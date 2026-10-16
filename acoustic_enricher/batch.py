"""Gemini Batch API mode for large one-off runs (about half the per-token price).

    beet listen-batch submit [--chunk N] [--max-usd X] [-n] QUERY
    beet listen-batch status
    beet listen-batch collect [--wait]

submit   extracts the clips, writes one JSONL request file per chunk (keyed
         `item-<id>`), estimates the cost and refuses to go past `max_usd`
         (counting everything already submitted), then uploads each file and
         creates a batch job. Job state lives in `batch_dir`.
collect  downloads finished results, validates and canonicalises them through
         the same path as `beet listen`, stores them, and records the real token
         usage. Raw result files are kept so a re-ingest never re-pays.

Results arrive asynchronously (usually hours, up to ~24h). Batch requests are
paid-tier only; there is no free tier for them.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time

from beets import ui

from . import providers as P
from . import schema as S
from . import vocab as V

AUDIO_TOKENS_PER_SECOND = 32          # Gemini audio input accounting
EST_TEXT_TOKENS = 2450        # ontology v3 instructions + schema, measured 2026-10-13
EST_OUTPUT_TOKENS = 450       # ontology v3 answer per run, measured 2026-10-13

# USD per 1M tokens at list (interactive) price; batch = list * batch_discount.
DEFAULT_PRICES = {
    'gemini-2.5-flash': {'audio': 1.00, 'text': 0.30, 'output': 2.50},
    'gemini-2.5-flash-lite': {'audio': 0.30, 'text': 0.10, 'output': 0.40},
    'gemini-3.5-flash-lite': {'audio': 0.50, 'text': 0.30, 'output': 2.50},
}

_DONE_STATES = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED',
                'JOB_STATE_EXPIRED'}


def _model_key(model: str) -> str:
    return model.split('/')[-1]


def _state_name(state) -> str:
    return getattr(state, 'name', None) or str(state)


class BatchRunner:
    def __init__(self, plugin):
        self.p = plugin
        self.cfg = plugin.config
        self.log = plugin._log
        self.dir = os.path.expanduser(self.cfg['batch_dir'].get())
        os.makedirs(self.dir, exist_ok=True)
        self.jobs_path = os.path.join(self.dir, 'jobs.json')
        self._client = None

    # -- state ----------------------------------------------------------------
    def _jobs(self) -> list[dict]:
        if not os.path.exists(self.jobs_path):
            return []
        with open(self.jobs_path, encoding='utf-8') as f:
            return json.load(f)

    def _save_jobs(self, jobs):
        tmp = self.jobs_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, indent=1)
        os.replace(tmp, self.jobs_path)

    def client(self):
        if self._client is None:
            from google import genai
            if self.cfg['use_vertex'].get(bool):
                self._client = genai.Client(vertexai=True, project=self.cfg['project'].get(),
                                            location=self.cfg['location'].get())
            else:
                key = P._secret(self.cfg, 'api_key', 'GEMINI_API_KEY')
                if not key:
                    raise ui.UserError('listen-batch needs a Gemini API key (api_key, '
                                       'api_key_file or GEMINI_API_KEY).')
                self._client = genai.Client(api_key=key)
        return self._client

    # -- pricing --------------------------------------------------------------
    def prices(self, model: str) -> dict:
        table = dict(DEFAULT_PRICES)
        table.update(self.cfg['prices'].get(dict) or {})
        key = _model_key(model)
        if key not in table:
            raise ui.UserError(f'No price for model {key!r}; add it under acoustic_enricher.prices')
        return table[key]

    def estimate_usd(self, model: str, audio_seconds: float, n_items: int) -> float:
        pr = self.prices(model)
        disc = self.cfg['batch_discount'].as_number()
        usd = (audio_seconds * AUDIO_TOKENS_PER_SECOND * pr['audio']
               + n_items * EST_TEXT_TOKENS * pr['text']
               + n_items * EST_OUTPUT_TOKENS * pr['output']) / 1e6
        return usd * disc

    def actual_usd(self, model: str, usage: dict) -> float:
        pr = self.prices(model)
        disc = self.cfg['batch_discount'].as_number()
        audio = usage.get('audio', 0)
        text = max(usage.get('prompt', 0) - audio, 0)
        out = usage.get('output', 0) + usage.get('thinking', 0)
        # cached prompt tokens (shared instruction prefix) bill at a fraction of the text price
        cached = min(usage.get('cached', 0), text)
        factor = self.cfg['cached_price_factor'].as_number()
        return (audio * pr['audio'] + (text - cached + cached * factor) * pr['text']
                + out * pr['output']) / 1e6 * disc

    def committed_usd(self) -> float:
        """Spend already committed: actual where known, else the estimate."""
        return sum(j.get('actual_usd', j.get('est_usd', 0)) for j in self._jobs()
                   if j.get('state') not in ('JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'))

    # -- request building -----------------------------------------------------
    def request_line(self, item_id: int, clip_path: str, run: int = 0, metadata: dict | None = None) -> str:
        with open(clip_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode()
        gen = {'responseMimeType': 'application/json', 'responseSchema': S.gemini_schema()}
        thinking = P.gemini_thinking(self.cfg['model'].get(), self.cfg['thinking_budget'].get(int),
                                     self.cfg['thinking_level'].get() or '')
        if thinking:
            gen['thinkingConfig'] = thinking
        temp = P.temperature(self.cfg)
        if temp is not None:
            gen['temperature'] = temp
        instructions, context = S.request_parts(metadata)       # instructions first: cacheable prefix
        req = {'contents': [{'role': 'user', 'parts': [
            {'text': instructions},
            {'inlineData': {'mimeType': 'audio/mp3', 'data': data}}]
            + ([{'text': context}] if context else [])}],
            'generationConfig': gen}
        key = f'item-{item_id}' if not run else f'item-{item_id}-r{run}'
        return json.dumps({'key': key, 'request': req})

    # -- submit ---------------------------------------------------------------
    def submit(self, lib, query, chunk: int, max_usd: float, dry_run: bool, force: bool):
        model = self.cfg['model'].get()
        if self.cfg['provider'].get() not in ('gemini', 'openai'):
            self.log.warning('listen-batch always uses the Gemini Batch API with model {0}.', model)
        runs = self.p._runs()
        pending = {i for j in self._jobs() if not j.get('ingested') for i in j['item_ids']}
        items = [it for it in lib.items(query)
                 if (force or it.get('listen_ver') != S.SCHEMA_VERSION
                     or int(it.get('listen_runs') or 1) < runs)
                 and it.id not in pending and float(it.length or 0) >= 3]
        if not items:
            self.log.info('Nothing to submit.')
            return
        audio_s = sum(sum(d for _, d in self.p.sample_windows(float(it.length))) for it in items)
        est = runs * self.estimate_usd(model, audio_s, len(items))
        committed = self.committed_usd()
        self.log.info('{0} items x {1} run(s), {2:.0f} min of clips, est ${3:.4f} at batch price '
                      '(already committed ${4:.4f}; cap ${5:.2f}).',
                      len(items), runs, audio_s / 60, est, committed, max_usd)
        if max_usd and committed + est > max_usd:
            raise ui.UserError(f'Estimated total ${committed + est:.2f} exceeds the cap '
                               f'${max_usd:.2f}; narrow the query or raise --max-usd.')
        if dry_run:
            return
        jobs = self._jobs()
        for start in range(0, len(items), chunk):
            part = items[start:start + chunk]
            self._submit_chunk(part, model, jobs)
            self._save_jobs(jobs)

    def _submit_chunk(self, items, model, jobs):
        from google.genai import types
        stamp = time.strftime('%Y%m%d-%H%M%S')
        runs = self.p._runs()
        req_path = os.path.join(self.dir, f'requests-{stamp}-{items[0].id}.jsonl')
        ids, audio_s = [], 0.0
        measured: dict[str, dict] = {}   # item id -> measured BPM, applied at ingest
        with open(req_path, 'w', encoding='utf-8') as out:
            for it in items:
                path = os.fsdecode(it.path)
                fd, clip = tempfile.mkstemp(suffix='.mp3')
                os.close(fd)
                try:
                    windows, analysed = self.p.clip_plan(path, float(it.length))
                    if analysed:
                        measured[str(it.id)] = {k: analysed.get(k) for k in (
                            'bpm', 'bpm_strength', 'intro_s', 'onset_rate', 'harsh',
                            'tail_silence_s', 'beat_strength')}
                    self.p.extract_clips(path, windows, clip)
                    for r in range(runs):
                        out.write(self.request_line(it.id, clip, r, self.p.metadata_of(it)) + '\n')
                    ids.append(it.id)
                    audio_s += sum(d for _, d in windows)
                except Exception as e:
                    self.log.error('Clip failed, skipping {0}: {1}', it.title, e)
                finally:
                    os.remove(clip)
        est = runs * self.estimate_usd(model, audio_s, len(ids))
        client = self.client()
        up = client.files.upload(file=req_path, config=types.UploadFileConfig(
            mime_type='jsonl', display_name=os.path.basename(req_path)))
        job = client.batches.create(model=model, src=up.name,
                                    config={'display_name': f'acoustic-enricher-{stamp}'})
        jobs.append({'name': job.name, 'model': model, 'requests': req_path,
                     'item_ids': ids, 'est_usd': round(est, 4), 'submitted': time.time(),
                     'state': _state_name(job.state), 'ingested': False,
                     'schema_version': S.SCHEMA_VERSION, 'runs': runs, 'measured': measured})
        self.log.info('Submitted {0}: {1} items, est ${2:.4f}.', job.name, len(ids), est)

    # -- status ---------------------------------------------------------------
    def refresh(self) -> list[dict]:
        jobs = self._jobs()
        for j in jobs:
            if j.get('state') in _DONE_STATES:
                continue
            job = self.client().batches.get(name=j['name'])
            j['state'] = _state_name(job.state)
            if job.dest is not None and getattr(job.dest, 'file_name', None):
                j['result_file'] = job.dest.file_name
        self._save_jobs(jobs)
        return jobs

    def status(self):
        for j in self.refresh():
            self.log.info('{0}  {1:24s} items={2:5d} est=${3:.4f} actual={4} ingested={5}',
                          j['name'], j['state'], len(j['item_ids']), j['est_usd'],
                          f"${j['actual_usd']:.4f}" if 'actual_usd' in j else '-',
                          j.get('ingested'))
        self.log.info('Committed spend: ${0:.4f}', self.committed_usd())

    # -- collect --------------------------------------------------------------
    def collect(self, lib, wait: bool, db_only: bool):
        while True:
            jobs = self.refresh()
            todo = [j for j in jobs if not j.get('ingested')]
            ready = [j for j in todo if j['state'] == 'JOB_STATE_SUCCEEDED']
            for j in ready:
                self._ingest(lib, j, db_only)
                self._save_jobs(jobs)
            running = [j for j in todo if j['state'] not in _DONE_STATES]
            if not wait or not running:
                break
            self.log.info('{0} job(s) still running; checking again in 5 min.', len(running))
            time.sleep(300)

    def _ingest(self, lib, job, db_only):
        raw_path = job.get('results') or os.path.join(
            self.dir, os.path.basename(job['requests']).replace('requests-', 'results-'))
        if not os.path.exists(raw_path):
            data = self.client().files.download(file=job['result_file'])
            with open(raw_path, 'wb') as f:
                f.write(data)
        job['results'] = raw_path
        ok, failed = 0, []
        usage_tot = {'prompt': 0, 'audio': 0, 'output': 0, 'thinking': 0}
        # Keys are item-<id> or item-<id>-r<n> (one line per run); runs of the same
        # item are canonicalised separately, then combined.
        per_item: dict[int, list] = {}
        errors: dict[int, str] = {}
        with open(raw_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                item_id = int(str(rec.get('key', 'item-0')).split('-')[1])
                per_item.setdefault(item_id, [])
                resp = rec.get('response') or {}
                um = resp.get('usageMetadata') or {}
                usage_tot['prompt'] += um.get('promptTokenCount', 0)
                usage_tot['output'] += um.get('candidatesTokenCount', 0)
                usage_tot['thinking'] += um.get('thoughtsTokenCount', 0)
                usage_tot['cached'] = usage_tot.get('cached', 0) + um.get('cachedContentTokenCount', 0)
                for d in um.get('promptTokensDetails', []) or []:
                    if d.get('modality') == 'AUDIO':
                        usage_tot['audio'] += d.get('tokenCount', 0)
                try:
                    if 'error' in rec:
                        raise ValueError(json.dumps(rec['error'])[:200])
                    text = resp['candidates'][0]['content']['parts'][0]['text']
                    can = V.canonicalize(P.parse_json(text))
                    if not can.moods or not can.shape:
                        raise ValueError(f'incomplete result: {text[:120]}')
                    per_item[item_id].append(can)
                except Exception as e:
                    errors[item_id] = str(e)[:200]
        for item_id, cans in per_item.items():
            item = lib.get_item(item_id)
            if item is None:
                failed.append({'id': item_id, 'error': 'item no longer in library'})
                continue
            if not cans:
                failed.append({'id': item_id, 'error': errors.get(item_id, 'no valid run')})
                continue
            try:
                can = V.combine(cans)
                self.p.apply(item, can, 'gemini-batch', _model_key(job['model']))
                self.p.apply_measured(item, can, (job.get('measured') or {}).get(str(item_id), {}))
                item.store()
                if not db_only:
                    self.p.write_tags(item, os.fsdecode(item.path))
                ok += 1
            except Exception as e:
                failed.append({'id': item_id, 'error': str(e)[:200]})
        job['ingested'] = True
        job['ok'] = ok
        job['failed'] = failed
        job['usage'] = usage_tot
        job['actual_usd'] = round(self.actual_usd(job['model'], usage_tot), 4)
        self.log.info('Ingested {0}: {1} ok, {2} failed, actual ${3:.4f} (est ${4:.4f}).',
                      job['name'], ok, len(failed), job['actual_usd'], job['est_usd'])
        if failed:
            self.log.info('Failed ids (re-run `beet listen-batch submit` on them, or `beet listen`): '
                          '{0}', ' '.join(f"id:{x['id']}" for x in failed[:50]))


def command(plugin):
    cmd = ui.Subcommand('listen-batch', help='run `listen` through the Gemini Batch API '
                                             '(submit | status | collect)')
    cmd.parser.usage += '\n  submit [--chunk N] [--max-usd X] [-n] [-f] QUERY\n  status\n  collect [--wait]'
    cmd.parser.add_option('--chunk', type='int', default=None, help='items per batch job')
    cmd.parser.add_option('--max-usd', type='float', default=None,
                          help='refuse to submit past this total spend')
    cmd.parser.add_option('-n', '--dry-run', action='store_true', default=False)
    cmd.parser.add_option('-f', '--force', action='store_true', default=False)
    cmd.parser.add_option('--wait', action='store_true', default=False)
    cmd.parser.add_option('--db-only', action='store_true', default=None)
    cmd.parser.add_option('--write-tags', action='store_false', dest='db_only')

    def run(lib, opts, args):
        if not args or args[0] not in ('submit', 'status', 'collect'):
            raise ui.UserError('usage: beet listen-batch submit|status|collect ...')
        action, query = args[0], args[1:]
        runner = BatchRunner(plugin)
        if action == 'submit':
            runner.submit(lib, query,
                          chunk=opts.chunk or plugin.config['batch_chunk'].get(int),
                          max_usd=opts.max_usd if opts.max_usd is not None
                          else plugin.config['batch_max_usd'].as_number(),
                          dry_run=opts.dry_run, force=opts.force)
        elif action == 'status':
            runner.status()
        else:
            runner.collect(lib, wait=opts.wait, db_only=plugin._db_only(opts))

    cmd.func = run
    return cmd
