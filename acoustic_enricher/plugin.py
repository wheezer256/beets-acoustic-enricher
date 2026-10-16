"""beet listen: sample a track, ask an audio-capable LLM to describe it, and
store the description in one controlled schema.

Fields (beets flexible attributes; * = also written to the file when tags are
written, which is the default unless `db_only` / --db-only):

  mood*                     1-3 terms from data/moods.yaml, comma-separated
  roontracktag*             "style:<s>; arc:<shape>; peak:<lvl>; <element>; ..."
  listen_style              canonical Discogs styles / MusicBrainz genres
  listen_elements           JSON list of canonical AudioSet class names
  listen_energy_shape/base/peak/narrative
  listen_tempo              slow / medium / fast / very fast (from listen_bpm)
  listen_bpm                felt BPM (LLM, median of runs; refined by the measured BPM
                            when clip_strategy is adaptive and the two agree up to 2x)
  listen_bpm_measured       BPM measured from the full track (adaptive strategy only)
  listen_rhythm             steady / shifting / fragmented
  listen_intro_s, listen_onset_rate, listen_harsh, listen_tail_silence_s,
  listen_beat_strength      whole-track measurements (adaptive strategy; analysis.py)
  listen_pulse              none / soft / steady / driving (beat prominence)
  listen_runs               number of LLM runs combined (config `runs`)
  listen_agreement          0..1 agreement between those runs (per-track confidence)
  listen_valence            mean circumplex valence of the mood terms (-1..1)
  listen_arousal            mean circumplex arousal of the mood terms (-1..1)
  listen_instrumental       1.0 no AudioSet vocal class heard, 0.0 vocals heard
  listen_dominance          -1 overwhelmed .. +1 in control, from the mood terms (ontology v3)
  listen_v3                 JSON: felt/use/timeplace/character term shares + listened dims (ontology v3)
  listen_unmapped           JSON of labels dropped because they were not in a vocabulary
  listen_ver                schema/prompt version (see schema.SCHEMA_VERSION)
  listen_provider / listen_model
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time

from beets import dbcore, ui
from beets.plugins import BeetsPlugin

from . import providers as P
from . import schema as S
from . import vocab as V


def _register_media_fields(plugin):
    try:
        from mediafile import (ASFStorageStyle, MediaField, MP3DescStorageStyle,
                               MP3StorageStyle, MP4StorageStyle, StorageStyle)
        plugin.add_media_field('roontracktag', MediaField(
            MP3DescStorageStyle('ROONTRACKTAG'), StorageStyle('ROONTRACKTAG'),
            ASFStorageStyle('ROONTRACKTAG'), MP4StorageStyle('----:com.apple.iTunes:ROONTRACKTAG'),
        ))
        # MOOD: ID3 TMOO, Vorbis/FLAC MOOD, MP4 freeform MOOD, ASF WM/Mood.
        # mediafile has no built-in mood field, which is why older versions of
        # this plugin never got mood into the file.
        plugin.add_media_field('mood', MediaField(
            MP3StorageStyle('TMOO'), StorageStyle('MOOD'),
            ASFStorageStyle('WM/Mood'), MP4StorageStyle('----:com.apple.iTunes:MOOD'),
        ))
    except Exception as e:  # pragma: no cover - depends on mediafile version
        plugin._log.warning('Could not register media fields: {0}', e)


class AcousticEnricherPlugin(BeetsPlugin):
    name = 'acoustic_enricher'  # config section; must stay stable for existing users

    item_types = {
        'mood': dbcore.types.STRING,
        'roontracktag': dbcore.types.STRING,
        'listen_style': dbcore.types.STRING,
        'listen_elements': dbcore.types.STRING,
        'listen_energy_shape': dbcore.types.STRING,
        'listen_energy_base': dbcore.types.STRING,
        'listen_energy_peak': dbcore.types.STRING,
        'listen_energy_narrative': dbcore.types.STRING,
        'listen_tempo': dbcore.types.STRING,
        'listen_pulse': dbcore.types.STRING,
        'listen_bpm': dbcore.types.FLOAT,
        'listen_bpm_measured': dbcore.types.FLOAT,
        'listen_rhythm': dbcore.types.STRING,
        'listen_bpm_runs': dbcore.types.STRING,
        'listen_intro_s': dbcore.types.INTEGER,
        'listen_onset_rate': dbcore.types.FLOAT,
        'listen_harsh': dbcore.types.FLOAT,
        'listen_tail_silence_s': dbcore.types.INTEGER,
        'listen_beat_strength': dbcore.types.FLOAT,
        'listen_runs': dbcore.types.INTEGER,
        'listen_agreement': dbcore.types.FLOAT,
        'listen_dominance': dbcore.types.FLOAT,
        'listen_v3': dbcore.types.STRING,
        'listen_valence': dbcore.types.FLOAT,
        'listen_arousal': dbcore.types.FLOAT,
        'listen_instrumental': dbcore.types.FLOAT,
        'listen_unmapped': dbcore.types.STRING,
        'listen_ver': dbcore.types.STRING,
        'listen_provider': dbcore.types.STRING,
        'listen_model': dbcore.types.STRING,
    }

    def __init__(self):
        super().__init__('acoustic_enricher')
        self.config.add({
            'provider': 'gemini',
            'model': 'gemini-3.5-flash-lite',
            # Gemini / Vertex AI
            'api_key': os.environ.get('GEMINI_API_KEY', ''),
            'api_key_file': '',
            'use_vertex': False,
            'project': os.environ.get('GOOGLE_CLOUD_PROJECT', ''),
            'location': 'us-central1',
            'thinking_budget': 0,          # 0 = no thinking tokens (cheaper); -1 = model default
            'thinking_level': 'minimal',   # Gemini 3.x: minimal|low|medium|high ('' = model default)
            'temperature': '',             # sampling temperature; '' = provider default
            # Calls per track, combined (vocab.combine). LLM labels vary between runs;
            # 3 runs measured ~0.92-0.95 agreement vs ~0.80-0.86 for one. Cost scales.
            'runs': 1,
            # Where the clips come from: 'fixed' (25/50/75 % of the track; reads only
            # the clips) or 'adaptive' (decodes the whole track once, see analysis.py,
            # and aims the clips at the most rhythmic / loudest / distinct sections;
            # also measures the BPM). Same total clip length, so same LLM cost.
            'clip_strategy': 'fixed',
            # Send catalogue context (artist/title/album/year/genre/label) with the audio. It is
            # used only for meaning the audio cannot carry (ritual purpose, irony, era); the
            # prompt tells the model to judge sound from the audio. Off by default.
            'include_metadata': False,
            # OpenAI-compatible (LiteLLM, vLLM, OpenRouter, ...)
            'openai_url': 'http://localhost:4000/v1/chat/completions',
            'openai_api_key': os.environ.get('OPENAI_API_KEY', ''),
            'openai_api_key_file': '',
            'reasoning_effort': 'disable',  # sent to OpenAI-compatible endpoints; '' to omit
            'structured_output': True,      # json_schema response_format (else json_object)
            # Anthropic (metadata-only)
            'anthropic_api_key': os.environ.get('ANTHROPIC_API_KEY', ''),
            # Ollama / llama.cpp
            'ollama_url': 'http://localhost:11434/api/chat',
            'llama_url': 'http://localhost:8080/v1/chat/completions',
            'timeout': 600,
            # Sampling
            'clip_count': 3,
            'clip_duration': 15,
            # Writing
            'db_only': False,               # True: store in the beets DB only, never touch files
            'preserve_mtime': True,         # restore file mtime after writing tags
            # Failure handling / accounting
            'max_consecutive_failures': 0,  # 0 = disabled
            'ledger': '',                   # path of a JSONL usage ledger ('' = off)
            # Gemini Batch API (`beet listen-batch`)
            'batch_dir': '',                # '' = <beets config dir>/listen-batches
            'batch_chunk': 1000,            # items per batch job
            'batch_max_usd': 10,            # refuse to submit past this committed spend
            'batch_discount': 0.5,          # batch price as a fraction of list price
            'cached_price_factor': 0.25,    # cached prompt tokens as a fraction of the text price
            'prices': {},                   # per-model USD/1M overrides {audio,text,output}
        })
        if not self.config['batch_dir'].get():
            from beets import config as _beets_config
            self.config['batch_dir'] = os.path.join(_beets_config.config_dir(), 'listen-batches')
        for k in ('api_key', 'openai_api_key', 'anthropic_api_key'):
            self.config[k].redact = True
        _register_media_fields(self)
        self._provider = None

    # -- commands -----------------------------------------------------------
    def commands(self):
        listen = ui.Subcommand('listen', help='describe tracks with an audio LLM '
                                              '(mood/style/elements/energy arc)')
        listen.parser.add_option('-n', '--dry-run', action='store_true', default=False,
                                 help='print the analysis, store nothing')
        listen.parser.add_option('-f', '--force', action='store_true', default=False,
                                 help='re-analyse tracks that already have the current listen_ver')
        listen.parser.add_option('--db-only', action='store_true', default=None,
                                 help='store in the beets DB only (do not write file tags)')
        listen.parser.add_option('--write-tags', action='store_false', dest='db_only',
                                 help='also write MOOD/ROONTRACKTAG file tags')
        listen.func = self._cmd_listen

        norm = ui.Subcommand('listen-normalize',
                             help='re-map stored enricher labels onto the controlled '
                                  'vocabularies (no LLM calls)')
        norm.parser.add_option('-n', '--dry-run', action='store_true', default=False)
        norm.func = self._cmd_normalize
        from .batch import command as batch_command
        return [listen, norm, batch_command(self)]

    def _get_provider(self):
        if self._provider is None:
            name = self.config['provider'].get()
            cls = P.PROVIDERS.get(name)
            if cls is None:
                raise ui.UserError(f'Unknown provider {name!r}; choose from {", ".join(P.PROVIDERS)}')
            self._provider = cls(self.config, self._log)
        return self._provider

    def clip_plan(self, path: str, duration: float) -> tuple[list, dict]:
        """Clip windows for a track, plus analysis results (measured BPM) when the
        adaptive strategy is on. Falls back to fixed windows if analysis fails."""
        if self.config['clip_strategy'].get() == 'adaptive' and duration >= 60:
            from . import analysis as A
            try:
                res = A.analyse(path, float(self.config['clip_duration'].get(int)),
                                self.config['clip_count'].get(int))
                if res.get('windows'):
                    return res['windows'], res
            except Exception as e:  # noqa: BLE001
                self._log.warning('Track analysis failed, using fixed clips: {0}', e)
        return self.sample_windows(duration), {}

    def apply_measured(self, item, can: V.Canonical, measured: dict):
        """Store the measured BPM and refine the felt BPM with it (analysis.reconcile_bpm)."""
        from . import analysis as A
        for key, field in (('intro_s', 'listen_intro_s'), ('onset_rate', 'listen_onset_rate'),
                           ('harsh', 'listen_harsh'), ('tail_silence_s', 'listen_tail_silence_s'),
                           ('beat_strength', 'listen_beat_strength')):
            if measured and measured.get(key) is not None:
                item[field] = measured[key]
        if not measured or not measured.get('bpm'):
            if 'listen_bpm_measured' in item:
                del item['listen_bpm_measured']
            return
        item['listen_bpm_measured'] = measured['bpm']
        bpm = A.reconcile_bpm(can.bpm, measured['bpm'], measured.get('bpm_strength', 0.0))
        if bpm:
            item['listen_bpm'] = bpm
            item['listen_tempo'] = V.tempo_of(bpm)

    def _runs(self) -> int:
        return max(1, self.config['runs'].get(int))

    def _db_only(self, opts) -> bool:
        flag = getattr(opts, 'db_only', None)
        return self.config['db_only'].get(bool) if flag is None else flag

    # -- listen -------------------------------------------------------------
    def _cmd_listen(self, lib, opts, args):
        items = list(lib.items(args))
        if not items:
            self._log.info('No tracks matched the query.')
            return
        provider_name = self.config['provider'].get()
        if provider_name not in P.AUDIO_PROVIDERS:
            self._log.info('Note: {0} infers from metadata only; it does not hear the audio.',
                           provider_name)
        provider = self._get_provider()
        db_only = self._db_only(opts)

        done = skipped = failed = streak = 0
        max_streak = self.config['max_consecutive_failures'].get(int)
        for item in items:
            if (not opts.force and item.get('listen_ver') == S.SCHEMA_VERSION
                    and int(item.get('listen_runs') or 1) >= self._runs()):
                skipped += 1
                continue
            try:
                self._analyze(item, provider, provider_name, opts.dry_run, db_only)
                done += 1
                streak = 0
            except KeyboardInterrupt:
                self._log.info('Interrupted.')
                break
            except P.QuotaError as e:
                self._log.error('Quota/budget exceeded, stopping: {0}', e)
                break
            except Exception as e:
                self._log.error('Failed: {0}: {1}', item.title, e)
                failed += 1
                streak += 1
                if max_streak and streak >= max_streak:
                    self._log.error('Stopping after {0} consecutive failures.', streak)
                    break
        self._log.info('Done [{0}/{1}]: {2} matched, {3} analysed, {4} already current, {5} failed.',
                       provider_name, self.config['model'].get(), len(items), done, skipped, failed)

    def _analyze(self, item, provider, provider_name, dry_run, db_only):
        path = os.fsdecode(item.path)
        duration = float(item.length or 0)
        if duration < 3:
            self._log.warning('Too short to analyse ({0:.0f}s): {1}', duration, item.title)
            return
        metadata = self.metadata_of(item)

        os.makedirs('/tmp/beets', exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix='.mp3', dir='/tmp/beets')
        os.close(fd)
        t0 = time.monotonic()
        runs, usage = [], {}
        try:
            measured = {}
            if provider_name in P.AUDIO_PROVIDERS:
                windows, measured = self.clip_plan(path, duration)
                self.extract_clips(path, windows, tmp_path)
            # The clip is extracted once and sent `runs` times; a failed run is
            # dropped as long as at least one succeeds (QuotaError always stops).
            last_err = None
            for _ in range(self._runs()):
                try:
                    runs.append(V.canonicalize(provider.analyze(tmp_path, metadata)))
                except P.QuotaError:
                    raise
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    continue
                for k, v in (getattr(provider, 'last_usage', None) or {}).items():
                    if isinstance(v, (int, float)):
                        usage[k] = usage.get(k, 0) + v
            if not runs:
                raise last_err or RuntimeError('no result')
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        elapsed = time.monotonic() - t0

        can = V.combine(runs)
        self._log.info('{0} | mood={1} style={2} arc={3}/{4}->{5} tempo={6} pulse={7} instr={8} '
                       'runs={9} agree={10} unmapped={11}',
                       item.title, can.moods, can.styles, can.shape, can.base, can.peak,
                       can.tempo, can.pulse, can.instrumental, can.runs, can.agreement,
                       can.unmapped or '-')
        self._ledger(item, provider_name, usage or None, elapsed, can)
        if dry_run:
            return
        self.apply(item, can, provider_name, self.config['model'].get())
        self.apply_measured(item, can, measured)
        item.store()
        if not db_only:
            self.write_tags(item, path)

    def metadata_of(self, item) -> dict | None:
        """Catalogue context for the prompt, or None when include_metadata is off."""
        if not self.config['include_metadata'].get(bool):
            return None
        return {k: str(item.get(k) or '') for k in ('artist', 'title', 'album', 'year', 'genre', 'label')}

    def apply(self, item, can: V.Canonical, provider_name: str, model: str):
        """Set the schema fields on an item (no store/write)."""
        item['mood'] = ', '.join(can.moods)
        item['listen_style'] = ', '.join(can.styles)
        item['listen_elements'] = json.dumps(can.elements)
        item['listen_energy_shape'] = can.shape
        item['listen_energy_base'] = can.base
        item['listen_energy_peak'] = can.peak
        item['listen_energy_narrative'] = can.narrative
        item['listen_tempo'] = can.tempo
        item['listen_pulse'] = can.pulse
        item['listen_rhythm'] = can.character
        item['listen_bpm_runs'] = ','.join(f'{b:g}' for b in (can.bpm_runs or ((can.bpm,) if can.bpm else ())))
        if can.bpm:
            item['listen_bpm'] = can.bpm
        elif 'listen_bpm' in item:
            del item['listen_bpm']
        item['listen_runs'] = can.runs
        if can.agreement is None:
            if 'listen_agreement' in item:
                del item['listen_agreement']
        else:
            item['listen_agreement'] = can.agreement
        item['roontracktag'] = can.roontracktag()
        item['listen_v3'] = json.dumps(can.v3, ensure_ascii=False) if can.v3 else ''
        for key, val in (('listen_valence', can.valence), ('listen_arousal', can.arousal),
                         ('listen_dominance', can.dominance),
                         ('listen_instrumental', can.instrumental)):
            if val is None:
                if key in item:
                    del item[key]
            else:
                item[key] = val
        item['listen_unmapped'] = json.dumps(can.unmapped) if can.unmapped else ''
        item['listen_ver'] = S.SCHEMA_VERSION
        item['listen_provider'] = provider_name
        item['listen_model'] = model

    def write_tags(self, item, path):
        stat = os.stat(path) if self.config['preserve_mtime'].get(bool) else None
        item.write()
        if stat is not None:
            os.utime(path, (stat.st_atime, stat.st_mtime))

    def _ledger(self, item, provider_name, usage, elapsed, can):
        path = self.config['ledger'].get()
        if not path:
            return
        rec = {'t': time.time(), 'id': item.id, 'provider': provider_name,
               'model': self.config['model'].get(), 'elapsed_s': round(elapsed, 2),
               'usage': usage, 'unmapped': can.unmapped}
        with open(os.path.expanduser(path), 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')

    # -- normalize ----------------------------------------------------------
    def _cmd_normalize(self, lib, opts, args):
        """Re-canonicalise older labels; fills listen_* from ROONTRACKTAG when
        the DB fields were lost. Mood is never invented (it is not in ROONTRACKTAG)."""
        changed = 0
        for item in lib.items(args):
            rt, have_db = item.get('roontracktag') or '', bool(item.get('listen_style'))
            if not rt and not have_db:
                continue
            if have_db:
                raw = {'mood': item.get('mood') or '',
                       'style': item.get('listen_style') or '',
                       'elements': json.loads(item.get('listen_elements') or '[]'),
                       'energy_arc': {'shape': item.get('listen_energy_shape') or '',
                                      'base': item.get('listen_energy_base') or '',
                                      'peak': item.get('listen_energy_peak') or '',
                                      'trajectory': item.get('listen_energy_narrative') or ''}}
            else:
                raw = V.parse_roontracktag(rt)
            can = V.canonicalize(raw)
            if opts.dry_run:
                self._log.info('{0} | {1} -> {2}', item.title, rt[:80], can.roontracktag()[:80])
                continue
            prov = item.get('listen_provider') or 'legacy'
            model = item.get('listen_model') or ''
            self.apply(item, can, prov, model)
            item['listen_ver'] = item.get('listen_ver') if have_db else f'{S.SCHEMA_VERSION}-normalized'
            item.store()
            changed += 1
        self._log.info('listen-normalize: {0} items updated{1}.', changed,
                       ' (dry run)' if opts.dry_run else '')

    # -- audio sampling -----------------------------------------------------
    def sample_windows(self, duration: float) -> list[tuple[float, float]]:
        clip_dur = self.config['clip_duration'].get(int)
        clip_count = self.config['clip_count'].get(int)
        if duration < 30:
            return [(0.0, duration)]
        if duration < 60:
            return [(max(0.0, duration / 2 - clip_dur / 2), min(clip_dur, duration))]
        positions = [i / (clip_count + 1) for i in range(1, clip_count + 1)]
        return [(min(duration * p, duration - clip_dur), float(clip_dur)) for p in positions]

    @staticmethod
    def extract_clips(input_path: str, samples, output_path: str):
        """Concatenate the sample windows into one low-bitrate mono MP3.

        Each window is opened with input seeking (-ss/-t before -i), so only the
        sampled seconds are read and decoded instead of the whole file."""
        cmd = ['ffmpeg', '-v', 'error', '-y']
        for start, dur in samples:
            cmd += ['-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', input_path]
        n = len(samples)
        graph = ''.join(f'[{i}:a]' for i in range(n)) + f'concat=n={n}:v=0:a=1[outa]'
        cmd += ['-filter_complex', graph, '-map', '[outa]', '-vn', '-map_metadata', '-1',
                '-codec:a', 'libmp3lame', '-b:a', '64k', '-ar', '22050', '-ac', '1', output_path]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f'ffmpeg failed: {result.stderr.decode(errors="replace")[-500:]}')
