import os
import json
import time
import base64
import subprocess
import tempfile

from beets import dbcore, ui
from beets.plugins import BeetsPlugin


_AUDIO_PROMPT = (
    'The audio contains clips sampled from the beginning, middle, and end of '
    'the track, in chronological order. Listen to how the energy and character '
    'evolve across these clips.\n\n'
    'Return a JSON object with exactly four keys:\n'
    '\n'
    '  "mood": 1-3 words describing emotional valence using the MIR '
    'valence-arousal literature (Russell circumplex model). Cover the full '
    'affective space: euphoric, triumphant, exhilarated, tense, frantic, '
    'anxious, serene, peaceful, tender, wistful, melancholic, desolate, '
    'somber, brooding, hypnotic, ethereal, eerie, nostalgic, dreamy, '
    'mysterious, cinematic — and any other precise valence term.\n'
    '\n'
    '  "style": 1-3 genre or style terms from the Discogs or MusicBrainz '
    'genre taxonomy. Use sub-genres where the recording warrants it '
    '(e.g. "minimal techno", "free jazz", "fingerstyle folk", '
    '"drum and bass", "post-bop", "darkwave", "shoegaze", '
    '"third stream", "cumbia").\n'
    '\n'
    '  "elements": an array of 8-15 labels drawn strictly from the AudioSet '
    'ontology (https://research.google.com/audioset/ontology/index.html). '
    'Cover all audible dimensions: instruments present, vocal characteristics, '
    'rhythmic and percussive elements, environmental and sound-design events, '
    'and production characteristics.\n'
    '\n'
    '  "energy_arc": an object describing how energy evolves through the '
    'track using the MIR temporal music narrative model. The audio clips are '
    'in chronological order — compare their relative energy, density, and '
    'tension to infer the contour shape. Return:\n'
    '    "shape": one of exactly [steady-low, steady-mid, steady-high, '
    'building, arch, valley, oscillating, explosive, breakdown-drop, '
    'decaying, cyclic]\n'
    '    "base": the resting / between-peak energy level — one of '
    '[low, medium, high]\n'
    '    "peak": the maximum energy level reached — one of [low, medium, high]\n'
    '    "trajectory": a 5-10 word narrative description of the arc\n'
    '\n'
    'Return only the JSON object — no explanation, no markdown fences.'
)

_TEXT_PROMPT = (
    'You are a music expert. Based on the track metadata below, infer the '
    'likely sonic character using your knowledge of the artist, album, and '
    'genre.\n\n'
    '{meta}\n\n'
    'Return a JSON object with exactly four keys:\n'
    '\n'
    '  "mood": 1-3 words describing emotional valence using the MIR '
    'valence-arousal literature (Russell circumplex model).\n'
    '\n'
    '  "style": 1-3 genre or style terms from the Discogs or MusicBrainz '
    'genre taxonomy. Use sub-genres where appropriate.\n'
    '\n'
    '  "elements": an array of 8-15 labels drawn from the AudioSet ontology '
    '(https://research.google.com/audioset/ontology/index.html). Include '
    'instruments, vocals, rhythmic elements, environmental sounds, and '
    'production characteristics typical of this artist and release.\n'
    '\n'
    '  "energy_arc": an object describing the typical energy arc for this '
    'track based on genre conventions and artist knowledge (note: inferred '
    'from metadata, not actual audio). Return:\n'
    '    "shape": one of [steady-low, steady-mid, steady-high, building, '
    'arch, valley, oscillating, explosive, breakdown-drop, decaying, cyclic]\n'
    '    "base": resting energy level [low, medium, high]\n'
    '    "peak": maximum energy reached [low, medium, high]\n'
    '    "trajectory": 5-10 word narrative summary\n'
    '\n'
    'Return only the JSON object — no explanation, no markdown fences.'
)


class _QuotaError(Exception):
    """Raised when a provider's quota or spend cap is exceeded."""


def _is_quota_error(exc):
    # requests-based providers (Ollama, LlamaCpp)
    try:
        if hasattr(exc, 'response') and exc.response.status_code == 429:
            return True
    except Exception:
        pass
    # Anthropic SDK
    if type(exc).__name__ == 'RateLimitError':
        return True
    # google-api-core / google-genai
    if type(exc).__name__ in ('ResourceExhausted', 'TooManyRequests'):
        return True
    if hasattr(exc, 'status_code') and exc.status_code == 429:
        return True
    msg = str(exc).lower()
    return 'quota' in msg or 'resource_exhausted' in msg


def _parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:])
        if text.endswith('```'):
            text = text[:text.rfind('```')]
        text = text.strip()
    return json.loads(text)


def _register_roon_media_fields(plugin):
    try:
        from mediafile import (
            MediaField,
            MP3DescStorageStyle,
            StorageStyle,
            ASFStorageStyle,
            MP4StorageStyle,
        )
        plugin.add_media_field(
            'roontracktag',
            MediaField(
                MP3DescStorageStyle('ROONTRACKTAG'),
                StorageStyle('ROONTRACKTAG'),
                ASFStorageStyle('ROONTRACKTAG'),
                MP4StorageStyle('----:com.apple.iTunes:ROONTRACKTAG'),
            ),
        )
    except Exception as e:
        plugin._log.warning('Could not register ROONTRACKTAG media field: {0}', e)


# ---------------------------------------------------------------------------
# LLM providers
#
# Interface: analyze(audio_path, metadata)
#   audio_path — path to the temp MP3 slice
#   metadata   — dict with title/artist/album/year/genre (str values)
#
# Audio providers (Gemini, Ollama, LlamaCpp) use the audio file.
# The Anthropic provider uses metadata only — Claude has no audio input API.
# ---------------------------------------------------------------------------

class _GeminiProvider:
    def __init__(self, cfg, log):
        from google import genai
        self._log = log
        model = cfg['model'].get()
        if not model.startswith('models/'):
            model = f'models/{model}'
        self._model = model
        if cfg['use_vertex'].get(bool):
            self._client = genai.Client(
                vertexai=True,
                project=cfg['project'].get(),
                location=cfg['location'].get(),
            )
        else:
            api_key = cfg['api_key'].get()
            if not api_key:
                raise RuntimeError(
                    'No Gemini API key. Set GEMINI_API_KEY or acoustic_enricher.api_key.'
                )
            self._client = genai.Client(api_key=api_key)

    def analyze(self, audio_path, metadata):
        from google.genai import types
        audio_file = self._client.files.upload(file=audio_path)
        try:
            for _ in range(60):
                state = audio_file.state
                if (state.name if hasattr(state, 'name') else str(state)) != 'PROCESSING':
                    break
                time.sleep(2)
                audio_file = self._client.files.get(name=audio_file.name)
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[audio_file, _AUDIO_PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type='application/json',
                    ),
                )
            except Exception as e:
                if _is_quota_error(e):
                    raise _QuotaError(str(e)) from e
                raise
        finally:
            try:
                self._client.files.delete(name=audio_file.name)
            except Exception:
                pass
        return _parse_json(response.text)


class _OllamaProvider:
    """Ollama /api/chat — requires a model with audio/multimodal support."""

    def __init__(self, cfg, log):
        self._log = log
        self._url = cfg['ollama_url'].get() or 'http://localhost:11434/api/chat'
        self._model = cfg['model'].get()

    def analyze(self, audio_path, metadata):
        import requests
        with open(audio_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            'model': self._model,
            'messages': [
                {'role': 'user', 'content': _AUDIO_PROMPT, 'images': [b64]},
            ],
            'format': 'json',
            'stream': False,
        }
        try:
            r = requests.post(self._url, json=payload, timeout=300)
            r.raise_for_status()
        except Exception as e:
            if _is_quota_error(e):
                raise _QuotaError(str(e)) from e
            raise
        return _parse_json(r.json()['message']['content'])


class _LlamaCppProvider:
    """llama-server OpenAI-compatible endpoint with native audio input."""

    def __init__(self, cfg, log):
        self._log = log
        self._url = cfg['llama_url'].get() or 'http://localhost:8080/v1/chat/completions'
        self._model = cfg['model'].get()

    def analyze(self, audio_path, metadata):
        import requests
        with open(audio_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            'model': self._model,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': _AUDIO_PROMPT},
                        {'type': 'input_audio', 'input_audio': {'data': b64, 'format': 'mp3'}},
                    ],
                }
            ],
            'response_format': {'type': 'json_object'},
        }
        try:
            r = requests.post(self._url, json=payload, timeout=300)
            r.raise_for_status()
        except Exception as e:
            if _is_quota_error(e):
                raise _QuotaError(str(e)) from e
            raise
        return _parse_json(r.json()['choices'][0]['message']['content'])


class _AnthropicProvider:
    """Claude via the Anthropic SDK.

    Claude has no audio input API, so this provider sends track metadata and
    asks Claude to infer mood/style/elements from its training knowledge.
    Results are knowledge-based rather than acoustic — useful for well-known
    artists but blind to the actual audio signal.
    """

    def __init__(self, cfg, log):
        import anthropic
        self._log = log
        api_key = cfg['anthropic_api_key'].get() or os.environ.get('ANTHROPIC_API_KEY', '')
        if not api_key:
            raise RuntimeError(
                'No Anthropic API key. Set ANTHROPIC_API_KEY or '
                'acoustic_enricher.anthropic_api_key in your beets config.'
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = cfg['model'].get()

    def analyze(self, audio_path, metadata):
        meta_lines = '\n'.join(
            f'{k.title()}: {v}'
            for k, v in metadata.items()
            if v
        )
        prompt = _TEXT_PROMPT.format(meta=meta_lines)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                messages=[{'role': 'user', 'content': prompt}],
            )
        except Exception as e:
            if _is_quota_error(e):
                raise _QuotaError(str(e)) from e
            raise
        return _parse_json(response.content[0].text)


_PROVIDERS = {
    'gemini':    _GeminiProvider,
    'ollama':    _OllamaProvider,
    'llamacpp':  _LlamaCppProvider,
    'anthropic': _AnthropicProvider,
}


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class AcousticEnricherPlugin(BeetsPlugin):
    """beet listen — slice audio, send to an LLM, write Roon-compatible tags.

    Tag mapping:
      mood         → TMOO (MP3) / MOOD (FLAC)         — Roon mood field
      roontracktag → TXXX:ROONTRACKTAG / ROONTRACKTAG — Roon tags
                     format: "style:<style>; <element>; <element>; ..."
      listen_style    → beets db only
      listen_elements → beets db only (raw JSON array)
    """

    name = 'acoustic_enricher'

    item_types = {
        'mood':                    dbcore.types.STRING,
        'roontracktag':            dbcore.types.STRING,
        'listen_style':            dbcore.types.STRING,
        'listen_elements':         dbcore.types.STRING,
        'listen_energy_shape':     dbcore.types.STRING,
        'listen_energy_peak':      dbcore.types.STRING,
        'listen_energy_narrative': dbcore.types.STRING,
        'listen_provider':         dbcore.types.STRING,
        'listen_model':            dbcore.types.STRING,
    }

    def __init__(self):
        super().__init__()
        self.config.add({
            'provider':         'gemini',
            'model':            'gemini-2.5-flash',
            # Gemini / Vertex AI
            'api_key':          os.environ.get('GEMINI_API_KEY', ''),
            'use_vertex':       False,
            'project':          os.environ.get('GOOGLE_CLOUD_PROJECT', ''),
            'location':         'us-central1',
            # Anthropic (Claude) — knowledge-based inference, not audio analysis
            'anthropic_api_key': os.environ.get('ANTHROPIC_API_KEY', ''),
            # Ollama
            'ollama_url':       'http://localhost:11434/api/chat',
            # llama.cpp
            'llama_url':        'http://localhost:8080/v1/chat/completions',
            # Sampling
            'clip_count':       3,
            'clip_duration':    15,
            # Failure handling
            'max_consecutive_failures': 0,  # 0 = disabled
        })
        _register_roon_media_fields(self)
        self._provider_instance = None

    # ------------------------------------------------------------------
    # Command registration
    # ------------------------------------------------------------------

    def commands(self):
        cmd = ui.Subcommand(
            'listen',
            help='Analyze audio with an LLM and write Roon-compatible tags',
        )
        cmd.parser.add_option(
            '-n', '--dry-run',
            action='store_true', dest='dry_run', default=False,
            help='print analysis without writing any tags',
        )
        cmd.parser.add_option(
            '-f', '--force',
            action='store_true', dest='force', default=False,
            help='re-analyze even if mood tag is already set',
        )
        cmd.func = self._cmd_listen
        return [cmd]

    # ------------------------------------------------------------------
    # Provider lazy-init
    # ------------------------------------------------------------------

    def _get_provider(self):
        if self._provider_instance is None:
            name = self.config['provider'].get()
            cls = _PROVIDERS.get(name)
            if cls is None:
                raise RuntimeError(
                    f'Unknown provider {name!r}. Valid choices: {", ".join(_PROVIDERS)}'
                )
            self._provider_instance = cls(self.config, self._log)
        return self._provider_instance

    # ------------------------------------------------------------------
    # Command handler
    # ------------------------------------------------------------------

    def _cmd_listen(self, lib, opts, args):
        items = list(lib.items(ui.decargs(args)))
        if not items:
            self._log.info('No tracks matched the query.')
            return

        provider_name = self.config['provider'].get()
        self._log.info('Initializing {0} provider...', provider_name)
        if provider_name == 'anthropic':
            self._log.info(
                'Note: Anthropic provider infers mood, style, elements, and '
                'energy arc from track metadata — Claude has no audio input '
                'API, so temporal arc is based on genre conventions, not '
                'actual audio analysis.'
            )
        provider = self._get_provider()

        total = len(items)
        analyzed = skipped = failed = consecutive_failures = 0
        max_consecutive = self.config['max_consecutive_failures'].get(int)
        for item in items:
            if not opts.force and item.get('mood'):
                self._log.debug('Skipping (already tagged): {0}', item.title)
                skipped += 1
                continue
            try:
                self._analyze(item, provider, provider_name, opts.dry_run)
                analyzed += 1
                consecutive_failures = 0
            except KeyboardInterrupt:
                self._log.info('Interrupted.')
                break
            except _QuotaError as e:
                self._log.error('Quota/spend cap exceeded — stopping: {0}', e)
                break
            except Exception as e:
                self._log.error('Failed — {0}: {1}', item.title, e)
                failed += 1
                consecutive_failures += 1
                if max_consecutive and consecutive_failures >= max_consecutive:
                    self._log.error(
                        'Stopping after {0} consecutive failures.', consecutive_failures
                    )
                    break

        scanned = analyzed + failed
        self._log.info(
            'Done [{0}/{1}]: {2} tracks matched — {3} scanned, {4} analyzed, {5} already tagged, {6} failed.',
            provider_name, self.config['model'].get(),
            total, scanned, analyzed, skipped, failed,
        )

    # ------------------------------------------------------------------
    # Per-track analysis
    # ------------------------------------------------------------------

    def _analyze(self, item, provider, provider_name, dry_run):
        path     = item.path.decode() if isinstance(item.path, bytes) else item.path
        duration = float(item.length or 0)

        if duration < 3:
            self._log.warning(
                'Track too short to analyze ({0:.0f}s): {1}', duration, item.title
            )
            return

        metadata = {
            'title':  str(item.get('title')  or ''),
            'artist': str(item.get('artist') or ''),
            'album':  str(item.get('album')  or ''),
            'year':   str(item.get('year')   or ''),
            'genre':  str(item.get('genre')  or ''),
        }

        samples = self._sample_windows(duration)

        os.makedirs('/tmp/beets', exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix='.mp3', dir='/tmp/beets', delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            self._ffmpeg_slice(path, samples, tmp_path)
            result = provider.analyze(tmp_path, metadata)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Coerce all values to plain strings — beets db must never see a list.
        mood  = str(result.get('mood',  '')).strip()
        style = str(result.get('style', '')).strip()
        elements = result.get('elements', [])
        if not isinstance(elements, list):
            elements = [str(elements)]
        elements = [str(e).strip() for e in elements if e]

        energy_arc = result.get('energy_arc', {})
        if not isinstance(energy_arc, dict):
            energy_arc = {}
        energy_shape     = str(energy_arc.get('shape',      '')).strip()
        energy_peak      = str(energy_arc.get('peak',       '')).strip()
        energy_narrative = str(energy_arc.get('trajectory', '')).strip()

        # ROONTRACKTAG: "style:X; arc:shape; peak:Y; <AudioSet elements...>"
        roon_parts = []
        if style:
            roon_parts.append(f'style:{style}')
        if energy_shape:
            roon_parts.append(f'arc:{energy_shape}')
        if energy_peak:
            roon_parts.append(f'peak:{energy_peak}')
        roon_parts.extend(elements)
        roon_tag = '; '.join(roon_parts)

        self._log.info(
            '{0}  |  mood={1!r}  style={2!r}  arc={3!r}  peak={4!r}  tags={5!r}',
            item.title, mood, style, energy_shape, energy_peak, roon_tag,
        )

        if dry_run:
            return

        item['mood']                   = mood
        item['listen_style']           = style
        item['listen_elements']        = json.dumps(elements)
        item['listen_energy_shape']    = energy_shape
        item['listen_energy_peak']     = energy_peak
        item['listen_energy_narrative'] = energy_narrative
        item['roontracktag']           = roon_tag
        item['listen_provider']        = provider_name
        item['listen_model']           = self.config['model'].get()
        item.store()   # write flexible fields to beets db
        stat = os.stat(path)
        item.write()   # write mood→TMOO and roontracktag→TXXX:ROONTRACKTAG to file
        os.utime(path, (stat.st_atime, stat.st_mtime))  # restore mtime so Roon won't re-import

    # ------------------------------------------------------------------
    # Audio slicing via ffmpeg
    # ------------------------------------------------------------------

    def _sample_windows(self, duration):
        clip_dur   = self.config['clip_duration'].get(int)
        clip_count = self.config['clip_count'].get(int)

        if duration < 30:
            return [(0.0, duration)]

        if duration < 60:
            start = max(0.0, duration / 2 - clip_dur / 2)
            return [(start, min(clip_dur, duration))]

        positions = [i / (clip_count + 1) for i in range(1, clip_count + 1)]
        return [
            (min(duration * p, duration - clip_dur), float(clip_dur))
            for p in positions
        ]

    def _ffmpeg_slice(self, input_path, samples, output_path):
        """Concatenate sample windows into a single low-bitrate mono MP3."""
        trim_filters = [
            f'[0:a]atrim=start={s:.3f}:duration={d:.3f},asetpts=PTS-STARTPTS[a{i}]'
            for i, (s, d) in enumerate(samples)
        ]
        concat_inputs = ''.join(f'[a{i}]' for i in range(len(samples)))
        filter_graph  = (
            ';'.join(trim_filters)
            + f';{concat_inputs}concat=n={len(samples)}:v=0:a=1[outa]'
        )
        result = subprocess.run(
            [
                'ffmpeg', '-y', '-i', input_path,
                '-filter_complex', filter_graph,
                '-map', '[outa]',
                '-codec:a', 'libmp3lame', '-b:a', '64k',
                '-ar', '22050', '-ac', '1',
                output_path,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f'ffmpeg failed: {result.stderr.decode(errors="replace")[-500:]}'
            )
