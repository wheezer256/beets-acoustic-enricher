# beets-acoustic-enricher

A [Beets](https://beets.io/) plugin (`acoustic_enricher`) that samples your
tracks with `ffmpeg`, has an audio-capable LLM listen to them, and stores what
it hears in **one controlled schema** built on established ontologies:

| Part | Vocabulary |
|---|---|
| **mood** (expressed) | 1–3 of 59 terms in 12 families, each placed on valence × arousal × dominance (Russell circumplex + Mehrabian PAD), see `acoustic_enricher/data/moods.yaml` |
| **felt** | 0–2 terms for what a listener feels (GEMS / Cowen & Keltner: awe, entranced, chills, comforted, urge to move, …) |
| **use** | 0–3 occasions (focus, dance party, night driving, wedding, worship, …; from the AllMusic themes) |
| **time & place** | 0–2 (early morning, late night, rain/monsoon, sacred space, …) |
| **character** | 0–2 sonic / aesthetic / cultural terms (hypnotic, raw, lush, sublime, saudade, yūgen, …) |
| **dimensions** | 16 single answers judged by ear: tension, stirring, musical happiness and sadness (separately: both high = bittersweet), attention demand, complexity, accessibility, rhythmic complexity, anthem, lyric intelligibility, speech (none/sampled/spoken), groove, recording fidelity, purpose, format, musical stance |
| **style** | Discogs styles and MusicBrainz genres |
| **elements** | AudioSet ontology class names (instruments, vocals, sounds, production) |
| **energy arc** | one of 11 shapes (steady-low … breakdown-drop), base and peak level |
| **rhythm** | felt BPM, pulse (none/soft/steady/driving), character (steady/shifting/fragmented) |

The facets and dimensions are "ontology v3" (`data/ontology_v3.yaml`); each facet answers
one question, so what the music *expresses*, what it makes you *feel*, what it is *for* and
how it *sounds* are never mixed in one list.

Mood and arc are enforced with structured output where the provider supports
it; style and elements are mapped onto the canonical lists after the call, and
anything that does not map is dropped and recorded in `listen_unmapped`. From
the mood terms the plugin derives numeric **valence/arousal**, and from the
AudioSet elements an **instrumental** flag, so tracks can be ranked and
filtered (e.g. for mood playlists) without a second labelling scheme.

## Features

- **`beet listen QUERY`** analyses tracks; skips those already at the current
  schema version (`-f` to redo, `-n` dry run).
- **`beet listen-batch submit|status|collect`**: the same analysis through the
  Gemini **Batch API** (about half price) for large libraries, with a cost
  estimate and a hard spend cap.
- **`beet listen-normalize QUERY`**: re-map labels written by older versions
  onto the vocabularies (no LLM calls).
- **Providers**
  - `gemini` (default): Google Gemini, audio sent inline, thinking disabled by default
  - `openai`: any OpenAI-compatible endpoint with `input_audio` (LiteLLM proxy, vLLM, OpenRouter, ...)
  - `llamacpp`: llama.cpp server with audio input
  - `ollama`: Ollama with an audio/multimodal model
  - `anthropic`: metadata-only inference via Claude (no audio input)
- **Sampling**: three 15 s clips, either at 25/50/75 % (`clip_strategy: fixed`, reads
  only those seconds) or aimed at the most rhythmic / loudest / distinct sections after
  one decode of the track (`adaptive`, which also measures BPM, intro length, onset rate,
  treble harshness, tail silence and beat strength).
- **Several runs per track** (`runs`, e.g. 3), combined: terms keep the share of runs
  that chose them, single answers take a majority vote, numbers the mean or median;
  `listen_agreement` is a per-track confidence.
- **Catalogue context** (`include_metadata`): artist / title / album / year / genre / label
  are sent after the audio, for meaning the audio cannot carry (ritual purpose, irony, era).
  In a blind test the owner preferred the context-aware reading 12:7, and it made repeated
  runs more consistent on every measure.
- **Prompt caching**: the instructions (identical for every track) go first, then the audio,
  then the per-track context, so providers can cache the shared prefix.
- **`--db-only`** stores everything in the beets database without touching
  your audio files; otherwise `MOOD` and `ROONTRACKTAG` are written to the
  files too (file mtime preserved).
- Optional JSONL **usage ledger** (tokens per track) for cost tracking.

## Fields

| Field | Stored |
|---|---|
| `mood` | DB + file (`TMOO` / `MOOD` / MP4 `MOOD` / `WM/Mood`) |
| `roontracktag` | DB + file (`ROONTRACKTAG`), `style:…; arc:…; peak:…; <elements>` |
| `listen_style`, `listen_elements` (JSON), `listen_energy_shape/base/peak/narrative` | DB |
| `listen_valence`, `listen_arousal` (−1…1, float) | DB, mean of the mood terms' circumplex coordinates |
| `listen_instrumental` (0…1) | DB, share of runs that heard no AudioSet vocal class |
| `listen_dominance` (−1…1, float) | DB, mean of the mood terms' dominance |
| `listen_v3` (JSON) | DB, felt / use / timeplace / character term shares and the 16 dimensions (answer, share, value) |
| `listen_tempo`, `listen_bpm`, `listen_bpm_runs`, `listen_pulse`, `listen_rhythm` | DB, felt tempo and rhythm |
| `listen_bpm_measured`, `listen_intro_s`, `listen_onset_rate`, `listen_harsh`, `listen_tail_silence_s`, `listen_beat_strength` | DB, measured (adaptive clips) |
| `listen_runs`, `listen_agreement` | DB, runs combined and their agreement |
| `listen_unmapped` | DB, labels dropped because they were not in a vocabulary |
| `listen_ver`, `listen_provider`, `listen_model` | DB |

Numeric fields are typed, so range queries work: `beet ls listen_arousal:..-0.3 listen_instrumental:1`.

## Install

Requires Python 3.10+, `beets`, `ffmpeg` on `$PATH`, and the SDK for your
provider (`google-genai` for gemini/batch, `requests` for openai/llamacpp/ollama,
`anthropic` for anthropic).

Clone the repository and point `pluginpath` at it:

```yaml
pluginpath:
  - /path/to/beets-acoustic-enricher
plugins:
  - acoustic_enricher
acoustic_enricher:
  provider: gemini
  model: gemini-3.5-flash-lite
  api_key: ${GEMINI_API_KEY}     # or api_key_file: /run/secrets/gemini
```

Through an OpenAI-compatible proxy such as LiteLLM:

```yaml
acoustic_enricher:
  provider: openai
  openai_url: http://localhost:4000/v1/chat/completions
  openai_api_key_file: /run/secrets/litellm_key
  model: gemini-flash            # the proxy's model alias
  reasoning_effort: disable      # no thinking tokens (cheaper, faster)
```

See `config.enricher.yaml` for every option.

## Usage

```bash
beet listen album:"Dark Side of the Moon"
beet listen -n artist:"Boards of Canada"        # preview, store nothing
beet listen --db-only genre:ambient            # DB only, files untouched

# Large runs via the Gemini Batch API
beet listen-batch submit -n --max-usd 50 added:2020..   # estimate only
beet listen-batch submit --max-usd 50 added:2020..
beet listen-batch status
beet listen-batch collect --wait

beet listen-normalize roontracktag::.          # tidy labels from older versions
```

## Cost

Gemini counts 32 tokens per second of audio, so the default 45 s of clips is
about 1,440 audio tokens. With ontology v3 the instructions add ~2,450 prompt tokens and
the answer ~450 output tokens per run (thinking minimal); the shared instructions are
cacheable (measured: 69 % of input tokens cached from the second track on). Per run,
3.5 Flash-Lite costs about $0.0026 interactive / $0.0013 batch before caching; multiply by
`runs`. The table below is the earlier, smaller v2 prompt on a 370-track sample (2026-10):

| Model | Interactive | Batch API |
|---|---|---|
| `gemini-3.5-flash-lite` (default) | ~$0.0011 / track | ~$0.0005 / track |
| `gemini-2.5-flash` | ~$0.0016 / track | ~$0.0008 / track |

On that sample Flash-Lite 3.5 matched or beat 2.5 Flash on arousal/valence
agreement with Essentia's models and kept to the vocabularies better, at about
two-thirds of the price. (`gemini-2.5-flash-lite` is not offered to new API
users.) Check current prices before large runs; override the built-in table
with `prices:`.

### Batch API trade-offs

- Results are asynchronous (usually hours, up to ~24 h); run the interactive
  command on a sample first to settle model and settings.
- No free tier for batch requests.
- Requests are split into chunks (`batch_chunk`, default 1000 tracks) because
  each carries inline audio.
- `batch_max_usd` (default **10**) refuses any submission that would push the
  committed total over the cap; raise it explicitly for big runs.
- Raw result files are kept in `batch_dir`, so re-ingesting never re-pays.

## Privacy

Audio providers receive short clips of your music (default three 15 s
excerpts). Use `ollama` or `llamacpp` to keep everything local.

## Development

```bash
pip install beets pyyaml requests pytest
pytest
```

## Licence

Code: MIT. Bundled vocabulary data: see `acoustic_enricher/data/ATTRIBUTION.md`
(AudioSet ontology CC BY-SA 4.0; MusicBrainz genres CC0; Discogs style names).
