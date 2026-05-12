# beets-acoustic-enricher

A [Beets](https://beets.io/) plugin (`acoustic_enricher`) that slices your
tracks with `ffmpeg`, sends the audio to an LLM for Music Information Retrieval
(MIR) analysis, and writes results as Roon-compatible tags.

## Features

- **`beet listen`** — analyze any query (`album:`, `artist:`, etc.) on demand
- **Roon-native tags** — writes `MOOD`/`TMOO` and `ROONTRACKTAG` directly into
  the audio file so Roon picks them up without re-importing
- **Energy arc** — shape, base/peak level, and a narrative description of how
  the track evolves
- **Four providers**
  - `gemini` (default) — real audio analysis via Google Gemini
  - `ollama` — local multimodal LLM via Ollama
  - `llamacpp` — llama.cpp OpenAI-compatible server with native audio input
  - `anthropic` — metadata-based inference via Claude (no audio API, but useful
    for well-known artists)
- **Smart sampling** — configurable number of clips spread across the track,
  concatenated into a single low-bitrate mono MP3 before upload
- `--dry-run` and `--force` flags; quota-error detection stops the batch cleanly

## Tag mapping

| Field | Where written |
|---|---|
| `mood` | `TMOO` (MP3) / `MOOD` (FLAC/Ogg) — Roon mood field |
| `roontracktag` | `TXXX:ROONTRACKTAG` / `ROONTRACKTAG` — Roon tags |
| `listen_style` | Beets DB only |
| `listen_elements` | Beets DB only (JSON array) |
| `listen_energy_shape` / `_peak` / `_narrative` | Beets DB only |
| `listen_provider` / `listen_model` | Beets DB only |

## Requirements

- Python 3.8+
- `beets`
- `ffmpeg` (must be on `$PATH`)
- Provider SDK: `google-genai`, `anthropic`, or `requests` (Ollama/llama.cpp)

```bash
pip install beets google-genai          # for Gemini
# pip install beets anthropic           # for Claude
# pip install beets requests            # for Ollama / llama.cpp
```

## Setup

Copy `acoustic_enricher.py` to your Beets plugin directory, then add to
`config.yaml`:

```yaml
pluginpath:
    - /path/to/plugins

plugins:
    - acoustic_enricher

acoustic_enricher:
    provider: gemini
    model: gemini-2.5-flash
    api_key: ${GEMINI_API_KEY}
```

See `config.enricher.yaml` for all options including Vertex AI, Anthropic,
Ollama, and llama.cpp examples.

## Usage

```bash
# Analyze an album (skips already-tagged tracks)
beet listen album:"Dark Side of the Moon"

# Re-analyze everything by an artist
beet listen -f artist:"Boards of Canada"

# Preview without writing tags
beet listen -n album:"Dummy"
```

## Docker

See `Dockerfile.enricher` for an example of adding the plugin to an existing
beets container.

## License

MIT
