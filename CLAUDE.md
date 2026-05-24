# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file Beets plugin (`acoustic_enricher.py`) that implements the `beet listen` command. It slices audio tracks with `ffmpeg`, sends clips to an LLM for MIR analysis, and writes results as Roon-compatible tags directly into audio files.

There are no tests, no build steps, and no package management — the plugin is installed by dropping the `.py` file into the beets `pluginpath`.

## Running the plugin locally

```bash
# Install dependencies for your provider
pip install beets google-genai          # Gemini (default)
pip install beets anthropic             # Claude
pip install beets requests              # Ollama / llama.cpp

# ffmpeg must be on $PATH
ffmpeg -version

# Run against your beets library
beet listen album:"Dark Side of the Moon"
beet listen -f artist:"Boards of Canada"   # force re-analyze
beet listen -n album:"Dummy"               # dry-run, no writes
```

## Architecture

Everything lives in `acoustic_enricher.py`. The file is structured in three layers:

1. **Prompts** (`_AUDIO_PROMPT`, `_TEXT_PROMPT`) — the LLM instructions. `_AUDIO_PROMPT` is sent with actual audio (Gemini, Ollama, llama.cpp); `_TEXT_PROMPT` is sent with track metadata only (Anthropic, which has no audio API).

2. **Provider classes** (`_GeminiProvider`, `_OllamaProvider`, `_LlamaCppProvider`, `_AnthropicProvider`) — each implements a single `analyze(audio_path, metadata)` method. All audio providers receive the temp MP3 concatenated by ffmpeg. Quota errors are normalized to `_QuotaError` so the batch loop can stop cleanly regardless of provider.

3. **`AcousticEnricherPlugin`** (the beets plugin) — registers the `listen` command and the custom item fields, lazily initializes the provider, drives the per-track loop with skip/force/failure logic, slices audio via `_ffmpeg_slice`, and writes tags in two steps: `item.store()` for beets DB fields then `item.write()` for audio file tags, followed by `os.utime()` to restore mtime so Roon won't re-import.

## Tag writing details

The plugin writes to two destinations per track:
- **Audio file**: `mood → TMOO` (MP3) / `MOOD` (FLAC/Ogg) and `roontracktag → TXXX:ROONTRACKTAG` — these are the tags Roon reads
- **Beets DB only**: `listen_style`, `listen_elements` (JSON array), `listen_energy_shape/peak/narrative`, `listen_provider`, `listen_model`

The `roontracktag` value is formatted as `style:<style>; arc:<shape>; peak:<level>; <AudioSet element>; ...`

`ROONTRACKTAG` is registered as a custom `MediaField` at plugin init via beets' `mediafile` API, since it's not a standard field.

## Adding a new provider

1. Add a class with `__init__(self, cfg, log)` and `analyze(self, audio_path, metadata) -> dict`
2. Wrap provider-specific quota/rate-limit exceptions in `_QuotaError`
3. Return a dict with keys: `mood` (str), `style` (str), `elements` (list[str]), `energy_arc` (dict with `shape`, `base`, `peak`, `trajectory`)
4. Register it in the `_PROVIDERS` dict
5. Add config defaults in `AcousticEnricherPlugin.__init__` if the provider needs new config keys
6. Document it in `config.enricher.yaml`
