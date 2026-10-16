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

The plugin is the package `acoustic_enricher/` (point beets' `pluginpath` at the repo root):

- `vocab.py` — loads the controlled vocabularies in `data/` (moods with circumplex
  coordinates, Discogs styles, MusicBrainz genres, AudioSet ontology + aliases) and
  `canonicalize()`s a raw provider result. No beets import; unit-testable.
- `schema.py` — prompts and the JSON schema generated from the vocabularies;
  `SCHEMA_VERSION` is stored per track as `listen_ver`.
- `providers.py` — gemini / openai(-compatible) / llamacpp / ollama / anthropic;
  each `analyze(audio_path, metadata) -> dict` and sets `last_usage`.
- `plugin.py` — the beets plugin: `listen`, `listen-normalize`, field writes, clip
  extraction (ffmpeg input seeking), ledger.
- `batch.py` — `listen-batch` (Gemini Batch API): chunked JSONL submit with cost cap,
  status, collect/ingest through the same canonicalisation path.

Tests live in `tests/` (`pytest`).

## Adding a new provider

1. Add a class with `__init__(self, cfg, log)` and `analyze(self, audio_path, metadata) -> dict`
2. Wrap provider-specific quota/rate-limit exceptions in `QuotaError`
3. Return a dict with keys: `mood` (list[str]), `style` (list[str] or str), `elements` (list[str]), `energy_arc` (dict with `shape`, `base`, `peak`, `trajectory`); set `self.last_usage`
4. Register it in `PROVIDERS` (and `AUDIO_PROVIDERS` if it hears audio)
5. Add config defaults in `AcousticEnricherPlugin.__init__` if the provider needs new config keys
6. Document it in `config.enricher.yaml`
