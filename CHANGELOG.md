# Changelog

## 2.0.0 (2026-10-16)

### Ontology v3
- Mood terms (`data/moods.yaml` v3): 59 terms in 12 families with valence, arousal and
  dominance; sound-character words (hypnotic, cinematic, …) moved to the character facet;
  new: defiant, humorous, wry, scary, dignified, vulnerable, resigned, bitter, carefree,
  and a desire family (flirtatious, seductive, sensual, erotic, yearning, obsessive, devoted).
- New facets (`data/ontology_v3.yaml`): felt, use, time & place, character; 16 listened
  dimensions with ordinal or categorical answers. Stored as JSON in `listen_v3`;
  `listen_dominance` added. `SCHEMA_VERSION` 5.
- `include_metadata`: catalogue context after the audio; `runs` with `combine()`;
  `clip_strategy: adaptive` with measured descriptors.
- Request order instructions → audio → context for prompt caching; batch cost accounts
  cached tokens (`cached_price_factor`).

### Changed
- One controlled schema: mood terms from `data/moods.yaml` (Russell circumplex),
  styles from Discogs/MusicBrainz, elements from the AudioSet ontology, energy arc
  from 11 shapes. Results are canonicalised; unmapped labels go to `listen_unmapped`.
- The plugin is now a package (`acoustic_enricher/`); point `pluginpath` at the
  repository root. Config keys and provider names are unchanged.
- Clips are read with input seeking instead of decoding the whole file.
- Gemini: audio sent inline, structured output enforced, thinking disabled by default.
- Default model is now `gemini-3.5-flash-lite` (Gemini 3 models use `thinking_level`).
- Tracks are skipped when already at the current `listen_ver` (was: when `mood` set).

### Added
- `openai` provider for OpenAI-compatible endpoints (LiteLLM, vLLM, OpenRouter, ...).
- `listen_valence`, `listen_arousal`, `listen_instrumental`, `listen_energy_base`,
  `listen_unmapped`, `listen_ver` fields.
- `--db-only` / `--write-tags`, `db_only` and `preserve_mtime` options.
- `beet listen-batch submit|status|collect` (Gemini Batch API) with cost estimate
  and `batch_max_usd` cap; `beet listen-normalize`.
- Optional JSONL usage ledger; tests (pytest).

### Fixed
- `mood` is now written to audio files (`MOOD` / `TMOO`); previously it was only
  set in the beets DB because mediafile has no built-in mood field.

## 1.0.x
- Initial release; configurable HTTP timeout for Ollama and llama.cpp.
