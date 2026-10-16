"""Prompts and the structured-output schema, both generated from the vocabularies.

Mood and the energy arc are constrained with enums, so providers that support
structured output can only return valid values. Style and elements are left as
free strings on purpose: enumerating ~2,300 genres and 632 AudioSet classes in
every request would add thousands of input tokens per track, so they are
canonicalised after the call instead (see vocab.canonicalize).
"""

from __future__ import annotations

from . import ontology as O
from . import vocab as V

# Bump when the prompt or schema changes in a way that makes older results
# non-comparable. Stored per track as `listen_ver`.
SCHEMA_VERSION = '5'   # 5 = ontology v3 (felt/use/time-place/character facets + listened dims)


def _mood_list() -> str:
    return ', '.join(V.load().mood_terms())


V3_KEYS = ['felt', 'use', 'timeplace', 'character', 'dims']
KEYS = ['mood', 'style', 'elements', 'energy_arc', 'rhythm'] + V3_KEYS


def context_text(metadata: dict | None) -> str:
    """Optional catalogue context, sent AFTER the audio (see request_parts). It may inform
    MEANING (ritual purpose, irony, era, cultural concepts) but never what is heard."""
    lines = [f'{k.title()}: {v}' for k, v in (metadata or {}).items() if v]
    if not lines:
        return ''
    return ('Catalogue context for this recording (may be incomplete or wrong). Use it only '
            'for meaning the audio alone cannot carry, such as ritual or ceremonial purpose, '
            'irony or pastiche, era and scene; judge everything you can hear from the audio '
            'itself:\n' + '\n'.join(lines))


def request_parts(metadata: dict | None = None) -> tuple[str, str]:
    """(instructions, context) around the audio: [instructions, AUDIO, context].

    The instructions are identical for every track and go FIRST so providers can cache
    them as a shared prompt prefix (and the audio too across a track's repeated runs);
    the per-track context goes LAST so it never breaks that prefix."""
    return audio_prompt(), context_text(metadata)


def audio_prompt(metadata: dict | None = None) -> str:
    """The instructions; with `metadata` the context is appended (single-text callers)."""
    ctx = context_text(metadata)
    return (
        'You will receive audio clips sampled from the beginning, middle, and end of '
        'a track, in chronological order. Listen to how the energy and character '
        'evolve across these clips.\n\n'
        f'Return a JSON object with exactly {len(KEYS)} keys:\n\n'
        '  "mood": 1-3 terms for the mood the music EXPRESSES (what it portrays, not '
        'what a listener feels; that goes in "felt"). Choose ONLY from: '
        f'{_mood_list()}.\n\n'
        '  "style": 1-3 genre or style names exactly as they appear in the Discogs '
        'style list or the MusicBrainz genre list. Use sub-genres where the '
        'recording warrants it (e.g. "minimal techno", "free jazz", "fingerstyle", '
        '"drum and bass", "post-bop", "darkwave", "shoegaze", "cumbia").\n\n'
        '  "elements": an array of 8-15 class names copied exactly from the AudioSet '
        'ontology (https://research.google.com/audioset/ontology/index.html), e.g. '
        '"Synthesizer", "Drum machine", "Bass drum", "Piano", "Acoustic guitar", '
        '"Male singing", "Female singing", "Rapping", "Choir", "Speech", '
        '"Reverberation", "Ambient music", "Rain". Cover instruments, vocals, '
        'percussion, environmental and sound-design events, and production. Do not '
        'invent labels that are not AudioSet class names.\n\n'
        '  "energy_arc": how energy evolves through the track. The clips are in '
        'chronological order; compare their relative energy, density and tension. '
        'Return "shape" (one of the allowed shapes), "base" (resting energy) and '
        '"peak" (maximum energy), each low/medium/high, and "trajectory" (a 5-10 '
        'word narrative of the arc).\n\n'
        '  "rhythm": judged by ear, not inferred from genre. "bpm": the FELT tempo '
        'as a number of beats per minute, the pace a listener would nod or tap to; '
        'count a half-time feel as heard (half-time dubstep is ~70, not 140; a slow '
        'jam over fast hi-hats is the slow figure). If there is no beat, estimate the '
        'pace from how fast the notes and phrases move (a racing solo cello is fast '
        'even without drums). "pulse": how prominent the beat is at its strongest in '
        'ANY clip: none (no regular beat: drones, rubato, free time), soft (light or '
        'implied: brushes, hand percussion, a plucked pulse), steady (a clear regular '
        'beat) or driving (a prominent, insistent beat, usually kick-drum led). '
        '"character": steady (regular throughout), shifting (tempo or feel changes '
        'between sections) or fragmented (stop-start, cut-up; beats or sounds '
        'dropping in and out).\n\n'
        + O.prompt_section() + '\n\n'
        'Return only the JSON object.'
        + ('\n\n' + ctx if ctx else '')
    )


def text_prompt(meta_lines: str) -> str:
    """Prompt for metadata-only providers (no audio input)."""
    return (
        'You are a music expert. Based on the track metadata below, infer the '
        'likely sonic character using your knowledge of the artist, album and '
        f'genre.\n\n{meta_lines}\n\n'
        'Return a JSON object with keys "mood" (1-3 terms, choose ONLY from: '
        f'{_mood_list()}), "style" (1-3 Discogs styles or MusicBrainz genres), '
        '"elements" (8-15 AudioSet ontology class names) and "energy_arc" '
        '(shape, base, peak, trajectory) and "rhythm" (bpm as a number, pulse '
        'none/soft/steady/driving, character steady/shifting/fragmented); note these '
        'are inferred from metadata, not audio. Return only the JSON object.'
    )


def json_schema() -> dict:
    """JSON Schema for the result (OpenAI `response_format` / Gemini schema)."""
    level = {'type': 'string', 'enum': list(V.ENERGY_LEVELS)}
    return {
        'type': 'object',
        'properties': {
            'mood': {
                'type': 'array',
                'items': {'type': 'string', 'enum': V.load().mood_terms()},
                'minItems': 1, 'maxItems': 3,
            },
            'style': {
                'type': 'array', 'items': {'type': 'string'},
                'minItems': 1, 'maxItems': 3,
            },
            'elements': {
                'type': 'array', 'items': {'type': 'string'},
                'minItems': 1, 'maxItems': 20,
            },
            'energy_arc': {
                'type': 'object',
                'properties': {
                    'shape': {'type': 'string', 'enum': list(V.ARC_SHAPES)},
                    'base': level,
                    'peak': level,
                    'trajectory': {'type': 'string'},
                },
                'required': ['shape', 'base', 'peak', 'trajectory'],
            },
            'rhythm': {
                'type': 'object',
                'properties': {
                    'bpm': {'type': 'number'},
                    'pulse': {'type': 'string', 'enum': list(V.PULSES)},
                    'character': {'type': 'string', 'enum': list(V.CHARACTERS)},
                },
                'required': ['bpm', 'pulse', 'character'],
            },
        **O.schema_properties(),
        },
        'required': KEYS,
    }


def gemini_schema() -> dict:
    """The same schema in the OpenAPI subset Gemini's `response_schema` accepts
    (no minItems/maxItems on nested arrays is fine; no `additionalProperties`)."""
    s = json_schema()
    s['propertyOrdering'] = list(KEYS)
    return s
