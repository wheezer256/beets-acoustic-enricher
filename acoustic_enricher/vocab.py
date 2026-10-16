"""Controlled vocabularies and canonicalisation for acoustic_enricher.

The enricher describes a recording in four parts, each tied to an existing
ontology so labels stay comparable across tracks, providers and runs:

  mood        terms from data/moods.yaml (Russell circumplex placements)
  style       Discogs styles (data/discogs_styles.txt) and MusicBrainz genres
              (data/musicbrainz_genres.txt)
  elements    AudioSet ontology class names (data/audioset.json)
  energy_arc  one of ARC_SHAPES, with base/peak in ENERGY_LEVELS
  rhythm      felt tempo in BPM (-> TEMPOS), beat prominence (PULSES) and
              rhythmic character (CHARACTERS)

Nothing here imports beets, so it can be unit-tested and reused on its own.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache

import yaml

from . import ontology as O

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

ARC_SHAPES = (
    'steady-low', 'steady-mid', 'steady-high', 'building', 'arch', 'valley',
    'oscillating', 'explosive', 'breakdown-drop', 'decaying', 'cyclic',
)
ENERGY_LEVELS = ('low', 'medium', 'high')
# Felt pace of the music, derived from the felt BPM (half-time feels count as heard,
# so 140 BPM dubstep in half-time is ~70). Bands: slow < 85 <= medium < 115 <= fast
# < 135 <= very fast.
TEMPOS = ('slow', 'medium', 'fast', 'very fast')
TEMPO_BANDS = ((85, 'slow'), (115, 'medium'), (135, 'fast'), (10_000, 'very fast'))
BPM_RANGE = (30, 250)
# How the rhythm is organised: steady (regular throughout), shifting (changes
# tempo/feel between sections), fragmented (stop-start, cut-up, beats or sounds
# dropping in and out).
CHARACTERS = ('steady', 'shifting', 'fragmented')


def tempo_of(bpm: float | None) -> str:
    if not bpm:
        return ''
    return next(name for limit, name in TEMPO_BANDS if bpm < limit)
# How prominent the beat is, at its strongest in any clip: none (no regular beat:
# drones, rubato, free time), soft (light or implied: brushes, hand percussion,
# plucked pulse), steady (clear regular beat), driving (prominent, insistent beat,
# usually kick-drum led).
PULSES = ('none', 'soft', 'steady', 'driving')

# AudioSet subtrees whose presence means the track has vocals: Speech, Singing
# (Choir, Chant, Rapping, Male/Female singing, ...), Humming, Vocal music
# (A capella, Beatboxing). Deliberately NOT the whole "Human voice" branch, which
# also holds laughter, crying, coughs and other non-vocal-performance sounds.
_VOCAL_ROOTS = ('/m/09x0r', '/m/015lz1', '/m/02fxyj', '/m/0y4f8')
# The bare "Human voice" class itself also counts as vocal (models often say just
# "Voice"), but not its other children (Laughter, Crying, Sigh, ...).
_VOCAL_NODES = ('/m/09l8g',)

# Close-match threshold for free-text style/element labels (difflib ratio).
_FUZZY_CUTOFF = 0.88


def _norm(s: str) -> str:
    """Case-, accent- and punctuation-insensitive key for matching labels."""
    s = unicodedata.normalize('NFKD', str(s)).casefold()
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.replace('&', ' and ')
    return re.sub(r'[\W_]+', ' ', s).strip()


@dataclass(frozen=True)
class MoodTerm:
    name: str
    valence: float
    arousal: float
    dominance: float = 0.0          # v3: -1 overwhelmed .. +1 in control


@dataclass
class Vocab:
    moods: dict[str, MoodTerm]
    mood_alias: dict[str, str]            # norm(alias or term) -> term
    styles: dict[str, str]                # norm(label) -> canonical display label
    elements: dict[str, str]              # norm(name) -> canonical AudioSet name
    vocal_elements: frozenset[str]        # canonical AudioSet names under Human voice
    version: str = field(default='1')

    # -- mood ---------------------------------------------------------------
    def mood_terms(self) -> list[str]:
        return list(self.moods)

    def canon_mood(self, label: str) -> str | None:
        return self.mood_alias.get(_norm(label))

    # -- style --------------------------------------------------------------
    def canon_style(self, label: str) -> str | None:
        key = _norm(label)
        if not key:
            return None
        if key in self.styles:
            return self.styles[key]
        hit = difflib.get_close_matches(key, self.styles.keys(), n=1, cutoff=_FUZZY_CUTOFF)
        return self.styles[hit[0]] if hit else None

    # -- elements -----------------------------------------------------------
    def canon_element(self, label: str) -> str | None:
        key = _norm(label)
        if not key:
            return None
        if key in self.elements:
            return self.elements[key]
        hit = difflib.get_close_matches(key, self.elements.keys(), n=1, cutoff=_FUZZY_CUTOFF)
        return self.elements[hit[0]] if hit else None


def _split_labels(value) -> list[str]:
    """Accept a list, or a comma/semicolon separated string, of labels."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = re.split(r'[;,]', str(value))
    return [str(p).strip() for p in parts if str(p).strip()]


@lru_cache(maxsize=1)
def load(data_dir: str = DATA_DIR) -> Vocab:
    with open(os.path.join(data_dir, 'moods.yaml'), encoding='utf-8') as f:
        mdoc = yaml.safe_load(f)
    moods, alias = {}, {}
    for name, spec in mdoc['terms'].items():
        moods[name] = MoodTerm(name, float(spec['valence']), float(spec['arousal']), float(spec.get('dominance', 0)))
        alias[_norm(name)] = name
        for a in spec.get('aliases', []) or []:
            alias.setdefault(_norm(a), name)

    styles: dict[str, str] = {}
    # MusicBrainz genres (CC0) are the primary display form: lowercase names.
    with open(os.path.join(data_dir, 'musicbrainz_genres.txt'), encoding='utf-8') as f:
        for line in f:
            g = line.strip()
            if g:
                styles.setdefault(_norm(g), g)
    # Discogs "Genre---Style": both the style and the parent genre are valid.
    with open(os.path.join(data_dir, 'discogs_styles.txt'), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            genre, _, style = line.partition('---')
            for label in (style, genre):
                if label:
                    styles.setdefault(_norm(label), label.lower())

    with open(os.path.join(data_dir, 'audioset.json'), encoding='utf-8') as f:
        onto = json.load(f)
    by_id = {c['id']: c for c in onto}
    elements: dict[str, str] = {}
    for c in onto:
        # Abstract/blacklisted classes are still valid names to keep.
        elements.setdefault(_norm(c['name']), c['name'])
    for c in onto:
        # "Male speech, man speaking" is also reachable as "Male speech" etc.
        if ',' in c['name']:
            for part in c['name'].split(','):
                elements.setdefault(_norm(part), c['name'])
    alias_path = os.path.join(data_dir, 'audioset_aliases.yaml')
    if os.path.exists(alias_path):
        with open(alias_path, encoding='utf-8') as f:
            for alias_name, target in (yaml.safe_load(f) or {}).items():
                canon = elements.get(_norm(target))
                if canon:
                    elements.setdefault(_norm(alias_name), canon)

    vocal: set[str] = set()
    stack = list(_VOCAL_ROOTS)
    while stack:
        cid = stack.pop()
        node = by_id.get(cid)
        if not node or node['name'] in vocal:
            continue
        vocal.add(node['name'])
        stack.extend(node.get('child_ids', []))

    for cid in _VOCAL_NODES:
        if cid in by_id:
            vocal.add(by_id[cid]['name'])

    return Vocab(moods=moods, mood_alias=alias, styles=styles, elements=elements,
                 vocal_elements=frozenset(vocal), version=str(mdoc.get('version', 1)))


@dataclass
class Canonical:
    """A provider result mapped onto the controlled vocabularies."""
    moods: list[str]
    styles: list[str]
    elements: list[str]
    shape: str
    base: str
    peak: str
    narrative: str
    unmapped: dict[str, list[str]]
    tempo: str = ''
    pulse: str = ''
    bpm: float | None = None        # felt BPM (LLM); see plugin for measured BPM
    character: str = ''
    bpm_runs: tuple = ()            # each run's felt BPM (diagnostics: run agreement)
    # Set by combine() when several runs were merged; a single run leaves these at
    # their defaults and the properties below derive from the labels.
    runs: int = 1
    agreement: float | None = None
    valence_override: float | None = None
    arousal_override: float | None = None
    instrumental_override: float | None = None
    dominance_override: float | None = None
    # Ontology v3 facets/dimensions (see ontology.py): one run's canonical dict, or after
    # combine() the merged form with shares. Stored as JSON in `listen_v3`.
    v3: dict | None = None

    @property
    def dominance(self) -> float | None:
        if self.dominance_override is not None:
            return self.dominance_override
        return _mean([load().moods[m].dominance for m in self.moods])

    @property
    def valence(self) -> float | None:
        if self.valence_override is not None:
            return self.valence_override
        return _mean([load().moods[m].valence for m in self.moods])

    @property
    def arousal(self) -> float | None:
        if self.arousal_override is not None:
            return self.arousal_override
        return _mean([load().moods[m].arousal for m in self.moods])

    @property
    def instrumental(self) -> float | None:
        """1.0 if no AudioSet vocal class was heard, 0.0 if one was; with several
        runs, the share of runs that heard no vocals.

        None when there are no elements at all (nothing to judge from)."""
        if self.instrumental_override is not None:
            return self.instrumental_override
        if not self.elements:
            return None
        vocal = load().vocal_elements
        return 0.0 if any(e in vocal for e in self.elements) else 1.0

    def roontracktag(self) -> str:
        """ROONTRACKTAG value: "style:<s>; arc:<shape>; peak:<lvl>; <element>; ..."."""
        parts = []
        if self.styles:
            parts.append('style:' + ', '.join(self.styles))
        if self.shape:
            parts.append(f'arc:{self.shape}')
        if self.peak:
            parts.append(f'peak:{self.peak}')
        parts.extend(self.elements)
        return '; '.join(parts)


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def _dedupe(xs: list[str]) -> list[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def canonicalize(result: dict, vocab: Vocab | None = None) -> Canonical:
    """Map a raw provider result onto the vocabularies.

    Labels that cannot be mapped are dropped and reported in `unmapped`, so a
    caller can log them; nothing outside the vocabularies is ever returned.
    """
    v = vocab or load()
    unmapped: dict[str, list[str]] = {'mood': [], 'style': [], 'elements': [], 'energy_arc': [],
                                      'rhythm': []}

    moods = []
    for m in _split_labels(result.get('mood')):
        c = v.canon_mood(m)
        (moods.append(c) if c else unmapped['mood'].append(m))

    styles = []
    for s in _split_labels(result.get('style')):
        c = v.canon_style(s)
        (styles.append(c) if c else unmapped['style'].append(s))

    elements = []
    raw_elements = result.get('elements') or []
    if not isinstance(raw_elements, (list, tuple)):
        raw_elements = _split_labels(raw_elements)
    for e in raw_elements:
        c = v.canon_element(e)
        (elements.append(c) if c else unmapped['elements'].append(str(e)))

    arc = result.get('energy_arc') or {}
    if not isinstance(arc, dict):
        arc = {}
    shape = str(arc.get('shape', '')).strip().lower()
    base = str(arc.get('base', '')).strip().lower()
    peak = str(arc.get('peak', '')).strip().lower()
    for key, val, allowed in (('shape', shape, ARC_SHAPES), ('base', base, ENERGY_LEVELS),
                              ('peak', peak, ENERGY_LEVELS)):
        if val and val not in allowed:
            unmapped['energy_arc'].append(f'{key}={val}')
    shape = shape if shape in ARC_SHAPES else ''
    base = base if base in ENERGY_LEVELS else ''
    peak = peak if peak in ENERGY_LEVELS else ''

    rhythm = result.get('rhythm') or {}
    if not isinstance(rhythm, dict):
        rhythm = {}
    pulse = str(rhythm.get('pulse', '')).strip().lower()
    character = str(rhythm.get('character', '')).strip().lower()
    for key, val, allowed in (('pulse', pulse, PULSES), ('character', character, CHARACTERS)):
        if val and val not in allowed:
            unmapped['rhythm'].append(f'{key}={val}')
    pulse = pulse if pulse in PULSES else ''
    character = character if character in CHARACTERS else ''
    bpm = None
    raw_bpm = rhythm.get('bpm')
    if raw_bpm not in (None, ''):
        try:
            bpm = float(raw_bpm)
        except (TypeError, ValueError):
            unmapped['rhythm'].append(f'bpm={raw_bpm}')
        else:
            if not BPM_RANGE[0] <= bpm <= BPM_RANGE[1]:
                unmapped['rhythm'].append(f'bpm={raw_bpm}')
                bpm = None
    # v3 results carried a tempo class instead of a BPM; keep reading them
    tempo = tempo_of(bpm) or str(rhythm.get('tempo', '')).strip().lower()
    tempo = tempo if tempo in TEMPOS else ''

    return Canonical(
        moods=_dedupe(moods)[:3],
        styles=_dedupe(styles)[:3],
        elements=_dedupe(elements),
        shape=shape, base=base, peak=peak,
        narrative=str(arc.get('trajectory', '') or '').strip(),
        tempo=tempo, pulse=pulse, bpm=bpm, character=character,
        unmapped={k: v for k, v in unmapped.items() if v},
        v3=O.canonicalize(result),
    )


def _vote(values: list[str], order: tuple = ()) -> str:
    """Majority value; ties go to the median of the tied values when the field is
    ordinal (`order` given), else to the value seen first."""
    vals = [v for v in values if v]
    if not vals:
        return ''
    counts = Counter(vals)
    best = max(counts.values())
    tied = [v for v in dict.fromkeys(vals) if counts[v] == best]
    if len(tied) == 1 or not order:
        return tied[0]
    ranked = sorted(tied, key=order.index)
    return ranked[(len(ranked) - 1) // 2]


def _frequent(lists: list[list[str]], limit: int | None) -> list[str]:
    """Labels named by at least half of the runs, most frequent first (ties keep
    first-seen order). Falls back to the single most frequent label so a field is
    never emptied by disagreement alone."""
    counts = Counter(x for xs in lists for x in dict.fromkeys(xs))
    if not counts:
        return []
    first = list(dict.fromkeys(x for xs in lists for x in xs))
    need = len(lists) / 2
    kept = sorted((x for x in first if counts[x] >= need), key=lambda x: (-counts[x], first.index(x)))
    if not kept:
        kept = sorted(first, key=lambda x: (-counts[x], first.index(x)))[:1]
    return kept[:limit] if limit else kept


def _quadrant(c: 'Canonical') -> tuple | None:
    if c.valence is None or c.arousal is None:
        return None
    return (c.valence >= 0, c.arousal >= 0)


def combine(cans: list[Canonical]) -> Canonical:
    """Merge several runs of the same track into one result.

    LLM labels vary from run to run (on a 370-track sample, one run vs another:
    valence rho ~0.80, arc shape agreeing ~70%), and averaging k runs improves that
    as for independent noise. Numbers are averaged, single-choice fields go to a
    majority vote, and list fields keep the labels at least half of the runs named.
    `agreement` (0..1) is the mean pairwise agreement on circumplex quadrant, tempo,
    pulse and arc shape: a per-track confidence.
    """
    cans = [c for c in cans if c is not None]
    if not cans:
        raise ValueError('combine() needs at least one result')
    if len(cans) == 1:
        c = cans[0]
        return replace(c, v3=O.combine([c.v3])) if c.v3 and 'runs' not in c.v3 else c
    shape = _vote([c.shape for c in cans])
    bpms = sorted(c.bpm for c in cans if c.bpm)
    bpm = bpms[len(bpms) // 2] if len(bpms) % 2 else (
        (bpms[len(bpms) // 2 - 1] + bpms[len(bpms) // 2]) / 2 if bpms else None)
    vals = [c.valence for c in cans if c.valence is not None]
    arous = [c.arousal for c in cans if c.arousal is not None]
    instr = [c.instrumental for c in cans if c.instrumental is not None]
    doms = [c.dominance for c in cans if c.dominance is not None]
    pairs = [(a, b) for i, a in enumerate(cans) for b in cans[i + 1:]]
    checks = [(_quadrant(a) == _quadrant(b), a.tempo == b.tempo, a.pulse == b.pulse,
               a.shape == b.shape) for a, b in pairs]
    agreement = round(sum(sum(c) for c in checks) / (4 * len(checks)), 3)
    unmapped: dict[str, list[str]] = {}
    for c in cans:
        for k, v in c.unmapped.items():
            unmapped.setdefault(k, [])
            unmapped[k].extend(x for x in v if x not in unmapped[k])
    return Canonical(
        moods=_frequent([c.moods for c in cans], 3),
        styles=_frequent([c.styles for c in cans], 3),
        elements=_frequent([c.elements for c in cans], None),
        shape=shape,
        base=_vote([c.base for c in cans], ENERGY_LEVELS),
        peak=_vote([c.peak for c in cans], ENERGY_LEVELS),
        narrative=next((c.narrative for c in cans if c.shape == shape and c.narrative), ''),
        unmapped=unmapped,
        tempo=tempo_of(bpm) or _vote([c.tempo for c in cans], TEMPOS),
        pulse=_vote([c.pulse for c in cans], PULSES),
        bpm=round(bpm, 1) if bpm else None,
        character=_vote([c.character for c in cans]),
        bpm_runs=tuple(c.bpm for c in cans if c.bpm),
        runs=len(cans),
        agreement=agreement,
        valence_override=_mean(vals),
        arousal_override=_mean(arous),
        instrumental_override=_mean(instr),
        dominance_override=_mean(doms),
        v3=O.combine([c.v3 for c in cans if c.v3]) if any(c.v3 for c in cans) else None,
    )


def parse_roontracktag(value: str) -> dict:
    """Turn a stored ROONTRACKTAG string back into a raw result dict.

    Used to normalise labels written by older plugin versions. Mood is not part
    of ROONTRACKTAG, so it is never recovered from here."""
    out = {'style': [], 'elements': [], 'energy_arc': {}}
    for part in (p.strip() for p in str(value or '').split(';')):
        if not part:
            continue
        key, sep, rest = part.partition(':')
        key = key.strip().lower()
        if sep and key == 'style':
            out['style'].extend(_split_labels(rest))
        elif sep and key == 'arc':
            out['energy_arc']['shape'] = rest.strip()
        elif sep and key == 'peak':
            out['energy_arc']['peak'] = rest.strip()
        else:
            out['elements'].append(part)
    return out
