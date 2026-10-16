"""Ontology v3 additions: felt / use / time-place / character facets and listened dimensions.

Facet A (expressed mood) stays in moods.yaml and goes through vocab.canonicalize like
before. Everything else v3 adds is loaded from data/ontology_v3.yaml and handled here:

  facets      controlled term lists (0..max terms each); after several runs each chosen
              term carries the share of runs that chose it.
  dimensions  single answers from a short ordinal or categorical list (LLMs are poorly
              calibrated on raw numbers); ordinal ones also map onto a number.

The result of one run is a plain dict; combine() merges runs. Stored as JSON in the
`listen_v3` field.
"""

from __future__ import annotations

import functools
import os
from collections import Counter

import yaml

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@functools.lru_cache(maxsize=None)
def load(data_dir: str = DATA_DIR) -> dict:
    with open(os.path.join(data_dir, 'ontology_v3.yaml'), encoding='utf-8') as f:
        doc = yaml.safe_load(f)
    for spec in doc['facets'].values():
        alias = {}
        for name, t in spec['terms'].items():
            alias[_key(name)] = name
            if t.get('label'):
                alias.setdefault(_key(t['label']), name)
            for a in t.get('aliases', []) or []:
                alias.setdefault(_key(a), name)
        spec['alias'] = alias
    return doc


def _key(s: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in str(s).lower()).strip('_')


# -- prompt & schema -------------------------------------------------------------
def prompt_section() -> str:
    """Instructions for the v3 keys, appended to the audio prompt."""
    o = load()
    parts = []
    for name, spec in o['facets'].items():
        terms = []
        for t, x in spec['terms'].items():
            label = x.get('label') or t.replace('_', ' ')
            # definitions only where the word alone is ambiguous (felt, character)
            terms.append(f"{t} ({x['def']})" if x.get('def') and name in ('felt', 'character') else
                         (t if label == t.replace('_', ' ') else f'{t} [{label}]'))
        parts.append(f'  "{name}": 0-{spec["max"]} terms. {spec["question"]} Choose ONLY from: '
                     + ', '.join(terms) + '. Use an empty array when none clearly applies.')
    dims = '; '.join(f'"{k}" ({v["def"]}) one of {"/".join(v["answers"])}' for k, v in o['dimensions'].items())
    parts.append('  "dims": an object judged by ear: ' + dims + '.')
    return '\n\n'.join(parts)


def schema_properties() -> dict:
    o = load()
    props = {name: {'type': 'array', 'items': {'type': 'string', 'enum': list(spec['terms'])}}
             for name, spec in o['facets'].items()}
    props['dims'] = {
        'type': 'object',
        'properties': {k: {'type': 'string', 'enum': list(v['answers'])} for k, v in o['dimensions'].items()},
        'required': list(o['dimensions']),
    }
    return props


# -- one run -----------------------------------------------------------------------
def canonicalize(result: dict) -> dict:
    """Map one provider result onto v3 terms/answers; unknown labels are dropped."""
    o, out = load(), {'facets': {}, 'dims': {}, 'unmapped': []}
    for name, spec in o['facets'].items():
        seen = []
        for label in result.get(name) or []:
            t = spec['alias'].get(_key(label))
            if t and t not in seen:
                seen.append(t)
            elif not t:
                out['unmapped'].append(f'{name}:{label}')
        out['facets'][name] = seen[:spec['max']]
    for k, spec in o['dimensions'].items():
        a = _key((result.get('dims') or {}).get(k, ''))
        if a in spec['answers']:
            out['dims'][k] = a
    return out


# -- several runs ------------------------------------------------------------------
def combine(runs: list[dict]) -> dict:
    """Merge runs: facet terms keep the share of runs that chose them (>= half are
    'current' picks, the rest stay visible with their share); dimensions take the
    majority answer with its share, and ordinal dimensions also the mean value."""
    o = load()
    runs = [r for r in runs if r]
    n = len(runs)
    out = {'runs': n, 'facets': {}, 'dims': {}}
    for name in o['facets']:
        c = Counter(t for r in runs for t in r['facets'].get(name, []))
        out['facets'][name] = {t: round(k / n, 3) for t, k in c.most_common()}
    for k, spec in o['dimensions'].items():
        answers = [r['dims'][k] for r in runs if k in r['dims']]
        if not answers:
            continue
        top, cnt = Counter(answers).most_common(1)[0]
        d = {'answer': top, 'share': round(cnt / len(answers), 3)}
        if 'to' in spec:
            nums = [spec['to'][spec['answers'].index(a)] for a in answers]
            d['value'] = round(sum(nums) / len(nums), 3)
        out['dims'][k] = d
    out['unmapped'] = sorted({u for r in runs for u in r.get('unmapped', [])})
    return out


def picks(combined: dict, facet: str) -> list[str]:
    """Terms chosen by at least half of the runs (what a single label would be)."""
    return [t for t, s in combined['facets'].get(facet, {}).items() if s >= 0.5]
