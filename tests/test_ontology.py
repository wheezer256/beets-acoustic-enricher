"""Ontology v3: facets/dimensions parsing, combining runs, prompt context, persistence."""
import json

from acoustic_enricher import ontology as O
from acoustic_enricher import schema as S
from acoustic_enricher import vocab as V


def _res(felt=(), character=(), dims=None, mood=('serene',)):
    return {'mood': list(mood), 'felt': list(felt), 'use': [], 'timeplace': [], 'character': list(character),
            'dims': dims or {}}


def test_canonicalize_maps_aliases_and_drops_unknown():
    r = O.canonicalize(_res(felt=['awe', 'Frisson', 'nonsense'], character=['tarab', 'hypnotic'],
                            dims={'tension': 'High', 'speech': 'Mostly spoken', 'groove': 'wobbly'}))
    assert r['facets']['felt'] == ['awe', 'chills']            # Frisson is an alias of chills
    assert r['facets']['character'] == ['ecstatic', 'hypnotic']  # tarab is an alias of ecstatic
    assert 'felt:nonsense' in r['unmapped']
    assert r['dims'] == {'tension': 'high', 'speech': 'mostly_spoken'}  # unknown answer dropped


def test_facet_max_is_enforced():
    r = O.canonicalize(_res(felt=['awe', 'moved', 'chills', 'entranced']))
    assert len(r['facets']['felt']) == O.load()['facets']['felt']['max']


def test_combine_shares_majorities_and_ordinal_means():
    runs = [O.canonicalize(_res(felt=f, dims={'tension': t, 'musical_stance': st}))
            for f, t, st in ((['awe'], 'high', 'sincere'), (['awe', 'moved'], 'medium', 'sincere'),
                             (['moved'], 'high', 'ironic'))]
    m = O.combine(runs)
    assert m['runs'] == 3
    assert m['facets']['felt'] == {'awe': 0.667, 'moved': 0.667}
    assert m['dims']['tension'] == {'answer': 'high', 'share': 0.667, 'value': round((.75 + .5 + .75) / 3, 3)}
    assert m['dims']['musical_stance'] == {'answer': 'sincere', 'share': 0.667}   # categorical: no value
    assert O.picks(m, 'felt') == ['awe', 'moved']


def test_prompt_context_only_with_metadata():
    assert 'Catalogue context' not in S.audio_prompt()
    p = S.audio_prompt({'artist': 'Halawa', 'title': 'Ceremonial gisalo performance', 'year': ''})
    assert 'Catalogue context' in p and 'Artist: Halawa' in p and 'Year:' not in p


def test_prompt_and_schema_cover_every_v3_term_and_dim():
    p, sch = S.audio_prompt(), S.json_schema()['properties']
    for name, spec in O.load()['facets'].items():
        assert sch[name]['items']['enum'] == list(spec['terms'])
        assert all(t in p for t in spec['terms'])
    assert set(sch['dims']['properties']) == set(O.load()['dimensions'])


def test_canonical_carries_v3_and_dominance():
    c = V.combine([V.canonicalize(_res(mood=('defiant',), felt=['empowered'])),
                   V.canonicalize(_res(mood=('defiant', 'angry'), felt=['empowered']))])
    assert c.v3['facets']['felt'] == {'empowered': 1.0}
    assert 0.6 < c.dominance <= 0.9          # defiant 0.9, angry 0.6
    assert json.loads(json.dumps(c.v3)) == c.v3
