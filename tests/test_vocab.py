import pytest

from acoustic_enricher import vocab as V


@pytest.fixture(scope='module')
def v():
    return V.load()


def test_vocab_sizes(v):
    assert len(v.moods) >= 40
    assert len(v.elements) >= 632          # names plus comma-part/alias keys
    assert len(set(v.elements.values())) == 632
    assert len(v.styles) > 2000


def test_mood_terms_and_aliases(v):
    assert v.canon_mood('Serene') == 'serene'
    assert v.canon_mood('melancholy') == 'melancholic'
    assert v.canon_mood('chill') == 'relaxed'
    assert v.canon_mood('happy') == 'joyful'
    assert v.canon_mood('flibbertigibbet') is None


def test_mood_coordinates_in_range(v):
    for t in v.moods.values():
        assert -1 <= t.valence <= 1 and -1 <= t.arousal <= 1


def test_style_canonical_forms(v):
    assert v.canon_style('Minimal Techno') == 'minimal techno'
    assert v.canon_style('minimal techno') == 'minimal techno'
    assert v.canon_style('Drum & Bass') == 'drum and bass'
    assert v.canon_style('Boogie Woogie') is not None      # Discogs-only style
    assert v.canon_style('totally invented genre xyz') is None


def test_element_exact_comma_alias_fuzzy(v):
    assert v.canon_element('synthesizer') == 'Synthesizer'
    assert v.canon_element('Male speech') == 'Male speech, man speaking'
    assert v.canon_element('Kick drum') == 'Bass drum'
    assert v.canon_element('Reverb') == 'Reverberation'
    assert v.canon_element('Keyboard') == 'Keyboard (musical)'
    assert v.canon_element('Synthesiser') == 'Synthesizer'  # fuzzy (British spelling)
    assert v.canon_element('Groove') is None                # not an AudioSet class


def test_vocal_classes(v):
    for name in ('Singing', 'Rapping', 'Choir', 'Speech', 'Male singing', 'Human voice'):
        assert name in v.vocal_elements, name
    for name in ('Laughter', 'Crying, sobbing', 'Sigh', 'Synthesizer'):
        assert name not in v.vocal_elements, name


def test_canonicalize_full_result():
    can = V.canonicalize({
        'mood': ['Serene', 'melancholy', 'nonsense'],
        'style': 'Ambient, Minimal Techno',
        'elements': ['Synthesizer', 'Kick drum', 'Groove', 'synthesizer'],
        'energy_arc': {'shape': 'Building', 'base': 'low', 'peak': 'HIGH',
                       'trajectory': 'slow build to a peak'},
    })
    assert can.moods == ['serene', 'melancholic']
    assert can.styles == ['ambient', 'minimal techno']
    assert can.elements == ['Synthesizer', 'Bass drum']     # deduped, canonical
    assert (can.shape, can.base, can.peak) == ('building', 'low', 'high')
    assert can.unmapped == {'mood': ['nonsense'], 'elements': ['Groove']}
    assert can.instrumental == 1.0


def test_canonicalize_rejects_bad_arc():
    can = V.canonicalize({'energy_arc': {'shape': 'zigzag', 'peak': 'extreme'}})
    assert can.shape == '' and can.peak == ''
    assert 'shape=zigzag' in can.unmapped['energy_arc']


def test_valence_arousal_projection(v):
    can = V.canonicalize({'mood': ['serene', 'euphoric']})
    exp_v = round((v.moods['serene'].valence + v.moods['euphoric'].valence) / 2, 3)
    exp_a = round((v.moods['serene'].arousal + v.moods['euphoric'].arousal) / 2, 3)
    assert (can.valence, can.arousal) == (exp_v, exp_a)
    assert V.canonicalize({}).valence is None


def test_instrumental_from_elements():
    assert V.canonicalize({'elements': ['Piano', 'Female singing']}).instrumental == 0.0
    assert V.canonicalize({'elements': ['Piano']}).instrumental == 1.0
    assert V.canonicalize({'elements': []}).instrumental is None


def test_roontracktag_roundtrip():
    tag = 'style:Industrial Techno, Experimental; arc:steady-high; peak:high; Music; Synthesizer'
    raw = V.parse_roontracktag(tag)
    assert raw['style'] == ['Industrial Techno', 'Experimental']
    assert raw['energy_arc'] == {'shape': 'steady-high', 'peak': 'high'}
    can = V.canonicalize(raw)
    assert can.roontracktag() == ('style:industrial techno, experimental; arc:steady-high; '
                                  'peak:high; Music; Synthesizer')


@pytest.mark.parametrize('bpm, tempo', [(70, 'slow'), (100, 'medium'), (124.5, 'fast'),
                                         (135, 'very fast'), (172, 'very fast')])
def test_tempo_bands_from_felt_bpm(bpm, tempo):
    can = V.canonicalize({'rhythm': {'bpm': bpm, 'pulse': 'DRIVING', 'character': 'Steady'}})
    assert (can.tempo, can.bpm, can.pulse, can.character) == (tempo, float(bpm), 'driving', 'steady')


def test_canonicalize_rhythm_rejects_bad_values():
    bad = V.canonicalize({'rhythm': {'bpm': 900, 'pulse': 'thumping', 'character': 'wobbly'}})
    assert (bad.bpm, bad.tempo, bad.pulse, bad.character) == (None, '', '', '')
    assert set(bad.unmapped['rhythm']) == {'bpm=900', 'pulse=thumping', 'character=wobbly'}
    assert V.canonicalize({}).tempo == ''


def test_v3_tempo_class_still_read():
    assert V.canonicalize({'rhythm': {'tempo': 'Fast', 'pulse': 'soft'}}).tempo == 'fast'


def test_combine_takes_median_felt_bpm():
    runs = [V.canonicalize({'mood': ['calm'], 'rhythm': {'bpm': b, 'character': c}})
            for b, c in ((70, 'steady'), (140, 'steady'), (72, 'fragmented'))]
    m = V.combine(runs)
    assert m.bpm == 72 and m.tempo == 'slow' and m.character == 'steady'


def _run(moods, shape='arch', tempo='slow', pulse='soft', peak='medium', elements=('Piano',)):
    return V.canonicalize({'mood': list(moods), 'style': ['ambient'], 'elements': list(elements),
                           'energy_arc': {'shape': shape, 'base': 'low', 'peak': peak,
                                          'trajectory': shape},
                           'rhythm': {'tempo': tempo, 'pulse': pulse}})


def test_combine_single_run_keeps_labels_and_merges_v3():
    c = _run(['serene'])
    m = V.combine([c])
    assert m.moods == c.moods and m.runs == 1 and m.agreement is None
    assert m.v3['runs'] == 1 and set(m.v3['facets']) == {'felt', 'use', 'timeplace', 'character'}


def test_combine_votes_and_averages():
    a = _run(['serene', 'calm'], shape='arch', tempo='slow')
    b = _run(['serene', 'dreamy'], shape='arch', tempo='medium')
    c = _run(['calm', 'serene'], shape='valley', tempo='slow', elements=('Piano', 'Female singing'))
    m = V.combine([a, b, c])
    assert m.runs == 3
    assert m.moods == ['serene', 'calm']            # dreamy named by 1 of 3 runs -> dropped
    assert m.shape == 'arch' and m.narrative == 'arch' and m.tempo == 'slow'
    assert m.valence == round((a.valence + b.valence + c.valence) / 3, 3)
    assert m.elements == ['Piano']                  # Female singing in 1 of 3 runs
    assert abs(m.instrumental - 2 / 3) < 0.01       # 2 of 3 runs heard no vocals
    assert 0 < m.agreement < 1


def test_combine_ordinal_tie_takes_median():
    m = V.combine([_run(['calm'], tempo='slow'), _run(['calm'], tempo='fast')])
    assert m.tempo in ('slow', 'fast')               # two-way tie: lower median of the tied pair
    m4 = V.combine([_run(['calm'], tempo=t) for t in ('slow', 'slow', 'fast', 'fast')])
    assert m4.tempo == 'slow'


def test_combine_never_empties_moods():
    m = V.combine([_run(['serene']), _run(['tense']), _run(['joyful'])])
    assert m.moods == ['serene']                     # no majority -> most frequent, first seen
