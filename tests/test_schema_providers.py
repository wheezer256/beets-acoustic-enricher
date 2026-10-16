import json

import confuse
import pytest

from acoustic_enricher import providers as P
from acoustic_enricher import schema as S
from acoustic_enricher import vocab as V


def test_schema_enums_match_vocab():
    sch = S.json_schema()
    assert sch['properties']['mood']['items']['enum'] == V.load().mood_terms()
    arc = sch['properties']['energy_arc']['properties']
    assert arc['shape']['enum'] == list(V.ARC_SHAPES)
    assert arc['peak']['enum'] == list(V.ENERGY_LEVELS)
    assert set(sch['required']) == {'mood', 'style', 'elements', 'energy_arc', 'rhythm',
                                    'felt', 'use', 'timeplace', 'character', 'dims'}
    rh = sch['properties']['rhythm']['properties']
    assert rh['bpm']['type'] == 'number'
    assert rh['pulse']['enum'] == list(V.PULSES)
    assert rh['character']['enum'] == list(V.CHARACTERS)


def test_prompt_lists_every_mood_term():
    prompt = S.audio_prompt()
    for term in V.load().mood_terms():
        assert term in prompt


def test_parse_json_strips_fences():
    assert P.parse_json('```json\n{"a": 1}\n```') == {'a': 1}
    assert P.parse_json(' {"a": 2} ') == {'a': 2}


@pytest.mark.parametrize('exc, expected', [
    (Exception('Resource_Exhausted: quota exceeded'), True),
    (Exception('Budget has been exceeded for key'), True),
    (Exception('connection reset'), False),
])
def test_quota_classification(exc, expected):
    assert P.is_quota_error(exc) is expected


def _cfg(**over):
    c = confuse.Configuration('acoustic_enricher_test', read=False)
    c.set({'openai_url': 'http://proxy/v1/chat/completions', 'model': 'gemini-flash',
           'timeout': 30, 'openai_api_key': 'k', 'openai_api_key_file': '',
           'reasoning_effort': 'disable', 'structured_output': True, **over})
    return c


def test_openai_payload_has_audio_schema_and_reasoning(tmp_path):
    clip = tmp_path / 'c.mp3'
    clip.write_bytes(b'ID3fake')
    prov = P.OpenAICompatProvider(_cfg(), log=None)
    p = prov._payload(str(clip))
    parts = p['messages'][0]['content']
    assert parts[1]['type'] == 'input_audio' and parts[1]['input_audio']['format'] == 'mp3'
    assert p['response_format']['type'] == 'json_schema'
    assert p['response_format']['json_schema']['strict'] is True
    assert p['reasoning_effort'] == 'disable'


def test_openai_payload_json_object_fallback(tmp_path):
    clip = tmp_path / 'c.mp3'
    clip.write_bytes(b'x')
    prov = P.OpenAICompatProvider(_cfg(structured_output=False, reasoning_effort=''), log=None)
    p = prov._payload(str(clip))
    assert p['response_format'] == {'type': 'json_object'}
    assert 'reasoning_effort' not in p


def test_openai_analyze_records_usage(tmp_path, monkeypatch):
    clip = tmp_path / 'c.mp3'
    clip.write_bytes(b'x')
    body = {'choices': [{'message': {'content': json.dumps({'mood': ['calm']})}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5,
                      'prompt_tokens_details': {'audio_tokens': 8}}}

    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return body

    import requests
    monkeypatch.setattr(requests, 'post', lambda *a, **k: R())
    prov = P.OpenAICompatProvider(_cfg(), log=None)
    assert prov.analyze(str(clip), {}) == {'mood': ['calm']}
    assert prov.last_usage['audio_tokens'] == 8 and prov.last_usage['output_tokens'] == 5


def test_openai_budget_error_in_body_is_quota(tmp_path, monkeypatch):
    clip = tmp_path / 'c.mp3'
    clip.write_bytes(b'x')

    class R:
        status_code = 400
        text = '{"error": {"message": "ExceededBudget: Budget has been exceeded! Current cost: 0.002"}}'

    import requests
    monkeypatch.setattr(requests, 'post', lambda *a, **k: R())
    with pytest.raises(P.QuotaError):
        P.OpenAICompatProvider(_cfg(), log=None).analyze(str(clip), {})


def test_openai_other_http_error_is_not_quota(tmp_path, monkeypatch):
    clip = tmp_path / 'c.mp3'
    clip.write_bytes(b'x')

    class R:
        status_code = 500
        text = 'upstream error'

    import requests
    monkeypatch.setattr(requests, 'post', lambda *a, **k: R())
    with pytest.raises(RuntimeError, match='HTTP 500'):
        P.OpenAICompatProvider(_cfg(), log=None).analyze(str(clip), {})


@pytest.mark.parametrize('model, budget, level, expected', [
    ('gemini-2.5-flash', 0, 'minimal', {'thinkingBudget': 0}),
    ('models/gemini-2.5-flash', -1, 'minimal', None),
    ('gemini-3.5-flash-lite', 0, 'minimal', {'thinkingLevel': 'MINIMAL'}),
    ('gemini-3.5-flash-lite', 0, '', None),
])
def test_gemini_thinking_by_generation(model, budget, level, expected):
    assert P.gemini_thinking(model, budget, level) == expected


def test_temperature_passed_when_set(tmp_path):
    clip = tmp_path / 'c.mp3'
    clip.write_bytes(b'x')
    assert 'temperature' not in P.OpenAICompatProvider(_cfg(), log=None)._payload(str(clip))
    p = P.OpenAICompatProvider(_cfg(temperature=0), log=None)._payload(str(clip))
    assert p['temperature'] == 0.0
    assert P.temperature(_cfg(temperature='')) is None
