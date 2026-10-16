import json

import pytest
from beets import config, ui
from beets.library import Item, Library

from acoustic_enricher import schema as S
from acoustic_enricher.batch import BatchRunner
from acoustic_enricher.plugin import AcousticEnricherPlugin


@pytest.fixture
def plugin(tmp_path):
    config.clear()
    config.read(user=False, defaults=True)
    p = AcousticEnricherPlugin()
    p.config['batch_dir'] = str(tmp_path / 'batches')
    return p


@pytest.fixture
def lib(tmp_path):
    lib = Library(str(tmp_path / 'lib.db'), str(tmp_path / 'music'))
    for i, title in enumerate(['A', 'B', 'C'], start=1):
        lib.add(Item(title=title, path=f'/nonexistent/{i}.flac', length=240.0))
    return lib


def test_request_line_shape(plugin, tmp_path):
    clip = tmp_path / 'c.mp3'
    clip.write_bytes(b'abc')
    line = json.loads(BatchRunner(plugin).request_line(7, str(clip)))
    assert line['key'] == 'item-7'
    parts = line['request']['contents'][0]['parts']
    # instructions first (shared, cacheable prefix), then the audio; no context without metadata
    assert 'Return a JSON object' in parts[0]['text'] and 'Catalogue context' not in parts[0]['text']
    assert parts[1]['inlineData']['mimeType'] == 'audio/mp3' and len(parts) == 2
    with_meta = json.loads(BatchRunner(plugin).request_line(7, str(clip), 0, {'artist': 'Laibach'}))
    p2 = with_meta['request']['contents'][0]['parts']
    assert p2[0]['text'] == parts[0]['text'] and p2[2]['text'].endswith('Artist: Laibach')
    gen = line['request']['generationConfig']
    assert gen['responseMimeType'] == 'application/json'
    assert gen['thinkingConfig'] == {'thinkingLevel': 'MINIMAL'}  # default model is Gemini 3.x
    assert gen['responseSchema']['properties']['mood']['items']['enum']


def test_estimate_matches_measured_cost(plugin):
    r = BatchRunner(plugin)
    # Ontology v3 on 3.5 Flash-Lite, measured 2026-10-13: per run ~1,440 audio + ~2,430 text
    # input and ~445 output tokens -> ~$0.0013 per track per run at batch price.
    est = r.estimate_usd('gemini-3.5-flash-lite', 45 * 1000, 1000)
    assert 1.1 < est < 1.5
    assert r.estimate_usd('gemini-2.5-flash-lite', 45 * 1000, 1000) < est


def test_cached_prompt_tokens_bill_at_factor(plugin):
    r = BatchRunner(plugin)
    u = {'prompt': 3870, 'audio': 1440, 'output': 445}
    full = r.actual_usd('gemini-3.5-flash-lite', u)
    cached = r.actual_usd('gemini-3.5-flash-lite', {**u, 'cached': 2430})
    saved = (2430 * 0.75 * 0.30) / 1e6 * 0.5          # 75 % of the text price, batch-discounted
    assert abs((full - cached) - saved) < 1e-9


def test_submit_refuses_over_cap(plugin, lib):
    with pytest.raises(ui.UserError, match='exceeds the cap'):
        BatchRunner(plugin).submit(lib, [], chunk=10, max_usd=0.0001, dry_run=False, force=False)


def test_submit_dry_run_does_not_upload(plugin, lib, monkeypatch):
    r = BatchRunner(plugin)
    monkeypatch.setattr(r, 'client', lambda: pytest.fail('dry run must not call the API'))
    r.submit(lib, [], chunk=10, max_usd=100, dry_run=True, force=False)
    assert r._jobs() == []


def test_ingest_results_file(plugin, lib, tmp_path):
    r = BatchRunner(plugin)
    good = {'mood': ['serene'], 'style': ['ambient'], 'elements': ['Synthesizer'],
            'energy_arc': {'shape': 'arch', 'base': 'low', 'peak': 'medium', 'trajectory': 'x'}}
    usage = {'promptTokenCount': 1940, 'candidatesTokenCount': 200,
             'promptTokensDetails': [{'modality': 'AUDIO', 'tokenCount': 1440},
                                     {'modality': 'TEXT', 'tokenCount': 500}]}
    lines = [
        {'key': 'item-1', 'response': {'candidates': [{'content': {'parts': [
            {'text': json.dumps(good)}]}}], 'usageMetadata': usage}},
        {'key': 'item-2', 'error': {'code': 13, 'message': 'internal'}},
        {'key': 'item-3', 'response': {'candidates': [{'content': {'parts': [
            {'text': json.dumps({'mood': ['nonsense']})}]}}], 'usageMetadata': usage}},
    ]
    res = tmp_path / 'results.jsonl'
    res.write_text('\n'.join(json.dumps(x) for x in lines))
    job = {'name': 'batches/x', 'model': 'gemini-2.5-flash', 'requests': str(tmp_path / 'r.jsonl'),
           'results': str(res), 'item_ids': [1, 2, 3], 'est_usd': 0.003}
    r._ingest(lib, job, db_only=True)
    assert job['ok'] == 1 and len(job['failed']) == 2
    it = lib.get_item(1)
    assert it['mood'] == 'serene' and it['listen_ver'] == S.SCHEMA_VERSION
    assert it['listen_provider'] == 'gemini-batch'
    assert lib.get_item(3).get('listen_ver') is None          # invalid result not stored
    assert job['usage']['audio'] == 2880 and job['actual_usd'] > 0


def test_request_lines_per_run(plugin, tmp_path):
    clip = tmp_path / 'c.mp3'
    clip.write_bytes(b'abc')
    r = BatchRunner(plugin)
    assert json.loads(r.request_line(7, str(clip)))['key'] == 'item-7'
    assert json.loads(r.request_line(7, str(clip), 2))['key'] == 'item-7-r2'


def test_ingest_combines_runs(plugin, lib, tmp_path):
    plugin.config['runs'] = 3
    r = BatchRunner(plugin)

    def res(key, mood, tempo):
        body = {'mood': [mood], 'style': ['ambient'], 'elements': ['Synthesizer'],
                'energy_arc': {'shape': 'arch', 'base': 'low', 'peak': 'medium', 'trajectory': 'x'},
                'rhythm': {'tempo': tempo, 'pulse': 'soft'}}
        return {'key': key, 'response': {'candidates': [{'content': {'parts': [
            {'text': json.dumps(body)}]}}], 'usageMetadata': {'promptTokenCount': 10}}}
    lines = [res('item-1', 'serene', 'slow'), res('item-1-r1', 'serene', 'slow'),
             res('item-1-r2', 'calm', 'medium'),
             {'key': 'item-2', 'error': {'code': 13}}, res('item-2-r1', 'tense', 'fast'),
             {'key': 'item-3', 'error': {'code': 13}}]
    p = tmp_path / 'results.jsonl'
    p.write_text('\n'.join(json.dumps(x) for x in lines))
    job = {'name': 'batches/y', 'model': 'gemini-3.5-flash-lite', 'requests': str(tmp_path / 'r.jsonl'),
           'results': str(p), 'item_ids': [1, 2, 3], 'est_usd': 0.01}
    r._ingest(lib, job, db_only=True)
    one = lib.get_item(1)
    assert one['mood'] == 'serene' and one['listen_tempo'] == 'slow'
    assert int(one['listen_runs']) == 3 and 0 < float(one['listen_agreement']) < 1
    assert int(lib.get_item(2)['listen_runs']) == 1        # one run failed, one usable
    assert job['ok'] == 2 and [f['id'] for f in job['failed']] == [3]
