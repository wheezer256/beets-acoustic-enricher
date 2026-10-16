import json
import shutil
import subprocess

import pytest
from beets import config
from beets.library import Item

from acoustic_enricher import schema as S
from acoustic_enricher import vocab as V
from acoustic_enricher.plugin import AcousticEnricherPlugin


@pytest.fixture
def plugin():
    config.clear()
    config.read(user=False, defaults=True)
    return AcousticEnricherPlugin()


def test_plugin_name_and_config_section(plugin):
    assert plugin.name == 'acoustic_enricher'
    assert plugin.config['provider'].get() == 'gemini'
    assert plugin.config['thinking_budget'].get(int) == 0


@pytest.mark.parametrize('duration, expected', [
    (20, [(0.0, 20)]),
    (45, [(15.0, 15)]),
    (240, [(60.0, 15.0), (120.0, 15.0), (180.0, 15.0)]),
])
def test_sample_windows(plugin, duration, expected):
    assert plugin.sample_windows(duration) == expected


def test_apply_sets_schema_fields(plugin):
    item = Item(title='t')
    can = V.canonicalize({'mood': ['serene', 'dreamy'], 'style': ['ambient'],
                          'elements': ['Synthesizer', 'Female singing'],
                          'energy_arc': {'shape': 'arch', 'base': 'low', 'peak': 'medium',
                                         'trajectory': 'gentle swell'},
                          'rhythm': {'tempo': 'slow', 'pulse': 'soft'}})
    plugin.apply(item, can, 'openai', 'gemini-flash')
    assert item['mood'] == 'serene, dreamy'
    assert item['listen_style'] == 'ambient'
    assert json.loads(item['listen_elements']) == ['Synthesizer', 'Female singing']
    assert item['listen_instrumental'] == 0.0
    assert item['listen_valence'] == can.valence and item['listen_arousal'] == can.arousal
    assert item['listen_energy_base'] == 'low' and item['listen_energy_peak'] == 'medium'
    assert item['roontracktag'].startswith('style:ambient; arc:arch; peak:medium; ')
    assert item['listen_tempo'] == 'slow' and item['listen_pulse'] == 'soft'
    assert item['listen_ver'] == S.SCHEMA_VERSION and item['listen_model'] == 'gemini-flash'


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not installed')
def test_extract_clips_produces_expected_length(tmp_path):
    src = tmp_path / 'tone.flac'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=f=440:d=120',
                    str(src)], check=True)
    out = tmp_path / 'clips.mp3'
    AcousticEnricherPlugin.extract_clips(str(src), [(10, 15), (60, 15), (100, 15)], str(out))
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                '-of', 'csv=p=0', str(out)], capture_output=True,
                               text=True, check=True).stdout)
    assert 44 <= dur <= 46


def test_listen_combines_runs(plugin, tmp_path, monkeypatch):
    from beets.library import Library
    plugin.config['runs'] = 3
    lib = Library(str(tmp_path / 'l.db'), str(tmp_path))
    item = Item(title='t', path=str(tmp_path / 'x.flac'), length=200.0)
    lib.add(item)
    answers = iter([{'mood': ['serene'], 'energy_arc': {'shape': 'arch'}},
                    RuntimeError('flaky'),
                    {'mood': ['serene', 'calm'], 'energy_arc': {'shape': 'arch'}},
                    {'mood': ['calm'], 'energy_arc': {'shape': 'valley'}}])

    class Prov:
        last_usage = {'prompt_tokens': 10}

        def analyze(self, path, meta):
            a = next(answers)
            if isinstance(a, Exception):
                raise a
            return a
    monkeypatch.setattr(AcousticEnricherPlugin, 'extract_clips', staticmethod(lambda *a, **k: None))
    plugin._analyze(item, Prov(), 'gemini', dry_run=False, db_only=True)
    got = lib.get_item(item.id)
    assert int(got['listen_runs']) == 2                    # 3 attempts, 1 failed
    assert got['mood'] == 'serene, calm' and got['listen_energy_shape'] == 'arch'
