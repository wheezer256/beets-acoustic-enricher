import numpy as np
import pytest

from acoustic_enricher import analysis as A

FPS = A.SR / A.HOP


def clicks(bpm, seconds=30, accent_every=0, swing=0.0):
    """Synthetic onset envelope: an impulse per beat (optionally swung/accented)."""
    env = np.zeros(int(seconds * FPS))
    period = 60.0 / bpm
    t, k = 0.0, 0
    while t < seconds - 0.1:
        off = swing * period if k % 2 else 0.0
        i = int((t + off) * FPS)
        if i < len(env):
            env[i] = 2.0 if accent_every and k % accent_every == 0 else 1.0
        t += period
        k += 1
    return np.convolve(env, np.hanning(5), mode='same')


@pytest.mark.parametrize('bpm', [70, 92, 124.5, 128, 140, 170])
def test_measure_bpm_on_clicks(bpm):
    got, strength = A.measure_bpm(clicks(bpm), FPS)
    ratios = [got / bpm * f for f in (0.5, 1, 2)]
    assert min(abs(r - 1) for r in ratios) < 0.03, got     # right up to an octave
    assert strength > 0.5


def test_measure_bpm_no_beat_is_weak():
    rng = np.random.default_rng(1)
    _, strength = A.measure_bpm(rng.random(int(30 * FPS)), FPS)
    assert strength < 0.3


@pytest.mark.parametrize('felt, measured, strength, expected', [
    (125, 124.5, 0.7, 124.5),     # agree: use the precise measured value
    (125, 83.4, 0.7, 125.1),      # measured locked on a 3:2 level -> x1.5
    (72, 142.4, 0.7, 71.2),       # half-time feel -> x0.5
    (170, 85.2, 0.7, 170.4),      # measured at half -> x2
    (110, 142.4, 0.7, 110),       # no candidate within 10% (94.9, 142.4): keep felt
    (125, 124.5, 0.1, 125),       # weak beat measurement: keep the felt BPM
    (None, 124.5, 0.7, None),     # no felt BPM: nothing to anchor the octave
])
def test_reconcile_bpm(felt, measured, strength, expected):
    assert A.reconcile_bpm(felt, measured, strength) == expected


def test_choose_windows_finds_late_beat():
    # 240 s track: quiet and beatless until 150 s, then a loud section with a beat
    pul = np.r_[np.full(150, 0.1), np.full(90, 0.8)]
    loud = np.r_[np.full(150, -40.0), np.full(90, -12.0)]
    segs = [(0, 150, 0.1, -40.0), (150, 240, 0.8, -12.0)]
    wins = A.choose_windows(loud, pul, segs, 240.0)
    assert len(wins) == 3 and all(d == 15.0 for _, d in wins)
    assert any(s >= 150 for s, _ in wins)            # the beat section is heard
    assert any(s < 150 for s, _ in wins)             # and so is the quiet part
    assert all(5 <= s <= 240 - 15 - 5 for s, _ in wins)


def test_descriptors_shapes_and_ranges():
    rng = np.random.default_rng(0)
    secs = 120
    loud = np.r_[np.full(30, -45.0), np.full(80, -15.0), np.full(10, -90.0)]   # quiet intro, body, silent tail
    pul = np.r_[np.full(30, 0.1), np.full(80, 0.7), np.full(10, 0.0)]
    env = clicks(170, secs)                                                     # ~2.8 onsets/s
    mag = np.abs(rng.normal(size=(len(env), A.NFFT // 2 + 1))).astype(np.float32)
    d = A.descriptors(mag, env, loud, pul)
    assert 25 <= d['intro_s'] <= 32
    assert d['tail_silence_s'] == 10
    assert 1.5 < d['onset_rate'] < 4.5
    assert 0.0 <= d['harsh'] <= 1.0 and d['beat_strength'] > 0.5
