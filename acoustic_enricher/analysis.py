"""Full-track analysis: where to take the clips, and the measured tempo.

The LLM only hears the clips, and its labels follow what it hears: on a 370-track
sample, moving the clips changed labels more than re-running the model did. Fixed
positions (25/50/75 %) missed a clearly different section (>= 20 s louder/quieter or
with a stronger/weaker beat) in 27 % of tracks; choosing positions from this analysis
cut that to ~5 % for the same total clip length, so the LLM cost is unchanged.

Needs numpy and ffmpeg only (no librosa/numba). The track is decoded once, at a low
sample rate, then summarised per second:

  loudness       RMS level in dB
  pulse          how periodic the onsets are in a 6 s window (0 = no beat, ~1 = metronomic)
  timbre         spectral centroid and three band energies

and four whole-track descriptors the owner's ratings kept asking for (the LLM, which
only hears 45 s, cannot judge them reliably):

  intro_s        seconds before the track reaches its typical loudness (and, if it has
                 a beat, its typical beat strength): long noisy/ambient intros
  onset_rate     note/sound onsets per second in the body of the track: "fast even
                 without a beat" (racing guitar, kora)
  harsh          share of energy above 5 kHz in the louder passages (90th percentile):
                 grating, high-frequency-heavy sound
  tail_silence_s seconds of near-silence at the end
  beat_strength  typical pulse clarity of the body (0 = no beat, ~1 = metronomic)

Sections are found where those features change, and the windows aim at the most
rhythmic stretch, the loudest stretch, and any clearly different section still not
covered (else the most typical stretch).
"""

from __future__ import annotations

import subprocess

import numpy as np

# 22.05 kHz keeps the treble up to ~11 kHz for the harshness measure; HOP/NFFT keep
# ~23 ms frames (fps ~43) as before.
SR = 22050
HOP = 512
NFFT = 2048
HARSH_HZ = 5000


def decode(path: str) -> np.ndarray:
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(SR),
                          '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _frames(y: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (len(y) - NFFT) // HOP)
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    return y[idx] * np.hanning(NFFT)[None, :]


def spectrum(y: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram, frames x bins (computed in chunks to bound memory)."""
    out = []
    step = 2048
    n = 1 + max(0, (len(y) - NFFT) // HOP)
    for s in range(0, n, step):
        seg = y[s * HOP: (min(n, s + step) - 1) * HOP + NFFT]
        out.append(np.abs(np.fft.rfft(_frames(seg), axis=1)).astype(np.float32))
    return np.vstack(out) if out else np.zeros((0, NFFT // 2 + 1), np.float32)


def onset_envelope(mag: np.ndarray) -> np.ndarray:
    """Spectral flux on log magnitude: positive change summed over frequency."""
    lm = np.log1p(10 * mag)
    flux = np.maximum(0.0, np.diff(lm, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    return flux - np.convolve(flux, np.ones(16) / 16, mode='same')   # remove slow trend


def _autocorr(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / ac[0] if ac[0] > 0 else np.zeros(n)


def pulse_clarity(env: np.ndarray, fps: float) -> float:
    ac = _autocorr(env)
    lo, hi = int(0.25 * fps), min(len(ac) - 1, int(2.0 * fps))
    return float(ac[lo:hi].max()) if hi > lo else 0.0


def measure_bpm(env: np.ndarray, fps: float) -> tuple[float | None, float]:
    """Tempo (60-200 BPM) from the onset autocorrelation.

    A beat period also produces peaks at 2x, 3x and 4x its lag, while a wrong
    metrical level (a triplet or dotted subdivision) lines up with fewer of them, so
    each candidate lag is scored by a comb over its multiples, times a broad
    log-normal tempo prior centred on 120 BPM. The winning lag is refined with
    parabolic interpolation. Octave (2x) ambiguity remains by design: the felt tempo
    decides that (see reconcile_bpm)."""
    ac = _autocorr(env)
    lo, hi = int(fps * 60 / 200), int(fps * 60 / 60)
    if 4 * hi + 2 >= len(ac):
        return None, 0.0
    lags = np.arange(lo, hi + 1)
    comb = sum(w * ac[m * lags] for m, w in ((1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25)))
    bpms = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)
    i = int(lags[int(np.argmax(comb * prior))])
    a, b, c = ac[i - 1], ac[i], ac[i + 1]
    den = a - 2 * b + c
    shift = 0.5 * (a - c) / den if den else 0.0
    lag = (i + shift) / fps
    return (round(60.0 / lag, 1) if lag > 0 else None), float(b)


def per_second(y: np.ndarray, mag: np.ndarray, env: np.ndarray):
    fps = SR / HOP
    sec = int(round(fps))
    n = min(len(env) // sec, len(y) // SR)
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    bands = [(0, 250), (250, 2000), (2000, SR / 2)]
    loud, pul, tim = [], [], []
    for s in range(n):
        seg = y[s * SR:(s + 1) * SR]
        loud.append(10 * np.log10(float(np.mean(seg ** 2)) + 1e-10))
        pul.append(pulse_clarity(env[max(0, (s - 3) * sec):(s + 3) * sec], fps))
        m = mag[s * sec:(s + 1) * sec]
        tot = m.sum() + 1e-9
        cen = float((m.sum(axis=0) * freqs).sum() / tot)
        tim.append([cen] + [float(m[:, (freqs >= a) & (freqs < b)].sum() / tot) for a, b in bands])
    return np.array(loud), np.array(pul), np.array(tim)


def sections(loud, pul, tim, min_len=20, max_k=8):
    """Change points where the per-second features differ most between the 10 s
    before and after; returns [(start, end, mean_pulse, mean_loud)]."""
    n = len(loud)
    feats = np.column_stack([(loud - loud.mean()) / (loud.std() + 1e-6),
                             (pul - pul.mean()) / (pul.std() + 1e-6),
                             (tim - tim.mean(0)) / (tim.std(0) + 1e-6)])
    w = 10
    nov = np.zeros(n)
    for t in range(w, n - w):
        nov[t] = np.linalg.norm(feats[t - w:t].mean(0) - feats[t:t + w].mean(0))
    cuts = []
    for t in np.argsort(nov)[::-1]:
        if nov[t] <= 0 or len(cuts) >= max_k - 1:
            break
        if all(abs(t - c) >= min_len for c in cuts) and min_len <= t <= n - min_len:
            cuts.append(int(t))
    edges = [0] + sorted(cuts) + [n]
    return [(a, b, float(pul[a:b].mean()), float(loud[a:b].mean())) for a, b in zip(edges, edges[1:]) if b > a]


def choose_windows(loud, pul, segs, dur, clip=15.0, k=3):
    n = len(pul)
    lo, hi = 5, max(5, int(dur - clip - 5))
    med_p, med_l = float(np.median(pul)), float(np.median(loud))
    distinct = [s for s in segs if s[1] - s[0] >= 20 and (abs(s[2] - med_p) > 0.2 or abs(s[3] - med_l) > 6)]

    def overlaps(a, b):
        return abs(a - b) < clip

    starts: list[float] = []
    for arr in (pul, loud):   # best 15 s stretch by mean, not a one-second spike
        span = range(lo, max(lo + 1, min(n - int(clip), hi) + 1))
        means = [float(np.mean(arr[i:i + int(clip)])) for i in span]
        if means:
            st = float(span[int(np.argmax(means))])
            if not any(overlaps(st, x) for x in starts):
                starts.append(st)
    for s in sorted(distinct, key=lambda s: s[1] - s[0], reverse=True):
        if len(starts) >= k:
            break
        if any(min(s[1], x + clip) - max(s[0], x) >= 5 for x in starts):
            continue
        st = float(min(max(lo, (s[0] + s[1]) / 2 - clip / 2), hi))
        if not any(overlaps(st, x) for x in starts):
            starts.append(st)
    z = np.abs((pul - np.median(pul)) / (pul.std() + 1e-6)) + np.abs((loud - np.median(loud)) / (loud.std() + 1e-6))
    for t in np.argsort(z):
        if len(starts) >= k:
            break
        st = float(min(max(lo, t - clip / 2), hi))
        if not any(overlaps(st, x) for x in starts):
            starts.append(st)
    return [(round(x, 1), float(clip)) for x in sorted(starts)[:k]]


def descriptors(mag: np.ndarray, env: np.ndarray, loud: np.ndarray, pul: np.ndarray) -> dict:
    """Whole-track descriptors (see module docstring)."""
    fps = SR / HOP
    n = len(loud)
    # trailing near-silence: below -50 dBFS or 30 dB under the track's median level
    quiet = max(-50.0, float(np.median(loud)) - 30)
    tail = 0
    for v in loud[::-1]:
        if v < quiet:
            tail += 1
        else:
            break
    body_end = max(1, n - tail)
    # intro: first second from which a 5 s stretch is at typical loudness (and beat)
    target = float(np.median(loud[:body_end])) - 6
    has_beat = float(np.percentile(pul[:body_end], 75)) >= 0.45
    beat_target = float(np.percentile(pul[:body_end], 75)) - 0.15
    intro = 0
    for t in range(0, max(1, body_end - 5)):
        if loud[t:t + 5].mean() >= target and (not has_beat or pul[t:t + 5].mean() >= beat_target):
            intro = t
            break
    # onset rate in the body: local maxima of the onset envelope above a robust threshold
    sec = int(round(fps))
    e = env[intro * sec: body_end * sec]
    rate = 0.0
    if len(e) > 3 * sec:
        thr = np.median(e) + 1.5 * np.median(np.abs(e - np.median(e)))
        peaks = (e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:]) & (e[1:-1] > max(thr, 1e-9))
        # at most one onset per 50 ms (avoid double-counting a single attack)
        idx = np.flatnonzero(peaks)
        kept, last = 0, -10 ** 9
        for i in idx:
            if i - last >= int(0.05 * fps):
                kept += 1
                last = i
        rate = kept / (len(e) / fps)
    # harshness: share of spectral energy above HARSH_HZ, loud frames, 90th percentile
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    power = mag.astype(np.float64) ** 2
    tot = power.sum(axis=1) + 1e-12
    hi = power[:, freqs >= HARSH_HZ].sum(axis=1) / tot
    frame_db = 10 * np.log10(tot / power.shape[1])
    loud_frames = frame_db >= np.percentile(frame_db, 50)
    harsh = float(np.percentile(hi[loud_frames], 90)) if loud_frames.any() else 0.0
    # beat strength: typical (75th percentile) pulse clarity of the body, 0 = no beat
    beat = float(np.percentile(pul[intro:body_end], 75)) if body_end > intro else 0.0
    return {'intro_s': int(intro), 'onset_rate': round(rate, 2), 'harsh': round(harsh, 4),
            'tail_silence_s': int(tail), 'beat_strength': round(beat, 3)}


def analyse(path: str, clip: float = 15.0, k: int = 3) -> dict:
    """Decode once; return {'windows', 'bpm', 'bpm_strength', 'duration', and the
    descriptors() fields}."""
    y = decode(path)
    dur = len(y) / SR
    mag = spectrum(y)
    env = onset_envelope(mag)
    fps = SR / HOP
    loud, pul, tim = per_second(y, mag, env)
    if len(loud) < 60:
        return {'duration': dur, 'windows': None, 'bpm': None, 'bpm_strength': 0.0}
    segs = sections(loud, pul, tim)
    # tempo from the most rhythmic 30 s, where the beat is clearest
    sec = int(round(fps))
    best = int(np.argmax([pul[i:i + 30].mean() for i in range(0, max(1, len(pul) - 30))]))
    bpm, strength = measure_bpm(env[best * sec:(best + 30) * sec], fps)
    return {'duration': dur, 'windows': choose_windows(loud, pul, segs, dur, clip, k),
            'bpm': bpm, 'bpm_strength': round(strength, 3), **descriptors(mag, env, loud, pul)}


def reconcile_bpm(felt: float | None, measured: float | None, strength: float,
                  min_strength: float = 0.3, tolerance: float = 0.10) -> float | None:
    """Measured BPM at the felt metrical level.

    The meter is precise but can lock onto a different level of the beat (x2, x1/2,
    or a 3:2 dotted/triplet level); the LLM knows the felt pace but not the exact
    number. Try the measured value at x1/2, x2/3, x1, x3/2 and x2 and use the one
    within `tolerance` of the felt BPM; otherwise keep the felt BPM."""
    if not felt:
        return None
    if not measured or strength < min_strength:
        return felt
    cand = min((measured * f for f in (0.5, 2 / 3, 1.0, 1.5, 2.0)), key=lambda c: abs(np.log(c / felt)))
    return round(cand, 1) if abs(cand / felt - 1) <= tolerance else felt
