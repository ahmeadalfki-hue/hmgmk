"""
tests/test_physio.py

Unit tests for physio engine utilities:
- bandpass_filter: ensure an embedded sinusoid near expected freq is preserved.
- eye_aspect_ratio: open vs closed eye behavior.
- compute_verdict_and_score: basic rule-based scoring behavior.
"""
import numpy as np
import math

from app.core.physio_engine import bandpass_filter, eye_aspect_ratio
from app.main import compute_verdict_and_score

def test_bandpass_filter_recovers_frequency():
    np.random.seed(0)
    fs = 30.0
    duration = 10.0
    t = np.arange(0, duration, 1.0/fs)
    freq = 1.5  # Hz
    sig = 1.0 * np.sin(2.0 * math.pi * freq * t) + 0.3 * np.random.randn(len(t))
    filtered = bandpass_filter(sig, fs=fs, low=0.7, high=4.0, order=4)
    freqs = np.fft.rfftfreq(len(filtered), d=1.0/fs)
    fft = np.abs(np.fft.rfft(filtered))
    peak_freq = freqs[np.argmax(fft)]
    assert abs(peak_freq - freq) < 0.2

def test_eye_aspect_ratio_open_vs_closed():
    open_coords = [(0,0), (1,-4), (2,-4), (3,0), (2,4), (1,4)]
    closed_coords = [(0,0), (1,-0.3), (2,-0.3), (3,0), (2,0.3), (1,0.3)]
    ear_open = eye_aspect_ratio(open_coords)
    ear_closed = eye_aspect_ratio(closed_coords)
    assert ear_open > ear_closed
    assert ear_closed < 0.25

def test_compute_verdict_and_score_cases():
    verdict1, score1, conf1 = compute_verdict_and_score(70.0, 8.0, 3)
    assert verdict1 == "likely_real"
    verdict2, score2, conf2 = compute_verdict_and_score(None, None, 0)
    assert verdict2 == "likely_fake"
    verdict3, score3, conf3 = compute_verdict_and_score(130.0, 4.0, 1)
    assert verdict3 == "inconclusive"
