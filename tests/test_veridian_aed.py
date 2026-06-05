"""Tests for the AED runner placeholder (sensveridian.runners.aed)."""
from __future__ import annotations

import numpy as np

from sensveridian.runners.aed import AEDRunner


def _silence_then_tone(sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    silence = np.zeros(sr // 2, dtype=np.float32)
    return np.concatenate([silence, tone])


def test_aed_detects_active_region():
    runner = AEDRunner(sample_rate=16000)
    samples = _silence_then_tone()
    out = runner.predict(samples, sample_rate=16000)
    assert out.summary.present is True
    assert out.summary.count >= 1
    assert out.raw["duration_s"] == 1.0
    # active span sits in the second half (the tone)
    seg = max(out.raw["segments"], key=lambda s: s["end_frac"])
    assert seg["label"] == "speech"
    assert seg["end_frac"] > 0.5
    assert 0.0 <= seg["start_frac"] < seg["end_frac"] <= 1.0


def test_aed_pure_silence_has_no_segments():
    runner = AEDRunner(sample_rate=16000)
    out = runner.predict(np.zeros(16000, dtype=np.float32), sample_rate=16000)
    assert out.summary.count == 0
    assert out.summary.present is False


def test_waveform_peaks_downsamples():
    runner = AEDRunner()
    peaks = runner.waveform_peaks(_silence_then_tone(), n=220)
    assert len(peaks) == 220
    assert all(p >= 0 for p in peaks)
    # second half (tone) is louder than first (silence)
    assert max(peaks[110:]) > max(peaks[:110])


def test_aed_handles_integer_pcm():
    runner = AEDRunner(sample_rate=16000)
    pcm = (_silence_then_tone() * 32767).astype(np.int16)
    out = runner.predict(pcm, sample_rate=16000)
    assert out.summary.count >= 1  # normalized from int16 range
