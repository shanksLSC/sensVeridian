"""AcousticEventDetection (AED) runner — the audio modality.

Adds the ``aed`` model the Veridian Studio audio screens expect. It follows the
runner contract (``model_id``/``display_name``/``version``/``load``) but operates
on an audio waveform rather than a BGR image, so it is driven by the audio
ingest path, not the image :class:`~sensveridian.orchestrator.Orchestrator`.

No acoustic model weights ship with sensVeridian yet, so ``predict`` uses a
**deterministic energy-based segmenter** as a clearly-labelled placeholder: it
finds active (non-silent) spans and emits them as segments. ``load()`` is the
seam for a real classifier (e.g. a YAMNet/PANNs-style model mapping frames to
``classmaps.AED_LABELS``). This keeps the audio shapes exercisable end-to-end
without fabricating model-quality confidences.

Pure (numpy only) so the segmentation logic is directly unit-testable.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import RunnerOutput, Summary

AED_LABELS = ["speech", "music", "siren", "alarm", "keyword", "noise", "silence"]


class AEDRunner:
    model_id = "aed"
    display_name = "AcousticEventDetection"
    version = "0.1.0"
    depends_on: tuple[str, ...] = ()

    def __init__(
        self,
        weights_path: Optional[str] = None,
        sample_rate: int = 16000,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        silence_db: float = -40.0,
        conf_threshold: float = 0.3,
    ):
        self.weights_path = weights_path
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.silence_db = silence_db
        self.conf_threshold = conf_threshold
        self.input_spec = f"{sample_rate // 1000}kHz mono"
        self.labels = AED_LABELS
        self.model = None

    def load(self) -> None:
        """SEAM: load a real acoustic event classifier from ``weights_path``.

        Until a model is provided this is a no-op and ``predict`` uses the
        energy-based placeholder. A real implementation would set ``self.model``
        and classify frames into ``self.labels``.
        """
        return None

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _to_mono_float(samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim > 1:  # average channels
            x = x.mean(axis=tuple(range(1, x.ndim)))
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1.0:  # integer PCM -> [-1, 1]
            x = x / 32768.0
        return x

    def _frame_rms_db(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        frame = max(1, int(sr * self.frame_ms / 1000))
        hop = max(1, int(sr * self.hop_ms / 1000))
        if x.size < frame:
            rms = np.array([np.sqrt(np.mean(x**2))]) if x.size else np.array([0.0])
            return _to_db(rms), hop
        n = 1 + (x.size - frame) // hop
        rms = np.empty(n, dtype=np.float32)
        for i in range(n):
            seg = x[i * hop : i * hop + frame]
            rms[i] = np.sqrt(np.mean(seg**2))
        return _to_db(rms), hop

    # ---- inference ---------------------------------------------------------
    def predict(self, samples: np.ndarray, sample_rate: Optional[int] = None) -> RunnerOutput:
        if self.model is None:
            self.load()
        sr = int(sample_rate or self.sample_rate)
        x = self._to_mono_float(samples)
        duration_s = (x.size / sr) if sr else 0.0
        db, hop = self._frame_rms_db(x, sr)

        active = db > self.silence_db
        segments: list[dict] = []
        i = 0
        n = len(active)
        # span scaling: frame i centres at i*hop/sr seconds
        span = (hop / sr) if sr else 0.0
        while i < n:
            if not active[i]:
                i += 1
                continue
            j = i
            while j < n and active[j]:
                j += 1
            start_s = i * span
            end_s = min(duration_s, j * span) if duration_s else j * span
            window = db[i:j]
            # confidence: how far above the silence floor, normalized
            conf = float(np.clip((np.mean(window) - self.silence_db) / abs(self.silence_db), 0.0, 1.0))
            if conf >= self.conf_threshold:
                segments.append(
                    {
                        "start_frac": (start_s / duration_s) if duration_s else 0.0,
                        "end_frac": (end_s / duration_s) if duration_s else 0.0,
                        "label": "speech",  # placeholder until a real classifier lands
                        "conf": round(conf, 3),
                        "keyword": None,
                    }
                )
            i = j

        summary = Summary(
            present=len(segments) > 0,
            count=len(segments),
            extras={"duration_s": round(duration_s, 3), "n_segments": len(segments)},
        )
        return RunnerOutput(
            summary=summary,
            raw={"segments": segments, "duration_s": duration_s, "sample_rate": sr},
        )

    def waveform_peaks(self, samples: np.ndarray, n: int = 220) -> list[float]:
        """Downsample to ~``n`` absolute peaks for the Clip.wave overview."""
        x = np.abs(self._to_mono_float(samples))
        if x.size == 0:
            return []
        if x.size <= n:
            return [round(float(v), 4) for v in x]
        bins = np.array_split(x, n)
        return [round(float(np.max(b)) if b.size else 0.0, 4) for b in bins]


def _to_db(rms: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(rms, 1e-8))
