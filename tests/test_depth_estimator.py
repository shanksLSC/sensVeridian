from __future__ import annotations

import types

import numpy as np

from sensveridian.augmentation.depth import ZoeDepthEstimator


def test_zoedepth_estimate_depth_ft_with_injected_model(tiny_image_bgr: np.ndarray) -> None:
    class _FakeModel:
        def infer_pil(self, _pil):
            return np.ones((tiny_image_bgr.shape[0], tiny_image_bgr.shape[1]), dtype=np.float32) * 2.0

    z = ZoeDepthEstimator(device="cpu")
    z.model = _FakeModel()
    depth_ft = z.estimate_depth_ft(tiny_image_bgr)
    assert depth_ft.shape == tiny_image_bgr.shape[:2]
    assert np.isclose(float(depth_ft[0, 0]), 2.0 * 3.28084, atol=1e-5)


def test_zoedepth_load_uses_torch_hub(monkeypatch) -> None:
    class _Loaded:
        def __init__(self):
            self.device = None

        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            return self

    loaded = _Loaded()

    class _Hub:
        @staticmethod
        def load(repo, model_name, pretrained=True):
            assert repo == "isl-org/ZoeDepth"
            assert model_name == "ZoeD_NK"
            assert pretrained is True
            return loaded

    fake_torch = types.SimpleNamespace(hub=_Hub)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    z = ZoeDepthEstimator(device="cpu")
    z.load()
    assert z.model is loaded
    assert loaded.device == "cpu"
