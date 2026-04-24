from __future__ import annotations

import types

import numpy as np

from sensveridian.augmentation.segment import SAMSegmenter, union_masks


def test_union_masks_resizes_and_unions() -> None:
    m1 = np.zeros((8, 8), dtype=np.uint8)
    m1[1:3, 1:3] = 1
    m2 = np.zeros((4, 4), dtype=np.uint8)
    m2[0:2, 0:2] = 1
    out = union_masks([m1, m2], (8, 8))
    assert out.shape == (8, 8)
    assert int(out.max()) == 1
    assert int(out.sum()) > 0


def test_sam_segmenter_segment_with_mocked_backend(monkeypatch) -> None:
    class _FakeSam:
        def to(self, device: str):
            return self

    class _FakePredictor:
        def __init__(self, _sam):
            self._image = None

        def set_image(self, image_rgb):
            self._image = image_rgb

        def predict(self, box, multimask_output=False):
            mask = np.zeros((1, 16, 16), dtype=np.uint8)
            mask[:, 2:6, 3:7] = 1
            return mask, None, None

    fake_mod = types.SimpleNamespace(
        sam_model_registry={"vit_b": lambda checkpoint: _FakeSam()},
        SamPredictor=_FakePredictor,
    )
    monkeypatch.setitem(__import__("sys").modules, "segment_anything", fake_mod)

    seg = SAMSegmenter(checkpoint_path="/tmp/sam.pth", model_type="vit_b", device="cpu")
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    masks = seg.segment(img, [[1, 1, 8, 8], [2, 2, 9, 9]])
    assert len(masks) == 2
    assert masks[0].shape == (16, 16)
