from __future__ import annotations

import json
from pathlib import Path

from sensveridian.augmentation.manual_distance import (
    DistanceOverrides,
    ImageOverride,
)


def test_empty_overrides_returns_none(tmp_path: Path) -> None:
    o = DistanceOverrides.empty()
    assert o.lookup(
        image_path=tmp_path / "foo.jpg",
        image_id="abc",
        model_id="amod",
        detection_idx=0,
    ) is None


def test_global_applies_when_no_image_entry(tmp_path: Path) -> None:
    o = DistanceOverrides(global_ft=5.5)
    assert o.lookup(
        image_path=tmp_path / "foo.jpg",
        image_id="abc",
        model_id="amod",
        detection_idx=0,
    ) == 5.5


def test_image_default_overrides_global(tmp_path: Path) -> None:
    o = DistanceOverrides(
        global_ft=5.0,
        images={"foo.jpg": ImageOverride(default=7.0)},
    )
    assert o.lookup(
        image_path=tmp_path / "foo.jpg",
        image_id="abc",
        model_id="amod",
        detection_idx=0,
    ) == 7.0


def test_detection_override_beats_image_default(tmp_path: Path) -> None:
    o = DistanceOverrides(
        global_ft=5.0,
        images={
            "foo.jpg": ImageOverride(
                default=7.0,
                detections={"amod:1": 9.25},
            )
        },
    )
    assert o.lookup(
        image_path=tmp_path / "foo.jpg",
        image_id="abc",
        model_id="amod",
        detection_idx=1,
    ) == 9.25
    assert o.lookup(
        image_path=tmp_path / "foo.jpg",
        image_id="abc",
        model_id="amod",
        detection_idx=0,
    ) == 7.0


def test_image_key_matches_on_path_basename_stem_or_image_id(tmp_path: Path) -> None:
    p = tmp_path / "door_01.jpg"
    o = DistanceOverrides(images={"door_01": ImageOverride(default=4.0)})
    assert o.lookup(image_path=p, image_id="xxxx", model_id="amod", detection_idx=0) == 4.0

    o2 = DistanceOverrides(images={str(p): ImageOverride(default=4.0)})
    assert o2.lookup(image_path=p, image_id="xxxx", model_id="amod", detection_idx=0) == 4.0

    o3 = DistanceOverrides(images={"xxxx": ImageOverride(default=4.0)})
    assert o3.lookup(image_path=p, image_id="xxxx", model_id="amod", detection_idx=0) == 4.0


def test_from_json_round_trip(tmp_path: Path) -> None:
    payload = {
        "global_ft": 5.0,
        "images": {
            "door_01.jpg": {
                "default": 4.5,
                "detections": {"amod:0": 4.1, "fd:0": 4.8},
                "real_sizes_m": {"amod:0": {"h": 1.7, "w": 0.45}},
            }
        },
    }
    p = tmp_path / "d0.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    o = DistanceOverrides.from_json(p)
    assert o.global_ft == 5.0
    assert "door_01.jpg" in o.images
    io = o.images["door_01.jpg"]
    assert io.default == 4.5
    assert io.detections["amod:0"] == 4.1
    assert io.detections["fd:0"] == 4.8
    assert io.real_sizes_m["amod:0"] == (1.7, 0.45)


def test_real_size_lookup(tmp_path: Path) -> None:
    p = tmp_path / "foo.jpg"
    o = DistanceOverrides(
        images={
            "foo.jpg": ImageOverride(
                default=6.0,
                real_sizes_m={"amod:0": (1.7, 0.45)},
            )
        }
    )
    assert o.real_size_lookup(
        image_path=p,
        image_id="abc",
        model_id="amod",
        detection_idx=0,
    ) == (1.7, 0.45)
    assert (
        o.real_size_lookup(
            image_path=p,
            image_id="abc",
            model_id="amod",
            detection_idx=1,
        )
        is None
    )


def test_covers_all(tmp_path: Path) -> None:
    p = tmp_path / "foo.jpg"
    refs = [("amod", 0), ("amod", 1)]

    assert not DistanceOverrides.empty().covers_all(
        image_path=p, image_id="abc", detection_refs=refs
    )
    assert DistanceOverrides(global_ft=3.0).covers_all(
        image_path=p, image_id="abc", detection_refs=refs
    )
    partial = DistanceOverrides(
        images={"foo.jpg": ImageOverride(detections={"amod:0": 3.0})}
    )
    assert not partial.covers_all(
        image_path=p, image_id="abc", detection_refs=refs
    )
