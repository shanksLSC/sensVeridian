from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from sensveridian.orchestrator import Orchestrator
from sensveridian.runners.base import RunnerOutput, Summary


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[:] = color
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Could not write {path}")


def _mock_runner(model_id: str, out: RunnerOutput):
    runner = MagicMock()
    runner.model_id = model_id
    runner.display_name = model_id.upper()
    runner.version = "vtest"
    runner.weights_path = f"/tmp/{model_id}.h5"
    runner.predict.return_value = out
    return runner


def _build_orch(duck_store, file_registry, monkeypatch) -> Orchestrator:
    monkeypatch.setattr("sensveridian.orchestrator.hash_file", lambda _p: "sha-test")
    orch = Orchestrator(store=duck_store, registry=file_registry)
    fd_out = RunnerOutput(
        summary=Summary(True, 1, {"n_FD": 1}),
        raw={
            "detections": [{"bbox": [1, 1, 12, 12], "conf": 0.9}],
            "crops": [np.zeros((8, 8, 3), dtype=np.uint8)],
        },
    )
    orch.runners = {
        "amod": _mock_runner("amod", RunnerOutput(Summary(True, 2, {}), {"detections": [{"bbox": [1, 2, 3, 4]}]})),
        "qrcode": _mock_runner("qrcode", RunnerOutput(Summary(False, 0, {}), {"detections": []})),
        "fd": _mock_runner("fd", fd_out),
        "fr": _mock_runner("fr", RunnerOutput(Summary(False, 0, {"n_FID": 0}), {"recognized": []})),
    }
    return orch


def test_orchestrator_ingest_skip_existing_and_force(tmp_path: Path, duck_store, file_registry, monkeypatch) -> None:
    _write_image(tmp_path / "a.png", (255, 0, 0))
    _write_image(tmp_path / "b.jpg", (0, 255, 0))
    (tmp_path / "ignore.txt").write_text("not an image", encoding="utf-8")

    orch = _build_orch(duck_store, file_registry, monkeypatch)
    selected = {"amod", "qrcode", "fd", "fr"}
    res1 = orch.ingest(tmp_path, run_id="r1", selected_models=selected, skip_existing=True)
    assert res1.images_seen == 2
    assert res1.images_ingested == 2
    assert res1.predictions_written == 8

    # skip_existing=True should ingest nothing on immediate rerun.
    res2 = orch.ingest(tmp_path, run_id="r1", selected_models=selected, skip_existing=True)
    assert res2.images_seen == 2
    assert res2.images_ingested == 0

    # skip_existing=False forces writes again.
    res3 = orch.ingest(tmp_path, run_id="r1", selected_models=selected, skip_existing=False)
    assert res3.images_ingested == 2
    assert res3.predictions_written == 8

    # Raw payload for FD should strip crops and keep crop_count.
    fd_raw = duck_store.query_df("select payload from predictions_raw where model_id='fd' limit 1")
    payload = fd_raw.iloc[0]["payload"]
    payload_str = str(payload)
    assert "crop_count" in payload_str
    assert "crops" not in payload_str


def test_orchestrator_order_and_no_fr_call_when_not_selected(tmp_path: Path, duck_store, file_registry, monkeypatch) -> None:
    _write_image(tmp_path / "x.png", (123, 50, 200))
    orch = _build_orch(duck_store, file_registry, monkeypatch)

    assert orch._ordered_models({"fr", "fd", "qrcode", "amod"}) == ["amod", "qrcode", "fd", "fr"]
    assert orch._ordered_models({"fd", "amod"}) == ["amod", "fd"]

    selected = {"amod", "fd"}  # fr excluded
    orch.ingest(tmp_path, run_id="r2", selected_models=selected, skip_existing=False)
    assert orch.runners["fr"].predict.call_count == 0
