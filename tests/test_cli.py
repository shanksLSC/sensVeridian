from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

import sensveridian.cli as cli_mod
from sensveridian.store.duck import DuckStore


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def _make_store_factory(tmp_path: Path):
    schema = Path(__file__).resolve().parents[1] / "src" / "sensveridian" / "store" / "schema.sql"
    db_path = tmp_path / "cli.duckdb"

    def _factory() -> DuckStore:
        s = DuckStore(db_path=db_path, schema_path=schema)
        s.migrate()
        return s

    return _factory, db_path


def _write_image(path: Path) -> None:
    img = np.ones((16, 16, 3), dtype=np.uint8) * 255
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Could not write test image {path}")


def test_cli_stats_query_export(cli_runner: CliRunner, tmp_path: Path, monkeypatch, file_registry) -> None:
    store_factory, _db = _make_store_factory(tmp_path)
    monkeypatch.setattr(cli_mod, "_store", store_factory)
    monkeypatch.setattr(cli_mod, "_registry", lambda: file_registry)

    r_stats = cli_runner.invoke(cli_mod.app, ["stats"])
    assert r_stats.exit_code == 0
    assert "images:" in r_stats.stdout
    assert "runs:" in r_stats.stdout

    r_query = cli_runner.invoke(cli_mod.app, ["query", "select 1 as x"])
    assert r_query.exit_code == 0
    assert "x" in r_query.stdout

    r_empty = cli_runner.invoke(cli_mod.app, ["query", "select * from images where 1=0"])
    assert r_empty.exit_code == 0
    assert "No rows." in r_empty.stdout

    out_parquet = tmp_path / "out.parquet"
    r_export = cli_runner.invoke(cli_mod.app, ["export", "--to", str(out_parquet)])
    assert r_export.exit_code == 0
    assert out_parquet.exists()


def test_cli_faces_seed_list_clear(cli_runner: CliRunner, tmp_path: Path, monkeypatch, file_registry) -> None:
    store_factory, _db = _make_store_factory(tmp_path)
    monkeypatch.setattr(cli_mod, "_store", store_factory)
    monkeypatch.setattr(cli_mod, "_registry", lambda: file_registry)

    r0 = cli_runner.invoke(cli_mod.app, ["faces", "list"])
    assert r0.exit_code == 0
    assert "No entries." in r0.stdout

    r_seed = cli_runner.invoke(cli_mod.app, ["faces", "seed", "--n", "3", "--clear-first"])
    assert r_seed.exit_code == 0
    assert "Seeded" in r_seed.stdout
    assert "dummy face records." in r_seed.stdout
    assert len(file_registry.list_entries()) == 3

    r_list = cli_runner.invoke(cli_mod.app, ["faces", "list"])
    assert r_list.exit_code == 0
    assert "person_001" in r_list.stdout

    r_clear = cli_runner.invoke(cli_mod.app, ["faces", "clear"])
    assert r_clear.exit_code == 0
    assert "Face registry cleared." in r_clear.stdout
    assert file_registry.list_entries() == []


def test_cli_augment_list_no_rows(cli_runner: CliRunner, tmp_path: Path, monkeypatch, file_registry) -> None:
    store_factory, _db = _make_store_factory(tmp_path)
    monkeypatch.setattr(cli_mod, "_store", store_factory)
    monkeypatch.setattr(cli_mod, "_registry", lambda: file_registry)

    r = cli_runner.invoke(cli_mod.app, ["augment", "list", "missing_parent"])
    assert r.exit_code == 0
    assert "No augmentations found." in r.stdout


def test_cli_ingest_with_patched_orchestrator(cli_runner: CliRunner, tmp_path: Path, monkeypatch, file_registry) -> None:
    store_factory, _db = _make_store_factory(tmp_path)
    monkeypatch.setattr(cli_mod, "_store", store_factory)
    monkeypatch.setattr(cli_mod, "_registry", lambda: file_registry)

    @dataclass
    class _Res:
        images_seen: int
        images_ingested: int
        predictions_written: int

    class _FakeOrchestrator:
        def __init__(self, store, registry):
            self.store = store
            self.registry = registry

        def ingest(self, image_root, run_id, selected_models, skip_existing=True):
            return _Res(images_seen=2, images_ingested=2, predictions_written=8)

    monkeypatch.setattr(cli_mod, "Orchestrator", _FakeOrchestrator)

    image_dir = tmp_path / "imgs"
    image_dir.mkdir(parents=True, exist_ok=True)
    _write_image(image_dir / "a.png")

    r = cli_runner.invoke(cli_mod.app, ["ingest", str(image_dir), "--run-id", "r1"])
    assert r.exit_code == 0
    assert "Ingest complete:" in r.stdout
    assert "seen=" in r.stdout
    assert "ingested=" in r.stdout
    assert "writes=" in r.stdout
