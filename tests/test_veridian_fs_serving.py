"""Tests for filesystem browse + image serving (local-source ingest plumbing)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from sensveridian.api import deps
from sensveridian.api.config import settings
from sensveridian.api.main import app
from sensveridian.api.routers import datasets as datasets_router
from sensveridian.api.routers import fs as fs_router
from sensveridian.ingest import kinds


# ---- pure: discovery -------------------------------------------------------
def test_discover_images_and_limit(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    sub = tmp_path / "sub"; sub.mkdir()
    (sub / "d.jpeg").write_bytes(b"x")
    found = kinds.discover_images(tmp_path)
    assert {p.name for p in found} == {"a.jpg", "b.png", "d.jpeg"}
    assert len(kinds.discover_images(tmp_path, limit=2)) == 2


def test_discover_videos(tmp_path: Path):
    (tmp_path / "v.mp4").write_bytes(b"x")
    (tmp_path / "x.jpg").write_bytes(b"x")
    assert {p.name for p in kinds.discover_videos(tmp_path)} == {"v.mp4"}


# ---- pure: traversal guard -------------------------------------------------
def test_safe_resolve_guards_escape(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_root", str(tmp_path))
    (tmp_path / "ds").mkdir()
    assert fs_router.safe_resolve("ds") == (tmp_path / "ds").resolve()
    with pytest.raises(HTTPException) as ei:
        fs_router.safe_resolve("../../etc")
    assert ei.value.status_code == 400


def test_allowed_image_path(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_root", str(tmp_path))
    monkeypatch.setattr(settings, "frames_root", str(tmp_path / "frames"))
    img = tmp_path / "ds" / "img.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\xff\xd8\xff")  # jpeg-ish
    assert datasets_router._allowed_image_path(str(img)) == img.resolve()
    assert datasets_router._allowed_image_path("/etc/passwd") is None
    assert datasets_router._allowed_image_path("") is None


# ---- API: fs browse --------------------------------------------------------
def test_fs_datasets_lists_media_folders(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_root", str(tmp_path))
    ds = tmp_path / "street"; (ds / "images").mkdir(parents=True)
    (ds / "images" / "a.jpg").write_bytes(b"x")
    (ds / "labels").mkdir(); (ds / "labels" / "a.txt").write_bytes(b"0 0.5 0.5 0.1 0.1")
    (tmp_path / "empty").mkdir()  # no media -> excluded
    with TestClient(app) as c:
        r = c.get("/api/v1/fs/datasets")
    assert r.status_code == 200
    entries = {e["name"]: e for e in r.json()["entries"]}
    assert "street" in entries and "empty" not in entries
    assert entries["street"]["images"] == 1 and entries["street"]["has_labels"] is True


def test_fs_browse_rejects_traversal(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_root", str(tmp_path))
    with TestClient(app) as c:
        assert c.get("/api/v1/fs/browse", params={"path": "../.."}).status_code == 400


# ---- API: image serving ----------------------------------------------------
class _RawStore:
    def __init__(self, path):
        self._path = path

    async def image_path(self, image_id):
        return self._path


def test_image_raw_serves_bytes_and_guards(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_root", str(tmp_path))
    monkeypatch.setattr(settings, "frames_root", str(tmp_path / "frames"))
    img = tmp_path / "ds" / "img.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\xff\xd8\xffhello")

    app.dependency_overrides[deps.get_store] = lambda: _RawStore(str(img))
    try:
        with TestClient(app) as c:
            ok = c.get("/api/v1/datasets/ds/images/abc/raw")
            assert ok.status_code == 200 and ok.content == b"\xff\xd8\xffhello"
        # path outside the allowed roots -> 404
        app.dependency_overrides[deps.get_store] = lambda: _RawStore("/etc/passwd")
        with TestClient(app) as c:
            assert c.get("/api/v1/datasets/ds/images/abc/raw").status_code == 404
    finally:
        app.dependency_overrides.clear()
