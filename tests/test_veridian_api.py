"""API surface tests for the Veridian Studio backend.

Uses FastAPI's TestClient with a dependency-overridden fake store, so the route
wiring, request/response shapes (vs the contract), and auth are exercised
without a live PostgreSQL/Redis.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sensveridian.api import deps
from sensveridian.api.main import app


class FakeStore:
    """Minimal async stand-in for AsyncPgStore returning canned, contract-shaped data."""

    def __init__(self):
        self.saved_reviews = []
        self.created_datasets = []
        self.created_connections = []
        self.created_jobs = []
        self.inserted_gt = 0

    async def list_datasets(self):
        return [{"id": "street_scenes", "name": "Street Scenes", "desc": "", "kind": "vision",
                 "models": ["amod"], "count": 28, "agreement": 0.81, "conflicts": 34,
                 "reviewed": 9, "runId": "baseline"}]

    async def get_dataset_row(self, dataset_id):
        return {"id": dataset_id, "name": "Street Scenes", "desc": "", "kind": "vision",
                "models": ["amod"], "palette": "dusk", "runId": "baseline"}

    async def dataset_aggregates(self, dataset_id, kind="vision", run_id="baseline"):
        return {"count": 28, "agreement": 0.81, "conflicts": 34, "reviewed": 9}

    async def list_dataset_images(self, dataset_id, offset=0, limit=200, filter=None, sort=None):
        return [{"id": "sha0", "w": 1280, "h": 800, "augmented": False, "d0_ft": 6.2, "captured": "2026-03-11"}]

    async def get_image(self, dataset_id, image_id, run_id="baseline"):
        return {
            "id": image_id, "datasetId": dataset_id, "w": 1280, "h": 800, "d0_ft": 6.2,
            "augmented": False, "status": "unreviewed", "captured": "2026-03-11",
            "objects": [{"id": f"{image_id}_amod_0", "cls": "car", "model": "amod",
                         "gt": [0.3, 0.4, 0.2, 0.2], "pred": [0.31, 0.41, 0.2, 0.2],
                         "conf": 0.91, "state": "match", "iou": 0.86}],
        }

    async def build_image_detections(self, dataset_id, image_id, run_id="baseline", model_id=None):
        return [{"id": f"{image_id}_amod_0", "cls": "car", "model": "amod",
                 "gt": [0.3, 0.4, 0.2, 0.2], "pred": [0.31, 0.41, 0.2, 0.2],
                 "conf": 0.91, "state": "match", "iou": 0.86}]

    async def get_clip(self, dataset_id, clip_id):
        return {"id": clip_id, "name": "clip_001.wav", "dur": 42.0, "wave": [0.1, 0.2],
                "segments": [{"id": "seg_0_1", "start": 0.04, "end": 0.22, "gt": "speech",
                              "pred": "speech", "conf": 0.93, "state": "match", "keyword": None}]}

    async def list_layers(self, dataset_id):
        return [{"id": "baseline:amod", "type": "prediction", "model": "amod",
                 "version": "8.2.0", "runId": "baseline", "source": None, "createdAt": None}]

    async def list_models(self):
        return [{"id": "amod", "display_name": "AutomotiveMultiObjectDetection", "short": "AMOD",
                 "input": "320×320×3", "weights_path": "/w/amod.h5", "classes": 6, "depends_on": None,
                 "versions": [{"version": "8.2.0", "weights_sha": "abc", "date": "2026-04-19",
                               "metrics": {"mAP": 0.91}, "notes": "current", "current": True}]}]

    async def regressions(self, model_id, base, candidate):
        return []

    async def promote_version(self, model_id, version):
        return None

    async def resolve_dataset_id(self, target_id):
        return "street_scenes"

    async def save_review(self, target_id, kind, verdict, dataset_id, box):
        self.saved_reviews.append((target_id, kind, verdict, dataset_id, box))

    async def bulk_review(self, target_ids, kinds, verdict, dataset_id):
        self.saved_reviews.extend((t, k, verdict, dataset_id, None) for t, k in zip(target_ids, kinds))
        return len(target_ids)

    async def lineage(self):
        return {"nodes": [{"id": "d:street", "type": "dataset", "label": "Street", "meta": "vision"}],
                "edges": [["d:street", "r:baseline:amod"]]}

    async def ensure_dataset(self, dataset_id, name, kind="vision", descr="", models=None, palette=None, run_id=None):
        self.created_datasets.append(dataset_id)

    async def upsert_layer(self, *a, **k):
        return None

    async def insert_gt_boxes(self, layer_id, dataset_id, annotations):
        self.inserted_gt += len(annotations)
        return len(annotations)

    async def create_ingest_job(self, job_id, source, spec, frames_total):
        self.created_jobs.append(job_id)

    async def get_ingest_job(self, job_id):
        return {"jobId": job_id, "source": "video", "status": "running", "stage": "Auto-labelling",
                "progress": 62, "framesDone": 526, "framesTotal": 848, "error": None}

    async def create_connection(self, connection_id, uri, schedule, pipeline):
        self.created_connections.append(connection_id)

    async def commit(self):
        return None


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def client(store: FakeStore):
    app.dependency_overrides[deps.get_store] = lambda: store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["storage"] == "postgres"


def test_list_datasets(client):
    r = client.get("/api/v1/datasets")
    assert r.status_code == 200
    body = r.json()
    assert body[0]["id"] == "street_scenes" and body[0]["agreement"] == 0.81


def test_get_dataset_attaches_images(client):
    r = client.get("/api/v1/datasets/street_scenes")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 28
    assert isinstance(body["images"], list) and body["images"][0]["w"] == 1280


def test_get_image_and_predictions(client):
    r = client.get("/api/v1/datasets/street_scenes/images/sha0")
    assert r.status_code == 200
    img = r.json()
    assert img["objects"][0]["state"] == "match"
    r2 = client.get("/api/v1/datasets/street_scenes/images/sha0/predictions?run_id=baseline&model_id=amod")
    assert r2.status_code == 200 and r2.json()[0]["cls"] == "car"


def test_models_and_promote(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200 and r.json()[0]["versions"][0]["current"] is True
    r2 = client.post("/api/v1/models/amod/versions/8.2.0:promote")
    assert r2.status_code == 200 and r2.json()["current"] == "8.2.0"


def test_reviews_put_resolves_dataset(client, store):
    r = client.put("/api/v1/reviews/sha0_amod_0", json={"verdict": "accepted"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert store.saved_reviews[0][3] == "street_scenes"  # dataset resolved server-side


def test_reviews_bulk(client, store):
    r = client.post("/api/v1/reviews:bulk", json={"target_ids": ["a_amod_0", "a_amod_1"],
                                                  "patch": {"verdict": "accepted"}})
    assert r.status_code == 200 and r.json()["n"] == 2


def test_import_runs_lint_and_persists(client, store):
    body = {
        "kind": "vision", "name": "Imported Eval Set", "format": "coco",
        "annotations": [{"image": "a.jpg", "cls": "car", "box": [0.1, 0.1, 0.2, 0.2], "img_w": 1000, "img_h": 1000}],
        "image_refs": ["a.jpg"],
    }
    r = client.post("/api/v1/datasets:import", json=body)
    assert r.status_code == 200
    res = r.json()
    assert res["datasetId"] == "imported_eval_set"
    assert res["score"] == 100
    assert store.inserted_gt == 1


def test_ingest_job_and_status(client, store, monkeypatch):
    async def _noop(job_id, spec):
        return None

    monkeypatch.setattr("sensveridian.ingest.worker.enqueue_ingest", _noop)
    r = client.post("/api/v1/ingest/jobs", json={"source": "video", "trustThreshold": 0.85,
                                                 "groups": [{"tag": "street", "label": "Street",
                                                             "models": ["amod"], "frames": 848}]})
    assert r.status_code == 200
    job_id = r.json()["jobId"]
    assert job_id in store.created_jobs
    r2 = client.get(f"/api/v1/ingest/jobs/{job_id}")
    assert r2.status_code == 200 and r2.json()["status"] == "running"


def test_connections(client, store):
    r = client.post("/api/v1/connections", json={"uri": "s3://b/incoming/", "schedule": "hourly",
                                                 "pipeline": "auto-label:amod"})
    assert r.status_code == 200 and r.json()["id"].startswith("conn_")
    assert store.created_connections


def test_lineage(client):
    r = client.get("/api/v1/lineage")
    assert r.status_code == 200
    assert r.json()["nodes"][0]["type"] == "dataset"


def test_auth_enforced_when_token_set(client, monkeypatch):
    monkeypatch.setattr(deps.settings, "auth_token", "secret")
    assert client.get("/api/v1/datasets").status_code == 401
    ok = client.get("/api/v1/datasets", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
