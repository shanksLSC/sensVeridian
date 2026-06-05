"""Pydantic request/response models — match the API contract and the shapes the
front-end's MockAdapter returns (design_reference/app/api.js)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel

Box = list[float]  # [x, y, w, h] normalized 0..1


class DatasetSummary(BaseModel):
    id: str
    name: str
    desc: str = ""
    kind: Literal["vision", "audio"] = "vision"
    models: list[str] = []
    count: int = 0
    agreement: float = 0.0
    conflicts: int = 0
    reviewed: int = 0
    runId: Optional[str] = None


class Detection(BaseModel):
    id: str
    cls: str
    model: str
    gt: Optional[Box] = None
    pred: Optional[Box] = None
    conf: float = 0.0
    state: Literal["match", "miss", "fp", "mismatch"] = "match"
    iou: float = 0.0
    mask: Optional[list[list[float]]] = None
    identity: Optional[dict[str, Any]] = None  # FR: {gt, pred, sim, person_id}
    decoded: Optional[dict[str, Any]] = None  # QR: {gt, pred}


class Image(BaseModel):
    id: str
    datasetId: str
    w: int
    h: int
    d0_ft: float = 6.0
    augmented: bool = False
    status: Literal["unreviewed", "verified", "flagged"] = "unreviewed"
    captured: Optional[str] = None
    objects: list[Detection] = []


class Segment(BaseModel):
    id: str
    start: float
    end: float
    gt: Optional[str] = None
    pred: Optional[str] = None
    conf: float = 0.0
    state: str = "match"
    keyword: Optional[str] = None


class Clip(BaseModel):
    id: str
    name: str
    dur: float
    wave: list[float]
    segments: list[Segment] = []


class ModelVersion(BaseModel):
    version: str
    weights_sha: str
    date: str
    metrics: dict[str, float]
    notes: str = ""
    current: bool = False


class Model(BaseModel):
    id: str
    display_name: str
    short: str
    input: str
    weights_path: str
    classes: int
    depends_on: Optional[str] = None
    versions: list[ModelVersion] = []


class Flip(BaseModel):
    datasetId: str
    imageId: str
    cls: str
    regress: bool
    confA: float
    confB: float
    detId: str


class ReviewPatch(BaseModel):
    verdict: Literal["accepted", "rejected", "edited", "flagged"]
    box: Optional[Box] = None


class BulkReview(BaseModel):
    target_ids: list[str]
    patch: ReviewPatch


class IngestGroup(BaseModel):
    tag: str
    label: str
    models: list[str]
    videos: int = 0
    frames: int = 0
    dataset: Optional[str] = None
    isNew: bool = False


class IngestJobSpec(BaseModel):
    source: Literal["video", "import", "connection"] = "video"
    trustThreshold: Optional[float] = None
    groups: list[IngestGroup] = []


class JobHandle(BaseModel):
    jobId: str
    status: str = "queued"


class ImportSpec(BaseModel):
    kind: Literal["vision", "audio"] = "vision"
    name: str
    desc: str = ""
    format: Optional[str] = None  # coco|yolo|csv|voc
    mapping: Optional[dict[str, str]] = None
    compareRun: Optional[str] = None
    models: list[str] = ["amod"]
    palette: str = "dusk"
    n: Optional[int] = None
    # Inline-import path: parsed annotations/segments may be supplied directly
    # (used by tests and when the server parses uploaded label files into the
    # request before lint/insert). The UI's presigned-upload flow populates
    # these server-side; the bare contract body omits them.
    annotations: Optional[list[dict[str, Any]]] = None
    image_refs: Optional[list[str]] = None
    segments: Optional[list[dict[str, Any]]] = None


class LintCheck(BaseModel):
    sev: Literal["ok", "warn", "error", "info"]
    label: str
    n: Optional[int] = None
    note: str = ""


class ImportResult(BaseModel):
    ok: bool = True
    datasetId: str
    score: Optional[int] = None
    checks: list[LintCheck] = []


class ConnectionSpec(BaseModel):
    uri: str
    schedule: Literal["hourly", "15min", "on-object"] = "hourly"
    pipeline: str = "auto-label:amod"


class LineageGraph(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[list[str]]
