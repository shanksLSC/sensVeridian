"""Detection fusion — the core of ``GET /datasets/{id}/images/{imageId}``.

Fuses a *prediction* layer (sensVeridian ``predictions_raw`` payloads) with the
verified *ground-truth* layer (imported ``gt_boxes`` and/or human ``reviews``)
into the ``Detection[]`` shape the canvas renders. This is the one piece of real
logic the handoff flags as "the one to get right".

It corrects three things the scaffold stub got wrong against the *actual* runner
outputs (see ``src/sensveridian/runners/*``):

* FaceRecognition emits ``raw["recognized"]`` (not ``"detections"``), each with
  ``matched_person_id`` / ``score`` / ``bbox`` — handled here for identity.
* FaceDetection boxes are **absolute pixels** (``safe_bbox_xyxy``) while AMOD /
  QR boxes come straight from the model head; normalization uses the image
  dimensions with a per-model coordinate-space policy.
* QR decoded text lives in ``raw["decoded_texts"]``.

State semantics match the front-end (``design_reference/app/data.js``):
``match`` (gt≈pred, same class), ``mismatch`` (overlap, wrong class/identity),
``fp`` (pred, no gt), ``miss`` (gt, no pred = false negative).

Pure module: stdlib only — no FastAPI / SQLAlchemy — so it is directly
unit-testable.
"""
from __future__ import annotations

from typing import Any, Optional

from .classmaps import class_name

Box = list[float]  # [x, y, w, h] normalized 0..1

# Per-model interpretation of the raw bbox coordinate space.
#   "abs_px" : absolute pixels relative to the source image (FD/FR crops path).
#   "auto"   : normalized if every coord <= ~1, else treated as absolute pixels.
BOX_SPACE = {"fd": "abs_px", "fr": "abs_px", "amod": "auto", "qrcode": "auto"}

DEFAULT_IOU_THR = 0.5


# ---- geometry --------------------------------------------------------------
def iou_xywh(a: Optional[Box], b: Optional[Box]) -> float:
    """IoU of two ``[x, y, w, h]`` boxes (same convention as data.js / the
    scaffold)."""
    if not a or not b:
        return 0.0
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    uni = a[2] * a[3] + b[2] * b[3] - inter
    return inter / uni if uni > 0 else 0.0


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def normalize_box(bbox_xyxy: Any, img_w: int, img_h: int, space: str = "auto") -> Optional[Box]:
    """Convert a detection bbox ``[x1, y1, x2, y2]`` into normalized
    ``[x, y, w, h]`` (top-left origin), clamped to ``0..1``.

    ``space`` selects how the input coordinates are interpreted:
    ``"abs_px"`` divides by the image size; ``"norm"`` assumes already 0..1;
    ``"auto"`` infers (normalized if all coords <= 1.5, else absolute pixels).
    """
    if not bbox_xyxy or len(bbox_xyxy) < 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy[:4])
    if space == "auto":
        space = "norm" if max(x1, y1, x2, y2) <= 1.5 else "abs_px"
    if space == "abs_px":
        w = max(img_w, 1)
        h = max(img_h, 1)
        x1, x2 = x1 / w, x2 / w
        y1, y2 = y1 / h, y2 / h
    # guard against x2<x1 (some heads emit unordered corners)
    nx1, nx2 = min(x1, x2), max(x1, x2)
    ny1, ny2 = min(y1, y2), max(y1, y2)
    x, y = _clamp01(nx1), _clamp01(ny1)
    return [x, y, _clamp01(nx2) - x, _clamp01(ny2) - y]


# ---- prediction extraction (per runner output shape) -----------------------
def extract_pred_detections(
    model_id: str,
    payload: dict,
    img_w: int,
    img_h: int,
    *,
    name_map: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Flatten one model's raw payload into normalized prediction records.

    Returns dicts with keys: ``model``, ``cls``, ``box`` (normalized xywh),
    ``conf``, and optionally ``identity`` (FR), ``decoded`` (QR), ``mask``.
    """
    payload = payload or {}
    space = BOX_SPACE.get(model_id, "auto")
    out: list[dict] = []

    if model_id == "fr":
        # FaceRecognition: raw["recognized"] = [{bbox, matched_person_id, score, embedding}]
        for rec in payload.get("recognized", []):
            pid = rec.get("matched_person_id")
            score = float(rec.get("score") or 0.0)
            pred_name = (name_map or {}).get(pid, pid) if pid else None
            out.append(
                {
                    "model": "fr",
                    "cls": "face",
                    "box": normalize_box(rec.get("bbox"), img_w, img_h, space),
                    "conf": score,
                    "identity": {"pred": pred_name, "sim": round(score, 3), "person_id": pid},
                }
            )
        return out

    decoded_texts = payload.get("decoded_texts", []) if model_id == "qrcode" else []
    for idx, det in enumerate(payload.get("detections", [])):
        rec: dict = {
            "model": model_id,
            "cls": class_name(model_id, int(det.get("class_id", 0))),
            "box": normalize_box(det.get("bbox"), img_w, img_h, space),
            "conf": float(det.get("conf", 0.0)),
        }
        if model_id == "qrcode":
            text = decoded_texts[idx] if idx < len(decoded_texts) else None
            rec["decoded"] = {"pred": text or None}
        if det.get("mask"):
            rec["mask"] = det["mask"]
        out.append(rec)
    return out


# ---- review / gt helpers ---------------------------------------------------
def _identity_matches(gt: dict, pred: dict) -> bool:
    g = (gt.get("identity") or {}).get("gt") if gt else None
    p = (pred.get("identity") or {}).get("pred") if pred else None
    if g is None or p is None:
        return True  # no identity to disagree on
    return str(g) == str(p)


def _merge_identity(pred: dict, gt: Optional[dict], state: str) -> Optional[dict]:
    if "identity" not in pred and not (gt and gt.get("identity")):
        return None
    ident = dict(pred.get("identity") or {})
    gt_name = (gt.get("identity") or {}).get("gt") if gt else None
    ident["gt"] = gt_name
    if state == "miss":
        ident["pred"] = None
    elif state == "fp":
        ident["pred"] = ident.get("pred") or "unknown"
    return ident


def _merge_decoded(pred: dict, gt: Optional[dict], state: str) -> Optional[dict]:
    if "decoded" not in pred and not (gt and gt.get("decoded")):
        return None
    dec = dict(pred.get("decoded") or {})
    gt_text = (gt.get("decoded") or {}).get("gt") if gt else None
    dec["gt"] = gt_text
    if state == "miss":
        dec["pred"] = None
    elif state == "fp":
        dec["pred"] = dec.get("pred") or "(garbled)"
    return dec


def _detection(det_id: str, pred: dict, gt_box: Optional[Box], state: str,
               gt_item: Optional[dict] = None) -> dict:
    pred_box = pred.get("box")
    out = {
        "id": det_id,
        "cls": pred.get("cls", "object"),
        "model": pred.get("model", ""),
        "gt": gt_box,
        "pred": None if state == "miss" else pred_box,
        "conf": round(float(pred.get("conf", 0.0)), 4),
        "state": state,
        "iou": round(iou_xywh(gt_box, pred_box), 3) if state != "miss" else 0.0,
    }
    if pred.get("mask"):
        out["mask"] = pred["mask"]
    ident = _merge_identity(pred, gt_item, state)
    if ident is not None:
        out["identity"] = ident
    dec = _merge_decoded(pred, gt_item, state)
    if dec is not None:
        out["decoded"] = dec
    return out


# ---- the fusion ------------------------------------------------------------
def fuse_detections(
    image_id: str,
    preds_by_model: dict[str, dict],
    img_w: int,
    img_h: int,
    *,
    gt_items: Optional[list[dict]] = None,
    reviews: Optional[dict[str, dict]] = None,
    iou_thr: float = DEFAULT_IOU_THR,
    name_map: Optional[dict[str, str]] = None,
    class_map: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Fuse predictions with ground truth into ``Detection[]``.

    Parameters
    ----------
    preds_by_model : ``{model_id: raw_payload}`` from ``predictions_raw``.
    gt_items : imported ground-truth boxes for this image, each
        ``{"cls", "box":[xywh], "identity"?, "decoded"?}``. When empty *and* no
        reviews exist, predictions are treated as provisional ground truth
        (state ``match``) — the auto-label semantic for a freshly ingested set,
        which avoids a misleading "everything is a false positive" view.
    reviews : ``{det_id: {"verdict", "box"?}}`` human verdicts. ``rejected`` ->
        ``fp``; ``edited`` replaces the gt box; ``accepted`` confirms the pred.
    iou_thr : match threshold (0.5, matching the scaffold/data.js).
    name_map : optional ``person_id -> display name`` for FR identities.
    class_map : optional ``{gt_label: model_class}`` alignment. When provided,
        GT boxes whose label is not in the map are dropped (the dataset carries
        classes the model never predicts), and the rest are relabelled to the
        model class so match/mismatch compares like with like.
    """
    reviews = reviews or {}
    gt_items = list(gt_items or [])
    if class_map:
        mapped = []
        for g in gt_items:
            m = class_map.get(g.get("cls"))
            if m is None:
                continue  # GT class the model does not predict -> ignore
            mapped.append({**g, "cls": m})
        gt_items = mapped
    has_gt_layer = len(gt_items) > 0

    # Flatten predictions with stable ids. The id embeds the FULL image_id so a
    # review verdict can be resolved back to its image (and dataset) without a
    # prefix collision: ``<image_id>_<model_id>_<idx>`` (model_id/image_id never
    # contain '_'). See detection_image_id() for the inverse.
    preds: list[tuple[str, dict]] = []
    for model_id, payload in preds_by_model.items():
        for idx, pred in enumerate(extract_pred_detections(model_id, payload, img_w, img_h, name_map=name_map)):
            preds.append((f"{image_id}_{model_id}_{idx}", pred))

    detections: list[dict] = []
    matched_any: set[int] = set()
    # GT is matched independently PER MODEL: when two models run on one dataset,
    # each should be scored against the full GT, so a box both models find is a
    # match for both (not a false positive for the second). A GT box is a miss
    # only when no model matched it.
    matched_by_model: dict[str, set[int]] = {}

    for det_id, pred in preds:
        review = reviews.get(det_id)
        pred_box = pred.get("box")
        model = pred.get("model", "")

        # 1) explicit human verdicts win
        if review and review.get("verdict") == "rejected":
            detections.append(_detection(det_id, pred, None, "fp"))
            continue
        if review and review.get("verdict") == "edited" and review.get("box"):
            gt_box = review["box"]
            state = "match" if iou_xywh(gt_box, pred_box) >= iou_thr else "mismatch"
            detections.append(_detection(det_id, pred, gt_box, state))
            continue

        # 2) match against the imported GT layer (per-model consumption)
        if has_gt_layer:
            used = matched_by_model.setdefault(model, set())
            best_j, best_iou = -1, 0.0
            for j, g in enumerate(gt_items):
                if j in used:
                    continue
                v = iou_xywh(g.get("box"), pred_box)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= iou_thr:
                used.add(best_j)
                matched_any.add(best_j)
                g = gt_items[best_j]
                same_cls = g.get("cls") in (None, pred.get("cls"))
                same_id = _identity_matches(g, pred)
                state = "match" if (same_cls and same_id) else "mismatch"
                detections.append(_detection(det_id, pred, g.get("box"), state, gt_item=g))
            else:
                detections.append(_detection(det_id, pred, None, "fp"))
            continue

        # 3) no GT layer: predictions are provisional ground truth (accepted or
        #    not-yet-reviewed both render as a match the human can later reject).
        detections.append(_detection(det_id, pred, pred_box, "match"))

    # 4) any GT box no model matched is a false negative (miss)
    if has_gt_layer:
        for j, g in enumerate(gt_items):
            if j in matched_any:
                continue
            miss = {"model": g.get("model", ""), "cls": g.get("cls", "object"), "box": None, "conf": 0.0}
            if g.get("identity"):
                miss["identity"] = {"pred": None}
            if g.get("decoded"):
                miss["decoded"] = {"pred": None}
            detections.append(_detection(f"{image_id}_gt_{j}", miss, g.get("box"), "miss", gt_item=g))

    return detections


def committed_gt(
    image_id: str,
    preds_by_model: dict[str, dict],
    img_w: int,
    img_h: int,
    *,
    gt_items: Optional[list[dict]] = None,
    reviews: Optional[dict[str, dict]] = None,
    class_map: Optional[dict[str, str]] = None,
    iou_thr: float = DEFAULT_IOU_THR,
) -> list[dict]:
    """Human-verified ground truth for curation write-back (Path 1), expressed in
    the **label file's** class vocabulary as ``[{cls, box:[xywh]}]``.

    The human curates *predicted* boxes; verdicts live in ``reviews`` keyed by the
    fusion detection id (``<image_id>_<model_id>_<idx>``):

    - ``rejected`` prediction -> not GT; if it overlapped an imported GT box, that
      box is dropped too (the human overrode it);
    - ``edited`` prediction -> its corrected box becomes GT;
    - ``accepted`` / IoU-matched prediction -> becomes GT, keeping the imported GT
      label when it matches one, else the model class mapped back through the
      class-map (its inverse);
    - imported GT boxes that no prediction touched are kept verbatim.

    When there is no imported GT layer, every non-rejected prediction is written
    (the provisional auto-label flow).
    """
    gt_items = list(gt_items or [])
    reviews = reviews or {}
    inv: dict[str, str] = {}
    for gt_label, model_cls in (class_map or {}).items():
        inv.setdefault(model_cls, gt_label)  # first GT label wins for a model class

    preds: list[tuple[str, dict]] = []
    for model_id, payload in preds_by_model.items():
        for idx, p in enumerate(extract_pred_detections(model_id, payload, img_w, img_h)):
            preds.append((f"{image_id}_{model_id}_{idx}", p))

    matched: set[int] = set()
    dropped: set[int] = set()

    def _best(box: Optional[Box]) -> int:
        if not box:
            return -1
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gt_items):
            if j in matched or j in dropped:
                continue
            v = iou_xywh(g.get("box"), box)
            if v >= best_iou:
                best_iou, best_j = v, j
        return best_j

    verified: list[dict] = []
    for det_id, p in preds:
        rv = reviews.get(det_id) or {}
        verdict = rv.get("verdict")
        pbox = p.get("box")
        if verdict == "rejected":
            j = _best(pbox)
            if j >= 0:
                dropped.add(j)
            continue
        box = rv["box"] if (verdict == "edited" and rv.get("box")) else pbox
        if not box:
            continue
        j = _best(box)
        if j >= 0:
            matched.add(j)
            verified.append({"cls": gt_items[j].get("cls"), "box": box})
        elif verdict == "accepted" or not gt_items:
            model_cls = p.get("cls")
            verified.append({"cls": inv.get(model_cls, model_cls), "box": box})

    for j, g in enumerate(gt_items):
        if j in matched or j in dropped:
            continue
        verified.append({"cls": g.get("cls"), "box": g.get("box")})

    return verified


def detection_image_id(target_id: str) -> Optional[str]:
    """Inverse of the det-id scheme: recover the ``image_id`` from a detection
    or flip target id. Returns None for ids that are not detection-shaped.

    ``<image_id>_<model_id>_<idx>`` -> ``image_id`` (also handles a leading
    ``flip:`` prefix). ``img:<image_id>`` is handled by the caller.
    """
    tid = target_id.split(":", 1)[1] if target_id.startswith("flip:") else target_id
    parts = tid.rsplit("_", 2)
    if len(parts) == 3 and parts[0]:
        return parts[0]
    return None


# ---- aggregates (match design_reference/app/data.js exactly) ---------------
def image_rollup(objects: list[dict]) -> dict:
    """Per-image agreement/conflicts, matching data.js makeImage()."""
    gt_count = sum(1 for o in objects if o.get("gt"))
    fp_count = sum(1 for o in objects if o.get("state") == "fp")
    matched = sum(1 for o in objects if o.get("state") == "match")
    conflicts = sum(1 for o in objects if o.get("state") in ("fp", "miss", "mismatch"))
    agreement = 1.0 if (gt_count + fp_count) == 0 else matched / max(1, matched + conflicts)
    return {"agreement": round(agreement, 3), "conflicts": conflicts, "matched": matched}


def dataset_rollup(image_rollups: list[dict], reviewed: int = 0) -> dict:
    """Dataset-level rollup, matching data.js dataset aggregates."""
    n = len(image_rollups)
    if n == 0:
        return {"agreement": 0.0, "conflicts": 0, "reviewed": reviewed, "count": 0}
    agreement = sum(r["agreement"] for r in image_rollups) / n
    conflicts = sum(r["conflicts"] for r in image_rollups)
    return {"agreement": round(agreement, 3), "conflicts": conflicts, "reviewed": reviewed, "count": n}
