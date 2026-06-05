"""Label-lint health check — returned by ``POST /datasets:import``.

Validates an imported ground-truth set *before* it is trusted. Unlike the
handoff stub (which returned all-zero counts), this computes each check by
scanning the parsed annotations from :mod:`sensveridian.api.importers`.

``sev`` in ``ok|warn|error|info``. Errors are auto-skipped on import; the score
gates it: ``score = 100 - 4*errors - int(0.4*warns)`` (matches the scaffold).

Pure module so it is unit-testable on in-memory annotation lists.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

from .fusion import iou_xywh

_MIN_PX = 8
_DUP_IOU = 0.95


def _class_balance_note(counts: Counter) -> str:
    total = sum(counts.values())
    if not total:
        return "no labels"
    top = counts.most_common(3)
    parts = [f"{label} {round(100 * n / total)}%" for label, n in top]
    other = total - sum(n for _, n in top)
    if other > 0:
        parts.append(f"other {round(100 * other / total)}%")
    return " · ".join(parts)


def _vision_checks(annotations: list[dict], image_refs: Optional[Iterable[str]]) -> list[dict]:
    boxes = len(annotations)
    images = len({a.get("image") for a in annotations})

    off_image = 0
    sub_px = 0
    by_group: dict[tuple, list[list[float]]] = {}
    class_counts: Counter = Counter()
    for a in annotations:
        x, y, w, h = a.get("box", [0, 0, 0, 0])
        if x < -1e-6 or y < -1e-6 or (x + w) > 1 + 1e-6 or (y + h) > 1 + 1e-6:
            off_image += 1
        iw, ih = a.get("img_w"), a.get("img_h")
        if iw and ih and (w * iw < _MIN_PX or h * ih < _MIN_PX):
            sub_px += 1
        by_group.setdefault((a.get("image"), a.get("cls")), []).append([x, y, w, h])
        class_counts[a.get("cls")] += 1

    duplicates = 0
    for group in by_group.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if iou_xywh(group[i], group[j]) > _DUP_IOU:
                    duplicates += 1
                    break

    unmatched = 0
    if image_refs is not None:
        refs = set(image_refs)
        unmatched = sum(1 for a in annotations if a.get("image") not in refs)

    return [
        {"sev": "ok", "label": "Boxes parsed", "n": boxes, "note": f"across {images} images"},
        {"sev": "warn", "label": "Boxes clipped to image bounds", "n": off_image, "note": "coords outside [0,1]"},
        {"sev": "warn", "label": "Duplicate / overlapping (IoU > .95)", "n": duplicates, "note": "likely double-labelled"},
        {"sev": "warn", "label": "Sub-8px boxes", "n": sub_px, "note": "below trainable size"},
        {"sev": "error", "label": "Label files without an image", "n": unmatched, "note": "unmatched — will skip"},
        {"sev": "info", "label": "Class imbalance", "n": None, "note": _class_balance_note(class_counts)},
    ]


def _audio_checks(segments: list[dict]) -> list[dict]:
    parsed = len(segments)
    zero_len = sum(1 for s in segments if float(s.get("end", 0)) == float(s.get("start", 0)))
    out_of_range = 0
    for s in segments:
        dur = s.get("dur")
        start, end = float(s.get("start", 0)), float(s.get("end", 0))
        if start < 0 or (dur is not None and end > float(dur) + 1e-6):
            out_of_range += 1

    overlaps = 0
    by_clip: dict[str, list[tuple[float, float]]] = {}
    label_counts: Counter = Counter()
    for s in segments:
        by_clip.setdefault(s.get("clip"), []).append((float(s.get("start", 0)), float(s.get("end", 0))))
        label_counts[s.get("label")] += 1
    for spans in by_clip.values():
        spans.sort()
        for i in range(1, len(spans)):
            if spans[i][0] < spans[i - 1][1] - 1e-9:
                overlaps += 1

    return [
        {"sev": "ok", "label": "Segments parsed", "n": parsed, "note": "across imported clips"},
        {"sev": "warn", "label": "Overlapping segments", "n": overlaps, "note": "same track, time overlap > 0"},
        {"sev": "warn", "label": "Zero-length segments", "n": zero_len, "note": "start == end"},
        {"sev": "error", "label": "Out-of-range timestamps", "n": out_of_range, "note": "end beyond clip duration"},
        {"sev": "info", "label": "Label balance", "n": None, "note": _class_balance_note(label_counts)},
    ]


def run_checks(spec, annotations: Optional[list[dict]] = None,
               image_refs: Optional[Iterable[str]] = None,
               segments: Optional[list[dict]] = None) -> dict:
    """Return ``{"score", "images", "boxes", "checks"[]}``.

    ``spec`` is the :class:`~sensveridian.api.schemas.ImportSpec` (only ``kind``
    is read). Pass the parsed ``annotations`` (vision) or ``segments`` (audio);
    when omitted the structural checks return with zero counts.
    """
    kind = getattr(spec, "kind", "vision")
    if kind == "audio":
        checks = _audio_checks(segments or [])
        n_images, n_boxes = 0, 0
    else:
        anns = annotations or []
        checks = _vision_checks(anns, image_refs)
        n_images = len({a.get("image") for a in anns})
        n_boxes = len(anns)

    errs = sum((c["n"] or 0) for c in checks if c["sev"] == "error")
    warns = sum((c["n"] or 0) for c in checks if c["sev"] == "warn")
    score = max(0, 100 - errs * 4 - int(warns * 0.4))
    return {"score": score, "images": n_images, "boxes": n_boxes, "checks": checks}
