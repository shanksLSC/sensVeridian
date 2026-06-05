"""Label-set import parsers — COCO / YOLO / CSV / VOC -> normalized annotations.

Each parser returns a list of annotation dicts of the form::

    {"image": <ref>, "cls": <label>, "box": [x, y, w, h], "img_w": W?, "img_h": H?}

where ``box`` is normalized ``[x, y, w, h]`` with a top-left origin. Boxes are
**not** clamped here so the label-lint can flag out-of-bounds annotations; the
store clamps on insert.

Pure module (stdlib only, incl. ``xml.etree``) so it is directly unit-testable
on in-memory structures without touching the filesystem.
"""
from __future__ import annotations

import csv as _csv
import io
import xml.etree.ElementTree as ET
from typing import Any, Optional

Ann = dict


# ---- COCO ------------------------------------------------------------------
def parse_coco(coco: dict) -> list[Ann]:
    """COCO detection JSON: images[], annotations[] (bbox [x,y,w,h] abs px),
    categories[]."""
    images = {im["id"]: im for im in coco.get("images", [])}
    cats = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}
    out: list[Ann] = []
    for a in coco.get("annotations", []):
        im = images.get(a.get("image_id"))
        if not im:
            continue
        W, H = float(im.get("width") or 0), float(im.get("height") or 0)
        x, y, w, h = (float(v) for v in (a.get("bbox") or [0, 0, 0, 0])[:4])
        if W <= 0 or H <= 0:
            continue
        out.append(
            {
                "image": im.get("file_name") or str(im["id"]),
                "cls": cats.get(a.get("category_id"), str(a.get("category_id"))),
                "box": [x / W, y / H, w / W, h / H],
                "img_w": int(W),
                "img_h": int(H),
            }
        )
    return out


# ---- YOLO ------------------------------------------------------------------
def parse_yolo(
    items: list[tuple[str, str]],
    names: Optional[list[str]] = None,
    dims: Optional[dict[str, tuple[int, int]]] = None,
) -> list[Ann]:
    """YOLO txt labels (already normalized, center-based ``cls cx cy w h``).

    ``items`` is ``[(image_ref, label_file_text), ...]``; ``names`` maps class
    index -> label; ``dims`` optionally maps image_ref -> (w, h) for sub-8px
    checks downstream.
    """
    names = names or []
    out: list[Ann] = []
    for image_ref, body in items:
        for line in body.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_i = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
            label = names[cls_i] if 0 <= cls_i < len(names) else f"class_{cls_i}"
            ann: Ann = {"image": image_ref, "cls": label, "box": [cx - w / 2, cy - h / 2, w, h]}
            if dims and image_ref in dims:
                ann["img_w"], ann["img_h"] = dims[image_ref]
            out.append(ann)
    return out


# ---- CSV -------------------------------------------------------------------
def parse_csv(text: str, mapping: dict[str, str],
              dims: Optional[dict[str, tuple[int, int]]] = None) -> list[Ann]:
    """Flat CSV with a column ``mapping`` (image, x1, y1, x2, y2, cls, conf...).

    Coordinates are treated as normalized when every value is <= 1, otherwise as
    absolute pixels (requiring ``dims`` for the image, or width/height columns
    named via the mapping).
    """
    m = mapping or {}
    img_c = m.get("image", "image")
    x1c, y1c, x2c, y2c = m.get("x1", "x1"), m.get("y1", "y1"), m.get("x2", "x2"), m.get("y2", "y2")
    cls_c = m.get("cls", "cls")
    wc, hc = m.get("width", "width"), m.get("height", "height")
    out: list[Ann] = []
    for row in _csv.DictReader(io.StringIO(text)):
        try:
            x1, y1, x2, y2 = float(row[x1c]), float(row[y1c]), float(row[x2c]), float(row[y2c])
        except (KeyError, ValueError, TypeError):
            continue
        image_ref = row.get(img_c, "")
        W = H = None
        if dims and image_ref in dims:
            W, H = dims[image_ref]
        elif wc in row and hc in row and row[wc] and row[hc]:
            W, H = float(row[wc]), float(row[hc])
        normalized = max(x1, y1, x2, y2) <= 1.5
        if not normalized and W and H:
            x1, x2 = x1 / W, x2 / W
            y1, y2 = y1 / H, y2 / H
        ann: Ann = {"image": image_ref, "cls": row.get(cls_c, ""),
                    "box": [x1, y1, x2 - x1, y2 - y1]}
        if W and H:
            ann["img_w"], ann["img_h"] = int(W), int(H)
        out.append(ann)
    return out


# ---- VOC -------------------------------------------------------------------
def parse_voc(xml_text: str) -> list[Ann]:
    """Pascal VOC XML for a single image (<size>, repeated <object>)."""
    root = ET.fromstring(xml_text)
    fname_el = root.find("filename")
    fname = fname_el.text if fname_el is not None else ""
    size = root.find("size")
    W = float(size.findtext("width", "0")) if size is not None else 0.0
    H = float(size.findtext("height", "0")) if size is not None else 0.0
    out: list[Ann] = []
    if W <= 0 or H <= 0:
        return out
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin", "0"))
        ymin = float(bb.findtext("ymin", "0"))
        xmax = float(bb.findtext("xmax", "0"))
        ymax = float(bb.findtext("ymax", "0"))
        out.append(
            {
                "image": fname,
                "cls": obj.findtext("name", ""),
                "box": [xmin / W, ymin / H, (xmax - xmin) / W, (ymax - ymin) / H],
                "img_w": int(W),
                "img_h": int(H),
            }
        )
    return out


# ---- dispatcher ------------------------------------------------------------
def parse_labels(fmt: str, data: Any, mapping: Optional[dict] = None, **kw: Any) -> list[Ann]:
    """Dispatch to the right parser by format name.

    ``data`` shape depends on ``fmt``: a dict for ``coco``; a list of
    ``(ref, text)`` for ``yolo``; a CSV string for ``csv``; an XML string (or
    list of XML strings) for ``voc``.
    """
    fmt = (fmt or "").lower()
    if fmt == "coco":
        return parse_coco(data)
    if fmt == "yolo":
        return parse_yolo(data, names=kw.get("names"), dims=kw.get("dims"))
    if fmt == "csv":
        return parse_csv(data, mapping or {}, dims=kw.get("dims"))
    if fmt == "voc":
        if isinstance(data, (list, tuple)):
            out: list[Ann] = []
            for xml_text in data:
                out.extend(parse_voc(xml_text))
            return out
        return parse_voc(data)
    raise ValueError(f"unsupported import format: {fmt!r}")
