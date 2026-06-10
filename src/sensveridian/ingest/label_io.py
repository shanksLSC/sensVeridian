"""Read/write on-disk label files for the two ingestion paths.

- **Read** (both paths): parse a frame's sibling label file into normalized
  ground-truth boxes so they can be stored in ``gt_boxes`` and fused in the
  canvas. Reuses the parsers in :mod:`sensveridian.api.importers`.
- **Write** (curation path only): overwrite a frame's label file with verified
  boxes, in its native format, keeping a one-time ``.bak`` backup.

YOLO is the primary format (``<root>/images/<stem>.ext`` ↔ ``<root>/labels/
<stem>.txt`` with ``cls cx cy w h`` normalized + a ``dataset.yaml`` ``names``
map). COCO/VOC reading goes through the importers; write-back currently targets
YOLO (the layout the datasets use).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..api import importers

YOLO_EXT = ".txt"


# ---- discovery -------------------------------------------------------------
def labels_dir_for(image_path: Path) -> Optional[Path]:
    """Locate the YOLO labels directory for an image. Replaces an ``images``
    path component with ``labels`` (the standard layout); falls back to a
    sibling ``labels/`` directory."""
    parts = list(Path(image_path).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            return Path(*parts[:i], "labels", *parts[i + 1:-1])
    sibling = Path(image_path).parent.parent / "labels"
    return sibling if sibling.exists() else None


def find_dataset_yaml(image_path: Path) -> Optional[Path]:
    for p in [Path(image_path).parent, *Path(image_path).parents]:
        y = p / "dataset.yaml"
        if y.exists():
            return y
    return None


def load_class_names(yaml_path: Optional[Path]) -> dict[int, str]:
    if not yaml_path or not Path(yaml_path).exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {}


def detect_labels(sample_image: Path) -> Optional[dict]:
    """Detect the label set for a dataset from one of its images. Returns
    ``{labels_dir, format, class_names}`` or None when no labels are present."""
    ld = labels_dir_for(sample_image)
    if not ld or not Path(ld).exists():
        return None
    names = load_class_names(find_dataset_yaml(sample_image))
    return {"labels_dir": str(ld), "format": "yolo", "class_names": names}


def _names_list(class_names: dict[int, str]) -> Optional[list[str]]:
    if not class_names:
        return None
    return [class_names.get(i, f"class_{i}") for i in range(max(class_names) + 1)]


def label_file_for(image_path: Path, labels_dir: Path) -> Path:
    return Path(labels_dir) / (Path(image_path).stem + YOLO_EXT)


# ---- read ------------------------------------------------------------------
def read_image_labels(image_path: Path, labels_dir: Path, fmt: str,
                      class_names: dict[int, str]) -> list[dict]:
    """Parse one frame's labels into ``[{cls, box:[x,y,w,h] normalized}]``."""
    fmt = (fmt or "yolo").lower()
    if fmt == "yolo":
        lf = label_file_for(image_path, labels_dir)
        if not lf.exists():
            return []
        anns = importers.parse_yolo([(Path(image_path).name, lf.read_text(encoding="utf-8"))],
                                    names=_names_list(class_names))
        return [{"cls": a["cls"], "box": a["box"]} for a in anns]
    return []


# ---- write (curation path) -------------------------------------------------
def write_image_labels(image_path: Path, labels_dir: Path, fmt: str,
                       class_names: dict[int, str], boxes: list[dict]) -> Path:
    """Overwrite a frame's label file with ``boxes`` (each ``{cls, box:[x,y,w,h]
    normalized}``) in the dataset's native format, keeping a one-time ``.bak``.

    Returns the written label file path.
    """
    fmt = (fmt or "yolo").lower()
    if fmt != "yolo":
        raise ValueError(f"label write-back not implemented for format {fmt!r}")
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    lf = label_file_for(image_path, labels_dir)
    if lf.exists():
        bak = lf.with_suffix(lf.suffix + ".bak")
        if not bak.exists():
            bak.write_bytes(lf.read_bytes())  # one-time backup of the original

    name_to_idx = {v: k for k, v in (class_names or {}).items()}
    lines = []
    for b in boxes:
        x, y, w, h = b["box"]
        cx, cy = x + w / 2.0, y + h / 2.0
        cls_idx = name_to_idx.get(b.get("cls"), 0)
        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    lf.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return lf
