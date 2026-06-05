"""Tests for label-set import parsers (sensveridian.api.importers)."""
from __future__ import annotations

import pytest

from sensveridian.api import importers


def test_parse_coco():
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 1000, "height": 800}],
        "annotations": [{"image_id": 1, "category_id": 7, "bbox": [100, 80, 200, 160]}],
        "categories": [{"id": 7, "name": "car"}],
    }
    anns = importers.parse_coco(coco)
    assert anns == [
        {"image": "a.jpg", "cls": "car", "box": pytest.approx([0.1, 0.1, 0.2, 0.2]),
         "img_w": 1000, "img_h": 800}
    ]


def test_parse_yolo_center_to_topleft():
    anns = importers.parse_yolo([("a.jpg", "0 0.5 0.5 0.2 0.2\n")], names=["car"])
    assert anns[0]["cls"] == "car"
    assert anns[0]["box"] == pytest.approx([0.4, 0.4, 0.2, 0.2])


def test_parse_csv_absolute_with_dims():
    text = "image,x_min,y_min,x_max,y_max,label\na.jpg,100,80,300,240,car\n"
    mapping = {"image": "image", "x1": "x_min", "y1": "y_min", "x2": "x_max", "y2": "y_max", "cls": "label"}
    anns = importers.parse_csv(text, mapping, dims={"a.jpg": (1000, 800)})
    assert anns[0]["cls"] == "car"
    assert anns[0]["box"] == pytest.approx([0.1, 0.1, 0.2, 0.2])


def test_parse_voc():
    xml = (
        "<annotation><filename>a.jpg</filename>"
        "<size><width>1000</width><height>800</height></size>"
        "<object><name>car</name>"
        "<bndbox><xmin>100</xmin><ymin>80</ymin><xmax>300</xmax><ymax>240</ymax></bndbox>"
        "</object></annotation>"
    )
    anns = importers.parse_voc(xml)
    assert anns[0]["image"] == "a.jpg"
    assert anns[0]["box"] == pytest.approx([0.1, 0.1, 0.2, 0.2])


def test_dispatcher_and_unknown_format():
    assert importers.parse_labels("yolo", [("a.jpg", "1 0.5 0.5 0.2 0.2")], names=["car", "truck"])[0]["cls"] == "truck"
    with pytest.raises(ValueError):
        importers.parse_labels("nope", {})
