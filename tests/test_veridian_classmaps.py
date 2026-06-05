"""Tests for model/class metadata (sensveridian.api.classmaps)."""
from __future__ import annotations

from sensveridian.api import classmaps


def test_amod_class_names_cover_six_classes():
    assert classmaps.class_name("amod", 0) == "car"
    assert classmaps.class_name("amod", 2) == "person"
    assert classmaps.class_name("amod", 5) == "stop_sign"
    assert classmaps.n_classes("amod") == 6
    # out of range falls back
    assert classmaps.class_name("amod", 99) == "class_99"


def test_single_class_detectors():
    assert classmaps.class_name("qrcode") == "qr"
    assert classmaps.class_name("fd") == "face"
    assert classmaps.class_name("fr") == "face"


def test_aed_labels():
    assert classmaps.class_name("aed", 0) == "speech"
    assert classmaps.n_classes("aed") == 7


def test_model_card_shapes():
    card = classmaps.model_card("fr")
    assert card["short"] == "FR"
    assert card["depends_on"] == "fd"
    assert card["classes"] == 1
    # unknown model gets sensible fallbacks
    unknown = classmaps.model_card("zzz")
    assert unknown["short"] == "ZZZ"
    assert unknown["depends_on"] is None
