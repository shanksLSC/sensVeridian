"""Seamless-Redis decision logic for the ingest worker (no live Redis needed).

``use_arq()`` honors the ``VERIDIAN_USE_ARQ`` override, else auto-detects a live
arq worker via its Redis health-check key. ``redis_reachable()`` fails fast and
returns False when nothing is listening.
"""
from __future__ import annotations

from sensveridian.ingest import worker


def test_use_arq_env_override(monkeypatch):
    monkeypatch.setenv("VERIDIAN_USE_ARQ", "1")
    assert worker.use_arq() is True
    monkeypatch.setenv("VERIDIAN_USE_ARQ", "0")
    assert worker.use_arq() is False


def test_use_arq_autodetects_live_worker(monkeypatch):
    monkeypatch.delenv("VERIDIAN_USE_ARQ", raising=False)
    monkeypatch.setattr(worker, "arq_worker_alive", lambda *a, **k: True)
    assert worker.use_arq() is True
    monkeypatch.setattr(worker, "arq_worker_alive", lambda *a, **k: False)
    assert worker.use_arq() is False


def test_redis_reachable_false_when_nothing_listening():
    # nothing listens on this port -> connection refused -> fast False
    assert worker.redis_reachable("redis://127.0.0.1:65001/0") is False


def test_arq_worker_alive_false_when_nothing_listening():
    assert worker.arq_worker_alive("redis://127.0.0.1:65001/0") is False
