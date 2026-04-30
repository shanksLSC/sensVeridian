from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Any
import numpy as np


@dataclass
class Summary:
    present: bool
    count: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerOutput:
    summary: Summary
    raw: dict[str, Any]


class Runner(Protocol):
    model_id: str
    display_name: str
    version: str
    weights_path: str
    depends_on: tuple[str, ...]

    def load(self) -> None:
        ...

    def predict(self, image_bgr: np.ndarray, deps: dict[str, RunnerOutput]) -> RunnerOutput:
        ...


def set_conf_threshold(runner: Any, conf_threshold: float | None) -> bool:
    if conf_threshold is None:
        return False
    if not hasattr(runner, "conf_threshold"):
        return False
    setattr(runner, "conf_threshold", float(conf_threshold))
    return True

