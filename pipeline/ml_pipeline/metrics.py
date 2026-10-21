"""The scores `main` reports. Plain numpy: the eval platform scores official runs with its own metrics."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def score(task: str, labels: Sequence[Any], predictions: Sequence[Any]) -> dict[str, float]:
    if len(labels) != len(predictions):
        raise ValueError(f"{len(labels)} labels but {len(predictions)} predictions")
    if not labels:
        raise ValueError("nothing to score: no rows")
    if task == "regression":
        errors = np.asarray(predictions, dtype=float) - np.asarray(labels, dtype=float)
        return {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2)))}
    correct = sum(1 for label, prediction in zip(labels, predictions, strict=True) if label == prediction)
    return {"accuracy": correct / len(labels)}
