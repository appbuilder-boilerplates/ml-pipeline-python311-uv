"""The pipeline end to end: train on one file, score on another, report each score as a `METRIC_<name>=<value>`
stdout line (the line format the platform's training runner reads). Configured only from the environment."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from .core.logging_config import configure_logging
from .data import load_data
from .metrics import score
from .model import predict, save_model, train

logger = logging.getLogger(__name__)


def run(train_file: str, test_file: str, params: dict[str, Any], *, label_column: str = "label",
        id_column: str = "id", model_dir: str | None = None) -> dict[str, float]:
    training = load_data(train_file, label_column=label_column, id_column=id_column)
    logger.info("data loaded", extra={"split": "train", "file": train_file, "rows": len(training["labels"])})
    testing = load_data(test_file, label_column=label_column, id_column=id_column)
    logger.info("data loaded", extra={"split": "test", "file": test_file, "rows": len(testing["labels"])})
    started = time.monotonic()
    model = train(params, training)
    logger.info("model trained", extra={"task": model.task,
                                        "duration_ms": round((time.monotonic() - started) * 1000, 1)})
    if model_dir:
        save_model(model, model_dir)
    metrics = score(model.task, testing["labels"], predict(model, testing))
    logger.info("model scored", extra={"metrics": metrics})
    return metrics


def _params() -> dict[str, Any]:
    raw = os.environ.get("MODEL_PARAMS_JSON", "{}")
    params = json.loads(raw)
    if not isinstance(params, dict):
        raise ValueError(f"MODEL_PARAMS_JSON must be a JSON object, not {raw!r}")
    return params


def main() -> None:
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    metrics = run(
        os.environ.get("TRAIN_FILE", "data_sources/train.csv"),
        os.environ.get("TEST_FILE", "data_sources/test.csv"),
        _params(),
        label_column=os.environ.get("LABEL_COLUMN", "label"),
        id_column=os.environ.get("ID_COLUMN", "id"),
        model_dir=os.environ.get("MODEL_DIR") or None,
    )
    for name, value in metrics.items():
        print(f"METRIC_{name}={value}")
