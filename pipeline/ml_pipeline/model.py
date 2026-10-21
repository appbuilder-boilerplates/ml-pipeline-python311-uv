"""Training and prediction. The model library is imported where it is used (never at import time), so importing the
package is cheap and the tests can stub it."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from .data import Dataset

Task = Literal["classification", "regression"]
DEFAULT_PARAMS: dict[str, Any] = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}
MODEL_FILE = "model.joblib"


@dataclass
class Model:
    task: Task
    estimator: Any
    feature_columns: list[str]
    # Classification: the file's label values; the estimator predicts their index.
    classes: list[Any]


def _task(params: dict[str, Any]) -> Task:
    task = params.get("task", "classification")
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', not {task!r}")
    return cast(Task, task)


def hyperparameters(params: dict[str, Any]) -> dict[str, Any]:
    """The estimator's arguments: the defaults overridden by `params` (without `task`)."""
    return {**DEFAULT_PARAMS, **{key: value for key, value in params.items() if key != "task"}}


def encode_features(features: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Numeric features: text columns one-hot encoded. With `columns` (the training columns), aligned to them --
    a category unseen in training is dropped and a missing one is 0."""
    encoded = pd.get_dummies(features, dtype=float)
    if columns is not None:
        encoded = encoded.reindex(columns=columns, fill_value=0.0)
    return encoded.astype(float)


def train(params: dict[str, Any], data: Dataset) -> Model:
    task = _task(params)
    features = encode_features(data["features"])
    if task == "regression":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(**hyperparameters(params))
        estimator.fit(features, np.asarray(data["labels"], dtype=float))
        return Model(task=task, estimator=estimator, feature_columns=list(features.columns), classes=[])
    from xgboost import XGBClassifier

    # A stable order whatever types the file's labels have.
    classes = sorted(set(data["labels"]), key=lambda label: (type(label).__name__, str(label)))
    if len(classes) < 2:
        raise ValueError(f"classification needs at least two label values in the training data, got {classes}")
    index = {label: position for position, label in enumerate(classes)}
    estimator = XGBClassifier(**hyperparameters(params))
    estimator.fit(features, np.asarray([index[label] for label in data["labels"]]))
    return Model(task=task, estimator=estimator, feature_columns=list(features.columns), classes=classes)


def predict(model: Model, data: Dataset) -> list[Any]:
    """One prediction per row of `data`, in the file's own label values (classification) or as floats."""
    raw = model.estimator.predict(encode_features(data["features"], model.feature_columns))
    if model.task == "regression":
        return [float(value) for value in raw]
    return [model.classes[int(value)] for value in raw]


def save_model(model: Model, directory: str | Path) -> str:
    import joblib

    path = Path(directory) / MODEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return str(path)


def load_model(path: str | Path) -> Model:
    import joblib

    model = joblib.load(path)
    if not isinstance(model, Model):
        raise TypeError(f"{path} does not hold a {Model.__name__}")
    return model


def describe(params: dict[str, Any]) -> dict[str, Any]:
    task = _task(params)
    return {
        "model": {"name": "xgboost", "type": "XGBRegressor" if task == "regression" else "XGBClassifier"},
        "hyperparameters": hyperparameters(params),
        "configuration": {"task": task},
    }
