"""A stand-in for xgboost (not installed in the test image): the classifier predicts the most common training class,
the regressor the mean training label. Enough to test the pipeline's own encoding and wiring, not the library."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class XGBClassifier:
    def __init__(self, **params: Any) -> None:
        self.params = params
        self.fitted_columns: list[str] = []
        self.majority = 0

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> XGBClassifier:
        self.fitted_columns = list(features.columns)
        values, counts = np.unique(labels, return_counts=True)
        self.majority = int(values[np.argmax(counts)])
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        assert list(features.columns) == self.fitted_columns, "predict got columns other than training's"
        return np.full(len(features), self.majority)


class XGBRegressor:
    def __init__(self, **params: Any) -> None:
        self.params = params
        self.mean = 0.0

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> XGBRegressor:
        self.mean = float(np.mean(labels))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.mean)
