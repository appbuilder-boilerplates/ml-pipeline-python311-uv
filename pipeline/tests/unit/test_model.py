from pathlib import Path

import pandas as pd
import pytest

from ml_pipeline import describe, predict, save_model, train
from ml_pipeline.data import Dataset
from ml_pipeline.model import encode_features, hyperparameters, load_model


def _data(labels: list[object], colours: list[str]) -> Dataset:
    return {
        "labels": labels,
        "ids": [str(i) for i in range(len(labels))],
        "features": pd.DataFrame({"size": [float(i) for i in range(len(labels))], "colour": colours}),
    }


def test_encode_features_aligns_unseen_and_missing_categories_to_training_columns() -> None:
    training = encode_features(pd.DataFrame({"size": [1.0, 2.0], "colour": ["red", "blue"]}))
    later = encode_features(pd.DataFrame({"size": [3.0], "colour": ["green"]}), list(training.columns))
    assert list(later.columns) == list(training.columns)
    assert later.iloc[0].to_dict() == {"size": 3.0, "colour_blue": 0.0, "colour_red": 0.0}


def test_classification_predicts_in_the_files_own_label_values(stub_xgboost: None) -> None:
    model = train({}, _data(["no", "yes", "yes"], ["red", "blue", "red"]))
    assert model.classes == ["no", "yes"]
    assert predict(model, _data(["no", "no"], ["green", "red"])) == ["yes", "yes"]


def test_classification_refuses_a_single_label_value(stub_xgboost: None) -> None:
    with pytest.raises(ValueError, match="at least two"):
        train({}, _data(["a", "a"], ["red", "blue"]))


def test_regression_predicts_floats(stub_xgboost: None) -> None:
    model = train({"task": "regression"}, _data([1, 3], ["red", "blue"]))
    assert predict(model, _data([0], ["red"])) == [2.0]


def test_an_unknown_task_is_refused(stub_xgboost: None) -> None:
    with pytest.raises(ValueError, match="task must be"):
        train({"task": "ranking"}, _data(["a", "b"], ["red", "blue"]))


def test_params_override_defaults_and_task_is_not_an_estimator_argument(stub_xgboost: None) -> None:
    model = train({"task": "classification", "max_depth": 3}, _data(["a", "b"], ["red", "blue"]))
    assert model.estimator.params == {**hyperparameters({}), "max_depth": 3}
    assert "task" not in model.estimator.params


def test_save_model_round_trips(stub_xgboost: None, tmp_path: Path) -> None:
    model = train({}, _data(["a", "b", "b"], ["red", "blue", "red"]))
    path = save_model(model, tmp_path / "artifacts")
    assert predict(load_model(path), _data(["a"], ["blue"])) == ["b"]


def test_describe_names_the_estimator_for_the_task() -> None:
    assert describe({"task": "regression", "n_estimators": 10})["model"]["type"] == "XGBRegressor"
    assert describe({})["hyperparameters"] == hyperparameters({})
