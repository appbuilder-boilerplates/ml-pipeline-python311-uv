import pytest

from ml_pipeline.metrics import score


def test_classification_scores_accuracy() -> None:
    assert score("classification", ["a", "b", "a", "a"], ["a", "b", "b", "a"]) == {"accuracy": 0.75}


def test_regression_scores_mae_and_rmse() -> None:
    assert score("regression", [1.0, 3.0], [2.0, 5.0]) == pytest.approx({"mae": 1.5, "rmse": (2.5) ** 0.5})


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="labels but"):
        score("classification", ["a"], [])


def test_no_rows_are_refused() -> None:
    with pytest.raises(ValueError, match="no rows"):
        score("classification", [], [])
