"""The pipeline end to end on real files: load -> encode -> train -> predict -> score -> METRIC lines."""
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from ml_pipeline.model import MODEL_FILE
from ml_pipeline.pipeline import main, run

TRAIN = "id,size,colour,label\n1,1.0,red,yes\n2,2.0,blue,yes\n3,3.0,red,no\n"
TEST = "id,size,colour,label\n4,1.5,green,yes\n5,2.5,red,no\n"


def test_run_trains_on_one_file_and_scores_the_other(
    stub_xgboost: None, write_csv: Callable[[str, str], Path], tmp_path: Path
) -> None:
    metrics = run(str(write_csv("train.csv", TRAIN)), str(write_csv("test.csv", TEST)), {},
                  model_dir=str(tmp_path / "model"))
    assert metrics == {"accuracy": 0.5}
    assert (tmp_path / "model" / MODEL_FILE).is_file()


def test_main_reports_each_score_as_a_metric_line(
    stub_xgboost: None,
    write_csv: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TRAIN_FILE", str(write_csv("train.csv", "id,size,colour,target\n1,1.0,red,1\n2,2.0,blue,3\n")))
    monkeypatch.setenv("TEST_FILE", str(write_csv("test.csv", "id,size,colour,target\n4,1.0,red,1\n5,2.0,red,3\n")))
    monkeypatch.setenv("LABEL_COLUMN", "target")
    monkeypatch.setenv("MODEL_PARAMS_JSON", '{"task": "regression"}')
    monkeypatch.delenv("MODEL_DIR", raising=False)
    main()
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["METRIC_mae=1.0", "METRIC_rmse=1.0"]  # the result channel, unchanged
    stages = [json.loads(line)["message"] for line in captured.err.splitlines() if line.startswith("{")]
    assert stages == ["data loaded", "data loaded", "model trained", "model scored"]


def test_main_refuses_params_that_are_not_an_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PARAMS_JSON", "[1]")
    with pytest.raises(ValueError, match="JSON object"):
        main()
