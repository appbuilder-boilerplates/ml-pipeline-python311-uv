"""The pipeline's operational log records go to stderr as JSON (stdout is the METRIC_ result channel the platform
reads, and stays exactly that)."""
import json
import logging

import pytest

from ml_pipeline.logging_config import configure_logging


def _records(captured: str) -> list[dict]:
    return [json.loads(line) for line in captured.splitlines() if line.strip()]


def test_a_record_is_one_json_line_on_stderr_with_its_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")

    logging.getLogger("ml_pipeline.example").info("model trained", extra={"rows": 3, "duration_ms": 5})

    captured = capsys.readouterr()
    [record] = [r for r in _records(captured.err) if r["message"] == "model trained"]
    assert (record["level"], record["logger"], record["rows"], record["duration_ms"]) == (
        "INFO", "ml_pipeline.example", 3, 5)
    assert captured.out == ""


def test_configuring_twice_writes_each_record_once(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    configure_logging("INFO")

    logging.getLogger("ml_pipeline.example").info("once")

    assert [r["message"] for r in _records(capsys.readouterr().err)].count("once") == 1
