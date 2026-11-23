"""Logging for the pipeline: one JSON object per line on stderr.

stdout is the pipeline's result channel -- its `METRIC_<name>=<value>` lines, which the platform's training runner
reads -- so it carries nothing else; every operational log record goes to stderr as JSON (`timestamp`, `level`,
`logger`, `message` and every field passed with it). `main()` calls `configure_logging()` once at startup; modules log
with `logging.getLogger(__name__)` and pass context as `extra={...}` -- never print, except the METRIC lines.
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import UTC, datetime
from typing import TextIO

_STANDARD_ATTRIBUTES = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRIBUTES and not key.startswith("_"):
                entry[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc_type"] = record.exc_info[0].__name__
            entry["exc_info"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(entry, default=str)


class _StderrHandler(logging.StreamHandler):
    """Writes to the current sys.stderr (resolved per record, so a replaced stderr is followed)."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self) -> TextIO:  # noqa: D401 -- logging's own attribute
        return sys.stderr

    @stream.setter
    def stream(self, _value: TextIO) -> None:
        pass


def configure_logging(level: str = "INFO") -> None:
    """One JSON stderr handler on the root logger at `level`. Idempotent: calling it again replaces the handler."""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, "_pipeline_json", False)]:
        root.removeHandler(handler)
    handler = _StderrHandler()
    handler.setFormatter(JsonFormatter())
    handler._pipeline_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level.upper())
