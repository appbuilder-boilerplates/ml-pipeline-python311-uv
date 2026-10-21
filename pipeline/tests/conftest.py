import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests import fake_xgboost


@pytest.fixture
def stub_xgboost(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """xgboost replaced by tests/fake_xgboost.py for one test (training imports it when it runs)."""
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgboost)
    yield


@pytest.fixture
def write_csv(tmp_path: Path) -> Callable[[str, str], Path]:
    def write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.write_text(text)
        return path

    return write
