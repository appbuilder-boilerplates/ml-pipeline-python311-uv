import multiprocessing
import os
import sys
import threading
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


# The test process exits once pytest has finished -- results printed, reports written -- even when a test left a
# non-daemon thread or a child process running (Python would otherwise wait for it at exit, forever). What is still
# running is named on stderr so it can be fixed.
_session_exit_status = {"value": 0}


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _session_exit_status["value"] = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    threads = [t for t in threading.enumerate() if t is not threading.main_thread() and t.is_alive() and not t.daemon]
    children = [p for p in multiprocessing.active_children() if not p.daemon]
    if not threads and not children:
        return
    names = [f"thread {t.name}" for t in threads] + [f"process {p.name} (pid {p.pid})" for p in children]
    sys.stderr.write(
        f"pytest: the tests left {len(names)} running after they finished: {', '.join(names)}. "
        "Exiting anyway; stop or join them in the test (or a fixture) that started them.\n"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_session_exit_status["value"])
