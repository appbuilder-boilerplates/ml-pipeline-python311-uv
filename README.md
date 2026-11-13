# ML pipeline (Python 3.11 + uv, XGBoost CPU) — vetted app builder boilerplate

Laid out like a generated project: the component lives in `pipeline/`. Reuse copies only `pipeline/` (catalogue
`component_path: pipeline`). No server, no preview, no production image: the pipeline's runs go to Modal through an eval
bundle the platform scaffolds (`evals/<feature>/<eval>/`).

## Stack
- Python **3.11** — the eval bundle's Modal image is `python:3.11-slim` and imports this code from the repository, so
  the code must run there (no 3.12+ syntax).
- Runtime packages: numpy 2.4, pandas 3.0, scikit-learn 1.9, **xgboost-cpu** 3.2 (plain `xgboost` pulls the CUDA wheels
  the platform's CPU-only check refuses), joblib 1.6.
- Dependencies: **uv** with `pyproject.toml` + `uv.lock`; the test tools are the `test` dependency group.
- Tests: pytest 9. Lint: ruff. Types: mypy (`disallow_untyped_defs`).

## The package (`pipeline/ml_pipeline/`)
Its public functions have the eval adapter's shape, so an eval's `adapter.py` only binds them:

| Function | What it does |
|---|---|
| `load_data(path, *, label_column="label", id_column="id")` | A CSV/TSV/JSONL file → `{"labels", "ids", "features"}`: labels exactly as the file holds them, ids as unique text |
| `train(params, data)` | XGBoost classifier (default) or regressor (`params["task"] = "regression"`); other params override `DEFAULT_PARAMS`; text features one-hot encoded; labels encoded inside |
| `predict(model, data)` | One prediction per row, in the file's own label values (classification) or floats |
| `save_model(model, directory)` / `load_model(path)` | joblib, `model.joblib` |
| `describe(params)` | `{"model", "hyperparameters", "configuration"}` |
| `pipeline.main()` | Train on `$TRAIN_FILE`, score `$TEST_FILE`, print `METRIC_<name>=<value>` lines (the platform's training runner reads these) |

Nothing heavy is imported at import time (xgboost and joblib are imported where they are used), and no path is fixed:
the files come from the arguments or the environment (`TRAIN_FILE`, `TEST_FILE`, `LABEL_COLUMN`, `ID_COLUMN`,
`MODEL_PARAMS_JSON`, `MODEL_DIR`). Data goes under `pipeline/data_sources/` (not committed).

Imports inside the package are relative, so it imports both as `ml_pipeline` (from `pipeline/`) and as
`pipeline.ml_pipeline` (from the repository root, which the eval's Modal run puts on `PYTHONPATH`).

### An eval's adapter
```python
from typing import Any

from pipeline.ml_pipeline import Dataset, describe, predict, save_model, train
from pipeline.ml_pipeline import load_data as _load_data

__all__ = ["describe", "load_data", "predict", "save_model", "train"]

LABEL_COLUMN = "label"
ID_COLUMN = "id"


def load_data(path: Any) -> Dataset:
    return _load_data(path, label_column=LABEL_COLUMN, id_column=ID_COLUMN)
```
The bundle's `requirements.txt` is this component's runtime dependencies (`[project].dependencies`, each with its
range); its `requirements-test.txt` needs only `numpy` and `pandas` (what importing the package needs).

## Contracts this boilerplate satisfies
- `test_command`: `export PATH=/deps/venv/bin:$PATH; ruff check . ; mypy ml_pipeline tests ; pytest tests` (with
  report placeholders).
- One `pipeline/Dockerfile`, two targets (R54.36):
  - `deps`: `uv sync --no-install-project --no-dev` into `/deps/venv` — what the pipeline runs with.
  - `test` (from a slim base, not `deps`): `uv sync --only-group test` — numpy, pandas, joblib and the test tools;
    no scikit-learn, scipy or xgboost. uv is removed again after the sync.

## Tests
- `tests/fake_xgboost.py` stands in for xgboost (majority class / mean label), installed per test by the
  `stub_xgboost` fixture — the tests check the pipeline's own encoding, label mapping and wiring, not the library.
- `tests/unit/` — data loading, the model wrapper, the metrics.
- `tests/integration/test_pipeline.py` — files → train → predict → score → `METRIC_` lines.

## Gotchas this boilerplate already handles
- `xgboost-cpu`, never `xgboost`: the CUDA wheels (nvidia-nccl-cu12) are hundreds of MB and the CPU-only check
  refuses them.
- Classification labels are mapped to indices inside `train` and back inside `predict`: the platform's label check
  compares `load_data`'s labels with the file's column, so they must never be re-encoded in `load_data`.
- Prediction features are aligned to the training columns: a category unseen in training is dropped, a missing one is
  0.
- `UV_PROJECT_ENVIRONMENT=/deps/venv` keeps the venv outside `/workspace` (hidden at runtime by the PVC mount).

## Logging

stdout is the result channel: only the `METRIC_<name>=<value>` lines the platform's training runner reads. Every
operational log record goes to stderr as one JSON object per line (`pipeline/ml_pipeline/logging_config.py`):
`timestamp`, `level`, `logger`, `message` and the fields passed with it (`data loaded`, `model trained`, `model scored`
stage records). Log with `logging.getLogger(__name__)` and `extra={...}`; never print anything but the METRIC lines.
