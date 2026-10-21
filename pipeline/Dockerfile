# One Dockerfile, two targets: `deps` (the locked runtime dependencies: what the pipeline runs with) and `test`
# (run_test's dependency image). Each is built on its own (`--target deps|test`). There is no server to preview and
# no production stage: the pipeline's runs go to Modal through the eval bundle (see README.md).
FROM python:3.11.17-slim-bookworm AS deps
RUN pip install --no-cache-dir "uv>=0.13.4,<0.14"
ENV UV_PROJECT_ENVIRONMENT=/deps/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH=/deps/venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /deps
# Lockfile optional (`*`): the first dependency build resolves it at /deps/uv.lock and run_test commits it into the
# component; `uv sync` honours a committed lockfile.
COPY pyproject.toml uv.lock* ./
# uv's download cache is a BuildKit cache mount: reused by the next rebuild on the same builder, never in the image.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-install-project --no-dev

# run_test's dependency/toolchain image, from a slim base rather than `deps`: only the `test` group (numpy, pandas,
# joblib and the test tools). xgboost and scikit-learn are stubbed in the tests, so the image stays small. Source is
# NOT copied: the runner clones the branch into /workspace/repo and runs test_command against /deps/venv.
FROM python:3.11.17-slim-bookworm AS test
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PROJECT_ENVIRONMENT=/deps/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH=/deps/venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /deps
COPY pyproject.toml uv.lock* ./
# `--only-group test`: none of the project's runtime dependencies. uv itself is removed again (the tests never call it).
RUN --mount=type=cache,target=/root/.cache/uv pip install --no-cache-dir "uv>=0.13.4,<0.14" \
    && uv sync --no-install-project --only-group test \
    && pip uninstall -y uv
