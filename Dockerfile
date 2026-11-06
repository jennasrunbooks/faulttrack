# FaultTrack service image (spec Section 7). Runtime dependencies only: no dev
# tools, tests, notebook, spike code or raw data (see .dockerignore).
#
# python:3.12-slim, multi-arch index digest read 2026-11-06 (arm64 for the Mac,
# amd64 for Fly.io and CI). Dependabot's docker ecosystem bumps it.
FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54

# uv 0.13.6, the version the devcontainer uses (multi-arch index digest).
COPY --from=ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8080

WORKDIR /app

# Layer 1: third-party dependencies. Cached until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Layer 2: the faulttrack package and the service files. README.md and LICENSE
# are listed in pyproject.toml, so the package build needs them.
COPY README.md LICENSE ./
COPY src/ src/
COPY predict.py ./
COPY models/ models/
COPY web/ web/
RUN uv sync --frozen --no-dev

RUN useradd --system --no-create-home --uid 10001 faulttrack
USER faulttrack

EXPOSE 8080

# sh expands $PORT; exec makes uvicorn PID 1 so it receives SIGTERM.
CMD ["sh", "-c", "exec uvicorn predict:app --host 0.0.0.0 --port \"$PORT\""]
