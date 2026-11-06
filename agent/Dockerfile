# Two stages, mirroring gateway/Dockerfile: build the virtualenv with
# uv, then copy only the venv and source into a clean runtime image.
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies only, so this layer caches until a manifest or the lockfile changes.
COPY pyproject.toml uv.lock ./
COPY packages/holster-catalog/pyproject.toml packages/holster-catalog/
RUN uv sync --frozen --no-dev --no-install-workspace

# Installed, not linked: the runtime image carries the library in the venv alone.
COPY packages packages
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY *.py ./

# nobody: the process listens on 8000 and writes nothing, so it never needs root.
USER 65534:65534
EXPOSE 8000
# main.py configures logging and serves — see its __main__ block.
CMD ["python", "main.py"]
