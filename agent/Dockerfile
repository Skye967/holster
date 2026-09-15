# Two stages, mirroring services/gateway/Dockerfile: build the virtualenv with
# uv, then copy only the venv and source into a clean runtime image.
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Resolve against the lockfile only, so this layer caches until it changes.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY . .

# nobody: the process listens on 8000 and writes nothing, so it never needs root.
USER 65534:65534
EXPOSE 8000
# main.py configures logging and serves — see its __main__ block.
CMD ["python", "main.py"]
