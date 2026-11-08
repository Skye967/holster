# holster-catalog

Holster's catalog core: `tmdb.py`, an async client for the TMDB API; `catalog_tool.py`,
which puts a model in front of it; `graph.py`, the turn's steps; and `turn.py`, a turn as
its callers see it. The FastAPI service in `agent/` imports it and is, so far, its only
consumer.

It serves no HTTP, its code reads no environment variable, it imports no model provider,
and it takes the chat model and the TMDB client as arguments.

## Install

It is a member of the uv workspace rooted at `agent/`, where `uv sync` installs it with
the service. It is not on PyPI.

## Modules

Import from the package root: `__all__` is the public API and what the version covers.
`holster_catalog.testing` is outside it and imported by name.

| Module | What |
|---|---|
| `holster_catalog.tmdb` | `TMDBClient`, its result types and its errors |
| `holster_catalog.catalog_tool` | The declared tools, the prompts and the catalog ladder |
| `holster_catalog.graph` | The turn as a LangGraph graph: `build_turn`, `CatalogGraph`, `TurnContext`, `TurnState` |
| `holster_catalog.turn` | `stream_turn`, `Turn`, the events `Opener` and `Intent`, the outcomes `Results` and `Message` |
| `holster_catalog.testing` | Test doubles: `FakeTMDB`, `ScriptedChatModel`, scripted replies |

`holster_catalog.testing` ships because the service's tests need it too. It is outside
the version: any release may change it.

## Versioning

Semver, from 0.1.0. Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## Tests

```sh
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

Run them from this directory. The suite is offline.
