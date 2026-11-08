# holster-catalog

Holster's catalog core: `tmdb.py`, an async client for the TMDB API; `catalog_tool.py`,
which puts a model in front of it; `graph.py`, the turn's steps; and `turn.py`, a turn as
its callers see it. The FastAPI service in `agent/` imports it and is, so far, its only
consumer.

It serves no HTTP, its code reads no environment variable, it imports no model provider,
and it takes the chat model and the TMDB client as arguments.

## Install

It is a member of the uv workspace rooted at `agent/`, where `uv sync` installs it with
the service. It is not on PyPI: each release's wheel and sdist are attached to its
GitHub Release, `holster-catalog/vX.Y.Z`, and `uv add <wheel URL>` adds it to a project.

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

Semver, pre-1.0. A change to the public API (a name added to or removed from `__all__`,
or a change to what one takes, returns or does) is a minor release; a fix that makes a
name do what it already claimed, or any other change to the dependency ranges in its
`pyproject.toml` or to its code outside `holster_catalog.testing`, is a patch. The PR
that makes the change also bumps `version` in `pyproject.toml` and adds its entry to
[CHANGELOG.md](CHANGELOG.md), dated the day it merges. A change only to
`holster_catalog.testing`, the tests, the docs or `uv.lock` is not a release and needs
neither.

After the merge, tag `holster-catalog/vX.Y.Z` on main. The tag runs
`.github/workflows/release-catalog.yml`, which fails before building when the tag is not
on main, when it names a version other than `pyproject.toml`'s or when the library's
tests fail, then builds the wheel and sdist and attaches them to a GitHub Release. That
Release's notes are generated from the repository's merged PRs, not only the library's;
CHANGELOG.md is the library's record.

## Tests

```sh
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

Run them from this directory. The suite is offline.
