# Changelog

All notable changes to `holster-catalog`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the package follows
[semver](https://semver.org/spec/v2.0.0.html). `holster_catalog.testing` is outside the
version and is not listed here.

## 0.2.0 - 2026-11-07

### Added

- The package: `holster_catalog.tmdb` and `holster_catalog.catalog_tool`, moved out of
  the agent service with no change in behaviour.
- `holster_catalog.graph`: the turn as a LangGraph `StateGraph`, built by
  `build_turn(model, composer)` as a `CatalogGraph`, with `TurnContext` and `TurnState`
  moved into it. `finish` and `compose` are nodes, and `compose` degrades to its template
  through a node timeout and error handler.
- `holster_catalog.turn`: `stream_turn`, `Turn`, the events `Opener` and `Intent`, and
  the outcomes `Results` and `Message`.
- The public API, as `__all__` at the package root.

### Changed

- `CATALOG_TOOLS` is three Pydantic schemas; nothing executes a tool.
- `Composer` takes three arguments: the turn's config reaches it from the graph step it
  runs in.
- Requires `langgraph>=1.2.18`, the first whose async runs honour a node's error handler.

### Removed

- `catalog_tool.catalog_agent`, `CatalogAgent`, `TwoCallTurn`, `MODEL_CALLS_PER_TURN`,
  the `search_catalog`, `look_up_title` and `answer` tool functions, `catalog_tool.compose`,
  `CatalogResult.cause`, and the `langchain` dependency.
