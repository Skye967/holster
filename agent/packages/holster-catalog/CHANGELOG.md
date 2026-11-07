# Changelog

All notable changes to `holster-catalog`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the package follows
[semver](https://semver.org/spec/v2.0.0.html). `holster_catalog.testing` is outside the
version and is not listed here.

## Unreleased

### Added

- The package: `holster_catalog.tmdb` and `holster_catalog.catalog_tool`, moved out of
  the agent service with no change in behaviour.
- `holster_catalog.graph`: the turn as a LangGraph `StateGraph`, built by `build_turn`
  as a `CatalogGraph`, with `TurnContext` and `TurnState` moved into it.

### Changed

- `CATALOG_TOOLS` is three Pydantic schemas; nothing executes a tool.

### Removed

- `catalog_tool.catalog_agent`, `CatalogAgent`, `TwoCallTurn`, `MODEL_CALLS_PER_TURN`,
  the `search_catalog`, `look_up_title` and `answer` tool functions, and the `langchain`
  dependency.
