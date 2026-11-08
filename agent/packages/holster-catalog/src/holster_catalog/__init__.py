"""Holster's catalog core: the TMDB client and the agent over it.

`__all__` is the public API and what the version covers. holster_catalog.testing
is outside it."""

from importlib.metadata import version

from holster_catalog.catalog_tool import (
    CatalogToolError,
    ComposedReply,
    Composer,
    LookUpArgs,
    OutcomeCause,
    RelaxedConstraint,
    SearchArgs,
    TitleRef,
    TitlesRequest,
    TitleVerdict,
    enrich_watchlist,
    google_composer,
)
from holster_catalog.graph import CatalogGraph, TurnContext, TurnState, build_turn
from holster_catalog.tmdb import RegionProvider, TMDBClient, TMDBUnavailable
from holster_catalog.turn import (
    Intent,
    Message,
    Opener,
    Outcome,
    Results,
    Turn,
    TurnEvent,
    stream_turn,
)

__all__ = [
    "CatalogGraph",
    "CatalogToolError",
    "ComposedReply",
    "Composer",
    "Intent",
    "LookUpArgs",
    "Message",
    "Opener",
    "Outcome",
    "OutcomeCause",
    "RegionProvider",
    "RelaxedConstraint",
    "Results",
    "SearchArgs",
    "TMDBClient",
    "TMDBUnavailable",
    "TitleRef",
    "TitleVerdict",
    "TitlesRequest",
    "Turn",
    "TurnContext",
    "TurnEvent",
    "TurnState",
    "__version__",
    "build_turn",
    "enrich_watchlist",
    "google_composer",
    "stream_turn",
]

__version__ = version("holster-catalog")
