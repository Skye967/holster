"""TMDB catalog client.

A thin async wrapper over the parts of the TMDB API the agent needs: title
search, ``discover`` (the workhorse — filter by streaming service, region,
runtime, year, genre, cast, crew, keywords), "more like this", and per-title
streaming availability.

What this module holds onto that the rest of the agent must not have to think
about:

- **Names, not IDs.** Callers pass "Tilda Swinton" and "heist"; the client
  resolves them to TMDB person and keyword IDs and caches the mapping for the
  life of the process. Genre names resolve against a frozen table — TMDB's genre
  list is reference data that changes about twice a decade.
- **The correctness floors from TASKS.md T12.** ``watch_region`` is required on
  every catalog call; a ``vote_count.gte`` floor is always applied and cannot be
  overridden; subscription matching is ``flatrate`` only, kept separate from
  rent/buy.
- **"Nothing matched" vs "something broke."** No results is an empty list, never
  an exception. Only TMDB being unreachable or returning 429/5xx raises
  (``TMDBUnavailable``), which the caller turns into "try again later" — never a
  status code on screen.

The client takes no URL, endpoint, or raw query from its caller — see the
invariant in ``../CLAUDE.md``. It reads no environment; the token is passed in.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable
from typing import Any, Literal, TypedDict, cast

import httpx2

logger = logging.getLogger("holster.tmdb")

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p"
# Sizes suit chat cards (TASKS.md T16). Other surfaces build their own from
# IMAGE_BASE and a raw path.
POSTER_SIZE = "w342"
LOGO_SIZE = "w92"
LANGUAGE = "en-US"

# A correctness floor, not a preference: without it "highest rated sci-fi"
# returns an obscure short with three 10/10 votes. The model must not be able to
# lower it, so it is not a parameter.
MIN_VOTE_COUNT = 200

REQUEST_TIMEOUT = 10.0
MAX_RETRIES = 2
MAX_RETRY_WAIT = 10.0

MediaType = Literal["movie", "tv"]
DiscoverSort = Literal["popularity.desc", "vote_average.desc"]
SearchKind = Literal["person", "keyword"]

# From /genre/movie/list and /genre/tv/list, 2026-09. Frozen rather than fetched:
# the list is stable reference data, and hardcoding keeps genre resolution a pure
# dict lookup with no I/O to mock. Keyed lower-case for case-insensitive matching.
MOVIE_GENRES: dict[str, int] = {
    "action": 28,
    "adventure": 12,
    "animation": 16,
    "comedy": 35,
    "crime": 80,
    "documentary": 99,
    "drama": 18,
    "family": 10751,
    "fantasy": 14,
    "history": 36,
    "horror": 27,
    "music": 10402,
    "mystery": 9648,
    "romance": 10749,
    "science fiction": 878,
    "tv movie": 10770,
    "thriller": 53,
    "war": 10752,
    "western": 37,
}
TV_GENRES: dict[str, int] = {
    "action & adventure": 10759,
    "animation": 16,
    "comedy": 35,
    "crime": 80,
    "documentary": 99,
    "drama": 18,
    "family": 10751,
    "kids": 10762,
    "mystery": 9648,
    "news": 10763,
    "reality": 10764,
    "sci-fi & fantasy": 10765,
    "soap": 10766,
    "talk": 10767,
    "war & politics": 10768,
    "western": 37,
}


class TMDBError(Exception):
    """TMDB returned something unexpected (e.g. a 4xx that is not 429). A bug in
    how the query was built — not transient, retrying will not help."""


class TMDBUnavailable(TMDBError):
    """TMDB was unreachable or returned 429/5xx. Transient: the caller surfaces
    'try again later', never a status code."""


class Title(TypedDict):
    tmdb_id: int
    media_type: MediaType
    title: str
    year: int | None
    overview: str
    poster_url: str | None
    vote_average: float
    vote_count: int
    genre_ids: list[int]


class Provider(TypedDict):
    provider_id: int
    provider_name: str
    logo_url: str | None


class WatchAvailability(TypedDict):
    # flatrate is "included with the subscription"; rent/buy are paid on top and
    # must never be shown as included (TASKS.md T12). link is the JustWatch-backed
    # watch page for the region.
    link: str | None
    flatrate: list[Provider]
    rent: list[Provider]
    buy: list[Provider]


_REGION_RE = re.compile(r"^[A-Z]{2}$")


def _validate_media_type(media_type: str) -> None:
    # media_type is typed as a Literal, but nothing enforces that at runtime and
    # it goes straight into the URL path — validate it here rather than trust
    # the caller, matching TASKS.md T13's "validate every argument before it
    # reaches a URL."
    if media_type not in ("movie", "tv"):
        raise ValueError(f"media_type must be 'movie' or 'tv', got {media_type!r}")


def _validate_region(region: str) -> None:
    if not _REGION_RE.match(region):
        raise ValueError(
            f"watch_region must be an ISO 3166-1 alpha-2 code, got {region!r}"
        )


def _validate_page(page: int) -> None:
    if not 1 <= page <= 500:
        raise ValueError(f"page must be between 1 and 500, got {page}")


def _validate_year(year: int) -> None:
    if not 1900 <= year <= 2100:
        raise ValueError(f"year out of range: {year}")


def _image_url(path: str | None, size: str) -> str | None:
    return f"{IMAGE_BASE}/{size}{path}" if path else None


def _year(date: str | None) -> int | None:
    return int(date[:4]) if date and date[:4].isdigit() else None


def _trim_title(raw: dict[str, Any], media_type: MediaType) -> Title:
    date_key = "release_date" if media_type == "movie" else "first_air_date"
    date = raw.get(date_key)
    return Title(
        tmdb_id=raw["id"],
        media_type=media_type,
        title=raw.get("title") or raw.get("name") or "",
        year=_year(date),
        overview=raw.get("overview") or "",
        poster_url=_image_url(raw.get("poster_path"), POSTER_SIZE),
        vote_average=float(raw.get("vote_average") or 0.0),
        vote_count=int(raw.get("vote_count") or 0),
        genre_ids=list(raw.get("genre_ids") or []),
    )


def _trim_provider(raw: dict[str, Any]) -> Provider:
    return Provider(
        provider_id=raw["provider_id"],
        provider_name=raw.get("provider_name") or "",
        logo_url=_image_url(raw.get("logo_path"), LOGO_SIZE),
    )


def _genres_into(
    params: dict[str, Any], key: str, names: list[str] | None, media_type: MediaType
) -> None:
    """Match genre names against the frozen table for ``media_type`` and add the
    IDs to ``params`` under ``key``. Unknown names are dropped with a warning."""
    if not names:
        return
    table = MOVIE_GENRES if media_type == "movie" else TV_GENRES
    ids: list[int] = []
    for name in names:
        gid = table.get(name.strip().lower())
        if gid is None:
            logger.warning("tmdb genre unresolved, constraint dropped (%s)", media_type)
        else:
            ids.append(gid)
    if ids:
        params[key] = ",".join(map(str, ids))


def _retry_wait(response: httpx2.Response | None, attempt: int) -> float:
    if response is not None:
        header = response.headers.get("retry-after", "")
        if header.isdigit():
            return min(float(header), MAX_RETRY_WAIT)
    return float(2**attempt)  # 1s, then 2s


# Indirection so tests can neutralise retry backoff without patching asyncio.
async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


async def _gather[T](*coros: Awaitable[T]) -> list[T]:
    """Concurrent gather that raises the first failure only after every
    coroutine has finished. Plain ``asyncio.gather`` leaves siblings of a
    failed task running in the background with no one to retrieve their
    result or exception; ``return_exceptions=True`` makes gather itself
    consume all of them, so nothing is left unobserved."""
    results = await asyncio.gather(*coros, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return cast(list[T], results)


class TMDBClient:
    """One instance per process. Owns an HTTP connection pool and the
    name-to-ID caches. Pass a ``transport`` to swap in a mock in tests;
    otherwise the default connects for real with one connect-error retry."""

    def __init__(
        self, token: str, *, transport: httpx2.AsyncBaseTransport | None = None
    ) -> None:
        self._client = httpx2.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=REQUEST_TIMEOUT,
            transport=transport or httpx2.AsyncHTTPTransport(retries=1),
        )
        # Process-lifetime, unbounded, no lock: person and keyword IDs never
        # change, a concurrent miss just re-fetches the same value, and misses
        # are cached too so a typo is not looked up twice. None means "no match".
        self._person_ids: dict[str, int | None] = {}
        self._keyword_ids: dict[str, int | None] = {}

    async def __aenter__(self) -> TMDBClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(
        self, path: str, params: dict[str, Any], *, allow_404: bool = False
    ) -> dict[str, Any]:
        logger.debug("tmdb GET %s keys=%s", path, sorted(params))
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self._client.get(path, params=params)
                response.raise_for_status()
                try:
                    data: dict[str, Any] = response.json()
                except ValueError as exc:
                    # A 2xx with a non-JSON body — a CDN error page or captive
                    # portal, say. Not a status code to retry on, but still
                    # "something broke," never "nothing matched."
                    raise TMDBUnavailable(
                        f"TMDB returned an unparseable response for {path}"
                    ) from exc
                return data
            except httpx2.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 404 and allow_404:
                    return {}
                transient = status == 429 or 500 <= status < 600
                if transient and attempt < MAX_RETRIES:
                    logger.warning("tmdb %s on %s, retrying", status, path)
                    await _sleep(_retry_wait(exc.response, attempt))
                    continue
                if transient:
                    raise TMDBUnavailable(f"TMDB {status} for {path}") from exc
                raise TMDBError(f"TMDB {status} for {path}") from exc
            except httpx2.RequestError as exc:
                if attempt < MAX_RETRIES:
                    logger.warning("tmdb request error on %s, retrying", path)
                    await _sleep(_retry_wait(None, attempt))
                    continue
                raise TMDBUnavailable(f"TMDB unreachable for {path}") from exc
        raise AssertionError("unreachable")  # loop either returns or raises

    async def _resolve_name(self, kind: SearchKind, name: str) -> int | None:
        """Look up one person or keyword name, cached for the life of the
        process — misses included, so a typo is not searched twice."""
        cache = self._person_ids if kind == "person" else self._keyword_ids
        key = name.strip().lower()
        if key in cache:
            return cache[key]
        data = await self._get(f"/search/{kind}", {"query": name})
        results = data.get("results") or []
        resolved: int | None = results[0]["id"] if results else None
        cache[key] = resolved
        if resolved is None:
            logger.warning("tmdb name unresolved, constraint dropped (%s)", kind)
        return resolved

    async def _resolve_into(
        self,
        params: dict[str, Any],
        key: str,
        names: list[str] | None,
        kind: SearchKind,
    ) -> None:
        """Resolve a list of names to IDs and add them to ``params`` under
        ``key``, comma-joined. Names that do not resolve are dropped; if none
        resolve, the parameter is omitted entirely."""
        if not names:
            return
        resolved = await _gather(*(self._resolve_name(kind, n) for n in names))
        ids = [i for i in resolved if i is not None]
        if ids:
            params[key] = ",".join(map(str, ids))

    async def search_titles(
        self, query: str, *, media_type: MediaType | None = None
    ) -> list[Title]:
        """Find titles by name. Without media_type, searches both movies and TV;
        merged results are ordered by vote count as a prominence proxy."""
        if media_type is not None:
            _validate_media_type(media_type)
        types: list[MediaType] = [media_type] if media_type else ["movie", "tv"]
        results: list[Title] = []
        for mt in types:
            data = await self._get(
                f"/search/{mt}",
                {"query": query, "include_adult": "false", "language": LANGUAGE},
            )
            results.extend(_trim_title(r, mt) for r in data.get("results", []))
        results.sort(key=lambda t: t["vote_count"], reverse=True)
        return results

    async def discover(
        self,
        *,
        media_type: MediaType,
        watch_region: str,
        watch_providers: list[int] | None = None,
        cast: list[str] | None = None,
        crew: list[str] | None = None,
        keywords: list[str] | None = None,
        without_keywords: list[str] | None = None,
        genres: list[str] | None = None,
        without_genres: list[str] | None = None,
        max_runtime_minutes: int | None = None,
        release_year_gte: int | None = None,
        release_year_lte: int | None = None,
        sort_by: DiscoverSort = "popularity.desc",
        page: int = 1,
    ) -> list[Title]:
        """The workhorse. TMDB applies every hard constraint here; the model's
        job is to rank what comes back, never to recall titles itself.

        Names for cast, crew and keywords are resolved to IDs; an unresolvable
        name is dropped with a warning rather than failing the call. Genres are
        matched against the frozen table for ``media_type``.
        """
        _validate_media_type(media_type)
        _validate_region(watch_region)
        _validate_page(page)

        params: dict[str, Any] = {
            "watch_region": watch_region,
            "sort_by": sort_by,
            "include_adult": "false",
            "language": LANGUAGE,
            "page": page,
            "vote_count.gte": MIN_VOTE_COUNT,
        }

        if watch_providers:
            ids_str = "|".join(str(int(p)) for p in watch_providers)
            params["with_watch_providers"] = ids_str
            # Subscription means included, not rentable — TASKS.md T12.
            params["with_watch_monetization_types"] = "flatrate"

        # Each field is an independent TMDB round trip; run them concurrently.
        await _gather(
            self._resolve_into(params, "with_cast", cast, "person"),
            self._resolve_into(params, "with_crew", crew, "person"),
            self._resolve_into(params, "with_keywords", keywords, "keyword"),
            self._resolve_into(params, "without_keywords", without_keywords, "keyword"),
        )
        _genres_into(params, "with_genres", genres, media_type)
        _genres_into(params, "without_genres", without_genres, media_type)

        if max_runtime_minutes is not None:
            if max_runtime_minutes <= 0:
                raise ValueError(
                    f"max_runtime_minutes must be positive, got {max_runtime_minutes}"
                )
            params["with_runtime.lte"] = max_runtime_minutes

        date_field = (
            "primary_release_date" if media_type == "movie" else "first_air_date"
        )
        if release_year_gte is not None:
            _validate_year(release_year_gte)
            params[f"{date_field}.gte"] = f"{release_year_gte}-01-01"
        if release_year_lte is not None:
            _validate_year(release_year_lte)
            params[f"{date_field}.lte"] = f"{release_year_lte}-12-31"

        data = await self._get(f"/discover/{media_type}", params)
        return [_trim_title(r, media_type) for r in data.get("results", [])]

    async def recommendations(
        self, *, media_type: MediaType, tmdb_id: int
    ) -> list[Title]:
        """ "More like this" for a known title. Does not filter by service — the
        caller cross-references availability."""
        _validate_media_type(media_type)
        data = await self._get(
            f"/{media_type}/{int(tmdb_id)}/recommendations", {"language": LANGUAGE}
        )
        return [_trim_title(r, media_type) for r in data.get("results", [])]

    async def watch_providers(
        self, *, media_type: MediaType, tmdb_id: int, watch_region: str
    ) -> WatchAvailability:
        """Where a title streams in one country. A title with no availability
        there returns empty arrays, not an error."""
        _validate_media_type(media_type)
        _validate_region(watch_region)
        data = await self._get(
            f"/{media_type}/{int(tmdb_id)}/watch/providers", {}, allow_404=True
        )
        region = (data.get("results") or {}).get(watch_region) or {}
        return WatchAvailability(
            link=region.get("link"),
            flatrate=[_trim_provider(p) for p in region.get("flatrate", [])],
            rent=[_trim_provider(p) for p in region.get("rent", [])],
            buy=[_trim_provider(p) for p in region.get("buy", [])],
        )
