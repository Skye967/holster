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
- **The correctness floors.** ``watch_region`` is required on
  every call whose answer depends on country — availability, and any browse that
  filters by service. Name search does not: ``/search/multi`` takes no region,
  and the availability step that follows it carries one. A ``vote_count.gte``
  floor is always applied to ``discover`` and cannot be overridden; subscription
  matching is ``flatrate`` only, kept separate from the ``rent|buy`` a caller
  has to ask for by name. The one URL that arrives whole rather than built
  here — a title's watch page — is dropped unless it is https.
- **"Nothing matched" vs "something broke."** No results is an empty list, never
  an exception. Only TMDB being unreachable or returning 429/5xx raises
  (``TMDBUnavailable``), which the caller turns into "try again later" — never a
  status code on screen.

The client takes no URL, endpoint, or raw query from its caller — the
invariant in ``../CLAUDE.md``. Every path is built here from a validated
``MediaType`` or a literal, never from a caller-supplied string. Search
*terms* are the one caller-supplied value that reaches TMDB, as a ``query``
parameter: person and keyword names via ``_resolve_name``, and a title via
``search_titles``. That is a value filled into a fixed destination, not a
destination. It reads no environment; the token is passed in.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable
from typing import Any, Literal, NoReturn, TypedDict, cast

import httpx2

logger = logging.getLogger("holster.tmdb")

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p"
# Sizes suit chat cards. Other surfaces build their own from
# IMAGE_BASE and a raw path.
POSTER_SIZE = "w342"
LOGO_SIZE = "w92"
LANGUAGE = "en-US"

# A correctness floor, not a preference: without it "highest rated sci-fi"
# returns an obscure short with three 10/10 votes. The model must not be able to
# lower it, so it is not a parameter. Unconditional on discover(); catalog_tool's
# name lookup reuses the number to rank namesakes, where it filters within a
# tier rather than gating (see lookup_title).
MIN_VOTE_COUNT = 200

# The same hazard one tier up, on the one sort exposed to it: 200 votes keeps an
# obscure short out of a *filtered* list but not off the top of one ordered by
# rating, where 9.4-from-888-votes outranks Breaking Bad's 8.9 from 18,579.
# Scoped to this sort rather than raised globally, which would answer every
# narrow request worse — Korean horror on a six-service account is 15 titles at
# 200 and 3 at 1000. Not raised further: at 3000 genuine classics with modest
# vote counts start dropping out (High and Low, 1190).
MIN_VOTE_COUNT_TOP_RATED = 1000

# TMDB's fixed page size for every list endpoint. Callers read it to tell
# "this page is all there is" from "there is another page" without spending a
# request to find out — a page short of this is the last one.
PAGE_SIZE = 20

REQUEST_TIMEOUT = 10.0
MAX_RETRIES = 2
MAX_RETRY_WAIT = 10.0

MediaType = Literal["movie", "tv"]
# "Most watched" and "best reviewed". TMDB's `popularity.desc` is deliberately
# not offered: it is a rolling trend metric, not a quality one, and there is no
# request it answers better — even "what's new" reads better by vote count. A
# real "what's new" would be a release-date window on top of vote_count.desc.
DiscoverSort = Literal["vote_count.desc", "vote_average.desc"]
# How a discover query is asked to treat availability. "subscription" pairs
# with a provider list and filters to what those services include; on its own
# it filters nothing, following the same "omit for none" convention as
# discover()'s other optional filters. "rent" filters to what can be paid for
# per title and takes no provider list. TMDB's free/ads types are not offered:
# "free with ads" is its own claim about where a title can be watched, and
# nothing here makes it yet.
Monetization = Literal["subscription", "rent"]
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

# The display-name counterpart to MOVIE_GENRES/TV_GENRES above: id -> TMDB's
# real casing, for showing a genre on a title card. Derived from the two
# tables above rather than a third hand-copied pair — those can never drift
# apart from each other since one is computed from the other. title()-casing
# matches TMDB's real casing for every entry except "tv movie", the one
# override below.
_GENRE_NAME_OVERRIDES = {10770: "TV Movie"}


def _genre_name_table(genres: dict[str, int]) -> dict[int, str]:
    return {
        gid: _GENRE_NAME_OVERRIDES.get(gid, name.title())
        for name, gid in genres.items()
    }


_MOVIE_GENRE_NAMES = _genre_name_table(MOVIE_GENRES)
_TV_GENRE_NAMES = _genre_name_table(TV_GENRES)


def genre_names(media_type: MediaType, genre_ids: list[int]) -> list[str]:
    """Resolve a Title's genre_ids to display names, in the order given. An id
    with no match is dropped silently rather than logged like other
    "unresolved" idioms here — it came from TMDB's own discover() response,
    not user or model input, so there's nothing a caller can act on."""
    table = _MOVIE_GENRE_NAMES if media_type == "movie" else _TV_GENRE_NAMES
    return [table[g] for g in genre_ids if g in table]


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


class RegionProvider(TypedDict):
    """One entry in the region-wide provider list behind the /connections
    picker — distinct from Provider because display_priority only makes sense
    for "every provider in a region," not for a single title's flatrate/rent/buy
    split."""

    provider_id: int
    provider_name: str
    logo_url: str | None
    display_priority: int


class WatchAvailability(TypedDict):
    # flatrate is "included with the subscription"; rent/buy are paid on top and
    # must never be shown as included. link is TMDB's own watch page for the
    # region, carrying the JustWatch-sourced store list — None when TMDB gave
    # none, and also when the one it gave failed watch_providers' scheme check,
    # since this one reaches a browser as an href.
    link: str | None
    flatrate: list[Provider]
    rent: list[Provider]
    buy: list[Provider]


# Top-billed cast shown on a title card — "two or three," capped
# here rather than left to the caller.
CAST_LIMIT = 3


class TitleDetails(TypedDict):
    runtime_minutes: int | None
    cast: list[str]  # top CAST_LIMIT billed names, in billing order


_REGION_RE = re.compile(r"^[A-Z]{2}$")


def _validate_media_type(media_type: str) -> None:
    # media_type is typed as a Literal, but nothing enforces that at runtime and
    # it goes straight into the URL path — validate it here rather than trust
    # the caller: every argument is validated before it reaches a URL.
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


def _unusable(what: str, exc: Exception) -> NoReturn:
    """A malformed TMDB body is a TMDBError, not a silent bug here — see
    _safe's docstring for why that, and not TMDBUnavailable, is the class
    that keeps this loud (logger.exception) rather than quietly downgraded
    as a transient, retry-later case. Shared so the five parsers below
    don't each restate the reasoning; only the noun changes per call site."""
    raise TMDBError(f"TMDB returned an unusable {what}") from exc


def _trim_title(raw: dict[str, Any], media_type: MediaType) -> Title:
    """One row of a list endpoint's results."""
    try:
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
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        _unusable("title row", exc)


def _trim_title_from_full(raw: dict[str, Any], media_type: MediaType) -> Title:
    """Same shape as _trim_title, for the single-title endpoint's raw
    response rather than discover()/search_titles()'s list shape -- genres
    arrive as [{"id","name"}] objects here, not a flat genre_ids list."""
    try:
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
            genre_ids=[g["id"] for g in raw.get("genres") or []],
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        _unusable("title page's fields", exc)


def _trim_details(raw: dict[str, Any], media_type: MediaType) -> TitleDetails:
    """Never reads "id" itself, so it cannot tell an error envelope from a
    real page on its own — both callers already gate on "id" being present
    (title_details() explicitly; title_with_details() by calling
    _trim_title_from_full on the same raw dict first) before reaching here."""
    try:
        if media_type == "movie":
            runtime = raw.get("runtime")
        else:
            # TMDB's own inconsistency: TV has no single "runtime", just a list of
            # typical episode lengths. The first is the best single number to show.
            episode_run_time = raw.get("episode_run_time") or []
            runtime = episode_run_time[0] if episode_run_time else None

        # TMDB already returns credits.cast in billing order; sorting on `order`
        # anyway is cheap insurance at this external boundary rather than relying
        # on an undocumented guarantee.
        cast_raw = sorted(
            (raw.get("credits") or {}).get("cast") or [],
            key=lambda c: c.get("order", 10**9),
        )
        return TitleDetails(
            # runtime: 0 is TMDB's own "not yet filled in" sentinel for titles
            # with incomplete metadata (same convention _trim_title leans on for
            # vote_average) — deliberately treated as "no data," not a real zero.
            runtime_minutes=int(runtime) if runtime else None,
            # Filter for a name before slicing, not after: a nameless entry among
            # the top CAST_LIMIT billed must not silently shrink the result below
            # CAST_LIMIT when a later-billed entry does have a name.
            cast=[c["name"] for c in cast_raw if c.get("name")][:CAST_LIMIT],
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        _unusable("title page's runtime/cast", exc)


def _trim_provider(raw: dict[str, Any]) -> Provider:
    """One provider entry."""
    try:
        return Provider(
            provider_id=raw["provider_id"],
            provider_name=raw.get("provider_name") or "",
            logo_url=_image_url(raw.get("logo_path"), LOGO_SIZE),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        _unusable("provider entry", exc)


def _trim_region_provider(raw: dict[str, Any], region: str) -> RegionProvider | None:
    # display_priorities is keyed by every region TMDB tracks; a provider with
    # no entry for this one is not actually available here, so it is dropped
    # rather than given a fake priority.
    try:
        priority = (raw.get("display_priorities") or {}).get(region)
        if priority is None:
            return None
        return RegionProvider(
            provider_id=raw["provider_id"],
            provider_name=raw.get("provider_name") or "",
            logo_url=_image_url(raw.get("logo_path"), LOGO_SIZE),
            display_priority=int(priority),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        _unusable("provider entry", exc)


def _genre_id(media_type: MediaType, name: str) -> int | None:
    """One genre name against the frozen table for ``media_type``, or None if
    it isn't in that vocabulary. The two tables differ — TV has no `horror`,
    movies have no `sci-fi & fantasy` — so the media type decides."""
    table = MOVIE_GENRES if media_type == "movie" else TV_GENRES
    return table.get(name.strip().lower())


def is_genre_resolved(media_type: MediaType, name: str) -> bool:
    """True if `name` is a real genre for `media_type`. Callers use it to tell
    a genre that filtered the query from one _genres_into dropped, the same
    question TMDBClient.is_keyword_resolved answers for keywords — a module
    function rather than a method because the genre tables are frozen
    constants, where the keyword and person caches are per-request state."""
    return _genre_id(media_type, name) is not None


def _genres_into(
    params: dict[str, Any], key: str, names: list[str] | None, media_type: MediaType
) -> None:
    """Match genre names against the frozen table for ``media_type`` and add the
    IDs to ``params`` under ``key``. Unknown names are dropped with a warning."""
    if not names:
        return
    ids: list[int] = []
    for name in names:
        gid = _genre_id(media_type, name)
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


async def gather_all[T](*coros: Awaitable[T]) -> list[T]:
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
        # Same rationale, same shape: a title's runtime and billing never
        # change once released. None means "no page for this id" and is
        # cached too — see title_details().
        self._title_details: dict[tuple[MediaType, int], TitleDetails | None] = {}

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
                    data = response.json()
                except ValueError as exc:
                    # A 2xx with a non-JSON body — a CDN error page or captive
                    # portal, say. Not a status code to retry on, but still
                    # "something broke," never "nothing matched."
                    raise TMDBUnavailable(
                        f"TMDB returned an unparseable response for {path}"
                    ) from exc
                # Parseable but not an object — a bare array or scalar. Same
                # category as the branch above, and the reason this is checked
                # rather than annotated: every caller does data.get(...), so
                # without it the annotation is a promise the code never keeps
                # and the AttributeError surfaces as a bug, not a TMDB fault.
                if not isinstance(data, dict):
                    raise TMDBUnavailable(f"TMDB returned a non-object body for {path}")
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

    def is_keyword_resolved(self, name: str) -> bool:
        """True if `name` resolved to a real TMDB keyword — i.e. a query
        using it actually filtered on something. Never triggers a lookup:
        the cache it reads is only populated by discover() calls, so only
        call this for a name the current request has already passed through
        discover(). Calling it before that conflates "not yet looked up"
        with "failed to resolve" — both read as False."""
        return self._keyword_ids.get(name.strip().lower()) is not None

    def is_person_resolved(self, name: str) -> bool:
        """True if `name` resolved to a real TMDB person. Same cache-only
        contract as is_keyword_resolved above — call it only for a name the
        current request has already passed through discover()."""
        return self._person_ids.get(name.strip().lower()) is not None

    async def _resolve_into(
        self,
        params: dict[str, Any],
        key: str,
        names: list[str] | None,
        kind: SearchKind,
        join: str = ",",
    ) -> None:
        """Resolve a list of names to IDs and add them to ``params`` under
        ``key``, joined by ``join``. Names that do not resolve are dropped; if
        none resolve, the parameter is omitted entirely.

        TMDB reads "," as AND and "|" as OR, so ``join`` is which of those a
        multi-value constraint means. It defaults to AND because that is what
        naming two people means — "with Hanks and Ryan" is both, not either —
        and only ``with_keywords`` passes "|". See discover()."""
        if not names:
            return
        resolved = await gather_all(*(self._resolve_name(kind, n) for n in names))
        ids = [i for i in resolved if i is not None]
        if ids:
            params[key] = join.join(map(str, ids))

    async def search_titles(self, query: str) -> list[Title]:
        """Find titles by name, in TMDB's own relevance order.

        /search/multi rather than /search/movie plus /search/tv: one request
        instead of two, and TMDB ranks films and series against each other
        rather than leaving us to merge two lists that each know nothing of
        the other. Its matching also handles what a person actually types —
        it resolves "Shogun" to "Shōgun", "Amelie" to "Amélie", and "Oceans
        Eleven" to "Ocean's Eleven".

        The order is returned exactly as TMDB gave it. Re-sorting — by vote
        count, say — discards the relevance signal and lifts a popular
        near-match above an exact one; a caller that wants a different
        ordering should apply it to this list, not ask for it here.

        Only movie and tv rows are kept: /search/multi returns people too,
        and a page can be mostly people ("Jackie Brown" returns nine)
        without costing the real answer, which is still in what remains.

        The isinstance guard is the same promise _get's non-object check
        makes one level up: this filter reads the row before _trim_title's
        own except can, so without it a malformed row surfaces as an
        AttributeError bug rather than the TMDBError it is.
        """
        data = await self._get(
            "/search/multi",
            {"query": query, "include_adult": "false", "language": LANGUAGE},
        )
        results = data.get("results") or []
        if not isinstance(results, list):
            # Present but the wrong shape. Left to the filter below this would
            # drop every row and read as "no such title" — a claim about the
            # user's spelling, not about TMDB. See the module docstring's
            # "'Nothing matched' vs 'something broke'".
            _unusable("search result list", TypeError(type(results).__name__))
        return [
            _trim_title(r, cast(MediaType, r["media_type"]))
            for r in results
            if isinstance(r, dict) and r.get("media_type") in ("movie", "tv")
        ]

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
        sort_by: DiscoverSort = "vote_count.desc",
        min_vote_count: int | None = None,
        monetization: Monetization = "subscription",
        page: int = 1,
    ) -> list[Title]:
        """The workhorse. TMDB applies every hard constraint here; the model's
        job is to rank what comes back, never to recall titles itself.

        Names for cast, crew and keywords are resolved to IDs; an unresolvable
        name is dropped with a warning rather than failing the call. Genres are
        matched against the frozen table for ``media_type``. ``watch_providers``
        follows the same "omit for none" convention as the other optional
        filters: pass ``None`` (the default) to skip provider filtering
        entirely; an explicit empty list raises ``ValueError`` rather than
        being silently treated the same as ``None``.

        ``monetization="rent"`` asks for what can be paid for per title rather
        than what a subscription includes, and carries no provider list:
        ``watch_providers`` is what the user subscribes to (``../CLAUDE.md``:
        "Users tick what they subscribe to"), and which of their subscriptions
        rents a title is a different question from where it can be rented.
        Passing both raises rather than quietly emitting a query that means
        neither.
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
            # Ordering by rating is the one thing this floor has to protect,
            # and 200 does not protect it — see MIN_VOTE_COUNT_TOP_RATED.
            # Caller-only, like watch_region above: the default follows the
            # sort, and search()'s last rung overrides it to bend the bar
            # without giving up the ordering the user asked for.
            "vote_count.gte": (
                min_vote_count
                if min_vote_count is not None
                else MIN_VOTE_COUNT_TOP_RATED
                if sort_by == "vote_average.desc"
                else MIN_VOTE_COUNT
            ),
        }

        if watch_providers is not None:
            if not watch_providers:
                # Omit the parameter for "no filter" — an explicit empty list
                # is a caller bug, not a request for every provider. Silently
                # treating [] like None would widen a zero-subscription
                # caller's query to the whole region instead of refusing it.
                raise ValueError(
                    "watch_providers must not be empty; omit it (None) for no filter"
                )
            if monetization == "rent":
                # TMDB ANDs the two, so this asks which of the user's own
                # subscriptions rents the title — a question neither the
                # caller nor the user asked. Raised rather than resolved
                # here for the same reason the empty list above is: a caller
                # bug that returns almost no rows reads downstream as
                # "nothing matched", which is a different claim.
                raise ValueError(
                    "watch_providers has no meaning on a rent query; omit it (None)"
                )
            ids_str = "|".join(str(int(p)) for p in watch_providers)
            params["with_watch_providers"] = ids_str

        if monetization == "rent":
            # "rent|buy", not "rent": a new release is often buy-only for its
            # first weeks, and a user who has said they will pay for it means
            # both.
            params["with_watch_monetization_types"] = "rent|buy"
        elif watch_providers is not None:
            # Subscription means included, not rentable. Paired with the
            # provider list rather than set unconditionally: a query with
            # neither asked for no availability filter at all, and adding one
            # here would silently drop every title the region streams nowhere.
            params["with_watch_monetization_types"] = "flatrate"

        # Each field is an independent TMDB round trip; run them concurrently.
        await gather_all(
            self._resolve_into(params, "with_cast", cast, "person"),
            self._resolve_into(params, "with_crew", crew, "person"),
            # OR, uniquely. Keywords are how a mood becomes a filter, and
            # TMDB's tag data is far too sparse to survive an AND: "thriller"
            # is 1459 titles, "thriller AND slow-burn" is zero, and
            # "thriller AND (slow-burn OR tense)" is 19. Zero then loses the
            # mood entirely to the relaxation ladder, so ANDing them meant a
            # mood was silently discarded on almost every request. OR lets
            # some noise in (Tokyo Drift is not a slow burn) and rank() is
            # what filters that, reading the user's actual words against 19
            # mood-adjacent candidates instead of a page of anything.
            self._resolve_into(params, "with_keywords", keywords, "keyword", join="|"),
            # Left as AND: TMDB returns the same rows either way here (1456 of
            # 1459 for "without gore, bleak" under both), so there is no
            # behaviour to buy by changing it.
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
        caller cross-references availability.

        Nothing calls this yet. Kept because "more like this" is in this
        module's brief; delete it if that stops being true."""
        _validate_media_type(media_type)
        data = await self._get(
            f"/{media_type}/{int(tmdb_id)}/recommendations", {"language": LANGUAGE}
        )
        return [_trim_title(r, media_type) for r in data.get("results", [])]

    async def title_details(
        self, *, media_type: MediaType, tmdb_id: int
    ) -> TitleDetails | None:
        """Runtime and top-billed cast for one title, via
        append_to_response=credits — one HTTP call for both. Cached for the
        life of the process, same rationale as _person_ids/_keyword_ids: a
        title's runtime and billing never change once released.

        Returns None, also cached, when TMDB has no page for this id. Raises
        TMDBError/TMDBUnavailable on request failure like every other method
        here — this does not self-degrade; a caller that wants "missing data,
        don't fail the turn" makes that choice itself (see
        catalog_tool.py's _safe_details). The cache is only written on the
        success path, so a transient failure is never cached as permanent
        "no data"."""
        _validate_media_type(media_type)
        key = (media_type, int(tmdb_id))
        if key in self._title_details:
            return self._title_details[key]
        data = await self._get(
            f"/{media_type}/{int(tmdb_id)}",
            {"language": LANGUAGE, "append_to_response": "credits"},
            allow_404=True,
        )
        # "id" as well as truthiness: a 2xx error envelope is a non-empty
        # dict that _trim_details reads happily, yielding empty details that
        # this cache would then serve as permanent truth.
        details = _trim_details(data, media_type) if data and "id" in data else None
        self._title_details[key] = details
        return details

    async def title_with_details(
        self, *, media_type: MediaType, tmdb_id: int
    ) -> tuple[Title, TitleDetails] | None:
        """Base title fields plus runtime/cast for an already-known id, via
        the same append_to_response=credits call title_details() makes --
        one HTTP request for both, not two. Unlike title_details(), always
        hits the network: title/year/vote_average/overview are live TMDB
        fields, not immutable metadata like runtime/cast, so caching them
        here would serve a stale rating as current -- the same class of bug
        "re-check availability at read time" exists to
        prevent, just applied to a rating instead of a service list. The
        runtime/cast half is still written into title_details()'s own
        cache, so a later title_details() call for this id is a cache hit,
        same as if discover() had produced this title first.

        Returns None when TMDB has no page for this id (the title was
        removed). Raises TMDBError/TMDBUnavailable like every other method
        here -- this does not self-degrade; see catalog_tool.py's _safe
        wrapper for the caller that wants "missing data, don't fail" instead.
        """
        _validate_media_type(media_type)
        data = await self._get(
            f"/{media_type}/{int(tmdb_id)}",
            {"language": LANGUAGE, "append_to_response": "credits"},
            allow_404=True,
        )
        if not data:
            return None
        title = _trim_title_from_full(data, media_type)
        details = _trim_details(data, media_type)
        self._title_details[(media_type, int(tmdb_id))] = details
        return title, details

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
        # Absent is not malformed: no results, no row for this region, or a
        # null list all mean "nothing here" and yield empty arrays, which the
        # card reads as "not on your services". A row that is present but the
        # wrong shape is a different thing and must not collapse into that
        # answer, so it raises through _unusable and reaches the card as
        # "availability unknown" instead. Without this, the AttributeError
        # would escape catalog_tool's _safe, which catches only TMDBError, and
        # break enrich_known_title's "never raises" contract — taking the
        # whole watchlist down rather than one row.
        try:
            region = (data.get("results") or {}).get(watch_region) or {}
            link = region.get("link")
            return WatchAvailability(
                # Every other URL this module hands out is built here from
                # IMAGE_BASE (see _image_url); this one arrives whole in a
                # response body and ends up as an href. Scheme only, and
                # case-insensitively, RFC 3986 schemes being case-insensitive:
                # checking the host would be a policy about TMDB's own body
                # rather than a check that this is a web address at all.
                # Anything else degrades to None, which callers render as text.
                link=link
                if isinstance(link, str) and link.lower().startswith("https://")
                else None,
                # `or []`, not a get() default: a key present with a null value
                # returns None and the default never fires. Same idiom as the
                # results lookup above and as _trim_title/_trim_details.
                flatrate=[_trim_provider(p) for p in region.get("flatrate") or []],
                rent=[_trim_provider(p) for p in region.get("rent") or []],
                buy=[_trim_provider(p) for p in region.get("buy") or []],
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            _unusable("watch-provider body", exc)

    async def watch_provider_list(self, *, watch_region: str) -> list[RegionProvider]:
        """Every provider available in one country, across movies and TV,
        merged by provider_id — a user thinks "I have Netflix," not "Netflix
        for movies," and the major platforms share one ID across media types.
        Powers the /connections picker; this call has no cache
        of its own, the caller (streaming_providers, gateway-side) owns that.
        """
        _validate_region(watch_region)
        movie_data, tv_data = await gather_all(
            self._get(
                "/watch/providers/movie",
                {"watch_region": watch_region, "language": LANGUAGE},
            ),
            self._get(
                "/watch/providers/tv",
                {"watch_region": watch_region, "language": LANGUAGE},
            ),
        )
        merged: dict[int, RegionProvider] = {}
        for data in (movie_data, tv_data):
            for raw in data.get("results", []):
                provider = _trim_region_provider(raw, watch_region)
                if provider is None:
                    continue
                existing = merged.get(provider["provider_id"])
                if existing is None or (
                    provider["display_priority"] < existing["display_priority"]
                ):
                    merged[provider["provider_id"]] = provider
        return sorted(merged.values(), key=lambda p: p["display_priority"])
