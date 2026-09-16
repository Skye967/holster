"""chat.py tests — offline, against fake interpret/rank models and a fake
TMDB transport, same shape as test_catalog_tool.py. These exercise the event
sequence and cancellation behaviour stream_chat() adds on top of search();
search()'s own behaviour (relaxation ladder, id-outside-candidates, etc.) is
covered there and not re-tested here.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, get_args
from unittest.mock import AsyncMock

import httpx2
import pytest
from pydantic import ValidationError

import chat
import tmdb
from catalog_tool import (
    _OUTCOME_SYSTEM_PROMPT,
    PROVIDERS_HEADER,
    CatalogToolError,
    ComposedReply,
    DiscoverIntent,
    OutcomeCause,
    RankedPick,
    RankResult,
    RelaxedConstraint,
    TitleRef,
    TitleVerdict,
)
from chat import (
    _RELAXED_LABELS,
    MAX_HISTORY_TURNS,
    ChatRequest,
    HistoryTurn,
    _nothing_found_message,
    _relaxed_note,
    _title_not_found_message,
    stream_chat,
)
from testutil import (
    ALL_JUDGED,
    CAPABILITY_QUESTION,
    COMPOSED_TEXT,
    GATEWAY_MAX_HISTORY_EXCHANGES,
    MOVIE_A,
    NEEDS_CLARIFICATION,
    NO_CANDIDATES,
    TITLE_LOOKUP,
    CountingModels,
    FakeTMDB,
    RecordingComposer,
    failing_compose,
    make_intent,
    ok_compose,
    ok_interpret,
    rank_must_not_run,
    raw_named,
)
from tmdb import Title, TMDBClient


async def _collect(
    req: ChatRequest,
    client: TMDBClient,
    interpret: Any,
    rank: Any,
    compose: Any = None,
) -> list[dict[str, Any]]:
    """`compose` defaults to a composer that answers without complaint. A turn
    that must not compose is asserted on a RecordingComposer's own `calls`
    rather than by a raising double — see that class for why."""
    return [
        e
        async for e in stream_chat(
            req,
            client=client,
            interpret_model=interpret,
            rank_model=rank,
            compose_model=compose or ok_compose,
        )
    ]


def test_no_providers_short_circuits_with_a_message() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="anything", watch_region="US", watch_providers=[])

    async def must_not_run(message: str) -> DiscoverIntent:
        raise AssertionError("interpret must not run with no subscriptions")

    composer = RecordingComposer()
    events = asyncio.run(
        _collect(req, fake_tmdb.client(), must_not_run, must_not_run, composer())
    )

    assert [e["type"] for e in events] == ["message", "done"]
    assert "Connections" in events[0]["text"]
    # No model call at all, which is the property services/gateway/chat.go's
    # guest throttle rests on: it meters a frame by the turn cap and the
    # per-IP limiter only when that frame carries providers.
    assert composer.calls == []
    assert fake_tmdb.requests == []


def test_capability_question_short_circuits_with_a_message() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="what can you do?",
        watch_region="US",
        watch_providers=[8],
        watch_provider_names=["Netflix", "Hulu"],
    )

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(is_capability_question=True)

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            fake_interpret,
            rank_must_not_run(CAPABILITY_QUESTION),
        )
    )

    # No "intent" event either — on_intent never fires for a capability
    # question (catalog_tool.py's search()), so there is nothing to
    # translate into an interpreting line.
    assert [e["type"] for e in events] == ["message", "done"]
    # "Netflix and Hulu", not "Netflix, Hulu" — matches services/gateway/
    # chat.go's humanJoin, used for the same provider list elsewhere in the
    # same conversation (the interpreting line).
    assert "Netflix and Hulu" in events[0]["text"]
    assert fake_tmdb.requests == []


def test_capability_question_degrades_gracefully_with_uncached_provider_names() -> None:
    """watch_providers non-empty but watch_provider_names empty is a real,
    reachable state (loadChatContext's per-country name cache hasn't warmed
    up yet — see chat.go) — must not be misread as "no services picked"."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="what can you do?",
        watch_region="US",
        watch_providers=[8],
        watch_provider_names=[],
    )

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(is_capability_question=True)

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            fake_interpret,
            rank_must_not_run(CAPABILITY_QUESTION),
        )
    )

    assert [e["type"] for e in events] == ["message", "done"]
    assert "Connections" not in events[0]["text"]
    assert "haven't picked" not in events[0]["text"]


def test_capability_question_takes_precedence_over_clarifying_question() -> None:
    """Nothing stops the model from setting both on the same response (the
    two fields are independent). Pins the deliberate elif ordering in
    stream_chat's run(): a capability question is a more specific signal
    than a clarifying nudge, so it wins and the clarifying text is dropped
    — not asserted anywhere else, so a future reordering could flip this
    silently without this test."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="help?",
        watch_region="US",
        watch_providers=[8],
        watch_provider_names=["Netflix"],
    )

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(
            is_capability_question=True,
            clarifying_question="What are you in the mood for?",
        )

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            fake_interpret,
            rank_must_not_run(CAPABILITY_QUESTION),
        )
    )

    assert [e["type"] for e in events] == ["message", "done"]
    assert "Netflix" in events[0]["text"]
    assert events[0]["text"] != "What are you in the mood for?"


def test_clarifying_question_short_circuits_with_the_models_question() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="recommend something", watch_region="US", watch_providers=[8]
    )

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(clarifying_question="What are you in the mood for tonight?")

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            fake_interpret,
            rank_must_not_run(NEEDS_CLARIFICATION),
        )
    )

    # No "intent" event either — on_intent never fires when search() short-
    # circuits for clarification (catalog_tool.py), same as a capability
    # question.
    assert [e["type"] for e in events] == ["message", "done"]
    assert events[0]["text"] == "What are you in the mood for tonight?"
    assert fake_tmdb.requests == []


def test_successful_search_emits_intent_then_results_then_done() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # MOVIE_A is id 101
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(keywords=["heist"])

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        pick = RankedPick(tmdb_id=candidates[0]["tmdb_id"], blurb="great fit")
        return RankResult(picks=[pick])

    events = asyncio.run(_collect(req, fake_tmdb.client(), fake_interpret, fake_rank))

    assert [e["type"] for e in events] == ["intent", "results", "done"]
    assert events[0]["intent"]["keywords"] == ["heist"]
    [pick] = events[1]["picks"]
    assert pick["title"] == "Fake Heist"
    assert pick["blurb"] == "great fit"
    assert events[1]["relaxed"] == []
    assert events[1]["kind"] == "search"
    # Enrichment reaches the wire event: genre_names is a pure
    # local lookup from MOVIE_A's genre_ids ([80] -> Crime), no TMDB call
    # needed; the other three fields come back empty because this test's
    # FakeTMDB has no /movie/101* endpoints queued, so title_details()/
    # watch_providers() succeed with a benign empty response — not the
    # None-on-failure degrade path, which test_catalog_tool.py covers
    # separately. This just confirms the fields reach the wire.
    assert pick["genre_names"] == ["Crime"]
    assert pick["runtime_minutes"] is None
    assert pick["cast"] == []
    assert pick["available_on"] == []


def test_intent_event_arrives_before_rank_is_called() -> None:
    """The whole reason for on_intent: the caller must be able to observe it
    before the (slower) rank step runs."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # MOVIE_A is id 101
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])
    order: list[str] = []

    async def fake_interpret(message: str) -> DiscoverIntent:
        order.append("interpret")
        return make_intent()

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        order.append("rank")
        return RankResult(picks=[RankedPick(tmdb_id=101, blurb="fits")])

    events = asyncio.run(_collect(req, fake_tmdb.client(), fake_interpret, fake_rank))

    assert [e["type"] for e in events] == ["intent", "results", "done"]
    assert order == ["interpret", "rank"]


def test_no_candidates_after_ladder_emits_a_message_naming_what_was_relaxed() -> None:
    fake_tmdb = FakeTMDB()
    # every /discover/movie call returns nothing; keyword resolves so it counts
    fake_tmdb.ok("/search/keyword", {"results": [{"id": 42, "name": "cozy"}]})
    req = ChatRequest(message="cozy and short", watch_region="US", watch_providers=[8])

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(max_runtime_minutes=90, keywords=["cozy"])

    composer = RecordingComposer()
    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            fake_interpret,
            rank_must_not_run(NO_CANDIDATES),
            composer(),
        )
    )

    assert [e["type"] for e in events] == ["intent", "message", "done"]
    assert events[1]["text"] == COMPOSED_TEXT
    # The labels are handed over as the composer's detail, so the sentence can
    # avoid advising a loosening that already happened. "the mood and how
    # long", not "the mood, how long" — _human_join, same grammar the
    # capability answer's provider list uses.
    assert composer.calls == [("nothing_matched", "the mood and how long")]


def test_nothing_matched_falls_back_to_the_template_naming_what_was_relaxed() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/search/keyword", {"results": [{"id": 42, "name": "cozy"}]})
    req = ChatRequest(message="cozy and short", watch_region="US", watch_providers=[8])

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(max_runtime_minutes=90, keywords=["cozy"])

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            fake_interpret,
            rank_must_not_run(NO_CANDIDATES),
            failing_compose,
        )
    )

    assert "the mood and how long" in events[1]["text"]


def test_tmdb_unavailable_maps_to_a_closed_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tmdb, "_sleep", AsyncMock())
    fake_tmdb = FakeTMDB()
    fake_tmdb.queue("/discover/movie", *[httpx2.Response(503) for _ in range(3)])
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])

    events = asyncio.run(_collect(req, fake_tmdb.client(), ok_interpret, ok_interpret))

    # error is terminal on its own — no trailing "done" (a turn ends with
    # exactly one of the two, never both; see chat.py's module docstring).
    assert [e["type"] for e in events] == ["intent", "error"]
    assert events[1]["reason"] == "tmdb_unavailable"


def test_interpret_failure_maps_to_model_unavailable() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])

    async def failing_interpret(message: str) -> DiscoverIntent:
        raise RuntimeError("model call timed out")

    events = asyncio.run(
        _collect(req, fake_tmdb.client(), failing_interpret, ok_interpret)
    )

    assert [e["type"] for e in events] == ["error"]
    assert events[0]["reason"] == "model_unavailable"


def test_bad_watch_region_maps_to_internal_not_tmdb_specific() -> None:
    """watch_region is caller-supplied, never model-derived (catalog_tool.py) —
    a malformed one is a gateway/caller bug, not 'try later'."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="something", watch_region="usa", watch_providers=[8])

    events = asyncio.run(_collect(req, fake_tmdb.client(), ok_interpret, ok_interpret))

    # on_intent already fired by the time discover() validates watch_region —
    # see catalog_tool.search(): interpret() -> on_intent() -> discover().
    assert [e["type"] for e in events] == ["intent", "error"]
    assert events[1]["reason"] == "internal"


def test_history_is_folded_into_the_message_text() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": []})
    req = ChatRequest(
        message="anything shorter?",
        watch_region="US",
        watch_providers=[8],
        history=[
            HistoryTurn(role="user", text="something like Heat"),
            HistoryTurn(role="assistant", text="Here are some heist films."),
        ],
    )
    seen: list[str] = []

    async def capturing_interpret(message: str) -> DiscoverIntent:
        seen.append(message)
        return make_intent()

    asyncio.run(_collect(req, fake_tmdb.client(), capturing_interpret, ok_interpret))

    [message] = seen
    assert "something like Heat" in message
    assert "Here are some heist films." in message
    assert message.endswith("Current message: anything shorter?")


def test_cold_start_folds_in_no_conversation() -> None:
    """A brand-new account has no verdicts and no history — both default to
    []. _with_history is a no-op on an empty list, so the message reaching
    interpret() ends with what the user typed and carries no prior turns. Same
    minimal shape as test_history_is_folded_into_the_message_text above — an
    empty discover() page means rank() is never reached, so there's nothing to
    stub there."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": []})
    req = ChatRequest(
        message="something funny for tonight", watch_region="US", watch_providers=[8]
    )
    seen: list[str] = []

    async def capturing_interpret(message: str) -> DiscoverIntent:
        seen.append(message)
        return make_intent()

    asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            capturing_interpret,
            rank_must_not_run(NO_CANDIDATES),
        )
    )

    [message] = seen
    assert "Recent conversation:" not in message
    assert "something funny for tonight" in message


def test_early_disconnect_cancels_the_background_search() -> None:
    """Simulates the caller (main.py's StreamingResponse) stopping iteration
    early, as it does on client disconnect — see chat.py's module docstring
    on cancellation.

    Deliberately does not assert *where* the cancellation lands (inside
    discover() or inside rank(), depending on scheduling) — only the
    contract that matters: the pipeline never reaches a result once the
    generator has been closed, and aclose() does not return until the
    background task has actually stopped (see stream_chat()'s finally,
    which awaits it rather than firing-and-forgetting task.cancel())."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # MOVIE_A is id 101
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])
    completed = asyncio.Event()

    async def slow_rank(message: str, candidates: list[Title]) -> RankResult:
        await asyncio.sleep(10)
        completed.set()
        return RankResult(picks=[])

    async def run() -> None:
        gen = stream_chat(
            req,
            client=fake_tmdb.client(),
            interpret_model=ok_interpret,
            rank_model=slow_rank,
            compose_model=ok_compose,
        )
        first = await gen.__anext__()
        assert first["type"] == "intent"
        await gen.aclose()

    asyncio.run(asyncio.wait_for(run(), timeout=2))
    assert not completed.is_set()


def test_chat_request_parses_the_gateways_verdict_json_verbatim() -> None:
    """The one place the Go->Python verdict contract is checked. The gateway
    marshals its own Verdict struct (services/gateway/verdicts.go), which is
    also the GET /api/verdicts response body and is mirrored a third time in
    web/src/lib/verdicts.ts -- so a rename made to suit the browser would
    otherwise reach production as a 422 on every chat turn, with the Go and
    Python suites both still green. This asserts the exact JSON tag names.
    """
    body = {
        "message": "something good",
        "watch_region": "US",
        "watch_providers": [8],
        # Verbatim from the Go struct's json tags.
        "verdicts": [
            {"tmdb_id": 101, "media_type": "movie", "verdict": "seen"},
            {"tmdb_id": 202, "media_type": "tv", "verdict": "liked"},
        ],
    }

    req = ChatRequest.model_validate(body)

    assert [(v.tmdb_id, v.media_type, v.verdict) for v in req.verdicts] == [
        (101, "movie", "seen"),
        (202, "tv", "liked"),
    ]


def test_chat_request_defaults_verdicts_when_the_gateway_omits_them() -> None:
    """agentChatRequest tags Verdicts omitempty, so a caller with no verdicts
    sends no key at all. Validated from a body because that is the shape
    omitempty produces — the key absent rather than null.
    The key must be absent, not null; omitempty on the Go side is what
    guarantees that, and services/gateway/verdicts_test.go pins it there."""
    req = ChatRequest.model_validate(
        {"message": "hi", "watch_region": "US", "watch_providers": [8]}
    )
    assert req.verdicts == []


def test_all_judged_message_replaces_the_loosening_advice() -> None:
    """all_judged wins over relaxed: if the ladder ran and the survivors were
    all already rated, "even after loosening how long" names the wrong cause."""
    assert "rated all of them" in _nothing_found_message("all_judged", [])
    assert "rated all of them" in _nothing_found_message("all_judged", ["runtime"])
    assert "loosening" in _nothing_found_message("nothing_matched", [])
    assert "how long" in _nothing_found_message("nothing_matched", ["runtime"])


def test_title_not_found_message_quotes_the_name_back() -> None:
    """Echoing the name is the point: a misheard or misspelled title is the
    likeliest reason a lookup found nothing, and the user can only see that
    if the text shows what was searched for."""
    text = _title_not_found_message("Nonexistent Film")

    assert '"Nonexistent Film"' in text
    assert "loosening" not in text


def test_chat_request_rejects_an_unknown_media_type() -> None:
    """TitleVerdict types media_type as a Literal while verdict is a plain str,
    and the docstring justifies the split at length - this is what holds the
    Literal half in place. A bad media_type must not reach the agent: it would
    never match an excluded key (so the title silently stops being excluded),
    and on a liked row _validate_media_type's ValueError isn't a TMDBError, so
    nothing on the taste path catches it - it fails the whole turn with a
    traceback, not quietly."""
    body = {
        "message": "hi",
        "watch_region": "US",
        "watch_providers": [8],
        "verdicts": [{"tmdb_id": 101, "media_type": "film", "verdict": "seen"}],
    }

    with pytest.raises(ValidationError):
        ChatRequest.model_validate(body)


def test_verdicts_reach_search_and_shape_the_reply() -> None:
    """The seam. stream_chat passing req.verdicts to search(), and search()
    settling on a cause from what the verdicts excluded, are each one line, and
    deleting either leaves every other test in this repo green - the feature
    would silently stop excluding anything while three suites stayed passing.
    This drives the whole path: a verdict on the only candidate must produce
    the already-rated message rather than results."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # MOVIE_A is id 101
    req = ChatRequest(
        message="a heist movie",
        watch_region="US",
        watch_providers=[8],
        verdicts=[TitleVerdict(tmdb_id=101, media_type="movie", verdict="seen")],
    )

    composer = RecordingComposer()
    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            ok_interpret,
            rank_must_not_run(ALL_JUDGED),
            composer(),
        )
    )

    # The cause is what carries the meaning here: "you have rated all of
    # these" is a different sentence from "nothing matched", and the composer
    # is told which, never left to infer it.
    assert composer.causes == ["all_judged"]
    assert [e["type"] for e in events] == ["intent", "message", "done"]
    assert events[1]["text"] == COMPOSED_TEXT


def test_unresolvable_title_gets_its_own_message_not_the_loosening_advice() -> None:
    """A name that TMDB can't resolve has no constraint to loosen, so
    _nothing_found_message's advice would read as nonsense in reply to it."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="do you have Nonexistent Film?", watch_region="US", watch_providers=[8]
    )

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(title="Nonexistent Film")

    composer = RecordingComposer()
    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            fake_interpret,
            rank_must_not_run(TITLE_LOOKUP),
            composer(),
        )
    )

    assert [e["type"] for e in events] == ["intent", "message", "done"]
    # The name reaches the composer as its detail, which is what lets the
    # sentence quote back what the user actually typed.
    assert composer.calls == [("title_not_found", "Nonexistent Film")]


def test_unresolvable_title_falls_back_to_a_template_quoting_the_name() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="is Nonexistent Film on netflix?",
        watch_region="US",
        watch_providers=[8],
    )

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(title="Nonexistent Film")

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            fake_interpret,
            rank_must_not_run(TITLE_LOOKUP),
            failing_compose,
        )
    )

    assert "Nonexistent Film" in events[1]["text"]
    assert "loosening" not in events[1]["text"]


def test_results_event_marks_a_lookup_so_history_says_what_happened() -> None:
    """services/gateway/chat.go's summarizePicks reads this to choose between
    "Looked up:" and "Suggested:", and that text is what the next turn's
    interpret() sees. Drop the field and a lookup is stored, and read back,
    as a recommendation the assistant never made."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/search/multi", {"results": [raw_named(101, "Heat", 8658)]})
    req = ChatRequest(
        message="is Heat on netflix?", watch_region="US", watch_providers=[8]
    )

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(title="Heat")

    events = asyncio.run(
        _collect(
            req, fake_tmdb.client(), fake_interpret, rank_must_not_run(TITLE_LOOKUP)
        )
    )

    assert [e["type"] for e in events] == ["intent", "results", "done"]
    assert events[1]["kind"] == "lookup"
    assert [p["title"] for p in events[1]["picks"]] == ["Heat"]


def test_intent_event_carries_the_title_for_the_gateways_line() -> None:
    """services/gateway/chat.go's interpretingLine templates off this field —
    excluded from the dump and the gateway silently says "Looking for movies"
    over a title lookup."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="is Heat on netflix?", watch_region="US", watch_providers=[8]
    )

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(title="Heat")

    events = asyncio.run(
        _collect(
            req, fake_tmdb.client(), fake_interpret, rank_must_not_run(TITLE_LOOKUP)
        )
    )

    assert events[0]["type"] == "intent"
    assert events[0]["intent"]["title"] == "Heat"


# --- the widening, said out loud --------------------------------------------


def test_all_shown_message_is_not_the_all_judged_one() -> None:
    """A query run dry by re-asking is not one whose every result the user
    rated, and saying so would be false."""
    shown = _nothing_found_message("all_shown", [])
    assert "rated" not in shown
    assert "already shown you" in shown


def test_all_shown_message_does_not_claim_the_catalog_is_exhausted() -> None:
    """No search earns "everything there is": the read can stop at the page
    ceiling, the clock or a failed call, and every query carries a vote floor
    the user never stated (tmdb.py's MIN_VOTE_COUNT). Claiming the catalog is
    out is the same class of false statement all_judged and all_shown were
    split apart to avoid."""
    shown = _nothing_found_message("all_shown", [])
    assert "everything I found" in shown
    # Both readings of the catalog claim, not just the one this branch tried:
    # "could find" reads as the catalog being out just as "there is" does.
    assert "everything there is" not in shown
    assert "everything I could find" not in shown


def test_relaxed_note_names_what_was_widened_and_how_far_it_is_exact() -> None:
    note = _relaxed_note(["genres"], exact_matches=2, total=5)
    assert "the first 2" in note
    assert "the genre" in note


def test_relaxed_note_reads_naturally_for_a_single_exact_match() -> None:
    assert "the first matches" in _relaxed_note(["runtime"], exact_matches=1, total=5)


def test_relaxed_note_claims_no_count_when_nothing_matched_exactly() -> None:
    """Counting survivors, not catalog rows: "only two matched" would be false
    for a page the user had simply already rated."""
    note = _relaxed_note(["runtime"], exact_matches=0, total=5)
    # "nothing left", not "nothing matched": the exact matches may simply have
    # all been shown on an earlier turn, and saying they never existed would
    # contradict the results still on screen above.
    assert "Nothing left that matches exactly" in note
    assert "the first" not in note


def test_every_relaxable_constraint_has_a_label() -> None:
    """The table is hand-synced with catalog_tool.RelaxedConstraint, and an
    unlabelled rung falls back to its raw code in user-facing copy —
    "loosening vote_average.desc". Cheap to assert, so the sync is checked
    rather than remembered."""
    assert set(get_args(RelaxedConstraint)) <= _RELAXED_LABELS.keys()


def test_the_rating_bar_is_spoken_in_both_sentences() -> None:
    """One rung, two places it can surface: above a widened list, and in the
    nothing-found reply."""
    assert "the rating bar" in _relaxed_note(["rating"], exact_matches=1, total=5)
    assert "the rating bar" in _nothing_found_message("nothing_matched", ["rating"])


def test_no_note_when_nothing_was_relaxed() -> None:
    assert _relaxed_note([], exact_matches=5, total=5) == ""


def test_no_note_when_widening_changed_nothing_on_screen() -> None:
    """A note about a wider search that contributed no pick sends the user
    looking for results that are not there."""
    assert _relaxed_note(["runtime"], exact_matches=5, total=5) == ""


def test_results_event_carries_the_note_for_a_widened_search() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": []})
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(message="something short", watch_region="US", watch_providers=[8])

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(max_runtime_minutes=90)

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        return RankResult(picks=[RankedPick(tmdb_id=101, blurb="fits")])

    events = asyncio.run(_collect(req, fake_tmdb.client(), fake_interpret, fake_rank))
    results = next(e for e in events if e["type"] == "results")

    assert results["relaxed"] == ["runtime"]
    assert "how long" in results["note"]


def test_results_event_note_is_empty_when_nothing_was_relaxed() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        return RankResult(picks=[RankedPick(tmdb_id=101, blurb="fits")])

    events = asyncio.run(_collect(req, fake_tmdb.client(), ok_interpret, fake_rank))
    results = next(e for e in events if e["type"] == "results")

    assert results["note"] == ""


def test_shown_reaches_search_and_keeps_a_title_off_the_screen() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(
        message="show me more",
        watch_region="US",
        watch_providers=[8],
        shown=[TitleRef(tmdb_id=101, media_type="movie")],
    )

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            ok_interpret,
            rank_must_not_run(NO_CANDIDATES),
        )
    )

    assert [e["type"] for e in events] == ["intent", "message", "done"]


def test_an_oversized_shown_set_is_cut_to_the_newest_and_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing upstream bounds `shown` any more, so if this ever fires the
    gateway is suppressing titles the search will not exclude. The oldest go,
    which is the wrong end — hence the warning: it is the only symptom."""
    monkeypatch.setattr(chat, "MAX_SHOWN", 2)
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # MOVIE_A is id 101
    req = ChatRequest(
        message="a heist movie",
        watch_region="US",
        watch_providers=[8],
        # 101 is the oldest, so the cut makes it eligible again.
        shown=[TitleRef(tmdb_id=i, media_type="movie") for i in (101, 900, 901)],
    )

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        return RankResult(
            picks=[RankedPick(tmdb_id=c["tmdb_id"], blurb="ok") for c in candidates]
        )

    with caplog.at_level("WARNING"):
        events = asyncio.run(_collect(req, fake_tmdb.client(), ok_interpret, fake_rank))

    assert [e["type"] for e in events] == ["intent", "results", "done"]
    assert [p["tmdb_id"] for p in events[1]["picks"]] == [101]
    assert "shown set truncated" in caplog.text


def test_the_history_backstop_stays_clear_of_the_gateways_window() -> None:
    """Pins MAX_HISTORY_TURNS against the gateway's maxHistoryExchanges.

    A backstop sitting at the operating point stops being a backstop and
    becomes a second, tighter window — the gateway would send five exchanges
    and the agent would quietly keep fewer, with no error anywhere."""
    gateway_window = GATEWAY_MAX_HISTORY_EXCHANGES * 2
    assert gateway_window <= MAX_HISTORY_TURNS, (
        f"MAX_HISTORY_TURNS {MAX_HISTORY_TURNS} is under the gateway's "
        f"{gateway_window}-message window; history would be cut here with no "
        f"error anywhere"
    )


# --- Voice -----------------------------------------------------------------
#
# The opener, the capability answer, and the per-turn model-call budget. The
# budget is asserted from the fakes' own counters rather than reasoned about:
# Gemini's free tier is 15 requests a minute, so a turn that quietly grew a
# third call costs the whole app a third of its throughput.


def test_the_opener_arrives_before_the_interpreting_line() -> None:
    """Order is the feature. The reply is prose above the italic interpreting
    line, and the gateway forwards in arrival order, so emitting it after
    "intent" would render it under the line it introduces."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(keywords=["heist"], reply="One sec while I look.")

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        return RankResult(picks=[RankedPick(tmdb_id=101, blurb="fits")])

    events = asyncio.run(_collect(req, fake_tmdb.client(), fake_interpret, fake_rank))

    assert [e["type"] for e in events] == ["reply", "intent", "results", "done"]
    assert events[0]["text"] == "One sec while I look."


def test_the_opener_is_not_a_search_parameter() -> None:
    """services/gateway's interpretingLine templates off the intent event, and
    agentIntent decodes only search fields — an opener arriving in there would
    be silently dropped there and duplicated here."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(keywords=["heist"], reply="One sec.")

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        return RankResult(picks=[RankedPick(tmdb_id=101, blurb="fits")])

    events = asyncio.run(_collect(req, fake_tmdb.client(), fake_interpret, fake_rank))

    intent = next(e for e in events if e["type"] == "intent")["intent"]
    assert "reply" not in intent
    assert "is_capability_question" not in intent
    assert "clarifying_question" not in intent


def test_a_blank_opener_sends_no_event() -> None:
    """reply is required of the model, which still leaves "" satisfying the
    schema — and an empty reply event renders as an empty paragraph."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(keywords=["heist"], reply="   ")

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        return RankResult(picks=[RankedPick(tmdb_id=101, blurb="fits")])

    events = asyncio.run(_collect(req, fake_tmdb.client(), fake_interpret, fake_rank))

    assert [e["type"] for e in events] == ["intent", "results", "done"]


def test_a_clarifying_question_is_its_own_reply() -> None:
    """The model's question already is the reply; an opener alongside it would
    say the same thing twice. search() returns before on_intent for this turn,
    which is what makes that structural rather than a check here."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="recommend something", watch_region="US", watch_providers=[8]
    )

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(
            clarifying_question="What are you in the mood for?",
            reply="An opener that must not be sent.",
        )

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            fake_interpret,
            rank_must_not_run(NEEDS_CLARIFICATION),
        )
    )

    assert [e["type"] for e in events] == ["message", "done"]
    assert events[0]["text"] == "What are you in the mood for?"


def test_a_capability_answer_is_the_models_own_words() -> None:
    """The whole answer rides on interpret(), so a capability question costs
    one model call rather than two."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="what can you do?",
        watch_region="US",
        watch_providers=[8],
        watch_provider_names=["Netflix", "Hulu"],
    )
    composer = RecordingComposer()

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(
            is_capability_question=True,
            reply="I can find you something on Netflix and Hulu.",
        )

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            fake_interpret,
            rank_must_not_run(CAPABILITY_QUESTION),
            composer(),
        )
    )

    assert [e["type"] for e in events] == ["message", "done"]
    assert events[0]["text"] == "I can find you something on Netflix and Hulu."
    assert composer.calls == []
    assert fake_tmdb.requests == []


def test_the_callers_services_reach_interpret_and_not_rank() -> None:
    """A capability answer can only name the caller's services if interpret()
    is told them. rank() is not: a blurb naming a service would sit beside a
    card whose availability line is computed from the real subscriptions, and
    the two contradicting each other is the trust bug that path guards."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(
        message="a heist movie",
        watch_region="US",
        watch_providers=[8],
        watch_provider_names=["Netflix", "Hulu"],
    )
    seen: dict[str, str] = {}

    async def fake_interpret(message: str) -> DiscoverIntent:
        seen["interpret"] = message
        return make_intent(keywords=["heist"])

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        seen["rank"] = message
        return RankResult(picks=[RankedPick(tmdb_id=101, blurb="fits")])

    asyncio.run(_collect(req, fake_tmdb.client(), fake_interpret, fake_rank))

    assert "Netflix" in seen["interpret"]
    assert "Hulu" in seen["interpret"]
    assert "Netflix" not in seen["rank"]
    assert "Hulu" not in seen["rank"]


def test_uncached_provider_names_still_send_a_real_list() -> None:
    """watch_providers non-empty with watch_provider_names empty is reachable
    — the gateway's per-country name cache may not have warmed up. The block
    is emitted anyway: it is the list the prompt calls real, and an empty one
    means the names are unresolved, not that the caller has no services."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(
        message="a heist movie",
        watch_region="US",
        watch_providers=[8],
        watch_provider_names=[],
    )
    seen: dict[str, str] = {}

    async def fake_interpret(message: str) -> DiscoverIntent:
        seen["interpret"] = message
        return make_intent(keywords=["heist"])

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        return RankResult(picks=[RankedPick(tmdb_id=101, blurb="fits")])

    asyncio.run(_collect(req, fake_tmdb.client(), fake_interpret, fake_rank))

    assert seen["interpret"].startswith(f"{PROVIDERS_HEADER}\n[]")
    assert seen["interpret"].endswith("a heist movie")


# --- The per-turn model-call budget ----------------------------------------


def _budget_for(
    req: ChatRequest,
    client: TMDBClient,
    models: CountingModels,
) -> list[dict[str, Any]]:
    return asyncio.run(
        _collect(req, client, models.interpret(), models.rank(), models.compose())
    )


def test_a_results_turn_spends_interpret_and_rank_and_nothing_else() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])
    models = CountingModels(
        intent=make_intent(keywords=["heist"], reply="One sec."),
        picks=RankResult(picks=[RankedPick(tmdb_id=101, blurb="fits")]),
    )

    events = _budget_for(req, fake_tmdb.client(), models)

    assert [e["type"] for e in events] == ["reply", "intent", "results", "done"]
    assert (models.interpret_calls, models.rank_calls, models.compose_calls) == (
        1,
        1,
        0,
    )
    assert models.total == 2


def test_an_empty_turn_spends_interpret_and_the_composer_and_nothing_else() -> None:
    """The spare call is real, not assumed: an empty pick list means an empty
    candidate list, and rank() returns before calling the model on one."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])
    models = CountingModels(intent=make_intent(reply="One sec."))

    events = _budget_for(req, fake_tmdb.client(), models)

    assert [e["type"] for e in events] == ["reply", "intent", "message", "done"]
    assert (models.interpret_calls, models.rank_calls, models.compose_calls) == (
        1,
        0,
        1,
    )
    assert models.total == 2


def test_a_capability_turn_spends_one_call() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="what can you do?", watch_region="US", watch_providers=[8]
    )
    models = CountingModels(
        intent=make_intent(is_capability_question=True, reply="Here's what I do.")
    )

    events = _budget_for(req, fake_tmdb.client(), models)

    assert [e["type"] for e in events] == ["message", "done"]
    assert models.total == 1


def test_a_clarifying_turn_spends_one_call() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="recommend something", watch_region="US", watch_providers=[8]
    )
    models = CountingModels(
        intent=make_intent(clarifying_question="What are you in the mood for?")
    )

    events = _budget_for(req, fake_tmdb.client(), models)

    assert [e["type"] for e in events] == ["message", "done"]
    assert models.total == 1


def test_a_lookup_turn_spends_one_call_when_the_title_resolves() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/search/multi", {"results": [raw_named(101, "Heat", 8658)]})
    req = ChatRequest(
        message="is Heat on netflix?", watch_region="US", watch_providers=[8]
    )
    models = CountingModels(intent=make_intent(title="Heat", reply="Checking."))

    events = _budget_for(req, fake_tmdb.client(), models)

    assert [e["type"] for e in events] == ["reply", "intent", "results", "done"]
    assert (models.interpret_calls, models.rank_calls, models.compose_calls) == (
        1,
        0,
        0,
    )


def test_an_unresolved_lookup_spends_interpret_and_the_composer() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="is Nonexistent on netflix?", watch_region="US", watch_providers=[8]
    )
    models = CountingModels(intent=make_intent(title="Nonexistent", reply="Checking."))

    events = _budget_for(req, fake_tmdb.client(), models)

    assert [e["type"] for e in events] == ["reply", "intent", "message", "done"]
    assert models.total == 2


def test_a_no_providers_turn_spends_nothing() -> None:
    """The cheapest turn in the app, and the first one most new visitors send.
    Composing it would put that message behind a call on a 15-request-a-minute
    tier, on a frame the gateway meters neither by cap nor by limiter."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="anything", watch_region="US", watch_providers=[])
    models = CountingModels()

    events = _budget_for(req, fake_tmdb.client(), models)

    assert [e["type"] for e in events] == ["message", "done"]
    assert models.total == 0


def test_a_failed_turn_does_not_spend_a_call_apologising() -> None:
    """An error turn ends at run()'s except, before every outcome branch, so
    the gateway's own copy answers it — which is the point: if the model is
    what failed, it cannot write its own apology."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])
    models = CountingModels()

    async def failing_interpret(message: str) -> DiscoverIntent:
        models.interpret_calls += 1
        raise CatalogToolError("model unavailable")

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            failing_interpret,
            models.rank(),
            models.compose(),
        )
    )

    # "error" is terminal on its own — a turn ends with exactly one of
    # "done" or "error", never both.
    assert [e["type"] for e in events] == ["error"]
    assert models.compose_calls == 0


def test_an_all_shown_turn_names_its_own_cause() -> None:
    """The cause decides which of three mutually exclusive sentences the user
    gets: "you've already seen these" and "you've rated these" are different
    claims, and only one is true of any turn."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(
        message="show me more",
        watch_region="US",
        watch_providers=[8],
        shown=[TitleRef(tmdb_id=101, media_type="movie")],
    )
    composer = RecordingComposer()

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            ok_interpret,
            rank_must_not_run(NO_CANDIDATES),
            composer(),
        )
    )

    assert [e["type"] for e in events] == ["intent", "message", "done"]
    # No detail: the relaxation story belongs to nothing_matched, and merging
    # it here would answer two questions at once.
    assert composer.calls == [("all_shown", "")]


def test_the_composer_falls_back_rather_than_outliving_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nothing_matched turn reaches the composer having already spent the
    ladder's budget, and the gateway drops a turn that passes its deadline —
    so a slow call would cost the answer that is already in hand. The real
    COMPOSE_TIMEOUT_SECONDS is shortened here rather than waited out; what is
    under test is that the bound exists, not its value."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])
    monkeypatch.setattr(chat, "COMPOSE_TIMEOUT_SECONDS", 0.01)

    async def slow_compose(
        message: str, cause: OutcomeCause, detail: str
    ) -> ComposedReply:
        await asyncio.sleep(5)
        return ComposedReply(text="too late")

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            ok_interpret,
            rank_must_not_run(NO_CANDIDATES),
            slow_compose,
        )
    )

    assert [e["type"] for e in events] == ["intent", "message", "done"]
    assert "Nothing matched" in events[1]["text"]


def test_every_outcome_cause_has_a_duty_in_the_prompt() -> None:
    """The prompt spells out one duty per cause, and they are mutually
    exclusive — a cause the prompt never names loses its duty silently, leaving
    the model to guess which of the others it is in."""
    for cause in get_args(OutcomeCause):
        # The labelled form, not a bare substring: a cause whose name happened
        # to occur in another duty's prose would otherwise pass with no duty
        # of its own.
        assert f"{cause}:" in _OUTCOME_SYSTEM_PROMPT


def test_a_composer_that_answers_with_the_wrong_shape_ends_the_turn() -> None:
    """with_structured_output can return something that is not a ComposedReply,
    and the cast in google_composer is erased at runtime; compose()'s isinstance
    check is what turns that into a CatalogToolError the turn degrades from."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])

    async def wrong_shape(message: str, cause: OutcomeCause, detail: str) -> Any:
        return None

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            ok_interpret,
            rank_must_not_run(NO_CANDIDATES),
            wrong_shape,
        )
    )

    # Degraded to the template rather than ending the turn: the answer was
    # already in hand, and only the wording was waiting on the model.
    assert [e["type"] for e in events] == ["intent", "message", "done"]
    assert "Nothing matched" in events[1]["text"]


def test_a_composer_that_raises_something_unexpected_still_ends_the_turn() -> None:
    """compose() narrows every Exception to CatalogToolError, so this is the
    residue: anything that is not an Exception at all. run()'s finally owes the
    sentinel regardless, or stream_chat never stops draining."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])

    class Fault(BaseException):
        pass

    async def hostile(message: str, cause: OutcomeCause, detail: str) -> Any:
        raise Fault("not an Exception")

    async def drain() -> list[dict[str, Any]]:
        seen: list[dict[str, Any]] = []
        with contextlib.suppress(Fault):
            async for e in stream_chat(
                req,
                client=fake_tmdb.client(),
                interpret_model=ok_interpret,
                rank_model=rank_must_not_run(NO_CANDIDATES),
                compose_model=hostile,
            ):
                seen.append(e)
        return seen

    events = asyncio.run(asyncio.wait_for(drain(), timeout=5))

    # The turn ends. It carries no outcome sentence — the gateway's dropped
    # stream fallback is what tells the user — but it ends.
    assert [e["type"] for e in events] == ["intent"]
