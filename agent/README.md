# agent

Python. FastAPI + LangChain. Turns a message into TMDB catalog queries and an answer.

## Responsibility

Receive a message plus the context the gateway assembled — the user's streaming
subscriptions, country, every verdict — decide which catalog operations to run,
execute them against TMDB, and return text. There is no service picker in the UI;
choosing what to reach for is this service's job.

The agent is a pure function: history and context in, answer out. It retains nothing
between turns and holds no conversation state — see `../ARCHITECTURE.md`'s
"One socket per session".

## Endpoints

| Method | Path | Does |
|---|---|---|
| `GET` | `/health` | liveness |
| `POST` | `/chat` | run the turn, stream the answer back to the gateway as NDJSON |

## Trust

This is the least trusted component in the system: it runs model-directed control flow
over text anyone can edit — film overviews, reviews, user messages. The design assumes
it will eventually be manipulated and makes that survivable rather than trying to
prevent it. See `../CLAUDE.md` for the invariants; the ones that bind this service:

- **It never writes.** Every mutation goes through the gateway. Its database role
  (`agent_ro`) has no `insert`, `update` or `delete` grant anywhere.
- **It is never reachable from the internet.** Only the gateway is exposed. Under
  compose it has no published port; in production it sits on an internal network with
  no ingress. `--host 0.0.0.0` in the container binds it for other compose services,
  not the world.
- **Its tools are a fixed, declared list.** No tool takes a URL, an endpoint, or a raw
  query from the model.
- **It holds only app-level secrets** — the TMDB key, the LLM provider key — neither of
  which opens any user's account.
- **Its database reads are scoped** to the catalog and conversation tables, nothing
  user-scoped beyond what the gateway hands it.

## Running locally

Config comes from the environment only — there is no `.env` loading in the service.
Under `docker compose up` the variables come from the repo-root `.env`.

```sh
uv sync
uv run uvicorn main:asgi_app --port 8000 --reload   # or: uv run python main.py
curl -s localhost:8000/health                        # {"status": "ok"}
```

The container runs `python main.py`, which configures logging and serves; the uvicorn
CLI above is the same thing with reload for local work.

| Variable | Required | Default | Notes |
|---|---|---|---|
| `TMDB_API_KEY` | yes | — | The v4 Read Access Token, sent as `Authorization: Bearer`. Read when the TMDB client is constructed. |
| `GOOGLE_API_KEY` | yes | — | Read when the catalog tool's model is constructed. |
| `GOOGLE_MODEL` | no | `gemini-3.5-flash-lite` | Powers all three catalog_tool steps — interpret, rank, and the outcome sentence. |
| `LOG_LEVEL` | no | `info` | `debug` \| `info` \| `warn` \| `error` |

Both keys are app-level: neither opens any user's account, so losing one costs a
rotation, not a user. See `CLAUDE.md`.

Logs are structured JSON to stdout, nothing else — routing and retention belong to
whatever runs the container. Each line carries the `X-Correlation-ID` the gateway
minted, so a request can be followed across services.

## TMDB client

`tmdb.py` wraps the catalog API: title search, `discover` (filter by streaming
service, region, runtime, year, genre, cast, crew, keywords), "more like this", and
per-title streaming availability. Callers pass names — "Tilda Swinton", "heist" — and
the client resolves them to TMDB IDs and caches the mapping. Results are ordered by
`vote_count` — most watched — by default, or by rating when the user asks for the best;
TMDB's own `popularity` is deliberately not offered, being a rolling trend metric that
ranked a 4.0-rated title above The Dark Knight. A `vote_count` floor is always applied
to `discover` and no caller can set it: it is derived from the sort, and is higher for
the rating sort, which is the only one a low floor actually misorders. (`search()`'s
last rung drops the rating sort for a query too narrow to fill under it, which lowers
the floor as a consequence — the sort is the only thing it sets.) Multiple cast or
crew names mean *all* of them; multiple mood keywords mean *any*, because TMDB's tag
data is far too sparse to survive an AND. Subscription matches are `flatrate` only; a
caller wanting rentals asks for them by name (`monetization="rent"`), which filters on
`rent|buy` and carries no provider list — passing one raises. No
results is an empty list; only TMDB being unreachable or 429/5xx
raises `TMDBUnavailable`. It takes no URL, endpoint, or raw query from its caller —
every path is built here from a validated media type or a literal — and reads no
environment — the token is passed to `TMDBClient(...)`.

## Catalog tool

`catalog_tool.py` puts a model in front of `tmdb.py`: two calls, not an agent loop —
or one, when the message names a title and `search()` looks it up instead. A lookup
answers with every title carrying that name — the film itself, and anything carrying
it on, so "Star Wars" reaches the sequels too.
`interpret()` turns a message into `DiscoverIntent` — `tmdb.discover()`'s creative
parameters, with no `watch_region`/`watch_providers` field, so which services results
come from is never the model's choice. What it can choose is whether the turn asks about
subscriptions or about rentals (`monetization`). `search()` calls `discover()` with
those parameters plus the caller's real region, and the caller's real providers unless
the message asked to rent — a rent turn has no provider list to filter on, and what a
card claims about streaming is cut to the caller's subscriptions either way, when
availability is merged in. `rank()` has the
model pick and explain the *real* candidates that came back — its output schema carries
no title metadata, only a selected id and a blurb, so a manipulated overview can win a
bad blurb at worst, never assert a fake title. `rank()` says nothing about how many
picks a turn shows; `search()` owns the whole size band.
A recommendation list aims for `RESULT_FLOOR`..`RESULT_CEILING` titles. Paging reads
toward the ceiling until TMDB has no next page — a thin page usually means exclusion ate
it, and the answer to that is more of what the user asked for, never a looser question.
Depth therefore scales with what is being excluded rather than sitting at a constant,
which is what lets a long conversation keep asking for more. Pages after the first go
out `DISCOVER_BATCH_PAGES` at a time, so a deep read costs a few round trips rather than
a dozen; `MAX_DISCOVER_PAGES` is a runaway ceiling on that, not a policy.
When paging cannot fill the floor — TMDB out of rows, the page ceiling, or the paging
budget spent — `search()` retries with soft constraints dropped in
`RelaxedConstraint`'s order, recording what it dropped in `CatalogResult.relaxed` for
the chat layer to explain. Never the user's services,
region, or exclusions, and never cast or crew: a named person is the request itself, so
padding two genuine matches with three unrelated films buys a count and loses the
answer. Genres bend only while a *resolved* cast or crew name still says what the search
is about — "a horror movie" widened to "any movie" answers a different question, and a
name TMDB could not resolve never reached the query, so it says nothing. Every rung is
guarded the same way on its own constraint: a keyword TMDB doesn't carry, or a genre
outside the media type's vocabulary (`horror` is a movie genre; TV has no equivalent),
never filtered anything, so dropping it would re-send an identical query and claim a
drop that never happened. The last rung is the rating sort itself, which carries a
higher vote floor — a query too narrow to fill at that floor falls back to the default
sort rather than answering short.

Rungs **accumulate**: a wider query's rows are appended to the narrower query's, never
substituted for them, so the titles that matched exactly survive being widened past.
Surviving is not the same as being picked, though: `rank()`'s candidate list carries no
rung marker, so it can name only widened titles. `search()` therefore guarantees every
exact match a place itself, tops the picks up to the floor, stable-sorts by the rung
they came from, and only then cuts to the limit — in that order, because cutting in
`rank()`'s own fit order could discard every exact match a widened query found. Both
the guarantee and the floor carry no blurb, being properties of this code rather than
requests made of the model, for the same reason `rank()` validates ids rather than
asking for valid ones.

The widening is bounded by `LADDER_BUDGET_SECONDS`, not run to exhaustion: paging
plus rungs is many sequential rounds of `discover()` calls at its ceiling, under
the gateway's 30s turn deadline. Paging may spend only `PAGING_BUDGET_SECONDS` of that, so
depth cannot leave the rungs without a clock on a slow TMDB — a spent paging budget
falls through to them, where a TMDB fault ends the ladder, a fault being the same call
a rung is about to make. It is a deadline around the whole widening, so a call already
in flight is cancelled rather than left to finish its own retries; page 1 sits outside
it, being the answer rather than the widening. What accumulated is then the reply —
which makes it a real answer rather than a truncated one.
LangChain is confined to `google_interpreter()`/`google_ranker()`; everything
else takes a plain async callable, the same shape as `TMDBClient`'s `transport=`.

## Chat pipeline

`chat.py`'s `stream_chat()` wraps `catalog_tool.search()` for `POST /chat`: a plain
chunked HTTP response (`application/x-ndjson`), not a second WebSocket — the browser's
socket belongs to the gateway alone (`../ARCHITECTURE.md`). Each line is one event:
`reply` (the model's one-line opener, ahead of `intent` on a turn that searches or looks
up), `intent` (as soon as `interpret()` resolves — well before the full pipeline finishes,
via `search()`'s `on_intent` hook), `results` or `message` (candidates, or a plain reply
when there are none — no subscriptions, nothing survived the relaxation ladder, or
everything found was already rated),
`error` (one of a fixed, closed set of reason codes — never raw exception text), and
`done`. `error` is terminal on its own; a turn ends with exactly one of `done` or
`error`, never both.

This is the agent-internal protocol, not the browser-facing one — the gateway owns the
public event vocabulary (`reply`/`interpreting`/`results`/`token`/`done`/`error`) and
templates it from these. Cancellation is plain `asyncio` cancellation: the gateway
aborting its HTTP call closes the response body, which stops `stream_chat()` iterating,
which cancels
the background `search()` call — nothing bespoke on this side.

`history` in the request body is the last few exchanges only, windowed by the gateway
from a `messages` table that persists the full conversation — this
agent still never reads that table itself, only the windowed slice handed to it per
call. It is folded into the text handed to `interpret()`/`rank()` rather than changing
`catalog_tool.py`'s `message: str` contract.

`shown` in the request body is what the gateway has already put on screen for this
conversation — read back from `messages.title_refs` for an account, kept on the
connection for a guest, who has no rows. It joins the same exclusion set as verdicts — but
unconditionally, including `want_to_watch` titles that verdicts alone deliberately keep
— so "show me 10 more" means ten *different* titles. It never reaches `all_judged`:
a query run dry by re-asking is not one whose every result the user rated, and that
case is `all_shown` instead. Like verdicts, it does not filter a named-title lookup —
and, symmetrically, a lookup turn does not *feed* the set either, nor does a turn whose
cards the gateway dropped as cancelled — for a guest. An account's set is read back from
`messages.title_refs`, which records neither distinction, so there a looked-up title is
barred from every later recommendation in that conversation, however long it runs.
Accepted, and the clean fix is a column rather than a heuristic on the stored text.

The gateway sends the whole set rather than a window of it — a title re-offered is a
visible bug, so nothing in it ages out — and paging above is what keeps that
affordable: the deeper the exclusion, the deeper the read. `MAX_SHOWN` here bounds
nothing upstream; it is there so that if a set ever does arrive oversized, the turn
says so rather than silently excluding less than the gateway believes it does.

`all_shown` therefore means this search found nothing new, never that TMDB ran out: the
read can stop at `MAX_DISCOVER_PAGES` or at the paging budget, and every query is
filtered by `tmdb.py`'s `MIN_VOTE_COUNT` — a floor the user never stated, which only
the rating rung bends and only back to that same floor. The copy says exactly that.

## Voice

Every sentence the user reads is written by the model, bar the ones that cannot be.
`_TONE` in `catalog_tool.py` is prepended to all three system prompts, so the opener, a
blurb, a clarifying question and an outcome sentence come from one assistant rather than
three. It says only how to sound — `rank()` is asked for a list of per-title blurbs, so a
tone rule forbidding lists or counts would contradict the job on the same call.

`_ANSWER_RULES` carries what a whole answer owes — name no film the user did not, give no
count, never say the catalog has run out, always say what to try next — and goes only to
the outcome prompt and to the capability case of the interpret prompt. Those two produce
the text the gateway stores as the turn's assistant text, which is read back into the next
turn's `interpret()`, where a named film reads as a title request and a number as a limit.

**A turn makes at most two model calls.** The opener rides on `interpret()` as
`DiscoverIntent.reply`, so it is free; a turn with results spends `interpret()` and
`rank()`. A turn that ends with nothing to show spends `interpret()` and one `Composer`
call instead, `rank()` having never reached the model — `chat.py` states why that spare
call is always there. Calls are not requests: interpret and rank are built with one
retry (`main.py`), so a results turn can cost three of the free tier's fifteen requests
a minute. The composer is built with none — its answer already exists in the template
beside it.

The composer is fed the cause (`nothing_matched`, `all_shown`, `all_judged`,
`title_not_found`), never asked to infer it: each owes the user a different sentence,
and telling someone they rated titles they were only shown is false. `search()` decides
the cause where it computes the facts behind it, so the sentence and the duty cannot
disagree. A composer call that fails or outlives `COMPOSE_TIMEOUT_SECONDS` falls back to
the template beside it rather than costing a turn the answer it already has.

What stays hand-written: `chat.py`'s `NO_PROVIDERS_MESSAGE`, because that turn reaches
no model at all and is the first message most new visitors see; `services/gateway/chat.go`'s
error copy, guest cap and rate-limit line, because if the model is what failed it cannot
write its own apology; and `_relaxed_note`, whose count is computed after `rank()` has cut
the picks, so no call is in flight to write it.

`verdicts` in the request body is the caller's whole `title_verdicts` set, loaded by the
gateway with the message. The gateway only reads the rows; what each verdict *means* for
a recommendation is decided here, in `search()`:

- **Every verdict but `want_to_watch` removes that title from the candidates `rank()`
  sees** (on the discover path; a named-title lookup runs no `rank()` and applies no
  verdict filter, because a factual question about a title is not a recommendation) —
  the four judgments are settled opinions, `want_to_watch` is an open
  intention. A set filter applied once, after paging and the relaxation ladder, never
  a prompt instruction, so it holds whatever the model does. An unrecognised verdict
  excludes too, which is the safe direction.
- **`liked` titles of the search's own media type become a hint prepended to `rank()`'s
  message**, and are never given to `interpret()` — that step sets hard TMDB filters,
  and a past like is a preference, not a constraint the user asked for. The hint is the
  first `MAX_TASTE_TITLES` likes, resolved to names via TMDB since the table stores only
  ids. Its effect is bounded: it reorders and re-selects within the one page
  `discover()` returned, it does not change what is in that page.

That second one is order-sensitive. The gateway sends verdicts newest-first and the
agent keeps the first few likes it sees, so the ordering is a contract rather than a
detail — `TitleVerdict` carries no timestamp, so the agent cannot re-sort, or even
notice if the order is lost. `services/gateway/verdicts.go` explains where that
ordering is itself only a proxy.

**Cold start:** a brand-new account has `verdicts == []` and
`history == []`. Both are no-ops at that length — `_with_taste`/`_with_history` return
the message unchanged — so `interpret()` and `rank()` see exactly what the caller typed,
nothing folded in. There is no special-cased "new user" code path; the cold-start answer
comes entirely from how well `interpret()`'s prompt extracts mood signal from a short,
colloquial first message, which is why `DiscoverIntent.keywords`'s field description in
`catalog_tool.py` explicitly tells the model to infer a mood-driven value rather than
leaving it empty on a vague-sounding request. It's the one field `_INTERPRET_SYSTEM_PROMPT`
names as an exception to its own "leave empty when unmentioned" rule — deliberately
keywords only, not genres: an unresolved keyword is dropped harmlessly, but `genres`
relaxes only when a cast or crew name TMDB actually resolved survives the drop, so on a
cold-start message naming no one — the common case — a wrong genre guess still zeroes
out the first results entirely. The `genres` field's own description carries a stricter
"only when unambiguous" caveat instead.

## Tests

```sh
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy .
```

The suite runs offline: `test_tmdb.py` against a mock transport, `test_catalog_tool.py`
against that plus fake interpret/rank callables. Nothing here calls TMDB or the model —
prompt behaviour is checked by running the app, not by asserting on a model's output.
