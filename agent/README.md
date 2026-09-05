# agent

Python. FastAPI + LangChain. Turns a message into TMDB catalog queries and an answer.

## Responsibility

Receive a message plus the context the gateway assembled — the user's streaming
subscriptions, country, every verdict — decide which catalog operations to run,
execute them against TMDB, and return text. There is no service picker in the UI;
choosing what to reach for is this service's job.

The agent is a pure function: history and context in, answer out. It retains nothing
between turns and holds no conversation state — see `../DECISIONS.md`.

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
| `ANTHROPIC_API_KEY` | yes | — | Read when the catalog tool's model is constructed. |
| `ANTHROPIC_MODEL` | no | `claude-haiku-4-5-20251001` | Powers both catalog_tool steps — interpreting a message and ranking candidates. |
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
the client resolves them to TMDB IDs and caches the mapping. A `vote_count` floor is
always applied and cannot be lowered; subscription matches are `flatrate` only. No
results is an empty list; only TMDB being unreachable or 429/5xx raises
`TMDBUnavailable`. It takes no URL or raw query from its caller, and reads no
environment — the token is passed to `TMDBClient(...)`.

## Catalog tool

`catalog_tool.py` puts a model in front of `tmdb.py`: two calls, not an agent loop.
`interpret()` turns a message into `DiscoverIntent` — `tmdb.discover()`'s creative
parameters, with no `watch_region`/`watch_providers` field, so the model can never
choose which streaming services results come from. `search()` calls `discover()` with
those parameters plus the caller's real region and providers, and `rank()` has the
model pick and explain up to `intent.limit` of the *real* candidates that came back —
its output schema carries no title metadata, only a selected id and a blurb, so a
manipulated overview can win a bad blurb at worst, never assert a fake title.
When the first query returns nothing, `search()` retries with soft constraints
dropped in a fixed order — runtime, then year, then mood keywords — never the
user's services, region, or exclusions, and records what it dropped in
`CatalogResult.relaxed` for the chat layer to explain ("nothing under 90 minutes
— here are the closest").
LangChain is confined to `anthropic_interpreter()`/`anthropic_ranker()`; everything
else takes a plain async callable, the same shape as `TMDBClient`'s `transport=`.

## Chat pipeline

`chat.py`'s `stream_chat()` wraps `catalog_tool.search()` for `POST /chat`: a plain
chunked HTTP response (`application/x-ndjson`), not a second WebSocket — the browser's
socket belongs to the gateway alone (`../DECISIONS.md`). Each line is one event:
`intent` (as soon as `interpret()` resolves — well before the full pipeline finishes,
via `search()`'s `on_intent` hook), `results` or `message` (candidates, or a plain reply
when there are none — no subscriptions, nothing survived the relaxation ladder, or
everything found was already rated),
`error` (one of a fixed, closed set of reason codes — never raw exception text), and
`done`. `error` is terminal on its own; a turn ends with exactly one of `done` or
`error`, never both.

This is the agent-internal protocol, not the browser-facing one — the gateway owns the
public event vocabulary (`interpreting`/`results`/`token`/`done`/`error`) and templates
it from these. Cancellation is plain `asyncio` cancellation: the gateway aborting its
HTTP call closes the response body, which stops `stream_chat()` iterating, which cancels
the background `search()` call — nothing bespoke on this side.

`history` in the request body is the last few exchanges only, windowed by the gateway
from a `messages` table that now persists the full conversation (`TASKS.md` T20) — this
agent still never reads that table itself, only the windowed slice handed to it per
call. It is folded into the text handed to `interpret()`/`rank()` rather than changing
`catalog_tool.py`'s `message: str` contract.

`verdicts` in the request body is the caller's whole `title_verdicts` set, loaded by the
gateway with the message. The gateway only reads the rows; what each verdict *means* for
a recommendation is decided here, in `search()`:

- **Every verdict but `want_to_watch` removes that title from the candidates `rank()`
  sees** — the four judgments are settled opinions, `want_to_watch` is an open
  intention. A set filter applied after the relaxation ladder, never a prompt
  instruction, so it holds whatever the model does. An unrecognised verdict excludes
  too, which is the safe direction.
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

## Tests

```sh
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy .
```

`test_tmdb.py` and `test_catalog_tool.py` run offline, against a mock transport and
fake models respectively. `test_tmdb_live.py` and `test_catalog_tool_live.py` make
real calls and skip unless `TMDB_API_KEY`/`ANTHROPIC_API_KEY` is set — the same
pattern as the gateway's `TEST_DATABASE_URL` tests.
