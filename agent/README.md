# agent

Python. FastAPI + LangChain. Turns a message into TMDB catalog queries and an answer.

## Responsibility

Receive a message plus the context the gateway assembled — the user's streaming
subscriptions, country, recent verdicts — decide which catalog operations to run,
execute them against TMDB, and return text. There is no service picker in the UI;
choosing what to reach for is this service's job.

The agent is a pure function: history and context in, answer out. It retains nothing
between turns and holds no conversation state — see `../DECISIONS.md`.

## Endpoints

| Method | Path | Does |
|---|---|---|
| `GET` | `/health` | liveness |
| `POST` | `/chat` | *(planned)* run the turn, stream the answer back to the gateway |

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

## Tests

```sh
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy .
```

`test_tmdb.py` and `test_catalog_tool.py` run offline, against a mock transport and
fake models respectively. `test_tmdb_live.py` and `test_catalog_tool_live.py` make
real calls and skip unless `TMDB_API_KEY`/`ANTHROPIC_API_KEY` is set — the same
pattern as the gateway's `TEST_DATABASE_URL` tests.
