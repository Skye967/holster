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
- **It holds exactly one secret** — the TMDB key, which is app-level and opens nobody's
  account.
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

`LOG_LEVEL` (`debug` | `info` | `warn` | `error`, default `info`) is the only knob so
far. Logs are structured JSON to stdout, nothing else — routing and retention belong
to whatever runs the container. Each line carries the `X-Correlation-ID` the gateway
minted, so a request can be followed across services.

## Tests

```sh
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy .
```
