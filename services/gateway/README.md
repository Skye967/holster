# gateway

Go. The only service the frontend talks to, and the only one that writes.

## Responsibility

Terminate requests from `web/`, verify the caller's Clerk session, own every database
write, and hand catalog questions to `agent/`. Nothing else in the system is exposed to
the browser.

## Endpoints (planned)

| Method | Path | Does |
|---|---|---|
| `GET` | `/health` | liveness |
| `POST` | `/api/chat` | assemble context, call `agent/`, persist the exchange |
| `GET` | `/api/subscriptions` | available providers for the user's country, plus their picks |
| `POST` | `/api/subscriptions/:id` | tick a service |
| `DELETE` | `/api/subscriptions/:id` | untick a service |
| `POST` | `/api/titles/:id/verdict` | liked, disliked, seen, not interested, want to watch |

## Running locally

Config comes from the environment only — there is no `.env` loading in the service.
Under `docker compose up` the variables come from the repo-root `.env`. To run it
directly, export them first:

```sh
set -a; source ../../.env; set +a
DATABASE_URL=postgresql://gateway_app:password@localhost:5432/holster go run .
```

`DATABASE_URL` is passed on the command line rather than kept in `.env` because the
two ways of running need different hosts: `localhost` here, `postgres` inside the
compose network. One variable cannot hold both, so `.env` leaves it unset, compose
supplies its own default, and a host run names the value it needs.

Required: `CLERK_JWKS_URL`, `CLERK_ISSUER`, `CLERK_AUTHORIZED_PARTIES`,
`CLERK_AUDIENCE`, `DATABASE_URL`. The service exits immediately if any is missing,
if the database is unreachable, or if the schema is not there. `GATEWAY_PORT`
defaults to 8080.

Set `DATABASE_URL` in `.env` only to point the gateway at Supabase. Use port 5432
(session mode), not 6543: the transaction pooler reassigns connections between
queries, which breaks the prepared statements pgx uses by default.

`statement_timeout`, `application_name`, and the pool size are set in code rather
than in the URL, so they survive that override. Pool size is a small fixed ceiling
(pgx's default is derived from the host CPU count, which is the wrong basis for a
gateway that makes one short query per request); if this ever runs more than one
replica, `replicas × MaxConns` must stay under the Supabase tier's pooler limit.

The local database is seeded from `supabase/migrations` on first start only.
Postgres runs those once, when its data directory is empty — after that the volume
persists and new migration files are ignored. `docker compose down -v` to re-init.

## Tests

```sh
go test ./...                                    # unit tests, no database needed
TEST_DATABASE_URL=postgresql://postgres:password@localhost:5432/holster go test ./...
```

Database tests skip unless `TEST_DATABASE_URL` is set. It is deliberately not
`DATABASE_URL`, so running the suite cannot write to a configured production
database. Point it at a superuser/owner connection — the RLS tests use `SET ROLE`
to drop into `gateway_app` and `agent_ro`; the local compose `postgres` role
works.

## User provisioning

Every authenticated request upserts the caller into `users`, keyed on the Clerk
user ID and carrying the `email` claim. Clerk creates the account on its side
only; nothing writes ours.

Per request rather than on a `user.created` webhook, and guarded by a `where … is
distinct from` clause so an unchanged row is not rewritten — see `../../DECISIONS.md`
and `upsertUser`.

A provisioning failure returns 503 rather than letting the request through, bounded
by a 2s deadline so an unresponsive database cannot hold the request open.

The gateway connects as `gateway_app` — a non-owner role with `select`, `insert`,
`update` and `delete` on the app tables and nothing else (no `drop`, no object
creation, no `pg_authid`). It is subject to row-level security: every user-scoped
operation runs through `withUser`, which opens a transaction, sets
`holster.user_id` to the caller's Clerk ID, and hands the handler a `pgx.Tx` the
policies pin to that ID. That is defence in depth behind each handler's own
`where user_id = $1` — the browser never reaches the database directly.

`agent_ro`, created in the same migration, has no write grant anywhere and no
read grants yet — each read is added by the task that needs it. Both roles are
`nologin` in the migration; a local login and password come from
`db/local-roles.sql`, and Supabase sets them in the dashboard.

## Logs

Structured JSON to stdout, nothing else — routing and retention belong to whatever
runs the container, not to the service.

`LOG_LEVEL` is the only verbosity control (default `info`):

| Level | What you get |
|---|---|
| `warn` | Anomalies only — bad signature, unknown key, unauthorized party |
| `info` | The above, plus startup and JWKS refresh failures |
| `debug` | The above, plus every rejected request with its reason |

Rejections log a reason code and correlation ID, never the token, the claimed
subject, or the raw error — an unverified token's contents are attacker-controlled.

A database error is logged as its SQLSTATE alone, since a constraint violation's
`DETAIL` carries row values. Connection and timeout errors log in full.

**One line per rejected request at `debug`.** That is unbounded by design: rate
limiting belongs in the log collector, which can be retuned without a redeploy.
Leave the level at `info` or above in production unless you are actively debugging.

`CLERK_AUTHORIZED_PARTIES` is the comma-separated list of origins allowed to mint
session tokens, checked against the token's `azp` claim. It must include every origin
the web app is served from, or every request fails with `unauthorized party`.

## Rules

- Resolves the Clerk user ID once, at the edge, and passes it inward. The agent trusts
  the gateway for identity and does not re-verify sessions.
- **Owns every write.** The agent has no write path to the database by design; if a
  feature needs one, it belongs here.
- Assembles the context the agent needs — subscriptions, country, every verdict — so
  the agent never needs broad database access.
- Serves the cached provider list from `streaming_providers`, refreshing it from TMDB
  when the row for that country is older than 24h. If TMDB fails, serve the stale row
  rather than failing the request.
