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
| `POST` | `/api/titles/:id/verdict` | liked, disliked, seen, not interested |

## Running locally

Config comes from the environment only — there is no `.env` loading in the service.
Under `docker compose up` the variables come from the repo-root `.env`. To run it
directly, export them first:

```sh
set -a; source ../../.env; set +a
go run .
```

Required: `CLERK_JWKS_URL`, `CLERK_ISSUER`, `CLERK_AUTHORIZED_PARTIES`. The service
exits immediately if any is missing. `GATEWAY_PORT` defaults to 8080.

`CLERK_AUTHORIZED_PARTIES` is the comma-separated list of origins allowed to mint
session tokens, checked against the token's `azp` claim. It must include every origin
the web app is served from, or every request fails with `unauthorized party`.

## Rules

- Resolves the Clerk user ID once, at the edge, and passes it inward. The agent trusts
  the gateway for identity and does not re-verify sessions.
- **Owns every write.** The agent has no write path to the database by design; if a
  feature needs one, it belongs here.
- Assembles the context the agent needs — subscriptions, country, recent verdicts — so
  the agent never needs broad database access.
- Serves the cached provider list from `streaming_providers`, refreshing it from TMDB
  when the row for that country is older than 24h. If TMDB fails, serve the stale row
  rather than failing the request.
