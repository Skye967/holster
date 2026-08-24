# gateway

Go. The only service the frontend talks to.

## Responsibility

Terminate requests from `web/`, verify the caller's Clerk session, and fan out to the
internal services. Nothing else in the system is exposed to the browser.

## Endpoints (planned)

| Method | Path | Goes to |
|---|---|---|
| `POST` | `/api/chat` | `agent/` |
| `GET` | `/api/connections` | `credentials/` |
| `POST` | `/api/connections/search` | `auth/` — find a provider by name |
| `POST` | `/api/connections/:provider` | `auth/` — begin OAuth |
| `DELETE` | `/api/connections/:provider` | `credentials/` — revoke and delete |

## Rules

- Resolves the Clerk user ID once, at the edge, and passes it inward. Internal
  services trust the gateway for identity and do not re-verify sessions.
- Never touches the master encryption key, and never proxies a raw token to the
  client.
