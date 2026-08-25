# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

Holster — a unified AI chat interface over the user's connected apps. See
[README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

Planning and process live **outside** the repo, in the parent directory:

| File | What |
|---|---|
| `../DEVOPS.md` | The work process, end to end. **Follow it for every task.** |
| `../TASKS.md` | Build plan — task order and branch names |
| `../DECISIONS.md` | Why the project is built this way |

## Code style

- Comments: short, and only where the code isn't self-evident.
- Match surrounding code. Don't introduce a second way to do something that already
  has one.

## Invariants

These are the point of the project. Do not break them for convenience.

- **Only `services/credentials` decrypts.** It alone reads `CREDENTIALS_MASTER_KEY`.
  Never pass that key to another service, and never import decryption as a library
  elsewhere.
- **Never return a plaintext token over the network** — not to the agent, not to the
  gateway, not to the browser. Callers ask `credentials` to perform the request.
- **`services/auth` stores nothing.** It forwards tokens to `credentials` and drops
  them.
- **Token columns are `bytea` and always ciphertext.** No migration may add a `text`
  token column.
- **Row-level security on every user-scoped table**, keyed on the Clerk user ID.
- **The agent never writes to a connected service.** It reads, searches, and drafts.
  Sending requires user approval.
- **`proxy.ts` is not an authorization boundary.** It only attaches auth state. Every
  page, route handler, and server action that touches protected data calls
  `await auth.protect()` itself. Do not reintroduce `createRouteMatcher` — it is
  deprecated, and path matching can diverge from how Next routes requests, leaving
  protected resources reachable.

## Scope

Search and recommendation across the streaming services a user subscribes to. Two
features: connection management, and one chat. Resist additions.

- **No service picker in the chat.** Choosing a tool is the agent's job. A dropdown to
  pick a service is scope creep, not a feature.
- **Two classes of tool.** Catalog tools (TMDB) use one app-level key and work for
  every user with no connection. Connected tools (Spotify, YouTube) need a per-user
  OAuth token and go through `credentials`. Do not blur them.
- **Streaming services are a preference, not a connection.** Netflix and Hulu publish
  no OAuth. Users tick what they subscribe to; those rows hold no secret and must not
  go near `credentials`.
- **Every catalog query carries `watch_region`.** Availability is country-specific.
- Adding a provider is config, not a new handler. If it needs Go changes, the registry
  design failed.

## Stack

| Path | Stack |
|---|---|
| `web/` | Next.js, TypeScript, Tailwind, shadcn/ui, Clerk |
| `services/*` | Go |
| `agent/` | Python, FastAPI, LangChain |
| `db/` | Supabase (PostgreSQL) |

**Package manager is `bun`.** Never run `npm`/`pnpm`/`yarn` in `web/` — it creates a
competing lockfile. Use `bun install`, `bun add`, `bun run dev`.

Icons are `lucide-react` — the set shadcn/ui ships. Do not add a second icon library.

Next.js 16 differs from older versions. `web/AGENTS.md` points at
`node_modules/next/dist/docs/` — read the relevant guide before writing app code.

## Secrets

Never commit a filled-in `.env`. `.env.example` lists every variable and holds only
placeholders. Never print a real key into a commit, comment, or PR body.
