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

## Scope

Two features: connection management, and one chat. Resist additions.

- **No service picker in the chat.** Choosing a provider is the agent's job. A
  dropdown to pick a service is scope creep, not a feature.
- Provider search is limited to providers with a registered OAuth client. "Any app"
  is not possible — OAuth requires prior registration.

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
