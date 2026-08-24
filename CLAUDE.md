# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

Holster — a unified AI chat interface over the user's connected apps. See
[README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

Planning docs live **outside** the repo, in the parent directory:
`../TASKS.md` (build plan) and `../DECISIONS.md` (why it's built this way).
Read `../TASKS.md` before starting work — it defines the task order and branch names.

## Workflow

- **One branch per task**, named in `../TASKS.md`. Never commit to `main`.
- **Ask before every commit. Ask again before every push.** Approval does not carry
  forward to the next commit.
- Open a PR; do not merge to `main` directly.

## Style

- Code comments: short and only where the code isn't self-evident.
- PR descriptions: what changed and why, a few lines. No ceremony sections.
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

Icons are `lucide-react` — the set shadcn/ui ships. Do not add a second icon library.

## Secrets

Never commit a filled-in `.env`. `.env.example` lists every variable and holds only
placeholders. Never print a real key into a commit, comment, or PR body.
