# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

Holster — chat-driven search and recommendation across the streaming services a user
subscribes to. See [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

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

The agent runs model-directed control flow over text anyone can edit — film overviews,
reviews, user messages. Assume it will eventually be manipulated, and keep that
survivable.

- **The agent never writes.** Not to the database, not anywhere. Every mutation goes
  through the gateway. Its database role has no `insert`, `update` or `delete` grant,
  and that grant must not be added.
- **The agent is never reachable from the internet.** Only the gateway is exposed.
- **Agent tools are a fixed, declared list.** Never add a tool that takes a URL, an
  endpoint, or a raw query from the model. Parameters may be filled in from a declared
  set; the destination may not.
- **The agent holds exactly one secret** — the TMDB key, which is app-level. If a
  change would give it a second, that change is wrong.
- **Row-level security on every user-scoped table**, keyed on the Clerk user ID.
- **Every catalog query carries `watch_region`.** Availability is country-specific.
- **No table holds a credential.** Holster stores no secret belonging to any other
  service, for any user. There is no encryption layer because there is nothing to
  encrypt — do not reintroduce one without reintroducing the thing it protects.
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
- **Streaming services are a preference, not a connection.** Netflix and Hulu publish
  no OAuth. Users tick what they subscribe to; those rows hold no secret.
- **No OAuth to third-party services.** Spotify and YouTube were planned and cut. Do
  not add a provider without revisiting `DECISIONS.md` — the whole credential
  architecture was removed with them, and adding one back means adding all of it back.
- **Taste comes from inside the app** — what the user subscribes to, what they type,
  and the verdicts they give on titles. Not from an external history feed.

## Stack

| Path | Stack |
|---|---|
| `web/` | Next.js, TypeScript, Tailwind, shadcn/ui, Clerk |
| `services/gateway/` | Go |
| `agent/` | Python, FastAPI, LangChain |
| `db/` | Supabase (PostgreSQL) |

**Package manager is `bun`.** Never run `npm`/`pnpm`/`yarn` in `web/` — it creates a
competing lockfile. Use `bun install`, `bun add`, `bun run dev`.

Icons are `lucide-react` — the set shadcn/ui ships. Do not add a second icon library.

Next.js 16 differs from older versions. `web/AGENTS.md` points at
`node_modules/next/dist/docs/` — read the relevant guide before writing app code.

## Attribution

TMDB's free tier is non-commercial and requires attribution: the TMDB logo plus
*"This product uses the TMDb API but is not endorsed or certified by TMDb"* in an About
or Credits section. Watch-provider data comes from JustWatch and must be credited to
them. This is a shipping requirement, not a nicety.

## Secrets

Never commit a filled-in `.env`. `.env.example` lists every variable and holds only
placeholders. Never print a real key into a commit, comment, or PR body.
