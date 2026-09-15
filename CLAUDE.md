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

- Match surrounding code. Don't introduce a second way to do something that already
  has one.

### Comments

**A comment is what the code cannot say. Short because history is gone, not because
detail is.** Length is an output of the filter below, not a budget: cutting mechanism to
hit a length is how a comment turns into a pointer.

**Keep** — the four things a comment is for, each answered *in the code*:

1. **Why this shape?** — the constraint that rules out the obvious simpler version.
2. **How does it work?** — the mechanism, when it spans more than the reader can see at
   once. Mechanism, never a restatement of the line beneath it.
3. **What is still true but not visible?** — a live limitation, an invariant, an
   accepted cost, stated as a fact rather than pointed at.
4. **What else must change with it?** — a coupling nothing enforces. This one *is* a
   reference, and must name a real symbol.

**Cut** — history (what the code used to be, what was tried, which bug prompted it),
provenance (task ids, PR numbers), restatement, and reassurance ("note that",
"obviously", "this is safe").

**Three things earn a reference:** a symbol; a fact already stated and then sourced
(`SQLSTATE 42501, "insufficient_privilege"`); or an invariant — but only from a doc that
ships with the code, such as `ARCHITECTURE.md`, a service's own README, or a migration.
`../TASKS.md` and `../DECISIONS.md` live outside the repo and a contributor cannot open
them, so an invariant sitting there is moved in, not cited. State the invariant, then
name the doc — never the name alone.

**Two tests.**

*Cover and reread* — does it explain itself? Cover every reference except the couplings
and ask whether you can still understand the code. If not, the comment is broken.

*Check, don't recall* — is it true? Verify every claim about anything outside this file
against that thing as you write it. Two shapes account for nearly every false comment:

- An **absolute** — only, never, every, nothing, the one, guarantees — which a single
  counterexample falsifies. If you have not checked exhaustively, weaken it: the weaker
  sentence is usually shorter and always truer.
- A **named reference** — open the file and find the sentence. Existence is not support;
  a heading can be present and still not carry your point. Schema objects are
  cumulative, so read the latest migration that touches one, not the one that created it.

**Name the coupling; don't describe the guard.** *"Mirrors `chat.go`'s `maxShownRefs`"*
is a fact a reader can check. *"TestX catches an edit here"* is a claim about program
behaviour that drifts the moment either side moves, and the test's own name already
carries it. If you do name a guard, you have run it and watched it fail.

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
  set; the destination may not. A *search term* is the one exception, and it is a
  narrow one: the model may name a person, a keyword, or a title, and that name
  reaches TMDB as a `query` value inside a path built here. It chooses what to look
  for, never where to look, and every row that comes back is a real TMDB row.
- **The agent holds no credential belonging to a user or a third-party account.** It
  may hold app-level API keys — TMDB, the LLM provider — whose compromise costs a key
  rotation, not a user. If a change would give it a credential that reaches a user's
  account anywhere, that change is wrong.
- **Row-level security on every user-scoped table**, keyed on the Clerk user ID.
- **Every catalog query carries `watch_region`, except `/search/multi`.**
  Availability is country-specific. That one endpoint accepts no region parameter
  at all, and the availability step that follows it carries one — so a card's
  streaming claim is still country-scoped. Named exception, not a judgement call:
  anything else that reaches TMDB without a region is a bug.
- **No table holds a credential.** Holster stores no secret belonging to any other
  service, for any user. There is no encryption layer because there is nothing to
  encrypt — do not reintroduce one without reintroducing the thing it protects.
- **`proxy.ts` is not an authorization boundary.** It only attaches auth state. Every
  page, route handler, and server action that touches user-scoped data calls
  `await auth.protect()` itself, or branches on `auth()` and touches none of it on
  the guest branch. Do not reintroduce `createRouteMatcher` — it is deprecated, and
  path matching can diverge from how Next routes requests, leaving protected
  resources reachable.
- **A guest is a browser with no session, and the gateway is what keeps it honest.**
  Guests get Connections (a cookie) and Chat (`/ws/chat?guest=1`, nothing persisted);
  everything user-scoped is a sign-in prompt. The gateway's `/guest/` prefix and the
  guest socket may never touch a user-scoped table — see `ARCHITECTURE.md`.

## Scope

Search and recommendation across the streaming services a user subscribes to. Two
features: connection management, and one chat. Resist additions.

- **No service picker in the chat.** Choosing a tool is the agent's job. A dropdown to
  pick a service is scope creep, not a feature.
- **Streaming services are a preference, not a connection.** Netflix and Hulu publish
  no OAuth. Users tick what they subscribe to; those rows hold no secret.
- **No OAuth to third-party services.** Adding a provider means reintroducing the whole
  credential architecture — user tokens, refresh, encryption at rest, revocation — which
  was removed with Spotify and YouTube. It is not an increment.
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
