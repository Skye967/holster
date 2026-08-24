# Decision record

Short notes on choices that would otherwise look arbitrary later.

---

## Monorepo, not one repo per service

**Decision.** All services live in `Skye967/holster`.

Monorepo and monolith are unrelated: service boundaries are the architecture, and how
many git repos hold them is a source-control question. Splitting would break atomic
API-contract changes (one change becomes three coordinated PRs), lose
`clone && docker compose up`, and scatter one project across repos nobody would
assemble.

Polyrepo solves an organizational problem — independent teams, independent release
cadence, per-service access control. None of those apply yet. Extracting a service
later is easy; merging repos with divergent histories is not.

Also rejected: a GitHub **Organization** (moves the work off the personal profile) and
GitHub **Projects** (a kanban board, not a container for repositories).

---

## Boundaries drawn around trust

**Decision.** The service split follows blast radius, not code size.

- `credentials/` alone can decrypt. It alone reads the master key.
- `auth/` alone speaks to external OAuth providers, and stores nothing.
- `agent/` runs model-directed control flow, so it gets the narrowest surface: it can
  request actions, never credentials.

A monolith would work functionally. It could not make these boundaries *enforceable* —
which is the part worth building.

---

## Per-user derived encryption keys, single database

**Decision.** One Postgres instance; tokens encrypted per user with
`HMAC(master_key, user_id)`.

A database per user was considered and rejected — provisioning and connection
management per signup is real operational cost for an MVP, and shipping the simpler
design cleanly beats shipping the complex one badly. Derived keys get most of the
isolation benefit: a database dump yields ciphertext under many distinct keys.

Revisit if Holster ever holds credentials for organizations rather than individuals.

---

## Clerk for authentication

**Decision.** Clerk handles Holster's own sign-in.

Auth is not what this project is demonstrating — the OAuth token lifecycle is. Clerk
removes session handling, MFA, and email verification so effort goes into
`credentials/` and `agent/` instead. NextAuth was the alternative and would show more
auth internals, at the cost of the parts that matter more here.

---

## No service picker in chat

**Decision.** The chat has one input and no dropdown.

Choosing which service answers a question is the agent's job. A picker would move that
decision back onto the user and reduce the product to a set of separate search boxes
sharing a page.

---

## Lucide for icons

**Decision.** `lucide-react`, matched by hand-built Lucide vector components in Figma.

It is the set shadcn/ui already ships, so design and code reference the same icons
rather than lookalikes. The Figma community kits subscribed to the file return nothing
searchable on a Starter plan, so the icons were drawn from Lucide's SVG paths directly.
