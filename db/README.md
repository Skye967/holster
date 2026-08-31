# db

Supabase (PostgreSQL). Schema reference for the tables below.

Migrations live in `supabase/migrations/` at the repo root, not here — that is the
only path the Supabase CLI discovers. Filenames are timestamp-prefixed and applied
in that order.

**Local stack:** the compose Postgres applies these automatically from the same
directory, but only on first start with an empty data directory. To re-apply after
editing or adding a migration:

```
docker compose down -v && docker compose up
```

**Supabase (hosted):** it has no auto-apply, so push explicitly:

```
supabase db push --db-url "$DATABASE_URL"
```

Do not run `supabase db push` against the local compose Postgres — the entrypoint
already applied the files without recording them in `supabase_migrations`, so the
push re-runs them and fails on the first `create table`.

## Planned schema

```
users                    mirror of Clerk identities
  id                     text primary key   -- Clerk user ID
  email                  text not null      -- Clerk instance requires email sign-up
  created_at             timestamptz

connected_services       dead as of the OAuth cut — dropped in T9
                         (kept here only until that migration lands)
  id                     uuid primary key
  user_id                text references users(id) on delete cascade
  provider               text               -- 'google', 'youtube', ...
  encrypted_access_token bytea              -- never plaintext
  encrypted_refresh_token bytea             -- never plaintext
  expires_at             timestamptz
  scopes                 text[]
  created_at             timestamptz
  unique (user_id, provider)

conversations
  id                     uuid primary key
  user_id                text references users(id) on delete cascade
  title                  text
  created_at             timestamptz

messages
  id                     uuid primary key
  conversation_id        uuid references conversations(id) on delete cascade
  role                   text               -- 'user' | 'assistant'
  content                text
  created_at             timestamptz
```

## Rules

- **No table holds a credential.** Holster stores no secret belonging to any other
  service, for any user. There is no encryption layer because there is nothing to
  encrypt — see *No OAuth, and no credential architecture* in `../DECISIONS.md`.
- **Row-level security on every user-scoped table**, keyed on the Clerk user ID.
- `on delete cascade` throughout, so deleting a user genuinely removes their rows.
