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

Skip the `down -v` and a newly added migration never runs, so anything it
introduced is missing — the `db-init` service then fails on the absent roles and
the gateway will not start. That is the intended loud failure for a stale schema;
`down -v` is the fix.

**Supabase (hosted):** it has no auto-apply, so push explicitly:

```
supabase db push --db-url "$DATABASE_URL"
```

Do not run `supabase db push` against the local compose Postgres — the entrypoint
already applied the files without recording them in `supabase_migrations`, so the
push re-runs them and fails on the first `create table`.

## Roles

The T10 migration creates two service roles, `gateway_app` (the sole writer) and
`agent_ro` (no write grant anywhere, and no read grants yet — later tasks add
them), both `NOLOGIN`. A committed migration runs everywhere, so it carries no
password.

- **Local:** compose's `db-init` service applies `db/local-roles.sql` once
  Postgres is healthy, granting both roles a login with the throwaway compose
  password. The gateway waits for it.
- **Supabase:** grant the login and set a password once, in the SQL editor or
  dashboard:
  ```
  alter role gateway_app with login password '…';
  alter role agent_ro   with login password '…';
  ```
  Then point `DATABASE_URL` at `gateway_app` and `DATABASE_URL_AGENT_READONLY` at
  `agent_ro`.

## Planned schema

```
users                    mirror of Clerk identities
  id                     text primary key   -- Clerk user ID
  email                  text not null      -- Clerk instance requires email sign-up
  country                text not null      -- ISO 3166-1 alpha-2, defaults 'US'
  created_at             timestamptz

streaming_subscriptions  one row per service the user subscribes to
  user_id                text references users(id) on delete cascade
  tmdb_provider_id       integer            -- TMDB watch-provider ID (Netflix = 8)
  created_at             timestamptz
  primary key (user_id, tmdb_provider_id)

streaming_providers      cache of TMDB's watch-provider list, one row per country
  country                text primary key
  providers              jsonb              -- [{provider_id, provider_name,
                                            --   logo_path, display_priority}]
  fetched_at             timestamptz        -- lazy-refreshed past 24h (T15)

title_verdicts           one row per user per title
  user_id                text references users(id) on delete cascade
  tmdb_id                integer            -- unique only within a media_type
  media_type             text               -- 'movie' | 'tv'
  verdict                text               -- liked | disliked | seen |
                                            -- not_interested | want_to_watch
  created_at             timestamptz
  primary key (user_id, tmdb_id, media_type)

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
