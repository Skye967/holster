# db

Supabase (PostgreSQL). Schema reference for the tables below.

Migrations live in `supabase/migrations/` at the repo root, not here — that is the
only path the Supabase CLI discovers. Filenames are timestamp-prefixed and applied
in that order:

```
supabase db push --db-url "$DATABASE_URL"
```

## Planned schema

```
users                    mirror of Clerk identities
  id                     text primary key   -- Clerk user ID
  email                  text
  created_at             timestamptz

connected_services       one row per user per authorized provider
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

- **Token columns are `bytea` and always ciphertext.** Encryption happens in
  `services/credentials` before insert. No plaintext token is ever written, and no
  migration should add a `text` token column.
- **Row-level security on every user-scoped table**, keyed on the Clerk user ID.
  Encryption is the second layer; RLS is the first.
- `on delete cascade` throughout, so deleting a user genuinely removes their tokens.
