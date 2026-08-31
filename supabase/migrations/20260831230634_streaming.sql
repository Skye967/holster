-- Streaming services as a preference, not a connection.
--
-- The OAuth cut (../DECISIONS.md) removed the credential architecture whole: no
-- table holds a secret belonging to another service. connected_services was its
-- last trace and is dropped here. The "token columns are bytea / ciphertext"
-- header on 20260824233907_init.sql describes a table that no longer exists --
-- that file is left untouched, since a migration is immutable once applied; this
-- comment supersedes it.

drop table connected_services;

-- Availability is country-specific, so every catalog query will carry
-- watch_region (ISO 3166-1 alpha-2). Hardcoded 'US' until onboarding asks it
-- (T15.5); the column exists now so that is a default change, not a schema one.
alter table users
  add column country text not null default 'US'
    check (country ~ '^[A-Z]{2}$');

-- What the user subscribes to. Self-reported, holds no secret. Values are TMDB
-- provider IDs (Netflix = 8, ...). Composite key: set membership, one row per
-- ticked service; the primary key index also serves "all subscriptions for a
-- user".
create table streaming_subscriptions (
  user_id          text not null references users (id) on delete cascade,
  tmdb_provider_id integer not null,
  created_at       timestamptz not null default now(),
  primary key (user_id, tmdb_provider_id)
);

-- Cache of TMDB's watch-provider list, one row per country. Refreshed lazily on
-- read when older than 24h, served stale if TMDB is unreachable (T15). Read and
-- written as a unit, so one jsonb blob rather than a normalised table anyone
-- maintains. providers shape:
--   [{"provider_id": int, "provider_name": text, "logo_path": text, "display_priority": int}]
create table streaming_providers (
  country    text primary key,
  providers  jsonb not null,
  fetched_at timestamptz not null default now()
);

-- Deny-by-default via PostgREST until T10 adds roles and policies; the gateway
-- connects as owner and is unaffected. streaming_providers is not user-scoped
-- but is still not for the browser to read directly.
alter table streaming_subscriptions enable row level security;
alter table streaming_providers enable row level security;
