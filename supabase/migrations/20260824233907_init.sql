-- Users and their connected services.
--
-- Token columns are bytea and hold ciphertext only. Encryption happens in
-- services/credentials before insert; no plaintext token is ever written.

create table users (
  id         text primary key,           -- Clerk user ID
  email      text not null,
  created_at timestamptz not null default now()
);

create table connected_services (
  id                      uuid primary key default gen_random_uuid(),
  user_id                 text not null references users (id) on delete cascade,
  provider                text not null,  -- 'google', 'youtube', ...
  encrypted_access_token  bytea not null,
  -- Nullable: providers return a refresh token on first consent only.
  encrypted_refresh_token bytea,
  -- Nullable: not every provider expires its access token.
  expires_at              timestamptz,
  scopes                  text[] not null default '{}',
  created_at              timestamptz not null default now(),
  unique (user_id, provider)
);

-- Deny-by-default until T10 adds policies. Supabase exposes public tables
-- through PostgREST, so tables without RLS are readable by anyone holding
-- the publishable key.
alter table users enable row level security;
alter table connected_services enable row level security;
