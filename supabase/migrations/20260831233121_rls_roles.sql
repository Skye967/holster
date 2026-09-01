-- T10: least-privilege service roles, and per-user row-level security.
--
-- The earlier migrations enabled RLS on every table but wrote no policies, so
-- everything was deny-all. This adds the two service roles and the policies that
-- let the gateway through. See ../../DECISIONS.md for why the gateway is the only
-- writer and the agent never writes.
--
-- Both roles are created NOLOGIN. LOGIN and a password are granted per
-- environment, never here: a committed migration runs on every database and must
-- carry no credential. Local dev does it in db/local-roles.sql; Supabase does it
-- in the dashboard (see db/README.md).

-- Roles are cluster-global -- they outlive the database and survive
-- `supabase db reset`, which re-runs every migration. `create role` alone would
-- fail with "role already exists" on the second run; `create table` needs no
-- such guard because it is database-scoped.
do $$
begin
  if not exists (select from pg_roles where rolname = 'gateway_app') then
    create role gateway_app nologin;
  end if;
  if not exists (select from pg_roles where rolname = 'agent_ro') then
    create role agent_ro nologin;
  end if;
end $$;

grant usage on schema public to gateway_app, agent_ro;

-- gateway_app: the sole writer. Full DML on the app tables and nothing else --
-- not the table owner, so a gateway compromise still cannot drop or alter a
-- table, create objects, or read pg_authid.
grant select, insert, update, delete
  on users, streaming_subscriptions, streaming_providers
  to gateway_app;

-- agent_ro: the least-trusted component -- it runs model-directed control flow
-- over text anyone can edit. T10 gives it no grants at all; what it must never
-- have is a write path, and that is asserted by TestAgentRoleIsReadOnly. Reads
-- are granted by the task that needs them (T20 for the conversation tables),
-- keeping the surface as narrow as the work in hand.

-- Per-user isolation. At the start of each request the gateway opens a
-- transaction and runs
--   select set_config('holster.user_id', <clerk id>, true)
-- and every policy below pins rows to that value. nullif(current_setting(...,
-- true), '') is NULL both when the GUC was never set and when it reverted to
-- empty, so id = NULL matches nothing -- a forgotten set_config denies rather
-- than exposes.
--
-- This is defence in depth behind the handler's own `where user_id = $1`, not a
-- replacement for it: it catches a query that forgets the filter. The browser
-- never reaches the database. On hosted Supabase, PostgREST stays denied by the
-- absence of any policy for anon/authenticated.
--
-- Deliberately not FORCE ROW LEVEL SECURITY: migrations and admin tasks connect
-- as the table owner and are meant to bypass. gateway_app is a non-owner role,
-- so policies apply to it without forcing.

create policy user_isolation on users
  for all to gateway_app
  using (id = nullif(current_setting('holster.user_id', true), ''))
  with check (id = nullif(current_setting('holster.user_id', true), ''));

create policy subscription_isolation on streaming_subscriptions
  for all to gateway_app
  using (user_id = nullif(current_setting('holster.user_id', true), ''))
  with check (user_id = nullif(current_setting('holster.user_id', true), ''));

-- streaming_providers is a global cache keyed by country, not by user. The
-- gateway reads and refreshes every country's row, so there is no per-user
-- predicate to apply -- only the grant and RLS being enabled matter here.
create policy provider_access on streaming_providers
  for all to gateway_app
  using (true) with check (true);
