-- Local development only.
--
-- Compose's `db-init` service applies this once Postgres is healthy, giving the
-- service roles a login and the throwaway compose password. The migration
-- creates them NOLOGIN so that no committed file carries a credential; on
-- Supabase these are set in the dashboard instead. ALTER ROLE is idempotent.

alter role gateway_app with login password 'password';
alter role agent_ro with login password 'password';
