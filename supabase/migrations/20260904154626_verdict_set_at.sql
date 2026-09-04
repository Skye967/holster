-- verdict_set_at tracks when a verdict was last written -- including a
-- want_to_watch -> liked transition -- unlike created_at, which is only the
-- row's first save and is deliberately left alone by every later write
-- (see 20260903002450_verdicts.sql). This is what T19's taste hint should
-- have been ordering by all along (T19.5).
--
-- `not null default now()` in one statement, not a nullable column locked
-- down after: Postgres satisfies NOT NULL on existing rows from the default
-- (metadata-only, no rewrite) without needing a separate constraint step, and
-- still sets the default new rows get going forward. The update below then
-- overwrites every pre-existing row with its real created_at -- backfilling
-- from the only prior signal those rows have, rather than leaving them all
-- stamped with this migration's apply time.
alter table title_verdicts add column verdict_set_at timestamptz not null default now();
update title_verdicts set verdict_set_at = created_at;
