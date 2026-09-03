-- One row per user per title. tmdb_id alone is not a title -- TMDB's numeric IDs
-- are only unique within a media type (movie 550 and tv 550 are unrelated), so
-- the composite key includes media_type. Set membership like
-- streaming_subscriptions: the primary key index also serves "all verdicts for
-- a user", which is what the chat context's "recent verdicts" needs (T19).
--
-- Four verdicts are judgments, set once and never changed. want_to_watch is the
-- exception -- an intention expected to later become seen via an update. Nothing
-- in this migration enforces "the other four never change"; that's an
-- application rule for whichever task builds the write path (T18/T19), not a
-- schema constraint.
create table title_verdicts (
  user_id    text not null references users (id) on delete cascade,
  tmdb_id    integer not null,
  media_type text not null check (media_type in ('movie', 'tv')),
  verdict    text not null
    check (verdict in ('liked', 'disliked', 'seen', 'not_interested', 'want_to_watch')),
  created_at timestamptz not null default now(),
  primary key (user_id, tmdb_id, media_type)
);

alter table title_verdicts enable row level security;

-- Same shape as T10 (20260831233121_rls_roles.sql): gateway_app is the sole
-- writer, scoped per-user via holster.user_id.
grant select, insert, update, delete on title_verdicts to gateway_app;

create policy verdict_isolation on title_verdicts
  for all to gateway_app
  using (user_id = nullif(current_setting('holster.user_id', true), ''))
  with check (user_id = nullif(current_setting('holster.user_id', true), ''));
