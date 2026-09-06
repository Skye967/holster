-- T20.5: multiple conversations per user, listed and deletable. Drops the
-- `unique (user_id)` constraint T20's migration (20260904191046_conversations.sql)
-- already flagged as "the easy part" of this task — conversations.go's
-- saveMessages now inserts one row per client-generated conversation id
-- (on conflict (id) do nothing) instead of upserting a single per-user row,
-- so two concurrent "new chat" threads for one account are two rows by
-- design, not a race to prevent.
alter table conversations drop constraint conversations_user_id_key;

-- Dropping the constraint above also drops the btree index Postgres was
-- backing it with. Both the sidebar list (loadConversationSummaries) and
-- conversation_isolation/message_isolation's own subqueries filter
-- conversations by user_id, so this replaces it rather than leaving those on
-- a sequential scan. created_at desc matches the sidebar's own ordering.
create index on conversations (user_id, created_at desc);
