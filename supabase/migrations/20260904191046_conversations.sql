-- T20: conversation persistence. One thread per user until T20.5 adds the
-- ability to start a new one or list past ones — `unique (user_id)` makes
-- that a database-enforced invariant, not just today's only code path.
-- Without it, two concurrent WS connections for one account (two tabs) could
-- each see no existing conversation and race to create their own, and a
-- transient load failure could look identical to "brand-new user" and do the
-- same (see conversations.go's loadConversation/saveMessages) — either way
-- silently forking one account's history across two rows that nothing
-- would ever read together correctly again. Dropping this constraint is the
-- easy part of T20.5's multi-conversation support and migrates no data
-- (every row created under it already satisfies a dropped constraint); the
-- harder part is that T20.5 also has to thread an explicit conversation id
-- through conversations.go's loadConversation/loadConversationTurns/
-- saveMessages and their WS call sites in chat.go, none of which take one
-- today — this constraint isn't standing in for that work.
--
-- title is nullable and unused until T20.5 generates one from the first
-- exchange; the column is declared now because ARCHITECTURE.md's data model
-- already names it and adding it later would be a second migration for no
-- reason.
create table conversations (
  id         uuid primary key default gen_random_uuid(),
  user_id    text not null references users (id) on delete cascade,
  title      text,
  created_at timestamptz not null default now(),
  unique (user_id)
);

-- content is the clean text shown to the user; title_refs is the other half
-- of DECISIONS.md's "One socket per session" entry — "What is persisted:
-- clean text and the title IDs shown. Never tool-call scaffolding." Null for
-- every user row and for an assistant row whose turn showed no picks.
--
-- seq, not created_at, is what conversations.go's readers order by: saveMessages
-- inserts a turn's user and assistant row in one statement, so both get the
-- exact same now() value and created_at alone cannot tell them apart -- two
-- rows with a tied timestamp have no guaranteed order. An identity column has
-- no such tie; created_at stays for its own sake (display, "created_at asc"
-- reads elsewhere), not as a sort key.
create table messages (
  id              uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references conversations (id) on delete cascade,
  role            text not null check (role in ('user', 'assistant')),
  content         text not null,
  title_refs      jsonb,
  created_at      timestamptz not null default now(),
  seq             bigint generated always as identity
);

-- Not auto-created by the foreign key. Both loadConversationTurns' read and
-- the messages RLS policy below (its subquery joins through conversation_id)
-- hit this on every call; seq is the second column because every such read
-- also orders by it.
create index on messages (conversation_id, seq);

alter table conversations enable row level security;
alter table messages enable row level security;

-- Same shape as T10 (20260831233121_rls_roles.sql) and T18
-- (20260903002450_verdicts.sql): gateway_app is the sole writer, scoped per-
-- user via holster.user_id. agent_ro's select is the first grant it's ever
-- had — T10 gave it none, deferring conversation-table reads to "the task
-- that needs them," naming this one. Nothing in the agent connects to the
-- database yet (it still only sees what the gateway hands it per turn, same
-- as verdicts); this grant exists so that when it does, per-user scoping is
-- already correct rather than bolted on later.
grant select, insert, update, delete on conversations, messages to gateway_app;
grant select on conversations, messages to agent_ro;

create policy conversation_isolation on conversations
  for all to gateway_app
  using (user_id = nullif(current_setting('holster.user_id', true), ''))
  with check (user_id = nullif(current_setting('holster.user_id', true), ''));

create policy conversation_read on conversations
  for select to agent_ro
  using (user_id = nullif(current_setting('holster.user_id', true), ''));

-- messages carries no user_id of its own, so both policies join through
-- conversation_id — the only user-scoped table in this schema whose
-- isolation predicate isn't a direct column comparison.
create policy message_isolation on messages
  for all to gateway_app
  using (conversation_id in (
    select id from conversations
    where user_id = nullif(current_setting('holster.user_id', true), '')
  ))
  with check (conversation_id in (
    select id from conversations
    where user_id = nullif(current_setting('holster.user_id', true), '')
  ));

create policy message_read on messages
  for select to agent_ro
  using (conversation_id in (
    select id from conversations
    where user_id = nullif(current_setting('holster.user_id', true), '')
  ));
