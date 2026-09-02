package main

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// testPool connects to TEST_DATABASE_URL, or skips. Kept separate from
// DATABASE_URL so running the suite can never write to a configured production
// database by accident.
func testPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("TEST_DATABASE_URL")
	if url == "" {
		t.Skip("TEST_DATABASE_URL not set; skipping database tests")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	if err := pool.Ping(ctx); err != nil {
		t.Fatalf("ping: %v", err)
	}
	t.Cleanup(pool.Close)
	return pool
}

// rowVersion reports the transaction that last wrote a row. An UPDATE gives the
// row a new xmin; a suppressed upsert only takes a lock, which stamps xmax.
// Read as text — xmin is the xid type, which pgx has no default codec for.
func rowVersion(t *testing.T, pool *pgxpool.Pool, id string) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var xmin string
	if err := pool.QueryRow(ctx,
		`select xmin::text from users where id = $1`, id).Scan(&xmin); err != nil {
		t.Fatalf("row version: %v", err)
	}
	return xmin
}

func TestUpsertUser(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	ensure := upsertUser(pool)

	const id = "user_upsert_test"
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := pool.Exec(cleanupCtx, `delete from users where id = $1`, id); err != nil {
			t.Errorf("cleanup: %v", err)
		}
	})
	if _, err := pool.Exec(ctx, `delete from users where id = $1`, id); err != nil {
		t.Fatal(err)
	}

	email := func() string {
		var e string
		if err := pool.QueryRow(ctx, `select email from users where id = $1`, id).Scan(&e); err != nil {
			t.Fatalf("select: %v", err)
		}
		return e
	}
	count := func() int {
		var n int
		if err := pool.QueryRow(ctx, `select count(*) from users where id = $1`, id).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}

	// First sight creates exactly one row — T8's done-when.
	if err := ensure(ctx, id, "first@example.com"); err != nil {
		t.Fatalf("first upsert: %v", err)
	}
	if n := count(); n != 1 {
		t.Fatalf("row count = %d, want 1", n)
	}

	// Repeats are idempotent and, crucially, rewrite no tuple. Delete the
	// IS DISTINCT FROM guard from upsertUser and this fails — which is the whole
	// point of asserting it.
	versionBefore := rowVersion(t, pool, id)
	for range 5 {
		if err := ensure(ctx, id, "first@example.com"); err != nil {
			t.Fatalf("repeat upsert: %v", err)
		}
	}
	if n := count(); n != 1 {
		t.Errorf("row count = %d after repeats, want 1", n)
	}
	if v := rowVersion(t, pool, id); v != versionBefore {
		t.Errorf("no-op upserts rewrote the row (xmin %s -> %s), want no write", versionBefore, v)
	}

	// A changed email is picked up, so Clerk stays the source of truth.
	if err := ensure(ctx, id, "second@example.com"); err != nil {
		t.Fatalf("changed upsert: %v", err)
	}
	if got := email(); got != "second@example.com" {
		t.Errorf("email = %q, want %q", got, "second@example.com")
	}
	if n := count(); n != 1 {
		t.Errorf("row count = %d after email change, want 1", n)
	}
}

// RLS must stay enabled on users: on hosted Supabase, PostgREST exposes public
// tables, so one without RLS is readable by anyone holding the anon key.
//
// This asserts the switch is on, not the policy behaviour behind it. The gateway
// connects as gateway_app -- a non-owner role, so RLS applies -- and sets
// holster.user_id per request; TestGatewayRoleEnforcesUserIsolation covers that
// path. TestUpsertUser runs as the TEST_DATABASE_URL superuser, which bypasses
// RLS, so it exercises upsert semantics only.
func TestRLSEnabledOnUsers(t *testing.T) {
	pool := testPool(t)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Qualified: a Supabase database also has auth.users, and an unqualified
	// relname would let this pass on that row while public.users has RLS off.
	var enabled bool
	err := pool.QueryRow(ctx,
		`select relrowsecurity from pg_class where oid = to_regclass('public.users')`).Scan(&enabled)
	if err != nil {
		t.Fatal(err)
	}
	if !enabled {
		t.Error("row level security is not enabled on users")
	}
}

// asRole runs fn inside a rolled-back transaction with the session role dropped
// to role. The suite connects as the migration/superuser role, which bypasses
// RLS and every grant; SET ROLE is what makes the T10 policies and grants under
// test actually apply. Nothing is committed.
func asRole(t *testing.T, pool *pgxpool.Pool, role string, fn func(context.Context, pgx.Tx)) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	defer tx.Rollback(ctx)

	// role is a test constant, never user input.
	if _, err := tx.Exec(ctx, "set role "+role); err != nil {
		t.Fatalf("set role %s: %v", role, err)
	}
	fn(ctx, tx)
}

// denied reports whether err is a permission-denied error (SQLSTATE 42501).
func denied(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "42501"
}

// probe runs one statement against a savepoint and rolls it back, so a
// permission failure does not abort the surrounding transaction and the next
// probe still runs.
func probe(t *testing.T, ctx context.Context, tx pgx.Tx, sql string) error {
	t.Helper()
	sp, err := tx.Begin(ctx)
	if err != nil {
		t.Fatalf("savepoint: %v", err)
	}
	_, execErr := sp.Exec(ctx, sql)
	_ = sp.Rollback(ctx)
	return execErr
}

// loadChatContext (chat.go) reads what T14's "gateway loads subscriptions,
// country and recent verdicts" actually has available today: subscriptions
// and country are real tables; the provider-name lookup must degrade to an
// empty list, not an error, when streaming_providers has no row yet for the
// country (T15 owns keeping that cache fresh — see chat.go's loadChatContext).
func TestLoadChatContext(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	load := loadChatContext(pool)

	const id = "chat_ctx_test"
	t.Cleanup(func() {
		c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		pool.Exec(c, `delete from users where id = $1`, id)
		pool.Exec(c, `delete from streaming_providers where country = 'ZZ'`)
	})

	if _, err := pool.Exec(ctx,
		`insert into users (id, email, country) values ($1, $2, 'ZZ')
		 on conflict (id) do update set country = excluded.country`,
		id, id+"@example.com"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx,
		`insert into streaming_subscriptions (user_id, tmdb_provider_id) values ($1, 8)
		 on conflict do nothing`, id); err != nil {
		t.Fatal(err)
	}

	// No streaming_providers row for 'ZZ' yet: names must come back empty,
	// not an error.
	cc, err := load(ctx, id)
	if err != nil {
		t.Fatalf("load with no provider cache: %v", err)
	}
	if cc.Region != "ZZ" || len(cc.Providers) != 1 || cc.Providers[0] != 8 {
		t.Errorf("cc = %+v, want Region=ZZ Providers=[8]", cc)
	}
	if len(cc.ProviderNames) != 0 {
		t.Errorf("ProviderNames = %v, want empty with no cache row", cc.ProviderNames)
	}

	// Now seed the cache and confirm the name resolves.
	if _, err := pool.Exec(ctx,
		`insert into streaming_providers (country, providers) values ('ZZ', $1)
		 on conflict (country) do update set providers = excluded.providers`,
		`[{"provider_id": 8, "provider_name": "Netflix"}]`); err != nil {
		t.Fatal(err)
	}

	cc, err = load(ctx, id)
	if err != nil {
		t.Fatalf("load with provider cache: %v", err)
	}
	if len(cc.ProviderNames) != 1 || cc.ProviderNames[0] != "Netflix" {
		t.Errorf("ProviderNames = %v, want [Netflix]", cc.ProviderNames)
	}
}

// T10's core invariant: agent_ro has no write path to any table. It also has no
// read grants yet -- those are added by the task that needs each one -- but
// "never writes" is the property that must not regress, so that is what this
// pins.
func TestAgentRoleIsReadOnly(t *testing.T) {
	pool := testPool(t)

	writes := map[string]string{
		"insert users":        `insert into users (id, email) values ('agent_probe', 'x@example.com')`,
		"update users":        `update users set email = 'y@example.com' where id = 'nobody'`,
		"delete users":        `delete from users where id = 'nobody'`,
		"insert subscription": `insert into streaming_subscriptions (user_id, tmdb_provider_id) values ('agent_probe', 8)`,
		"delete subscription": `delete from streaming_subscriptions where user_id = 'nobody'`,
		"insert provider":     `insert into streaming_providers (country, providers) values ('ZZ', '[]'::jsonb)`,
		"update provider":     `update streaming_providers set fetched_at = now() where country = 'ZZ'`,
	}
	asRole(t, pool, "agent_ro", func(ctx context.Context, tx pgx.Tx) {
		for name, sql := range writes {
			if err := probe(t, ctx, tx, sql); !denied(err) {
				t.Errorf("%s: err = %v, want SQLSTATE 42501", name, err)
			}
		}
	})
}

// T10's first half: under gateway_app, holster.user_id fences every user-scoped
// table -- a user sees only their rows and cannot write rows tagged with another
// user's id.
func TestGatewayRoleEnforcesUserIsolation(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	const uA, uB = "rls_test_a", "rls_test_b"
	t.Cleanup(func() {
		c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := pool.Exec(c, `delete from users where id = any($1)`, []string{uA, uB}); err != nil {
			t.Errorf("cleanup: %v", err)
		}
	})
	// Seed as the owner, which bypasses RLS.
	for _, u := range []string{uA, uB} {
		if _, err := pool.Exec(ctx,
			`insert into users (id, email) values ($1, $2) on conflict (id) do nothing`,
			u, u+"@example.com"); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx,
			`insert into streaming_subscriptions (user_id, tmdb_provider_id) values ($1, 8)
			 on conflict do nothing`, u); err != nil {
			t.Fatal(err)
		}
	}

	asRole(t, pool, "gateway_app", func(ctx context.Context, tx pgx.Tx) {
		if _, err := tx.Exec(ctx, `select set_config('holster.user_id', $1, true)`, uA); err != nil {
			t.Fatalf("set_config: %v", err)
		}

		var users []string
		rows, err := tx.Query(ctx, `select id from users where id = any($1) order by id`,
			[]string{uA, uB})
		if err != nil {
			t.Fatal(err)
		}
		for rows.Next() {
			var id string
			if err := rows.Scan(&id); err != nil {
				t.Fatal(err)
			}
			users = append(users, id)
		}
		rows.Close()
		if len(users) != 1 || users[0] != uA {
			t.Errorf("as %s, visible users = %v, want [%s]", uA, users, uA)
		}

		var subs int
		if err := tx.QueryRow(ctx,
			`select count(*) from streaming_subscriptions where user_id = any($1)`,
			[]string{uA, uB}).Scan(&subs); err != nil {
			t.Fatal(err)
		}
		if subs != 1 {
			t.Errorf("as %s, visible subscriptions = %d, want 1", uA, subs)
		}

		// A write for the acting user passes WITH CHECK...
		if err := probe(t, ctx, tx,
			`insert into streaming_subscriptions (user_id, tmdb_provider_id) values ('`+uA+`', 99)`); err != nil {
			t.Errorf("as %s, writing its own subscription: %v", uA, err)
		}
		// ...including the on-conflict upsert path upsertUser runs.
		if err := probe(t, ctx, tx, `
			insert into users (id, email) values ('`+uA+`', 'updated@example.com')
			on conflict (id) do update set email = excluded.email
			where users.email is distinct from excluded.email`); err != nil {
			t.Errorf("as %s, upserting its own users row: %v", uA, err)
		}
		// ...but a write for another user is rejected by WITH CHECK.
		if err := probe(t, ctx, tx,
			`insert into streaming_subscriptions (user_id, tmdb_provider_id) values ('`+uB+`', 9)`); !denied(err) {
			t.Errorf("writing a row for %s while acting as %s: err = %v, want SQLSTATE 42501", uB, uA, err)
		}
	})
}

// The fail-closed half: gateway_app with holster.user_id unset sees no rows and
// can write none. A handler that forgets withUser's set_config gets an empty
// result or a permission error -- never another user's data. This is what the
// empty-string guard on current_setting in the policies buys.
func TestGatewayRoleDeniesWithoutUserContext(t *testing.T) {
	pool := testPool(t)

	// A row must exist for "sees no rows" to mean anything.
	ctx := context.Background()
	const seed = "rls_no_context_seed"
	if _, err := pool.Exec(ctx,
		`insert into users (id, email) values ($1, $2) on conflict (id) do nothing`,
		seed, seed+"@example.com"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := pool.Exec(c, `delete from users where id = $1`, seed); err != nil {
			t.Errorf("cleanup: %v", err)
		}
	})

	asRole(t, pool, "gateway_app", func(ctx context.Context, tx pgx.Tx) {
		var n int
		if err := tx.QueryRow(ctx, `select count(*) from users`).Scan(&n); err != nil {
			t.Fatalf("select users: %v", err)
		}
		if n != 0 {
			t.Errorf("with no holster.user_id, visible users = %d, want 0", n)
		}

		if err := probe(t, ctx, tx,
			`insert into users (id, email) values ('rls_no_context', 'x@example.com')`); !denied(err) {
			t.Errorf("insert with no holster.user_id: err = %v, want SQLSTATE 42501", err)
		}
	})
}
