package main

import (
	"context"
	"os"
	"testing"
	"time"

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

// RLS must stay enabled on users: Supabase exposes tables through PostgREST, so
// one without it is readable by anyone holding the publishable key.
//
// Deliberately does not assert how the gateway gets to write. It currently does
// so as the table owner, which bypasses RLS, but pinning that would turn T10's
// hardening into a red test that reads like a regression. TestUpsertUser already
// proves writes work, by outcome, under whatever role is configured.
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
