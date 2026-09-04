package main

import (
	"context"
	"errors"
	"os"
	"slices"
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

// loadChatContext (chat.go) reads the subscriptions-and-country half of the
// context; verdicts are loaded separately by runTurn — see chat.go's
// chatContext for why. The provider-name lookup must degrade to an empty
// list, not an error, when streaming_providers has no row yet for the
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

	// Zero subscriptions: Providers must come back as [], not nil — the
	// subscriptions handler (providers.go) marshals this slice directly to
	// JSON, and nil marshals to `null`.
	cc, err := load(ctx, id)
	if err != nil {
		t.Fatalf("load with no subscriptions: %v", err)
	}
	if cc.Providers == nil || len(cc.Providers) != 0 {
		t.Errorf("Providers = %#v, want non-nil empty slice", cc.Providers)
	}

	if _, err := pool.Exec(ctx,
		`insert into streaming_subscriptions (user_id, tmdb_provider_id) values ($1, 8)
		 on conflict do nothing`, id); err != nil {
		t.Fatal(err)
	}

	// No streaming_providers row for 'ZZ' yet: names must come back empty,
	// not an error.
	cc, err = load(ctx, id)
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

// loadVerdicts (verdicts.go) serves two consumers: the browser's hydration of
// GET /api/verdicts, and the chat context runTurn hands the agent (T19). The
// ordering is what the agent depends on -- it keeps only the first few liked
// titles and has no timestamp of its own to sort by.
func TestLoadVerdicts(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	load := loadVerdicts(pool)

	const id = "verdict_load_test"
	// Before as well as after, like TestUpsertUser: a run killed between the
	// seed and the cleanup would otherwise leave rows behind and every later
	// run would fail on the "nil before any are set" assertion, pointing at
	// loadVerdicts rather than at the leftovers.
	if _, err := pool.Exec(ctx, `delete from users where id = $1`, id); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		// title_verdicts cascades on user delete.
		pool.Exec(c, `delete from users where id = $1`, id)
	})

	if _, err := pool.Exec(ctx,
		`insert into users (id, email, country) values ($1, $2, 'ZZ')
		 on conflict (id) do nothing`, id, id+"@example.com"); err != nil {
		t.Fatal(err)
	}

	// Nothing set yet: nil, not an error. The handler is what turns that into
	// [] for the browser (see TestGetVerdictsReturnsEmptyArray).
	vs, err := load(ctx, id)
	if err != nil {
		t.Fatalf("load with no verdicts: %v", err)
	}
	// nil specifically, not just empty: verdicts.go's handler relies on this
	// to substitute [] for the browser (TestGetVerdictsReturnsEmptyArray),
	// and len() == 0 would pass for an eagerly-allocated slice too, leaving
	// that branch dead with nothing to flag it.
	if vs != nil {
		t.Errorf("Verdicts = %+v, want nil before any are set", vs)
	}

	// Explicit, distinct created_at values rather than the column default:
	// now() is transaction start time, so a multi-row insert stamps every row
	// identically and "newest first" would be asserting on heap order.
	if _, err := pool.Exec(ctx,
		`insert into title_verdicts (user_id, tmdb_id, media_type, verdict, created_at)
		 values ($1, 101, 'movie', 'seen', now() - interval '2 hours'),
		        ($1, 202, 'tv', 'liked', now() - interval '1 hour'),
		        ($1, 303, 'movie', 'want_to_watch', now())
		 on conflict do nothing`, id); err != nil {
		t.Fatal(err)
	}

	vs, err = load(ctx, id)
	if err != nil {
		t.Fatalf("load with verdicts: %v", err)
	}
	want := []Verdict{
		{TMDBID: 303, MediaType: "movie", Verdict: "want_to_watch"},
		{TMDBID: 202, MediaType: "tv", Verdict: "liked"},
		{TMDBID: 101, MediaType: "movie", Verdict: "seen"},
	}
	if !slices.Equal(vs, want) {
		t.Errorf("Verdicts = %+v, want %+v", vs, want)
	}

	// The tie-break, exercised the only way ties actually arise: one statement
	// writing several rows, so now() stamps them identically. Without the
	// tie-break their order is whatever the heap scan returns.
	if _, err := pool.Exec(ctx,
		`insert into title_verdicts (user_id, tmdb_id, media_type, verdict, created_at)
		 values ($1, 505, 'movie', 'liked', now() + interval '1 hour'),
		        ($1, 404, 'movie', 'liked', now() + interval '1 hour')
		 on conflict do nothing`, id); err != nil {
		t.Fatal(err)
	}

	vs, err = load(ctx, id)
	if err != nil {
		t.Fatalf("load with tied timestamps: %v", err)
	}
	if len(vs) < 2 || vs[0].TMDBID != 404 || vs[1].TMDBID != 505 {
		t.Errorf("tied rows = %+v, want 404 before 505 (tmdb_id ascending)", vs[:min(2, len(vs))])
	}
}

// loadProviders (providers.go, T15): no cache refreshes and caches; a fresh
// cache is served without another refresh; a stale cache refreshes and
// overwrites; a failing refresh degrades to whatever is cached.
func TestLoadProviders(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	const country = "YY"
	t.Cleanup(func() {
		c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		pool.Exec(c, `delete from streaming_providers where country = $1`, country)
	})
	if _, err := pool.Exec(ctx, `delete from streaming_providers where country = $1`, country); err != nil {
		t.Fatal(err)
	}

	var refreshCalls int
	fresh := []Provider{{ProviderID: 8, ProviderName: "Netflix", DisplayPriority: 1}}
	load := loadProviders(pool, func(context.Context, string) ([]Provider, error) {
		refreshCalls++
		return fresh, nil
	})

	got, err := load(ctx, country)
	if err != nil {
		t.Fatalf("load with no cache: %v", err)
	}
	if len(got) != 1 || got[0].ProviderID != 8 {
		t.Errorf("got = %+v, want the refreshed list", got)
	}
	if refreshCalls != 1 {
		t.Errorf("refreshCalls = %d, want 1", refreshCalls)
	}

	got, err = load(ctx, country)
	if err != nil {
		t.Fatalf("load with fresh cache: %v", err)
	}
	if len(got) != 1 {
		t.Errorf("got = %+v, want the cached list", got)
	}
	if refreshCalls != 1 {
		t.Errorf("refreshCalls = %d after a fresh read, want still 1", refreshCalls)
	}

	if _, err := pool.Exec(ctx,
		`update streaming_providers set fetched_at = now() - interval '25 hours' where country = $1`,
		country); err != nil {
		t.Fatal(err)
	}
	stale := []Provider{{ProviderID: 337, ProviderName: "Disney Plus", DisplayPriority: 1}}
	load = loadProviders(pool, func(context.Context, string) ([]Provider, error) {
		refreshCalls++
		return stale, nil
	})
	got, err = load(ctx, country)
	if err != nil {
		t.Fatalf("load with stale cache: %v", err)
	}
	if len(got) != 1 || got[0].ProviderID != 337 {
		t.Errorf("got = %+v, want the newly-refreshed list", got)
	}
	if refreshCalls != 2 {
		t.Errorf("refreshCalls = %d, want 2 after a stale read", refreshCalls)
	}

	if _, err := pool.Exec(ctx,
		`update streaming_providers set fetched_at = now() - interval '25 hours' where country = $1`,
		country); err != nil {
		t.Fatal(err)
	}
	failing := loadProviders(pool, func(context.Context, string) ([]Provider, error) {
		return nil, errors.New("agent unreachable")
	})
	got, err = failing(ctx, country)
	if err != nil {
		t.Fatalf("load with a failing refresh: %v", err)
	}
	if len(got) != 1 || got[0].ProviderID != 337 {
		t.Errorf("got = %+v, want the stale cached list on refresh failure", got)
	}
}

// A country with neither a cache row nor a working agent gets an empty list,
// not an error — the picker renders nothing to toggle rather than failing the
// page.
func TestLoadProvidersDegradesToEmptyWithNoCacheAndFailingRefresh(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	const country = "XX"
	t.Cleanup(func() {
		c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		pool.Exec(c, `delete from streaming_providers where country = $1`, country)
	})
	if _, err := pool.Exec(ctx, `delete from streaming_providers where country = $1`, country); err != nil {
		t.Fatal(err)
	}

	load := loadProviders(pool, func(context.Context, string) ([]Provider, error) {
		return nil, errors.New("agent unreachable")
	})
	got, err := load(ctx, country)
	if err != nil {
		t.Fatalf("load with no cache and a failing refresh: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got = %+v, want empty", got)
	}
}

// saveSubscription (providers.go, T15): subscribing inserts, unsubscribing
// deletes, and repeating either is a no-op rather than an error — matches
// the picker's "flip a switch" semantics, not a form submit.
func TestSaveSubscription(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveSubscription(pool)

	const id = "sub_write_test"
	t.Cleanup(func() {
		c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		pool.Exec(c, `delete from users where id = $1`, id)
	})
	if _, err := pool.Exec(ctx,
		`insert into users (id, email) values ($1, $2) on conflict (id) do nothing`,
		id, id+"@example.com"); err != nil {
		t.Fatal(err)
	}

	exists := func() bool {
		var n int
		if err := pool.QueryRow(ctx,
			`select count(*) from streaming_subscriptions where user_id = $1 and tmdb_provider_id = 8`,
			id).Scan(&n); err != nil {
			t.Fatalf("select: %v", err)
		}
		return n == 1
	}

	if err := save(ctx, id, 8, true); err != nil {
		t.Fatalf("subscribe: %v", err)
	}
	if !exists() {
		t.Error("subscribing did not insert a row")
	}
	if err := save(ctx, id, 8, true); err != nil {
		t.Fatalf("subscribing an already-ticked provider: %v", err)
	}

	if err := save(ctx, id, 8, false); err != nil {
		t.Fatalf("unsubscribe: %v", err)
	}
	if exists() {
		t.Error("unsubscribing did not delete the row")
	}
	if err := save(ctx, id, 8, false); err != nil {
		t.Fatalf("unsubscribing an already-absent provider: %v", err)
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
		"insert verdict": `insert into title_verdicts (user_id, tmdb_id, media_type, verdict)
			values ('agent_probe', 550, 'movie', 'liked')`,
		"update verdict": `update title_verdicts set verdict = 'seen'
			where user_id = 'nobody' and tmdb_id = 550 and media_type = 'movie'`,
		"delete verdict": `delete from title_verdicts
			where user_id = 'nobody' and tmdb_id = 550 and media_type = 'movie'`,
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
		if _, err := pool.Exec(ctx,
			`insert into title_verdicts (user_id, tmdb_id, media_type, verdict) values ($1, 550, 'movie', 'liked')
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

		var verdicts int
		if err := tx.QueryRow(ctx,
			`select count(*) from title_verdicts where user_id = any($1)`,
			[]string{uA, uB}).Scan(&verdicts); err != nil {
			t.Fatal(err)
		}
		if verdicts != 1 {
			t.Errorf("as %s, visible verdicts = %d, want 1", uA, verdicts)
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

		// Same isolation on title_verdicts: own verdict passes WITH CHECK...
		if err := probe(t, ctx, tx,
			`insert into title_verdicts (user_id, tmdb_id, media_type, verdict) values ('`+uA+`', 551, 'movie', 'seen')`); err != nil {
			t.Errorf("as %s, writing its own verdict: %v", uA, err)
		}
		// ...but a verdict for another user is rejected by WITH CHECK.
		if err := probe(t, ctx, tx,
			`insert into title_verdicts (user_id, tmdb_id, media_type, verdict) values ('`+uB+`', 551, 'movie', 'seen')`); !denied(err) {
			t.Errorf("writing a verdict for %s while acting as %s: err = %v, want SQLSTATE 42501", uB, uA, err)
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

// saveVerdict (verdicts.go, T18): want_to_watch is fully mutable and the only
// value a judgment may still be set from; once one of the four judgments is
// set, the row is locked — title_verdicts' migration comment
// (20260903002450_verdicts.sql) states those "never change." A fake-backed
// unit test (verdicts_test.go) covers the HTTP layer; only a real database
// proves the WHERE-guarded upsert's SQL and its RowsAffected-based lock
// detection actually work, the same reason TestSaveSubscription and
// TestUpsertUser run against testPool rather than a fake.
func TestSaveVerdictLocksJudgments(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveVerdict(pool)

	const id = "verdict_write_test"
	t.Cleanup(func() {
		c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		pool.Exec(c, `delete from users where id = $1`, id)
	})
	if _, err := pool.Exec(ctx,
		`insert into users (id, email) values ($1, $2) on conflict (id) do nothing`,
		id, id+"@example.com"); err != nil {
		t.Fatal(err)
	}

	current := func() (string, bool) {
		var v string
		err := pool.QueryRow(ctx,
			`select verdict from title_verdicts where user_id = $1 and tmdb_id = 550 and media_type = 'movie'`,
			id).Scan(&v)
		if errors.Is(err, pgx.ErrNoRows) {
			return "", false
		}
		if err != nil {
			t.Fatalf("select: %v", err)
		}
		return v, true
	}
	strPtr := func(s string) *string { return &s }

	// want_to_watch is freely settable from nothing, and the expected
	// transition into a judgment succeeds (T17: "expected to become seen
	// later").
	if err := save(ctx, id, 550, "movie", strPtr("want_to_watch")); err != nil {
		t.Fatalf("set want_to_watch: %v", err)
	}
	if v, ok := current(); !ok || v != "want_to_watch" {
		t.Fatalf("current = %q, %v; want want_to_watch", v, ok)
	}
	if err := save(ctx, id, 550, "movie", strPtr("seen")); err != nil {
		t.Fatalf("want_to_watch -> seen: %v", err)
	}
	if v, ok := current(); !ok || v != "seen" {
		t.Fatalf("current = %q, %v; want seen", v, ok)
	}

	// Once "seen" (a judgment), re-setting the same value is an idempotent
	// no-op, but changing to a different judgment or back to want_to_watch is
	// rejected.
	if err := save(ctx, id, 550, "movie", strPtr("seen")); err != nil {
		t.Errorf("idempotent re-set of the same judgment: %v", err)
	}
	if err := save(ctx, id, 550, "movie", strPtr("liked")); !errors.Is(err, errVerdictLocked) {
		t.Errorf("seen -> liked: err = %v, want errVerdictLocked", err)
	}
	if err := save(ctx, id, 550, "movie", strPtr("want_to_watch")); !errors.Is(err, errVerdictLocked) {
		t.Errorf("seen -> want_to_watch: err = %v, want errVerdictLocked", err)
	}
	if v, ok := current(); !ok || v != "seen" {
		t.Errorf("current = %q, %v after rejected writes; want unchanged seen", v, ok)
	}

	// A locked judgment isn't deletable either — the clear (nil verdict) is a
	// no-op, not an error, matching setSubscription's idempotent DELETE.
	if err := save(ctx, id, 550, "movie", nil); err != nil {
		t.Errorf("clear on a locked judgment: %v", err)
	}
	if v, ok := current(); !ok || v != "seen" {
		t.Errorf("current = %q, %v after clear on a locked judgment; want unchanged seen", v, ok)
	}

	// A fresh title (no row yet) can be judged directly, no want_to_watch
	// detour required.
	if err := save(ctx, id, 551, "movie", strPtr("disliked")); err != nil {
		t.Fatalf("fresh judgment: %v", err)
	}
	if err := pool.QueryRow(ctx,
		`select verdict from title_verdicts where user_id = $1 and tmdb_id = 551 and media_type = 'movie'`,
		id).Scan(new(string)); err != nil {
		t.Errorf("fresh judgment row missing: %v", err)
	}
}
