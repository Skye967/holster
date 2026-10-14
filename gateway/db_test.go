package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
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

	// First sight creates exactly one row.
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

// TestVerifyRoleIsSafeRejectsTheMigrationRole proves main.go's startup check
// actually rejects a role that bypasses RLS: testPool's own connection
// is the TEST_DATABASE_URL superuser and the owner of every migrated table
// (see TestRLSEnabledOnUsers' comment), exactly the shape production must
// never boot as.
func TestVerifyRoleIsSafeRejectsTheMigrationRole(t *testing.T) {
	pool := testPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := verifyRoleIsSafe(ctx, pool); err == nil {
		t.Error("verifyRoleIsSafe = nil for the migration/superuser role, want an error")
	}
}

// TestVerifyRoleIsSafeAcceptsGatewayApp proves the check passes for the role
// production actually connects as, using asRole the same way every other
// gateway_app-scoped test in this file does.
func TestVerifyRoleIsSafeAcceptsGatewayApp(t *testing.T) {
	pool := testPool(t)
	asRole(t, pool, "gateway_app", func(ctx context.Context, tx pgx.Tx) {
		if err := verifyRoleIsSafe(ctx, tx); err != nil {
			t.Errorf("verifyRoleIsSafe = %v for gateway_app, want nil", err)
		}
	})
}

// asRole runs fn inside a rolled-back transaction with the session role dropped
// to role. The suite connects as the migration/superuser role, which bypasses
// RLS and every grant; SET ROLE is what makes the policies and grants under
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
// country (providers.go keeps that cache fresh — see chat.go's
// loadChatContext).
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
// GET /api/verdicts, and the chat context runTurn hands the agent. The
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

	// Explicit, distinct verdict_set_at values rather than the column default:
	// now() is transaction start time, so a multi-row insert stamps every row
	// identically and "newest first" would be asserting on heap order.
	//
	// tmdb_id 303 deliberately has the *oldest* created_at (a title bookmarked
	// long ago) but the *newest* verdict_set_at (liked just now) — the
	// want_to_watch -> liked transition verdict_set_at exists for. If load
	// were still ordering by created_at, 303 would sort last, not first.
	if _, err := pool.Exec(ctx,
		`insert into title_verdicts (user_id, tmdb_id, media_type, verdict, created_at, verdict_set_at)
		 values ($1, 101, 'movie', 'seen', now() - interval '3 hours', now() - interval '3 hours'),
		        ($1, 202, 'tv', 'liked', now() - interval '2 hours', now() - interval '2 hours'),
		        ($1, 303, 'movie', 'liked', now() - interval '5 hours', now() - interval '1 hour')
		 on conflict do nothing`, id); err != nil {
		t.Fatal(err)
	}

	vs, err = load(ctx, id)
	if err != nil {
		t.Fatalf("load with verdicts: %v", err)
	}
	want := []Verdict{
		{TMDBID: 303, MediaType: "movie", Verdict: "liked"},
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
		`insert into title_verdicts (user_id, tmdb_id, media_type, verdict, verdict_set_at)
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

// loadProviders (providers.go): no cache refreshes and caches; a fresh
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

// Refreshes are single-flighted, so one fetch serves every concurrent caller —
// which means it must not be tied to whichever request happened to start it.
// GET /guest/providers put this path in front of signed-out browsers, where a
// leader closing its tab mid-refresh is ordinary: if that cancelled the flight,
// every follower would be handed the leader's context.Canceled and degrade to
// an empty catalog, and the cache row would never be written, so the next wave
// would repeat it.
func TestLoadProvidersRefreshSurvivesTheLeaderGoingAway(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	const country = "XZ"
	clear := func() {
		c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		pool.Exec(c, `delete from streaming_providers where country = $1`, country)
	}
	t.Cleanup(clear)
	clear()

	var refreshCalls atomic.Int32
	entered := make(chan struct{})
	release := make(chan struct{})
	fresh := []Provider{{ProviderID: 8, ProviderName: "Netflix", DisplayPriority: 1}}
	load := loadProviders(pool, func(context.Context, string) ([]Provider, error) {
		if refreshCalls.Add(1) == 1 {
			close(entered)
			<-release
		}
		return fresh, nil
	})

	// The leader, on a context that is cancelled while its refresh is in flight.
	leaderCtx, cancelLeader := context.WithCancel(ctx)
	go func() {
		load(leaderCtx, country) //nolint:errcheck // the follower is what's asserted
	}()
	<-entered

	// A follower joins the same flight, then the leader goes away.
	followerDone := make(chan []Provider, 1)
	go func() {
		got, err := load(ctx, country)
		if err != nil {
			t.Errorf("follower load: %v", err)
		}
		followerDone <- got
	}()
	// The follower needs to be waiting on the flight before the leader is
	// cancelled, or it would start a second one and prove nothing.
	time.Sleep(100 * time.Millisecond)
	cancelLeader()
	close(release)

	select {
	case got := <-followerDone:
		if len(got) != 1 || got[0].ProviderID != 8 {
			t.Errorf("follower got = %+v, want the leader's fresh list", got)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("follower never returned")
	}
	if n := refreshCalls.Load(); n != 1 {
		t.Errorf("refreshCalls = %d, want 1 — the flight was not shared", n)
	}

	// The cache write must have outlived the cancelled leader too, or every
	// later caller pays the fetch again.
	var cachedJSON []byte
	if err := pool.QueryRow(ctx,
		`select providers from streaming_providers where country = $1`, country).
		Scan(&cachedJSON); err != nil {
		t.Fatalf("cache row after a cancelled leader: %v", err)
	}
}

// saveSubscription (providers.go): subscribing inserts, unsubscribing
// deletes, and repeating either is a no-op rather than an error — matches
// the picker's "flip a switch" semantics, not a form submit.
func TestSaveSubscription(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveSubscription(pool)

	const id = "sub_write_test"
	newTestUser(t, pool, ctx, id)

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

// The agent's core invariant: agent_ro has no write path to any table. Its read
// grants are narrow and added one at a time (today: select on conversations and
// messages, 20260904191046_conversations.sql), but "never writes" is the property
// that must not regress, so that is what this pins.
//
// Probes every table the gateway writes, including the two agent_ro can read --
// a table it can reach is where a stray write grant would matter most. Coverage
// is per statement and not exhaustive. Add a table here when you add one.
//
// Every policy on these tables is scoped to gateway_app, so an INSERT that
// agent_ro was wrongly granted still raises 42501 from RLS and this probe
// stays green. UPDATE and DELETE have no such second layer: RLS filters rows
// rather than raising, so a stray grant there returns "0 rows" and no error,
// and the probe fails.
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
		"insert conversation": `insert into conversations (id, user_id, title)
			values (gen_random_uuid(), 'agent_probe', 'x')`,
		"update conversation": `update conversations set title = 'y' where user_id = 'nobody'`,
		"delete conversation": `delete from conversations where user_id = 'nobody'`,
		"insert message": `insert into messages (conversation_id, role, content)
			values (gen_random_uuid(), 'user', 'x')`,
		"update message": `update messages set content = 'y' where role = 'nobody'`,
		"delete message": `delete from messages where role = 'nobody'`,
	}
	asRole(t, pool, "agent_ro", func(ctx context.Context, tx pgx.Tx) {
		for name, sql := range writes {
			if err := probe(t, ctx, tx, sql); !denied(err) {
				t.Errorf("%s: err = %v, want SQLSTATE 42501", name, err)
			}
		}
	})
}

// Under gateway_app, holster.user_id fences every user-scoped
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

// TestGatewayRoleEnforcesIsolationOnDeleteAndUpdate closes the coverage gap
// TestGatewayRoleEnforcesUserIsolation leaves: deleteUser/updateUserEmail
// (webhooks.go) lean on this same user_isolation policy as their only
// backstop against a webhook payload naming the wrong id -- unlike every
// other withUser caller, id there comes from a Clerk webhook payload, not a
// verified JWT subject.
//
// UPDATE/DELETE aren't WITH CHECK's job here -- that clause only gates the
// row a write would produce, and neither of these statements changes id, so
// it never comes into play. USING is what matters: it filters which rows
// UPDATE/DELETE can even see, so a cross-user attempt matches zero rows
// silently rather than erroring -- the same shape deleteUser's own
// idempotent "already gone" retries already depend on.
func TestGatewayRoleEnforcesIsolationOnDeleteAndUpdate(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	const uA, uB = "rls_write_a", "rls_write_b"
	newTestUser(t, pool, ctx, uA)
	newTestUser(t, pool, ctx, uB)

	// Everything below runs inside one rolled-back transaction (asRole never
	// commits), so uA's own-row delete near the end doesn't need its own
	// undo and both rows are still there for t.Cleanup afterward.
	asRole(t, pool, "gateway_app", func(ctx context.Context, tx pgx.Tx) {
		if _, err := tx.Exec(ctx, `select set_config('holster.user_id', $1, true)`, uA); err != nil {
			t.Fatalf("set_config: %v", err)
		}

		tag, err := tx.Exec(ctx, `delete from users where id = $1`, uB)
		if err != nil {
			t.Errorf("delete attempt on %s while acting as %s: %v", uB, uA, err)
		}
		if tag.RowsAffected() != 0 {
			t.Errorf("deleting %s's row while acting as %s affected %d rows, want 0", uB, uA, tag.RowsAffected())
		}

		tag, err = tx.Exec(ctx, `update users set email = 'hijacked@example.com' where id = $1`, uB)
		if err != nil {
			t.Errorf("update attempt on %s while acting as %s: %v", uB, uA, err)
		}
		if tag.RowsAffected() != 0 {
			t.Errorf("updating %s's row while acting as %s affected %d rows, want 0", uB, uA, tag.RowsAffected())
		}

		// The positive case deleteUser/updateUserEmail depend on: acting as
		// its own id succeeds.
		tag, err = tx.Exec(ctx, `delete from users where id = $1`, uA)
		if err != nil {
			t.Errorf("deleting its own row while acting as %s: %v", uA, err)
		}
		if tag.RowsAffected() != 1 {
			t.Errorf("deleting its own row while acting as %s affected %d rows, want 1", uA, tag.RowsAffected())
		}
	})
}

// currentVerdict reads one title's stored verdict, or false if no row
// exists — shared by every saveVerdict test below rather than each defining
// its own copy of the same lookup.
func currentVerdict(t *testing.T, pool *pgxpool.Pool, ctx context.Context, userID string, tmdbID int) (string, bool) {
	t.Helper()
	var v string
	err := pool.QueryRow(ctx,
		`select verdict from title_verdicts where user_id = $1 and tmdb_id = $2 and media_type = 'movie'`,
		userID, tmdbID).Scan(&v)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", false
	}
	if err != nil {
		t.Fatalf("select verdict for tmdb %d: %v", tmdbID, err)
	}
	return v, true
}

// newTestUser inserts (or reuses) a user row and registers its
// cascade-delete cleanup — shared by every test below whose setup is exactly
// this. Several other tests in this file need a differently-shaped seed (an
// extra column, a pre-delete guard, multiple users) and are left as their own
// inline blocks rather than bent to fit this signature.
func newTestUser(t *testing.T, pool *pgxpool.Pool, ctx context.Context, id string) {
	t.Helper()
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
}

// saveVerdict (verdicts.go): want_to_watch is fully mutable and the only
// value a judgment may still be set from; once one of the four judgments is
// set, a direct overwrite to a *different* verdict is locked — title_verdicts'
// migration comment (20260903002450_verdicts.sql) states those "never change."
// The row can still be cleared and re-judged (ARCHITECTURE.md's "A locked
// judgment can be cleared"); only the one-step overwrite stays rejected.
// A fake-backed unit test (verdicts_test.go) covers the HTTP layer; only a
// real database proves the WHERE-guarded upsert's SQL and its
// RowsAffected-based lock detection actually work, the same reason
// TestSaveSubscription and TestUpsertUser run against testPool rather than a
// fake.
func TestSaveVerdictLocksJudgments(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveVerdict(pool)
	strPtr := func(s string) *string { return &s }

	const id = "verdict_write_test"
	newTestUser(t, pool, ctx, id)

	// want_to_watch is freely settable from nothing, and the expected
	// transition into a judgment succeeds — want_to_watch is an intention
	// expected to become seen later.
	if err := save(ctx, id, 550, "movie", strPtr("want_to_watch"), ""); err != nil {
		t.Fatalf("set want_to_watch: %v", err)
	}
	if v, ok := currentVerdict(t, pool, ctx, id, 550); !ok || v != "want_to_watch" {
		t.Fatalf("current = %q, %v; want want_to_watch", v, ok)
	}
	if err := save(ctx, id, 550, "movie", strPtr("seen"), ""); err != nil {
		t.Fatalf("want_to_watch -> seen: %v", err)
	}
	if v, ok := currentVerdict(t, pool, ctx, id, 550); !ok || v != "seen" {
		t.Fatalf("current = %q, %v; want seen", v, ok)
	}

	// Once "seen" (a judgment), re-setting the same value is an idempotent
	// no-op, but changing to a different judgment or back to want_to_watch is
	// rejected.
	if err := save(ctx, id, 550, "movie", strPtr("seen"), ""); err != nil {
		t.Errorf("idempotent re-set of the same judgment: %v", err)
	}
	if err := save(ctx, id, 550, "movie", strPtr("liked"), ""); !errors.Is(err, errVerdictLocked) {
		t.Errorf("seen -> liked: err = %v, want errVerdictLocked", err)
	}
	if err := save(ctx, id, 550, "movie", strPtr("want_to_watch"), ""); !errors.Is(err, errVerdictLocked) {
		t.Errorf("seen -> want_to_watch: err = %v, want errVerdictLocked", err)
	}
	if v, ok := currentVerdict(t, pool, ctx, id, 550); !ok || v != "seen" {
		t.Errorf("current = %q, %v after rejected writes; want unchanged seen", v, ok)
	}

	// A fresh title (no row yet) can be judged directly, no want_to_watch
	// detour required.
	if err := save(ctx, id, 551, "movie", strPtr("disliked"), ""); err != nil {
		t.Fatalf("fresh judgment: %v", err)
	}
	if v, ok := currentVerdict(t, pool, ctx, id, 551); !ok || v != "disliked" {
		t.Errorf("current = %q, %v; want disliked", v, ok)
	}
}

// verdict_set_at (20260904154626_verdict_set_at.sql) must only refresh when
// the verdict actually changes — an idempotent resubmit (a client retry
// after a timed-out-but-succeeded request, say) must not bump a title's
// recency and jump it ahead of a genuinely more recent one in loadVerdicts'
// ordering. Asserted by exact equality between two reads, not "recent
// enough": both a correct implementation and a regressed one (the upsert's
// CASE bumping unconditionally) would pass a mere recency check, since both
// writes happen within the same test run.
func TestSaveVerdictSetAtOnlyBumpsOnRealChange(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveVerdict(pool)
	strPtr := func(s string) *string { return &s }

	const id = "verdict_set_at_test"
	newTestUser(t, pool, ctx, id)

	readSetAt := func() time.Time {
		t.Helper()
		var setAt time.Time
		if err := pool.QueryRow(ctx,
			`select verdict_set_at from title_verdicts where user_id = $1 and tmdb_id = 550 and media_type = 'movie'`,
			id).Scan(&setAt); err != nil {
			t.Fatalf("select verdict_set_at: %v", err)
		}
		return setAt
	}

	if err := save(ctx, id, 550, "movie", strPtr("want_to_watch"), ""); err != nil {
		t.Fatalf("set want_to_watch: %v", err)
	}
	if err := save(ctx, id, 550, "movie", strPtr("seen"), ""); err != nil {
		t.Fatalf("want_to_watch -> seen: %v", err)
	}
	afterRealChange := readSetAt()

	if err := save(ctx, id, 550, "movie", strPtr("seen"), ""); err != nil {
		t.Fatalf("idempotent re-set: %v", err)
	}
	if afterIdempotentReset := readSetAt(); !afterIdempotentReset.Equal(afterRealChange) {
		t.Errorf("verdict_set_at = %v after an idempotent re-set, want unchanged %v", afterIdempotentReset, afterRealChange)
	}
}

// Clearing (verdict == nil) is a compare-and-delete: it only removes the row
// when it still holds the exact verdict the caller expects — the same idiom
// TestSaveVerdictLocksJudgments' upsert already uses. This is what keeps a
// stale client (a second tab that hasn't seen a write made elsewhere) safe:
// a mismatched clear is rejected with errVerdictStale rather than either
// silently succeeding or silently touching the wrong row, whether the
// mismatch is between want_to_watch and a judgment or between two different
// judgments. Each scenario below is independent, not a narrative on one
// title, so each runs as its own subtest — a failure in one must not mask
// or abort the others.
func TestSaveVerdictClearIsScopedToExpectedVerdict(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveVerdict(pool)
	strPtr := func(s string) *string { return &s }

	const id = "verdict_clear_scope_test"
	newTestUser(t, pool, ctx, id)

	t.Run("bookmark happy path", func(t *testing.T) {
		// Previously untested against a real database: clearing a
		// want_to_watch row when the expected value still matches actually
		// removes it.
		if err := save(ctx, id, 601, "movie", strPtr("want_to_watch"), ""); err != nil {
			t.Fatalf("seed want_to_watch: %v", err)
		}
		if err := save(ctx, id, 601, "movie", nil, "want_to_watch"); err != nil {
			t.Errorf("clear matching want_to_watch: %v", err)
		}
		if _, ok := currentVerdict(t, pool, ctx, id, 601); ok {
			t.Errorf("still present after a matching want_to_watch clear")
		}
	})

	t.Run("bookmark clear against an already absent row is a silent no-op", func(t *testing.T) {
		// The row never existed — an idempotent double-click or a retried
		// request must not error just because there's nothing to remove.
		if err := save(ctx, id, 605, "movie", nil, "want_to_watch"); err != nil {
			t.Errorf("clear on an absent row: %v", err)
		}
	})

	t.Run("stale bookmark clear against a locked judgment is rejected, not silent", func(t *testing.T) {
		// A stale bookmark tab expecting want_to_watch against a row that's
		// since become a locked judgment — the original multi-tab race this
		// scoping exists to prevent. Rejected with errVerdictStale rather
		// than a silent no-op, so the caller's optimistic UI has a signal
		// to correct itself instead of drifting from the server forever.
		if err := save(ctx, id, 602, "movie", strPtr("want_to_watch"), ""); err != nil {
			t.Fatalf("seed want_to_watch: %v", err)
		}
		if err := save(ctx, id, 602, "movie", strPtr("seen"), ""); err != nil {
			t.Fatalf("want_to_watch -> seen: %v", err)
		}
		if err := save(ctx, id, 602, "movie", nil, "want_to_watch"); !errors.Is(err, errVerdictStale) {
			t.Errorf("mismatched want_to_watch clear against seen: err = %v, want errVerdictStale", err)
		}
		if v, ok := currentVerdict(t, pool, ctx, id, 602); !ok || v != "seen" {
			t.Errorf("current = %q, %v after a mismatched clear; want unchanged seen", v, ok)
		}
	})

	t.Run("judgment happy path", func(t *testing.T) {
		// Clearing a locked judgment when the expected value matches
		// exactly actually removes it (ARCHITECTURE.md's "A locked
		// judgment can be cleared").
		if err := save(ctx, id, 603, "movie", strPtr("liked"), ""); err != nil {
			t.Fatalf("seed liked: %v", err)
		}
		if err := save(ctx, id, 603, "movie", nil, "liked"); err != nil {
			t.Errorf("clear matching liked: %v", err)
		}
		if _, ok := currentVerdict(t, pool, ctx, id, 603); ok {
			t.Errorf("still present after a matching liked clear")
		}
	})

	t.Run("stale judgment clear against a different judgment is rejected", func(t *testing.T) {
		// A stale "Clear rating" click expecting the judgment it last saw,
		// against a row that's since become a *different* judgment.
		// Comparing against the exact expected value (not just "any
		// judgment") is what closes the race between two different
		// judgments a coarser scope would leave open. Seeded directly as
		// "disliked" — no need to route through "liked" first, since only
		// the end state (a locked judgment other than "liked") matters here.
		if err := save(ctx, id, 604, "movie", strPtr("disliked"), ""); err != nil {
			t.Fatalf("seed disliked: %v", err)
		}
		if err := save(ctx, id, 604, "movie", nil, "liked"); !errors.Is(err, errVerdictStale) {
			t.Errorf("mismatched liked clear against disliked: err = %v, want errVerdictStale", err)
		}
		if v, ok := currentVerdict(t, pool, ctx, id, 604); !ok || v != "disliked" {
			t.Errorf("current = %q, %v after a mismatched judgment clear; want unchanged disliked", v, ok)
		}

		// Clearing and re-judging stay two explicit steps: after a real
		// clear, the row is gone, so a new judgment is a fresh insert, not
		// a rejected overwrite of a locked one.
		if err := save(ctx, id, 604, "movie", nil, "disliked"); err != nil {
			t.Fatalf("clear disliked: %v", err)
		}
		if err := save(ctx, id, 604, "movie", strPtr("liked"), ""); err != nil {
			t.Errorf("re-judge after clearing a locked judgment: %v", err)
		}
		if v, ok := currentVerdict(t, pool, ctx, id, 604); !ok || v != "liked" {
			t.Errorf("current = %q, %v after re-judging a cleared row; want liked", v, ok)
		}
	})
}

// --- conversations.go ---------------------------------------------------------

// TestLoadConversationReturnsEmptyForANewCaller proves loadConversation
// degrades to nil, not an error, for a conversation id with nothing saved yet
// — the shape both a brand-new "new chat" id and a not-yet-created id share.
// runChatConnection treats that as "start blank."
func TestLoadConversationReturnsEmptyForANewCaller(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	load := loadConversation(pool)

	const id = "conv_empty_test"
	newTestUser(t, pool, ctx, id)

	history, shown, err := load(ctx, id, uuid.NewString())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if history != nil {
		t.Errorf("history = %#v, want nil", history)
	}
	if shown != nil {
		t.Errorf("shown = %#v, want nil", shown)
	}
}

// TestLoadConversationOrdersAndCapsHistory seeds more than loadConversation's
// own cap (maxHistoryMessages, not maxStoredHistoryMessages — that one
// bounds GET /api/chat/history/{conversationID} instead, see
// TestLoadConversationTurns*) and confirms it returns only the most recent
// ones, oldest-first — windowHistory and the agent both depend on
// chronological order, and the cap exists precisely so a long-running
// conversation doesn't load its entire history on every reconnect or
// conversation switch just to discard most of it.
func TestLoadConversationOrdersAndCapsHistory(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	load := loadConversation(pool)

	const id = "conv_cap_test"
	newTestUser(t, pool, ctx, id)

	convID := uuid.NewString()
	if _, err := pool.Exec(ctx,
		`insert into conversations (id, user_id) values ($1, $2)`, convID, id); err != nil {
		t.Fatal(err)
	}

	// One more pair than the cap allows, tagged by index so ordering is
	// verifiable — content isn't otherwise unique.
	const pairs = maxHistoryMessages/2 + 1
	for i := range pairs {
		if _, err := pool.Exec(ctx, `
			insert into messages (conversation_id, role, content) values
				($1, 'user', $2), ($1, 'assistant', $3)`,
			convID, fmt.Sprintf("q%d", i), fmt.Sprintf("a%d", i)); err != nil {
			t.Fatal(err)
		}
	}

	history, _, err := load(ctx, id, convID)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(history) != maxHistoryMessages {
		t.Fatalf("history length = %d, want %d", len(history), maxHistoryMessages)
	}
	// The oldest pair (q0/a0) must have been dropped by the cap, and what
	// remains must still be oldest-first.
	if history[0].Text != "q1" {
		t.Errorf("history[0].Text = %q, want %q (oldest surviving pair)", history[0].Text, "q1")
	}
	last := history[len(history)-1]
	if last.Text != fmt.Sprintf("a%d", pairs-1) {
		t.Errorf("last history entry = %q, want the most recent assistant text", last.Text)
	}
}

// TestLoadConversationReadsEveryStoredTitleRef proves the already-shown set
// round-trips through the database and is not bounded by the history beside
// it: the refs come from their own query, so a conversation longer than
// maxHistoryMessages still suppresses everything it ever showed. That is what
// makes a reconnect cost an account nothing. Rows with a null title_refs — a
// user row, and an assistant row that showed no cards — contribute nothing
// rather than erroring.
func TestLoadConversationReadsEveryStoredTitleRef(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	load := loadConversation(pool)

	const id = "conv_refs_test"
	newTestUser(t, pool, ctx, id)

	convID := uuid.NewString()
	if _, err := pool.Exec(ctx,
		`insert into conversations (id, user_id) values ($1, $2)`, convID, id); err != nil {
		t.Fatal(err)
	}

	// More turns than the history cap, so anything derived from the history
	// window would lose the oldest refs.
	pairs := maxHistoryMessages
	var want []agentTitleRef
	for i := range pairs {
		ref := agentTitleRef{TMDBID: 100 + i, MediaType: "movie"}
		refsJSON, err := json.Marshal([]agentTitleRef{ref})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
			insert into messages (conversation_id, role, content, title_refs) values
				($1, 'user', 'a heist movie', null), ($1, 'assistant', 'Suggested', $2)`,
			convID, refsJSON); err != nil {
			t.Fatal(err)
		}
		want = append(want, ref)
	}

	history, shown, err := load(ctx, id, convID)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(history) != maxHistoryMessages {
		t.Fatalf("history length = %d, want the cap %d", len(history), maxHistoryMessages)
	}
	// Every ref, oldest-first — including the ones whose messages the history
	// window dropped.
	if !slices.Equal(shown, want) {
		t.Errorf("shown = %+v, want %+v", shown, want)
	}
}

// TestLoadConversationTurnsPairsMessages proves GET
// /api/chat/history/{conversationID}'s loader pairs stored role/content rows
// into completed exchanges in chronological order.
func TestLoadConversationTurnsPairsMessages(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	load := loadConversationTurns(pool)

	const id = "conv_turns_test"
	newTestUser(t, pool, ctx, id)

	convID := uuid.NewString()
	if _, err := pool.Exec(ctx,
		`insert into conversations (id, user_id) values ($1, $2)`, convID, id); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		insert into messages (conversation_id, role, content) values
			($1, 'user', 'a heist movie'), ($1, 'assistant', 'Suggested: Heat'),
			($1, 'user', 'something shorter'), ($1, 'assistant', 'Suggested: Ronin')`,
		convID); err != nil {
		t.Fatal(err)
	}

	turns, exists, err := load(ctx, id, convID)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !exists {
		t.Error("exists = false, want true for a conversation that was just inserted")
	}
	want := []conversationTurn{
		{UserText: "a heist movie", AssistantText: "Suggested: Heat"},
		{UserText: "something shorter", AssistantText: "Suggested: Ronin"},
	}
	if len(turns) != len(want) {
		t.Fatalf("turns = %+v, want %+v", turns, want)
	}
	for i := range want {
		if turns[i] != want[i] {
			t.Errorf("turns[%d] = %+v, want %+v", i, turns[i], want[i])
		}
	}
}

// TestLoadConversationTurnsReportsNotExistsForAnUnknownID is the regression
// guard for chatHistory's 404: a conversation id nothing has ever written reports
// exists=false (and, per that field's own contract, so does one belonging to
// another user — see TestLoadConversationTurnsHidesAForeignConversation and
// loadConversationTurns' own doc comment for why the two can't be told apart
// here). turns stays the non-nil empty slice either way, matching the `[]`,
// never `null`, wire contract every other list endpoint holds.
func TestLoadConversationTurnsReportsNotExistsForAnUnknownID(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	load := loadConversationTurns(pool)

	const id = "conv_turns_empty_test"
	newTestUser(t, pool, ctx, id)

	turns, exists, err := load(ctx, id, uuid.NewString())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if exists {
		t.Error("exists = true, want false for a conversation id nothing has ever written")
	}
	if turns == nil || len(turns) != 0 {
		t.Errorf("turns = %#v, want non-nil empty slice", turns)
	}
}

// TestLoadConversationTurnsHidesAForeignConversation proves RLS still
// independently blocks this on its own, as a second layer behind
// loadConversationTurns' own explicit user_id filter
// (TestLoadConversationAndTurnsHideAForeignConversationAtTheAppLevel below) —
// with a naive, unfiltered `select exists(...)`, run under an explicit
// asRole("gateway_app") with
// holster.user_id bound to the *other* user, the only way to make RLS apply
// at all (testPool otherwise connects as the table owner, which bypasses it).
func TestLoadConversationTurnsHidesAForeignConversation(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	const uA, uB = "conv_turns_foreign_a", "conv_turns_foreign_b"
	newTestUser(t, pool, ctx, uA)
	newTestUser(t, pool, ctx, uB)

	convA := uuid.NewString()
	if _, err := pool.Exec(ctx,
		`insert into conversations (id, user_id) values ($1, $2)`, convA, uA); err != nil {
		t.Fatal(err)
	}

	asRole(t, pool, "gateway_app", func(ctx context.Context, tx pgx.Tx) {
		if _, err := tx.Exec(ctx, `select set_config('holster.user_id', $1, true)`, uB); err != nil {
			t.Fatalf("set_config: %v", err)
		}
		var exists bool
		if err := tx.QueryRow(ctx,
			`select exists(select 1 from conversations where id = $1)`, convA).Scan(&exists); err != nil {
			t.Fatalf("exists query: %v", err)
		}
		if exists {
			t.Errorf("%s's conversation is visible while acting as %s, want hidden by RLS", uA, uB)
		}
	})
}

// TestLoadConversationAndTurnsHideAForeignConversationAtTheAppLevel proves
// the explicit user_id join independently of RLS: unlike the
// asRole-based tests above, this calls loadConversation and
// loadConversationTurns directly through testPool, which connects as the
// migration/superuser role and bypasses RLS entirely, so the join itself, not
// RLS, is what has to fail this.
func TestLoadConversationAndTurnsHideAForeignConversationAtTheAppLevel(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	const uA, uB = "conv_app_level_a", "conv_app_level_b"
	newTestUser(t, pool, ctx, uA)
	newTestUser(t, pool, ctx, uB)

	convA := uuid.NewString()
	if _, err := pool.Exec(ctx,
		`insert into conversations (id, user_id) values ($1, $2)`, convA, uA); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx,
		`insert into messages (conversation_id, role, content) values ($1, 'user', 'a-only')`,
		convA); err != nil {
		t.Fatal(err)
	}

	history, shown, err := loadConversation(pool)(ctx, uB, convA)
	if err != nil {
		t.Fatalf("loadConversation: %v", err)
	}
	if shown != nil {
		t.Errorf("loadConversation(uB, convA) shown = %#v, want nil (convA belongs to uA)", shown)
	}
	if history != nil {
		t.Errorf("loadConversation(uB, convA) = %#v, want nil (convA belongs to uA)", history)
	}

	turns, exists, err := loadConversationTurns(pool)(ctx, uB, convA)
	if err != nil {
		t.Fatalf("loadConversationTurns: %v", err)
	}
	if exists {
		t.Error("loadConversationTurns(uB, convA) exists = true, want false (convA belongs to uA)")
	}
	if len(turns) != 0 {
		t.Errorf("loadConversationTurns(uB, convA) turns = %+v, want empty", turns)
	}
}

// TestSaveMessagesCreatesConversationOnFirstUseThenAppends proves the
// lazy-create shape: saveMessages' first call against a client-generated
// conversation id creates the row, and a second call against the same id
// appends to it rather than creating a second one — ids are always supplied by
// the caller, never discovered here.
func TestSaveMessagesCreatesConversationOnFirstUseThenAppends(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveMessages(pool)

	const id = "conv_save_lazy_test"
	newTestUser(t, pool, ctx, id)
	convID := uuid.NewString()

	if _, err := save(ctx, id, convID, "a heist movie", "Suggested: Heat", nil); err != nil {
		t.Fatalf("first save: %v", err)
	}
	if _, err := save(ctx, id, convID, "something shorter", "Suggested: Ronin", nil); err != nil {
		t.Fatalf("second save: %v", err)
	}

	var count int
	if err := pool.QueryRow(ctx,
		`select count(*) from conversations where user_id = $1`, id).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Errorf("conversations for %s = %d, want 1 (no second row from the second save)", id, count)
	}

	var msgCount int
	if err := pool.QueryRow(ctx,
		`select count(*) from messages where conversation_id = $1`, convID).Scan(&msgCount); err != nil {
		t.Fatal(err)
	}
	if msgCount != 4 {
		t.Errorf("messages in %s = %d, want 4 (two turns of two rows each)", convID, msgCount)
	}
}

// TestSaveMessagesCreatesConversationWithDerivedTitle proves the create side
// of saveMessages' on-conflict insert sets title from the first message
// — generated from the first exchange, once, and stored.
func TestSaveMessagesCreatesConversationWithDerivedTitle(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveMessages(pool)

	const id = "conv_save_title_test"
	newTestUser(t, pool, ctx, id)
	convID := uuid.NewString()

	if _, err := save(ctx, id, convID, "a heist movie set in Tokyo", "Suggested: Heat", nil); err != nil {
		t.Fatalf("save: %v", err)
	}

	var title string
	if err := pool.QueryRow(ctx,
		`select title from conversations where id = $1`, convID).Scan(&title); err != nil {
		t.Fatal(err)
	}
	if title != "a heist movie set in Tokyo" {
		t.Errorf("title = %q, want the derived title from the first user message", title)
	}
}

// TestSaveMessagesSecondCallDoesNotOverwriteTitle proves title is set once —
// a second message on the same conversation must not replace it, even though
// titleFromMessage is recomputed from the second call's own userText every
// time (it's simply discarded by the on-conflict-do-nothing insert).
func TestSaveMessagesSecondCallDoesNotOverwriteTitle(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveMessages(pool)

	const id = "conv_save_title_locked_test"
	newTestUser(t, pool, ctx, id)
	convID := uuid.NewString()

	if _, err := save(ctx, id, convID, "first message", "reply one", nil); err != nil {
		t.Fatalf("first save: %v", err)
	}
	if _, err := save(ctx, id, convID, "second message", "reply two", nil); err != nil {
		t.Fatalf("second save: %v", err)
	}

	var title string
	if err := pool.QueryRow(ctx,
		`select title from conversations where id = $1`, convID).Scan(&title); err != nil {
		t.Fatal(err)
	}
	if title != "first message" {
		t.Errorf("title = %q, want the first message's title, unchanged by the second save", title)
	}
}

// TestSaveMessagesPersistsTitleRefsOnAssistantRowOnly proves saveMessages
// writes the shown title ids onto the assistant row and leaves the user row's
// title_refs null, and that a nil
// slice persists as SQL NULL rather than the JSON literal "null".
func TestSaveMessagesPersistsTitleRefsOnAssistantRowOnly(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveMessages(pool)

	const id = "conv_save_refs_test"
	newTestUser(t, pool, ctx, id)
	convID := uuid.NewString()

	refs := []agentTitleRef{{TMDBID: 550, MediaType: "movie"}, {TMDBID: 603, MediaType: "movie"}}
	if _, err := save(ctx, id, convID, "a heist movie", "Suggested: Fight Club, The Matrix", refs); err != nil {
		t.Fatalf("save: %v", err)
	}

	rows, err := pool.Query(ctx,
		`select role, title_refs from messages where conversation_id = $1 order by created_at`, convID)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()

	var sawUser, sawAssistant bool
	for rows.Next() {
		var role string
		var refsJSON []byte
		if err := rows.Scan(&role, &refsJSON); err != nil {
			t.Fatal(err)
		}
		switch role {
		case "user":
			sawUser = true
			if refsJSON != nil {
				t.Errorf("user row title_refs = %s, want SQL NULL", refsJSON)
			}
		case "assistant":
			sawAssistant = true
			var got []agentTitleRef
			if err := json.Unmarshal(refsJSON, &got); err != nil {
				t.Fatalf("decode title_refs: %v", err)
			}
			if len(got) != 2 || got[0].TMDBID != 550 || got[1].TMDBID != 603 {
				t.Errorf("assistant row title_refs = %+v, want %+v", got, refs)
			}
		}
	}
	if !sawUser || !sawAssistant {
		t.Fatalf("expected one user and one assistant row, sawUser=%v sawAssistant=%v", sawUser, sawAssistant)
	}
}

// TestSaveMessagesConcurrentCallsSameNewIDConvergeOnOneRow proves the
// id-scoped create: several callers racing to
// create the *same client-generated conversation id* (two tabs opening "New
// chat" would never collide on an id in practice, but two turns on one
// connection completing close enough together against a brand-new id can)
// must converge on one conversations row, not error or fork.
func TestSaveMessagesConcurrentCallsSameNewIDConvergeOnOneRow(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveMessages(pool)

	const id = "conv_concurrent_create_test"
	newTestUser(t, pool, ctx, id)
	convID := uuid.NewString()

	const racers = 5
	errs := make([]error, racers)
	var wg sync.WaitGroup
	for i := range racers {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, errs[i] = save(ctx, id, convID, fmt.Sprintf("q%d", i), fmt.Sprintf("a%d", i), nil)
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("racer %d: %v", i, err)
		}
	}

	var convCount int
	if err := pool.QueryRow(ctx,
		`select count(*) from conversations where id = $1`, convID).Scan(&convCount); err != nil {
		t.Fatal(err)
	}
	if convCount != 1 {
		t.Errorf("conversations with id %s = %d, want 1", convID, convCount)
	}

	var msgCount int
	if err := pool.QueryRow(ctx,
		`select count(*) from messages where conversation_id = $1`, convID).Scan(&msgCount); err != nil {
		t.Fatal(err)
	}
	if msgCount != racers*2 {
		t.Errorf("messages = %d, want %d (every racer's pair landed on the one conversation)", msgCount, racers*2)
	}
}

// TestSaveMessagesReportsWhetherItCreatedTheConversation proves the outcome
// bool chat.go's finishTurn needs to know, definitively, when to tell the
// browser a new conversation exists — created only on the call whose insert
// actually added the row, mirroring deleteConversation's own
// RowsAffected()-derived outcome bool.
func TestSaveMessagesReportsWhetherItCreatedTheConversation(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveMessages(pool)

	const id = "conv_save_created_test"
	newTestUser(t, pool, ctx, id)
	convID := uuid.NewString()

	created, err := save(ctx, id, convID, "a heist movie", "Suggested: Heat", nil)
	if err != nil {
		t.Fatalf("first save: %v", err)
	}
	if !created {
		t.Error("created = false, want true for a brand-new conversation id")
	}

	created, err = save(ctx, id, convID, "something shorter", "Suggested: Ronin", nil)
	if err != nil {
		t.Fatalf("second save: %v", err)
	}
	if created {
		t.Error("created = true, want false for a second message on an existing conversation")
	}
}

// TestSaveMessagesRejectsForeignConversationID proves message_isolation's RLS
// policy still independently blocks this on its own, as a second layer
// behind saveMessages' own explicit join
// (TestSaveMessagesRejectsForeignConversationIDAtTheAppLevel below) — not by
// calling the production closure (whose current statement text this no
// longer mirrors), but with a plain, naive multi-row insert carrying no
// ownership join of its own, run under an explicit
// asRole("gateway_app") — the only way to make RLS apply at all, since
// testPool otherwise connects as the table owner, which bypasses it.
func TestSaveMessagesRejectsForeignConversationID(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	const uA, uB = "conv_foreign_a", "conv_foreign_b"
	newTestUser(t, pool, ctx, uA)
	newTestUser(t, pool, ctx, uB)

	convA := uuid.NewString()
	if _, err := pool.Exec(ctx,
		`insert into conversations (id, user_id) values ($1, $2)`, convA, uA); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx,
		`insert into messages (conversation_id, role, content) values ($1, 'user', 'a-only')`,
		convA); err != nil {
		t.Fatal(err)
	}

	asRole(t, pool, "gateway_app", func(ctx context.Context, tx pgx.Tx) {
		if _, err := tx.Exec(ctx, `select set_config('holster.user_id', $1, true)`, uB); err != nil {
			t.Fatalf("set_config: %v", err)
		}
		// The conversations insert's DO NOTHING silently no-ops against uA's
		// pre-existing row (RLS never blocks that — the row simply isn't
		// touched), so it's the naive messages insert immediately after that
		// must fail — proving RLS alone would still catch this even without
		// saveMessages' own join.
		if _, err := tx.Exec(ctx, `
			insert into conversations (id, user_id, title) values ($1, $2, $3)
			on conflict (id) do nothing`,
			convA, uB, "intrusion"); err != nil {
			t.Fatalf("conversations insert (expected to no-op, not error): %v", err)
		}
		if err := probe(t, ctx, tx, `
			insert into messages (conversation_id, role, content) values ('`+convA+`', 'user', 'intrusion')`,
		); !denied(err) {
			t.Errorf("messages insert against %s's conversation while acting as %s: err = %v, want SQLSTATE 42501", uA, uB, err)
		}
	})

	var msgCount int
	if err := pool.QueryRow(ctx,
		`select count(*) from messages where conversation_id = $1`, convA).Scan(&msgCount); err != nil {
		t.Fatal(err)
	}
	if msgCount != 1 {
		t.Errorf("messages in %s's conversation = %d, want 1 (only %s's own message)", uA, msgCount, uA)
	}
}

// TestSaveMessagesRejectsForeignConversationIDAtTheAppLevel proves the
// explicit join, independently of RLS: it calls the real saveMessages
// closure directly through testPool, which connects as the migration/
// superuser role and bypasses RLS entirely (see TestRLSEnabledOnUsers'
// comment), so the join itself, not RLS, is what has to reject it.
func TestSaveMessagesRejectsForeignConversationIDAtTheAppLevel(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	save := saveMessages(pool)

	const uA, uB = "conv_app_level_save_a", "conv_app_level_save_b"
	newTestUser(t, pool, ctx, uA)
	newTestUser(t, pool, ctx, uB)

	convA := uuid.NewString()
	if _, err := pool.Exec(ctx,
		`insert into conversations (id, user_id) values ($1, $2)`, convA, uA); err != nil {
		t.Fatal(err)
	}

	_, err := save(ctx, uB, convA, "intrusion", "should never be saved", nil)
	if !errors.Is(err, errConversationNotWritable) {
		t.Errorf("save(uB, convA, ...) err = %v, want errConversationNotWritable", err)
	}

	var msgCount int
	if err := pool.QueryRow(ctx,
		`select count(*) from messages where conversation_id = $1`, convA).Scan(&msgCount); err != nil {
		t.Fatal(err)
	}
	if msgCount != 0 {
		t.Errorf("messages in %s's conversation after uB's rejected save = %d, want 0", uA, msgCount)
	}
}

// TestLoadConversationSummariesOrdersNewestFirst proves the sidebar's list
// is ordered by created_at desc and reports a fallback
// title for a row saved before titles existed.
func TestLoadConversationSummariesOrdersNewestFirst(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	list := loadConversationSummaries(pool)

	const id = "conv_list_test"
	newTestUser(t, pool, ctx, id)

	oldID, newID := uuid.NewString(), uuid.NewString()
	if _, err := pool.Exec(ctx,
		`insert into conversations (id, user_id, title, created_at) values
			($1, $2, null, now() - interval '1 hour'),
			($3, $2, 'a heist movie', now())`,
		oldID, id, newID); err != nil {
		t.Fatal(err)
	}

	got, err := list(ctx, id)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	want := []conversationSummary{
		{ID: newID, Title: "a heist movie"},
		{ID: oldID, Title: defaultConversationTitle},
	}
	if len(got) != len(want) {
		t.Fatalf("summaries = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("summaries[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// TestLoadConversationSummariesBreaksCreatedAtTiesStably proves a genuine
// created_at tie (two rows sharing the exact same timestamp) still returns
// in a deterministic order, matching id desc.
func TestLoadConversationSummariesBreaksCreatedAtTiesStably(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	list := loadConversationSummaries(pool)

	const id = "conv_tie_test"
	newTestUser(t, pool, ctx, id)

	idA, idB := uuid.NewString(), uuid.NewString()
	tied := time.Now()
	if _, err := pool.Exec(ctx,
		`insert into conversations (id, user_id, title, created_at) values
			($1, $3, 'first', $4),
			($2, $3, 'second', $4)`,
		idA, idB, id, tied); err != nil {
		t.Fatal(err)
	}

	titleByID := map[string]string{idA: "first", idB: "second"}
	firstID, secondID := idA, idB
	if idB > idA {
		firstID, secondID = idB, idA
	}
	want := []conversationSummary{
		{ID: firstID, Title: titleByID[firstID]},
		{ID: secondID, Title: titleByID[secondID]},
	}

	got, err := list(ctx, id)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("summaries = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("summaries[%d] = %+v, want %+v (id desc must break the tie deterministically)", i, got[i], want[i])
		}
	}
}

// TestDeleteConversationCascadesMessages proves deleteConversation removes
// both the conversation row and its messages via the foreign key's `on
// delete cascade` (20260904191046_conversations.sql).
func TestDeleteConversationCascadesMessages(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	del := deleteConversation(pool)

	const id = "conv_delete_test"
	newTestUser(t, pool, ctx, id)
	convID := uuid.NewString()
	if _, err := pool.Exec(ctx,
		`insert into conversations (id, user_id) values ($1, $2)`, convID, id); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx,
		`insert into messages (conversation_id, role, content) values ($1, 'user', 'hi')`, convID); err != nil {
		t.Fatal(err)
	}

	deleted, err := del(ctx, id, convID)
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	if !deleted {
		t.Error("deleted = false, want true for a row that actually existed")
	}

	var convCount, msgCount int
	if err := pool.QueryRow(ctx, `select count(*) from conversations where id = $1`, convID).Scan(&convCount); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `select count(*) from messages where conversation_id = $1`, convID).Scan(&msgCount); err != nil {
		t.Fatal(err)
	}
	if convCount != 0 || msgCount != 0 {
		t.Errorf("after delete: conversations = %d, messages = %d, want 0, 0", convCount, msgCount)
	}
}

// TestDeleteConversationIsIdempotent proves deleting an already-gone or
// never-existed conversation id is a silent success, matching
// setSubscription/setVerdict's existing DELETE convention.
func TestDeleteConversationIsIdempotent(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	del := deleteConversation(pool)

	const id = "conv_delete_idempotent_test"
	newTestUser(t, pool, ctx, id)

	deleted, err := del(ctx, id, uuid.NewString())
	if err != nil {
		t.Errorf("delete of a nonexistent conversation returned an error: %v", err)
	}
	if deleted {
		t.Error("deleted = true, want false for a conversation that never existed")
	}
}

// TestAgentRoleCanReadConversationsAndMessages is agent_ro's first positive
// read test — it starts with no grants at all, and the conversations migration
// is what adds these. Proves the select grant and per-user
// policy actually work, not just that writes are still denied.
func TestAgentRoleCanReadConversationsAndMessages(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	const id = "agent_read_test"
	newTestUser(t, pool, ctx, id)
	var convID string
	if err := pool.QueryRow(ctx,
		`insert into conversations (user_id) values ($1) returning id`, id).Scan(&convID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx,
		`insert into messages (conversation_id, role, content) values ($1, 'user', 'hi')`,
		convID); err != nil {
		t.Fatal(err)
	}

	asRole(t, pool, "agent_ro", func(ctx context.Context, tx pgx.Tx) {
		if _, err := tx.Exec(ctx, `select set_config('holster.user_id', $1, true)`, id); err != nil {
			t.Fatalf("set_config: %v", err)
		}

		var gotConvID string
		if err := tx.QueryRow(ctx, `select id from conversations where user_id = $1`, id).
			Scan(&gotConvID); err != nil {
			t.Errorf("agent_ro select on conversations: %v", err)
		} else if gotConvID != convID {
			t.Errorf("conversation id = %q, want %q", gotConvID, convID)
		}

		var content string
		if err := tx.QueryRow(ctx, `select content from messages where conversation_id = $1`, convID).
			Scan(&content); err != nil {
			t.Errorf("agent_ro select on messages: %v", err)
		} else if content != "hi" {
			t.Errorf("content = %q, want %q", content, "hi")
		}
	})

	// Extends TestAgentRoleIsReadOnly's coverage rather than duplicating its
	// whole table: conversations/messages are agent_ro's first-ever grant, so
	// the negative case matters here specifically, not just generically.
	asRole(t, pool, "agent_ro", func(ctx context.Context, tx pgx.Tx) {
		if _, err := tx.Exec(ctx, `select set_config('holster.user_id', $1, true)`, id); err != nil {
			t.Fatalf("set_config: %v", err)
		}
		writes := map[string]string{
			"insert conversation": `insert into conversations (user_id) values ('` + id + `')`,
			"insert message":      `insert into messages (conversation_id, role, content) values ('` + convID + `', 'user', 'x')`,
			"delete message":      `delete from messages where conversation_id = '` + convID + `'`,
		}
		for name, sql := range writes {
			if err := probe(t, ctx, tx, sql); !denied(err) {
				t.Errorf("%s: err = %v, want SQLSTATE 42501", name, err)
			}
		}
	})
}

// TestMessagePolicyEnforcesUserIsolation covers the one policy shape in this
// schema that isn't a direct user_id comparison: messages has no user_id
// column, so its isolation predicate joins through conversations via a
// subquery (20260904191046_conversations.sql). Proves that shape actually
// denies cross-user access rather than assuming the direct-column pattern
// (TestGatewayRoleEnforcesUserIsolation) transfers unverified.
func TestMessagePolicyEnforcesUserIsolation(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	const uA, uB = "msg_rls_a", "msg_rls_b"
	newTestUser(t, pool, ctx, uA)
	newTestUser(t, pool, ctx, uB)

	var convA string
	if err := pool.QueryRow(ctx,
		`insert into conversations (user_id) values ($1) returning id`, uA).Scan(&convA); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx,
		`insert into messages (conversation_id, role, content) values ($1, 'user', 'a-only')`,
		convA); err != nil {
		t.Fatal(err)
	}

	asRole(t, pool, "gateway_app", func(ctx context.Context, tx pgx.Tx) {
		if _, err := tx.Exec(ctx, `select set_config('holster.user_id', $1, true)`, uB); err != nil {
			t.Fatalf("set_config: %v", err)
		}

		var count int
		if err := tx.QueryRow(ctx, `select count(*) from messages where conversation_id = $1`, convA).
			Scan(&count); err != nil {
			t.Fatal(err)
		}
		if count != 0 {
			t.Errorf("as %s, visible messages in %s's conversation = %d, want 0", uB, uA, count)
		}

		if err := probe(t, ctx, tx,
			`insert into messages (conversation_id, role, content) values ('`+convA+`', 'user', 'intrusion')`); !denied(err) {
			t.Errorf("writing into %s's conversation while acting as %s: err = %v, want SQLSTATE 42501", uA, uB, err)
		}
	})
}

// --- webhooks.go -------------------------------------------------------------

// TestDeleteUser proves the task's done-when line directly: removing the
// users row takes every dependent table with it via ON DELETE CASCADE, and a
// second call against an already-deleted id (Clerk retries webhook delivery)
// is a no-op, not an error.
func TestDeleteUser(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	remove := deleteUser(pool)

	const id = "webhook_delete_test"
	newTestUser(t, pool, ctx, id)

	if _, err := pool.Exec(ctx,
		`insert into streaming_subscriptions (user_id, tmdb_provider_id) values ($1, 8)
		 on conflict do nothing`, id); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx,
		`insert into title_verdicts (user_id, tmdb_id, media_type, verdict) values ($1, 550, 'movie', 'liked')
		 on conflict do nothing`, id); err != nil {
		t.Fatal(err)
	}
	convID := uuid.NewString()
	if _, err := pool.Exec(ctx,
		`insert into conversations (id, user_id) values ($1, $2)`, convID, id); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx,
		`insert into messages (conversation_id, role, content) values ($1, 'user', 'hi')`, convID); err != nil {
		t.Fatal(err)
	}

	if err := remove(ctx, id); err != nil {
		t.Fatalf("delete: %v", err)
	}

	counts := map[string]string{
		"users":                   `select count(*) from users where id = $1`,
		"streaming_subscriptions": `select count(*) from streaming_subscriptions where user_id = $1`,
		"title_verdicts":          `select count(*) from title_verdicts where user_id = $1`,
		"conversations":           `select count(*) from conversations where user_id = $1`,
	}
	for table, query := range counts {
		var n int
		if err := pool.QueryRow(ctx, query, id).Scan(&n); err != nil {
			t.Fatal(err)
		}
		if n != 0 {
			t.Errorf("%s rows for %s after delete = %d, want 0", table, id, n)
		}
	}
	var msgCount int
	if err := pool.QueryRow(ctx, `select count(*) from messages where conversation_id = $1`, convID).
		Scan(&msgCount); err != nil {
		t.Fatal(err)
	}
	if msgCount != 0 {
		t.Errorf("messages for %s after delete = %d, want 0", convID, msgCount)
	}

	// Idempotent: a second call against a user already gone must not error.
	if err := remove(ctx, id); err != nil {
		t.Errorf("repeat delete on an already-gone user: %v", err)
	}
}

// TestUpdateUserEmail proves user.updated's write: a changed email is
// stored, and a same-email call and a call for a nonexistent id are both
// silent no-ops.
func TestUpdateUserEmail(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	update := updateUserEmail(pool)

	const id = "webhook_email_test"
	newTestUser(t, pool, ctx, id)

	email := func() string {
		var e string
		if err := pool.QueryRow(ctx, `select email from users where id = $1`, id).Scan(&e); err != nil {
			t.Fatalf("select: %v", err)
		}
		return e
	}

	if got := email(); got != id+"@example.com" {
		t.Fatalf("seed email = %q, want %q", got, id+"@example.com")
	}

	if err := update(ctx, id, "updated@example.com"); err != nil {
		t.Fatalf("update: %v", err)
	}
	if got := email(); got != "updated@example.com" {
		t.Errorf("email = %q, want %q", got, "updated@example.com")
	}

	// Same email again is a no-op, not an error.
	if err := update(ctx, id, "updated@example.com"); err != nil {
		t.Errorf("idempotent re-update: %v", err)
	}

	// A user who has never hit the gateway has no row -- silently doing
	// nothing here is deliberate (see updateUserEmail's doc comment):
	// upsertUser provisions the row on their first real request.
	if err := update(ctx, "webhook_email_nonexistent", "new@example.com"); err != nil {
		t.Errorf("update for a nonexistent id: %v", err)
	}
}

// cachedProviderNames is the one cached-row read both loadChatContext (via
// its transaction) and loadGuestChatContext (straight off the pool) share.
// A throwaway 'ZZ' row, as TestLoadChatContext uses — never a real country's.
func TestCachedProviderNames(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	t.Cleanup(func() {
		c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		pool.Exec(c, `delete from streaming_providers where country = 'ZZ'`)
	})

	if _, err := pool.Exec(ctx, `delete from streaming_providers where country = 'ZZ'`); err != nil {
		t.Fatal(err)
	}
	names, err := cachedProviderNames(ctx, pool, "ZZ", []int{8})
	if err != nil {
		t.Fatalf("with no cache row: %v", err)
	}
	if names == nil || len(names) != 0 {
		t.Errorf("names = %#v, want non-nil empty without a cache row", names)
	}

	if _, err := pool.Exec(ctx,
		`insert into streaming_providers (country, providers) values ('ZZ', $1)
		 on conflict (country) do update set providers = excluded.providers`,
		`[{"provider_id": 8, "provider_name": "Netflix"}, {"provider_id": 15, "provider_name": "Hulu"}]`); err != nil {
		t.Fatal(err)
	}
	names, err = cachedProviderNames(ctx, pool, "ZZ", []int{15, 8, 999})
	if err != nil {
		t.Fatalf("with cache row: %v", err)
	}
	if want := []string{"Hulu", "Netflix"}; len(names) != 2 || names[0] != want[0] || names[1] != want[1] {
		t.Errorf("names = %v, want %v (caller's order, unknown ids dropped)", names, want)
	}
}
