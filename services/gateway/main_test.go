package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math/big"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MicahParks/keyfunc/v3"
	"github.com/golang-jwt/jwt/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

const (
	testIssuer   = "https://holster.clerk.accounts.dev"
	testOrigin   = "https://holster.app"
	testAudience = "holster-gateway"
)

func jwksJSON(kid string, key *rsa.PrivateKey) []byte {
	b, err := json.Marshal(map[string]any{"keys": []map[string]string{{
		"kty": "RSA", "kid": kid, "use": "sig", "alg": "RS256",
		"n": base64.RawURLEncoding.EncodeToString(key.N.Bytes()),
		"e": base64.RawURLEncoding.EncodeToString(big.NewInt(int64(key.E)).Bytes()),
	}}})
	if err != nil {
		panic(err)
	}
	return b
}

func mint(t *testing.T, kid string, key *rsa.PrivateKey, claims *Claims) string {
	t.Helper()
	token := jwt.NewWithClaims(jwt.SigningMethodRS256, claims)
	token.Header["kid"] = kid
	signed, err := token.SignedString(key)
	if err != nil {
		t.Fatal(err)
	}
	return signed
}

// validClaims returns claims a real Clerk session token would carry.
func validClaims() *Claims {
	return &Claims{Azp: testOrigin, Email: "user@example.com",
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   "user_abc",
			Issuer:    testIssuer,
			Audience:  jwt.ClaimStrings{testAudience},
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Minute)),
		}}
}

// newTestHandler serves the given key from a stub JWKS endpoint.
func newTestHandler(t *testing.T, key *rsa.PrivateKey, kid string) *Handler {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(jwksJSON(kid, key))
	}))
	t.Cleanup(srv.Close)

	jwks, err := keyfunc.NewDefaultCtx(t.Context(), []string{srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	// Provisioning is a no-op here; the SQL has its own test.
	h, err := newHandler(jwks.Keyfunc, testIssuer, testAudience, map[string]struct{}{testOrigin: {}},
		func(context.Context, string, string) error { return nil },
		noopChatCtx, noopAgentCaller, t.Context(),
		noopLoadProviders, noopSaveSubscription,
	)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

// Chat's own dependencies, for the many tests below that exercise auth/
// provisioning only and never reach a chat endpoint — chat_test.go covers
// loadChatCtx/callAgent for real.
func noopChatCtx(context.Context, string) (chatContext, error) { return chatContext{}, nil }
func noopAgentCaller(context.Context, agentChatRequest) (<-chan agentEvent, error) {
	return nil, nil
}

// providers.go's own dependencies, for the same reason — providers_test.go
// covers loadProviders/saveSubscription for real.
func noopLoadProviders(context.Context, string) ([]Provider, error) { return nil, nil }
func noopSaveSubscription(context.Context, string, int, bool) error { return nil }

func TestVerifyTokenAcceptsValidToken(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")

	claims, err := h.verifyToken(mint(t, "kid_A", key, validClaims()))
	if err != nil {
		t.Fatalf("valid token rejected: %v", err)
	}
	if claims.Subject != "user_abc" {
		t.Errorf("subject = %q, want %q", claims.Subject, "user_abc")
	}
}

func TestVerifyTokenRejects(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")

	mutate := map[string]func(*Claims){
		"expired":      func(c *Claims) { c.ExpiresAt = jwt.NewNumericDate(time.Now().Add(-time.Hour)) },
		"no exp":       func(c *Claims) { c.ExpiresAt = nil },
		"empty sub":    func(c *Claims) { c.Subject = "" },
		"wrong issuer": func(c *Claims) { c.Issuer = "https://other.clerk.accounts.dev" },
		"wrong azp":    func(c *Claims) { c.Azp = "https://evil.example.com" },
		"absent azp":   func(c *Claims) { c.Azp = "" },
		"future nbf":   func(c *Claims) { c.NotBefore = jwt.NewNumericDate(time.Now().Add(time.Hour)) },
	}

	for name, apply := range mutate {
		t.Run(name, func(t *testing.T) {
			claims := validClaims()
			apply(claims)
			if _, err := h.verifyToken(mint(t, "kid_A", key, claims)); err == nil {
				t.Errorf("accepted a token that should be rejected (%s)", name)
			}
		})
	}

	t.Run("forged signature", func(t *testing.T) {
		other, _ := rsa.GenerateKey(rand.Reader, 2048)
		if _, err := h.verifyToken(mint(t, "kid_A", other, validClaims())); err == nil {
			t.Error("accepted a token signed by an unknown key")
		}
	})

	t.Run("alg none", func(t *testing.T) {
		token := jwt.NewWithClaims(jwt.SigningMethodNone, validClaims())
		token.Header["kid"] = "kid_A"
		signed, err := token.SignedString(jwt.UnsafeAllowNoneSignatureType)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := h.verifyToken(signed); err == nil {
			t.Error("accepted an unsigned token")
		}
	})

	t.Run("unknown kid", func(t *testing.T) {
		if _, err := h.verifyToken(mint(t, "kid_missing", key, validClaims())); err == nil {
			t.Error("accepted a token with an unknown key id")
		}
	})

	// Without this the key set is searched exhaustively instead of by key id.
	t.Run("no kid header", func(t *testing.T) {
		token := jwt.NewWithClaims(jwt.SigningMethodRS256, validClaims())
		signed, err := token.SignedString(key)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := h.verifyToken(signed); err == nil {
			t.Error("accepted a token with no kid header")
		}
	})

	t.Run("no issuer", func(t *testing.T) {
		claims := validClaims()
		claims.Issuer = ""
		if _, err := h.verifyToken(mint(t, "kid_A", key, claims)); err == nil {
			t.Error("accepted a token with no iss claim")
		}
	})
}

// Rejections must carry a distinguishable reason, so a test cannot pass because
// the right token failed for the wrong cause.
func TestRejectionReasons(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")

	noSub := validClaims()
	noSub.Subject = ""
	badAzp := validClaims()
	badAzp.Azp = "https://evil.example.com"
	expired := validClaims()
	expired.ExpiresAt = jwt.NewNumericDate(time.Now().Add(-time.Hour))

	cases := []struct {
		name   string
		claims *Claims
		want   error
	}{
		{"empty sub", noSub, errNoSubject},
		{"wrong azp", badAzp, errUnauthorizedParty},
		{"expired", expired, jwt.ErrTokenExpired},
	}
	for _, c := range cases {
		_, err := h.verifyToken(mint(t, "kid_A", key, c.claims))
		if !errors.Is(err, c.want) {
			t.Errorf("%s: err = %v, want errors.Is(_, %v)", c.name, err, c.want)
		}
	}

	token := jwt.NewWithClaims(jwt.SigningMethodRS256, validClaims())
	signed, err := token.SignedString(key)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.verifyToken(signed); !errors.Is(err, errNoKID) {
		t.Errorf("no kid: err = %v, want errors.Is(_, errNoKID)", err)
	}
}

// A token minted for a different consumer of the same Clerk instance must not
// authenticate here. RFC 8725 requires the audience check; azp is not a substitute.
func TestAudienceIsRequired(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")

	wrong := validClaims()
	wrong.Audience = jwt.ClaimStrings{"some-other-service"}
	if _, err := h.verifyToken(mint(t, "kid_A", key, wrong)); err == nil {
		t.Error("accepted a token minted for another audience")
	}

	absent := validClaims()
	absent.Audience = nil
	_, err := h.verifyToken(mint(t, "kid_A", key, absent))
	if err == nil {
		t.Error("accepted a token with no audience — the default session token shape")
	}
	// A wholly-absent aud reports missing_claim, not bad_audience: jwt collapses
	// "required claim absent" into one error and cannot say which claim. Both
	// reject and both are anomalies; this pins the label a dashboard would see.
	if reason, _ := authFailure(err); reason != "missing_claim" {
		t.Errorf("no-audience reason = %q, want %q", reason, "missing_claim")
	}
}

// The Clerk instance requires email sign-up, so a token without the claim is
// malformed. Not a schema constraint — an absent claim unmarshals to "", which
// users.email NOT NULL would accept.
func TestEmailIsRequired(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")

	noEmail := validClaims()
	noEmail.Email = ""
	_, err := h.verifyToken(mint(t, "kid_A", key, noEmail))
	if !errors.Is(err, errNoEmail) {
		t.Errorf("err = %v, want errors.Is(_, errNoEmail)", err)
	}
	if reason, _ := authFailure(err); reason != "no_email" {
		t.Errorf("reason = %q, want %q", reason, "no_email")
	}
}

// Provisioning runs on every authenticated request and carries the token's
// identity, not anything the caller supplied.
func TestProvisioningReceivesTokenIdentity(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(jwksJSON("kid_A", key))
	}))
	defer srv.Close()
	jwks, err := keyfunc.NewDefaultCtx(t.Context(), []string{srv.URL})
	if err != nil {
		t.Fatal(err)
	}

	var gotID, gotEmail string
	var calls int
	h, err := newHandler(jwks.Keyfunc, testIssuer, testAudience, map[string]struct{}{testOrigin: {}},
		func(_ context.Context, id, email string) error {
			calls++
			gotID, gotEmail = id, email
			return nil
		},
		noopChatCtx, noopAgentCaller, t.Context(), noopLoadProviders, noopSaveSubscription)
	if err != nil {
		t.Fatal(err)
	}

	handler := correlationID(h.authMiddleware(http.HandlerFunc(h.example)))
	req := httptest.NewRequest("GET", "/api/example", nil)
	req.Header.Set("Authorization", "Bearer "+mint(t, "kid_A", key, validClaims()))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if calls != 1 {
		t.Errorf("provisioner called %d times, want 1", calls)
	}
	if gotID != "user_abc" || gotEmail != "user@example.com" {
		t.Errorf("provisioned (%q, %q), want (user_abc, user@example.com)", gotID, gotEmail)
	}

	// A rejected token must not reach provisioning at all.
	calls = 0
	req = httptest.NewRequest("GET", "/api/example", nil)
	req.Header.Set("Authorization", "Bearer aaa.bbb.ccc")
	handler.ServeHTTP(httptest.NewRecorder(), req)
	if calls != 0 {
		t.Errorf("provisioner ran for an unauthenticated request")
	}
}

// A provisioning failure must not let the request through: anything with a user
// foreign key would fail later and less clearly.
func TestProvisioningFailureBlocksRequest(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(jwksJSON("kid_A", key))
	}))
	defer srv.Close()
	jwks, err := keyfunc.NewDefaultCtx(t.Context(), []string{srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	h, err := newHandler(jwks.Keyfunc, testIssuer, testAudience, map[string]struct{}{testOrigin: {}},
		func(context.Context, string, string) error { return errors.New("connection refused") },
		noopChatCtx, noopAgentCaller, t.Context(), noopLoadProviders, noopSaveSubscription)
	if err != nil {
		t.Fatal(err)
	}

	reached := false
	handler := correlationID(h.authMiddleware(http.HandlerFunc(
		func(http.ResponseWriter, *http.Request) { reached = true })))
	req := httptest.NewRequest("GET", "/api/example", nil)
	req.Header.Set("Authorization", "Bearer "+mint(t, "kid_A", key, validClaims()))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", rec.Code)
	}
	if reached {
		t.Error("handler ran despite provisioning failing")
	}
}

// A caller that hangs up mid-request is not an outage. Ordinary navigation
// cancels the request context, and filing that as a provisioning failure puts it
// in the same signal as a database that is genuinely down.
func TestClientDisconnectIsNotAProvisioningFailure(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(jwksJSON("kid_A", key))
	}))
	defer srv.Close()
	jwks, err := keyfunc.NewDefaultCtx(t.Context(), []string{srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	h, err := newHandler(jwks.Keyfunc, testIssuer, testAudience, map[string]struct{}{testOrigin: {}},
		func(ctx context.Context, _, _ string) error { return ctx.Err() },
		noopChatCtx, noopAgentCaller, t.Context(), noopLoadProviders, noopSaveSubscription)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil))) })

	reached := false
	handler := correlationID(h.authMiddleware(http.HandlerFunc(
		func(http.ResponseWriter, *http.Request) { reached = true })))

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // the client is already gone
	req := httptest.NewRequest("GET", "/api/example", nil).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer "+mint(t, "kid_A", key, validClaims()))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if reached {
		t.Error("handler ran despite provisioning being cancelled")
	}
	if buf.Len() != 0 {
		t.Errorf("logged %q, want nothing for a client that hung up", buf.String())
	}
	if rec.Body.Len() != 0 {
		t.Errorf("wrote %q to a connection that is already gone", rec.Body.String())
	}
}

// A database that accepts the connection but never answers must surface as a
// 503 on the provisioning deadline, not hang the request until the client or
// the OS gives up.
func TestProvisioningDeadlineReturns503(t *testing.T) {
	old := provisionTimeout
	provisionTimeout = 50 * time.Millisecond
	t.Cleanup(func() { provisionTimeout = old })

	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(jwksJSON("kid_A", key))
	}))
	defer srv.Close()
	jwks, err := keyfunc.NewDefaultCtx(t.Context(), []string{srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	h, err := newHandler(jwks.Keyfunc, testIssuer, testAudience, map[string]struct{}{testOrigin: {}},
		func(ctx context.Context, _, _ string) error {
			<-ctx.Done()
			return ctx.Err()
		},
		noopChatCtx, noopAgentCaller, t.Context(), noopLoadProviders, noopSaveSubscription)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil))) })

	reached := false
	handler := correlationID(h.authMiddleware(http.HandlerFunc(
		func(http.ResponseWriter, *http.Request) { reached = true })))
	req := httptest.NewRequest("GET", "/api/example", nil)
	req.Header.Set("Authorization", "Bearer "+mint(t, "kid_A", key, validClaims()))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", rec.Code)
	}
	if reached {
		t.Error("handler ran despite provisioning timing out")
	}
	if buf.Len() == 0 {
		t.Error("provisioning timeout logged nothing")
	}
}

func TestNewHandlerRejectsEmptyParties(t *testing.T) {
	kf := func(*jwt.Token) (any, error) { return nil, nil }
	noop := func(context.Context, string, string) error { return nil }
	_, err := newHandler(kf, testIssuer, testAudience, map[string]struct{}{}, noop,
		noopChatCtx, noopAgentCaller, t.Context(), noopLoadProviders, noopSaveSubscription)
	if err == nil {
		t.Error("built a Handler with no authorized parties, which would reject every request")
	}
}

// A server-side error must reduce to its SQLSTATE — a constraint violation's
// Message and Detail carry row values. Everything else logs in full, because
// losing the connection error is losing the only diagnosis of an outage.
func TestDBErrorHidesRowDataOnly(t *testing.T) {
	pgErr := &pgconn.PgError{
		Code:    "23505",
		Message: `duplicate key value violates unique constraint "users_pkey"`,
		Detail:  `Key (id)=(user_abc) already exists.`,
	}
	if got := dbError(fmt.Errorf("wrapped: %w", pgErr)); got != "23505" {
		t.Errorf("dbError(PgError) = %q, want %q", got, "23505")
	}

	plain := errors.New("failed to connect to `user=postgres database=holster`: dial error")
	if got := dbError(plain); got != plain.Error() {
		t.Errorf("dbError(connection error) = %q, want the full message", got)
	}
}

// authFailure is a log field and a future metric label, so its range must stay
// closed and must never carry token-derived data.
func TestAuthFailureIsClosedSet(t *testing.T) {
	allowed := map[string]bool{
		"expired": true, "malformed": true, "not_yet_valid": true,
		"no_kid": true, "no_subject": true, "no_email": true,
		"unauthorized_party": true, "bad_signature": true, "bad_issuer": true,
		"bad_audience": true, "missing_claim": true,
		"unverifiable": true, "other": true,
	}

	// "unverifiable" means no key could be obtained. An unrelated failure must
	// not borrow that label and send the reader to JWKS.
	for _, err := range []error{
		errors.New("database exploded"),
		context.DeadlineExceeded,
	} {
		if reason, _ := authFailure(err); reason != "other" {
			t.Errorf("authFailure(%v) = %q, want %q", err, reason, "other")
		}
	}
	routine := map[string]bool{"expired": true, "malformed": true, "not_yet_valid": true}

	for _, err := range []error{
		jwt.ErrTokenExpired, jwt.ErrTokenMalformed, jwt.ErrTokenNotValidYet,
		errNoKID, errNoSubject, errNoEmail, errUnauthorizedParty,
		jwt.ErrTokenSignatureInvalid, jwt.ErrTokenInvalidIssuer,
		jwt.ErrTokenInvalidAudience,
		jwt.ErrTokenRequiredClaimMissing, jwt.ErrTokenUnverifiable,
		errors.New("something nobody anticipated"),
		fmt.Errorf("attacker controlled %s", strings.Repeat("A", 5000)),
	} {
		reason, anomaly := authFailure(fmt.Errorf("wrapped: %w", err))
		if !allowed[reason] {
			t.Errorf("authFailure(%v) = %q, outside the closed set", err, reason)
		}
		if anomaly == routine[reason] {
			t.Errorf("authFailure(%v) = %q, anomaly=%v — misclassified", err, reason, anomaly)
		}
	}
}

// Nothing token-derived may reach the log, at any size. The raw error is not
// logged precisely because keyfunc embeds the key id in it.
func TestOversizedKIDNeverReachesLog(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")

	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil))) })

	srv := correlationID(h.authMiddleware(http.HandlerFunc(h.example)))
	req := httptest.NewRequest("GET", "/api/example", nil)
	req.Header.Set("Authorization", "Bearer "+mint(t, strings.Repeat("P", 4000), key, validClaims()))
	srv.ServeHTTP(httptest.NewRecorder(), req)

	t.Logf("log line for a 4000-char kid: %d bytes", buf.Len())
	if bytes.Contains(buf.Bytes(), []byte("PPPP")) {
		t.Error("the key id reached the log")
	}
	if buf.Len() > 300 {
		t.Errorf("log line is %d bytes, want a bounded reason code", buf.Len())
	}
}

// Routine lifecycle logs at DEBUG, anomalies at WARN, so verbosity is a level
// decision the environment makes rather than something dropped in code.
func TestLogLevelByFailureKind(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")

	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil))) })
	srv := correlationID(h.authMiddleware(http.HandlerFunc(h.example)))

	send := func(token string) string {
		buf.Reset()
		req := httptest.NewRequest("GET", "/api/example", nil)
		req.Header.Set("Authorization", "Bearer "+token)
		srv.ServeHTTP(httptest.NewRecorder(), req)
		var rec map[string]any
		if err := json.Unmarshal(buf.Bytes(), &rec); err != nil {
			t.Fatalf("no log record: %v (%q)", err, buf.String())
		}
		return rec["level"].(string) + " " + rec["reason"].(string)
	}

	expired := validClaims()
	expired.ExpiresAt = jwt.NewNumericDate(time.Now().Add(-time.Hour))
	if got := send(mint(t, "kid_A", key, expired)); got != "DEBUG expired" {
		t.Errorf("expired token logged as %q, want \"DEBUG expired\"", got)
	}

	other, _ := rsa.GenerateKey(rand.Reader, 2048)
	if got := send(mint(t, "kid_A", other, validClaims())); got != "WARN bad_signature" {
		t.Errorf("forged signature logged as %q, want \"WARN bad_signature\"", got)
	}

	badAzp := validClaims()
	badAzp.Azp = "https://evil.example.com"
	if got := send(mint(t, "kid_A", key, badAzp)); got != "WARN unauthorized_party" {
		t.Errorf("wrong azp logged as %q, want \"WARN unauthorized_party\"", got)
	}
}

// The default level must be quiet: a deploy that configures nothing must not
// emit a line per expired token, which is the highest-volume rejection there is.
func TestDefaultLevelSuppressesRoutineFailures(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")

	var buf bytes.Buffer
	var level slog.LevelVar // zero value is LevelInfo, matching the default
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: &level})))
	t.Cleanup(func() { slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil))) })
	srv := correlationID(h.authMiddleware(http.HandlerFunc(h.example)))

	expired := validClaims()
	expired.ExpiresAt = jwt.NewNumericDate(time.Now().Add(-time.Hour))
	for range 1000 {
		req := httptest.NewRequest("GET", "/api/example", nil)
		req.Header.Set("Authorization", "Bearer "+mint(t, "kid_A", key, expired))
		srv.ServeHTTP(httptest.NewRecorder(), req)
	}
	if buf.Len() != 0 {
		t.Errorf("1000 expired tokens wrote %d bytes at the default level, want 0", buf.Len())
	}

	// Anomalies must still surface at that level.
	buf.Reset()
	attacker, _ := rsa.GenerateKey(rand.Reader, 2048)
	req := httptest.NewRequest("GET", "/api/example", nil)
	req.Header.Set("Authorization", "Bearer "+mint(t, "kid_A", attacker, validClaims()))
	srv.ServeHTTP(httptest.NewRecorder(), req)
	if !bytes.Contains(buf.Bytes(), []byte(`"reason":"bad_signature"`)) {
		t.Errorf("anomaly suppressed at the default level: %q", buf.String())
	}
}

// A live bearer credential must never reach the log.
func TestTokenNeverLogged(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")

	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil))) })

	attacker, _ := rsa.GenerateKey(rand.Reader, 2048)
	token := mint(t, "kid_A", attacker, validClaims())
	srv := correlationID(h.authMiddleware(http.HandlerFunc(h.example)))
	req := httptest.NewRequest("GET", "/api/example", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	srv.ServeHTTP(httptest.NewRecorder(), req)

	if bytes.Contains(buf.Bytes(), []byte(token)) {
		t.Error("the token was written to the log")
	}
	// The signature segment alone must not leak either.
	if sig := token[strings.LastIndex(token, ".")+1:]; bytes.Contains(buf.Bytes(), []byte(sig)) {
		t.Error("the token signature was written to the log")
	}
}

// T7's done-when: a valid token returns the user ID, without one 401.
func TestRoutesRequireAuthOnEveryMethod(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")

	// Register a future-shaped write route to prove it inherits auth. (The
	// real PUT/DELETE /api/subscriptions/{providerID} now exists — see
	// providers.go and providers_test.go — so it is no longer a placeholder
	// here; this one stands in for any route not yet built.)
	api := http.NewServeMux()
	api.HandleFunc("GET /api/example", h.example)
	api.HandleFunc("POST /api/chat", h.example)
	root := http.NewServeMux()
	root.HandleFunc("GET /health", h.health)
	root.Handle("/api/", h.authMiddleware(api))
	srv := correlationID(root)

	protected := []struct{ method, path string }{
		{"GET", "/api/example"},
		{"POST", "/api/chat"},
		{"DELETE", "/api/not-registered"},
	}
	for _, p := range protected {
		rec := httptest.NewRecorder()
		srv.ServeHTTP(rec, httptest.NewRequest(p.method, p.path, nil))
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%s %s without a token = %d, want 401", p.method, p.path, rec.Code)
		}
		if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
			t.Errorf("%s %s content-type = %q, want application/json", p.method, p.path, ct)
		}
	}

	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest("GET", "/health", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("/health = %d, want 200", rec.Code)
	}

	req := httptest.NewRequest("POST", "/api/chat", nil)
	req.Header.Set("Authorization", "Bearer "+mint(t, "kid_A", key, validClaims()))
	rec = httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("authenticated POST = %d, want 200", rec.Code)
	}
	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["user_id"] != "user_abc" {
		t.Errorf("user_id = %q, want %q", body["user_id"], "user_abc")
	}
	if body["correlation_id"] == "" {
		t.Error("correlation_id missing from response")
	}
	if rec.Header().Get("X-Correlation-ID") == "" {
		t.Error("X-Correlation-ID header missing")
	}
}

// T15: the picker is the first caller to hit /api/ with a plain browser
// fetch() rather than a WebSocket upgrade or a server-side request, so this
// is the first test that actually exercises corsMiddleware.
func TestCORSPreflightFromAllowedOrigin(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")
	srv := h.routes()

	req := httptest.NewRequest(http.MethodOptions, "/api/providers", nil)
	req.Header.Set("Origin", testOrigin)
	req.Header.Set("Access-Control-Request-Method", "GET")
	req.Header.Set("Access-Control-Request-Headers", "authorization")
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Errorf("preflight status = %d, want 204", rec.Code)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != testOrigin {
		t.Errorf("Access-Control-Allow-Origin = %q, want %q", got, testOrigin)
	}
	if rec.Header().Get("Access-Control-Allow-Headers") == "" {
		t.Error("Access-Control-Allow-Headers missing")
	}
}

func TestCORSOmittedForUnknownOrigin(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")
	srv := h.routes()

	req := httptest.NewRequest(http.MethodOptions, "/api/providers", nil)
	req.Header.Set("Origin", "https://evil.example")
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	// Still answers the preflight (so it never hangs waiting on a real
	// server), but with no Allow-Origin — the browser enforces the block, the
	// same way an unrecognised azp fails token verification rather than the
	// gateway refusing to respond at all.
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Errorf("Access-Control-Allow-Origin = %q for an unknown origin, want empty", got)
	}
}

func TestCORSHeadersPresentOnRealResponse(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")
	srv := h.routes()

	// Deliberately no Authorization header: proves CORS headers land even on
	// the 401 an unauthenticated request gets, which is what lets the browser
	// surface that 401 to application code instead of an opaque CORS error.
	req := httptest.NewRequest(http.MethodGet, "/api/providers", nil)
	req.Header.Set("Origin", testOrigin)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != testOrigin {
		t.Errorf("Access-Control-Allow-Origin = %q, want %q", got, testOrigin)
	}
}

func TestMalformedAuthorizationHeader(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")
	srv := correlationID(h.authMiddleware(http.HandlerFunc(h.example)))

	for _, header := range []string{"", "Bearer", "Basic abc", "bearer abc", "Bearer  ", "Bearer a b"} {
		req := httptest.NewRequest("GET", "/api/example", nil)
		if header != "" {
			req.Header.Set("Authorization", header)
		}
		rec := httptest.NewRecorder()
		srv.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("Authorization=%q gave %d, want 401", header, rec.Code)
		}
	}
}

// A rotated signing key must recover without a restart.
func TestKeyRotationSelfHeals(t *testing.T) {
	keyA, _ := rsa.GenerateKey(rand.Reader, 2048)
	keyB, _ := rsa.GenerateKey(rand.Reader, 2048)

	var rotated atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rotated.Load() {
			w.Write(jwksJSON("kid_B", keyB))
		} else {
			w.Write(jwksJSON("kid_A", keyA))
		}
	}))
	defer srv.Close()

	jwks, err := keyfunc.NewDefaultCtx(t.Context(), []string{srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	h, err := newHandler(jwks.Keyfunc, testIssuer, testAudience, map[string]struct{}{testOrigin: {}},
		func(context.Context, string, string) error { return nil },
		noopChatCtx, noopAgentCaller, t.Context(), noopLoadProviders, noopSaveSubscription)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := h.verifyToken(mint(t, "kid_A", keyA, validClaims())); err != nil {
		t.Fatalf("pre-rotation token rejected: %v", err)
	}

	rotated.Store(true)
	if _, err := h.verifyToken(mint(t, "kid_B", keyB, validClaims())); err != nil {
		t.Errorf("post-rotation token rejected, key rotation did not self-heal: %v", err)
	}
}

// Unknown key ids must not turn the gateway into a JWKS flood.
func TestUnknownKidDoesNotAmplify(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	var fetches atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fetches.Add(1)
		w.Write(jwksJSON("kid_A", key))
	}))
	defer srv.Close()

	jwks, err := keyfunc.NewDefaultCtx(t.Context(), []string{srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	h, err := newHandler(jwks.Keyfunc, testIssuer, testAudience, map[string]struct{}{testOrigin: {}},
		func(context.Context, string, string) error { return nil },
		noopChatCtx, noopAgentCaller, t.Context(), noopLoadProviders, noopSaveSubscription)
	if err != nil {
		t.Fatal(err)
	}
	before := fetches.Load()

	const requests = 200
	attacker, _ := rsa.GenerateKey(rand.Reader, 2048)
	var wg sync.WaitGroup
	for range requests {
		wg.Add(1)
		go func() {
			defer wg.Done()
			h.verifyToken(mint(t, "kid_unknown", attacker, validClaims()))
		}()
	}
	wg.Wait()

	if got := fetches.Load() - before; got > 20 {
		t.Errorf("%d unknown-kid requests caused %d JWKS fetches, want <= 20", requests, got)
	}
}

func TestAuthorizedParties(t *testing.T) {
	cases := map[string]int{
		"https://holster.app":           1,
		"https://holster.app,":          1,
		"https://holster.app, ":         1,
		"https://a.app,,https://b.app":  2,
		"https://a.app , https://b.app": 2,
		" ":                             0,
		"":                              0,
	}
	for env, want := range cases {
		parties := authorizedParties(env)
		if len(parties) != want {
			t.Errorf("authorizedParties(%q) has %d entries, want %d", env, len(parties), want)
		}
		if _, empty := parties[""]; empty {
			t.Errorf("authorizedParties(%q) admits an empty azp", env)
		}
	}
}
