package main

import (
	"bytes"
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
	"unicode/utf8"

	"github.com/MicahParks/keyfunc/v3"
	"github.com/golang-jwt/jwt/v5"
)

const (
	testIssuer = "https://holster.clerk.accounts.dev"
	testOrigin = "https://holster.app"
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
	return &Claims{Azp: testOrigin, RegisteredClaims: jwt.RegisteredClaims{
		Subject:   "user_abc",
		Issuer:    testIssuer,
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
	h, err := newHandler(jwks.Keyfunc, testIssuer, map[string]struct{}{testOrigin: {}})
	if err != nil {
		t.Fatal(err)
	}
	return h
}

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

func TestNewHandlerRejectsEmptyParties(t *testing.T) {
	kf := func(*jwt.Token) (any, error) { return nil, nil }
	if _, err := newHandler(kf, testIssuer, map[string]struct{}{}); err == nil {
		t.Error("built a Handler with no authorized parties, which would reject every request")
	}
}

// failureClass must stay a closed set: the sampler keys on it, so an unbounded
// label set would be a memory-exhaustion vector.
func TestFailureClassIsClosed(t *testing.T) {
	allowed := map[string]bool{
		"no_kid": true, "no_subject": true, "unauthorized_party": true,
		"bad_signature": true, "bad_issuer": true, "missing_claim": true,
		"unverifiable": true, "other": true,
	}
	inputs := []error{
		errNoKID, errNoSubject, errUnauthorizedParty,
		jwt.ErrTokenSignatureInvalid, jwt.ErrTokenInvalidIssuer,
		jwt.ErrTokenRequiredClaimMissing, jwt.ErrTokenUnverifiable,
		errors.New("something nobody anticipated"),
		fmt.Errorf("attacker controlled %s", strings.Repeat("A", 5000)),
	}
	for _, err := range inputs {
		if class := failureClass(err); !allowed[class] {
			t.Errorf("failureClass(%v) = %q, outside the closed set", err, class)
		}
	}
}

func TestTruncateError(t *testing.T) {
	short := errors.New("token has no subject")
	if got := truncateError(short); got != short.Error() {
		t.Errorf("short error was altered: %q", got)
	}

	long := fmt.Errorf(`key not found %q`, strings.Repeat("P", 4000))
	got := truncateError(long)
	if len(got) > maxLoggedError+len("…(truncated)") {
		t.Errorf("truncated length = %d, want <= %d", len(got), maxLoggedError+len("…(truncated)"))
	}
	if !strings.HasSuffix(got, "(truncated)") {
		t.Errorf("truncation not marked: %q", got)
	}

	// A multi-byte sequence must not be split into invalid UTF-8, or the log
	// handler will emit replacement characters for attacker-chosen input.
	multi := errors.New(strings.Repeat("é", 4000))
	if got := truncateError(multi); !utf8.ValidString(got) {
		t.Error("truncation produced invalid UTF-8")
	}
}

// Attacker-controlled data reaches error strings via the kid. keyfunc's refresh
// limiter changes the error after the first request, so this asserts on the
// FIRST one — testing later requests reads a false negative.
func TestOversizedKIDDoesNotInflateLog(t *testing.T) {
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
	if buf.Len() > 1000 {
		t.Errorf("a 4000-char kid produced a %d byte log line, want bounded", buf.Len())
	}
	// A run longer than the truncation limit means truncation did not apply.
	if bytes.Contains(buf.Bytes(), bytes.Repeat([]byte("P"), maxLoggedError+1)) {
		t.Error("the oversized kid reached the log untruncated")
	}
}

// One noisy class must not consume another's budget.
func TestSamplerDoesNotStarveOtherClasses(t *testing.T) {
	s := newLogSampler(time.Second, 5)

	flooded := 0
	for range 10000 {
		if ok, _ := s.allow("bad_signature"); ok {
			flooded++
		}
	}
	if flooded > 5 {
		t.Errorf("flood produced %d lines, want <= burst of 5", flooded)
	}

	genuine := 0
	for range 50 {
		if ok, _ := s.allow("unauthorized_party"); ok {
			genuine++
		}
	}
	if genuine == 0 {
		t.Error("a different class was starved out by the flood")
	}
}

// Suppression must not be silent.
func TestSamplerReportsDrops(t *testing.T) {
	s := newLogSampler(10*time.Millisecond, 1)
	if ok, dropped := s.allow("k"); !ok || dropped != 0 {
		t.Fatalf("first call: ok=%v dropped=%d, want true/0", ok, dropped)
	}
	for range 99 {
		s.allow("k")
	}
	time.Sleep(20 * time.Millisecond)
	ok, dropped := s.allow("k")
	if !ok {
		t.Fatal("limiter did not refill")
	}
	if dropped != 99 {
		t.Errorf("dropped = %d, want 99", dropped)
	}
	// The counter resets once reported, so drops are not double counted.
	time.Sleep(20 * time.Millisecond)
	if _, dropped := s.allow("k"); dropped != 0 {
		t.Errorf("dropped = %d after reporting, want 0", dropped)
	}
}

// Routine token lifecycle must not be logged; anomalies must be.
func TestRoutineFailureClassification(t *testing.T) {
	routine := []error{jwt.ErrTokenExpired, jwt.ErrTokenMalformed, jwt.ErrTokenNotValidYet}
	for _, err := range routine {
		if !routineFailure(fmt.Errorf("wrapped: %w", err)) {
			t.Errorf("%v classified as an anomaly, want routine", err)
		}
	}

	anomalies := []error{
		jwt.ErrTokenSignatureInvalid,
		jwt.ErrTokenInvalidIssuer,
		jwt.ErrTokenUnverifiable,
		errNoKID,
		errNoSubject,
		errUnauthorizedParty,
	}
	for _, err := range anomalies {
		if routineFailure(fmt.Errorf("wrapped: %w", err)) {
			t.Errorf("%v classified as routine, want anomaly", err)
		}
	}
}

// An attacker must not be able to drive unbounded log volume.
func TestAnomalyLoggingIsRateLimited(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	h := newTestHandler(t, key, "kid_A")

	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil))) })

	srv := correlationID(h.authMiddleware(http.HandlerFunc(h.example)))
	attacker, _ := rsa.GenerateKey(rand.Reader, 2048)

	// 500 forged signatures: an anomaly class, so eligible for logging.
	for range 500 {
		req := httptest.NewRequest("GET", "/api/example", nil)
		req.Header.Set("Authorization", "Bearer "+mint(t, "kid_A", attacker, validClaims()))
		srv.ServeHTTP(httptest.NewRecorder(), req)
	}
	if lines := bytes.Count(buf.Bytes(), []byte("\n")); lines > 10 {
		t.Errorf("500 forged tokens produced %d log lines, want <= 10 (burst)", lines)
	}

	// Routine expiry must produce nothing at all.
	buf.Reset()
	expired := validClaims()
	expired.ExpiresAt = jwt.NewNumericDate(time.Now().Add(-time.Hour))
	for range 100 {
		req := httptest.NewRequest("GET", "/api/example", nil)
		req.Header.Set("Authorization", "Bearer "+mint(t, "kid_A", key, expired))
		srv.ServeHTTP(httptest.NewRecorder(), req)
	}
	if buf.Len() != 0 {
		t.Errorf("expired tokens logged %d bytes, want 0: %s", buf.Len(), buf.String())
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

	// Register future-shaped write routes to prove they inherit auth.
	api := http.NewServeMux()
	api.HandleFunc("GET /api/example", h.example)
	api.HandleFunc("POST /api/chat", h.example)
	api.HandleFunc("PUT /api/subscriptions", h.example)
	root := http.NewServeMux()
	root.HandleFunc("GET /health", h.health)
	root.Handle("/api/", h.authMiddleware(api))
	srv := correlationID(root)

	protected := []struct{ method, path string }{
		{"GET", "/api/example"},
		{"POST", "/api/chat"},
		{"PUT", "/api/subscriptions"},
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
	h, err := newHandler(jwks.Keyfunc, testIssuer, map[string]struct{}{testOrigin: {}})
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
	h, err := newHandler(jwks.Keyfunc, testIssuer, map[string]struct{}{testOrigin: {}})
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
