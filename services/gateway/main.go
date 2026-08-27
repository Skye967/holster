package main

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"

	"github.com/MicahParks/keyfunc/v3"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"golang.org/x/time/rate"
)

// Private key type so context values cannot collide with another package's.
type ctxKey int

const (
	ctxUserID ctxKey = iota
	ctxCorrelationID
)

// Clock skew allowed on exp and nbf. Clerk session tokens live 60s, so this
// stays well under a full lifetime.
const clockLeeway = 5 * time.Second

type Claims struct {
	// Origin the token was issued to. Clerk's defence against a token minted
	// for one frontend being replayed against another.
	Azp string `json:"azp"`
	jwt.RegisteredClaims
}

var (
	errNoKID             = errors.New("token has no kid header")
	errNoSubject         = errors.New("token has no subject")
	errUnauthorizedParty = errors.New("unauthorized party")
)

// failureClass maps a rejection onto a closed set of labels. It must stay closed:
// logSampler keys on it, so a label derived from error text would let anyone who
// can influence an error message grow the sampler's map without bound.
func failureClass(err error) string {
	switch {
	case errors.Is(err, errNoKID):
		return "no_kid"
	case errors.Is(err, errNoSubject):
		return "no_subject"
	case errors.Is(err, errUnauthorizedParty):
		return "unauthorized_party"
	case errors.Is(err, jwt.ErrTokenSignatureInvalid):
		return "bad_signature"
	case errors.Is(err, jwt.ErrTokenInvalidIssuer):
		return "bad_issuer"
	case errors.Is(err, jwt.ErrTokenRequiredClaimMissing):
		return "missing_claim"
	case errors.Is(err, jwt.ErrTokenUnverifiable):
		return "unverifiable"
	default:
		return "other"
	}
}

// maxLoggedError caps the error text on a log line. Error strings can embed
// attacker-controlled data — an unknown kid reaches keyfunc's `key not found "…"`
// error, and a token header may carry up to MaxHeaderBytes of it. failureClass
// bounds the log's key; this bounds its value.
const maxLoggedError = 200

func truncateError(err error) string {
	s := err.Error()
	if len(s) <= maxLoggedError {
		return s
	}
	// Cut on a rune boundary so an attacker-supplied multi-byte sequence cannot
	// be split into invalid UTF-8.
	cut := maxLoggedError
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…(truncated)"
}

// logSampler bounds how often each class may be logged and counts what it drops
// in between. Keying per class is what stops a flood of one failure from hiding
// another; the drop count is what stops suppression from being silent.
//
// Modelled on zap's sampler, which keys the same way, but with a hard bucket
// rather than zap's "every Mth thereafter" — output must stay bounded under a
// flood, not merely proportional.
type logSampler struct {
	mu       sync.Mutex
	every    time.Duration
	burst    int
	limiters map[string]*rate.Limiter
	dropped  map[string]uint64
}

func newLogSampler(every time.Duration, burst int) *logSampler {
	return &logSampler{
		every:    every,
		burst:    burst,
		limiters: map[string]*rate.Limiter{},
		dropped:  map[string]uint64{},
	}
}

// allow reports whether to log this class now, and how many of it were dropped
// since the last one that was logged.
func (s *logSampler) allow(class string) (bool, uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	l, ok := s.limiters[class]
	if !ok {
		l = rate.NewLimiter(rate.Every(s.every), s.burst)
		s.limiters[class] = l
	}
	if !l.Allow() {
		s.dropped[class]++
		return false, 0
	}
	n := s.dropped[class]
	delete(s.dropped, class)
	return true, n
}

type Handler struct {
	keyfunc jwt.Keyfunc
	issuer  string
	parties map[string]struct{}
	sampler *logSampler
}

// newHandler owns the Handler's invariants: the keyfunc is guarded, the sampler
// is set, and the origin allowlist is non-empty. An empty allowlist would reject
// every request, so it is a startup error rather than a runtime surprise.
func newHandler(kf jwt.Keyfunc, issuer string, parties map[string]struct{}) (*Handler, error) {
	if len(parties) == 0 {
		return nil, errors.New("no authorized parties configured")
	}
	return &Handler{
		keyfunc: requireKID(kf),
		issuer:  issuer,
		parties: parties,
		// 1/s per class, burst 5: enough to watch a fault develop, bounded enough
		// that a flood cannot fill the disk.
		sampler: newLogSampler(time.Second, 5),
	}, nil
}

// requireKID rejects a token that names no signing key. Do not remove this as
// redundant: keyfunc has no option to disable it, and when kid is absent it
// falls back to trying every key in the set, skipping the alg cross-check it
// performs on the kid path.
func requireKID(next jwt.Keyfunc) jwt.Keyfunc {
	return func(token *jwt.Token) (any, error) {
		if kid, ok := token.Header["kid"].(string); !ok || kid == "" {
			return nil, errNoKID
		}
		return next(token)
	}
}

// routineFailure reports whether a rejection is ordinary token lifecycle rather
// than a security event. Clerk tokens live 60s, so every idle tab produces these
// (TASKS.md: "401 is routine, not an error") and logging them buries the
// anomalies. Clerk performs authentication; this service only verifies tokens,
// so an expired one is a timer elapsing, not a failed credential attempt.
func routineFailure(err error) bool {
	return errors.Is(err, jwt.ErrTokenExpired) ||
		errors.Is(err, jwt.ErrTokenMalformed) ||
		errors.Is(err, jwt.ErrTokenNotValidYet)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (h *Handler) verifyToken(tokenString string) (*Claims, error) {
	claims := &Claims{}
	_, err := jwt.ParseWithClaims(tokenString, claims, h.keyfunc,
		jwt.WithValidMethods([]string{"RS256"}),
		jwt.WithIssuer(h.issuer),
		jwt.WithExpirationRequired(),
		jwt.WithLeeway(clockLeeway),
	)
	if err != nil {
		return nil, err
	}

	if claims.Subject == "" {
		return nil, errNoSubject
	}

	// Clerk's docs allow skipping this when azp is absent. We don't: tokens
	// minted through the Backend API carry no azp, and those are not sessions.
	if _, ok := h.parties[claims.Azp]; !ok {
		return nil, errUnauthorizedParty
	}

	return claims, nil
}

func correlationID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := uuid.New().String()
		w.Header().Set("X-Correlation-ID", id)
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), ctxCorrelationID, id)))
	})
}

func (h *Handler) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		parts := strings.SplitN(r.Header.Get("Authorization"), " ", 2)
		if len(parts) != 2 || parts[0] != "Bearer" {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid authorization header"})
			return
		}

		claims, err := h.verifyToken(parts[1])
		if err != nil {
			// Anomalies are logged so a JWKS or config fault is distinguishable
			// from users with stale tokens. Never log the token: it is a live
			// bearer credential. Never log a claimed sub: it is attacker
			// controlled until the signature verifies.
			if class := failureClass(err); !routineFailure(err) {
				if ok, dropped := h.sampler.allow(class); ok {
					slog.Warn("auth rejected",
						"class", class,
						"correlation_id", r.Context().Value(ctxCorrelationID),
						"error", truncateError(err),
						"dropped_since_last", dropped)
				}
			}
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid token"})
			return
		}

		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), ctxUserID, claims.Subject)))
	})
}

func (h *Handler) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (h *Handler) example(w http.ResponseWriter, r *http.Request) {
	userID, _ := r.Context().Value(ctxUserID).(string)
	corrID, _ := r.Context().Value(ctxCorrelationID).(string)
	writeJSON(w, http.StatusOK, map[string]string{
		"user_id":        userID,
		"correlation_id": corrID,
	})
}

// routes wraps auth around the whole /api/ prefix rather than around individual
// routes, so a handler registered on the api mux cannot be reachable without it.
// Public routes go on root.
func (h *Handler) routes() http.Handler {
	api := http.NewServeMux()
	api.HandleFunc("GET /api/example", h.example)

	root := http.NewServeMux()
	root.HandleFunc("GET /health", h.health)
	root.Handle("/api/", h.authMiddleware(api))

	return correlationID(root)
}

func mustEnv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("%s not set", key)
	}
	return v
}

// authorizedParties parses the comma-separated origin allowlist. Empty entries
// are dropped so a stray comma cannot admit tokens that carry no azp.
func authorizedParties(env string) map[string]struct{} {
	parties := map[string]struct{}{}
	for _, p := range strings.Split(env, ",") {
		if p = strings.TrimSpace(p); p != "" {
			parties[p] = struct{}{}
		}
	}
	return parties
}

func main() {
	// JSON to stdout. keyfunc logs its JWKS refresh failures to slog.Default(),
	// so setting this puts those in the same stream as our own lines.
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))

	jwksURL := mustEnv("CLERK_JWKS_URL")
	issuer := mustEnv("CLERK_ISSUER")
	parties := authorizedParties(mustEnv("CLERK_AUTHORIZED_PARTIES"))

	port := os.Getenv("GATEWAY_PORT")
	if port == "" {
		port = "8080"
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// keyfunc owns the JWKS: it refreshes on an unknown kid, rate-limits those
	// refreshes, and bounds each fetch. An unreachable JWKS is not fatal here —
	// it retries in the background, and auth fails closed until it succeeds.
	//
	// Its defaults were reviewed and taken as-is: hourly background refresh, at
	// most one unknown-kid refresh per 5 minutes. Rotation recovers through
	// either path, and JWKS providers publish old and new keys through an
	// overlap far longer than an hour.
	jwks, err := keyfunc.NewDefaultCtx(ctx, []string{jwksURL})
	if err != nil {
		log.Fatalf("jwks: %v", err)
	}

	h, err := newHandler(jwks.Keyfunc, issuer, parties)
	if err != nil {
		log.Fatalf("CLERK_AUTHORIZED_PARTIES: %v", err)
	}

	server := &http.Server{
		Addr:              ":" + port,
		Handler:           h.routes(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	go func() {
		log.Printf("gateway listening on :%s", port)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("listen: %v", err)
		}
	}()

	<-ctx.Done()
	log.Print("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
	}
}
