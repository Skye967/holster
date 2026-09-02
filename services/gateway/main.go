package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"log"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/MicahParks/keyfunc/v3"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
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

// Ceiling on the per-request provisioning upsert — a single-row write on a
// primary key, so this is generous even to Supabase. A var, not a const, so a
// test can shorten it without waiting the full two seconds.
var provisionTimeout = 2 * time.Second

type Claims struct {
	// Origin the token was issued to. Clerk's defence against a token minted
	// for one frontend being replayed against another.
	Azp string `json:"azp"`
	// Added by the gateway JWT template. Required: the Clerk instance is
	// configured for email sign-up, so a token without one is malformed rather
	// than a legitimate account. Not a schema constraint — an absent claim
	// unmarshals to "", which users.email NOT NULL would accept.
	Email string `json:"email"`
	jwt.RegisteredClaims
}

var (
	errNoKID             = errors.New("token has no kid header")
	errNoSubject         = errors.New("token has no subject")
	errNoEmail           = errors.New("token has no email")
	errUnauthorizedParty = errors.New("unauthorized party")
)

// authFailure is why a token was rejected. Fixed set: it is a log field and an
// eventual metric label, so it must never carry token-derived data. The bool
// separates ordinary token lifecycle from anomalies worth a warning.
//
// Clerk performs authentication; this service only verifies tokens, so an expired
// one is a 60s timer elapsing, not a failed credential attempt.
//
// Anomalies come first because jwt's validator joins every claim failure into one
// error: errors.Is matches all of them, so this order alone decides the label.
// Expiry first would file a replayed cross-audience token as a routine expiry.
func authFailure(err error) (reason string, anomaly bool) {
	switch {
	case errors.Is(err, errNoKID):
		return "no_kid", true
	case errors.Is(err, errNoSubject):
		return "no_subject", true
	case errors.Is(err, errNoEmail):
		return "no_email", true
	case errors.Is(err, errUnauthorizedParty):
		return "unauthorized_party", true
	case errors.Is(err, jwt.ErrTokenSignatureInvalid):
		return "bad_signature", true
	case errors.Is(err, jwt.ErrTokenInvalidIssuer):
		return "bad_issuer", true
	case errors.Is(err, jwt.ErrTokenInvalidAudience):
		return "bad_audience", true
	case errors.Is(err, jwt.ErrTokenRequiredClaimMissing):
		return "missing_claim", true
	case errors.Is(err, jwt.ErrTokenUnverifiable):
		return "unverifiable", true
	case errors.Is(err, jwt.ErrTokenExpired):
		return "expired", false
	case errors.Is(err, jwt.ErrTokenMalformed):
		return "malformed", false
	case errors.Is(err, jwt.ErrTokenNotValidYet):
		return "not_yet_valid", false
	default:
		// Deliberately not "unverifiable": that means no key could be obtained,
		// and mislabelling an unrelated failure sends the reader to JWKS.
		return "other", true
	}
}

type Handler struct {
	keyfunc  jwt.Keyfunc
	issuer   string
	audience string
	parties  map[string]struct{}
	// Injected so the auth path is testable without a database. Production
	// wiring is upsertUser; the SQL itself is covered by its own test.
	ensureUser func(ctx context.Context, id, email string) error

	// The chat socket — see chat.go. loadChatCtx and callAgent follow
	// ensureUser's own injection shape so both are fakeable in tests without a
	// real database or agent process. rootCtx is the process's own
	// SIGINT/SIGTERM-cancelled context (not any single request's), used to tie
	// every open WebSocket's lifetime to server shutdown.
	loadChatCtx    func(ctx context.Context, userID string) (chatContext, error)
	callAgent      agentCaller
	tickets        *chatTicketStore
	originPatterns []string
	rootCtx        context.Context

	// providers.go (T15): the region catalog cache and the per-user
	// subscription write, injected the same way as ensureUser/loadChatCtx
	// above so both are fakeable in tests without a real database or agent.
	loadProviders    func(ctx context.Context, country string) ([]Provider, error)
	saveSubscription func(ctx context.Context, userID string, providerID int, subscribed bool) error
}

// newHandler guards the keyfunc and rejects an empty origin allowlist, which
// would 401 every request — a startup error rather than a runtime surprise.
func newHandler(kf jwt.Keyfunc, issuer, audience string, parties map[string]struct{},
	ensureUser func(ctx context.Context, id, email string) error,
	loadChatCtx func(ctx context.Context, userID string) (chatContext, error),
	callAgent agentCaller,
	rootCtx context.Context,
	loadProviders func(ctx context.Context, country string) ([]Provider, error),
	saveSubscription func(ctx context.Context, userID string, providerID int, subscribed bool) error,
) (*Handler, error) {

	if len(parties) == 0 {
		return nil, errors.New("no authorized parties configured")
	}
	if audience == "" {
		return nil, errors.New("no audience configured")
	}
	if ensureUser == nil {
		return nil, errors.New("no user provisioner configured")
	}
	if loadChatCtx == nil {
		return nil, errors.New("no chat context loader configured")
	}
	if callAgent == nil {
		return nil, errors.New("no agent caller configured")
	}
	if rootCtx == nil {
		return nil, errors.New("no root context configured")
	}
	if loadProviders == nil {
		return nil, errors.New("no provider loader configured")
	}
	if saveSubscription == nil {
		return nil, errors.New("no subscription writer configured")
	}

	origins := make([]string, 0, len(parties))
	for p := range parties {
		origins = append(origins, p)
	}

	return &Handler{
		keyfunc:          requireKID(kf),
		issuer:           issuer,
		audience:         audience,
		parties:          parties,
		ensureUser:       ensureUser,
		loadChatCtx:      loadChatCtx,
		callAgent:        callAgent,
		tickets:          newChatTicketStore(),
		originPatterns:   origins,
		rootCtx:          rootCtx,
		loadProviders:    loadProviders,
		saveSubscription: saveSubscription,
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
		jwt.WithAudience(h.audience),
		jwt.WithExpirationRequired(),
		jwt.WithLeeway(clockLeeway),
	)
	if err != nil {
		return nil, err
	}

	if claims.Subject == "" {
		return nil, errNoSubject
	}

	// Authorization before payload adequacy: a token minted for an origin we do
	// not allow must report unauthorized_party, not a missing-claim reason it
	// would also fail on. Backend-API tokens carry neither azp nor the template's
	// email, and they are the case this ordering exists for.
	//
	// Clerk's docs allow skipping this when azp is absent. We don't: tokens
	// minted through the Backend API carry no azp, and those are not sessions.
	if !h.originAllowed(claims.Azp) {
		return nil, errUnauthorizedParty
	}

	if claims.Email == "" {
		return nil, errNoEmail
	}

	return claims, nil
}

// withUser runs fn inside a transaction with holster.user_id bound to the
// caller's Clerk ID, so the RLS policies from 20260831233121_rls_roles.sql scope
// every statement fn issues. This is the shape every user-scoped database
// operation follows — see ../../DECISIONS.md and TASKS.md T10.
//
// The transaction is not optional. The gateway connects as gateway_app, a
// non-owner role subject to row-level security. set_config with is_local => true
// scopes the binding to this transaction, so it cannot leak to the next caller
// on a pooled connection; a session-level SET would. Omitting the binding is not
// an exposure but a self-inflicted outage: current_setting then returns NULL and
// every policy predicate denies.
func withUser(ctx context.Context, db *pgxpool.Pool, userID string, fn func(pgx.Tx) error) error {
	return pgx.BeginFunc(ctx, db, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx,
			`select set_config('holster.user_id', $1, true)`, userID); err != nil {
			return err
		}
		return fn(tx)
	})
}

// upsertUser mirrors the Clerk identity into users. Runs on every request rather
// than on a user.created webhook because it repairs itself; see ../../DECISIONS.md.
//
// The WHERE guard is what makes that affordable: without it ON CONFLICT DO UPDATE
// rewrites the row every time. TestUpsertUser holds it to that.
func upsertUser(db *pgxpool.Pool) func(context.Context, string, string) error {
	return func(ctx context.Context, id, email string) error {
		return withUser(ctx, db, id, func(tx pgx.Tx) error {
			_, err := tx.Exec(ctx, `
				insert into users (id, email) values ($1, $2)
				on conflict (id) do update set email = excluded.email
				where users.email is distinct from excluded.email`, id, email)
			return err
		})
	}
}

// dbError renders a database error for logging. A server-side error becomes its
// SQLSTATE alone: PostgreSQL puts row values in a constraint violation's DETAIL,
// so the message would carry the user's own data. Everything else — connection,
// timeout, cancellation — has no row data in it and is logged as-is.
func dbError(err error) string {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		return pgErr.Code
	}
	return err.Error()
}

// originAllowed reports whether origin is one of the app's configured origins
// (CLERK_AUTHORIZED_PARTIES). Shared by verifyToken's azp check and
// corsMiddleware below — two different trust decisions (who may mint a
// token, vs. who may read a response in a browser) that happen to share one
// value today. Split them into separate allowlists only once something
// actually needs them to diverge; today it would be two copies of the same
// set.
func (h *Handler) originAllowed(origin string) bool {
	_, ok := h.parties[origin]
	return ok
}

// corsMiddleware answers a browser's CORS preflight and stamps the same
// headers on the real response, scoped to originAllowed — the same origin
// allowlist azp validation and the WebSocket's own OriginPatterns already
// trust, so there is one allowlist, not two.
//
// allowMethods is derived once in routes() from the routes actually
// registered on the api mux, rather than hardcoded here — a literal list
// would silently drift the first time a route added a verb without anyone
// remembering a second place to update.
//
// Wraps only the /api/ mux: /health is never called from a browser, and
// /ws/chat's own handshake isn't subject to fetch's CORS rules. Nothing under
// /api/ needed this until T15 — chat's WebSocket and its ticket mint are both
// same-origin-safe in a way a browser fetch() with an Authorization header is
// not, so this had no reason to exist before the picker's plain GET/PUT/DELETE
// calls.
//
// Runs outside authMiddleware: a preflight OPTIONS request never carries the
// real Authorization header, so answering it inside auth would 401 every
// preflight and the browser would never get far enough to see the headers set
// here.
func (h *Handler) corsMiddleware(allowMethods string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if h.originAllowed(origin) {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin")
			w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
			w.Header().Set("Access-Control-Allow-Methods", allowMethods)
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
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
			// Only the reason code is logged. Never the token, which is a live
			// bearer credential; never a claimed sub, which is attacker
			// controlled until the signature verifies; and never the raw error,
			// which can embed an attacker-chosen key id. keyfunc logs the
			// underlying JWKS detail itself. Volume is the collector's job.
			reason, anomaly := authFailure(err)
			// Routine lifecycle is DEBUG so the default level is quiet: an
			// unconfigured deploy must not emit a line per expired token.
			level := slog.LevelDebug
			if anomaly {
				level = slog.LevelWarn
			}
			slog.Log(r.Context(), level, "auth rejected",
				"reason", reason,
				"correlation_id", r.Context().Value(ctxCorrelationID))
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid token"})
			return
		}

		// Fail closed: without the row, anything with a user foreign key would
		// fail later and less clearly.
		//
		// The deadline matters because r.Context() has none — no http.Server
		// timeout cancels it, only client disconnect — so a database that
		// blackholes packets would otherwise hold this goroutine until TCP gives
		// up, minutes later, with no 503 ever written.
		provisionCtx, cancel := context.WithTimeout(r.Context(), provisionTimeout)
		err = h.ensureUser(provisionCtx, claims.Subject, claims.Email)
		cancel()
		if err != nil {
			// A caller that hung up cancels provisionCtx, and pgx surfaces that
			// as context.Canceled — not a provisioning failure, and nothing to
			// write to either since the connection is already gone. Our own 2s
			// deadline is context.DeadlineExceeded and a real database error is
			// neither, so both of those still log and return 503.
			if errors.Is(err, context.Canceled) {
				return
			}
			slog.ErrorContext(r.Context(), "user provisioning failed",
				"correlation_id", r.Context().Value(ctxCorrelationID),
				"error", dbError(err))
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
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
//
// /ws/chat is the one deliberate exception: it authenticates via a ticket
// (chat.go), not a Bearer header, so it cannot sit on the authMiddleware-wrapped
// api mux — putting it under /api/ would silently break the "everything under
// /api/ requires the Bearer path" invariant this comment states. Its own ticket
// mint endpoint, POST /api/chat/ticket, is an ordinary Bearer-authenticated
// route and does live on api.
func (h *Handler) routes() http.Handler {
	api := http.NewServeMux()

	// register collects the method set as a side effect of registering each
	// route, so corsMiddleware's Access-Control-Allow-Methods is derived from
	// what's actually on this mux instead of a separate literal that could
	// drift from it.
	methods := map[string]struct{}{}
	register := func(pattern string, handler http.HandlerFunc) {
		method, _, _ := strings.Cut(pattern, " ")
		methods[method] = struct{}{}
		api.HandleFunc(pattern, handler)
	}
	register("GET /api/example", h.example)
	register("POST /api/chat/ticket", h.chatTicket)
	register("GET /api/providers", h.providers)
	register("GET /api/subscriptions", h.subscriptions)
	register("PUT /api/subscriptions/{providerID}", h.setSubscription)
	register("DELETE /api/subscriptions/{providerID}", h.setSubscription)

	allowMethods := make([]string, 0, len(methods))
	for m := range methods {
		allowMethods = append(allowMethods, m)
	}
	sort.Strings(allowMethods) // deterministic header value

	root := http.NewServeMux()
	root.HandleFunc("GET /health", h.health)
	root.HandleFunc("GET /ws/chat", h.chatWS)
	root.Handle("/api/", h.corsMiddleware(strings.Join(allowMethods, ", "), h.authMiddleware(api)))

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

// setupLogging sends structured JSON to stdout and nothing else: routing and
// retention belong to the execution environment, not here. LOG_LEVEL is the only
// verbosity control. LevelVar rather than a plain Level so it can be changed at
// runtime later without restructuring.
func setupLogging() {
	var level slog.LevelVar
	if err := level.UnmarshalText([]byte(cmp.Or(os.Getenv("LOG_LEVEL"), "info"))); err != nil {
		log.Fatalf("LOG_LEVEL: %v", err)
	}
	// keyfunc logs its JWKS refresh failures to slog.Default(), so this puts
	// those in the same stream, at the same level.
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: &level})))
}

func main() {
	setupLogging()

	jwksURL := mustEnv("CLERK_JWKS_URL")
	issuer := mustEnv("CLERK_ISSUER")
	audience := mustEnv("CLERK_AUDIENCE")
	parties := authorizedParties(mustEnv("CLERK_AUTHORIZED_PARTIES"))
	databaseURL := mustEnv("DATABASE_URL")
	agentURL := mustEnv("AGENT_SERVICE_URL")

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

	// Set here rather than in DATABASE_URL so they survive pointing the gateway at
	// Supabase.
	poolConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		log.Fatalf("DATABASE_URL: %v", err)
	}
	poolConfig.ConnConfig.RuntimeParams["application_name"] = "holster-gateway"
	// Backstop, not the request budget. pgx's default cancellation handler only
	// puts a deadline on the client socket, so a query provisionTimeout gave up on
	// keeps running on the server. Set above provisionTimeout so the request
	// deadline is what normally fires and this catches only what outlives it.
	poolConfig.ConnConfig.RuntimeParams["statement_timeout"] = "3000"
	// pgx sizes the pool from runtime.NumCPU() — host cores, not the cgroup
	// limit — so the default is 4 on a small VM but 16+ on a large node, and
	// neither is the right basis for a gateway that makes one short DB round
	// trip per request. Pick a small explicit ceiling instead. If this ever runs
	// more than one replica, replicas × MaxConns must stay under the Supabase
	// tier's pooler limit.
	poolConfig.MaxConns = 10

	db, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		log.Fatalf("database: %v", err)
	}
	defer db.Close()

	// Reachability and schema in one round trip, at startup rather than on the
	// first request: a bad DATABASE_URL should stop a deploy, not surface as a 503
	// to a user. A ping alone would not catch the likelier failure — compose runs
	// migrations only when the data directory is empty, so a volume from before
	// they existed answers fine and has no tables.
	startupCtx, cancelStartup := context.WithTimeout(ctx, 10*time.Second)
	defer cancelStartup()
	var schemaReady bool
	if err := db.QueryRow(startupCtx,
		`select to_regclass('public.users') is not null`).Scan(&schemaReady); err != nil {
		log.Fatalf("database unreachable: %v", err)
	}
	if !schemaReady {
		log.Fatal("database has no users table; run 'docker compose down -v' to re-seed the " +
			"local stack, or apply migrations to your configured database (see db/README.md)")
	}

	// No client-level Timeout: a chat turn's own bound is turnDeadline (chat.go),
	// applied via the request context, not a fixed transport timeout that would
	// also cap how long the streamed NDJSON body may legitimately stay open.
	agentClient := &http.Client{}

	h, err := newHandler(jwks.Keyfunc, issuer, audience, parties, upsertUser(db),
		loadChatContext(db), newAgentCaller(agentClient, agentURL), ctx,
		loadProviders(db, newAgentProviderCaller(agentClient, agentURL)), saveSubscription(db))
	if err != nil {
		log.Fatalf("handler: %v", err)
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
