package main

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// maxWebhookBodyBytes bounds the one externally reachable, signature-gated
// endpoint: the body must be read in full before the signature can even be
// checked, so an unbounded read would let an unauthenticated caller tie up a
// goroutine on an arbitrarily large upload. Clerk's user payloads are a few
// KB at most.
const maxWebhookBodyBytes = 64 * 1024

// webhookTimestampTolerance bounds how old (or how far in the future, for
// clock skew) a signed request may be, per the Standard Webhooks spec Svix
// co-authored. Without it a captured, still-correctly-signed request could be
// replayed indefinitely.
const webhookTimestampTolerance = 5 * time.Minute

// maxSignatureTokens caps how many "v1,<sig>" entries in svix-signature get
// base64-decoded and HMAC-compared. A secret rotation sends at most two (the
// outgoing and incoming key) — generous headroom over that, not a tight fit.
// svix-signature is attacker-controlled and unverified at this point, so
// without a cap its token count is unbounded work per rejected request.
const maxSignatureTokens = 5

var (
	errMissingSvixHeaders  = errors.New("missing svix headers")
	errBadTimestamp        = errors.New("invalid svix-timestamp")
	errTimestampOutOfRange = errors.New("svix-timestamp outside tolerance")
	errNoMatchingSignature = errors.New("no matching signature")
)

// decodeWebhookSecret unpacks a Clerk/Svix signing secret — the dashboard
// shows it as "whsec_<base64>" — into the raw bytes HMAC needs. Called once
// at startup so a malformed secret is a boot failure, not a per-request one.
func decodeWebhookSecret(raw string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(strings.TrimPrefix(raw, "whsec_"))
}

// svixHeadersPresent reports whether all three Svix signature headers are
// set, without validating their contents. The one shared source for the
// header names: clerkWebhook checks this before reading the body (so it
// never buffers a request that could not possibly verify), and
// verifyClerkSignature checks it again on the parsed values it actually
// needs — two call sites, one list of header names to keep in sync.
func svixHeadersPresent(header http.Header) bool {
	return header.Get("svix-id") != "" && header.Get("svix-timestamp") != "" && header.Get("svix-signature") != ""
}

// verifyClerkSignature checks body against the svix-id/svix-timestamp/
// svix-signature headers Clerk (via Svix) sends on every webhook delivery.
// See https://docs.svix.com/receiving/verifying-payloads/how-manual.
//
// Hand-rolled rather than pulling in Svix's Go SDK, which is the full
// management-API client (application/endpoint/message-attempt CRUD, hundreds
// of files) for a library that would be used for nothing but this one check —
// not a fit for a gateway with four direct dependencies. The signed-content
// shape and comparison are exactly what the spec (and every vetted
// implementation of it) does: HMAC-SHA256 over "{id}.{timestamp}.{body}",
// checked against every "v1,<base64>" token in svix-signature (there can be
// more than one during a secret rotation), not just the first.
func verifyClerkSignature(secret []byte, header http.Header, body []byte) error {
	if !svixHeadersPresent(header) {
		return errMissingSvixHeaders
	}
	id := header.Get("svix-id")
	timestamp := header.Get("svix-timestamp")
	signatures := header.Get("svix-signature")

	ts, err := strconv.ParseInt(timestamp, 10, 64)
	if err != nil {
		return errBadTimestamp
	}
	// Both directions: too old is a replayed request, too far in the future
	// is a clock-skew anomaly — neither is a request to act on now.
	if age := time.Since(time.Unix(ts, 0)); age > webhookTimestampTolerance || age < -webhookTimestampTolerance {
		return errTimestampOutOfRange
	}

	mac := hmac.New(sha256.New, secret)
	mac.Write([]byte(id + "." + timestamp + "."))
	mac.Write(body)
	expected := mac.Sum(nil)

	for i, sig := range strings.Fields(signatures) {
		if i >= maxSignatureTokens {
			break
		}
		version, encoded, ok := strings.Cut(sig, ",")
		if !ok || version != "v1" {
			continue
		}
		got, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			continue
		}
		if hmac.Equal(got, expected) {
			return nil
		}
	}
	return errNoMatchingSignature
}

// clerkWebhookEvent is the envelope every Clerk webhook delivery shares;
// data is decoded per event type below.
type clerkWebhookEvent struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

// clerkEmailAddress is one entry in user.updated's email_addresses array.
type clerkEmailAddress struct {
	ID           string `json:"id"`
	EmailAddress string `json:"email_address"`
}

// clerkUserPayload is user.updated's data — a Clerk user can hold several
// email addresses, so the primary one has to be looked up by id rather than
// assumed to be the first entry.
type clerkUserPayload struct {
	ID                    string              `json:"id"`
	PrimaryEmailAddressID string              `json:"primary_email_address_id"`
	EmailAddresses        []clerkEmailAddress `json:"email_addresses"`
}

// primaryEmail resolves the address primary_email_address_id points at, or
// "" if the payload names one that isn't present in email_addresses — or
// names none at all, which the early return guards separately so an absent
// id never matches an equally-absent (empty-string) address id.
func (p clerkUserPayload) primaryEmail() string {
	if p.PrimaryEmailAddressID == "" {
		return ""
	}
	for _, addr := range p.EmailAddresses {
		if addr.ID == p.PrimaryEmailAddressID {
			return addr.EmailAddress
		}
	}
	return ""
}

// deleteUser removes id's row (TASKS.md T22.5). withUser is called with the
// target's own id, not a caller's — there is no authenticated caller here,
// only Clerk telling us which account is gone — which is exactly the id the
// users RLS policy requires to let the delete through. Every other table
// with a user_id foreign key cascades from here (ON DELETE CASCADE); FK
// referential-integrity actions bypass row security, so the cascade needs no
// policy of its own on those tables.
//
// Idempotent: Clerk retries webhook delivery, and a user already gone (a
// retry, or a row that was never provisioned) is a no-op, not an error.
func deleteUser(db *pgxpool.Pool) func(ctx context.Context, id string) error {
	return func(ctx context.Context, id string) error {
		return withUser(ctx, db, id, func(tx pgx.Tx) error {
			_, err := tx.Exec(ctx, `delete from users where id = $1`, id)
			return err
		})
	}
}

// updateUserEmail keeps the stored email current on Clerk's user.updated
// event (TASKS.md T22.5). Update only, not upsert: a user who has never hit
// the gateway has no row yet, and upsertUser provisions one with the current
// email the first time they do — the same self-healing reasoning T8 gives
// for running upsertUser per-request instead of off user.created.
func updateUserEmail(db *pgxpool.Pool) func(ctx context.Context, id, email string) error {
	return func(ctx context.Context, id, email string) error {
		return withUser(ctx, db, id, func(tx pgx.Tx) error {
			_, err := tx.Exec(ctx,
				`update users set email = $2 where id = $1 and email is distinct from $2`, id, email)
			return err
		})
	}
}

// clerkWebhook is POST /webhooks/clerk — the one endpoint that authenticates
// by signature instead of a Bearer session token, so it sits on root rather
// than the /api/ mux (see routes()).
func (h *Handler) clerkWebhook(w http.ResponseWriter, r *http.Request) {
	corrID := r.Context().Value(ctxCorrelationID)

	// Reject a request carrying none of Clerk's signature headers before
	// reading any of its body. This is the one gateway endpoint reachable
	// with no auth at all — a scanner or misdirected request with no svix
	// headers can never pass verifyClerkSignature below, so there's no
	// reason to pay for a buffered read first. verifyClerkSignature still
	// checks these itself (a genuine delivery always carries them, so this
	// is a no-op there); this is only about failing the header-less case
	// fast.
	if !svixHeadersPresent(r.Header) {
		slog.WarnContext(r.Context(), "webhook signature rejected",
			"reason", errMissingSvixHeaders.Error(), "correlation_id", corrID)
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid signature"})
		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxWebhookBodyBytes))
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "body too large"})
		} else {
			// A transport-level failure (client hangup, timeout) reading a
			// body under the cap — not the size limit.
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "malformed request"})
		}
		return
	}

	if err := verifyClerkSignature(h.webhookSecret, r.Header, body); err != nil {
		// Never the signature, the timestamp, or the body: the body is a
		// live Clerk payload (email addresses), and an invalid signature
		// leaves everything else attacker-controlled until it verifies.
		slog.WarnContext(r.Context(), "webhook signature rejected",
			"reason", err.Error(), "correlation_id", corrID)
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid signature"})
		return
	}

	var event clerkWebhookEvent
	if err := json.Unmarshal(body, &event); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "malformed payload"})
		return
	}

	switch event.Type {
	case "user.deleted":
		var data struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(event.Data, &data); err != nil || data.ID == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "malformed user.deleted payload"})
			return
		}
		// Bounded for the same reason authMiddleware bounds ensureUser: r.Context()
		// has no deadline of its own, only client disconnect, so a stalled DB
		// would otherwise hold this goroutine open indefinitely.
		deleteCtx, cancel := context.WithTimeout(r.Context(), provisionTimeout)
		err := h.deleteUser(deleteCtx, data.ID)
		cancel()
		if err != nil {
			slog.ErrorContext(r.Context(), "webhook user delete failed",
				"correlation_id", corrID, "error", dbError(err))
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
			return
		}

	case "user.updated":
		var data clerkUserPayload
		if err := json.Unmarshal(event.Data, &data); err != nil || data.ID == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "malformed user.updated payload"})
			return
		}
		email := data.primaryEmail()
		if email == "" {
			// A signed-up-with-email account always has one; a payload
			// without a resolvable primary address is unexpected but not
			// this webhook's problem to error out over — log it and move
			// on, the same as an unrecognized event type below.
			slog.WarnContext(r.Context(), "user.updated with no resolvable primary email",
				"correlation_id", corrID)
			break
		}
		updateCtx, cancel := context.WithTimeout(r.Context(), provisionTimeout)
		err := h.updateUserEmail(updateCtx, data.ID, email)
		cancel()
		if err != nil {
			slog.ErrorContext(r.Context(), "webhook user email update failed",
				"correlation_id", corrID, "error", dbError(err))
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
			return
		}
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}
