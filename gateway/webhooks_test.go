package main

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"
)

// signWebhook computes the svix-id/svix-timestamp/svix-signature values a
// real Clerk delivery would carry for body, signed with secret.
func signWebhook(secret []byte, id string, ts time.Time, body []byte) (svixID, svixTimestamp, svixSignature string) {
	svixID = id
	svixTimestamp = strconv.FormatInt(ts.Unix(), 10)
	mac := hmac.New(sha256.New, secret)
	mac.Write([]byte(svixID + "." + svixTimestamp + "."))
	mac.Write(body)
	svixSignature = "v1," + base64.StdEncoding.EncodeToString(mac.Sum(nil))
	return
}

// --- verifyClerkSignature: pure unit tests, no HTTP -------------------------

func TestVerifyClerkSignatureAcceptsValidSignature(t *testing.T) {
	body := []byte(`{"type":"user.deleted","data":{"id":"user_1"}}`)
	id, ts, sig := signWebhook(testWebhookSecretBytes, "msg_1", time.Now(), body)
	header := http.Header{}
	header.Set("svix-id", id)
	header.Set("svix-timestamp", ts)
	header.Set("svix-signature", sig)

	if err := verifyClerkSignature(testWebhookSecretBytes, header, body); err != nil {
		t.Errorf("valid signature rejected: %v", err)
	}
}

func TestVerifyClerkSignatureRejectsWrongSecret(t *testing.T) {
	body := []byte(`{"type":"user.deleted","data":{"id":"user_1"}}`)
	id, ts, sig := signWebhook(testWebhookSecretBytes, "msg_1", time.Now(), body)
	header := http.Header{}
	header.Set("svix-id", id)
	header.Set("svix-timestamp", ts)
	header.Set("svix-signature", sig)

	other, err := decodeWebhookSecret("whsec_" + base64.StdEncoding.EncodeToString([]byte("a-different-signing-secret-value")))
	if err != nil {
		t.Fatal(err)
	}
	if err := verifyClerkSignature(other, header, body); !errors.Is(err, errNoMatchingSignature) {
		t.Errorf("err = %v, want errNoMatchingSignature", err)
	}
}

func TestVerifyClerkSignatureRejectsTamperedBody(t *testing.T) {
	body := []byte(`{"type":"user.deleted","data":{"id":"user_1"}}`)
	id, ts, sig := signWebhook(testWebhookSecretBytes, "msg_1", time.Now(), body)
	header := http.Header{}
	header.Set("svix-id", id)
	header.Set("svix-timestamp", ts)
	header.Set("svix-signature", sig)

	tampered := []byte(`{"type":"user.deleted","data":{"id":"user_2"}}`)
	if err := verifyClerkSignature(testWebhookSecretBytes, header, tampered); !errors.Is(err, errNoMatchingSignature) {
		t.Errorf("err = %v, want errNoMatchingSignature", err)
	}
}

func TestVerifyClerkSignatureRejectsStaleTimestamp(t *testing.T) {
	body := []byte(`{}`)
	id, ts, sig := signWebhook(testWebhookSecretBytes, "msg_1", time.Now().Add(-10*time.Minute), body)
	header := http.Header{}
	header.Set("svix-id", id)
	header.Set("svix-timestamp", ts)
	header.Set("svix-signature", sig)

	if err := verifyClerkSignature(testWebhookSecretBytes, header, body); !errors.Is(err, errTimestampOutOfRange) {
		t.Errorf("err = %v, want errTimestampOutOfRange", err)
	}
}

func TestVerifyClerkSignatureRejectsFutureSkewedTimestamp(t *testing.T) {
	body := []byte(`{}`)
	id, ts, sig := signWebhook(testWebhookSecretBytes, "msg_1", time.Now().Add(10*time.Minute), body)
	header := http.Header{}
	header.Set("svix-id", id)
	header.Set("svix-timestamp", ts)
	header.Set("svix-signature", sig)

	if err := verifyClerkSignature(testWebhookSecretBytes, header, body); !errors.Is(err, errTimestampOutOfRange) {
		t.Errorf("err = %v, want errTimestampOutOfRange", err)
	}
}

func TestVerifyClerkSignatureRejectsMissingHeaders(t *testing.T) {
	if err := verifyClerkSignature(testWebhookSecretBytes, http.Header{}, []byte(`{}`)); !errors.Is(err, errMissingSvixHeaders) {
		t.Errorf("err = %v, want errMissingSvixHeaders", err)
	}
}

// A secret rotation briefly leaves two "v1,<sig>" tokens in svix-signature —
// the current key's and the previous one's. Any matching token must verify,
// not just the first.
func TestVerifyClerkSignatureAcceptsAnyMatchingToken(t *testing.T) {
	body := []byte(`{}`)
	id, ts, sig := signWebhook(testWebhookSecretBytes, "msg_1", time.Now(), body)
	header := http.Header{}
	header.Set("svix-id", id)
	header.Set("svix-timestamp", ts)
	header.Set("svix-signature", "v1,bm90LWEtcmVhbC1zaWduYXR1cmU= "+sig)

	if err := verifyClerkSignature(testWebhookSecretBytes, header, body); err != nil {
		t.Errorf("valid token among several rejected: %v", err)
	}
}

// An absent primary_email_address_id must resolve to "", never to an
// email_addresses entry whose own id also happens to be empty — "" is not
// a real Clerk id, but it is a valid zero value for the field, and matching
// it against email_addresses (rather than treating an absent primary id as
// unresolvable up front) would pick an arbitrary address.
func TestPrimaryEmailWithEmptyPrimaryIDDoesNotMatchEmptyAddressID(t *testing.T) {
	payload := clerkUserPayload{
		ID:                    "user_1",
		PrimaryEmailAddressID: "",
		EmailAddresses:        []clerkEmailAddress{{ID: "", EmailAddress: "unexpected@example.com"}},
	}
	if got := payload.primaryEmail(); got != "" {
		t.Errorf("primaryEmail() = %q, want \"\" for an unset primary_email_address_id", got)
	}
}

// --- HTTP-level handler tests ------------------------------------------------

// newWebhookTestServer wires a full Handler (real routing/signature check)
// with fake deleteUser/updateUserEmail. Unlike newVerdictsTestServer and its
// siblings, this needs no real JWKS/keyfunc: /webhooks/clerk sits on root,
// never behind authMiddleware (see routes()), so newHandler's keyfunc
// argument — unlike every other constructor argument — goes unused on
// every path these tests exercise. newHandler never nil-checks it, so nil
// is safe here.
func newWebhookTestServer(t *testing.T,
	deleteUser func(context.Context, string) error,
	updateUserEmail func(context.Context, string, string) error,
) *httptest.Server {
	t.Helper()

	h, err := newHandler(nil, testIssuer, testAudience, map[string]struct{}{testOrigin: {}},
		func(context.Context, string, string) error { return nil },
		noopChatCtx, noopLoadGuestChatCtx, noopAgentCaller, t.Context(),
		noopLoadProviders, noopSaveSubscription,
		noopLoadVerdicts, noopSaveVerdict,
		noopLoadWatchlistItems, noopCallAgentTitles,
		noopLoadConversation, noopLoadConversationTurns, noopSaveMessages,
		noopLoadConversationSummaries, noopDeleteConversation,
		testWebhookSecretBytes, deleteUser, updateUserEmail,
	)
	if err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(h.routes())
	t.Cleanup(srv.Close)
	return srv
}

// postWebhook sends body to /webhooks/clerk, signed with secret — a wrong
// secret is how TestClerkWebhookRejectsBadSignature exercises the 401 path.
func postWebhook(t *testing.T, srv *httptest.Server, secret []byte, body []byte) *http.Response {
	t.Helper()
	id, ts, sig := signWebhook(secret, "msg_"+t.Name(), time.Now(), body)
	req, err := http.NewRequest(http.MethodPost, srv.URL+"/webhooks/clerk", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("svix-id", id)
	req.Header.Set("svix-timestamp", ts)
	req.Header.Set("svix-signature", sig)
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func TestClerkWebhookUserDeletedCallsDeleteUser(t *testing.T) {
	var gotID string
	deleteUser := func(_ context.Context, id string) error {
		gotID = id
		return nil
	}
	srv := newWebhookTestServer(t, deleteUser, noopUpdateUserEmail)

	body := []byte(`{"type":"user.deleted","data":{"id":"user_123","object":"user","deleted":true}}`)
	resp := postWebhook(t, srv, testWebhookSecretBytes, body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if gotID != "user_123" {
		t.Errorf("deleteUser called with %q, want user_123", gotID)
	}
}

func TestClerkWebhookUserDeletedMissingIDIsBadRequest(t *testing.T) {
	deleteCalled := false
	deleteUser := func(context.Context, string) error { deleteCalled = true; return nil }
	srv := newWebhookTestServer(t, deleteUser, noopUpdateUserEmail)

	body := []byte(`{"type":"user.deleted","data":{}}`)
	resp := postWebhook(t, srv, testWebhookSecretBytes, body)

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	if deleteCalled {
		t.Error("deleteUser called despite a missing id")
	}
}

func TestClerkWebhookUserUpdatedCallsUpdateUserEmailWithPrimaryAddress(t *testing.T) {
	var gotID, gotEmail string
	updateUserEmail := func(_ context.Context, id, email string) error {
		gotID, gotEmail = id, email
		return nil
	}
	srv := newWebhookTestServer(t, noopDeleteUser, updateUserEmail)

	body := []byte(`{"type":"user.updated","data":{
		"id":"user_456",
		"primary_email_address_id":"idn_2",
		"email_addresses":[
			{"id":"idn_1","email_address":"old@example.com"},
			{"id":"idn_2","email_address":"new@example.com"}
		]
	}}`)
	resp := postWebhook(t, srv, testWebhookSecretBytes, body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if gotID != "user_456" || gotEmail != "new@example.com" {
		t.Errorf("updateUserEmail called with (%q, %q), want (user_456, new@example.com)", gotID, gotEmail)
	}
}

func TestClerkWebhookUserUpdatedWithNoResolvablePrimaryEmailSkipsUpdate(t *testing.T) {
	called := false
	updateUserEmail := func(context.Context, string, string) error {
		called = true
		return nil
	}
	srv := newWebhookTestServer(t, noopDeleteUser, updateUserEmail)

	// primary_email_address_id names an id absent from email_addresses.
	body := []byte(`{"type":"user.updated","data":{
		"id":"user_789",
		"primary_email_address_id":"idn_missing",
		"email_addresses":[{"id":"idn_1","email_address":"a@example.com"}]
	}}`)
	resp := postWebhook(t, srv, testWebhookSecretBytes, body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if called {
		t.Error("updateUserEmail called despite no resolvable primary address")
	}
}

func TestClerkWebhookUnknownEventTypeIsNoOp(t *testing.T) {
	deleteCalled, updateCalled := false, false
	deleteUser := func(context.Context, string) error { deleteCalled = true; return nil }
	updateUserEmail := func(context.Context, string, string) error { updateCalled = true; return nil }
	srv := newWebhookTestServer(t, deleteUser, updateUserEmail)

	body := []byte(`{"type":"session.created","data":{}}`)
	resp := postWebhook(t, srv, testWebhookSecretBytes, body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if deleteCalled || updateCalled {
		t.Error("an unrecognized event type triggered a write")
	}
}

func TestClerkWebhookRejectsBadSignature(t *testing.T) {
	deleteCalled := false
	deleteUser := func(context.Context, string) error { deleteCalled = true; return nil }
	srv := newWebhookTestServer(t, deleteUser, noopUpdateUserEmail)

	wrongSecret, err := decodeWebhookSecret("whsec_" + base64.StdEncoding.EncodeToString([]byte("a-different-signing-secret-value")))
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"type":"user.deleted","data":{"id":"user_123"}}`)
	resp := postWebhook(t, srv, wrongSecret, body)

	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", resp.StatusCode)
	}
	if deleteCalled {
		t.Error("deleteUser called despite a bad signature")
	}
}

func TestClerkWebhookMalformedPayloadWithValidSignatureIsBadRequest(t *testing.T) {
	srv := newWebhookTestServer(t, noopDeleteUser, noopUpdateUserEmail)
	body := []byte(`not json`)
	resp := postWebhook(t, srv, testWebhookSecretBytes, body)

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}

func TestClerkWebhookRejectsOversizedBody(t *testing.T) {
	srv := newWebhookTestServer(t, noopDeleteUser, noopUpdateUserEmail)
	body := bytes.Repeat([]byte("a"), maxWebhookBodyBytes+1)
	resp := postWebhook(t, srv, testWebhookSecretBytes, body)

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}

func TestClerkWebhookDeleteFailureIsServiceUnavailable(t *testing.T) {
	deleteUser := func(context.Context, string) error { return errors.New("db down") }
	srv := newWebhookTestServer(t, deleteUser, noopUpdateUserEmail)

	body := []byte(`{"type":"user.deleted","data":{"id":"user_123"}}`)
	resp := postWebhook(t, srv, testWebhookSecretBytes, body)

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", resp.StatusCode)
	}
}

func TestClerkWebhookUpdateFailureIsServiceUnavailable(t *testing.T) {
	updateUserEmail := func(context.Context, string, string) error { return errors.New("db down") }
	srv := newWebhookTestServer(t, noopDeleteUser, updateUserEmail)

	body := []byte(`{"type":"user.updated","data":{
		"id":"user_123",
		"primary_email_address_id":"idn_1",
		"email_addresses":[{"id":"idn_1","email_address":"a@example.com"}]
	}}`)
	resp := postWebhook(t, srv, testWebhookSecretBytes, body)

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", resp.StatusCode)
	}
}
