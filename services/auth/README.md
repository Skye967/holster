# auth

Go. Owns every conversation with an external OAuth provider.

## Responsibility

1. **Provider search.** Given a name the user typed ("gmail", "youtube"), return
   candidate providers with descriptions so they can confirm which one they meant.
2. **OAuth initiation.** Build the authorization URL for the chosen provider,
   including PKCE challenge and CSRF `state`, and hand it back for redirect.
3. **Callback handling.** Validate `state`, exchange the authorization code for an
   access/refresh token pair, and hand that pair to `credentials/`.

## Rules

- **Never stores a token.** Tokens are forwarded to `credentials/` and dropped. This
  service holds no master key and has no long-lived secret material beyond the OAuth
  client credentials.
- **Never sees a user's password.** Authentication happens on the provider's own
  domain. MFA is the provider's concern entirely.
- `state` is single-use and bound to the Clerk user ID.

## Adding a provider

Providers are config, not code paths: client ID/secret, authorization and token URLs,
and the scope set. Adding one should not require a new handler.
