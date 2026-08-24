# credentials

Go. The token vault. **The only service that can decrypt.**

## Responsibility

Store OAuth tokens encrypted, refresh them before expiry, and perform authorized
calls on behalf of other services without ever handing out the secret.

## Encryption

- `CREDENTIALS_MASTER_KEY` is read by this service and no other.
- A per-user key is derived from it — `HMAC(master_key, user_id)` — so tokens for
  different users are encrypted under different keys. Compromising one user's derived
  key does not unlock anyone else's rows.
- Tokens are encrypted before insert and decrypted only in memory, at call time.

A stolen database dump is therefore not a stolen set of accounts.

## Refresh

Access tokens are short-lived (~1h); refresh tokens are long-lived. On use, the
service checks `expires_at`, refreshes if needed, persists the new pair, and proceeds.
The user is never prompted to reconnect unless the refresh token itself is revoked.

## The rule that shapes the API

**Never return a decrypted token across the network.** Callers — including `agent/` —
ask this service to *make the call*, or request a short-lived scoped handle. They do
not receive credentials. This is why the service exists as its own boundary rather
than as a library the agent imports.
