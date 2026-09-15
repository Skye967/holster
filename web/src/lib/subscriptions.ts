import {
  GATEWAY_CALL_TIMEOUT_MS,
  gatewayFetch,
  withTimeout,
  type GetToken,
} from "@/lib/gateway"
import type { Session } from "@/lib/session"

// Derived, not stored — no flag of its own. A guest's answer is its
// cookie (lib/guest.ts); an account's is its rows. Getting the account branch
// wrong fails safe — gatewayFetch throws GatewaySessionExpiredError, caught
// below and treated the same as any other failure, so a mistake here would
// silently skip a nudge, not crash or leak data.
export async function needsOnboarding(
  session: Pick<Session, "getToken" | "guestProviders">,
): Promise<boolean> {
  // Explicitly null, not truthiness: a guest with nothing picked is `[]`,
  // which is truthy today but is the value most likely to be "tidied" into a
  // .length check that would send it down the account branch.
  if (session.guestProviders !== null) {
    return session.guestProviders.length === 0
  }
  return accountNeedsOnboarding(session.getToken)
}

// The one writer of an account's streaming_subscriptions row, shared by the
// picker's toggle and the sign-in migration (app/signed-in/route.ts) so both
// carry the module's standard timeout instead of one of them hanging.
export async function setSubscription(
  getToken: GetToken,
  providerId: number,
  subscribed: boolean,
): Promise<void> {
  const res = await withTimeout(
    gatewayFetch(
      `/api/subscriptions/${providerId}`,
      {
        method: subscribed ? "PUT" : "DELETE",
        signal: AbortSignal.timeout(GATEWAY_CALL_TIMEOUT_MS),
      },
      getToken,
    ),
    GATEWAY_CALL_TIMEOUT_MS,
  )
  if (!res.ok) throw new Error(`status ${res.status}`)
}

async function accountNeedsOnboarding(getToken: GetToken): Promise<boolean> {
  try {
    const res = await withTimeout(
      gatewayFetch(
        "/api/subscriptions",
        { signal: AbortSignal.timeout(GATEWAY_CALL_TIMEOUT_MS) },
        getToken,
      ),
      GATEWAY_CALL_TIMEOUT_MS,
    )
    if (!res.ok) return false
    const ids: number[] = await res.json()
    return ids.length === 0
  } catch {
    return false
  }
}
