import {
  GATEWAY_CALL_TIMEOUT_MS,
  gatewayFetch,
  withTimeout,
  type GetToken,
} from "@/lib/gateway"

// Derived, not stored (TASKS.md T15.5: "no new flag"). Callers must have already
// resolved auth (auth.protect()) before calling this — documented here, not
// type-enforced (GetToken is a structural type; nothing stops a caller from
// passing one sourced elsewhere). Accepted: both current call sites are correct,
// and getting this wrong fails safe — gatewayFetch throws
// GatewaySessionExpiredError, caught below and treated the same as any other
// failure, so a future mistake here would silently skip a nudge, not crash or
// leak data. Not worth enforcing via the type system for a 2-call-site app.
export async function needsOnboarding(getToken: GetToken): Promise<boolean> {
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
