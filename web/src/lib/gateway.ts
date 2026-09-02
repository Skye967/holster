// A fetch call to the gateway, hardened the way TASKS.md's T7 notes require:
// Clerk session tokens live 60 seconds, so any idle tab's next request can
// hit a 401 that isn't a real error. This catches it, fetches a fresh token,
// and retries once — only a failed retry becomes a visible error.

export class GatewaySessionExpiredError extends Error {}

type GetToken = (opts: {
  template: string
  skipCache?: boolean
}) => Promise<string | null>

export async function gatewayFetch(
  path: string,
  init: RequestInit,
  getToken: GetToken,
): Promise<Response> {
  const token = await getToken({ template: "gateway" })
  if (!token) throw new GatewaySessionExpiredError()

  const doFetch = (t: string) => {
    // Headers, not a plain-object spread: init.headers may legally be a
    // Headers instance or a tuple array, either of which would silently
    // spread into {} or numeric keys instead of real header entries.
    const headers = new Headers(init.headers)
    headers.set("Authorization", `Bearer ${t}`)
    return fetch(`${process.env.NEXT_PUBLIC_GATEWAY_URL}${path}`, {
      ...init,
      headers,
    })
  }

  let res = await doFetch(token)
  if (res.status === 401) {
    // skipCache: true so this isn't handed back the same about-to-expire
    // token that just failed.
    const fresh = await getToken({ template: "gateway", skipCache: true })
    if (!fresh) throw new GatewaySessionExpiredError()
    res = await doFetch(fresh)
    if (res.status === 401) throw new GatewaySessionExpiredError()
  }
  return res
}
