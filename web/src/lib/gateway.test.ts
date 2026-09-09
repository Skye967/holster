import { afterEach, describe, expect, test } from "bun:test"

import {
  GatewaySessionExpiredError,
  SESSION_EXPIRED_TEXT,
  gatewayErrorText,
  gatewayFetch,
  withTimeout,
} from "./gateway"

describe("gatewayFetch with getToken null", () => {
  const originalFetch = globalThis.fetch
  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test("sends no Authorization header and does not retry a 401", async () => {
    const calls: RequestInit[] = []
    globalThis.fetch = (async (_url: unknown, init?: RequestInit) => {
      calls.push(init ?? {})
      return new Response("", { status: 401 })
    }) as typeof fetch

    const res = await gatewayFetch("/guest/providers", {}, null)

    expect(res.status).toBe(401)
    expect(calls).toHaveLength(1)
    expect(new Headers(calls[0].headers).has("Authorization")).toBe(false)
  })
})

describe("gatewayErrorText", () => {
  test("returns the shared session-expired text for a GatewaySessionExpiredError", () => {
    expect(gatewayErrorText(new GatewaySessionExpiredError(), "fallback")).toBe(
      SESSION_EXPIRED_TEXT,
    )
  })

  test("returns the caller's fallback for any other error", () => {
    expect(gatewayErrorText(new Error("boom"), "fallback")).toBe("fallback")
  })
})

describe("withTimeout", () => {
  test("resolves with the promise's value when it settles first", async () => {
    const fast = new Promise<string>((resolve) =>
      setTimeout(() => resolve("done"), 5),
    )
    await expect(withTimeout(fast, 50)).resolves.toBe("done")
  })

  test("rejects once the timeout elapses before the promise settles", async () => {
    const slow = new Promise<string>((resolve) =>
      setTimeout(() => resolve("too late"), 50),
    )
    await expect(withTimeout(slow, 5)).rejects.toThrow("gateway call timed out")
  })

  test("propagates the wrapped promise's own rejection, not a timeout error", async () => {
    const failing = Promise.reject(new Error("boom"))
    await expect(withTimeout(failing, 50)).rejects.toThrow("boom")
  })
})
