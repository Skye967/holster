import { describe, expect, test } from "bun:test"

import { withTimeout } from "./gateway"

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
