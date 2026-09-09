import { describe, expect, test } from "bun:test"

import { needsOnboarding } from "./subscriptions"

// A guest's answer comes from its cookie alone. These pin the `!== null`
// check: `[]` is a guest who has picked nothing, and the guest branch must
// never fall through to the account one — a `.length`/truthiness "tidy-up"
// there would send every guest to the gateway with no token.
describe("needsOnboarding, guest branch", () => {
  // getToken would throw if the account branch were ever reached, so these
  // assert the branch taken as much as the value returned.
  const getToken = (() => {
    throw new Error("needsOnboarding reached the account branch for a guest")
  }) as never

  test("a guest with nothing picked needs onboarding", async () => {
    expect(await needsOnboarding({ getToken, guestProviders: [] })).toBe(true)
  })

  test("a guest with picks does not", async () => {
    expect(await needsOnboarding({ getToken, guestProviders: [8] })).toBe(false)
  })
})
