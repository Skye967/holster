import { describe, expect, test } from "bun:test"

import { isValidConversationId } from "./conversations"

describe("isValidConversationId", () => {
  test("accepts a well-formed uuid", () => {
    expect(isValidConversationId("11111111-1111-1111-1111-111111111111")).toBe(
      true,
    )
  })
  test("rejects an empty string", () => {
    expect(isValidConversationId("")).toBe(false)
  })
  test("rejects a malformed id", () => {
    expect(isValidConversationId("not-a-uuid")).toBe(false)
  })
})
