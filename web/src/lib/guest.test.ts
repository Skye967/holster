import { describe, expect, test } from "bun:test"

import {
  MAX_GUEST_PROVIDERS,
  MAX_PROVIDER_ID,
  isValidProviderId,
  parseGuestProviders,
  safeNext,
} from "./guest"

const base = new URL("https://holster.example/signed-in")

describe("safeNext", () => {
  test("keeps a same-origin path with its query", () => {
    expect(safeNext("/chat/abc?x=1", base)).toBe("/chat/abc?x=1")
    expect(safeNext("https://holster.example/watchlist", base)).toBe(
      "/watchlist",
    )
  })

  test("sends anything off-origin home", () => {
    for (const value of [
      "//evil.example",
      "/\\evil.example",
      "https://evil.example/x",
      "javascript:alert(1)",
      null,
      "",
    ]) {
      expect(safeNext(value, base)).toBe("/")
    }
  })

  // The vectors a character-level guard misses: URL parsing strips these
  // before resolving, so "/<tab>/evil" becomes protocol-relative "//evil".
  test("sends control characters home rather than resolving them", () => {
    for (const c of ["\t", "\n", "\r"]) {
      expect(safeNext(`/${c}/evil.example`, base)).toBe("/")
    }
  })
})

describe("parseGuestProviders", () => {
  test("keeps positive integers, drops the rest, and dedupes", () => {
    expect(parseGuestProviders("8,8,0,-2,x,,15")).toEqual([8, 15])
    expect(parseGuestProviders(undefined)).toEqual([])
  })

  test("caps the list", () => {
    const many = Array.from({ length: MAX_GUEST_PROVIDERS + 10 }, (_, i) =>
      String(i + 1),
    ).join(",")
    expect(parseGuestProviders(many)).toHaveLength(MAX_GUEST_PROVIDERS)
  })

  // Number.isInteger says yes to 1e21 and to Number.MAX_VALUE, so without a
  // ceiling either could reach the cookie — and from there setSubscription at
  // sign-in, and the gateway's own providers frame.
  test("drops ids too large to be a provider", () => {
    expect(parseGuestProviders(`8,${MAX_PROVIDER_ID + 1}`)).toEqual([8])
    expect(parseGuestProviders("1e21")).toEqual([])
    expect(parseGuestProviders(String(Number.MAX_VALUE))).toEqual([])
  })
})

describe("isValidProviderId", () => {
  test("accepts a real provider id and nothing exotic", () => {
    expect(isValidProviderId(8)).toBe(true)
    expect(isValidProviderId(MAX_PROVIDER_ID)).toBe(true)
    for (const bad of [0, -1, 1.5, NaN, Infinity, 1e21, MAX_PROVIDER_ID + 1]) {
      expect(isValidProviderId(bad)).toBe(false)
    }
  })
})
