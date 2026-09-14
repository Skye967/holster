import { NextResponse } from "next/server"

// NextResponse.redirect() requires an absolute URL, built here from
// request.nextUrl — which trusts whatever Host header the server received
// for this request. Self-hosted behind Docker (and specifically running the
// standalone server under bun, not node — see ../../Dockerfile's other bun
// runtime caveat), that Host can come back wrong even when the browser is on
// a perfectly good origin, sending the user to the container's own hostname.
// Both callers only ever redirect same-origin, so there is nothing an
// absolute URL adds: a bare relative Location header is valid per the HTTP
// spec, and the browser resolves it against the origin it actually used,
// never the server's own (possibly wrong) idea of that origin.
export function redirectTo(path: string): NextResponse {
  return new NextResponse(null, { status: 307, headers: { Location: path } })
}
