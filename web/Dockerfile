# Two stages, mirroring agent/Dockerfile and services/gateway/Dockerfile: build
# with bun (this repo's package manager, per ../CLAUDE.md), then copy only the
# standalone output into a clean runtime image.
#
# Pinned to 1.4.2, not package.json's 1.3.13: that version segfaults running
# Next's compiled Turbopack runtime under bun on Linux/arm64 (reproduces in this
# image; not a musl-vs-glibc issue, and doesn't reproduce on macOS). 1.4.2 is
# lockfile-compatible and builds clean — bump package.json's pin separately.
#
# Shared by both FROMs below so a version bump can't update one and miss the
# other.
ARG BUN_VERSION=1.4.2-alpine

FROM oven/bun:${BUN_VERSION} AS builder

WORKDIR /app

COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY . .

# NEXT_PUBLIC_* vars are inlined into the client bundle at build time, not read
# at runtime — they must arrive as build args, unlike every other env var this
# app uses.

# Two required at build time. NEXT_PUBLIC_* are compiled in rather than read at
# runtime, so a placeholder key builds clean and then 500s every request. The
# shape is checked here instead, against Clerk's own rule: a pk_test_/pk_live_
# prefix plus a base64 segment, three underscore-separated parts exactly —
# `.env.example`'s pk_test_replace_me has four.
ARG NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ARG NEXT_PUBLIC_GATEWAY_URL
RUN case "$NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY" in \
      pk_test_?*_*|pk_live_?*_*) ok=no ;; \
      pk_test_?*|pk_live_?*) ok=yes ;; \
      *) ok=no ;; \
    esac; \
    [ "$ok" = yes ] || { echo "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY is not a real Clerk key (got '${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY:-<empty>}')" >&2; exit 1; }
RUN [ -n "$NEXT_PUBLIC_GATEWAY_URL" ] || (echo "NEXT_PUBLIC_GATEWAY_URL not set" >&2 && exit 1)

# Clerk's own custom-page routing — same values as ../.env.example. Defaulted
# since they're static app routes, not deployment-specific config. Where a
# sign-in lands is no longer one of these: both components set
# fallbackRedirectUrl to /signed-in in code (app/signed-in/route.ts).
ARG NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
ARG NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up

RUN bun run build

FROM oven/bun:${BUN_VERSION}

WORKDIR /app

# output: "standalone" (next.config.ts) traces only the deps the server needs;
# static assets and public/ ship separately, per Next's own docs.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# next/image writes optimised posters here, and the copies above are root-owned,
# so without this the optimiser fails every write and silently caches nothing.
# The rest of the tree stays root-owned on purpose: the renderer cannot rewrite
# its own code. That also means .next/server is read-only, so ISR is not
# supported in this image — a route adding `revalidate` would fail the same
# quiet way.
RUN mkdir -p .next/cache && chown 65534:65534 .next/cache

# standalone's server.js hardcodes a localhost bind unless HOSTNAME overrides
# it, which is unreachable from outside the container.
ENV HOSTNAME=0.0.0.0

# nobody: the process listens on 3000 and writes nothing outside that cache.
USER 65534:65534
EXPOSE 3000
CMD ["bun", "server.js"]
