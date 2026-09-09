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

# Two required at build time — the app is not meaningfully functional without
# them, and an empty value only surfaces at container runtime (500s), not here.
ARG NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ARG NEXT_PUBLIC_GATEWAY_URL
RUN [ -n "$NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY" ] || (echo "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY not set" >&2 && exit 1)
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

# nobody: the process listens on 3000 and writes nothing, so it never needs root.
USER 65534:65534
EXPOSE 3000
CMD ["bun", "server.js"]
