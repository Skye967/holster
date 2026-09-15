import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  // Emits .next/standalone — a minimal server tracing only the deps actually
  // used, copied wholesale into the runtime stage of web/Dockerfile.
  output: "standalone",
  images: {
    // TMDB's CDN for poster/logo images — agent/tmdb.py builds these URLs
    // (see ../CLAUDE.md's attribution note); every surface that renders one
    // (the picker's provider logos, the chat's title cards) needs this entry.
    remotePatterns: [{ protocol: "https", hostname: "image.tmdb.org" }],
  },
}

export default nextConfig
