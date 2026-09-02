import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  images: {
    // TMDB's CDN for poster/logo images — agent/tmdb.py builds these URLs
    // (see ../CLAUDE.md's attribution note); every surface that renders one
    // (this task's provider logos, T16's title cards) needs this entry.
    remotePatterns: [{ protocol: "https", hostname: "image.tmdb.org" }],
  },
}

export default nextConfig
