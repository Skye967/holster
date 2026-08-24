# web

Next.js frontend. TypeScript, Tailwind, shadcn/ui, Clerk for auth.

## Routes

| Route | Purpose |
|---|---|
| `/sign-in`, `/sign-up` | Clerk-hosted authentication |
| `/chat` | The single unified conversation (default view after sign-in) |
| `/connections` | Add and remove authorized apps |

## Notes

- **Icons are `lucide-react`** — the same set the Figma design uses, so the two match
  exactly. Don't introduce a second icon library.
- **The browser never receives an OAuth token.** Connecting an app is a redirect to
  the provider; the resulting tokens go straight to `services/credentials` and stay
  server-side. The frontend only ever learns *that* a service is connected.
- **No service picker in chat.** The agent decides what to reach for. If you find
  yourself adding a dropdown to choose a service, that's scope creep.

## Design reference

Figma: https://www.figma.com/design/j8EQVXshnq7J6HYGUmUKt6

The Figma file is a rough reference, not a spec — it has known layout defects and is
missing the add-connection flow. Prefer shadcn defaults over reproducing it pixel for
pixel.
