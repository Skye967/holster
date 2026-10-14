# web

Next.js frontend. TypeScript, Tailwind, shadcn/ui, Clerk for auth.

## Routes

| Route                       | Purpose                                                           |
| --------------------------- | ----------------------------------------------------------------- |
| `/`                         | Redirects to `/onboarding` or `/chat`                             |
| `/sign-in`, `/sign-up`      | Clerk-hosted authentication                                       |
| `/chat`, `/chat/[id]`       | The single conversation; bare `/chat` reopens an account's latest |
| `/connections`              | Tick the streaming services you subscribe to                      |
| `/onboarding`               | The same picker, where `/` lands while nothing is ticked          |
| `/watchlist`                | Titles saved from chat; a sign-in prompt for a guest              |
| `/credits`                  | TMDB and JustWatch attribution                                    |
| `/signed-in`, `/signed-out` | Clerk's landings; carry a guest's picks in, clear the cookie      |

## Notes

- **Icons are `lucide-react`** — the same set the Figma design uses, so the two match
  exactly. Don't introduce a second icon library.
- **Streaming services are a preference, not a connection.** Users tick what they
  subscribe to; those picks hold no secret, and there is no OAuth to any provider
  (`../CLAUDE.md`).
- **No service picker in chat.** The agent decides what to reach for. If you find
  yourself adding a dropdown to choose a service, that's scope creep.

## Design reference

Figma: https://www.figma.com/design/j8EQVXshnq7J6HYGUmUKt6

The Figma file is a rough reference, not a spec — it has known layout defects. Prefer
shadcn defaults over reproducing it pixel for pixel.
