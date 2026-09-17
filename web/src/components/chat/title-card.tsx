import {
  Ban,
  Bookmark,
  BookmarkCheck,
  Eye,
  MoreHorizontal,
  RotateCcw,
  Star,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react"
import Image from "next/image"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import type { AgentPick, AgentProvider } from "@/lib/chat-socket"
import type { Verdict } from "@/lib/verdicts"

// The four judgments, set from the smaller judgment control below —
// want_to_watch is
// deliberately not one of these, since it has its own always-visible bookmark
// control. Order matches title_verdicts' check constraint
// (supabase/migrations/20260903002450_verdicts.sql).
const JUDGMENTS: { value: Verdict; label: string; icon: typeof ThumbsUp }[] = [
  { value: "liked", label: "Liked", icon: ThumbsUp },
  { value: "disliked", label: "Disliked", icon: ThumbsDown },
  { value: "seen", label: "Seen", icon: Eye },
  { value: "not_interested", label: "Not interested", icon: Ban },
]

function formatRuntime(minutes: number | null): string | null {
  if (minutes == null) return null
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

// available_on is null when the availability check itself failed, distinct
// from an empty array (confirmed on none of the caller's services) — these
// must read differently, never collapsed to the same "nothing here" state
// (see AgentPick's own comment). available_on has already been cut to the
// user's own services by the agent; rent_on has not been, because those are
// stores rather than subscriptions.
//
// One rule for every card, with no per-turn flag, which is what gives the
// name-lookup and the watchlist their rent line for free. Included wins over
// rentable and the two never share a line: a title the user already pays for
// must not be offered to them for money again. Renting is offered only
// alongside "Not on your services", so the card says the title isn't included
// before it says what it would cost to watch. A rent search whose title has
// no stores in this region therefore reads as the bare "Not on your
// services" — accurate, and the price of having no per-turn flag.
function Availability({
  availableOn,
  rentOn,
  watchLink,
}: {
  availableOn: AgentProvider[] | null
  rentOn: AgentProvider[] | null
  watchLink: string | null
}) {
  if (availableOn === null) {
    return (
      <p className="text-xs text-muted-foreground">
        Availability unknown right now
      </p>
    )
  }
  if (availableOn.length > 0) {
    return (
      <p className="text-xs text-muted-foreground">
        Streaming on {availableOn.map((p) => p.provider_name).join(", ")}
      </p>
    )
  }
  const stores = (rentOn ?? []).map((p) => p.provider_name).join(", ")
  return (
    <p className="text-xs text-muted-foreground">
      Not on your services
      {stores && (
        <>
          {" · Rent or buy on "}
          {/* No price anywhere: TMDB's watch-provider data says where a title
              can be rented, never for how much, so any figure here would be
              one we cannot stand behind. py-1 pads the 16px line box so the
              link is not a hairline target; it is still an inline link in
              running prose, not the thumb-sized target web/CLAUDE.md asks
              for elsewhere. */}
          {watchLink ? (
            <a
              href={watchLink}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block py-1 text-primary underline underline-offset-4"
            >
              {stores}
            </a>
          ) : (
            stores
          )}
        </>
      )}
    </p>
  )
}

// The MVP's main visual component, shared with the watchlist
// view. Meant to sit in a horizontal snap-scroll row (see chat-panel.tsx),
// which is why it has a fixed width rather than flexing to its container.
//
// Verdict state and its mutations live in the caller (chat-panel.tsx), not
// here — this stays a presentational component, and the same verdict map
// keys every card showing the same title, so a toggle on one instance is
// reflected on every other (e.g. a title repeated by "show me more").
export function TitleCard({
  pick,
  verdict,
  pending,
  error,
  onSetVerdict,
  onClearVerdict,
}: {
  pick: AgentPick
  verdict?: Verdict
  pending?: boolean
  error?: string
  onSetVerdict: (
    tmdbId: number,
    mediaType: "movie" | "tv",
    verdict: Verdict,
  ) => void
  // expectedVerdict is the verdict this card is currently showing — the
  // caller sends it as a compare-and-delete guard, so a stale click (this
  // card re-rendered from state that's since changed elsewhere) clears
  // nothing instead of erasing whatever's actually there now.
  onClearVerdict: (
    tmdbId: number,
    mediaType: "movie" | "tv",
    expectedVerdict: Verdict,
  ) => void
}) {
  const runtime = formatRuntime(pick.runtime_minutes)
  const cast = pick.cast.slice(0, 3)

  const saved = verdict === "want_to_watch"
  const judgment = JUDGMENTS.find((j) => j.value === verdict)
  // Shared by the bookmark button and the radio group below, but *not* the
  // dropdown trigger (see its own disabled prop) — a locked judgment still
  // blocks switching straight to a different one, or to want_to_watch, but
  // the menu itself must stay reachable so its "Clear rating" item can fire —
  // a locked judgment can be cleared (ARCHITECTURE.md).
  const locked = pending || judgment != null

  return (
    <Card className="w-64 shrink-0 snap-start">
      {pick.poster_url ? (
        <Image
          src={pick.poster_url}
          alt=""
          width={256}
          height={384}
          className="aspect-2/3 w-full object-cover"
        />
      ) : (
        <div className="aspect-2/3 w-full bg-muted" />
      )}
      <CardHeader>
        <CardTitle className="truncate">{pick.title}</CardTitle>
        <CardAction className="flex items-center gap-1">
          {/* The primary save control — one tap, toggles,
              saves immediately, no confirmation. Disabled once a judgment is
              locked — want_to_watch must not be silently set over a locked
              judgment, and the gateway rejects that write anyway. Unlike the
              dropdown trigger below, this stays disabled while locked: there
              is no "clear from the bookmark" path, only via the dropdown's
              own clear item. */}
          <Button
            variant="ghost"
            size="icon-sm"
            disabled={locked}
            aria-label={saved ? "Remove from watchlist" : "Save to watchlist"}
            onClick={() =>
              saved
                ? onClearVerdict(pick.tmdb_id, pick.media_type, "want_to_watch")
                : onSetVerdict(pick.tmdb_id, pick.media_type, "want_to_watch")
            }
          >
            {saved ? (
              <BookmarkCheck className="size-4" />
            ) : (
              <Bookmark className="size-4" />
            )}
          </Button>
          {/* The smaller judgment control. Disabled only by `pending`, not
              `locked` — once one of the four judgments is set the menu must
              still open, so its "Clear rating" item (below) stays reachable:
              a locked judgment can be cleared (ARCHITECTURE.md). Swaps
              to that judgment's own icon so the locked state is still
              visible, not just inert. */}
          <DropdownMenu>
            <DropdownMenuTrigger
              render={
                <Button
                  variant="ghost"
                  size="icon-sm"
                  disabled={pending}
                  aria-label={
                    judgment ? `Rated: ${judgment.label}` : "Rate this title"
                  }
                >
                  {judgment ? (
                    <judgment.icon className="size-4" />
                  ) : (
                    <MoreHorizontal className="size-4" />
                  )}
                </Button>
              }
            />
            <DropdownMenuContent align="end">
              {/* disabled here, not just on the trigger: switching directly
                  to a *different* judgment stays blocked (the gateway's
                  upsert guard still rejects it) — clearing first, below, is
                  the only path to a new judgment. Also closes the same gap
                  the trigger's disabled state can't: if the menu is already
                  open when a write locks the judgment (or a separate
                  bookmark click starts one), base-ui's MenuRadioGroup
                  disabled prop propagates to every item regardless. */}
              <DropdownMenuRadioGroup
                value={judgment?.value}
                disabled={locked}
                onValueChange={(value) =>
                  onSetVerdict(pick.tmdb_id, pick.media_type, value as Verdict)
                }
              >
                {JUDGMENTS.map((j) => (
                  <DropdownMenuRadioItem key={j.value} value={j.value}>
                    <j.icon />
                    {j.label}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
              {judgment && (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    disabled={pending}
                    onClick={() =>
                      onClearVerdict(
                        pick.tmdb_id,
                        pick.media_type,
                        judgment.value,
                      )
                    }
                  >
                    <RotateCcw />
                    Clear rating
                  </DropdownMenuItem>
                </>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
          {pick.year != null && <span>{pick.year}</span>}
          {runtime && <span>{runtime}</span>}
          {pick.genre_names.length > 0 && (
            <span>{pick.genre_names.join(", ")}</span>
          )}
          <span className="flex items-center gap-0.5">
            <Star className="size-3 fill-current" aria-hidden />
            {pick.vote_average.toFixed(1)}
          </span>
        </div>
        {cast.length > 0 && (
          <p className="truncate text-xs text-muted-foreground">
            {cast.join(", ")}
          </p>
        )}
        {pick.blurb && <p className="text-sm">{pick.blurb}</p>}
        <Availability
          availableOn={pick.available_on}
          rentOn={pick.rent_on}
          watchLink={pick.watch_link}
        />
        {error && <p className="text-xs text-destructive">{error}</p>}
      </CardContent>
    </Card>
  )
}
