import { Bookmark, Star } from "lucide-react"
import Image from "next/image"

import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { AgentPick, AgentProvider } from "@/lib/chat-socket"

function formatRuntime(minutes: number | null): string | null {
  if (minutes == null) return null
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

// available_on is null when the availability check itself failed, distinct
// from an empty array (confirmed on none of the caller's services) — these
// must read differently, never collapsed to the same "nothing here" state
// (see AgentPick's own comment, and CLAUDE.md's availability invariant:
// never show a service list including one the user doesn't have).
function Availability({
  availableOn,
}: {
  availableOn: AgentProvider[] | null
}) {
  if (availableOn === null) {
    return (
      <p className="text-xs text-muted-foreground">
        Availability unknown right now
      </p>
    )
  }
  if (availableOn.length === 0) {
    return <p className="text-xs text-muted-foreground">Not on your services</p>
  }
  return (
    <p className="text-xs text-muted-foreground">
      Streaming on {availableOn.map((p) => p.provider_name).join(", ")}
    </p>
  )
}

// The MVP's main visual component (TASKS.md T16.5) — reused as-is by T18
// (verdict controls) and T18.5 (watchlist) once those land. Meant to sit in
// a horizontal snap-scroll row (see chat-panel.tsx), which is why it has a
// fixed width rather than flexing to its container.
export function TitleCard({ pick }: { pick: AgentPick }) {
  const runtime = formatRuntime(pick.runtime_minutes)
  const cast = pick.cast.slice(0, 3)

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
        <CardAction>
          {/* Visual stub only (TASKS.md T18 wires the real save control) —
              a plain icon, not a button, so it can't be tapped into implying
              a save that doesn't happen. */}
          <Bookmark className="size-4 text-muted-foreground" aria-hidden />
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
        <p className="text-sm">{pick.blurb}</p>
        <Availability availableOn={pick.available_on} />
      </CardContent>
    </Card>
  )
}
