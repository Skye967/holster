import { cookies } from "next/headers"

import { AppSidebar } from "@/components/app-sidebar"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { getSession } from "@/lib/session"

// Guest-visible by default, on purpose: the shell touches no user data, and
// each page that reads user-scoped data decides for itself. The gateway is the
// boundary. Why: DECISIONS.md "Optional sign-in".
export default async function AppLayout({ children }: LayoutProps<"/">) {
  const { userId } = await getSession()

  // SidebarProvider writes this cookie on toggle — read it back so a collapsed
  // sidebar survives a reload.
  const cookieStore = await cookies()
  const defaultOpen = cookieStore.get("sidebar_state")?.value !== "false"

  return (
    <SidebarProvider defaultOpen={defaultOpen}>
      <AppSidebar guest={!userId} />
      <SidebarInset>
        <header className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger />
        </header>
        {/* min-h-0: without it a flex item defaults to min-height:auto (its
            content's height), so a full-height child like ChatPanel can
            never get a bounded height to scroll within — the whole page
            would scroll instead, taking the pinned input with it. */}
        <div className="min-h-0 flex-1">{children}</div>
      </SidebarInset>
    </SidebarProvider>
  )
}
