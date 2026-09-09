"use client"

import { useAuth } from "@clerk/nextjs"
import {
  Bookmark,
  Info,
  LogIn,
  MessageSquare,
  Plug,
  Plus,
  Trash2,
} from "lucide-react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { useCallback, useEffect, useRef, useState } from "react"

import { UserMenu } from "@/components/user-menu"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupAction,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { useRowStatus } from "@/hooks/use-row-status"
import { gatewayErrorText } from "@/lib/gateway"
import {
  CONVERSATIONS_CHANGED_EVENT,
  deleteConversation,
  fetchConversations,
  type ConversationSummary,
} from "@/lib/conversations"
import { signInHref } from "@/lib/guest"

const nav = [
  { title: "Chat", href: "/chat", icon: MessageSquare },
  { title: "Watchlist", href: "/watchlist", icon: Bookmark },
  { title: "Connections", href: "/connections", icon: Plug },
]

// The sidebar's conversation list (TASKS.md T20.5) — newest first, a "New
// chat" action, and per-row delete. Kept local to the sidebar rather than a
// shared context: nothing else in the tree needs this list, only the one
// cross-subtree signal a completed turn sends (CONVERSATIONS_CHANGED_EVENT).
// A guest keeps the header and "New chat" — the one way to reset context —
// but has no list to load, since nothing is persisted for it.
function ConversationsGroup({ guest }: { guest: boolean }) {
  const { getToken, isLoaded } = useAuth()
  const pathname = usePathname()
  const router = useRouter()
  const [conversations, setConversations] = useState<
    ConversationSummary[] | null
  >(null)
  const {
    status: deleteStatus,
    setPending,
    setFailure,
  } = useRowStatus<string>()

  // Guards against an out-of-order response overwriting fresher state: an
  // earlier-fired GET that resolves after a later one (two refresh() calls
  // racing, e.g. mount plus a CONVERSATIONS_CHANGED_EVENT arriving close
  // together) must not clobber what the later one fetched. Bumped by
  // handleDelete too, so its own optimistic removal can't be overwritten by
  // a slower, pre-delete GET landing after it and briefly resurrecting a
  // just-deleted conversation.
  const requestSeqRef = useRef(0)

  const refresh = useCallback(() => {
    if (guest || !isLoaded) return
    const seq = ++requestSeqRef.current
    fetchConversations(getToken)
      .then((data) => {
        if (seq === requestSeqRef.current) setConversations(data)
      })
      .catch(() => {
        // Best-effort: the sidebar just keeps showing whatever it last had.
      })
  }, [guest, isLoaded, getToken])

  useEffect(() => {
    refresh()
  }, [refresh])

  // Catches a brand-new conversation's first message landing, and its
  // title being derived for the first time — chat-panel.tsx dispatches this
  // on every completed turn rather than this component polling.
  useEffect(() => {
    window.addEventListener(CONVERSATIONS_CHANGED_EVENT, refresh)
    return () =>
      window.removeEventListener(CONVERSATIONS_CHANGED_EVENT, refresh)
  }, [refresh])

  const handleDelete = useCallback(
    async (id: string) => {
      setPending(id)
      try {
        await deleteConversation(getToken, id)
        requestSeqRef.current++ // invalidate any in-flight refresh from before this delete
        setConversations((prev) => prev?.filter((c) => c.id !== id) ?? prev)
        if (pathname === `/chat/${id}`) router.push("/chat")
      } catch (err) {
        setFailure(
          id,
          gatewayErrorText(err, "Couldn't delete that — try again"),
        )
      }
    },
    [getToken, pathname, router, setPending, setFailure],
  )

  return (
    <SidebarGroup>
      <SidebarGroupLabel>Conversations</SidebarGroupLabel>
      <SidebarGroupAction
        aria-label="New chat"
        onClick={() => router.push(`/chat/${crypto.randomUUID()}`)}
      >
        <Plus />
      </SidebarGroupAction>
      {guest ? (
        <p className="px-2 text-xs text-muted-foreground">
          <Link
            href={signInHref(pathname)}
            className="text-primary underline underline-offset-4"
          >
            Sign in
          </Link>{" "}
          to save future chats.
        </p>
      ) : (
        <SidebarMenu>
          {conversations?.map((c) => (
            <SidebarMenuItem key={c.id}>
              <SidebarMenuButton
                isActive={pathname === `/chat/${c.id}`}
                render={
                  <Link href={`/chat/${c.id}`}>
                    <span className="truncate">{c.title}</span>
                  </Link>
                }
              />
              <SidebarMenuAction
                showOnHover
                aria-label={`Delete "${c.title}"`}
                disabled={deleteStatus[c.id]?.pending}
                onClick={() => handleDelete(c.id)}
              >
                <Trash2 />
              </SidebarMenuAction>
              {deleteStatus[c.id]?.error && (
                <p className="px-2 pt-1 text-xs text-destructive">
                  {deleteStatus[c.id]?.error}
                </p>
              )}
            </SidebarMenuItem>
          ))}
        </SidebarMenu>
      )}
    </SidebarGroup>
  )
}

// Credits stays reachable for a guest: TMDB and JustWatch attribution is a
// shipping requirement (CLAUDE.md), not something behind the user menu.
function GuestMenu() {
  const pathname = usePathname()

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <SidebarMenuButton
          size="lg"
          render={
            <Link href={signInHref(pathname)}>
              <LogIn />
              <span>Sign in</span>
            </Link>
          }
        />
      </SidebarMenuItem>
      <SidebarMenuItem>
        <SidebarMenuButton
          render={
            <Link href="/credits">
              <Info />
              <span>Credits</span>
            </Link>
          }
        />
      </SidebarMenuItem>
    </SidebarMenu>
  )
}

// guest comes from the server layout's auth(), not Clerk's client state, so
// the shell renders in the right mode on the first paint with no flicker.
export function AppSidebar({ guest }: { guest: boolean }) {
  const pathname = usePathname()

  return (
    <Sidebar>
      <SidebarHeader className="px-3 py-3">
        <Link href="/chat" className="text-base font-semibold tracking-tight">
          Holster
        </Link>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <nav aria-label="Main">
            <SidebarMenu>
              {nav.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    isActive={
                      pathname === item.href ||
                      pathname.startsWith(`${item.href}/`)
                    }
                    render={
                      // A guest's thread lives only on the open socket, so
                      // sending them to /chat — which mints a fresh id — would
                      // discard it with no warning. Pointing the item at the
                      // conversation they are already in makes the click the
                      // no-op it looks like; "New chat" above is the explicit
                      // reset. An account resumes its last conversation from
                      // the database, so it needs none of this.
                      <Link
                        href={
                          guest &&
                          item.href === "/chat" &&
                          pathname.startsWith("/chat/")
                            ? pathname
                            : item.href
                        }
                      >
                        <item.icon />
                        <span>{item.title}</span>
                      </Link>
                    }
                  />
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </nav>
        </SidebarGroup>
        <ConversationsGroup guest={guest} />
      </SidebarContent>
      <SidebarFooter>{guest ? <GuestMenu /> : <UserMenu />}</SidebarFooter>
    </Sidebar>
  )
}
