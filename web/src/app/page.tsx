import { redirect } from "next/navigation"

import { getSession } from "@/lib/session"
import { needsOnboarding } from "@/lib/subscriptions"

export default async function Home() {
  const session = await getSession()
  redirect((await needsOnboarding(session)) ? "/onboarding" : "/chat")
}
