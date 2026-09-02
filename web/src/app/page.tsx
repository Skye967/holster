import { auth } from "@clerk/nextjs/server"
import { redirect } from "next/navigation"

import { needsOnboarding } from "@/lib/subscriptions"

export default async function Home() {
  const { getToken } = await auth.protect()
  redirect((await needsOnboarding(getToken)) ? "/onboarding" : "/chat")
}
