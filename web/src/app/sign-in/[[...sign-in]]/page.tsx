import { SignIn } from "@clerk/nextjs"

// Every sign-in lands on /signed-in (app/signed-in/route.ts) unless a
// redirect_url says otherwise — and signInHref always points that at
// /signed-in too, so a guest's picks are carried into the account either way.
export default function SignInPage() {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <SignIn fallbackRedirectUrl="/signed-in" />
    </div>
  )
}
