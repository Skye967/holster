import { SignUp } from "@clerk/nextjs"

// See sign-in/page.tsx — same landing, same reason.
export default function SignUpPage() {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <SignUp fallbackRedirectUrl="/signed-in" />
    </div>
  )
}
