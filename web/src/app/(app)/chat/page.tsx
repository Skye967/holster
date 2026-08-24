import { auth } from "@clerk/nextjs/server";

export default async function ChatPage() {
  await auth.protect();

  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold tracking-tight">Chat</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        The agent isn&apos;t wired up yet.
      </p>
    </div>
  );
}
