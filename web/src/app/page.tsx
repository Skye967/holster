import { UserButton } from "@clerk/nextjs";
import { auth } from "@clerk/nextjs/server";
import { MessageSquare } from "lucide-react";

import { Button } from "@/components/ui/button";

export default async function Home() {
  await auth.protect();

  return (
    <div className="flex min-h-svh flex-col items-center justify-center gap-4">
      <h1 className="text-2xl font-semibold tracking-tight">Holster</h1>
      <Button>
        <MessageSquare />
        New chat
      </Button>
      <UserButton />
    </div>
  );
}
