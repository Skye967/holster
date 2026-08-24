import { auth } from "@clerk/nextjs/server";

export default async function ConnectionsPage() {
  await auth.protect();

  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold tracking-tight">Connections</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        No services connected yet.
      </p>
    </div>
  );
}
