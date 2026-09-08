import { auth } from "@clerk/nextjs/server"
import Image from "next/image"

export default async function CreditsPage() {
  await auth.protect()

  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold tracking-tight">Credits</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Holster&apos;s catalog and streaming-availability data come from these
        sources.
      </p>
      <div className="mt-6 max-w-md space-y-6">
        <div>
          <a
            href="https://www.themoviedb.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              src="/tmdb-logo.svg"
              alt="The Movie Database"
              width={276}
              height={20}
            />
          </a>
          <p className="mt-2 text-sm text-muted-foreground">
            This product uses the TMDb API but is not endorsed or certified
            by TMDb.
          </p>
        </div>
        <p className="text-sm text-muted-foreground">
          Streaming availability data provided by{" "}
          <a
            href="https://www.justwatch.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary underline underline-offset-4"
          >
            JustWatch
          </a>
          .
        </p>
      </div>
    </div>
  )
}
