import * as React from "react"

const MOBILE_BREAKPOINT = 768
const MOBILE_QUERY = `(max-width: ${MOBILE_BREAKPOINT - 1}px)`

let mediaQueryList: MediaQueryList | undefined

// Created on first use — window doesn't exist when this module is imported.
function getMediaQueryList() {
  return (mediaQueryList ??= window.matchMedia(MOBILE_QUERY))
}

function subscribe(onStoreChange: () => void) {
  const mql = getMediaQueryList()
  mql.addEventListener("change", onStoreChange)
  return () => mql.removeEventListener("change", onStoreChange)
}

export function useIsMobile() {
  return React.useSyncExternalStore(
    subscribe,
    () => getMediaQueryList().matches,
    // No viewport on the server — render the desktop sidebar.
    () => false
  )
}
