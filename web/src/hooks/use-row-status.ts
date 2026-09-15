import { useCallback, useState } from "react"

export interface RowStatus {
  pending: boolean
  error?: string
}

// The per-row {pending, error?} status map shared by app-sidebar.tsx's delete,
// watchlist-view.tsx's remove/judge, streaming-picker.tsx's subscribe toggle
// and chat-panel.tsx's verdict write. K is string or number so
// streaming-picker's numeric provider_id keys fit alongside the others'.
export function useRowStatus<K extends string | number>() {
  const [status, setStatus] = useState<Record<K, RowStatus>>(
    {} as Record<K, RowStatus>,
  )

  const setPending = useCallback((key: K) => {
    setStatus((prev) => ({ ...prev, [key]: { pending: true } }))
  }, [])

  const setSuccess = useCallback((key: K) => {
    setStatus((prev) => ({ ...prev, [key]: { pending: false } }))
  }, [])

  const setFailure = useCallback((key: K, error: string) => {
    setStatus((prev) => ({ ...prev, [key]: { pending: false, error } }))
  }, [])

  return { status, setPending, setSuccess, setFailure }
}
