import { useEffect, useState } from 'react'

export const MOBILE_QUERY = '(max-width: 820px)'

/** True on narrow screens; the layout switches to a single column with the
 *  inspector as a bottom sheet. */
export function useIsMobile(): boolean {
  const [m, setM] = useState(() => typeof window !== 'undefined' && window.matchMedia(MOBILE_QUERY).matches)
  useEffect(() => {
    const mq = window.matchMedia(MOBILE_QUERY)
    const on = () => setM(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return m
}
