import { useEffect, useState, type ReactNode } from 'react'
import { useStore } from '../store'
import { useIsMobile } from '../useMobile'
import { leafName } from '../fmt'

/** On phones the side panel becomes a bottom sheet: collapsed to a handle
 *  by default, opened when a node is selected, toggled by tapping the handle.
 *  On wide screens it renders the children unchanged (the sidebar). */
export function Sheet({ children, title }: { children: ReactNode; title?: string }) {
  const mobile = useIsMobile()
  const selected = useStore((s) => s.selected)
  const [open, setOpen] = useState(false)
  useEffect(() => {
    if (mobile && selected != null) setOpen(true)
  }, [selected, mobile])
  // the sheet takes real layout space: ask canvases to refit after it animates
  useEffect(() => {
    if (!mobile) return
    const t = setTimeout(() => window.dispatchEvent(new Event('mm:refit')), 300)
    return () => clearTimeout(t)
  }, [open, mobile])
  if (!mobile) return <>{children}</>
  const label = title ?? (selected != null ? leafName(selected) : 'details')
  return (
    <div className={`mm-sheet ${open ? 'is-open' : ''}`}>
      <button className="mm-sheet-handle" onClick={() => setOpen((v) => !v)} aria-expanded={open} aria-label="Toggle details panel">
        <span className="mm-sheet-grip" aria-hidden="true" />
        <span className="mm-sheet-title">{label}</span>
        <span className="mm-sheet-chev">{open ? '▾' : '▴'}</span>
      </button>
      <div className="mm-sheet-body">{children}</div>
    </div>
  )
}
