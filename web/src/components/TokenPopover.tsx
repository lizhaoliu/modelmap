import { useEffect, useRef, useState } from 'react'
import { getToken, setToken } from '../api'

/** Optional HF token for gated/private repos. Stored only in this browser,
 *  sent per request as X-HF-Token; the server never persists it and skips
 *  the shared cache for tokened requests. */
export function TokenPopover({ onClose }: { onClose: () => void }) {
  const [val, setVal] = useState(getToken())
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as globalThis.Node)) onClose()
    }
    window.addEventListener('keydown', onKey)
    // deferred so the opening click doesn't immediately close it
    const t = setTimeout(() => document.addEventListener('mousedown', onDown), 0)
    return () => {
      window.removeEventListener('keydown', onKey)
      clearTimeout(t)
      document.removeEventListener('mousedown', onDown)
    }
  }, [onClose])
  return (
    <div className="mm-pop" role="dialog" aria-label="Hugging Face token" ref={ref}>
      <label htmlFor="mm-token">Hugging Face token <span className="mm-dim">(for gated / private repos)</span></label>
      <input
        id="mm-token"
        type="password"
        autoFocus
        value={val}
        placeholder="hf_…"
        spellCheck={false}
        autoComplete="off"
        onChange={(e) => setVal(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') { setToken(val); onClose() }
          if (e.key === 'Escape') onClose()
        }}
      />
      <p className="mm-pop-note">
        Stays in this browser only and is sent with each request; tokened results are never
        cached on the server. Create one at huggingface.co/settings/tokens (read scope).
      </p>
      <div className="mm-pop-actions">
        <button className="mm-btn" onClick={() => { setToken(''); setVal(''); onClose() }}>clear</button>
        <button className="mm-btn is-on" onClick={() => { setToken(val); onClose() }}>save</button>
      </div>
    </div>
  )
}
