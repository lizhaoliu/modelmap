import { useEffect, useMemo, useRef, useState } from 'react'
import { fmtParams } from '../fmt'
import { useStore } from '../store'
import type { GNode } from '../types'

/** In-graph module search (§28): `/` opens it, typing matches module paths
 *  and classes, Enter reveals the module — every ancestor opened, the node
 *  selected and framed. Twelve layers is browsable; eighty is not. */
export function NodeFinder() {
  const doc = useStore((s) => s.doc)
  const index = useStore((s) => s.index)
  const reveal = useStore((s) => s.reveal)
  const [open, setOpen] = useState(false)
  const [q, setQ] = useState('')
  const [hi, setHi] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const boxRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      const typing = t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT'
      if (e.key === '/' && !typing && !e.metaKey && !e.ctrlKey) {
        e.preventDefault()
        setOpen(true)
        setTimeout(() => inputRef.current?.select(), 0)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => !boxRef.current?.contains(e.target as Node) && setOpen(false)
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const hits = useMemo(() => {
    if (!doc || !index) return []
    const terms = q.trim().toLowerCase().split(/\s+/).filter(Boolean)
    if (!terms.length) return []
    const score = (n: GNode): number => {
      const id = n.id.toLowerCase(), cls = n.cls.toLowerCase(), leaf = id.split('.').pop() ?? ''
      let s = 0
      for (const t of terms) {
        if (leaf === t) s += 6
        else if (leaf.startsWith(t)) s += 4
        else if (id.includes(t)) s += 2
        else if (cls.includes(t)) s += 1
        else return 0
      }
      return s + (n.depth <= 3 ? 0.5 : 0) // shallow matches first on ties
    }
    return doc.nodes
      .filter((n) => n.id)
      .map((n) => ({ n, s: score(n) }))
      .filter((x) => x.s > 0)
      .sort((a, b) => b.s - a.s || a.n.id.length - b.n.id.length)
      .slice(0, 12)
      .map((x) => x.n)
  }, [doc, index, q])
  useEffect(() => setHi(0), [q])

  if (!doc) return null
  const go = (n: GNode) => {
    reveal(n.id)
    setOpen(false)
    setQ('')
  }
  return (
    <div className={`mm-finder ${open ? 'is-open' : ''}`} ref={boxRef}>
      {!open && (
        <button className="mm-btn mm-finder-btn" onClick={() => { setOpen(true); setTimeout(() => inputRef.current?.focus(), 0) }} title="Find a module in this graph (/)">
          find module <kbd>/</kbd>
        </button>
      )}
      {open && (
        <div className="mm-finder-box" role="combobox" aria-expanded={hits.length > 0} aria-haspopup="listbox">
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="module path or class — q_proj, layers.17, RMSNorm"
            aria-label="Find a module"
            spellCheck={false}
            autoFocus
            onKeyDown={(e) => {
              if (e.key === 'Escape') { setOpen(false); setQ('') }
              else if (e.key === 'ArrowDown') { e.preventDefault(); setHi((h) => Math.min(hits.length - 1, h + 1)) }
              else if (e.key === 'ArrowUp') { e.preventDefault(); setHi((h) => Math.max(0, h - 1)) }
              else if (e.key === 'Enter' && hits[hi]) go(hits[hi])
            }}
          />
          {q.trim() && (
            <ul role="listbox" className="mm-finder-list">
              {hits.length === 0 && <li className="mm-dim">no module matches</li>}
              {hits.map((n, i) => {
                const rep = index?.repeatByRep.get(n.id)
                return (
                  <li key={n.id} role="option" aria-selected={i === hi} className={i === hi ? 'is-hi' : ''} onMouseEnter={() => setHi(i)} onMouseDown={(e) => { e.preventDefault(); go(n) }}>
                    <span className={`mm-finder-kind kind-${n.kind}`} />
                    <span className="mm-finder-id">{n.id}{rep ? <span className="mm-dim"> ×{rep.count}</span> : null}</span>
                    <span className="mm-finder-meta">{n.cls}{n.params ? ` · ${fmtParams(n.params)}` : ''}</span>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
