import { useEffect, useRef, useState } from 'react'
import { searchModels, type SearchHit } from '../api'
import { fmtCount } from '../fmt'
import { useStore } from '../store'

export function ModelSearch({ big = false }: { big?: boolean }) {
  const loadModel = useStore((s) => s.loadModel)
  const [q, setQ] = useState('')
  const [hits, setHits] = useState<SearchHit[]>([])
  const [open, setOpen] = useState(false)
  const [hi, setHi] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const boxRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (q.trim().length < 2) {
      setHits([])
      return
    }
    const t = setTimeout(async () => {
      const res = await searchModels(q.trim())
      setHits(res)
      setHi(0)
      setOpen(true)
    }, 250)
    return () => clearTimeout(t)
  }, [q])

  // ⌘K / Ctrl-K focuses whichever search is mounted
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        inputRef.current?.focus()
        inputRef.current?.select()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as globalThis.Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [])

  const pick = (id: string) => {
    setOpen(false)
    setQ('')
    inputRef.current?.blur()
    void loadModel(id)
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHi((h) => Math.min(h + 1, hits.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHi((h) => Math.max(h - 1, 0))
    } else if (e.key === 'Enter') {
      const target = hits[hi]?.id ?? (q.includes('/') ? q.trim() : null)
      if (target) pick(target)
    } else if (e.key === 'Escape') {
      setOpen(false)
      inputRef.current?.blur()
    }
  }

  return (
    <div className={`mm-search ${big ? 'is-big' : ''}`} ref={boxRef}>
      <input
        ref={inputRef}
        value={q}
        placeholder={big ? 'Search Hugging Face models — try "qwen3"' : 'Search models'}
        spellCheck={false}
        onChange={(e) => setQ(e.target.value)}
        onFocus={() => hits.length && setOpen(true)}
        onKeyDown={onKeyDown}
        aria-label="Search Hugging Face models"
      />
      {!big && <kbd className="mm-kbd-hint">⌘K</kbd>}
      {open && hits.length > 0 && (
        <ul className="mm-search-results" role="listbox">
          {hits.map((h, i) => (
            <li
              key={h.id}
              role="option"
              aria-selected={i === hi}
              className={i === hi ? 'is-hi' : ''}
              onMouseEnter={() => setHi(i)}
              onMouseDown={(e) => {
                e.preventDefault()
                pick(h.id)
              }}
            >
              <span className="mm-hit-id">{h.id}</span>
              <span className="mm-hit-meta">
                {h.pipeline_tag ?? ''}
                {h.downloads != null && ` · ↓${fmtCount(h.downloads)}`}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
