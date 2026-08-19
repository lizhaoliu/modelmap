import { useEffect, useRef, useState } from 'react'
import { useCostStore } from '../analytics/costStore'
import { getToken } from '../api'
import { downloadBlob, fileStem, renderSvg, svgToPng } from '../export/svg'
import { useViewStore } from '../graph/viewStore'
import { useStore } from '../store'

/** "export ▾": take the map with you — images of the current view, the data
 *  behind it in formats other tools read, and links / embed code that
 *  reproduce this exact view (design doc §16). */
export function ExportMenu() {
  const doc = useStore((s) => s.doc)
  const selected = useStore((s) => s.selected)
  const setToast = useStore((s) => s.setToast)
  const lens = useCostStore((s) => s.lens)
  const report = useCostStore((s) => s.report)
  const a = useCostStore((s) => s.assumptions)
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const ref = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => !ref.current?.contains(e.target as globalThis.Node) && setOpen(false)
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', onDown)
    window.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onDown); window.removeEventListener('keydown', onKey) }
  }, [open])
  if (!doc) return null
  const stem = fileStem(doc.model_id)
  const theme = (document.documentElement.dataset.theme as 'light' | 'dark' | undefined)
    ?? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')

  const svg = () => {
    const view = useViewStore.getState()
    return renderSvg(doc, view, {
      theme, title: `${doc.model_id}${doc.variant ? ` · ${doc.variant}` : ''} — ${doc.architecture ?? 'architecture'}`,
      lens, report, selected,
    })
  }
  const run = async (label: string, fn: () => Promise<void>) => {
    setBusy(label)
    try {
      await fn()
      setOpen(false)
    } catch (e) {
      setToast(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }
  const saveSvg = () => run('svg', async () => {
    const s = svg()
    if (!s) throw new Error('nothing to export yet')
    downloadBlob(`${stem}.svg`, new Blob([s], { type: 'image/svg+xml;charset=utf-8' }))
    setToast('SVG saved — the current view, at any zoom')
  })
  const savePng = () => run('png', async () => {
    const s = svg()
    if (!s) throw new Error('nothing to export yet')
    downloadBlob(`${stem}.png`, await svgToPng(s, 2))
    setToast('PNG saved (2×)')
  })
  const saveJson = () => run('json', async () => {
    downloadBlob(`${stem}.graph.json`, new Blob([JSON.stringify(doc)], { type: 'application/json' }))
    setToast('Graph document saved')
  })
  const apiUrl = (format: string) =>
    `/api/export/${doc.model_id}?format=${format}&T=${a.T}&B=${a.B}&dtype=${encodeURIComponent(a.dtypeLabel)}`
  const saveServer = (format: 'csv' | 'md' | 'dot', ext: string, what: string) => run(format, async () => {
    const headers: Record<string, string> = {}
    const tok = getToken()
    if (tok) headers['X-HF-Token'] = tok
    const res = await fetch(apiUrl(format) + '&download=1', { headers })
    if (!res.ok) throw new Error(`export failed (${res.status})`)
    downloadBlob(`${stem}.${ext}`, await res.blob())
    setToast(`${what} saved`)
  })
  const copy = (label: string, text: string) => run(label, async () => {
    await navigator.clipboard.writeText(text)
    setToast(`${label} copied`)
  })
  const viewUrl = location.href
  const embedUrl = (() => {
    const u = new URL(location.href)
    u.searchParams.set('embed', '1')
    return u.toString()
  })()
  const iframe = `<iframe src="${embedUrl}" width="100%" height="520" style="border:1px solid #d9dee6;border-radius:10px" loading="lazy" title="${doc.model_id} — modelmap"></iframe>`

  return (
    <span className="mm-topbar-rel" ref={ref}>
      <button className={`mm-btn mm-btn-export ${open ? 'is-on' : ''}`} onClick={() => setOpen((v) => !v)} aria-expanded={open} title="Download, copy or embed this view">
        export ▾
      </button>
      {open && (
        <div className="mm-pop mm-export-pop" role="menu" aria-label="Export">
          <div className="mm-export-grp">image of this view</div>
          <button role="menuitem" onClick={savePng} disabled={busy != null}>PNG <span className="mm-dim">2×, current expansion</span></button>
          <button role="menuitem" onClick={saveSvg} disabled={busy != null}>SVG <span className="mm-dim">vector, editable</span></button>
          <div className="mm-export-grp">data</div>
          <button role="menuitem" onClick={() => saveServer('md', 'md', 'Markdown summary')} disabled={busy != null}>Markdown summary <span className="mm-dim">config · cost · top weights</span></button>
          <button role="menuitem" onClick={() => saveServer('csv', 'csv', 'CSV')} disabled={busy != null}>CSV module table <span className="mm-dim">params · shapes · cost per module</span></button>
          <button role="menuitem" onClick={saveJson} disabled={busy != null}>JSON document <span className="mm-dim">what this page renders</span></button>
          <button role="menuitem" onClick={() => saveServer('dot', 'dot', 'Graphviz DOT')} disabled={busy != null}>Graphviz DOT</button>
          <div className="mm-export-grp">share</div>
          <button role="menuitem" onClick={() => copy('Link', viewUrl)}>Copy link <span className="mm-dim">reproduces this exact view</span></button>
          <button role="menuitem" onClick={() => copy('Embed code', iframe)}>Copy embed code <span className="mm-dim">&lt;iframe&gt; for model cards, blogs, docs</span></button>
          <button role="menuitem" onClick={() => copy('API URL', location.origin + apiUrl('csv').replace('format=csv', 'format=json'))}>Copy API URL <span className="mm-dim">/api/export · /api/summary · /docs</span></button>
          {busy && <div className="mm-pop-note">working…</div>}
          {!busy && <div className="mm-pop-note">CLI: <code>uvx modelmap dump {doc.model_id} -f md</code> · MCP: <code>uvx --from 'modelmap[mcp]' modelmap mcp</code></div>}
        </div>
      )}
    </span>
  )
}
