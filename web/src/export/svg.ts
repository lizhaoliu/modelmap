/**
 * Static renderings of the current view (design doc §16): an SVG built from
 * the same geometry the canvas uses (ELK positions, fixed node sizes, the
 * bezier edges of graph/edges.tsx) — so what you download is what you saw,
 * at any expansion state, with no dependence on DOM measurement or fonts
 * being embeddable. PNG is the SVG rasterized at 2× in an offscreen canvas.
 */
import type { Edge } from '@xyflow/react'
import { fmtLens, lensValue, type CostReport, type Lens } from '../analytics/cost'
import { fmtParams, leafName } from '../fmt'
import type { MMNode, Rect } from '../graph/layout'
import type { GraphDoc, Kind } from '../types'

interface Palette {
  paper: string; card: string; ink: string; muted: string; line: string; flow: string
  kind: Record<Kind, string>
}
const LIGHT: Palette = {
  paper: '#F7F8FA', card: '#FFFFFF', ink: '#1B2430', muted: '#5D6878', line: '#D9DEE6', flow: '#E08A00',
  kind: { embedding: '#4E63C8', head: '#4E63C8', attention: '#C6537E', mlp: '#2E8F84', moe: '#2E8F84', norm: '#8593A6', linear: '#7C8798', conv: '#7C8798', container: '#AAB4C2', module: '#AAB4C2' },
}
const DARK: Palette = {
  paper: '#12161D', card: '#1A2029', ink: '#E8ECF2', muted: '#97A2B3', line: '#2A3340', flow: '#FFB454',
  kind: { embedding: '#8194E8', head: '#8194E8', attention: '#E27EA2', mlp: '#4FB3A7', moe: '#4FB3A7', norm: '#7E8CA0', linear: '#93A0B2', conv: '#93A0B2', container: '#AAB4C2', module: '#AAB4C2' },
}

export interface SvgOptions {
  theme: 'light' | 'dark'
  title?: string
  lens?: Lens
  report?: CostReport | null
  selected?: string | null
  /** show the "modelmap.cc/m/…" footer */
  credit?: boolean
}

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
const PAD = 28
const FOOT = 26

function wash(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16)
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`
}

/** Cubic bezier between node sides — same endpoints as graph/edges.tsx */
function edgePath(s: Rect, t: Rect, dir: 'h' | 'v'): string {
  if (dir === 'v') {
    const x1 = s.x + s.w / 2, y1 = s.y + s.h, x2 = t.x + t.w / 2, y2 = t.y
    const c = Math.max(18, Math.abs(y2 - y1) / 2)
    return `M${x1},${y1} C${x1},${y1 + c} ${x2},${y2 - c} ${x2},${y2}`
  }
  const x1 = s.x + s.w, y1 = s.y + s.h / 2, x2 = t.x, y2 = t.y + t.h / 2
  const c = Math.max(18, Math.abs(x2 - x1) / 2)
  return `M${x1},${y1} C${x1 + c},${y1} ${x2 - c},${y2} ${x2},${y2}`
}

export function renderSvg(
  doc: GraphDoc,
  view: { nodes: MMNode[]; edges: Edge[]; positions: Record<string, Rect> },
  opts: SvgOptions,
): string {
  const P = opts.theme === 'dark' ? DARK : LIGHT
  const rects = Object.values(view.positions)
  if (!rects.length) return ''
  const minX = Math.min(...rects.map((r) => r.x)), minY = Math.min(...rects.map((r) => r.y))
  const maxX = Math.max(...rects.map((r) => r.x + r.w)), maxY = Math.max(...rects.map((r) => r.y + r.h))
  const W = maxX - minX + PAD * 2, H = maxY - minY + PAD * 2 + (opts.title ? 30 : 0) + (opts.credit === false ? 0 : FOOT)
  const top = PAD + (opts.title ? 30 : 0)
  const ox = PAD - minX, oy = top - minY
  const lens = opts.lens ?? 'none'
  const total = doc.params_total
  const out: string[] = []
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="IBM Plex Sans, Inter, system-ui, sans-serif">`)
  out.push(`<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${P.line === '#2A3340' ? '#8A94A4' : '#8A94A4'}"/></marker></defs>`)
  out.push(`<rect width="100%" height="100%" fill="${P.paper}"/>`)
  if (opts.title) {
    out.push(`<text x="${PAD}" y="${PAD + 4}" font-size="15" font-weight="600" fill="${P.ink}">${esc(opts.title)}</text>`)
  }
  // containers first (back), then leaves, then edges (front, like the canvas)
  const byId = new Map(view.nodes.map((n) => [n.id, n]))
  const containers = view.nodes.filter((n) => n.type === 'containerNode').sort((a, b) => a.data.g.depth - b.data.g.depth)
  const leaves = view.nodes.filter((n) => n.type !== 'containerNode')
  const badge = (n: MMNode): string | null => {
    const g = n.data.g
    if (lens === 'none') return g.params > 0 ? fmtParams(g.params) : null
    const v = lensValue(lens, g, opts.report?.byNode.get(g.id))
    return v > 0 ? fmtLens(lens, v) : null
  }
  const heatOf = (n: MMNode): number => {
    if (lens === 'none' || !opts.report) return 0
    const g = n.data.g
    const v = lensValue(lens, g, opts.report.byNode.get(g.id))
    const rootV = lens === 'params' ? total : lensValue(lens, g, opts.report.root)
    const share = rootV > 0 ? Math.min(1, v / rootV) : 0
    return v > 0 ? Math.min(0.85, 0.06 + 0.8 * Math.sqrt(share)) : 0
  }
  for (const n of containers) {
    const r = view.positions[n.id]
    const k = P.kind[n.data.g.kind]
    const heat = heatOf(n)
    const depthWash = n.data.g.depth % 2 ? 0.06 : 0.03
    out.push(`<g>`)
    out.push(`<rect x="${r.x + ox}" y="${r.y + oy}" width="${r.w}" height="${r.h}" rx="10" fill="${wash(k, depthWash + heat * 0.18)}" stroke="${n.id === opts.selected ? P.flow : wash(k, 0.7)}" stroke-width="${n.id === opts.selected ? 2 : 1.2}"/>`)
    const b = badge(n)
    const rep = n.data.repeat, stack = n.data.stackOf
    const tag = rep ? `1 of ${rep.count}` : stack ? `×${n.data.stackTotal ?? stack.count}` : ''
    out.push(`<text x="${r.x + ox + 14}" y="${r.y + oy + 22}" font-size="12.5" font-weight="600" fill="${P.ink}">${esc(leafName(n.id))}<tspan fill="${P.muted}" font-weight="400"> ${esc(n.data.g.cls === '?' ? '' : n.data.g.cls)}</tspan>${tag ? `<tspan fill="${P.muted}" font-weight="500" font-size="11"> ${esc(tag)}</tspan>` : ''}</text>`)
    if (b) out.push(`<text x="${r.x + ox + r.w - 12}" y="${r.y + oy + 22}" text-anchor="end" font-size="11" font-family="IBM Plex Mono, monospace" fill="${P.muted}">${esc(b)}</text>`)
    out.push(`</g>`)
  }
  for (const n of leaves) {
    const r = view.positions[n.id]
    const g = n.data.g
    const k = P.kind[g.kind]
    const heat = heatOf(n)
    const sel = n.id === opts.selected
    const stack = n.data.stackOf, rep = n.data.repeat
    out.push(`<g>`)
    if (stack) {
      // the stack's "deck" behind the card
      out.push(`<rect x="${r.x + ox + 6}" y="${r.y + oy + 6}" width="${r.w - 4}" height="${r.h - 4}" rx="8" fill="${P.card}" stroke="${wash(k, 0.5)}"/>`)
      out.push(`<rect x="${r.x + ox + 3}" y="${r.y + oy + 3}" width="${r.w - 4}" height="${r.h - 4}" rx="8" fill="${P.card}" stroke="${wash(k, 0.6)}"/>`)
    }
    out.push(`<rect x="${r.x + ox}" y="${r.y + oy}" width="${r.w - (stack ? 4 : 0)}" height="${r.h - (stack ? 4 : 0)}" rx="8" fill="${heat ? wash(k, heat * 0.35) : P.card}" stroke="${sel ? P.flow : wash(k, 0.85)}" stroke-width="${sel ? 2.2 : 1.3}"/>`)
    out.push(`<rect x="${r.x + ox}" y="${r.y + oy + 8}" width="3" height="${r.h - 16 - (stack ? 4 : 0)}" fill="${k}"/>`)
    const name = leafName(n.id)
    const tag = stack ? `×${n.data.stackTotal ?? stack.count}` : rep ? `1 of ${rep.count}` : n.data.hasChildren ? '+' : ''
    out.push(`<text x="${r.x + ox + 12}" y="${r.y + oy + 21}" font-size="12.5" font-weight="600" fill="${P.ink}">${esc(name)}${tag ? `<tspan fill="${stack ? P.flow : P.muted}" font-weight="600" font-size="11"> ${esc(tag)}</tspan>` : ''}</text>`)
    const cls = g.cls === '?' ? g.kind : g.cls
    const b = badge(n)
    out.push(`<text x="${r.x + ox + 12}" y="${r.y + oy + 40}" font-size="10.5" fill="${P.muted}">${esc(cls.length > 22 ? cls.slice(0, 21) + '…' : cls)}</text>`)
    if (b) out.push(`<text x="${r.x + ox + r.w - 10 - (stack ? 4 : 0)}" y="${r.y + oy + 40}" text-anchor="end" font-size="10.5" font-family="IBM Plex Mono, monospace" fill="${P.muted}">${esc(b)}</text>`)
    out.push(`</g>`)
  }
  for (const e of view.edges) {
    const s = byId.get(e.source), t = byId.get(e.target)
    const sr = view.positions[e.source], tr = view.positions[e.target]
    if (!s || !t || !sr || !tr) continue
    const aux = Boolean((e.data as { aux?: boolean } | undefined)?.aux)
    const d = edgePath({ ...sr, x: sr.x + ox, y: sr.y + oy }, { ...tr, x: tr.x + ox, y: tr.y + oy }, s.data.dir)
    out.push(`<path d="${d}" fill="none" stroke="#8A94A4" stroke-width="${aux ? 1.1 : 1.3}" ${aux ? 'stroke-dasharray="4 4" opacity="0.8"' : ''} marker-end="url(#ah)"/>`)
  }
  if (opts.credit !== false) {
    out.push(`<text x="${W - PAD}" y="${H - 10}" text-anchor="end" font-size="10.5" fill="${P.muted}">modelmap.cc/m/${esc(doc.model_id)} · ${esc(doc.architecture ?? '')} · ${fmtParams(doc.params_total)} params${lens !== 'none' ? ` · ${lens} lens` : ''}</text>`)
  }
  out.push('</svg>')
  return out.join('\n')
}

export async function svgToPng(svg: string, scale = 2): Promise<Blob> {
  const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  try {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image()
      i.onload = () => res(i)
      i.onerror = () => rej(new Error('could not rasterize the SVG'))
      i.src = url
    })
    const m = svg.match(/width="(\d+(?:\.\d+)?)" height="(\d+(?:\.\d+)?)"/)
    const w = m ? parseFloat(m[1]) : img.width, h = m ? parseFloat(m[2]) : img.height
    const canvas = document.createElement('canvas')
    // cap at ~16 Mpx so giant expansions still rasterize
    const s = Math.min(scale, Math.sqrt(16e6 / (w * h)))
    canvas.width = Math.round(w * s)
    canvas.height = Math.round(h * s)
    const ctx = canvas.getContext('2d')!
    ctx.scale(s, s)
    ctx.drawImage(img, 0, 0, w, h)
    return await new Promise<Blob>((res, rej) => canvas.toBlob((b) => (b ? res(b) : rej(new Error('PNG encode failed'))), 'image/png'))
  } finally {
    URL.revokeObjectURL(url)
  }
}

export function downloadBlob(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function fileStem(modelId: string): string {
  return modelId.replace(/^local:/, '').replace(/[/:]/g, '--').replace(/[^\w.\-]+/g, '_')
}
