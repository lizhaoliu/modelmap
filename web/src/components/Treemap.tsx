import { useMemo } from 'react'
import { fmtLens, lensValue } from '../analytics/cost'
import { useCostStore } from '../analytics/costStore'
import { fmtParams, leafName } from '../fmt'
import { useStore } from '../store'
import type { GNode } from '../types'

interface Cell {
  node: GNode
  x: number
  y: number
  w: number
  h: number
}

/** Squarified treemap (Bruls et al.) — good aspect ratios, deterministic. */
function squarify(items: { node: GNode; v: number }[], x: number, y: number, w: number, h: number): Cell[] {
  const out: Cell[] = []
  let rest = items.filter((i) => i.v > 0)
  const total = rest.reduce((a, b) => a + b.v, 0)
  if (!total) return out
  const scale = (w * h) / total
  rest = rest.map((i) => ({ ...i, v: i.v * scale }))
  let rx = x, ry = y, rw = w, rh = h
  while (rest.length) {
    const vertical = rw >= rh
    const side = vertical ? rh : rw
    let row: typeof rest = []
    let worst = Infinity
    let i = 0
    for (; i < rest.length; i++) {
      const trial = [...row, rest[i]]
      const s = trial.reduce((a, b) => a + b.v, 0)
      const wst = Math.max(...trial.map((t) => Math.max((side * side * t.v) / (s * s), (s * s) / (side * side * t.v))))
      if (wst > worst) break
      row = trial
      worst = wst
    }
    const s = row.reduce((a, b) => a + b.v, 0)
    const thick = s / side
    let off = 0
    for (const r of row) {
      const len = r.v / thick
      out.push(
        vertical
          ? { node: r.node, x: rx, y: ry + off, w: thick, h: len }
          : { node: r.node, x: rx + off, y: ry, w: len, h: thick },
      )
      off += len
    }
    if (vertical) { rx += thick; rw -= thick } else { ry += thick; rh -= thick }
    rest = rest.slice(row.length)
  }
  return out
}

const W = 264
const H = 132

/** Parameter share of a container's children, clickable.
 *  With `overview`, a dominant container child (≥ 60% of params — typically
 *  the base model under a task head) is replaced by its own children, so the
 *  root view reads embed / layers ×N / norm / lm_head instead of model / head. */
export function Treemap({ parent, overview = false }: { parent: string; overview?: boolean }) {
  const index = useStore((s) => s.index)
  const doc = useStore((s) => s.doc)
  const select = useStore((s) => s.select)
  const lens = useCostStore((s) => s.lens)
  const report = useCostStore((s) => s.report)
  const assumptions = useCostStore((s) => s.assumptions)
  const metric = lens === 'none' ? 'params' : lens
  const valueOf = (k: GNode) => lensValue(metric, k, report?.byNode.get(k.id), assumptions)
  const cells = useMemo(() => {
    if (!index) return []
    let kids = (index.children.get(parent) ?? []).filter((k) => valueOf(k) > 0)
    if (overview) {
      const total = kids.reduce((a, k) => a + k.params, 0)
      const dominant = kids.find(
        (k) => k.params >= 0.6 * total && (index.children.get(k.id) ?? []).length > 1,
      )
      if (dominant) {
        kids = kids.flatMap((k) =>
          k.id === dominant.id ? (index.children.get(k.id) ?? []).filter((c) => c.params > 0) : [k],
        )
      }
    }
    const items = kids
      .map((k) => ({ node: k, v: valueOf(k) * (index.repeatByRep.get(k.id)?.count ?? 1) }))
      .sort((a, b) => b.v - a.v)
    return squarify(items, 0, 0, W, H)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, parent, overview, metric, report])
  if (!index || !doc || cells.length < 2) return null
  return (
    <figure className="mm-treemap">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="parameter share of children">
        {cells.map((c) => {
          const rep = index.repeatByRep.get(c.node.id)
          const label = leafName(c.node.id) + (rep ? ` ×${rep.count}` : '')
          const big = c.w > 46 && c.h > 26
          return (
            <g
              key={c.node.id}
              className={`mm-tm-cell kind-${c.node.kind}`}
              onClick={() => select(c.node.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && select(c.node.id)}
            >
              <title>{`${c.node.id} · ${fmtLens(metric, valueOf(c.node))}${rep ? ` × ${rep.count}` : ''}`}</title>
              <rect x={c.x + 0.6} y={c.y + 0.6} width={Math.max(0, c.w - 1.2)} height={Math.max(0, c.h - 1.2)} rx="2.5" />
              {big && (
                <>
                  <text x={c.x + 6} y={c.y + 13} className="mm-tm-name">{label}</text>
                  <text x={c.x + 6} y={c.y + 24} className="mm-tm-val">{metric === 'params' ? fmtParams(c.node.params * (rep?.count ?? 1)) : fmtLens(metric, valueOf(c.node) * (rep?.count ?? 1))}</text>
                </>
              )}
            </g>
          )
        })}
      </svg>
      <figcaption>{metric === 'params' ? 'parameters' : metric === 'compute' ? 'compute (MACs)' : metric === 'memory' ? 'activation memory' : metric === 'vram' ? 'GPU memory (weights + KV)' : 'KV cache'} by child · click to select</figcaption>
    </figure>
  )
}
