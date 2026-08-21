import { Handle, Position, type NodeProps } from '@xyflow/react'
import { fmtLens, lensValue } from '../analytics/cost'
import { useCostStore } from '../analytics/costStore'
import { fmtParams, leafName } from '../fmt'
import { useStore } from '../store'
import { stackTitle, type MMNode } from './layout'
import type { GNode } from '../types'

/** Lens-aware badge text, heat (0–100) and hover detail for a node. */
function useLens(g: GNode): { badge: string | null; heat: number; title?: string } {
  const lens = useCostStore((s) => s.lens)
  const report = useCostStore((s) => s.report)
  const a = useCostStore((s) => s.assumptions)
  const total = useStore((s) => s.doc?.params_total ?? 0)
  if (lens === 'none') return { badge: g.params > 0 ? fmtParams(g.params) : null, heat: 0 }
  const cost = report?.byNode.get(g.id)
  const v = lensValue(lens, g, cost, a)
  const rootV = lens === 'params' ? total : report ? lensValue(lens, g, report.root, a) : 0
  const share = rootV > 0 ? Math.min(1, v / rootV) : 0
  const heat = v > 0 ? Math.min(85, 6 + 80 * Math.sqrt(share)) : 0
  let title: string | undefined
  if (lens === 'vram' && cost && v > 0) {
    const kv = cost.kvPerToken * a.T * a.B
    title = `weights ${fmtLens('vram', cost.paramBytes)}${kv ? ` + KV cache ${fmtLens('vram', kv)} at T ${a.T.toLocaleString()}${a.B > 1 ? ` × B ${a.B}` : ''}` : ''}`
  }
  return { badge: v > 0 ? fmtLens(lens, v) : null, heat, title }
}

function handles(dir: 'h' | 'v') {
  return (
    <>
      <Handle
        type="target"
        position={dir === 'v' ? Position.Top : Position.Left}
        className="mm-handle"
        isConnectable={false}
      />
      <Handle
        type="source"
        position={dir === 'v' ? Position.Bottom : Position.Right}
        className="mm-handle"
        isConnectable={false}
      />
    </>
  )
}

/** Leaf module, or a collapsed container / repeat stack. */
export function ModuleNode({ data, selected }: NodeProps<MMNode>) {
  const toggleExpand = useStore((s) => s.toggleExpand)
  const expandMany = useStore((s) => s.expandMany)
  const setToast = useStore((s) => s.setToast)
  // a leaf has nothing to open: say so instead of silently ignoring the double-click
  const leafHint = () => setToast(`${leafName(g.id)} is a leaf module — nothing inside to open. Its weights and shapes are in the inspector.`)
  const { g, repeat, stackOf, stackTotal, stackRuns, stackLoose, hasChildren, dir } = data
  const { badge, heat, title } = useLens(g)
  const diff = useStore((s) => s.diff?.get(g.id))
  const cls = [
    'mm-node',
    `kind-${g.kind}`,
    selected ? 'is-selected' : '',
    stackOf ? 'is-stack' : '',
    hasChildren ? 'is-openable' : '',
    heat ? 'has-heat' : '',
    diff ? `diff-${diff}` : '',
  ]
    .filter(Boolean)
    .join(' ')

  // opening a stack goes straight into the representative block(s)
  const index = useStore((s) => s.index)
  const open = () => {
    if (stackOf) expandMany([g.id, ...(index?.repeatsByParent.get(g.id) ?? []).map((r) => r.representative)])
    else toggleExpand(g.id)
  }

  return (
    <div className={cls} onDoubleClick={hasChildren ? open : leafHint} style={{ '--heat': heat } as React.CSSProperties}>
      {handles(dir)}
      <div className="mm-row">
        {(diff === 'added' || diff === 'removed') && <span className="mm-diffchip">{diff === 'added' ? '+' : '−'}</span>}
        <span className="mm-name">{leafName(g.id)}</span>
        {stackOf && (
          <button
            className="mm-badge"
            title={`${stackTitle(stackTotal ?? stackOf.count, stackRuns, stackLoose)} — click to open one${stackRuns ? ' of each' : ''}`}
            onClick={(e) => {
              e.stopPropagation()
              open()
            }}
          >
            ×{stackTotal ?? stackOf.count}
          </button>
        )}
        {!stackOf && repeat && (
          <span className="mm-badge is-static" title={`one of ${repeat.count} identical siblings`}>
            1 of {repeat.count}
          </span>
        )}
        {!stackOf && hasChildren && (
          <button
            className="mm-chevron"
            title="Expand"
            onClick={(e) => {
              e.stopPropagation()
              open()
            }}
          >
            +
          </button>
        )}
      </div>
      <div className="mm-row mm-row-sub">
        <span className="mm-cls">{g.cls === '?' ? g.kind : g.cls}</span>
        {badge && <span className="mm-params" title={title}>{badge}</span>}
      </div>
    </div>
  )
}

/** Expanded container: header strip; children are separate flow nodes inside. */
export function ContainerNode({ data, selected }: NodeProps<MMNode>) {
  const toggleExpand = useStore((s) => s.toggleExpand)
  const { g, repeat, stackOf, stackTotal, stackRuns, stackLoose, dir } = data
  const { badge, heat, title } = useLens(g)
  const diff = useStore((s) => s.diff?.get(g.id))
  const cls = [
    'mm-container',
    `kind-${g.kind}`,
    g.depth % 2 ? 'lvl-odd' : 'lvl-even', // alternating surfaces keep nesting legible
    selected ? 'is-selected' : '',
    heat ? 'has-heat' : '',
    diff ? `diff-${diff}` : '',
  ]
    .filter(Boolean)
    .join(' ')
  return (
    <div className={cls} style={{ '--heat': heat } as React.CSSProperties}>
      {handles(dir)}
      <div className="mm-container-head">
        <button
          className="mm-chevron"
          title="Collapse"
          onClick={(e) => {
            e.stopPropagation()
            toggleExpand(g.id)
          }}
        >
          −
        </button>
        {(diff === 'added' || diff === 'removed') && <span className="mm-diffchip">{diff === 'added' ? '+' : '−'}</span>}
        <span className="mm-name">{leafName(g.id)}</span>
        <span className="mm-cls">{g.cls === '?' ? '' : g.cls}</span>
        {repeat && (
          <span className="mm-badge is-static" title={`showing one of ${repeat.count} identical blocks`}>
            1 of {repeat.count}
          </span>
        )}
        {stackOf && (
          <span className="mm-badge is-static" title={`children: ${stackTitle(stackTotal ?? stackOf.count, stackRuns, stackLoose)}`}>
            ×{stackTotal ?? stackOf.count}
          </span>
        )}
        {badge && <span className="mm-params" title={title}>{badge}</span>}
      </div>
    </div>
  )
}

export const nodeTypes = { moduleNode: ModuleNode, containerNode: ContainerNode }
