import { Handle, Position, type NodeProps } from '@xyflow/react'
import { fmtLens, lensValue } from '../analytics/cost'
import { useCostStore } from '../analytics/costStore'
import { fmtParams, leafName } from '../fmt'
import { useStore } from '../store'
import type { MMNode } from './layout'
import type { GNode } from '../types'

/** Lens-aware badge text and heat (0–100) for a node. */
function useLens(g: GNode): { badge: string | null; heat: number } {
  const lens = useCostStore((s) => s.lens)
  const report = useCostStore((s) => s.report)
  const total = useStore((s) => s.doc?.params_total ?? 0)
  if (lens === 'none') return { badge: g.params > 0 ? fmtParams(g.params) : null, heat: 0 }
  const v = lensValue(lens, g, report?.byNode.get(g.id))
  const rootV = lens === 'params' ? total : report ? lensValue(lens, g, report.root) : 0
  const share = rootV > 0 ? Math.min(1, v / rootV) : 0
  const heat = v > 0 ? Math.min(85, 6 + 80 * Math.sqrt(share)) : 0
  return { badge: v > 0 ? fmtLens(lens, v) : null, heat }
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
  const { g, repeat, stackOf, hasChildren, dir } = data
  const { badge, heat } = useLens(g)
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

  // opening a stack goes straight into the representative block
  const open = () => {
    if (stackOf) expandMany([g.id, stackOf.representative])
    else toggleExpand(g.id)
  }

  return (
    <div className={cls} onDoubleClick={hasChildren ? open : undefined} style={{ '--heat': heat } as React.CSSProperties}>
      {handles(dir)}
      <div className="mm-row">
        {(diff === 'added' || diff === 'removed') && <span className="mm-diffchip">{diff === 'added' ? '+' : '−'}</span>}
        <span className="mm-name">{leafName(g.id)}</span>
        {stackOf && (
          <button
            className="mm-badge"
            title={`${stackOf.count} structurally identical blocks — click to open one`}
            onClick={(e) => {
              e.stopPropagation()
              open()
            }}
          >
            ×{stackOf.count}
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
        {badge && <span className="mm-params">{badge}</span>}
      </div>
    </div>
  )
}

/** Expanded container: header strip; children are separate flow nodes inside. */
export function ContainerNode({ data, selected }: NodeProps<MMNode>) {
  const toggleExpand = useStore((s) => s.toggleExpand)
  const { g, repeat, stackOf, dir } = data
  const { badge, heat } = useLens(g)
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
          <span className="mm-badge is-static" title={`children are ${stackOf.count} identical blocks`}>
            ×{stackOf.count}
          </span>
        )}
        {badge && <span className="mm-params">{badge}</span>}
      </div>
    </div>
  )
}

export const nodeTypes = { moduleNode: ModuleNode, containerNode: ContainerNode }
