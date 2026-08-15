import { Handle, Position, type NodeProps } from '@xyflow/react'
import { fmtParams, leafName } from '../fmt'
import { useStore } from '../store'
import type { MMNode } from './layout'

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
  const cls = [
    'mm-node',
    `kind-${g.kind}`,
    selected ? 'is-selected' : '',
    stackOf ? 'is-stack' : '',
    hasChildren ? 'is-openable' : '',
  ]
    .filter(Boolean)
    .join(' ')

  // opening a stack goes straight into the representative block
  const open = () => {
    if (stackOf) expandMany([g.id, stackOf.representative])
    else toggleExpand(g.id)
  }

  return (
    <div className={cls} onDoubleClick={hasChildren ? open : undefined}>
      {handles(dir)}
      <div className="mm-row">
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
        {g.params > 0 && <span className="mm-params">{fmtParams(g.params)}</span>}
      </div>
    </div>
  )
}

/** Expanded container: header strip; children are separate flow nodes inside. */
export function ContainerNode({ data, selected }: NodeProps<MMNode>) {
  const toggleExpand = useStore((s) => s.toggleExpand)
  const { g, repeat, stackOf, dir } = data
  const cls = [
    'mm-container',
    `kind-${g.kind}`,
    g.depth % 2 ? 'lvl-odd' : 'lvl-even', // alternating surfaces keep nesting legible
    selected ? 'is-selected' : '',
  ]
    .filter(Boolean)
    .join(' ')
  return (
    <div className={cls}>
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
        {g.params > 0 && <span className="mm-params">{fmtParams(g.params)}</span>}
      </div>
    </div>
  )
}

export const nodeTypes = { moduleNode: ModuleNode, containerNode: ContainerNode }
