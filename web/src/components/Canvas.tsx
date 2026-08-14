import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  useReactFlow,
  type Edge,
} from '@xyflow/react'
import { layoutGraph, type MMNode } from '../graph/layout'
import { nodeTypes } from '../graph/nodes'
import { useStore } from '../store'
import type { Kind } from '../types'

const KIND_HEX: Record<Kind, string> = {
  embedding: '#4E63C8',
  head: '#4E63C8',
  attention: '#C6537E',
  mlp: '#2E8F84',
  moe: '#2E8F84',
  norm: '#8593A6',
  linear: '#7C8798',
  conv: '#7C8798',
  container: '#AAB4C2',
  module: '#AAB4C2',
}

export function Canvas() {
  const doc = useStore((s) => s.doc)
  const index = useStore((s) => s.index)
  const expanded = useStore((s) => s.expanded)
  const selected = useStore((s) => s.selected)
  const select = useStore((s) => s.select)
  const expand = useStore((s) => s.expand)
  const collapse = useStore((s) => s.collapse)
  const { fitView } = useReactFlow()

  const [flow, setFlow] = useState<{ nodes: MMNode[]; edges: Edge[] }>({ nodes: [], edges: [] })
  const lastModel = useRef<string | null>(null)

  useEffect(() => {
    if (!doc || !index) return
    let alive = true
    layoutGraph(doc, index, expanded).then((f) => {
      if (!alive) return
      setFlow(f)
      if (lastModel.current !== doc.model_id) {
        lastModel.current = doc.model_id
        requestAnimationFrame(() => fitView({ padding: 0.12 }))
      }
    })
    return () => {
      alive = false
    }
  }, [doc, index, expanded, fitView])

  const nodes = useMemo(
    () => flow.nodes.map((n) => ({ ...n, selected: n.id === selected })),
    [flow.nodes, selected],
  )

  // keyboard: E expand · C collapse · 0 fit · Esc deselect (design doc §07)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || e.metaKey || e.ctrlKey) return
      if (e.key === 'Escape') select(null)
      else if (e.key === '0') fitView({ padding: 0.12, duration: 250 })
      else if (e.key === 'e' && selected) expand(selected)
      else if (e.key === 'c' && selected) collapse(selected)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selected, select, expand, collapse, fitView])

  return (
    <div className="mm-canvas">
      <ReactFlow
        nodes={nodes}
        edges={flow.edges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, n) => select(n.id)}
        onPaneClick={() => select(null)}
        minZoom={0.08}
        maxZoom={2.5}
        fitView
        proOptions={{ hideAttribution: false }}
        nodesConnectable={false}
        nodesDraggable={false}
        elementsSelectable
      >
        <Background variant={BackgroundVariant.Dots} gap={26} size={1.2} className="mm-bg" />
        <MiniMap
          pannable
          zoomable
          className="mm-minimap"
          nodeColor={(n) => {
            const data = n.data as Partial<MMNode['data']> | undefined
            return (data?.g && KIND_HEX[data.g.kind]) || '#AAB4C2'
          }}
          nodeStrokeWidth={0}
        />
        <Controls showInteractive={false} className="mm-controls" />
      </ReactFlow>
    </div>
  )
}
