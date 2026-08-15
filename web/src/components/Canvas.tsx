import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ViewportPortal,
  useReactFlow,
  type Edge,
} from '@xyflow/react'
import { buildFlowScript } from '../flow/beats'
import { useFlowEngine } from '../flow/engine'
import { useFlowStore } from '../flow/flowStore'
import { layoutGraph, type MMNode, type Rect } from '../graph/layout'
import { nodeTypes } from '../graph/nodes'
import { useStore } from '../store'
import type { Kind } from '../types'
import { Breadcrumb } from './Breadcrumb'
import { FlowBar } from './FlowBar'

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

interface View {
  nodes: MMNode[]
  edges: Edge[]
  positions: Record<string, Rect>
}

export function Canvas() {
  const doc = useStore((s) => s.doc)
  const index = useStore((s) => s.index)
  const expanded = useStore((s) => s.expanded)
  const selected = useStore((s) => s.selected)
  const select = useStore((s) => s.select)
  const expand = useStore((s) => s.expand)
  const collapse = useStore((s) => s.collapse)
  const lastExpanded = useStore((s) => s.lastExpanded)
  const clearLastExpanded = useStore((s) => s.clearLastExpanded)
  const { fitView, fitBounds } = useReactFlow()

  const [view, setView] = useState<View>({ nodes: [], edges: [], positions: {} })
  const lastModel = useRef<string | null>(null)
  const pulseRef = useRef<HTMLDivElement | null>(null)

  const flowActive = useFlowStore((s) => s.active)
  const script = useMemo(
    () => (doc && index ? buildFlowScript(doc, index, expanded) : { beats: [], total: 0 }),
    [doc, index, expanded],
  )
  const api = useFlowEngine(script, view.positions, pulseRef)

  useEffect(() => {
    if (!doc || !index) return
    let alive = true
    layoutGraph(doc, index, expanded).then((v) => {
      if (!alive) return
      setView(v)
      if (lastModel.current !== doc.model_id) {
        lastModel.current = doc.model_id
        requestAnimationFrame(() => fitView({ padding: 0.12 }))
      } else if (lastExpanded && v.positions[lastExpanded]) {
        // frame the container that just opened so its extent is unmistakable
        const r = v.positions[lastExpanded]
        requestAnimationFrame(() => {
          fitBounds({ x: r.x, y: r.y, width: r.w, height: r.h }, { padding: 0.25, duration: 350 })
          clearLastExpanded()
        })
      }
    })
    return () => {
      alive = false
    }
  }, [doc, index, expanded, fitView, fitBounds, lastExpanded, clearLastExpanded])

  const nodes = useMemo(() => {
    // ancestors of the selection get a strengthened border — the trail
    // that pairs with the breadcrumb
    const ancestors = new Set<string>()
    if (selected) {
      const parts = selected.split('.')
      for (let i = 1; i < parts.length; i++) ancestors.add(parts.slice(0, i).join('.'))
    }
    return view.nodes.map((n) => ({
      ...n,
      selected: n.id === selected,
      className: ancestors.has(n.id) ? 'mm-ancestor' : undefined,
    }))
  }, [view.nodes, selected])

  // keyboard: E expand · C collapse · 0 fit · F flow · Space play/pause ·
  // ←/→ step beats · Esc exit flow / deselect (design doc §07–08)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLInputElement
      const isRange = t.tagName === 'INPUT' && t.type === 'range'
      const isText = (t.tagName === 'INPUT' && !isRange) || t.tagName === 'TEXTAREA'
      if (isText || e.metaKey || e.ctrlKey) return
      // a focused scrubber keeps native ←/→ fine-stepping
      if (isRange && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) return
      const flow = useFlowStore.getState()
      if (e.key === 'Escape') {
        if (flow.active) flow.deactivate()
        else select(null)
      } else if (e.key === '0') fitView({ padding: 0.12, duration: 250 })
      else if (e.key === 'f' && doc?.trace.length) {
        flow.active ? flow.deactivate() : flow.activate()
      } else if (e.key === ' ' && flow.active) {
        e.preventDefault()
        flow.playing ? flow.pause() : flow.play()
      } else if (e.key === 'ArrowRight' && flow.active) api.stepBeat(1)
      else if (e.key === 'ArrowLeft' && flow.active) api.stepBeat(-1)
      else if (e.key === 'e' && selected) expand(selected)
      else if (e.key === 'c' && selected) collapse(selected)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selected, select, expand, collapse, fitView, doc, api])

  return (
    <div className="mm-canvas">
      <ReactFlow
        nodes={nodes}
        edges={view.edges}
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
        {flowActive && (
          <ViewportPortal>
            <div className="mm-flow-pulse" ref={pulseRef} aria-hidden="true" />
          </ViewportPortal>
        )}
      </ReactFlow>
      <Breadcrumb />
      <FlowBar script={script} api={api} />
    </div>
  )
}
