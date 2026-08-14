// Mirrors src/modelmap/schema.py (SCHEMA_VERSION 1)

export type Fidelity = 'full' | 'structural' | 'weights'

export type Kind =
  | 'embedding' | 'attention' | 'mlp' | 'moe' | 'norm'
  | 'linear' | 'conv' | 'head' | 'container' | 'module'

export interface GNode {
  id: string
  kind: Kind
  cls: string
  parent: string | null
  depth: number
  order: number
  params: number
  dtype?: string | null
  weight_shapes?: Record<string, number[]> | null
}

export interface Repeat {
  parent: string
  representative: string
  count: number
  signature: string
  members: string[]
}

export interface GEdge {
  src: string
  dst: string
}

export interface TraceStep {
  step: number
  node: string
  inputs: number[][]
  outputs: number[][]
}

export interface GraphDoc {
  schema_version: number
  model_id: string
  revision: string
  fidelity: Fidelity
  architecture: string | null
  params_total: number
  config: Record<string, unknown>
  nodes: GNode[]
  repeats: Repeat[]
  edges: GEdge[]
  trace: TraceStep[]
  notes: string[]
}

/** Precomputed lookups for one loaded document. */
export interface GraphIndex {
  byId: Map<string, GNode>
  children: Map<string | null, GNode[]>
  repeatByRep: Map<string, Repeat>
  /** container id → repeat among its children (a collapsed container with one
   *  of these renders as a "×N" stack node) */
  repeatByParent: Map<string, Repeat>
  traceByNode: Map<string, TraceStep>
}

export function buildIndex(doc: GraphDoc): GraphIndex {
  const byId = new Map<string, GNode>()
  const children = new Map<string | null, GNode[]>()
  for (const n of doc.nodes) {
    byId.set(n.id, n)
    const list = children.get(n.parent) ?? []
    list.push(n)
    children.set(n.parent, list)
  }
  for (const list of children.values()) list.sort((a, b) => a.order - b.order)
  const repeatByRep = new Map(doc.repeats.map((r) => [r.representative, r]))
  const repeatByParent = new Map(doc.repeats.map((r) => [r.parent, r]))
  const traceByNode = new Map<string, TraceStep>()
  for (const t of doc.trace) if (!traceByNode.has(t.node)) traceByNode.set(t.node, t)
  return { byId, children, repeatByRep, repeatByParent, traceByNode }
}
