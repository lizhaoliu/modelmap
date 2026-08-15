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
  /** dim value → semantic label ("hidden", "vocab", …); see buildDimLabels */
  dimLabels: Map<number, string>
  /** batch size of the traced dummy input (labels index 0 of activations) */
  traceBatch: number | undefined
}

/** Label tensor dims by VALUE-matching against known quantities from the
 *  config and the traced input — never by position, which breaks on
 *  transposed weights (gpt2 Conv1D) and exotic layouts. A value claimed by
 *  two different labels is left unlabeled: wrong beats unlabeled, never. */
export function buildDimLabels(doc: GraphDoc): Map<number, string> {
  const c = doc.config as Record<string, unknown>
  const num = (k: string) => (typeof c[k] === 'number' ? (c[k] as number) : undefined)
  const hidden = num('hidden_size') ?? num('n_embd')
  const heads = num('num_attention_heads') ?? num('n_head')
  const candidates: [number | undefined, string][] = [
    [hidden, 'hidden'],
    [num('intermediate_size'), 'ffn'],
    [num('moe_intermediate_size'), 'ffn'],
    [num('vocab_size'), 'vocab'],
    [heads, 'heads'],
    [num('num_key_value_heads'), 'kv heads'],
    [num('head_dim'), 'head dim'],
    [num('num_experts') ?? num('n_routed_experts') ?? num('num_local_experts'), 'experts'],
    [num('max_position_embeddings') ?? num('n_positions'), 'max pos'],
    [num('patch_size'), 'patch'],
  ]
  if (!num('head_dim') && hidden && heads && hidden % heads === 0) {
    candidates.push([hidden / heads, 'head dim'])
  }
  const first = doc.trace[0]?.inputs?.[0]
  if (first?.length === 2) candidates.push([first[1], 'seq'])
  if (first?.length === 4) candidates.push([first[1], 'ch'], [first[2], 'px'], [first[3], 'px'])

  const map = new Map<number, string>()
  const clash = new Set<number>()
  for (const [v, label] of candidates) {
    if (v == null || v < 3 || clash.has(v)) continue // tiny values match everything
    const prev = map.get(v)
    if (prev && prev !== label) {
      map.delete(v)
      clash.add(v)
      continue
    }
    map.set(v, label)
  }
  return map
}

export function buildIndex(doc: GraphDoc): GraphIndex {
  const byId = new Map<string, GNode>()
  const children = new Map<string | null, GNode[]>()
  for (const n of doc.nodes) {
    byId.set(n.id, n)
    // weights-view graphs have no "" root module — their top-level nodes carry
    // parent null; normalize them under '' so the canvas treats them as roots
    const key = n.parent ?? (n.id === '' ? null : '')
    const list = children.get(key) ?? []
    list.push(n)
    children.set(key, list)
  }
  for (const list of children.values()) list.sort((a, b) => a.order - b.order)
  const repeatByRep = new Map(doc.repeats.map((r) => [r.representative, r]))
  const repeatByParent = new Map(doc.repeats.map((r) => [r.parent, r]))
  const traceByNode = new Map<string, TraceStep>()
  for (const t of doc.trace) if (!traceByNode.has(t.node)) traceByNode.set(t.node, t)
  return {
    byId,
    children,
    repeatByRep,
    repeatByParent,
    traceByNode,
    dimLabels: buildDimLabels(doc),
    traceBatch: doc.trace[0]?.inputs?.[0]?.[0],
  }
}
