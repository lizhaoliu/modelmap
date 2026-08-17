/**
 * Compare (design doc §15): align two module trees and diff the paired nodes.
 *
 * Alignment is recursive from the roots: children pair by leaf name first
 * (HF naming has converged — model.layers.N.self_attn.q_proj is the same in
 * Llama, Qwen, Mistral, Gemma), then unpaired children pair by role (same
 * kind, in order — wte ↔ embed_tokens). What is left is added / removed.
 * Repeat stacks compare representative-to-representative plus counts.
 */
import type { GNode, GraphDoc, GraphIndex } from '../types'

export type PairStatus = 'same' | 'changed' | 'added' | 'removed'

export interface Change {
  field: string
  a: string | null
  b: string | null
}

export interface Pair {
  key: string // A's id when present, else "+" + B's id
  a: GNode | null
  b: GNode | null
  status: PairStatus
  changes: Change[]
  /** true when this node or any descendant differs */
  dirty: boolean
}

export interface Alignment {
  pairs: Pair[]
  byA: Map<string, Pair>
  byB: Map<string, Pair>
  counts: { same: number; changed: number; added: number; removed: number }
  configDiff: Change[]
}

const CONFIG_KEYS = [
  'model_type', 'architectures', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
  'num_key_value_heads', 'head_dim', 'intermediate_size', 'vocab_size', 'max_position_embeddings',
  'rope_theta', 'rope_scaling', 'tie_word_embeddings', 'num_experts', 'num_experts_per_tok',
  'moe_intermediate_size', 'sliding_window', 'attention_bias', 'hidden_act', 'rms_norm_eps',
  'layer_norm_epsilon', 'dtype', 'torch_dtype', 'n_layer', 'n_head', 'n_embd', 'n_positions',
]

const leaf = (id: string) => id.split('.').pop() ?? id
const fmt = (v: unknown): string | null =>
  v == null ? null : typeof v === 'object' ? JSON.stringify(v) : String(v)

export function align(docA: GraphDoc, ia: GraphIndex, docB: GraphDoc, ib: GraphIndex): Alignment {
  const pairs: Pair[] = []
  const byA = new Map<string, Pair>()
  const byB = new Map<string, Pair>()

  const push = (a: GNode | null, b: GNode | null): Pair => {
    const p: Pair = { key: a ? a.id : `+${b!.id}`, a, b, status: 'same', changes: [], dirty: false }
    if (a && b) {
      p.changes = diffNodes(a, ia, b, ib)
      p.status = p.changes.length ? 'changed' : 'same'
    } else {
      p.status = a ? 'removed' : 'added'
    }
    pairs.push(p)
    if (a) byA.set(a.id, p)
    if (b) byB.set(b.id, p)
    return p
  }

  const alignChildren = (pa: string, pb: string): boolean => {
    const ca = [...(ia.children.get(pa) ?? [])]
    const cb = [...(ib.children.get(pb) ?? [])]
    const usedB = new Set<string>()
    const matched: [GNode, GNode][] = []
    // 1) by leaf name
    for (const a of ca) {
      const b = cb.find((x) => !usedB.has(x.id) && leaf(x.id) === leaf(a.id))
      if (b) { matched.push([a, b]); usedB.add(b.id) }
    }
    // 2) by role: same kind, in order, among what is left. Containers only
    //    pair when unambiguous (one unpaired container on each side —
    //    "transformer" ↔ "model"), never by position among several.
    const restA = ca.filter((a) => !matched.some(([x]) => x.id === a.id))
    const isBox = (n: GNode) => n.kind === 'container' || n.kind === 'module'
    const boxesA = restA.filter(isBox)
    const boxesB = cb.filter((x) => !usedB.has(x.id) && isBox(x))
    for (const a of restA) {
      let b: GNode | undefined
      if (isBox(a)) {
        if (boxesA.length === 1 && boxesB.length === 1) b = boxesB[0]
      } else {
        b = cb.find((x) => !usedB.has(x.id) && x.kind === a.kind && !isBox(x))
      }
      if (b) { matched.push([a, b]); usedB.add(b.id) }
    }
    // keep A's order
    matched.sort((x, y) => x[0].order - y[0].order)
    let dirty = false
    for (const [a, b] of matched) {
      const p = push(a, b)
      const childDirty = alignChildren(a.id, b.id)
      p.dirty = p.status !== 'same' || childDirty
      dirty = dirty || p.dirty
    }
    for (const a of ca) if (!matched.some(([x]) => x.id === a.id)) { markSubtree(a, ia, 'removed'); dirty = true }
    for (const b of cb) if (!usedB.has(b.id)) { markSubtree(b, ib, 'added'); dirty = true }
    return dirty
  }

  const markSubtree = (n: GNode, idx: GraphIndex, status: 'added' | 'removed') => {
    const p = status === 'removed' ? push(n, null) : push(null, n)
    p.dirty = true
    for (const c of idx.children.get(n.id) ?? []) markSubtree(c, idx, status)
  }

  // roots ('' in transformers graphs; several parentless roots in weights views)
  const rootsA = docA.nodes.filter((n) => n.parent === null)
  const rootsB = docB.nodes.filter((n) => n.parent === null)
  if (rootsA.length === 1 && rootsB.length === 1) {
    const p = push(rootsA[0], rootsB[0])
    p.dirty = alignChildren(rootsA[0].id, rootsB[0].id) || p.status !== 'same'
  } else {
    alignChildren('', '')
  }

  const counts = { same: 0, changed: 0, added: 0, removed: 0 }
  for (const p of pairs) counts[p.status]++

  const configDiff: Change[] = []
  for (const k of CONFIG_KEYS) {
    const a = fmt(docA.config[k]), b = fmt(docB.config[k])
    if (a !== b && (a != null || b != null)) configDiff.push({ field: k, a, b })
  }
  return { pairs, byA, byB, counts, configDiff }
}

function diffNodes(a: GNode, ia: GraphIndex, b: GNode, ib: GraphIndex): Change[] {
  const out: Change[] = []
  const cmp = (field: string, x: unknown, y: unknown) => {
    const fx = fmt(x), fy = fmt(y)
    if (fx !== fy) out.push({ field, a: fx, b: fy })
  }
  cmp('kind', a.kind, b.kind)
  cmp('class', a.cls, b.cls)
  cmp('params', a.params, b.params)
  cmp('dtype', a.dtype ?? null, b.dtype ?? null)
  const wa = a.weight_shapes ?? {}, wb = b.weight_shapes ?? {}
  for (const k of new Set([...Object.keys(wa), ...Object.keys(wb)])) cmp(`weight ${k}`, wa[k], wb[k])
  const aa = a.attrs ?? {}, ab = b.attrs ?? {}
  for (const k of new Set([...Object.keys(aa), ...Object.keys(ab)])) {
    if (k.startsWith('_')) continue
    cmp(k, aa[k], ab[k])
  }
  cmp('repeats', ia.repeatByRep.get(a.id)?.count ?? null, ib.repeatByRep.get(b.id)?.count ?? null)
  const ta = ia.traceByNode.get(a.id), tb = ib.traceByNode.get(b.id)
  if (ta && tb) {
    cmp('input', ta.inputs[0], tb.inputs[0])
    cmp('output', ta.outputs[0], tb.outputs[0])
  }
  return out
}

/** Per-side status maps for node styling: dirty containers get 'inside'. */
export function sideStatus(al: Alignment): { a: Map<string, PairStatus | 'inside'>; b: Map<string, PairStatus | 'inside'> } {
  const a = new Map<string, PairStatus | 'inside'>()
  const b = new Map<string, PairStatus | 'inside'>()
  for (const p of al.pairs) {
    const st: PairStatus | 'inside' = p.status === 'same' && p.dirty ? 'inside' : p.status
    if (p.a) a.set(p.a.id, st)
    if (p.b) b.set(p.b.id, st)
  }
  return { a, b }
}
