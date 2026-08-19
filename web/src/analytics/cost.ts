/**
 * Cost lens (design doc §14): compute, memory and KV-cache estimates derived
 * on the client from traced shapes, module attributes and config — so a
 * change of assumptions re-derives everything instantly.
 *
 * Every number is an analytic estimate: MACs for weight matmuls plus the
 * attention core, bytes from shapes × dtype. Fused kernels, softmax, rope and
 * the like are counted as "other" (elementwise) or not at all, and every
 * value carries the formula it came from.
 */
import type { GNode, GraphDoc, GraphIndex } from '../types'

export interface Assumptions {
  T: number // sequence length
  B: number // batch
  bytes: number // bytes per activation / weight element for the what-if dtype
  dtypeLabel: string
}

export const DEFAULT_ASSUMPTIONS: Assumptions = { T: 4096, B: 1, bytes: 2, dtypeLabel: 'bf16' }

export interface Cost {
  /** weight-matmul + attention-core MACs for the whole forward at (B, T) */
  macs: number
  /** elementwise work (norms, activations, rope, embeddings lookup) — elements touched */
  other: number
  /** parameter bytes at the *stored* dtype */
  paramBytes: number
  /** parameters that actually run for one token (MoE: k/E of experts) */
  activeParams: number
  /** output activation bytes at (B, T) — summed over the subtree */
  actBytes: number
  /** largest single output activation in the subtree */
  maxAct: number
  maxActNode: string
  /** KV-cache bytes per token contributed by attention layers in the subtree */
  kvPerToken: number
  /** own-node formula (for the tooltip) */
  formula?: string
}

export interface CostReport {
  byNode: Map<string, Cost>
  root: Cost
  assumptions: Assumptions
  /** how many attention layers were counted for KV, and how many skipped (linear attention) */
  kvLayers: number
  kvSkipped: number
  notes: string[]
}

/** bytes per element at a stored dtype; GGUF block quants are fractional
 *  (bits per weight incl. scales / 8) — mirrors analytics.py DTYPE_BYTES */
export const DTYPE_BYTES: Record<string, number> = {
  f64: 8, f32: 4, float32: 4, f16: 2, float16: 2, bf16: 2, bfloat16: 2,
  f8_e4m3: 1, f8_e5m2: 1, float8_e4m3fn: 1, float8_e5m2: 1, i8: 1, int8: 1, u8: 1, bool: 1,
  i16: 2, i32: 4, i64: 8, int4: 0.5, u4: 0.5, i4: 0.5, nf4: 0.5, fp4: 0.5,
  q4_0: 4.5 / 8, q4_1: 5 / 8, q5_0: 5.5 / 8, q5_1: 6 / 8, q8_0: 8.5 / 8, q8_1: 9 / 8,
  q2_k: 2.625 / 8, q3_k: 3.4375 / 8, q4_k: 4.5 / 8, q5_k: 5.5 / 8, q6_k: 6.5625 / 8,
  q8_k: 8.5 / 8, iq2_xxs: 2.0625 / 8, iq2_xs: 2.3125 / 8, iq2_s: 2.5 / 8,
  iq3_xxs: 3.0625 / 8, iq3_s: 3.4375 / 8, iq1_s: 1.5625 / 8, iq1_m: 1.75 / 8,
  iq4_nl: 4.5 / 8, iq4_xs: 4.25 / 8, tq1_0: 1.6875 / 8, tq2_0: 2.0625 / 8, mxfp4: 4.25 / 8,
}
export function bytesOf(dtype: string | null | undefined, fallback: number): number {
  if (!dtype) return fallback
  return DTYPE_BYTES[dtype.toLowerCase()] ?? fallback
}
/** "q4_k" → "Q4_K · 4.5 bpw"; plain dtypes pass through */
export function fmtDtype(dtype: string | null | undefined): string {
  if (!dtype) return '—'
  const d = dtype.toLowerCase()
  if (/^(i?q\d|tq\d|mxfp4)/.test(d)) {
    const b = DTYPE_BYTES[d]
    return `${dtype.toUpperCase()}${b ? ` · ${+(b * 8).toFixed(2)} bpw` : ''}`
  }
  return dtype
}

const VISION = /(^|\.)(visual|vision|vision_tower|vision_model|image_encoder)(\.|$)/
const LINEAR_ATTN = /(DeltaNet|Mamba|LinearAttention|GatedLinear|SSM|Retention)/i

function num(c: Record<string, unknown>, ...keys: string[]): number | undefined {
  for (const k of keys) if (typeof c[k] === 'number') return c[k] as number
  return undefined
}
const prod = (xs: number[]) => xs.reduce((a, b) => a * b, 1)

export function computeCosts(doc: GraphDoc, index: GraphIndex, a: Assumptions): CostReport {
  const c = doc.config as Record<string, unknown>
  const notes: string[] = []
  const T0 = (index.dimLabels && [...index.dimLabels.entries()].find(([, l]) => l === 'seq')?.[0]) ?? 7
  const B0 = index.traceBatch ?? 1

  // multiplicity: how many times a node really occurs (repeat stacks)
  const mult = new Map<string, number>()
  const multOf = (id: string): number => {
    const m = mult.get(id)
    if (m != null) return m
    const n = index.byId.get(id)
    const own = index.repeatByRep.get(id)?.count ?? 1
    const v = own * (n?.parent != null ? multOf(n.parent) : 1)
    mult.set(id, v)
    return v
  }

  // MoE: E and k, from the fused weight or config
  const E = num(c, 'num_experts', 'n_routed_experts', 'num_local_experts') ?? 0
  const K = num(c, 'num_experts_per_tok', 'moe_top_k', 'num_experts_per_token') ?? 0
  const underExperts = (id: string) => /(^|\.)experts(\.|$)/.test(id) && !/shared_experts/.test(id)
  const expertFrac = (id: string, weights: number[][]): number => {
    if (!underExperts(id) || !K) return 1
    const fused = weights.find((w) => w.length === 3)
    const e = fused ? fused[0] : E
    return e ? Math.min(1, K / e) : 1
  }

  // shape scaling: seq dims → T, batch dim → B; vision towers keep their patch count
  const scale = (shape: number[], id: string): number[] => {
    const vis = VISION.test(id)
    return shape.map((d, i) => {
      if (i === 0 && d === B0 && shape.length > 1) return a.B
      if (!vis && d === T0) return a.T
      return d
    })
  }
  const tokensOf = (shape: number[] | undefined, id: string): number => {
    if (!shape || shape.length < 2) return a.B * a.T
    const s = scale(shape, id)
    let t = prod(s.slice(0, -1))
    if (!(shape[0] === B0 && shape.length > 1)) t *= a.B // batch folded into tokens
    return t
  }

  // attention geometry (text and vision)
  const heads = num(c, 'num_attention_heads', 'n_head') ?? 0
  const kvHeads = num(c, 'num_key_value_heads') ?? heads
  const hidden = num(c, 'hidden_size', 'n_embd') ?? 0
  const headDim = num(c, 'head_dim') ?? (heads ? hidden / heads : 0)
  const qkDim = (num(c, 'qk_nope_head_dim') ?? 0) + (num(c, 'qk_rope_head_dim') ?? 0) || headDim
  const vDim = num(c, 'v_head_dim') ?? headDim
  const kvLora = num(c, 'kv_lora_rank')
  const ropeDim = num(c, 'qk_rope_head_dim') ?? 0
  const vc = (c.vision_config ?? {}) as Record<string, unknown>
  const vHeads = num(vc, 'num_heads', 'num_attention_heads') ?? 0
  const vHidden = num(vc, 'hidden_size') ?? 0
  const vHeadDim = vHeads ? vHidden / vHeads : 0

  // a tied lm_head shares the embedding matrix: real compute, but its
  // parameters are stored (and counted in params_total) only once
  const tiedHeads = new Set<string>()
  if (c.tie_word_embeddings === true) {
    const embShapes = new Set(
      doc.nodes.filter((n) => n.kind === 'embedding').flatMap((n) => Object.values(n.weight_shapes ?? {}).map((w) => w.join('x'))),
    )
    for (const n of doc.nodes) {
      if (n.kind === 'head' && Object.values(n.weight_shapes ?? {}).some((w) => embShapes.has(w.join('x')))) tiedHeads.add(n.id)
    }
  }

  const own = new Map<string, Cost>()
  let kvLayers = 0
  let kvSkipped = 0

  for (const n of doc.nodes) {
    const io = index.traceByNode.get(n.id)
    const weights = Object.values(n.weight_shapes ?? {})
    const cost: Cost = {
      macs: 0, other: 0, paramBytes: 0, activeParams: 0,
      actBytes: 0, maxAct: 0, maxActNode: n.id, kvPerToken: 0,
    }
    // own params (non-recursive) → bytes at stored dtype, active share
    const ownParams = tiedHeads.has(n.id) ? 0 : weights.reduce((s, w) => s + prod(w), 0)
    const frac = expertFrac(n.id, weights)
    cost.paramBytes = ownParams * bytesOf(n.dtype, a.bytes)
    cost.activeParams = ownParams * frac

    // output activations
    if (io?.outputs?.length) {
      const out = io.outputs[0]
      const bytes = prod(scale(out, n.id)) * a.bytes
      cost.actBytes = bytes
      cost.maxAct = bytes
    }

    const kids = index.children.get(n.id) ?? []
    const isLeaf = kids.length === 0
    const formula: string[] = []

    const matmulWeights = weights.filter((w) => w.length === 2 || w.length === 3)
    if (!['embedding', 'norm', 'conv'].includes(n.kind) && matmulWeights.length) {
      // any module owning a ≥2-D weight multiplies by it: nn.Linear, Conv1D,
      // routers, fused experts (3-D, only k of E run per token → × k/E)
      const tokens = tokensOf(io?.inputs?.[0], n.id)
      for (const w of matmulWeights) cost.macs += tokens * prod(w) * frac
      formula.push(frac < 1 ? `tokens × k/E × prod(W) = ${fmtInt(tokens)} × ${K}/${E} × …` : `tokens × in × out = ${fmtInt(tokens)} × …`)
    } else if (n.kind === 'conv') {
      const out = io?.outputs?.[0]
      const w = weights.find((x) => x.length >= 3)
      if (out && w) {
        const outElems = prod(scale(out, n.id))
        const perOut = prod(w.slice(1)) // in_ch/groups × kernel
        cost.macs = outElems * perOut
        formula.push(`out_elems × in_ch/groups × kernel = ${fmtInt(outElems)} × ${fmtInt(perOut)}`)
      }
    } else if (n.kind === 'attention') {
      // analytic core: QKᵀ and AV — not modules, so not in the trace
      const vis = VISION.test(n.id)
      const linear = LINEAR_ATTN.test(n.cls)
      if (!linear) {
        const h = vis ? vHeads : heads
        const dq = vis ? vHeadDim : qkDim
        const dv = vis ? vHeadDim : vDim
        const seq = vis ? (io?.inputs?.[0]?.[0] ?? 0) : a.T
        const b = vis ? a.B : a.B
        cost.macs = b * h * seq * seq * (dq + dv)
        formula.push(`attention core: B × heads × T² × (d_qk + d_v) = ${b} × ${h} × ${fmtInt(seq)}² × ${dq + dv}`)
        if (!vis) {
          const perLayer = kvLora ? (kvLora + ropeDim) * a.bytes : 2 * kvHeads * headDim * a.bytes
          cost.kvPerToken = perLayer
          kvLayers += multOf(n.id)
        }
      } else {
        kvSkipped += multOf(n.id)
        formula.push('linear attention: T-linear core not modeled')
      }
    } else if (isLeaf && io?.outputs?.length) {
      // elementwise-ish leaves (norms, activations, rope, embeddings)
      cost.other = prod(scale(io.outputs[0], n.id))
    }
    if (formula.length) cost.formula = formula.join('; ')
    own.set(n.id, cost)
  }

  // roll-ups (children × repeat multiplicity), bottom-up by depth
  const byNode = new Map<string, Cost>()
  const ordered = [...doc.nodes].sort((x, y) => y.depth - x.depth)
  for (const n of ordered) {
    const base = own.get(n.id)!
    const total: Cost = { ...base }
    for (const k of index.children.get(n.id) ?? []) {
      const kc = byNode.get(k.id)
      if (!kc) continue
      const m = index.repeatByRep.get(k.id)?.count ?? 1
      total.macs += kc.macs * m
      total.other += kc.other * m
      total.paramBytes += kc.paramBytes * m
      total.activeParams += kc.activeParams * m
      total.actBytes += kc.actBytes * m
      total.kvPerToken += kc.kvPerToken * m
      if (kc.maxAct > total.maxAct) {
        total.maxAct = kc.maxAct
        total.maxActNode = kc.maxActNode
      }
    }
    byNode.set(n.id, total)
  }
  const rootNode = doc.nodes.find((n) => n.parent === null && n.id === '') ?? doc.nodes.find((n) => n.parent === null)
  let root = rootNode ? byNode.get(rootNode.id)! : emptyCost()
  if (!rootNode || (index.children.get('')?.length ?? 0) > 1 && rootNode.id !== '') {
    // weights-view graphs: several parentless roots — sum them
    root = emptyCost()
    for (const n of doc.nodes.filter((x) => x.parent === null)) {
      const kc = byNode.get(n.id)!
      root.macs += kc.macs; root.other += kc.other; root.paramBytes += kc.paramBytes
      root.activeParams += kc.activeParams; root.actBytes += kc.actBytes; root.kvPerToken += kc.kvPerToken
      if (kc.maxAct > root.maxAct) { root.maxAct = kc.maxAct; root.maxActNode = kc.maxActNode }
    }
  }
  if (kvSkipped) notes.push(`${kvSkipped} linear-attention layers hold no KV cache`)
  if (tiedHeads.size) notes.push('lm_head is tied to the embedding matrix (stored once)')
  if (E && K) notes.push(`MoE: ${K} of ${E} experts run per token`)
  return { byNode, root, assumptions: a, kvLayers, kvSkipped, notes }
}

function emptyCost(): Cost {
  return { macs: 0, other: 0, paramBytes: 0, activeParams: 0, actBytes: 0, maxAct: 0, maxActNode: '', kvPerToken: 0 }
}

// ---------------------------------------------------------------- formatting

export function fmtInt(n: number): string {
  return Math.round(n).toLocaleString('en-US')
}
export function fmtBig(n: number, unit = ''): string {
  const abs = Math.abs(n)
  const [v, s] =
    abs >= 1e15 ? [n / 1e15, 'P'] : abs >= 1e12 ? [n / 1e12, 'T'] : abs >= 1e9 ? [n / 1e9, 'G']
    : abs >= 1e6 ? [n / 1e6, 'M'] : abs >= 1e3 ? [n / 1e3, 'K'] : [n, '']
  const digits = Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2
  return `${v.toFixed(digits)} ${s}${unit}`
}
export function fmtBytes(n: number): string {
  const abs = Math.abs(n)
  const [v, s] =
    abs >= 2 ** 40 ? [n / 2 ** 40, 'TB'] : abs >= 2 ** 30 ? [n / 2 ** 30, 'GB'] : abs >= 2 ** 20 ? [n / 2 ** 20, 'MB']
    : abs >= 1024 ? [n / 1024, 'KB'] : [n, 'B']
  const digits = Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2
  return `${v.toFixed(digits)} ${s}`
}
export function fmtMacs(n: number): string {
  return fmtBig(n, 'MAC')
}

export type Lens = 'none' | 'params' | 'compute' | 'memory' | 'kv'
export const LENSES: { id: Lens; label: string; hint: string }[] = [
  { id: 'params', label: 'params', hint: 'parameter count' },
  { id: 'compute', label: 'compute', hint: 'MACs per forward at the chosen T and B (estimate)' },
  { id: 'memory', label: 'memory', hint: 'output activation bytes at the chosen T, B and dtype' },
  { id: 'kv', label: 'kv', hint: 'KV-cache bytes per token' },
]
export function lensValue(lens: Lens, node: GNode, cost: Cost | undefined): number {
  if (!cost) return lens === 'params' ? node.params : 0
  switch (lens) {
    case 'params': return node.params
    case 'compute': return cost.macs
    case 'memory': return cost.actBytes
    case 'kv': return cost.kvPerToken
    default: return node.params
  }
}
export function fmtLens(lens: Lens, v: number): string {
  switch (lens) {
    case 'compute': return fmtMacs(v)
    case 'memory': return fmtBytes(v)
    case 'kv': return v ? `${fmtBytes(v)}/tok` : '—'
    default: return fmtBig(v)
  }
}
