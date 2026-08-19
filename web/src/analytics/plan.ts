/**
 * Serving planner (design doc §17): TP × PP placement estimate. Twin of
 * analytics.plan_serving in Python — keep the arithmetic identical so the UI,
 * CLI and MCP agree (web/tests/plan.test.ts pins the same fixtures).
 */
import type { GraphDoc, GraphIndex } from '../types'
import { computeCosts, type Assumptions } from './cost'

export interface PlanRequest {
  gpus: number
  gpuMemoryGb: number
  tp: number
  pp: number
  T: number
  B: number
  dtypeLabel: string
  bytes: number
  headroom: number
}

export interface Stage {
  stage: number
  gpus: number[]
  layers: [number, number] | []
  layerCount: number
  weightBytesPerGpu: number
  kvBytesPerGpu: number
  actBytesPerGpu: number
  totalBytesPerGpu: number
  fits: boolean
  boundaryBytesOut: number
}

export interface Plan {
  request: PlanRequest
  fits: boolean
  stages: Stage[]
  weightBytes: number
  kvBytes: number
  actBytes: number
  perGpuCapacityBytes: number
  maxContextTokens: number
  notes: string[]
}

export const GPU_PRESETS: [string, number][] = [
  ['H100 80GB', 80], ['H200 141GB', 141], ['A100 80GB', 80], ['A100 40GB', 40], ['L40S 48GB', 48],
  ['L4 24GB', 24], ['A10G 24GB', 24], ['RTX 4090 24GB', 24], ['RTX 3090 24GB', 24], ['RTX 5090 32GB', 32],
  ['MI300X 192GB', 192], ['B200 180GB', 180], ['T4 16GB', 16],
]

const num = (c: Record<string, unknown>, ...keys: string[]) => {
  for (const k of keys) if (typeof c[k] === 'number') return c[k] as number
  return undefined
}

export function planServing(doc: GraphDoc, index: GraphIndex, req: PlanRequest): Plan {
  const notes: string[] = []
  const a: Assumptions = { T: req.T, B: req.B, bytes: req.bytes, dtypeLabel: req.dtypeLabel }
  const rep = computeCosts(doc, index, a)
  const c = doc.config as Record<string, unknown>
  const tp = Math.max(1, req.tp), pp = Math.max(1, req.pp)
  if (tp * pp !== req.gpus) notes.push(`tp × pp = ${tp * pp} ≠ gpus = ${req.gpus}; planning for ${tp * pp} GPUs`)
  const cap = req.gpuMemoryGb * 2 ** 30 * (1 - req.headroom)
  const tokens = req.T * req.B
  const hidden = num(c, 'hidden_size', 'n_embd') ?? 0

  const reps = [...doc.repeats].sort(
    (x, y) => y.count * (index.byId.get(y.representative)?.params ?? 0) - x.count * (index.byId.get(x.representative)?.params ?? 0),
  )
  const stack = reps[0]
  const items: [string, number, number][] = []
  let outsideW = 0, outsideKv = 0
  if (stack) {
    const parent = stack.parent
    for (const s of index.children.get(parent) ?? []) {
      const r = index.repeatByRep.get(s.id)
      const cost = rep.byNode.get(s.id)!
      if (r) for (const m of r.members) items.push([`${parent}.${m}`, cost.paramBytes, cost.kvPerToken])
      else items.push([s.id || 'model', cost.paramBytes, cost.kvPerToken])
    }
    outsideW = rep.root.paramBytes - rep.byNode.get(parent)!.paramBytes
    outsideKv = rep.root.kvPerToken - rep.byNode.get(parent)!.kvPerToken
  } else {
    notes.push('no repeated layer stack found; pipeline stages split the top-level modules')
    for (const s of index.children.get('') ?? []) {
      const cost = rep.byNode.get(s.id)!
      items.push([s.id, cost.paramBytes, cost.kvPerToken])
    }
  }
  const [firstExtra, lastExtra] = pp > 1 ? [outsideW / 2, outsideW / 2] : [outsideW, 0]
  const totalW = items.reduce((s, [, w]) => s + w, 0)
  const target = (totalW + outsideW) / pp
  const groups: number[][] = Array.from({ length: pp }, () => [])
  let acc = firstExtra, g = 0
  items.forEach(([, w], i) => {
    if (g < pp - 1 && acc + w > target && groups[g].length) { g++; acc = 0 }
    groups[g].push(i)
    acc += w
  })
  let largestAct = 0
  for (const n of doc.nodes) {
    if (n.kind === 'head' || (index.children.get(n.id)?.length ?? 0) > 0) continue
    largestAct = Math.max(largestAct, rep.byNode.get(n.id)?.maxAct ?? 0)
  }
  const stages: Stage[] = []
  let fits = true
  for (let s = 0; s < pp; s++) {
    const idx = groups[s]
    const w = idx.reduce((t, i) => t + items[i][1], 0) + (s === 0 ? firstExtra : 0) + (s === pp - 1 ? lastExtra : 0)
    const kv = idx.reduce((t, i) => t + items[i][2], 0) * tokens + (s === 0 ? outsideKv * tokens : 0)
    const wGpu = w / tp, kvGpu = kv / tp, actGpu = largestAct / tp
    const total = wGpu + kvGpu + actGpu
    const layerIds = idx.map((i) => items[i][0].split('.').pop()!).filter((x) => /^\d+$/.test(x)).map(Number)
    stages.push({
      stage: s,
      gpus: Array.from({ length: tp }, (_, k) => s * tp + k),
      layers: layerIds.length ? [Math.min(...layerIds), Math.max(...layerIds)] : [],
      layerCount: layerIds.length,
      weightBytesPerGpu: wGpu,
      kvBytesPerGpu: kvGpu,
      actBytesPerGpu: actGpu,
      totalBytesPerGpu: total,
      fits: total <= cap,
      boundaryBytesOut: s < pp - 1 ? req.B * req.T * hidden * a.bytes : 0,
    })
    fits = fits && total <= cap
  }
  let maxCtx = Infinity
  for (const st of stages) {
    const kvTok = tokens ? st.kvBytesPerGpu / tokens : 0
    const free = cap - st.weightBytesPerGpu - st.actBytesPerGpu
    if (kvTok > 0) maxCtx = Math.min(maxCtx, free / kvTok / Math.max(1, req.B))
    else if (free < 0) maxCtx = 0
  }
  let maxContextTokens = maxCtx === Infinity ? 0 : Math.max(0, Math.floor(maxCtx))
  if (!stages.some((st) => st.kvBytesPerGpu)) {
    notes.push('no KV cache (no standard attention layers found); context is not memory-bound here')
    maxContextTokens = 0
  }
  if (tp > 1) {
    const heads = num(c, 'num_attention_heads', 'n_head') ?? 0
    const kvh = num(c, 'num_key_value_heads') ?? heads
    if (heads && heads % tp) notes.push(`${heads} attention heads do not divide evenly across tp=${tp}`)
    if (kvh && kvh < tp) notes.push(`only ${kvh} KV heads: tp=${tp} replicates K/V (KV memory per GPU is higher than shown)`)
  }
  if (req.gpuMemoryGb <= 0) notes.push('GPU memory is 0 — unified-memory devices: compare totals against system RAM')
  notes.push(
    'activations = largest single non-logits activation at T, B (prefill peak; decode needs far less); weights at stored dtypes; KV at the activation dtype; no framework workspace beyond the headroom',
  )
  return {
    request: req, fits, stages, weightBytes: rep.root.paramBytes, kvBytes: rep.root.kvPerToken * tokens,
    actBytes: largestAct, perGpuCapacityBytes: cap, maxContextTokens, notes,
  }
}
