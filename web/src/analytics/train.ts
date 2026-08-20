/**
 * Training planner + roofline throughput (design doc §22). Twin of
 * analytics.py plan_training / estimate_throughput — keep the arithmetic
 * identical (web/tests/train.test.ts pins the same fixture numbers).
 */
import type { GraphDoc, GraphIndex } from '../types'
import { computeCosts, type Assumptions } from './cost'

export const LORA_TARGETS: Record<string, string[]> = {
  attention: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv_proj', 'c_attn', 'query', 'key', 'value', 'dense', 'out_proj'],
  'attn-mlp': [
    'q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv_proj', 'c_attn', 'query', 'key', 'value', 'dense', 'out_proj',
    'gate_proj', 'up_proj', 'down_proj', 'c_fc', 'c_proj', 'fc1', 'fc2', 'wi', 'wo',
  ],
  'all-linear': [],
}

export const OPTIMIZER_BYTES: Record<string, number> = { adamw: 12, adamw8bit: 6 }
export const QLORA_BASE_BYTES = 0.55

export interface TrainRequest {
  method: 'full' | 'lora' | 'qlora'
  optimizer: 'adamw' | 'adamw8bit'
  loraRank: number
  loraTargets: 'attention' | 'attn-mlp' | 'all-linear'
  gpus: number
  gpuMemoryGb: number
  sharding: 'none' | 'zero2' | 'zero3'
  T: number
  B: number
  gradCheckpoint: boolean
  flashAttention: boolean
  headroom: number
  gpu?: string
}

export interface TrainPlan {
  request: TrainRequest
  trainableParams: number
  totalParams: number
  weightBytesPerGpu: number
  gradBytesPerGpu: number
  optimizerBytesPerGpu: number
  activationBytesPerGpu: number
  totalBytesPerGpu: number
  perGpuCapacityBytes: number
  fits: boolean
  maxMicrobatch: number
  trainTokensPerSec: number | null
  notes: string[]
}

const num = (c: Record<string, unknown>, ...keys: string[]) => {
  for (const k of keys) if (typeof c[k] === 'number') return c[k] as number
  return undefined
}

function loraTrainable(doc: GraphDoc, mult: Map<string, number>, targets: string, rank: number): { total: number; matched: number } {
  const leaves = LORA_TARGETS[targets] ?? []
  let total = 0
  let matched = 0
  for (const n of doc.nodes) {
    const w = n.weight_shapes?.weight
    if (!w || w.length !== 2) continue
    if (n.kind === 'embedding' || n.kind === 'head' || n.kind === 'norm') continue
    const leaf = n.id.split('.').pop()!
    if (targets !== 'all-linear' && !leaves.includes(leaf)) continue
    const m = mult.get(n.id) ?? 1
    total += rank * (w[0] + w[1]) * m
    matched += m
  }
  return { total, matched }
}

export function planTraining(doc: GraphDoc, index: GraphIndex, req: TrainRequest): TrainPlan {
  const notes: string[] = []
  const a: Assumptions = { T: req.T, B: req.B, bytes: 2, dtypeLabel: 'bf16' }
  const rep = computeCosts(doc, index, a)
  const c = doc.config as Record<string, unknown>
  const params = doc.params_total
  const gpus = Math.max(1, req.gpus)
  const cap = req.gpuMemoryGb * 2 ** 30 * (1 - req.headroom)

  // multiplicity map (same recursion the cost roll-up uses)
  const mult = new Map<string, number>()
  const multOf = (id: string): number => {
    const hit = mult.get(id)
    if (hit != null) return hit
    const n = index.byId.get(id)
    const own = index.repeatByRep.get(id)?.count ?? 1
    const v = own * (n?.parent != null ? multOf(n.parent) : 1)
    mult.set(id, v)
    return v
  }
  for (const n of doc.nodes) multOf(n.id)

  let trainable: number
  let weightBytes: number
  if (req.method === 'full') {
    trainable = params
    weightBytes = params * 2
    notes.push('full fine-tune: every parameter trains (bf16 weights)')
  } else {
    const { total, matched } = loraTrainable(doc, mult, req.loraTargets, req.loraRank)
    trainable = total
    if (!matched) notes.push(`no linear modules matched targets '${req.loraTargets}'`)
    else notes.push(`LoRA r=${req.loraRank} on ${matched} linear modules (${req.loraTargets})`)
    if (doc.nodes.some((n) => Object.values(n.weight_shapes ?? {}).some((w) => w.length === 3)))
      notes.push('fused 3-D expert weights are not LoRA targets here')
    const base = params * (req.method === 'qlora' ? QLORA_BASE_BYTES : 2)
    if (req.method === 'qlora') notes.push('QLoRA: frozen base at NF4 (≈0.55 B/param incl. quant constants)')
    weightBytes = base + total * 2
  }
  const gradBytes = trainable * 2
  const optBytes = trainable * (OPTIMIZER_BYTES[req.optimizer] ?? 12)

  const layers = num(c, 'num_hidden_layers', 'n_layer') ?? 0
  const hidden = num(c, 'hidden_size', 'n_embd') ?? 0
  const heads = num(c, 'num_attention_heads', 'n_head') ?? 0
  const reps = [...doc.repeats].sort(
    (x, y) => y.count * (index.byId.get(y.representative)?.params ?? 0) - x.count * (index.byId.get(x.representative)?.params ?? 0),
  )
  const blockAct = reps.length ? rep.byNode.get(reps[0].representative)!.actBytes : 0
  const fullAct = rep.root.actBytes
  const scores = req.flashAttention ? 0 : layers * heads * req.B * req.T * req.T * 2
  let actBytes: number
  if (req.gradCheckpoint) {
    actBytes = layers * req.B * req.T * hidden * 2 + blockAct + (req.flashAttention ? 0 : scores / Math.max(layers, 1))
    notes.push("gradient checkpointing: layer inputs kept, one block's activations resident during recompute")
  } else {
    actBytes = fullAct + scores
    notes.push('no gradient checkpointing: every traced activation held for backward')
  }
  if (!req.flashAttention) notes.push('without flash attention, softmax scores add heads × T² per layer')

  let wGpu = weightBytes
  let gGpu = gradBytes
  let oGpu = optBytes
  if (req.sharding === 'zero3') {
    wGpu /= gpus
    gGpu /= gpus
    oGpu /= gpus
    notes.push('ZeRO-3 / FSDP: weights, grads and optimizer sharded across GPUs (gathering adds transient overhead)')
  } else if (req.sharding === 'zero2') {
    gGpu /= gpus
    oGpu /= gpus
    notes.push('ZeRO-2: grads and optimizer sharded; each GPU keeps full weights')
  } else if (gpus > 1) {
    notes.push('plain data parallel: every GPU holds a full replica')
  }
  const total = wGpu + gGpu + oGpu + actBytes
  const fits = total <= cap
  const fixed = wGpu + gGpu + oGpu
  const actPerB = actBytes / Math.max(req.B, 1)
  const maxMicrobatch = actPerB > 0 && cap > fixed ? Math.floor((cap - fixed) / actPerB) : 0

  let tps: number | null = null
  if (req.gpu && GPU_SPECS[req.gpu]) {
    const spec = GPU_SPECS[req.gpu]
    const macsTok = rep.root.macs / Math.max(1, req.T * req.B)
    tps = (spec.tflops * 1e12 * TRAIN_MFU * gpus) / (6 * macsTok)
    notes.push(`speed assumes ${Math.round(TRAIN_MFU * 100)}% MFU on ${req.gpu}; fwd+bwd ≈ 3× forward FLOPs`)
  }
  return {
    request: req, trainableParams: trainable, totalParams: params,
    weightBytesPerGpu: wGpu, gradBytesPerGpu: gGpu, optimizerBytesPerGpu: oGpu,
    activationBytesPerGpu: actBytes, totalBytesPerGpu: total, perGpuCapacityBytes: cap,
    fits, maxMicrobatch, trainTokensPerSec: tps, notes,
  }
}

// ------------------------------------------- roofline throughput

export const GPU_SPECS: Record<string, { tflops: number; bw: number }> = {
  'H100 80GB': { tflops: 989, bw: 3350 }, 'H200 141GB': { tflops: 989, bw: 4800 },
  'A100 80GB': { tflops: 312, bw: 2039 }, 'A100 40GB': { tflops: 312, bw: 1555 },
  'L40S 48GB': { tflops: 362, bw: 864 }, 'L4 24GB': { tflops: 121, bw: 300 },
  'A10G 24GB': { tflops: 70, bw: 600 }, 'RTX 4090 24GB': { tflops: 165, bw: 1008 },
  'RTX 3090 24GB': { tflops: 71, bw: 936 }, 'RTX 5090 32GB': { tflops: 210, bw: 1792 },
  'MI300X 192GB': { tflops: 1307, bw: 5300 }, 'B200 180GB': { tflops: 2250, bw: 8000 },
  'T4 16GB': { tflops: 65, bw: 320 },
}
export const PREFILL_MFU = 0.4
export const DECODE_BW_EFF = 0.6
export const TRAIN_MFU = 0.35

export interface Throughput {
  gpu: string
  prefillTokPerSec: number
  decodeTokPerSecB1: number
  decodeTokPerSecAtB: number
  batch: number
  bytesReadPerToken: number
  notes: string[]
}

export function estimateThroughput(
  doc: GraphDoc, index: GraphIndex, gpu: string,
  opts: { tp?: number; T?: number; B?: number; bytes?: number; dtypeLabel?: string } = {},
): Throughput | null {
  const spec = GPU_SPECS[gpu]
  if (!spec) return null
  const tp = opts.tp ?? 1
  const T = opts.T ?? 4096
  const B = opts.B ?? 1
  const rep = computeCosts(doc, index, { T, B, bytes: opts.bytes ?? 2, dtypeLabel: opts.dtypeLabel ?? 'bf16' })
  const tokens = Math.max(1, T * B)
  const macsTok = rep.root.macs / tokens
  const params = doc.params_total || 1
  const activeFrac = rep.root.activeParams / params
  const activeBytes = rep.root.paramBytes * activeFrac
  const kvRead = rep.root.kvPerToken * T
  const perSeqBytes = activeBytes + kvRead
  const bw = spec.bw * 1e9 * DECODE_BW_EFF * tp
  const flops = spec.tflops * 1e12 * PREFILL_MFU * tp
  const notes = [
    `roofline estimate: prefill at ${Math.round(PREFILL_MFU * 100)}% MFU, decode at ${Math.round(DECODE_BW_EFF * 100)}% of ${spec.bw.toFixed(0)} GB/s; interconnect and scheduler overhead not modeled`,
  ]
  if (activeFrac < 0.95) notes.push(`MoE: decode streams the ≈${Math.round(activeFrac * 100)}% of weights that are active per token`)
  return {
    gpu,
    prefillTokPerSec: flops / (2 * macsTok),
    decodeTokPerSecB1: bw / perSeqBytes,
    decodeTokPerSecAtB: (B * bw) / (activeBytes + B * kvRead),
    batch: B,
    bytesReadPerToken: perSeqBytes,
    notes,
  }
}
