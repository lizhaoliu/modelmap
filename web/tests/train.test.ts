import { describe, expect, it } from 'vitest'
import { gunzipSync } from 'node:zlib'
import { readFileSync } from 'node:fs'
import { buildIndex, type GraphDoc } from '../src/types'
import { estimateThroughput, planTraining, type TrainRequest } from '../src/analytics/train'

const load = (name: string): GraphDoc =>
  JSON.parse(gunzipSync(readFileSync(new URL(`./fixtures/${name}.graph.json.gz`, import.meta.url))).toString())
const req = (o: Partial<TrainRequest>): TrainRequest => ({
  method: 'lora', optimizer: 'adamw', loraRank: 16, loraTargets: 'attn-mlp',
  gpus: 1, gpuMemoryGb: 24, sharding: 'none', T: 2048, B: 1,
  gradCheckpoint: true, flashAttention: true, headroom: 0.1, ...o,
})

// the same cases tests/test_analytics.py pins in Python — the twins must agree
describe('training planner', () => {
  const doc = load('qwen3-8b')
  const index = buildIndex(doc)
  it('LoRA r=16 attn-mlp on Qwen3-8B trains 43.6M params and fits 24 GB', () => {
    const p = planTraining(doc, index, req({}))
    expect(Math.round(p.trainableParams / 1e5) / 10).toBe(43.6)
    expect(p.gradBytesPerGpu).toBe(p.trainableParams * 2)
    expect(p.optimizerBytesPerGpu).toBe(p.trainableParams * 12)
    expect(p.fits).toBe(true)
  })
  it('QLoRA shrinks the frozen base below a third of bf16 and fits with headroom', () => {
    const l = planTraining(doc, index, req({}))
    const q = planTraining(doc, index, req({ method: 'qlora' }))
    expect(q.weightBytesPerGpu).toBeLessThan(l.weightBytesPerGpu / 3)
    expect(q.fits).toBe(true)
    expect(q.maxMicrobatch).toBeGreaterThanOrEqual(8)
  })
  it('full fine-tune follows the 16 B/param convention; ZeRO-3 divides by N', () => {
    const f = planTraining(doc, index, req({ method: 'full', gpuMemoryGb: 80 }))
    expect(f.weightBytesPerGpu + f.gradBytesPerGpu + f.optimizerBytesPerGpu).toBe(doc.params_total * 16)
    expect(f.fits).toBe(false)
    const z3 = planTraining(doc, index, req({ method: 'full', gpus: 8, gpuMemoryGb: 80, sharding: 'zero3' }))
    expect(z3.fits).toBe(true)
    expect(z3.weightBytesPerGpu).toBe(f.weightBytesPerGpu / 8)
  })
  it('checkpointing and flash attention shape activation memory', () => {
    const ck = planTraining(doc, index, req({}))
    const no = planTraining(doc, index, req({ gradCheckpoint: false }))
    expect(no.activationBytesPerGpu).toBeGreaterThan(3 * ck.activationBytesPerGpu)
    const nf = planTraining(doc, index, req({ gradCheckpoint: false, flashAttention: false }))
    expect(nf.activationBytesPerGpu - no.activationBytesPerGpu).toBe(36 * 32 * 2048 * 2048 * 2)
  })
})

describe('roofline throughput', () => {
  it('A100: Qwen3-8B decodes ≈72 tok/s at B=1 and amortizes weights at B=8', () => {
    const doc = load('qwen3-8b')
    const index = buildIndex(doc)
    const t = estimateThroughput(doc, index, 'A100 80GB', { T: 4096 })!
    expect(t.decodeTokPerSecB1).toBeGreaterThan(65)
    expect(t.decodeTokPerSecB1).toBeLessThan(80)
    expect(t.prefillTokPerSec).toBeGreaterThan(6000)
    const t8 = estimateThroughput(doc, index, 'A100 80GB', { T: 4096, B: 8 })!
    expect(t8.decodeTokPerSecAtB).toBeGreaterThan(5 * t.decodeTokPerSecB1)
    expect(estimateThroughput(doc, index, 'GTX 9999', {})).toBeNull()
  })
  it('MoE decode streams active weights only', () => {
    const doc = load('qwen3-235b-a22b')
    const t = estimateThroughput(doc, buildIndex(doc), 'H100 80GB', { tp: 8, T: 4096 })!
    expect(t.bytesReadPerToken).toBeLessThan(60e9)
    expect(t.notes.join()).toMatch(/MoE/)
  })
})
