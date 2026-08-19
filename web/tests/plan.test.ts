import { describe, expect, it } from 'vitest'
import { gunzipSync } from 'node:zlib'
import { readFileSync } from 'node:fs'
import { buildIndex, type GraphDoc } from '../src/types'
import { planServing, type PlanRequest } from '../src/analytics/plan'

const load = (name: string): GraphDoc =>
  JSON.parse(gunzipSync(readFileSync(new URL(`./fixtures/${name}.graph.json.gz`, import.meta.url))).toString())
const req = (o: Partial<PlanRequest>): PlanRequest => ({
  gpus: 1, gpuMemoryGb: 24, tp: 1, pp: 1, T: 4096, B: 1, dtypeLabel: 'bf16', bytes: 2, headroom: 0.1, ...o,
})

// the same cases tests/test_analytics.py::test_planner_tp_pp_fits_and_max_context pins in Python
describe('serving planner', () => {
  const doc = load('qwen3-8b')
  const index = buildIndex(doc)
  it('single 24 GB GPU fits Qwen3-8B at T=4096 and reports a KV-limited max context', () => {
    const one = planServing(doc, index, req({}))
    expect(one.fits).toBe(true)
    expect(one.stages).toHaveLength(1)
    expect(one.weightBytes).toBeGreaterThan(15e9)
    expect(one.weightBytes).toBeLessThan(17e9)
    expect(one.maxContextTokens).toBeGreaterThan(4096)
  })
  it('pp=2 splits the 36 layers into contiguous halves with B·T·hidden·bytes boundary traffic', () => {
    const two = planServing(doc, index, req({ gpus: 2, pp: 2 }))
    expect(two.stages).toHaveLength(2)
    expect(two.stages[0].layers[0]).toBe(0)
    expect(two.stages[1].layers[1]).toBe(35)
    expect(two.stages[0].layerCount + two.stages[1].layerCount).toBe(36)
    expect(two.stages[0].boundaryBytesOut).toBe(4096 * 4096 * 2)
    const one = planServing(doc, index, req({}))
    expect(two.maxContextTokens).toBeGreaterThan(one.maxContextTokens)
  })
  it('tp=2 halves per-GPU weights; an 8 GB GPU does not fit', () => {
    const one = planServing(doc, index, req({}))
    const tp2 = planServing(doc, index, req({ gpus: 2, tp: 2 }))
    expect(Math.abs(tp2.stages[0].weightBytesPerGpu - one.stages[0].weightBytesPerGpu / 2)).toBeLessThan(1)
    const tiny = planServing(doc, index, req({ gpuMemoryGb: 8 }))
    expect(tiny.fits).toBe(false)
    expect(tiny.maxContextTokens).toBe(0)
  })
  it('DeepSeek two-stack layers still partition all 61 layers across 8 stages', () => {
    const ds = load('deepseek-v3.1')
    const p = planServing(ds, buildIndex(ds), req({ gpus: 8, pp: 8, gpuMemoryGb: 80 }))
    expect(p.stages.reduce((s, st) => s + st.layerCount, 0)).toBe(61)
  })
})
