import { describe, expect, it } from 'vitest'
import { gunzipSync } from 'node:zlib'
import { readFileSync } from 'node:fs'
import { buildIndex, type GraphDoc } from '../src/types'
import { computeCosts, DEFAULT_ASSUMPTIONS, lensValue, textConfig } from '../src/analytics/cost'

const load = (name: string): GraphDoc =>
  JSON.parse(gunzipSync(readFileSync(new URL(`./fixtures/${name}.graph.json.gz`, import.meta.url))).toString())

const run = (name: string, T = 7) => {
  const doc = load(name)
  const index = buildIndex(doc)
  const rep = computeCosts(doc, index, { ...DEFAULT_ASSUMPTIONS, T })
  return { doc, index, rep }
}

describe('cost lens', () => {
  it('Qwen3-235B-A22B: ≈22B active params per token (the A22B), all 235B stored', () => {
    const { doc, rep } = run('qwen3-235b-a22b')
    expect(rep.root.activeParams / 1e9).toBeGreaterThan(20)
    expect(rep.root.activeParams / 1e9).toBeLessThan(24)
    expect(rep.root.paramBytes / 2 / 1e9).toBeCloseTo(doc.params_total / 1e9, 0)
    expect(rep.notes.join()).toMatch(/8 of 128 experts/)
  })

  it('dense: MACs per token ≈ non-embedding params (the 2·N FLOPs rule) at short T', () => {
    for (const name of ['qwen3-8b', 'gpt2']) {
      const { doc, rep } = run(name, 7)
      const perToken = rep.root.macs / (7 * DEFAULT_ASSUMPTIONS.B)
      const own = (n: (typeof doc.nodes)[0]) =>
        Object.values(n.weight_shapes ?? {}).reduce((s, w) => s + w.reduce((a, b) => a * b, 1), 0)
      const embed = doc.nodes.filter((n) => n.kind === 'embedding').reduce((s, n) => s + own(n), 0)
      const head = doc.nodes.filter((n) => n.kind === 'head').reduce((s, n) => s + own(n), 0)
      // a tied lm_head shares the embedding matrix: real compute, but not in params_total
      const tied = doc.config.tie_word_embeddings === true
      const expected = doc.params_total - embed + (tied ? head : 0)
      expect(perToken / expected).toBeGreaterThan(0.9)
      expect(perToken / expected).toBeLessThan(1.1)
    }
  })

  it('MoE: MACs per token ≈ active params, far below total', () => {
    const { rep } = run('mixtral-8x7b', 7)
    const perToken = rep.root.macs / 7
    expect(perToken / rep.root.activeParams).toBeGreaterThan(0.75)
    expect(perToken / rep.root.activeParams).toBeLessThan(1.35)
    expect(rep.root.activeParams / 1e9).toBeGreaterThan(11)
    expect(rep.root.activeParams / 1e9).toBeLessThan(14)
  })

  it('KV cache per token from config (GQA) and MLA for DeepSeek', () => {
    const q = run('qwen3-8b')
    // 36 layers × 2 × 8 kv heads × 128 × 2 bytes = 147 456
    expect(q.rep.root.kvPerToken).toBe(36 * 2 * 8 * 128 * 2)
    expect(q.rep.kvLayers).toBe(36)
    const d = run('deepseek-v3.1')
    // 61 layers × (512 + 64) × 2 bytes
    expect(d.rep.root.kvPerToken).toBe(61 * (512 + 64) * 2)
  })

  it('attention share grows with T (quadratic core) and shapes scale', () => {
    const short = run('qwen3-8b', 128)
    const long = run('qwen3-8b', 32768)
    const attn = (r: ReturnType<typeof run>) => {
      const a = r.rep.byNode.get('model.layers.0.self_attn')!
      const blk = r.rep.byNode.get('model.layers.0')!
      return a.macs / blk.macs
    }
    expect(attn(long)).toBeGreaterThan(attn(short) * 2)
    // per-token activation memory of a norm output scales linearly with T
    const nShort = short.rep.byNode.get('model.layers.0.input_layernorm')!.actBytes
    const nLong = long.rep.byNode.get('model.layers.0.input_layernorm')!.actBytes
    expect(nLong / nShort).toBeCloseTo(32768 / 128, 5)
  })
})

describe('tied embeddings', () => {
  it('gpt2: a tied lm_head is stored once — active params equal params_total', () => {
    const { doc, rep } = run('gpt2')
    expect(rep.root.activeParams).toBe(doc.params_total)
    expect(rep.root.paramBytes).toBe(doc.params_total * 4)
    expect(rep.notes.join()).toMatch(/tied/)
  })
})

describe('quantized dtypes', () => {
  it('GGUF quant types carry fractional bytes per weight and format with bpw', async () => {
    const { bytesOf, fmtDtype } = await import('../src/analytics/cost')
    expect(bytesOf('q4_k', 2)).toBeCloseTo(4.5 / 8, 6)
    expect(bytesOf('Q8_0', 2)).toBeCloseTo(8.5 / 8, 6)
    expect(bytesOf('nf4', 2)).toBe(0.5)
    expect(bytesOf('mystery', 2)).toBe(2)
    expect(fmtDtype('q4_k')).toBe('Q4_K · 4.5 bpw')
    expect(fmtDtype('bf16')).toBe('bf16')
    const doc = load('qwen3-8b')
    for (const n of doc.nodes) if (n.weight_shapes) n.dtype = 'q4_k'
    const rep = computeCosts(doc, buildIndex(doc), DEFAULT_ASSUMPTIONS)
    // 8.19B params at 4.5 bits ≈ 4.3 GiB of weights
    expect(rep.root.paramBytes / 2 ** 30).toBeGreaterThan(4.2)
    expect(rep.root.paramBytes / 2 ** 30).toBeLessThan(4.4)
  })
})

describe('serving precision + vram lens (§26)', () => {
  it('"serve weights as int4" re-prices every weight tensor; activations and KV keep the activation dtype', () => {
    const doc = load('qwen3-8b')
    const index = buildIndex(doc)
    const stored = computeCosts(doc, index, { ...DEFAULT_ASSUMPTIONS, T: 4096 })
    const int4 = computeCosts(doc, index, { ...DEFAULT_ASSUMPTIONS, T: 4096, weights: 'int4', weightBytes: 0.5 })
    expect(stored.root.paramBytes / 1e9).toBeCloseTo(16.38, 1)
    expect(int4.root.paramBytes / 1e9).toBeCloseTo(4.095, 2) // pinned with analytics.py
    expect(int4.root.kvPerToken).toBe(stored.root.kvPerToken)
    expect(int4.root.actBytes).toBe(stored.root.actBytes)
  })

  it('vram lens value = weights + KV cache at T × B, so attention grows with context and MLP does not', () => {
    const doc = load('qwen3-8b')
    const index = buildIndex(doc)
    const attn = doc.nodes.find((n) => n.id === 'model.layers.0.self_attn')!
    const mlp = doc.nodes.find((n) => n.id === 'model.layers.0.mlp')!
    const at = (T: number) => {
      const rep = computeCosts(doc, index, { ...DEFAULT_ASSUMPTIONS, T })
      return { attn: lensValue('vram', attn, rep.byNode.get(attn.id), rep.assumptions), mlp: lensValue('vram', mlp, rep.byNode.get(mlp.id), rep.assumptions), rep }
    }
    const short = at(4096), long = at(131072)
    expect(long.mlp).toBe(short.mlp)
    expect(long.attn).toBeGreaterThan(short.attn)
    // one layer's KV at 128k: 2 × 8 kv heads × 128 × 2 bytes × 131072 = 512 MiB on top of its weights
    expect(long.attn - short.attn).toBeCloseTo(2 * 8 * 128 * 2 * (131072 - 4096), -3)
    // at 128k the cache for the whole model outweighs the bf16 weights (18 GB vs 16.4 GB)
    expect(long.rep.root.kvPerToken * 131072).toBeGreaterThan(long.rep.root.paramBytes)
  })

  it('VLM configs nest the language model under text_config — KV and attention compute still count', () => {
    const doc = load('qwen2.5-vl-3b')
    const index = buildIndex(doc)
    const rep = computeCosts(doc, index, { ...DEFAULT_ASSUMPTIONS, T: 4096 })
    // 36 layers × 2 × 2 kv heads × 128 × 2 bytes
    expect(rep.root.kvPerToken).toBe(36 * 2 * 2 * 128 * 2)
    expect(rep.kvLayers).toBe(36)
    expect(textConfig(doc).num_key_value_heads).toBe(2)
  })
})
