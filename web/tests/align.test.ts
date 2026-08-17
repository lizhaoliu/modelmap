import { describe, expect, it } from 'vitest'
import { gunzipSync } from 'node:zlib'
import { readFileSync } from 'node:fs'
import { buildIndex, type GraphDoc } from '../src/types'
import { align } from '../src/compare/align'

const load = (name: string): GraphDoc =>
  JSON.parse(gunzipSync(readFileSync(new URL(`./fixtures/${name}.graph.json.gz`, import.meta.url))).toString())
const cmp = (x: string, y: string) => {
  const a = load(x), b = load(y)
  return align(a, buildIndex(a), b, buildIndex(b))
}
const byField = (al: ReturnType<typeof align>, id: string) =>
  Object.fromEntries((al.byA.get(id)?.changes ?? []).map((c) => [c.field, [c.a, c.b]]))

describe('compare alignment', () => {
  it('a base and its fine-tune: identical structure, nothing changed', () => {
    const al = cmp('qwen3-8b-base', 'qwen3-8b')
    expect(al.counts.added + al.counts.removed).toBe(0)
    // only tie/dtype-level config or nothing; no shape/param changes anywhere
    const structural = al.pairs.filter((p) => p.changes.some((c) => /params|weight|input|output|repeats|class/.test(c.field)))
    expect(structural).toHaveLength(0)
  })

  it('gpt2 vs gpt2-medium: same design, only widths and counts change; everything pairs by path', () => {
    const al = cmp('gpt2', 'gpt2-medium')
    expect(al.counts.added + al.counts.removed).toBe(0)
    expect(byField(al, 'transformer.h.0')['repeats']).toEqual(['12', '24'])
    expect(byField(al, 'transformer.wte')['weight weight']).toEqual(['[50257,768]', '[50257,1024]'])
    expect(al.configDiff.map((c) => c.field)).toEqual(expect.arrayContaining(['n_embd', 'n_head', 'n_layer']))
  })

  it('Qwen2.5-7B vs Qwen3-8B: q/k norms added, biases dropped, ffn and layer count changed', () => {
    const al = cmp('qwen2.5-7b', 'qwen3-8b')
    const added = al.pairs.filter((p) => p.status === 'added').map((p) => p.b!.id)
    expect(added).toEqual(expect.arrayContaining(['model.layers.0.self_attn.q_norm', 'model.layers.0.self_attn.k_norm']))
    expect(byField(al, 'model.layers.0')['repeats']).toEqual(['28', '36'])
    // Qwen2.5 projections carry a bias; Qwen3's do not
    expect(byField(al, 'model.layers.0.self_attn.q_proj')['bias']).toEqual(['True', 'False'])
    expect(byField(al, 'model.layers.0.mlp.gate_proj')['weight weight']).toEqual(['[18944,3584]', '[12288,4096]'])
    // every Qwen2.5 layer module found its twin (nothing removed)
    expect(al.counts.removed).toBe(0)
    expect(al.byA.get('model.layers')?.dirty).toBe(true)
    expect(al.byA.get('model.norm')?.status).toBe('changed') // width changed
  })

  it('unrelated architectures still pair by role rather than mis-pairing', () => {
    const al = cmp('gpt2', 'qwen3-8b')
    // embeddings pair by role: wte ↔ embed_tokens
    const wte = al.byA.get('transformer.wte')
    expect(wte?.b?.id).toBe('model.embed_tokens')
    // gpt2's Conv1D projections and Qwen's Linear ones do not silently pair as identical
    expect(al.counts.same).toBeLessThan(al.pairs.length / 2)
  })
})
