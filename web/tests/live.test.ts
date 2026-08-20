import { describe, expect, it } from 'vitest'
import { gunzipSync } from 'node:zlib'
import { readFileSync } from 'node:fs'
import { LiveModel, topK } from '../src/live/model'
import { buildTokenizer, type TokenizerJSON } from '../src/live/tokenizer'
import { parseSafetensors, tensorF32 } from '../src/live/safetensors'

/** Live-mode engine vs real transformers: fixtures from scripts/gen_live_fixtures.py
 *  pin logits, attention maps, logit-lens argmaxes and greedy continuations. */

const FIX = (name: string) => new URL(`./fixtures/live/${name}`, import.meta.url)
const buf = (name: string): ArrayBuffer => {
  const b = readFileSync(FIX(name))
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)
}
const json = <T>(name: string): T => JSON.parse(readFileSync(FIX(name), 'utf-8'))

interface Expected {
  prompt: number[]
  logits_last: number[]
  attn_layer0: number[][][]
  attn_lastlayer_lastrow: number[][]
  lens_top1: number[]
  greedy: number[]
  top1_margin: number
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]))
  return m
}

function checkModel(name: 'llama-tiny' | 'gpt2-tiny') {
  const cfg = json<Record<string, unknown>>(`${name}.config.json`)
  const exp = json<Expected>(`${name}.expected.json`)
  const model = new LiveModel(cfg as never, buf(`${name}.safetensors`))
  const logits = model.forward(exp.prompt)

  it(`${name}: last-position logits match transformers`, () => {
    expect(logits.length).toBe(exp.logits_last.length)
    expect(maxAbsDiff(logits, exp.logits_last)).toBeLessThan(2e-4)
  })

  it(`${name}: layer-0 attention probabilities match`, () => {
    const m = model.attnMatrix(0)
    const S = exp.prompt.length
    expect(m.seq).toBe(S)
    for (let h = 0; h < m.heads; h++)
      for (let q = 0; q < S; q++)
        for (let k = 0; k <= q; k++)
          expect(Math.abs(m.data[h * S * S + q * S + k] - exp.attn_layer0[h][q][k])).toBeLessThan(2e-5)
  })

  it(`${name}: last layer's last attention row matches`, () => {
    const cfgL = (cfg.num_hidden_layers ?? cfg.n_layer) as number
    const m = model.attnMatrix(cfgL - 1)
    const S = exp.prompt.length
    for (let h = 0; h < m.heads; h++)
      for (let k = 0; k < S; k++)
        expect(Math.abs(m.data[h * S * S + (S - 1) * S + k] - exp.attn_lastlayer_lastrow[h][k])).toBeLessThan(2e-5)
  })

  it(`${name}: logit lens argmax per layer matches`, () => {
    const lens = model.logitLens()
    lens.forEach((lg, li) => expect(topK(lg, 1)[0].id).toBe(exp.lens_top1[li]))
  })

  it(`${name}: greedy continuation matches token for token (with KV cache)`, () => {
    const m2 = new LiveModel(cfg as never, buf(`${name}.safetensors`))
    let lg = m2.forward(exp.prompt)
    const out: number[] = []
    for (let i = 0; i < exp.greedy.length; i++) {
      const id = topK(lg, 1)[0].id
      out.push(id)
      lg = m2.forward([id])
    }
    expect(out).toEqual(exp.greedy)
  })
}

describe('live engine: llama architecture', () => checkModel('llama-tiny'))
describe('live engine: gpt2 architecture', () => checkModel('gpt2-tiny'))

describe('live engine: bf16 checkpoints', () => {
  it('llama-tiny.bf16 stays close to the f32 logits and agrees on top-1', () => {
    const cfg = json<Record<string, unknown>>('llama-tiny.config.json')
    const exp = json<Expected>('llama-tiny.expected.json')
    const model = new LiveModel(cfg as never, buf('llama-tiny.bf16.safetensors'))
    const logits = model.forward(exp.prompt)
    expect(maxAbsDiff(logits, exp.logits_last)).toBeLessThan(0.08)
    if (exp.top1_margin > 0.1) {
      expect(topK(logits, 1)[0].id).toBe(topK(Float32Array.from(exp.logits_last), 1)[0].id)
    }
  })
})

describe('safetensors reader', () => {
  it('parses shapes and converts bf16', () => {
    const f = parseSafetensors(buf('llama-tiny.safetensors'))
    const emb = f.tensors.get('model.embed_tokens.weight')!
    expect(emb.shape).toEqual([128, 32])
    const f32 = tensorF32(f, emb)
    const fb = parseSafetensors(buf('llama-tiny.bf16.safetensors'))
    const emb16 = tensorF32(fb, fb.tensors.get('model.embed_tokens.weight')!)
    expect(maxAbsDiff(f32, emb16)).toBeLessThan(0.02)
  })
})

describe('tokenizers vs Hugging Face `tokenizers`', () => {
  const expected = json<Record<string, { repo: string; cases: { text: string; ids: number[]; decoded: string }[] }>>('tok.expected.json')
  for (const key of ['tinyllama', 'smollm2', 'gpt2'] as const) {
    const tj = JSON.parse(gunzipSync(readFileSync(FIX(`tok-${key}.json.gz`))).toString()) as TokenizerJSON
    const tok = buildTokenizer(tj)
    it(`${key}: encodes ${expected[key].cases.length} strings identically`, () => {
      for (const c of expected[key].cases) {
        expect(tok.encode(c.text), JSON.stringify(c.text)).toEqual(c.ids)
      }
    })
    it(`${key}: decode matches the reference decoder`, () => {
      for (const c of expected[key].cases) {
        expect(tok.decode(c.ids), JSON.stringify(c.text)).toBe(c.decoded)
      }
    })
  }
})
