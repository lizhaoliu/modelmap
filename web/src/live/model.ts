/** CPU inference for Live mode (design doc §20): llama-family and GPT-2
 *  checkpoints, implemented directly on Float32Arrays so the forward pass can
 *  expose what frameworks hide — per-head attention probabilities and
 *  per-layer hidden states (for the logit lens). Runs inside a Web Worker;
 *  small models only (the supported list is capped by download size).
 *
 *  Correctness is pinned by fixtures generated with real `transformers`
 *  (scripts/gen_live_fixtures.py → web/tests/live.engine.test.ts): logits,
 *  attention maps and greedy continuations must match within f32 tolerance.
 */

import { numel, parseSafetensors, tensorF32 } from './safetensors'

export interface LlamaConfig {
  model_type: string
  hidden_size: number
  num_hidden_layers: number
  num_attention_heads: number
  num_key_value_heads?: number | null
  intermediate_size: number
  vocab_size: number
  rms_norm_eps?: number
  rope_theta?: number | null
  tie_word_embeddings?: boolean
  max_position_embeddings?: number
  head_dim?: number | null
  // gpt2 names
  n_embd?: number
  n_layer?: number
  n_head?: number
  n_positions?: number
  layer_norm_epsilon?: number
}

export interface ModelInfo {
  arch: 'llama' | 'gpt2'
  /** the config's model_type (llama, qwen2, qwen3, gpt2…) */
  modelType: string
  layers: number
  heads: number
  kvHeads: number
  hidden: number
  headDim: number
  vocab: number
  maxSeq: number
  params: number
}

interface Layer {
  // llama
  wIn?: Float32Array
  wPost?: Float32Array
  /** qwen3: per-head RMSNorm on q and k (length head_dim) */
  qNorm?: Float32Array
  kNorm?: Float32Array
  wq?: Float32Array
  wk?: Float32Array
  wv?: Float32Array
  wo?: Float32Array
  bq?: Float32Array
  bk?: Float32Array
  bv?: Float32Array
  wGate?: Float32Array
  wUp?: Float32Array
  wDown?: Float32Array
  // gpt2
  ln1w?: Float32Array
  ln1b?: Float32Array
  ln2w?: Float32Array
  ln2b?: Float32Array
  wqkv?: Float32Array // [3H, H] (transposed from Conv1D)
  bqkv?: Float32Array
  wproj?: Float32Array // [H, H]
  bproj?: Float32Array
  wfc?: Float32Array // [I, H]
  bfc?: Float32Array
  wfproj?: Float32Array // [H, I]
  bfproj?: Float32Array
  // kv cache [maxSeq, kvHeads * headDim]
  kCache: Float32Array
  vCache: Float32Array
  /** captured attention rows: for absolute position p, [heads, p+1] probs */
  attnRows: Float32Array[]
}

const MAX_SEQ = 512

// ------------------------------------------------------------- primitives

/** y[S,N] = x[S,K] · Wᵀ where W is [N,K] row-major (the safetensors linear layout). */
function matmul(x: Float32Array, S: number, K: number, W: Float32Array, N: number, y: Float32Array, bias?: Float32Array): void {
  for (let s = 0; s < S; s++) {
    const xo = s * K
    const yo = s * N
    for (let n = 0; n < N; n++) {
      const wo = n * K
      let acc = 0
      let k = 0
      const kEnd = K - 3
      for (; k < kEnd; k += 4) {
        acc +=
          x[xo + k] * W[wo + k] +
          x[xo + k + 1] * W[wo + k + 1] +
          x[xo + k + 2] * W[wo + k + 2] +
          x[xo + k + 3] * W[wo + k + 3]
      }
      for (; k < K; k++) acc += x[xo + k] * W[wo + k]
      y[yo + n] = acc + (bias ? bias[n] : 0)
    }
  }
}

function rmsnorm(x: Float32Array, S: number, H: number, w: Float32Array, eps: number, y: Float32Array): void {
  for (let s = 0; s < S; s++) {
    const o = s * H
    let ss = 0
    for (let i = 0; i < H; i++) ss += x[o + i] * x[o + i]
    const inv = 1 / Math.sqrt(ss / H + eps)
    for (let i = 0; i < H; i++) y[o + i] = x[o + i] * inv * w[i]
  }
}

function layernorm(x: Float32Array, S: number, H: number, w: Float32Array, b: Float32Array, eps: number, y: Float32Array): void {
  for (let s = 0; s < S; s++) {
    const o = s * H
    let mean = 0
    for (let i = 0; i < H; i++) mean += x[o + i]
    mean /= H
    let vs = 0
    for (let i = 0; i < H; i++) {
      const d = x[o + i] - mean
      vs += d * d
    }
    const inv = 1 / Math.sqrt(vs / H + eps)
    for (let i = 0; i < H; i++) y[o + i] = (x[o + i] - mean) * inv * w[i] + b[i]
  }
}

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI)
function geluNew(v: number): number {
  return 0.5 * v * (1 + Math.tanh(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)))
}
function silu(v: number): number {
  return v / (1 + Math.exp(-v))
}

export function softmaxInPlace(x: Float32Array, off: number, n: number): void {
  let max = -Infinity
  for (let i = 0; i < n; i++) if (x[off + i] > max) max = x[off + i]
  let sum = 0
  for (let i = 0; i < n; i++) {
    const e = Math.exp(x[off + i] - max)
    x[off + i] = e
    sum += e
  }
  for (let i = 0; i < n; i++) x[off + i] /= sum
}

// ------------------------------------------------------------------ model

export class LiveModel {
  info: ModelInfo
  private layers: Layer[] = []
  private embed!: Float32Array // [V, H]
  private wpe?: Float32Array // gpt2 positions [P, H]
  private normW!: Float32Array
  private normB?: Float32Array
  private lmHead!: Float32Array // [V, H]
  private eps: number
  private theta: number
  /** hidden state of the LAST position after each layer (for the logit lens) */
  lensHidden: Float32Array[] = []
  seq = 0
  /** "layer:head" pairs whose attention output is zeroed (design doc §24) */
  ablated = new Set<string>()

  constructor(cfg: LlamaConfig, buf: ArrayBuffer) {
    const file = parseSafetensors(buf)
    const arch: 'llama' | 'gpt2' = cfg.model_type === 'gpt2' ? 'gpt2' : 'llama'
    const hidden = cfg.hidden_size ?? cfg.n_embd!
    const layers = cfg.num_hidden_layers ?? cfg.n_layer!
    const heads = cfg.num_attention_heads ?? cfg.n_head!
    const kvHeads = cfg.num_key_value_heads ?? heads
    const headDim = cfg.head_dim ?? Math.floor(hidden / heads)
    this.eps = cfg.rms_norm_eps ?? cfg.layer_norm_epsilon ?? 1e-5
    this.theta = cfg.rope_theta ?? 10000
    const maxSeq = Math.min(MAX_SEQ, cfg.max_position_embeddings ?? cfg.n_positions ?? MAX_SEQ)
    let params = 0
    for (const m of file.tensors.values()) params += numel(m.shape)

    const pick = (...names: string[]): Float32Array | undefined => {
      for (const n of names) {
        for (const cand of [n, `model.${n}`, `transformer.${n}`, n.replace(/^model\./, '')]) {
          const m = file.tensors.get(cand)
          if (m) return tensorF32(file, m)
        }
      }
      return undefined
    }
    const need = (...names: string[]): Float32Array => {
      const t = pick(...names)
      if (!t) throw new Error(`checkpoint is missing tensor ${names[0]}`)
      return t
    }

    if (arch === 'llama') {
      this.embed = need('model.embed_tokens.weight', 'embed_tokens.weight')
      this.normW = need('model.norm.weight', 'norm.weight')
      this.lmHead = cfg.tie_word_embeddings ? this.embed : need('lm_head.weight')
      for (let i = 0; i < layers; i++) {
        const p = `model.layers.${i}`
        this.layers.push({
          wIn: need(`${p}.input_layernorm.weight`),
          wPost: need(`${p}.post_attention_layernorm.weight`),
          wq: need(`${p}.self_attn.q_proj.weight`),
          wk: need(`${p}.self_attn.k_proj.weight`),
          wv: need(`${p}.self_attn.v_proj.weight`),
          wo: need(`${p}.self_attn.o_proj.weight`),
          bq: pick(`${p}.self_attn.q_proj.bias`),
          bk: pick(`${p}.self_attn.k_proj.bias`),
          bv: pick(`${p}.self_attn.v_proj.bias`),
          qNorm: pick(`${p}.self_attn.q_norm.weight`),
          kNorm: pick(`${p}.self_attn.k_norm.weight`),
          wGate: need(`${p}.mlp.gate_proj.weight`),
          wUp: need(`${p}.mlp.up_proj.weight`),
          wDown: need(`${p}.mlp.down_proj.weight`),
          kCache: new Float32Array(maxSeq * kvHeads * headDim),
          vCache: new Float32Array(maxSeq * kvHeads * headDim),
          attnRows: [],
        })
      }
    } else {
      this.embed = need('wte.weight')
      this.wpe = need('wpe.weight')
      this.normW = need('ln_f.weight')
      this.normB = need('ln_f.bias')
      this.lmHead = this.embed // gpt2 is always tied
      for (let i = 0; i < layers; i++) {
        const p = `h.${i}`
        // Conv1D stores [in, out] — transpose once so matmul is uniform
        const t = (w: Float32Array, nIn: number, nOut: number) => {
          const out = new Float32Array(w.length)
          for (let a = 0; a < nIn; a++) for (let b = 0; b < nOut; b++) out[b * nIn + a] = w[a * nOut + b]
          return out
        }
        const inter = (cfg.intermediate_size ?? 4 * hidden)
        this.layers.push({
          ln1w: need(`${p}.ln_1.weight`),
          ln1b: need(`${p}.ln_1.bias`),
          ln2w: need(`${p}.ln_2.weight`),
          ln2b: need(`${p}.ln_2.bias`),
          wqkv: t(need(`${p}.attn.c_attn.weight`), hidden, 3 * hidden),
          bqkv: need(`${p}.attn.c_attn.bias`),
          wproj: t(need(`${p}.attn.c_proj.weight`), hidden, hidden),
          bproj: need(`${p}.attn.c_proj.bias`),
          wfc: t(need(`${p}.mlp.c_fc.weight`), hidden, inter),
          bfc: need(`${p}.mlp.c_fc.bias`),
          wfproj: t(need(`${p}.mlp.c_proj.weight`), inter, hidden),
          bfproj: need(`${p}.mlp.c_proj.bias`),
          kCache: new Float32Array(maxSeq * heads * headDim),
          vCache: new Float32Array(maxSeq * heads * headDim),
          attnRows: [],
        })
      }
    }
    this.info = {
      arch, modelType: cfg.model_type, layers, heads,
      kvHeads: arch === 'gpt2' ? heads : kvHeads,
      hidden, headDim, vocab: cfg.vocab_size, maxSeq, params,
    }
  }

  reset(): void {
    this.seq = 0
    this.lensHidden = []
    for (const l of this.layers) l.attnRows = []
  }

  private rope(vec: Float32Array, S: number, nHeads: number, past: number): void {
    const dh = this.info.headDim
    const half = dh >> 1
    for (let s = 0; s < S; s++) {
      const p = past + s
      for (let h = 0; h < nHeads; h++) {
        const o = (s * nHeads + h) * dh
        for (let i = 0; i < half; i++) {
          const freq = Math.pow(this.theta, -(2 * i) / dh)
          const cos = Math.cos(p * freq)
          const sin = Math.sin(p * freq)
          const a = vec[o + i]
          const b = vec[o + i + half]
          vec[o + i] = a * cos - b * sin
          vec[o + i + half] = b * cos + a * sin
        }
      }
    }
  }

  /** Run `ids` through the model appended to the cache; returns logits of the
   *  last position (length vocab). Captures attention rows + lens hiddens. */
  forward(ids: number[]): Float32Array {
    const { arch, hidden: H, heads: nh, kvHeads: nkv, headDim: dh, vocab: V, layers: L, maxSeq } = this.info
    const S = ids.length
    const past = this.seq
    if (past + S > maxSeq) throw new Error(`sequence exceeds the ${maxSeq}-token cap`)
    const x = new Float32Array(S * H)
    for (let s = 0; s < S; s++) {
      x.set(this.embed.subarray(ids[s] * H, (ids[s] + 1) * H), s * H)
      if (this.wpe) {
        const o = s * H
        const w = this.wpe.subarray((past + s) * H, (past + s + 1) * H)
        for (let i = 0; i < H; i++) x[o + i] += w[i]
      }
    }
    const h1 = new Float32Array(S * H)
    const q = new Float32Array(S * nh * dh)
    const kv = new Float32Array(S * nkv * dh)
    const attnOut = new Float32Array(S * nh * dh)
    const scores = new Float32Array(maxSeq)
    const scale = 1 / Math.sqrt(dh)
    const group = nh / nkv
    this.lensHidden = []

    for (let li = 0; li < L; li++) {
      const l = this.layers[li]
      // ---- attention
      if (arch === 'llama') {
        rmsnorm(x, S, H, l.wIn!, this.eps, h1)
        matmul(h1, S, H, l.wq!, nh * dh, q, l.bq)
        matmul(h1, S, H, l.wk!, nkv * dh, kv, l.bk)
        // qwen3-style per-head q/k RMSNorm, before rope
        if (l.qNorm) for (let s2 = 0; s2 < S; s2++) rmsnorm(q.subarray(s2 * nh * dh, (s2 + 1) * nh * dh), nh, dh, l.qNorm, this.eps, q.subarray(s2 * nh * dh, (s2 + 1) * nh * dh))
        if (l.kNorm) for (let s2 = 0; s2 < S; s2++) rmsnorm(kv.subarray(s2 * nkv * dh, (s2 + 1) * nkv * dh), nkv, dh, l.kNorm, this.eps, kv.subarray(s2 * nkv * dh, (s2 + 1) * nkv * dh))
        this.rope(q, S, nh, past)
        this.rope(kv, S, nkv, past)
        for (let s = 0; s < S; s++) l.kCache.set(kv.subarray(s * nkv * dh, (s + 1) * nkv * dh), (past + s) * nkv * dh)
        matmul(h1, S, H, l.wv!, nkv * dh, kv, l.bv)
        for (let s = 0; s < S; s++) l.vCache.set(kv.subarray(s * nkv * dh, (s + 1) * nkv * dh), (past + s) * nkv * dh)
      } else {
        layernorm(x, S, H, l.ln1w!, l.ln1b!, this.eps, h1)
        const qkv = new Float32Array(S * 3 * H)
        matmul(h1, S, H, l.wqkv!, 3 * H, qkv, l.bqkv)
        for (let s = 0; s < S; s++) {
          q.set(qkv.subarray(s * 3 * H, s * 3 * H + H), s * H)
          l.kCache.set(qkv.subarray(s * 3 * H + H, s * 3 * H + 2 * H), (past + s) * H)
          l.vCache.set(qkv.subarray(s * 3 * H + 2 * H, s * 3 * H + 3 * H), (past + s) * H)
        }
      }
      for (let s = 0; s < S; s++) {
        const p = past + s
        const row = new Float32Array(nh * (p + 1))
        for (let h = 0; h < nh; h++) {
          const kvh = arch === 'gpt2' ? h : Math.floor(h / group)
          const qo = (s * nh + h) * dh
          for (let j = 0; j <= p; j++) {
            const ko = (j * nkv + kvh) * dh
            let acc = 0
            for (let d = 0; d < dh; d++) acc += q[qo + d] * l.kCache[ko + d]
            scores[j] = acc * scale
          }
          softmaxInPlace(scores, 0, p + 1)
          row.set(scores.subarray(0, p + 1), h * (p + 1))
          const ao = (s * nh + h) * dh
          attnOut.fill(0, ao, ao + dh)
          if (!this.ablated.has(`${li}:${h}`)) {
            for (let j = 0; j <= p; j++) {
              const vo = (j * nkv + kvh) * dh
              const w = scores[j]
              for (let d = 0; d < dh; d++) attnOut[ao + d] += w * l.vCache[vo + d]
            }
          }
        }
        l.attnRows.push(row)
      }
      const proj = new Float32Array(S * H)
      matmul(attnOut, S, nh * dh, arch === 'llama' ? l.wo! : l.wproj!, H, proj, arch === 'gpt2' ? l.bproj : undefined)
      for (let i = 0; i < S * H; i++) x[i] += proj[i]
      // ---- mlp
      if (arch === 'llama') {
        rmsnorm(x, S, H, l.wPost!, this.eps, h1)
        const I = l.wGate!.length / H
        const g = new Float32Array(S * I)
        const u = new Float32Array(S * I)
        matmul(h1, S, H, l.wGate!, I, g)
        matmul(h1, S, H, l.wUp!, I, u)
        for (let i = 0; i < S * I; i++) g[i] = silu(g[i]) * u[i]
        matmul(g, S, I, l.wDown!, H, proj)
      } else {
        layernorm(x, S, H, l.ln2w!, l.ln2b!, this.eps, h1)
        const I = l.wfc!.length / H
        const f = new Float32Array(S * I)
        matmul(h1, S, H, l.wfc!, I, f, l.bfc)
        for (let i = 0; i < S * I; i++) f[i] = geluNew(f[i])
        matmul(f, S, I, l.wfproj!, H, proj, l.bfproj)
      }
      for (let i = 0; i < S * H; i++) x[i] += proj[i]
      this.lensHidden.push(x.slice((S - 1) * H, S * H))
    }
    this.seq = past + S

    const xn = new Float32Array(H)
    this.finalNorm(x.subarray((S - 1) * H, S * H), xn)
    const logits = new Float32Array(V)
    matmul(xn, 1, H, this.lmHead, V, logits)
    return logits
  }

  finalNorm(x: Float32Array, out: Float32Array): void {
    const H = this.info.hidden
    if (this.normB) layernorm(x, 1, H, this.normW, this.normB, this.eps, out)
    else rmsnorm(x, 1, H, this.normW, this.eps, out)
  }

  /** logit lens: project each layer's last-position hidden through the final
   *  norm + head; returns per-layer logits (computed on demand — full vocab
   *  matmuls are not free). */
  logitLens(): Float32Array[] {
    const { hidden: H, vocab: V } = this.info
    const out: Float32Array[] = []
    const xn = new Float32Array(H)
    for (const hid of this.lensHidden) {
      this.finalNorm(hid, xn)
      const logits = new Float32Array(V)
      matmul(xn, 1, H, this.lmHead, V, logits)
      out.push(logits)
    }
    return out
  }

  /** dense [heads, T, T] attention matrix for one layer (rows are ragged in
   *  storage; padded with zeros above the diagonal). */
  attnMatrix(layer: number): { heads: number; seq: number; data: Float32Array } {
    const l = this.layers[layer]
    const T = this.seq
    const nh = this.info.heads
    const data = new Float32Array(nh * T * T)
    for (let p = 0; p < l.attnRows.length; p++) {
      const row = l.attnRows[p]
      const width = row.length / nh
      for (let h = 0; h < nh; h++) {
        data.set(row.subarray(h * width, (h + 1) * width), h * T * T + p * T)
      }
    }
    return { heads: nh, seq: T, data }
  }

  /** attention-pattern statistics per head of one layer, over the current
   *  sequence: how much each head looks at the previous token, position 0
   *  (the "attention sink"), itself, and how spread out it is. */
  headStats(layer: number): { prev: number; first: number; self: number; entropy: number; tag: string }[] {
    const l = this.layers[layer]
    const nh = this.info.heads
    const out: { prev: number; first: number; self: number; entropy: number; tag: string }[] = []
    for (let h = 0; h < nh; h++) {
      let prev = 0
      let first = 0
      let selfW = 0
      let ent = 0
      let n = 0
      for (let p = 1; p < l.attnRows.length; p++) {
        const row = l.attnRows[p]
        const width = row.length / nh
        const o = h * width
        prev += row[o + p - 1]
        first += row[o]
        selfW += row[o + p]
        let e = 0
        for (let j = 0; j <= p; j++) {
          const v = row[o + j]
          if (v > 1e-9) e -= v * Math.log(v)
        }
        ent += e / Math.log(p + 1) // normalized: 1 = uniform
        n++
      }
      if (!n) {
        out.push({ prev: 0, first: 0, self: 0, entropy: 0, tag: '' })
        continue
      }
      prev /= n
      first /= n
      selfW /= n
      ent /= n
      const tag =
        prev > 0.4 ? 'prev-token' : first > 0.55 ? 'sink' : selfW > 0.4 ? 'self' : ent > 0.85 ? 'broad' : ''
      out.push({ prev, first, self: selfW, entropy: ent, tag })
    }
    return out
  }
}

// ------------------------------------------------------------- sampling

export function topK(logits: Float32Array, k: number): { id: number; logit: number }[] {
  const out: { id: number; logit: number }[] = []
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i]
    if (out.length < k) {
      out.push({ id: i, logit: v })
      out.sort((a, b) => b.logit - a.logit)
    } else if (v > out[k - 1].logit) {
      out[k - 1] = { id: i, logit: v }
      out.sort((a, b) => b.logit - a.logit)
    }
  }
  return out
}

export function softmaxOver(entries: { id: number; logit: number }[], temperature: number): { id: number; p: number }[] {
  const t = Math.max(temperature, 1e-6)
  const max = entries[0]?.logit ?? 0
  const exps = entries.map((e) => Math.exp((e.logit - max) / t))
  const sum = exps.reduce((a, b) => a + b, 0)
  return entries.map((e, i) => ({ id: e.id, p: exps[i] / sum }))
}

export function sample(logits: Float32Array, temperature: number, k: number): number {
  const top = topK(logits, Math.max(1, k))
  if (temperature <= 0) return top[0].id
  const probs = softmaxOver(top, temperature)
  let r = Math.random()
  for (const { id, p } of probs) {
    r -= p
    if (r <= 0) return id
  }
  return probs[probs.length - 1].id
}
