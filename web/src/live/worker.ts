/// <reference lib="webworker" />
/** Live-mode worker (design doc §20): downloads a checkpoint from the Hub
 *  (the *browser* fetches the weights — the modelmap server never sees them),
 *  builds the model, and answers run / generate / attn / lens requests.
 *  All heavy math stays off the main thread. */

import { LiveModel, sample, topK, type LlamaConfig } from './model'
import { buildTokenizer, type Tokenizer, type TokenizerJSON } from './tokenizer'

interface LoadMsg { t: 'load'; repo: string; revision?: string }
interface RunMsg { t: 'run'; prompt: string }
interface GenMsg { t: 'gen'; maxNew: number; temperature: number; topK: number }
interface AttnMsg { t: 'attn'; layer: number }
interface LensMsg { t: 'lens' }
interface StopMsg { t: 'stop' }
interface AblateMsg { t: 'ablate'; layer: number; head: number; on: boolean }
interface HeadStatsMsg { t: 'headstats'; layer: number }
export type InMsg = LoadMsg | RunMsg | GenMsg | AttnMsg | LensMsg | StopMsg | AblateMsg | HeadStatsMsg

export interface TopEntry { id: number; tok: string; p: number }

let model: LiveModel | null = null
let tokenizer: Tokenizer | null = null
let ids: number[] = []
let lastLogits: Float32Array | null = null
let stopFlag = false

const post = (msg: unknown, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(msg, transfer)

function hubUrl(repo: string, file: string, revision = 'main'): string {
  return `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/${file}`
}

async function fetchJson<T>(url: string): Promise<T | null> {
  const r = await fetch(url)
  if (!r.ok) return null
  return r.json()
}

async function fetchWithProgress(url: string, phase: string): Promise<ArrayBuffer> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${url.split('/').pop()}: HTTP ${r.status}`)
  const total = Number(r.headers.get('content-length')) || 0
  if (!r.body) return r.arrayBuffer()
  const reader = r.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  let lastPost = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.byteLength
    const now = Date.now()
    if (now - lastPost > 120) {
      lastPost = now
      post({ t: 'progress', phase, loaded, total })
    }
  }
  post({ t: 'progress', phase, loaded, total: total || loaded })
  const out = new Uint8Array(loaded)
  let off = 0
  for (const c of chunks) {
    out.set(c, off)
    off += c.byteLength
  }
  return out.buffer
}

/** full-vocab softmax, then the top k entries */
function topkProbs(logits: Float32Array, k: number): TopEntry[] {
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i]
  let sum = 0
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max)
  return topK(logits, k).map((e) => ({
    id: e.id,
    tok: tokenizer ? tokenizer.pretty(e.id) : String(e.id),
    p: Math.exp(e.logit - max) / sum,
  }))
}

async function load(msg: LoadMsg): Promise<void> {
  const rev = msg.revision ?? 'main'
  post({ t: 'progress', phase: 'config', loaded: 0, total: 0 })
  const cfg = await fetchJson<LlamaConfig>(hubUrl(msg.repo, 'config.json', rev))
  if (!cfg) throw new Error('could not fetch config.json')
  const tj = await fetchJson<TokenizerJSON>(hubUrl(msg.repo, 'tokenizer.json', rev))
  if (!tj) throw new Error('this repo has no tokenizer.json — Live mode needs one')
  const tcfg = await fetchJson<{ add_bos_token?: boolean }>(hubUrl(msg.repo, 'tokenizer_config.json', rev))
  tokenizer = buildTokenizer(tj, tcfg ?? undefined)
  const buf = await fetchWithProgress(hubUrl(msg.repo, 'model.safetensors', rev), 'weights')
  post({ t: 'progress', phase: 'building', loaded: 0, total: 0 })
  await new Promise((r) => setTimeout(r, 0))
  model = new LiveModel(cfg, buf)
  post({ t: 'ready', info: model.info })
}

function run(prompt: string): void {
  if (!model || !tokenizer) throw new Error('model is not loaded')
  model.reset()
  ids = tokenizer.encode(prompt)
  if (!ids.length) throw new Error('the prompt tokenized to nothing')
  if (ids.length > model.info.maxSeq - 8) ids = ids.slice(0, model.info.maxSeq - 8)
  const t0 = performance.now()
  const logits = model.forward(ids)
  lastLogits = logits
  post({
    t: 'ran',
    ids: [...ids],
    tokens: ids.map((i) => tokenizer!.pretty(i)),
    text: tokenizer.decode(ids),
    topk: topkProbs(logits, 8),
    seq: model.seq,
    ms: Math.round(performance.now() - t0),
  })
}

async function generate(msg: GenMsg): Promise<void> {
  if (!model || !tokenizer) throw new Error('model is not loaded')
  stopFlag = false
  if (lastLogits === null || model.seq !== ids.length) {
    // no run yet (or state drifted): prefill first
    model.reset()
    lastLogits = model.forward(ids)
  }
  for (let i = 0; i < msg.maxNew; i++) {
    if (stopFlag || model.seq >= model.info.maxSeq - 1) break
    const id = sample(lastLogits, msg.temperature, msg.topK)
    if (tokenizer.eosId != null && id === tokenizer.eosId) break
    ids.push(id)
    const t0 = performance.now()
    lastLogits = model.forward([id])
    post({
      t: 'tok',
      id,
      tok: tokenizer.pretty(id),
      text: tokenizer.decode(ids),
      topk: topkProbs(lastLogits, 8),
      seq: model.seq,
      ms: Math.round(performance.now() - t0),
    })
    await new Promise((r) => setTimeout(r, 0)) // let stop messages land
  }
  post({ t: 'genDone', seq: model.seq })
}

self.onmessage = async (ev: MessageEvent<InMsg>) => {
  const msg = ev.data
  try {
    if (msg.t === 'load') await load(msg)
    else if (msg.t === 'run') run(msg.prompt)
    else if (msg.t === 'gen') await generate(msg)
    else if (msg.t === 'stop') stopFlag = true
    else if (msg.t === 'attn') {
      if (!model) throw new Error('model is not loaded')
      const m = model.attnMatrix(msg.layer)
      post({ t: 'attn', layer: msg.layer, ...m }, [m.data.buffer])
    } else if (msg.t === 'lens') {
      if (!model) throw new Error('model is not loaded')
      const rows = model.logitLens().map((logits, layer) => ({ layer, top: topkProbs(logits, 5) }))
      post({ t: 'lens', rows })
    } else if (msg.t === 'headstats') {
      if (!model) throw new Error('model is not loaded')
      post({ t: 'headstats', layer: msg.layer, stats: model.headStats(msg.layer) })
    } else if (msg.t === 'ablate') {
      if (!model || !ids.length) throw new Error('run a prompt first')
      const key = `${msg.layer}:${msg.head}`
      if (msg.on) model.ablated.add(key)
      else model.ablated.delete(key)
      // re-run the whole sequence under the new ablation set
      model.reset()
      lastLogits = model.forward(ids)
      post({
        t: 'ablated',
        ablated: [...model.ablated],
        topk: topkProbs(lastLogits, 8),
        seq: model.seq,
      })
    }
  } catch (e) {
    post({ t: 'error', message: e instanceof Error ? e.message : String(e) })
  }
}
