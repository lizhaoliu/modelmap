import { create } from 'zustand'
import { useFlowStore } from '../flow/flowStore'
import type { GraphDoc } from '../types'
import type { ModelInfo } from './model'
import type { TopEntry } from './worker'

/** Live mode (design doc §20): real inference in the visitor's browser.
 *  The store owns the worker; components read plain state. */

export type LiveStatus = 'idle' | 'loading' | 'ready' | 'running' | 'generating' | 'error'

export interface LensRow {
  layer: number
  top: TopEntry[]
}

export interface AttnData {
  layer: number
  heads: number
  seq: number
  data: Float32Array
}

export interface HeadStat {
  prev: number
  first: number
  self: number
  entropy: number
  tag: string
}

interface LiveState {
  open: boolean
  status: LiveStatus
  error: string | null
  progress: { phase: string; loaded: number; total: number } | null
  info: ModelInfo | null
  prompt: string
  tokens: string[]
  text: string
  topk: TopEntry[]
  lens: LensRow[] | null
  lensStale: boolean
  attn: AttnData | null
  attnLayer: number
  attnHead: number // -1 = mean over heads
  genCount: number // generated tokens this session (drives the canvas ripple)
  tokMs: number | null
  temperature: number
  /** "layer:head" pairs currently silenced (§24) */
  ablated: string[]
  /** next-token top-k with no ablations, for the Δ display */
  baselineTopk: TopEntry[] | null
  headStats: HeadStat[] | null

  toggle: (doc: GraphDoc | null) => void
  close: () => void
  load: (repo: string) => void
  setPrompt: (p: string) => void
  run: () => void
  generate: (n?: number) => void
  stop: () => void
  setAttnLayer: (l: number) => void
  setAttnHead: (h: number) => void
  requestLens: () => void
  toggleAblate: (layer: number, head: number) => void
  clearAblations: () => void
}

/** Which models can run live: single-file f32/bf16/f16 safetensors of a
 *  llama-family or GPT-2 checkpoint, small enough to download casually. */
export const LIVE_MAX_MB = 700

export function liveSupport(doc: GraphDoc | null): { ok: boolean; sizeMB: number; reason?: string } {
  if (!doc) return { ok: false, sizeMB: 0, reason: 'no model' }
  const c = doc.config as Record<string, unknown>
  const arch = c.model_type
  if (doc.model_id.startsWith('local:')) return { ok: false, sizeMB: 0, reason: 'local checkpoints stay on your disk' }
  if (arch !== 'llama' && arch !== 'gpt2' && arch !== 'qwen2' && arch !== 'qwen3')
    return { ok: false, sizeMB: 0, reason: `live inference supports llama / qwen2 / qwen3 / gpt2 checkpoints (this is ${String(arch ?? 'unknown')})` }
  if (doc.weights_format && doc.weights_format !== 'safetensors')
    return { ok: false, sizeMB: 0, reason: 'live inference needs a safetensors checkpoint' }
  if (doc.fidelity !== 'full') return { ok: false, sizeMB: 0, reason: 'no full trace' }
  const bytesPer = typeof c.torch_dtype === 'string' && c.torch_dtype.includes('16') ? 2 : typeof c.dtype === 'string' && String(c.dtype).includes('16') ? 2 : 4
  const sizeMB = Math.round((doc.params_total * bytesPer) / 1e6)
  if (sizeMB > LIVE_MAX_MB) return { ok: false, sizeMB, reason: `${sizeMB} MB of weights is beyond the in-browser cap (${LIVE_MAX_MB} MB)` }
  return { ok: true, sizeMB }
}

/** Curated models the landing / empty state points at. */
export const LIVE_PICKS = [
  { id: 'Maykeye/TinyLLama-v0', size: '9 MB', blurb: 'llama in miniature, trained on TinyStories — instant' },
  { id: 'HuggingFaceTB/SmolLM2-135M', size: '269 MB', blurb: 'a real small LM — slower, much smarter' },
  { id: 'distilbert/distilgpt2', size: '353 MB', blurb: 'the classic, distilled' },
]

// flow and live share the bottom of the canvas: starting a replay closes live
useFlowStore.subscribe((s, prev) => {
  if (s.active && !prev.active && useLiveStore.getState().open) useLiveStore.getState().close()
})

let worker: Worker | null = null

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    worker.onmessage = (ev) => handle(ev.data)
    worker.onerror = (e) => useLiveStore.setState({ status: 'error', error: e.message || 'the live worker crashed' })
  }
  return worker
}

function killWorker(): void {
  worker?.terminate()
  worker = null
}

function handle(msg: Record<string, unknown> & { t: string }): void {
  const set = useLiveStore.setState
  const get = useLiveStore.getState
  switch (msg.t) {
    case 'progress':
      set({ progress: msg as unknown as LiveState['progress'] })
      break
    case 'ready':
      set({ status: 'ready', info: msg.info as ModelInfo, progress: null })
      break
    case 'ran':
      set((s) => ({
        status: 'ready',
        tokens: msg.tokens as string[],
        text: msg.text as string,
        topk: msg.topk as TopEntry[],
        baselineTopk: s.ablated.length ? s.baselineTopk : (msg.topk as TopEntry[]),
        tokMs: msg.ms as number,
        lens: null,
        lensStale: true,
        attn: null,
      }))
      get().requestLens()
      refreshAttn()
      refreshHeadStats()
      break
    case 'ablated':
      set({ ablated: msg.ablated as string[], topk: msg.topk as TopEntry[], lensStale: true })
      get().requestLens()
      refreshAttn()
      break
    case 'headstats':
      if ((msg.layer as number) === get().attnLayer) set({ headStats: msg.stats as HeadStat[] })
      break
    case 'tok':
      set((s) => ({
        tokens: [...s.tokens, msg.tok as string],
        text: msg.text as string,
        topk: msg.topk as TopEntry[],
        baselineTopk: s.ablated.length ? s.baselineTopk : (msg.topk as TopEntry[]),
        tokMs: msg.ms as number,
        genCount: s.genCount + 1,
        lensStale: true,
      }))
      break
    case 'genDone':
      set({ status: 'ready' })
      get().requestLens()
      refreshAttn()
      break
    case 'attn':
      set({ attn: { layer: msg.layer as number, heads: msg.heads as number, seq: msg.seq as number, data: msg.data as Float32Array } })
      break
    case 'lens':
      set({ lens: msg.rows as LensRow[], lensStale: false })
      break
    case 'error':
      set({ status: get().info ? 'ready' : 'error', error: String(msg.message) })
      break
  }
}

function refreshAttn(): void {
  const s = useLiveStore.getState()
  if (s.info && s.tokens.length) getWorker().postMessage({ t: 'attn', layer: s.attnLayer })
}

function refreshHeadStats(): void {
  const s = useLiveStore.getState()
  if (s.info && s.tokens.length) getWorker().postMessage({ t: 'headstats', layer: s.attnLayer })
}

export const useLiveStore = create<LiveState>((set, get) => ({
  open: false,
  status: 'idle',
  error: null,
  progress: null,
  info: null,
  prompt: 'Once upon a time there was a little',
  tokens: [],
  text: '',
  topk: [],
  lens: null,
  lensStale: false,
  attn: null,
  attnLayer: 0,
  attnHead: -1,
  genCount: 0,
  tokMs: null,
  temperature: 0.7,
  ablated: [],
  baselineTopk: null,
  headStats: null,

  toggle(doc) {
    const opening = !get().open
    if (opening) {
      useFlowStore.getState().deactivate()
      set({ open: true })
      void doc
    } else {
      get().close()
    }
  },

  close() {
    set({ open: false })
    if (get().status === 'generating') getWorker().postMessage({ t: 'stop' })
  },

  load(repo) {
    set({ status: 'loading', error: null, progress: { phase: 'config', loaded: 0, total: 0 } })
    getWorker().postMessage({ t: 'load', repo })
  },

  setPrompt: (prompt) => set({ prompt }),

  run() {
    const s = get()
    if (s.status !== 'ready' || !s.prompt.trim()) return
    set({ status: 'running', error: null, tokens: [], topk: [], text: '', genCount: 0, baselineTopk: null, headStats: null })
    getWorker().postMessage({ t: 'run', prompt: s.prompt })
    // 'ran' flips status back; a long prefill keeps the spinner honest
    setTimeout(() => {
      if (get().status === 'running') set({ status: 'running' })
    }, 50)
  },

  generate(n = 24) {
    const s = get()
    if (s.status !== 'ready' || !s.tokens.length) return
    set({ status: 'generating', error: null })
    getWorker().postMessage({ t: 'gen', maxNew: n, temperature: s.temperature, topK: 40 })
  },

  stop() {
    getWorker().postMessage({ t: 'stop' })
  },

  setAttnLayer(l) {
    set({ attnLayer: l, headStats: null })
    refreshAttn()
    refreshHeadStats()
  },

  toggleAblate(layer, head) {
    const key = `${layer}:${head}`
    const on = !get().ablated.includes(key)
    getWorker().postMessage({ t: 'ablate', layer, head, on })
  },

  clearAblations() {
    for (const key of get().ablated) {
      const [layer, head] = key.split(':').map(Number)
      getWorker().postMessage({ t: 'ablate', layer, head, on: false })
    }
  },

  setAttnHead: (attnHead) => set({ attnHead }),

  requestLens() {
    if (get().info && get().tokens.length) getWorker().postMessage({ t: 'lens' })
  },
}))

/** Full reset when the user navigates to another model. */
export function resetLive(): void {
  killWorker()
  useLiveStore.setState({
    open: false, status: 'idle', error: null, progress: null, info: null,
    tokens: [], text: '', topk: [], lens: null, lensStale: false, attn: null,
    attnLayer: 0, attnHead: -1, genCount: 0, tokMs: null,
    ablated: [], baselineTopk: null, headStats: null,
  })
}
