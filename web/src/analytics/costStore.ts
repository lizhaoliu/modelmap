import { create } from 'zustand'
import type { GraphDoc, GraphIndex } from '../types'
import {
  computeCosts, DEFAULT_ASSUMPTIONS, type Assumptions, type CostReport, type Lens,
} from './cost'

const DTYPES: Record<string, number> = { bf16: 2, f16: 2, f32: 4, f8: 1, int8: 1, int4: 0.5 }

interface CostState {
  lens: Lens
  assumptions: Assumptions
  report: CostReport | null
  setLens: (l: Lens) => void
  setAssumptions: (a: Partial<Pick<Assumptions, 'T' | 'B'>> & { dtypeLabel?: string }) => void
  recompute: (doc: GraphDoc | null, index: GraphIndex | null) => void
}

function fromUrl(): { lens: Lens; assumptions: Assumptions } {
  const p = new URL(location.href).searchParams
  const lens = (p.get('lens') as Lens) || 'none'
  const dtypeLabel = p.get('dtype') && DTYPES[p.get('dtype')!] ? p.get('dtype')! : DEFAULT_ASSUMPTIONS.dtypeLabel
  return {
    lens: ['params', 'compute', 'memory', 'kv'].includes(lens) ? lens : 'none',
    assumptions: {
      T: clampInt(p.get('T'), DEFAULT_ASSUMPTIONS.T, 1, 1 << 20),
      B: clampInt(p.get('B'), DEFAULT_ASSUMPTIONS.B, 1, 4096),
      bytes: DTYPES[dtypeLabel],
      dtypeLabel,
    },
  }
}
function clampInt(v: string | null, d: number, lo: number, hi: number) {
  const n = v ? parseInt(v, 10) : NaN
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d
}
function toUrl(lens: Lens, a: Assumptions) {
  const url = new URL(location.href)
  const set = (k: string, v: string | null) => (v == null ? url.searchParams.delete(k) : url.searchParams.set(k, v))
  set('lens', lens === 'none' ? null : lens)
  const def = a.T === DEFAULT_ASSUMPTIONS.T && a.B === DEFAULT_ASSUMPTIONS.B && a.dtypeLabel === DEFAULT_ASSUMPTIONS.dtypeLabel
  set('T', def ? null : String(a.T))
  set('B', def ? null : String(a.B))
  set('dtype', def ? null : a.dtypeLabel)
  history.replaceState({}, '', url)
}

let lastDoc: GraphDoc | null = null
let lastIndex: GraphIndex | null = null

export const useCostStore = create<CostState>((set, get) => ({
  ...fromUrl(),
  report: null,
  setLens: (lens) => {
    set({ lens })
    toUrl(lens, get().assumptions)
  },
  setAssumptions: (patch) => {
    const cur = get().assumptions
    const dtypeLabel = patch.dtypeLabel ?? cur.dtypeLabel
    const a: Assumptions = { T: patch.T ?? cur.T, B: patch.B ?? cur.B, bytes: DTYPES[dtypeLabel] ?? 2, dtypeLabel }
    set({ assumptions: a })
    toUrl(get().lens, a)
    get().recompute(lastDoc, lastIndex)
  },
  recompute: (doc, index) => {
    lastDoc = doc
    lastIndex = index
    set({ report: doc && index ? computeCosts(doc, index, get().assumptions) : null })
  },
}))

export const DTYPE_OPTIONS = Object.keys(DTYPES)
