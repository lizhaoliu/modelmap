/**
 * Planner settings shared by the "fit?" popover (§17/§22) and the on-graph
 * VRAM strip (§26): GPU preset, count, parallelism, fine-tune recipe.
 * Persisted to localStorage; the serving layout also travels in the URL.
 */
import { create } from 'zustand'
import { GPU_PRESETS } from './plan'

const KEY = 'mm-plan'

export interface PlanSettings {
  gpus: number; mem: number; tp: number; pp: number; headroom: number
  mode: 'serve' | 'train'
  gpuName: string
  method: 'full' | 'lora' | 'qlora'; rank: number; targets: 'attention' | 'attn-mlp' | 'all-linear'
  optim: 'adamw' | 'adamw8bit'; sharding: 'none' | 'zero2' | 'zero3'; ckpt: boolean; flash: boolean
}
export const PLAN_DEFAULTS: PlanSettings = {
  gpus: 1, mem: 80, tp: 1, pp: 1, headroom: 0.1,
  mode: 'serve', gpuName: 'H100 80GB',
  method: 'qlora', rank: 16, targets: 'attn-mlp', optim: 'adamw', sharding: 'none', ckpt: true, flash: true,
}

function load(): PlanSettings {
  try {
    const p = new URL(location.href).searchParams
    const fromUrl: Partial<PlanSettings> = {}
    if (p.get('gpus')) fromUrl.gpus = +p.get('gpus')!
    if (p.get('gmem')) fromUrl.mem = +p.get('gmem')!
    if (p.get('tp')) fromUrl.tp = +p.get('tp')!
    if (p.get('pp')) fromUrl.pp = +p.get('pp')!
    if (p.get('gpu') && GPU_PRESETS.some(([n]) => n === p.get('gpu'))) {
      fromUrl.gpuName = p.get('gpu')!
      if (!p.get('gmem')) fromUrl.mem = GPU_PRESETS.find(([n]) => n === p.get('gpu'))![1]
    }
    const saved = JSON.parse(localStorage.getItem(KEY) ?? '{}') as Partial<PlanSettings>
    return { ...PLAN_DEFAULTS, ...saved, ...fromUrl }
  } catch {
    return PLAN_DEFAULTS
  }
}
function persist(s: PlanSettings) {
  try {
    localStorage.setItem(KEY, JSON.stringify(s))
  } catch { /* private mode */ }
  const u = new URL(location.href)
  const set = (k: string, v: number, d: number) => (v === d ? u.searchParams.delete(k) : u.searchParams.set(k, String(v)))
  set('gpus', s.gpus, PLAN_DEFAULTS.gpus); set('gmem', s.mem, PLAN_DEFAULTS.mem); set('tp', s.tp, PLAN_DEFAULTS.tp); set('pp', s.pp, PLAN_DEFAULTS.pp)
  if (s.gpuName === PLAN_DEFAULTS.gpuName) u.searchParams.delete('gpu')
  else u.searchParams.set('gpu', s.gpuName)
  history.replaceState({}, '', u)
}

interface PlanStore {
  s: PlanSettings
  upd: (patch: Partial<PlanSettings>) => void
  /** pick a preset by name: sets its memory too */
  pickGpu: (name: string) => void
}

export const usePlanStore = create<PlanStore>((set, get) => ({
  s: load(),
  upd: (patch) => {
    const cur = get().s
    const next = { ...cur, ...patch }
    // keep tp × pp = gpus: changing gpus re-derives pp; changing tp/pp re-derives gpus
    if (patch.gpus != null) {
      next.tp = Math.min(next.tp, next.gpus)
      next.pp = Math.max(1, Math.floor(next.gpus / next.tp))
      if (next.tp * next.pp !== next.gpus) { next.tp = 1; next.pp = next.gpus }
    } else if (patch.tp != null || patch.pp != null) next.gpus = next.tp * next.pp
    persist(next)
    set({ s: next })
  },
  pickGpu: (name) => {
    const gb = GPU_PRESETS.find(([n]) => n === name)?.[1]
    get().upd(gb != null ? { gpuName: name, mem: gb } : { gpuName: name })
  },
}))
