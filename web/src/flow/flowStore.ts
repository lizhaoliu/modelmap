import { create } from 'zustand'

/** UI-facing flow state. The engine keeps precise time in a ref and pushes
 *  coarse updates here (~8 Hz) so React never renders per animation frame. */
interface FlowUI {
  active: boolean
  playing: boolean
  speed: number
  beatIdx: number
  tCoarse: number
  total: number
  activate: () => void
  deactivate: () => void
  play: () => void
  pause: () => void
  setSpeed: (s: number) => void
  _setBeat: (i: number) => void
  _setCoarse: (t: number) => void
  _setTotal: (t: number) => void
}

function syncUrl(on: boolean) {
  const url = new URL(location.href)
  if (on) url.searchParams.set('mode', 'flow')
  else url.searchParams.delete('mode')
  history.replaceState({}, '', url)
}

export const useFlowStore = create<FlowUI>((set) => ({
  active: false,
  playing: false,
  speed: 1,
  beatIdx: 0,
  tCoarse: 0,
  total: 0,
  activate: () => {
    syncUrl(true)
    set({ active: true, playing: true, beatIdx: 0, tCoarse: 0 })
  },
  deactivate: () => {
    syncUrl(false)
    set({ active: false, playing: false })
  },
  play: () => set({ playing: true }),
  pause: () => set({ playing: false }),
  setSpeed: (speed) => set({ speed }),
  _setBeat: (beatIdx) => set({ beatIdx }),
  _setCoarse: (tCoarse) => set({ tCoarse }),
  _setTotal: (total) => set({ total }),
}))
