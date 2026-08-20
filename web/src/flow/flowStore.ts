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
  /** camera tracks the pulse; a manual pan/zoom during flow turns it off */
  follow: boolean
  /** this activation was the first-visit autoplay, not a user action */
  auto: boolean
  activate: (opts?: { auto?: boolean }) => void
  deactivate: () => void
  play: () => void
  pause: () => void
  setSpeed: (s: number) => void
  setFollow: (v: boolean) => void
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
  follow: true,
  auto: false,
  activate: (opts) => {
    syncUrl(true)
    localStorage.setItem('mm-flow-used', '1')
    set({ active: true, playing: true, beatIdx: 0, tCoarse: 0, follow: true, auto: Boolean(opts?.auto) })
  },
  deactivate: () => {
    syncUrl(false)
    set({ active: false, playing: false, auto: false })
  },
  play: () => set({ playing: true }),
  pause: () => set({ playing: false }),
  setSpeed: (speed) => set({ speed }),
  setFollow: (follow) => set({ follow }),
  _setBeat: (beatIdx) => set({ beatIdx }),
  _setCoarse: (tCoarse) => set({ tCoarse }),
  _setTotal: (total) => set({ total }),
}))
