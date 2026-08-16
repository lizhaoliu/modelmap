import { useEffect, useMemo, useRef } from 'react'
import type { FlowScript } from './beats'
import { useFlowStore } from './flowStore'

export interface Rect {
  x: number
  y: number
  w: number
  h: number
}

export interface FlowApi {
  seek: (t: number) => void
  stepBeat: (delta: number) => void
}

const TRAVEL_FRAC = 0.35 // first part of a beat travels; the rest dwells

/** One rAF clock drives everything: the pulse moves via direct DOM transforms
 *  and node highlighting via classList — React only re-renders on beat
 *  boundaries (HUD) and ~8 Hz for the scrubber (design doc §08). */
export function useFlowEngine(
  script: FlowScript,
  positions: Record<string, Rect>,
  pulseRef: React.RefObject<HTMLDivElement | null>,
): FlowApi {
  const active = useFlowStore((s) => s.active)
  const st = useRef({ t: 0, beat: -1, raf: 0, last: 0 })
  const scriptRef = useRef(script)
  scriptRef.current = script
  const posRef = useRef(positions)
  posRef.current = positions

  const api = useMemo<FlowApi>(() => {
    const nodeEl = (id: string) =>
      document.querySelector(`.react-flow__node[data-id="${CSS.escape(id)}"]`)

    const clearClasses = () => {
      document
        .querySelectorAll('.mm-flow-active, .mm-flow-spent')
        .forEach((el) => el.classList.remove('mm-flow-active', 'mm-flow-spent'))
    }

    const applyBeatClasses = (idx: number) => {
      clearClasses()
      const beats = scriptRef.current.beats
      const spent = new Set<string>()
      for (let i = 0; i < idx; i++) spent.add(beats[i].node)
      const cur = beats[idx]?.node
      if (cur) spent.delete(cur)
      for (const id of spent) nodeEl(id)?.classList.add('mm-flow-spent')
      if (cur) nodeEl(cur)?.classList.add('mm-flow-active')
    }

    const center = (id: string) => {
      const r = posRef.current[id]
      return r ? { x: r.x + r.w / 2, y: r.y + r.h / 2 } : null
    }

    const applyFrame = (t: number, force = false) => {
      const { beats, total } = scriptRef.current
      if (!beats.length) return
      const tt = Math.min(Math.max(t, 0), Math.max(total - 1e-6, 0))
      let idx = beats.findIndex((b) => tt < b.start + b.dur)
      if (idx === -1) idx = beats.length - 1
      if (idx !== st.current.beat || force) {
        st.current.beat = idx
        applyBeatClasses(idx)
        useFlowStore.getState()._setBeat(idx)
      }
      const b = beats[idx]
      const prev = idx > 0 ? beats[idx - 1] : null
      const p = (tt - b.start) / b.dur
      let pos = center(b.node)
      if (prev && prev.node !== b.node && p < TRAVEL_FRAC) {
        const a = center(prev.node)
        const z = center(b.node)
        if (a && z) {
          const k = p / TRAVEL_FRAC
          pos = { x: a.x + (z.x - a.x) * k, y: a.y + (z.y - a.y) * k }
        }
      }
      if (pos && pulseRef.current) {
        pulseRef.current.style.transform = `translate(${pos.x}px, ${pos.y}px)`
      }
    }

    return {
      seek(t: number) {
        st.current.t = Math.min(Math.max(t, 0), scriptRef.current.total)
        applyFrame(st.current.t, true)
        useFlowStore.getState()._setCoarse(st.current.t)
      },
      stepBeat(delta: number) {
        const beats = scriptRef.current.beats
        if (!beats.length) return
        const idx = Math.min(Math.max(st.current.beat + delta, 0), beats.length - 1)
        this.seek(beats[idx].start + 1e-3)
      },
      /* internal — reused by the rAF loop */
      _applyFrame: applyFrame,
      _clear: clearClasses,
    } as FlowApi & { _applyFrame: typeof applyFrame; _clear: typeof clearClasses }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pulseRef])

  useEffect(() => {
    const internal = api as FlowApi & {
      _applyFrame: (t: number, force?: boolean) => void
      _clear: () => void
    }
    if (!active) {
      internal._clear()
      return
    }
    useFlowStore.getState()._setTotal(scriptRef.current.total)
    st.current.t = 0
    st.current.beat = -1
    st.current.last = 0
    internal._applyFrame(0, true)
    let coarseLast = 0
    const loop = (ts: number) => {
      const s = useFlowStore.getState()
      const dt = st.current.last ? (ts - st.current.last) / 1000 : 0
      st.current.last = ts
      if (s.playing) {
        st.current.t += dt * s.speed
        if (st.current.t >= scriptRef.current.total) {
          st.current.t = scriptRef.current.total
          s.pause()
        }
        internal._applyFrame(st.current.t)
        if (ts - coarseLast > 120) {
          coarseLast = ts
          s._setCoarse(st.current.t)
        }
      }
      st.current.raf = requestAnimationFrame(loop)
    }
    st.current.raf = requestAnimationFrame(loop)
    return () => {
      cancelAnimationFrame(st.current.raf)
      internal._clear()
    }
  }, [active, script, api])

  return api
}
