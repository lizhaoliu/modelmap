import { useEffect, useRef } from 'react'
import { DTYPE_OPTIONS, useCostStore } from '../analytics/costStore'
import { fmtInt } from '../analytics/cost'

const T_STOPS = [1, 8, 32, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

/** Assumptions behind every cost number: sequence length, batch, dtype. */
export function WhatIf({ onClose }: { onClose: () => void }) {
  const a = useCostStore((s) => s.assumptions)
  const setA = useCostStore((s) => s.setAssumptions)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    const onDown = (e: MouseEvent) => !ref.current?.contains(e.target as globalThis.Node) && onClose()
    window.addEventListener('keydown', onKey)
    const t = setTimeout(() => document.addEventListener('mousedown', onDown), 0)
    return () => { window.removeEventListener('keydown', onKey); clearTimeout(t); document.removeEventListener('mousedown', onDown) }
  }, [onClose])
  const idx = Math.max(0, T_STOPS.findIndex((t) => t >= a.T))
  return (
    <div className="mm-pop mm-whatif" role="dialog" aria-label="Cost assumptions" ref={ref}>
      <label>
        sequence length <b>{fmtInt(a.T)}</b>
        <input type="range" min={0} max={T_STOPS.length - 1} step={1} value={idx}
          onChange={(e) => setA({ T: T_STOPS[Number(e.target.value)] })} aria-label="sequence length" />
      </label>
      <label>
        batch
        <input type="number" min={1} max={4096} value={a.B} onChange={(e) => setA({ B: Math.max(1, Number(e.target.value) || 1) })} />
      </label>
      <label>
        dtype (activations, and weights whose stored dtype is unknown)
        <select value={a.dtypeLabel} onChange={(e) => setA({ dtypeLabel: e.target.value })}>
          {DTYPE_OPTIONS.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
      </label>
      <p className="mm-pop-note">
        Compute is analytic MACs (weight matmuls + the attention core); memory is output-activation bytes; KV cache from
        the config. Estimates — hover any number for its formula. Assumptions travel in the URL.
      </p>
    </div>
  )
}
