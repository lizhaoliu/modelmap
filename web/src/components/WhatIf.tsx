import { useEffect, useRef } from 'react'
import { DTYPE_OPTIONS, WEIGHT_OPTIONS, useCostStore } from '../analytics/costStore'
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
      <label>
        serve weights as
        <select value={a.weights ?? 'stored'} onChange={(e) => setA({ weights: e.target.value })} aria-label="weight precision">
          {WEIGHT_OPTIONS.map((w) => <option key={w} value={w}>{w === 'stored' ? 'as stored' : w === 'int4' ? 'int4 (Q4 / NF4 / AWQ)' : w === 'int8' ? 'int8' : w === 'f8' ? 'fp8' : w}</option>)}
        </select>
      </label>
      <p className="mm-pop-note">
        Compute is analytic MACs (weight matmuls + the attention core); memory is output-activation bytes; KV cache from
        the config. "Serve weights as" re-prices every weight tensor for a quantized deployment (the <b>vram</b> lens and
        the planner follow). Estimates — hover any number for its formula. Assumptions travel in the URL.
      </p>
    </div>
  )
}
