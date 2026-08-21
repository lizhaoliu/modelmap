import { useState } from 'react'
import { fmtBytes, fmtInt, LENSES } from '../analytics/cost'
import { useCostStore } from '../analytics/costStore'
import { Planner } from './Planner'
import { WhatIf } from './WhatIf'

export function LensBar() {
  const lens = useCostStore((s) => s.lens)
  const setLens = useCostStore((s) => s.setLens)
  const a = useCostStore((s) => s.assumptions)
  const report = useCostStore((s) => s.report)
  const [open, setOpen] = useState(false)
  const [planOpen, setPlanOpen] = useState(false)
  return (
    <div className="mm-lens">
      <div className="mm-lens-seg" role="radiogroup" aria-label="Lens">
        {LENSES.map((l) => (
          <button
            key={l.id}
            role="radio"
            aria-checked={lens === l.id}
            className={`mm-lens-btn ${lens === l.id ? 'is-on' : ''}`}
            title={l.hint}
            onClick={() => setLens(lens === l.id ? 'none' : l.id)}
          >
            {l.label}
          </button>
        ))}
      </div>
      <span className="mm-topbar-rel">
        <button className="mm-btn mm-lens-assume" onClick={() => setOpen((v) => !v)} title="Assumptions behind the cost numbers" aria-expanded={open}>
          T {fmtInt(a.T)}{a.B > 1 ? ` · B ${a.B}` : ''} · {a.dtypeLabel}{a.weightBytes != null ? ` · ${a.weights} weights` : ''}
          {report && lens === 'kv' && ` · ${fmtBytes(report.root.kvPerToken * a.T * a.B)} KV`}
        </button>
        {open && <WhatIf onClose={() => setOpen(false)} />}
      </span>
      <span className="mm-topbar-rel">
        <button className={`mm-btn mm-btn-plan ${planOpen ? 'is-on' : ''}`} onClick={() => setPlanOpen((v) => !v)} title="Serving planner: does it fit on my GPUs, how to split it, max context" aria-expanded={planOpen}>
          fit?
        </button>
        {planOpen && <Planner onClose={() => setPlanOpen(false)} />}
      </span>
    </div>
  )
}
