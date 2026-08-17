import { useState } from 'react'
import { fmtBytes, fmtInt, LENSES } from '../analytics/cost'
import { useCostStore } from '../analytics/costStore'
import { WhatIf } from './WhatIf'

export function LensBar() {
  const lens = useCostStore((s) => s.lens)
  const setLens = useCostStore((s) => s.setLens)
  const a = useCostStore((s) => s.assumptions)
  const report = useCostStore((s) => s.report)
  const [open, setOpen] = useState(false)
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
          T {fmtInt(a.T)} · B {a.B} · {a.dtypeLabel}
          {report && lens === 'kv' && ` · ${fmtBytes(report.root.kvPerToken * a.T * a.B)} KV`}
        </button>
        {open && <WhatIf onClose={() => setOpen(false)} />}
      </span>
    </div>
  )
}
