import { useMemo } from 'react'
import { fmtBytes, fmtInt } from '../analytics/cost'
import { useCostStore, WEIGHT_OPTIONS } from '../analytics/costStore'
import { GPU_PRESETS, planServing, type Plan, type PlanRequest } from '../analytics/plan'
import { usePlanStore } from '../analytics/planStore'
import { useStore } from '../store'

const T_STOPS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

/** The planner painted onto the graph (design doc §26): while the vram lens
 *  is on, this strip holds the knobs that move the picture — context, batch,
 *  weight precision, GPU — and the verdict: weights + KV + activations
 *  stacked against one GPU's capacity, and how many GPUs it takes. */
export function VramBar() {
  const doc = useStore((s) => s.doc)
  const index = useStore((s) => s.index)
  const a = useCostStore((s) => s.assumptions)
  const setA = useCostStore((s) => s.setAssumptions)
  const s = usePlanStore((st) => st.s)
  const pickGpu = usePlanStore((st) => st.pickGpu)
  const upd = usePlanStore((st) => st.upd)

  const fit = useMemo(() => {
    if (!doc || !index) return null
    const base: PlanRequest = {
      gpus: 1, gpuMemoryGb: s.mem, tp: 1, pp: 1, T: a.T, B: a.B, dtypeLabel: a.dtypeLabel, bytes: a.bytes,
      weights: a.weights, weightBytes: a.weightBytes, headroom: s.headroom,
    }
    const one = planServing(doc, index, base)
    // the smallest tensor-parallel group this GPU needs (same ladder as the planner's tp menu)
    let needs: { tp: number; plan: Plan } | null = one.fits ? { tp: 1, plan: one } : null
    if (!needs) {
      for (const tp of [2, 4, 8, 16]) {
        const p = planServing(doc, index, { ...base, gpus: tp, tp })
        if (p.fits) { needs = { tp, plan: p }; break }
      }
    }
    return { one, needs }
  }, [doc, index, a, s.mem, s.headroom])
  if (!doc || !fit) return null

  const st = fit.one.stages[0]
  const cap = fit.one.perGpuCapacityBytes
  const total = st.totalBytesPerGpu
  const pct = (b: number) => (cap > 0 ? Math.min(100, (b / cap) * 100) : 0)
  const over = total > cap
  const tIdx = Math.max(0, T_STOPS.findIndex((t) => t >= a.T))
  const gpuShort = s.gpuName.replace(/ \d+GB$/, '')
  const kvWins = st.kvBytesPerGpu > st.weightBytesPerGpu

  return (
    <div className="mm-vrambar" role="region" aria-label="GPU memory to serve">
      <div className="mm-vram-knobs">
        <label className="mm-vram-ctx">
          context <b>{fmtInt(a.T)}</b>
          <input type="range" min={0} max={T_STOPS.length - 1} step={1} value={tIdx}
            onChange={(e) => setA({ T: T_STOPS[Number(e.target.value)] })} aria-label="context length" />
        </label>
        <label>batch
          <input type="number" min={1} max={4096} value={a.B} onChange={(e) => setA({ B: Math.max(1, Number(e.target.value) || 1) })} aria-label="batch" />
        </label>
        <label>weights
          <select value={a.weights ?? 'stored'} onChange={(e) => setA({ weights: e.target.value })} aria-label="weight precision">
            {WEIGHT_OPTIONS.map((w) => <option key={w} value={w}>{w === 'stored' ? 'as stored' : w === 'f8' ? 'fp8' : w}</option>)}
          </select>
        </label>
        <label>GPU
          <select value={s.gpuName} onChange={(e) => pickGpu(e.target.value)} aria-label="GPU">
            {GPU_PRESETS.map(([name]) => <option key={name} value={name}>{name}</option>)}
          </select>
        </label>
        <label>headroom
          <select value={s.headroom} onChange={(e) => upd({ headroom: Number(e.target.value) })} aria-label="headroom">
            {[0, 0.05, 0.1, 0.15, 0.2, 0.3].map((h) => <option key={h} value={h}>{Math.round(h * 100)}%</option>)}
          </select>
        </label>
      </div>
      <div className="mm-vram-verdict">
        <div className={`mm-vram-bar ${over ? 'is-over' : ''}`} title={`${Math.round(pct(total))}% of ${fmtBytes(cap)} usable`}>
          <i className="w" style={{ width: `${pct(st.weightBytesPerGpu)}%` }} title={`weights ${fmtBytes(st.weightBytesPerGpu)}`} />
          <i className="k" style={{ width: `${pct(st.kvBytesPerGpu)}%` }} title={`KV cache ${fmtBytes(st.kvBytesPerGpu)}`} />
          <i className="a" style={{ width: `${pct(st.actBytesPerGpu)}%` }} title={`activations ${fmtBytes(st.actBytesPerGpu)}`} />
        </div>
        <span className="mm-vram-legend">
          <i className="w" /> weights {fmtBytes(st.weightBytesPerGpu)}
          <i className="k" /> KV {fmtBytes(st.kvBytesPerGpu)}{kvWins && <em> — the cache outweighs the model</em>}
          <i className="a" /> act {fmtBytes(st.actBytesPerGpu)}
          <span className="mm-dim"> = {fmtBytes(total)} of {fmtBytes(cap)}</span>
        </span>
        <span className={`mm-vram-fit ${fit.needs ? (fit.needs.tp === 1 ? 'is-ok' : 'is-multi') : 'is-over'}`}>
          {fit.needs
            ? fit.needs.tp === 1
              ? <>fits on 1× {gpuShort}{fit.one.maxContextTokens ? <span className="mm-dim"> · up to {fmtInt(fit.one.maxContextTokens)} tokens</span> : null}</>
              : <>needs {fit.needs.tp}× {gpuShort} <span className="mm-dim">(tensor-parallel)</span></>
            : <>does not fit on 16× {gpuShort}</>}
        </span>
      </div>
    </div>
  )
}
