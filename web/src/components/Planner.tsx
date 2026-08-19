import { useEffect, useMemo, useRef, useState } from 'react'
import { fmtBytes, fmtInt } from '../analytics/cost'
import { useCostStore } from '../analytics/costStore'
import { GPU_PRESETS, planServing, type PlanRequest } from '../analytics/plan'
import { useStore } from '../store'

const KEY = 'mm-plan'

interface Saved { gpus: number; mem: number; tp: number; pp: number; headroom: number }
const DEFAULTS: Saved = { gpus: 1, mem: 80, tp: 1, pp: 1, headroom: 0.1 }

function load(): Saved {
  try {
    const p = new URL(location.href).searchParams
    const fromUrl: Partial<Saved> = {}
    if (p.get('gpus')) fromUrl.gpus = +p.get('gpus')!
    if (p.get('gmem')) fromUrl.mem = +p.get('gmem')!
    if (p.get('tp')) fromUrl.tp = +p.get('tp')!
    if (p.get('pp')) fromUrl.pp = +p.get('pp')!
    const saved = JSON.parse(localStorage.getItem(KEY) ?? '{}') as Partial<Saved>
    return { ...DEFAULTS, ...saved, ...fromUrl }
  } catch {
    return DEFAULTS
  }
}
function persist(s: Saved) {
  localStorage.setItem(KEY, JSON.stringify(s))
  const u = new URL(location.href)
  const set = (k: string, v: number, d: number) => (v === d ? u.searchParams.delete(k) : u.searchParams.set(k, String(v)))
  set('gpus', s.gpus, DEFAULTS.gpus); set('gmem', s.mem, DEFAULTS.mem); set('tp', s.tp, DEFAULTS.tp); set('pp', s.pp, DEFAULTS.pp)
  history.replaceState({}, '', u)
}

/** "fit?" — the serving planner (design doc §17): does this model fit on my
 *  GPUs, how would TP/PP split it, and how much context is left for KV. */
export function Planner({ onClose }: { onClose: () => void }) {
  const doc = useStore((s) => s.doc)
  const index = useStore((s) => s.index)
  const a = useCostStore((s) => s.assumptions)
  const [s, setS] = useState<Saved>(load)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => persist(s), [s])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    const onDown = (e: MouseEvent) => !ref.current?.contains(e.target as globalThis.Node) && onClose()
    window.addEventListener('keydown', onKey)
    const t = setTimeout(() => document.addEventListener('mousedown', onDown), 0)
    return () => { window.removeEventListener('keydown', onKey); clearTimeout(t); document.removeEventListener('mousedown', onDown) }
  }, [onClose])

  const plan = useMemo(() => {
    if (!doc || !index) return null
    const req: PlanRequest = { gpus: s.gpus, gpuMemoryGb: s.mem, tp: s.tp, pp: s.pp, T: a.T, B: a.B, dtypeLabel: a.dtypeLabel, bytes: a.bytes, headroom: s.headroom }
    return planServing(doc, index, req)
  }, [doc, index, s, a])
  if (!doc || !plan) return null

  const upd = (patch: Partial<Saved>) => setS((cur) => {
    const next = { ...cur, ...patch }
    // keep tp × pp = gpus: changing gpus re-derives pp; changing tp/pp re-derives gpus
    if (patch.gpus != null) {
      next.tp = Math.min(next.tp, next.gpus)
      next.pp = Math.max(1, Math.floor(next.gpus / next.tp))
      if (next.tp * next.pp !== next.gpus) { next.tp = 1; next.pp = next.gpus }
    } else if (patch.tp != null || patch.pp != null) next.gpus = next.tp * next.pp
    return next
  })
  const tpOptions = [1, 2, 4, 8, 16].filter((x) => x <= 64)
  const cap = plan.perGpuCapacityBytes
  const pct = (b: number) => (cap > 0 ? Math.min(100, (b / cap) * 100) : 0)
  const cmd = `uvx modelmap plan ${doc.model_id} --gpus ${s.gpus} --gpu-memory ${s.mem} --tp ${s.tp} --pp ${s.pp} -T ${a.T} -B ${a.B} --dtype ${a.dtypeLabel}`

  return (
    <div className="mm-pop mm-planner" role="dialog" aria-label="Serving planner" ref={ref}>
      <div className="mm-plan-head">
        <b>fit on GPUs?</b>
        <span className={`mm-plan-verdict ${plan.fits ? 'is-ok' : 'is-over'}`}>{plan.fits ? 'fits' : 'does not fit'}</span>
      </div>
      <div className="mm-plan-grid">
        <label>GPU
          <select value={String(s.mem)} onChange={(e) => upd({ mem: Number(e.target.value) })}>
            {GPU_PRESETS.map(([name, gb]) => <option key={name} value={gb}>{name}</option>)}
            {!GPU_PRESETS.some(([, gb]) => gb === s.mem) && <option value={s.mem}>{s.mem} GB (custom)</option>}
          </select>
        </label>
        <label>memory GB <input type="number" min={0} max={100000} value={s.mem} onChange={(e) => upd({ mem: Math.max(0, Number(e.target.value) || 0) })} /></label>
        <label>GPUs <input type="number" min={1} max={4096} value={s.gpus} onChange={(e) => upd({ gpus: Math.max(1, Math.floor(Number(e.target.value) || 1)) })} /></label>
        <label>tensor ∥ <select value={s.tp} onChange={(e) => upd({ tp: Number(e.target.value) })}>{tpOptions.map((x) => <option key={x} value={x}>{x}</option>)}</select></label>
        <label>pipeline ∥ <input type="number" min={1} max={4096} value={s.pp} onChange={(e) => upd({ pp: Math.max(1, Math.floor(Number(e.target.value) || 1)) })} /></label>
        <label>headroom <select value={s.headroom} onChange={(e) => upd({ headroom: Number(e.target.value) })}>{[0, 0.05, 0.1, 0.15, 0.2, 0.3].map((h) => <option key={h} value={h}>{Math.round(h * 100)}%</option>)}</select></label>
      </div>
      <p className="mm-plan-assume">at T {fmtInt(a.T)} · B {a.B} · {a.dtypeLabel} (change in the assumptions chip) · capacity {fmtBytes(cap)}/GPU</p>
      <table className="mm-plan-table">
        <thead><tr><th>stage</th><th>layers</th><th>weights</th><th>kv</th><th>act</th><th>total / GPU</th></tr></thead>
        <tbody>
          {plan.stages.map((st) => (
            <tr key={st.stage} className={st.fits ? '' : 'is-over'}>
              <td>{st.stage}{s.tp > 1 ? <span className="mm-dim"> · gpus {st.gpus[0]}–{st.gpus[st.gpus.length - 1]}</span> : ''}</td>
              <td>{st.layers.length ? `${st.layers[0]}–${st.layers[1]}` : '—'} <span className="mm-dim">({st.layerCount})</span></td>
              <td>{fmtBytes(st.weightBytesPerGpu)}</td>
              <td>{fmtBytes(st.kvBytesPerGpu)}</td>
              <td>{fmtBytes(st.actBytesPerGpu)}</td>
              <td>
                <div className="mm-plan-bar" title={`${Math.round(pct(st.totalBytesPerGpu))}% of capacity`}>
                  <i className="w" style={{ width: `${pct(st.weightBytesPerGpu)}%` }} />
                  <i className="k" style={{ width: `${pct(st.kvBytesPerGpu)}%` }} />
                  <i className="a" style={{ width: `${pct(st.actBytesPerGpu)}%` }} />
                </div>
                {fmtBytes(st.totalBytesPerGpu)}{st.fits ? '' : ' ✗'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mm-plan-summary">
        <b>max context at B {a.B}: {plan.maxContextTokens ? fmtInt(plan.maxContextTokens) + ' tokens' : '—'}</b>
        {plan.stages.length > 1 && <> · {fmtBytes(plan.stages[0].boundaryBytesOut)} crosses each stage boundary per forward</>}
        <> · weights {fmtBytes(plan.weightBytes)} · KV {fmtBytes(plan.kvBytes)} at T</>
      </p>
      <p className="mm-pop-note">{plan.notes.join(' · ')}</p>
      <p className="mm-pop-note">
        <code>{cmd}</code>
        <button className="mm-link" onClick={() => void navigator.clipboard.writeText(cmd)} title="copy the CLI command">copy</button>
      </p>
    </div>
  )
}
