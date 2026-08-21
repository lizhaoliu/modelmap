import { useEffect, useMemo, useRef } from 'react'
import { fmtBytes, fmtInt } from '../analytics/cost'
import { useCostStore } from '../analytics/costStore'
import { GPU_PRESETS, planServing, type PlanRequest } from '../analytics/plan'
import { usePlanStore, type PlanSettings } from '../analytics/planStore'
import { estimateThroughput, planTraining, type TrainRequest } from '../analytics/train'
import { fmtParams } from '../fmt'
import { useStore } from '../store'

/** "fit?" — the serving planner (design doc §17): does this model fit on my
 *  GPUs, how would TP/PP split it, and how much context is left for KV. */
export function Planner({ onClose }: { onClose: () => void }) {
  const doc = useStore((s) => s.doc)
  const index = useStore((s) => s.index)
  const a = useCostStore((s) => s.assumptions)
  const s = usePlanStore((st) => st.s)
  const upd = usePlanStore((st) => st.upd)
  const pickGpu = usePlanStore((st) => st.pickGpu)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    const onDown = (e: MouseEvent) => !ref.current?.contains(e.target as globalThis.Node) && onClose()
    window.addEventListener('keydown', onKey)
    const t = setTimeout(() => document.addEventListener('mousedown', onDown), 0)
    return () => { window.removeEventListener('keydown', onKey); clearTimeout(t); document.removeEventListener('mousedown', onDown) }
  }, [onClose])

  const plan = useMemo(() => {
    if (!doc || !index || s.mode !== 'serve') return null
    const req: PlanRequest = { gpus: s.gpus, gpuMemoryGb: s.mem, tp: s.tp, pp: s.pp, T: a.T, B: a.B, dtypeLabel: a.dtypeLabel, bytes: a.bytes, weights: a.weights, weightBytes: a.weightBytes, headroom: s.headroom }
    return planServing(doc, index, req)
  }, [doc, index, s, a])
  const tput = useMemo(() => {
    if (!doc || !index || s.mode !== 'serve') return null
    return estimateThroughput(doc, index, s.gpuName, { tp: s.tp, T: a.T, B: a.B, bytes: a.bytes, dtypeLabel: a.dtypeLabel, weights: a.weights, weightBytes: a.weightBytes })
  }, [doc, index, s, a])
  const train = useMemo(() => {
    if (!doc || !index || s.mode !== 'train') return null
    const req: TrainRequest = {
      method: s.method, optimizer: s.optim, loraRank: s.rank, loraTargets: s.targets,
      gpus: s.gpus, gpuMemoryGb: s.mem, sharding: s.sharding, T: a.T, B: a.B,
      gradCheckpoint: s.ckpt, flashAttention: s.flash, headroom: s.headroom, gpu: s.gpuName,
    }
    return planTraining(doc, index, req)
  }, [doc, index, s, a])
  if (!doc || (!plan && !train)) return null

  const tpOptions = [1, 2, 4, 8, 16].filter((x) => x <= 64)
  const cap = (plan ?? train)!.perGpuCapacityBytes
  const pct = (b: number) => (cap > 0 ? Math.min(100, (b / cap) * 100) : 0)
  const cmd =
    s.mode === 'serve'
      ? `uvx modelmap plan ${doc.model_id} --gpus ${s.gpus} --gpu-memory ${s.mem} --tp ${s.tp} --pp ${s.pp} -T ${a.T} -B ${a.B} --dtype ${a.dtypeLabel}${a.weightBytes != null ? ` --weights ${a.weights}` : ''} --gpu "${s.gpuName}"`
      : `uvx modelmap train ${doc.model_id} --method ${s.method}${s.method !== 'full' ? ` --rank ${s.rank} --targets ${s.targets}` : ''} --optimizer ${s.optim} --gpus ${s.gpus} --gpu-memory ${s.mem} --sharding ${s.sharding} -T ${a.T} -B ${a.B} --gpu "${s.gpuName}"`

  return (
    <div className="mm-pop mm-planner" role="dialog" aria-label="Serving planner" ref={ref}>
      <div className="mm-plan-head">
        <span className="mm-plan-tabs" role="tablist" aria-label="Planner mode">
          <button role="tab" aria-selected={s.mode === 'serve'} className={s.mode === 'serve' ? 'is-on' : ''} onClick={() => upd({ mode: 'serve' })}>serve</button>
          <button role="tab" aria-selected={s.mode === 'train'} className={s.mode === 'train' ? 'is-on' : ''} onClick={() => upd({ mode: 'train' })}>fine-tune</button>
        </span>
        <span className={`mm-plan-verdict ${(plan ?? train)!.fits ? 'is-ok' : 'is-over'}`}>{(plan ?? train)!.fits ? 'fits' : 'does not fit'}</span>
      </div>
      <div className="mm-plan-grid">
        <label>GPU
          <select value={s.gpuName} onChange={(e) => pickGpu(e.target.value)}>
            {GPU_PRESETS.map(([name]) => <option key={name} value={name}>{name}</option>)}
          </select>
        </label>
        <label>memory GB <input type="number" min={0} max={100000} value={s.mem} onChange={(e) => upd({ mem: Math.max(0, Number(e.target.value) || 0) })} /></label>
        <label>GPUs <input type="number" min={1} max={4096} value={s.gpus} onChange={(e) => upd({ gpus: Math.max(1, Math.floor(Number(e.target.value) || 1)) })} /></label>
        {s.mode === 'serve' && <label>tensor ∥ <select value={s.tp} onChange={(e) => upd({ tp: Number(e.target.value) })}>{tpOptions.map((x) => <option key={x} value={x}>{x}</option>)}</select></label>}
        {s.mode === 'serve' && <label>pipeline ∥ <input type="number" min={1} max={4096} value={s.pp} onChange={(e) => upd({ pp: Math.max(1, Math.floor(Number(e.target.value) || 1)) })} /></label>}
        {s.mode === 'train' && (
          <label>method
            <select value={s.method} onChange={(e) => upd({ method: e.target.value as PlanSettings['method'] })}>
              <option value="qlora">QLoRA</option>
              <option value="lora">LoRA</option>
              <option value="full">full FT</option>
            </select>
          </label>
        )}
        {s.mode === 'train' && s.method !== 'full' && (
          <label>rank
            <select value={s.rank} onChange={(e) => upd({ rank: Number(e.target.value) })}>
              {[8, 16, 32, 64, 128].map((r) => <option key={r} value={r}>r {r}</option>)}
            </select>
          </label>
        )}
        {s.mode === 'train' && s.method !== 'full' && (
          <label>targets
            <select value={s.targets} onChange={(e) => upd({ targets: e.target.value as PlanSettings['targets'] })}>
              <option value="attention">attention</option>
              <option value="attn-mlp">attn + mlp</option>
              <option value="all-linear">all linear</option>
            </select>
          </label>
        )}
        {s.mode === 'train' && (
          <label>optimizer
            <select value={s.optim} onChange={(e) => upd({ optim: e.target.value as PlanSettings['optim'] })}>
              <option value="adamw">AdamW</option>
              <option value="adamw8bit">AdamW 8-bit</option>
            </select>
          </label>
        )}
        {s.mode === 'train' && (
          <label>sharding
            <select value={s.sharding} onChange={(e) => upd({ sharding: e.target.value as PlanSettings['sharding'] })}>
              <option value="none">none</option>
              <option value="zero2">ZeRO-2</option>
              <option value="zero3">ZeRO-3 / FSDP</option>
            </select>
          </label>
        )}
        {s.mode === 'train' && (
          <label className="mm-plan-checks">
            <span><input type="checkbox" checked={s.ckpt} onChange={(e) => upd({ ckpt: e.target.checked })} /> grad ckpt</span>
            <span><input type="checkbox" checked={s.flash} onChange={(e) => upd({ flash: e.target.checked })} /> flash attn</span>
          </label>
        )}
        <label>headroom <select value={s.headroom} onChange={(e) => upd({ headroom: Number(e.target.value) })}>{[0, 0.05, 0.1, 0.15, 0.2, 0.3].map((h) => <option key={h} value={h}>{Math.round(h * 100)}%</option>)}</select></label>
      </div>
      <p className="mm-plan-assume">at T {fmtInt(a.T)} · B {a.B}{s.mode === 'train' ? '/gpu · bf16 training' : ` · ${a.dtypeLabel}${a.weightBytes != null ? ` · weights ${a.weights}` : ''}`} (change in the assumptions chip) · capacity {fmtBytes(cap)}/GPU</p>
      {plan && <table className="mm-plan-table">
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
      </table>}
      {plan && <p className="mm-plan-summary">
        <b>max context at B {a.B}: {plan.maxContextTokens ? fmtInt(plan.maxContextTokens) + ' tokens' : '—'}</b>
        {plan.stages.length > 1 && <> · {fmtBytes(plan.stages[0].boundaryBytesOut)} crosses each stage boundary per forward</>}
        <> · weights {fmtBytes(plan.weightBytes)} · KV {fmtBytes(plan.kvBytes)} at T</>
      </p>}
      {tput && (
        <p className="mm-plan-summary">
          <b>speed on {tput.gpu}{s.tp > 1 ? ` × ${s.tp}` : ''} ≈ {fmtInt(tput.prefillTokPerSec)} tok/s prefill · {fmtInt(tput.decodeTokPerSecB1)} tok/s decode at B 1</b>
          {a.B > 1 && <> · {fmtInt(tput.decodeTokPerSecAtB)} tok/s total at B {a.B}</>}
        </p>
      )}
      {train && (
        <>
          <table className="mm-plan-table">
            <thead><tr><th>trainable</th><th>weights</th><th>grads</th><th>optimizer</th><th>activations</th><th>total / GPU</th></tr></thead>
            <tbody>
              <tr className={train.fits ? '' : 'is-over'}>
                <td>{fmtParams(train.trainableParams)} <span className="mm-dim">of {fmtParams(train.totalParams)}</span></td>
                <td>{fmtBytes(train.weightBytesPerGpu)}</td>
                <td>{fmtBytes(train.gradBytesPerGpu)}</td>
                <td>{fmtBytes(train.optimizerBytesPerGpu)}</td>
                <td>{fmtBytes(train.activationBytesPerGpu)}</td>
                <td>
                  <div className="mm-plan-bar" title={`${Math.round(pct(train.totalBytesPerGpu))}% of capacity`}>
                    <i className="w" style={{ width: `${pct(train.weightBytesPerGpu)}%` }} />
                    <i className="k" style={{ width: `${pct(train.gradBytesPerGpu + train.optimizerBytesPerGpu)}%` }} />
                    <i className="a" style={{ width: `${pct(train.activationBytesPerGpu)}%` }} />
                  </div>
                  {fmtBytes(train.totalBytesPerGpu)}{train.fits ? '' : ' ✗'}
                </td>
              </tr>
            </tbody>
          </table>
          <p className="mm-plan-summary">
            <b>largest micro-batch at T {fmtInt(a.T)}: {train.maxMicrobatch || '—'}/GPU</b>
            {train.trainTokensPerSec != null && <> · ≈ {fmtInt(train.trainTokensPerSec)} training tok/s across {s.gpus} GPU{s.gpus > 1 ? 's' : ''}</>}
          </p>
        </>
      )}
      <p className="mm-pop-note">{(s.mode === 'serve' ? [...(plan?.notes ?? []), ...(tput?.notes ?? [])] : train?.notes ?? []).join(' · ')}</p>
      <p className="mm-pop-note">
        <code>{cmd}</code>
        <button className="mm-link" onClick={() => void navigator.clipboard.writeText(cmd)} title="copy the CLI command">copy</button>
      </p>
    </div>
  )
}
