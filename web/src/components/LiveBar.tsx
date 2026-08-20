import { useEffect, useMemo, useRef, useState } from 'react'
import { LIVE_PICKS, liveSupport, useLiveStore } from '../live/liveStore'
import { fmtParams } from '../fmt'
import { gotoModel } from '../api'
import { useStore } from '../store'

/** Live mode (design doc §20): download the weights into the browser, type a
 *  prompt, and the map stops being a diagram — real next-token probabilities,
 *  real per-head attention, a logit lens over the layers. */

function fmtMB(b: number): string {
  return b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${Math.round(b / 1e6)} MB`
}

const PHASES: Record<string, string> = {
  config: 'fetching config + tokenizer…',
  weights: 'downloading weights',
  building: 'building the model…',
}

export function LiveBar() {
  const doc = useStore((s) => s.doc)
  const selected = useStore((s) => s.selected)
  const open = useLiveStore((s) => s.open)
  const status = useLiveStore((s) => s.status)
  const error = useLiveStore((s) => s.error)
  const progress = useLiveStore((s) => s.progress)
  const info = useLiveStore((s) => s.info)
  const prompt = useLiveStore((s) => s.prompt)
  const setPrompt = useLiveStore((s) => s.setPrompt)
  const tokens = useLiveStore((s) => s.tokens)
  const topk = useLiveStore((s) => s.topk)
  const lens = useLiveStore((s) => s.lens)
  const tokMs = useLiveStore((s) => s.tokMs)
  const temperature = useLiveStore((s) => s.temperature)
  const load = useLiveStore((s) => s.load)
  const run = useLiveStore((s) => s.run)
  const generate = useLiveStore((s) => s.generate)
  const stop = useLiveStore((s) => s.stop)
  const close = useLiveStore((s) => s.close)
  const sup = useMemo(() => liveSupport(doc), [doc])
  const tokensEndRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    tokensEndRef.current?.scrollIntoView({ block: 'nearest', inline: 'end' })
  }, [tokens.length])

  // clicking an attention node on the canvas retargets the heatmap layer
  const setAttnLayer = useLiveStore((s) => s.setAttnLayer)
  useEffect(() => {
    if (!selected || !info) return
    const m = /(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)/.exec(selected)
    if (m) {
      const l = Number(m[1])
      if (l < info.layers && l !== useLiveStore.getState().attnLayer) setAttnLayer(l)
    }
  }, [selected, info, setAttnLayer])

  if (!open || !doc) return null

  return (
    <div className="mm-livebar" role="region" aria-label="Live inference">
      <div className="mm-live-head">
        <span className="mm-live-title">⚡ live</span>
        <span className="mm-live-sub">
          {info
            ? <>{fmtParams(info.params)} params on your CPU{tokMs != null && ` · ${tokMs} ms/token`}</>
            : 'real inference in your browser — the weights download to you; the server never sees them'}
        </span>
        <button className="mm-btn mm-live-close" onClick={close} title="Close live mode">✕</button>
      </div>

      {status === 'idle' && sup.ok && (
        <div className="mm-live-cta">
          <button className="mm-btn mm-btn-primary" onClick={() => load(doc.model_id)}>
            download weights · {sup.sizeMB} MB
          </button>
          <span className="mm-live-note">one fetch from huggingface.co, kept in memory only</span>
        </div>
      )}
      {status === 'idle' && !sup.ok && (
        <div className="mm-live-cta">
          <span className="mm-live-note">{sup.reason}. Try one of these instead:</span>
          <span className="mm-live-picks">
            {LIVE_PICKS.map((p) => (
              <button key={p.id} className="mm-btn" title={p.blurb} onClick={() => gotoModel(p.id)}>
                {p.id.split('/')[1]} · {p.size}
              </button>
            ))}
          </span>
        </div>
      )}

      {status === 'loading' && progress && (
        <div className="mm-live-progress">
          <span>{PHASES[progress.phase] ?? progress.phase}</span>
          {progress.phase === 'weights' && progress.total > 0 && (
            <>
              <span className="mm-live-bar"><i style={{ width: `${(100 * progress.loaded) / progress.total}%` }} /></span>
              <span className="mm-dim">{fmtMB(progress.loaded)} / {fmtMB(progress.total)}</span>
            </>
          )}
        </div>
      )}
      {error && <p className="mm-live-error">{error}</p>}

      {info && (
        <>
          <form
            className="mm-live-promptrow"
            onSubmit={(e) => {
              e.preventDefault()
              run()
            }}
          >
            <input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="type a prompt…"
              spellCheck={false}
              aria-label="Prompt"
              disabled={status === 'generating' || status === 'running'}
            />
            <button className="mm-btn mm-btn-primary" type="submit" disabled={status !== 'ready'}>
              {status === 'running' ? 'running…' : 'run ▸'}
            </button>
            {status === 'generating' ? (
              <button className="mm-btn" type="button" onClick={stop}>stop</button>
            ) : (
              <button className="mm-btn" type="button" disabled={status !== 'ready' || !tokens.length} onClick={() => generate(24)} title="Sample 24 tokens and watch them ripple through the map">
                generate ▶
              </button>
            )}
            <select
              value={temperature}
              onChange={(e) => useLiveStore.setState({ temperature: Number(e.target.value) })}
              title="Sampling temperature (0 = greedy)"
              aria-label="Temperature"
            >
              <option value={0}>greedy</option>
              <option value={0.7}>t 0.7</option>
              <option value={1}>t 1.0</option>
            </select>
          </form>

          {tokens.length > 0 && (
            <div className="mm-live-grid">
              <div className="mm-live-left">
                <div className="mm-live-toks" aria-label="Tokens">
                  {tokens.map((t, i) => (
                    <i key={i} className={i >= tokens.length - 1 && status === 'generating' ? 'is-new' : ''}>{t}</i>
                  ))}
                  <div ref={tokensEndRef} />
                </div>
                <div className="mm-live-topk" aria-label="Next-token probabilities">
                  <span className="mm-live-lbl">next token</span>
                  {topk.slice(0, 5).map((e) => (
                    <span key={e.id} className="mm-live-cand" title={`p = ${(e.p * 100).toFixed(2)}%`}>
                      <i style={{ width: `${Math.max(2, e.p * 100)}%` }} />
                      <b>{e.tok}</b>
                      <em>{(e.p * 100).toFixed(e.p >= 0.1 ? 0 : 1)}%</em>
                    </span>
                  ))}
                </div>
                {lens && <LensStrip />}
              </div>
              <AttnHeatmap />
            </div>
          )}
        </>
      )}
    </div>
  )
}

/** Logit lens: what the model would predict if each layer were the last —
 *  watch the answer sharpen layer by layer. */
function LensStrip() {
  const lens = useLiveStore((s) => s.lens)
  const stale = useLiveStore((s) => s.lensStale)
  if (!lens) return null
  return (
    <div className={`mm-live-lens ${stale ? 'is-stale' : ''}`} aria-label="Logit lens">
      <span className="mm-live-lbl" title="Project each layer's hidden state straight to vocabulary scores — the prediction forming through the depth of the model">
        logit lens
      </span>
      <div className="mm-live-lensrow">
        {lens.map((r) => (
          <span
            key={r.layer}
            className="mm-live-lenscell"
            style={{ '--p': r.top[0]?.p ?? 0 } as React.CSSProperties}
            title={`after layer ${r.layer}:\n${r.top.map((t) => `${t.tok}  ${(t.p * 100).toFixed(1)}%`).join('\n')}`}
          >
            <em>{r.layer}</em>
            <b>{r.top[0]?.tok ?? ''}</b>
          </span>
        ))}
      </div>
    </div>
  )
}

const CELL_MAX = 14

function AttnHeatmap() {
  const attn = useLiveStore((s) => s.attn)
  const tokens = useLiveStore((s) => s.tokens)
  const info = useLiveStore((s) => s.info)
  const layer = useLiveStore((s) => s.attnLayer)
  const head = useLiveStore((s) => s.attnHead)
  const setLayer = useLiveStore((s) => s.setAttnLayer)
  const setHead = useLiveStore((s) => s.setAttnHead)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<string | null>(null)

  useEffect(() => {
    const cv = canvasRef.current
    if (!cv || !attn) return
    const { seq, heads, data } = attn
    const cell = Math.max(3, Math.min(CELL_MAX, Math.floor(280 / seq)))
    const pad = 2
    const W = seq * cell + pad * 2
    cv.width = W * devicePixelRatio
    cv.height = W * devicePixelRatio
    cv.style.width = `${W}px`
    cv.style.height = `${W}px`
    const ctx = cv.getContext('2d')!
    ctx.scale(devicePixelRatio, devicePixelRatio)
    const styles = getComputedStyle(document.documentElement)
    const paper = styles.getPropertyValue('--code-bg').trim() || '#0D1117'
    ctx.fillStyle = paper
    ctx.fillRect(0, 0, W, W)
    const at = (h: number, q: number, k: number) => data[h * seq * seq + q * seq + k]
    for (let qi = 0; qi < seq; qi++) {
      for (let ki = 0; ki <= qi; ki++) {
        let v = 0
        if (head === -1) {
          for (let h = 0; h < heads; h++) v += at(h, qi, ki)
          v /= heads
        } else v = at(head, qi, ki)
        const a = Math.min(1, Math.pow(v, 0.6))
        ctx.fillStyle = `rgba(224, 138, 0, ${a})`
        ctx.fillRect(pad + ki * cell, pad + qi * cell, cell - 0.5, cell - 0.5)
      }
    }
  }, [attn, head])

  if (!info) return null
  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!attn) return
    const rect = e.currentTarget.getBoundingClientRect()
    const cell = Math.max(3, Math.min(CELL_MAX, Math.floor(280 / attn.seq)))
    const qi = Math.floor((e.clientY - rect.top - 2) / cell)
    const ki = Math.floor((e.clientX - rect.left - 2) / cell)
    if (qi >= 0 && ki >= 0 && qi < attn.seq && ki <= qi) {
      const v =
        head === -1
          ? Array.from({ length: attn.heads }, (_, h) => attn.data[h * attn.seq * attn.seq + qi * attn.seq + ki]).reduce((a, b) => a + b, 0) / attn.heads
          : attn.data[head * attn.seq * attn.seq + qi * attn.seq + ki]
      setHover(`"${tokens[qi] ?? qi}" ← "${tokens[ki] ?? ki}" · ${(v * 100).toFixed(1)}%`)
    } else setHover(null)
  }

  return (
    <div className="mm-live-attn" aria-label="Attention heatmap">
      <div className="mm-live-attn-head">
        <span className="mm-live-lbl" title="Each row: where that token looked. Click an attention block on the map to jump to its layer.">
          attention
        </span>
        <label>
          layer {layer}
          <input type="range" min={0} max={info.layers - 1} value={layer} onChange={(e) => setLayer(Number(e.target.value))} aria-label="Attention layer" />
        </label>
        <select value={head} onChange={(e) => setHead(Number(e.target.value))} aria-label="Attention head">
          <option value={-1}>mean of {info.heads} heads</option>
          {Array.from({ length: info.heads }, (_, h) => (
            <option key={h} value={h}>head {h}</option>
          ))}
        </select>
      </div>
      <canvas ref={canvasRef} onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
      <span className="mm-live-hover">{hover ?? 'rows attend to columns · hover for values'}</span>
    </div>
  )
}
