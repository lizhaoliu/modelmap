import { fmtShape, leafName } from '../fmt'
import type { FlowScript } from '../flow/beats'
import { caption } from '../flow/captions'
import type { FlowApi } from '../flow/engine'
import { useFlowStore } from '../flow/flowStore'
import { useStore } from '../store'

const SPEEDS = [0.5, 1, 2, 4]

function fmtTime(s: number): string {
  const m = Math.floor(s / 60)
  return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}`
}

export function FlowBar({ script, api }: { script: FlowScript; api: FlowApi }) {
  const active = useFlowStore((s) => s.active)
  const playing = useFlowStore((s) => s.playing)
  const speed = useFlowStore((s) => s.speed)
  const beatIdx = useFlowStore((s) => s.beatIdx)
  const tCoarse = useFlowStore((s) => s.tCoarse)
  const total = useFlowStore((s) => s.total)
  const play = useFlowStore((s) => s.play)
  const pause = useFlowStore((s) => s.pause)
  const setSpeed = useFlowStore((s) => s.setSpeed)
  const deactivate = useFlowStore((s) => s.deactivate)
  const doc = useStore((s) => s.doc)
  const index = useStore((s) => s.index)

  if (!active || !doc || !index || !script.beats.length) return null
  const beat = script.beats[Math.min(beatIdx, script.beats.length - 1)]
  const node = index.byId.get(beat.node)
  const inS = beat.inShapes[0] ? fmtShape(beat.inShapes[0]) : '—'
  const outS = beat.outShapes[0] ? fmtShape(beat.outShapes[0]) : '—'

  return (
    <div className="mm-flowbar-wrap">
      <div className="mm-hud">
        <span className="mm-hud-name">{leafName(beat.node)}</span>
        {beat.member && (
          <span className="mm-hud-member">
            layer {beat.member.ordinal} / {beat.member.count}
          </span>
        )}
        <span className="mm-hud-shapes">
          {inS} <span className="mm-hud-arrow">→</span> {outS}
        </span>
        {node && <span className="mm-hud-caption">{caption(node, doc, beat)}</span>}
      </div>
      <div className="mm-flowbar">
        <button
          className="mm-flow-play"
          onClick={() => (playing ? pause() : play())}
          title="Play / pause (Space)"
          aria-label={playing ? 'Pause' : 'Play'}
        >
          {playing ? '❚❚' : '▶'}
        </button>
        <input
          type="range"
          className="mm-flow-scrub"
          min={0}
          max={total}
          step={0.01}
          value={Math.min(tCoarse, total)}
          onChange={(e) => api.seek(Number(e.target.value))}
          aria-label="Replay position"
        />
        <span className="mm-flow-time">
          {fmtTime(tCoarse)} / {fmtTime(total)}
        </span>
        <span className="mm-flow-step">
          beat {beatIdx + 1}/{script.beats.length}
        </span>
        {doc.fidelity === 'structural' && (
          <span className="mm-flow-approx" title={doc.notes.join('\n')}>
            approximate order
          </span>
        )}
        <button
          className="mm-btn"
          onClick={() => setSpeed(SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length])}
          title="Playback speed"
        >
          {speed}×
        </button>
        <button className="mm-btn" onClick={deactivate} title="Exit flow mode (Esc)">
          ✕
        </button>
      </div>
    </div>
  )
}
