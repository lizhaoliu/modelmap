import { useMemo } from 'react'
import { useFlowStore } from '../flow/flowStore'
import type { MicroScript } from '../flow/micro'
import { Shape } from './Shape'
import { useStore } from '../store'

/** Animated strip of a beat's inner stages. CSS-only motion keyed to the
 *  beat: stages light up in sequence and a dot travels the row; duration
 *  follows playback speed, play-state follows pause. Remounted per beat via
 *  the parent's `key`, with a negative delay when joined mid-beat. */
export function MicroView({
  script,
  beatDur,
  elapsed,
}: {
  script: MicroScript
  beatDur: number
  elapsed: number
}) {
  const speed = useFlowStore((s) => s.speed)
  const playing = useFlowStore((s) => s.playing)
  const index = useStore((s) => s.index)
  const n = script.stages.length
  const dur = Math.max(0.6, beatDur / speed)
  const style = useMemo(
    () =>
      ({
        '--mm-dur': `${dur}s`,
        '--mm-n': n,
        '--mm-delay': `${-Math.max(0, elapsed / speed)}s`,
        animationPlayState: playing ? 'running' : 'paused',
      }) as React.CSSProperties,
    [dur, n, elapsed, speed, playing],
  )
  if (!index) return null
  return (
    <div className={`mm-micro ${playing ? '' : 'is-paused'}`} style={style} aria-label={script.title}>
      <div className="mm-micro-title">{script.title}</div>
      <div className="mm-micro-row">
        {script.stages.map((st, i) => (
          <div className="mm-micro-seg" key={i}>
            <div className={`mm-micro-stage kind-${st.kind}`} style={{ '--i': i } as React.CSSProperties}>
              <div className="mm-micro-label">{st.label}</div>
              <div className="mm-micro-shapes">
                {st.shapes.map((sh, j) => (
                  <div key={j}>
                    <Shape shape={sh} labels={index.dimLabels} batch={index.traceBatch} />
                  </div>
                ))}
              </div>
              <div className="mm-micro-note">{st.note}</div>
            </div>
            {i < n - 1 && <div className="mm-micro-arrow">→</div>}
          </div>
        ))}
        <div className="mm-micro-dot" aria-hidden="true" />
      </div>
    </div>
  )
}
