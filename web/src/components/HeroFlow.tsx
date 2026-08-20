import { useEffect, useState } from 'react'

/** The landing hero is itself a tiny flow replay (design doc §19): a canned
 *  decoder map with a pulse making laps — embed → blocks ×12 → norm → head —
 *  so the first paint of modelmap is the animation, not a wall of cards.
 *  Pure CSS keyframes on one shared loop; JS only cycles the caption.
 *  `prefers-reduced-motion` freezes it into a static diagram. */

const LOOP_S = 9

// caption windows line up with the pulse's --d delays below
const CAPTIONS: [number, string][] = [
  [0.0, '“The cat sat on the” → token ids'],
  [0.12, 'each id becomes a 4096-number vector'],
  [0.3, 'attention mixes context · MLP transforms it — ×12 blocks'],
  [0.62, 'normalize, keep scales stable'],
  [0.78, 'project to 50k vocabulary scores'],
  [0.9, '“mat” — the next token'],
]

const STOPS = [
  { x: 3, w: 17, label: 'tokens', cls: 'tok' },
  { x: 24, w: 14, label: 'embed', cls: 'embedding' },
  { x: 42, w: 16, label: 'blocks', cls: 'attention', stack: true },
  { x: 62, w: 11, label: 'norm', cls: 'norm' },
  { x: 77, w: 13, label: 'lm_head', cls: 'head' },
]

export function HeroFlow() {
  const [tick, setTick] = useState(0)
  useEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const t0 = performance.now()
    const t = setInterval(() => setTick(((performance.now() - t0) / 1000 / LOOP_S) % 1), 250)
    return () => clearInterval(t)
  }, [])
  const caption = CAPTIONS.reduce((acc, [at, text]) => (tick >= at ? text : acc), CAPTIONS[0][1])

  return (
    <div className="mm-hero" aria-hidden="true" style={{ '--loop': `${LOOP_S}s` } as React.CSSProperties}>
      <div className="mm-hero-canvas">
        <div className="mm-hero-wire" />
        {STOPS.map((s, i) => (
          <div
            key={s.label}
            className={`mm-hero-node kind-${s.cls} ${s.stack ? 'is-stack' : ''}`}
            style={{ left: `${s.x}%`, width: `${s.w}%`, '--d': `${i / STOPS.length}` } as React.CSSProperties}
          >
            {s.cls === 'tok' ? (
              <span className="mm-hero-toks">
                {['The', 'cat', 'sat', 'on'].map((t, j) => (
                  <i key={j} style={{ '--j': j } as React.CSSProperties}>{t}</i>
                ))}
              </span>
            ) : (
              <>
                <b>{s.label.split(' ')[0]}</b>
                {s.stack && <span className="mm-hero-badge">×12</span>}
              </>
            )}
          </div>
        ))}
        <div className="mm-hero-pulse" />
        {/* rising logits at the far end */}
        <div className="mm-hero-logits">
          {[0.9, 0.35, 0.2, 0.12, 0.07].map((h, i) => (
            <i key={i} style={{ '--h': h, '--j': i } as React.CSSProperties} />
          ))}
        </div>
        <span className="mm-hero-out">mat</span>
      </div>
      <p className="mm-hero-caption" key={caption}>{caption}</p>
    </div>
  )
}
