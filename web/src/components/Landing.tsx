import { useEffect, useState } from 'react'
import { fetchGallery, gotoCompare, type GalleryEntry } from '../api'
import { fmtParams } from '../fmt'
import { useStore } from '../store'
import { ModelSearch } from './ModelSearch'

export function Landing() {
  const loadModel = useStore((s) => s.loadModel)
  const [gallery, setGallery] = useState<GalleryEntry[] | null>(null)
  const [cmpA, setCmpA] = useState('')
  useEffect(() => {
    void fetchGallery().then(setGallery)
  }, [])

  const openFlow = (id: string) => {
    void loadModel(id).then(async () => {
      if ((useStore.getState().doc?.trace.length ?? 0) > 0) {
        const { useFlowStore } = await import('../flow/flowStore')
        useFlowStore.getState().activate()
      }
    })
  }

  return (
    <div className="mm-landing">
      <div className="mm-landing-hero">
        <h1 className="mm-wordmark">modelmap</h1>
        <p className="mm-tagline">
          Paste a Hugging Face model id. Get a living map of the network — no weights downloaded.
        </p>
        <ModelSearch big />
      </div>
      <section className="mm-gallery" aria-label="Gallery">
        {gallery === null && <p className="mm-gallery-loading">loading gallery…</p>}
        {gallery?.map((g) => (
          <article className={`mm-card ${g.cached ? '' : 'is-cold'}`} key={g.id}>
            <button className="mm-card-main" onClick={() => void loadModel(g.id)}>
              <span className="mm-card-id">{g.id}</span>
              <span className="mm-card-blurb">{g.blurb}</span>
              <span className="mm-card-meta">
                {g.summary ? (
                  <>
                    <b>{fmtParams(g.summary.params_total)}</b> params · {g.summary.architecture ?? 'weights view'}
                    {g.summary.fidelity !== 'full' && ` · ${g.summary.fidelity}`}
                  </>
                ) : (
                  'first visit extracts it (a few seconds)'
                )}
              </span>
            </button>
            {(g.summary?.trace_steps ?? 1) > 0 && (
              <button className="mm-card-flow" title="Replay the forward pass" onClick={() => openFlow(g.id)}>
                ▶ flow
              </button>
            )}
          </article>
        ))}
      </section>
      <section className="mm-cmp-entry" aria-label="Compare two models">
        <span className="mm-cmp-entry-label">compare</span>
        <ModelSearch placeholder={cmpA || 'first model'} onPick={(id) => setCmpA(id)} />
        <span className="mm-vs">vs</span>
        <ModelSearch placeholder="second model" onPick={(id) => cmpA && gotoCompare(cmpA, id)} />
        <span className="mm-cmp-entry-hint">e.g. <button className="mm-link" onClick={() => gotoCompare('Qwen/Qwen2.5-7B', 'Qwen/Qwen3-8B')}>Qwen2.5-7B vs Qwen3-8B</button></span>
      </section>
      <p className="mm-landing-foot">
        Structure comes from a meta-device instantiation; shapes come from a traced fake forward pass. Any public repo works — gated ones after you add a token.
      </p>
    </div>
  )
}
