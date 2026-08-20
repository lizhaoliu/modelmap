import { useEffect, useState } from 'react'
import { fetchGallery, fetchHealth, gotoCompare, type Gallery, type GalleryEntry } from '../api'
import { fmtCount, fmtParams } from '../fmt'
import { useStore } from '../store'
import { HeroFlow } from './HeroFlow'
import { ModelSearch } from './ModelSearch'

export function Landing() {
  const loadModel = useStore((s) => s.loadModel)
  const [gallery, setGallery] = useState<Gallery | null>(null)
  const [cmpA, setCmpA] = useState('')
  const [allowLocal, setAllowLocal] = useState(false)
  const [localPath, setLocalPath] = useState('')
  useEffect(() => {
    void fetchGallery().then(setGallery)
    void fetchHealth().then((h) => setAllowLocal(Boolean(h?.allow_local)))
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
        <HeroFlow />
        <ModelSearch big />
      </div>
      {gallery === null && <p className="mm-gallery-loading">loading gallery…</p>}
      {gallery && gallery.trending.length > 0 && (
        <section className="mm-gallery-block" aria-label="Trending on Hugging Face">
          <h2 className="mm-gallery-h">Trending on Hugging Face <span className="mm-dim">right now · ungated, transformers-loadable</span></h2>
          <div className="mm-gallery">
            {gallery.trending.map((g, i) => <Card key={g.id} g={g} rank={i + 1} loadModel={loadModel} openFlow={openFlow} />)}
          </div>
        </section>
      )}
      {gallery && (
        <section className="mm-gallery-block" aria-label="Classics">
          <h2 className="mm-gallery-h">Classics <span className="mm-dim">reference architectures, always instant</span></h2>
          <div className="mm-gallery">
            {gallery.classics.map((g) => <Card key={g.id} g={g} loadModel={loadModel} openFlow={openFlow} />)}
          </div>
        </section>
      )}
      {allowLocal && (
        <form
          className="mm-local-entry"
          aria-label="Open a local checkpoint"
          onSubmit={(e) => {
            e.preventDefault()
            const p = localPath.trim()
            if (p) void loadModel(p.startsWith('local:') ? p : `local:${p}`)
          }}
        >
          <span className="mm-cmp-entry-label">local</span>
          <input
            value={localPath}
            onChange={(e) => setLocalPath(e.target.value)}
            placeholder="/path/to/checkpoint-dir, model.safetensors or model.gguf"
            spellCheck={false}
            aria-label="Local checkpoint path"
          />
          <button className="mm-btn" type="submit">open</button>
          <span className="mm-cmp-entry-hint">this server runs on your machine, so your own fine-tunes and GGUFs work too</span>
        </form>
      )}
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

function Card({ g, rank, loadModel, openFlow }: { g: GalleryEntry; rank?: number; loadModel: (id: string) => Promise<void>; openFlow: (id: string) => void }) {
  const blurb =
    g.blurb ??
    [g.pipeline_tag, g.downloads != null && `↓ ${fmtCount(g.downloads)}`, g.likes != null && `♥ ${fmtCount(g.likes)}`]
      .filter(Boolean)
      .join(' · ')
  return (
    <article className={`mm-card ${g.cached ? '' : 'is-cold'}`}>
      <button className="mm-card-main" onClick={() => void loadModel(g.id)}>
        <span className="mm-card-id">
          {rank != null && <i className="mm-card-rank">{rank}</i>}
          {g.id}
        </span>
        <span className="mm-card-blurb">{blurb}</span>
        <span className="mm-card-meta">
          {g.summary ? (
            <>
              <b>{fmtParams(g.summary.params_total)}</b> params · {g.summary.architecture ?? 'weights view'}
              {g.summary.fidelity !== 'full' && ` · ${g.summary.fidelity}`}
            </>
          ) : (
            <>{g.architecture ? `${g.architecture} · ` : ''}first visit extracts it (a few seconds)</>
          )}
        </span>
      </button>
      {(g.summary?.trace_steps ?? 1) > 0 && (
        <button className="mm-card-flow" title="Replay the forward pass" onClick={() => openFlow(g.id)}>
          ▶ flow
        </button>
      )}
    </article>
  )
}
