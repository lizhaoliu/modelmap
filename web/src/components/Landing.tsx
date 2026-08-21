import { useEffect, useState } from 'react'
import { fetchGallery, fetchHealth, gotoCompare, type Gallery, type GalleryEntry } from '../api'
import { fmtCount, fmtParams } from '../fmt'
import { useStore } from '../store'
import { useCostStore } from '../analytics/costStore'
import { GPU_PRESETS } from '../analytics/plan'
import { usePlanStore } from '../analytics/planStore'
import { HeroFlow } from './HeroFlow'
import { ModelSearch } from './ModelSearch'

/** open a model straight into the vram lens for the chosen GPU (§26) */
export async function openFit(loadModel: (id: string) => Promise<void>, id: string, gpuName?: string) {
  if (gpuName) usePlanStore.getState().pickGpu(gpuName)
  await loadModel(id)
  if (useStore.getState().doc) {
    useCostStore.getState().setLens('vram')
    usePlanStore.getState().upd({}) // loadModel replaced the URL: put the GPU back into it
  }
}

export function Landing() {
  const loadModel = useStore((s) => s.loadModel)
  const [gallery, setGallery] = useState<Gallery | null>(null)
  const [cmpA, setCmpA] = useState('')
  const gpuName = usePlanStore((st) => st.s.gpuName)
  const pickGpu = usePlanStore((st) => st.pickGpu)
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
          Paste a Hugging Face model id. See the whole network, watch a forward pass, and find out whether it fits on your GPU — no weights downloaded.
        </p>
        <HeroFlow />
        <ModelSearch big />
        <form className="mm-fit-entry" aria-label="Will it fit on my GPU?" onSubmit={(e) => e.preventDefault()}>
          <span className="mm-fit-q">will it fit on</span>
          <select value={gpuName} onChange={(e) => pickGpu(e.target.value)} aria-label="GPU">
            {GPU_PRESETS.map(([name]) => <option key={name} value={name}>{name}</option>)}
          </select>
          <span className="mm-fit-q">?</span>
          <ModelSearch placeholder="model id" onPick={(id) => void openFit(loadModel, id, gpuName)} />
          <span className="mm-cmp-entry-hint">
            weights + KV cache + activations on the map itself · try{' '}
            <button className="mm-link" onClick={() => void openFit(loadModel, 'Qwen/Qwen3-8B', gpuName)}>Qwen3-8B</button>
            {' '}or{' '}
            <button className="mm-link" onClick={() => void openFit(loadModel, 'deepseek-ai/DeepSeek-V3.1', gpuName)}>DeepSeek-V3.1</button>
          </span>
        </form>
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
      <section className="mm-cmp-entry" aria-label="Browse the zoo">
        <span className="mm-cmp-entry-label">families</span>
        {['qwen', 'llama', 'deepseek', 'gpt', 'mistral', 'bert', 't5', 'whisper'].map((k) => (
          <button key={k} className="mm-zoo-tagbtn" onClick={() => { history.pushState({}, '', `/arch/${k}`); window.dispatchEvent(new Event('mm:navigate')) }}>
            {k}
          </button>
        ))}
        <button className="mm-link" onClick={() => { history.pushState({}, '', '/models'); window.dispatchEvent(new Event('mm:navigate')) }}>
          all mapped models →
        </button>
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
        Repos that need <code>trust_remote_code</code>: run <code>modelmap dump &lt;id&gt; --trust-remote-code</code> on your machine and drop the <code>.graph.json</code> anywhere on this page.
      </p>
      <p className="mm-landing-links">
        <a href="https://github.com/lizhaoliu/modelmap" target="_blank" rel="noreferrer">
          <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" aria-hidden="true">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>
          </svg>
          open source (MIT) — github.com/lizhaoliu/modelmap
        </a>
        <span className="mm-dim">·</span>
        <a href="/docs" target="_blank" rel="noreferrer">REST API</a>
        <span className="mm-dim">·</span>
        <a href="https://github.com/lizhaoliu/modelmap/blob/main/docs/API.md" target="_blank" rel="noreferrer">CLI · Python · MCP</a>
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
      <span className="mm-card-actions">
        {(g.summary?.trace_steps ?? 1) > 0 && (
          <button className="mm-card-flow" title="Replay the forward pass" onClick={() => openFlow(g.id)}>
            ▶ flow
          </button>
        )}
        <button className="mm-card-flow mm-card-fit" title="Does it fit on my GPU? Opens the vram lens" onClick={() => void openFit(loadModel, g.id)}>
          fit?
        </button>
      </span>
    </article>
  )
}
