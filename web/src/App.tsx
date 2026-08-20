import { useEffect, useState } from 'react'
import { ReactFlowProvider } from '@xyflow/react'
import { Canvas } from './components/Canvas'
import { CompareView } from './components/CompareView'
import { Inspector } from './components/Inspector'
import { Landing } from './components/Landing'
import { LiveBar } from './components/LiveBar'
import { Sheet } from './components/Sheet'
import { TopBar } from './components/TopBar'
import { resetLive } from './live/liveStore'
import { useStore } from './store'

const LOAD_STAGES = [
  'fetching config.json…',
  'instantiating on the meta device…',
  'tracing a fake forward pass…',
  'collapsing repeated blocks…',
  'reading safetensors headers…',
]

function LoadingOverlay({ modelId }: { modelId: string }) {
  const [stage, setStage] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setStage((s) => (s + 1) % LOAD_STAGES.length), 1300)
    return () => clearInterval(t)
  }, [])
  return (
    <div className="mm-overlay">
      <div className="mm-loader">
        <span className="mm-pulse-dot" />
        <div>
          <div className="mm-loader-model">{modelId}</div>
          <div className="mm-loader-stage">{LOAD_STAGES[stage]}</div>
        </div>
      </div>
    </div>
  )
}

/** ?embed=1: chrome-less view for iframes (model cards, blogs, docs) */
export function isEmbed(): boolean {
  return new URL(location.href).searchParams.get('embed') === '1'
}

function urlModel(): string | null {
  const m = location.pathname.match(/^\/m\/(.+)$/)
  return m ? decodeURIComponent(m[1]) : null
}
/** /compare/{A}...{B} */
export function urlCompare(): [string, string] | null {
  const m = location.pathname.match(/^\/compare\/(.+)$/)
  if (!m) return null
  const [a, b] = decodeURIComponent(m[1]).split('...')
  return a && b ? [a, b] : null
}

export default function App() {
  const doc = useStore((s) => s.doc)
  const loading = useStore((s) => s.loading)
  // a different model means a different live session (weights, tokenizer, cache)
  const modelId = doc?.model_id
  useEffect(() => {
    resetLive()
  }, [modelId])
  const error = useStore((s) => s.error)
  const errorModel = useStore((s) => s.errorModel)
  const toast = useStore((s) => s.toast)
  const loadModel = useStore((s) => s.loadModel)
  const select = useStore((s) => s.select)
  const [compare, setCompare] = useState<[string, string] | null>(() => urlCompare())
  const embed = isEmbed()

  useEffect(() => {
    if (embed) document.documentElement.dataset.embed = '1'
    const boot = async () => {
      const id = urlModel()
      if (!id) return
      await loadModel(id, { push: false })
      const params = new URL(location.href).searchParams
      const sel = params.get('sel')
      if (sel != null) select(sel)
      if (params.get('mode') === 'flow' && (useStore.getState().doc?.trace.length ?? 0) > 0) {
        const { useFlowStore } = await import('./flow/flowStore')
        useFlowStore.getState().activate()
      }
    }
    void boot()
    const onPop = () => {
      setCompare(urlCompare())
      const id = urlModel()
      if (id && id !== useStore.getState().doc?.model_id) void loadModel(id, { push: false })
    }
    const onNav = () => setCompare(urlCompare())
    window.addEventListener('mm:navigate', onNav)
    window.addEventListener('popstate', onPop)
    return () => {
      window.removeEventListener('popstate', onPop)
      window.removeEventListener('mm:navigate', onNav)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const fullUrl = (() => {
    const u = new URL(location.href)
    u.searchParams.delete('embed')
    return u.toString()
  })()

  return (
    <ReactFlowProvider>
      <div className={`mm-app ${embed ? 'is-embed' : ''}`}>
        {!embed && <TopBar />}
        <div className="mm-main">
          {compare ? (
            <CompareView idA={compare[0]} idB={compare[1]} />
          ) : doc ? (
            <>
              <Canvas />
              {!embed && <LiveBar />}
              {!embed && (
                <Sheet>
                  <Inspector />
                </Sheet>
              )}
              {embed && (
                <a className="mm-embed-badge" href={fullUrl} target="_blank" rel="noreferrer" title="Open the full interactive map on modelmap">
                  <b>modelmap</b> {doc.model_id} <span className="mm-dim">↗</span>
                </a>
              )}
            </>
          ) : (
            !loading && !error && <Landing />
          )}
          {loading && <LoadingOverlay modelId={loading} />}
          {toast && (
            <div className="mm-toast" role="status">
              {toast}
            </div>
          )}
          {error && (
            <div className="mm-error" role="alert">
              <strong>Could not load this model.</strong>
              <p>{error}</p>
              {errorModel && (
                <button className="mm-btn mm-error-retry" onClick={() => void loadModel(errorModel)}>
                  try again
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </ReactFlowProvider>
  )
}
