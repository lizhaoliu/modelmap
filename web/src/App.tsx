import { useEffect, useState } from 'react'
import { ReactFlowProvider } from '@xyflow/react'
import { Canvas } from './components/Canvas'
import { Inspector } from './components/Inspector'
import { Landing } from './components/Landing'
import { TopBar } from './components/TopBar'
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

function urlModel(): string | null {
  const m = location.pathname.match(/^\/m\/(.+)$/)
  return m ? decodeURIComponent(m[1]) : null
}

export default function App() {
  const doc = useStore((s) => s.doc)
  const loading = useStore((s) => s.loading)
  const error = useStore((s) => s.error)
  const loadModel = useStore((s) => s.loadModel)
  const select = useStore((s) => s.select)

  useEffect(() => {
    const boot = async () => {
      const id = urlModel()
      if (!id) return
      await loadModel(id, { push: false })
      const sel = new URL(location.href).searchParams.get('sel')
      if (sel != null) select(sel)
    }
    void boot()
    const onPop = () => {
      const id = urlModel()
      if (id && id !== useStore.getState().doc?.model_id) void loadModel(id, { push: false })
    }
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <ReactFlowProvider>
      <div className="mm-app">
        <TopBar />
        <div className="mm-main">
          {doc ? (
            <>
              <Canvas />
              <Inspector />
            </>
          ) : (
            !loading && !error && <Landing />
          )}
          {loading && <LoadingOverlay modelId={loading} />}
          {error && (
            <div className="mm-error" role="alert">
              <strong>Could not load this model.</strong>
              <p>{error}</p>
            </div>
          )}
        </div>
      </div>
    </ReactFlowProvider>
  )
}
