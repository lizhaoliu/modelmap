import { useEffect, useState } from 'react'
import { ReactFlowProvider } from '@xyflow/react'
import { Canvas } from './components/Canvas'
import { CompareView } from './components/CompareView'
import { Inspector } from './components/Inspector'
import { Landing } from './components/Landing'
import { LiveBar } from './components/LiveBar'
import { CatalogView, FamilyView } from './components/Zoo'
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
function urlZoo(): { page: 'models' } | { page: 'arch'; key: string } | null {
  if (location.pathname === '/models') return { page: 'models' }
  const m = location.pathname.match(/^\/arch\/([\w-]+)$/)
  return m ? { page: 'arch', key: m[1] } : null
}

/** /compare/{A}...{B} */
export function urlCompare(): [string, string] | null {
  const m = location.pathname.match(/^\/compare\/(.+)$/)
  if (!m) return null
  const [a, b] = decodeURIComponent(m[1]).split('...')
  return a && b ? [a, b] : null
}

/** Read a dropped .graph.json (or .graph.json.gz) into a document. */
async function readDroppedGraph(file: File): Promise<unknown> {
  if (file.name.endsWith('.gz')) {
    const ds = new DecompressionStream('gzip')
    const text = await new Response(file.stream().pipeThrough(ds)).text()
    return JSON.parse(text)
  }
  return JSON.parse(await file.text())
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
  const [compare, setCompare] = useState<[string, string] | null>(() => urlCompare())
  const [zoo, setZoo] = useState(() => urlZoo())
  const embed = isEmbed()
  const [dropHover, setDropHover] = useState(false)
  const loadDocFromFile = useStore((s) => s.loadDocFromFile)
  const setToast = useStore((s) => s.setToast)

  // drag a `modelmap dump` file anywhere onto the app — the client renders it
  // without a server round trip (how trust_remote_code models get their map)
  useEffect(() => {
    if (embed) return
    let depth = 0
    const over = (e: DragEvent) => {
      if (e.dataTransfer?.types.includes('Files')) {
        e.preventDefault()
      }
    }
    const enter = (e: DragEvent) => {
      if (e.dataTransfer?.types.includes('Files')) {
        depth++
        setDropHover(true)
      }
    }
    const leave = () => {
      depth = Math.max(0, depth - 1)
      if (depth === 0) setDropHover(false)
    }
    const drop = async (e: DragEvent) => {
      depth = 0
      setDropHover(false)
      const file = e.dataTransfer?.files?.[0]
      if (!file) return
      e.preventDefault()
      if (!/\.(graph\.)?json(\.gz)?$/.test(file.name)) {
        setToast(`${file.name}: expected a .graph.json from \`modelmap dump\``)
        return
      }
      try {
        loadDocFromFile((await readDroppedGraph(file)) as never, file.name)
      } catch {
        setToast(`${file.name} could not be parsed as a graph document`)
      }
    }
    window.addEventListener('dragover', over)
    window.addEventListener('dragenter', enter)
    window.addEventListener('dragleave', leave)
    window.addEventListener('drop', drop)
    return () => {
      window.removeEventListener('dragover', over)
      window.removeEventListener('dragenter', enter)
      window.removeEventListener('dragleave', leave)
      window.removeEventListener('drop', drop)
    }
  }, [embed, loadDocFromFile, setToast])

  useEffect(() => {
    if (embed) document.documentElement.dataset.embed = '1'
    const boot = async () => {
      const id = urlModel()
      if (!id) return
      await loadModel(id, { push: false })
      const params = new URL(location.href).searchParams
      const sel = params.get('sel')
      // a deep link to a module opens the way to it (ancestors expanded, framed)
      if (sel != null) useStore.getState().reveal(sel)
      if (params.get('mode') === 'flow' && (useStore.getState().doc?.trace.length ?? 0) > 0) {
        const { useFlowStore } = await import('./flow/flowStore')
        useFlowStore.getState().activate()
      }
    }
    void boot()
    const onPop = () => {
      setCompare(urlCompare())
      setZoo(urlZoo())
      const id = urlModel()
      if (id && id !== useStore.getState().doc?.model_id) void loadModel(id, { push: false })
    }
    const onNav = () => {
      setCompare(urlCompare())
      setZoo(urlZoo())
    }
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
          {zoo ? (
            zoo.page === 'models' ? <CatalogView /> : <FamilyView familyKey={zoo.key} />
          ) : compare ? (
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
          {dropHover && (
            <div className="mm-drop-overlay" aria-hidden="true">
              <div className="mm-drop-card">drop a <code>.graph.json</code> to open it<br /><span className="mm-dim">from <code>modelmap dump</code> — stays in your browser</span></div>
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
