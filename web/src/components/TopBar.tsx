import { useEffect, useState } from 'react'
import { getToken, gotoCompare, gotoModel } from '../api'
import { useCompareStore } from '../compare/compareStore'
import { urlCompare } from '../App'
import { useFlowStore } from '../flow/flowStore'
import { liveSupport, useLiveStore } from '../live/liveStore'
import { fmtParams } from '../fmt'
import { useStore } from '../store'
import { ExportMenu } from './ExportMenu'
import { HelpOverlay } from './HelpOverlay'
import { LensBar } from './LensBar'
import { ModelSearch } from './ModelSearch'
import { TokenPopover } from './TokenPopover'

type Theme = 'system' | 'light' | 'dark'

function applyTheme(t: Theme) {
  if (t === 'system') delete document.documentElement.dataset.theme
  else document.documentElement.dataset.theme = t
}

export function TopBar() {
  const doc = useStore((s) => s.doc)
  const flowActive = useFlowStore((s) => s.active)
  const activateFlow = useFlowStore((s) => s.activate)
  const deactivateFlow = useFlowStore((s) => s.deactivate)
  const fileDoc = useStore((s) => s.fileDoc)
  const liveOpen = useLiveStore((s) => s.open)
  const toggleLive = useLiveStore((s) => s.toggle)
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem('mm-theme') as Theme) || 'dark', // dark by default
  )
  const [copied, setCopied] = useState(false)
  const [tokenOpen, setTokenOpen] = useState(false)
  const [helpOpen, setHelpOpen] = useState(false)
  const [cmpOpen, setCmpOpen] = useState(false)
  const cmpIds = useCompareStore((s) => s.ids)
  const inCompare = urlCompare() != null && cmpIds != null
  const setToast = useStore((s) => s.setToast)

  useEffect(() => applyTheme(theme), [theme])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA') return
      if (e.key === '?') setHelpOpen((v) => !v)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const cycleTheme = () => {
    const next: Theme = theme === 'dark' ? 'light' : theme === 'light' ? 'system' : 'dark'
    localStorage.setItem('mm-theme', next)
    setTheme(next)
  }

  const share = async () => {
    try {
      await navigator.clipboard.writeText(location.href)
      setCopied(true)
      setToast('Link copied — it reproduces this exact view')
      setTimeout(() => setCopied(false), 1600)
    } catch {
      // clipboard blocked (permissions / insecure context): the URL *is* the view
      setToast('Copy the address bar — the URL reproduces this exact view')
    }
  }

  return (
    <header className="mm-topbar">
      <a
        className="mm-brand"
        href="/"
        onClick={(e) => {
          e.preventDefault()
          location.href = '/'
        }}
      >
        modelmap
      </a>
      {inCompare && cmpIds && (
        <span className="mm-model-chip">{cmpIds[0]} <i className="mm-vs">vs</i> {cmpIds[1]}</span>
      )}
      {!inCompare && doc && (
        <span className="mm-model-chip" title={`revision ${doc.revision} · fidelity ${doc.fidelity}${doc.weights_format ? ` · ${doc.weights_format}` : ''}${doc.notes.length ? '\n' + doc.notes.join('\n') : ''}`}>
          {fileDoc && <i className="mm-file-tag">file</i>}
          {doc.model_id.replace(/:[^/]+$/, '')}
        </span>
      )}
      {!inCompare && doc && (doc.variants?.length ?? 0) > 0 && (
        <label className="mm-variant" title="GGUF quantization variants in this repo — switch to see real bytes per weight">
          <span className="mm-variant-tag">gguf</span>
          <select
            value={doc.variant ?? ''}
            onChange={(e) => gotoModel(`${doc.model_id.replace(/:[^/]+$/, '')}:${e.target.value}`)}
            aria-label="GGUF variant"
          >
            {(doc.variants ?? []).map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
      )}
      <div className="mm-topbar-center">{!inCompare && doc && <ModelSearch />}</div>
      {!inCompare && doc && <LensBar />}
      {!inCompare && doc && (
        <span className="mm-topbar-rel">
          <button className={`mm-btn mm-btn-cmp ${cmpOpen ? 'is-on' : ''}`} onClick={() => setCmpOpen((v) => !v)} title="Compare this model with another">
            vs…
          </button>
          {cmpOpen && (
            <div className="mm-pop mm-cmp-pop">
              <label>Compare <b>{doc.model_id}</b> with</label>
              <ModelSearch placeholder="other model id" onPick={(id) => { setCmpOpen(false); gotoCompare(doc.model_id, id) }} />
            </div>
          )}
        </span>
      )}
      {!inCompare && doc && (
        <>
          {doc.fidelity !== 'full' && (
            <span className={`mm-fidelity is-${doc.fidelity}`} title={doc.notes.join('\n') || 'traced fake forward completed'}>
              {doc.fidelity}
            </span>
          )}
          <span className="mm-total-params">{fmtParams(doc.params_total)} params</span>
          {doc.trace.length > 0 ? (
            <button
              className={`mm-btn mm-btn-flow ${flowActive ? 'is-on' : ''} ${!flowActive && !localStorage.getItem('mm-flow-used') ? 'mm-glow' : ''}`}
              onClick={() => (flowActive ? deactivateFlow() : activateFlow())}
              title="Watch a token flow through the model (F)"
            >
              {flowActive ? 'exit flow' : '▶ flow'}
            </button>
          ) : (
            <button
              className="mm-btn"
              disabled
              title={
                (doc.notes.find((n) => n.includes('trust_remote_code')) &&
                  'No forward trace: this repo needs trust_remote_code (its own Python), which the server refuses. ' +
                    'Run `modelmap dump <id> --trust-remote-code` locally and drop the .graph.json here for the full map.') ||
                doc.notes.find((n) => n.includes('weights view') || n.includes('instantiate')) ||
                'No trace available (weights view)'
              }
            >
              ▶ flow
            </button>
          )}
          <button
            className={`mm-btn mm-btn-live ${liveOpen ? 'is-on' : ''}`}
            onClick={() => toggleLive(doc)}
            title={
              liveSupport(doc).ok
                ? 'Run this model in your browser: type a prompt, see real attention and next-token probabilities'
                : 'Live inference (this model cannot run in the browser — the panel suggests ones that can)'
            }
          >
            ⚡ live
          </button>
          <button className="mm-btn mm-btn-share" onClick={share}>
            {copied ? 'copied ✓' : 'share'}
          </button>
          <ExportMenu />
        </>
      )}
      <span className="mm-topbar-rel">
        <button
          className={`mm-btn mm-btn-token ${getToken() ? 'is-on' : ''}`}
          onClick={() => setTokenOpen((v) => !v)}
          title="Hugging Face token for gated / private repos"
          aria-expanded={tokenOpen}
        >
          token
        </button>
        {tokenOpen && <TokenPopover onClose={() => setTokenOpen(false)} />}
      </span>
      <button className="mm-btn mm-btn-theme" onClick={cycleTheme} title="Theme">
        {theme === 'system' ? 'auto' : theme}
      </button>
      <a
        className="mm-btn mm-btn-gh"
        href="https://github.com/lizhaoliu/modelmap"
        target="_blank"
        rel="noreferrer"
        title="Source on GitHub (MIT)"
        aria-label="GitHub repository"
      >
        <svg viewBox="0 0 16 16" width="15" height="15" fill="currentColor" aria-hidden="true">
          <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>
        </svg>
      </a>
      <button className="mm-btn mm-btn-help" onClick={() => setHelpOpen(true)} title="Shortcuts (?)" aria-label="Help">
        ?
      </button>
      {helpOpen && <HelpOverlay onClose={() => setHelpOpen(false)} />}
    </header>
  )
}
