import { useEffect, useState } from 'react'
import { getToken, gotoCompare, gotoModel } from '../api'
import { useCompareStore } from '../compare/compareStore'
import { urlCompare } from '../App'
import { useFlowStore } from '../flow/flowStore'
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
              className={`mm-btn mm-btn-flow ${flowActive ? 'is-on' : ''}`}
              onClick={() => (flowActive ? deactivateFlow() : activateFlow())}
              title="Replay the forward pass (F)"
            >
              {flowActive ? 'exit flow' : '▶ flow'}
            </button>
          ) : (
            <button className="mm-btn" disabled title="No trace available (weights view)">
              ▶ flow
            </button>
          )}
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
      <button className="mm-btn mm-btn-help" onClick={() => setHelpOpen(true)} title="Shortcuts (?)" aria-label="Help">
        ?
      </button>
      {helpOpen && <HelpOverlay onClose={() => setHelpOpen(false)} />}
    </header>
  )
}
