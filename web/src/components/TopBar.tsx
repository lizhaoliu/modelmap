import { useEffect, useState } from 'react'
import { getToken } from '../api'
import { useFlowStore } from '../flow/flowStore'
import { fmtParams } from '../fmt'
import { useStore } from '../store'
import { HelpOverlay } from './HelpOverlay'
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
    () => (localStorage.getItem('mm-theme') as Theme) || 'system',
  )
  const [copied, setCopied] = useState(false)
  const [tokenOpen, setTokenOpen] = useState(false)
  const [helpOpen, setHelpOpen] = useState(false)
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
    const next: Theme = theme === 'system' ? 'dark' : theme === 'dark' ? 'light' : 'system'
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
      {doc && (
        <span className="mm-model-chip" title={`revision ${doc.revision}`}>
          {doc.model_id}
        </span>
      )}
      <div className="mm-topbar-center">{doc && <ModelSearch />}</div>
      {doc && (
        <>
          <span className={`mm-fidelity is-${doc.fidelity}`} title={doc.notes.join('\n') || 'traced fake forward completed'}>
            {doc.fidelity}
          </span>
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
          <button className="mm-btn" onClick={share}>
            {copied ? 'copied ✓' : 'share'}
          </button>
        </>
      )}
      <span className="mm-topbar-rel">
        <button
          className={`mm-btn ${getToken() ? 'is-on' : ''}`}
          onClick={() => setTokenOpen((v) => !v)}
          title="Hugging Face token for gated / private repos"
          aria-expanded={tokenOpen}
        >
          token
        </button>
        {tokenOpen && <TokenPopover onClose={() => setTokenOpen(false)} />}
      </span>
      <button className="mm-btn" onClick={cycleTheme} title="Theme">
        {theme === 'system' ? 'auto' : theme}
      </button>
      <button className="mm-btn" onClick={() => setHelpOpen(true)} title="Shortcuts (?)" aria-label="Help">
        ?
      </button>
      {helpOpen && <HelpOverlay onClose={() => setHelpOpen(false)} />}
    </header>
  )
}
