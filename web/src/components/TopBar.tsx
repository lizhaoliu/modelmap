import { useEffect, useState } from 'react'
import { fmtParams } from '../fmt'
import { useStore } from '../store'
import { ModelSearch } from './ModelSearch'

type Theme = 'system' | 'light' | 'dark'

function applyTheme(t: Theme) {
  if (t === 'system') delete document.documentElement.dataset.theme
  else document.documentElement.dataset.theme = t
}

export function TopBar() {
  const doc = useStore((s) => s.doc)
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem('mm-theme') as Theme) || 'system',
  )
  const [copied, setCopied] = useState(false)

  useEffect(() => applyTheme(theme), [theme])

  const cycleTheme = () => {
    const next: Theme = theme === 'system' ? 'dark' : theme === 'dark' ? 'light' : 'system'
    localStorage.setItem('mm-theme', next)
    setTheme(next)
  }

  const share = async () => {
    try {
      await navigator.clipboard.writeText(location.href)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch { /* clipboard unavailable */ }
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
          <button className="mm-btn" onClick={share}>
            {copied ? 'copied ✓' : 'share'}
          </button>
        </>
      )}
      <button className="mm-btn" onClick={cycleTheme} title="Theme">
        {theme === 'system' ? 'auto' : theme}
      </button>
    </header>
  )
}
