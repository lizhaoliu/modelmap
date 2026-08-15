import { useStore } from '../store'

/** Clickable path of the selected module — the "you are here" trail
 *  (design doc §07 wireframe). Each segment selects that ancestor. */
export function Breadcrumb() {
  const doc = useStore((s) => s.doc)
  const selected = useStore((s) => s.selected)
  const select = useStore((s) => s.select)
  if (!doc || !selected) return null

  const parts = selected.split('.')
  const crumbs = parts.map((label, i) => ({ label, id: parts.slice(0, i + 1).join('.') }))

  return (
    <nav className="mm-crumbs" aria-label="Selected module path">
      {crumbs.map((c, i) => (
        <span className="mm-crumb-seg" key={c.id}>
          {i > 0 && <span className="mm-crumb-sep">›</span>}
          <button
            className={`mm-crumb ${i === crumbs.length - 1 ? 'is-current' : ''}`}
            onClick={() => select(c.id)}
          >
            {c.label}
          </button>
        </span>
      ))}
    </nav>
  )
}
