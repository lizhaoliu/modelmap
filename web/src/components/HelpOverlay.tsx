const ROWS: [string, string][] = [
  ['⌘K / Ctrl-K', 'search models (or type a full id)'],
  ['click / double-click', 'inspect a module / open it'],
  ['E · C', 'expand · collapse the selected module'],
  ['0', 'fit the whole graph in view'],
  ['F', 'toggle Flow mode (replay the forward pass)'],
  ['Space', 'play / pause the replay'],
  ['← →', 'step one beat back / forward'],
  ['Esc', 'exit Flow mode · clear the selection'],
  ['?', 'this help'],
]

export function HelpOverlay({ onClose }: { onClose: () => void }) {
  return (
    <div className="mm-help-backdrop" onClick={onClose} role="presentation">
      <div className="mm-help" role="dialog" aria-label="Keyboard shortcuts" onClick={(e) => e.stopPropagation()}>
        <h2>Shortcuts</h2>
        <dl>
          {ROWS.map(([k, v]) => (
            <div key={k}>
              <dt><kbd>{k}</kbd></dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
        <p className="mm-help-foot">
          Colors: <span className="mm-swatch kind-embedding" /> embedding / head ·{' '}
          <span className="mm-swatch kind-attention" /> attention ·{' '}
          <span className="mm-swatch kind-mlp" /> MLP / experts ·{' '}
          <span className="mm-swatch kind-norm" /> norm. Amber is always live data or your selection.
        </p>
        <button className="mm-btn" onClick={onClose}>close</button>
      </div>
    </div>
  )
}
