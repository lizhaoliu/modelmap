import { fmtParams, fmtPct, fmtShape, leafName, traceLegend } from '../fmt'
import { useStore } from '../store'

const CONFIG_KEYS = [
  'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
  'intermediate_size', 'vocab_size', 'max_position_embeddings', 'num_experts',
  'num_experts_per_tok', 'model_type',
]

export function Inspector() {
  const doc = useStore((s) => s.doc)
  const index = useStore((s) => s.index)
  const selected = useStore((s) => s.selected)
  if (!doc || !index) return null

  const node = selected != null ? index.byId.get(selected) : undefined

  if (!node) {
    const cfg = CONFIG_KEYS.map((k) => [k, doc.config[k]] as const).filter(
      ([, v]) => v != null && (typeof v === 'number' || typeof v === 'string'),
    )
    return (
      <aside className="mm-inspector">
        <div className="mm-insp-head">
          <h2>{doc.model_id}</h2>
          <span className="mm-cls">{doc.architecture ?? 'unknown architecture'}</span>
        </div>
        <dl className="mm-kv">
          <dt>parameters</dt>
          <dd>{doc.params_total.toLocaleString()}</dd>
          <dt>fidelity</dt>
          <dd>{doc.fidelity}</dd>
          <dt>trace steps</dt>
          <dd>{doc.trace.length || '—'}</dd>
          {cfg.map(([k, v]) => (
            <FragmentRow key={k} k={k} v={String(v)} />
          ))}
        </dl>
        {doc.notes.length > 0 && (
          <div className="mm-notes">
            {doc.notes.map((n, i) => (
              <p key={i}>{n}</p>
            ))}
          </div>
        )}
        <p className="mm-hint">
          Click a node to inspect it. Double-click or <kbd>E</kbd> expands, <kbd>C</kbd> collapses,{' '}
          <kbd>0</kbd> fits the view.
        </p>
      </aside>
    )
  }

  const repeat = index.repeatByRep.get(node.id)
  const io = index.traceByNode.get(node.id)
  const weights = node.weight_shapes ? Object.entries(node.weight_shapes) : []

  return (
    <aside className="mm-inspector">
      <div className="mm-insp-head">
        <h2>{leafName(node.id)}</h2>
        <span className={`mm-kind-chip kind-${node.kind}`}>{node.kind}</span>
      </div>
      <p className="mm-path">{node.id || '(root)'}</p>
      <dl className="mm-kv">
        <dt>class</dt>
        <dd>{node.cls === '?' ? 'unknown (weights view)' : node.cls}</dd>
        <dt>params</dt>
        <dd>
          {node.params.toLocaleString()} <span className="mm-dim">· {fmtPct(node.params, doc.params_total)} of model</span>
        </dd>
        {node.dtype && (
          <>
            <dt>dtype</dt>
            <dd>{node.dtype}</dd>
          </>
        )}
        {repeat && (
          <>
            <dt>repeats</dt>
            <dd>×{repeat.count} identical (members {repeat.members[0]}…{repeat.members[repeat.members.length - 1]})</dd>
          </>
        )}
        {io && (
          <>
            <dt>input</dt>
            <dd>{io.inputs.map(fmtShape).join('  ') || '—'}</dd>
            <dt>output</dt>
            <dd>{io.outputs.map(fmtShape).join('  ') || '—'}</dd>
          </>
        )}
      </dl>
      {io && traceLegend(doc.trace) && <p className="mm-io-note">{traceLegend(doc.trace)}</p>}
      <div className="mm-param-bar" title="share of total parameters">
        <div
          className={`mm-param-fill kind-${node.kind}`}
          style={{ width: `${Math.max(0.5, (node.params / doc.params_total) * 100)}%` }}
        />
      </div>
      {weights.length > 0 && (
        <>
          <h3>weights</h3>
          <table className="mm-wtable">
            <tbody>
              {weights.map(([name, shape]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{fmtShape(shape)}</td>
                  <td className="mm-dim">{fmtParams(shape.reduce((a, b) => a * b, 1))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </aside>
  )
}

function FragmentRow({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </>
  )
}
