import { fmtBytes, fmtDtype, fmtInt, fmtMacs, type Cost } from '../analytics/cost'
import { useCostStore } from '../analytics/costStore'
import { fmtParams, fmtPct, leafName } from '../fmt'
import { useStore } from '../store'
import { Shape } from './Shape'
import { Treemap } from './Treemap'

const CONFIG_KEYS = [
  'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
  'intermediate_size', 'vocab_size', 'max_position_embeddings', 'num_experts',
  'num_experts_per_tok', 'model_type',
]

/** Cost rows for a node (or the whole model when `cost` is the root). */
function CostRows({ cost, rootCost, T, B, isRoot, kvLayers, weights }: { cost: Cost; rootCost: Cost; T: number; B: number; isRoot?: boolean; kvLayers?: number; weights?: string }) {
  const tokens = Math.max(1, T * B)
  const share = (a: number, b: number) => (b > 0 ? ` · ${fmtPct(a, b)}` : '')
  return (
    <>
      <h3>cost <span className="mm-dim">(estimates · T {fmtInt(T)} · B {B})</span></h3>
      <dl className="mm-kv mm-cost">
        <dt title={cost.formula ?? 'sum over children: weight matmuls + attention core'}>compute</dt>
        <dd title={cost.formula}>
          {fmtMacs(cost.macs / tokens)}/tok <span className="mm-dim">· {fmtMacs(cost.macs)} / forward{share(cost.macs, rootCost.macs)}</span>
        </dd>
        {isRoot && cost.activeParams > 0 && (
          <>
            <dt title="parameters that run for one token; MoE counts k/E of the experts">active params</dt>
            <dd>{fmtParams(cost.activeParams)}/tok</dd>
          </>
        )}
        <dt title="parameters × bytes at the stored dtype (or the chosen serving precision)">weights</dt>
        <dd>{fmtBytes(cost.paramBytes)}{weights ? <span className="mm-dim"> · {weights}</span> : null}{share(cost.paramBytes, rootCost.paramBytes)}</dd>
        <dt title="output activation bytes at T, B, dtype (summed over the subtree)">activations</dt>
        <dd>
          {fmtBytes(cost.actBytes)}
          {isRoot && cost.maxAct > 0 && <span className="mm-dim"> · largest {fmtBytes(cost.maxAct)} ({leafName(cost.maxActNode)})</span>}
        </dd>
        {cost.kvPerToken > 0 && (
          <>
            <dt title="KV cache: layers × 2 × kv_heads × head_dim × bytes (MLA: kv_lora_rank + rope dim)">kv cache</dt>
            <dd>
              {fmtBytes(cost.kvPerToken)}/tok <span className="mm-dim">· {fmtBytes(cost.kvPerToken * tokens)} at T{isRoot && kvLayers ? ` · ${kvLayers} layers` : ''}</span>
            </dd>
          </>
        )}
      </dl>
    </>
  )
}

export function Inspector() {
  const doc = useStore((s) => s.doc)
  const index = useStore((s) => s.index)
  const selected = useStore((s) => s.selected)
  const report = useCostStore((s) => s.report)
  const lens = useCostStore((s) => s.lens)
  if (!doc || !index) return null

  const node = selected != null ? index.byId.get(selected) : undefined
  const showCost = lens !== 'none' && report
  const setToast = useStore((s) => s.setToast)
  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(location.href)
      setToast(node ? `Link to ${leafName(node.id)} copied` : 'Link copied')
    } catch {
      setToast('Copy the address bar — the URL points at this module')
    }
  }

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
          {doc.weights_format && (
            <>
              <dt>checkpoint</dt>
              <dd>
                {doc.weights_format}
                {doc.variant && <> · <b>{doc.variant}</b></>}
                {(doc.variants?.length ?? 0) > 1 && <span className="mm-dim"> · {doc.variants!.length} variants in the repo</span>}
              </dd>
            </>
          )}
          <dt>trace steps</dt>
          <dd>{doc.trace.length || '—'}</dd>
          {cfg.map(([k, v]) => (
            <FragmentRow key={k} k={k} v={String(v)} />
          ))}
        </dl>
        <Treemap parent="" overview />
        {showCost && <CostRows cost={report.root} rootCost={report.root} T={report.assumptions.T} B={report.assumptions.B} isRoot kvLayers={report.kvLayers} weights={report.assumptions.weightBytes != null ? report.assumptions.weights : undefined} />}
        {showCost && report.notes.length > 0 && <p className="mm-hint">{report.notes.join(' · ')}</p>}
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
  const attrs = Object.entries(node.attrs ?? {}).filter(([k]) => !k.startsWith('_'))
  const src = node.attrs?._src
  const srcUrl = node.attrs?._src_url

  return (
    <aside className="mm-inspector">
      <div className="mm-insp-head">
        <h2>{leafName(node.id)}</h2>
        <span className={`mm-kind-chip kind-${node.kind}`}>{node.kind}</span>
        <button className="mm-link mm-insp-copy" onClick={copyLink} title="Copy a link that opens this model with this module selected">
          copy link
        </button>
      </div>
      <p className="mm-path">{node.id || '(root)'}</p>
      <dl className="mm-kv">
        <dt>class</dt>
        <dd>
          {node.cls === '?' ? 'unknown (weights view)' : node.cls}
          {src && (
            <span className="mm-src">
              {' '}
              {srcUrl ? (
                <a href={srcUrl} target="_blank" rel="noreferrer" title="open the class definition">
                  {src}
                </a>
              ) : (
                src
              )}
            </span>
          )}
        </dd>
        {attrs.map(([k, v]) => (
          <FragmentRow key={k} k={k} v={String(v)} />
        ))}
        <dt>params</dt>
        <dd>
          {node.params.toLocaleString()} <span className="mm-dim">· {fmtPct(node.params, doc.params_total)} of model</span>
        </dd>
        {node.dtype && (
          <>
            <dt>dtype</dt>
            <dd>{fmtDtype(node.dtype)}</dd>
          </>
        )}
        {repeat && (
          <>
            <dt>repeats</dt>
            <dd>×{repeat.count} identical <span className="mm-dim">({fmtMembers(repeat.members)})</span></dd>
          </>
        )}
        {io && (
          <>
            <dt>input</dt>
            <dd>
              {io.inputs.length
                ? io.inputs.map((s, i) => (
                    <span key={i}>
                      {i > 0 && '  '}
                      <Shape shape={s} labels={index.dimLabels} batch={index.traceBatch} />
                    </span>
                  ))
                : '—'}
            </dd>
            <dt>output</dt>
            <dd>
              {io.outputs.length
                ? io.outputs.map((s, i) => (
                    <span key={i}>
                      {i > 0 && '  '}
                      <Shape shape={s} labels={index.dimLabels} batch={index.traceBatch} />
                    </span>
                  ))
                : '—'}
            </dd>
          </>
        )}
      </dl>
      <div className="mm-param-bar" title="share of total parameters">
        <div
          className={`mm-param-fill kind-${node.kind}`}
          style={{ width: `${Math.max(0.5, (node.params / doc.params_total) * 100)}%` }}
        />
      </div>
      <Treemap parent={node.id} />
      {showCost && report.byNode.get(node.id) && (
        <CostRows cost={report.byNode.get(node.id)!} rootCost={report.root} T={report.assumptions.T} B={report.assumptions.B} weights={report.assumptions.weightBytes != null ? report.assumptions.weights : undefined} />
      )}
      {weights.length > 0 && (
        <>
          <h3>weights</h3>
          <table className="mm-wtable">
            <tbody>
              {weights.map(([name, shape]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>
                    <Shape shape={shape} labels={index.dimLabels} />
                  </td>
                  <td className="mm-dim">{fmtParams(shape.reduce((a, b) => a * b, 1))}{node.dtype ? ` · ${fmtDtype(node.dtype).split(' · ')[0]}` : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </aside>
  )
}

/** "0…35" for a contiguous run; "0, 1, 3, 5 … 41 (interleaved)" otherwise */
export function fmtMembers(members: string[]): string {
  if (!members.length) return ''
  const nums = members.map(Number)
  const contiguous = nums.every((n, i) => !Number.isNaN(n) && (i === 0 || n === nums[i - 1] + 1))
  if (contiguous || members.length <= 4) return members.length <= 4 ? members.join(', ') : `members ${members[0]}…${members[members.length - 1]}`
  return `members ${members.slice(0, 4).join(', ')} … ${members[members.length - 1]} — interleaved`
}

function FragmentRow({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </>
  )
}
