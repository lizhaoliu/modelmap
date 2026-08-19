import { useEffect, useMemo, useState } from 'react'
import { ReactFlowProvider } from '@xyflow/react'
import { computeCosts, DEFAULT_ASSUMPTIONS, fmtBytes, fmtMacs } from '../analytics/cost'
import { useCompareStore } from '../compare/compareStore'
import type { Pair } from '../compare/align'
import { fmtParams, leafName } from '../fmt'
import { StoreContext, type GraphStore } from '../store'
import type { GraphDoc, GraphIndex } from '../types'
import { Canvas, LinkGroup } from './Canvas'
import { Sheet } from './Sheet'
import { Shape } from './Shape'

const CFG_ROWS: [string, string[]][] = [
  ['layers', ['num_hidden_layers', 'n_layer']], ['hidden', ['hidden_size', 'n_embd']],
  ['heads', ['num_attention_heads', 'n_head']], ['kv heads', ['num_key_value_heads']],
  ['ffn', ['intermediate_size', 'n_inner', 'moe_intermediate_size']], ['vocab', ['vocab_size']],
  ['context', ['max_position_embeddings', 'n_positions']], ['experts', ['num_experts', 'n_routed_experts', 'num_local_experts']],
  ['top-k', ['num_experts_per_tok']],
]

function cfg(doc: GraphDoc, keys: string[]): string {
  for (const k of keys) {
    const v = doc.config[k]
    if (typeof v === 'number') return v.toLocaleString('en-US')
    if (typeof v === 'string') return v
  }
  return '—'
}

function Summary({ a, b }: { a: { doc: GraphDoc; index: GraphIndex }; b: { doc: GraphDoc; index: GraphIndex } }) {
  const al = useCompareStore((s) => s.alignment)
  const diffOnly = useCompareStore((s) => s.diffOnly)
  const setDiffOnly = useCompareStore((s) => s.setDiffOnly)
  const costs = useMemo(
    () => [computeCosts(a.doc, a.index, DEFAULT_ASSUMPTIONS), computeCosts(b.doc, b.index, DEFAULT_ASSUMPTIONS)],
    [a, b],
  )
  const rows = (
    [
      ['params', fmtParams(a.doc.params_total), fmtParams(b.doc.params_total)],
      ...CFG_ROWS.map(([label, keys]) => [label, cfg(a.doc, keys), cfg(b.doc, keys)]),
      ['compute/tok', fmtMacs(costs[0].root.macs / DEFAULT_ASSUMPTIONS.T), fmtMacs(costs[1].root.macs / DEFAULT_ASSUMPTIONS.T)],
      ['kv/tok', costs[0].root.kvPerToken ? fmtBytes(costs[0].root.kvPerToken) : '—', costs[1].root.kvPerToken ? fmtBytes(costs[1].root.kvPerToken) : '—'],
    ] as [string, string, string][]
  ).filter(([, x, y]) => !(x === '—' && y === '—'))
  return (
    <div className="mm-cmp-summary">
      <div className="mm-cmp-heads">
        <div className="mm-cmp-head"><b>{a.doc.model_id}</b><span>{a.doc.architecture}</span></div>
        <div className="mm-cmp-mid">
          {al && (
            <span className="mm-cmp-counts" title="module pairs: changed · added · removed">
              <em>~{al.counts.changed}</em> <i className="add">+{al.counts.added}</i> <i className="rem">−{al.counts.removed}</i>
            </span>
          )}
          <label className="mm-cmp-toggle"><input type="checkbox" checked={diffOnly} onChange={(e) => setDiffOnly(e.target.checked)} /> differences only</label>
        </div>
        <div className="mm-cmp-head is-b"><b>{b.doc.model_id}</b><span>{b.doc.architecture}</span></div>
      </div>
      <div className="mm-cmp-rows">
        {rows.map(([label, x, y]) => (
          <div key={label} className={`mm-cmp-row ${x !== y ? 'is-diff' : ''}`}>
            <span className="mm-cmp-val">{x}</span>
            <span className="mm-cmp-label">{label}</span>
            <span className="mm-cmp-val is-b">{y}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function DiffInspector({ pair, a, b }: { pair: Pair | null; a: GraphIndex; b: GraphIndex }) {
  const al = useCompareStore((s) => s.alignment)
  if (!pair) {
    return (
      <aside className="mm-inspector mm-diffinsp">
        <h2>Config differences</h2>
        {al && al.configDiff.length === 0 && <p className="mm-hint">Identical on every compared key.</p>}
        {al && al.configDiff.length > 0 && (
          <table className="mm-difftable">
            <tbody>
              {al.configDiff.map((c) => (
                <tr key={c.field} className="is-diff"><td>{c.field}</td><td>{c.a ?? '—'}</td><td>{c.b ?? '—'}</td></tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="mm-hint">Click any module: paired modules show both sides, differences in amber; <kbd>+</kbd>/<kbd>−</kbd> mark modules only one model has.</p>
      </aside>
    )
  }
  const A = pair.a, B = pair.b
  const changed = new Set(pair.changes.map((c) => c.field))
  const cell = (side: 'a' | 'b', field: string, val: React.ReactNode) => (
    <td className={changed.has(field) ? 'is-diff' : ''}>{val ?? '—'}</td>
  )
  const w = (n: typeof A) => Object.entries(n?.weight_shapes ?? {})
  const attrs = (n: typeof A) => Object.entries(n?.attrs ?? {}).filter(([k]) => !k.startsWith('_'))
  const io = (n: typeof A, idx: GraphIndex) => (n ? idx.traceByNode.get(n.id) : undefined)
  const ioA = io(A, a), ioB = io(B, b)
  const keys = new Set([...w(A).map(([k]) => `weight ${k}`), ...w(B).map(([k]) => `weight ${k}`)])
  const akeys = new Set([...attrs(A).map(([k]) => k), ...attrs(B).map(([k]) => k)])
  return (
    <aside className="mm-inspector mm-diffinsp">
      <h2>
        {A ? leafName(A.id) : leafName(B!.id)}
        <span className={`mm-diffstatus is-${pair.status}`}>{pair.status}</span>
      </h2>
      <p className="mm-path">{A?.id ?? '—'}{B && A?.id !== B.id ? ` ↔ ${B.id}` : ''}</p>
      <table className="mm-difftable">
        <tbody>
          <tr>{cell('a', 'kind', A?.kind)}<th>kind</th>{cell('b', 'kind', B?.kind)}</tr>
          <tr>{cell('a', 'class', A?.cls)}<th>class</th>{cell('b', 'class', B?.cls)}</tr>
          <tr>{cell('a', 'params', A && fmtParams(A.params))}<th>params</th>{cell('b', 'params', B && fmtParams(B.params))}</tr>
          <tr>{cell('a', 'dtype', A?.dtype)}<th>dtype</th>{cell('b', 'dtype', B?.dtype)}</tr>
          <tr>{cell('a', 'repeats', A && a.repeatByRep.get(A.id) ? `×${a.repeatByRep.get(A.id)!.count}` : null)}<th>repeats</th>{cell('b', 'repeats', B && b.repeatByRep.get(B.id) ? `×${b.repeatByRep.get(B.id)!.count}` : null)}</tr>
          {[...keys].map((k) => {
            const name = k.slice(7)
            const sa = A?.weight_shapes?.[name], sb = B?.weight_shapes?.[name]
            return <tr key={k}>{cell('a', k, sa && <Shape shape={sa} labels={a.dimLabels} />)}<th>{name}</th>{cell('b', k, sb && <Shape shape={sb} labels={b.dimLabels} />)}</tr>
          })}
          {[...akeys].map((k) => <tr key={k}>{cell('a', k, A?.attrs?.[k])}<th>{k}</th>{cell('b', k, B?.attrs?.[k])}</tr>)}
          {(ioA || ioB) && (
            <>
              <tr>{cell('a', 'input', ioA?.inputs[0] && <Shape shape={ioA.inputs[0]} labels={a.dimLabels} batch={a.traceBatch} />)}<th>input</th>{cell('b', 'input', ioB?.inputs[0] && <Shape shape={ioB.inputs[0]} labels={b.dimLabels} batch={b.traceBatch} />)}</tr>
              <tr>{cell('a', 'output', ioA?.outputs[0] && <Shape shape={ioA.outputs[0]} labels={a.dimLabels} batch={a.traceBatch} />)}<th>output</th>{cell('b', 'output', ioB?.outputs[0] && <Shape shape={ioB.outputs[0]} labels={b.dimLabels} batch={b.traceBatch} />)}</tr>
            </>
          )}
        </tbody>
      </table>
    </aside>
  )
}

function Side({ store, id, group }: { store: GraphStore; id: string; group: LinkGroup }) {
  return (
    <StoreContext.Provider value={store}>
      <ReactFlowProvider>
        <Canvas link={{ group, id }} flowEnabled={false} costs={false} />
      </ReactFlowProvider>
    </StoreContext.Provider>
  )
}

export function CompareView({ idA, idB }: { idA: string; idB: string }) {
  const load = useCompareStore((s) => s.load)
  const a = useCompareStore((s) => s.a)
  const b = useCompareStore((s) => s.b)
  const loading = useCompareStore((s) => s.loading)
  const error = useCompareStore((s) => s.error)
  const alignment = useCompareStore((s) => s.alignment)
  const [group] = useState(() => new LinkGroup())
  useEffect(() => { void load(idA, idB) }, [idA, idB, load])

  const docA = a((s) => s.doc), indexA = a((s) => s.index)
  const docB = b((s) => s.doc), indexB = b((s) => s.index)
  const selA = a((s) => s.selected), selB = b((s) => s.selected)
  const pair = alignment
    ? (selA != null ? alignment.byA.get(selA) : selB != null ? alignment.byB.get(selB) : null) ?? null
    : null

  if (error) return <div className="mm-error" role="alert"><strong>Could not compare.</strong><p>{error}</p></div>
  if (loading || !docA || !indexA || !docB || !indexB || !alignment) {
    return <div className="mm-overlay"><div className="mm-loader"><span className="mm-pulse-dot" /><div><div className="mm-loader-model">{idA} vs {idB}</div><div className="mm-loader-stage">loading both graphs and aligning…</div></div></div></div>
  }
  return (
    <div className="mm-compare">
      <Summary a={{ doc: docA, index: indexA }} b={{ doc: docB, index: indexB }} />
      <div className="mm-cmp-body">
        <div className="mm-cmp-canvases">
          <Side store={a} id="a" group={group} />
          <div className="mm-cmp-divider" />
          <Side store={b} id="b" group={group} />
        </div>
        <StoreContext.Provider value={a}>
          <Sheet title={pair ? (pair.a ? pair.a.id.split('.').pop() : pair.b?.id.split('.').pop()) : 'differences'}>
            <DiffInspector pair={pair} a={indexA} b={indexB} />
          </Sheet>
        </StoreContext.Provider>
      </div>
    </div>
  )
}
