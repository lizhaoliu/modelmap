import { useEffect, useMemo, useState } from 'react'
import { gotoCompare, gotoModel } from '../api'
import { fmtBytes } from '../analytics/cost'
import { fmtParams } from '../fmt'

/** The architecture zoo (design doc §23): /models — every cached graph as a
 *  filterable table of structural facts; /arch/<family> — a curated lineage
 *  whose arrows are live structural diffs. */

export interface CatalogEntry {
  model_id: string
  architecture: string | null
  model_type: string | null
  family: string | null
  fidelity: string
  params_total: number
  active_params: number
  layers: number | null
  hidden: number | null
  heads: number | null
  kv_heads: number | null
  context: number | null
  kv_bytes_per_token: number
  macs_per_token: number
  tags: string[]
}

interface FamilyMember {
  id: string
  entry: CatalogEntry | null
}

export interface Family {
  key: string
  title: string
  blurb: string
  members: FamilyMember[]
  extra: FamilyMember[]
}

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url)
    return r.ok ? r.json() : null
  } catch {
    return null
  }
}

const fmtCtx = (n: number | null) => (n == null ? '—' : n >= 1 << 20 ? `${n / (1 << 20)}M` : n >= 1024 ? `${Math.round(n / 1024)}k` : String(n))

function Tag({ t }: { t: string }) {
  return <span className={`mm-zoo-tag ${/^(moe|mla)/.test(t) ? 'is-hot' : ''}`}>{t}</span>
}

function Row({ e }: { e: CatalogEntry }) {
  const moe = e.active_params > 0 && e.active_params < 0.9 * e.params_total
  return (
    <tr onClick={() => gotoModel(e.model_id)} tabIndex={0} onKeyDown={(ev) => ev.key === 'Enter' && gotoModel(e.model_id)}>
      <td className="mm-zoo-id">{e.model_id}</td>
      <td>{fmtParams(e.params_total)}{moe && <span className="mm-dim"> · {fmtParams(e.active_params)} act</span>}</td>
      <td>{e.layers ?? '—'}</td>
      <td>{e.hidden ?? '—'}</td>
      <td>{e.heads ?? '—'}{e.kv_heads != null && e.kv_heads !== e.heads ? `/${e.kv_heads}` : ''}</td>
      <td>{fmtCtx(e.context)}</td>
      <td>{e.kv_bytes_per_token ? `${fmtBytes(e.kv_bytes_per_token)}` : '—'}</td>
      <td className="mm-zoo-tags">{e.tags.map((t) => <Tag key={t} t={t} />)}</td>
    </tr>
  )
}

export function CatalogView() {
  const [models, setModels] = useState<CatalogEntry[] | null>(null)
  const [q, setQ] = useState('')
  const [tag, setTag] = useState<string | null>(null)
  useEffect(() => {
    document.title = 'models · modelmap'
    void fetchJson<{ models: CatalogEntry[] }>('/api/models').then((d) => setModels(d?.models ?? []))
  }, [])
  const tagKinds = useMemo(() => {
    const counts = new Map<string, number>()
    for (const m of models ?? []) for (const t of m.tags) {
      const k = t.split(' ')[0].split(':')[0]
      counts.set(k, (counts.get(k) ?? 0) + 1)
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([k]) => k)
  }, [models])
  const shown = useMemo(
    () =>
      (models ?? []).filter(
        (m) =>
          (!q || m.model_id.toLowerCase().includes(q.toLowerCase()) || (m.architecture ?? '').toLowerCase().includes(q.toLowerCase())) &&
          (!tag || m.tags.some((t) => t.startsWith(tag))),
      ),
    [models, q, tag],
  )
  return (
    <div className="mm-zoo">
      <header className="mm-zoo-head">
        <h1>every model this instance has mapped</h1>
        <p className="mm-dim">
          structural facts straight from the graphs — click a row to open its map · also at <code>/api/models</code>
        </p>
        <div className="mm-zoo-filters">
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="filter by id or class…" spellCheck={false} aria-label="Filter models" />
          {tagKinds.map((k) => (
            <button key={k} className={`mm-zoo-tagbtn ${tag === k ? 'is-on' : ''}`} onClick={() => setTag(tag === k ? null : k)}>
              {k}
            </button>
          ))}
        </div>
      </header>
      {models === null ? (
        <p className="mm-gallery-loading">loading…</p>
      ) : (
        <div className="mm-zoo-scroll">
          <table className="mm-zoo-table">
            <thead>
              <tr><th>model</th><th>params</th><th>layers</th><th>hidden</th><th>heads</th><th>ctx</th><th>kv/tok</th><th>structure</th></tr>
            </thead>
            <tbody>{shown.map((e) => <Row key={e.model_id} e={e} />)}</tbody>
          </table>
          {!shown.length && <p className="mm-gallery-loading">nothing matches</p>}
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------- family page

interface LineageDiff {
  changed: number
  added: number
  removed: number
  fields: string[]
}

function useLineageDiff(a: string | undefined, b: string | undefined): LineageDiff | null {
  const [d, setD] = useState<LineageDiff | null>(null)
  useEffect(() => {
    if (!a || !b) return
    let alive = true
    void fetchJson<{ counts: Record<string, number>; config_diff: { field: string }[]; pairs: { changes: { field: string }[] }[] }>(
      `/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`,
    ).then((r) => {
      if (!alive || !r) return
      // config keys are the semantic story (hidden_size, layers…); skip bookkeeping keys
      const top = r.config_diff
        .map((c) => c.field)
        .filter((f) => !['architectures', 'model_type', 'torch_dtype', 'dtype'].includes(f))
        .slice(0, 4)
      setD({ changed: r.counts.changed, added: r.counts.added, removed: r.counts.removed, fields: top })
    })
    return () => {
      alive = false
    }
  }, [a, b])
  return d
}

function LineageStep({ from, to }: { from: FamilyMember; to: FamilyMember }) {
  const d = useLineageDiff(from.entry ? from.id : undefined, to.entry ? to.id : undefined)
  return (
    <div className="mm-zoo-step">
      <span className="mm-zoo-arrow">→</span>
      <button className="mm-link" onClick={() => gotoCompare(from.id, to.id)} title="Open the full side-by-side diff">
        {d
          ? d.changed + d.added + d.removed === 0
            ? 'identical structure'
            : `${d.changed} changed${d.added ? ` · +${d.added}` : ''}${d.removed ? ` · −${d.removed}` : ''}${d.fields.length ? ` — ${d.fields.join(', ')}` : ''}`
          : 'diff…'}
      </button>
    </div>
  )
}

function MemberCard({ m }: { m: FamilyMember }) {
  const e = m.entry
  return (
    <button className="mm-zoo-member" onClick={() => gotoModel(m.id)}>
      <b>{m.id.split('/').pop()}</b>
      {e ? (
        <>
          <span>{fmtParams(e.params_total)}{e.active_params < 0.9 * e.params_total ? ` · ${fmtParams(e.active_params)} active` : ''}{e.layers != null ? ` · ${e.layers}L` : ''}{e.hidden != null ? ` · ${e.hidden}h` : ''}</span>
          <span className="mm-zoo-tags">{e.tags.slice(0, 4).map((t) => <Tag key={t} t={t} />)}</span>
        </>
      ) : (
        <span className="mm-dim">first visit extracts it</span>
      )}
    </button>
  )
}

export function FamilyView({ familyKey }: { familyKey: string }) {
  const [families, setFamilies] = useState<Family[] | null>(null)
  useEffect(() => {
    void fetchJson<{ families: Family[] }>('/api/families').then((d) => setFamilies(d?.families ?? []))
  }, [])
  const fam = families?.find((f) => f.key === familyKey)
  useEffect(() => {
    if (fam) document.title = `${fam.title} · modelmap`
  }, [fam])
  if (families === null) return <div className="mm-zoo"><p className="mm-gallery-loading">loading…</p></div>
  if (!fam)
    return (
      <div className="mm-zoo">
        <p className="mm-gallery-loading">no such family — try {families.map((f) => f.key).join(', ')}</p>
      </div>
    )
  return (
    <div className="mm-zoo">
      <header className="mm-zoo-head">
        <nav className="mm-zoo-nav">
          {families.map((f) => (
            <button key={f.key} className={`mm-zoo-tagbtn ${f.key === familyKey ? 'is-on' : ''}`} onClick={() => { history.pushState({}, '', `/arch/${f.key}`); window.dispatchEvent(new Event('mm:navigate')) }}>
              {f.key}
            </button>
          ))}
        </nav>
        <h1>{fam.title}</h1>
        <p className="mm-zoo-blurb">{fam.blurb}</p>
      </header>
      <div className="mm-zoo-lineage">
        {fam.members.map((m, i) => (
          <span key={m.id} className="mm-zoo-lineage-item">
            {i > 0 && <LineageStep from={fam.members[i - 1]} to={m} />}
            <MemberCard m={m} />
          </span>
        ))}
      </div>
      {fam.extra.length > 0 && (
        <div className="mm-zoo-extra">
          <span className="mm-live-lbl">related</span>
          {fam.extra.map((m) => <MemberCard key={m.id} m={m} />)}
        </div>
      )}
      <p className="mm-zoo-foot">
        every number and diff on this page is derived live from the graphs — nothing is hand-maintained ·{' '}
        <a className="mm-link" href="/models" onClick={(e) => { e.preventDefault(); history.pushState({}, '', '/models'); window.dispatchEvent(new Event('mm:navigate')) }}>all mapped models</a>
      </p>
    </div>
  )
}
