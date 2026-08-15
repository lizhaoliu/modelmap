export function fmtParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e11 ? 0 : 2)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return `${n}`
}

export function fmtPct(part: number, total: number): string {
  if (!total) return '—'
  const p = (part / total) * 100
  if (p >= 10) return `${p.toFixed(0)}%`
  if (p >= 0.1) return `${p.toFixed(1)}%`
  return p > 0 ? '<0.1%' : '0%'
}

export function fmtShape(shape: number[]): string {
  return `[${shape.join(' × ')}]`
}

export function fmtCount(n: number | null): string {
  if (n == null) return ''
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`
  return `${n}`
}

/** Last path segment: "model.layers.0.self_attn" → "self_attn". */
export function leafName(id: string): string {
  const seg = id.split('.').pop() ?? id
  return seg === '' ? 'model' : seg
}

/** Explains the recurring constants in traced activation shapes (batch, seq)
 *  from the dummy input the trace actually ran — true for any architecture. */
export function traceLegend(trace: { inputs: number[][] }[]): string | null {
  const first = trace[0]?.inputs?.[0]
  if (!first || first.length < 2) return null
  const [b, ...rest] = first
  if (first.length === 2) return `traced with batch ${b} · ${rest[0]} input tokens`
  if (first.length === 4) return `traced with batch ${b} · ${rest.join(' × ')} image`
  return `traced with input [${first.join(' × ')}]`
}
