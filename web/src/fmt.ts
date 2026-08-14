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
