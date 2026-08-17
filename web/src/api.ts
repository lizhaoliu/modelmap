import type { GraphDoc } from './types'

export const TOKEN_KEY = 'mm-hf-token'
export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? ''
}
export function setToken(t: string) {
  if (t.trim()) localStorage.setItem(TOKEN_KEY, t.trim())
  else localStorage.removeItem(TOKEN_KEY)
}

export async function fetchGraph(modelId: string, revision = 'main'): Promise<GraphDoc> {
  const headers: Record<string, string> = {}
  const tok = getToken()
  if (tok) headers['X-HF-Token'] = tok // gated/private repos; bypasses the shared cache
  const res = await fetch(
    `/api/graph/${modelId}?revision=${encodeURIComponent(revision)}`,
    { headers },
  )
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body.detail) detail = body.detail
    } catch { /* non-JSON error body */ }
    throw new Error(detail)
  }
  return res.json()
}

export interface SearchHit {
  id: string
  downloads: number | null
  likes: number | null
  pipeline_tag: string | null
}

export async function searchModels(q: string, limit = 8): Promise<SearchHit[]> {
  const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=${limit}`)
  if (!res.ok) return []
  return res.json()
}

export interface GalleryEntry {
  id: string
  blurb: string
  cached: boolean
  summary: {
    architecture: string | null
    params_total: number
    fidelity: 'full' | 'structural' | 'weights'
    nodes: number
    trace_steps: number
  } | null
}

export async function fetchGallery(): Promise<GalleryEntry[]> {
  const res = await fetch('/api/gallery')
  if (!res.ok) return []
  return res.json()
}

/** Client-side navigation to /compare/{A}...{B} (no reload). */
export function gotoCompare(a: string, b: string) {
  history.pushState({}, '', `/compare/${a}...${b}`)
  window.dispatchEvent(new Event('mm:navigate'))
}
export function gotoModel(id: string) {
  history.pushState({}, '', `/m/${id}`)
  window.dispatchEvent(new Event('mm:navigate'))
  window.dispatchEvent(new PopStateEvent('popstate'))
}
