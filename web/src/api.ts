import type { GraphDoc } from './types'

export const TOKEN_KEY = 'mm-hf-token'
export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? ''
}
export function setToken(t: string) {
  if (t.trim()) localStorage.setItem(TOKEN_KEY, t.trim())
  else localStorage.removeItem(TOKEN_KEY)
}

// mirrors ids.normalize_model_id on the server: full Hub URLs, ollama-style
// "hf.co/owner/name:Q4", typographic dashes and file-tree links all collapse
// to canonical "owner/name[:variant]" before they reach the URL bar or the API
const DASHES = /[\u2010-\u2015\u2212]/g
const ZERO_WIDTH = /[\u200b-\u200d\u2060\ufeff]/g
const URL_PREFIX = /^(?:https?:\/\/)?(?:www\.)?(?:huggingface\.co|hf\.co)\//i
const URL_TAIL = /\/(?:tree|blob|resolve|raw|commit|discussions|blame)(?:\/.*)?$/
const QUANTISH = /(?:^|[-_.])(i?q\d|f16|bf16|f32|fp\d|mxfp\d|nvfp\d|int\d)/i

export function normalizeModelId(raw: string): string {
  let s = raw.trim().replace(DASHES, '-').replace(ZERO_WIDTH, '')
  if (s.startsWith('local:')) return s
  s = s.replace(URL_PREFIX, '')
  s = s.split('?')[0].split('#')[0].replace(URL_TAIL, '').replace(/^\/+|\/+$/g, '')
  const parts = s.split('/')
  if (parts.length > 2) {
    const tail = parts[parts.length - 1]
    const i = tail.lastIndexOf('.')
    const base = i > 0 ? tail.slice(0, i) : ''
    const ext = i >= 0 ? tail.slice(i + 1) : ''
    s = parts.slice(0, 2).join('/')
    if (ext.toLowerCase() === 'gguf' && base) s += ':' + base
    else if (!tail.includes(':') && QUANTISH.test(tail)) s += ':' + tail
    else if (tail.includes(':')) s += ':' + tail.slice(tail.lastIndexOf(':') + 1)
  }
  return s
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
  blurb?: string
  pipeline_tag?: string | null
  downloads?: number | null
  likes?: number | null
  architecture?: string | null
  cached: boolean
  summary: {
    architecture: string | null
    params_total: number
    fidelity: 'full' | 'structural' | 'weights'
    nodes: number
    trace_steps: number
  } | null
}

export interface Gallery {
  trending: GalleryEntry[]
  classics: GalleryEntry[]
}
export async function fetchGallery(): Promise<Gallery> {
  const res = await fetch('/api/gallery')
  if (!res.ok) return { trending: [], classics: [] }
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

export interface Health { ok: boolean; version: string; allow_local?: boolean; cache_entries?: number }
export async function fetchHealth(): Promise<Health | null> {
  try {
    const res = await fetch('/api/health')
    return res.ok ? res.json() : null
  } catch {
    return null
  }
}

/** GET json, null on any failure (network, non-2xx). */
export async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url)
    return r.ok ? ((await r.json()) as T) : null
  } catch {
    return null
  }
}
