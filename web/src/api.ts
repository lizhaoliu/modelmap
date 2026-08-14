import type { GraphDoc } from './types'

export async function fetchGraph(modelId: string, revision = 'main'): Promise<GraphDoc> {
  const res = await fetch(
    `/api/graph/${modelId}?revision=${encodeURIComponent(revision)}`,
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
