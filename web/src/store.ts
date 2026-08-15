import { create } from 'zustand'
import { fetchGraph } from './api'
import { buildIndex, type GraphDoc, type GraphIndex } from './types'

interface State {
  doc: GraphDoc | null
  index: GraphIndex | null
  loading: string | null // model id being loaded
  error: string | null
  errorModel: string | null // id whose load failed, for the retry button
  expanded: Set<string>
  selected: string | null
  loadModel: (id: string, opts?: { push?: boolean }) => Promise<void>
  toggleExpand: (id: string) => void
  expand: (id: string) => void
  expandMany: (ids: string[]) => void
  collapse: (id: string) => void
  select: (id: string | null) => void
}

/** Containers open by default: the root and its immediate container children.
 *  Everything deeper starts closed (design doc §07: first view ≈ 10 nodes). */
function defaultExpanded(index: GraphIndex): Set<string> {
  const out = new Set<string>()
  for (const [id, kids] of index.children.entries()) {
    if (id === null || !kids.length) continue
    const node = index.byId.get(id)
    if (node && node.depth <= 1) out.add(id)
  }
  out.add('')
  return out
}

export const useStore = create<State>((set, get) => ({
  doc: null,
  index: null,
  loading: null,
  error: null,
  errorModel: null,
  expanded: new Set(),
  selected: null,

  async loadModel(id, opts) {
    if (get().loading === id) return
    set({ loading: id, error: null, errorModel: null })
    try {
      const doc = await fetchGraph(id)
      const index = buildIndex(doc)
      set({
        doc,
        index,
        loading: null,
        error: null,
        errorModel: null,
        expanded: defaultExpanded(index),
        selected: null,
      })
      if (opts?.push !== false) {
        history.pushState({}, '', `/m/${id}`)
      }
      document.title = `${id.split('/').pop()} · modelmap`
    } catch (e) {
      set({ loading: null, error: e instanceof Error ? e.message : String(e), errorModel: id })
    }
  },

  toggleExpand(id) {
    const expanded = new Set(get().expanded)
    if (expanded.has(id)) expanded.delete(id)
    else expanded.add(id)
    set({ expanded })
  },

  expand(id) {
    get().expandMany([id])
  },

  expandMany(ids) {
    const { index } = get()
    const expanded = new Set(get().expanded)
    for (const id of ids) {
      if (index?.children.get(id)?.length) expanded.add(id)
    }
    set({ expanded })
  },

  collapse(id) {
    const { index, expanded } = get()
    const next = new Set(expanded)
    if (next.has(id)) {
      next.delete(id)
    } else {
      // collapsing a leaf collapses its nearest expanded ancestor container
      const parent = index?.byId.get(id)?.parent
      if (parent == null || parent === '') return
      next.delete(parent)
    }
    set({ expanded: next })
  },

  select(id) {
    set({ selected: id })
    const url = new URL(location.href)
    if (id) url.searchParams.set('sel', id)
    else url.searchParams.delete('sel')
    history.replaceState({}, '', url)
  },
}))
