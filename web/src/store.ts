import { createContext, useContext } from 'react'
import { create, type StoreApi, type UseBoundStore } from 'zustand'
import { fetchGraph } from './api'
import { buildIndex, type GraphDoc, type GraphIndex } from './types'

export type DiffStatus = 'same' | 'changed' | 'added' | 'removed' | 'inside'

export interface State {
  doc: GraphDoc | null
  index: GraphIndex | null
  loading: string | null // model id being loaded
  error: string | null
  errorModel: string | null // id whose load failed, for the retry button
  expanded: Set<string>
  selected: string | null
  /** container just opened — the canvas frames it once, then clears this */
  lastExpanded: string | null
  clearLastExpanded: () => void
  /** transient message (render-budget collapses, copied links, …) */
  toast: string | null
  setToast: (msg: string | null) => void
  /** compare mode: per-node diff status for this side (set by the compare view) */
  diff: Map<string, DiffStatus> | null
  setDiff: (d: Map<string, DiffStatus> | null) => void
  /** compare mode: apply an expansion set computed elsewhere (linked expansion) */
  setExpanded: (e: Set<string>) => void
  loadModel: (id: string, opts?: { push?: boolean }) => Promise<void>
  toggleExpand: (id: string) => void
  expand: (id: string) => void
  expandMany: (ids: string[]) => void
  collapse: (id: string) => void
  select: (id: string | null) => void
}

/** Containers open by default: the root and its immediate container children.
 *  Everything deeper starts closed (design doc §07: first view ≈ 10 nodes).
 *  Vision-language models get one more level — their depth-2 children are the
 *  vision tower and the language model, each of which is the real story. */
function defaultExpanded(index: GraphIndex, doc: GraphDoc): Set<string> {
  const out = new Set<string>()
  const isVLM = doc.config.vision_config != null
  for (const [id, kids] of index.children.entries()) {
    if (id === null || !kids.length) continue
    const node = index.byId.get(id)
    if (!node) continue
    if (node.depth <= 1) out.add(id)
    else if (isVLM && node.depth === 2 && !index.repeatsByParent.has(id) && kids.length >= 2) out.add(id)
  }
  out.add('')
  return out
}

/** Design doc §10 render budget: never mount more than this many nodes.
 *  When an expansion would exceed it, the least recently opened containers
 *  (never the one just opened, nor its ancestors) are collapsed, with a toast. */
export const RENDER_BUDGET = 300
const openOrder: string[] = [] // least-recent first

function countVisible(index: GraphIndex, expanded: Set<string>): number {
  let n = 0
  const walk = (parent: string) => {
    for (const c of index.children.get(parent) ?? []) {
      n++
      if (expanded.has(c.id)) walk(c.id)
    }
  }
  walk('')
  return n
}

function enforceBudget(state: State, expanded: Set<string>, opened: string | null): Set<string> {
  const { index } = state
  if (!index) return expanded
  for (const id of expanded) if (!openOrder.includes(id)) openOrder.push(id)
  for (let i = openOrder.length - 1; i >= 0; i--) if (!expanded.has(openOrder[i])) openOrder.splice(i, 1)
  const protect = new Set<string>([''])
  if (opened) {
    let cur: string | null = opened
    while (cur != null) {
      protect.add(cur)
      cur = index.byId.get(cur)?.parent ?? null
    }
  }
  let collapsed = 0
  while (countVisible(index, expanded) > RENDER_BUDGET) {
    const victim = openOrder.find((id) => !protect.has(id) && expanded.has(id))
    if (!victim) break
    expanded.delete(victim)
    openOrder.splice(openOrder.indexOf(victim), 1)
    collapsed++
  }
  if (collapsed) {
    setTimeout(() =>
      useStore.getState().setToast(
        `Collapsed ${collapsed} container${collapsed > 1 ? 's' : ''} to stay under ${RENDER_BUDGET} visible modules`,
      ),
    )
  }
  return expanded
}

export interface StoreOptions {
  /** write ?sel= and push /m/{id} history — only the primary store should */
  syncUrl: boolean
}

export type GraphStore = UseBoundStore<StoreApi<State>>

export function createGraphStore(opts: StoreOptions): GraphStore {
  return create<State>((set, get) => ({
  doc: null,
  index: null,
  loading: null,
  error: null,
  errorModel: null,
  expanded: new Set(),
  selected: null,
  lastExpanded: null,
  clearLastExpanded: () => set({ lastExpanded: null }),
  toast: null,
  setToast: (msg) => {
    set({ toast: msg })
    if (msg) setTimeout(() => get().toast === msg && set({ toast: null }), 3200)
  },

  async loadModel(id, loadOpts) {
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
        expanded: defaultExpanded(index, doc),
        selected: null,
      })
      if (opts.syncUrl && loadOpts?.push !== false) {
        history.pushState({}, '', `/m/${id}`)
      }
      if (opts.syncUrl) document.title = `${id.split('/').pop()} · modelmap`
    } catch (e) {
      set({ loading: null, error: e instanceof Error ? e.message : String(e), errorModel: id })
    }
  },

  toggleExpand(id) {
    const expanded = new Set(get().expanded)
    if (expanded.has(id)) {
      expanded.delete(id)
      set({ expanded, lastExpanded: null })
    } else {
      expanded.add(id)
      set({ expanded: enforceBudget(get(), expanded, id), lastExpanded: id })
    }
  },

  expand(id) {
    get().expandMany([id])
  },

  expandMany(ids) {
    const { index } = get()
    const expanded = new Set(get().expanded)
    let opened: string | null = null
    for (const id of ids) {
      if (index?.children.get(id)?.length && !expanded.has(id)) {
        expanded.add(id)
        opened = opened ?? id
      }
    }
    set({ expanded: enforceBudget(get(), expanded, opened), lastExpanded: opened })
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
    if (!opts.syncUrl) return
    const url = new URL(location.href)
    if (id) url.searchParams.set('sel', id)
    else url.searchParams.delete('sel')
    history.replaceState({}, '', url)
  },

  diff: null,
  setDiff: (diff) => set({ diff }),
  setExpanded: (expanded) => set({ expanded, lastExpanded: null }),
  }))
}

/** The primary store (the /m/{id} view). Compare mode creates two more. */
export const primaryStore = createGraphStore({ syncUrl: true })
export const StoreContext = createContext<GraphStore>(primaryStore)

/** Components read whichever store their subtree is bound to. */
export function useStore<T>(selector: (s: State) => T): T {
  const store = useContext(StoreContext)
  return store(selector)
}
useStore.getState = () => primaryStore.getState()
