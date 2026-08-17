import { create } from 'zustand'
import { createGraphStore, type GraphStore } from '../store'
import { align, sideStatus, type Alignment } from './align'

/** Compare mode: two independent graph stores plus their alignment, with
 *  expansion and selection mirrored across the pair. */
interface CompareState {
  ids: [string, string] | null
  a: GraphStore
  b: GraphStore
  alignment: Alignment | null
  loading: boolean
  error: string | null
  diffOnly: boolean
  load: (idA: string, idB: string) => Promise<void>
  setDiffOnly: (v: boolean) => void
  reset: () => void
}

let unsub: (() => void)[] = []

export const useCompareStore = create<CompareState>((set, get) => ({
  ids: null,
  a: createGraphStore({ syncUrl: false }),
  b: createGraphStore({ syncUrl: false }),
  alignment: null,
  loading: false,
  error: null,
  diffOnly: false,

  async load(idA, idB) {
    get().reset()
    const a = createGraphStore({ syncUrl: false })
    const b = createGraphStore({ syncUrl: false })
    set({ ids: [idA, idB], a, b, loading: true, error: null, alignment: null })
    await Promise.all([a.getState().loadModel(idA, { push: false }), b.getState().loadModel(idB, { push: false })])
    const sa = a.getState(), sb = b.getState()
    const err = sa.error ?? sb.error
    if (err || !sa.doc || !sa.index || !sb.doc || !sb.index) {
      set({ loading: false, error: err ?? 'could not load both models' })
      return
    }
    const alignment = align(sa.doc, sa.index, sb.doc, sb.index)
    const st = sideStatus(alignment)
    sa.setDiff(st.a)
    sb.setDiff(st.b)
    set({ alignment, loading: false })
    document.title = `${idA.split('/').pop()} vs ${idB.split('/').pop()} · modelmap`

    // ---- link expansion + selection both ways (guarded against ping-pong)
    let busy = false
    const mirror = (from: GraphStore, to: GraphStore, map: (id: string) => string | undefined) =>
      from.subscribe((s, prev) => {
        if (busy) return
        busy = true
        try {
          if (s.expanded !== prev.expanded) {
            const next = new Set<string>()
            for (const id of s.expanded) {
              const twin = map(id)
              if (twin != null) next.add(twin)
            }
            // keep the other side's expansions that have no twin (added subtrees)
            for (const id of to.getState().expanded) if (!mapBack(to, from, id)) next.add(id)
            to.getState().setExpanded(next)
          }
          if (s.selected !== prev.selected) {
            to.getState().select(s.selected != null ? map(s.selected) ?? null : null)
          }
        } finally {
          busy = false
        }
      })
    const aToB = (id: string) => alignment.byA.get(id)?.b?.id
    const bToA = (id: string) => alignment.byB.get(id)?.a?.id
    const mapBack = (side: GraphStore, _other: GraphStore, id: string) =>
      side === b ? bToA(id) != null : aToB(id) != null
    unsub = [mirror(a, b, aToB), mirror(b, a, bToA)]
    // seed: mirror A's default expansion onto B once
    const seed = new Set<string>()
    for (const id of sa.expanded) { const t = aToB(id); if (t != null) seed.add(t) }
    for (const id of sb.expanded) if (bToA(id) == null) seed.add(id)
    sb.setExpanded(seed)
  },

  setDiffOnly(v) {
    set({ diffOnly: v })
    const { a, alignment } = get()
    if (!alignment) return
    if (v) {
      // collapse containers whose subtree is unchanged
      const keepA = new Set([...a.getState().expanded].filter((id) => id === '' || alignment.byA.get(id)?.dirty))
      a.getState().setExpanded(keepA) // mirroring carries it to B
    }
  },

  reset() {
    for (const u of unsub) u()
    unsub = []
  },
}))
