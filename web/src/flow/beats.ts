import type { GraphDoc, GraphIndex } from '../types'

/** One animation beat: the pulse dwells on a visible node, showing real
 *  tensor shapes from the trace. Consecutive trace steps that land on the
 *  same visible node (e.g. everything inside a collapsed block) compress
 *  into a single beat; passes through a repeat stack keep a member counter
 *  ("layer 12 / 36") instead of expanding into 36× the beats. */
export interface BeatMember {
  index: string
  ordinal: number
  count: number
}

export interface Beat {
  node: string
  member?: BeatMember
  inShapes: number[][]
  outShapes: number[][]
  start: number
  dur: number
}

export interface FlowScript {
  beats: Beat[]
  total: number
}

const LEAF_DWELL = 0.9 // seconds per ordinary beat at 1×
const STACK_BUDGET = 8 // seconds across ALL repeat-member beats at 1×
const MIN_BEAT = 0.05

export function buildFlowScript(
  doc: GraphDoc,
  index: GraphIndex,
  expanded: Set<string>,
): FlowScript {
  if (!doc.trace.length) return { beats: [], total: 0 }
  // outermost repeats first, so the member counter reflects the layer stack
  // rather than e.g. an expert stack nested inside it
  const repeats = [...doc.repeats].sort((a, b) => a.parent.length - b.parent.length)

  const isVisible = (id: string): boolean => {
    let p = index.byId.get(id)?.parent
    while (p != null && p !== '') {
      if (!expanded.has(p)) return false
      p = index.byId.get(p)?.parent ?? null
    }
    return true
  }

  const climbToVisible = (id: string): string | null => {
    let cur = id
    while (cur !== '') {
      if (index.byId.has(cur) && isVisible(cur)) return cur
      cur = cur.includes('.') ? cur.slice(0, cur.lastIndexOf('.')) : ''
    }
    return null
  }

  /** Trace node → (visible node, member pass). Collapsed members rewrite to
   *  the representative ("layers.12.x" → "layers.0.x") before climbing. */
  const mapNode = (tid: string): { vid: string; key: string | null; member?: BeatMember } | null => {
    let id = tid
    let member: BeatMember | undefined
    let key: string | null = null
    let changed = true
    while (changed) {
      changed = false
      for (const r of repeats) {
        const pfx = r.parent + '.'
        if (!id.startsWith(pfx)) continue
        const seg = id.slice(pfx.length).split('.', 1)[0]
        const ordinal = r.members.indexOf(seg)
        if (ordinal === -1) continue
        if (!member) {
          member = { index: seg, ordinal: ordinal + 1, count: r.count }
          key = `${r.parent}#${seg}`
        }
        const repLeaf = r.representative.slice(pfx.length)
        if (seg !== repLeaf) {
          id = r.representative + id.slice(pfx.length + seg.length)
          changed = true
        }
      }
    }
    const vid = climbToVisible(id)
    if (vid == null || vid === '') return null
    // expanded containers don't dwell — their internals carry the animation
    const kids = index.children.get(vid) ?? []
    if (kids.length && expanded.has(vid)) return null
    return { vid, key, member }
  }

  type Run = Beat & { _key: string | null }
  const runs: Run[] = []
  for (const t of doc.trace) {
    const m = mapNode(t.node)
    if (!m) continue
    const last = runs[runs.length - 1]
    if (last && last.node === m.vid && last._key === m.key) {
      last.outShapes = t.outputs.length ? t.outputs : last.outShapes
      continue
    }
    runs.push({
      node: m.vid,
      member: m.member,
      inShapes: t.inputs,
      outShapes: t.outputs,
      start: 0,
      dur: 0,
      _key: m.key,
    })
  }

  const stackCount = runs.filter((b) => b.member).length
  const stackDur = stackCount ? Math.max(STACK_BUDGET / stackCount, MIN_BEAT) : 0
  let t = 0
  for (const b of runs) {
    b.dur = b.member ? stackDur : LEAF_DWELL
    b.start = t
    t += b.dur
  }
  return { beats: runs.map(({ _key, ...b }) => b), total: t }
}
