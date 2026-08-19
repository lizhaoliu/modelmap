import ELK from 'elkjs/lib/elk-api.js'
import type { Edge, Node } from '@xyflow/react'
import { MarkerType } from '@xyflow/react'
import type { GNode, GraphDoc, GraphIndex, Repeat } from '../types'

export interface MMData extends Record<string, unknown> {
  g: GNode
  /** set when this node IS a repeat's representative ("1 of N") */
  repeat?: Repeat
  /** set when this container's children collapsed into a repeat ("×N" stack);
   *  the first run — opening the stack opens its representative */
  stackOf?: Repeat
  /** total children behind the stack: every run's count + un-collapsed siblings */
  stackTotal?: number
  /** counts of each repeated design when a parent holds several (DeepSeek-V3
   *  dense+MoE [3, 58]; DeepSeek-V4 interleaved [20, 20]) and how many
   *  children stayed unique (un-collapsed) */
  stackRuns?: number[]
  stackLoose?: number
  hasChildren: boolean
  expanded: boolean
  /** layout direction of the parent container — decides handle placement */
  dir: 'h' | 'v'
}

export type MMNode = Node<MMData>

const NODE_W = 176
const NODE_H = 56
const CLOSED_W = 198
const CLOSED_H = 60

const EDGE_COLOR = '#8A94A4'

const BASE_OPTS = {
  'elk.algorithm': 'layered',
  'elk.layered.spacing.nodeNodeBetweenLayers': '44',
  'elk.spacing.nodeNode': '24',
}

function mkEdge(id: string, source: string, target: string, aux: boolean): Edge {
  return {
    id,
    type: 'mm',
    source,
    target,
    data: { aux },
    style: aux
      ? { stroke: EDGE_COLOR, strokeWidth: 1.1, strokeDasharray: '4 4', opacity: 0.8 }
      : { stroke: EDGE_COLOR, strokeWidth: 1.2 },
    markerEnd: { type: MarkerType.ArrowClosed, width: 15, height: 15, color: EDGE_COLOR },
  }
}

/** Count and describe what a collapsed "×N" stack stands for. */
export function stackInfo(index: GraphIndex, id: string): { stackTotal?: number; stackRuns?: number[]; stackLoose?: number } {
  const runs = index.repeatsByParent.get(id)
  if (!runs?.length) return {}
  const kids = index.children.get(id) ?? []
  const reps = new Set(runs.map((r) => r.representative))
  const loose = kids.filter((k) => !reps.has(k.id)).length
  const total = runs.reduce((s, r) => s + r.count, 0) + loose
  return { stackTotal: total, stackRuns: runs.length > 1 || loose ? runs.map((r) => r.count) : undefined, stackLoose: loose }
}

/** "43 blocks — 2 repeated designs (×20, ×20) + 3 unique" */
export function stackTitle(total: number, runs: number[] | undefined, loose: number | undefined): string {
  if (!runs) return `${total} structurally identical blocks`
  return `${total} blocks — ${runs.length} repeated design${runs.length > 1 ? 's' : ''} (${runs.map((c) => '×' + c).join(', ')})${loose ? ` + ${loose} unique` : ''}`
}

/** Top levels read left-to-right like the design-doc hero; block internals
 *  stack top-to-bottom so expanded models stay compact. */
function dirFor(depth: number): 'RIGHT' | 'DOWN' {
  return depth <= 2 ? 'RIGHT' : 'DOWN'
}

/** One ELK instance, running in a Web Worker so layout of large expansions
 *  never blocks the main thread (design doc §07). The worker chunk is
 *  code-split out of the main bundle. */
let _elk: InstanceType<typeof ELK> | null = null
function getElk() {
  if (!_elk) {
    _elk = new ELK({
      workerFactory: () =>
        new Worker(new URL('elkjs/lib/elk-worker.min.js', import.meta.url), { type: 'classic' }),
    })
  }
  return _elk
}

interface ElkShape {
  id: string
  width?: number
  height?: number
  x?: number
  y?: number
  children?: ElkShape[]
  edges?: { id: string; sources: string[]; targets: string[] }[]
  layoutOptions?: Record<string, string>
}

export interface Rect {
  x: number
  y: number
  w: number
  h: number
}

export async function layoutGraph(
  doc: GraphDoc,
  index: GraphIndex,
  expanded: Set<string>,
): Promise<{ nodes: MMNode[]; edges: Edge[]; positions: Record<string, Rect> }> {
  const edgesByParent = new Map<string, { src: string; dst: string; kind?: 'flow' | 'aux' }[]>()
  for (const e of doc.edges) {
    const parent = index.byId.get(e.src)?.parent
    if (parent == null) continue
    const list = edgesByParent.get(parent) ?? []
    list.push(e)
    edgesByParent.set(parent, list)
  }

  const auxIds = new Set(doc.edges.filter((e) => e.kind === 'aux').map((e) => `${e.src}→${e.dst}`))
  const elkEdgesFor = (parentId: string) =>
    (edgesByParent.get(parentId) ?? []).map((e) => ({
      id: `${e.src}→${e.dst}`,
      sources: [e.src],
      targets: [e.dst],
    }))

  const build = (n: GNode): ElkShape => {
    const kids = expanded.has(n.id) ? index.children.get(n.id) ?? [] : []
    if (!kids.length) {
      const isClosedContainer = (index.children.get(n.id) ?? []).length > 0
      return {
        id: n.id,
        width: isClosedContainer ? CLOSED_W : NODE_W,
        height: isClosedContainer ? CLOSED_H : NODE_H,
      }
    }
    return {
      id: n.id,
      layoutOptions: {
        ...BASE_OPTS,
        'elk.direction': dirFor(n.depth + 1),
        'elk.padding': '[top=46.0,left=16.0,bottom=16.0,right=16.0]',
      },
      children: kids.map(build),
      edges: elkEdgesFor(n.id),
    }
  }

  const topLevel = index.children.get('') ?? []
  const root: ElkShape = {
    id: '__root__',
    layoutOptions: { ...BASE_OPTS, 'elk.direction': dirFor(0) },
    children: topLevel.map(build),
    edges: elkEdgesFor(''),
  }

  const laid = (await getElk().layout(
    root as unknown as Parameters<ReturnType<typeof getElk>['layout']>[0],
  )) as unknown as ElkShape

  const nodes: MMNode[] = []
  const edges: Edge[] = []
  const positions: Record<string, Rect> = {}

  const walk = (
    shapes: ElkShape[],
    parentId: string | undefined,
    dir: 'h' | 'v',
    offX = 0,
    offY = 0,
  ) => {
    for (const s of shapes) {
      const g = index.byId.get(s.id)
      if (!g) continue
      const kids = s.children ?? []
      const isExpanded = kids.length > 0
      const childDir: 'h' | 'v' = dirFor(g.depth + 1) === 'RIGHT' ? 'h' : 'v'
      const absX = offX + (s.x ?? 0)
      const absY = offY + (s.y ?? 0)
      positions[s.id] = { x: absX, y: absY, w: s.width ?? 0, h: s.height ?? 0 }
      nodes.push({
        id: s.id,
        type: isExpanded ? 'containerNode' : 'moduleNode',
        position: { x: s.x ?? 0, y: s.y ?? 0 },
        // explicit dimensions (from ELK) so fitView and the MiniMap work
        // before/without DOM measurement
        width: s.width,
        height: s.height,
        ...(parentId ? { parentId } : {}),
        data: {
          g,
          repeat: index.repeatByRep.get(s.id),
          stackOf: index.repeatsByParent.get(s.id)?.[0],
          ...stackInfo(index, s.id),
          hasChildren: (index.children.get(s.id) ?? []).length > 0,
          expanded: isExpanded,
          dir,
        },
        draggable: false,
      })
      for (const e of s.edges ?? []) edges.push(mkEdge(e.id, e.sources[0], e.targets[0], auxIds.has(e.id)))
      walk(kids, s.id, childDir, absX, absY)
    }
  }
  walk(laid.children ?? [], undefined, 'h')
  for (const e of laid.edges ?? []) edges.push(mkEdge(e.id, e.sources[0], e.targets[0], auxIds.has(e.id)))

  return { nodes, edges, positions }
}
