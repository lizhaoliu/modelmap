import { BaseEdge, Position, getBezierPath, useInternalNode, type EdgeProps } from '@xyflow/react'
import type { MMData } from './layout'

/** Edge whose endpoints come from node geometry (position + explicit size)
 *  rather than DOM-measured handle bounds. Renders like the default bezier,
 *  but never depends on layout measurement timing or CSS transforms — our
 *  nodes have known sizes and fixed handle sides, so geometry is exact. */
export function MMEdge({ id, source, target, markerEnd, style }: EdgeProps) {
  const s = useInternalNode(source)
  const t = useInternalNode(target)
  if (!s || !t) return null
  const dir = (s.data as MMData).dir
  const sp = s.internals.positionAbsolute
  const tp = t.internals.positionAbsolute
  const sw = s.width ?? s.measured?.width ?? 0
  const sh = s.height ?? s.measured?.height ?? 0
  const tw = t.width ?? t.measured?.width ?? 0
  const th = t.height ?? t.measured?.height ?? 0

  const geo =
    dir === 'v'
      ? {
          sourceX: sp.x + sw / 2, sourceY: sp.y + sh, sourcePosition: Position.Bottom,
          targetX: tp.x + tw / 2, targetY: tp.y, targetPosition: Position.Top,
        }
      : {
          sourceX: sp.x + sw, sourceY: sp.y + sh / 2, sourcePosition: Position.Right,
          targetX: tp.x, targetY: tp.y + th / 2, targetPosition: Position.Left,
        }
  const [path] = getBezierPath(geo)
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />
}

export const edgeTypes = { mm: MMEdge }
