import { create } from 'zustand'
import type { Edge } from '@xyflow/react'
import type { MMNode, Rect } from './layout'

/** The primary canvas publishes its laid-out view here so the export menu
 *  can render exactly what is on screen (SVG / PNG) without touching the DOM. */
interface ViewState {
  nodes: MMNode[]
  edges: Edge[]
  positions: Record<string, Rect>
  set: (v: { nodes: MMNode[]; edges: Edge[]; positions: Record<string, Rect> }) => void
}
export const useViewStore = create<ViewState>((set) => ({
  nodes: [], edges: [], positions: {},
  set: (v) => set(v),
}))
