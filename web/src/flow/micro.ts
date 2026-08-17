import type { GNode, GraphDoc, GraphIndex } from '../types'
import type { Beat } from './beats'

/** Micro-views: the inner choreography of a block / attention / MLP / MoE
 *  beat (design doc §08). One template per kind, filled from the model's
 *  config and real traced shapes, so it generalizes across families. */

export type MicroKind = 'block' | 'attention' | 'mlp' | 'moe'

export interface Stage {
  /** short mono label, e.g. "q_proj · k_proj · v_proj" */
  label: string
  /** one shape per row (multi-row for parallel branches) */
  shapes: number[][]
  /** plain-language caption */
  note: string
  kind: 'embedding' | 'attention' | 'mlp' | 'moe' | 'norm' | 'linear' | 'op' | 'residual'
}

export interface MicroScript {
  kind: MicroKind
  title: string
  stages: Stage[]
}

function num(c: Record<string, unknown>, ...keys: string[]): number | undefined {
  for (const k of keys) if (typeof c[k] === 'number') return c[k] as number
  return undefined
}

/** Which micro-view a beat's node gets, if any. */
export function microKindFor(node: GNode, index: GraphIndex): MicroKind | null {
  if (node.kind === 'attention') return 'attention'
  if (node.kind === 'mlp') return 'mlp'
  if (node.kind === 'moe') return 'moe'
  const block = blockOf(node, index)
  return block ? 'block' : null
}

/** The decoder/encoder block a beat represents: the node itself if it has an
 *  attention child, or (for a collapsed repeat stack) its representative. */
function blockOf(node: GNode, index: GraphIndex): GNode | null {
  const hasAttn = (n: GNode) =>
    (index.children.get(n.id) ?? []).some((c) => c.kind === 'attention')
  if (hasAttn(node)) return node
  const rep = index.repeatByParent.get(node.id)?.representative
  const repNode = rep ? index.byId.get(rep) : undefined
  return repNode && hasAttn(repNode) ? repNode : null
}

export function buildMicro(
  node: GNode,
  doc: GraphDoc,
  index: GraphIndex,
  beat: Beat,
): MicroScript | null {
  const kind = microKindFor(node, index)
  if (!kind) return null
  const c = doc.config as Record<string, unknown>
  const T = beat.inShapes[0]?.[1] ?? doc.trace[0]?.inputs?.[0]?.[1] ?? 7
  const d = num(c, 'hidden_size', 'n_embd') ?? beat.inShapes[0]?.at(-1) ?? 0
  const heads = num(c, 'num_attention_heads', 'n_head') ?? 0
  const kv = num(c, 'num_key_value_heads') ?? heads
  const hd = num(c, 'head_dim') ?? (heads ? Math.round(d / heads) : 0)
  const ffn = num(c, 'intermediate_size', 'moe_intermediate_size', 'n_inner') ?? 4 * d
  const E = num(c, 'num_experts', 'n_routed_experts', 'num_local_experts') ?? 0
  const k = num(c, 'num_experts_per_tok', 'moe_top_k') ?? 0
  const io = [1, T, d]

  if (kind === 'attention') {
    const kvNote = kv !== heads ? ` (${kv} shared K/V heads — grouped-query)` : ''
    return {
      kind,
      title: 'inside attention',
      stages: [
        { label: 'x', shapes: [io], note: 'hidden states in', kind: 'op' },
        {
          label: 'q_proj · k_proj · v_proj',
          shapes: [[1, heads, T, hd], [1, kv, T, hd], [1, kv, T, hd]],
          note: `project into ${heads} heads of ${hd} dims${kvNote}`,
          kind: 'linear',
        },
        { label: 'q·kᵀ / √d', shapes: [[1, heads, T, T]], note: 'every position scores every earlier position', kind: 'op' },
        { label: 'softmax', shapes: [[1, heads, T, T]], note: 'scores become weights that sum to 1 per row', kind: 'op' },
        { label: 'weights · v', shapes: [[1, heads, T, hd]], note: 'mix values by their weights', kind: 'op' },
        { label: 'o_proj', shapes: [io], note: 'merge heads back into the residual stream', kind: 'linear' },
      ],
    }
  }

  if (kind === 'mlp') {
    const kids = index.children.get(node.id) ?? []
    const gated = kids.some((x) => /gate/.test(x.id.split('.').pop() ?? ''))
    return {
      kind,
      title: gated ? 'inside the gated MLP' : 'inside the MLP',
      stages: gated
        ? [
            { label: 'x', shapes: [io], note: 'hidden states in', kind: 'op' },
            { label: 'gate_proj · up_proj', shapes: [[1, T, ffn], [1, T, ffn]], note: `expand ${d} → ${ffn}, twice`, kind: 'linear' },
            { label: 'act(gate) ⊙ up', shapes: [[1, T, ffn]], note: 'the gate decides what passes through', kind: 'op' },
            { label: 'down_proj', shapes: [io], note: `project back to ${d}`, kind: 'linear' },
          ]
        : [
            { label: 'x', shapes: [io], note: 'hidden states in', kind: 'op' },
            { label: 'up', shapes: [[1, T, ffn]], note: `expand ${d} → ${ffn}`, kind: 'linear' },
            { label: 'activation', shapes: [[1, T, ffn]], note: 'nonlinearity', kind: 'op' },
            { label: 'down', shapes: [io], note: `project back to ${d}`, kind: 'linear' },
          ],
    }
  }

  if (kind === 'moe') {
    return {
      kind,
      title: 'inside the mixture of experts',
      stages: [
        { label: 'x', shapes: [io], note: 'hidden states in', kind: 'op' },
        { label: 'router', shapes: [[1, T, E]], note: `score all ${E} experts for each token`, kind: 'linear' },
        { label: `top-${k}`, shapes: [[T, k]], note: `keep the ${k} best experts per token`, kind: 'op' },
        { label: `${k} of ${E} experts`, shapes: Array.from({ length: Math.min(k, 3) }, () => io), note: 'only the chosen experts run', kind: 'mlp' },
        { label: 'weighted sum', shapes: [io], note: 'combine expert outputs by router weight', kind: 'op' },
      ],
    }
  }

  // block: real children in traced execution order, with residual adds
  const block = blockOf(node, index)!
  const kids = [...(index.children.get(block.id) ?? [])]
    .filter((x) => ['norm', 'attention', 'mlp', 'moe'].includes(x.kind))
    .sort((a, b) => (index.traceByNode.get(a.id)?.step ?? 1e9) - (index.traceByNode.get(b.id)?.step ?? 1e9))
  const stages: Stage[] = [{ label: 'x', shapes: [io], note: 'residual stream in', kind: 'op' }]
  for (const ch of kids) {
    const t = index.traceByNode.get(ch.id)
    const shape = t?.outputs?.[0] ?? io
    const leaf = ch.id.split('.').pop() ?? ch.id
    if (ch.kind === 'norm') stages.push({ label: leaf, shapes: [shape], note: 'normalize before the sub-layer', kind: 'norm' })
    else if (ch.kind === 'attention') {
      stages.push({ label: leaf, shapes: [shape], note: 'positions exchange information', kind: 'attention' })
      stages.push({ label: '⊕', shapes: [io], note: 'add back to the residual stream', kind: 'residual' })
    } else {
      stages.push({ label: leaf, shapes: [shape], note: ch.kind === 'moe' ? 'routed experts transform each position' : 'feed-forward transforms each position', kind: ch.kind === 'moe' ? 'moe' : 'mlp' })
      stages.push({ label: '⊕', shapes: [io], note: 'add back to the residual stream', kind: 'residual' })
    }
  }
  return { kind, title: 'inside one block', stages }
}
