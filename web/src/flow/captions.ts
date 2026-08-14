import type { GNode, GraphDoc } from '../types'
import type { Beat } from './beats'

const int = (n: number) => n.toLocaleString('en-US')

/** Plain-language, per-kind caption templated from config values — the
 *  educational layer of Flow mode (design doc §08, decided core). */
export function caption(node: GNode, doc: GraphDoc, beat: Beat): string {
  const c = doc.config as Record<string, number | undefined>
  const d = c.hidden_size ?? c.n_embd
  const heads = c.num_attention_heads ?? c.n_head
  const kv = c.num_key_value_heads
  const inter = c.intermediate_size ?? c.moe_intermediate_size
  const vocab = c.vocab_size
  const leaf = node.id.split('.').pop() ?? ''

  if (beat.member && (node.kind === 'container' || node.kind === 'module')) {
    return `one full block — attention mixes context between positions, the feed-forward transforms each one — repeated ${beat.member.count} times`
  }

  switch (node.kind) {
    case 'embedding':
      return vocab && d
        ? `look up each token id in a ${int(vocab)} × ${int(d)} table — ids become ${int(d)}-dim vectors`
        : 'token ids become dense vectors'
    case 'attention':
      return heads
        ? `${heads} heads${kv && kv !== heads ? ` (${kv} shared KV)` : ''} let every position attend to earlier positions and mix in their information`
        : 'positions attend to each other and exchange information'
    case 'mlp':
      return inter && d
        ? `position-wise MLP: expand ${int(d)} → ${int(inter)}, nonlinearity, project back to ${int(d)}`
        : 'position-wise feed-forward transform'
    case 'moe': {
      const experts = c.num_experts ?? c.n_routed_experts ?? c.num_local_experts
      const topk = c.num_experts_per_tok ?? c.moe_top_k
      return experts
        ? `a router sends each token to its top ${topk ?? '?'} of ${experts} experts — only those run`
        : 'mixture-of-experts: a router picks which experts process each token'
    }
    case 'norm':
      return 'normalize each position’s vector so scales stay stable from layer to layer'
    case 'head':
      return vocab
        ? `project to ${int(vocab)} vocabulary scores — the next-token logits`
        : 'project hidden states to output scores'
    case 'linear': {
      if (/^(q_proj|query)/.test(leaf)) return heads ? `project into ${heads} query heads` : 'query projection'
      if (/^(k_proj|key)/.test(leaf)) return `project into ${kv ?? heads ?? ''} key heads`.replace('  ', ' ')
      if (/^(v_proj|value)/.test(leaf)) return `project into ${kv ?? heads ?? ''} value heads`.replace('  ', ' ')
      if (/^(o_proj|out|dense)/.test(leaf)) return 'merge the attention heads back into the residual stream'
      if (/gate/.test(leaf)) return 'gating branch of the gated feed-forward'
      if (/up/.test(leaf)) return 'expand into the feed-forward inner dimension'
      if (/down/.test(leaf)) return 'project back down into the residual stream'
      return 'linear transformation'
    }
    default:
      if (/rotary/i.test(node.cls)) return 'precompute rotary position phases used by every attention layer'
      if (/dropout/i.test(node.cls)) return 'dropout — identity at inference time'
      return node.cls
  }
}
