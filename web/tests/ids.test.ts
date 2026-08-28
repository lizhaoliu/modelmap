import { describe, expect, it } from 'vitest'

import { normalizeModelId } from '../src/api'

// mirrors tests/test_ids.py — the client and server must agree on the
// canonical spelling, or the URL bar and the cache key drift apart
describe('normalizeModelId', () => {
  it('passes canonical ids through', () => {
    expect(normalizeModelId('Qwen/Qwen3-8B')).toBe('Qwen/Qwen3-8B')
    expect(normalizeModelId('  Qwen/Qwen3-8B ')).toBe('Qwen/Qwen3-8B')
    expect(normalizeModelId('local:/tmp/ckpt')).toBe('local:/tmp/ckpt')
  })

  it('strips Hub URL shapes', () => {
    expect(normalizeModelId('https://huggingface.co/Qwen/Qwen3-8B')).toBe('Qwen/Qwen3-8B')
    expect(normalizeModelId('http://www.huggingface.co/Qwen/Qwen3-8B/')).toBe('Qwen/Qwen3-8B')
    expect(normalizeModelId('hf.co/Qwen/Qwen3-8B')).toBe('Qwen/Qwen3-8B')
    expect(normalizeModelId('https://huggingface.co/Qwen/Qwen3-8B/tree/main')).toBe('Qwen/Qwen3-8B')
    expect(normalizeModelId('https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json')).toBe('Qwen/Qwen3-8B')
    expect(normalizeModelId('https://huggingface.co/Qwen/Qwen3-8B?x=1#y')).toBe('Qwen/Qwen3-8B')
  })

  it('keeps ollama-style variants', () => {
    expect(normalizeModelId('hf.co/ryzdfm/some-model:Q4_K_M')).toBe('ryzdfm/some-model:Q4_K_M')
  })

  it('turns GGUF file paths into variant requests', () => {
    expect(normalizeModelId('o/repo-GGUF/model-Q8_0.gguf')).toBe('o/repo-GGUF:model-Q8_0')
    expect(normalizeModelId('o/repo-GGUF/Ornith-1.5-35B-Q8_0')).toBe('o/repo-GGUF:Ornith-1.5-35B-Q8_0')
  })

  it('drops non-variant repo subpaths', () => {
    expect(normalizeModelId('o/repo/model.safetensors')).toBe('o/repo')
    expect(normalizeModelId('a/b/c')).toBe('a/b')
  })

  it('fixes typographic dashes and zero-width junk', () => {
    expect(normalizeModelId('Qwen/Qwen3.5‑27B')).toBe('Qwen/Qwen3.5-27B')
    expect(normalizeModelId('Qwen/Qwen3–8B')).toBe('Qwen/Qwen3-8B')
    expect(normalizeModelId('Qwen/​Qwen3-8B')).toBe('Qwen/Qwen3-8B')
  })
})
