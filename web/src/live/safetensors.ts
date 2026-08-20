/** Minimal safetensors reader for Live mode (design doc §20): parse the JSON
 *  header of a fetched checkpoint and hand out tensors as Float32Arrays.
 *  BF16/F16 convert on read; every tensor is copied out of the file buffer
 *  (safetensors data offsets carry no alignment guarantee), so the original
 *  download buffer can be released after loading. */

export interface STMeta {
  dtype: string
  shape: number[]
  begin: number // absolute byte offset into the buffer
  end: number
}

export interface STFile {
  tensors: Map<string, STMeta>
  buf: ArrayBuffer
}

export function parseSafetensors(buf: ArrayBuffer): STFile {
  const dv = new DataView(buf)
  const n = Number(dv.getBigUint64(0, true))
  if (n <= 0 || n > 100_000_000 || 8 + n > buf.byteLength) throw new Error('not a safetensors file')
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)))
  const base = 8 + n
  const tensors = new Map<string, STMeta>()
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue
    const t = info as { dtype: string; shape: number[]; data_offsets: [number, number] }
    tensors.set(name, {
      dtype: t.dtype.toUpperCase(),
      shape: t.shape,
      begin: base + t.data_offsets[0],
      end: base + t.data_offsets[1],
    })
  }
  return { tensors, buf }
}

function f16ToF32(u: number): number {
  const sign = (u & 0x8000) ? -1 : 1
  const exp = (u >> 10) & 0x1f
  const frac = u & 0x3ff
  if (exp === 0) return sign * frac * 2 ** -24
  if (exp === 31) return frac ? NaN : sign * Infinity
  return sign * (1 + frac / 1024) * 2 ** (exp - 15)
}

/** Copy a tensor out as Float32Array (converting BF16 / F16). */
export function tensorF32(file: STFile, meta: STMeta): Float32Array {
  const bytes = meta.end - meta.begin
  if (meta.dtype === 'F32') {
    const out = new Float32Array(bytes / 4)
    // byte-wise copy: the source offset may not be 4-aligned
    new Uint8Array(out.buffer).set(new Uint8Array(file.buf, meta.begin, bytes))
    return out
  }
  if (meta.dtype === 'BF16') {
    const src = new Uint8Array(file.buf, meta.begin, bytes)
    const n = bytes / 2
    const out = new Float32Array(n)
    const u32 = new Uint32Array(out.buffer)
    for (let i = 0; i < n; i++) u32[i] = (src[2 * i] | (src[2 * i + 1] << 8)) << 16
    return out
  }
  if (meta.dtype === 'F16') {
    const src = new Uint8Array(file.buf, meta.begin, bytes)
    const n = bytes / 2
    const out = new Float32Array(n)
    for (let i = 0; i < n; i++) out[i] = f16ToF32(src[2 * i] | (src[2 * i + 1] << 8))
    return out
  }
  throw new Error(`unsupported tensor dtype ${meta.dtype} — Live mode needs an F32/BF16/F16 checkpoint`)
}

export function numel(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1)
}
