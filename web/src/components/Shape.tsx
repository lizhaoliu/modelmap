/** A tensor shape with inline semantic dim labels: [1 batch × 7 seq × 5120 hidden].
 *  Labels come from value-matching (types.buildDimLabels); unmatched dims stay
 *  bare numbers. Pass `batch` only for activation shapes — weight tensors have
 *  no batch dimension. */
export function Shape({
  shape,
  labels,
  batch,
}: {
  shape: number[]
  labels: Map<number, string>
  batch?: number
}) {
  return (
    <span className="mm-shape">
      [
      {shape.map((d, i) => {
        const label =
          i === 0 && batch != null && d === batch && shape.length > 1
            ? 'batch'
            : labels.get(d)
        return (
          <span key={i}>
            {i > 0 && ' × '}
            {d}
            {label && <i className="mm-dimlab"> {label}</i>}
          </span>
        )
      })}
      ]
    </span>
  )
}
