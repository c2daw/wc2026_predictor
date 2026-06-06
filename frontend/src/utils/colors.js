function hexToRgb(hex) {
  const h = hex.replace('#', '')
  return [parseInt(h.slice(0,2), 16), parseInt(h.slice(2,4), 16), parseInt(h.slice(4,6), 16)]
}

function lerpColor(c1, c2, t) {
  return c1.map((v, i) => Math.round(v + (c2[i] - v) * t))
}

function makeScale(stops) {
  return stops.map(([pos, hex]) => [pos, hexToRgb(hex)])
}

const SCALES = {
  a: makeScale([[0,'#F3F4F6'],[0.2,'#FFD0D0'],[0.6,'#FF4444'],[1,'#D60000']]),
  b: makeScale([[0,'#F3F4F6'],[0.2,'#D6E4FF'],[0.6,'#304FFF'],[1,'#0D1B6E']]),
  d: makeScale([[0,'#F3F4F6'],[0.2,'#EAD5EC'],[0.6,'#9B35A8'],[1,'#5C0E6A']]),
}

export function getCellColor(type, intensity) {
  const scale = SCALES[type]
  const s = Math.min(Math.max(intensity, 0), 1)
  let seg = scale.length - 2
  for (let i = 0; i < scale.length - 1; i++) {
    if (s <= scale[i+1][0]) { seg = i; break }
  }
  const t0 = scale[seg][0], t1 = scale[seg+1][0]
  const tt = t0 === t1 ? 0 : (s - t0) / (t1 - t0)
  const [r, g, b] = lerpColor(scale[seg][1], scale[seg+1][1], tt)
  return `rgb(${r},${g},${b})`
}

export function gradFill(value, vmax, hex, lo = 0.06, hi = 0.80) {
  const a = lo + (hi - lo) * Math.min(value / Math.max(vmax, 1e-6), 1.0)
  const [r, g, b] = hexToRgb(hex)
  return `rgba(${r},${g},${b},${a.toFixed(2)})`
}

export const NAVY  = '#19237C'
export const LIME  = '#AFEA00'
export const RED   = '#D60000'
export const BLUE2 = '#304FFF'
export const GREEN = '#00C651'
