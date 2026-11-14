// Colours. One accent for the interface; the categorical series colours are
// only used to give model families an identity in the curve chart and its
// legend (validated for adjacent-pair and colour-vision separation on the
// panel surface).

export const ACCENT = '#46e0b4';
export const NULL_COLOR = '#8a94a0';

export const SERIES = [
  '#3987e5', // blue
  '#d95926', // orange
  '#199e70', // aqua
  '#c98500', // yellow
  '#d55181', // magenta
  '#0ca30c', // green
  '#9085e9', // violet
  '#e66767', // red
];

/** Colour by proposal order among the non-null options, so it never changes with rank. */
export function familyColors(order: string[], isNull: (id: string) => boolean): Record<string, string> {
  const out: Record<string, string> = {};
  let i = 0;
  for (const id of order) {
    if (isNull(id)) {
      out[id] = NULL_COLOR;
      continue;
    }
    out[id] = SERIES[i % SERIES.length] ?? NULL_COLOR;
    i += 1;
  }
  return out;
}
