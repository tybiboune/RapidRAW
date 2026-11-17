// "Affected areas" (#2059): hold Alt on a tone, Colour Mixer or Colour Grading control and the preview
// shows where that control acts, at any value including zero. The pixels it acts on keep their
// resulting colour (tonal controls: their brightness, in grey), scaled by how strongly; the rest is
// black. It is only a view: it travels with the render request, never with the adjustments, so it
// cannot reach history, sidecars, exports or caches.

export type InfluenceKind = 'hsl' | 'tone' | 'grading';

export interface InfluenceTarget {
  kind: InfluenceKind;
  // hsl: band * 3 + slider (0 hue, 1 saturation, 2 luminance); tone: 0 highlights, 1 shadows,
  // 2 whites, 3 blacks; grading: 0 shadows, 1 midtones, 2 highlights
  index: number;
  label: string;
}

export const TONE_INFLUENCE = { highlights: 0, shadows: 1, whites: 2, blacks: 3 } as const;
export const GRADING_INFLUENCE = { shadows: 0, midtones: 1, highlights: 2 } as const;
export const HSL_INFLUENCE_PROPERTIES = ['hue', 'saturation', 'luminance'] as const;

const ATTRIBUTE = 'data-influence';
const LABEL_ATTRIBUTE = 'data-influence-label';

export const influenceAttributes = (target?: InfluenceTarget | null): Record<string, string> =>
  target ? { [ATTRIBUTE]: `${target.kind}:${target.index}`, [LABEL_ATTRIBUTE]: target.label } : {};

export const influenceTargetOf = (element: EventTarget | null): InfluenceTarget | null => {
  if (!(element instanceof Element)) return null;
  const host = element.closest(`[${ATTRIBUTE}]`);
  const [kind, index] = host?.getAttribute(ATTRIBUTE)?.split(':') ?? [];
  if (kind !== 'hsl' && kind !== 'tone' && kind !== 'grading') return null;
  return { kind, index: Number(index), label: host?.getAttribute(LABEL_ATTRIBUTE) ?? '' };
};

export const sameInfluence = (a: InfluenceTarget | null, b: InfluenceTarget | null) =>
  a === b || (!!a && !!b && a.kind === b.kind && a.index === b.index && a.label === b.label);
