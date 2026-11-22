export enum Mask {
  AiDepth = 'ai-depth',
  AiForeground = 'ai-foreground',
  AiSky = 'ai-sky',
  AiSubject = 'ai-subject',
  AiText = 'ai-text',
  AiFace = 'ai-face',
  All = 'all',
  Brush = 'brush',
  Flow = 'flow',
  MatchBrush = 'match-brush',
  Color = 'color',
  Linear = 'linear',
  Luminance = 'luminance',
  QuickEraser = 'quick-eraser',
  Radial = 'radial',
  Clone = 'clone',
  Heal = 'heal',
  Liquify = 'liquify',
  Retouch = 'retouch',
  FaceRestore = 'face-restore',
  Deblur = 'deblur',
  DustRemoval = 'dust-removal',
  WireRemoval = 'wire-removal',
}

// The brushed repairs whose patch the backend works out from the strokes: restoring faces, deblurring,
// sensor dust and power lines. They are drawn and handled like Retouch.
export const REPAIR_TOOL_TYPES: Array<string> = [Mask.FaceRestore, Mask.Deblur, Mask.DustRemoval, Mask.WireRemoval];

export const isRepairTool = (type?: string | null): boolean => !!type && REPAIR_TOOL_TYPES.includes(type);

export enum SubMaskMode {
  Additive = 'additive',
  Subtractive = 'subtractive',
  Intersect = 'intersect',
}

export enum ToolType {
  AiSeletor = 'ai-selector',
  Brush = 'brush',
  Eraser = 'eraser',
  GenerativeReplace = 'generative-replace',
  SelectSubject = 'select-subject',
}

export interface MaskType {
  disabled: boolean;
  icon: any;
  id?: string;
  name: string;
  type: Mask;
}

export interface SubMask {
  id: string;
  invert: boolean;
  mode: SubMaskMode;
  name?: string;
  opacity: number;
  parameters?: any;
  type: Mask;
  visible: boolean;
}
