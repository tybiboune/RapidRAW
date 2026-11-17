import type React from 'react';

export const FINE_ADJUSTMENT_MULTIPLIER = 0.2;

// Shift only: Alt held on a control shows its affected areas instead (utils/influencePreview.ts)
export const hasFineAdjustmentModifier = (event: MouseEvent | TouchEvent | React.MouseEvent | React.TouchEvent) =>
  'shiftKey' in event && event.shiftKey;
