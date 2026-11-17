import { useEffect } from 'react';
import { useEditorStore } from '../store/useEditorStore';
import { InfluenceTarget, influenceTargetOf, sameInfluence } from '../utils/influencePreview';

// Shows the affected areas of the control under the pointer while Alt is held (#2059). A control
// pressed with the pointer keeps the view until it is released, so a drag that strays off the
// slider or wheel goes on showing it.
export function useInfluencePreview(enabled: boolean) {
  useEffect(() => {
    const show = (target: InfluenceTarget | null) => {
      const { influence, setEditor } = useEditorStore.getState();
      if (!sameInfluence(influence, target)) setEditor({ influence: target });
    };
    if (!enabled) {
      show(null);
      return;
    }

    let altHeld = false;
    let hovered: InfluenceTarget | null = null;
    let pressed: InfluenceTarget | null = null;
    const update = () => show(altHeld ? (pressed ?? hovered) : null);

    const onKey = (event: KeyboardEvent) => {
      const held = event.key === 'Alt' ? event.type === 'keydown' : event.altKey;
      // a lone Alt would otherwise move the focus to the window menu on release
      if (event.key === 'Alt' && (pressed ?? hovered)) event.preventDefault();
      if (held === altHeld) return;
      altHeld = held;
      update();
    };
    const onPointerMove = (event: PointerEvent) => {
      hovered = influenceTargetOf(event.target);
      altHeld = event.altKey;
      update();
    };
    const onPointerDown = (event: PointerEvent) => {
      pressed = influenceTargetOf(event.target);
      onPointerMove(event);
    };
    const onPointerUp = (event: PointerEvent) => {
      pressed = null;
      onPointerMove(event);
    };
    const onBlur = () => {
      altHeld = false;
      pressed = null;
      update();
    };

    window.addEventListener('keydown', onKey, true);
    window.addEventListener('keyup', onKey, true);
    window.addEventListener('pointermove', onPointerMove, true);
    window.addEventListener('pointerdown', onPointerDown, true);
    window.addEventListener('pointerup', onPointerUp, true);
    window.addEventListener('pointercancel', onPointerUp, true);
    window.addEventListener('blur', onBlur);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('keyup', onKey, true);
      window.removeEventListener('pointermove', onPointerMove, true);
      window.removeEventListener('pointerdown', onPointerDown, true);
      window.removeEventListener('pointerup', onPointerUp, true);
      window.removeEventListener('pointercancel', onPointerUp, true);
      window.removeEventListener('blur', onBlur);
      show(null);
    };
  }, [enabled]);
}
