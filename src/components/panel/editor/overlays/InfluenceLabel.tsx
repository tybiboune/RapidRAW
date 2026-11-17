import { AnimatePresence, motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { useEditorStore } from '../../../../store/useEditorStore';

// Names what the preview shows while Alt is held on a control, so a mostly black picture reads as
// the affected-areas view rather than a broken render.
export default function InfluenceLabel() {
  const { t } = useTranslation();
  const influence = useEditorStore((state) => state.influence);

  return (
    <AnimatePresence>
      {influence && (
        <motion.div
          key="influence-label"
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.12 }}
          className="pointer-events-none absolute top-3 left-1/2 -translate-x-1/2 z-50 rounded-full bg-black/75 px-3 py-1 text-xs text-white shadow-lg whitespace-nowrap"
        >
          <span className="opacity-70">{t('editor.affectedAreas')}</span>
          {influence.label && <span className="ml-1.5 font-medium">{influence.label}</span>}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
