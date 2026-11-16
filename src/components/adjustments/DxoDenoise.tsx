import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'react-toastify';
import { Loader2 } from 'lucide-react';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextVariants } from '../../types/typography';
import { Invokes } from '../../types/invokes';
import { useUIStore } from '../../store/useUIStore';
import { useEditorStore } from '../../store/useEditorStore';

interface DxoStatus {
  available: boolean;
  photolabPath: string | null;
  method: string | null;
  denoisedPath: string | null;
}

/**
 * Denoises the open photo with DxO PhotoLab's most advanced denoiser and develops it from the result from
 * then on. Only shown when PhotoLab is installed. PhotoLab's window is used while it works, which takes a
 * minute or more.
 */
export default function DxoDenoise() {
  const { t } = useTranslation();
  const path = useEditorStore((state) => state.selectedImage?.path);
  const [status, setStatus] = useState<DxoStatus | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const refresh = useCallback(async () => {
    if (!path) return;
    try {
      setStatus(await invoke<DxoStatus>(Invokes.GetDxoStatus, { path }));
    } catch (error) {
      console.warn('Could not read the DxO status:', error);
      setStatus(null);
    }
  }, [path]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // the photo is developed from a different file now, so it is opened again
  const reopen = () => {
    if (path) useUIStore.getState().imageSelectHandler?.(path, true, true);
  };

  const run = async () => {
    if (!path || isRunning) return;
    setIsRunning(true);
    const toastId = toast.loading(t('adjustments.details.dxoRunning'));
    try {
      await invoke(Invokes.RunDxoDenoise, { path });
      toast.update(toastId, {
        render: t('adjustments.details.dxoDone'),
        type: 'success',
        isLoading: false,
        autoClose: 4000,
      });
      await refresh();
      reopen();
    } catch (error) {
      toast.update(toastId, {
        render: t('adjustments.details.dxoFailed', { error: String(error) }),
        type: 'error',
        isLoading: false,
        autoClose: 8000,
      });
    } finally {
      setIsRunning(false);
    }
  };

  const remove = async () => {
    if (!path) return;
    try {
      await invoke(Invokes.RemoveDxoDenoise, { path });
      await refresh();
      reopen();
    } catch (error) {
      toast.error(String(error));
    }
  };

  if (!status?.available) return null;

  return (
    <div className="mt-2 flex flex-col gap-2">
      <Text variant={TextVariants.label}>{t('adjustments.details.dxoDenoise')}</Text>
      <Text variant={TextVariants.small} className="text-text-secondary">
        {status.denoisedPath
          ? t('adjustments.details.dxoInUse', { method: status.method })
          : t('adjustments.details.dxoHint', { method: status.method })}
      </Text>
      {isRunning ? (
        <div className="flex items-center gap-2 text-text-secondary">
          <Loader2 size={16} className="animate-spin shrink-0" />
          <Text variant={TextVariants.small}>{t('adjustments.details.dxoRunning')}</Text>
        </div>
      ) : (
        <div className="flex gap-2">
          <Button onClick={() => void run()}>
            {status.denoisedPath ? t('adjustments.details.dxoRedo') : t('adjustments.details.dxoRun')}
          </Button>
          {status.denoisedPath && <Button onClick={() => void remove()}>{t('adjustments.details.dxoRemove')}</Button>}
        </div>
      )}
    </div>
  );
}
