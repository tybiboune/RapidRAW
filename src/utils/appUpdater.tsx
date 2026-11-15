import { create } from 'zustand';
import i18n from 'i18next';
import { toast, type Id } from 'react-toastify';
import { check, type Update } from '@tauri-apps/plugin-updater';
import { relaunch } from '@tauri-apps/plugin-process';
import { type as osType } from '@tauri-apps/plugin-os';

const FIRST_CHECK_DELAY_MS = 8_000;
const CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000;
const UPDATE_TOAST_ID = 'app-update';
const PROGRESS_TOAST_ID = 'app-update-progress';

interface UpdaterState {
  update: Update | null;
  isInstalling: boolean;
  setUpdate(update: Update | null): void;
  setInstalling(isInstalling: boolean): void;
}

export const useUpdaterStore = create<UpdaterState>((set) => ({
  update: null,
  isInstalling: false,
  setUpdate: (update) => set({ update }),
  setInstalling: (isInstalling) => set({ isInstalling }),
}));

/**
 * Asks the update server whether a newer signed version exists. A failure (offline, no release yet, a
 * server that is down) is not an error worth showing: the app simply carries on and asks again later.
 */
async function findUpdate(): Promise<Update | null> {
  try {
    return await check();
  } catch (error) {
    console.warn('Update check failed:', error);
    return null;
  }
}

/** Downloads the update, installs it and restarts the app. */
export async function installUpdate(update: Update) {
  const store = useUpdaterStore.getState();
  if (store.isInstalling) return;
  store.setInstalling(true);
  toast.dismiss(UPDATE_TOAST_ID);

  let total = 0;
  let received = 0;
  let progressToast: Id | undefined;
  const showProgress = (message: string, progress?: number) => {
    const options = { toastId: PROGRESS_TOAST_ID, autoClose: false as const, closeOnClick: false, progress };
    if (progressToast === undefined) {
      progressToast = toast.info(message, options);
    } else {
      toast.update(progressToast, { render: message, ...options });
    }
  };

  try {
    showProgress(i18n.t('library.splash.updateDownloading', { percent: 0 }), 0);
    await update.downloadAndInstall((event) => {
      if (event.event === 'Started') {
        total = event.data.contentLength ?? 0;
      } else if (event.event === 'Progress') {
        received += event.data.chunkLength;
        if (total > 0) {
          const fraction = Math.min(received / total, 1);
          showProgress(i18n.t('library.splash.updateDownloading', { percent: Math.round(fraction * 100) }), fraction);
        }
      } else if (event.event === 'Finished') {
        showProgress(i18n.t('library.splash.updateInstalling'));
      }
    });
    // On Windows the installer takes over and the app exits by itself; elsewhere it restarts here.
    if (osType() !== 'windows') {
      await relaunch();
    }
  } catch (error) {
    toast.dismiss(PROGRESS_TOAST_ID);
    toast.error(i18n.t('library.splash.updateFailed', { error: String(error) }));
    store.setInstalling(false);
  }
}

function announce(update: Update) {
  toast.info(
    ({ closeToast }) => (
      <div className="flex flex-col gap-2">
        <span>{i18n.t('library.splash.updateAvailable', { version: update.version })}</span>
        <div className="flex gap-2">
          <button
            type="button"
            className="px-3 py-1 rounded-md bg-accent text-button-text text-sm"
            onClick={() => {
              closeToast?.();
              void installUpdate(update);
            }}
          >
            {i18n.t('library.splash.updateInstall')}
          </button>
          <button
            type="button"
            className="px-3 py-1 rounded-md bg-card-active text-text-primary text-sm"
            onClick={() => closeToast?.()}
          >
            {i18n.t('library.splash.updateLater')}
          </button>
        </div>
      </div>
    ),
    { toastId: UPDATE_TOAST_ID, autoClose: false, closeOnClick: false },
  );
}

/**
 * Looks for an update shortly after the app opens and again every few hours. The first time a given
 * version turns up it is announced once; after that it stays available from the version label on the
 * home screen, so a version that was put off is not announced again and again.
 */
export function startUpdateChecks(): () => void {
  let announced: string | null = null;

  const run = async () => {
    if (useUpdaterStore.getState().isInstalling) return;
    const update = await findUpdate();
    useUpdaterStore.getState().setUpdate(update);
    if (update && update.version !== announced) {
      announced = update.version;
      announce(update);
    }
  };

  const first = window.setTimeout(run, FIRST_CHECK_DELAY_MS);
  const repeat = window.setInterval(run, CHECK_INTERVAL_MS);
  return () => {
    window.clearTimeout(first);
    window.clearInterval(repeat);
  };
}
