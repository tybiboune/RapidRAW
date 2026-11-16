//! DxO PhotoLab as a denoiser.
//!
//! DxO's DeepPRIME denoisers are the best there are, and DxO offers no way to call them from another
//! program. What it does have is a window, so this drives an installed PhotoLab through Windows UI
//! Automation (see `dxo_driver.ps1`): it opens the photo, picks the denoiser and exports a linear DNG next to
//! the photo as `<name>_DxO.dng`. From then on the DNG is what RapiderRaw develops instead of the original
//! raw, so the denoising is under every edit and nothing is denoised twice.
//!
//! Needs the user's own licensed PhotoLab; nothing of DxO's is copied or bypassed.

use std::path::{Path, PathBuf};

use serde::Serialize;

// only the Windows side drives PhotoLab, and only it reads the script
#[cfg(target_os = "windows")]
const DRIVER_SCRIPT: &str = include_str!("dxo_driver.ps1");
const SUFFIX: &str = "_DxO";

/// The denoiser for a photo: the most advanced one that exists for its sensor. DeepPRIME XD3 is for
/// Fujifilm X-Trans sensors only; every other camera is Bayer and gets DeepPRIME XD/XD2s.
pub fn method_for(photo: &Path) -> &'static str {
    let is_fuji = photo
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("raf"));
    if is_fuji {
        "DeepPRIME XD3 X-Trans"
    } else {
        "DeepPRIME XD/XD2s"
    }
}

/// Where the denoised DNG of a photo lives: next to it.
pub fn denoised_path(photo: &Path) -> PathBuf {
    let stem = photo
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    photo.with_file_name(format!("{stem}{SUFFIX}.dng"))
}

/// The denoised DNG of a photo, if there is one that is still good: made after the photo was last changed.
pub fn existing_denoised(photo: &Path) -> Option<PathBuf> {
    let dng = denoised_path(photo);
    let dng_meta = dng.metadata().ok()?;
    if dng_meta.len() == 0 {
        return None;
    }
    let photo_modified = photo.metadata().ok()?.modified().ok()?;
    (dng_meta.modified().ok()? >= photo_modified).then_some(dng)
}

/// PhotoLab's program, newest installed version.
#[cfg(target_os = "windows")]
pub fn find_photolab() -> Option<PathBuf> {
    let roots = [
        std::env::var_os("ProgramFiles"),
        std::env::var_os("ProgramFiles(x86)"),
    ];
    let mut found: Vec<PathBuf> = Vec::new();
    for root in roots.into_iter().flatten() {
        let Ok(entries) = std::fs::read_dir(Path::new(&root).join("DxO")) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            let exe = entry.path().join("DxO.PhotoLab.exe");
            if name.starts_with("dxo photolab") && exe.is_file() {
                found.push(exe);
            }
        }
    }
    // "DxO PhotoLab 8" sorts after "DxO PhotoLab 7" and (as text) before "DxO PhotoLab 10": compare numbers
    found.sort_by_key(|exe| {
        exe.parent()
            .and_then(|dir| dir.file_name())
            .map(|name| {
                name.to_string_lossy()
                    .chars()
                    .filter(char::is_ascii_digit)
                    .collect::<String>()
                    .parse::<u32>()
                    .unwrap_or(0)
            })
            .unwrap_or(0)
    });
    found.pop()
}

#[cfg(not(target_os = "windows"))]
pub fn find_photolab() -> Option<PathBuf> {
    None
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DxoStatus {
    pub available: bool,
    pub photolab_path: Option<String>,
    pub method: Option<&'static str>,
    /// the denoised DNG that is in use for this photo, if any
    pub denoised_path: Option<String>,
}

#[tauri::command]
pub fn get_dxo_status(path: String) -> DxoStatus {
    let source = crate::library::sidecars::parse_virtual_path(&path).0;
    let photolab = find_photolab();
    DxoStatus {
        available: photolab.is_some(),
        photolab_path: photolab.map(|p| p.to_string_lossy().into_owned()),
        method: Some(method_for(&source)),
        denoised_path: existing_denoised(&source).map(|p| p.to_string_lossy().into_owned()),
    }
}

/// Denoises a photo with PhotoLab and returns the path of the DNG. It takes a minute or more: PhotoLab's
/// window is used while it runs.
#[cfg(target_os = "windows")]
#[tauri::command]
pub async fn run_dxo_denoise(path: String) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let source = crate::library::sidecars::parse_virtual_path(&path).0;
        let photolab =
            find_photolab().ok_or_else(|| "DxO PhotoLab is not installed".to_string())?;
        let output = denoised_path(&source);

        // PhotoLab opens the photo given to it, starting up first if it is not running
        std::process::Command::new(&photolab)
            .arg(&source)
            .spawn()
            .map_err(|e| format!("Could not start PhotoLab: {e}"))?;

        let script = std::env::temp_dir().join("rapiderraw_dxo_driver.ps1");
        // a byte-order mark, so that PowerShell 5 reads the script as UTF-8
        let mut text = String::from('\u{feff}');
        text.push_str(DRIVER_SCRIPT);
        std::fs::write(&script, text).map_err(|e| format!("Could not write the driver: {e}"))?;

        let started = std::time::Instant::now();
        let result = std::process::Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
            ])
            .arg(&script)
            .arg("-Photo")
            .arg(&source)
            .arg("-Method")
            .arg(method_for(&source))
            .arg("-Output")
            .arg(&output)
            .output()
            .map_err(|e| format!("Could not run PowerShell: {e}"))?;
        let _ = std::fs::remove_file(&script);

        let stdout = String::from_utf8_lossy(&result.stdout);
        let last = stdout
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        if let Some(reason) = last.strip_prefix("error ") {
            return Err(reason.to_string());
        }
        if !last.starts_with("ok ") || existing_denoised(&source).is_none() {
            return Err(format!(
                "PhotoLab gave no result ({}): {}",
                result.status,
                String::from_utf8_lossy(&result.stderr).trim()
            ));
        }
        log::info!(
            "Denoised '{}' with {} in {:.0}s",
            source.display(),
            method_for(&source),
            started.elapsed().as_secs_f32()
        );
        Ok(output.to_string_lossy().into_owned())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg(not(target_os = "windows"))]
#[tauri::command]
pub async fn run_dxo_denoise(_path: String) -> Result<String, String> {
    Err("DxO PhotoLab denoising is only available on Windows".to_string())
}

/// Goes back to developing the original raw: removes the denoised DNG.
#[tauri::command]
pub fn remove_dxo_denoise(path: String) -> Result<(), String> {
    let source = crate::library::sidecars::parse_virtual_path(&path).0;
    let dng = denoised_path(&source);
    if dng.exists() {
        std::fs::remove_file(&dng).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_denoised_file_sits_next_to_the_photo() {
        assert_eq!(
            denoised_path(Path::new("D:/shoot/DSC_0982.NEF")),
            PathBuf::from("D:/shoot/DSC_0982_DxO.dng")
        );
    }

    #[test]
    fn each_sensor_gets_the_most_advanced_denoiser_it_has() {
        assert_eq!(method_for(Path::new("a.NEF")), "DeepPRIME XD/XD2s");
        assert_eq!(method_for(Path::new("a.cr3")), "DeepPRIME XD/XD2s");
        assert_eq!(method_for(Path::new("a.RAF")), "DeepPRIME XD3 X-Trans");
    }

    #[test]
    fn only_a_denoised_file_made_after_the_photo_counts() {
        let dir = std::env::temp_dir().join("rapiderraw_dxo_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let photo = dir.join("a.nef");
        std::fs::write(&photo, b"raw").unwrap();
        assert!(existing_denoised(&photo).is_none());
        std::fs::write(denoised_path(&photo), b"").unwrap();
        assert!(
            existing_denoised(&photo).is_none(),
            "an empty file is no result"
        );
        std::fs::write(denoised_path(&photo), b"dng").unwrap();
        assert!(existing_denoised(&photo).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
