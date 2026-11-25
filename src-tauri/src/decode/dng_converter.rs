// Nikon "High Efficiency" (HE / HE*, TicoRAW) NEFs cannot be decoded by rawler. Adobe's free DNG Converter
// can, so for those files only: convert once to a lossless DNG in the app cache and develop that instead of
// falling back to the embedded JPEG.
//
// The converter is a Windows / macOS application. If it is missing the user is asked (event
// `dng-converter-needed`) and Adobe's own installer is downloaded and launched, so its licence is accepted by
// the user in Adobe's installer.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime};

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

pub const ADOBE_DNG_PAGE: &str =
    "https://helpx.adobe.com/camera-raw/using/adobe-dng-converter.html";

static CONVERSION_LOCK: Mutex<()> = Mutex::new(());
static PROMPTED: AtomicBool = AtomicBool::new(false);

/// True for the rawler error raised on Nikon HE / HE* NEFs (possibly wrapped by the loader).
pub fn is_unsupported_nef_compression(err: &anyhow::Error) -> bool {
    is_unsupported_nef_compression_text(&err.to_string())
}

pub fn is_unsupported_nef_compression_text(text: &str) -> bool {
    let lowered = text.to_ascii_lowercase();
    lowered.contains("nef compression")
        && lowered.contains("not supported")
        && lowered.contains("highefficen")
}

fn default_converter_locations() -> Vec<PathBuf> {
    let mut found = Vec::new();
    if cfg!(target_os = "windows") {
        for var in ["ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"] {
            if let Some(base) = std::env::var_os(var) {
                found.push(
                    PathBuf::from(base)
                        .join("Adobe")
                        .join("Adobe DNG Converter")
                        .join("Adobe DNG Converter.exe"),
                );
            }
        }
        found.push(PathBuf::from(
            r"C:\Program Files\Adobe\Adobe DNG Converter\Adobe DNG Converter.exe",
        ));
    } else if cfg!(target_os = "macos") {
        let rel = "Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter";
        found.push(PathBuf::from("/Applications").join(rel));
        if let Some(home) = std::env::var_os("HOME") {
            found.push(PathBuf::from(home).join("Applications").join(rel));
        }
    }
    found
}

/// The converter executable: the user's configured path first, then the default install locations.
pub fn find_converter(configured: Option<&str>) -> Option<PathBuf> {
    configured
        .filter(|p| !p.trim().is_empty())
        .map(PathBuf::from)
        .into_iter()
        .chain(default_converter_locations())
        .find(|p| p.is_file())
}

fn cache_dir(app: &AppHandle) -> Result<PathBuf> {
    let dir = app.path().app_cache_dir()?.join("dng_converted");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub(crate) fn cache_key(src: &Path) -> Result<String> {
    let meta = fs::metadata(src)?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    let mut hasher = DefaultHasher::new();
    src.hash(&mut hasher);
    (meta.len(), mtime).hash(&mut hasher);
    Ok(format!("{:016x}", hasher.finish()))
}

/// Arguments of Adobe DNG Converter's command line: lossless compression, no embedded preview.
fn converter_args(out_dir: &Path, out_name: &str, src: &Path) -> Vec<std::ffi::OsString> {
    vec![
        "-c".into(),
        "-p0".into(),
        "-d".into(),
        out_dir.as_os_str().to_owned(),
        "-o".into(),
        out_name.into(),
        src.as_os_str().to_owned(),
    ]
}

/// Converts `src` to a DNG (cached by path, size and modification time) and returns the DNG path.
pub fn convert_to_dng(app: &AppHandle, converter: &Path, src: &Path) -> Result<PathBuf> {
    let dir = cache_dir(app)?;
    let key = cache_key(src)?;
    let dng = dir.join(format!("{key}.dng"));
    if dng.metadata().is_ok_and(|m| m.len() > 0) {
        return Ok(dng);
    }

    let _guard = CONVERSION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if dng.metadata().is_ok_and(|m| m.len() > 0) {
        return Ok(dng);
    }

    let work = dir.join(format!("{key}.work"));
    let _ = fs::remove_dir_all(&work);
    fs::create_dir_all(&work)?;
    let started = Instant::now();
    let output = Command::new(converter)
        .args(converter_args(&work, &format!("{key}.dng"), src))
        .output()
        .with_context(|| format!("Could not run {}", converter.display()))?;
    let produced = work.join(format!("{key}.dng"));
    let ok = produced.metadata().is_ok_and(|m| m.len() > 0);
    if !ok {
        let _ = fs::remove_dir_all(&work);
        bail!(
            "Adobe DNG Converter produced no output for '{}' (status {}): {}",
            src.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    fs::rename(&produced, &dng)?;
    let _ = fs::remove_dir_all(&work);
    log::info!(
        "Converted '{}' to DNG in {:.1}s",
        src.display(),
        started.elapsed().as_secs_f32()
    );
    Ok(dng)
}

/// The DNG an HE NEF has already been converted to, if there is one; converts nothing. For what the NEF
/// keeps in its image data - the camera's as-shot neutral and colour matrices, the size of the picture -
/// which rawler cannot read from an HE NEF but reads from this DNG without decoding a pixel.
pub fn converted_dng(src: &Path) -> Option<PathBuf> {
    let app = crate::render::gpu::context::app_handle()?;
    let dng = app
        .path()
        .app_cache_dir()
        .ok()?
        .join("dng_converted")
        .join(format!("{}.dng", cache_key(src).ok()?));
    dng.metadata().is_ok_and(|m| m.len() > 0).then_some(dng)
}

/// Returns the DNG for an HE NEF, converting if needed. When the converter is not installed, asks the user
/// (once per session) whether to download and install it and returns an error so the caller can fall back.
pub fn dng_for_he_nef(configured: Option<&str>, path: &str) -> Result<PathBuf> {
    let app = crate::render::gpu::context::app_handle()
        .ok_or_else(|| anyhow!("Application is not ready"))?;
    let Some(converter) = find_converter(configured) else {
        if !PROMPTED.swap(true, Ordering::AcqRel) {
            let _ = app.emit("dng-converter-needed", path);
        }
        bail!("Adobe DNG Converter is not installed");
    };
    convert_to_dng(app, &converter, Path::new(path))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DngConverterStatus {
    pub supported_platform: bool,
    pub installed: bool,
    pub path: Option<String>,
    pub download_page: &'static str,
}

#[tauri::command]
pub fn get_dng_converter_status(configured_path: Option<String>) -> DngConverterStatus {
    let path = find_converter(configured_path.as_deref());
    DngConverterStatus {
        supported_platform: cfg!(any(target_os = "windows", target_os = "macos")),
        installed: path.is_some(),
        path: path.map(|p| p.to_string_lossy().into_owned()),
        download_page: ADOBE_DNG_PAGE,
    }
}

/// Picks Adobe's installer link for this platform out of the DNG Converter page.
fn pick_installer_url(page: &str) -> Option<String> {
    let re =
        regex::Regex::new(r#"https?://[A-Za-z0-9./_\-]*DNGConverter[A-Za-z0-9._\-]*\.(?:exe|dmg)"#)
            .ok()?;
    let links: Vec<&str> = re.find_iter(page).map(|m| m.as_str()).collect();
    let ext = if cfg!(target_os = "windows") {
        ".exe"
    } else {
        ".dmg"
    };
    let wants_arm = cfg!(target_arch = "aarch64");
    let is_arm = |l: &str| {
        let l = l.to_ascii_lowercase();
        l.contains("arm") || l.contains("aarch64")
    };
    links
        .iter()
        .copied()
        .filter(|l| l.ends_with(ext))
        // on macOS Adobe ships one universal dmg; on Windows pick the build matching the CPU
        .filter(|l| !cfg!(target_os = "windows") || is_arm(l) == wants_arm)
        .max_by_key(|l| version_of(l))
        .map(str::to_string)
}

fn version_of(url: &str) -> Vec<u32> {
    let file = url.rsplit('/').next().unwrap_or(url);
    file.split(|c: char| !c.is_ascii_digit())
        .filter_map(|p| p.parse().ok())
        .collect()
}

/// Downloads Adobe's installer and launches it; the user finishes the installation in Adobe's own installer.
#[tauri::command]
pub async fn install_dng_converter(app: AppHandle) -> Result<(), String> {
    if !cfg!(any(target_os = "windows", target_os = "macos")) {
        return Err("Adobe DNG Converter is only available for Windows and macOS".to_string());
    }
    run_install(&app)
        .await
        .map_err(|e| e.to_string())
        .inspect_err(|e| {
            let _ = app.emit(
                "dng-converter-download-finish",
                serde_json::json!({ "error": e }),
            );
        })
}

async fn run_install(app: &AppHandle) -> Result<()> {
    const NAME: &str = "Adobe DNG Converter";
    let _ = app.emit("dng-converter-download-start", NAME);
    let page = reqwest::get(ADOBE_DNG_PAGE)
        .await?
        .error_for_status()?
        .text()
        .await?;
    let url = pick_installer_url(&page).ok_or_else(|| {
        anyhow!("No installer link found on {ADOBE_DNG_PAGE}; install Adobe DNG Converter manually")
    })?;
    log::info!("Downloading Adobe DNG Converter from {url}");

    let mut response = reqwest::get(&url).await?.error_for_status()?;
    let total = response.content_length().unwrap_or(0);
    let file_name = url.rsplit('/').next().unwrap_or("DNGConverter_installer");
    let dest = cache_dir(app)?.join(file_name);
    let mut file = fs::File::create(&dest)?;
    let (mut current, mut last) = (0u64, Instant::now());
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk)?;
        current += chunk.len() as u64;
        if total > 0 && last.elapsed().as_millis() >= 200 {
            last = Instant::now();
            let _ = app.emit(
                "dng-converter-download-progress",
                serde_json::json!({ "name": NAME, "current": current, "total": total }),
            );
        }
    }
    file.flush()?;
    drop(file);
    let _ = app.emit(
        "dng-converter-download-finish",
        serde_json::json!({ "error": null }),
    );

    launch_installer(&dest)
}

#[cfg(target_os = "windows")]
fn launch_installer(installer: &Path) -> Result<()> {
    Command::new(installer)
        .spawn()
        .with_context(|| format!("Could not start {}", installer.display()))?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn launch_installer(dmg: &Path) -> Result<()> {
    // opens the disk image in Finder, where Adobe's installer package is started by the user
    Command::new("open")
        .arg(dmg)
        .spawn()
        .with_context(|| format!("Could not open {}", dmg.display()))?;
    Ok(())
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn launch_installer(_installer: &Path) -> Result<()> {
    Err(anyhow!(
        "Adobe DNG Converter is only available for Windows and macOS"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_he_errors() {
        assert!(is_unsupported_nef_compression_text(
            "NEF compression Some(HighEfficency) is not supported"
        ));
        assert!(is_unsupported_nef_compression_text(
            "Unsupported RAW compression format for 'a.NEF'. Original error: Decoder failed: NEF compression Some(HighEfficencyStar) is not supported"
        ));
        assert!(!is_unsupported_nef_compression_text(
            "NEF compression Lossy is not supported"
        ));
        assert!(!is_unsupported_nef_compression_text("Load cancelled"));
    }

    #[test]
    fn picks_newest_installer() {
        let page = r#"<a href="https://download.adobe.com/pub/adobe/dng/win/DNGConverter_17_2.exe">x</a>
            <a href="https://download.adobe.com/pub/adobe/dng/win/DNGConverter_18_1.exe">x</a>
            <a href="https://download.adobe.com/pub/adobe/dng/win/DNGConverter_18_1_ARM64.exe">x</a>
            <a href="https://download.adobe.com/pub/adobe/dng/mac/DNGConverter_18_1.dmg">x</a>"#;
        let url = pick_installer_url(page).unwrap();
        if cfg!(target_os = "windows") && !cfg!(target_arch = "aarch64") {
            assert!(url.ends_with("DNGConverter_18_1.exe"));
        } else if cfg!(target_os = "windows") {
            assert!(url.ends_with("DNGConverter_18_1_ARM64.exe"));
        } else {
            assert!(url.ends_with("DNGConverter_18_1.dmg"));
        }
    }

    #[test]
    fn args_are_lossless_cached_output() {
        let args = converter_args(Path::new("/o"), "k.dng", Path::new("/s/a.NEF"));
        let s: Vec<_> = args
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(s, ["-c", "-p0", "-d", "/o", "-o", "k.dng", "/s/a.NEF"]);
    }
}
