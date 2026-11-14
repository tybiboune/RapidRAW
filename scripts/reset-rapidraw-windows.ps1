# Resets RapidRAW on this Windows PC so the next launch starts as a fresh install.
# Removes settings, last session, caches, thumbnails, AI models, converted DNGs and the webview state.
# Does NOT touch your photos or the .rrdata / .rrexif edit files stored next to them.
# Run in PowerShell:  powershell -ExecutionPolicy Bypass -File .\reset-rapidraw-windows.ps1

$id = 'io.github.CyberTimon.RapidRAW'

Get-Process -Name 'RapidRAW' -ErrorAction SilentlyContinue | Stop-Process -Force

$paths = @(
    (Join-Path $env:APPDATA $id),        # settings, last session, data, AI models
    (Join-Path $env:LOCALAPPDATA $id)    # caches, thumbnails, converted DNGs, WebView2 state
)
foreach ($p in $paths) {
    if (Test-Path $p) { Remove-Item -LiteralPath $p -Recurse -Force; Write-Host "removed $p" }
    else { Write-Host "not found $p" }
}

# saved Immich API key, if any
cmdkey /delete:"api-key.$id.immich" 2>$null | Out-Null
cmdkey /delete:"$id.immich" 2>$null | Out-Null

Write-Host 'Done. Next launch starts fresh.'
