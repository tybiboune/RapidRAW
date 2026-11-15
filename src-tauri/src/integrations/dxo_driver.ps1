# Drives an installed DxO PhotoLab (8.x) to denoise one photo with its most advanced denoiser and export the
# result as a DNG next to the photo.
#
# DxO offers no command line, so this uses Windows UI Automation to read PhotoLab's window and a real mouse
# click for the one control that ignores automation (the denoiser choice). Run by RapiderRaw; prints one
# line, "ok <path>" or "error <reason>".
param(
    [Parameter(Mandatory = $true)][string]$Photo,
    [Parameter(Mandatory = $true)][string]$Method,
    [Parameter(Mandatory = $true)][string]$Output,
    [int]$TimeoutSec = 900
)
$ErrorActionPreference = 'Stop'
function Fail($reason) { Write-Output "error $reason"; exit 1 }

Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class DxoWin {
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, UIntPtr e);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
    public static void Click(int x, int y) {
        SetCursorPos(x, y);
        mouse_event(2, 0, 0, 0, UIntPtr.Zero);
        mouse_event(4, 0, 0, 0, UIntPtr.Zero);
    }
}
'@

$ae = [System.Windows.Automation.AutomationElement]
$Scope = [System.Windows.Automation.TreeScope]
$Types = [System.Windows.Automation.ControlType]
$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
$selItem = [System.Windows.Automation.SelectionItemPattern]::Pattern
$invoke = [System.Windows.Automation.InvokePattern]::Pattern
$expand = [System.Windows.Automation.ExpandCollapsePattern]::Pattern
$toggle = [System.Windows.Automation.TogglePattern]::Pattern
function Prop($prop, $value) { [System.Windows.Automation.PropertyCondition]::new($prop, $value) }

# PhotoLab is started by RapiderRaw with the photo as its argument; wait for its window.
$proc = $null
for ($i = 0; $i -lt 90 -and -not $proc; $i++) {
    $proc = Get-Process -Name 'DxO.PhotoLab' -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
    if (-not $proc) { Start-Sleep 1 }
}
if (-not $proc) { Fail 'PhotoLab did not open' }
$wasMinimized = [DxoWin]::IsIconic($proc.MainWindowHandle)
if ($wasMinimized) { [void][DxoWin]::ShowWindow($proc.MainWindowHandle, 9) }
[void][DxoWin]::SetForegroundWindow($proc.MainWindowHandle)

function TopWindow {
    $ae::RootElement.FindAll($Scope::Children, (Prop $ae::ProcessIdProperty $proc.Id)) | Select-Object -First 1
}
$top = TopWindow
if (-not $top) { Fail 'PhotoLab window not found' }

# wait until the photo is the open one: the film strip names it
$stem = [IO.Path]::GetFileName($Photo)
$ready = $false
for ($i = 0; $i -lt 120 -and -not $ready; $i++) {
    $texts = $top.FindAll($Scope::Descendants, (Prop $ae::ControlTypeProperty $Types::Text))
    foreach ($t in $texts) { if ($t.Current.Name -like "*$stem") { $ready = $true; break } }
    if (-not $ready) { Start-Sleep 1 }
}
if (-not $ready) { Fail 'PhotoLab did not open the photo' }

# Customize tab, Detail palette
$customize = $top.FindFirst($Scope::Descendants, (Prop $ae::AutomationIdProperty 'CustomizeButton'))
$customize.GetCurrentPattern($selItem).Select()
Start-Sleep 2
$tabs = $top.FindAll($Scope::Descendants, (Prop $ae::ControlTypeProperty $Types::ListItem))
$tabs[2].GetCurrentPattern($selItem).Select()
Start-Sleep 3

# the denoiser: a real click, as automation alone does not make PhotoLab apply it
$label = $top.FindFirst($Scope::Descendants, (Prop $ae::NameProperty $Method))
if (-not $label) { Fail "denoiser '$Method' is not offered by this PhotoLab" }
$radio = $walker.GetParent($label)
if (-not $radio.Current.IsEnabled) { Fail "denoiser '$Method' is not available in this PhotoLab edition" }
$rect = $radio.Current.BoundingRectangle
[DxoWin]::Click([int]($rect.X + 20), [int]($rect.Y + $rect.Height / 2))
Start-Sleep 4
if (-not $radio.GetCurrentPattern($selItem).Current.IsSelected) { Fail 'the denoiser could not be selected' }

# File > Export to disk
$file = $top.FindFirst($Scope::Descendants, (Prop $ae::AutomationIdProperty 'MenuFile'))
$file.GetCurrentPattern($expand).Expand()
Start-Sleep 1
$item = $file.FindAll($Scope::Descendants, (Prop $ae::NameProperty 'Export to disk')) |
    Where-Object { $_.Current.ControlType -eq $Types::MenuItem } | Select-Object -First 1
if (-not $item) { Fail 'export menu not found' }
$item.GetCurrentPattern($invoke).Invoke()
Start-Sleep 3
$dlg = $top.FindFirst($Scope::Children, (Prop $ae::NameProperty 'Export to disk - Options'))
if (-not $dlg) { Fail 'export options did not open' }

# only the DNG output ticked, going to the photo's own folder
$list = $dlg.FindAll($Scope::Descendants, (Prop $ae::ControlTypeProperty $Types::ListItem))
$dngItem = $null
foreach ($li in $list) {
    $name = $li.FindFirst($Scope::Descendants, (Prop $ae::AutomationIdProperty 'NameTextBlock'))
    $box = $li.FindFirst($Scope::Descendants, (Prop $ae::AutomationIdProperty 'OutputSettingCheckBox'))
    if (-not $name -or -not $box) { continue }
    $isDng = ($name.Current.Name -eq 'DNG')
    $tp = $box.GetCurrentPattern($toggle)
    if ($isDng -ne ($tp.Current.ToggleState -eq 'On')) { $tp.Toggle() }
    if ($isDng) { $dngItem = $li }
}
if (-not $dngItem) { Fail 'no DNG output option in PhotoLab' }
$dngItem.GetCurrentPattern($selItem).Select()
Start-Sleep 2
$combos = $dlg.FindAll($Scope::Descendants, (Prop $ae::ControlTypeProperty $Types::ComboBox))
$format = ($combos[0].GetCurrentPattern([System.Windows.Automation.SelectionPattern]::Pattern).Current.GetSelection() | Select-Object -First 1).Current.Name
$where = ($combos[1].GetCurrentPattern([System.Windows.Automation.SelectionPattern]::Pattern).Current.GetSelection() | Select-Object -First 1).Current.Name
if ($format -notmatch 'Dng') { Fail "the DNG output option is set to '$format'" }
if ($where -notmatch 'OriginalFolder') { Fail "the DNG output option is not set to the photo's own folder ('$where')" }

# a leftover from an earlier run would make PhotoLab number the new file instead
Remove-Item -LiteralPath $Output -Force -ErrorAction SilentlyContinue
$button = $dlg.FindAll($Scope::Descendants, (Prop $ae::NameProperty 'Export')) |
    Where-Object { $_.Current.ControlType -eq $Types::Button } | Select-Object -First 1
$button.GetCurrentPattern($invoke).Invoke()

# wait for the file to appear and stop growing
$last = -1
$stable = 0
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    Start-Sleep 2
    $f = Get-Item -LiteralPath $Output -ErrorAction SilentlyContinue
    if ($f -and $f.Length -gt 0) {
        if ($f.Length -eq $last) { $stable++ } else { $stable = 0; $last = $f.Length }
        if ($stable -ge 3) { break }
    }
}
if (-not (Test-Path -LiteralPath $Output) -or $stable -lt 3) { Fail 'PhotoLab did not finish the export in time' }

# put the window back the way it was found
if ($wasMinimized) { [void][DxoWin]::ShowWindow($proc.MainWindowHandle, 6) }
Write-Output "ok $Output"
