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
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
    [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool attach);
    [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);
    // Windows does not let a background process take the foreground, which is what happens when PhotoLab was
    // already running: attaching to the foreground window's input and flipping the window to topmost and back
    // gets it forward all the same.
    public static void BringToFront(IntPtr h) {
        if (IsIconic(h)) ShowWindow(h, 9);
        uint pid;
        uint foreground = GetWindowThreadProcessId(GetForegroundWindow(), out pid);
        uint me = GetCurrentThreadId();
        if (foreground != me) AttachThreadInput(me, foreground, true);
        SetWindowPos(h, new IntPtr(-1), 0, 0, 0, 0, 0x0003);
        SetWindowPos(h, new IntPtr(-2), 0, 0, 0, 0, 0x0003);
        SetForegroundWindow(h);
        if (foreground != me) AttachThreadInput(me, foreground, false);
    }
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

# PhotoLab is started by RapiderRaw with the photo as its argument. It shows a splash window first and its
# real window some time later, so the real window is looked for until it is there and names the photo in
# its film strip: it is the one with the Customize button.
$stem = [IO.Path]::GetFileName($Photo)
$proc = $null
$top = $null
$sawProcess = $false
$wasMinimized = $false
$deadline = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $deadline -and -not $top) {
    foreach ($p in @(Get-Process -Name 'DxO.PhotoLab*' -ErrorAction SilentlyContinue)) {
        $sawProcess = $true
        # a minimized window does not show its contents to automation
        if ($p.MainWindowHandle -ne 0 -and [DxoWin]::IsIconic($p.MainWindowHandle)) {
            $wasMinimized = $true
            [void][DxoWin]::ShowWindow($p.MainWindowHandle, 9)
        }
        foreach ($w in @($ae::RootElement.FindAll($Scope::Children, (Prop $ae::ProcessIdProperty $p.Id)))) {
            try {
                if (-not $w.FindFirst($Scope::Descendants, (Prop $ae::AutomationIdProperty 'CustomizeButton'))) { continue }
                foreach ($t in $w.FindAll($Scope::Descendants, (Prop $ae::ControlTypeProperty $Types::Text))) {
                    if ($t.Current.Name -like "*$stem") { $top = $w; $proc = $p; break }
                }
            } catch { }
            if ($top) { break }
        }
        if ($top) { break }
    }
    if (-not $top) { Start-Sleep 2 }
}
if (-not $top) {
    if ($sawProcess) { Fail 'PhotoLab did not open the photo' } else { Fail 'PhotoLab did not open' }
}
$hwnd = [IntPtr]$top.Current.NativeWindowHandle
[DxoWin]::BringToFront($hwnd)

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

# The click only counts if PhotoLab's window is what is under the pointer, so it is brought forward and
# checked first; a click that did not take is repeated.
function ClickRadio {
    $rect = $radio.Current.BoundingRectangle
    $x = [int]($rect.X + 20)
    $y = [int]($rect.Y + $rect.Height / 2)
    for ($try = 0; $try -lt 4; $try++) {
        [DxoWin]::BringToFront($hwnd)
        Start-Sleep 1
        $under = $ae::FromPoint([System.Windows.Point]::new($x, $y))
        if ($under -and $under.Current.ProcessId -eq $proc.Id) {
            [DxoWin]::Click($x, $y)
            return $true
        }
    }
    return $false
}
$selected = $radio.GetCurrentPattern($selItem).Current.IsSelected
for ($attempt = 0; $attempt -lt 3 -and -not $selected; $attempt++) {
    if (-not (ClickRadio)) { Fail 'PhotoLab is covered by another window that could not be moved aside' }
    Start-Sleep 4
    $selected = $radio.GetCurrentPattern($selItem).Current.IsSelected
}
if (-not $selected) { Fail 'the denoiser could not be selected' }

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
