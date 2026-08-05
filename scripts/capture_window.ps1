param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class WindowCaptureNative {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [StructLayout(LayoutKind.Sequential)]
    public struct Rect {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out Rect rect);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int maximumCount);

    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();

    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr hWnd, IntPtr deviceContext, uint flags);
}
'@

[WindowCaptureNative]::SetProcessDPIAware() | Out-Null
$windows = [Collections.Generic.List[object]]::new()
$callback = [WindowCaptureNative+EnumWindowsProc]{
    param([IntPtr]$window, [IntPtr]$state)
    $windowProcessId = 0
    [WindowCaptureNative]::GetWindowThreadProcessId($window, [ref]$windowProcessId) | Out-Null
    if ($windowProcessId -eq $ProcessId -and [WindowCaptureNative]::IsWindowVisible($window)) {
        $rect = [WindowCaptureNative+Rect]::new()
        if ([WindowCaptureNative]::GetWindowRect($window, [ref]$rect)) {
            $title = [Text.StringBuilder]::new(512)
            [WindowCaptureNative]::GetWindowText($window, $title, $title.Capacity) | Out-Null
            $windows.Add([PSCustomObject]@{
                Handle = $window
                Title = $title.ToString()
                Left = $rect.Left
                Top = $rect.Top
                Width = $rect.Right - $rect.Left
                Height = $rect.Bottom - $rect.Top
            })
        }
    }
    return $true
}
[WindowCaptureNative]::EnumWindows($callback, [IntPtr]::Zero) | Out-Null

$target = $windows | Where-Object { $_.Width -gt 0 -and $_.Height -gt 0 } |
    Sort-Object { $_.Width * $_.Height } -Descending | Select-Object -First 1
if (-not $target) {
    throw "No visible window found for process $ProcessId."
}

$absoluteOutputPath = [IO.Path]::GetFullPath($OutputPath)
$bitmap = [Drawing.Bitmap]::new($target.Width, $target.Height)
$graphics = [Drawing.Graphics]::FromImage($bitmap)
try {
    $deviceContext = $graphics.GetHdc()
    try {
        $rendered = [WindowCaptureNative]::PrintWindow($target.Handle, $deviceContext, 2)
    } finally {
        $graphics.ReleaseHdc($deviceContext)
    }
    if (-not $rendered) {
        $graphics.CopyFromScreen($target.Left, $target.Top, 0, 0, $bitmap.Size)
    }
    $bitmap.Save($absoluteOutputPath, [Drawing.Imaging.ImageFormat]::Png)
} finally {
    $graphics.Dispose()
    $bitmap.Dispose()
}

$windows | Select-Object Title,Left,Top,Width,Height
Write-Host "Screenshot: $absoluteOutputPath"
