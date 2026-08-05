param(
    [string]$DependencyRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) '.deps'),
    [string]$Python = 'python',
    [switch]$SkipQt
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dependencyRoot = [IO.Path]::GetFullPath($DependencyRoot)
New-Item -ItemType Directory -Force -Path $dependencyRoot | Out-Null

$vcpkgRoot = Join-Path $dependencyRoot 'vcpkg'
if (-not (Test-Path -LiteralPath (Join-Path $vcpkgRoot 'vcpkg.exe'))) {
    if (-not (Test-Path -LiteralPath $vcpkgRoot)) {
        git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
    }
    & (Join-Path $vcpkgRoot 'bootstrap-vcpkg.bat') -disableMetrics
}
$env:VCPKG_ROOT = $vcpkgRoot
& (Join-Path $vcpkgRoot 'vcpkg.exe') install --triplet x64-windows --x-manifest-root=$projectRoot

$qtRoot = Join-Path $dependencyRoot 'Qt\6.8.3\msvc2022_64'
if (-not $SkipQt -and -not (Test-Path -LiteralPath (Join-Path $qtRoot 'bin\windeployqt.exe'))) {
    & $Python -m pip install 'aqtinstall>=3.2,<4'
    & $Python -m aqt install-qt windows desktop 6.8.3 win64_msvc2022_64 -m qtimageformats --outputdir (Join-Path $dependencyRoot 'Qt')
} elseif (-not $SkipQt -and -not (Test-Path -LiteralPath (Join-Path $qtRoot 'plugins\imageformats\qwebp.dll'))) {
    & $Python -m aqt install-qt windows desktop 6.8.3 win64_msvc2022_64 -m qtimageformats --outputdir (Join-Path $dependencyRoot 'Qt')
}

$ortRoot = Join-Path $dependencyRoot 'onnxruntime-win-x64-gpu-1.22.0'
if (-not (Test-Path -LiteralPath (Join-Path $ortRoot 'lib\onnxruntime.lib'))) {
    $archive = Join-Path $dependencyRoot 'onnxruntime-gpu.zip'
    Invoke-WebRequest -Uri 'https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-gpu-1.22.0.zip' -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $dependencyRoot -Force
    Remove-Item -LiteralPath $archive -Force
}

$environmentFile = Join-Path $dependencyRoot 'deps-env.ps1'
@"
`$env:VCPKG_ROOT = '$vcpkgRoot'
`$env:QT_ROOT = '$qtRoot'
`$env:ONNXRUNTIME_ROOT = '$ortRoot'
"@ | Set-Content -LiteralPath $environmentFile -Encoding UTF8

Write-Host "Dependencies are ready. Environment: $environmentFile"
Write-Host 'CUDA and cuDNN redistributable DLLs must be supplied through CUDA_REDIST_ROOT when packaging.'
