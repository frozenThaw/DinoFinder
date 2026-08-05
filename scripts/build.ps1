param(
    [string]$DependencyRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) '.deps'),
    [switch]$Clean,
    [switch]$CpuOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentFile = Join-Path ([IO.Path]::GetFullPath($DependencyRoot)) 'deps-env.ps1'
if (-not (Test-Path -LiteralPath $environmentFile)) {
    throw 'Dependencies are not configured. Run scripts\bootstrap_dependencies.ps1 first.'
}
. $environmentFile
$projectVcpkgRoot = $env:VCPKG_ROOT
$projectQtRoot = $env:QT_ROOT
$projectOnnxRuntimeRoot = $env:ONNXRUNTIME_ROOT

$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    throw 'Visual Studio with the Desktop C++ workload is required.'
}
$devCmd = Join-Path $vsPath 'Common7\Tools\VsDevCmd.bat'
$developerEnvironment = & cmd.exe /d /s /c "`"$devCmd`" -arch=x64 -host_arch=x64 >nul && set"
if ($LASTEXITCODE -ne 0) {
    throw "Visual Studio developer environment failed with exit code $LASTEXITCODE"
}
$developerPathEntry = $developerEnvironment | Where-Object { $_ -cmatch '^PATH=' } | Select-Object -First 1
foreach ($entry in $developerEnvironment) {
    if ($entry -imatch '^PATH=') {
        continue
    }
    if ($entry -match '^([^=]+)=(.*)$') {
        Set-Item -LiteralPath "Env:$($matches[1])" -Value $matches[2]
    }
}
if (-not $developerPathEntry) {
    throw 'Visual Studio developer environment did not provide PATH.'
}
[Environment]::SetEnvironmentVariable('PATH', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $developerPathEntry.Substring(5), 'Process')

# Visual Studio's developer shell sets VCPKG_ROOT to its bundled checkout.
# Restore the dependency roots provisioned for this project before configuring.
$env:VCPKG_ROOT = $projectVcpkgRoot
$env:QT_ROOT = $projectQtRoot
$env:ONNXRUNTIME_ROOT = $projectOnnxRuntimeRoot
$env:Path = (Join-Path $env:QT_ROOT 'bin') + [IO.Path]::PathSeparator + $env:Path

$cmake = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ctest = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'
$env:NINJA_EXE = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$env:CMAKE_TOOLCHAIN_FILE = Join-Path $env:VCPKG_ROOT 'scripts\buildsystems\vcpkg.cmake'
$env:CXX = Join-Path $env:VCToolsInstallDir 'bin\Hostx64\x64\cl.exe'

$configurePreset = 'windows-msvc'
$buildPreset = 'windows-release'
$testPreset = 'windows-release'
if ($CpuOnly) {
    $configurePreset = 'windows-msvc-cpu'
    $buildPreset = 'windows-release-cpu'
    $testPreset = 'windows-release-cpu'
}
$buildDirectory = Join-Path $projectRoot ("build\" + $configurePreset)
if ($Clean -and (Test-Path -LiteralPath $buildDirectory)) {
    Remove-Item -LiteralPath $buildDirectory -Recurse -Force
}
& $cmake --preset $configurePreset --fresh
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE" }
& $cmake --build --preset $buildPreset --parallel
if ($LASTEXITCODE -ne 0) { throw "CMake build failed with exit code $LASTEXITCODE" }
& $ctest --preset $testPreset
if ($LASTEXITCODE -ne 0) { throw "CTest failed with exit code $LASTEXITCODE" }
