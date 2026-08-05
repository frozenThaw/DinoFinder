param(
    [string]$DependencyRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) '.deps'),
    [string]$CudaRedistRoot = $env:CUDA_REDIST_ROOT
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dependencyRoot = [IO.Path]::GetFullPath($DependencyRoot)
. (Join-Path $dependencyRoot 'deps-env.ps1')
if (-not $CudaRedistRoot) {
    $CudaRedistRoot = Join-Path $dependencyRoot 'cuda-redist\bin'
}
if (-not (Test-Path -LiteralPath $CudaRedistRoot)) {
    throw 'CUDA redistributables are missing. Run scripts\bootstrap_cuda_redist.ps1 first or set CUDA_REDIST_ROOT.'
}
& (Join-Path $PSScriptRoot 'build.ps1') -DependencyRoot $dependencyRoot

$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    throw 'Visual Studio with the Desktop C++ workload is required.'
}
$cmake = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'

$stage = Join-Path $projectRoot 'dist\SimilarFinder'
if (Test-Path -LiteralPath $stage) {
    Remove-Item -LiteralPath $stage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stage | Out-Null
& $cmake --install (Join-Path $projectRoot 'build\windows-msvc') --prefix $stage --config Release
if ($LASTEXITCODE -ne 0) { throw "CMake install failed with exit code $LASTEXITCODE" }
& (Join-Path $env:QT_ROOT 'bin\windeployqt.exe') --release --no-translations --no-compiler-runtime (Join-Path $stage 'SimilarFinder.exe')
if ($LASTEXITCODE -ne 0) { throw "windeployqt failed with exit code $LASTEXITCODE" }

$msvcRedistRoot = Join-Path $vsPath 'VC\Redist\MSVC'
$msvcVersion = Get-ChildItem -LiteralPath $msvcRedistRoot -Directory |
    Where-Object Name -Match '^\d+\.' |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (-not $msvcVersion) {
    throw 'Cannot locate the MSVC x64 redistributable directory.'
}
$msvcCrtDirectory = Get-ChildItem -LiteralPath (Join-Path $msvcVersion.FullName 'x64') -Directory -Filter 'Microsoft.VC*.CRT' |
    Select-Object -First 1
if (-not $msvcCrtDirectory) {
    throw 'Cannot locate the MSVC x64 CRT DLLs.'
}
Copy-Item -Path (Join-Path $msvcCrtDirectory.FullName '*.dll') -Destination $stage -Force

$windowsSdkBin = Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\bin' -Directory -ErrorAction SilentlyContinue |
    Where-Object Name -Match '^\d+\.' |
    Sort-Object Name -Descending |
    Select-Object -First 1
if ($windowsSdkBin) {
    foreach ($directXCompilerDll in 'dxcompiler.dll','dxil.dll') {
        $directXCompilerPath = Join-Path $windowsSdkBin.FullName "x64\$directXCompilerDll"
        if (Test-Path -LiteralPath $directXCompilerPath) {
            Copy-Item -LiteralPath $directXCompilerPath -Destination $stage -Force
        }
    }
}

Copy-Item -Path (Join-Path $env:ONNXRUNTIME_ROOT 'lib\*.dll') -Destination $stage -Force
Copy-Item -Path (Join-Path $projectRoot 'vcpkg_installed\x64-windows\bin\*.dll') -Destination $stage -Force

Get-ChildItem -LiteralPath $CudaRedistRoot -Filter '*.dll' -File -Recurse | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $stage -Force
}

$licenseDirectory = Join-Path $stage 'licenses'
New-Item -ItemType Directory -Force -Path $licenseDirectory | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'THIRD_PARTY_NOTICES.txt') -Destination $licenseDirectory
Copy-Item -Path (Join-Path $projectRoot 'third_party_licenses\*') -Destination $licenseDirectory -Force
Copy-Item -LiteralPath (Join-Path $env:ONNXRUNTIME_ROOT 'LICENSE') -Destination (Join-Path $licenseDirectory 'ONNXRuntime-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $env:ONNXRUNTIME_ROOT 'ThirdPartyNotices.txt') -Destination (Join-Path $licenseDirectory 'ONNXRuntime-ThirdPartyNotices.txt')
$vcpkgShareRoot = Join-Path $projectRoot 'vcpkg_installed\x64-windows\share'
Get-ChildItem -LiteralPath $vcpkgShareRoot -Directory | ForEach-Object {
    $copyright = Join-Path $_.FullName 'copyright'
    if (Test-Path -LiteralPath $copyright) {
        Copy-Item -LiteralPath $copyright -Destination (Join-Path $licenseDirectory ($_.Name + '-copyright.txt'))
    }
}
$cudaLicenseRoot = Join-Path (Split-Path -Parent $CudaRedistRoot) 'licenses'
if (Test-Path -LiteralPath $cudaLicenseRoot) {
    Copy-Item -Path (Join-Path $cudaLicenseRoot '*') -Destination $licenseDirectory -Force
}
$cudaManifest = Join-Path (Split-Path -Parent $CudaRedistRoot) 'manifest.json'
if (Test-Path -LiteralPath $cudaManifest) {
    Copy-Item -LiteralPath $cudaManifest -Destination (Join-Path $licenseDirectory 'NVIDIA-redist-manifest.json')
}

$checksums = Get-ChildItem -LiteralPath $stage -File -Recurse | ForEach-Object {
    $relative = $_.FullName.Substring($stage.Length).TrimStart('\').Replace('\','/')
    $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $relative"
}
$checksums | Set-Content -LiteralPath (Join-Path $stage 'SHA256SUMS.txt') -Encoding ASCII

$archive = Join-Path $projectRoot 'dist\SimilarFinder-windows-x64.zip'
if (Test-Path -LiteralPath $archive) {
    Remove-Item -LiteralPath $archive -Force
}
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Portable package: $archive"
