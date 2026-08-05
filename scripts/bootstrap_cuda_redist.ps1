param(
    [string]$DependencyRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) '.deps')
)

$ErrorActionPreference = 'Stop'
Import-Module BitsTransfer
$dependencyRoot = [IO.Path]::GetFullPath($DependencyRoot)
$redistRoot = Join-Path $dependencyRoot 'cuda-redist'
$archiveRoot = Join-Path $redistRoot 'archives'
$extractRoot = Join-Path $redistRoot 'extracted'
$binRoot = Join-Path $redistRoot 'bin'
$licenseRoot = Join-Path $redistRoot 'licenses'
New-Item -ItemType Directory -Force -Path $archiveRoot,$extractRoot,$binRoot,$licenseRoot | Out-Null

$baseUrl = 'https://developer.download.nvidia.com/compute'
$packages = @(
    @{
        Name = 'cuda-cudart-12.8.90'
        Url = "$baseUrl/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.8.90-archive.zip"
        Sha256 = '4a39058fd8519444a81cfc7ae055d136f48d1a31ffa41ae255b35b2edd61e13b'
    },
    @{
        Name = 'libcublas-12.8.4.1'
        Url = "$baseUrl/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.8.4.1-archive.zip"
        Sha256 = '57a470112cec7e112c95253dde8b3c7184d795dbd92b0bde77a4cb7f8c94c8aa'
    },
    @{
        Name = 'libcufft-11.3.3.83'
        Url = "$baseUrl/cuda/redist/libcufft/windows-x86_64/libcufft-windows-x86_64-11.3.3.83-archive.zip"
        Sha256 = 'cc6e0ba958cf23387b462017a24464c72bd901549046133f3d1ebcc3d7444c90'
    },
    @{
        Name = 'cudnn-9.8.0.87-cuda12'
        Url = "$baseUrl/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.8.0.87_cuda12-archive.zip"
        Sha256 = 'd8a23705e3884b137b7e05449fb2b61bfa524e7cfc3fda80743d633f423c6ce4'
    }
)

foreach ($package in $packages) {
    $archivePath = Join-Path $archiveRoot ($package.Name + '.zip')
    $validArchive = (Test-Path -LiteralPath $archivePath) -and
        ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $package.Sha256)
    if (-not $validArchive) {
        Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
        $bitsArguments = @{
            Source = $package.Url
            Destination = $archivePath
            DisplayName = "SimilarFinder-$($package.Name)"
            Description = 'SimilarFinder NVIDIA redistributable'
            Priority = 'Foreground'
            TransferType = 'Download'
        }
        Start-BitsTransfer @bitsArguments
    }
    $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $package.Sha256) {
        throw "SHA-256 mismatch for $($package.Name): $actualHash"
    }

    $packageExtractRoot = Join-Path $extractRoot $package.Name
    if (-not (Test-Path -LiteralPath (Join-Path $packageExtractRoot '.complete'))) {
        if (Test-Path -LiteralPath $packageExtractRoot) {
            Remove-Item -LiteralPath $packageExtractRoot -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path $packageExtractRoot | Out-Null
        Expand-Archive -LiteralPath $archivePath -DestinationPath $packageExtractRoot -Force
        New-Item -ItemType File -Path (Join-Path $packageExtractRoot '.complete') | Out-Null
    }

    Get-ChildItem -LiteralPath $packageExtractRoot -Filter '*.dll' -File -Recurse | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $binRoot -Force
    }
    Get-ChildItem -LiteralPath $packageExtractRoot -File -Recurse | Where-Object {
        $_.Name -match '^(LICENSE|License|license|EULA|NOTICE)'
    } | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $licenseRoot ($package.Name + '-' + $_.Name)) -Force
    }
}

$dlls = Get-ChildItem -LiteralPath $binRoot -Filter '*.dll' -File | Sort-Object Name | ForEach-Object {
    [ordered]@{
        name = $_.Name
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        size = $_.Length
    }
}
$manifest = [ordered]@{
    cuda = '12.8.1'
    cudnn = '9.8.0.87'
    packages = $packages
    dlls = $dlls
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $redistRoot 'manifest.json') -Encoding UTF8

Write-Host "CUDA redistributables are ready: $binRoot"
