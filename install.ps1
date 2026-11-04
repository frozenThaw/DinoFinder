$ErrorActionPreference = 'Stop'

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw '未在 PATH 中找到 Python。请确认 Python 3.12.5 已加入 PATH 后重试。'
}

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $projectDir '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host '创建 Python 虚拟环境...'
    & $pythonCommand.Source -m venv $venvDir
}

Write-Host '安装 DINOv2 / FAISS 运行依赖...'
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $projectDir 'requirements.txt')

Write-Host ''
Write-Host '安装完成。运行 .\run.ps1 开始检索。'
