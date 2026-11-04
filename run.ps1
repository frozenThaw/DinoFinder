$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectDir '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython) {
    $python = $venvPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    $standardPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    if ($pythonCommand) {
        $python = $pythonCommand.Source
    } elseif (Test-Path -LiteralPath $standardPython) {
        $python = $standardPython
    } else {
        throw '未找到可用的 Python，请先执行 powershell -ExecutionPolicy Bypass -File .\install.ps1'
    }
}

& $python (Join-Path $projectDir 'similar_finder.py') @args
exit $LASTEXITCODE
