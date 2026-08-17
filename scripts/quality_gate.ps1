$ErrorActionPreference = 'Stop'

$python = 'python'
if (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
    $python = '.venv\Scripts\python.exe'
}

& $python -m ruff check .
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -m pytest -q
exit $LASTEXITCODE
