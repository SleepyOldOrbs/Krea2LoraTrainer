$ErrorActionPreference = "Stop"

$script = Join-Path $PSScriptRoot "run-in-wsl.ps1"
& $script wizard @args
exit $LASTEXITCODE
