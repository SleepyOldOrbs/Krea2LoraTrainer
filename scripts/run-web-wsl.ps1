param(
    [int] $Port = 8765,
    [string] $HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$resolved = Resolve-Path -LiteralPath $repo
$drive = $resolved.Path.Substring(0, 1).ToLowerInvariant()
$tail = $resolved.Path.Substring(2).Replace("\", "/")
$wslRepo = "/mnt/$drive$tail"

$command = "cd '$wslRepo' && python3 web_app.py --host '$HostAddress' --port $Port"
wsl.exe -e bash -lc $command
