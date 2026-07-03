param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CliArgs
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$resolved = Resolve-Path -LiteralPath $repo
$drive = $resolved.Path.Substring(0, 1).ToLowerInvariant()
$tail = $resolved.Path.Substring(2).Replace("\", "/")
$wslRepo = "/mnt/$drive$tail"

$quotedArgs = @()
foreach ($arg in $CliArgs) {
    $quotedArgs += "'" + ($arg -replace "'", "'\''") + "'"
}

$command = "cd '$wslRepo' && python3 krea2_lora.py " + ($quotedArgs -join " ")
wsl.exe -e bash -lc $command
exit $LASTEXITCODE
