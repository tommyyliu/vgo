param(
    [string]$Candidates = "",
    [string]$Cases = "",
    [int]$Side = 0,
    [int]$Budget = 0,
    [int]$TimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"

$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (-not (Test-Path -LiteralPath $chrome)) { throw "Chrome was not found at $chrome" }

$query = @()
if ($Candidates) { $query += "candidates=$Candidates" }
if ($Cases)      { $query += "cases=$Cases" }
if ($Side -gt 0) { $query += "side=$Side" }
if ($Budget -gt 0) { $query += "budget=$Budget" }
$suffix = if ($query.Count) { "?" + ($query -join "&") } else { "" }

$page = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "run.html")).Path.Replace("\", "/")
$profile = Join-Path $env:TEMP ("vgo-bench-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $profile | Out-Null
$stdout = Join-Path $profile "stdout.txt"
$stderr = Join-Path $profile "stderr.txt"

try {
    $process = Start-Process -FilePath $chrome -ArgumentList @(
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--allow-file-access-from-files",
        "--user-data-dir=$profile",
        "--dump-dom",
        "--virtual-time-budget=$($TimeoutSeconds * 1000)",
        "file:///$page$suffix"
    ) -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr

    $dom = Get-Content -LiteralPath $stdout -Raw
    if ($dom -match '(?s)<pre id="out"[^>]*>(.*?)</pre>') {
        $report = $Matches[1].Replace("&lt;", "<").Replace("&gt;", ">").Replace("&amp;", "&")
        Write-Output $report
        if ($report -match "HARNESS ERROR") { exit 1 }
    } else {
        Write-Output "no report (exit $($process.ExitCode))"
        Write-Output (Get-Content -LiteralPath $stderr -Raw)
        exit 1
    }
} finally {
    Remove-Item -LiteralPath $profile -Recurse -Force -ErrorAction SilentlyContinue
}
