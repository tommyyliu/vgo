$ErrorActionPreference = "Stop"

$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (-not (Test-Path -LiteralPath $chrome)) {
    throw "Chrome was not found at $chrome"
}

function Invoke-BrowserTest([string]$Page) {
    $testPage = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot $Page)).Path.Replace("\", "/")
    $profile = Join-Path $env:TEMP ("vgo-browser-tests-" + [Guid]::NewGuid().ToString("N"))
    $tempRoot = [IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\'
    $profile = [IO.Path]::GetFullPath($profile)
    if (-not $profile.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to use a test profile outside the temp directory: $profile"
    }
    $stdout = Join-Path $profile "stdout.txt"
    $stderr = Join-Path $profile "stderr.txt"
    New-Item -ItemType Directory -Path $profile | Out-Null

    try {
        $process = Start-Process -FilePath $chrome -ArgumentList @(
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--allow-file-access-from-files",
            "--user-data-dir=$profile",
            "--dump-dom",
            "file:///$testPage"
        ) -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $dom = Get-Content -LiteralPath $stdout -Raw
        if ($process.ExitCode -ne 0 -or $dom -notmatch 'data-status="pass"') {
            Write-Error ($Page + " failed:`n" + $dom + "`n" + (Get-Content -LiteralPath $stderr -Raw))
        }
        if ($dom -match 'PASS [0-9]+/[0-9]+') { Write-Output ($Page + ': ' + $Matches[0]) }
    } finally {
        Remove-Item -LiteralPath $profile -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Invoke-BrowserTest "engine-tests.html"
Invoke-BrowserTest "game-tree-tests.html"
Invoke-BrowserTest "ui-tests.html"
