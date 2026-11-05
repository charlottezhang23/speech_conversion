# 启动 Ollama 服务并拉取默认模型
param([string]$Model = "qwen3:8b")

$ErrorActionPreference = "Continue"
$exe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
if (-not (Test-Path $exe)) {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { $exe = $cmd.Source }
}
if (-not (Test-Path $exe)) {
    Write-Output "OLLAMA_NOT_FOUND"
    exit 1
}
Write-Output "OLLAMA_EXE=$exe"

$listening = $false
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -UseBasicParsing -TimeoutSec 3
    $listening = ($r.StatusCode -eq 200)
} catch { $listening = $false }

if (-not $listening) {
    Start-Process -FilePath $exe -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { $listening = $true; break }
        } catch { }
    }
}
Write-Output "OLLAMA_ONLINE=$listening"
& $exe --version

& $exe pull $Model
Write-Output "PULL_EXIT=$LASTEXITCODE"
& $exe list
