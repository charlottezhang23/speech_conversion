#requires -Version 5.1
<#
.SYNOPSIS
    启动本地访谈转录与会议纪要系统（默认 http://127.0.0.1:7860）

.DESCRIPTION
    - 检查 .venv 与 FFmpeg
    - 如果 Ollama 已安装但未运行，会在后台启动 ollama serve
    - 启动 FastAPI 服务并自动打开浏览器

.PARAMETER Port
    监听端口，默认 7860

.PARAMETER NoBrowser
    不自动打开浏览器

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\start.ps1
#>
[CmdletBinding()]
param(
    [int]$Port = 7860,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "未找到 .venv，请先运行：powershell -ExecutionPolicy Bypass -File .\install.ps1" -ForegroundColor Red
    exit 1
}

$ffmpeg = Join-Path $Root "tools\ffmpeg\bin"
if (Test-Path (Join-Path $ffmpeg "ffmpeg.exe")) {
    $env:PATH = "$ffmpeg;$env:PATH"
}
else {
    Write-Host "警告：tools\ffmpeg 不存在，音频转码将依赖系统 PATH 中的 ffmpeg。" -ForegroundColor Yellow
}

# 如果 Ollama 已安装但没在运行，后台启动（纪要功能需要）
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) {
    $listening = Test-NetConnection -ComputerName 127.0.0.1 -Port 11434 -InformationLevel Quiet -WarningAction SilentlyContinue
    if (-not $listening) {
        Write-Host "启动 Ollama 服务 ..." -ForegroundColor DarkGray
        Start-Process -FilePath $ollama.Source -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3
    }
}

$args = @("-m", "app.main", "--port", "$Port")
if ($NoBrowser) { $args += "--no-browser" }

Write-Host "启动中：http://127.0.0.1:$Port（按 Ctrl+C 停止）" -ForegroundColor Cyan
& $venvPy @args
