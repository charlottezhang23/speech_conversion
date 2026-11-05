#requires -Version 5.1
<#
.SYNOPSIS
    本地访谈转录与会议纪要系统 — 一键安装（Windows 10/11，CPU 推理）

.DESCRIPTION
    1) 检查系统环境（CPU / 内存 / Python / 磁盘）
    2) 安装本地 FFmpeg 到 tools\ffmpeg
    3) 创建 Python 虚拟环境 .venv 并安装依赖（PyTorch CPU + WhisperX + pyannote）
    4) 检测 / 可选安装 Ollama，并拉取默认模型 qwen3:8b（约 5GB）
    5) 打印自检结果与后续步骤

.PARAMETER SkipPython
    跳过 pip 安装（依赖已经装好时使用）

.PARAMETER SkipOllama
    跳过 Ollama 检测与模型下载

.PARAMETER Model
    要拉取的 Ollama 模型，默认 qwen3:8b

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipPython,
    [switch]$SkipOllama,
    [string]$Model = "qwen3:8b",
    [switch]$ForceFfmpeg
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Write-Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Write-Ok($text) { Write-Host "    [OK] $text" -ForegroundColor Green }
function Write-Warn2($text) { Write-Host "    [!] $text" -ForegroundColor Yellow }
function Write-Err($text) { Write-Host "    [X] $text" -ForegroundColor Red }

Write-Host "===============================================" -ForegroundColor DarkGray
Write-Host " 本地访谈转录与会议纪要 — 安装程序" -ForegroundColor White
Write-Host "===============================================" -ForegroundColor DarkGray

# ---------------------------------------------------------------- environment
Write-Step "检查系统环境"
$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
$freeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
Write-Host "    操作系统 : $($os.Caption) $($os.Version)"
Write-Host "    CPU      : $($cpu.Name.Trim())（$($cpu.NumberOfCores) 核 / $($cpu.NumberOfLogicalProcessors) 线程）"
Write-Host "    内存     : $ramGB GB"
Write-Host "    C 盘剩余 : $freeGB GB"

if ($ramGB -lt 16) { Write-Warn2 "内存低于 16GB，large-v3 可能不够用，建议使用 medium 模型。" }
if ($freeGB -lt 15) { Write-Warn2 "磁盘剩余不足 15GB，模型文件可能放不下。" }

$pyCmd = $null
foreach ($cand in @("py", "python")) {
    $cmd = Get-Command $cand -ErrorAction SilentlyContinue
    if ($cmd -and $cand -eq "python" -and $cmd.Source -like "*WindowsApps*") {
        # WindowsApps 别名通常仍可用，继续尝试
    }
    if ($cmd) {
        try {
            if ($cand -eq "py") {
                $v = & py -3 -c "import sys; print(sys.version)" 2>$null
                if ($LASTEXITCODE -eq 0) { $pyCmd = "py -3"; break }
            }
            else {
                $v = & python -c "import sys; print(sys.version)" 2>$null
                if ($LASTEXITCODE -eq 0) { $pyCmd = "python"; break }
            }
        }
        catch { }
    }
}
if (-not $pyCmd) {
    Write-Err "未找到可用的 Python 3。请先从 https://www.python.org/downloads/windows/ 安装 Python 3.11/3.12/3.13（勾选 Add to PATH），然后重新运行本脚本。"
    exit 1
}
$pyVersion = & cmd /c "$pyCmd -c ""import sys; print('.'.join(map(str,sys.version_info[:3])))"""
Write-Ok "Python $pyVersion（$pyCmd）"
if ([version]$pyVersion -lt [version]"3.11") {
    Write-Err "需要 Python 3.11 或更高版本。"
    exit 1
}

foreach ($d in @("projects", "models", "config", "secrets", "logs", "tmp", "tools")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null
}

# --------------------------------------------------------------------- ffmpeg
Write-Step "检查 FFmpeg"
$ffmpegExe = Join-Path $Root "tools\ffmpeg\bin\ffmpeg.exe"
if ($ForceFfmpeg -or -not (Test-Path $ffmpegExe)) {
    Write-Host "    正在下载 FFmpeg（约 90MB）..."
    $zip = Join-Path $Root "tools\ffmpeg.zip"
    # shared build: ships avcodec-*.dll next to the exe (needed by native audio libs)
    $url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"
    try {
        & curl.exe -L --fail --retry 2 -o $zip $url | Out-Null
        if (-not (Test-Path $zip)) { throw "下载失败" }
        Expand-Archive -Path $zip -DestinationPath (Join-Path $Root "tools") -Force
        $dir = Get-ChildItem (Join-Path $Root "tools") -Directory | Where-Object { $_.Name -like "ffmpeg-*" } | Select-Object -First 1
        $target = Join-Path $Root "tools\ffmpeg"
        if (Test-Path $target) { Remove-Item $target -Recurse -Force }
        Rename-Item $dir.FullName "ffmpeg"
        Remove-Item $zip -Force
        Write-Ok "FFmpeg 已安装到 tools\ffmpeg"
    }
    catch {
        Write-Err "FFmpeg 下载失败：$($_.Exception.Message)"
        Write-Warn2 "也可以手动下载 https://www.gyan.dev/ffmpeg/builds/ 并解压到 tools\ffmpeg\bin\ffmpeg.exe"
    }
}
else {
    Write-Ok "已存在：$ffmpegExe"
}

# --------------------------------------------------------------------- venv
Write-Step "准备 Python 虚拟环境"
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "    创建 .venv ..."
    & cmd /c "$pyCmd -m venv .venv"
    if (-not (Test-Path $venvPy)) {
        Write-Err "虚拟环境创建失败。"
        exit 1
    }
    Write-Ok "已创建 .venv"
}
else {
    Write-Ok "复用已有 .venv"
}

if (-not $SkipPython) {
    Write-Step "安装 Python 依赖（首次约 2–4GB 下载）"
    & $venvPy -m pip install --upgrade pip setuptools wheel
    & $venvPy -m pip install -r (Join-Path $Root "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Write-Err "依赖安装失败，请检查网络或查看上方错误信息。"
        exit 1
    }
    Write-Ok "Python 依赖安装完成"
}

# --------------------------------------------------------------------- ollama
if (-not $SkipOllama) {
    Write-Step "检查 Ollama（本地大模型运行时）"
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollama) {
        Write-Warn2 "未检测到 Ollama。纪要功能需要它。"
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            $answer = Read-Host "    是否现在用 winget 安装 Ollama？(y/N)"
            if ($answer -match "^(y|Y)") {
                & winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements --silent
                Start-Sleep -Seconds 3
                $ollama = Get-Command ollama -ErrorAction SilentlyContinue
            }
        }
        if (-not $ollama) {
            Write-Warn2 "请手动从 https://ollama.com/download/windows 安装 Ollama，然后运行：ollama pull $Model"
        }
    }
    if ($ollama) {
        Write-Ok "Ollama 已安装：$($ollama.Source)"
        $listening = Test-NetConnection -ComputerName 127.0.0.1 -Port 11434 -InformationLevel Quiet -WarningAction SilentlyContinue
        if (-not $listening) {
            Write-Host "    启动 ollama serve ..."
            Start-Process -FilePath $ollama.Source -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 4
        }
        Write-Host "    拉取模型 $Model（约 5GB，仅首次需要）..."
        & ollama pull $Model
        if ($LASTEXITCODE -eq 0) { Write-Ok "模型 $Model 已就绪" } else { Write-Warn2 "模型拉取失败，可稍后运行：ollama pull $Model" }
    }
}
else {
    Write-Host "`n    已跳过 Ollama 检查。"
}

# ------------------------------------------------------------------ self check
Write-Step "环境自检"
& $venvPy (Join-Path $Root "tools\verify_env.py")

Write-Host "`n===============================================" -ForegroundColor DarkGray
Write-Host " 安装完成。启动应用：" -ForegroundColor White
Write-Host "   powershell -ExecutionPolicy Bypass -File .\start.ps1" -ForegroundColor Cyan
Write-Host " 首次使用说话人分离前，请在网页「设置」中填入 Hugging Face Token，" -ForegroundColor Gray
Write-Host " 并先在 https://huggingface.co/pyannote/speaker-diarization-community-1 接受条款。" -ForegroundColor Gray
Write-Host "===============================================" -ForegroundColor DarkGray
