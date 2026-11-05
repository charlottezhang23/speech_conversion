@echo off
REM 对最近一个已转录项目生成两种纪要（需要 Ollama 正在运行）
cd /d "%~dp0.."
set PATH=%CD%\tools\ffmpeg\bin;%PATH%
".venv\Scripts\python.exe" tools\e2e_summary.py %1 > logs\e2e_summary_console.txt 2>&1
type logs\e2e_summary.txt
