@echo off
REM 启动本地服务（仅监听 127.0.0.1:7860）
cd /d "%~dp0.."
set PATH=%CD%\tools\ffmpeg\bin;%PATH%
".venv\Scripts\python.exe" -m app.main --no-browser --port 7860
