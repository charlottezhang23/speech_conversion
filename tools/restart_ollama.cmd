@echo off
REM 重启本地 Ollama 服务（清理卡住的推理进程）
taskkill /F /IM ollama.exe /T >nul 2>&1
taskkill /F /IM llama-server.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul
set OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
start "" "%OLLAMA%" serve
timeout /t 5 /nobreak >nul
curl.exe -s http://127.0.0.1:11434/api/version
echo.
