@echo off
set LOG=%LOCALAPPDATA%\Ollama\server.log
if exist "%LOG%" (
  powershell -NoProfile -Command "Get-Content -Path '%LOG%' -Tail 40"
) else (
  echo no server log at %LOG%
  dir /b "%LOCALAPPDATA%\Ollama" 2>nul
)
