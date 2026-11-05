@echo off
cd /d "%~dp0.."
(
  ".venv\Scripts\python.exe" tools\ps.py
  echo ---
  curl.exe -s -m 5 http://127.0.0.1:11434/api/ps
  echo.
) > logs\ps.txt 2>&1
