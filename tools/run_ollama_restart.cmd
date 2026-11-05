@echo off
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\ollama_restart.py
