@echo off
set OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
if not exist "%OLLAMA%" set OLLAMA=ollama
"%OLLAMA%" list
