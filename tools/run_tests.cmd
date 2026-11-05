@echo off
REM 运行自动化测试并把结果写入 logs\pytest.txt
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m pytest tests -q > logs\pytest.txt 2>&1
type logs\pytest.txt
exit /b %errorlevel%
