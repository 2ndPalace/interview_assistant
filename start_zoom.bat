@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" zoom_transcriber.py
if errorlevel 1 pause
