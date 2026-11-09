@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" interview_assistant.py
if errorlevel 1 pause
