@echo off
rem Launch the voice chat UI without a console window hanging around.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" talk.py
