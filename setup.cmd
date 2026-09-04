@echo off
rem One-time setup: create the venv and install dependencies.
cd /d "%~dp0"
uv venv --python 3.12 .venv || goto :err
uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :err
echo.
echo Done. Run talk.cmd to start the app.
goto :eof
:err
echo Setup failed. Is uv installed and on PATH?
exit /b 1
