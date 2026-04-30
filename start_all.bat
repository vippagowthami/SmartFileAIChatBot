@echo off
REM Start backend and frontend using the local venv python and run_all.py
SET ROOT=%~dp0
IF EXIST "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" "%ROOT%run_all.py"
) ELSE (
  python "%ROOT%run_all.py"
)
