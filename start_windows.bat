@echo off
echo ============================================================
echo Smart File AI Chatbot - Windows Quick Start
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check if Ollama is available
curl http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo WARNING: Ollama is not running
    echo Please start Ollama first:
    echo 1. Install from https://ollama.ai
    echo 2. Run 'ollama pull llama2' in a terminal
    echo 3. Ollama will run automatically in background
    echo.
    pause
)

REM Navigate to backend directory
cd backend

REM Check if venv exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

REM Start the server
echo.
echo ============================================================
echo Starting FastAPI Server...
echo ============================================================
echo.
echo Backend will be available at: http://localhost:8000
echo API Docs at: http://localhost:8000/docs
echo.
echo Open frontend in browser: frontend/index.html
echo Or use local server: python -m http.server 8080 --directory ..\frontend
echo.
python main.py

pause
