@echo off
REM ============================================
REM AEMER Backend Startup Script
REM ============================================

echo.
echo ================================================
echo   AEMER Backend Setup
echo ================================================
echo.

REM Check if Python is installed using py launcher (Windows default)
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Navigate to backend directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    py -m venv venv
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo [INFO] Installing dependencies (this may take a few minutes first time)...
pip install -r requirements.txt

REM Check if model exists
if exist "..\AudioModel\best_model.pth" (
    echo.
    echo [OK] Model found: AudioModel\best_model.pth
) else (
    echo.
    echo [WARNING] Model NOT FOUND at AudioModel\best_model.pth
    echo Please ensure your trained model is in the correct location.
)

echo.
echo ================================================
echo   Starting AEMER Backend Server...
echo ================================================
echo.
echo API will be available at: http://localhost:8000
echo API docs at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Start the server
python main.py
