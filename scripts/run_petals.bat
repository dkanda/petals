@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Petals One-Click Runner (Windows)
echo ==========================================

:: Check if python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Define the virtual environment directory
set VENV_DIR=petals_venv

:: Check if the virtual environment exists, create if it doesn't
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment in %VENV_DIR%...
    python -m venv %VENV_DIR%
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate the virtual environment
echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

:: Install dependencies if not already installed
:: Using git+https as per README, or pip install . if in the repo root
:: First try to install from the local repository if we are in it
if exist "setup.cfg" (
    echo [INFO] Installing Petals from local repository...
    python -m pip install -e .
) else (
    echo [INFO] Installing Petals from git...
    python -m pip install git+https://github.com/bigscience-workshop/petals
)

if !errorlevel! neq 0 (
    echo [ERROR] Failed to install Petals dependencies.
    pause
    exit /b 1
)

echo [INFO] Setup complete!

:: Start the server with any arguments passed to the script
echo [INFO] Starting Petals server...
python -m petals.cli.run_server %*

:: Keep window open if the server exits
pause
