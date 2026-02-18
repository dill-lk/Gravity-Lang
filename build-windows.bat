@echo off
REM ========================================
REM Gravity-Lang Windows Build Script
REM ========================================
REM
REM This script builds a standalone Windows executable (.exe)
REM for Gravity-Lang using PyInstaller.
REM
REM Prerequisites:
REM   - Python 3.8 or higher installed
REM   - Internet connection for installing dependencies
REM
REM Usage:
REM   1. Open Command Prompt or PowerShell
REM   2. Navigate to the Gravity-Lang directory
REM   3. Run: build-windows.bat
REM
REM ========================================

echo.
echo ========================================
echo   Gravity-Lang Windows Build Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [1/4] Python version:
python --version
echo.

REM Install dependencies
echo [2/4] Installing dependencies...
pip install --quiet pyinstaller numpy
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo     PyInstaller and NumPy installed successfully
echo.

REM Build the executable
echo [3/4] Building Windows executable...
echo     This may take 2-3 minutes...
python gravity_lang_interpreter.py build-exe --name gravity-lang-windows --outdir dist
if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)
echo.

REM Test the executable
echo [4/4] Testing executable...
if exist "dist\gravity-lang-windows.exe" (
    dist\gravity-lang-windows.exe --version
    echo.
    echo ========================================
    echo   BUILD SUCCESSFUL!
    echo ========================================
    echo.
    echo Executable location: dist\gravity-lang-windows.exe
    echo File size: 
    dir dist\gravity-lang-windows.exe | find "gravity-lang-windows.exe"
    echo.
    echo Test the executable:
    echo   dist\gravity-lang-windows.exe run examples\moon_orbit.gravity
    echo.
) else (
    echo ERROR: Executable not found
    pause
    exit /b 1
)

pause
