# ========================================
# Gravity-Lang Windows Build Script
# ========================================
#
# This PowerShell script builds a standalone Windows executable (.exe)
# for Gravity-Lang using PyInstaller.
#
# Prerequisites:
#   - Python 3.8 or higher installed
#   - Internet connection for installing dependencies
#
# Usage:
#   1. Open PowerShell
#   2. Navigate to the Gravity-Lang directory
#   3. Run: .\build-windows.ps1
#
# If you get an execution policy error, run this once:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#
# ========================================

Write-Host ""
Write-Host "========================================"
Write-Host "  Gravity-Lang Windows Build Script"
Write-Host "========================================"
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[1/4] Python version:"
    Write-Host "    $pythonVersion"
    Write-Host ""
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.8+ from https://www.python.org/downloads/"
    Write-Host "Make sure to check 'Add Python to PATH' during installation"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Install dependencies
Write-Host "[2/4] Installing dependencies..."
try {
    pip install --quiet pyinstaller numpy
    Write-Host "    PyInstaller and NumPy installed successfully" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Build the executable
Write-Host "[3/4] Building Windows executable..."
Write-Host "    This may take 2-3 minutes..."
try {
    python gravity_lang_interpreter.py build-exe --name gravity-lang-windows --outdir dist
    Write-Host ""
} catch {
    Write-Host "ERROR: Build failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Test the executable
Write-Host "[4/4] Testing executable..."
if (Test-Path "dist\gravity-lang-windows.exe") {
    $version = & "dist\gravity-lang-windows.exe" --version
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  BUILD SUCCESSFUL!"
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Executable location: dist\gravity-lang-windows.exe"
    $fileSize = (Get-Item "dist\gravity-lang-windows.exe").Length / 1MB
    Write-Host "File size: $([math]::Round($fileSize, 2)) MB"
    Write-Host ""
    Write-Host "Version: $version"
    Write-Host ""
    Write-Host "Test the executable:"
    Write-Host "  .\dist\gravity-lang-windows.exe run examples\moon_orbit.gravity"
    Write-Host ""
} else {
    Write-Host "ERROR: Executable not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Read-Host "Press Enter to exit"
