@echo off
setlocal
title VoxCPM Local API Server

echo ========================================================
echo         Starting VoxCPM Local API Server
echo ========================================================
echo.

cd /d "%~dp0"

REM Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    echo Activating .venv virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment .venv not found, using system Python...
)

REM Run API Server
python api_server.py --host 0.0.0.0 --port 8000 --device auto %*

pause
