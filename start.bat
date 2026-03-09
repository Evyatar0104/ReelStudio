@echo off
title ReelStudio
echo ============================================
echo   ReelStudio - Video Editor Productivity
echo ============================================
echo.

:: Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)
echo [OK] Python found.

:: Check ffmpeg (warn only)
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARN] ffmpeg not found in PATH. Silence Cutter will not work.
    echo        Download from https://ffmpeg.org/download.html
    echo.
) else (
    echo [OK] ffmpeg found.
)

:: Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt --quiet 2>nul
echo [OK] Dependencies ready.

:: Open browser after delay
echo.
echo Starting ReelStudio on http://localhost:5177 ...
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5177"

:: Run server
python app.py
