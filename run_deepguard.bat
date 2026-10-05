@echo off
setlocal
title DeepGuard AI - Real-Time Call Deepfake Audio Detection Sentinel
echo ====================================================================
echo   DEEPGUARD AI - REAL-TIME CALL DEEPFAKE DETECTION SENTINEL
echo ====================================================================

cd /d "%~dp0"

:: Default host is 0.0.0.0 (listen on all network adapters/hosts)
set TARGET_HOST=%1
if "%TARGET_HOST%"=="" set TARGET_HOST=0.0.0.0

set TARGET_PORT=%2
if "%TARGET_PORT%"=="" set TARGET_PORT=0

echo [*] Target Host Binding: %TARGET_HOST%
if "%TARGET_PORT%"=="0" (
    echo [*] Target Port:         Auto-Detect (checks 8000, 8050, 8080)
) else (
    echo [*] Target Port:         %TARGET_PORT%
)
echo [*] Usage: run_deepguard.bat [HOST] [PORT]
echo --------------------------------------------------------------------

python run_deepguard.py --host %TARGET_HOST% --port %TARGET_PORT%

pause
