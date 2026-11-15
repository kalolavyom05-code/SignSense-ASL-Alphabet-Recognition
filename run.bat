@echo off
title ASL Sign Language Detector
cd /d "%~dp0"
echo ===================================================
echo     Starting ASL Real-Time Hand Sign Detector
echo ===================================================

if exist ".\.venv\Scripts\python.exe" (
    set "PY_EXE=.\.venv\Scripts\python.exe"
) else if exist "C:\Users\kalol\.gemini\antigravity\scratch\sign-language-detector\.venv\Scripts\python.exe" (
    set "PY_EXE=C:\Users\kalol\.gemini\antigravity\scratch\sign-language-detector\.venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

"%PY_EXE%" realtime_inference.py
if errorlevel 1 (
    echo.
    echo An error occurred while running the detector.
    pause
)
