@echo off
title Record ASL Signs
cd /d "%~dp0"
.\.venv\Scripts\python.exe collect_data.py
pause
