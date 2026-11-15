@echo off
title Train ASL Model
cd /d "%~dp0"
.\.venv\Scripts\python.exe train_classifier.py
pause
