@echo off
title Grab and Go - AI Smart Retail Checkout
echo ===================================================
echo     STARTING GRAB ^& GO SMART RETAIL SYSTEM
echo ===================================================
cd /d "%~dp0"
echo Starting FastAPI Web Server at http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause
