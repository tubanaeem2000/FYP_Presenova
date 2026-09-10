@echo off
title AI Presentation Hub

echo ====================================================================
echo               AI Presentation Hub - Launcher                        
echo ====================================================================
echo.

:: 1. Start Flask Backend
echo [1/2] Starting Flask Backend Server (Port 5000)...
start "AI Hub Backend (Port 5000)" cmd /k "set PYTHONNOUSERSITE=1 && if exist .venv\Scripts\python.exe (.venv\Scripts\python.exe -s main.py) else (python -s main.py)"

:: 2. Start React Frontend
echo [2/2] Starting React Vite Frontend Server (Port 3000)...
start "AI Hub Frontend (Port 3000)" cmd /k "cd frontend && npm run dev"

echo.
echo ====================================================================
echo   Services launched!
echo   - Backend: http://localhost:5000
echo   - Frontend: http://localhost:3000 (usually opens automatically)
echo.
echo   Press any key in this window to close the launcher.
echo   (The backend/frontend terminal windows will remain running)
echo ====================================================================
pause
