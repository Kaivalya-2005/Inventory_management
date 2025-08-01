@echo off
echo Starting Sparkathon Inventory Management System...
echo.

echo Starting FastAPI Backend...
cd backend
start "FastAPI Backend" cmd /k "uvicorn app:app --reload --host 0.0.0.0 --port 8000"
cd ..

echo.
echo Starting React Frontend...
cd frontend/client
start "React Frontend" cmd /k "npm start"
cd ../..

echo.
echo Both servers are starting...
echo Backend will be available at: http://localhost:8000
echo Frontend will be available at: http://localhost:3000
echo.
echo Press any key to close this window...
pause > nul 