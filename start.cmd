@echo off
REM ===================================================================
REM  Kelvin6K - start everything after a laptop restart.
REM
REM  Double-click this file, or run:  start.cmd
REM
REM  It opens two windows:
REM    1. the API backend  (camera, database, speech-to-text, speech)
REM    2. the dashboard    (the web page you talk to)
REM
REM  Close either window with Ctrl+C. Closing with the X leaves the
REM  process running and holding its port, which is what causes the
REM  "port 8000 already in use" message on the next start.
REM ===================================================================

cd /d "%~dp0"

echo.
echo  Starting the Kelvin6K backend...
REM --restart stops anything still holding port 8000 from a previous run.
start "Kelvin6K API" cmd /k "python run_api.py --restart"

REM Give the backend a moment so the dashboard's first requests succeed.
timeout /t 6 /nobreak >nul

echo  Starting the Kelvin6K dashboard...
start "Kelvin6K UI" cmd /k "cd ui && npm run dev"

timeout /t 6 /nobreak >nul

echo.
echo  ------------------------------------------------------------
echo   Open this address in Chrome or Edge:
echo.
echo       http://localhost:5173
echo.
echo   IMPORTANT: always use localhost, not 127.0.0.1. The browser
echo   remembers microphone permission per address, so the two are
echo   different sites to it -- and permission granted on one does
echo   not apply to the other.
echo  ------------------------------------------------------------
echo.
start "" http://localhost:5173
