@echo off
rem ============================================================================
rem  Marketing Intelligence Platform - double-click this file to start the app.
rem
rem  - Works from any folder name (spaces and "&" included): it uses its own
rem    location (%~dp0) and always quotes paths.
rem  - Runs Streamlit "headless" so it never stops to ask for an e-mail address
rem    on first start, then opens the browser itself.
rem  - Close this window (or press Ctrl+C) to stop the app.
rem  - If anything goes wrong, the window stays open so the message can be read.
rem ============================================================================
setlocal
title Marketing Intelligence Platform
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  Could not find the project's Python environment:
    echo    "%~dp0.venv\Scripts\python.exe"
    echo.
    echo  Create it first ^(see START_HERE.md^), then double-click run_app.bat again.
    echo.
    pause
    exit /b 1
)

echo.
echo  Starting the Marketing Intelligence Platform ...
echo  The app will open in your browser at http://localhost:8501
echo  Keep this window open while you use the app. Close it to stop the app.
echo.

rem Open the browser after a few seconds, when the server is ready.
rem (Set MI_NO_BROWSER=1 before running to skip this, e.g. for automated tests.)
if not defined MI_NO_BROWSER (
    start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process 'http://localhost:8501'"
)

".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --server.port 8501
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo  The app stopped with an error ^(code %EXITCODE%^). Please read the message above.
    echo  Tip: if it says the port is already in use, the app may already be running -
    echo  look for another "Marketing Intelligence Platform" window or open http://localhost:8501
    echo.
    pause
)
endlocal
exit /b %EXITCODE%
