@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Follow the setup instructions in README.md.
    pause
    exit /b 1
)

if /i "%~1"=="--joint-test" goto jointtest
if /i "%~1"=="--trained" goto trained
goto baseline

:trained
if not exist "models\four-dof-v1\tongue_a_ppo.zip" goto baseline
if not exist "models\four-dof-v1\tongue_b_ppo.zip" goto baseline

echo Starting French Kiss with the trained agents...
".venv\Scripts\python.exe" -m scripts.demo --trained
goto finished

:jointtest
echo Starting the joint test panel...
".venv\Scripts\python.exe" -m scripts.demo --joint-test
goto finished

:baseline
echo Starting the active scripted interaction...
".venv\Scripts\python.exe" -m scripts.demo

:finished
if errorlevel 1 (
    echo.
    echo French Kiss stopped with an error. See the message above.
    pause
    exit /b 1
)
endlocal
