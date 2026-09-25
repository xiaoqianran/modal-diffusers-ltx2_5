@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
set "APP_NAME=ltx25-nvfp4"
set "MODAL_ENV=main"

if exist ".env" (
  for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if /I "%%A"=="LTX25_MODAL_APP" set "LTX25_MODAL_APP=%%B"
    if /I "%%A"=="LTX25_MODAL_ENVIRONMENT" set "LTX25_MODAL_ENVIRONMENT=%%B"
  )
)

if not exist "%PYTHON%" (
  echo [ERROR] Project Python was not found:
  echo         %PYTHON%
  echo.
  pause
  exit /b 1
)

rem Keep this in sync with modal_app.py:
rem APP_NAME = os.environ.get("LTX25_MODAL_APP", "ltx25-nvfp4")
if defined LTX25_MODAL_APP set "APP_NAME=%LTX25_MODAL_APP%"
if defined LTX25_MODAL_ENVIRONMENT set "MODAL_ENV=%LTX25_MODAL_ENVIRONMENT%"

echo ========================================
echo   LTX-2.5 / Qwen Modal Stop
echo ========================================
echo.
echo App: %APP_NAME%
echo Environment: %MODAL_ENV%
echo.
echo This will stop the deployed Modal app and terminate its running GPU container.
echo Volumes, Secrets, model weights, and cached assets will NOT be deleted.
echo.

"%PYTHON%" -m modal app stop "%APP_NAME%" --env "%MODAL_ENV%" --yes
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo [OK] Modal app stopped: %APP_NAME%
  echo [OK] The resident GPU container has been terminated.
  echo [INFO] Run deploy-modal.bat whenever you want to deploy it again.
) else (
  echo [ERROR] Failed to stop Modal app. Exit code: %RC%
)

echo.
pause
exit /b %RC%
