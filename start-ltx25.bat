@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Load simple KEY=VALUE entries so media backend settings in .env reach the
rem local FastAPI router. Comment lines are ignored; shell env set by the user
rem later in this script still takes precedence where explicit defaults exist.
if exist ".env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" set "%%A=%%B"
  )
)

set "APP_URL=http://127.0.0.1:5187"
set "LOCAL_API=http://127.0.0.1:48125"
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not defined LTX25_LOCAL_KEEP_GPU_WARM set "LTX25_LOCAL_KEEP_GPU_WARM=1"
if not defined LTX25_MODAL_GPU_IDLE_SECONDS set "LTX25_MODAL_GPU_IDLE_SECONDS=600"
if not defined LTX25_MEDIA_BACKEND set "LTX25_MEDIA_BACKEND=volume"

if not exist "%PYTHON%" (
  echo [LTX-2.5] Python venv was not found: %PYTHON%
  pause
  exit /b 1
)

"%PYTHON%" -c "import fastapi,uvicorn,multipart,PIL,modal" >nul 2>nul || (
  echo [LTX-2.5] Installing lightweight local router dependencies...
  where uv >nul 2>nul || (
    echo [LTX-2.5] uv was not found in PATH.
    pause
    exit /b 1
  )
  uv pip install --python "%PYTHON%" -r requirements-local.txt || (pause & exit /b 1)
)

if /I "%LTX25_MEDIA_BACKEND%"=="s3" (
  "%PYTHON%" -c "import boto3" >nul 2>nul || (
    echo [LTX-2.5] Installing S3-compatible client dependency...
    uv pip install --python "%PYTHON%" "boto3>=1.40,<2" || (pause & exit /b 1)
  )
)

where npm >nul 2>nul || (
  echo [LTX-2.5] npm was not found in PATH.
  pause
  exit /b 1
)

if not exist "frontend\node_modules" (
  echo [LTX-2.5] Installing frontend dependencies...
  pushd frontend
  call npm install || (popd & pause & exit /b 1)
  popd
)

echo [LTX-2.5] Cleaning stale listeners on ports 48125 / 5187...
powershell -NoProfile -Command "$ports=48125,5187; foreach($p in $ports){ Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }"
timeout /t 1 /nobreak >nul

echo [LTX-2.5] Starting local app supervisor...
"%PYTHON%" "%~dp0tools\local_launcher.py"
set "RC=%ERRORLEVEL%"

endlocal
exit /b %RC%
