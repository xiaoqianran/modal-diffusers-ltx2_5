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

echo [LTX-2.5] Starting cloud APP...
echo [Frontend] %APP_URL%
echo [Router  ] %LOCAL_API%  ^(local Modal SDK -^> RTX PRO 6000^)

start "LTX25 Local Router" /D "%~dp0" cmd /k ""%PYTHON%" -m uvicorn ltx25.api:app --host 127.0.0.1 --port 48125"

powershell -NoProfile -Command "$u='%LOCAL_API%/api/health'; for($i=0;$i -lt 80;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 $u; if($r.StatusCode -lt 500){ exit 0 } } catch {}; Start-Sleep -Milliseconds 250 }; exit 1" >nul 2>nul || (
  echo [LTX-2.5] Local Modal router did not start.
  pause
  exit /b 1
)

start "LTX25 Frontend" /D "%~dp0frontend" cmd /k "set VITE_API_TARGET=%LOCAL_API%&& npm run dev"

powershell -NoProfile -Command "$u='%APP_URL%'; for($i=0;$i -lt 40;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 $u; if($r.StatusCode -lt 500){ exit 0 } } catch {}; Start-Sleep -Milliseconds 250 }; exit 1" >nul 2>nul
start "" "%APP_URL%"

endlocal
