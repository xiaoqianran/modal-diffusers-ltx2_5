@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:5187"
set "API_URL=http://127.0.0.1:8000"

where npm >nul 2>nul || (
  echo [LTX-2.5] npm was not found in PATH.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [LTX-2.5] Missing .venv.
  echo Create it and install requirements before using Local APP mode.
  pause
  exit /b 1
)

.venv\Scripts\python.exe -c "import fastapi, uvicorn" >nul 2>nul || (
  echo [LTX-2.5] Local backend dependencies are missing.
  echo Run:
  echo   uv pip install --python .venv\Scripts\python.exe -r requirements.txt
  echo.
  pause
  exit /b 1
)

if not exist "frontend\node_modules" (
  echo [LTX-2.5] Installing frontend dependencies...
  pushd frontend
  call npm install || (popd & pause & exit /b 1)
  popd
)

echo [LTX-2.5] Local APP mode
echo [Frontend] %APP_URL%
echo [Backend ] %API_URL%

start "LTX25 Backend" /D "%~dp0" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
start "LTX25 Frontend" /D "%~dp0frontend" cmd /k "set VITE_API_TARGET=%API_URL%&& npm run dev"

powershell -NoProfile -Command "$u='%APP_URL%'; for($i=0;$i -lt 60;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 $u; if($r.StatusCode -lt 500){ exit 0 } } catch {}; Start-Sleep -Milliseconds 250 }; exit 1" >nul 2>nul
start "" "%APP_URL%"

endlocal
