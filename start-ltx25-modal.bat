@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:5187"
set "API_CONFIG=%~dp0.ltx25-api-base"

if not defined LTX25_API_BASE if exist "%API_CONFIG%" set /p LTX25_API_BASE=<"%API_CONFIG%"
if not defined LTX25_API_BASE set "LTX25_API_BASE=https://weiranzhiqian--ltx25-nvfp4-ltx25server-web.modal.run"

>"%API_CONFIG%" echo %LTX25_API_BASE%

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

echo [LTX-2.5] Modal APP mode
echo [Frontend] %APP_URL%
echo [Backend ] %LTX25_API_BASE%

start "LTX25 Frontend" /D "%~dp0frontend" cmd /k "set VITE_API_TARGET=%LTX25_API_BASE%&& npm run dev"

powershell -NoProfile -Command "$u='%APP_URL%'; for($i=0;$i -lt 40;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 $u; if($r.StatusCode -lt 500){ exit 0 } } catch {}; Start-Sleep -Milliseconds 250 }; exit 1" >nul 2>nul
start "" "%APP_URL%"

endlocal
