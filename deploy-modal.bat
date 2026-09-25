@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   LTX-2.5 / Qwen One-Click Modal Deploy
echo   Media: Cloudflare R2 -^> MinIO fallback
echo ========================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\deploy_modal.ps1" %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo [OK] Deployment completed.
) else (
  echo [ERROR] Deployment failed with exit code %RC%.
)

pause
exit /b %RC%
