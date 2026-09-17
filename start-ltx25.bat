@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if /I "%~1"=="modal" goto modal
if /I "%~1"=="local" goto local
if /I "%~1"=="q" exit /b 0

cls
echo ========================================
echo            LTX-2.5 APP Launcher
echo ========================================
echo.
echo   [1] Modal GPU mode   ^(recommended^)
echo   [2] Local GPU mode
echo   [Q] Quit
echo.
choice /C 12Q /N /M "Select mode [1/2/Q]: "
if errorlevel 3 exit /b 0
if errorlevel 2 goto local

:modal
call "%~dp0start-ltx25-modal.bat"
exit /b %errorlevel%

:local
call "%~dp0start-ltx25-local.bat"
exit /b %errorlevel%
