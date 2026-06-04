@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%scripts\Install-VRGDG-ComfyUI-Portable.ps1"

if not exist "%PS_SCRIPT%" (
  echo Could not find "%PS_SCRIPT%".
  echo Make sure this BAT file is next to the scripts folder from comfyui-vrgamedevgirl.
  pause
  exit /b 1
)

powershell.exe -Sta -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
  echo Installer finished.
) else (
  echo Installer stopped with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
