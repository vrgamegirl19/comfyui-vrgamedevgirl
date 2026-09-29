@echo off
setlocal
cd /d "%~dp0"

if not exist node_modules\playwright (
  echo Installing local dependencies...
  call npm install --strict-ssl=false
  if errorlevel 1 (
    echo.
    echo Dependency install failed.
    pause
    exit /b 1
  )
)

set "FLOW_URL=https://labs.google/fx/tools/flow"
set "FLOW_PROFILE=%~dp0chrome-flow-profile"
set "CHROME_EXE=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME_EXE%" set "CHROME_EXE=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"

if not exist "%CHROME_EXE%" (
  echo Could not find Chrome at the standard install location.
  pause
  exit /b 1
)

echo Flow Image Edit Automatic Run
echo.
echo This uploads an image, adds it to the prompt, enters your edit prompt, submits, and downloads 2K.
echo Your normal Chrome and ComfyUI can stay open.
echo.
set /p "IMAGE_PATH=Image file path: "
if "%IMAGE_PATH%"=="" (
  echo Image path is required.
  pause
  exit /b 1
)
set /p "FLOW_PROMPT=Edit prompt: "
set "OUTPUT_DIR=%~dp0outputs"
if "%FLOW_PROMPT%"=="" set "FLOW_PROMPT=edit this image"

echo.
echo Starting Flow Chrome on debug port 9222...
start "Flow Chrome" "%CHROME_EXE%" --remote-debugging-port=9222 --user-data-dir="%FLOW_PROFILE%" --window-size=1600,950 "%FLOW_URL%"

echo Waiting for Chrome debug port...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(20); do { try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:9222/json/version' -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; Start-Sleep -Milliseconds 500 } while ((Get-Date) -lt $deadline); exit 1"
if errorlevel 1 (
  echo Chrome debug port did not become ready.
  pause
  exit /b 1
)

echo Running image edit automation now. No browser clicks should be needed.
call npm start -- --url "%FLOW_URL%" --prompt "%FLOW_PROMPT%" --image "%IMAGE_PATH%" --out "%OUTPUT_DIR%" --connect-cdp "http://127.0.0.1:9222"

echo.
echo Done. Check the downloads folder.
pause


