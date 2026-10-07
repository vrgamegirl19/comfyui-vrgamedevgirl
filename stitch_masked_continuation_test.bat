@echo off
REM Trim and join the two clips the masked continuation test workflow rendered (use after running it on the canvas).
REM
REM The raw clips repeat the warm-up, so they must not be joined as they are. This trims each clip the same way the
REM Video Builder does and writes ComfyUI\output\MaskedContinuationTest\MaskedContinuation_FINAL.mp4.

setlocal
set "PACK=%~dp0"
set "PYTHON=%PACK%..\..\..\python_embeded\python.exe"

if not exist "%PYTHON%" (
    echo [VRGDG] Portable Python not found at "%PYTHON%".
    pause
    exit /b 1
)

cd /d "%PACK%"
"%PYTHON%" "%PACK%scripts\run_masked_continuation_test.py" --stitch
echo.
pause
exit /b %ERRORLEVEL%
