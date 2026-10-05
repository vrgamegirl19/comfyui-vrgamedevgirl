@echo off
REM Build the two-scene Latent Continuation Masked test workflow, and render it through the running ComfyUI.
REM
REM Start ComfyUI first. Scene 1 renders, then scene 2 continues it in the same graph. Results are in
REM ComfyUI\output\MaskedContinuationTest. The workflow to load on the canvas is
REM Workflows\masked_continuation_test\MaskedContinuation_2Scene_Test_API.json.
REM
REM Options are passed through, for example:
REM   run_masked_continuation_test.bat --audio "C:\music\song.mp3" --start 45 --context-frames 90

setlocal
set "PACK=%~dp0"
set "PYTHON=%PACK%..\..\..\python_embeded\python.exe"

if not exist "%PYTHON%" (
    echo [VRGDG] Portable Python not found at "%PYTHON%".
    pause
    exit /b 1
)

cd /d "%PACK%"
"%PYTHON%" "%PACK%scripts\run_masked_continuation_test.py" --run %*
echo.
pause
exit /b %ERRORLEVEL%
