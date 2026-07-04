@echo off
setlocal
cd /d "%~dp0"

echo Make sure the Flow page is open with the generated image and download button visible.
echo This will inspect the page DOM and write debug files.
echo.
node inspect-flow-buttons.mjs

echo.
echo The debug files are in the debug folder.
pause
