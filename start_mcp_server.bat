@echo off
REM Start the VRGDG MCP server (stdio). Local only: mcp_server/ is git-ignored.
REM
REM An MCP client (Claude Code, Claude Desktop, ...) normally starts this itself from its config, and talks to it
REM over stdin/stdout. Do not print anything to stdout here: that stream is the protocol. Messages go to stderr.
REM
REM Run it by hand only to check that it starts. It then waits for JSON-RPC lines on stdin (Ctrl+C to stop).
REM
REM Settings (set before running, or in the client's config):
REM   VRGDG_API_URL    ComfyUI Agent API address. Default http://127.0.0.1:8188/vrgdg/api/v1
REM   VRGDG_API_TOKEN  Bearer token, only if the server is set to require one.

setlocal
set "PACK=%~dp0"
set "PYTHON=%PACK%..\..\..\python_embeded\python.exe"

if not exist "%PYTHON%" (
    echo [VRGDG MCP] Portable Python not found at "%PYTHON%". 1>&2
    exit /b 1
)

if "%VRGDG_API_URL%"=="" set "VRGDG_API_URL=http://127.0.0.1:8188/vrgdg/api/v1"

echo [VRGDG MCP] Starting. API: %VRGDG_API_URL% 1>&2

cd /d "%PACK%"
REM The portable Python ignores the current folder for "-m", so run the entry file (it adds the pack folder itself).
"%PYTHON%" "%PACK%mcp_server\__main__.py"
exit /b %ERRORLEVEL%
