"""Entrypoint for launching MCP server (Section 8).

Supports both `python -m mcp_server` and `python mcp_server/__main__.py`.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

try:
    from .server import run_stdio_server
except ImportError:
    from mcp_server.server import run_stdio_server

if __name__ == "__main__":
    run_stdio_server()
