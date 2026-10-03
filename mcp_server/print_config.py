"""Print the MCP client config for this install, using this install's own paths.

Run with the portable Python so the printed ``command`` is the right interpreter::

    ..\\..\\..\\python_embeded\\python.exe mcp_server\\print_config.py

Paste the JSON into the client's MCP config (LM Studio ``mcp.json``, Claude Desktop). Claude Code users can
run the printed ``claude mcp add`` line instead. Nothing is written to disk.
"""

import json
import os
import sys

API_URL = "http://127.0.0.1:8188/vrgdg/api/v1"


def build_config(python_exe: str, entry_file: str, api_url: str = API_URL) -> dict:
    """Return the ``mcpServers`` config for the given interpreter and entry file."""
    return {
        "mcpServers": {
            "vrgdg": {
                "command": python_exe.replace("\\", "/"),
                "args": [entry_file.replace("\\", "/")],
                "env": {"VRGDG_API_URL": api_url},
            }
        }
    }


def main() -> None:
    pack = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    entry_file = os.path.join(pack, "mcp_server", "__main__.py")
    portable = os.path.abspath(os.path.join(pack, "..", "..", "..", "python_embeded", "python.exe"))
    python_exe = portable if os.path.isfile(portable) else sys.executable
    if python_exe != portable:
        print(f"[VRGDG MCP] Portable Python not found at {portable}; using {python_exe}.", file=sys.stderr)

    print(json.dumps(build_config(python_exe, entry_file), indent=2))
    print(f'\nClaude Code: claude mcp add vrgdg -- "{python_exe}" "{entry_file}"')


if __name__ == "__main__":
    main()
