"""MCP stdio server loop and JSON-RPC dispatch (Section 8, Appendix C)."""

import json
import logging
import sys
from typing import Any, Dict, Optional, TextIO

from .client import VrgdgApiClient
from .prompts import get_prompt, list_prompts
from .protocol import (
    PROTOCOL_VERSION,
    SERVER_NAME,
    SERVER_VERSION,
    format_tool_result,
    make_jsonrpc_error,
    make_jsonrpc_response,
)
from .resources import list_resources, read_resource
from .tools import ALL_TOOLS

logger = logging.getLogger("vrgdg.mcp.server")


SERVER_INSTRUCTIONS = (
    "VRGDG Music Video Builder. These tools drive the same project the Video Builder edits in ComfyUI.\n"
    "To make a music video while chatting with the user, get the prompt `chat_make_music_video`: it asks the user what "
    "they want, confirms, then builds everything. If you already have all the details, get `make_music_video` (or read "
    "resource `vrgdg://docs/music-video-playbook`) and follow it step by step.\n"
    "Resource `vrgdg://docs/endpoints` lists every API endpoint and what it does, and `vrgdg://docs/openapi` is the "
    "full contract. Every endpoint has a tool: the named tools first, `api_*` tools for the rest, and `api_request` "
    "for any path. Call `llm_active` before any step that writes text: the LLM is whatever is loaded in LM Studio "
    "and must never be changed.\n"
    "Long operations (LLM steps, renders, stitch) return a job id. Call `job_wait` until the job is succeeded or failed."
)


class McpServer:
    """Zero-dependency Model Context Protocol stdio server."""

    def __init__(self, client: Optional[VrgdgApiClient] = None):
        self.client = client or VrgdgApiClient()

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatch a single JSON-RPC request to the appropriate MCP handler."""
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}

        # 1. Lifecycle
        if method == "initialize":
            return make_jsonrpc_response(
                req_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {},
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                    "instructions": SERVER_INSTRUCTIONS,
                },
            )

        if method == "notifications/initialized":
            # Notifications do not return a response
            return None

        if method == "ping":
            return make_jsonrpc_response(req_id, {})

        # 2. Tools
        if method == "tools/list":
            tools_list = [tool.to_mcp_dict() for tool in ALL_TOOLS.values()]
            return make_jsonrpc_response(req_id, {"tools": tools_list})

        if method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments") or {}

            tool = ALL_TOOLS.get(tool_name)
            if not tool:
                result = format_tool_result(f"Tool not found: '{tool_name}'", is_error=True)
                return make_jsonrpc_response(req_id, result)

            tool_result = tool.handler(self.client, tool_args)
            return make_jsonrpc_response(req_id, tool_result)

        # 3. Resources
        if method == "resources/list":
            resources_list = list_resources(self.client)
            return make_jsonrpc_response(req_id, {"resources": resources_list})

        if method == "resources/read":
            uri = params.get("uri", "")
            try:
                res_content = read_resource(self.client, uri)
                return make_jsonrpc_response(req_id, res_content)
            except Exception as exc:
                return make_jsonrpc_error(req_id, -32603, f"Failed to read resource '{uri}': {exc}")

        # 4. Prompts
        if method == "prompts/list":
            prompts_list = list_prompts()
            return make_jsonrpc_response(req_id, {"prompts": prompts_list})

        if method == "prompts/get":
            prompt_name = params.get("name", "")
            prompt_args = params.get("arguments") or {}
            try:
                p_content = get_prompt(prompt_name, prompt_args)
                return make_jsonrpc_response(req_id, p_content)
            except KeyError:
                return make_jsonrpc_error(req_id, -32602, f"Prompt not found: '{prompt_name}'")
            except Exception as exc:
                return make_jsonrpc_error(req_id, -32603, f"Failed to get prompt '{prompt_name}': {exc}")

        # Method not found
        return make_jsonrpc_error(req_id, -32601, f"Method not found: '{method}'")

    def run(self, in_stream: Optional[TextIO] = None, out_stream: Optional[TextIO] = None) -> None:
        """Run the stdio message processing loop."""
        in_io = in_stream if in_stream is not None else sys.stdin
        out_io = out_stream if out_stream is not None else sys.stdout

        logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION} on stdio...")
        for line in in_io:
            clean_line = line.strip()
            if not clean_line:
                continue

            try:
                request = json.loads(clean_line.lstrip("﻿"))  # some Windows shells prefix a BOM
            except json.JSONDecodeError as jde:
                err_resp = make_jsonrpc_error(None, -32700, f"Parse error: {jde}")
                out_io.write(json.dumps(err_resp) + "\n")
                out_io.flush()
                continue

            response = self.handle_request(request)
            if response is not None:
                out_io.write(json.dumps(response) + "\n")
                out_io.flush()


def run_stdio_server(client: Optional[VrgdgApiClient] = None, in_stream: Optional[TextIO] = None, out_stream: Optional[TextIO] = None) -> None:
    server = McpServer(client=client)
    server.run(in_stream=in_stream, out_stream=out_stream)
