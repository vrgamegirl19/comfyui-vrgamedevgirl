"""MCP JSON-RPC 2.0 Protocol definitions and helper utilities."""

import json
from typing import Any, Dict, List, Optional, Union

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "vrgdg-music-video-mcp"
SERVER_VERSION = "1.0.0"


def make_jsonrpc_response(req_id: Optional[Union[str, int]], result: Any) -> Dict[str, Any]:
    """Format successful JSON-RPC 2.0 response."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    }


def make_jsonrpc_error(
    req_id: Optional[Union[str, int]],
    code: int,
    message: str,
    data: Optional[Any] = None,
) -> Dict[str, Any]:
    """Format JSON-RPC 2.0 error response."""
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": err,
    }


def format_tool_result(
    text: Union[str, Dict[str, Any], List[Any]],
    image_base64: Optional[str] = None,
    image_mime_type: str = "image/jpeg",
    is_error: bool = False,
) -> Dict[str, Any]:
    """Format standard MCP tool execution content blocks (Rule 4, Rule 5)."""
    content: List[Dict[str, Any]] = []

    # If image payload is provided (e.g. contact sheet, thumbnail)
    if image_base64:
        content.append({
            "type": "image",
            "data": image_base64,
            "mimeType": image_mime_type,
        })

    # Text content block
    if isinstance(text, (dict, list)):
        formatted_text = json.dumps(text, indent=2)
    else:
        formatted_text = str(text)

    content.append({
        "type": "text",
        "text": formatted_text,
    })

    return {
        "content": content,
        "isError": is_error,
    }
