"""MCP tools for every Agent API endpoint that has no hand-written tool, plus a generic ``api_request`` tool.

The endpoint list is ``agent_api/endpoints.json`` (generated from the router by
``scripts/export_api_endpoints.py``), so a new endpoint becomes a tool without editing this server.
Hand-written tools in ``tools.py`` keep their names. An endpoint they already call is not repeated.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .client import VrgdgApiClient
from .protocol import format_tool_result
from .tools import ToolDefinition, _safe_call

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDPOINTS_PATH = os.path.join(PACK_ROOT, "agent_api", "endpoints.json")
TOOLS_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools.py")

_PARAM = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ARG_NAMES = {"pid": "project_id", "sid": "scene_id", "rid": "reference_id"}


# What an agent needs to know about the arguments the endpoints share. Only wording, never a type: the route decides what
# it accepts, so a hint here cannot make a valid call fail a client's schema check.
_ARG_HINTS = {
    "project_id": "The project's id (its folder name). Get it from project_list.",
    "scene_id": "The scene's id, e.g. seg_1. Get it from scene_list.",
    "reference_id": "The reference card's id. Get it from references_get.",
    "job_id": "A job id returned by a call that runs as a job.",
    "id": "The id from the path.",
    "name": "The name from the path (a preset or file name).",
    "field": "Which prompt field to write, e.g. t2i_prompt or i2v_prompt.",
    "kind": "The reference group named in the path, e.g. subjects or locations.",
    "n": "The number from the path (an anchor or run index).",
}
_BODY_HINTS = {
    "keys": "Reference keys to use, as listed by the matching GET (`available` / `selected`).",
    "automatic": "true lets the Builder choose the references itself; false uses `keys`.",
    "prompt": "The prompt text.",
    "mode": "The MiniMax H3 mode, e.g. text_to_video, image_to_video, reference_to_video.",
    "save": "true stores the result on the scene; false only returns it.",
    "shots": "The shot list to assemble, in order.",
    "origin": "Where the text came from, e.g. llm or manual.",
    "beats": "Beat times in seconds, ascending.",
    "tempo_bpm": "Tempo in beats per minute.",
    "offset_seconds": "Seconds to shift the beat grid by.",
    "scope": "Which scenes to touch, e.g. selected or all.",
    "edge": "Which scene edge to snap, start or end.",
    "settings": "The settings object to save.",
    "provider": "The LLM provider name.",
    "source_video_path": "Path of the rendered video to take the frame from.",
    "run_index": "Which face-fix run to use.",
    "order": "Which anchor, by order.",
}
_QUERY_HINTS = {
    "project_id": "Only this project.",
    "status": "Only jobs in this status, e.g. running, queued, done, error.",
    "type": "Only jobs of this type.",
    "since": "Return log lines after this position.",
    "include": "Comma-separated groups and/or top-level session keys.",
    "peaks": "How many waveform points to return.",
    "provider": "The LLM provider name.",
    "kind": "Which prompt context to return.",
    "folder": "RefMod folder to list.",
}
# Calls that overwrite or remove something that cannot be brought back from the tool result.
_DESTRUCTIVE_POSTS = ("/minimax/cleanup", "/llm/unload", "/video/trim")


def _annotations(method: str, path: str) -> Dict[str, Any]:
    """MCP tool annotations from the method: GET reads, DELETE removes, PUT sets, POST changes (idempotent only for PUT)."""
    if method == "GET":
        return {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
    destructive = method == "DELETE" or any(path.endswith(tail) for tail in _DESTRUCTIVE_POSTS)
    return {
        "readOnlyHint": False,
        "destructiveHint": destructive,
        "idempotentHint": method in ("PUT", "DELETE"),
        "openWorldHint": False,
    }


def _normalize(path: str) -> str:
    return _PARAM.sub("{}", path.split("?")[0])


def load_endpoints(path: str = ENDPOINTS_PATH) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return list(json.load(handle).get("endpoints") or [])
    except (OSError, ValueError):
        return []


def covered_by_handwritten_tools(source_path: str = TOOLS_SOURCE_PATH) -> Set[Tuple[str, str]]:
    """(METHOD, normalized path) pairs the hand-written tools already call."""
    try:
        with open(source_path, "r", encoding="utf-8") as handle:
            source = handle.read()
    except OSError:
        return set()
    covered: Set[Tuple[str, str]] = set()
    for match in re.finditer(r'client\.(get|post|put|patch|delete)\(\s*f?"([^"]+)"', source):
        covered.add((match.group(1).upper(), _normalize(match.group(2))))
    # Paths built into a variable first, then passed to the client: count the path for any method.
    for match in re.finditer(r'(?:path|url|endpoint)\s*=\s*f?"(/[^"]+)"', source):
        covered.add(("*", _normalize(match.group(1))))
    return covered


def _is_covered(endpoint: Dict[str, Any], covered: Set[Tuple[str, str]]) -> bool:
    path = _normalize(endpoint["path"])
    return (endpoint["method"], path) in covered or ("*", path) in covered


def _arg_name(param: str, path: str) -> str:
    if param == "project_id":
        return "project_id"
    if param == "id" and path.startswith("/jobs"):
        return "job_id"
    return _ARG_NAMES.get(param, param)


def _slug(endpoint: Dict[str, Any]) -> str:
    """Readable tool name from the path: ``/projects/{pid}/scenes/{sid}/video/trim`` -> ``scene_video_trim``."""
    parts = [p for p in endpoint["path"].split("/") if p]
    words = []
    for part in parts:
        if _PARAM.fullmatch(part) or part == "projects":
            continue
        words.append(re.sub(r"[^a-z0-9]+", "_", part.lower()).strip("_"))
    name = "_".join(w for w in words if w) or "root"
    return f"{endpoint['method'].lower()}_{name}"


def _unique_names(endpoints: List[Dict[str, Any]], taken: Set[str]) -> Dict[int, str]:
    """A distinct ``api_`` name per endpoint. A clash adds the method, then the trailing path parameter, then a number."""
    names: Dict[int, str] = {}
    used = set(taken)
    for index, endpoint in enumerate(endpoints):
        base = "api_" + _slug(endpoint)
        candidates = [base]
        tail = _PARAM.findall(endpoint["path"])[-1:] if endpoint["path"].rstrip("/").endswith("}") else []
        if tail:
            candidates.append(f"{base}_by_{tail[0]}")
        choice = next((c for c in candidates if c not in used), None)
        number = 2
        while choice is None:
            choice = f"{base}_{number}" if f"{base}_{number}" not in used else None
            number += 1
        used.add(choice)
        names[index] = choice[:64]
    return names


def _build_tool(endpoint: Dict[str, Any], name: str) -> ToolDefinition:
    method, path = endpoint["method"], endpoint["path"]
    arg_for = {param: _arg_name(param, path) for param in endpoint["path_params"]}
    properties: Dict[str, Any] = {
        arg: ({"type": "string", "description": _ARG_HINTS[arg]} if arg in _ARG_HINTS else {"type": "string"})
        for arg in arg_for.values()
    }
    required = list(arg_for.values())
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        # The ids in the path are arguments of their own, so they are not repeated as body keys.
        keys = [key for key in endpoint["body_keys"] if key not in arg_for.values() and key not in endpoint["path_params"]]
        body: Dict[str, Any] = {
            "type": "object",
            "description": "JSON request body." + (
                f" Fields: {', '.join(keys)}." if keys
                else " This endpoint's fields are not listed here: see its entry in resource vrgdg://docs/endpoints."
            ),
        }
        if keys:
            body["properties"] = {key: ({"description": _BODY_HINTS[key]} if key in _BODY_HINTS else {}) for key in keys}
            body["additionalProperties"] = True
        properties["body"] = body
    if endpoint["query"]:
        properties["query"] = {
            "type": "object",
            "description": f"Query parameters: {', '.join(endpoint['query'])}.",
            "properties": {key: ({"description": _QUERY_HINTS[key]} if key in _QUERY_HINTS else {}) for key in endpoint["query"]},
            "additionalProperties": True,
        }
    if endpoint["if_match"]:
        properties["if_match_revision"] = {"type": "integer", "description": "Fail with a conflict if the project revision differs."}
    description = f"{endpoint['description']} ({method} {path})"
    if endpoint["job"]:
        description += " Runs as a background job: returns a job id, then call job_wait."

    @_safe_call
    def handler(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
        real_path = _PARAM.sub(lambda m: str(args.get(arg_for[m.group(1)], "")).strip(), path)
        missing = [arg for arg in required if not str(args.get(arg, "")).strip()]
        if missing:
            return format_tool_result(f"Error [VALIDATION_ERROR]: missing {', '.join(missing)}.", is_error=True)
        body = args.get("body") if isinstance(args.get("body"), dict) else None
        query = args.get("query") if isinstance(args.get("query"), dict) else None
        revision = args.get("if_match_revision")
        if method == "GET":
            res = client.get(real_path, params=query)
        elif method == "DELETE":
            res = client.delete(real_path, params=query, json_data=body, if_match_revision=revision)
        else:
            res = getattr(client, method.lower())(real_path, json_data=body if body is not None else {}, params=query, if_match_revision=revision)
        return format_tool_result(res)

    return ToolDefinition(
        name=name, description=description,
        input_schema={"type": "object", "properties": properties, "required": required},
        handler=handler, annotations=_annotations(method, path),
    )


@_safe_call
def _api_request(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    method = str(args.get("method") or "GET").upper()
    path = str(args.get("path") or "").strip()
    if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
        return format_tool_result("Error [VALIDATION_ERROR]: method must be GET, POST, PUT, PATCH or DELETE.", is_error=True)
    if not path.startswith("/") or "://" in path or ".." in path:
        return format_tool_result("Error [VALIDATION_ERROR]: path must start with / and be relative to the API base, e.g. /projects.", is_error=True)
    body = args.get("body") if isinstance(args.get("body"), dict) else None
    query = args.get("query") if isinstance(args.get("query"), dict) else None
    revision = args.get("if_match_revision")
    if method == "GET":
        res = client.get(path, params=query)
    elif method == "DELETE":
        res = client.delete(path, params=query, json_data=body, if_match_revision=revision)
    else:
        res = getattr(client, method.lower())(path, json_data=body if body is not None else {}, params=query, if_match_revision=revision)
    return format_tool_result(res)


API_REQUEST_TOOL = ToolDefinition(
    name="api_request",
    description=(
        "Call any Agent API endpoint directly. The full list is in resource vrgdg://docs/endpoints. `path` is relative to "
        "the API base, with ids filled in, e.g. /projects/MySong/scenes. Use this for anything the named tools do not cover. "
        "Endpoints that run as jobs return a job id: call job_wait."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
            "path": {"type": "string", "description": "e.g. /projects/MySong/scenes/seg_1/video/trim"},
            "body": {"type": "object", "description": "JSON request body"},
            "query": {"type": "object", "description": "Query parameters"},
            "if_match_revision": {"type": "integer"},
        },
        "required": ["method", "path"],
    },
    handler=_api_request, annotations={"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False},
)


def build_endpoint_tools(existing: Dict[str, ToolDefinition], endpoints_path: Optional[str] = None) -> Dict[str, ToolDefinition]:
    """Tools for endpoints the hand-written tools do not call, keyed by name, plus ``api_request``."""
    covered = covered_by_handwritten_tools()
    uncovered = [e for e in load_endpoints(endpoints_path or ENDPOINTS_PATH) if not _is_covered(e, covered)]
    names = _unique_names(uncovered, set(existing))
    tools = {names[i]: _build_tool(endpoint, names[i]) for i, endpoint in enumerate(uncovered)}
    tools[API_REQUEST_TOOL.name] = API_REQUEST_TOOL
    return tools
