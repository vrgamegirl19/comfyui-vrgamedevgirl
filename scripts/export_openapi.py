"""Export the VRGDG Agent API v1 contract as an OpenAPI 3.1 document (Apireport K6, D12).

aiohttp does not generate OpenAPI, so the document is built from two sources:

* ``agent_api/router.py`` is parsed with ``ast`` (never imported, so ComfyUI is not
  needed) to list every route: method, path, handler, path/query/header inputs,
  request body keys, and the success status.
* ``agent_api/schemas.py``, ``errors.py`` and ``jobs/models.py`` supply the shared
  component schemas (settings groups, error codes, job statuses).

Usage (from the pack root, with the portable Python)::

    ..\\..\\..\\python_embeded\\python.exe scripts/export_openapi.py          # write agent_api/openapi.json
    ..\\..\\..\\python_embeded\\python.exe scripts/export_openapi.py --check  # exit 1 if the file is stale

The committed ``agent_api/openapi.json`` is the contract snapshot. Any diff needs a
deliberate version decision, so ``tests/test_agent_api_contract.py`` fails when the
file does not match the router.
"""

import argparse
import ast
import dataclasses
import importlib.util
import json
import re
import sys
import types
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
AGENT_API_DIR = ROOT / "agent_api"
ROUTER_PATH = AGENT_API_DIR / "router.py"
OPENAPI_PATH = AGENT_API_DIR / "openapi.json"

API_PREFIX = "/vrgdg/api/v1"
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
# Bump when the contract changes in a way clients must notice (Apireport 11, item 4).
CONTRACT_VERSION = "1.0.0"
_PATH_PARAM = re.compile(r"{([A-Za-z_][A-Za-z0-9_]*)}")


def _load_agent_api_module(name: str) -> types.ModuleType:
    """Import ``agent_api.<name>`` (or ``minimax.<name>``) without running package ``__init__`` files.

    ``agent_api/__init__.py`` imports the router, which needs ComfyUI's ``server``.
    Bare stand-in packages let the dependency-free modules load in any Python while
    keeping their relative imports (``..minimax``, ``.errors``) working.
    """
    def ensure_package(qualified: str, directory: Path) -> types.ModuleType:
        package = sys.modules.get(qualified)
        if package is None:
            package = types.ModuleType(qualified)
            package.__path__ = [str(directory)]
            sys.modules[qualified] = package
        return package

    ensure_package("_vrgdg_pack", ROOT)
    ensure_package("_vrgdg_pack.agent_api", AGENT_API_DIR)
    ensure_package("_vrgdg_pack.agent_api.jobs", AGENT_API_DIR / "jobs")
    ensure_package("_vrgdg_pack.minimax", ROOT / "minimax")
    qualified = f"_vrgdg_pack.agent_api.{name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    file_path = AGENT_API_DIR / (name.replace(".", "/") + ".py")
    spec = importlib.util.spec_from_file_location(qualified, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def _resolve_route_path(node: ast.expr) -> Optional[str]:
    """Turn the f-string passed to ``routes.<method>`` into a plain path."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if not isinstance(node, ast.JoinedStr):
        return None
    parts: List[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append(str(value.value))
        elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name) and value.value.id == "_API_V1_PREFIX":
            parts.append(API_PREFIX)
        else:
            return None
    return "".join(parts)


def _route_decorator(func: ast.AsyncFunctionDef) -> Optional[Tuple[str, str]]:
    """Return ``(method, path)`` when the function is registered with ``routes.<method>(...)``."""
    for decorator in func.decorator_list:
        if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
            continue
        method = decorator.func.attr
        owner = decorator.func.value
        if method in HTTP_METHODS and isinstance(owner, ast.Attribute) and owner.attr == "routes" and decorator.args:
            path = _resolve_route_path(decorator.args[0])
            if path:
                return method, path
    return None


def _first_str_arg(call: ast.Call) -> Optional[str]:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


def _analyze_handler(func: ast.AsyncFunctionDef) -> Dict[str, Any]:
    """Collect what one handler reads from the request and how it responds."""
    info: Dict[str, Any] = {
        "query": set(),
        "body_keys": set(),
        "reads_json": False,
        "reads_multipart": False,
        "if_match": False,
        "status": 200,
        "binary": False,
    }
    body_vars = set()
    for node in ast.walk(func):
        # payload = await request.json()
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Await, ast.IfExp)):
            text = ast.unparse(node.value)
            if "request.json()" in text:
                info["reads_json"] = True
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        body_vars.add(target.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = ast.unparse(node.func.value)
            attr = node.func.attr
            if owner == "request.query" and attr == "get":
                key = _first_str_arg(node)
                if key:
                    info["query"].add(key)
            elif owner == "request.headers" and attr == "get" and _first_str_arg(node) == "If-Match":
                info["if_match"] = True
            elif owner in body_vars and attr == "get":
                key = _first_str_arg(node)
                if key:
                    info["body_keys"].add(key)
            elif owner == "request" and attr == "json":
                info["reads_json"] = True
            elif owner == "request" and attr in {"multipart", "post"}:
                info["reads_multipart"] = True
            elif owner == "web" and attr in {"FileResponse", "StreamResponse"}:
                info["binary"] = True
            elif owner == "web" and attr == "Response":
                info["binary"] = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "api_success":
            for keyword in node.keywords:
                if keyword.arg == "status" and isinstance(keyword.value, ast.Constant):
                    info["status"] = int(keyword.value.value)
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "request.query":
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                info["query"].add(key.value)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in body_vars:
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                info["body_keys"].add(key.value)
    return info


def extract_routes(router_source: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return one record per registered route, sorted by path then method."""
    source = router_source if router_source is not None else ROUTER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    routes: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        decorated = _route_decorator(node)
        if not decorated:
            continue
        method, path = decorated
        info = _analyze_handler(node)
        docstring = ast.get_docstring(node) or ""
        routes.append({
            "method": method,
            "path": path,
            "operation_id": node.name,
            "summary": docstring.strip().splitlines()[0] if docstring.strip() else node.name.replace("api_", "").replace("_", " "),
            "path_params": _PATH_PARAM.findall(path),
            **info,
        })
    routes.sort(key=lambda item: (item["path"], item["method"]))
    return routes


_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}


def _annotation_schema(annotation: Any) -> Dict[str, Any]:
    """Map a dataclass field annotation to a JSON Schema fragment."""
    origin = typing.get_origin(annotation)
    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}
    if origin in (dict, Dict):
        return {"type": "object"}
    if origin in (list, List):
        args = typing.get_args(annotation)
        return {"type": "array", "items": _annotation_schema(args[0]) if args else {}}
    if origin is typing.Union:
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        inner = _annotation_schema(args[0]) if len(args) == 1 else {}
        return {"oneOf": [inner, {"type": "null"}]} if inner else {}
    if dataclasses.is_dataclass(annotation):
        return {"$ref": f"#/components/schemas/{annotation.__name__}"}
    return {}


def _dataclass_schema(cls: type, schemas: Dict[str, Any]) -> None:
    """Add ``cls`` and any nested dataclasses to ``schemas``."""
    if cls.__name__ in schemas:
        return
    hints = typing.get_type_hints(cls)
    properties: Dict[str, Any] = {}
    for item in dataclasses.fields(cls):
        annotation = hints.get(item.name, Any)
        if dataclasses.is_dataclass(annotation):
            _dataclass_schema(annotation, schemas)
        prop = _annotation_schema(annotation)
        if item.default is not dataclasses.MISSING:
            prop["default"] = item.default
        properties[item.name] = prop
    schemas[cls.__name__] = {
        "type": "object",
        "properties": properties,
        "required": [item.name for item in dataclasses.fields(cls)],
    }


def _minimax_settings_schema() -> Dict[str, Any]:
    """JSON Schema for the full MiniMax H3 settings group, from the shared defaults and limits."""
    _load_agent_api_module("errors")  # registers the stand-in packages
    module = importlib.import_module("_vrgdg_pack.minimax.settings_payload")
    properties: Dict[str, Any] = {}
    for key, entry in module.minimax_h3_settings_schema()["settings"].items():
        prop: Dict[str, Any] = {"type": entry["type"], "default": entry["default"]}
        for field in ("enum", "minimum", "maximum", "multipleOf"):
            if field in entry:
                prop[field] = entry[field]
        properties[key] = prop
    return {
        "type": "object",
        "description": "Every MiniMax H3 video option the Video Builder saves. Send any subset under the `minimax_h3` group.",
        "properties": dict(sorted(properties.items())),
    }


def build_components() -> Dict[str, Any]:
    """Shared schemas: envelopes, error codes, job statuses and the settings groups."""
    errors = _load_agent_api_module("errors")
    schemas_module = _load_agent_api_module("schemas")
    job_models = _load_agent_api_module("jobs.models")

    error_codes = sorted(
        value for name, value in vars(errors).items()
        if name.isupper() and isinstance(value, str) and value == name
    )
    schemas: Dict[str, Any] = {
        "ErrorCode": {"type": "string", "enum": error_codes},
        "JobStatus": {"type": "string", "enum": sorted(job_models.JobStatus.ALL)},
        "ErrorBody": {
            "type": "object",
            "properties": {
                "code": {"$ref": "#/components/schemas/ErrorCode"},
                "message": {"type": "string"},
                "details": {"type": "object"},
                "retryable": {"type": "boolean"},
            },
            "required": ["code", "message", "details", "retryable"],
        },
        "ErrorEnvelope": {
            "type": "object",
            "properties": {
                "ok": {"const": False},
                "error": {"$ref": "#/components/schemas/ErrorBody"},
            },
            "required": ["ok", "error"],
        },
        "SuccessEnvelope": {
            "type": "object",
            "description": "`data` and `revision` are present when the operation has a payload or mutates the project.",
            "properties": {
                "ok": {"const": True},
                "data": {},
                "revision": {"type": "integer"},
            },
            "required": ["ok"],
        },
    }
    _dataclass_schema(schemas_module.EffectiveSettings, schemas)
    schemas["MiniMaxH3Settings"] = _minimax_settings_schema()
    schemas["EffectiveSettings"]["properties"]["minimax_h3"] = {"$ref": "#/components/schemas/MiniMaxH3Settings"}
    _dataclass_schema(job_models.JobProgress, schemas)
    return {
        "schemas": dict(sorted(schemas.items())),
        "securitySchemes": {
            "bearerAuth": {"type": "http", "scheme": "bearer", "description": "Token from VRGDG_Model_Defaults/agent_api.json. Optional on loopback unless require_token_on_loopback is set."},
        },
    }


def _operation(route: Dict[str, Any]) -> Dict[str, Any]:
    """Build the OpenAPI operation object for one route record."""
    parameters: List[Dict[str, Any]] = [
        {"name": name, "in": "path", "required": True, "schema": {"type": "string"}}
        for name in route["path_params"]
    ]
    parameters += [
        {"name": name, "in": "query", "required": False, "schema": {"type": "string"}}
        for name in sorted(route["query"])
    ]
    if route["if_match"]:
        parameters.append({
            "name": "If-Match", "in": "header", "required": False,
            "description": "Project revision the caller last saw. A mismatch returns REVISION_CONFLICT.",
            "schema": {"type": "integer"},
        })
    operation: Dict[str, Any] = {
        "operationId": route["operation_id"],
        "summary": route["summary"],
        "tags": [_tag(route["path"])],
        "security": [{"bearerAuth": []}, {}],
        "parameters": parameters,
        "responses": {},
    }
    if route["reads_json"]:
        body_schema: Dict[str, Any] = {"type": "object"}
        if route["body_keys"]:
            body_schema["properties"] = {key: {} for key in sorted(route["body_keys"])}
        operation["requestBody"] = {
            "required": False,
            "content": {"application/json": {"schema": body_schema}},
        }
    elif route["reads_multipart"]:
        operation["requestBody"] = {
            "required": True,
            "content": {"multipart/form-data": {"schema": {"type": "object"}}},
        }
    if route["binary"]:
        operation["responses"][str(route["status"])] = {
            "description": "Binary or non-JSON payload.",
            "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
        }
    else:
        operation["responses"][str(route["status"])] = {
            "description": "Success.",
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SuccessEnvelope"}}},
        }
    operation["responses"]["default"] = {
        "description": "Error. The HTTP status mirrors error.code.",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorEnvelope"}}},
    }
    return operation


def _tag(path: str) -> str:
    relative = path[len(API_PREFIX):].strip("/")
    parts = relative.split("/")
    if parts[0] == "projects" and len(parts) > 2:
        return parts[2] if not parts[2].startswith("{") else "projects"
    return parts[0] or "root"


def build_openapi(router_source: Optional[str] = None) -> Dict[str, Any]:
    """Build the full OpenAPI 3.1 document."""
    paths: Dict[str, Dict[str, Any]] = {}
    for route in extract_routes(router_source):
        paths.setdefault(route["path"], {})[route["method"]] = _operation(route)
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "VRGDG Agent API",
            "version": CONTRACT_VERSION,
            "description": "Contract shared by the in-pack Agent API and any host that implements /api/v1 (Apireport D12).",
        },
        "servers": [{"url": "http://127.0.0.1:8188"}],
        "paths": dict(sorted(paths.items())),
        "components": build_components(),
    }


def render(document: Dict[str, Any]) -> str:
    """Stable text form used for the snapshot file."""
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export agent_api/openapi.json from router.py and schemas.py.")
    parser.add_argument("--check", action="store_true", help="Exit 1 when agent_api/openapi.json is out of date.")
    parser.add_argument("--output", default=str(OPENAPI_PATH), help="Output path (default: agent_api/openapi.json).")
    args = parser.parse_args(argv)

    text = render(build_openapi())
    target = Path(args.output)
    if args.check:
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        if current != text:
            print("[VRGDG OpenAPI] agent_api/openapi.json is out of date. Run scripts/export_openapi.py.")
            return 1
        print("[VRGDG OpenAPI] agent_api/openapi.json is up to date.")
        return 0

    # core/atomic_write.py has no ComfyUI imports, so load it by path (CLAUDE.md rule 2).
    spec = importlib.util.spec_from_file_location("_vrgdg_atomic_write", ROOT / "core" / "atomic_write.py")
    atomic_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(atomic_module)
    atomic_module.atomic_write_text(str(target), text)
    print(f"[VRGDG OpenAPI] Wrote {target} ({len(json.loads(text)['paths'])} paths).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
