"""MCP Resources definitions and handlers (Section 8.2)."""

import json
import os
import re
from typing import Any, Dict, List, Optional

from .client import ApiClientError, VrgdgApiClient


PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Documents an agent can read to learn the API: (uri, file relative to the pack root, name, mime type, description).
DOC_RESOURCES = [
    ("vrgdg://docs/music-video-playbook", "MUSIC_VIDEO_AGENT_PROMPT.md", "Music Video Playbook", "text/markdown",
     "Step-by-step instructions for building a finished music video with the tools, with a form for the song, lyrics, character and style"),
    ("vrgdg://docs/chat-agent-prompt", "CHAT_AGENT_PROMPT.md", "Chat Agent Prompt", "text/markdown",
     "Instructions for an agent talking with a user: it asks for the song, lyrics, character and style, confirms, then builds the whole video"),
    ("vrgdg://docs/endpoints", "Api_Endpoints.md", "API Endpoints", "text/markdown",
     "Every Agent API endpoint and what it does (the tools call these)"),
    ("vrgdg://docs/openapi", os.path.join("agent_api", "openapi.json"), "OpenAPI Contract", "application/json",
     "Machine-readable contract: paths, parameters, request body keys, settings schemas and error codes"),
]

RESOURCE_DESCRIPTORS: List[Dict[str, Any]] = [
    {"uri": uri, "name": name, "description": description, "mimeType": mime}
    for uri, _path, name, mime, description in DOC_RESOURCES
] + [
    {
        "uri": "vrgdg://projects",
        "name": "Projects List",
        "description": "List of all active Music Video projects",
        "mimeType": "application/json",
    },
    {
        "uri": "vrgdg://modes",
        "name": "Modes Catalog",
        "description": "Available image, video, and engine modes",
        "mimeType": "application/json",
    },
    {
        "uriTemplate": "vrgdg://project/{project_id}",
        "name": "Project Session",
        "description": "Full JSON state of a project session",
        "mimeType": "application/json",
    },
    {
        "uriTemplate": "vrgdg://project/{project_id}/scene/{scene_id}",
        "name": "Scene Details",
        "description": "JSON details of a single timeline scene segment",
        "mimeType": "application/json",
    },
    {
        "uriTemplate": "vrgdg://project/{project_id}/lyrics",
        "name": "Project Lyrics",
        "description": "Plain text lyrics for a project",
        "mimeType": "text/plain",
    },
    {
        "uriTemplate": "vrgdg://jobs/{job_id}/log",
        "name": "Job Status and Log",
        "description": "Status, progress, and log messages for a job",
        "mimeType": "text/plain",
    },
]


def list_resources(client: VrgdgApiClient) -> List[Dict[str, Any]]:
    """Return all static and templated resources."""
    return RESOURCE_DESCRIPTORS


def read_resource(client: VrgdgApiClient, uri: str) -> Dict[str, Any]:
    """Resolve and fetch the content of an MCP resource URI."""
    clean_uri = str(uri or "").strip()

    for doc_uri, doc_path, _name, mime, _description in DOC_RESOURCES:
        if clean_uri == doc_uri:
            full_path = os.path.join(PACK_ROOT, doc_path)
            try:
                with open(full_path, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except OSError as exc:
                raise ApiClientError("RESOURCE_NOT_FOUND", f"Could not read {doc_path}: {exc}")
            return {"contents": [{"uri": clean_uri, "mimeType": mime, "text": text}]}

    if clean_uri == "vrgdg://projects":
        data = client.get("/projects")
        return {
            "contents": [
                {
                    "uri": clean_uri,
                    "mimeType": "application/json",
                    "text": json.dumps(data, indent=2),
                }
            ]
        }

    if clean_uri == "vrgdg://modes":
        data = client.get("/modes")
        return {
            "contents": [
                {
                    "uri": clean_uri,
                    "mimeType": "application/json",
                    "text": json.dumps(data, indent=2),
                }
            ]
        }

    # vrgdg://project/{id}/scene/{sid}
    scene_m = re.match(r"^vrgdg://project/([^/]+)/scene/([^/]+)$", clean_uri)
    if scene_m:
        pid, sid = scene_m.group(1), scene_m.group(2)
        data = client.get(f"/projects/{pid}/scenes/{sid}")
        return {
            "contents": [
                {
                    "uri": clean_uri,
                    "mimeType": "application/json",
                    "text": json.dumps(data, indent=2),
                }
            ]
        }

    # vrgdg://project/{id}/lyrics
    lyrics_m = re.match(r"^vrgdg://project/([^/]+)/lyrics$", clean_uri)
    if lyrics_m:
        pid = lyrics_m.group(1)
        data = client.get(f"/projects/{pid}/lyrics")
        raw_text = data.get("lyrics_text") or json.dumps(data, indent=2)
        return {
            "contents": [
                {
                    "uri": clean_uri,
                    "mimeType": "text/plain",
                    "text": str(raw_text),
                }
            ]
        }

    # vrgdg://project/{id}
    project_m = re.match(r"^vrgdg://project/([^/]+)$", clean_uri)
    if project_m:
        pid = project_m.group(1)
        data = client.get(f"/projects/{pid}")
        return {
            "contents": [
                {
                    "uri": clean_uri,
                    "mimeType": "application/json",
                    "text": json.dumps(data, indent=2),
                }
            ]
        }

    # vrgdg://jobs/{id}/log
    job_m = re.match(r"^vrgdg://jobs/([^/]+)/log$", clean_uri)
    if job_m:
        jid = job_m.group(1)
        data = client.get(f"/jobs/{jid}")
        job = data.get("job") or data
        log_lines = [
            f"Job ID: {job.get('id')}",
            f"Type: {job.get('type')}",
            f"Status: {job.get('status')}",
            f"Progress: {job.get('progress')}%",
            f"Message: {job.get('message')}",
            f"Error: {job.get('error') or 'None'}",
        ]
        return {
            "contents": [
                {
                    "uri": clean_uri,
                    "mimeType": "text/plain",
                    "text": "\n".join(log_lines),
                }
            ]
        }

    raise ApiClientError("RESOURCE_NOT_FOUND", f"Unknown resource URI: {clean_uri}")
