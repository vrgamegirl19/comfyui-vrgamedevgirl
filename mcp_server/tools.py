"""MCP Tools catalog (T1 to T50) for VRGDG Agent API (Section 8.1)."""

import base64
import json
import logging
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from .client import ApiClientError, VrgdgApiClient
from .protocol import format_tool_result

logger = logging.getLogger("vrgdg.mcp.tools")


class ToolDefinition:
    """Definition of an MCP tool including schema and handler."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[[VrgdgApiClient, Dict[str, Any]], Dict[str, Any]],
        annotations: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler
        self.annotations = annotations

    def to_mcp_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.annotations:
            result["annotations"] = dict(self.annotations)
        return result


# ==============================================================================
# Helper for Next Steps (Rule 5)
# ==============================================================================

def _derive_next_steps(code: str, details: Optional[Dict[str, Any]] = None) -> List[str]:
    steps: List[str] = []
    c = str(code).upper()
    if c == "PREDECESSOR_MISSING":
        steps.append("Render predecessor scene video first or disable latent continuity.")
    elif c == "LATENT_STALE":
        steps.append("Call latents_status_rebuild with action='rebuild' to regenerate the dirty latent chain.")
    elif c == "REVISION_CONFLICT":
        steps.append("Fetch the latest project state via project_get and retry with the updated revision.")
    elif c == "SCENE_NOT_FOUND":
        steps.append("Call scene_list to verify existing scene IDs in this project.")
    elif c == "PROJECT_NOT_FOUND":
        steps.append("Call project_list to check available projects.")
    elif c == "VALIDATION_ERROR":
        steps.append("Call project_validate to inspect missing settings or invalid parameters.")
    return steps


def _safe_call(func: Callable[[VrgdgApiClient, Dict[str, Any]], Dict[str, Any]]) -> Callable[[VrgdgApiClient, Dict[str, Any]], Dict[str, Any]]:
    def wrapper(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return func(client, args)
        except ApiClientError as ace:
            steps = ace.next_steps or _derive_next_steps(ace.code, ace.details)
            steps_text = ("\n\nNext steps:\n" + "\n".join(f"- {s}" for s in steps)) if steps else ""
            msg = f"Error [{ace.code}]: {ace.message}{steps_text}"
            return format_tool_result(msg, is_error=True)
        except Exception as exc:
            logger.exception(f"Unexpected error in tool: {exc}")
            return format_tool_result(f"Error [INTERNAL_ERROR]: {str(exc)}", is_error=True)
    return wrapper


# ==============================================================================
# T1 - T12: Project and System Tools
# ==============================================================================

@_safe_call
def _t1_system_health(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    res = client.get("/health")
    return format_tool_result(res)

@_safe_call
def _t2_list_modes(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    res = client.get("/modes")
    return format_tool_result(res)

@_safe_call
def _t3_list_models(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    res = client.get("/models")
    return format_tool_result(res)

@_safe_call
def _t4_project_list(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    res = client.get("/projects")
    return format_tool_result(res)

@_safe_call
def _t5_project_create(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    body = {"name": args.get("project_name") or args.get("name", "")}
    if args.get("template_from"):
        body["template_from"] = args["template_from"]
    res = client.post("/projects", json_data=body)
    data = res.get("data") if isinstance(res, dict) and isinstance(res.get("data"), dict) else res
    pid = (data or {}).get("project_id") or (data or {}).get("id")
    if pid and args.get("audio_file"):
        client.put(f"/projects/{pid}/audio", json_data={"audio_path": args["audio_file"]})
    elif pid and args.get("duration"):
        client.post(f"/projects/{pid}/audio/silent", json_data={"duration": float(args["duration"])})
    return format_tool_result(res)

@_safe_call
def _t6_project_get(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    params = {"include": args["include"]} if "include" in args else None
    res = client.get(f"/projects/{pid}", params=params)
    return format_tool_result(res)

@_safe_call
def _t7_project_summary(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    res = client.get(f"/projects/{pid}/summary")
    return format_tool_result(res)

@_safe_call
def _t8_project_update_settings(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    settings = args.get("settings", {})
    if_match = args.get("if_match_revision")
    res = client.patch(f"/projects/{pid}/settings", json_data=settings, if_match_revision=if_match)
    return format_tool_result(res)

@_safe_call
def _t51_project_get_settings(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    res = client.get(f"/projects/{pid}/settings")
    return format_tool_result(res)

@_safe_call
def _t52_minimax_settings_schema(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    res = client.get("/settings/minimax-h3/schema")
    return format_tool_result(res)

@_safe_call
def _t9_project_duplicate(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    new_name = args.get("new_name") or args.get("new_project_name")
    payload = {"new_name": new_name} if new_name else {}
    if isinstance(args.get("options"), dict):
        payload["options"] = args["options"]
    res = client.post(f"/projects/{pid}/duplicate", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t10_project_validate(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    payload = args.get("settings", {})
    res = client.post(f"/projects/{pid}/settings/preflight", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t11_project_export(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    payload = {"export_type": args.get("export_type", "zip")}
    res = client.post(f"/projects/{pid}/export", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t12_project_delete(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    if not args.get("confirm"):
        return format_tool_result("Deletion aborted: parameter 'confirm' must be explicitly true.", is_error=True)
    res = client.delete(f"/projects/{pid}", params={"confirm": pid})
    return format_tool_result(res)


# ==============================================================================
# T13 - T24: Audio, Lyrics, and Timeline Tools
# ==============================================================================

@_safe_call
def _t13_audio_attach(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    payload = {"audio_path": args["audio_file"]}
    res = client.put(f"/projects/{pid}/audio", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t14_audio_analyze(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    res = client.get(f"/projects/{pid}/audio/beats")
    return format_tool_result(res)

@_safe_call
def _t15_lyrics_set(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    lyrics = args["lyrics"]
    if isinstance(lyrics, list):
        lyrics = "\n".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in lyrics)
    res = client.put(f"/projects/{pid}/lyrics", json_data={"lyrics_text": str(lyrics)})
    return format_tool_result(res)

@_safe_call
def _t16_lyrics_transcribe(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    language = {"en": "english"}.get(str(args.get("language", "english")).lower(), args.get("language", "english"))
    res = client.post(f"/projects/{pid}/lyrics/align", json_data={"language": language})
    return format_tool_result(res)

@_safe_call
def _t17_timeline_build(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    payload = {
        "text": args.get("text", ""),
        "mode": args.get("mode", "durations"),
        "action": args.get("action", "replace"),
        "append_start": float(args.get("append_start", 0.0)),
        "clear_media": bool(args.get("clear_media", False)),
    }
    res = client.post(f"/projects/{pid}/timeline/bulk", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t18_scene_list(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    res = client.get(f"/projects/{pid}/scenes")
    return format_tool_result(res)

@_safe_call
def _t19_scene_get(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    sid = args["scene_id"]
    res = client.get(f"/projects/{pid}/scenes/{sid}")
    return format_tool_result(res)

@_safe_call
def _t20_scene_update(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    sid = args["scene_id"]
    patch_data = args.get("patch", {})
    if_match = args.get("if_match_revision")
    res = client.patch(f"/projects/{pid}/scenes/{sid}", json_data=patch_data, if_match_revision=if_match)
    return format_tool_result(res)

@_safe_call
def _t21_scene_insert(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    scene_payload = args.get("scene", {})
    res = client.post(f"/projects/{pid}/scenes", json_data=scene_payload)
    return format_tool_result(res)

@_safe_call
def _t22_scene_delete(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    sid = args["scene_id"]
    ripple = args.get("ripple", False)
    res = client.delete(f"/projects/{pid}/scenes/{sid}", params={"ripple": "true" if ripple else "false"})
    return format_tool_result(res)

@_safe_call
def _t23_scene_split_merge_move_resize(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    sid = args["scene_id"]
    op = str(args.get("op", "")).strip().lower()

    if op == "split":
        res = client.post(f"/projects/{pid}/scenes/{sid}/split", json_data={"at_time": args.get("at_time", args.get("split_time"))})
    elif op == "merge":
        res = client.post(f"/projects/{pid}/scenes/{sid}/merge", json_data={"with_direction": args.get("with_direction", args.get("direction", "next"))})
    elif op == "move":
        res = client.post(f"/projects/{pid}/scenes/{sid}/move", json_data={"start_time": args.get("start_time", 0.0), "ripple": args.get("ripple", False)})
    elif op == "resize":
        p = {"ripple": args.get("ripple", False)}
        if "duration" in args:
            p["duration"] = args["duration"]
        if "end_time" in args:
            p["end_time"] = args["end_time"]
        res = client.post(f"/projects/{pid}/scenes/{sid}/resize", json_data=p)
    else:
        return format_tool_result(f"Invalid op: '{op}'. Must be one of: split, merge, move, resize.", is_error=True)

    return format_tool_result(res)

@_safe_call
def _t24_scenes_bulk_edit(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    ops = args.get("operations", [])
    res = client.post(f"/projects/{pid}/scenes/bulk", json_data={"operations": ops})
    return format_tool_result(res)


# ==============================================================================
# T25 - T30: Story and References Tools
# ==============================================================================

@_safe_call
def _t25_story_generate(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    what = str(args.get("what", "brief")).strip().lower()
    params = args.get("params", {})
    if what in ("arc", "brief", "beats"):
        res = client.post(f"/projects/{pid}/story/{what}", json_data=params)
    elif what == "motion_notes":
        res = client.post(f"/projects/{pid}/prompts/motion-notes", json_data=params)
    else:
        res = client.post(f"/projects/{pid}/prompts/concepts", json_data=params)
    return format_tool_result(res)

@_safe_call
def _t26_story_get_set(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    if "story" in args:
        res = client.put(f"/projects/{pid}/story", json_data=args["story"])
    else:
        res = client.get(f"/projects/{pid}/story")
    return format_tool_result(res)

@_safe_call
def _t27_references_get(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    res = client.get(f"/projects/{pid}/references")
    return format_tool_result(res)

@_safe_call
def _t28_reference_upsert(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    kind = "locations" if str(args.get("kind", "subjects")).strip().lower().startswith("loc") else "subjects"
    rid = args["reference_id"]
    payload = args.get("payload", {})
    res = client.put(f"/projects/{pid}/references/{kind}/{rid}", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t29_references_extract(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    res = client.post(f"/projects/{pid}/references/locations/extract", json_data=args.get("params", {}))
    return format_tool_result(res)

@_safe_call
def _t30_reference_scene_mapping_set(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    mapping = args.get("mapping", {})
    res = client.put(f"/projects/{pid}/references/scene-mapping", json_data=mapping)
    return format_tool_result(res)


# ==============================================================================
# T31 - T32: Prompts & Instructions Tools
# ==============================================================================

@_safe_call
def _t31_prompts_generate(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    kind = str(args.get("kind", "image")).strip().lower()
    scope = str(args.get("scope", "all")).strip().lower()
    scene_id = args.get("scene_id")

    if scope == "one" and scene_id:
        kind = kind if kind in ("image", "video", "video-chained", "enhance", "edit") else "video"
        endpoint = f"/projects/{pid}/scenes/{scene_id}/prompts/{kind}"
        res = client.post(endpoint, json_data=args.get("params", {}))
    else:
        endpoint = f"/projects/{pid}/prompts/batch"
        res = client.post(endpoint, json_data={"kind": kind, "scope": scope, **args.get("params", {})})

    return format_tool_result(res)

@_safe_call
def _t32_instructions_get_set(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    key = args["key"]
    action = str(args.get("action", "get")).strip().lower()
    pid = args.get("project_id")

    if action == "save":
        payload = {"text": args.get("text", args.get("prompt", "")), "scope": args.get("scope", "all")}
        if args.get("scene_id"):
            payload["scene_id"] = args["scene_id"]
            payload["scope"] = args.get("scope", "scene")
        if pid:
            payload["project_id"] = pid
        res = client.put(f"/instructions/{key}", json_data=payload)
    elif action == "reset":
        payload = {"project_id": pid} if pid else {}
        payload["scope"] = args.get("scope", "all")
        if args.get("scene_id"):
            payload["scene_id"] = args["scene_id"]
            payload["scope"] = args.get("scope", "scene")
        res = client.delete(f"/instructions/{key}/override", json_data=payload)
    else:
        params = {"project_id": pid} if pid else None
        res = client.get(f"/instructions/{key}", params=params)

    return format_tool_result(res)


# ==============================================================================
# T33 - T43: Generation, Post-Processing, Pipelines Tools
# ==============================================================================

@_safe_call
def _t33_image_generate(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    sid = args.get("scene_id")
    p = args.get("params", {})

    if sid:
        res = client.post(f"/projects/{pid}/scenes/{sid}/image/generate", json_data=p)
    else:
        res = client.post(f"/projects/{pid}/images/generate", json_data=p)

    return format_tool_result(res)

@_safe_call
def _t34_image_set_from_upload(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    sid = args["scene_id"]
    payload = {}
    if "image_data" in args:
        payload["image_data"] = args["image_data"]
    if "source_path" in args:
        payload["source_path"] = args["source_path"]
    res = client.put(f"/projects/{pid}/scenes/{sid}/image", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t35_image_approve_revert(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    sid = args["scene_id"]
    action = str(args.get("action", "approve")).strip().lower()

    if action == "approve":
        p = {"image_path": args.get("image_path")} if "image_path" in args else {}
        res = client.post(f"/projects/{pid}/scenes/{sid}/image/approve", json_data=p)
    elif action == "revert":
        p = {"delta": args.get("delta", -1)}
        if "index" in args:
            p["index"] = args["index"]
        res = client.post(f"/projects/{pid}/scenes/{sid}/image/revert", json_data=p)
    elif action == "delete":
        res = client.delete(f"/projects/{pid}/scenes/{sid}/image")
    else:
        return format_tool_result(f"Invalid image action: '{action}'.", is_error=True)

    return format_tool_result(res)

@_safe_call
def _t36_video_render(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    sid = args.get("scene_id")
    p = args.get("params", {})

    if sid:
        res = client.post(f"/projects/{pid}/scenes/{sid}/video/render", json_data=p)
    else:
        res = client.post(f"/projects/{pid}/video/render", json_data=p)

    return format_tool_result(res)

@_safe_call
def _t37_video_recover(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    sid = args["scene_id"]
    res = client.post(f"/projects/{pid}/scenes/{sid}/video/recover", json_data=args.get("params", {}))
    return format_tool_result(res)

@_safe_call
def _t38_video_select_take(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    sid = args["scene_id"]
    res = client.post(f"/projects/{pid}/scenes/{sid}/video/select", json_data={"source_path": args["source_path"]})
    return format_tool_result(res)

@_safe_call
def _t39_latents_status_rebuild(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    action = str(args.get("action", "status")).strip().lower()

    if action == "dirty":
        res = client.get(f"/projects/{pid}/latents/dirty")
    elif action == "rebuild":
        res = client.post(f"/projects/{pid}/latents/rebuild", json_data=args.get("params", {}))
    elif action == "delete":
        sid = args.get("scene_id", "scene_001")
        all_flag = "true" if args.get("all") else "false"
        res = client.delete(f"/projects/{pid}/scenes/{sid}/latent", params={"all": all_flag})
    else:
        res = client.get(f"/projects/{pid}/latents")

    return format_tool_result(res)

@_safe_call
def _t40_post_apply(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    op = str(args.get("op", "lut")).strip().lower()
    sid = args.get("scene_id")
    p = args.get("settings", {})

    if op == "apply_all":
        res = client.post(f"/projects/{pid}/post/apply-all", json_data=p)
    elif op == "face_fix" and sid:
        res = client.post(f"/projects/{pid}/scenes/{sid}/face-fix/auto", json_data=p)
    elif op == "grain" and sid:
        res = client.post(f"/projects/{pid}/scenes/{sid}/post/film-grain", json_data=p)
    elif op == "adjust" and sid:
        res = client.post(f"/projects/{pid}/scenes/{sid}/post/adjust", json_data=p)
    elif sid:
        # Default lut
        res = client.post(f"/projects/{pid}/scenes/{sid}/post/lut", json_data=p)
    else:
        return format_tool_result(f"Invalid post operation or scene_id missing for '{op}'.", is_error=True)

    return format_tool_result(res)

@_safe_call
def _t41_stitch_final(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    res = client.post(f"/projects/{pid}/stitch", json_data=args.get("params", {}))
    return format_tool_result(res)

@_safe_call
def _t42_pipeline_build_full_video(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    pipeline = str(args.get("pipeline", "full_video")).strip().lower()
    payload = dict(args)
    payload.pop("project_id", None)
    payload.pop("pipeline", None)

    if pipeline == "flf":
        res = client.post(f"/projects/{pid}/pipelines/build-flf", json_data=payload)
    else:
        res = client.post(f"/projects/{pid}/pipelines/build-full-video", json_data=payload)

    return format_tool_result(res)

@_safe_call
def _t54_lyrics_align(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    payload = {k: v for k, v in args.items() if k != "project_id"}
    res = client.post(f"/projects/{pid}/lyrics/align", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t55_timeline_from_lines(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    payload = {k: v for k, v in args.items() if k != "project_id"}
    res = client.post(f"/projects/{pid}/timeline/from-lines", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t56_timeline_enforce_length(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    payload = {k: v for k, v in args.items() if k != "project_id"}
    res = client.post(f"/projects/{pid}/timeline/enforce-length", json_data=payload)
    return format_tool_result(res)

_TIMELINE_NOTE_FIELDS = ("id", "start", "end", "type", "label", "note")


@_safe_call
def _t64_timeline_notes_list(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    res = client.get(f"/projects/{pid}/timeline/notes")
    return format_tool_result(res)

@_safe_call
def _t65_timeline_note_create(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    payload = {k: args[k] for k in _TIMELINE_NOTE_FIELDS if k in args}
    res = client.post(f"/projects/{pid}/timeline/notes", json_data=payload,
                      if_match_revision=args.get("if_match_revision"))
    return format_tool_result(res)

@_safe_call
def _t66_timeline_note_update(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid, nid = args["project_id"], args["note_id"]
    payload = {k: args[k] for k in _TIMELINE_NOTE_FIELDS if k in args and k != "id"}
    res = client.patch(f"/projects/{pid}/timeline/notes/{nid}", json_data=payload,
                       if_match_revision=args.get("if_match_revision"))
    return format_tool_result(res)

@_safe_call
def _t67_timeline_note_delete(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid, nid = args["project_id"], args["note_id"]
    res = client.delete(f"/projects/{pid}/timeline/notes/{nid}", if_match_revision=args.get("if_match_revision"))
    return format_tool_result(res)

@_safe_call
def _t57_reference_describe(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid, kind, rid = args["project_id"], args.get("kind", "subjects"), args["ref_id"]
    payload = {k: v for k, v in args.items() if k not in ("project_id", "kind", "ref_id")}
    res = client.post(f"/projects/{pid}/references/{kind}/{rid}/describe", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t58_reference_extract_locations(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    payload = {k: v for k, v in args.items() if k != "project_id"}
    res = client.post(f"/projects/{pid}/references/locations/extract", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t59_reference_assign_scenes(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    payload = {k: v for k, v in args.items() if k != "project_id"}
    res = client.post(f"/projects/{pid}/references/assign-scenes", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t60_llm_active(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    params = {"project_id": args["project_id"]} if args.get("project_id") else None
    res = client.get("/llm/active", params=params)
    return format_tool_result(res)

@_safe_call
def _t61_story_settings(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    payload = {k: v for k, v in args.items() if k != "project_id"}
    res = client.put(f"/projects/{pid}/story/settings", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t62_story_create(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    step = args["step"]
    payload = {k: v for k, v in args.items() if k not in ("project_id", "step")}
    res = client.post(f"/projects/{pid}/story/{step}", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t63_minimax_prompts(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    payload = {k: v for k, v in args.items() if k != "project_id"}
    res = client.post(f"/projects/{pid}/minimax-prompts", json_data=payload)
    return format_tool_result(res)

@_safe_call
def _t53_pipeline_from_song(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    res = client.post("/pipelines/from-song", json_data=dict(args))
    return format_tool_result(res)

@_safe_call
def _t43_pipeline_plan(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    pid = args["project_id"]
    params = {k: v for k, v in args.items() if k != "project_id"}
    res = client.get(f"/projects/{pid}/pipelines/plan", params=params)
    return format_tool_result(res)


# ==============================================================================
# T44 - T50: Jobs and Media Tools
# ==============================================================================

@_safe_call
def _t44_job_get(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    jid = args["job_id"]
    res = client.get(f"/jobs/{jid}")
    return format_tool_result(res)

@_safe_call
def _t45_job_wait(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    jid = args["job_id"]
    timeout = float(args.get("timeout_seconds", 60.0))
    res = client.job_wait(jid, timeout_seconds=timeout)
    return format_tool_result(res)

@_safe_call
def _t46_job_cancel(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    jid = args["job_id"]
    res = client.post(f"/jobs/{jid}/cancel")
    return format_tool_result(res)

@_safe_call
def _t47_job_retry(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    jid = args["job_id"]
    res = client.post(f"/jobs/{jid}/retry")
    return format_tool_result(res)

_IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}


def _scene_picture_path(client: VrgdgApiClient, project_id: str, scene_id: str) -> str:
    """The best picture of a scene: the video thumbnail, else the approved image."""
    scene = client.get(f"/projects/{project_id}/scenes/{scene_id}")
    project = client.get(f"/projects/{project_id}", params={"include": "audio"})
    folder = str((project or {}).get("project_folder") or "")
    for key in ("video_thumbnail", "approved_image"):
        asset = (scene or {}).get(key)
        rel = str((asset or {}).get("path_rel") or "")
        path = rel if os.path.isabs(rel) else os.path.join(folder, rel)
        if rel and os.path.isfile(path):
            return path
    return ""


@_safe_call
def _t48_asset_view(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    path = str(args.get("path") or args.get("asset_id") or "").strip()
    if not path and args.get("project_id") and args.get("scene_id"):
        path = _scene_picture_path(client, args["project_id"], args["scene_id"])
    if not path:
        return format_tool_result("Give `path` to an image file, or `project_id` and `scene_id` of a scene that has a picture.", is_error=True)
    mime = _IMAGE_TYPES.get(os.path.splitext(path)[1].lower())
    if not mime or not os.path.isfile(path):
        return format_tool_result(f"No image file found at: {path}", is_error=True)
    with open(path, "rb") as handle:
        data = base64.b64encode(handle.read()).decode("ascii")
    return format_tool_result({"path": path, "bytes": os.path.getsize(path)}, image_base64=data, image_mime_type=mime)

@_safe_call
def _t49_asset_download_url(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    path = str(args.get("path") or args.get("asset_id") or "").strip()
    if not path or not os.path.isfile(path):
        return format_tool_result(f"No file found at: {path}", is_error=True)
    return format_tool_result({"path": os.path.abspath(path), "bytes": os.path.getsize(path), "uri": pathlib.Path(os.path.abspath(path)).as_uri()})

@_safe_call
def _t50_upload_file(client: VrgdgApiClient, args: Dict[str, Any]) -> Dict[str, Any]:
    filename = args["filename"]
    kind = str(args.get("kind", "general")).strip().lower()
    content = args.get("content", "")
    pid, sid = args.get("project_id"), args.get("scene_id")

    if kind == "lut" or filename.lower().endswith(".cube"):
        res = client.post("/post/luts/upload", json_data={"filename": filename, "data": content})
    elif kind == "audio" and pid:
        res = client.put(f"/projects/{pid}/audio", json_data={"audio_data": content, "audio_name": filename})
    elif kind == "image" and pid and sid:
        res = client.put(f"/projects/{pid}/scenes/{sid}/image", json_data={"image_data": content})
    else:
        return format_tool_result(
            "Uploads: kind 'lut' (a LUT file), 'audio' (with project_id) or 'image' (with project_id and scene_id). "
            "Files already on this machine do not need uploading: pass their path to audio_attach, reference_upsert or image_set_from_upload.",
            is_error=True,
        )
    return format_tool_result(res)


# ==============================================================================
# Complete Tool Registry
# ==============================================================================

ALL_TOOLS: Dict[str, ToolDefinition] = {
    "system_health": ToolDefinition(
        name="system_health",
        description="Check health of ComfyUI, GPU, queue, and FFmpeg (T1).",
        input_schema={"type": "object", "properties": {}},
        handler=_t1_system_health,
    ),
    "list_modes": ToolDefinition(
        name="list_modes",
        description="List supported image and video modes, requirements, and capabilities (T2).",
        input_schema={"type": "object", "properties": {}},
        handler=_t2_list_modes,
    ),
    "list_models": ToolDefinition(
        name="list_models",
        description="List installed models, checkpoints, and LoRAs available in ComfyUI (T3).",
        input_schema={"type": "object", "properties": {}},
        handler=_t3_list_models,
    ),
    "project_list": ToolDefinition(
        name="project_list",
        description="List all available Music Video projects (T4).",
        input_schema={"type": "object", "properties": {}},
        handler=_t4_project_list,
    ),
    "project_create": ToolDefinition(
        name="project_create",
        description="Create a new music video builder project (T5).",
        input_schema={
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "Name for the project"},
                "audio_file": {"type": "string", "description": "Optional path to audio track"},
                "duration": {"type": "number", "description": "Optional silent timeline duration"},
            },
            "required": ["project_name"],
        },
        handler=_t5_project_create,
    ),
    "project_get": ToolDefinition(
        name="project_get",
        description="Get complete project session details, optionally filtering fields (T6).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "include": {"type": "string", "description": "Comma-separated keys to include"},
            },
            "required": ["project_id"],
        },
        handler=_t6_project_get,
    ),
    "project_summary": ToolDefinition(
        name="project_summary",
        description="Get compact project summary with scene counts, rendered progress, and duration (T7).",
        input_schema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
        handler=_t7_project_summary,
    ),
    "project_update_settings": ToolDefinition(
        name="project_update_settings",
        description=(
            "Update project-level video, image, or engine settings (T8). `settings` is grouped: "
            "{\"minimax_h3\": {...}, \"project\": {...}, \"ltx_video\": {...}, \"zimage\": {...}, "
            "\"flux_klein\": {...}, \"llm\": {...}, \"post_process\": {...}}. Only the keys you send change. "
            "Call minimax_settings_schema for every MiniMax H3 key (render passes, 2 Pass Advanced tiles, "
            "seam controls, LoRAs) with types, allowed values and limits. Invalid values are rejected "
            "with SETTINGS_INVALID."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "settings": {"type": "object", "description": "Grouped settings to update, e.g. {\"minimax_h3\": {\"render_pass\": \"three_pass\"}}"},
                "if_match_revision": {"type": "integer"},
            },
            "required": ["project_id", "settings"],
        },
        handler=_t8_project_update_settings,
    ),
    "project_get_settings": ToolDefinition(
        name="project_get_settings",
        description="Get a project's effective settings in every group, including all MiniMax H3 options (T51).",
        input_schema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
        handler=_t51_project_get_settings,
    ),
    "minimax_settings_schema": ToolDefinition(
        name="minimax_settings_schema",
        description=(
            "List every MiniMax H3 video setting with type, default, allowed values and limits (T52). "
            "Use it to build a valid `minimax_h3` patch for project_update_settings."
        ),
        input_schema={"type": "object", "properties": {}},
        handler=_t52_minimax_settings_schema,
    ),
    "project_duplicate": ToolDefinition(
        name="project_duplicate",
        description="Duplicate an existing project into a new one (T9).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "new_project_name": {"type": "string"},
            },
            "required": ["project_id"],
        },
        handler=_t9_project_duplicate,
    ),
    "project_validate": ToolDefinition(
        name="project_validate",
        description="Preflight check project settings and dependencies (T10).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "settings": {"type": "object"},
            },
            "required": ["project_id"],
        },
        handler=_t10_project_validate,
    ),
    "project_export": ToolDefinition(
        name="project_export",
        description="Export project bundle archive (T11).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "export_type": {"type": "string", "default": "zip"},
            },
            "required": ["project_id"],
        },
        handler=_t11_project_export,
    ),
    "project_delete": ToolDefinition(
        name="project_delete",
        description="Delete a project from disk. Requires confirm=true (T12).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "confirm": {"type": "boolean", "description": "Must be true to proceed"},
            },
            "required": ["project_id", "confirm"],
        },
        handler=_t12_project_delete,
    ),
    "audio_attach": ToolDefinition(
        name="audio_attach",
        description="Attach an audio track to the project (T13).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "audio_file": {"type": "string"},
            },
            "required": ["project_id", "audio_file"],
        },
        handler=_t13_audio_attach,
    ),
    "audio_analyze": ToolDefinition(
        name="audio_analyze",
        description="Analyze audio waveform and extract musical beats/onsets (T14).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "method": {"type": "string", "default": "librosa"},
            },
            "required": ["project_id"],
        },
        handler=_t14_audio_analyze,
    ),
    "lyrics_set": ToolDefinition(
        name="lyrics_set",
        description="Set raw lyric text or timed lyric lines for the project (T15).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "lyrics": {"description": "String or list of timed lyric objects"},
            },
            "required": ["project_id", "lyrics"],
        },
        handler=_t15_lyrics_set,
    ),
    "lyrics_transcribe": ToolDefinition(
        name="lyrics_transcribe",
        description="Transcribe audio into timed lyrics using Whisper (T16).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "language": {"type": "string", "default": "en"},
                "model": {"type": "string", "default": "base"},
            },
            "required": ["project_id"],
        },
        handler=_t16_lyrics_transcribe,
    ),
    "timeline_build": ToolDefinition(
        name="timeline_build",
        description="Build scene segments from text durations, beats, or lyrics (T17).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "text": {"type": "string"},
                "mode": {"type": "string", "default": "durations"},
                "action": {"type": "string", "default": "replace"},
                "append_start": {"type": "number", "default": 0.0},
                "clear_media": {"type": "boolean", "default": False},
            },
            "required": ["project_id"],
        },
        handler=_t17_timeline_build,
    ),
    "scene_list": ToolDefinition(
        name="scene_list",
        description="List all scene segments in the project timeline (T18).",
        input_schema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
        handler=_t18_scene_list,
    ),
    "scene_get": ToolDefinition(
        name="scene_get",
        description="Get single scene details including prompts, images, takes, and timing (T19).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene_id": {"type": "string"},
            },
            "required": ["project_id", "scene_id"],
        },
        handler=_t19_scene_get,
    ),
    "scene_update": ToolDefinition(
        name="scene_update",
        description=(
            "Edit a scene card (T20). The change is saved on the timeline scene and on its Storyboard card, the way "
            "the Builder saves an edit, and appears in the open Video Builder and Storyboard. Only the keys sent "
            "change; an empty string, false or [] clears or disables a field. Read the current card with scene_get "
            "(scene_card). Notes: timeline_note (Director Notes), i2v_notes (Video Notes), notes (Planning Notes), "
            "prompt_summary. Lyrics: lyric_text, lyric_section, lyric_singers, lyric_cue_map, lyric_no_lip_sync, "
            "lyric_instrumental, lyric_performance_mode, lyric_shot_word_timing_enabled, speaker_assignments. "
            "Camera and character: shot_type, camera_motion, character_motion, no_character_present. Performance: "
            "performance_mode (singing|speaking|no_lip_sync), performance_style, facial_performance, "
            "facial_performance_custom, include_microphone. Audio and continuity: audio_direction, continuity, "
            "flf_start_state, flf_transformation, flf_end_state, flf_carry_forward. Look: video_style, "
            "video_style_custom, temporal_world_effect_override, temporal_world_effect_custom, trigger_phrase, "
            "trigger_position. References: subject_ids (list of character ids), location_id. Prompts: image_prompt, "
            "video_prompt, video_prompt_origin, minimax_h3_pass2_prompt, plus the single prompt fields and timing. "
            "Timed Timeline Notes are separate: use timeline_note_create."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene_id": {"type": "string"},
                "patch": {"type": "object", "description": "Scene-card fields to set, for example {\"timeline_note\": \"...\", \"include_microphone\": false}."},
                "if_match_revision": {"type": "integer"},
            },
            "required": ["project_id", "scene_id", "patch"],
        },
        handler=_t20_scene_update,
    ),
    "scene_insert": ToolDefinition(
        name="scene_insert",
        description="Insert a new scene segment into the timeline (T21).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene": {"type": "object"},
            },
            "required": ["project_id", "scene"],
        },
        handler=_t21_scene_insert,
    ),
    "scene_delete": ToolDefinition(
        name="scene_delete",
        description="Delete a scene segment from the timeline (T22).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene_id": {"type": "string"},
                "ripple": {"type": "boolean", "default": False},
            },
            "required": ["project_id", "scene_id"],
        },
        handler=_t22_scene_delete,
    ),
    "scene_split_merge_move_resize": ToolDefinition(
        name="scene_split_merge_move_resize",
        description="Perform split, merge, move, or resize operations on scenes (T23).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene_id": {"type": "string"},
                "op": {"type": "string", "enum": ["split", "merge", "move", "resize"]},
                "split_time": {"type": "number"},
                "direction": {"type": "string", "default": "next"},
                "start_time": {"type": "number"},
                "duration": {"type": "number"},
                "end_time": {"type": "number"},
                "ripple": {"type": "boolean", "default": False},
            },
            "required": ["project_id", "scene_id", "op"],
        },
        handler=_t23_scene_split_merge_move_resize,
    ),
    "scenes_bulk_edit": ToolDefinition(
        name="scenes_bulk_edit",
        description="Execute multiple scene operations atomically (T24).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "operations": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["project_id", "operations"],
        },
        handler=_t24_scenes_bulk_edit,
    ),
    "story_generate": ToolDefinition(
        name="story_generate",
        description="Generate story concepts, story arc, or scene beat notes with LLM (T25).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "what": {"type": "string", "enum": ["brief", "arc", "scene_beats", "dialogue"]},
                "params": {"type": "object"},
            },
            "required": ["project_id", "what"],
        },
        handler=_t25_story_generate,
    ),
    "story_get_set": ToolDefinition(
        name="story_get_set",
        description="Get or update project story data (concept, arc, characters) (T26).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "story": {"type": "object"},
            },
            "required": ["project_id"],
        },
        handler=_t26_story_get_set,
    ),
    "references_get": ToolDefinition(
        name="references_get",
        description="Get project reference cast and locations (T27).",
        input_schema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
        handler=_t27_references_get,
    ),
    "reference_upsert": ToolDefinition(
        name="reference_upsert",
        description="Add or update a reference subject or location (T28).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "kind": {"type": "string", "enum": ["subjects", "locations"]},
                "reference_id": {"type": "string"},
                "payload": {"type": "object"},
            },
            "required": ["project_id", "kind", "reference_id", "payload"],
        },
        handler=_t28_reference_upsert,
    ),
    "references_extract": ToolDefinition(
        name="references_extract",
        description="Extract characters and settings from lyrics or story using LLM (T29).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["project_id"],
        },
        handler=_t29_references_extract,
    ),
    "reference_scene_mapping_set": ToolDefinition(
        name="reference_scene_mapping_set",
        description="Bind reference subjects and locations to scenes (T30).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "mapping": {"type": "object"},
            },
            "required": ["project_id", "mapping"],
        },
        handler=_t30_reference_scene_mapping_set,
    ),
    "prompts_generate": ToolDefinition(
        name="prompts_generate",
        description="Generate image or video prompts for scenes using LLM (T31).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "kind": {"type": "string", "enum": ["image", "video", "chained", "enhance", "edit"]},
                "scope": {"type": "string", "enum": ["one", "all", "selected"]},
                "scene_id": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["project_id"],
        },
        handler=_t31_prompts_generate,
    ),
    "instructions_get_set": ToolDefinition(
        name="instructions_get_set",
        description="Inspect or update system prompt instructions for prompt creator (T32).",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "action": {"type": "string", "enum": ["get", "save", "reset"]},
                "project_id": {"type": "string"},
                "prompt": {"type": "string"},
                "negative_prompt": {"type": "string"},
            },
            "required": ["key"],
        },
        handler=_t32_instructions_get_set,
    ),
    "image_generate": ToolDefinition(
        name="image_generate",
        description="Submit an image generation job for a scene or batch (T33).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene_id": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["project_id"],
        },
        handler=_t33_image_generate,
    ),
    "image_set_from_upload": ToolDefinition(
        name="image_set_from_upload",
        description="Set a scene's starting image from base64 data or existing file path (T34).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene_id": {"type": "string"},
                "image_data": {"type": "string"},
                "source_path": {"type": "string"},
            },
            "required": ["project_id", "scene_id"],
        },
        handler=_t34_image_set_from_upload,
    ),
    "image_approve_revert": ToolDefinition(
        name="image_approve_revert",
        description="Approve, revert, or delete scene image preview (T35).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene_id": {"type": "string"},
                "action": {"type": "string", "enum": ["approve", "revert", "delete"]},
                "image_path": {"type": "string"},
                "delta": {"type": "integer"},
                "index": {"type": "integer"},
            },
            "required": ["project_id", "scene_id", "action"],
        },
        handler=_t35_image_approve_revert,
    ),
    "video_render": ToolDefinition(
        name="video_render",
        description="Submit a video render job for a scene or batch (T36).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene_id": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["project_id"],
        },
        handler=_t36_video_render,
    ),
    "video_recover": ToolDefinition(
        name="video_recover",
        description="Recover scene video from rendered backup takes (T37).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene_id": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["project_id", "scene_id"],
        },
        handler=_t37_video_recover,
    ),
    "video_select_take": ToolDefinition(
        name="video_select_take",
        description="Select active take from scene video history (T38).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene_id": {"type": "string"},
                "source_path": {"type": "string"},
            },
            "required": ["project_id", "scene_id", "source_path"],
        },
        handler=_t38_video_select_take,
    ),
    "latents_status_rebuild": ToolDefinition(
        name="latents_status_rebuild",
        description="Check latent status or rebuild dirty latent chain (Invariant 4) (T39).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "action": {"type": "string", "enum": ["status", "dirty", "rebuild", "delete"]},
                "scene_id": {"type": "string"},
                "all": {"type": "boolean"},
                "params": {"type": "object"},
            },
            "required": ["project_id"],
        },
        handler=_t39_latents_status_rebuild,
    ),
    "post_apply": ToolDefinition(
        name="post_apply",
        description="Apply LUT, film grain, color adjust, or face fix (T40).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "op": {"type": "string", "enum": ["lut", "grain", "adjust", "face_fix", "apply_all"]},
                "scene_id": {"type": "string"},
                "settings": {"type": "object"},
            },
            "required": ["project_id", "op"],
        },
        handler=_t40_post_apply,
    ),
    "stitch_final": ToolDefinition(
        name="stitch_final",
        description="Stitch rendered scene videos into FINAL_VIDEO.mp4 (T41).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["project_id"],
        },
        handler=_t41_stitch_final,
    ),
    "pipeline_build_full_video": ToolDefinition(
        name="pipeline_build_full_video",
        description="Execute complete automated end-to-end video pipeline (T42).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "pipeline": {"type": "string", "enum": ["full_video", "flf"]},
                "build_mode": {"type": "string", "enum": ["resume_missing", "fresh_rebuild", "redo_videos", "redo_i2v_prompts_videos"]},
                "scope": {"type": "string", "enum": ["all", "selected", "from_selected"]},
                "scene_ids": {"type": "array", "items": {"type": "string"}},
                "stitch": {"type": "boolean", "default": True},
            },
            "required": ["project_id"],
        },
        handler=_t42_pipeline_build_full_video,
    ),
    "lyrics_align": ToolDefinition(
        name="lyrics_align",
        description=(
            "Time the project's lyrics against its song with the Builder's Stable-ts workflow (T54). Runs as a job "
            "(use job_wait). Needs project audio and lyrics. Saves the timed lines in the project."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "segment_mode": {"type": "string", "enum": ["reference_lines", "exact_reference_lines", "reference_stanzas", "whisper_chunks"], "default": "reference_lines"},
                "language": {"type": "string", "default": "english"},
                "include_instrumental_gaps": {"type": "boolean", "default": True},
                "min_gap_seconds": {"type": "number", "default": 2.0},
                "vocal_tail_padding_seconds": {"type": "number", "default": 0.6},
            },
            "required": ["project_id"],
        },
        handler=_t54_lyrics_align,
    ),
    "timeline_from_lines": ToolDefinition(
        name="timeline_from_lines",
        description=(
            "Create the timeline scenes from the project's lyrics, like Line Mapping in the Builder (T55). Times the "
            "lyrics (or reuses a saved timing), makes one scene per lyric line, fills instrumental gaps, then merges "
            "scenes shorter than min_scene_seconds and cuts scenes longer than max_scene_seconds. Runs as a job "
            "(use job_wait). Refuses to replace existing scenes unless replace_existing is true, and never "
            "replaces scenes that already have images or videos."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "min_scene_seconds": {"type": "number", "default": 3.5},
                "max_scene_seconds": {"type": "number", "default": 10.0},
                "segment_mode": {"type": "string", "enum": ["reference_lines", "exact_reference_lines", "reference_stanzas", "whisper_chunks"], "default": "reference_lines"},
                "enforce_lengths": {"type": "boolean", "default": True, "description": "False keeps one scene per lyric line whatever its length"},
                "replace_existing": {"type": "boolean", "default": False},
                "use_saved_alignment": {"type": "boolean", "default": False, "description": "Reuse the timing from lyrics_align instead of running it again"},
                "reference_lyrics": {"type": "string", "description": "Defaults to the project lyrics"},
            },
            "required": ["project_id"],
        },
        handler=_t55_timeline_from_lines,
    ),
    "timeline_enforce_length": ToolDefinition(
        name="timeline_enforce_length",
        description=(
            "Merge scenes shorter than min_scene_seconds and cut scenes longer than max_scene_seconds (T56). A 12 s "
            "scene becomes two 6 s scenes with its lyrics divided; short scenes merge into the neighbor that gives the "
            "shorter result. Uses the safe merge/split edits that keep image and video numbering correct. Scenes "
            "with rendered video are left alone and listed. Use dry_run to preview."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "min_scene_seconds": {"type": "number"},
                "max_scene_seconds": {"type": "number"},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["project_id", "min_scene_seconds", "max_scene_seconds"],
        },
        handler=_t56_timeline_enforce_length,
    ),
    "timeline_notes_list": ToolDefinition(
        name="timeline_notes_list",
        description=(
            "List the project's timed Timeline Notes (the Builder's + Timeline Note markers, T64): id, start and end "
            "seconds (end is null for a point note), type, label, note text and the scene_ids each note overlaps. "
            "Story Arc generation (story_create step arc) uses them as timed story events."
        ),
        input_schema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
        handler=_t64_timeline_notes_list,
    ),
    "timeline_note_create": ToolDefinition(
        name="timeline_note_create",
        description=(
            "Add a timed Timeline Note (T65). start is seconds on the project timeline. Give end for a range note or "
            "leave it out for a point note (an event at start). The open Video Builder shows it right away."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "start": {"type": "number", "minimum": 0},
                "end": {"type": ["number", "null"], "description": "Later than start; omit or null for a point note."},
                "type": {"type": "string", "description": "Free text, for example note, chorus, verse, beat, female vocal."},
                "label": {"type": "string"},
                "note": {"type": "string", "description": "The story direction for this moment."},
                "id": {"type": "string", "description": "Optional id; one is made when omitted."},
                "if_match_revision": {"type": "integer"},
            },
            "required": ["project_id", "start"],
        },
        handler=_t65_timeline_note_create,
    ),
    "timeline_note_update": ToolDefinition(
        name="timeline_note_update",
        description=(
            "Change a timed Timeline Note (T66). Only the fields sent change; the id never changes. Send end: null to "
            "turn a range note into a point note."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "note_id": {"type": "string"},
                "start": {"type": "number", "minimum": 0},
                "end": {"type": ["number", "null"]},
                "type": {"type": "string"},
                "label": {"type": "string"},
                "note": {"type": "string"},
                "if_match_revision": {"type": "integer"},
            },
            "required": ["project_id", "note_id"],
        },
        handler=_t66_timeline_note_update,
    ),
    "timeline_note_delete": ToolDefinition(
        name="timeline_note_delete",
        description="Delete a timed Timeline Note (T67).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "note_id": {"type": "string"},
                "if_match_revision": {"type": "integer"},
            },
            "required": ["project_id", "note_id"],
        },
        handler=_t67_timeline_note_delete,
    ),
    "reference_describe": ToolDefinition(
        name="reference_describe",
        description=(
            "Describe a character or location reference image with the project's LLM, like Gemma Describe (T57). "
            "Saves the description on the reference. Runs as a job (use job_wait). With LM Studio it uses the "
            "model that is already loaded and never changes it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "kind": {"type": "string", "enum": ["subjects", "locations"], "default": "subjects"},
                "ref_id": {"type": "string", "description": "Id of the subject or location"},
            },
            "required": ["project_id", "ref_id"],
        },
        handler=_t57_reference_describe,
    ),
    "reference_extract_locations": ToolDefinition(
        name="reference_extract_locations",
        description=(
            "Ask the project's LLM for 18-22 filming locations from the lyrics and style notes, like LM Extract, and "
            "add them to the Reference Builder (T58). Needs scenes with lyrics. Runs as a job (use job_wait)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "style_theme": {"type": "string", "description": "Overall look of the scenes, e.g. 'Los Angeles nightlife, neon, night time'"},
            },
            "required": ["project_id"],
        },
        handler=_t58_reference_extract_locations,
    ),
    "reference_assign_scenes": ToolDefinition(
        name="reference_assign_scenes",
        description=(
            "Assign saved characters and locations to scenes with a pattern, like Assign Scenes (T59). "
            "location_pattern 'blocks' repeats each location for location_block_size scenes, then moves to the next. "
            "Existing mappings are only replaced when replace_existing is true. Use dry_run to preview."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "character_pattern": {"type": "string", "enum": ["random", "rotate", "blocks", "unchanged"], "default": "unchanged"},
                "character_block_size": {"type": "integer", "default": 10},
                "location_pattern": {"type": "string", "enum": ["random", "rotate", "blocks", "unchanged"], "default": "unchanged"},
                "location_block_size": {"type": "integer", "default": 4},
                "replace_existing": {"type": "boolean", "default": False},
                "avoid_location_repeat": {"type": "boolean", "default": True},
                "scope": {"type": "string", "enum": ["all", "range", "selected"], "default": "all"},
                "range_start": {"type": "integer"},
                "range_end": {"type": "integer"},
                "scene_ids": {"type": "array", "items": {"type": "string"}},
                "seed": {"type": "integer"},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["project_id"],
        },
        handler=_t59_reference_assign_scenes,
    ),
    "llm_active": ToolDefinition(
        name="llm_active",
        description="Show which LLM the API will use (T60). For LM Studio this is the model currently loaded; the API never loads or switches models.",
        input_schema={"type": "object", "properties": {"project_id": {"type": "string"}}},
        handler=_t60_llm_active,
    ),
    "story_settings": ToolDefinition(
        name="story_settings",
        description=(
            "Save the Storyboard scene defaults and the story idea (T61). defaults: video_style (e.g. 'Cinematic realism'), "
            "camera_flow (e.g. 'intimate_closeups'), camera_motion_speed 0-10, character_motion_speed 0-10, "
            "minimax_h3_cut_frequency, performance_style, story_arc_detail. story: overall_story_idea (written by the agent), "
            "lyric_story_strength 0-10, image_world_style."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "defaults": {"type": "object"},
                "story": {"type": "object"},
            },
            "required": ["project_id"],
        },
        handler=_t61_story_settings,
    ),
    "story_create": ToolDefinition(
        name="story_create",
        description=(
            "Write a story layer step with the project's LLM, as a background job (T62). step 'arc' writes the story arc from the "
            "story idea, 'brief' writes the song story brief, 'beats' writes a beat per scene (only scenes without one unless "
            "replace_existing; limit caps the count). Returns a job id for job_wait. Order: arc, brief, beats. "
            "For LM Studio the loaded model is used and never changed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "step": {"type": "string", "enum": ["arc", "brief", "beats"]},
                "story_idea": {"type": "string", "description": "arc only; defaults to the saved idea."},
                "story_arc_detail": {"type": "string", "enum": ["compact", "standard", "detailed", "rich"]},
                "scene_ids": {"type": "array", "items": {"type": "string"}},
                "replace_existing": {"type": "boolean", "default": False},
                "limit": {"type": "integer"},
            },
            "required": ["project_id", "step"],
        },
        handler=_t62_story_create,
    ),
    "minimax_prompts": ToolDefinition(
        name="minimax_prompts",
        description=(
            "Write MiniMax H3 reference-to-video prompts with the project's LLM, as a background job (T63). Only scenes without a "
            "prompt are written unless replace_existing; limit caps the count. Every scene needs a mapped character with an image. "
            "Run after story_create beats. Returns a job id for job_wait. For LM Studio the loaded model is used and never changed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene_ids": {"type": "array", "items": {"type": "string"}},
                "replace_existing": {"type": "boolean", "default": False},
                "limit": {"type": "integer"},
            },
            "required": ["project_id"],
        },
        handler=_t63_minimax_prompts,
    ),
    "pipeline_from_song": ToolDefinition(
        name="pipeline_from_song",
        description=(
            "Turn a song into a finished video in one background job (T53): create the project if needed, "
            "attach audio and lyrics, cut scenes, then run the full build. Returns a job id for job_wait. "
            "A project that already has scenes keeps them."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "Creates a new project with this name"},
                "project_id": {"type": "string", "description": "Use an existing project instead"},
                "audio_path": {"type": "string", "description": "Song file; optional if the project already has audio"},
                "lyrics_text": {"type": "string"},
                "scene_seconds": {"type": "number", "default": 4.0},
                "snap_to_beats": {"type": "boolean", "default": True},
                "build_mode": {"type": "string", "enum": ["resume_missing", "fresh_rebuild", "redo_videos", "redo_i2v_prompts_videos"]},
                "max_auto_retries": {"type": "integer", "default": 3},
                "stitch": {"type": "boolean", "default": True},
            },
        },
        handler=_t53_pipeline_from_song,
    ),
    "pipeline_plan": ToolDefinition(
        name="pipeline_plan",
        description="Dry-run plan calculating steps, GPU minutes, and missing prerequisites (T43).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "pipeline": {"type": "string", "enum": ["full_video", "flf"]},
                "build_mode": {"type": "string"},
                "scope": {"type": "string"},
                "scene_ids": {"type": "string"},
            },
            "required": ["project_id"],
        },
        handler=_t43_pipeline_plan,
    ),
    "job_get": ToolDefinition(
        name="job_get",
        description="Check status, progress percentage, and results of a background job (T44).",
        input_schema={
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
        handler=_t44_job_get,
    ),
    "job_wait": ToolDefinition(
        name="job_wait",
        description="Block up to timeout_seconds waiting for job completion without client timeout (T45).",
        input_schema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "timeout_seconds": {"type": "number", "default": 60.0},
            },
            "required": ["job_id"],
        },
        handler=_t45_job_wait,
    ),
    "job_cancel": ToolDefinition(
        name="job_cancel",
        description="Cancel a running or queued background job (T46).",
        input_schema={
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
        handler=_t46_job_cancel,
    ),
    "job_retry": ToolDefinition(
        name="job_retry",
        description="Retry a failed background job (T47).",
        input_schema={
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
        handler=_t47_job_retry,
    ),
    "asset_view": ToolDefinition(
        name="asset_view",
        description="View thumbnail or contact sheet as an MCP image content block (T48).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "scene_id": {"type": "string"},
                "asset_id": {"type": "string"},
                "kind": {"type": "string", "enum": ["thumbnail", "contact_sheet"], "default": "thumbnail"},
            },
        },
        handler=_t48_asset_view,
    ),
    "asset_download_url": ToolDefinition(
        name="asset_download_url",
        description="Get streaming / download URL for an asset (T49).",
        input_schema={
            "type": "object",
            "properties": {"asset_id": {"type": "string"}},
            "required": ["asset_id"],
        },
        handler=_t49_asset_download_url,
    ),
    "upload_file": ToolDefinition(
        name="upload_file",
        description="Upload an audio, image, or LUT file (T50).",
        input_schema={
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string", "description": "Base64 encoded file content"},
                "kind": {"type": "string", "enum": ["general", "lut"]},
                "project_id": {"type": "string"},
            },
            "required": ["filename"],
        },
        handler=_t50_upload_file,
    ),
}


# Every Agent API endpoint without a tool above becomes an `api_*` tool, and `api_request` reaches any endpoint.
from .endpoint_tools import build_endpoint_tools  # noqa: E402  (needs ToolDefinition and _safe_call defined above)

ALL_TOOLS.update(build_endpoint_tools(ALL_TOOLS))

# MCP tool annotations for the hand-written tools. Clients use them to decide what needs a confirmation: a tool without
# ``readOnlyHint`` may change the project, and ``destructiveHint`` marks one that deletes or overwrites. A tool in neither
# list is only marked as not read-only, so a client treats it with care. The ``api_*`` tools set their own from the method.
_READ_ONLY_TOOLS = (
    "system_health", "list_modes", "list_models", "project_list", "project_get", "project_summary", "project_get_settings",
    "minimax_settings_schema", "scene_list", "scene_get", "references_get", "llm_active", "job_get", "job_wait",
    "asset_view", "asset_download_url", "timeline_notes_list",
)
_DESTRUCTIVE_TOOLS = (
    "project_delete", "scene_delete", "scenes_bulk_edit", "scene_split_merge_move_resize", "job_cancel",
    "timeline_note_delete",
)
for _name, _tool in ALL_TOOLS.items():
    if _tool.annotations is not None or _name.startswith("api_"):
        continue
    if _name in _READ_ONLY_TOOLS:
        _tool.annotations = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
    elif _name in _DESTRUCTIVE_TOOLS:
        _tool.annotations = {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False}
    else:
        _tool.annotations = {"readOnlyHint": False, "openWorldHint": False}
