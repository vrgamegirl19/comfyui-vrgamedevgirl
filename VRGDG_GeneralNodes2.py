import json
import numbers
import os
import threading
import torch
import time

import folder_paths
from aiohttp import web
from nodes import PreviewImage
from server import PromptServer


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")

_VRGDG_TEST_SAVE_ROUTE_REGISTERED = False
_VRGDG_TEST_TEXT_TARGETS = {
    "full_lyrics": ("VRGDG_TEMP", "TextFiles", "fulllyrics", "full_lyrics.txt"),
    "style_theme": ("VRGDG_TEMP", "TextFiles", "themestyle", "themestyle.txt"),
    "story_idea": ("VRGDG_TEMP", "TextFiles", "storyconcept", "storyconcept.txt"),
    "subjects_and_scenes": ("VRGDG_TEMP", "TextFiles", "subjectandscenes", "subjectsandscenes.txt"),
    "text_to_image_notes": ("VRGDG_TEMP", "TextFiles", "t2iNotes", "t2iNotes.txt"),
    "image_to_video_notes": ("VRGDG_TEMP", "TextFiles", "i2vNotes", "i2vNotes.txt"),
}


def _apply_backend_node_action(node_id, action):
    action = str(action or "mute").lower()
    if action == "active":
        PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": True})
    elif action == "bypass":
        PromptServer.instance.send_sync(
            "impact-bridge-continue",
            {"node_id": str(node_id), "bypasses": [str(node_id)], "mutes": [], "actives": []},
        )
    else:
        PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": False})


def _get_workflow_from_extra_pnginfo(extra_pnginfo):
    if not isinstance(extra_pnginfo, list) or not extra_pnginfo:
        return None
    first = extra_pnginfo[0]
    if not isinstance(first, dict):
        return None
    workflow = first.get("workflow")
    return workflow if isinstance(workflow, dict) else None


def _workflow_groups_sorted_alpha(workflow):
    groups = workflow.get("groups", []) if isinstance(workflow, dict) else []
    if not isinstance(groups, list):
        return []
    filtered = [g for g in groups if isinstance(g, dict) and str(g.get("title", "")).strip()]
    filtered.sort(key=lambda g: str(g.get("title", "")).strip().lower())
    return filtered


def _workflow_nodes_for_group(workflow, group):
    if not isinstance(workflow, dict) or not isinstance(group, dict):
        return []
    bounds = group.get("bounding")
    if not isinstance(bounds, list) or len(bounds) < 4:
        return []
    try:
        gx, gy, gw, gh = [float(v) for v in bounds[:4]]
    except Exception:
        return []

    resolved = []
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict):
            continue
        try:
            node_id = int(node.get("id"))
        except Exception:
            continue
        pos = node.get("pos") or [0, 0]
        size = node.get("size") or [140, 80]
        try:
            center_x = float(pos[0]) + float(size[0]) * 0.5
            center_y = float(pos[1]) + float(size[1]) * 0.5
        except Exception:
            continue
        if gx <= center_x < gx + gw and gy <= center_y < gy + gh:
            resolved.append(node_id)
    return resolved


def _resolve_preset_targets(extra_pnginfo, target_specs):
    workflow = _get_workflow_from_extra_pnginfo(extra_pnginfo)
    if workflow is None:
        return []

    groups_sorted = _workflow_groups_sorted_alpha(workflow)
    resolved_targets = []
    for spec in target_specs:
        if not isinstance(spec, dict):
            continue
        slot = spec.get("slot")
        title = str(spec.get("title", "")).strip()
        matched_group = None

        if title:
            matched_group = next(
                (group for group in groups_sorted if str(group.get("title", "")).strip() == title),
                None,
            )

        if matched_group is None:
            try:
                slot_index = int(slot) - 1
            except Exception:
                slot_index = -1
            if 0 <= slot_index < len(groups_sorted):
                matched_group = groups_sorted[slot_index]

        node_ids = _workflow_nodes_for_group(workflow, matched_group) if matched_group is not None else []
        resolved_targets.append(
            {
                "slot": slot,
                "title": title,
                "action": spec.get("action", "mute"),
                "node_ids": node_ids,
            }
        )
    return resolved_targets


def _run_preset_group_state(signal, extra_pnginfo, target_specs, queue_delay=5.0):
    targets = _resolve_preset_targets(extra_pnginfo, target_specs)
    for target in targets:
        for node_id in target.get("node_ids", []):
            _apply_backend_node_action(node_id, target.get("action", "mute"))

    if targets:
        PromptServer.instance.send_sync("vrgdg-apply-node-modes", {"targets": targets})

    def _delayed_queue():
        time.sleep(max(0.0, float(queue_delay or 0.0)))
        PromptServer.instance.send_sync("impact-add-queue", {})

    threading.Thread(target=_delayed_queue, daemon=True).start()
    return (signal,)


def _get_test_popup_audio_dir():
    return os.path.normpath(os.path.join(folder_paths.get_output_directory(), "VRGDG_AudioFiles"))


def _get_test_popup_text_path(field_name):
    parts = _VRGDG_TEST_TEXT_TARGETS[field_name]
    return os.path.normpath(os.path.join(folder_paths.get_output_directory(), *parts))


def _ensure_test_save_route_registered():
    global _VRGDG_TEST_SAVE_ROUTE_REGISTERED
    if _VRGDG_TEST_SAVE_ROUTE_REGISTERED:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.get("/vrgdg/test_popup/config")
    async def vrgdg_test_popup_config(request):
        text_targets = {
            field_name: _get_test_popup_text_path(field_name)
            for field_name in _VRGDG_TEST_TEXT_TARGETS
        }
        return web.json_response(
            {
                "ok": True,
                "audio_dir": _get_test_popup_audio_dir(),
                "text_targets": text_targets,
            }
        )

    @server_instance.routes.post("/vrgdg/test_popup/save_text")
    async def vrgdg_test_popup_save_text(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

        saved_paths = {}
        for field_name in _VRGDG_TEST_TEXT_TARGETS:
            target_path = _get_test_popup_text_path(field_name)
            folder_path = os.path.dirname(target_path)
            text_value = str(payload.get(field_name, "") or "")

            try:
                os.makedirs(folder_path, exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as handle:
                    handle.write(text_value)
            except Exception as exc:
                return web.json_response(
                    {
                        "ok": False,
                        "error": f"Could not write {field_name}: {exc}",
                        "field": field_name,
                    },
                    status=400,
                )

            saved_paths[field_name] = target_path

        return web.json_response({"ok": True, "saved_paths": saved_paths})

    @server_instance.routes.post("/vrgdg/test_popup/upload_audio")
    async def vrgdg_test_popup_upload_audio(request):
        post = await request.post()
        audio = post.get("audio")
        if audio is None or not getattr(audio, "file", None):
            return web.json_response({"ok": False, "error": "Missing audio file."}, status=400)

        filename = os.path.basename(str(getattr(audio, "filename", "") or "").strip())
        if not filename:
            return web.json_response({"ok": False, "error": "Invalid audio filename."}, status=400)

        target_dir = _get_test_popup_audio_dir()
        target_path = os.path.join(target_dir, filename)

        try:
            os.makedirs(target_dir, exist_ok=True)
            for existing_name in os.listdir(target_dir):
                existing_path = os.path.join(target_dir, existing_name)
                if os.path.isfile(existing_path):
                    os.remove(existing_path)
            with open(target_path, "wb") as handle:
                handle.write(audio.file.read())
        except Exception as exc:
            return web.json_response(
                {"ok": False, "error": f"Could not write audio file: {exc}"},
                status=400,
            )

        return web.json_response({"ok": True, "path": target_path, "filename": filename})

    @server_instance.routes.post("/vrgdg/apply_node_modes")
    async def vrgdg_apply_node_modes(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

        targets = payload.get("targets", [])
        if not isinstance(targets, list):
            return web.json_response({"ok": False, "error": "targets must be a list."}, status=400)

        applied = 0
        for target in targets:
            if not isinstance(target, dict):
                continue
            action = target.get("action", "mute")
            node_ids = target.get("node_ids", [])
            if not isinstance(node_ids, list):
                continue
            for raw_id in node_ids:
                try:
                    node_id = int(raw_id)
                except Exception:
                    continue
                if node_id < 0:
                    continue
                _apply_backend_node_action(node_id, action)
                applied += 1

        return web.json_response({"ok": True, "applied": applied})

    _VRGDG_TEST_SAVE_ROUTE_REGISTERED = True


class VRGDG_PromptCreatorUI:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "VRGDG/General"

    def noop(self):
        return ()


class VRGDG_ShowText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "VRGDG/General"

    def notify(self, text=None, unique_id=None, extra_pnginfo=None):
        if text is None:
            text = [""]
        elif not isinstance(text, list):
            text = [text]

        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("VRGDG_ShowText: extra_pnginfo is not a list")
            elif not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]:
                print("VRGDG_ShowText: extra_pnginfo[0] is not a dict or missing 'workflow'")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow.get("nodes", []) if str(x.get("id")) == str(unique_id[0])),
                    None,
                )
                if node is not None:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}


class VRGDG_ShowAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "value": (any_typ, {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "VRGDG/General"

    def _format_value(self, value):
        if isinstance(value, str):
            return value

        try:
            return json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    def notify(self, value=None, unique_id=None, extra_pnginfo=None):
        if value is None:
            value = [None]
        elif not isinstance(value, list):
            value = [value]

        text = [self._format_value(item) for item in value]

        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("VRGDG_ShowAny: extra_pnginfo is not a list")
            elif not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]:
                print("VRGDG_ShowAny: extra_pnginfo[0] is not a dict or missing 'workflow'")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow.get("nodes", []) if str(x.get("id")) == str(unique_id[0])),
                    None,
                )
                if node is not None:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}


class VRGDG_TextBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "output_mode": (["string", "json"], {"default": "string"}),
            }
        }

    RETURN_TYPES = ("STRING", "JSON")
    RETURN_NAMES = ("text_output", "json_output")
    FUNCTION = "output_text"
    CATEGORY = "VRGDG/General"

    def output_text(self, text, output_mode):
        if output_mode == "json":
            try:
                json_output = json.loads(text)
            except Exception as e:
                raise ValueError(f"VRGDG_TextBox: output_mode is 'json' but input is not valid JSON: {e}")
            return (text, json_output)

        return (text, {})


class VRGDG_String2Json:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True, "default": ""}),
                "auto_fix": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("json_output",)
    FUNCTION = "to_json"
    CATEGORY = "VRGDG/General"

    def _basic_cleanup(self, text):
        cleaned = str(text).strip()
        cleaned = cleaned.replace("\ufeff", "").replace("\u200b", "")
        cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
        return cleaned

    def _escape_unescaped_inner_quotes(self, s):
        chars = []
        in_string = False
        escaped = False
        n = len(s)
        i = 0

        while i < n:
            ch = s[i]

            if not in_string:
                chars.append(ch)
                if ch == '"':
                    in_string = True
                    escaped = False
                i += 1
                continue

            # in string
            if escaped:
                chars.append(ch)
                escaped = False
                i += 1
                continue

            if ch == "\\":
                chars.append(ch)
                escaped = True
                i += 1
                continue

            if ch == '"':
                # If this quote is not followed by a valid JSON token boundary,
                # treat it as an inner quote and escape it.
                j = i + 1
                while j < n and s[j].isspace():
                    j += 1
                next_ch = s[j] if j < n else ""
                if next_ch not in [",", "}", "]", ":", ""]:
                    chars.append("\\")
                    chars.append('"')
                    i += 1
                    continue

                chars.append(ch)
                in_string = False
                i += 1
                continue

            chars.append(ch)
            i += 1

        return "".join(chars)

    def _remove_trailing_commas(self, s):
        import re
        return re.sub(r",(\s*[}\]])", r"\1", s)

    def _auto_fix_json_text(self, text):
        cleaned = self._basic_cleanup(text)
        cleaned = self._escape_unescaped_inner_quotes(cleaned)
        cleaned = self._remove_trailing_commas(cleaned)
        return cleaned

    def to_json(self, text, auto_fix=True):
        raw = self._basic_cleanup(text)

        try:
            return (json.loads(raw),)
        except Exception as e:
            if not auto_fix:
                raise ValueError(f"VRGDG_String2Json: invalid JSON input: {e}")

            fixed = self._auto_fix_json_text(raw)
            try:
                return (json.loads(fixed),)
            except Exception as e2:
                raise ValueError(
                    "VRGDG_String2Json: invalid JSON input after auto-fix attempt: "
                    f"{e2}"
                )


class VRGDG_Json2String:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("JSON", {"forceInput": True}),
                "pretty": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "to_string"
    CATEGORY = "VRGDG/General"

    def to_string(self, json_input, pretty=True):
        try:
            if pretty:
                text_output = json.dumps(json_input, indent=2, ensure_ascii=False, default=str)
            else:
                text_output = json.dumps(json_input, separators=(",", ":"), ensure_ascii=False, default=str)
        except Exception:
            text_output = str(json_input)

        return (text_output,)


class VRGDG_ShowImage(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "show_image"
    OUTPUT_NODE = True
    CATEGORY = "VRGDG/General"

    def _is_empty_image(self, image):
        if image is None:
            return True

        if isinstance(image, numbers.Number):
            return image == 0

        if isinstance(image, (list, tuple)):
            return len(image) == 0

        if hasattr(image, "numel"):
            try:
                return image.numel() == 0
            except Exception:
                pass

        if hasattr(image, "shape"):
            try:
                if len(image.shape) == 0:
                    return False
                return image.shape[0] == 0
            except Exception:
                pass

        return False

    def show_image(self, image=None, prompt=None, extra_pnginfo=None):
        if self._is_empty_image(image):
            return {"ui": {"images": []}}

        return self.save_images(
            image,
            filename_prefix="VRGDG_ShowImage",
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )


class VRGDG_BoxIT:
    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "BoxIT", "multiline": False}),
            }
        }

    def run(self, label):
        return ()


class VRGDG_IntToFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "convert"
    CATEGORY = "VRGDG/General"

    def convert(self, value):
        return (float(value),)


class VRGDG_ImageIndex0HUMOEDIT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_index": ("STRING", {"default": "0", "multiline": False}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "build_image"
    CATEGORY = "VRGDG/General"
    DESCRIPTION = "Returns an EmptyImage-style image when the comma-separated index string contains 0."

    def _parse_indices(self, image_index):
        indices = []
        parts = [part.strip() for part in str(image_index or "").replace(";", ",").split(",") if part.strip()]
        for part in parts:
            try:
                value = int(part)
            except ValueError:
                continue
            if value not in indices:
                indices.append(value)
        return indices

    def build_image(self, image_index, width, height):
        indices = self._parse_indices(image_index)
        if 0 not in indices:
            return (None,)

        empty_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
        return (empty_image,)


class VRGDG_NoteBox:
    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "title": ("STRING", {"default": "Note", "multiline": False}),
                "note": (
                    "STRING",
                    {
                        "default": "Write your workflow notes here.",
                        "multiline": True,
                    },
                ),
                "font_size": ("INT", {"default": 18, "min": 12, "max": 120, "step": 1}),
            }
        }

    def run(self, title, note, font_size):
        return ()


class VRGDG_SetMuteStateMulti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (any_typ,),
                "node_ids": ("STRING", {"default": "", "multiline": False}),
                "set_state": ("BOOLEAN", {"default": True, "label_on": "active", "label_off": "mute"}),
                "off_mode": (["mute", "bypass"], {"default": "mute"}),
            }
        }

    FUNCTION = "doit"
    CATEGORY = "VRGDG/General"
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("signal_opt",)
    OUTPUT_NODE = True

    def _parse_node_ids(self, node_ids):
        parsed = []
        parts = [part.strip() for part in str(node_ids or "").replace(";", ",").split(",") if part.strip()]
        for part in parts:
            try:
                value = int(part)
            except ValueError:
                continue
            if value < 0:
                continue
            if value not in parsed:
                parsed.append(value)
        return parsed

    def doit(self, signal, node_ids, set_state, off_mode):
        for node_id in self._parse_node_ids(node_ids):
            if set_state:
                PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": True})
            elif off_mode == "bypass":
                # Reuse Impact Pack bridge event to set a single node to bypass (mode=4).
                PromptServer.instance.send_sync(
                    "impact-bridge-continue",
                    {"node_id": str(node_id), "bypasses": [str(node_id)], "mutes": [], "actives": []},
                )
            else:
                PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": False})
        return (signal,)


class VRGDG_SetGroupStateMulti:
    MAX_GROUP_SLOTS = 12
    NONE_OPTION = "<none>"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "signal": (any_typ,),
            "group_count": ("INT", {"default": 1, "min": 1, "max": cls.MAX_GROUP_SLOTS, "step": 1}),
            # Legacy global fallback action (used only if per-group target map is empty).
            "group_action": (["active", "mute", "bypass"], {"default": "mute"}),
            "auto_queue_next": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
            "queue_delay_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1}),
            # Populated by frontend helper from selected group dropdowns (legacy fallback).
            "node_ids_csv": ("STRING", {"default": ""}),
            # Populated by frontend helper with per-group actions and node ids.
            "group_targets_json": ("STRING", {"default": "[]"}),
        }
        for i in range(1, cls.MAX_GROUP_SLOTS + 1):
            # Must be STRING on backend so dynamic frontend dropdown values validate.
            required[f"group_{i}"] = ("STRING", {"default": cls.NONE_OPTION})
            required[f"group_{i}_action"] = (["active", "mute", "bypass"], {"default": "mute"})
        return {"required": required}

    FUNCTION = "doit"
    CATEGORY = "VRGDG/General"
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("signal_opt",)
    OUTPUT_NODE = True

    def _parse_node_ids(self, node_ids_csv):
        parsed = []
        parts = [part.strip() for part in str(node_ids_csv or "").replace(";", ",").split(",") if part.strip()]
        for part in parts:
            try:
                value = int(part)
            except ValueError:
                continue
            if value < 0:
                continue
            if value not in parsed:
                parsed.append(value)
        return parsed

    def _apply_action(self, node_id, action):
        action = str(action or "mute").lower()
        if action == "active":
            PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": True})
        elif action == "bypass":
            PromptServer.instance.send_sync(
                "impact-bridge-continue",
                {"node_id": str(node_id), "bypasses": [str(node_id)], "mutes": [], "actives": []},
            )
        else:
            PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": False})

    def doit(
        self,
        signal,
        group_count,
        group_action,
        auto_queue_next,
        queue_delay_seconds,
        node_ids_csv,
        group_targets_json,
        **kwargs,
    ):
        applied = False

        # Preferred path: explicit per-group action + node id targets generated by frontend helper.
        try:
            targets = json.loads(str(group_targets_json or "[]"))
        except Exception:
            targets = []

        if isinstance(targets, list):
            for target in targets:
                if not isinstance(target, dict):
                    continue
                action = target.get("action", "mute")
                node_ids = target.get("node_ids", [])
                if not isinstance(node_ids, list):
                    continue
                for raw_id in node_ids:
                    try:
                        node_id = int(raw_id)
                    except Exception:
                        continue
                    if node_id < 0:
                        continue
                    self._apply_action(node_id, action)
                    applied = True

        # Backward fallback: one action for all collected ids.
        if not applied:
            node_ids = self._parse_node_ids(node_ids_csv)
            for node_id in node_ids:
                self._apply_action(node_id, group_action)
                applied = True

        # Apply mode changes on frontend for root+subgraphs (including subgraph contents).
        # Impact Pack events can miss some subgraph internals depending on graph context.
        if isinstance(targets, list) and len(targets) > 0:
            PromptServer.instance.send_sync("vrgdg-apply-node-modes", {"targets": targets})

        # Important: mode changes affect future runs; queue one more run to continue the chain.
        if applied and bool(auto_queue_next):
            delay = max(0.0, float(queue_delay_seconds or 0.0))
            if delay <= 0:
                PromptServer.instance.send_sync("impact-add-queue", {})
            else:
                # Non-blocking delayed queue to allow mode changes to settle first.
                def _delayed_queue():
                    time.sleep(delay)
                    PromptServer.instance.send_sync("impact-add-queue", {})

                threading.Thread(target=_delayed_queue, daemon=True).start()
        return (signal,)


class VRGDG_MuteUnmute4PromptCreatorWF_1(VRGDG_SetGroupStateMulti):
    pass


class VRGDG_MuteUnmute4PromptCreatorWF_2(VRGDG_SetGroupStateMulti):
    pass


_ensure_test_save_route_registered()


class VRGDG_MuteUnmute4PromptCreatorWF_0(VRGDG_SetGroupStateMulti):
    pass


_ensure_test_save_route_registered()

NODE_CLASS_MAPPINGS = {
    "VRGDG_ShowText": VRGDG_ShowText,
    "VRGDG_ShowAny": VRGDG_ShowAny,
    "VRGDG_TextBox": VRGDG_TextBox,
    "VRGDG_String2Json": VRGDG_String2Json,
    "VRGDG_Json2String": VRGDG_Json2String,
    "VRGDG_ShowImage": VRGDG_ShowImage,
    "VRGDG_BoxIT": VRGDG_BoxIT,
    "VRGDG_IntToFloat": VRGDG_IntToFloat,
    "VRGDG_ImageIndex0HUMOEDIT": VRGDG_ImageIndex0HUMOEDIT,
    "VRGDG_NoteBox": VRGDG_NoteBox,
    "VRGDG_SetMuteStateMulti": VRGDG_SetMuteStateMulti,
    "VRGDG_SetGroupStateMulti": VRGDG_SetGroupStateMulti,
    "VRGDG_MuteUnmute4PromptCreatorWF_1": VRGDG_MuteUnmute4PromptCreatorWF_1,
    "VRGDG_MuteUnmute4PromptCreatorWF_2": VRGDG_MuteUnmute4PromptCreatorWF_2,
    "VRGDG_MuteUnmute4PromptCreatorWF_0": VRGDG_MuteUnmute4PromptCreatorWF_0,
    "VRGDG_PromptCreatorUI": VRGDG_PromptCreatorUI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_ShowText": "VRGDG_ShowText",
    "VRGDG_ShowAny": "VRGDG_ShowAny",
    "VRGDG_TextBox": "VRGDG_TextBox",
    "VRGDG_String2Json": "VRGDG_String2Json",
    "VRGDG_Json2String": "VRGDG_Json2String",
    "VRGDG_ShowImage": "VRGDG_ShowImage",
    "VRGDG_BoxIT": "VRGDG_BoxIT",
    "VRGDG_IntToFloat": "VRGDG_IntToFloat",
    "VRGDG_ImageIndex0HUMOEDIT": "VRGDG_ImageIndex0HUMOEDIT",
    "VRGDG_NoteBox": "VRGDG_NoteBox",
    "VRGDG_SetMuteStateMulti": "VRGDG_SetMuteStateMulti",
    "VRGDG_SetGroupStateMulti": "VRGDG_SetGroupStateMulti",
    "VRGDG_MuteUnmute4PromptCreatorWF_1": "VRGDG_MuteUnmute4PromptCreatorWF_1",
    "VRGDG_MuteUnmute4PromptCreatorWF_2": "VRGDG_MuteUnmute4PromptCreatorWF_2",
    "VRGDG_MuteUnmute4PromptCreatorWF_0": "VRGDG_MuteUnmute4PromptCreatorWF_0",
    "VRGDG_PromptCreatorUI": "VRGDG_PromptCreatorUI",
}
