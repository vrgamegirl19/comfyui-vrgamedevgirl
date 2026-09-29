import json

import math

import numbers



import threading

import torch

import time


from typing import List


import comfy


import node_helpers

from aiohttp import web

from nodes import PreviewImage

from server import PromptServer



from .VRGDG_AnyType import any_typ


_VRGDG_TEST_SAVE_ROUTE_REGISTERED = False

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


def _ensure_test_save_route_registered():
    global _VRGDG_TEST_SAVE_ROUTE_REGISTERED
    if _VRGDG_TEST_SAVE_ROUTE_REGISTERED:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

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


class VRGDG_MuteUnmute4PromptCreatorWF_0(VRGDG_SetGroupStateMulti):
    pass


class VRGDG_StoryGroupJsonFixer:
    REQUIRED_GROUP_KEYS = ("index", "subject", "camera", "scene_and_lighting", "frame")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "JSON", "BOOLEAN", "STRING")
    RETURN_NAMES = ("fixed_text", "json_output", "was_fixed", "notes")
    FUNCTION = "fix_json"
    CATEGORY = "VRGDG/General"

    def _strip_markdown_json_fence(self, text):
        value = str(text or "").strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if lines:
                first = lines[0].strip().lower()
                if first == "```" or first.startswith("```json"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                value = "\n".join(lines).strip()
        return value

    def _basic_cleanup(self, text):
        cleaned = self._strip_markdown_json_fence(text)
        cleaned = cleaned.replace("\ufeff", "").replace("\u200b", "")
        cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
        return cleaned.strip()

    def _extract_json_slice(self, text):
        start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
        if not start_candidates:
            return text
        start = min(start_candidates)
        end_obj = text.rfind("}")
        end_arr = text.rfind("]")
        end = max(end_obj, end_arr)
        if end >= start:
            return text[start : end + 1]
        return text[start:]

    def _remove_duplicate_open_braces(self, text):
        chars = []
        in_string = False
        escaped = False
        i = 0
        changes = 0

        while i < len(text):
            ch = text[i]
            if in_string:
                chars.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue

            if ch == '"':
                in_string = True
                chars.append(ch)
                i += 1
                continue

            if ch == "{":
                chars.append(ch)
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and text[j] == "{":
                    changes += 1
                    i = j
                    continue
                i += 1
                continue

            chars.append(ch)
            i += 1

        return "".join(chars), changes

    def _remove_trailing_commas(self, text):
        import re

        updated = re.sub(r",(\s*[}\]])", r"\1", text)
        return updated, int(updated != text)

    def _insert_missing_object_commas(self, text):
        chars = []
        in_string = False
        escaped = False
        changes = 0
        i = 0

        while i < len(text):
            ch = text[i]
            chars.append(ch)
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue

            if ch == '"':
                in_string = True
                i += 1
                continue

            if ch == "}":
                j = i + 1
                whitespace = []
                while j < len(text) and text[j].isspace():
                    whitespace.append(text[j])
                    j += 1
                if j < len(text) and text[j] == "{":
                    chars.extend(whitespace)
                    chars.append(",")
                    changes += 1
                    i = j
                    continue
            i += 1

        return "".join(chars), changes

    def _balance_outer_structure(self, text):
        stripped = text.strip()
        changes = 0
        if stripped.startswith("{") and stripped.count("{") > stripped.count("}"):
            text += "}" * (stripped.count("{") - stripped.count("}"))
            changes += 1
        if stripped.startswith("[") and stripped.count("[") > stripped.count("]"):
            text += "]" * (stripped.count("[") - stripped.count("]"))
            changes += 1
        if '"groups"' in text:
            prefix = text.split('"groups"', 1)[0]
            if prefix.count("[") > prefix.count("]") + 1:
                text += "]" * (prefix.count("[") - prefix.count("]") - 1)
                changes += 1
            if text.count("[") > text.count("]"):
                text += "]" * (text.count("[") - text.count("]"))
                changes += 1
        return text, changes

    def _format_json_error(self, exc, text, label):
        if not isinstance(exc, json.JSONDecodeError):
            return f"{label}: {exc}"

        lines = str(text or "").splitlines()
        context = ""
        if 1 <= exc.lineno <= len(lines):
            line = lines[exc.lineno - 1]
            pointer = " " * max(0, exc.colno - 1) + "^"
            context = f" Line {exc.lineno}, column {exc.colno}:\n{line}\n{pointer}"
        return f"{label}: {exc.msg}.{context}"

    def _validate_story_payload(self, data):
        errors = []
        if not isinstance(data, dict):
            return ["Top-level JSON must be an object with 'story_summary' and 'groups'."]

        if "story_summary" not in data:
            errors.append("Missing top-level key 'story_summary'.")
        elif not isinstance(data.get("story_summary"), str):
            errors.append("'story_summary' must be a string.")

        if "groups" not in data:
            errors.append("Missing top-level key 'groups'.")
            return errors

        groups = data.get("groups")
        if not isinstance(groups, list):
            errors.append("'groups' must be a list.")
            return errors

        seen_indexes = set()
        for pos, group in enumerate(groups, start=1):
            if not isinstance(group, dict):
                errors.append(f"groups[{pos}] must be an object.")
                continue

            missing = [key for key in self.REQUIRED_GROUP_KEYS if key not in group]
            if missing:
                errors.append(f"groups[{pos}] is missing keys: {', '.join(missing)}.")

            if "index" in group:
                try:
                    index_value = int(group.get("index"))
                    if index_value <= 0:
                        errors.append(f"groups[{pos}].index must be greater than 0.")
                    elif index_value in seen_indexes:
                        errors.append(f"Duplicate group index {index_value}.")
                    else:
                        seen_indexes.add(index_value)
                except Exception:
                    errors.append(f"groups[{pos}].index must be an integer.")

            for key in self.REQUIRED_GROUP_KEYS[1:]:
                if key in group and not isinstance(group.get(key), str):
                    errors.append(f"groups[{pos}].{key} must be a string.")

        return errors

    def _normalize_group(self, item, fallback_index):
        if not isinstance(item, dict):
            item = {}

        normalized = {}
        raw_index = item.get("index", fallback_index)
        try:
            normalized["index"] = int(raw_index)
        except Exception:
            normalized["index"] = fallback_index

        for key in self.REQUIRED_GROUP_KEYS[1:]:
            value = item.get(key, "")
            if value is None:
                value = ""
            elif not isinstance(value, str):
                value = str(value)
            normalized[key] = value
        return normalized

    def _normalize_story_payload(self, data):
        validation_errors = self._validate_story_payload(data)
        if validation_errors:
            raise ValueError(" ".join(validation_errors))

        story_summary = data.get("story_summary", "")
        groups = data.get("groups", [])

        normalized_groups = []
        for idx, group in enumerate(groups, start=1):
            normalized_groups.append(self._normalize_group(group, idx))

        normalized_groups.sort(key=lambda item: item.get("index", 0))
        for idx, group in enumerate(normalized_groups, start=1):
            if group.get("index") <= 0:
                group["index"] = idx

        return {
            "story_summary": story_summary,
            "groups": normalized_groups,
        }

    def _parse_json_preserving_order(self, text):
        return json.loads(text)

    def _repair_schema_text(self, text):
        notes = []
        working = self._basic_cleanup(text)
        sliced = self._extract_json_slice(working)
        if sliced != working:
            notes.append("trimmed extra text outside JSON")
            working = sliced

        working, duplicate_count = self._remove_duplicate_open_braces(working)
        if duplicate_count:
            notes.append(f"removed duplicate '{{' x{duplicate_count}")

        working, comma_cleanup = self._remove_trailing_commas(working)
        if comma_cleanup:
            notes.append("removed trailing commas")

        working, inserted_commas = self._insert_missing_object_commas(working)
        if inserted_commas:
            notes.append(f"inserted missing commas between objects x{inserted_commas}")

        working, balance_changes = self._balance_outer_structure(working)
        if balance_changes:
            notes.append("balanced closing brackets/braces")

        return working, notes

    def fix_json(self, text):
        original = self._basic_cleanup(text)
        notes = []

        try:
            parsed = self._parse_json_preserving_order(original)
        except json.JSONDecodeError as exc:
            repaired_text, notes = self._repair_schema_text(text)
            try:
                parsed = self._parse_json_preserving_order(repaired_text)
            except json.JSONDecodeError as repaired_exc:
                original_error = self._format_json_error(exc, original, "Original JSON parse failed")
                repaired_error = self._format_json_error(repaired_exc, repaired_text, "Repair attempt still invalid")
                raise ValueError(f"VRGDG_StoryGroupJsonFixer: {original_error}\n{repaired_error}")
        else:
            repaired_text = original

        try:
            normalized = self._normalize_story_payload(parsed)
        except ValueError as exc:
            raise ValueError(f"VRGDG_StoryGroupJsonFixer schema error: {exc}")
        fixed_text = json.dumps(normalized, indent=2, ensure_ascii=False)
        was_fixed = bool(notes) or fixed_text.strip() != original.strip()
        note_text = "; ".join(notes) if notes else ("normalized formatting" if was_fixed else "")
        return (fixed_text, normalized, was_fixed, note_text)


class VRGDG_MultiReferenceConditioning:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    MAX_IMAGES = 50

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"image{i}": ("IMAGE", {"tooltip": f"Optional reference image {i}."})
            for i in range(1, cls.MAX_IMAGES + 1)
        }
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image_count": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": cls.MAX_IMAGES,
                        "step": 1,
                        "tooltip": "How many image inputs to show and process.",
                    },
                ),
                "upscale_method": (cls.upscale_methods, {"default": "nearest-exact"}),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 16.0,
                        "step": 0.01,
                    },
                ),
                "resolution_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "IMAGE")
    FUNCTION = "apply"
    CATEGORY = "VRGDG/Conditioning"
    DESCRIPTION = (
        "Dynamically scales each connected reference image, VAE encodes it, "
        "and appends the resulting reference latent to positive and negative conditioning."
    )

    @classmethod
    def _scale_to_total_pixels(cls, image, upscale_method, megapixels, resolution_steps):
        samples = image.movedim(-1, 1)
        total = float(megapixels) * 1024 * 1024
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        steps = max(1, int(resolution_steps))
        width = max(1, round(samples.shape[3] * scale_by / steps) * steps)
        height = max(1, round(samples.shape[2] * scale_by / steps) * steps)
        scaled = comfy.utils.common_upscale(
            samples,
            int(width),
            int(height),
            upscale_method,
            "disabled",
        )
        return scaled.movedim(1, -1)

    @staticmethod
    def _append_reference_latent(conditioning, latent):
        return node_helpers.conditioning_set_values(
            conditioning,
            {"reference_latents": [latent["samples"]]},
            append=True,
        )

    @staticmethod
    def _batch_for_image_output(images: List[torch.Tensor]):
        if not images:
            raise ValueError("VRGDG Multi Reference Conditioning needs at least one connected image input.")
        if len(images) == 1:
            return images[0]

        base = images[0]
        batched = [base]
        for image in images[1:]:
            next_image = image
            if next_image.shape[-1] != base.shape[-1]:
                max_channels = max(next_image.shape[-1], base.shape[-1])
                if base.shape[-1] < max_channels:
                    base = torch.nn.functional.pad(base, (0, max_channels - base.shape[-1]), value=1.0)
                    batched[0] = base
                if next_image.shape[-1] < max_channels:
                    next_image = torch.nn.functional.pad(next_image, (0, max_channels - next_image.shape[-1]), value=1.0)
            if next_image.shape[1:] != base.shape[1:]:
                next_image = comfy.utils.common_upscale(
                    next_image.movedim(-1, 1),
                    base.shape[2],
                    base.shape[1],
                    "bilinear",
                    "center",
                ).movedim(1, -1)
            batched.append(next_image)
        return torch.cat(batched, dim=0)

    def apply(self, positive, negative, vae, image_count, upscale_method, megapixels, resolution_steps, **kwargs):
        count = max(1, min(self.MAX_IMAGES, int(image_count)))
        positive_out = positive
        negative_out = negative
        scaled_images = []

        for index in range(1, count + 1):
            image = kwargs.get(f"image{index}")
            if image is None:
                continue

            scaled_image = self._scale_to_total_pixels(image, upscale_method, megapixels, resolution_steps)
            latent = {"samples": vae.encode(scaled_image)}
            positive_out = self._append_reference_latent(positive_out, latent)
            negative_out = self._append_reference_latent(negative_out, latent)
            scaled_images.append(scaled_image)

        return (positive_out, negative_out, self._batch_for_image_output(scaled_images))


_ensure_test_save_route_registered()


NODE_CLASS_MAPPINGS = {
    "VRGDG_String2Json": VRGDG_String2Json,
    "VRGDG_Json2String": VRGDG_Json2String,
    "VRGDG_ShowImage": VRGDG_ShowImage,
    "VRGDG_BoxIT": VRGDG_BoxIT,
    "VRGDG_NoteBox": VRGDG_NoteBox,
    "VRGDG_SetMuteStateMulti": VRGDG_SetMuteStateMulti,
    "VRGDG_SetGroupStateMulti": VRGDG_SetGroupStateMulti,
    "VRGDG_MuteUnmute4PromptCreatorWF_1": VRGDG_MuteUnmute4PromptCreatorWF_1,
    "VRGDG_MuteUnmute4PromptCreatorWF_2": VRGDG_MuteUnmute4PromptCreatorWF_2,
    "VRGDG_MuteUnmute4PromptCreatorWF_0": VRGDG_MuteUnmute4PromptCreatorWF_0,
    "VRGDG_StoryGroupJsonFixer": VRGDG_StoryGroupJsonFixer,
    "VRGDG_MultiReferenceConditioning": VRGDG_MultiReferenceConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_String2Json": "VRGDG_String2Json",
    "VRGDG_Json2String": "VRGDG_Json2String",
    "VRGDG_ShowImage": "VRGDG_ShowImage",
    "VRGDG_BoxIT": "VRGDG_BoxIT",
    "VRGDG_NoteBox": "VRGDG_NoteBox",
    "VRGDG_SetMuteStateMulti": "VRGDG_SetMuteStateMulti",
    "VRGDG_SetGroupStateMulti": "VRGDG_SetGroupStateMulti",
    "VRGDG_MuteUnmute4PromptCreatorWF_1": "VRGDG_MuteUnmute4PromptCreatorWF_1",
    "VRGDG_MuteUnmute4PromptCreatorWF_2": "VRGDG_MuteUnmute4PromptCreatorWF_2",
    "VRGDG_MuteUnmute4PromptCreatorWF_0": "VRGDG_MuteUnmute4PromptCreatorWF_0",
    "VRGDG_StoryGroupJsonFixer": "VRGDG_StoryGroupJsonFixer",
    "VRGDG_MultiReferenceConditioning": "VRGDG Multi Reference Conditioning",
}
