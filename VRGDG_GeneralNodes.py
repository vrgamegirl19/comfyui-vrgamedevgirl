import ast
import json
import math
import os
import re
import shutil
import sys
import time

import folder_paths
from aiohttp import web
from server import PromptServer

IMAGE2VIDEO_BATCH_FOLDER_PREFIX = "Image2Video_Batch_"
TEXT2IMAGE_BATCH_FOLDER_PREFIX = "Text2Image_Batch_"
LLM_BATCHES_FOLDER_NAME = "llm_batches"
COMBINED_JSON_SUFFIX = "_COMBINED.json"
EMPTY_COMBINED_JSON_OPTION = "<no files found>"
BATCH_TYPE_TEXT2IMAGE = "Text2Image"
BATCH_TYPE_IMAGE2VIDEO = "Image2Video"
MAX_COMBINED_JSON_PROMPT_EDIT_SLOTS = 20


def _get_llm_batches_root():
    return os.path.normpath(
        os.path.join(folder_paths.get_output_directory(), LLM_BATCHES_FOLDER_NAME)
    )


def _find_latest_batch_folder(prefix=None):
    root = _get_llm_batches_root()
    if not os.path.isdir(root):
        return None

    latest_path = None
    latest_mtime = -1.0

    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if prefix and not name.startswith(prefix):
            continue

        try:
            mtime = os.path.getmtime(full)
        except OSError:
            continue

        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = full

    return latest_path


def _normalize_batch_type(batch_type):
    t = str(batch_type or "").strip()
    if t == BATCH_TYPE_IMAGE2VIDEO:
        return BATCH_TYPE_IMAGE2VIDEO
    return BATCH_TYPE_TEXT2IMAGE


def _batch_prefix_for_type(batch_type):
    batch_type = _normalize_batch_type(batch_type)
    if batch_type == BATCH_TYPE_IMAGE2VIDEO:
        return IMAGE2VIDEO_BATCH_FOLDER_PREFIX
    return TEXT2IMAGE_BATCH_FOLDER_PREFIX


def _list_latest_combined_json_files(batch_type=BATCH_TYPE_TEXT2IMAGE):
    prefix = _batch_prefix_for_type(batch_type)
    latest_folder = _find_latest_batch_folder(prefix)
    if not latest_folder:
        return [], None

    files = []
    for name in os.listdir(latest_folder):
        full = os.path.join(latest_folder, name)
        if not os.path.isfile(full):
            continue
        if name.endswith(COMBINED_JSON_SUFFIX):
            files.append(name)

    files.sort(key=str.lower)
    return files, latest_folder


def _resolve_latest_combined_json_file_path(batch_type, combined_json_file):
    selected = os.path.basename(str(combined_json_file or "").strip())
    if not selected or selected == EMPTY_COMBINED_JSON_OPTION:
        return None, "No combined JSON file selected."

    batch_type = _normalize_batch_type(batch_type)
    files, latest_folder = _list_latest_combined_json_files(batch_type)
    if not latest_folder:
        return None, f"No latest {batch_type} batch folder found."
    if selected not in files:
        return None, f"Selected file not found in latest {batch_type} batch folder."

    file_path = os.path.normpath(os.path.join(latest_folder, selected))
    if not os.path.isfile(file_path):
        return None, "Selected combined JSON file does not exist on disk."
    return file_path, ""


def _read_text_with_utf8_fallback(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return f.read()


def _load_combined_json_object(file_path):
    raw = _read_text_with_utf8_fallback(file_path) or ""
    parsed = json.loads(raw) if raw.strip() else {}
    if not isinstance(parsed, dict):
        raise ValueError("Combined JSON must be a JSON object.")
    return parsed


def _write_combined_json_object(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _parse_prompt_number_from_key(key):
    m = re.match(r"^prompt(\d+)$", str(key or ""), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except Exception:
        return None
    return n if n > 0 else None


def _normalize_image_index_list(value):
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out
    return []


def _parse_image_index_input(raw):
    if raw is None:
        return False, []
    if isinstance(raw, list):
        return True, _normalize_image_index_list(raw)

    text = str(raw).strip()
    if not text:
        return True, []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return True, _normalize_image_index_list(parsed)
    except Exception:
        pass

    if "," in text:
        parts = [p.strip() for p in text.split(",")]
    else:
        parts = [text]

    out = []
    for part in parts:
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return True, out


def _extract_prompt_rows_for_ui(data, max_items=MAX_COMBINED_JSON_PROMPT_EDIT_SLOTS):
    rows = []
    if not isinstance(data, dict):
        return rows

    for key, value in data.items():
        prompt_number = _parse_prompt_number_from_key(key)
        if prompt_number is None:
            continue

        prompt_text = ""
        image_index = []
        if isinstance(value, dict):
            if "text" in value:
                prompt_text = value.get("text", "")
                if prompt_text is None:
                    prompt_text = ""
                elif not isinstance(prompt_text, str):
                    prompt_text = str(prompt_text)
            else:
                try:
                    prompt_text = json.dumps(value, ensure_ascii=False, indent=2)
                except Exception:
                    prompt_text = str(value)
            image_index = _normalize_image_index_list(value.get("imageIndex"))
        else:
            prompt_text = str(value if value is not None else "")

        rows.append(
            {
                "prompt_number": prompt_number,
                "prompt": prompt_text,
                "image_index": image_index,
            }
        )

    rows.sort(key=lambda r: r["prompt_number"])
    return rows[:max_items]


def _coerce_prompt_updates(raw_updates, max_items=MAX_COMBINED_JSON_PROMPT_EDIT_SLOTS):
    rows = []
    if not isinstance(raw_updates, list):
        return rows

    for item in raw_updates:
        if not isinstance(item, dict):
            continue

        try:
            prompt_number = int(item.get("prompt_number"))
        except Exception:
            continue

        if prompt_number <= 0:
            continue

        prompt_text = item.get("prompt", "")
        if prompt_text is None:
            prompt_text = ""
        elif not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)

        has_image_index, image_index = _parse_image_index_input(item.get("image_index"))

        rows.append(
            {
                "prompt_number": prompt_number,
                "prompt": prompt_text,
                "has_image_index": has_image_index,
                "image_index": image_index,
            }
        )
        if len(rows) >= max_items:
            break

    return rows


def _apply_prompt_updates_to_data(data, updates, batch_type=BATCH_TYPE_TEXT2IMAGE):
    changed_count = 0
    updated_keys = []
    batch_type = _normalize_batch_type(batch_type)
    is_text2image = batch_type == BATCH_TYPE_TEXT2IMAGE

    for item in updates:
        prompt_number = item.get("prompt_number")
        prompt_text = item.get("prompt", "")
        has_image_index = bool(item.get("has_image_index", False))
        image_index = item.get("image_index", [])
        key = f"prompt{prompt_number}"
        old = data.get(key)

        if isinstance(old, dict):
            old_text = old.get("text")
            if old_text != prompt_text:
                old["text"] = prompt_text
                changed_count += 1
            if is_text2image and has_image_index:
                old_image_index = _normalize_image_index_list(old.get("imageIndex"))
                if old_image_index != image_index:
                    old["imageIndex"] = image_index
                    changed_count += 1
        else:
            if is_text2image:
                next_value = {"text": prompt_text}
                if has_image_index:
                    next_value["imageIndex"] = image_index
                if old != next_value:
                    data[key] = next_value
                    changed_count += 1
            else:
                if old != prompt_text:
                    data[key] = prompt_text
                    changed_count += 1

        updated_keys.append(key)

    return changed_count, updated_keys


_VRGDG_COMBINED_ROUTE_REGISTERED = False


def _ensure_combined_files_route_registered():
    global _VRGDG_COMBINED_ROUTE_REGISTERED
    if _VRGDG_COMBINED_ROUTE_REGISTERED:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.get("/vrgdg/llm_batches/combined_files")
    async def vrgdg_list_combined_files(request):
        batch_type = request.query.get("batch_type", BATCH_TYPE_TEXT2IMAGE)
        batch_type = _normalize_batch_type(batch_type)
        files, latest_folder = _list_latest_combined_json_files(batch_type)
        return web.json_response(
            {
                "batch_type": batch_type,
                "files": files,
                "latest_folder": latest_folder or "",
            }
        )

    @server_instance.routes.get("/vrgdg/llm_batches/combined_file_prompt_values")
    async def vrgdg_get_combined_file_prompt_values(request):
        batch_type = _normalize_batch_type(request.query.get("batch_type", BATCH_TYPE_TEXT2IMAGE))
        combined_json_file = request.query.get("combined_json_file", "")

        file_path, resolve_error = _resolve_latest_combined_json_file_path(
            batch_type, combined_json_file
        )
        if not file_path:
            return web.json_response(
                {"ok": False, "error": resolve_error or "Unable to resolve target file."},
                status=400,
            )

        try:
            data = _load_combined_json_object(file_path)
        except Exception as e:
            return web.json_response(
                {"ok": False, "error": f"Failed to parse combined JSON: {type(e).__name__}: {e}"},
                status=400,
            )

        rows = _extract_prompt_rows_for_ui(data)
        return web.json_response(
            {
                "ok": True,
                "batch_type": batch_type,
                "file_path": file_path,
                "prompt_count": len(rows),
                "prompts": rows,
            }
        )

    @server_instance.routes.post("/vrgdg/llm_batches/combined_file_update_prompts")
    async def vrgdg_update_combined_file_prompts(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "Invalid JSON body."},
                status=400,
            )

        remake_mode = _normalize_bool(payload.get("remake_mode", False))
        batch_type = _normalize_batch_type(payload.get("batch_type", BATCH_TYPE_TEXT2IMAGE))
        use_plain_text = _normalize_bool(payload.get("use_plain_text", False))
        combined_json_file = payload.get("combined_json_file", "")
        updates = _coerce_prompt_updates(payload.get("updates", []))

        if not remake_mode:
            return web.json_response(
                {
                    "ok": True,
                    "ignored": True,
                    "updated": 0,
                    "updated_keys": [],
                    "file_path": "",
                    "message": "Remake mode is disabled; update ignored.",
                }
            )

        if not updates:
            return web.json_response(
                {"ok": False, "error": "No valid prompt updates were provided."},
                status=400,
            )

        file_path, resolve_error = _resolve_latest_combined_json_file_path(
            batch_type, combined_json_file
        )
        if not file_path:
            return web.json_response(
                {"ok": False, "error": resolve_error or "Unable to resolve target file."},
                status=400,
            )

        try:
            data = _load_combined_json_object(file_path)
        except Exception as e:
            return web.json_response(
                {"ok": False, "error": f"Failed to parse combined JSON: {type(e).__name__}: {e}"},
                status=400,
            )

        apply_batch_type = BATCH_TYPE_IMAGE2VIDEO if use_plain_text else batch_type
        changed_count, updated_keys = _apply_prompt_updates_to_data(
            data, updates, batch_type=apply_batch_type
        )

        try:
            _write_combined_json_object(file_path, data)
        except Exception as e:
            return web.json_response(
                {"ok": False, "error": f"Failed to write combined JSON: {type(e).__name__}: {e}"},
                status=500,
            )

        return web.json_response(
            {
                "ok": True,
                "ignored": False,
                "updated": changed_count,
                "updated_keys": updated_keys,
                "file_path": file_path,
            }
        )

    _VRGDG_COMBINED_ROUTE_REGISTERED = True


_ensure_combined_files_route_registered()


class VRGDG_GeneralPromptBatcher:
    RETURN_TYPES = (
        "STRING",
        "INT",
        "INT",
        "BOOLEAN",
        "STRING",
        "STRING",
    )

    RETURN_NAMES = (
        "prompts",
        "batch_index",
        "total_batches",
        "is_final_batch",
        "output_folder",
        "file_prefix",
    )

    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("INT", {"forceInput": True}),
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 9999}),
                "file_path": (
                    "STRING",
                    {
                        "default": "general_prompt_batches",
                        "placeholder": "Relative to ComfyUI/output OR full path",
                    },
                ),
                "file_prefix": ("STRING", {"default": "Batch"}),
                "enable_auto_queue": ("BOOLEAN", {"default": True}),
                "input_1": ("STRING", {"multiline": True, "forceInput": True}),
            },
            "optional": {
                "global_input_1": ("STRING", {"multiline": True, "forceInput": True}),
                "global_input_2": ("STRING", {"multiline": True, "forceInput": True}),
                "input_2": ("STRING", {"multiline": True, "forceInput": True}),
                "input_3": ("STRING", {"multiline": True, "forceInput": True}),
                "input_4": ("STRING", {"multiline": True, "forceInput": True}),
            },
        }

    def _resolve_output_path(self, file_path):
        base_output = folder_paths.get_output_directory()
        if os.path.isabs(file_path):
            return os.path.normpath(file_path)
        return os.path.normpath(os.path.join(base_output, file_path))

    def _find_latest_batch_folder(self, root_folder):
        if not os.path.isdir(root_folder):
            return None

        highest_num = -1
        highest_path = None

        for name in os.listdir(root_folder):
            full = os.path.join(root_folder, name)
            if not os.path.isdir(full):
                continue
            if not name.startswith(IMAGE2VIDEO_BATCH_FOLDER_PREFIX):
                continue
            suffix = name[len(IMAGE2VIDEO_BATCH_FOLDER_PREFIX) :]
            if not suffix.isdigit():
                continue

            n = int(suffix)
            if n > highest_num:
                highest_num = n
                highest_path = full

        return highest_path

    def _is_unfinished_batch_folder(self, folder, file_prefix):
        if not os.path.isdir(folder):
            return False

        combined_name = f"{file_prefix}_COMBINED.json"
        if os.path.isfile(os.path.join(folder, combined_name)):
            return False

        prefix = f"{file_prefix}_"
        for fname in os.listdir(folder):
            if (
                fname.startswith(prefix)
                and fname.lower().endswith(".txt")
                and "COMBINED" not in fname
            ):
                return True
        return False

    def _create_next_batch_folder(self, root_folder):
        os.makedirs(root_folder, exist_ok=True)
        next_num = 1

        while True:
            candidate = os.path.join(
                root_folder, f"{IMAGE2VIDEO_BATCH_FOLDER_PREFIX}{next_num:03d}"
            )
            if not os.path.exists(candidate):
                os.makedirs(candidate, exist_ok=True)
                return candidate
            next_num += 1

    def _extract_index(self, text, loose=False):
        if text is None:
            return None
        s = str(text)
        m = re.search(
            r'(?i)^\s*["\']?(?:lyricsegment|prompt|segment|group|index)\s*[_#:\-\s]*([0-9]+)',
            s,
        )
        if m:
            return int(m.group(1))
        # Optional relaxed mode for dictionary keys/ids where only a number may exist.
        if loose:
            m = re.search(r"\b([0-9]+)\b", s)
            if m:
                return int(m.group(1))
        return None

    def _parse_json_to_groups(self, data):
        if isinstance(data, list):
            out = {}
            for i, item in enumerate(data, start=1):
                if isinstance(item, dict):
                    idx = self._extract_index(item.get("index"), loose=True)
                    if idx is None:
                        idx = self._extract_index(item.get("id"), loose=True)
                    if idx is None:
                        idx = self._extract_index(item.get("name"), loose=True)
                    if idx is None:
                        idx = i
                    out[idx] = json.dumps(item, ensure_ascii=False, indent=2)
                else:
                    out[i] = str(item).strip()
            return {k: v for k, v in out.items() if v}

        if isinstance(data, dict):
            for k in ("groups", "items", "prompts", "segments", "lines"):
                if isinstance(data.get(k), list):
                    return self._parse_json_to_groups(data[k])

            out = {}
            seq = 1
            for key, value in data.items():
                idx = self._extract_index(key, loose=True)
                if idx is None and isinstance(value, dict):
                    idx = self._extract_index(value.get("index"), loose=True)
                    if idx is None:
                        idx = self._extract_index(value.get("id"), loose=True)
                    if idx is None:
                        idx = self._extract_index(value.get("name"), loose=True)
                if idx is None:
                    while seq in out:
                        seq += 1
                    idx = seq

                if isinstance(value, (dict, list)):
                    text = json.dumps(value, ensure_ascii=False, indent=2)
                else:
                    text = str(value).strip()

                if text:
                    out[idx] = text
            return out

        return {}

    def _extract_line_group_index(self, line):
        if line is None:
            return None
        s = str(line)
        # Explicit labels: index/prompt/segment/group/lyricsegment...
        idx = self._extract_index(s, loose=False)
        if idx is not None:
            return idx

        # Numbered list/group starters only at line start (e.g., "1:", "2.", "#3 -")
        m = re.search(r"^\s*#?\s*([0-9]+)\s*[:.)-]\s*", s)
        if m:
            return int(m.group(1))

        return None

    def _parse_plain_text_to_groups(self, text):
        lines = text.splitlines()
        out = {}
        current_idx = None
        pending_object_open = False

        for line in lines:
            raw = line.rstrip()
            if not raw.strip():
                continue
            stripped = raw.strip()
            if stripped == "{":
                pending_object_open = True
                continue
            idx = self._extract_line_group_index(raw)
            if idx is not None:
                current_idx = idx
                out.setdefault(current_idx, [])
                if pending_object_open:
                    out[current_idx].append("{")
                    pending_object_open = False
                out[current_idx].append(raw)
            elif current_idx is not None:
                out[current_idx].append(raw)

        if out:
            return {k: "\n".join(v).strip() for k, v in out.items() if v}

        blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
        if not blocks:
            return {}
        if len(blocks) == 1:
            blocks = [l.strip() for l in text.splitlines() if l.strip()]
        filtered = []
        for b in blocks:
            if b in ("[", "]", "{", "}", "],", "},"):
                continue
            filtered.append(b)
        return {i + 1: b for i, b in enumerate(filtered)}

    def _extract_groups_array_text(self, text):
        if not isinstance(text, str):
            return None
        m = re.search(r'(?i)"groups"\s*:\s*\[', text)
        if not m:
            return None

        # Start at "[" after "groups":
        start = text.find("[", m.start())
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text[start:]

    def _parse_groups_array_fallback(self, groups_text):
        if not isinstance(groups_text, str):
            return {}

        # Extract top-level {...} objects from inside the groups array text.
        objects = []
        depth = 0
        obj_start = None
        in_string = False
        escaped = False

        for i, ch in enumerate(groups_text):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        objects.append(groups_text[obj_start : i + 1])
                        obj_start = None

        out = {}
        seq = 1
        for obj_text in objects:
            idx = None
            text_value = None

            # Try clean JSON object first.
            try:
                obj = json.loads(obj_text)
                idx = self._extract_index(obj.get("index"), loose=True)
                if idx is None:
                    idx = seq
                text_value = json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                # Fallback for near-JSON text objects.
                m = re.search(r'(?i)"index"\s*:\s*([0-9]+)', obj_text)
                if m:
                    idx = int(m.group(1))
                if idx is None:
                    idx = seq
                text_value = obj_text.strip()

            if text_value:
                out[idx] = text_value
                seq += 1

        return out

    def _parse_input_groups(self, value):
        if not isinstance(value, str):
            return {}
        cleaned = value.strip()
        if not cleaned:
            return {}

        # Prefer explicit "groups" payload if present; ignore wrapper keys like story_summary.
        groups_array_text = self._extract_groups_array_text(cleaned)
        if groups_array_text:
            try:
                groups_data = json.loads(groups_array_text)
                return self._parse_json_to_groups(groups_data)
            except Exception:
                fallback_groups = self._parse_groups_array_fallback(groups_array_text)
                if fallback_groups:
                    return fallback_groups
                # Strict groups-mode fallback: parse only inside groups array,
                # never from wrapper text (prevents story_summary leakage).
                return self._parse_plain_text_to_groups(groups_array_text)

        if cleaned.startswith("{") or cleaned.startswith("["):
            try:
                return self._parse_json_to_groups(json.loads(cleaned))
            except Exception:
                pass
        return self._parse_plain_text_to_groups(cleaned)

    def _slice(self, items, batch_index, batch_size):
        start = batch_index * batch_size
        end = start + batch_size
        return items[start:end]

    def _next_batch_index(self, output_path, file_prefix):
        if not os.path.isdir(output_path):
            return 0

        highest = -1
        pattern = re.compile(rf"^{re.escape(file_prefix)}_(\d+)(?:\..+)?$")
        for fname in os.listdir(output_path):
            full = os.path.join(output_path, fname)
            if not os.path.isfile(full):
                continue
            m = pattern.match(fname)
            if not m:
                continue
            idx = int(m.group(1))
            if idx > highest:
                highest = idx

        return highest + 1

    def _maybe_auto_queue(self, total_batches, batch_index, enable_auto_queue):
        if not enable_auto_queue:
            return
        if batch_index != 0:
            return
        for _ in range(max(0, total_batches - 1)):
            PromptServer.instance.send_sync("impact-add-queue", {})

    def _send_popup_notification(self, message, message_type="info", title="Prompt Batch Instructions"):
        try:
            PromptServer.instance.send_sync(
                "vrgdg_instructions_popup",
                {"message": message, "type": message_type, "title": title},
            )
        except Exception as e:
            print(f"[Popup] Failed: {e}")

    def _build_prompt(self, batch_indices, grouped_inputs, global_input_1, global_input_2):
        sections = []

        if isinstance(global_input_1, str) and global_input_1.strip():
            sections.append(global_input_1.strip())
        if isinstance(global_input_2, str) and global_input_2.strip():
            sections.append(global_input_2.strip())

        for idx in batch_indices:
            parts = [f"### Group {idx}"]
            for input_name in ("input_1", "input_2", "input_3", "input_4"):
                value = grouped_inputs[input_name].get(idx)
                if value:
                    parts.append(f"{input_name}:\n{value}")
            sections.append("\n\n".join(parts))

        return "\n\n".join(sections).strip()

    def run(
        self,
        trigger,
        batch_size,
        file_path,
        file_prefix,
        enable_auto_queue,
        input_1,
        global_input_1=None,
        global_input_2=None,
        input_2=None,
        input_3=None,
        input_4=None,
    ):
        print("========== VRGDG General Prompt Batcher ==========")
        print("Trigger value:", trigger)

        base_output = folder_paths.get_output_directory()
        llm_batches_root = os.path.normpath(os.path.join(base_output, "llm_batches"))
        os.makedirs(llm_batches_root, exist_ok=True)

        latest_batch_folder = self._find_latest_batch_folder(llm_batches_root)
        if latest_batch_folder and self._is_unfinished_batch_folder(
            latest_batch_folder, file_prefix
        ):
            output_path = latest_batch_folder
            print("Reusing unfinished batch folder:", output_path)
        else:
            output_path = self._create_next_batch_folder(llm_batches_root)
            print("Created new batch folder:", output_path)

        os.makedirs(output_path, exist_ok=True)

        grouped_inputs = {
            "input_1": self._parse_input_groups(input_1),
            "input_2": self._parse_input_groups(input_2),
            "input_3": self._parse_input_groups(input_3),
            "input_4": self._parse_input_groups(input_4),
        }

        all_indices = sorted({k for d in grouped_inputs.values() for k in d.keys()})
        if not all_indices:
            raise ValueError("No grouped data found in inputs.")

        total_items = len(all_indices)
        total_batches = max(1, math.ceil(total_items / batch_size))
        batch_index = self._next_batch_index(output_path, file_prefix)
        is_final_batch = (batch_index + 1) >= total_batches

        if total_batches <= 1:
            instructions = "✅ 1 prompt batch required. Running now."
        elif batch_index == 0:
            if enable_auto_queue:
                instructions = (
                    f"⚠️ {total_batches} prompt batches required\n"
                    f"✅ Auto-queuing remaining {total_batches - 1} batch(es)"
                )
            else:
                instructions = (
                    f"⚠️ {total_batches} prompt batches required\n"
                    f"🔴 Auto-queue is DISABLED — run each batch manually"
                )
        elif is_final_batch:
            instructions = f"🏁 Final prompt batch ({batch_index + 1} of {total_batches})"
        else:
            instructions = f"⏳ Prompt batch {batch_index + 1} of {total_batches} in progress"

        batch_indices = self._slice(all_indices, batch_index, batch_size)
        prompts = self._build_prompt(batch_indices, grouped_inputs, global_input_1, global_input_2)

        if batch_index == 0:
            self._send_popup_notification(
                instructions,
                "info",
                "🧠 LLM Prompt Batching Started",
            )
        elif is_final_batch:
            self._send_popup_notification(
                instructions,
                "green",
                "🏁 LLM Final Prompt Batching, then it will be Complete",
            )
        else:
            self._send_popup_notification(
                instructions,
                "yellow",
                "⏳ LLM Prompt Batch Progress",
            )

        self._maybe_auto_queue(total_batches, batch_index, enable_auto_queue)

        print(f"Output path: {output_path}")
        print(f"File prefix: {file_prefix}")
        print(f"Groups: {total_items} | Batch size: {batch_size}")
        print(f"Batch index: {batch_index} / {max(0, total_batches - 1)}")
        print(f"Is final batch: {is_final_batch}")

        return (
            prompts,
            int(batch_index),
            int(total_batches),
            bool(is_final_batch),
            output_path,
            file_prefix,
        )


class VRGDG_PythonCodeRunner:
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("result_text", "result_json", "has_error")
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"
    MAX_CODE_LENGTH = 8000
    MAX_AST_NODES = 1200
    MAX_EXEC_STEPS = 20000
    MAX_EXEC_SECONDS = 1.5
    ALLOWED_IMPORTS = {"re", "json", "math"}
    BLOCKED_NAMES = {
        "__import__",
        "eval",
        "exec",
        "open",
        "input",
        "compile",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "help",
        "breakpoint",
        "os",
        "sys",
        "subprocess",
        "pathlib",
        "shutil",
        "ctypes",
        "socket",
        "requests",
        "urllib",
        "importlib",
        "builtins",
    }
    SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
    }
    BLOCKED_NODE_TYPES = (
        ast.Global,
        ast.Nonlocal,
        ast.With,
        ast.AsyncWith,
        ast.AsyncFor,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
        ast.While,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
    )

    @classmethod
    def _safe_import(cls, name, globals=None, locals=None, fromlist=(), level=0):
        if level and int(level) > 0:
            raise ImportError("Relative imports are not allowed.")
        root = str(name or "").split(".")[0]
        if root not in cls.ALLOWED_IMPORTS:
            raise ImportError(f"Import blocked: {name}")
        return __import__(name, globals, locals, fromlist, level)

    @classmethod
    def _validate_code(cls, python_code):
        code = str(python_code or "")
        if len(code) > cls.MAX_CODE_LENGTH:
            raise ValueError(
                f"Code is too long ({len(code)} chars). Max allowed is {cls.MAX_CODE_LENGTH}."
            )

        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            raise ValueError(f"Syntax error: {e}") from e

        node_count = 0
        for node in ast.walk(tree):
            node_count += 1
            if node_count > cls.MAX_AST_NODES:
                raise ValueError(
                    f"Code is too complex ({node_count} AST nodes). Max allowed is {cls.MAX_AST_NODES}."
                )

            if isinstance(node, cls.BLOCKED_NODE_TYPES):
                raise ValueError(f"Disallowed syntax: {type(node).__name__}.")

            if isinstance(node, ast.FunctionDef):
                if node.decorator_list:
                    raise ValueError("Function decorators are not allowed.")

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in cls.ALLOWED_IMPORTS:
                        raise ValueError(f"Disallowed import: {alias.name}.")
                    if alias.asname:
                        raise ValueError("Import aliases are not allowed.")

            if isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    raise ValueError("Relative imports are not allowed.")
                mod = node.module or ""
                if mod not in cls.ALLOWED_IMPORTS:
                    raise ValueError(f"Disallowed import: {mod}.")
                for alias in node.names:
                    if alias.asname:
                        raise ValueError("Import aliases are not allowed.")

            if isinstance(node, ast.Name):
                if node.id in cls.BLOCKED_NAMES:
                    raise ValueError(f"Disallowed name: {node.id}.")
                if node.id.startswith("__"):
                    raise ValueError("Dunder names are not allowed.")

            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__"):
                    raise ValueError("Dunder attribute access is not allowed.")

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in cls.BLOCKED_NAMES:
                    raise ValueError(f"Disallowed function call: {node.func.id}.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "python_code": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "# Available vars: input_text, input_json, json, math, re\n"
                            "# Safety mode: imports, filesystem/process/network APIs are blocked.\n"
                            "# Set `result` to any value.\n"
                            "data = json.loads(input_json) if input_json.strip() else {}\n"
                            "result = json.dumps(data, indent=2)"
                        ),
                    },
                ),
            },
            "optional": {
                "input_text": ("STRING", {"multiline": True, "forceInput": True}),
                "input_json": ("STRING", {"multiline": True, "forceInput": True}),
            },
        }

    def run(self, python_code, input_text="", input_json=""):
        self._validate_code(python_code)

        shared_values = {
            "input_text": input_text or "",
            "input_json": input_json or "",
            "json": json,
            "math": math,
            "re": re,
        }
        local_scope = {
            **shared_values,
            "result": "",
        }
        safe_builtins = dict(self.SAFE_BUILTINS)
        safe_builtins["__import__"] = self._safe_import
        global_scope = {"__builtins__": safe_builtins, **shared_values}

        try:
            steps = 0
            start = time.monotonic()
            previous_trace = sys.gettrace()

            def _trace(frame, event, arg):
                nonlocal steps
                if event == "line":
                    steps += 1
                    if steps > self.MAX_EXEC_STEPS:
                        raise TimeoutError(
                            f"Execution step limit exceeded ({self.MAX_EXEC_STEPS})."
                        )
                    if (time.monotonic() - start) > self.MAX_EXEC_SECONDS:
                        raise TimeoutError(
                            f"Execution time limit exceeded ({self.MAX_EXEC_SECONDS}s)."
                        )
                return _trace

            sys.settrace(_trace)
            try:
                exec(python_code, global_scope, local_scope)
            finally:
                sys.settrace(previous_trace)
            result_value = local_scope.get("result", "")

            if isinstance(result_value, str):
                result_text = result_value
            else:
                try:
                    result_text = json.dumps(result_value, ensure_ascii=False, indent=2)
                except Exception:
                    result_text = str(result_value)

            result_json = ""
            if isinstance(result_value, (dict, list)):
                result_json = json.dumps(result_value, ensure_ascii=False, indent=2)
            elif isinstance(result_text, str) and result_text.strip():
                try:
                    parsed = json.loads(result_text)
                    result_json = json.dumps(parsed, ensure_ascii=False, indent=2)
                except Exception:
                    result_json = ""

            return (result_text, result_json, False)
        except Exception as e:
            return (f"{type(e).__name__}: {e}", "", True)

class VRGDG_LoadLatestCombinedJsonText:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        _ensure_combined_files_route_registered()

        all_files = set()
        for batch_type in (BATCH_TYPE_TEXT2IMAGE, BATCH_TYPE_IMAGE2VIDEO):
            files, _ = _list_latest_combined_json_files(batch_type)
            all_files.update(files)

        choices = sorted(all_files, key=str.lower) if all_files else [EMPTY_COMBINED_JSON_OPTION]
        return {
            "required": {
                "batch_type": ([BATCH_TYPE_TEXT2IMAGE, BATCH_TYPE_IMAGE2VIDEO],),
                "combined_json_file": (choices,),
            }
        }

    def run(self, batch_type, combined_json_file):
        selected = str(combined_json_file or "").strip()
        if not selected or selected == EMPTY_COMBINED_JSON_OPTION:
            return ("",)

        batch_type = _normalize_batch_type(batch_type)
        files, latest_folder = _list_latest_combined_json_files(batch_type)
        if not latest_folder or selected not in files:
            return ("",)

        file_path = os.path.join(latest_folder, selected)
        if not os.path.isfile(file_path):
            return ("",)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                raw = f.read()

        raw = raw or ""

        try:
            parsed = json.loads(raw)
            text_out = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            text_out = raw

        return (text_out,)


class VRGDG_UpdateLatestCombinedJsonPrompts:
    RETURN_TYPES = ("STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("updated_json_text", "file_path", "updated_count", "ignored")
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        _ensure_combined_files_route_registered()

        all_files = set()
        for batch_type in (BATCH_TYPE_TEXT2IMAGE, BATCH_TYPE_IMAGE2VIDEO):
            files, _ = _list_latest_combined_json_files(batch_type)
            all_files.update(files)

        choices = sorted(all_files, key=str.lower) if all_files else [EMPTY_COMBINED_JSON_OPTION]
        optional = {}
        for i in range(1, MAX_COMBINED_JSON_PROMPT_EDIT_SLOTS + 1):
            optional[f"prompt_number_{i}"] = (
                "INT",
                {"default": i, "min": 1, "max": 999999},
            )
            optional[f"prompt_text_{i}"] = (
                "STRING",
                {"default": "", "multiline": True},
            )
            optional[f"prompt_image_index_{i}"] = (
                "STRING",
                {"default": "", "multiline": False},
            )

        return {
            "required": {
                "remake_mode": ("BOOLEAN", {"default": False}),
                "batch_type": ([BATCH_TYPE_TEXT2IMAGE, BATCH_TYPE_IMAGE2VIDEO],),
                "combined_json_file": (choices,),
                "prompt_count": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_COMBINED_JSON_PROMPT_EDIT_SLOTS},
                ),
            },
            "optional": optional,
        }

    @classmethod
    def IS_CHANGED(cls, remake_mode, batch_type, combined_json_file, prompt_count, **kwargs):
        file_path, _ = _resolve_latest_combined_json_file_path(batch_type, combined_json_file)
        if not file_path:
            return f"{batch_type}|{combined_json_file}|missing"
        try:
            mtime = os.path.getmtime(file_path)
        except Exception:
            mtime = -1.0
        return f"{batch_type}|{os.path.basename(file_path)}|{mtime}"

    def _collect_updates(self, prompt_count, kwargs):
        try:
            count = int(prompt_count)
        except Exception:
            count = 0
        count = max(0, min(MAX_COMBINED_JSON_PROMPT_EDIT_SLOTS, count))

        updates = []
        for i in range(1, count + 1):
            number_key = f"prompt_number_{i}"
            text_key = f"prompt_text_{i}"

            try:
                prompt_number = int(kwargs.get(number_key, i))
            except Exception:
                continue
            if prompt_number <= 0:
                continue

            prompt_text = kwargs.get(text_key, "")
            if prompt_text is None:
                prompt_text = ""
            elif not isinstance(prompt_text, str):
                prompt_text = str(prompt_text)

            if not prompt_text.strip():
                continue

            updates.append((prompt_number, prompt_text))

        return updates

    def run(self, remake_mode, batch_type, combined_json_file, prompt_count, **kwargs):
        file_path, resolve_error = _resolve_latest_combined_json_file_path(
            batch_type, combined_json_file
        )
        if not file_path:
            return (resolve_error or "Unable to resolve target file.", "", 0, False)

        try:
            data = _load_combined_json_object(file_path)
        except Exception as e:
            return (f"Failed to parse combined JSON: {type(e).__name__}: {e}", file_path, 0, False)
        return (json.dumps(data, ensure_ascii=False, indent=2), file_path, 0, not _normalize_bool(remake_mode))


class VRGDG_UpdateLatestCombinedJsonPrompts_zimage:
    RETURN_TYPES = ("STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("updated_json_text", "file_path", "updated_count", "ignored")
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        _ensure_combined_files_route_registered()

        files, _ = _list_latest_combined_json_files(BATCH_TYPE_TEXT2IMAGE)
        choices = sorted(files, key=str.lower) if files else [EMPTY_COMBINED_JSON_OPTION]
        optional = {}
        for i in range(1, MAX_COMBINED_JSON_PROMPT_EDIT_SLOTS + 1):
            optional[f"prompt_number_{i}"] = (
                "INT",
                {"default": i, "min": 1, "max": 999999},
            )
            optional[f"prompt_text_{i}"] = (
                "STRING",
                {"default": "", "multiline": True},
            )

        return {
            "required": {
                "remake_mode": ("BOOLEAN", {"default": False}),
                "combined_json_file": (choices,),
                "prompt_count": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_COMBINED_JSON_PROMPT_EDIT_SLOTS},
                ),
            },
            "optional": optional,
        }

    @classmethod
    def IS_CHANGED(cls, remake_mode, combined_json_file, prompt_count, **kwargs):
        file_path, _ = _resolve_latest_combined_json_file_path(
            BATCH_TYPE_TEXT2IMAGE, combined_json_file
        )
        if not file_path:
            return f"{BATCH_TYPE_TEXT2IMAGE}|{combined_json_file}|missing"
        try:
            mtime = os.path.getmtime(file_path)
        except Exception:
            mtime = -1.0
        return f"{BATCH_TYPE_TEXT2IMAGE}|{os.path.basename(file_path)}|{mtime}"

    def run(self, remake_mode, combined_json_file, prompt_count, **kwargs):
        file_path, resolve_error = _resolve_latest_combined_json_file_path(
            BATCH_TYPE_TEXT2IMAGE, combined_json_file
        )
        if not file_path:
            return (resolve_error or "Unable to resolve target file.", "", 0, False)

        try:
            data = _load_combined_json_object(file_path)
        except Exception as e:
            return (f"Failed to parse combined JSON: {type(e).__name__}: {e}", file_path, 0, False)
        return (json.dumps(data, ensure_ascii=False, indent=2), file_path, 0, not _normalize_bool(remake_mode))


TEXT_FILE_CATEGORY_OPTIONS = [
    "subject1",
    "subject2",
    "scene1",
    "scene2",
    "other1",
    "other2",
]
TEXT_FILES_ROOT_FOLDER = "VRGDG_TEMP"
TEXT_FILES_SUBFOLDER = "TextFiles"
EMPTY_TEXT_FILE_OPTION = "<no files found>"
EMPTY_TEXT_FOLDER_OPTION = "<no folders found>"
_VRGDG_TEXT_FILES_ROUTE_REGISTERED = False


def _normalize_text_file_category(category):
    c = str(category or "").strip().lower()
    if c in TEXT_FILE_CATEGORY_OPTIONS:
        return c
    return TEXT_FILE_CATEGORY_OPTIONS[0]


def _normalize_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _sanitize_text_segment(value, fallback):
    s = str(value or "").strip()
    s = re.sub(r"[^A-Za-z0-9_\- ]+", "_", s)
    s = s.strip(" .")
    return s or fallback


def _resolve_comfy_output_directory():
    custom_nodes_dir = os.path.dirname(os.path.abspath(__file__))
    comfy_root_dir = os.path.normpath(os.path.join(custom_nodes_dir, "..", ".."))
    comfy_output_dir = os.path.normpath(os.path.join(comfy_root_dir, "output"))
    if os.path.isdir(comfy_output_dir):
        return comfy_output_dir

    try:
        output_dir = folder_paths.get_output_directory()
        if output_dir:
            return os.path.normpath(output_dir)
    except Exception:
        pass

    return comfy_output_dir


def _get_text_files_root():
    return os.path.normpath(
        os.path.join(
            _resolve_comfy_output_directory(),
            TEXT_FILES_ROOT_FOLDER,
            TEXT_FILES_SUBFOLDER,
        )
    )


def _get_text_files_category_folder(category):
    category = _normalize_text_file_category(category)
    return os.path.normpath(os.path.join(_get_text_files_root(), category))


def _get_text_files_manual_folder(folder_name):
    safe_folder = _sanitize_text_segment(folder_name, "default")
    return os.path.normpath(os.path.join(_get_text_files_root(), safe_folder)), safe_folder


def _list_text_files_for_category(category):
    folder = _get_text_files_category_folder(category)
    if not os.path.isdir(folder):
        return [], folder

    files = []
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            continue
        if name.lower().endswith(".txt"):
            files.append(name)

    files.sort(key=str.lower)
    return files, folder


def _list_text_folder_names():
    root = _get_text_files_root()
    if not os.path.isdir(root):
        return []

    folders = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            folders.append(name)

    folders.sort(key=str.lower)
    return folders


def _list_text_files_for_folder(folder_name, use_most_recent=False):
    folder_path, safe_folder = _get_text_files_manual_folder(folder_name)
    if not os.path.isdir(folder_path):
        return [], folder_path, safe_folder

    rows = []
    for name in os.listdir(folder_path):
        full = os.path.join(folder_path, name)
        if not os.path.isfile(full):
            continue
        if not name.lower().endswith(".txt"):
            continue
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            mtime = 0.0
        rows.append((name, mtime))

    rows.sort(key=lambda x: (-x[1], x[0].lower()))
    files = [name for name, _ in rows]
    if use_most_recent and files:
        files = [files[0]]

    return files, folder_path, safe_folder


def _list_all_text_file_names():
    root = _get_text_files_root()
    if not os.path.isdir(root):
        return []

    names = set()
    for folder_name in os.listdir(root):
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for file_name in os.listdir(folder_path):
            full = os.path.join(folder_path, file_name)
            if os.path.isfile(full) and file_name.lower().endswith(".txt"):
                names.add(file_name)

    return sorted(names, key=str.lower)



def _coerce_text_payload(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    if isinstance(text, (dict, list)):
        return json.dumps(text, ensure_ascii=False, indent=2)
    return str(text)


def _ensure_text_files_route_registered():
    global _VRGDG_TEXT_FILES_ROUTE_REGISTERED
    if _VRGDG_TEXT_FILES_ROUTE_REGISTERED:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.get("/vrgdg/text_files/list")
    async def vrgdg_list_text_files(request):
        category = _normalize_text_file_category(request.query.get("category"))
        files, folder = _list_text_files_for_category(category)
        return web.json_response(
            {
                "category": category,
                "files": files,
                "folder": folder,
            }
        )

    @server_instance.routes.get("/vrgdg/text_files/folders")
    async def vrgdg_list_text_folders(request):
        folders = _list_text_folder_names()
        return web.json_response(
            {
                "folders": folders,
                "root": _get_text_files_root(),
            }
        )

    @server_instance.routes.get("/vrgdg/text_files/files")
    async def vrgdg_list_text_files_for_folder(request):
        folder_name = request.query.get("folder", "")
        use_most_recent = _normalize_bool(request.query.get("use_most_recent", "false"))
        files, folder_path, safe_folder = _list_text_files_for_folder(
            folder_name,
            use_most_recent,
        )
        return web.json_response(
            {
                "folder": safe_folder,
                "folder_path": folder_path,
                "use_most_recent": bool(use_most_recent),
                "files": files,
            }
        )

    _VRGDG_TEXT_FILES_ROUTE_REGISTERED = True


_ensure_text_files_route_registered()


class VRGDG_SaveTextAdvanced:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "file_path")
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_name": ("STRING", {"default": "story"}),
                "file_name": ("STRING", {"default": "story"}),
                "overwrite": ("BOOLEAN", {"default": True}),
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "trigger": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    def run(self, folder_name, file_name, overwrite, text, trigger):
        folder_path, _ = _get_text_files_manual_folder(folder_name)
        os.makedirs(folder_path, exist_ok=True)

        safe_base_name = _sanitize_text_segment(file_name, "text")
        if overwrite:
            final_name = f"{safe_base_name}.txt"
        else:
            final_name = _next_incremental_prefixed_file_name(folder_path, safe_base_name)

        file_path = os.path.normpath(os.path.join(folder_path, final_name))
        text_to_write = _coerce_text_payload(text)
        print(
            f"[VRGDG_SaveTextAdvanced] len={len(text_to_write)} "
            f"start={repr(text_to_write[:1])} end={repr(text_to_write[-1:])}"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_to_write)

        return (text_to_write, file_path)


class VRGDG_LoadTextAdvanced:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "file_path")
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        _ensure_text_files_route_registered()
        folders = _list_text_folder_names()
        folder_choices = folders if folders else [EMPTY_TEXT_FOLDER_OPTION]
        file_choices = _list_all_text_file_names() or [EMPTY_TEXT_FILE_OPTION]

        return {
            "required": {
                "folder_name": (folder_choices,),
                "use_most_recent": ("BOOLEAN", {"default": True}),
                "text_file": (file_choices,),
            }
        }

    @classmethod
    def IS_CHANGED(cls, folder_name, use_most_recent, text_file):
        selected_folder = str(folder_name or "").strip()
        if not selected_folder or selected_folder == EMPTY_TEXT_FOLDER_OPTION:
            return "empty-folder"

        files, folder_path, _ = _list_text_files_for_folder(
            selected_folder,
            bool(use_most_recent),
        )
        if not files:
            return f"{selected_folder}|no-files"

        if bool(use_most_recent):
            chosen_name = files[0]
        else:
            selected_name = os.path.basename(str(text_file or "").strip())
            if selected_name in files:
                chosen_name = selected_name
            else:
                return f"{selected_folder}|missing-selection|{selected_name}"

        file_path = os.path.normpath(os.path.join(folder_path, chosen_name))
        try:
            stats = os.stat(file_path)
            return (
                f"{file_path}|{int(bool(use_most_recent))}|"
                f"{stats.st_mtime_ns}|{stats.st_size}"
            )
        except OSError:
            return f"{file_path}|missing"

    def run(self, folder_name, use_most_recent, text_file):
        selected_folder = str(folder_name or "").strip()
        if not selected_folder or selected_folder == EMPTY_TEXT_FOLDER_OPTION:
            return ("", "")

        files, folder_path, _ = _list_text_files_for_folder(
            selected_folder,
            bool(use_most_recent),
        )
        if not files:
            return ("", "")

        if bool(use_most_recent):
            chosen_name = files[0]
        else:
            selected_name = os.path.basename(str(text_file or "").strip())
            if selected_name in files:
                chosen_name = selected_name
            else:
                return ("", "")

        file_path = os.path.normpath(os.path.join(folder_path, chosen_name))
        if not os.path.isfile(file_path):
            return ("", "")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                text = f.read()

        return (text or "", file_path)


class VRGDG_SaveText:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "file_path")
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_type": (TEXT_FILE_CATEGORY_OPTIONS,),
                "text": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    def run(self, text_type, text):
        category = _normalize_text_file_category(text_type)
        folder = _get_text_files_category_folder(category)
        os.makedirs(folder, exist_ok=True)

        file_name = f"{category}.txt"
        file_path = os.path.join(folder, file_name)

        text_to_write = _coerce_text_payload(text)
        print(
            f"[VRGDG_SaveText] len={len(text_to_write)} "
            f"start={repr(text_to_write[:1])} end={repr(text_to_write[-1:])}"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_to_write)

        return (text_to_write, file_path)


class VRGDG_LoadText:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "file_path")
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        _ensure_text_files_route_registered()
        files, _ = _list_text_files_for_category(TEXT_FILE_CATEGORY_OPTIONS[0])
        choices = files if files else [EMPTY_TEXT_FILE_OPTION]
        return {
            "required": {
                "text_type": (TEXT_FILE_CATEGORY_OPTIONS,),
                "text_file": (choices,),
            }
        }

    def run(self, text_type, text_file):
        category = _normalize_text_file_category(text_type)
        selected = str(text_file or "").strip()
        if not selected or selected == EMPTY_TEXT_FILE_OPTION:
            return ("", "")

        selected_name = os.path.basename(selected)
        if selected_name != selected or not selected_name.lower().endswith(".txt"):
            return ("", "")

        folder = _get_text_files_category_folder(category)
        file_path = os.path.normpath(os.path.join(folder, selected_name))
        if not os.path.isfile(file_path):
            return ("", "")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                text = f.read()

        return (text or "", file_path)


class VRGDG_SaveAudioFilePath:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("audio_file_path", "saved_txt_path")
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "txt_name": ("STRING", {"default": "audio_file_path"}),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    def _extract_audio_path(self, audio):
        if not isinstance(audio, dict):
            return ""

        for key in (
            "path",
            "file_path",
            "filepath",
            "filename",
            "audio_path",
            "source_path",
            "source",
            "url",
        ):
            value = audio.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        metadata = audio.get("metadata")
        if isinstance(metadata, dict):
            for key in (
                "path",
                "file_path",
                "filepath",
                "filename",
                "audio_path",
                "source_path",
                "source",
                "url",
            ):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return ""

    def _coerce_candidate_path(self, value):
        if not isinstance(value, str):
            return ""
        candidate = value.strip()
        if not candidate:
            return ""
        if os.path.isabs(candidate):
            return os.path.normpath(candidate)

        bases = [
            folder_paths.get_input_directory(),
            folder_paths.get_output_directory(),
        ]
        get_temp = getattr(folder_paths, "get_temp_directory", None)
        if callable(get_temp):
            bases.append(get_temp())

        for base in bases:
            full = os.path.normpath(os.path.join(base, candidate))
            if os.path.exists(full):
                return full

        return candidate

    def _extract_audio_path_from_prompt(self, prompt, unique_id):
        if not isinstance(prompt, dict):
            return ""

        current_id = str(unique_id or "").strip()
        if not current_id:
            return ""

        current_node = prompt.get(current_id)
        if not isinstance(current_node, dict):
            return ""

        current_inputs = current_node.get("inputs", {})
        audio_link = current_inputs.get("audio")
        if not (
            isinstance(audio_link, (list, tuple))
            and len(audio_link) >= 1
            and str(audio_link[0]) in prompt
        ):
            return ""

        visited = set()
        source_id = str(audio_link[0])

        for _ in range(24):
            if source_id in visited:
                break
            visited.add(source_id)

            source_node = prompt.get(source_id, {})
            if not isinstance(source_node, dict):
                break

            source_inputs = source_node.get("inputs", {})
            for key in (
                "audio",
                "path",
                "file_path",
                "filepath",
                "filename",
                "audio_path",
                "source_path",
                "source",
                "url",
            ):
                found = self._coerce_candidate_path(source_inputs.get(key))
                if found:
                    return found

            next_id = None
            for value in source_inputs.values():
                if (
                    isinstance(value, (list, tuple))
                    and len(value) >= 1
                    and str(value[0]) in prompt
                ):
                    next_id = str(value[0])
                    break
            if not next_id:
                break
            source_id = next_id

        return ""

    def run(self, audio, txt_name, overwrite, prompt=None, unique_id=None):
        audio_file_path = self._extract_audio_path(audio)
        if not audio_file_path:
            audio_file_path = self._extract_audio_path_from_prompt(prompt, unique_id)

        target_folder = os.path.normpath(
            os.path.join(
                folder_paths.get_output_directory(),
                "VRGDG_TEMP",
                "FilePaths",
            )
        )
        os.makedirs(target_folder, exist_ok=True)

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(txt_name or "").strip())
        safe_name = safe_name.strip("._")
        if not safe_name:
            safe_name = "audio_file_path"

        if overwrite:
            final_name = f"{safe_name}.txt"
        else:
            i = 1
            final_name = f"{safe_name}_{i:03d}.txt"
            while os.path.exists(os.path.join(target_folder, final_name)):
                i += 1
                final_name = f"{safe_name}_{i:03d}.txt"

        saved_txt_path = os.path.normpath(os.path.join(target_folder, final_name))
        with open(saved_txt_path, "w", encoding="utf-8") as f:
            f.write(audio_file_path)

        return (audio_file_path, saved_txt_path)


class VRGDG_LoadAudioFilePath:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("audio_file_path", "audio_file_name")
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refresh": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    @staticmethod
    def _paths_folder():
        return os.path.normpath(
            os.path.join(
                folder_paths.get_output_directory(),
                "VRGDG_TEMP",
                "FilePaths",
            )
        )

    @classmethod
    def _latest_txt(cls):
        folder = cls._paths_folder()
        if not os.path.isdir(folder):
            return ("", 0.0)

        latest_path = ""
        latest_mtime = 0.0
        for name in os.listdir(folder):
            if not name.lower().endswith(".txt"):
                continue
            full = os.path.join(folder, name)
            if not os.path.isfile(full):
                continue
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = full

        return (latest_path, latest_mtime)

    @classmethod
    def IS_CHANGED(cls, refresh):
        latest_path, latest_mtime = cls._latest_txt()
        return f"{refresh}|{latest_path}|{latest_mtime}"

    def run(self, refresh):
        latest_txt, _ = self._latest_txt()
        if not latest_txt:
            return ("", "")

        try:
            with open(latest_txt, "r", encoding="utf-8") as f:
                audio_file_path = str(f.read() or "").strip()
        except UnicodeDecodeError:
            with open(latest_txt, "r", encoding="utf-8-sig") as f:
                audio_file_path = str(f.read() or "").strip()

        audio_file_name = os.path.basename(audio_file_path) if audio_file_path else ""
        return (audio_file_path, audio_file_name)


class VRGDG_IntToString:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0}),
            }
        }

    def run(self, value):
        return (str(value),)


class VRGDG_ArchiveLlmBatchFolders:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("trigger", "details")
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("STRING", {"default": ""}),
            }
        }

    @staticmethod
    def _unique_destination(base_path):
        if not os.path.exists(base_path):
            return base_path

        i = 1
        while True:
            candidate = f"{base_path}_{i:03d}"
            if not os.path.exists(candidate):
                return candidate
            i += 1

    def run(self, trigger):
        llm_batches_root = _get_llm_batches_root()
        os.makedirs(llm_batches_root, exist_ok=True)

        old_folder = os.path.join(llm_batches_root, "old")
        os.makedirs(old_folder, exist_ok=True)

        moved = []
        skipped = []

        for name in os.listdir(llm_batches_root):
            source = os.path.join(llm_batches_root, name)
            if not os.path.isdir(source):
                continue
            if os.path.normcase(name) == os.path.normcase("old"):
                continue

            destination = self._unique_destination(os.path.join(old_folder, name))
            try:
                shutil.move(source, destination)
                moved.append(f"{name} -> {os.path.basename(destination)}")
            except Exception as e:
                skipped.append(f"{name}: {e}")

        details = f"Archived {len(moved)} folder(s) from llm_batches to old."
        if moved:
            details += " Moved: " + ", ".join(moved)
        if skipped:
            details += " Skipped: " + "; ".join(skipped)

        return (str(trigger), details)


NODE_CLASS_MAPPINGS = {
    "VRGDG_GeneralPromptBatcher": VRGDG_GeneralPromptBatcher,
    "VRGDG_PythonCodeRunner": VRGDG_PythonCodeRunner,
    "VRGDG_LoadLatestCombinedJsonText": VRGDG_LoadLatestCombinedJsonText,
    "VRGDG_UpdateLatestCombinedJsonPrompts": VRGDG_UpdateLatestCombinedJsonPrompts,
    "VRGDG_UpdateLatestCombinedJsonPrompts_zimage": VRGDG_UpdateLatestCombinedJsonPrompts_zimage,
    "VRGDG_SaveText": VRGDG_SaveText,
    "VRGDG_LoadText": VRGDG_LoadText,
    "VRGDG_SaveTextAdvanced": VRGDG_SaveTextAdvanced,
    "VRGDG_LoadTextAdvanced": VRGDG_LoadTextAdvanced,
    "VRGDG_SaveAudioFilePath": VRGDG_SaveAudioFilePath,
    "VRGDG_LoadAudioFilePath": VRGDG_LoadAudioFilePath,
    "VRGDG_IntToString": VRGDG_IntToString,
    "VRGDG_ArchiveLlmBatchFolders": VRGDG_ArchiveLlmBatchFolders,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_GeneralPromptBatcher": "VRGDG_GeneralPromptBatcher",
    "VRGDG_PythonCodeRunner": "VRGDG_PythonCodeRunner",
    "VRGDG_LoadLatestCombinedJsonText": "VRGDG_LoadLatestCombinedJsonText",
    "VRGDG_UpdateLatestCombinedJsonPrompts": "VRGDG_UpdateLatestCombinedJsonPrompts",
    "VRGDG_UpdateLatestCombinedJsonPrompts_zimage": "VRGDG_UpdateLatestCombinedJsonPrompts_zimage",
    "VRGDG_SaveText": "VRGDG_SaveText",
    "VRGDG_LoadText": "VRGDG_LoadText",
    "VRGDG_SaveTextAdvanced": "VRGDG_SaveTextAdvanced",
    "VRGDG_LoadTextAdvanced": "VRGDG_LoadTextAdvanced",
    "VRGDG_SaveAudioFilePath": "VRGDG_SaveAudioFilePath",
    "VRGDG_LoadAudioFilePath": "VRGDG_LoadAudioFilePath",
    "VRGDG_IntToString": "VRGDG_IntToString",
    "VRGDG_ArchiveLlmBatchFolders": "VRGDG_ArchiveLlmBatchFolders",
}





