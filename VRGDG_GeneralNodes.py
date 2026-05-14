import ast
import json
import math
import os
import re
import shutil
import sys
import time
import random


import folder_paths
from aiohttp import web
from server import PromptServer


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


IMAGE2VIDEO_BATCH_FOLDER_PREFIX = "Image2Video_Batch_"
TEXT2IMAGE_BATCH_FOLDER_PREFIX = "Text2Image_Batch_"
LLM_BATCHES_FOLDER_NAME = "llm_batches"
COMBINED_JSON_SUFFIX = "_COMBINED.json"
EMPTY_COMBINED_JSON_OPTION = "<no files found>"
BATCH_TYPE_TEXT2IMAGE = "Text2Image"
BATCH_TYPE_IMAGE2VIDEO = "Image2Video"
MAX_COMBINED_JSON_PROMPT_EDIT_SLOTS = 120


def _get_llm_batches_root():
    return os.path.normpath(
        os.path.join(folder_paths.get_output_directory(), LLM_BATCHES_FOLDER_NAME)
    )


def _find_latest_batch_folder(prefix=None):
    root = _get_llm_batches_root()
    if not os.path.isdir(root):
        return None

    highest_num = -1
    highest_path = None
    latest_path = None
    latest_mtime = -1.0

    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if prefix and not name.startswith(prefix):
            continue

        if prefix:
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                batch_num = int(suffix)
                if batch_num > highest_num:
                    highest_num = batch_num
                    highest_path = full

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


def _combined_json_choices(batch_types=None):
    batch_types = batch_types or (BATCH_TYPE_TEXT2IMAGE, BATCH_TYPE_IMAGE2VIDEO)
    all_files = []
    seen = set()

    for batch_type in batch_types:
        files, _ = _list_latest_combined_json_files(batch_type)
        for name in files:
            if name in seen:
                continue
            seen.add(name)
            all_files.append(name)

    all_files.sort(key=str.lower)
    return [EMPTY_COMBINED_JSON_OPTION, *all_files]


def _find_latest_combined_json_file_path(batch_type):
    batch_type = _normalize_batch_type(batch_type)
    files, latest_folder = _list_latest_combined_json_files(batch_type)
    if not latest_folder or not files:
        return None

    latest_path = None
    latest_mtime = -1.0
    for name in files:
        full = os.path.normpath(os.path.join(latest_folder, name))
        if not os.path.isfile(full):
            continue
        try:
            mtime = max(os.path.getctime(full), os.path.getmtime(full))
        except OSError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = full

    return latest_path


def _resolve_latest_combined_json_file_path(batch_type, combined_json_file, allow_auto_latest=False):
    selected = os.path.basename(str(combined_json_file or "").strip())
    if not selected or selected == EMPTY_COMBINED_JSON_OPTION:
        if allow_auto_latest:
            latest_path = _find_latest_combined_json_file_path(batch_type)
            if latest_path:
                return latest_path, ""
        return None, "No combined JSON file selected."

    batch_type = _normalize_batch_type(batch_type)
    files, latest_folder = _list_latest_combined_json_files(batch_type)
    if not latest_folder:
        return None, f"No latest {batch_type} batch folder found."
    if selected not in files:
        if allow_auto_latest:
            latest_path = _find_latest_combined_json_file_path(batch_type)
            if latest_path:
                return latest_path, ""
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


def _clean_user_folder_path(folder_path):
    raw = str(folder_path or "").strip()
    if not raw:
        return ""

    # Accept common pasted path formats like quoted strings or file URLs.
    if raw.startswith("file:///"):
        raw = raw[8:]
    raw = raw.strip().strip("\"'`")
    return raw.strip()


def _resolve_remake_folder_path(folder_path):
    raw = _clean_user_folder_path(folder_path)
    if not raw:
        return None, "Folder path is empty."

    normalized = os.path.normpath(raw)
    if os.path.basename(normalized).lower() == "remake":
        remake_folder = normalized
    else:
        remake_folder = os.path.normpath(os.path.join(normalized, "remake"))

    if not os.path.isdir(remake_folder):
        return None, f"Remake folder not found: {remake_folder}"
    return remake_folder, ""


def _extract_prompt_indexes_from_remake_folder(folder_path, max_items=MAX_COMBINED_JSON_PROMPT_EDIT_SLOTS):
    remake_folder, error = _resolve_remake_folder_path(folder_path)
    if not remake_folder:
        return None, error

    indexes = set()
    for name in os.listdir(remake_folder):
        full_path = os.path.join(remake_folder, name)
        if not os.path.isfile(full_path):
            continue

        match = re.match(r"^video_(\d+)_", name, flags=re.IGNORECASE)
        if not match:
            continue

        try:
            prompt_number = int(match.group(1))
        except Exception:
            continue

        if prompt_number > 0:
            indexes.add(prompt_number)

    sorted_indexes = sorted(indexes)
    if not sorted_indexes:
        return [], ""
    return sorted_indexes[:max_items], ""


def _extract_prompt_rows_for_ui(data, max_items=None):
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
    if isinstance(max_items, int) and max_items > 0:
        return rows[:max_items]
    return rows


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
        combined_json_file = request.query.get("combined_json_file", "")
        files, latest_folder = _list_latest_combined_json_files(batch_type)
        resolved_file_path, _ = _resolve_latest_combined_json_file_path(
            batch_type, combined_json_file, allow_auto_latest=True
        )
        return web.json_response(
            {
                "batch_type": batch_type,
                "files": files,
                "latest_folder": latest_folder or "",
                "resolved_file": os.path.basename(resolved_file_path) if resolved_file_path else "",
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

    @server_instance.routes.post("/vrgdg/llm_batches/remake_prompt_indexes")
    async def vrgdg_get_remake_prompt_indexes(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "Invalid JSON body."},
                status=400,
            )

        indexes, error = _extract_prompt_indexes_from_remake_folder(payload.get("folder_path", ""))
        if indexes is None:
            return web.json_response(
                {"ok": False, "error": error or "Unable to inspect remake folder."},
                status=400,
            )

        remake_folder, _ = _resolve_remake_folder_path(payload.get("folder_path", ""))
        return web.json_response(
            {
                "ok": True,
                "folder_path": str(payload.get("folder_path", "") or ""),
                "remake_folder": remake_folder or "",
                "prompt_count": len(indexes),
                "prompt_numbers": indexes,
                "empty": len(indexes) == 0,
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
                "trigger": (any_typ, {"forceInput": True}),
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
                if self._has_meaningful_group_value(value):
                    parts.append(f"{input_name}:\n{value}")
            sections.append("\n\n".join(parts))

        return "\n\n".join(sections).strip()

    def _has_meaningful_group_value(self, value):
        if value is None:
            return False

        text = str(value).strip()
        if not text:
            return False
        if text in ("{}", "[]", '""', "null", "None"):
            return False

        return True

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
        safe_builtins = dict(self.SAFE_BUILTINS)
        safe_builtins["__import__"] = self._safe_import
        exec_scope = {
            "__builtins__": safe_builtins,
            **shared_values,
            "result": "",
        }

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
                # Run in a single namespace so user-defined helpers are visible
                # to comprehensions and other nested evaluation contexts.
                exec(python_code, exec_scope, exec_scope)
            finally:
                sys.settrace(previous_trace)
            result_value = exec_scope.get("result", "")

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
        return {
            "required": {
                "batch_type": ([BATCH_TYPE_TEXT2IMAGE, BATCH_TYPE_IMAGE2VIDEO],),
                "combined_json_file": (_combined_json_choices(),),
                "refresh": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, batch_type, combined_json_file, refresh):
        batch_type = _normalize_batch_type(batch_type)
        selected_path, _ = _resolve_latest_combined_json_file_path(
            batch_type, combined_json_file, allow_auto_latest=True
        )
        selected_mtime = 0.0
        if selected_path:
            try:
                selected_mtime = os.path.getmtime(selected_path)
            except OSError:
                selected_mtime = 0.0

        return f"{refresh}|{batch_type}|{selected_path or ''}|{selected_mtime}"

    def run(self, batch_type, combined_json_file, refresh):
        batch_type = _normalize_batch_type(batch_type)
        file_path, _ = _resolve_latest_combined_json_file_path(
            batch_type, combined_json_file, allow_auto_latest=True
        )
        if not file_path:
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
                "combined_json_file": (_combined_json_choices(),),
                "prompt_count": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_COMBINED_JSON_PROMPT_EDIT_SLOTS},
                ),
            },
            "optional": {
                "folder_path": (
                    "STRING",
                    {"default": "", "forceInput": True, "multiline": False},
                ),
                **optional,
            },
        }

    @classmethod
    def IS_CHANGED(cls, remake_mode, batch_type, combined_json_file, prompt_count, folder_path="", **kwargs):
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

    def run(self, remake_mode, batch_type, combined_json_file, prompt_count, folder_path="", **kwargs):
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
                "combined_json_file": (_combined_json_choices((BATCH_TYPE_TEXT2IMAGE,)),),
                "prompt_count": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_COMBINED_JSON_PROMPT_EDIT_SLOTS},
                ),
            },
            "optional": {
                "folder_path": (
                    "STRING",
                    {"default": "", "forceInput": True, "multiline": False},
                ),
                **optional,
            },
        }

    @classmethod
    def IS_CHANGED(cls, remake_mode, combined_json_file, prompt_count, folder_path="", **kwargs):
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

    def run(self, remake_mode, combined_json_file, prompt_count, folder_path="", **kwargs):
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
    try:
        output_dir = folder_paths.get_output_directory()
        if output_dir:
            return os.path.normpath(output_dir)
    except Exception:
        pass

    custom_nodes_dir = os.path.dirname(os.path.abspath(__file__))
    comfy_root_dir = os.path.normpath(os.path.join(custom_nodes_dir, "..", ".."))
    return os.path.normpath(os.path.join(comfy_root_dir, "output"))


def _resolve_default_comfy_output_directory():
    base_path = getattr(folder_paths, "base_path", None)
    if base_path:
        return os.path.normpath(os.path.join(base_path, "output"))

    custom_nodes_dir = os.path.dirname(os.path.abspath(__file__))
    comfy_root_dir = os.path.normpath(os.path.join(custom_nodes_dir, "..", ".."))
    return os.path.normpath(os.path.join(comfy_root_dir, "output"))


def _get_output_directory_candidates():
    candidates = [
        _resolve_comfy_output_directory(),
        _resolve_default_comfy_output_directory(),
    ]
    unique = []
    seen = set()
    for path in candidates:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _get_text_files_root():
    return os.path.normpath(
        os.path.join(
            _resolve_comfy_output_directory(),
            TEXT_FILES_ROOT_FOLDER,
            TEXT_FILES_SUBFOLDER,
        )
    )


def _get_text_files_root_candidates():
    return [
        os.path.normpath(os.path.join(output_dir, TEXT_FILES_ROOT_FOLDER, TEXT_FILES_SUBFOLDER))
        for output_dir in _get_output_directory_candidates()
    ]


def _get_text_files_category_folder(category):
    category = _normalize_text_file_category(category)
    return os.path.normpath(os.path.join(_get_text_files_root(), category))


def _get_text_files_manual_folder(folder_name):
    safe_folder = _sanitize_text_segment(folder_name, "default")
    return os.path.normpath(os.path.join(_get_text_files_root(), safe_folder)), safe_folder


def _normalize_custom_text_files_root(custom_base_path):
    raw_path = str(custom_base_path or "").strip().strip("\"'")
    if not raw_path:
        return ""
    expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(raw_path)))
    path = os.path.normpath(expanded)
    base_name = os.path.basename(path).lower()
    parent_name = os.path.basename(os.path.dirname(path)).lower()
    grandparent_name = os.path.basename(os.path.dirname(os.path.dirname(path))).lower()

    if base_name == TEXT_FILES_SUBFOLDER.lower() and parent_name == TEXT_FILES_ROOT_FOLDER.lower():
        return path
    if base_name == TEXT_FILES_ROOT_FOLDER.lower():
        return os.path.normpath(os.path.join(path, TEXT_FILES_SUBFOLDER))
    if parent_name == TEXT_FILES_SUBFOLDER.lower() and grandparent_name == TEXT_FILES_ROOT_FOLDER.lower():
        return os.path.dirname(path)
    return os.path.normpath(os.path.join(path, TEXT_FILES_ROOT_FOLDER, TEXT_FILES_SUBFOLDER))


def _get_text_files_root_candidates_for_custom(use_custom_base_path=False, custom_base_path=""):
    if use_custom_base_path:
        custom_root = _normalize_custom_text_files_root(custom_base_path)
        return [custom_root] if custom_root else []
    return _get_text_files_root_candidates()


def _list_text_files_for_category(category):
    category = _normalize_text_file_category(category)
    first_folder = None
    for root in _get_text_files_root_candidates():
        folder = os.path.normpath(os.path.join(root, category))
        if first_folder is None:
            first_folder = folder
        if not os.path.isdir(folder):
            continue

        files = []
        for name in os.listdir(folder):
            full = os.path.join(folder, name)
            if not os.path.isfile(full):
                continue
            if name.lower().endswith(".txt"):
                files.append(name)

        files.sort(key=str.lower)
        if files:
            return files, folder

    return [], first_folder or _get_text_files_category_folder(category)


def _list_text_folder_names():
    folders = set()
    for root in _get_text_files_root_candidates():
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if os.path.isdir(full):
                folders.add(name)

    return sorted(folders, key=str.lower)


def _list_text_files_for_folder(folder_name, use_most_recent=False, use_custom_base_path=False, custom_base_path=""):
    safe_folder = _sanitize_text_segment(folder_name, "default")
    first_folder_path = None
    selected_folder_path = None
    rows_by_name = {}
    for root in _get_text_files_root_candidates_for_custom(use_custom_base_path, custom_base_path):
        folder_path = os.path.normpath(os.path.join(root, safe_folder))
        if first_folder_path is None:
            first_folder_path = folder_path
        if not os.path.isdir(folder_path):
            continue

        for name in os.listdir(folder_path):
            full = os.path.join(folder_path, name)
            if not os.path.isfile(full):
                continue
            if not name.lower().endswith(".txt"):
                continue
            try:
                mtime = os.path.getmtime(full)
                size = os.path.getsize(full)
            except OSError:
                mtime = 0.0
                size = 0
            existing = rows_by_name.get(name)
            if existing is None or (size > 0 and existing[2] == 0) or mtime > existing[1]:
                rows_by_name[name] = (name, mtime, size, folder_path)
                selected_folder_path = folder_path

    if not rows_by_name:
        if first_folder_path:
            return [], first_folder_path, safe_folder
        return [], _get_text_files_manual_folder(folder_name)[0], safe_folder

    rows = list(rows_by_name.values())
    rows.sort(key=lambda x: (-x[1], x[0].lower()))
    files = [name for name, _, _, _ in rows]
    if use_most_recent and files:
        files = [files[0]]

    return files, selected_folder_path or first_folder_path, safe_folder


def _list_all_text_file_names():
    names = set()
    for root in _get_text_files_root_candidates():
        if not os.path.isdir(root):
            continue
        for folder_name in os.listdir(root):
            folder_path = os.path.join(root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for file_name in os.listdir(folder_path):
                full = os.path.join(folder_path, file_name)
                if os.path.isfile(full) and file_name.lower().endswith(".txt"):
                    names.add(file_name)

    return sorted(names, key=str.lower)


def _resolve_text_file_path(folder_name, file_name, use_custom_base_path=False, custom_base_path=""):
    safe_folder = _sanitize_text_segment(folder_name, "default")
    selected_name = os.path.basename(str(file_name or "").strip())
    if not selected_name or selected_name != str(file_name or "").strip():
        return ""
    if not selected_name.lower().endswith(".txt"):
        return ""

    fallback = ""
    for root in _get_text_files_root_candidates_for_custom(use_custom_base_path, custom_base_path):
        file_path = os.path.normpath(os.path.join(root, safe_folder, selected_name))
        if not fallback:
            fallback = file_path
        if not os.path.isfile(file_path):
            continue
        try:
            if os.path.getsize(file_path) > 0:
                return file_path
        except OSError:
            return file_path
    return fallback if fallback and os.path.isfile(fallback) else ""



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
        use_custom_base_path = _normalize_bool(request.query.get("use_custom_base_path", "false"))
        custom_base_path = request.query.get("custom_base_path", "")
        files, folder_path, safe_folder = _list_text_files_for_folder(
            folder_name,
            use_most_recent,
            use_custom_base_path,
            custom_base_path,
        )
        return web.json_response(
            {
                "folder": safe_folder,
                "folder_path": folder_path,
                "use_most_recent": bool(use_most_recent),
                "use_custom_base_path": bool(use_custom_base_path),
                "custom_text_files_root": _normalize_custom_text_files_root(custom_base_path) if use_custom_base_path else "",
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
                "folder_name_override": ("STRING", {"default": "", "multiline": False}),
                "use_custom_base_path": ("BOOLEAN", {"default": False}),
                "custom_base_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    @classmethod
    def _resolve_folder_name(cls, folder_name, folder_name_override):
        override_folder = str(folder_name_override or "").strip()
        if override_folder:
            return override_folder
        return str(folder_name or "").strip()

    @classmethod
    def IS_CHANGED(cls, folder_name, use_most_recent, text_file, folder_name_override="", use_custom_base_path=False, custom_base_path=""):
        selected_folder = cls._resolve_folder_name(folder_name, folder_name_override)
        if not selected_folder or selected_folder == EMPTY_TEXT_FOLDER_OPTION:
            return "empty-folder"

        files, folder_path, _ = _list_text_files_for_folder(
            selected_folder,
            bool(use_most_recent),
            bool(use_custom_base_path),
            custom_base_path,
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

        file_path = _resolve_text_file_path(selected_folder, chosen_name, bool(use_custom_base_path), custom_base_path)
        if not file_path:
            file_path = os.path.normpath(os.path.join(folder_path, chosen_name))
        try:
            stats = os.stat(file_path)
            return (
                f"{file_path}|{int(bool(use_most_recent))}|"
                f"{stats.st_mtime_ns}|{stats.st_size}"
            )
        except OSError:
            return f"{file_path}|missing"

    def run(self, folder_name, use_most_recent, text_file, folder_name_override="", use_custom_base_path=False, custom_base_path=""):
        selected_folder = self._resolve_folder_name(folder_name, folder_name_override)
        if not selected_folder or selected_folder == EMPTY_TEXT_FOLDER_OPTION:
            return ("", "")

        files, folder_path, _ = _list_text_files_for_folder(
            selected_folder,
            bool(use_most_recent),
            bool(use_custom_base_path),
            custom_base_path,
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

        file_path = _resolve_text_file_path(selected_folder, chosen_name, bool(use_custom_base_path), custom_base_path)
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

        file_path = _resolve_text_file_path(category, selected_name)
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
    def _audio_folder():
        return os.path.normpath(
            os.path.join(
                folder_paths.get_output_directory(),
                "VRGDG_AudioFiles",
            )
        )

    @classmethod
    def _latest_audio(cls):
        folder = cls._audio_folder()
        if not os.path.isdir(folder):
            return ("", 0.0)

        files = [name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]
        try:
            files = folder_paths.filter_files_content_types(files, ["audio"])
        except Exception:
            # Fallback extensions if content-type filtering is unavailable.
            exts = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif"}
            files = [name for name in files if os.path.splitext(name)[1].lower() in exts]

        latest_path = ""
        latest_mtime = 0.0
        for name in files:
            full = os.path.join(folder, name)
            try:
                mtime = max(os.path.getctime(full), os.path.getmtime(full))
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = full

        return (latest_path, latest_mtime)

    @classmethod
    def IS_CHANGED(cls, refresh):
        latest_path, latest_mtime = cls._latest_audio()
        return f"{refresh}|{latest_path}|{latest_mtime}"

    @staticmethod
    def _clean_audio_name(file_path):
        name = os.path.basename(str(file_path or "")).strip()
        name = os.path.splitext(name)[0]
        name = re.sub(r"\s*\(\d+\)\s*$", "", name)
        name = re.sub(r"[^A-Za-z_]+$", "", name)
        return name

    def run(self, refresh):
        latest_audio, _ = self._latest_audio()
        if not latest_audio:
            return ("", "")

        audio_file_path = os.path.normpath(latest_audio)
        audio_file_name = self._clean_audio_name(audio_file_path)
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

class VRGDG_CyclingTextPicker:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("formatted_text", "selected_item", "selected_items", "wrapped_index", "item_count")
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": -999999,
                        "max": 999999,
                        "tooltip": (
                            "The step number used to choose from the list. In index mode this wraps around automatically: "
                            "with 5 items, index 0 selects item 1, index 4 selects item 5, and index 5 starts over at item 1. "
                            "In random modes, this still acts like the current step so each frame/batch/index can pick a different item."
                        ),
                    },
                ),
                "items": (
                    "STRING",
                    {
                        "default": "slow push in\nwide orbit\nhandheld follow\nlow angle reveal\ncrane up",
                        "multiline": True,
                        "tooltip": (
                            "The list of text choices to pick from. Works with one item per line, blank-line-separated chunks, "
                            "comma-separated text, pipe-separated text, JSON arrays like [\"push in\", \"orbit left\"], "
                            "Python-style lists/sets, or JSON objects with an items/values/motions key. "
                            "Example: slow push in, orbit left, handheld follow."
                        ),
                    },
                ),
                "label": (
                    "STRING",
                    {
                        "default": "Camera Motion",
                        "multiline": False,
                        "tooltip": (
                            "Optional name placed before the selected text in the formatted output. "
                            "Example label Camera Motion outputs: Camera Motion = slow push in. "
                            "Leave blank if you only want the selected text with no prefix."
                        ),
                    },
                ),
                "max_items": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "tooltip": (
                            "Optional limit for how many parsed list items are used. Leave at 0 to use the full list automatically. "
                            "Example: if the list has 20 items and max_items is 5, only the first 5 items are used and the index loops through those 5."
                        ),
                    },
                ),
                "split_mode": (
                    ["auto", "json/python", "line", "blank line", "comma", "pipe"],
                    {
                        "tooltip": (
                            "How to split the items text into a list. Auto tries JSON/Python first, then detects blank lines, commas, pipes, or normal lines. "
                            "Use line for one item per line, blank line for paragraph/chunk lists, comma for a,b,c, pipe for a|b|c, "
                            "or json/python for structured input only."
                        ),
                    },
                ),
                "selection_mode": (
                    ["index", "random", "random no repeat"],
                    {
                        "tooltip": (
                            "How items are chosen. index selects by index and wraps around the list. random picks a seeded random item for each index. "
                            "random no repeat creates a seeded shuffled order, walks through every item once, then reshuffles for the next cycle. "
                            "Example with 5 items: indexes 0-4 use all 5 in random order before any item repeats."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": (
                            "Controls the random order for random and random no repeat modes. Same seed plus same index gives the same result, "
                            "which makes workflows repeatable. Change the seed to get a different random order. Ignored by normal index mode."
                        ),
                    },
                ),
                "multi_format": (
                    ["auto", "lines", "comma", "sentence"],
                    {
                        "tooltip": (
                            "How multiple selected items are combined in formatted_text. auto uses the two_item_template when pick_count is 2, "
                            "otherwise it uses commas. lines outputs each selected item on its own line. comma outputs item1, item2, item3. "
                            "sentence uses the two_item_template for exactly 2 items."
                        ),
                    },
                ),
                "two_item_template": (
                    "STRING",
                    {
                        "default": "start with {item1} then follow with {item2}",
                        "multiline": False,
                        "tooltip": (
                            "Editable sentence template used when pick_count is 2 and multi_format is auto or sentence. "
                            "Keep {item1} and {item2} where you want the selected items inserted. You can also use {items} for both items joined by commas. "
                            "Example: begin with {item1}, then transition into {item2}."
                        ),
                    },
                ),
                "keep_empty": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Whether blank entries count as selectable items. Usually leave this off. "
                            "If off, blank lines are ignored. If on, blank lines can be selected and may output an empty value, "
                            "for example: Camera Motion = ."
                        ),
                    },
                ),
                "pick_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 50,
                        "tooltip": (
                            "How many items to select at once. Use 1 for a single motion. Use 2 to combine two motions, "
                            "such as: Camera Motion = start with slow push in then follow with orbit left. "
                            "For more than 2, the selected items can be output as lines or comma-separated text."
                        ),
                    },
                ),
            }
        }

    @staticmethod
    def _stringify_item(value):
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list, tuple, set)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    @classmethod
    def _items_from_structured_value(cls, value):
        if isinstance(value, dict):
            for key in ("items", "values", "motions", "camera_motions", "camera motions"):
                found = value.get(key)
                if isinstance(found, (list, tuple, set)):
                    return [cls._stringify_item(item) for item in found]
            return [cls._stringify_item(item) for item in value.values()]
        if isinstance(value, (list, tuple, set)):
            return [cls._stringify_item(item) for item in value]
        return None

    @classmethod
    def _parse_structured_items(cls, text):
        stripped = str(text or "").strip()
        if not stripped:
            return []

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except Exception:
                continue
            items = cls._items_from_structured_value(parsed)
            if items is not None:
                return items

        return None

    @staticmethod
    def _clean_split_item(item):
        cleaned = str(item or "").strip()
        cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", cleaned)
        return cleaned.strip().strip(",")

    @classmethod
    def _split_items(cls, text, split_mode):
        raw = str(text or "")
        mode = str(split_mode or "auto").strip().lower()

        if mode in ("auto", "json/python"):
            structured = cls._parse_structured_items(raw)
            if structured is not None:
                return structured
            if mode == "json/python":
                return []

        if mode == "blank line" or (mode == "auto" and re.search(r"\n\s*\n", raw)):
            return re.split(r"\n\s*\n+", raw.strip())
        if mode == "comma" or (mode == "auto" and "\n" not in raw and "," in raw):
            return raw.split(",")
        if mode == "pipe" or (mode == "auto" and "\n" not in raw and "|" in raw):
            return raw.split("|")
        return raw.splitlines() if "\n" in raw else [raw]

    @classmethod
    def _parse_items(cls, text, split_mode, keep_empty):
        items = [cls._clean_split_item(item) for item in cls._split_items(text, split_mode)]
        if not keep_empty:
            items = [item for item in items if item]
        return items

    @staticmethod
    def _random_index(index, item_count, seed):
        rng = random.Random(f"{int(seed)}:{int(index)}:{item_count}")
        return rng.randrange(item_count)

    @classmethod
    def _random_no_repeat_index(cls, index, item_count, seed):
        if item_count <= 1:
            return 0

        index = int(index)
        cycle = index // item_count
        offset = index % item_count

        rng = random.Random(f"{int(seed)}:{cycle}:{item_count}")
        order = list(range(item_count))
        rng.shuffle(order)

        if cycle > 0:
            prev_rng = random.Random(f"{int(seed)}:{cycle - 1}:{item_count}")
            prev_order = list(range(item_count))
            prev_rng.shuffle(prev_order)
            if order[0] == prev_order[-1]:
                order[0], order[1] = order[1], order[0]

        return order[offset]

    @classmethod
    def _select_index(cls, index, item_count, selection_mode, seed):
        mode = str(selection_mode or "index").strip().lower()
        if mode == "random":
            return cls._random_index(index, item_count, seed)
        if mode == "random no repeat":
            return cls._random_no_repeat_index(index, item_count, seed)
        return int(index) % item_count

    @staticmethod
    def _format_selected_items(selected_items, multi_format, two_item_template):
        count = len(selected_items)
        if count <= 0:
            return ""
        if count == 1:
            return selected_items[0]

        mode = str(multi_format or "auto").strip().lower()
        if count == 2 and mode in ("auto", "sentence"):
            template = str(two_item_template or "").strip()
            if not template:
                template = "start with {item1} then follow with {item2}"
            try:
                return template.format(
                    item1=selected_items[0],
                    item2=selected_items[1],
                    items=", ".join(selected_items),
                )
            except Exception:
                return f"start with {selected_items[0]} then follow with {selected_items[1]}"

        if mode == "lines":
            return "\n".join(selected_items)
        return ", ".join(selected_items)

    def run(
        self,
        index,
        items,
        label,
        max_items,
        split_mode,
        selection_mode,
        seed,
        multi_format,
        two_item_template,
        keep_empty,
        pick_count,
    ):
        parsed_items = self._parse_items(items, split_mode, keep_empty)
        if max_items and max_items > 0:
            parsed_items = parsed_items[:max_items]

        item_count = len(parsed_items)
        if item_count <= 0:
            return ("", "", "", 0, 0)

        selected_indexes = [
            self._select_index(int(index) + i, item_count, selection_mode, seed)
            for i in range(max(1, int(pick_count)))
        ]
        wrapped_index = selected_indexes[0]
        selected_items = [parsed_items[i] for i in selected_indexes]
        selected_item = selected_items[0]
        selected_items_text = "\n".join(selected_items)
        formatted_value = self._format_selected_items(
            selected_items,
            multi_format,
            two_item_template,
        )
        label_text = str(label or "").strip()
        formatted_text = f"{label_text} = {formatted_value}" if label_text else formatted_value
        return (formatted_text, selected_item, selected_items_text, wrapped_index, item_count)


class VRGDG_MultiCyclingTextPicker(VRGDG_CyclingTextPicker):
    MAX_PICKERS = 20
    PRESET_LABELS = [
        "Camera Motion",
        "Character Movement/Motion",
        "Lighting",
        "Time of Day",
        "Weather",
        "Dialogue",
        "Facial Expression",
        "Emotion",
        "Custom",
    ]
    PRESET_ITEMS = {
        "Camera Motion": """Slow push-in
Track right
Track left
Dolly backward
Handheld follow
Over-the-shoulder push-in
Slow pan right
Slow pan left
Tilt up
Tilt down
Arc around subject
Orbit shot
Low-angle tracking shot
Crane rising move
Slow zoom-in""",
        "Character Movement/Motion": """Walks toward camera with confident swagger
Strides across the frame
Leans toward the camera
Points into the lens
Throws arms wide
Raises both hands overhead
Runs a hand through their hair
Slowly backs away from the camera
Drops to one knee
Throws their head back
Whips a jacket off one shoulder
Stomps forward with attitude
Tilts chin upward
Reaches toward the camera
Collapses dramatically to the floor""",
        "Lighting": """Soft natural light
Hard direct sunlight
Warm tungsten light
Cool fluorescent light
Neon nightclub light
Moody low-key lighting
High-key studio lighting
Backlit silhouette
Rim lighting
Side lighting
Top-down lighting
Underlighting
Golden hour light
Blue hour light
Strobe lighting""",
        "Time of Day": """Pre-dawn
Dawn
Early morning
Mid-morning
Late morning
Noon
Early afternoon
Mid-afternoon
Late afternoon
Golden hour
Sunset
Dusk
Blue hour
Night
After midnight""",
        "Weather": """Clear sky
Partly cloudy
Overcast
Light rain
Heavy rain
Thunderstorm
Drizzle
Fog
Mist
Snowfall
Blizzard
Hail
Strong wind
Dust storm
Humid haze""",
        "Dialogue": "",
        "Facial Expression": """Calm expression
Serious expression
Confident smirk
Cold stare
Worried expression
Sad expression
Angry glare
Fearful expression
Surprised expression
Blank expression
Dreamy expression
Suspicious look
Pained expression
Defiant expression
Soft smile""",
        "Emotion": """Joyful
Melancholic
Anxious
Furious
Heartbroken
Hopeful
Jealous
Lonely
Nostalgic
Conflicted
Euphoric
Ashamed
Determined
Vengeful
Peaceful""",
        "Custom": "",
    }
    RETURN_TYPES = ("STRING", "STRING") + tuple(["STRING"] * MAX_PICKERS)
    RETURN_NAMES = (
        "combined_formatted_text",
        "results_json",
    ) + tuple(f"formatted_text_{i}" for i in range(1, MAX_PICKERS + 1))
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "picker_count": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": cls.MAX_PICKERS,
                    "tooltip": "How many independent cycling text pickers to use. Set 0 to disable advanced prompt details.",
                },
            ),
            "joiner": (
                ["newline", "blank line", "comma", "pipe"],
                {
                    "tooltip": "How to combine the active formatted_text outputs into combined_formatted_text.",
                },
            ),
        }

        optional = {}
        for i in range(1, cls.MAX_PICKERS + 1):
            picker_inputs = cls._picker_input_types(i)
            for name, input_type in picker_inputs.items():
                if name.startswith("index_") or name.startswith("seed_"):
                    optional[name] = input_type
                else:
                    required[name] = input_type

        return {"required": required, "optional": optional}

    @staticmethod
    def _picker_input_types(i):
        return {
            f"preset_{i}": (
                "STRING",
                {
                    "default": "Camera Motion" if i == 1 else "Custom",
                    "multiline": False,
                    "tooltip": f"Picker {i}: choose a preset label/list, or Custom for your own label and editable list.",
                },
            ),
            f"index_{i}": (
                "INT",
                {
                    "default": 0,
                    "forceInput": True,
                    "min": -999999,
                    "max": 999999,
                    "tooltip": f"Picker {i}: step number used to choose from this picker list.",
                },
            ),
            f"items_{i}": (
                "STRING",
                {
                    "default": VRGDG_MultiCyclingTextPicker.PRESET_ITEMS["Camera Motion"] if i == 1 else "",
                    "multiline": True,
                    "tooltip": f"Picker {i}: list of text choices. Supports lines, blank-line chunks, comma, pipe, JSON/Python lists, or objects with items/values/motions.",
                },
            ),
            f"label_{i}": (
                "STRING",
                {
                    "default": "Camera Motion" if i == 1 else "",
                    "multiline": False,
                    "tooltip": f"Picker {i}: optional label placed before this picker output.",
                },
            ),
            f"selection_mode_{i}": (
                "STRING",
                {
                    "default": "index",
                    "multiline": False,
                    "tooltip": f"Picker {i}: choose by wrapped index, seeded random, or seeded random no-repeat.",
                },
            ),
            f"seed_{i}": (
                "INT",
                {
                    "default": 0,
                    "forceInput": True,
                    "min": -0xFFFFFFFFFFFFFFFF,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": f"Picker {i}: seed used by random modes.",
                },
            ),
            f"two_item_template_{i}": (
                "STRING",
                {
                    "default": "start with {item1} then follow with {item2}",
                    "multiline": False,
                    "tooltip": f"Picker {i}: sentence template used when this picker selects exactly two items.",
                },
            ),
            f"pick_count_{i}": (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": 50,
                    "tooltip": f"Picker {i}: how many items to select from this picker at once.",
                },
            ),
        }

    @staticmethod
    def _join_formatted_text(values, joiner):
        mode = str(joiner or "newline").strip().lower()
        if mode == "blank line":
            return "\n\n".join(values)
        if mode == "comma":
            return ", ".join(values)
        if mode == "pipe":
            return " | ".join(values)
        return "\n".join(values)

    @staticmethod
    def _extract_item_directives(raw_items):
        text = str(raw_items or "")
        lines = text.splitlines()
        directives = {}
        body_lines = []
        in_directive_header = True
        directive_names = {
            "LABEL": "label",
            "SELECTION_MODE": "selection_mode",
            "PICK_COUNT": "pick_count",
            "TEMPLATE": "template",
        }

        for line in lines:
            stripped = line.strip()
            directive_key = None
            directive_value = None
            if stripped.startswith("#") and in_directive_header:
                raw_directive = stripped[1:].strip()
                if ":" in raw_directive:
                    raw_name, raw_value = raw_directive.split(":", 1)
                    raw_name = raw_name.strip().upper()
                    if raw_name.startswith("VRGDG_"):
                        raw_name = raw_name[6:]
                    directive_key = directive_names.get(raw_name)
                    directive_value = raw_value.strip()

            if directive_key:
                directives[directive_key] = directive_value
                continue

            in_directive_header = False
            body_lines.append(line)

        return directives, "\n".join(body_lines)

    def _run_one_picker(self, i, kwargs):
        preset = str(kwargs.get(f"preset_{i}", "Custom") or "Custom")
        raw_items = kwargs.get(f"items_{i}", "")
        directives, raw_items = self._extract_item_directives(raw_items)
        if not str(raw_items or "").strip() and preset in self.PRESET_ITEMS:
            raw_items = self.PRESET_ITEMS[preset]
        parsed_items = self._parse_items(
            raw_items,
            "auto",
            False,
        )

        item_count = len(parsed_items)
        if item_count <= 0:
            return {
                "picker": i,
                "formatted_text": "",
                "selected_item": "",
                "selected_items": [],
                "wrapped_index": 0,
                "item_count": 0,
            }

        base_index = int(kwargs.get(f"index_{i}", 0) or 0)
        pick_count_raw = directives.get("pick_count", kwargs.get(f"pick_count_{i}", 1))
        pick_count = max(1, int(pick_count_raw or 1))
        selection_mode = directives.get("selection_mode") or kwargs.get(f"selection_mode_{i}", "index")
        template = directives.get("template") or kwargs.get(f"two_item_template_{i}", "")
        selected_indexes = [
            self._select_index(
                base_index + offset,
                item_count,
                selection_mode,
                kwargs.get(f"seed_{i}", 0),
            )
            for offset in range(pick_count)
        ]
        selected_items = [parsed_items[index] for index in selected_indexes]
        formatted_value = self._format_selected_items(
            selected_items,
            "auto",
            template,
        )
        label_text = directives.get("label") or str(kwargs.get(f"label_{i}", "") or "").strip()
        if not label_text and preset != "Custom":
            label_text = preset
        if not label_text:
            normalized_items = "\n".join(parsed_items).strip()
            for preset_name, preset_items in self.PRESET_ITEMS.items():
                if normalized_items == str(preset_items or "").strip():
                    label_text = preset_name
                    break
        formatted_text = f"{label_text} = {formatted_value}" if label_text else formatted_value
        return {
            "picker": i,
            "formatted_text": formatted_text,
            "selected_item": selected_items[0],
            "selected_items": selected_items,
            "wrapped_index": selected_indexes[0],
            "item_count": item_count,
        }

    def run(self, picker_count=2, joiner="newline", **kwargs):
        active_count = max(0, min(self.MAX_PICKERS, int(picker_count or 0)))
        if active_count <= 0:
            return ("", "[]", *([""] * self.MAX_PICKERS))
        results = [self._run_one_picker(i, kwargs) for i in range(1, active_count + 1)]
        formatted_outputs = [result["formatted_text"] for result in results]
        active_formatted_outputs = [value for value in formatted_outputs if value]
        combined = self._join_formatted_text(active_formatted_outputs, joiner)
        results_json = json.dumps(results, ensure_ascii=False, indent=2)
        padded_outputs = formatted_outputs + [""] * (self.MAX_PICKERS - len(formatted_outputs))
        return (combined, results_json, *padded_outputs)


class VRGDG_EasyMultiCyclingTextPicker(VRGDG_MultiCyclingTextPicker):
    CATEGORY = "VRGDG/General"
    DESCRIPTION = "A friendlier UI wrapper around VRGDG_MultiCyclingTextPicker. Uses the same picker behavior and outputs."


class VRGDG_SaveTextAdvancedConcat:
    RETURN_TYPES = ("STRING", "STRING", "JSON", "STRING")
    RETURN_NAMES = ("text", "file_path", "json", "json_string")
    FUNCTION = "run"
    CATEGORY = "VRGDG/General"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_name": ("STRING", {"default": "story"}),
                "file_name": ("STRING", {"default": "story"}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "concat": ("BOOLEAN", {"default": False}),
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "trigger": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    def _clean_prompt_for_json(self, text):
        return "\n".join(
            line.rstrip()
            for line in str(text or "").splitlines()
            if line.strip()
        ).strip()

    def _load_existing_prompt_json(self, json_path, existing_text):
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    prompts = [
                        str(loaded[key])
                        for key in sorted(
                            loaded.keys(),
                            key=lambda key: int(key[6:]) if re.fullmatch(r"Prompt\d+", key) else 999999,
                        )
                        if str(loaded[key]).strip()
                    ]
                    if prompts:
                        return {
                            f"Prompt{i}": prompt
                            for i, prompt in enumerate(prompts, start=1)
                        }
            except Exception as exc:
                print(f"[VRGDG_SaveTextAdvancedConcat] Could not read JSON sidecar: {exc}")

        cleaned_existing = self._clean_prompt_for_json(existing_text)
        if cleaned_existing:
            return {"Prompt1": cleaned_existing}
        return {}

    def _build_prompt_json(self, existing_prompt_json, text_to_add):
        prompt_json = dict(existing_prompt_json or {})
        cleaned_prompt = self._clean_prompt_for_json(text_to_add)
        if cleaned_prompt:
            prompt_json[f"Prompt{len(prompt_json) + 1}"] = cleaned_prompt
        return prompt_json, json.dumps(prompt_json, ensure_ascii=False, indent=2)

    def run(self, folder_name, file_name, overwrite, concat, text, trigger):
        folder_path, _ = _get_text_files_manual_folder(folder_name)
        os.makedirs(folder_path, exist_ok=True)

        safe_base_name = _sanitize_text_segment(file_name, "text")
        if concat:
            final_name = f"{safe_base_name}.txt"
        elif overwrite:
            final_name = f"{safe_base_name}.txt"
        else:
            final_name = _next_incremental_prefixed_file_name(folder_path, safe_base_name)

        file_path = os.path.normpath(os.path.join(folder_path, final_name))
        json_path = os.path.splitext(file_path)[0] + ".json"
        text_to_write = _coerce_text_payload(text)
        saved_text = text_to_write
        existing_text = ""

        if concat and os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_text = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    existing_text = f.read()

            if existing_text and text_to_write:
                saved_text = existing_text.rstrip("\r\n") + "\n\n" + text_to_write.lstrip("\r\n")
            elif existing_text:
                saved_text = existing_text

        print(
            f"[VRGDG_SaveTextAdvancedConcat] len={len(saved_text)} concat={bool(concat)} "
            f"start={repr(saved_text[:1])} end={repr(saved_text[-1:])}"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(saved_text)

        existing_prompt_json = {}
        if concat:
            existing_prompt_json = self._load_existing_prompt_json(json_path, existing_text)

        prompt_json, prompt_json_string = self._build_prompt_json(existing_prompt_json, text_to_write)

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(prompt_json_string)

        return (saved_text, file_path, prompt_json, prompt_json_string)


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
    "VRGDG_CyclingTextPicker": VRGDG_CyclingTextPicker,
    "VRGDG_MultiCyclingTextPicker": VRGDG_MultiCyclingTextPicker,
    "VRGDG_EasyMultiCyclingTextPicker": VRGDG_EasyMultiCyclingTextPicker,
    "VRGDG_SaveTextAdvancedConcat": VRGDG_SaveTextAdvancedConcat,
    

    
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
    "VRGDG_CyclingTextPicker": "VRGDG Cycling Text Picker",
    "VRGDG_MultiCyclingTextPicker": "VRGDG Multi Cycling Text Picker",
    "VRGDG_EasyMultiCyclingTextPicker": "VRGDG Easy Multi Cycling Text Picker",
    "VRGDG_SaveTextAdvancedConcat": "VRGDG_SaveTextAdvancedConcat",
    
}







