import json
import os
import re
import time
import base64
import math
from io import BytesIO
from urllib.parse import quote

import folder_paths
import torch
import torchaudio.functional as AF
from aiohttp import web
from PIL import Image
from server import PromptServer


class AnyType(str):
    def __ne__(self, value):
        return False


any_typ = AnyType("*")
_VRGDG_VIDEO_EDITOR_ROUTES_REGISTERED = False
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".avi")
_VISUAL_T2I_INSTRUCTIONS = """Create one text-to-image prompt from the provided image and user input.

User input includes:
- a reference image
- optional user notes

Use all parts of the input together.

Priority:
- Use the provided image as the main visual foundation.
- Preserve the visible subject, setting, outfit, mood, lighting, and scene identity from the image unless the user clearly asks to change them.
- Use the user notes to adjust framing, pose, camera distance, lighting, mood, action, wardrobe refinement, or environment details.

Rules:
- Create one polished text-to-image prompt.
- Treat the provided image as the base scene reference.
- Describe only what should be visible in the final generated image.
- If the user asks for a closer shot, wider shot, different pose, different lighting, different camera angle, or different mood, apply that change while keeping the same core subject and scene identity.
- Keep the image prompt concrete and visual.
- Do not use metaphors, abstract symbolic wording, or non-visible language.
- Do not explain your choices.
- Only send the final prompt text.

Use this exact format:

A high resolution cinematic photograph of a [subject], [action or pose based on the reference image and user notes], in [environment/location based on the reference image], during [time of day]. The subject is wearing [main outfit visible in the reference image refined by the user notes], [shoes/accessories visible in the reference image refined by the user notes], and [additional visible style details inspired by the user notes]. Their hair is [hair color], [hair length/style], and [movement or texture]. The environment is [visual style of location from the reference image shaped by the user notes] with [background details visible in the reference image], [lighting and color details based on the reference image and user notes], and [surface/reflection/material details connected to the reference image]. Camera is [camera angle/framing requested by the user or inferred from the image] with a [lens type or framing]. The weather is [weather condition appropriate to the scene], with [atmospheric detail influenced by the reference image and user notes], creating a [mood] mood.

[subject] = character gender! don't just say "subject"!

Only send the final prompt text. Do not include labels, notes, quotes, or extra text.

User Input:"""

_TEXT_ONLY_T2I_INSTRUCTIONS = """Create one text-to-image prompt from the user input.

User input includes:
- user notes describing the desired image

Use the user notes as the full scene foundation. Preserve all concrete visible details from the user notes, including subject, setting, outfit, pose, mood, lighting, camera framing, and environment details. If details are missing, infer only the visual details needed to make one complete image prompt.

Rules:
- Create one polished text-to-image prompt.
- Keep the image prompt concrete and visual.
- Do not use metaphors, abstract symbolic wording, or non-visible language.
- Do not explain your choices.
- Only send the final prompt text.

Use this exact format:

A high resolution cinematic photograph of a [subject], [action or pose based on the user notes], in [environment/location based on the user notes], during [time of day]. The subject is wearing [main outfit based on the user notes], [shoes/accessories based on the user notes], and [additional visible style details based on the user notes]. Their hair is [hair color], [hair length/style], and [movement or texture]. The environment is [visual style of location from the user notes] with [background details], [lighting and color details], and [surface/reflection/material details]. Camera is [camera angle/framing requested by the user or inferred from the notes] with a [lens type or framing]. The weather is [weather condition appropriate to the scene], with [atmospheric detail], creating a [mood] mood.

[subject] = character gender! don't just say "subject"!

Only send the final prompt text. Do not include labels, notes, quotes, or extra text.

User Input:"""

_I2V_INSTRUCTIONS = """Convert the user's text-to-image prompt into a dynamic image-to-video prompt.

Use the image prompt only as the visual reference. Preserve the original subject, setting, outfit, mood, atmosphere, and scene identity. Do not repeat or describe color grading, lighting style, camera photo style, or static image-quality terms unless needed for motion clarity.

Add fast, cinematic motion by giving the subject a clear action sequence, expressive face expressions and body movement, strong gestures, and intentional camera movement. Keep the subject visible and framed throughout.

Output one polished paragraph using this structure:

The [Subject] who is singing with passion in [setting/environment] during [time/weather]. The subject [dynamic performance action]. Their clothing/hair [reacts to movement]. The camera [Camera Motion] while maintaining [subject visibility]. The environment [reacts dynamically].

Each word in brackets should be chosen based on user input that would best fit the scene.
NOTE: DO NOT USE ORBIT TYPE CAMERA MOTION, DO NOT USE THE WORD "SPIN" SUBJECT SHOULD NEVER SPIN.
Subject should always be physically singing!

Do not add audio, dialogue, captions, text overlays, unrelated characters, new locations, major story changes, color style, lighting style, or image-quality descriptions. Keep it vivid, fast, cinematic, dynamic, and video-ready.

User input always takes priority over the text-to-image prompt when the user asks for specific camera motion, character movement, performance direction, or scene changes.

User Input, must follow:"""


def _resolve_editor_folder(raw_path):
    text = str(raw_path or "").strip().strip('"')
    if not text:
        raise ValueError("Output folder path is empty.")

    candidates = []
    if os.path.isabs(text):
        candidates.append(text)
    else:
        candidates.extend(
            [
                text,
                os.path.join(folder_paths.get_output_directory(), text),
                os.path.join(folder_paths.get_input_directory(), text),
            ]
        )
        get_temp = getattr(folder_paths, "get_temp_directory", None)
        if callable(get_temp):
            candidates.append(os.path.join(get_temp(), text))

    for candidate in candidates:
        folder = os.path.normpath(os.path.abspath(candidate))
        if os.path.isdir(folder):
            return folder

    raise FileNotFoundError(f"Output folder was not found: {text}")


def _parse_extensions(raw_extensions):
    values = []
    for item in re.split(r"[,;\s]+", str(raw_extensions or ""), flags=re.IGNORECASE):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        values.append(ext)
    return tuple(values or _VIDEO_EXTENSIONS)


def _natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text or ""))]


def _guess_clip_number(filename, fallback_index):
    match = re.match(r"video_(\d+)", str(filename or ""), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", str(filename or ""))
    if match:
        return int(match.group(1))
    return fallback_index


def _session_path(folder):
    return os.path.join(folder, "vrgdg_temp", "editor_session.json")


def _frames_folder(folder):
    return os.path.join(folder, "vrgdg_editor_frames")


def _round_up_8n1(n):
    n = max(1, int(n))
    return ((n - 1 + 7) // 8) * 8 + 1


def _format_seconds(sec):
    sec = max(0.0, float(sec or 0.0))
    minutes = int(sec // 60)
    seconds = sec % 60
    return f"{minutes}:{seconds:06.3f}"


def _parse_srt_segments(path):
    srt_path = str(path or "").strip().strip('"')
    if not srt_path or not os.path.isfile(srt_path):
        raise FileNotFoundError(f"SRT file was not found: {srt_path}")

    with open(srt_path, "r", encoding="utf-8-sig") as handle:
        text = handle.read().strip()
    blocks = re.split(r"\n\s*\n", text)
    segments = []

    def to_seconds(value):
        match = re.match(r"\s*(\d+):(\d+):(\d+),(\d+)\s*", str(value or ""))
        if not match:
            raise ValueError(f"Invalid SRT timecode: {value}")
        hours, minutes, seconds, millis = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_line = next((line for line in lines if "-->" in line), "")
        if not time_line:
            continue
        start_text, end_text = time_line.split("-->", 1)
        segments.append((to_seconds(start_text), to_seconds(end_text)))

    if not segments:
        raise ValueError("No valid SRT entries were found.")
    return segments


def _load_editor_session_file(session_path):
    path = str(session_path or "").strip().strip('"')
    if not path:
        raise ValueError("session_path is empty.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Editor session file was not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Editor session must be a JSON object.")
    clips = data.get("clips", {})
    if not isinstance(clips, dict):
        raise ValueError("Editor session JSON does not contain a valid clips object.")
    return path, data, clips


def _selected_editor_clips(clips_obj):
    items = [item for item in clips_obj.values() if isinstance(item, dict) and item.get("selected_for_remake")]
    items.sort(key=lambda item: int(item.get("clip_number", 0) or 0))
    return items


def _list_clips(folder_path, raw_extensions):
    folder = _resolve_editor_folder(folder_path)
    extensions = _parse_extensions(raw_extensions)
    clips = []

    for filename in os.listdir(folder):
        full_path = os.path.join(folder, filename)
        if not os.path.isfile(full_path):
            continue
        lower = filename.lower()
        if not lower.endswith(extensions):
            continue
        if lower.startswith("final_video") or lower == "00001.mp4":
            continue
        try:
            stat = os.stat(full_path)
        except OSError:
            continue
        clips.append(
            {
                "name": filename,
                "path": full_path,
                "size": int(stat.st_size),
                "mtime": float(stat.st_mtime),
                "clip_number": 0,
                "url": f"/vrgdg/video_editor/video?path={quote(full_path)}&v={int(stat.st_mtime)}_{int(stat.st_size)}",
            }
        )

    clips.sort(key=lambda item: _natural_key(item["name"]))
    for index, item in enumerate(clips, start=1):
        item["clip_number"] = _guess_clip_number(item["name"], index)

    return {
        "folder_path": folder,
        "remake_folder": os.path.join(folder, "remake"),
        "session_path": _session_path(folder),
        "clips": clips,
    }


def _load_session(folder_path):
    folder = _resolve_editor_folder(folder_path)
    path = _session_path(folder)
    if not os.path.isfile(path):
        return {"project_folder": folder, "clips": {}, "updated": None}
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Editor session must be a JSON object.")
    return data


def _save_session(folder_path, session):
    folder = _resolve_editor_folder(folder_path)
    if not isinstance(session, dict):
        raise ValueError("Session must be a JSON object.")
    path = _session_path(folder)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(session)
    staged = _stage_selected_remakes(folder, payload)
    payload = {
        **payload,
        "project_folder": folder,
        "updated": time.time(),
        "staged_remakes": staged,
    }
    _clear_remake_queue_state(folder)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path, payload


def _clear_remake_queue_state(folder):
    state_path = os.path.join(folder, "vrgdg_temp", "remake_clip_queue_state.json")
    queue_cls = globals().get("VRGDG_RemakeClipQueue")
    if queue_cls is not None:
        try:
            key = os.path.normcase(os.path.abspath(os.path.join(folder, "vrgdg_temp", "editor_session.json")))
            queue_cls._autoqueue_memory.pop(key, None)
        except Exception:
            pass
    try:
        if os.path.isfile(state_path):
            os.remove(state_path)
    except Exception:
        pass


def _stage_selected_remakes(folder, session):
    clips = session.get("clips", {}) if isinstance(session, dict) else {}
    if not isinstance(clips, dict):
        return []
    remake_dir = os.path.join(folder, "remake")
    os.makedirs(remake_dir, exist_ok=True)
    staged = []
    for item in clips.values():
        if not isinstance(item, dict) or not item.get("selected_for_remake"):
            continue
        clip_path = str(item.get("path", "") or "").strip()
        basename = os.path.basename(clip_path) if clip_path else str(item.get("name", "") or "").strip()
        if not basename:
            continue
        main_path = os.path.join(folder, basename)
        remake_path = os.path.join(remake_dir, basename)
        if os.path.isfile(remake_path):
            item["path"] = remake_path
            staged.append({"name": basename, "from": "", "to": remake_path, "already_staged": True})
            continue
        if not os.path.isfile(main_path):
            continue
        os.replace(main_path, remake_path)
        item["path"] = remake_path
        staged.append({"name": basename, "from": main_path, "to": remake_path, "already_staged": False})
    return staged


def _safe_frame_filename(clip_name, frame_time):
    stem = os.path.splitext(os.path.basename(str(clip_name or "clip")))[0]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "clip"
    time_tag = f"{max(0.0, float(frame_time or 0.0)):09.3f}".replace(".", "_")
    return f"{stem}_frame_{time_tag}.png"


def _image_from_data_url(data_url):
    text = str(data_url or "").strip()
    match = re.match(r"^data:image/(?:png|jpeg|jpg|webp);base64,(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Expected a base64 image data URL.")
    raw = base64.b64decode(match.group(1))
    return Image.open(BytesIO(raw)).convert("RGB")


def _save_editor_frame(payload):
    folder = _resolve_editor_folder(payload.get("folder_path", ""))
    image = _image_from_data_url(payload.get("image_data", ""))
    clip_name = str(payload.get("clip_name", "") or "clip")
    frame_time = float(payload.get("frame_time", 0.0) or 0.0)
    target_dir = _frames_folder(folder)
    os.makedirs(target_dir, exist_ok=True)
    frame_path = os.path.join(target_dir, _safe_frame_filename(clip_name, frame_time))
    image.save(frame_path, format="PNG")
    return {
        "frame_path": frame_path,
        "frames_folder": target_dir,
        "filename": os.path.basename(frame_path),
    }


def _clean_visual_gemma_text(text):
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"^(?:Assistant|Answer|Final prompt)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    control_patterns = [
        r"_?\s*<\|channel>\s*(?:thought|analysis|reasoning)?\s*",
        r"_?\s*<\|?channel\|?>\s*(?:thought|analysis|reasoning)?\s*",
        r"_?\s*<channel\|>\s*(?:thought|analysis|reasoning)?\s*",
        r"^\s*(?:thought|analysis|reasoning)\s*[:\-]?\s*",
    ]
    previous = None
    while cleaned and previous != cleaned:
        previous = cleaned
        for pattern in control_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"^(?:Assistant|Answer|Final prompt)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    return paragraphs[0] if paragraphs else cleaned


def _clean_gemma_prompt_text(text):
    return _clean_visual_gemma_text(text)


def _generate_visual_t2i_prompt(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    frame_path = str(payload.get("frame_path", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    has_frame = bool(frame_path and os.path.isfile(frame_path))
    if not model_file:
        raise ValueError("Choose a Gemma4 model first.")
    if has_frame and not mmproj_file:
        raise ValueError("Choose an mmproj file first.")
    if not has_frame and not user_notes:
        raise ValueError("Enter generation notes or capture a frame first.")

    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    mmproj_path = llm._resolve_dropdown_path(mmproj_file, llm.MISSING_MMPROJ_OPTION) if has_frame else ""
    image = Image.open(frame_path).convert("RGB") if has_frame else None
    if has_frame:
        prompt = _VISUAL_T2I_INSTRUCTIONS
        if user_notes:
            prompt += f"\n\nUser notes:\n{user_notes}"
        else:
            prompt += "\n\nUser notes:\nUse the reference image as the guide."
    else:
        prompt = f"{_TEXT_ONLY_T2I_INSTRUCTIONS}\n\nUser notes:\n{user_notes}"

    n_ctx = int(payload.get("n_ctx") or 13000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.6)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or 1200)
    unload_after = bool(payload.get("unload_after"))

    try:
        model = llm._load_gguf_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
            mmproj_path=mmproj_path,
        )
        if has_frame:
            text = llm._run_gguf_vision_pipeline(
                model=model,
                pil_images=[image],
                instruction_text=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        else:
            text = llm._run_gguf_text_pipeline(
                model=model,
                instruction_text=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        text = _clean_visual_gemma_text(text)
        if not text:
            raise ValueError("Gemma returned an empty prompt.")
        return {
            "prompt": text,
            "frame_path": frame_path,
            "used_frame": has_frame,
            "used_model": model_path,
            "used_mmproj": mmproj_path,
            "unloaded": unload_after,
        }
    finally:
        if unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )


def _generate_i2v_prompt(payload):
    from .LLM import VRGDG_SuperGemmaGGUFChat

    model_file = str(payload.get("model_file", "") or "").strip()
    t2i_prompt = str(payload.get("t2i_prompt", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    if not model_file:
        raise ValueError("Choose an I2V Gemma model first.")
    if not model_file.lower().endswith(".gguf"):
        raise ValueError("The I2V model field is not a GGUF model. Recreate this editor node or choose a Gemma .gguf in the I2V model dropdown.")
    if not t2i_prompt:
        raise ValueError("Create or paste a T2I prompt first.")

    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    prompt = (
        f"{_I2V_INSTRUCTIONS}\n\n"
        f"Text-to-image prompt:\n{t2i_prompt}\n\n"
    )
    if user_notes:
        prompt += f"User motion/camera notes:\n{user_notes}"
    else:
        prompt += "User motion/camera notes:\nCreate fast cinematic performance motion that fits the scene."

    n_ctx = int(payload.get("n_ctx") or 13000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.7)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or 1200)
    unload_after = bool(payload.get("unload_after"))

    try:
        model = llm._load_gguf_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
            mmproj_path="",
        )
        text = llm._run_gguf_text_pipeline(
            model=model,
            instruction_text=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        text = _clean_gemma_prompt_text(text)
        if not text:
            raise ValueError("Gemma returned an empty I2V prompt.")
        return {
            "prompt": text,
            "used_model": model_path,
            "unloaded": unload_after,
        }
    finally:
        if unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path="",
            )


def _ensure_video_editor_routes():
    global _VRGDG_VIDEO_EDITOR_ROUTES_REGISTERED
    if _VRGDG_VIDEO_EDITOR_ROUTES_REGISTERED:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.post("/vrgdg/video_editor/list_clips")
    async def vrgdg_video_editor_list_clips(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _list_clips(payload.get("folder_path", ""), payload.get("extensions", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/video_editor/load_session")
    async def vrgdg_video_editor_load_session(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            session = _load_session(payload.get("folder_path", ""))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "session": session})

    @server_instance.routes.post("/vrgdg/video_editor/save_session")
    async def vrgdg_video_editor_save_session(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            path, session = _save_session(payload.get("folder_path", ""), payload.get("session", {}))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "session_path": path, "session": session})

    @server_instance.routes.post("/vrgdg/video_editor/save_frame")
    async def vrgdg_video_editor_save_frame(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            result = _save_editor_frame(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/video_editor/generate_visual_t2i")
    async def vrgdg_video_editor_generate_visual_t2i(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            import asyncio
            result = await asyncio.to_thread(_generate_visual_t2i_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.post("/vrgdg/video_editor/generate_i2v")
    async def vrgdg_video_editor_generate_i2v(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)
        try:
            import asyncio
            result = await asyncio.to_thread(_generate_i2v_prompt, payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    @server_instance.routes.get("/vrgdg/video_editor/video")
    async def vrgdg_video_editor_video(request):
        raw_path = str(request.query.get("path", "") or "").strip()
        video_path = os.path.normpath(os.path.abspath(raw_path))
        if not os.path.isfile(video_path):
            return web.json_response({"ok": False, "error": "Video file was not found."}, status=404)
        return web.FileResponse(video_path)

    _VRGDG_VIDEO_EDITOR_ROUTES_REGISTERED = True


class VRGDG_VideoEditorUI:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            from .LLM import VRGDG_SuperGemmaGGUFChat
            gemma_choices = VRGDG_SuperGemmaGGUFChat._list_local_gemma_gguf_choices()
            mmproj_choices = VRGDG_SuperGemmaGGUFChat._list_local_mmproj_choices()
        except Exception:
            gemma_choices = ["[No Gemma GGUF found in models/LLM]"]
            mmproj_choices = ["[No mmproj GGUF found in models/LLM]"]
        preferred_i2v = next(
            (
                choice
                for choice in gemma_choices
                if "supergemma4-26b-uncensored-fast-v2_q4_k_m.gguf" in str(choice).lower()
            ),
            gemma_choices[0],
        )
        return {
            "required": {
                "output_folder": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Folder created by the workflow, containing video_0001... clips.",
                        "tooltip": "Project output folder to load into the editor.",
                    },
                ),
                "video_extensions": (
                    "STRING",
                    {
                        "default": ".mp4,.mov,.webm,.mkv",
                        "tooltip": "Comma-separated video extensions to show in the editor.",
                    },
                ),
                "selected_clip_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Clip currently selected in the editor UI.",
                    },
                ),
                "session_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to the saved editor session JSON.",
                    },
                ),
                "model_file": (
                    gemma_choices,
                    {
                        "default": gemma_choices[0],
                        "tooltip": "Gemma GGUF model used by the editor's visual prompt button.",
                    },
                ),
                "mmproj_file": (
                    mmproj_choices,
                    {
                        "default": mmproj_choices[0],
                        "tooltip": "mmproj GGUF used so Gemma can see the captured frame.",
                    },
                ),
                "captured_frame_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Last frame captured by the editor UI.",
                    },
                ),
                "generated_t2i_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Last visual Gemma text-to-image prompt generated by the editor UI.",
                    },
                ),
                "generated_i2v_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Last Gemma image-to-video prompt generated by the editor UI.",
                    },
                ),
                "i2v_model_file": (
                    gemma_choices,
                    {
                        "default": preferred_i2v,
                        "tooltip": "Text-only Gemma GGUF model used by the editor's I2V prompt button. This does not need an mmproj.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_folder", "session_path", "captured_frame_path", "generated_t2i_prompt", "generated_i2v_prompt")
    FUNCTION = "noop"
    CATEGORY = "VRGDG/Video Editor"

    def noop(
        self,
        output_folder,
        video_extensions,
        selected_clip_path,
        session_path,
        model_file,
        mmproj_file,
        captured_frame_path,
        generated_t2i_prompt,
        generated_i2v_prompt,
        i2v_model_file,
    ):
        return (output_folder, session_path, captured_frame_path, generated_t2i_prompt, generated_i2v_prompt)


class VRGDG_VideoEditorSessionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to vrgdg_temp/editor_session.json saved by VRGDG Video Editor UI.",
                    },
                ),
                "clip_number": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 999999,
                        "step": 1,
                        "tooltip": "Clip number to load from the saved editor session.",
                    },
                ),
                "clip_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional exact clip path. If set, this is used before clip_number.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("t2i_prompt", "i2v_prompt", "captured_frame_path", "selected_for_remake", "clip_name", "clip_path")
    FUNCTION = "load"
    CATEGORY = "VRGDG/Video Editor"

    @staticmethod
    def _norm_path(value):
        text = str(value or "").strip().strip('"')
        if not text:
            return ""
        try:
            return os.path.normcase(os.path.normpath(os.path.abspath(text)))
        except Exception:
            return os.path.normcase(os.path.normpath(text))

    def _find_clip(self, clips, clip_number, clip_path):
        wanted_path = self._norm_path(clip_path)
        if wanted_path:
            for key, item in clips:
                item_path = self._norm_path(item.get("path", "") or key)
                if item_path == wanted_path:
                    return item

        wanted_number = int(clip_number)
        for _, item in clips:
            try:
                number = int(item.get("clip_number", 0) or 0)
            except Exception:
                number = 0
            if number == wanted_number:
                return item

        return None

    def load(self, session_path, clip_number, clip_path):
        path = str(session_path or "").strip().strip('"')
        if not path:
            return ("", "", "", False, "", "")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Editor session file was not found: {path}")

        with open(path, "r", encoding="utf-8-sig") as handle:
            session = json.load(handle)
        clips_obj = session.get("clips", {}) if isinstance(session, dict) else {}
        if not isinstance(clips_obj, dict):
            raise ValueError("Editor session JSON does not contain a valid clips object.")

        clips = [(key, item) for key, item in clips_obj.items() if isinstance(item, dict)]
        item = self._find_clip(clips, clip_number, clip_path)
        if item is None:
            return ("", "", "", False, "", "")

        return (
            str(item.get("t2i_prompt", "") or ""),
            str(item.get("i2v_prompt", "") or ""),
            str(item.get("captured_frame_path", "") or ""),
            bool(item.get("selected_for_remake", False)),
            str(item.get("name", "") or ""),
            str(item.get("path", "") or ""),
        )


class VRGDG_RemakeClipQueue:
    _autoqueue_memory = {}

    RETURN_TYPES = (
        "DICT",
        "FLOAT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "INT",
        "INT",
        "DICT",
        "STRING",
        "BOOLEAN",
    ) + ("AUDIO",) + (any_typ, "INT", "STRING", "STRING", "INT")

    RETURN_NAMES = (
        "meta",
        "total_duration",
        "clip_number",
        "frames_for_ltx",
        "start_time",
        "end_time",
        "t2i_prompt",
        "i2v_prompt",
            "captured_frame_path",
            "clip_path",
            "index",
            "total_selected",
            "frames_per_scene",
            "audio_meta",
            "instructions",
        "is_valid",
    ) + ("audio", "signal_out", "pre_frames", "output_folder", "overwrite_mode", "total_sets")

    FUNCTION = "run"
    CATEGORY = "VRGDG/Video Editor"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return time.time()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "trigger": (any_typ,),
                "session_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to vrgdg_temp/editor_session.json saved by the editor.",
                    },
                ),
                "srt_file": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Original SRT file used to make the video clips.",
                    },
                ),
                "queue_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "step": 1,
                        "tooltip": "0 = use internal auto queue position. 1..N = load that selected remake item manually.",
                    },
                ),
                "fps": ("INT", {"default": 24, "min": 1}),
                "enable_auto_queue": ("BOOLEAN", {"default": False}),
                "reset_queue": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Reset the internal remake queue position before this run.",
                    },
                ),
                "tail_loss_frames": ("INT", {"default": 5, "min": 0}),
                "pre_frames": ("INT", {"default": 0, "min": 0}),
            }
        }

    def _empty_audio(self, audio):
        sample_rate = int(audio.get("sample_rate", 44100)) if isinstance(audio, dict) else 44100
        return {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": sample_rate}

    def _queue_state_path(self, session_path):
        return os.path.join(os.path.dirname(str(session_path)), "remake_clip_queue_state.json")

    def _queue_memory_key(self, session_path):
        return os.path.normcase(os.path.abspath(str(session_path or "")))

    def _session_output_folder(self, session_path, session):
        folder = str(session.get("project_folder", "") or "").strip()
        if folder:
            return folder
        return os.path.dirname(os.path.dirname(str(session_path)))

    def _clip_paths_for_folder(self, item, output_folder):
        clip_path = str(item.get("path", "") or "").strip()
        basename = os.path.basename(clip_path) if clip_path else str(item.get("name", "") or "").strip()
        if not basename:
            basename = f"video_{int(item.get('clip_number', 0) or 0):04d}.mp4"
        main_path = os.path.join(output_folder, basename)
        remake_path = os.path.join(output_folder, "remake", basename)
        return main_path, remake_path, basename

    def _file_matches_clip_number(self, filename, clip_number):
        try:
            target = int(clip_number)
        except Exception:
            return False
        match = re.match(r"video_(\d+)", str(filename or ""), flags=re.IGNORECASE)
        if not match:
            return False
        try:
            return int(match.group(1)) == target
        except Exception:
            return False

    def _find_clip_file_in_folder(self, folder, item, fallback_name=""):
        if not folder or not os.path.isdir(folder):
            return ""
        clip_number = int(item.get("clip_number", 0) or 0)
        fallback_name = os.path.basename(str(fallback_name or ""))
        exact = os.path.join(folder, fallback_name) if fallback_name else ""
        if exact and os.path.isfile(exact):
            return exact
        matches = []
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            if self._file_matches_clip_number(filename, clip_number):
                matches.append(path)
        matches.sort(key=lambda value: _natural_key(os.path.basename(value)))
        return matches[0] if matches else ""

    def _prepare_remake_files(self, selected, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        remake_dir = os.path.join(output_folder, "remake")
        backup_dir = os.path.join(output_folder, "backup")
        os.makedirs(remake_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)

        prepared = []
        for item in selected:
            main_path, remake_path, basename = self._clip_paths_for_folder(item, output_folder)
            remake_path = self._find_clip_file_in_folder(remake_dir, item, basename) or remake_path
            remake_exists = os.path.isfile(remake_path)
            backup_path = os.path.join(backup_dir, basename)
            existing_backup = self._find_clip_file_in_folder(backup_dir, item, basename)
            if existing_backup:
                backup_path = existing_backup

            done = bool(existing_backup) and not remake_exists
            pending = bool(remake_exists)
            prepared.append(
                {
                    "item": item,
                    "main_path": main_path,
                    "remake_path": remake_path,
                    "backup_path": backup_path,
                    "basename": basename,
                    "done": done,
                    "pending": pending,
                }
            )
        return prepared

    def _move_selected_remake_to_backup(self, selected_entry, output_folder):
        remake_path = str(selected_entry.get("remake_path", "") or "")
        if not remake_path or not os.path.isfile(remake_path):
            return str(selected_entry.get("backup_path", "") or "")

        backup_dir = os.path.join(output_folder, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        basename = os.path.basename(remake_path)
        backup_path = os.path.join(backup_dir, basename)
        if os.path.exists(backup_path):
            root, ext = os.path.splitext(basename)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"{root}_{stamp}{ext}")
        print(f"[VRGDG RemakeQueue] move remake -> backup: {remake_path} -> {backup_path}")
        os.replace(remake_path, backup_path)
        selected_entry["backup_path"] = backup_path
        selected_entry["remake_path"] = ""
        selected_entry["pending"] = False
        selected_entry["done"] = True
        return backup_path

    def _select_queue_item(self, session_path, prepared, queue_index, reset_queue, enable_auto_queue):
        total = len(prepared)
        pending = [entry for entry in prepared if entry["pending"]]
        if total <= 0:
            return None, 0, pending
        if int(queue_index) > 0:
            pos = int(queue_index) - 1
            if pos < 0 or pos >= total:
                return None, int(queue_index), pending
            return prepared[pos], int(queue_index), pending

        state_path = self._queue_state_path(session_path)
        memory_key = self._queue_memory_key(session_path)
        if reset_queue:
            self._autoqueue_memory.pop(memory_key, None)
            try:
                if os.path.isfile(state_path):
                    os.remove(state_path)
            except Exception:
                pass

        if not pending:
            self._autoqueue_memory.pop(memory_key, None)
            try:
                if os.path.isfile(state_path):
                    os.remove(state_path)
            except Exception:
                pass
            return None, total + 1, pending

        selected_signature = [int(entry["item"].get("clip_number", 0) or 0) for entry in prepared]
        pending_signature = [int(entry["item"].get("clip_number", 0) or 0) for entry in pending]
        state = self._autoqueue_memory.get(memory_key, {})
        already_queued = state.get("selected_signature") == selected_signature
        print(
            "[VRGDG RemakeQueue] autoqueue "
            f"enabled={bool(enable_auto_queue)} "
            f"selected={selected_signature} "
            f"pending={pending_signature} "
            f"state_selected={state.get('selected_signature')} "
            f"already_queued={already_queued} "
            f"will_queue={bool(enable_auto_queue and len(pending) > 1 and not already_queued)} "
            f"queue_add={max(0, len(pending) - 1)}"
        )
        if enable_auto_queue and len(pending) > 1 and not already_queued:
            for _ in range(len(pending) - 1):
                PromptServer.instance.send_sync("impact-add-queue", {})
            self._autoqueue_memory[memory_key] = {
                "selected_signature": selected_signature,
                "pending_signature": pending_signature,
                "queued_count": len(pending) - 1,
                "updated": time.time(),
            }

        active = pending[0]
        active_clip_number = int(active["item"].get("clip_number", 0) or 0)
        active_queue_index = next(
            (index for index, entry in enumerate(prepared, start=1) if int(entry["item"].get("clip_number", 0) or 0) == active_clip_number),
            1,
        )
        print(
            "[VRGDG RemakeQueue] selected "
            f"clip_number={active_clip_number} "
            f"queue_position={active_queue_index} "
            f"index_out={max(0, active_clip_number - 1)} "
            f"remake_path={active.get('remake_path', '')}"
        )
        return active, active_queue_index, pending

    def _slice_audio(self, audio, start_sec, end_sec, fps, tail_loss_frames, pre_frames, clip_number):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        total_duration = total_samples / sample_rate

        start_frame = int(round(float(start_sec) * fps))
        end_frame = int(round(float(end_sec) * fps))
        start_sec = start_frame / fps
        end_sec = end_frame / fps
        frames_per_scene = max(1, end_frame - start_frame)

        pre = int(pre_frames)
        if int(clip_number) <= 1:
            pre = 0
        tail = int(tail_loss_frames)
        base_frames_for_ltx = frames_per_scene + pre + tail
        frames_for_ltx = _round_up_8n1(base_frames_for_ltx)

        samples_per_frame = sample_rate / fps
        pre_samples = int(round(pre * samples_per_frame))
        start_sample = max(0, int(round(start_frame * samples_per_frame)) - pre_samples)
        end_sample = min(total_samples, start_sample + int(round(base_frames_for_ltx * samples_per_frame)))
        segment = waveform[..., start_sample:end_sample].contiguous().clone()

        target_sr = 44100
        output_sr = sample_rate
        if output_sr != target_sr:
            batch, channels, samples = segment.shape
            segment = segment.reshape(batch * channels, samples)
            segment = AF.resample(segment, output_sr, target_sr)
            segment = segment.reshape(batch, channels, -1)
            output_sr = target_sr

        desired_samples = int(round(frames_for_ltx * output_sr / fps))
        current_samples = segment.shape[-1]
        if current_samples < desired_samples:
            segment = torch.nn.functional.pad(segment, (0, desired_samples - current_samples))
        elif current_samples > desired_samples:
            segment = segment[..., :desired_samples]

        return {
            "audio": {"waveform": segment, "sample_rate": output_sr},
            "total_duration": total_duration,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "frames_per_scene": frames_per_scene,
            "frames_for_ltx": frames_for_ltx,
            "pre_frames": pre,
        }

    def run(
        self,
        audio,
        trigger,
        session_path,
        srt_file,
        queue_index,
        fps,
        enable_auto_queue,
        reset_queue,
        tail_loss_frames,
        pre_frames,
    ):
        path, session, clips_obj = _load_editor_session_file(session_path)
        selected = _selected_editor_clips(clips_obj)
        total_selected = len(selected)
        output_folder = self._session_output_folder(path, session)
        prepared = self._prepare_remake_files(selected, output_folder) if selected else []
        selected_entry, active_queue_index, pending = self._select_queue_item(path, prepared, queue_index, reset_queue, enable_auto_queue)

        if selected_entry is None:
            if total_selected == 0:
                instructions = "No selected remake clips were found. Open the VRGDG Video Editor UI, select clips for remake, then save the editor session."
            else:
                instructions = "No clips are currently in the remake folder. Open the VRGDG Video Editor UI and click Save Editor Session to move selected clips into remake."
            return (
                {},
                0.0,
                0,
                0,
                "",
                "",
                "",
                "",
                "",
                "",
                int(active_queue_index),
                int(total_selected),
                0,
                {"durations_frames": []},
                instructions,
                False,
                self._empty_audio(audio),
                trigger,
                0,
                str(output_folder or ""),
                "overwrite",
                0,
            )

        item = selected_entry["item"]
        clip_number = int(item.get("clip_number", 0) or 0)
        backup_path = self._move_selected_remake_to_backup(selected_entry, output_folder)
        srt_segments = _parse_srt_segments(srt_file)
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        total_duration = waveform.shape[-1] / sample_rate
        if srt_segments and srt_segments[-1][1] < total_duration:
            srt_segments[-1] = (srt_segments[-1][0], total_duration)
        if clip_number < 1 or clip_number > len(srt_segments):
            raise ValueError(f"Clip number {clip_number} is out of range for SRT entries ({len(srt_segments)}).")

        start_sec, end_sec = srt_segments[clip_number - 1]
        sliced = self._slice_audio(audio, start_sec, end_sec, int(fps), int(tail_loss_frames), int(pre_frames), clip_number)

        t2i_prompt = str(item.get("t2i_prompt", "") or "")
        i2v_prompt = str(item.get("i2v_prompt", "") or "")
        captured_frame_path = str(item.get("captured_frame_path", "") or "")
        clip_path = str(backup_path or item.get("path", "") or "")
        clip_name = str(item.get("name", "") or "")
        instructions = (
            f"VRGDG remake queue\n"
            f"Item {active_queue_index} / {total_selected}\n"
            f"Remaining remakes after this one: {max(0, len(pending) - 1)}\n"
            f"Clip {clip_number}: {clip_name}\n"
            f"Moved original to backup: {backup_path}\n"
            f"Timing: {_format_seconds(sliced['start_sec'])} -> {_format_seconds(sliced['end_sec'])}"
        )
        meta = {
            "output_folder": output_folder,
            "session_path": path,
            "clip_number": clip_number,
            "clip_name": clip_name,
            "clip_path": clip_path,
            "index": max(0, int(clip_number) - 1),
            "queue_position": int(active_queue_index),
            "total_selected": int(total_selected),
            "offset_seconds": sliced["start_sec"],
            "start_seconds": sliced["start_sec"],
            "end_seconds": sliced["end_sec"],
            "frames_for_ltx": sliced["frames_for_ltx"],
            "frames_per_scene": sliced["frames_per_scene"],
            "pre_frames": sliced["pre_frames"],
            "remaining_remakes": max(0, len(pending) - 1),
            "remake_path": selected_entry.get("remake_path", ""),
            "backup_path": backup_path,
            "replacement_path": selected_entry.get("main_path", ""),
        }
        audio_meta = {"durations_frames": [sliced["frames_per_scene"]]}

        return (
            meta,
            sliced["total_duration"],
            clip_number,
            sliced["frames_for_ltx"],
            _format_seconds(sliced["start_sec"]),
            _format_seconds(sliced["end_sec"]),
            t2i_prompt,
            i2v_prompt,
            captured_frame_path,
            clip_path,
            max(0, int(clip_number) - 1),
            int(total_selected),
            sliced["frames_per_scene"],
            audio_meta,
            instructions,
            True,
            sliced["audio"],
            trigger,
            int(sliced["pre_frames"]),
            str(output_folder or ""),
            "overwrite",
            len(srt_segments),
        )


_ensure_video_editor_routes()


NODE_CLASS_MAPPINGS = {
    "VRGDG_VideoEditorUI": VRGDG_VideoEditorUI,
    "VRGDG_VideoEditorSessionLoader": VRGDG_VideoEditorSessionLoader,
    "VRGDG_RemakeClipQueue": VRGDG_RemakeClipQueue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_VideoEditorUI": "VRGDG Video Editor UI",
    "VRGDG_VideoEditorSessionLoader": "VRGDG Video Editor Session Loader",
    "VRGDG_RemakeClipQueue": "VRGDG Remake Clip Queue",
}
