import json
import os
import shutil
import time
import base64
import re

from aiohttp import web
from server import PromptServer

from .VRGDG_BrowserImageRoutes import _newest_manual_download, _normalize_provider


STORYBOARD_FOLDER = "start_image_storyboard"


def _project_folder(value):
    folder = os.path.abspath(str(value or "").strip().strip('"'))
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose an existing Video Builder project folder.")
    if not os.path.isfile(os.path.join(folder, "vrgdg_builder_session.json")):
        raise ValueError("That folder is not a Video Builder project. Choose a folder containing vrgdg_builder_session.json.")
    return folder


def _board_folder(project_folder):
    return os.path.join(project_folder, STORYBOARD_FOLDER)


def _board_path(project_folder):
    return os.path.join(_board_folder(project_folder), "storyboard.json")


def _images_folder(project_folder):
    return os.path.join(_board_folder(project_folder), "images")


def _project_image_path(project_folder, path):
    value = str(path or "").strip().strip('"')
    if not value:
        return ""
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(project_folder, value))


def _image_url(project_folder, path):
    path = _project_image_path(project_folder, path)
    if not path:
        return ""
    stamp = int(os.path.getmtime(path)) if os.path.isfile(path) else int(time.time())
    from urllib.parse import quote
    return f"/vrgdg/start_storyboard/image?project_folder={quote(project_folder)}&path={quote(path)}&v={stamp}"


def _read_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _builder_session_path(project_folder):
    return os.path.join(project_folder, "vrgdg_builder_session.json")


def _reference_builder_from_session(data):
    if not isinstance(data, dict):
        return {}
    for key in ("flux_reference_builder", "fluxReferenceBuilder", "reference_builder", "referenceBuilder"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    for key in ("session", "state", "project"):
        found = _reference_builder_from_session(data.get(key))
        if found:
            return found
    return {}


def _reference_image(item):
    source = item if isinstance(item, dict) else {}
    image = source.get("image") if isinstance(source.get("image"), dict) else source
    return {
        "path": str(image.get("path") or source.get("image_path") or source.get("imagePath") or source.get("path") or "").strip(),
        "data": str(image.get("data") or source.get("image_data") or source.get("imageData") or source.get("data") or "").strip(),
        "name": str(image.get("name") or source.get("image_name") or source.get("imageName") or "").strip(),
    }


def _scene_map_location_id(scene_map, candidates):
    if not isinstance(scene_map, dict):
        return ""
    for candidate in candidates:
        key = str(candidate or "").strip()
        if not key or key not in scene_map:
            continue
        value = scene_map.get(key)
        if isinstance(value, dict):
            value = value.get("location_id") or value.get("locationId") or value.get("location") or value.get("id")
        value = str(value or "").strip()
        if value:
            return value
    return ""


def _apply_project_location_mappings(project_folder, board):
    session_path = _builder_session_path(project_folder)
    session = _read_json(session_path) if os.path.isfile(session_path) else {}
    refs = _reference_builder_from_session(session)
    locations = refs.get("locations") if isinstance(refs.get("locations"), list) else []
    scene_map = refs.get("scene_map") or refs.get("sceneMap") or {}
    locations_cleared = bool(refs.get("locations_cleared") or refs.get("locationsCleared"))
    locations_by_id = {str(item.get("id") or "").strip(): item for item in locations if isinstance(item, dict) and str(item.get("id") or "").strip()}
    locations_by_name = {str(item.get("name") or item.get("label") or "").strip().lower(): item for item in locations if isinstance(item, dict) and str(item.get("name") or item.get("label") or "").strip()}
    session_scenes = _segment_list(session)
    imported_count = 0

    for index, scene in enumerate(board.get("scenes") or []):
        if not isinstance(scene, dict):
            continue
        prior_source = str((scene.get("location_ref") or {}).get("source") or "") if isinstance(scene.get("location_ref"), dict) else ""
        if locations_cleared:
            if prior_source == "video_builder" or prior_source == "":
                scene.pop("location_ref", None)
            continue
        session_scene = session_scenes[index] if index < len(session_scenes) and isinstance(session_scenes[index], dict) else {}
        project_scene_id = str(session_scene.get("id") or scene.get("project_scene_id") or scene.get("id") or "").strip()
        if project_scene_id:
            scene["project_scene_id"] = project_scene_id
        number = index + 1
        candidates = [
            scene.get("id"),
            scene.get("project_scene_id"),
            session_scene.get("id"),
            number,
            f"scene{number}",
            f"scene_{number}",
            f"scene_{number:04d}",
        ]
        location_id = _scene_map_location_id(scene_map, candidates)
        if not location_id:
            direct = session_scene.get("location_ref") or session_scene.get("locationRef")
            if isinstance(direct, dict):
                location_id = str(direct.get("id") or direct.get("name") or "").strip()
            else:
                location_id = str(session_scene.get("mapped_location") or session_scene.get("location_id") or "").strip()
        location = locations_by_id.get(location_id) or locations_by_name.get(location_id.lower())
        if not location:
            if prior_source == "video_builder":
                scene.pop("location_ref", None)
            continue
        image = _reference_image(location)
        image["path"] = _project_image_path(project_folder, image.get("path"))
        scene["location_ref"] = {
            "id": str(location.get("id") or location_id).strip(),
            "name": str(location.get("name") or location.get("label") or "Mapped location").strip(),
            "description": str(location.get("description") or location.get("prompt") or "").strip(),
            "image": image,
            "source": "video_builder",
        }
        imported_count += 1

    board["project_location_catalog"] = [
        {
            "id": str(item.get("id") or "").strip(),
            "name": str(item.get("name") or item.get("label") or "Mapped location").strip(),
            "description": str(item.get("description") or item.get("prompt") or "").strip(),
            "image": _reference_image(item),
        }
        for item in locations if isinstance(item, dict)
    ]
    board["imported_location_count"] = imported_count
    return board


def _find_lyric_source(project_folder):
    candidates = [
        os.path.join(project_folder, "prompts", "lyric_segments.json"),
        os.path.join(project_folder, "lyric_segments.json"),
        os.path.join(project_folder, "project_context", "lyric_segments.json"),
        os.path.join(project_folder, "vrgdg_builder_session.json"),
        os.path.join(project_folder, "session.json"),
        os.path.join(project_folder, "music_video_builder_session.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("No lyric_segments.json or Video Builder session was found in this project.")


def _segment_list(data):
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("segments", "scenes", "lyric_segments", "timelineSegments", "timeline_segments"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    for key in ("session", "state", "project"):
        found = _segment_list(data.get(key))
        if found:
            return found
    return []


def _normalize_scenes(items):
    scenes = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            lyric = item.strip()
            item = {}
        elif isinstance(item, dict):
            lyric = str(item.get("lyric_text") or item.get("lyrics") or item.get("text") or item.get("line") or "").strip()
        else:
            continue
        if not lyric and isinstance(item, dict) and str(item.get("type") or "").lower() in {"overlay", "marker"}:
            continue
        number = len(scenes) + 1
        scenes.append({
            "id": str(item.get("id") or f"scene_{number:04d}"),
            "number": number,
            "lyric": lyric,
            "note": "",
            "preset": "",
            "end_transition_preset": "",
            "end_frame_note": "",
            "prompt": "",
            "image_path": "",
            "end_image_path": "",
            "location_area": "",
        })
    return scenes


def _load_board(project_folder, import_if_missing=True):
    path = _board_path(project_folder)
    created = False
    if os.path.isfile(path):
        board = _read_json(path)
    elif import_if_missing:
        source = _find_lyric_source(project_folder)
        board = {
            "version": 1,
            "project_folder": project_folder,
            "global_idea": "",
            "scenes": _normalize_scenes(_segment_list(_read_json(source))),
            "lyric_source": source,
        }
        if not board["scenes"]:
            raise ValueError(f"No lyric scenes were found in {source}.")
        created = True
    else:
        board = {"version": 1, "project_folder": project_folder, "global_idea": "", "scenes": []}
    _apply_project_location_mappings(project_folder, board)
    if created:
        _save_board(project_folder, board)
    for scene in board.get("scenes", []):
        scene["image_url"] = _image_url(project_folder, scene.get("image_path", ""))
        scene["end_image_url"] = _image_url(project_folder, scene.get("end_image_path", ""))
        scene["reference_url"] = _image_url(project_folder, scene.get("reference_path", ""))
        location = scene.get("location_ref") if isinstance(scene.get("location_ref"), dict) else {}
        location_image = location.get("image") if isinstance(location.get("image"), dict) else {}
        location_data = str(location_image.get("data") or "").strip()
        scene["location_image_url"] = location_data if location_data.startswith("data:image/") else _image_url(project_folder, location_image.get("path", ""))
    board["global_reference_url"] = _image_url(project_folder, board.get("global_reference_path", ""))
    return board


def _save_board(project_folder, board):
    os.makedirs(_images_folder(project_folder), exist_ok=True)
    clean = dict(board or {})
    clean["version"] = 2
    clean["project_folder"] = project_folder
    clean["updated_at"] = int(time.time())
    scenes = []
    for index, source in enumerate(clean.get("scenes") or [], start=1):
        scene = dict(source or {})
        scene.pop("image_url", None)
        scene.pop("end_image_url", None)
        scene.pop("reference_url", None)
        scene.pop("location_image_url", None)
        scene["number"] = index
        scene["id"] = str(scene.get("id") or f"scene_{index:04d}")
        scenes.append(scene)
    clean["scenes"] = scenes
    os.makedirs(_board_folder(project_folder), exist_ok=True)
    temp_path = _board_path(project_folder) + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(clean, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, _board_path(project_folder))
    return clean


def _frame_field(frame):
    return "end_image_path" if str(frame or "start").strip().lower() == "end" else "image_path"


def _frame_stem(scene_number, frame):
    suffix = "_end" if _frame_field(frame) == "end_image_path" else ""
    return f"scene_{int(scene_number):04d}{suffix}"


def _import_latest(project_folder, provider, scene_number, frame="start"):
    provider = _normalize_provider(provider)
    source = _newest_manual_download(provider)
    scene_number = int(scene_number)
    ext = os.path.splitext(source)[1].lower() or ".png"
    stem = _frame_stem(scene_number, frame)
    target = os.path.join(_images_folder(project_folder), f"{stem}{ext}")
    attempts = os.path.join(_images_folder(project_folder), "attempts", stem)
    os.makedirs(attempts, exist_ok=True)
    if os.path.isfile(target):
        old_ext = os.path.splitext(target)[1]
        archive = os.path.join(attempts, f"attempt_{int(time.time() * 1000)}{old_ext}")
        shutil.copy2(target, archive)
    for name in os.listdir(_images_folder(project_folder)):
        if name.startswith(f"{stem}."):
            old = os.path.join(_images_folder(project_folder), name)
            if os.path.isfile(old) and old != target:
                os.remove(old)
    shutil.copy2(source, target)
    board = _load_board(project_folder)
    if scene_number < 1 or scene_number > len(board.get("scenes", [])):
        raise ValueError("Scene number is outside this storyboard.")
    board["scenes"][scene_number - 1][_frame_field(frame)] = target
    _save_board(project_folder, board)
    return {"saved_path": target, "image_url": _image_url(project_folder, target), "provider": provider}


def _save_reference(project_folder, data_url, name, scene_number=None):
    match = re.match(r"^data:image/([A-Za-z0-9.+-]+);base64,(.+)$", str(data_url or ""), flags=re.S)
    if not match:
        raise ValueError("Reference upload did not contain valid image data.")
    subtype = match.group(1).lower()
    ext = ".jpg" if subtype in {"jpeg", "jpg"} else ".webp" if subtype == "webp" else ".png"
    refs = os.path.join(_board_folder(project_folder), "references")
    os.makedirs(refs, exist_ok=True)
    target_name = "global_character_reference" if not scene_number else f"scene_{int(scene_number):04d}_reference"
    target = os.path.join(refs, target_name + ext)
    for existing in os.listdir(refs):
        if existing.startswith(target_name + "."):
            old = os.path.join(refs, existing)
            if os.path.isfile(old):
                os.remove(old)
    with open(target, "wb") as handle:
        handle.write(base64.b64decode(match.group(2)))
    board = _load_board(project_folder)
    if scene_number:
        number = int(scene_number)
        if number < 1 or number > len(board.get("scenes", [])):
            raise ValueError("Scene number is outside this storyboard.")
        board["scenes"][number - 1]["reference_path"] = target
    else:
        board["global_reference_path"] = target
        board["use_global_reference"] = True
    _save_board(project_folder, board)
    return {"saved_path": target, "image_url": _image_url(project_folder, target)}


def _save_uploaded_scene_image(project_folder, data_url, scene_number, frame="start"):
    match = re.match(r"^data:image/([A-Za-z0-9.+-]+);base64,(.+)$", str(data_url or ""), flags=re.S)
    if not match:
        raise ValueError("Dropped file did not contain valid image data.")
    subtype = match.group(1).lower()
    ext = ".jpg" if subtype in {"jpeg", "jpg"} else ".webp" if subtype == "webp" else ".png"
    scene_number = int(scene_number)
    board = _load_board(project_folder)
    if scene_number < 1 or scene_number > len(board.get("scenes", [])):
        raise ValueError("Scene number is outside this storyboard.")
    images = _images_folder(project_folder)
    stem = _frame_stem(scene_number, frame)
    attempts = os.path.join(images, "attempts", stem)
    os.makedirs(attempts, exist_ok=True)
    prefix = f"{stem}."
    for existing in os.listdir(images):
        if not existing.startswith(prefix):
            continue
        old = os.path.join(images, existing)
        if os.path.isfile(old):
            archive = os.path.join(attempts, f"attempt_{int(time.time() * 1000)}{os.path.splitext(old)[1]}")
            shutil.copy2(old, archive)
            os.remove(old)
    target = os.path.join(images, f"{stem}{ext}")
    with open(target, "wb") as handle:
        handle.write(base64.b64decode(match.group(2)))
    board["scenes"][scene_number - 1][_frame_field(frame)] = target
    _save_board(project_folder, board)
    return {"saved_path": target, "image_url": _image_url(project_folder, target)}


class VRGDG_StartImageStoryboard:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    OUTPUT_NODE = True
    CATEGORY = "VRGDG/UI"
    DESCRIPTION = "One-button standalone start-image storyboard creator with Browser AI."

    def noop(self):
        return ()


NODE_CLASS_MAPPINGS = {"VRGDG_StartImageStoryboard": VRGDG_StartImageStoryboard}
NODE_DISPLAY_NAME_MAPPINGS = {"VRGDG_StartImageStoryboard": "VRGDG Storyboard Creator with Browser AI — Open This"}


@PromptServer.instance.routes.post("/vrgdg/start_storyboard/load")
async def load_start_storyboard(request):
    try:
        payload = await request.json()
        folder = _project_folder(payload.get("project_folder"))
        return web.json_response({"ok": True, "storyboard": _load_board(folder)})
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


@PromptServer.instance.routes.post("/vrgdg/start_storyboard/reimport")
async def reimport_start_storyboard(request):
    try:
        payload = await request.json()
        folder = _project_folder(payload.get("project_folder"))
        source = _find_lyric_source(folder)
        old = _load_board(folder, False)
        imported = _normalize_scenes(_segment_list(_read_json(source)))
        for index, scene in enumerate(imported):
            if index < len(old.get("scenes", [])):
                previous = old["scenes"][index]
                for key in ("note", "preset", "end_transition_preset", "end_frame_note", "prompt", "image_path", "end_image_path", "reference_path", "location_area"):
                    scene[key] = previous.get(key, scene.get(key, ""))
        old["scenes"] = imported
        old["lyric_source"] = source
        _apply_project_location_mappings(folder, old)
        _save_board(folder, old)
        return web.json_response({"ok": True, "storyboard": _load_board(folder)})
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


@PromptServer.instance.routes.post("/vrgdg/start_storyboard/save")
async def save_start_storyboard(request):
    try:
        payload = await request.json()
        folder = _project_folder(payload.get("project_folder"))
        _save_board(folder, payload.get("storyboard") or {})
        return web.json_response({"ok": True, "storyboard": _load_board(folder)})
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


@PromptServer.instance.routes.post("/vrgdg/start_storyboard/import_latest")
async def import_latest_start_storyboard(request):
    try:
        payload = await request.json()
        folder = _project_folder(payload.get("project_folder"))
        result = _import_latest(folder, payload.get("provider"), payload.get("scene_number"), payload.get("frame"))
        return web.json_response({"ok": True, **result})
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


@PromptServer.instance.routes.post("/vrgdg/start_storyboard/save_reference")
async def save_start_storyboard_reference(request):
    try:
        payload = await request.json()
        folder = _project_folder(payload.get("project_folder"))
        result = _save_reference(folder, payload.get("image_data"), payload.get("image_name"), payload.get("scene_number"))
        return web.json_response({"ok": True, **result, "storyboard": _load_board(folder)})
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


@PromptServer.instance.routes.post("/vrgdg/start_storyboard/save_scene_upload")
async def save_start_storyboard_scene_upload(request):
    try:
        payload = await request.json()
        folder = _project_folder(payload.get("project_folder"))
        result = _save_uploaded_scene_image(folder, payload.get("image_data"), payload.get("scene_number"), payload.get("frame"))
        return web.json_response({"ok": True, **result})
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


@PromptServer.instance.routes.get("/vrgdg/start_storyboard/image")
async def start_storyboard_image(request):
    try:
        folder = _project_folder(request.query.get("project_folder"))
        path = os.path.abspath(str(request.query.get("path") or "").strip())
        roots = [
            os.path.abspath(folder),
            os.path.abspath(_images_folder(folder)),
            os.path.abspath(os.path.join(_board_folder(folder), "references")),
        ]
        if not os.path.isfile(path) or not any(os.path.commonpath([root, path]) == root for root in roots):
            raise FileNotFoundError("Storyboard image was not found.")
        return web.FileResponse(path)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
