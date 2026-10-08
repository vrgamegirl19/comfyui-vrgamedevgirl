"""Per-take scene latents.

Every MiniMax render saves the scene's latent to ``latents/scene_NNN.latent``, which the next scene reads for Latent
Continuation Masked. A re-render overwrote it, so switching a scene back to an earlier take left the next scene
continuing from the newest take. This module keeps one archived latent per take in ``latents/scene_NNN.takes/`` and
puts the selected take's latent back as the active one before the next scene renders.

A take is matched to its video by a fingerprint of the video file (size plus its first and last bytes), so it still
matches after the clip is moved to ``rendered_scene_videos_backup`` under a new name. The flow:

1. ``archive_latent`` copies a freshly saved latent into the take archive as *pending*.
2. ``attach_video`` runs when the render's final video is collected and ties the pending latent to that video.
3. ``activate_for_video`` copies the archived latent of a selected video over ``scene_NNN.latent``.
"""

import hashlib
import filecmp
import glob
import json
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional

from ..core.atomic_write import atomic_write_json

TAKES_SUFFIX = ".takes"
INDEX_NAME = "index.json"
_FOLDER_PATTERN = re.compile(r"^scene_(\d{3,})\.takes$", re.IGNORECASE)
_SAMPLE_BYTES = 256 * 1024


def video_fingerprint(path: str) -> str:
    """Identify a video file by its size and first and last bytes. Empty when the file cannot be read."""
    try:
        size = os.path.getsize(path)
        digest = hashlib.sha1(str(size).encode("ascii"))
        with open(path, "rb") as handle:
            digest.update(handle.read(_SAMPLE_BYTES))
            if size > _SAMPLE_BYTES:
                handle.seek(max(_SAMPLE_BYTES, size - _SAMPLE_BYTES))
                digest.update(handle.read(_SAMPLE_BYTES))
        return digest.hexdigest()
    except OSError:
        return ""


def _folder(latents_dir: str, scene_number: int) -> str:
    return os.path.join(latents_dir, f"scene_{int(scene_number):03d}{TAKES_SUFFIX}")


def _read_index(folder: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(folder, INDEX_NAME), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {"takes": []}
    if not isinstance(data, dict) or not isinstance(data.get("takes"), list):
        return {"takes": []}
    return data


def _write_index(folder: str, index: Dict[str, Any]) -> None:
    os.makedirs(folder, exist_ok=True)
    atomic_write_json(os.path.join(folder, INDEX_NAME), index)


def _take_files(folder: str, take_id: str) -> List[str]:
    return [os.path.join(folder, f"{take_id}.latent"), os.path.join(folder, f"{take_id}.latent.json")]


def _remove_take_files(folder: str, take_id: str) -> None:
    for path in _take_files(folder, take_id):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as exc:
            print(f"[VRGDG Latent Takes] Could not remove {path}: {exc}")


def archive_latent(latents_dir: str, scene_number: int, latent_path: str) -> str:
    """Copy a just-saved latent into the take archive as the pending take. Returns its id."""
    folder = _folder(latents_dir, scene_number)
    os.makedirs(folder, exist_ok=True)
    index = _read_index(folder)
    previous = str(index.get("pending") or "")
    if previous:
        _remove_take_files(folder, previous)  # a render whose video was never collected
    take_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"
    shutil.copy2(latent_path, os.path.join(folder, f"{take_id}.latent"))
    if os.path.isfile(latent_path + ".json"):
        shutil.copy2(latent_path + ".json", os.path.join(folder, f"{take_id}.latent.json"))
    index["pending"] = take_id
    _write_index(folder, index)
    return take_id


def _live_fingerprints(project_folder: str, scene_number: int) -> set:
    """Fingerprints of every video file of this scene: the selected clip and its backups."""
    number = int(scene_number)
    patterns = [
        os.path.join(project_folder, "rendered_scene_videos", f"video_{number:04d}-audio*.mp4"),
        os.path.join(project_folder, "rendered_scene_videos_backup", f"scene_{number:04d}", "*.mp4"),
    ]
    found = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            fingerprint = video_fingerprint(path)
            if fingerprint:
                found.add(fingerprint)
    return found


def attach_video(project_folder: str, scene_number: int, video_path: str) -> Optional[str]:
    """Tie the pending latent to the collected final video, and drop takes whose video no longer exists."""
    latents_dir = os.path.join(os.path.abspath(project_folder), "latents")
    folder = _folder(latents_dir, scene_number)
    index = _read_index(folder)
    pending = str(index.get("pending") or "")
    if not pending or not os.path.isfile(_take_files(folder, pending)[0]):
        return None
    fingerprint = video_fingerprint(video_path)
    if not fingerprint:
        return None
    takes = []
    for take in index["takes"]:
        if take.get("video_fp") == fingerprint:
            _remove_take_files(folder, str(take.get("id")))
        else:
            takes.append(take)
    takes.append({
        "id": pending,
        "video_fp": fingerprint,
        "video_name": os.path.basename(video_path),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    index["takes"] = takes
    index["pending"] = ""
    live = _live_fingerprints(project_folder, scene_number)
    if live:
        kept = []
        for take in index["takes"]:
            if take.get("video_fp") in live:
                kept.append(take)
            else:
                _remove_take_files(folder, str(take.get("id")))
        index["takes"] = kept
    _write_index(folder, index)
    return pending


def rename_video(project_folder: str, scene_number: int, old_fingerprint: str, new_fingerprint: str) -> None:
    """A clip was rewritten in place (for example by the opening color match): keep its take."""
    if not old_fingerprint or not new_fingerprint or old_fingerprint == new_fingerprint:
        return
    folder = _folder(os.path.join(os.path.abspath(project_folder), "latents"), scene_number)
    index = _read_index(folder)
    changed = False
    for take in index["takes"]:
        if take.get("video_fp") == old_fingerprint:
            take["video_fp"] = new_fingerprint
            changed = True
    if changed:
        _write_index(folder, index)


def list_takes(project_folder: str, scene_number: int) -> List[Dict[str, Any]]:
    """The archived takes of a scene, oldest first."""
    folder = _folder(os.path.join(os.path.abspath(project_folder), "latents"), scene_number)
    return [
        {"id": take.get("id"), "video_name": take.get("video_name", ""), "saved_at": take.get("saved_at", "")}
        for take in _read_index(folder)["takes"]
    ]


def activate_for_video(project_folder: str, scene_number: int, video_path: str) -> Dict[str, Any]:
    """Make the latent of the take that produced ``video_path`` the scene's active latent.

    ``status`` is ``activated`` or ``already_active`` when the take was found, ``no_archive`` when this scene has
    archived takes but none belongs to the video, and ``no_takes`` when the scene predates per-take latents (the
    active latent is then left alone).
    """
    latents_dir = os.path.join(os.path.abspath(project_folder), "latents")
    folder = _folder(latents_dir, scene_number)
    takes = _read_index(folder)["takes"]
    if not takes:
        return {"status": "no_takes", "take_count": 0}
    fingerprint = video_fingerprint(video_path) if video_path else ""
    take = next((item for item in takes if fingerprint and item.get("video_fp") == fingerprint), None)
    if take is None:
        return {"status": "no_archive", "take_count": len(takes)}
    source = _take_files(folder, str(take["id"]))[0]
    if not os.path.isfile(source):
        return {"status": "no_archive", "take_count": len(takes)}
    active = os.path.join(latents_dir, f"scene_{int(scene_number):03d}.latent")
    result = {"take_id": take["id"], "take_count": len(takes), "video_name": take.get("video_name", "")}
    if os.path.isfile(active) and filecmp.cmp(source, active, shallow=False):
        return {"status": "already_active", **result}
    shutil.copy2(source, active)
    if os.path.isfile(source + ".json"):
        shutil.copy2(source + ".json", active + ".json")
    print(f"[VRGDG Latent Takes] Scene {int(scene_number):03d}: using the latent of take {take['id']} ({take.get('video_name', '')})")
    return {"status": "activated", **result}


def delete_scene_takes(latents_dir: str, scene_number: int) -> None:
    shutil.rmtree(_folder(latents_dir, scene_number), ignore_errors=True)


def delete_all_takes(latents_dir: str) -> None:
    if not os.path.isdir(latents_dir):
        return
    for name in os.listdir(latents_dir):
        if _FOLDER_PATTERN.match(name):
            shutil.rmtree(os.path.join(latents_dir, name), ignore_errors=True)


def shift_takes(latents_dir: str, first_scene: int, delta: int) -> None:
    """Renumber the take folders of every scene from ``first_scene`` on, when scenes are inserted or removed."""
    if not os.path.isdir(latents_dir):
        return
    numbers = sorted(
        {int(m.group(1)) for m in map(_FOLDER_PATTERN.match, os.listdir(latents_dir)) if m and int(m.group(1)) >= int(first_scene)},
        reverse=delta > 0,
    )
    for number in numbers:
        source = _folder(latents_dir, number)
        target = _folder(latents_dir, number + delta)
        try:
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            os.rename(source, target)
        except OSError as exc:
            print(f"[VRGDG Latent Takes] Renumbering {source} -> {target} failed: {exc}")


def copy_takes(source_latents_dir: str, target_latents_dir: str) -> None:
    if not os.path.isdir(source_latents_dir):
        return
    for name in os.listdir(source_latents_dir):
        if _FOLDER_PATTERN.match(name):
            try:
                shutil.copytree(os.path.join(source_latents_dir, name), os.path.join(target_latents_dir, name), dirs_exist_ok=True)
            except OSError as exc:
                print(f"[VRGDG Latent Takes] Copying {name} failed: {exc}")
