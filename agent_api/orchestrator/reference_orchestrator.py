"""Reference Builder steps for agents: describe a reference image, extract locations, assign them to scenes.

These are the Video Builder's "Gemma Describe", "LM Extract" and "Assign Scenes" buttons. The LLM calls
use the project's saved runner settings and, for LM Studio, the model that is already loaded (see
``agent_api.llm_runtime``): the API never loads or switches a model.
"""

import os
import random
import time
from typing import Any, Dict, List, Optional

from ...llm import image_prompt_generation as img_gen
from ..errors import ValidationError
from ..jobs.manager import JobManager, get_job_manager
from ..jobs.models import Job
from ..llm_runtime import llm_payload_from_session, prepare_llm_payload
from ..mutations import _BUILDER_SAVE_LOCK, _get_active_session_and_folder, _persist_session

REFERENCE_KINDS = {"subject": "subjects", "subjects": "subjects", "location": "locations", "locations": "locations"}
PATTERNS = ("random", "rotate", "blocks", "unchanged")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _refs(session: Dict[str, Any]) -> Dict[str, Any]:
    return session.setdefault("flux_reference_builder", {})


def _extra_target(subject: Dict[str, Any]) -> str:
    return _text(subject.get("extra_reference_for") or subject.get("same_subject_as"))


def logical_subjects(refs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Characters that can be mapped to scenes (extra views of a character are not separate people)."""
    return [s for s in refs.get("subjects") or [] if isinstance(s, dict) and not _extra_target(s)]


def _new_location_id(taken: Optional[set] = None) -> str:
    """``loc_<ms>_<n>`` like the UI, never repeating an id already in use (several are made in one millisecond)."""
    taken = taken if taken is not None else set()
    while True:
        candidate = f"loc_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate


# ---------------------------------------------------------------------------
# Describe a reference image ("Gemma Describe")
# ---------------------------------------------------------------------------

def describe_reference(project_id: str, kind: str, ref_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Describe a subject or location's image with the project's LLM and save the description on it."""
    params = dict(params or {})
    list_key = REFERENCE_KINDS.get(str(kind or "").strip().lower())
    if not list_key:
        raise ValidationError("kind must be 'subjects' or 'locations'.")
    folder, session = _get_active_session_and_folder(project_id)
    item = next((r for r in _refs(session).get(list_key) or [] if isinstance(r, dict) and r.get("id") == ref_id), None)
    if item is None:
        raise ValidationError(f"{list_key[:-1].capitalize()} '{ref_id}' was not found in the Reference Builder.")
    image = item.get("image") if isinstance(item.get("image"), dict) else {}
    image_path = _text(image.get("path"))
    if not image_path or not os.path.isfile(image_path):
        raise ValidationError(f"{item.get('name') or ref_id} has no image file to describe. Upload an image first.")

    request = {
        **llm_payload_from_session(session),
        "reference_type": "location" if list_key == "locations" else (_text(item.get("reference_type")) or "character"),
        "name": _text(item.get("name")),
        "image_path": image_path,
        # Local-model options. They do nothing for LM Studio, and nothing here may unload a model.
        "unload_after": False,
        "clear_before_load": False,
        **{k: v for k, v in params.items() if k in ("temperature", "top_p", "max_new_tokens", "seed")},
    }
    result = img_gen._generate_builder_reference_description(prepare_llm_payload(request))
    description = _text(result.get("description"))
    if not description:
        raise ValidationError("The LLM returned an empty reference description.")

    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        for stored in _refs(session).get(list_key) or []:
            if stored.get("id") == ref_id:
                stored["description"] = description
        saved = _persist_session(folder, session)
    return {
        "id": ref_id,
        "kind": list_key[:-1],
        "name": _text(item.get("name")),
        "description": description,
        "model": result.get("used_model") or request.get("lmstudio_model"),
        "revision": saved.get("revision"),
    }


# ---------------------------------------------------------------------------
# Locations ("LM Extract")
# ---------------------------------------------------------------------------

def _global_theme_text(folder: str) -> str:
    path = os.path.join(folder, "project_context", "themestyle.txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return ""


def location_style_theme_text(folder: str, session: Dict[str, Any], local_notes: str = "") -> str:
    """Mirror ``locationExtractionStyleTheme``: the project's theme/style file plus the location notes."""
    parts: List[str] = []
    global_theme = _global_theme_text(folder) if session.get("use_vrgdg_text_context", True) else ""
    if global_theme:
        parts.append(f"Global theme/style:\n{global_theme}")
    if _text(local_notes):
        parts.append(f"Location extraction notes:\n{_text(local_notes)}")
    return "\n\n".join(parts).strip()


def subject_context_for_locations(refs: Dict[str, Any]) -> str:
    """Mirror ``referenceSubjectContextForLocations``."""
    lines = []
    for subject in refs.get("subjects") or []:
        name, description = _text(subject.get("name")), _text(subject.get("description"))
        if not name and not description:
            continue
        kind = _text(subject.get("reference_type")) or "character"
        lines.append(f"{name or 'Reference'} ({kind})" + (f": {description}" if description else ""))
    return "\n".join(lines)


def extract_locations(project_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ask the project's LLM for filming locations (LM Extract) and add them to the Reference Builder."""
    params = dict(params or {})
    folder, session = _get_active_session_and_folder(project_id)
    scout = getattr(img_gen, "_generate_lm_scout_locations", None)
    if scout is None:
        raise ValidationError("This install does not have the LM Extract location scout.")
    segments = [s for s in session.get("segments") or [] if isinstance(s, dict)]
    lyric_lines, planning = [], []
    for index, segment in enumerate(segments, start=1):
        lyric = _text(segment.get("lyric_text"))
        if lyric:
            lyric_lines.append((len(lyric_lines) + 1, lyric))
        concept = _text(segment.get("t2i_prompt") or segment.get("flux_prompt") or segment.get("notes") or segment.get("flux_notes"))
        notes = "\n".join(p for p in (_text(segment.get("notes")), _text(segment.get("timeline_note")), _text(segment.get("i2v_notes"))) if p)
        if concept or notes:
            planning.append({"id": segment.get("id"), "label": segment.get("label") or f"Scene {index}", "concept": concept, "notes": notes})
    if not lyric_lines and not planning:
        raise ValidationError("LM Extract needs lyrics, scene notes, or concept prompts first. Create the scenes from the lyrics first.")

    refs = _refs(session)
    local_notes = _text(params.get("style_theme") or refs.get("location_style_theme"))
    request = {
        **llm_payload_from_session(session),
        "lyrics_text": "\n".join(f"Scene {number}: {lyric}" for number, lyric in lyric_lines),
        "scenes": planning,
        "style_theme": location_style_theme_text(folder, session, local_notes),
        "subject_context": subject_context_for_locations(refs),
        "existing_locations": [{"name": loc.get("name", ""), "description": loc.get("description", "")} for loc in refs.get("locations") or []],
        "max_new_tokens": int(params.get("max_new_tokens") or 4000),
        "unload_after": False,
    }
    result = scout(prepare_llm_payload(request))

    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        refs = _refs(session)
        locations = refs.setdefault("locations", [])
        refs["locations_cleared"] = False
        by_name = {_text(loc.get("name")).lower(): loc for loc in locations}
        used_ids = {loc.get("id") for loc in locations}
        added = updated = 0
        for item in result.get("locations") or []:
            name = _text(item.get("name"))
            if not name:
                continue
            existing = by_name.get(name.lower())
            if existing is None:
                location = {"id": _new_location_id(used_ids), "name": name, "description": _text(item.get("description")), "image": {"path": "", "data": "", "name": ""}}
                locations.append(location)
                by_name[name.lower()] = location
                added += 1
            elif not _text(existing.get("description")) and _text(item.get("description")):
                existing["description"] = _text(item.get("description"))
                updated += 1
        refs["location_style_theme"] = local_notes
        refs["use_location_references"] = bool(locations)
        saved = _persist_session(folder, session)
    return {
        "added": added,
        "updated": updated,
        "total_locations": len(locations),
        "locations": [{"id": loc["id"], "name": loc["name"]} for loc in locations],
        "model": result.get("used_model"),
        "revision": saved.get("revision"),
    }


# ---------------------------------------------------------------------------
# Assign characters and locations to scenes ("Assign Scenes")
# ---------------------------------------------------------------------------

def plan_scene_assignment(
    session: Dict[str, Any],
    *,
    scope: str = "all",
    range_start: int = 1,
    range_end: Optional[int] = None,
    scene_ids: Optional[List[str]] = None,
    character_pattern: str = "unchanged",
    character_block_size: int = 10,
    location_pattern: str = "unchanged",
    location_block_size: int = 4,
    replace_existing: bool = False,
    avoid_location_repeat: bool = True,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Mirror ``createProposal`` in reference_scene_assignment.mjs. Returns one entry per target scene."""
    if character_pattern not in PATTERNS or location_pattern not in PATTERNS:
        raise ValidationError(f"Patterns must be one of: {', '.join(PATTERNS)}.")
    rng = random.Random(seed)
    refs = _refs(session)
    scenes = [s for s in session.get("segments") or [] if isinstance(s, dict)]
    subjects = logical_subjects(refs)
    locations = [loc for loc in refs.get("locations") or [] if isinstance(loc, dict)]

    if scope == "range":
        start = max(1, min(len(scenes), int(range_start or 1)))
        end = max(start, min(len(scenes), int(range_end or len(scenes))))
        targets = [(s, i) for i, s in enumerate(scenes) if start <= i + 1 <= end]
    elif scope in ("selected", "scene_ids"):
        wanted = {str(x) for x in (scene_ids or [])}
        if not wanted:
            raise ValidationError("scene_ids is required when scope is 'selected'.")
        targets = [(s, i) for i, s in enumerate(scenes) if str(s.get("id")) in wanted or str(i + 1) in wanted]
    else:
        targets = [(s, i) for i, s in enumerate(scenes)]

    subject_map = refs.get("subject_scene_map") if isinstance(refs.get("subject_scene_map"), dict) else {}
    location_map = refs.get("scene_map") if isinstance(refs.get("scene_map"), dict) else {}

    def pick_random(items: List[Dict[str, Any]], previous_id: str = "", avoid: bool = False) -> Optional[Dict[str, Any]]:
        pool = [i for i in items if i.get("id") != previous_id] if avoid and len(items) > 1 else items
        return rng.choice(pool) if pool else None

    plan: List[Dict[str, Any]] = []
    previous_location = ""
    char_block = max(1, int(character_block_size or 1))
    loc_block = max(1, int(location_block_size or 1))
    for target_index, (scene, index) in enumerate(targets):
        existing_subjects = subject_map.get(scene.get("id"))
        existing_subjects = [str(x) for x in (existing_subjects if isinstance(existing_subjects, list) else str(existing_subjects or "").split(",")) if str(x).strip()]
        existing_location = _text(location_map.get(scene.get("id")))
        subject_id = location_id = ""
        if not scene.get("no_character_present") and character_pattern != "unchanged" and subjects:
            if character_pattern == "random":
                subject_id = (pick_random(subjects) or {}).get("id", "")
            elif character_pattern == "rotate":
                subject_id = subjects[target_index % len(subjects)].get("id", "")
            else:
                subject_id = subjects[(target_index // char_block) % len(subjects)].get("id", "")
        if location_pattern != "unchanged" and locations:
            if location_pattern == "rotate":
                chosen = locations[target_index % len(locations)]
            elif location_pattern == "blocks":
                chosen = locations[(target_index // loc_block) % len(locations)]
            else:
                chosen = pick_random(locations, previous_location, avoid_location_repeat)
            location_id = (chosen or {}).get("id", "")
            previous_location = location_id or previous_location
        if scene.get("no_character_present"):
            final_subjects: List[str] = []
        elif character_pattern == "unchanged" or (not replace_existing and existing_subjects):
            final_subjects = existing_subjects
        else:
            final_subjects = [subject_id] if subject_id else []
        final_location = existing_location if (location_pattern == "unchanged" or (not replace_existing and existing_location)) else location_id
        plan.append({"scene_id": scene.get("id"), "scene_number": index + 1, "subject_ids": final_subjects, "location_id": final_location})
    return plan


def assign_scenes(project_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assign saved characters and locations to scenes with a pattern, and save the mapping."""
    params = dict(params or {})
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        refs = _refs(session)
        character_pattern = str(params.get("character_pattern") or "unchanged")
        location_pattern = str(params.get("location_pattern") or "unchanged")
        plan = plan_scene_assignment(
            session,
            scope=str(params.get("scope") or "all"),
            range_start=int(params.get("range_start") or 1),
            range_end=params.get("range_end"),
            scene_ids=params.get("scene_ids"),
            character_pattern=character_pattern,
            character_block_size=int(params.get("character_block_size") or 10),
            location_pattern=location_pattern,
            location_block_size=int(params.get("location_block_size") or 4),
            replace_existing=bool(params.get("replace_existing", False)),
            avoid_location_repeat=bool(params.get("avoid_location_repeat", True)),
            seed=params.get("seed"),
        )
        if not params.get("dry_run"):
            subject_map = refs.setdefault("subject_scene_map", {})
            location_map = refs.setdefault("scene_map", {})
            for item in plan:
                if character_pattern != "unchanged":
                    if item["subject_ids"]:
                        subject_map[item["scene_id"]] = list(item["subject_ids"])
                    else:
                        subject_map.pop(item["scene_id"], None)
                if location_pattern != "unchanged":
                    if item["location_id"]:
                        location_map[item["scene_id"]] = item["location_id"]
                    else:
                        location_map.pop(item["scene_id"], None)
            refs["use_subject_reference"] = bool(logical_subjects(refs))
            refs["use_location_references"] = bool(refs.get("locations"))
            saved = _persist_session(folder, session)
            revision = saved.get("revision")
        else:
            revision = None
    names = {loc.get("id"): loc.get("name") for loc in refs.get("locations") or []}
    return {
        "dry_run": bool(params.get("dry_run")),
        "scenes_assigned": len(plan),
        "location_use": {name: sum(1 for item in plan if item["location_id"] == lid) for lid, name in names.items() if any(item["location_id"] == lid for item in plan)},
        "plan": plan,
        "revision": revision,
    }


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def run_reference_describe_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    params = dict(job.params or {})
    manager.update_progress(job.id, 10.0, "describing", message="Describing the reference image...")
    return describe_reference(job.project_id, params.pop("kind", "subjects"), params.pop("ref_id", ""), params)


def run_extract_locations_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    manager.update_progress(job.id, 10.0, "extracting_locations", message="Asking the LLM location scout...")
    return extract_locations(job.project_id, job.params)


def register_reference_orchestrator_handlers(manager: Optional[JobManager] = None) -> None:
    manager = manager or get_job_manager()
    manager.register_handler("reference.describe", run_reference_describe_job)
    manager.register_handler("reference.extract_locations", run_extract_locations_job)
