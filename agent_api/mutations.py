"""Transactional mutation operations for the VRGDG Agent API (Phase A3, Section 15 & 16)."""

import os
import re
import copy
import shutil
import uuid
from typing import Any, Dict, List, Optional

from ..builder.audio import (
    _convert_audio_to_wav,
    _create_silent_audio,
    _read_audio_peaks,
    _save_project_audio,
    _save_project_srt,
    _srt_path,
)
from ..builder.lyric_scenes import (
    apply_length_fix,
    carry_lyrics_over,
    is_instrumental_lyric_text,
    merge_lyric_text,
    next_length_fix,
    split_lyric_text,
)
from ..builder.project import (
    _BUILDER_SAVE_LOCK,
    _SCENE_ASSET_NAME,
    _delete_builder_project,
    _load_builder_session,
    _load_model_defaults,
    _new_builder_project,
    _prepare_builder_project_export,
    _renumber_scene_assets_after_insert,
    _renumber_scene_assets_after_removal,
    _save_builder_project_as,
    _save_builder_session,
    _save_canonical_full_lyrics,
)
from ..builder.timeline import (
    TimelineJournal,
    close_all_base_timeline_gaps,
    close_base_timeline_gap,
    has_locked_video,
    move_scene_timing,
    normalize_segments,
    parse_bulk_scenes,
    recover_pending_journal,
    renumber_generic_base_scene_labels,
    resize_scene_timing,
    rewrite_renamed_scene_paths,
    shift_segment_timing,
    snap_to_beat,
    sort_segments,
    validate_timeline_consistency,
)
from ..minimax.latent_manager import SceneLatentManager
from ..minimax.scene_inputs import reference_choices, validate_reference_keys
from ..minimax.settings_payload import minimax_h3_settings_for_scene
from ..minimax.prompt_assembly import (
    assemble_minimax_h3_prompt,
    build_minimax_prompt_context,
    effective_mode as effective_minimax_mode,
    validate_minimax_h3_prompt,
)

from ..builder import timeline_markers as markers_service
from ..core.atomic_write import atomic_write_text
from ..storyboard import persistence as storyboard_store
from ..storyboard import scene_card_fields as card_fields
from ..storyboard import session_sync

from .errors import (
    ProjectNotFoundError,
    RevisionConflictError,
    SceneNotFoundError,
    SettingsInvalidError,
    TimelineNoteNotFoundError,
    ValidationError,
)
from .paths import get_project_id, resolve_project_folder
from .project_events import notify_project_changed
from .schemas import extract_effective_settings, validate_settings_patch


def _generate_scene_id() -> str:
    """Generate a unique segment ID conforming to the seg_<hex> standard."""
    return f"seg_{uuid.uuid4().hex[:12]}"


def _get_active_session_and_folder(project_id: str) -> tuple:
    folder = resolve_project_folder(project_id)
    recover_pending_journal(folder)
    load_result = _load_builder_session(folder)
    session = load_result.get("session")
    if not isinstance(session, dict):
        raise ProjectNotFoundError(project_id)
    return folder, session


def _persist_session(folder: str, session: Dict[str, Any], change: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist session changes through _save_builder_session with correct payload and revision.

    Open Video Builder windows are told about the save (``project_events``); ``change`` says what changed
    so they can merge it instead of reloading the whole project.
    """
    # The UI counts saves in builder_save_revision while the server counts them in
    # revision, and the two can drift apart. Stay above both so this save is never
    # mistaken for a stale snapshot and dropped.
    current_rev = max(int(session.get("revision") or 0), int(session.get("builder_save_revision") or 0))
    next_rev = current_rev + 1
    session["revision"] = next_rev
    session["builder_save_revision"] = next_rev
    session["project_folder"] = folder
    payload = {
        "project_folder": folder,
        "session": session,
        "audio_path": session.get("audio_path", ""),
        "project_name": session.get("project_name", ""),
    }
    result = _save_builder_session(payload)
    if isinstance(result, dict) and result.get("stale"):
        # Never report success for a write the server discarded.
        raise RevisionConflictError(int(result.get("current_revision") or 0), int(next_rev))
    saved = result.get("session") if isinstance(result, dict) and isinstance(result.get("session"), dict) else session
    notify_project_changed(folder, saved, change)
    return result


def scene_field_change(scene_id: str, segment_keys: List[str], card_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """A ``scene_fields`` change notice for one scene (see ``project_events``)."""
    return {
        "kind": "scene_fields",
        "scenes": {str(scene_id): {"segment": sorted(segment_keys), "card": sorted(card_keys or []), "references": []}},
        "storyboard": bool(card_keys),
    }


# ==============================================================================
# 1. Scene CRUD and Structural Mutations (Section 15.4, 15.6)
# ==============================================================================

def create_scene(
    project_id: str,
    position: str = "append",
    ref_scene_id: Optional[str] = None,
    duration: float = 4.0,
    label: str = "",
    t2i_prompt: str = "",
    i2v_prompt: str = "",
    notes: str = "",
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a new scene with file slot reservation, renumbering, and transaction journaling (Section 15.4 T1, T2, T3, 15.6)."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        segments = session.setdefault("segments", [])
        sort_segments(segments)

        pos = str(position or "append").lower()
        dur = max(0.1, float(duration or 4.0))

        insert_idx = len(segments)
        start_time = 0.0

        if segments:
            last_end = float(segments[-1].get("end", 0.0) or 0.0)
            start_time = last_end

        if pos in ("before", "after") and ref_scene_id:
            ref_idx = next((i for i, s in enumerate(segments) if s.get("id") == ref_scene_id or str(i + 1) == str(ref_scene_id)), -1)
            if ref_idx < 0:
                raise SceneNotFoundError(ref_scene_id, project_id)

            if pos == "before":
                insert_idx = ref_idx
                start_time = float(segments[ref_idx].get("start", 0.0) or 0.0)
            else:
                insert_idx = ref_idx + 1
                start_time = float(segments[ref_idx].get("end", 0.0) or 0.0)

        slot_number = insert_idx + 1
        renamed = []

        journal = TimelineJournal(folder, "create_scene")
        journal.start()
        try:
            if insert_idx < len(segments):
                renamed = _renumber_scene_assets_after_insert(folder, slot_number)
                journal.record_renames(renamed)
                rewrite_renamed_scene_paths(session, renamed)

            new_scene = {
                "id": _generate_scene_id(),
                "label": label or f"Scene {slot_number}",
                "start": round(start_time, 4),
                "end": round(start_time + dur, 4),
                "t2i_prompt": t2i_prompt,
                "i2v_prompt": i2v_prompt,
                "notes": notes,
                "source": "inserted",
            }

            if insert_idx < len(segments):
                for s in segments[insert_idx:]:
                    shift_segment_timing(s, dur)

            segments.insert(insert_idx, new_scene)
            renumber_generic_base_scene_labels(segments)

            save_result = _persist_session(folder, session)
            journal.commit()

            return {
                "scene": new_scene,
                "renamed": renamed,
                "revision": save_result.get("revision", current_rev + 1),
            }
        except Exception:
            journal.rollback()
            raise


def delete_scene(
    project_id: str,
    scene_id: str,
    ripple: bool = True,
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Delete a scene, renumber later scene files down by 1, and ripple close the timeline gap (T4, 15.6)."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        segments = session.setdefault("segments", [])
        sort_segments(segments)

        target_idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
        if target_idx < 0:
            raise SceneNotFoundError(scene_id, project_id)

        target_scene = segments[target_idx]
        slot_number = target_idx + 1
        start_time = float(target_scene.get("start", 0.0) or 0.0)
        end_time = float(target_scene.get("end", 0.0) or 0.0)

        journal = TimelineJournal(folder, "delete_scene")
        journal.start()
        try:
            renamed = _renumber_scene_assets_after_removal(folder, slot_number)
            journal.record_renames(renamed)
            rewrite_renamed_scene_paths(session, renamed)

            segments.pop(target_idx)

            if ripple:
                close_base_timeline_gap(segments, session.get("overlay_segments"), start_time, end_time)

            renumber_generic_base_scene_labels(segments)
            save_result = _persist_session(folder, session)
            journal.commit()

            return {
                "deleted_scene_id": target_scene.get("id"),
                "renamed": renamed,
                "revision": save_result.get("revision", current_rev + 1),
            }
        except Exception:
            journal.rollback()
            raise


def split_scene(
    project_id: str,
    scene_id: str,
    at_time: float,
    clear_right_media: bool = True,
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Split a base scene at a specified timestamp into two scenes (Section 15.4 T7, 15.6)."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        segments = session.setdefault("segments", [])
        sort_segments(segments)

        idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
        if idx < 0:
            raise SceneNotFoundError(scene_id, project_id)

        left = segments[idx]
        if has_locked_video(left):
            raise ValidationError("Scene has rendered video. Clear or trim video before splitting.")

        start = float(left.get("start", 0.0) or 0.0)
        end = float(left.get("end", start) or start)
        t = float(at_time)

        if t < start + 0.05 or t > end - 0.05:
            raise ValidationError(f"Split time {t}s must be between {start + 0.05}s and {end - 0.05}s.")

        slot_number = idx + 2
        renamed = []

        journal = TimelineJournal(folder, "split_scene")
        journal.start()
        try:
            if slot_number <= len(segments):
                renamed = _renumber_scene_assets_after_insert(folder, slot_number)
                journal.record_renames(renamed)
                rewrite_renamed_scene_paths(session, renamed)

            left["end"] = round(t, 4)

            right = dict(left)
            right["id"] = _generate_scene_id()
            right["start"] = round(t, 4)
            right["end"] = round(end, 4)
            right["source"] = "split"

            # Each half keeps the lyrics sung during its own part of the scene.
            if str(left.get("lyric_text") or "").strip():
                left_text, right_text = split_lyric_text(left.get("lyric_text"), (t - start) / max(end - start, 0.01))
                left["lyric_text"], right["lyric_text"] = left_text, right_text
                left["lyric_no_lip_sync"] = is_instrumental_lyric_text(left_text)
                right["lyric_no_lip_sync"] = is_instrumental_lyric_text(right_text)

            if clear_right_media:
                right.pop("approved_image_path", None)
                right.pop("video_path", None)
                right.pop("video_output", None)
                right.pop("video_status", None)
                right.pop("image_history", None)
                right.pop("video_history", None)
                right["video_status"] = "none"

            segments.insert(idx + 1, right)
            renumber_generic_base_scene_labels(segments)

            save_result = _persist_session(folder, session)
            journal.commit()

            return {
                "left_scene": left,
                "right_scene": right,
                "renamed": renamed,
                "revision": save_result.get("revision", current_rev + 1),
            }
        except Exception:
            journal.rollback()
            raise


def merge_scenes(
    project_id: str,
    scene_id: str,
    with_direction: str = "next",
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Merge two adjacent base scenes without video into one (Section 15.4 T9, 15.6)."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        segments = session.setdefault("segments", [])
        sort_segments(segments)

        target_idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
        if target_idx < 0:
            raise SceneNotFoundError(scene_id, project_id)

        direction = str(with_direction or "next").lower()
        if direction == "next":
            left_idx = target_idx
            right_idx = target_idx + 1
        elif direction == "previous":
            left_idx = target_idx - 1
            right_idx = target_idx
        else:
            raise ValidationError(f"Invalid merge direction '{with_direction}'. Must be 'next' or 'previous'.")

        if left_idx < 0 or right_idx >= len(segments):
            raise ValidationError(f"Cannot merge scene {scene_id} with {direction}: no adjacent neighbor.")

        left = segments[left_idx]
        right = segments[right_idx]

        if has_locked_video(left) or has_locked_video(right):
            raise ValidationError("Cannot merge scenes: one or both scenes already have rendered video.")

        slot_number = right_idx + 1
        journal = TimelineJournal(folder, "merge_scenes")
        journal.start()
        try:
            renamed = _renumber_scene_assets_after_removal(folder, slot_number)
            journal.record_renames(renamed)
            rewrite_renamed_scene_paths(session, renamed)

            left["start"] = min(float(left.get("start", 0.0) or 0.0), float(right.get("start", 0.0) or 0.0))
            left["end"] = max(float(left.get("end", 0.0) or 0.0), float(right.get("end", 0.0) or 0.0))

            notes_parts = [str(left.get("notes") or "").strip(), str(right.get("notes") or "").strip()]
            left["notes"] = "\n\n".join(p for p in notes_parts if p)
            if str(left.get("lyric_text") or "").strip() or str(right.get("lyric_text") or "").strip():
                left["lyric_text"] = merge_lyric_text(left.get("lyric_text"), right.get("lyric_text"))
                left["lyric_no_lip_sync"] = is_instrumental_lyric_text(left["lyric_text"])
            note_parts = [str(left.get("timeline_note") or "").strip(), str(right.get("timeline_note") or "").strip()]
            left["timeline_note"] = "\n".join(p for p in note_parts if p)

            segments.pop(right_idx)
            renumber_generic_base_scene_labels(segments)

            save_result = _persist_session(folder, session)
            journal.commit()

            return {
                "merged_scene": left,
                "renamed": renamed,
                "revision": save_result.get("revision", current_rev + 1),
            }
        except Exception:
            journal.rollback()
            raise


def move_scene(
    project_id: str,
    scene_id: str,
    start_time: float,
    ripple: bool = False,
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Move scene start boundary (Section 15.4 T10)."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        segments = session.setdefault("segments", [])
        res = move_scene_timing(segments, scene_id, start_time, ripple=ripple)
        save_result = _persist_session(folder, session)
        return {
            "scene": res["scene"],
            "delta": res["delta"],
            "revision": save_result.get("revision", current_rev + 1),
        }


def resize_scene(
    project_id: str,
    scene_id: str,
    duration: Optional[float] = None,
    end_time: Optional[float] = None,
    ripple: bool = False,
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Resize scene duration or end boundary (Section 15.4 T10)."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        segments = session.setdefault("segments", [])
        res = resize_scene_timing(segments, scene_id, new_duration=duration, new_end=end_time, ripple=ripple)
        save_result = _persist_session(folder, session)
        return {
            "scene": res["scene"],
            "delta": res["delta"],
            "revision": save_result.get("revision", current_rev + 1),
        }


# Scene fields PATCH /scenes/{id} can change. Anything else is rejected, so a request can never report
# success while silently saving nothing. Scene-card fields (notes, camera, performance, references, ...) are
# listed in ``storyboard/scene_card_fields.py`` and are saved on both the timeline and the Storyboard card.
_SCENE_PATCH_TEXT_FIELDS = (
    "t2i_prompt",
    "i2v_prompt",
    "enhance_prompt",
    "minimax_h3_prompt",
    "minimax_h3_i2v_frame_mode",
    "first_last_frame_end_image_path",
    "minimax_h3_continuation_direction",
    "flux_prompt",
    "nb_prompt",
    "flow_gpt_prompt",
    "ernie_t2i_prompt",
    "video_path",
    "video_output",
    "video_status",
    "approved_image_path",
    "audio_path",
    "custom_audio_path",
)
_SCENE_PATCH_NUMBER_FIELDS = ("start", "end")
# A number, or null to go back to the default (0.5 s). It is kept between 0.5 s and half of the scene when it is used.
_SCENE_PATCH_OPTIONAL_NUMBER_FIELDS = ("minimax_h3_continuation_start_seconds",)
_SCENE_PATCH_ECHOED_FIELDS = ("id",)  # clients often send back what they read; the id cannot change
_PROMPT_KEYS = ("t2i_prompt", "i2v_prompt", "minimax_h3_prompt", "minimax_h3_pass2_prompt")


def _supported_scene_fields() -> List[str]:
    return sorted(
        set(_SCENE_PATCH_TEXT_FIELDS) | set(_SCENE_PATCH_NUMBER_FIELDS)
        | set(_SCENE_PATCH_OPTIONAL_NUMBER_FIELDS) | set(card_fields.field_names())
    )


def _unsupported_scene_fields(patch: Dict[str, Any]) -> List[str]:
    known = set(_supported_scene_fields()) | set(_SCENE_PATCH_ECHOED_FIELDS)
    return sorted(
        key for key in patch
        if key not in known and not (key.startswith("use_scene_") or key.endswith("_settings"))
    )


def _apply_scene_references(session: Dict[str, Any], scene_id: str, values: Dict[str, Any]) -> List[str]:
    """Map the scene's characters (``subject_ids``) and location (``location_id``) like the Reference Builder.

    An empty list or empty id is saved as an explicit "none" for this scene, so it does not fall back to an
    older number-keyed mapping. Returns the scene maps that changed ("subjects", "locations").
    """
    refs = session.setdefault("flux_reference_builder", {})
    changed = []
    for name, api_name, list_key, switch in (
        ("subject_ids", "subjects", "subjects", "use_subject_reference"),
        ("location_id", "locations", "locations", "use_location_references"),
    ):
        if name not in values:
            continue
        known = {str(item.get("id")) for item in refs.get(list_key) or [] if isinstance(item, dict)}
        wanted = values[name] if name == "subject_ids" else ([values[name]] if values[name] else [])
        unknown = [item for item in wanted if item not in known]
        if unknown:
            raise ValidationError(
                f"Unknown {list_key[:-1]} id{'s' if len(unknown) > 1 else ''}: {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(known)) or 'none'}."
            )
        scene_map = _reference_map(session, api_name)
        new_value = list(values[name]) if name == "subject_ids" else values[name]
        if scene_map.get(scene_id) != new_value:
            scene_map[scene_id] = new_value
            changed.append(api_name)
        refs[switch] = bool(refs.get(list_key) or any(scene_map.values()))
    return changed


def _storyboard_card_updates(session: Dict[str, Any], scene: Dict[str, Any], values: Dict[str, Any],
                             legacy: Dict[str, Any], references_changed: List[str]) -> Dict[str, Any]:
    """Storyboard card keys to write for a scene patch, keyed by card field."""
    updates: Dict[str, Any] = {}
    plain = {name: value for name, value in values.items() if name not in ("subject_ids", "location_id")}
    card = {}
    card_fields.apply_to_card(card, plain)
    updates.update(card)
    engine = session_sync.video_engine(session)
    if any(key in legacy for key in ("t2i_prompt", "flux_prompt", "nb_prompt", "flow_gpt_prompt", "ernie_t2i_prompt")):
        # The card shows the prompt the Builder picks for the project's image model.
        updates["image_prompt"] = session_sync.storyboard_prompt_for_segment(scene, session_sync.image_mode(session))
    prompt_key, origin_key = card_fields.video_prompt_keys(engine)
    if prompt_key in legacy:
        updates["video_prompt"] = str(scene.get(prompt_key) or "")
        updates["video_prompt_origin"] = "gemma" if str(scene.get(origin_key) or "").lower() == "gemma" else "manual"
    if references_changed or "no_character_present" in values:
        fresh = next((c for c in session_sync.scene_cards(session) if c["id"] == scene.get("id")), None)
        if fresh:
            for key in ("subject_refs", "subjects", "location_ref", "setting"):
                updates[key] = fresh[key]
    return updates


def _write_storyboard_card(folder: str, session: Dict[str, Any], scene_id: str, updates: Dict[str, Any],
                           card_only: List[str]) -> tuple:
    """Save the card changes into ``storyboard.json``. Returns (changed card keys, undo).

    Without a saved Storyboard nothing is written unless a field only the card stores is set; the Storyboard
    then starts from the timeline, as it does when it opens. A card the user deleted from the Storyboard stays
    deleted, and setting a card-only field on it is an error.
    """
    if not updates:
        return [], None
    saved = storyboard_store._load_storyboard({"project_folder": folder})
    exists = bool(saved.get("exists"))
    if not exists and not card_only:
        return [], None
    if exists:
        storyboard = {key: value for key, value in saved.items() if key not in ("path", "exists")}
    else:
        storyboard = session_sync.merge_storyboard(None, session)
    scenes = storyboard.setdefault("scenes", [])
    card = next((item for item in scenes if isinstance(item, dict) and item.get("id") == scene_id), None)
    if card is None:
        source_ids = storyboard.get("source_scene_ids")
        if isinstance(source_ids, list) and scene_id in source_ids:
            if card_only:
                raise ValidationError(
                    f"Scene {scene_id} was removed from the Storyboard, so {', '.join(card_only)} cannot be saved "
                    "on its card. The timeline fields were not changed either."
                )
            return [], None
        card = next((dict(c) for c in session_sync.scene_cards(session) if c["id"] == scene_id), None)
        if card is None:
            return [], None
        card["status"] = session_sync.video_prompt_status(card)
        scenes.append(card)
        storyboard["source_scene_ids"] = list(source_ids or []) + [scene_id]
    before = copy.deepcopy(card)
    card.update(copy.deepcopy(updates))
    # A timeline scene's card uses the project's engine (the Storyboard sets it when it opens); an LTX card
    # would get LTX facial wording added to its video prompt on save.
    card["project_video_engine"] = session_sync.video_engine(session)
    if "video_prompt" in updates:
        card["status"] = session_sync.video_prompt_status(card, str(before.get("status") or ""))
    changed = sorted(key for key in set(before) | set(card) if before.get(key) != card.get(key))
    if not changed and exists:
        return [], None
    path = saved.get("path") or ""
    previous = None
    if exists and path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            previous = handle.read()
    storyboard_store._save_storyboard({"project_folder": folder, "storyboard": storyboard})

    def undo() -> None:
        if previous is not None:
            atomic_write_text(path, previous)
        elif path and os.path.isfile(path):
            os.remove(path)

    return changed, undo


def patch_scene(
    project_id: str,
    scene_id: str,
    patch: Dict[str, Any],
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Merge-patch scene fields (prompts, scene-card fields, timing, overrides) and mark latents dirty on edits.

    Scene-card fields are written with the Builder's keys on the timeline segment and, when the project has a
    saved Storyboard (or a field only the card stores is set), on the Storyboard card too. Unsupported fields
    raise a ValidationError that lists what can be changed; nothing is saved then.
    """
    if not isinstance(patch, dict):
        raise ValidationError("The scene patch must be a JSON object.")
    if "minimax_h3_i2v_frame_mode" in patch and patch["minimax_h3_i2v_frame_mode"] not in ("normal", "flf", "", None):
        raise ValidationError("minimax_h3_i2v_frame_mode must be normal or flf.")
    try:
        values, legacy = card_fields.parse_scene_card_patch(patch)
    except card_fields.SceneCardFieldError as exc:
        raise ValidationError(str(exc)) from None
    unsupported = _unsupported_scene_fields(legacy)
    if unsupported:
        raise ValidationError(
            f"Unsupported scene field{'s' if len(unsupported) > 1 else ''}: {', '.join(unsupported)}. "
            f"Supported fields: {', '.join(_supported_scene_fields())}, plus use_scene_* and *_settings. "
            "POST /scenes/bulk with op=patch can write other fields."
        )
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        segments = session.setdefault("segments", [])
        sort_segments(segments)

        idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
        if idx < 0:
            raise SceneNotFoundError(scene_id, project_id)

        scene = segments[idx]
        slot_number = idx + 1
        resolved_id = str(scene.get("id") or scene_id)
        before = copy.deepcopy(scene)

        for key in _SCENE_PATCH_TEXT_FIELDS:
            if key in legacy:
                scene[key] = str(legacy[key] or "")

        for key in _SCENE_PATCH_NUMBER_FIELDS:
            if key in legacy:
                scene[key] = float(legacy[key])

        for key in _SCENE_PATCH_OPTIONAL_NUMBER_FIELDS:
            if key in legacy:
                value = legacy[key]
                if value is None or value == "":
                    scene[key] = None
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    raise ValidationError(f"{key} must be a number of seconds, or null for the default.")
                if number != number or number in (float("inf"), float("-inf")) or number < 0:
                    raise ValidationError(f"{key} must be a number of seconds that is 0 or more, or null for the default.")
                scene[key] = round(number, 2)

        references_changed = _apply_scene_references(session, resolved_id, values)
        card_fields.apply_to_segment(
            scene,
            {name: value for name, value in values.items() if name not in ("subject_ids", "location_id")},
            video_engine=session_sync.video_engine(session),
            image_mode=session_sync.image_mode(session),
        )

        # Same as editing the lyric in the Builder: the scene is marked instrumental from its text.
        if "lyric_text" in values and "lyric_no_lip_sync" not in values:
            scene["lyric_no_lip_sync"] = is_instrumental_lyric_text(scene.get("lyric_text"))

        for key, val in legacy.items():
            if key.startswith("use_scene_") or key.endswith("_settings"):
                scene[key] = val

        prompt_changed = any(str(before.get(key) or "") != str(scene.get(key) or "") for key in _PROMPT_KEYS)
        old_dur = float(before.get("end", 0.0) or 0.0) - float(before.get("start", 0.0) or 0.0)
        new_dur = float(scene.get("end", 0.0) or 0.0) - float(scene.get("start", 0.0) or 0.0)
        duration_changed = abs(old_dur - new_dur) > 0.05

        if prompt_changed or duration_changed:
            try:
                SceneLatentManager.mark_dirty(folder, slot_number)
            except Exception:
                pass

        if "start" in legacy or "end" in legacy:
            normalize_segments(segments, active_index=idx)

        segment_changed = sorted(key for key in set(before) | set(scene) if before.get(key) != scene.get(key))
        card_updates = _storyboard_card_updates(session, scene, values, legacy, references_changed)
        card_changed, undo_card = _write_storyboard_card(
            folder, session, resolved_id, card_updates, card_fields.card_only_names(values)
        )
        timing_changed = "start" in legacy or "end" in legacy
        change = {
            "kind": "project" if timing_changed else "scene_fields",
            "scenes": {resolved_id: {
                "segment": segment_changed, "card": card_changed, "references": references_changed,
            }},
            "storyboard": bool(card_changed),
        }
        try:
            save_result = _persist_session(folder, session, change)
        except Exception:
            if undo_card:
                undo_card()
            raise
        return {
            "scene": scene,
            "storyboard_card": card_changed,
            "revision": save_result.get("revision", current_rev + 1),
        }


def bulk_scene_operations(
    project_id: str,
    operations: List[Dict[str, Any]],
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a batch of scene mutations atomically in one write (Section 6.4)."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        segments = session.setdefault("segments", [])
        applied = 0
        for op in operations:
            op_type = op.get("op") or op.get("action")
            if op_type == "move":
                move_scene_timing(segments, op["scene_id"], float(op["start"]), ripple=bool(op.get("ripple", False)))
                applied += 1
            elif op_type == "resize":
                resize_scene_timing(
                    segments,
                    op["scene_id"],
                    new_duration=op.get("duration"),
                    new_end=op.get("end"),
                    ripple=bool(op.get("ripple", False)),
                )
                applied += 1
            elif op_type == "patch":
                idx = next((i for i, s in enumerate(segments) if s.get("id") == op["scene_id"]), -1)
                if idx >= 0:
                    fields = op.get("fields", {})
                    for k, v in fields.items():
                        segments[idx][k] = v
                    if "lyric_text" in fields and "lyric_no_lip_sync" not in fields:
                        segments[idx]["lyric_no_lip_sync"] = is_instrumental_lyric_text(segments[idx].get("lyric_text"))
                    applied += 1

        save_result = _persist_session(folder, session)
        return {
            "applied_operations": applied,
            "revision": save_result.get("revision", current_rev + 1),
        }


# ==============================================================================
# 2. Timeline Batch Operations (Section 15.4 T13, T14, T15, T16)
# ==============================================================================

def timeline_close_gaps(project_id: str) -> Dict[str, Any]:
    """Close all gaps between base scenes (T15)."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        segments = session.setdefault("segments", [])
        overlays = session.setdefault("overlay_segments", [])

        removed = close_all_base_timeline_gaps(segments, overlays)
        save_result = _persist_session(folder, session)
        return {
            "removed_duration": removed,
            "revision": save_result.get("revision", 1),
        }


def enforce_scene_lengths_on_project(
    project_id: str,
    min_scene_seconds: float,
    max_scene_seconds: float,
    dry_run: bool = False,
    max_edits: int = 400,
) -> Dict[str, Any]:
    """Merge scenes shorter than ``min`` and cut scenes longer than ``max``, using the transactional edits.

    Long scenes are split into equal parts (a 12 s scene becomes 6 s + 6 s) and their lyrics are divided
    between the parts. Short scenes are merged into the neighbor that gives the shorter result. Scenes
    with rendered video are left alone and reported. ``dry_run`` returns the edits without making them.
    """
    edits: List[Dict[str, Any]] = []
    _folder, session = _get_active_session_and_folder(project_id)
    scenes_before = len(session.get("segments") or [])
    working = [dict(s) for s in session.get("segments") or []] if dry_run else None

    for _ in range(max(1, int(max_edits))):
        if dry_run:
            locked = {s.get("id") for s in working if has_locked_video(s)}
            fix = next_length_fix(working, min_scene_seconds, max_scene_seconds, locked)
            if fix is None:
                break
            apply_length_fix(working, fix)
        else:
            _folder, session = _get_active_session_and_folder(project_id)
            segments = session.get("segments") or []
            locked = {s.get("id") for s in segments if has_locked_video(s)}
            fix = next_length_fix(segments, min_scene_seconds, max_scene_seconds, locked)
            if fix is None:
                break
            if fix["op"] == "split":
                split_scene(project_id, fix["scene_id"], fix["at_time"])
            else:
                merge_scenes(project_id, fix["scene_id"], with_direction=fix["with"])
        edits.append(fix)
    else:
        raise ValidationError(f"Stopped after {max_edits} edits without reaching the limits; check the minimum and maximum.")

    final = working if dry_run else (_get_active_session_and_folder(project_id)[1].get("segments") or [])
    locked_ids = {s.get("id") for s in final if has_locked_video(s)}
    lengths = [float(s.get("end", 0)) - float(s.get("start", 0)) for s in final]
    still_outside = [
        {"scene_id": s.get("id"), "seconds": round(float(s["end"]) - float(s["start"]), 2), "reason": "has rendered video" if s.get("id") in locked_ids else "no valid edit"}
        for s in final
        if (float(s["end"]) - float(s["start"]) > float(max_scene_seconds) + 0.01
            or (len(final) > 1 and float(s["end"]) - float(s["start"]) < float(min_scene_seconds) - 0.01))
    ]
    return {
        "dry_run": bool(dry_run),
        "splits": sum(1 for e in edits if e["op"] == "split"),
        "merges": sum(1 for e in edits if e["op"] == "merge"),
        "scenes_before": scenes_before,
        "scenes_after": len(final),
        "shortest_seconds": round(min(lengths), 2) if lengths else None,
        "longest_seconds": round(max(lengths), 2) if lengths else None,
        "still_outside_limits": still_outside,
    }


def timeline_snap(
    project_id: str,
    scope: str = "edge",
    scene_id: Optional[str] = None,
    edge: str = "start",
) -> Dict[str, Any]:
    """Snap scene boundaries to nearest beats (Section 15.4 T13, T14)."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        segments = session.setdefault("segments", [])
        beats = _session_beats(session)
        if not beats:
            raise ValidationError("No beats loaded in project. Analyze audio first.")

        if scope == "edge":
            idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), 0)
            if idx < len(segments):
                scene = segments[idx]
                target_key = "start" if edge == "start" else "end"
                scene[target_key] = snap_to_beat(float(scene.get(target_key, 0.0) or 0.0), beats)
                normalize_segments(segments, active_index=idx)
        elif scope == "all_starts":
            for i, s in enumerate(segments):
                if i > 0 and not has_locked_video(s):
                    s["start"] = snap_to_beat(float(s.get("start", 0.0) or 0.0), beats)
            for i in range(len(segments)):
                normalize_segments(segments, active_index=i)

        save_result = _persist_session(folder, session)
        return {
            "snapped": True,
            "revision": save_result.get("revision", 1),
        }


def _bulk_scene(start: float, end: float, lyric: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    scene: Dict[str, Any] = {"id": _generate_scene_id(), "start": start, "end": end, "source": "manual"}
    for key in ("lyric_text", "lyric_no_lip_sync", "lyric_singers"):
        if lyric and key in lyric:
            scene[key] = lyric[key]
    return scene


def timeline_bulk(
    project_id: str,
    text: str,
    mode: str = "durations",
    action: str = "replace",
    append_start: float = 0.0,
    clear_media: bool = False,
) -> Dict[str, Any]:
    """Apply bulk timings from text (Section 15.4 T16).

    A line can end with the words for its scene (``12.5 --> 16.0 Hello darkness``). When ``action`` is
    ``replace`` and no line carries words, the lyrics of the scenes being replaced move onto the new
    scenes by time, so replacing the timeline no longer wipes them.
    """
    parsed = parse_bulk_scenes(text, mode=mode)
    if not parsed:
        raise ValidationError("No valid timing lines found in input.")
    if action not in ("replace", "append"):
        raise ValidationError("action must be 'replace' or 'append'.")

    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        segments = session.setdefault("segments", [])

        explicit_lyrics = any(item.get("lyric_text") for item in parsed)
        lyrics_source = "from_text" if explicit_lyrics else "none"
        lyric_fields: List[Dict[str, Any]] = [{} for _ in parsed]
        if explicit_lyrics:
            for index, item in enumerate(parsed):
                if item.get("lyric_text"):
                    lyric_fields[index] = {
                        "lyric_text": item["lyric_text"],
                        "lyric_no_lip_sync": is_instrumental_lyric_text(item["lyric_text"]),
                    }

        new_segments: List[Dict[str, Any]] = []
        if action == "replace":
            if not explicit_lyrics and any(str(s.get("lyric_text") or "").strip() for s in segments):
                lyric_fields = carry_lyrics_over(segments, [(item["start"], item["end"]) for item in parsed])
                lyrics_source = "carried_over" if any(lyric_fields) else "none"
            for item, lyric in zip(parsed, lyric_fields):
                new_segments.append(_bulk_scene(item["start"], item["end"], lyric))
            session["segments"] = new_segments
            session["timing_frozen"] = False
        else:
            start_cursor = float(append_start) if append_start > 0 else (float(segments[-1]["end"]) if segments else 0.0)
            for item, lyric in zip(parsed, lyric_fields):
                dur = item["end"] - item["start"]
                new_segments.append(_bulk_scene(round(start_cursor, 4), round(start_cursor + dur, 4), lyric))
                start_cursor += dur
            segments.extend(new_segments)

        renumber_generic_base_scene_labels(session["segments"])
        save_result = _persist_session(folder, session)
        return {
            "scene_count": len(session["segments"]),
            "scenes_with_lyrics": sum(1 for scene in new_segments if str(scene.get("lyric_text") or "").strip()),
            "lyrics": lyrics_source,
            "revision": save_result.get("revision", 1),
        }


# ==============================================================================
# 2b. Timed Timeline Notes (the Builder's "+ Timeline Note" markers)
# ==============================================================================

def _timeline_note_view(marker: Dict[str, Any], segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A saved note plus the scenes it overlaps (the scenes Story Arc planning applies it to)."""
    return {**marker, "scene_ids": markers_service.overlapping_scene_ids(marker, segments)}


def list_timeline_notes(project_id: str) -> Dict[str, Any]:
    """The project's timed Timeline Notes, sorted by start time."""
    _folder, session = _get_active_session_and_folder(project_id)
    segments = [s for s in session.get("segments") or [] if isinstance(s, dict)]
    notes = markers_service.normalize_markers(session.get("timeline_markers"))
    return {
        "notes": [_timeline_note_view(marker, segments) for marker in notes],
        "count": len(notes),
        "revision": int(session.get("revision") or session.get("builder_save_revision") or 0),
    }


def _edit_timeline_notes(project_id: str, if_match_revision: Optional[int], edit) -> Dict[str, Any]:
    """Load the notes under the save lock, apply ``edit(notes) -> (note, marker_id)``, save and notify."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)
        notes = markers_service.normalize_markers(session.get("timeline_markers"))
        try:
            note, marker_id = edit(notes, session)
        except markers_service.TimelineMarkerError as exc:
            raise ValidationError(str(exc)) from None
        except KeyError as exc:
            raise TimelineNoteNotFoundError(str(exc.args[0]), project_id) from None
        session["timeline_markers"] = notes
        save_result = _persist_session(folder, session, {"kind": "timeline_markers", "marker_ids": [marker_id]})
        segments = [s for s in session.get("segments") or [] if isinstance(s, dict)]
        return {
            "note": _timeline_note_view(note, segments) if note else None,
            "notes": [_timeline_note_view(marker, segments) for marker in notes],
            "revision": save_result.get("revision", current_rev + 1),
        }


def create_timeline_note(
    project_id: str,
    fields: Dict[str, Any],
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Add a timed note. ``start`` is required; leave ``end`` out (or null) for a point note."""
    def edit(notes, _session):
        note = markers_service.create_marker(notes, fields)
        return note, note["id"]
    return _edit_timeline_notes(project_id, if_match_revision, edit)


def update_timeline_note(project_id: str, note_id: str, fields: Dict[str, Any],
                         if_match_revision: Optional[int] = None) -> Dict[str, Any]:
    """Change only the given fields of a note. ``end: null`` turns a range note into a point note."""
    def edit(notes, _session):
        return markers_service.update_marker(notes, note_id, fields), note_id
    return _edit_timeline_notes(project_id, if_match_revision, edit)


def delete_timeline_note(project_id: str, note_id: str, if_match_revision: Optional[int] = None) -> Dict[str, Any]:
    """Remove a note. The Builder's active-note selection is cleared when it pointed at it."""
    def edit(notes, session):
        removed = markers_service.delete_marker(notes, note_id)
        if session.get("active_timeline_marker_id") == note_id:
            session["active_timeline_marker_id"] = ""
        return removed, note_id
    result = _edit_timeline_notes(project_id, if_match_revision, edit)
    result["deleted"] = result.pop("note")
    return result


# ==============================================================================
# 3. References CRUD & Mapping (Section 6.5, Section 24, T25-T33)
# ==============================================================================

# Reference scene maps the Video Builder keeps inside session["flux_reference_builder"],
# keyed by API name. Older versions of this API wrote subjects/locations at the top level
# of the session, where the UI never reads them, so those are migrated on write.
_REFERENCE_MAP_KEYS = {
    "subjects": "subject_scene_map",
    "locations": "scene_map",
    "ingredients": "ingredients_scene_map",
    "extras": "extra_scene_map",
}


def _reference_map(session: Dict[str, Any], api_name: str) -> Dict[str, Any]:
    """Return the live scene map for ``api_name``, folding in any legacy top-level map."""
    key = _REFERENCE_MAP_KEYS[api_name]
    ref_builder = session.setdefault("flux_reference_builder", {})
    current = ref_builder.setdefault(key, {})
    legacy = session.get(key) if api_name in ("subjects", "locations") else None
    if isinstance(legacy, dict) and legacy:
        for scene_id, value in legacy.items():
            current.setdefault(scene_id, value)
        session.pop(key, None)
    return current


def get_project_references(project_id: str) -> Dict[str, Any]:
    """Get all subjects, locations, and scene mappings for a project."""
    _folder, session = _get_active_session_and_folder(project_id)
    ref_builder = session.get("flux_reference_builder", {})
    return {
        "subjects": ref_builder.get("subjects", []),
        "locations": ref_builder.get("locations", []),
        "scene_mapping": {name: dict(_reference_map(session, name)) for name in _REFERENCE_MAP_KEYS},
    }


def upsert_reference_subject(
    project_id: str,
    subject_id: str,
    payload: Dict[str, Any],
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Create or update a subject reference."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        ref_builder = session.setdefault("flux_reference_builder", {})
        subjects = ref_builder.setdefault("subjects", [])

        idx = next((i for i, s in enumerate(subjects) if s.get("id") == subject_id), -1)
        existing = subjects[idx] if idx >= 0 else {}
        voice = payload.get("minimax_voice")
        if voice is None:
            voice = existing.get("minimax_voice", "none")
        subj = {
            "id": subject_id,
            "name": payload.get("name", existing.get("name", "Character")),
            "description": payload.get("description", existing.get("description", "")),
            "face_description": payload.get("face_description", existing.get("face_description", "")),
            "reference_type": payload.get("reference_type", existing.get("reference_type", "character")),
            "minimax_voice": voice,
            "trigger_phrase": payload.get("trigger_phrase", existing.get("trigger_phrase", "")),
            "trigger_position": payload.get("trigger_position", existing.get("trigger_position", "start")),
            "extra_reference_for": payload.get("extra_reference_for", existing.get("extra_reference_for", "")),
            "extra_reference_note": payload.get("extra_reference_note", existing.get("extra_reference_note", "")),
            "image": payload.get("image", existing.get("image", {})),
        }

        if idx >= 0:
            subjects[idx].update(subj)
            item = subjects[idx]
        else:
            subjects.append(subj)
            item = subj

        save_result = _persist_session(folder, session)
        return {
            "subject": item,
            "revision": save_result.get("revision", current_rev + 1),
        }


def upsert_reference_location(
    project_id: str,
    location_id: str,
    payload: Dict[str, Any],
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Create or update a location reference."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        ref_builder = session.setdefault("flux_reference_builder", {})
        locations = ref_builder.setdefault("locations", [])

        idx = next((i for i, loc in enumerate(locations) if loc.get("id") == location_id), -1)
        loc = {
            "id": location_id,
            "name": payload.get("name", "Location"),
            "description": payload.get("description", ""),
            "trigger_phrase": payload.get("trigger_phrase", ""),
            "trigger_position": payload.get("trigger_position", "start"),
            "image": payload.get("image", {}),
        }

        if idx >= 0:
            locations[idx].update(loc)
            item = locations[idx]
        else:
            locations.append(loc)
            item = loc

        save_result = _persist_session(folder, session)
        return {
            "location": item,
            "revision": save_result.get("revision", current_rev + 1),
        }


def delete_reference(
    project_id: str,
    kind: str,
    ref_id: str,
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Delete a reference subject or location and clean up scene maps (Q-T5)."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        ref_builder = session.setdefault("flux_reference_builder", {})
        target_list_key = "subjects" if kind in ("subjects", "subject") else "locations"
        items = ref_builder.setdefault(target_list_key, [])

        initial_len = len(items)
        items[:] = [item for item in items if item.get("id") != ref_id]

        # Clean scene mappings (Q-T5)
        if target_list_key == "subjects":
            subj_map = _reference_map(session, "subjects")
            for sid, sub_list in list(subj_map.items()):
                if isinstance(sub_list, list):
                    subj_map[sid] = [s for s in sub_list if s != ref_id]
        else:
            loc_map = _reference_map(session, "locations")
            for sid, loc_val in list(loc_map.items()):
                if loc_val == ref_id:
                    loc_map.pop(sid, None)

        save_result = _persist_session(folder, session)
        return {
            "deleted": len(items) < initial_len,
            "kind": target_list_key,
            "id": ref_id,
            "revision": save_result.get("revision", current_rev + 1),
        }


def update_scene_reference_mapping(
    project_id: str,
    mapping: Dict[str, Any],
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Update subject and location assignments for scenes."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        for api_name in _REFERENCE_MAP_KEYS:
            current = _reference_map(session, api_name)
            if isinstance(mapping.get(api_name), dict):
                current.update(mapping[api_name])

        # Same switches the Builder sets after a scene mapping (reference_scene_mapping.mjs), so the
        # mapped subjects and locations are actually used.
        ref_builder = session["flux_reference_builder"]
        for api_name, list_key, switch in (
            ("subjects", "subjects", "use_subject_reference"),
            ("locations", "locations", "use_location_references"),
        ):
            if isinstance(mapping.get(api_name), dict):
                ref_builder[switch] = bool(ref_builder.get(list_key) or _reference_map(session, api_name))

        save_result = _persist_session(folder, session)
        return {
            "scene_mapping": {name: dict(_reference_map(session, name)) for name in _REFERENCE_MAP_KEYS},
            "revision": save_result.get("revision", current_rev + 1),
        }


def _sorted_scene_with_index(session: Dict[str, Any], scene_id: str, project_id: str):
    """The scene (by id or 1-based number) and its position in timeline order, which the reference maps use."""
    segments = sorted(session.get("segments") or [], key=lambda s: float(s.get("start", 0.0) or 0.0))
    index = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if index < 0:
        raise SceneNotFoundError(scene_id, project_id)
    return segments[index], index


def _scene_reference_choices(session: Dict[str, Any], scene_id: str, project_id: str) -> Dict[str, Any]:
    segment, index = _sorted_scene_with_index(session, scene_id, project_id)
    mode = str(minimax_h3_settings_for_scene(session, segment).get("video_mode") or "")
    return reference_choices(session, segment, index, mode)


def get_scene_minimax_references(project_id: str, scene_id: str) -> Dict[str, Any]:
    """What "Choose MiniMax References" shows for a scene: every reference it can use, and the order it sends."""
    _folder, session = _get_active_session_and_folder(project_id)
    return _scene_reference_choices(session, scene_id, project_id)


def set_scene_minimax_references(
    project_id: str,
    scene_id: str,
    keys: Any = None,
    automatic: bool = False,
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Choose the ordered MiniMax references for one scene (``keys``), or hand it back to the scene mappings."""
    if automatic and keys is not None:
        raise ValidationError("Send either `keys` (a chosen order) or `automatic: true`, not both.")
    if not automatic and keys is None:
        raise ValidationError("Send `keys` (a list of reference keys) or `automatic: true`.")
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        segment, _index = _sorted_scene_with_index(session, scene_id, project_id)
        if automatic:
            segment["minimax_h3_reference_keys"] = None
        else:
            try:
                segment["minimax_h3_reference_keys"] = validate_reference_keys(
                    _scene_reference_choices(session, scene_id, project_id), keys,
                )
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc

        save_result = _persist_session(folder, session)
        return {
            **_scene_reference_choices(session, scene_id, project_id),
            "revision": save_result.get("revision", current_rev + 1),
        }


# ==============================================================================
# 4. Audio, Lyrics, Beats & Timing (Section 6.3)
# ==============================================================================

def get_project_lyrics(project_id: str) -> Dict[str, Any]:
    """Get canonical lyrics text and SRT content."""
    folder, session = _get_active_session_and_folder(project_id)
    lyrics_text = str(session.get("canonical_lyrics") or "")
    if not lyrics_text:
        possible_paths = [
            os.path.join(folder, "project_context", "full_lyrics.txt"),
            os.path.join(folder, "full_lyrics.txt"),
            os.path.join(folder, "canonical_full_lyrics.txt"),
        ]
        for p in possible_paths:
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        lyrics_text = f.read()
                    if lyrics_text.strip():
                        break
                except Exception:
                    pass

    srt_file = _srt_path(folder)
    srt_text = ""
    if os.path.isfile(srt_file):
        try:
            with open(srt_file, "r", encoding="utf-8") as f:
                srt_text = f.read()
        except Exception:
            pass

    return {
        "lyrics_text": lyrics_text,
        "srt_text": srt_text,
        "has_srt": bool(srt_text.strip()),
    }


def set_project_lyrics(
    project_id: str,
    lyrics_text: Optional[str] = None,
    srt_text: Optional[str] = None,
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Set canonical lyrics text and or SRT file for the project."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        if lyrics_text is not None:
            _save_canonical_full_lyrics(folder, lyrics_text)
            session["canonical_lyrics"] = lyrics_text

        if srt_text is not None:
            _save_project_srt({"project_folder": folder, "srt_text": srt_text})
            session["srt_mode"] = True

        save_result = _persist_session(folder, session)
        return {
            "lyrics_saved": lyrics_text is not None,
            "srt_saved": srt_text is not None,
            "revision": save_result.get("revision", current_rev + 1),
        }


def attach_project_audio(
    project_id: str,
    audio_path: Optional[str] = None,
    audio_data: Optional[str] = None,
    audio_name: Optional[str] = None,
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Attach audio file or base64 data to project."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        payload = {
            "project_folder": folder,
            "source_path": audio_path,
            "audio_data": audio_data,
            # Keep the source file's own extension (an .mp3 saved as .wav would be a mislabeled file).
            "audio_name": audio_name or (os.path.basename(str(audio_path)) if audio_path else "") or "audio.wav",
        }
        res = _save_project_audio(payload)
        saved_path = res.get("saved_path") or res.get("audio_path")
        if not saved_path:
            raise ValidationError("The audio file could not be saved into the project.")
        # Same keys the Video Builder saves: audio_path, audio_duration, audio_peaks, beat_markers.
        session["audio_path"] = saved_path
        session["audio_duration"] = float(res.get("duration") or 0.0)
        peaks = res.get("peaks") if isinstance(res.get("peaks"), list) else []
        session["audio_peaks"] = peaks
        beats = res.get("beats") if isinstance(res.get("beats"), list) else []
        session["beat_markers"] = beats
        if res.get("tempo_bpm"):
            session["detected_tempo_bpm"] = float(res["tempo_bpm"])

        save_result = _persist_session(folder, session)
        return {
            "audio_path": saved_path,
            "duration": session["audio_duration"],
            "peaks_count": len(peaks),
            "beat_count": len(beats),
            "tempo_bpm": session.get("detected_tempo_bpm"),
            "revision": save_result.get("revision", current_rev + 1),
        }


def create_project_silent_audio(
    project_id: str,
    duration: Optional[float] = None,
    scope: str = "project",
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a silent WAV audio track."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        payload = {
            "project_folder": folder,
            "duration": duration or 10.0,
            "scope": scope,
        }
        res = _create_silent_audio(payload)
        session["audio_path"] = res.get("audio_path")
        session["audio_duration"] = res.get("duration", 0.0)
        if isinstance(res.get("peaks"), list):
            session["audio_peaks"] = res["peaks"]

        save_result = _persist_session(folder, session)
        return {
            "audio_path": res.get("audio_path"),
            "duration": res.get("duration"),
            "revision": save_result.get("revision", current_rev + 1),
        }


def get_audio_waveform(project_id: str, target_peaks: int = 1600) -> Dict[str, Any]:
    """Get audio waveform peaks for timeline display."""
    folder, session = _get_active_session_and_folder(project_id)
    audio_path = session.get("audio_path")
    if not audio_path or not os.path.isfile(audio_path):
        return {"peaks": [], "duration": 0.0}

    peaks = _read_audio_peaks(audio_path, target_peaks=target_peaks)
    return {
        "peaks": peaks,
        "duration": float(session.get("audio_duration", 0.0) or 0.0),
    }


def _session_beats(session: Dict[str, Any]) -> List[Any]:
    """Beat markers as the Video Builder saves them (``beat_markers``), falling back to the older ``beats`` key."""
    beats = session.get("beat_markers")
    if not isinstance(beats, list) or not beats:
        beats = session.get("beats")
    return beats if isinstance(beats, list) else []


def get_audio_beats(project_id: str) -> Dict[str, Any]:
    """Get detected audio beat markers and tempo."""
    _folder, session = _get_active_session_and_folder(project_id)
    return {
        "beats": _session_beats(session),
        "tempo_bpm": session.get("detected_tempo_bpm") or session.get("tempo_bpm") or session.get("bpm"),
        "beat_calibration": session.get("beat_calibration"),
    }


def set_audio_beats(
    project_id: str,
    beats: List[Any],
    tempo_bpm: Optional[float] = None,
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Update beat markers on project."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        session["beat_markers"] = beats
        session.pop("beats", None)
        if tempo_bpm is not None:
            session["detected_tempo_bpm"] = float(tempo_bpm)

        save_result = _persist_session(folder, session)
        return {
            "beat_count": len(beats),
            "tempo_bpm": session.get("detected_tempo_bpm"),
            "revision": save_result.get("revision", current_rev + 1),
        }


def calibrate_beats(
    project_id: str,
    offset_seconds: float = 0.0,
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Offset all beat times by offset_seconds."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        beats = _session_beats(session)
        offset = float(offset_seconds or 0.0)
        calibrated = []
        for b in beats:
            if isinstance(b, dict):
                nb = dict(b)
                nb["time"] = round(float(nb.get("time", 0.0)) + offset, 4)
                calibrated.append(nb)
            else:
                calibrated.append(round(float(b) + offset, 4))

        session["beat_markers"] = calibrated
        session.pop("beats", None)
        session["beat_calibration"] = {
            "offset_seconds": offset,
            "calibrated": True,
        }

        save_result = _persist_session(folder, session)
        return {
            "calibrated": True,
            "offset_seconds": offset,
            "beat_count": len(calibrated),
            "revision": save_result.get("revision", current_rev + 1),
        }


# ==============================================================================
# 5. Prompts: Context, Assembly, and Validation (Section 18.7)
# ==============================================================================

def get_prompt_context(project_id: str, scene_id: str, kind: str = "minimax") -> Dict[str, Any]:
    """Get structured prompt context brief for external agent authoring."""
    _folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    if idx < 0:
        raise SceneNotFoundError(scene_id, project_id)

    segment = segments[idx]
    return build_minimax_prompt_context(segment, session)


def assemble_minimax_prompt_endpoint(
    project_id: str,
    scene_id: str,
    shots: List[str],
    mode: Optional[str] = None,
    save: bool = False,
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble official MiniMax H3 prompt from shot descriptions and optionally save."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        segments = session.get("segments", [])
        idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
        if idx < 0:
            raise SceneNotFoundError(scene_id, project_id)

        segment = segments[idx]
        assembled = assemble_minimax_h3_prompt(segment, session, shots, mode=mode)
        validation = validate_minimax_h3_prompt(assembled["prompt"], segment=segment, mode=assembled["mode"])

        revision = current_rev
        if save:
            segment["minimax_h3_prompt"] = assembled["prompt"]
            segment["minimax_h3_prompt_origin"] = "agent"
            save_result = _persist_session(folder, session)
            revision = save_result.get("revision", current_rev + 1)

        return {
            "prompt": assembled["prompt"],
            "characters": assembled["characters"],
            "shots_used": assembled["shots_used"],
            "valid": validation["valid"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "saved": save,
            "revision": revision,
        }


def validate_minimax_prompt_endpoint(
    project_id: str,
    scene_id: str,
    prompt: str,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate MiniMax prompt against rules."""
    _folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])
    idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
    segment = segments[idx] if idx >= 0 else None
    resolved_mode = effective_minimax_mode(segment or {}, session, mode)
    return validate_minimax_h3_prompt(prompt, segment=segment, mode=resolved_mode, fail_on_invalid_prompt_formats=True)


def set_scene_prompt_field_endpoint(
    project_id: str,
    scene_id: str,
    field: str,
    prompt: str,
    origin: str = "agent",
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Directly set a prompt field on a scene."""
    allowed_fields = {
        "t2i_prompt",
        "i2v_prompt",
        "minimax_h3_prompt",
        "minimax_h3_pass2_prompt",
        "flux_prompt",
        "nb_prompt",
        "flow_gpt_prompt",
        "ernie_t2i_prompt",
        "enhance_prompt",
    }
    if field not in allowed_fields:
        raise ValidationError(f"Field '{field}' is not an editable prompt field.")

    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        segments = session.get("segments", [])
        idx = next((i for i, s in enumerate(segments) if s.get("id") == scene_id or str(i + 1) == str(scene_id)), -1)
        if idx < 0:
            raise SceneNotFoundError(scene_id, project_id)

        scene = segments[idx]
        scene[field] = str(prompt or "")
        scene[f"{field}_origin"] = origin

        save_result = _persist_session(folder, session)
        return {
            "scene_id": scene_id,
            "field": field,
            "prompt": scene[field],
            "origin": origin,
            "revision": save_result.get("revision", current_rev + 1),
        }


# ==============================================================================
# 6. Project Lifecycle: Export, Story, Settings & Preflight
# ==============================================================================

def export_project(project_id: str) -> Dict[str, Any]:
    """Export project as a zip archive."""
    folder = resolve_project_folder(project_id)
    res = _prepare_builder_project_export(folder)
    return res


def get_project_story(project_id: str) -> Dict[str, Any]:
    """Get builder story brief, arc, and beats."""
    _folder, session = _get_active_session_and_folder(project_id)
    return session.get("builder_story_layer") or session.get("builderStoryLayer") or session.get("story") or {}


def put_project_story(
    project_id: str,
    story_data: Dict[str, Any],
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Set builder story layer data."""
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        session["builder_story_layer"] = story_data
        session.pop("builderStoryLayer", None)
        save_result = _persist_session(folder, session)
        return {
            "story": story_data,
            "revision": save_result.get("revision", current_rev + 1),
        }


def preflight_project_settings(project_id: str) -> Dict[str, Any]:
    """Preflight check settings against selected video/image modes (Section 16)."""
    _folder, session = _get_active_session_and_folder(project_id)
    effective = extract_effective_settings(session)

    errors = []
    warnings = []

    video_engine = effective.get("project", {}).get("video_engine", "minimax_h3")
    if video_engine == "minimax_h3":
        mm = effective.get("minimax_h3", {})
        if not mm.get("aspect_ratio"):
            errors.append("MiniMax aspect_ratio is required.")
        if mm.get("steps", 0) <= 0:
            errors.append("MiniMax steps must be greater than 0.")
    elif video_engine == "ltx":
        ltx = effective.get("ltx_video", {})
        if ltx.get("fps", 0) <= 0:
            errors.append("LTX fps must be greater than 0.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "effective_settings": effective,
    }


# Settings groups exposed by the API and where the Video Builder keeps them in the session file.
_SETTINGS_GROUP_SESSION_KEYS = {
    "minimax_h3": "minimax_h3_settings",
    "ltx_video": "i2v_video_settings",
    "zimage": "zimage_settings",
    "flux_klein": "flux_klein_settings",
}
_SETTINGS_FLAT_GROUPS = {"project", "llm", "post_process"}


def patch_project_settings(
    project_id: str,
    patch: Dict[str, Any],
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Patch settings on a project session (Section 16)."""
    validate_settings_patch(patch)
    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)

        for key, val in patch.items():
            group_key = _SETTINGS_GROUP_SESSION_KEYS.get(key)
            if key in _SETTINGS_FLAT_GROUPS and isinstance(val, dict):
                # Flat groups live as top-level session keys (e.g. lut_enabled, gemma_context_limit).
                session.update(val)
            elif group_key and isinstance(val, dict):
                # Nested groups live under the key the Video Builder UI saves and reads.
                current = session.get(group_key) if isinstance(session.get(group_key), dict) else {}
                session[group_key] = {**current, **val}
            elif isinstance(val, dict) and isinstance(session.get(key), dict):
                session[key].update(val)
            else:
                session[key] = val

        save_result = _persist_session(folder, session)
        saved_session = save_result.get("session") if isinstance(save_result.get("session"), dict) else session
        return {
            "settings": extract_effective_settings(saved_session),
            "revision": save_result.get("revision", current_rev + 1),
        }


def create_project(name: str, template_from: Optional[str] = None) -> Dict[str, Any]:
    """Create a new project folder and empty session structure."""
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValidationError("Project name cannot be empty.")

    payload = {"project_name": clean_name}
    created = _new_builder_project(payload)
    folder = created["project_folder"]
    pid = get_project_id(folder)

    revision = 1
    if template_from:
        src_folder = resolve_project_folder(template_from)
        src_session_file = os.path.join(src_folder, "vrgdg_builder_session.json")
        if os.path.isfile(src_session_file):
            shutil.copy2(src_session_file, created["session_path"])
    else:
        # Start from the saved model defaults, like a new project in the UI. Saving a session
        # also rewrites those defaults from the keys it contains, so an empty session would
        # replace them with nothing.
        defaults = copy.deepcopy(_load_model_defaults().get("defaults") or {})
        session = {
            **defaults,
            "project_name": clean_name,
            "project_folder": folder,
            "segments": [],
            "video_engine": defaults.get("video_engine") or "minimax_h3",
        }
        saved = _save_builder_session({"project_folder": folder, "project_name": clean_name, "session": session})
        revision = int(saved.get("revision") or 1)

    return {
        "project_id": pid,
        "name": clean_name,
        "project_folder": folder,
        "revision": revision,
    }


def delete_project_by_id(project_id: str, confirm: Optional[str] = None) -> Dict[str, Any]:
    """Delete a project directory after explicit confirmation matching."""
    if not confirm or confirm.strip() != project_id.strip():
        raise ValidationError(f"Confirm parameter must match project_id '{project_id}'.")

    folder = resolve_project_folder(project_id)
    _delete_builder_project({"project_folder": folder})
    return {"deleted": True, "project_id": project_id}


def duplicate_project(project_id: str, new_name: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Branch a project under a new name with optional asset filtering."""
    folder = resolve_project_folder(project_id)
    payload = {
        "source_project_folder": folder,
        "target_project_folder": new_name,
        **(options or {}),
    }
    result = _save_builder_project_as(payload)
    new_folder = result.get("target_project_folder") or result.get("project_folder", "")
    return {
        "project_id": get_project_id(new_folder),
        "project_folder": new_folder,
        "revision": 1,
    }


def validate_project(project_id: str) -> Dict[str, Any]:
    """Run comprehensive validation on a project."""
    folder, session = _get_active_session_and_folder(project_id)
    segments = session.get("segments", [])
    overlays = session.get("overlay_segments", [])
    audio_dur = float(session.get("audio_duration", 0.0) or 0.0)

    issues = validate_timeline_consistency(segments, overlays, audio_dur)

    # Check for mismatched asset numbering (C17 / R10)
    for idx, scene in enumerate(segments):
        expected_num = idx + 1
        img_path = scene.get("approved_image_path")
        if img_path:
            base = os.path.basename(img_path)
            m = _SCENE_ASSET_NAME.match(base)
            if m and int(m.group(2)) != expected_num:
                actual_num = int(m.group(2))
                issues.append({
                    "type": "mismatched_asset_numbering",
                    "scene_id": scene.get("id"),
                    "scene_number": expected_num,
                    "asset_type": "image",
                    "expected_number": expected_num,
                    "actual_number": actual_num,
                    "path": img_path,
                    "message": f"Scene {expected_num} image asset '{base}' has mismatched numbering (expected image_{expected_num:04d}, got {m.group(1)}{actual_num:04d}).",
                })

        vid_path = scene.get("video_path") or scene.get("video_full_path")
        if vid_path:
            base = os.path.basename(vid_path)
            m = _SCENE_ASSET_NAME.match(base)
            if m and int(m.group(2)) != expected_num:
                actual_num = int(m.group(2))
                issues.append({
                    "type": "mismatched_asset_numbering",
                    "scene_id": scene.get("id"),
                    "scene_number": expected_num,
                    "asset_type": "video",
                    "expected_number": expected_num,
                    "actual_number": actual_num,
                    "path": vid_path,
                    "message": f"Scene {expected_num} video asset '{base}' has mismatched numbering (expected video_{expected_num:04d}, got {m.group(1)}{actual_num:04d}).",
                })

    dirty_latents = []
    try:
        dirty_latents = SceneLatentManager.list_dirty(folder)
    except Exception:
        pass

    if dirty_latents:
        issues.append({
            "type": "dirty_latents",
            "message": f"{len(dirty_latents)} scenes have modified prompts or timing without updated latents.",
            "scenes": dirty_latents,
        })

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "scene_count": len(segments),
    }
