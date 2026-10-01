"""Transactional mutation operations for the VRGDG Agent API (Phase A3, Section 15 & 16)."""

import os
import re
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
from ..builder.project import (
    _BUILDER_SAVE_LOCK,
    _SCENE_ASSET_NAME,
    _delete_builder_project,
    _load_builder_session,
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
    parse_bulk_timings,
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
from ..minimax.prompt_assembly import (
    assemble_minimax_h3_prompt,
    build_minimax_prompt_context,
    validate_minimax_h3_prompt,
)

from .errors import (
    ProjectNotFoundError,
    RevisionConflictError,
    SceneNotFoundError,
    SettingsInvalidError,
    ValidationError,
)
from .paths import get_project_id, resolve_project_folder
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


def _persist_session(folder: str, session: Dict[str, Any]) -> Dict[str, Any]:
    """Persist session changes through _save_builder_session with correct payload and revision."""
    current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
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
    return _save_builder_session(payload)


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


def patch_scene(
    project_id: str,
    scene_id: str,
    patch: Dict[str, Any],
    if_match_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Merge-patch scene fields (prompts, timing, overrides) and mark latents dirty on edits."""
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

        prompt_changed = False
        duration_changed = False

        old_t2i = str(scene.get("t2i_prompt") or "")
        old_i2v = str(scene.get("i2v_prompt") or "")
        old_dur = float(scene.get("end", 0.0) or 0.0) - float(scene.get("start", 0.0) or 0.0)

        for key in (
            "t2i_prompt",
            "i2v_prompt",
            "enhance_prompt",
            "minimax_h3_prompt",
            "notes",
            "label",
            "timeline_note",
            "video_path",
            "video_output",
            "video_status",
            "approved_image_path",
            "audio_path",
            "custom_audio_path",
        ):
            if key in patch:
                scene[key] = str(patch[key] or "")

        for key in ("start", "end"):
            if key in patch:
                scene[key] = float(patch[key])

        for key, val in patch.items():
            if key.startswith("use_scene_") or key.endswith("_settings"):
                scene[key] = val

        new_t2i = str(scene.get("t2i_prompt") or "")
        new_i2v = str(scene.get("i2v_prompt") or "")
        new_dur = float(scene.get("end", 0.0) or 0.0) - float(scene.get("start", 0.0) or 0.0)

        if old_t2i != new_t2i or old_i2v != new_i2v:
            prompt_changed = True
        if abs(old_dur - new_dur) > 0.05:
            duration_changed = True

        if prompt_changed or duration_changed:
            try:
                SceneLatentManager.mark_dirty(folder, slot_number)
            except Exception:
                pass

        if "start" in patch or "end" in patch:
            normalize_segments(segments, active_index=idx)

        save_result = _persist_session(folder, session)
        return {
            "scene": scene,
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
                    for k, v in op.get("fields", {}).items():
                        segments[idx][k] = v
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
        beats = session.get("beats") or []
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


def timeline_bulk(
    project_id: str,
    text: str,
    mode: str = "durations",
    action: str = "replace",
    append_start: float = 0.0,
    clear_media: bool = False,
) -> Dict[str, Any]:
    """Apply bulk timings from text (Section 15.4 T16)."""
    parsed = parse_bulk_timings(text, mode=mode)
    if not parsed:
        raise ValidationError("No valid timing lines found in input.")

    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        segments = session.setdefault("segments", [])

        if action == "replace":
            new_segments = []
            for start, end in parsed:
                new_segments.append({
                    "id": _generate_scene_id(),
                    "start": start,
                    "end": end,
                    "source": "manual",
                })
            session["segments"] = new_segments
            session["timing_frozen"] = False
        elif action == "append":
            start_cursor = float(append_start) if append_start > 0 else (float(segments[-1]["end"]) if segments else 0.0)
            for dur_start, dur_end in parsed:
                dur = dur_end - dur_start
                new_segments.append({
                    "id": _generate_scene_id(),
                    "start": round(start_cursor, 4),
                    "end": round(start_cursor + dur, 4),
                    "source": "manual",
                })
                start_cursor += dur
            segments.extend(new_segments)

        renumber_generic_base_scene_labels(session["segments"])
        save_result = _persist_session(folder, session)
        return {
            "scene_count": len(session["segments"]),
            "revision": save_result.get("revision", 1),
        }


# ==============================================================================
# 3. References CRUD & Mapping (Section 6.5, Section 24, T25-T33)
# ==============================================================================

def get_project_references(project_id: str) -> Dict[str, Any]:
    """Get all subjects, locations, and scene mappings for a project."""
    _folder, session = _get_active_session_and_folder(project_id)
    ref_builder = session.get("flux_reference_builder", {})
    return {
        "subjects": ref_builder.get("subjects", []),
        "locations": ref_builder.get("locations", []),
        "scene_mapping": {
            "subjects": session.get("subject_scene_map", {}),
            "locations": session.get("scene_map", {}),
        },
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
        subj = {
            "id": subject_id,
            "name": payload.get("name", "Character"),
            "description": payload.get("description", ""),
            "face_description": payload.get("face_description", ""),
            "reference_type": payload.get("reference_type", "character"),
            "minimax_voice": payload.get("minimax_voice", "none"),
            "trigger_phrase": payload.get("trigger_phrase", ""),
            "trigger_position": payload.get("trigger_position", "start"),
            "extra_reference_for": payload.get("extra_reference_for", ""),
            "extra_reference_note": payload.get("extra_reference_note", ""),
            "image": payload.get("image", {}),
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
            subj_map = session.get("subject_scene_map", {})
            for sid, sub_list in list(subj_map.items()):
                if isinstance(sub_list, list):
                    subj_map[sid] = [s for s in sub_list if s != ref_id]
        else:
            loc_map = session.get("scene_map", {})
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

        if "subjects" in mapping:
            session.setdefault("subject_scene_map", {}).update(mapping["subjects"])
        if "locations" in mapping:
            session.setdefault("scene_map", {}).update(mapping["locations"])

        save_result = _persist_session(folder, session)
        return {
            "scene_mapping": {
                "subjects": session.get("subject_scene_map", {}),
                "locations": session.get("scene_map", {}),
            },
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
            "audio_name": audio_name or "audio.wav",
        }
        res = _save_project_audio(payload)
        session["audio_path"] = res.get("audio_path")
        session["audio_duration"] = res.get("duration", 0.0)

        peaks = _read_audio_peaks(res["audio_path"])
        session["audio_peaks"] = len(peaks)

        save_result = _persist_session(folder, session)
        return {
            "audio_path": res.get("audio_path"),
            "duration": res.get("duration"),
            "peaks_count": len(peaks),
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


def get_audio_beats(project_id: str) -> Dict[str, Any]:
    """Get detected audio beat markers and tempo."""
    _folder, session = _get_active_session_and_folder(project_id)
    return {
        "beats": session.get("beats", []),
        "tempo_bpm": session.get("tempo_bpm") or session.get("bpm"),
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

        session["beats"] = beats
        if tempo_bpm is not None:
            session["tempo_bpm"] = float(tempo_bpm)

        save_result = _persist_session(folder, session)
        return {
            "beat_count": len(beats),
            "tempo_bpm": session.get("tempo_bpm"),
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

        beats = session.get("beats") or []
        offset = float(offset_seconds or 0.0)
        calibrated = []
        for b in beats:
            if isinstance(b, dict):
                nb = dict(b)
                nb["time"] = round(float(nb.get("time", 0.0)) + offset, 4)
                calibrated.append(nb)
            else:
                calibrated.append(round(float(b) + offset, 4))

        session["beats"] = calibrated
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
    return validate_minimax_h3_prompt(prompt, segment=segment, mode=mode, fail_on_invalid_prompt_formats=True)


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
    return session.get("builderStoryLayer") or session.get("story") or {}


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

        session["builderStoryLayer"] = story_data
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
            if isinstance(val, dict) and isinstance(session.get(key), dict):
                session[key].update(val)
            else:
                session[key] = val

        save_result = _persist_session(folder, session)
        return {
            "settings": save_result.get("session", {}).get("settings") or patch,
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

    if template_from:
        src_folder = resolve_project_folder(template_from)
        src_session_file = os.path.join(src_folder, "vrgdg_builder_session.json")
        if os.path.isfile(src_session_file):
            shutil.copy2(src_session_file, created["session_path"])

    return {
        "project_id": pid,
        "name": clean_name,
        "project_folder": folder,
        "revision": 1,
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
