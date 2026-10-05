"""Scene-level MiniMax H3 render inputs: reference images, continuity frames, last frame, video references.

Python twin of the browser logic that decides which images and videos a scene render
receives (``minimax_references.mjs``, ``minimax_prompt.mjs`` ``miniMaxRenderReferenceImagePaths``
and ``prepareMiniMaxH3ContinuityReference`` in ``video_render.mjs``). The Agent API
orchestrator uses it so a server-side render sends the same ``image_paths``,
``video_references``, ``last_frame_path`` and ``latent_exact_frame_path`` as the UI.

Reference maps live inside ``session["flux_reference_builder"]`` (``scene_map``,
``subject_scene_map``, ``extra_scene_map``, ``ingredients_scene_map``), exactly where
the Video Builder saves them.

Pure Python. Frame extraction is injected so this module needs no ffmpeg or ComfyUI.
"""

import os
from typing import Any, Callable, Dict, List, Optional

MAX_REFERENCE_IMAGES = 9
_IMAGE_REFERENCE_MODES = ("reference_to_video", "image_reference_to_video")
_REFERENCE_BUILDER_MODES = ("reference_to_video", "image_reference_to_video", "video_to_video")
_CONTINUITY_MODES = ("off", "spatial_reference", "exact_start_frame", "latent_continuation", "latent_continuation_exact_frame", "latent_continuation_masked")
_LATENT_MODES = ("latent_continuation", "latent_continuation_exact_frame", "latent_continuation_masked")


def media_path_key(path: Any) -> str:
    """Comparison key for media paths (mirrors ``mediaPathKey`` in timeline_state.mjs)."""
    text = str(path or "").strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.lower()


def normalize_continuity_mode(value: Any) -> str:
    """Mirror ``normalizeMiniMaxH3ContinuityMode``."""
    clean = "_".join(str(value or "").strip().lower().replace("-", " ").split())
    if clean in ("latent_exact", "latent_exact_frame", "latent_continuation_exact", "latent_continuation_exact_frame"):
        return "latent_continuation_exact_frame"
    if clean in ("latent_masked", "latent_masked_av", "latent_continuation_masked"):
        return "latent_continuation_masked"
    if clean in ("latent", "latent_continuation", "continuation"):
        return "latent_continuation"
    if clean in ("spatial", "spatial_reference", "continuity_reference"):
        return "spatial_reference"
    if clean in ("exact", "exact_start", "exact_start_frame", "continuous_start"):
        return "exact_start_frame"
    return "off"


def _uses_scene_image_as_start_frame(segment: Dict[str, Any]) -> bool:
    """The UI saves both the flag and the choice it derives from ("exact_start_frame")."""
    choice = "_".join(str(segment.get("minimax_h3_scene_image_use") or "").strip().lower().replace("-", " ").split())
    return bool(segment.get("minimax_h3_use_scene_image_as_start_frame")) or choice in ("exact", "exact_start", "exact_start_frame")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _has_image(image: Any) -> bool:
    return isinstance(image, dict) and bool(_text(image.get("path")) or _text(image.get("data")))


def _split_ids(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def scene_map_value(mapping: Any, segment: Dict[str, Any], index: int) -> Any:
    """Mirror ``sceneReferenceMapValue``: look up by scene id, else by 1-based number when keys are numbers."""
    source = mapping if isinstance(mapping, dict) else {}
    scene_id = _text(segment.get("id"))
    if scene_id and scene_id in source:
        return source[scene_id]
    if any(not _text(key).isdigit() for key in source):
        return None
    return source.get(str(index + 1))


def _builder(session: Dict[str, Any]) -> Dict[str, Any]:
    raw = session.get("flux_reference_builder") if isinstance(session, dict) else None
    return raw if isinstance(raw, dict) else {}


def _extra_target_id(subject: Dict[str, Any]) -> str:
    return _text(
        subject.get("extra_reference_for") or subject.get("extraReferenceFor")
        or subject.get("same_subject_as") or subject.get("sameSubjectAs")
    )


def _logical_subjects(refs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in (refs.get("subjects") or []) if isinstance(s, dict) and not _extra_target_id(s)]


def _subject_ids_for_scene(refs: Dict[str, Any], segment: Dict[str, Any], index: int) -> List[str]:
    ids = _split_ids(scene_map_value(refs.get("subject_scene_map"), segment, index))
    subjects = [s for s in (refs.get("subjects") or []) if isinstance(s, dict)]
    logical_ids = {s.get("id") for s in _logical_subjects(refs)}
    mapped: List[str] = []
    for subject_id in ids:
        subject = next((s for s in subjects if s.get("id") == subject_id), None)
        if not subject:
            continue
        target = _extra_target_id(subject) or subject.get("id")
        if target and target in logical_ids and target not in mapped:
            mapped.append(target)
    return mapped


def _expand_subject_references(refs: Dict[str, Any], subjects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = {_text(s.get("id")) for s in subjects if _text(s.get("id"))}
    expanded = list(subjects)
    for subject in refs.get("subjects") or []:
        if isinstance(subject, dict) and _extra_target_id(subject) in selected:
            expanded.append(subject)
    seen = set()
    result = []
    for subject in expanded:
        subject_id = _text(subject.get("id"))
        if subject_id and subject_id not in seen:
            seen.add(subject_id)
            result.append(subject)
    return result


def _primary_subject(refs: Dict[str, Any]) -> Dict[str, Any]:
    subjects = [s for s in (refs.get("subjects") or []) if isinstance(s, dict)]
    primary = refs.get("subject") if isinstance(refs.get("subject"), dict) else {}
    first = subjects[0] if subjects else {}
    return {
        **first,
        "description": primary.get("description") or first.get("description") or "",
        "image": primary.get("image") or first.get("image") or {},
    }


def _has_meaningful_subject(refs: Dict[str, Any]) -> bool:
    """Mirror ``hasMeaningfulSubject`` in normalizeFluxReferenceBuilder.

    The UI switches subject references on by itself once a subject has a name, description,
    trigger or image, so a project built through the API (which never sets the flag) behaves
    like one built in the UI.
    """
    if refs.get("cleared"):
        return False
    primary = refs.get("subject") if isinstance(refs.get("subject"), dict) else {}
    image = primary.get("image") if isinstance(primary.get("image"), dict) else {}
    if _text(primary.get("description")) or any(_text(image.get(key)) for key in ("path", "data", "name")):
        return True
    if refs.get("subject_scene_map"):
        return True
    for subject in refs.get("subjects") or []:
        if not isinstance(subject, dict):
            continue
        name = _text(subject.get("name"))
        placeholder = name.lower().startswith("character ") and name.split(" ", 1)[1].isdigit()
        subject_image = subject.get("image") if isinstance(subject.get("image"), dict) else {}
        if (name and not placeholder) or _text(subject.get("description")) or _text(subject.get("trigger_phrase")) \
                or any(_text(subject_image.get(key)) for key in ("path", "data", "name")):
            return True
    return False


def _subject_items_for_segment(refs: Dict[str, Any], segment: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
    """Mirror ``referenceBuilderSubjectItemsForSegment`` for MiniMax projects."""
    primary_has_image = _has_image((refs.get("subject") or {}).get("image"))
    subjects = [s for s in (refs.get("subjects") or []) if isinstance(s, dict)]
    if not (refs.get("use_subject_reference") or _has_meaningful_subject(refs) or primary_has_image):
        return []
    if segment.get("no_character_present"):
        backed = [s for s in subjects if _has_image(s.get("image"))]
        if backed:
            return [backed[0]]
        return [_primary_subject(refs)] if primary_has_image else []
    subject_count = int(refs.get("subject_count") or len(subjects))
    if subject_count > 1:
        id_set = set(_subject_ids_for_scene(refs, segment, index))
        mapped = [s for s in subjects if s.get("id") in id_set]
        if mapped:
            return _expand_subject_references(refs, mapped)
        backed = [s for s in _logical_subjects(refs) if _has_image(s.get("image"))]
        if len(backed) == 1:
            return backed
        if not backed and primary_has_image:
            return [_primary_subject(refs)]
        return []
    subject = _primary_subject(refs)
    if _has_image(subject.get("image")) or _text(subject.get("name") or subject.get("description")):
        return [subject]
    return []


def reference_catalog(refs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Mirror ``miniMaxReferenceBuilderCatalog``: every reference that has an image."""
    catalog: List[Dict[str, Any]] = []

    def add(kind: str, item: Any, fallback: str) -> None:
        if not isinstance(item, dict):
            return
        item_id = _text(item.get("id"))
        image = item.get("image") if isinstance(item.get("image"), dict) else {}
        if not item_id or not (_text(image.get("path")) or _text(image.get("data"))):
            return
        catalog.append({
            "key": f"{kind}:{item_id}",
            "kind": kind,
            "source_id": item_id,
            "label": _text(item.get("name") or item.get("title") or fallback) or "Reference",
            "description": _text(item.get("description")),
            "reference_image_type": _text(item.get("reference_image_type")) or "single",
            "image": image,
        })

    for subject in refs.get("subjects") or []:
        add("subject", subject, "Character reference")
    for extra in refs.get("extra_subjects") or []:
        if isinstance(extra, dict) and extra.get("send_to_minimax"):
            add("extra", extra, "Extra character reference")
    for location in refs.get("locations") or []:
        add("location", location, "Location reference")
    for sheet in refs.get("ingredients_sheets") or []:
        add("ingredients", sheet, "Ingredients sheet")
    return catalog


def _forced_extra_keys(refs: Dict[str, Any], segment: Dict[str, Any], index: int) -> List[str]:
    """Mirror ``miniMaxForcedExtraReferenceKeysForSegment`` (extras mapped to this scene that go to MiniMax)."""
    if segment.get("no_character_present") or not refs.get("extras_enabled"):
        return []
    extras = {_text(e.get("id")): e for e in (refs.get("extra_subjects") or []) if isinstance(e, dict)}
    entries = scene_map_value(refs.get("extra_scene_map"), segment, index)
    keys: List[str] = []
    for entry in entries if isinstance(entries, list) else []:
        extra = extras.get(_text((entry or {}).get("extra_id") or (entry or {}).get("extraId")))
        if extra and _text(extra.get("description")) and extra.get("send_to_minimax") and _has_image(extra.get("image")):
            key = f"extra:{_text(extra.get('id'))}"
            if key not in keys:
                keys.append(key)
    return keys


def _mapped_keys(refs: Dict[str, Any], segment: Dict[str, Any], index: int, catalog_keys: set) -> List[str]:
    keys: List[str] = []

    def add(key: str) -> None:
        if key and key in catalog_keys and key not in keys:
            keys.append(key)

    if not segment.get("no_character_present"):
        for subject in _subject_items_for_segment(refs, segment, index):
            add(f"subject:{_text(subject.get('id'))}")
        for key in _forced_extra_keys(refs, segment, index):
            add(key)
    location_id = _text(scene_map_value(refs.get("scene_map"), segment, index))
    if location_id:
        add(f"location:{location_id}")
    sheet_id = _text((refs.get("ingredients_scene_map") or {}).get(segment.get("id")))
    if sheet_id:
        add(f"ingredients:{sheet_id}")
    return keys


def desired_reference_keys(refs: Dict[str, Any], segment: Dict[str, Any], index: int) -> List[str]:
    """Mirror ``miniMaxReferenceKeysForSegment``: explicit keys else mapped keys, plus forced extras, max 9."""
    catalog_keys = {item["key"] for item in reference_catalog(refs)}
    explicit = segment.get("minimax_h3_reference_keys")
    selected = [_text(k) for k in explicit] if isinstance(explicit, list) else _mapped_keys(refs, segment, index, catalog_keys)
    forced = _forced_extra_keys(refs, segment, index)
    seen: set = set()
    keys: List[str] = []
    for key in [*selected, *forced]:
        if not key or key not in catalog_keys or (key.startswith("extra:") and key not in forced) or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys[:MAX_REFERENCE_IMAGES]


def segment_image_path(segment: Dict[str, Any]) -> str:
    """Mirror ``selectedSegmentImagePath``: the selected history image, else approved, else custom."""
    history = segment.get("image_history") if isinstance(segment.get("image_history"), list) else []
    if history:
        position = max(0, min(len(history) - 1, int(segment.get("image_history_index") or 0)))
        if history[position]:
            return _text(history[position])
    return _text(segment.get("approved_image_path") or segment.get("custom_image_path"))


def ordered_reference_items(session: Dict[str, Any], segment: Dict[str, Any], mode: str, index: int) -> List[Dict[str, Any]]:
    """Mirror ``miniMaxOrderedImageReferenceItemsForSegment``: start frame first, then mapped references, max 9."""
    refs = _builder(session)
    ordered: List[Dict[str, Any]] = []
    if mode in _IMAGE_REFERENCE_MODES and _uses_scene_image_as_start_frame(segment):
        start = segment_image_path(segment)
        if start:
            ordered.append({"key": "scene:start_frame", "kind": "start_frame", "label": "Scene start frame", "image": {"path": start}})
    by_key = {item["key"]: item for item in reference_catalog(refs)}
    ordered.extend(by_key[key] for key in desired_reference_keys(refs, segment, index) if key in by_key)
    seen: set = set()
    result = []
    for item in ordered:
        path = _text((item.get("image") or {}).get("path"))
        fingerprint = f"path:{media_path_key(path)}" if path else ""
        if fingerprint and fingerprint not in seen:
            seen.add(fingerprint)
            result.append(item)
    return result[:MAX_REFERENCE_IMAGES]


def render_reference_image_paths(
    session: Dict[str, Any],
    segment: Dict[str, Any],
    mode: str,
    index: int,
    configured: Optional[List[Any]] = None,
) -> List[str]:
    """Mirror ``miniMaxRenderReferenceImagePaths``."""
    def normalize(value: Any) -> List[str]:
        out = []
        for item in value if isinstance(value, list) else []:
            path = _text(item.get("path") or item.get("file")) if isinstance(item, dict) else _text(item)
            if path:
                out.append(path)
        return out

    uses_builder = mode in _REFERENCE_BUILDER_MODES
    builder_paths = []
    if uses_builder and configured is None:
        builder_paths = [
            _text((item.get("image") or {}).get("path"))
            for item in ordered_reference_items(session, segment, mode, index)
            if _text((item.get("image") or {}).get("path"))
        ][:MAX_REFERENCE_IMAGES]
    extra_paths = normalize(
        configured if configured is not None
        else (segment.get("minimax_h3_image_paths") or segment.get("minimax_image_paths") or [])
    )
    paths: List[str] = []
    if uses_builder:
        seen: set = set()
        for path in [*builder_paths, *extra_paths]:
            key = media_path_key(path)
            if key and key not in seen:
                seen.add(key)
                paths.append(path)
        paths = paths[:MAX_REFERENCE_IMAGES]
    if mode in ("image_to_video", "image_reference_to_video"):
        selected = segment_image_path(segment)
        if selected:
            paths = [selected, *[p for p in paths if media_path_key(p) != media_path_key(selected)]][:MAX_REFERENCE_IMAGES]
    return paths


def video_references_for_scene(segment: Dict[str, Any], mode: str, configured: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Reference videos, only for video_to_video (mirrors the browser)."""
    raw = configured if configured is not None else (
        segment.get("minimax_h3_video_references") or segment.get("minimax_video_references") or []
    )
    return list(raw) if mode == "video_to_video" and isinstance(raw, list) else []


def last_frame_path_for_scene(segment: Dict[str, Any], mode: str) -> str:
    """Image to Video can end on an explicit last frame (``first_last_frame_end_image_path``)."""
    return _text(segment.get("first_last_frame_end_image_path")) if mode == "image_to_video" else ""


def resolve_scene_inputs(
    session: Dict[str, Any],
    segment: Dict[str, Any],
    mode: str,
    scene_index: int,
    *,
    continuity_mode: str,
    previous_segment: Optional[Dict[str, Any]],
    project_folder: str,
    scene_number: int,
    extract_final_frame: Callable[[str, str, int], str],
    configured_image_paths: Optional[List[Any]] = None,
    configured_video_references: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Everything scene-specific a MiniMax render needs besides settings, prompt and timing.

    ``extract_final_frame(project_folder, video_path, scene_number)`` must return the saved
    frame path. Raises ``ValueError`` with the browser's wording when inputs are missing.
    """
    continuity = normalize_continuity_mode(continuity_mode) if mode in ("reference_to_video", "video_to_video") else "off"
    if continuity == "exact_start_frame" and _uses_scene_image_as_start_frame(segment):
        raise ValueError("A scene cannot use both its scene image and the previous rendered final frame as the exact start frame.")

    image_paths = render_reference_image_paths(session, segment, mode, scene_index, configured_image_paths)
    result: Dict[str, Any] = {"continuity_mode": continuity, "latent_exact_frame_path": "", "continuity_image_number": 0}

    if continuity != "off" and previous_segment is not None:
        if continuity in _LATENT_MODES:
            if scene_number <= 1:
                raise ValueError("Scene 1 cannot use Latent Continuation because there is no predecessor scene. Set continuity to off.")
            if continuity == "latent_continuation_exact_frame":
                video = _text(previous_segment.get("video_path") or previous_segment.get("rendered_video_path"))
                if not video:
                    raise ValueError(
                        f"Latent Continuation + Exact Last Frame needs Scene {scene_number - 1}'s rendered video to read its "
                        f"last frame, but it has none. Render Scene {scene_number - 1} first."
                    )
                frame = extract_final_frame(project_folder, video, scene_number)
                if not frame:
                    raise ValueError("Could not extract the previous scene's last frame for exact-frame continuity.")
                result["latent_exact_frame_path"] = frame
        else:
            video = _text(previous_segment.get("video_path") or previous_segment.get("rendered_video_path"))
            if video:
                frame = extract_final_frame(project_folder, video, scene_number)
                if not frame:
                    raise ValueError("MiniMax continuity final-frame extraction did not return an image path.")
                key = media_path_key(frame)
                if not any(media_path_key(p) == key for p in image_paths):
                    if len(image_paths) >= MAX_REFERENCE_IMAGES:
                        raise ValueError(
                            "This scene has nine MiniMax image references already. Remove one Reference Builder image so the "
                            "previous-scene continuity frame can use the reserved ninth slot."
                        )
                    image_paths.append(frame)
                result["continuity_image_number"] = next(i for i, p in enumerate(image_paths) if media_path_key(p) == key) + 1

    videos = video_references_for_scene(segment, mode, configured_video_references)
    if mode in ("image_to_video", "image_reference_to_video") and not image_paths:
        raise ValueError("This scene needs a selected scene image for MiniMax Image to Video.")
    if mode == "reference_to_video" and not image_paths:
        raise ValueError("This scene needs at least one ordered Reference Builder image.")
    if mode == "video_to_video" and not any(_text((v or {}).get("path")) for v in videos if isinstance(v, dict)):
        raise ValueError("This scene needs at least one reference video path.")

    missing = [p for p in image_paths if not os.path.isfile(p)]
    result["missing_image_paths"] = missing
    result["image_paths"] = image_paths
    result["video_references"] = videos
    last_frame = last_frame_path_for_scene(segment, mode)
    if last_frame:
        result["last_frame_path"] = last_frame
    return result
