"""MiniMax H3 video prompts for agents: the Video Builder's "create MiniMax prompts" for reference-to-video scenes.

The project's LLM writes the creative shot descriptions with the saved ``minimax_h3_reference_to_video``
instruction. The builder side (shot labels, cut times, the ``detailed_description`` wrapper and the format
checks, and the reference definitions and soundscape around the shots) is ``minimax.shot_prompt``. For LM Studio the model that is already loaded is used and never changed
(see ``llm_runtime``). Prompts are saved on each scene as ``minimax_h3_prompt``.
"""

from typing import Any, Callable, Dict, List, Optional

from ...builder.lyric_scenes import is_instrumental_lyric_text
from ...llm import video_prompt_generation as vid_gen
from ...minimax import shot_prompt as sp
from ...minimax.prompt_assembly import storyboard_cut_plan_for_duration
from ...minimax.scene_inputs import ordered_reference_items
from ..errors import ValidationError
from ..jobs.manager import JobManager, get_job_manager
from ..jobs.models import Job
from ..llm_runtime import llm_payload_from_session, prepare_llm_payload
from ..mutations import _BUILDER_SAVE_LOCK, _get_active_session_and_folder, _persist_session
from .storyboard_orchestrator import scene_cards, sync_storyboard_files

MODE = "reference_to_video"
INSTRUCTION_KEY = "minimax_h3_reference_to_video"
MAX_ATTEMPTS = 3


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _scene_context(session: Dict[str, Any], segment: Dict[str, Any], card: Dict[str, Any], index: int) -> Dict[str, Any]:
    defaults = session.get("builder_storyboard_defaults") if isinstance(session.get("builder_storyboard_defaults"), dict) else {}
    settings = session.get("minimax_h3_settings") if isinstance(session.get("minimax_h3_settings"), dict) else {}
    start, end = _number(segment.get("start"), 0.0), _number(segment.get("end"), 0.0)
    duration = max(0.1, end - start)
    items = ordered_reference_items(session, segment, MODE, index)
    lyric = _text(segment.get("lyric_text"))
    no_character = bool(segment.get("no_character_present"))
    visual_only = bool(segment.get("lyric_no_lip_sync")) or no_character or is_instrumental_lyric_text(lyric)
    return {
        "items": items,
        "duration": duration,
        "cut_plan": storyboard_cut_plan_for_duration(duration, int(_number(defaults.get("minimax_h3_cut_frequency"), 0))),
        "style": _text(segment.get("minimax_h3_video_style") or defaults.get("video_style")),
        "aspect_ratio": _text(settings.get("aspect_ratio")) or "16:9",
        "audio_mode": _text(settings.get("audio_mode")) or "input_audio",
        "camera_speed": _number(segment.get("camera_motion_speed", defaults.get("camera_motion_speed")), 4.0),
        "character_speed": _number(segment.get("character_motion_speed", defaults.get("character_motion_speed")), 4.0),
        "camera_guidance": _text(segment.get("camera_motion_speed_guidance") or defaults.get("camera_guidance")),
        "character_guidance": _text(segment.get("character_motion_guidance") or defaults.get("character_guidance")),
        "lyric": "" if visual_only else " ".join(lyric.split()),
        "visual_only": visual_only,
        "no_character": no_character,
        "singers": [] if visual_only else list(card.get("lyric_singers") or []),
        "subject_text": "\n".join(f"{r['name']}: {r['description']}" if r["description"] else r["name"] for r in card.get("subject_refs") or []),
        "location_text": "\n".join(filter(None, [
            _text((card.get("location_ref") or {}).get("name")), _text((card.get("location_ref") or {}).get("description"))])),
    }


def generate_scene_prompt(session: Dict[str, Any], folder: str, segment: Dict[str, Any], card: Dict[str, Any], index: int,
                          params: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the LLM for the shot descriptions of one scene and assemble the saved prompt. Retries when too long."""
    ctx = _scene_context(session, segment, card, index)
    if not ctx["items"]:
        raise ValidationError(f"{card['label']} has no mapped Reference Builder image. Map a character to the scene first.")
    labels = sp.reference_labels(ctx["items"])
    plan = ctx["cut_plan"]
    # The saved prompt is the Builder's full format: definitions before the shots, soundscape after. Both count
    # toward the 7,000 characters, so the shots get what is left.
    frame = sp.reference_frame(ctx["items"], plan, ctx["style"], ctx["audio_mode"], _text(segment.get("audio_direction")))
    target_limit = sp.HARD_LIMIT - len(frame["head"]) - len(frame["tail"])
    last_length = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        budget = sp.character_budget(plan, ctx["style"], target_limit)
        if budget["fixed_chars"] >= budget["hard_limit"]:
            raise ValidationError("The fixed MiniMax H3 format already exceeds 7,000 characters.")
        task = sp.build_shot_task(
            mode_label="Reference to Video", duration=ctx["duration"], aspect_ratio=ctx["aspect_ratio"], audio_mode=ctx["audio_mode"],
            cut_plan=plan, style=ctx["style"], labels=labels, camera_speed=ctx["camera_speed"], character_speed=ctx["character_speed"],
            camera_guidance=ctx["camera_guidance"], character_guidance=ctx["character_guidance"], lyric_text=ctx["lyric"],
            visual_only=ctx["visual_only"], no_character=ctx["no_character"], story_beat=card.get("story_beat", ""),
            lyric_section=card.get("lyric_section", ""), scene_notes=_text(segment.get("notes") or segment.get("director_note")),
            subject_text=ctx["subject_text"], location_text=ctx["location_text"], seed=_text(segment.get("id") or card.get("label")),
            target_limit=target_limit,
        )
        request = prepare_llm_payload({
            **llm_payload_from_session(session),
            "project_folder": folder,
            "scene_id": segment.get("id") or "",
            "builder_instruction_key": INSTRUCTION_KEY,
            "t2i_prompt": task,
            "user_notes": "",
            "subject_context": "",
            "location_context": "",
            "no_character_present": ctx["no_character"],
            "performance_mode": "no_lip_sync" if ctx["visual_only"] else "singing",
            "lyric_text": ctx["lyric"],
            "singers": ctx["singers"],
            "audio_mode": ctx["audio_mode"],
            "camera_motion_speed": ctx["camera_speed"],
            "character_motion_speed": ctx["character_speed"],
            "unload_after": False,
            "temperature": _number(params.get("temperature"), 0.45),
            "top_p": _number(params.get("top_p"), 0.92),
            "max_new_tokens": min(int(_number(params.get("max_new_tokens"), 4000)), max(600, int(budget["shot_chars"] / 2.5) + 250)),
        })
        result = vid_gen._generate_builder_t2v_prompt(request)
        raw = _text(result.get("prompt") if isinstance(result, dict) else result)
        descriptions = sp.parse_shot_descriptions(raw, budget["shot_count"])
        if ctx["lyric"] and ctx["audio_mode"] != "built_in_audio" and labels:
            # The prompt must say what the character sings: the lyric goes in the shot in double quotes.
            cast = [l for l in labels if l["kind"] == "subject"] or labels
            performer = f"{cast[0]['label']} ({cast[0]['name']})" if cast[0]["name"] else cast[0]["label"]
            descriptions = sp.ensure_quoted_lyrics(descriptions, segment.get("lyric_text") or ctx["lyric"], performer)
        core = sp.assemble_prompt(descriptions, plan, ctx["style"])
        prompt = sp.wrap_reference_prompt(core, frame)
        try:
            sp.validate_prompt(core, plan)
            if len(prompt) > sp.HARD_LIMIT:
                raise sp.ShotPromptError(
                    f"The MiniMax H3 prompt is {len(prompt)} characters, over the {sp.HARD_LIMIT} maximum by {len(prompt) - sp.HARD_LIMIT}.",
                    "MINIMAX_H3_PROMPT_TOO_LONG", len(prompt))
        except sp.ShotPromptError as exc:
            if exc.code != "MINIMAX_H3_PROMPT_TOO_LONG" or attempt >= MAX_ATTEMPTS:
                raise
            last_length = exc.length
            target_limit = max(budget["fixed_chars"], target_limit - max(350, exc.length - sp.HARD_LIMIT + 250))
            continue
        return {"prompt": prompt, "characters": len(prompt), "shots": budget["shot_count"], "attempts": attempt,
                "model": request.get("lmstudio_model") or request.get("model_file")}
    raise ValidationError(f"Could not fit the prompt in {sp.HARD_LIMIT} characters after {MAX_ATTEMPTS} tries (last {last_length}).")


def create_minimax_prompts(
    project_id: str,
    params: Optional[Dict[str, Any]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Write a MiniMax prompt for each scene that has none, saving after every scene.

    ``replace_existing`` rewrites all, ``scene_ids`` or ``limit`` narrow the set. A scene whose prompt
    fails is reported in ``failures`` and the rest continue.
    """
    params = dict(params or {})
    folder, session = _get_active_session_and_folder(project_id)
    cards = scene_cards(session)
    if not cards:
        raise ValidationError("The project has no scenes. Create the timeline first.")
    wanted = {str(x) for x in (params.get("scene_ids") or [])}
    replace = bool(params.get("replace_existing"))
    segments = {s.get("id"): (i, s) for i, s in enumerate(sorted((x for x in session.get("segments") or [] if isinstance(x, dict)),
                                                                   key=lambda x: _number(x.get("start"), 0.0)))}
    targets = [
        c for c in cards
        if (not wanted or str(c["id"]) in wanted or str(c["scene_number"]) in wanted)
        and (replace or wanted or not _text(segments[c["id"]][1].get("minimax_h3_prompt")))
    ]
    limit = int(params.get("limit") or 0)
    if limit > 0:
        targets = targets[:limit]
    created: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    revision = session.get("revision")
    for done, card in enumerate(targets):
        if progress:
            progress(done, len(targets), f"Scene {card['scene_number']}")
        index, segment = segments[card["id"]]
        try:
            result = generate_scene_prompt(session, folder, segment, card, index, params)
        except (ValidationError, sp.ShotPromptError) as exc:
            failures.append({"scene_id": card["id"], "scene_number": card["scene_number"], "error": str(exc)})
            continue
        with _BUILDER_SAVE_LOCK:
            folder, session = _get_active_session_and_folder(project_id)
            for stored in session.get("segments") or []:
                if isinstance(stored, dict) and stored.get("id") == card["id"]:
                    stored["minimax_h3_prompt"] = result["prompt"]
                    stored["minimax_h3_prompt_origin"] = "gemma"
                    stored["minimax_h3_mode"] = MODE
                    stored["video_prompt_type"] = "rtv"  # what the Video Builder saves for reference-to-video scenes
            revision = _persist_session(folder, session).get("revision")
        created.append({"scene_id": card["id"], "scene_number": card["scene_number"], "characters": result["characters"],
                        "shots": result["shots"], "attempts": result["attempts"]})
    if progress:
        progress(len(targets), len(targets), "done")
    # The Storyboard Builder reads its own saved copy (storyboard/storyboard.json) and the exported prompt files.
    synced = sync_storyboard_files(project_id) if created else {}
    return {"created": len(created), "failed": len(failures), "skipped": len(cards) - len(targets), "prompts": created,
            "failures": failures, "storyboard": synced, "revision": revision}


def run_minimax_prompts_job(job: Job, manager: JobManager) -> Dict[str, Any]:
    def report(done: int, total: int, label: str) -> None:
        manager.update_progress(job.id, 5.0 + 90.0 * done / max(1, total), "minimax_prompts", message=f"MiniMax prompts {done}/{total} ({label})")
    return create_minimax_prompts(job.project_id, job.params, progress=report)


def register_minimax_prompt_orchestrator_handlers(manager: Optional[JobManager] = None) -> None:
    manager = manager or get_job_manager()
    manager.register_handler("minimax.prompts", run_minimax_prompts_job)
