"""Pure-Python MiniMax H3 Prompt Assembly and Validation (Section 18).

Provides:
- Cut plan calculation (storyboard_cut_plan_for_duration)
- Timecode formatting (format_timecode)
- Context construction for external agents (build_minimax_prompt_context)
- Shot description cleanup and official prompt assembly (assemble_minimax_h3_prompt)
- Rule validation (validate_minimax_h3_prompt)
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple


_MAX_MINIMAX_PROMPT_CHARS = 7000
_MAX_MINIMAX_REFERENCE_IMAGES = 9


def format_timecode(seconds: float) -> str:
    """Format seconds into MM:SS.mmm timecode."""
    secs = max(0.0, float(seconds or 0.0))
    minutes = int(secs // 60)
    remainder = secs - (minutes * 60)
    whole_secs = int(remainder)
    millis = int(round((remainder - whole_secs) * 1000))
    if millis >= 1000:
        whole_secs += 1
        millis -= 1000
    if whole_secs >= 60:
        minutes += 1
        whole_secs -= 60
    return f"{minutes:02d}:{whole_secs:02d}.{millis:03d}"


def storyboard_cut_plan_for_duration(duration: float, frequency: int = 0) -> Dict[str, Any]:
    """Calculate shot cuts for a given scene duration and frequency (0 to 10) (Section 18.2)."""
    dur = max(0.1, float(duration or 4.0))
    freq = max(0, min(10, int(frequency or 0)))

    # Maximum 1 cut per second
    maximum_cuts = max(0, math.ceil(dur - 1e-6) - 1)

    if freq <= 0 or maximum_cuts <= 0:
        cut_count = 0
    elif freq >= 10:
        cut_count = maximum_cuts
    else:
        cut_count = min(max(1, maximum_cuts - 1), max(1, int(round(maximum_cuts * freq / 10.0))))

    shot_count = cut_count + 1

    if cut_count == 0:
        cut_times = []
    elif cut_count == maximum_cuts:
        cut_times = [float(i + 1) for i in range(cut_count)]
    else:
        cut_times = [round(dur * (i + 1) / (cut_count + 1), 3) for i in range(cut_count)]

    return {
        "frequency": freq,
        "exact_duration_seconds": round(dur, 3),
        "maximum_one_per_second_cuts": maximum_cuts,
        "cut_count": cut_count,
        "shot_count": shot_count,
        "cut_times_seconds": cut_times,
        "continuous_shot": cut_count == 0,
    }


def normalize_minimax_mode(mode: Optional[str]) -> str:
    """Normalize MiniMax H3 mode string."""
    m = str(mode or "").strip().lower()
    if m in ("image_to_video", "i2v"):
        return "image_to_video"
    if m in ("reference_to_video", "r2v"):
        return "reference_to_video"
    if m in ("image_reference_to_video", "image_plus_reference_to_video", "i2v_r2v"):
        return "image_reference_to_video"
    if m in ("video_to_video", "v2v"):
        return "video_to_video"
    return "text_to_video"


def strip_negative_prompt_phrasing(text: str) -> str:
    """Strip unwanted negative phrasing or disclaimers from LLM shot descriptions."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        l_str = line.strip()
        if re.search(r"\b(no blur|no distortion|high quality|photorealistic|4k|hyperrealistic|cinematic lighting)\b", l_str, re.IGNORECASE):
            continue
        cleaned.append(l_str)
    return " ".join(cleaned).strip()


def normalize_shot_description(text: str) -> str:
    """Normalize whitespace and punctuation on shot description."""
    t = str(text or "").strip()
    # Strip any accidental [Shot N] or At MM:SS prefixes
    t = re.sub(r"^\[Shot\s*\d+\]\s*(?:At\s*\d+:\d+(?:\.\d+)?,\s*)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s+", " ", t)
    t = strip_negative_prompt_phrasing(t)
    if t and not re.search(r"[.!?…]$", t):
        t += "."
    return t


def effective_mode(segment: Dict[str, Any], session: Dict[str, Any], mode: Optional[str] = None) -> str:
    """The scene's MiniMax mode: the caller's, else the scene's, else the project's saved ``video_mode``."""
    settings = session.get("minimax_h3_settings") if isinstance(session.get("minimax_h3_settings"), dict) else {}
    return normalize_minimax_mode(
        mode or segment.get("minimax_h3_mode") or segment.get("video_mode") or settings.get("video_mode") or session.get("video_mode")
    )


def scene_cut_plan(segment: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
    """The scene's shot plan from its length and the project's saved cut frequency."""
    defaults = session.get("builder_storyboard_defaults") if isinstance(session.get("builder_storyboard_defaults"), dict) else {}
    duration = max(0.1, float(segment.get("end", 0.0) or 0.0) - float(segment.get("start", 0.0) or 0.0))
    try:
        frequency = int(float(defaults.get("minimax_h3_cut_frequency") or 0))
    except (TypeError, ValueError):
        frequency = 0
    return storyboard_cut_plan_for_duration(duration, frequency)


def assemble_minimax_h3_prompt(
    segment: Dict[str, Any],
    session: Dict[str, Any],
    descriptions: List[str],
    mode: Optional[str] = None,
    apply_fx: bool = False,
) -> Dict[str, Any]:
    """Assemble official MiniMax prompt text from a list of shot descriptions (Section 18.3, 18.5)."""
    norm_mode = effective_mode(segment, session, mode)
    dur = float(segment.get("end", 0.0) or 0.0) - float(segment.get("start", 0.0) or 0.0)
    dur = max(0.1, dur)

    cut_plan = scene_cut_plan(segment, session)
    expected_shots = cut_plan["shot_count"]

    if norm_mode == "reference_to_video":
        # The saved format is the wrapper plus the shots; the renderer adds the reference definitions.
        from . import shot_prompt

        wanted = len(shot_prompt.shot_plan(cut_plan))
        cleaned = [shot_prompt.normalize_description(shot_prompt.strip_negative_sentences(d)) or shot_prompt.FALLBACK_SHOT for d in descriptions][:wanted]
        cleaned += [shot_prompt.FALLBACK_SHOT] * (wanted - len(cleaned))
        style = str(segment.get("minimax_h3_video_style") or session.get("builder_storyboard_defaults", {}).get("video_style") or "")
        prompt_text = shot_prompt.assemble_prompt(cleaned, cut_plan, style)
        return {"prompt": prompt_text, "characters": len(prompt_text), "shots_used": wanted, "mode": norm_mode, "cut_plan": cut_plan}

    # Clean descriptions
    cleaned_descs = [normalize_shot_description(d) for d in descriptions]
    if len(cleaned_descs) < expected_shots:
        # Pad with fallback descriptions
        while len(cleaned_descs) < expected_shots:
            cleaned_descs.append("A stable cinematic shot with natural camera movement.")
    elif len(cleaned_descs) > expected_shots:
        cleaned_descs = cleaned_descs[:expected_shots]

    # Build shot body
    shots_body_parts = []
    for i, desc in enumerate(cleaned_descs):
        shot_num = i + 1
        if shot_num == 1:
            shots_body_parts.append(f"[Shot 1] {desc}")
        else:
            time_sec = cut_plan["cut_times_seconds"][i - 1] if (i - 1) < len(cut_plan["cut_times_seconds"]) else dur * i / expected_shots
            timecode = format_timecode(time_sec)
            shots_body_parts.append(f"[Shot {shot_num}] At {timecode}, {desc}")

    shots_body = "\n\n".join(shots_body_parts)

    style = str(segment.get("minimax_h3_video_style") or session.get("builder_storyboard_defaults", {}).get("video_style") or "photorealistic")

    if norm_mode in ("text_to_video", "image_to_video"):
        header = f"integrated_multimodal_description:\nGenerate a {round(dur, 1)}-second 16:9 {style} video."
        if norm_mode == "image_to_video":
            header += " <Picture 1> is the exact opening composition, character identity, and visual anchor."
        prompt_text = f"{header}\n\n{shots_body}\n\noverall_soundscape:\nAmbient music video soundstage.\n\nnon_diegetic_music:\nFull synchronized musical performance."
    else:
        # Reference-to-video / Image-reference-to-video / Video-to-video
        subject_defs = [
            "<Subject 1> is the featured performer in <Picture 1>.",
            "<Picture 1> is the first frame of [Shot 1], used as the opening composition and visual anchor.",
            "<Audio 1> is the complete synchronized vocal track.",
        ]
        summary = f"summary:\n[reference generation + audio reuse] The target video is a {style} cinematic scene featuring <Subject 1>."
        retention = [
            "retention_analysis:",
            "<Subject 1> (appears in all shots): fully_preserved - the character identity and performance are retained.",
            "<Picture 1> ([Shot 1] first frame): fully_preserved - the opening framing and lighting are retained.",
            "<Audio 1>: fully_copy - the synchronized music and vocal audio track is fully copied.",
        ]
        detailed = f"detailed_description:\nThe target video is in a {style} music-video style.\n\n{shots_body}"
        prompt_text = "\n\n".join([
            "subject_definitions:\n" + "\n".join(subject_defs),
            summary,
            "\n".join(retention),
            detailed,
        ])

    return {
        "prompt": prompt_text,
        "characters": len(prompt_text),
        "shots_used": len(cleaned_descs),
        "mode": norm_mode,
        "cut_plan": cut_plan,
    }


def validate_minimax_h3_prompt(
    prompt: str,
    segment: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    fail_on_invalid_prompt_formats: bool = False,
) -> Dict[str, Any]:
    """Validate an assembled MiniMax H3 prompt against the specification rules (Section 18.4)."""
    text = str(prompt or "").strip()
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if not text:
        errors.append({"code": "EMPTY_PROMPT", "message": "The MiniMax prompt is empty."})
        return {"valid": False, "errors": errors, "warnings": warnings, "length": 0}

    length = len(text)
    if length > _MAX_MINIMAX_PROMPT_CHARS:
        errors.append({
            "code": "MINIMAX_H3_PROMPT_TOO_LONG",
            "message": f"Prompt length ({length}) exceeds the 7,000 character maximum by {length - _MAX_MINIMAX_PROMPT_CHARS}.",
            "length": length,
            "limit": _MAX_MINIMAX_PROMPT_CHARS,
            "over_by": length - _MAX_MINIMAX_PROMPT_CHARS,
        })

    # Reference capacity check
    pic_matches = set(re.findall(r"<Picture\s+(\d+)>", text, re.IGNORECASE))
    if len(pic_matches) > _MAX_MINIMAX_REFERENCE_IMAGES:
        errors.append({
            "code": "TOO_MANY_REFERENCES",
            "message": f"Prompt references {len(pic_matches)} pictures, exceeding the maximum of {_MAX_MINIMAX_REFERENCE_IMAGES}.",
        })

    norm_mode = normalize_minimax_mode(mode or (segment.get("video_mode") if segment else None))

    if fail_on_invalid_prompt_formats:
        # Check required section headers
        if norm_mode in ("text_to_video", "image_to_video"):
            if "integrated_multimodal_description:" not in text:
                errors.append({
                    "code": "MISSING_HEADER",
                    "message": "Missing 'integrated_multimodal_description:' header in prompt.",
                })
        else:
            # The renderer adds the reference definitions, so a saved prompt holds only this section.
            required_sections = ["detailed_description:"] if norm_mode == "reference_to_video" else [
                "subject_definitions:", "summary:", "retention_analysis:", "detailed_description:"]
            for sec in required_sections:
                if sec not in text:
                    errors.append({
                        "code": "MISSING_SECTION",
                        "message": f"Missing required section '{sec}' in prompt.",
                    })

        # Check shot blocks
        shots_found = re.findall(r"\[Shot\s*(\d+)\]", text, re.IGNORECASE)
        if not shots_found:
            errors.append({"code": "NO_SHOTS_FOUND", "message": "No [Shot N] blocks found in prompt."})
        else:
            shot_nums = [int(n) for n in shots_found]
            if shot_nums[0] != 1:
                warnings.append({"code": "SHOT_ORDER", "message": f"First shot is numbered {shot_nums[0]} instead of 1."})

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "length": length,
        "character_budget_remaining": max(0, _MAX_MINIMAX_PROMPT_CHARS - length),
    }


def build_minimax_prompt_context(
    segment: Dict[str, Any],
    session: Dict[str, Any],
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Build structured context brief for an agent authoring MiniMax shot descriptions (Section 18.6, 18.7)."""
    norm_mode = effective_mode(segment, session, mode)
    dur = float(segment.get("end", 0.0) or 0.0) - float(segment.get("start", 0.0) or 0.0)
    dur = max(0.1, dur)

    cut_plan = scene_cut_plan(segment, session)
    shot_count = cut_plan["shot_count"]

    # Budget calculation
    fixed_overhead = 400 if norm_mode in ("text_to_video", "image_to_video") else (300 if norm_mode == "reference_to_video" else 1200)
    available_chars = max(500, _MAX_MINIMAX_PROMPT_CHARS - fixed_overhead)
    per_shot_chars = available_chars // shot_count

    # Cast & references
    ref_builder = session.get("flux_reference_builder", {})
    subjects = ref_builder.get("subjects", [])
    locations = ref_builder.get("locations", [])

    cast = []
    for idx, s in enumerate(subjects, start=1):
        cast.append({
            "label": f"<Subject {idx}>",
            "name": s.get("name", f"Character {idx}"),
            "role": s.get("auto_build_role") or "lead performer",
            "reference_type": s.get("reference_type", "character"),
        })

    instruction_text = (
        f"Generate {shot_count} cinematic shot descriptions for a {round(dur, 1)}s {norm_mode} scene.\n"
        f"Return JSON: {{\"shots\": [{{\"description\": \"...\"}}]}}.\n"
        f"Target approx {per_shot_chars} characters per shot description."
    )

    return {
        "mode": norm_mode,
        "duration_seconds": round(dur, 2),
        "shot_plan": cut_plan,
        "budget": {
            "hard_limit": _MAX_MINIMAX_PROMPT_CHARS,
            "estimated_overhead_chars": fixed_overhead,
            "available_description_chars": available_chars,
            "per_shot_chars": per_shot_chars,
        },
        "cast": cast,
        "references": {
            "subjects_count": len(subjects),
            "locations_count": len(locations),
        },
        "notes": segment.get("notes", ""),
        "t2i_prompt": segment.get("t2i_prompt", ""),
        "i2v_prompt": segment.get("i2v_prompt", ""),
        "instruction_text": instruction_text,
    }
