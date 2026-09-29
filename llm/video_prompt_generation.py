"""Video Builder video prompt generation: T2V, I2V, chained I2V, motion notes, MiniMax H3 formatting, and video prompt edits."""

import json
import os
import re
import time
from PIL import Image
from ..core.atomic_write import atomic_write_json
from ..builder.video_editor import _image_from_data_url
from .text_cleaning import _clean_gemma_prompt_text
from .prompts.video import _I2V_INSTRUCTIONS, _T2V_INSTRUCTIONS

from .prompts.video import _video_prompt_edit_instructions, _video_prompt_enhancement_instructions
from ..builder.paths import _read_text_file, _resolve_existing_file, _safe_project_name
from .output_checks import _extract_json_object_from_text, _looks_like_gemma_repeat_failure, _looks_like_unfilled_prompt_template
from ..builder.media import _image_from_prompt_payload
from .builder_instructions import _BUILDER_INSTRUCTION_LABELS, _effective_builder_instruction, _safe_builder_instruction_key
from .builder_runner import _EXTERNAL_LLM_RUNNERS, _builder_local_llm, _builder_local_mmproj_file, _builder_local_model_file, _clean_lm_studio_plain_text, _llm_runner_from_payload, _repair_and_validate_builder_gemma_prompt, _resolve_mmproj_dropdown_path, _run_builder_text_llm, _runner_output_token_limit, _try_run_remote_vision


def _format_minimax_h3_prompt(text, payload=None, instruction_key=""):
    """Keep H3 prompts readable and enforce the selected MiniMax audio contract."""
    payload = payload if isinstance(payload, dict) else {}
    cleaned = _clean_lm_studio_plain_text(text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"<\s*Picture\s+(\d+)\s*>", r"Image \1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*Video\s+(\d+)\s*>", r"Video \1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*Audio\s+1\s*>", "Audio 1", cleaned, flags=re.IGNORECASE)

    raw_audio_mode = str(payload.get("audio_mode") or payload.get("minimax_h3_audio_mode") or "input_audio").strip().lower().replace("-", "_").replace(" ", "_")
    native_audio = raw_audio_mode in {"built_in_audio", "native_audio", "generated_audio"}
    speaker_assignments_raw = payload.get("speaker_assignments") or payload.get("minimax_speaker_assignments") or payload.get("dialogue_cues") or []
    if isinstance(speaker_assignments_raw, str):
        try:
            speaker_assignments_raw = json.loads(speaker_assignments_raw)
        except Exception:
            speaker_assignments_raw = []
    speaker_assignments = []
    if isinstance(speaker_assignments_raw, list):
        for item in speaker_assignments_raw:
            if not isinstance(item, dict):
                continue
            speaker_name = str(item.get("speaker_name") or item.get("speaker") or item.get("name") or "").strip()
            cue_text = str(item.get("text") or item.get("dialogue") or item.get("line") or "").strip()
            if cue_text:
                speaker_assignments.append({"speaker_name": speaker_name or "The assigned speaker", "text": cue_text})

    timestamp_pattern = r"\[\s*\d+(?:\.\d+)?s?\s*[-\u2013\u2014]\s*\d+(?:\.\d+)?s?\s*\]"
    section_pattern = rf"(?:Image\s+\d+|Video\s+\d+|Audio\s+1|Native\s+audio|Audio|Continuity)\s*:|{timestamp_pattern}"
    cleaned = re.sub(rf"[ \t]*(?={section_pattern})", "\n\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(rf"({timestamp_pattern})[ \t]*(?:\n[ \t]*)?", r"\1\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    lyric_text = re.sub(r"\s+", " ", str(payload.get("lyric_text") or "")).strip().strip('"\'\u201c\u201d\u2018\u2019')
    performance_mode = str(payload.get("performance_mode") or "singing").strip().lower().replace("-", "_").replace(" ", "_")
    no_visible_character = bool(payload.get("no_character_present"))
    no_lip_sync = no_visible_character or performance_mode in {"no_lip_sync", "nolipsync", "no_lipsync", "visual_only", "instrumental"}
    singers_raw = payload.get("singers") or []
    if isinstance(singers_raw, str):
        singer_names = [item.strip() for item in re.split(r"[,;\n]+", singers_raw) if item.strip()]
    elif isinstance(singers_raw, list):
        singer_names = [str(item or "").strip() for item in singers_raw if str(item or "").strip()]
    else:
        singer_names = []
    performer = " and ".join(singer_names[:2]) if singer_names else "the visible performer"
    ordered_native_dialogue = bool(native_audio and performance_mode == "speaking" and speaker_assignments)

    def insert_before_first_timeline(block):
        nonlocal cleaned
        match = re.search(timestamp_pattern, cleaned)
        if match:
            cleaned = f"{cleaned[:match.start()].rstrip()}\n\n{block}\n\n{cleaned[match.start():].lstrip()}"
        else:
            cleaned = f"{cleaned.rstrip()}\n\n{block}"

    def insert_before_continuity(block):
        nonlocal cleaned
        match = re.search(r"(?im)^Continuity\s*:", cleaned)
        if match:
            cleaned = f"{cleaned[:match.start()].rstrip()}\n\n{block}\n\n{cleaned[match.start():].lstrip()}"
        else:
            cleaned = f"{cleaned.rstrip()}\n\n{block}"

    def enforce_visual_only_timeline():
        """Remove lyric-synchronised vocal actions from H3 timestamps and add a final B-roll contract."""
        nonlocal cleaned
        positive_vocal = re.compile(
            r"\b(?:sing(?:s|ing)?|sang|sung|rap(?:s|ping)?|lip[ -]?sync(?:s|ing)?|"
            r"speak(?:s|ing)?|say(?:s|ing)?|said|mouth(?:s|ed|ing)?|whisper(?:s|ing)?|"
            r"perform(?:s|ing)?\s+(?:the\s+)?(?:saved\s+|exact\s+)?(?:lyric|dialogue|words?))\b",
            flags=re.IGNORECASE,
        )
        negative_safety = re.compile(
            r"\b(?:no|not|never|without|does\s+not|do\s+not|must\s+not|cannot|can['’]t|don['’]t)\b",
            flags=re.IGNORECASE,
        )
        timeline_block_pattern = re.compile(
            rf"({timestamp_pattern}\n)(.*?)(?=\n\n(?:{timestamp_pattern}|Audio\s*:|Continuity\s*:)|$)",
            flags=re.IGNORECASE | re.DOTALL,
        )

        def clean_timeline_block(match):
            parts = [
                part.strip()
                for part in re.split(r"(?<=[.!?])\s+|\n+", match.group(2).strip())
                if part.strip()
            ]
            kept = [
                part for part in parts
                if not (positive_vocal.search(part) and not negative_safety.search(part))
            ]
            if not kept:
                kept.append("Continue the requested visual action, camera movement, and environmental motion naturally.")
            kept.append(
                "No visible subject sings, raps, mouths the saved lyric, or lip-syncs; any visible conversation remains natural and unsynchronised to the saved song."
            )
            return f"{match.group(1)}{' '.join(kept)}"

        cleaned = timeline_block_pattern.sub(clean_timeline_block, cleaned)
        safety_block = (
            "VISUAL-ONLY B-ROLL / NO-LIP-SYNC SAFETY — MANDATORY AND FINAL:\n"
            "This scene is explicitly B-roll / no-lip-sync. This safety block overrides any conflicting lyric, singer, "
            "song-performance, or lyric-sync instruction elsewhere in the prompt. It does not remove visible characters "
            "or prohibit natural non-lyric interaction.\n"
            "No visible subject sings, raps, lip-syncs, or mouths the saved lyric. Characters may appear, act, interact, "
            "and have natural unsynchronised conversation—for example, band members backstage talking—but no visible mouth "
            "may synchronise to the saved song.\n"
            + (
                "Built-in MiniMax Audio: generate only requested environmental ambience and sound effects. Do not "
                "generate speech, singing, dialogue, lyrics, narration, voices, or vocal layers."
                if native_audio else
                "Input Audio: preserve Audio 1 completely unchanged as the soundtrack and timing reference, including "
                "any audible vocals, but treat those vocals as off-screen soundtrack only. No visible subject may synchronize to them."
            )
            + "\nTreat the saved lyric only as hidden mood/story context; never quote it as dialogue or describe anyone performing it."
        )
        insert_before_first_timeline(safety_block)

    if instruction_key.endswith("image_to_video") and not re.search(r"(?im)^Image\s+1\s*:", cleaned):
        image_assignment = (
            "Image 1: use as the exact start frame, character appearance and clothing reference, "
            "environment reference, lighting reference, and composition reference."
        )
        first_audio = re.search(r"(?im)^Audio\s+1\s*:", cleaned)
        if first_audio:
            cleaned = f"{cleaned[:first_audio.start()].rstrip()}\n\n{image_assignment}\n\n{cleaned[first_audio.start():].lstrip()}"
        else:
            insert_before_first_timeline(image_assignment)

    if native_audio:
        native_assignment_pattern = re.compile(
            rf"(?ims)^(?:Audio\s+1|Native\s+audio)\s*:.*?(?=\n\n(?:{timestamp_pattern}|Image\s+\d+\s*:|Video\s+\d+\s*:|Audio\s*:|Continuity\s*:)|\Z)"
        )
        cleaned = native_assignment_pattern.sub("", cleaned).strip()
        if ordered_native_dialogue:
            ordered_cues = "; ".join(
                f'{cue["speaker_name"]} says exactly “{cue["text"]}”'
                for cue in speaker_assignments
            )
            native_assignment = (
                f"Native audio: Generate spoken dialogue in this exact order: {ordered_cues}. "
                "Only the assigned speaker speaks during each cue. Every other visible character remains silent with their mouth closed and reacts naturally. "
                "Synchronize each assigned speaker’s lips, mouth shapes, jaw movement, facial muscles, and breathing precisely to that speaker’s generated words. "
                "Speak only in the same language used by the exact supplied dialogue; do not translate or switch languages. "
                "Pronounce every supplied word clearly at a natural measured pace. Do not generate gibberish, babble, invented syllables, filler, phonetic substitutions, repeated phrases, or extra vocalizations. "
                "Begin the first cue within the first 0.15 seconds, pace the exact words naturally across the available clip, and land the final word approximately 0.15-0.30 seconds before the clip ends. Never finish early and fill time with invented speech. "
                "Do not merge speakers, overlap or reorder cues, transfer words, rewrite, restart, repeat, improvise, omit, or add dialogue."
            )
        elif lyric_text and not no_lip_sync:
            action = "speaking" if performance_mode == "speaking" else "singing"
            native_assignment = (
                f"Native audio: Generate {performer} {action} the exact line “{lyric_text}”. "
                "Synchronize the visible performance precisely to the generated words and preserve the exact wording without additions, repetition, or replacement."
            )
        elif no_lip_sync:
            native_assignment = (
                "Native audio: Generate only the requested environmental ambience and sound effects. "
                "Do not generate speech, singing, lyrics, dialogue, voices, or visible mouth synchronization."
            )
        else:
            native_assignment = (
                "Native audio: Generate only appropriate environmental ambience and requested sound effects. "
                "Do not invent speech, singing, lyrics, dialogue, or voices."
            )
        insert_before_first_timeline(native_assignment)
    else:
        if lyric_text and not no_lip_sync:
            action = "says" if performance_mode == "speaking" else "is singing"
            vocal_kind = "spoken" if performance_mode == "speaking" else "sung"
            audio_assignment = (
                "Audio 1: use unchanged as the primary and only audio track. Preserve its exact music, vocals, timing, "
                f"rhythm, phrasing, tone, and duration, and use it as the exact vocal, timing, and lip-sync reference. "
                f"{performer} {action} the exact line "
                f"“{lyric_text}”. Synchronize lips, mouth shapes, jaw movement, facial muscles, and breathing precisely "
                f"to that {vocal_kind} line in Audio 1. Do not generate, add, replace, remix, or extend any music, "
                "ambience, dialogue, vocals, or sound effects."
            )
        elif no_lip_sync:
            audio_assignment = (
                "Audio 1: use unchanged as the primary and only audio track. Preserve its exact music, vocals, timing, "
                "rhythm, phrasing, tone, and duration. This portion contains no vocal line for the visible character, "
                "so the character does not sing, speak, or lip-sync; keep the mouth naturally relaxed or closed. "
                "Do not generate, add, replace, remix, or extend any music, ambience, dialogue, vocals, or sound effects."
            )
        else:
            audio_assignment = (
                "Audio 1: use unchanged as the primary and only audio track. Preserve its exact music, vocals, timing, "
                "rhythm, phrasing, tone, and duration. Use it as the exact performance and movement reference. "
                "Do not generate, add, replace, remix, or extend any music, ambience, dialogue, vocals, or sound effects."
            )
        input_audio_assignment_pattern = re.compile(
            rf"(?ims)^Audio\s+1\s*:.*?(?=\n\n(?:{timestamp_pattern}|Image\s+\d+\s*:|Video\s+\d+\s*:|Audio\s*:|Continuity\s*:|REFERENCE\s+SUBJECT\s+COUNT\b)|\Z)"
        )
        cleaned = input_audio_assignment_pattern.sub("", cleaned).strip()
        insert_before_first_timeline(audio_assignment)

    if no_lip_sync:
        enforce_visual_only_timeline()

    if lyric_text and not no_lip_sync and not ordered_native_dialogue:
        lines = cleaned.split("\n")
        exact_line_lower = lyric_text.casefold()
        for index, line in enumerate(lines):
            if re.match(r"^Audio\s+1\s*:", line, flags=re.IGNORECASE):
                if exact_line_lower not in line.casefold() or "lip" not in line.casefold():
                    action = "spoken" if performance_mode == "speaking" else "sung"
                    lines[index] = (
                        f"{line.rstrip()} Preserve the exact {action} line “{lyric_text}” and synchronize lips, mouth shapes, "
                        "jaw movement, facial muscles, and breathing precisely to Audio 1."
                    )
                break
        cleaned = "\n".join(lines)

        vocal_verb = "says" if performance_mode == "speaking" else "visibly sings"
        sync_kind = "spoken dialogue" if performance_mode == "speaking" else "sung lyric"
        timeline_block_pattern = re.compile(
            rf"({timestamp_pattern}\n)(.*?)(?=\n\n(?:{timestamp_pattern}|Audio\s*:|Continuity\s*:)|$)",
            flags=re.IGNORECASE | re.DOTALL,
        )

        def ensure_vocal_in_timeline(match):
            header = match.group(1)
            body = match.group(2).strip()
            body_lower = body.casefold()
            visibly_performed = (
                lyric_text.casefold() in body_lower
                and bool(re.search(r"\b(?:sing|sings|singing|sung|lip[ -]?sync|say|says|speaking|spoken)\b", body_lower))
            )
            if visibly_performed:
                return f"{header}{body}"
            required_action = (
                f"During only the portion of this interval where the {sync_kind} is audible in Audio 1, {performer} "
                f"{vocal_verb} “{lyric_text}” with precise lip, mouth-shape, jaw, facial-muscle, and breathing synchronization. "
                "When the supplied vocal ends, the mouth closes or relaxes naturally; never stretch, restart, or repeat the line to fill the interval."
            )
            return f"{header}{body.rstrip()} {required_action}".strip()

        cleaned = timeline_block_pattern.sub(ensure_vocal_in_timeline, cleaned)

    if native_audio:
        native_summary_pattern = re.compile(
            r"(?ims)^Audio\s*:.*?(?=\n\n(?:Continuity\s*:|MINIMAX\s+NATIVE\s+VOICE\s+IDENTITY\b)|\Z)"
        )
        cleaned = native_summary_pattern.sub("", cleaned).strip()
        if ordered_native_dialogue:
            audio_summary = (
                "Audio: Generate the ordered spoken dialogue as the primary audio track using each character’s assigned voice. "
                "Preserve every word and speaker turn exactly, keep the visible speaker’s lip sync precise, and keep all non-speaking characters silent. "
                "Add only subtle low-level requested ambience and sound effects underneath. Do not add narration, extra dialogue, music, or vocal layers."
            )
        elif lyric_text and not no_lip_sync:
            vocal_kind = "spoken line" if performance_mode == "speaking" else "sung lyric"
            audio_summary = (
                f"Audio: Generate the exact {vocal_kind} “{lyric_text}” as the primary audio track and synchronize the visible performance precisely to it. "
                "Add only subtle low-level requested ambience and sound effects underneath. Do not add replacement words, extra dialogue, or vocal layers."
            )
        else:
            audio_summary = (
                "Audio: Generate only the requested environmental ambience and sound effects. "
                "Do not add speech, singing, dialogue, lyrics, narration, or vocal layers."
            )
        insert_before_continuity(audio_summary)
    else:
        if lyric_text and not no_lip_sync:
            audio_summary = (
                "Audio: Audio 1 remains unchanged as the primary and only audio track. Preserve its exact music, vocals, "
                f"timing, rhythm, phrasing, tone, and duration, and keep the lip sync exact to “{lyric_text}”. "
                "Do not generate, add, replace, remix, or extend any music, ambience, dialogue, vocals, or sound effects."
            )
        elif no_lip_sync:
            audio_summary = (
                "Audio: Audio 1 remains unchanged as the primary and only audio track. Preserve its exact music, vocals, "
                "timing, rhythm, phrasing, tone, and duration. The visible character does not sing, speak, or lip-sync "
                "during this non-vocal portion and keeps the mouth naturally relaxed or closed. Do not generate, add, "
                "replace, remix, or extend any music, ambience, dialogue, vocals, or sound effects."
            )
        else:
            audio_summary = (
                "Audio: Audio 1 remains unchanged as the primary and only audio track. Preserve its exact music, vocals, "
                "timing, rhythm, phrasing, tone, and duration. Do not generate, add, replace, remix, or extend any music, "
                "ambience, dialogue, vocals, or sound effects."
            )
        input_audio_summary_pattern = re.compile(
            r"(?ims)^Audio\s*:.*?(?=\n\n(?:Audio\s+1\s*:|Continuity\s*:|REFERENCE\s+SUBJECT\s+COUNT\b)|\Z)"
        )
        cleaned = input_audio_summary_pattern.sub("", cleaned).strip()
        insert_before_continuity(audio_summary)

    if not re.search(r"(?im)^Continuity\s*:", cleaned):
        continuity = (
            "Continuity: Preserve the same visible character identity, appearance, clothing, environment, lighting, and spatial "
            "relationships throughout. Do not introduce new characters, objects, text, logos, captions, or scene changes unless explicitly requested."
        )
        cleaned = f"{cleaned.rstrip()}\n\n{continuity}"

    if native_audio:
        cleaned = re.sub(r"\bAudio\s+1\b", "the generated native audio", cleaned, flags=re.IGNORECASE)

    if ordered_native_dialogue:
        # Keep the exact dialogue script in one authoritative place. Repeating a full
        # quoted cue in timeline/summary sections can make H3 restart or repeat it.
        native_block = re.search(
            rf"(?ims)^Native\s+audio\s*:.*?(?=\n\n{timestamp_pattern}|\Z)",
            cleaned,
        )
        if native_block:
            prefix = cleaned[:native_block.end()]
            suffix = cleaned[native_block.end():]
            for cue in speaker_assignments:
                cue_text = str(cue.get("text") or "").strip()
                if not cue_text:
                    continue
                escaped = re.escape(cue_text)
                suffix = re.sub(
                    rf"[\u201c\"]\s*{escaped}\s*[\u201d\"]",
                    "the assigned cue",
                    suffix,
                    flags=re.IGNORECASE,
                )
                suffix = re.sub(escaped, "the assigned cue", suffix, flags=re.IGNORECASE)
            cleaned = prefix + suffix

    try:
        camera_speed = max(0.0, min(10.0, float(payload.get("camera_motion_speed", 4))))
    except Exception:
        camera_speed = 4.0
    try:
        character_speed = max(0.0, min(10.0, float(payload.get("character_motion_speed", 4))))
    except Exception:
        character_speed = 4.0

    if camera_speed >= 7:
        camera_replacements = [
            (r"\bslow cinematic drift\b", "energetic cinematic tracking drift"),
            (r"\bslow orbit\b", "energetic orbit"),
            (r"\bslow (left|right) orbit\b", r"energetic \1 orbit"),
            (r"\bslow zoom out\b", "brisk pull-back reveal"),
            (r"\bslow (left|right|side|lateral) drift\b", r"brisk \1 tracking drift"),
            (r"\bslow (pan|tilt|track|tracking|pull[ -]?back|drift)\b", r"brisk \1"),
            (r"\bgentle lateral drift\b", "energetic lateral tracking"),
            (r"\bgentle pan reveal\b", "brisk pan reveal"),
            (r"\bgentle (pan|tilt|orbit|drift|camera movement)\b", r"brisk \1"),
            (r"\bsubtle handheld movement\b", "active handheld tracking"),
            (r"\bsubtle handheld camera\b", "active handheld camera"),
            (r"\bsubtle handheld follow\b", "energetic handheld follow"),
            (r"\bsubtle rack focus\b", "quick rack focus"),
            (r"\bsubtle energetic orbit\b", "energetic orbit"),
            (r"\bsubtle settling pause\b", "active reframing beat"),
            (r"\bsubtle orbit movement\b", "energetic orbit movement"),
            (r"\b(?:quiet handheld hold|locked-off reaction hold|locked-off shot)\b", "active handheld reaction tracking"),
            (r"\brestrained pan\b", "brisk pan"),
        ]
        for pattern, replacement in camera_replacements:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        if not re.search(r"\b(?:tracking|orbit|whip pan|pan|tilt|crane|pullback|pull-back|push|dolly|handheld|reveal)\b", cleaned, flags=re.IGNORECASE):
            first_timeline = re.search(rf"(?ims)({timestamp_pattern}\n)(.*?)(?=\n\n(?:{timestamp_pattern}|Audio\s*:|Continuity\s*:)|$)", cleaned)
            if first_timeline:
                body = first_timeline.group(2).rstrip()
                replacement = f"{first_timeline.group(1)}{body} The camera uses energetic tracking throughout this interval and never settles into a static hold."
                cleaned = cleaned[:first_timeline.start()] + replacement + cleaned[first_timeline.end():]

    if character_speed >= 4 and not no_visible_character:
        physical_action_pattern = r"\b(?:walks?|steps?|strides?|runs?|sprints?|dances?|crosses?|lunges?|reaches?|pushes?|pulls?|climbs?|fights?|brushes?|sweeps?|gestures?|interacts?|grabs?|lifts?|paces?)\b"
        timeline_text = "\n".join(
            match.group(2)
            for match in re.finditer(
                rf"(?ims)({timestamp_pattern}\n)(.*?)(?=\n\n(?:{timestamp_pattern}|Audio\s*:|Continuity\s*:)|$)",
                cleaned,
            )
        )
        if not re.search(physical_action_pattern, timeline_text, flags=re.IGNORECASE):
            first_timeline = re.search(rf"(?ims)({timestamp_pattern}\n)(.*?)(?=\n\n(?:{timestamp_pattern}|Audio\s*:|Continuity\s*:)|$)", cleaned)
            if first_timeline:
                body = first_timeline.group(2).rstrip()
                replacement = (
                    f"{first_timeline.group(1)}{body} The visible subject performs a clear physical action with the body, hands, "
                    "or surrounding set instead of relying on facial movement alone."
                )
                cleaned = cleaned[:first_timeline.start()] + replacement + cleaned[first_timeline.end():]

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _generate_builder_motion_notes(payload):
    source_mode = str(payload.get("source_mode") or "concept").strip().lower()
    story_idea = str(payload.get("story_idea") or "").strip()
    theme_style = str(payload.get("theme_style") or "").strip()
    previous_summary = str(payload.get("previous_summary") or "").strip()
    video_mode = str(payload.get("video_mode") or "i2v").strip().lower()
    scenes = payload.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("No scenes were provided for motion note creation.")

    cleaned_scenes = []
    for scene in scenes[:20]:
        if not isinstance(scene, dict):
            continue
        try:
            number = int(scene.get("scene_number") or scene.get("number") or len(cleaned_scenes) + 1)
        except Exception:
            number = len(cleaned_scenes) + 1
        timeline_notes = scene.get("timeline_notes") or []
        if not isinstance(timeline_notes, list):
            timeline_notes = []
        cleaned_scenes.append({
            "scene_number": max(1, number),
            "label": str(scene.get("label") or f"Scene {number}").strip()[:160],
            "concept_prompt": str(scene.get("concept_prompt") or "").strip()[:1600],
            "director_note": str(scene.get("director_note") or "").strip()[:1200],
            "timeline_notes": [
                {
                    "label": str(item.get("label") or item.get("type") or "note").strip()[:160],
                    "note": str(item.get("note") or "").strip()[:800],
                    "start": item.get("start"),
                    "end": item.get("end"),
                }
                for item in timeline_notes[:8]
                if isinstance(item, dict)
            ],
        })
    if not cleaned_scenes:
        raise ValueError("No valid scenes were provided for motion note creation.")

    source_note = {
        "concept": "Use only concept prompts.",
        "concept_director": "Use concept prompts and director notes.",
        "concept_timeline": "Use concept prompts and overlapping timeline notes.",
        "all": "Use concept prompts, director notes, and overlapping timeline notes.",
    }.get(source_mode, "Use concept prompts.")

    instruction = (
        "You will create concise I2V/T2V motion notes for scene-by-scene music video generation.\n"
        "Return only valid JSON in this exact flat format:\n"
        "{\n"
        "  \"Motion1\": \"\",\n"
        "  \"Motion2\": \"\"\n"
        "}\n\n"
        f"Source mode: {source_mode}. {source_note}\n"
        f"Target video mode: {video_mode.upper()}.\n\n"
        "Write one motion note for each provided scene.\n"
        "Each motion note should describe camera motion, subject movement or interaction, environmental motion, and emotional pacing.\n"
        "Use the concept prompt as the main visual action source. Use director notes and timeline notes only when provided by the source mode.\n\n"
        "Rules:\n"
        "- Keep each motion note one sentence or one short paragraph.\n"
        "- Do not write a full video prompt.\n"
        "- Do not repeat character appearance details.\n"
        "- Do not include camera settings, frame rate, aspect ratio, render quality, or model tags.\n"
        "- For instrumental/no-subject scenes, focus on environmental movement, transitions, symbolic motion, light, particles, atmosphere, or camera drift.\n"
        "- For solo character scenes, include performance/reaction movement, facial/emotional energy, body motion, and camera movement.\n"
        "- For two-character scenes, include interaction, blocking, distance, reaction, duet/confrontation energy, and camera movement.\n"
        "- Keep motion connected across the batch like shots from the same music video.\n"
        "- Return only the JSON object.\n\n"
        f"Story idea:\n{story_idea or '[none provided]'}\n\n"
        f"Style/theme:\n{theme_style or '[none provided]'}\n\n"
        f"Previous batch motion summary:\n{previous_summary or '[none yet]'}\n\n"
        "Scenes:\n"
        f"{json.dumps(cleaned_scenes, ensure_ascii=False, indent=2)}"
    )
    text, info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.45),
        top_p=float(payload.get("top_p") or 0.95),
        max_new_tokens=int(payload.get("max_new_tokens") or 1200),
        label="Motion Note Creator",
        preserve_paragraphs=True,
    )
    raw = _clean_lm_studio_plain_text(text)
    notes = {}
    try:
        parsed = _extract_json_object_from_text(raw)
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                match = re.search(r"(?:motion|scene)\s*(\d+)", str(key), re.I)
                if not match:
                    continue
                note_text = str(value or "").strip()
                if note_text:
                    notes[f"Motion{int(match.group(1))}"] = note_text
    except Exception:
        notes = {}
    if not notes:
        for match in re.finditer(r'"?\b(?:Motion|Scene)\s*(\d+)"?\s*:\s*"([^"]+)"', raw, re.I | re.S):
            note_text = match.group(2).strip()
            if note_text:
                notes[f"Motion{int(match.group(1))}"] = note_text
    if not notes:
        raise ValueError("Gemma did not return any MotionN notes.")
    summary_lines = []
    for key in sorted(notes.keys(), key=lambda item: int(re.search(r"\d+", item).group(0))):
        text_value = notes[key]
        summary_lines.append(f"{key}: {text_value[:180]}")
    return {
        "notes": notes,
        "raw": raw,
        "summary": "\n".join(summary_lines)[:1200],
        "run_info": info,
    }


def _generate_builder_i2v_prompt(payload):
    from .cache import _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    t2i_prompt = str(payload.get("t2i_prompt", "") or "").strip()
    image_reference_path = str(payload.get("image_reference_path", "") or "").strip().strip('"')
    image_reference_data = str(payload.get("image_reference_data", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    subject_context = str(payload.get("subject_context", "") or "").strip()
    location_context = str(payload.get("location_context", "") or "").strip()
    no_character_present = bool(payload.get("no_character_present") or payload.get("no_subject") or payload.get("no_visible_subject"))
    performance_mode = str(
        payload.get("performance_mode")
        or payload.get("performanceMode")
        or payload.get("video_type")
        or payload.get("videoType")
        or ""
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if performance_mode in {"speaking", "short_film", "dialogue", "dialog"}:
        performance_mode = "speaking"
    elif performance_mode in {"no_lip_sync", "nolipsync", "no_lipsync", "no_sync", "silent", "visual_only"}:
        performance_mode = "no_lip_sync"
    else:
        performance_mode = "singing"
    if performance_mode == "speaking":
        mode_note = (
            "Video Type / performance mode:\nspeaking / short film. "
            "If a line is present, the visible speaker says it naturally. Do not use singing, rapping, vocals, lyric, lip-sync, or music-performance wording."
        )
    elif performance_mode == "no_lip_sync":
        mode_note = (
            "Video Type / performance mode:\nno lip sync / visual-only. "
            "Do not quote lyric text. Do not mention saying, speaking, dialogue, singing, rapping, vocals, lyric, lip-sync, mouth movement, or no-vocal status. "
            "Use visible action, camera motion, environmental motion, mood, and physical movement instead."
        )
    else:
        mode_note = (
            "Video Type / performance mode:\nsinging / music video. "
            "Use singing behavior only when the scene notes or lyric context call for a vocal performance."
        )
    text_runner = _llm_runner_from_payload(payload)
    if not model_file and text_runner not in _EXTERNAL_LLM_RUNNERS:
        raise ValueError("Choose an I2V Gemma model first.")
    if model_file and text_runner not in _EXTERNAL_LLM_RUNNERS and not model_file.lower().endswith(".gguf"):
        raise ValueError("The I2V model field is not a GGUF model.")

    image = None
    has_image_reference = False
    if image_reference_data:
        image = _image_from_data_url(image_reference_data).convert("RGB")
        has_image_reference = True
    elif image_reference_path:
        image_path = _resolve_existing_file(image_reference_path, "I2V image reference")
        image = Image.open(image_path).convert("RGB")
        has_image_reference = True

    if has_image_reference and text_runner not in _EXTERNAL_LLM_RUNNERS and not model_file:
        raise ValueError("Choose an I2V vision Gemma model first.")
    if not has_image_reference and not t2i_prompt:
        raise ValueError("Create or paste a T2I prompt first, or save/load an image reference.")
    if not has_image_reference:
        theme_style = _read_text_file(payload.get("theme_style_path", ""), "Theme/style file")
        story_idea = _read_text_file(payload.get("story_idea_path", ""), "Story idea file")
        subject_scene = _read_text_file(payload.get("subject_scene_path", ""), "Subject/scene file")
        context_parts = []
        if no_character_present:
            context_parts.append("Subject visibility:\nNo main character, singer, performer, person, mapped subject, or character reference is present in this scene. Use location, props, objects, atmosphere, and camera motion instead.")
        elif subject_scene:
            context_parts.append(f"Subject/scene:\n{subject_scene}")
        if subject_context and not no_character_present:
            context_parts.append(f"Mapped scene character(s):\n{subject_context}")
        if location_context:
            context_parts.append(f"Mapped scene location:\n{location_context}")
        if theme_style:
            context_parts.append(f"Theme/style:\n{theme_style}")
        if story_idea:
            context_parts.append(f"Story idea:\n{story_idea}")
        if context_parts:
            user_notes = "\n\n".join(context_parts + ([f"Segment motion notes:\n{user_notes}"] if user_notes else []))

    user_notes = "\n\n".join([mode_note, user_notes]).strip()

    llm = _builder_local_llm(payload) if text_runner not in _EXTERNAL_LLM_RUNNERS else None
    model_file = _builder_local_model_file(payload, model_file)
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION) if llm else ""
    mmproj_file = _builder_local_mmproj_file(payload, mmproj_file)
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file) if has_image_reference and llm else ""
    i2v_instructions = _effective_builder_instruction(payload, "i2v", _I2V_INSTRUCTIONS)
    if has_image_reference:
        prompt = (
            f"{i2v_instructions}\n\n"
            "Use the provided image as the primary visual reference. Preserve the visible subject, setting, clothing, mood, and scene identity from the image. "
            "Use the text-to-image prompt only as extra scene context. "
            "Use only the provided image, the text-to-image prompt, and the user motion/camera notes. Do not use concept prompts, global story text, theme files, or subject files. "
            "Use the user motion/camera notes as the highest priority when deciding motion, performance, camera movement, and energy.\n\n"
            f"Text-to-image prompt:\n{t2i_prompt or 'Use the image as the main visual reference.'}\n\n"
        )
    else:
        prompt = f"{i2v_instructions}\n\nText-to-image prompt:\n{t2i_prompt}\n\n"
    prompt += f"User motion/camera notes:\n{user_notes or 'Create fast cinematic performance motion that fits the scene.'}"

    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or (0.25 if has_image_reference else 0.7))
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or 4000))
    unload_after = bool(payload.get("unload_after", True))
    seed = payload.get("seed")

    try:
        if has_image_reference and text_runner in _EXTERNAL_LLM_RUNNERS:
            text, run_info = _try_run_remote_vision(
                payload,
                prompt,
                [image],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        elif text_runner in _EXTERNAL_LLM_RUNNERS:
            text, run_info = _run_builder_text_llm(
                payload,
                prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                label="I2V LLM",
            )
        else:
            model = llm._load_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            if has_image_reference:
                text = llm._run_gguf_vision_pipeline(
                    model=model,
                    pil_images=[image],
                    instruction_text=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    seed=int(seed) if seed is not None else None,
                )
                run_info = {"runner": "builtin", "used_model": model_path, "unloaded": unload_after}
            else:
                text = llm._run_gguf_text_pipeline(
                    model=model,
                    instruction_text=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    seed=int(seed) if seed is not None else None,
                )
                run_info = {"runner": "builtin", "used_model": model_path, "unloaded": unload_after}
        text = _clean_gemma_prompt_text(text)
        text = _repair_and_validate_builder_gemma_prompt(payload, text, "I2V")
        return {"prompt": text, "used_model": run_info.get("used_model", model_path), "used_mmproj": mmproj_path, "used_image_reference": has_image_reference, "runner": run_info.get("runner", "builtin"), "unloaded": run_info.get("unloaded", unload_after)}
    finally:
        if llm and unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _chained_i2v_meta_language_error(text):
    forbidden_patterns = [
        r"\bcurrent\s+(?:frame|image|picture|photo)\b",
        r"\bprovided\s+(?:frame|image|picture|photo)\b",
        r"\bprevious\s+(?:frame|image|picture|photo|scene|video)\b",
        r"\blast\s+(?:frame|image|picture|photo)\b",
        r"\bfirst\s+(?:frame|image|picture|photo)\b",
        r"\bstart(?:ing)?\s+(?:frame|image|picture|photo)\b",
        r"\b(?:this|the)\s+(?:frame|image|picture|photo)\b",
        r"\bfrom\s+(?:the\s+)?(?:frame|image|picture|photo)\b",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, text or "", flags=re.I):
            return pattern
    return ""


def _validate_chained_i2v_prompt(text):
    if _chained_i2v_meta_language_error(text):
        raise ValueError(
            "Gemma returned a chained I2V prompt with frame/image meta language. "
            "Try again or simplify the chain direction."
        )


def _repair_chained_i2v_meta_prompt(payload, text, transition_lora_prompt=False, transition_lora_trigger="zhuanchang"):
    original = str(text or "").strip()
    if not original or not _chained_i2v_meta_language_error(original):
        return original
    repair_payload = dict(payload or {})
    repair_model = str(
        repair_payload.get("repair_model_file")
        or repair_payload.get("text_model_file")
        or repair_payload.get("model_file")
        or ""
    ).strip()
    if repair_model:
        repair_payload["model_file"] = repair_model
    repair_payload["mmproj_file"] = ""
    repair_payload["use_vision"] = False
    trigger = str(transition_lora_trigger or "zhuanchang").strip() or "zhuanchang"
    trigger_rule = (
        f"\n- End the prompt with exactly one trigger phrase: {trigger}"
        if transition_lora_prompt else ""
    )
    instruction = (
        "Rewrite this chained LTX image-to-video prompt into one normal final video prompt paragraph.\n\n"
        "The prompt is already conceptually useful, but it contains forbidden meta language about frames/images/references/sources. "
        "Remove that meta language while preserving the visible starting subject, setting, action, camera motion, transformation, lyrics/performance intent, and ending state.\n\n"
        "Rules:\n"
        "- Do not mention frames, images, pictures, photos, references, sources, current visuals, provided visuals, previous scenes, or previous videos.\n"
        "- Do not say use/using/based on/from the image or frame.\n"
        "- Keep it as a normal cinematic image-to-video prompt only.\n"
        "- No markdown, labels, quotes around the whole prompt, JSON, or bullet points."
        f"{trigger_rule}\n\n"
        f"Prompt to rewrite:\n{original[:5000]}"
    )
    try:
        repaired, _run_info = _run_builder_text_llm(
            repair_payload,
            instruction,
            temperature=0.2,
            top_p=0.85,
            max_new_tokens=900,
            label="Chained I2V repair Gemma",
        )
        repaired = _clean_gemma_prompt_text(repaired)
        if transition_lora_prompt and trigger:
            repaired = re.sub(rf"(?:,\s*)?{re.escape(trigger)}\s*$", "", repaired, flags=re.I).strip().rstrip(".,;")
            repaired = f"{repaired}, {trigger}"
        if repaired and not _chained_i2v_meta_language_error(repaired):
            return repaired
    except Exception:
        pass
    return original


def _fallback_chained_i2v_prompt(
    scene_context="",
    user_notes="",
    story_context="",
    chain_style="continuous",
    transition_lora_prompt=False,
    transition_lora_trigger="zhuanchang",
):
    context = " ".join(
        part
        for part in (
            str(scene_context or "").strip(),
            str(user_notes or "").strip(),
            str(story_context or "").strip(),
        )
        if part
    )
    context = re.sub(r"\s+", " ", context).strip()
    if len(context) > 700:
        context = context[:700].rsplit(" ", 1)[0].strip()
    style = str(chain_style or "continuous").strip().lower().replace("-", "_").replace(" ", "_")
    if style in {"transformation", "surreal"} or transition_lora_prompt:
        prompt = (
            "A cinematic shot begins from the visible subject and setting, preserving the existing pose, lighting, colors, and composition. "
            "As the camera moves smoothly, the subject's outfit, hair, materials, and silhouette begin to transform with fluid detail, while the surrounding environment shifts into a new expressive location shaped by the scene's story and mood. "
            "Lighting changes across the subject's face and clothing, textures ripple and reform, and the background evolves into a more dramatic visual world while the motion remains continuous and natural."
        )
    elif style == "environment_shift":
        prompt = (
            "A cinematic shot begins from the visible subject and setting, preserving the existing pose, lighting, colors, and composition. "
            "As the camera moves smoothly, the surrounding environment transforms with changing atmosphere, architecture, weather, and light, while the visible subject remains grounded in the scene. "
            "The location gradually becomes a new expressive space shaped by the story and mood, with continuous motion and natural visual flow."
        )
    else:
        prompt = (
            "A cinematic shot begins from the visible subject and setting, preserving the existing pose, lighting, colors, and composition. "
            "The camera moves smoothly as the subject continues with natural performance energy, subtle expression changes, and environmental motion. "
            "The scene develops toward the next story beat while maintaining continuous visual flow."
        )
    if context:
        prompt += f" The transformation direction follows this scene context: {context}"
    trigger = str(transition_lora_trigger or "zhuanchang").strip() or "zhuanchang"
    if transition_lora_prompt:
        prompt = re.sub(rf"(?:,\s*)?{re.escape(trigger)}\s*$", "", prompt, flags=re.I).strip().rstrip(".,;")
        prompt = f"{prompt}, {trigger}"
    return prompt


def _chained_i2v_style_note(chain_style, chain_direction):
    style = str(chain_style or "continuous").strip().lower().replace("-", "_").replace(" ", "_")
    if style not in {"continuous", "surreal", "transformation", "environment_shift"}:
        style = "continuous"
    direction = str(chain_direction or "").strip()
    if style == "surreal":
        note = "Style mode: surreal continuity. Keep the opening visual state recognizable, then introduce dreamlike impossible motion, altered light, strange materials, or poetic environmental behavior."
    elif style == "transformation":
        note = (
            "Style mode: subject and environment transformation. Start from the visible subject, clothing, pose, lighting, and place exactly as they appear, "
            "then visibly change them during the shot. Include at least one clear wardrobe/material/body-silhouette transformation and one clear environment, lighting, weather, architecture, or location transformation when a character is visible. "
            "Do not leave the subject in the same outfit and same location with only posing or wind; the shot must evolve into something else while remaining continuous."
        )
    elif style == "environment_shift":
        note = "Style mode: environment shift. Keep the opening visual state recognizable, then gradually change the surrounding place, weather, architecture, lighting, or atmosphere while maintaining one continuous shot."
    else:
        note = "Style mode: continuous video. Keep the opening visual state recognizable and extend it with natural action, camera motion, lighting changes, and environmental motion."
    if direction:
        note += f"\nUser chain direction: {direction}"
    return note


def _generate_builder_chained_i2v_prompt(payload):
    from .cache import _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    image_reference_path = str(payload.get("image_reference_path", "") or payload.get("source_image_path", "") or "").strip().strip('"')
    image_reference_data = str(payload.get("image_reference_data", "") or "").strip()
    scene_context = str(payload.get("scene_context", "") or payload.get("t2i_prompt", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    scene_notes = str(payload.get("scene_notes", "") or "").strip()
    director_note = str(payload.get("director_note", "") or "").strip()
    story_beat = str(payload.get("story_beat", "") or "").strip()
    lyric_text = str(payload.get("lyric_text", "") or payload.get("lyrics", "") or "").strip()
    lyric_section = str(payload.get("lyric_section", "") or "").strip()
    subject_context = str(payload.get("subject_context", "") or "").strip()
    location_context = str(payload.get("location_context", "") or "").strip()
    no_character_present = bool(payload.get("no_character_present", False))
    reference_context = payload.get("reference_context") or {}
    chain_style = str(payload.get("chain_style", "") or payload.get("continuity_style", "") or "continuous").strip()
    chain_direction = str(payload.get("chain_direction", "") or payload.get("continuity_direction", "") or "").strip()
    transition_lora_prompt = bool(payload.get("transition_lora_prompt") or payload.get("use_transition_lora_prompt") or False)
    transition_lora_trigger = str(payload.get("transition_lora_trigger", "") or "zhuanchang").strip() or "zhuanchang"
    performance_mode = str(payload.get("performance_mode") or payload.get("video_type") or "").strip()
    text_runner = _llm_runner_from_payload(payload)

    if not image_reference_data and not image_reference_path:
        raise ValueError("Chained I2V needs an extracted final-frame image.")
    if not model_file and text_runner not in _EXTERNAL_LLM_RUNNERS:
        raise ValueError("Choose an I2V vision Gemma model first.")
    if model_file and text_runner not in _EXTERNAL_LLM_RUNNERS and not model_file.lower().endswith(".gguf"):
        raise ValueError("The I2V vision model field is not a GGUF model.")

    if image_reference_data:
        image = _image_from_data_url(image_reference_data).convert("RGB")
    else:
        image_path = _resolve_existing_file(image_reference_path, "Chained I2V image reference")
        image = Image.open(image_path).convert("RGB")

    style_note = _chained_i2v_style_note(chain_style, chain_direction)
    reference_subject_context = ""
    reference_location_context = ""
    if isinstance(reference_context, dict):
        reference_subject_context = str(reference_context.get("subject_context", "") or "").strip()
        reference_location_context = str(reference_context.get("location_context", "") or "").strip()
        if not reference_subject_context:
            subject_refs = reference_context.get("subject_refs") or []
            if isinstance(subject_refs, list):
                subject_lines = []
                for subject in subject_refs:
                    if not isinstance(subject, dict):
                        continue
                    name = str(subject.get("name", "") or "").strip()
                    description = str(subject.get("description", "") or "").strip()
                    trigger = str(subject.get("trigger_phrase", "") or "").strip()
                    line = " - ".join(part for part in (name, description, f"trigger: {trigger}" if trigger else "") if part)
                    if line:
                        subject_lines.append(line)
                reference_subject_context = "\n".join(subject_lines)
        if not reference_location_context:
            location_ref = reference_context.get("location_ref")
            if isinstance(location_ref, dict):
                name = str(location_ref.get("name", "") or "").strip()
                description = str(location_ref.get("description", "") or "").strip()
                trigger = str(location_ref.get("trigger_phrase", "") or "").strip()
                reference_location_context = " - ".join(part for part in (name, description, f"trigger: {trigger}" if trigger else "") if part)
    elif reference_context:
        reference_subject_context = str(reference_context).strip()

    subject_context = subject_context or reference_subject_context
    location_context = location_context or reference_location_context
    story_parts = [
        ("Scene concept", scene_context),
        ("Scene notes", scene_notes),
        ("Director note", director_note),
        ("Story beat", story_beat),
        ("Lyric section", lyric_section),
        ("Lyrics/dialogue", lyric_text),
        ("Reference Builder subject", subject_context if not no_character_present else ""),
        ("Reference Builder location", location_context),
    ]
    story_context = "\n".join(f"{label}: {value}" for label, value in story_parts if value)
    no_character_rule = (
        "\n- The next shot is marked as no-character-present. Do not add the mapped subject unless the visible scene already clearly contains them."
        if no_character_present else ""
    )
    transition_lora_rules = ""
    if transition_lora_prompt:
        transition_lora_rules = (
            "\nTransition LoRA prompt style is active.\n"
            f"- Write in a transition-LoRA style: visible starting shot, detailed transformation process, changed ending state, lighting/material/environment details, and strong continuous camera movement.\n"
            "- Make the transformation explicit and temporal with phrases such as gradually, as the camera moves, the clothing shifts into, the environment morphs into, the lighting changes from/to, or the scene reveals.\n"
            "- For a visible character, include a clear outfit/material/hair/silhouette change and a clear environment/location/lighting/style change unless the user direction forbids one of them.\n"
            "- For an environment-only shot, transform the place, weather, architecture, terrain, lighting, or style into a different readable destination.\n"
            "- End the prompt with the transition trigger phrase exactly once.\n"
            f"- Trigger phrase: {transition_lora_trigger}\n"
        )
    i2v_instructions = _effective_builder_instruction(payload, "i2v", _I2V_INSTRUCTIONS)
    instruction = (
        f"{i2v_instructions}\n\n"
        "Write one normal image-to-video prompt for LTX. The video model will receive a visual source separately, "
        "but your output must read like an ordinary video prompt only.\n\n"
        "Rules for the output:\n"
        "- Begin with a concrete description of only what is actually visible: subject, clothing, pose, lighting, camera angle, colors, and setting. Do not invent a different location, outfit, pose, shot angle, or material from the story context in the opening description.\n"
        "- Treat the story, lyrics, subject, and location context as the direction the shot may evolve toward, not as permission to replace the visible starting facts.\n"
        "- Continue with motion and changes that naturally develop from the visible state while moving toward the story context below.\n"
        "- If transformation style is active, the prompt must include visible change: clothing/materials/body silhouette should transform and the surrounding environment or lighting should transform. Avoid a boring continuation where the subject only poses, stares, walks, wind blows, or the camera moves while the outfit and place stay basically the same.\n"
        "- Silently assess the visible shot scale before choosing camera motion. If it is already a close-up or extreme close-up, do not zoom or push farther into the face; use a slow pullback, orbit, pan, rack focus, expression change, environmental motion, or transformation instead. If it is already a wide or distant shot, do not pull farther away; use a push-in toward the subject, orbiting push-in, tracking move, subject action, or environmental change instead. If it is a medium shot, choose motion that changes the composition without repeating what is already true.\n"
        "- Use the story context to make this shot meaningfully different from the last shot when it fits the requested chain style.\n"
        "- Do not mention frames, images, pictures, photos, references, sources, current visuals, provided visuals, previous scenes, or previous videos.\n"
        "- Do not write instructions to the model. Output only the final cinematic prompt paragraph.\n"
        f"- No markdown, labels, quotes, JSON, or bullet points.{no_character_rule}\n"
        f"{transition_lora_rules}\n"
        f"{style_note}\n\n"
        f"Story context for the next shot:\n{story_context or '(none)'}\n\n"
        f"Motion/performance notes:\n{user_notes or 'Create cinematic motion that fits the visible scene.'}\n\n"
        f"Performance mode:\n{performance_mode or 'Use the visible scene and notes. Do not force singing unless the notes call for it.'}"
    )

    llm = _builder_local_llm(payload) if text_runner not in _EXTERNAL_LLM_RUNNERS else None
    model_file = _builder_local_model_file(payload, model_file)
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION) if llm else ""
    mmproj_file = _builder_local_mmproj_file(payload, mmproj_file)
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file) if llm else ""
    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.9)
    max_new_tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or 1200))
    unload_after = bool(payload.get("unload_after", True))
    seed = payload.get("seed")

    try:
        model = None
        if text_runner in _EXTERNAL_LLM_RUNNERS:
            text, run_info = _try_run_remote_vision(
                payload,
                instruction,
                [image],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        else:
            model = llm._load_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            text = llm._run_gguf_vision_pipeline(
                model=model,
                pil_images=[image],
                instruction_text=instruction,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=int(seed) if seed is not None else None,
            )
            run_info = {"runner": "builtin", "used_model": model_path, "unloaded": unload_after}
        text = _clean_gemma_prompt_text(text)
        try:
            text = _repair_and_validate_builder_gemma_prompt(payload, text, "Chained I2V")
        except Exception:
            text = _fallback_chained_i2v_prompt(
                scene_context=scene_context,
                user_notes=user_notes,
                story_context=story_context,
                chain_style=chain_style,
                transition_lora_prompt=transition_lora_prompt,
                transition_lora_trigger=transition_lora_trigger,
            )
        if transition_lora_prompt and transition_lora_trigger:
            text = re.sub(rf"(?:,\s*)?{re.escape(transition_lora_trigger)}\s*$", "", text, flags=re.I).strip().rstrip(".,;")
            text = f"{text}, {transition_lora_trigger}"
        text = _repair_chained_i2v_meta_prompt(
            payload,
            text,
            transition_lora_prompt=transition_lora_prompt,
            transition_lora_trigger=transition_lora_trigger,
        )
        if _looks_like_gemma_repeat_failure(text) or _looks_like_unfilled_prompt_template(text):
            text = _fallback_chained_i2v_prompt(
                scene_context=scene_context,
                user_notes=user_notes,
                story_context=story_context,
                chain_style=chain_style,
                transition_lora_prompt=transition_lora_prompt,
                transition_lora_trigger=transition_lora_trigger,
            )
        meta_language_warning = bool(_chained_i2v_meta_language_error(text))
        return {
            "prompt": text,
            "used_model": run_info.get("used_model", model_path),
            "used_mmproj": mmproj_path,
            "used_image_reference": True,
            "runner": run_info.get("runner", "builtin"),
            "unloaded": run_info.get("unloaded", unload_after),
            "chain_style": chain_style or "continuous",
            "transition_lora_prompt": transition_lora_prompt,
            "transition_lora_trigger": transition_lora_trigger if transition_lora_prompt else "",
            "meta_language_warning": meta_language_warning,
        }
    finally:
        if llm and unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _normalize_flf_vision_observation(text):
    """Return canonical START/END lines without discarding paragraph breaks."""
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json|text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = re.sub(
        r"^(?:Assistant|Answer|Final answer|Final response|Observation)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    descriptions = {}
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            normalized_key = re.sub(r"[^a-z]", "", str(key or "").lower())
            if normalized_key.startswith("start") and str(value or "").strip():
                descriptions.setdefault("START", str(value).strip())
            elif normalized_key.startswith("end") and str(value or "").strip():
                descriptions.setdefault("END", str(value).strip())

    if len(descriptions) < 2:
        label_pattern = re.compile(
            r"(?im)^[ \t]*(?:[-+]\s+|\d+[.)]\s+|#{1,6}[ \t]+)?"
            r"[*_]{0,2}[ \t]*(START|END)\b"
            r"(?:[ \t]+(?:FRAME|IMAGE|DESCRIPTION|OBSERVATION|STATE))?"
            r"[ \t]*(?::|-)?[ \t]*[*_]{0,2}[ \t]*(?::|-)?[ \t]*"
        )
        matches = list(label_pattern.finditer(cleaned))
        for index, match in enumerate(matches):
            label = match.group(1).upper()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
            body = re.sub(r"\s+", " ", cleaned[match.end():end]).strip(" \t\n-*_:;")
            if body:
                descriptions.setdefault(label, body)

    missing = [label for label in ("START", "END") if not descriptions.get(label)]
    normalized = "\n".join(
        f"{label}: {descriptions[label]}"
        for label in ("START", "END")
        if descriptions.get(label)
    )
    return normalized, missing


def _generate_builder_t2v_prompt(payload):
    from .cache import _clear_vrgdg_llm_caches

    model_file = str(payload.get("model_file", "") or "").strip()
    mmproj_file = str(payload.get("mmproj_file", "") or "").strip()
    scene_prompt = str(payload.get("t2i_prompt", "") or payload.get("scene_prompt", "") or "").strip()
    image_reference_path = str(payload.get("image_reference_path", "") or "").strip().strip('"')
    image_reference_data = str(payload.get("image_reference_data", "") or "").strip()
    user_notes = str(payload.get("user_notes", "") or "").strip()
    lyric_cue_map = payload.get("lyric_cue_map")
    if isinstance(lyric_cue_map, list) and lyric_cue_map:
        # Keep the assignment in the actual LLM text context as well as in
        # the request metadata. This makes the singer/shot handoff auditable
        # and prevents generic scene context from replacing explicit cue rows.
        cue_lines = []
        for index, cue in enumerate(lyric_cue_map, start=1):
            if not isinstance(cue, dict):
                continue
            cue_type = "instrumental" if str(cue.get("type") or "").strip().lower() == "instrumental" else "vocal"
            singer = str(cue.get("singer_name") or cue.get("speaker_name") or "the assigned singer").strip()
            text = str(cue.get("text") or "").strip()
            start = cue.get("start")
            end = cue.get("end")
            timing = ""
            if start is not None and end is not None:
                timing = f" [{start}s-{end}s]"
            if cue_type == "instrumental":
                cue_lines.append(f"Cue {index}{timing}: INSTRUMENTAL — no subject sings, speaks, or lip-syncs.")
            else:
                cue_lines.append(f"Cue {index}{timing}: {singer} is the only singer for the exact words: {text!r}.")
        if cue_lines:
            user_notes = "\n\n".join(filter(None, [user_notes, "AUTHORITATIVE SINGER CUE MAP — follow exactly:\n" + "\n".join(cue_lines)]))
    subject_context = str(payload.get("subject_context", "") or "").strip()
    location_context = str(payload.get("location_context", "") or "").strip()
    no_character_present = bool(payload.get("no_character_present") or payload.get("no_subject") or payload.get("no_visible_subject"))
    instruction_key = _safe_builder_instruction_key(payload.get("builder_instruction_key") or payload.get("instruction_key") or "t2v")
    is_minimax_h3_prompt = instruction_key.startswith("minimax_h3_")
    is_minimax_h3_shot_json_task = is_minimax_h3_prompt and scene_prompt.lstrip().startswith("MiniMax H3 shot-description task.")
    prompt_only_scene_inspiration = bool(payload.get("prompt_only_scene_inspiration"))
    frame_continuity_prompt = bool(payload.get("frame_continuity_prompt")) or instruction_key == "minimax_h3_frame_continuity"
    text_runner = _llm_runner_from_payload(payload)
    if not model_file and text_runner not in _EXTERNAL_LLM_RUNNERS:
        raise ValueError("Choose a T2V Gemma model first.")
    if model_file and text_runner not in _EXTERNAL_LLM_RUNNERS and not model_file.lower().endswith(".gguf"):
        raise ValueError("The T2V model field is not a GGUF model.")
    vision_images = []
    has_image_reference = False
    image_references = payload.get("image_references") or []
    if isinstance(image_references, str):
        try:
            image_references = json.loads(image_references)
        except Exception:
            image_references = [{"path": line.strip()} for line in image_references.splitlines() if line.strip()]
    if isinstance(image_references, list):
        reference_limit = 10 if is_minimax_h3_prompt and (prompt_only_scene_inspiration or frame_continuity_prompt) else 9 if is_minimax_h3_prompt else 4
        for index, item in enumerate(image_references[:reference_limit], start=1):
            if isinstance(item, str):
                item = {"path": item}
            if not isinstance(item, dict):
                continue
            try:
                vision_images.append(_image_from_prompt_payload(item.get("path", ""), item.get("data", ""), f"T2V Gemma image reference {index}").convert("RGB"))
            except Exception as exc:
                raise ValueError(f"Could not load T2V Gemma image reference {index}: {exc}") from exc
    if not vision_images:
        if image_reference_data:
            vision_images.append(_image_from_data_url(image_reference_data).convert("RGB"))
        elif image_reference_path:
            image_path = _resolve_existing_file(image_reference_path, "T2V Gemma image reference")
            vision_images.append(Image.open(image_path).convert("RGB"))
    first_last_frame_mode = bool(payload.get("first_last_frame_mode") or payload.get("firstLastFrameMode"))
    transition_lora_active = bool(payload.get("transition_lora_active") or payload.get("transitionLoraActive"))
    flf_context_mode = str(payload.get("flf_context_mode") or "images_story").strip().lower()
    if flf_context_mode not in {"images_only", "images_story", "full"}:
        flf_context_mode = "images_story"
    flf_start_state = str(payload.get("flf_start_state") or "").strip()[:1800]
    flf_transformation = str(payload.get("flf_transformation") or "").strip()[:2400]
    flf_end_state = str(payload.get("flf_end_state") or "").strip()[:1800]
    flf_carry_forward = str(payload.get("flf_carry_forward") or "").strip()[:1800]
    if first_last_frame_mode and len(vision_images) < 2:
        raise ValueError(
            "First Last Frame Gemma requires two resolved image references, "
            f"but the backend received {len(vision_images)}. Reassign the scene's start and end images and try again."
        )
    if first_last_frame_mode and len(vision_images) >= 2:
        first_image, last_image = vision_images[0], vision_images[1]
        total_width = max(1, first_image.width + last_image.width)
        if total_width > 1920:
            scale = 1920.0 / total_width
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
            first_image = first_image.resize((max(1, int(round(first_image.width * scale))), max(1, int(round(first_image.height * scale)))), resample)
            last_image = last_image.resize((max(1, int(round(last_image.width * scale))), max(1, int(round(last_image.height * scale)))), resample)
        if first_image.width + last_image.width > 1920:
            last_image = last_image.crop((0, 0, max(1, 1920 - first_image.width), last_image.height))
        canvas_width = first_image.width + last_image.width
        canvas_height = max(first_image.height, last_image.height)
        combined = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
        combined.paste(first_image, (0, (canvas_height - first_image.height) // 2))
        combined.paste(last_image, (first_image.width, (canvas_height - last_image.height) // 2))
        vision_images = [combined]
    has_image_reference = bool(vision_images)
    if has_image_reference and text_runner not in _EXTERNAL_LLM_RUNNERS and not model_file:
        raise ValueError("Choose a T2V vision Gemma model first.")
    if has_image_reference:
        max_height = 512
        resized_images = []
        for image in vision_images:
            if image.height > max_height:
                resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
                width = max(1, int(image.width * (max_height / max(1, image.height))))
                image = image.resize((width, max_height), resample)
            resized_images.append(image)
        vision_images = resized_images

    theme_style = _read_text_file(payload.get("theme_style_path", ""), "Theme/style file")
    story_idea = _read_text_file(payload.get("story_idea_path", ""), "Story idea file")
    subject_scene = _read_text_file(payload.get("subject_scene_path", ""), "Subject/scene file")
    context_parts = []
    if no_character_present:
        context_parts.append("Subject visibility:\nNo main character, singer, performer, person, mapped subject, or character reference is present in this scene. Use location, props, objects, atmosphere, and camera motion instead.")
    elif subject_scene:
        context_parts.append(f"Subject/scene:\n{subject_scene}")
    if subject_context and not no_character_present:
        context_parts.append(f"Mapped scene character(s):\n{subject_context}")
    if location_context:
        context_parts.append(f"Mapped scene location:\n{location_context}")
    if theme_style:
        context_parts.append(f"Theme/style:\n{theme_style}")
    if story_idea:
        context_parts.append(f"Story idea:\n{story_idea}")
    if context_parts:
        user_notes = "\n\n".join(context_parts + ([f"Segment motion notes:\n{user_notes}"] if user_notes else []))
    if not scene_prompt:
        if user_notes or subject_context or location_context or theme_style or story_idea or subject_scene or no_character_present:
            scene_prompt = "Use the available scene notes, mapped references, lyrics/performance context, location details, and motion notes as the scene concept."
        else:
            raise ValueError("Create or paste scene notes, mapped references, motion notes, or a T2I/concept prompt first.")

    t2v_instructions = _effective_builder_instruction(payload, instruction_key, _T2V_INSTRUCTIONS)
    prompt_label = (
        _BUILDER_INSTRUCTION_LABELS.get(instruction_key, "MiniMax H3")
        if is_minimax_h3_prompt
        else "ID-LoRA I2V" if instruction_key == "id_lora"
        else "Reference to Video" if instruction_key == "rtv"
        else "T2V"
    )
    if has_image_reference and first_last_frame_mode:
        # Generic I2V instructions describe a single first-frame reference and
        # can conflict with FLF endpoint framing (for example, demanding that a
        # face remain visible). FLF uses its dedicated contract below instead.
        t2v_instructions = (
            "Write only one polished First Last Frame video prompt paragraph. "
            "Use the locked START and END facts, scene performance context, and transition direction exactly. "
            "Do not output analysis, labels, headings, or hidden reasoning."
        )
        transition_trigger_guidance = (
            "- The LTX 2.3 transition LoRA is active. End the final prompt with the trigger word 'zhuanchang' exactly once, and do not place it anywhere else.\n"
            if transition_lora_active
            else "- No transition LoRA is active. Do not include the trigger word 'zhuanchang'.\n"
        )
        image_guidance = (
            "First Last Frame guidance:\n"
            "- You receive one side-by-side image: the LEFT half is the opening visual state and the RIGHT half is the ending visual state.\n"
            "- Use both halves as the absolute visual truth for subject identity, wardrobe, setting, lighting, composition, body pose, subject position, camera endpoint, visible objects, and visible creatures. Lyrics, storyboard beats, transformation notes, and user text may explain how to travel between the endpoints, but they must never override, omit, or contradict what is visibly present in either half.\n"
            "- The final paragraph must explicitly account for every major endpoint difference: standing/sitting/lying posture, front/back/profile orientation, hand placement, subject location in frame, wardrobe state, foreground/background objects, creature type and color, and camera distance/crop. Do not substitute a related object or creature (for example moths for butterflies, birds for moths, or skin for a mirror) unless that exact change is visibly supported by the endpoints.\n"
            "- Write ONE compact, flowing paragraph in this exact motion structure; do not use headings, bullets, timestamps, numbered phases, or separate staged blocks.\n"
            "- Sentence 1 establishes the actual LEFT composition, subject, setting, and starting condition without inventing a wider establishing shot.\n"
            "- Sentence 2 begins with an explicit natural physical action by the subject (for example shoulders lift, the body rises, the head turns, or the subject steps/moves as supported by the endpoints) and joins all material/anatomical changes into ONE continuous progressive transformation. Do not merely change textures on a motionless subject. Describe a few precise visible changes with coordinated verbs such as softens, closes, separates, lengthens, opens, or forms. Do not schedule them as disconnected events.\n"
            "- Sentence 3 starts a continuous camera move at the same time as the transformation and follows it toward the exact RIGHT crop. Keep that camera motion active throughout instead of delaying it until late in the clip.\n"
            "- Sentence 4 describes the most distinctive RIGHT destination anatomy, object, material, color, pose, and focal detail forming gradually and coherently. Never replace an unusual endpoint with generic wording such as 'fleshy anatomy,' 'human visage,' or 'living form.'\n"
            "- Sentence 5 anchors shared environmental motion and explicitly keeps the setting, lighting, and spatial layout consistent.\n"
            "- Sentence 6 states that the transformation completes as the camera settles on the exact RIGHT destination, preserving continuous movement and coherent anatomy/structure throughout.\n"
            "- Both subject motion and camera travel begin immediately and proceed continuously. Do not hold one endpoint, introduce a second figure, overlay both compositions, or wait until the middle of the clip to reframe.\n"
            "- Never move, collapse, or recede facial features into another body region. Preserve coherent identity and anatomy while the camera moves away from any region excluded by the END crop.\n"
            "- When framing or camera angle differs, derive the camera direction strictly from the relative location of the RIGHT endpoint crop, then describe a precise continuous trajectory using direction and distance. If the LEFT shows a face and the RIGHT is centered lower on the neck, chest, shoulders, torso, or legs, the camera must tilt/pan DOWNWARD and must never say upward. If the RIGHT endpoint is physically above the LEFT focal region, only then may it say upward. The camera must end at the exact crop and focal region shown on the right. Explicitly name what remains visible at that endpoint and do not claim the face or full subject remains visible when the ending crop excludes it.\n"
            "- Preserve environmental elements that remain shared between the endpoints so the background does not reset unnecessarily.\n"
            "- Describe every change as progressive physical formation rather than a fade, crossfade, cut, overlay, double exposure, collage, split screen, or abrupt replacement.\n"
            "- Let the images decide the camera/character endpoints. Use motion/camera notes and speed/pacing guidance only to decide how quickly, smoothly, forcefully, or emotionally the subject and camera travel between those endpoints.\n"
            + transition_trigger_guidance
            +
            "- Do not include an aspect ratio, resolution, widescreen label, or phrases such as 'cinematic 16:9' in the final prompt. Width and height are controlled by the workflow settings.\n"
            "- Before answering, silently check that every camera-direction word moves toward the RIGHT crop and that no sentence contradicts the endpoint. Correct any upward/downward, closer/wider, or visible/out-of-frame contradiction.\n"
            "- Preserve any supplied singing, lyric, dialogue, emotional-performance, and selected facial-performance direction; integrate it naturally without disrupting the continuous visual motion.\n"
            "- Preserve the normal one-paragraph video prompt structure from the instructions. Do not mention images, frames, references, inputs, MSR, or LoRA in the final prompt.\n\n"
        )
        if flf_context_mode == "full" and any((flf_start_state, flf_transformation, flf_end_state, flf_carry_forward)):
            image_guidance += (
                "Storyboard FLF motion contract:\n"
                f"- Opening state: {flf_start_state or '[derive exactly from the LEFT image]'}\n"
                f"- Required continuous transformation: {flf_transformation or '[derive one continuous physical transition from both images]'}\n"
                f"- Required destination state: {flf_end_state or '[derive exactly from the RIGHT image]'}\n"
                f"- Carry-forward continuity: {flf_carry_forward or '[preserve all shared continuity]'}\n"
                "- The images remain the visual truth. The storyboard transformation is the primary motion plan and must be expressed as progressive physical action rather than replaced by generic morph language.\n"
                "- Begin the planned action and camera travel immediately, keep them continuous, and finish at the exact RIGHT destination.\n"
                "- Preserve all supplied singing, lyric, emotional-performance, and selected facial-performance directions while carrying out this transformation.\n"
                "- Do not output these labels or quote this contract in the final paragraph.\n\n"
            )
        elif flf_context_mode == "images_only":
            image_guidance += (
                "Gemma context mode: IMAGES ONLY.\n"
                "- Design the visual transition only from the locked LEFT and RIGHT observations.\n"
                "- Ignore lyrics, story arc, storyboard endpoint prose, mapped descriptions, scene notes, and camera presets when deciding visible action or camera travel.\n"
                "- The application adds any required exact vocal line and facial-performance direction after this visual prompt is returned.\n\n"
            )
        elif flf_context_mode == "images_story":
            image_guidance += (
                "Gemma context mode: IMAGES + STORY BEAT.\n"
                "- Use the locked LEFT and RIGHT observations as absolute endpoint truth.\n"
                "- Use only the supplied short scene story beat to choose a meaningful continuous bridge. Discard any beat detail that conflicts with either image.\n"
                "- Ignore lyrics, story arc, mapped descriptions, detailed endpoint prose, other scene notes, and camera presets when deciding visible action or camera travel.\n"
                "- The application adds any required exact vocal line and facial-performance direction after this visual prompt is returned.\n\n"
            )
    elif has_image_reference and frame_continuity_prompt:
        image_guidance = (
            "Vision attachment mapping:\n"
            "- Picture 1 is the previous render's final frame and the authoritative opening state. It is prompt-writing input, not a renderer Image N label.\n"
            "- Later pictures are the supporting inputs documented by the resolved Scene concept and cannot replace Picture 1's opening state.\n\n"
        )
    elif has_image_reference and is_minimax_h3_shot_json_task:
        image_guidance = (
            "MiniMax H3 JSON-shot visual-reference guidance:\n"
            "- Inspect the attached pictures only according to the exact assignments and limits in the Scene concept.\n"
            "- Use permitted visual facts inside the creative shot descriptions, but do not define pictures, subjects, audio, retention, continuity, or final prompt sections.\n"
            "- Return only the requested valid JSON object.\n\n"
        )
    elif has_image_reference and is_minimax_h3_prompt:
        image_guidance = (
            "MiniMax H3 prompt-only scene-inspiration guidance:\n"
            "- Attached <Picture 1> is vision input for prompt writing only. It is never supplied to the MiniMax renderer and must never appear as Image 1 or any Image N in the finished prompt.\n"
            "- Follow the scene context's exact environment-only or environment-plus-framing extraction limits for <Picture 1>. Always ignore all character identity, appearance, clothing, body, pose, placement, and activity visible in it.\n"
            "- Attached <Picture 2> corresponds to renderer Image 1, <Picture 3> to renderer Image 2, and so on. Inspect each renderer picture only for its assigned purpose.\n"
            "- In the finished prompt, use only renderer Image N labels. Convert permitted observations from <Picture 1> into direct scene prose without mentioning pictures, inspiration, analysis, or source imagery.\n"
            "- Do not merge a storyboard grid into a collage output; interpret its panels as ordered visual guidance.\n\n"
            if prompt_only_scene_inspiration
            else
            "MiniMax H3 ordered visual-reference guidance:\n"
            "- The attached pictures are in the exact <Picture 1>, <Picture 2>, and subsequent order stated in the scene context.\n"
            "- Inspect every attached picture and use it only for the purpose assigned to its matching tag.\n"
            "- Keep the exact <Picture N> tags in the final MiniMax prompt wherever the mode instructions require them.\n"
            "- Do not merge a storyboard grid into a collage output; interpret its panels as ordered visual guidance.\n\n"
        )
    elif has_image_reference and instruction_key == "id_lora":
        image_guidance = (
            "Use the provided image as the primary visual truth for the [VISUAL] section. "
            "Describe the visible subject, framing, setting, clothing/style, mood, and camera-ready action from the image, then adapt motion to the scene notes. "
            "Do not say 'reference image' in the final script.\n\n"
        )
    else:
        image_guidance = (
            "Use the provided reference image only to guide pose, framing, composition, mood, visible styling, or other user-requested visual details. "
            "Do not describe it as a reference image in the final prompt.\n\n"
            if has_image_reference else ""
        )
    prompt = (
        f"{t2v_instructions}\n\n"
        f"{image_guidance}"
        f"Scene concept:\n{scene_prompt}\n\n"
        f"User motion/camera notes:\n{user_notes or 'Create cinematic camera movement and natural subject/environment motion that fits the scene.'}"
    )

    llm_request_audit_path = ""
    llm_request_audit = None
    if is_minimax_h3_prompt:
        project_folder = os.path.abspath(str(payload.get("project_folder") or "").strip().strip('"')) if payload.get("project_folder") else ""
        if project_folder and os.path.isdir(project_folder):
            scene_id = _safe_project_name(str(payload.get("scene_id") or "scene"))
            audit_folder = os.path.join(project_folder, "llm_request_audits")
            llm_request_audit_path = os.path.join(audit_folder, f"last_minimax_h3_{scene_id}.json")
            llm_request_audit = {
                "saved_at": time.time(),
                "scene_id": str(payload.get("scene_id") or ""),
                "instruction_key": instruction_key,
                "performance_mode": str(payload.get("performance_mode") or ""),
                "audio_mode": str(payload.get("audio_mode") or ""),
                "singers": payload.get("singers") or [],
                "lyric_text": str(payload.get("lyric_text") or ""),
                "lyric_cue_map": payload.get("lyric_cue_map") or [],
                "performer_assignment": payload.get("performer_assignment") or {},
                "scene_concept": scene_prompt,
                "user_notes": user_notes,
                "actual_llm_instruction": prompt,
            }
            atomic_write_json(llm_request_audit_path, llm_request_audit)

    llm = _builder_local_llm(payload) if has_image_reference and text_runner not in _EXTERNAL_LLM_RUNNERS else None
    model_file = _builder_local_model_file(payload, model_file)
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION) if llm else ""
    mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file) if llm else ""
    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.7)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or 4000))
    unload_after = bool(payload.get("unload_after", True))

    try:
        model = None
        if has_image_reference and text_runner not in _EXTERNAL_LLM_RUNNERS:
            model = llm._load_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )

        def run_vision_instruction(instruction_text, token_limit):
            if text_runner in _EXTERNAL_LLM_RUNNERS:
                vision_payload = dict(payload)
                vision_payload["max_new_tokens"] = int(token_limit)
                return _try_run_remote_vision(
                    vision_payload,
                    instruction_text,
                    vision_images,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=int(token_limit),
                )
            result_text = llm._run_gguf_vision_pipeline(
                model=model,
                pil_images=vision_images,
                instruction_text=instruction_text,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=int(token_limit),
            )
            return result_text, {"runner": "builtin", "used_model": model_path, "unloaded": unload_after}

        if has_image_reference:
            if first_last_frame_mode:
                observation_prompt = (
                    "Inspect the side-by-side visual carefully. The LEFT half is START and the RIGHT half is END. "
                    "Do visual observation only; do not write a video prompt, story, transition, or camera direction. "
                    "Return exactly two dense labeled lines. Each line must literally inventory: subject identity; full-body posture (standing, sitting, kneeling, lying, leaning); front/back/profile orientation; head, torso, arm, and hand placement; location and scale within the frame; clothing and hair; all major props and surfaces; setting; lighting; exact crop; and every visible creature/object with its type and dominant color. "
                    "START: state all visible opening facts. END: state all visible destination facts and every major difference from START, including where the subject moved and which opening objects disappeared or remained. "
                    "Prioritize unusual anatomy, openings, cavities, translucent structures, distinctive materials, and which body regions are out of frame when present. "
                    "Do not merge the two halves, infer a story, rename one creature as another, or describe what ought to be present. State only what is visibly present."
                )
                locked_observation, _ = run_vision_instruction(observation_prompt, 900)
                locked_observation, missing_observations = _normalize_flf_vision_observation(locked_observation)
                if missing_observations:
                    retry_observation_prompt = (
                        observation_prompt
                        + "\n\nYour prior response did not provide two readable endpoint descriptions. Try again. "
                        "You MUST output both labeled lines, START: and END:. "
                        "Do not stop after START. Keep each line compact enough to fit."
                    )
                    locked_observation, _ = run_vision_instruction(retry_observation_prompt, 1200)
                    locked_observation, missing_observations = _normalize_flf_vision_observation(locked_observation)
                if missing_observations:
                    raise ValueError(
                        "FLF Gemma vision observation was incomplete after retry. "
                        "Missing readable description(s): "
                        + ", ".join(missing_observations)
                        + ". Both START and END descriptions are required."
                    )
                prompt += (
                    "\n\nLOCKED VISUAL OBSERVATION FROM THE REQUIRED FIRST PASS:\n"
                    f"{locked_observation}\n\n"
                    "Use these locked facts as literal visual truth. The final prompt must reach every distinctive END fact without replacing it with generic wording."
                    " If any lyric, storyboard transformation, motion note, or scene concept conflicts with the locked facts, discard the conflicting detail and write a continuous physical bridge that reaches the visible END instead."
                )
            text, run_info = run_vision_instruction(prompt, max_new_tokens)
        else:
            text, run_info = _run_builder_text_llm(
                payload,
                prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                label=f"{prompt_label} Gemma",
                preserve_paragraphs=is_minimax_h3_prompt,
            )
        if llm_request_audit is not None:
            llm_request_audit["raw_llm_response"] = str(text or "")
            atomic_write_json(llm_request_audit_path, llm_request_audit)
        text = _clean_lm_studio_plain_text(text) if is_minimax_h3_prompt else _clean_gemma_prompt_text(text)
        if is_minimax_h3_prompt and not is_minimax_h3_shot_json_task:
            text = _format_minimax_h3_prompt(text, payload, instruction_key)
        if first_last_frame_mode:
            text = re.sub(r"(?i)\b(?:cinematic\s+)?(?:aspect\s+ratio\s*[:=]?\s*)?(?:16\s*:\s*9|9\s*:\s*16|21\s*:\s*9|4\s*:\s*3|3\s*:\s*4|1\s*:\s*1)(?:\s+(?:aspect\s+ratio|widescreen|portrait|landscape))?\b[,]?\s*", "", text)
            text = re.sub(r"(?i)\b(?:widescreen|portrait|landscape)\s+aspect\s+ratio\b[,]?\s*", "", text)
            text = re.sub(r"(?i)\b(?:fades?|dissolves?|crossfades?|blends?)\s+(?:smoothly\s+|seamlessly\s+|gradually\s+)?into\b", "progressively transforms into", text)
            text = re.sub(r"(?i),?\s*(?:while\s+)?maintaining (?:her|his|their|the subject(?:'s)?) visibility throughout(?: the transformation)?", "", text)
            text = re.sub(r"\s{2,}", " ", text).strip(" ,")
        if is_minimax_h3_prompt:
            if not text:
                raise ValueError(f"{prompt_label} LLM returned an empty prompt.")
        else:
            text = _repair_and_validate_builder_gemma_prompt(payload, text, prompt_label)
        if first_last_frame_mode:
            text = re.sub(r"(?i)(?:\s*[,.;:-]?\s*)\bzhuanchang\b", "", text).strip(" ,.;:-")
            if transition_lora_active:
                text = f"{text}. zhuanchang"
        return {
            "prompt": text,
            "used_model": run_info.get("used_model", model_path if has_image_reference else ""),
            "used_mmproj": mmproj_path,
            "runner": run_info.get("runner", "builtin"),
            "used_image_reference": has_image_reference,
            "unloaded": run_info.get("unloaded", unload_after),
            "llm_request_audit_path": llm_request_audit_path,
        }
    finally:
        if llm and has_image_reference and unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _enhance_builder_video_prompt(payload):
    draft_prompt = str(payload.get("draft_prompt") or "").strip()
    if not draft_prompt:
        raise ValueError("Draft video prompt is empty.")
    model_file = str(payload.get("model_file") or payload.get("repair_model_file") or "").strip()
    if model_file:
        payload = dict(payload)
        payload["model_file"] = model_file
    instruction = _video_prompt_enhancement_instructions(payload)
    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("enhance_temperature") or 0.25),
        top_p=float(payload.get("enhance_top_p") or 0.9),
        max_new_tokens=int(payload.get("enhance_max_new_tokens") or 1200),
        label="I2V prompt enhancement",
    )
    text = _clean_gemma_prompt_text(text)
    text = _repair_and_validate_builder_gemma_prompt(payload, text, str(payload.get("mode_label") or "I2V"))
    return {
        "prompt": text,
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", bool(payload.get("unload_after", True))),
    }


def _edit_builder_video_prompt(payload):
    from .cache import _clear_vrgdg_llm_caches

    current_prompt = str(payload.get("current_prompt") or "").strip()
    if not current_prompt:
        raise ValueError("Current video prompt is empty.")
    model_file = str(payload.get("model_file") or payload.get("repair_model_file") or "").strip()
    mmproj_file = str(payload.get("mmproj_file") or "").strip()
    image_reference_path = str(payload.get("image_reference_path") or "").strip().strip('"')
    image_reference_data = str(payload.get("image_reference_data") or "").strip()
    use_vision_reference = bool(payload.get("use_vision_reference"))
    text_runner = _llm_runner_from_payload(payload)
    if model_file:
        payload = dict(payload)
        payload["model_file"] = model_file
    instruction = _video_prompt_edit_instructions(payload)
    image = None
    if use_vision_reference:
        if image_reference_data:
            image = _image_from_data_url(image_reference_data).convert("RGB")
        elif image_reference_path:
            image_path = _resolve_existing_file(image_reference_path, "Prompt edit starting image")
            image = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("Prompt edit requested the starting image, but no image reference was provided.")

    temperature = float(payload.get("temperature") or 0.25)
    top_p = float(payload.get("top_p") or 0.9)
    max_new_tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or 1200))
    unload_after = bool(payload.get("unload_after", True))
    n_ctx = int(payload.get("n_ctx") or 8000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format") or "").strip()
    seed = payload.get("seed")
    llm = None
    model_path = ""
    mmproj_path = ""
    try:
        if use_vision_reference and text_runner in _EXTERNAL_LLM_RUNNERS:
            text, run_info = _try_run_remote_vision(
                payload,
                instruction,
                [image],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        elif use_vision_reference:
            if not model_file:
                raise ValueError("Choose an I2V vision Gemma model first.")
            if not model_file.lower().endswith(".gguf"):
                raise ValueError("The I2V model field is not a GGUF model.")
            llm = _builder_local_llm(payload)
            model_file = _builder_local_model_file(payload, model_file)
            model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
            mmproj_file = _builder_local_mmproj_file(payload, mmproj_file)
            mmproj_path = _resolve_mmproj_dropdown_path(llm, mmproj_file)
            model = llm._load_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            text = llm._run_gguf_vision_pipeline(
                model=model,
                pil_images=[image],
                instruction_text=instruction,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=int(seed) if seed is not None else None,
            )
            run_info = {"runner": "builtin", "used_model": model_path, "unloaded": unload_after}
        else:
            text, run_info = _run_builder_text_llm(
                payload,
                instruction,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                label="video prompt edit",
            )
        text = _clean_gemma_prompt_text(text)
        text = _repair_and_validate_builder_gemma_prompt(payload, text, str(payload.get("mode_label") or "Video"))
        return {
            "prompt": text,
            "runner": run_info.get("runner", "builtin"),
            "used_model": run_info.get("used_model", model_path),
            "used_mmproj": mmproj_path,
            "used_image_reference": use_vision_reference,
            "unloaded": run_info.get("unloaded", unload_after),
        }
    finally:
        if llm and unload_after:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)
