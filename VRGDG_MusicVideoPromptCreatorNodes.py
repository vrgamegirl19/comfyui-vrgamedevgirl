import json
import copy
import math
import os
import re
import shutil
import time

import folder_paths
from aiohttp import web
from server import PromptServer

from .VRGDG_MusicVideoBuilderNodes import (
    _context_folder,
    _prompts_folder,
    _run_builder_text_llm,
    _llm_runner_from_payload,
    _safe_project_name,
    _session_path,
    _srt_path,
)
from .VRGDG_GeneralNodes2 import (
    _VRGDG_GEMMA4_LYRICS_INSTRUCTIONS,
    _VRGDG_GEMMA4_STYLE_INSTRUCTIONS,
    _VRGDG_GEMMA4_STORY_INSTRUCTIONS,
    _VRGDG_GEMMA4_SUBJECTS_INSTRUCTIONS,
)
from .VRGDG_GeneralNodes2 import (
    VRGDG_LyricSegmentJsonFixer,
    VRGDG_PromptMapJsonFixer,
)
from .VRGDG_VideoEditorNodes import _clean_gemma_prompt_text


_VRGDG_MUSIC_PROMPT_CREATOR_ROUTES_REGISTERED = False

_LLM_SETTINGS = {
    "n_ctx": 14848,
    "max_new_tokens": 32000,
    "temperature": 0.30,
    "top_p": 0.80,
    "n_gpu_layers": 99,
    "n_threads": 8,
    "chat_format": "",
}


_WHISPER_REPAIR_INSTRUCTIONS = r"""Whisper Segment Repair Prompt

INPUTS
You will receive exactly:
1. WHISPER_TRANSCRIPTION: numbered Whisper segments.
2. MAIN_LYRICS: the correct full lyrics.
3. STYLE_THEME: style/theme for filler replacements only.

TASK
Repair each Whisper segment in place.

MOST IMPORTANT RULE
Do not create a clean lyric sheet.
Do not copy MAIN_LYRICS line by line into the JSON.
Do not make segment1 the first lyric line, segment2 the second lyric line, segment3 the third lyric line, etc.

HOW TO WORK
For each input segment number:
1. Start with the text from that exact same Whisper segment.
2. Find the matching words in MAIN_LYRICS.
3. Fix only misheard words inside that same segment.
4. Write the fixed text to the same output number.

The output for segmentN must be a corrected version of lyricSegmentN.
segmentN must not be based on lyricSegmentN+1.
segmentN must not be based on lyricSegmentN-1.
segmentN must not be based on the next clean line in MAIN_LYRICS.

KEEP WHISPER CHUNKS
If lyricSegmentN contains parts of two or three lyric lines, segmentN must also contain those parts.
If lyricSegmentN starts in the middle of a lyric line, segmentN should also start in the middle.
If lyricSegmentN ends before the lyric line is finished, segmentN should also end before that lyric line is finished.
Do not move leftover words to the next segment unless Whisper already put them there.

FILLER SEGMENTS
Whisper often writes "Thank you." when it detects silence or non-lyric audio.
If lyricSegmentN is filler such as "Thank you." and does not match sung lyrics at that time, replace it with a short new lyric phrase.
The phrase should feel like it happens right after the nearest previous real lyric segment.
Use STYLE_THEME for the phrase.
Use 3 to 8 words.
Do not use an empty string.
Do not use the next real lyric line from MAIN_LYRICS.

KEY CHECK
Return exactly one key for every input segment.
Keys must be exactly "segment1", "segment2", "segment3", and so on.
Never write "segments14".
Never write "lyricSegment14".
Never skip a number.

FINAL SELF-CHECK BEFORE ANSWERING
Check segment2 against lyricSegment2.
Check segment11 against lyricSegment11.
Check segment16 against lyricSegment16.
If any output is just the next clean lyric line instead of the same Whisper chunk, fix it before answering.

OUTPUT
Return JSON only.
No markdown.
No explanation.
Use double quotes.
No trailing commas.
No line breaks inside string values.

FORMAT
{
  "segment1": "corrected version of lyricSegment1",
  "segment2": "corrected version of lyricSegment2",
  "segment3": "corrected version of lyricSegment3"
}"""


_WHISPER_BATCH_REPAIR_INSTRUCTIONS = r"""Repair a small batch of Whisper lyric segments by aligning them to a nearby real lyric window.

INPUTS
You will receive:
1. TARGET_WHISPER_SEGMENTS: the exact segment keys to repair.
2. REAL_LYRIC_WINDOW: nearby real lyrics in song order. Use only these lyric words for corrections.
3. PREVIOUS_REPAIRED_CONTEXT: already repaired segments just before this batch, for continuity only.

TASK
Return one corrected value for each TARGET_WHISPER_SEGMENTS key.

STRICT RULES
- Use only words from REAL_LYRIC_WINDOW for sung lyrics.
- You may fix punctuation, capitalization, spacing, and obvious partial-word breaks.
- Do not invent new lyrics.
- Do not use visual story, style/theme, mood, color, or setting words.
- If a Whisper segment is filler such as "Thank you." and no lyric from REAL_LYRIC_WINDOW belongs there, output "[instrumental]".
- INTRO FILLER EXCEPTION: When a short Whisper segment (1-3 words) at the very start of the song contains the beginning of a longer sung line whose remaining words appear in the next segment, that short segment is Whisper picking up the leading edge of the first vocal entry during a silent or instrumental intro. Output "[instrumental]" for it, even though its individual words appear in REAL_LYRIC_WINDOW. The test: would the full line in REAL_LYRIC_WINDOW still make sense if these few words moved to the start of the next segment? If yes, the short segment is intro filler.
- OUTRO FILLER EXCEPTION: Same pattern at the end of the song. A short trailing fragment (1-3 words) that duplicates the last few words of the previous segment's sung phrase, or appears after the last full sung line, is outro filler. Output "[instrumental]".
- If a filler segment clearly sits where a lyric from REAL_LYRIC_WINDOW belongs, use the matching real lyric words.
- Keep the segment count and key names exactly the same as TARGET_WHISPER_SEGMENTS.
- Keep the batch in song order. Do not jump backward to earlier lyrics outside REAL_LYRIC_WINDOW.
- Do not copy the whole lyric window into every segment.

OUTPUT
Return valid JSON only.
No markdown.
No explanation.
Use double quotes.
No trailing commas.
No line breaks inside string values.

FORMAT
{
  "segment10": "corrected lyric chunk",
  "segment11": "corrected lyric chunk"
}"""


_CONCEPT_PROMPT_INSTRUCTIONS = r"""You are a lyric-to-visual-concept converter.

INPUTS
You will receive:
1. LYRIC_SEGMENT_JSON: corrected lyric segments in order.
2. STORY: the overall story arc.
3. THEME_STYLE: visual style, mood, genre, world, and atmosphere.
4. SUBJECT: the main subject details, provided for downstream use only.
5. LOCATIONS: an optional list of locations or settings that should be used for the visual concepts.

TASK
Create one visual concept for each lyric segment.
These concepts will be sent to another LLM that writes the final text-to-image prompt.
Do not write the final image prompt here.

IMPORTANT
Do not describe the main subject.
Do not include character gender, hair, clothing, face, body, age, identity, or repeated subject details.
The SUBJECT input is provided for downstream use only and must be ignored when writing the concepts.
If a LOCATIONS list is provided, each concept must use one location from that list as its primary setting.
Do not invent a different primary location when LOCATIONS is provided.

STORY FLOW
Make the concepts feel like one continuous story sequence.
Each concept should feel like the next small beat after the previous one.
Show progression in action, location, emotion, stakes, or visual transformation.
When a LOCATIONS list is provided, build the story using locations from that list.
Prefer moving through the listed locations across the sequence in a way that feels like a journey or evolving story.
Avoid making every segment a disconnected literal illustration.
Do not repeat the same scene idea unless the lyrics repeat and the story beat needs to echo.

CONCEPT RULES
Use the matching lyric segment as the main source for the moment.
Use STORY to keep the scene connected across segments.
Use THEME_STYLE for mood, lighting, setting, color, genre, and surreal details.
Each concept must be one sentence.
Each concept must include a clear setting.
If LOCATIONS is provided, that setting must be one of the locations from the LOCATIONS list.
Focus on visible action, environment, emotional tone, props, symbols, and motion.
Make each concept useful for both image generation and later image-to-video motion.
Do not mention camera moves unless the lyric clearly needs motion.
Do not quote the lyric directly unless it is necessary.
Do not explain anything.

LYRIC ANCHOR RULES
Before writing each Prompt, silently identify the concrete anchors in that exact lyric segment:
- objects and places: window, rain, name, silence, flowers, table, thorns, room, heart, glass roses, floor, sugar, kiss, shoulder, bruise, door, shadow, mirror, monster, hands, petals, etc.
- actions and interactions: spelling, talking, answering, breathing, holding, trying not to bleed, cutting, falling, kissing, bruising, asking, whispering, walking out, etc.
- visible states: broken, sharp, poisonous, trembling, damaged, half out of mind, freedom, pain, etc.

Every non-instrumental Prompt must include at least one concrete object or action from its matching lyric segment.
If the lyric segment contains a specific object, that object must appear in the Prompt unless it is impossible to visualize.
If the lyric segment contains an action or interaction, the Prompt must show that action or a clear visual equivalent.
Do not replace lyric anchors with generic mood, glow, color, haze, landscape, or abstract atmosphere.
Do not write a concept that only describes lighting or scenery when the lyric contains an object or action.
Use THEME_STYLE to transform the lyric anchors visually, but never erase them.
For "Instrumental section." or other no-vocal placeholders, create a visual transition that follows STORY and THEME_STYLE; do not invent fake lyric objects.

LOCATION RULES
If LOCATIONS is provided, every Prompt value must begin with one exact location phrase copied from the LOCATIONS list, followed by a colon.
Do not use a location unless it appears in LOCATIONS.
Do not shorten, rename, paraphrase, or replace the location.
If the story needs to stay in the same place, reuse the same exact location phrase.

OUTPUT KEYS
Return one key for every input segment.
Use keys named "Prompt1", "Prompt2", "Prompt3", etc.
Never use "lyricSegment" keys.
Never skip, merge, split, or reorder prompts.

OUTPUT
Return valid JSON only.
No markdown.
No explanation.
Use double quotes.
No trailing commas.
No line breaks inside string values.

FORMAT
{
  "Prompt1": "Exact provided location: a short visual story beat that includes a concrete object or action from segment1, shaped by the story and theme, without describing the subject.",
  "Prompt2": "Exact provided location: the next connected visual story beat that includes a concrete object or action from segment2, continuing the story without repeating subject details."
}"""


_I2V_MOTION_NOTES_INSTRUCTIONS = r"""You are an image-to-video motion note writer.

INPUTS
You will receive:
1. CONCEPT_PROMPT_JSON: one visual concept per scene.
2. STORY: the overall story arc.
3. THEME_STYLE: visual style, mood, genre, world, and atmosphere.
4. SUBJECT: the main subject details, for character movement and performance only.

TASK
Create one short image-to-video motion note for each concept prompt.
These notes will be placed into per-scene I2V motion notes in the video builder.

RULES
- Write camera motion, character/performance motion, environmental motion, and mood pacing.
- Use the concept prompt as the main source for each scene.
- Use SUBJECT only for broad performance or body motion when useful.
- Do not rewrite the image prompt.
- Do not mention text, captions, lyrics, prompts, JSON, source images, or reference images.
- Keep each note practical for image-to-video generation.
- Keep each value one sentence, under 45 words.
- Avoid impossible object transformations unless the concept already implies surreal motion.
- If a scene is quiet or instrumental, use subtle camera/environment movement.

OUTPUT KEYS
Return one key for every input prompt.
Use keys named "Motion1", "Motion2", "Motion3", etc.
Never use Prompt keys.
Never skip, merge, split, or reorder notes.

OUTPUT
Return valid JSON only.
No markdown.
No explanation.
Use double quotes.
No trailing commas.
No line breaks inside string values.

FORMAT
{
  "Motion1": "Slow dolly toward the subject as background light drifts and small environmental details move gently.",
  "Motion2": "Wide lateral camera drift through the setting with subtle character movement and atmospheric motion."
}"""


_CONCEPT_MATCH_PRESETS = {
    "super_tight_literal": r"""LYRIC MATCH PRESET: SUPER TIGHT AND LITERAL
Every non-instrumental Prompt must visibly represent the exact lyric segment.
Use the lyric's concrete nouns and actions directly.
If the lyric says window, rain, flowers, thorns, glass roses, floor, kiss, shoulder, bruise, door, shadow, mirror, hands, petals, or walking out, those exact things must appear.
Do not substitute a different symbol when the lyric gives a concrete object.
Keep theme/style as surface treatment only; it must not override the lyric event.""",
    "medium": r"""LYRIC MATCH PRESET: MEDIUM
Each non-instrumental Prompt must include at least one concrete object or action from the lyric segment.
You may adapt the lyric into a cinematic visual metaphor, but the original lyric anchor must still be recognizable.
Balance the lyric event with STORY and THEME_STYLE.""",
    "loose": r"""LYRIC MATCH PRESET: LOOSE
Each Prompt should be inspired by the lyric segment's emotional intent, with at least one concrete lyric anchor when the segment gives a strong object or action.
You may prioritize STORY flow and THEME_STYLE over literal depiction when needed.
Avoid generic scenery; keep a visible connection to the lyric whenever possible.""",
    "super_light": r"""LYRIC MATCH PRESET: SUPER LIGHT
Use the lyric segment mostly as emotional timing.
Prioritize STORY flow, THEME_STYLE, and visual continuity.
Only include concrete lyric objects/actions when they naturally fit the visual sequence.
Instrumental and sparse lyric segments may become pure visual transitions.""",
}


def _concept_match_instructions(mode):
    key = str(mode or "medium").strip().lower()
    key = re.sub(r"[\s-]+", "_", key)
    return _CONCEPT_MATCH_PRESETS.get(key, _CONCEPT_MATCH_PRESETS["medium"])


_SUBJECT_EXTRACT_INSTRUCTIONS = r"""Extract only the subject from the user input.

Return one clean sentence in this format:
A/An [subject].

Rules:
- Ignore locations and all other fields.
- Preserve the subject details.
- Add commas only where they improve readability.
- End with a period.
- Do not add extra text.

Example input:
subject: female with blond hair wearing a red dress
locations:
woods
kitchen
van
beach

Example output:
A female with blond hair, wearing a red dress.

user input:"""


_PROMPT_CREATOR_INSTRUCTION_DEFAULTS = {
    "full_lyrics": _VRGDG_GEMMA4_LYRICS_INSTRUCTIONS,
    "style_theme": _VRGDG_GEMMA4_STYLE_INSTRUCTIONS,
    "story_idea": _VRGDG_GEMMA4_STORY_INSTRUCTIONS,
    "subject_locations": _VRGDG_GEMMA4_SUBJECTS_INSTRUCTIONS,
    "concept_prompts": _CONCEPT_PROMPT_INSTRUCTIONS,
    "subject_extract": _SUBJECT_EXTRACT_INSTRUCTIONS,
    "i2v_motion_notes": _I2V_MOTION_NOTES_INSTRUCTIONS,
}

_PROMPT_CREATOR_INSTRUCTION_LABELS = {
    "full_lyrics": "Full Lyrics",
    "style_theme": "Style / Theme",
    "story_idea": "Story Idea",
    "subject_locations": "Subject and Locations",
    "concept_prompts": "Concept Prompts",
    "subject_extract": "Subject Extraction",
    "i2v_motion_notes": "I2V Motion Notes",
}


def _safe_instruction_key(value):
    key = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    if key not in _PROMPT_CREATOR_INSTRUCTION_DEFAULTS:
        raise ValueError(f"Unknown Prompt Creator instruction key: {value}")
    return key


def _safe_preset_name(value):
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_. -]+", "_", text).strip(" ._")
    if not text:
        raise ValueError("Preset name is empty.")
    return text[:80]


def _instruction_folder(project_folder):
    return os.path.join(_context_folder(project_folder), "custom_llm_instructions")


def _instruction_path(project_folder, key):
    return os.path.join(_instruction_folder(project_folder), f"{_safe_instruction_key(key)}.txt")


def _instruction_preset_root():
    return os.path.join(folder_paths.get_output_directory(), "VRGDG_LLM_Instruction_Presets", "prompt_creator")


def _instruction_preset_path(key, name):
    return os.path.join(_instruction_preset_root(), _safe_instruction_key(key), f"{_safe_preset_name(name)}.txt")


def _prompt_creator_instruction(project_folder, key, default_text):
    path = _instruction_path(project_folder, key)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as handle:
                text = handle.read().strip()
            if text:
                return text
        except Exception:
            pass
    return str(default_text or "")


def _workflow_template_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Workflows",
        "UsedForUIDoNotTouch",
        "LTX2.3_Music_Video_Creator_Prompt_Creator_API.json",
    )


def _load_prompt_creator_workflow_template():
    workflow_path = os.path.abspath(_workflow_template_path())
    if not os.path.isfile(workflow_path):
        raise FileNotFoundError(f"Hidden Whisper workflow was not found: {workflow_path}")
    with open(workflow_path, "r", encoding="utf-8") as handle:
        prompt = json.load(handle)
    if not isinstance(prompt, dict) or not prompt:
        raise ValueError("Hidden Whisper workflow is not a valid ComfyUI API workflow JSON.")
    return workflow_path, prompt


def _project_folder_from_payload(payload):
    raw = str(payload.get("project_folder", "") or "").strip().strip('"')
    if raw:
        return os.path.abspath(raw)
    name = str(payload.get("project_name", "") or "").strip()
    if not name:
        name = f"VRGDG_Project_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    return os.path.join(folder_paths.get_output_directory(), _safe_project_name(name))


def _ensure_project_folders(project_folder):
    os.makedirs(project_folder, exist_ok=True)
    os.makedirs(_context_folder(project_folder), exist_ok=True)
    os.makedirs(_prompts_folder(project_folder), exist_ok=True)


def _project_audio_folder(project_folder):
    folder = os.path.join(project_folder, "audio")
    os.makedirs(folder, exist_ok=True)
    return folder


def _prompt_creator_debug_folder(project_folder):
    folder = os.path.join(project_folder, "prompt_creator_debug")
    os.makedirs(folder, exist_ok=True)
    return folder


def _write_prompt_creator_debug_file(project_folder, stem, content):
    try:
        folder = _prompt_creator_debug_folder(project_folder)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"{stamp}_{stem}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(str(content or ""))
        return path
    except Exception as exc:
        print(f"[PromptCreator] Failed to write debug file {stem}: {exc}")
        return ""


def _safe_file_name(name, fallback="vrgdg_audio.wav"):
    safe_name = re.sub(r'[<>:"/\\|?*]+', "_", os.path.basename(str(name or ""))).strip()
    return safe_name or fallback


def _stage_audio_for_upload_node(audio_path):
    raw_path = str(audio_path or "").strip().strip('"')
    if not raw_path:
        raise ValueError("Choose an audio file before running Prompt Creator.")

    source_path = os.path.abspath(raw_path)
    if not os.path.isfile(source_path):
        input_candidate = os.path.join(folder_paths.get_input_directory(), raw_path)
        if os.path.isfile(input_candidate):
            source_path = os.path.abspath(input_candidate)
        else:
            raise FileNotFoundError(f"Audio file was not found: {raw_path}")

    input_dir = folder_paths.get_input_directory()
    os.makedirs(input_dir, exist_ok=True)

    safe_name = _safe_file_name(
        source_path,
        f"vrgdg_prompt_creator_audio{os.path.splitext(source_path)[1] or '.wav'}",
    )

    staged_path = os.path.abspath(os.path.join(input_dir, safe_name))
    if os.path.abspath(source_path) != staged_path:
        if (
            not os.path.isfile(staged_path)
            or os.path.getsize(staged_path) != os.path.getsize(source_path)
            or int(os.path.getmtime(staged_path)) != int(os.path.getmtime(source_path))
        ):
            shutil.copy2(source_path, staged_path)

    return os.path.basename(staged_path), staged_path


def _clean_llm_json_text(text):
    cleaned = _clean_gemma_prompt_text(text)
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def _repair_json_like_text(text):
    repaired = str(text or "").strip()
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    repaired = re.sub(r"//.*?$", "", repaired, flags=re.MULTILINE)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(
        r'([{\[,]\s*)([A-Za-z_]*(?:Prompt|prompt|I2V|i2v|Motion|motion|segment|Segment|lyricSegment|LyricSegment|segments|Segments)\d+)\s*:',
        r'\1"\2":',
        repaired,
    )
    repaired = re.sub(
        r'(^\s*)([A-Za-z_]*(?:Prompt|prompt|I2V|i2v|Motion|motion|segment|Segment|lyricSegment|LyricSegment|segments|Segments)\d+)\s*:',
        r'\1"\2":',
        repaired,
        flags=re.MULTILINE,
    )
    return repaired


def _parse_json_like_key_value_lines(text):
    values = {}
    current_key = None
    current_parts = []
    key_pattern = re.compile(
        r'^\s*"?([A-Za-z_]*(?:Prompt|prompt|I2V|i2v|Motion|motion|segment|Segment|lyricSegment|LyricSegment|segments|Segments)\s*\d+)"?\s*[:=]\s*(.*?)(?:,\s*)?$'
    )
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line in ("{", "}", "[", "]"):
            continue
        match = key_pattern.match(line)
        if match:
            if current_key:
                values[current_key] = "\n".join(current_parts).strip().strip('"')
            current_key = match.group(1)
            value = match.group(2).strip().rstrip(",").strip()
            current_parts = [value.strip('"')]
            continue
        if current_key:
            current_parts.append(line.rstrip(",").strip('"'))
    if current_key:
        values[current_key] = "\n".join(current_parts).strip().strip('"')
    if not values:
        raise ValueError("Gemma did not return a JSON object.")
    return values


def _extract_json_object(text):
    cleaned = _clean_llm_json_text(text)
    candidates = [cleaned, _repair_json_like_text(cleaned)]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        sliced = cleaned[start:end + 1]
        candidates.extend([sliced, _repair_json_like_text(sliced)])
    last_error = None
    for candidate in candidates:
        if not str(candidate or "").strip():
            continue
        try:
            return json.loads(candidate)
        except Exception as error:
            last_error = error
    try:
        return _parse_json_like_key_value_lines(cleaned)
    except Exception:
        if last_error:
            raise last_error
        raise ValueError("Gemma did not return a JSON object.")


def _fix_lyric_segment_json_like_old_workflow(text):
    fixer = VRGDG_LyricSegmentJsonFixer()
    fixed_text, json_output, was_fixed, notes = fixer.fix_json(text)
    return {
        "fixed_text": fixed_text,
        "json_output": json_output,
        "was_fixed": was_fixed,
        "notes": notes,
    }


def _fix_prompt_map_json_like_old_workflow(text):
    fixer = VRGDG_PromptMapJsonFixer()
    fixed_text, json_output, was_fixed, notes, prompt_count = fixer.fix_json(text, False, "")
    return {
        "fixed_text": fixed_text,
        "json_output": json_output,
        "was_fixed": was_fixed,
        "notes": notes,
        "prompt_count": prompt_count,
    }


def _parse_whisper_segments(text):
    segments = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(?:lyricSegment|segment)?\s*(\d+)\s*[:=.-]\s*(.+)$", line, flags=re.IGNORECASE)
        if match:
            segments.append((int(match.group(1)), match.group(2).strip()))
    if not segments:
        raise ValueError("No numbered Whisper segments were found.")
    segments.sort(key=lambda item: item[0])
    return {f"lyricSegment{index}": value for index, value in segments}


def _whisper_segments_to_legacy_text(mapping):
    lines = []
    for key in sorted(mapping, key=lambda item: int(re.search(r"\d+", item).group(0))):
        lines.append(f"{key}={str(mapping.get(key, '') or '').strip()}")
    return "\n".join(lines)


def _split_real_lyric_lines(text):
    lines = []
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if re.match(r"^\s*(?:verse|chorus|bridge|intro|outro|pre[-\s]?chorus)\b", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    if not lines:
        compact = re.sub(r"\s+", " ", str(text or "")).strip()
        if compact:
            lines.append(compact)
    return lines


def _lyric_window_for_segment_batch(lyric_lines, start_index, end_index, total_segments, overlap=4):
    if not lyric_lines:
        return []
    total_lines = len(lyric_lines)
    start_ratio = max(0.0, (start_index - 1) / max(1, total_segments))
    end_ratio = min(1.0, end_index / max(1, total_segments))
    start_line = max(0, int(math.floor(start_ratio * total_lines)) - overlap)
    end_line = min(total_lines, int(math.ceil(end_ratio * total_lines)) + overlap)
    if end_line <= start_line:
        end_line = min(total_lines, start_line + 1)
    return [
        f"line{line_number + 1}={lyric_lines[line_number]}"
        for line_number in range(start_line, end_line)
    ]


def _segment_count_from_mapping(mapping):
    return len([key for key in mapping if re.match(r"^(?:lyricSegment|segment|segments)\s*\d+$", str(key), flags=re.IGNORECASE)])


def _canonical_segment_mapping(value):
    fixed = {}
    for raw_key, raw_value in (value or {}).items():
        match = re.match(r"^(?:lyricSegment|segment|segments)\s*(\d+)$", str(raw_key), flags=re.IGNORECASE)
        if match:
            fixed[f"segment{int(match.group(1))}"] = str(raw_value or "").strip()
    return {key: fixed[key] for key in sorted(fixed, key=lambda item: int(re.search(r"\d+", item).group(0)))}


def _payload_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on", "y"}:
        return True
    if text in {"false", "0", "no", "off", "n", ""}:
        return False
    return default


def _format_srt_timestamp(seconds):
    value = max(0.0, float(seconds or 0))
    whole = int(math.floor(value))
    millis = int(round((value - whole) * 1000))
    if millis >= 1000:
        whole += 1
        millis -= 1000
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _parse_srt_timestamp(value):
    match = re.match(r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*$", str(value or ""))
    if not match:
        return None
    hours, minutes, seconds, millis = [int(part) for part in match.groups()]
    return (hours * 3600) + (minutes * 60) + seconds + (millis / 1000.0)


def _srt_total_duration_hint(srt_text):
    last_end = None
    for match in re.finditer(r"-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})", str(srt_text or "")):
        parsed = _parse_srt_timestamp(match.group(1))
        if parsed is not None:
            last_end = parsed
    return last_end


def _fixed_duration_srt_from_segments(segments, fixed_scene_duration=4, total_duration_hint=None):
    canonical = _canonical_segment_mapping(segments)
    if not canonical:
        return ""
    duration = max(0.05, float(fixed_scene_duration or 4))
    total_hint = float(total_duration_hint or 0)
    lines = []
    start = 0.0
    items = list(canonical.items())
    for index, (_key, text) in enumerate(items, start=1):
        end = start + duration
        if index == len(items) and total_hint > start:
            end = total_hint
        lines.extend([
            str(index),
            f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}",
            str(text or "Instrumental section."),
            "",
        ])
        start = end
    return "\n".join(lines).rstrip() + "\n"


def _canonical_prompt_mapping(value):
    fixed = {}
    for raw_key, raw_value in (value or {}).items():
        match = re.match(r"^Prompt\s*(\d+)$", str(raw_key), flags=re.IGNORECASE)
        if match:
            fixed[f"Prompt{int(match.group(1))}"] = str(raw_value or "").strip()
    return {key: fixed[key] for key in sorted(fixed, key=lambda item: int(re.search(r"\d+", item).group(0)))}


def _is_scene_label_only_prompt_mapping(value):
    items = list((value or {}).items())
    if not items:
        return False
    for key, prompt in items:
        key_match = re.search(r"(\d+)", str(key or ""))
        value_match = re.match(r"^\s*scene\s*(\d+)\s*$", str(prompt or ""), flags=re.IGNORECASE)
        if not key_match or not value_match or int(key_match.group(1)) != int(value_match.group(1)):
            return False
    return True


def _prompt_map_key_number(key):
    match = re.search(r"(\d+)", str(key or ""))
    return int(match.group(1)) if match else 999999


def _normalize_inline_text(value):
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _strip_leading_subject(prompt, subjects):
    prompt_text = _normalize_inline_text(prompt)
    subject_list = [_normalize_inline_text(item) for item in (subjects or []) if _normalize_inline_text(item)]
    guard = 0
    changed = True
    while changed and guard < 8:
        changed = False
        guard += 1
        for subject_text in subject_list:
            if not prompt_text:
                break
            lower_prompt = prompt_text.lower()
            lower_subject = subject_text.lower()
            if lower_prompt == lower_subject:
                prompt_text = ""
                changed = True
                break
            if lower_prompt.startswith(lower_subject):
                prompt_text = prompt_text[len(subject_text):].lstrip()
                prompt_text = re.sub(r"^[,;:.-]\s*", "", prompt_text).strip()
                changed = True
                break
    return prompt_text


def _prepend_subject_to_prompts(prompts, subject, separator=", ", previous_subjects=None):
    subject_text = _normalize_inline_text(subject)
    if not subject_text or not isinstance(prompts, dict):
        return prompts

    known_subjects = [subject_text]
    if isinstance(previous_subjects, (list, tuple, set)):
        known_subjects.extend(previous_subjects)
    elif previous_subjects:
        known_subjects.append(previous_subjects)

    output = {}
    for key, value in prompts.items():
        prompt_text = _strip_leading_subject(value, known_subjects)
        if prompt_text:
            prompt_text = f"{subject_text}{separator}{prompt_text}"
        else:
            prompt_text = subject_text
        output[str(key)] = prompt_text
    return output


def _validate_segment_json(value, expected_count):
    if not isinstance(value, dict):
        raise ValueError("Segment output is not a JSON object.")
    indexed = {int(re.search(r"\d+", key).group(0)): raw_value for key, raw_value in _canonical_segment_mapping(value).items()}
    fixed = {}
    for index in range(1, int(expected_count) + 1):
        key = f"segment{index}"
        if index not in indexed:
            raise ValueError(f"Segment output is missing {key}.")
        text = str(indexed.get(index, "") or "").strip()
        if not text:
            raise ValueError(f"{key} is empty.")
        fixed[key] = text
    return fixed


def _validate_segment_subset(value, expected_keys):
    if not isinstance(value, dict):
        raise ValueError("Segment batch output is not a JSON object.")
    canonical = _canonical_segment_mapping(value)
    fixed = {}
    for key in expected_keys:
        text = str(canonical.get(key, "") or "").strip()
        if not text:
            raise ValueError(f"Segment batch output is missing {key}.")
        fixed[key] = text
    return fixed


def _segment_subset_with_fallback(value, expected_keys, target_segments):
    canonical = _canonical_segment_mapping(value) if isinstance(value, dict) else {}
    fixed = {}
    for key in expected_keys:
        text = str(canonical.get(key, "") or "").strip()
        if not text:
            original = str(target_segments.get(key, "") or "").strip()
            text = "[instrumental]" if re.fullmatch(r"(?:thank you\.?|thanks\.?|oh[,\s.]*)+", original, flags=re.IGNORECASE) else original
        if not text:
            text = "[instrumental]"
        fixed[key] = text
    return fixed


def _validate_prompt_json(value, expected_count):
    if not isinstance(value, dict):
        raise ValueError("Prompt output is not a JSON object.")
    indexed = {int(re.search(r"\d+", key).group(0)): raw_value for key, raw_value in _canonical_prompt_mapping(value).items()}
    fixed = {}
    for index in range(1, int(expected_count) + 1):
        key = f"Prompt{index}"
        if index not in indexed:
            raise ValueError(f"Prompt output is missing {key}.")
        text = str(indexed.get(index, "") or "").strip()
        if not text:
            raise ValueError(f"{key} is empty.")
        fixed[key] = text
    return fixed


def _missing_prompt_indices(value, expected_count):
    if not isinstance(value, dict):
        return list(range(1, int(expected_count) + 1))
    indexed = {}
    for key, raw_value in _canonical_prompt_mapping(value).items():
        match = re.search(r"\d+", key)
        if match:
            indexed[int(match.group(0))] = str(raw_value or "").strip()
    missing = []
    for index in range(1, int(expected_count) + 1):
        if not indexed.get(index):
            missing.append(index)
    return missing


def _split_subject_locations(value):
    text = str(value or "").strip()
    subject = text
    locations = ""
    subject_match = re.search(r"subject\s*:\s*(.+?)(?:\n\s*locations?\s*:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if subject_match:
        subject = subject_match.group(1).strip()
    locations_match = re.search(r"locations?\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if locations_match:
        locations = locations_match.group(1).strip()
    return subject, locations


def _run_text_gemma(model_file, prompt, overrides=None, payload=None):
    from .LLM import VRGDG_SuperGemmaGGUFChat, _clear_vrgdg_llm_caches

    runner_payload = dict(payload or {})
    runner = _llm_runner_from_payload(runner_payload)
    if runner != "builtin":
        settings = dict(_LLM_SETTINGS)
        if isinstance(overrides, dict):
            settings.update({key: value for key, value in overrides.items() if value not in (None, "")})
        text, run_info = _run_builder_text_llm(
            runner_payload,
            prompt,
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            max_new_tokens=int(settings["max_new_tokens"]),
            label="Gemma4",
            preserve_paragraphs=True,
        )
        text = _clean_llm_json_text(text)
        if not text:
            raise ValueError("Gemma returned an empty response.")
        return {"text": text, "used_model": run_info.get("used_model", ""), "runner": run_info.get("runner", runner)}

    if not str(model_file or "").strip():
        raise ValueError("Choose a Gemma4 text model first.")

    settings = dict(_LLM_SETTINGS)
    if isinstance(overrides, dict):
        settings.update({key: value for key, value in overrides.items() if value not in (None, "")})

    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(str(model_file), llm.MISSING_MODEL_OPTION)
    model = None
    try:
        model = llm._load_gguf_model(
            model_path=model_path,
            n_ctx=int(settings["n_ctx"]),
            n_gpu_layers=int(settings["n_gpu_layers"]),
            n_threads=int(settings["n_threads"]),
            chat_format=str(settings["chat_format"] or ""),
            mmproj_path="",
        )
        text = llm._run_gguf_text_pipeline(
            model=model,
            instruction_text=prompt,
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            max_new_tokens=int(settings["max_new_tokens"]),
        )
        text = _clean_llm_json_text(text)
        if not text:
            raise ValueError("Gemma returned an empty response.")
        return {"text": text, "used_model": model_path, "runner": "builtin"}
    finally:
        llm._unload_gguf_model(
            model_path=model_path,
            n_ctx=int(settings["n_ctx"]),
            n_gpu_layers=int(settings["n_gpu_layers"]),
            n_threads=int(settings["n_threads"]),
            chat_format=str(settings["chat_format"] or ""),
            mmproj_path="",
        )
        _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def _run_text_gemma_custom(model_file, custom_instructions, user_input, overrides=None, payload=None):
    from .LLM import VRGDG_SuperGemmaGGUFChat

    runner_payload = dict(payload or {})
    runner = _llm_runner_from_payload(runner_payload)
    if runner != "builtin":
        settings = dict(_LLM_SETTINGS)
        if isinstance(overrides, dict):
            settings.update({key: value for key, value in overrides.items() if value not in (None, "")})
        prompt = f"{str(custom_instructions or '').strip()}\n\nUser input:\n{str(user_input or '').strip()}"
        text, run_info = _run_builder_text_llm(
            runner_payload,
            prompt,
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            max_new_tokens=int(settings["max_new_tokens"]),
            label="Gemma4",
            preserve_paragraphs=True,
        )
        text = _clean_llm_json_text(text)
        if not text:
            raise ValueError("Gemma returned an empty response.")
        return {"text": text, "used_model": run_info.get("used_model", ""), "runner": run_info.get("runner", runner)}

    if not str(model_file or "").strip():
        raise ValueError("Choose a Gemma4 text model first.")

    settings = dict(_LLM_SETTINGS)
    if isinstance(overrides, dict):
        settings.update({key: value for key, value in overrides.items() if value not in (None, "")})

    llm = VRGDG_SuperGemmaGGUFChat()
    text, used_model, status = llm.generate_prompt(
        model_file=str(model_file),
        mmproj_file=llm.MISSING_MMPROJ_OPTION,
        task_preset="custom",
        user_input=str(user_input or ""),
        custom_instructions=str(custom_instructions or ""),
        trigger_word="",
        image_count=0,
        advanced=True,
        unload_after_run=True,
        n_ctx=int(settings["n_ctx"]),
        n_gpu_layers=int(settings["n_gpu_layers"]),
        n_threads=int(settings["n_threads"]),
        chat_format=str(settings["chat_format"] or ""),
        temperature=float(settings["temperature"]),
        top_p=float(settings["top_p"]),
        max_new_tokens=int(settings["max_new_tokens"]),
    )
    if str(status or "").strip().lower() != "ok":
        raise ValueError(str(status or "Gemma failed to run."))
    text = _clean_llm_json_text(text)
    if not text:
        raise ValueError("Gemma returned an empty response.")
    return {"text": text, "used_model": used_model, "runner": "builtin"}


def _repair_segments(payload):
    whisper_map = _parse_whisper_segments(payload.get("whisper_segments", ""))
    expected_count = len(whisper_map)
    lyric_lines = _split_real_lyric_lines(payload.get("full_lyrics", ""))
    batch_size = 8
    repaired_segments = {}
    raw_outputs = []
    fixer_notes = []
    repair_settings = dict(payload.get("llm_settings") or {})
    repair_settings.update({
        "temperature": 0.08,
        "top_p": 0.70,
        "max_new_tokens": min(6000, int(repair_settings.get("max_new_tokens") or _LLM_SETTINGS["max_new_tokens"])),
    })

    for batch_start in range(1, expected_count + 1, batch_size):
        batch_end = min(expected_count, batch_start + batch_size - 1)
        batch_keys = [f"segment{index}" for index in range(batch_start, batch_end + 1)]
        target_segments = {
            f"segment{index}": whisper_map.get(f"lyricSegment{index}", "")
            for index in range(batch_start, batch_end + 1)
        }
        previous_context = {
            f"segment{index}": repaired_segments.get(f"segment{index}", "")
            for index in range(max(1, batch_start - 3), batch_start)
            if repaired_segments.get(f"segment{index}")
        }
        lyric_window = _lyric_window_for_segment_batch(lyric_lines, batch_start, batch_end, expected_count)
        batch_input = (
            f"TARGET_WHISPER_SEGMENTS:\n{json.dumps(target_segments, ensure_ascii=False, indent=2)}\n\n"
            f"REAL_LYRIC_WINDOW:\n" + "\n".join(lyric_window) + "\n\n"
            f"PREVIOUS_REPAIRED_CONTEXT:\n{json.dumps(previous_context, ensure_ascii=False, indent=2)}"
        )
        result = _run_text_gemma_custom(payload.get("model_file", ""), _WHISPER_BATCH_REPAIR_INSTRUCTIONS, batch_input, repair_settings, payload)
        raw_outputs.append(result["text"])
        fixed = _fix_lyric_segment_json_like_old_workflow(result["text"])
        try:
            repaired_segments.update(_validate_segment_subset(fixed["json_output"], batch_keys))
        except Exception:
            retry_input = (
                f"{batch_input}\n\n"
                f"PREVIOUS_INVALID_ANSWER:\n{result['text']}\n\n"
                f"Return only these exact keys: {', '.join(batch_keys)}"
            )
            result = _run_text_gemma_custom(payload.get("model_file", ""), _WHISPER_BATCH_REPAIR_INSTRUCTIONS, retry_input, repair_settings, payload)
            raw_outputs.append(result["text"])
            fixed = _fix_lyric_segment_json_like_old_workflow(result["text"])
            repaired_segments.update(_segment_subset_with_fallback(fixed.get("json_output"), batch_keys, target_segments))
        if fixed.get("notes"):
            fixer_notes.append(str(fixed["notes"]))

    fixed = {
        "fixed_text": json.dumps(repaired_segments, indent=2, ensure_ascii=False),
        "json_output": repaired_segments,
        "was_fixed": True,
        "notes": "; ".join(fixer_notes) or "Batched lyric-window repair.",
    }
    retry_used = False
    if not isinstance(fixed.get("json_output"), dict):
        retry_used = True
        user_input = (
            f"WHISPER_TRANSCRIPTION:\n{_whisper_segments_to_legacy_text(whisper_map)}\n\n"
            f"MAIN_LYRICS:\n{str(payload.get('full_lyrics', '') or '').strip()}"
        )
        retry_instructions = (
            "Your previous answer was not valid JSON. Redo the task now.\n"
            "Return valid JSON only. No markdown. No explanation. No intro sentence.\n"
            f"Return exactly {expected_count} keys named segment1 through segment{expected_count}.\n"
            "Each value must be a corrected version of the matching WHISPER_TRANSCRIPTION chunk.\n"
            "Use double quotes and no trailing commas."
        )
        retry_input = (
            f"{user_input}\n\n"
            f"PREVIOUS_INVALID_ANSWER:\n{result['text']}"
        )
        result = _run_text_gemma_custom(payload.get("model_file", ""), retry_instructions, retry_input, payload.get("llm_settings"), payload)
        fixed = _fix_lyric_segment_json_like_old_workflow(result["text"])
    data = _validate_segment_json(fixed["json_output"], expected_count)
    return {
        "segments": data,
        "segment_count": expected_count,
        "raw_text": "\n\n--- BATCH ---\n\n".join(raw_outputs),
        "fixed_text": fixed["fixed_text"],
        "fixer_notes": fixed["notes"],
        "was_fixed": fixed["was_fixed"],
        "retry_used": retry_used,
        "used_model": result["used_model"],
        "unloaded": True,
    }


def _create_concepts(payload):
    segment_data = payload.get("segments")
    if isinstance(segment_data, str):
        segment_data = _extract_json_object(segment_data)
    if not isinstance(segment_data, dict):
        raise ValueError("Lyric segment JSON is required.")
    expected_count = _segment_count_from_mapping(segment_data)
    if expected_count <= 0:
        raise ValueError("Lyric segment JSON did not contain any segments.")
    subject_text, locations_text = _split_subject_locations(payload.get("subject_locations", ""))
    concept_match_mode = str(payload.get("concept_match_mode", "medium") or "medium")
    project_folder = _project_folder_from_payload(payload)
    concept_instructions = _prompt_creator_instruction(project_folder, "concept_prompts", _CONCEPT_PROMPT_INSTRUCTIONS)
    user_input = (
        f"{concept_instructions}\n\n"
        f"{_concept_match_instructions(concept_match_mode)}\n\n"
        f"LYRIC_SEGMENT_JSON:\n{json.dumps(segment_data, ensure_ascii=False, indent=2)}\n\n"
        f"STORY:\n{str(payload.get('story_idea', '') or '').strip()}\n\n"
        f"THEME_STYLE:\n{str(payload.get('style_theme', '') or '').strip()}\n\n"
        f"SUBJECT:\n{subject_text}\n\n"
        f"LOCATIONS:\n{locations_text}"
    )
    result = _run_text_gemma(payload.get("model_file", ""), user_input, payload.get("llm_settings"), payload)
    fixed = _fix_prompt_map_json_like_old_workflow(result["text"])
    retry_used = False
    try:
        data = _validate_prompt_json(fixed["json_output"], expected_count)
    except Exception:
        retry_used = True
        base_prompts = _canonical_prompt_mapping(fixed.get("json_output") if isinstance(fixed.get("json_output"), dict) else {})
        missing = _missing_prompt_indices(base_prompts, expected_count)
        repaired = dict(base_prompts)
        for missing_index in missing:
            context_start = max(1, missing_index - 3)
            context_end = min(expected_count, missing_index + 3)
            nearby_segments = {
                f"segment{index}": segment_data.get(f"segment{index}", "")
                for index in range(context_start, context_end + 1)
            }
            nearby_existing_prompts = {
                f"Prompt{index}": repaired.get(f"Prompt{index}", "")
                for index in range(context_start, context_end + 1)
                if repaired.get(f"Prompt{index}")
            }
            target_key = f"Prompt{missing_index}"
            repair_prompt = (
                "Create one missing visual concept prompt.\n"
                "Return valid JSON only. No markdown. No explanation. No intro sentence.\n"
                f"Return exactly one key named {target_key}.\n"
                "Do not return any other Prompt keys.\n"
                "The value must be one visual concept sentence for the matching lyric segment.\n"
                "Use nearby segments and existing prompts only for story continuity.\n"
                "Use double quotes and no trailing commas.\n\n"
                f"TARGET_SEGMENT_NUMBER:\n{missing_index}\n\n"
                f"TARGET_CORRECTED_LYRIC:\n{segment_data.get(f'segment{missing_index}', '')}\n\n"
                f"NEARBY_CORRECTED_LYRICS:\n{json.dumps(nearby_segments, ensure_ascii=False, indent=2)}\n\n"
                f"NEARBY_EXISTING_PROMPTS:\n{json.dumps(nearby_existing_prompts, ensure_ascii=False, indent=2)}\n\n"
                f"STORY:\n{str(payload.get('story_idea', '') or '').strip()}\n\n"
                f"THEME_STYLE:\n{str(payload.get('style_theme', '') or '').strip()}\n\n"
                f"SUBJECT:\n{subject_text}\n\n"
                f"LOCATIONS:\n{locations_text}"
            )
            repair_result = _run_text_gemma(payload.get("model_file", ""), repair_prompt, payload.get("llm_settings"), payload)
            repair_fixed = _fix_prompt_map_json_like_old_workflow(repair_result["text"])
            repair_map = _canonical_prompt_mapping(repair_fixed.get("json_output") if isinstance(repair_fixed.get("json_output"), dict) else {})
            repaired_text = str(repair_map.get(target_key, "") or "").strip()
            if not repaired_text:
                raise ValueError(f"Targeted repair did not return {target_key}.")
            repaired[target_key] = repaired_text
        try:
            data = _validate_prompt_json(repaired, expected_count)
            fixed["json_output"] = data
            fixed["fixed_text"] = json.dumps(data, indent=2, ensure_ascii=False)
            fixed["was_fixed"] = True
            fixed["notes"] = f"Targeted repair filled missing prompt(s): {', '.join(f'Prompt{i}' for i in missing)}"
        except Exception:
            retry_prompt = (
                "Your previous answer was not valid for the required prompt map. Redo the task now.\n"
                "Return valid JSON only. No markdown. No explanation. No intro sentence.\n"
                f"Return exactly {expected_count} keys named Prompt1 through Prompt{expected_count}.\n"
                "Do not skip any number. Do not add extra keys.\n"
                "Each value must be one visual concept sentence for the matching lyric segment.\n"
                "Use double quotes and no trailing commas.\n\n"
                f"LYRIC_SEGMENT_JSON:\n{json.dumps(segment_data, ensure_ascii=False, indent=2)}\n\n"
                f"STORY:\n{str(payload.get('story_idea', '') or '').strip()}\n\n"
                f"THEME_STYLE:\n{str(payload.get('style_theme', '') or '').strip()}\n\n"
                f"SUBJECT:\n{subject_text}\n\n"
                f"LOCATIONS:\n{locations_text}\n\n"
                f"PREVIOUS_INVALID_ANSWER:\n{result['text']}"
            )
            result = _run_text_gemma(payload.get("model_file", ""), retry_prompt, payload.get("llm_settings"), payload)
            fixed = _fix_prompt_map_json_like_old_workflow(result["text"])
            data = _validate_prompt_json(fixed["json_output"], expected_count)
    return {
        "prompts": data,
        "prompt_count": expected_count,
        "raw_text": result["text"],
        "fixed_text": fixed["fixed_text"],
        "fixer_notes": fixed["notes"],
        "was_fixed": fixed["was_fixed"],
        "retry_used": retry_used,
        "used_model": result["used_model"],
        "unloaded": True,
    }


def _create_i2v_motion_notes(payload):
    project_folder = _project_folder_from_payload(payload)
    _ensure_project_folders(project_folder)
    prompt_data = payload.get("prompts") or payload.get("concept_prompts")
    if isinstance(prompt_data, str):
        prompt_data = _extract_json_object(prompt_data)
    if not isinstance(prompt_data, dict):
        raise ValueError("ConceptPrompts JSON is required.")
    prompt_data = _canonical_prompt_mapping(prompt_data)
    expected_count = _segment_count_from_mapping({f"segment{_prompt_map_key_number(key)}": value for key, value in prompt_data.items()})
    if expected_count <= 0:
        raise ValueError("ConceptPrompts JSON did not contain any prompts.")
    motion_instructions = _prompt_creator_instruction(project_folder, "i2v_motion_notes", _I2V_MOTION_NOTES_INSTRUCTIONS)
    user_input = (
        f"{motion_instructions}\n\n"
        f"CONCEPT_PROMPT_JSON:\n{json.dumps(prompt_data, ensure_ascii=False, indent=2)}\n\n"
        f"STORY:\n{str(payload.get('story_idea', '') or '').strip()}\n\n"
        f"THEME_STYLE:\n{str(payload.get('style_theme', '') or '').strip()}\n\n"
        f"SUBJECT:\n{str(payload.get('subject', '') or '').strip()}"
    )
    debug_input_path = _write_prompt_creator_debug_file(project_folder, "i2v_motion_notes_input", user_input)
    result = _run_text_gemma(payload.get("model_file", ""), user_input, payload.get("llm_settings"), payload)
    debug_raw_path = _write_prompt_creator_debug_file(project_folder, "i2v_motion_notes_raw_output", result["text"])
    try:
        raw = _extract_json_object(result["text"])
        fixed_notes = "Parsed I2V motion notes JSON."
    except Exception:
        fixed = _fix_prompt_map_json_like_old_workflow(result["text"])
        raw = fixed.get("json_output") if isinstance(fixed.get("json_output"), dict) else {}
        fixed_notes = fixed.get("notes", "")
    data = {}
    for index in range(1, expected_count + 1):
        value = ""
        for key, item in raw.items():
            if _prompt_map_key_number(key) == index:
                value = str(item or "").strip()
                break
        if not value:
            value = "Subtle cinematic camera movement with gentle environmental motion that fits the scene."
        data[f"Motion{index}"] = value
    fallback_count = sum(
        1 for value in data.values()
        if value == "Subtle cinematic camera movement with gentle environmental motion that fits the scene."
    )
    if debug_raw_path:
        fixed_notes = f"{fixed_notes} Raw Gemma output saved to: {debug_raw_path}"
    if debug_input_path:
        fixed_notes = f"{fixed_notes} Input saved to: {debug_input_path}"
    if fallback_count:
        fixed_notes = f"{fixed_notes} Fallback motion notes used: {fallback_count}/{expected_count}."
    return {
        "motion_notes": data,
        "motion_count": expected_count,
        "raw_text": result["text"],
        "fixed_text": json.dumps(data, indent=2, ensure_ascii=False),
        "fixer_notes": fixed_notes,
        "debug_input_path": debug_input_path,
        "debug_raw_output_path": debug_raw_path,
        "fallback_count": fallback_count,
        "was_fixed": True,
        "used_model": result["used_model"],
        "unloaded": True,
    }


def _extract_subject(payload):
    subject_locations = str(payload.get("subject_locations", "") or "").strip()
    project_folder = _project_folder_from_payload(payload)
    subject_instructions = _prompt_creator_instruction(project_folder, "subject_extract", _SUBJECT_EXTRACT_INSTRUCTIONS)
    result = _run_text_gemma(
        payload.get("model_file", ""),
        f"{subject_instructions}\n{subject_locations}",
        payload.get("llm_settings"),
        payload,
    )
    text = _clean_gemma_prompt_text(result["text"])
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not line:
        raise ValueError("Gemma returned an empty subject.")
    return {
        "subject": line,
        "raw_text": result["text"],
        "used_model": result["used_model"],
        "unloaded": True,
    }


def _save_prompt_creator_outputs(payload):
    project_folder = _project_folder_from_payload(payload)
    _ensure_project_folders(project_folder)
    context = _context_folder(project_folder)
    prompts = _prompts_folder(project_folder)

    corrected_segments = payload.get("segments") or {}
    concept_prompts = payload.get("prompts") or {}
    i2v_motion_notes = payload.get("i2v_motion_notes") or {}
    if isinstance(corrected_segments, str) and corrected_segments.strip():
        corrected_segments = _extract_json_object(corrected_segments)
    if isinstance(concept_prompts, str) and concept_prompts.strip():
        concept_prompts = _extract_json_object(concept_prompts)
    if isinstance(i2v_motion_notes, str) and i2v_motion_notes.strip():
        i2v_motion_notes = _extract_json_object(i2v_motion_notes)
    if corrected_segments:
        corrected_segments = _canonical_segment_mapping(corrected_segments)
    if concept_prompts:
        concept_prompts = _canonical_prompt_mapping(concept_prompts)
        if _is_scene_label_only_prompt_mapping(concept_prompts):
            raise ValueError("ConceptPrompts only contains scene labels like SCENE 1. Create or paste real concept prompts before sending to Video Creator.")
        if _payload_bool(payload.get("append_subject_to_prompts", True), True):
            concept_prompts = _prepend_subject_to_prompts(
                concept_prompts,
                str(payload.get("subject", "") or ""),
                separator=", ",
                previous_subjects=[str(payload.get("previous_subject", "") or "")],
            )

    files = {}
    values = {
        os.path.join(context, "full_lyrics.txt"): str(payload.get("full_lyrics", "") or ""),
        os.path.join(context, "themestyle.txt"): str(payload.get("style_theme", "") or ""),
        os.path.join(context, "storyconcept.txt"): str(payload.get("story_idea", "") or ""),
        os.path.join(context, "subjectsandscenes.txt"): str(payload.get("subject_locations", "") or ""),
        os.path.join(context, "subject.txt"): str(payload.get("subject", "") or ""),
    }
    for path, value in values.items():
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(value)
        files[os.path.basename(path)] = path

    if corrected_segments:
        path = os.path.join(prompts, "lyric_segments.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(corrected_segments, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        files["lyric_segments.json"] = path
    if concept_prompts:
        path = os.path.join(context, "ConceptPrompts.txt")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(concept_prompts, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        files["ConceptPrompts.txt"] = path
    if i2v_motion_notes:
        path = os.path.join(context, "I2VMotionNotes.txt")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(i2v_motion_notes, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        files["I2VMotionNotes.txt"] = path

    use_srt_durations = _payload_bool(payload.get("use_srt_durations", True), True)
    fixed_scene_duration = float(payload.get("fixed_scene_duration", 4) or 4)
    srt_text = str(payload.get("srt_text", "") or "")
    if corrected_segments and not use_srt_durations:
        srt_text = _fixed_duration_srt_from_segments(
            corrected_segments,
            fixed_scene_duration,
            total_duration_hint=_srt_total_duration_hint(srt_text),
        )
    if srt_text.strip():
        with open(_srt_path(project_folder), "w", encoding="utf-8") as handle:
            handle.write(srt_text)
        files["builder_segments.srt"] = _srt_path(project_folder)

    marker_path = os.path.join(context, "prompt_creator_output.json")
    marker = {
        "type": "vrgdg_prompt_creator_output",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "has_concept_prompts": bool(concept_prompts),
        "has_i2v_motion_notes": bool(i2v_motion_notes),
        "has_srt": bool(srt_text.strip()),
    }
    with open(marker_path, "w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    files["prompt_creator_output.json"] = marker_path
    latest_path = os.path.join(folder_paths.get_output_directory(), "VRGDG_LastPromptCreatorProject.json")
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump({
            "type": "vrgdg_last_prompt_creator_project",
            "project_folder": project_folder,
            "context_folder": context,
            "saved_at": marker["saved_at"],
        }, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return {
        "project_folder": project_folder,
        "session_path": _session_path(project_folder),
        "srt_path": _srt_path(project_folder),
        "context_folder": context,
        "prompts_folder": prompts,
        "files": files,
    }


def _draft_path(project_folder):
    return os.path.join(project_folder, "prompt_creator_draft.json")


def _read_text_file_if_exists(path):
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return handle.read()
    except Exception:
        return ""


def _write_prompt_creator_pointer(project_folder, context, saved_at, marker=None):
    marker_path = os.path.join(context, "prompt_creator_output.json")
    marker_data = marker or {
        "type": "vrgdg_prompt_creator_output",
        "saved_at": saved_at,
        "has_concept_prompts": os.path.isfile(os.path.join(context, "ConceptPrompts.txt")),
        "has_i2v_motion_notes": os.path.isfile(os.path.join(context, "I2VMotionNotes.txt")),
        "has_srt": os.path.isfile(_srt_path(project_folder)),
    }
    with open(marker_path, "w", encoding="utf-8") as handle:
        json.dump(marker_data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    latest_path = os.path.join(folder_paths.get_output_directory(), "VRGDG_LastPromptCreatorProject.json")
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump({
            "type": "vrgdg_last_prompt_creator_project",
            "project_folder": project_folder,
            "context_folder": context,
            "saved_at": saved_at,
        }, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return marker_path


def _save_prompt_creator_draft(payload):
    project_folder = _project_folder_from_payload(payload)
    _ensure_project_folders(project_folder)
    context = _context_folder(project_folder)
    prompts = _prompts_folder(project_folder)
    saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
    draft = {
        "audio_path": str(payload.get("audio_path", "") or ""),
        "min_duration": payload.get("min_duration", 4),
        "max_duration": payload.get("max_duration", 10),
        "bias": payload.get("bias", 0.7),
        "duration_preset": str(payload.get("duration_preset", "varied_no_repeat") or "varied_no_repeat"),
        "use_srt_durations": _payload_bool(payload.get("use_srt_durations", True), True),
        "fixed_scene_duration": payload.get("fixed_scene_duration", 4),
        "empty_segment_text": str(payload.get("empty_segment_text", "Instrumental section.") or "Instrumental section."),
        "concept_match_mode": str(payload.get("concept_match_mode", "medium") or "medium"),
        "append_subject_to_prompts": _payload_bool(payload.get("append_subject_to_prompts", True), True),
        "repair_lyric_segments": _payload_bool(payload.get("repair_lyric_segments", False), False),
        "text_gemma_runner": str(payload.get("text_gemma_runner") or payload.get("text_runner") or "builtin"),
        "lm_studio_base_url": str(payload.get("lm_studio_base_url") or payload.get("lmstudio_base_url") or "http://127.0.0.1:1234/v1"),
        "lm_studio_model": str(payload.get("lm_studio_model") or payload.get("lmstudio_model") or ""),
        "lm_studio_api_key": "",
        "llm_api_provider": str(payload.get("llm_api_provider") or "openai"),
        "llm_api_model": str(payload.get("llm_api_model") or ""),
        "full_lyrics": str(payload.get("full_lyrics", "") or ""),
        "style_theme": str(payload.get("style_theme", "") or ""),
        "story_idea": str(payload.get("story_idea", "") or ""),
        "subject_locations": str(payload.get("subject_locations", "") or ""),
        "whisper_segments": str(payload.get("whisper_segments", "") or ""),
        "srt_text": str(payload.get("srt_text", "") or ""),
        "corrected_segments_text": str(payload.get("corrected_segments_text", "") or ""),
        "concept_prompts_text": str(payload.get("concept_prompts_text", "") or ""),
        "i2v_motion_notes_text": str(payload.get("i2v_motion_notes_text", "") or ""),
        "subject": str(payload.get("subject", "") or ""),
        "saved_at": saved_at,
    }
    path = _draft_path(project_folder)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(draft, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    files_written = {}
    context_values = {
        os.path.join(context, "full_lyrics.txt"): draft["full_lyrics"],
        os.path.join(context, "themestyle.txt"): draft["style_theme"],
        os.path.join(context, "storyconcept.txt"): draft["story_idea"],
        os.path.join(context, "subjectsandscenes.txt"): draft["subject_locations"],
        os.path.join(context, "subject.txt"): draft["subject"],
    }
    for context_path, value in context_values.items():
        with open(context_path, "w", encoding="utf-8") as handle:
            handle.write(str(value or ""))
        files_written[os.path.basename(context_path)] = context_path

    corrected_segments = {}
    if draft["corrected_segments_text"].strip():
        corrected_segments = _canonical_segment_mapping(_extract_json_object(draft["corrected_segments_text"]))
        if corrected_segments:
            lyric_path = os.path.join(prompts, "lyric_segments.json")
            with open(lyric_path, "w", encoding="utf-8") as handle:
                json.dump(corrected_segments, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            files_written["lyric_segments.json"] = lyric_path

    if draft["concept_prompts_text"].strip():
        concept_prompts = _canonical_prompt_mapping(_extract_json_object(draft["concept_prompts_text"]))
        if concept_prompts:
            if _is_scene_label_only_prompt_mapping(concept_prompts):
                raise ValueError("ConceptPrompts only contains scene labels like SCENE 1. Create or paste real concept prompts before saving.")
            concept_path = os.path.join(context, "ConceptPrompts.txt")
            with open(concept_path, "w", encoding="utf-8") as handle:
                json.dump(concept_prompts, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            files_written["ConceptPrompts.txt"] = concept_path

    if draft["i2v_motion_notes_text"].strip():
        raw_motion_notes = _extract_json_object(draft["i2v_motion_notes_text"])
        motion_notes = {}
        for raw_key, raw_value in (raw_motion_notes or {}).items():
            match = re.search(r"(\d+)", str(raw_key or ""))
            if match:
                motion_notes[f"Motion{int(match.group(1))}"] = str(raw_value or "").strip()
        if motion_notes:
            motion_path = os.path.join(context, "I2VMotionNotes.txt")
            with open(motion_path, "w", encoding="utf-8") as handle:
                json.dump(motion_notes, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            files_written["I2VMotionNotes.txt"] = motion_path

    srt_text = draft["srt_text"]
    regenerated_fixed_srt = False
    if corrected_segments and not draft["use_srt_durations"]:
        srt_text = _fixed_duration_srt_from_segments(
            corrected_segments,
            draft["fixed_scene_duration"],
            total_duration_hint=_srt_total_duration_hint(srt_text),
        )
        draft["srt_text"] = srt_text
        regenerated_fixed_srt = True

    if srt_text.strip():
        with open(_srt_path(project_folder), "w", encoding="utf-8") as handle:
            handle.write(srt_text)
        files_written["builder_segments.srt"] = _srt_path(project_folder)

    if regenerated_fixed_srt:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(draft, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    _write_prompt_creator_pointer(project_folder, _context_folder(project_folder), saved_at, {
        "type": "vrgdg_prompt_creator_output",
        "saved_at": saved_at,
        "from_draft": True,
        "has_concept_prompts": os.path.isfile(os.path.join(_context_folder(project_folder), "ConceptPrompts.txt")),
        "has_i2v_motion_notes": os.path.isfile(os.path.join(_context_folder(project_folder), "I2VMotionNotes.txt")),
        "has_srt": os.path.isfile(_srt_path(project_folder)),
    })
    return {
        "project_folder": project_folder,
        "draft_path": path,
        "draft": draft,
        "files": files_written,
    }


def _load_prompt_creator_draft(payload):
    project_folder = _project_folder_from_payload(payload)
    path = _draft_path(project_folder)
    if not os.path.isfile(path):
        context = _context_folder(project_folder)
        prompts = _prompts_folder(project_folder)
        synthetic = {
            "full_lyrics": _read_text_file_if_exists(os.path.join(context, "full_lyrics.txt")),
            "style_theme": _read_text_file_if_exists(os.path.join(context, "themestyle.txt")),
            "story_idea": _read_text_file_if_exists(os.path.join(context, "storyconcept.txt")),
            "subject_locations": _read_text_file_if_exists(os.path.join(context, "subjectsandscenes.txt")),
            "srt_text": _read_text_file_if_exists(_srt_path(project_folder)),
            "corrected_segments_text": _read_text_file_if_exists(os.path.join(prompts, "lyric_segments.json")),
            "concept_prompts_text": _read_text_file_if_exists(os.path.join(context, "ConceptPrompts.txt")),
            "i2v_motion_notes_text": _read_text_file_if_exists(os.path.join(context, "I2VMotionNotes.txt")),
            "subject": _read_text_file_if_exists(os.path.join(context, "subject.txt")).strip(),
        }
        if any(str(value or "").strip() for value in synthetic.values()):
            audio_folder = os.path.join(project_folder, "audio")
            audio_path = ""
            try:
                for filename in sorted(os.listdir(audio_folder), reverse=True):
                    candidate = os.path.join(audio_folder, filename)
                    if os.path.isfile(candidate) and filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg", ".mp4")):
                        audio_path = candidate
                        break
            except Exception:
                audio_path = ""
            synthetic["audio_path"] = audio_path
            synthetic["use_srt_durations"] = True
            synthetic["fixed_scene_duration"] = 4
            synthetic["empty_segment_text"] = "Instrumental section."
            synthetic["concept_match_mode"] = "medium"
            synthetic["append_subject_to_prompts"] = True
            return {
                "project_folder": project_folder,
                "draft_path": path,
                "found": True,
                "draft": synthetic,
                "synthetic": True,
            }
        return {
            "project_folder": project_folder,
            "draft_path": path,
            "found": False,
            "draft": {},
        }
    with open(path, "r", encoding="utf-8") as handle:
        draft = json.load(handle)
    if not isinstance(draft, dict):
        draft = {}
    return {
        "project_folder": project_folder,
        "draft_path": path,
        "found": True,
        "draft": draft,
    }


def _list_prompt_creator_drafts():
    output_dir = os.path.abspath(folder_paths.get_output_directory())
    projects = []
    if not os.path.isdir(output_dir):
        return {"projects": projects, "output_dir": output_dir}

    for name in os.listdir(output_dir):
        folder = os.path.join(output_dir, name)
        if not os.path.isdir(folder):
            continue

        draft_path = _draft_path(folder)
        context = _context_folder(folder)
        marker_path = os.path.join(context, "prompt_creator_output.json")
        concept_path = os.path.join(context, "ConceptPrompts.txt")
        i2v_path = os.path.join(context, "I2VMotionNotes.txt")
        srt_path = _srt_path(folder)

        has_draft = os.path.isfile(draft_path)
        has_marker = os.path.isfile(marker_path)
        has_outputs = os.path.isfile(concept_path) or os.path.isfile(i2v_path) or os.path.isfile(srt_path)
        if not (has_draft or has_marker or has_outputs):
            continue

        updated_candidates = []
        for candidate in (draft_path, marker_path, concept_path, i2v_path, srt_path):
            if os.path.isfile(candidate):
                try:
                    updated_candidates.append(os.path.getmtime(candidate))
                except OSError:
                    pass
        updated = max(updated_candidates) if updated_candidates else 0

        scene_count = 0
        if os.path.isfile(srt_path):
            try:
                with open(srt_path, "r", encoding="utf-8-sig") as handle:
                    scene_count = len(re.findall(r"(?m)^\s*\d+\s*$", handle.read()))
            except Exception:
                scene_count = 0
        if not scene_count and os.path.isfile(concept_path):
            try:
                with open(concept_path, "r", encoding="utf-8-sig") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    scene_count = len([key for key in data.keys() if re.match(r"^(?:Prompt|prompt)\d+$", str(key))])
            except Exception:
                scene_count = 0

        projects.append({
            "name": name,
            "project_folder": os.path.abspath(folder),
            "draft_path": os.path.abspath(draft_path) if has_draft else "",
            "context_folder": os.path.abspath(context),
            "updated": updated,
            "scene_count": scene_count,
            "has_draft": has_draft,
            "has_outputs": has_outputs,
        })

    projects.sort(key=lambda item: item.get("updated", 0), reverse=True)
    return {"projects": projects, "output_dir": output_dir}


def _get_prompt_creator_instruction(payload):
    project_folder = _project_folder_from_payload(payload)
    key = _safe_instruction_key(payload.get("key"))
    default_text = _PROMPT_CREATOR_INSTRUCTION_DEFAULTS[key]
    path = _instruction_path(project_folder, key)
    custom_text = _read_text_file_if_exists(path).strip()
    return {
        "project_folder": project_folder,
        "key": key,
        "label": _PROMPT_CREATOR_INSTRUCTION_LABELS.get(key, key),
        "default_text": default_text,
        "custom_text": custom_text,
        "text": custom_text or default_text,
        "has_custom": bool(custom_text),
        "path": path,
    }


def _save_prompt_creator_instruction(payload):
    project_folder = _project_folder_from_payload(payload)
    key = _safe_instruction_key(payload.get("key"))
    text = str(payload.get("text", "") or "").strip()
    if not text:
        raise ValueError("Instruction text is empty.")
    folder = _instruction_folder(project_folder)
    os.makedirs(folder, exist_ok=True)
    path = _instruction_path(project_folder, key)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
    return _get_prompt_creator_instruction({"project_folder": project_folder, "key": key})


def _reset_prompt_creator_instruction(payload):
    project_folder = _project_folder_from_payload(payload)
    key = _safe_instruction_key(payload.get("key"))
    path = _instruction_path(project_folder, key)
    if os.path.isfile(path):
        os.remove(path)
    return _get_prompt_creator_instruction({"project_folder": project_folder, "key": key})


def _list_prompt_creator_instruction_presets(payload):
    key = _safe_instruction_key(payload.get("key"))
    folder = os.path.join(_instruction_preset_root(), key)
    presets = []
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if not filename.lower().endswith(".txt"):
                continue
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            try:
                updated = os.path.getmtime(path)
            except OSError:
                updated = 0
            presets.append({
                "name": os.path.splitext(filename)[0],
                "path": os.path.abspath(path),
                "updated": updated,
            })
    presets.sort(key=lambda item: item.get("updated", 0), reverse=True)
    return {
        "key": key,
        "label": _PROMPT_CREATOR_INSTRUCTION_LABELS.get(key, key),
        "presets": presets,
        "preset_folder": folder,
    }


def _save_prompt_creator_instruction_preset(payload):
    key = _safe_instruction_key(payload.get("key"))
    name = _safe_preset_name(payload.get("name"))
    text = str(payload.get("text", "") or "").strip()
    if not text:
        raise ValueError("Preset instruction text is empty.")
    path = _instruction_preset_path(key, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
    return {"key": key, "name": name, "path": path}


def _load_prompt_creator_instruction_preset(payload):
    key = _safe_instruction_key(payload.get("key"))
    name = _safe_preset_name(payload.get("name"))
    path = _instruction_preset_path(key, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Instruction preset was not found: {path}")
    text = _read_text_file_if_exists(path).strip()
    if not text:
        raise ValueError("Instruction preset is empty.")
    return {"key": key, "name": name, "path": path, "text": text}


def _build_whisper_workflow_prompt(payload):
    workflow_path, prompt = _load_prompt_creator_workflow_template()
    prompt = copy.deepcopy(prompt)

    project_folder = _project_folder_from_payload(payload)
    _ensure_project_folders(project_folder)

    audio_path = str(payload.get("audio_path", "") or payload.get("audio_file", "") or "").strip().strip('"')
    audio_upload_name, staged_audio_path = _stage_audio_for_upload_node(audio_path)

    min_duration = float(payload.get("min_duration", 4) or 4)
    max_duration = float(payload.get("max_duration", 10) or 10)
    bias = float(payload.get("bias", 0.7) or 0.7)
    duration_preset = str(payload.get("duration_preset", "varied_no_repeat") or "varied_no_repeat")
    use_srt_durations = _payload_bool(payload.get("use_srt_durations", True), True)
    fixed_scene_duration = float(payload.get("fixed_scene_duration", 4) or 4)
    empty_segment_text = str(payload.get("empty_segment_text", "Instrumental section.") or "Instrumental section.").strip() or "Instrumental section."
    whisper_language = str(payload.get("whisper_language", "english") or "english").strip() or "english"
    full_lyrics = str(payload.get("full_lyrics", "") or "")
    output_filename = f"builder_segments_{time.strftime('%Y%m%d_%H%M%S')}.srt"

    def node(node_id):
        key = str(node_id)
        if key not in prompt:
            raise KeyError(f"Hidden Whisper workflow node {key} was not found.")
        return prompt[key]

    if "954" in prompt:
        node(954).setdefault("inputs", {})["audio"] = audio_upload_name
    elif "964" in prompt:
        node(964).setdefault("inputs", {})["audio"] = audio_upload_name

    if "28:114" in prompt:
        node("28:114").setdefault("inputs", {})["audio_file_path"] = staged_audio_path
    elif "955" in prompt and "audio_file_path" in node(955).setdefault("inputs", {}):
        node(955).setdefault("inputs", {})["audio_file_path"] = staged_audio_path

    if "955" in prompt and prompt["955"].get("class_type") == "VRGDG_TextBox":
        node(955).setdefault("inputs", {})["text"] = full_lyrics

    if "960" in prompt:
        extractor_inputs = node(960).setdefault("inputs", {})
        extractor_inputs["scene_duration_seconds"] = fixed_scene_duration
        extractor_inputs["reference_lyrics"] = full_lyrics
        extractor_inputs["language"] = whisper_language

    if "28:933" in prompt:
        node("28:933").setdefault("inputs", {})["switch"] = use_srt_durations

    if "28:887" in prompt:
        node("28:887").setdefault("inputs", {})["use_srt_durations"] = use_srt_durations

    if "28:920" in prompt:
        node("28:920").setdefault("inputs", {})["use_srt_file"] = use_srt_durations

    if "28:949" in prompt:
        node("28:949").setdefault("inputs", {})["empty_segment_text"] = empty_segment_text

    duration_node_id = "28:80" if "28:80" in prompt else "963"
    duration_inputs = node(duration_node_id).setdefault("inputs", {})
    duration_inputs["min_duration"] = min_duration
    duration_inputs["max_duration"] = max_duration
    duration_inputs["bias"] = bias
    duration_inputs["duration_preset"] = duration_preset
    duration_inputs["output_filename"] = output_filename

    return {
        "workflow_template_path": workflow_path,
        "prompt": prompt,
        "project_folder": project_folder,
        "expected_srt_path": _srt_path(project_folder),
        "source_srt_filename": output_filename,
    }


async def _import_prompt_creator_audio(request):
    reader = await request.multipart()
    project_folder = ""
    audio_part = None

    async for part in reader:
        if part.name == "project_folder":
            project_folder = (await part.text()).strip().strip('"')
        elif part.name == "audio":
            audio_part = part
            break

    project_folder = os.path.abspath(project_folder) if project_folder else _project_folder_from_payload({})
    _ensure_project_folders(project_folder)

    if audio_part is None:
        raise ValueError("No audio file was received.")

    original_name = audio_part.filename or "prompt_creator_audio.wav"
    safe_name = _safe_file_name(original_name, "prompt_creator_audio.wav")
    save_path = os.path.abspath(os.path.join(_project_audio_folder(project_folder), safe_name))

    with open(save_path, "wb") as handle:
        while True:
            chunk = await audio_part.read_chunk()
            if not chunk:
                break
            handle.write(chunk)

    if not os.path.isfile(save_path) or os.path.getsize(save_path) <= 0:
        raise ValueError("Audio import failed because the saved file is empty.")

    return {
        "project_folder": project_folder,
        "audio_path": save_path,
        "audio_name": safe_name,
    }


def _json_response(result=None, error=None, status=200):
    if error:
        return web.json_response({"ok": False, "error": str(error)}, status=status)
    data = {"ok": True}
    if isinstance(result, dict):
        data.update(result)
    elif result is not None:
        data["result"] = result
    return web.json_response(data)


def _register_routes():
    global _VRGDG_MUSIC_PROMPT_CREATOR_ROUTES_REGISTERED
    if _VRGDG_MUSIC_PROMPT_CREATOR_ROUTES_REGISTERED:
        return
    server_instance = PromptServer.instance
    if server_instance is None:
        return

    @server_instance.routes.get("/vrgdg/music_prompt_creator/config")
    async def vrgdg_music_prompt_creator_config(_request):
        path = _workflow_template_path()
        return _json_response({
            "workflow_template_path": path,
            "workflow_template_exists": os.path.isfile(path),
            "llm_settings": dict(_LLM_SETTINGS),
        })

    @server_instance.routes.post("/vrgdg/music_prompt_creator/repair_segments")
    async def vrgdg_music_prompt_creator_repair_segments(request):
        try:
            return _json_response(_repair_segments(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/create_concepts")
    async def vrgdg_music_prompt_creator_create_concepts(request):
        try:
            return _json_response(_create_concepts(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/extract_subject")
    async def vrgdg_music_prompt_creator_extract_subject(request):
        try:
            return _json_response(_extract_subject(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/create_i2v_motion_notes")
    async def vrgdg_music_prompt_creator_create_i2v_motion_notes(request):
        try:
            return _json_response(_create_i2v_motion_notes(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/save_outputs")
    async def vrgdg_music_prompt_creator_save_outputs(request):
        try:
            return _json_response(_save_prompt_creator_outputs(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/save_draft")
    async def vrgdg_music_prompt_creator_save_draft(request):
        try:
            return _json_response(_save_prompt_creator_draft(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/load_draft")
    async def vrgdg_music_prompt_creator_load_draft(request):
        try:
            return _json_response(_load_prompt_creator_draft(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.get("/vrgdg/music_prompt_creator/list_drafts")
    async def vrgdg_music_prompt_creator_list_drafts(_request):
        try:
            return _json_response(_list_prompt_creator_drafts())
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/get_instruction")
    async def vrgdg_music_prompt_creator_get_instruction(request):
        try:
            return _json_response(_get_prompt_creator_instruction(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/save_instruction")
    async def vrgdg_music_prompt_creator_save_instruction(request):
        try:
            return _json_response(_save_prompt_creator_instruction(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/reset_instruction")
    async def vrgdg_music_prompt_creator_reset_instruction(request):
        try:
            return _json_response(_reset_prompt_creator_instruction(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/list_instruction_presets")
    async def vrgdg_music_prompt_creator_list_instruction_presets(request):
        try:
            return _json_response(_list_prompt_creator_instruction_presets(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/save_instruction_preset")
    async def vrgdg_music_prompt_creator_save_instruction_preset(request):
        try:
            return _json_response(_save_prompt_creator_instruction_preset(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/load_instruction_preset")
    async def vrgdg_music_prompt_creator_load_instruction_preset(request):
        try:
            return _json_response(_load_prompt_creator_instruction_preset(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/build_whisper_prompt")
    async def vrgdg_music_prompt_creator_build_whisper_prompt(request):
        try:
            return _json_response(_build_whisper_workflow_prompt(await request.json()))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    @server_instance.routes.post("/vrgdg/music_prompt_creator/import_audio")
    async def vrgdg_music_prompt_creator_import_audio(request):
        try:
            return _json_response(await _import_prompt_creator_audio(request))
        except Exception as exc:
            return _json_response(error=exc, status=400)

    _VRGDG_MUSIC_PROMPT_CREATOR_ROUTES_REGISTERED = True


_register_routes()


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
