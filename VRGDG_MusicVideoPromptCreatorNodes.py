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
    _safe_project_name,
    _session_path,
    _srt_path,
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
        r'([{\[,]\s*)([A-Za-z_]*(?:Prompt|prompt|segment|Segment|lyricSegment|LyricSegment|segments|Segments)\d+)\s*:',
        r'\1"\2":',
        repaired,
    )
    repaired = re.sub(
        r'(^\s*)([A-Za-z_]*(?:Prompt|prompt|segment|Segment|lyricSegment|LyricSegment|segments|Segments)\d+)\s*:',
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
        r'^\s*"?([A-Za-z_]*(?:Prompt|prompt|segment|Segment|lyricSegment|LyricSegment|segments|Segments)\s*\d+)"?\s*[:=]\s*(.*?)(?:,\s*)?$'
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


def _canonical_prompt_mapping(value):
    fixed = {}
    for raw_key, raw_value in (value or {}).items():
        match = re.match(r"^Prompt\s*(\d+)$", str(raw_key), flags=re.IGNORECASE)
        if match:
            fixed[f"Prompt{int(match.group(1))}"] = str(raw_value or "").strip()
    return {key: fixed[key] for key in sorted(fixed, key=lambda item: int(re.search(r"\d+", item).group(0)))}


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


def _run_text_gemma(model_file, prompt, overrides=None):
    from .LLM import VRGDG_SuperGemmaGGUFChat, _clear_vrgdg_llm_caches

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
        return {"text": text, "used_model": model_path}
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


def _run_text_gemma_custom(model_file, custom_instructions, user_input, overrides=None):
    from .LLM import VRGDG_SuperGemmaGGUFChat

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
    return {"text": text, "used_model": used_model}


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
        result = _run_text_gemma_custom(payload.get("model_file", ""), _WHISPER_BATCH_REPAIR_INSTRUCTIONS, batch_input, repair_settings)
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
            result = _run_text_gemma_custom(payload.get("model_file", ""), _WHISPER_BATCH_REPAIR_INSTRUCTIONS, retry_input, repair_settings)
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
        result = _run_text_gemma_custom(payload.get("model_file", ""), retry_instructions, retry_input, payload.get("llm_settings"))
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
    user_input = (
        f"{_CONCEPT_PROMPT_INSTRUCTIONS}\n\n"
        f"{_concept_match_instructions(concept_match_mode)}\n\n"
        f"LYRIC_SEGMENT_JSON:\n{json.dumps(segment_data, ensure_ascii=False, indent=2)}\n\n"
        f"STORY:\n{str(payload.get('story_idea', '') or '').strip()}\n\n"
        f"THEME_STYLE:\n{str(payload.get('style_theme', '') or '').strip()}\n\n"
        f"SUBJECT:\n{subject_text}\n\n"
        f"LOCATIONS:\n{locations_text}"
    )
    result = _run_text_gemma(payload.get("model_file", ""), user_input, payload.get("llm_settings"))
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
            repair_result = _run_text_gemma(payload.get("model_file", ""), repair_prompt, payload.get("llm_settings"))
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
            result = _run_text_gemma(payload.get("model_file", ""), retry_prompt, payload.get("llm_settings"))
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


def _extract_subject(payload):
    subject_locations = str(payload.get("subject_locations", "") or "").strip()
    result = _run_text_gemma(
        payload.get("model_file", ""),
        f"{_SUBJECT_EXTRACT_INSTRUCTIONS}\n{subject_locations}",
        payload.get("llm_settings"),
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
    if isinstance(corrected_segments, str) and corrected_segments.strip():
        corrected_segments = _extract_json_object(corrected_segments)
    if isinstance(concept_prompts, str) and concept_prompts.strip():
        concept_prompts = _extract_json_object(concept_prompts)
    if corrected_segments:
        corrected_segments = _canonical_segment_mapping(corrected_segments)
    if concept_prompts:
        concept_prompts = _canonical_prompt_mapping(concept_prompts)

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

    srt_text = str(payload.get("srt_text", "") or "")
    if srt_text.strip():
        with open(_srt_path(project_folder), "w", encoding="utf-8") as handle:
            handle.write(srt_text)
        files["builder_segments.srt"] = _srt_path(project_folder)

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


def _save_prompt_creator_draft(payload):
    project_folder = _project_folder_from_payload(payload)
    _ensure_project_folders(project_folder)
    draft = {
        "audio_path": str(payload.get("audio_path", "") or ""),
        "min_duration": payload.get("min_duration", 4),
        "max_duration": payload.get("max_duration", 10),
        "bias": payload.get("bias", 0.7),
        "duration_preset": str(payload.get("duration_preset", "varied_no_repeat") or "varied_no_repeat"),
        "use_srt_durations": bool(payload.get("use_srt_durations", True)),
        "fixed_scene_duration": payload.get("fixed_scene_duration", 4),
        "empty_segment_text": str(payload.get("empty_segment_text", "Instrumental section.") or "Instrumental section."),
        "concept_match_mode": str(payload.get("concept_match_mode", "medium") or "medium"),
        "full_lyrics": str(payload.get("full_lyrics", "") or ""),
        "style_theme": str(payload.get("style_theme", "") or ""),
        "story_idea": str(payload.get("story_idea", "") or ""),
        "subject_locations": str(payload.get("subject_locations", "") or ""),
        "whisper_segments": str(payload.get("whisper_segments", "") or ""),
        "srt_text": str(payload.get("srt_text", "") or ""),
        "corrected_segments_text": str(payload.get("corrected_segments_text", "") or ""),
        "concept_prompts_text": str(payload.get("concept_prompts_text", "") or ""),
        "subject": str(payload.get("subject", "") or ""),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = _draft_path(project_folder)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(draft, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return {
        "project_folder": project_folder,
        "draft_path": path,
        "draft": draft,
    }


def _load_prompt_creator_draft(payload):
    project_folder = _project_folder_from_payload(payload)
    path = _draft_path(project_folder)
    if not os.path.isfile(path):
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
    use_srt_durations = bool(payload.get("use_srt_durations", True))
    fixed_scene_duration = float(payload.get("fixed_scene_duration", 4) or 4)
    empty_segment_text = str(payload.get("empty_segment_text", "Instrumental section.") or "Instrumental section.").strip() or "Instrumental section."
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
