import json
import copy
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
  "Prompt1": "A short visual story beat using one provided location as the setting, with action, mood, and image-to-video friendly motion, without describing the subject.",
  "Prompt2": "The next connected visual story beat using one provided location as the setting, continuing the previous moment without repeating subject details."
}"""


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
        "whipsterandbeatonly_API.json",
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


def _repair_segments(payload):
    whisper_map = _parse_whisper_segments(payload.get("whisper_segments", ""))
    expected_count = len(whisper_map)
    user_input = (
        f"{_WHISPER_REPAIR_INSTRUCTIONS}\n\n"
        f"WHISPER_TRANSCRIPTION:\n{json.dumps(whisper_map, ensure_ascii=False, indent=2)}\n\n"
        f"MAIN_LYRICS:\n{str(payload.get('full_lyrics', '') or '').strip()}\n\n"
        f"STYLE_THEME:\n{str(payload.get('style_theme', '') or '').strip()}"
    )
    result = _run_text_gemma(payload.get("model_file", ""), user_input, payload.get("llm_settings"))
    fixed = _fix_lyric_segment_json_like_old_workflow(result["text"])
    data = _validate_segment_json(fixed["json_output"], expected_count)
    return {
        "segments": data,
        "segment_count": expected_count,
        "raw_text": result["text"],
        "fixed_text": fixed["fixed_text"],
        "fixer_notes": fixed["notes"],
        "was_fixed": fixed["was_fixed"],
        "used_model": result["used_model"],
        "unloaded": True,
    }


def _create_concepts(payload):
    segment_data = payload.get("segments")
    if isinstance(segment_data, str):
        segment_data = _extract_json_object(segment_data)
    if not isinstance(segment_data, dict):
        raise ValueError("Corrected lyric segment JSON is required.")
    expected_count = _segment_count_from_mapping(segment_data)
    if expected_count <= 0:
        raise ValueError("Corrected lyric segment JSON did not contain any segments.")
    subject_text, locations_text = _split_subject_locations(payload.get("subject_locations", ""))
    user_input = (
        f"{_CONCEPT_PROMPT_INSTRUCTIONS}\n\n"
        f"LYRIC_SEGMENT_JSON:\n{json.dumps(segment_data, ensure_ascii=False, indent=2)}\n\n"
        f"STORY:\n{str(payload.get('story_idea', '') or '').strip()}\n\n"
        f"THEME_STYLE:\n{str(payload.get('style_theme', '') or '').strip()}\n\n"
        f"SUBJECT:\n{subject_text}\n\n"
        f"LOCATIONS:\n{locations_text}"
    )
    result = _run_text_gemma(payload.get("model_file", ""), user_input, payload.get("llm_settings"))
    fixed = _fix_prompt_map_json_like_old_workflow(result["text"])
    data = _validate_prompt_json(fixed["json_output"], expected_count)
    return {
        "prompts": data,
        "prompt_count": expected_count,
        "raw_text": result["text"],
        "fixed_text": fixed["fixed_text"],
        "fixer_notes": fixed["notes"],
        "was_fixed": fixed["was_fixed"],
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
    output_filename = f"builder_segments_{time.strftime('%Y%m%d_%H%M%S')}.srt"

    def node(node_id):
        key = str(node_id)
        if key not in prompt:
            raise KeyError(f"Hidden Whisper workflow node {key} was not found.")
        return prompt[key]

    node(964).setdefault("inputs", {})["audio"] = audio_upload_name
    node(955).setdefault("inputs", {})["audio_file_path"] = staged_audio_path
    duration_inputs = node(963).setdefault("inputs", {})
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
