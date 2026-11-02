"""Back up source prompts, retag references, and ask the LLM for a lossless layout plan."""

import json
import os
import re
import uuid
from datetime import datetime, timezone

from .prompts.seedance import SEEDANCE_CONVERSION_INSTRUCTIONS


HEADINGS = ("References", "Scene", "Preservation", "Shots", "Sound", "Music", "Final assembly")
SOURCE_HEADINGS = {
    "subject_definitions": "References", "reference_definitions": "References",
    "summary": "Scene", "retention_analysis": "Preservation",
    "detailed_description": "Shots", "integrated_multimodal_description": "Shots",
    "overall_soundscape": "Sound", "non_diegetic_music": "Music",
}
PLAN_SCHEMA = {
    "type": "object", "properties": {"sections": {"type": "array", "minItems": 1,
        "items": {"type": "object", "properties": {
            "heading": {"type": "string", "enum": list(HEADINGS)},
            "block_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        }, "required": ["heading", "block_ids"], "additionalProperties": False}}},
    "required": ["sections"], "additionalProperties": False,
}

SEEDANCE_VOCAL_ROLE_GUIDANCE = (
    "Follow the singer/speaker roles explicitly assigned in this scene. Instrument-only performers keep their mouths closed unless the source explicitly gives them a vocal part. "
    "Keep each lyric phrase assigned to its source performer, including parenthetical words. Backing singers already identified in the scene perform their distinct backing-vocal responses where supported by the source assignment and @Audio 1, at their actual recorded times. "
    "Preserve the source's lead-vocal assignment for ambiguous phrases. Each performer follows their own assigned vocal or instrumental part. "
    "Keep the active singer's face and mouth readable during each vocal phrase."
)


def _seedance_narration_only(text, transform):
    """Never repair camera/staging wording inside quoted lyrics or dialogue."""
    parts = re.split(r'("(?:\\.|[^"\\])*"|“[^”]*”)', text, flags=re.S)
    return "".join(part if index % 2 else transform(part) for index, part in enumerate(parts))


def _normalize_seedance_shot_safety(text):
    # A paragraph can contain several shots: scope staging repairs to each shot.
    shots = re.split(r"(?=\[Shot \d+\])", text, flags=re.I)
    return "".join(_normalize_seedance_single_shot(shot) for shot in shots)


def _normalize_seedance_single_shot(text):
    def continuity(part):
        part = re.sub(r"\bHard cut,\s*the same moment on the same stage\b",
                      "Hard cut, continuing immediately from the preceding shot on the same stage", part, flags=re.I)
        return re.sub(r"\bHard cut,\s*(?:the )?same moment\b",
                      "Hard cut, continuing immediately from the preceding shot", part, flags=re.I)
    text = _seedance_narration_only(text, continuity)
    # Detect a vocal close-up from narration, excluding the quoted performance itself.
    narration = re.sub(r'"(?:\\.|[^"\\])*"|“[^”]*”', "", text, flags=re.S)
    if re.search(r"\bclose[- ]?up\b", narration, flags=re.I) and re.search(r"\b(?:sings?|singing|speaks?|speaking|lip[- ]?syncs?)\b", narration, flags=re.I):
        text = _seedance_narration_only(text, lambda part: re.sub(
            r"\bspinning (?:once|a full turn)\b",
            "making a partial turn toward the camera while keeping the singing mouth visible", part, flags=re.I))
    return text


def _clean_seedance_performance_syntax(text):
    """Translate MiniMax delivery tags without changing a single sung/spoken word."""
    def delivery(match):
        body = match[1].strip()
        metadata = re.match(r"^\[([^\]]+)\]\s*", body)
        directions = metadata[1] if metadata else ""
        lyric = body[metadata.end():] if metadata else body
        emphasis = re.findall(r"<i>(.*?)</i>", lyric, flags=re.I | re.S)
        lyric = re.sub(r"</?i>", "", lyric, flags=re.I)
        result = (f"({directions}) " if directions else "") + '"' + lyric + '"'
        if emphasis:
            result += " (emphasize " + ", ".join('"' + word + '"' for word in emphasis) + ")"
        return result
    text = re.sub(r"<d>(.*?)</d>", delivery, text, flags=re.I | re.S)
    text = re.sub(r"<i>(.*?)</i>", r"\1", text, flags=re.I | re.S)
    text = re.sub(r"</?d>", "", text, flags=re.I)
    text = text.replace(
        "preserving every person's exact position, posture, and movement across every camera cut",
        "continuing every person's position, posture, and movement naturally across every camera cut without resetting or freezing the action",
    )
    return text


def _backup_folder(project_folder):
    root = os.path.abspath(str(project_folder or "").strip().strip('"'))
    if not str(project_folder or "").strip() or not os.path.isdir(root):
        raise ValueError("Create or load an existing project before converting prompts.")
    return os.path.join(root, "project_context", "prompt_backups")


def backup_seedance_prompts(payload):
    from ..core.atomic_write import atomic_write_json

    folder = _backup_folder(payload.get("project_folder"))
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("No scene prompts were supplied for backup.")
    allowed = {"id", "label", "start", "end", "source_key", "source_prompt", "minimax_h3_prompt", "i2v_prompt",
               "api_video_prompt", "api_video_settings", "api_video_image_paths", "api_video_image_references",
               "api_video_video_paths", "api_video_audio_paths", "api_video_prompt_reference_binding",
               "image_references", "reference_aliases", "video_paths", "audio_paths", "original_soundtrack", "use_scene_audio"}
    clean, ids = [], set()
    for scene in scenes:
        if not isinstance(scene, dict) or not scene.get("id") or scene["id"] in ids:
            raise ValueError("Every backup scene needs a unique ID.")
        if not str(scene.get("source_prompt") or "").strip():
            raise ValueError("A backup source prompt is empty.")
        ids.add(scene["id"])
        clean.append({key: value for key, value in scene.items() if key in allowed})
    token = uuid.uuid4().hex
    path = os.path.join(folder, f"seedance_{token}.json")
    atomic_write_json(path, {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                            "video_engine": payload.get("video_engine"), "scenes": clean})
    return {"backup_id": token, "backup_path": path, "scene_count": len(clean)}


def load_backup_scene(payload):
    token = str(payload.get("backup_id") or "")
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ValueError("Save a prompt backup before converting.")
    path = os.path.join(_backup_folder(payload.get("project_folder")), f"seedance_{token}.json")
    with open(path, encoding="utf-8") as handle:
        backup = json.load(handle)
    for scene in backup["scenes"]:
        if str(scene["id"]) == str(payload.get("scene_id")):
            return scene
    raise ValueError("Scene was not found in the saved prompt backup.")


def prepare_seedance_blocks(scene):
    text = str(scene.get("source_prompt") or "").strip()
    if not text:
        raise ValueError("The source prompt is empty.")
    aliases = dict(scene.get("reference_aliases") or {})
    images = scene.get("image_references") or []
    for index in range(1, len(images) + 1):
        aliases.setdefault(f"<Picture {index}>", f"@Image {index}")
        aliases.setdefault(f"<Image {index}>", f"@Image {index}")
    for kind, key in (("Video", "video_paths"), ("Audio", "audio_paths")):
        for index in range(1, len(scene.get(key) or []) + 1):
            aliases.setdefault(f"<{kind} {index}>", f"@{kind} {index}")
    if scene.get("original_soundtrack") and not scene.get("audio_paths"):
        aliases.setdefault("<Audio 1>", "Original soundtrack")
    def replace(match):
        key = f"<{match[1].title()} {int(match[2])}>"
        if key not in aliases:
            if match[1].lower() == "subject":
                return f"Subject {int(match[2])}"
            raise ValueError(f"Cannot map {key} to an actual reference. Check this scene's reference assignments.")
        return str(aliases[key])
    # Native tags and bare Picture/Image labels both occur in saved Builder prompts.
    text = re.sub(r"(?<![\w@])<?(Subject|Picture|Image|Video|Audio)\s+(\d+)>?", replace, text, flags=re.I)
    text = text.replace("fully_preserved", "preserve exactly").replace("fully_copy", "retain without alteration in final assembly")
    text = text.replace("[reference generation + audio reuse]", "[Reference-guided generation; original soundtrack retained in final assembly]")
    text = _clean_seedance_performance_syntax(text)
    blocks, current, hint = [], [], "Scene"
    def flush():
        if current:
            blocks.append({"id": f"b{len(blocks) + 1}", "heading_hint": hint, "text": "\n".join(current)})
            current.clear()
    for line in text.splitlines():
        heading = re.fullmatch(r"\s*([a-z_ ]+):\s*", line, flags=re.I)
        heading_key = heading[1].strip().lower() if heading else ""
        destination_heading = SOURCE_HEADINGS.get(heading_key) or next((value for value in HEADINGS if value.lower() == heading_key), None)
        if destination_heading:
            flush()
            hint = destination_heading
        elif not line.strip():
            flush()
        else:
            current.append(line)
    flush()
    for block in blocks:
        if block["heading_hint"] == "Shots" or re.search(r"\[?Shot \d+", block["text"], flags=re.I):
            block["text"] = _normalize_seedance_shot_safety(block["text"])
    if scene.get("use_scene_audio"):
        if scene.get("audio_paths"):
            raise ValueError("Scene-audio mode requires an empty audio-reference file list so the scene excerpt can be Audio 1.")
        for block in blocks:
            if block["heading_hint"] == "Shots" or re.search(r"\[Shot \d+\]|(?:sings?|speaks?) in sync|lip.sync", block["text"], flags=re.I):
                block["text"] = block["text"].replace("Original soundtrack", "@Audio 1")
        definition = "@Audio 1 is the supplied project/custom audio excerpt for this scene, beginning at scene time 0. Follow its original words, vocal phrasing, pauses and tempo. The full original soundtrack is retained by the Builder during final assembly."
        # On reconversion this definition is already an immutable source block.
        if not any(definition in block["text"] for block in blocks):
            blocks.insert(0, {"id": "scene_audio", "heading_hint": "References", "text": definition})
        vocal_narration = "\n".join(re.sub(r'"(?:\\.|[^"\\])*"|“[^”]*”', "", block["text"], flags=re.S) for block in blocks if block["heading_hint"] == "Shots" or re.search(r"\[?Shot \d+", block["text"], flags=re.I))
        if re.search(r"\b(?:sings?|singing|speaks?|speaking|lip[- ]?syncs?)\b", vocal_narration, flags=re.I) and not any(SEEDANCE_VOCAL_ROLE_GUIDANCE in block["text"] for block in blocks):
            blocks.append({"id": "vocal_roles", "heading_hint": "Sound", "text": SEEDANCE_VOCAL_ROLE_GUIDANCE})
    for block in blocks:
        for kind, number in re.findall(r"@(Image|Video|Audio)\s+(\d+)", block["text"]):
            limit = len(images) if kind == "Image" else (1 if kind == "Audio" and scene.get("use_scene_audio") else len(scene.get("video_paths" if kind == "Video" else "audio_paths") or []))
            if not 1 <= int(number) <= limit:
                raise ValueError(f"@{kind} {number} does not have an attached reference. Check the scene's reference assignments.")
    if not blocks:
        raise ValueError("No prompt content remains to format.")
    return blocks


def assemble_seedance_prompt(blocks, plan):
    sections = plan.get("sections") if isinstance(plan, dict) else None
    if not isinstance(sections, list) or not sections:
        raise ValueError("The LLM did not return a Seedance formatting plan. No prompts were changed.")
    ids = []
    for section in sections:
        if not isinstance(section, dict) or section.get("heading") not in HEADINGS or not isinstance(section.get("block_ids"), list) or not section["block_ids"]:
            raise ValueError("The LLM returned an invalid formatting section. No prompts were changed.")
        ids.extend(section["block_ids"])
    if ids != [block["id"] for block in blocks]:
        raise ValueError("The LLM dropped, duplicated, or reordered source content. No prompts were changed.")
    by_id = {block["id"]: block["text"] for block in blocks}
    prompt = "\n\n".join(section["heading"] + ":\n" + "\n\n".join(by_id[key] for key in section["block_ids"]) for section in sections)
    if len(prompt) > 15000:
        raise ValueError("The complete converted prompt exceeds Seedance's 15,000-character limit. Nothing was truncated.")
    return prompt


def convert_seedance_prompt(payload):
    from .builder_instructions import _effective_builder_instruction
    from .builder_runner import _run_builder_text_llm
    from .output_checks import _extract_json_object_from_text

    scene = load_backup_scene(payload)
    blocks = prepare_seedance_blocks(scene)
    instructions = _effective_builder_instruction(payload, "seedance25_prompt_conversion", SEEDANCE_CONVERSION_INSTRUCTIONS)
    text, info = _run_builder_text_llm(
        payload, instructions + "\n\nIMMUTABLE SOURCE BLOCKS:\n" + json.dumps(blocks, ensure_ascii=False),
        temperature=0.1, top_p=0.8, max_new_tokens=2500, label="Seedance format conversion",
        preserve_paragraphs=True, json_schema=PLAN_SCHEMA,
    )
    prompt = assemble_seedance_prompt(blocks, _extract_json_object_from_text(text))
    return {"prompt": prompt, "characters": len(prompt), "preserved_blocks": len(blocks),
            "runner": info.get("runner", "builtin"), "used_model": info.get("used_model", "")}
