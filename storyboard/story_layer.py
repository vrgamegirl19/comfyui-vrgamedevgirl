import json
import os
import re

from ..llm.prompts.storyboard import (
    _storyboard_flf_endpoint_repair_instruction,
    _storyboard_flf_endpoint_schema,
    _storyboard_scene_beat_audio_language_repair_instruction,
    _storyboard_scene_beat_cast_repair_instruction,
    _storyboard_scene_beat_extra_mapping_repair_instruction,
    _storyboard_scene_beat_instruction,
    _storyboard_scene_beat_location_repair_instruction,
    _storyboard_script_story_instruction,
    _storyboard_story_arc_format_retry_instruction,
    _storyboard_story_arc_instruction,
    _storyboard_story_arc_json_retry_instruction,
    _storyboard_story_arc_schema,
    _storyboard_story_brief_instruction,
)
from .cast_guard import build_cast_guard, cast_leaks, cast_wall_text, strip_cast_leaks, strip_story_arc_entry_leaks
from .persistence import _normalize_script_import, _normalize_storyboard_scene
from .scene_helpers import (
    _clean_scene_text,
    _STORY_ARC_DETAIL_PROFILES,
    _STORY_ARC_ENTRY_WORDS,
    _lyric_story_strength_guidance,
    _normalize_story_arc_detail,
    _normalize_story_layer,
    _selected_storyboard_scene,
    _storyboard_dialogue_reference_catalog,
)


def _authoritative_script_from_payload(payload):
    source = payload.get("script_import") or payload.get("scriptImport")
    if not source and isinstance(payload.get("storyboard"), dict):
        source = payload["storyboard"].get("script_import") or payload["storyboard"].get("scriptImport")
    normalized = _normalize_script_import(source or {})
    return normalized if normalized.get("enabled") and normalized.get("cues") else None


def _authoritative_script_text(script_import):
    raw_text = _clean_scene_text((script_import or {}).get("raw_text") or "", 100000)
    if raw_text:
        return raw_text
    return "\n".join(
        f'{cue.get("speaker_alias") or cue.get("speaker_name") or "Speaker"}: {cue.get("text") or ""}'
        for cue in (script_import or {}).get("cues") or []
        if cue.get("text")
    )


def _build_short_film_script_story_text(payload, script_import, purpose="premise"):
    story_layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {})
    subjects, locations = _storyboard_dialogue_reference_catalog(payload)
    script_text = _authoritative_script_text(script_import)
    planned_scenes = (script_import.get("scene_plan") or {}).get("scenes") or []
    compact_plan = []
    for scene in planned_scenes[:240]:
        compact_plan.append({
            "segment": scene.get("index"),
            "duration_seconds": scene.get("duration_seconds"),
            "continuation_of_previous": bool(scene.get("continuation_of_previous")),
            "dialogue": [
                {
                    "speaker_id": cue.get("speaker_id", ""),
                    "speaker": cue.get("speaker_name") or cue.get("speaker_alias") or "Speaker",
                    "exact_text": cue.get("text", ""),
                }
                for cue in scene.get("speaker_assignments") or []
            ],
        })
    if purpose == "brief":
        max_tokens = 1000
        label = "Storyboard Authoritative Script Film Brief"
    else:
        max_tokens = 1100
        label = "Storyboard Authoritative Script Film Premise"
    instruction = _storyboard_script_story_instruction(
        purpose,
        story_layer.get("overall_story_idea") or "",
        json.dumps(subjects, ensure_ascii=False, indent=2) if subjects else "[none]",
        json.dumps(locations, ensure_ascii=False, indent=2) if locations else "[none]",
        script_text,
        json.dumps(compact_plan, ensure_ascii=False, indent=2),
    )
    from ..llm.builder_runner import _run_builder_text_llm

    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.35),
        top_p=float(payload.get("top_p") or 0.90),
        max_new_tokens=int(payload.get("max_new_tokens") or max_tokens),
        label=label,
        preserve_paragraphs=True,
    )
    text = _clean_scene_text(text, 10000)
    if not text:
        raise ValueError("The LLM returned an empty short-film story result.")
    return text, run_info


def _build_story_layer_brief(payload):
    lyrics = _clean_scene_text(payload.get("lyrics") or payload.get("lyrics_text") or "", 16000)
    story_layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or {})
    authoritative_script = _authoritative_script_from_payload(payload)
    if authoritative_script:
        text, run_info = _build_short_film_script_story_text(payload, authoritative_script, "brief")
        return {
            "story_brief": text,
            "runner": run_info.get("runner", "builtin"),
            "used_model": run_info.get("used_model", ""),
            "unloaded": run_info.get("unloaded", True),
            "authoritative_script_used": True,
        }
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        scenes = []
    compact_scenes = []
    for index, scene in enumerate(scenes[:160], start=1):
        if not isinstance(scene, dict):
            continue
        normalized = _normalize_storyboard_scene(scene, index)
        compact_scenes.append({
            "scene_number": normalized["scene_number"],
            "label": normalized["label"],
            "lyric_section": normalized.get("lyric_section", ""),
            "lyrics": normalized.get("lyrics", "")[:500],
            "mapped_extras": [
                {
                    "name": item.get("name", ""),
                    "count": item.get("count", 1),
                    "interaction": item.get("interaction", "background"),
                }
                for item in (normalized.get("extra_subjects") or [])
                if isinstance(item, dict) and item.get("name")
            ],
        })
    if not lyrics and not compact_scenes and not story_layer.get("user_story_arc"):
        raise ValueError("Lyrics, scene lyrics, or a user story arc are required to create a story brief.")
    instruction = _storyboard_story_brief_instruction(
        _lyric_story_strength_guidance(story_layer),
        story_layer.get("user_story_arc") or "",
        lyrics,
        json.dumps(compact_scenes, ensure_ascii=False, indent=2),
    )
    from ..llm.builder_runner import _run_builder_text_llm

    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.35),
        top_p=float(payload.get("top_p") or 0.90),
        max_new_tokens=int(payload.get("max_new_tokens") or 800),
        label="Storyboard Story Brief Gemma",
        preserve_paragraphs=True,
    )
    text = _clean_scene_text(text, 4000)
    if not text:
        raise ValueError("Gemma returned an empty story brief.")
    return {
        "story_brief": text,
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", True),
    }


def _parse_story_arc_lyric_sections(lyrics, collapse_adjacent=True):
    """Return ordered (display label, body) pairs from bracketed lyric headers."""
    structural_pattern = re.compile(
        r"^(?:intro|verse|pre[\s-]?chorus|chorus|post[\s-]?chorus|bridge|outro|"
        r"refrain|hook|breakdown|drop|interlude|instrumental(?:\s+break)?|solo|break|"
        r"spoken(?:\s+word)?|rap)(?:\s+(?:\d+|[ivxlcdm]+))?$",
        re.IGNORECASE,
    )
    annotation_pattern = re.compile(
        r"^(?:whispered|spoken|sung|dark atmosphere|building energy|high energy|"
        r"emotional climax|explosive|quiet arrangement|falling tension|rising tension|"
        r"silence|soft|loud|gentle|intense|energetic|calm|dramatic|atmospheric)$",
        re.IGNORECASE,
    )

    def parse_header_line(raw_line):
        """Return (section label, lyric remainder, terminal marker)."""
        stripped = str(raw_line or "").strip()
        if not stripped.startswith("["):
            return "", raw_line, False
        labels = []
        position = 0
        while position < len(stripped):
            match = re.match(r"\s*\[([^\]\n]{1,80})\]", stripped[position:])
            if not match:
                break
            labels.append(re.sub(r"\s+", " ", match.group(1)).strip())
            position += match.end()
        if not labels:
            return "", raw_line, False
        remainder = stripped[position:].strip()
        terminal = any(label.casefold() in {"end", "end of song"} for label in labels)
        structural = next((label for label in labels if structural_pattern.fullmatch(label)), "")
        if not structural:
            first = labels[0]
            if not annotation_pattern.fullmatch(first) and first.casefold() not in {"end", "end of song"}:
                # Preserve custom section names such as [Part A], while avoiding
                # common performance/mood annotations used beside real headers.
                structural = first
        return structural, remainder, terminal and not structural

    sections = []
    current_label = ""
    current_lines = []
    for raw_line in str(lyrics or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        header_label, remainder, terminal = parse_header_line(raw_line)
        if header_label:
            if current_label:
                sections.append((current_label, "\n".join(current_lines).strip()))
            current_label = header_label
            current_lines = [remainder] if remainder else []
        elif terminal:
            if current_label:
                sections.append((current_label, "\n".join(current_lines).strip()))
            current_label = ""
            current_lines = []
        elif current_label:
            # Lines containing annotation-only tags may still carry lyric text.
            current_lines.append(remainder if remainder != raw_line else raw_line)
    if current_label:
        sections.append((current_label, "\n".join(current_lines).strip()))
    if not sections:
        return []

    # Timeline/storyboard payloads repeat the scene's section header before every
    # lyric chunk.  Treat adjacent copies as one real song section while keeping
    # later recurrences (for example, a chorus after Verse 2) as separate blocks.
    collapsed = []
    for label, body in sections:
        if collapse_adjacent and collapsed and collapsed[-1][0].casefold() == label.casefold():
            previous_label, previous_body = collapsed[-1]
            merged_body = "\n".join(part for part in (previous_body, body) if part).strip()
            collapsed[-1] = (previous_label, merged_body)
        else:
            collapsed.append((label, body))

    counts = {}
    numbered = []
    for label, body in collapsed:
        key = label.casefold()
        counts[key] = counts.get(key, 0) + 1
        occurrence = counts[key]
        display = label if occurrence == 1 else f"{label} {occurrence}"
        numbered.append((display, body))
    return numbered


def _cap_story_arc_words(text, maximum=100):
    # A section written as "Scene N (...)" entries keeps one entry per line, each capped to its share of the maximum.
    entry_lines = [line.strip() for line in str(text or "").split("\n") if line.strip()]
    if len(entry_lines) > 1 and all(re.match(r"^(?:[-*\u2022][ \t]*)?Scene[ \t]+\d+\b", line) for line in entry_lines):
        per_entry = max(12, -(-int(maximum) // len(entry_lines)))
        return "\n".join(_cap_story_arc_words(line, per_entry) for line in entry_lines)
    words = re.findall(r"\S+", str(text or ""))
    if len(words) <= maximum:
        return " ".join(words)
    clipped = " ".join(words[:maximum])
    sentence_end = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
    if sentence_end >= max(80, len(clipped) // 2):
        return clipped[:sentence_end + 1].strip()
    return clipped.rstrip(" ,;:") + "…"


def _story_arc_section_word_limit(section_count, per_section_max=100, total_budget=1500):
    """Keep long song structures within the Story Arc word budget for the chosen detail level."""
    try:
        count = max(0, int(section_count))
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        return per_section_max
    return max(30, min(per_section_max, total_budget // count))


def _story_arc_entry_word_limit(scene_count, per_entry_max=55, total_budget=1500):
    """Words allowed per scene entry when the Story Arc is written scene by scene."""
    try:
        count = max(1, int(scene_count))
    except (TypeError, ValueError):
        count = 1
    return max(22, min(per_entry_max, total_budget // count))


def _story_arc_scene_map(scene_rows, required_labels):
    """Group timeline scenes under the lyric section each one belongs to.

    scene_rows are dicts with scene_number, lyric_section, location and lyric in timeline order.
    Returns [(section_label, [scene_row, ...]), ...] in required_labels order, or [] when the
    scenes cannot be aligned to the lyric sections.
    """
    labels = [str(label) for label in (required_labels or [])]
    rows = [row for row in (scene_rows or []) if isinstance(row, dict)]
    if not labels or not rows:
        return []

    def base(value):
        return re.sub(r"\s+\d+$", "", re.sub(r"\s+", " ", str(value or "")).strip()).casefold()

    runs = []
    for row in rows:
        key = base(row.get("lyric_section"))
        if runs and runs[-1][0] == key:
            runs[-1][1].append(row)
        else:
            runs.append([key, [row]])
    if len(runs) == len(labels):
        return [(label, run[1]) for label, run in zip(labels, runs)]
    # Timeline runs and lyric sections differ in count (for example instrumental scenes), so match by
    # section name and occurrence, then attach every unmatched run to the nearest earlier matched section.
    run_owner = {}
    used_runs = set()
    for label_index, label in enumerate(labels):
        key = base(label)
        run_index = next((index for index, run in enumerate(runs) if run[0] == key and index not in used_runs), None)
        if run_index is None:
            return []
        used_runs.add(run_index)
        run_owner[run_index] = label_index
    owners = []
    current = None
    for index in range(len(runs)):
        if index in run_owner:
            current = run_owner[index]
        owners.append(current)
    first_owner = next((owner for owner in owners if owner is not None), None)
    owners = [first_owner if owner is None else owner for owner in owners]
    grouped = [[] for _ in labels]
    for index, run in enumerate(runs):
        grouped[owners[index]].extend(run[1])
    return [(label, group) for label, group in zip(labels, grouped)]


class StoryArcFormatError(ValueError):
    """Format failure carrying bounded model output for the UI diagnostics panel."""

    def __init__(self, message, *, raw_output="", cleaned_output="", expected_sections=None, runner="LLM"):
        super().__init__(message)
        self.raw_output = str(raw_output or "")[-12000:]
        self.cleaned_output = str(cleaned_output or "")[-12000:]
        self.expected_sections = [str(item) for item in (expected_sections or [])]
        self.runner = str(runner or "LLM")


def _normalize_story_arc_output(text, required_labels, maximum_words=100, runner_label="LLM"):
    """Enforce the detected headings and configured per-section word limit."""
    raw = str(text or "").strip()
    runner_label = str(runner_label or "LLM").strip() or "LLM"
    # Scene entries ("Scene 12 (Location) - ...") must never look like headings, so a stray colon in
    # one becomes a dash before headings are detected.
    if required_labels:
        entry_safe_keys = {re.sub(r"\s+", " ", str(label)).strip().casefold() for label in required_labels}
        entry_lines = []
        for line in raw.split("\n"):
            entry = re.match(r"^([ \t]*(?:[-*\u2022][ \t]*)?Scene[ \t]+\d+\b[^:\n]*?):[ \t]*(.*)$", line)
            if entry and re.sub(r"\s+", " ", entry.group(1)).strip().casefold() not in entry_safe_keys:
                line = f"{entry.group(1).rstrip()} \u2014 {entry.group(2)}".rstrip()
            entry_lines.append(line)
        raw = "\n".join(entry_lines)
    # Local runners may put the section body on the same line as the heading.
    # Keep the line-start anchor so ordinary colons inside prose are not treated
    # as headings, while accepting inline and standalone heading formats.
    heading_pattern = re.compile(r"(?m)^[ \t]*([^\n:]{1,80}):[ \t]*")
    matches = list(heading_pattern.finditer(raw))
    if not matches:
        if required_labels:
            preview = re.sub(r"\s+", " ", raw)[:240]
            expected = ", ".join(str(label) for label in required_labels[:8])
            if len(required_labels) > 8:
                expected += ", …"
            raise ValueError(
                f"{runner_label} did not return the required lyric section headings. "
                f"No heading lines were detected. Expected: {expected}. "
                f"Response preview: {preview or '[empty]'}."
            )
        return _cap_story_arc_words(raw, maximum_words)
    blocks = []
    for index, match in enumerate(matches):
        label = re.sub(r"\s+", " ", match.group(1)).strip()
        bracketed = re.fullmatch(r"\[([^\]\n]{1,80})\]", label)
        if bracketed:
            label = re.sub(r"\s+", " ", bracketed.group(1)).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        blocks.append((label, raw[match.end():end].strip()))
    if required_labels:
        required_keys = {label.casefold() for label in required_labels}
        nonstructural_scene_markers = {
            "instrumental", "instrumental section", "instrumental break",
            "break", "interlude", "solo", "music only", "no vocals",
            "no vocal", "silence", "b roll", "b-roll",
        }
        # The timeline uses values such as [instrumental] as scene-content
        # markers. Gemma can mistakenly promote one into a ninth story heading
        # even when it also returned every required lyric heading in order. Fold
        # that prose into the nearest real section instead of rejecting an
        # otherwise valid arc. Unknown invented headings still fail below.
        folded_blocks = []
        pending_prefix = []
        for label, body in blocks:
            key = label.casefold()
            if key not in required_keys and key in nonstructural_scene_markers:
                if folded_blocks:
                    previous_label, previous_body = folded_blocks[-1]
                    folded_blocks[-1] = (
                        previous_label,
                        "\n".join(part for part in (previous_body, body) if part).strip(),
                    )
                elif body:
                    pending_prefix.append(body)
                continue
            if pending_prefix:
                body = "\n".join([*pending_prefix, body] if body else pending_prefix).strip()
                pending_prefix = []
            folded_blocks.append((label, body))
        if pending_prefix and folded_blocks:
            last_label, last_body = folded_blocks[-1]
            folded_blocks[-1] = (
                last_label,
                "\n".join(part for part in (last_body, *pending_prefix) if part).strip(),
            )
        blocks = folded_blocks
        # Some runners emit one heading per timeline scene. Merge only adjacent
        # same-name runs, and only keep the merge when it produces the exact
        # number of required sections. This preserves strict ordering checks.
        if len(blocks) != len(required_labels):
            merged_blocks = []
            for label, body in blocks:
                if merged_blocks and merged_blocks[-1][0].casefold() == label.casefold():
                    previous_label, previous_body = merged_blocks[-1]
                    merged_blocks[-1] = (
                        previous_label,
                        "\n".join(part for part in (previous_body, body) if part).strip(),
                    )
                else:
                    merged_blocks.append((label, body))
            if len(merged_blocks) == len(required_labels):
                blocks = merged_blocks
        required = [label.casefold() for label in required_labels]
        meta_heading_pattern = re.compile(
            r"\b(?:user|instruction|requirement|preserve|output|heading|exact sections?|"
            r"requested format|response format|story arc request)\b",
            re.IGNORECASE,
        )
        # Qwen and some server models may echo a short instruction heading
        # before the requested answer. Ignore only clearly meta/instructional
        # preamble blocks; invented story sections remain strict failures.
        while blocks:
            first_key = blocks[0][0].casefold()
            first_matches_required = first_key == required[0] or first_key == f"{required[0]} 1"
            if first_matches_required or not meta_heading_pattern.search(blocks[0][0]):
                break
            blocks.pop(0)
        returned = [label.casefold() for label, _body in blocks]
        # Some local Gemma models add "1" to the first occurrence of a
        # heading even though only repeated occurrences are numbered.
        returned = [
            required[index] if index < len(required) and actual == f"{required[index]} 1" else actual
            for index, actual in enumerate(returned)
        ]
        normalized_returned = []
        for index, actual in enumerate(returned):
            if index < len(required):
                expected = required[index]
                expected_base = re.sub(r"\s+\d+$", "", expected)
                if (
                    re.search(r"\s+\d+$", expected)
                    and not re.search(r"\s+\d+$", actual)
                    and expected_base == actual
                ):
                    actual = expected
            normalized_returned.append(actual)
        returned = normalized_returned

        if len(blocks) > len(required_labels) and returned[:len(required)] == required:            # Gemma occasionally preserves every required lyric heading, then
            # appends invented sections such as an extra Instrumental or Outro.
            # The required prefix is already a complete valid story arc, so
            # discard only those trailing additions instead of failing the
            # entire generation. Missing, renamed, reordered, or interleaved
            # headings still fail the strict comparison below.
            blocks = blocks[:len(required_labels)]
            returned = returned[:len(required)]
        if returned != required:
            mismatch_index = next(
                (index for index, (expected, actual) in enumerate(zip(required, returned)) if expected != actual),
                min(len(required), len(returned)),
            )
            expected_label = required_labels[mismatch_index] if mismatch_index < len(required_labels) else "[none]"
            returned_label = blocks[mismatch_index][0] if mismatch_index < len(blocks) else "[missing]"
            missing = [label for label in required_labels if label.casefold() not in returned]
            extra = [label for label, _body in blocks if label.casefold() not in required]
            details = [
                f"Expected {len(required_labels)} headings but {runner_label} returned {len(blocks)}.",
                f"First mismatch at section {mismatch_index + 1}: expected '{expected_label}', received '{returned_label}'.",
                "Detected headings: " + (", ".join(label for label, _body in blocks[:12]) or "[none]") + ("…" if len(blocks) > 12 else "") + ".",
            ]
            if missing:
                details.append("Missing: " + ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "") + ".")
            if extra:
                details.append("Extra: " + ", ".join(extra[:8]) + ("…" if len(extra) > 8 else "") + ".")
            raise ValueError(
                f"{runner_label} changed the lyric structure. "
                + " ".join(details)
            )
        blocks = [
            (required_labels[index], body)
            for index, (_label, body) in enumerate(blocks)
        ]
    return "\n\n".join(
        f"{label}:\n{_cap_story_arc_words(body, maximum_words)}"
        for label, body in blocks
        if body
    )


def _build_story_layer_arc(payload):
    authoritative_script = _authoritative_script_from_payload(payload)
    if authoritative_script:
        text, run_info = _build_short_film_script_story_text(payload, authoritative_script, "premise")
        return {
            "story_arc": text,
            "lyrics_source": "Authoritative Script Mapper import",
            "story_arc_seed": _clean_scene_text(payload.get("story_arc_seed") or payload.get("storyArcSeed") or payload.get("seed") or "", 80),
            "runner": run_info.get("runner", "builtin"),
            "used_model": run_info.get("used_model", ""),
            "unloaded": run_info.get("unloaded", True),
            "authoritative_script_used": True,
        }
    storyboard = payload.get("storyboard") if isinstance(payload.get("storyboard"), dict) else {}
    timeline_lyrics = _clean_scene_text(payload.get("lyrics") or payload.get("lyrics_text") or "", 40000)
    line_mapping_lyrics = _clean_scene_text(payload.get("line_mapping_lyrics") or payload.get("lineMappingLyrics") or "", 40000)
    project_value = payload.get("project_folder") or payload.get("projectFolder") or ""
    project_folder = os.path.abspath(str(project_value).strip().strip('"')) if project_value else ""
    prompt_creator_lyrics = ""
    if project_folder:
        full_lyrics_path = os.path.join(project_folder, "project_context", "full_lyrics.txt")
        if os.path.isfile(full_lyrics_path):
            try:
                with open(full_lyrics_path, "r", encoding="utf-8-sig") as handle:
                    prompt_creator_lyrics = _clean_scene_text(handle.read(), 40000)
            except OSError:
                prompt_creator_lyrics = ""
    # project_context/full_lyrics.txt is the canonical, user-pasted song source.
    # Timeline lyrics contain detected instrumental gaps and must only be a
    # fallback when the project has no saved complete lyric text yet.
    lyrics = prompt_creator_lyrics or line_mapping_lyrics or timeline_lyrics
    lyrics_source = "project full_lyrics.txt" if prompt_creator_lyrics else ("Line Mapping reference lyrics" if line_mapping_lyrics else "timeline scene lyrics")
    lyric_sections = _parse_story_arc_lyric_sections(
        lyrics,
        collapse_adjacent=not bool(line_mapping_lyrics or prompt_creator_lyrics),
    )
    required_section_labels = [item[0] for item in lyric_sections]
    story_arc_detail = _normalize_story_arc_detail(
        payload.get("story_arc_detail") or payload.get("storyArcDetail") or storyboard.get("story_arc_detail")
    )
    section_word_limit = _story_arc_section_word_limit(len(required_section_labels), *_STORY_ARC_DETAIL_PROFILES[story_arc_detail])
    # Larger detail levels need more output tokens and a higher raw-text cap than the 2400 / 14000 defaults.
    arc_word_estimate = section_word_limit * max(1, len(required_section_labels))
    arc_max_new_tokens = max(int(payload.get("max_new_tokens") or 2400), int(arc_word_estimate * 1.6) + 400)
    arc_char_cap = max(14000, arc_word_estimate * 9)
    story_layer = _normalize_story_layer(payload.get("story_layer") or payload.get("storyLayer") or storyboard.get("story_layer") or {})
    story_idea = _clean_scene_text(payload.get("story_idea") or payload.get("storyIdea") or story_layer.get("overall_story_idea") or "", 4000)
    story_arc_seed = _clean_scene_text(payload.get("story_arc_seed") or payload.get("storyArcSeed") or payload.get("seed") or "", 80)
    previous_story_arc = _clean_scene_text(payload.get("previous_story_arc") or payload.get("previousStoryArc") or "", 5000)
    style_theme = _clean_scene_text(payload.get("style_theme") or payload.get("styleTheme") or payload.get("theme") or "", 1600)
    performance_style = _clean_scene_text(payload.get("performance_style") or payload.get("performanceStyle") or storyboard.get("performance_style_default") or "", 200)
    facial_performance = _clean_scene_text(payload.get("facial_performance") or payload.get("facialPerformance") or storyboard.get("facial_performance_default") or "", 200)
    camera_flow = _clean_scene_text(payload.get("camera_flow") or payload.get("cameraFlow") or storyboard.get("camera_flow") or "", 200)
    try:
        camera_motion_speed = int(float(payload.get("camera_motion_speed", payload.get("cameraMotionSpeed", storyboard.get("camera_motion_speed", 4)))))
    except Exception:
        camera_motion_speed = 4
    camera_motion_speed = max(0, min(10, camera_motion_speed))
    try:
        character_motion = int(float(payload.get(
            "character_motion",
            payload.get("characterMotion", payload.get("character_motion_speed", payload.get("characterMotionSpeed", storyboard.get("character_motion_speed", 7))))
        )))
    except Exception:
        character_motion = 7
    character_motion = max(0, min(10, character_motion))
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        scenes = []
    compact_scenes = []
    scene_rows = []
    scene_casts = {}
    subjects = []
    locations = []
    seen_subjects = set()
    seen_locations = set()
    for index, scene in enumerate(scenes[:160], start=1):
        if not isinstance(scene, dict):
            continue
        normalized = _normalize_storyboard_scene(scene, index)
        scene_cast = [] if normalized.get("no_character_present") else [
            {
                "name": _clean_scene_text(subject.get("name") or "", 120),
                "description": _clean_scene_text(subject.get("description") or "", 500),
            }
            for subject in (normalized.get("subject_refs") or [])
            if isinstance(subject, dict) and _clean_scene_text(subject.get("name") or "", 120)
        ]
        scene_casts[normalized["scene_number"]] = scene_cast
        compact_scenes.append({
            "scene_number": normalized["scene_number"],
            "label": normalized["label"],
            "lyric_section": normalized.get("lyric_section", ""),
            "cast": [item["name"] for item in scene_cast] or "no characters",
        })
        for subject in normalized.get("subject_refs") or []:
            if not isinstance(subject, dict):
                continue
            name = _clean_scene_text(subject.get("name") or "", 120)
            description = _clean_scene_text(subject.get("description") or "", 500)
            key = name.lower()
            if key and key not in seen_subjects:
                seen_subjects.add(key)
                subjects.append({"name": name, "description": description})
        location = normalized.get("location_ref")
        if isinstance(location, dict):
            name = _clean_scene_text(location.get("name") or "", 120)
            description = _clean_scene_text(location.get("description") or "", 500)
            key = name.lower()
            if key and key not in seen_locations:
                seen_locations.add(key)
                locations.append({"name": name, "description": description})
        scene_rows.append({
            "scene_number": normalized["scene_number"],
            "lyric_section": normalized.get("lyric_section", ""),
            "location": _clean_scene_text(location.get("name") or "", 120) if isinstance(location, dict) else "",
            "lyric": _clean_scene_text(normalized.get("lyrics", ""), 90),
            "cast": [item["name"] for item in scene_cast],
        })
    reference_builder = payload.get("reference_builder") or payload.get("referenceBuilder") or {}
    if isinstance(reference_builder, dict):
        for subject in reference_builder.get("subjects") or []:
            if not isinstance(subject, dict):
                continue
            name = _clean_scene_text(subject.get("name") or "", 120)
            description = _clean_scene_text(subject.get("description") or "", 500)
            key = name.lower()
            if key and key not in seen_subjects:
                seen_subjects.add(key)
                subjects.append({"name": name, "description": description})
        for location in reference_builder.get("locations") or []:
            if not isinstance(location, dict):
                continue
            name = _clean_scene_text(location.get("name") or "", 120)
            description = _clean_scene_text(location.get("description") or "", 500)
            key = name.lower()
            if key and key not in seen_locations:
                seen_locations.add(key)
                locations.append({"name": name, "description": description})
    # Hard wall: a scene entry may only mention the characters selected for that scene.
    cast_guards = {
        number: build_cast_guard(subjects, cast)
        for number, cast in scene_casts.items()
    }
    # When the timeline scenes can be matched to the lyric sections, the arc is written scene by scene so it
    # uses every mapped location and gives each scene its own story entry.
    section_scene_map = _story_arc_scene_map(scene_rows, required_section_labels)
    entry_word_limit = 0
    if section_scene_map:
        mapped_scene_count = sum(len(rows) for _label, rows in section_scene_map)
        entry_word_limit = _story_arc_entry_word_limit(
            mapped_scene_count,
            _STORY_ARC_ENTRY_WORDS[story_arc_detail],
            _STORY_ARC_DETAIL_PROFILES[story_arc_detail][1],
        )
        largest_section = max(len(rows) for _label, rows in section_scene_map)
        section_word_limit = max(section_word_limit, largest_section * entry_word_limit + 20)
        arc_word_estimate = mapped_scene_count * entry_word_limit
        arc_max_new_tokens = max(int(payload.get("max_new_tokens") or 2400), int(arc_word_estimate * 1.6) + 400)
        arc_char_cap = max(14000, arc_word_estimate * 9)
    instruction = _storyboard_story_arc_instruction(
        required_section_labels=required_section_labels,
        section_word_limit=section_word_limit,
        section_scene_map=section_scene_map,
        entry_word_limit=entry_word_limit,
        song_story_brief=story_layer.get("song_story_brief") or "",
        story_arc_seed=story_arc_seed,
        camera_flow=camera_flow,
        camera_motion_speed=camera_motion_speed,
        character_motion_speed=character_motion,
        performance_style=performance_style,
        facial_performance=facial_performance,
        lyric_story_strength_guidance_text=_lyric_story_strength_guidance(story_layer),
        story_idea=story_idea,
        previous_story_arc=previous_story_arc,
        style_theme=style_theme,
        lyrics_source=lyrics_source,
        lyrics=lyrics,
        compact_scenes_json=json.dumps(compact_scenes, ensure_ascii=False, indent=2) if compact_scenes else "[not provided]",
        subjects_json=json.dumps(subjects[:24], ensure_ascii=False, indent=2) if subjects else "[not provided]",
        locations_json=json.dumps(locations[:40], ensure_ascii=False, indent=2) if locations else "[not provided]",
    )
    from ..llm.builder_runner import _llm_runner_display_name, _run_builder_text_llm, _runner_supports_json_schema

    runner_label = _llm_runner_display_name(payload)

    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.45),
        top_p=float(payload.get("top_p") or 0.92),
        max_new_tokens=arc_max_new_tokens,
        label=f"Storyboard Story Arc {runner_label}",
        preserve_paragraphs=True,
    )
    text = _clean_scene_text(text, arc_char_cap)
    if not text:
        raise ValueError(f"{runner_label} returned an empty story arc.")
    try:
        text = _normalize_story_arc_output(text, required_section_labels, section_word_limit, runner_label)
    except ValueError as first_error:
        if not required_section_labels:
            raise
        # Runners that support schemas put each required heading in a JSON key, so the headings cannot drift.
        structured = _runner_supports_json_schema(payload)
        if structured:
            retry_instruction = _storyboard_story_arc_json_retry_instruction(instruction)
        else:
            retry_instruction = _storyboard_story_arc_format_retry_instruction(required_section_labels, instruction)
        retry_payload = dict(payload or {})
        try:
            retry_payload["seed"] = (int(payload.get("seed") or 0) + 1) % 2147483647
        except (TypeError, ValueError):
            retry_payload["seed"] = 1
        retry_text, retry_info = _run_builder_text_llm(
            retry_payload,
            retry_instruction,
            temperature=0.2,
            top_p=0.85,
            max_new_tokens=arc_max_new_tokens,
            label=f"Storyboard Story Arc {runner_label} format retry",
            preserve_paragraphs=True,
            json_schema=_storyboard_story_arc_schema(required_section_labels) if structured else None,
        )
        try:
            if structured:
                sections = json.loads(retry_text)
                retry_text = "\n\n".join(f"{label}:\n{str(sections.get(label) or '').strip()}" for label in required_section_labels)
            retry_text = _clean_scene_text(retry_text, arc_char_cap)
            text = _normalize_story_arc_output(
                retry_text,
                required_section_labels,
                section_word_limit,
                runner_label,
            )
            run_info = retry_info
        except ValueError as retry_error:
            raise StoryArcFormatError(
                f"{runner_label} could not preserve the lyric-section structure after an automatic format retry. "
                f"{retry_error}",
                raw_output=retry_text or text,
                cleaned_output=retry_text or text,
                expected_sections=required_section_labels,
                runner=runner_label,
            ) from first_error
    text = strip_story_arc_entry_leaks(
        text,
        {number: guard for number, guard in cast_guards.items() if guard},
        {number: [item["name"] for item in cast] for number, cast in scene_casts.items()},
    )
    return {
        "story_arc": text,
        "lyrics_source": lyrics_source,
        "story_arc_seed": story_arc_seed,
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", True),
    }


_STORYBOARD_DRIFT_LOCATION_PATTERNS = [
    (r"\bwarehouse\b", "warehouse"),
    (r"\bloading\s+dock\b", "loading dock"),
    (r"\bindustrial\b", "industrial"),
    (r"\bbackstage\s+corridor\b", "backstage corridor"),
    (r"\bnarrow,\s*dimly\s*lit\s+corridor\b", "dimly lit corridor"),
    (r"\bdark,\s*narrow\s+corridor\b", "dark corridor"),
    (r"\bheavy\s+(?:metal|steel)\s+door\b", "heavy metal/steel door"),
    (r"\bmassive\s+window\b", "massive window"),
    (r"\bconcrete\b", "concrete"),
    (r"\bmetal\s+pipes?\b", "metal pipes"),
    (r"\bsteel\s+stairs?\b", "steel stairs"),
    (r"\bvast,\s*silent\s+hall\b", "vast hall"),
    (r"\bvast\s+empty\s+space\b", "vast empty space"),
]


def _storyboard_scene_location_context(scene):
    if not isinstance(scene, dict):
        return ""
    location = scene.get("location_ref") if isinstance(scene.get("location_ref"), dict) else {}
    parts = []
    if isinstance(location, dict):
        parts.extend([
            location.get("name"),
            location.get("description"),
            location.get("trigger_phrase") or location.get("trigger") or location.get("Trigger"),
        ])
    parts.extend([scene.get("setting"), scene.get("location")])
    return _clean_scene_text(" ".join(str(part or "") for part in parts if str(part or "").strip()), 2400)


def _storyboard_location_drift_terms(text, location_context):
    text_lower = str(text or "").lower()
    location_lower = str(location_context or "").lower()
    if not text_lower or not location_lower:
        return []
    drift_terms = []
    for pattern, label in _STORYBOARD_DRIFT_LOCATION_PATTERNS:
        if re.search(pattern, text_lower, flags=re.IGNORECASE) and not re.search(pattern, location_lower, flags=re.IGNORECASE):
            drift_terms.append(label)
    return drift_terms


def _parse_flf_endpoint_json(text):
    raw = str(text or "").strip()
    raw = re.sub(
        r"^\s*[^A-Za-z0-9]*(?:(?:user|assistant|model)\b)?[^A-Za-z0-9]*(?:thought|analysis|reasoning)(?=[A-Z]|[^A-Za-z0-9]|$)[^A-Za-z0-9]*",
        "",
        raw,
        flags=re.I,
    ).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    raw = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    match = re.search(r"\{.*\}", raw, flags=re.S)
    candidate = match.group(0) if match else raw
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return json.loads(candidate)


_SCENE_BEAT_AUDIO_LANGUAGE = re.compile(
    r"\b(?:lip[\s-]?sync(?:ing)?|sings?|singing|sang|lyrics?|lyric|"
    r"vocals?|vocalizing|vocalizes?|rapping|raps?|music|instrumental|"
    r"performing vocals?)\b",
    flags=re.IGNORECASE,
)


def _scene_beat_has_audio_language(text):
    """Return whether a narrative scene beat leaks performance/audio metadata."""
    return bool(_SCENE_BEAT_AUDIO_LANGUAGE.search(str(text or "")))


def _strip_scene_beat_audio_language(text):
    """Remove common performance phrasing from a beat as a final safety net.

    Scene beats are consumed as visual story guidance. Audio/performance metadata
    belongs to the video-prompt stage and must never become part of this field.
    """
    cleaned = _clean_scene_text(text, 1800)
    if not cleaned:
        return ""
    # Preserve the useful visual clause in constructions such as:
    # "Singer visibly sings ... as her fingertips trace ...".
    cleaned = re.sub(
        r"\b(?:the\s+)?(?:singer|performer|vocalist)\s+(?:visibly\s+)?"
        r"(?:sings?|sang|is\s+singing|performs?)\b"
        r"(?:\s+(?:through|during|over)\s+[^,.;!?]+)?\s+as\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:while|as)\s+(?:she|he|they|the\s+(?:singer|performer|vocalist))\s+"
        r"(?:sings?|sang|is\s+singing|performs?)\b\s*[,]?\s*",
        "while ",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Drop any residual sentence that is solely an audio/performance aside.
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [
        sentence for sentence in sentences
        if not _scene_beat_has_audio_language(sentence)
    ]
    cleaned = " ".join(sentences)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" \t\n,;:-")
    return cleaned


def _build_story_layer_scene_beat(payload):
    scene_bundle = payload.get("storyboard_payload") or payload.get("scene_bundle") or payload.get("gpt_payload")
    if not isinstance(scene_bundle, dict):
        raise ValueError("Storyboard scene-card payload is missing.")
    scene = _selected_storyboard_scene(scene_bundle)
    if not scene:
        raise ValueError("Storyboard scene-card payload has no selected scene.")
    story_layer = _normalize_story_layer(payload.get("story_layer") or scene_bundle.get("story_layer") or {})
    previous_beat = _clean_scene_text(payload.get("previous_beat") or "", 1200)
    previous_lyrics = _clean_scene_text(payload.get("previous_lyrics") or "", 800)
    previous_end_state = _clean_scene_text(payload.get("previous_end_state") or "", 1800)
    previous_carry_forward = _clean_scene_text(payload.get("previous_carry_forward") or "", 1800)
    current_lyrics = _clean_scene_text(payload.get("current_lyrics") or scene.get("lyrics") or scene.get("lyric_text") or "", 1200)
    next_lyrics = _clean_scene_text(payload.get("next_lyrics") or "", 800)
    flf_mode = bool(payload.get("flf_mode")) or str(scene.get("video_prompt_type") or "").strip().lower() == "flf"
    vocal_status = scene.get("vocal_status") if isinstance(scene.get("vocal_status"), dict) else {}
    # Scene-beat generation is also called from Image Prep, where the normal
    # prompt payload intentionally clears vocal_status.singers. The storyboard
    # scene's lyric_singers is the performer assignment that remains valid in
    # both Image Prep and Video Prep.
    assigned_performers = scene.get("lyric_singers") if isinstance(scene.get("lyric_singers"), list) else []
    if not assigned_performers and isinstance(vocal_status.get("singers"), list):
        assigned_performers = vocal_status.get("singers")
    assigned_performers = [_clean_scene_text(item, 180) for item in assigned_performers if _clean_scene_text(item, 180)]
    performer_assignment = scene.get("performer_assignment") if isinstance(scene.get("performer_assignment"), dict) else {}
    assigned_performers = [
        _clean_scene_text(item, 180)
        for item in (performer_assignment.get("singing") if isinstance(performer_assignment.get("singing"), list) else assigned_performers)
        if _clean_scene_text(item, 180)
    ]
    if not assigned_performers:
        mapped_subject_names = []
        for item in scene.get("subject_refs") or []:
            if isinstance(item, dict):
                name = _clean_scene_text(item.get("name") or "", 180)
                if name:
                    mapped_subject_names.append(name)
        for item in scene.get("subjects") or []:
            name = _clean_scene_text(item.get("name") if isinstance(item, dict) else item, 180)
            if name:
                mapped_subject_names.append(name)
        singer_named_subjects = [
            name for name in dict.fromkeys(mapped_subject_names)
            if re.search(r"\b(?:singer|performer|vocalist|rapper)\b", name, flags=re.IGNORECASE)
        ]
        if len(singer_named_subjects) == 1:
            assigned_performers = singer_named_subjects
    silent_performers = [
        _clean_scene_text(item, 180)
        for item in (performer_assignment.get("silent") if isinstance(performer_assignment.get("silent"), list) else [])
        if _clean_scene_text(item, 180)
    ]
    scene_defaults = {
        "shot_type": _clean_scene_text(scene.get("shot_type") or scene.get("shot") or "", 240),
        "camera_motion": _clean_scene_text(scene.get("camera_motion") or scene.get("camera_motion_preset") or "", 500),
        "camera_flow": _clean_scene_text(scene.get("camera_flow") or scene.get("cameraFlow") or "", 120),
        "camera_flow_guidance": _clean_scene_text(scene.get("camera_flow_guidance") or "", 1200),
        "character_motion": _clean_scene_text(scene.get("character_motion") or scene.get("character_motion_preset") or "", 700),
        "performance_direction": _clean_scene_text(scene.get("performance_direction") or scene.get("performance_style") or "", 1000),
        "facial_performance_direction": _clean_scene_text(scene.get("facial_performance_direction") or scene.get("facial_performance_custom") or scene.get("facial_performance") or "", 1200),
    }
    raw_extra_subjects = scene.get("extra_subjects") or scene.get("extraSubjects") or []
    extra_subjects = []
    if isinstance(raw_extra_subjects, list):
        for index, item in enumerate(raw_extra_subjects[:100], start=1):
            if not isinstance(item, dict):
                continue
            name = _clean_scene_text(item.get("name") or item.get("title") or f"Extra {index}", 180)
            if not name:
                continue
            interaction = str(item.get("interaction") or "background").strip()
            if interaction not in {"background", "background_dancing", "alongside", "dancing_with", "direct"}:
                interaction = "background"
            try:
                count = max(1, min(100, int(round(float(item.get("count") or 1)))))
            except (TypeError, ValueError):
                count = 1
            extra_subjects.append({
                "name": name,
                "count": count,
                "interaction": interaction,
                "identity": _clean_scene_text(item.get("identity") or item.get("description") or "", 240),
            })
    if scene.get("no_character_present") or scene.get("noCharacterPresent") or scene.get("no_visible_subject") or scene.get("no_subject"):
        extra_subjects = []
    beat_word_limit = 100 if extra_subjects else 80
    # Hard wall: characters not selected for this scene may not appear in its beat.
    scene_cast = [] if (scene.get("no_character_present") or scene.get("noCharacterPresent")) else [
        item for item in (scene.get("subject_refs") or []) if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    all_project_subjects = [item for item in (payload.get("all_subjects") or []) if isinstance(item, dict)]
    reference_builder = payload.get("reference_builder") or payload.get("referenceBuilder") or {}
    if isinstance(reference_builder, dict):
        all_project_subjects += [item for item in (reference_builder.get("subjects") or []) if isinstance(item, dict)]
    cast_guard = build_cast_guard(all_project_subjects + scene_cast, scene_cast)
    instruction = _storyboard_scene_beat_instruction(
        flf_mode=flf_mode,
        beat_word_limit=beat_word_limit,
        lyric_story_strength_guidance_text=_lyric_story_strength_guidance(story_layer),
        user_story_arc=story_layer.get("user_story_arc") or "",
        song_story_brief=story_layer.get("song_story_brief") or "",
        previous_beat=previous_beat,
        previous_lyrics=previous_lyrics,
        previous_end_state=previous_end_state,
        previous_carry_forward=previous_carry_forward,
        current_lyrics=current_lyrics,
        next_lyrics=next_lyrics,
        scene_defaults_json=json.dumps(scene_defaults, ensure_ascii=False, indent=2),
        performance_assignment_json=json.dumps({"singing": assigned_performers, "silent": silent_performers}, ensure_ascii=False, indent=2),
        extra_subjects_json=json.dumps(extra_subjects, ensure_ascii=False, indent=2) if extra_subjects else "[none]",
        scene_json=json.dumps(scene, ensure_ascii=False, indent=2),
        cast_wall=cast_wall_text(cast_guard),
    )
    from ..llm.builder_runner import _run_builder_text_llm, _runner_supports_json_schema

    text, run_info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.35),
        top_p=float(payload.get("top_p") or 0.90),
        max_new_tokens=int(payload.get("max_new_tokens") or 360),
        label="Storyboard Scene Beat Gemma",
        preserve_paragraphs=True,
    )
    flf_fields = {}
    if flf_mode:
        try:
            parsed = _parse_flf_endpoint_json(text)
        except Exception as parse_error:
            # Runners that support schemas re-run the beat constrained to the five FLF keys; the text repair prompt is the fallback.
            if _runner_supports_json_schema(payload):
                repair_instruction = instruction
                repair_schema = _storyboard_flf_endpoint_schema()
            else:
                repair_instruction = _storyboard_flf_endpoint_repair_instruction(text, json.dumps(scene, ensure_ascii=False))
                repair_schema = None
            repaired_text, repair_info = _run_builder_text_llm(
                payload,
                repair_instruction,
                temperature=0.05,
                top_p=0.75,
                max_new_tokens=max(900, int(payload.get("max_new_tokens") or 360)),
                label="Storyboard FLF Endpoint JSON Repair",
                preserve_paragraphs=True,
                json_schema=repair_schema,
            )
            try:
                parsed = _parse_flf_endpoint_json(repaired_text)
                run_info = {**run_info, "json_repaired": True, "repair_runner": repair_info.get("runner", "")}
            except Exception as repair_error:
                raise ValueError(
                    f"Gemma returned malformed FLF endpoint JSON and automatic repair failed. "
                    f"Original parse error: {parse_error}; repair parse error: {repair_error}"
                ) from repair_error
        flf_fields = {
            key: _clean_scene_text(parsed.get(key) or "", 1800)
            for key in ("flf_start_state", "flf_transformation", "flf_end_state", "flf_carry_forward")
        }
        if previous_end_state:
            flf_fields["flf_start_state"] = previous_end_state
        text = _clean_scene_text(parsed.get("story_beat") or "", 1800)
        missing = [key for key, value in flf_fields.items() if not value]
        if not text or missing:
            raise ValueError("Gemma returned incomplete FLF endpoint fields: " + ", ".join((["story_beat"] if not text else []) + missing))
    else:
        text = re.sub(r"^\s*(scene\s+story\s+beat|story\s+beat|beat)\s*:\s*", "", _clean_scene_text(text, 1800), flags=re.I)
    if not text:
        raise ValueError("Gemma returned an empty scene story beat.")
    if _scene_beat_has_audio_language(text):
        repair_instruction = _storyboard_scene_beat_audio_language_repair_instruction(text)
        repaired_text, repair_info = _run_builder_text_llm(
            payload,
            repair_instruction,
            temperature=0.10,
            top_p=0.80,
            max_new_tokens=300,
            label="Storyboard Scene Beat Audio Language Repair Gemma",
            preserve_paragraphs=True,
        )
        repaired_text = re.sub(r"^\s*(scene\s+story\s+beat|story\s+beat|beat)\s*:\s*", "", _clean_scene_text(repaired_text, 1800), flags=re.I)
        if repaired_text and not _scene_beat_has_audio_language(repaired_text):
            text = repaired_text
            run_info = {
                **run_info,
                "audio_language_repaired": True,
                "audio_language_repair_runner": repair_info.get("runner", ""),
                "audio_language_repair_model": repair_info.get("used_model", ""),
            }
        else:
            text = _strip_scene_beat_audio_language(text)
            run_info = {**run_info, "audio_language_repaired": True, "audio_language_repair_fallback": True}
    else:
        text = _strip_scene_beat_audio_language(text)
    location_context = _storyboard_scene_location_context(scene)
    drift_terms = _storyboard_location_drift_terms(text, location_context)
    if drift_terms:
        repair_instruction = _storyboard_scene_beat_location_repair_instruction(location_context, drift_terms, text)
        repaired_text, repair_info = _run_builder_text_llm(
            payload,
            repair_instruction,
            temperature=0.20,
            top_p=0.85,
            max_new_tokens=300,
            label="Storyboard Scene Beat Location Repair Gemma",
            preserve_paragraphs=True,
        )
        repaired_text = re.sub(r"^\s*(scene\s+story\s+beat|story\s+beat|beat)\s*:\s*", "", _clean_scene_text(repaired_text, 1800), flags=re.I)
        repaired_drift_terms = _storyboard_location_drift_terms(repaired_text, location_context)
        if repaired_text and not repaired_drift_terms:
            text = repaired_text
            run_info = {
                **run_info,
                "location_repaired": True,
                "location_repair_terms": drift_terms,
                "location_repair_runner": repair_info.get("runner", ""),
                "location_repair_model": repair_info.get("used_model", ""),
            }
        else:
            subject_hint = "The scene subject" if not scene.get("subject_refs") else _clean_scene_text((scene.get("subject_refs") or [{}])[0].get("name") or "The scene subject", 120)
            text = (
                f"{subject_hint} channels the story arc's defiant, boundary-breaking energy inside {location_context}. "
                "The beat focuses on tension, control, and release through posture, expression, and interaction with the mapped studio environment, without changing the physical location."
            )
            run_info = {
                **run_info,
                "location_repaired": True,
                "location_repair_terms": drift_terms,
                "location_repair_fallback": True,
            }
    missing_extras = [item["name"] for item in extra_subjects if item["name"].casefold() not in text.casefold()]
    if missing_extras:
        repair_instruction = _storyboard_scene_beat_extra_mapping_repair_instruction(json.dumps(extra_subjects, ensure_ascii=False, indent=2), text)
        repaired_text, repair_info = _run_builder_text_llm(
            payload,
            repair_instruction,
            temperature=0.15,
            top_p=0.82,
            max_new_tokens=360,
            label="Storyboard Scene Beat Extra Mapping Repair Gemma",
            preserve_paragraphs=True,
        )
        repaired_text = re.sub(r"^\s*(scene\s+story\s+beat|story\s+beat|beat)\s*:\s*", "", _clean_scene_text(repaired_text, 1800), flags=re.I)
        still_missing = [item["name"] for item in extra_subjects if item["name"].casefold() not in repaired_text.casefold()]
        if not repaired_text or still_missing:
            raise ValueError("Gemma omitted mapped extras from the scene story beat after repair: " + ", ".join(still_missing or missing_extras))
        text = repaired_text
        run_info = {
            **run_info,
            "extra_mapping_repaired": True,
            "extra_mapping_repair_runner": repair_info.get("runner", ""),
            "extra_mapping_repair_model": repair_info.get("used_model", ""),
        }
    text = _strip_scene_beat_audio_language(text)
    if not text:
        raise ValueError("Scene beat became empty after removing audio/performance language.")
    leaked_terms = cast_leaks(text, cast_guard)
    if leaked_terms:
        repaired_text, repair_info = _run_builder_text_llm(
            payload,
            _storyboard_scene_beat_cast_repair_instruction(cast_wall_text(cast_guard), leaked_terms, text),
            temperature=0.15,
            top_p=0.80,
            max_new_tokens=300,
            label="Storyboard Scene Beat Cast Repair Gemma",
            preserve_paragraphs=True,
        )
        repaired_text = re.sub(r"^\s*(scene\s+story\s+beat|story\s+beat|beat)\s*:\s*", "", _clean_scene_text(repaired_text, 1800), flags=re.I)
        run_info = {**run_info, "cast_repaired": True, "cast_repair_terms": leaked_terms}
        # The filter is the guarantee: whatever still refers to an unselected character is removed.
        text = strip_cast_leaks(repaired_text if repaired_text and not cast_leaks(repaired_text, cast_guard) else text, cast_guard)
        if not text:
            names = ", ".join(cast_guard["cast_names"]) if cast_guard and cast_guard["cast_names"] else "The scene"
            text = f"The scene stays focused on {names}, expressed through posture, expression, and the mapped location."
    flf_fields = {key: strip_cast_leaks(value, cast_guard) for key, value in flf_fields.items()}
    return {
        "story_beat": text,
        **flf_fields,
        "runner": run_info.get("runner", "builtin"),
        "used_model": run_info.get("used_model", ""),
        "unloaded": run_info.get("unloaded", True),
    }
