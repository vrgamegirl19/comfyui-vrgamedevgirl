"""MiniMax H3 reference-to-video prompts, the way the Video Builder writes them.

The LLM writes only creative shot descriptions (JSON). The builder owns the ``[Shot N]`` labels, the cut
times and the ``detailed_description`` wrapper. ``wrap_reference_prompt`` then puts the reference definitions
(``subject_definitions``, ``summary``, ``retention_analysis``) before it and the soundscape sections after it, like the
prompts the Video Builder saves. The render sends the saved text as it is, so without them ``<Subject N>`` would never
be tied to ``<Picture N>``.

Python twin of ``minimax_prompt.mjs`` (``miniMaxH3CreativePromptContextForSegment``,
``parseMiniMaxH3ShotDescriptionPayload``, ``miniMaxH3OfficialShotBodyFromDescriptions``,
``assertValidMiniMaxH3FinalPrompt``) for scenes with singing, speaking-free performance and no timed cue map.
"""

import json
import re
from typing import Any, Dict, List, Optional

HARD_LIMIT = 7000
FALLBACK_SHOT = "A steady cinematic shot continues the scene with natural camera movement and continuous physical action."

_LIFE_SUBTLE = [
    "breath catching before a held note", "weight shifting from one foot to the other", "fingers tightening then loosening on an object",
    "a glance away and back", "shoulders dropping after a held pose", "a slow blink", "a head tilt on a sustained note",
    "a hand brushing hair or collar", "a small smile that fades", "jaw tension then release", "a lean toward or away from the camera",
    "an unfinished gesture",
]
_LIFE_ACTIVE = [
    "a snap turn that whips the hair", "a quick step then a hard stop", "shoulders driving on the downbeat", "a spin that ends in a pose",
    "a fast reach and grab", "a head throw back on the high note", "a jump landing with bent knees", "a quick glance at the other person mid-stride",
    "hands slicing the air with the rhythm", "a lean into the camera then a push away", "a stumble that turns into a dance step",
    "a laugh breaking out mid-motion",
]


class ShotPromptError(ValueError):
    """The LLM output or the assembled prompt cannot be used. ``code`` names the problem."""

    def __init__(self, message: str, code: str = "INVALID_SHOT_PROMPT", length: int = 0):
        super().__init__(message)
        self.code = code
        self.length = length


def timecode(seconds: Any) -> str:
    """``MM:SS.mmm`` (``miniMaxH3Timecode``)."""
    total = max(0.0, float(seconds or 0))
    minutes = int(total // 60)
    return f"{minutes:02d}:{total - minutes * 60:06.3f}"


def shot_plan(cut_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    cuts = cut_plan.get("cut_times_seconds") or []
    return [{"number": 1, "time": 0.0, "timecode": "00:00.000"}] + [
        {"number": i + 2, "time": float(t), "timecode": timecode(t)} for i, t in enumerate(cuts)
    ]


def cut_plan_instruction(cut_plan: Dict[str, Any]) -> str:
    """Port of ``miniMaxH3OfficialCutPlanInstruction`` (frequency-driven plans)."""
    duration = round(float(cut_plan.get("exact_duration_seconds") or 0), 3)
    cuts = cut_plan.get("cut_times_seconds") or []
    if not cuts:
        return (
            f"EDITING / CUT PLAN — MANDATORY: Use one smooth, continuous, uninterrupted shot for the full {duration}-second segment. "
            "Output only [Shot 1]. Use no additional shot label, hard cut, angle reset, montage, dissolve, scene change, or transition. "
            "Camera and character movement may develop inside the same continuous take."
        )
    times = ", ".join(timecode(t) for t in cuts)
    plural = "" if len(cuts) == 1 else "s"
    return (
        f"EDITING / CUT PLAN — MANDATORY: Cut frequency {cut_plan.get('frequency')}/10 for this exact {duration}-second segment requires exactly "
        f"{len(cuts)} hard cut{plural} at {times}, creating {len(cuts) + 1} coherent shots. The builder will write [Shot 1] and every later "
        "[Shot N] At MM:SS.mmm label. Return only the creative description for each shot. Each later description should stage a "
        "continuity-preserving camera/shot cut. Do not omit, merge, add, or shift a scheduled cut. Do not add extra shots, montage beats, "
        "dissolves, scene changes, or transitions outside this schedule."
    )


def motion_energy_text(camera_speed: float, character_speed: float) -> str:
    camera = "The camera is fast and energetic: whips, orbits, push-ins, and quick tracking." if camera_speed >= 7 else (
        "The camera moves at a steady pace with a clear direction." if camera_speed >= 4 else "The camera moves slowly, or holds a locked frame.")
    character = "The performers are high energy: pop, fast motion, jumps, spins, runs, dance hits, and quick reactions." if character_speed >= 7 else (
        "The performers use steady physical action: walking, turning, reaching, and set interaction." if character_speed >= 4
        else "The performers use small movements and held poses with life detail.")
    return f"{camera} {character}"


def life_movements(character_speed: float, seed: str = "") -> List[str]:
    if character_speed >= 7:
        bank = _LIFE_ACTIVE
    elif character_speed >= 4:
        bank = _LIFE_SUBTLE[:6] + _LIFE_ACTIVE[:6]
    else:
        bank = _LIFE_SUBTLE
    offset = sum(ord(c) for c in str(seed)) % len(bank)
    return [bank[(offset + i) % len(bank)] for i in range(6)]


def opening_wrapper(style: str) -> str:
    return f"detailed_description:\nThe target video is in a {style or 'photorealistic cinematic'} music-video style.\n\n"


def _clean_sentence_start(text: str) -> str:
    return re.sub(r"^(\s*[\"'“‘(]*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def strip_negative_sentences(description: str) -> str:
    """Keep only positive visual sentences (``stripMiniMaxH3NegativePromptSentences``)."""
    text = str(description or "").strip()
    if not text:
        return ""
    negative = re.compile(r"\b(?:do\s+not|don['’]t|never|without|avoid|must\s+not|cannot|can['’]t|not|no)\b", re.I)
    # Quoted lyric words are sung text, not instructions, so "don't" or "no" inside quotes is never a reason to drop a sentence.
    quotes: List[str] = []

    def mask(match: "re.Match[str]") -> str:
        quotes.append(match.group(0))
        return f"VRGDGQUOTE{len(quotes) - 1}TOKEN"

    masked = re.sub(r'["“][^"”]*["”]', mask, text)
    # A decimal point (1.5 seconds) is not a sentence end.
    protected = re.sub(r"(\d)\.(?=\d)", lambda m: m.group(1) + "VRGDGDECIMALTOKEN", masked)
    sentences = [s.replace("VRGDGDECIMALTOKEN", ".") for s in re.findall(r"[^.!?…]+[.!?…]+|[^.!?…]+$", protected)] or [masked]
    kept = " ".join(s.strip() for s in sentences if not negative.search(re.sub(r"VRGDGQUOTE\d+TOKEN", " ", s)))
    kept = re.sub(r"VRGDGQUOTE(\d+)TOKEN", lambda m: quotes[int(m.group(1))], kept)
    return re.sub(r"\s{2,}", " ", kept).strip()


def normalize_description(text: str) -> str:
    """Port of ``normalizeMiniMaxH3ShotDescription`` for the parts that apply without dialogue tags."""
    clean = str(text or "")
    clean = re.sub(r"<\|[^<>]*\|>", "", clean)
    clean = re.sub(r"<\s*(Subject\s+\d+)\s*\(\s*([^)<>]+)\s*>", r"<\1> (\2)", clean, flags=re.I)
    clean = re.sub(r"(?<!<)\bSubject\s+(\d+)\b(?!\s*>)", r"<Subject \1>", clean, flags=re.I)
    clean = re.sub(r"\bAudio\s+1\b", "<Audio 1>", clean)
    clean = re.sub(r"<+Audio 1>+", "<Audio 1>", clean)
    clean = re.sub(r"\bImage\s+\d+\b[,.]?", "", clean, flags=re.I)
    if clean.count('"') % 2:
        clean = clean.replace('"', "")  # an unpaired quote is LLM noise, not a quotation
    return re.sub(r"\s+", " ", clean).strip()


def _strip_cut_directive(text: str) -> str:
    clean = normalize_description(text)
    previous = None
    while clean and clean != previous:
        previous = clean
        clean = re.sub(r"^(?:(?:the\s+camera\s+)?cuts?\s+to|cut\s+to)\s*(?::|[-–—.]|\s)*", "", clean, flags=re.I).strip()
    return clean


def post_cut_text(text: str) -> str:
    """The text after ``[Shot N] At MM:SS.mmm,`` (``miniMaxH3PostCutShotText``, grammar clean-up omitted)."""
    return f"the camera cuts. {_clean_sentence_start(_strip_cut_directive(text))}"


def _leading_json(source: str) -> str:
    source = source.strip()
    start = source.find("{")
    if start < 0:
        return source
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(source)):
        ch = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    return source


def parse_shot_descriptions(raw: str, expected: int) -> List[str]:
    """The creative shot descriptions from the LLM's JSON, exactly ``expected`` of them."""
    text = str(raw or "").strip()
    if not text:
        raise ShotPromptError("The LLM returned an empty MiniMax shot-description payload.", "EMPTY_SHOTS")
    try:
        parsed = json.loads(_leading_json(text))
    except ValueError as exc:
        raise ShotPromptError(f"The LLM did not return valid JSON shot descriptions. Raw output:\n{text[:600]}", "BAD_JSON") from exc
    shots = parsed.get("shots") if isinstance(parsed, dict) else None
    shots = shots if isinstance(shots, list) else []
    if len(shots) > expected:
        raise ShotPromptError(f"The LLM returned {len(shots)} shot descriptions, but {expected} were expected.", "WRONG_SHOT_COUNT")
    shots = list(shots) + [{}] * (expected - len(shots))
    result = []
    for index, item in enumerate(shots):
        description = item if isinstance(item, str) else str((item or {}).get("description") or (item or {}).get("text") or (item or {}).get("shot") or "").strip()
        if not description:
            result.append(FALLBACK_SHOT)
            continue
        if re.search(r"\[\s*Shot\s+\d+\s*\]", description, re.I) or re.search(r"\bAt\s+\d{1,2}:\d{2}(?:\.\d{1,3})?\b", description, re.I):
            raise ShotPromptError(f"The LLM included shot labels or cut times inside shot {index + 1}.", "LABELS_IN_SHOT")
        positive = strip_negative_sentences(description)
        result.append(normalize_description(positive) if positive else FALLBACK_SHOT)
    return result


def shot_body(descriptions: List[str], cut_plan: Dict[str, Any]) -> str:
    plan = shot_plan(cut_plan)
    if len(descriptions) != len(plan):
        raise ShotPromptError(f"Cannot assemble MiniMax shots: expected {len(plan)} descriptions, got {len(descriptions)}.", "WRONG_SHOT_COUNT")
    blocks = []
    for shot, description in zip(plan, descriptions):
        if shot["number"] == 1:
            blocks.append(f"[Shot 1] {description}".strip())
        else:
            blocks.append(f"[Shot {shot['number']}] At {shot['timecode']}, {post_cut_text(description)}".strip())
    return "\n\n".join(blocks)


def assemble_prompt(descriptions: List[str], cut_plan: Dict[str, Any], style: str) -> str:
    """The saved prompt for a reference-to-video scene."""
    return f"{opening_wrapper(style)}{shot_body(descriptions, cut_plan)}".strip()


AUDIO_DEFINITION = (
    "<Audio 1> is the complete synchronized song and vocal track for the target video, reused as the target video's "
    "complete final soundtrack and timing reference."
)
_PURPOSE = {
    "subject": "character identity, face, hair, clothing, and body-proportion reference",
    "extra": "character identity, face, hair, clothing, and body-proportion reference for this non-speaking extra",
    "location": "environment, location, architecture, layout, and atmosphere reference",
    "ingredients": "ordered ingredients, props, identity, or visual-detail reference",
}
_DEFAULT_SOUNDSCAPE = "Subtle location-appropriate ambience, physical movement sounds, breathing, and expressive performance sounds support the scene."


def _compact_description(text: str, limit: int) -> str:
    """A reference description cut at a clause or word boundary (``miniMaxH3CompactReferenceDescription``)."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    clean = re.sub(r"[.;,\s]+$", "", clean)
    if len(clean) <= limit:
        return clean
    window = clean[: limit + 1]
    comma = max(window.rfind(","), window.rfind(";"))
    cut = comma if comma >= int(limit * 0.55) else window.rfind(" ")
    return re.sub(r"[.;,\s]+$", "", window[: cut if cut > 0 else limit]).strip()


def reference_frame(items: List[Dict[str, Any]], cut_plan: Dict[str, Any], style: str, audio_mode: str,
                    audio_direction: str = "") -> Dict[str, str]:
    """The text a reference-to-video prompt needs around its ``detailed_description`` section.

    ``items`` are the scene's ordered reference images (``scene_inputs.ordered_reference_items``). Picture numbers
    follow that order and subject numbers skip start/end frames, the same numbering ``reference_labels`` gives the LLM.
    Returns ``head`` (subject_definitions, summary, retention_analysis) and ``tail`` (overall_soundscape,
    non_diegetic_music).
    """
    built_in = audio_mode == "built_in_audio"
    shots = ", ".join(f"[Shot {shot['number']}]" for shot in shot_plan(cut_plan)) or "[Shot 1]"
    definitions: List[str] = []
    retention: List[str] = []
    featuring: List[str] = []
    number = 0
    for index, item in enumerate(items, start=1):
        kind = str(item.get("kind") or "")
        picture = f"<Picture {index}>"
        if kind == "start_frame":
            definitions.append(f"{picture} is the first frame of [Shot 1], used as the exact opening composition and visual-state anchor.")
            retention.append(f"{picture} ([Shot 1] first frame): fully_preserved - the opening composition, subject placement, environment, lighting, and visual state are retained as the starting frame.")
            continue
        if kind == "end_frame":
            last = shot_plan(cut_plan)[-1]["number"] if shot_plan(cut_plan) else 1
            definitions.append(f"{picture} is the last frame of [Shot {last}], used as the exact final composition and visual-state anchor.")
            retention.append(f"{picture} ([Shot {last}] last frame): fully_preserved - the final composition, subject placement, environment, lighting, and visual state are reached exactly at the end.")
            continue
        number += 1
        label = f"<Subject {number}>"
        raw_name = re.sub(r"\s+", " ", str(item.get("label") or item.get("name") or "")).strip() or ("location" if kind == "location" else "reference")
        name = "environment" if kind == "location" else re.sub(r"^(?:the\s+)+", "the ", raw_name, flags=re.I)
        noun = name if kind in ("subject", "extra", "location") else f"{name} reference"
        phrase = noun if re.match(r"^(?:the|a|an)\s+", noun, re.I) else f"the {noun}"
        purpose = _PURPOSE.get(kind, "visual reference")
        description = _compact_description(item.get("description"), 240)
        if kind == "subject" and number == 1:
            definitions.append(f"{label} is {phrase} in {picture}; {picture} is the visual authority for {purpose}.")
        else:
            definitions.append(f"{label} is {phrase} in {picture}, used as {purpose}{': ' + description + '.' if description else '.'}")
        if kind in ("subject", "extra"):
            keep = "reference identity, face, hair, wardrobe, and accessories" if kind == "subject" else "identity and wardrobe"
        else:
            keep = (_compact_description(item.get("description"), 140) or name) + " and its reference identity"
        retention.append(f"{label} (appears in {shots}): fully_preserved - {keep} remain consistent.")
        featuring.append(f"{label} ({name})")
    tasks = []
    if any(str(i.get("kind")) == "start_frame" for i in items):
        tasks.append("keyframe completion")
    tasks.append("reference generation")
    if not built_in:
        tasks.append("audio reuse")
        definitions.append(AUDIO_DEFINITION)
        retention.append("<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.")
    subject_text = " and ".join(featuring) if featuring else "the described target scene"
    audio_text = (" MiniMax generates the native audio requested by the scene." if built_in
                  else " <Audio 1> is reused as the complete soundtrack and timing reference.")
    head = (
        "subject_definitions:\n" + "\n".join(definitions)
        + f"\n\nsummary:\n[{' + '.join(tasks)}] The target video is a {style or 'photorealistic cinematic'} scene featuring {subject_text}.{audio_text}"
        + "\n\nretention_analysis:\n" + "\n".join(retention) + "\n\n"
    )
    if built_in:
        tail = (f"\n\noverall_soundscape:\n{audio_direction.strip() or _DEFAULT_SOUNDSCAPE}"
                "\n\nnon_diegetic_music:\nThe scene's native soundtrack follows the requested musical and atmospheric direction.")
    else:
        tail = ("\n\noverall_soundscape:\n<Audio 1> remains the sole complete audience-facing soundtrack with its original "
                "mix, timing, vocals, music, and dynamics intact.\n\nnon_diegetic_music:\n<Audio 1> is reused as the "
                "complete audience-facing song/music track.")
    return {"head": head, "tail": tail}


def wrap_reference_prompt(prompt: str, frame: Dict[str, str]) -> str:
    """The saved prompt with its reference definitions and soundscape. A prompt that already has them is returned as is."""
    text = str(prompt or "").strip()
    if not text or "subject_definitions:" in text:
        return text
    return f"{frame['head']}{text}{frame['tail']}"


def character_budget(cut_plan: Dict[str, Any], style: str, target_limit: int = 6500) -> Dict[str, int]:
    """How many characters the shot descriptions may use in total (``miniMaxH3PromptCharacterBudget``)."""
    target = max(0, min(HARD_LIMIT, int(target_limit)))
    fixed = len(assemble_prompt([""] * len(shot_plan(cut_plan)), cut_plan, style))
    return {"hard_limit": HARD_LIMIT, "target_limit": target, "fixed_chars": fixed, "shot_chars": max(0, target - fixed),
            "shot_count": len(shot_plan(cut_plan))}


def validate_prompt(prompt: str, cut_plan: Dict[str, Any]) -> str:
    """Raise ``ShotPromptError`` unless the prompt follows the H3 format. Returns the prompt."""
    text = str(prompt or "").strip()
    if not text:
        raise ShotPromptError("The assembled MiniMax H3 prompt is empty.", "EMPTY_PROMPT")
    if len(text) > HARD_LIMIT:
        raise ShotPromptError(f"The MiniMax H3 prompt is {len(text)} characters, over the {HARD_LIMIT} maximum by {len(text) - HARD_LIMIT}.",
                              "MINIMAX_H3_PROMPT_TOO_LONG", len(text))
    if "detailed_description:" not in text:
        raise ShotPromptError("The MiniMax H3 prompt is missing the detailed_description section.", "MISSING_SECTION")
    creative = text[text.index("detailed_description:") + len("detailed_description:"):]
    plan = shot_plan(cut_plan)
    for shot in plan:
        found = re.findall(rf"\[Shot\s+{shot['number']}\]", creative)
        if len(found) != 1:
            raise ShotPromptError(f"Expected exactly one [Shot {shot['number']}] block, found {len(found)}.", "SHOT_COUNT")
        if shot["number"] > 1 and not re.search(rf"\[Shot\s+{shot['number']}\]\s+At\s+{re.escape(shot['timecode'])},", creative):
            raise ShotPromptError(f"[Shot {shot['number']}] must begin with At {shot['timecode']},.", "SHOT_TIMING")
    unexpected = [m for m in re.findall(r"\[Shot\s+(\d+)\]", creative) if int(m) > len(plan)]
    if unexpected:
        raise ShotPromptError(f"The MiniMax H3 prompt contains unexpected [Shot {unexpected[0]}].", "SHOT_COUNT")
    last = f"[Shot {plan[-1]['number']}]"
    final = creative[creative.index(last) + len(last):].strip()
    if len(final) < 24 or not re.search(r"[.!?…]\s*$", final):
        raise ShotPromptError("The MiniMax H3 prompt has an incomplete final shot description.", "INCOMPLETE_FINAL_SHOT")
    if re.search(r"<(?:Subject|Picture|Video|Audio)\b[^>\n]{0,40}(?:$|\n)", text, re.I | re.M):
        raise ShotPromptError("The MiniMax H3 prompt contains an incomplete reference label.", "BAD_REFERENCE_LABEL")
    return text


def reference_labels(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """``<Subject N>`` labels for the ordered reference images (start/end frames are pictures, not subjects)."""
    labels = []
    number = 0
    for index, item in enumerate(items, start=1):
        kind = str(item.get("kind") or "")
        if kind in ("start_frame", "end_frame"):
            continue
        number += 1
        labels.append({
            "label": f"<Subject {number}>",
            "picture": f"<Picture {index}>",
            "kind": "location" if kind == "location" else ("extra" if kind == "extra" else "subject"),
            "name": str(item.get("label") or item.get("name") or "").strip(),
        })
    return labels


def lyric_lines(lyric_text: str) -> List[str]:
    """The scene's lyric as clean single lines (section tags and blanks dropped)."""
    lines = []
    for raw in re.split(r"[\r\n]+", str(lyric_text or "")):
        line = re.sub(r"\s+", " ", raw).strip()
        if line and not re.fullmatch(r"\[[^\]]*\]", line):
            lines.append(line)
    return lines


def lyric_chunks(lyric_text: str, shot_count: int) -> List[str]:
    """Split the lyric lines, in order, into ``shot_count`` groups. A single shot gets all of it."""
    lines = lyric_lines(lyric_text)
    count = max(1, int(shot_count))
    if not lines:
        return [""] * count
    if count == 1:
        return [" ".join(lines)]
    size, extra = divmod(len(lines), count)
    chunks, start = [], 0
    for index in range(count):
        take = size + (1 if index < extra else 0)
        chunks.append(" ".join(lines[start:start + take]))
        start += take
    return chunks


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower().replace("’", "'"))


def _quote_existing(description: str, lyric: str) -> Optional[str]:
    """Wrap an unquoted copy of ``lyric`` in the description in double quotes. None when it is not there."""
    words = _words(lyric)
    if not words:
        return None
    pattern = r"[\W_]*".join(re.escape(w) for w in words)
    match = re.search(pattern, description.replace("’", "'"), re.IGNORECASE)
    if not match:
        return None
    start, end = match.span()
    return description[:start] + '"' + description[start:end].strip() + '"' + description[end:]


def _has_quoted_lyric(description: str, lyric: str) -> bool:
    words = _words(lyric)
    if not words:
        return True
    lead = " ".join(words[:6])
    for quoted in re.findall(r'["“]([^"”]+)["”]', description):
        if lead in " ".join(_words(quoted)):
            return True
    return False


def ensure_quoted_lyrics(descriptions: List[str], lyric_text: str, performer: str = "the singer") -> List[str]:
    """Make every shot carry its lyric in double quotes: quote a plain copy, or add the sentence.

    The saved prompt must say what the character sings, so this does not rely on the LLM.
    """
    chunks = lyric_chunks(lyric_text, len(descriptions))
    result = []
    for description, chunk in zip(descriptions, chunks):
        if not chunk or _has_quoted_lyric(description, chunk):
            result.append(description)
            continue
        quoted = _quote_existing(description, chunk)
        if quoted is not None:
            result.append(quoted)
            continue
        sentence = f'{performer} sings the lyric line, "{chunk.rstrip(" ,;")}".'
        base = description.rstrip()
        result.append(f"{base} {sentence}" if base else sentence)
    return result


def build_shot_task(
    *,
    mode_label: str,
    duration: float,
    aspect_ratio: str,
    audio_mode: str,
    cut_plan: Dict[str, Any],
    style: str,
    labels: List[Dict[str, str]],
    camera_speed: float,
    character_speed: float,
    camera_guidance: str = "",
    character_guidance: str = "",
    lyric_text: str = "",
    visual_only: bool = False,
    no_character: bool = False,
    story_beat: str = "",
    lyric_section: str = "",
    scene_notes: str = "",
    subject_text: str = "",
    location_text: str = "",
    seed: str = "",
    target_limit: int = 7000,
) -> str:
    """The ``MiniMax H3 shot-description task`` text the saved instruction expects."""
    plan = shot_plan(cut_plan)
    exact = f"{round(float(duration), 3):g}"
    budget = character_budget(cut_plan, style, target_limit)
    per_shot = max(1, budget["shot_chars"] // max(1, len(plan)))
    cast = [l for l in labels if l["kind"] in ("subject", "extra")] if not no_character else []

    def compact(value: Any, limit: int = 900) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text[:limit].rstrip() + "..." if len(text) > limit else text

    def labelize(value: Any) -> str:
        text = str(value or "")
        for item in cast:
            name = item["name"]
            if name:
                text = re.sub(rf"\b{re.escape(name)}\b", item["label"], text, flags=re.I)
        return text

    parts = [
        "MiniMax H3 shot-description task.",
        f'Return exactly {len(plan)} JSON shot description{"" if len(plan) == 1 else "s"}: {{"shots":[{{"description":"..."}}]}}',
        "Write only creative shot text. Do not include [Shot N] labels, cut times, markdown, or analysis.",
        f"Mode: {mode_label}",
        f"Duration: {exact}s, {aspect_ratio}",
        f"Audio mode: {'Built-in MiniMax audio' if audio_mode == 'built_in_audio' else 'Input Audio 1 preserved by Builder'}",
        f"Shot count: {len(plan)}",
        cut_plan_instruction(cut_plan),
        f"MANDATORY CHARACTER BUDGET: The combined text inside all {len(plan)} JSON description values must not exceed {budget['shot_chars']} "
        f"characters total (about {per_shot} per shot). Stay within this combined limit; be concise without omitting required subjects, actions, or camera direction.",
        "SHOT FORMAT — MANDATORY: Write each shot as 2 to 4 complete sentences, about 60 to 110 words, in this order. "
        "1) Camera: the opening framing, one named camera move with its direction and speed, and the ending framing. "
        "2) Subject action: each person in the cast does their own continuous physical action, written by what the body does. "
        f"3) Life detail: one or two small human movements placed inside the action, for example {'; '.join(life_movements(character_speed, seed))}. "
        "4) Light and set: one short line using only the mapped location's own light and objects. "
        "Describe only what the camera sees. Show emotion only as visible movement. Do not use feeling words such as feel, grief, longing, memory, soul, or emotional. "
        "Write complete grammatical prose, not notes, labels, or fragments. "
        "Character appearance is already carried by the reference images and the Builder. Do not list clothing, hair, accessories, jewelry, or facial features. "
        "Mention a garment or feature only when it moves or reacts in the action, in one brief clause at most.",
        f"MOTION ENERGY — MANDATORY: {motion_energy_text(camera_speed, character_speed)} This scene is {exact} seconds long. "
        "Fill the full shot length with continuous action at this energy. Do not slow down or hold still.",
    ]
    cut_times = [s["timecode"] for s in plan[1:]]
    parts.append(f"Builder cut times for your planning only: {', '.join(cut_times)}. Do not write these times." if cut_times
                 else "Continuous shot: return one description only.")
    if cast:
        first = cast[0]
        lines = [
            f'CAST FOR THIS SCENE — MANDATORY: Character names in the scene text below have been replaced by these labels. Refer to every character in the shot text by label, '
            f'writing the label followed by the assigned name in parentheses on its first mention in each shot, for example "{first["label"]} ({first["name"] or "assigned name"})", '
            "then the label alone. Never refer to a character only by name or pronoun."
        ]
        for item in cast:
            role = "does not sing, speak, or lip sync; acts and reacts silently with a relaxed closed mouth" if (visual_only or not lyric_text) \
                else "SINGS and lip-syncs the exact lyric line with visible mouth, lip, and jaw movement"
            lines.append(f"- {item['label']} ({item['name'] or 'mapped subject'}): {role}")
        lines.append("ONLY THESE PEOPLE APPEAR — MANDATORY: " + ", ".join(f"{i['label']} ({i['name'] or 'mapped subject'})" for i in cast)
                     + ". No other person, hand, arm, shadow, reflection, silhouette, or crowd appears in any shot.")
        parts.append("\n".join(lines))
    parts.append(f"Camera speed: {camera_speed:g}/10{' - ' + compact(camera_guidance, 240) if camera_guidance else ''}.")
    parts.append(f"Character speed: {character_speed:g}/10{' - ' + compact(character_guidance, 240) if character_guidance else ''}.")
    if camera_speed >= 7:
        parts.append("Camera rule: use energetic, visibly active camera movement; avoid slow/static/locked-off language.")
    if no_character:
        parts.append("Vocal performance: no visible character / no lip sync. Do not invent a visible singer or speaker.")
    elif visual_only or not lyric_text:
        parts.append("Vocal performance: no exact lyric is assigned to this scene. Do not add singing, speaking, or lip sync.")
    else:
        chunks = lyric_chunks(lyric_text, len(plan))
        singer = f"{cast[0]['label']} ({cast[0]['name']})" if cast and cast[0]["name"] else (cast[0]["label"] if cast else "the singer")
        parts.append("MANDATORY VOCAL PERFORMANCE: The assigned subject is visibly singing the exact supplied lyric/audio during this scene. "
                     "Use the stable visible subject label. Show clear, natural mouth, lip, jaw, and facial movement synchronized to the audible vocal. "
                     "LYRIC IN THE SHOT — MANDATORY: write the exact lyric words, unchanged, inside the shot description in double quotes, "
                     f'introduced like this: {singer} sings the lyric line, "the exact words". Every lyric line given below must appear in a shot, in order.')
        if len(chunks) == 1:
            parts.append(f"Exact lyric line:\n{compact(chunks[0])}")
        else:
            parts.append("Exact lyric lines, one group per shot:\n" + "\n".join(
                f"Shot {i + 1}: {compact(chunk)}" for i, chunk in enumerate(chunks) if chunk))
    for label, value, limit in (
        ("Story beat", labelize(story_beat), 900),
        ("Lyric section", lyric_section, 900),
        ("Scene notes", labelize(scene_notes), 900),
        ("Subject (identity context only; do not restate appearance or clothing in the shot text)",
         "No main character is visible in this scene." if no_character else subject_text, 900),
        ("Location", location_text, 900),
    ):
        text = compact(value, limit)
        if text:
            parts.append(f"{label}:\n{text}")
    pictures = [f"{l['picture']}" for l in labels]
    if pictures:
        parts.append(f"Available renderer reference labels: {', '.join(pictures)}. Do not define labels in the shot text.")
    return "\n\n".join(parts)
