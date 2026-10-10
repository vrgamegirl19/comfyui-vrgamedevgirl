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


def literal_scene_instruction() -> str:
    """Require explicit anatomy, ownership and spatial continuity in shot prose."""
    return (
        "LITERAL SCENE CLARITY — MANDATORY: H3 follows the final wording literally; it cannot fill in unstated "
        "ownership or spatial connections from your reasoning. Rewrite ambiguous scene-card or image-prompt wording "
        "into a self-contained physical description rather than copying it. Establish the actor before describing "
        "their body parts or worn clothing; use that actor's possessive at first mention and unambiguous pronouns "
        "afterward. A worn boot belongs to a foot on that same person's leg, a glove is worn on their hand, and a "
        "facial reaction belongs to their face. For example: '<Subject 1> strides past the ground-level camera; her "
        "boot, worn on her foot, passes close to the lens during that stride.' Keep foreground limbs and the rest of "
        "their owner as one anatomically connected person at normal scale and consistent depth. A close camera can "
        "enlarge a nearby foot by perspective, but must not stage it as a separate object beside a distant copy of "
        "its owner. If the body is cropped, describe the close framing of that person's body part; do not introduce "
        "another body elsewhere. A genuinely loose garment or detached prop must be explicitly introduced as such and "
        "placed on a surface or in someone's possession. Keep positions consistent as the person and camera move; the "
        "ending must be reachable from the opening through the described action. Use only objects supported by this "
        "scene's directions or mapped references. Adjacent scenes supply context; carry an object forward only when "
        "the current scene explicitly does so. Before returning, read the shot alone as literal staging: identify who "
        "owns every limb, garment, expression and gesture, where each object is, and whether all positions and "
        "movements can coexist. Rewrite any ambiguity without adding a new person, prop or action. Return only the "
        "requested shot JSON, not this check."
    )


def scene_timing_instruction(duration: float, cut_plan: Dict[str, Any]) -> str:
    """Give the creative LLM the real scene and per-shot action budgets."""
    exact = round(float(duration), 3)
    plan = shot_plan(cut_plan)
    windows = []
    for index, shot in enumerate(plan):
        start = float(shot["time"])
        end = float(plan[index + 1]["time"]) if index + 1 < len(plan) else exact
        available = round(max(0.0, end - start), 3)
        windows.append(f"Shot {index + 1}: {start:g}–{end:g}s ({available:g} seconds available).")
    return "\n".join([
        f"SCENE TIMING — MANDATORY: The complete scene lasts exactly {exact:g} seconds.",
        *windows,
        "Use the scene beat and storyboard directions together to stage only what can physically finish within each "
        "shot's available time. Reserve time for the required singing or speaking and any supplied entrance, reveal, "
        "or transition. For a brief shot, express the essential beat through one economical continuous action and the "
        "requested camera move; let performance and camera movement happen together where physically possible. Keep "
        "optional gestures and reactions only when time remains. Do not invent sequential steps, pivots, glances, or "
        "framing changes to fill a checklist. Preserve explicitly requested action and exact vocal words; simplify "
        "optional choreography rather than rushing required performance. Motion speed sets the energy of the chosen "
        "movement, not the number of actions. Before returning, check that the opening, action, camera travel, and "
        "ending fit the available time without an extra cut, rushed final beat, or action continuing beyond the shot.",
    ])


def motion_energy_text(camera_speed: float, character_speed: float) -> str:
    camera = "The requested camera movement is fast and energetic." if camera_speed >= 7 else (
        "The camera moves at a steady pace with a clear direction." if camera_speed >= 4 else "The camera moves slowly, or holds a locked frame.")
    character = "The requested performer action has high energy and decisive physical movement." if character_speed >= 7 else (
        "The requested performer action has steady physical energy." if character_speed >= 4
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
    clean = re.sub(r"\bImage\s+(\d+)\b", r"<Picture \1>", clean, flags=re.I)
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
    anchor_patterns = (
        r"\b(?:in|from|matching|copying|reproducing|preserving|using|following)\s+(?:the\s+)?(?:exact\s+)?(?:composition|framing|camera\s+angle|pose)\s+(?:of|from|in)\s+<Picture\s+\d+>",
        r"<Picture\s+\d+>\s+(?:is|defines|controls|sets)\s+(?:the\s+)?(?:exact\s+)?(?:first|start|opening)\s+(?:frame|composition|framing)",
        r"\b(?:begin|begins|start|starts|open|opens)\s+(?:exactly\s+)?(?:from|on|with)\s+<Picture\s+\d+>",
    )
    if any(re.search(pattern, text, re.I) for pattern in anchor_patterns):
        raise ShotPromptError("Reference to Video cannot use a reference picture as the opening frame or composition.", "MINIMAX_H3_REFERENCE_COMPOSITION_LEAK")
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


def compact_reference_prompt(prompt: str, items: List[Dict[str, Any]]) -> str:
    """Bind pictures within the shot prose, without a reference-definition paragraph."""
    labels = reference_labels(items)
    blocks = re.split(r"(?=\[Shot\s+\d+\])", str(prompt or ""))
    for index, block in enumerate(blocks):
        if not re.match(r"\[Shot\s+\d+\]", block):
            continue
        text = block.strip()
        for item in labels:
            if item["kind"] != "location":
                pattern = re.escape(item["label"]) + r"\s*\(([^)]*)\)"
                text = re.sub(pattern, lambda m: m.group(0) if re.fullmatch(r"\s*S\d+\s*", m.group(1), re.I) else item["label"], text)
            elif item["picture"] not in text:
                text += f" The action takes place in the environment from {item['picture']}."
        blocks[index] = text
    return "\n\n".join(block.strip() for block in blocks if block.strip())


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

    def normalize(value: str) -> str:
        return re.sub(r"[^\w']+", " ", value.lower().replace("’", "'")).strip()

    for description, chunk in zip(descriptions, chunks):
        tagged_lyrics = re.findall(r"<d>\s*\[[^\]]+\]\s*(.*?)</d>", description, re.I | re.S)
        normalized_chunk = normalize(chunk)
        if normalized_chunk and any(normalized_chunk in normalize(re.sub(r"<[^>]+>", "", tagged)) for tagged in tagged_lyrics):
            result.append(description)
            continue
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


def last_shot_text(saved_prompt: str, limit: int = 700) -> str:
    """The last ``[Shot N]`` description of a saved prompt: what is on screen when the next scene starts."""
    text = str(saved_prompt or "")
    starts = [m.end() for m in re.finditer(r"\[Shot\s+\d+\]\s*(?:At\s+[0-9:.]+,\s*)?", text)]
    if not starts:
        return ""
    block = text[starts[-1]:].split("\n\n", 1)[0]
    block = re.sub(r"\s+", " ", block).strip()
    return block[:limit].rstrip() + "..." if len(block) > limit else block


def _location_identity(location: Optional[Dict[str, Any]], fallback: str) -> str:
    name = str((location or {}).get("name") or "").strip() or fallback
    description = " ".join(str((location or {}).get("description") or "").split())
    if not description:
        return name
    match = re.match(r"^.*?[.!?](?:\s|$)", description)
    first = match.group(0).strip() if match else description
    if len(first) > 240:
        first = first[:237].strip() + "..."
    return f"{name} ({first})"


def _location_key(location: Optional[Dict[str, Any]]) -> str:
    location = location or {}
    return " ".join(str(location.get("id") or location.get("name") or location.get("description") or "").strip().lower().split())


def performance_line(has_vocals: bool, part: str) -> str:
    """The sentence that keeps a singing scene singing. ``part`` is ``opening`` (the first frame) or ``transition`` (a movement)."""
    if not has_vocals:
        return ""
    if part == "transition":
        return (" The performer keeps singing the scene's lyrics on camera through the whole movement, lips and mouth in continuous "
                "sync with <Audio 1>, and the movement keeps their face in view (it may travel around them but never hides the face "
                "for more than a moment).")
    return (" The performer is mid-performance, so keep singing the scene's lyrics without a pause from the first frame, lips and mouth "
            "in continuous sync with <Audio 1>.")


def location_continuity_contract(
    current: Optional[Dict[str, Any]],
    previous: Optional[Dict[str, Any]],
    preset: str = "normal",
    custom: str = "",
    hold_seconds: float = 1.5,
    has_vocals: bool = False,
) -> str:
    """Python twin of ``miniMaxH3FrameLocationContinuityContract`` in ``minimax_prompt.mjs``.

    Stays in the established location, or at a change of mapped location performs the chosen transition once.
    Returns "" when the scene has no mapped location.
    """
    current_key = _location_key(current)
    if not current_key:
        return ""
    current_name = _location_identity(current, "the current mapped location")
    previous_key = _location_key(previous)
    if previous_key and previous_key != current_key:
        prev = _location_identity(previous, "the preceding mapped location")
        cur = current_name
        hold = f"{float(hold_seconds):g}"
        common_ending = (
            f"Complete the transition inside this uninterrupted scene and end looking deeper into {cur}, with its mapped geography "
            "filling the image as the location inherited by the following scene."
        )
        directions = {
            "normal": (
                f"LOCATION PHASE — PHYSICAL THRESHOLD TURN: This scene performs the single physical passage from {prev} into {cur}. "
                "Track the subject to a visible doorway, gateway, solid foreground edge, or natural threshold already present in the opening frame. "
                "After the subject crosses its plane, let that foreground edge sweep fully across the image as a natural full-frame occlusion. "
                f"During that continuous occlusion, cross the threshold and execute a clear camera arc around the subject toward {cur}. "
                f"As the occluding edge clears, the camera faces forward into {cur}; {prev} has passed fully beyond the rear camera plane while the subject "
                "and destination reappear through coherent motion and parallax. Describe the threshold crossing, full-frame occlusion, camera arc, and final viewing direction explicitly."
            ),
            "surreal": (
                f"LOCATION PHASE — SURREAL MATERIAL TRANSFORMATION: Transform {prev} progressively into {cur} through imaginative, visually connected material changes that travel across the frame. "
                "Use forms, textures, weather, particles, light, and movement visible in the actual opening image as the transformation source. "
                f"Preserve the subject's continuous identity, action, position, and camera momentum while each physical element evolves into a corresponding element of {cur}. "
                "Make the transformation richly creative, spatially progressive, and complete."
            ),
            "cinematic": (
                "LOCATION PHASE — CINEMATIC CONCEAL AND REVEAL: Choose a visually suitable detail present in the actual opening image, such as the subject's eye, hair, clothing, a shadow, bright light, fog, doorway, or foreground object. "
                f"Move the camera into that detail until it fills the complete frame, carry the camera continuously through the concealed moment, then pull or glide outward to reveal the subject physically present in {cur}. "
                "Preserve the subject's action and camera momentum across the full-frame concealment."
            ),
            "inner_world": (
                f"LOCATION PHASE — INNER WORLD PORTAL: Reveal {cur} living inside a visually suitable surface already present in the opening image, such as an eye reflection, mirror, window, pool, pendant, crystal, smoke formation, or luminous opening. "
                f"Let the destination become visibly dimensional inside that surface, move the camera continuously through it, and emerge with the subject inside the full-scale geography of {cur}. "
                "Treat the portal as one coherent passage with continuous scale, perspective, light, and motion."
            ),
            "match": (
                f"LOCATION PHASE — VISUAL MATCH TRANSITION: Inspect the actual opening image and find a strong shared shape, color, texture, light pattern, or motion that can connect {prev} to {cur}. "
                f"Track that matching element as it fills or commands the composition, then let the same visual form resolve seamlessly as a real element in {cur}. "
                "Carry the subject's movement and camera trajectory through the visual correspondence with precise composition and spatial flow."
            ),
            "motion": (
                "LOCATION PHASE — MOTION-DRIVEN TRANSITION: Use an energetic camera action suited to the actual opening frame, such as a whip pan, rapid orbit, fast push, foreground sweep, or close pass around the subject. "
                f"Let directional motion and natural motion blur carry the complete image across the location boundary, then resolve the same movement and screen direction clearly inside {cur}. "
                "Preserve the subject's action, rhythm, and camera momentum throughout."
            ),
            "creative_auto": (
                f"LOCATION PHASE — CREATIVE IMAGE-AWARE TRANSITION: Inspect the actual opening image, {prev}, and {cur}, then choose the most visually convincing imaginative transition for their specific forms, materials, lighting, subject action, and camera trajectory. "
                "Build one explicit on-screen mechanism with readable progression and continuous movement, using a physical passage, material transformation, cinematic concealment, portal, visual match, motion bridge, or an equally coherent original idea. "
                "Make the chosen mechanism concrete in the shot description."
            ),
            "masked": (
                f"LOCATION PHASE — MASKED CONTINUATION TRANSITION: The renderer already holds the previous scene's last moments, so for the first {hold} seconds simply continue the opening frame's action in {prev} with the same subject, framing, pace, and camera motion. No cut, no change of angle, and no new setup. "
                f"At about {hold} seconds, begin ONE smooth, motivated movement that carries the shot into {cur}. Choose the movement that best suits the opening frame: a camera move (push-in, pull-back, pan, tilt, track, or arc around the subject) or a natural subject move (turning, stepping through, walking on) that the camera follows. "
                "Let the surroundings change progressively through that movement and its parallax, with no wipe, flash, portal, morph, or cut. Describe the move and where the shot has arrived when it finishes."
                + performance_line(has_vocals, "transition")
            ),
        }
        custom_text = " ".join(str(custom or "").split())
        directions["custom"] = (
            f"LOCATION PHASE — CUSTOM TRANSITION: Apply this scene's authored transition direction: {custom_text} "
            f"Translate it into explicit positive visual action that carries the actual opening image from {prev} into {cur} through one coherent continuous camera experience."
            if custom_text else
            f"LOCATION PHASE — CUSTOM TRANSITION FALLBACK: Inspect the actual opening image, {prev}, and {cur}, then create one highly imaginative, visually readable transition whose on-screen mechanism follows the subject and camera momentum into the destination."
        )
        return f"{directions.get(preset) or directions['normal']} {common_ending}"
    return (
        f"LOCATION PHASE — ESTABLISHED CURRENT LOCATION: {current_name} is now the complete established environment. "
        "Continue forward through the exact visible opening state and deeper into its mapped geography. "
        f"Build every newly revealed environmental feature from {current_name}, preserve its established spatial logic, and let the camera trajectory create the next composition. "
        f"End looking deeper into {current_name}, fully grounded in that location."
    )


def continuation_task_text(
    continuation: Dict[str, Any],
    previous_shot: str = "",
    has_vocals: bool = False,
    with_picture: bool = False,
    location_contract: str = "",
) -> str:
    """The rules an LLM needs to write a scene that continues the previous one with ``latent_continuation_masked``.

    Python twin of the masked FRAME-TO-FRAME CONTINUITY block, the opening subject visibility block, the location
    contract and the author's direction block in ``web/music_video_builder/minimax_prompt.mjs``. With ``with_picture``
    the LLM is shown the previous scene's final frame as Attached Picture 1, like the Builder. Without it the previous
    scene's last shot description stands in for the picture.
    """
    hold = f"{float(continuation.get('hold_seconds') or 0.5):g}"
    scene_seconds = float(continuation.get("scene_seconds") or 0.0)
    seconds_left = float(continuation.get("seconds_left") or max(0.0, scene_seconds - float(hold)))
    direction = str(continuation.get("direction") or "").strip()
    follow_up = (
        "then carry on exactly as the AUTHOR'S DIRECTION at the end of this scene concept says. " if direction
        else "then advance them one small natural step at a time. "
    )
    if with_picture:
        parts = [
            "FRAME-TO-FRAME CONTINUITY — HIGHEST PRIORITY:\n"
            "Attached Picture 1 is the previous rendered scene's actual final frame. The renderer already holds the last moments of the previous "
            "scene's motion and audio as the start of this render, so this scene is the very next moment of the same uninterrupted take, not a new shot. "
            "Begin the returned description with exactly: ‘Continuing seamlessly from the previous shot, the camera maintains its established course as’ "
            "and immediately name the same camera movement continuing at the same speed. "
            "Keep the subject's action, pace, pose, framing, camera angle, lighting, wardrobe, and environment exactly as Attached Picture 1 shows them, "
            + follow_up
            + "Do not restage, reset, re-establish, change framing, or cut. "
            + ("The author's direction decides what happens next." if direction
               else "The current story beat, lyrics/audio timing, and mapped location decide where the action goes next, reached through the continuing motion.")
            + performance_line(has_vocals, "opening")
            + " Write every finished shot sentence as a positive description of the desired visible result. Attached Picture 1 remains an LLM-only "
            "observation source; finished prose uses direct visual description and the documented renderer labels.",
            "OPENING SUBJECT VISIBILITY — IMAGE-AWARE: Inspect Attached Picture 1 and inventory the human or character subjects actually visible in its opening composition. "
            "Visible subjects continue from their exact observed position and state. Every currently mapped subject absent from Attached Picture 1 begins physically offscreen. "
            "Bring an offscreen subject into view through a clearly described continuous screen-space event: an entrance through a frame edge, doorway, path, foreground layer, "
            "or the selected transition mechanism, or a camera pan, track, or orbit that reaches and reveals them in a connected position. "
            "State the observed empty or partially occupied opening composition first, then the exact entrance or camera-reveal route, then that subject's performance action. "
            "Their first visible moment occurs through that physical entrance or reveal.",
        ]
    else:
        parts = [
            "CONTINUATION — HIGHEST PRIORITY:\n"
            "This scene is the very next moment of the same uninterrupted take the previous scene ended on. The renderer already "
            "holds the previous scene's last moments as the start of this render, so this is not a new shot. "
            "Begin the returned description with exactly: ‘Continuing seamlessly from the previous shot, the camera maintains its "
            "established course as’ and immediately name the same camera movement continuing at the same speed. "
            "Keep the subject's action, pace, pose, framing, camera angle, lighting, and environment as the previous scene's last shot "
            "leaves them, "
            + follow_up.replace("at the end of this scene concept", "below")
            + "Do not restage, reset, re-establish, change framing, or cut. The camera line of Shot 1 continues the previous "
            "scene's camera and framing, it does not choose a new opening framing." + performance_line(has_vocals, "opening")
        ]
        if previous_shot:
            parts.append(f"PREVIOUS SCENE'S LAST SHOT (what is on screen when this scene starts):\n{previous_shot}")
    if location_contract:
        parts.append(location_contract)
    if direction:
        parts.append(
            f"AUTHOR'S DIRECTION FOR THIS SCENE — MANDATORY, THE FINISHED DESCRIPTION MUST CONTAIN IT: \"{direction}\"\n"
            + (
                f"SCENE TIMING: This scene is {scene_seconds:g} seconds long. The direction starts at {hold} seconds and has to be "
                f"completely finished before the scene ends, which leaves {seconds_left:g} seconds for it. Perform every action of the "
                f"direction in the author's order at a brisk pace that fits those {seconds_left:g} seconds, and end the shot with the "
                "last action fully done, never cut off or left unfinished. "
                if scene_seconds > 0 else ""
            )
            + f"Write the one shot description in two timed parts and put the timing in the text itself. First: \"For the first {hold} seconds, ...\" "
            "continuing the opening frame's action with the same camera motion, framing, and pace. "
            f"Then: \"At about {hold} seconds, ...\" performing every action in the direction above, in the author's order and with the "
            "author's own verbs and objects, as one smooth continuous movement in the same take. "
            "If the direction needs a body position or facing different from "
            + ("Attached Picture 1" if with_picture else "the previous scene's last shot")
            + " (for example standing up, walking, or turning), first describe the natural movement that gets the subject there, inside the same take. "
            "Do not skip or replace any action in the direction. If it is a lot for the time left, perform its actions in quicker "
            "succession rather than leaving any out. Never cut, change shot, or restart the action to reach it. "
            "Write it as the shot's one movement: if the mapped location differs from the previous scene's, let that same movement carry the shot into the new "
            "location instead of adding a second one. Finish by stating where the shot ends."
            + performance_line(has_vocals, "transition")
        )
    return "\n\n".join(parts)


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
    continuation: Optional[Dict[str, Any]] = None,
    previous_shot: str = "",
    with_picture: bool = False,
    location_contract: str = "",
    motion_request: str = "",
    audio_direction: str = "",
    continuity: str = "",
    storyboard_context: str = "",
) -> str:
    """The ``MiniMax H3 shot-description task`` text the saved instruction expects.

    ``motion_request`` (the scene's Video Notes), ``audio_direction``, ``continuity`` and
    ``storyboard_context`` (the scene-card block) are the Builder's "Motion/camera request", staging
    notes and "Storyboard Builder context" sections (``minimax_prompt.mjs``).
    """
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
        "SHOT FORMAT — MANDATORY: Write concise complete sentences, with length and action complexity suited to the available shot time, in this order. "
        "1) Camera: the opening framing, one named camera move with its direction and speed, and the ending framing. "
        "2) Subject action: each person in the cast does their own continuous physical action, written by what the body does. "
        "3) Acting detail: include only context-supported expression or reaction that fits naturally inside the existing action and available time. "
        "4) Light and set: one short line using only the mapped location's own light and objects. "
        "Describe only what the camera sees. Show emotion only as visible movement. Do not use feeling words such as feel, grief, longing, memory, soul, or emotional. "
        "Write complete grammatical prose, not notes, labels, or fragments. "
        "Character appearance is already carried by the reference images and the Builder. Do not list clothing, hair, accessories, jewelry, or facial features. "
        "Mention a garment or feature only when it moves or reacts in the action, in one brief clause at most.",
        f"MOTION ENERGY — MANDATORY: {motion_energy_text(camera_speed, character_speed)} This scene is {exact} seconds long. "
        "Apply this energy to the requested action within the available time; speed does not require additional actions.",
        scene_timing_instruction(duration, cut_plan),
    ]
    cut_times = [s["timecode"] for s in plan[1:]]
    parts.append(f"Builder cut times for your planning only: {', '.join(cut_times)}. Do not write these times." if cut_times
                 else "Continuous shot: return one description only.")
    if cast:
        lines = [
            "CAST FOR THIS SCENE — MANDATORY: Character names in the scene text below have been replaced by these labels. "
            "Identify each character by its subject label on first mention in each shot, then use natural pronouns and possessives when the actor is unambiguous. "
            "For a single character, continue with she/he/they and her/his/their rather than repeating the label for every action. "
            "With multiple characters, repeat a label when the actor or speaker changes or a pronoun would be ambiguous. "
            "Do not append character names or picture origins in parentheses."
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
        ("Motion/camera request", labelize(motion_request), 900),
        ("Subject (identity context only; do not restate appearance or clothing in the shot text)",
         "No main character is visible in this scene." if no_character else subject_text, 900),
        ("Location", location_text, 900),
        ("Manual audio direction for staging only", audio_direction, 900),
        ("Continuity notes for staging only", labelize(continuity), 900),
    ):
        text = compact(value, limit)
        if text:
            parts.append(f"{label}:\n{text}")
    if str(storyboard_context or "").strip():
        parts.append(f"Storyboard Builder context:\n{str(storyboard_context).strip()}")
    pictures = [f"{l['picture']}" for l in labels]
    if pictures:
        parts.append(f"Available renderer reference labels: {', '.join(pictures)}. Do not define labels in the shot text.")
        parts.append("Renderer picture assignments for planning only:\n" + "\n".join(
            f"{item['picture']}: {item['label']} ({item['name']}), {item['kind']} reference." for item in labels))
    if mode_label == "Reference to Video":
        parts.append(
            "REFERENCE SCENE GROUNDING — MANDATORY: Generate a complete new scene from the assigned character and environment pictures. "
            "Character pictures supply identity and appearance; environment pictures supply the set. The prompt determines opening framing, camera angle, staging, pose, composition, and action. Never copy a reference picture's composition, framing, camera angle, or pose as the opening shot. A separately enabled continuation task follows its own previous-frame rules. "
            "A saved image prompt is a proposed scene idea, not proof that its props, poses, or layout exist in a supplied picture. "
            "Use props supported by the mapped references or current explicit scene directions; omit unsupported carryover props. "
            "When a requested action needs a new prop, introduce its appearance and physical placement before using it, rather than an unexplained 'the rusted worktable'. "
            "COHERENT SHOT WRITING: Treat the story beat and scene card as story context to translate into a self-contained, physically coherent shot, "
            "not prose to splice into a camera template. Introduce each prop, its owner or containing object, and its placement before any action or camera instruction refers to it. "
            "For a coat-and-zipper beat, establish 'a battered coat lies across a rusted worktable, its zipper caught half-open' before referring to 'the zipper'. "
            "Establish any character's position relative to those objects before describing interaction. Write the opening setup, action, camera movement, "
            "and final framing in chronological order; do not describe the camera's ending before establishing its target. "
            "Ensure the final framing is physically possible from the stated character and prop positions. Preserve the beat's intended visual emphasis and endpoint; "
            "do not replace a prop-focused ending with a generic singing close-up or invent prop handling merely because a singer is present. "
            "Integrate a required performance only through staging consistent with the scene directions. Before returning the shot, reread it independently "
            "of the beat and resolve every unexplained object, gesture, spatial relationship, and camera target. "
            "STAGING AND ACTION OWNERSHIP: Establish where each character stands and where any handled prop is relative to them. "
            "Make the character the actor: '<Subject 1> grips the zipper pull with her gloved right hand', not 'a studded glove grips the zipper'. "
            "Attribute every gesture and facial reaction to that character; use '<Subject 1> looks down at the zipper, then back toward the camera', not unowned 'eyes flick down and back'. "
            "Do not invent gloves or accessories. Describe the environment as a physical setting around the character, with concrete spatial and lighting relationships; "
            "a picture label identifies the environment reference, not a moving background object. State each singing/audio synchronization direction once, integrated into the action. "
            "LOCATION REFERENCE: Establish the physical setting and bind the location picture to that setting. "
            "Then describe the camera independently. Derive framing, focus, camera motion, staging, and how much of the location is visible "
            "from this scene's story beat, storyboard details, scene-card directions, and selected camera settings. "
            "The location picture supplies the environment's visual appearance; the scene directions determine how it is filmed. "
            "Identify each character by its subject label on first mention in each shot, then use natural pronouns and possessives when the actor is unambiguous. "
            "For a single character, continue with she/he/they and her/his/their rather than repeating the label for every action. "
            "With multiple characters, repeat a label when the actor or speaker changes or a pronoun would be ambiguous. "
            "Do not append character names or picture origins in parentheses. "
            "Do not add a standalone reference-definition paragraph to the shot description."
        )
    parts.append(literal_scene_instruction())
    if continuation:
        # Last, where the model weighs it most. The vocal rule applies only when the scene sings.
        has_vocals = not (visual_only or no_character or not lyric_text)
        parts.append(continuation_task_text(continuation, previous_shot, has_vocals, with_picture, location_contract))
    return "\n\n".join(parts)
