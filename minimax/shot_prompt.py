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


# Cut style "One continuous moment" (MINIMAX_H3_CONTINUOUS_CUTS_* in minimax_prompt.mjs): every cut keeps one unbroken
# performance and only moves the camera.
CONTINUOUS_CUTS_OPENING = (
    "All shots follow one continuous performance, preserving every person's exact position, body orientation, "
    "posture, and ongoing movement across every camera cut. Each new shot continues directly from the final physical "
    "moment of the previous shot, with only the camera angle and framing changing."
)
CONTINUOUS_CUTS_RULES = (
    "CONTINUOUS MOMENT ACROSS CUTS — MANDATORY: All shots are one unbroken moment filmed from different camera angles. "
    "Each later shot starts exactly where the previous shot ended: the same spot, the same position and body "
    "orientation, the same wardrobe, props, light and weather, and the action already in progress. Only the camera "
    "angle, distance and framing change at a cut. The action may move forward naturally (a step, a turn, a gesture "
    "finishing), but never jumps in time, never resets a pose, and never moves anyone to a new place. Begin each later "
    "description by naming the new framing and what it continues from, for example: 'the camera cuts to a low-angle "
    "close-up of <Subject 1>, continuing her raised-hand gesture from the shot before'."
)


def cut_style_value(value: Any) -> str:
    """``storyboardCutStyleValue``: "continuous" or "free"."""
    return "continuous" if str(value or "").strip().lower() == "continuous" else "free"


def apply_cut_style(cut_plan: Dict[str, Any], cut_style: Any) -> Dict[str, Any]:
    """``miniMaxH3ApplyCutStyle``: a plan with cuts is marked ``continuous`` when the style asks for it."""
    plan = dict(cut_plan or {})
    plan["continuous"] = bool(plan.get("cut_times_seconds")) and cut_style_value(cut_style) == "continuous"
    return plan


def cut_plan_instruction(cut_plan: Dict[str, Any]) -> str:
    """Port of ``miniMaxH3OfficialCutPlanInstruction`` (frequency-driven plans), with the continuous-moment rules."""
    text = _frequency_cut_plan_instruction(cut_plan)
    return f"{text} {CONTINUOUS_CUTS_RULES}" if cut_plan.get("continuous") else text


def _frequency_cut_plan_instruction(cut_plan: Dict[str, Any]) -> str:
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
        "\n\n"
        "MULTI-SHOT CONTINUITY — MANDATORY: When a scene has multiple shots, write them as consecutive camera views of one uninterrupted physical performance in the same place and moment. Shot 1 establishes staging. In every later description, write a complete sentence explicitly continuing directly from the previous shot's final moment. At the instant of the cut, preserve the previous ending position, posture, body orientation, eyeline and prop contact, then carry the ongoing movement forward at its existing pace and direction. For a scene with active singing, for example: 'Continuing directly from the previous shot's final moment, <Subject 1> remains seated in the same position on the curb, her torso still angled toward the doorway, and continues the same uninterrupted singing performance in sync with the assigned audio.' Use the actual preceding physical state, not a generic continuity phrase. Do not restart an action, reset a pose, or replace continued performance with an unrelated beauty shot. An ongoing action may span scheduled cuts. Only camera angle, framing and camera movement change at a cut unless the scene explicitly requests a new action or transition. Show any requested physical change developing naturally from the inherited state. During every shot's active vocal interval, explicitly state that the assigned performer continues singing or speaking in sync with the assigned audio. Preserve cue timing and speaker assignments; instrumental, no-lip-sync and silent intervals remain non-vocal. Follow supplied lyric/dialogue allocations without duplicating an already quoted cue or restarting the scene's lyrics after a cut. Use complete grammatical sentences rather than fragments such as 'Still seated on the curb.' Self-contained shot prose and varied camera coverage must preserve the shared staging. Before returning, check every adjacent pair for a matching end-to-start physical state and continuous performance."
    )


# Performance craft for any song or story (MINIMAX_H3_CRAFT_* in minimax_prompt.mjs): what kept faces acting,
# lip-sync readable and props in place on real H3 renders. The emotions come from each scene, never from a stock list.
MINIMAX_H3_CRAFT_ACTING = (
    "PERFORMANCE CRAFT — MANDATORY: Give every visible face its own concrete expression in every shot, "
    "written as what the brows, eyes, smile, head and shoulders do, for example eyes narrowing, brows lifting, a slow smile spreading or the head turning away. "
    "Take each emotion from this scene's lyric meaning, story beat and song energy rather than a stock mood, choose two specific emotions wherever an emotion header is written, "
    "and let the expression change at least once in a longer shot. Scale every expression to the {style} video style and the scene's energy: "
    "stylized, animated or high-energy scenes use big, exaggerated expressions; realistic or quiet scenes use small, precise ones. "
    "Write each emotion as a visible action instead of a feeling word or a face colour: anger is a jabbed finger, gritted teeth or brows slammed down, never a red, flushed or pale face. "
    "Faces stay dry, with no tears, crying, wet eyes, glistening skin or sweat."
)
MINIMAX_H3_CRAFT_SILENT = (
    "SILENT PEOPLE: Name every visible person who is not singing or speaking at that moment and state that their lips stay pressed together, "
    "with their own expression, for example '<Subject 2> watches with her lips pressed together and her brows lifted.'"
)
MINIMAX_H3_CRAFT_FRAMING = (
    "FACES WHILE SINGING OR SPEAKING: Unless a per-shot framing above says otherwise, frame a person who is singing or speaking "
    "from a medium shot to a close-up, with the camera near their eye level, so the face stays large in frame; keep wide and full-body framing for moments without vocals."
)
MINIMAX_H3_CRAFT_CLEAR_FACE = (
    "Keep hands, drinks, food, hair and kisses away from a singing or speaking person's face while words are heard, "
    "and finish any such gesture before the vocal starts."
)
MINIMAX_H3_CRAFT_OPENING = (
    "OPENING AND ENDING: Open Shot 1 on the scene's main subject, and end the last shot on a clear held state such as a pose, a look or a finished gesture."
)
MINIMAX_H3_CRAFT_ACTION = (
    "ACTIONS AND PROPS: Give each shot one main action written as visible steps that reach a clear final state. "
    "Once a prop is picked up, put down, handed over or put on, name it with where it is in every later shot of this scene."
)


def performance_craft_instruction(style: str = "", has_cast: bool = True, has_vocals: bool = False, continuation: bool = False) -> str:
    """Twin of ``miniMaxH3PerformanceCraftInstruction``. A continued scene keeps its inherited framing and opening."""
    lines = []
    if has_cast:
        lines += [MINIMAX_H3_CRAFT_ACTING.replace("{style}", str(style or "chosen").strip() or "chosen"), MINIMAX_H3_CRAFT_SILENT]
        if has_vocals and not continuation:
            lines.append(MINIMAX_H3_CRAFT_FRAMING)
        if has_vocals:
            lines.append(MINIMAX_H3_CRAFT_CLEAR_FACE)
    if not continuation:
        lines.append(MINIMAX_H3_CRAFT_OPENING)
    lines.append(MINIMAX_H3_CRAFT_ACTION)
    return "\n".join(lines)


_CUE_LANGUAGES = re.compile(
    r"^(?:english|spanish|french|german|italian|portuguese|dutch|swedish|norwegian|danish|finnish|polish|czech|russian|ukrainian|greek|"
    r"turkish|arabic|hebrew|hindi|bengali|urdu|japanese|korean|chinese|mandarin|cantonese|thai|vietnamese|indonesian|malay|tagalog|filipino)$",
    re.I)
_CUE_DELIVERY_RULE = (
    "Write each vocal cue as <d>[Language, emotion, emotion, singing] exact words</d>: choose two specific emotions that fit this line's meaning, the story beat and the song, "
    "wrap the 1–3 words the singer leans on (a rhyme word, a key verb) in <i>...</i>, and put <breath> between two lines where the singer breathes. "
    "Keep every word unchanged and in order, and follow the cue with that singer's concrete facial acting."
)


def cue_header(header: Any) -> str:
    """Twin of ``miniMaxH3CueHeader``: the LLM's emotion header with the language first, else "English"."""
    items = [re.sub(r"[^\w '-]", "", item).strip() for item in str(header or "").split(",")]
    items = [item for item in items if item]
    if not items:
        return "English"
    if not _CUE_LANGUAGES.match(items[0]):
        items.insert(0, "English")
    return ", ".join(items[:5])


def cue_delivery(cue_text: str, llm_body: Any) -> str:
    """Twin of ``miniMaxH3CueDelivery``: the cue's own words with the LLM's <i> stress and <breath> marks."""
    canonical = str(cue_text or "").split()

    def has_word(token: str) -> bool:
        return bool(re.search(r"[^\W_]", token))

    def word_key(token: str) -> str:
        return re.sub(r"[^\w']", "", token.lower().replace("’", "'")).replace("_", "")

    marks, italic, breath = [], False, False
    spaced = re.sub(r"<\s*(/?i|breath)\s*>", r" <\1> ", str(llm_body or ""), flags=re.I)
    for token in spaced.split():
        tag = re.fullmatch(r"<(/?i|breath)>", token, re.I)
        name = tag.group(1).lower() if tag else ""
        if name == "i":
            italic = True
        elif name == "/i":
            italic = False
        elif name == "breath":
            breath = True
        elif has_word(re.sub(r"<[^>]*>", "", token)):
            marks.append({"key": word_key(re.sub(r"<[^>]*>", "", token)), "stress": italic, "breath": breath})
            breath = False
    words = [token for token in canonical if has_word(token)]
    if not canonical or len(marks) != len(words) or any(word_key(t) != m["key"] for t, m in zip(words, marks)):
        return " ".join(canonical)
    stressed = sum(1 for mark in marks if mark["stress"])
    use_stress = 0 < stressed <= 4
    out, word_index = [], -1
    for index, token in enumerate(canonical):
        if not has_word(token):
            out.append(token)
            continue
        word_index += 1
        mark = marks[word_index]
        text = token
        if use_stress and mark["stress"]:
            match = re.match(r"^(.*?[\w'’])([^\w'’]*)$", token)
            word, tail = (match.group(1), match.group(2)) if match else (token, "")
            text = f"<i>{word}</i>{tail}"
        out.append(("<breath> " if mark["breath"] and index > 0 else "") + text)
    return re.sub(r"</i>(\s+)<i>", r"\1", " ".join(out))


def timed_cue_rows(segment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The scene's timed cue map (``lyric_performance_mode`` "cue_map", every row with a start), scene-relative. [] otherwise."""
    if str(segment.get("lyric_performance_mode") or "together") != "cue_map":
        return []
    rows = []
    for cue in segment.get("lyric_cue_map") or []:
        if not isinstance(cue, dict):
            continue
        kind = "instrumental" if str(cue.get("type") or "").strip() == "instrumental" else "vocal"
        text = " ".join(str(cue.get("text") or "").split())

        def number(key: str) -> Optional[float]:
            value = cue.get(key)
            try:
                return max(0.0, float(value)) if value not in (None, "") else None
            except (TypeError, ValueError):
                return None

        if kind == "vocal" and not text:
            continue
        rows.append({"type": kind, "text": text, "singer_id": str(cue.get("singer_id") or "").strip(),
                     "singer_name": str(cue.get("singer_name") or "").strip(), "start": number("start"),
                     "vocal_start": number("vocal_start") if kind == "vocal" else None,
                     "vocal_end": number("vocal_end") if kind == "vocal" else None,
                     "action_note": str(cue.get("action_note") or "").strip()})
    if not rows or any(row["start"] is None for row in rows):
        return []
    return rows


def cue_cut_plan(cut_plan: Dict[str, Any], cues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The Builder's cue-driven plan (``miniMaxH3SceneCutPlan``): a hard cut at every later cue's start."""
    duration = float(cut_plan.get("exact_duration_seconds") or 0)
    cuts = sorted({round(float(c["start"]), 3) for c in cues[1:] if 0.04 < float(c["start"]) < duration - 0.04})
    if len(cues) < 2 or not cuts:
        return cut_plan
    plan = dict(cut_plan)
    plan.update(cut_times_seconds=cuts, cut_count=len(cuts), shot_count=len(cuts) + 1, continuous_shot=False, cue_driven=True)
    return plan


def _cue_text(text: str) -> str:
    clean = str(text or "").strip()
    if clean and not re.search(r"[.!?…][\"')\]]?$", re.sub(r"<[^>]+>", "", clean).strip()):
        clean += "."
    return re.sub(r"^(\s*[\"'“‘(]*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), clean)


def _cue_window(cue: Dict[str, Any]) -> str:
    start, end = cue.get("vocal_start"), cue.get("vocal_end")
    if start is None or end is None or end <= start:
        return ""
    return f" from {round(start, 2):g} s to {round(end, 2):g} s"


def cue_shot_contract(cues: List[Dict[str, Any]]) -> str:
    """Twin of ``miniMaxH3CueShotContractText`` for shot cue maps. Each row needs ``label``, the singer's subject label."""
    lines = ["Timed singer/lyric shot contract — authoritative:"]
    for index, cue in enumerate(cues):
        if cue["type"] == "instrumental":
            note = f" Visual action note: {cue['action_note']}" if cue.get("action_note") else ""
            lines.append(f"[Shot {index + 1}]: use only the assigned visual action and camera direction.{note}")
            continue
        window = _cue_window(cue)
        sync = f" The singer sings{window}." if window else ""
        lines.append(f"[Shot {index + 1}]: {cue.get('label') or 'The singer'} is the only performer singing <d>[English] {_cue_text(cue['text'])}</d> "
                     f"from <Audio 1>.{sync} Other visible performers remain silent, mouth closed or naturally reacting.")
    lines.append("Each listed cue is its own shot. Do not swap singers, merge lyric cues, or anticipate a later vocal cue. "
                 "Apply vocal direction only to the assigned vocal row and its exact timing.")
    lines.append(_CUE_DELIVERY_RULE)
    return "\n".join(lines)


_VOCAL_MARKER = re.compile(
    r"<d>|</d>|\b(?:sings?|singing|sung|speaks?|speaking|says?|saying|performs?|performing|dialogue|lip[ -]?sync(?:s|ing|ed)?|lyrics?|vocals?)\b", re.I)


def enforce_cue_shot(description: str, cue: Dict[str, Any]) -> str:
    """Twin of ``enforceMiniMaxH3CueOnShotDescription``: an instrumental shot loses its vocal sentences; a vocal shot keeps
    the LLM's prose and ends with the one canonical cue, carrying the LLM's emotion header and delivery marks."""
    text = normalize_description(description)
    if cue["type"] == "instrumental":
        sentences = re.findall(r"[^.!?…]+[.!?…]+|[^.!?…]+$", text)
        clean = " ".join(s.strip() for s in sentences if not _VOCAL_MARKER.search(s)).strip()
        return normalize_description(clean or FALLBACK_SHOT)
    cue_text = _cue_text(cue["text"])
    llm_cue = re.search(r"<d>\s*(?:\[([^\]]*)\])?\s*([\s\S]*?)</d>", text, re.I)
    header = cue_header(llm_cue.group(1) if llm_cue else "")
    delivered = cue_delivery(cue_text, llm_cue.group(2) if llm_cue else "")
    text = re.sub(r"<Subject\s+\d+>[^.\n]*?is the only visible performer singing[\s\S]*?every other visible performer remains silent\.?", "", text, flags=re.I)
    for variant in sorted({str(cue["text"]).strip(), cue_text.rstrip(".!?…").strip()} - {""}, key=len, reverse=True):
        text = re.sub(r"[“\"']\s*" + re.escape(variant) + r"[.!?…]*\s*[”\"']", "the assigned vocal cue", text, flags=re.I)
    clean = re.sub(r"<d>[\s\S]*?</d>", "the assigned vocal cue", text, flags=re.I)
    clean = re.sub(r"\s+([,.;!?])", r"\1", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    clean = re.sub(r"[,;:]\s*([.!?…])", r"\1", clean).rstrip(".!?… ") or FALLBACK_SHOT.rstrip(".")
    contract = (f"{cue.get('label') or 'The singer'} is the only visible performer singing, in sync with <Audio 1>{_cue_window(cue)}, "
                f"<d>[{header}] {delivered}</d>; every other visible performer remains silent.")
    return normalize_description(f"{clean}. {contract}")


def literal_scene_instruction() -> str:
    """Require natural action-led prose with clear actors and placement."""
    return (
        "LITERAL SCENE CLARITY — MANDATORY: Write natural, fluent shot prose that H3 can follow literally. Lead "
        "physical action with the person doing it; use their possessive for a body part or worn item and clear "
        "pronouns afterward. For example: '<Subject 1> steps past the ground-level camera, her boot briefly crossing "
        "the foreground.' Describe how the camera films that same action instead of independently placing a body part "
        "in the foreground and its owner elsewhere. A close-up identifies whose body part is framed; describe "
        "cropping only when needed to understand the shot. Keep ownership and spatial checks in your reasoning; "
        "finished sentences describe visible action, not explanations of anatomy or routine clothing attachment. "
        "Rewrite ambiguous scene-card or image-prompt wording into ordinary action-led sentences rather than copying "
        "it. Keep character, camera and prop positions consistent from the opening through the ending. Place loose "
        "props explicitly and use only objects supported by this scene's directions or mapped references. Carry an "
        "object from an adjacent scene only when the current scene explicitly does so. Before returning, read the "
        "shot by itself: every action has an actor, every item has a clear owner or placement, and the camera "
        "movement fits the same physical action. Resolve uncertainty with clear verbs and possessives, not "
        "explanatory qualifiers. Return only the requested shot JSON with natural visual prose."
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


def opening_wrapper(style: str, continuous: bool = False, guidance: str = "") -> str:
    """The opening of detailed_description: the style name and its look, then the continuity sentence."""
    continuity = f" {CONTINUOUS_CUTS_OPENING}" if continuous else ""
    look = f": {guidance.strip()}" if guidance and guidance.strip() else "."
    return f"detailed_description:\nThe target video is in a {style or 'photorealistic cinematic'} music-video style{look}{continuity}\n\n"


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

    masked = re.sub(r'<d>[\s\S]*?</d>|["“][^"”]*["”]', mask, text, flags=re.I)
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


def continuous_cut_text(text: str, shot_number: int) -> str:
    """``miniMaxH3ContinuousCutText``: a later shot of a continuous-moment scene says what it carries on from."""
    if re.search(r"\bcontinu", text, re.IGNORECASE):
        return text
    carried = f"the camera cuts, continuing directly from the final moment of Shot {shot_number - 1}."
    if re.match(r"^the camera cuts\.", text, re.IGNORECASE):
        return re.sub(r"^the camera cuts\.", carried, text, count=1, flags=re.IGNORECASE)
    return f"{carried} {text}"


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
            text = post_cut_text(description)
            if cut_plan.get("continuous"):
                text = continuous_cut_text(text, shot["number"])
            blocks.append(f"[Shot {shot['number']}] At {shot['timecode']}, {text}".strip())
    return "\n\n".join(blocks)


def assemble_prompt(descriptions: List[str], cut_plan: Dict[str, Any], style: str, guidance: str = "") -> str:
    """The saved prompt for a reference-to-video scene. ``guidance`` is the video style's look for the opening line."""
    return f"{opening_wrapper(style, bool(cut_plan.get('continuous')), guidance)}{shot_body(descriptions, cut_plan)}".strip()


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
        purpose = str(item.get("purpose") or _PURPOSE.get(kind, "visual reference"))
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


def character_reference_bindings(creative: str, items: List[Dict[str, Any]]) -> str:
    """Keep used character/picture assignments outside the shot action."""
    bindings = []
    number = 0
    for picture, item in enumerate(items, start=1):
        kind = str(item.get("kind") or "")
        if kind in ("start_frame", "end_frame"):
            continue
        number += 1
        label = f"<Subject {number}>"
        if kind not in ("subject", "extra") or label not in creative:
            continue
        purpose = str(item.get("purpose") or _PURPOSE.get(kind, "character appearance reference")).split(";")[0]
        purpose = re.sub(r"\s+reference(?:\s+only)?(?:\s+for.*)?$", "", purpose, flags=re.I)
        bindings.append(f"Use <Picture {picture}> for {label}'s {purpose}.")
    return " ".join(bindings)


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
                f"LOCATION PHASE — MASKED CONTINUATION TRANSITION: The renderer already holds the previous scene's last moments, "
                + (
                    f"so for the first {hold} seconds simply continue the opening frame's action in {prev} with the same subject, framing, pace, and camera motion. No cut, no change of angle, and no new setup. "
                    f"At about {hold} seconds, "
                    if float(hold) > 0 else
                    f"so from the first frame carry on the opening frame's action in {prev} with the same subject, framing, pace, and camera motion. No cut, no change of angle, and no new setup. From the first frame, "
                )
                + f"begin ONE smooth, motivated movement that carries the shot into {cur}. Choose the movement that best suits the opening frame: a camera move (push-in, pull-back, pan, tilt, track, or arc around the subject) or a natural subject move (turning, stepping through, walking on) that the camera follows. "
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
    hold_value = continuation.get("hold_seconds")
    hold = f"{float(0.0 if hold_value is None else hold_value):g}"
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
            + (
                f"Write the one shot description in two timed parts and put the timing in the text itself. First: \"For the first {hold} seconds, ...\" "
                "continuing the opening frame's action with the same camera motion, framing, and pace. "
                f"Then: \"At about {hold} seconds, ...\" performing every action in the direction above, in the author's order and with the "
                "author's own verbs and objects, as one smooth continuous movement in the same take. "
                if float(hold) > 0 else
                "Write the one shot description so the direction starts from the first frame, with no timed lead-in: perform every action "
                "in the direction above, in the author's order and with the author's own verbs and objects, as one smooth continuous movement in the same take. "
            )
            + "If the direction needs a body position or facing different from "
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
    cues: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """The ``MiniMax H3 shot-description task`` text the saved instruction expects.

    ``cues`` are the scene's timed cue rows (``timed_cue_rows``), one per shot, with ``label`` set to the singer's
    subject label. With them each shot gets its own words and vocal window, like the Builder's cue map.

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
        "3) Acting detail: each visible face's concrete expression for this moment, as the PERFORMANCE CRAFT rules below describe. "
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
            if visual_only or not lyric_text:
                role = "does not sing, speak, or lip sync; acts and reacts silently with a relaxed closed mouth"
            elif audio_mode == "built_in_audio":
                role = "SINGS and lip-syncs the exact lyric line with visible mouth, lip, and jaw movement"
            else:
                # Supplied audio drives the mouth, so the prose describes the face and body (SUPPLIED-AUDIO SINGING below).
                role = "SINGS the exact lyric line in sync with <Audio 1>, acting it with the face and body"
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
    elif cues:
        parts.append("TIMED VOCAL CUES ONLY: Follow the timed singer/shot contract exactly. Never place, anticipate, continue, or repeat a lyric in an "
                     "instrumental shot. A performer sings only inside the specifically assigned vocal cue shot.")
        parts.append("Timed singer / shot contract:\n" + cue_shot_contract(cues))
    else:
        chunks = lyric_chunks(lyric_text, len(plan))
        singer = f"{cast[0]['label']} ({cast[0]['name']})" if cast and cast[0]["name"] else (cast[0]["label"] if cast else "the singer")
        mouth = ("Show clear, natural mouth, lip, jaw, and facial movement synchronized to the audible vocal. " if audio_mode == "built_in_audio"
                 else "Audio 1 drives the singer's mouth, so act the line with the face and body. ")
        parts.append("MANDATORY VOCAL PERFORMANCE: The assigned subject is visibly singing the exact supplied lyric/audio during this scene. "
                     f"Use the stable visible subject label. {mouth}"
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
    parts.append(performance_craft_instruction(
        style, has_cast=bool(cast), has_vocals=bool(lyric_text) and not visual_only and not no_character, continuation=bool(continuation)))
    parts.append(literal_scene_instruction())
    if audio_mode != "built_in_audio" and not visual_only and not no_character:
        parts.append("SUPPLIED-AUDIO SINGING: Include the assigned singer and Audio 1 synchronization cue once, only during audible vocals. "
                     "Do not describe mouth, lip or jaw movement or ask for singing from the first frame. Preserve any supplied vocal timing; introductory silence remains non-vocal.")
    if continuation:
        # Last, where the model weighs it most. The vocal rule applies only when the scene sings.
        has_vocals = not (visual_only or no_character or not lyric_text)
        parts.append(continuation_task_text(continuation, previous_shot, has_vocals, with_picture, location_contract))
    return "\n\n".join(parts)
