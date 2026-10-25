"""Custom-audio policy; twin of web/music_video_builder/lyric_free_performance.mjs."""

import re
from typing import Any, Dict, List, Optional, Tuple

from ..storyboard.theatrical_presets import theatrical_presets


def enabled(value: bool, performance_mode: str, audio_mode: str) -> bool:
    """Apply only to singing with supplied audio."""
    return bool(value) and performance_mode == "singing" and audio_mode != "built_in_audio"


def strip_singing_articulation(text: str) -> str:
    """Remove oral movement directions while retaining action, emotion and quoted words."""
    saved = []

    def protect(match):
        saved.append(match[0])
        return f"VRGDGPROTECTED{len(saved) - 1}TOKEN"

    # A silent person's closed lips ("<Subject 2> keeps his lips pressed together") are kept: they stop that person mouthing the song.
    clean = re.sub(r'<d>[\s\S]*?</d>|“[^”]*”|"[^"]*"|\[[^\]]*\]'
                   r'|\blips\s+(?:(?:stay|stays|staying|remain|remains|remaining|still|kept|firmly|tightly)\s+){0,2}'
                   r'pressed\s+(?:together|shut|closed)\b',
                   protect, str(text or ""), flags=re.I)
    clean = re.sub(r"(\d)\.(?=\d)", r"\1VRGDGDECIMALTOKEN", clean)
    clean = re.sub(
        r"(<Subject\s+\d+>(?:\s+\([^)]*\))?|\b(?:she|he|they))\s+"
        r"(?:keeps?|holds?|lets?|maintains?)\s+(?:(?:her|his|their)\s+)?"
        r"(?:mouth|lips?|jaw)\b[^,;.!?]*,\s*(?:then|and)\s+", r"\1 ", clean, flags=re.I)
    clean = re.sub(
        r"\b(?:her|his|their)\s+(?:mouth|lips?|jaw)\s+"
        r"(?:close|closes|relax|relaxes|settle|settles)\b[^,;.!?]*?\s+and\s+"
        r"(?=(?:she|he|they)\b)", "", clean, flags=re.I)
    clean = re.sub(r"\bwith\s+(?:natural\s+)?(?:mouth|lips?|jaw)\b[^,;.!?]*,\s*", "with ", clean, flags=re.I)
    result = []
    for sentence in re.split(r"(?<=[.!?…])\s+", clean):
        parts = re.split(r"([,;])\s*", sentence)
        rebuilt = ""
        for i in range(0, len(parts), 2):
            clause = parts[i]
            clause = re.sub(r"\s+(?:with|and|while)\s+(?:(?:her|his|their|natural)\s+)*(?:mouth|lips?|jaw)\b[^.!?]*",
                            "", clause, flags=re.I).strip()
            if re.fullmatch(r"(?:after|before|between|during)\s+(?:the\s+)?(?:vocal\s+)?"
                            r"(?:cues?|lines?|phrases?|vocals?|singing)(?:\s+(?:ends?|finishes?))?",
                            clause, flags=re.I) and (
                    i + 2 >= len(parts) or re.search(r"\b(?:mouth\w*|lips?|jaw)\b", parts[i + 2], re.I)):
                continue
            if clause and not re.search(r"\b(?:mouth\w*|lips?|jaw)\b", clause, re.I):
                rebuilt += ((parts[i - 1] or ",") + " " if rebuilt else "") + clause
        rebuilt = re.sub(r"[,;]\s*([.!?])", r"\1", rebuilt).rstrip(" ,;")
        if rebuilt and re.search(r"[.!?…]$", sentence) and not re.search(r"[.!?…]$", rebuilt):
            rebuilt += sentence[-1]
        result.append(rebuilt)
    clean = " ".join(filter(None, result)).replace("VRGDGDECIMALTOKEN", ".")
    clean = re.sub(r"VRGDGPROTECTED(\d+)TOKEN", lambda m: saved[int(m[1])], clean)
    return re.sub(r"\s+", " ", clean).strip()


def merge_contract(description: str, contract: str) -> Tuple[str, str]:
    """Merge named or unambiguously pronoun-owned singing, retaining tags and action."""
    protect = lambda v: re.sub(r"(\d)\.(?=\d)", r"\1DECIMAL", v)
    singing, *rest = re.split(r"(?<=[.!?])\s+", protect(contract))
    actor = re.match(r"(.*?) sings\b", singing, re.I)
    if not actor:
        return description, contract
    actor = actor[1]
    suffix = r"(?:\s*\[[^\]]+\])*\s+sings(?:\s+[a-z]+ly|\s+with passion)?\b"
    owned = re.compile(re.escape(actor) + suffix, re.I)
    pronoun = re.compile(r"\b(?:she|he|they)" + suffix, re.I)
    sentences = re.split(r"(?<=[.!?])\s+", protect(description))
    requested_time = re.search(r" during (.+)\.$", singing)
    preceding = ""
    index = None
    cue = None
    for i, sentence in enumerate(sentences):
        visible = re.sub(r'<d>[\s\S]*?</d>|“[^”]*”|"[^"]*"',
                         lambda m: " " * len(m[0]), sentence, flags=re.I)
        named = owned.search(visible)
        inferred = pronoun.search(visible)
        labels = set(re.findall(r"<Subject\s+\d+>",
                                preceding + visible[:inferred.start() if inferred else 0], re.I))
        match = named or (inferred if len(labels) == 1 and actor in labels else None)
        existing_time = re.search(r" during (\d+(?:DECIMAL\d+)?s\s*[–—-]\s*\d+(?:DECIMAL\d+)?s)", sentence)
        if match and (not requested_time or not existing_time or existing_time[1] == requested_time[1]):
            index, cue = i, match
            break
        preceding += visible + " "
    if index is None:
        return description, contract
    sync = re.sub(r"^.*? sings(?: with passion)?\s*", "", singing, flags=re.I).rstrip(".")
    tail = sentences[index][cue.end():]
    tail = re.sub(r"^(\s*(?:\[[^\]]+\]\s*)*)in sync with\s+(?:the\s+)?(?:audible\s+)?(?:vocals(?:\s+in)?\s*(?:<Audio 1>)?|<Audio 1>)", r"\1", tail, flags=re.I)
    tail = re.sub(r"^(\s*(?:\[[^\]]+\]\s*)*)during\s+\d+(?:DECIMAL\d+)?s\s*[–—-]\s*\d+(?:DECIMAL\d+)?s", r"\1", tail, flags=re.I)
    tail = re.sub(r"^\s+([.,;!?])", r"\1", tail)
    tail = re.sub(r"^\s+", " ", tail)
    sentences[index] = sentences[index][:cue.end()] + f" {sync}" + tail
    canonical = lambda v: re.sub(r"\s+", " ", re.sub(r" sings with passion", " sings", v, flags=re.I)).strip()
    kept = [sentence for i, sentence in enumerate(sentences) if i == index or canonical(sentence) != canonical(singing)]
    return " ".join(kept).replace("DECIMAL", "."), " ".join(rest).replace("DECIMAL", ".")


def clean_description(text: str, lyrics: List[str], preserve_performance: bool = False,
                      single_shot: bool = False) -> str:
    """Remove stale lyric and vocal movement sentences before adding timed direction."""
    clean = re.sub(r"<d>[\s\S]*?</d>", "", str(text or ""), flags=re.I)
    for lyric in sorted(filter(None, lyrics), key=len, reverse=True):
        pattern = r"\s+".join(re.escape(part) for part in lyric.split())
        clean = re.sub(pattern, "", clean, flags=re.I)
    clean = re.sub(r"[\"“”']\s*[\"“”']", "", clean)
    vocal = re.compile(r"\b(?:sing\w*|sang|sung|lip[ -]?sync\w*|lyrics?|vocals?|dialogue|mouth\w*|lips?|jaw)\b", re.I)
    protected = re.sub(r"(\d)\.(?=\d)", r"\1VRGDGDECIMALTOKEN", clean)
    if preserve_performance:
        protected = strip_singing_articulation(protected)
        return re.sub(r"\s+", " ", protected).replace("VRGDGDECIMALTOKEN", ".").strip()
    sentences = re.findall(r"[^.!?…]+[.!?…]+|[^.!?…]+$", protected)
    kept = " ".join(s for s in sentences if not vocal.search(s))
    return re.sub(r"\s+", " ", kept).replace("VRGDGDECIMALTOKEN", ".").strip()


def shot_direction(performer: str, single_shot: bool, timing: str = "",
                   include_expression: bool = True) -> str:
    """One audio-based singing direction without explicit articulation."""
    return (
        f"{performer} sings with passion in sync with the vocals in <Audio 1>"
        + (f" during {timing}" if timing else "") + "."
        + (f" {performer}'s engaged eyes and expressive brows convey the song's intensity." if include_expression else "")
    )


def remaining_contract(description: str, contract: str) -> str:
    singing, *rest = re.split(r"(?<=[.!?])\s+", contract)
    actor = re.match(r"(.*?) sings\b", singing, re.I)
    timing = re.search(r" during (.+)\.$", singing)
    protect = lambda value: re.sub(r"(\d)\.(?=\d)", r"\1DECIMAL", value)
    sentences = re.split(r"(?<=[.!?])\s+", protect(description))
    covered = actor and any(
        re.search(re.escape(actor[1]) + r"\s+sings\b.*\bin sync\b.*<Audio 1>", s, re.I)
        and (not timing or protect(timing[1]) in s) for s in sentences)
    if not covered:
        return contract
    return " ".join(s for s in rest if not (
        re.search(r"\bmouth|\bjaw", s, re.I) and any(
            actor[1] in existing and re.search(r"\bmouth|\bjaw", existing, re.I)
            and re.search(r"audible vocal phrasing", existing, re.I) for existing in sentences)))


_FACIAL_PRESETS = {
    "theatrical": theatrical_presets()["facial"]["direction"],
    "off": (
        ""
    ),
    "": (
        "Use natural expressive facial performance: engaged eyes, subtle natural eye movement, active "
        "brows, subtle cheek and jaw movement, visible emotion that fits the lyric or scene, and "
        "occasional natural blinking."
    ),
    "pop_polished": (
        "Use polished pop-star facial performance: bright eyes, subtle natural eye movement, direct "
        "camera gaze, soft confident smile, playful smirk, relaxed brows, slight head tilts, lips "
        "slightly parted while singing, charming camera-ready expression, and occasional natural "
        "blinking."
    ),
    "pop_flirty": (
        "Use playful pop facial performance: flirty smile, coy glance, subtle natural eye movement, "
        "light pout, glossy pout, raised brows, charming direct gaze, playful smirk, subtle head "
        "tilt, lips slightly parted while singing, and occasional natural blinking."
    ),
    "love_tender": (
        "Use tender love-song facial performance: softened eyes, subtle natural eye movement, warm "
        "smile, affectionate gaze, raised inner brows, gentle head tilt, relaxed cheeks, subtle "
        "vulnerable emotion, and occasional natural blinking."
    ),
    "sad_wounded": (
        "Use wounded sad-song facial performance: lowered gaze, heavy or watery eyes, subtle natural "
        "eye movement, raised inner brows, pinched brows, downturned mouth, trembling lips or chin "
        "when appropriate, defeated expression, and occasional natural blinking."
    ),
    "happy_joyful": (
        "Use joyful facial performance: bright smile, smiling eyes, subtle natural eye movement, "
        "raised cheeks, delighted expression, playful gaze, lifted mouth corners, relaxed brows, head "
        "tilt with smile, and occasional natural blinking."
    ),
    "rock_intense": (
        "Use intense rock facial performance: focused stare, subtle natural eye movement, furrowed "
        "brows, defiant smirk, clenched jaw, gritty emotional strain, sharp eye contact, forceful "
        "singing expression, and occasional natural blinking."
    ),
    "metal_rage": (
        "Use aggressive heavy metal facial performance: fierce stare, subtle natural eye movement, "
        "furrowed brows, wild eyes, clenched jaw, snarling mouth shapes during vocals, bared teeth on "
        "powerful notes, flared nostrils, strained neck intensity, raw emotional scream expression, "
        "and occasional natural blinking."
    ),
    "rap_high_intensity": (
        "Use high-intensity rap facial performance: intense stare, sharp eye contact, subtle natural "
        "eye movement, furrowed brows, animated eyes, confident smirk, tight jaw, mouth open "
        "mid-verse, fast-moving mouth during delivery, challenging look, victory grin, and occasional "
        "natural blinking."
    ),
    "custom": (
        ""
    ),
}


def facial_text(segment: Dict[str, Any], session: Optional[Dict[str, Any]] = None) -> str:
    """Preserve selected facial emotion without introducing vocal articulation."""
    session = session or {}
    scoped = segment.get("scene_defaults")
    scoped = scoped if isinstance(scoped, dict) else {}
    key = str(segment.get("facial_performance") or scoped.get(
        "facial_performance", session.get("default_facial_performance")) or "")
    if key == "off":
        return ""
    custom = str(segment.get("facial_performance_custom") or scoped.get(
        "facial_performance_custom", session.get("default_facial_performance_custom")) or "")
    text = custom if key == "custom" else " ".join(filter(None, [_FACIAL_PRESETS.get(key, ""), custom]))
    clauses = [c.strip() for c in re.split(r"[,.;]", text) if c.strip() and not re.search(
        r"\b(?:mouth\w*|lips?|jaw|sing\w*|sang|sung|vocals?|lyrics?|teeth|notes)\b", c, re.I)]
    return re.sub(r",\s*and\s+", ", ", ", ".join(clauses)) + ("." if clauses else "")


def scene_enabled(session: Dict[str, Any], segment: Dict[str, Any]) -> bool:
    """Resolve the durable project option and effective scene audio mode."""
    from .settings_payload import minimax_h3_settings_for_scene

    performance = str(segment.get("performance_mode") or session.get("video_type") or "singing")
    return enabled(session.get("omit_lyrics_from_video_prompts", False), performance,
                   minimax_h3_settings_for_scene(session, segment).get("audio_mode", "input_audio"))


def scene_performers(session: Dict[str, Any], segment: Dict[str, Any],
                     mode: str) -> Tuple[str, Dict[str, str]]:
    """Resolve actual renderer labels for the selected singers."""
    from .scene_inputs import ordered_reference_items
    from .shot_prompt import reference_labels

    ordered = sorted(session.get("segments") or [], key=lambda s: float(s.get("start") or 0))
    index = next((i for i, s in enumerate(ordered) if s.get("id") == segment.get("id")), 0)
    items = ordered_reference_items(session, segment, mode, index)
    label_items = reference_labels(items)
    labels = {item["name"]: item["label"] for item in label_items if item["kind"] == "subject"}
    for item in items:
        name = item.get("label") or item.get("name")
        subject_id = item.get("source_id") or item.get("id")
        if subject_id and name in labels:
            labels[str(subject_id)] = labels[name]
    singers = [labels[name] for name in segment.get("lyric_singers") or [] if name in labels]
    performer = " and ".join(dict.fromkeys(singers)) or next(iter(labels.values()), "The assigned performer")
    return performer, labels


def directions(segment: Dict[str, Any], cut_plan: Dict[str, Any], performer: str,
               index: int = -1, labels: Optional[Dict[str, str]] = None) -> List[str]:
    """Assign singing only to overlapping vocal intervals; instrumental intervals stay visual."""
    from ..builder.lyric_scenes import is_instrumental_lyric_text

    lyric = str(segment.get("lyric_text") or "")
    if segment.get("no_character_present") or segment.get("lyric_no_lip_sync") or is_instrumental_lyric_text(lyric):
        return []
    times = [0.0] + list(cut_plan.get("cut_times_seconds") or [])
    single = len(times) == 1
    cues = (segment.get("lyric_cue_map") or []) if segment.get("lyric_performance_mode") == "cue_map" else []
    result = []
    for cue_index, cue in enumerate(cues):
        if cue.get("type") == "instrumental":
            continue
        start, end = cue.get("vocal_start", cue.get("start")), cue.get("vocal_end", cue.get("end"))
        start = cue.get("start") if start is None else start
        end = cue.get("end") if end is None else end
        if index >= 0:
            if start is None or end is None:
                if not single and cue_index != index:
                    continue
            else:
                shot_end = times[index + 1] if index + 1 < len(times) else cut_plan["exact_duration_seconds"]
                if float(start) >= float(shot_end) or float(end) <= times[index]:
                    continue
        who = ((labels or {}).get(str(cue.get("singer_id")))
               or (labels or {}).get(str(cue.get("singer_name"))) or performer)
        timing = f"{start}s–{end}s" if start is not None and end is not None else ""
        result.append(shot_direction(who, single, timing))
    if not cues and lyric.strip():
        result.append(shot_direction(performer, single))
    return result


def apply_shots(descriptions: List[str], segment: Dict[str, Any], cut_plan: Dict[str, Any],
                performer: str = "The assigned performer", labels: Optional[Dict[str, str]] = None,
                session: Optional[Dict[str, Any]] = None) -> List[str]:
    """Final assembly safety net, independent of LLM compliance."""
    lyrics = [str(segment.get("lyric_text") or "")] + str(segment.get("lyric_text") or "").splitlines()
    lyrics += [str(c.get("text") or "") for c in segment.get("lyric_cue_map") or []]
    from ..llm.prompts.emotion_expression import emotion_expression_input, has_emotion_expression_input

    interpret_emotion = has_emotion_expression_input(emotion_expression_input(segment, session or {}))
    result = []
    for index, text in enumerate(descriptions):
        contracts = directions(segment, cut_plan, performer, index, labels)
        clean = clean_description(text, lyrics, bool(contracts) and interpret_emotion,
                                  not cut_plan.get("cut_times_seconds"))
        if interpret_emotion:
            contracts = [re.sub(r" (?:[^.]+?'s )?engaged eyes and expressive brows convey the song's intensity\.", "", c.replace("sings with passion", "sings"), flags=re.I)
                         for c in contracts]
            remaining = []
            for contract in contracts:
                clean, extra = merge_contract(clean, contract)
                remaining.append(extra)
            contracts = remaining
        result.append(" ".join(filter(None, [clean or
                      "The camera follows continuous physical action through the scene.",
                      "" if interpret_emotion else facial_text(segment, session), *contracts])))
    return result


def prompt_context(text: str, segment: Dict[str, Any], cut_plan: Dict[str, Any],
                   performer: str = "The assigned performer", labels: Optional[Dict[str, str]] = None,
                   session: Optional[Dict[str, Any]] = None) -> str:
    """Replace literal vocal contracts with audio-based, timed performance instructions."""
    vocal_contract = (r"(?:MANDATORY VOCAL PERFORMANCE|Vocal performance|Exact lyric|Timed singer|"
                      r"Performer / vocal|Multi-performer rule|AUTHORITATIVE PERFORMER)")
    paragraphs = [p for p in str(text).split("\n\n") if not re.match(vocal_contract, p.strip(), re.I)]
    clean = "\n\n".join(paragraphs)
    lyrics = [str(segment.get("lyric_text") or "")] + str(segment.get("lyric_text") or "").splitlines()
    lyrics += [str(c.get("text") or "") for c in segment.get("lyric_cue_map") or []]
    for lyric in sorted(filter(None, lyrics), key=len, reverse=True):
        clean = clean.replace(lyric, "[audio vocal cue]")
    clean = re.sub(r"[^\n.!?]*\b(?:mouth\w*|lips?|jaw)\b[^\n.!?]*[.!?]?", "", clean, flags=re.I)
    single = not cut_plan.get("cut_times_seconds")
    rule = "Do not mention mouth, lip, or jaw movement anywhere in this prompt. Write each assigned singing cue once, only during its audible vocals in Audio 1."
    return (clean + "\n\nLYRIC-FREE CUSTOM AUDIO — MANDATORY:\n"
            "Do not quote, invent, or add lyric lines or <d> tags. " + rule +
            " Instrumental intervals use only visual action and camera direction; do not mention singing in them. "
            "Preserve the supplied audio unchanged. Use selected facial direction through eyes, brows and expression.\n"
            + facial_text(segment, session) + "\n"
            + "\n".join(directions(segment, cut_plan, performer, labels=labels)))
