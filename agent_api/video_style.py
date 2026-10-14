"""The Video Style and FX wording the Video Builder adds to a finished MiniMax prompt.

Python twin of ``storyboardMiniMaxVideoStyleVerbiage`` / ``storyboardSceneVideoStyle`` / ``storyboardFxContract`` in
``web/storyboard_builder/video_style.mjs`` and of ``ensureBuilderManagedFx`` in ``web/music_video_builder/minimax_prompt.mjs``.
The preset wording is ``video_style_data.json``, exported from the JS by ``scripts/export_video_style_data.mjs``.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

_DATA = json.loads((Path(__file__).with_name("video_style_data.json")).read_text(encoding="utf-8"))
VIDEO_STYLES: Dict[str, Dict[str, str]] = _DATA["video_styles"]
FX_PRESETS: Dict[str, Dict[str, Any]] = _DATA["fx"]

_SHOT = re.compile(r"(\[\s*Shot\s+\d+[^\]]*\]\s*\n?)([\s\S]*?)(?=\n\s*\[\s*Shot\s+\d+[^\]]*\]|$)", re.I)


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def video_style_value(card: Dict[str, Any], defaults: Dict[str, Any]) -> str:
    """The style key for a scene: its own choice, else the project default."""
    return _text(card.get("video_style")) or _text(defaults.get("video_style"))


def video_style_label(card: Dict[str, Any], defaults: Dict[str, Any]) -> str:
    """The name written in the prompt's opening line. A saved label or custom wording is used as typed."""
    value = video_style_value(card, defaults)
    if value == "custom":
        return _text(card.get("video_style_custom") if card.get("video_style") else defaults.get("video_style_custom"))
    return VIDEO_STYLES[value]["label"] if value in VIDEO_STYLES else value


def video_style_verbiage(card: Dict[str, Any], defaults: Dict[str, Any]) -> str:
    """The exact style paragraph ("Label: guidance", or the custom text), or "" when no style is set."""
    value = video_style_value(card, defaults)
    if value == "custom":
        return video_style_label(card, defaults)
    preset = VIDEO_STYLES.get(value)
    return f"{preset['label']}: {preset['guidance']}" if preset and preset["guidance"] else ""


def insert_video_style(prompt: str, verbiage: str) -> str:
    """Put the style paragraph right after the prompt's first paragraph, once."""
    text = _text(prompt)
    if not text or not verbiage or verbiage in text:
        return text
    head, sep, rest = text.partition("\n\n")
    if sep:
        return f"{head}\n\n{verbiage}\n\n{rest}"
    return f"{text}\n\n{verbiage}"


def _custom_fx(raw: Any) -> Dict[str, Any]:
    """``normalizeStoryboardCustomFxJson``: the cues and timing of a custom FX JSON, or {} when it holds none."""
    source = raw
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except ValueError:
            return {}
    if isinstance(source, list):
        source = {"cues": source}
    if not isinstance(source, dict):
        return {}
    items = source.get("cues") if isinstance(source.get("cues"), list) else (source.get("effects") if isinstance(source.get("effects"), list) else [])
    cues: List[str] = []
    for item in items:
        if isinstance(item, str):
            cues.append(item.strip())
        elif isinstance(item, dict):
            cues.append(_text(item.get("text") or item.get("cue") or item.get("description")))
    cues = [c for c in cues if c][:12]
    primary = _text(source.get("primary") or source.get("primary_effect"))
    secondary = _text(source.get("secondary") or source.get("secondary_effect"))
    if primary:
        cues.insert(0, primary)
    if secondary:
        cues.append(secondary)
    if not cues:
        return {}
    return {"cues": list(dict.fromkeys(cues))[:12], "timing": (_text(source.get("timing")) or "on the strongest musical or action accent")[:240]}


def fx_accent(defaults: Dict[str, Any], shot_index: int) -> str:
    """The FX sentence for one shot, or "" when FX are off."""
    preset = _text(defaults.get("fx_preset"))
    if not preset:
        return ""
    custom = _custom_fx(defaults.get("fx_custom_json")) if preset == "custom" else {}
    cues = custom.get("cues") or (FX_PRESETS.get(preset) or {}).get("cues") or []
    if not cues:
        return ""
    timing = custom["timing"] if custom else "on a readable musical, lyric, or action accent when one is present"
    return f"FX accent: {cues[max(0, shot_index) % len(cues)]} Timing: {timing}. Keep the mapped subject stable and readable."


def fx_reserve(defaults: Dict[str, Any], shot_count: int) -> int:
    """Characters the FX sentences will add to a prompt of ``shot_count`` shots."""
    return sum(len(accent) + 1 for accent in (fx_accent(defaults, i) for i in range(shot_count)) if accent)


def apply_fx(core: str, defaults: Dict[str, Any]) -> str:
    """Add each shot's FX sentence at the end of that shot, like the Video Builder. ``core`` holds only the shots."""
    if not _text(core) or not fx_accent(defaults, 0):
        return core
    index = [0]

    def add(match: "re.Match[str]") -> str:
        accent = fx_accent(defaults, index[0])
        index[0] += 1
        body = match.group(2)
        cue = accent.split("FX accent: ", 1)[1].split(" Timing:", 1)[0]
        return match.group(0) if cue in body else f"{match.group(1)}{body.strip()} {accent}"

    return _SHOT.sub(add, core)
