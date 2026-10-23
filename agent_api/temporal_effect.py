"""The Temporal / World Effect contract the Video Builder appends to a finished MiniMax prompt.

Python twin of ``storyboardTemporalWorldEffectForScene`` in ``web/storyboard_builder/video_style.mjs`` (its
``exact_verbiage``) and of ``ensureStoryboardRequiredTemporalWorldEffect`` in
``web/music_video_builder/storyboard_bridge.mjs``. Both are checked against ``tests/temporal_effect_cases.json``.
"""

from typing import Any, Dict

PRESETS: Dict[str, Dict[str, str]] = {
    "realtime_subjects_timelapse_world": {"label": "Real-time characters / time-lapse world", "guidance": "Create a clearly separated two-speed reality: protected characters remain in natural real time while only the unprotected background world moves in smooth accelerated time-lapse."},
    "frozen_world": {"label": "Characters move / world frozen", "guidance": "Protected characters move naturally through a world frozen at one instant. Unprotected people, particles, vehicles, liquids, smoke, weather, and environmental motion remain suspended unless the scene explicitly releases one element."},
    "reverse_world": {"label": "Characters forward / world reverses", "guidance": "Protected characters move and perform forward in natural time while unprotected background people and environmental events visibly run in reverse, including recoverable spills, retracing traffic, returning debris, and reversed weather or smoke."},
    "day_night_sweep": {"label": "Real-time characters / day-to-night sweep", "guidance": "Protected characters remain in natural time while the environment passes rapidly through a readable day-to-night or night-to-day cycle, with accelerated sky color, sunlight angle, shadow travel, window light, and practical lights switching on or off."},
    "seasonal_passage": {"label": "Real-time characters / seasons pass", "guidance": "Protected characters remain in natural time while the environment transitions through accelerated seasonal change: vegetation, weather, ground cover, atmospheric color, and daylight evolve coherently without changing the characters' identities or wardrobe unless explicitly requested."},
    "crowd_flow": {"label": "Real-time characters / crowd river", "guidance": "Protected characters remain sharply readable in natural time while anonymous unreferenced extras flow around them as an accelerated crowd river, forming continuous directional streams without duplicating, replacing, or obscuring the protected cast."},
    "looping_background": {"label": "Real-time characters / looping background", "guidance": "Protected characters continue naturally while unprotected background actions repeat in deliberate seamless temporal loops. Keep each loop spatially anchored and visually distinct from the protected characters' unrepeated performance."},
    "delayed_world": {"label": "Characters lead / world echoes behind", "guidance": "Protected characters move in natural time while unprotected environmental reactions lag behind them in visible delayed echoes, as though the world responds one beat late. Preserve physical readability and avoid duplicating the protected characters."},
    "living_shadows": {"label": "Real-time characters / living shadows", "guidance": "Protected characters remain in natural time while cast shadows and environmental shadows move independently at accelerated speed, changing direction and shape without changing the protected characters' bodies, faces, or identity."},
    "reflection_delay": {"label": "Real-time characters / delayed reflections", "guidance": "Protected characters move in natural time while their reflections and the environment's reflections respond with a deliberate temporal delay. Keep the real characters singular and stable; the delayed imagery exists only inside physically plausible reflective surfaces."},
    "gravity_separation": {"label": "Real-time characters / altered-gravity world", "guidance": "Protected characters remain grounded and move in natural time while unprotected loose objects, dust, fabric scraps, droplets, leaves, and environmental debris rise, fall, or drift under visibly altered gravity."},
    "custom": {"label": "Custom temporal effect", "guidance": ""},
}

AUDIO_DIRECTION = ("Temporal speed separation is visual only. Keep supplied or generated dialogue, singing, lip sync, facial timing, "
                   "and primary audio at normal speed; never time-stretch, reverse, or accelerate protected voices.")


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def temporal_world_effect_verbiage(card: Dict[str, Any], defaults: Dict[str, Any]) -> str:
    """The exact temporal / world effect block for one scene, or "" when the scene has no effect.

    ``card`` is a scene card (``temporal_world_effect_override``, ``temporal_world_effect_custom``).
    ``defaults`` is the project's ``builder_storyboard_defaults``.
    """
    override = _text(card.get("temporal_world_effect_override")) or "global"
    key = "" if override == "off" else (override if override != "global" else _text(defaults.get("temporal_world_effect")))
    preset = PRESETS.get(key)
    if not preset:
        return ""
    own_custom = key == "custom" and override != "global"
    custom = _text(card.get("temporal_world_effect_custom") if own_custom else defaults.get("temporal_world_effect_custom"))
    base = custom if key == "custom" else preset["guidance"]
    if not base:
        return ""
    mode = _text(defaults.get("temporal_protected_characters"))
    protected_custom = _text(defaults.get("temporal_protected_custom"))
    if mode == "lead_only":
        protected = "Protect only the first mapped/reference character at natural 1x real-time speed; other mapped characters may receive the selected temporal effect."
    elif mode == "custom" and protected_custom:
        protected = f"Protect only these named mapped/reference characters at natural 1x real-time speed: {protected_custom}."
    else:
        protected = ("Protect every mapped/reference character in the scene at natural 1x real-time speed, including secondary referenced "
                     "characters. Never accelerate, freeze, reverse, echo, duplicate, or temporally distort any protected character.")
    return (
        f"Temporal / World Effect — {preset['label']} — Mandatory:\n\n{base}\n\n{protected}\n\n"
        "Only the unprotected background/world may receive this effect, including location-appropriate environmental motion or "
        "anonymous extras when permitted. Keep the effect secondary; never alter, duplicate, replace, or obscure a mapped/reference character.\n\n"
        "Timestamp staging requirement: every shot must visibly show at least one concrete, readable effect affecting only the unprotected "
        "background/world while protected characters continue moving naturally at normal 1x speed.\n\n"
        f"{AUDIO_DIRECTION}"
    )


def append_temporal_world_effect(prompt: str, verbiage: str) -> str:
    """Put the contract once, last. A copy the LLM wrote inside a shot is removed first, like the Video Builder does."""
    text = _text(prompt)
    if not text or not verbiage:
        return text
    text = text.replace(verbiage, "")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return f"{text.strip()}\n\n{verbiage}".strip()
