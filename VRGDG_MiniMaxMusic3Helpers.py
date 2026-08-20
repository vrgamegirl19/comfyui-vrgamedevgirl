"""Practical tuning helpers for ComfyUI's built-in MiniMax Music 3 nodes.

They expose and document four useful control stages around the built-in AR
planner and diffusion transformer.
"""

from __future__ import annotations

import copy
import json
import os
import re

import torch
import torch.nn.functional as F


CATEGORY = "VRGDG/Audio/MiniMax Music 3"


# The shared JSON file is also read by the browser extension that visibly
# populates every Caption Builder widget. Keeping one source prevents the
# backend caption and the displayed preset fields from drifting apart.
_PRESET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "VRGDG_Music3CaptionPresets.json")
with open(_PRESET_FILE, "r", encoding="utf-8") as _preset_handle:
    _STYLE_PRESETS = json.load(_preset_handle)


_TUNING_PRESETS = {
    "Balanced / Built-in Baseline": (1.5, 50, 1.7, 30, 1.0),
    "Balanced Quality": (1.6, 45, 1.7, 36, 1.0),
    "Raw Live Rock": (1.7, 40, 1.8, 36, 1.10),
    "Lyrics + Structure Focus": (2.0, 24, 1.75, 36, 1.05),
    "Creative Variations": (1.15, 100, 1.4, 28, 0.90),
    "Intimate + Dry": (1.8, 30, 1.6, 34, 1.05),
    "Aggressive + Dense": (1.55, 70, 1.9, 40, 1.15),
    "Conservative / Stable": (1.5, 30, 1.45, 36, 0.90),
    "Strict Lyrics": (2.1, 20, 1.75, 36, 1.05),
    "Strong Song Structure": (1.95, 28, 1.8, 40, 1.10),
    "Creative Melody": (1.2, 90, 1.5, 32, 0.95),
    "Maximum Arrangement Variation": (1.05, 120, 1.35, 30, 0.85),
    "Stable / Low Artifact": (1.5, 30, 1.45, 38, 0.90),
    "Strong Acoustic Plan": (1.7, 40, 1.85, 38, 1.20),
    "Loose Acoustic Plan": (1.25, 80, 1.4, 30, 0.80),
    "Detailed Rendering": (1.5, 50, 1.75, 48, 1.0),
    "Long-Song Stability": (1.8, 32, 1.6, 44, 1.0),
    "Fast Draft": (1.5, 50, 1.5, 20, 0.95),
    "High-Energy / Dense": (1.6, 65, 1.95, 42, 1.15),
    "Soft / Restrained": (1.7, 35, 1.45, 36, 0.95),
}

_VOCAL_PROFILES = (
    "Use Style Default",
    "Female Lead",
    "Female Alto / Mezzo",
    "Female Soprano / Bright",
    "Female Rock / Rasp",
    "Female Soft / Airy",
    "Male Lead",
    "Male Baritone",
    "Male Tenor",
    "Male Rock / Grit",
    "Male Soft / Intimate",
    "Androgynous / Neutral",
    "Female + Male Duet",
    "Custom / keep fields",
)


def _vocal_profile_tooltip() -> str:
    return (
        "Automatically fills Singer Name/Phrase and Vocal Qualities. "
        "Use Style Default restores the selected style's generic vocal description. "
        "Female or Male Lead gives a general voice; register/timbre variants add more specific direction. "
        "Custom / keep fields leaves both text fields untouched."
    )

_TUNING_DETAILS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "VRGDG_Music3TuningDetails.json")
with open(_TUNING_DETAILS_FILE, "r", encoding="utf-8") as _details_handle:
    _TUNING_PRESET_DETAILS = json.load(_details_handle)


def _tuning_preset_tooltip() -> str:
    lines = ["Caption presets choose WHAT style to request. Tuning presets choose HOW Music3 interprets it.", ""]
    lines.extend(f"{name}: {_TUNING_PRESET_DETAILS.get(name, '')}" for name in _TUNING_PRESETS)
    return "\n".join(lines)


_SECTION_NAMES = {
    "intro": "Intro",
    "verse": "Verse",
    "pre-chorus": "Pre-Chorus",
    "pre chorus": "Pre-Chorus",
    "prechorus": "Pre-Chorus",
    "chorus": "Chorus",
    "final chorus": "Final Chorus",
    "post-chorus": "Post-Chorus",
    "post chorus": "Post-Chorus",
    "postchorus": "Post-Chorus",
    "bridge": "Bridge",
    "refrain": "Refrain",
    "hook": "Hook",
    "break": "Break",
    "breakdown": "Breakdown",
    "interlude": "Interlude",
    "instrumental": "Instrumental",
    "solo": "Solo",
    "outro": "Outro",
}


def _clean_piece(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).strip(" ,.;")


def _normalise_section_tag(raw: str) -> tuple[str, bool]:
    content = _clean_piece(raw).lower()
    match = re.fullmatch(r"(.+?)(?:\s+(\d+))?", content)
    base = match.group(1) if match else content
    number = match.group(2) if match else None
    canonical = _SECTION_NAMES.get(base)
    if canonical is None:
        return raw.strip(), False
    return f"{canonical} {number}" if number else canonical, True


def prepare_music3_lyrics(lyrics: str, normalise_headers: bool, remove_metadata_lines: bool) -> tuple[str, str, int]:
    lines = str(lyrics or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    output = []
    normalised = 0
    removed = 0
    unsupported = []
    lyric_lines = 0
    metadata_prefixes = ("genre:", "mood:", "style:", "tempo:", "instruments:", "production:", "vocal:")
    for original in lines:
        stripped = original.strip()
        if remove_metadata_lines and stripped.lower().startswith(metadata_prefixes):
            removed += 1
            continue
        tag_match = re.fullmatch(r"[\[\(\{]\s*(.+?)\s*[\]\)\}]", stripped)
        if tag_match:
            canonical, supported = _normalise_section_tag(tag_match.group(1))
            if supported:
                rewritten = f"[{canonical}]"
                normalised += int(rewritten != stripped)
                output.append(rewritten if normalise_headers else stripped)
            else:
                output.append(stripped)
                unsupported.append(stripped)
            continue
        output.append(original.rstrip())
        if stripped:
            lyric_lines += 1
    while output and not output[0].strip():
        output.pop(0)
    while output and not output[-1].strip():
        output.pop()
    cleaned = "\n".join(output)
    sections = sum(1 for line in output if re.fullmatch(r"\[.+\]", line.strip()))
    report_parts = [f"{lyric_lines} lyric lines", f"{sections} section headers", f"{normalised} headers normalized"]
    if removed:
        report_parts.append(f"{removed} metadata lines removed")
    if unsupported:
        report_parts.append("Warning: unsupported bracket tags may be treated as lyrics: " + ", ".join(sorted(set(unsupported))))
    if sections == 0 and lyric_lines:
        report_parts.append("Warning: no section headers found; add [Verse 1], [Chorus], etc. for clearer structure")
    return cleaned, "; ".join(report_parts), lyric_lines


class VRGDG_Music3PromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        tip = "Leave blank to use the selected preset. Nonblank text overrides that preset field."
        return {"required": {
            "preset": (list(_STYLE_PRESETS) + ["Custom / fields only"], {"tooltip": "Choosing a preset visibly fills every field below. You can then edit any field for this song."}),
            "vocal_profile": (list(_VOCAL_PROFILES), {"tooltip": _vocal_profile_tooltip()}),
            "singer_name": ("STRING", {"default": "", "tooltip": "Singer name or vocal-direction phrase written into the caption. This is ordinary caption text and remains fully editable."}),
            "style_name": ("STRING", {"default": "", "tooltip": "Short style label written near the beginning of the caption. Presets fill it automatically."}),
            "genre_subgenre": ("STRING", {"default": "", "multiline": True, "tooltip": tip}),
            "mood_themes": ("STRING", {"default": "", "multiline": True, "tooltip": tip}),
            "instruments": ("STRING", {"default": "", "multiline": True, "tooltip": tip}),
            "vocal_qualities": ("STRING", {"default": "", "multiline": True, "tooltip": tip}),
            "production": ("STRING", {"default": "", "multiline": True, "tooltip": tip}),
            "tempo_range": ("STRING", {"default": "", "multiline": True, "tooltip": tip}),
            "song_arc": ("STRING", {"default": "", "multiline": True, "tooltip": tip}),
            "avoid_characteristics": ("STRING", {"default": "", "multiline": True, "tooltip": "Appended as an Avoid sentence. Music3 does not have a separate negative text prompt, so exclusions are guidance rather than guarantees."}),
            "extra_direction": ("STRING", {"default": "", "multiline": True, "tooltip": "Any final song-specific direction, such as 'start with voice and dry guitar only'."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("caption", "caption_breakdown")
    OUTPUT_TOOLTIPS = ("Connect to MiniMax Music3 Text Encode → caption.", "Human-readable audit of the fields used.")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Builds a structured Music3 caption with editable singer, style, genre, vocal, instrument, production, tempo, arc, and avoidance fields."

    def build(self, preset, vocal_profile, singer_name, style_name, genre_subgenre, mood_themes, instruments, vocal_qualities, production, tempo_range, song_arc, avoid_characteristics, extra_direction):
        values = {} if preset == "Custom / fields only" else dict(_STYLE_PRESETS[preset])
        overrides = {
            "singer": singer_name, "style": style_name,
            "genre": genre_subgenre, "mood": mood_themes, "instruments": instruments,
            "vocal": vocal_qualities, "production": production, "tempo": tempo_range,
            "arc": song_arc, "avoid": avoid_characteristics,
        }
        for key, value in overrides.items():
            if _clean_piece(value):
                values[key] = _clean_piece(value)
        prefix = ", ".join(piece for piece in (values.get("singer"), values.get("style")) if piece)
        labelled = [
            ("Genre", values.get("genre")), ("Mood and emotional direction", values.get("mood")),
            ("Instrumentation", values.get("instruments")), ("Lead vocal", values.get("vocal")),
            ("Production", values.get("production")), ("Tempo and groove", values.get("tempo")),
            ("Song arc", values.get("arc")),
        ]
        sentences = [prefix + "."] if prefix else []
        sentences.extend(f"{label}: {value}." for label, value in labelled if value)
        if values.get("avoid"):
            sentences.append(f"Avoid: {values['avoid']}.")
        if _clean_piece(extra_direction):
            sentences.append(f"Additional direction: {_clean_piece(extra_direction)}.")
        caption = " ".join(sentences).replace("..", ".")
        identity = [f"Singer/name phrase: {values.get('singer')}", f"Style label: {values.get('style')}"]
        breakdown = "\n".join([f"Preset: {preset}", f"Vocal profile: {vocal_profile}"] + [line for line in identity if not line.endswith("None")] + [f"{label}: {value}" for label, value in labelled if value] + ([f"Avoid: {values.get('avoid')}"] if values.get("avoid") else []))
        return caption, breakdown


class VRGDG_Music3TuningPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"preset": (list(_TUNING_PRESETS), {"tooltip": _tuning_preset_tooltip()})}}

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("ar_cfg_scale", "ar_top_k", "sampler_cfg", "steps", "conditioning_strength", "notes")
    OUTPUT_TOOLTIPS = (
        "Connect to Music3 Text Encode cfg_scale. Higher follows caption/lyrics more strongly while generating the AR plan.",
        "Connect to Music3 Text Encode top_k. Lower is more constrained/repeatable; higher explores more token alternatives.",
        "Connect to KSampler cfg. Controls denoiser guidance between the acoustic plan and zeroed conditioning.",
        "Connect to KSampler steps. More steps can refine detail but do not guarantee better composition.",
        "Connect through Music3 Conditioning Strength after Text Encode. Scales the acoustic plan presented to the DiT.",
        "Preset-specific notes and explanations.",
    )
    FUNCTION = "select"
    CATEGORY = CATEGORY
    DESCRIPTION = "Behavior-based presets for adherence, structure, variation, stability, plan strength, rendering detail, and speed. These remain independent from genre/style captions."

    def select(self, preset):
        ar_cfg, top_k, sampler_cfg, steps, strength = _TUNING_PRESETS[preset]
        notes = (
            f"{preset}\n"
            f"Expected tendency: {_TUNING_PRESET_DETAILS.get(preset, 'Coordinated Music3 tuning values.')}\n\n"
            f"AR cfg_scale={ar_cfg}: caption/lyrics guidance while the autoregressive acoustic plan is composed.\n"
            f"AR top_k={top_k}: candidate-token diversity; lower is tighter, higher is more exploratory.\n"
            f"Sampler cfg={sampler_cfg}: diffusion guidance toward the completed acoustic plan.\n"
            f"Steps={steps}: denoising iterations.\n"
            f"Conditioning strength={strength}: direct multiplier on the acoustic hidden plan.\n"
            "Change one family at a time and keep the same seed for meaningful comparisons."
        )
        return float(ar_cfg), int(top_k), float(sampler_cfg), int(steps), float(strength), notes


class VRGDG_Music3LyricsPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lyrics": ("STRING", {"default": "[Verse 1]\n\n[Chorus]\n", "multiline": True, "dynamicPrompts": False, "tooltip": "Lyrics only. Square-bracket section labels are structural hints; Verse 1 and Verse 2 are valid."}),
            "normalise_headers": ("BOOLEAN", {"default": True, "tooltip": "Converts recognized (), {}, or inconsistently capitalized section tags to canonical [Verse 1], [Pre-Chorus], etc."}),
            "remove_metadata_lines": ("BOOLEAN", {"default": True, "tooltip": "Removes lines beginning Genre:, Mood:, Style:, Tempo:, Instruments:, Production:, or Vocal: so Music3 does not try to sing them."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("lyrics", "validation_report", "lyric_line_count")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = "Normalizes recognized song-section headers, removes accidental metadata, and warns about bracket text Music3 may mistake for lyrics."

    def prepare(self, lyrics, normalise_headers, remove_metadata_lines):
        return prepare_music3_lyrics(lyrics, normalise_headers, remove_metadata_lines)


class VRGDG_Music3ConditioningStrength:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING", {"tooltip": "Acoustic conditioning from MiniMax Music3 Text Encode."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05, "tooltip": "Directly scales Music3's acoustic hidden plan. 1.0 is built-in behavior; lower loosens it; higher emphasizes it and may become harsh or unstable."}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "Scales Music3's model-specific conditioning_scale metadata without altering the AR plan tensor."

    def apply(self, conditioning, strength):
        output = []
        for hidden, metadata in conditioning:
            meta = dict(metadata)
            current = meta.get("conditioning_scale")
            if torch.is_tensor(current):
                meta["conditioning_scale"] = current * float(strength)
            else:
                meta["conditioning_scale"] = torch.ones((hidden.shape[0], 1, 1), device=hidden.device, dtype=hidden.dtype) * float(strength)
            output.append([hidden, meta])
        return (output,)


def _align_hidden(a: torch.Tensor, b: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim != b.ndim or a.shape[0] != b.shape[0] or a.shape[2:] != b.shape[2:]:
        raise ValueError(f"Music3 conditioning shapes are incompatible: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.shape[1] == b.shape[1]:
        return a, b
    if mode == "crop to shorter":
        length = min(a.shape[1], b.shape[1])
        return a[:, :length], b[:, :length]
    length = max(a.shape[1], b.shape[1])
    if a.shape[1] < length:
        a = F.pad(a, (0, 0, 0, length - a.shape[1]))
    if b.shape[1] < length:
        b = F.pad(b, (0, 0, 0, length - b.shape[1]))
    return a, b


class VRGDG_Music3ConditioningBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_a": ("CONDITIONING", {"tooltip": "Primary Music3 acoustic plan. For useful results, A and B should use the same lyrics, duration, and preferably the same seed."}),
            "conditioning_b": ("CONDITIONING", {"tooltip": "Alternative Music3 acoustic plan from another caption or seed."}),
            "blend_b": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "0 uses only A; 1 uses only B. Intermediate values linearly blend the hidden acoustic plans. Experimental: some combinations can become incoherent."}),
            "length_mode": (["crop to shorter", "pad shorter with zeros"], {"tooltip": "Plans can end at different times. Cropping is safer; padding preserves the longer requested duration."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "blend_report")
    FUNCTION = "blend"
    CATEGORY = CATEGORY
    DESCRIPTION = "Experimental linear blend of two Music3 acoustic hidden plans. Best used for controlled same-lyrics A/B caption experiments."

    def blend(self, conditioning_a, conditioning_b, blend_b, length_mode):
        if not conditioning_a or not conditioning_b:
            raise ValueError("Both Music3 conditionings are required")
        count = min(len(conditioning_a), len(conditioning_b))
        output = []
        shapes = []
        weight = float(blend_b)
        for index in range(count):
            a, meta_a = conditioning_a[index]
            b, meta_b = conditioning_b[index]
            b = b.to(device=a.device, dtype=a.dtype)
            a, b = _align_hidden(a, b, length_mode)
            hidden = torch.lerp(a, b, weight)
            meta = copy.copy(meta_a)
            scale_a = meta_a.get("conditioning_scale")
            scale_b = meta_b.get("conditioning_scale")
            if torch.is_tensor(scale_a) and torch.is_tensor(scale_b):
                meta["conditioning_scale"] = torch.lerp(scale_a, scale_b.to(scale_a), weight)
            output.append([hidden, meta])
            shapes.append(str(tuple(hidden.shape)))
        report = f"Blended {count} conditioning entry/entries: A={1.0-weight:.2f}, B={weight:.2f}, mode={length_mode}, output={', '.join(shapes)}. Experimental; compare with A-only using the same sampler seed."
        return output, report


class VRGDG_Music3SeedBank:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_seed": ("INT", {"default": 1000, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Starting seed for repeatable A/B exploration."}),
            "spacing": ("INT", {"default": 101, "min": 1, "max": 1000000, "tooltip": "Offset between seed candidates."}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("seed_a", "seed_b", "seed_c", "seed_d", "notes")
    FUNCTION = "make"
    CATEGORY = CATEGORY
    DESCRIPTION = "Creates four deterministic seed candidates. Keep one seed fixed while tuning parameters; change seeds only when exploring composition alternatives."

    def make(self, base_seed, spacing):
        maximum = 0xFFFFFFFFFFFFFFFF
        seeds = tuple((int(base_seed) + int(spacing) * index) & maximum for index in range(4))
        return (*seeds, f"Seeds: {seeds}. Use one seed for parameter A/B tests; changing AR seed can change melody, duration, vocal phrasing, and arrangement." )


NODE_CLASS_MAPPINGS = {
    "VRGDG_Music3PromptBuilder": VRGDG_Music3PromptBuilder,
    "VRGDG_Music3TuningPreset": VRGDG_Music3TuningPreset,
    "VRGDG_Music3LyricsPrepare": VRGDG_Music3LyricsPrepare,
    "VRGDG_Music3ConditioningStrength": VRGDG_Music3ConditioningStrength,
    "VRGDG_Music3ConditioningBlend": VRGDG_Music3ConditioningBlend,
    "VRGDG_Music3SeedBank": VRGDG_Music3SeedBank,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_Music3PromptBuilder": "VRGDG Music3 Caption Builder",
    "VRGDG_Music3TuningPreset": "VRGDG Music3 Tuning Presets",
    "VRGDG_Music3LyricsPrepare": "VRGDG Music3 Lyrics Prepare + Validate",
    "VRGDG_Music3ConditioningStrength": "VRGDG Music3 Conditioning Strength",
    "VRGDG_Music3ConditioningBlend": "VRGDG Music3 Conditioning Blend (Experimental)",
    "VRGDG_Music3SeedBank": "VRGDG Music3 Seed Bank",
}
