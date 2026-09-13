"""Build clear LLM instructions for one LongShot chunk."""

from __future__ import annotations

import re

try:
    from .VRGDG_LongShotH3PromptGuide import build_longshot_h3_guide
except ImportError:  # Supports direct loading in lightweight tests.
    import importlib.util
    from pathlib import Path

    _guide_path = Path(__file__).with_name("VRGDG_LongShotH3PromptGuide.py")
    _guide_spec = importlib.util.spec_from_file_location(
        "vrgdg_longshot_h3_prompt_guide", _guide_path
    )
    if _guide_spec is None or _guide_spec.loader is None:
        raise ImportError(f"Unable to load LongShot H3 guide from {_guide_path}")
    _guide_module = importlib.util.module_from_spec(_guide_spec)
    _guide_spec.loader.exec_module(_guide_module)
    build_longshot_h3_guide = _guide_module.build_longshot_h3_guide


FPS = 24
FRAME_SETUPS = (
    "first frame only",
    "first + last frames",
    "previous ending only",
    "previous ending + last frame",
    "no frame anchors",
)
VOCAL_MODES = (
    "singing",
    "spoken dialogue",
    "instrumental / no vocals",
)
AUDIO_MODES = (
    "custom audio",
    "built-in audio",
)
VISUAL_REFERENCE_MODES = (
    "normal references",
    "storyboard grid",
)
STORYBOARD_LAYOUTS = (
    "automatic",
    "horizontal",
    "2x2",
    "2x3",
    "3x2",
)
STORYBOARD_IMAGE_SLOTS = ("Image 1", "Image 2", "Image 3", "Image 4")


def _text(value):
    return str(value or "").strip()


def _without_previous_dialogue(value):
    """Keep continuity prose while withholding old words that H3 may repeat."""
    text = _text(value)
    if not text:
        return ""
    return re.sub(
        r"<d(?:\s+[^>]*)?>.*?</d>",
        "[previous dialogue omitted]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _continuity_excerpt(value, limit=1800):
    """Retain the prior prompt's ending action without resending its whole prompt."""
    text = _without_previous_dialogue(value)
    if not text:
        return ""
    for header in ("detailed_description:", "integrated_multimodal_description:"):
        if header in text:
            text = text.split(header, 1)[1]
            break
    for header in ("overall_soundscape:", "non_diegetic_music:"):
        if header in text:
            text = text.split(header, 1)[0]
    text = text.strip()
    if len(text) <= limit:
        return text
    excerpt = text[-limit:]
    boundary = re.search(r"(?:\n\s*\n|(?<=[.!?])\s+)", excerpt[:300])
    if boundary:
        excerpt = excerpt[boundary.end():]
    return "[earlier prior action omitted] " + excerpt.strip()


def _timing(chunk_duration, context_frames, frame_setup, chunk_number):
    body = max(1, round(float(chunk_duration) * FPS))
    uses_previous_tail = frame_setup.startswith("previous ending") or (
        frame_setup == "no frame anchors" and int(chunk_number) > 1
    )
    prefix = int(context_frames) if uses_previous_tail else 0
    requested = prefix + body
    generated = requested + (5 - requested) % 17
    return {
        "body": body,
        "prefix": prefix,
        "generated": generated,
        "padding": generated - prefix - body,
        "endpoint": (prefix + body - 1) / FPS,
    }


class VRGDG_LongShotLLMContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_idea": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "chunk_direction": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "lyrics_this_chunk": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "previous_lyric_carryover": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "chunk_number": ("INT", {"default": 1, "min": 1, "max": 999}),
                "total_chunks": ("INT", {"default": 4, "min": 1, "max": 999}),
                "chunk_duration": (
                    "FLOAT",
                    {"default": 7.5, "min": 0.1, "max": 60.0, "step": 0.1},
                ),
                "context_frames": (["1", "5", "22", "39"], {"default": "22"}),
                "frame_setup": (list(FRAME_SETUPS), {"default": "previous ending + last frame"}),
                "vocal_mode": (list(VOCAL_MODES), {"default": "singing"}),
                "allow_cuts": ("BOOLEAN", {"default": False}),
                "audio_mode": (list(AUDIO_MODES), {"default": "custom audio"}),
                "visual_reference_mode": (
                    list(VISUAL_REFERENCE_MODES),
                    {"default": "normal references"},
                ),
                "storyboard_panel_count": ("INT", {"default": 4, "min": 2, "max": 6}),
                "storyboard_layout": (
                    list(STORYBOARD_LAYOUTS),
                    {"default": "automatic"},
                ),
                "storyboard_image_slot": (
                    list(STORYBOARD_IMAGE_SLOTS),
                    {"default": "Image 3"},
                ),
            },
            "optional": {
                "previous_chunk_prompt": (
                    "STRING",
                    {"forceInput": True, "multiline": True},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("llm_instructions",)
    FUNCTION = "build"
    CATEGORY = "VRGDG/Video/Long Shot"
    DESCRIPTION = (
        "Combines the H3 guide, story, exact dialogue or lyrics, audio mode, timing, frame "
        "roles, and previous chunk prompt into one multimodal LLM instruction string."
    )

    def build(
        self,
        video_idea,
        chunk_direction,
        lyrics_this_chunk,
        previous_lyric_carryover,
        chunk_number,
        total_chunks,
        chunk_duration,
        context_frames,
        frame_setup,
        vocal_mode,
        allow_cuts,
        audio_mode="custom audio",
        visual_reference_mode="normal references",
        storyboard_panel_count=4,
        storyboard_layout="automatic",
        storyboard_image_slot="Image 3",
        previous_chunk_prompt=None,
    ):
        chunk_number = int(chunk_number)
        total_chunks = int(total_chunks)
        if chunk_number < 1 or total_chunks < 1 or chunk_number > total_chunks:
            raise ValueError("chunk_number must be between 1 and total_chunks.")
        if frame_setup not in FRAME_SETUPS:
            raise ValueError(f"Unknown frame_setup: {frame_setup}")
        if vocal_mode not in VOCAL_MODES:
            raise ValueError(f"Unknown vocal_mode: {vocal_mode}")
        if audio_mode not in AUDIO_MODES:
            raise ValueError(f"Unknown audio_mode: {audio_mode}")
        if visual_reference_mode not in VISUAL_REFERENCE_MODES:
            raise ValueError(f"Unknown visual_reference_mode: {visual_reference_mode}")
        if storyboard_layout not in STORYBOARD_LAYOUTS:
            raise ValueError(f"Unknown storyboard_layout: {storyboard_layout}")
        if storyboard_image_slot not in STORYBOARD_IMAGE_SLOTS:
            raise ValueError(f"Unknown storyboard_image_slot: {storyboard_image_slot}")
        storyboard_panel_count = int(storyboard_panel_count)
        capacities = {"2x2": 4, "2x3": 6, "3x2": 6}
        if storyboard_layout in capacities and storyboard_panel_count > capacities[storyboard_layout]:
            raise ValueError(
                f"{storyboard_layout} cannot contain {storyboard_panel_count} panels."
            )
        reserved_slots = {
            "first frame only": {"Image 1"},
            "first + last frames": {"Image 1", "Image 2"},
            "previous ending only": {"Image 1"},
            "previous ending + last frame": {"Image 1", "Image 2"},
            "no frame anchors": set(),
        }[frame_setup]
        if visual_reference_mode == "storyboard grid" and storyboard_image_slot in reserved_slots:
            raise ValueError(
                f"{storyboard_image_slot} is already assigned by frame_setup '{frame_setup}'. "
                "Choose a different storyboard_image_slot."
            )

        timing = _timing(chunk_duration, context_frames, frame_setup, chunk_number)
        idea = _text(video_idea) or "No overall idea was supplied; devise a coherent idea from the images."
        direction = _text(chunk_direction) or "No additional chunk-specific direction was supplied."
        lyrics = _text(lyrics_this_chunk)
        carryover = _text(previous_lyric_carryover)
        previous = _continuity_excerpt(previous_chunk_prompt)

        if not lyrics and vocal_mode != "instrumental / no vocals":
            lyric_contract = (
                "No new exact words supplied. Do not invent dialogue or lyrics."
            )
        else:
            lyric_contract = f"EXACT NEW WORDS — preserve verbatim:\n{lyrics or 'N/A'}"

        if timing["prefix"]:
            context_seconds = timing["prefix"] / FPS
            overlap_contract = (
                f"Discarded prefix: {context_seconds:.6f}s ({timing['prefix']} frames) of prior "
                f"motion/audio. New retained body: {timing['body'] / FPS:.6f}s "
                f"({timing['body']} frames)."
            )
            if carryover:
                overlap_contract += (
                    " Use this only to understand the response; never copy it into the H3 prompt.\n"
                    f"PRIVATE CONTINUITY MEMORY — NEVER OUTPUT:\n{carryover}"
                )
        else:
            overlap_contract = "No discarded prefix; retained action begins at local 00:00.000."

        previous_contract = (
            "PRIOR ENDING CONTINUITY EXCERPT — preserve its final physical state and momentum; "
            f"do not copy old timing or words:\n{previous}"
            if previous
            else "No prior prompt supplied; use the assigned images and story."
        )

        final_contract = (
            "This is the final chunk. Resolve the action and camera move into a satisfying final "
            "state without an unrequested fade or cut."
            if chunk_number == total_chunks
            else
            "This is not the final chunk. End with a specific pose, action, screen direction, and "
            "camera trajectory that can continue into the next chunk."
        )
        task_description = (
            "custom-audio Ref2VA" if audio_mode == "custom audio"
            else "built-in synchronized audiovisual"
        )

        h3_guide = build_longshot_h3_guide(
            audio_mode=audio_mode,
            frame_setup=frame_setup,
            allow_cuts=allow_cuts,
            vocal_mode=vocal_mode,
            endpoint_seconds=timing["endpoint"],
            prefix_seconds=timing["prefix"] / FPS,
            visual_reference_mode=visual_reference_mode,
            storyboard_panel_count=storyboard_panel_count,
            storyboard_layout=storyboard_layout,
            storyboard_image_slot=storyboard_image_slot,
        )

        prompt = f"""TASK
Write the finished MiniMax H3 prompt for LongShot chunk {chunk_number} of {total_chunks}: {task_description}.

SELECTED RULES — settings have already removed irrelevant modes
{h3_guide}

VIDEO IDEA
{idea}

CHUNK DIRECTION
{direction}

TIMING
Requested visible duration: {float(chunk_duration):.3f}s.
{overlap_contract}
Last retained frame: local {timing['endpoint']:.3f}s. Generated: {timing['generated']} frames. Discarded trailing padding: {timing['padding']} frames; place no required action there.

CURRENT PERFORMANCE
{lyric_contract}

CONTINUITY
{previous_contract}
{final_contract}

Return only the H3 prompt with no analysis, notes, JSON, or Markdown fences."""

        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "VRGDG_LongShotLLMContext": VRGDG_LongShotLLMContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LongShotLLMContext": "VRGDG LongShot LLM Prompt Context",
}
