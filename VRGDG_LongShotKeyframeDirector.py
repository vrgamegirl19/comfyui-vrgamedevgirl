"""Plan, extract, and collect four coordinated LongShot keyframes."""

from __future__ import annotations

import json
import re

import torch


REFERENCE_SETUPS = ("no reference images", "image 1 only", "images 1 and 2")
REFERENCE_ROLES = (
    "protagonist", "second character", "multi-character cast sheet", "location",
    "product", "creature or prop", "wardrobe", "visual style inspiration",
)
IMAGE_MODELS = ("GPT Image 2", "Nano Banana", "generic image model")


def _clean(value):
    return str(value or "").strip()


def _extract_json(value):
    raw = _clean(value)
    raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The Keyframe Director response does not contain a JSON object.")
    try:
        result = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The Keyframe Director returned invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(result, dict):
        raise ValueError("The Keyframe Director plan must be a JSON object.")
    return result


def _reference_instructions(setup, role_1, role_2):
    if setup == "no reference images":
        return (
            "No user reference images are attached. Establish the complete visual world in "
            "Keyframe 1. Repeat the same precise identity, wardrobe, location, and style details "
            "in all four prompts. Generated Keyframe 1 may be passed to later image generators."
        )
    lines = [f"Attached Image 1 role: {role_1}. Preserve its relevant visible details."]
    if setup == "images 1 and 2":
        lines.append(f"Attached Image 2 role: {role_2}. Preserve its relevant visible details.")
    lines.extend((
        "A character sheet describes one character; do not reproduce its panel layout, duplicate "
        "figures, blanked faces, labels, or neutral studio background in the story image.",
        "A location reference defines architecture, materials, light sources, and physical "
        "geography. Show it from the required camera position instead of copying its exact view.",
    ))
    return "\n".join(lines)


class VRGDG_LongShotKeyframeDirectorContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "director_plan": ("STRING", {"forceInput": True, "multiline": True}),
            "image_model": (list(IMAGE_MODELS), {"default": "GPT Image 2"}),
            "chunk_duration": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 60.0, "step": 0.1}),
            "allow_cuts": ("BOOLEAN", {"default": False}),
            "reference_setup": (list(REFERENCE_SETUPS), {"default": "no reference images"}),
            "image_1_role": (list(REFERENCE_ROLES), {"default": "protagonist"}),
            "image_2_role": (list(REFERENCE_ROLES), {"default": "location"}),
            "extra_image_direction": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("keyframe_planner_instructions",)
    FUNCTION = "build"
    CATEGORY = "VRGDG/Video/Long Shot"
    DESCRIPTION = "Creates one LLM request for four coordinated 16:9 LongShot keyframe prompts."

    def build(self, director_plan, image_model, chunk_duration, allow_cuts,
              reference_setup, image_1_role, image_2_role, extra_image_direction):
        if image_model not in IMAGE_MODELS:
            raise ValueError(f"Unknown image_model: {image_model}")
        if reference_setup not in REFERENCE_SETUPS:
            raise ValueError(f"Unknown reference_setup: {reference_setup}")
        duration = float(chunk_duration)
        times = (0.0, duration * 2, duration * 3, duration * 4)
        editing = (
            "Cuts are allowed. Each keyframe may represent a distinct motivated shot, but story, "
            "identity, wardrobe, geography, props, lighting, and action remain continuous."
            if allow_cuts else
            "This is one continuous physical camera take. Every camera position must be reachable "
            "from the preceding position in the available time. Maintain lens behavior and screen "
            "direction; do not cross through walls, furniture, equipment, or people; do not use "
            "occlusion or foreground wipes as hidden cuts."
        )
        extra = _clean(extra_image_direction) or "No additional image direction was supplied."
        refs = _reference_instructions(reference_setup, image_1_role, image_2_role)
        prompt = f"""You are the cinematic Keyframe Director for a four-chunk MiniMax H3 LongShot.

Convert the supplied Auto Director story plan into FOUR separate, coordinated 16:9 image-generation prompts for {image_model}. Do not create a storyboard grid. The prompts will be sent separately to four image-generation nodes.

AUTO DIRECTOR PLAN
{_clean(director_plan)}

REFERENCE HANDLING
{refs}

EXTRA USER DIRECTION
{extra}

KEYFRAME MAPPING
- Keyframe 1: global 0.000 seconds, exact first frame of Chunk 1.
- Keyframe 2: global {times[1]:.3f} seconds, exact last frame of Chunk 2.
- Keyframe 3: global {times[2]:.3f} seconds, exact last frame of Chunk 3.
- Keyframe 4: global {times[3]:.3f} seconds, exact last frame of Chunk 4.
Chunk 1 has no supplied last frame. Its ending and Chunk 2's opening use motion continuity.

CAMERA CONTINUITY
{editing}
Plan one stable physical map and one camera route. Make every composition meaningfully different through camera position, height, foreground relationships, subject blocking, and environmental reveal, never through a simple zoom or crop. A changed view must reveal what physically exists on that side of the characters.

REFERENCE FIDELITY
- Preserve identity, face, hair, body proportions, wardrobe, distinguishing details, product design, and important props across all four prompts.
- Preserve fixed architecture and light-source positions. Describe believable perspective and lighting changes as the camera moves.
- Existing keyframes and references guide continuity without forcing the same camera angle.
- Never copy reference captions, arrows, watermarks, blanked faces, labels, or panel borders.

PERFORMANCE AND COMPOSITION
- Show chronological progress through pose, expression, gaze, hands, interaction, and environmental action.
- Depict one exact photographic instant per prompt. Do not claim a still image is moving or lip-synced.
- Avoid impossible anatomy, duplicate people, extra limbs, fused hands, and changing props.
- For performers, preserve instrument, strap, microphone, and stand geometry. Keep mouths visible when planned video requires speech or singing.
- Demand a separate full-frame 16:9 image with no text, captions, subtitles, labels, numbering, borders, split screens, collages, or storyboard grids.

PROMPT RULES
- Each image_prompt must be standalone and 1,200 characters or fewer.
- Repeat only identity and world details needed by the image model. Avoid screenplay timing, H3 fields, dialogue transcripts, audio instructions, and abstract commentary.
- State subject placement, exact physical action, facial emotion, camera position and height, framing, lens character, foreground/background relationship, lighting, and environment.
- Keep the same people, wardrobe, props, and world in every prompt.

OUTPUT
Return only valid JSON with exactly four keyframes and no Markdown fences:
{{
  "continuity_bible": "compact shared identity, wardrobe, location, lighting and camera-path rules",
  "keyframes": [
    {{
      "keyframe": 1,
      "workflow_role": "chunk 1 first frame",
      "global_time_seconds": 0.0,
      "camera_position": "physical camera position and direction",
      "subject_state": "exact pose, expression, interaction and prop state",
      "handoff_logic": "how this composition connects physically to the next keyframe",
      "image_prompt": "standalone 16:9 prompt, at most 1200 characters"
    }}
  ]
}}

Before returning, verify exactly four objects numbered 1–4, exact mapped times, stable identities and geography, a plausible camera route, no text or grids, prompts at most 1,200 characters, and valid JSON."""
        return (prompt,)


class VRGDG_LongShotKeyframePromptExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"keyframe_plan": ("STRING", {"forceInput": True, "multiline": True})}}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("keyframe_1_prompt", "keyframe_2_prompt", "keyframe_3_prompt",
                    "keyframe_4_prompt", "continuity_bible", "status")
    FUNCTION = "extract"
    CATEGORY = "VRGDG/Video/Long Shot"
    DESCRIPTION = "Extracts four standalone image prompts from Keyframe Director JSON."

    def extract(self, keyframe_plan):
        plan = _extract_json(keyframe_plan)
        frames = plan.get("keyframes")
        if not isinstance(frames, list):
            raise ValueError("The Keyframe Director plan has no keyframes array.")
        by_number = {}
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            try:
                number = int(frame.get("keyframe", index + 1))
            except (TypeError, ValueError):
                continue
            by_number[number] = frame
        missing = [number for number in range(1, 5) if number not in by_number]
        if missing:
            raise ValueError(f"The Keyframe Director plan is missing keyframes: {missing}.")
        prompts = []
        for number in range(1, 5):
            prompt = _clean(by_number[number].get("image_prompt"))
            if not prompt:
                raise ValueError(f"Keyframe {number} has no image_prompt.")
            if len(prompt) > 1200:
                raise ValueError(f"Keyframe {number} image_prompt is {len(prompt)} characters; maximum is 1200.")
            prompts.append(prompt)
        bible = _clean(plan.get("continuity_bible"))
        status = "Four keyframe prompts validated: " + ", ".join(
            f"K{i + 1}={len(prompt)} chars" for i, prompt in enumerate(prompts)
        )
        return (*prompts, bible, status)


class VRGDG_LongShotCollectFourKeyframes:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "keyframe_1": ("IMAGE",), "keyframe_2": ("IMAGE",),
            "keyframe_3": ("IMAGE",), "keyframe_4": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("chunk_1_first_frame", "chunk_2_last_frame", "chunk_3_last_frame",
                    "chunk_4_last_frame", "keyframe_batch")
    FUNCTION = "collect"
    CATEGORY = "VRGDG/Video/Long Shot"
    DESCRIPTION = "Waits for all four images, exposes their LongShot roles, and builds a preview batch."

    def collect(self, keyframe_1, keyframe_2, keyframe_3, keyframe_4):
        frames = [keyframe_1[:1], keyframe_2[:1], keyframe_3[:1], keyframe_4[:1]]
        shapes = [tuple(frame.shape[1:]) for frame in frames]
        if len(set(shapes)) != 1:
            raise ValueError(
                "All four keyframes must have the same height, width, and channels to form a "
                f"batch; received {shapes}. Generate every image at the same 16:9 resolution."
            )
        batch = torch.cat(frames, dim=0)
        return frames[0], frames[1], frames[2], frames[3], batch


NODE_CLASS_MAPPINGS = {
    "VRGDG_LongShotKeyframeDirectorContext": VRGDG_LongShotKeyframeDirectorContext,
    "VRGDG_LongShotKeyframePromptExtractor": VRGDG_LongShotKeyframePromptExtractor,
    "VRGDG_LongShotCollectFourKeyframes": VRGDG_LongShotCollectFourKeyframes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LongShotKeyframeDirectorContext": "VRGDG LongShot Keyframe Director Context",
    "VRGDG_LongShotKeyframePromptExtractor": "VRGDG LongShot Keyframe Prompt Extractor",
    "VRGDG_LongShotCollectFourKeyframes": "VRGDG LongShot Collect 4 Keyframes",
}
