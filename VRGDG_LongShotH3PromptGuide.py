"""Mode-selected MiniMax H3 knowledge for the LongShot LLM context node."""

from __future__ import annotations


CORE_RULES = """CORE RULES
- User story, characters, actions, and exact current dialogue/lyrics are authoritative.
- Analyze attached images according to their assigned roles. Preserve identity, wardrobe, props, geography, lighting, pose, screen direction, and camera state.
- Describe chronological action, facial acting, interaction, and a concrete physical camera path. Do not substitute vague words such as "cinematic" for camera direction.
- Never add captions, subtitles, credits, logos, signs, or readable text unless requested.
- Return only the finished H3 prompt. Maximum 7,000 characters including whitespace; target 5,500–6,500. Count before returning and state each rule once.
"""

CUSTOM_AUDIO_RULES = """CUSTOM AUDIO — USE ONLY THIS FORMAT
Use once, in order: subject_definitions:, summary:, retention_analysis:, detailed_description:, overall_soundscape:, non_diegetic_music:.
Define <Audio 1>; begin summary with [audio reuse] without frame anchors or [keyframe completion + audio reuse] with anchors; mark <Audio 1> fully_copy in retention_analysis. Reuse its exact waveform without regenerating, rewriting, restarting, extending, or rearranging it. Put action, camera, performance, synchronization, and exact words in detailed_description. Put copied physical ambience in overall_soundscape and copied audience-only music in non_diegetic_music without repeating words. Suggested budgets: 650/450/550/3900/300/180 characters respectively.
"""

BUILTIN_AUDIO_RULES = """BUILT-IN AUDIO — USE ONLY THIS FORMAT
Use once, in order: integrated_multimodal_description:, overall_soundscape:, non_diegetic_music:. Never mention <Audio 1>, fully_copy, subject_definitions, summary, or retention_analysis. H3 generates synchronized picture and sound. Put action, camera, speakers, exact words, delivery, and synchronized events in integrated_multimodal_description; ambience and physical sounds in overall_soundscape; audience-only music in non_diegetic_music or N/A. Suggested budgets: 5700/450/250 characters.
"""


def _shot_rules(allow_cuts):
    if allow_cuts:
        return (
            "EDITING — CUTS: Use sequential [Shot N]. Shot 1 has no timestamp; every later shot "
            "starts with [Shot N] At MM:SS.mmm using strictly increasing local time. Preserve "
            "identity, geography, action, and audio across motivated cuts. Use <scenetrans> only "
            "when one line truly crosses a cut."
        )
    return (
        "EDITING — ONE TAKE: Use exactly one [Shot 1]. No cuts, transitions, angle jumps, hidden "
        "edits, digital zooms, foreground wipes, or frame-filling occlusions. Maintain lens behavior "
        "and the established side of the action. Use physically reachable camera travel and begin "
        "large endpoint changes early enough to avoid a final snap."
    )


def _frame_rules(frame_setup, endpoint_seconds):
    endpoint = f"{float(endpoint_seconds):.2f}"
    return {
        "first frame only": (
            "FRAMES — I2VA: Attached Image 1 is the exact opening at 0.00 seconds. Begin there and "
            "develop active motion; no supplied endpoint. The native first-frame guide is not a "
            "Picture reference."
        ),
        "first + last frames": (
            f"FRAMES — FL2VA: Image 1 is the exact opening at 0.00 seconds; Image 2 is the exact "
            f"endpoint at {endpoint} seconds. Describe one believable path between them. Both are "
            "native frame guides, not Picture references."
        ),
        "previous ending only": (
            "FRAMES — CONTINUATION: Image 1 is the actual retained ending of the preceding chunk. "
            "Continue its action and camera momentum immediately; no supplied endpoint."
        ),
        "previous ending + last frame": (
            f"FRAMES — CONTINUATION + ENDPOINT: Image 1 is the actual preceding ending; Image 2 is "
            f"the native last-frame guide at {endpoint} seconds. Continue from Image 1 and approach "
            "Image 2 gradually without teleporting or replacing the scene."
        ),
        "no frame anchors": (
            "FRAMES — T2VA: No explicit frame anchor. For later chunks, infer the opening from the "
            "observed previous ending and continuity excerpt; leave a clear moving handoff."
        ),
    }[frame_setup]


def _vocal_rules(vocal_mode, audio_mode):
    if vocal_mode == "instrumental / no vocals":
        return "VOCALS — NONE: Add no speech, singing, lyrics, vocal sounds, or mouth synchronization."
    if vocal_mode == "singing":
        source = "<Audio 1>" if audio_mode == "custom audio" else "the generated vocal"
        return (
            f"VOCALS — SINGING: Perform only the exact supplied lyrics, once, inside "
            f"<d>[English] exact words</d>, visibly synchronized to {source} through breaths, jaw, "
            "vowels, and consonant closures."
        )
    source = "<Audio 1>" if audio_mode == "custom audio" else "the generated voices"
    return (
        f"VOCALS — DIALOGUE: Bind each stable ID to one unmistakable visible person. Introduce every "
        f"line with its speaker ID and use only <d>[English] exact words</d>, synchronized to {source}. "
        "[English] has no closing tag; never write [/English]. Never invent, repeat, swap, merge, or "
        "transfer lines. While one person speaks, visible listeners remain silent with lips closed "
        "and react nonverbally."
    )


def _timeline_rules(prefix_seconds):
    prefix = max(0.0, float(prefix_seconds))
    if prefix <= 0:
        return "TIMELINE: No discarded prefix; current words may begin at local 00:00.000."
    stamp = f"00:{prefix:06.3f}"
    return (
        f"TIMELINE: Local 00:00.000–{stamp} is discarded previous motion/audio context. Start no new "
        f"current-chunk words before {stamp}; never assign the previous voice to a new speaker."
    )


def _panel_times(panel_count, prefix_seconds, endpoint_seconds):
    if panel_count < 2:
        return [float(prefix_seconds)]
    start, end = float(prefix_seconds), float(endpoint_seconds)
    return [start + (end - start) * index / (panel_count - 1) for index in range(panel_count)]


def _storyboard_rules(panel_count, layout, image_slot, allow_cuts,
                      prefix_seconds, endpoint_seconds):
    times = _panel_times(int(panel_count), prefix_seconds, endpoint_seconds)
    mapping = "; ".join(
        f"Panel {index + 1} at {value:.3f}s" for index, value in enumerate(times)
    )
    transition = (
        "Treat panels as editorial shot compositions. Shot 1 includes opening continuity; start "
        "later panel shots at their mapped timestamps."
        if allow_cuts else
        "Treat panels as visual states along the single physical take. Interpolate pose, action, "
        "camera position, and environment continuously between them."
    )
    return f"""VISUAL REFERENCE — STORYBOARD GRID ONLY
Attached {image_slot} is this chunk's {int(panel_count)}-panel storyboard grid in {layout} layout, read left-to-right and then top-to-bottom. Analyze every panel in that order. In the H3 prompt call the grid <Picture 1>, because it must connect to the chunk's reference_image socket. The grid is planning imagery only: render full-screen video and never reproduce its grid, borders, panels, labels, captions, or text.
Panel mapping: {mapping}.
{transition} Use only this chunk's grid; do not ask the model to choose which panels matter. Preserve panel chronology while adding believable intermediate action."""


def _normal_reference_rules():
    return (
        "VISUAL REFERENCE — NORMAL: Use attached non-frame images only for their stated character, "
        "location, product, prop, or style roles. Do not treat them as a storyboard or invent panels."
    )


def build_longshot_h3_guide(audio_mode, frame_setup, allow_cuts, vocal_mode,
                            endpoint_seconds, prefix_seconds=0.0,
                            visual_reference_mode="normal references",
                            storyboard_panel_count=4,
                            storyboard_layout="automatic",
                            storyboard_image_slot="Image 3"):
    """Return only rule blocks selected by the node's current settings."""
    blocks = [
        CORE_RULES.strip(),
        (CUSTOM_AUDIO_RULES if audio_mode == "custom audio" else BUILTIN_AUDIO_RULES).strip(),
        _shot_rules(bool(allow_cuts)),
        _frame_rules(frame_setup, endpoint_seconds),
        _vocal_rules(vocal_mode, audio_mode),
        _timeline_rules(prefix_seconds),
    ]
    if visual_reference_mode == "storyboard grid":
        blocks.append(_storyboard_rules(
            storyboard_panel_count, storyboard_layout, storyboard_image_slot,
            bool(allow_cuts), prefix_seconds, endpoint_seconds,
        ))
    else:
        blocks.append(_normal_reference_rules())
    return "\n\n".join(blocks)
