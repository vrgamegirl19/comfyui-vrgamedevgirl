"""Chunked second pass for the MiniMax H3 2 Pass graph.

A 2 Pass render normally upscales the whole first-pass clip and refines it in one sampler call, which needs memory
for every frame of the scene. When the clip is longer than the "Pass 2 chunking" threshold, this module rewires the
built graph so the same upscale and refinement run in overlapping windows through ``VRGDG_MiniMaxH3ChunkedPass2``
(``minimax/chunked_pass2.py``). The first pass, the audio, the references, the latent saves and the decode are
untouched. A clip at or under the threshold keeps the normal graph.

The windows restart their clock at 0, so a prompt with whole-scene shot times and lyric cues would mislead them. The
second pass therefore reads the same prompt with its lyric text and times removed (``strip_prompt_timing``).
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

from ..minimax.chunk_plan import (
    CHUNK_SECONDS, OVERLAP_MODE, OVERLAP_SECONDS, TAIL_TRIM_SECONDS, UPSCALE_MODE, chunking_decision,
)

LOG_PREFIX = "[VRGDG Chunked Pass 2]"
CHUNK_CLASS = "VRGDG_MiniMaxH3ChunkedPass2"
SUPPORTED_ENCODERS = ("MiniMaxH3ReferenceToVideo", "MiniMaxH3RefModTextEncode")

_CUE_WITH_TEXT = re.compile(r"[ \t]*from \d+(?:\.\d+)? s to \d+(?:\.\d+)? s,?[ \t]*<d>.*?</d>", re.DOTALL)
_DIALOGUE = re.compile(r"[ \t]*<d>.*?</d>", re.DOTALL)
_CUE_TIME = re.compile(r"[ \t]*from \d+(?:\.\d+)? s to \d+(?:\.\d+)? s")
_SHOT_TIME = re.compile(r"^(\[Shot \d+\][ \t]*)At \d\d:\d\d(?:\.\d+)?,[ \t]*(\S)", re.MULTILINE)
_SHOT_PARAGRAPH = re.compile(r"^\[Shot (\d+)\][ \t]*(.*)$", re.DOTALL)
_SECTION_LINE = re.compile(r"^[a-z_]{4,}:[ \t]*$", re.MULTILINE)


class Unsupported(Exception):
    """The built graph does not have the shape this rewrite expects."""


def _merge_repeated_shots(text: str) -> str:
    """Drop a ``[Shot n]`` paragraph that repeats the one before it word for word, and renumber what is left.

    Shots in one unbroken take often differ only by their lyric and start time. Once those are removed they read the
    same, and a window would be told the same shot six times. Only directly repeated shots go; a shot that returns
    after a different one stays, and a prompt with no repeats comes back exactly as it was.
    """
    parts = re.split(r"(\n[ \t]*\n)", text)
    kept: List[str] = []
    previous: Optional[str] = None
    number = dropped = 0
    for position in range(0, len(parts), 2):
        paragraph = parts[position]
        separator = parts[position - 1] if position else ""
        match = _SHOT_PARAGRAPH.match(paragraph)
        if match and not _SECTION_LINE.search(paragraph):
            body = match.group(2).strip()
            if body == previous:
                dropped += 1
                continue
            previous = body
            number += 1
            if dropped:
                paragraph = f"[Shot {number}] {body}"
        else:
            previous = None
        kept.append(separator + paragraph)
    if not dropped:
        return text
    return "".join(kept).rstrip() + text[len(text.rstrip()):]


def strip_prompt_timing(text: str) -> str:
    """The prompt with its lyric cues, dialogue text and shot start times removed.

    ``sings in sync with the vocals in <Audio 1> from 1.0 s to 4.4 s, <d>[tag] words</d>, delivered softly`` becomes
    ``sings in sync with the vocals in <Audio 1>, delivered softly``. A ``[Shot 2] At 00:04.600, continuing ...``
    becomes ``[Shot 2] Continuing ...``, and shots that then repeat the one before them are merged into it.
    Everything else, including the look, the references and the shot descriptions, is kept.
    """
    result = _CUE_WITH_TEXT.sub("", text)
    result = _DIALOGUE.sub("", result)
    result = _CUE_TIME.sub("", result)
    result = _SHOT_TIME.sub(lambda match: match.group(1) + match.group(2).upper(), result)
    result = re.sub(r"[ \t]+,", ",", result)
    result = re.sub(r"(?<=\S)[ \t]{2,}", " ", result)
    return _merge_repeated_shots(result)


def _link(value: Any) -> Optional[Tuple[str, int]]:
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
        return str(value[0]), int(value[1])
    return None


def _source(prompt: Dict[str, Any], ref: Any, class_type: str, what: str) -> str:
    """The id of the node a link points to, checked to be ``class_type``."""
    link = _link(ref)
    node = prompt.get(link[0]) if link else None
    if not node or node.get("class_type") != class_type:
        raise Unsupported(f"{what} is not a {class_type} node.")
    return link[0]


def _new_id(prompt: Dict[str, Any], base: int) -> str:
    number = base
    while str(number) in prompt:
        number += 1
    return str(number)


def _dangling(prompt: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [
        (key, name) for key, node in prompt.items() for name, value in node.get("inputs", {}).items()
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] not in prompt
    ]


def _find_second_pass(prompt: Dict[str, Any]) -> Dict[str, Any]:
    """The nodes of the whole-clip second pass: separate, upscale, replace, optional continuation head, sampler."""
    replaces = [key for key, node in prompt.items() if node.get("class_type") == "VRGDG_MiniMaxH3ReplaceUpscaledVideoLatent"]
    if len(replaces) != 1:
        raise Unsupported("the graph has no single learned-upscale replace node.")
    replace_id = replaces[0]
    found = []
    for key, node in prompt.items():
        if node.get("class_type") != "SamplerCustomAdvanced":
            continue
        link = _link(node["inputs"].get("latent_image"))
        apply_id = None
        if link and prompt.get(link[0], {}).get("class_type") == "VRGDG_MiniMaxH3ApplyMaskedContinuation":
            apply_id = link[0]
            link = _link(prompt[apply_id]["inputs"].get("latent"))
        if link and link[0] == replace_id:
            found.append((key, apply_id))
    if len(found) != 1:
        raise Unsupported("the graph has no single second-pass sampler.")
    sampler_id, apply_id = found[0]
    sampler = prompt[sampler_id]["inputs"]
    replace = prompt[replace_id]["inputs"]
    upscale_id = _source(prompt, replace.get("upscaled_video_latent"), "VRGDG_MiniMaxH3LearnedLatentUpscale", "the upscaled latent")
    separate_id = _source(prompt, prompt[upscale_id]["inputs"].get("latent"), "MiniMaxH3AVLatentSeparateT8", "the upscaler input")
    first_pass = prompt[separate_id]["inputs"].get("av_latent")
    if replace.get("original_av_latent") != first_pass:
        raise Unsupported("the upscale and the replace node read different first-pass latents.")
    guider_id = _source(prompt, sampler.get("guider"), "BasicGuider", "the second-pass guider")
    noise_id = _source(prompt, sampler.get("noise"), "RandomNoise", "the second-pass noise")
    guider = prompt[guider_id]["inputs"]
    encoder_link = _link(guider.get("conditioning"))
    encoder = prompt.get(encoder_link[0]) if encoder_link else None
    if not encoder or encoder.get("class_type") not in SUPPORTED_ENCODERS:
        raise Unsupported(
            "the second pass is conditioned by "
            f"{encoder.get('class_type') if encoder else 'an unknown node'}; only Reference to Video and RefMod are chunked."
        )
    text_id = _source(prompt, encoder["inputs"].get("prompt"), "PrimitiveStringMultiline", "the prompt")
    if not isinstance(prompt[text_id]["inputs"].get("value"), str):
        raise Unsupported("the prompt text is not a plain string.")
    return {
        "sampler": sampler_id, "apply": apply_id, "replace": replace_id, "upscale": upscale_id, "separate": separate_id,
        "guider": guider_id, "noise": noise_id, "encoder": encoder_link[0], "text": text_id,
    }


def _rewire(prompt: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    """Replace the whole-clip second pass in ``prompt`` with the chunk node. Raises ``Unsupported`` on a foreign graph."""
    found = _find_second_pass(prompt)
    sampler = prompt[found["sampler"]]["inputs"]
    upscale = prompt[found["upscale"]]["inputs"]
    guider = prompt[found["guider"]]["inputs"]
    first_pass = prompt[found["separate"]]["inputs"]["av_latent"]
    try:
        seed = int(prompt[found["noise"]]["inputs"]["noise_seed"])
    except (TypeError, ValueError, KeyError) as exc:
        raise Unsupported("the second-pass seed is not a number.") from exc

    text_id = _new_id(prompt, 9500)
    prompt[text_id] = {
        "class_type": "PrimitiveStringMultiline",
        "inputs": {"value": strip_prompt_timing(prompt[found["text"]]["inputs"]["value"])},
        "_meta": {"title": "Pass 2 prompt (no lyric text, no times)"},
    }
    encode_id = _new_id(prompt, 9501)
    prompt[encode_id] = copy.deepcopy(prompt[found["encoder"]])
    prompt[encode_id]["inputs"]["prompt"] = [text_id, 0]
    prompt[encode_id]["_meta"] = {"title": "Pass 2 conditioning (windows)"}

    chunk_id = _new_id(prompt, 9502)
    inputs: Dict[str, Any] = {
        "latent": first_pass,
        "model": guider["model"],
        "conditioning": [encode_id, 0],
        "upscale_model": upscale["upscale_model"],
        "sampler": sampler["sampler"],
        "sigmas": sampler["sigmas"],
        "scale": upscale["scale"],
        "chunk_seconds": CHUNK_SECONDS,
        "overlap_seconds": OVERLAP_SECONDS,
        "overlap_mode": OVERLAP_MODE,
        "seed": seed,
        "tail_trim_seconds": TAIL_TRIM_SECONDS,
        "max_window_tokens": 0,
        "upscale": UPSCALE_MODE,
    }
    head_locked = found["apply"] is not None
    if head_locked:
        apply = prompt[found["apply"]]["inputs"]
        inputs["head_latent"] = apply["context_latent"]
        if apply.get("vae") is not None:
            inputs["head_vae"] = apply["vae"]
    prompt[chunk_id] = {
        "class_type": CHUNK_CLASS, "inputs": inputs,
        "_meta": {"title": f"Pass 2 in windows ({decision['rendered_seconds']:g} s clip)"},
    }

    for node in prompt.values():
        for name, value in node.get("inputs", {}).items():
            link = _link(value)
            if link and link[0] == found["sampler"]:
                node["inputs"][name] = [chunk_id, 0]
    removed = [found[key] for key in ("sampler", "guider", "noise", "separate", "upscale", "replace", "apply") if found[key]]
    for node_id in removed:
        prompt.pop(node_id, None)
    dangling = _dangling(prompt)
    if dangling:
        raise Unsupported(f"another node still reads the second pass it replaces: {dangling[:3]}")
    return {
        "chunk_node_id": chunk_id, "encode_node_id": encode_id, "prompt_node_id": text_id,
        "removed_nodes": removed, "head_locked": head_locked,
    }


def apply_chunked_pass2(prompt: Dict[str, Any], payload: Dict[str, Any], timing: Any) -> Dict[str, Any]:
    """Switch the second pass of a built 2 Pass graph to windows when the clip is long enough.

    Returns a summary with ``enabled``. A clip at or under the threshold, a turned-off setting, or a graph this
    rewrite does not recognise leaves ``prompt`` exactly as it was and says why in ``reason``.
    """
    frames = getattr(timing, "h3_frame_count", None)
    if frames is None and isinstance(timing, dict):
        frames = timing.get("h3_frame_count")
    decision = chunking_decision(payload.get("pass2_chunking"), payload.get("pass2_chunking_min_seconds"), frames)
    if not decision["enabled"]:
        return decision
    work = copy.deepcopy(prompt)
    try:
        decision.update(_rewire(work, decision))
    except Unsupported as exc:
        decision["enabled"] = False
        decision["reason"] = f"Pass 2 runs in one piece because {exc}"
        print(f"{LOG_PREFIX} {decision['reason']}")
        return decision
    prompt.clear()
    prompt.update(work)
    decision.update({
        "chunk_seconds": CHUNK_SECONDS, "overlap_seconds": OVERLAP_SECONDS, "overlap_mode": OVERLAP_MODE,
        "tail_trim_seconds": TAIL_TRIM_SECONDS, "upscale": UPSCALE_MODE,
    })
    print(f"{LOG_PREFIX} {decision['reason']} ({decision.get('windows')} windows"
          f"{', predecessor head locked' if decision['head_locked'] else ''}).")
    return decision
