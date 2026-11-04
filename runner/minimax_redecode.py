"""Make a MiniMax H3 scene video again from its saved latent, at a chosen quality.

A render saves the sampler's whole output (picture and sound) to ``latents/scene_NNN.latent`` right before the VAE
decode. Decoding it again gives the same frames the render encoded, and the encoder can then be given a finer CRF than
the render used. The graph the Builder compiles for the scene is cut down to what that needs: load the latent, decode,
resize, encode. Nothing is sampled again.
"""

import copy
from typing import Any, Dict

from ..minimax.latent_manager import SceneLatentManager
from .api_graph import _find_final_vae_decode_id

REDECODE_LOADER_CLASS = "VRGDG_MiniMaxH3LoadLatentFull"
DEFAULT_REDECODE_CRF = 12
MAX_REDECODE_CRF = 30
_AUDIO_DRIVE_CLASS = "VRGDG_MiniMaxH3AudioDrive"
# The sockets a decode node reads the sampled latent from.
_LATENT_INPUTS = ("samples", "av_latent", "latent")


def redecode_crf(value: Any) -> int:
    """Return the CRF for a re-decode: 12 when not given, else a whole number from 0 to 30.

    Raises:
        ValueError: when the value is not a number in range.
    """
    if value in (None, ""):
        return DEFAULT_REDECODE_CRF
    try:
        crf = int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"crf must be a whole number from 0 to {MAX_REDECODE_CRF}, got {value!r}.") from None
    if not 0 <= crf <= MAX_REDECODE_CRF:
        raise ValueError(f"crf must be from 0 to {MAX_REDECODE_CRF}, got {crf}.")
    return crf


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int))


def _reachable(prompt: Dict[str, Any], root: str) -> set:
    """The ids of ``root`` and every node it reads from, directly or through other nodes."""
    keep: set = set()
    todo = [str(root)]
    while todo:
        node_id = todo.pop()
        if node_id in keep or node_id not in prompt:
            continue
        keep.add(node_id)
        for value in prompt[node_id].get("inputs", {}).values():
            if _is_link(value):
                todo.append(str(value[0]))
    return keep


def _final_video_node_id(prompt: Dict[str, Any], graph_res: Dict[str, Any]) -> str:
    """The node that writes the scene video: the one the compiler named, else the first video combine."""
    named = str(graph_res.get("final_video_node_id") or "")
    if named in prompt:
        return named
    for candidate in ("353", "142"):
        if prompt.get(candidate, {}).get("class_type") == "VHS_VideoCombine":
            return candidate
    for node_id, node in prompt.items():
        if node.get("class_type") == "VHS_VideoCombine":
            return str(node_id)
    raise ValueError("The scene graph has no video output node to re-decode into.")


def build_redecode_graph(
    graph_res: Dict[str, Any],
    project_folder: str,
    scene_number: int,
    crf: int,
    loader_class: str = REDECODE_LOADER_CLASS,
) -> Dict[str, Any]:
    """Cut a compiled scene graph down to "load the saved latent, decode, resize, encode at ``crf``".

    ``graph_res`` is what the MiniMax graph builders return. The result has the same shape with the smaller prompt,
    so the caller's trim and collect steps work as they do after a render.

    Raises:
        ValueError: when the scene cannot be re-decoded: no saved latent, a latent that does not match the scene's
            current settings, built-in audio, or a graph without a decode or a video output.
    """
    if str(graph_res.get("audio_mode") or "") == "built_in_audio":
        raise ValueError(
            "This scene uses built-in audio, which is decoded from the first-pass latent. Re-decoding supports scenes "
            "that play the song or another audio file."
        )
    prompt = copy.deepcopy(graph_res.get("prompt") or {})
    video_node = _final_video_node_id(prompt, graph_res)
    decode_id = _find_final_vae_decode_id(prompt)
    decode = prompt.get(decode_id)
    latent_input = next((name for name in _LATENT_INPUTS if decode and _is_link(decode["inputs"].get(name))), None)
    if not decode or latent_input is None:
        raise ValueError("The scene graph has no VAE decode to feed from a saved latent.")

    info = SceneLatentManager.get_latent_info(project_folder, scene_number)
    if not info.get("exists"):
        raise ValueError(f"Scene {scene_number} has no saved latent. Render the scene first; MiniMax H3 renders save one.")
    expected = int((graph_res.get("timing") or {}).get("h3_frame_count") or 0)
    saved = int(info.get("frame_count") or 0)
    if expected and saved and expected != saved:
        raise ValueError(
            f"Scene {scene_number}'s saved latent has {saved} frames, but its current settings and timing call for "
            f"{expected}. The scene was changed after it was rendered, so the latent cannot be trimmed to the scene "
            "exactly. Render the scene again."
        )

    loader_id = str(max([int(key) for key in prompt if str(key).isdigit()] + [9000]) + 1)
    prompt[loader_id] = {
        "class_type": loader_class,
        "inputs": {"project_folder": project_folder, "scene_number": int(scene_number)},
        "_meta": {"title": f"Saved Latent (Scene {int(scene_number):03d})"},
    }
    decode["inputs"][latent_input] = [loader_id, 0]

    # The video is muxed with the source audio file, not with anything sampled. The audio drive only passes that file
    # through (and needs the text encoder and reference nodes to build its latent), so read the file directly.
    video_inputs = prompt[video_node]["inputs"]
    audio = video_inputs.get("audio")
    if _is_link(audio) and prompt.get(str(audio[0]), {}).get("class_type") == _AUDIO_DRIVE_CLASS and int(audio[1]) == 1:
        source = prompt[str(audio[0])]["inputs"].get("source_audio")
        if _is_link(source):
            video_inputs["audio"] = list(source)

    keep = _reachable(prompt, video_node)
    prompt = {node_id: node for node_id, node in prompt.items() if str(node_id) in keep}
    prompt[video_node]["inputs"]["crf"] = int(crf)
    return {
        **graph_res,
        "prompt": prompt,
        "final_video_node_id": video_node,
        "redecode": {"crf": int(crf), "frame_count": saved, "latent_path": info.get("path", ""), "node_count": len(prompt)},
    }
