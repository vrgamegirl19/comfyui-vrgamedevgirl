"""RefMod pipeline for the MiniMax H3 graph builders.

When a render payload says ``pipeline == "refmod"`` the standard graph is rewired after it is built: the reference
media node and the reference inputs of the Reference to Video node are removed, and the guiders read their conditioning
from *Text Encode with RefMods* instead. That node shows every saved RefMod to the text encoder with a numbered label
(``<Video 1>``, ``<Picture 1>``, ``<Audio 1>``), which is what lets the prompt tell several characters apart.

The scene audio (input audio mode) is turned into an audio RefMod on the fly so the ``<Audio 1>`` label and the
audio reference block still exist. Built-in audio graphs have no scene audio, so only the visual mods are used.

Every standard patch (LoRAs, Turbo, TE-Speed, feed-forward, block-sparse attention, fast decode, latent continuation)
has already been applied to the graph when this runs, so they work unchanged.
"""

from typing import Any, Dict, List

from ..minimax.refmod_library import find_refmod
from ..minimax.refmod_scene import assign_labels, token_report

LOADER_SLOTS = 8
MAX_LOADERS = 3
MAX_REFERENCES = LOADER_SLOTS * MAX_LOADERS
_REFERENCE_INPUT_PREFIXES = ("ref_images.", "ref_videos.", "ref_video_audios.", "ref_audios.")


def refmod_requested(payload: Dict[str, Any]) -> bool:
    return str(payload.get("pipeline") or "").strip().lower() == "refmod"


def _new_id(prompt: Dict[str, Any], base: int) -> str:
    number = base
    while str(number) in prompt:
        number += 1
    return str(number)


def validate_references(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check the ordered ``refmod_references`` of a payload against the RefMods on disk."""
    references = payload.get("refmod_references")
    if not isinstance(references, list) or not references:
        raise ValueError("The RefMod pipeline needs at least one RefMod reference for this scene.")
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"A scene can use at most {MAX_REFERENCES} RefMods ({len(references)} were given).")
    checked: List[Dict[str, Any]] = []
    for position, reference in enumerate(references, start=1):
        if not isinstance(reference, dict):
            raise ValueError(f"RefMod reference {position} is not an object.")
        name = str(reference.get("name") or "").strip()
        entry = find_refmod(name)
        if entry is None:
            raise FileNotFoundError(
                f"RefMod '{name}' was not found in models/refmods. Pick it again in the Reference Builder or remove the card.")
        try:
            strength = float(reference.get("strength", 1.0))
        except (TypeError, ValueError):
            raise ValueError(f"RefMod '{name}' has a strength that is not a number.")
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"RefMod '{name}' strength must be between 0 and 1.")
        checked.append({
            "name": name, "strength": strength, "kind": entry["kind"], "tokens": entry["tokens"],
            "display_name": str(reference.get("display_name") or name), "category": str(reference.get("category") or ""),
            "prompt_label": str(reference.get("label") or "").strip(),
        })
    return checked


def _loader_node(slots: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {"show_info": False}
    for index in range(1, LOADER_SLOTS + 1):
        slot = slots[index - 1] if index <= len(slots) else None
        inputs[f"mod_{index}"] = slot["name"] if slot else "(none)"
        inputs[f"strength_{index}"] = slot["strength"] if slot else 1.0
        inputs[f"copies_{index}"] = 1
    return {"class_type": "MiniMaxH3RefModsLoader", "inputs": inputs, "_meta": {"title": title}}


def apply_refmod_pipeline(prompt: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rewire a built MiniMax H3 graph to the RefMod pipeline. Changes ``prompt`` in place and returns a summary."""
    references = validate_references(payload)
    reference_nodes = [key for key, node in prompt.items() if node.get("class_type") == "MiniMaxH3ReferenceToVideo"]
    if len(reference_nodes) != 1:
        raise ValueError("The MiniMax H3 graph changed: the RefMod pipeline needs one Reference to Video node.")
    r2v = reference_nodes[0]
    media_nodes = [key for key, node in prompt.items() if node.get("class_type") == "VRGDG_MiniMaxH3ReferenceMediaFromPaths"]
    prompt_node = prompt[r2v]["inputs"].get("prompt")
    if not (isinstance(prompt_node, list) and len(prompt_node) == 2):
        raise ValueError("The MiniMax H3 graph changed: the Reference to Video prompt is not linked to the prompt node.")

    # No images, videos or reference audio go into Reference to Video any more. It only makes the empty AV latent.
    for name in [name for name in prompt[r2v]["inputs"] if name.startswith(_REFERENCE_INPUT_PREFIXES)]:
        del prompt[r2v]["inputs"][name]
    for key in media_nodes:
        prompt.pop(key, None)
    prompt[r2v]["inputs"]["prompt"] = " "

    loaders: List[str] = []
    for start in range(0, len(references), LOADER_SLOTS):
        loader_id = _new_id(prompt, 9301 + len(loaders))
        prompt[loader_id] = _loader_node(references[start:start + LOADER_SLOTS], f"RefMods for this scene ({len(loaders) + 1})")
        loaders.append(loader_id)

    combine_inputs: Dict[str, Any] = {f"mods_{i + 1}": [loader_id, 0] for i, loader_id in enumerate(loaders)}
    audio_node = next((key for key, node in prompt.items() if node.get("class_type") == "VHS_LoadAudio"), "")
    audio_vae = prompt[r2v]["inputs"].get("audio_vae")
    with_audio = bool(audio_node) and str(payload.get("audio_mode") or "input_audio") != "built_in_audio" and audio_vae
    if with_audio:
        extract_id = _new_id(prompt, 9310)
        prompt[extract_id] = {
            "class_type": "MiniMaxH3RefModAudioExtract",
            "inputs": {
                "audio": [audio_node, 0], "audio_vae": audio_vae, "name": "scene_audio", "max_seconds": 60.0,
                "max_tokens": 0, "budget_policy": "error", "concept_type": "singing", "description": "",
                "subfolder": "", "save": False,
            },
            "_meta": {"title": "Scene audio as an audio RefMod"},
        }
        combine_inputs[f"mods_{len(loaders) + 1}"] = [extract_id, 0]
    combine_id = _new_id(prompt, 9320)
    prompt[combine_id] = {
        "class_type": "VRGDG_RefModCombine", "inputs": combine_inputs, "_meta": {"title": "RefMods and scene audio"},
    }

    clip = next((key for key, node in prompt.items() if node.get("class_type") == "CLIPLoader"), "")
    video_vae = prompt[r2v]["inputs"].get("vae")
    if not clip or not video_vae:
        raise ValueError("The MiniMax H3 graph changed: the RefMod pipeline needs a text encoder and the video VAE.")
    encode_id = _new_id(prompt, 9330)
    prompt[encode_id] = {
        "class_type": "MiniMaxH3RefModTextEncode",
        "inputs": {
            "clip": [clip, 0], "mods": [combine_id, 0], "prompt": prompt_node, "reference_fps": 24.0,
            "max_total_tokens": 0, "vae": video_vae,
        },
        "_meta": {"title": "Text Encode with RefMods (labels the references)"},
    }
    rewired = 0
    for node in prompt.values():
        if node.get("inputs", {}).get("conditioning") == [r2v, 0]:
            node["inputs"]["conditioning"] = [encode_id, 0]
            rewired += 1
    if not rewired:
        raise ValueError("The MiniMax H3 graph changed: no guider reads the Reference to Video conditioning.")

    dangling = [
        (key, name) for key, node in prompt.items() for name, value in node.get("inputs", {}).items()
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] not in prompt
    ]
    if dangling:
        raise ValueError(f"The RefMod rewire left dangling links: {dangling}")

    labelled = assign_labels(
        [{"strength": item["strength"], "kind": item["kind"], "name": item["display_name"], "mod_name": item["name"],
          "tokens": item["tokens"], "key": "", "card_id": "", "category": item["category"], "reference_type": "", "description": "",
          "wears": ""} for item in references],
        include_audio=bool(with_audio),
    )
    labels = [entry["label"] for entry in labelled]
    report = token_report(labelled)
    if report["imbalance"]:
        print(f"[VRGDG RefMod] {report['imbalance']['weak']} is {report['imbalance']['ratio']}x lighter than "
              f"{report['imbalance']['strong']} and may be duplicated.")
    if report["over_limit"]:
        print(f"[VRGDG RefMod] Scene uses {report['total']} RefMod tokens (above {report['limit']}).")
    text = str(payload.get("prompt") or "")
    relabelled = _relabel_prompt(prompt, prompt_node, references, labels)
    if relabelled:
        text = prompt[prompt_node[0]]["inputs"]["value"]
        print(f"[VRGDG RefMod] Prompt labels updated to match the saved RefMods: {relabelled}")
    return {
        "references": [{"name": item["name"], "strength": item["strength"], "kind": item["kind"], "tokens": item["tokens"]} for item in references],
        "labels": labels,
        # <Audio 1> is only named in multi-performer prompts, so it is not reported.
        "labels_missing_from_prompt": [label for label in labels if label and not label.startswith("<Audio") and label not in text],
        "tokens": sum(item["tokens"] for item in references if item["strength"] > 0),
        "token_report": report,
        "scene_audio_mod": bool(with_audio),
        "guiders_rewired": rewired,
        "labels_relabelled": relabelled,
    }


def _relabel_prompt(prompt: Dict[str, Any], prompt_node: List[Any], references: List[Dict[str, Any]], labels: List[str]) -> Dict[str, str]:
    """Swap labels the prompt was written with for the labels the render gives, when they differ.

    The prompt is written with the kind a card saved when its RefMod was picked. If the file was saved again since
    (one image became several), ``<Picture 1>`` is now ``<Video 1>`` and the prompt would name the wrong reference.
    Returns ``{old: new}`` for every label it changed.
    """
    mapping = {
        item["prompt_label"]: label for item, label in zip(references, labels)
        if item.get("prompt_label") and label and item["prompt_label"] != label
    }
    node = prompt.get(str(prompt_node[0])) or {}
    value = node.get("inputs", {}).get("value")
    if not mapping or not isinstance(value, str):
        return {}
    # Two steps, so swapping <Picture 1> and <Video 1> does not turn both into the same label.
    for index, old in enumerate(mapping):
        value = value.replace(old, f"\x00refmod{index}\x00")
    for index, new in enumerate(mapping.values()):
        value = value.replace(f"\x00refmod{index}\x00", new)
    node["inputs"]["value"] = value
    return mapping
