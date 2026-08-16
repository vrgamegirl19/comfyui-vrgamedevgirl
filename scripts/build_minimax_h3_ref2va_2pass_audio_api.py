"""Build the cleaned MiniMax H3 Ref2VA two-pass external-audio API template.

This intentionally materializes the active graph instead of carrying ComfyUI UI
bypass state into an API prompt.  The source workflow remains untouched.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
SOURCE = Path(
    r"Z:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\MiniMax-H3-Two-Stage-Sampling\workflows"
    + "\\MiniMax H3 - Reference to Video - 2 Stage (EN).json"
)
OUTPUT = REPO_ROOT / "Workflows" / "UsedForUIDoNotTouch" / "minimax_ref2video_2pass_audio_driven_api.json"


def _source_ref(value):
    return isinstance(value, list) and len(value) == 2 and str(value[0])


def _dependencies(prompt, roots):
    needed = set(str(item) for item in roots)
    pending = list(needed)
    while pending:
        node_id = pending.pop()
        node = prompt.get(node_id)
        if not isinstance(node, dict):
            continue
        for value in (node.get("inputs") or {}).values():
            if _source_ref(value) and str(value[0]) not in needed:
                needed.add(str(value[0]))
                pending.append(str(value[0]))
    return needed


def _find_node(prompt, class_type, predicate=None):
    for node_id, node in prompt.items():
        if node.get("class_type") != class_type:
            continue
        if predicate is None or predicate(node_id, node):
            return str(node_id), node
    raise RuntimeError(f"Could not find API node class {class_type}")


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    sys.path.insert(0, str(COMFY_ROOT))
    sys.path.insert(0, str(REPO_ROOT))
    # Load the runner as a synthetic package because the repository directory
    # contains hyphens and cannot be imported as a normal Python package.
    package_name = "vrgdg_builder_tools"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT)]
    sys.modules[package_name] = package
    runner_spec = importlib.util.spec_from_file_location(
        f"{package_name}.VRGDG_WorkflowRunnerNodes",
        REPO_ROOT / "VRGDG_WorkflowRunnerNodes.py",
    )
    runner = importlib.util.module_from_spec(runner_spec)
    sys.modules[runner_spec.name] = runner
    runner_spec.loader.exec_module(runner)

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    bypassed = {
        str(node.get("id"))
        for node in source.get("nodes", [])
        if int(node.get("mode", 0) or 0) == 4
    }
    ui_only = {
        "Fast Groups Bypasser (rgthree)",
        "MarkdownNote",
        "Note",
        "PreviewImage",
        "SaveVideo",
        "CreateVideo",
    }
    nodes = [
        copy.deepcopy(node)
        for node in source.get("nodes", [])
        if str(node.get("id")) not in bypassed and node.get("type") not in ui_only
    ]
    active_ids = {str(node.get("id")) for node in nodes}
    links = [
        link
        for link in source.get("links", [])
        if isinstance(link, list)
        and len(link) >= 6
        and str(link[1]) in active_ids
        and str(link[3]) in active_ids
    ]

    workflow = copy.deepcopy(source)
    workflow["nodes"] = nodes
    workflow["links"] = links

    # The standalone conversion process does not start the ComfyUI server, so
    # some optional/core node packs are not registered yet.  Their workflow
    # input declarations are still present in the source JSON and are enough
    # for API conversion.
    source_inputs = {
        str(node.get("type")): [
            str(item.get("name"))
            for item in (node.get("inputs") or [])
            if item.get("name")
        ]
        for node in nodes
    }
    original_input_names = runner._input_names_for_node

    def input_names_with_source_fallback(class_type, mappings):
        try:
            return original_input_names(class_type, mappings)
        except KeyError:
            return source_inputs.get(class_type, [])

    runner._input_names_for_node = input_names_with_source_fallback
    prompt = runner._workflow_to_api_prompt(workflow)

    # Reroute is a canvas convenience node, not an executable API node.
    # Resolve any retained reroute references to their upstream source.
    changed = True
    while changed:
        changed = False
        for node_id, node in list(prompt.items()):
            if node.get("class_type") != "Reroute":
                continue
            upstream = next(
                (value for value in (node.get("inputs") or {}).values() if _source_ref(value)),
                None,
            )
            if upstream is None:
                prompt.pop(node_id, None)
                changed = True
                continue
            for consumer in prompt.values():
                for key, value in list((consumer.get("inputs") or {}).items()):
                    if _source_ref(value) and str(value[0]) == str(node_id):
                        consumer["inputs"][key] = list(upstream)
            prompt.pop(node_id, None)
            changed = True

    h3_id, h3 = _find_node(prompt, "MiniMaxH3ReferenceToVideo")
    outputs = [
        (node_id, node)
        for node_id, node in prompt.items()
        if node.get("class_type") == "VHS_VideoCombine"
        and "Stage2" in str(node.get("inputs", {}).get("filename_prefix", ""))
    ]
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one Stage 2 VHS output, found {len(outputs)}")
    output_id, output_node = outputs[0]

    # The source workflow's final Stage 2 image path is retained.  Its VAE
    # decode input identifies the final Stage 2 AV latent for AudioDrive.
    final_image_ref = output_node.get("inputs", {}).get("images")
    if not _source_ref(final_image_ref):
        raise RuntimeError("Stage 2 VHS output is missing its image source")
    decode_id = str(final_image_ref[0])
    decode_node = prompt.get(decode_id)
    if not decode_node or decode_node.get("class_type") != "VAEDecode":
        raise RuntimeError("Stage 2 output does not use the expected VAEDecode node")
    stage2_latent = decode_node.get("inputs", {}).get("samples")
    if not _source_ref(stage2_latent):
        raise RuntimeError("Stage 2 VAEDecode is missing its latent source")

    video_vae_id, _ = _find_node(
        prompt,
        "VAELoader",
        lambda _id, node: "video_vae" in json.dumps(node).lower()
        or "video" in str(node.get("inputs", {}).get("vae_name", "")).lower(),
    )
    audio_vae_candidates = [
        (node_id, node)
        for node_id, node in prompt.items()
        if node.get("class_type") == "VAELoader" and str(node_id) != video_vae_id
    ]
    if len(audio_vae_candidates) != 1:
        raise RuntimeError("Could not identify the audio VAE loader")
    audio_vae_id = audio_vae_candidates[0][0]

    # Use stable builder-facing inputs.  The same source audio is both the H3
    # reference-audio conditioning input and the final audio-drive source.
    prompt["9000"] = {
        "class_type": "VRGDG_MiniMaxH3ReferenceMediaFromPaths",
        "inputs": {"image_paths": "[]", "video_references": "[]"},
        "_meta": {"title": "BUILDER INPUTS - Ordered H3 Reference Images And Videos"},
    }
    prompt["9001"] = {
        "class_type": "VHS_LoadAudio",
        "inputs": {"audio_file": "", "seek_seconds": 0, "duration": 0},
        "_meta": {"title": "BUILDER INPUT - Project Audio"},
    }
    prompt["9002"] = {
        "class_type": "VRGDG_MiniMaxH3AudioDrive",
        "inputs": {
            "av_latent": list(stage2_latent),
            "source_audio": ["9001", 0],
            "audio_vae": [audio_vae_id, 0],
        },
        "_meta": {"title": "BUILDER OUTPUT - External Audio Drive"},
    }

    # Replace all manual reference branches with the builder adapter.  The
    # source workflow has fewer visible image slots, but the native node and
    # builder adapter support the full reference slot contract.
    for index in range(9):
        h3["inputs"][f"ref_images.ref_image_{index}"] = ["9000", index]
    for index in range(3):
        h3["inputs"][f"ref_videos.ref_video_{index}"] = ["9000", 9 + index]
        h3["inputs"][f"ref_video_audios.ref_video_audio_{index}"] = ["9000", 12 + index]
    h3["inputs"]["ref_audios.ref_audio_0"] = ["9001", 0]
    h3["inputs"]["ref_image_size"] = "max"
    for index in (1, 2):
        h3["inputs"].pop(f"ref_audios.ref_audio_{index}", None)

    resize_id, resize_node = _find_node(prompt, "ImageResizeKJv2")
    resize_node["inputs"].update({
        "width": ["297", 0],
        "height": ["297", 1],
        "upscale_method": "nvidia_rtx_vsr",
        "keep_proportion": "crop",
        "pad_color": "0, 0, 0",
        "crop_position": "center",
        "divisible_by": 0,
        "device": "cpu",
    })
    resize_node["inputs"].pop("mask", None)

    # External audio replaces the generated audio at the final output while
    # preserving the Stage 2 video decode.
    output_node["inputs"]["audio"] = ["9002", 0]
    output_node["inputs"].update({
        "pix_fmt": "yuv420p",
        "crf": 19,
        "save_metadata": True,
        "trim_to_audio": False,
    })

    # Remove the bypassed TE-Speed optimization from the API graph.  It is an
    # optional accelerator and should not affect the first correctness test.
    prompt = {
        node_id: node
        for node_id, node in prompt.items()
        if node.get("class_type") != "TESpeedMiniMaxH3"
    }
    needed = _dependencies(prompt, [output_id])
    prompt = {node_id: node for node_id, node in prompt.items() if node_id in needed}

    # Use the model filenames already exposed by the Builder's MiniMax panel.
    for node_id, node in prompt.items():
        if node.get("class_type") == "UNETLoader":
            node["inputs"]["unet_name"] = "minimax_h3_ref2va_int8_convrot.safetensors"
        elif node.get("class_type") == "CLIPLoader":
            node["inputs"]["clip_name"] = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        elif node.get("class_type") == "VAELoader":
            if node_id == video_vae_id:
                node["inputs"]["vae_name"] = "minimax_h3_video_vae_fp16.safetensors"
            elif node_id == audio_vae_id:
                node["inputs"]["vae_name"] = "minimax_h3_audio_vae_fp32.safetensors"

    OUTPUT.write_text(json.dumps(prompt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Nodes: {len(prompt)}; output node: {output_id}; stage2 latent: {stage2_latent}")


if __name__ == "__main__":
    main()
