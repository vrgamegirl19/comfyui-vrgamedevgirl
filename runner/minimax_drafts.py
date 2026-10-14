"""Split MiniMax two-pass graphs at the original pass-1 sampler output."""

import copy
import json
import os
import uuid


def _draft_folder(project_folder, draft_id):
    if not os.path.isdir(project_folder):
        raise ValueError("The draft project folder does not exist.")
    if not isinstance(draft_id, str) or len(draft_id) != 32 or any(c not in "0123456789abcdef" for c in draft_id):
        raise ValueError("Invalid MiniMax draft id.")
    root = os.path.realpath(project_folder)
    folder = os.path.join(root, "minimax_drafts", draft_id)
    if os.path.commonpath([root, os.path.realpath(folder)]) != root:
        raise ValueError("Draft checkpoint must remain inside the project.")
    return folder


def _remove_other_video_outputs(prompt):
    for key in list(prompt):
        if key != "142" and prompt[key]["class_type"] == "VHS_VideoCombine":
            del prompt[key]


def _prune_draft_graph(prompt):
    roots = ["142"] + [key for key, node in prompt.items() if node["class_type"] == "VRGDG_MiniMaxH3SaveLatent"]
    keep = set()
    def visit(key):
        if key in keep:
            return
        keep.add(key)
        for value in prompt[key].get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in prompt and isinstance(value[1], int):
                visit(value[0])
    for key in roots:
        visit(key)
    return {key: node for key, node in prompt.items() if key in keep}


def _build_minimax_draft(result, payload):
    if "125" not in result["prompt"] or not ("two_pass" in result or "advanced_two_pass" in result):
        raise ValueError("Draft rendering requires a MiniMax two-pass workflow.")
    draft_id = uuid.uuid4().hex
    folder = _draft_folder(payload["project_folder"], draft_id)
    os.makedirs(folder)
    saved = copy.deepcopy(result)
    saved["draft_id"] = draft_id
    saved["draft_scene_number"] = int(payload["scene_number"])
    saved["draft_start"] = float(payload["timeline_start_seconds"])
    saved["draft_end"] = float(payload["timeline_end_seconds"])
    saved["draft_restore_audio"] = payload.get("draft_restore_audio") or {}
    saved["draft_steps"] = {"pass1": result["prompt"]["124"]["inputs"]["steps"], "pass2": result["prompt"]["190"]["inputs"]["value"]}
    with open(os.path.join(folder, "workflow.json"), "w", encoding="utf-8") as handle:
        json.dump(saved, handle)
    draft = copy.deepcopy(saved)
    prompt = draft["prompt"]
    _remove_other_video_outputs(prompt)
    prompt["draft_save"] = {"class_type": "VRGDG_MiniMaxH3SaveDraft", "inputs": {
        "latent": ["125", 0], "checkpoint_path": os.path.join(folder, "pass1.pt"),
    }}
    prompt["draft_decode"] = {"class_type": "VAEDecode", "inputs": {
        "samples": ["draft_save", 0], "vae": ["119", 0],
    }}
    prompt["142"]["inputs"]["images"] = ["draft_decode", 0]
    prompt["142"]["inputs"]["filename_prefix"] += "_draft_stage1"
    for node in prompt.values():
        if node["class_type"] == "VRGDG_MiniMaxH3SaveLatent":
            node["inputs"]["latent"] = ["125", 0]
    draft["prompt"] = _prune_draft_graph(prompt)
    return draft


def _build_minimax_finalize(payload):
    folder = _draft_folder(payload["project_folder"], payload["draft_id"])
    checkpoint = os.path.join(folder, "pass1.pt")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError("The saved pass-1 latent is missing. Render a new draft first.")
    with open(os.path.join(folder, "workflow.json"), encoding="utf-8") as handle:
        result = json.load(handle)
    if int(payload["scene_number"]) != result["draft_scene_number"]:
        raise ValueError("This draft belongs to a different scene.")
    if abs(float(payload["timeline_start_seconds"]) - result["draft_start"]) > 0.001 or abs(float(payload["timeline_end_seconds"]) - result["draft_end"]) > 0.001:
        raise ValueError("The scene timing changed. Render a new draft before finalizing.")
    prompt = result["prompt"]
    _remove_other_video_outputs(prompt)
    # Replace the sampler itself, so every pass-2 consumer reuses exactly the saved output.
    prompt["125"] = {"class_type": "VRGDG_MiniMaxH3LoadDraft", "inputs": {"checkpoint_path": checkpoint}}
    prompt["142"]["inputs"]["filename_prefix"] += "_finalized"
    result["prompt"] = _prune_draft_graph(prompt)
    return result
