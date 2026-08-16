import json
from pathlib import Path


SOURCE = Path(r"Z:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\VRGDG_FaceFix_LTX25_FullCrop_OpaqueComposite_ShotAware.json")
TARGET = Path(r"Z:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\VRGDG_FaceFix_LTX25_FullCrop_OpaqueComposite_NormalSampler.json")


def main():
    workflow = json.loads(SOURCE.read_text(encoding="utf-8"))
    nodes = workflow["nodes"]
    prepare = next(n for n in nodes if str(n.get("id")) == "5370")
    prepare["type"] = "VRGDGFaceFixPrepareShotAware"
    prepare["widgets_values"] = [0.4, 0.15, 12, "Off (fastest)", "Far faces (recommended)", 9, "16 frames (recommended)", 2, 0.28]
    prepare["properties"] = {"cnr_id": "comfyui-vrgamedevgirl", "Node name for S&R": "VRGDGFaceFixPrepareShotAware"}
    prepare["title"] = "Face Fix - Prepare Full Video With Shot Tracking"

    ltx_group = next(n for n in nodes if str(n.get("id")) == "5343")
    ltx_group["title"] = "LTX Normal Sampler - Full Crop V2V"
    ltx_group["properties"]["proxyWidgets"] = [["1241", "frame_rate"], ["2483", "text"]]

    definitions = workflow.setdefault("definitions", {})
    subgraphs = definitions.get("subgraphs") or []
    ltx_subgraph = next(s for s in subgraphs if s.get("id") == "f9abaaee-15e1-4c0b-9fee-546d258c9553")
    sampler = next(n for n in ltx_subgraph["nodes"] if n.get("id") == 4880)
    sampler["type"] = "SamplerCustomAdvanced"
    sampler["title"] = "LTX Normal Sampler"
    sampler["properties"] = {"cnr_id": "comfy-core", "Node name for S&R": "SamplerCustomAdvanced"}
    sampler["inputs"] = [
        {"localized_name": "noise", "name": "noise", "type": "NOISE", "link": 32},
        {"localized_name": "guider", "name": "guider", "type": "GUIDER", "link": 35},
        {"localized_name": "sampler", "name": "sampler", "type": "SAMPLER", "link": 33},
        {"localized_name": "sigmas", "name": "sigmas", "type": "SIGMAS", "link": 34},
        {"localized_name": "latent_image", "name": "latent_image", "type": "LATENT", "link": 36},
    ]
    sampler["outputs"] = [
        {"localized_name": "output", "name": "output", "type": "LATENT", "links": []},
        {"localized_name": "denoised_output", "name": "denoised_output", "type": "LATENT", "links": [40]},
    ]
    sampler["widgets_values"] = []

    ltx_subgraph["links"] = [link for link in ltx_subgraph["links"] if link["target_id"] != 4880 and link["origin_id"] != 4880]
    ltx_subgraph["links"].extend([
        {"id": 32, "origin_id": 4638, "origin_slot": 0, "target_id": 4880, "target_slot": 0, "type": "NOISE"},
        {"id": 35, "origin_id": 4878, "origin_slot": 0, "target_id": 4880, "target_slot": 1, "type": "GUIDER"},
        {"id": 33, "origin_id": 4637, "origin_slot": 0, "target_id": 4880, "target_slot": 2, "type": "SAMPLER"},
        {"id": 34, "origin_id": 4896, "origin_slot": 0, "target_id": 4880, "target_slot": 3, "type": "SIGMAS"},
        {"id": 36, "origin_id": 5290, "origin_slot": 0, "target_id": 4880, "target_slot": 4, "type": "LATENT"},
        {"id": 40, "origin_id": 4880, "origin_slot": 1, "target_id": 5289, "target_slot": 0, "type": "LATENT"},
    ])
    # Keep the source-preserving sigma schedule used for the successful test.
    sigmas = next(n for n in ltx_subgraph["nodes"] if n.get("id") == 4896)
    sigmas["widgets_values"] = ["0.65, 0.50, 0.30, 0.0"]

    workflow["extra"]["description"] = (
        "Shot-aware LTX-2.5 full-crop face repair using the normal SamplerCustomAdvanced "
        "instead of temporal looping. Opaque face composite; original audio preserved."
    )
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
