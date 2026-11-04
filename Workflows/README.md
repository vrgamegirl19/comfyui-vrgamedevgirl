# Workflow library

Download a workflow JSON and drag it onto the ComfyUI canvas. Select your installed models and install any additional nodes required by the workflow before running it.

| Folder | Contents |
| --- | --- |
| [Example](Example/) | Example workflows and VRGDG settings demonstrations |
| [MiniMax Workflows](MiniMax%20Workflows/) | MiniMax H3 and FastH3 workflows, API graphs, two-pass refinement, latent upscaling, and the two-stage sampling archive |
| [Music](Music/) | MiniMax Music 3 tuning workflow |
| [RefMods](RefMods/) | RefMod information and training workflows |
| [LTX-2 Workflows](LTX-2_Workflows/) | LTX workflows and the Video Builder guide |
| [Face Fix](FaceFix/) | Face-repair workflow |
| [VoxCPM2](VoxCPM2/) | Speech and voice-cloning workflow |
| [YuE2](Yue2/) | Music generation, setup, and cover workflows |
| [Z-Image Upscale](Z-ImageUpscale/) | Z-Image upscaling workflows |

The [FastH3 Omni Two-Model Refinement Test](MiniMax%20Workflows/FastH3_Omni_Two_Model_Refinement_Test.json) is included in the MiniMax folder. Other existing workflow collections remain in their named subfolders; `UsedForUIDoNotTouch/` contains templates used internally by the Builder.

## Standalone Nodes and Older Examples

This library contains Builder templates and canvas examples collected over time. A workflow JSON can reference nodes supplied by another repository.

Standalone face-repair, utility, LLM, long-video, LoRA-training, LTX/MiniMax, music/audio, and workflow-UI nodes are maintained in [VRGDG Custom Node Pack](https://github.com/vrgamegirl19/VRGDG_CustomNodePack). Use that repository for the current standalone-node documentation and examples. Install it separately when a workflow needs its nodes; installing the AI Video Builder alone does not provide the moved tools.

For Builder setup, use the [AI Video Builder Guide](LTX-2_Workflows/Video_Builder/readme.md). Keep `UsedForUIDoNotTouch/` intact: the Builder uses those templates internally.

## Videos and community

---

[discord server here](https://discord.gg/FJ9VvCDXw3)

[New walkthough video for Nano Banana / LTX-2 workflow](https://youtu.be/06bPil9vrPk)

[Youtube playlist of sample vids created using these new LTX-2 workflows](https://www.youtube.com/playlist?list=PLQ0zxAQhttlbkHBNHUgvzIOL610JW-_hZ)

---
