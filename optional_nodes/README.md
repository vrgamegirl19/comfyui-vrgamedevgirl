# Optional node packs

These nodes are not used by the Music Video Builder and are not loaded by the main pack. Each folder is a standalone ComfyUI node pack with its own `__init__.py`, `requirements.txt`, `web/` and workflows. Node names are unchanged, so existing workflows keep working once the pack is installed.

## Install

Copy (or symlink) the pack folder into `ComfyUI/custom_nodes/`, install its requirements, and restart ComfyUI:

```
cp -r optional_nodes/face_fix ComfyUI/custom_nodes/vrgdg_face_fix
python -m pip install -r ComfyUI/custom_nodes/vrgdg_face_fix/requirements.txt
```

Packs can be installed alongside the main pack and each other.

| Pack | Nodes | Contents |
| --- | --- | --- |
| `face_fix` | 20 | Face repair, LTX face-crop round trips, video enhancement, image paste-back |
| `general_utilities` | 70 | Text/JSON/prompt tools, switches, group/mute toggles, grain, sharpening, color match, LUTs, image/video compare, audio split loaders |
| `llm_nodes` | 6 | Qwen 3.5/2.5, General VLM, General GGUF, Local LLM, llama.cpp doctor |
| `long_video` | 12 | Overlap meta-batch, long-video meta-batch loader, LongShot nodes (needs VideoHelperSuite) |
| `lora_training` | 14 | LTX 2.3, Krea 2 and Z-Image LoRA trainers, installers, XYZ plots, LoRA Dataset Creator |
| `ltx_minimax_tools` | 16 | LTX guiders, first/last guides, IC grid, looping sampler, sigma presets, MiniMax chunks/still/trim, upscaler panel |
| `music_audio` | 19 | YuE2, SheetSage2, MiniMax Music 3 helpers, VoxCPM2, audio load/save |
| `ui_tools` | 10 | Video Editor, Prompt Creator V1/V2, Start Image Storyboard, Node Canvas |

## Workflows that need more than one pack

- LTX 2.3 V5.x workflows (in the main `Workflows/`): `general_utilities`, `ui_tools`, `llm_nodes`
- MiniMax upscaler workflows: `long_video`, `general_utilities`
- Z-Image Upscale AnyImage / Wan 2.2 workflows: `general_utilities`
- YuE2 Cover/Starter workflows: main pack
- LoRA Dataset Creator workflows: `general_utilities` and the main pack
