# MiniMax H3 — Two-Stage Sampling Workflows

Resource pack for the video. Two ComfyUI workflows, both English, both using
two-stage sampling: generate fast at low resolution, then resample at high
resolution instead of paying full price for every step.

**Working title keywords:** best MiniMax workflow · two-stage sampling ·
efficient · better character consistency

---

## The two workflows

### 1. Image to Video — `MiniMax H3 - Image to Video - 2 Stage (EN).json`

77 nodes. Model: **fl2va**.

| Group | What it does |
|---|---|
| Model Loading | UNet + turbo LoRA + Sage attention patch, CLIP, video VAE, audio VAE |
| First Frame / Last Frame | Two image loaders, each independently switchable |
| Stage 1 — FL2VA Generation | Generates at low res (default ~0.5 MP) |
| Stage 2 — FL2VA Resample + RTX Upscale | Re-samples at high res (default ~1.4 MP) |

**Three switches**, all rgthree group bypassers:

| First Frame | Last Frame | Mode |
|---|---|---|
| ON | ON | First + last frame |
| ON | OFF | Image to video |
| OFF | OFF | **Text to video** |

Both keyframe inputs are optional, so turning both off makes the model
generate from the prompt alone (t2va) — one workflow covers all three modes.

Plus **Stage 2 ON / OFF** — off gives you the Stage 1 output only.

### 2. Reference to Video — `MiniMax H3 - Reference to Video - 2 Stage (EN).json`

109 nodes. Model: **ref2va**.

Up to 6 reference images, 3 reference videos and 3 reference audio clips,
each in its own bypassable group. This is the one for character consistency —
references carry identity across shots.

---

## Models required

All go under `ComfyUI/models/`. **Total download is roughly 75 GB** — start it before
you go to bed.

| File | Folder | Size | Download |
|---|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models` | 20.9 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) |
| `minimax_h3_ref2va_int8_convrot.safetensors` | `diffusion_models` | 34.0 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors) |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders` | 15.7 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| `minimax_h3_video_vae_int8_convrot.safetensors` | `vae` | 3.2 GB | [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/minimax_h3_video_vae_int8_convrot.safetensors) |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae` | 0.6 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) |
| `minimax_h3_turbo_v4_step600_ema.safetensors` | `loras` | 0.8 GB | [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/blob/main/minimax_h3_turbo_v4_step600_ema.safetensors) |

**Note the three different repos.** Everything comes from `Comfy-Org/MiniMax-H3`
*except* the video VAE (Kijai's experimental repo) and the turbo LoRA (larryvrh).
That's the part people miss — they download the Comfy-Org repo, don't find those two,
and assume the workflow is broken.

### Copy-paste download

```bash
pip install -U "huggingface_hub[cli]"
cd ComfyUI/models

hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir .

hf download Kijai/MiniMax-H3-experimental \
  minimax_h3_video_vae_int8_convrot.safetensors --local-dir vae

hf download larryvrh/MiniMax-H3-Turbo-Lora \
  minimax_h3_turbo_v4_step600_ema.safetensors --local-dir loras
```

The Comfy-Org command already writes into the right subfolders. The other two need
`--local-dir` pointed at the destination because those repos are flat.

### Short on disk or VRAM?

`minimax_h3_ref2va_int8_convrot` is the 34 GB full version. Comfy-Org also ships
`minimax_h3_ref2va_pruned_int8_convrot.safetensors` at **20.9 GB** — same folder, drop-in
replacement, just point the UNETLoader at it instead. Quality is slightly lower; if
you're tight on space it's the first cut to make.

### Get the model right

`MiniMaxH3ImageToVideo` is the **t2va / fl2va** node — its own docstring in
`comfy_extras/nodes_minimax_h3.py` says so. It needs **fl2va** weights.
`MiniMaxH3ReferenceToVideo` is the **ref2va** node and needs ref2va weights.
Feeding the wrong one is a silent mismatch that only shows up later.

---

## Custom nodes required

All of these install into `ComfyUI/custom_nodes/`. On the Windows portable build
that's `ComfyUI_windows_portable\ComfyUI\custom_nodes`.

### ⚠️ Read this before you try ComfyUI Manager

**"Install Missing Custom Nodes" will not fully solve these workflows.** Some nodes
were saved with missing or incorrect source metadata, so Manager either can't
identify them or points at the wrong repo (several report `comfyui-workflow-encrypt`,
which does not contain them — ignore that suggestion). Install the four marked
**manual** below by cloning them yourself.

### Required — both workflows

| Pack | Provides | Install |
|---|---|---|
| rgthree-comfy | Fast Groups Bypasser (all ON/OFF switches) | Manager, or `git clone https://github.com/rgthree/rgthree-comfy` |
| ComfyUI-KJNodes | SetNode / GetNode, ImageResizeKJv2, Sage attention patch | Manager, or `git clone https://github.com/kijai/ComfyUI-KJNodes` |
| ComfyUI_LayerStyle | `LayerUtility: ImageScaleByAspectRatio V2` | Manager, or `git clone https://github.com/chflame163/ComfyUI_LayerStyle` |
| ComfyUI-VideoHelperSuite | VHS_VideoCombine, VHS_LoadVideo | Manager, or `git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite` |
| ComfyUI_Text_Translation | `Text` (the prompt box) | **manual** — `git clone https://github.com/TFL-TFL/ComfyUI_Text_Translation` |
| comfyui-minimax-h3-audio-T8 | MiniMaxH3AVDecodeT8 | **manual** — `git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8` |
| ComfyUI-PT_H3ConcatAVLatent | PT_H3ConcatAVLatent | **manual** — `git clone https://github.com/ptmaster/ComfyUI-PT_H3ConcatAVLatent` |

### Optional — bypassed by default

These ship switched **off**. The workflow runs without them; you'll just see a red
node until you install them. Only install if you want to turn that group on.

| Pack | Provides | Used by | Install |
|---|---|---|---|
| TE-Speed-MiniMaxH3-OSS | TESpeedMiniMaxH3 | Reference to Video | **manual** — `git clone https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS` |
| comfyui_nvidia_rtx_nodes | RTXVideoSuperResolution | Image to Video | Manager, or `git clone https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI` |
| Comfyui-Memory_Cleanup | RAMCleanup / VRAMCleanup | Image to Video | Manager, or `git clone https://github.com/LAOGOU-666/Comfyui-Memory_Cleanup` |

### Copy-paste install (the four Manager can't find)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TFL-TFL/ComfyUI_Text_Translation
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8
git clone https://github.com/ptmaster/ComfyUI-PT_H3ConcatAVLatent
git clone https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS
```

Restart ComfyUI fully afterwards — a browser refresh is not enough, the node
registry only rebuilds on server start.

### If `ImageScaleByAspectRatio V2` is still red

Two different packs publish a node by that name — ComfyUI_LayerStyle and
`aining2022/ComfyUI_Swwan`. LayerStyle is the one these workflows were built
against. If you already have Swwan installed, the two can shadow each other;
install LayerStyle and restart, and if the node still won't resolve, remove Swwan.

`ResolutionSelector`, `ComfyMathExpression` and `PrimitiveFloat` are core
ComfyUI (tested on 0.31.1) — nothing to install.

---

## The one thing that trips people up

Stage 1 and Stage 2 run at **different resolutions**, and the two model types
handle that differently.

- **ref2va** stores each reference with its own `latent_h` / `latent_w`, so a
  reference is resolution-independent. Stage 2 can reuse Stage 1's
  conditioning directly.
- **fl2va** pins keyframe rows to the *target* grid. In
  `comfy/ldm/minimax/model.py`: *"fl2va: keyframe cond rows right after text,
  sharing the target spatial grid."*

So in the Image to Video workflow, Stage 2 **cannot** reuse Stage 1's
conditioning — it has its own `MiniMaxH3ImageToVideo` node that re-encodes the
keyframes at the Stage 2 resolution. Sharing it instead throws:

```
RuntimeError: shape mismatch: value tensor of shape [510, 96]
cannot be broadcast to indexing result of shape [1400, 96]
```

510 = 30x17 patch grid at 960x544 (Stage 1). 1400 = 50x28 at 1600x896
(Stage 2). Both stages must also use the **same frame count**.

---

## Settings worth mentioning on camera

- Stage 1 resolution ~0.4–0.5 MP, Stage 2 ~1.4 MP. Lower Stage 1 = faster;
  Stage 2 does the quality work at 4 steps with denoise 0.2.
- Turbo LoRA strength: 0.88 in Image to Video, 0.7 in Reference to Video.
  First dial to touch if motion drifts.
- RTX Video Super Resolution is set to 2x, ULTRA.

## Known UI quirk

These workflows were authored on frontend 1.23.0. On a current frontend
(1.48.x) node boxes may resize oddly on first load. Widen and re-save once and
it settles. Do **not** run "Resize Selected Nodes" on the rgthree Fast Groups
Bypasser nodes — their toggle labels are custom-drawn and not counted by
LiteGraph's `computeSize()`, so it collapses them and the rows spill outside
the frame.
