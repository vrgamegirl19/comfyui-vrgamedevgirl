# 🎬 VRGameDevGirl’s AI Video Builder for ComfyUI

**Create AI music videos, short films, and scene-based video projects inside ComfyUI.** This repository contains the AI Video Builder, its supporting nodes, and the workflow templates it uses.

The flagship tool is the **AI Video Builder**: a scene-by-scene production workspace for LTX 2.3, MiniMax H3, and future video engines that brings planning, prompting, media generation, timing, review, and final assembly together inside ComfyUI.

Standalone face-repair, utility, LLM, long-video, LoRA-training, LTX/MiniMax, music/audio, and workflow-UI nodes are now maintained in [VRGDG Custom Node Pack](https://github.com/vrgamegirl19/VRGDG_CustomNodePack). Install that pack separately for those canvas nodes and their example workflows. The AI Video Builder and its core nodes remain in this repository.

---

## Resource monitor

The AI Video Builder shows host RAM and NVIDIA VRAM usage in its top toolbar, refreshing about every two seconds while the Builder is open. The compact display hides automatically when the toolbar is too narrow. GPU readings require the driver's `nvidia-smi`; RAM remains available when GPU sensors are unsupported.

The optional canvas strip can be enabled under **Settings → VRGDG → Resource Monitor**. It adds GPU load, temperature, RAM/VRAM bars, optional fan/clock/power details, and an idle-only **Clear memory** action. Readings come from the machine running ComfyUI.

## 🎬 AI Video Builder

Add the node named **`VRGDG AI Video Builder UI`** to open the Builder.

Use it to:

- 🎵 Build projects from songs, audio, SRT files, lyrics, or manually timed scenes.
- 🧙 Start quickly with the guided Wizard, Storyboard Builder, and Reference Builder.
- 🖼️ Plan and generate scene images with character and location references.
- 🎞️ Create image-to-video, text-to-video, ID-LoRA, and First/Last Frame scenes.
- 🔗 Build chained or independent First/Last Frame sequences for stronger continuity.
- 🗣️ Review lyrics, map singers, plan lip-sync scenes, and mark instrumental or B-roll sections.
- ✨ Repair faces, enhance clips, apply post-processing, and compare results.
- 🎚️ Preview timing on the visual timeline, calibrate beat markers, and stitch the final video.
- 💾 Save, branch, export, import, and continue portable Builder projects.
- 🤖 Use built-in LLM, local LM Studio, API, and Browser AI options where supported.
- 🌐 Render with **API Nodes → Seedance 2.5 Reference to Video**, using your ComfyUI account and credits. Convert existing prompts with backups, preserve reference order, and use project or custom scene audio. See the [API video setup guide](docs/API_VIDEO_MODELS.md).

> **Overlays, transitions, effects and filters** (grain, vignette, text, LUTs, clips such as dust and light leaks, and the transitions, effects and filters on the timeline) belong to the **H3 Studio** desktop app, which draws them into the exported video. This ComfyUI Builder does not show them. A project made in H3 Studio keeps them when you open it here.

📖 **New here? Start with the full [AI Video Builder Guide](Workflows/LTX-2_Workflows/Video_Builder/readme.md).**

Agents can render MiniMax two-pass scenes through the scene-video API using `render_phase: "draft"`, `"finalize"`, or `"full"` (default). Drafts save pass-1 latents for review; finalize runs pass 2 from the saved draft. `render_phase: "redecode"` makes a scene's video again from its saved latent at a finer `crf` than the render's, without sampling. See [API endpoints](Api_Endpoints.md).

✨ **Or chat with a GPT and ask any question about the video builder. [HERE](https://chatgpt.com/g/g-6a6b4799d0e48191acf5a97fb2132ba9-ltx-2-3-music-video-builder-guide)** 

---

## 📦 Which Repository Do I Need?

| What you want to use | Repository |
| --- | --- |
| AI Video Builder, Wizard, Storyboard Builder, Reference Builder, scene rendering, project timing, and final assembly | [AI Video Builder — this repository](https://github.com/vrgamegirl19/comfyui-vrgamedevgirl) |
| Standalone face-repair, utility, LLM, long-video, LoRA-training, LTX/MiniMax, music/audio, and workflow-UI nodes | [VRGDG Custom Node Pack](https://github.com/vrgamegirl19/VRGDG_CustomNodePack) |

The standalone tools moved to **VRGDG Custom Node Pack**. Use that repository for their installation instructions, documentation, and example workflows, including dataset/training tools and standalone speech/music tools.

The Builder keeps its integrated tools and the core nodes needed by its workflows. Features such as scene face repair, enhancement, LLM prompting, and audio preparation remain part of the Builder workflow; they are separate from the standalone canvas-node collection.

Some example workflow JSONs remain in the [workflow library](Workflows/README.md). Their presence does not mean every node they use is bundled here. Install the separate pack when an example requires its nodes, along with any other dependencies listed for that workflow.

---

## 🚀 Quick Start

1. Install this AI Video Builder repository and restart ComfyUI.
2. Hard refresh the ComfyUI browser page so the latest JavaScript UI files load.
3. Add **`VRGDG AI Video Builder UI`**.
4. Create a project and add audio, SRT timing, or scenes.
5. Use the Wizard or Storyboard Builder to plan the project.
6. Generate and approve scene images, render scene videos, then stitch the final video.

---

## 📦 Installation

### 🧰 ComfyUI Manager — Recommended

1. Open **Manager** → **Install Custom Nodes**.
2. Search for `vrgamedev`, or install from:

```text
https://github.com/vrgamegirl19/comfyui-vrgamedevgirl
```

3. Restart ComfyUI and hard refresh the browser page.

### 🖐️ Manual Install

Clone this repository into `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/vrgamegirl19/comfyui-vrgamedevgirl.git
```

Then install the Python requirements using the Python environment that runs ComfyUI. For the Windows portable build, run this from `ComfyUI_windows_portable`:

```bat
python_embeded\python.exe -m pip install --upgrade pip setuptools wheel
python_embeded\python.exe -m pip install Cython scikit-build-core
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui-vrgamedevgirl\requirements.txt
```

The first two commands prepare the build tooling needed by `voxcpm` and `llama-cpp-python`, especially on Windows with Python 3.13. `llama-cpp-python` may also need a working CMake/Ninja and C++ compiler when a compatible prebuilt wheel is unavailable. Python 3.12 is the safer choice for older Windows portable environments.

Restart ComfyUI and hard refresh the browser page after installation.

---

## 💡 Good to Know

- The **Video Builder guide** is the main source for setup, model paths, screenshots, and step-by-step help.
- Start with a short project or a few scenes before committing to a long render.
- Save often, and use **Branch Project** or **Export Shareable Project ZIP** before major experiments.
- Builder rendering, Browser AI, face-repair, and enhancement workflows may need additional models and custom nodes; see the guide’s [Required Custom Nodes](Workflows/LTX-2_Workflows/Video_Builder/readme.md#required-custom-nodes).
- MiniMax H3 2 Pass and 3 Pass finish with a resize set to **nvidia_rtx_vsr** (NVIDIA RTX Video Super Resolution), which needs the optional `nvidia-vfx` library and an NVIDIA RTX GPU. Without it the pack resizes with `lanczos` instead and prints a `[VRGDG MiniMax]` warning. To use RTX Video Super Resolution, press **Install** on the **nvidia-vfx** card in the Video Builder's Custom Nodes dialog (about 800 MB), or install it yourself with the extra index (plain PyPI only has a stub) and restart ComfyUI:

  ```bat
  python_embeded\python.exe -m pip install nvidia-vfx --extra-index-url https://pypi.nvidia.com
  ```
- For standalone nodes and training tools, install [VRGDG Custom Node Pack](https://github.com/vrgamegirl19/VRGDG_CustomNodePack) and follow its setup instructions.

---

## 🧑‍💻 Author & Community

Created by **VRGameDevGirl** ✨

- 💬 [Join the Discord community](https://discord.gg/FJ9VvCDXw3)
- ☕ [Support VR Game Dev Girl](https://buymeacoffee.com/vrgamedevgirl)
- 📺 [Videos created with these workflows](https://www.youtube.com/playlist?list=PLQ0zxAQhttlbkHBNHUgvzIOL610JW-_hZ)
- 🎓 [Walkthroughs and update videos](https://www.youtube.com/playlist?list=PLegqSfZi4nfo) — Some features have been updated since these were recorded, so the current UI may look different.

---

## 📜 License

Covered original code is available under the **VRGameDevGirl Community and Commercial License**. Copyright (c) 2026 **Jean Thompson (VRGameDevGirl)**.

**You may create, sell, and monetize videos and other creative media without paying a software license fee**, including paid client work and internal studio production. Model licenses and other third-party rights still apply.

Free community forks and modifications are permitted under the license. **Commercializing the covered software itself requires a separate paid written license**: this includes selling the code or forks, commercial integrations, paid software access, and commercial hosted generation services.

For commercial licensing, contact **Jean Thompson** at **[jeanthompson1984@gmail.com](mailto:jeanthompson1984@gmail.com)**. A usage license does not transfer copyright ownership.

This is a source-available license. Models, ComfyUI, dependencies, and third-party code retain their own terms. Previously granted AGPL and other rights are not revoked. See [LICENSE](LICENSE) for the full terms and the [licensing guide](docs/licensing/README.md) for examples and transition details.
