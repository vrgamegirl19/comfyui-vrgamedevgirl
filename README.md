# 🎮 VRGameDevGirl’s LTX 2.3 Video, Image & Creative Workflow Nodes for ComfyUI

A growing collection of custom ComfyUI nodes for **AI video creation, music videos, storyboarding, image generation, video enhancement, face repair, editing, LoRA training, and workflow automation**.

The flagship tool is the **LTX 2.3 Video Builder**: a scene-by-scene production workspace that brings planning, prompting, media generation, timing, review, and final assembly together inside ComfyUI.

---

## 🎬 LTX 2.3 Video Builder

Add the node named **`VRGDG Music Video Builder UI`** to open the Builder.

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

📖 **New here? Start with the full [LTX 2.3 Video Builder Guide](Workflows/LTX-2_Workflows/Video_Builder/readme.md).**

---

## 🌟 Useful Nodes & Tools

These are some of the most useful tools included in the pack. Many can be used on their own in a normal ComfyUI workflow.

### 🧠 Planning & Project Tools

- **`VRGDG Storyboard Creator with Browser AI — Open This`** — Create project-aware start/end storyboard images with supported browser image tools.

### ✨ Repair, Enhance & Finish

- **Face Fix node set** — Detect and track a face, enhance guided anchors, process the crop through LTX, and composite the repaired face back into the source video. Start with the included [Face Fix workflow](Workflows/FaceFix/VRGDG_FaceFix_Workflow.json).
- **Video Enhance node set** — Create guided enhancement anchors, process a full video through LTX, and restore the exact original resolution and frame count.
- **Z-Image upscaler workflows** — Upscale and refine images with Z-Image using the included workflows for several source pipelines. Browse the [Z-Image Upscale workflows](Workflows/Z-ImageUpscale/).
- **Image comparison tools** — Compare an original and processed image directly inside ComfyUI.
- **Fast Film Grain, Color Match, and Sharpening nodes** — Add cinematic grain, match a reference palette, or restore edge detail efficiently across image batches.

### 🎨 Dataset & LoRA Tools

- **`VRGDG LoRA Dataset Creator UI`** — Build and review captioned datasets for styles, characters, and experimental edit pairs.
- **LTX 2.3 and Z-Image LoRA training workflows** — Train standard video, audio, audio/video, and Speed LoRAs with the included [updated LoRA training workflows](Workflows/LTX-2_Workflows/Lora_Training/UpdatedWorkflows/).
- **`VRGDG Musubi-Tuner Installer`** and **Krea 2 tools** — Set up supported training environments and use preset-based training, sampling, and comparison tools.
- **Preview and grid plot nodes** — Compare checkpoints, prompts, strengths, and generated video folders.

### 🔊 Audio, Prompt & Workflow Utilities

- **`VRGDG VoxCPM2 Voice Clone / TTS`** — Generate speech from text, design a voice, continue spoken audio, or clone a voice from a reference clip. Start with the included [VoxCPM2 Voice Clone / TTS workflow](Workflows/VoxCPM2/VRGDG_VoxCPM_TTS.json).
- Audio loading, splitting, timing, transcription, and silent-audio helpers.
- Local and API-based LLM prompt tools for structured image and video prompting.
- Image, text, switch, folder, workflow-runner, and batch-processing utilities.
- LUT, color, grain, sharpening, resize, combine, and general video-processing nodes.

> Some advanced tools need additional models or external components. The Builder guide explains the supported workflows, required custom nodes, model locations, and optional setup.

---

## 🚀 Quick Start

1. Install the node pack and restart ComfyUI.
2. Hard refresh the ComfyUI browser page so the latest JavaScript UI files load.
3. Add **`VRGDG Music Video Builder UI`**.
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
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui-vrgamedevgirl\requirements.txt
```

Restart ComfyUI and hard refresh the browser page after installation.

---

## 💡 Good to Know

- The **Video Builder guide** is the main source for setup, model paths, screenshots, and step-by-step help.
- Start with a short project or a few scenes before committing to a long render.
- Save often, and use **Branch Project** or **Export Shareable Project ZIP** before major experiments.
- Optional training, Browser AI, face-repair, and enhancement tools may have their own setup requirements.

---

## 🧑‍💻 Author & Community

Created by **VRGameDevGirl** ✨

- 💬 [Join the Discord community](https://discord.gg/FJ9VvCDXw3)
- ☕ [Support VR Game Dev Girl](https://buymeacoffee.com/vrgamedevgirl)
- 📺 [Videos created with these workflows](https://www.youtube.com/playlist?list=PLQ0zxAQhttlbkHBNHUgvzIOL610JW-_hZ)
- 🎓 [Walkthroughs and update videos](https://www.youtube.com/playlist?list=PLegqSfZi4nfo) — Some features have been updated since these were recorded, so the current UI may look different.

---

## 📜 License

Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

Commercial use is allowed only when the AGPL-3.0 terms are followed. Closed-source paid apps, hosted services, SaaS products, or commercial wrappers may not use this code without complying with the license and providing the complete corresponding source code under the same license.

See [LICENSE](LICENSE) for the full terms.
