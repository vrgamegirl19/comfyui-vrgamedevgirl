# V9 Video Builder Guide

This guide is for someone opening the Video Builder for the first time. It explains what each main area does, the usual workflow, and where to look when something is missing.

If this guide or the V9 Video Builder helps you, you can support VR Game Dev Girl here: [buymeacoffee.com/vrgamedevgirl](https://buymeacoffee.com/vrgamedevgirl).

![Full Video Builder Window](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Full%20V9%20Video%20Builder%20window.png)

## Table of Contents

- [What the V9 Video Builder Does](#what-the-v9-video-builder-does)
- [What's New In V9](#whats-new-in-v9)
- [Installing Or Switching To V9](#installing-or-switching-to-v9)
- [Opening the Builder](#opening-the-builder)
- [The Main Layout](#the-main-layout)
- [Top Bar Buttons](#top-bar-buttons)
- [Starting or Loading a Project](#starting-or-loading-a-project)
- [Adding Audio and SRT Timing](#adding-audio-and-srt-timing)
- [Working With Scenes](#working-with-scenes)
- [Using the Timeline](#using-the-timeline)
- [Scene Tab](#scene-tab)
- [Image Tab](#image-tab)
- [Video Tab](#video-tab)
- [Audio Tab](#audio-tab)
- [Lyric Mapping](#lyric-mapping)
- [Review Lyrics and Map Singers](#review-lyrics-and-map-singers)
- [Option 2: Create Scenes From Lyrics](#option-2-create-scenes-from-lyrics)
- [Reference Builder](#reference-builder)
- [Video Wizard](#video-wizard)
- [Storyboard Builder](#storyboard-builder)
- [Builder Agent](#builder-agent)
- [Prompt Options](#prompt-options)
- [LLM Runner](#llm-runner)
- [Batch Buttons and Full Builds](#batch-buttons-and-full-builds)
- [Post Process](#post-process)
- [Prompt Creator Panel](#prompt-creator-panel)
- [Prompt Creator Import](#prompt-creator-import)
- [Settings And Audio Notifications](#settings-and-audio-notifications)
- [Required Custom Nodes](#required-custom-nodes)
- [Models and Downloads](#models-and-downloads)
- [Saving Projects](#saving-projects)
- [Recommended Beginner Workflow](#recommended-beginner-workflow)
- [Common Problems](#common-problems)
- [Screenshot Checklist](#screenshot-checklist)

## What the V9 Video Builder Does

V9 Video Builder is a scene-by-scene video creation UI inside ComfyUI. It helps you build a project from audio, SRT timing, lyric timing, scene notes, prompts, images, video clips, and final stitching.

The basic idea is:

1. Create or load a project.
2. Add global audio or per-scene audio.
3. Add scenes manually, from SRT, from Prompt Creator data, or from timestamped lyrics.
4. Add scene notes, lyric notes, video notes, and optional Reference Builder data.
5. Write or generate image prompts.
6. Generate or import images.
7. Write or generate video prompts.
8. Render scene videos.
9. Stitch the final video.

V9 adds a guided Wizard, Storyboard Builder planning tools, Reference-to-Video and Ingredients-to-Video modes, more image backends, per-scene model overrides, an LLM API runner option, and post-process tools for LUTs, grain, compare previews, and overlays.

## What's New In V9

The biggest V9 upgrades are:

| New feature | What it adds |
| --- | --- |
| `Wizard` | A guided music-video setup path for mode choice, model settings, audio, lyric scenes, references, story structure, prompt runs, and final builds |
| `Storyboard Builder` | A planning surface for scene cards, story briefs, story arcs, camera flow, performance style, facial performance, image prompts, and video prompts |
| `Reference to Video` | LTX/MSR reference-video mode that uses mapped subject references instead of only a generated first frame |
| `Ingredients to Video` | LTX Ingredients mode that maps complete ingredients-sheet images to scenes and uses the required Ingredients LoRA |
| `Krea 2` image mode | Two-pass Krea image generation with its own model settings, LoRA strengths, image-to-image controls, and prompt box |
| Per-scene settings | Individual scenes can override ZImage, Ernie, Krea 2, Flux/Klein, NanoBanana, and video model/LoRA settings |
| `LLM Runner` | Text-only prompt writing can use the built-in runner, LM Studio, or an OpenAI-compatible LLM API endpoint |
| `Post Process` | A left-panel tab for LUT browsing, film grain, FX/overlay packs, and before/after compare previews |
| Flow tools | Flow Browser and Flow image-edit helpers are available as separate VRGDG nodes for browser-driven image generation/edit workflows |

## Installing Or Switching To V9

If you do not have the `comfyui-vrgamedevgirl` custom nodes installed yet, install the V9 branch first.

### New Install With ComfyUI Manager

1. Open ComfyUI.
2. Open `Manager` -> `Install Custom Nodes`.
3. Search for `vrgamedev` or paste this Git URL:

```text
https://github.com/vrgamegirl19/comfyui-vrgamedevgirl
```

4. If Manager lets you choose a branch, choose `dev/music-video-builder-ui-test-v9`.
5. Install, restart ComfyUI, then hard refresh the browser page.

If Manager installs the default branch instead of V9, open a terminal in the installed `ComfyUI/custom_nodes/comfyui-vrgamedevgirl` folder and use the branch-switch commands below.

### New Install With Git

Open a terminal in your `ComfyUI/custom_nodes` folder and run:

```bash
git clone --branch dev/music-video-builder-ui-test-v9 https://github.com/vrgamegirl19/comfyui-vrgamedevgirl.git
```

Then restart ComfyUI and hard refresh the browser page.

### If You Already Have The Nodes Installed

If you are already using another branch, stop ComfyUI first, save any open project, and back up anything important before switching.

Open a terminal in your existing `ComfyUI/custom_nodes/comfyui-vrgamedevgirl` folder and run:

```bash
git fetch origin
git switch dev/music-video-builder-ui-test-v9
git pull
```

If Git says the branch does not exist locally yet, run:

```bash
git fetch origin dev/music-video-builder-ui-test-v9:dev/music-video-builder-ui-test-v9
git switch dev/music-video-builder-ui-test-v9
```

To confirm you are on the right branch:

```bash
git branch --show-current
```

It should show:

```bash
dev/music-video-builder-ui-test-v9
```

If you download from GitHub instead of using Git, use the branch dropdown on the repository page, choose `dev/music-video-builder-ui-test-v9`, then download that branch as a ZIP.

After installing or switching branches, restart ComfyUI and hard refresh the browser page so the new JavaScript UI files load.

## Opening the Builder

Add the node named `VRGDG Music Video Builder UI` in ComfyUI.

When the builder opens, it may show a welcome window where you can create a new project or open an existing project.

![ComfyUI Builder Node](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/ComfyUI%20Builder%20Node.png)

![Welcome Window](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Welcome%20Window.png)

Short snippet:

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/Opening%20the%20Video%20Builder%20node%20and%20welcome%20window.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/Opening%20the%20Video%20Builder%20node%20and%20welcome%20window.gif" alt="Opening the Video Builder node and welcome window" width="820"></a>

## The Main Layout

The builder has five main areas:

| Area | What it is for |
| --- | --- |
| Top bar | Project menu, save, Prompt Creator tools, utilities, stop button, model downloads, fullscreen |
| Left panel | Scene list |
| Center preview | Selected image or video preview |
| Right panel | Scene settings, split into Scene, Image, Video, and Audio tabs |
| Bottom timeline | Timing, playback, scene blocks, inserts, notes, beat markers, and selected media tools |

The full builder screenshot at the start of this guide shows these areas together.

## Top Bar Buttons

The top bar contains project-wide tools. These are not tied to only one scene.

| Button | What it does |
| --- | --- |
| `Menu` | Opens project actions such as New Project, Load Project, Prompt Creator import, batch runs, and settings |
| `Quick Save` | Saves the current project immediately |
| `Wizard` | Opens the guided V9 setup flow |
| `Storyboard Builder` | Opens the scene-card planning and Storyboard prompt workspace |
| `Reference Builder` | Opens character/location reference setup for Flux/Klein and Nano B |
| `Line Mapping` | Opens lyric transcription, lyric review, singer mapping, and timing correction tools |
| `LLM Runner` | Chooses whether text-only LLM/Gemma calls use the built-in runner, LM Studio, or an API endpoint |
| `Agent` | Opens the Builder Agent chat helper |
| `Prompt Options` | Opens prompt editing, reload, clear, and prompt-file tools |
| `LLM T2I All` | Creates image prompts for multiple scenes |
| `LLM Video All` | Creates video prompts for multiple scenes |
| `Stop` | Stops the current running builder workflow |
| `Download Models` | Opens model links and model folder guidance |
| `Clear Memory` | Runs memory cleanup |
| `Fullscreen` | Expands the builder UI without closing it |
| `Close` | Closes the builder UI |

If a button opens a modal, use that modal's `Close` button to return to the main builder.

![Top bar buttons](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Top%20bar%20buttons%20screenshot.png)

## Starting or Loading a Project

Use the `Menu` button in the top-left area of the builder.

Important project options:

| Button | Use it when |
| --- | --- |
| `New Project` | You want to start a fresh builder project |
| `Load Project` | You want to open an existing project folder |
| `Load Last Project` | You want to return to the most recent project |
| `Save Project As` | You want to duplicate the current project into a new folder |
| `Quick Save` | You want to save the current project state |
| `Auto save` | You want the builder to save changes while you work |

Projects are saved under the ComfyUI output folder. A builder project contains the session JSON, SRT, generated images, scene videos, prompt files, reference images, and copied audio assets.

![Menu Dropdown](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Menu%20Dropdown.png)

![Load Project Window](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Load%20Project%20Window.png)

Short snippets:

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/Menu%20Start%20new%20Project%20window%20display.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/Menu%20Start%20new%20Project%20window%20display.gif" alt="Create a new project" width="820"></a>

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/menu%20load%20project.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/menu%20load%20project.gif" alt="Load an existing project" width="820"></a>

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/quicksave%20and%20save%20project%20as.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/quicksave%20and%20save%20project%20as.gif" alt="Quick Save and Save Project As" width="820"></a>

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/menu%20save%20project%20as%20window%20display.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/menu%20save%20project%20as%20window%20display.gif" alt="Save Project As window" width="820"></a>

## Adding Audio and SRT Timing

There are two audio styles:

| Audio style | Best for |
| --- | --- |
| Global Audio | Music videos, songs, visualizers, lyric-timed projects |
| Scene Audio | Dialogue, short films, ads, or projects where each scene has its own audio clip |

For a music video, start with global audio.

To add global audio:

1. Open the `Audio` tab on the right.
2. Use the `Timeline Audio` section.
3. Drag audio into the drop zone or use `Choose Global Audio`.

To use SRT timing:

1. Use `Choose SRT` / `Load SRT` if available in the project flow.
2. Or import timing from Prompt Creator.
3. Check the timeline to make sure scenes line up with the audio.

Supported audio formats include WAV, MP3, FLAC, M4A, and OGG.

The `Audio Tab` section later in this guide shows the audio controls.

## Working With Scenes

Scenes are the main building blocks of the video. Each scene has timing, notes, image settings, video settings, and optional audio.

Use the left scene list to select a scene. The selected scene appears in the center preview and its settings appear on the right.

Common scene actions:

| Action | Where |
| --- | --- |
| Add a new scene | Timeline `+ Segment` |
| Add an insert/overlay | Timeline `+ Insert` |
| Delete selected scene | Timeline `x` button |
| Edit scene name | Right panel `Scene` tab, `Scene label` |
| Edit timing | Right panel `Scene` tab, `Start` and `End` |
| Prevent timing changes from SRT import | `Freeze SRT timing` |

![Left Scene List](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Left%20Scene%20List.png)

Short snippets:

Click a GIF to open the MP4.

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/adding%20segmnets%20and%20bulk%20segments.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/adding%20segmnets%20and%20bulk%20segments.gif" alt="Adding segments and bulk segments" width="820"></a>

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/delete%20a%20segment.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/delete%20a%20segment.gif" alt="Delete a segment" width="820"></a>

## Using the Timeline

The timeline is at the bottom of the builder. It shows the global timing of the project, the playhead, scene blocks, notes, waveform, inserts, selected media tools, and beat markers.

Timeline controls:

| Control | What it does |
| --- | --- |
| `Bulk Segments` | Create many manual scenes from pasted durations or start/end times |
| `+ Scene Note` | Show editable note boxes below scenes |
| `+ Video Note` | Show editable video-motion/performance notes below scenes |
| `+ Lyric Note` / `Hide Lyric Notes` | Show or hide the timeline lyric notes lane |
| `Set In` / `Set Out` | Mark a selected range using the playhead |
| `Clear Range` | Remove the selected range |
| `+ Timeline Note` | Add a timeline marker or note |
| `+ Segment` | Add a normal scene |
| `+ Insert` | Add an insert/overlay segment without changing the base timeline |
| Undo / Redo | Revert or restore timeline edits |
| Play / Stop | Preview audio/timeline playback |
| `Select Multi` | Select multiple scenes for batch settings or preview stitching |
| Waveform size | Choose small, medium, or large waveform |
| `Snap beats` | Snap edits to detected beats |
| Zoom `-` / `+` | Zoom the timeline view |
| `Use Frame as Image` | Save the current video frame as the selected scene image |
| `Delete Image/Video` | Remove selected media from the scene |

Timeline lanes:

| Lane | What it is for |
| --- | --- |
| `Base` | The main scene timeline. These clips define the normal order of the final video |
| `Inserts` | Extra insert clips that sit above the base timeline |
| `Director Notes` / Scene Notes | Image/scene direction notes, often used by image prompting |
| `Video Notes` | Motion, camera, acting, and performance notes for video prompting |
| `Lyric Notes` | Lyrics/vocal line for each scene, used by Gemma for I2V/T2V prompting |
| Waveform | Visual display of the audio |

Use `Video Notes` when you want to describe what should happen in motion. Use `Director Notes` or image notes for the still image idea. Use `Lyric Notes` for the exact lyric or vocal line.

![Timeline Controls](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Timeline%20Controls.png)

![Timeline With Scenes](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Timeline%20Scene%20Blocks.png)

![Video Notes lane on the timeline](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Video%20Notes%20lane%20on%20the%20timeline.png)

Short snippets:

Click a GIF to open the MP4.

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/timeline%20base%20and%20insert%20clips.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/timeline%20base%20and%20insert%20clips.gif" alt="Timeline base and insert clips" width="820"></a>

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/timeline%20showing%20all%20note%20lanes%20.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/timeline%20showing%20all%20note%20lanes%20.gif" alt="Timeline showing all note lanes" width="820"></a>

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/hide%20notes.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/hide%20notes.gif" alt="Hide notes" width="820"></a>

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/unfreeze%20and%20freeze%20timeline%20and%20timeline%20edits.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/unfreeze%20and%20freeze%20timeline%20and%20timeline%20edits.gif" alt="Freeze and timeline edits" width="820"></a>

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/select%20multi.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/select%20multi.gif" alt="Select multi" width="820"></a>

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/waveform%20settings.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/waveform%20settings.gif" alt="Waveform settings" width="820"></a>

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/beat%20markers%20and%20snap%20to%20beats.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/beat%20markers%20and%20snap%20to%20beats.gif" alt="Beat markers and snap to beats" width="820"></a>

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/delete%20image.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/delete%20image.gif" alt="Delete image" width="820"></a>

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/delete%20a%20video.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/delete%20a%20video.gif" alt="Delete video" width="820"></a>

## Scene Tab

The `Scene` tab stores the general information for the selected scene.

Main fields:

| Field | Meaning |
| --- | --- |
| `Scene label` | Name of the scene |
| `Freeze SRT timing` | Keeps this scene timing from being changed by timing imports |
| `Start` / `End` | Scene start and end time |
| `Prompt JSON path` | Concept prompt JSON to import scene ideas |
| `I2V motion notes JSON path` | Motion notes to import for video prompting |
| `Use VRGDG text context files` | Use the default/global text context files |
| `Global theme/style text file` | Overall style, look, and mood reference |
| `Global story idea text file` | Overall story/concept reference |
| `Global subject/scene text file` | Character, subject, and scene reference |

Use this tab first when a scene needs better direction before image or video generation.

![Scene Tab](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Scene%20Tab.png)

## Image Tab

The `Image` tab creates or manages the selected scene image. Pick the image model at the top of the tab.

Image modes:

| Mode | Use it for |
| --- | --- |
| `ZImage` | Main local image generation workflow |
| `Flux Klein` | Flux/Klein image generation, including reference image ingredients |
| `Nano B` | NanoBanana image generation with optional reference images and API key |
| `Ernie` | Ernie Image generation |
| `Krea 2` | Two-pass Krea image generation with optional LoRAs and image-to-image input |
| `Enhance` | Upscale or enhance a selected image |
| `+ Custom` | Load your own image for the scene |

Each image mode usually has three subtabs:

| Subtab | What it controls |
| --- | --- |
| `Models` | Model files, VAE/CLIP, Gemma models, LoRAs |
| `Image Settings` | Size, seed, batch size, image-to-image, reference images, trigger phrase |
| `LLM Prompting` | Notes, Gemma prompt generation, and the final image prompt |

Basic image workflow:

1. Select a scene.
2. Open `Image`.
3. Pick an image mode, usually `ZImage` for a first test.
4. Add notes in `LLM Prompting`.
5. Click `Gemma T2I` to create a prompt, or type your own prompt.
6. Click the create button for that model, such as `Create Z-Image`.
7. Review the image in the center preview.

### ZImage

`ZImage` is the local text-to-image workflow. It can use a standard text prompt, optional LoRAs, and optional image/reference prompting depending on the current settings.

Common ZImage controls:

| Control | What it does |
| --- | --- |
| `ZImage model` | Diffusion model file |
| `CLIP` | Text encoder file |
| `VAE` | VAE file |
| `Non-Vision text Gemma model` | Gemma model used for text-only prompt creation |
| `Vision Gemma model` / `Vision mmproj` | Used when Gemma needs to look at an image reference |
| `Use LoRAs?` | Enables LoRA settings |
| `LoRA count` | Number of LoRAs to show/use |
| `Pass 1` / `Pass 2` strengths | LoRA strength for each pass when the workflow supports two-pass generation |
| `Gemma T2I` | Generates the text-to-image prompt |
| `Create Z-Image` | Runs the ZImage image workflow |

### Flux/Klein

`Flux Klein` supports image ingredients and Reference Builder images.

Use Flux/Klein when you want:

- a character reference plus a location reference
- multiple image ingredients
- a global subject reference used across every scene
- scene-specific images such as props, backgrounds, or style references

Common Flux/Klein controls:

| Control | What it does |
| --- | --- |
| `Image trigger phrase` | Optional phrase added to the start of prompts |
| `Use global image ingredients` | Adds global image references to every scene |
| `Image ingredients` drop area | Scene-specific images for character, background, props, or style |
| `Upload Images` | Opens file picker for image ingredients |
| `Clear Images` | Removes loaded image ingredients |
| `Gemma Flux Prompt` | Uses Gemma vision/text to create a Flux/Klein prompt |
| `Create with Flux/Klein` | Runs the Flux/Klein image workflow |

If Reference Builder is enabled, Flux/Klein can automatically include the mapped subject and location images for each scene.

### Nano B

`Nano B` is the NanoBanana image mode. It uses an API key and can use reference images. It can also receive Reference Builder subject/location references.

Common Nano B controls:

| Control | What it does |
| --- | --- |
| `API key` | NanoBanana/Google API key used by the hidden workflow |
| `Model` | NanoBanana model choice |
| `NanoBanana reference images` | Drop or upload reference images for the scene |
| `Global reference images` | Shared references used across scenes when enabled |
| `Gemma NB Prompt` | Creates a NanoBanana prompt from notes and references |
| `Create with NanoBanana` | Runs the NanoBanana image workflow |

Nano B prompts do not need strict section headers. If Gemma creates a normal usable image prompt, Nano B can still run.

Nano B works best when its prompt clearly says the reference images are identity/location references, not images to paste into the output. If the output looks like a character was dropped into the reference location, rewrite the prompt with stronger camera language such as `close-up`, `upper body`, `low angle`, `profile`, or `new camera position`.

### Ernie

`Ernie` is another image-generation mode. It works similarly to ZImage from the UI side: choose models, set image settings, optionally use Gemma, then create the image.

Use Ernie when you want to compare a scene image against ZImage, Flux/Klein, or Nano B.

### Krea 2

`Krea 2` is a two-pass image mode. It has its own model picks, optional LoRAs, image-to-image controls, prompt trigger phrase, and per-scene override toggle.

Use Krea 2 when you want:

- a polished alternate image pass for the same scene prompt
- separate LoRA strengths for the first and second image pass
- image-to-image testing from a loaded reference image
- a scene-specific image model setup without changing global ZImage, Ernie, Flux/Klein, or Nano B settings

Common Krea 2 controls:

| Control | What it does |
| --- | --- |
| `Use custom Krea 2 settings for this scene` | Lets the selected scene override global Krea 2 model and generation settings |
| `Krea 2 Models` | Sets the Krea workflow model files |
| `Use LoRAs?` | Enables Krea 2 LoRA rows |
| `Pass 1` / `Pass 2` strengths | Controls LoRA influence in each generation pass |
| `Load I2I Image` | Loads a source image for image-to-image generation |
| `Gemma T2I` | Creates a Krea 2 image prompt |
| `Create with Krea 2` | Runs the Krea 2 image workflow |

### Enhance

`Enhance` works on an existing selected image. Use it to upscale, enhance, or perform image-to-image improvement.

The LLM Prompting tab in Enhance can use the selected/custom image as context to create a better enhancement prompt.

### Load Custom

`+ Custom` / `Load Custom` lets you use your own image for the selected scene instead of generating one.

Use this for:

- images created outside ComfyUI
- screenshots
- previous workflow outputs
- manually curated scene images

The loaded image becomes the selected scene image, so video generation can use it the same way it uses generated images.

### Prompt Sharing Across Image Models

When Gemma creates a T2I prompt for a scene, the builder can copy that prompt into the matching prompt boxes for the other image models. This makes it easier to try ZImage, Flux/Klein, Nano B, Ernie, or Enhance without asking Gemma to rewrite the same scene every time.

You can still edit each model's prompt after it is copied.

### Image Trigger Phrase

The image trigger phrase is added at the start of image prompts when it is filled in.

Use it for model-specific trigger words, LoRA trigger phrases, or a short global style phrase. Leave it blank if the model does not need one.

### Per-Scene Image Settings

V9 lets a single scene override the global settings for each supported image mode.

Use the `Use custom ... settings for this scene` toggles when one scene needs a different model, seed, LoRA, image-to-image source, reference setup, resolution, or trigger phrase. Multi-select can apply many of these model/settings changes to several selected scenes at once.

If the toggle is off, the scene follows the global settings for that image mode again.

![Image Tab Model Chooser](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Image%20Tab%20Model%20Chooser.png)

![ZImage Prompting](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/ZImage%20Prompting.png)

![Flux Reference Images](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Flux%20Reference%20Images.png)

![Nano B model settings](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Nano%20B%20model%20settings.png)

![Nano B image settings](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Nano%20B%20image%20settings.png)

![Nano B LLM Prompting settings](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Nano%20B%20LLM%20Prompting%20settings.png)

## Video Tab

The `Video` tab creates the selected scene video. At the top, choose between:

| Mode | What it does |
| --- | --- |
| `Image to Video` | Uses the selected scene image plus a video prompt |
| `Text to Video` | Uses a text prompt directly, without requiring a scene image |
| `Reference to Video` | Uses LTX/MSR reference images and text prompting for the selected scene |
| `Ingredients to Video` | Uses a mapped Ingredients sheet image and the required Ingredients LoRA |

The Video tab has three subtabs:

| Subtab | What it controls |
| --- | --- |
| `Models` | LTX/video model files, audio VAE, Gemma models, video LoRAs |
| `Video Settings` | FPS, width, height, seed, warm up frames, cool down frames |
| `LLM Prompting` | Motion notes, Gemma I2V/T2V prompt generation, final video prompt |

Basic Image-to-Video workflow:

1. Create or load an image for the scene.
2. Open `Video`.
3. Choose `Image to Video`.
4. In `LLM Prompting`, write motion notes.
5. Keep `Use image reference for I2V prompt?` checked if you want Gemma to look at the scene image.
6. Click `Gemma I2V`.
7. Review or edit the `Video prompt`.
8. Click `Create Scene Video`.

Basic Text-to-Video workflow:

1. Open `Video`.
2. Choose `Text to Video`.
3. Write motion notes or a full video prompt.
4. Optionally use a reference image for Gemma prompt writing.
5. Click `Create Scene Video`.

### Image To Video

`Image to Video` uses the selected scene image as the first-frame visual source. This is the normal workflow after creating scene images.

Important controls:

| Control | What it does |
| --- | --- |
| `Use image reference for I2V prompt?` | Gemma looks at the scene image while writing the video prompt |
| `I2V motion notes` | User notes for camera movement, acting, motion, and performance |
| `Gemma I2V` | Creates the image-to-video prompt |
| `I2V prompt` | Final prompt sent to the video workflow |
| `Create Scene Video` | Renders the selected scene video |

If Lyric Mapping has been saved, Gemma can use the lyric line, singer choices, instrumental flag, and B-roll/no-lip-sync flag while writing the I2V prompt.

### Text To Video

`Text to Video` creates video without requiring a scene image.

Use it when:

- you want to skip image generation
- you are building from text prompts only
- you are making quick motion tests
- you want a scene generated directly from concept and video notes

For T2V, Gemma uses the concept/image prompt, video notes, lyric notes, singer mapping, and user motion notes to create the video prompt.

Use T2V when image generation is not needed, or when you want the video model to invent the first frame from text.

### Reference To Video

`Reference to Video` is the LTX/MSR reference-video mode. It is useful when the scene should be driven by a mapped subject reference rather than only a generated scene image.

Use it when:

- the same character identity must stay consistent across scenes
- you have MSR subject references in Reference Builder
- you want Storyboard Builder prompts to describe motion while references carry identity
- you are making a lyric-driven video where singer mapping and subject mapping matter

Important controls:

| Control | What it does |
| --- | --- |
| `Required MSR LoRA` | Required LoRA for Reference-to-Video |
| `MSR strength` | Strength for the MSR reference-video LoRA |
| `Gemma Reference Video` | Writes the reference-video prompt for the selected scene |
| `Reference Builder` | Opens the MSR reference/mapping setup when this mode is active |

Reference-to-Video works best after saving lyric/singer mapping and Reference Builder subject mapping.

### Ingredients To Video

`Ingredients to Video` uses complete Ingredients sheet images mapped to scenes. Each scene can receive its own Ingredients sheet from the Ingredients Reference Builder.

Use it when:

- you have a prepared reference sheet for a character, pose, outfit, scene, or location
- you want a reference sheet to act as the scene's visual source
- you want lyric review and scene mapping to decide which sheet belongs to which moment

Important controls:

| Control | What it does |
| --- | --- |
| `Required Ingredients LoRA` | Required LoRA applied for Ingredients-to-Video |
| `Pass 1` | Strength for the required Ingredients LoRA on the first pass |
| Width/height | Defaults to the Ingredients training-friendly resolution, with a warning when needed |
| `Gemma Ingredients Video` | Writes the Ingredients-to-Video prompt |
| `Ingredients Reference Builder` | Uploads, describes, and maps Ingredients sheets to scenes |

The Ingredients LoRA was trained around `768x448`. Other sizes can work, but unusual aspect ratios may reduce composition quality.

### Video Notes

Video Notes are separate from image/scene notes. Use them for motion-specific instructions such as:

- camera movement
- character movement
- performance energy
- lip-sync direction
- environmental motion
- action beats

Example:

```text
Slow side dolly as the singer leans against the door frame, hair moving slightly in the wind.
```

### I2V Prompt Enhancement Pass

`I2V prompt enhancement pass` is an optional extra Gemma cleanup pass for video prompts.

When enabled, the builder creates the first video prompt, then asks Gemma to clean it into a stronger video-ready structure.

Use it when:

| Situation | Why |
| --- | --- |
| Prompts feel too loose | The pass makes them more structured |
| Lyrics are not being handled clearly | The pass can reinforce singer/lyric behavior |
| Multiple singers are confusing Gemma | The pass can keep all listed singers active |
| Instrumental/B-roll scenes mention singing | The pass can remove singing/no-vocal wording |

Leave it off if you prefer to manually write or preserve the exact prompt.

### Seeds And Custom Scene Settings

The global video settings apply to all scenes by default.

Turn on custom scene settings when a specific scene needs different:

- model files
- LoRAs
- LoRA strengths
- trigger phrase
- FPS
- seed
- width/height
- warm up frames
- cool down frames

Use this for one-off scenes, special LoRA tests, or different video resolutions.

### Video LoRAs

Video LoRAs can use separate strengths for pass 1 and pass 2 when the hidden workflow supports it.

| Setting | What it means |
| --- | --- |
| `Use video LoRAs?` | Enables video LoRA selection |
| `Video LoRA count` | How many LoRA rows are visible |
| `Pass 1` | Strength used during the first video pass |
| `Pass 2` | Strength used during the second video pass |

Use a lower pass 1 strength when a LoRA hurts motion. For example, some style LoRAs trained on images work better at `0.5` on pass 1 and `1.0` on pass 2.

### Warm Up And Cool Down Frames

Warm up and cool down frames help the hidden video workflow create smoother scene clips.

| Field | What it is for |
| --- | --- |
| `Warm Up Frames` | Gives the workflow extra lead-in frames before the final trimmed section |
| `Cool Down Frames` | Gives the workflow extra frames after the final trimmed section |

If a scene starts too stiffly, check that warm up frames are enabled and set to a useful number.

![Video Mode Chooser](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Video%20Mode%20Chooser.png)

![Video Prompting](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Video%20Prompting.png)

## Audio Tab

The `Audio` tab controls audio for the selected scene and the overall timeline.

Sections:

| Section | What it is for |
| --- | --- |
| `Scene Audio` | Add or manage audio attached to the selected scene |
| `Timeline Audio` | Add or manage the global project audio |

Use `Scene Audio` for scene-specific dialogue or clips. Use `Timeline Audio` for music-video timing.

![Audio Tab](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Audio%20Tab%20Timeline%20Audio.png)

## Lyric Mapping

The `Lyric Mapping` button opens the tools used to connect audio, lyrics, scene timing, singers, and no-lip-sync sections.

Use Lyric Mapping when you want Gemma to know:

| Question | Why it matters |
| --- | --- |
| What lyric is happening in each scene? | Gemma can include the correct vocal line in video prompts |
| Who is singing? | Duets and multi-character scenes can keep the right person lip-syncing |
| Is this scene instrumental? | Gemma can avoid creating singing or mouth movement |
| Is this scene B-roll? | A person can appear on screen without lip-syncing |
| Are the scene timings correct? | LTX receives better audio and prompt timing |

Lyric Mapping has two main jobs:

1. Put lyric notes onto the timeline.
2. Review and correct those notes before creating video prompts.

The usual flow is:

1. Load the project audio in the `Audio` tab.
2. Open `Lyric Mapping`.
3. Choose whether you already have timeline scenes.
4. Transcribe lyrics or create scenes from lyrics.
5. Open `Review Lyrics + Map Singers`.
6. Correct lyrics, timing, singers, instrumental sections, B-roll, and locations.
7. Save the lyric mapping.
8. Run Gemma video prompting.

![Lyric Mapping Step 1 window](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Lyric%20Mapping%20Step%201%20window.png)

### Step 1: Transcribe Lyrics Or Create Scenes

The first Lyric Mapping screen has two starting options.

| Option | Use it when |
| --- | --- |
| `Option 1: Existing scenes` | You already have timeline scenes and want to fill lyric notes into them |
| `Option 2: Create scenes from lyrics` | You do not have timeline scenes yet and want the builder to create scenes from the song |

### Option 1: Existing Scenes

Use `Transcribe Existing Scenes` when your timeline already has scene blocks.

This keeps the current scene timing and sends the global audio plus the builder SRT timing to the transcription workflow. The result is written into each scene's `Lyrics / vocal line` field.

Use this when:

| Situation | Why |
| --- | --- |
| You imported scenes from Prompt Creator | The scene timing already exists |
| You manually created scenes | You want lyrics attached to those existing timings |
| You adjusted timing by hand | You do not want the transcriber to replace the whole timeline |

After it finishes, open `Review Lyrics + Map Singers` to fix any timing or lyric mistakes.

### Option 2: Create Scenes From Lyrics

Use `Create Scenes From Lyrics` when you do not have scenes yet.

This option listens to the loaded audio, uses optional reference lyrics, and creates timeline scene blocks from the detected lyric timing.

Before using Option 2:

1. Open the `Audio` tab.
2. Load the song or voice track as timeline/global audio.
3. Return to `Lyric Mapping`.
4. Click `Create Scenes From Lyrics`.

The Create Scenes window includes these controls:

| Control | What it does |
| --- | --- |
| `Reference lyrics` | Optional, but recommended. Paste the real lyrics so transcription and alignment are more accurate |
| `Language` | The language Whisper should use, such as `english` |
| `Segment mode` | Controls how reference lyrics become timeline scenes |
| `Include instrumental gaps` | Adds no-vocal scenes for long gaps between vocal sections |
| `Instrumental text` | Text used for no-vocal scenes, usually `[instrumental]` |
| `Min gap seconds` | Minimum no-vocal gap length before the builder creates an instrumental scene |
| `Min scene seconds` | Prevents very tiny scene blocks |
| `Max scene seconds` | Prevents one lyric or instrumental section from becoming too long |
| `Vocal tail padding` | Adds a little extra time after vocal chunks so last words are less likely to get cut off |
| `Create Timeline Scenes` | Runs the timestamped transcription workflow and replaces the current base timeline with generated lyric scenes |
| `?` hint | Explains the timestamped lyric settings |

![Create Scenes From Lyrics window](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Create%20Scenes%20From%20Lyrics%20window.png)

Important notes:

- Option 2 replaces the current base timeline scenes with the generated lyric scenes.
- Existing generated media is not deleted, but it may no longer line up after timing changes.
- Reference lyrics do not need to be perfect, but cleaner lyrics usually produce cleaner timing.
- Blank lines in pasted lyrics are treated as spacing, not instrumental sections.
- To request a no-vocal section in the lyrics, use marker lines like `[instrumental]`, `[break]`, `[intro]`, `[outro]`, or `[b-roll]`.

### Segment Mode

`Segment mode` controls how the lyric text is divided before timing is created.

| Mode | What it means |
| --- | --- |
| One scene per lyric line | Each lyric line becomes its own scene target |
| Reference chunks / grouped lines | Larger lyric chunks can become longer scenes |
| Whisper chunks | Uses Whisper's detected chunks more directly |

For most music-video projects, start with one scene per lyric line. It is easier to review and fix.

### Include Instrumental Gaps

When `Include instrumental gaps` is on, the builder creates scene blocks for no-vocal areas.

Use this for:

| Song section | Result |
| --- | --- |
| Intro with music but no singing | Creates an instrumental scene |
| Break between verses | Creates an instrumental or no-vocal scene |
| Outro after vocals end | Creates an instrumental scene |

This helps prevent Gemma from making the character sing during parts of the song where nobody should be singing.

### Min Gap Seconds

`Min gap seconds` decides how long a no-vocal gap must be before it becomes its own scene.

Example:

| Value | Behavior |
| --- | --- |
| `1.0` | Creates more instrumental gaps, including shorter pauses |
| `2.0` | Good default for music videos |
| `4.0` | Only longer no-vocal sections become separate scenes |

If you get too many tiny instrumental scenes, increase this value. If instrumental sections are being missed, lower it.

### Min And Max Scene Seconds

`Min scene seconds` prevents scenes that are too short to be useful.

`Max scene seconds` prevents a lyric or instrumental section from stretching too long.

If a lyric line is being stretched across a long intro or break, lower `Max scene seconds` and keep `Include instrumental gaps` enabled.

### Vocal Tail Padding

`Vocal tail padding` adds a little extra time after a vocal phrase. This is useful when the last word of a line feels like it is spilling into the next scene.

Use small values first:

| Value | Use |
| --- | --- |
| `0` | No extra tail |
| `0.25` | Small safety buffer |
| `0.5` | More forgiving for held words |
| `1.0` | Large buffer, but may push into the next section |

Padding helps, but it does not replace manual review. Always check the timing in `Review Lyrics + Map Singers`.

## Review Lyrics And Map Singers

`Review Lyrics + Map Singers` is the main cleanup window after transcription.

This is where you listen scene by scene, fix lyric text, correct timing, choose singers, mark instrumental/B-roll scenes, and save the data that Gemma uses for video prompting.

![Review Lyrics and Map Singers window](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Review%20Lyrics%20%20Map%20Singers%20window.png)

### Top Controls

| Control | What it does |
| --- | --- |
| `?` | Opens the help window for this review screen |
| `Close` | Closes the review window without applying unsaved edits |
| `Single character singer label` | The natural label Gemma/LTX should use for a one-character project |
| `Timing edit mode` | Controls how timing changes affect neighboring scenes |
| Audio player | Plays the project audio for review |
| `Prev` | Moves to the previous scene |
| `Play Selected Scene` | Plays the currently selected scene |
| `Next` | Moves to the next scene |
| `Save Lyrics + Timing + Singers + Locations` | Applies the reviewed lyrics, timing, singer choices, and locations to the real timeline |

### Single Character Singer Label

Use this when the project has one character.

Instead of forcing Gemma to say `Character 1`, type a natural label such as:

- `the woman`
- `the man`
- `the singer`
- `the lead vocalist`

This label is used in video prompts, so choose wording that sounds natural in a prompt.

### Timing Edit Mode

Timing edit mode controls what happens when you change a scene's start or end time.

| Mode | What it does | Best for |
| --- | --- | --- |
| `Lock rest of timeline` | Moves only the shared boundary between neighboring scenes. Later scenes keep their timing | Fixing one boundary without shifting the whole project |
| `Ripple following scenes` | Moves following scenes when you change timing | Large timing edits where everything after the edit should shift |

For lyric cleanup, `Lock rest of timeline` is usually safest.

### Scene Rows

Each scene row contains the tools for one scene.

| Control | What it does |
| --- | --- |
| Scene name/time display | Shows scene number, start time, end time, and duration |
| `Start` | Editable start time for the scene |
| `End` | Editable end time for the scene |
| `Set Start` | Sets the scene start to the current audio playhead time |
| `Set End` | Sets the scene end to the current audio playhead time |
| `Split At Playhead` | Splits the scene into two scenes at the current audio playhead |
| Lyric text box | The lyric or vocal text Gemma will read for that scene |
| Singer checkboxes | Choose who should visibly sing or lip-sync this scene |
| `All` next to a singer | Applies that singer choice across scenes |
| `Group / all visible singers` | Tells Gemma all visible singers should perform together |
| `Location` | Connects the scene to a Reference Builder location |
| `Instrumental` | Marks the scene as no sung lyrics |
| `B-roll / no lip-sync` | Allows visuals/people, but tells Gemma nobody should lip-sync |
| `Play Scene` | Plays only this scene's audio range |
| `Play From Here` | Starts playback at this scene and continues forward |
| `Select` | Selects this scene in the main builder timeline |

### Start, End, Set Start, And Set End

Use the `Start` and `End` boxes when you know the exact time.

Use `Set Start` and `Set End` when listening:

1. Play the audio.
2. Pause where the scene should start or end.
3. Click `Set Start` or `Set End`.

This is useful when the last word of a lyric is getting cut off, or when a scene starts too early.

### Split At Playhead

`Split At Playhead` cuts the selected scene into two scenes at the current audio playhead.

Use it when:

| Problem | Why split helps |
| --- | --- |
| One scene contains two lyric lines | Each lyric can get its own scene |
| A vocal line is followed by an instrumental section | Split the vocal and no-vocal parts |
| A long scene needs more visual variety | Split it into smaller scenes |

The split copies useful scene data into both new pieces, including lyric text, singer settings, B-roll/instrumental state, motion notes, and location mapping. Review both new scenes after splitting.

### Lyric Text Box

This text is what Gemma sees as the vocal line for that scene.

You can type plain lyrics:

```text
I feel the air changing
```

You can also write call-and-response or duet notes:

```text
Male: "I feel the air changing" Female: "Something close, something far"
```

If you do not type quotes, the builder can add quotes around the vocal line when it builds the video prompt.

### Singer Choices

Singer choices tell Gemma who should visibly sing.

| Choice | What it means |
| --- | --- |
| One character checked | Only that character should sing or lip-sync |
| Multiple characters checked | All checked characters should sing together |
| `Group / all visible singers` | Everyone visible in the shot may sing |
| No singer checked | Gemma may treat the visible subject as the singer unless the scene is instrumental or B-roll |

For duets, make sure both singers are checked on the duet lines. Otherwise Gemma may make one person sing while the other only reacts.

### The All Button

The `All` button applies a singer choice across the project.

Use it when one character appears or sings in most scenes.

Be careful with `All` in duets or multi-character projects, because it can assign a singer more broadly than intended.

### Location Dropdown

The `Location` dropdown connects a scene to a Reference Builder location.

This only has options after you create locations in Reference Builder.

Use it when:

| Goal | Result |
| --- | --- |
| Keep a scene in a specific place | Gemma and supported image modes can use that mapped location |
| Use a location reference image | The scene can receive that location reference |
| Keep story continuity | Scenes can stay in the same location across lyrics |

Leave it `Unassigned` if you do not want a mapped location reference.

### Instrumental

Use `Instrumental` when nobody is singing in that scene.

This should be used for:

- intros
- outros
- musical breaks
- instrumental solos
- no-vocal sections

When a scene is marked instrumental, Gemma should avoid singing, lip-syncing, or mouth movement instructions.

### B-roll / No Lip-sync

Use `B-roll / no lip-sync` when the scene can show people, movement, or story visuals, but nobody should mouth the words.

Examples:

| Scene idea | Use B-roll? |
| --- | --- |
| Character walking through a hallway during vocals | Yes, if they should not sing |
| Close-up of hands, props, or environment | Yes |
| Singer performing the lyric | No |
| Instrumental intro with a character preparing on stage | Yes or Instrumental |

B-roll is different from Instrumental. Instrumental means no sung lyric exists in that section. B-roll means a lyric may exist, but the visible shot should not lip-sync it.

### Play Scene And Play From Here

`Play Scene` plays the current scene range only.

`Play From Here` starts at that scene and keeps playing forward. This is useful when you need to find the exact place where a lyric ends, because playback does not stop at the old scene boundary.

### Save Lyrics + Timing + Singers + Locations

This is the most important button in the review window.

It applies the edited review rows back into the actual builder timeline.

It saves:

| Saved item | Where it goes |
| --- | --- |
| Corrected lyric text | Scene lyric notes |
| Start/end timing | Scene timing |
| Singer choices | Lyric/singer mapping |
| Instrumental and B-roll flags | Scene no-lip-sync behavior |
| Location choices | Reference Builder scene mapping |

If you close the window without saving, the review edits may not be applied to the timeline.

### How Gemma Uses Lyric Mapping

After saving, Gemma uses this information when creating I2V or T2V prompts.

| Saved data | How Gemma uses it |
| --- | --- |
| Lyric text | Adds the exact vocal line when needed |
| Singer choice | Makes the correct person sing |
| Multiple singers | Keeps duet/group singing together |
| Instrumental | Avoids singing and lip-sync instructions |
| B-roll/no lip-sync | Allows visual action without mouth movement |
| Location | Helps keep the scene tied to the mapped location |

This is why reviewing lyrics before running Gemma can improve lip-sync, reduce wrong singers, and prevent characters from singing during instrumental sections.

## Reference Builder

The `Reference Builder` button opens the `Reference Image Builder`. This is for projects where scenes need consistent characters, locations, or visual references across many generated images.

Reference Builder can feed references into `Flux/Klein` or `Nano B`, depending on the current image mode. In V9 it also supports the video-side reference flows for `Reference to Video` and `Ingredients to Video`.

Use it when:

| Goal | What to use |
| --- | --- |
| Keep the same character across scenes | `Character References` |
| Keep locations consistent | `Location References` |
| Connect scenes to specific location images | `Scene Mapping` |
| Drive LTX/MSR Reference-to-Video | `MSR References` and subject mapping |
| Drive LTX Ingredients-to-Video | `Ingredients Sheets` and scene mapping |
| Include manually loaded image references too | `Also include manually loaded reference images` |

Main areas:

| Area | What it does |
| --- | --- |
| `Use subject reference` | Sends the character/subject reference into supported image generations |
| `Use mapped location references` | Sends the mapped location image for each scene |
| `Character References` | Upload or generate subject reference images |
| `Extract Subjects` | Uses project prompts/director notes to find subjects |
| `Create Subject with ZImage` | Generates a subject reference image |
| `Location References` | Add, upload, or generate location reference images |
| `Extract Locations` | Uses project prompts/director notes to find locations |
| `Auto Map Locations with Gemma` | Lets Gemma choose which location reference fits each scene |
| `Scene Mapping` | Manually choose the location reference for each scene |
| `Ingredients Sheets` | Upload complete Ingredients sheet images for Ingredients-to-Video |
| `Describe Ingredients Sheets` | Uses vision Gemma to summarize sheet images for mapping/prompt context |
| `Ingredients Scene Mapping` | Chooses which sheet each scene should use |
| `Save Reference Builder` | Saves the reference setup into the project |

Basic Reference Builder workflow:

1. Choose `Flux Klein` or `Nano B` in the `Image` tab, or choose `Reference to Video` / `Ingredients to Video` in the `Video` tab.
2. Click `Reference Builder` in the top bar.
3. Turn on `Use subject reference` and/or `Use mapped location references`.
4. Add character and location references.
5. Map locations to scenes.
6. Click `Save Reference Builder`.
7. Generate images normally from the `Image` tab.

For `Reference to Video`, focus on subject/MSR references and singer/subject mapping. For `Ingredients to Video`, open the Ingredients Reference Builder, upload sheet images, describe them if needed, and map sheets to scenes before running video prompts.

### Character References

Character references are for identity. They help supported image models keep a face, hair, outfit, or character design consistent.

Important controls:

| Control | What it does |
| --- | --- |
| `Character count` | Number of character reference slots |
| Character name/label | The natural name used in mapping, such as `the woman`, `the man`, or a character name |
| `Subject description` | Text description used for creating or understanding the subject |
| Drop/upload box | Add a subject reference image |
| `Create Subject with ZImage` | Generates a subject/reference sheet from the description |
| `Upload Subject Image` | Uploads a custom subject reference |
| `Clear Subject` | Removes the subject image |

If there is only one character, give it a useful label. Avoid leaving it as `Character 1` if you want prompts to sound natural.

### Location References

Location references are for environment identity. They help supported image models understand the location without forcing the exact same camera angle.

Important controls:

| Control | What it does |
| --- | --- |
| `Add Location` | Adds a location slot |
| `Location name` | Short label shown in mapping dropdowns |
| `Description / prompt` | Text description for the location |
| Drop/upload box | Add a location reference image |
| `Create with ZImage` | Generates a location reference image |
| `Upload` | Uploads a custom location image |
| `Clear` | Removes the image but keeps the location slot |
| `Remove` | Deletes the location slot |

### Extract And Map Locations

`Extract Locations` asks Gemma to find location ideas from existing scene prompts, director notes, video notes, lyric context, or project context.

`Auto Map Locations with Gemma` asks Gemma to assign the existing location list to scenes. You can always change the dropdowns manually afterward.

Use `Auto Map Locations with Gemma` after location slots exist. If no locations exist yet, use `Extract Locations` or add locations manually first.

### Map Subjects From Lyrics

`Map Subjects From Lyrics` uses saved lyric/singer mapping to assign characters to scenes. This works best after you have used `Review Lyrics + Map Singers` and saved the lyric mapping.

It does not overwrite your lyric text. It only helps connect scene references to the subjects used in those lyric/singer choices.

### Ingredients Sheet Mapping

Ingredients sheet mapping connects a complete reference sheet image to one or more scenes.

Use it when the selected video mode is `Ingredients to Video`.

The usual flow is:

1. Open `Reference Builder` while `Ingredients to Video` is active.
2. Add one or more Ingredients sheet images.
3. Use vision Gemma to describe sheets when the description is blank or unclear.
4. Map each scene to the matching sheet.
5. Save Reference Builder.
6. Run `Gemma Ingredients Video` or `LLM Video All`.

If lyric/singer mapping already knows which subject appears in each scene, V9 can sync Ingredients scene mapping from those subject mappings.

![Reference Builder Window](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Reference%20Builder%20Window.png)

![Reference Builder with character and location mappings](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Reference%20Builder%20with%20character%20and%20location%20mappings.png)

## Video Wizard

The `Wizard` button opens a guided V9 setup flow. It does not replace the main builder; it calls the same builder tools in a more ordered path.

Use the Wizard when you want the builder to walk through:

| Step | What it helps with |
| --- | --- |
| Mode setup | Choose `Image to Video` or `Reference to Video` and choose the image backend |
| Model settings | Apply video, image, LoRA, Gemma, MSR, and Ingredients defaults back to the builder |
| Audio | Load global timeline audio |
| Lyrics and scenes | Paste reference lyrics and create timeline scenes |
| References | Open the correct Reference Builder mode for the selected video mode |
| Lyric review | Open lyric timing, singer, B-roll, instrumental, location, and Ingredients mapping tools |
| Scene defaults | Fill camera flow, performance style, and facial performance across scenes |
| Story layer | Create a story brief, story arc, lyric sections, and per-scene story beats |
| Prompts/build | Run LLM image prompts, Storyboard/LLM video prompts, and Build Full Video |

The Wizard saves draft progress into the project when possible, so you can close it and continue later.

For `Reference to Video`, the Wizard can run the same Storyboard prompt writer used by Storyboard Builder. `Ingredients to Video` remains available from the main UI and Ingredients Reference Builder.

## Storyboard Builder

`Storyboard Builder` is a planning workspace for scene cards before or during image/video generation.

Use it when you want stronger control over:

| Tool | What it helps with |
| --- | --- |
| Scene cards | Review scene status, prompts, selected images, video prompts, and notes |
| Story brief | Create a compact project summary from lyrics, sections, subjects, and locations |
| Story arc | Build a song-structure story plan across verses, chorus, bridge, intro, or outro |
| Scene beats | Add per-scene story intent from the larger arc |
| Camera motion | Choose from camera-flow presets or specific camera motion categories |
| Still style | Choose image prompt style presets such as cinematic, editorial, beauty, analog, surreal, or studio |
| Performance style | Apply music-video performance presets such as pop, rock, metal, rap, ballad, EDM, or B-roll |
| Facial performance | Keep singing/acting prompts consistent with the lyric or instrumental state |
| Storyboard Gemma All | Write video prompts across scenes using Storyboard context |

Storyboard Builder works especially well with saved lyric mapping and Reference Builder data. For Reference-to-Video projects, it can enforce clearer facial/lip-sync behavior and add reference-aware trigger phrasing before writing video prompts back into the Video Builder scenes.

## Builder Agent

The `Agent` button opens the `Builder Agent`, a chat helper for planning, editing, troubleshooting, and optionally applying changes to the project.

The Agent is useful when you want help with:

| Task | Example request |
| --- | --- |
| Learning the workflow | `Walk me through what to do next.` |
| Scene planning | `Rewrite Scene 5 notes to match the previous scene.` |
| Prompt cleanup | `Make this image prompt more cinematic but keep the subject the same.` |
| Video motion | `Create stronger camera motion for Scene 8.` |
| Story planning | `Plan this whole song as a multi-character video.` |
| Troubleshooting | `Why is this scene not ready to render?` |

Agent controls:

| Control | What it means |
| --- | --- |
| `Context sent to Gemma` | How much project context the Agent receives |
| `Active scene only` | Best for focused changes to the selected scene |
| `Active scene + neighbors` | Best for continuity around the selected scene |
| `Project brief` | Best for overall advice without sending every scene |
| `Full scene plan` | Best for Story Builder and large planning tasks |
| `Agent mode: Manual` | Suggest only; does not apply changes |
| `Agent mode: Auto` | Can update fields, switch modes, select scenes, and run supported actions |
| `Purpose: Beginner help` | Helps a new user step through setup |
| `Purpose: Scene work` | Helps with scene notes, prompts, images, and video |
| `Purpose: Story Builder` | Helps plan a multi-scene story or character structure |
| `Purpose: Troubleshoot` | Helps diagnose missing setup or failed steps |
| `Hint` | Shows examples of what the Agent can do |
| `Pop Out` | Moves the Agent to its own browser window |
| `Min` | Minimizes the Agent without closing the chat |
| `Clear Chat` | Clears the current Agent conversation |

Reference and story tools:

| Tool | Use it for |
| --- | --- |
| `Drop reference image` | Give the Agent a visual reference for the active scene |
| `Upload Ref` | Upload one or more reference images |
| `Add Audio` | Add global/timeline audio while using Story Builder |
| `Story Source` | Paste or edit lyrics/script/source text for Story Builder |
| `Upload Story Images` | Add singers, characters, locations, or aesthetic images |
| `Analyze Images` | Turn story images into compact notes for text-only planning |

Recommended beginner use:

1. Open `Agent`.
2. Set `Agent mode` to `Manual: suggest only`.
3. Set `Purpose` to `Beginner help`.
4. Ask what to do next.
5. Switch to `Scene work` when you want help with a selected scene.
6. Use `Auto: update fields` only when you are comfortable letting it make edits.

![Builder Agent Window](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Builder%20Agent%20Window.png)

![Builder Agent Hints](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Builder%20Agent%20Hints.png)

## Prompt Options

The `Prompt Options` button opens tools for editing, creating, reloading, or clearing final prompt text files for the project.

Use this when the scene prompt boxes need to be refreshed from saved prompt files, or when you want to edit generated prompts in a larger text editor-style workflow.

Prompt Options has two groups:

| Group | Buttons |
| --- | --- |
| `Image` | `Create Concept Prompts`, `Edit Text to Image Prompts`, `Reload Text to Image Prompts`, `Reload Original T2I Prompts`, `Clear All T2I Prompts` |
| `Video` | `Create Motion Notes`, `Edit Image to Video Prompts`, `Reload Image to Video Prompts`, `Reload Original I2V Prompts`, `Clear All I2V Prompts` |

If the current image mode is `Flux/Klein` or `Nano B`, the image prompt buttons change names to match that mode.

Use caution with the clear buttons. They remove prompt text from the project stage they describe.

### Editing Prompt Files

The editor accepts several formats:

Important: after you edit and save T2I or I2V prompt text in the Prompt Options editor, use the matching reload button before running builds. Saving updates the prompt file on disk. Reloading copies the saved prompt file back into the scene prompt boxes the builder uses.

Blank-line format:

```text
Prompt for scene 1

Prompt for scene 2

Prompt for scene 3
```

Key/value format:

```text
Prompt1=Prompt for scene 1
Prompt2=Prompt for scene 2
Prompt3=Prompt for scene 3
```

For I2V prompts:

```text
I2V1=Video prompt for scene 1
I2V2=Video prompt for scene 2
```

JSON format:

```json
{
  "Prompt1": "Prompt for scene 1",
  "Prompt2": "Prompt for scene 2"
}
```

For I2V prompts:

```json
{
  "I2V1": "Video prompt for scene 1",
  "I2V2": "Video prompt for scene 2"
}
```

### Reloading And Clearing Prompts

| Button | What it changes |
| --- | --- |
| `Reload Text to Image Prompts` | Loads the current T2I prompt file into scene prompt boxes after editing/saving |
| `Reload Original T2I Prompts` | Restores the first backup made by the prompt editor |
| `Clear All T2I Prompts` | Clears saved image prompts only |
| `Reload Image to Video Prompts` | Loads the current I2V prompt file into scene prompt boxes after editing/saving |
| `Reload Original I2V Prompts` | Restores the first backup made by the prompt editor |
| `Clear All I2V Prompts` | Clears saved video prompts only |

Clearing prompts does not delete images, videos, LoRAs, reference images, model choices, seeds, scene notes, video notes, or lyric notes.

![Prompt Options Window](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Prompt%20Options%20Window.png)

![Prompt Options image and video groups](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Prompt%20Options%20image%20and%20video%20groups.png)

## LLM Runner

The `LLM Runner` button controls which runner is used for text-only prompt-writing steps. Vision/image-reference Gemma calls still use the built-in vision path.

Options:

| Runner | What it means |
| --- | --- |
| `builtin` | Uses the built-in local GGUF runner |
| `lm_studio` | Uses LM Studio's local server for text-only Gemma calls |
| `llm_api` | Uses an OpenAI-compatible API endpoint for text-only LLM calls |

LM Studio is only for text-only Gemma steps. Vision/image-reference Gemma still uses the built-in GGUF runner.

LM Studio setup fields:

| Field | What to enter |
| --- | --- |
| `LM Studio base URL` | Usually `http://127.0.0.1:1234/v1` |
| `Available LM Studio models` | Click `Load LM Studio Models` to fill this |
| `LM Studio model name` | The loaded chat model name from LM Studio |
| `API key` | Usually blank for local LM Studio |
| `Test LM Studio` | Sends a tiny test prompt to confirm it works |

LLM API setup fields:

| Field | What to enter |
| --- | --- |
| API base URL | OpenAI-compatible `/v1` base URL |
| Model name | Chat/completions model name exposed by that server |
| API key | Key required by the server, if any |
| Test API | Sends a tiny test prompt to confirm the endpoint works |

When a batch run is active, progress windows show the runner name so you can tell whether a text-only pass is using `LM Studio`, `API LLM`, or the built-in runner.

If LM Studio does not list models:

1. Open LM Studio.
2. Go to the Local Server tab.
3. Load a chat model.
4. Start the server.
5. Return to the builder and click `Load LM Studio Models`.

![Gemma Runner Window](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Gemma%20Runner%20Window.png)

![Gemma Runner with LM Studio model dropdown](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Gemma%20Runner%20with%20LM%20Studio%20model%20dropdown.png)

## Batch Buttons and Full Builds

The `Menu` contains batch tools that can work across many scenes.

| Button | What it does |
| --- | --- |
| `LLM T2I All` | Creates image prompts for multiple scenes |
| `LLM Video All` | Creates I2V, T2V, Reference-to-Video, or Ingredients-to-Video prompts for multiple scenes |
| `Image All` | Creates missing image prompts if needed, then creates missing images |
| `Render All` | Renders missing scene videos and can stitch when possible |
| `Stitch Preview` | Stitches selected or ranged existing scene videos into a preview |
| `Build Full Video` | Runs the larger pipeline from prompts/images/videos through final stitch |
| `Remake Mode` | Helps rerun or rebuild outputs |
| `Stop` | Stops the current workflow run |

In `Reference to Video` mode, `LLM Video All` can use the Storyboard prompt writer when launched through the Wizard. In `Ingredients to Video` mode, make sure Ingredients sheets are mapped before running the batch prompt step.

When prompted, choose the safest option first:

| Option | Best for |
| --- | --- |
| Resume missing only | Continue without replacing finished work |
| Keep prompts, redo images/videos | Make new media but preserve prompt work |
| Redo prompts and images/videos | Start fresh for the selected stage |

Build Full Video options:

| Option | What it does |
| --- | --- |
| `Resume missing only` | Keeps existing prompts, selected images, and selected videos. Only creates missing pieces, then stitches |
| `Fresh full rebuild` | Regenerates prompts, images, video prompts, and videos |
| `Keep images, redo I2V prompts and videos` | Keeps selected images, regenerates video prompts and scene videos |
| `Keep images and prompts, redo videos` | Keeps selected images and existing video prompts, creates new scene videos |

Some rebuild choices let you keep the current seeds or randomize them. Keep seeds when you want a similar result with changed settings, such as adding a LoRA. Randomize seeds when you want a fresh variation.

`Image All` is image-only. It stops after the images are created so you can review them before generating videos.

`Render All` is video/stitch focused. It does not regenerate image prompts or images unless the selected build option says so.

![Build Full Video Options](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Build%20Full%20Video%20Options.png)

When the finished video is stitched, the builder shows a `Final Video Ready` popup. Use `Open Video` to preview it.

![Final Video Ready Popup](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Final%20Video%20Ready%20Popup.png)

## Post Process

The left panel has three tabs: `Scenes`, `Tools`, and `Post Process`.

Use `Post Process` after images or videos exist and you want to style, compare, or finish them.

Post Process tools:

| Tool | What it does |
| --- | --- |
| `LUTS` | Shows installed `.cube` LUTs and applies a look to the selected scene |
| `Film Grain` | Applies film-grain settings to the selected scene media |
| `FX` | Opens overlay and visual-FX tools |
| Compare preview | Shows before/after post-process previews in the center preview area |

LUT tips:

- Click a LUT to apply it to the selected scene.
- Drag a LUT onto a scene when you want to target a specific scene.
- Use `Refresh` if you add new `.cube` files while ComfyUI is open.
- The included LUT examples live under the repo's `LUTS/examples` folder.

Post-process changes are scene-level choices. Save the project after choosing looks so the builder can keep those selections with the session.

## Prompt Creator Panel

The Prompt Creator is the companion UI for turning a song, lyrics, SRT timing, and user ideas into concept prompts and motion notes.

Use Prompt Creator when you want the builder to start from a planned prompt package instead of writing every scene manually.

Prompt Creator can create:

| Output | Used by |
| --- | --- |
| SRT/scene timing | Video Builder scene timing |
| Lyric segments | Concept prompt creation and no-vocal detection |
| Extracted subject | Subject prefix and reference context |
| Theme/style | Global visual direction |
| Story idea | Overall story direction |
| Subject and locations | Characters, locations, and scene context |
| Concept prompts | Image/T2I notes in Video Builder |
| I2V motion notes | Video motion notes in Video Builder |

Main Prompt Creator controls:

| Control | What it does |
| --- | --- |
| `Audio file` | Song/audio file used for Whisper/SRT timing |
| `Language` | Whisper language hint |
| `Use SRT duration file` | Uses detected beat/SRT timing instead of fixed duration |
| `Fixed scene duration` | Used when SRT duration is off |
| `Empty lyric segment text` | Text used for no-vocal/blank segments, such as `Instrumental section.` |
| `Append subject to Concept Prompts` | Adds the extracted subject to the start of each concept prompt |
| `Min duration` / `Max duration` | Controls scene duration bounds |
| `Bias` | Guides how strongly the duration logic prefers longer/shorter timing choices |
| `Duration preset` | Chooses the scene timing style |
| `Concept lyric match` | Controls how tightly concept prompts follow the lyrics |
| `Gemma4 text model` | Non-vision model used for Prompt Creator text steps |

User input boxes:

| Box | What to put there |
| --- | --- |
| `Full lyrics` | Full song lyrics. Use the `Sonauto` button if you need a free music creator link |
| `Style/theme` | Visual style, color, mood, genre, or art direction |
| `Story idea` | Overall music video story or structure |
| `Subject and locations` | Characters, outfits, locations, props, and setting details |

Prompt Creator buttons:

| Button | What it does |
| --- | --- |
| `Gemma4 Lyrics` | Helps clean or draft lyric text |
| `Gemma4` buttons on context boxes | Drafts that specific context field |
| `Use GPT` | Uses the alternate GPT helper path when available |
| `Edit Instructions` | Opens custom instructions for that Prompt Creator step |
| `Run` | Runs the full Prompt Creator pipeline |
| `Run: Skip Whisper/SRT` | Uses existing SRT/segment data and runs the later prompt steps |
| `Save Project Draft` | Saves the Prompt Creator draft |
| `Load Project Draft` | Opens a saved Prompt Creator draft |
| `Send To Video Creator` | Sends saved Prompt Creator output into Video Builder |
| `Back To Video Creator` | Returns to Video Builder without necessarily importing new data |

### Concept Lyric Match

`Concept lyric match` controls how literal concept prompts should be.

| Option | Meaning |
| --- | --- |
| Super tight/literal | Use visible lyric objects and actions whenever possible |
| Medium | Keep at least one recognizable lyric object or action while still following story/style |
| Loose | Use the lyric as inspiration, but allow the story and visuals more freedom |
| Super light | Treat lyrics mostly as mood and pacing |

Use tighter settings for lyric-symbolic videos. Use looser settings for abstract, cinematic, or story-first videos.

### Custom Prompt Creator Instructions

Each Prompt Creator Gemma step can have custom instructions.

Use this only when you know what you want to change. Bad instructions can make Gemma produce invalid, short, repeated, or unusable outputs.

Use presets or restore defaults if a custom instruction causes problems.

### Prompt Creator To Video Builder

After running or editing Prompt Creator:

1. Click `Save Project Draft`.
2. Click `Send To Video Creator`.
3. In Video Builder, confirm scenes, notes, prompt paths, audio, and SRT timing came over.
4. Use `Import Data From Prompt Creator` or Prompt Options reload buttons if needed.

## Prompt Creator Import

The builder can import data from the Prompt Creator, and Prompt Creator can send data back into the Video Builder.

Useful buttons:

| Button | What it does |
| --- | --- |
| `Prompt Creator` | Opens the Prompt Creator panel |
| `Import Data From Prompt Creator` | Copies Prompt Creator outputs into the current builder project |
| `Send To Video Creator` | From Prompt Creator, saves/imports the current prompt creator project into Video Builder |
| `Send To Prompt Creator` | From Video Builder, sends audio/SRT/lyrics back to Prompt Creator so you can create concept prompts |
| `Back To Video Creator` | Returns to Video Builder; it is navigation, not the same as importing |
| `Prompt Options` | Opens prompt-related settings/options |
| `LLM Runner` | Opens text-only LLM/Gemma runner tools |
| `Agent` | Opens the builder assistant/agent |
| `Reference Builder` | Helps build reference material for image workflows |

Imported data can include audio, SRT, concept prompts, lyric segments, motion notes, theme/style text, story idea text, and subject/scene text.

If you manually edit Prompt Creator outputs, use the save buttons before sending/importing. The builder reads the saved files, not unsaved text sitting in a box.

If the file paths appear in the Scene tab but the scene note boxes are empty, use the matching import/reload button so the file contents are copied into the scene fields.

![Prompt Creator Import Buttons](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Prompt%20Creator%20Import%20Buttons.png)

## Settings And Audio Notifications

Open `Settings` from the `Menu`.

Settings are project/user preferences that should carry forward between sessions when saved.

Common settings:

| Setting | What it does |
| --- | --- |
| Custom model root | Optional alternate models folder root |
| Audio notifications | Plays a sound when selected events finish or fail |
| Notification volume | Controls notification sound volume |
| Notify on error | Plays an error sound when a run fails |
| Notify on finished item | Plays a sound after scene/image/video tasks finish |
| Notify on full run complete | Plays a sound when a batch/full build finishes |
| Custom success/error sound | Lets you choose your own audio file for notifications |

### Custom Model Root

Use this if your models are not inside the normal ComfyUI `models` folder.

Example:

```text
H:\AIStuff\models
```

Inside that folder, keep the normal ComfyUI model subfolders:

```text
models
  diffusion_models
  text_encoders
  vae
  LLM
  upscale_models
  latent_upscale_models
```

The model pickers look inside the configured root and its known subfolders. If a model is not visible after changing this setting, save settings and refresh/restart the UI.

### Audio Notifications

Audio notifications are optional. They are useful for long overnight runs.

Use them for:

- errors
- each finished scene/image/video
- full build completion
- custom success/failure sounds

Browsers may block sound until you have clicked somewhere in the page at least once.

![Settings window with custom model root and audio notifications](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/Settings%20window%20with%20custom%20model%20root%20and%20audio%20notifications.png)

Short snippet:

<a href="https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/Menu%20settings%2C%20all%20settings.mp4"><img src="https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/video/gifs/Menu%20settings%2C%20all%20settings.gif" alt="Settings and audio notifications" width="820"></a>

## Required Custom Nodes

The Builder UI comes from this repo, but the hidden workflows it launches also use several external custom-node packs. Install these before running full image/video builds.

This list was checked against the hidden workflow templates used by the Builder and Prompt Creator: `text2image_zimage_API.json`, `Krea2_TextToImage_API.json`, `Krea2_API_2Pass.json`, `image_ernie_image_turbo_API.json`, `fluxKleinMultiImage_API.json`, `NB_API.json`, `z_upscaleEnhance_API.json`, `Singlei2vForUI_API.json`, `Singlet2vForUI_API.json`, `SingleRef2VidForUI_API.json`, `SingleIngredients2Video_ForUI_API.json`, `ClearMemory_API.json`, `LTX2.3_Transcribe_API.json`, `LTX2.3_Transcribe_2_API.json`, and `LTX2.3_Music_Video_Creator_Prompt_Creator_API.json`.

Use ComfyUI Manager when possible: open `Manager` -> `Install Custom Nodes`, search the name, install it, then restart ComfyUI. If a workflow still opens with red missing nodes, use ComfyUI Manager's missing-node installer on that workflow.

Core requirement:

| Custom node pack | Needed for |
| --- | --- |
| `comfyui-vrgamedevgirl` | The Video Builder UI, Prompt Creator, Storyboard Builder, lyric tools, VRGDG audio/SRT/project nodes, NanoBanana node, LUT/post-process helpers, and VRGDG workflow glue |

Hidden workflow requirements:

| Custom node pack | Needed for |
| --- | --- |
| `ComfyUI-VideoHelperSuite` | Loading audio/video/image paths and combining video outputs. Look for nodes such as `VHS_LoadAudio`, `VHS_LoadVideo`, and `VHS_VideoCombine` |
| `ComfyUI-LTXVideo` | LTX 2.3 video, audio VAE, I2V/T2V, Reference-to-Video, Ingredients-to-Video, latent upscaling, and LTX guide/reference nodes |
| `ComfyUI-GGUF` | GGUF model loading for video/text models. Look for nodes such as `UnetLoaderGGUF` and `DualCLIPLoaderGGUF` |
| `ComfyUI-KJNodes` | Utility image/video nodes used by the hidden workflows, including resize, image size/count, and KJ VAE loader helpers |
| `comfyui_memory_cleanup` | RAM/VRAM cleanup nodes used between heavy video and image passes. Look for `RAMCleanup` and `VRAMCleanup` |
| `erosdiffusion-eulerflowmatchingdiscretescheduler` | Custom FlowMatch scheduler used by ZImage/Krea-style image workflows. Look for `FlowMatchEulerDiscreteScheduler (Custom)` |

Mode-specific notes:

| Builder feature | Extra dependency notes |
| --- | --- |
| `Image to Video` / `Text to Video` | Requires the LTXVideo, VideoHelperSuite, GGUF, and KJNodes packs above |
| `Reference to Video` | Requires LTXVideo plus the MSR LoRA/model files shown in `Download Models` |
| `Ingredients to Video` | Requires LTXVideo plus the required Ingredients LoRA and Ingredients workflow files |
| `ZImage`, `Flux/Klein`, `Ernie`, `Krea 2` | Use the model files in `Download Models`; if a hidden workflow reports a missing node, run Manager's missing-node installer for that workflow |
| `Nano B` | Uses the VRGDG NanoBanana node in this repo and requires the API key/model setting in the Nano B tab |
| `Prompt Creator` / lyric transcription | Uses this repo's VRGDG lyric/SRT nodes and the Python packages from `requirements.txt`; Whisper/transcription also needs the matching models and packages available in your ComfyUI environment |

Quick missing-node checklist:

1. Restart ComfyUI after installing custom nodes.
2. Open ComfyUI Manager and use `Install Missing Custom Nodes` if any hidden workflow reports missing classes.
3. Check the browser console or ComfyUI terminal for the exact missing `class_type`.
4. Install the missing pack, restart, then reopen the Builder.

## Models and Downloads

Use `Download Models` in the top bar if you need model links and folder locations.

The model download window includes groups for:

| Model group | Used by |
| --- | --- |
| `LLM / Vision` | Gemma text and vision prompt creation |
| `ZImage` | ZImage generation |
| `Flux/Klein 9B` | Higher quality Flux/Klein generation |
| `Flux/Klein 4B` | Smaller/lighter Flux/Klein generation |
| `Ernie Image` | Ernie image generation |
| `LTX 2.3` | Video generation |

After placing models in the correct ComfyUI model folders, restart ComfyUI if dropdowns do not refresh.

Folder examples:

ZImage:

```text
ComfyUI/
  models/
    text_encoders/qwen_3_4b.safetensors
    diffusion_models/z_image_turbo_bf16.safetensors
    vae/ae.safetensors
```

Flux/Klein 9B:

```text
ComfyUI/
  models/
    diffusion_models/flux-2-klein-9b-fp8.safetensors
    text_encoders/qwen_3_8b_fp8mixed.safetensors
    vae/full_encoder_small_decoder.safetensors
```

Flux/Klein 4B:

```text
ComfyUI/
  models/
    diffusion_models/flux-2-klein-4b-fp8.safetensors
    text_encoders/qwen_3_4b.safetensors
    vae/flux2-vae.safetensors
```

LTX 2.3:

```text
ComfyUI/
  models/
    diffusion_models/ltx-2.3-distilled_1.1-Q6_k.gguf
    text_encoders/ltx-2.3-text_projection_bf16.safetensors
    text_encoders/abliterated-sikaworld-high-fidelity-edition.safetensors
    vae/LTX2.3_video_vae_bf16.safetensors
    vae/LTX2.3_audio_vae_bf16.safetensors
    latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
```

![Download Models Window](https://raw.githubusercontent.com/vrgamegirl19/comfyui-vrgamedevgirl/refs/heads/dev/music-video-builder-ui-test-v9/Workflows/LTX-2_Workflows/Video_Builder/images/2026-06-01%2016_02_27-.png)

## Saving Projects

Use `Quick Save` often. Keep `Auto save` on unless you have a reason to turn it off.

Use `Save Project As` before major experiments. This creates a separate copy so you can test new prompts, models, or remake settings without damaging the original project.

Project folders may contain:

| Folder or file | Purpose |
| --- | --- |
| `vrgdg_builder_session.json` | Main builder session |
| `builder_segments.srt` | Scene timing |
| `SceneNotes.json` | Scene notes export |
| `project_context` | Theme, story, subject, and prompt context files |
| `zimage_approved` | Approved/generated scene images |
| `scene_image_previews` | Image preview versions |
| `scene_audio` | Per-scene audio |
| `rendered_scene_videos` | Generated scene videos |
| `prompts` | Prompt and lyric segment files |

## Recommended Beginner Workflow

Use this if you are new and just want the first successful video.

1. Add/open `VRGDG Music Video Builder UI`.
2. Click `New Project`.
3. Add global audio in the `Audio` tab.
4. Import SRT or create scenes with `+ Segment` / `Bulk Segments`.
5. Select the first scene.
6. In `Scene`, name the scene and check the timing.
7. In `Image`, choose `ZImage`.
8. In `LLM Prompting`, write simple scene notes.
9. Click `Gemma T2I`, then `Create Z-Image`.
10. In `Video`, choose `Image to Video`.
11. Add motion notes, click `Gemma I2V`, then `Create Scene Video`.
12. Repeat for a few scenes.
13. Use `Stitch Preview` to check the flow.
14. Use `Build Full Video` or `Render All` once the scenes are ready.
15. Use `Quick Save`.

### Music Video Workflow With Lyrics

Use this when you have a song and want better lip-sync behavior.

1. Create or load a project.
2. Add global audio.
3. Open `Lyric Mapping`.
4. If scenes do not exist, use `Create Scenes From Lyrics`.
5. If scenes already exist, use `Transcribe Existing Scenes`.
6. Open `Review Lyrics + Map Singers`.
7. Correct lyrics, singer choices, instrumental sections, B-roll, locations, and timing.
8. Save lyrics/timing/singers/locations.
9. Add or import image/concept notes.
10. Run `Gemma T2I All` or `Image All`.
11. Review images.
12. Run `Gemma I2V All` or `Build Full Video`.
13. Stitch or build the final video.

### Prompt Creator Workflow

Use this when Prompt Creator makes your concept prompts and motion notes.

1. Open Prompt Creator.
2. Add audio and full lyrics.
3. Run the prompt creator pipeline.
4. Review/edit concept prompts and motion notes.
5. Click `Send To Video Creator`.
6. In Video Builder, verify scenes, notes, prompts, audio, and SRT timing.
7. Generate images and videos.

### No Prompt Creator Workflow

Use this when you want to work directly in Video Builder.

1. Add audio.
2. Use Lyric Mapping to create or fill scenes.
3. Add Director Notes for still image direction.
4. Add Video Notes for motion direction.
5. Use Reference Builder if you need consistent characters or locations.
6. Run Gemma prompts and generate media.

## Common Problems

| Problem | What to check |
| --- | --- |
| No scene is editable | Select a scene from the left list or timeline |
| The Image/Video/Audio tabs are disabled | No scene is selected |
| Model dropdowns are empty | Install models, then restart ComfyUI |
| Gemma prompt buttons fail | Make sure the correct Gemma model and mmproj are selected |
| NanoBanana fails | Add the API key in the Nano B `Models` tab |
| Render All skips scenes | Scenes with selected videos may already be complete |
| Build Full Video replaced too much | Next time choose `Resume missing only` or a "keep prompts" option |
| Audio does not play | Check whether the project uses Global Audio or Scene Audio |
| Timing changed after import | Enable `Freeze SRT timing` on scenes you do not want changed |
| Image-to-video prompt looks wrong | Turn `Use image reference for I2V prompt?` on/off depending on whether the image should guide Gemma |
| Characters sing during instrumental sections | Mark the scene `Instrumental` or `B-roll / no lip-sync`, then save lyric mapping |
| Wrong singer performs a duet line | Open `Review Lyrics + Map Singers`, check both singers, then save |
| Lyric notes do not show on the timeline | Use `Show Lyric Notes`, then save the lyric review |
| Location dropdowns are empty | Add locations in Reference Builder and save it |
| Reference Builder auto map fails | Extract/add locations first, then auto map; reduce overly long location lists if needed |
| Prompt Creator data does not populate notes | Use `Send To Video Creator` or `Import Data From Prompt Creator`, then reload/import prompt files if needed |
| LM Studio is selected but not used | Vision Gemma still uses built-in GGUF; only text-only passes use LM Studio |
| Audio notification does not play | Click inside the browser once and check notification settings |
| Model path is wrong on Linux | Use forward slashes and make sure the model picker shows the exact model name |

## Screenshot Checklist

Already added:

- Full V9 Video Builder window
- ComfyUI Builder Node
- Welcome Window
- Menu Dropdown
- Load Project Window
- Left Scene List
- Timeline Controls
- Timeline With Scenes
- Scene Tab
- Image Tab Model Chooser
- ZImage Prompting
- Flux Reference Images
- Top bar buttons
- Video Notes lane on the timeline
- Nano B model settings
- Nano B image settings
- Nano B LLM Prompting settings
- Video Mode Chooser
- Video Prompting
- Audio Tab
- Lyric Mapping Step 1 window
- Create Scenes From Lyrics window
- Review Lyrics + Map Singers window
- Reference Builder with character and location mappings
- Prompt Options image/video groups
- Gemma Runner with LM Studio model dropdown
- Settings window with custom model root and audio notifications
- Build Full Video Options
- Final Video Ready Popup
- Download Models Window

