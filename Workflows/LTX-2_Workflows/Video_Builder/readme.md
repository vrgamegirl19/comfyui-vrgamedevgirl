
# V7 Video Builder Guide

This guide is for someone opening the V7 Video Builder for the first time. It explains what each main area does, the usual workflow, and where to look when something is missing.

<screenshot of the full V7 Video Builder window goes here>

## Table of Contents

- [What the V7 Video Builder Does](#what-the-v7-video-builder-does)
- [Opening the Builder](#opening-the-builder)
- [The Main Layout](#the-main-layout)
- [Starting or Loading a Project](#starting-or-loading-a-project)
- [Adding Audio and SRT Timing](#adding-audio-and-srt-timing)
- [Working With Scenes](#working-with-scenes)
- [Using the Timeline](#using-the-timeline)
- [Scene Tab](#scene-tab)
- [Image Tab](#image-tab)
- [Video Tab](#video-tab)
- [Audio Tab](#audio-tab)
- [Batch Buttons and Full Builds](#batch-buttons-and-full-builds)
- [Prompt Creator Import](#prompt-creator-import)
- [Models and Downloads](#models-and-downloads)
- [Saving Projects](#saving-projects)
- [Recommended Beginner Workflow](#recommended-beginner-workflow)
- [Common Problems](#common-problems)
- [Screenshot Checklist](#screenshot-checklist)

## What the V7 Video Builder Does

V7 Video Builder is a scene-by-scene video creation UI inside ComfyUI. It helps you build a project from audio, SRT timing, scene notes, prompts, images, video clips, and final stitching.

The basic idea is:

1. Create or load a project.
2. Add global audio or per-scene audio.
3. Add scenes manually, from SRT, or from Prompt Creator data.
4. Write or generate image prompts.
5. Generate or import images.
6. Write or generate video prompts.
7. Render scene videos.
8. Stitch the final video.

## Opening the Builder

Add the node named `VRGDG Music Video Builder UI` in ComfyUI.

When the builder opens, it may show a welcome window where you can create a new project or open an existing project.

<screenshot of the ComfyUI node and open builder button goes here>

<screenshot of the Welcome to Video Creator window goes here>

## The Main Layout

The builder has five main areas:

| Area | What it is for |
| --- | --- |
| Top bar | Project menu, save, Prompt Creator tools, utilities, stop button, model downloads, fullscreen |
| Left panel | Scene list |
| Center preview | Selected image or video preview |
| Right panel | Scene settings, split into Scene, Image, Video, and Audio tabs |
| Bottom timeline | Timing, playback, scene blocks, inserts, notes, beat markers, and selected media tools |

<screenshot of the layout with labels for top bar, scene list, preview, inspector, and timeline goes here>

## Starting or Loading a Project

Use the `Menu` button in the top-left area of the builder.

Important project options:

| Button | Use it when |
| --- | --- |
| `New Project` | You want to start a fresh V7 builder project |
| `Load Project` | You want to open an existing project folder |
| `Load Last Project` | You want to return to the most recent project |
| `Save Project As` | You want to duplicate the current project into a new folder |
| `Quick Save` | You want to save the current project state |
| `Auto save` | You want the builder to save changes while you work |

Projects are saved under the ComfyUI output folder. A builder project contains the session JSON, SRT, generated images, scene videos, prompt files, reference images, and copied audio assets.

<screenshot of the Menu dropdown goes here>

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

<screenshot of the Audio tab with Timeline Audio expanded goes here>

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

<screenshot of the left scene list with one scene selected goes here>

## Using the Timeline

The timeline is at the bottom of the builder. It shows the global timing of the project, the playhead, scene blocks, notes, waveform, inserts, selected media tools, and beat markers.

Timeline controls:

| Control | What it does |
| --- | --- |
| `Bulk Segments` | Create many manual scenes from pasted durations or start/end times |
| `+ Scene Note` | Show editable note boxes below scenes |
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

<screenshot of the timeline controls goes here>

<screenshot of scene blocks on the timeline goes here>

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

<screenshot of the Scene tab goes here>

## Image Tab

The `Image` tab creates or manages the selected scene image. Pick the image model at the top of the tab.

Image modes:

| Mode | Use it for |
| --- | --- |
| `ZImage` | Main local image generation workflow |
| `Flux Klein` | Flux/Klein image generation, including reference image ingredients |
| `Nano B` | NanoBanana image generation with reference images and API key |
| `Ernie` | Ernie Image generation |
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

<screenshot of the Image tab model chooser goes here>

<screenshot of ZImage LLM Prompting with Gemma T2I and T2I prompt goes here>

<screenshot of Flux/Klein reference image ingredients goes here>

## Video Tab

The `Video` tab creates the selected scene video. At the top, choose between:

| Mode | What it does |
| --- | --- |
| `Image to Video` | Uses the selected scene image plus a video prompt |
| `Text to Video` | Uses a text prompt directly, without requiring a scene image |

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

<screenshot of the Video tab mode chooser goes here>

<screenshot of Video LLM Prompting with motion notes and Gemma I2V goes here>

## Audio Tab

The `Audio` tab controls audio for the selected scene and the overall timeline.

Sections:

| Section | What it is for |
| --- | --- |
| `Scene Audio` | Add or manage audio attached to the selected scene |
| `Timeline Audio` | Add or manage the global project audio |

Use `Scene Audio` for scene-specific dialogue or clips. Use `Timeline Audio` for music-video timing.

<screenshot of the Audio tab showing Scene Audio and Timeline Audio goes here>

## Batch Buttons and Full Builds

The `Menu` contains batch tools that can work across many scenes.

| Button | What it does |
| --- | --- |
| `Gemma T2I All` | Creates image prompts for multiple scenes |
| `Gemma Video All` | Creates I2V/T2V prompts for multiple scenes |
| `Image All` | Runs image generation across scenes |
| `Render All` | Renders missing scene videos and stitches when possible |
| `Stitch Preview` | Stitches selected/existing scene videos into a preview |
| `Build Full Video` | Runs the larger pipeline from prompts/images/videos through final stitch |
| `Remake Mode` | Helps rerun or rebuild outputs |
| `Stop` | Stops the current workflow run |

When prompted, choose the safest option first:

| Option | Best for |
| --- | --- |
| Resume missing only | Continue without replacing finished work |
| Keep prompts, redo images/videos | Make new media but preserve prompt work |
| Redo prompts and images/videos | Start fresh for the selected stage |

<screenshot of Build Full Video confirmation options goes here>

## Prompt Creator Import

V7 can import data from the Prompt Creator.

Useful buttons:

| Button | What it does |
| --- | --- |
| `Prompt Creator` | Opens the Prompt Creator panel |
| `Import Data From Prompt Creator` | Copies prompt creator outputs into the current builder project |
| `Prompt Options` | Opens prompt-related settings/options |
| `Gemma Runner` | Opens Gemma runner tools |
| `Agent` | Opens the builder assistant/agent |
| `Reference Builder` | Helps build reference material for image workflows |

Imported data can include audio, SRT, concept prompts, lyric segments, motion notes, theme/style text, story idea text, and subject/scene text.

<screenshot of Prompt Creator import buttons in the top bar goes here>

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

<screenshot of the Download Models window goes here>

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

## Screenshot Checklist

Add screenshots in these places when you are ready:

- Full V7 Video Builder window
- ComfyUI node used to open the builder
- Welcome window
- Menu dropdown
- Top bar buttons
- Left scene list
- Timeline controls
- Scene tab
- Image tab model chooser
- ZImage prompt workflow
- Flux/Klein reference images
- Video tab mode chooser
- Video prompt workflow
- Audio tab
- Build Full Video options
- Download Models window

