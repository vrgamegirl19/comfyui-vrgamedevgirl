# V7 Video Builder Guide

This guide is for someone opening the V7 Video Builder for the first time. It explains what each main area does, the usual workflow, and where to look when something is missing.

![Full V7 Video Builder Window](images/Full%20V7%20Video%20Builder%20Window.png)

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
- [Reference Builder](#reference-builder)
- [Builder Agent](#builder-agent)
- [Prompt Options](#prompt-options)
- [Gemma Runner](#gemma-runner)
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

![ComfyUI Builder Node](images/ComfyUI%20Builder%20Node.png)

When the builder opens, it may show a welcome window where you can create a new project or open an existing project.

![Welcome Window](images/Welcome%20Window.png)

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

![Menu Dropdown](images/Menu%20Dropdown.png)

![Load Project Window](images/Load%20Project%20Window.png)

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

![Left Scene List](images/Left%20Scene%20List.png)

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

![Timeline With Scenes](images/Timeline%20Scene%20Blocks.png)

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

![Scene Tab](images/Scene%20Tab.png)

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

![Image Tab Model Chooser](images/Image%20Tab%20Model%20Chooser.png)

![ZImage Prompting](images/ZImage%20Prompting.png)

![Flux Reference Images](images/Flux%20Reference%20Images.png)

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

![Video Mode Chooser](images/Video%20Mode%20Chooser.png)

![Video Prompting](images/Video%20Prompting.png)

## Audio Tab

The `Audio` tab controls audio for the selected scene and the overall timeline.

Sections:

| Section | What it is for |
| --- | --- |
| `Scene Audio` | Add or manage audio attached to the selected scene |
| `Timeline Audio` | Add or manage the global project audio |

Use `Scene Audio` for scene-specific dialogue or clips. Use `Timeline Audio` for music-video timing.

![Audio Tab](images/Audio%20Tab%20Timeline%20Audio.png)

## Reference Builder

The `Reference Builder` button opens the `Reference Image Builder`. This is for projects where scenes need consistent characters, locations, or visual references across many generated images.

Reference Builder can feed references into `Flux/Klein` or `Nano B`, depending on the current image mode.

Use it when:

| Goal | What to use |
| --- | --- |
| Keep the same character across scenes | `Character References` |
| Keep locations consistent | `Location References` |
| Connect scenes to specific location images | `Scene Mapping` |
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
| `Save Reference Builder` | Saves the reference setup into the project |

Basic Reference Builder workflow:

1. Choose `Flux Klein` or `Nano B` in the `Image` tab.
2. Click `Reference Builder` in the top bar.
3. Turn on `Use subject reference` and/or `Use mapped location references`.
4. Add character and location references.
5. Map locations to scenes.
6. Click `Save Reference Builder`.
7. Generate images normally from the `Image` tab.

![Reference Builder Mapping](images/Reference%20Builder%20Mapping.png)

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
| `Purpose: Walkthrough` | Helps a new user step through setup |
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
3. Set `Purpose` to `Walkthrough`.
4. Ask what to do next.
5. Switch to `Scene work` when you want help with a selected scene.
6. Use `Auto: update fields` only when you are comfortable letting it make edits.

![Builder Agent Window](images/Builder%20Agent%20Window.png)

![Builder Agent Hints](images/Builder%20Agent%20Hints.png)

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

![Prompt Options Window](images/Prompt%20Options%20Window.png)

## Gemma Runner

The `Gemma Runner` button controls which runner is used for text-only Gemma steps.

Options:

| Runner | What it means |
| --- | --- |
| `builtin` | Uses the built-in local GGUF runner |
| `lm_studio` | Uses LM Studio's local server for text-only Gemma calls |

LM Studio is only for text-only Gemma steps. Vision/image-reference Gemma still uses the built-in GGUF runner.

LM Studio setup fields:

| Field | What to enter |
| --- | --- |
| `LM Studio base URL` | Usually `http://127.0.0.1:1234/v1` |
| `Available LM Studio models` | Click `Load LM Studio Models` to fill this |
| `LM Studio model name` | The loaded chat model name from LM Studio |
| `API key` | Usually blank for local LM Studio |
| `Test LM Studio` | Sends a tiny test prompt to confirm it works |

![Gemma Runner Window](images/Gemma%20Runner%20Window.png)

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

![Build Full Video Options](images/Build%20Full%20Video%20Options.png)

When the finished video is stitched, the builder shows a `Final Video Ready` popup. Use `Open Video` to preview it.

![Final Video Ready Popup](images/Final%20Video%20Ready%20Popup.png)

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

![Prompt Creator Import Buttons](images/Prompt%20Creator%20Import%20Buttons.png)

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

![Download Models Window](images/Download%20Models%20Window.png)

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
