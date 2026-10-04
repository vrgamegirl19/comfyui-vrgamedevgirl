# vrgdg-h3-film: a Claude skill for making AI short films with the VRGDG Video Builder

This skill lets Claude (in Claude Code) make a complete short film on your own PC, using your own ComfyUI. Claude
writes the story and dialogue, designs the characters and locations, renders every scene with MiniMax H3 (including
the built-in voices and sound), scores it, checks its own work, re-renders what's wrong, and edits the final film.
You give it one prompt, then chat with it to fix anything you'd like changed.

It drives ComfyUI through the VRGDG Video Builder's own routes. It never edits, updates or deletes your existing
nodes, models or ComfyUI files, and everything it makes goes in a project folder you choose. It can install missing
pieces, but only after showing you the list and getting your OK (see "Missing something?").

## What you need

- **[Claude Code](https://claude.com/claude-code):** the desktop app, the VS Code extension or the terminal.
- **ComfyUI, recent:** the skill was tested on ComfyUI 0.37.0. It needs ComfyUI's built-in MiniMax H3 and MiniMax
  Music 3 nodes, so update ComfyUI if they're missing.
- **GPU:** 24–32 GB of VRAM. Each 8-second scene takes about 7–10 minutes on a 32 GB card.
- **Disk:** about 45 GB for the required H3 models, about 21 GB more for Z-Image, and about 14 GB for Music 3.

### Custom nodes (all required)

| Pack | Why it's needed |
|---|---|
| [comfyui-vrgamedevgirl](https://github.com/vrgamegirl19/comfyui-vrgamedevgirl) | the Video Builder, its routes, the H3 helper nodes and the YuE2 nodes |
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | video and audio loading and saving in the Builder workflows |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | the H3 model loader and image resize nodes |
| [comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) | audio/video latent separation in the H3 2-pass workflow |
| [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | the latent-upscaler backend used by the Builder's 2-pass upscale |
| [ComfyUI-EulerDiscreteScheduler](https://github.com/erosDiffusion/ComfyUI-EulerDiscreteScheduler) | the scheduler in the Builder's Z-Image 2-pass workflow |

### Models

Put each file in the `ComfyUI/models/` subfolder shown. Any folder that ComfyUI already reads also works, including
paths in `extra_model_paths.yaml` and the Video Builder's custom model folder.

| | Model | Folder | Size |
|---|---|---|---|
| **Required** (video, voices and sound) | [MiniMax H3 Ref2VA diffusion model](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | `diffusion_models` | 21.0 GB |
| | [MiniMax H3 Qwen3-VL text encoder](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | `text_encoders` | 15.7 GB |
| | [MiniMax H3 video VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) | `vae` | 5.2 GB |
| | [MiniMax H3 audio VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `vae` | 0.6 GB |
| | [MiniMax H3 latent upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors) | `latent_upscale_models` | 0.7 GB |
| | [MiniMax H3 turbo LoRA, 4-step 768p v1.1](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors) | `loras` | 2.0 GB |
| **Recommended** (reference images) | [Z-Image Turbo](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors) | `diffusion_models` | 12.3 GB |
| | [Qwen 3 4B text encoder](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) | `text_encoders` | 8.0 GB |
| | [Z-Image VAE](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors) | `vae` | 0.3 GB |
| **Optional** (music) | [MiniMax Music 3 DiT](https://huggingface.co/Comfy-Org/MiniMax-Music-3/resolve/main/diffusion_models/minimax_music3_dit_fp16.safetensors) | `diffusion_models` | 4.9 GB |
| | [MiniMax Music 3 text encoder](https://huggingface.co/Comfy-Org/MiniMax-Music-3/resolve/main/text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors) | `text_encoders` | 9.2 GB |
| | [MiniMax Music 3 audio VAE](https://huggingface.co/Comfy-Org/MiniMax-Music-3/resolve/main/vae/minimax_music3_dav.safetensors) | `vae` | 0.2 GB |

**Notes on the models:**
- **Use this turbo LoRA:** other H3 turbo LoRAs leaked the reference images into the video in testing.
- **Music alternatives:** instead of MiniMax Music 3 you can use **YuE2** (install it with the VRGDG *YuE2 Installer*
  node; see the VRGDG YuE2 README), or your own music files.
- **Optional dialogue checks:** install `openai-whisper` in ComfyUI's Python so Claude can check every line of
  dialogue. Without it, Claude checks only the pictures.

## Install the skill

1. Copy the whole `vrgdg-h3-film` folder into your Claude skills folder:
   - **Windows:** `C:\Users\<you>\.claude\skills\vrgdg-h3-film\`
   - **Mac or Linux:** `~/.claude/skills/vrgdg-h3-film/`

   `SKILL.md` must end up directly inside that folder. The skill ships with the VRGDG nodes, so if you installed them
   it's already at `ComfyUI/custom_nodes/comfyui-vrgamedevgirl/skills/vrgdg-h3-film/`. Copy it again after updating
   the nodes to get the latest skill fixes.
2. Restart Claude Code, or start a new session.
3. Start ComfyUI the usual way (for example `run_nvidia_gpu.bat`) and leave its console window open. You don't need
   to open ComfyUI in the browser; Claude talks to the server directly.

## Missing something? Claude can install it for you

When Claude starts, it checks your setup. If a model or custom node is missing, it shows you a report: what's
missing, the download sizes, and where each file will go. **Nothing is installed until you say yes.** Then:

- **Models** are downloaded straight from the links above into `ComfyUI/models/...`. Downloads resume if they're
  interrupted.
- **Custom nodes** are cloned into `ComfyUI/custom_nodes/`, and their requirements are installed into ComfyUI's Python.
  You restart ComfyUI afterwards. If you'd rather install nodes yourself, use ComfyUI-Manager or the links above.
- **Whisper** for the dialogue checks can be added the same way.

It only adds what's missing. Existing files and node folders are never overwritten, updated or deleted. You can
also run the check yourself:

```
python "%USERPROFILE%\.claude\skills\vrgdg-h3-film\scripts\install_missing.py" --comfy-root "D:\ComfyUI_windows_portable"
```

## What to tell Claude

- **Where ComfyUI is installed,** for example `D:\ComfyUI_windows_portable`.
- **Where YuE2 is installed,** only if you use it.

You **don't** need to say where your models are. Claude asks your running ComfyUI for its model list, so anything
that shows up in ComfyUI's model dropdowns is found automatically. If you renamed a model file and it isn't picked
up, tell Claude its name.

## Use it

Open Claude Code in any folder and ask, for example:

> Make me a 3-minute short film with my VRGDG Video Builder. ComfyUI is in `D:\ComfyUI_windows_portable`. Come up
> with your own story, adults only. Put the project in `D:\Films\MyFirstFilm`.

Claude loads the skill and works through these steps:

1. Checks your setup and offers to install anything missing.
2. Creates the project folder, writes the story and screenplay, and makes the reference images.
3. Renders a few test scenes, checks them, and then renders the rest.
4. Makes the score, checks every cut between scenes, and assembles `final/<TITLE>.mp4`.

A 3-minute film takes roughly 3–5 hours of rendering. You can leave it running, then watch the result and tell
Claude what to fix ("in scene 12 she should look at him, not the camera").

**Optional details to tell Claude:**
- **YuE2 for music:** "YuE2 is installed in `E:\Yue2`."
- **Your own music:** "Use my song `D:\music\theme.mp3` as the score."
- **A whole song for the credits:** "Make an end-credits song with YuE2."

## What's in the folder

| File | What it does |
|---|---|
| `SKILL.md` | the instructions Claude follows: the pipeline, plus every prompt and voice rule learned the hard way |
| `requirements.json` | the list of required custom nodes and models with their download links |
| `scripts/install_missing.py` | reports what's missing and, only with your OK, downloads or clones it |
| `scripts/check_env.py` | inspects your ComfyUI, picks models, writes `env.json` |
| `scripts/new_project.py` | creates a project with `settings.json`, starter screenplay, image specs and score files |
| `scripts/gen_refs.py` | Z-Image reference images through the Builder |
| `scripts/build_prompts.py` | turns the screenplay into MiniMax H3 prompts, with a lint that catches common mistakes |
| `scripts/render_scenes.py` | renders scenes with the Builder's H3 2-pass route, using H3's own voices |
| `scripts/qa.py` | dialogue check (Whisper), voice and pitch check, clipped-ending check, review sheets |
| `scripts/cut_sheet.py` | the last frame and first frame of every cut between scenes |
| `scripts/retrim.py`, `continuity_frame.py`, `adopt_render.py` | fix-up tools for trims, scene-to-scene continuity, and interrupted renders |
| `scripts/gen_music.py` | score cues with MiniMax Music 3 or YuE2 |
| `scripts/assemble.py` | Builder stitch, ducked score, title and end cards, loudness |
| `scripts/sync_check.py` | checks every scene's audio and picture sync in the finished film |
| `scripts/social_kit.py` | poster reference images and the film's facts for the social media posts |
| `templates/` | an example two-scene screenplay, image specs and score |

## Troubleshooting

- **A scene says "interrupted":** something cancelled the running job, such as Cancel in the ComfyUI web page. Ask
  Claude to re-run that scene.
- **The setup check reports NOT READY:** the message says what's missing. Ask Claude to install it, or update the
  VRGDG nodes or ComfyUI.
- **Renders take much longer than 10 minutes:** close other GPU programs. The skill frees ComfyUI's memory after
  each scene.
- **Last words of a line cut off:** the skill pads every scene. Each take is kept to its last frame, the last line
  gets a trailing breath, and prompts ask for a silent beat after the last word. The lint warns when a line is too
  long for its scene; shorten the line or lengthen the scene.
- **Audio out of sync:** assemble.py builds the dialogue track from each clip's own audio and prints a sync check at
  the end. Over 40 ms means a clip's audio is wrong. If you edit a clip's audio by hand, keep it at 32 kHz like the H3
  clips (assemble.py converts mismatched clips anyway).
- **Licences:** check each model's licence before commercial use. For example, the YuE2 weights are CC BY-NC 4.0.
