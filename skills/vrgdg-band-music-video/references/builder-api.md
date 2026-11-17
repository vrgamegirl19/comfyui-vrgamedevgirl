# VRGDG Video Builder Agent API: what this skill uses, and the quirks

**Base URL:** `http://127.0.0.1:8188/vrgdg/api/v1`.

**The full list:** `Api_Endpoints.md` in the node pack is generated from the router, so trust it over this page, and
read it once per session. `scripts/common.py` (class `Api`) wraps every call used below.

## Calls in pipeline order

| Step | Call | Notes |
|---|---|---|
| Health | `GET /health`, `GET /queue`, ComfyUI `GET :8188/queue` | before every GPU job, so you never queue behind the user's render |
| Create | `POST /projects` `{"name"}` | returns `project_id` and `project_folder` (`ComfyUI/output/<name>`) |
| Settings | `PATCH /projects/{pid}/settings` `{"project": {...}, "minimax_h3": {...}}` | key names: `GET /settings/minimax-h3/schema` |
| Audio | `PUT /projects/{pid}/audio` `{"audio_path"}` | copies the file to `project_audio/project_audio.wav`, finds beats |
| Lyrics | `PUT /projects/{pid}/lyrics` `{"srt_text"}` | full-track times |
| Timeline | `POST /projects/{pid}/timeline/bulk` `{"mode": "ranges", "action": "replace", "text": "0.00 --> 8.00 label\n...", "clear_media": true}` | replaces every scene; never on a rendered project without the user's OK |
| Cards | `PUT /projects/{pid}/references/subjects/{rid}` and `/locations/{rid}` `{"name", "description", "reference_type", "image": {"path", "name"}}` | a PUT **resets fields it leaves out**: always send the whole card |
| Scene fields | `PATCH /projects/{pid}/scenes/{n}` | `subject_ids`, `location_id`, `performance_mode`, `lyric_no_lip_sync`, `include_microphone`, `story_beat`, `use_scene_minimax_h3_settings`, `minimax_h3_settings`, `minimax_h3_continuation_start_seconds` |
| Pictures | `PUT /projects/{pid}/scenes/{n}/minimax-references` `{"keys": ["subject:singer", ..., "location:kitchen"]}` | the order is the `<Picture N>` order; at most 9 |
| Prompt | `PUT /projects/{pid}/scenes/{n}/prompts/minimax_h3_prompt` `{"prompt", "origin": "manual"}` | rendered exactly as written |
| Render | `POST /projects/{pid}/scenes/{n}/video/render` `{"mode": "minimax_h3", "render_phase": "draft"\|"finalize"\|"full"[, "audio_path": "<track>"]}` | a job; `draft` = pass 1 only (latent saved, scene holds a draft), `finalize` = pass 2 from the scene's **latest** draft; `audio_path` overrides the project song for this render only (vocal-free, vocal-gated, or song + silence) |
| Add a scene | `POST /projects/{pid}/scenes` `{"position": "append", "duration": 5.0, "label"}` | append starts at the last scene's end (an ending after the song); before/after a `ref_scene_id` renumbers files |
| Memory | ComfyUI `POST /vrgdg/resource-monitor/clear-memory`, `GET /vrgdg/resource-monitor`; ComfyUI `POST /free {"unload_models": true, "free_memory": true}` | before every render (render_guarded.py does it) |
| Job | `GET /jobs/{id}`, `GET /jobs/{id}/log?since=n` | states: queued, running, succeeded, failed, cancelled, interrupted |
| Scene | `GET /projects/{pid}/scenes/{n}` | `rendered_video.url` has the absolute path in its `path=` parameter |
| Stitch | `POST /projects/{pid}/stitch` `{"output_prefix"}` | a job; frame-accurate at 24 fps; default audio = the project song |
| Latents | `GET /projects/{pid}/latents` | per-scene latent file, `dirty` flag |
| Copy | `POST /projects/{pid}/duplicate` | see the quirks below |
| Continuation brief | `GET /projects/{pid}/scenes/{n}/prompts/context?kind=minimax` | the `continuation` block: hold seconds and the Builder's rules |

## Settings that matter

**DON'T LOOK NOW (2026-10-25, the current reference):** `Minimax-h3_Singularity_ref2va_v2_int8.safetensors`,
`reference_to_video`, `two_pass`, `2k`, pass 1 25 steps, pass 2 `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16
.safetensors` at 4 steps and denoise 0.35, **warmup_frames 0 and cooldown_frames 0**, seed 69 on every pass,
feedforward on both passes only for scenes over 8 s, `ref_image_size` `max` (`match` is the out-of-memory fallback).
Re-rolls change the prompt first, the seed (1001, 2002, 3003) only when the wording is right.

**Memory:** start ComfyUI WITHOUT `--disable-dynamic-vram` / `--disable-smart-memory` (the log must say "DynamicVRAM
support detected and enabled"); with them, pass 2 ran out of memory in the latent upscaler on 10–12 s scenes. Even
with them off it happens at random after long sessions: retry the finalize (up to 4 times), then `ref_image_size`
`match`, then a re-draft at `match` (drive_queue.py does this). A PC restart clears what a memory clear can't.

**FUCK THE STATIC (older):**

- **Model:** `diffusion_model_name`: `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors` ("Singularity").
- **Mode:** `video_mode` `reference_to_video`, `render_pass` `two_pass`, `resolution_preset` `2k` (1920×1088),
  `audio_mode` `input_audio`.
- **Pass 1:** `two_pass_pass1_steps` 25.
- **Pass 2:** `two_pass_lora_name` `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` (the "8 step 768
  LoRA"), `two_pass_lora_preset` 8, `two_pass_pass2_steps` 4, `two_pass_pass2_denoise` 0.35.
- **Warm-up and cool-down:** `warmup_frames` 5, `cooldown_frames` 5. Same as the user's "the riot of one" project.
- **Acceleration:** the user wanted feedforward on for **both passes, only on scenes over 8 s**, and no other
  acceleration (te_speed, block_sparse_attention, fast_vae_decode all off). Feedforward is per scene: see the quirks.
- **Copy the user's settings:** another of their projects shows their preferences:
  `GET /projects/<their project>/settings`.

**Check what really runs:** while a scene renders, read the graph from ComfyUI `GET /queue` → `queue_running[0][2]`.
Node inputs show the model loader name, the BasicScheduler steps and denoise, the LoRA loader name, the pass-2 steps
and denoise primitives, and the image paths sent. Tonight that confirmed 25 steps, the 8-step LoRA, 4 steps and 0.35.

## Quirks (each one cost time tonight)

1. **Project ids with spaces:** `Fuck the Static` must be URL-encoded in paths. `common.Api` quotes `{pid}`.
   vb.py from the vrgdg-builder-api skill does not, and Python's urllib refuses the URL.
2. **Git Bash rewrites paths that start with `/`** (`/projects` → `C:/Program Files/Git/projects`, then a 404). Set
   `MSYS_NO_PATHCONV=1` when passing API paths as arguments, or call the API from Python, as these scripts do.
3. **The portable Python ignores the current folder:** `import api` fails unless the script adds its own folder to
   `sys.path` (these scripts do). Pass Python Windows-style paths (`Z:/...`), not Git Bash `/z/...` paths.
4. **Per-scene settings replace the project's for that scene.** `use_scene_minimax_h3_settings: true` plus
   `minimax_h3_settings` must carry the **full** settings: the project settings with your changes, such as
   feedforward or continuation. build_prompts.py builds them from `GET /settings`. Re-applying a scene with partial
   settings would silently drop the model, LoRA or steps.
5. **Duplicate creates an empty project:** `POST /duplicate` builds the copy's session from a `session` field and
   writes an empty session without it, and it attaches no audio. `duplicate_project.py` sends the source's saved
   session (read-only) in `options.session`, then `PUT /audio` on the copy. The whole folder is copied, including
   videos, latents and `_claude_work`. **Prompts inside the copy's `_claude_work` are a snapshot from copy time**:
   export them again from the copy's scenes if the user will read them there.
6. **LATENT_STALE:** a continued scene refuses to render when its predecessor's latent is flagged dirty. That happens
   when any earlier scene was re-rendered after it, even when the predecessor itself didn't change. Only a render
   clears the flag, and the session file must not be edited by hand. So render the predecessor again in the copy
   (that take can serve as an alternate too), then the continued scene.
7. **409 REVISION_CONFLICT:** the open Builder page saved in between. `Api.mutate` re-reads the revision and retries.
   Tell the user to **reload the project in the Builder** after your changes, or their next save can overwrite them.
8. **Connection reset:** a single `ConnectionResetError` can happen while ComfyUI is busy; retry, then check
   `/health`. `Api.call` retries twice.
9. **The `rendered_video` object** carries `path_rel` (relative to the project) and a `url` with the absolute path.
   Use `common.video_path()`.
10. **The Builder's own prompt writer** (`POST /minimax-prompts`) uses the loaded LLM. This skill writes prompts
    itself, so no LLM is needed. Never load, unload or switch the LM Studio model.

11. **Finalize only takes the scene's latest draft.** The API has no draft id parameter yet. To finalize an older
    take, re-draft it with its exact saved prompt and the same seed: it comes back pixel-identical (DLN scene 23,
    PSNR inf against the chosen take), then finalize. Save every attempt's prompt (`prompts/scene_NN_dK.txt`) and
    keep `logs/draft_index.json` (render_guarded.py). The scene GET does not return the draft; read
    `minimax_h3_draft` from the session file (read-only, `render_guarded.segment()`).
12. **A failed finalize can still have rendered.** "Removing existing scene video ... stayed locked" (the user had
    the draft open in a player) failed the job after pass 2 finished. The finished file is still in ComfyUI's
    output: `GET :8188/history/<prompt_id>` (prompt id from `GET /jobs/{id}` → `comfy`) lists
    `VRGDG_MiniMaxH3/<project>_<hash>/scene_NNNN/MiniMaxH3_scene_NNNN_stage2_finalized_00001-audio.mp4`. It has the
    render padding at the end (124 frames for a 120-frame scene): keep the first N frames, after checking them
    against the draft.
13. **Stitch `overlays` add one frame per insert,** so the picture after each insert runs a frame late (the result
    still reports the timeline frame count). Use `splice_scene.py` on a plain stitch instead; it is frame-exact.
14. **Latent Continuation Masked** (`continuity_mode: latent_continuation_masked`, `latent_context_frames: 39`,
    `continuity_prompt_from_last_frame: false` to keep your prompt) on a scene's own settings (`set_h3.py`). Warm-up
    frames are forced to 0 in masked mode. **Finalize the predecessor first:** pass 2 re-applies the head from the
    predecessor's pass-2 latent. The scene records `minimax_h3_continuity_mode_used` after the render.
15. **Image to Video 2 Pass needs an FL2V LoRA:** with a ref2v LoRA the render fails with "Image to Video 2 Pass
    requires an FL2V/I2V Turbo LoRA". DLN used `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`
    with `minimax_h3_fl2va_pruned_int8_convrot.safetensors`. First/last frame: PATCH the scene
    `approved_image_path` (first), `minimax_h3_i2v_frame_mode: "flf"`, `first_last_frame_end_image_path` (last),
    plus scene settings with `video_mode: image_to_video`. Masked continuation is not available in I2V.
16. **Audio past the song's end:** a scene after the last sung second needs a render track that long. Append
    silence to the song (`song_plus_5s_silence.wav`) and pass it as the render's `audio_path`, rather than changing
    the project audio.
17. **Project copies are named by id:** the copy "<name>" lives in `ComfyUI/output/<name>`. Do experiments (single
    shots to splice, I2V clips, extra scenes) there, never in the user's project.
18. **Never edit a chain or queue script that is already running** (bash reads it as it goes; PLM lost a whole
    re-roll queue that way). Start a new one.

## Rules for driving it

- **One GPU job at a time**, and only when ComfyUI is idle. render_queue.py waits.
- **Only the API writes project files.** Never edit `vrgdg_builder_session.json`; reading it is fine. Your own files
  live in the work folder (`<project>/_claude_work`).
- **Never start, stop or restart ComfyUI,** and never cancel a job you didn't start.
- **Experiments go in a duplicate** (continuation alternates), never in the project the user is using. Ask which
  project they treat as "theirs". Tonight they adopted the alternates copy, and later fixes went there.
