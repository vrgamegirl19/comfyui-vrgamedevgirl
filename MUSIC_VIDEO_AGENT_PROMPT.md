# Music Video Agent Prompt

Give an AI agent this whole file after filling in **Part 1**. The agent builds a music video through the
VRGDG MCP server, doing the same steps as the Video Builder, in the same order.

The agent must have the `vrgdg` MCP server connected (`python -m mcp_server` from this repository, with ComfyUI
running at `http://127.0.0.1:8188`). If the tools below are missing, stop and tell the user.

---

## Part 1 - Fill this in

```yaml
project_name:            # e.g. "Busting a Nut"
audio_file:              # full path to the song, e.g. C:\Users\me\Music\song.mp3
lyrics: |                # paste the full lyrics, with section tags like [Verse 1] if you have them

character:
  name:                  # e.g. Darrel
  image_file:            # full path to the character image
  notes:                 # optional, anything the agent must know about the character
location_style_theme:    # where and what look, e.g. "Los Angeles nightlife, rooftop lounges, neon, night time"
story_idea:              # one or two sentences the agent expands from. Leave empty to let the agent write one
                         # from the lyrics and theme.
continue_scenes: false   # true makes every scene after the first continue the previous scene's take (see step 16)
video:
  style: cinematic_realism           # saved as the video style key
  camera_flow: intimate_closeups
  camera_motion_speed: 7             # 0-10
  character_motion_speed: 5          # 0-10
scene_length_seconds:
  min: 3.5
  max: 10
```

---

## Part 2 - Rules (read before you start)

1. **Never change the LLM.** All writing (character description, locations, story arc, story brief, scene beats,
   MiniMax prompts) is done by the model that is already loaded in LM Studio. Never load, unload or switch models.
   Call `llm_active` first. If it reports no loaded model, stop and ask the user to load one.
2. **The agent supplies only the story idea.** Everything else is written by the LLM through the tools below.
   Do not write scene beats or prompts yourself.
3. **MiniMax H3 only.** Reference to Video, plain **2 Pass** (`render_pass: two_pass`), final size **1920x1088**.
   Never use 2 Pass Advanced.
4. **One GPU job at a time.** Rendering and LLM jobs run as background jobs. Start one, then call `job_wait`
   in a loop until it is `succeeded` or `failed`. Do not start a second render while one is running.
5. **Do not edit files by hand.** Use the tools. They save the same session the Video Builder reads, so the
   project opens in the Video Builder exactly as if it had been built there.
6. **If a tool fails,** read the error and `next_steps`, fix the cause once, and retry. If it fails again, stop
   and report the exact error. Do not work around it with other tools.
7. Report after each step in one or two lines: what ran, what the result was, any warning.

---

## Part 3 - Steps

Use the project id returned by `project_create` in every later call.

### A. Project
1. `system_health`, then `llm_active`. Stop if either reports a problem.
2. `project_create` with `project_name`.
3. `project_update_settings` with
   `{"project": {"video_engine": "minimax_h3"}, "minimax_h3": {"video_mode": "reference_to_video", "render_pass": "two_pass", "resolution_preset": "2k"}}`.
4. `audio_attach` with `audio_file`.
5. `lyrics_set` with the pasted lyrics.

### B. Line Mapping (scenes)
6. `timeline_from_lines` with `min_scene_seconds` and `max_scene_seconds` from Part 1. Wait for the job.
   It times the lyrics against the song, makes one scene per lyric line, merges scenes shorter than the minimum and
   splits scenes longer than the maximum. Check the result: every scene length is inside the range, and the
   lyrics appear on the timeline.

### C. Reference Builder
7. `reference_upsert` with `kind: "subjects"`, a short `reference_id` (the character name in lower case) and
   `payload: {"name": ..., "reference_type": "character", "image": {"path": <image_file>, "name": <file name>}}`.
8. `reference_describe` for that character (Gemma Describe). Wait for the job.
9. `reference_extract_locations` with `style_theme` = `location_style_theme` (LM Extract). Wait for the job.
   No location images are needed.
10. `reference_assign_scenes` with `character_pattern: "blocks"`, `character_block_size: 1000`,
    `location_pattern: "blocks"`, `location_block_size: 4`, `replace_existing: true`.
    This puts the character on every scene and repeats each location for 4 scenes.
    A scene's mapping holds one location. To give a MiniMax scene several references (more than one location, or a
    different order), use `GET /projects/{pid}/scenes/{sid}/minimax-references`: `available` lists every character,
    extra, location and ingredients sheet with its `key`, and `selected` is the order MiniMax receives. Then
    `PUT` the same path with `{"keys": ["subject:ava", "location:roof", "location:alley"]}`, or `{"automatic": true}`
    to follow the mappings again. At most 9 images are sent (8 chosen when the scene image is Image 1).

### D. Storyboard (story layer)
11. `story_settings` with
    `defaults: {"video_style": <video.style>, "camera_flow": <video.camera_flow>, "camera_motion_speed": ..., "character_motion_speed": ...}`
    and `story: {"overall_story_idea": <story_idea>}`.
    If `story_idea` is empty, write one or two sentences from the lyrics and `location_style_theme`.
12. `story_create` with `step: "arc"`. Wait for the job.
13. `story_create` with `step: "brief"`. Wait for the job.
14. `story_create` with `step: "beats"` (all scenes in one call, no `limit`). Wait for the job.

### E. MiniMax prompts
15. `minimax_prompts`. Only scenes without a prompt are written. Wait for the job. If `failed` is not 0, read
    `failures`, fix the cause (usually a scene with no mapped character) and run it again.

### F. Render and stitch
15b. Only when `continue_scenes` is true: `project_update_settings` with
    `{"minimax_h3": {"continuity_mode": "latent_continuation_masked", "latent_context_frames": 39, "location_transition_preset": "masked", "continuity_prompt_from_last_frame": true}}`
    before any prompt is written. With `continuity_prompt_from_last_frame` each continued scene's prompt is written again from the
    previous scene's real final frame right before it renders (the loaded LLM must be able to read images, otherwise it uses the
    previous scene's last shot). Scene 1 renders normally. Every later scene starts from the previous scene's last moments, so its
    prompt is the next moment of the same take: carry on for the first part of the scene, then make one smooth movement, never a cut.
    `minimax_prompts` already writes it that way. To direct a scene yourself, `scene_update` with
    `patch: {"minimax_h3_continuation_direction": "he turns to the camera, then sits down"}` (the action only, no lyrics). The direction starts 0.5 s into the scene; add `"minimax_h3_continuation_start_seconds": 1.2` to start later (at most half the scene). The prompt writer is told the scene length and must finish the action before it ends. Then
    `minimax_prompts` with `scene_ids: [that scene]` and `replace_existing: true`. The scenes must be rendered in timeline order with the same render pass and
    resolution. A scene whose predecessor has no saved video fails with `PREDECESSOR_MISSING`, render the predecessor first.
16. `video_render` for every scene in timeline order, one at a time, with `params: {"mode": "minimax_h3"}`.
    Wait for each job before starting the next. If a scene fails, retry it once, then continue and list it in
    your final report.
17. `scene_get` on each scene as it finishes: `video_path` is set and the clip length matches the scene length
    within about 0.1 second.
18. `stitch_final`. Wait for the job, then report the `final_video_path`, the duration, and any scene that failed.

---

## Part 4 - What "done" means

- Every scene has a story beat, a MiniMax prompt and a rendered clip.
- `FINAL_VIDEO.mp4` exists in the project folder and is about as long as the song.
- The project opens in the Video Builder with the lyrics lane, scene pictures and prompts visible.
