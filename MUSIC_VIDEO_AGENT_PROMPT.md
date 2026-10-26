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
                         # from the lyrics and theme. Required in Story Mode.
story_mode: false        # true runs the Story Mode path (Part 3b): the story is a book, locations come from it
story_text: |            # Story Mode only, optional: a finished story to use word for word instead of story_idea
scenes_per_spot: 3       # Story Mode only: fewest scenes in a row at one spot (e.g. one room)
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
8. **Know what this build can do.** The answer of `system_health` may carry a `features` block (the H3 Studio app does).
   Plan with it. Do not run a step it says is missing.
9. **Ask before you do more than was asked.** Ask first before you generate more than a few images, change a setting the user
   did not mention (for example `automatic_memory_cleanup`), start a GPU job the user did not ask for (a RefMod or the stems),
   or delete or overwrite anything. Try a new call on a copy made with `project_duplicate`, not on the user's project, and
   delete only what you made.
10. **The Video Builder may be open on the project.** It shows what you change as you change it: scene edits and finished renders
    appear at once, and a change to the settings, a card or the story reloads the project by itself. A scene field the user has
    typed and not saved is kept. The window waits, then reloads, only while the user is typing or has a render running in it
    (or has unsaved changes to the settings or cards), and it refuses to save its older copy meanwhile, so nothing you saved is
    overwritten. There is nothing to tell the user to reload.

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
11b. Optional story events at exact times: `timeline_note_create` with `start` and `end` seconds (leave `end` out for a
    moment) and the event in `note`, for example `{"start": 42, "end": 58, "label": "Reveal", "note": "She finds the letter"}`.
    The arc applies each note to the scenes it overlaps. Per-scene directions go on the scene card instead:
    `scene_update` with `timeline_note` (Director Notes), `i2v_notes` (Video Notes), `notes` (Planning Notes),
    camera, performance, audio and continuity fields; `scene_get` shows the whole card as `scene_card`. The open
    Video Builder and Storyboard show these edits.
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

## Part 3b - Story Mode steps (when `story_mode` is true)

Story Mode tells a story written as a book: chapters, each in one general setting (Home, The Night Train), split
into acts, with every scene at one spot inside the setting (Home - Kitchen). The lyrics set mood and emotion only;
they never decide an event or a place. Do steps 1-6 of Part 3, then:

7. `project_update_settings` with `{"project": {"video_type": "story"}}`.
8. Characters first: steps 7-8 of Part 3 (`reference_upsert`, `reference_describe`) for every main character.
9. `story_settings` with the video defaults from step 11 of Part 3 and
   `story: {"overall_story_idea": <story_idea>, "story_spot_min_scenes": <scenes_per_spot>}`.
   Timed events (`timeline_note_create`) and per-scene Director / Video Notes (`scene_update`) go in now: the
   story steps follow them and they win over the story.
10. The book: `story_create` with `step: "book"` (from `story_idea`), or with `step: "split"` and `text: <story_text>`
    when the user gave a finished story. Wait for the job. Read it back with `GET /projects/{pid}/story`
    (`story_book_text`). To fix a chapter: `step: "chapter"`, `chapter_id`, `direction`.
11. `story_create` with `step: "cast"`. For each listed character the user wants, `reference_upsert` a subject with
    the given name and description (they need an image before rendering).
12. Places, as the Story Builder's Places tab does before the acts: `story_create` with `step: "locations"`
    (`style_theme` = `location_style_theme`). Wait. It finds the places inside each chapter setting, writes each
    setting's look and makes one location card per place. Places the user named go in first as location cards
    (`reference_upsert`, `kind: "locations"`) and are kept word for word.
13. `story_create` with `step: "acts"`. Wait. The acts happen inside the places.
14. `story_create` with `step: "plan"`. Wait. Every scene now has a story slice, a spot, a mood and a performance
    (singing, speaking, story only or instrumental), and each act's cast is mapped onto scenes that had none.
15. `story_create` with `step: "locations"` and `map_only: true`. It asks the LLM nothing: it maps every planned scene
    to the card of its spot. Do not run `reference_extract_locations` or `reference_assign_scenes` in Story Mode.
16. `story_create` with `step: "beats"`. Wait. Then continue with Part 3 from step 15 (MiniMax prompts, render,
    stitch).

---

## Part 3c - Cards that render from a saved RefMod, and scenes without the singer

### Cards that render from a RefMod
When the project renders with RefMods (`minimax_h3.pipeline` is `refmod`), H3 receives each scene's RefMods and its audio and
nothing else: no scene image and no location image. Every scene needs at least one RefMod.
1. `api_get_refmods` lists the saved RefMods, each as `<type>/<name>`.
2. Pick one on a card with `reference_upsert`:
   `payload: {"name": "Corvette", "reference_type": "vehicle", "source": "refmod", "refmod": {"name": "vehicle/C8_Corvette", "strength": 1}}`.
   A location card takes the same fields. A name that is not saved is an error that lists the saved ones. `source: "image"` takes
   the RefMod off again.
3. To make a RefMod from pictures, such as a place or a person you generated, call `refmod_create` with `name`, `type` and
   `paths`. It is a GPU job: wait with `job_wait`. Then pick `<type>/<name>` as in step 2.
4. Map the cards to scenes with `reference_scene_mapping_set`. `api_get_scenes_minimax_references` for a scene shows what its
   render sends (`items`, and `warnings` when a RefMod is missing).

### Scenes without the singer
H3 makes every face in a shot lip-sync to the audio it hears. For a scene where the singer is not on screen, render it against
the song without vocals:
1. `audio_stems_split` once. Wait for the job. The result has `no_vocals_path`.
2. `video_render` for that scene with `params: {"mode": "minimax_h3", "audio_path": <no_vocals_path>}`.
The stitch puts the project song under the final video, so the vocals are back in the final.

---

## Part 4 - What "done" means

- Every scene has a story beat, a MiniMax prompt and a rendered clip.
- `FINAL_VIDEO.mp4` exists in the project folder and is about as long as the song.
- The project opens in the Video Builder with the lyrics lane, scene pictures and prompts visible.
