# Agent API endpoints

Every endpoint of the VRGDG Agent API, and what it does. Generated from `agent_api/router.py` by
`scripts/export_api_endpoints.py`. The machine-readable contract is `agent_api/openapi.json`.

**Base URL:** `http://127.0.0.1:8188/vrgdg/api/v1` (ComfyUI's address plus the prefix).
Every path below is relative to it.

**Auth:** requests from the same machine need no token unless the server is configured to require one. Other
hosts send `Authorization: Bearer <token>`.

**Responses:** `{"ok": true, "data": ..., "revision": n}` on success and
`{"ok": false, "error": {"code", "message", "details", "retryable"}}` on failure. The HTTP status mirrors the error.

**Jobs:** an endpoint marked **job** returns `202` with a `job_id` at once and keeps working in the background.
Poll `GET /jobs/{id}` until `status` is `succeeded`, `failed`, `cancelled` or `interrupted`. The result is in
`result`, the reason for a failure in `error`.

**Revisions:** a project has a `revision` that goes up on every save. Endpoints marked **If-Match** accept the
header `If-Match: <revision>` and answer `409` when the project changed since you read it.

**LLM steps** use the runner saved in the project. For LM Studio they use the model that is already loaded and
never load or switch models. They answer `503 LLM_UNAVAILABLE` when nothing is loaded.

**Path names:** `{pid}`, `{project_id}` and `{id}` are the project or job id. `{sid}` and `{scene_id}` are a scene id
or its 1-based number.


**Total:** 137 endpoints.

## Service and discovery

| Method | Path | What it does |
|---|---|---|
| `GET` | `/events` | Server-sent events stream of job and project events, with a keep-alive every 15 seconds. Filter with `project_id`. _(query: `project_id`)_ |
| `GET` | `/health` | Health check: `healthy` or `degraded`, whether ComfyUI is up, its queue depth, whether FFmpeg is found and whether a GPU is available. |
| `GET` | `/meta` | API version (`v1`), pack version and update date, schema version and the list of capabilities (projects, scenes, prompts, videos, latents, post, face fix, pipelines, jobs, events). |
| `GET` | `/models` | Models, checkpoints and LoRAs installed in ComfyUI, grouped by type. |
| `GET` | `/modes` | Supported image and video modes with their requirements and capabilities. |
| `GET` | `/queue` | Summary of the API job queue: running, queued and finished jobs. |
| `GET` | `/refmods` | Saved RefMods in models/refmods with their type (folder), kind, frame count, tokens and description. Filter with `folder` (for example `identity`). A project switched to the RefMod pipeline (`pipeline: refmod` in the MiniMax H3 settings) renders from these. _(query: `folder`)_ |

## Project endpoints

### Project

| Method | Path | What it does |
|---|---|---|
| `GET` | `/projects` | List projects found in the allowed project roots. `root` limits the search to one root. _(query: `root`)_ |
| `POST` | `/projects` | Create a project: folder, empty session seeded with your saved model defaults. Body: `name`, optional `template_from`. _(body: `name`, `template_from`)_ |
| `DELETE` | `/projects/{pid}` | Delete a project folder from disk. Needs `confirm` equal to the project id. _(query: `confirm`)_ |
| `GET` | `/projects/{pid}` | The project with its settings, scenes, audio, story and references. `include` picks some of those groups and/or top-level session keys by name (e.g. `audio_path`, `detected_tempo_bpm`, `flux_reference_builder`; API keys come back blank). A Builder session key the project has not saved yet comes back empty (`{}` for `flux_reference_builder`, `builder_story_layer` and `builder_storyboard_defaults`, otherwise `null`). A name that is not a Builder session key is a 400 that lists it. _(query: `include`)_ |
| `GET` | `/projects/{pid}/assets` | Files in the project folder: images, videos, thumbnails, audio and final videos. |
| `POST` | `/projects/{pid}/duplicate` | Copy a project to `new_name`. `options` chooses what to keep (scenes, mappings, notes, prompts, media). _(body: `new_name`, `options`)_ |
| `POST` | `/projects/{pid}/export` | Build a zip of the project for backup or sharing. |
| `GET` | `/projects/{pid}/settings` | Effective settings in every group, including all MiniMax H3 options. |
| `PATCH` | `/projects/{pid}/settings` | Change settings by group, for example `{"minimax_h3": {"render_pass": "two_pass"}}`. Saved under the same keys the Video Builder uses. _(If-Match)_ |
| `POST` | `/projects/{pid}/settings/preflight` | Check the saved settings against the chosen video and image modes and report problems before rendering. |
| `GET` | `/projects/{pid}/summary` | Compact status: scene counts, how many have images, prompts and videos, total duration. |
| `POST` | `/projects/{pid}/validate` | Check the project is consistent: files exist, scene timing, references, settings. |

### Audio and lyrics

| Method | Path | What it does |
|---|---|---|
| `PUT` | `/projects/{pid}/audio` | Attach the song: a file path (`audio_path`) or uploaded data (`audio_data`, `audio_name`). Saves the waveform peaks, beats and tempo. _(If-Match; body: `audio_data`, `audio_name`, `audio_path`)_ |
| `GET` | `/projects/{pid}/audio/beats` | Beat markers and detected tempo. |
| `PUT` | `/projects/{pid}/audio/beats` | Replace the beat markers (`beats`) and tempo (`tempo_bpm`). _(If-Match; body: `beats`, `tempo_bpm`)_ |
| `POST` | `/projects/{pid}/audio/silent` | Create a silent audio track of a given `duration` instead of a song. _(If-Match; body: `duration`, `scope`)_ |
| `GET` | `/projects/{pid}/audio/waveform` | Waveform peaks for drawing the audio. `peaks` sets how many. _(query: `peaks`)_ |
| `GET` | `/projects/{pid}/lyrics` | The project's lyrics text and the lyric text on each scene. |
| `PUT` | `/projects/{pid}/lyrics` | Set the lyrics (`lyrics_text`) or an SRT (`srt_text`). The text is the reference Line Mapping works from. _(If-Match; body: `lyrics_text`, `srt_text`)_ |
| `POST` | `/projects/{pid}/lyrics/align` | Time the lyrics against the song with the Stable-ts workflow in ComfyUI and save the timed lines. _(**job**)_ |

### Timeline

| Method | Path | What it does |
|---|---|---|
| `POST` | `/projects/{pid}/timeline/bulk` | Rebuild or extend the whole timeline from text. `mode` is `durations`, `ranges` or `markers`; `action` is `replace` or `append` (`append_start` sets where appended scenes begin). A line can end with the words for its scene, for example `12.5 --> 16.0 Hello darkness`. On `replace`, when no line carries words, the lyrics of the scenes being replaced move onto the new scenes by time, so they are not lost. Returns `scene_count`, `scenes_with_lyrics` and `lyrics` (`from_text`, `carried_over` or `none`). _(body: `action`, `append_start`, `clear_media`, `mode`, `text`)_ |
| `POST` | `/projects/{pid}/timeline/calibrate` | Shift all beat markers by `offset_seconds`. _(If-Match; body: `offset_seconds`)_ |
| `POST` | `/projects/{pid}/timeline/close-gaps` | Remove gaps between scenes by moving later scenes earlier. |
| `POST` | `/projects/{pid}/timeline/enforce-length` | Merge scenes shorter than `min_scene_seconds` and cut scenes longer than `max_scene_seconds`. `dry_run` shows the plan only. Scenes with rendered video are left alone. _(body: `dry_run`, `max_scene_seconds`, `min_scene_seconds`)_ |
| `POST` | `/projects/{pid}/timeline/from-lines` | Make the timeline scenes from the lyrics, like Line Mapping: one scene per lyric line, short scenes merged and long ones split to fit `min_scene_seconds` and `max_scene_seconds`. Turns on the lyric lane. _(**job**)_ |
| `POST` | `/projects/{pid}/timeline/snap` | Snap a scene edge to the nearest beat (`scene_id`, `edge`, `scope`). _(body: `edge`, `scope`)_ |

### Scenes

| Method | Path | What it does |
|---|---|---|
| `GET` | `/projects/{pid}/scenes` | List scenes. Filter with `has_image`, `has_prompt`, `has_video` and `status`. _(query: `has_image`, `has_prompt`, `has_video`, `status`)_ |
| `POST` | `/projects/{pid}/scenes` | Insert a scene (`position`, `ref_scene_id`, `duration`, `label`, `notes`, prompts). Later scene files are renumbered. _(If-Match; body: `duration`, `i2v_prompt`, `label`, `notes`, `position`, `ref_scene_id`, `t2i_prompt`)_ |
| `POST` | `/projects/{pid}/scenes/bulk` | Apply several scene operations in one atomic change (`operations`). _(If-Match; body: `operations`)_ |
| `DELETE` | `/projects/{pid}/scenes/{sid}` | Delete a scene. `ripple` closes the gap. Later scene files are renumbered. _(If-Match; query: `ripple`)_ |
| `GET` | `/projects/{pid}/scenes/{sid}` | One scene: timing, lyrics, story beat, prompts (including `minimax_h3_prompt`), approved image, rendered video with its thumbnail, and scene audio. Each file is `null` when it does not exist. |
| `PATCH` | `/projects/{pid}/scenes/{sid}` | Change scene fields: `lyric_text` (sets `lyric_no_lip_sync` from the text unless you send it), `lyric_singers`, `story_beat`, prompts (`t2i_prompt`, `i2v_prompt`, `enhance_prompt`, `minimax_h3_prompt`, `minimax_h3_pass2_prompt`, `flux_prompt`, `nb_prompt`, `flow_gpt_prompt`, `ernie_t2i_prompt`), `minimax_h3_continuation_direction` (what a scene continued with `latent_continuation_masked` does after its first moments, one smooth movement, no cut), `minimax_h3_continuation_start_seconds` (second of that scene where the direction starts: 0.5 by default, at most half the scene, `null` resets), `notes`, `label`, `start`, `end`, `no_character_present`, `lyric_no_lip_sync`, or per-scene `use_scene_*` / `*_settings`. A field that cannot be patched returns a validation error that lists the supported fields, and nothing is saved. _(If-Match)_ |
| `POST` | `/projects/{pid}/scenes/{sid}/merge` | Merge a scene with its neighbour (`with_direction`: previous or next). Lyrics are joined. _(If-Match; body: `with_direction`)_ |
| `GET` | `/projects/{pid}/scenes/{sid}/minimax-references` | What the Video Builder's Choose MiniMax References button shows for a scene: `available` (every character, extra, location and ingredients sheet with an image, each with its `key`, whether the scene mapping already picks it, and its `image_number` when selected), `selected` (the order sent to MiniMax), `custom` (chosen by hand or following the scene mappings), `automatic_keys` and the `limits` (9 images, 8 choices when the scene image is Image 1). |
| `PUT` | `/projects/{pid}/scenes/{sid}/minimax-references` | Choose the ordered MiniMax references for a scene: `keys` is the list from `available` (for example `["subject:ava", "location:roof", "location:alley"]`, several locations are allowed), or `automatic: true` to follow the scene mappings again. Unknown or repeated keys and too many keys are refused with the allowed keys listed. _(If-Match)_ _(If-Match; body: `automatic`, `keys`)_ |
| `POST` | `/projects/{pid}/scenes/{sid}/move` | Move a scene to `start_time`. `ripple` shifts the others. _(If-Match; body: `ripple`, `start_time`)_ |
| `POST` | `/projects/{pid}/scenes/{sid}/resize` | Change a scene's length (`duration` or `end_time`). `ripple` shifts the others. _(If-Match; body: `duration`, `end_time`, `ripple`)_ |
| `POST` | `/projects/{pid}/scenes/{sid}/split` | Split a scene at `at_time`. `clear_right_media` drops the media on the new right half. Lyrics follow their words. _(If-Match; body: `at_time`, `clear_right_media`)_ |

### References (characters and locations)

| Method | Path | What it does |
|---|---|---|
| `GET` | `/projects/{pid}/references` | Characters, locations and the scene mappings. |
| `POST` | `/projects/{pid}/references/assign-scenes` | Assign characters and locations to scenes by pattern (`random`, `rotate`, `blocks`, `unchanged`). `dry_run` previews. |
| `POST` | `/projects/{pid}/references/locations/extract` | Ask the project's LLM for filming locations from the lyrics and `style_theme` (LM Extract) and add them. _(**job**)_ |
| `PUT` | `/projects/{pid}/references/locations/{rid}` | Create or update a location (name, description, optional image). _(If-Match)_ |
| `GET` | `/projects/{pid}/references/scene-mapping` | Read which characters, locations, ingredients and extras each scene uses. A scene maps to one location here. To see every reference a MiniMax scene can pick from, and the order it sends, use `GET /projects/{pid}/scenes/{sid}/minimax-references`. |
| `PUT` | `/projects/{pid}/references/scene-mapping` | Set which characters, locations, ingredients and extras each scene uses. Sending `subjects` or `locations` also turns on that "use reference" switch, as the Builder does. _(If-Match)_ |
| `PUT` | `/projects/{pid}/references/subjects/{rid}` | Create or update a character (name, description, `reference_type`, voice, trigger phrase, `image`). _(If-Match)_ |
| `DELETE` | `/projects/{pid}/references/{kind}/{rid}` | Delete a character or location (`kind` is `subjects` or `locations`) and its scene mappings. _(If-Match)_ |
| `POST` | `/projects/{pid}/references/{kind}/{rid}/describe` | Describe a character or location image with the project's LLM (Gemma Describe) and save the description. _(**job**)_ |

### Story

| Method | Path | What it does |
|---|---|---|
| `GET` | `/projects/{pid}/story` | The saved story layer: idea, arc, brief. |
| `PUT` | `/projects/{pid}/story` | Replace the saved story layer. _(If-Match)_ |
| `PUT` | `/projects/{pid}/story/settings` | Save the Storyboard scene defaults (`defaults`: every key the Builder saves in `builder_storyboard_defaults`, e.g. video style, camera flow, motion speeds, cut frequency, short film planning, temporal and FX settings, plus `story_arc_detail`) and the story fields (`story`: every key of `builder_story_layer`, e.g. `enabled`, idea, strength, world style). |
| `POST` | `/projects/{pid}/story/{step}` | Write a story step with the project's LLM. `step` is `arc` (from the story idea), `brief` or `beats` (a beat per scene without one; `replace_existing`, `scene_ids`, `limit`). Each step also updates the Storyboard Builder's saved copy. _(**job**)_ |

### Prompts

| Method | Path | What it does |
|---|---|---|
| `POST` | `/projects/{pid}/minimax-prompts` | Write MiniMax H3 reference-to-video prompts with the project's LLM for scenes that have none (`replace_existing`, `scene_ids`, `limit`). Each singing scene's prompt has its lyric in double quotes after 'sings the lyric line,'. Saves them on the scenes and in the Storyboard Builder's copy (`storyboard/storyboard.json` and the `prompts/` files). Needs a mapped character with an image on each scene. Each prompt is the Builder's full format: subject definitions tying `<Subject N>` to `<Picture N>`, summary, retention analysis, the shots, and the soundscape. _(**job**)_ |
| `POST` | `/projects/{pid}/prompts/batch` | Write image or video prompts for many scenes (`kind`, `scope`, `run_mode`, `scene_ids`). Image prompts use each scene's notes, lyric and references. _(**job**)_ |
| `POST` | `/projects/{pid}/prompts/concepts` | Write scene concept prompts for the whole project with the LLM. _(**job**)_ |
| `POST` | `/projects/{pid}/prompts/motion-notes` | Write motion and camera notes for scenes with the LLM. _(**job**)_ |
| `GET` | `/projects/{pid}/scenes/{sid}/prompts/context` | The context brief an outside agent needs to write a prompt itself: cast, cut plan and character budget (`kind`). A scene continued with `latent_continuation_masked` also gets a `continuation` block: `hold_seconds`, the scene's `direction` and the `rules` for writing it as the next moment of the previous scene's take. _(query: `kind`)_ |
| `POST` | `/projects/{pid}/scenes/{sid}/prompts/edit` | Rewrite an existing scene prompt following an instruction. _(**job**)_ |
| `POST` | `/projects/{pid}/scenes/{sid}/prompts/enhance` | Improve an existing scene prompt with the LLM. _(**job**)_ |
| `POST` | `/projects/{pid}/scenes/{sid}/prompts/image` | Write one scene's image prompt with the LLM, from the scene's notes, lyric and references (`user_notes` overrides them). _(**job**)_ |
| `POST` | `/projects/{pid}/scenes/{sid}/prompts/minimax/assemble` | Build a MiniMax prompt from shot descriptions you provide (`shots`, `mode`). `save` stores it. A reference-to-video prompt gets the same subject definitions and soundscape as `minimax-prompts`. _(If-Match; body: `mode`, `save`, `shots`)_ |
| `POST` | `/projects/{pid}/scenes/{sid}/prompts/minimax/validate` | Check a MiniMax prompt against the length and format rules. _(body: `mode`, `prompt`)_ |
| `POST` | `/projects/{pid}/scenes/{sid}/prompts/video` | Write one scene's video prompt with the LLM (`mode`), from its image prompt, motion notes and references. _(**job**)_ |
| `POST` | `/projects/{pid}/scenes/{sid}/prompts/video-chained` | Write one scene's video prompt continuing from the previous scene's last frame, using the scene's own prompt, notes and references. _(**job**)_ |
| `PUT` | `/projects/{pid}/scenes/{sid}/prompts/{field}` | Set a prompt field directly (`t2i_prompt`, `i2v_prompt`, `minimax_h3_prompt`, ...). Body: `prompt`, `origin`. _(If-Match; body: `origin`, `prompt`)_ |

### Images

| Method | Path | What it does |
|---|---|---|
| `POST` | `/projects/{project_id}/images/generate` | Generate images for many scenes. _(**job**)_ |
| `DELETE` | `/projects/{project_id}/scenes/{scene_id}/image` | Remove the scene's image. |
| `PUT` | `/projects/{project_id}/scenes/{scene_id}/image` | Set the scene image from a file (`source_path`) or upload (`image_data`). _(body: `image_data`, `source_path`)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/image/approve` | Approve one generated image as the scene's image (`image_path`). _(body: `image_path`)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/image/from-video-frame` | Take the scene image from a frame of a video (`source_video_path`). _(body: `source_video_path`)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/image/generate` | Generate a scene image with ComfyUI. _(**job**)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/image/revert` | Step through the scene's image history (`index` or `delta`). _(body: `delta`, `index`)_ |

### Video, stitch and finals

| Method | Path | What it does |
|---|---|---|
| `GET` | `/projects/{project_id}/finals` | List the project's final videos. |
| `DELETE` | `/projects/{project_id}/scenes/{scene_id}/video` | Clear the scene's active video. Files stay on disk. |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/video/match-start-color` | Match the opening colors of a scene video to the previous scene's last frame. _(**job**)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/video/minimax-stage-recover` | Recover a MiniMax pass-1 or pass-2 clip from the scratch folder. |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/video/recover` | Bring back a scene video from a backup or file (`source_path`). _(body: `source_path`)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/video/render` | Render one scene's video with ComfyUI (`mode`, e.g. `minimax_h3`). Trims to the exact timeline length and saves it as `video_NNNN-audio.mp4`. Set `audio_mode: built_in_audio` in the MiniMax settings for H3 voices and sound (Single or 2 Pass; 2 Pass Advanced needs input audio). With `continuity_mode: latent_continuation_masked` the previous scene must be rendered first (`PREDECESSOR_MISSING` otherwise); it works in Single and 2 Pass. When `continuity_prompt_from_last_frame` is also true, the scene's prompt is first written by the LLM from the previous scene's rendered final frame (the Video Builder's automatic prompt), saved on the scene, and then rendered. A `prompt` in `params` skips that. _(**job**)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/video/select` | Choose which take is the scene's active video (`source_path`). _(body: `source_path`)_ |
| `GET` | `/projects/{project_id}/scenes/{scene_id}/video/takes` | List the scene's raw (untrimmed) renders, newest first, with length, frame count and whether the file still exists. Takes in a sibling scratch folder with the same project name are marked `other_folder`. Only renders made through the API are always kept: the Video Builder deletes its scratch renders after each render. Use `take` on the trim call. |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/video/trim` | Trim a scene video as a job. Name the source with `source_path` (default: the scene's current video) or with `take` (`latest` or an index from `video/takes`, the raw render, so the clip can start earlier or end later than the current one). `start` is seconds into the source; give `duration` (and `frames`), or `to_end: true` to end on the source's last frame. Records the new clip on the scene and keeps the old one in its video history. _(**job**)_ |
| `POST` | `/projects/{project_id}/slideshow` | Make a video from the scene images instead of rendered clips. _(**job**)_ |
| `POST` | `/projects/{project_id}/stitch` | Join the scenes' videos into `FINAL_VIDEO.mp4` with the project audio. _(**job**)_ |
| `POST` | `/projects/{project_id}/video/graph` | Build the ComfyUI graph for a video `mode` and the given parameters without running it, to inspect what would be sent. |
| `POST` | `/projects/{project_id}/video/render` | Render many scenes' videos one after another as a GPU job, and stitch them when asked. _(**job**)_ |
| `POST` | `/projects/{project_id}/video/scan` | Scan the project folder for rendered scene videos and thumbnails and re-link them to scenes. |

### MiniMax files

| Method | Path | What it does |
|---|---|---|
| `POST` | `/projects/{project_id}/minimax/cleanup` | Delete MiniMax scratch outputs that are no longer needed. |
| `GET` | `/projects/{project_id}/minimax/index` | Index of the MiniMax scratch outputs per scene. |

### Latents

| Method | Path | What it does |
|---|---|---|
| `GET` | `/projects/{project_id}/latents` | Latent chain status for every scene (MiniMax continuity). |
| `GET` | `/projects/{project_id}/latents/dirty` | Scenes whose latent is out of date because an earlier scene changed. |
| `POST` | `/projects/{project_id}/latents/rebuild` | Rebuild out-of-date latents in order. _(**job**)_ |
| `DELETE` | `/projects/{project_id}/scenes/{scene_id}/latent` | Delete a scene's latent (`all`, `reindex`). _(query: `all`, `reindex`; body: `all`, `reindex`)_ |
| `GET` | `/projects/{project_id}/scenes/{scene_id}/latent` | One scene's latent file and status. |

### Post-processing (per project and scene)

| Method | Path | What it does |
|---|---|---|
| `POST` | `/projects/{project_id}/post/apply-all` | Apply the project's post-processing settings to every scene video. _(**job**)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/post/adjust` | Apply color and tone adjustments to a scene video. _(**job**)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/post/adjust/preview` | Make a still preview of the adjustments on the scene. |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/post/film-grain` | Apply film grain to a scene video. _(**job**)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/post/film-grain/preview` | Make a still preview of film grain on the scene. |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/post/lut` | Apply a LUT to a scene video. _(**job**)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/post/lut/preview` | Make a still preview of a LUT on the scene. |

### Face fix

| Method | Path | What it does |
|---|---|---|
| `POST` | `/projects/{project_id}/scenes/{scene_id}/face-fix/anchors/{n}/enhance` | Enhance one anchor frame (`order`). _(**job**; body: `order`)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/face-fix/auto` | Run prepare, enhance, LTX and finalize automatically. _(**job**)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/face-fix/estimate` | Estimate the work and GPU time to fix faces in a scene video. |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/face-fix/finalize` | Combine the fixed runs into the scene video. _(**job**)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/face-fix/prepare` | Find the faces and prepare anchor frames. _(**job**)_ |
| `POST` | `/projects/{project_id}/scenes/{scene_id}/face-fix/runs/{n}/ltx` | Run the LTX face-fix pass for one run (`run_index`). _(**job**; body: `run_index`)_ |

### Pipelines

| Method | Path | What it does |
|---|---|---|
| `POST` | `/pipelines/from-song` | Song to final video in one job: create the project if needed, attach audio and lyrics, make the scenes, then run the full build. _(**job**; body: `project_name`)_ |
| `POST` | `/projects/{project_id}/pipelines/build-flf` | Run the first/last-frame (FLF) build for an existing project as one job. _(**job**)_ |
| `POST` | `/projects/{project_id}/pipelines/build-full-video` | Run the whole build for an existing project (prompts, images, videos, stitch) as one job. _(**job**)_ |
| `GET` | `/projects/{project_id}/pipelines/plan` | Dry-run plan for the full build: steps, estimated GPU minutes and missing prerequisites. |

## Jobs

| Method | Path | What it does |
|---|---|---|
| `GET` | `/jobs` | List jobs, newest first. Filter with `project_id`, `status` and `type`. _(query: `project_id`, `status`, `type`)_ |
| `GET` | `/jobs/{id}` | One job: status (`queued`, `running`, `succeeded`, `failed`, `cancelled`, `interrupted`), progress, result and error. |
| `POST` | `/jobs/{id}/cancel` | Cancel a queued or running job. A running ComfyUI render is interrupted. |
| `GET` | `/jobs/{id}/log` | The job's progress log. Pass `since` (a sequence number) to read only new lines. _(query: `since`)_ |
| `POST` | `/jobs/{id}/retry` | Start a failed or interrupted job again. `resume: true` skips work that already finished. _(**job**; body: `resume`)_ |

## LLM and prompt instructions

| Method | Path | What it does |
|---|---|---|
| `GET` | `/instructions` | List the prompt-writing instructions (system prompts) the LLM steps use, with their override state. |
| `PUT` | `/instructions/presets/{name}` | Save the current text of an instruction as a named preset. _(body: `key`, `name`)_ |
| `POST` | `/instructions/presets/{name}/load` | Load a named preset into an instruction. _(body: `key`, `name`)_ |
| `GET` | `/instructions/{key}` | One instruction's current text, its default and whether a project override is set. |
| `PUT` | `/instructions/{key}` | Save an instruction override for a project or for all projects. _(body: `key`, `project_folder`)_ |
| `DELETE` | `/instructions/{key}/override` | Remove an override and return to the built-in instruction. _(body: `key`, `project_folder`)_ |
| `GET` | `/llm/active` | Which LLM the API would use right now, for a project (`project_id`) or, with none, for a new project (your saved model defaults). For LM Studio this is the model that is loaded. The API never loads or switches a model. _(query: `project_id`)_ |
| `GET` | `/llm/models` | Models the configured LLM provider offers. `provider` selects LM Studio, an own server or an API provider. _(query: `provider`)_ |
| `POST` | `/llm/test` | Send a short test request to the configured LLM runner and report whether it answers. _(body: `provider`, `text_runner`)_ |
| `POST` | `/llm/unload` | Free the memory held by the built-in LLM runners (clears their caches and the CUDA cache). It does not touch LM Studio. |

## Post-processing library

| Method | Path | What it does |
|---|---|---|
| `GET` | `/post/adjust/presets` | List the saved color and tone adjustment presets. |
| `PUT` | `/post/adjust/presets/{name}` | Save a color and tone adjustment preset (`settings`). _(body: `settings`)_ |
| `GET` | `/post/luts` | List the available LUT (color look-up table) files. |
| `DELETE` | `/post/luts/previews/{id}` | Delete a LUT, grain or color preview file the API created. |
| `POST` | `/post/luts/upload` | Upload a LUT file (base64 `content` and `filename`). _(query: `filename`; body: `content`, `data`, `filename`)_ |

## Settings schema

| Method | Path | What it does |
|---|---|---|
| `GET` | `/settings/minimax-h3/schema` | Every MiniMax H3 video setting that can be patched under the `minimax_h3` group: type, default, allowed values and limits. `continuity_mode` takes `latent_continuation_masked` (the previous scene's latent is protected at the start of the next, Single and 2 Pass), `latent_context_frames` takes 39, 90, 141 or 192 for it, and `location_transition_preset` takes `masked`. |
