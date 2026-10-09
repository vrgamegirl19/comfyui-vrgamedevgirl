"""Write Api_Endpoints.md: every Agent API v1 endpoint and what it does.

The route list (method, path, query parameters, body keys, If-Match, job or immediate) is read from
``agent_api/router.py`` by the same parser that builds ``openapi.json``. The descriptions are written here, in
``DESCRIPTIONS``. A route with no description fails the run, so the document cannot fall behind the router.

Usage (from the pack root, with the portable Python)::

    ..\\..\\..\\python_embeded\\python.exe scripts/export_api_endpoints.py          # write Api_Endpoints.md
    ..\\..\\..\\python_embeded\\python.exe scripts/export_api_endpoints.py --check  # exit 1 if stale or incomplete
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_openapi as oa  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "Api_Endpoints.md"
JSON_PATH = ROOT / "agent_api" / "endpoints.json"

# (section title, path prefix test) in display order. The first matching section wins.
SECTIONS = [
    ("Service and discovery", ("/health", "/meta", "/modes", "/refmods", "/models", "/events", "/queue")),
    ("Projects", ("/projects", "/pipelines")),
    ("Jobs", ("/jobs",)),
    ("LLM and prompt instructions", ("/llm", "/instructions")),
    ("Post-processing library", ("/post",)),
    ("Settings schema", ("/settings",)),
]

DESCRIPTIONS: Dict[str, str] = {
    # --- Service and discovery -------------------------------------------------------------
    "GET /health": "Health check: `healthy` or `degraded`, whether ComfyUI is up, its queue depth, whether FFmpeg is found and whether a GPU is available.",
    "GET /meta": "API version (`v1`), pack version and update date, schema version and the list of capabilities (projects, scenes, prompts, videos, latents, post, face fix, pipelines, jobs, events).",
    "GET /modes": "Supported image and video modes with their requirements and capabilities.",
    "GET /refmods": "Saved RefMods in models/refmods with their type (folder), kind, frame count, tokens and description. Filter with `folder` (for example `identity`). A project switched to the RefMod pipeline (`pipeline: refmod` in the MiniMax H3 settings) renders from these.",
    "GET /models": "Models, checkpoints and LoRAs installed in ComfyUI, grouped by type.",
    "GET /events": "Server-sent events stream of job and project events, with a keep-alive every 15 seconds. Filter with `project_id`.",
    "GET /queue": "Summary of the API job queue: running, queued and finished jobs.",
    # --- Jobs ------------------------------------------------------------------------------
    "GET /jobs": "List jobs, newest first. Filter with `project_id`, `status` and `type`.",
    "GET /jobs/{id}": "One job: status (`queued`, `running`, `succeeded`, `failed`, `cancelled`, `interrupted`), progress, result and error.",
    "GET /jobs/{id}/log": "The job's progress log. Pass `since` (a sequence number) to read only new lines.",
    "POST /jobs/{id}/cancel": "Cancel a queued or running job. A running ComfyUI render is interrupted.",
    "POST /jobs/{id}/retry": "Start a failed or interrupted job again. `resume: true` skips work that already finished.",
    # --- LLM and instructions --------------------------------------------------------------
    "GET /llm/active": "Which LLM the API would use right now, for a project (`project_id`) or, with none, for a new project (your saved model defaults). For LM Studio this is the model that is loaded. The API never loads or switches a model.",
    "GET /llm/models": "Models the configured LLM provider offers. `provider` selects LM Studio, an own server or an API provider.",
    "POST /llm/test": "Send a short test request to the configured LLM runner and report whether it answers.",
    "POST /llm/unload": "Free the memory held by the built-in LLM runners (clears their caches and the CUDA cache). It does not touch LM Studio.",
    "GET /instructions": "List the prompt-writing instructions (system prompts) the LLM steps use, with their override state.",
    "GET /instructions/{key}": "One instruction's current text, its default and whether a project override is set.",
    "PUT /instructions/{key}": "Save an instruction override for a project or for all projects.",
    "DELETE /instructions/{key}/override": "Remove an override and return to the built-in instruction.",
    "PUT /instructions/presets/{name}": "Save the current text of an instruction as a named preset.",
    "POST /instructions/presets/{name}/load": "Load a named preset into an instruction.",
    # --- Post-processing library -----------------------------------------------------------
    "GET /post/luts": "List the available LUT (color look-up table) files.",
    "POST /post/luts/upload": "Upload a LUT file (base64 `content` and `filename`).",
    "DELETE /post/luts/previews/{id}": "Delete a LUT, grain or color preview file the API created.",
    "GET /post/adjust/presets": "List the saved color and tone adjustment presets.",
    "PUT /post/adjust/presets/{name}": "Save a color and tone adjustment preset (`settings`).",
    # --- Settings schema -------------------------------------------------------------------
    "GET /settings/minimax-h3/schema": "Every MiniMax H3 video setting that can be patched under the `minimax_h3` group: type, default, allowed values and limits. `continuity_mode` takes `latent_continuation_masked` (the previous scene's latent is protected at the start of the next, Single and 2 Pass), `latent_context_frames` takes 39, 90, 141 or 192 for it, and `location_transition_preset` takes `masked`.",
    # --- Pipelines -------------------------------------------------------------------------
    "POST /pipelines/from-song": "Song to final video in one job: create the project if needed, attach audio and lyrics, make the scenes, then run the full build.",
    "GET /projects/{project_id}/pipelines/plan": "Dry-run plan for the full build: steps, estimated GPU minutes and missing prerequisites.",
    "POST /projects/{project_id}/pipelines/build-full-video": "Run the whole build for an existing project (prompts, images, videos, stitch) as one job.",
    "POST /projects/{project_id}/pipelines/build-flf": "Run the first/last-frame (FLF) build for an existing project as one job.",
    # --- Projects --------------------------------------------------------------------------
    "GET /projects": "List projects found in the allowed project roots. `root` limits the search to one root.",
    "POST /projects": "Create a project: folder, empty session seeded with your saved model defaults. Body: `name`, optional `template_from`.",
    "GET /projects/{pid}": "The project with its settings, scenes, audio, story and references. `include` picks some of those groups and/or top-level session keys by name (e.g. `audio_path`, `detected_tempo_bpm`, `flux_reference_builder`; API keys come back blank). A Builder session key the project has not saved yet comes back empty (`{}` for `flux_reference_builder`, `builder_story_layer` and `builder_storyboard_defaults`, otherwise `null`). A name that is not a Builder session key is a 400 that lists it.",
    "DELETE /projects/{pid}": "Delete a project folder from disk. Needs `confirm` equal to the project id.",
    "POST /projects/{pid}/duplicate": "Copy a project to `new_name`. `options` chooses what to keep (scenes, mappings, notes, prompts, media).",
    "POST /projects/{pid}/export": "Build a zip of the project for backup or sharing.",
    "POST /projects/{pid}/validate": "Check the project is consistent: files exist, scene timing, references, settings.",
    "GET /projects/{pid}/summary": "Compact status: scene counts, how many have images, prompts and videos, total duration.",
    "GET /projects/{pid}/assets": "Files in the project folder: images, videos, thumbnails, audio and final videos.",
    "GET /projects/{pid}/settings": "Effective settings in every group, including all MiniMax H3 options.",
    "PATCH /projects/{pid}/settings": "Change settings by group, for example `{\"minimax_h3\": {\"render_pass\": \"two_pass\"}}`. Saved under the same keys the Video Builder uses.",
    "POST /projects/{pid}/settings/preflight": "Check the saved settings against the chosen video and image modes and report problems before rendering.",
    # --- Audio and lyrics ------------------------------------------------------------------
    "PUT /projects/{pid}/audio": "Attach the song: a file path (`audio_path`) or uploaded data (`audio_data`, `audio_name`). Saves the waveform peaks, beats and tempo.",
    "GET /projects/{pid}/audio/waveform": "Waveform peaks for drawing the audio. `peaks` sets how many.",
    "POST /projects/{pid}/audio/silent": "Create a silent audio track of a given `duration` instead of a song.",
    "GET /projects/{pid}/audio/beats": "Beat markers and detected tempo.",
    "PUT /projects/{pid}/audio/beats": "Replace the beat markers (`beats`) and tempo (`tempo_bpm`).",
    "GET /projects/{pid}/lyrics": "The project's lyrics text and the lyric text on each scene.",
    "PUT /projects/{pid}/lyrics": "Set the lyrics (`lyrics_text`) or an SRT (`srt_text`). The text is the reference Line Mapping works from.",
    "POST /projects/{pid}/lyrics/align": "Time the lyrics against the song with the Stable-ts workflow in ComfyUI and save the timed lines.",
    # --- Timeline --------------------------------------------------------------------------
    "POST /projects/{pid}/timeline/from-lines": "Make the timeline scenes from the lyrics, like Line Mapping: one scene per lyric line, short scenes merged and long ones split to fit `min_scene_seconds` and `max_scene_seconds`. Turns on the lyric lane.",
    "POST /projects/{pid}/timeline/enforce-length": "Merge scenes shorter than `min_scene_seconds` and cut scenes longer than `max_scene_seconds`. `dry_run` shows the plan only. Scenes with rendered video are left alone.",
    "POST /projects/{pid}/timeline/close-gaps": "Remove gaps between scenes by moving later scenes earlier.",
    "POST /projects/{pid}/timeline/snap": "Snap a scene edge to the nearest beat (`scene_id`, `edge`, `scope`).",
    "POST /projects/{pid}/timeline/bulk": "Rebuild or extend the whole timeline from text. `mode` is `durations`, `ranges` or `markers`; `action` is `replace` or `append` (`append_start` sets where appended scenes begin). A line can end with the words for its scene, for example `12.5 --> 16.0 Hello darkness`. On `replace`, when no line carries words, the lyrics of the scenes being replaced move onto the new scenes by time, so they are not lost. Returns `scene_count`, `scenes_with_lyrics` and `lyrics` (`from_text`, `carried_over` or `none`).",
    "POST /projects/{pid}/timeline/calibrate": "Shift all beat markers by `offset_seconds`.",
    # --- Scenes ----------------------------------------------------------------------------
    "GET /projects/{pid}/scenes": "List scenes. Filter with `has_image`, `has_prompt`, `has_video` and `status`.",
    "POST /projects/{pid}/scenes": "Insert a scene (`position`, `ref_scene_id`, `duration`, `label`, `notes`, prompts). Later scene files are renumbered.",
    "POST /projects/{pid}/scenes/bulk": "Apply several scene operations in one atomic change (`operations`).",
    "GET /projects/{pid}/scenes/{sid}": "One scene: timing, lyrics, story beat, prompts (including `minimax_h3_prompt`), approved image, rendered video with its thumbnail, and scene audio. Each file is `null` when it does not exist.",
    "PATCH /projects/{pid}/scenes/{sid}": "Change scene fields: `lyric_text` (sets `lyric_no_lip_sync` from the text unless you send it), `lyric_singers`, `story_beat`, prompts (`t2i_prompt`, `i2v_prompt`, `enhance_prompt`, `minimax_h3_prompt`, `minimax_h3_pass2_prompt`, `flux_prompt`, `nb_prompt`, `flow_gpt_prompt`, `ernie_t2i_prompt`), `minimax_h3_i2v_frame_mode` (`normal` or `flf`, per scene), `first_last_frame_end_image_path` (explicit last image for I2V FLF), `minimax_h3_continuation_direction` (what a scene continued with `latent_continuation_masked` does after its first moments, one smooth movement, no cut), `minimax_h3_continuation_start_seconds` (second of that scene where the direction starts: 0.5 by default, at most half the scene, `null` resets), `notes`, `label`, `start`, `end`, `no_character_present`, `lyric_no_lip_sync`, or per-scene `use_scene_*` / `*_settings`. A field that cannot be patched returns a validation error that lists the supported fields, and nothing is saved.",
    "DELETE /projects/{pid}/scenes/{sid}": "Delete a scene. `ripple` closes the gap. Later scene files are renumbered.",
    "POST /projects/{pid}/scenes/{sid}/split": "Split a scene at `at_time`. `clear_right_media` drops the media on the new right half. Lyrics follow their words.",
    "POST /projects/{pid}/scenes/{sid}/merge": "Merge a scene with its neighbour (`with_direction`: previous or next). Lyrics are joined.",
    "POST /projects/{pid}/scenes/{sid}/move": "Move a scene to `start_time`. `ripple` shifts the others.",
    "POST /projects/{pid}/scenes/{sid}/resize": "Change a scene's length (`duration` or `end_time`). `ripple` shifts the others.",
    # --- References ------------------------------------------------------------------------
    "GET /projects/{pid}/references": "Characters, locations and the scene mappings.",
    "PUT /projects/{pid}/references/subjects/{rid}": "Create or update a character (name, description, `reference_type`, voice, trigger phrase, `image`).",
    "PUT /projects/{pid}/references/locations/{rid}": "Create or update a location (name, description, optional image).",
    "DELETE /projects/{pid}/references/{kind}/{rid}": "Delete a character or location (`kind` is `subjects` or `locations`) and its scene mappings.",
    "GET /projects/{pid}/references/scene-mapping": "Read which characters, locations, ingredients and extras each scene uses. A scene maps to one location here. To see every reference a MiniMax scene can pick from, and the order it sends, use `GET /projects/{pid}/scenes/{sid}/minimax-references`.",
    "PUT /projects/{pid}/references/scene-mapping": "Set which characters, locations, ingredients and extras each scene uses. Sending `subjects` or `locations` also turns on that \"use reference\" switch, as the Builder does.",
    "GET /projects/{pid}/scenes/{sid}/minimax-references": "What the Video Builder's Choose MiniMax References button shows for a scene: `available` (every character, extra, location and ingredients sheet with an image, each with its `key`, whether the scene mapping already picks it, and its `image_number` when selected), `selected` (the order sent to MiniMax), `custom` (chosen by hand or following the scene mappings), `automatic_keys` and the `limits` (9 images, 8 choices when the scene image is Image 1).",
    "PUT /projects/{pid}/scenes/{sid}/minimax-references": "Choose the ordered MiniMax references for a scene: `keys` is the list from `available` (for example `[\"subject:ava\", \"location:roof\", \"location:alley\"]`, several locations are allowed), or `automatic: true` to follow the scene mappings again. Unknown or repeated keys and too many keys are refused with the allowed keys listed. _(If-Match)_",
    "POST /projects/{pid}/references/{kind}/{rid}/describe": "Describe a character or location image with the project's LLM (Gemma Describe) and save the description.",
    "POST /projects/{pid}/references/locations/extract": "Ask the project's LLM for filming locations from the lyrics and `style_theme` (LM Extract) and add them.",
    "POST /projects/{pid}/references/assign-scenes": "Assign characters and locations to scenes by pattern (`random`, `rotate`, `blocks`, `unchanged`). `dry_run` previews.",
    # --- Story -----------------------------------------------------------------------------
    "GET /projects/{pid}/story": "The saved story layer: idea, arc, brief.",
    "PUT /projects/{pid}/story": "Replace the saved story layer.",
    "PUT /projects/{pid}/story/settings": "Save the Storyboard scene defaults (`defaults`: every key the Builder saves in `builder_storyboard_defaults`, e.g. video style, camera flow, motion speeds, cut frequency, short film planning, temporal and FX settings, plus `story_arc_detail`) and the story fields (`story`: every key of `builder_story_layer`, e.g. `enabled`, idea, strength, world style).",
    "POST /projects/{pid}/story/{step}": "Write a story step with the project's LLM. `step` is `arc` (from the story idea, lyrics and saved timed Timeline Notes), `brief` or `beats` (a beat per scene without one; `replace_existing`, `scene_ids`, `limit`). Each step also updates the Storyboard Builder's saved copy.",
    # --- Prompts ---------------------------------------------------------------------------
    "POST /projects/{pid}/minimax-prompts": "Write MiniMax H3 reference-to-video prompts with the project's LLM for scenes that have none (`replace_existing`, `scene_ids`, `limit`). Each singing scene's prompt has its lyric in double quotes after 'sings the lyric line,'. Saves them on the scenes and in the Storyboard Builder's copy (`storyboard/storyboard.json` and the `prompts/` files). Needs a mapped character with an image on each scene. Each prompt is the Builder's full format: subject definitions tying `<Subject N>` to `<Picture N>`, summary, retention analysis, the shots, and the soundscape.",
    "POST /projects/{pid}/prompts/concepts": "Write scene concept prompts for the whole project with the LLM.",
    "POST /projects/{pid}/prompts/motion-notes": "Write motion and camera notes for scenes with the LLM.",
    "POST /projects/{pid}/prompts/batch": "Write image or video prompts for many scenes (`kind`, `scope`, `run_mode`, `scene_ids`). Image prompts use each scene's notes, lyric and references.",
    "POST /projects/{pid}/scenes/{sid}/prompts/image": "Write one scene's image prompt with the LLM, from the scene's notes, lyric and references (`user_notes` overrides them).",
    "POST /projects/{pid}/scenes/{sid}/prompts/video": "Write one scene's video prompt with the LLM (`mode`), from its image prompt, motion notes and references.",
    "POST /projects/{pid}/scenes/{sid}/prompts/video-chained": "Write one scene's video prompt continuing from the previous scene's last frame, using the scene's own prompt, notes and references.",
    "POST /projects/{pid}/scenes/{sid}/prompts/enhance": "Improve an existing scene prompt with the LLM.",
    "POST /projects/{pid}/scenes/{sid}/prompts/edit": "Rewrite an existing scene prompt following an instruction.",
    "GET /projects/{pid}/scenes/{sid}/prompts/context": "The context brief an outside agent needs to write a prompt itself: cast, cut plan and character budget (`kind`). A scene continued with `latent_continuation_masked` also gets a `continuation` block: `hold_seconds`, the scene's `direction` and the `rules` for writing it as the next moment of the previous scene's take.",
    "POST /projects/{pid}/scenes/{sid}/prompts/minimax/assemble": "Build a MiniMax prompt from shot descriptions you provide (`shots`, `mode`). `save` stores it. A reference-to-video prompt gets the same subject definitions and soundscape as `minimax-prompts`.",
    "POST /projects/{pid}/scenes/{sid}/prompts/minimax/validate": "Check a MiniMax prompt against the length and format rules.",
    "PUT /projects/{pid}/scenes/{sid}/prompts/{field}": "Set a prompt field directly (`t2i_prompt`, `i2v_prompt`, `minimax_h3_prompt`, ...). Body: `prompt`, `origin`.",
    # --- Images ----------------------------------------------------------------------------
    "POST /projects/{project_id}/scenes/{scene_id}/image/generate": "Generate a scene image with ComfyUI.",
    "POST /projects/{project_id}/images/generate": "Generate images for many scenes.",
    "POST /projects/{project_id}/scenes/{scene_id}/image/approve": "Approve one generated image as the scene's image (`image_path`).",
    "POST /projects/{project_id}/scenes/{scene_id}/image/revert": "Step through the scene's image history (`index` or `delta`).",
    "PUT /projects/{project_id}/scenes/{scene_id}/image": "Set the scene image from a file (`source_path`) or upload (`image_data`).",
    "DELETE /projects/{project_id}/scenes/{scene_id}/image": "Remove the scene's image.",
    "POST /projects/{project_id}/scenes/{scene_id}/image/from-video-frame": "Take the scene image from a frame of a video (`source_video_path`).",
    # --- Video -----------------------------------------------------------------------------
    "POST /projects/{project_id}/scenes/{scene_id}/video/render": "Render one scene's video with ComfyUI (`mode`, e.g. `minimax_h3`). Trims to the exact timeline length and saves it as `video_NNNN-audio.mp4`. Set `audio_mode: built_in_audio` in the MiniMax settings for H3 voices and sound (Single or 2 Pass; 2 Pass Advanced needs input audio). With `continuity_mode: latent_continuation_masked` the previous scene must be rendered first (`PREDECESSOR_MISSING` otherwise); it works in Single and 2 Pass. When `continuity_prompt_from_last_frame` is also true, the scene's prompt is first written by the LLM from the previous scene's rendered final frame (the Video Builder's automatic prompt), saved on the scene, and then rendered. A `prompt` in `params` skips that.",
    "POST /projects/{project_id}/video/render": "Render many scenes' videos one after another as a GPU job, and stitch them when asked.",
    "POST /projects/{project_id}/video/graph": "Build the ComfyUI graph for a video `mode` and the given parameters without running it, to inspect what would be sent.",
    "GET /projects/{project_id}/scenes/{scene_id}/video/takes": "List the scene's raw (untrimmed) renders, newest first, with length, frame count and whether the file still exists. Takes in a sibling scratch folder with the same project name are marked `other_folder`. Only renders made through the API are always kept: the Video Builder deletes its scratch renders after each render. Use `take` on the trim call.",
    "POST /projects/{project_id}/scenes/{scene_id}/video/trim": "Trim a scene video as a job. Name the source with `source_path` (default: the scene's current video) or with `take` (`latest` or an index from `video/takes`, the raw render, so the clip can start earlier or end later than the current one). `start` is seconds into the source; give `duration` (and `frames`), or `to_end: true` to end on the source's last frame. Records the new clip on the scene and keeps the old one in its video history.",
    "POST /projects/{project_id}/scenes/{scene_id}/video/match-start-color": "Match the opening colors of a scene video to the previous scene's last frame.",
    "POST /projects/{project_id}/scenes/{scene_id}/video/recover": "Bring back a scene video from a backup or file (`source_path`).",
    "POST /projects/{project_id}/scenes/{scene_id}/video/select": "Choose which take is the scene's active video (`source_path`).",
    "DELETE /projects/{project_id}/scenes/{scene_id}/video": "Clear the scene's active video. Files stay on disk.",
    "POST /projects/{project_id}/video/scan": "Scan the project folder for rendered scene videos and thumbnails and re-link them to scenes.",
    "POST /projects/{project_id}/scenes/{scene_id}/video/minimax-stage-recover": "Recover a MiniMax pass-1 or pass-2 clip from the scratch folder.",
    "POST /projects/{project_id}/minimax/cleanup": "Delete MiniMax scratch outputs that are no longer needed.",
    "GET /projects/{project_id}/minimax/index": "Index of the MiniMax scratch outputs per scene.",
    "POST /projects/{project_id}/stitch": "Join the scenes' videos into one video (`output_prefix`, default `FINAL_VIDEO`), frame-accurate to the timeline like the Builder's stitch: in a MiniMax H3 project every clip is synced to its scene's frames at 24 fps, so the output has exactly the timeline's frames. `scene_ids` picks scenes by number (1 = first scene) or id and stitches them in timeline order (default: every scene); an unknown scene or a selected scene without a video is a 400. `audio`: `auto` (default) uses the scenes' own audio when every selected MiniMax scene renders with built-in audio, the project song otherwise; `embedded` or `project` forces one. With the project song (or `audio_path`) a selection is cut to its window of the song, so it must be contiguous. `overlays` adds insert clips (`path`, `start`, `end`, `source_start`). The job result lists the stitched `scene_ids`, `audio`, the requested song window (`requested_audio_start`, `requested_audio_duration`; `audio_start` repeats the start) and `expected_frame_count`, and describes the written file from ffprobe: `output_width`, `output_height`, `output_duration` and `audio_duration` (= `output_audio_duration`, the audio actually in the file). An unknown body field is a 400.",
    "POST /projects/{project_id}/slideshow": "Make a video from the scene images instead of rendered clips.",
    "GET /projects/{project_id}/finals": "List the project's final videos.",
    # --- Latents ---------------------------------------------------------------------------
    "GET /projects/{project_id}/latents": "Latent chain status for every scene (MiniMax continuity).",
    "GET /projects/{project_id}/latents/dirty": "Scenes whose latent is out of date because an earlier scene changed.",
    "POST /projects/{project_id}/latents/rebuild": "Rebuild out-of-date latents in order.",
    "GET /projects/{project_id}/scenes/{scene_id}/latent": "One scene's latent file and status.",
    "DELETE /projects/{project_id}/scenes/{scene_id}/latent": "Delete a scene's latent (`all`, `reindex`).",
    # --- Post-processing -------------------------------------------------------------------
    "POST /projects/{project_id}/scenes/{scene_id}/post/lut": "Apply a LUT to a scene video.",
    "POST /projects/{project_id}/scenes/{scene_id}/post/lut/preview": "Make a still preview of a LUT on the scene.",
    "POST /projects/{project_id}/scenes/{scene_id}/post/film-grain": "Apply film grain to a scene video.",
    "POST /projects/{project_id}/scenes/{scene_id}/post/film-grain/preview": "Make a still preview of film grain on the scene.",
    "POST /projects/{project_id}/scenes/{scene_id}/post/adjust": "Apply color and tone adjustments to a scene video.",
    "POST /projects/{project_id}/scenes/{scene_id}/post/adjust/preview": "Make a still preview of the adjustments on the scene.",
    "POST /projects/{project_id}/post/apply-all": "Apply the project's post-processing settings to every scene video.",
    # --- Face fix --------------------------------------------------------------------------
    "POST /projects/{project_id}/scenes/{scene_id}/face-fix/estimate": "Estimate the work and GPU time to fix faces in a scene video.",
    "POST /projects/{project_id}/scenes/{scene_id}/face-fix/prepare": "Find the faces and prepare anchor frames.",
    "POST /projects/{project_id}/scenes/{scene_id}/face-fix/anchors/{n}/enhance": "Enhance one anchor frame (`order`).",
    "POST /projects/{project_id}/scenes/{scene_id}/face-fix/runs/{n}/ltx": "Run the LTX face-fix pass for one run (`run_index`).",
    "POST /projects/{project_id}/scenes/{scene_id}/face-fix/finalize": "Combine the fixed runs into the scene video.",
    "POST /projects/{project_id}/scenes/{scene_id}/face-fix/auto": "Run prepare, enhance, LTX and finalize automatically.",
}


def _section_for(path: str) -> str:
    for title, prefixes in SECTIONS:
        if any(path == p or path.startswith(p + "/") for p in prefixes):
            if title == "Projects" and path.startswith("/projects/"):
                break
            return title
    return ""


def _group(path: str) -> str:
    """Sub-group for project routes, by what the route works on."""
    rest = path.split("/", 3)
    tail = rest[3] if len(rest) > 3 else ""
    parts = tail.split("/")
    # /projects/{pid}/<thing>/...
    first = parts[0] if parts else ""
    if path.startswith("/pipelines") or first == "pipelines":
        return "Pipelines"
    if "/prompts" in path or first in ("minimax-prompts", "prompts"):
        return "Prompts"
    if "/face-fix" in path:
        return "Face fix"
    if "/post/" in path:
        return "Post-processing (per project and scene)"
    if "/latent" in path or first == "latents":
        return "Latents"
    if first in ("minimax",):
        return "MiniMax files"
    if "/image" in path or first == "images":
        return "Images"
    if first in ("video", "stitch", "slideshow", "finals") or "/video" in path:
        return "Video, stitch and finals"
    if first in ("audio", "lyrics"):
        return "Audio and lyrics"
    if first == "timeline":
        return "Timeline"
    if first == "references":
        return "References (characters and locations)"
    if first in ("story",):
        return "Story"
    if first == "scenes":
        return "Scenes"
    return "Project"


_GROUP_ORDER = ["Project", "Audio and lyrics", "Timeline", "Scenes", "References (characters and locations)", "Story", "Prompts",
                "Images", "Video, stitch and finals", "MiniMax files", "Latents", "Post-processing (per project and scene)",
                "Face fix", "Pipelines"]

INTRO = """# Agent API endpoints

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

"""


def _row(route: Dict, description: str) -> str:
    path = route["path"].replace(oa.API_PREFIX, "")
    notes: List[str] = []
    if route["status"] == 202:
        notes.append("**job**")
    if route["if_match"]:
        notes.append("If-Match")
    if route["query"]:
        notes.append("query: " + ", ".join(f"`{q}`" for q in sorted(route["query"])))
    if route["body_keys"]:
        notes.append("body: " + ", ".join(f"`{k}`" for k in sorted(route["body_keys"]) if k not in ("project_id", "scene_id")))
    suffix = f" _({'; '.join(n for n in notes if n and not n.endswith(': '))})_" if notes else ""
    return f"| `{route['method'].upper()}` | `{path}` | {description}{suffix} |"


def build_document() -> str:
    routes = oa.extract_routes()
    missing = []
    buckets: Dict[str, Dict[str, List[str]]] = {}
    for route in routes:
        path = route["path"].replace(oa.API_PREFIX, "")
        key = f"{route['method'].upper()} {path}"
        description = DESCRIPTIONS.get(key)
        if not description:
            missing.append(key)
            continue
        section = _section_for(path)
        group = ""
        if section in ("", "Projects"):
            section = "Project endpoints"
            group = _group(path)
        buckets.setdefault(section, {}).setdefault(group, []).append(_row(route, description))
    if missing:
        raise SystemExit("No description for:\n  " + "\n  ".join(missing))
    unknown = sorted(set(DESCRIPTIONS) - {f"{r['method'].upper()} {r['path'].replace(oa.API_PREFIX, '')}" for r in routes})
    if unknown:
        raise SystemExit("Description for a route that does not exist:\n  " + "\n  ".join(unknown))

    out = [INTRO, f"**Total:** {len(routes)} endpoints.\n"]
    table_head = "| Method | Path | What it does |\n|---|---|---|"
    for title in ["Service and discovery"]:
        out += [f"## {title}\n", table_head] + sum(buckets.get(title, {}).values(), []) + [""]
    out.append("## Project endpoints\n")
    groups = buckets.get("Project endpoints", {})
    for name in _GROUP_ORDER:
        rows = groups.get(name)
        if rows:
            out += [f"### {name}\n", table_head] + rows + [""]
    for title in ("Jobs", "LLM and prompt instructions", "Post-processing library", "Settings schema"):
        rows = sum(buckets.get(title, {}).values(), [])
        if rows:
            out += [f"## {title}\n", table_head] + rows + [""]
    return "\n".join(out).rstrip() + "\n"


def build_records() -> str:
    """Every endpoint as JSON (for the MCP server, which turns them into tools)."""
    records = []
    for route in oa.extract_routes():
        path = route["path"].replace(oa.API_PREFIX, "")
        key = f"{route['method'].upper()} {path}"
        records.append({
            "method": route["method"].upper(),
            "path": path,
            "description": DESCRIPTIONS[key],
            "path_params": route["path_params"],
            "query": sorted(route["query"]),
            "body_keys": sorted(route["body_keys"]),
            "job": route["status"] == 202,
            "if_match": bool(route["if_match"]),
        })
    return json.dumps({"endpoints": records}, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if Api_Endpoints.md is stale")
    args = parser.parse_args(argv)
    document = build_document()
    records = build_records()
    if args.check:
        current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.is_file() else ""
        current_json = JSON_PATH.read_text(encoding="utf-8") if JSON_PATH.is_file() else ""
        if current.replace("\r\n", "\n") != document or current_json.replace("\r\n", "\n") != records:
            print("[VRGDG] Api_Endpoints.md or agent_api/endpoints.json is stale. Run scripts/export_api_endpoints.py.")
            return 1
        return 0
    OUTPUT_PATH.write_text(document, encoding="utf-8")
    JSON_PATH.write_text(records, encoding="utf-8")
    print(f"[VRGDG] Wrote {OUTPUT_PATH} ({document.count(chr(10))} lines) and {JSON_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
