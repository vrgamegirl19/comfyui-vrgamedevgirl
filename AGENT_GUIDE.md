# 🤖 AI Agent Engineering Guide: comfyui-vrgamedevgirl

Welcome to **comfyui-vrgamedevgirl**. This repository is an enterprise-grade ComfyUI custom node suite and production application ecosystem. Its flagship tool is the **AI Video Builder**—a full digital audio workstation (DAW) and non-linear video editor (NLE) operating directly inside ComfyUI to produce AI music videos, narrative films, and cinematic sequences using engines like LTX-Video, MiniMax H3, FLUX, SDXL, and Z-Image.

This guide serves as the definitive technical manual for AI coding agents and human contributors. It provides an exhaustive map of the project architecture, detailed documentation of every active core file, guidelines for strict PEP 8 compliance, separation of concerns (SoC), and actionable recipes for extending the platform safely.

> [!IMPORTANT]
> **Operating the Video Builder through its API or MCP (not editing code)?** Read [Api_Endpoints.md](Api_Endpoints.md) first: every endpoint and what it does. Then follow [MUSIC_VIDEO_AGENT_PROMPT.md](MUSIC_VIDEO_AGENT_PROMPT.md), the step-by-step playbook for turning a song into a finished video. Connected MCP clients get both as resources (`vrgdg://docs/endpoints`, `vrgdg://docs/music-video-playbook`) and the playbook as the `make_music_video` prompt. See [Agent API Subsystem](#agent-api-subsystem-agent_api).

> [!NOTE]
> Per project directives, legacy and optional standalone modules located in `optional_nodes/` are intentionally omitted from this guide. All documentation here focuses on the active core runtime registered in [__init__.py](__init__.py).

---

## Table of Contents

1. [Architectural Overview & Runtime Lifecycle](#1-architectural-overview--runtime-lifecycle)
2. [Complete Project Map & File Reference](#2-complete-project-map--file-reference)
   - [Core Infrastructure (`core/`)](#core-infrastructure-core)
   - [AI Video Builder Engine (`builder/`)](#ai-video-builder-engine-builder)
   - [Workflow Runner & Graph Compiler (`runner/`)](#workflow-runner--graph-compiler-runner)
   - [Unified LLM Subsystem (`llm/`)](#unified-llm-subsystem-llm)
   - [MiniMax H3 Video Engine (`minimax/`)](#minimax-h3-video-engine-minimax)
   - [Post-Processing & Enhancement (`post_process/`)](#post-processing--enhancement-post_process)
   - [Storyboard Subsystem (`storyboard/`)](#storyboard-subsystem-storyboard)
   - [Prompt Creator Subsystem (`prompt_creator/`)](#prompt-creator-subsystem-prompt_creator)
   - [General Custom Nodes (`general/`)](#general-custom-nodes-general)
   - [Browser AI Automation (`browser/` & `flow_automation/`)](#browser-ai-automation-browser--flow_automation)
   - [Frontend Web Applications (`web/`)](#frontend-web-applications-web)
   - [Agent API Subsystem (`agent_api/`)](#agent-api-subsystem-agent_api)
   - [Model Context Protocol Server (`mcp_server/`)](#model-context-protocol-server-mcp_server)
   - [Utility Scripts (`scripts/`)](#utility-scripts-scripts)
   - [Test Suites (`tests/`)](#test-suites-tests)
3. [Separation of Concerns (SoC) Principles](#3-separation-of-concerns-soc-principles)
4. [PEP 8 Coding Standards & Repository Conventions](#4-pep-8-coding-standards--repository-conventions)
5. [Developer Implementation Recipes](#5-developer-implementation-recipes)
   - [Recipe 1: Creating a New ComfyUI Custom Node](#recipe-1-creating-a-new-comfyui-custom-node)
   - [Recipe 2: Registering a Backend HTTP Route](#recipe-2-registering-a-backend-http-route)
   - [Recipe 3: Adding or Extending a Workflow Runner Pipeline](#recipe-3-adding-or-extending-a-workflow-runner-pipeline)
   - [Recipe 4: Modifying Builder Frontend Modules](#recipe-4-modifying-builder-frontend-modules)
   - [Recipe 5: Writing Unit & Integration Tests](#recipe-5-writing-unit--integration-tests)
6. [Critical Invariants & Anti-Patterns to Avoid](#6-critical-invariants--anti-patterns-to-avoid)

---

## 1. Architectural Overview & Runtime Lifecycle

### Dual Execution Paradigms

This codebase operates under two distinct paradigms:

```
+-----------------------------------------------------------------------------------+
|                                 ComfyUI Process                                   |
|                                                                                   |
|  +-------------------------------------+   +-----------------------------------+  |
|  |     Canvas Node Graph Execution     |   |   Headless API Prompt Execution   |  |
|  |   (Standard ComfyUI Canvas Mode)    |   |     (Video Builder Subsystem)     |  |
|  |                                     |   |                                   |  |
|  | - User connects nodes on canvas.    |   | - Web UI runs inside modal window.|  |
|  | - Execution scheduled via queue.    |   | - Server compiles workflow graph. |  |
|  | - Modules: general, minimax, llm   |   | - Dispatches to ComfyUI /prompt.  |  |
|  +-------------------------------------+   +-----------------------------------+  |
|                     ^                                        ^                    |
|                     |                                        |                    |
|  +------------------+----------------------------------------+-----------------+  |
|  |                          __init__.py Entry Point                            |  |
|  |  - Iterates _VRGDG_SUBMODULES tuple.                                        |  |
|  |  - Dynamically imports modules with fault tolerance.                       |  |
|  |  - Merges NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.               |  |
|  |  - Exports WEB_DIRECTORY = "./web" for browser asset auto-registration.    |  |
|  |  - Binds HTTP route listeners to server.PromptServer.instance.routes.       |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

1. **Canvas Node Execution**: Custom nodes (e.g., `VRGDG_ShowText`, `VRGDG_LLM_Multi`, `H3FastVAEDecode`, `VRGDG_AudioCrop`) are placed directly onto the ComfyUI canvas, connected with noodles, and executed by ComfyUI's standard execution scheduler.
2. **Headless API Graph Compilation**: The AI Video Builder frontend ([web/music_video_builder/](web/music_video_builder)) acts as an integrated production studio. Instead of requiring users to wire up dozens of complex nodes manually, the builder dispatches HTTP commands to [runner/routes.py](runner/routes.py). The backend dynamically compiles complete execution graphs (in ComfyUI `/prompt` API format via [runner/api_graph.py](runner/api_graph.py)), sends them to ComfyUI's internal queue, streams progress back via WebSockets, and captures rendered frames or video clips.

### Initialization Sequence

1. ComfyUI discovers this custom node folder and executes [__init__.py](__init__.py).
2. `_VRGDG_SUBMODULES` is iterated. Each submodule (e.g., `.builder.nodes`, `.runner.nodes`, `.core.system_routes`) is imported.
3. Submodule import triggers route attachment to `server.PromptServer.instance.routes`.
4. Node dictionaries (`NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`) are merged into the global scope. Collisions trigger console warnings.
5. ComfyUI serves static web assets from `WEB_DIRECTORY = "./web"`.

### Project Data Organization

Every Video Builder project resides in a dedicated directory on disk:

```
<project_root>/
├── vrgdg_builder_session.json   # Core project state (scenes, timeline, parameters, active models, revision)
├── builder_segments.srt         # Master SRT subtitle timing file
├── SceneNotes.json              # Per-scene prompt notes and instructions
├── project_context/             # Context files (full_lyrics.txt, ConceptPrompts.txt, I2VMotionNotes.txt)
│   └── full_lyrics.txt
├── latents/                     # Serialized MiniMax H3 latent tensors (scene_001.latent)
│   └── scene_001.latent
├── zimage_approved/             # Approved start frame images for scenes (image_0001.png)
├── scene_image_previews/        # Candidate image generation previews
│   └── scene_0001/
├── rendered_scene_videos/       # Rendered video clips per scene (video_0001-audio.mp4)
├── rendered_scene_videos_backup/# Backup of previous renders when overwriting
├── scene_audio/                 # Extracted audio clips per scene
├── scene_audio_trimmed/         # Trimmed audio clips matching scene durations
├── render_logs/                 # MiniMax / LTX render logs and diagnostics
├── session_backups/             # Automated snapshots of session state
└── removed_scene_assets/        # Quarantined assets from deleted scenes
```

---

## 2. Complete Project Map & File Reference

### Directory Overview Table

| Directory | Primary Responsibility | Key Files |
| :--- | :--- | :--- |
| [core/](core) | Shared primitives: atomic file writes, wildcard sockets, model directory resolution, resource monitoring | `atomic_write.py`, `any_type.py`, `model_paths.py`, `system_routes.py` |
| [builder/](builder) | AI Video Builder backend: session persistence, project branching, audio beat detection, media indexing, routes | `project.py`, `audio.py`, `media.py`, `paths.py`, `routes.py`, `ui_profiles.py`, `video_profiles.py`, `nodes.py` |
| [runner/](runner) | Dynamic workflow graph generation and rendering engine for LTX, MiniMax H3, Z-Image, Flux | `api_graph.py`, `ltx_workflows.py`, `minimax_workflows.py`, `routes.py` |
| [llm/](llm) | Multi-provider LLM integrations (GGUF, API, Google), prompt expansion, agent chat, JSON validation | `api.py`, `gguf.py`, `builder_agent.py`, `image_prompt_generation.py` |
| [minimax/](minimax) | MiniMax H3 video pipeline: latent caching, frame-token math, latent continuation, fast VAE decoding | `latent_manager.py`, `latent_continuation.py`, `latent_upscaler.py`, `tile_plan.py`, `resolution.py`, `settings_payload.py`, `scene_inputs.py`, `nodes.py` |
| [post_process/](post_process) | Face tracking, anchor enhancement, face paste-back compositing, 3D LUT grading, film grain | `face_fix.py`, `luts.py`, `lut_video_tools.py` |
| [storyboard/](storyboard) | Storyboard generation, three-act structure planning, scene beats, dialogue allocation | `story_layer.py`, `scene_prompts.py`, `dialogue_scenes.py`, `nodes.py` |
| [prompt_creator/](prompt_creator) | Structured prompt brainstorming, concept maps, motion notes, draft persistence routes | `nodes.py` |
| [general/](general) | General-purpose canvas nodes: lyrics extraction (stable-ts), video analysis, audio stems, LoRA utilities | `lyrics.py`, `video.py`, `audio.py`, `utility.py`, `ltx_msr_reference.py` |
| [browser/](browser) | External browser automation nodes bridging Flow, Meta AI, and ChatGPT image generators into ComfyUI | `nodes.py` |
| [flow_automation/](flow_automation) | Node.js / Playwright / Puppeteer automation scripts for browser-driven generation workflows | `flow-poc.mjs`, `manual-bridge.mjs`, `meta-ai-poc.mjs` |
| [web/](web) | ComfyUI web extensions, Video Builder application (79 ESM modules), Storyboard UI (22 ESM modules) | `music_video_builder/`, `storyboard_builder/`, `VRGDG_*.js` |
| [agent_api/](agent_api) | Headless REST API (`/vrgdg/api/v1`), transactional mutations, job management, SSE events, orchestrator | `router.py`, `mutations.py`, `jobs.py`, `envelope.py`, `paths.py`, `orchestrator/` |
| [mcp_server/](mcp_server) | Local-only (git-ignored) zero-dependency MCP server (JSON-RPC 2.0 stdio): 63 named tools plus one generated `api_*` tool per other endpoint and `api_request`, doc resources, prompt templates | `__main__.py`, `server.py`, `tools.py`, `endpoint_tools.py`, `resources.py`, `prompts.py`, `client.py` |
| [scripts/](scripts) | Standalone tooling, workflow generation scripts, backport utilities, Photoshop integration, contract exporters | `build_minimax_h3_ref2va_2pass_audio_api.py`, `far_face_repair_backend.py`, `export_openapi.py`, `export_api_endpoints.py`, `export_minimax_defaults.mjs` |
| [tests/](tests) | 84 test suites covering Python backend logic, node collision checks, and JavaScript UI contracts | `test_node_registration.py`, `test_atomic_write.py`, `builder_source.py` |

---

### Core Infrastructure (`core/`)

The `core` package houses critical cross-cutting utilities used across the entire codebase.

#### [core/atomic_write.py](core/atomic_write.py)
- **Purpose**: Provides crash-safe atomic file writing. Prevents corruption of project files, session state, and settings if a crash, timeout, or power disruption occurs mid-write.
- **Key Functions**:
  - `atomic_write_text(path, content, encoding="utf-8")`: Writes content to a hidden sibling temporary file (`.filename.tmp`) using `tempfile.mkstemp`, performs `handle.flush()` and `os.fsync()`, and replaces the target path atomically via `os.replace()`. Cleans up temporary files if an exception is raised.
  - `atomic_write_json(path, value)`: Formats data with `json.dumps(value, indent=2, ensure_ascii=False)` and calls `atomic_write_text`.
- **Usage Rule**: **Never** use raw `open(path, 'w')` when writing state or configuration files. Always use `atomic_write_json` or `atomic_write_text`.

#### [core/any_type.py](core/any_type.py)
- **Purpose**: Implements the ComfyUI wildcard socket type pattern.
- **Key Symbols**:
  - `class AnyType(str)`: Overrides `__ne__(self, value)` to always return `False`.
  - `any_typ = AnyType("*")`: The singleton wildcard socket instance. When assigned to node inputs or outputs, it connects to any ComfyUI slot type regardless of type checking.

#### [core/model_paths.py](core/model_paths.py)
- **Purpose**: Manages custom model directory registration, persistence, and discovery outside the standard ComfyUI `models/` directory.
- **Key Functions**:
  - `load_custom_model_root()`: Reads `custom_model_root.json` from `VRGDG_Model_Defaults`.
  - `save_custom_model_root(value)`: Atomically writes a new custom model root directory.
  - `register_custom_model_root(root=None)`: Recursively scans and registers subfolders (`diffusion_models`, `unet`, `text_encoders`, `clip`, `vae`, `loras`, `upscale_models`, `latent_upscale_models`, `LLM`) into ComfyUI's central resolver using `folder_paths.add_model_folder_path`.

#### [core/system_routes.py](core/system_routes.py)
- **Purpose**: Implements server-level system telemetry, GPU monitoring, process introspection, and memory management routes.
- **Key Endpoints**:
  - `GET /vrgdg/resource-monitor`: Queries host RAM (via `psutil`) and GPU metrics (via `nvidia-smi` without console popups) including utilization, VRAM usage, temperature, fan speed, clock frequencies, and power draw. Readings are protected by an async mutex and throttled to 1-second intervals.
  - `POST /vrgdg/resource-monitor/clear-memory`: Forces aggressive memory reclamation: runs Python `gc.collect()`, PyTorch `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()`, ComfyUI model cache clearing, and logs exact before/after RSS memory deltas.
  - `GET /vrgdg/update/v10/status` & `POST /vrgdg/update/v10`: Provides self-updating mechanisms for the node pack.
  - `GET /vrgdg/video_builder/custom_nodes/status` & `POST /vrgdg/video_builder/custom_nodes/install`: Allowlist-based dependency checker and installer for prerequisite custom nodes.

---

### AI Video Builder Engine (`builder/`)

The `builder` package contains the backend business logic and HTTP API powering the **AI Video Builder UI**.

#### [builder/nodes.py](builder/nodes.py)
- **Purpose**: Defines the primary canvas node `VRGDG_MusicVideoBuilderUI` and triggers the registration of builder server routes.
- **Node Registered**:
  - Class: `VRGDG_MusicVideoBuilderUI`
  - Display Name: `VRGDG Music Video Builder UI`
  - Category: `VRGDG/UI`
  - Inputs: `audio_path`, `project_folder`, `session_path`, `srt_path`
  - Outputs: `(project_folder, session_path, srt_path)`

#### [builder/routes.py](builder/routes.py)
- **Purpose**: The central API gateway for the Video Builder frontend. Implements over 50 async REST endpoints on `server.PromptServer.instance.routes`.
- **Key Route Categories**:
  - **Audio & Beat Analysis**: `/vrgdg/music_builder/analyze_audio`, `/vrgdg/music_builder/import_capcut_beats`, `/vrgdg/music_builder/save_scene_audio`, `/vrgdg/music_builder/trim_scene_audio`, `/vrgdg/music_builder/create_silent_audio`, `/vrgdg/music_builder/prepare_scene_audio_mix`.
  - **Session & Project State**: `/vrgdg/music_builder/save_session`, `/vrgdg/music_builder/load_session`, `/vrgdg/music_builder/new_project`, `/vrgdg/music_builder/save_project_as`, `/vrgdg/music_builder/delete_project`, `/vrgdg/music_builder/list_projects`, `/vrgdg/music_builder/export_project`, `/vrgdg/music_builder/import_project`.
  - **Media Asset Management**: `/vrgdg/music_builder/save_scene_image`, `/vrgdg/music_builder/archive_scene_image`, `/vrgdg/music_builder/extract_video_final_frame`, `/vrgdg/music_builder/scan_scene_videos`, `/vrgdg/music_builder/restore_scene_video`.
  - **Latent Lifecycle**: `/vrgdg/music_builder/latent_status`, `/vrgdg/music_builder/check_latent_predecessor`, `/vrgdg/music_builder/delete_scene_latent`, `/vrgdg/music_builder/list_dirty_latents`.
  - **Timeline Asset Renumbering**: `/vrgdg/music_builder/renumber_scenes_after_removal`, `/vrgdg/music_builder/renumber_scenes_after_insert`.
  - **Video Profiles**: `/vrgdg/music_builder/list_video_profiles`, `/vrgdg/music_builder/load_video_profile`, `/vrgdg/music_builder/save_video_profile` (409 with `exists` when the name is taken and `overwrite` is not set), `/vrgdg/music_builder/delete_video_profile`. Backed by `builder/video_profiles.py`, which also remembers the profile chosen last (`/vrgdg/music_builder/set_last_video_profile`, and `last` in the list response).
  - **UI Layout Profiles**: `/vrgdg/music_builder/list_ui_profiles`, `/vrgdg/music_builder/load_ui_profile`, `/vrgdg/music_builder/set_last_ui_profile`, `/vrgdg/music_builder/save_ui_profile` (409 with `exists` when the name is taken and `overwrite` is not set), `/vrgdg/music_builder/update_ui_profile_layout`, `/vrgdg/music_builder/delete_ui_profile`. Backed by `builder/ui_profiles.py`.
  - **LLM Prompt Generation & Agent**: `/vrgdg/music_builder/generate_t2i`, `/vrgdg/music_builder/generate_i2v`, `/vrgdg/music_builder/generate_chained_i2v`, `/vrgdg/music_builder/generate_t2v`, `/vrgdg/music_builder/agent_chat`, `/vrgdg/music_builder/flux_reference_extract_subjects`, `/vrgdg/music_builder/flux_reference_extract_locations`.

#### [builder/video_profiles.py](builder/video_profiles.py)
- **Purpose**: Named MiniMax H3 video profiles shared by every project. A profile is the user's video type, render pass and every setting that belongs to them, saved as one JSON file under `<ComfyUI output>/VRGDG_Video_Profiles/minimax_h3/`. `EXCLUDED_PROFILE_KEYS` lists what a profile never carries (audio mode, between-scene continuity, the project's Standard / RefMod pipeline, the per-pass cache and settings-version markers); the server filters on save and again on load. The UI is `web/music_video_builder/video_profiles.mjs` (the row above the video type buttons); it applies a profile through the same steps as the video type and pass buttons. Profiles are a Builder feature and are not part of the Agent API.

#### [builder/ui_profiles.py](builder/ui_profiles.py)
- **Purpose**: Named Video Builder layout profiles shared by every project, independent of the video profiles. A profile stores `left_collapsed`, `right_collapsed`, `left_panel_width`, `right_panel_width`, `timeline_panel_height` and the floating LLM Prompting window's `llm_popout_open`, `llm_popout_width`, `llm_popout_height`, `llm_popout_x` and `llm_popout_y` as one JSON file under `<ComfyUI output>/VRGDG_UI_Profiles/music_video_builder/`, and `_last_selected.json` there remembers the profile chosen last. `normalize_layout` clamps every size on save and load. The UI is `web/music_video_builder/ui_profiles.mjs` (the UI Layout selector next to Video Type): the last profile loads when the Builder starts and wins over a project's own saved sizes, and while one is selected every layout change (side panel tab, panel drag, timeline drag) is written back to it. Builder feature only, not part of the Agent API.
- **LLM Prompting window**: `web/music_video_builder/llm_popout.mjs` opens a floating window (a checkbox at the top of the right panel) with mirrors of the MiniMax H3 prompt, the Save Updated Prompt button, the character count line and the 2nd pass prompt. The right panel and the Builder grid are not changed. Typing in a mirror sets the real field and fires its `input` event, the mirror's button clicks the real Save button, and a 200 ms timer copies the panel's values, status text and button state back, because panel code sets values without events. The window is a fixed-position element inside the Builder overlay, so it closes with the Builder. Its open state, size and screen position are saved with the project and the UI layout (`llm_popout_open`, `llm_popout_width`, `llm_popout_height`, `llm_popout_x`, `llm_popout_y`).

#### [builder/project.py](builder/project.py)
- **Purpose**: Project serialization, initialization, asset renumbering, and migration logic.
- **Key Functions**:
  - `_new_builder_project(payload)`: Sets up directory scaffolding (`zimage_approved`, `prompts`, `project_context`, `latents`, etc.) and writes the initial `vrgdg_builder_session.json`.
  - `_load_builder_session(payload)` / `_save_builder_session(payload)`: Reads and atomically persists project scene data, timeline markers, and render flags.
  - `_renumber_scene_assets_after_insert(project_folder, inserted_index)`: Renumbers all disk assets (`image_NNNN.*`, `video_NNNN.*`, etc.) backwards from the end to make room for an inserted scene without overwriting existing files.
  - `_renumber_scene_assets_after_removal(project_folder, removed_index)`: Shifts all subsequent assets forward by one to close the gap left by a deleted scene.

#### [builder/project_copy.py](builder/project_copy.py)
- **Purpose**: Non-destructive project duplication and branching.
- **Key Functions**:
  - `_save_builder_project_as(payload)`: Duplicates a project folder, selectively filters scene media based on user choices (e.g., keep approved images only, discard failed video renders), rewrites internal path references inside `vrgdg_builder_session.json`, and clones serialized MiniMax latents.

#### [builder/audio.py](builder/audio.py)
- **Purpose**: Digital audio processing, waveform analysis, and beat detection.
- **Key Functions**:
  - `_read_audio_peaks(audio_path, target_peaks=1600)`: Extracts downsampled waveform peak envelopes for high-performance frontend timeline rendering.
  - `_estimate_beats_from_audio(audio_path, ...)`: Leverages `librosa` to compute audio onset envelopes, extract tempo (BPM), and determine precise beat frame timings.
  - `_convert_audio_to_wav(audio_path, target_path)`: Converts incoming audio formats (MP3, M4A, FLAC) to uncompressed 16-bit 44.1/48kHz WAV via `av` or `ffmpeg`.
  - `_trim_scene_audio(...)` & `_prepare_scene_audio_mix(...)`: Slices audio segments matching scene start/duration parameters and mixes final multi-track audio for video assembly.

#### [builder/media.py](builder/media.py)
- **Purpose**: Visual asset tracking, frame extraction, and reference media management.
- **Key Functions**:
  - `_extract_video_final_frame_as_scene_image(video_path, target_image_path)`: Uses `torchcodec` or `cv2` to grab the exact final frame of a rendered scene video to use as the starting frame of the subsequent scene (First/Last Frame continuity).
  - `_archive_scene_image(project_folder, scene_num)`: Moves superseded scene images to a history folder before new iterations overwrite them.
  - `_scan_builder_scene_videos(project_folder)`: Traverses scene video directories and indexes render versions, timestamps, and resolutions.

#### [builder/paths.py](builder/paths.py)
- **Purpose**: Path sanitation, directory containment checks, native OS dialog integration.
- **Key Functions**:
  - `_resolve_existing_file(path, label)`: Validates that a file exists and normalizes Windows/POSIX path separators.
  - `_open_native_picker(type="file", ...)`: Spawns the operating system's native file/folder explorer dialog.
  - `_open_local_file(path)`: Launches the default system media player or image viewer for a rendered asset.

#### [builder/video_editor.py](builder/video_editor.py)
- **Purpose**: Static asset delivery endpoints for high-throughput video/image timeline streaming.
- **Key Endpoints**:
  - `GET /vrgdg/video_editor/video?path=<path>`: Streams video files with `Cache-Control: public, max-age=31536000, immutable` headers keyed to scene cache busters to enable instant browser timeline playback without re-fetching.
  - `GET /vrgdg/video_editor/image?path=<path>`: Serves image files and thumbnails.

---

### Workflow Runner & Graph Compiler (`runner/`)

The `runner` package is the code generation and execution engine that converts project settings into executable ComfyUI workflow prompt graphs.

#### [runner/nodes.py](runner/nodes.py)
- **Purpose**: Canvas UI nodes for workflow runners and system memory clearing.
- **Nodes Registered**:
  - `VRGDG_MiniMaxH3TurboLoRACompat`: "VRGDG MiniMax-H3 Turbo LoRA Compatibility"
  - `VRGDG_ZImageWorkflowRunnerUI`: "VRGDG Z-Image Workflow Runner UI"
  - `VRGDG_ClearMemoryButtonUI`: "VRGDG Clear Memory Button"

#### [runner/routes.py](runner/routes.py)
- **Purpose**: Compiles requested workflows and coordinates background generation jobs.
- **Key Endpoints**:
  - `/vrgdg/workflow_runner/build_zimage_prompt`: Assembles Z-Image / SDXL / Flux prompt graphs.
  - `/vrgdg/workflow_runner/build_i2v_prompt` & `build_t2v_prompt`: Assembles LTX-Video generation graphs.
  - `/vrgdg/workflow_runner/build_minimax_h3_prompt`, `build_minimax_h3_2pass_prompt`, `build_minimax_h3_advanced_2pass_prompt`: Assembles single- and multi-pass MiniMax H3 generation graphs.
  - `/vrgdg/workflow_runner/build_flf_prompt`: Assembles First/Last Frame guided interpolation graphs.
  - `/vrgdg/workflow_runner/collect_scene_video`: Locates rendered outputs in ComfyUI output folders and imports them into the project.
  - `/vrgdg/workflow_runner/match_scene_video_start_color`: Matches output video start frames to source image color profiles to prevent color shifts.
  - `/vrgdg/workflow_runner/stitch_scene_videos`: Invokes `ffmpeg` to stitch all approved scene video clips into a single continuous video with muxed audio.

#### [runner/api_graph.py](runner/api_graph.py)
- **Purpose**: Graph construction builder primitives. Generates unique string node IDs, links inputs between nodes, and formats the output into the standard ComfyUI API schema: `{node_id: {"class_type": ..., "inputs": {...}}}`.

#### [runner/ltx_workflows.py](runner/ltx_workflows.py)
- **Purpose**: Compiler for LTX-Video pipelines (Text-to-Video, Image-to-Video, First/Last Frame guidance, STG guidance, frame rate, and aspect ratio conditioning).

#### [runner/minimax_workflows.py](runner/minimax_workflows.py)
- **Purpose**: Compiler for MiniMax H3 pipelines. Assembles model loading, text conditioning, image reference attachment, audio drive conditioning, latent upscale stages, and fast VAE decoding.

#### [runner/minimax_inputs.py](runner/minimax_inputs.py)
- **Purpose**: Input parsing, validation, and token/frame duration alignment specifically for MiniMax H3 executions.
- **I2V 2 Pass**: selects `minimax_i2v_audio_driven_builder_latent_upscale_2pass_api.json` in `Workflows/UsedForUIDoNotTouch`. The shared two-pass compiler applies the same video settings as Reference to Video, using FL2VA/FL2V model and LoRA choices and separate first-frame conditioning at each pass's resolution. Input audio only; 2 Pass Advanced is not offered for I2V.

#### [runner/minimax_patches.py](runner/minimax_patches.py)
- **Purpose**: Injects optional patches (such as Turbo LoRA or camera motion control weights) into MiniMax H3 model nodes.

#### [runner/image_workflows.py](runner/image_workflows.py)
- **Purpose**: Compiles prompt graphs for image generators (Flux Schnell/Dev, Z-Image, SDXL) with support for reference conditioning images.

#### [runner/utility_workflows.py](runner/utility_workflows.py)
- **Purpose**: Compiles background helper workflows such as Whisper speech-to-text transcription and standalone memory cleanup prompts.

#### [runner/video_files.py](runner/video_files.py)
- **Purpose**: Output file handling, video trimming, format verification, and ffmpeg assembly.

#### [runner/models.py](runner/models.py)
- **Purpose**: Dataclasses and type definitions representing model configurations, samplers, schedulers, and resolution settings.

---

### Unified LLM Subsystem (`llm/`)

The `llm` package provides unified text generation, prompt rewriting, concept generation, and interactive agent capabilities across multiple backends.

#### [llm/api.py](llm/api.py)
- **Purpose**: Unified multi-provider API client node. Supports OpenAI, Anthropic Claude, Google Gemini, Grok (xAI), Ollama, and LM Studio.
- **Node Registered**:
  - `VRGDG_LLM_Multi`: "🤖 VRGDG LLM Multi 🤖"
- **Features**: Handles multimodal image inputs, system instructions, temperature/seed control, structured JSON schema enforcement, and retry loops.

#### [llm/gguf.py](llm/gguf.py)
- **Purpose**: Direct local GGUF model execution using `llama-cpp-python` with CUDA acceleration.
- **Nodes Registered**:
  - `VRGDG_QwenGGUF`: "🧠 VRGDG Qwen GGUF 🧠"
  - `VRGDG_SuperGemmaGGUFChat`: "🧠 VRGDG SuperGemma GGUF Chat 🧠"
  - `VRGDG_UnloadGemmaModels`: "VRGDG Unload Gemma/GGUF Models"

#### [llm/google.py](llm/google.py)
- **Purpose**: Google Gemini and Imagen integration.
- **Node Registered**:
  - `VRGDG_NanoBananaPro`: "🚀 VRGDG NanoBanana Pro 🚀"

#### [llm/builder_agent.py](llm/builder_agent.py)
- **Purpose**: Conversational AI assistant logic embedded inside the Video Builder UI. Handles user queries about project planning, scene direction, prompt critique, and shot progression.

#### [llm/builder_instructions.py](llm/builder_instructions.py)
- **Purpose**: Manages system prompts, persona templates, and instruction presets for image and video prompt generation.

#### [llm/builder_runner.py](llm/builder_runner.py)
- **Purpose**: Dispatch router for builder LLM tasks. Dynamically routes requests to LM Studio, local GGUF, or cloud APIs with automatic fallback and seed retry.

#### [llm/image_prompt_generation.py](llm/image_prompt_generation.py)
- **Purpose**: Domain-specific prompt expansion for image models (Flux, Z-Image, SDXL). Generates consistent character descriptions, architectural lighting details, and color palettes from raw lyrics or concept notes.

#### [llm/video_prompt_generation.py](llm/video_prompt_generation.py)
- **Purpose**: Domain-specific prompt expansion for video models (LTX, MiniMax). Crafts dynamic camera motions (pan, tilt, crane, dolly, orbit), action beats, physical dynamics, and scene transitions.

#### [llm/output_checks.py](llm/output_checks.py)
- **Purpose**: Resilient JSON parsing and validation. Extracts JSON payloads enclosed in markdown code fences, fixes trailing commas, validates schema fields, and recovers gracefully from truncated responses.

#### [llm/text_cleaning.py](llm/text_cleaning.py)
- **Purpose**: Strips conversational preambles, trailing commentary, markdown formatting, and hallucinated prompt tags from raw LLM outputs.

#### [llm/cache.py](llm/cache.py)
- **Purpose**: In-memory and disk-based caching for prompt expansions to minimize redundant API costs.

---

### MiniMax H3 Video Engine (`minimax/`)

The `minimax` package implements high-performance conditioning, latent management, and decoding nodes for the MiniMax H3 video architecture.

- `minimax/vae_decode.py` owns batched spatial VAE decoding; `H3FastVAEDecode` delegates to it. Its seam blending follows the installed stock VAE's policy, including newer ComfyUI versions that blend already-composited overlap strips.
- I2V stores its Single/2 Pass caches in `i2v_pass_profiles`, separate from `ref_pass_profiles`. `i2v_pass_settings_version` migrates older I2V saves to Single because their saved `render_pass` was previously ignored. Keep these migrations and mode-specific defaults synchronized in `minimax_h3.mjs` and `minimax/settings_payload.py`.
- Image to Video and Image + Reference use per-scene frames and do not support between-scene continuity. Hide the continuity settings section in these modes; show continuation direction/timing only while final-frame prompt continuation is active. Keep frontend eligibility and `minimax/scene_inputs.py` synchronized, and ignore saved continuity modes when building image-mode render payloads.
- FLF transition style and optional direction are shared MiniMax settings (`i2v_transition_style`, `i2v_transition_direction`), inherited globally unless the existing scene settings lock is enabled. `minimax_h3.mjs` owns presets/default normalization; `minimax_i2v_transition.mjs` owns creative prompt guidance; keyframe UI controls save through the existing panel and event modules. Normal I2V ignores these settings. Changing them requires regenerating Create Prompt and does not change frame conditioning or sampler settings. Transition settings stay shared when switching Single / 2 Pass.
- I2V Normal / FLF is stored per scene as `minimax_h3_i2v_frame_mode`; the selected scene image is the first frame and `first_last_frame_end_image_path` is its independent last frame. Legacy last images imply FLF until an explicit mode is saved. `minimax_keyframe_state.mjs` owns eligibility/validation, and `minimax_keyframes.mjs` owns the pickers/previews. Existing image generators and scene image history supply frame choices. Both passes accept these same inputs. `runner/minimax_keyframes.py` inserts `VRGDG_MiniMaxH3KeyframeTiming`, whose service in `minimax/keyframes.py` aligns conditioning to the range retained by exact scene trimming without copying latent tensors. Normal ignores saved last images; FLF requires both frames. No previous-scene frame resolution is used.
- MiniMax I2V FLF uses the existing A → B timeline thumbnail layout. Eligibility uses the effective MiniMax mode for that scene and `miniMaxI2VFLFEnabled`; its thumbnail sources are the selected scene image and explicit last image, never LTX chaining resolvers. Keyframe edits repaint timeline segments directly as well as the scene list.

#### [minimax/latent_manager.py](minimax/latent_manager.py)
- **Purpose**: Serialized latent storage and frame-token math for MiniMax H3. Eliminates pixel-space VAE re-encoding drift across chained scene passes.
- **Key Symbols**:
  - `_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`: MiniMax H3 temporal latent token compression pattern.
  - `_tokens_to_frames(token_count)` & `_frames_to_tokens(frame_count)`: Performs exact conversions between video frame counts and temporal latent tokens.
  - `class SceneLatentManager`: Manages `.latent` safetensors files in `latents/`, tracks dirty flags, handles predecessor dependencies, and renames latent files when scenes are reordered.
  - `scene_latent_manager`: Singleton instance.

#### [minimax/nodes.py](minimax/nodes.py)
- **Purpose**: Core canvas nodes for MiniMax H3 generation.
- **Nodes Registered**:
  - `H3FastVAEDecode`: "H3 VAE Decode Fast (Batched Tiles)" — Memory-efficient tiled VAE decoder preventing out-of-memory errors on long clips.
  - `VRGDG_MiniMaxH3AudioDrive`: "VRGDG MiniMax H3 Audio Drive" — Injects audio waveform conditioning into MiniMax H3 to drive motion/sync.
  - `VRGDG_MiniMaxH3ReferenceMediaFromPaths`: "VRGDG MiniMax H3 Reference Media From Paths" — Binds character and background reference images.
  - `VRGDG_MiniMaxH3ImageReferenceToVideo`: "MiniMax H3 Image + Reference to Video" — High-level conditioning node for Image-to-Video with multiple reference images.

#### RefMod pipeline (`minimax/refmod_*.py`, `runner/minimax_refmod.py`)
- **Purpose**: A project-wide pipeline (`minimax_h3_settings.pipeline == "refmod"`) that renders scenes from saved RefMods (`models/refmods/<type>/<name>.safetensors`, made in RefMods Studio) instead of reference images. Full design: [docs/REFMOD_INTEGRATION_SPEC.md](docs/REFMOD_INTEGRATION_SPEC.md).
- **Files**:
  - `minimax/refmod_picker.py`: file-dialog image picker, describe prompts per type, metadata and combine nodes.
  - `minimax/refmod_studio.py`: create a RefMod from images (crops, quality presets, fixed folder per type) and save a Reference Builder image as a RefMod.
  - `minimax/refmod_library.py`: list saved RefMods and their previews (`/vrgdg/refmod/library`, `/vrgdg/refmod/preview`).
  - `minimax/refmod_scene.py`: which RefMods a scene uses, their order and the `<Video n>` / `<Picture n>` / `<Audio n>` labels. Twin of `web/music_video_builder/refmod_labels.mjs`; both are checked against `tests/refmod_scene_cases.json`.
  - `runner/minimax_refmod.py`: rewires a built single or 2 pass graph so the guiders read *Text Encode with RefMods* (needs ComfyUI-MiniMaxH3Mod). The scene audio becomes an audio RefMod so `<Audio 1>` still exists.
  - `token_report` (`refmod_scene.py`) / `tokenReport` (`refmod_labels.mjs`): scene token total (warns above 6,000) and the character balance (a character under half the strongest by tokens x strength tends to be duplicated). The panel status line and the render log use it.
  - Storyboard: `_refmod_card_fields` (`storyboard/scene_helpers.py`) and `refmodCardFields` keep RefMod fields on storyboard cards; `gpt_payload.mjs` adds `refmod_label` per subject and a `refmod_pipeline` block when the Builder opens the storyboard with `refmodPipeline`.
- **Rules**: one mode (reference_to_video), single or 2 pass only, no scene images. Settings normalisation keeps those rules in both `minimax_h3.mjs` and `minimax/settings_payload.py`.

#### [minimax/latent_continuation.py](minimax/latent_continuation.py)
- **Purpose**: Native latent-space continuation between adjacent scenes.
- **Nodes Registered**:
  - `VRGDG_MiniMaxH3SaveLatent`: "VRGDG H3 Save Latent"
  - `VRGDG_MiniMaxH3LoadLatent`: "VRGDG H3 Load Latent"
  - `VRGDG_MiniMaxH3ApplyLatentGuide`: "VRGDG H3 Apply Latent Continuation Guide"
  - `VRGDG_MiniMaxH3LoadExactFrame`: "VRGDG H3 Load Exact Last Frame"
  - `VRGDG_MiniMaxH3ApplyMaskedContinuation`: "VRGDG H3 Apply Masked Continuation"
- **Latent Continuation Masked** (`continuity_mode` = `latent_continuation_masked`): Single pass, 2 Pass and 2 Pass Advanced. `Load Latent` (`masked_av`) slices a phase-aligned window from `plan_masked_context` (39/90/141/192 frames, starts on a 5-token boundary, ends before the predecessor's padding). `Apply Masked Continuation` copies it into the head of the sampler's input latent and zeroes the denoise mask there (needs ComfyUI PR 15375, v0.34.0+). Nothing is added to the conditioning. With Audio Drive the song audio stays locked and only the video head is protected, with built-in audio the predecessor's audio ticks are copied too. The head is trimmed through the timing plan's warm-up (`_patch_minimax_h3_latent_continuation_masked` in `runner/minimax_patches.py`). In 2 Pass the learned upscale resets the video mask on purpose, so the same window is applied again to the second sampler (the predecessor is saved at the pass 2 size, so that head is exact). In 2 Pass Advanced the second pass runs inside MMH3 Ultimate Upscale, which builds its own masks, so only pass 1 is protected. Any other sampler layout raises a clear error.

#### [minimax/tile_plan.py](minimax/tile_plan.py) and [minimax/resolution.py](minimax/resolution.py)
- **Purpose**: Pure helpers (no ComfyUI or torch imports). `resolution.py` is the one output resolution shared by single pass, 2 Pass and 2 Pass Advanced (`resolution_preset` + `megapixels`, plus the migration of older saves); its JS twins are `miniMaxH3FrameSize` and `miniMaxH3PresetMegapixels` in `minimax_h3.mjs`. `tile_plan.py` plans 2 Pass Advanced tiling from the Pass 2 size and the VRAM preset and holds the fixed (hidden) MMH3 settings; `_build_minimax_h3_advanced_2pass_api_prompt` adds `VRGDG_MiniMaxH3SpatialTilePlan` to the graph so the plan is made from the real size at run time. Keep the JS and Python copies in step (`tests/test_minimax_tile_plan.py`, `tests/test_minimax_resolution.py`, `tests/minimax_pass_settings.cjs`).

#### [minimax/latent_upscaler.py](minimax/latent_upscaler.py)
- **Purpose**: Learned latent upscaling nodes operating directly on MiniMax latent tensors.
- **Nodes Registered**:
  - `VRGDG_MiniMaxH3LatentUpscaleModelLoader`: "Load MiniMax H3 Learned Latent Upscaler"
  - `VRGDG_MiniMaxH3UltimateUpscaleParams`: "MiniMax H3 Ultimate Upscale Params (VRGDG)"
  - `VRGDG_MiniMaxH3SpatialTilePlan`: "MiniMax H3 Spatial Tile Plan (VRGDG)" — derives grid, overlaps, fades, minimum tile and temporal chunk settings from the Pass 2 size and a VRAM preset (8-24 GB).
  - `VRGDG_MiniMaxH3LearnedLatentUpscale`: "MiniMax H3 Learned Latent Upscale"
  - `VRGDG_MiniMaxH3ReplaceUpscaledVideoLatent`: "MiniMax H3 Replace with Upscaled Video Latent"

---

### Post-Processing & Enhancement (`post_process/`)

#### [post_process/face_fix.py](post_process/face_fix.py)
- **Purpose**: High-precision video face repair backend. Tracks faces across video frames, extracts guided anchor crops, processes crops through enhancement models, and composites repaired faces back into the source video with feathered edge masks.
- **Key Functions**:
  - `register_face_fix_routes(server_instance)`: Registers endpoints `/vrgdg/face_fix/detect_anchors`, `/vrgdg/face_fix/accept_ltx_frames`, and `/vrgdg/face_fix/finalize`.

#### [post_process/luts.py](post_process/luts.py)
- **Purpose**: 3D LUT (.cube) parsing and PyTorch tensor application with strength blending.

#### [post_process/lut_video_tools.py](post_process/lut_video_tools.py)
- **Purpose**: Video grading and preview routes.
- **Key Functions**:
  - `register_lut_routes(server_instance)`: Registers endpoints for LUT discovery (`/vrgdg/music_builder/luts`), LUT image/video application (`apply_image`, `apply_video`), procedural film grain generation, and color adjustments (brightness, contrast, saturation).

---

### Storyboard Subsystem (`storyboard/`)

The `storyboard` package provides script breakdown, shot planning, and narrative structure tools.

#### [storyboard/nodes.py](storyboard/nodes.py)
- **Purpose**: Storyboard canvas UI node and HTTP route registration.
- **Node Registered**:
  - `VRGDG_StoryboardBuilderUI`: "VRGDG Storyboard Builder UI"
- **Key Routes**: `/vrgdg/storyboard/load`, `/vrgdg/storyboard/save`, `/vrgdg/storyboard/story_brief`, `/vrgdg/storyboard/story_arc`, `/vrgdg/storyboard/scene_story_beat`, `/vrgdg/storyboard/export_prompts`.

#### [storyboard/story_layer.py](storyboard/story_layer.py)
- **Purpose**: Narrative arc management. Breaks stories into Three-Act structures, defines emotional intensity curves, and maps plot points to visual beats.

#### [storyboard/timeline_notes.py](storyboard/timeline_notes.py)
- **Purpose**: Preserves user Timeline Notes and their timestamps, and maps range or point notes to overlapping scene cards for Story Arc planning. The UI supplies current `timeline_markers`; the Agent API uses the project's saved markers. Story Arc format retries and detailed scene entries keep this timing context.

#### [storyboard/scene_prompts.py](storyboard/scene_prompts.py)
- **Purpose**: Translates high-level storyboard cards into specific visual prompt strings for image generation and video rendering.

#### [storyboard/dialogue_scenes.py](storyboard/dialogue_scenes.py)
- **Purpose**: Parses scripts and song lyrics to identify character dialogue, attribute speakers, and time scene transitions.

#### [storyboard/persistence.py](storyboard/persistence.py)
- **Purpose**: Serialization of storyboard cards, beats, and shot configurations to `storyboard.json` using atomic writes.

#### [storyboard/scene_helpers.py](storyboard/scene_helpers.py)
- **Purpose**: Math and timing utilities for calculating scene lengths, frame offsets, and shot classifications.

---

### Prompt Creator Subsystem (`prompt_creator/`)

#### [prompt_creator/nodes.py](prompt_creator/nodes.py)
- **Purpose**: Registers backend HTTP API routes for the interactive Prompt Creator tool (`/vrgdg/music_prompt_creator/*`): concept creation, motion notes extraction, segment repair, draft saving/loading, and Whisper prompt generation.

---

### General Custom Nodes (`general/`)

The `general` package contains modular canvas nodes that can be placed in any standard ComfyUI workflow.

#### [general/lyrics.py](general/lyrics.py)
- **Purpose**: Lyric transcription, alignment, and SRT segment processing.
- **Nodes Registered**:
  - `VRGDG_PromptTemplateBuilder`: Assembles multi-token prompt templates with variable replacement.
  - `VRGDG_ManualLyricsExtractor_SRT`: Extracts text and timing from standard SRT subtitle files.
  - `VRGDG_ManualLyricsExtractor_SRT_Advanced`: Advanced SRT extraction with character identification.
  - `VRGDG_TimestampedLyricsExtractor`: Generates word-level timestamped SRT subtitles directly from audio using `stable-ts` / Whisper.

#### [general/video.py](general/video.py)
- **Purpose**: Video analysis, beat-aligned scene sizing, image sequence management.
- **Nodes Registered**:
  - `VRGDG_BuildVideoOutputPath_General_SRT`: Generates deterministic, formatted output file paths.
  - `BeatImpactAnalysisNode`: Analyzes audio transients to detect cut opportunities.
  - `BeatSceneDurationNode`: Quantizes scene durations to musical bars and beats.
  - `IndexedImageFromFolder_ForRemakeMode`: Sequentially iterates image directories by index.
  - `VRGDG_LatestSRTAutoLoader`: Automatically finds and loads the newest SRT file in a directory.
  - `VRGDG_LoadAudioSplit_SRTOnly`: Extracts audio matching an SRT segment's time bounds.
  - `VRGDG_TrimImageBatch_SRTOnly`: Trims image batches to match target video frame counts.

#### [general/audio.py](general/audio.py)
- **Purpose**: Audio manipulation and stem separation nodes.
- **Nodes Registered**:
  - `VRGDG_AudioCrop`: Crops audio tensors by start and end timestamps.
  - `VRGDG_GetStems`: Performs 4-stem separation (vocals, drums, bass, other) using `demucs`.

#### [general/utility.py](general/utility.py)
- **Purpose**: Swiss-army utility nodes for types, strings, LoRAs, and prompt JSONs.
- **Nodes Registered**:
  - `VRGDG_ShowText`, `VRGDG_ShowAny`, `VRGDG_TextBox`: UI display and multiline text input.
  - `VRGDG_IntToFloat`: Type casting node.
  - `VRGDG_OptionalMultiLoraModelOnly`, `VRGDG_OptionalMultiLoraTwoPassStrengths`, `VRGDG_LoraFromPathModelOnly`: Dynamic multi-LoRA loading with strength control.
  - `VRGDG_MultiStringConcat`: Concatenates up to 10 strings with configurable delimiters.
  - `VRGDG_LyricSegmentJsonFixer`, `VRGDG_LyricSegmentTextCleaner`: Text sanitization and JSON repair.
  - `VRGDG_PromptMapJsonFixer`, `VRGDG_PromptJsonSubjectPrepender`: Injects subjects into prompt maps.
  - `VRGDG_LyricSegmentDurationMerger`: Merges short segments into coherent scenes.
  - `VRGDG_MultiReferenceConditioningFromPaths`, `VRGDG_ImageBatchMultiFromPaths`: Loads batches of reference images from paths for model conditioning.

#### [general/ltx_msr_reference.py](general/ltx_msr_reference.py)
- **Purpose**: Multi-Scale Reference (MSR) conditioning builder for LTX-Video.
- **Nodes Registered**:
  - `VRGDG_LTXMSRReferenceBuilder`: "VRGDG LTX MSR Reference Builder"
  - `VRGDG_LTX25MSRReferenceLoader`: "VRGDG LTX 2.5 MSR Reference Loader"

---

### Browser AI Automation (`browser/` & `flow_automation/`)

#### [browser/nodes.py](browser/nodes.py)
- **Purpose**: Browser automation nodes bridging external web-based image generators (Flow, Meta AI, ChatGPT) directly into ComfyUI workflows.
- **Nodes Registered**:
  - `VRGDG_FlowBrowserImageEdit`: "VRGDG Flow Browser Image Edit"
  - `VRGDG_FlowBrowserSetup`: "VRGDG Flow Browser Setup"
  - `VRGDG_ChatGPTImagesBrowser`: "VRGDG ChatGPT Images Browser"
  - `VRGDG_MetaAIBrowserImage`: "VRGDG Meta AI Browser Image"
- **Routes Registered**: `/vrgdg/browser_image/status`, `setup`, `open_login`, `manual_open`, `manual_upload`, `manual_submit`, `manual_finish`, `manual_wait_download`, `manual_import_latest`.

#### `flow_automation/`
- **Purpose**: Standalone Node.js automation scripts utilizing Puppeteer / Playwright to control external browser sessions:
  - `flow-poc.mjs`: Automation script for Flow.
  - `chatgpt-images-poc.mjs`: Automation script for ChatGPT DALL-E/image generator.
  - `meta-ai-poc.mjs`: Automation script for Meta AI generation.
  - `manual-bridge.mjs`: WebSocket bridge facilitating user login and manual image downloads.

---

### Frontend Web Applications (`web/`)

ComfyUI automatically loads all extensions placed in [web/](web) because `WEB_DIRECTORY = "./web"` is declared in [__init__.py](__init__.py).

#### Standalone Extension Scripts
- [web/VRGDG_MusicVideoBuilderUI.js](web/VRGDG_MusicVideoBuilderUI.js): Entry point registering the Video Builder modal dialog on the ComfyUI canvas node.
- [web/VRGDG_StoryboardBuilderUI.js](web/VRGDG_StoryboardBuilderUI.js): Entry point registering the Storyboard Builder modal dialog.
- [web/VRGDG_ResourceMonitor.js](web/VRGDG_ResourceMonitor.js): Canvas and top-bar widget rendering live host RAM and NVIDIA VRAM gauges.
- [web/VRGDG_UIThemes.js](web/VRGDG_UIThemes.js): Theme engine supporting dark, cinematic, and modern styling tokens across VRGDG UI dialogs.
- [web/VRGDG_FaceFixUI.js](web/VRGDG_FaceFixUI.js): Interactive face-repair preview and anchor selection interface.
- [web/VRGDG_RenderETA.js](web/VRGDG_RenderETA.js): Calculates and displays remaining render time based on historical frame rendering rates.

#### Video Builder Application Modules (`web/music_video_builder/`)
A modular 79-file ESM (`.mjs`) application structure:
- **Core Orchestration**: `builder.mjs` (main coordinator), `session.mjs` (session state), `history.mjs` (undo/redo).
- **Timeline & Playback**: `timeline_view.mjs`, `timeline_state.mjs`, `timeline_actions.mjs`, `timeline_events.mjs`, `timeline_edit.mjs`, `beat_calibration.mjs`.
- **Inspector & Settings**: `inspector.mjs`, `inspector_events.mjs`, `video_settings_panel.mjs`, `model_settings.mjs`, `model_pickers.mjs`.
- **Media & References**: `media_import.mjs`, `image_panels.mjs`, `reference_builder.mjs`, `reference_data.mjs`, `reference_subjects.mjs`, `reference_locations.mjs`, `reference_scene_mapping.mjs`.
- **Prompting & LLM**: `prompt_editing.mjs`, `prompt_creator.mjs`, `builder_agent.mjs`, `llm_runner.mjs`, `gemma_runner.mjs`.
- **Rendering & Output**: `batch_render.mjs`, `scene_render_prep.mjs`, `video_render.mjs`, `render_log.mjs`, `post_process.mjs`.
- **MiniMax H3 Panel**: `minimax_panel.mjs`, `minimax_panel_layout.mjs`, `minimax_panel_events.mjs`, `minimax_prompt.mjs`, `minimax_references.mjs`, `minimax_h3.mjs`.
- **Wizard**: `wizard.mjs`, `wizard_bridge.mjs`, `auto_build.mjs`.

#### Storyboard Builder Application Modules (`web/storyboard_builder/`)
A modular 22-file ESM (`.mjs`) application structure:
- `storyboard.mjs`: Main application entry and coordinator.
- `story_layer.mjs` & `story_layer_layout.mjs`: Narrative structure and act editor.
- `scene_editor.mjs` & `scene_table.mjs`: Scene card grid and table views.
- `shot_presets.mjs` & `video_style.mjs`: Cinematic shot presets and visual style definitions.
- `gpt_payload.mjs` & `prompt_generation.mjs`: Prepares structured generation payloads for external LLMs.

---

### Agent API Subsystem (`agent_api/`)

The `agent_api` package provides a headless, production-grade REST API served at `/vrgdg/api/v1` by ComfyUI. Designed for autonomous external AI agents, headless worker processes, and IDE extensions, it exposes full control over the AI Video Builder without requiring a browser session.

#### Architecture & Design Principles
- **Uniform Envelopes (D6)**:
  - Success responses return: `{"ok": True, "data": {...}, "revision": n}`.
  - Failure responses return structured error blocks: `{"ok": False, "error": {"code": "...", "message": "...", "details": {...}, "retryable": bool}}`.
- **Revision-Based Optimistic Concurrency (D4)**:
  - Every project save increments a monotonically increasing `revision` stored in `vrgdg_builder_session.json`.
  - Mutation endpoints support optimistic concurrency via the HTTP `If-Match: <revision>` header, rejecting stale edits with `409 Conflict` (`REVISION_CONFLICT`) to prevent silent overwrites.
- **Transactional Timeline Journal (Section 15.6)**:
  - All structural scene operations (split, merge, insert, delete, move, resize) employ `TimelineJournal`.
  - Disk renumbering of assets (`image_NNNN.png`, `video_NNNN.mp4`, etc.) is recorded before disk renames and rolled back automatically if an error occurs mid-transaction.
- **SQLite Job Manager (Section 5)**:
  - Long-running generation jobs run in background worker threads, tracked in `agent_jobs.db` located inside the project directory.
  - Supports live Server-Sent Events (SSE) streaming (`GET /vrgdg/api/v1/jobs/{id}/events`).
  - Interrupted jobs left behind across ComfyUI restarts are safely recovered into `interrupted` status on startup.

#### Endpoint reference (agents read this first)
- **[Api_Endpoints.md](Api_Endpoints.md)** lists every endpoint with its method, path, what it does, and whether it is a **job** (returns `202` and a `job_id`), takes `If-Match`, or reads query parameters and body keys. Read it before calling anything. It is generated, so it always matches `router.py`.
- **Base URL** is `http://127.0.0.1:8188/vrgdg/api/v1`. Every path in the reference is relative to it.
- **Jobs**: an endpoint marked job returns at once. Poll `GET /jobs/{id}` until `status` is `succeeded`, `failed`, `cancelled` or `interrupted`. Run one GPU job at a time.
- **LLM steps use the loaded model.** Story, reference, description and prompt steps run on the project's saved runner. For LM Studio that is the model already loaded, found with the read-only `GET /api/v0/models`. The API never loads, unloads or switches a model (`agent_api/llm_runtime.py`). `GET /llm/active` shows what would be used and `503 LLM_UNAVAILABLE` means nothing is loaded.
- **The workflow** from song to final video, in order: create the project, set the MiniMax H3 settings, attach audio, set lyrics, `timeline/from-lines`, add the character, `references/{kind}/{rid}/describe`, `references/locations/extract`, `references/assign-scenes`, `story/settings`, `story/arc`, `story/brief`, `story/beats`, `minimax-prompts`, `scenes/{sid}/video/render` per scene, `stitch`. [MUSIC_VIDEO_AGENT_PROMPT.md](MUSIC_VIDEO_AGENT_PROMPT.md) spells out each call and its arguments.
- **Keeping the reference current**: `scripts/export_api_endpoints.py` writes `Api_Endpoints.md` and `agent_api/endpoints.json` from the router and the `DESCRIPTIONS` map inside the script. It stops if a route has no description. `tests/test_api_endpoints_doc.py` fails when either file is stale. The MCP server builds its generated tools from `endpoints.json`.

#### Contract and parity with the Video Builder UI
- **OpenAPI contract (D12)**: `agent_api/openapi.json` is generated from `router.py` (parsed with `ast`, never imported) and `schemas.py` by `scripts/export_openapi.py`. Run it after changing a route, a schema, or an error code. `tests/test_agent_api_contract.py` fails when the file is stale, when registered routes and documented routes differ, or when envelopes and settings groups drift from the schemas.
- **MiniMax H3 settings**: the UI saves every option in `session["minimax_h3_settings"]`. `minimax/h3_settings_defaults.json` lists all of them and is generated from the UI by `node scripts/export_minimax_defaults.mjs`. `minimax/settings_payload.py` validates patches, applies scene overrides, and builds the same render payload as `video_render.mjs`. A test fails when the browser payload gains a key the Python builder does not send. Agents read the full list with `GET /settings/minimax-h3/schema` and change settings with `PATCH /projects/{pid}/settings` under the `minimax_h3` group.
- **Scene inputs**: `minimax/scene_inputs.py` resolves reference images (start frame, mapped subjects, extras, locations, ingredients), previous-scene continuity frames, reference videos and the last frame, like the browser. Reference scene maps live inside `session["flux_reference_builder"]`, where the UI reads them.
- **Session keys the API must match**: project audio is `audio_path` (use `agent_api.paths.session_audio_path`), settings groups map to the UI's keys (see `_SETTINGS_GROUP_SESSION_KEYS` in `mutations.py`), and `builder_save_revision` (UI counter) can run ahead of `revision` (server counter). `_persist_session` stays above both and raises `REVISION_CONFLICT` instead of reporting a discarded write as saved.
- **Files the API writes must look like the Builder's.** A rendered scene is finished the Builder's way: trim the raw render to the exact timeline length (label `minimax_exact`, marked as the audio video), then collect it as `rendered_scene_videos/video_NNNN-audio.mp4`. Record it on the segment with `agent_api/scene_video.apply_scene_video()`, which sets `video_path`, `video_thumbnail_path`, `video_history`, `video_thumbnail_history`, `video_status` and `video_cache_bust`. Setting only `video_path` leaves the timeline without a picture. Scene audio for MiniMax is cut at render time into `minimax_h3_scene_audio/`. The lyric lane shows when `show_timeline_lyric_notes` is true, which `timeline/from-lines` sets.
- **Scene views read what the session saved.** `agent_api/projects.py` reports each scene's `lyric_text`, `story_beat`, `minimax_h3_prompt`, image, rendered video, thumbnail and scene audio from the session's own paths, with the old fixed file names only as a fallback.
- **The Storyboard Builder reads its own copy.** It shows video prompts, story beats and the green or red status from `storyboard/storyboard.json`, not from the session. The story steps and `minimax-prompts` call `sync_storyboard_files()` (`storyboard_orchestrator.py`), which runs the same save and export as "Save Storyboard" (`storyboard/persistence.py`): `storyboard.json` with each scene's `video_prompt` and `status`, plus `prompts/i2v_prompts.txt` and `video_prompts.json`. Segments also get `video_prompt_type: "rtv"`.
- **Lyrics are in the prompt, in double quotes.** A singing scene's shot says what is sung: `<Subject 1> (name) sings the lyric line, "the exact words"`. The task text asks for it, and `minimax/shot_prompt.ensure_quoted_lyrics()` (Python) and `miniMaxH3EnsureQuotedLyricInShot` (`minimax_prompt.mjs`, used by the normal Builder) quote a plain copy or add the sentence, so it does not depend on the LLM. Lyric lines are shared across cuts in order. Quoted words are never dropped as negative wording.
- **MiniMax prompt format**: a saved reference-to-video prompt is the Builder's full format: `subject_definitions:` (each `<Subject N>` tied to its `<Picture N>`, plus `<Audio 1>` for input audio), `summary:`, `retention_analysis:`, then `detailed_description:` with the style line and the `[Shot N]` blocks, then `overall_soundscape:` and `non_diegetic_music:`. The render sends the saved text as it is, so the definitions must be in it. `minimax/shot_prompt.py` builds, parses and validates the shots and `reference_frame()` / `wrap_reference_prompt()` add the sections around them (picture numbers follow `scene_inputs.ordered_reference_items`, the order the images are sent). Both `minimax-prompts` and `prompts/minimax/assemble` use it, and the 7,000-character budget counts the added sections.
- **Built-in audio and 2 Pass**: `audio_mode: built_in_audio` renders with Single or 2 Pass. 2 Pass takes the audio-driven template and `runner/minimax_workflows._use_minimax_h3_native_audio()` rewires it by node class: the audio file and `VRGDG_MiniMaxH3AudioDrive` go away, pass 1 samples the Reference to Video node's own latent and its audio is decoded with `LTXVAudioVAEDecode`. 2 Pass Advanced still takes input audio only, and so does the Video Builder's own 2 Pass render (`video_render.mjs`). With built-in audio the warm-up frames are never cut short by a missing source file.
- **Latent Continuation Masked over the API**: `continuity_mode` accepts `latent_continuation_masked` (every spelling is saved as that value, an unknown mode is rejected by `validate_minimax_h3_patch`), `latent_context_frames` accepts 16, 22, 39, 56, 90, 141 and 192, and `location_transition_preset` accepts `masked`; `minimax_h3_settings_schema()` lists them with notes. The render payload carries the mode for Single and 2 Pass, and the orchestrator checks the predecessor's latent (`PREDECESSOR_MISSING`). The scene field `minimax_h3_continuation_direction` is PATCHable and shown on the scene. The browser's frame-to-frame prompt writer is ported: with `continuity_prompt_from_last_frame` on, `render_scene_video_async` extracts the previous scene's final frame, calls `minimax_prompt_orchestrator.write_continued_scene_prompt()` (the LLM is shown the frame as Attached Picture 1 with the `minimax_h3_frame_continuity` instruction), saves the prompt on the scene (`minimax_h3_prompt_origin` `previous_final_frame`, plus the `minimax_h3_continuity_prompt_*` fields the Builder saves) and renders it. The task text is `shot_prompt.continuation_task_text(with_picture=True)` and `shot_prompt.location_continuity_contract()`, Python twins of the contracts in `minimax_prompt.mjs`, so a change to one needs the other. A loaded model that cannot read images falls back to the previous scene's last shot as text and reports it in `warnings`. Writing prompts ahead of time (`minimax-prompts`) has no final frame yet, so it always uses that text stand-in, and `build_minimax_prompt_context` adds a `continuation` block (hold seconds, the direction and the writing rules) for agents that write the shots themselves. `continuation_hold_seconds()` (Python) and `miniMaxH3ContinuationHoldSeconds` (`minimax_prompt.mjs`) must stay equal, `tests/test_minimax_masked_continuation_api.py` pins the values. After a MiniMax render the API saves `minimax_h3_continuity_mode_used` and `minimax_h3_continuity_source_scene_id` on the scene like the Builder does, taken from the graph's `latent_continuation_settings` (what was really loaded, so a scene's locked settings win over the project default); `tests/test_agent_api_render_records_continuity.py` covers it.
- **Known gaps against the browser render**: per-scene custom audio ranges, the supporting renderer pictures in the frame-based prompt request (only the final frame is attached), the older latent modes' frame-based prompt (only `latent_continuation_masked` has it), and LLM prompt payload parity are not ported. Server renders use the saved prompts.
- **Run on server (opt-in)**: "Build Full Video" offers "On the server (background job)". It saves the project, starts `pipelines/build-full-video` (`web/music_video_builder/server_pipeline.mjs`), follows the job, and reloads the project. The browser loop remains the default.

#### Key Modules & Endpoints
- **[agent_api/router.py](agent_api/router.py)**:
  - Registers `/vrgdg/api/v1` route endpoints across all subsystems.
- **[agent_api/mutations.py](agent_api/mutations.py)**:
  - Scene CRUD, timeline snapping, gap closing, bulk edits, beat calibration, reference management, lyrics attachment, settings patching, and comprehensive project validation (`validate_project`).
- **[agent_api/jobs.py](agent_api/jobs.py)**:
  - Thread-safe job execution manager, log file management, status transitions, and SSE subscriber broker.
- **[agent_api/schemas.py](agent_api/schemas.py)**:
  - JSON Schema definitions and validator functions for mode-specific settings (LTX 6 modes, MiniMax H3 5 modes, image engines).
- **[agent_api/paths.py](agent_api/paths.py)**:
  - Multi-root project discovery, project ID validation, and path traversal containment safeguards.
- **[agent_api/orchestrator/](agent_api/orchestrator)**:
  - `pipeline_orchestrator.py`: End-to-end dry-run planning (`GET /pipelines/plan`) and full autonomous builds (`pipeline.build_full_video`, `pipeline.build_flf`, `pipeline.from_song`). `POST /pipelines/from-song` creates the project if needed, attaches audio and lyrics, cuts even scenes (`plan_scene_boundaries`), then runs the full build. A project that already has scenes keeps them.
  - `video_orchestrator.py`: Video rendering jobs (`video.render`), latent caching, color matching, and final video stitching.
  - `image_orchestrator.py`: Single-scene and batch image generation (`image.generate`) across Z-Image, Flux/Klein, NanoBanana, Ernie, and Krea 2.
  - `prompt_orchestrator.py`: LLM batch prompt generation jobs (`prompt.batch_generate`).
  - `post_orchestrator.py`: Video LUT application, film grain overlay, color adjust presets, and Face Fix pipelines.
  - `lyrics_orchestrator.py`: `lyrics.align` (Stable-ts timing through ComfyUI) and `timeline.from_lines` (Line Mapping with a min and max scene length). Pure length rules are in `builder/lyric_scenes.py`.
  - `reference_orchestrator.py`: Gemma Describe (`reference.describe`), LM Extract for locations (`reference.extract_locations`) and Assign Scenes patterns.
  - `storyboard_orchestrator.py`: scene defaults and story idea, then the story arc, story brief and scene beats (`storyboard.story_arc`, `storyboard.story_brief`, `storyboard.scene_beats`) through `storyboard/story_layer.py`.
  - `minimax_prompt_orchestrator.py`: MiniMax H3 reference-to-video prompts (`minimax.prompts`) with the saved instruction and `minimax/shot_prompt.py`.
- **[agent_api/llm_runtime.py](agent_api/llm_runtime.py)**: picks the LLM for a request. The saved runner settings become the generator keys, and for LM Studio the loaded model and its context length are used.
- **[agent_api/scene_video.py](agent_api/scene_video.py)**: records a rendered video on a segment like the Builder does.
- **[agent_api/session_keys.py](agent_api/session_keys.py)**: the top-level keys of the Builder session, a Python twin of `currentSessionData()` in `web/music_video_builder/session.mjs` (`tests/test_agent_api_project_include_fresh.py` fails when they drift). `GET /projects/{pid}?include=` accepts these even before a project has saved them.

---

### Model Context Protocol Server (`mcp_server/`)

The `mcp_server` package is a standalone, zero-dependency Model Context Protocol (MCP) server over stdio (JSON-RPC 2.0). An MCP client such as Claude Code, Claude Desktop or LM Studio starts it as a subprocess, and it calls the Agent API on the same machine. ComfyUI must be running.

#### Server features and design
- **Zero external dependencies**: Python standard library only, so it runs on the portable `python_embeded`.
- **Protocol**: MCP `2024-11-05` over stdin and stdout. Nothing else may print to stdout.
- **Start it**: a client runs the entry file. The portable Python ignores the current folder for `-m`, so do not use `-m mcp_server` with it.
  ```bash
  ..\..\..\python_embeded\python.exe mcp_server\__main__.py
  ```
  `start_mcp_server.bat` (git-ignored) does the same, finds the portable Python and sets `VRGDG_API_URL` to the default. Settings: `VRGDG_API_URL` (default `http://127.0.0.1:8188/vrgdg/api/v1`) and `VRGDG_API_TOKEN` when a token is required.
- **Client config**: Claude Code: `claude mcp add vrgdg -- <python_embeded\python.exe> <pack>\mcp_server\__main__.py`. LM Studio `mcp.json`: `{"mcpServers": {"vrgdg": {"command": "<python>", "args": ["<pack>/mcp_server/__main__.py"], "env": {"VRGDG_API_URL": "http://127.0.0.1:8188/vrgdg/api/v1"}}}}`.
- **Server instructions**: the `initialize` reply carries `instructions` that point the agent to the `make_music_video` prompt and the doc resources and remind it never to change the LM Studio model.
- **Actionable errors**: a failed call returns `isError: true` with `next_steps` in the text, so an agent is not left retrying blindly.

> `mcp_server/` is local-only (listed in `.gitignore`). `tests/test_mcp_server.py` and `tests/test_mcp_endpoints.py` need the folder.

#### Tools
- **Named tools (63)**, written by hand in `tools.py`:
  - *System and projects*: `system_health`, `list_modes`, `list_models`, `llm_active`, `project_list`, `project_create`, `project_get`, `project_summary`, `project_get_settings`, `project_update_settings`, `minimax_settings_schema`, `project_duplicate`, `project_validate`, `project_export`, `project_delete`.
  - *Audio, lyrics, timeline*: `audio_attach`, `audio_analyze`, `lyrics_set`, `lyrics_transcribe`, `lyrics_align`, `timeline_build`, `timeline_from_lines`, `timeline_enforce_length`.
  - *Scenes*: `scene_list`, `scene_get`, `scene_update`, `scene_insert`, `scene_delete`, `scene_split_merge_move_resize`, `scenes_bulk_edit`.
  - *References and story*: `references_get`, `reference_upsert`, `references_extract`, `reference_describe`, `reference_extract_locations`, `reference_assign_scenes`, `reference_scene_mapping_set`, `story_get_set`, `story_generate`, `story_settings`, `story_create`.
  - *Prompts*: `prompts_generate`, `minimax_prompts`, `instructions_get_set`.
  - *Generation and pipelines*: `image_generate`, `image_set_from_upload`, `image_approve_revert`, `video_render`, `video_recover`, `video_select_take`, `latents_status_rebuild`, `post_apply`, `stitch_final`, `pipeline_build_full_video`, `pipeline_from_song`, `pipeline_plan`.
  - *Jobs and files*: `job_get`, `job_wait`, `job_cancel`, `job_retry`, `asset_view`, `asset_download_url`, `upload_file`.
- **Generated `api_*` tools**: `endpoint_tools.py` reads `agent_api/endpoints.json` and adds one tool for every endpoint the named tools do not call, for example `api_post_scenes_video_trim` or `api_get_finals`. Names are `api_<method>_<path words>`. Arguments are the path ids (`project_id`, `scene_id`, ...), an optional `body`, `query` and `if_match_revision`. Every endpoint can therefore be reached through a named tool.
- **`api_request`**: calls any method and path directly, for example `{"method": "GET", "path": "/projects/MySong/scenes"}`.
- **Tests that keep this honest**: `tests/test_mcp_endpoints.py` fails when a named tool calls a route the router does not have, when an endpoint has no tool, or when names collide. When you change a route, regenerate `Api_Endpoints.md` and `endpoints.json` (see above) and the generated tools follow.

#### Resources and prompt templates
- **Docs an agent can read**:
  - `vrgdg://docs/endpoints`: [Api_Endpoints.md](Api_Endpoints.md), every endpoint and what it does.
  - `vrgdg://docs/music-video-playbook`: [MUSIC_VIDEO_AGENT_PROMPT.md](MUSIC_VIDEO_AGENT_PROMPT.md), the song-to-video steps.
  - `vrgdg://docs/openapi`: `agent_api/openapi.json`, the machine-readable contract.
- **Live data**: `vrgdg://projects`, `vrgdg://modes`, `vrgdg://project/{id}`, `vrgdg://project/{id}/scene/{sid}`, `vrgdg://project/{id}/lyrics`, `vrgdg://jobs/{id}/log`.
- **Prompts**:
  - `make_music_video`: takes the project name, audio, lyrics, character name and image, location style theme and story idea, and returns them followed by the full playbook.
  - `review_scene`, `fix_failed_render`, `polish_timeline`: scene critique, failed-job diagnosis and timeline pacing.

---

### Utility Scripts (`scripts/`)

- [scripts/build_minimax_h3_ref2va_2pass_audio_api.py](scripts/build_minimax_h3_ref2va_2pass_audio_api.py): Offline standalone generator for 2-pass MiniMax H3 reference-to-video API workflow JSON files.
- [scripts/build_ltx25_normal_sampler_workflow.py](scripts/build_ltx25_normal_sampler_workflow.py): Generates baseline LTX 2.5 sampler API graphs.
- [scripts/export_api_endpoints.py](scripts/export_api_endpoints.py): Writes `Api_Endpoints.md` and `agent_api/endpoints.json` from the router and the descriptions in the script. `--check` exits 1 when they are stale or a route has no description.
- [scripts/far_face_repair_backend.py](scripts/far_face_repair_backend.py): Standalone backend testing script for small/distant face detection and super-resolution repair.
- [scripts/Backport-Krea2ToMusubi.ps1](scripts/Backport-Krea2ToMusubi.ps1): PowerShell script for migrating dataset captions and LoRA configurations between training engines.

---

### Test Suites (`tests/`)

The repository includes 84 test suites verifying backend logic, API contracts, and JavaScript source code integrity.

- [tests/test_node_registration.py](tests/test_node_registration.py): Parses AST of all submodules in `_VRGDG_SUBMODULES` and asserts that every node identifier in `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` is globally unique. **Must always pass.**
- [tests/test_atomic_write.py](tests/test_atomic_write.py): Simulates disk interruptions, asserts file replacement integrity, and verifies temp file cleanup.
- [tests/test_builder_branch_project.py](tests/test_builder_branch_project.py): Tests non-destructive project cloning and path rewriting.
- [tests/test_minimax_h3_latent_continuation.py](tests/test_minimax_h3_latent_continuation.py): Validates frame-to-token conversions, latent tensor guide alignment, and safetensors persistence.
- [tests/test_minimax_masked_continuation_api.py](tests/test_minimax_masked_continuation_api.py): Latent Continuation Masked through the Agent API: settings enums and aliases, the PATCHable `minimax_h3_continuation_direction`, the hold time (twin of the Builder's), the `continuation` prompt context block, and the LLM task text for a continued scene.
- [tests/builder_source.py](tests/builder_source.py): AST and regex extraction helper allowing Python unit tests to validate frontend JavaScript logic and contracts without spinning up a headless browser.

---

## 3. Separation of Concerns (SoC) Principles

Every module in this codebase has a single, strictly bounded responsibility. Adhere to these architectural boundaries:

```
+-----------------------------------------------------------------------------------+
|                        PRESENTATION LAYER (web/**/*.mjs)                          |
|  - Renders UI controls, timeline tracks, inspector panels, and dialogs.           |
|  - Handles DOM events, keyboard shortcuts, and local UI state.                    |
|  - NEVER accesses local filesystem directly; communicates strictly via REST API.  |
+-----------------------------------------+-----------------------------------------+
                                          | HTTP REST / WebSockets
                                          v
+-----------------------------------------------------------------------------------+
|                           API LAYER (**/routes.py)                                |
|  - Validates request payloads and query parameters.                               |
|  - Returns standardized JSON envelopes: {"ok": true, ...} or {"ok": false, ...}. |
|  - Dispatches heavy CPU/GPU tasks to asyncio.to_thread.                           |
|  - NEVER contains complex business algorithms directly.                           |
+-----------------------------------------+-----------------------------------------+
                                          | Function Calls
                                          v
+-----------------------------------------------------------------------------------+
|                         SERVICE LAYER (**/project.py, etc.)                       |
|  - Implements core business logic: project lifecycle, beat detection, latents.   |
|  - Performs disk I/O using atomic_write primitives.                               |
|  - Stateless where possible; receives arguments and returns structured results.   |
+-------------------+-------------------------------------+-------------------------+
                    |                                     |
                    v                                     v
+-----------------------------------+ +---------------------------------------------+
|    GRAPH COMPILER (runner/*.py)   | |         COMFYUI NODES (**/nodes.py)         |
| - Builds /prompt API dicts.       | | - Declares INPUT_TYPES, RETURN_TYPES, etc.  |
| - Connects node slots via IDs.    | | - Receives tensors/strings from ComfyUI.    |
| - Independent of UI state.        | | - Calls service layer and returns outputs.  |
+-----------------------------------+ +---------------------------------------------+
```

### Golden Invariants
1. **Never Mix Node Logic with Server Routes**: A file declaring `NODE_CLASS_MAPPINGS` should only handle ComfyUI socket inputs and outputs. HTTP endpoints belong in `routes.py` files.
2. **Never Perform Blocking I/O on the Event Loop**: All synchronous disk I/O, subprocess executions (e.g. `ffmpeg`, `nvidia-smi`), or heavy math calculations within route handlers must be offloaded via `await asyncio.to_thread(...)`.
3. **Always Use Atomic Writes for State Mutation**: Any file written by the backend (`session.json`, `storyboard.json`, settings) must be written via `atomic_write_json` or `atomic_write_text`.
4. **Preserve Latent-Space Fidelity**: When implementing video continuation between scenes (as in MiniMax H3), always pass the raw latent tensor guide. Never round-trip through VAE decode and pixel re-encode unless explicitly requested by the user.

---

## 4. PEP 8 Coding Standards & Repository Conventions

All Python code must strictly follow **PEP 8** standards along with repository-specific conventions:

### Formatting & Layout
- **Indentation**: Exactly 4 spaces per indentation level. No tabs.
- **Line Length**: Soft limit of 100 characters; hard limit of 120 characters where readability is improved.
- **Blank Lines**:
  - 2 blank lines between top-level functions and class definitions.
  - 1 blank line between methods inside a class.
- **Quotes**: Double quotes (`"`) preferred for docstrings and standard strings; single quotes (`'`) acceptable for dictionary keys or internal identifiers if consistent within the module.

### Naming Conventions
- **ComfyUI Node Classes**: `VRGDG_<DescriptivePascalCase>` (e.g., `VRGDG_MiniMaxH3AudioDrive`, `VRGDG_AudioCrop`).
- **Internal Helper Functions**: Leading underscore with lowercase snake_case (e.g., `_read_audio_peaks`, `_save_builder_session`).
- **Public Service Functions & Classes**: Standard `snake_case` for functions, `PascalCase` for classes (e.g., `SceneLatentManager`, `atomic_write_json`).
- **Constants**: Upper snake_case (e.g., `_FRAME_PER_TOKEN`, `_CUSTOM_MODEL_ROOT_FILE`).

### Import Ordering
Imports must be grouped in three distinct blocks separated by single blank lines:
```python
# 1. Standard library imports
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 2. Third-party dependencies
from aiohttp import web
import psutil
import torch

# 3. ComfyUI core & local package imports
import folder_paths
from server import PromptServer

from ..core.atomic_write import atomic_write_json
from ..core.model_paths import load_custom_model_root
```

### Type Annotations & Docstrings
- Type annotate all new function signatures: parameters and return values.
- Include concise, descriptive docstrings explaining the contract, side effects, and potential exceptions:

```python
def calculate_minimax_h3_timing(
    duration_seconds: float,
    fps: int = 25,
) -> Tuple[int, int]:
    """Calculate the required frame count and latent token count for MiniMax H3.

    Args:
        duration_seconds: Desired video duration in seconds.
        fps: Video frame rate (default 25 fps).

    Returns:
        A tuple of (frame_count, token_count).

    Raises:
        ValueError: If duration_seconds is negative or zero.
    """
    if duration_seconds <= 0:
        raise ValueError("Duration must be strictly positive.")
    ...
```

### Logging & Error Envelopes
- Never use bare `except:`. Always catch specific exception types (`OSError`, `ValueError`, `json.JSONDecodeError`) or `Exception` when wrapping top-level route handlers.
- Prepend console messages with `[VRGDG]` or `[VRGDG <Subsystem>]` (e.g., `print(f"[VRGDG] Loaded {len(nodes)} nodes.")`).
- HTTP endpoints must return standardized JSON response dictionaries:
  - Success: `web.json_response({"ok": True, "data": ...})`
  - Client Error: `web.json_response({"ok": False, "error": str(exc)}, status=400)`
  - Server Error: `web.json_response({"ok": False, "error": str(exc)}, status=500)`

---

## 5. Developer Implementation Recipes

### Recipe 1: Creating a New ComfyUI Custom Node

When adding a new canvas node:

1. **Choose the appropriate package** (e.g., [general/utility.py](general/utility.py), [general/video.py](general/video.py), or [minimax/nodes.py](minimax/nodes.py)).
2. **Implement the node class conforming to ComfyUI standards**:

```python
class VRGDG_ExampleColorGrade:
    """Applies a cinematic tint adjustment to an input image batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tint_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "tint_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "tint_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply_grade"
    CATEGORY = "VRGDG/PostProcess"
    DESCRIPTION = "Adjusts RGB channel tint multipliers across image tensors."

    def apply_grade(self, images: torch.Tensor, tint_r: float, tint_g: float, tint_b: float, mask: Optional[torch.Tensor] = None):
        # images tensor shape: [B, H, W, C]
        multipliers = torch.tensor([tint_r, tint_g, tint_b], device=images.device, dtype=images.dtype)
        processed = torch.clamp(images * multipliers, 0.0, 1.0)
        if mask is not None:
            mask = mask.unsqueeze(-1).to(images.device)
            processed = (processed * mask) + (images * (1.0 - mask))
        return (processed,)
```

3. **Register the node in the module's mappings**:

```python
NODE_CLASS_MAPPINGS = {
    "VRGDG_ExampleColorGrade": VRGDG_ExampleColorGrade,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_ExampleColorGrade": "🎨 VRGDG Example Color Grade",
}
```

4. **Verify [__init__.py](__init__.py)**: Ensure the file containing the node is listed in `_VRGDG_SUBMODULES`.
5. **Run the node registration test**:
```bash
python -m unittest tests/test_node_registration.py
```

---

### Recipe 2: Registering a Backend HTTP Route

When adding a route to the Agent API (`/vrgdg/api/v1`, in `agent_api/router.py`) also:

1. Give the handler a one-line docstring and return `api_success(...)` or raise an `AgentApiError`.
2. Add `"METHOD /path": "what it does"` to `DESCRIPTIONS` in `scripts/export_api_endpoints.py`.
3. Run `scripts/export_openapi.py` and `scripts/export_api_endpoints.py`. The contract and endpoint-doc tests fail until you do.
4. The MCP server adds an `api_*` tool for the route by itself. Add a named tool in `mcp_server/tools.py` only when a friendlier tool is worth it, and call the real route (`tests/test_mcp_endpoints.py` checks this).

When adding a backend route:

1. **Locate the appropriate `routes.py`** (e.g., [builder/routes.py](builder/routes.py) or [runner/routes.py](runner/routes.py)).
2. **Implement an async handler wrapped in a registration guard**:

```python
from aiohttp import web
from server import PromptServer

_VRGDG_MY_ROUTES_REGISTERED = False

def _ensure_my_routes():
    global _VRGDG_MY_ROUTES_REGISTERED
    if _VRGDG_MY_ROUTES_REGISTERED:
        return
    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.post("/vrgdg/my_feature/process_task")
    async def vrgdg_my_feature_process_task(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            param = str(payload.get("param", "")).strip()
            if not param:
                return web.json_response({"ok": False, "error": "Missing 'param' field."}, status=400)

            # Offload synchronous business logic or disk I/O to a worker thread
            result = await asyncio.to_thread(_heavy_computation_service, param)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

        return web.json_response({"ok": True, "result": result})

    _VRGDG_MY_ROUTES_REGISTERED = True

_ensure_my_routes()
```

---

### Recipe 3: Adding or Extending a Workflow Runner Pipeline

When adding a new generation model or pipeline to the headless Workflow Runner:

1. **Define the model parameters in [runner/models.py](runner/models.py)**.
2. **Implement the graph compiler function** in a dedicated file (e.g. `runner/my_new_model_workflows.py`):

```python
def build_my_model_prompt_graph(
    positive_prompt: str,
    negative_prompt: str,
    checkpoint_name: str,
    steps: int = 30,
    cfg: float = 7.0,
    width: int = 1024,
    height: int = 576,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compiles a ComfyUI /prompt API graph dictionary."""
    graph = {}
    
    # 1. Checkpoint Loader
    graph["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": checkpoint_name},
    }
    
    # 2. Positive Conditioning
    graph["2"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": positive_prompt, "clip": ["1", 1]},
    }
    
    # 3. Negative Conditioning
    graph["3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative_prompt, "clip": ["1", 1]},
    }
    
    # 4. Latent Space
    graph["4"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    
    # 5. KSampler
    graph["5"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        },
    }
    
    # 6. VAE Decode & Save
    graph["6"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    }
    graph["7"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["6", 0], "filename_prefix": "VRGDG_MyModel"},
    }
    
    return graph
```

3. **Expose the compiler via [runner/routes.py](runner/routes.py)**:
   - Add a POST route `/vrgdg/workflow_runner/build_my_model_prompt` that accepts the JSON payload, calls your compiler, and returns the assembled graph.

---

### Recipe 4: Modifying Builder Frontend Modules

When updating or adding UI features in [web/music_video_builder/](web/music_video_builder):

1. **Follow ESM Conventions**: All files are `.mjs` modules. Use explicit imports and exports.
2. **Never store transient UI state in `session.json`**:
   - `session.json` is reserved for durable project data (scenes, prompts, assigned references, render versions).
   - Ephemeral UI state (e.g., active tab, dropdown visibility, timeline dragging offsets) should remain in frontend module variables.
3. **Use the Central ComfyUI API Fetch Helper**:
   - Make HTTP requests using `api.fetchApi(...)` so authentication tokens, reverse proxies, and ComfyUI path prefixes are properly handled.
4. **Cache Busting**:
   - When referencing rendered media assets, always append the scene's `video_cache_bust` query parameter (e.g., `/vrgdg/video_editor/video?path=...&cb=169823412`) so the browser reuses cached video during scrubbing but fetches fresh bytes immediately upon re-render.

---

### Recipe 5: Writing Unit & Integration Tests

All new functionality must be accompanied by tests in [tests/](tests):

1. **Use `unittest`**:
2. **Mock ComfyUI Dependencies**: Many modules import `server` or `folder_paths`. Use `importlib.util` or `unittest.mock.patch` to test backend components without booting the full ComfyUI server:

```python
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

# Dynamic loading of isolated modules
SPEC = importlib.util.spec_from_file_location("my_module", ROOT / "core/atomic_write.py")
my_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(my_module)

class MyFeatureTests(unittest.TestCase):
    def test_basic_behavior(self):
        self.assertTrue(hasattr(my_module, "atomic_write_json"))

if __name__ == "__main__":
    unittest.main()
```

3. **Run the Test Suite**:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 6. Critical Invariants & Anti-Patterns to Avoid

> [!CAUTION]
> Violating any of the following rules will introduce regressions, corrupt user projects, or cause ComfyUI startup failures.

1. **DO NOT introduce duplicate node names**: Every key in `NODE_CLASS_MAPPINGS` across all submodules must be unique. Run `python -m unittest tests/test_node_registration.py` after editing any node dictionary.
2. **DO NOT perform raw file writes for project state**: Never do `open("session.json", "w").write(...)`. If the process is terminated mid-write, the project is destroyed. Always use `atomic_write_json`.
3. **DO NOT block the server event loop**: Never call `time.sleep()`, synchronous `subprocess.run()`, or heavy tensor loops directly inside an async route handler. Use `await asyncio.to_thread(...)`.
4. **DO NOT hardcode path separators**: Never use string concatenation like `folder + "\\" + filename`. Always use `os.path.join(...)` or `pathlib.Path` to maintain cross-platform compatibility across Windows and Linux.
5. **DO NOT modify `optional_nodes/`**: The `optional_nodes` folder is deprecated/optional. Put all new nodes in their proper domain packages (`general/`, `builder/`, `runner/`, `minimax/`, `post_process/`, `llm/`).
6. **DO NOT forget `[VRGDG]` log prefixes**: All stdout/stderr output from custom nodes must begin with `[VRGDG]` so users and developers can filter custom node output from ComfyUI core logs.
7. **DO NOT create global state without mutex locks**: If shared state must be cached in memory (such as GPU resource readings in `system_routes.py`), protect updates with an `asyncio.Lock()`.
8. **DO NOT change or load an LLM model from the API.** For LM Studio use the model that is already loaded (`agent_api/llm_runtime.py`). Naming a model that is not loaded makes LM Studio load it and unload the current one.
9. **DO NOT let API-written project files differ from the Builder's.** Use the Builder's own functions and key names, finish renders as described under "Files the API writes", and set the session keys the UI reads. A project must open in the Video Builder as if it had been built there.
