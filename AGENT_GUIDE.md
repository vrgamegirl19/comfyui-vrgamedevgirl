# 🤖 AI Agent Engineering Guide: comfyui-vrgamedevgirl

Welcome to **comfyui-vrgamedevgirl**. This repository is an enterprise-grade ComfyUI custom node suite and production application ecosystem. Its flagship tool is the **AI Video Builder**—a full digital audio workstation (DAW) and non-linear video editor (NLE) operating directly inside ComfyUI to produce AI music videos, narrative films, and cinematic sequences using engines like LTX-Video, MiniMax H3, FLUX, SDXL, and Z-Image.

This guide serves as the definitive technical manual for AI coding agents and human contributors. It provides an exhaustive map of the project architecture, detailed documentation of every active core file, guidelines for strict PEP 8 compliance, separation of concerns (SoC), and actionable recipes for extending the platform safely.

> [!NOTE]
> Per project directives, legacy and optional standalone modules located in `optional_nodes/` are intentionally omitted from this guide. All documentation here focuses on the active core runtime registered in [__init__.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/__init__.py).

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
2. **Headless API Graph Compilation**: The AI Video Builder frontend ([web/music_video_builder/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/web/music_video_builder)) acts as an integrated production studio. Instead of requiring users to wire up dozens of complex nodes manually, the builder dispatches HTTP commands to [runner/routes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/routes.py). The backend dynamically compiles complete execution graphs (in ComfyUI `/prompt` API format via [runner/api_graph.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/api_graph.py)), sends them to ComfyUI's internal queue, streams progress back via WebSockets, and captures rendered frames or video clips.

### Initialization Sequence

1. ComfyUI discovers this custom node folder and executes [__init__.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/__init__.py).
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
| [core/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/core) | Shared primitives: atomic file writes, wildcard sockets, model directory resolution, resource monitoring | `atomic_write.py`, `any_type.py`, `model_paths.py`, `system_routes.py` |
| [builder/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/builder) | AI Video Builder backend: session persistence, project branching, audio beat detection, media indexing, routes | `project.py`, `audio.py`, `media.py`, `paths.py`, `routes.py`, `nodes.py` |
| [runner/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner) | Dynamic workflow graph generation and rendering engine for LTX, MiniMax H3, Z-Image, Flux | `api_graph.py`, `ltx_workflows.py`, `minimax_workflows.py`, `routes.py` |
| [llm/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/llm) | Multi-provider LLM integrations (GGUF, API, Google), prompt expansion, agent chat, JSON validation | `api.py`, `gguf.py`, `builder_agent.py`, `image_prompt_generation.py` |
| [minimax/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/minimax) | MiniMax H3 video pipeline: latent caching, frame-token math, latent continuation, fast VAE decoding | `latent_manager.py`, `latent_continuation.py`, `latent_upscaler.py`, `nodes.py` |
| [post_process/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/post_process) | Face tracking, anchor enhancement, face paste-back compositing, 3D LUT grading, film grain | `face_fix.py`, `luts.py`, `lut_video_tools.py` |
| [storyboard/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/storyboard) | Storyboard generation, three-act structure planning, scene beats, dialogue allocation | `story_layer.py`, `scene_prompts.py`, `dialogue_scenes.py`, `nodes.py` |
| [prompt_creator/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/prompt_creator) | Structured prompt brainstorming, concept maps, motion notes, draft persistence routes | `nodes.py` |
| [general/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/general) | General-purpose canvas nodes: lyrics extraction (stable-ts), video analysis, audio stems, LoRA utilities | `lyrics.py`, `video.py`, `audio.py`, `utility.py`, `ltx_msr_reference.py` |
| [browser/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/browser) | External browser automation nodes bridging Flow, Meta AI, and ChatGPT image generators into ComfyUI | `nodes.py` |
| [flow_automation/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/flow_automation) | Node.js / Playwright / Puppeteer automation scripts for browser-driven generation workflows | `flow-poc.mjs`, `manual-bridge.mjs`, `meta-ai-poc.mjs` |
| [web/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/web) | ComfyUI web extensions, Video Builder application (79 ESM modules), Storyboard UI (22 ESM modules) | `music_video_builder/`, `storyboard_builder/`, `VRGDG_*.js` |
| [scripts/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/scripts) | Standalone tooling, workflow generation scripts, backport utilities, Photoshop integration | `build_minimax_h3_ref2va_2pass_audio_api.py`, `far_face_repair_backend.py` |
| [tests/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/tests) | 84 test suites covering Python backend logic, node collision checks, and JavaScript UI contracts | `test_node_registration.py`, `test_atomic_write.py`, `builder_source.py` |

---

### Core Infrastructure (`core/`)

The `core` package houses critical cross-cutting utilities used across the entire codebase.

#### [core/atomic_write.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/core/atomic_write.py)
- **Purpose**: Provides crash-safe atomic file writing. Prevents corruption of project files, session state, and settings if a crash, timeout, or power disruption occurs mid-write.
- **Key Functions**:
  - `atomic_write_text(path, content, encoding="utf-8")`: Writes content to a hidden sibling temporary file (`.filename.tmp`) using `tempfile.mkstemp`, performs `handle.flush()` and `os.fsync()`, and replaces the target path atomically via `os.replace()`. Cleans up temporary files if an exception is raised.
  - `atomic_write_json(path, value)`: Formats data with `json.dumps(value, indent=2, ensure_ascii=False)` and calls `atomic_write_text`.
- **Usage Rule**: **Never** use raw `open(path, 'w')` when writing state or configuration files. Always use `atomic_write_json` or `atomic_write_text`.

#### [core/any_type.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/core/any_type.py)
- **Purpose**: Implements the ComfyUI wildcard socket type pattern.
- **Key Symbols**:
  - `class AnyType(str)`: Overrides `__ne__(self, value)` to always return `False`.
  - `any_typ = AnyType("*")`: The singleton wildcard socket instance. When assigned to node inputs or outputs, it connects to any ComfyUI slot type regardless of type checking.

#### [core/model_paths.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/core/model_paths.py)
- **Purpose**: Manages custom model directory registration, persistence, and discovery outside the standard ComfyUI `models/` directory.
- **Key Functions**:
  - `load_custom_model_root()`: Reads `custom_model_root.json` from `VRGDG_Model_Defaults`.
  - `save_custom_model_root(value)`: Atomically writes a new custom model root directory.
  - `register_custom_model_root(root=None)`: Recursively scans and registers subfolders (`diffusion_models`, `unet`, `text_encoders`, `clip`, `vae`, `loras`, `upscale_models`, `latent_upscale_models`, `LLM`) into ComfyUI's central resolver using `folder_paths.add_model_folder_path`.

#### [core/system_routes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/core/system_routes.py)
- **Purpose**: Implements server-level system telemetry, GPU monitoring, process introspection, and memory management routes.
- **Key Endpoints**:
  - `GET /vrgdg/resource-monitor`: Queries host RAM (via `psutil`) and GPU metrics (via `nvidia-smi` without console popups) including utilization, VRAM usage, temperature, fan speed, clock frequencies, and power draw. Readings are protected by an async mutex and throttled to 1-second intervals.
  - `POST /vrgdg/resource-monitor/clear-memory`: Forces aggressive memory reclamation: runs Python `gc.collect()`, PyTorch `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()`, ComfyUI model cache clearing, and logs exact before/after RSS memory deltas.
  - `GET /vrgdg/update/v10/status` & `POST /vrgdg/update/v10`: Provides self-updating mechanisms for the node pack.
  - `GET /vrgdg/video_builder/custom_nodes/status` & `POST /vrgdg/video_builder/custom_nodes/install`: Allowlist-based dependency checker and installer for prerequisite custom nodes.

---

### AI Video Builder Engine (`builder/`)

The `builder` package contains the backend business logic and HTTP API powering the **AI Video Builder UI**.

#### [builder/nodes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/builder/nodes.py)
- **Purpose**: Defines the primary canvas node `VRGDG_MusicVideoBuilderUI` and triggers the registration of builder server routes.
- **Node Registered**:
  - Class: `VRGDG_MusicVideoBuilderUI`
  - Display Name: `VRGDG Music Video Builder UI`
  - Category: `VRGDG/UI`
  - Inputs: `audio_path`, `project_folder`, `session_path`, `srt_path`
  - Outputs: `(project_folder, session_path, srt_path)`

#### [builder/routes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/builder/routes.py)
- **Purpose**: The central API gateway for the Video Builder frontend. Implements over 50 async REST endpoints on `server.PromptServer.instance.routes`.
- **Key Route Categories**:
  - **Audio & Beat Analysis**: `/vrgdg/music_builder/analyze_audio`, `/vrgdg/music_builder/import_capcut_beats`, `/vrgdg/music_builder/save_scene_audio`, `/vrgdg/music_builder/trim_scene_audio`, `/vrgdg/music_builder/create_silent_audio`, `/vrgdg/music_builder/prepare_scene_audio_mix`.
  - **Session & Project State**: `/vrgdg/music_builder/save_session`, `/vrgdg/music_builder/load_session`, `/vrgdg/music_builder/new_project`, `/vrgdg/music_builder/save_project_as`, `/vrgdg/music_builder/delete_project`, `/vrgdg/music_builder/list_projects`, `/vrgdg/music_builder/export_project`, `/vrgdg/music_builder/import_project`.
  - **Media Asset Management**: `/vrgdg/music_builder/save_scene_image`, `/vrgdg/music_builder/archive_scene_image`, `/vrgdg/music_builder/extract_video_final_frame`, `/vrgdg/music_builder/scan_scene_videos`, `/vrgdg/music_builder/restore_scene_video`.
  - **Latent Lifecycle**: `/vrgdg/music_builder/latent_status`, `/vrgdg/music_builder/check_latent_predecessor`, `/vrgdg/music_builder/delete_scene_latent`, `/vrgdg/music_builder/list_dirty_latents`.
  - **Timeline Asset Renumbering**: `/vrgdg/music_builder/renumber_scenes_after_removal`, `/vrgdg/music_builder/renumber_scenes_after_insert`.
  - **LLM Prompt Generation & Agent**: `/vrgdg/music_builder/generate_t2i`, `/vrgdg/music_builder/generate_i2v`, `/vrgdg/music_builder/generate_chained_i2v`, `/vrgdg/music_builder/generate_t2v`, `/vrgdg/music_builder/agent_chat`, `/vrgdg/music_builder/flux_reference_extract_subjects`, `/vrgdg/music_builder/flux_reference_extract_locations`.

#### [builder/project.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/builder/project.py)
- **Purpose**: Project serialization, initialization, asset renumbering, and migration logic.
- **Key Functions**:
  - `_new_builder_project(payload)`: Sets up directory scaffolding (`project_audio`, `project_images`, `scene_videos`, etc.) and writes the initial `session.json`.
  - `_load_builder_session(payload)` / `_save_builder_session(payload)`: Reads and atomically persists project scene data, timeline markers, and render flags.
  - `_renumber_scene_assets_after_insert(project_folder, inserted_index)`: Renumbers all disk assets (`scene_XXX.*`) backwards from the end to make room for an inserted scene without overwriting existing files.
  - `_renumber_scene_assets_after_removal(project_folder, removed_index)`: Shifts all subsequent assets forward by one to close the gap left by a deleted scene.

#### [builder/project_copy.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/builder/project_copy.py)
- **Purpose**: Non-destructive project duplication and branching.
- **Key Functions**:
  - `_save_builder_project_as(payload)`: Duplicates a project folder, selectively filters scene media based on user choices (e.g., keep approved images only, discard failed video renders), rewrites internal path references inside `session.json`, and clones serialized MiniMax latents.

#### [builder/audio.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/builder/audio.py)
- **Purpose**: Digital audio processing, waveform analysis, and beat detection.
- **Key Functions**:
  - `_read_audio_peaks(audio_path, target_peaks=1600)`: Extracts downsampled waveform peak envelopes for high-performance frontend timeline rendering.
  - `_estimate_beats_from_audio(audio_path, ...)`: Leverages `librosa` to compute audio onset envelopes, extract tempo (BPM), and determine precise beat frame timings.
  - `_convert_audio_to_wav(audio_path, target_path)`: Converts incoming audio formats (MP3, M4A, FLAC) to uncompressed 16-bit 44.1/48kHz WAV via `av` or `ffmpeg`.
  - `_trim_scene_audio(...)` & `_prepare_scene_audio_mix(...)`: Slices audio segments matching scene start/duration parameters and mixes final multi-track audio for video assembly.

#### [builder/media.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/builder/media.py)
- **Purpose**: Visual asset tracking, frame extraction, and reference media management.
- **Key Functions**:
  - `_extract_video_final_frame_as_scene_image(video_path, target_image_path)`: Uses `torchcodec` or `cv2` to grab the exact final frame of a rendered scene video to use as the starting frame of the subsequent scene (First/Last Frame continuity).
  - `_archive_scene_image(project_folder, scene_num)`: Moves superseded scene images to a history folder before new iterations overwrite them.
  - `_scan_builder_scene_videos(project_folder)`: Traverses scene video directories and indexes render versions, timestamps, and resolutions.

#### [builder/paths.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/builder/paths.py)
- **Purpose**: Path sanitation, directory containment checks, native OS dialog integration.
- **Key Functions**:
  - `_resolve_existing_file(path, label)`: Validates that a file exists and normalizes Windows/POSIX path separators.
  - `_open_native_picker(type="file", ...)`: Spawns the operating system's native file/folder explorer dialog.
  - `_open_local_file(path)`: Launches the default system media player or image viewer for a rendered asset.

#### [builder/video_editor.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/builder/video_editor.py)
- **Purpose**: Static asset delivery endpoints for high-throughput video/image timeline streaming.
- **Key Endpoints**:
  - `GET /vrgdg/video_editor/video?path=<path>`: Streams video files with `Cache-Control: public, max-age=31536000, immutable` headers keyed to scene cache busters to enable instant browser timeline playback without re-fetching.
  - `GET /vrgdg/video_editor/image?path=<path>`: Serves image files and thumbnails.

---

### Workflow Runner & Graph Compiler (`runner/`)

The `runner` package is the code generation and execution engine that converts project settings into executable ComfyUI workflow prompt graphs.

#### [runner/nodes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/nodes.py)
- **Purpose**: Canvas UI nodes for workflow runners and system memory clearing.
- **Nodes Registered**:
  - `VRGDG_MiniMaxH3TurboLoRACompat`: "VRGDG MiniMax-H3 Turbo LoRA Compatibility"
  - `VRGDG_ZImageWorkflowRunnerUI`: "VRGDG Z-Image Workflow Runner UI"
  - `VRGDG_ClearMemoryButtonUI`: "VRGDG Clear Memory Button"

#### [runner/routes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/routes.py)
- **Purpose**: Compiles requested workflows and coordinates background generation jobs.
- **Key Endpoints**:
  - `/vrgdg/workflow_runner/build_zimage_prompt`: Assembles Z-Image / SDXL / Flux prompt graphs.
  - `/vrgdg/workflow_runner/build_i2v_prompt` & `build_t2v_prompt`: Assembles LTX-Video generation graphs.
  - `/vrgdg/workflow_runner/build_minimax_h3_prompt`, `build_minimax_h3_2pass_prompt`, `build_minimax_h3_advanced_2pass_prompt`: Assembles single- and multi-pass MiniMax H3 generation graphs.
  - `/vrgdg/workflow_runner/build_flf_prompt`: Assembles First/Last Frame guided interpolation graphs.
  - `/vrgdg/workflow_runner/collect_scene_video`: Locates rendered outputs in ComfyUI output folders and imports them into the project.
  - `/vrgdg/workflow_runner/match_scene_video_start_color`: Matches output video start frames to source image color profiles to prevent color shifts.
  - `/vrgdg/workflow_runner/stitch_scene_videos`: Invokes `ffmpeg` to stitch all approved scene video clips into a single continuous video with muxed audio.

#### [runner/api_graph.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/api_graph.py)
- **Purpose**: Graph construction builder primitives. Generates unique string node IDs, links inputs between nodes, and formats the output into the standard ComfyUI API schema: `{node_id: {"class_type": ..., "inputs": {...}}}`.

#### [runner/ltx_workflows.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/ltx_workflows.py)
- **Purpose**: Compiler for LTX-Video pipelines (Text-to-Video, Image-to-Video, First/Last Frame guidance, STG guidance, frame rate, and aspect ratio conditioning).

#### [runner/minimax_workflows.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/minimax_workflows.py)
- **Purpose**: Compiler for MiniMax H3 pipelines. Assembles model loading, text conditioning, image reference attachment, audio drive conditioning, latent upscale stages, and fast VAE decoding.

#### [runner/minimax_inputs.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/minimax_inputs.py)
- **Purpose**: Input parsing, validation, and token/frame duration alignment specifically for MiniMax H3 executions.

#### [runner/minimax_patches.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/minimax_patches.py)
- **Purpose**: Injects optional patches (such as Turbo LoRA or camera motion control weights) into MiniMax H3 model nodes.

#### [runner/image_workflows.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/image_workflows.py)
- **Purpose**: Compiles prompt graphs for image generators (Flux Schnell/Dev, Z-Image, SDXL) with support for reference conditioning images.

#### [runner/utility_workflows.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/utility_workflows.py)
- **Purpose**: Compiles background helper workflows such as Whisper speech-to-text transcription and standalone memory cleanup prompts.

#### [runner/video_files.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/video_files.py)
- **Purpose**: Output file handling, video trimming, format verification, and ffmpeg assembly.

#### [runner/models.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/models.py)
- **Purpose**: Dataclasses and type definitions representing model configurations, samplers, schedulers, and resolution settings.

---

### Unified LLM Subsystem (`llm/`)

The `llm` package provides unified text generation, prompt rewriting, concept generation, and interactive agent capabilities across multiple backends.

#### [llm/api.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/llm/api.py)
- **Purpose**: Unified multi-provider API client node. Supports OpenAI, Anthropic Claude, Google Gemini, Grok (xAI), Ollama, and LM Studio.
- **Node Registered**:
  - `VRGDG_LLM_Multi`: "🤖 VRGDG LLM Multi 🤖"
- **Features**: Handles multimodal image inputs, system instructions, temperature/seed control, structured JSON schema enforcement, and retry loops.

#### [llm/gguf.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/llm/gguf.py)
- **Purpose**: Direct local GGUF model execution using `llama-cpp-python` with CUDA acceleration.
- **Nodes Registered**:
  - `VRGDG_QwenGGUF`: "🧠 VRGDG Qwen GGUF 🧠"
  - `VRGDG_SuperGemmaGGUFChat`: "🧠 VRGDG SuperGemma GGUF Chat 🧠"
  - `VRGDG_UnloadGemmaModels`: "VRGDG Unload Gemma/GGUF Models"

#### [llm/google.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/llm/google.py)
- **Purpose**: Google Gemini and Imagen integration.
- **Node Registered**:
  - `VRGDG_NanoBananaPro`: "🚀 VRGDG NanoBanana Pro 🚀"

#### [llm/builder_agent.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/llm/builder_agent.py)
- **Purpose**: Conversational AI assistant logic embedded inside the Video Builder UI. Handles user queries about project planning, scene direction, prompt critique, and shot progression.

#### [llm/builder_instructions.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/llm/builder_instructions.py)
- **Purpose**: Manages system prompts, persona templates, and instruction presets for image and video prompt generation.

#### [llm/builder_runner.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/llm/builder_runner.py)
- **Purpose**: Dispatch router for builder LLM tasks. Dynamically routes requests to LM Studio, local GGUF, or cloud APIs with automatic fallback and seed retry.

#### [llm/image_prompt_generation.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/llm/image_prompt_generation.py)
- **Purpose**: Domain-specific prompt expansion for image models (Flux, Z-Image, SDXL). Generates consistent character descriptions, architectural lighting details, and color palettes from raw lyrics or concept notes.

#### [llm/video_prompt_generation.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/llm/video_prompt_generation.py)
- **Purpose**: Domain-specific prompt expansion for video models (LTX, MiniMax). Crafts dynamic camera motions (pan, tilt, crane, dolly, orbit), action beats, physical dynamics, and scene transitions.

#### [llm/output_checks.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/llm/output_checks.py)
- **Purpose**: Resilient JSON parsing and validation. Extracts JSON payloads enclosed in markdown code fences, fixes trailing commas, validates schema fields, and recovers gracefully from truncated responses.

#### [llm/text_cleaning.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/llm/text_cleaning.py)
- **Purpose**: Strips conversational preambles, trailing commentary, markdown formatting, and hallucinated prompt tags from raw LLM outputs.

#### [llm/cache.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/llm/cache.py)
- **Purpose**: In-memory and disk-based caching for prompt expansions to minimize redundant API costs.

---

### MiniMax H3 Video Engine (`minimax/`)

The `minimax` package implements high-performance conditioning, latent management, and decoding nodes for the MiniMax H3 video architecture.

#### [minimax/latent_manager.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/minimax/latent_manager.py)
- **Purpose**: Serialized latent storage and frame-token math for MiniMax H3. Eliminates pixel-space VAE re-encoding drift across chained scene passes.
- **Key Symbols**:
  - `_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`: MiniMax H3 temporal latent token compression pattern.
  - `_tokens_to_frames(token_count)` & `_frames_to_tokens(frame_count)`: Performs exact conversions between video frame counts and temporal latent tokens.
  - `class SceneLatentManager`: Manages `.latent` safetensors files in `latents/`, tracks dirty flags, handles predecessor dependencies, and renames latent files when scenes are reordered.
  - `scene_latent_manager`: Singleton instance.

#### [minimax/nodes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/minimax/nodes.py)
- **Purpose**: Core canvas nodes for MiniMax H3 generation.
- **Nodes Registered**:
  - `H3FastVAEDecode`: "H3 VAE Decode Fast (Batched Tiles)" — Memory-efficient tiled VAE decoder preventing out-of-memory errors on long clips.
  - `VRGDG_MiniMaxH3AudioDrive`: "VRGDG MiniMax H3 Audio Drive" — Injects audio waveform conditioning into MiniMax H3 to drive motion/sync.
  - `VRGDG_MiniMaxH3ReferenceMediaFromPaths`: "VRGDG MiniMax H3 Reference Media From Paths" — Binds character and background reference images.
  - `VRGDG_MiniMaxH3ImageReferenceToVideo`: "MiniMax H3 Image + Reference to Video" — High-level conditioning node for Image-to-Video with multiple reference images.

#### [minimax/latent_continuation.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/minimax/latent_continuation.py)
- **Purpose**: Native latent-space continuation between adjacent scenes.
- **Nodes Registered**:
  - `VRGDG_MiniMaxH3SaveLatent`: "VRGDG H3 Save Latent"
  - `VRGDG_MiniMaxH3LoadLatent`: "VRGDG H3 Load Latent"
  - `VRGDG_MiniMaxH3ApplyLatentGuide`: "VRGDG H3 Apply Latent Continuation Guide"
  - `VRGDG_MiniMaxH3LoadExactFrame`: "VRGDG H3 Load Exact Last Frame"

#### [minimax/latent_upscaler.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/minimax/latent_upscaler.py)
- **Purpose**: Learned latent upscaling nodes operating directly on MiniMax latent tensors.
- **Nodes Registered**:
  - `VRGDG_MiniMaxH3LatentUpscaleModelLoader`: "Load MiniMax H3 Learned Latent Upscaler"
  - `VRGDG_MiniMaxH3UltimateUpscaleParams`: "MiniMax H3 Ultimate Upscale Params (VRGDG)"
  - `VRGDG_MiniMaxH3LearnedLatentUpscale`: "MiniMax H3 Learned Latent Upscale"
  - `VRGDG_MiniMaxH3ReplaceUpscaledVideoLatent`: "MiniMax H3 Replace with Upscaled Video Latent"

---

### Post-Processing & Enhancement (`post_process/`)

#### [post_process/face_fix.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/post_process/face_fix.py)
- **Purpose**: High-precision video face repair backend. Tracks faces across video frames, extracts guided anchor crops, processes crops through enhancement models, and composites repaired faces back into the source video with feathered edge masks.
- **Key Functions**:
  - `register_face_fix_routes(server_instance)`: Registers endpoints `/vrgdg/face_fix/detect_anchors`, `/vrgdg/face_fix/accept_ltx_frames`, and `/vrgdg/face_fix/finalize`.

#### [post_process/luts.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/post_process/luts.py)
- **Purpose**: 3D LUT (.cube) parsing and PyTorch tensor application with strength blending.

#### [post_process/lut_video_tools.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/post_process/lut_video_tools.py)
- **Purpose**: Video grading and preview routes.
- **Key Functions**:
  - `register_lut_routes(server_instance)`: Registers endpoints for LUT discovery (`/vrgdg/music_builder/luts`), LUT image/video application (`apply_image`, `apply_video`), procedural film grain generation, and color adjustments (brightness, contrast, saturation).

---

### Storyboard Subsystem (`storyboard/`)

The `storyboard` package provides script breakdown, shot planning, and narrative structure tools.

#### [storyboard/nodes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/storyboard/nodes.py)
- **Purpose**: Storyboard canvas UI node and HTTP route registration.
- **Node Registered**:
  - `VRGDG_StoryboardBuilderUI`: "VRGDG Storyboard Builder UI"
- **Key Routes**: `/vrgdg/storyboard/load`, `/vrgdg/storyboard/save`, `/vrgdg/storyboard/story_brief`, `/vrgdg/storyboard/story_arc`, `/vrgdg/storyboard/scene_story_beat`, `/vrgdg/storyboard/export_prompts`.

#### [storyboard/story_layer.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/storyboard/story_layer.py)
- **Purpose**: Narrative arc management. Breaks stories into Three-Act structures, defines emotional intensity curves, and maps plot points to visual beats.

#### [storyboard/scene_prompts.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/storyboard/scene_prompts.py)
- **Purpose**: Translates high-level storyboard cards into specific visual prompt strings for image generation and video rendering.

#### [storyboard/dialogue_scenes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/storyboard/dialogue_scenes.py)
- **Purpose**: Parses scripts and song lyrics to identify character dialogue, attribute speakers, and time scene transitions.

#### [storyboard/persistence.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/storyboard/persistence.py)
- **Purpose**: Serialization of storyboard cards, beats, and shot configurations to `storyboard.json` using atomic writes.

#### [storyboard/scene_helpers.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/storyboard/scene_helpers.py)
- **Purpose**: Math and timing utilities for calculating scene lengths, frame offsets, and shot classifications.

---

### Prompt Creator Subsystem (`prompt_creator/`)

#### [prompt_creator/nodes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/prompt_creator/nodes.py)
- **Purpose**: Registers backend HTTP API routes for the interactive Prompt Creator tool (`/vrgdg/music_prompt_creator/*`): concept creation, motion notes extraction, segment repair, draft saving/loading, and Whisper prompt generation.

---

### General Custom Nodes (`general/`)

The `general` package contains modular canvas nodes that can be placed in any standard ComfyUI workflow.

#### [general/lyrics.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/general/lyrics.py)
- **Purpose**: Lyric transcription, alignment, and SRT segment processing.
- **Nodes Registered**:
  - `VRGDG_PromptTemplateBuilder`: Assembles multi-token prompt templates with variable replacement.
  - `VRGDG_ManualLyricsExtractor_SRT`: Extracts text and timing from standard SRT subtitle files.
  - `VRGDG_ManualLyricsExtractor_SRT_Advanced`: Advanced SRT extraction with character identification.
  - `VRGDG_TimestampedLyricsExtractor`: Generates word-level timestamped SRT subtitles directly from audio using `stable-ts` / Whisper.

#### [general/video.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/general/video.py)
- **Purpose**: Video analysis, beat-aligned scene sizing, image sequence management.
- **Nodes Registered**:
  - `VRGDG_BuildVideoOutputPath_General_SRT`: Generates deterministic, formatted output file paths.
  - `BeatImpactAnalysisNode`: Analyzes audio transients to detect cut opportunities.
  - `BeatSceneDurationNode`: Quantizes scene durations to musical bars and beats.
  - `IndexedImageFromFolder_ForRemakeMode`: Sequentially iterates image directories by index.
  - `VRGDG_LatestSRTAutoLoader`: Automatically finds and loads the newest SRT file in a directory.
  - `VRGDG_LoadAudioSplit_SRTOnly`: Extracts audio matching an SRT segment's time bounds.
  - `VRGDG_TrimImageBatch_SRTOnly`: Trims image batches to match target video frame counts.

#### [general/audio.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/general/audio.py)
- **Purpose**: Audio manipulation and stem separation nodes.
- **Nodes Registered**:
  - `VRGDG_AudioCrop`: Crops audio tensors by start and end timestamps.
  - `VRGDG_GetStems`: Performs 4-stem separation (vocals, drums, bass, other) using `demucs`.

#### [general/utility.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/general/utility.py)
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

#### [general/ltx_msr_reference.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/general/ltx_msr_reference.py)
- **Purpose**: Multi-Scale Reference (MSR) conditioning builder for LTX-Video.
- **Nodes Registered**:
  - `VRGDG_LTXMSRReferenceBuilder`: "VRGDG LTX MSR Reference Builder"
  - `VRGDG_LTX25MSRReferenceLoader`: "VRGDG LTX 2.5 MSR Reference Loader"

---

### Browser AI Automation (`browser/` & `flow_automation/`)

#### [browser/nodes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/browser/nodes.py)
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

ComfyUI automatically loads all extensions placed in [web/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/web) because `WEB_DIRECTORY = "./web"` is declared in [__init__.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/__init__.py).

#### Standalone Extension Scripts
- [web/VRGDG_MusicVideoBuilderUI.js](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/web/VRGDG_MusicVideoBuilderUI.js): Entry point registering the Video Builder modal dialog on the ComfyUI canvas node.
- [web/VRGDG_StoryboardBuilderUI.js](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/web/VRGDG_StoryboardBuilderUI.js): Entry point registering the Storyboard Builder modal dialog.
- [web/VRGDG_ResourceMonitor.js](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/web/VRGDG_ResourceMonitor.js): Canvas and top-bar widget rendering live host RAM and NVIDIA VRAM gauges.
- [web/VRGDG_UIThemes.js](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/web/VRGDG_UIThemes.js): Theme engine supporting dark, cinematic, and modern styling tokens across VRGDG UI dialogs.
- [web/VRGDG_FaceFixUI.js](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/web/VRGDG_FaceFixUI.js): Interactive face-repair preview and anchor selection interface.
- [web/VRGDG_RenderETA.js](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/web/VRGDG_RenderETA.js): Calculates and displays remaining render time based on historical frame rendering rates.

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

### Utility Scripts (`scripts/`)

- [scripts/build_minimax_h3_ref2va_2pass_audio_api.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/scripts/build_minimax_h3_ref2va_2pass_audio_api.py): Offline standalone generator for 2-pass MiniMax H3 reference-to-video API workflow JSON files.
- [scripts/build_ltx25_normal_sampler_workflow.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/scripts/build_ltx25_normal_sampler_workflow.py): Generates baseline LTX 2.5 sampler API graphs.
- [scripts/far_face_repair_backend.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/scripts/far_face_repair_backend.py): Standalone backend testing script for small/distant face detection and super-resolution repair.
- [scripts/Backport-Krea2ToMusubi.ps1](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/scripts/Backport-Krea2ToMusubi.ps1): PowerShell script for migrating dataset captions and LoRA configurations between training engines.

---

### Test Suites (`tests/`)

The repository includes 84 test suites verifying backend logic, API contracts, and JavaScript source code integrity.

- [tests/test_node_registration.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/tests/test_node_registration.py): Parses AST of all submodules in `_VRGDG_SUBMODULES` and asserts that every node identifier in `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` is globally unique. **Must always pass.**
- [tests/test_atomic_write.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/tests/test_atomic_write.py): Simulates disk interruptions, asserts file replacement integrity, and verifies temp file cleanup.
- [tests/test_builder_branch_project.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/tests/test_builder_branch_project.py): Tests non-destructive project cloning and path rewriting.
- [tests/test_minimax_h3_latent_continuation.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/tests/test_minimax_h3_latent_continuation.py): Validates frame-to-token conversions, latent tensor guide alignment, and safetensors persistence.
- [tests/builder_source.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/tests/builder_source.py): AST and regex extraction helper allowing Python unit tests to validate frontend JavaScript logic and contracts without spinning up a headless browser.

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

1. **Choose the appropriate package** (e.g., [general/utility.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/general/utility.py), [general/video.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/general/video.py), or [minimax/nodes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/minimax/nodes.py)).
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

4. **Verify [__init__.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/__init__.py)**: Ensure the file containing the node is listed in `_VRGDG_SUBMODULES`.
5. **Run the node registration test**:
```bash
python -m unittest tests/test_node_registration.py
```

---

### Recipe 2: Registering a Backend HTTP Route

When adding a backend route:

1. **Locate the appropriate `routes.py`** (e.g., [builder/routes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/builder/routes.py) or [runner/routes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/routes.py)).
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

1. **Define the model parameters in [runner/models.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/models.py)**.
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

3. **Expose the compiler via [runner/routes.py](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/runner/routes.py)**:
   - Add a POST route `/vrgdg/workflow_runner/build_my_model_prompt` that accepts the JSON payload, calls your compiler, and returns the assembled graph.

---

### Recipe 4: Modifying Builder Frontend Modules

When updating or adding UI features in [web/music_video_builder/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/web/music_video_builder):

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

All new functionality must be accompanied by tests in [tests/](file:///c:/Users/NVMax/Desktop/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-vrgamedevgirl/tests):

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
