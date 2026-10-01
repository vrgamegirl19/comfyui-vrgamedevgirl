# CLAUDE.md

## Primary Directive: Follow AGENT_GUIDE.md

Before modifying, designing, or debugging any code in this repository, you **MUST** read and adhere to the guidelines, architectural boundaries, and recipes documented in [AGENT_GUIDE.md](AGENT_GUIDE.md).

`AGENT_GUIDE.md` is the primary ground-truth technical specification for this repository. It defines:
- The dual execution paradigms (ComfyUI canvas node graph vs. headless API prompt execution via the Video Builder).
- The complete project map across all active core modules (`core/`, `builder/`, `runner/`, `llm/`, `minimax/`, `post_process/`, `storyboard/`, `prompt_creator/`, `general/`, `browser/`, `web/`, `agent_api/`, `mcp_server/`).
- Strict separation of concerns (SoC) rules.
- PEP 8 coding standards and repository-specific conventions.
- Developer recipes for adding nodes, routes, runner pipelines, and frontend features.

---

## Critical Rules & Invariants

1. **Exclude `optional_nodes/`**:
   - The `optional_nodes/` directory contains legacy and optional standalone modules. **Do NOT edit, reference, or import from `optional_nodes/`**.
   - Work strictly within the active core submodules defined in `_VRGDG_SUBMODULES` in [__init__.py](__init__.py).

2. **Atomic Disk Operations**:
   - **Never** write state files (e.g., `vrgdg_builder_session.json`, `storyboard.json`, settings) using raw `open(path, 'w')`.
   - **Always** use `atomic_write_json` or `atomic_write_text` from `core.atomic_write` to prevent corruption on unexpected termination.

3. **No Blocking on Event Loop**:
   - Server routes on `PromptServer.instance.routes` are asynchronous.
   - Never run blocking I/O, heavy CPU/GPU calculations, or synchronous subprocesses (`ffmpeg`, `nvidia-smi`) directly on the event loop. Always offload via `await asyncio.to_thread(...)`.

4. **Globally Unique Node Names**:
   - Node class names in `NODE_CLASS_MAPPINGS` must be globally unique across all submodules.
   - Always run the node registration test after touching node mappings:
     ```bash
     ..\..\..\python_embeded\python.exe tests/test_node_registration.py
     ```

5. **Separation of Concerns**:
   - **Canvas Nodes** (`nodes.py`): Only handle socket IO (`INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`). Delegate business logic to services.
   - **HTTP Routes** (`routes.py`): Only parse payloads, validate parameters, invoke services, and return standard JSON envelopes (`{"ok": True, ...}` / `{"ok": False, "error": str(exc)}`).
   - **Services** (`project.py`, `audio.py`, `media.py`, etc.): Implement pure business logic and disk operations.
   - **Graph Compiler** (`runner/*.py`): Build ComfyUI `/prompt` API graph dictionaries independently of UI state.
   - **Agent API** (`agent_api/`): Headless REST surface (`/vrgdg/api/v1`), background job manager, SSE progress streaming, and full pipeline orchestrator.
   - **MCP Server** (`mcp_server/`): Standard I/O bridge exposing tools (T1–T50), resources, and prompt templates to LLM coding assistants.
   - **Web UI** (`web/**/*.mjs`): Interact with the backend strictly via REST APIs and WebSockets; never directly access the filesystem.

6. **Consistent Logging**:
   - Prefix all stdout/stderr messages with `[VRGDG]` or `[VRGDG <Subsystem>]` (e.g., `[VRGDG Latent]`, `[VRGDG Clear Memory]`, `[VRGDG API]`).

7. **Agent API & MCP Server Invariants**:
   - **Agent API** is served at `/vrgdg/api/v1` and returns standard envelopes (`api_success` / `api_error`).
   - Structural timeline mutations must preserve on-disk file numbering (`image_NNNN.png`, `video_NNNN.mp4`) using `TimelineJournal` rollback protection and `_renumber_scene_assets_after_insert` / `_removal`.
   - Concurrency is protected via optimistic revision checking (`If-Match` header).
   - The standalone `mcp_server/` uses zero external dependencies (Python stdlib only) and communicates over stdio JSON-RPC 2.0 with actionable Rule 5 error handling (`isError=True`, `next_steps`). Launch with `python -m mcp_server`.

---

## Testing & Validation Commands

Run tests using the portable Python environment located at `..\..\..\python_embeded\python.exe`:

- **Verify node name uniqueness (no collisions)**:
  ```bash
  ..\..\..\python_embeded\python.exe tests/test_node_registration.py
  ```

- **Verify atomic write integrity**:
  ```bash
  ..\..\..\python_embeded\python.exe tests/test_atomic_write.py
  ```

- **Run all unit tests**:
  ```bash
  ..\..\..\python_embeded\python.exe -m unittest discover -s tests -p "test_*.py"
  ```

---

## Code Quality & Style

- **Python**: Strict PEP 8 compliance. 4 spaces indentation, type annotations on function signatures, Google/Sphinx style docstrings, 3-block import ordering (standard library, third-party, local package).
- **JavaScript**: Modern ES Modules (`.mjs`). Modular single-responsibility files in `web/music_video_builder/` and `web/storyboard_builder/`. Use `api.fetchApi(...)` for backend communication.
- **Media Streaming**: Append `video_cache_bust` query parameters when serving scene videos or images so browser cache reuses unchanged frames during timeline scrubbing.
