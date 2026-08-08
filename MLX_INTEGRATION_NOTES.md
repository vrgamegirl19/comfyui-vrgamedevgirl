# MLX Integration — Session Notes

Context dump covering every Apple Silicon / MLX engine added to this repo, across
multiple sessions. Originally scoped to just LTX-2 video (hence any file paths that
still say "ltx2mlx" below); renamed to `MLX_INTEGRATION_NOTES.md` once FLUX.2 Klein,
Krea-2, and Z-Image MLX engines were added too, since the same environment setup,
node-pack pattern, and 3-layer testing methodology apply to all of them. Written so
none of this has to be re-derived after a context reset.

Sessions documented here, oldest first:
1. **LTX-2 MLX** (video, MusicVideoBuilder third render engine) — original content below.
2. **FLUX.2 Klein / Krea-2 / Z-Image MLX** (images, Reference Builder "Create Subject"
   engines) — see [Krea-2 + Z-Image MLX Integration](#krea-2--z-image-mlx-integration----session-notes)
   near the end of this file.

## Why this exists

`ComfyUI-LTXVideo`'s fp8 checkpoint (`ltx-2.3-22b-dev-fp8.safetensors`) only runs on
CUDA or (very slowly) CPU. Confirmed directly: PyTorch's MPS backend has no kernel for
`float8_e4m3fn` at all —

```
RuntimeError: Undefined type Float8_e4m3fn
```

— and a bf16 fallback doesn't fit comfortably on a 64GB Mac for a 22B model. The fix
isn't a PyTorch/MPS patch; it's routing generation through
[dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx), a pure MLX port of LTX-2.3
with its own quantization (q4/q8) built for Apple's unified memory.

## Two repos touched

1. **`comfyui-ltx2-mlx`** (new standalone repo, sibling to this one under `custom_nodes/`,
   branch/repo not yet pushed to GitHub — local only as of this session). Wraps
   `ltx-2-mlx` as real ComfyUI nodes. MIT licensed, meant to be useful to anyone on
   Apple Silicon, not just MusicVideoBuilder users.
2. **`comfyui-vrgamedevgirl`** (this repo, `mlx` branch). Wires the above in as a third
   `project_video_engine` option (`ltx` / `minimax_h3` / `ltx2mlx`) using the exact same
   build-prompt → queue → poll-history → collect-video pattern already used for LTX and
   MiniMax H3.

## Environment setup

ComfyUI's main venv here is Python 3.10; `ltx-core-mlx`/`ltx-pipelines-mlx` require
Python ≥3.11. Rather than touching the shared production venv (risky — ~50 other custom
nodes depend on it), a second venv was created:

```bash
cd /Volumes/CORSAIR/StabilityMatrix/Packages/ComfyUI
/opt/homebrew/bin/python3.13 -m venv venv-3.13
./venv-3.13/bin/pip install -r requirements.txt                              # ComfyUI core
./venv-3.13/bin/pip install -r custom_nodes/comfyui-ltx2-mlx/requirements.txt # mlx + ltx-2-mlx
./venv-3.13/bin/pip install -r custom_nodes/ComfyUI-Manager/requirements.txt
./venv-3.13/bin/pip install -r custom_nodes/comfyui-vrgamedevgirl/requirements.txt
```

`llama-cpp-python` (in vrgamedevgirl's requirements) fails to build from source on this
machine — first with `-arch x86_64` (needs `CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64"`),
then with a linker error (`ld: symbol(s) not found for architecture arm64` for OpenSSL
X509 symbols in llama.cpp's bundled `cpp-httplib`) that persisted even pointing at
Homebrew OpenSSL. **Left unresolved** — it's an unrelated environment/toolchain issue.
Harmless for this feature: `LLM.py` is the only file that imports it, and every submodule
in `comfyui-vrgamedevgirl/__init__.py` is loaded in its own try/except, so one missing dep
doesn't block the modules that matter here (`VRGDG_WorkflowRunnerNodes`,
`VRGDG_MusicVideoBuilderNodes`, `VRGDG_StoryboardBuilderNodes` all load fine without it).

**Model weight download location** — HF cache defaults to `~/.cache/huggingface`, which
was tight on disk. Redirected via:

```bash
export HF_HOME="/Volumes/CORSAIR/StabilityMatrix/Models/LTX2MLX"
```

Set this before launching ComfyUI (or the venv-3.13 Python directly) so weights land in
the shared Models folder instead of the internal disk. The q4 model + Gemma-3-12B-4bit
text encoder together came to ~63GB on disk (larger than the ~19GB naive estimate —
includes VAE/audio/vocoder decoder weights too). **Disk space was tight throughout this
session (30-40GB free range)** — check before pulling q8 (~21GB) or bf16 (~42GB) tiers.

Launch command used for testing:

```bash
cd /Volumes/CORSAIR/StabilityMatrix/Packages/ComfyUI
HF_HOME=/Volumes/CORSAIR/StabilityMatrix/Models/LTX2MLX \
  ./venv-3.13/bin/python3 main.py --port 8199 --listen 127.0.0.1
```

(Use a Bash tool call with `run_in_background: true` for this — a manual shell `&` gets
killed when that tool call ends, even with `disown`. Learned this the hard way: first
server launch died silently mid-startup because of exactly that.)

## comfyui-ltx2-mlx: node design

Four `io.ComfyNode` (v3 schema) nodes, registered via a small `nodes_registry.py`
(`comfy_node` decorator) mirroring the pattern in `ComfyUI-LTXVideo`:

- `LTX2MLXModelLoader` → `LTX2MLX_PIPELINE` (T2V/I2V pipelines: two_stage, two_stage_hq,
  one_stage, distilled)
- `LTX2MLXGenerate` — text-to-video, or image-to-video if `image` is connected
- `LTX2MLXAudioModelLoader` → `LTX2MLX_A2V_PIPELINE`
- `LTX2MLXAudioToVideo` — audio-driven generation (the actual song-mode use case),
  with a `match_audio_length` toggle that snaps video duration to the input audio's
  actual length (respecting the VAE's `8k+1` frame constraint)

Only one pipeline is cached/resident at a time — loading a new one evicts the previous
(the 22B model doesn't fit twice even at q4/q8).

**Real API signature bugs found only by importing the actual installed package**
(the GitHub README was stale/wrong in a couple of places):
- Constructor kwarg is `low_ram_streaming`, not `low_ram`.
- `generate_and_save` requires `frame_rate` as a **mandatory keyword-only** arg — no
  default. The first draft of `LTX2MLXGenerate` never passed it and would have crashed
  immediately on first real use.
- `LTX2MLXAudioToVideo` originally had no `height`/`width`/`num_frames` inputs at all and
  silently used the pipeline's default ~4s video length regardless of audio length —
  crashed with a RoPE shape mismatch (`Shapes (1,32,26,32) and (1,32,101,32) cannot be
  broadcast`) when fed 1s of audio. Fixed by adding explicit dimension controls plus
  `match_audio_length`.

**Output node pattern**: both generation nodes mark `is_output_node=True` and return
`io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(...)]))`, saving through
`folder_paths.get_save_image_path()` with a `filename_prefix` — the same convention the
stock `SaveVideo` node uses. This means ComfyUI's native `/history` endpoint reports the
output automatically; no separate `SaveVideo` node is needed downstream, and
MusicVideoBuilder's existing `waitForVideos`/`extractVideosFromHistory` JS just works
against it unmodified.

## comfyui-vrgamedevgirl: integration points

- **`VRGDG_WorkflowRunnerNodes.py`**: `_build_ltx2mlx_api_prompt(payload)` (dispatches to
  t2v/i2v vs a2v based on `ltx2mlx_mode`), two new API-format workflow JSON templates in
  `Workflows/UsedForUIDoNotTouch/` (`ltx2mlx_t2v_i2v_API.json`, `ltx2mlx_a2v_API.json`),
  a `_require_ltx2mlx_available()` capability check (macOS+arm64 and required node
  classes registered — mirrors the MiniMax Turbo missing-node guard), routes
  `build_ltx2mlx_prompt` / `ltx2mlx_capability` / `trim_scene_audio` (the last one is a
  generic, reusable version of the MiniMax-specific audio-trim helper — used by Song
  Mode to cut each scene's exact audio slice before generation).
- **`VRGDG_StoryboardBuilderNodes.py`**: `project_video_engine` normalizer extended to
  3-way (`ltx`/`minimax_h3`/`ltx2mlx`). Storyboard doesn't get its own LTX2MLX-specific
  reference/video-to-video controls (that's MiniMax-only UI); `ltx2mlx` there just
  behaves like plain `ltx`.
- **`VRGDG_MusicVideoBuilderUI.js`**: new `ltx2MlxEnginePanel` (model/pipeline/dimensions/
  song-mode toggle), 3-way `normalizeProjectVideoEngine`, capability check on selection
  (hits the `ltx2mlx_capability` route once, shows a warning note if unavailable),
  `renderLtx2MlxSceneVideoWithProgress` (mirrors `renderMiniMaxSceneVideoWithProgress`'s
  structure: build prompt → `queueWorkflowPrompt` → `waitForVideos` → `collect_scene_video`),
  a standalone "Create LTX-2 MLX Scene Video" button, batch "Render All" dispatch,
  session save/load of engine + settings, `validateRenderAllReady` coverage.

## Testing methodology (do this again after any further changes)

Three layers, each catching different bugs — don't skip to the top layer:

1. **Direct Python calls** to the node classes (`Loader.execute(...)`,
   `Generate.execute(...)`) — fast, catches signature mismatches against the real
   installed package.
2. **Real ComfyUI server + curl** — launch the server, POST to
   `/vrgdg/workflow_runner/build_ltx2mlx_prompt`, then POST the returned `prompt` to
   ComfyUI's native `/prompt`, poll `/history/{id}`. This is the layer that actually
   proved the ComfyUI graph/type-linking works (custom `LTX2MLX_PIPELINE` type matching,
   node execution, `is_output_node` reporting) — none of that is exercised by layer 1.
3. **Real browser (Playwright) against the real server** — this is the layer that caught
   the two worst bugs, both invisible to layers 1-2:
   - `syncLtx2MlxPanelFromState()` called eagerly right after its own definition, before
     `const state = {...}` existed later in the same enclosing function — a JS temporal
     dead zone `ReferenceError` on *every single builder open*, engine-agnostic.
   - The engine-switch toast was hardcoded to only ever say "MiniMax H3" or "LTX" —
     selecting LTX-2 MLX announced "This project now uses the existing LTX scene
     renderer."

   `chromium-cli` wasn't available in this environment; used `npx playwright` instead
   (`npm install playwright` in a scratch dir, `playwright install chromium`). Key
   gotcha: the Music Video Builder isn't a page route — it's a modal opened by clicking
   a button widget on a `VRGDG_MusicVideoBuilderUI` graph node. To drive it headlessly:

   ```js
   const app = window.comfyAPI.app.app;   // NOT window.app -- that's undefined
   const node = window.LiteGraph.createNode("VRGDG_MusicVideoBuilderUI");
   app.graph.add(node);
   const button = node.widgets.find(w => w.type === "button" && w.name === "Open Music Video Builder");
   button.callback();
   ```

   Also: a pre-existing, unrelated crash (`Cannot read properties of null (reading
   'image_history')` in `segmentImageSource`) fires in this synthetic "brand new node,
   zero real segments" test scenario — confirmed via `git show <parent-commit>` that it
   happens on the *original* unmodified file too. Not a regression; the modal still
   renders fully despite it (the DOM is appended to `document.body` before the crash
   point in the init sequence).

## Verified working end-to-end

- T2V and A2V both queued through the real `/prompt` endpoint with zero `node_errors`,
  executed successfully, produced valid mp4s with correct dimensions/duration (A2V
  correctly matched a 1s test audio clip → 1.04s video, snapped to `8k+1` frames).
- Full UI flow: open builder → Menu → Settings → Project Video Engine → select
  "LTX-2 MLX (Apple Silicon)" → badge updates, correct panel renders with correct
  defaults, toast is correct.

## Bug found after this doc was first written: "no video path was found in history"

Reported by actually using the button this doc had flagged as untested. Two compounding
bugs in the shared history-parsing helpers, both now fixed:

1. `extractVideosFromHistory()` only checked `gifs`/`videos`/a legacy `animated`-as-
   file-array convention. LTX2MLX's nodes report outputs the way ComfyUI's *native*
   `SaveVideo` does — files under `images`, with a sibling `animated: [true]` flag — a
   shape the function didn't recognize at all.
2. Even once found, `resolveComfyVideoPath()` still couldn't turn `{filename, subfolder,
   type}` into a path — it only understands a `fullpath` param or a Windows-absolute
   `subfolder` (every other engine in this codebase feeds an absolute `filename_prefix`,
   so `subfolder` in history already *is* the absolute directory). LTX2MLX uses
   `folder_paths.get_save_image_path()`, ComfyUI's own relative-path convention, which is
   a genuinely different shape nothing else in this codebase produces.

Fix: added `_resolve_output_media_path()` / `/vrgdg/workflow_runner/resolve_output_path`
(Python, `folder_paths`-based, scoped to output/input/temp dirs) as a fallback used only
from `renderLtx2MlxSceneVideoWithProgress` when `resolveComfyVideoPath` returns empty —
other engines untouched. Re-verified against a live server: build → queue → history
entry correctly extracted → `resolve_output_path` returns the real, existing file path.

**Lesson**: layers 1-3 in the testing methodology above (direct calls, curl+history,
Playwright UI) each verified a *segment* of the pipeline in isolation but never the full
chain from click to collected file in one pass — which is exactly where this bug lived,
at the seam between "history contains an output" and "here is its path on disk." Wiring
a new engine into shared helpers written for a different output-path convention needs a
real click-through test, not just its component pieces individually.

## Known gaps / not done

- Only tested at low resolution/frame count with q4 + distilled/dev pipelines — real
  generation time/memory at production settings (q8, higher res, `two_stage`) unknown.
- `venv-3.13` has ComfyUI core + comfyui-ltx2-mlx + comfyui-vrgamedevgirl deps only, not
  the other ~50 custom nodes' deps — not a drop-in replacement for the main venv yet.
- `comfyui-ltx2-mlx` repo not pushed to GitHub (local commit only).
- Storyboard Builder has no LTX2MLX-specific reference/V2V UI (treated as plain `ltx`).
- `llama-cpp-python` still doesn't build in venv-3.13 (see Environment setup above).

---

## Krea-2 + Z-Image MLX Integration — Session Notes

Context dump from the session(s) that added FLUX.2 Klein, Krea-2, and Z-Image image
generation on Apple Silicon (via MLX / `mflux`) — first as a `flux2klein_mlx` engine
option in MusicVideoBuilder's per-scene image modes, then as two new engine cards in
the Reference Builder's "Create Subject"/"Create Location" flow (`ZImage MLX` and
`Krea2 + ZImage MLX Enhancer`). Same shape of problem as LTX-2: the existing
`flux_klein`/`zimage`/`krea2` engines assume a CUDA-capable ComfyUI backend;
[filipstrand/mflux](https://github.com/filipstrand/mflux) is a pure-MLX implementation
of all three model families, reusing the shared `venv-3.13` set up for LTX2MLX (no new
venv needed — `mflux` was `pip install`ed into it alongside `ltx-2-mlx`, with only a
`mlx`/`mlx-metal` downgrade from 0.32.0 to 0.31.2 to satisfy mflux's pin; LTX2MLX was
re-verified working afterward).

### Three new standalone node-pack repos

Same isolation pattern as `comfyui-ltx2-mlx`: each is its own `custom_nodes/` git repo
(now git-initialized with LICENSE/README/.gitignore and an initial commit), never
imported directly by `comfyui-vrgamedevgirl`, with a `_check_apple_silicon()` guard in
every loader's `execute()`.

- **`comfyui-flux2klein-mlx`** — `Flux2KleinModelLoader`/`Flux2KleinGenerate` (T2I/
  img2img) and `Flux2KleinEditModelLoader`/`Flux2KleinEdit` (multi-image conditioned
  editing, `image_paths` one-per-line or JSON list, matching
  `VRGDG_MultiReferenceConditioningFromPaths`'s format). See its own `SCOPING.md` and
  `README.md` for the full node/API details.
- **`comfyui-krea2-mlx`** — `Krea2ModelLoader`/`Krea2Generate` (T2I/img2img). Model name
  `"krea2"` resolves fine via `ModelConfig.from_name()` in either underscore or
  hyphenated form (unlike Flux2Klein/Z-Image, which require hyphens).
- **`comfyui-zimage-mlx`** — `ZImageModelLoader`/`ZImageGenerate` (T2I/img2img). Model
  names must be hyphenated (`"z-image"`, `"z-image-turbo"`) — `ModelConfig.from_name()`
  raises `Cannot infer base_model` on the underscore form, same gotcha as Flux2Klein.

**Real API findings, from `inspect.signature`/`inspect.getsource` against the actually
installed `mflux` 0.18.1, not README prose:**
- `Krea2`/`ZImage` both accept `negative_prompt` directly (unlike Flux2Klein, which
  hard-rejects it) — Krea-2 turbo CLI defaults are 8 steps, guidance 1.0, shift 1.15.
- `ZImage.generate_image()`'s return type annotation says plain `PIL.Image.Image`, but
  tracing through `ImageUtil.to_image()` confirms it actually returns the same
  `GeneratedImage` wrapper Krea2/Flux2Klein do (with a `.image` attribute) — the
  annotation is stale, not the real contract.
- `image_strength`'s inverted-from-usual-convention semantics (higher = closer to
  source, fewer denoise steps) is *presumed* to carry over from Flux2Klein's confirmed
  behavior for Krea2/ZImage too, but has not been independently re-verified — flagged
  in both node tooltips and READMEs as provisional.

### No combiner node needed for the Krea2 + ZImage "enhancer" chain

CUDA's Reference Builder wires "Krea2 + ZImage Enhancer" as a hidden two-pass workflow
(Krea2 T2I → ZImage img2img refine) inside one JSON template. The MLX equivalent needed
no new node type at all: `Krea2Generate`'s `IMAGE` output connects straight into
`ZImageGenerate`'s `image` input on the graph, since both packs speak plain ComfyUI
`IMAGE` tensors. Confirmed with a real chained generation (Krea2 seed=42 → ZImage
enhancer seed=7, `image_strength=0.3`): composition/pose/scene held across both passes,
enhancer pass visibly sharpened fur/snow texture without regenerating the scene.

### Testing methodology — same three layers as LTX2MLX, all three run for this work too

1. **Direct Python calls** (`Krea2(...).generate_image(...)`, `ZImage(...)`, and the
   chained pair) — real weight downloads (`krea/Krea-2-Turbo` is a **gated** HF repo;
   needed the user to visit the model page and request/accept access before
   `HF_TOKEN`-authenticated downloads succeeded — 401 before auth, 403-not-authorized
   after auth but before approval, 200 after approval), real generations, real images
   saved and visually inspected.
2. **Real ComfyUI server + curl** — `build_zimage_mlx_prompt`/`build_krea2_zimage_mlx_prompt`
   POSTed, then the returned `prompt` POSTed to `/prompt`, polled `/history/{id}`.
   Zero `node_errors`, `status_str: "success"`, correct files landed in `output/` via
   the same `ImageSaveHelper`/`folder_paths.get_save_image_path()` convention the stock
   `SaveImage` node uses — unlike LTX2MLX, this needed **no** history-parsing fallback
   fix, since it's the same output shape every other engine in this codebase already
   produces.
3. **Real browser (Playwright)** — same `window.comfyAPI.app.app` /
   `LiteGraph.createNode("VRGDG_MusicVideoBuilderUI")` / button-widget-callback pattern
   as LTX2MLX. Click path: open builder → dismiss the "Welcome to Video Creator"
   dialog (a *second* modal stacked on top of the builder on first open, not mentioned
   in the original LTX2MLX playwright notes — text-based Playwright locators kept
   matching ComfyUI's own top-level `close-workflow-button` instead of this dialog's
   `Close`, needed a direct DOM query/click instead) → Reference Builder → Flux/Nano
   Image References → Add Subject → Generate Subject. Confirmed: both new option cards
   render with all fields, both buttons enabled (capability check passed), zero
   `PAGE_ERRORS`. One tolerated, **pre-existing, already-documented** error reproduced
   again during builder open (`segmentImageSource`'s `image_history` crash on a
   brand-new node with zero real segments, from the original LTX2MLX playwright
   session) — confirmed still non-regression, modal still renders around it.

### Reference Builder integration points (`comfyui-vrgamedevgirl`)

- **`VRGDG_WorkflowRunnerNodes.py`**: `_build_zimage_mlx_api_prompt`/
  `_build_krea2_zimage_mlx_api_prompt`, `_require_zimage_mlx_available`/
  `_require_krea2_zimage_mlx_available` capability guards, two new API-format workflow
  JSON templates (`zimageMlx_API.json`, `krea2ZimageMlx_API.json`), four new routes
  (`build_zimage_mlx_prompt`, `build_krea2_zimage_mlx_prompt`, `zimage_mlx_capability`,
  `krea2_zimage_mlx_capability`) — exact mirror of the `flux2klein_mlx` route shape.
- **`VRGDG_MusicVideoBuilderUI.js`**: two new option cards — "ZImage MLX (Apple
  Silicon)" and "Krea2 + ZImage MLX Enhancer (Apple Silicon)" — added to all three
  Reference Builder dialogs (single subject/location generate, batch missing subjects,
  batch missing locations), each capability-gated (`applyMlxCapabilityGate`, disables
  the button and shows the server's reason if unavailable). New `workflow` values
  (`"zimage_mlx"`, `"krea2_zimage_mlx"`) threaded through the existing
  `createMissingSubjectReferencesWithImageWorkflow`/`createMissingLocationReferencesWithZImage`
  batch functions alongside `"zimage"`/`"krea2"`/`"flow_gpt"`. Cards expose real
  settings (quantize, width/height, steps, seed/seed mode, enhancer strength) via a new
  `buildMlxReferenceGeneratorControls()`, persisted the same way the CUDA cards persist
  their settings — models themselves are *not* configurable (fixed to `krea2`/
  `z-image-turbo`), since that's the whole point of the MLX path.
- LoRA slots are **not** exposed on either MLX Reference Builder card — a deliberate
  scope cut, not an oversight; CUDA's first/second-pass LoRA-strength UI wasn't ported.

### Known gaps / not done (Krea-2 + Z-Image MLX)

- LoRA support on the Reference Builder MLX cards (both node packs support
  `lora_paths`/`lora_scales` already; just not wired into these two cards' UI yet).
- `image_strength`'s direction not independently re-verified for Krea2/ZImage, only
  presumed to match Flux2Klein's confirmed-inverted behavior.
- No reference-image resize/downscale controls, same gap as Flux2Klein's.
- Krea-2's HF gating means first-run setup requires a manual "request access" step on
  huggingface.co before generation works — not automatable, should be called out
  wherever this engine is documented for end users.
- `comfyui-krea2-mlx`/`comfyui-zimage-mlx`/`comfyui-flux2klein-mlx` are git-initialized
  locally but not pushed to GitHub yet, same as `comfyui-ltx2-mlx` at the equivalent
  point in its own session.
