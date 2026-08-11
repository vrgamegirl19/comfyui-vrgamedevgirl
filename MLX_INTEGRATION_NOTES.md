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
3. **Gemma MLX chat/vision backend** (`LLM.py`, used by prompt/lyric/story writing and
   LoRA-dataset image captioning across the repo) — see
   [Gemma MLX Backend Integration](#gemma-mlx-backend-integration----session-notes)
   at the very end of this file.

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
- **No standalone Krea-2 MLX build path exists — confirmed real gap, not by design.**
  On the CUDA side there are genuinely two separate builders:
  `_build_krea2_api_prompt` (Krea-2 alone, loads `Krea2_TextToImage_API.json`) and
  `_build_krea2_2pass_api_prompt` (Krea-2 → Z-Image enhancer, loads
  `Krea2_API_2Pass.json`). The single-pass one is **actively used**, not dead code — the
  Reference Builder's "Use Krea2 + Enhancer For All Missing" buttons (JS
  `VRGDG_MusicVideoBuilderUI.js:26582`, `:26731`, `:26886`) all POST to
  `/vrgdg/workflow_runner/build_krea2_prompt`, i.e. the plain single-pass endpoint,
  despite the button label saying "+ Enhancer" (a pre-existing UI copy inaccuracy, not
  something this session touched). On the MLX side, only the enhancer-chain equivalent
  was ever wired up (`_build_krea2_zimage_mlx_api_prompt` /
  `krea2ZimageMlx_API.json`) — there is no `krea2Mlx_API.json`, no
  `_build_krea2_mlx_api_prompt`, and no `/vrgdg/workflow_runner/build_krea2_mlx_prompt`
  route. The underlying `comfyui-krea2-mlx` node pack already supports standalone use
  (`Krea2ModelLoader` + `Krea2Generate` alone was tested standalone earlier in this same
  session, see above) — it just was never surfaced as its own Reference Builder option.
  Deferred by explicit user request as of 2026-08-08; would be a small, contained
  addition mirroring the Z-Image MLX wiring (new API template + builder function +
  route + a JS card) if picked up later.

## Console logging added for every MLX engine (2026-08-08)

None of the MLX engines printed anything to the ComfyUI console before this — the only
confirmation a user had that MLX (vs. the CUDA equivalent, vs. a different MLX tier)
actually ran was client-side toast text, output `filename_prefix`, or node titles in
the graph. Added a `print(f"[VRGDG ...] Engine=... ...", flush=True)` line — matching
this file's existing `[VRGDG WorkflowRunner]`/`[VRGDG FLF]` convention — at the point
each engine's prompt is built/dispatched, so the actual engine, model, quantize level,
and seed used are now visible in the server log for every single generation:

- **LTX-2 MLX** — `_build_ltx2mlx_api_prompt()` (`VRGDG_WorkflowRunnerNodes.py`): logs
  `mode` (t2v/i2v/a2v), `model_dir`, `pipeline_type`, `seed`.
- **FLUX.2 Klein MLX** — `_build_flux2klein_mlx_api_prompt()`: logs `mode` (t2i/edit),
  `model_name`, `quantize`, `seed`.
- **Z-Image MLX** — `_build_zimage_mlx_api_prompt()`: logs `model_name`, `quantize`,
  `seed`.
- **Krea2 + Z-Image MLX Enhancer** — `_build_krea2_zimage_mlx_api_prompt()`: logs both
  passes' `model_name`/`quantize` plus `seed`.
- **Gemma MLX (text and vision)** — `LLM.py`'s `_load_gguf_model()`, the single
  dispatch point every internal call site and `generate_prompt()` funnel through: logs
  `Engine=Gemma MLX (Apple Silicon)` with `kind=text|vision`, the resolved
  `mlx-community/gemma-3-*` repo, and the original GGUF `model_path` it was mapped
  from — **or**, when no MLX mapping applies (or MLX isn't available), logs
  `Engine=GGUF (llama-cpp-python)` instead. Placed before either path's own model
  cache check, so it fires on every call, not just on first load/cache-miss — cached
  reuse is exactly as visible as a fresh load, deliberately kept consistent between the
  two branches (the GGUF branch's log line was initially placed after its cache check,
  which would have hidden cache hits; moved earlier to match the MLX branch's
  behavior).

## Publishing the 4 node packs to the ComfyUI Registry

All four standalone repos (`comfyui-ltx2-mlx`, `comfyui-flux2klein-mlx`,
`comfyui-krea2-mlx`, `comfyui-zimage-mlx`) got pushed to GitHub under `tanis2000` and
published to [registry.comfy.org](https://registry.comfy.org) in a later session, per
the [official publishing guide](https://docs.comfy.org/registry/publishing). All four
already had `pyproject.toml` with a `[tool.comfy]` block (`PublisherId`, `DisplayName`)
from when they were first created, so this was just wiring up CI + auth, not writing
registry metadata from scratch.

Steps taken, identical across all four repos:

1. Added `.github/workflows/publish.yml` to each repo — the standard
   `Comfy-Org/publish-node-action@main` workflow, triggered on `workflow_dispatch` or on
   push to `main` when `pyproject.toml` changes:

   ```yaml
   name: Publish to Comfy registry
   on:
     workflow_dispatch:
     push:
       branches:
         - main
       paths:
         - "pyproject.toml"

   permissions:
     issues: write

   jobs:
     publish-node:
       name: Publish Custom Node to registry
       runs-on: ubuntu-latest
       steps:
         - name: Check out code
           uses: actions/checkout@v4
         - name: Publish Custom Node
           uses: Comfy-Org/publish-node-action@main
           with:
             personal_access_token: ${{ secrets.REGISTRY_ACCESS_TOKEN }}
   ```

   Note the trigger path is `main`, not `master` — worth double-checking against each
   repo's actual default branch name if a future push doesn't fire the workflow.

2. Changed `PublisherId` in each `pyproject.toml` from the placeholder `vrgamedevgirl`
   to `tanis2000` (the real registry publisher these repos are published under).

3. User generated a `REGISTRY_ACCESS_TOKEN` from the ComfyUI Registry publisher
   dashboard and added it as a repo secret (Settings → Secrets and variables → Actions)
   on all four GitHub repos — this step has to happen in the GitHub UI, no CLI path was
   used.

4. Committed (`pyproject.toml` + `.github/workflows/publish.yml`) and pushed `master` to
   `origin` on all four repos — remotes were already configured
   (`git@github.com:tanis2000/<repo>.git`), so this was the first real push for all of
   them.

5. Verified via each repo's GitHub Actions tab (`gh` CLI wasn't installed in this
   environment, so this had to be checked manually in the browser) — all four
   `publish-node` runs succeeded.

6. Confirmed all four packages showed up immediately on
   [registry.comfy.org](https://registry.comfy.org) (publisher page:
   `registry.comfy.org/publishers/tanis2000`) right after the workflow run completed —
   registry publish is synchronous with the Action, not a delayed/batched process.

**Gotcha avoided**: the workflow's push-trigger path filter is `pyproject.toml` only.
Any future metadata bump (new `version`, description, etc.) needs to touch that file
specifically to auto-publish on push — editing other files and pushing to `main` won't
trigger a new registry version. `workflow_dispatch` is always available as a manual
fallback regardless of what changed.

---

## Gemma MLX Backend Integration — Session Notes

Every "write a prompt/lyric/story/JSON extraction" call and every LoRA-dataset image
caption in this repo goes through Gemma via `LLM.py`'s `VRGDG_GeneralGGUF`/
`VRGDG_SuperGemmaGGUFChat` (llama-cpp-python, GGUF) or `VRGDG_GeneralVLM` (HF
`transformers`, no `mps` device support). On this machine `llama-cpp-python` is the
dependency documented above as failing to build in `venv-3.13` (the OpenSSL/
`cpp-httplib` linker error) — so on Apple Silicon, Gemma had no working local backend
at all until this session, short of very slow CPU `transformers`.

### Why an MLX backend was scoped differently from LTX2MLX/Flux2Klein/Krea2/Z-Image

Those four engines are each a single diffusion pipeline invoked from exactly one place
(a ComfyUI graph node). Gemma is the opposite: investigation found **~15+ internal call
sites** across `VRGDG_MusicVideoBuilderNodes.py`, `VRGDG_LoraDatasetCreatorNodes.py`,
`VRGDG_MusicVideoPromptCreatorNodes.py`, `VRGDG_VideoEditorNodes.py`,
`VRGDG_GeneralNodes2.py`, and `LTXLoraTrain.py` that instantiate
`VRGDG_SuperGemmaGGUFChat()` directly in Python and call its low-level helpers
(`_load_gguf_model`, `_run_gguf_text_pipeline`, `_run_gguf_vision_pipeline`) — bypassing
the ComfyUI graph and the node's `generate_prompt()` entrypoint entirely. A sibling
node-pack repo (the LTX2MLX pattern) would have needed every one of those ~15 call sites
edited individually to route to it.

Instead, the MLX backend was built **in place inside `LLM.py`**, dispatching from
inside those same three low-level methods based on Apple-Silicon + `mlx-lm`/`mlx-vlm`
availability. Method signatures and return types (`str`) are unchanged, so **every**
call site — the ~15 direct callers and the `generate_prompt()` node entrypoint alike —
gets MLX for free with zero edits outside `LLM.py`. No new node pack, no new UI.

### `mlx-vlm` install and the `mlx` version pin

`mlx-lm` (0.31.3) was already installed in `venv-3.13`, pulled in transitively by
`ltx-core-mlx` (LTX2MLX's own Gemma-3 text-encoder dependency), and already working
against the `mlx==0.31.2`/`mlx-metal==0.31.2` pin mflux needs.

`mlx-vlm` was not installed. `pip install mlx-vlm` pulled `mlx-vlm==0.6.10`, which
bumped `mlx`/`mlx-metal` to `0.32.0` — breaking mflux's `mlx<0.32.0` pin, exactly the
conflict flagged as a risk before starting. Downgrading back with
`pip install "mlx==0.31.2" "mlx-metal==0.31.2"` left pip reporting a *declared*
incompatibility (`mlx-vlm 0.6.10 requires mlx>=0.32.0, but you have mlx 0.31.2`), but
this turned out to be an overly strict pin, not a real runtime requirement — confirmed
by direct testing, not by trusting the pip warning:

- `mlx_vlm` imports cleanly and `mlx_vlm.load`/`mlx_vlm.generate` run successfully
  against `mlx==0.31.2`.
- `mflux` (`Krea2`), `ltx_core_mlx`, and `mlx_lm` all still import cleanly together
  with `mlx-vlm` installed and `mlx` pinned back to `0.31.2`.
- A real Krea2/LTX2MLX generation smoke-test wasn't re-run end-to-end (basic
  tensor-op + real-package-import checks were judged sufficient here, unlike the
  original mlx-metal 0.32.0→0.31.2 downgrade session which did re-verify with full
  generations) — worth a full regression pass before relying on this in production if
  mflux/LTX2MLX behavior looks off after this change.

**Net effect**: `mlx`/`mlx-metal` stayed pinned at `0.31.2` in `venv-3.13`; `mlx-vlm`
runs fine despite its own stricter declared requirement. Neither `mlx-lm` nor
`mlx-vlm` were added to this repo's root `requirements.txt` (it has no platform
markers and would apply the install to every OS/CUDA target) — they're documented here
instead, same as the other three sibling MLX node-pack repos' dependencies.

### Design in `LLM.py`

- New module-level caches `_MLX_TEXT_MODEL_CACHE`/`_MLX_VISION_MODEL_CACHE`
  (single-slot, same evict-on-load-new policy as every other MLX engine in this repo),
  cleared by `_clear_vrgdg_llm_caches()` alongside the existing GGUF caches.
- `_gemma_mlx_available(vision=False)` — Apple Silicon (`sys.platform == "darwin" and
  platform.machine() == "arm64"`) plus `importlib.util.find_spec("mlx_lm")`
  (and `mlx_vlm` too when `vision=True`). No sibling-node-pack "required classes"
  check like `_require_ltx2mlx_available()` uses, since there's no node pack here —
  just direct library imports.
- `_GEMMA_MLX_MODEL_MAP` + `_GEMMA_MLX_SIZE_HINTS` + `_resolve_gemma_mlx_repo()` —
  maps known GGUF preset repo ids (`unsloth/gemma-4-26B-A4B-it-GGUF`, `Jiunsong/
  supergemma4-26b-uncensored-gguf-v2`) and size-tier substrings in a GGUF filename
  (`27b`/`26b`→`gemma-3-27b-it-4bit`, `12b`→`gemma-3-12b-it-4bit`, `4b`→
  `gemma-3-4b-it-4bit`, `1b`→`gemma-3-1b-it-4bit`, default `gemma-3-12b-it-4bit`) to an
  `mlx-community/gemma-3-*-it-4bit` repo id. Matched against the GGUF `model_path`'s
  basename alone (not a separately threaded `model_id`), since that's all the low-level
  helpers ever receive, and `VRGDG_SuperGemmaGGUFChat`'s local-file dropdown already
  filters filenames for `"gemma"` — confirmed via `_list_local_gemma_gguf_choices()` —
  so real call sites reliably have "gemma" in the basename already.
- **Caveat, deliberately not papered over**: MLX cannot load an arbitrary local
  `.gguf` file — different format entirely from `mlx-lm`/`mlx-vlm`'s safetensors-based
  MLX format. When `_resolve_gemma_mlx_repo()` returns `None` (no `"gemma"` in the
  path, or a model with no known MLX equivalent), `_load_gguf_model` falls straight
  through to the existing `llama_cpp.Llama(...)` path unchanged — so a genuinely custom
  local GGUF still needs `llama-cpp-python` working, same as before this change.
- `_load_mlx_text_model(repo)` / `_load_mlx_vision_model(repo)` — lazy `import mlx_lm`
  / `import mlx_vlm`, wrapped in the exact same try/except-with-clear-message idiom
  already used for `llama_cpp` (`LLM.py`'s existing pattern), so machines without these
  packages installed get a clear error instead of an ImportError deep in a stack trace.
  Return a tagged `_MLXTextHandle`/`_MLXVisionHandle` (plain attribute holder,
  `model`/`tokenizer`/`repo` or `model`/`processor`/`config`/`repo`) rather than the
  library's raw model object, so `_run_gguf_text_pipeline`/`_run_gguf_vision_pipeline`
  can `isinstance()`-check and dispatch without changing their own signatures.
- `_run_mlx_text_pipeline()` — `tokenizer.apply_chat_template(messages,
  add_generation_prompt=True)` → `mlx_lm.generate(model, tokenizer, prompt=...,
  max_tokens=..., sampler=make_sampler(temp=..., top_p=...))`. **Real bug caught by
  actual end-to-end testing, not just import checks**: `mlx_lm.generate()` doesn't stop
  at Gemma's `<end_of_turn>` token by default — first response came back correct,
  followed by the model endlessly repeating garbage tokens for the rest of
  `max_new_tokens`. Fixed by calling `tokenizer.add_eos_token(tok)` for every entry in
  `VRGDG_GeneralGGUF._GEMMA_STOP_SEQUENCES` right after `mlx_lm.load()` — the same stop
  sequences already used to bound the GGUF/llama.cpp path, now reused for MLX too.
- `_run_mlx_vision_pipeline()` — saves incoming PIL images to temp PNG files (mflux's
  sibling packs use the identical `_tensor_to_image_path`-style pattern for image
  conditioning), builds the prompt via `mlx_vlm.prompt_utils.apply_chat_template(processor,
  config, instruction_text, num_images=len(image_paths))`, then
  `mlx_vlm.generate(model, processor, prompt=..., image=image_paths, max_tokens=...,
  temperature=..., top_p=...)`, reading `.text` off the returned `GenerationResult`.
  Temp files cleaned up in a `finally`.
- Dispatch itself lives at the top of `_load_gguf_model()` (branches to
  `_load_mlx_text_model`/`_load_mlx_vision_model` when available+mapped, else falls
  through unchanged) and at the top of `_run_gguf_text_pipeline()`/
  `_run_gguf_vision_pipeline()` (`isinstance(model, _MLXTextHandle/_MLXVisionHandle)` →
  MLX path, else unchanged GGUF path). `_unload_gguf_model()` was left untouched — it
  only pops from the GGUF cache dict by GGUF cache key, which MLX handles never enter,
  so it's a harmless no-op for MLX-backed sessions; MLX models are only evicted by the
  single-slot cache loading a different model, or by `_clear_vrgdg_llm_caches()`.

### Real-API findings from `inspect` against the actually installed packages

- `mlx_lm.load(repo) -> (model, TokenizerWrapper)`; `mlx_lm.generate(model, tokenizer,
  prompt, max_tokens=, sampler=)`; sampler built via
  `mlx_lm.sample_utils.make_sampler(temp=, top_p=)`.
- `TokenizerWrapper.add_eos_token(token_str_or_id)` is the real (undocumented-in-README)
  way to add extra stop tokens beyond the tokenizer's default `eos_token_id` — needed
  for the `<end_of_turn>` fix above.
- `mlx_vlm.load(repo) -> (model, processor)`; config for chat-templating comes from
  `mlx_vlm.utils.load_config(repo)`, a separate call, not part of `load()`'s return.
  `mlx_vlm.generate()` returns a `GenerationResult` dataclass-like object with a `.text`
  attribute (plus token counts, timing, etc.) — not a plain string like `mlx_lm.generate()`.

### Testing performed (adapted from the repo's 3-layer methodology — no new UI, so no
Playwright layer was needed)

1. **Direct Python calls**: `_gemma_mlx_available()`/`_resolve_gemma_mlx_repo()` unit
   checks (mapping correctness, non-Gemma paths correctly return `None`); a real
   `_load_mlx_text_model`/`_run_mlx_text_pipeline` call against
   `mlx-community/gemma-3-1b-it-4bit` (smallest tier, chosen to conserve the ~35-46GB
   free disk on this machine) — caught and fixed the `<end_of_turn>` stopping bug above.
2. **Through the real call-site pattern**: instantiated `VRGDG_SuperGemmaGGUFChat()`
   and called `_load_gguf_model()` → `_run_gguf_text_pipeline()` exactly as the ~15
   internal call sites do, with a fake path containing `"gemma-4-26B-A4B"` (matching
   the real default preset's naming) — confirmed it dispatched to `_MLXTextHandle` and
   produced a clean response. Repeated for vision: `_load_gguf_model()` with a non-empty
   `mmproj_path` + a real solid-orange test image through
   `_run_gguf_vision_pipeline()`, against `mlx-community/gemma-3-4b-it-4bit` — dispatched
   to `_MLXVisionHandle`, correctly described the image's color.
3. **mflux/LTX2MLX regression check**: re-imported `mflux`'s `Krea2` and `ltx_core_mlx`
   after the `mlx-vlm` install + `mlx` re-pin — both clean. A full real-generation
   regression (not just imports) was **not** re-run this session — flagged as a gap.
4. **Real ComfyUI server / UI click-through** — performed after this doc was first
   written, by the user manually exercising the real Music Video Builder UI (not
   Playwright). Confirmed working:
   - **Text, 27B tier**: real prompt/lyric/story-type generation through the actual
     builder UI ran successfully against the `mlx-community/gemma-3-27b-it-4bit`
     mapping (the default tier for the `gemma-4-26B-A4B`-style preset naming).
   - **Vision**: confirmed working too. The user wasn't initially sure which builder
     feature actually exercises the vision path (it's gated behind `use_vision` +
     an attached reference image, not always-on) — traced it to the
     `use_vision`/`has_ref_image` gate in `_generate_builder_t2i_prompt()` (and the
     same pattern in `_generate_builder_i2v_prompt`/`_generate_builder_chained_i2v_prompt`/
     `_generate_builder_t2v_prompt`/`_edit_builder_video_prompt`), plus the always-vision
     `/vrgdg/music_builder/describe_reference_image` route and
     `VRGDG_LoraDatasetCreatorNodes.py`'s captioning step. The `[VRGDG LLM] Engine=Gemma
     MLX ... kind=vision` console log line added earlier this session was what let the
     user confirm which backend actually ran, resolving the ambiguity.

### Known gaps / not done

- No full mflux/LTX2MLX real-generation regression re-run after the `mlx-vlm` install
  (only import-level + basic tensor-op checks were done).
- The 12B default text tier and other Gemma-3 sizes below 27B haven't been separately
  confirmed by the user (27B and 4B/vision were the ones actually exercised) — output
  quality at those tiers, at real generation lengths/settings used by this repo's
  prompt/lyric/story tasks, is unverified.
- `_GEMMA_MLX_MODEL_MAP`/`_GEMMA_MLX_SIZE_HINTS` is a best-effort mapping, not an
  exhaustive one — a custom local GGUF with unusual naming (no size-tier substring, no
  `"gemma"` in the filename at all) silently falls back to the GGUF/llama-cpp path
  rather than MLX, which on this machine means it'll hit the still-unresolved
  `llama-cpp-python` build failure. No user-facing override/warning surfaces this today.
- No explicit user-facing override to force GGUF over MLX (or vice versa) when both are
  technically available — dispatch is fully automatic today.

---

## Follow-up session (2026-08-09): engine visibility, memory diagnosis, pre-quantized downloads

### FLUX.2 Klein MLX "Gemma Klein MLX Prompt" button fix

Reported as "does nothing" — turned out unrelated to Gemma/MLX entirely. Both the CUDA
"Gemma Flux Prompt" button and the MLX one already called the same endpoint
(`/vrgdg/music_builder/generate_flux_klein_prompt`) and correctly updated the segment's
data model (`segment.flux_prompt`). The bug: `syncSegmentT2IPrompt()` (the function both
buttons funnel through to push the result into visible textareas) was missing the line
that writes into the FLUX.2 Klein MLX card's textarea (`flux2KleinMlxPrompt`) — its
sibling function `syncSegmentFlowGptPrompt()` already had it. So the prompt was saved
but the box stayed stale until the user navigated away from the scene and back (full
re-sync). Fixed by adding `flux2KleinMlxPrompt.value = cleanPrompt;` to
`syncSegmentT2IPrompt()` (`web/VRGDG_MusicVideoBuilderUI.js`).

### `[VRGDG WorkflowRunner]`/`[VRGDG LLM]` console logging for every MLX engine

None of the MLX engines printed anything to the ComfyUI console before this — the only
confirmation a user had that MLX (vs. CUDA, vs. a different MLX tier) actually ran was
client-side toast text, output `filename_prefix`, or node titles in the graph. Added
`print(f"[VRGDG ...] Engine=... ...", flush=True)` lines at the point each engine's
prompt is built/dispatched:

- **LTX-2 MLX** — `_build_ltx2mlx_api_prompt()`: `mode`, `model_dir`, `pipeline_type`, `seed`.
- **FLUX.2 Klein MLX** — `_build_flux2klein_mlx_api_prompt()`: `mode`, `model_name`, `quantize`, `seed`.
- **Z-Image MLX** — `_build_zimage_mlx_api_prompt()`: `model_name`, `quantize`, `seed`.
- **Krea2 + Z-Image MLX Enhancer** — `_build_krea2_zimage_mlx_api_prompt()`: both passes' `model_name`/`quantize`, `seed`.
- **Gemma MLX (text and vision)** — `LLM.py`'s `_load_gguf_model()`, the single dispatch
  point every internal call site and `generate_prompt()` funnel through: logs
  `Engine=Gemma MLX (Apple Silicon)` with `kind=text|vision`, the resolved
  `mlx-community/gemma-3-*` repo, and the original GGUF `model_path`, or
  `Engine=GGUF (llama-cpp-python)` when no MLX mapping applies. Placed before either
  path's own cache check so cache hits log too, not just fresh loads.

User confirmed via this logging: the 27B Gemma-3 MLX text tier and vision (image
captioning) both work end-to-end through the real Music Video Builder UI. Vision
specifically fires from `_generate_builder_t2i_prompt()`/`_generate_builder_i2v_prompt()`/
etc. (gated behind a `use_vision` toggle + an attached reference image, not always-on),
the always-vision `/vrgdg/music_builder/describe_reference_image` route, and
`VRGDG_LoraDatasetCreatorNodes.py`'s captioning step.

### LTX2MLX 8h+ render investigation: memory pressure, not (only) inherent slowness

User reported LTX-2.3 MLX at q8, 1920x1080, 8 seconds, exceeding the already-generous 8h
render timeout. Diagnosis: `vm.swapusage` showed 24.4GB/25.6GB swap used (only ~1.2GB
free) — heavy swapping on Apple Silicon severely degrades unified-memory-bound
workloads. After the user closed some apps, physical free RAM improved (~4.7GB → ~7.8GB)
and `memory_pressure`'s own verdict moved to "System-wide memory free percentage: 62%"
(healthy) — but the swap "used" figure barely moved, since macOS doesn't proactively
reclaim swap once written; it's a stale signature of the render's *actual* peak demand,
not a live indicator once pressure eases. No single foreground app was found hogging
tens of GB — top RSS consumers were all under ~1GB. Most likely explanation: some
combination of (a) 1080p/8s/q8/`two_stage` genuinely being a very large, previously
untested workload for a 22B model on Apple Silicon's unified memory, and (b) that load
itself triggering swap thrashing that made it much worse than the "just slow" baseline.

Available levers on `LTX2MLXModelLoader`/`LTX2MLXAudioModelLoader` (not yet tested
against each other head-to-head): `low_ram` toggle (→ `low_ram_streaming=True`, streams
weights from disk instead of holding q8's ~21GB resident at once — trades throughput for
much lower peak memory, worth trying first given the swap evidence); `model_dir` quant
tier (`dgrauet/ltx-2.3-mlx-q4` vs. the current `-q8`, roughly halves resident weight
size); `pipeline_type` (`two_stage`/`two_stage_hq` run generation twice vs. `one_stage`;
`distilled` is fastest/cheapest via fewer steps, some quality tradeoff).

### Pre-quantized HuggingFace downloads for FLUX.2 Klein / Krea2 / Z-Image MLX

Discovered while debugging why selecting "FLUX.2 Klein MLX, 9B, 8-bit" still triggered a
`Downloading model from HuggingFace: black-forest-labs/FLUX.2-klein-9B...` log line for
a ~32GB+ bf16 checkpoint. Root cause (not a bug, confirmed via real `mflux` source):
`mflux` has no separate pre-quantized repo concept baked into `ModelConfig.from_name()`
— it always resolves a bare `model_name` like `flux2-klein-9b` to the original
full-precision HF repo, downloads that, and quantizes locally on first load, regardless
of the requested `quantize` level. The `quantize` savings only apply to compute/memory
after that point, not to the download itself.

However, all three sibling loader nodes already had an existing, previously
under-utilized `custom_model_path` field ("Overrides model_name if set — local path or
HF repo id"). Traced through `Flux2Initializer.init()` → `WeightLoader.load()` →
`WeightApplier.apply_and_quantize_single()`: if `custom_model_path` points at a repo
whose weights already carry `quantization_level` metadata (i.e. a genuine pre-quantized
"mflux save format" export), `QuantizationResolution.resolve(stored=stored_q,
requested=quantize_arg)` uses the stored quantization instead of re-quantizing — safe
and correct, confirmed by reading `mflux`'s actual resolution rule table
(`quantization_resolution.py`), not assumed.

**Verified real repos** (via `HfApi.model_info(..., files_metadata=True)` — actual
existence, non-gated status, and real file sizes checked, not guessed from naming
conventions) and wired into each pack's `nodes/loaders.py` as an automatic
`_PREQUANTIZED_REPO_MAP` lookup (used only when `custom_model_path` is left blank):

| Engine | Model + quantize | Repo | Size | vs. bf16 original |
|---|---|---|---|---|
| FLUX.2 Klein | `flux2-klein-4b` @ 4/8-bit | `mlx-community/flux2-klein-4b-4bit`/`-8bit` | 4.6GB / 8.6GB | ~15GB+ |
| FLUX.2 Klein | `flux2-klein-9b` @ 4/8-bit | `mlx-community/flux2-klein-9b-4bit`/`-8bit` | 9.5GB / 17.9GB | ~32GB+ |
| Krea2 | `krea2` @ 4/8-bit | `MLXBits/krea-2-mlx-q4`/`-q8` | ~7GB / 22.2GB | full `krea/Krea-2-Turbo` bf16 (also gated) |
| Z-Image | `z-image-turbo` @ 8-bit | `deepsweet/Z-Image-Turbo-6B-MLX-Q8` | 11GB | `Tongyi-MAI/Z-Image-Turbo` bf16 |

Every repo above was confirmed compatible the same way: its own `model.safetensors.index.json`
`metadata` field carries `quantization_level`/`mflux_version`, and its README explicitly
states `library_name: mflux` with an `mflux-generate-*` usage example — i.e. it was
actually exported by `mflux` itself, not just similarly named.

**Rejected**: `SceneWorks/krea-2-turbo-mlx` (user-suggested) — real repo, real MLX
weights, but its `q8/transformer/config.json` shows `_class_name:
"Krea2Transformer2DModel"` and `q8/model_index.json` declares `diffusers`/`transformers`
library components (`Krea2Pipeline`) — a `diffusers`-native pipeline export, a
fundamentally different weight-key schema than what this pack's `mflux`-based `Krea2`
loader expects. Not wired in; would very likely throw a key/shape-mismatch error or
silently produce broken output if pointed at directly.

Also added: a `_log_resolved_source()` print in each of the three packs' loaders
(`[comfyui-flux2klein-mlx]`/`[comfyui-krea2-mlx]`/`[comfyui-zimage-mlx]` prefix),
logging whether the pre-quantized repo, a `custom_model_path` override, or the original
bf16-then-quantize-locally path was actually used — visible in the ComfyUI console for
every single model load.

**Not done**: `flux2-klein-9b-kv` (no pre-quantized mirror found), plain `z-image`
(non-turbo), and Z-Image's 4-bit tier all still fall through to the original
download-and-quantize-locally behavior — no verified repo exists for these yet.

### Frontend: pre-quantized vs. full-precision status indicator

Added a live status note under every model/quantize dropdown pair across the three
engines above, so the choice's cost is visible before clicking generate — not just
knowable after the fact from the console log:

- FLUX.2 Klein MLX scene-image card: updates on both model and quantize changes.
- Z-Image MLX (standalone card, and the enhancer second pass inside the "Krea2 + Z-Image
  MLX Enhancer" combo card): updates on quantize (model is a fixed constant per card).
- Krea2 MLX (first pass of the enhancer combo card): same.

Implementation: a client-side `MLX_PREQUANTIZED_REPO_MAP` in
`web/VRGDG_MusicVideoBuilderUI.js` mirrors the Python-side maps above (with an explicit
comment noting they must be kept in sync — this is display-only, the real routing
decision is always made server-side, so any drift would only produce a misleading label,
never wrong behavior). `describeMlxQuantStatus()`/`makeMlxQuantStatusNote()`/
`updateMlxQuantStatusNote()` render either "⚡ Pre-quantized build available: downloads
`<repo> (<size>)` directly" (green) or "⏳ No pre-quantized build known ... downloads the
full-size original and quantizes it locally (slow, large one-time download)" / "⏳ Full
precision: downloads the full-size original checkpoint..." (yellow) depending on the
current selection. Wired into `buildMlxReferenceGeneratorControls()` (shared by all
three Reference Builder modal call sites) and the FLUX.2 Klein MLX scene-image panel's
existing `change` listeners.

---

## Follow-up session (2026-08-09/10): "Gemma returned repeated/thought junk" investigation

Reported symptom: generating a video prompt from the Edit Scene Card's video prep flow
sometimes threw `"Gemma returned repeated/thought junk instead of a usable prompt. Try
again or shorten the notes."` This took several rounds to pin down because two real,
independent bugs were fixed along the way before finding the actual root cause — worth
recording all three so the wrong ones aren't "found" again.

### Fix 1 (real, but not the reported bug): MLX text/vision generation had no stop-sequence enforcement

`_run_mlx_text_pipeline`/`_run_mlx_vision_pipeline` (`LLM.py`) relied only on
`tokenizer.add_eos_token()` at model-load time (see the original Gemma MLX Backend
Integration section above), silently swallowed via a bare `try/except: pass`. Several of
`_GEMMA_STOP_SEQUENCES`'s entries (`_end_turn`, `|end_of_turn|`, etc.) aren't real single
vocab tokens for this tokenizer, so registration for those silently failed. Unlike the
GGUF/llama-cpp path, which passes a `stop=` list checked against generated *text* on every
call, MLX had no per-call text-based stop check at all — so a generation that drifted past
`<end_of_turn>` could run to `max_tokens` producing repeated channel/thought text.

Fixed by adding `_truncate_at_stop_sequence()` (mirrors the GGUF path's substring
matching) plus a repetition-penalty logits processor
(`make_repetition_penalty`/`repetition_penalty` kwarg) to both MLX pipelines, with the
penalty's lookback window scaled to `max_new_tokens` (`max(64, min(max_new_tokens, 512))`)
since video-prep calls request up to 4000 tokens vs. ~1000-1200 for image-prep — a fixed
64-token window left long generations unprotected against loops recurring outside it.

Verified via direct `venv-3.13` calls: clean, correctly-terminated output at both short
and full 4000-token budgets, including through the real `VRGDG_SuperGemmaGGUFChat`
dispatch path. **This was a real gap and is a legitimate improvement, but it turned out
not to be what the user was hitting** — real MLX generations were coming back clean the
whole time.

### Fix 2 (real, but also not the reported bug): frontend junk-detector's blanket bracket rule

`looksLikeGeneratedPromptJunk()` (`web/VRGDG_MusicVideoBuilderUI.js`) had
`if (bracketed.length >= 2) return true;` — flagging *any* text with two or more
bracket-enclosed spans regardless of content. The I2V prompt-enhancement instruction
template (`VRGDG_MusicVideoBuilderNodes.py`'s `_video_prompt_enhancement_instructions`)
is itself built from bracket-placeholder examples (`[subject]`, `[camera motion]`, etc.),
so a legitimately-cleaned response containing even two unrelated bracketed asides (e.g. a
shot annotation) could get rejected client-side after the backend's own placeholder-aware
validator had already passed it. Removed the blanket rule, keeping only the targeted check
for brackets that actually contain placeholder words
(`subject|setting|environment|camera|motion|weather|lighting|dynamic|framing`). Verified
with a small harness: legitimate bracket use no longer trips it, genuine unfilled-template
leaks still do. **Also a real fix, also not the actual cause of the user's failure.**

### Actual root cause: `cleanNaturalSubjectLabel()`'s colon-split regex

Neither fix above reproduced the failure — 10+ live `venv-3.13` generations against the
user's exact scene notes (character/location/T2I text, both text-only and real-image
vision paths, the enhancement pass, RTV mode) all came back clean. The backend's own
`/vrgdg/music_builder/generate_i2v` request/response, captured from the user's real
browser session via DevTools, also came back clean and passed the (already-patched) junk
detector when tested in isolation.

Added a `console.error` at the actual throw site in `applyTriggerPhrase()` so the next
failure would print the exact text being rejected regardless of whether the existing
`gemma_debug` file-save path fired (it wasn't producing a file for this user — likely
because `project_folder` wasn't populated at the moment the error handler ran in this
particular call path). That surfaced the real text: the user's full multi-sentence
character bio (`"the singer: The singer has long, dark, wavy hair..."`) appearing
**twice** in the final prompt, back-to-back with unrelated sentences between.

Traced to `cleanNaturalSubjectLabel()` (`web/VRGDG_MusicVideoBuilderUI.js:12270`), which
tries to shorten a `"label: description"` string down to just the label by splitting on
`/\s+[:|]\s+/` — requiring whitespace on **both sides** of the colon/pipe. Standard
English punctuation writes `"the singer:"` with no space before the colon, so the split
never matched, `beforeDetail` fell back to the *entire original paragraph*, and since that
paragraph starts with "the singer" it passed the `/^(?:the\s+)?(?:singer|performer|...)/i`
check and was returned unchanged — the whole bio, not a short label.

That mislabeled "performer" then got used twice by different downstream call sites:
once prepended as the vocal directive in `applyVocalDirectiveToVideoPrompt()`
(`"<performer> visibly sings "..." in sync with the audio."`), and again substituted into
Gemma's own generated paragraph via `replaceGenericSubjectLabels()`. The result was a
genuinely duplicated character bio in the final text — the junk detector was correctly
catching a real defect, just not a Gemma/MLX generation problem at all.

**Fix**: changed the split regex to `/\s*[:|]\s+/` (space before the delimiter now
optional, space after still required) at `web/VRGDG_MusicVideoBuilderUI.js:12274`.
Verified: `"the singer: <full bio>"` now correctly resolves to `"the singer"`; the
previously-working spaced (`"the singer : ..."`) and generic-placeholder
(`"Subject 1: ..."`) cases are unaffected. Confirmed working end-to-end by the user
against the real builder after this fix (Fixes 1 and 2 above did not resolve it on their
own; this one did).

**Lesson**: when a Gemma/MLX-labeled error message shows up, don't assume the bug is in
generation — several call sites downstream mutate the model's raw output client-side
(vocal directive injection, generic-subject-label replacement, trigger-phrase mapping)
before it reaches the user, and a bug in any of those can produce text that legitimately
looks like model repetition. The `console.error` added to `applyTriggerPhrase()`'s throw
site is a permanent diagnostic now — future junk-detector failures will log the exact
rejected text to the browser console even if the file-based `gemma_debug` save path
doesn't fire.

### TODO (not yet done): LTX2MLX A2V trims video slightly shorter than its audio slice — scene splitting needs to account for this

Found while fixing an unrelated crash in the sibling `comfyui-ltx2-mlx` repo (see its own
commit history: `4528ec3`, superseded by `e71109b`). `LTX2MLXAudioToVideo`'s
`match_audio_length` used to round the derived frame count to the *nearest* valid `8k+1`
value and pad the audio with trailing silence if that rounded up past the clip's real
duration — which fixed a RoPE broadcast-shape crash
(`[broadcast_shapes] Shapes (1,32,100,32) and (1,32,101,32) cannot be broadcast`) but added
synthetic silence to the song. Revised to instead round the frame count *down* to the
largest `8k+1` value that fits within the audio's actual duration
(`_snap_frame_count_leq` in `comfyui-ltx2-mlx/nodes/audio.py`) — no padding, but the
rendered video can now be up to just under one frame-snap step
(8 frames / frame_rate, ~333ms at 24fps) **shorter** than the audio slice it was given.

**Why this matters here**: this repo's scene-splitting logic (`_trim_scene_audio_clip` in
`VRGDG_WorkflowRunnerNodes.py:3156`, and whatever computes each scene's `start_seconds`/
`duration_seconds` slice of the full song before calling it) currently assumes each
scene's rendered video will cover its audio slice exactly, back-to-back with no gap, so
consecutive scenes concatenate seamlessly into the full song. With the LTX2MLX A2V engine
specifically, that assumption is no longer exactly true — a scene's video can now end
up to ~333ms *before* its audio slice ends, which would show up as a small silent/frozen
gap at each scene boundary in the final concatenated music video (LTX2MLX A2V only; other
video engines — LTX, MiniMax H3 — aren't affected, since they don't derive frame count
from `match_audio_length`'s 8k+1 snap-down).

**Not yet investigated/fixed** — needs a follow-up session to look at:
- Whether scene concatenation is even audio-driven end-to-end for LTX2MLX A2V today, or
  whether each scene's own generated audio (LTX2MLX A2V outputs audio too, not just
  video — it's driven by a vocoder/audio decoder per the pipeline load log) is what
  actually gets stitched, in which case the video/audio *within* a single scene's output
  are already in sync and the only question is whether the next scene's clip starts
  immediately after or leaves a gap.
- Whether the fix belongs in `_trim_scene_audio_clip`/scene-boundary computation (e.g.
  snap each scene's duration down to the same `8k+1`-at-24fps boundary *before* slicing,
  so consecutive slices already tile with no gap) or in how the render pipeline
  reassembles per-scene clips into the final timeline (e.g. compensate the next scene's
  start time by the previous scene's actual trimmed length rather than its requested
  audio-slice length).
- Whether this is specific to the current default `frame_rate=24` (8-frame step ≈ 333ms)
  or should be computed dynamically per the project's actual frame rate setting.

---

## Follow-up session (2026-08-11): example workflows for all 4 MLX node packs, Krea2 mflux bug fix

### Example workflows added to all four standalone node-pack repos

Each of `comfyui-zimage-mlx`, `comfyui-flux2klein-mlx`, `comfyui-ltx2-mlx`, and
`comfyui-krea2-mlx` now ships an `examples/` folder: a loadable ComfyUI workflow
graph (Loader → Generate [→ Preview]), a real output produced by actually running
it against a live ComfyUI server (`/prompt` → `/history`, zero `node_errors`), and
a Playwright screenshot of the loaded graph — all committed to each repo's own
GitHub remote (`tanis2000/comfyui-{zimage,flux2klein,ltx2,krea2}-mlx`).

**Real bug found and fixed along the way, worth remembering for any future
hand-authored workflow JSON**: ComfyUI's frontend auto-inserts a
`control_after_generate` widget immediately after any INT widget literally named
`seed` — even though none of these node packs' `define_schema()`s declare one
(`control_after_generate=None` by default, no opt-in). This is invisible from the
Python schema alone; it only shows up once the workflow is actually loaded in the
browser. A hand-written `widgets_values` array that doesn't budget a slot for it
silently shifts every value after `seed` out of alignment — e.g. a string like
`"zimage_mlx/image"` landing in a `FLOAT` widget's slot. Root-caused by loading
the graph in a real headless browser via Playwright and reading
`node.widgets.map(w => ({name: w.name, value: w.value}))` directly, not by
guessing from the schema — the widget's *serialized position*, not its declared
name, is what actually gets consumed. **Lesson: always verify a hand-authored
workflow JSON's `widgets_values` against a real loaded graph in the browser
(`app.loadGraphData()` + reading live widget values), never trust the schema
order alone when a `seed`-named INT widget is present anywhere in the node.**

Also confirmed (contrary to an earlier assumption checked and discarded this
session): the newer Vue-based frontend's saved `links` array uses the classic
`[id, from_node, from_slot, to_node, to_slot, type]` tuple format, not an
object form — that wasn't the actual cause of an early workflow failing to load;
the `control_after_generate` misalignment was.

### Real mflux bug found and fixed in `comfyui-krea2-mlx`

Krea2 generation via the pre-quantized `MLXBits/krea-2-mlx-q8`/`-q4` repos
(previously recorded as "verified genuinely compatible" in this doc's original
Krea-2 + Z-Image session) started failing with `FileNotFoundError: Missing
specified weight files ... ['turbo.safetensors']` — the installed `mflux` version
had apparently moved on since that verification. Root-caused by reading the
actually-installed `mflux` 0.18.1 source directly: `Krea2WeightDefinition`
(`mflux/models/krea2/weights/krea2_weight_definition.py`) hardcodes the
transformer component's weight file as a single literal `turbo.safetensors`, and
its own `get_download_patterns()` only ever fetches that exact filename — but the
MLXBits pre-quantized repos ship the transformer as root-level numbered shards
(`0.safetensors`..`6.safetensors` + `model.safetensors.index.json`) instead of a
single file, a genuinely different, incompatible layout. Not fixable in mflux
from this repo; worked around in `comfyui-krea2-mlx/nodes/loaders.py` instead:
`_ensure_turbo_safetensors()` does its own separate, narrowly-scoped
`snapshot_download` for the root-level shards (mflux's own restrictive
`allow_patterns` never fetches them), then merges them via `mlx.load`/
`mx.save_safetensors` into a `turbo.safetensors` file inside the same HF cache
snapshot directory — one-time and idempotent, since `PathResolution.resolve`'s
`exists_locally` rule (confirmed by reading `mflux/models/common/resolution/
path_resolution.py`) takes priority over re-downloading once a local path is
handed back.

**Verified for real, not just at the merge-logic level**: loaded the actual
updated `Krea2ModelLoader`/`Krea2Generate` node classes directly (via
`importlib.util.spec_from_file_location` with `submodule_search_locations`, to
handle the hyphenated package directory name the same way ComfyUI itself does)
and ran a full pipeline load + real generation end-to-end — produced a clean
image matching the prompt. This is the repo's existing "layer 1: direct Python
calls" testing methodology, used here specifically because the live ComfyUI
server's already-imported Python module can't hot-reload edited node code
without a restart, and restarting risked losing an undocumented `HF_HOME`
env var override that isn't set in the interactive shell (confirmed via
`lsof`/`ps` that the running server process has it set somehow, but it doesn't
appear in the shell's own environment — likely set by whatever launched it,
StabilityMatrix's own process launcher).

**Disk space note**: this session's Krea2 q8 download (root-level shards, since
the vae/text_encoder portion was already cached from an earlier failed attempt)
needed ~22GB total and briefly pushed free disk down to ~18GB before the merge
step (which needs its own ~14GB for the merged `turbo.safetensors`, in addition
to the shard files it reads from) completed. Also found and cleaned up ~2.8GB of
orphaned `.incomplete` blob files left over from the original failed attempt
(different content hashes than the successful retry, so not deduplicated
automatically) — worth checking for on any machine that hit the original bug
before this fix existed.

### Krea2 remaining known gaps

- The other three MLXBits→turbo.safetensors-shape mismatches this fix doesn't
  address (if they exist): only `krea2` was affected/tested here, since it's the
  only model in `comfyui-krea2-mlx`. `flux2-klein-*`/`z-image-*`'s pre-quantized
  `mlx-community` mirrors were not affected — confirmed via their own successful
  example-workflow runs this same session, so this is a `Krea2WeightDefinition`
  -specific quirk, not a general mflux pre-quantized-repo problem.
- `_ensure_turbo_safetensors()` doesn't verify the merged shards' key names
  actually match what `Krea2WeightMapping.get_transformer_mapping()` expects
  beyond "generation ran and produced a sane-looking image" — that's fairly
  strong evidence, but not the same as inspecting the key schema directly.
