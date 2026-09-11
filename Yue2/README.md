# VRGDG YuE2 nodes

Experimental ComfyUI integration for [YuE2](https://github.com/multimodal-art-projection/YuE).

## Nodes

- **VRGDG YuE2 Settings** selects the runtime, model, VAE, backend, and memory policy.
- **VRGDG YuE2 Generate Song** generates 48 kHz stereo ComfyUI `AUDIO` plus the ABC score and complete artifact directory.
- **VRGDG YuE2 Create Plan** generates an editable symbolic ABC score without rendering audio.
- **VRGDG YuE2 Render ABC** renders an original or edited ABC score with lyrics and style.
- **VRGDG YuE2 Unload Models** releases cached models used by the experimental in-process runtime.
- **VRGDG SheetSage2 Settings** selects the separate audio-transcription runtime.
- **VRGDG SheetSage2 Transcribe Cover** converts uploaded song audio into an editable ABC lead sheet and retains its MIDI/event artifacts.

The Generate Song, Create Plan, and Render ABC nodes include 35 editable style
presets: five each for rock, pop, country, rap, hip hop, metal, and 1990s
alternative. Selecting a preset fills the style prompt. Editing the filled text
automatically switches the selector back to **Custom / Keep typed style** so the
customized wording is used.

Generate Song and Render ABC expose `ode_steps` for final audio synthesis. The
official default is 32. Lower values such as 16 render the synthesis stage faster
with reduced fidelity; 48–64 are slower and usually have diminishing returns.
This setting does not reduce the earlier ABC-planning or semantic-token stages.

## Starter workflow

Import `Workflows/Yue2/VRGDG_YuE2_Starter.json` into ComfyUI and edit the style and lyrics directly in **VRGDG YuE2 Generate Song**. The installer node's `yue2_config` output is already connected to Generate Song. Change its single `target_root` value if YuE2 is installed somewhere other than `E:\Yue2`.

The workspace starter workflow is currently configured for this local installation:

- Python: `E:\Yue2\.venv\Scripts\python.exe`
- Generator: `E:\Yue2\models\YuE2-3B`
- VAE: `E:\Yue2\models\YuE2-Vae`
- Offline/local-files-only mode: enabled

## Cover workflow

Import `Workflows/Yue2/VRGDG_YuE2_Cover.json`. Select a legally usable source song in **Load Source Song**, paste lyrics that match the source song's phrasing, describe the new arrangement/voice in **Render Cover**, then queue the graph.

The installer node is already connected twice: `sheetsage2_config` feeds the
transcription node and `yue2_config` feeds Render ABC. Change its single
`target_root` value if necessary. The graph then runs in two isolated processes:

1. SheetSage2 transcribes the recording into `score.abc` with `melody_only` enabled.
2. YuE2 renders that melody as a new performance using the supplied style and lyrics.

The local paths expected by the workflow are:

- SheetSage2 Python: `E:\Yue2\SheetSage2-venv\Scripts\python.exe`
- SheetSage2 model: `E:\Yue2\models\SheetSage2`
- MERT parent model: `E:\Yue2\models\MERT-v2-FullSong`
- Hugging Face runtime cache: `E:\Yue2\hf-cache`
- YuE2 Python/model/VAE: the same paths listed above

SheetSage2 does not transcribe sung words. Lyrics are intentionally a manual field so syllables and sections can be corrected before rendering. `melody_only=true` is the intended cover mode: it keeps the vocal/instrument melody voices while leaving YuE2 freedom to create a new arrangement.

## Recommended isolated installation

YuE2 pins its own Torch and Transformers versions. Keep those packages out of ComfyUI's embedded Python environment.

Create a separate environment using the official YuE2 installation instructions, then enter that environment's Python executable in **VRGDG YuE2 Settings**. Typical paths are:

- Windows venv: `C:\path\to\YuE\.venv\Scripts\python.exe`
- Linux/WSL venv: `/path/to/YuE/.venv/bin/python`

The official runtime currently targets Linux, Python 3.10+, an NVIDIA BF16 GPU, and 24 GB VRAM. Native Windows operation is experimental.

### Installer node

Add **VRGDG YuE2 Installer + Settings**, choose a dedicated installation folder
(for example `D:\Yue2`), and use one of its explicit buttons. **Install
Everything** installs the official YuE2 source, a separate YuE2 virtual
environment, YuE2-3B and YuE2-Vae, a dependency-isolated SheetSage2 virtual
environment, SheetSage2, and MERT-v2-FullSong. It then verifies CUDA, BF16,
imports, and model files and writes `vrgdg_yue2_install_report.json`.

Installation never starts merely because a workflow is queued. After setup,
the same node acts as portable settings: connect `yue2_config` to the YuE2
generation/render node and `sheetsage2_config` to the transcription node. Model
downloads retain their upstream licenses and can require substantial disk space.

The combined installer/settings node retains the runtime controls needed for
normal use: device, memory budget, backend, quantization, AR offload, offline
mode, ComfyUI model unloading, and optional model-weight hash verification.

For a standalone setup screen, load
`Workflows/Yue2/VRGDG_YuE2_Setup.json`. On Windows, install Python 3.10 or
3.11 with the `py` launcher before using **Install Everything**. The installer
node installs isolated environments, packages, and model files, but it does not
install or modify system-wide Python. [Python 3.11.9 for
Windows](https://www.python.org/downloads/release/python-3119/) is a compatible
choice; confirm it is registered by running `py -3.11 --version`, then restart
ComfyUI. The generation
environment may use Python 3.10–3.12; the separately pinned cover environment
uses Python 3.10 or 3.11. Full installation can take many minutes because it
downloads CUDA-enabled Torch builds and several large model snapshots. Progress
is printed in the ComfyUI console.

`torch` is the recommended backend and uses CUDA graphs. `torch-eager` is available as a troubleshooting fallback but is substantially slower for autoregressive score and song generation. FP8 is experimental in YuE2 and is not enabled by default.

On Windows, some Torch wheels expose the Flash Attention operator even though
its CUDA implementation was not compiled. The worker detects that condition
and keeps the fast `torch` CUDA-graph backend active using cuDNN attention (or
SDPA when cuDNN is unavailable). Restart ComfyUI after updating the nodes.

Isolated workers stream YuE2/SheetSage2 stage updates into the ComfyUI console and also preserve complete `stdout.log` and `stderr.log` files beside each run. A 15-second heartbeat reports the worker PID and elapsed time if an upstream stage temporarily produces no messages.

`verify_hashes=false` skips rereading and hashing all model weights on every isolated run. The local `E:\Yue2` weights were integrity-verified during installation, so both included workflows disable the repeated scan. Turn it back on after replacing or updating model files.

The first run downloads `m-a-p/YuE2-3B` and `m-a-p/YuE2-Vae` from Hugging Face unless `local_files_only` is enabled.

## Output

Each run creates a unique folder under `ComfyUI/output/Yue2`. A complete generation retains audio, ABC, semantic tokens, acoustic latents, configuration, timing, hashes, and truncation information.

## License note

YuE2 source code is Apache-2.0, while the official model weights are CC BY-NC 4.0. Review the upstream licenses before distribution or commercial use.

SheetSage2 and MERT-v2-FullSong weights are also CC BY-NC 4.0. Only make covers when you have the rights or permission required for the source composition, recording, lyrics, and intended distribution.
