# VRGDG Flow Automation

Browser automation helper used by the ComfyUI node `VRGDG Flow Browser Image Edit`.
The same setup is also shared by `VRGDG ChatGPT Images Browser`.

## Setup

The easiest setup path is to add and run this ComfyUI node once:

```text
VRGDG Flow Browser Setup
```

On Windows it can download a private portable Node.js runtime into `runtime/`. On Linux and macOS it uses system Node.js/npm. It then runs `npm install` in this folder for you.

Chrome or Chromium is still required because the automation controls the providers in a real browser profile.

Manual setup is also available:

Linux/macOS:

```bash
cd ComfyUI/custom_nodes/comfyui-vrgamedevgirl/flow_automation
npm install
```

Windows PowerShell:

```powershell
cd ComfyUI\custom_nodes\comfyui-vrgamedevgirl\flow_automation
npm install
```

If npm has certificate issues on your machine, this local project can be installed with:

```powershell
npm install --strict-ssl=false
```

## First Login

Run either launcher once:

```text
Flow Fully Automatic.bat
```

or:

```text
Flow Image Edit Automatic.bat
```

Chrome opens with a separate profile stored in `chrome-flow-profile/`. Sign into Google Flow once in that browser profile. Do not share or commit `chrome-flow-profile/`.

GPT Image uses `chrome-chatgpt-profile/`, and Meta AI uses `chrome-meta-profile/`. Each provider must be logged into separately. The `.bat` launchers are Windows-only.

## ComfyUI Node

The ComfyUI node saves temporary input images to:

```text
inputs/
```

Flow browser downloads are staged internally at:

```text
outputs/
```

ChatGPT Images browser downloads are staged internally at:

```text
chatgpt_outputs/
```

These folders are generated/runtime data and are ignored by git except for placeholder `.gitkeep` files.
In normal ComfyUI usage, connect the browser node's `image` output to a standard `Save Image` node and let ComfyUI save the final image into the normal output folder.

The nodes find this `flow_automation/` folder automatically from the installed custom node location, so shared workflows should not need a machine-specific `flow_dir` path.

## Notes

- Requires Chrome or Chromium. Set `VRGDG_CHROME_PATH` (or `CHROME_PATH`) when the browser executable is in a custom location.
- Node.js can be installed locally by the setup node on Windows, unless you disable `install_portable_node`.
- Linux and macOS require system Node.js/npm available on `PATH`.
- Requires Google Flow access on the signed-in account.
- Requires ChatGPT image generation access on the signed-in account when using the ChatGPT Images node.
- Requires Meta AI access on the signed-in account when using Meta AI.
- This is UI automation; Google Flow UI changes may require selector updates.
- ChatGPT UI changes may require selector updates.
