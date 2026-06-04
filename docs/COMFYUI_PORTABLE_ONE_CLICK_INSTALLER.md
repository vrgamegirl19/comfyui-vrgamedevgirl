# VRGDG ComfyUI Portable One-Click Installer

Use `install_vrgdg_comfyui_portable.bat` for Windows 11 users who need a simple portable ComfyUI install.

## What It Does

- Lets the user choose an install folder.
- Downloads the latest official ComfyUI Windows portable Nvidia build.
- Installs `comfyui-vrgamedevgirl` from the `main` branch.
- Installs the repo `requirements.txt` files with the portable embedded Python.
- Installs helpful custom nodes for the bundled workflows:
  - ComfyUI Manager
  - ComfyUI VideoHelperSuite
  - ComfyUI LTXVideo
  - rgthree-comfy
  - ComfyUI Crystools
  - ComfyUI KJNodes
- Downloads a local FFmpeg build if FFmpeg is not already on PATH.
- Checks or installs the Microsoft Visual C++ 2015-2022 x64 runtime.
- Creates `run_vrgdg_nvidia_gpu.bat` inside `ComfyUI_windows_portable`.

## Recommended User Steps

1. Download or clone this repo.
2. Double-click `install_vrgdg_comfyui_portable.bat`.
3. Choose an install folder.
4. Wait for downloads and Python requirements to finish.
5. Start ComfyUI with:

```bat
ComfyUI_windows_portable\run_vrgdg_nvidia_gpu.bat
```

## Build Choice

The script defaults to `cu126-py312` because VRGDG workflows currently include packages such as `llama-cpp-python`, video/audio tooling, and Gemma-related nodes that are usually less painful on Python 3.12.

For the newest official RTX portable build, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-VRGDG-ComfyUI-Portable.ps1 -PortableBuild cu130-py313
```

## Notes

This script does not download model weights. Model files can be large, workflow-specific, and may have license gates. If ComfyUI still reports missing custom nodes after startup, open ComfyUI Manager and use **Install Missing Custom Nodes**.
