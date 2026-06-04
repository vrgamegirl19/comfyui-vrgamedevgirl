VRGDG ComfyUI Portable Installer
================================

This folder installs the portable Windows version of ComfyUI for an Nvidia GPU.
It is meant for Windows 11 users with an RTX card, such as an RTX 4090.

Quick Start
-----------

1. Extract this zip somewhere easy to find.
2. Double-click:

   install_vrgdg_comfyui_portable.bat

3. When asked, choose the folder where you want ComfyUI installed.
   Example:

   C:\AI\ComfyUI

4. Wait for the installer to finish.
   It may take a while because ComfyUI portable is large and Python packages need
   to install.

5. When it is done, start ComfyUI with:

   ComfyUI_windows_portable\run_vrgdg_nvidia_gpu.bat

What This Installs
------------------

- ComfyUI portable for Windows/Nvidia.
- VRGameDevGirl's ComfyUI custom nodes.
- Python requirements needed by the VRGDG nodes.
- ComfyUI Manager.
- Several extra custom node packs commonly needed by the included workflows.
- FFmpeg, if it is not already installed.
- Microsoft Visual C++ runtime, if Windows needs it.

Important Notes
---------------

- Keep the black command window open while using ComfyUI.
- ComfyUI normally opens at:

  http://127.0.0.1:8188

- The installer does not download AI model files. Some workflows may still need
  models downloaded separately.
- If ComfyUI says a workflow has missing custom nodes, open ComfyUI Manager and
  use "Install Missing Custom Nodes".

If Something Goes Wrong
-----------------------

Take a screenshot or copy the error text from the installer window and send it
to VRGameDevGirl.

The most useful things to send are:

- The last 20-30 lines shown in the installer window.
- Any red error text.
- Your install folder path.
- Whether you are using Windows 11 and an Nvidia GPU.

Advanced Option
---------------

The installer defaults to the compatibility build because it is usually friendlier
for video, audio, Gemma, and llama-cpp-python related packages.

If VRGameDevGirl asks you to use the newest RTX/CUDA 13 build instead, open
PowerShell in this folder and run:

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-VRGDG-ComfyUI-Portable.ps1 -PortableBuild cu130-py313
