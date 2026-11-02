# MiniMax H3 pass 2 enhancer

Select **MiniMax H3 → 2 Pass**, then open **Video Settings → Pass 2 LoRA / Refinement** and check **Use NVIDIA Enhancer on Pass 2**.

Download both LoRAs using the links beneath the checkbox:

- [DMAD full-critic 4-step LoRA, ComfyUI format](https://huggingface.co/ZhengmingYu/DMAD/resolve/main/minimax_h3/dmad_minimax_h3_4step_full_critic_comfyui.safetensors?download=true).
- [Open-H3 2K Refiner, rank 16 BF16](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors?download=true).

Place the files in `ComfyUI/models/loras` (subfolders are supported), refresh the model list, and select them in the two enhancer fields. The DMAD download uses the ComfyUI key layout; the upstream file without `_comfyui` uses the Diffusers layout. The Builder checks that both selected files are registered before queueing a new render and names any missing file in the error.

When enabled, pass 2 starts from the base model, applies **DMAD at 1.0**, then **Open-H3 Refiner at 1.0**, and uses **er_sde** with manual sigmas **`0.7, 0.494118, 0`** (two refinement steps). This replaces the usual pass 2 Turbo LoRA and scheduler. The overridden sampling fields are hidden; the pass 2 seed remains editable. Optional additional LoRAs and acceleration controls retain their configured behavior.

Pass 1 keeps its model settings and optional LoRAs. No Enhancer LoRA is required or automatically added to pass 1. Learned latent upscaling and the selected audio mode keep their existing behavior.

Uncheck the option to restore normal pass 2 controls and their saved values. The option is saved with project settings, locked scene overrides, and video profiles. It applies only to **2 Pass**, not Single or 2 Pass Advanced.

Drafts save the recipe when pass 1 is rendered. Finalizing uses that saved graph. Enable this option before making the draft; changing it afterward does not rewrite an existing draft's pass 2 workflow.

Agents can set `two_pass_use_enhancer`, `two_pass_enhancer_dmad_name`, and `two_pass_enhancer_refiner_name` in the `minimax_h3` settings group. The backend enforces the recipe even if a caller supplies different pass 2 steps or sampler settings.
