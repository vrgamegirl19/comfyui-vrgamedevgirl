export const LTX_23_MODEL_DOWNLOADS = [
  { label: "LTX GGUF", url: "https://huggingface.co/Abiray/LTX-2.3-22B-DISTILLED-1.1-GGUF/tree/main" },
  { label: "Video VAE", url: "https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/vae" },
  { label: "Gemma Clip", url: "https://huggingface.co/Sikaworld1990/gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition-Ltx-2/resolve/main/gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors" },
  { label: "Text Projection", url: "https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/text_encoders" },
  { label: "Latent Upscaler", url: "https://huggingface.co/prince-canuma/LTX-2.3-distilled/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" },
  { label: "Audio VAE", url: "https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/vae" },
];
export const LTX_25_MODEL_DOWNLOADS = [
  { label: "Gemma 4 E2B Text Encoder", url: "https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/gemma4_e2b_it_bf16.safetensors" },
  { label: "Gemma 4 12B LTX 2.5 Text Encoder", url: "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" },
  { label: "LTX 2.5 Distilled Transformer", url: "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" },
  { label: "LTX 2.5 Video VAE", url: "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors" },
  { label: "LTX 2.5 Audio VAE", url: "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors" },
  { label: "LTX 2.5 Latent Upscaler", url: "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" },
];
export const MINIMAX_H3_MODEL_DOWNLOADS = [
  { label: "Diffusion model", url: "https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" },
  { label: "Qwen3-VL text encoder", url: "https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" },
  { label: "Video VAE", url: "https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors" },
  { label: "Audio VAE", url: "https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors" },
  { label: "Kijai MiniMax H3 LoRAs", url: "https://huggingface.co/Kijai/MiniMax-H3_comfy/tree/main/loras" },
  { label: "MMH3 Ultimate Upscale custom nodes", url: "https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale" },
  { label: "MiniMax H3 latent upscaler models", url: "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler" },
];
export const VIDEO_BUILDER_CUSTOM_NODES = [
  { id: "vrgdg", label: "VRGameDevGirl Video Builder", note: "The builder’s own custom-node pack. This is the pack currently providing this interface and its VRGDG nodes.", url: "https://github.com/vrgamegirl19/comfyui-vrgamedevgirl", current: true },
  { id: "videohelpersuite", label: "ComfyUI-VideoHelperSuite", note: "Video loading, combining, and output nodes used by the builder workflows.", url: "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite" },
  { id: "kjnodes", label: "ComfyUI-KJNodes", note: "KJ utility, video, model-loader, and optimization nodes used by LTX and MiniMax workflows.", url: "https://github.com/kijai/ComfyUI-KJNodes" },
  { id: "gguf", label: "ComfyUI-GGUF", note: "GGUF diffusion and text-encoder loaders used by the LTX 2.3 and local LLM options.", url: "https://github.com/city96/ComfyUI-GGUF" },
  { id: "ltxvideo", label: "ComfyUI-LTXVideo", note: "Official LTX-Video support, including the sampler wrappers used by LTX builder workflows.", url: "https://github.com/Lightricks/ComfyUI-LTXVideo" },
  { id: "te_speed_minimax_h3", label: "TE-Speed-MiniMaxH3-OSS", note: "Optional MiniMax H3 acceleration node used by the TE-Speed setting in two-pass workflows.", url: "https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS" },
  { id: "mmh3_ultimate_upscale", label: "Comfyui-MMH3-UltimateUpscale", note: "Optional MiniMax H3 advanced upscale node used by the advanced upscale/refinement path.", url: "https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale" },
  { id: "minimax_h3_audio_t8", label: "comfyui-minimax-h3-audio-T8", note: "Required by MiniMax H3 Ref to Video 2 Pass for audio/video latent separation.", url: "https://github.com/T8mars/comfyui-minimax-h3-audio-T8" },
  { id: "minimax_h3_latent_upscaler", label: "Comfyui_Minimax_h3_latent_Upscaler", note: "Required by MiniMax H3 Ref to Video 2 Pass and the experimental video upscaler. The latent-upscaler checkpoint must also be installed separately.", url: "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler" },
];
export const ZIMAGE_MODEL_DOWNLOADS = [
  { label: "Z-Image Turbo", url: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" },
  { label: "Qwen CLIP", url: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" },
  { label: "Z-Image VAE", url: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" },
];
export const KREA2_MODEL_DOWNLOADS = [
  { label: "Krea2 diffusion model", url: "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors" },
  { label: "Krea2 text encoder", url: "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors" },
  { label: "Krea2 VAE", url: "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors" },
];
export const DEFAULT_KREA2_REFERENCE_SETTINGS = {
  krea_unet_name: "krea2_turbo_fp8_scaled.safetensors",
  krea_clip_name: "qwen3vl_4b_fp8_scaled.safetensors",
  krea_vae_name: "qwen_image_vae.safetensors",
  z_unet_name: "z_image_turbo_bf16.safetensors",
  z_clip_name: "qwen_3_4b.safetensors",
  z_vae_name: "ae.safetensors",
  first_pass_width: 1024,
  first_pass_height: 576,
  width: 1920,
  height: 1080,
  seed: 1,
  seed_mode: "fixed",
};

export function cloneKrea2ReferenceSettings(settings = {}) {
  return {
    ...DEFAULT_KREA2_REFERENCE_SETTINGS,
    ...(settings && typeof settings === "object" ? settings : {}),
  };
}
export const FLUX_KLEIN_9B_MODEL_DOWNLOADS = [
  { label: "9B diffusion model", url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors" },
  { label: "9B Qwen CLIP", url: "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" },
  { label: "9B VAE", url: "https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors" },
];
export const FLUX_KLEIN_4B_MODEL_DOWNLOADS = [
  { label: "4B diffusion model", url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors" },
  { label: "4B Qwen CLIP", url: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" },
  { label: "4B VAE", url: "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" },
];
export const ERNIE_MODEL_DOWNLOADS = [
  { label: "Ernie diffusion model", url: "https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/diffusion_models/ernie-image-turbo.safetensors" },
  { label: "Ministral text encoder", url: "https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/text_encoders/ministral-3-3b.safetensors" },
  { label: "Ernie VAE", url: "https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/vae/flux2-vae.safetensors" },
];
export const GEMMA_LLM_MODEL_DOWNLOADS = [
  { label: "SuperGemma GGUF", url: "https://huggingface.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2/resolve/main/supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf" },
  { label: "Gemma Vision GGUF", url: "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-IQ2_M.gguf" },
  { label: "Gemma Vision mmproj", url: "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/mmproj-BF16.gguf" },
];
export const QWEN_LLM_MODEL_DOWNLOADS = [
  { label: "Qwen3.8-27B GGUF (choose a quantization/model)", url: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main" },
  { label: "Qwen3.8 vision mmproj (rename to qwen-mmproj-BF16.gguf)", url: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-BF16.gguf" },
];
export const MODEL_FOLDER_HINTS = {
  "LLM / Gemma": `ComfyUI/
models/
  LLM/
    supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf
    gemma-4-26B-A4B-it-UD-IQ2_M.gguf
    mmproj-BF16.gguf

Gemma Vision requires both the model GGUF and its matching mmproj.`,
  "LLM / Qwen": `ComfyUI/
models/
  LLM/
    <chosen Qwen3.8 GGUF model files>
    qwen-mmproj-BF16.gguf

Qwen3.8 requires BOTH the chosen model GGUF (including all shards for that quantization) and its matching vision mmproj.
Rename the downloaded Qwen projector to qwen-mmproj-BF16.gguf.`,
  "ZImage": `ComfyUI/
models/
  text_encoders/
    qwen_3_4b.safetensors
  diffusion_models/
    z_image_turbo_bf16.safetensors
  vae/
    ae.safetensors`,
  "Krea2": `ComfyUI/
models/
  diffusion_models/
    krea2_turbo_fp8_scaled.safetensors
  text_encoders/
    qwen3vl_4b_fp8_scaled.safetensors
  vae/
    qwen_image_vae.safetensors`,
  "Flux/Klein 9B": `ComfyUI/
models/
  diffusion_models/
    flux-2-klein-9b-fp8.safetensors
  text_encoders/
    qwen_3_8b_fp8mixed.safetensors
  vae/
    full_encoder_small_decoder.safetensors`,
  "Flux/Klein 4B": `ComfyUI/
models/
  text_encoders/
    qwen_3_4b.safetensors
  diffusion_models/
    flux-2-klein-4b-fp8.safetensors
  vae/
    flux2-vae.safetensors`,
  "Ernie Image": `ComfyUI/
models/
  diffusion_models/
    ernie-image-turbo.safetensors
  text_encoders/
    ministral-3-3b.safetensors
  vae/
    flux2-vae.safetensors`,
  "LTX 2.3": `ComfyUI/
models/
  diffusion_models/
    ltx-2.3-distilled_1.1-Q6_k.gguf
  text_encoders/
    ltx-2.3-text_projection_bf16.safetensors
    abliterated-sikaworld-high-fidelity-edition.safetensors
  vae/
    LTX2.3_video_vae_bf16.safetensors
    LTX2.3_audio_vae_bf16.safetensors
  latent_upscale_models/
    ltx-2.3-spatial-upscaler-x2-1.1.safetensors`,
  "LTX 2.5": `ComfyUI/
models/
  text_encoders/
    gemma4_e2b_it_bf16.safetensors
    gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
  diffusion_models/
    ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
  vae/
    ltx-2.5-video-vae-bf16.safetensors
    ltx-2.5-audio-vae-bf16.safetensors
  latent_upscale_models/
    ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`,
  "MiniMax H3": `ComfyUI/
models/
  diffusion_models/
    minimax_h3_ref2va_pruned_int8_convrot.safetensors
  text_encoders/
    qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
  vae/
    minimax_h3_video_vae_fp16.safetensors
    minimax_h3_audio_vae_fp32.safetensors`,
};
