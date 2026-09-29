export const BUILDER_UI_VERSION = "welcome-startup-2026-05-20";
export const DEFAULT_I2V_UNET = "LTX-2.3-22B-distilled-1.1-Q6_K.gguf";
export const DEFAULT_I2V_DIFFUSION_MODEL = "LTX_8bit\\ltx-2.3-22b-dev_transformer_only_int8_convrot.safetensors";
export const BAD_I2V_UNET_ALIASES = new Set(["LTX-2.3-22B-distilled-11-Q6_K.gguf"]);
export const REQUIRED_LTX_MSR_LORA = "licon\\LTX-2.3-Licon-MSR-V1.safetensors";
export const REQUIRED_LTX25_MSR_LORA = "LTX-2.5-Licon-MSR-V1.safetensors";
export const REQUIRED_LTX_INGREDIENTS_LORA = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors";
export const REQUIRED_LTX_ID_LORA = "lora_weights.safetensors";
export const REQUIRED_LTX_ID_LORA_URL = "https://huggingface.co/AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K";
export const DEFAULT_LTX_INGREDIENTS_WIDTH = 768;
export const DEFAULT_LTX_INGREDIENTS_HEIGHT = 448;
export const I2V_SAMPLER_OPTIONS = [
  "euler_ancestral",
  "euler",
  "euler_cfg_pp",
  "euler_ancestral_cfg_pp",
  "dpmpp_2m",
  "dpmpp_2m_sde",
  "dpmpp_3m_sde",
  "uni_pc",
];
export const DEFAULT_I2V_PASS1_SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0";
export const DEFAULT_I2V_PASS2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0";
export const DEFAULT_INGREDIENTS_SAMPLER = "euler_ancestral_cfg_pp";
export const LOCATION_MAPPER_GPT_URL = "https://chatgpt.com/g/g-6a2df090651c819190b00d7974677ad2-ltx-2-3-video-builder-location-creator-mapper";
export const LOCATION_SCOUT_GPT_URL = "https://chatgpt.com/g/g-6a3ff63879048191b30df4168cbea80a-music-video-location-scout";
export const LOCATION_SCOUT_ADVANCED_CHATGPT_URL = "https://chatgpt.com/";
export const SCENE_MAPPING_GPT_URL = "https://chatgpt.com/g/g-6a3a00f5cd508191a0a94ab5356e0b63-ltx-2-3-scene-mapping-assistant";
export const BUY_ME_A_COFFEE_URL = "https://buymeacoffee.com/vrgamedevgirl";
export const TIMELINE_HEIGHT = 210;
export const TIMELINE_OVERLAY_TOP = 24;
export const TIMELINE_OVERLAY_HEIGHT = 50;
export const TIMELINE_SEGMENT_TOP = 86;
export const TIMELINE_SEGMENT_HEIGHT = 62;
export const TIMELINE_SCENE_AUDIO_TOP = TIMELINE_SEGMENT_TOP + TIMELINE_SEGMENT_HEIGHT + 10;
export const TIMELINE_SCENE_AUDIO_HEIGHT = 28;
export const TIMELINE_NOTE_HEIGHT = 58;
export const TIMELINE_NOTE_GAP = 12;
const TIMELINE_WAVE_TOP = 98;
export const TIMELINE_MARKER_HEIGHT = 58;
export const TIMELINE_MARKER_MIN_WIDTH = 110;
export const FLUX_GEMMA_TIMEOUT_MS = 30 * 60 * 1000;
export const DEFAULT_NON_VISION_GEMMA_MODEL = "supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf";
export const NB_IMAGE_MODELS = ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"];
export const DEFAULT_NB_IMAGE_MODEL = "gemini-3-pro-image-preview";
export const BUILDER_FONT_STACK = "Segoe UI, Inter, Roboto, Arial, sans-serif";
export const WAVEFORM_MODES = {
  small: { label: "Small wave", height: 150, gain: 1 },
  medium: { label: "Medium wave", height: 190, gain: 1.35 },
  large: { label: "Large wave", height: 240, gain: 1.85 },
};

// Safe rollout switch: the legacy side-panel generators remain in the builder and can
// be restored by changing this one value to false without deleting them.
export const USE_STORYBOARD_PROMPT_PIPELINE_FOR_SIDE_PANEL = true;
