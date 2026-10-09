export const MINIMAX_H3_MODE_OPTIONS = [
  { value: "text_to_video", label: "Text to Video", buttonLabel: "T2V" },
  { value: "image_to_video", label: "Image to Video", buttonLabel: "I2V" },
  { value: "image_reference_to_video", label: "Image + Reference 2 Pass", buttonLabel: "Image + Ref\n2 Pass" },
  { value: "reference_to_video", label: "Reference to Video", buttonLabel: "Ref to\nVideo" },
  { value: "video_to_video", label: "Video to Video", buttonLabel: "V2V" },
];
const MINIMAX_H3_INSTRUCTION_KEYS = {
  text_to_video: "minimax_h3_text_to_video",
  image_to_video: "minimax_h3_image_to_video",
  reference_to_video: "minimax_h3_reference_to_video",
  image_reference_to_video: "minimax_h3_image_reference_to_video",
  video_to_video: "minimax_h3_video_to_video",
};
export const MINIMAX_H3_VIDEO_REFERENCE_PURPOSES = [
  { value: "continuation", label: "Continuation / Extension" },
  { value: "movement", label: "Movement Guide" },
  { value: "camera", label: "Camera Guide" },
  { value: "edit_rhythm", label: "Edit / Rhythm Guide" },
  { value: "transformation", label: "Transformation Source" },
  { value: "visual_style", label: "Visual Style Guide" },
];
export const MINIMAX_H3_AUDIO_MODE_OPTIONS = [
  { value: "input_audio", label: "Input Audio (exact supplied audio)" },
  { value: "built_in_audio", label: "Built-in MiniMax Audio" },
];
export const MINIMAX_H3_VOICE_PRESETS = [
  { value: "none", label: "No voice preset", gender: "", name: "", description: "" },
  {
    value: "female_velvet_ember",
    label: "Woman — Velvet Ember (American)",
    gender: "female",
    name: "VELVET EMBER",
    description: "an adult feminine low contralto with a warm smoky texture, a soft neutral American accent, measured confident cadence, crisp consonants, gentle breathiness, and clean intimate close-microphone delivery.",
  },
  {
    value: "female_silver_thistle",
    label: "Woman — Silver Thistle (British)",
    gender: "female",
    name: "SILVER THISTLE",
    description: "an adult feminine clear mid-range voice with an unmistakable polished British Received Pronunciation accent, non-rhotic pronunciation, elongated broad-A vowels in words such as can’t, after, and glass, measured aristocratic English cadence, light airy resonance, precise consonants, elegant warmth, and clean intimate close-microphone delivery.",
  },
  {
    value: "female_sunlit_wattle",
    label: "Woman — Sunlit Wattle (Australian)",
    gender: "female",
    name: "SUNLIT WATTLE",
    description: "an adult feminine bright mezzo voice with an unmistakable contemporary General Australian English accent, non-rhotic pronunciation, broad open vowels, relaxed upward inflection, confident friendly cadence, crisp natural consonants, lively sunlit warmth, and clean intimate close-microphone delivery.",
  },
  { value: "female_custom", label: "Woman — Custom Voice", gender: "female", name: "", description: "" },
  {
    value: "male_iron_cedar",
    label: "Man — Iron Cedar (American)",
    gender: "male",
    name: "IRON CEDAR",
    description: "an adult masculine deep baritone voice with a clear neutral American accent, grounded resonant timbre, measured confident cadence, crisp consonants, subtle gravel, calm authority, natural warmth, and clean intimate close-microphone delivery.",
  },
  {
    value: "male_crowned_ash",
    label: "Man — Crowned Ash (British)",
    gender: "male",
    name: "CROWNED ASH",
    description: "an adult masculine polished baritone voice with an unmistakable British Received Pronunciation accent, non-rhotic pronunciation, elongated broad-A vowels in words such as can’t, after, and glass, measured articulate cadence, precise consonants, restrained dry warmth, quiet authority, and clean intimate close-microphone delivery.",
  },
  {
    value: "male_red_gum",
    label: "Man — Red Gum (Australian)",
    gender: "male",
    name: "RED GUM",
    description: "an adult masculine warm tenor-baritone voice with an unmistakable contemporary General Australian English accent, non-rhotic pronunciation, broad open vowels, relaxed upward inflection, easygoing confident cadence, clear natural consonants, bright resonant warmth, and clean intimate close-microphone delivery.",
  },
  { value: "male_custom", label: "Man — Custom Voice", gender: "male", name: "", description: "" },
];
// The pipeline is set for the whole project. "standard" renders from reference images. "refmod" renders from saved
// RefMods (see refmod_labels.mjs) and offers Single and 2 Pass only, with video_mode fixed to reference_to_video.
export const MINIMAX_H3_PIPELINE_OPTIONS = [
  { value: "standard", label: "Standard" },
  { value: "refmod", label: "RefMod" },
];
export const MINIMAX_I2V_TRANSITION_OPTIONS = [
  {value: "natural", label: "Natural movement"},
  {value: "surreal_morph", label: "Surreal morph"},
  {value: "dreamlike_dissolve", label: "Dreamlike dissolve"},
  {value: "environment_transformation", label: "Environment transformation"},
  {value: "camera_reveal", label: "Camera reveal"},
  {value: "custom", label: "Custom"},
];
export function normalizeMiniMaxI2VTransitionStyle(value) {
  const style = String(value || "").trim().toLowerCase();
  return MINIMAX_I2V_TRANSITION_OPTIONS.some(option => option.value === style) ? style : "natural";
}
export const DEFAULT_MINIMAX_H3_SETTINGS = {
  pipeline: "standard",
  video_mode: "text_to_video",
  render_pass: "two_pass",
  i2v_transition_style: "natural",
  i2v_transition_direction: "",
  i2v_pass_settings_version: 1,
  i2v_pass_profiles: {},
  audio_mode: "input_audio",
  continuity_mode: "off",
  continuity_prompt_from_last_frame: false,
  location_transition_preset: "normal",
  location_transition_custom: "",
  latent_context_frames: 39,
  diffusion_model_name: "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
  clip_name: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
  video_vae_name: "minimax_h3_video_vae_fp16.safetensors",
  audio_vae_name: "minimax_h3_audio_vae_fp32.safetensors",
  aspect_ratio: "16:9 (Widescreen)",
  // One output resolution for single pass, 2 Pass and 2 Pass Advanced (Pass 2 size).
  // `megapixels` always holds the resolved value; a preset recomputes it per aspect ratio.
  resolution_preset: "1k",
  megapixels: 0.5625,
  seed: 69,
  warmup_frames: 0,
  cooldown_frames: 0,
  sampler_name: "res_multistep",
  scheduler: "simple",
  steps: 20,
  steps_before_turbo: 20,
  denoise: 1,
  easy_cache_bypass: false,
  easy_cache_bypass_before_turbo: false,
  easy_cache_reuse_threshold: 0.3,
  easy_cache_start_percent: 0.2,
  easy_cache_end_percent: 0.9,
  easy_cache_verbose: false,
  sage_attention: "auto",
  use_memory_efficient_sage_attention: false,
  enable_fp16_accumulation: true,
  use_loras: false,
  lora_count: 0,
  loras: [],
  use_turbo_lora: false,
  turbo_lora_name: "minimax_h3_turbo_4step_ema_ckpt850.safetensors",
  turbo_lora_strength: 1,
  ref_image_size: "max",
  two_pass_lora_name: "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
  two_pass_lora_strength: 1,
  two_pass_lora_preset: 4,
  two_pass_defaults_version: 1,
  two_pass_latent_upscale_scale: 2,
  two_pass_latent_upscaler_name: "minimax_h3_latent_upscaler_3d_bf16.safetensors",
  use_te_speed: false,
  use_fast_vae_decode: false,
  two_pass_use_te_speed: false,
  use_feedforward: false,
  two_pass_use_feedforward: false,
  use_block_sparse_attention: false,
  two_pass_use_block_sparse_attention: false,
  two_pass_use_fast_vae_decode: false,
  two_pass_te_speed_processing_control: 0.07,
  two_pass_te_speed_start_percent: 0.1,
  two_pass_te_speed_end_percent: 0.9,
  two_pass_te_speed_mcs: 2,
  two_pass_te_speed_cache_depth: 0.75,
  two_pass_te_speed_device: "auto",
  two_pass_final_resize_method: "nvidia_rtx_vsr",
  two_pass_output_crf: 19,
  three_pass_lightx_lora_name: "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors",
  three_pass_lightx_lora_strength: 0.5,
  two_pass_pass1_megapixels: 0.5,
  two_pass_pass1_steps: 20,
  two_pass_pass1_denoise: 1,
  two_pass_pass1_sampler: "res_multistep",
  two_pass_pass1_scheduler: "simple",
  two_pass_pass1_seed: -1,
  two_pass_pass2_megapixels: 1.5,
  two_pass_pass2_steps: 2,
  two_pass_pass2_denoise: 0.2,
  two_pass_pass2_sampler: "res_multistep",
  two_pass_pass2_scheduler: "simple",
  two_pass_pass2_seed: -1,
  three_pass_pass1_megapixels: 0.4,
  three_pass_pass1_steps: 20,
  three_pass_pass1_denoise: 1,
  three_pass_pass1_sampler: "euler",
  three_pass_pass1_scheduler: "beta",
  three_pass_pass1_seed: 69,
  three_pass_pass1_te_speed: true,
  three_pass_pass2_megapixels: 2,
  three_pass_pass2_steps: 1,
  three_pass_pass2_denoise: 0.2,
  three_pass_pass2_sampler: "sa_solver",
  three_pass_pass2_scheduler: "simple",
  three_pass_pass2_seed: 69,
  three_pass_pass2_te_speed: false,
  three_pass_pass3_megapixels: 2,
  three_pass_pass3_steps: 5,
  three_pass_pass3_denoise: 0.2,
  three_pass_pass3_sampler: "euler",
  three_pass_pass3_scheduler: "beta",
  three_pass_pass3_seed: 69,
  three_pass_pass3_te_speed: false,
  // Tiles, chunks, fades and the upscaler device are derived from the output resolution at run time
  // (minimax/tile_plan.py); the only tiling choice is the VRAM preset (8-24 GB).
  advanced_two_pass_vram_preset: "16gb",
  advanced_two_pass_defaults_version: 4,
  advanced_two_pass_pass1_megapixels: 0.4,
  advanced_two_pass_pass1_resolution_preset: "custom",
  advanced_two_pass_pass1_steps: 20,
  advanced_two_pass_pass1_denoise: 1,
  advanced_two_pass_pass1_sampler: "euler",
  advanced_two_pass_pass1_scheduler: "beta",
  advanced_two_pass_pass1_seed: 69,
  advanced_two_pass_pass2_steps: 1,
  advanced_two_pass_pass2_denoise: 0.2,
  advanced_two_pass_pass2_sampler: "sa_solver",
  advanced_two_pass_pass2_scheduler: "simple",
  advanced_two_pass_pass2_seed: 69,
};

// 2 Pass Advanced VRAM presets (8-24 GB: bigger cards should use 2 Pass or single pass). Activations
// scale with chunk_length x tile area, so each card gets a tile-area target (megapixels). Mirrors
// minimax/tile_plan.py, which builds the real tile settings at run time.
export const MINIMAX_H3_VRAM_PRESETS = {
  "8gb": { tileMegapixels: 0.2, chunk: 51, overlap: 128 },
  "12gb": { tileMegapixels: 0.3, chunk: 85, overlap: 128 },
  "16gb": { tileMegapixels: 0.43, chunk: 119, overlap: 128 },
  "24gb": { tileMegapixels: 0.65, chunk: 153, overlap: 160 },
};
export const MINIMAX_H3_DEFAULT_VRAM_PRESET = "16gb";
export const MINIMAX_H3_RESOLUTION_PRESETS = [
  { value: "custom", label: "Custom megapixels" },
  { value: "1k", label: "H3 1K (1024×576)" },
  { value: "2k", label: "H3 2K (1920×1088)" },
  { value: "1440p", label: "1440p (2560×1440)" },
  { value: "4k", label: "H3 4K (3840×2176)" },
];

// Saved projects from before the 24 GB cap may hold 32gb or custom.
export function normalizeMiniMaxH3VramPreset(value) {
  const key = String(value || "").trim().toLowerCase();
  if (MINIMAX_H3_VRAM_PRESETS[key]) return key;
  if (key === "32gb" || key === "custom") return "24gb";
  return MINIMAX_H3_DEFAULT_VRAM_PRESET;
}

// Frame size in pixels (multiples of 32) for a megapixel target and an aspect ratio label such as
// "16:9 (Widescreen)". Mirrors minimax/resolution.py frame_size().
export function miniMaxH3FrameSize(megapixels, aspectRatio) {
  const match = String(aspectRatio || "16:9").match(/(\d+)\s*:\s*(\d+)/);
  const ratioWidth = Number(match?.[1] || 16);
  const ratioHeight = Number(match?.[2] || 9);
  const scale = Math.sqrt((Math.max(0.1, Number(megapixels) || 2) * 1048576) / (ratioWidth * ratioHeight));
  return {
    width: Math.round((ratioWidth * scale) / 32) * 32,
    height: Math.round((ratioHeight * scale) / 32) * 32,
  };
}

// Megapixels of a resolution preset at an aspect ratio, or null for "custom".
// Mirrors minimax/resolution.py preset_megapixels().
export function miniMaxH3PresetMegapixels(preset, aspectRatio) {
  const match = String(aspectRatio || "16:9").match(/(\d+)\s*:\s*(\d+)/);
  const ratioWidth = Number(match?.[1] || 16);
  const ratioHeight = Number(match?.[2] || 9);
  const longEdge = { "1k": 1024, "2k": 1920, "1440p": 2560, "4k": 3840 }[String(preset || "").toLowerCase()];
  if (!longEdge) return null;
  const scale = longEdge / Math.max(ratioWidth, ratioHeight);
  const width = Math.round(ratioWidth * scale / 32) * 32;
  const height = Math.round(ratioHeight * scale / 32) * 32;
  return Number(((width * height) / 1048576).toFixed(4));
}

// Preview of the tile grid the run-time planner picks for a VRAM preset at an output size: an equal
// rows x cols grid (max 9 x 9) scored on tile area (going over the target costs double) and tile shape.
export function miniMaxH3TilePlan(presetKey, megapixels, aspectRatio) {
  const preset = MINIMAX_H3_VRAM_PRESETS[normalizeMiniMaxH3VramPreset(presetKey)];
  const { width, height } = miniMaxH3FrameSize(megapixels, aspectRatio);
  const targetArea = preset.tileMegapixels * 1048576;
  let best = null;
  for (let rows = 1; rows <= 9; rows += 1) {
    for (let cols = 1; cols <= 9; cols += 1) {
      const tileWidth = width / cols;
      const tileHeight = height / rows;
      const areaError = Math.log((tileWidth * tileHeight) / targetArea);
      const shapeError = Math.log(tileWidth / tileHeight / (width / height));
      const cost = (areaError > 0 ? 2 * areaError : -areaError) + 0.5 * Math.abs(shapeError);
      if (!best || cost < best.cost - 1e-9) best = { rows, cols, cost };
    }
  }
  return { rows: best.rows, cols: best.cols, width, height, chunk: preset.chunk, overlap: preset.overlap };
}

export const MINIMAX_H3_CONTINUITY_OPTIONS = [
  { value: "off", label: "Off" },
  { value: "latent_continuation_masked", label: "Latent Continuation Masked (protected predecessor latent)" },
];

export const MINIMAX_H3_LOCATION_TRANSITION_OPTIONS = [
  { value: "normal", label: "Normal — physical route" },
  { value: "surreal", label: "Surreal transformation" },
  { value: "cinematic", label: "Cinematic conceal" },
  { value: "inner_world", label: "Inner world / portal" },
  { value: "match", label: "Match transition" },
  { value: "motion", label: "Motion transition" },
  { value: "creative_auto", label: "Creative auto" },
  { value: "masked", label: "Masked — continue, then one smooth move (Latent Continuation Masked)" },
  { value: "custom", label: "Custom" },
];

export const MINIMAX_H3_START_FRAME_CHARACTER_INFLUENCE_OPTIONS = [
  { value: "face_hair_only", label: "Face + hair only (keep the rest of the start frame)" },
  { value: "full_character", label: "Full character identity (face, hair, clothing, and body)" },
];

export const MINIMAX_H3_SCENE_IMAGE_USE_OPTIONS = [
  { value: "off", label: "Do not use scene image" },
  { value: "exact_start_frame", label: "Exact start frame (LLM + MiniMax)" },
  { value: "environment_inspiration", label: "Environment inspiration only (LLM only — ignore framing)" },
  { value: "environment_framing_inspiration", label: "Environment + framing inspiration (LLM only)" },
];

export const MINIMAX_H3_SAGE_ATTENTION_OPTIONS = [
  { value: "disabled", label: "Disabled" },
  { value: "auto", label: "Auto" },
  { value: "sageattn_qk_int8_pv_fp16_cuda", label: "SageAttention 2 — FP16 CUDA" },
  { value: "sageattn_qk_int8_pv_fp16_triton", label: "SageAttention 2 — FP16 Triton" },
  { value: "sageattn_qk_int8_pv_fp8_cuda", label: "SageAttention 2 — FP8 CUDA" },
  { value: "sageattn_qk_int8_pv_fp8_cuda++", label: "SageAttention 2 — FP8 CUDA++" },
  { value: "sageattn3", label: "SageAttention 3" },
  { value: "sageattn3_per_block_mean", label: "SageAttention 3 — Per-block mean" },
];

export function normalizeMiniMaxH3Mode(value) {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (["image_reference_to_video", "image_plus_reference_to_video", "i2v_r2v"].includes(clean)) return "image_reference_to_video";
  return MINIMAX_H3_MODE_OPTIONS.some((item) => item.value === clean) ? clean : "text_to_video";
}

export function normalizeMiniMaxH3Pipeline(value) {
  return String(value || "").trim().toLowerCase() === "refmod" ? "refmod" : "standard";
}

export function normalizeMiniMaxH3AudioMode(value) {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return ["built_in_audio", "native_audio", "generated_audio"].includes(clean) ? "built_in_audio" : "input_audio";
}

export function normalizeMiniMaxH3ContinuityMode(value) {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  // The standard and exact-last-frame latent modes were retired. Saved projects that still name them continue masked.
  if (["latent_exact", "latent_exact_frame", "latent_continuation_exact", "latent_continuation_exact_frame",
    "latent_masked", "latent_masked_av", "latent_continuation_masked",
    "latent", "latent_continuation", "continuation"].includes(clean)) return "latent_continuation_masked";
  // The previous-final-frame modes were retired too. Saved projects that still name them are off.
  return "off";
}

export function normalizeMiniMaxH3LocationTransitionPreset(value) {
  const clean = String(value || "normal").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return MINIMAX_H3_LOCATION_TRANSITION_OPTIONS.some((item) => item.value === clean) ? clean : "normal";
}

// Where a continued scene's own movement may begin: at least 0.5 s in, at most half of the scene. The author picks it
// per scene (segment.minimax_h3_continuation_start_seconds), 0.5 s when nothing is set.
// Python twin: continuation_start_limits / continuation_hold_seconds in minimax/prompt_assembly.py.
export const MINIMAX_H3_CONTINUATION_MIN_START_SECONDS = 0.5;

export function miniMaxH3ContinuationStartLimits(sceneSeconds) {
  const low = MINIMAX_H3_CONTINUATION_MIN_START_SECONDS;
  const high = Math.max(low, Math.floor(Math.max(0, Number(sceneSeconds) || 0) * 0.5 * 10 + 1e-9) / 10);
  return { low, high };
}

export function miniMaxH3ContinuationStartSeconds(sceneSeconds, requested) {
  const { low, high } = miniMaxH3ContinuationStartLimits(sceneSeconds);
  const number = requested === null || requested === undefined || requested === "" ? NaN : Number(requested);
  const value = Number.isFinite(number) ? number : low;
  return Math.round(Math.min(high, Math.max(low, value)) * 10) / 10;
}

export function isMiniMaxH3LatentContinuationMode(mode) {
  return mode === "latent_continuation_masked";
}

// I2V scenes use their own frames; between-scene continuation is unavailable in either image mode.
// Latent Continuation Masked works in the other MiniMax modes, and in every render pass
// except 2 Pass Advanced (render_pass "three_pass", which only applies to Reference to Video).
// Python twin: minimax.scene_inputs.continuity_allowed_for_mode.
export function isMiniMaxH3ContinuityAllowedForMode(continuityMode, videoMode, renderPass = "single") {
  const continuity = normalizeMiniMaxH3ContinuityMode(continuityMode);
  const mode = normalizeMiniMaxH3Mode(videoMode);
  if (continuity === "latent_continuation_masked") {
    return !["image_to_video", "image_reference_to_video"].includes(mode) && !(mode === "reference_to_video" && renderPass === "three_pass");
  }
  return continuity === "off";
}

export function normalizeMiniMaxH3StartFrameCharacterInfluence(value) {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return ["face_hair_only", "face_and_hair_only", "identity_face_hair_only"].includes(clean)
    ? "face_hair_only"
    : "full_character";
}

export function normalizeMiniMaxH3SceneImageUse(value, legacyExactStartFrame = false) {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (["exact", "exact_start", "exact_start_frame"].includes(clean)) return "exact_start_frame";
  if (["environment", "environment_only", "environment_inspiration", "inspiration"].includes(clean)) return "environment_inspiration";
  if (["environment_framing", "environment_framing_inspiration", "inspiration_with_framing"].includes(clean)) return "environment_framing_inspiration";
  return legacyExactStartFrame ? "exact_start_frame" : "off";
}

export function normalizeMiniMaxH3Voice(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const rawPreset = String(source.preset_id || source.presetId || source.preset || "none").trim().toLowerCase();
  const preset = MINIMAX_H3_VOICE_PRESETS.find((item) => item.value === rawPreset) || MINIMAX_H3_VOICE_PRESETS[0];
  const custom = preset.value.endsWith("_custom");
  return {
    preset_id: preset.value,
    gender: preset.gender,
    preset_name: custom ? String(source.preset_name || source.presetName || source.name || "").trim() : preset.name,
    description: custom ? String(source.description || source.voice_description || source.voiceDescription || "").trim() : preset.description,
  };
}

export function normalizeMiniMaxSpeakerAssignments(value = []) {
  const source = Array.isArray(value) ? value : [];
  return source
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || item.cue_id || item.cueId || `speaker_cue_${Date.now()}_${index}_${Math.floor(Math.random() * 10000)}`),
      type: String(item.type || item.kind || "").trim().toLowerCase() === "instrumental" ? "instrumental" : "dialogue",
      speaker_id: String(item.speaker_id || item.speakerId || item.subject_id || item.subjectId || ""),
      speaker_name: String(item.speaker_name || item.speakerName || item.speaker || item.character || "").trim(),
      text: String(item.text || item.dialogue || item.line || item.lyric || "").trim(),
      action_note: String(item.action_note || item.actionNote || item.note || "").trim(),
      start: Number.isFinite(Number(item.start)) ? Math.max(0, Number(item.start)) : null,
      end: Number.isFinite(Number(item.end)) ? Math.max(0, Number(item.end)) : null,
    }))
    .slice(0, 40);
}

export function miniMaxH3ModeLabel(value) {
  const mode = normalizeMiniMaxH3Mode(value);
  if (mode === "image_reference_to_video") return "Image to Video 2 Pass";
  return MINIMAX_H3_MODE_OPTIONS.find((item) => item.value === mode)?.label || "Text to Video";
}

export function miniMaxH3InstructionKey(value) {
  return MINIMAX_H3_INSTRUCTION_KEYS[normalizeMiniMaxH3Mode(value)] || MINIMAX_H3_INSTRUCTION_KEYS.text_to_video;
}

export function normalizeMiniMaxShortFilmPlanningMode(value = "") {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return clean === "fully_custom" || clean === "custom" ? "fully_custom" : "guided_film";
}

function miniMaxH3ShortFilmInstructionKey(value, planningMode = "guided_film") {
  const mode = normalizeMiniMaxH3Mode(value);
  const profile = normalizeMiniMaxShortFilmPlanningMode(planningMode) === "fully_custom" ? "custom" : "guided";
  return `minimax_h3_short_film_${profile}_${mode}`;
}

export function normalizeMiniMaxH3VideoPurpose(value) {
  const clean = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return MINIMAX_H3_VIDEO_REFERENCE_PURPOSES.some((item) => item.value === clean) ? clean : "continuation";
}

export function miniMaxInstalledPass2Lora(preset, installed, mode = "reference_to_video") {
  const candidates = Number(preset) === 8 ? [
    "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors",
  ] : [
    DEFAULT_MINIMAX_H3_SETTINGS.two_pass_lora_name,
    "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
    "minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors",
    "minimax_h3_fl2v_lightx2v_turbo_4step_v1.0_768p_resized_avg_rank_31_bf16.safetensors",
    "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
    "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors",
  ];
  for (const candidate of candidates) {
    if (mode === "image_to_video" && candidate.includes("ref2v")) continue;
    const match = installed.find((name) => name.split(/[\\/]/).pop().toLowerCase() === candidate.toLowerCase());
    if (match) return match;
  }
  return "";
}

export function selectMiniMaxH3PassSettings(settings, passMode) {
  const { ref_pass_profiles = {}, i2v_pass_profiles = {}, ref_pass_mode: _legacyPassMode, ...current } = settings;
  const isI2V = normalizeMiniMaxH3Mode(current.video_mode) === "image_to_video";
  const profiles = { ...(isI2V ? i2v_pass_profiles : ref_pass_profiles) };
  if (profiles.advanced && !profiles.three_pass) profiles.three_pass = profiles.advanced;
  delete profiles.advanced;
  profiles[current.render_pass || "single"] = current;
  return cloneMiniMaxH3Settings({
    ...current,
    ...(profiles[passMode] || {}),
    // The output resolution is shared by every pass mode, not stored per profile.
    i2v_transition_style: current.i2v_transition_style,
    i2v_transition_direction: current.i2v_transition_direction,
    aspect_ratio: current.aspect_ratio,
    resolution_preset: current.resolution_preset,
    megapixels: current.megapixels,
    video_mode: isI2V ? "image_to_video" : "reference_to_video",
    render_pass: isI2V && passMode === "three_pass" ? "two_pass" : passMode,
    i2v_pass_settings_version: 1,
    ref_pass_profiles: isI2V ? ref_pass_profiles : profiles,
    i2v_pass_profiles: isI2V ? profiles : i2v_pass_profiles,
  });
}

// One output resolution for every render pass. New saves carry resolution_preset; older saves took the
// resolution of the pass type they were rendering with (single megapixels, 2 Pass final size, or
// 2 Pass Advanced Pass 2 megapixels).
const MINIMAX_H3_RESOLUTION_PRESET_KEYS = ["custom", "1k", "2k", "1440p", "4k"];
function resolveMiniMaxH3Resolution(source, renderPass) {
  const aspect = String(source.aspect_ratio || DEFAULT_MINIMAX_H3_SETTINGS.aspect_ratio);
  const savedPreset = String(source.resolution_preset || "").trim().toLowerCase();
  const clamp = (value) => Math.max(0.1, Math.min(16, Number(value) || DEFAULT_MINIMAX_H3_SETTINGS.megapixels));
  if (MINIMAX_H3_RESOLUTION_PRESET_KEYS.includes(savedPreset)) {
    const presetMegapixels = miniMaxH3PresetMegapixels(savedPreset, aspect);
    return { resolution_preset: savedPreset, megapixels: presetMegapixels ?? clamp(source.megapixels ?? DEFAULT_MINIMAX_H3_SETTINGS.megapixels) };
  }
  const hasLegacyResolution = source.megapixels != null
    || (Number(source.two_pass_final_width) > 0 && Number(source.two_pass_final_height) > 0)
    || source.advanced_two_pass_pass2_megapixels != null
    || source.advanced_two_pass_pass2_resolution_preset != null;
  if (!hasLegacyResolution) {
    // A new project: use the default preset.
    const defaultPreset = DEFAULT_MINIMAX_H3_SETTINGS.resolution_preset;
    return { resolution_preset: defaultPreset, megapixels: miniMaxH3PresetMegapixels(defaultPreset, aspect) ?? DEFAULT_MINIMAX_H3_SETTINGS.megapixels };
  }
  let preset = "custom";
  let megapixels;
  if (renderPass === "three_pass") {
    const legacyPreset = String(source.advanced_two_pass_pass2_resolution_preset || "").trim().toLowerCase();
    preset = MINIMAX_H3_RESOLUTION_PRESET_KEYS.includes(legacyPreset) ? legacyPreset : "custom";
    megapixels = miniMaxH3PresetMegapixels(preset, aspect) ?? clamp(source.advanced_two_pass_pass2_megapixels ?? 2);
  } else if (renderPass === "two_pass" && Number(source.two_pass_final_width) > 0 && Number(source.two_pass_final_height) > 0) {
    megapixels = clamp((Number(source.two_pass_final_width) * Number(source.two_pass_final_height)) / 1048576);
  } else {
    megapixels = clamp(source.megapixels ?? 0.9);
  }
  if (preset === "custom") {
    // Keep the preset label when the migrated size is exactly a preset's frame size.
    const size = miniMaxH3FrameSize(megapixels, aspect);
    const matching = MINIMAX_H3_RESOLUTION_PRESET_KEYS.find((key) => {
      const presetMp = miniMaxH3PresetMegapixels(key, aspect);
      if (presetMp == null) return false;
      const presetSize = miniMaxH3FrameSize(presetMp, aspect);
      return presetSize.width === size.width && presetSize.height === size.height;
    });
    if (matching) return { resolution_preset: matching, megapixels: miniMaxH3PresetMegapixels(matching, aspect) };
  }
  return { resolution_preset: preset, megapixels: Number(megapixels.toFixed(4)) };
}

export function cloneMiniMaxH3Settings(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const hasCurrentTwoPassDefaults = Number(source.two_pass_defaults_version || 0) >= 1;
  const hasCurrentAdvancedTwoPassDefaults = Number(source.advanced_two_pass_defaults_version || 0) >= 4;
  let renderPass = ["single", "two_pass", "three_pass"].includes(String(source.render_pass || "").trim().toLowerCase())
    ? String(source.render_pass).trim().toLowerCase()
    : ["single", "two_pass", "advanced"].includes(String(source.ref_pass_mode || "").trim().toLowerCase())
      ? (source.ref_pass_mode === "advanced" ? "three_pass" : source.ref_pass_mode)
      : source.render_pass == null && normalizeMiniMaxH3Mode(source.video_mode || source.mode) === "image_reference_to_video"
        ? "two_pass"
        : DEFAULT_MINIMAX_H3_SETTINGS.render_pass;
  const pipeline = normalizeMiniMaxH3Pipeline(source.pipeline);
  const isI2V = pipeline !== "refmod" && normalizeMiniMaxH3Mode(source.video_mode || source.mode) === "image_to_video";
  // I2V previously ignored render_pass, even when a save carried the global two-pass default.
  if (isI2V && Number(source.i2v_pass_settings_version || 0) < 1) renderPass = "single";
  if (isI2V && renderPass === "three_pass") renderPass = "two_pass";
  const resolution = resolveMiniMaxH3Resolution(source, renderPass);
  const sourceLoras = Array.isArray(source.loras)
    ? source.loras
    : Array.from({ length: 4 }, (_, index) => ({
      name: source[`lora_${index + 1}`],
      strength: source[`lora_${index + 1}_strength`],
    }));
  const loras = sourceLoras
    .map((item) => ({
      name: String(item?.name || item?.lora_name || item?.loraName || "").trim(),
      strength: Math.max(-10, Math.min(10, Number(item?.strength ?? item?.strength_model ?? item?.strengthModel ?? 1))),
      // A LoRA with no saved target goes on the first pass.
      apply_to: ["both", "pass1", "pass2"].includes(String(item?.apply_to || item?.applyTo || "pass1"))
        ? String(item?.apply_to || item?.applyTo || "pass1")
        : "pass1",
    }))
    .filter((item) => item.name && item.name !== "[none]")
    .slice(0, 4);
  const turboEnabled = normalizeMiniMaxH3Mode(source.video_mode || source.mode || DEFAULT_MINIMAX_H3_SETTINGS.video_mode) !== "reference_to_video" && Boolean(source.use_turbo_lora ?? source.useTurboLora ?? DEFAULT_MINIMAX_H3_SETTINGS.use_turbo_lora);
  const loraEnabled = Boolean(source.use_loras ?? source.useLoras ?? source.use_custom_loras ?? source.useCustomLoras ?? DEFAULT_MINIMAX_H3_SETTINGS.use_loras) && !turboEnabled;
  const rawSteps = Math.max(1, Math.min(1000, Math.trunc(Number(source.steps ?? DEFAULT_MINIMAX_H3_SETTINGS.steps) || DEFAULT_MINIMAX_H3_SETTINGS.steps)));
  const hasSavedPreTurboSteps = source.steps_before_turbo != null || source.stepsBeforeTurbo != null;
  const migrateOldTurboDefault = turboEnabled
    && !hasSavedPreTurboSteps
    && rawSteps === DEFAULT_MINIMAX_H3_SETTINGS.steps;
  const hasSavedPreTurboEasyCache = source.easy_cache_bypass_before_turbo != null || source.easyCacheBypassBeforeTurbo != null;
  const rawEasyCacheBypass = Boolean(source.easy_cache_bypass ?? DEFAULT_MINIMAX_H3_SETTINGS.easy_cache_bypass);
  const continuityMode = normalizeMiniMaxH3ContinuityMode(source.continuity_mode || source.continuityMode || DEFAULT_MINIMAX_H3_SETTINGS.continuity_mode);
  const videoMode = pipeline === "refmod" ? "reference_to_video" : normalizeMiniMaxH3Mode(source.video_mode || source.mode || DEFAULT_MINIMAX_H3_SETTINGS.video_mode);
  // Masked continuation plans its own head and tail; extra render frames stay at zero.
  const maskedContinuation = continuityMode === "latent_continuation_masked"
    && isMiniMaxH3ContinuityAllowedForMode(continuityMode, videoMode,
      pipeline === "refmod" && renderPass === "three_pass" ? "two_pass" : renderPass);
  return {
    ...DEFAULT_MINIMAX_H3_SETTINGS,
    ...source,
    pipeline,
    // The RefMod pipeline only has one mode.
    video_mode: videoMode,
    // Pre-existing saves never had render_pass; Image + Reference 2 Pass was always a two-pass
    // workflow by mode alone, so an absent field must still mean two_pass for that mode.
    render_pass: pipeline === "refmod" && renderPass === "three_pass" ? "two_pass" : renderPass,
    i2v_transition_style: normalizeMiniMaxI2VTransitionStyle(source.i2v_transition_style),
    i2v_transition_direction: String(source.i2v_transition_direction || "").trim(),
    audio_mode: normalizeMiniMaxH3AudioMode(source.audio_mode || source.audioMode || DEFAULT_MINIMAX_H3_SETTINGS.audio_mode),
    continuity_mode: continuityMode,
    continuity_prompt_from_last_frame: Boolean(source.continuity_prompt_from_last_frame ?? source.continuityPromptFromLastFrame ?? DEFAULT_MINIMAX_H3_SETTINGS.continuity_prompt_from_last_frame),
    location_transition_preset: normalizeMiniMaxH3LocationTransitionPreset(source.location_transition_preset ?? source.locationTransitionPreset ?? DEFAULT_MINIMAX_H3_SETTINGS.location_transition_preset),
    location_transition_custom: String(source.location_transition_custom ?? source.locationTransitionCustom ?? DEFAULT_MINIMAX_H3_SETTINGS.location_transition_custom).trim(),
    latent_context_frames: [39, 90, 141, 192].includes(Number(source.latent_context_frames ?? source.latentContextFrames))
      ? Number(source.latent_context_frames ?? source.latentContextFrames)
      : DEFAULT_MINIMAX_H3_SETTINGS.latent_context_frames,
    diffusion_model_name: isI2V && (!source.diffusion_model_name || source.diffusion_model_name === DEFAULT_MINIMAX_H3_SETTINGS.diffusion_model_name)
      ? "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
      : String(source.diffusion_model_name || DEFAULT_MINIMAX_H3_SETTINGS.diffusion_model_name),
    two_pass_lora_name: isI2V && (!source.two_pass_lora_name || source.two_pass_lora_name === DEFAULT_MINIMAX_H3_SETTINGS.two_pass_lora_name)
      ? "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors"
      : String(source.two_pass_lora_name || DEFAULT_MINIMAX_H3_SETTINGS.two_pass_lora_name),
    i2v_pass_settings_version: 1,
    i2v_pass_profiles: source.i2v_pass_profiles && typeof source.i2v_pass_profiles === "object" && !Array.isArray(source.i2v_pass_profiles)
      ? source.i2v_pass_profiles : {},
    clip_name: String(source.clip_name || DEFAULT_MINIMAX_H3_SETTINGS.clip_name),
    video_vae_name: String(source.video_vae_name || DEFAULT_MINIMAX_H3_SETTINGS.video_vae_name),
    audio_vae_name: String(source.audio_vae_name || DEFAULT_MINIMAX_H3_SETTINGS.audio_vae_name),
    aspect_ratio: String(source.aspect_ratio || DEFAULT_MINIMAX_H3_SETTINGS.aspect_ratio),
    resolution_preset: resolution.resolution_preset,
    megapixels: resolution.megapixels,
    seed: Number.isFinite(Number(source.seed)) ? Number(source.seed) : DEFAULT_MINIMAX_H3_SETTINGS.seed,
    warmup_frames: maskedContinuation ? 0 : Math.max(0, Math.trunc(Number(source.warmup_frames || 0))),
    cooldown_frames: maskedContinuation ? 0 : Math.max(0, Math.trunc(Number(source.cooldown_frames || 0))),
    sampler_name: String(source.sampler_name || DEFAULT_MINIMAX_H3_SETTINGS.sampler_name),
    scheduler: String(source.scheduler || DEFAULT_MINIMAX_H3_SETTINGS.scheduler),
    steps: migrateOldTurboDefault ? 4 : rawSteps,
    steps_before_turbo: Math.max(1, Math.min(1000, Math.trunc(Number(source.steps_before_turbo ?? source.stepsBeforeTurbo ?? rawSteps) || DEFAULT_MINIMAX_H3_SETTINGS.steps_before_turbo))),
    denoise: Math.max(0, Math.min(1, Number(source.denoise ?? DEFAULT_MINIMAX_H3_SETTINGS.denoise))),
    easy_cache_bypass: turboEnabled && !hasSavedPreTurboEasyCache ? true : rawEasyCacheBypass,
    easy_cache_bypass_before_turbo: Boolean(source.easy_cache_bypass_before_turbo ?? source.easyCacheBypassBeforeTurbo ?? rawEasyCacheBypass),
    easy_cache_reuse_threshold: Math.max(0, Math.min(1, Number(source.easy_cache_reuse_threshold ?? DEFAULT_MINIMAX_H3_SETTINGS.easy_cache_reuse_threshold))),
    easy_cache_start_percent: Math.max(0, Math.min(1, Number(source.easy_cache_start_percent ?? DEFAULT_MINIMAX_H3_SETTINGS.easy_cache_start_percent))),
    easy_cache_end_percent: Math.max(0, Math.min(1, Number(source.easy_cache_end_percent ?? DEFAULT_MINIMAX_H3_SETTINGS.easy_cache_end_percent))),
    easy_cache_verbose: Boolean(source.easy_cache_verbose ?? DEFAULT_MINIMAX_H3_SETTINGS.easy_cache_verbose),
    sage_attention: MINIMAX_H3_SAGE_ATTENTION_OPTIONS.some((item) => item.value === source.sage_attention)
      ? source.sage_attention
      : DEFAULT_MINIMAX_H3_SETTINGS.sage_attention,
    use_memory_efficient_sage_attention: Boolean(source.use_memory_efficient_sage_attention ?? DEFAULT_MINIMAX_H3_SETTINGS.use_memory_efficient_sage_attention),
    enable_fp16_accumulation: Boolean(source.enable_fp16_accumulation ?? DEFAULT_MINIMAX_H3_SETTINGS.enable_fp16_accumulation),
    use_loras: loraEnabled,
    lora_count: loraEnabled ? Math.max(0, Math.min(4, Math.trunc(Number(source.lora_count ?? source.loraCount ?? loras.length) || loras.length))) : 0,
    loras,
    use_turbo_lora: turboEnabled,
    turbo_lora_name: String(source.turbo_lora_name || source.turboLoraName || DEFAULT_MINIMAX_H3_SETTINGS.turbo_lora_name),
    turbo_lora_strength: Math.max(-10, Math.min(10, Number(source.turbo_lora_strength ?? source.turboLoraStrength ?? DEFAULT_MINIMAX_H3_SETTINGS.turbo_lora_strength))),
    ref_image_size: ["match", "max"].includes(String(source.ref_image_size || "").trim().toLowerCase())
      ? String(source.ref_image_size).trim().toLowerCase()
      : DEFAULT_MINIMAX_H3_SETTINGS.ref_image_size,
    two_pass_defaults_version: DEFAULT_MINIMAX_H3_SETTINGS.two_pass_defaults_version,
    two_pass_latent_upscale_scale: Math.max(1, Math.min(8, Number(hasCurrentTwoPassDefaults
      ? (source.two_pass_latent_upscale_scale ?? DEFAULT_MINIMAX_H3_SETTINGS.two_pass_latent_upscale_scale)
      : DEFAULT_MINIMAX_H3_SETTINGS.two_pass_latent_upscale_scale))),
    two_pass_latent_upscaler_name: String(source.two_pass_latent_upscaler_name || DEFAULT_MINIMAX_H3_SETTINGS.two_pass_latent_upscaler_name),
    two_pass_lora_preset: Number(source.two_pass_lora_preset) === 8 ? 8 : 4,
    two_pass_use_te_speed: Boolean(source.two_pass_use_te_speed ?? DEFAULT_MINIMAX_H3_SETTINGS.two_pass_use_te_speed),
    use_feedforward: Boolean(source.use_feedforward ?? DEFAULT_MINIMAX_H3_SETTINGS.use_feedforward),
    two_pass_use_feedforward: Boolean(source.two_pass_use_feedforward ?? DEFAULT_MINIMAX_H3_SETTINGS.two_pass_use_feedforward),
    use_block_sparse_attention: Boolean(source.use_block_sparse_attention ?? DEFAULT_MINIMAX_H3_SETTINGS.use_block_sparse_attention),
    two_pass_use_block_sparse_attention: Boolean(source.two_pass_use_block_sparse_attention ?? DEFAULT_MINIMAX_H3_SETTINGS.two_pass_use_block_sparse_attention),
    two_pass_use_fast_vae_decode: Boolean(source.two_pass_use_fast_vae_decode ?? DEFAULT_MINIMAX_H3_SETTINGS.two_pass_use_fast_vae_decode),
    two_pass_te_speed_processing_control: Math.max(0, Math.min(1, Number(source.two_pass_te_speed_processing_control ?? DEFAULT_MINIMAX_H3_SETTINGS.two_pass_te_speed_processing_control))),
    two_pass_te_speed_start_percent: Math.max(0, Math.min(1, Number(source.two_pass_te_speed_start_percent ?? DEFAULT_MINIMAX_H3_SETTINGS.two_pass_te_speed_start_percent))),
    two_pass_te_speed_end_percent: Math.max(0, Math.min(1, Number(source.two_pass_te_speed_end_percent ?? DEFAULT_MINIMAX_H3_SETTINGS.two_pass_te_speed_end_percent))),
    two_pass_te_speed_mcs: Math.max(1, Math.min(64, Math.trunc(Number(source.two_pass_te_speed_mcs ?? DEFAULT_MINIMAX_H3_SETTINGS.two_pass_te_speed_mcs)))),
    two_pass_te_speed_cache_depth: Math.max(0, Math.min(1, Number(source.two_pass_te_speed_cache_depth ?? DEFAULT_MINIMAX_H3_SETTINGS.two_pass_te_speed_cache_depth))),
    two_pass_te_speed_device: String(source.two_pass_te_speed_device || DEFAULT_MINIMAX_H3_SETTINGS.two_pass_te_speed_device),
    two_pass_final_resize_method: String(source.two_pass_final_resize_method || DEFAULT_MINIMAX_H3_SETTINGS.two_pass_final_resize_method),
    two_pass_output_crf: Math.max(0, Math.min(100, Math.trunc(Number(source.two_pass_output_crf ?? DEFAULT_MINIMAX_H3_SETTINGS.two_pass_output_crf)))),
    advanced_two_pass_vram_preset: normalizeMiniMaxH3VramPreset(source.advanced_two_pass_vram_preset),
    advanced_two_pass_pass1_resolution_preset: ["custom", "1k", "2k", "1440p", "4k"].includes(String(source.advanced_two_pass_pass1_resolution_preset || "").toLowerCase())
      ? String(source.advanced_two_pass_pass1_resolution_preset).toLowerCase()
      : DEFAULT_MINIMAX_H3_SETTINGS.advanced_two_pass_pass1_resolution_preset,
    advanced_two_pass_defaults_version: DEFAULT_MINIMAX_H3_SETTINGS.advanced_two_pass_defaults_version,
    ...Object.fromEntries([1, 2].flatMap((pass) => {
      const prefix = `advanced_two_pass_pass${pass}_`;
      const defaults = DEFAULT_MINIMAX_H3_SETTINGS;
      return [
        ...(pass === 1 ? [[`${prefix}megapixels`, Math.max(0.1, Math.min(16, Number(source[`${prefix}megapixels`] ?? defaults[`${prefix}megapixels`])))]] : []),
        [`${prefix}steps`, Math.max(1, Math.min(1000, Math.trunc(Number(source[`${prefix}steps`] ?? defaults[`${prefix}steps`]) || defaults[`${prefix}steps`])))],
        [`${prefix}denoise`, Math.max(0, Math.min(1, Number(source[`${prefix}denoise`] ?? defaults[`${prefix}denoise`])))],
        [`${prefix}sampler`, String(source[`${prefix}sampler`] || defaults[`${prefix}sampler`])],
        [`${prefix}scheduler`, String(source[`${prefix}scheduler`] || defaults[`${prefix}scheduler`])],
        [`${prefix}seed`, Number.isFinite(Number(source[`${prefix}seed`])) ? Number(source[`${prefix}seed`]) : defaults[`${prefix}seed`]],
      ];
    })),
    ...Object.fromEntries([1, 2].flatMap((pass) => {
      const prefix = `two_pass_pass${pass}_`;
      const defaults = DEFAULT_MINIMAX_H3_SETTINGS;
      return [
        [`${prefix}steps`, Math.max(1, Math.min(1000, Math.trunc(Number(hasCurrentTwoPassDefaults ? source[`${prefix}steps`] : defaults[`${prefix}steps`]) || defaults[`${prefix}steps`])))],
        [`${prefix}denoise`, Math.max(0, Math.min(1, Number(hasCurrentTwoPassDefaults ? (source[`${prefix}denoise`] ?? defaults[`${prefix}denoise`]) : defaults[`${prefix}denoise`])))],
        [`${prefix}sampler`, String(hasCurrentTwoPassDefaults ? (source[`${prefix}sampler`] || defaults[`${prefix}sampler`]) : defaults[`${prefix}sampler`])],
        [`${prefix}scheduler`, String(hasCurrentTwoPassDefaults ? (source[`${prefix}scheduler`] || defaults[`${prefix}scheduler`]) : defaults[`${prefix}scheduler`])],
        [`${prefix}seed`, hasCurrentTwoPassDefaults && Number.isFinite(Number(source[`${prefix}seed`])) ? Number(source[`${prefix}seed`]) : defaults[`${prefix}seed`]],
      ];
    })),
    ...Object.fromEntries([1, 2, 3].flatMap((pass) => {
      const prefix = `three_pass_pass${pass}_`;
      const defaults = DEFAULT_MINIMAX_H3_SETTINGS;
      return [
        [`${prefix}megapixels`, Math.max(0.1, Number(hasCurrentAdvancedTwoPassDefaults ? (source[`${prefix}megapixels`] ?? defaults[`${prefix}megapixels`]) : defaults[`${prefix}megapixels`]))],
        [`${prefix}steps`, Math.max(1, Math.min(1000, Math.trunc(Number(hasCurrentAdvancedTwoPassDefaults ? (source[`${prefix}steps`] ?? defaults[`${prefix}steps`]) : defaults[`${prefix}steps`]) || defaults[`${prefix}steps`])))],
        [`${prefix}denoise`, Math.max(0, Math.min(1, Number(hasCurrentAdvancedTwoPassDefaults ? (source[`${prefix}denoise`] ?? defaults[`${prefix}denoise`]) : defaults[`${prefix}denoise`])) )],
        [`${prefix}sampler`, String(hasCurrentAdvancedTwoPassDefaults ? (source[`${prefix}sampler`] || defaults[`${prefix}sampler`]) : defaults[`${prefix}sampler`])],
        [`${prefix}scheduler`, String(hasCurrentAdvancedTwoPassDefaults ? (source[`${prefix}scheduler`] || defaults[`${prefix}scheduler`]) : defaults[`${prefix}scheduler`])],
        [`${prefix}seed`, hasCurrentAdvancedTwoPassDefaults && Number.isFinite(Number(source[`${prefix}seed`])) ? Number(source[`${prefix}seed`]) : defaults[`${prefix}seed`]],
        [`${prefix}te_speed`, Boolean(hasCurrentAdvancedTwoPassDefaults ? (source[`${prefix}te_speed`] ?? defaults[`${prefix}te_speed`]) : defaults[`${prefix}te_speed`])],
      ];
    })),
  };
}
