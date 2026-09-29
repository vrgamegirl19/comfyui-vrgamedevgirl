import { BROWSER_IMAGE_PROVIDERS } from "../VRGDG_BrowserImageBridge.js";
import { storyboardCutFrequencyValue } from "../storyboard_builder/scenes.mjs";
import { normalizeStoryboardCustomCameraFlowSequence } from "../storyboard_builder/shot_presets.mjs";
import { getJson, normalizeOwnServerTimeoutMinutes } from "./comfy_api.mjs";
import {
  BAD_I2V_UNET_ALIASES,
  DEFAULT_I2V_DIFFUSION_MODEL,
  DEFAULT_I2V_PASS1_SIGMAS,
  DEFAULT_I2V_PASS2_SIGMAS,
  DEFAULT_I2V_UNET,
  DEFAULT_INGREDIENTS_SAMPLER,
  DEFAULT_LTX_INGREDIENTS_HEIGHT,
  DEFAULT_LTX_INGREDIENTS_WIDTH,
  DEFAULT_NB_IMAGE_MODEL,
  REQUIRED_LTX_ID_LORA,
  REQUIRED_LTX_INGREDIENTS_LORA,
  REQUIRED_LTX25_MSR_LORA,
} from "./constants.mjs";
import { normalizeProjectVideoEngine, normalizeVideoType } from "./controls.mjs";
import { defaultNBImageSettings } from "./llm_runner.mjs";
import { normalizeMiniMaxShortFilmPlanningMode } from "./minimax_h3.mjs";
import { cloneKrea2ReferenceSettings } from "./models.mjs";
import {
  normalizeGemmaContextLimit,
  normalizeGemmaGpuLayers,
  normalizeLmStudioContextLimit,
  normalizeOutputTokenLimit,
} from "./prompt_text.mjs";

export function defaultZImageSettings() {
  return {
    unet_name: "z_image_turbo_bf16.safetensors",
    clip_name: "qwen_3_4b.safetensors",
    vae_name: "ae.safetensors",
    first_pass_width: 1280,
    first_pass_height: 720,
    second_pass_width: 1920,
    second_pass_height: 1080,
    seed: 1,
    seed_mode: "fixed",
    batch_size: 1,
    use_loras: false,
    lora_count: 0,
    loras: [],
    use_image_to_image: false,
    image_to_image_start_at_step: 5,
    image_to_image_path: "",
    image_to_image_data: "",
    image_to_image_name: "",
    image_trigger_phrase: "",
  };
}

export function defaultFluxKleinSettings() {
  return {
    enabled: false,
    image_model_mode: "",
    use_text_only_gemma_prompt: false,
    use_director_notes: false,
    unet_name: "flux\\flux-2-klein-4b-fp8.safetensors",
    clip_name: "qwen_3_4b.safetensors",
    vae_name: "flux\\flux2-vae.safetensors",
    width: 1024,
    height: 576,
    seed: 100,
    use_loras: false,
    lora_count: 0,
    loras: [],
    image_trigger_phrase: "",
  };
}

export function defaultFlowGptBrowserSettings() {
  return {
    provider: BROWSER_IMAGE_PROVIDERS.FLOW_NANO_BANANA,
    aspect_ratio: "16:9",
    timeout_seconds: 600,
    flow_timeout_seconds: 420,
    gpt_timeout_seconds: 600,
    meta_timeout_seconds: 600,
    max_retries: 10,
    failure_mode: "last_successful_image",
    ask_previous_scene_image: false,
    manual_chat_prompt: "Using the character and location reference images, create 5 new images. Place the character naturally within the location in different areas, poses, and compositions. Vary the camera position and angle for each image. Integrate the character into the environment instead of simply pasting the character onto the scene. Make them look like stills from a cinematic music video. 16:9 aspect ratio.",
    browser_ai_prompt_character_count: 0,
    browser_ai_prompt_sequence_key: "",
    auto_advance_reference_group: true,
    band_sequence_enabled: false,
    band_sequence_singer_references: [],
    band_sequence_extra_references: [],
    band_sequence_member_references: [],
    band_sequence_location_references: [],
    band_sequence_location_index: 0,
    band_sequence_set_index: 0,
    reference_groups: [{ id: "group_1", name: "Group 1", images: [] }],
    active_reference_group_id: "group_1",
    location_reference: null,
    setup_note_collapsed: false,
  };
}

export function normalizeFlowGptBrowserProvider(provider) {
  const value = String(provider || "").trim();
  if (value === BROWSER_IMAGE_PROVIDERS.GPT_IMAGE) return BROWSER_IMAGE_PROVIDERS.GPT_IMAGE;
  if (value === BROWSER_IMAGE_PROVIDERS.META_AI) return BROWSER_IMAGE_PROVIDERS.META_AI;
  return BROWSER_IMAGE_PROVIDERS.FLOW_NANO_BANANA;
}

export function browserImageProviderLabel(provider) {
  const normalized = normalizeFlowGptBrowserProvider(provider);
  if (normalized === BROWSER_IMAGE_PROVIDERS.GPT_IMAGE) return "GPT Image";
  if (normalized === BROWSER_IMAGE_PROVIDERS.META_AI) return "Meta AI";
  return "Flow Nano Banana";
}

export function browserImageProviderShortLabel(provider) {
  const normalized = normalizeFlowGptBrowserProvider(provider);
  if (normalized === BROWSER_IMAGE_PROVIDERS.GPT_IMAGE) return "GPT Image";
  if (normalized === BROWSER_IMAGE_PROVIDERS.META_AI) return "Meta AI";
  return "Flow";
}

export function browserImageProviderDebugPort(provider) {
  const normalized = normalizeFlowGptBrowserProvider(provider);
  if (normalized === BROWSER_IMAGE_PROVIDERS.GPT_IMAGE) return 9223;
  if (normalized === BROWSER_IMAGE_PROVIDERS.META_AI) return 9224;
  return 9222;
}

export function browserImageProviderTimeout(settings = {}) {
  const normalized = normalizeFlowGptBrowserProvider(settings.provider);
  if (normalized === BROWSER_IMAGE_PROVIDERS.GPT_IMAGE) return settings.gpt_timeout_seconds || settings.timeout_seconds || 600;
  if (normalized === BROWSER_IMAGE_PROVIDERS.META_AI) return settings.meta_timeout_seconds || settings.timeout_seconds || 600;
  return settings.flow_timeout_seconds || settings.timeout_seconds || 420;
}

export function browserImageLoginStatus(provider, settings = {}) {
  const normalized = normalizeFlowGptBrowserProvider(provider);
  if (normalized === BROWSER_IMAGE_PROVIDERS.GPT_IMAGE) {
    return `GPT Image login browser opened.\nGPT prompts will append aspect ratio: ${settings.aspect_ratio || "16:9"}.`;
  }
  if (normalized === BROWSER_IMAGE_PROVIDERS.META_AI) {
    return "Meta AI login browser opened.\nLog into meta.ai first; Meta may look prompt-ready while still requiring login on submit.";
  }
  return `Flow login browser opened.\nSet Flow to 1 image and choose your aspect ratio before batch runs.\nRetries: ${settings.max_retries || 10}`;
}

export function defaultErnieImageSettings() {
  return {
    unet_name: "ernie\\ernie-image-turbo.safetensors",
    clip_name: "ministral-3-3b.safetensors",
    vae_name: "flux\\flux2-vae.safetensors",
    width: 1280,
    height: 720,
    seed: 1,
    seed_mode: "fixed",
    batch_size: 1,
    use_loras: false,
    lora_count: 0,
    loras: [],
    use_image_to_image: false,
    image_to_image_start_at_step: 5,
    image_to_image_path: "",
    image_to_image_data: "",
    image_to_image_name: "",
    image_trigger_phrase: "",
  };
}

export function defaultKrea2TwoPassSettings() {
  return {
    unet_name: "krea2_turbo_fp8_scaled.safetensors",
    clip_name: "qwen3vl_4b_fp8_scaled.safetensors",
    vae_name: "qwen_image_vae.safetensors",
    use_loras: false,
    lora_count: 0,
    loras: [],
    aspect_ratio: "16:9 (Widescreen)",
    sampler_name: "euler_ancestral_cfg_pp",
    cfg: 1.2,
    seed: 1,
    seed_mode: "fixed",
    batch_size: 1,
    use_image_to_image: false,
    image_to_image_creativity: 5,
    image_to_image_path: "",
    image_to_image_data: "",
    image_to_image_name: "",
    image_trigger_phrase: "",
  };
}

export function defaultZEnhanceSettings() {
  return {
    unet_name: "z_image_turbo_bf16.safetensors",
    clip_name: "qwen_3_4b.safetensors",
    vae_name: "ae.safetensors",
    width: 1920,
    height: 1080,
    seed: 1,
    seed_mode: "randomize",
    enhance_amount: 8,
    use_loras: false,
    lora_count: 0,
    loras: [],
    video_trigger_phrase: "",
  };
}

export function defaultI2VVideoSettings() {
  return {
    ltx_version: "2.5",
    use_gguf_model: false,
    unet_name: DEFAULT_I2V_UNET,
    diffusion_model_name: "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    use_sage_attention: false,
    enable_fp16_accumulation: false,
    vae_name: "ltx-2.5-video-vae-conv-bf16.safetensors",
    clip_name1: "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    clip_name2: "ltx-2.3_text_projection_bf16.safetensors",
    upscale_model_name: "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    audio_vae_name: "ltx-2.5-audio-vae-bf16.safetensors",
    fps: 24,
    width: 1920,
    height: 1080,
    resolution_aspect_ratio: "16:9 (Widescreen)",
    resolution_megapixels: 1.2,
    seed: 69,
    tail_loss_frames: 25,
    pre_frames: 50,
    flf_pre_frames: 0,
    flf_first_guide_strength: 0.7,
    flf_last_guide_strength: 0.7,
    flf_first_guide_frame_idx: 0,
    flf_last_guide_frame_idx: -1,
    flf_first_guide_crf: 29,
    flf_last_guide_crf: 29,
    flf_first_guide_blur_radius: 1,
    flf_last_guide_blur_radius: 1,
    flf_first_guide_interpolation: "lanczos",
    flf_last_guide_interpolation: "lanczos",
    flf_first_guide_crop: "center",
    flf_last_guide_crop: "center",
    flf_first_attention_strength: 0.9,
    flf_last_attention_strength: 1,
    flf_chain_previous_end_frame: true,
    flf_render_chain_start_source: "rendered_frame",
    flf_pregenerate_prompts_from_scene_images: false,
    flf_match_previous_clip_color: false,
    flf_color_match_strength: 0.85,
    flf_color_match_fade_seconds: 1.0,
    flf_global_transition_type: "auto",
    flf_gemma_context_mode: "images_story",
    msr_lora_name: REQUIRED_LTX25_MSR_LORA,
    msr_first_pass_strength: 1,
    msr_second_pass_strength: 0,
    msr_reference_strength: "auto - based on subject count",
    msr_background_mode: "no background reference",
    ingredients_lora_name: REQUIRED_LTX_INGREDIENTS_LORA,
    ingredients_first_pass_strength: 1,
    ingredients_second_pass_strength: 0,
    ingredients_width: DEFAULT_LTX_INGREDIENTS_WIDTH,
    ingredients_height: DEFAULT_LTX_INGREDIENTS_HEIGHT,
    id_lora_name: REQUIRED_LTX_ID_LORA,
    id_lora_first_pass_strength: 1,
    id_lora_second_pass_strength: 1,
    id_lora_reference_audio_path: "",
    identity_guidance_scale: 3,
    identity_start_percent: 0,
    identity_end_percent: 1,
    id_lora_duration: 5,
    use_loras: false,
    lora_count: 0,
    loras: [],
    pass1_sampler_name: "euler_ancestral",
    pass1_sigmas: DEFAULT_I2V_PASS1_SIGMAS,
    pass1_inplace_strength: 1,
    pass1_inplace_bypass: false,
    pass2_sampler_name: "euler_ancestral",
    pass2_sigmas: DEFAULT_I2V_PASS2_SIGMAS,
    pass2_inplace_strength: 1,
    pass2_inplace_bypass: false,
    t2v_pass1_sampler_name: "euler_ancestral",
    t2v_pass1_sigmas: DEFAULT_I2V_PASS1_SIGMAS,
    t2v_pass2_sampler_name: "euler_ancestral",
    t2v_pass2_sigmas: DEFAULT_I2V_PASS2_SIGMAS,
    rtv_pass1_sampler_name: "euler_ancestral",
    rtv_pass1_sigmas: DEFAULT_I2V_PASS1_SIGMAS,
    rtv_pass2_sampler_name: "euler_ancestral",
    rtv_pass2_sigmas: DEFAULT_I2V_PASS2_SIGMAS,
    ingredients_pass1_sampler_name: DEFAULT_INGREDIENTS_SAMPLER,
    ingredients_pass1_sigmas: DEFAULT_I2V_PASS1_SIGMAS,
    ingredients_pass2_sampler_name: DEFAULT_INGREDIENTS_SAMPLER,
    ingredients_pass2_sigmas: DEFAULT_I2V_PASS2_SIGMAS,
  };
}

export function repairI2VVideoSettingDimensions(settings = {}) {
  const repaired = settings && typeof settings === "object" ? settings : {};
  const regularWidth = Number(repaired.width || 1920);
  const regularHeight = Number(repaired.height || 1080);
  if (regularWidth === DEFAULT_LTX_INGREDIENTS_WIDTH && regularHeight === DEFAULT_LTX_INGREDIENTS_HEIGHT) {
    repaired.width = 1920;
    repaired.height = 1080;
  }
  const ingredientsWidth = Number(repaired.ingredients_width || DEFAULT_LTX_INGREDIENTS_WIDTH);
  const ingredientsHeight = Number(repaired.ingredients_height || DEFAULT_LTX_INGREDIENTS_HEIGHT);
  if (ingredientsWidth === 1920 && ingredientsHeight === 1080) {
    repaired.ingredients_width = DEFAULT_LTX_INGREDIENTS_WIDTH;
    repaired.ingredients_height = DEFAULT_LTX_INGREDIENTS_HEIGHT;
  }
  return repaired;
}

export function normalizeBuilderStoryLayer(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const lyricStoryStrength = Math.max(0, Math.min(10, Number(source.lyric_story_strength ?? source.lyricStoryStrength ?? 7)));
  const imageWorldStyle = ["natural", "surreal_subject", "balanced_surreal", "full_surreal", "abstract", "custom"].includes(String(source.image_world_style || source.imageWorldStyle || "natural"))
    ? String(source.image_world_style || source.imageWorldStyle || "natural")
    : "natural";
  return {
    enabled: source.enabled !== false,
    overall_story_idea: String(source.overall_story_idea || source.overallStoryIdea || source.story_idea || source.storyIdea || ""),
    user_story_arc: String(source.user_story_arc || source.userStoryArc || ""),
    song_story_brief: String(source.song_story_brief || source.songStoryBrief || ""),
    lyric_story_strength: Number.isFinite(lyricStoryStrength) ? lyricStoryStrength : 7,
    image_world_style: imageWorldStyle,
    image_custom_style_direction: String(source.image_custom_style_direction || source.imageCustomStyleDirection || ""),
  };
}

export function normalizeBuilderStoryboardDefaults(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const motionDefaults = source.motion_defaults && typeof source.motion_defaults === "object" ? source.motion_defaults : {};
  const cameraSpeed = Math.max(0, Math.min(10, Number(source.camera_motion_speed ?? source.cameraMotionSpeed ?? motionDefaults.camera_motion_speed ?? 4)));
  const characterSpeed = Math.max(0, Math.min(10, Number(source.character_motion_speed ?? source.characterMotionSpeed ?? motionDefaults.character_motion_speed ?? 4)));
  const cutFrequency = storyboardCutFrequencyValue(source.minimax_h3_cut_frequency ?? source.cut_frequency ?? source.cutFrequency);
  const temporalIntensity = Math.max(0, Math.min(10, Number(source.temporal_background_intensity ?? source.temporalBackgroundIntensity ?? 8)));
  return {
    global_consistency_phrase: String(source.global_consistency_phrase || source.globalConsistencyPhrase || ""),
    camera_motion_speed: Number.isFinite(cameraSpeed) ? cameraSpeed : 4,
    character_motion_speed: Number.isFinite(characterSpeed) ? characterSpeed : 4,
    minimax_h3_cut_frequency: cutFrequency,
    camera_guidance: String(source.camera_guidance || motionDefaults.camera_guidance || ""),
    character_guidance: String(source.character_guidance || motionDefaults.character_guidance || ""),
    performance_style: String(source.performance_style || source.performanceStyle || source.performance_style_default || ""),
    short_film_planning_mode: normalizeMiniMaxShortFilmPlanningMode(source.short_film_planning_mode || source.shortFilmPlanningMode),
    camera_flow: String(source.camera_flow || source.cameraFlow || ""),
    custom_camera_flow_sequence: normalizeStoryboardCustomCameraFlowSequence(source.custom_camera_flow_sequence || source.customCameraFlowSequence),
    image_shot_flow: String(source.image_shot_flow || source.imageShotFlow || ""),
    image_aesthetic: String(source.image_aesthetic || source.imageAesthetic || ""),
    video_style: String(source.video_style || source.videoStyle || ""),
    video_style_custom: String(source.video_style_custom || source.videoStyleCustom || ""),
    temporal_world_effect: String(source.temporal_world_effect || source.temporalWorldEffect || ""),
    temporal_world_effect_custom: String(source.temporal_world_effect_custom || source.temporalWorldEffectCustom || ""),
    temporal_allow_background_extras: (source.temporal_allow_background_extras ?? source.temporalAllowBackgroundExtras) !== false,
    temporal_background_intensity: Number.isFinite(temporalIntensity) ? temporalIntensity : 8,
    temporal_environment_time_passage: (source.temporal_environment_time_passage ?? source.temporalEnvironmentTimePassage) !== false,
    temporal_protected_characters: ["all_referenced", "lead_only", "custom"].includes(String(source.temporal_protected_characters || source.temporalProtectedCharacters || ""))
      ? String(source.temporal_protected_characters || source.temporalProtectedCharacters)
      : "all_referenced",
    temporal_protected_custom: String(source.temporal_protected_custom || source.temporalProtectedCustom || ""),
    fx_preset: String(source.fx_preset || source.fxPreset || ""),
    fx_custom_json: String(source.fx_custom_json || source.fxCustomJson || ""),
  };
}

export function builderMotionSpeedGuidance(value, kind = "camera") {
  const speed = Math.max(0, Math.min(10, Number(value ?? 4)));
  if (kind === "character") {
    if (speed <= 0) return "Character speed 0/10: subject stays still or holds a pose.";
    if (speed <= 3) return `Character speed ${speed}/10: subtle motion like gestures, turns, swaying, reaching, or small steps.`;
    if (speed <= 6) return `Character speed ${speed}/10: active performance like walking, dancing, interacting with objects, or using the set.`;
    if (speed <= 8) return `Character speed ${speed}/10: energetic action like running, hard dancing, climbing, struggling, spinning, or crossing the space.`;
    return `Character speed ${speed}/10: fast action movement, rapid direction changes, and intense physical performance while keeping the subject readable.`;
  }
  if (speed <= 0) return "Camera speed 0/10: locked-off static camera.";
  if (speed <= 3) return `Camera speed ${speed}/10: slow, gentle camera motion; one simple move at most.`;
  if (speed <= 6) return `Camera speed ${speed}/10: controlled cinematic movement like tracking, pan, dolly, crane, reveal, or orbit.`;
  if (speed <= 8) return `Camera speed ${speed}/10: energetic movement with stronger tracking, orbit, whip pan, rise, reveal, or compound motion.`;
  return `Camera speed ${speed}/10: fast action camera language with multiple coordinated moves while keeping the subject readable.`;
}

export function normalizeAutoBuildPreparation(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  return {
    input_fingerprint: String(source.input_fingerprint || source.inputFingerprint || ""),
    lyrics_fingerprint: String(source.lyrics_fingerprint || source.lyricsFingerprint || ""),
    segment_duration: Number(source.segment_duration ?? source.segmentDuration ?? 0) || 0,
    engine: String(source.engine || ""),
    prepared_at: String(source.prepared_at || source.preparedAt || ""),
  };
}

export function autoBuildFingerprint(parts = []) {
  const text = parts.map((part) => String(part ?? "").trim()).join("\u001f");
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `${(hash >>> 0).toString(16)}-${text.length}`;
}

export function cloneZImageSettings(settings) {
  const source = settings || {};
  return {
    unet_name: source.unet_name || "z_image_turbo_bf16.safetensors",
    clip_name: source.clip_name || "qwen_3_4b.safetensors",
    vae_name: source.vae_name || "ae.safetensors",
    first_pass_width: Number(source.first_pass_width || 1280),
    first_pass_height: Number(source.first_pass_height || 720),
    second_pass_width: Number(source.second_pass_width || 1920),
    second_pass_height: Number(source.second_pass_height || 1080),
    seed: Number(source.seed || 1),
    seed_mode: source.seed_mode || "fixed",
    batch_size: Math.max(1, Math.min(16, Number(source.batch_size || 1))),
    use_loras: Boolean(source.use_loras),
    lora_count: Math.max(0, Math.min(4, Number(source.lora_count || 0))),
    loras: Array.isArray(source.loras) ? source.loras.map((item) => ({
      name: item?.name || "[none]",
      first_pass_strength: Number(item?.first_pass_strength ?? item?.strength ?? 0.5),
      second_pass_strength: 0,
      strength: Number(item?.first_pass_strength ?? item?.strength ?? 1),
    })) : [],
    use_image_to_image: Boolean(source.use_image_to_image),
    image_to_image_start_at_step: Math.max(1, Math.min(8, Number(source.image_to_image_start_at_step || 5))),
    image_to_image_path: source.image_to_image_path || "",
    image_to_image_data: source.image_to_image_data || "",
    image_to_image_name: source.image_to_image_name || "",
    image_trigger_phrase: source.image_trigger_phrase || "",
  };
}

export function cloneErnieImageSettings(settings) {
  const source = settings || {};
  return {
    ...defaultErnieImageSettings(),
    ...source,
    width: Number(source.width || 1280),
    height: Number(source.height || 720),
    seed: Number(source.seed || 1),
    batch_size: Math.max(1, Math.min(16, Number(source.batch_size || 1))),
    lora_count: Math.max(0, Math.min(4, Number(source.lora_count || 0))),
    loras: Array.isArray(source.loras) ? source.loras.map((item) => ({ name: item?.name || "[none]", strength: Number(item?.strength ?? 1) })) : [],
    image_trigger_phrase: source.image_trigger_phrase || "",
  };
}

function pathLooksInsideFolder(path = "", folder = "") {
  const item = String(path || "").trim().replace(/\\/g, "/").toLowerCase();
  const root = String(folder || "").trim().replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
  if (!item || !root) return false;
  return item === root || item.startsWith(`${root}/`);
}

export function scrubGlobalImageToImageSourceForProject(settings, projectFolder = "") {
  if (!settings || typeof settings !== "object") return settings;
  const path = String(settings.image_to_image_path || "").trim();
  if (settings.use_image_to_image && path && projectFolder && !pathLooksInsideFolder(path, projectFolder)) {
    settings.use_image_to_image = false;
    settings.image_to_image_path = "";
    settings.image_to_image_data = "";
    settings.image_to_image_name = "";
  }
  return settings;
}

export function cloneKrea2TwoPassSettings(settings) {
  const source = settings || {};
  const legacyLora = source.enhancer_lora_name && source.enhancer_lora_name !== "[none]"
    ? [{
      name: source.enhancer_lora_name,
      first_pass_strength: Number(source.enhancer_lora_first_pass_strength ?? source.enhancer_lora_strength ?? 0.5),
      second_pass_strength: Number(source.enhancer_lora_second_pass_strength ?? source.enhancer_lora_strength ?? 0),
      strength: Number(source.enhancer_lora_second_pass_strength ?? source.enhancer_lora_strength ?? 0),
    }]
    : [];
  const sourceLoras = Array.isArray(source.loras) && source.loras.length ? source.loras : legacyLora;
  return {
    ...defaultKrea2TwoPassSettings(),
    ...source,
    cfg: Math.max(1, Math.min(1.2, Number(source.cfg ?? 1.2))),
    seed: Number(source.seed || 1),
    batch_size: Math.max(1, Math.min(16, Number(source.batch_size || 1))),
    use_loras: Boolean(source.use_loras ?? source.use_custom_loras ?? legacyLora.length),
    lora_count: Math.max(0, Math.min(4, Number(source.lora_count || sourceLoras.length || 0))),
    loras: sourceLoras.map((item) => ({
      name: item?.name || "[none]",
      first_pass_strength: Number(item?.first_pass_strength ?? item?.strength ?? 0.5),
      second_pass_strength: Number(item?.second_pass_strength ?? item?.strength ?? 0),
      strength: Number(item?.second_pass_strength ?? item?.strength ?? 0),
    })),
    image_to_image_creativity: Math.max(0, Math.min(10, Number(source.image_to_image_creativity ?? 5))),
    image_trigger_phrase: source.image_trigger_phrase || "",
  };
}

export function cloneNBImageSettings(settings) {
  const source = settings || {};
  return {
    ...defaultNBImageSettings(),
    ...source,
    api_key: source.api_key || "",
    model: source.model || DEFAULT_NB_IMAGE_MODEL,
    use_text_only_gemma_prompt: Boolean(source.use_text_only_gemma_prompt),
    use_director_notes: Boolean(source.use_director_notes),
  };
}

export function cloneFlowGptBrowserSettings(settings) {
  const source = settings || {};
  const provider = normalizeFlowGptBrowserProvider(source.provider);
  const normalizeBrowserReferences = (items) => (Array.isArray(items) ? items : []).map((item) => ({
    path: String(item?.path || ""),
    data: String(item?.data || ""),
    name: String(item?.name || "reference.png"),
    location_label: String(item?.location_label || ""),
  })).filter((item) => item.path || item.data);
  const referenceGroups = (Array.isArray(source.reference_groups) ? source.reference_groups : [])
    .map((group, index) => ({
      id: String(group?.id || `group_${index + 1}`),
      name: String(group?.name || `Group ${index + 1}`),
      images: normalizeBrowserReferences(group?.images),
    }));
  if (!referenceGroups.length) referenceGroups.push({ id: "group_1", name: "Group 1", images: [] });
  const requestedActiveGroupId = String(source.active_reference_group_id || "");
  const activeReferenceGroupId = referenceGroups.some((group) => group.id === requestedActiveGroupId)
    ? requestedActiveGroupId
    : referenceGroups[0].id;
  const locationSource = source.location_reference || null;
  const locationReference = locationSource && (locationSource.path || locationSource.data)
    ? {
        path: String(locationSource.path || ""),
        data: String(locationSource.data || ""),
        name: String(locationSource.name || "location.png"),
        location_label: String(locationSource.location_label || ""),
      }
    : null;
  return {
    ...defaultFlowGptBrowserSettings(),
    ...source,
    provider,
    aspect_ratio: String(source.aspect_ratio || source.aspectRatio || "16:9").trim() || "16:9",
    timeout_seconds: Math.max(60, Math.min(2400, Number(source.timeout_seconds || 600))),
    flow_timeout_seconds: Math.max(60, Math.min(1800, Number(source.flow_timeout_seconds || 420))),
    gpt_timeout_seconds: Math.max(60, Math.min(2400, Number(source.gpt_timeout_seconds || 600))),
    meta_timeout_seconds: Math.max(60, Math.min(2400, Number(source.meta_timeout_seconds || 600))),
    max_retries: Math.max(1, Math.min(20, Number(source.max_retries || 10))),
    failure_mode: ["last_successful_image", "try_other_provider", "stop"].includes(source.failure_mode) ? source.failure_mode : "last_successful_image",
    ask_previous_scene_image: Boolean(source.ask_previous_scene_image || source.askPreviousSceneImage),
    manual_chat_prompt: String(source.manual_chat_prompt || defaultFlowGptBrowserSettings().manual_chat_prompt),
    auto_advance_reference_group: source.auto_advance_reference_group !== false,
    band_sequence_enabled: Boolean(source.band_sequence_enabled),
    band_sequence_singer_references: normalizeBrowserReferences(source.band_sequence_singer_references),
    band_sequence_extra_references: normalizeBrowserReferences(source.band_sequence_extra_references),
    band_sequence_member_references: normalizeBrowserReferences(source.band_sequence_member_references),
    band_sequence_location_references: normalizeBrowserReferences(source.band_sequence_location_references),
    band_sequence_location_index: Math.max(0, Number(source.band_sequence_location_index || 0)),
    band_sequence_set_index: Math.max(0, Math.min(3, Number(source.band_sequence_set_index || 0))),
    reference_groups: referenceGroups,
    active_reference_group_id: activeReferenceGroupId,
    location_reference: locationReference,
    setup_note_collapsed: Boolean(source.setup_note_collapsed),
  };
}

export function cloneFluxKleinSettings(settings) {
  const source = settings || {};
  return {
    ...defaultFluxKleinSettings(),
    ...source,
    width: Number(source.width || 1024),
    height: Number(source.height || 576),
    seed: Number(source.seed || 100),
    use_text_only_gemma_prompt: Boolean(source.use_text_only_gemma_prompt),
    use_director_notes: Boolean(source.use_director_notes),
    use_loras: Boolean(source.use_loras),
    lora_count: Math.max(0, Math.min(4, Number(source.lora_count || 0))),
    loras: Array.isArray(source.loras) ? source.loras.map((item) => ({
      name: item?.name || "[none]",
      strength: Number(item?.strength ?? 1),
    })) : [],
    image_trigger_phrase: source.image_trigger_phrase || "",
  };
}

export function cloneI2VVideoSettings(settings) {
  const source = settings || {};
  const ltxVersion = String(source.ltx_version || "2.5") === "2.3" ? "2.3" : "2.5";
  const requestedUpscaler = String(source.upscale_model_name || "").trim();
  // Older builds populated this field from the regular image-upscaler list.
  // Migrate those invalid .pth selections back to the matching LTX latent
  // upscaler so existing projects render without manual repair.
  const latentUpscaler = !requestedUpscaler || /\.pth$/i.test(requestedUpscaler)
    ? (ltxVersion === "2.3"
      ? "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
      : "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")
    : requestedUpscaler;
  return repairI2VVideoSettingDimensions({
    ...defaultI2VVideoSettings(),
    ...source,
    ltx_version: ltxVersion,
    upscale_model_name: latentUpscaler,
    use_gguf_model: source.use_gguf_model ?? source.useGgufModel ?? (String(source.ltx_version || "2.5") === "2.3"),
    unet_name: BAD_I2V_UNET_ALIASES.has(source.unet_name) ? DEFAULT_I2V_UNET : source.unet_name || DEFAULT_I2V_UNET,
    diffusion_model_name: source.diffusion_model_name || source.model_name || (String(source.ltx_version || "2.5") === "2.3" ? DEFAULT_I2V_DIFFUSION_MODEL : "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
    use_sage_attention: Boolean(source.use_sage_attention ?? false),
    enable_fp16_accumulation: Boolean(source.enable_fp16_accumulation ?? false),
    fps: Number(source.fps || 24),
    width: Number(source.width || 1920),
    height: Number(source.height || 1080),
    resolution_aspect_ratio: source.resolution_aspect_ratio || "16:9 (Widescreen)",
    resolution_megapixels: Number(source.resolution_megapixels || 1.2),
    seed: Number(source.seed || 69),
    tail_loss_frames: Math.max(0, Number(source.tail_loss_frames ?? 25)),
    pre_frames: Math.max(0, Number(source.pre_frames ?? 50)),
    ingredients_lora_name: source.ingredients_lora_name || REQUIRED_LTX_INGREDIENTS_LORA,
    ingredients_first_pass_strength: Number(source.ingredients_first_pass_strength ?? 1),
    ingredients_second_pass_strength: 0,
    ingredients_width: Number(source.ingredients_width || DEFAULT_LTX_INGREDIENTS_WIDTH),
    ingredients_height: Number(source.ingredients_height || DEFAULT_LTX_INGREDIENTS_HEIGHT),
    id_lora_name: source.id_lora_name || REQUIRED_LTX_ID_LORA,
    id_lora_first_pass_strength: Number(source.id_lora_first_pass_strength ?? 1),
    id_lora_second_pass_strength: Number(source.id_lora_second_pass_strength ?? 1),
    id_lora_reference_audio_path: String(source.id_lora_reference_audio_path || source.reference_audio_path || ""),
    identity_guidance_scale: Number(source.identity_guidance_scale ?? 3),
    identity_start_percent: Number(source.identity_start_percent ?? 0),
    identity_end_percent: Number(source.identity_end_percent ?? 1),
    id_lora_duration: Number(source.id_lora_duration || 5),
    pass1_sampler_name: source.pass1_sampler_name || "euler_ancestral",
    pass1_sigmas: source.pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS,
    pass1_inplace_strength: Number(source.pass1_inplace_strength ?? 1),
    pass1_inplace_bypass: Boolean(source.pass1_inplace_bypass),
    pass2_sampler_name: source.pass2_sampler_name || "euler_ancestral",
    pass2_sigmas: source.pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS,
    pass2_inplace_strength: Number(source.pass2_inplace_strength ?? 1),
    pass2_inplace_bypass: Boolean(source.pass2_inplace_bypass),
    t2v_pass1_sampler_name: source.t2v_pass1_sampler_name || "euler_ancestral",
    t2v_pass1_sigmas: source.t2v_pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS,
    t2v_pass2_sampler_name: source.t2v_pass2_sampler_name || "euler_ancestral",
    t2v_pass2_sigmas: source.t2v_pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS,
    rtv_pass1_sampler_name: source.rtv_pass1_sampler_name || "euler_ancestral",
    rtv_pass1_sigmas: source.rtv_pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS,
    rtv_pass2_sampler_name: source.rtv_pass2_sampler_name || "euler_ancestral",
    rtv_pass2_sigmas: source.rtv_pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS,
    ingredients_pass1_sampler_name: source.ingredients_pass1_sampler_name || DEFAULT_INGREDIENTS_SAMPLER,
    ingredients_pass1_sigmas: source.ingredients_pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS,
    ingredients_pass2_sampler_name: source.ingredients_pass2_sampler_name || DEFAULT_INGREDIENTS_SAMPLER,
    ingredients_pass2_sigmas: source.ingredients_pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS,
    lora_count: Math.max(0, Math.min(4, Number(source.lora_count || 0))),
    loras: Array.isArray(source.loras) ? source.loras.map((item) => ({
      name: item?.name || "[none]",
      first_pass_strength: Number(item?.first_pass_strength ?? item?.strength ?? 1),
      second_pass_strength: Number(item?.second_pass_strength ?? item?.strength ?? 1),
      strength: Number(item?.second_pass_strength ?? item?.first_pass_strength ?? item?.strength ?? 1),
    })) : [],
    video_trigger_phrase: source.video_trigger_phrase || "",
  });
}

export function createModelSettings({
  activeSegment, currentVideoMode, ensureAllSegmentRuntimeFields, ernieImageTriggerInput,
  fluxImageTriggerInput, imageTriggerInput, krea2TwoPassImageTriggerInput, ltxVideoPanel, miniMaxEnginePanel,
  projectVideoEngineBadge, segmentImageSource, settingsModalControls, state, syncErnieImagePanel,
  syncFluxKleinPanel, syncI2VVideoSettingsPanel, syncKrea2TwoPassPanel, syncMiniMaxH3Panel, syncNBImagePanel,
  syncVideoModePanel, syncVideoTypeControl, syncZEnhanceSettingsPanel, syncZImageSettingsPanel,
  videoSettingsSegment, videoTriggerInput,
}) {
  function cloneFlowGptBrowserSettingsForLoadedProject(savedSettings) {
    if (savedSettings) return cloneFlowGptBrowserSettings(savedSettings);
    const defaults = defaultFlowGptBrowserSettings();
    return cloneFlowGptBrowserSettings({
      ...state.flowGptBrowserSettings,
      reference_groups: defaults.reference_groups,
      active_reference_group_id: defaults.active_reference_group_id,
      location_reference: null,
      band_sequence_enabled: defaults.band_sequence_enabled,
      band_sequence_singer_references: defaults.band_sequence_singer_references,
      band_sequence_extra_references: defaults.band_sequence_extra_references,
      band_sequence_member_references: defaults.band_sequence_member_references,
      band_sequence_location_references: defaults.band_sequence_location_references,
      band_sequence_location_index: 0,
      band_sequence_set_index: 0,
    });
  }

  function applyModelDefaults(defaults) {
    if (!defaults || typeof defaults !== "object" || Array.isArray(defaults)) return false;
    const legacyLlmMaxTokens = defaults.llm_max_tokens ?? defaults.llmMaxTokens;
    if (defaults.text_gemma_runner || defaults.textGemmaRunner) {
      state.textGemmaRunner = defaults.text_gemma_runner || defaults.textGemmaRunner || state.textGemmaRunner || "builtin";
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "gemma_gpu_layers") || Object.prototype.hasOwnProperty.call(defaults, "gemmaGpuLayers") || Object.prototype.hasOwnProperty.call(defaults, "n_gpu_layers")) {
      state.gemmaGpuLayers = normalizeGemmaGpuLayers(defaults.gemma_gpu_layers ?? defaults.gemmaGpuLayers ?? defaults.n_gpu_layers ?? state.gemmaGpuLayers);
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "gemma_context_limit") || Object.prototype.hasOwnProperty.call(defaults, "gemmaContextLimit") || Object.prototype.hasOwnProperty.call(defaults, "n_ctx") || legacyLlmMaxTokens != null) {
      state.gemmaContextLimit = normalizeGemmaContextLimit(defaults.gemma_context_limit ?? defaults.gemmaContextLimit ?? defaults.n_ctx ?? legacyLlmMaxTokens ?? state.gemmaContextLimit);
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "gemma_output_token_limit") || Object.prototype.hasOwnProperty.call(defaults, "gemmaOutputTokenLimit") || legacyLlmMaxTokens != null) {
      state.gemmaOutputTokenLimit = normalizeOutputTokenLimit(defaults.gemma_output_token_limit ?? defaults.gemmaOutputTokenLimit ?? legacyLlmMaxTokens ?? state.gemmaOutputTokenLimit);
    }
    if (defaults.lm_studio_base_url || defaults.lmStudioBaseUrl) {
      state.lmStudioBaseUrl = defaults.lm_studio_base_url || defaults.lmStudioBaseUrl || state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1";
    }
    if (defaults.lm_studio_model || defaults.lmStudioModel) {
      state.lmStudioModel = defaults.lm_studio_model || defaults.lmStudioModel || state.lmStudioModel || "";
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "lm_studio_api_key") || Object.prototype.hasOwnProperty.call(defaults, "lmStudioApiKey")) {
      state.lmStudioApiKey = defaults.lm_studio_api_key ?? defaults.lmStudioApiKey ?? state.lmStudioApiKey ?? "";
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "lm_studio_context_limit") || Object.prototype.hasOwnProperty.call(defaults, "lmStudioContextLimit")) {
      state.lmStudioContextLimit = normalizeLmStudioContextLimit(defaults.lm_studio_context_limit ?? defaults.lmStudioContextLimit ?? state.lmStudioContextLimit);
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "lm_studio_output_token_limit") || Object.prototype.hasOwnProperty.call(defaults, "lmStudioOutputTokenLimit") || legacyLlmMaxTokens != null) {
      state.lmStudioOutputTokenLimit = normalizeOutputTokenLimit(defaults.lm_studio_output_token_limit ?? defaults.lmStudioOutputTokenLimit ?? legacyLlmMaxTokens ?? state.lmStudioOutputTokenLimit);
    }
    if (defaults.llm_api_provider || defaults.llmApiProvider) {
      state.llmApiProvider = defaults.llm_api_provider || defaults.llmApiProvider || state.llmApiProvider || "openai";
    }
    if (defaults.llm_api_model || defaults.llmApiModel) {
      state.llmApiModel = defaults.llm_api_model || defaults.llmApiModel || state.llmApiModel || "";
    }
    if (defaults.own_server_url || defaults.ownServerUrl) {
      state.ownServerUrl = defaults.own_server_url || defaults.ownServerUrl || state.ownServerUrl || "http://127.0.0.1:8000/v1";
    }
    if (defaults.own_server_model || defaults.ownServerModel) {
      state.ownServerModel = defaults.own_server_model || defaults.ownServerModel || state.ownServerModel || "";
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "own_server_output_token_limit") || Object.prototype.hasOwnProperty.call(defaults, "ownServerOutputTokenLimit")) {
      state.ownServerOutputTokenLimit = normalizeOutputTokenLimit(defaults.own_server_output_token_limit ?? defaults.ownServerOutputTokenLimit ?? state.ownServerOutputTokenLimit);
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "own_server_timeout") || Object.prototype.hasOwnProperty.call(defaults, "ownServerTimeoutMinutes") || Object.prototype.hasOwnProperty.call(defaults, "own_server_timeout_minutes")) {
      const rawTimeout = defaults.ownServerTimeoutMinutes ?? defaults.own_server_timeout_minutes ?? (Number(defaults.own_server_timeout) > 15 ? Number(defaults.own_server_timeout) / 60 : defaults.own_server_timeout);
      state.ownServerTimeoutMinutes = normalizeOwnServerTimeoutMinutes(rawTimeout ?? state.ownServerTimeoutMinutes);
    }
    state.imageModelMode = defaults.image_model_mode || defaults.imageModelMode || defaults.flux_klein_settings?.image_model_mode || defaults.fluxKleinSettings?.image_model_mode || state.imageModelMode || "zimage";
    state.videoType = normalizeVideoType(defaults.video_type || defaults.videoType || state.videoType);
    if (defaults.zimage_settings || defaults.zimageSettings) {
      state.zimageSettings = scrubGlobalImageToImageSourceForProject(cloneZImageSettings(defaults.zimage_settings || defaults.zimageSettings), state.projectFolder);
    }
    if (defaults.reference_krea2_settings || defaults.referenceKrea2Settings) {
      state.referenceKrea2Settings = cloneKrea2ReferenceSettings(defaults.reference_krea2_settings || defaults.referenceKrea2Settings);
    }
    if (defaults.flux_klein_settings || defaults.fluxKleinSettings) {
      state.fluxKleinSettings = cloneFluxKleinSettings(defaults.flux_klein_settings || defaults.fluxKleinSettings);
    }
    if (defaults.flow_gpt_browser_settings || defaults.flowGptBrowserSettings) {
      state.flowGptBrowserSettings = cloneFlowGptBrowserSettings(defaults.flow_gpt_browser_settings || defaults.flowGptBrowserSettings);
    }
    if (defaults.ernie_image_settings || defaults.ernieImageSettings) {
      state.ernieImageSettings = scrubGlobalImageToImageSourceForProject(cloneErnieImageSettings(defaults.ernie_image_settings || defaults.ernieImageSettings), state.projectFolder);
    }
    if (defaults.krea2_2pass_settings || defaults.krea2TwoPassSettings) {
      state.krea2TwoPassSettings = scrubGlobalImageToImageSourceForProject(cloneKrea2TwoPassSettings(defaults.krea2_2pass_settings || defaults.krea2TwoPassSettings), state.projectFolder);
    }
    if (defaults.nb_image_settings || defaults.nbImageSettings) {
      state.nbImageSettings = cloneNBImageSettings(defaults.nb_image_settings || defaults.nbImageSettings);
    }
    if (defaults.z_enhance_settings || defaults.zEnhanceSettings) {
      state.zEnhanceSettings = {
        ...defaultZEnhanceSettings(),
        ...(defaults.z_enhance_settings || defaults.zEnhanceSettings || {}),
      };
    }
    state.videoModelMode = defaults.video_model_mode || defaults.videoModelMode || state.videoModelMode || "i2v";
    if (defaults.i2v_video_settings || defaults.i2vVideoSettings) {
      state.i2vVideoSettings = cloneI2VVideoSettings(defaults.i2v_video_settings || defaults.i2vVideoSettings);
    }
    syncZImageSettingsPanel();
    syncFluxKleinPanel();
    syncErnieImagePanel();
    syncKrea2TwoPassPanel();
    syncNBImagePanel();
    syncZEnhanceSettingsPanel();
    syncVideoTypeControl();
    syncI2VVideoSettingsPanel();
    syncVideoModePanel();
    return true;
  }

  async function loadGlobalModelDefaultsQuiet() {
    try {
      const data = await getJson("/vrgdg/music_builder/model_defaults");
      const applied = applyModelDefaults(data.defaults || {});
      if (applied) {
        console.log("[VRGDG Music Builder] Loaded global model defaults:", data.path || "");
      }
      return applied;
    } catch (error) {
      console.warn("[VRGDG Music Builder] Could not load global model defaults:", error);
      return false;
    }
  }

  function clearTriggerPhrasesForFreshProject() {
    state.imageTriggerPhrase = "";
    state.videoTriggerPhrase = "";
    if (state.zimageSettings) state.zimageSettings.image_trigger_phrase = "";
    if (state.fluxKleinSettings) state.fluxKleinSettings.image_trigger_phrase = "";
    if (state.ernieImageSettings) state.ernieImageSettings.image_trigger_phrase = "";
    if (state.krea2TwoPassSettings) state.krea2TwoPassSettings.image_trigger_phrase = "";
    if (state.i2vVideoSettings) state.i2vVideoSettings.video_trigger_phrase = "";
    imageTriggerInput.value = "";
    ernieImageTriggerInput.value = "";
    krea2TwoPassImageTriggerInput.value = "";
    fluxImageTriggerInput.value = "";
    videoTriggerInput.value = "";
  }

  function activeZImageSettings() {
    const segment = activeSegment();
    if (segment?.use_scene_zimage_settings) {
      if (!segment.zimage_settings) segment.zimage_settings = cloneZImageSettings(state.zimageSettings);
      return segment.zimage_settings;
    }
    return state.zimageSettings;
  }

  function activeErnieImageSettings() {
    const segment = activeSegment();
    if (segment?.use_scene_ernie_image_settings) {
      if (!segment.ernie_image_settings) segment.ernie_image_settings = cloneErnieImageSettings(state.ernieImageSettings);
      return segment.ernie_image_settings;
    }
    return state.ernieImageSettings;
  }

  function activeKrea2TwoPassSettings() {
    const segment = activeSegment();
    if (segment?.use_scene_krea2_2pass_settings) {
      if (!segment.krea2_2pass_settings) segment.krea2_2pass_settings = cloneKrea2TwoPassSettings(state.krea2TwoPassSettings);
      return segment.krea2_2pass_settings;
    }
    return state.krea2TwoPassSettings;
  }

  function activeNBImageSettings() {
    const segment = activeSegment();
    if (segment?.use_scene_nb_image_settings) {
      if (!segment.nb_image_settings) segment.nb_image_settings = cloneNBImageSettings(state.nbImageSettings);
      return segment.nb_image_settings;
    }
    return state.nbImageSettings;
  }

  function activeFluxKleinSettings() {
    const segment = activeSegment();
    if (segment?.use_scene_flux_klein_settings) {
      if (!segment.flux_klein_settings) segment.flux_klein_settings = cloneFluxKleinSettings(state.fluxKleinSettings);
      return segment.flux_klein_settings;
    }
    return state.fluxKleinSettings;
  }

  function activeI2VVideoSettings() {
    const segment = videoSettingsSegment();
    if (segment?.use_scene_i2v_video_settings) {
      if (!segment.i2v_video_settings) segment.i2v_video_settings = cloneI2VVideoSettings(state.i2vVideoSettings);
      return segment.i2v_video_settings;
    }
    return state.i2vVideoSettings;
  }

  function videoVisionReferenceEnabled(segment) {
    const mode = currentVideoMode();
    if (mode === "id_lora") return Boolean(segmentImageSource(segment));
    if (mode === "rtv" || mode === "ingredients") return false;
    return mode === "t2v" ? Boolean(segment?.use_t2v_vision_reference) : segment?.use_i2v_vision_reference !== false;
  }

  function setVideoVisionReferenceEnabled(segment, enabled) {
    if (!segment) return;
    if (currentVideoMode() === "t2v") segment.use_t2v_vision_reference = Boolean(enabled);
    else segment.use_i2v_vision_reference = Boolean(enabled);
  }

  function syncProjectVideoEngineUI() {
    // Auto Build can open while project restoration has not assigned an active
    // scene yet. Engine-panel synchronization expects a real scene whenever
    // the project has one, so restore that selection before refreshing panels.
    ensureAllSegmentRuntimeFields();
    if (!activeSegment()) {
      state.activeId = state.segments[0]?.id || state.overlaySegments[0]?.id || "";
    }
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    if (settingsModalControls.projectVideoEngineSelect) settingsModalControls.projectVideoEngineSelect.value = miniMaxProject ? "minimax_h3" : "ltx";
    projectVideoEngineBadge.dataset.engine = miniMaxProject ? "minimax_h3" : "ltx";
    projectVideoEngineBadge.textContent = miniMaxProject ? "◈ MiniMax" : "◈ LTX";
    projectVideoEngineBadge.title = miniMaxProject
      ? "Switch project video engine: MiniMax H3"
      : "Switch project video engine: LTX";
    projectVideoEngineBadge.setAttribute("aria-label", miniMaxProject
      ? "Switch project video engine from MiniMax H3 to LTX"
      : "Switch project video engine from LTX to MiniMax H3");
    projectVideoEngineBadge.style.background = miniMaxProject ? "#083344" : "#172554";
    projectVideoEngineBadge.style.borderColor = miniMaxProject ? "#22d3ee" : "#60a5fa";
    projectVideoEngineBadge.style.color = miniMaxProject ? "#a5f3fc" : "#bfdbfe";
    ltxVideoPanel.style.display = miniMaxProject ? "none" : "flex";
    miniMaxEnginePanel.style.display = miniMaxProject ? "flex" : "none";
    syncMiniMaxH3Panel();
  }

  return {
    activeErnieImageSettings, activeFluxKleinSettings, activeI2VVideoSettings, activeKrea2TwoPassSettings,
    activeNBImageSettings, activeZImageSettings, clearTriggerPhrasesForFreshProject,
    cloneFlowGptBrowserSettingsForLoadedProject, loadGlobalModelDefaultsQuiet, setVideoVisionReferenceEnabled,
    syncProjectVideoEngineUI, videoVisionReferenceEnabled,
  };
}
