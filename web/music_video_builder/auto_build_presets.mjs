import { SPEED_DEFAULT_PRESETS, musicDefaultSettings } from "../storyboard_builder/defaults_presets.mjs";
import { storyboardCameraFlowEntry } from "../storyboard_builder/shot_presets.mjs";
import { builderMotionSpeedGuidance, normalizeBuilderStoryboardDefaults } from "./model_settings.mjs";

// Update the same defaults and scene fields consumed by Story Board and prompt generation.
export function applyAutoBuildPresets(state, { recommended = false, speed, music } = {}) {
  const patch = recommended ? {
    video_style: "", video_style_custom: "", temporal_world_effect: "", temporal_world_effect_custom: "",
    fx_preset: "", fx_custom_json: "", global_consistency_phrase: "", camera_flow: "medium_shots",
    camera_motion_speed: 6, character_motion_speed: 4, minimax_h3_cut_frequency: 3, performance_style: "",
    defaults_speed_preset: "", defaults_music_preset: "",
  } : {};
  const speedSettings = SPEED_DEFAULT_PRESETS[speed];
  const musicSettings = musicDefaultSettings(music);
  if (speed !== undefined) patch.defaults_speed_preset = speedSettings ? speed : "";
  if (music !== undefined) patch.defaults_music_preset = musicSettings ? music : "";
  if (speedSettings) Object.assign(patch, {
    camera_motion_speed: speedSettings.cameraMotionSpeed,
    character_motion_speed: speedSettings.characterMotionSpeed,
    minimax_h3_cut_frequency: speedSettings.cutFrequency,
  });
  if (musicSettings) Object.assign(patch, {
    camera_flow: musicSettings.cameraFlow, performance_style: musicSettings.performanceStyle,
  });
  if (speedSettings || musicSettings) Object.assign(patch, {
    video_style: "", video_style_custom: "", temporal_world_effect: "", temporal_world_effect_custom: "",
    fx_preset: "", fx_custom_json: "",
  });
  if ("camera_motion_speed" in patch) patch.camera_guidance = builderMotionSpeedGuidance(patch.camera_motion_speed, "camera");
  if ("character_motion_speed" in patch) patch.character_guidance = builderMotionSpeedGuidance(patch.character_motion_speed, "character");
  state.builderStoryboardDefaults = normalizeBuilderStoryboardDefaults({ ...state.builderStoryboardDefaults, ...patch });
  if (recommended || musicSettings) {
    state.defaultFacialPerformance = musicSettings?.facialPerformance || "";
    state.defaultFacialPerformanceCustom = "";
  }
  let previousMotion = "";
  for (const [index, segment] of (state.segments || []).entries()) {
    segment.scene_defaults = { ...segment.scene_defaults, ...patch };
    if ("camera_flow" in patch) {
      const camera = storyboardCameraFlowEntry(patch.camera_flow, index, previousMotion);
      segment.shot_type = camera?.shot || "";
      segment.camera_motion = camera?.camera || "";
      previousMotion = segment.camera_motion;
    }
    for (const key of ["camera_motion_speed", "character_motion_speed", "minimax_h3_cut_frequency", "performance_style", "video_style", "video_style_custom"]) {
      if (key in patch) segment[key] = patch[key];
    }
    if ("character_guidance" in patch) segment.character_motion = patch.character_guidance;
    if ("camera_guidance" in patch) segment.camera_motion_speed_guidance = patch.camera_guidance;
    if ("character_guidance" in patch) segment.character_motion_guidance = patch.character_guidance;
    if ("temporal_world_effect" in patch) {
      segment.temporal_world_effect_override = "off";
      segment.temporal_world_effect_custom = "";
    }
    if ("fx_preset" in patch) { segment.fx_preset = ""; segment.fx_custom_json = ""; }
    if (recommended || musicSettings) {
      segment.facial_performance = state.defaultFacialPerformance;
      segment.facial_performance_custom = "";
    }
  }
}

// New timeline segments still need concrete framing when preserving edited defaults.
export function fillAutoBuildSceneFraming(state) {
  if (!state.builderStoryboardDefaults?.camera_flow) {
    state.builderStoryboardDefaults = normalizeBuilderStoryboardDefaults({ ...state.builderStoryboardDefaults, camera_flow: "medium_shots" });
  }
  const defaults = state.builderStoryboardDefaults;
  let previousMotion = "";
  for (const [index, segment] of (state.segments || []).entries()) {
    const sceneDefaults = { ...defaults, ...segment.scene_defaults };
    const camera = storyboardCameraFlowEntry(sceneDefaults.camera_flow || "medium_shots", index, previousMotion, sceneDefaults.custom_camera_flow_sequence);
    if (!String(segment.shot_type || "").trim()) segment.shot_type = camera?.shot || "";
    if (!String(segment.camera_motion || "").trim()) segment.camera_motion = camera?.camera || "";
    previousMotion = segment.camera_motion;
  }
}
