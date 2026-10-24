// The same snake-case keys as builder_storyboard_defaults, restricted to scene-level settings.
export const SCENE_DEFAULT_KEYS = {
  cameraFlow: "camera_flow", customCameraFlowSequence: "custom_camera_flow_sequence",
  cameraGuidance: "camera_guidance", characterGuidance: "character_guidance",
  cameraMotionSpeed: "camera_motion_speed", characterMotionSpeed: "character_motion_speed",
  cutFrequency: "minimax_h3_cut_frequency", cutStyle: "minimax_h3_cut_style",
  performanceStyle: "performance_style", facialPerformance: "facial_performance",
  facialPerformanceCustom: "facial_performance_custom", globalConsistencyPhrase: "global_consistency_phrase",
  imageAesthetic: "image_aesthetic", imageShotFlow: "image_shot_flow",
  videoStyle: "video_style", videoStyleCustom: "video_style_custom",
  temporalWorldEffect: "temporal_world_effect", temporalWorldEffectCustom: "temporal_world_effect_custom",
  temporalAllowBackgroundExtras: "temporal_allow_background_extras",
  temporalBackgroundIntensity: "temporal_background_intensity",
  temporalEnvironmentTimePassage: "temporal_environment_time_passage",
  temporalProtectedCharacters: "temporal_protected_characters", temporalProtectedCustom: "temporal_protected_custom",
  fxPreset: "fx_preset", fxCustomJson: "fx_custom_json",
};
export function normalizeSceneDefaults(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(Object.values(SCENE_DEFAULT_KEYS).filter(key => Object.hasOwn(value, key) && value[key] !== undefined)
    .map(key => [key, JSON.parse(JSON.stringify(value[key]))]));
}
export function effectiveSceneDefaultsState(state, scene) {
  const overrides = normalizeSceneDefaults(scene?.scene_defaults);
  const resolved = { ...state };
  for (const [field, key] of Object.entries(SCENE_DEFAULT_KEYS)) {
    if (Object.hasOwn(overrides, key)) resolved[field] = overrides[key];
  }
  return resolved;
}
