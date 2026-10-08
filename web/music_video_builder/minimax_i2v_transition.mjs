import { MINIMAX_I2V_TRANSITION_OPTIONS, normalizeMiniMaxI2VTransitionStyle } from "./minimax_h3.mjs";
import { miniMaxI2VFrameMode } from "./minimax_keyframe_state.mjs";

// Creative prompt guidance only: frame conditioning and sampler settings remain in their own modules.
const TRANSITION_GUIDANCE = {
  natural: "Connect the frames through believable character movement and smooth camera movement. Preserve physical geometry and character identity; reach the ending pose and camera angle without morphing.",
  surreal_morph: "Creatively morph shapes, scenery and visual motifs from the first image into the last through a coherent surreal transformation. Keep the intended character recognizable and the action readable.",
  dreamlike_dissolve: "Reveal the ending composition through drifting mist, light, particles or layered imagery, with a gradual dreamlike visual dissolve.",
  environment_transformation: "Transform the setting, weather or lighting from the opening image into the ending image while keeping the character recognizable and their movement coherent.",
  camera_reveal: "Use a motivated pan, orbit, push-in or pull-back to reveal the ending composition. Preserve plausible spatial relationships and describe the camera's path.",
  custom: "Use the author's transition direction to connect the opening and ending compositions. If no direction is supplied, choose a coherent transition supported by the two images and scene context.",
};

export function miniMaxI2VTransitionPrompt(segment, mode, settings = {}) {
  if (mode !== "image_to_video" || miniMaxI2VFrameMode(segment) !== "flf") return "";
  const style = normalizeMiniMaxI2VTransitionStyle(settings.i2v_transition_style);
  const label = MINIMAX_I2V_TRANSITION_OPTIONS.find(option => option.value === style).label;
  const direction = String(settings.i2v_transition_direction || "").trim();
  return `FIRST / LAST FRAME TRANSITION — ${label}:\n${TRANSITION_GUIDANCE[style]} `
    + "The first image defines this scene's opening and the last image defines its ending. Complete the transition within the exact scene duration. "
    + "Keep the story beat, lyrics/audio timing, performance requirements and scene defaults; use this preset to resolve conflicting generic transition guidance. "
    + "Describe the visible progression in the finished shot prose. These are this scene's own frames, not a continuation from another scene."
    + (direction ? `\nAuthor's transition direction: ${direction}` : "");
}
