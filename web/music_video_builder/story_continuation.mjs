// Story Mode scene continuations on the timeline. A plan entry with continues_previous makes its scene a MiniMax
// Latent Continuation Masked scene, set exactly like the timeline's chain button (locked scene settings, Masked
// transition, prompt written from the previous scene's final frame at render). The plan's direction becomes the
// scene's continuation direction. Python twin: apply_continuations_to_segments in
// agent_api/orchestrator/story_mode_orchestrator.py.
import { cloneMiniMaxH3Settings, isMiniMaxH3ContinuityAllowedForMode } from "./minimax_h3.mjs";
import { normalizeStoryScenePlan } from "./story_mode_data.mjs";

export function sceneIsMasked(segment) {
  return Boolean(segment?.use_scene_minimax_h3_settings)
    && segment?.minimax_h3_settings?.continuity_mode === "latent_continuation_masked";
}

// Returns { masked, cleared, blocked } scene ids. Blocked scenes want a continuation their MiniMax mode cannot do.
// Only scenes the story made masked (story_masked_continuation) are turned back off.
export function applyStoryContinuationsToSegments(segments, plan, projectSettings) {
  const byId = Object.fromEntries(normalizeStoryScenePlan(plan).map((entry) => [entry.scene_id, entry]));
  const result = { masked: [], cleared: [], blocked: [] };
  (Array.isArray(segments) ? segments : []).forEach((segment, index) => {
    const entry = byId[segment?.id];
    if (entry?.continues_previous && index > 0) {
      const base = cloneMiniMaxH3Settings(segment.use_scene_minimax_h3_settings && segment.minimax_h3_settings
        ? segment.minimax_h3_settings
        : projectSettings);
      if (!isMiniMaxH3ContinuityAllowedForMode("latent_continuation_masked", base.video_mode, base.render_pass)) {
        result.blocked.push(segment.id);
        return;
      }
      if (!sceneIsMasked(segment)) {
        segment.minimax_h3_masked_quick_prev = {
          locked: Boolean(segment.use_scene_minimax_h3_settings),
          continuity_mode: base.continuity_mode,
          location_transition_preset: base.location_transition_preset,
          continuity_prompt_from_last_frame: Boolean(base.continuity_prompt_from_last_frame),
        };
        segment.use_scene_minimax_h3_settings = true;
        segment.minimax_h3_settings = cloneMiniMaxH3Settings({
          ...base,
          continuity_mode: "latent_continuation_masked",
          location_transition_preset: "masked",
          continuity_prompt_from_last_frame: true,
        });
        segment.minimax_h3_mode = segment.minimax_h3_settings.video_mode;
        segment.story_masked_continuation = true;
        result.masked.push(segment.id);
      }
      segment.minimax_h3_continuation_direction = entry.continuation_direction;
      return;
    }
    if (!segment?.story_masked_continuation) return;
    delete segment.story_masked_continuation;
    if (!sceneIsMasked(segment)) return;  // already turned off on the timeline
    const before = segment.minimax_h3_masked_quick_prev;
    if (before && before.locked === false) {
      segment.use_scene_minimax_h3_settings = false;
      delete segment.minimax_h3_settings;
    } else {
      segment.minimax_h3_settings = cloneMiniMaxH3Settings({
        ...segment.minimax_h3_settings,
        continuity_mode: before?.continuity_mode || "off",
        location_transition_preset: before?.location_transition_preset || "normal",
        continuity_prompt_from_last_frame: Boolean(before?.continuity_prompt_from_last_frame),
      });
    }
    delete segment.minimax_h3_masked_quick_prev;
    segment.minimax_h3_continuation_direction = "";
    result.cleared.push(segment.id);
  });
  return result;
}
