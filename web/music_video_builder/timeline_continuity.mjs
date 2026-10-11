import { cloneMiniMaxH3Settings } from "./minimax_h3.mjs";
import { toast } from "./controls.mjs";

export function disableAllLatentContinuation(state) {
  const turnOff = (settings) => cloneMiniMaxH3Settings({
    ...settings,
    continuity_mode: "off",
    continuity_prompt_from_last_frame: false,
  });
  state.miniMaxH3Settings = turnOff(state.miniMaxH3Settings);
  const scenes = [...(state.segments || []), ...(state.overlaySegments || [])];
  for (const scene of scenes) {
    if (scene.minimax_h3_settings) {
      scene.minimax_h3_settings = turnOff(scene.minimax_h3_settings);
    }
    delete scene.minimax_h3_masked_quick_prev;
  }
}

export function wireLatentContinuationOffAll({ button, state, pushHistory, syncInspector, render, autoSaveSessionQuiet }) {
  button.onclick = async () => {
    pushHistory();
    disableAllLatentContinuation(state);
    syncInspector();
    render();
    try {
      await autoSaveSessionQuiet("latent continuation turned off for all scenes");
      toast("Latent continuation off for all scenes.");
    } catch (error) {
      toast(String(error), true);
    }
  };
}
