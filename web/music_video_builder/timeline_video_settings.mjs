import { cloneMiniMaxH3Settings } from "./minimax_h3.mjs";
import { toast } from "./controls.mjs";

export function unlockAllSceneVideoSettings(state) {
  const mode = cloneMiniMaxH3Settings(state.miniMaxH3Settings).video_mode;
  for (const scene of [...(state.segments || []), ...(state.overlaySegments || [])]) {
    scene.use_scene_minimax_h3_settings = false;
    scene.use_scene_i2v_video_settings = false;
    scene.minimax_h3_mode = mode;
    // The quick-continuation toggle must not restore a previous settings lock.
    delete scene.minimax_h3_masked_quick_prev;
  }
}

export function wireUnlockAllScenes({ button, state, pushHistory, syncInspector, render, autoSaveSessionQuiet }) {
  button.onclick = async () => {
    pushHistory();
    unlockAllSceneVideoSettings(state);
    syncInspector();
    render();
    try {
      await autoSaveSessionQuiet("all scene video settings unlocked");
      toast("All scenes now follow the global video settings.");
    } catch (error) {
      toast(String(error), true);
    }
  };
}
