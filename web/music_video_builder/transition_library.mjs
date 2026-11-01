// The transitions the player and the picker can draw. The H3 Studio app serves them (`GET /vrgdg/music_builder/transitions`, each with its GLSL ES 3.00
// source) and says it can draw them in the final video (`GET /vrgdg/music_builder/transitions/capabilities`). This pack has no transition render route
// yet, so here the capability is false and the transition screens stay out of sight, as the overlays do. A project's `scene_transitions` are kept all the same.

import { api } from "../../../scripts/api.js";

let libraryPromise = null;
let libraryItems = [];

// The transitions loaded so far (empty until `loadTransitionLibrary` has answered).
export function transitionLibrary() {
  return libraryItems;
}

export function loadTransitionLibrary() {
  libraryPromise ||= api.fetchApi("/vrgdg/music_builder/transitions", { timeoutMs: null })
    .then(async (response) => {
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !Array.isArray(data.transitions)) throw new Error(data.error || `The transition library did not load (${response.status}).`);
      libraryItems = data.transitions;
      return data.transitions;
    })
    .catch((error) => {
      libraryPromise = null;
      throw error;
    });
  return libraryPromise;
}

// Asks whether this build can draw transitions. It asks with a plain request, since a missing route is an answer here and not an error for the
// project's error log.
export async function loadTransitionCapabilities(state) {
  try {
    const response = await api.fetchApi("/vrgdg/music_builder/transitions/capabilities", { timeoutMs: null });
    const data = response.ok ? await response.json().catch(() => null) : null;
    state.transitionsAvailable = Boolean(data && data.ok !== false && data.version);
  } catch {
    state.transitionsAvailable = false;
  }
  return state.transitionsAvailable;
}
