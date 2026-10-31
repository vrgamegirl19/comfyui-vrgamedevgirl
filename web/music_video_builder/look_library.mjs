// The effects and filters the player and the picker can draw. The H3 Studio app serves them (`GET /vrgdg/music_builder/effects` and `/filters`, each look with its
// GLSL ES 3.00 source) and says it can draw them in the final video (`GET .../capabilities`). This pack has no render route for them yet, so here the capability is
// false and the effect and filter screens stay out of sight, as the overlays and transitions do. A project's `video_effects` and `video_filters` are kept all the same.

import { api } from "../../../scripts/api.js";
import { LOOK_KINDS, LOOK_ORDER } from "./look_model.mjs";

const lookLibraries = {};

// The looks of a kind loaded so far (empty until `loadLookLibrary` has answered).
export function lookLibrary(kind) {
  return lookLibraries[kind]?.items || [];
}

export function loadLookLibrary(kind) {
  const { plural } = LOOK_KINDS[kind];
  const library = (lookLibraries[kind] ||= { promise: null, items: [] });
  library.promise ||= api.fetchApi(`/vrgdg/music_builder/${plural}`, { timeoutMs: null })
    .then(async (response) => {
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !Array.isArray(data[plural])) throw new Error(data.error || `The ${plural} library did not load (${response.status}).`);
      library.items = data[plural];
      return data[plural];
    })
    .catch((error) => {
      library.promise = null;
      throw error;
    });
  return library.promise;
}

// Asks whether this build can draw effects and filters. It asks with plain requests, since a missing route is an answer here and not an error for the project's
// error log. Returns whether any kind can be drawn.
export async function loadLookCapabilities(state) {
  await Promise.all(LOOK_ORDER.map(async (kind) => {
    const config = LOOK_KINDS[kind];
    try {
      const response = await api.fetchApi(`/vrgdg/music_builder/${config.plural}/capabilities`, { timeoutMs: null });
      const data = response.ok ? await response.json().catch(() => null) : null;
      state[config.availableKey] = Boolean(data && data.ok !== false && data.version);
    } catch {
      state[config.availableKey] = false;
    }
  }));
  return LOOK_ORDER.some((kind) => state[LOOK_KINDS[kind].availableKey]);
}
