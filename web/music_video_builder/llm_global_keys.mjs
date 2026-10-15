// API keys kept for every project: the LLM API key, the custom server key and the ElevenLabs key.
// They are saved to their own server file (never into a project's session) with every session save, and filled
// into a project when it opens or is created, unless the project holds a key of its own.

let boundState = null;
const changed = new Set();
let known = {};

export function bindGlobalLlmKeys(state) {
  boundState = state;
}

// The keys the server holds, from any response that carries them.
export function rememberGlobalLlmKeys(keys) {
  if (keys && typeof keys === "object") known = { ...keys };
}

// Call after the user edits or clears a key so the change is saved (an empty value clears the global key).
export function markGlobalLlmKeyChanged(field) {
  changed.add(field);
}

const sameUrl = (a, b) => String(a || "").trim().replace(/\/+$/, "") === String(b || "").trim().replace(/\/+$/, "");

// The keys to send with a session save, or null when there is nothing to send.
export function globalLlmKeysForSave() {
  const state = boundState;
  if (!state) return null;
  const sources = {
    llm_api_key: [state.llmApiKey, state.llmApiKeyProject],
    own_server_api_key: [state.ownServerApiKey, state.ownServerApiKeyProject],
    elevenlabs_api_key: [state.elevenLabsApiKey, state.elevenLabsApiKeyProject],
  };
  const keys = {};
  for (const [field, [live, project]] of Object.entries(sources)) {
    const value = String(live || "").trim();
    if (value && value !== String(project || "").trim()) keys[field] = value;
    else if (!value && changed.has(field)) keys[field] = "";
  }
  changed.clear();
  if (keys.own_server_api_key) keys.own_server_url = String(state.ownServerUrl || "").trim();
  for (const [field, value] of Object.entries(keys)) {
    if (value) known[field] = value;
    else delete known[field];
  }
  return Object.keys(keys).length ? keys : null;
}

// Fills the keys a project does not hold itself. The custom server key only goes to the server it was saved for.
export function applyGlobalLlmKeys(state, keys = known) {
  if (!state || !keys || typeof keys !== "object") return;
  if (keys.llm_api_key && !state.llmApiKeyProject && !state.llmApiKey) state.llmApiKey = keys.llm_api_key;
  if (keys.own_server_api_key && !state.ownServerApiKeyProject && !state.ownServerApiKey
    && sameUrl(state.ownServerUrl, keys.own_server_url)) {
    state.ownServerApiKey = keys.own_server_api_key;
  }
  if (keys.elevenlabs_api_key && !state.elevenLabsApiKeyProject && !state.elevenLabsApiKey) {
    state.elevenLabsApiKey = keys.elevenlabs_api_key;
  }
}
