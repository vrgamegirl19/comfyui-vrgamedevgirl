// LLM settings kept for every project: the API keys (LLM API, custom server, ElevenLabs) and the LLM Runner settings
// (runner, servers, models and limits). They live in their own server files, never in a project's session, and the
// LLM API and custom server keys and the runner settings always win over a project's own copy. Switching runners or
// projects therefore keeps every saved server, model and key. Python twin: builder/project.py (GLOBAL_LLM_*).

let boundState = null;
const changed = new Set();
let known = {};
// The server's runner settings; null until they arrive, so a save made before that cannot overwrite them.
let knownRunner = null;

// [session key, state key]. The custom server timeout is seconds in the session and minutes in the state.
const RUNNER_FIELDS = [
  ["text_gemma_runner", "textGemmaRunner"], ["gemma_model_file", "gemmaModelFile"],
  ["qwen_model_file", "qwenModelFile"], ["qwen_mmproj_file", "qwenMmprojFile"],
  ["gemma_context_limit", "gemmaContextLimit"], ["gemma_output_token_limit", "gemmaOutputTokenLimit"],
  ["gemma_gpu_layers", "gemmaGpuLayers"], ["gemma_gpu_device", "gemmaGpuDevice"], ["lm_studio_base_url", "lmStudioBaseUrl"], ["lm_studio_model", "lmStudioModel"],
  ["lm_studio_api_key", "lmStudioApiKey"], ["lm_studio_context_limit", "lmStudioContextLimit"],
  ["lm_studio_output_token_limit", "lmStudioOutputTokenLimit"], ["llm_api_provider", "llmApiProvider"],
  ["llm_api_model", "llmApiModel"], ["own_server_url", "ownServerUrl"], ["own_server_model", "ownServerModel"],
  ["own_server_output_token_limit", "ownServerOutputTokenLimit"], ["own_server_timeout", "ownServerTimeoutMinutes"],
];

export function bindGlobalLlmKeys(state) {
  boundState = state;
}

// The keys and runner settings the server holds, from any response that carries them.
export function rememberGlobalLlmKeys(keys, runner) {
  if (keys && typeof keys === "object") known = { ...keys };
  if (runner && typeof runner === "object") knownRunner = { ...runner };
}

// Call after the user edits or clears a key so the change is saved (an empty value clears the global key).
export function markGlobalLlmKeyChanged(field) {
  changed.add(field);
}

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
  for (const [field, value] of Object.entries(keys)) {
    if (value) known[field] = value;
    else delete known[field];
  }
  return Object.keys(keys).length ? keys : null;
}

// The runner settings in the state, in session key names.
export function globalLlmRunnerFromState(state = boundState) {
  if (!state) return {};
  const runner = {};
  for (const [field, key] of RUNNER_FIELDS) {
    const value = key === "ownServerTimeoutMinutes" ? Number(state[key]) * 60 : state[key];
    if (value !== undefined && value !== null && value !== "" && !Number.isNaN(value)) runner[field] = value;
  }
  return runner;
}

// The runner settings to send with a session save, or null before the server's copy has arrived.
export function globalLlmRunnerForSave() {
  if (!boundState || knownRunner === null) return null;
  knownRunner = globalLlmRunnerFromState(boundState);
  return knownRunner;
}

// The body for POST /vrgdg/music_builder/llm_runner (LLM Runner > Save Runner): every runner setting and both LLM keys.
export function globalLlmSettingsPayload(state = boundState) {
  const runner = globalLlmRunnerFromState(state);
  knownRunner = runner;
  return {
    runner,
    keys: { llm_api_key: String(state?.llmApiKey || "").trim(), own_server_api_key: String(state?.ownServerApiKey || "").trim() },
  };
}

// Fills the saved keys and runner settings into the state. The LLM API and custom server keys and the runner settings
// replace a project's own copy. A key an older project saved is used when no global key exists, and moves to the
// global keys with the next save.
export function applyGlobalLlmKeys(state, keys = known, runner = knownRunner) {
  if (!state) return;
  if (keys && typeof keys === "object") {
    for (const [field, live, project] of [["llm_api_key", "llmApiKey", "llmApiKeyProject"], ["own_server_api_key", "ownServerApiKey", "ownServerApiKeyProject"]]) {
      const projectKey = String(state[project] || "").trim();
      if (keys[field]) state[live] = keys[field];
      else if (projectKey) state[live] = projectKey;
      state[project] = "";
    }
    if (keys.elevenlabs_api_key && !state.elevenLabsApiKeyProject && !state.elevenLabsApiKey) {
      state.elevenLabsApiKey = keys.elevenlabs_api_key;
    }
  }
  if (runner && typeof runner === "object") {
    for (const [field, key] of RUNNER_FIELDS) {
      if (runner[field] === undefined || runner[field] === null || runner[field] === "") continue;
      state[key] = key === "ownServerTimeoutMinutes" ? Number(runner[field]) / 60 : runner[field];
    }
  }
}
