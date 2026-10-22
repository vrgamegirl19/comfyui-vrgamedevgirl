import assert from "node:assert/strict";
import test from "node:test";
import {
  applyGlobalLlmKeys, bindGlobalLlmKeys, globalLlmKeysForSave, globalLlmRunnerForSave, globalLlmSettingsPayload,
  markGlobalLlmKeyChanged, rememberGlobalLlmKeys,
} from "../web/music_video_builder/llm_global_keys.mjs";

const fresh = (extra = {}) => ({
  llmApiKey: "", llmApiKeyProject: "", ownServerApiKey: "", ownServerApiKeyProject: "", elevenLabsApiKey: "",
  elevenLabsApiKeyProject: "", ownServerUrl: "http://host:8000/v1", ...extra,
});

test("typed keys are sent with a save", () => {
  bindGlobalLlmKeys(fresh({ llmApiKey: " sk-1 ", ownServerApiKey: "own-1" }));
  assert.deepEqual(globalLlmKeysForSave(), { llm_api_key: "sk-1", own_server_api_key: "own-1" });
  assert.equal(globalLlmRunnerForSave(), null, "no runner is sent before the saved one arrives");
});

test("clearing a key sends an empty value only after the user changed it", () => {
  bindGlobalLlmKeys(fresh());
  assert.equal(globalLlmKeysForSave(), null);
  markGlobalLlmKeyChanged("llm_api_key");
  assert.deepEqual(globalLlmKeysForSave(), { llm_api_key: "" });
  assert.equal(globalLlmKeysForSave(), null);
});

test("the saved LLM keys replace a project's own, whatever its server URL", () => {
  const keys = { llm_api_key: "sk-g", own_server_api_key: "own-g", elevenlabs_api_key: "el-g" };
  const state = fresh({ ownServerUrl: "http://elsewhere/v1", llmApiKeyProject: "sk-p", ownServerApiKeyProject: "own-p" });
  applyGlobalLlmKeys(state, keys, null);
  assert.deepEqual([state.llmApiKey, state.ownServerApiKey, state.elevenLabsApiKey], ["sk-g", "own-g", "el-g"]);
  assert.deepEqual([state.llmApiKeyProject, state.ownServerApiKeyProject], ["", ""]);
});

test("an older project's key is used when none is saved globally, and moves to the global keys", () => {
  const state = fresh({ llmApiKeyProject: "sk-p" });
  bindGlobalLlmKeys(state);
  applyGlobalLlmKeys(state, {}, null);
  assert.equal(state.llmApiKey, "sk-p");
  assert.deepEqual(globalLlmKeysForSave(), { llm_api_key: "sk-p" });
});

test("the saved runner settings replace the project's, with the timeout in minutes", () => {
  const state = fresh({ textGemmaRunner: "builtin", ownServerModel: "old", ownServerTimeoutMinutes: 2 });
  applyGlobalLlmKeys(state, {}, { text_gemma_runner: "own_server", own_server_url: "http://a/v1", own_server_model: "gemma", own_server_timeout: 300 });
  assert.deepEqual([state.textGemmaRunner, state.ownServerUrl, state.ownServerModel, state.ownServerTimeoutMinutes], ["own_server", "http://a/v1", "gemma", 5]);
});

test("a session save carries the runner only after the saved runner has arrived", () => {
  const state = fresh({ textGemmaRunner: "lm_studio", ownServerTimeoutMinutes: 4 });
  bindGlobalLlmKeys(state);
  rememberGlobalLlmKeys(null, null);
  const payload = globalLlmSettingsPayload(state);
  assert.equal(payload.runner.text_gemma_runner, "lm_studio");
  assert.equal(payload.runner.own_server_timeout, 240);
  assert.deepEqual(payload.keys, { llm_api_key: "", own_server_api_key: "" });
  assert.equal(globalLlmRunnerForSave().text_gemma_runner, "lm_studio");
});
