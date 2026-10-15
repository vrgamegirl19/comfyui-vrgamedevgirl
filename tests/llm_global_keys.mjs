import assert from "node:assert/strict";
import test from "node:test";
import {
  applyGlobalLlmKeys, bindGlobalLlmKeys, globalLlmKeysForSave, markGlobalLlmKeyChanged, rememberGlobalLlmKeys,
} from "../web/music_video_builder/llm_global_keys.mjs";

const fresh = (extra = {}) => ({
  llmApiKey: "", llmApiKeyProject: "", ownServerApiKey: "", ownServerApiKeyProject: "", elevenLabsApiKey: "",
  elevenLabsApiKeyProject: "", ownServerUrl: "http://host:8000/v1", ...extra,
});

test("typed keys are sent with a save, with the server URL the custom key belongs to", () => {
  const state = fresh({ llmApiKey: " sk-1 ", ownServerApiKey: "own-1" });
  bindGlobalLlmKeys(state);
  assert.deepEqual(globalLlmKeysForSave(), { llm_api_key: "sk-1", own_server_api_key: "own-1", own_server_url: "http://host:8000/v1" });
});

test("a project key is not copied to the global keys", () => {
  bindGlobalLlmKeys(fresh({ llmApiKey: "sk-p", llmApiKeyProject: "sk-p" }));
  assert.equal(globalLlmKeysForSave(), null);
});

test("clearing a key sends an empty value only after the user changed it", () => {
  bindGlobalLlmKeys(fresh());
  assert.equal(globalLlmKeysForSave(), null);
  markGlobalLlmKeyChanged("llm_api_key");
  assert.deepEqual(globalLlmKeysForSave(), { llm_api_key: "" });
  assert.equal(globalLlmKeysForSave(), null);
});

test("saved keys fill a project unless it holds its own, and the custom key only fits its own server", () => {
  const keys = { llm_api_key: "sk-g", own_server_api_key: "own-g", own_server_url: "http://host:8000/v1/", elevenlabs_api_key: "el-g" };
  const state = fresh();
  applyGlobalLlmKeys(state, keys);
  assert.deepEqual([state.llmApiKey, state.ownServerApiKey, state.elevenLabsApiKey], ["sk-g", "own-g", "el-g"]);
  const other = fresh({ ownServerUrl: "http://elsewhere/v1", llmApiKeyProject: "sk-p" });
  applyGlobalLlmKeys(other, keys);
  assert.deepEqual([other.llmApiKey, other.ownServerApiKey], ["", ""]);
});

test("remembered keys are applied when none are passed", () => {
  rememberGlobalLlmKeys({ llm_api_key: "sk-r" });
  const state = fresh();
  applyGlobalLlmKeys(state);
  assert.equal(state.llmApiKey, "sk-r");
});
