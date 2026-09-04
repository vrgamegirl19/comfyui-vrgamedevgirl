import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_VideoPromptReconstructor";
const API_RUNNER = "LLM API";
const API_DEFAULTS = {
  openai: "gpt-4o",
  anthropic: "claude-sonnet-4-6",
  google: "gemini-2.5-flash",
  openrouter: "openai/gpt-4o",
  grok: "grok-2-vision-1212",
};

const API_MODELS = {
  openai: ["gpt-5.6-luna", "gpt-5.5", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"],
  anthropic: ["claude-sonnet-4-6", "claude-opus-4-8", "claude-3-5-sonnet-20241022"],
  google: ["gemini-3.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"],
  openrouter: ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-2.5-flash"],
  grok: ["grok-4.6", "grok-4.5", "grok-2-vision-1212"],
};

function getWidget(node, name) {
  return (node.widgets || []).find((item) => item.name === name);
}

function setVisible(item, visible) {
  if (!item) return;
  if (!Object.prototype.hasOwnProperty.call(item, "__vrgdgOriginalComputeSize")) {
    item.__vrgdgOriginalComputeSize = item.computeSize;
  }
  item.hidden = !visible;
  item.serialize = true;
  item.computeSize = visible ? item.__vrgdgOriginalComputeSize : () => [0, -4];
}

function setModelOptions(model, values, preferred) {
  const models = [...new Set((values || []).map((value) => String(value || "").trim()).filter(Boolean))];
  if (!model || !models.length) return;
  model.options = model.options || {};
  model.options.values = models;
  if (!models.includes(String(model.value || ""))) model.value = preferred || models[0];
}

async function loadServerModels(node, runner, url) {
  const model = getWidget(node, "model_name");
  const base = String(url?.value || "").trim();
  if (!model || !base) return;
  const route = runner === "LM Studio" ? "/vrgdg/music_builder/lm_studio_models" : "/vrgdg/music_builder/own_server_models";
  const payload = runner === "LM Studio"
    ? { lmstudio_base_url: base, lmstudio_api_key: getWidget(node, "api_key")?.value || "" }
    : { own_server_url: base, own_server_api_key: getWidget(node, "api_key")?.value || "" };
  try {
    const response = await fetch(route, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (response.ok && Array.isArray(data.models) && data.models.length) setModelOptions(model, data.models);
  } catch (_error) {
    // The model can still be entered manually if the local server is not running yet.
  }
}

function refresh(node) {
  const runner = getWidget(node, "llm_runner");
  const provider = getWidget(node, "llm_provider");
  const model = getWidget(node, "model_name");
  const url = getWidget(node, "api_url");
  if (!runner || !provider || !model || !url) return;

  const isApi = runner.value === API_RUNNER;
  setVisible(provider, isApi);
  setVisible(url, !isApi);

  if (isApi) {
    setModelOptions(model, API_MODELS[provider.value] || API_MODELS.openai, API_DEFAULTS[provider.value]);
  }
  else {
    loadServerModels(node, runner.value, url);
  }

  if (isApi && API_DEFAULTS[provider.value] && (!model.value || model.value === "qwen2.5-vl-7b-instruct")) {
    model.value = API_DEFAULTS[provider.value];
  } else if (runner.value === "LM Studio" && (!url.value || url.value.includes("api.openai.com"))) {
    url.value = "http://127.0.0.1:1234/v1";
  } else if (runner.value === "Custom Server" && (!url.value || url.value.includes("127.0.0.1:1234"))) {
    url.value = "http://127.0.0.1:8000/v1";
  }

  node.setSize([Math.max(300, node.size?.[0] || 300), node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bind(node) {
  if (node.__vrgdgPromptReconstructorBound) return;
  for (const name of ["llm_runner", "llm_provider"]) {
    const item = getWidget(node, name);
    if (!item) continue;
    const oldCallback = item.callback;
    item.callback = function () {
      if (oldCallback) oldCallback.apply(this, arguments);
      refresh(node);
    };
  }
  node.__vrgdgPromptReconstructorBound = true;
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".dynamic",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const originalCreated = nodeType.prototype.onNodeCreated;
    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      bind(this);
      setTimeout(() => refresh(this), 0);
      return result;
    };
    nodeType.prototype.onConfigure = function () {
      const result = originalConfigure?.apply(this, arguments);
      bind(this);
      setTimeout(() => refresh(this), 0);
      return result;
    };
  },
});
