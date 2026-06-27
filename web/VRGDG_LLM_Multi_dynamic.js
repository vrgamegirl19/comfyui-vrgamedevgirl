import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_LLM_Multi";

const PROVIDER_MODELS = {
  openai: [
    "gpt-image-2",
    "gpt-image-1",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o4-mini",
    "gpt-4.1",
    "gpt-4o",
  ],
  anthropic: [
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
  ],
  google: [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
  ],
  xai: [
    "grok-4.3",
    "grok-4.3-latest",
    "grok-build-0.1",
    "grok-4.1-fast",
    "grok-4.1-fast-latest",
    "grok-4",
    "grok-4-latest",
    "grok-3",
    "grok-3-latest",
    "grok-3-mini",
    "grok-3-mini-latest",
  ],
  grok: [
    "grok-4.3",
    "grok-4.3-latest",
    "grok-build-0.1",
    "grok-4.1-fast",
    "grok-4.1-fast-latest",
    "grok-4",
    "grok-4-latest",
    "grok-3",
    "grok-3-latest",
    "grok-3-mini",
    "grok-3-mini-latest",
  ],
  deepseek: ["deepseek-chat", "deepseek-reasoner"],
  openrouter: [
    "openai/gpt-5.5",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.8",
    "google/gemini-3.5-flash",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-3.1-70b-instruct",
  ],
  apifreellm: ["apifreellm"],
};

const DEFAULT_MODEL = {
  openai: "gpt-5.4-mini",
  anthropic: "claude-sonnet-4-6",
  google: "gemini-3.5-flash",
  xai: "grok-4.3",
  grok: "grok-4.3",
  deepseek: "deepseek-chat",
  openrouter: "openai/gpt-5.4-mini",
  apifreellm: "apifreellm",
};

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function syncModelWidget(node) {
  const providerWidget = getWidget(node, "provider");
  const modelWidget = getWidget(node, "model");
  if (!providerWidget || !modelWidget) return;

  const provider = String(providerWidget.value || "openai").toLowerCase();
  const models = PROVIDER_MODELS[provider] || [];
  if (!models.length) return;

  modelWidget.options = modelWidget.options || {};
  modelWidget.options.values = [...models];

  if (!models.includes(modelWidget.value)) {
    modelWidget.value = DEFAULT_MODEL[provider] || models[0];
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bindProviderChange(node) {
  const providerWidget = getWidget(node, "provider");
  if (!providerWidget || providerWidget.__vrgdgLlmMultiBound) return;

  const oldCallback = providerWidget.callback;
  providerWidget.callback = function () {
    if (oldCallback) oldCallback.apply(this, arguments);
    syncModelWidget(node);
  };

  providerWidget.__vrgdgLlmMultiBound = true;
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      bindProviderChange(this);
      setTimeout(() => syncModelWidget(this), 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindProviderChange(this);
      syncModelWidget(this);
      return r;
    };
  },
});
