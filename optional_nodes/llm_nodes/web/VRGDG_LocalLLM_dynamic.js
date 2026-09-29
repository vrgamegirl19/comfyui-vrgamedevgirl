import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_LocalLLM";

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function normalizeBaseUrl(url, backend) {
  const u = String(url || "").trim().replace(/\/+$/, "");
  if (!u) return backend === "ollama" ? "http://127.0.0.1:11434" : "http://127.0.0.1:1234/v1";
  return u;
}

function ensureModelPicker(node, models) {
  const modelWidget = getWidget(node, "model");
  if (!modelWidget || !models?.length) return;

  let picker = getWidget(node, "model_picker");
  if (!picker) {
    picker = node.addWidget("combo", "model_picker", models[0], (v) => {
      modelWidget.value = String(v || "").trim();
      app.graph.setDirtyCanvas(true, true);
    }, { values: [...models] });
  } else {
    picker.options = picker.options || {};
    picker.options.values = [...models];
  }

  const current = String(modelWidget.value || "").trim();
  if (models.includes(current)) {
    picker.value = current;
  } else {
    picker.value = models[0];
  }
}

async function fetchModels(node) {
  const backendWidget = getWidget(node, "backend");
  const baseUrlWidget = getWidget(node, "base_url");
  const apiKeyWidget = getWidget(node, "api_key");
  const modelWidget = getWidget(node, "model");
  if (!backendWidget || !baseUrlWidget || !modelWidget) return;

  const backend = String(backendWidget.value || "ollama");
  const baseUrl = normalizeBaseUrl(baseUrlWidget.value, backend);

  let models = [];
  try {
    if (backend === "ollama") {
      const res = await fetch(`${baseUrl}/api/tags`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      models = (data.models || []).map((m) => m.name).filter(Boolean);
    } else {
      const headers = {};
      const apiKey = String(apiKeyWidget?.value || "").trim();
      if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
      const res = await fetch(`${baseUrl}/models`, { headers });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      models = (data.data || []).map((m) => m.id).filter(Boolean);
    }
  } catch (e) {
    models = [];
  }

  if (!models.length) return;

  ensureModelPicker(node, models);

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bindCallbacks(node) {
  const backendWidget = getWidget(node, "backend");
  const baseUrlWidget = getWidget(node, "base_url");
  const apiKeyWidget = getWidget(node, "api_key");
  if (node.__vrgdgLocalLlmBound) return;

  const wrap = (widget) => {
    if (!widget) return;
    const old = widget.callback;
    widget.callback = function () {
      if (old) old.apply(this, arguments);
      fetchModels(node);
    };
  };

  wrap(backendWidget);
  wrap(baseUrlWidget);
  wrap(apiKeyWidget);
  wrap(getWidget(node, "model"));
  node.__vrgdgLocalLlmBound = true;
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      bindCallbacks(this);
      if (!(this.widgets || []).some((w) => w.type === "button" && w.name === "Refresh Models")) {
        this.addWidget("button", "Refresh Models", null, () => fetchModels(this));
      }
      setTimeout(() => fetchModels(this), 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindCallbacks(this);
      fetchModels(this);
      return r;
    };
  },
});
