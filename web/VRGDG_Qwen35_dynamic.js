import { app } from "../../../scripts/app.js";

const NODE_NAMES = new Set([
  "VRGDG_Qwen3.5",
  "VRGDG_Qwen2.5",
  "VRGDG_GeneralVLM",
  "VRGDG_GeneralGGUF",
  "VRGDG_SuperGemmaGGUFChat",
]);

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!Object.prototype.hasOwnProperty.call(widget, "__vrgdgOriginalType")) {
    widget.__vrgdgOriginalType = widget.type;
    widget.__vrgdgOriginalComputeSize = widget.computeSize;
  }

  widget.hidden = !visible;
  widget.serialize = true;

  if (visible) {
    widget.type = widget.__vrgdgOriginalType;
    if (widget.__vrgdgOriginalComputeSize) {
      widget.computeSize = widget.__vrgdgOriginalComputeSize;
    } else {
      delete widget.computeSize;
    }
  } else {
    widget.type = widget.__vrgdgOriginalType;
    widget.computeSize = () => [0, -4];
  }
}

function asBoolean(value) {
  if (typeof value === "boolean") return value;
  const text = String(value ?? "").trim().toLowerCase();
  return text === "true" || text === "1" || text === "yes" || text === "on";
}
function isSuperGemmaNode(node) {
  return node?.comfyClass === "VRGDG_SuperGemmaGGUFChat" || node?.type === "VRGDG_SuperGemmaGGUFChat";
}

function isSuperGemmaTextOnlyModel(node) {
  if (!isSuperGemmaNode(node)) return false;
  const modelWidget = getWidget(node, "model_file");
  const value = String(modelWidget?.value || "").toLowerCase();
  return /26/.test(value) && /uncensored/.test(value) && !/vision|mmproj/.test(value);
}

function removeRefreshButton(node) {
  if (!node?.widgets) return;
  node.widgets = node.widgets.filter((w) => !(w.type === "button" && w.name === "Refresh Inputs"));
}

function ensureRefreshButton(node) {
  if (!node?.widgets) return;
  if ((node.widgets || []).some((w) => w.type === "button" && w.name === "Refresh Inputs")) return;
  node.addWidget("button", "Refresh Inputs", null, () => refreshInputs(node));
}
function refreshInputs(node) {
  const countWidget = getWidget(node, "image_count");
  if (!countWidget) return;

  const count = isSuperGemmaTextOnlyModel(node) ? 0 : Math.max(0, Math.min(24, Number(countWidget.value ?? 0)));
  const keepNames = new Set(Array.from({ length: count }, (_, i) => `image${i + 1}`));

  node.inputs = (node.inputs || []).filter((input) => {
    if (!input?.name?.match?.(/^image\d+$/)) return true;
    return keepNames.has(input.name);
  });

  for (let i = 1; i <= count; i++) {
    const name = `image${i}`;
    if (!(node.inputs || []).some((input) => input.name === name)) {
      node.addInput(name, "IMAGE");
    }
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function refreshPresetWidgets(node) {
  const presetWidget = getWidget(node, "task_preset");
  const triggerWidget = getWidget(node, "trigger_word");
  const customInstructionsWidget = getWidget(node, "custom_instructions");
  const modelWidget = getWidget(node, "model_preset");
  const hfTokenWidget = getWidget(node, "hf_token");
  if (!presetWidget) return;

  const preset = String(presetWidget.value || "");
  setWidgetVisible(triggerWidget, preset === "captioner_training");
  setWidgetVisible(customInstructionsWidget, preset === "custom");

  // GeneralVLM and GGUF variants expose hf_token. Show for likely gated/custom model choices.
  if (hfTokenWidget && modelWidget) {
    const modelValue = String(modelWidget.value || "");
    const needsToken =
      modelValue === "custom" ||
      /^meta-llama\//i.test(modelValue) ||
      /^cohereforai\//i.test(modelValue) ||
      /aya-vision/i.test(modelValue) ||
      /^jiunsong\//i.test(modelValue) ||
      /\.gguf$/i.test(modelValue);
    setWidgetVisible(hfTokenWidget, needsToken);
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function refreshSuperGemmaModelWidgets(node) {
  if (!isSuperGemmaNode(node)) return;

  const textOnly = isSuperGemmaTextOnlyModel(node);
  setWidgetVisible(getWidget(node, "mmproj_file"), !textOnly);
  setWidgetVisible(getWidget(node, "image_count"), !textOnly);
  if (textOnly) {
    removeRefreshButton(node);
  } else {
    ensureRefreshButton(node);
  }
  refreshInputs(node);

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function refreshAdvancedWidgets(node) {
  const advancedWidget = getWidget(node, "advanced");
  if (!advancedWidget) return;

  const visible = asBoolean(advancedWidget.value);
  const advancedNames = [
    "n_ctx",
    "n_gpu_layers",
    "n_threads",
    "chat_format",
    "temperature",
    "top_p",
    "max_new_tokens",
  ];

  for (const name of advancedNames) {
    const widget = getWidget(node, name);
    if (widget) {
      widget.serialize = true;
      setWidgetVisible(widget, visible);
    }
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bindCallbacks(node) {
  if (node.__vrgdgQwen35Bound) return;

  const countWidget = getWidget(node, "image_count");
  const presetWidget = getWidget(node, "task_preset");
  const modelWidget = getWidget(node, "model_preset");
  const superGemmaModelWidget = getWidget(node, "model_file");
  const advancedWidget = getWidget(node, "advanced");
  if (countWidget) {
    const oldCallback = countWidget.callback;
    countWidget.callback = function () {
      if (oldCallback) oldCallback.apply(this, arguments);
      refreshInputs(node);
    };
  }
  if (presetWidget) {
    const oldCallback = presetWidget.callback;
    presetWidget.callback = function () {
      if (oldCallback) oldCallback.apply(this, arguments);
      refreshPresetWidgets(node);
    };
  }
  if (modelWidget) {
    const oldCallback = modelWidget.callback;
    modelWidget.callback = function () {
      if (oldCallback) oldCallback.apply(this, arguments);
      refreshPresetWidgets(node);
    };
  }
  if (superGemmaModelWidget) {
    const oldCallback = superGemmaModelWidget.callback;
    superGemmaModelWidget.callback = function () {
      if (oldCallback) oldCallback.apply(this, arguments);
      refreshSuperGemmaModelWidgets(node);
    };
  }
  if (advancedWidget) {
    const oldCallback = advancedWidget.callback;
    advancedWidget.callback = function () {
      if (oldCallback) oldCallback.apply(this, arguments);
      refreshAdvancedWidgets(node);
    };
  }

  node.__vrgdgQwen35Bound = true;
}

app.registerExtension({
  name: "vrgdg.qwen.dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.has(nodeData.name)) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      bindCallbacks(this);
      if (!isSuperGemmaTextOnlyModel(this)) {
        ensureRefreshButton(this);
      }
      const refreshAll = () => {
        refreshInputs(this);
        refreshPresetWidgets(this);
        refreshSuperGemmaModelWidgets(this);
        refreshAdvancedWidgets(this);
      };
      setTimeout(refreshAll, 0);
      setTimeout(refreshAll, 100);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindCallbacks(this);
      refreshInputs(this);
      refreshPresetWidgets(this);
      refreshSuperGemmaModelWidgets(this);
      refreshAdvancedWidgets(this);
      return r;
    };
  },
});








