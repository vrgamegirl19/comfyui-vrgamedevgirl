import { app } from "../../../scripts/app.js";

const NODE_NAMES = new Set(["VRGDG_Qwen3.5", "VRGDG_Qwen2.5", "VRGDG_GeneralVLM"]);

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!Object.prototype.hasOwnProperty.call(widget, "__vrgdgOriginalType")) {
    widget.__vrgdgOriginalType = widget.type;
    widget.__vrgdgOriginalComputeSize = widget.computeSize;
  }

  if (visible) {
    widget.type = widget.__vrgdgOriginalType;
    if (widget.__vrgdgOriginalComputeSize) {
      widget.computeSize = widget.__vrgdgOriginalComputeSize;
    } else {
      delete widget.computeSize;
    }
  } else {
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
  }
}

function refreshInputs(node) {
  const countWidget = getWidget(node, "image_count");
  if (!countWidget) return;

  const count = Math.max(0, Math.min(24, Number(countWidget.value ?? 0)));
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

  // Only GeneralVLM has hf_token. Show for likely gated/custom model choices.
  if (hfTokenWidget && modelWidget) {
    const modelValue = String(modelWidget.value || "");
    const needsToken =
      modelValue === "custom" ||
      /^meta-llama\//i.test(modelValue) ||
      /^cohereforai\//i.test(modelValue) ||
      /aya-vision/i.test(modelValue);
    setWidgetVisible(hfTokenWidget, needsToken);
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bindCallbacks(node) {
  if (node.__vrgdgQwen35Bound) return;

  const countWidget = getWidget(node, "image_count");
  const presetWidget = getWidget(node, "task_preset");
  const modelWidget = getWidget(node, "model_preset");
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
      if (!(this.widgets || []).some((w) => w.type === "button" && w.name === "Refresh Inputs")) {
        this.addWidget("button", "Refresh Inputs", null, () => refreshInputs(this));
      }
      setTimeout(() => {
        refreshInputs(this);
        refreshPresetWidgets(this);
      }, 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindCallbacks(this);
      refreshInputs(this);
      refreshPresetWidgets(this);
      return r;
    };
  },
});
