import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_LTX25SigmaPreset";
const CUSTOM_PRESET = "Custom — Enter Your Own Values";
const PRESET_SIGMAS = {
  "Enhance Only — Preserve Original": "0.30, 0.18, 0.06, 0.0",
  "Very Low — Subtle Detail Recovery": "0.50, 0.35, 0.15, 0.0",
  "Low — Light Enhancement": "0.65, 0.50, 0.25, 0.0",
  "Medium — Balanced Enhancement": "0.75, 0.60, 0.35, 0.0",
  "High — Noticeable Refinement": "0.85, 0.725, 0.4219, 0.0",
  "Very High — Strong Reinterpretation": "0.95, 0.80, 0.50, 0.0",
};

function getWidget(node, name) {
  return (node.widgets || []).find((item) => item.name === name);
}

function setWidgetVisible(item, visible) {
  if (!item) return;
  if (!Object.prototype.hasOwnProperty.call(item, "__vrgdgOriginalComputeSize")) {
    item.__vrgdgOriginalComputeSize = item.computeSize;
  }
  item.hidden = !visible;
  item.serialize = true;
  item.computeSize = visible ? item.__vrgdgOriginalComputeSize : () => [0, -4];
}

function refresh(node) {
  const preset = getWidget(node, "preset");
  const custom = getWidget(node, "custom_sigmas");
  const preview = getWidget(node, "sigma_values");
  if (!preset || !custom || !preview) return;

  const isCustom = preset.value === CUSTOM_PRESET;
  setWidgetVisible(custom, isCustom);
  setWidgetVisible(preview, !isCustom);

  preview.value = isCustom
    ? (custom.value || "Enter custom values above")
    : (PRESET_SIGMAS[preset.value] || preview.value);
  if (preview.inputEl) preview.inputEl.readOnly = true;
  node.setSize([Math.max(260, node.size?.[0] || 260), node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bind(node) {
  if (node.__vrgdgLTX25SigmaBound) return;
  for (const name of ["preset", "custom_sigmas"]) {
    const item = getWidget(node, name);
    if (!item) continue;
    const oldCallback = item.callback;
    item.callback = function () {
      if (oldCallback) oldCallback.apply(this, arguments);
      refresh(node);
    };
  }
  node.__vrgdgLTX25SigmaBound = true;
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
