import { app } from "../../../scripts/app.js";

const NODE_NAMES = new Set([
  "VRGDG_OptionalMultiLoraModelOnly",
]);
const OPTIONAL_MODEL_ONLY_NODE = "VRGDG_OptionalMultiLoraModelOnly";
const MAX_LORA_SLOTS = 20;
const OPTIONAL_OUTPUTS = [
  ["first_pass_model", "MODEL"],
  ["second_pass_model", "MODEL"],
  ["lora_names", "STRING"],
];

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function asBoolean(value) {
  if (typeof value === "boolean") return value;
  const text = String(value ?? "").trim().toLowerCase();
  return text === "true" || text === "1" || text === "yes" || text === "on";
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

function isOptionalModelOnlyNode(node) {
  return node?.comfyClass === OPTIONAL_MODEL_ONLY_NODE || node?.type === OPTIONAL_MODEL_ONLY_NODE;
}

function normalizeOptionalOutputs(node) {
  if (!isOptionalModelOnlyNode(node) || !Array.isArray(node.outputs)) return;

  while (node.outputs.length > OPTIONAL_OUTPUTS.length) {
    if (typeof node.removeOutput === "function") {
      node.removeOutput(node.outputs.length - 1);
    } else {
      node.outputs.pop();
    }
  }

  for (let i = 0; i < OPTIONAL_OUTPUTS.length; i++) {
    const [name, type] = OPTIONAL_OUTPUTS[i];
    if (!node.outputs[i] && typeof node.addOutput === "function") {
      node.addOutput(name, type);
    }
    if (node.outputs[i]) {
      node.outputs[i].name = name;
      node.outputs[i].localized_name = name;
      node.outputs[i].type = type;
    }
  }
}

function refreshWidgets(node) {
  const countWidget = getWidget(node, "lora_count");
  if (!countWidget) return;

  normalizeOptionalOutputs(node);

  const isOptionalNode = isOptionalModelOnlyNode(node);
  const useCustomWidget = getWidget(node, "use_custom_loras");
  const useCustom = !isOptionalNode || asBoolean(useCustomWidget?.value);
  const minCount = isOptionalNode ? 0 : 1;
  const fallbackCount = isOptionalNode ? 0 : 4;
  const count = useCustom
    ? Math.max(minCount, Math.min(MAX_LORA_SLOTS, Number(countWidget.value ?? fallbackCount)))
    : 0;

  setWidgetVisible(countWidget, useCustom);
  setWidgetVisible(getWidget(node, "ltx_two_pass_mode"), useCustom);

  for (let i = 1; i <= MAX_LORA_SLOTS; i++) {
    const visible = useCustom && i <= count;
    setWidgetVisible(getWidget(node, `lora_${i}`), visible);
    setWidgetVisible(getWidget(node, `strength_${i}`), visible);
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bindCountChange(node) {
  const countWidget = getWidget(node, "lora_count");
  const useCustomWidget = getWidget(node, "use_custom_loras");

  if (countWidget && !countWidget.__vrgdgLoraSwitchBound) {
    const oldCallback = countWidget.callback;
    countWidget.callback = function () {
      if (oldCallback) oldCallback.apply(this, arguments);
      refreshWidgets(node);
    };

    countWidget.__vrgdgLoraSwitchBound = true;
  }

  if (useCustomWidget && !useCustomWidget.__vrgdgLoraSwitchBound) {
    const oldCallback = useCustomWidget.callback;
    useCustomWidget.callback = function () {
      if (oldCallback) oldCallback.apply(this, arguments);
      refreshWidgets(node);
    };

    useCustomWidget.__vrgdgLoraSwitchBound = true;
  }
}

app.registerExtension({
  name: "vrgdg.optional_lora_model_only.dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.has(nodeData.name)) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      bindCountChange(this);
      setTimeout(() => refreshWidgets(this), 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindCountChange(this);
      refreshWidgets(this);
      return r;
    };
  },
});
