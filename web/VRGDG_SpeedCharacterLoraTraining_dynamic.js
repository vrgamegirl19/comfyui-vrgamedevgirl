import { app } from "../../../scripts/app.js";

const NODE_NAMES = new Set([
  "VRGDG_SpeedCharacterLoraTraining",
  "VRGDG_LTXLoraTrainChunkZImagePreset",
]);

const MAX_IMAGE_SLOTS = 20;

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

function refreshNode(node) {
  const countWidget = getWidget(node, "image_count");
  if (!countWidget) return;

  const count = Math.max(1, Math.min(MAX_IMAGE_SLOTS, Number(countWidget.value ?? 4)));
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

  for (let i = 1; i <= MAX_IMAGE_SLOTS; i++) {
    setWidgetVisible(getWidget(node, `caption_${i}`), i <= count);
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bindCountChange(node) {
  const countWidget = getWidget(node, "image_count");
  if (!countWidget || countWidget.__vrgdgSpeedCharacterPresetBound) return;

  const oldCallback = countWidget.callback;
  countWidget.callback = function () {
    if (oldCallback) oldCallback.apply(this, arguments);
    refreshNode(node);
  };

  countWidget.__vrgdgSpeedCharacterPresetBound = true;
}

app.registerExtension({
    name: "vrgdg.ltx_lora_train_speed_character.dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.has(nodeData.name)) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      bindCountChange(this);
      if (!(this.widgets || []).some((w) => w.type === "button" && w.name === "Refresh Inputs")) {
        this.addWidget("button", "Refresh Inputs", null, () => refreshNode(this));
      }
      setTimeout(() => refreshNode(this), 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindCountChange(this);
      refreshNode(this);
      return r;
    };
  },
});
