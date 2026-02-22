import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_ImageSwitchMultiDynamic";

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function refreshInputs(node) {
  const countWidget = getWidget(node, "image_count");
  if (!countWidget) return;

  const count = Math.max(1, Math.min(50, Number(countWidget.value ?? 4)));

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

function bindCountChange(node) {
  const countWidget = getWidget(node, "image_count");
  if (!countWidget || countWidget.__vrgdgImageSwitchMultiBound) return;

  const oldCallback = countWidget.callback;
  countWidget.callback = function () {
    if (oldCallback) oldCallback.apply(this, arguments);
    refreshInputs(node);
  };

  countWidget.__vrgdgImageSwitchMultiBound = true;
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);

      bindCountChange(this);
      if (!(this.widgets || []).some((w) => w.type === "button" && w.name === "Refresh Inputs")) {
        this.addWidget("button", "Refresh Inputs", null, () => refreshInputs(this));
      }

      setTimeout(() => refreshInputs(this), 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindCountChange(this);
      refreshInputs(this);
      return r;
    };
  },
});
