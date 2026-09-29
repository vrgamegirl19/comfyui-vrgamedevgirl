import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_LTXICIngredientsGrid";
const MAX_IMAGES = 24;

function getWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name);
}

function refreshInputs(node) {
  const countWidget = getWidget(node, "image_count");
  if (!countWidget) return;

  const count = Math.max(1, Math.min(MAX_IMAGES, Number(countWidget.value ?? 6)));
  const keepNames = new Set(Array.from({ length: count }, (_, index) => `image${index + 1}`));

  node.inputs = (node.inputs || []).filter((input) => {
    if (!input?.name?.match?.(/^image\d+$/)) return true;
    return keepNames.has(input.name);
  });

  for (let index = 1; index <= count; index += 1) {
    const name = `image${index}`;
    if (!(node.inputs || []).some((input) => input.name === name)) {
      node.addInput(name, "IMAGE");
    }
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bindCountChange(node) {
  const countWidget = getWidget(node, "image_count");
  if (!countWidget || countWidget.__vrgdgLtxIcIngredientsBound) return;

  const oldCallback = countWidget.callback;
  countWidget.callback = function () {
    oldCallback?.apply(this, arguments);
    refreshInputs(node);
  };

  countWidget.__vrgdgLtxIcIngredientsBound = true;
}

app.registerExtension({
  name: "vrgdg.ltx_ic_ingredients_grid.dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated?.apply(this, arguments);
      bindCountChange(this);

      if (!(this.widgets || []).some((widget) => widget.type === "button" && widget.name === "Refresh Inputs")) {
        this.addWidget("button", "Refresh Inputs", null, () => refreshInputs(this));
      }

      setTimeout(() => refreshInputs(this), 0);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = origOnConfigure?.apply(this, arguments);
      bindCountChange(this);
      refreshInputs(this);
      return result;
    };
  },
});
