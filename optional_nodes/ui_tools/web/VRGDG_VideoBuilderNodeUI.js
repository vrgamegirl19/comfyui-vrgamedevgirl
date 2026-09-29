import { app } from "../../scripts/app.js";

const COMFY_NODE_NAME = "VRGDG_VideoBuilderNodeCanvas";

// The canvas lives in node_canvas/node_canvas.mjs and is imported on first open so it does not load with every page.
async function openNodeCanvas(...args) {
  const { openNodeCanvas } = await import("./node_canvas/node_canvas.mjs");
  return openNodeCanvas(...args);
}

app.registerExtension({
  name: "VRGDG.VideoBuilderNodeCanvasPrototype",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== COMFY_NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onConfigure = nodeType.prototype.onConfigure;

    function ensureOpenButton(node) {
      const buttonName = "Open Node Canvas";
      node.widgets = (node.widgets || []).filter(
        (widget) => !(widget?.type === "button" && widget?.name === buttonName)
      );
      const widget = node.addWidget("button", buttonName, null, () => {
        openNodeCanvas();
      });
      if (widget) widget.serialize = false;
      node.size = [
        Math.max(node.size?.[0] || 320, 320),
        Math.max(node.size?.[1] || 120, 120),
      ];
    }

    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      ensureOpenButton(this);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      ensureOpenButton(this);
      return result;
    };
  },
});
