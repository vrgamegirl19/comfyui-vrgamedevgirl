import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_PromptSplitter";
const MAX_SCENES = 50;
const DEFAULT_SCENES = 2;

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    function doRefresh(node) {
      const sceneCountWidget = node.widgets?.find(w => w.name === "scene_count");

      if (sceneCountWidget && (sceneCountWidget.value == null || sceneCountWidget.value === "")) {
        sceneCountWidget.value = DEFAULT_SCENES;
      }

      const count = Math.max(1, Math.min(MAX_SCENES, Number(sceneCountWidget?.value ?? DEFAULT_SCENES)));

      let textOuts = (node.outputs || []).filter(o => o?.name?.startsWith?.("text_output_")).length;

      for (let i = textOuts + 1; i <= count; i++) {
        node.addOutput(`text_output_${i}`, "TEXT");
      }

      for (let i = (node.outputs?.length ?? 0) - 1; i >= 0; i--) {
        const out = node.outputs[i];
        if (out?.name?.startsWith?.("text_output_")) {
          const idx = Number(out.name.split("_")[2]);
          if (idx > count) node.removeOutput(i);
        }
      }

      node.setSize([node.size[0], node.computeSize()[1]]);
      app.graph.setDirtyCanvas(true, true);
    }

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this);

      if (!(this.widgets || []).some(w => w.type === "button" && w.label === "Refresh Scene Outputs")) {
        this.addWidget("button", "Refresh Scene Outputs", null, () => doRefresh(this));
      }

      // Force correct initial state
      setTimeout(() => doRefresh(this), 50);

      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      doRefresh(this);
      return r;
    };
  },
});
