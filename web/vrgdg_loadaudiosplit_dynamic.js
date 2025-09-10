import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_LoadAudioSplitDynamic";

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure   = nodeType.prototype.onConfigure;

    function doRefresh(node) {
      const sceneCountWidget = node.widgets?.find(w => w.name === "scene_count");
      if (!sceneCountWidget) return;

      const count = Math.max(1, Math.min(50, Number(sceneCountWidget.value ?? 1)));

      // keep any existing duration values
      const currentDur = new Map();
      (node.widgets || []).forEach(w => {
        if (w.name?.startsWith("duration_")) currentDur.set(w.name, w.value);
      });

      // remove extra duration widgets (but keep first 'count')
      const kept = new Set([...Array(count)].map((_, i) => `duration_${i+1}`));
      node.widgets = (node.widgets || []).filter(w => {
        if (!w.name?.startsWith?.("duration_")) return true;
        return kept.has(w.name);
      });

      // add missing duration widgets
      for (let i = 1; i <= count; i++) {
        const name = `duration_${i}`;
        if (!node.widgets.find(w => w.name === name)) {
          const v = currentDur.get(name) ?? 3.0;
          node.addWidget("number", name, v, () => {}, { min: 0.1, step: 0.1 });
        }
      }

      // adjust outputs without destroying existing connections for first N
      // add missing
      let audioOuts = (node.outputs || []).filter(o => o?.name?.startsWith?.("audio_")).length;
      for (let i = audioOuts + 1; i <= count; i++) node.addOutput(`audio_${i}`, "AUDIO");
      // remove extras from the end
      for (let i = (node.outputs?.length ?? 0) - 1; i >= 0; i--) {
        const out = node.outputs[i];
        if (out?.name?.startsWith?.("audio_")) {
          const idx = Number(out.name.split("_")[1]);
          if (idx > count) node.removeOutput(i);
        }
      }

      node.setSize([node.size[0], node.computeSize()[1]]);
      app.graph.setDirtyCanvas(true, true);
    }

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this);

      // add button
      this.addWidget("button", "Refresh Scene Inputs", null, () => doRefresh(this));

      // bind change handler so manual scene_count edits reflow safely
      const sc = this.widgets?.find(w => w.name === "scene_count");
      if (sc) sc.callback = () => doRefresh(this);

      // IMPORTANT: don't refresh now; wait for onConfigure to load saved values
      return r;
    };

    nodeType.prototype.onConfigure = function (o) {
      const r = origOnConfigure?.apply(this, arguments);
      // now values from the workflow are applied; safe to reflow
      doRefresh(this);
      return r;
    };
  },
});
