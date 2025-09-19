// import { app } from "../../../scripts/app.js";

// const NODE_NAME     = "VRGDG_LoadAudioSplitUpload";
// const MAX_SCENES    = 50;
// const DEFAULT_COUNT = 2;

// function getSceneCount(node) {
//   const w = (node.widgets || []).find(w => w.name === "scene_count");
//   let val = Number(w?.value ?? DEFAULT_COUNT);
//   if (!Number.isFinite(val)) val = DEFAULT_COUNT;
//   return Math.max(1, Math.min(MAX_SCENES, val));
// }

// function doRefresh(node) {
//   const count = getSceneCount(node);

//   // Count existing audio_* outputs
//   let audioOuts = (node.outputs || []).filter(o => o?.name?.startsWith?.("audio_")).length;

//   // Add missing outputs
//   for (let i = audioOuts + 1; i <= count; i++) {
//     node.addOutput(`audio_${i}`, "AUDIO");
//   }

//   // Remove extras beyond count
//   for (let i = (node.outputs?.length ?? 0) - 1; i >= 0; i--) {
//     const out = node.outputs[i];
//     if (out?.name?.startsWith?.("audio_")) {
//       const idx = Number(out.name.split("_")[1]);
//       if (idx > count) node.removeOutput(i);
//     }
//   }

//   // Reset duration widgets
//   node.widgets = (node.widgets || []).filter(w => !w.name?.startsWith?.("duration_"));
//   for (let i = 1; i <= count; i++) {
//     node.addWidget("number", `duration_${i}`, 3.88, () => {}, { min: 0.1, step: 0.1 });
//   }

//   node.setSize([node.size[0], node.computeSize()[1]]);
//   app.graph.setDirtyCanvas(true, true);
// }

// app.registerExtension({
//   name: "vrgdg." + NODE_NAME,

//   async beforeRegisterNodeDef(nodeType, nodeData) {
//     if (nodeData.name !== NODE_NAME) return;

//     const origCreated   = nodeType.prototype.onNodeCreated;
//     const origConfigure = nodeType.prototype.onConfigure;

//     nodeType.prototype.onNodeCreated = function () {
//       const r = origCreated?.apply(this, arguments);

//       // Add refresh button if missing
//       if (!(this.widgets || []).some(w => w.type === "button" && w.label === "Refresh Scene Inputs")) {
//         this.addWidget("button", "Refresh Scene Inputs", null, () => doRefresh(this));
//       }

//       // Auto-refresh on fresh drop (default 2)
//       setTimeout(() => doRefresh(this), 50);

//       return r;
//     };

//     nodeType.prototype.onConfigure = function () {
//       const r = origConfigure?.apply(this, arguments);
//       // Re-sync on workflow load
//       doRefresh(this);
//       return r;
//     };
//   },
// });
import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_LoadAudioSplitUpload";
const MAX_SCENES = 50;
const DEFAULT_SCENES = 2;
const DEFAULT_DURATION = 3.88;
const INFINITE_DURATION = 5.00;

function doRefresh(node) {
  const sceneCountWidget = node.widgets?.find(w => w.name === "scene_count");
  const infiniteWidget = node.widgets?.find(w => w.name === "using_infinite_talk");

  // Safety check: fallback if not set
  if (sceneCountWidget && (sceneCountWidget.value == null || sceneCountWidget.value === "")) {
    sceneCountWidget.value = DEFAULT_SCENES;
  }

  const count = Math.max(1, Math.min(MAX_SCENES, Number(sceneCountWidget?.value ?? DEFAULT_SCENES)));
  const useInfinite = infiniteWidget?.value === "true";
  const defaultDuration = useInfinite ? INFINITE_DURATION : DEFAULT_DURATION;

  // Remove existing duration widgets
  node.widgets = (node.widgets || []).filter(w => !w.name?.startsWith?.("duration_"));

  // Add duration widgets for current count
  for (let i = 1; i <= count; i++) {
    node.addWidget("number", `duration_${i}`, defaultDuration, () => {}, { min: 0.1, step: 0.1 });
  }

  // Sync audio outputs
  const audioOuts = (node.outputs || []).filter(o => o?.name?.startsWith?.("audio_")).length;

  for (let i = audioOuts + 1; i <= count; i++) {
    node.addOutput(`audio_${i}`, "AUDIO");
  }

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

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this);

      // Add refresh button if missing
      if (!(this.widgets || []).some(w => w.type === "button" && w.label === "Refresh Scene Inputs")) {
        this.addWidget("button", "Refresh Scene Inputs", null, () => doRefresh(this));
      }

      // Delay refresh until after default widgets are initialized
      setTimeout(() => doRefresh(this), 50);

      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      // Do NOT run doRefresh here — or it will overwrite manually set values
      return r;
    };
  },
});
