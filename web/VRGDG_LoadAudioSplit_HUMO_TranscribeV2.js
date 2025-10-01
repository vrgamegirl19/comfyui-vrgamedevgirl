import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_LoadAudioSplit_HUMO_TranscribeV2";
const SCENES = 16;

function doRefresh(node) {
  console.log(">>> doRefresh CALLED for", NODE_NAME);

  // 🔨 Remove unwanted "Refresh Scene Inputs" if ComfyUI injected it
  if (node.widgets) {
    node.widgets = node.widgets.filter(w => w?.label !== "Refresh Scene Inputs");
  }

  // Remove old play buttons
  if (node.widgets) {
    for (let i = node.widgets.length - 1; i >= 0; i--) {
      const w = node.widgets[i];
      if (/^play_\d+$/.test(w?.name || "")) {
        node.widgets.splice(i, 1);
      }
    }
  }

  // Insert play buttons above each context_i
  for (let i = 1; i <= SCENES; i++) {
    const ctxIndex = node.widgets.findIndex(w => w.name === `context_${i}`);

    const playButton = {
      type: "button",
      name: `play_${i}`, // match Python
      callback: () => {
        if (!node._audioPlayers) node._audioPlayers = {};
        const existing = node._audioPlayers[i];
        if (existing && !existing.paused) {
          existing.pause();
          existing.currentTime = 0;
          node._audioPlayers[i] = null;
        } else {
          const url = `/view?filename=audiochunks/audio_${i}.wav&type=input`;
          const player = new Audio(url);
          player.play();
          node._audioPlayers[i] = player;
        }
      }
    };

    if (ctxIndex !== -1) {
      node.widgets.splice(ctxIndex, 0, playButton);
    } else {
      node.widgets.push(playButton);
    }

    const ctxWidget = node.widgets.find(w => w.name === `context_${i}`);
    if (ctxWidget && (ctxWidget.value === null || ctxWidget.value === undefined)) {
      ctxWidget.value = "";
    }
  }

  // 🔨 Force exactly 16 audio outputs at all times
  for (let i = 1; i <= SCENES; i++) {
    if (!node.outputs.find(o => o.name === `audio_${i}`)) {
      node.addOutput(`audio_${i}`, "AUDIO");
    }
  }
  for (let i = (node.outputs?.length ?? 0) - 1; i >= 0; i--) {
    const out = node.outputs[i];
    if (out?.name?.startsWith?.("audio_")) {
      const idx = Number(out.name.split("_")[1]);
      if (idx > SCENES) node.removeOutput(i);
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
      setTimeout(() => doRefresh(this), 0); // force build once
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      setTimeout(() => doRefresh(this), 0); // 🔨 ensure fixes apply after reload
      return r;
    };
  },
});

