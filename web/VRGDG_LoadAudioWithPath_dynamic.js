import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_LoadAudioWithPath";
const EMPTY_OPTION = "<no audio files found>";

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

async function fetchAudioFiles() {
  try {
    const res = await api.fetchApi("/vrgdg/audio/list", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (Array.isArray(data.files) && data.files.length) {
      return data.files.map((v) => String(v));
    }
  } catch (e) {
    // fall through
  }
  return [EMPTY_OPTION];
}

async function refreshAudioWidget(node, preferred = null) {
  const audioWidget = getWidget(node, "audio");
  if (!audioWidget) return;

  const options = await fetchAudioFiles();
  const current = String(audioWidget.value || "");
  const selected = preferred && options.includes(preferred) ? preferred : current;

  audioWidget.options = audioWidget.options || {};
  audioWidget.options.values = [...options];
  audioWidget.value = options.includes(selected) ? selected : options[0];

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

async function uploadAudioFile(file) {
  const form = new FormData();
  form.append("audio", file);

  const res = await api.fetchApi("/vrgdg/audio/upload", {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    let message = `Upload failed (${res.status})`;
    try {
      const data = await res.json();
      if (data?.error) message = String(data.error);
    } catch (e) {
      // ignore parse errors
    }
    throw new Error(message);
  }
  return await res.json();
}

function attachUploadButton(node) {
  if ((node.widgets || []).some((w) => w.type === "button" && w.name === "Upload Audio")) {
    return;
  }

  node.addWidget("button", "Upload Audio", null, async () => {
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = "audio/*,video/*";
    picker.style.display = "none";
    document.body.appendChild(picker);

    picker.addEventListener("change", async () => {
      const file = picker.files?.[0];
      document.body.removeChild(picker);
      if (!file) return;

      try {
        const data = await uploadAudioFile(file);
        const uploadedName = String(data?.name || "");
        await refreshAudioWidget(node, uploadedName);
      } catch (e) {
        alert(`[VRGDG] ${String(e?.message || e)}`);
      }
    });

    picker.click();
  });
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      attachUploadButton(this);
      setTimeout(() => refreshAudioWidget(this, null), 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      attachUploadButton(this);
      refreshAudioWidget(this, null);
      return r;
    };
  },
});
