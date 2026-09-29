import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_LoadLatestCombinedJsonText";
const EMPTY_OPTION = "<no files found>";

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

async function refreshFiles(node, keepSelection = true) {
  const typeWidget = getWidget(node, "batch_type");
  const fileWidget = getWidget(node, "combined_json_file");
  if (!typeWidget || !fileWidget) return;

  const batchType = encodeURIComponent(String(typeWidget.value || "Text2Image"));
  const current = String(fileWidget.value || "");
  const selected = encodeURIComponent(current);
  let options = [EMPTY_OPTION];
  let resolvedFile = "";
  try {
    const res = await api.fetchApi(
      `/vrgdg/llm_batches/combined_files?batch_type=${batchType}&combined_json_file=${selected}`,
      { cache: "no-store" }
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (Array.isArray(data.files) && data.files.length) {
      options = data.files.map((v) => String(v));
    }
    resolvedFile = String(data.resolved_file || "");
  } catch (e) {
    options = [EMPTY_OPTION];
  }

  fileWidget.options = fileWidget.options || {};
  fileWidget.options.values = [...options];

  if (resolvedFile && options.includes(resolvedFile)) {
    fileWidget.value = resolvedFile;
  } else if (resolvedFile && resolvedFile !== EMPTY_OPTION) {
    fileWidget.options.values = [resolvedFile, ...options.filter((v) => v !== resolvedFile)];
    fileWidget.value = resolvedFile;
  } else {
    const canKeepCurrent =
      keepSelection &&
      current &&
      current !== EMPTY_OPTION &&
      options.includes(current);

    if (canKeepCurrent) {
      fileWidget.value = current;
    } else {
      fileWidget.value = options[0];
    }
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}


function bindTypeRefresh(node) {
  if (node.__vrgdgBatchTypeBound) return;
  const typeWidget = getWidget(node, "batch_type");
  if (!typeWidget) return;

  const old = typeWidget.callback;
  typeWidget.callback = function () {
    if (old) old.apply(this, arguments);
    refreshFiles(node, false);
  };

  node.__vrgdgBatchTypeBound = true;
}

function bindRefreshInputAutoPick(node) {
  if (node.__vrgdgRefreshAutoPickBound) return;

  const origOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function () {
    const r = origOnConnectionsChange?.apply(this, arguments);
    setTimeout(() => refreshFiles(this, true), 0);
    return r;
  };

  const origOnExecute = node.onExecute;
  node.onExecute = function () {
    setTimeout(() => refreshFiles(this, true), 0);
    return origOnExecute?.apply(this, arguments);
  };

  node.__vrgdgRefreshAutoPickBound = true;
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      bindTypeRefresh(this);
      bindRefreshInputAutoPick(this);
      if (!(this.widgets || []).some((w) => w.type === "button" && w.name === "Refresh Files")) {
        this.addWidget("button", "Refresh Files", null, () => refreshFiles(this, true));
      }
      setTimeout(() => refreshFiles(this, true), 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindTypeRefresh(this);
      bindRefreshInputAutoPick(this);
      refreshFiles(this, true);
      return r;
    };
  },
});
