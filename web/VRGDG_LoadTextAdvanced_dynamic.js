import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_LoadTextAdvanced";
const EMPTY_FILE_OPTION = "<no files found>";
const EMPTY_FOLDER_OPTION = "<no folders found>";

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

async function refreshFolders(node, keepSelection = true) {
  const folderWidget = getWidget(node, "folder_name");
  if (!folderWidget) return;

  let options = [EMPTY_FOLDER_OPTION];
  try {
    const res = await api.fetchApi("/vrgdg/text_files/folders", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (Array.isArray(data.folders) && data.folders.length) {
      options = data.folders.map((v) => String(v));
    }
  } catch (e) {
    options = [EMPTY_FOLDER_OPTION];
  }

  const current = String(folderWidget.value || "");
  folderWidget.options = folderWidget.options || {};
  folderWidget.options.values = [...options];

  if (keepSelection && options.includes(current)) {
    folderWidget.value = current;
  } else {
    folderWidget.value = options[0];
  }
}

async function refreshFiles(node, keepSelection = true) {
  const folderWidget = getWidget(node, "folder_name");
  const mostRecentWidget = getWidget(node, "use_most_recent");
  const fileWidget = getWidget(node, "text_file");
  if (!folderWidget || !mostRecentWidget || !fileWidget) return;

  const folder = encodeURIComponent(String(folderWidget.value || ""));
  const useMostRecent = Boolean(mostRecentWidget.value);

  let options = [EMPTY_FILE_OPTION];
  try {
    const url = `/vrgdg/text_files/files?folder=${folder}&use_most_recent=${useMostRecent ? "true" : "false"}`;
    const res = await api.fetchApi(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (Array.isArray(data.files) && data.files.length) {
      options = data.files.map((v) => String(v));
    }
  } catch (e) {
    options = [EMPTY_FILE_OPTION];
  }

  const current = String(fileWidget.value || "");
  fileWidget.options = fileWidget.options || {};
  fileWidget.options.values = [...options];

  if (keepSelection && options.includes(current)) {
    fileWidget.value = current;
  } else {
    fileWidget.value = options[0];
  }

  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bindRefreshCallbacks(node) {
  if (node.__vrgdgLoadTextAdvancedBound) return;

  const folderWidget = getWidget(node, "folder_name");
  const mostRecentWidget = getWidget(node, "use_most_recent");

  if (folderWidget) {
    const oldFolder = folderWidget.callback;
    folderWidget.callback = function () {
      if (oldFolder) oldFolder.apply(this, arguments);
      refreshFiles(node, false);
    };
  }

  if (mostRecentWidget) {
    const oldMostRecent = mostRecentWidget.callback;
    mostRecentWidget.callback = function () {
      if (oldMostRecent) oldMostRecent.apply(this, arguments);
      refreshFiles(node, false);
    };
  }

  node.__vrgdgLoadTextAdvancedBound = true;
}

async function fullRefresh(node, keepSelection = true) {
  await refreshFolders(node, keepSelection);
  await refreshFiles(node, keepSelection);
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      bindRefreshCallbacks(this);

      if (!(this.widgets || []).some((w) => w.type === "button" && w.name === "Refresh List")) {
        this.addWidget("button", "Refresh List", null, () => fullRefresh(this, true));
      }

      setTimeout(() => fullRefresh(this, true), 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindRefreshCallbacks(this);
      fullRefresh(this, true);
      return r;
    };
  },
});
