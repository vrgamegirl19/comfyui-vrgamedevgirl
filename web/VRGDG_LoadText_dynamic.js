import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_LoadText";
const EMPTY_OPTION = "<no files found>";

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

async function refreshFiles(node, keepSelection = true) {
  const typeWidget = getWidget(node, "text_type");
  const fileWidget = getWidget(node, "text_file");
  if (!typeWidget || !fileWidget) return;

  const category = encodeURIComponent(String(typeWidget.value || "subject1"));
  let options = [EMPTY_OPTION];

  try {
    const res = await api.fetchApi(`/vrgdg/text_files/list?category=${category}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (Array.isArray(data.files) && data.files.length) {
      options = data.files.map((v) => String(v));
    }
  } catch (e) {
    options = [EMPTY_OPTION];
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

function bindTypeRefresh(node) {
  if (node.__vrgdgTextTypeBound) return;

  const typeWidget = getWidget(node, "text_type");
  if (!typeWidget) return;

  const old = typeWidget.callback;
  typeWidget.callback = function () {
    if (old) old.apply(this, arguments);
    refreshFiles(node, false);
  };

  node.__vrgdgTextTypeBound = true;
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

      if (!(this.widgets || []).some((w) => w.type === "button" && w.name === "Refresh Files")) {
        this.addWidget("button", "Refresh Files", null, () => refreshFiles(this, true));
      }

      setTimeout(() => refreshFiles(this, true), 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindTypeRefresh(this);
      refreshFiles(this, true);
      return r;
    };
  },
});
