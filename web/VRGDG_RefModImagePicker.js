import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_RefModImagePicker";

function getWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name);
}

function readPaths(widget) {
  const text = String(widget?.value || "").trim();
  if (!text) return [];
  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) return parsed.map(String);
  } catch {
    // One path per line is also accepted.
  }
  return text.split(/\r?\n/).map((line) => line.trim().replace(/^"|"$/g, "")).filter(Boolean);
}

function writePaths(node, paths) {
  const widget = getWidget(node, "image_paths");
  if (!widget) return;
  widget.value = JSON.stringify(paths, null, 1);
  widget.callback?.(widget.value);
  node.title = paths.length ? `VRGDG RefMod Image Picker (${paths.length})` : "VRGDG RefMod Image Picker";
  node.setSize([Math.max(node.size?.[0] || 360, 420), Math.max(node.size?.[1] || 0, node.computeSize()[1])]);
  app.graph.setDirtyCanvas(true, true);
}

async function browse(node, button) {
  const previous = button.name;
  button.name = "Waiting for file dialog...";
  app.graph.setDirtyCanvas(true, true);
  try {
    const response = await api.fetchApi("/vrgdg/refmod/pick_images", { method: "POST" });
    const result = await response.json();
    if (!result.ok) throw new Error(result.error || "File dialog failed.");
    if (result.paths?.length) {
      const existing = readPaths(getWidget(node, "image_paths"));
      const merged = existing.concat(result.paths.filter((path) => !existing.includes(path)));
      writePaths(node, merged);
    }
  } catch (error) {
    alert(`[VRGDG RefMod] ${error.message || error}`);
  } finally {
    button.name = previous;
    app.graph.setDirtyCanvas(true, true);
  }
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".ui",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated?.apply(this, arguments);
      const node = this;
      const browseButton = this.addWidget("button", "Browse images...", null, () => browse(node, browseButton));
      this.addWidget("button", "Clear images", null, () => writePaths(node, []));
      this.setSize([Math.max(this.size?.[0] || 360, 420), this.computeSize()[1]]);
      return result;
    };
  },
});
