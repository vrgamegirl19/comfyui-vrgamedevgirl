import { app } from "../../scripts/app.js";

// The Storyboard Builder lives in .mjs modules so ComfyUI does not load it on every page;
// it is imported the first time it is opened.
const NODE_NAME = "VRGDG_StoryboardBuilderUI";
const HIDDEN_WIDGETS = new Set(["project_folder"]);

function hideInternalWidgets(node) {
  for (const widget of node.widgets || []) {
    if (!HIDDEN_WIDGETS.has(widget.name)) continue;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
  }
}

async function openStoryboardBuilder(payload) {
  const { openStoryboardBuilder } = await import("./storyboard_builder/storyboard.mjs");
  openStoryboardBuilder(payload);
}

window.VRGDGStoryboardBuilder = window.VRGDGStoryboardBuilder || {};
window.VRGDGStoryboardBuilder.open = openStoryboardBuilder;

function ensureButton(node) {
  const buttonName = "Open Storyboard Builder";
  hideInternalWidgets(node);
  node.widgets = (node.widgets || []).filter((widget) => !(widget?.type === "button" && widget?.name === buttonName));
  const widget = node.addWidget("button", buttonName, null, () => {
    const projectWidget = (node.widgets || []).find((item) => item.name === "project_folder");
    openStoryboardBuilder({ projectFolder: projectWidget?.value || "" });
  });
  if (widget) widget.serialize = false;
  hideInternalWidgets(node);
}

app.registerExtension({
  name: "vrgdg.StoryboardBuilderUI",
  loadedGraphNode(node) {
    if ((node?.comfyClass || node?.type) === NODE_NAME) ensureButton(node);
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      ensureButton(this);
      return result;
    };
    nodeType.prototype.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      ensureButton(this);
      return result;
    };
  },
});
