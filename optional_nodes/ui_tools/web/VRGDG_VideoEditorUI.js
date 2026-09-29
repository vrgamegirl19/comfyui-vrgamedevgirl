import { app } from "../../scripts/app.js";

const NODE_NAME = "VRGDG_VideoEditorUI";

const HIDDEN_WIDGETS = new Set([
  "selected_clip_path",
  "session_path",
  "captured_frame_path",
  "generated_t2i_prompt",
  "generated_i2v_prompt",
]);

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!Object.prototype.hasOwnProperty.call(widget, "__vrgdgVideoEditorOriginalType")) {
    widget.__vrgdgVideoEditorOriginalType = widget.type;
    widget.__vrgdgVideoEditorOriginalComputeSize = widget.computeSize;
    widget.__vrgdgVideoEditorOriginalDraw = widget.draw;
  }
  widget.serialize = true;
  widget.hidden = !visible;
  if (visible) {
    widget.type = widget.__vrgdgVideoEditorOriginalType;
    if (widget.__vrgdgVideoEditorOriginalComputeSize) widget.computeSize = widget.__vrgdgVideoEditorOriginalComputeSize;
    else delete widget.computeSize;
    if (widget.__vrgdgVideoEditorOriginalDraw) widget.draw = widget.__vrgdgVideoEditorOriginalDraw;
    else delete widget.draw;
    return;
  }
  widget.type = "hidden";
  widget.computeSize = () => [0, 0];
  widget.draw = () => {};
}

function hideInternalWidgets(node) {
  for (const widget of node?.widgets || []) {
    if (HIDDEN_WIDGETS.has(widget?.name)) setWidgetVisible(widget, false);
  }
  const width = Math.max(520, node?.size?.[0] || 520);
  const height = Math.max(120, node?.computeSize?.()[1] || 120);
  node?.setSize?.([width, height]);
  app.graph?.setDirtyCanvas?.(true, true);
}

function ensureButton(node) {
  const buttonName = "Open Video Editor";
  hideInternalWidgets(node);
  node.widgets = (node.widgets || []).filter((widget) => !(widget?.type === "button" && widget?.name === buttonName));
  const widget = node.addWidget("button", buttonName, null, () => openEditor(node));
  if (widget) widget.serialize = false;
  hideInternalWidgets(node);
}

// The editor lives in video_editor/editor.mjs and is imported on first open so it does not load with every page.
async function openEditor(...args) {
  const { openEditor } = await import("./video_editor/editor.mjs");
  return openEditor(...args);
}

app.registerExtension({
  name: "vrgdg.VideoEditorUI",

  loadedGraphNode(node) {
    if ((node?.comfyClass || node?.type) === NODE_NAME) {
      ensureButton(node);
      setTimeout(() => hideInternalWidgets(node), 0);
      setTimeout(() => hideInternalWidgets(node), 100);
    }
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    const originalOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      ensureButton(this);
      setTimeout(() => hideInternalWidgets(this), 0);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      ensureButton(this);
      setTimeout(() => hideInternalWidgets(this), 0);
      return result;
    };
  },
});
