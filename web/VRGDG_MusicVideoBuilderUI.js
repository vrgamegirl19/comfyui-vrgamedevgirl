import { app } from "../../scripts/app.js";
import { takeBuilderRefresh } from "./music_video_builder/builder_refresh.mjs";

// The builder lives in .mjs modules so ComfyUI does not load it on every page;
// it is imported the first time the builder is opened.
const NODE_NAME = "VRGDG_MusicVideoBuilderUI";
const HIDDEN_WIDGETS = new Set(["audio_path", "project_folder", "session_path", "srt_path"]);

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!Object.prototype.hasOwnProperty.call(widget, "__vrgdgBuilderOriginalType")) {
    widget.__vrgdgBuilderOriginalType = widget.type;
    widget.__vrgdgBuilderOriginalComputeSize = widget.computeSize;
    widget.__vrgdgBuilderOriginalDraw = widget.draw;
  }
  widget.serialize = true;
  widget.hidden = !visible;
  if (visible) {
    widget.type = widget.__vrgdgBuilderOriginalType;
    if (widget.__vrgdgBuilderOriginalComputeSize) widget.computeSize = widget.__vrgdgBuilderOriginalComputeSize;
    else delete widget.computeSize;
    if (widget.__vrgdgBuilderOriginalDraw) widget.draw = widget.__vrgdgBuilderOriginalDraw;
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
  node?.setSize?.([420, 96]);
  app.graph?.setDirtyCanvas?.(true, true);
}

async function openBuilder(node, options) {
  const { openBuilder } = await import("./music_video_builder/builder.mjs");
  openBuilder(node, options);
}

function ensureButton(node) {
  const buttonName = "Open Music Video Builder";
  hideInternalWidgets(node);
  node.widgets = (node.widgets || []).filter((widget) => !(widget?.type === "button" && widget?.name === buttonName));
  const widget = node.addWidget("button", buttonName, null, () => openBuilder(node));
  if (widget) widget.serialize = false;
  hideInternalWidgets(node);
}

app.registerExtension({
  name: "vrgdg.MusicVideoBuilderUI",
  setup() {
    const resume = takeBuilderRefresh(window.sessionStorage);
    if (!resume) return;
    // Let ComfyUI restore its graph before bringing the Builder back to the front.
    window.setTimeout(() => {
      const node = app.graph?._nodes?.find((item) =>
        item.id === resume.nodeId && (item.comfyClass || item.type) === NODE_NAME) || null;
      openBuilder(node, { resume }).catch((error) => {
        console.error("VRGDG Video Builder could not resume after refresh:", error);
      });
    }, 1500);
  },
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
