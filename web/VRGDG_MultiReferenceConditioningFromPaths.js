import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_MultiReferenceConditioningFromPaths";

function getWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name);
}

function parsePathCount(value) {
  const text = String(value || "").trim();
  if (!text) return 0;

  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) return parsed.length;
    if (parsed && Array.isArray(parsed.image_paths)) return parsed.image_paths.length;
    if (parsed && Array.isArray(parsed.images)) return parsed.images.length;
  } catch {
    // Plain newline-separated paths are the normal UI case.
  }

  return text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).length;
}

function refreshPathSummary(node) {
  const widget = getWidget(node, "image_paths");
  const count = parsePathCount(widget?.value);

  if (widget?.inputEl) {
    widget.inputEl.style.minHeight = "130px";
    widget.inputEl.style.whiteSpace = "pre";
  }

  node.title = count
    ? `VRGDG UI Multi Reference Conditioning (${count} image${count === 1 ? "" : "s"})`
    : "VRGDG UI Multi Reference Conditioning";

  node.setSize([Math.max(node.size?.[0] || 320, 420), node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bindPathWidget(node) {
  const widget = getWidget(node, "image_paths");
  if (!widget || widget.__vrgdgPathSummaryBound) return;

  const oldCallback = widget.callback;
  widget.callback = function () {
    oldCallback?.apply(this, arguments);
    refreshPathSummary(node);
  };

  widget.__vrgdgPathSummaryBound = true;
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".ui",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated?.apply(this, arguments);
      bindPathWidget(this);

      if (!(this.widgets || []).some((widget) => widget.type === "button" && widget.name === "Clear Image Paths")) {
        this.addWidget("button", "Clear Image Paths", null, () => {
          const widget = getWidget(this, "image_paths");
          if (widget) widget.value = "";
          refreshPathSummary(this);
        });
      }

      setTimeout(() => refreshPathSummary(this), 0);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = origOnConfigure?.apply(this, arguments);
      bindPathWidget(this);
      setTimeout(() => refreshPathSummary(this), 0);
      return result;
    };
  },
});
