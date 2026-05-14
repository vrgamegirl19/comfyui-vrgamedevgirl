import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_MultiStringConcat";
const MAX_STRING_SLOTS = 20;
const STRING_WIDGET_HEIGHT = 58;

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!Object.prototype.hasOwnProperty.call(widget, "__vrgdgOriginalType")) {
    widget.__vrgdgOriginalType = widget.type;
    widget.__vrgdgOriginalComputeSize = widget.computeSize;
  }

  widget.hidden = !visible;
  widget.serialize = true;

  if (visible) {
    widget.type = widget.__vrgdgOriginalType;
    if (String(widget.name || "").startsWith("string_")) {
      widget.computeSize = () => [0, STRING_WIDGET_HEIGHT];
    } else if (widget.__vrgdgOriginalComputeSize) {
      widget.computeSize = widget.__vrgdgOriginalComputeSize;
    } else {
      delete widget.computeSize;
    }
  } else {
    widget.type = widget.__vrgdgOriginalType;
    widget.computeSize = () => [0, -4];
  }
}

function refreshWidgets(node) {
  const countWidget = getWidget(node, "string_count");
  if (!countWidget) return;

  const count = Math.max(1, Math.min(MAX_STRING_SLOTS, Number(countWidget.value ?? 2)));
  for (let i = 1; i <= MAX_STRING_SLOTS; i++) {
    setWidgetVisible(getWidget(node, `string_${i}`), i <= count);
  }

  node.setSize([Math.max(220, node.size?.[0] || 220), node.computeSize()[1]]);
  app.graph.setDirtyCanvas(true, true);
}

function bindCountChange(node) {
  const countWidget = getWidget(node, "string_count");
  if (!countWidget || countWidget.__vrgdgMultiStringConcatBound) return;

  const oldCallback = countWidget.callback;
  countWidget.callback = function () {
    if (oldCallback) oldCallback.apply(this, arguments);
    refreshWidgets(node);
  };

  countWidget.__vrgdgMultiStringConcatBound = true;
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME + ".dynamic",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConfigure = nodeType.prototype.onConfigure;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      bindCountChange(this);
      setTimeout(() => refreshWidgets(this), 0);
      return r;
    };

    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure?.apply(this, arguments);
      bindCountChange(this);
      setTimeout(() => refreshWidgets(this), 0);
      return r;
    };
  },
});
