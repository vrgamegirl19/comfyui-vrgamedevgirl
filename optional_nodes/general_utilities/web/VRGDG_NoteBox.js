import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_NoteBox";
const MIN_WIDTH = 260;
const MIN_HEIGHT = 180;
const DEFAULT_WIDTH = 380;
const DEFAULT_HEIGHT = 240;
const HEADER_HEIGHT = 36;
const WIDGETS_START_Y = 44;
const TEXT_PADDING = 14;

function getWidgetValue(node, name, fallback) {
  return node.widgets?.find((widget) => widget.name === name)?.value ?? fallback;
}

function roundedRect(ctx, x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius, width / 2, height / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function wrapText(ctx, text, maxWidth) {
  const rawLines = String(text || "").replace(/\r/g, "").split("\n");
  const wrapped = [];

  for (const rawLine of rawLines) {
    const words = rawLine.split(/\s+/).filter(Boolean);
    if (!words.length) {
      wrapped.push("");
      continue;
    }

    let line = words[0];
    for (let i = 1; i < words.length; i += 1) {
      const candidate = `${line} ${words[i]}`;
      if (ctx.measureText(candidate).width <= maxWidth) {
        line = candidate;
      } else {
        wrapped.push(line);
        line = words[i];
      }
    }
    wrapped.push(line);
  }

  return wrapped;
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onConfigure = nodeType.prototype.onConfigure;
    const onResize = nodeType.prototype.onResize;
    const onDrawBackground = nodeType.prototype.onDrawBackground;
    const onDrawForeground = nodeType.prototype.onDrawForeground;

    function requestRedraw() {
      app.graph?.setDirtyCanvas?.(true, true);
    }

    function applyLayout(node) {
      node.widgets_start_y = WIDGETS_START_Y;
      node.widget_start_y = WIDGETS_START_Y;
    }

    function bindWidgetRefresh(node) {
      for (const widget of node.widgets || []) {
        if (widget._vrgdgNoteWrapped) continue;
        const originalCallback = widget.callback;
        widget.callback = function (...args) {
          const result = originalCallback?.apply(this, args);
          requestRedraw();
          return result;
        };
        widget._vrgdgNoteWrapped = true;
      }
    }

    function getBodyTop(node, fontSize) {
      let maxBottom = WIDGETS_START_Y;

      for (const widget of node.widgets || []) {
        if (widget.last_y == null) continue;

        let widgetHeight = 24;
        if (widget.computeSize) {
          const size = widget.computeSize(node.size?.[0] || DEFAULT_WIDTH);
          if (Array.isArray(size) && size.length > 1) {
            widgetHeight = size[1];
          }
        }

        maxBottom = Math.max(maxBottom, widget.last_y + widgetHeight);
      }

      return Math.max(maxBottom + 12, HEADER_HEIGHT + fontSize + 14);
    }

    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.resizable = true;
      this.serialize_widgets = true;
      this.color = "#00000000";
      this.bgcolor = "#00000000";
      this.boxcolor = "#00000000";
      this.title = NODE_NAME;
      this.size = [
        Math.max(MIN_WIDTH, this.size?.[0] || DEFAULT_WIDTH),
        Math.max(MIN_HEIGHT, this.size?.[1] || DEFAULT_HEIGHT),
      ];
      applyLayout(this);
      bindWidgetRefresh(this);
      requestAnimationFrame(() => requestRedraw());
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.size = [
        Math.max(MIN_WIDTH, this.size?.[0] || DEFAULT_WIDTH),
        Math.max(MIN_HEIGHT, this.size?.[1] || DEFAULT_HEIGHT),
      ];
      applyLayout(this);
      return result;
    };

    nodeType.prototype.onResize = function (size) {
      const result = onResize?.apply(this, arguments);
      this.size = [
        Math.max(MIN_WIDTH, Math.round(size?.[0] || this.size?.[0] || DEFAULT_WIDTH)),
        Math.max(MIN_HEIGHT, Math.round(size?.[1] || this.size?.[1] || DEFAULT_HEIGHT)),
      ];
      applyLayout(this);
      return result;
    };

    nodeType.prototype.onDrawBackground = function (ctx) {
      onDrawBackground?.apply(this, arguments);

      const width = Math.max(MIN_WIDTH, this.size?.[0] || DEFAULT_WIDTH);
      const height = Math.max(MIN_HEIGHT, this.size?.[1] || DEFAULT_HEIGHT);

      ctx.save();

      roundedRect(ctx, 1, 1, width - 2, height - 2, 18);
      ctx.fillStyle = "rgba(222, 193, 97, 0.22)";
      ctx.fill();

      ctx.lineWidth = 2;
      ctx.strokeStyle = "#e2c161";
      ctx.stroke();

      roundedRect(ctx, 1, 1, width - 2, HEADER_HEIGHT, 16);
      ctx.fillStyle = "rgba(222, 193, 97, 0.34)";
      ctx.fill();

      ctx.restore();
    };

    nodeType.prototype.onDrawForeground = function (ctx) {
      onDrawForeground?.apply(this, arguments);

      const title = String(getWidgetValue(this, "title", "Note") || "Note");
      const note = String(getWidgetValue(this, "note", "") || "");
      const fontSize = Math.max(12, Math.min(120, Number(getWidgetValue(this, "font_size", 18)) || 18));
      const width = Math.max(MIN_WIDTH, this.size?.[0] || DEFAULT_WIDTH);
      const height = Math.max(MIN_HEIGHT, this.size?.[1] || DEFAULT_HEIGHT);
      const lineHeight = Math.round(fontSize * 1.3);
      const textWidth = Math.max(80, width - TEXT_PADDING * 2);

      ctx.save();
      ctx.fillStyle = "#fff8de";
      ctx.font = `700 ${Math.max(14, Math.min(28, fontSize))}px Arial`;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(title, TEXT_PADDING, Math.floor(HEADER_HEIGHT / 2) + 1);

      ctx.fillStyle = "#f5ecd0";
      ctx.font = `${fontSize}px Arial`;
      ctx.textBaseline = "top";

      const lines = wrapText(ctx, note, textWidth);
      let y = getBodyTop(this, fontSize);
      const maxY = height - TEXT_PADDING;
      for (const line of lines) {
        if (y + lineHeight > maxY) break;
        ctx.fillText(line, TEXT_PADDING, y);
        y += lineHeight;
      }

      ctx.restore();
    };
  },
});
