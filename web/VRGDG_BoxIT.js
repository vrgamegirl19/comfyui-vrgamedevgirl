import { app } from "../../../scripts/app.js";

const NODE_NAME = "VRGDG_BoxIT";
const MIN_WIDTH = 160;
const MIN_HEIGHT = 120;
const DEFAULT_WIDTH = 320;
const DEFAULT_HEIGHT = 220;
const DEFAULT_OPACITY = 0.2;
const DEFAULT_RADIUS = 18;
const DEFAULT_FONT_SIZE = 20;
const DEFAULT_COLORS = {
  fill: "rgba(63, 112, 176, ALPHA)",
  border: "#7eaef1",
  title: "#f2f7ff",
};

function getWidgetValue(node, name, fallback) {
  return node.widgets?.find((widget) => widget.name === name)?.value ?? fallback;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function moveNodeToBack(node) {
  const nodes = node?.graph?._nodes;
  if (!Array.isArray(nodes)) return;
  const index = nodes.indexOf(node);
  if (index <= 0) return;
  nodes.splice(index, 1);
  nodes.unshift(node);
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

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onConfigure = nodeType.prototype.onConfigure;
    const onResize = nodeType.prototype.onResize;
    const onDrawBackground = nodeType.prototype.onDrawBackground;
    const onDrawForeground = nodeType.prototype.onDrawForeground;
    const isPointInside = nodeType.prototype.isPointInside;
    const onMouseDown = nodeType.prototype.onMouseDown;
    const onMouseUp = nodeType.prototype.onMouseUp;

    function getHeaderHeight(node) {
      const fontSize = DEFAULT_FONT_SIZE;
      return Math.max(34, fontSize + 16);
    }

    function applyHeaderLayout(node) {
      const headerHeight = Math.max(42, getHeaderHeight(node) + 6);
      node.widgets_start_y = headerHeight;
      node.widget_start_y = headerHeight;
    }

    function isResizeCornerHit(node, x, y) {
      const localX = x - (node.pos?.[0] || 0);
      const localY = y - (node.pos?.[1] || 0);
      return localX >= (node.size?.[0] || 0) - 20 && localY >= (node.size?.[1] || 0) - 20;
    }

    function getLocalPos(node, e, localPos) {
      if (Array.isArray(localPos) && localPos.length >= 2) {
        return localPos;
      }
      const canvasX = e?.canvasX ?? e?.clientX ?? node.pos?.[0] ?? 0;
      const canvasY = e?.canvasY ?? e?.clientY ?? node.pos?.[1] ?? 0;
      return [canvasX - (node.pos?.[0] || 0), canvasY - (node.pos?.[1] || 0)];
    }

    function isNodeInsideBox(boxNode, candidateNode) {
      if (!candidateNode || candidateNode === boxNode) return false;
      if (!Array.isArray(candidateNode.pos) || !Array.isArray(candidateNode.size)) return false;

      const boxLeft = boxNode.pos?.[0] || 0;
      const boxTop = boxNode.pos?.[1] || 0;
      const boxRight = boxLeft + (boxNode.size?.[0] || 0);
      const boxBottom = boxTop + (boxNode.size?.[1] || 0);

      const nodeLeft = candidateNode.pos[0];
      const nodeTop = candidateNode.pos[1];
      const nodeRight = nodeLeft + (candidateNode.size[0] || 0);
      const nodeBottom = nodeTop + (candidateNode.size[1] || 0);

      const overlaps =
        nodeRight > boxLeft &&
        nodeLeft < boxRight &&
        nodeBottom > boxTop &&
        nodeTop < boxBottom;

      return overlaps;
    }

    function captureBoxChildren(boxNode) {
      const nodes = boxNode?.graph?._nodes;
      if (!Array.isArray(nodes)) return [];
      return nodes.filter((node) => isNodeInsideBox(boxNode, node));
    }

    function moveCapturedChildren(boxNode, dx, dy) {
      if (!dx && !dy) return;
      for (const node of boxNode._boxitChildren || []) {
        if (!node || node === boxNode || !Array.isArray(node.pos)) continue;
        node.pos[0] += dx;
        node.pos[1] += dy;
      }
    }

    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);

      this.resizable = true;
      this.serialize_widgets = true;
      this.isVirtualNode = false;
      this.color = "#00000000";
      this.bgcolor = "#00000000";
      this.boxcolor = "#00000000";
      this.title = NODE_NAME;

      const width = DEFAULT_WIDTH;
      const height = DEFAULT_HEIGHT;
      this.size = [Math.max(MIN_WIDTH, width), Math.max(MIN_HEIGHT, height)];
      applyHeaderLayout(this);

      requestAnimationFrame(() => {
        moveNodeToBack(this);
        app.graph?.setDirtyCanvas?.(true, true);
      });

      return result;
    };

    nodeType.prototype.isPointInside = function (x, y, margin, skipTitle) {
      const inside = isPointInside?.call(this, x, y, margin, skipTitle);
      if (!inside) return false;

      const localX = x - (this.pos?.[0] || 0);
      const localY = y - (this.pos?.[1] || 0);
      const headerHeight = Math.max(42, getHeaderHeight(this) + 6);

      if (localY <= headerHeight) {
        return true;
      }

      if (isResizeCornerHit(this, x, y)) {
        return true;
      }

      return false;
    };

    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      const width = Number(this.size?.[0] ?? DEFAULT_WIDTH) || DEFAULT_WIDTH;
      const height = Number(this.size?.[1] ?? DEFAULT_HEIGHT) || DEFAULT_HEIGHT;
      this.size = [Math.max(MIN_WIDTH, width), Math.max(MIN_HEIGHT, height)];
      applyHeaderLayout(this);
      this._boxitLastPos = [this.pos?.[0] || 0, this.pos?.[1] || 0];
      return result;
    };

    nodeType.prototype.onMouseDown = function (e, localPos, graphCanvas) {
      const result = onMouseDown?.apply(this, arguments);
      const headerHeight = Math.max(42, getHeaderHeight(this) + 6);
      const resolvedLocalPos = getLocalPos(this, e, localPos);
      const clickedHeader = resolvedLocalPos[1] >= 0 && resolvedLocalPos[1] <= headerHeight;
      const clickedResizeCorner = isResizeCornerHit(this, e?.canvasX || 0, e?.canvasY || 0);

      if (clickedHeader && !clickedResizeCorner) {
        this._boxitDragging = true;
        this._boxitChildren = captureBoxChildren(this);
        this._boxitLastPos = [this.pos?.[0] || 0, this.pos?.[1] || 0];
      } else {
        this._boxitDragging = false;
        this._boxitChildren = [];
        this._boxitLastPos = [this.pos?.[0] || 0, this.pos?.[1] || 0];
      }

      return result;
    };

    nodeType.prototype.onMouseUp = function () {
      const result = onMouseUp?.apply(this, arguments);
      this._boxitDragging = false;
      this._boxitChildren = [];
      this._boxitLastPos = [this.pos?.[0] || 0, this.pos?.[1] || 0];
      return result;
    };

    nodeType.prototype.onResize = function (size) {
      const result = onResize?.apply(this, arguments);
      const nextWidth = Math.max(MIN_WIDTH, Math.round(size[0] || this.size?.[0] || MIN_WIDTH));
      const nextHeight = Math.max(MIN_HEIGHT, Math.round(size[1] || this.size?.[1] || MIN_HEIGHT));
      this.size = [nextWidth, nextHeight];

      applyHeaderLayout(this);

      return result;
    };

    nodeType.prototype.onDrawBackground = function (ctx) {
      onDrawBackground?.apply(this, arguments);
      moveNodeToBack(this);

      const lastPos = this._boxitLastPos || [this.pos?.[0] || 0, this.pos?.[1] || 0];
      const currentPos = [this.pos?.[0] || 0, this.pos?.[1] || 0];
      const dx = currentPos[0] - lastPos[0];
      const dy = currentPos[1] - lastPos[1];
      if (this._boxitDragging && (dx || dy)) {
        moveCapturedChildren(this, dx, dy);
        app.graph?.setDirtyCanvas?.(true, true);
      }
      this._boxitLastPos = currentPos;

      const preset = DEFAULT_COLORS;
      const opacity = DEFAULT_OPACITY;
      const radius = DEFAULT_RADIUS;

      const x = 0;
      const y = 0;
      const width = Math.max(MIN_WIDTH, this.size?.[0] || MIN_WIDTH);
      const height = Math.max(MIN_HEIGHT, this.size?.[1] || MIN_HEIGHT);

      ctx.save();

      roundedRect(ctx, x + 1, y + 1, width - 2, height - 2, radius);
      ctx.fillStyle = preset.fill.replace("ALPHA", opacity.toFixed(2));
      ctx.fill();

      ctx.lineWidth = 2;
      ctx.strokeStyle = preset.border;
      ctx.stroke();

      const headerHeight = getHeaderHeight(this);
      roundedRect(ctx, x + 1, y + 1, width - 2, headerHeight, Math.max(0, radius - 2));
      ctx.fillStyle = preset.fill.replace("ALPHA", Math.min(1, opacity + 0.12).toFixed(2));
      ctx.fill();

      ctx.restore();
    };

    nodeType.prototype.onDrawForeground = function (ctx) {
      onDrawForeground?.apply(this, arguments);

      const label = String(getWidgetValue(this, "label", "BoxIT") || "BoxIT");
      const preset = DEFAULT_COLORS;
      const fontSize = DEFAULT_FONT_SIZE;
      const headerHeight = getHeaderHeight(this);

      ctx.save();
      ctx.fillStyle = preset.title;
      ctx.font = `600 ${fontSize}px Arial`;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(label, 14, Math.floor(headerHeight / 2) + 1);
      ctx.restore();
    };
  },
});
