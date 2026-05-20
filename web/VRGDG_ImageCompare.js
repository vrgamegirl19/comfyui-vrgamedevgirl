import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_ImageCompare";
const MIN_WIDTH = 420;
const MIN_HEIGHT = 360;
const PADDING = 10;

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function getWidgetValue(node, name, fallback) {
  return getWidget(node, name)?.value ?? fallback;
}

function setWidgetValue(node, name, value) {
  const widget = getWidget(node, name);
  if (!widget) return;
  widget.value = value;
  widget.callback?.(value);
}

function requestRedraw() {
  app.graph?.setDirtyCanvas?.(true, true);
}

function imageUrl(info) {
  const params = new URLSearchParams();
  params.set("filename", info.filename);
  params.set("type", info.type || "output");
  if (info.subfolder) params.set("subfolder", info.subfolder);
  params.set("rand", Math.random().toString());
  return api.apiURL(`/view?${params.toString()}`);
}

function loadPreviewImages(node, infos) {
  const ordered = [null, null];
  for (const info of infos || []) {
    const role = info.compare_role;
    if (role === "a" && !ordered[0]) ordered[0] = info;
    if (role === "b" && !ordered[1]) ordered[1] = info;
  }

  if (!ordered[0]) ordered[0] = infos?.[0] || null;
  if (!ordered[1]) ordered[1] = infos?.[1] || null;

  node._vrgdgCompareImages = ordered.map((info) => {
    if (!info) return null;
    const img = new Image();
    img.onload = requestRedraw;
    img.onerror = requestRedraw;
    img.src = imageUrl(info);
    return img;
  });

  requestRedraw();
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
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

function drawImageContain(ctx, img, x, y, width, height) {
  if (!img || !img.complete || !img.naturalWidth || !img.naturalHeight) return false;

  const scale = Math.min(width / img.naturalWidth, height / img.naturalHeight);
  const drawWidth = img.naturalWidth * scale;
  const drawHeight = img.naturalHeight * scale;
  const drawX = x + (width - drawWidth) / 2;
  const drawY = y + (height - drawHeight) / 2;

  ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);
  return true;
}

function drawLabel(ctx, text, x, y) {
  ctx.save();
  ctx.font = "600 12px Arial";
  const width = Math.ceil(ctx.measureText(text).width) + 14;
  drawRoundedRect(ctx, x, y, width, 22, 5);
  ctx.fillStyle = "rgba(0, 0, 0, 0.68)";
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + 7, y + 11);
  ctx.restore();
}

function drawEmptyState(ctx, rect) {
  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, 0.68)";
  ctx.font = "13px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Run the node to load comparison previews", rect.x + rect.w / 2, rect.y + rect.h / 2);
  ctx.restore();
}

function clipRect(ctx, rect) {
  drawRoundedRect(ctx, rect.x, rect.y, rect.w, rect.h, 8);
  ctx.clip();
}

function drawCompare(node, ctx) {
  const top = Math.max(118, (node.widgets_start_y || 0) + (node.widgets?.length || 0) * 24 + 8);
  const rect = {
    x: PADDING,
    y: top,
    w: Math.max(1, (node.size?.[0] || MIN_WIDTH) - PADDING * 2),
    h: Math.max(1, (node.size?.[1] || MIN_HEIGHT) - top - PADDING),
  };
  node._vrgdgCompareRect = rect;

  ctx.save();
  drawRoundedRect(ctx, rect.x, rect.y, rect.w, rect.h, 8);
  ctx.fillStyle = "#101216";
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.16)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();

  const [imgA, imgB] = node._vrgdgCompareImages || [];
  if (!imgA || !imgB) {
    drawEmptyState(ctx, rect);
    return;
  }

  const mode = getWidgetValue(node, "mode", "slider");
  const showLabels = !!getWidgetValue(node, "show_labels", true);

  ctx.save();
  clipRect(ctx, rect);

  if (mode === "side_by_side") {
    const gap = 6;
    const half = (rect.w - gap) / 2;
    const left = { x: rect.x, y: rect.y, w: half, h: rect.h };
    const right = { x: rect.x + half + gap, y: rect.y, w: half, h: rect.h };
    drawImageContain(ctx, imgA, left.x, left.y, left.w, left.h);
    drawImageContain(ctx, imgB, right.x, right.y, right.w, right.h);
    ctx.fillStyle = "rgba(255, 255, 255, 0.16)";
    ctx.fillRect(rect.x + half, rect.y, gap, rect.h);
    if (showLabels) {
      drawLabel(ctx, "A", left.x + 8, left.y + 8);
      drawLabel(ctx, "B", right.x + 8, right.y + 8);
    }
  } else if (mode === "overlay") {
    const opacity = Number(getWidgetValue(node, "overlay_opacity", 0.5));
    drawImageContain(ctx, imgA, rect.x, rect.y, rect.w, rect.h);
    ctx.globalAlpha = Math.max(0, Math.min(1, opacity));
    drawImageContain(ctx, imgB, rect.x, rect.y, rect.w, rect.h);
    ctx.globalAlpha = 1;
    if (showLabels) drawLabel(ctx, `B ${(opacity * 100).toFixed(0)}%`, rect.x + 8, rect.y + 8);
  } else if (mode === "difference") {
    drawImageContain(ctx, imgA, rect.x, rect.y, rect.w, rect.h);
    ctx.globalCompositeOperation = "difference";
    drawImageContain(ctx, imgB, rect.x, rect.y, rect.w, rect.h);
    ctx.globalCompositeOperation = "source-over";
    if (showLabels) drawLabel(ctx, "Difference", rect.x + 8, rect.y + 8);
  } else if (mode === "blink") {
    const speed = Math.max(0.1, Number(getWidgetValue(node, "blink_speed", 1.0)));
    const phase = Math.floor((Date.now() / 1000) * speed * 2) % 2;
    drawImageContain(ctx, phase ? imgB : imgA, rect.x, rect.y, rect.w, rect.h);
    if (showLabels) drawLabel(ctx, phase ? "B" : "A", rect.x + 8, rect.y + 8);
    requestAnimationFrame(requestRedraw);
  } else {
    const slider = Math.max(0, Math.min(1, Number(getWidgetValue(node, "slider_position", 0.5))));
    const splitX = rect.x + rect.w * slider;

    drawImageContain(ctx, imgA, rect.x, rect.y, rect.w, rect.h);
    ctx.save();
    ctx.beginPath();
    ctx.rect(splitX, rect.y, rect.x + rect.w - splitX, rect.h);
    ctx.clip();
    drawImageContain(ctx, imgB, rect.x, rect.y, rect.w, rect.h);
    ctx.restore();

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(splitX, rect.y);
    ctx.lineTo(splitX, rect.y + rect.h);
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(splitX, rect.y + rect.h / 2, 6, 0, Math.PI * 2);
    ctx.fill();

    if (showLabels) {
      drawLabel(ctx, "A", rect.x + 8, rect.y + 8);
      drawLabel(ctx, "B", rect.x + rect.w - 28, rect.y + 8);
    }
  }

  ctx.restore();
}

function pointInRect(pos, rect) {
  return pos && rect && pos[0] >= rect.x && pos[0] <= rect.x + rect.w && pos[1] >= rect.y && pos[1] <= rect.y + rect.h;
}

function updateSliderFromMouse(node, localPos) {
  const rect = node._vrgdgCompareRect;
  if (!rect || !localPos) return;
  const value = Math.max(0, Math.min(1, (localPos[0] - rect.x) / rect.w));
  setWidgetValue(node, "slider_position", Math.round(value * 100) / 100);
  requestRedraw();
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onConfigure = nodeType.prototype.onConfigure;
    const onExecuted = nodeType.prototype.onExecuted;
    const onResize = nodeType.prototype.onResize;
    const onDrawForeground = nodeType.prototype.onDrawForeground;
    const onMouseDown = nodeType.prototype.onMouseDown;
    const onMouseMove = nodeType.prototype.onMouseMove;
    const onMouseUp = nodeType.prototype.onMouseUp;

    function bindWidgets(node) {
      for (const widget of node.widgets || []) {
        if (widget._vrgdgCompareBound) continue;
        const callback = widget.callback;
        widget.callback = function () {
          const result = callback?.apply(this, arguments);
          requestRedraw();
          return result;
        };
        widget._vrgdgCompareBound = true;
      }
    }

    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.resizable = true;
      this.serialize_widgets = true;
      this.size = [Math.max(MIN_WIDTH, this.size?.[0] || MIN_WIDTH), Math.max(MIN_HEIGHT, this.size?.[1] || MIN_HEIGHT)];
      bindWidgets(this);
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.size = [Math.max(MIN_WIDTH, this.size?.[0] || MIN_WIDTH), Math.max(MIN_HEIGHT, this.size?.[1] || MIN_HEIGHT)];
      bindWidgets(this);
      return result;
    };

    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      loadPreviewImages(this, message?.compare_images || []);
    };

    nodeType.prototype.onResize = function (size) {
      const result = onResize?.apply(this, arguments);
      this.size = [Math.max(MIN_WIDTH, size?.[0] || MIN_WIDTH), Math.max(MIN_HEIGHT, size?.[1] || MIN_HEIGHT)];
      return result;
    };

    nodeType.prototype.onDrawForeground = function (ctx) {
      onDrawForeground?.apply(this, arguments);
      drawCompare(this, ctx);
    };

    nodeType.prototype.onMouseDown = function (e, localPos) {
      const result = onMouseDown?.apply(this, arguments);
      if (getWidgetValue(this, "mode", "slider") === "slider" && pointInRect(localPos, this._vrgdgCompareRect)) {
        this._vrgdgCompareDragging = true;
        updateSliderFromMouse(this, localPos);
        return true;
      }
      return result;
    };

    nodeType.prototype.onMouseMove = function (e, localPos) {
      const result = onMouseMove?.apply(this, arguments);
      if (this._vrgdgCompareDragging) {
        updateSliderFromMouse(this, localPos);
        return true;
      }
      return result;
    };

    nodeType.prototype.onMouseUp = function () {
      const result = onMouseUp?.apply(this, arguments);
      this._vrgdgCompareDragging = false;
      return result;
    };
  },
});
