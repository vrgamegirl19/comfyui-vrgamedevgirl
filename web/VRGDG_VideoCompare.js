import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDG_VideoCompareSlider";
const MIN_WIDTH = 560;
const MIN_HEIGHT = 560;

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function widgetValue(node, name, fallback) {
  return getWidget(node, name)?.value ?? fallback;
}

function setWidgetValue(node, name, value) {
  const widget = getWidget(node, name);
  if (!widget) return;
  widget.value = value;
  widget.callback?.(value);
}

function videoUrl(path) {
  return api.apiURL(`/vrgdg/video_editor/video?path=${encodeURIComponent(path)}&v=${Date.now()}`);
}

function formatTime(value) {
  const seconds = Math.max(0, Number(value || 0));
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function createButton(label, title = "") {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.title = title;
  button.style.cssText = [
    "min-width:38px",
    "height:30px",
    "padding:4px 10px",
    "border:1px solid #52525b",
    "border-radius:5px",
    "background:#27272a",
    "color:#f4f4f5",
    "font:700 12px Arial,sans-serif",
    "cursor:pointer",
  ].join(";");
  return button;
}

function createCompareUI(node) {
  const root = document.createElement("div");
  root.style.cssText = [
    "width:100%",
    "height:430px",
    "display:flex",
    "flex-direction:column",
    "gap:7px",
    "box-sizing:border-box",
    "padding:6px",
    "background:#18181b",
    "color:#f4f4f5",
    "font-family:Arial,sans-serif",
    "user-select:none",
    "overflow:hidden",
  ].join(";");

  const stage = document.createElement("div");
  stage.style.cssText = [
    "position:relative",
    "flex:1 1 auto",
    "min-height:260px",
    "overflow:hidden",
    "border:1px solid #3f3f46",
    "border-radius:7px",
    "background:#050505",
    "touch-action:none",
    "cursor:ew-resize",
  ].join(";");

  const beforeVideo = document.createElement("video");
  const afterVideo = document.createElement("video");
  for (const video of [beforeVideo, afterVideo]) {
    video.preload = "metadata";
    video.playsInline = true;
    video.disablePictureInPicture = true;
    video.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#050505;pointer-events:none;";
  }
  afterVideo.muted = true;

  const afterClip = document.createElement("div");
  afterClip.style.cssText = "position:absolute;inset:0;overflow:hidden;pointer-events:none;";
  afterClip.append(afterVideo);

  const divider = document.createElement("div");
  divider.style.cssText = [
    "position:absolute",
    "top:0",
    "bottom:0",
    "width:2px",
    "background:#fff",
    "box-shadow:0 0 0 1px rgba(0,0,0,.7),0 0 14px rgba(255,255,255,.35)",
    "transform:translateX(-1px)",
    "pointer-events:none",
  ].join(";");

  const handle = document.createElement("div");
  handle.textContent = "↔";
  handle.style.cssText = [
    "position:absolute",
    "top:50%",
    "width:34px",
    "height:34px",
    "display:flex",
    "align-items:center",
    "justify-content:center",
    "border:1px solid rgba(0,0,0,.55)",
    "border-radius:999px",
    "background:rgba(255,255,255,.94)",
    "color:#111",
    "font:900 17px Arial,sans-serif",
    "box-shadow:0 3px 14px rgba(0,0,0,.55)",
    "transform:translate(-50%,-50%)",
    "pointer-events:none",
  ].join(";");

  const beforeLabel = document.createElement("div");
  const afterLabel = document.createElement("div");
  for (const label of [beforeLabel, afterLabel]) {
    label.style.cssText = [
      "position:absolute",
      "top:9px",
      "padding:5px 8px",
      "border-radius:5px",
      "background:rgba(0,0,0,.68)",
      "color:#fff",
      "font:900 11px Arial,sans-serif",
      "pointer-events:none",
    ].join(";");
  }
  beforeLabel.style.left = "9px";
  afterLabel.style.right = "9px";

  const empty = document.createElement("div");
  empty.textContent = "Run the node to load the before and after videos";
  empty.style.cssText = [
    "position:absolute",
    "inset:0",
    "display:flex",
    "align-items:center",
    "justify-content:center",
    "padding:20px",
    "color:#a1a1aa",
    "font:700 12px Arial,sans-serif",
    "text-align:center",
    "pointer-events:none",
  ].join(";");

  stage.append(beforeVideo, afterClip, divider, handle, beforeLabel, afterLabel, empty);

  const controls = document.createElement("div");
  controls.style.cssText = "display:grid;grid-template-columns:auto auto minmax(80px,1fr) auto auto;gap:7px;align-items:center;";
  const playButton = createButton("▶", "Play or pause both videos");
  const restartButton = createButton("↺", "Restart both videos");
  const scrubber = document.createElement("input");
  scrubber.type = "range";
  scrubber.min = "0";
  scrubber.max = "0";
  scrubber.step = "0.01";
  scrubber.value = "0";
  scrubber.style.cssText = "width:100%;accent-color:#22d3ee;cursor:pointer;";
  const timeLabel = document.createElement("div");
  timeLabel.textContent = "00:00.00 / 00:00.00";
  timeLabel.style.cssText = "font:700 11px Arial,sans-serif;color:#67e8f9;white-space:nowrap;font-variant-numeric:tabular-nums;";
  const muteButton = createButton("🔇", "Mute or unmute before-video audio");
  controls.append(playButton, restartButton, scrubber, timeLabel, muteButton);

  const wipeRow = document.createElement("div");
  wipeRow.style.cssText = "display:grid;grid-template-columns:auto minmax(80px,1fr) auto;gap:8px;align-items:center;";
  const wipeTitle = document.createElement("div");
  wipeTitle.textContent = "Before / After";
  wipeTitle.style.cssText = "font:800 11px Arial,sans-serif;color:#d4d4d8;white-space:nowrap;";
  const wipeSlider = document.createElement("input");
  wipeSlider.type = "range";
  wipeSlider.min = "0";
  wipeSlider.max = "1";
  wipeSlider.step = "0.01";
  wipeSlider.style.cssText = "width:100%;accent-color:#f4f4f5;cursor:ew-resize;";
  const wipeValue = document.createElement("div");
  wipeValue.style.cssText = "min-width:34px;text-align:right;font:700 11px Arial,sans-serif;color:#d4d4d8;";
  wipeRow.append(wipeTitle, wipeSlider, wipeValue);
  root.append(stage, controls, wipeRow);

  let dragging = false;
  let animationFrame = 0;
  let commonDuration = 0;

  function showLabels() {
    const visible = !!widgetValue(node, "show_labels", true);
    beforeLabel.textContent = String(widgetValue(node, "before_label", "Before") || "Before");
    afterLabel.textContent = String(widgetValue(node, "after_label", "After") || "After");
    beforeLabel.style.display = visible ? "" : "none";
    afterLabel.style.display = visible ? "" : "none";
  }

  function setWipe(value, updateWidget = false) {
    const position = Math.max(0, Math.min(1, Number(value ?? 0.5)));
    const percent = position * 100;
    afterClip.style.clipPath = `inset(0 0 0 ${percent}%)`;
    divider.style.left = `${percent}%`;
    handle.style.left = `${percent}%`;
    wipeSlider.value = String(position);
    wipeValue.textContent = `${Math.round(percent)}%`;
    if (updateWidget) {
      const rounded = Math.round(position * 100) / 100;
      const widget = getWidget(node, "slider_position");
      if (widget && Number(widget.value) !== rounded) {
        widget.value = rounded;
      }
    }
  }

  function setWipeFromPointer(event) {
    const rect = stage.getBoundingClientRect();
    if (!rect.width) return;
    setWipe((event.clientX - rect.left) / rect.width, true);
  }

  function calculateDuration() {
    const durations = [Number(beforeVideo.duration), Number(afterVideo.duration)]
      .filter((value) => Number.isFinite(value) && value > 0);
    commonDuration = durations.length ? Math.min(...durations) : 0;
    scrubber.max = String(commonDuration);
    updateTime();
  }

  function updateTime() {
    const current = Math.min(commonDuration || Number(beforeVideo.currentTime || 0), Number(beforeVideo.currentTime || 0));
    if (document.activeElement !== scrubber) scrubber.value = String(Math.max(0, current));
    timeLabel.textContent = `${formatTime(current)} / ${formatTime(commonDuration)}`;
  }

  function seekBoth(value) {
    const target = Math.max(0, Math.min(commonDuration || Number(value || 0), Number(value || 0)));
    try {
      beforeVideo.currentTime = target;
      afterVideo.currentTime = target;
    } catch {
      // Metadata may still be loading. The next scrub/play event retries.
    }
    updateTime();
  }

  function pauseBoth() {
    beforeVideo.pause();
    afterVideo.pause();
    playButton.textContent = "▶";
    if (animationFrame) cancelAnimationFrame(animationFrame);
    animationFrame = 0;
  }

  function animationSync() {
    if (beforeVideo.paused) {
      animationFrame = 0;
      return;
    }
    if (commonDuration > 0 && beforeVideo.currentTime >= commonDuration - 0.025) {
      if (widgetValue(node, "loop", true)) {
        seekBoth(0);
      } else {
        seekBoth(commonDuration);
        pauseBoth();
        return;
      }
    }
    if (Math.abs(Number(afterVideo.currentTime || 0) - Number(beforeVideo.currentTime || 0)) > 0.08) {
      try {
        afterVideo.currentTime = beforeVideo.currentTime;
      } catch {
        // Retry on the next animation frame.
      }
    }
    updateTime();
    animationFrame = requestAnimationFrame(animationSync);
  }

  async function playBoth() {
    if (!beforeVideo.src || !afterVideo.src) return;
    if (commonDuration > 0 && beforeVideo.currentTime >= commonDuration - 0.025) seekBoth(0);
    afterVideo.muted = true;
    beforeVideo.muted = !!widgetValue(node, "muted", true);
    try {
      afterVideo.currentTime = beforeVideo.currentTime;
      await Promise.all([beforeVideo.play(), afterVideo.play()]);
      playButton.textContent = "Ⅱ";
      if (!animationFrame) animationFrame = requestAnimationFrame(animationSync);
    } catch (error) {
      pauseBoth();
      console.warn("[VRGDG Video Compare] Playback failed:", error);
    }
  }

  function clearVideo(video) {
    video.pause();
    video.removeAttribute("src");
    video.load();
  }

  function applyWidgets() {
    setWipe(widgetValue(node, "slider_position", 0.5), false);
    showLabels();
    beforeVideo.muted = !!widgetValue(node, "muted", true);
    muteButton.textContent = beforeVideo.muted ? "🔇" : "🔊";
  }

  stage.addEventListener("pointerdown", (event) => {
    dragging = true;
    stage.setPointerCapture?.(event.pointerId);
    setWipeFromPointer(event);
  });
  stage.addEventListener("pointermove", (event) => {
    if (dragging) setWipeFromPointer(event);
  });
  stage.addEventListener("pointerup", (event) => {
    dragging = false;
    stage.releasePointerCapture?.(event.pointerId);
  });
  stage.addEventListener("pointercancel", () => {
    dragging = false;
  });
  wipeSlider.addEventListener("input", () => setWipe(wipeSlider.value, true));
  scrubber.addEventListener("input", () => seekBoth(scrubber.value));
  playButton.addEventListener("click", () => {
    if (beforeVideo.paused) playBoth();
    else pauseBoth();
  });
  restartButton.addEventListener("click", () => {
    const wasPlaying = !beforeVideo.paused;
    seekBoth(0);
    if (wasPlaying) playBoth();
  });
  muteButton.addEventListener("click", () => {
    const muted = !beforeVideo.muted;
    beforeVideo.muted = muted;
    setWidgetValue(node, "muted", muted);
    muteButton.textContent = muted ? "🔇" : "🔊";
  });
  beforeVideo.addEventListener("loadedmetadata", calculateDuration);
  afterVideo.addEventListener("loadedmetadata", calculateDuration);
  beforeVideo.addEventListener("pause", () => {
    if (!beforeVideo.ended) pauseBoth();
  });
  beforeVideo.addEventListener("ratechange", () => {
    afterVideo.playbackRate = beforeVideo.playbackRate;
  });
  root.addEventListener("pointerdown", (event) => event.stopPropagation());
  root.addEventListener("pointermove", (event) => event.stopPropagation());
  root.addEventListener("pointerup", (event) => event.stopPropagation());
  root.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
  root.addEventListener("contextmenu", (event) => event.stopPropagation());

  applyWidgets();

  return {
    root,
    applyWidgets,
    load(items = [], settings = {}) {
      pauseBoth();
      const beforeInfo = items.find((item) => item?.compare_role === "before") || items[0];
      const afterInfo = items.find((item) => item?.compare_role === "after") || items[1];
      if (!beforeInfo?.path || !afterInfo?.path) return;
      if (settings.slider_position !== undefined) setWidgetValue(node, "slider_position", settings.slider_position);
      for (const [name, value] of [
        ["before_label", settings.before_label],
        ["after_label", settings.after_label],
        ["show_labels", settings.show_labels],
        ["loop", settings.loop],
        ["muted", settings.muted],
      ]) {
        if (value !== undefined) {
          const widget = getWidget(node, name);
          if (widget) widget.value = value;
        }
      }
      commonDuration = 0;
      scrubber.max = "0";
      scrubber.value = "0";
      beforeVideo.src = videoUrl(beforeInfo.path);
      afterVideo.src = videoUrl(afterInfo.path);
      beforeVideo.load();
      afterVideo.load();
      empty.style.display = "none";
      applyWidgets();
      node.setDirtyCanvas?.(true, true);
    },
    destroy() {
      pauseBoth();
      clearVideo(beforeVideo);
      clearVideo(afterVideo);
    },
  };
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onConfigure = nodeType.prototype.onConfigure;
    const onExecuted = nodeType.prototype.onExecuted;
    const onRemoved = nodeType.prototype.onRemoved;
    const onResize = nodeType.prototype.onResize;

    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.resizable = true;
      this.serialize_widgets = true;
      this.size = [
        Math.max(MIN_WIDTH, this.size?.[0] || MIN_WIDTH),
        Math.max(MIN_HEIGHT, this.size?.[1] || MIN_HEIGHT),
      ];
      const ui = createCompareUI(this);
      this._vrgdgVideoCompareUI = ui;
      const domWidget = this.addDOMWidget("video_compare", "vrgdg-video-compare", ui.root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 430,
        getMaxHeight: () => 430,
      });
      if (domWidget) {
        domWidget.serialize = false;
        domWidget.computeSize = (width) => [width, 430];
      }
      for (const widget of this.widgets || []) {
        if (widget.name === "video_compare" || widget._vrgdgVideoCompareBound) continue;
        const callback = widget.callback;
        widget.callback = function () {
          const callbackResult = callback?.apply(this, arguments);
          ui.applyWidgets();
          return callbackResult;
        };
        widget._vrgdgVideoCompareBound = true;
      }
      return result;
    };

    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.size = [
        Math.max(MIN_WIDTH, this.size?.[0] || MIN_WIDTH),
        Math.max(MIN_HEIGHT, this.size?.[1] || MIN_HEIGHT),
      ];
      this._vrgdgVideoCompareUI?.applyWidgets();
      return result;
    };

    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      this._vrgdgVideoCompareUI?.load(
        message?.compare_videos || [],
        message?.compare_video_settings || {},
      );
    };

    nodeType.prototype.onResize = function (size) {
      const result = onResize?.apply(this, arguments);
      this.size = [
        Math.max(MIN_WIDTH, size?.[0] || MIN_WIDTH),
        Math.max(MIN_HEIGHT, size?.[1] || MIN_HEIGHT),
      ];
      return result;
    };

    nodeType.prototype.onRemoved = function () {
      this._vrgdgVideoCompareUI?.destroy();
      this._vrgdgVideoCompareUI = null;
      return onRemoved?.apply(this, arguments);
    };
  },
});
