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
    `min-width:${MIN_WIDTH}px`,
    "max-width:none",
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
  controls.style.cssText = "display:grid;grid-template-columns:auto auto auto minmax(80px,1fr) auto auto auto;gap:7px;align-items:center;";
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
  const recordButton = createButton("● Record", "Record the labeled slider preview for the configured duration");
  const fullscreenButton = createButton("⛶ Fullscreen", "Open the compare slider in a fullscreen viewer without recording");
  recordButton.style.background = "#7f1d1d";
  const recordStatus = document.createElement("div");
  recordStatus.textContent = "";
  recordStatus.style.cssText = "font:700 10px Arial,sans-serif;color:#fca5a5;white-space:nowrap;";
  controls.append(playButton, restartButton, recordButton, scrubber, timeLabel, muteButton, fullscreenButton);

  const wipeRow = document.createElement("div");
  wipeRow.style.cssText = "display:grid;grid-template-columns:auto minmax(80px,1fr) auto;gap:8px;align-items:center;";
  const wipeTitle = document.createElement("div");
  wipeTitle.textContent = "Slider";
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

  const fileRow = document.createElement("div");
  fileRow.style.cssText = "display:flex;flex-wrap:wrap;gap:7px;align-items:center;";
  const chooseBeforeButton = createButton("Choose Before Video", "Upload a video directly from your computer as the original video");
  const chooseAfterButton = createButton("Choose After Video", "Upload a video directly from your computer as the processed video");
  const beforeFileInput = document.createElement("input");
  const afterFileInput = document.createElement("input");
  for (const input of [beforeFileInput, afterFileInput]) { input.type = "file"; input.accept = ".mp4,.mov,.webm,.mkv,.avi,.m4v,video/*"; input.style.display = "none"; }
  const fileStatus = document.createElement("div");
  fileStatus.style.cssText = "font:700 10px Arial,sans-serif;color:#a1a1aa;white-space:nowrap;";
  fileRow.append(chooseBeforeButton, chooseAfterButton, beforeFileInput, afterFileInput, fileStatus);
  root.append(stage, controls, wipeRow, fileRow, recordStatus);

  let dragging = false;
  let animationFrame = 0;
  let commonDuration = 0;
  let recorder = null;
  let recordingTimer = 0;
  let recordingFrame = 0;
  let recordingChunks = [];
  const selectedPaths = { before: "", after: "" };
  let fullscreenOverlay = null;
  let fullscreenStageStyle = "";
  let fullscreenZoom = 1;
  let fullscreenZoomOrigin = { x: 50, y: 50 };
  let fullscreenZoomLabel = null;
  let fullscreenResetButton = null;

  function showLabels() {
    const visible = !!widgetValue(node, "show_labels", true);
    beforeLabel.textContent = String(widgetValue(node, "before_label", "Before") || "Before");
    afterLabel.textContent = String(widgetValue(node, "after_label", "After") || "After");
    beforeLabel.style.display = visible ? "" : "none";
    afterLabel.style.display = visible ? "" : "none";
  }

  function drawContained(ctx, video, width, height) {
    if (!video.videoWidth || !video.videoHeight) return;
    const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
    const drawWidth = video.videoWidth * scale;
    const drawHeight = video.videoHeight * scale;
    ctx.drawImage(video, (width - drawWidth) / 2, (height - drawHeight) / 2, drawWidth, drawHeight);
  }

  function drawRecordingFrame(ctx, width, height) {
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, width, height);
    drawContained(ctx, beforeVideo, width, height);
    const position = Math.max(0, Math.min(1, Number(wipeSlider.value || 0.5)));
    ctx.save();
    ctx.beginPath();
    ctx.rect(width * position, 0, width * (1 - position), height);
    ctx.clip();
    drawContained(ctx, afterVideo, width, height);
    ctx.restore();
    if (widgetValue(node, "show_labels", true)) {
      ctx.font = "900 22px Arial,sans-serif";
      const beforeText = beforeLabel.textContent || "Before";
      const afterText = afterLabel.textContent || "After";
      const beforeWidth = ctx.measureText(beforeText).width;
      const afterWidth = ctx.measureText(afterText).width;
      ctx.fillStyle = "rgba(0,0,0,.68)";
      ctx.fillRect(10, 10, beforeWidth + 16, 32);
      ctx.fillRect(width - afterWidth - 26, 10, afterWidth + 16, 32);
      ctx.fillStyle = "#fff";
      ctx.fillText(beforeText, 18, 34);
      ctx.fillText(afterText, width - afterWidth - 18, 34);
    }
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = Math.max(2, width / 640);
    ctx.beginPath();
    ctx.moveTo(width * position, 0);
    ctx.lineTo(width * position, height);
    ctx.stroke();
  }

  function stopRecording() {
    if (!recorder) return;
    clearTimeout(recordingTimer);
    cancelAnimationFrame(recordingFrame);
    recorder.stop();
    pauseBoth();
    recordButton.disabled = false;
    recordButton.textContent = "● Record";
    recordStatus.textContent = "Finishing…";
  }

  function startRecording() {
    if (recorder || !beforeVideo.src || !afterVideo.src) return;
    const canvas = document.createElement("canvas");
    const width = Math.max(640, Math.round((beforeVideo.videoWidth || stage.clientWidth || 640)));
    const height = Math.max(360, Math.round((beforeVideo.videoHeight || stage.clientHeight || 360)));
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const stream = canvas.captureStream(30);
    const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]
      .find((type) => MediaRecorder.isTypeSupported(type));
    if (!mimeType) {
      recordStatus.textContent = "Recording is not supported by this browser.";
      return;
    }
    recordingChunks = [];
    recorder = new MediaRecorder(stream, { mimeType });
    recorder.ondataavailable = (event) => { if (event.data.size) recordingChunks.push(event.data); };
    recorder.onstop = async () => {
      const blob = new Blob(recordingChunks, { type: mimeType });
      recordStatus.textContent = "Saving MP4 to ComfyUI output…";
      try {
        const form = new FormData();
        form.append("file", blob, "vrgdg-video-compare.webm");
        const response = await api.fetchApi("/vrgdg/video_compare/save_recording", {
          method: "POST",
          body: form,
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "The recording could not be saved.");
        const link = document.createElement("a");
        link.href = videoUrl(data.path);
        link.download = data.name || "vrgdg-video-compare.mp4";
        link.textContent = `Download MP4 (${data.name || "saved"})`;
        link.title = data.path;
        link.style.cssText = "color:#67e8f9;font:700 11px Arial,sans-serif;white-space:nowrap;";
        recordStatus.replaceChildren(link);
      } catch (error) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "vrgdg-video-compare.webm";
        link.textContent = "Download temporary WebM (MP4 save failed)";
        link.style.cssText = "color:#fca5a5;font:700 11px Arial,sans-serif;white-space:nowrap;";
        recordStatus.replaceChildren(link);
        setTimeout(() => URL.revokeObjectURL(url), 60000);
        console.error("[VRGDG Video Compare] Could not save MP4:", error);
      }
      recorder = null;
    };
    recorder.start(200);
    recordButton.disabled = true;
    recordButton.textContent = "■ Recording";
    recordStatus.textContent = "Move the slider…";
    playBoth();
    const startedAt = performance.now();
    const durationMs = Math.max(500, Number(widgetValue(node, "record_duration", 5)) * 1000);
    const render = () => {
      drawRecordingFrame(ctx, width, height);
      if (recorder && performance.now() - startedAt < durationMs) recordingFrame = requestAnimationFrame(render);
    };
    render();
    recordingTimer = setTimeout(stopRecording, durationMs);
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

  function closeFullscreen(exitBrowserFullscreen = true) {
    if (!fullscreenOverlay) return;
    if (exitBrowserFullscreen && document.fullscreenElement === fullscreenOverlay) {
      document.exitFullscreen?.().catch?.(() => {});
    }
    if (fullscreenOverlay.parentNode) fullscreenOverlay.parentNode.removeChild(fullscreenOverlay);
    root.insertBefore(stage, root.firstChild);
    stage.style.cssText = fullscreenStageStyle;
    setFullscreenZoom(1, 50, 50);
    fullscreenZoomLabel = null;
    fullscreenResetButton = null;
    fullscreenOverlay = null;
    fullscreenButton.textContent = "⛶ Fullscreen";
    app.graph?.setDirtyCanvas?.(true, true);
  }

  function setFullscreenZoom(value, originX = fullscreenZoomOrigin.x, originY = fullscreenZoomOrigin.y) {
    fullscreenZoom = Math.max(1, Math.min(6, Number(value) || 1));
    fullscreenZoomOrigin = {
      x: Math.max(0, Math.min(100, Number(originX) || 50)),
      y: Math.max(0, Math.min(100, Number(originY) || 50)),
    };
    const transform = `scale(${fullscreenZoom})`;
    beforeVideo.style.transform = transform;
    afterVideo.style.transform = transform;
    beforeVideo.style.transformOrigin = `${fullscreenZoomOrigin.x}% ${fullscreenZoomOrigin.y}%`;
    afterVideo.style.transformOrigin = `${fullscreenZoomOrigin.x}% ${fullscreenZoomOrigin.y}%`;
    if (fullscreenZoomLabel) fullscreenZoomLabel.textContent = `${Math.round(fullscreenZoom * 100)}%`;
    if (fullscreenResetButton) {
      fullscreenResetButton.disabled = fullscreenZoom === 1;
      fullscreenResetButton.style.opacity = fullscreenZoom === 1 ? "0.55" : "1";
    }
  }

  function openFullscreen() {
    if (fullscreenOverlay) return;
    fullscreenStageStyle = stage.style.cssText;
    fullscreenOverlay = document.createElement("div");
    fullscreenOverlay.style.cssText = "position:fixed;inset:0;z-index:100000;display:flex;flex-direction:column;gap:10px;padding:14px;background:#09090b;color:#f4f4f5;font-family:Arial,sans-serif;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:34px;";
    const title = document.createElement("div");
    title.textContent = "VRGDG Video Compare Slider";
    title.style.cssText = "font:900 15px Arial,sans-serif;color:#cffafe;";
    fullscreenZoomLabel = document.createElement("div");
    fullscreenZoomLabel.textContent = "100%";
    fullscreenZoomLabel.title = "Fullscreen zoom level. Scroll over the video to zoom.";
    fullscreenZoomLabel.style.cssText = "font:800 12px Arial,sans-serif;color:#a5f3fc;min-width:42px;text-align:center;";
    fullscreenResetButton = createButton("Reset Zoom", "Reset fullscreen zoom to 100%");
    fullscreenResetButton.disabled = true;
    fullscreenResetButton.style.opacity = "0.55";
    fullscreenResetButton.addEventListener("click", () => setFullscreenZoom(1));
    const closeButton = createButton("✕ Exit Fullscreen", "Return to the ComfyUI node");
    closeButton.addEventListener("click", () => closeFullscreen(true));
    const headerActions = document.createElement("div");
    headerActions.style.cssText = "display:flex;align-items:center;gap:8px;";
    headerActions.append(fullscreenZoomLabel, fullscreenResetButton, closeButton);
    header.append(title, headerActions);
    fullscreenOverlay.append(header, stage);
    stage.style.cssText = "position:relative;flex:1 1 auto;min-height:0;width:100%;overflow:hidden;border:1px solid #3f3f46;border-radius:8px;background:#050505;touch-action:none;cursor:ew-resize;";
    setFullscreenZoom(1, 50, 50);
    document.body.append(fullscreenOverlay);
    fullscreenButton.textContent = "✕ Exit Fullscreen";
    fullscreenOverlay.addEventListener("fullscreenchange", () => {
      if (fullscreenOverlay && document.fullscreenElement !== fullscreenOverlay) closeFullscreen(false);
    });
    fullscreenOverlay.requestFullscreen?.().catch?.(() => {});
  }

  async function uploadSelectedVideo(file, slot) {
    if (!file) return;
    fileStatus.textContent = `Uploading ${file.name}…`;
    const form = new FormData();
    form.append("video", file, file.name);
    try {
      const response = await api.fetchApi("/vrgdg/video_compare/upload", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok || !data.ok || !data.path) throw new Error(data.error || `Upload failed (${response.status})`);
      selectedPaths[slot] = data.path;
      setWidgetValue(node, slot === "before" ? "before_selected" : "after_selected", data.path);
      fileStatus.textContent = `${slot === "before" ? "Before" : "After"}: ${data.name || file.name}`;
    } catch (error) {
      fileStatus.textContent = String(error?.message || error);
      fileStatus.style.color = "#fca5a5";
    }
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
  stage.addEventListener("wheel", (event) => {
    if (!fullscreenOverlay) return;
    event.preventDefault();
    event.stopPropagation();
    const direction = event.deltaY < 0 ? 1 : -1;
    const factor = direction > 0 ? 1.12 : 1 / 1.12;
    const rect = stage.getBoundingClientRect();
    const originX = rect.width ? ((event.clientX - rect.left) / rect.width) * 100 : 50;
    const originY = rect.height ? ((event.clientY - rect.top) / rect.height) * 100 : 50;
    setFullscreenZoom(fullscreenZoom * factor, originX, originY);
  }, { passive: false });
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
  recordButton.addEventListener("click", startRecording);
  chooseBeforeButton.addEventListener("click", () => beforeFileInput.click());
  chooseAfterButton.addEventListener("click", () => afterFileInput.click());
  beforeFileInput.addEventListener("change", () => uploadSelectedVideo(beforeFileInput.files?.[0], "before"));
  afterFileInput.addEventListener("change", () => uploadSelectedVideo(afterFileInput.files?.[0], "after"));
  fullscreenButton.addEventListener("click", () => { if (fullscreenOverlay) closeFullscreen(true); else openFullscreen(); });
  beforeVideo.addEventListener("loadedmetadata", calculateDuration);
  afterVideo.addEventListener("loadedmetadata", calculateDuration);
  for (const video of [beforeVideo, afterVideo]) {
    video.addEventListener("error", () => {
      empty.textContent = "Could not load one of the videos. Check the ComfyUI console for the media URL/error.";
      empty.style.display = "flex";
      console.error("[VRGDG Video Compare] Video load failed:", video.currentSrc || video.src, video.error);
    });
  }
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
        ["record_duration", settings.record_duration],
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
      closeFullscreen(true);
      if (recorder) stopRecording();
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
        domWidget.element.style.width = "100%";
        domWidget.element.style.minWidth = `${MIN_WIDTH}px`;
        domWidget.element.style.maxWidth = "none";
        ui.root.style.width = `${Math.max(MIN_WIDTH, this.size[0])}px`;
        // LiteGraph may pass the DOM widget a default width that is narrower
        // than the node itself.  Returning that width makes the embedded UI
        // render in a small column and leaves the rest of the node blank.
        // Keep the widget's measured width aligned with the node minimum.
        domWidget.computeSize = () => [MIN_WIDTH, 430];
      }
      for (const widget of this.widgets || []) {
        if (widget.name === "before_selected" || widget.name === "after_selected") {
          widget.hidden = true;
          widget.serialize = true;
          widget.computeSize = () => [0, -4];
        }
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
      if (this._vrgdgVideoCompareUI?.root) {
        this._vrgdgVideoCompareUI.root.style.width = `${Math.max(MIN_WIDTH, this.size[0])}px`;
      }
      this._vrgdgVideoCompareUI?.applyWidgets();
      return result;
    };

    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const payload = message?.output || message?.ui || message || {};
      this._vrgdgVideoCompareUI?.load(
        payload?.compare_videos || [],
        payload?.compare_video_settings || {},
      );
    };

    nodeType.prototype.onResize = function (size) {
      const result = onResize?.apply(this, arguments);
      this.size = [
        Math.max(MIN_WIDTH, size?.[0] || MIN_WIDTH),
        Math.max(MIN_HEIGHT, size?.[1] || MIN_HEIGHT),
      ];
      const root = this._vrgdgVideoCompareUI?.root;
      if (root) {
        root.style.width = `${Math.max(MIN_WIDTH, this.size[0])}px`;
        root.style.minWidth = `${MIN_WIDTH}px`;
      }
      return result;
    };

    nodeType.prototype.onRemoved = function () {
      this._vrgdgVideoCompareUI?.destroy();
      this._vrgdgVideoCompareUI = null;
      return onRemoved?.apply(this, arguments);
    };
  },
});
