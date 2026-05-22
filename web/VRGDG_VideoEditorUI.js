import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "VRGDG_VideoEditorUI";
const HIDDEN_WIDGETS = new Set([
  "selected_clip_path",
  "session_path",
  "captured_frame_path",
  "generated_t2i_prompt",
  "generated_i2v_prompt",
]);

function getWidget(node, name) {
  return (node?.widgets || []).find((widget) => widget?.name === name);
}

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

function setWidgetValue(node, name, value) {
  const widget = getWidget(node, name);
  if (!widget) return false;
  widget.value = value;
  widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse);
  const index = (node.widgets || []).indexOf(widget);
  if (Array.isArray(node.widgets_values) && index >= 0) node.widgets_values[index] = value;
  app.graph?.setDirtyCanvas?.(true, true);
  return true;
}

function makeButton(label, kind = "neutral") {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.style.cssText = `
    border: 1px solid ${kind === "primary" ? "#0891b2" : "#3f3f46"};
    border-radius: 6px;
    background: ${kind === "primary" ? "#06b6d4" : "#27272a"};
    color: ${kind === "primary" ? "#082f49" : "#f4f4f5"};
    font-size: 12px;
    font-weight: 800;
    padding: 8px 11px;
    cursor: pointer;
  `;
  return button;
}

function makeInput(value = "", type = "text") {
  const input = document.createElement("input");
  input.type = type;
  input.value = value;
  input.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-size:12px;";
  return input;
}

function makeField(label, control) {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;color:#d4d4d8;font-weight:700;";
  const text = document.createElement("span");
  text.textContent = label;
  wrapper.append(text, control);
  return wrapper;
}

function toast(message, isError = false) {
  const element = document.createElement("div");
  element.textContent = message;
  element.style.cssText = `
    position: fixed;
    right: 18px;
    bottom: 18px;
    z-index: 100003;
    max-width: min(520px, calc(100vw - 36px));
    border: 1px solid ${isError ? "#991b1b" : "#155e75"};
    border-radius: 8px;
    background: ${isError ? "#450a0a" : "#083344"};
    color: ${isError ? "#fecaca" : "#cffafe"};
    padding: 12px 14px;
    white-space: pre-wrap;
    font-size: 12px;
    line-height: 1.4;
    box-shadow: 0 18px 60px rgba(0,0,0,.45);
  `;
  document.body.appendChild(element);
  setTimeout(() => element.remove(), 5200);
}

async function postJson(url, payload) {
  const response = await api.fetchApi(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data?.ok) throw new Error(String(data?.error || `Request failed (${response.status})`));
  return data;
}

function formatBytes(value) {
  const size = Number(value || 0);
  if (size > 1024 * 1024 * 1024) return `${(size / 1024 / 1024 / 1024).toFixed(2)} GB`;
  if (size > 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`;
  if (size > 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${size} B`;
}

function formatTime(value) {
  const total = Math.max(0, Number(value || 0));
  const minutes = Math.floor(total / 60);
  const seconds = Math.floor(total % 60);
  const frames = Math.floor((total - Math.floor(total)) * 100);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(frames).padStart(2, "0")}`;
}

function loadVideoDuration(url) {
  return new Promise((resolve) => {
    const probe = document.createElement("video");
    let finished = false;
    const done = (duration) => {
      if (finished) return;
      finished = true;
      probe.removeAttribute("src");
      probe.load();
      resolve(Number.isFinite(duration) && duration > 0 ? duration : 8);
    };
    probe.preload = "metadata";
    probe.muted = true;
    probe.addEventListener("loadedmetadata", () => done(probe.duration), { once: true });
    probe.addEventListener("error", () => done(8), { once: true });
    setTimeout(() => done(8), 3500);
    probe.src = url;
  });
}

function loadVideoThumbnail(url, seekTime = 0.25) {
  return new Promise((resolve) => {
    const probe = document.createElement("video");
    let finished = false;
    const done = (thumbnail) => {
      if (finished) return;
      finished = true;
      probe.removeAttribute("src");
      probe.load();
      resolve(thumbnail || "");
    };
    probe.preload = "metadata";
    probe.muted = true;
    probe.playsInline = true;
    probe.crossOrigin = "anonymous";
    probe.addEventListener("loadedmetadata", () => {
      const duration = Number.isFinite(probe.duration) && probe.duration > 0 ? probe.duration : 0;
      const target = duration > 0 ? Math.min(Math.max(0.05, seekTime), Math.max(0.05, duration - 0.05)) : 0;
      try {
        probe.currentTime = target;
      } catch (_) {
        done("");
      }
    }, { once: true });
    probe.addEventListener("seeked", () => {
      try {
        const canvas = document.createElement("canvas");
        const width = Math.max(1, probe.videoWidth || 320);
        const height = Math.max(1, probe.videoHeight || 180);
        canvas.width = 240;
        canvas.height = Math.max(80, Math.round((canvas.width / width) * height));
        canvas.getContext("2d")?.drawImage(probe, 0, 0, canvas.width, canvas.height);
        done(canvas.toDataURL("image/jpeg", 0.72));
      } catch (_) {
        done("");
      }
    }, { once: true });
    probe.addEventListener("error", () => done(""), { once: true });
    setTimeout(() => done(""), 5000);
    probe.src = url;
  });
}

function clipId(clip) {
  return String(clip?.path || clip?.name || "");
}

function clipVersion(clip) {
  return `${clip?.mtime || 0}:${clip?.size || 0}`;
}

function defaultClipState(clip) {
  return {
    path: clip.path,
    name: clip.name,
    clip_number: clip.clip_number,
    selected_for_remake: false,
    user_notes: "",
    i2v_user_notes: "",
    t2i_prompt: "",
    i2v_prompt: "",
    captured_frame_path: "",
  };
}

function createEditablePromptBox(placeholder = "") {
  const box = document.createElement("textarea");
  box.placeholder = placeholder;
  box.style.cssText = "width:100%;box-sizing:border-box;min-height:118px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#111113;color:#fafafa;padding:9px;font-size:12px;line-height:1.45;";
  return box;
}

function getSelectedGemmaModel(node, widgetName) {
  const value = String(getWidget(node, widgetName)?.value || "").trim();
  return value.toLowerCase().endsWith(".gguf") ? value : "";
}

function openEditor(node) {
  const overlay = document.createElement("div");
  overlay.style.cssText = `
    position: fixed;
    inset: 0;
    z-index: 100000;
    background: rgba(0,0,0,.72);
    display: flex;
    align-items: center;
    justify-content: center;
  `;

  const shell = document.createElement("div");
  shell.style.cssText = `
    width: min(1920px, calc(100vw - 24px));
    height: min(860px, calc(100vh - 24px));
    display: grid;
    grid-template-rows: auto 1fr 170px;
    background: #18181b;
    color: #fafafa;
    border: 1px solid #3f3f46;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 24px 90px rgba(0,0,0,.55);
  `;

  const topbar = document.createElement("div");
  topbar.style.cssText = "display:grid;grid-template-columns:minmax(260px,1fr) 140px auto auto;gap:10px;align-items:end;padding:12px;border-bottom:1px solid #27272a;background:#202024;";
  const folderInput = makeInput(String(getWidget(node, "output_folder")?.value || ""));
  const extensionsInput = makeInput(String(getWidget(node, "video_extensions")?.value || ".mp4,.mov,.webm,.mkv"));
  const refreshButton = makeButton("Load Clips", "primary");
  const closeButton = makeButton("Close");
  closeButton.addEventListener("click", () => overlay.remove());
  topbar.append(makeField("Output folder", folderInput), makeField("Extensions", extensionsInput), refreshButton, closeButton);

  const main = document.createElement("div");
  main.style.cssText = "display:grid;grid-template-columns:360px 1fr 300px;min-height:0;";

  const clipList = document.createElement("div");
  clipList.style.cssText = "overflow:auto;padding:10px;border-right:1px solid #27272a;background:#202024;";

  const previewArea = document.createElement("div");
  previewArea.style.cssText = "display:grid;grid-template-rows:1fr auto auto;min-width:0;background:#111113;";
  const videoWrap = document.createElement("div");
  videoWrap.style.cssText = "display:flex;align-items:center;justify-content:center;min-height:0;background:#09090b;";
  const video = document.createElement("video");
  video.controls = true;
  video.style.cssText = "max-width:100%;max-height:100%;background:#000;";
  videoWrap.appendChild(video);
  const globalScrubWrap = document.createElement("div");
  globalScrubWrap.style.cssText = "display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:9px;padding:8px 12px;border-top:1px solid #27272a;background:#111113;";
  const globalScrubLabel = document.createElement("div");
  globalScrubLabel.textContent = "Global scrub";
  globalScrubLabel.style.cssText = "font-size:11px;font-weight:800;color:#d4d4d8;white-space:nowrap;";
  const globalScrub = document.createElement("input");
  globalScrub.type = "range";
  globalScrub.min = "0";
  globalScrub.max = "0";
  globalScrub.step = "0.01";
  globalScrub.value = "0";
  globalScrub.disabled = true;
  globalScrub.style.cssText = "width:100%;accent-color:#22d3ee;cursor:pointer;";
  const globalScrubTime = document.createElement("div");
  globalScrubTime.textContent = "00:00.00 / 00:00.00";
  globalScrubTime.style.cssText = "font-size:11px;color:#67e8f9;font-variant-numeric:tabular-nums;white-space:nowrap;";
  globalScrubWrap.append(globalScrubLabel, globalScrub, globalScrubTime);
  const activeInfo = document.createElement("div");
  activeInfo.textContent = "No clip selected";
  activeInfo.style.cssText = "padding:10px 12px;border-top:1px solid #27272a;font-size:12px;color:#d4d4d8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
  previewArea.append(videoWrap, globalScrubWrap, activeInfo);

  const inspector = document.createElement("div");
  inspector.style.cssText = "display:flex;flex-direction:column;gap:10px;padding:10px;border-left:1px solid #27272a;background:#202024;min-height:0;overflow:auto;";
  const selectedToggle = document.createElement("input");
  selectedToggle.type = "checkbox";
  const selectedWrap = document.createElement("label");
  selectedWrap.style.cssText = "display:flex;align-items:center;gap:8px;font-size:13px;font-weight:800;color:#fafafa;";
  selectedWrap.append(selectedToggle, document.createTextNode("Select for remake"));
  const tabBar = document.createElement("div");
  tabBar.style.cssText = "display:grid;grid-template-columns:repeat(3,1fr);gap:5px;";
  const reviewTab = makeButton("Review");
  const t2iTab = makeButton("T2I");
  const i2vTab = makeButton("I2V");
  tabBar.append(reviewTab, t2iTab, i2vTab);
  const reviewPanel = document.createElement("div");
  reviewPanel.style.cssText = "display:flex;flex-direction:column;gap:10px;";
  const t2iPanel = document.createElement("div");
  t2iPanel.style.cssText = "display:none;flex-direction:column;gap:10px;";
  const i2vPanel = document.createElement("div");
  i2vPanel.style.cssText = "display:none;flex-direction:column;gap:10px;";
  const notes = document.createElement("textarea");
  notes.placeholder = "Extra generation details for this clip...";
  notes.style.cssText = "width:100%;box-sizing:border-box;min-height:120px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:9px;font-size:12px;line-height:1.45;";
  const t2iNote = document.createElement("div");
  t2iNote.textContent = "T2I can use a captured frame if you grab one. Without a frame, it creates the prompt from your generation notes.";
  t2iNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;";
  const frameButton = makeButton("Grab Current Frame", "primary");
  const promptButton = makeButton("Create T2I Prompt", "primary");
  const framePath = document.createElement("div");
  framePath.textContent = "No frame captured";
  framePath.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;word-break:break-word;";
  const promptBox = createEditablePromptBox("Gemma prompt will appear here, or type your own...");
  const promptActions = document.createElement("div");
  promptActions.style.cssText = "display:grid;grid-template-columns:1fr;gap:8px;";
  promptActions.append(frameButton, promptButton);
  const i2vNote = document.createElement("div");
  i2vNote.textContent = "I2V uses the T2I prompt as visual reference. Pick a text Gemma model on the node; no mmproj is needed.";
  i2vNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;";
  const i2vNotes = document.createElement("textarea");
  i2vNotes.placeholder = "Optional: camera motion, character movement, performance energy, or anything else Gemma must follow...";
  i2vNotes.style.cssText = "width:100%;box-sizing:border-box;min-height:112px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:9px;font-size:12px;line-height:1.45;";
  const i2vButton = makeButton("Create I2V Prompt", "primary");
  const i2vPromptBox = createEditablePromptBox("Image-to-video prompt will appear here, or type your own...");
  const selectedCount = document.createElement("div");
  selectedCount.style.cssText = "font-size:12px;color:#a1a1aa;";
  const saveButton = makeButton("Save Editor Session", "primary");
  const clearButton = makeButton("Clear Session");
  const sessionActions = document.createElement("div");
  sessionActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  sessionActions.append(saveButton, clearButton);
  reviewPanel.append(selectedWrap, selectedCount);
  t2iPanel.append(makeField("Generation notes", notes), t2iNote, promptActions, makeField("Captured frame", framePath), makeField("T2I prompt", promptBox));
  i2vPanel.append(i2vNote, makeField("I2V motion notes", i2vNotes), i2vButton, makeField("I2V prompt", i2vPromptBox));
  inspector.append(
    tabBar,
    reviewPanel,
    t2iPanel,
    i2vPanel,
    sessionActions
  );

  main.append(clipList, previewArea, inspector);

  const timeline = document.createElement("div");
  timeline.style.cssText = "display:grid;grid-template-rows:auto 1fr;border-top:1px solid #27272a;background:#111113;min-width:0;min-height:0;";
  const timelineHeader = document.createElement("div");
  timelineHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:8px 12px;border-bottom:1px solid #27272a;font-size:12px;color:#d4d4d8;";
  const timelineTitle = document.createElement("div");
  timelineTitle.textContent = "Timeline";
  timelineTitle.style.cssText = "font-weight:800;color:#f4f4f5;";
  const timelineTime = document.createElement("div");
  timelineTime.textContent = "00:00.00 / 00:00.00";
  timelineTime.style.cssText = "font-variant-numeric:tabular-nums;color:#67e8f9;";
  timelineHeader.append(timelineTitle, timelineTime);
  const timelineViewport = document.createElement("div");
  timelineViewport.style.cssText = "position:relative;overflow:auto;padding:12px 12px 10px;min-height:0;cursor:pointer;";
  const timelineTrack = document.createElement("div");
  timelineTrack.style.cssText = "position:relative;display:flex;align-items:stretch;gap:0;height:104px;min-width:100%;";
  const playhead = document.createElement("div");
  playhead.style.cssText = "position:absolute;top:0;bottom:0;width:2px;background:#f4f4f5;box-shadow:0 0 0 1px rgba(0,0,0,.65),0 0 12px rgba(103,232,249,.75);z-index:4;pointer-events:none;transform:translateX(-1px);";
  const playheadHandle = document.createElement("div");
  playheadHandle.style.cssText = "position:absolute;top:-5px;left:50%;width:12px;height:12px;border-radius:50%;background:#67e8f9;transform:translateX(-50%);box-shadow:0 0 0 2px rgba(0,0,0,.7);";
  playhead.appendChild(playheadHandle);
  timelineViewport.appendChild(timelineTrack);
  timeline.append(timelineHeader, timelineViewport);

  shell.append(topbar, main, timeline);
  overlay.appendChild(shell);
  document.body.appendChild(overlay);

  const state = {
    clips: [],
    session: { clips: {} },
    activeId: "",
    sessionPath: "",
    totalDuration: 0,
    pxPerSecond: 42,
    isScrubbing: false,
    isGlobalScrubbing: false,
    activeTab: "review",
  };

  function setTab(name) {
    state.activeTab = name;
    const activeStyle = "border-color:#06b6d4;background:#164e63;color:#f4f4f5;";
    const inactiveStyle = "border-color:#3f3f46;background:#27272a;color:#f4f4f5;";
    reviewPanel.style.display = name === "review" ? "flex" : "none";
    t2iPanel.style.display = name === "t2i" ? "flex" : "none";
    i2vPanel.style.display = name === "i2v" ? "flex" : "none";
    reviewTab.style.cssText += name === "review" ? activeStyle : inactiveStyle;
    t2iTab.style.cssText += name === "t2i" ? activeStyle : inactiveStyle;
    i2vTab.style.cssText += name === "i2v" ? activeStyle : inactiveStyle;
  }

  function activeClip() {
    return state.clips.find((clip) => clipId(clip) === state.activeId) || null;
  }

  function clipState(clip) {
    const id = clipId(clip);
    if (!state.session.clips[id]) state.session.clips[id] = defaultClipState(clip);
    return state.session.clips[id];
  }

  function updateSelectedCount() {
    const count = Object.values(state.session.clips || {}).filter((item) => item?.selected_for_remake).length;
    selectedCount.textContent = `${count} clip${count === 1 ? "" : "s"} selected for remake`;
  }

  function syncInspector() {
    const clip = activeClip();
    if (!clip) {
      selectedToggle.checked = false;
      notes.value = "";
      i2vNotes.value = "";
      framePath.textContent = "No frame captured";
      promptBox.value = "";
      i2vPromptBox.value = "";
      activeInfo.textContent = "No clip selected";
      return;
    }
    const item = clipState(clip);
    selectedToggle.checked = Boolean(item.selected_for_remake);
    notes.value = item.user_notes || "";
    i2vNotes.value = item.i2v_user_notes || "";
    framePath.textContent = item.captured_frame_path || "No frame captured";
    promptBox.value = item.t2i_prompt || "";
    i2vPromptBox.value = item.i2v_prompt || "";
    activeInfo.textContent = `${clip.name} | clip ${clip.clip_number} | ${formatBytes(clip.size)}`;
    setWidgetValue(node, "selected_clip_path", clip.path || "");
    setWidgetValue(node, "captured_frame_path", item.captured_frame_path || "");
    setWidgetValue(node, "generated_t2i_prompt", item.t2i_prompt || "");
    setWidgetValue(node, "generated_i2v_prompt", item.i2v_prompt || "");
  }

  function buildTimelineModel() {
    let cursor = 0;
    for (const clip of state.clips) {
      clip.timeline_start = cursor;
      clip.duration = Number.isFinite(clip.duration) && clip.duration > 0 ? clip.duration : 8;
      cursor += clip.duration;
    }
    state.totalDuration = cursor;
  }

  async function ensureDurations() {
    await Promise.all(state.clips.map(async (clip) => {
      clip.duration = await loadVideoDuration(clip.url);
    }));
    buildTimelineModel();
  }

  async function ensureThumbnails() {
    for (const clip of state.clips) {
      if (clip.thumbnail_url) continue;
      clip.thumbnail_url = await loadVideoThumbnail(clip.url, Math.min(0.5, Math.max(0.1, (clip.duration || 1) * 0.2)));
      render();
    }
  }

  function timelinePositionForVideo() {
    const clip = activeClip();
    if (!clip) return 0;
    const current = Number.isFinite(video.currentTime) ? video.currentTime : 0;
    return Math.min(state.totalDuration, (clip.timeline_start || 0) + current);
  }

  function updatePlayhead(keepVisible = false) {
    const absoluteTime = timelinePositionForVideo();
    const left = absoluteTime * state.pxPerSecond;
    playhead.style.left = `${left}px`;
    timelineTime.textContent = `${formatTime(absoluteTime)} / ${formatTime(state.totalDuration)}`;
    globalScrub.max = String(Math.max(0, state.totalDuration));
    globalScrub.disabled = !state.clips.length || state.totalDuration <= 0;
    if (!state.isGlobalScrubbing) globalScrub.value = String(Math.min(absoluteTime, state.totalDuration));
    globalScrubTime.textContent = `${formatTime(absoluteTime)} / ${formatTime(state.totalDuration)}`;
    if (!keepVisible) return;
    const viewportLeft = timelineViewport.scrollLeft;
    const viewportRight = viewportLeft + timelineViewport.clientWidth;
    if (left > viewportRight - 80) timelineViewport.scrollLeft = Math.max(0, left - timelineViewport.clientWidth + 120);
    if (left < viewportLeft + 60) timelineViewport.scrollLeft = Math.max(0, left - 80);
  }

  function selectClip(clip, seekTime = null, options = {}) {
    const id = clipId(clip);
    const shouldPlay = Boolean(options.play);
    const targetTime = Math.max(0, Number(seekTime || 0));
    const version = clipVersion(clip);
    const sameClip = state.activeId === id && video.dataset.clipId === id && video.dataset.clipVersion === version;
    state.activeId = clipId(clip);
    if (sameClip) {
      video.currentTime = Math.min(targetTime, Math.max(0, Number(clip.duration || 0) - 0.05));
      if (shouldPlay) video.play().catch(() => {});
      syncInspector();
      render();
      updatePlayhead(true);
      return;
    }
    video.dataset.clipId = id;
    video.dataset.clipVersion = version;
    video.src = clip.url;
    video.load();
    video.addEventListener("loadedmetadata", () => {
      if (seekTime !== null) video.currentTime = Math.min(targetTime, Math.max(0, Number(video.duration || clip.duration || 0) - 0.05));
      updatePlayhead(true);
      if (shouldPlay) video.play().catch(() => {});
    }, { once: true });
    syncInspector();
    render();
    updatePlayhead(true);
  }

  function renderClipRow(clip) {
    const item = clipState(clip);
    const row = document.createElement("button");
    row.type = "button";
    row.style.cssText = `
      width: 100%;
      text-align: left;
      border: 1px solid ${clipId(clip) === state.activeId ? "#06b6d4" : item.selected_for_remake ? "#0e7490" : "#3f3f46"};
      border-radius: 7px;
      background: ${clipId(clip) === state.activeId ? "#164e63" : item.selected_for_remake ? "#083344" : "#27272a"};
      color: #fafafa;
      padding: 8px;
      margin-bottom: 8px;
      cursor: pointer;
    `;
    const name = document.createElement("div");
    name.textContent = clip.name;
    name.style.cssText = "font-size:12px;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    const meta = document.createElement("div");
    meta.textContent = `${item.selected_for_remake ? "Selected" : "Clip"} ${clip.clip_number} | ${formatBytes(clip.size)}`;
    meta.style.cssText = "font-size:11px;color:#a1a1aa;margin-top:4px;";
    row.append(name, meta);
    row.addEventListener("click", () => selectClip(clip));
    return row;
  }

  function renderTimelineClip(clip) {
    const item = clipState(clip);
    const block = document.createElement("button");
    block.type = "button";
    block.title = clip.name;
    const width = Math.max(92, Math.round((clip.duration || 8) * state.pxPerSecond));
    block.style.cssText = `
      flex: 0 0 ${width}px;
      border: 1px solid ${clipId(clip) === state.activeId ? "#67e8f9" : item.selected_for_remake ? "#06b6d4" : "#3f3f46"};
      border-radius: 5px;
      background: ${item.selected_for_remake ? "#0e7490" : "#27272a"};
      color: #fafafa;
      padding: 7px;
      font-size: 11px;
      cursor: pointer;
      overflow: hidden;
      height: 88px;
      margin-top: 10px;
      position: relative;
    `;
    const label = document.createElement("div");
    label.textContent = `video_${String(clip.clip_number).padStart(4, "0")}`;
    label.style.cssText = "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:800;";
    const duration = document.createElement("div");
    duration.textContent = formatTime(clip.duration || 0);
    duration.style.cssText = "position:absolute;right:7px;top:7px;font-size:10px;color:#d4d4d8;font-variant-numeric:tabular-nums;";
    const bar = document.createElement("div");
    const background = clip.thumbnail_url
      ? `linear-gradient(rgba(0,0,0,.05),rgba(0,0,0,.05)), url("${clip.thumbnail_url}") center / cover repeat-x`
      : "repeating-linear-gradient(90deg,#155e75 0 18px,#0891b2 18px 36px,#164e63 36px 54px)";
    bar.style.cssText = `position:absolute;left:7px;right:7px;bottom:8px;height:42px;border-radius:3px;background:${background};`;
    block.append(label, duration, bar);
    return block;
  }

  function render() {
    clipList.textContent = "";
    timelineTrack.textContent = "";
    if (!state.clips.length) {
      const empty = document.createElement("div");
      empty.textContent = "Load an output folder to start.";
      empty.style.cssText = "font-size:13px;color:#a1a1aa;padding:8px;";
      clipList.appendChild(empty);
    }
    for (const clip of state.clips) {
      clipList.appendChild(renderClipRow(clip));
      timelineTrack.appendChild(renderTimelineClip(clip));
    }
    const trackWidth = Math.max(timelineViewport.clientWidth - 24, Math.round(state.totalDuration * state.pxPerSecond));
    timelineTrack.style.width = `${trackWidth}px`;
    timelineTrack.appendChild(playhead);
    updatePlayhead();
    updateSelectedCount();
  }

  function pointerToTimelineTime(event) {
    const bounds = timelineTrack.getBoundingClientRect();
    const x = Math.max(0, event.clientX - bounds.left);
    return Math.min(state.totalDuration, x / state.pxPerSecond);
  }

  function seekAbsoluteTime(absoluteTime, keepPlayback = false) {
    if (!state.clips.length) return;
    const targetTime = Math.max(0, Math.min(state.totalDuration, Number(absoluteTime || 0)));
    const clip = state.clips.find((item) => {
      const start = item.timeline_start || 0;
      return targetTime >= start && targetTime < start + (item.duration || 8);
    }) || state.clips[state.clips.length - 1];
    const offset = Math.max(0, targetTime - (clip.timeline_start || 0));
    if (!keepPlayback) video.pause();
    selectClip(clip, offset, { play: keepPlayback });
  }

  function seekTimeline(event, keepPlayback = false) {
    seekAbsoluteTime(pointerToTimelineTime(event), keepPlayback);
  }

  function startTimelineScrub(event) {
    if (!state.clips.length || event.button > 0) return;
    state.isScrubbing = true;
    timelineViewport.setPointerCapture?.(event.pointerId);
    seekTimeline(event, false);
    event.preventDefault();
  }

  function moveTimelineScrub(event) {
    if (!state.isScrubbing) return;
    seekTimeline(event, false);
    event.preventDefault();
  }

  function stopTimelineScrub(event) {
    if (!state.isScrubbing) return;
    state.isScrubbing = false;
    timelineViewport.releasePointerCapture?.(event.pointerId);
  }

  function scrubGlobalToValue() {
    seekAbsoluteTime(Number(globalScrub.value || 0), false);
  }

  selectedToggle.addEventListener("change", () => {
    const clip = activeClip();
    if (!clip) return;
    clipState(clip).selected_for_remake = selectedToggle.checked;
    render();
    updateSelectedCount();
  });

  notes.addEventListener("input", () => {
    const clip = activeClip();
    if (!clip) return;
    clipState(clip).user_notes = notes.value;
  });

  i2vNotes.addEventListener("input", () => {
    const clip = activeClip();
    if (!clip) return;
    clipState(clip).i2v_user_notes = i2vNotes.value;
  });

  promptBox.addEventListener("input", () => {
    const clip = activeClip();
    if (!clip) return;
    const value = promptBox.value || "";
    clipState(clip).t2i_prompt = value;
    setWidgetValue(node, "generated_t2i_prompt", value);
  });

  i2vPromptBox.addEventListener("input", () => {
    const clip = activeClip();
    if (!clip) return;
    const value = i2vPromptBox.value || "";
    clipState(clip).i2v_prompt = value;
    setWidgetValue(node, "generated_i2v_prompt", value);
  });

  async function captureCurrentFrame() {
    const clip = activeClip();
    if (!clip) {
      toast("Select a clip first.", true);
      return null;
    }
    if (!video.videoWidth || !video.videoHeight) {
      toast("The selected video is not ready yet. Play or seek the clip once, then try again.", true);
      return null;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL("image/png");
    const data = await postJson("/vrgdg/video_editor/save_frame", {
      folder_path: folderInput.value,
      clip_name: clip.name,
      clip_path: clip.path,
      frame_time: video.currentTime || 0,
      image_data: imageData,
    });
    const item = clipState(clip);
    item.captured_frame_path = data.frame_path || "";
    framePath.textContent = item.captured_frame_path || "No frame captured";
    setWidgetValue(node, "captured_frame_path", item.captured_frame_path || "");
    toast(`Frame saved:\n${item.captured_frame_path}`);
    return item.captured_frame_path;
  }

  async function createVisualT2IPrompt() {
    const clip = activeClip();
    if (!clip) {
      toast("Select a clip first.", true);
      return;
    }
    const item = clipState(clip);
    let currentFramePath = item.captured_frame_path || "";
    try {
      frameButton.disabled = true;
      promptButton.disabled = true;
      promptButton.textContent = "Gemma is thinking...";
      const data = await postJson("/vrgdg/video_editor/generate_visual_t2i", {
        folder_path: folderInput.value,
        clip_path: clip.path,
        clip_name: clip.name,
        frame_path: currentFramePath,
        user_notes: notes.value || "",
        model_file: String(getWidget(node, "model_file")?.value || ""),
        mmproj_file: String(getWidget(node, "mmproj_file")?.value || ""),
        unload_after: true,
      });
      item.t2i_prompt = String(data.prompt || "").trim();
      item.captured_frame_path = data.frame_path || currentFramePath;
      promptBox.value = item.t2i_prompt;
      framePath.textContent = item.captured_frame_path || "No frame captured";
      setWidgetValue(node, "captured_frame_path", item.captured_frame_path || "");
      setWidgetValue(node, "generated_t2i_prompt", item.t2i_prompt || "");
      const source = data.used_frame ? "from the captured frame" : "from notes only";
      toast(data.unloaded ? `Gemma created the T2I prompt ${source} and unloaded.` : `Gemma created the T2I prompt ${source}.`);
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      frameButton.disabled = false;
      promptButton.disabled = false;
      promptButton.textContent = "Create T2I Prompt";
    }
  }

  async function createI2VPrompt() {
    const clip = activeClip();
    if (!clip) {
      toast("Select a clip first.", true);
      return;
    }
    const item = clipState(clip);
    const t2iPrompt = String(promptBox.value || item.t2i_prompt || "").trim();
    if (!t2iPrompt) {
      toast("Create a T2I prompt first, then make the I2V prompt.", true);
      setTab("t2i");
      return;
    }
    item.t2i_prompt = t2iPrompt;
    setWidgetValue(node, "generated_t2i_prompt", item.t2i_prompt || "");
    try {
      i2vButton.disabled = true;
      i2vButton.textContent = "Gemma is thinking...";
      const data = await postJson("/vrgdg/video_editor/generate_i2v", {
        folder_path: folderInput.value,
        clip_path: clip.path,
        clip_name: clip.name,
        t2i_prompt: t2iPrompt,
        user_notes: i2vNotes.value || "",
        model_file: getSelectedGemmaModel(node, "i2v_model_file"),
        unload_after: true,
      });
      item.i2v_user_notes = i2vNotes.value || "";
      item.i2v_prompt = String(data.prompt || "").trim();
      i2vPromptBox.value = item.i2v_prompt;
      setWidgetValue(node, "generated_i2v_prompt", item.i2v_prompt || "");
      toast(data.unloaded ? "Gemma created the I2V prompt and unloaded." : "Gemma created the I2V prompt.");
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      i2vButton.disabled = false;
      i2vButton.textContent = "Create I2V Prompt";
    }
  }

  async function loadProject() {
    try {
      refreshButton.disabled = true;
      refreshButton.textContent = "Loading...";
      const [clipsData, sessionData] = await Promise.all([
        postJson("/vrgdg/video_editor/list_clips", {
          folder_path: folderInput.value,
          extensions: extensionsInput.value,
        }),
        postJson("/vrgdg/video_editor/load_session", {
          folder_path: folderInput.value,
        }).catch(() => ({ session: { clips: {} } })),
      ]);
      state.clips = clipsData.clips || [];
      state.session = sessionData.session || { clips: {} };
      state.session.clips = state.session.clips || {};
      state.sessionPath = clipsData.session_path || "";
      folderInput.value = clipsData.folder_path || folderInput.value;
      setWidgetValue(node, "output_folder", folderInput.value);
      setWidgetValue(node, "session_path", state.sessionPath);
      await ensureDurations();
      if (state.clips.length) selectClip(state.clips[0]);
      render();
      syncInspector();
      ensureThumbnails().catch((error) => console.warn("[VRGDG] Timeline thumbnails failed:", error));
      toast(`Loaded ${state.clips.length} clip${state.clips.length === 1 ? "" : "s"}.\nSession:\n${state.sessionPath}`);
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      refreshButton.disabled = false;
      refreshButton.textContent = "Load Clips";
    }
  }

  async function saveSession() {
    try {
      const session = {
        ...state.session,
        clips: state.session.clips || {},
      };
      const data = await postJson("/vrgdg/video_editor/save_session", {
        folder_path: folderInput.value,
        session,
      });
      state.session = data.session || session;
      state.sessionPath = data.session_path || state.sessionPath;
      setWidgetValue(node, "output_folder", folderInput.value);
      setWidgetValue(node, "video_extensions", extensionsInput.value);
      setWidgetValue(node, "session_path", state.sessionPath);
      const stagedCount = Array.isArray(state.session.staged_remakes) ? state.session.staged_remakes.length : 0;
      const stagedLine = stagedCount ? `\nMoved/staged ${stagedCount} selected clip${stagedCount === 1 ? "" : "s"} into remake.` : "";
      toast(`Saved editor session.${stagedLine}\n${state.sessionPath}`);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  async function clearSession() {
    try {
      state.session = { clips: {} };
      for (const clip of state.clips) {
        state.session.clips[clipId(clip)] = defaultClipState(clip);
      }
      const data = await postJson("/vrgdg/video_editor/save_session", {
        folder_path: folderInput.value,
        session: state.session,
      });
      state.session = data.session || state.session;
      state.sessionPath = data.session_path || state.sessionPath;
      setWidgetValue(node, "session_path", state.sessionPath);
      render();
      syncInspector();
      updateSelectedCount();
      toast(`Cleared editor session.\nNo video files were moved or deleted.\n${state.sessionPath}`);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  refreshButton.addEventListener("click", loadProject);
  reviewTab.addEventListener("click", () => setTab("review"));
  t2iTab.addEventListener("click", () => setTab("t2i"));
  i2vTab.addEventListener("click", () => setTab("i2v"));
  frameButton.addEventListener("click", () => {
    captureCurrentFrame().catch((error) => toast(String(error?.message || error), true));
  });
  promptButton.addEventListener("click", createVisualT2IPrompt);
  i2vButton.addEventListener("click", createI2VPrompt);
  saveButton.addEventListener("click", saveSession);
  clearButton.addEventListener("click", clearSession);
  globalScrub.addEventListener("pointerdown", () => {
    state.isGlobalScrubbing = true;
    video.pause();
  });
  globalScrub.addEventListener("input", scrubGlobalToValue);
  globalScrub.addEventListener("change", () => {
    scrubGlobalToValue();
    state.isGlobalScrubbing = false;
    updatePlayhead(true);
  });
  globalScrub.addEventListener("pointerup", () => {
    state.isGlobalScrubbing = false;
    updatePlayhead(true);
  });
  globalScrub.addEventListener("pointercancel", () => {
    state.isGlobalScrubbing = false;
    updatePlayhead(true);
  });
  timelineTrack.addEventListener("click", (event) => seekTimeline(event, false));
  timelineViewport.addEventListener("pointerdown", startTimelineScrub);
  timelineViewport.addEventListener("pointermove", moveTimelineScrub);
  timelineViewport.addEventListener("pointerup", stopTimelineScrub);
  timelineViewport.addEventListener("pointercancel", stopTimelineScrub);
  video.addEventListener("timeupdate", () => updatePlayhead(true));
  video.addEventListener("seeked", () => updatePlayhead(true));
  video.addEventListener("ended", () => {
    const index = state.clips.findIndex((clip) => clipId(clip) === state.activeId);
    const next = state.clips[index + 1];
    if (next) {
      selectClip(next, 0, { play: true });
      return;
    }
    updatePlayhead(true);
  });
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) overlay.remove();
  });

  render();
  setTab("review");
  if (folderInput.value.trim()) loadProject();
}

function ensureButton(node) {
  const buttonName = "Open Video Editor";
  hideInternalWidgets(node);
  node.widgets = (node.widgets || []).filter((widget) => !(widget?.type === "button" && widget?.name === buttonName));
  const widget = node.addWidget("button", buttonName, null, () => openEditor(node));
  if (widget) widget.serialize = false;
  hideInternalWidgets(node);
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
