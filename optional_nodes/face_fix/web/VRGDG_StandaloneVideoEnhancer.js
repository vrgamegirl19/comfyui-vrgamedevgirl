import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "VRGDGStandaloneVideoEnhancer";
const UI_CLASS = "vrgdg-standalone-video-enhancer";

function mediaUrl(path) {
  return api.apiURL(`/vrgdg/video_enhancer/media?path=${encodeURIComponent(path)}&v=${Date.now()}`);
}

async function readJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data?.ok) {
    throw new Error(String(data?.error || `Request failed (HTTP ${response.status})`));
  }
  return data;
}

async function postJson(path, payload) {
  return readJson(await api.fetchApi(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  }));
}

function formatTime(value) {
  const seconds = Math.max(0, Number(value || 0));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  const prefix = hours ? `${String(hours).padStart(2, "0")}:` : "";
  return `${prefix}${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function formatBytes(value) {
  const bytes = Math.max(0, Number(value || 0));
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let amount = bytes;
  let index = -1;
  do {
    amount /= 1024;
    index += 1;
  } while (amount >= 1024 && index < units.length - 1);
  return `${amount.toFixed(amount >= 100 ? 0 : amount >= 10 ? 1 : 2)} ${units[index]}`;
}

function upscaleDimensions(video, resolution) {
  const width = Math.max(1, Number(video?.width || 1));
  const height = Math.max(1, Number(video?.height || 1));
  const targetLongEdge = { "2k": 2560, "3k": 3072, "4k": 3840 }[String(resolution || "").toLowerCase()] || 0;
  const sourceLongEdge = Math.max(width, height);
  if (!targetLongEdge || sourceLongEdge >= targetLongEdge) {
    return [Math.round(width), Math.round(height)];
  }
  const scale = targetLongEdge / sourceLongEdge;
  const outputWidth = Math.max(2, Math.round((width * scale) / 2) * 2);
  const outputHeight = Math.max(2, Math.round((height * scale) / 2) * 2);
  return [outputWidth, outputHeight];
}

function button(label, primary = false) {
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = label;
  element.style.cssText = [
    "min-height:34px",
    "border:1px solid " + (primary ? "#0891b2" : "#475569"),
    "border-radius:7px",
    "background:" + (primary ? "#0e7490" : "#1e293b"),
    "color:#f8fafc",
    "padding:7px 12px",
    "font:800 12px Arial,sans-serif",
    "cursor:pointer",
  ].join(";");
  return element;
}

function input(type, value, min, max, step) {
  const element = document.createElement("input");
  element.type = type;
  element.value = value;
  if (min !== undefined) element.min = String(min);
  if (max !== undefined) element.max = String(max);
  if (step !== undefined) element.step = String(step);
  element.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #475569;border-radius:6px;background:#0f172a;color:#f8fafc;padding:7px 8px;font:12px Arial,sans-serif;";
  return element;
}

function checkbox(checked = false) {
  const element = document.createElement("input");
  element.type = "checkbox";
  element.checked = checked;
  element.style.accentColor = "#06b6d4";
  return element;
}

function select(options, selected) {
  const element = document.createElement("select");
  element.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #475569;border-radius:6px;background:#0f172a;color:#f8fafc;padding:7px 8px;font:12px Arial,sans-serif;";
  for (const [value, label] of options) {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = label;
    element.append(option);
  }
  element.value = String(selected);
  return element;
}

function field(label, control, hint = "") {
  const wrap = document.createElement("label");
  wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;min-width:0;color:#cbd5e1;font:800 11px Arial,sans-serif;";
  const title = document.createElement("span");
  title.textContent = label;
  wrap.append(title, control);
  if (hint) {
    const help = document.createElement("span");
    help.textContent = hint;
    help.style.cssText = "color:#64748b;font:10px/1.35 Arial,sans-serif;";
    wrap.append(help);
  }
  return wrap;
}

function checkField(label, control, hint = "") {
  const wrap = document.createElement("label");
  wrap.style.cssText = "display:grid;grid-template-columns:auto minmax(0,1fr);gap:8px;align-items:start;color:#e2e8f0;font:800 12px Arial,sans-serif;cursor:pointer;";
  const text = document.createElement("span");
  text.textContent = label;
  wrap.append(control, text);
  if (hint) {
    const help = document.createElement("span");
    help.textContent = hint;
    help.style.cssText = "grid-column:2;color:#64748b;font:10px/1.35 Arial,sans-serif;";
    wrap.append(help);
  }
  return wrap;
}

function section(title) {
  const wrap = document.createElement("section");
  wrap.style.cssText = "border:1px solid #334155;border-radius:9px;background:#111827;padding:12px;display:flex;flex-direction:column;gap:10px;";
  const heading = document.createElement("div");
  heading.textContent = title;
  heading.style.cssText = "font:900 13px Arial,sans-serif;color:#a5f3fc;";
  wrap.append(heading);
  return wrap;
}

function createWipeStage(kind = "image") {
  const stage = document.createElement("div");
  stage.style.cssText = "position:relative;width:100%;min-height:300px;aspect-ratio:16/9;overflow:hidden;border:1px solid #334155;border-radius:9px;background:#020617;touch-action:none;cursor:ew-resize;";
  const before = document.createElement(kind === "video" ? "video" : "img");
  const after = document.createElement(kind === "video" ? "video" : "img");
  for (const media of [before, after]) {
    media.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#020617;pointer-events:none;";
    if (kind === "video") {
      media.preload = "metadata";
      media.playsInline = true;
      media.disablePictureInPicture = true;
    } else {
      media.alt = kind === "image" ? "Enhancement preview" : "";
    }
  }
  if (kind === "video") after.muted = true;
  const afterClip = document.createElement("div");
  afterClip.style.cssText = "position:absolute;inset:0;overflow:hidden;pointer-events:none;";
  afterClip.append(after);
  const divider = document.createElement("div");
  divider.style.cssText = "position:absolute;top:0;bottom:0;width:2px;background:#fff;box-shadow:0 0 12px rgba(255,255,255,.75);transform:translateX(-1px);pointer-events:none;";
  const handle = document.createElement("div");
  handle.textContent = "↔";
  handle.style.cssText = "position:absolute;top:50%;width:36px;height:36px;border-radius:50%;background:#fff;color:#0f172a;display:flex;align-items:center;justify-content:center;font:900 18px Arial;box-shadow:0 4px 18px rgba(0,0,0,.65);transform:translate(-50%,-50%);pointer-events:none;";
  const beforeLabel = document.createElement("div");
  beforeLabel.textContent = "BEFORE";
  beforeLabel.style.cssText = "position:absolute;left:10px;top:10px;padding:5px 7px;border-radius:5px;background:rgba(0,0,0,.7);color:#fff;font:900 10px Arial;pointer-events:none;";
  const afterLabel = document.createElement("div");
  afterLabel.textContent = "AFTER";
  afterLabel.style.cssText = beforeLabel.style.cssText.replace("left:10px", "right:10px");
  stage.append(before, afterClip, divider, handle, beforeLabel, afterLabel);
  let dragging = false;
  let wipe = 0.5;
  const setWipe = (value) => {
    wipe = Math.max(0, Math.min(1, Number(value ?? 0.5)));
    const percent = wipe * 100;
    afterClip.style.clipPath = `inset(0 0 0 ${percent}%)`;
    divider.style.left = `${percent}%`;
    handle.style.left = `${percent}%`;
  };
  const fromPointer = (event) => {
    const rect = stage.getBoundingClientRect();
    if (rect.width) setWipe((event.clientX - rect.left) / rect.width);
  };
  stage.addEventListener("pointerdown", (event) => {
    dragging = true;
    stage.setPointerCapture?.(event.pointerId);
    fromPointer(event);
  });
  stage.addEventListener("pointermove", (event) => {
    if (dragging) fromPointer(event);
  });
  stage.addEventListener("pointerup", (event) => {
    dragging = false;
    stage.releasePointerCapture?.(event.pointerId);
  });
  stage.addEventListener("pointercancel", () => { dragging = false; });
  setWipe(0.5);
  return { stage, before, after, setWipe, getWipe: () => wipe };
}

function setOutputWidget(node, value) {
  const widget = node.widgets?.find((item) => item.name === "output_path");
  if (!widget) return;
  widget.value = String(value || "");
  widget.callback?.(widget.value);
}

function openEnhancer(node) {
  document.querySelector(`.${UI_CLASS}`)?.remove();

  const state = {
    source: null,
    jobId: "",
    renderedPath: "",
    pollTimer: 0,
    closed: false,
  };
  const saved = node.properties?.vrgdg_video_enhancer || {};

  const backdrop = document.createElement("div");
  backdrop.className = UI_CLASS;
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100100;background:rgba(2,6,23,.82);display:flex;align-items:center;justify-content:center;padding:12px;box-sizing:border-box;";
  const panel = document.createElement("div");
  panel.style.cssText = "width:min(1600px,calc(100vw - 24px));height:calc(100vh - 24px);border:1px solid #155e75;border-radius:12px;background:#0b1220;color:#f8fafc;box-shadow:0 28px 100px rgba(0,0,0,.72);display:flex;flex-direction:column;overflow:hidden;";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 16px;border-bottom:1px solid #334155;background:#0f172a;";
  const headingWrap = document.createElement("div");
  const heading = document.createElement("div");
  heading.textContent = "VRGDG Standalone Video Enhancer";
  heading.style.cssText = "font:950 18px Arial,sans-serif;color:#cffafe;";
  const subtitle = document.createElement("div");
  subtitle.textContent = "Fast 2K–4K resize + sharpen + film grain with checkpointed rendering and comparison";
  subtitle.style.cssText = "margin-top:3px;color:#67e8f9;font:11px Arial,sans-serif;";
  headingWrap.append(heading, subtitle);
  const close = button("Close");
  header.append(headingWrap, close);
  const body = document.createElement("div");
  body.style.cssText = "flex:1 1 auto;min-height:0;overflow:auto;padding:14px;display:grid;grid-template-columns:minmax(0,1.7fr) minmax(320px,.85fr);gap:14px;align-items:start;";
  const left = document.createElement("div");
  left.style.cssText = "display:flex;flex-direction:column;gap:12px;min-width:0;";
  const right = document.createElement("div");
  right.style.cssText = "display:flex;flex-direction:column;gap:12px;min-width:0;";

  const sourceSection = section("1. Source video");
  const drop = document.createElement("div");
  drop.textContent = "Drop a video here or click Choose Video";
  drop.style.cssText = "border:2px dashed #0e7490;border-radius:9px;background:#082f49;color:#cffafe;padding:18px;text-align:center;font:900 12px Arial;cursor:pointer;";
  const filePicker = document.createElement("input");
  filePicker.type = "file";
  filePicker.accept = "video/mp4,video/quicktime,video/webm,video/x-matroska,video/x-msvideo,.m4v";
  filePicker.style.display = "none";
  const sourceActions = document.createElement("div");
  sourceActions.style.cssText = "display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:8px;align-items:center;";
  const choose = button("Choose Video", true);
  const localPath = input("text", saved.source_path || "");
  localPath.placeholder = "Or enter a video path on the ComfyUI server";
  const loadPath = button("Load Path");
  sourceActions.append(choose, localPath, loadPath);
  const metadataLine = document.createElement("div");
  metadataLine.textContent = "No video loaded.";
  metadataLine.style.cssText = "color:#94a3b8;font:11px/1.45 Arial,sans-serif;";
  sourceSection.append(drop, filePicker, sourceActions, metadataLine);

  const sourceVideo = document.createElement("video");
  sourceVideo.controls = true;
  sourceVideo.preload = "metadata";
  sourceVideo.playsInline = true;
  sourceVideo.style.cssText = "width:100%;max-height:430px;border:1px solid #334155;border-radius:9px;background:#020617;";
  const playheadLine = document.createElement("div");
  playheadLine.textContent = "Playhead: 00:00.00";
  playheadLine.style.cssText = "color:#67e8f9;font:800 11px Arial,sans-serif;";

  const previewSection = section("2. Current-frame preview");
  const stillCompare = createWipeStage("image");
  stillCompare.stage.style.display = "none";
  const previewHint = document.createElement("div");
  previewHint.textContent = "Seek the source video to any frame, then generate a fast one-frame comparison.";
  previewHint.style.cssText = "display:flex;align-items:center;justify-content:center;min-height:180px;border:1px dashed #334155;border-radius:9px;color:#64748b;font:800 12px Arial;text-align:center;padding:18px;";
  const previewButton = button("Preview Current Frame", true);
  previewButton.disabled = true;
  previewSection.append(previewHint, stillCompare.stage, previewButton);

  const finalSection = section("Final video comparison");
  finalSection.style.display = "none";
  finalSection.style.gridColumn = "1 / -1";
  const videoCompare = createWipeStage("video");
  const compareToolbar = document.createElement("div");
  compareToolbar.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;";
  const compareHelp = document.createElement("div");
  compareHelp.textContent = "Drag anywhere on the video to move the before/after divider.";
  compareHelp.style.cssText = "color:#94a3b8;font:11px/1.4 Arial,sans-serif;";
  const expandCompare = button("⛶ Expand to Screen", true);
  compareToolbar.append(compareHelp, expandCompare);
  const videoControls = document.createElement("div");
  videoControls.style.cssText = "display:grid;grid-template-columns:auto auto minmax(100px,1fr) auto;gap:8px;align-items:center;";
  const comparePlay = button("▶");
  const compareRestart = button("↺");
  const compareScrub = input("range", "0", 0, 0, 0.01);
  const compareTime = document.createElement("div");
  compareTime.textContent = "00:00.00 / 00:00.00";
  compareTime.style.cssText = "white-space:nowrap;color:#67e8f9;font:800 10px Arial;";
  videoControls.append(comparePlay, compareRestart, compareScrub, compareTime);
  finalSection.append(compareToolbar, videoCompare.stage, videoControls);

  const expandedCompare = document.createElement("div");
  expandedCompare.style.cssText = "position:fixed;inset:0;z-index:100300;display:none;flex-direction:column;gap:10px;padding:10px;box-sizing:border-box;background:#020617;color:#f8fafc;";
  const expandedHeader = document.createElement("div");
  expandedHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;flex:0 0 auto;";
  const expandedTitle = document.createElement("div");
  expandedTitle.textContent = "Before / After Video Comparison";
  expandedTitle.style.cssText = "font:950 16px Arial,sans-serif;color:#cffafe;";
  const collapseCompare = button("Exit Expanded View");
  expandedHeader.append(expandedTitle, collapseCompare);
  const expandedStageHost = document.createElement("div");
  expandedStageHost.style.cssText = "flex:1 1 auto;min-height:0;display:flex;align-items:stretch;justify-content:stretch;";
  const expandedControlsHost = document.createElement("div");
  expandedControlsHost.style.cssText = "flex:0 0 auto;";
  expandedCompare.append(expandedHeader, expandedStageHost, expandedControlsHost);

  const upscaleSection = section("Fake upscale / Output size");
  const upscaleResolution = select([
    ["original", "Original resolution"],
    ["2k", "2K / 1440p (2560px long edge)"],
    ["3k", "3K (3072px long edge)"],
    ["4k", "4K UHD / 2160p (3840px long edge)"],
  ], saved.upscale_resolution || "original");
  const upscaleSummary = document.createElement("div");
  upscaleSummary.style.cssText = "border:1px solid #164e63;border-radius:7px;background:#082f49;color:#a5f3fc;padding:8px;font:800 11px/1.4 Arial,sans-serif;";
  const upscaleNote = document.createElement("div");
  upscaleNote.textContent = "Uses high-quality Lanczos resizing before sharpen and grain. This enlarges the video without inventing AI detail.";
  upscaleNote.style.cssText = "color:#94a3b8;font:10px/1.4 Arial,sans-serif;";
  upscaleSection.append(
    field("Target resolution", upscaleResolution, "Preserves aspect ratio and orientation; never reduces a larger source."),
    upscaleSummary,
    upscaleNote,
  );

  const sharpenSection = section("Sharpen");
  const sharpenEnabled = checkbox(saved.sharpen_enabled ?? true);
  const sharpenStrength = input("number", String(saved.sharpen_strength ?? 0.5), 0, 10, 0.01);
  sharpenSection.append(
    checkField("Enable Fast Unsharp Sharpen", sharpenEnabled),
    field("Strength", sharpenStrength, "0 is unchanged; 10 is the strongest setting."),
  );

  const grainSection = section("Film grain");
  const grainEnabled = checkbox(saved.grain_enabled ?? false);
  const grainIntensity = input("number", String(saved.grain_intensity ?? 0.04), 0, 1, 0.001);
  const saturationMix = input("number", String(saved.saturation_mix ?? 0.5), 0, 1, 0.01);
  const grainSeed = input("number", String(saved.seed ?? 42), 0, 2147483647, 1);
  grainSection.append(
    checkField("Enable Fast Film Grain", grainEnabled),
    field("Grain intensity", grainIntensity),
    field("Saturation mix", saturationMix, "0 is monochrome grain; 1 is fully colored grain."),
    field("Seed", grainSeed, "Combined with the absolute frame number for stable results."),
  );

  const renderSection = section("Render");
  const preserveAudio = checkbox(saved.preserve_audio ?? true);
  const outputName = input("text", saved.output_name || "enhanced_video.mp4");
  const quality = select([
    [16, "Very high quality (CRF 16)"],
    [18, "High quality (CRF 18)"],
    [20, "Balanced (CRF 20)"],
    [23, "Smaller file (CRF 23)"],
  ], saved.encode_crf ?? 18);
  const renderButton = button("Render Enhanced Video", true);
  renderButton.disabled = true;
  const cancelButton = button("Cancel Render");
  cancelButton.disabled = true;
  const resumeButton = button("Resume From Checkpoint");
  resumeButton.style.display = "none";
  const renderActions = document.createElement("div");
  renderActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  renderActions.append(renderButton, cancelButton, resumeButton);
  const progressTrack = document.createElement("div");
  progressTrack.style.cssText = "height:12px;border:1px solid #334155;border-radius:999px;background:#020617;overflow:hidden;";
  const progressFill = document.createElement("div");
  progressFill.style.cssText = "width:0%;height:100%;background:linear-gradient(90deg,#0891b2,#22d3ee);transition:width .2s;";
  progressTrack.append(progressFill);
  const renderStatus = document.createElement("pre");
  renderStatus.textContent = "Load a video to begin.";
  renderStatus.style.cssText = "min-height:66px;white-space:pre-wrap;margin:0;border:1px solid #334155;border-radius:7px;background:#020617;color:#94a3b8;padding:9px;font:10px/1.45 monospace;";
  const outputActions = document.createElement("div");
  outputActions.style.cssText = "display:none;grid-template-columns:1fr 1fr;gap:8px;";
  const download = document.createElement("a");
  download.textContent = "Download Result";
  download.style.cssText = button("", true).style.cssText + "text-decoration:none;text-align:center;box-sizing:border-box;";
  const openOutput = document.createElement("a");
  openOutput.textContent = "Open Result";
  openOutput.target = "_blank";
  openOutput.rel = "noopener";
  openOutput.style.cssText = button("").style.cssText + "text-decoration:none;text-align:center;box-sizing:border-box;";
  outputActions.append(download, openOutput);
  renderSection.append(
    checkField("Preserve original audio", preserveAudio),
    field("Output filename", outputName),
    field("Quality", quality),
    renderActions,
    progressTrack,
    renderStatus,
    outputActions,
  );

  const advanced = document.createElement("details");
  advanced.style.cssText = "border:1px solid #334155;border-radius:9px;background:#111827;padding:10px;";
  const advancedTitle = document.createElement("summary");
  advancedTitle.textContent = "Advanced batching";
  advancedTitle.style.cssText = "cursor:pointer;color:#a5f3fc;font:900 12px Arial;";
  const advancedGrid = document.createElement("div");
  advancedGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:10px;";
  const useGpu = checkbox(saved.use_gpu ?? true);
  const batchSize = input("number", String(saved.batch_size ?? 0), 0, 128, 1);
  const segmentSeconds = input("number", String(saved.segment_seconds ?? 30), 5, 300, 1);
  const preset = select([
    ["veryfast", "Very fast"],
    ["fast", "Fast"],
    ["medium", "Medium"],
    ["slow", "Slow"],
  ], saved.encode_preset || "medium");
  advancedGrid.append(
    checkField("Use GPU when available", useGpu),
    field("Frame batch size", batchSize, "0 chooses automatically and retries smaller on GPU OOM."),
    field("Checkpoint seconds", segmentSeconds),
    field("Final encode preset", preset),
  );
  advanced.append(advancedTitle, advancedGrid);

  left.append(sourceSection, sourceVideo, playheadLine, previewSection);
  right.append(upscaleSection, sharpenSection, grainSection, renderSection, advanced);
  body.append(left, right, finalSection);
  panel.append(header, body);
  backdrop.append(panel, expandedCompare);
  document.body.append(backdrop);

  const collapsedStageStyle = videoCompare.stage.style.cssText;
  const collapsedControlsStyle = videoControls.style.cssText;
  const showExpandedComparison = () => {
    if (expandedCompare.style.display === "flex") return;
    expandedStageHost.append(videoCompare.stage);
    expandedControlsHost.append(videoControls);
    videoCompare.stage.style.cssText = `${collapsedStageStyle};height:100%;min-height:0;aspect-ratio:auto;flex:1 1 auto;`;
    videoControls.style.cssText = `${collapsedControlsStyle};padding:2px 4px;`;
    expandedCompare.style.display = "flex";
    videoCompare.setWipe(videoCompare.getWipe());
  };
  const hideExpandedComparison = () => {
    if (expandedCompare.style.display !== "flex") return;
    compareToolbar.after(videoCompare.stage);
    videoCompare.stage.after(videoControls);
    videoCompare.stage.style.cssText = collapsedStageStyle;
    videoControls.style.cssText = collapsedControlsStyle;
    expandedCompare.style.display = "none";
    videoCompare.setWipe(videoCompare.getWipe());
  };
  expandCompare.onclick = showExpandedComparison;
  collapseCompare.onclick = hideExpandedComparison;
  videoCompare.stage.addEventListener("dblclick", () => {
    if (expandedCompare.style.display === "flex") hideExpandedComparison();
    else showExpandedComparison();
  });

  const settings = () => ({
    upscale_resolution: String(upscaleResolution.value || "original"),
    sharpen_enabled: Boolean(sharpenEnabled.checked),
    sharpen_strength: Number(sharpenStrength.value || 0),
    grain_enabled: Boolean(grainEnabled.checked),
    grain_intensity: Number(grainIntensity.value || 0),
    saturation_mix: Number(saturationMix.value || 0),
    seed: Number(grainSeed.value || 42),
    use_gpu: Boolean(useGpu.checked),
    batch_size: Number(batchSize.value || 0),
    segment_seconds: Number(segmentSeconds.value || 30),
    encode_crf: Number(quality.value || 18),
    encode_preset: String(preset.value || "medium"),
    preserve_audio: Boolean(preserveAudio.checked),
    output_name: String(outputName.value || "enhanced_video.mp4"),
  });

  const persist = () => {
    node.properties = node.properties || {};
    node.properties.vrgdg_video_enhancer = {
      source_path: state.source?.path || localPath.value.trim(),
      ...settings(),
    };
  };

  const updateUpscaleSummary = () => {
    if (!state.source) {
      upscaleSummary.textContent = "Load a video to see the exact output dimensions.";
      return;
    }
    const [width, height] = upscaleDimensions(state.source, upscaleResolution.value);
    const unchanged = width === Number(state.source.width) && height === Number(state.source.height);
    upscaleSummary.textContent = upscaleResolution.value === "original"
      ? `Output: ${width}×${height} (no resize)`
      : unchanged
      ? `Output: ${width}×${height} (source is already this size or larger)`
      : `Output: ${width}×${height} from ${state.source.width}×${state.source.height}`;
  };
  upscaleResolution.onchange = () => {
    updateUpscaleSummary();
    persist();
  };
  updateUpscaleSummary();

  const setBusy = (busy) => {
    renderButton.disabled = busy || !state.source;
    previewButton.disabled = busy || !state.source;
    cancelButton.disabled = !busy;
    choose.disabled = busy;
    loadPath.disabled = busy;
  };

  const setSource = (video) => {
    state.source = video;
    localPath.value = video.path || "";
    sourceVideo.src = mediaUrl(video.path);
    sourceVideo.load();
    metadataLine.textContent = `${video.name} · ${video.width}×${video.height} · ${Number(video.fps).toFixed(3)} FPS · ${formatTime(video.duration)} · ${formatBytes(video.size)} · Audio: ${video.has_audio === false ? "No" : video.has_audio === true ? "Yes" : "Unknown"}`;
    updateUpscaleSummary();
    renderStatus.textContent = "Ready. Preview a frame or render the complete video.";
    previewButton.disabled = false;
    renderButton.disabled = false;
    finalSection.style.display = "none";
    state.renderedPath = "";
    outputActions.style.display = "none";
    persist();
  };

  const uploadFile = async (file) => {
    if (!file) return;
    setBusy(true);
    renderStatus.textContent = `Uploading ${file.name}…`;
    try {
      const form = new FormData();
      form.append("video", file, file.name);
      const data = await readJson(await api.fetchApi("/vrgdg/video_enhancer/upload", {
        method: "POST",
        body: form,
      }));
      setSource(data.video);
    } catch (error) {
      renderStatus.textContent = `Upload error: ${String(error?.message || error)}`;
    } finally {
      setBusy(false);
    }
  };

  choose.onclick = () => filePicker.click();
  drop.onclick = () => filePicker.click();
  filePicker.onchange = () => {
    const file = filePicker.files?.[0];
    filePicker.value = "";
    uploadFile(file);
  };
  for (const eventName of ["dragenter", "dragover"]) {
    drop.addEventListener(eventName, (event) => {
      event.preventDefault();
      drop.style.borderColor = "#67e8f9";
    });
  }
  drop.addEventListener("dragleave", () => { drop.style.borderColor = "#0e7490"; });
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.style.borderColor = "#0e7490";
    uploadFile(event.dataTransfer?.files?.[0]);
  });
  loadPath.onclick = async () => {
    const path = localPath.value.trim();
    if (!path) return;
    setBusy(true);
    renderStatus.textContent = "Loading video metadata…";
    try {
      const data = await postJson("/vrgdg/video_enhancer/load", { path });
      setSource(data.video);
    } catch (error) {
      renderStatus.textContent = `Load error: ${String(error?.message || error)}`;
    } finally {
      setBusy(false);
    }
  };

  sourceVideo.addEventListener("timeupdate", () => {
    playheadLine.textContent = `Playhead: ${formatTime(sourceVideo.currentTime)} / ${formatTime(sourceVideo.duration)}`;
  });

  previewButton.onclick = async () => {
    if (!state.source) return;
    persist();
    previewButton.disabled = true;
    renderStatus.textContent = `Processing frame at ${formatTime(sourceVideo.currentTime)}…`;
    try {
      const data = await postJson("/vrgdg/video_enhancer/preview", {
        source_path: state.source.path,
        timestamp: sourceVideo.currentTime || 0,
        settings: settings(),
      });
      stillCompare.before.src = mediaUrl(data.before_path);
      stillCompare.after.src = mediaUrl(data.after_path);
      previewHint.style.display = "none";
      stillCompare.stage.style.display = "block";
      renderStatus.textContent = `Preview ready at ${data.output_width}×${data.output_height}: frame ${Number(data.frame_index).toLocaleString()} at ${formatTime(data.timestamp)}. Drag the divider to compare.`;
    } catch (error) {
      renderStatus.textContent = `Preview error: ${String(error?.message || error)}`;
    } finally {
      previewButton.disabled = false;
    }
  };

  let compareDuration = 0;
  let compareAnimation = 0;
  const updateCompareTime = () => {
    const current = Number(videoCompare.before.currentTime || 0);
    if (document.activeElement !== compareScrub) compareScrub.value = String(current);
    compareTime.textContent = `${formatTime(current)} / ${formatTime(compareDuration)}`;
  };
  const pauseCompare = () => {
    videoCompare.before.pause();
    videoCompare.after.pause();
    comparePlay.textContent = "▶";
    if (compareAnimation) cancelAnimationFrame(compareAnimation);
    compareAnimation = 0;
  };
  const syncCompare = () => {
    if (videoCompare.before.paused) {
      compareAnimation = 0;
      return;
    }
    if (Math.abs(videoCompare.after.currentTime - videoCompare.before.currentTime) > 0.08) {
      videoCompare.after.currentTime = videoCompare.before.currentTime;
    }
    updateCompareTime();
    compareAnimation = requestAnimationFrame(syncCompare);
  };
  const playCompare = async () => {
    try {
      videoCompare.after.currentTime = videoCompare.before.currentTime;
      videoCompare.after.muted = true;
      await Promise.all([videoCompare.before.play(), videoCompare.after.play()]);
      comparePlay.textContent = "Ⅱ";
      compareAnimation = requestAnimationFrame(syncCompare);
    } catch (error) {
      pauseCompare();
      renderStatus.textContent = `Comparison playback error: ${String(error?.message || error)}`;
    }
  };
  const seekCompare = (value) => {
    const time = Math.max(0, Math.min(compareDuration || 0, Number(value || 0)));
    videoCompare.before.currentTime = time;
    videoCompare.after.currentTime = time;
    updateCompareTime();
  };
  const loadFinalComparison = (beforePath, afterPath) => {
    pauseCompare();
    videoCompare.before.src = mediaUrl(beforePath);
    videoCompare.after.src = mediaUrl(afterPath);
    videoCompare.before.load();
    videoCompare.after.load();
    const metadataLoaded = () => {
      const durations = [videoCompare.before.duration, videoCompare.after.duration]
        .map(Number)
        .filter((value) => Number.isFinite(value) && value > 0);
      compareDuration = durations.length ? Math.min(...durations) : 0;
      compareScrub.max = String(compareDuration);
      updateCompareTime();
    };
    videoCompare.before.onloadedmetadata = metadataLoaded;
    videoCompare.after.onloadedmetadata = metadataLoaded;
    finalSection.style.display = "flex";
    finalSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };
  comparePlay.onclick = () => videoCompare.before.paused ? playCompare() : pauseCompare();
  compareRestart.onclick = () => {
    const playing = !videoCompare.before.paused;
    seekCompare(0);
    if (playing) playCompare();
  };
  compareScrub.oninput = () => seekCompare(compareScrub.value);

  const finishJob = (job) => {
    setBusy(false);
    progressFill.style.width = "100%";
    state.renderedPath = job.output_path;
    setOutputWidget(node, job.output_path);
    outputActions.style.display = "grid";
    download.href = mediaUrl(job.output_path);
    download.download = String(job.output_path || "").split(/[\\/]/).pop() || "enhanced_video.mp4";
    openOutput.href = mediaUrl(job.output_path);
    const outputSize = job.output_metadata?.width && job.output_metadata?.height
      ? `\nOutput: ${job.output_metadata.width}×${job.output_metadata.height}`
      : "";
    renderStatus.textContent = `Enhancement complete.\n${job.output_path}${outputSize}\n${Number(job.frames_processed || 0).toLocaleString()} frames rendered.`;
    loadFinalComparison(state.source.path, job.output_path);
    resumeButton.style.display = "none";
    persist();
  };

  const pollJob = async () => {
    if (!state.jobId || state.closed) return;
    try {
      const response = await api.fetchApi(`/vrgdg/video_enhancer/render/status?job_id=${encodeURIComponent(state.jobId)}`);
      const data = await readJson(response);
      const job = data.job || {};
      progressFill.style.width = `${Math.max(0, Math.min(100, Number(job.progress || 0) * 100))}%`;
      const checkpoint = Number(job.total_segments || 0)
        ? `\nCheckpoint ${job.segment_index || 0}/${job.total_segments}`
        : "";
      const batch = Number(job.batch_size || 0) ? ` · batch ${job.batch_size}` : "";
      renderStatus.textContent = `${job.message || job.status || "Working…"}${checkpoint}${batch}\nJob: ${state.jobId}`;
      if (job.status === "complete") {
        window.clearTimeout(state.pollTimer);
        finishJob(job);
        return;
      }
      if (job.status === "failed" || job.status === "canceled") {
        window.clearTimeout(state.pollTimer);
        setBusy(false);
        resumeButton.style.display = job.can_resume ? "" : "none";
        renderStatus.textContent = `${job.message || job.status}${job.error ? `\n${job.error}` : ""}\nCompleted checkpoints were kept for resume.\nJob: ${state.jobId}`;
        return;
      }
      state.pollTimer = window.setTimeout(pollJob, 750);
    } catch (error) {
      renderStatus.textContent = `Status error: ${String(error?.message || error)}\nRetrying…`;
      state.pollTimer = window.setTimeout(pollJob, 1500);
    }
  };

  const startRender = async (resume = false) => {
    if (!state.source) return;
    persist();
    setBusy(true);
    resumeButton.style.display = "none";
    outputActions.style.display = "none";
    progressFill.style.width = "0%";
    renderStatus.textContent = resume ? "Resuming completed checkpoints…" : "Starting batched render…";
    try {
      const data = await postJson("/vrgdg/video_enhancer/render/start", {
        source_path: state.source.path,
        settings: settings(),
        resume_job_id: resume ? state.jobId : "",
      });
      state.jobId = data.job.job_id;
      window.clearTimeout(state.pollTimer);
      pollJob();
    } catch (error) {
      setBusy(false);
      renderStatus.textContent = `Render error: ${String(error?.message || error)}`;
    }
  };
  renderButton.onclick = () => startRender(false);
  resumeButton.onclick = () => startRender(true);
  cancelButton.onclick = async () => {
    if (!state.jobId) return;
    cancelButton.disabled = true;
    renderStatus.textContent = "Canceling after the current frame batch…";
    try {
      await postJson("/vrgdg/video_enhancer/render/cancel", { job_id: state.jobId });
    } catch (error) {
      renderStatus.textContent = `Cancel error: ${String(error?.message || error)}`;
    }
  };

  const onKeyDown = (event) => {
    if (event.key !== "Escape") return;
    if (expandedCompare.style.display === "flex") {
      hideExpandedComparison();
      return;
    }
    destroy();
  };
  const destroy = () => {
    state.closed = true;
    window.clearTimeout(state.pollTimer);
    sourceVideo.pause();
    pauseCompare();
    document.removeEventListener("keydown", onKeyDown);
    backdrop.remove();
  };
  document.addEventListener("keydown", onKeyDown);
  close.onclick = destroy;
  backdrop.addEventListener("pointerdown", (event) => {
    if (event.target === backdrop) destroy();
  });
  backdrop.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
  panel.addEventListener("pointerdown", (event) => event.stopPropagation());

  if (saved.source_path) {
    localPath.value = saved.source_path;
    loadPath.click();
  }
}

app.registerExtension({
  name: "vrgdg." + NODE_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.size = [Math.max(360, this.size?.[0] || 360), Math.max(150, this.size?.[1] || 150)];
      const launch = button("Open Standalone Video Enhancer", true);
      launch.style.width = "100%";
      launch.style.height = "48px";
      launch.onclick = () => openEnhancer(this);
      const root = document.createElement("div");
      root.style.cssText = "width:100%;height:62px;box-sizing:border-box;padding:7px;background:#111827;";
      root.append(launch);
      const widget = this.addDOMWidget("video_enhancer_ui", "vrgdg-video-enhancer", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 62,
        getMaxHeight: () => 62,
      });
      if (widget) {
        widget.serialize = false;
        widget.computeSize = (width) => [width, 62];
      }
      return result;
    };
  },
});
