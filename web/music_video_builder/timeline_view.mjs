import { normalizeOverlayClip } from "../VRGDG_OverlayTrack.js";
import { makeEditorImageUrl, makeEditorThumbnailUrl } from "./comfy_api.mjs";
import {
  TIMELINE_HEIGHT,
  TIMELINE_MARKER_HEIGHT,
  TIMELINE_MARKER_MIN_WIDTH,
  TIMELINE_NOTE_GAP,
  TIMELINE_NOTE_HEIGHT,
  TIMELINE_OVERLAY_HEIGHT,
  TIMELINE_OVERLAY_TOP,
  TIMELINE_SCENE_AUDIO_HEIGHT,
  TIMELINE_SCENE_AUDIO_TOP,
  TIMELINE_SEGMENT_HEIGHT,
  TIMELINE_SEGMENT_TOP,
  WAVEFORM_MODES,
} from "./constants.mjs";
import { escapeHtml, makeButton, makeCheckbox, makeInput, makeSelect, normalizeProjectVideoEngine, toast } from "./controls.mjs";
import { miniMaxI2VFLFEnabled } from "./minimax_keyframe_state.mjs";
import { formatDurationSeconds, formatTime } from "./format.mjs";
import { applyLyricSectionsFromReferenceText } from "./lyric_transcription.mjs";
import {
  filmGrainLabel,
  lutLabelFromName,
  normalizeSceneFilmGrain,
  normalizeSceneLut,
  sceneFilmGrainBadgeHtml,
  sceneLutBadgeHtml,
} from "./post_process.mjs";
import { cloneMiniMaxH3Settings, isMiniMaxH3ContinuityAllowedForMode } from "./minimax_h3.mjs";
import { isInstrumentalLyricText } from "./prompt_text.mjs";
import { newTimelineMarker, sortSegments } from "./segments.mjs";
import { appendTimelineVideoThumbnail, hasLockedVideo, selectedSegmentVideoPath } from "./selection_preview.mjs";
import { parseBulkTimeValue } from "./timeline_actions.mjs";
import { audioChunkDuration, audioTimelineStart, markerEnd, normalizeTimelineMarkers } from "./timeline_state.mjs";
import { mappedLocation } from "./scene_locations.mjs";
import { createTimelineToolWindows } from "./timeline_tool_windows.mjs";
import {
  STEM_COLORS, STEM_LABELS, getStemLanes, requestOpenAudioMask, requestStemEdit, stemRowNames,
} from "./audio_mask_store.mjs";

// One track per stem (vocals, drums, bass, guitar, piano, other) and a masked mix track, under the scenes that have been
// split in the Audio Mask window. A track is as wide as its scene and as tall as a waveform row.
const STEM_ROW_HEIGHT = TIMELINE_SEGMENT_HEIGHT; // as tall as a scene block
const STEM_ROW_GAP = 4;
const STEM_EDGE_GRAB_PX = 6;
const STEM_MIN_REGION_SECONDS = 0.02;
// The rows shown, top to bottom ("mix" is last), and the height of the band between the scene-audio row and everything
// below it. Both are empty or 0 when no stem tracks are shown.
let stemRowList = [];
let stemBandHeight = 0;

export function shiftSegmentTiming(segment, delta) {
  const amount = Number(delta || 0);
  if (!segment || !amount) return;
  segment.start = Number(segment.start || 0) + amount;
  segment.end = Number(segment.end || 0) + amount;
  if (Number.isFinite(Number(segment.custom_audio_timeline_start))) {
    segment.custom_audio_timeline_start = Number(segment.custom_audio_timeline_start || 0) + amount;
  }
}

function timelineNoteTop() {
  return TIMELINE_SCENE_AUDIO_TOP + TIMELINE_SCENE_AUDIO_HEIGHT + stemBandHeight + TIMELINE_NOTE_GAP;
}

// Drawn stem tracks, reused between timeline redraws while nothing about them has changed. A redraw happens on every scene
// change, and drawing every track of every scene each time would stall playback.
const stemRowCache = new Map();

// A track is drawn when it is about to scroll into view, not when the timeline is built: a project with many scenes has
// hundreds of tracks and only a few are on screen. A drawn track stays drawn (it is in the cache above).
const stemDrawWaiting = new WeakMap();
const stemDrawObserver = typeof IntersectionObserver === "function"
  ? new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const draw = stemDrawWaiting.get(entry.target);
      stemDrawWaiting.delete(entry.target);
      stemDrawObserver.unobserve(entry.target);
      if (draw) draw();
    }
  }, { rootMargin: "400px" })
  : null;

function drawStemRowWhenVisible(canvas, draw) {
  if (!stemDrawObserver) {
    draw();
    return;
  }
  stemDrawWaiting.set(canvas, draw);
  stemDrawObserver.observe(canvas);
}

// What a track looks like depends on exactly these things.
function stemRowSignature(rowName, lanes, stem, width) {
  const peaks = stem ? stem.peaks : lanes.mix?.peaks;
  const regions = stem ? stem.regions.map((region) => `${region.start}-${region.end}`).join(",") : "";
  return [width, rowName, stem ? `${stem.mask ? 1 : 0}${stem.mute ? 1 : 0}|${stem.db}|${regions}` : "", lanes.duration, Boolean(lanes.enabled)].join("|")
    + `|${Array.isArray(peaks) ? peaks.length : 0}`;
}

// Draws one stem track for one scene. ``regions`` is what to show as kept (the stored ones, or the ones being dragged).
function drawStemRow(canvas, rowName, lanes, stem, regions) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(17,17,19,.96)";
  ctx.fillRect(0, 0, width, height);
  const color = STEM_COLORS[rowName] || STEM_COLORS.mix;
  const peaks = Array.isArray(stem ? stem.peaks : lanes.mix?.peaks) ? (stem ? stem.peaks : lanes.mix.peaks) : [];
  const duration = Math.max(0.05, Number(lanes.duration) || 1);
  const masked = Boolean(stem?.mask);
  const muted = Boolean(stem?.mute);
  const mid = height / 2;
  // Which columns are kept, worked out per region instead of testing every region at every column.
  const keptColumns = masked ? new Uint8Array(width) : null;
  if (keptColumns) {
    for (const region of regions) {
      keptColumns.fill(1, Math.max(0, Math.floor((region.start / duration) * width)), Math.min(width, Math.ceil((region.end / duration) * width)));
    }
  }
  // One path for the kept columns and one for the dimmed ones: two fills instead of one per column.
  const keptPath = new Path2D();
  const dimPath = new Path2D();
  for (let x = 0; x < width; x += 1) {
    const level = Math.min(1, peaks[Math.floor((x / width) * peaks.length)] || 0);
    const half = Math.max(0.5, level * (height / 2 - 2));
    (!keptColumns || keptColumns[x] ? keptPath : dimPath).rect(x, mid - half, 1, half * 2);
  }
  ctx.fillStyle = color;
  ctx.globalAlpha = muted ? 0.12 : 1;
  ctx.fill(keptPath);
  ctx.globalAlpha = muted ? 0.12 : 0.2;
  ctx.fill(dimPath);
  ctx.globalAlpha = 1;
  if (masked && !muted) {
    for (const region of regions) {
      const x = (region.start / duration) * width;
      const regionWidth = Math.max(1, ((region.end - region.start) / duration) * width);
      ctx.fillStyle = "rgba(34,197,94,.16)";
      ctx.fillRect(x, 0, regionWidth, height);
      ctx.fillStyle = "#22c55e";
      ctx.fillRect(x, 0, 2, height);
      ctx.fillRect(x + regionWidth - 2, 0, 2, height);
    }
  }
  if (width > 70) {
    const label = stem
      ? `${STEM_LABELS[rowName] || rowName}${muted ? " · muted, double-click to turn on" : masked && !regions.length ? " · silent, double-click to turn on" : masked ? " · only regions" : ""}${stem.db ? ` · ${stem.db > 0 ? "+" : ""}${stem.db} dB` : ""}`
      : peaks.length ? "Masked mix" : "Masked mix (press Build)";
    ctx.font = "bold 12px sans-serif";
    ctx.fillStyle = "rgba(9,9,11,.7)";
    ctx.fillRect(2, 2, Math.min(width - 4, ctx.measureText(label).width + 8), 17);
    ctx.fillStyle = "rgba(244,244,245,.95)";
    ctx.fillText(label, 6, 14);
  }
}

// Drag on a stem track to keep that part of the stem, drag a region's edges to resize it or its middle to move it, and
// double-click a region to delete it. The change is sent to Audio Mask, which saves it and rebuilds the mix.
function attachStemEditing(canvas, sceneId, rowName, lanes, stem) {
  const duration = Math.max(0.05, Number(lanes.duration) || 1);
  let working = stem.mask ? stem.regions.map((region) => ({ ...region })) : [];
  let drag = null;
  const timeAt = (event) => {
    const rect = canvas.getBoundingClientRect();
    return Math.max(0, Math.min(duration, ((event.clientX - rect.left) / Math.max(1, rect.width)) * duration));
  };
  const hit = (event) => {
    if (!stem.mask) return { kind: "new", index: -1 };
    const rect = canvas.getBoundingClientRect();
    const grab = (STEM_EDGE_GRAB_PX / Math.max(1, rect.width)) * duration;
    const time = timeAt(event);
    for (let index = 0; index < working.length; index += 1) {
      if (Math.abs(time - working[index].start) <= grab) return { kind: "start", index };
      if (Math.abs(time - working[index].end) <= grab) return { kind: "end", index };
    }
    const inside = working.findIndex((region) => time > region.start && time < region.end);
    return inside >= 0 ? { kind: "move", index: inside } : { kind: "new", index: -1 };
  };
  const round = (value) => Math.round(value * 1000) / 1000;
  const send = (regions) => requestStemEdit(sceneId, rowName, { mask: true, regions });

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    const target = hit(event);
    const time = timeAt(event);
    if (target.kind === "new") {
      working.push({ start: round(time), end: round(time), fade_ms: 30 });
      drag = { kind: "end", index: working.length - 1, anchor: time, created: true };
    } else {
      const region = working[target.index];
      drag = { kind: target.kind, index: target.index, offset: time - region.start, length: region.end - region.start };
    }
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) {
      const target = hit(event);
      canvas.style.cursor = target.kind === "start" || target.kind === "end" ? "ew-resize" : target.kind === "move" ? "grab" : "crosshair";
      return;
    }
    const time = timeAt(event);
    const region = working[drag.index];
    if (drag.created) {
      region.start = round(Math.min(drag.anchor, time));
      region.end = round(Math.max(drag.anchor, time));
    } else if (drag.kind === "start") {
      region.start = round(Math.min(time, region.end - STEM_MIN_REGION_SECONDS));
    } else if (drag.kind === "end") {
      region.end = round(Math.max(time, region.start + STEM_MIN_REGION_SECONDS));
    } else {
      const start = Math.max(0, Math.min(duration - drag.length, time - drag.offset));
      region.start = round(start);
      region.end = round(start + drag.length);
    }
    drawStemRow(canvas, rowName, lanes, stem, working);
  });
  const finish = () => {
    if (!drag) return;
    const kept = working.filter((region) => region.end - region.start >= STEM_MIN_REGION_SECONDS);
    // A plain click changes nothing and must send nothing: the timeline redraws on every edit, which would replace
    // this canvas in the middle of a double-click.
    const before = stem.mask ? stem.regions : [];
    const changed = kept.length !== before.length
      || kept.some((region, index) => Math.abs(region.start - before[index].start) > 0.0005 || Math.abs(region.end - before[index].end) > 0.0005);
    drag = null;
    if (changed) send(kept);
    else {
      working = stem.mask ? stem.regions.map((region) => ({ ...region })) : [];
      drawStemRow(canvas, rowName, lanes, stem, working);
    }
  };
  canvas.addEventListener("pointerup", finish);
  canvas.addEventListener("pointercancel", finish);
  canvas.addEventListener("dblclick", (event) => {
    event.stopPropagation();
    // A double-click is an on/off switch for the whole stem, except inside a region, where it deletes that region.
    if (stem.mute) {
      requestStemEdit(sceneId, rowName, { mute: false });
      return;
    }
    if (stem.mask && !stem.regions.length) {
      // Masked with no region left, so silent: play the whole stem again (the saved masking is switched off).
      requestStemEdit(sceneId, rowName, { mask: false });
      return;
    }
    const time = timeAt(event);
    const remaining = stem.mask ? stem.regions.filter((region) => !(time >= region.start && time <= region.end)) : stem.regions;
    if (remaining.length !== stem.regions.length) send(remaining);
    else requestStemEdit(sceneId, rowName, { mute: true });
  });
  return () => drawStemRow(canvas, rowName, lanes, stem, working);
}

function drawSegmentAudioWaveform(canvas, peaks) {
  const values = Array.isArray(peaks) && peaks.length ? peaks : [];
  const width = Math.max(1, canvas.width || 1);
  const height = Math.max(1, canvas.height || 1);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  if (!values.length) return;
  const mid = height / 2;
  ctx.strokeStyle = "rgba(216, 180, 254, .9)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < width; x++) {
    const index = Math.floor((x / width) * values.length);
    const amp = Math.min(1, Math.max(0.03, Number(values[index] || 0) * 1.6));
    ctx.moveTo(x, mid - amp * (height / 2));
    ctx.lineTo(x, mid + amp * (height / 2));
  }
  ctx.stroke();
}

function timelineMarkerColor(type) {
  const value = String(type || "").toLowerCase();
  if (value.includes("female")) return { bg: "rgba(190,24,93,.88)", border: "#f9a8d4", text: "#fce7f3" };
  if (value.includes("male")) return { bg: "rgba(37,99,235,.88)", border: "#93c5fd", text: "#dbeafe" };
  if (value.includes("chorus")) return { bg: "rgba(126,34,206,.88)", border: "#d8b4fe", text: "#f3e8ff" };
  if (value.includes("verse")) return { bg: "rgba(21,128,61,.88)", border: "#86efac", text: "#dcfce7" };
  if (value.includes("beat")) return { bg: "rgba(217,119,6,.88)", border: "#fcd34d", text: "#fffbeb" };
  return { bg: "rgba(8,47,73,.9)", border: "#67e8f9", text: "#ecfeff" };
}

export function buildTimelineView({ overlay, preview, previewStage, getDeleteAvailability }) {
  const timeline = document.createElement("div");
  timeline.style.cssText = "display:grid;grid-template-rows:7px auto 1fr;border-top:1px solid #27272a;background:#111113;min-height:0;";
  const timelineResizeHandle = document.createElement("div");
  timelineResizeHandle.title = "Drag to resize timeline";
  timelineResizeHandle.style.cssText = "cursor:row-resize;background:#18181b;border-bottom:1px solid #27272a;";
  const timelineHeader = document.createElement("div");
  timelineHeader.style.cssText = "display:flex;gap:8px;align-items:center;padding:8px 12px;border-bottom:1px solid #27272a;font-size:12px;overflow-x:auto;overflow-y:hidden;white-space:nowrap;";
  const bulkSegmentsButton = makeButton("Bulk Segments");
  const sceneNoteButton = makeButton("+ Scene Note");
  const videoNoteButton = makeButton("+ Video Note");
  const lyricNoteButton = makeButton("+ Line Note");
  const setInButton = makeButton("Set In");
  const setOutButton = makeButton("Set Out");
  const clearRangeButton = makeButton("Clear Range");
  const closeTimelineGapsButton = makeButton("Close Gaps");
  const snapSceneEdgeButton = makeButton("Snap Scene Edge");
  const splitSceneButton = makeButton("✂");
  const idLoraTrimModeButton = makeButton("Trim Mode");
  const overlayTrackToggleButton = makeButton("Overlay Track: Off");
  const overlayTrackHintButton = makeButton("?");
  const addTimelineMarkerButton = makeButton("+ Timeline Note");
  const addSegmentButton = makeButton("Add Segment", "primary");
  const addOverlaySegmentButton = makeButton("+ Overlay Track", "primary");
  const undoButton = makeButton("Undo");
  const redoButton = makeButton("Redo");
  const playButton = makeButton("Play");
  const stopButton = makeButton("Stop");
  const multiSelectButton = makeButton("Select Multi");
  const multiSelectHintButton = makeButton("?");
  const deleteSegmentButton = makeButton("Del");
  const deleteAllSegmentsButton = makeButton("Delete ALL segments");
  const zoomOutButton = makeButton("-");
  const zoomInButton = makeButton("+");
  bulkSegmentsButton.title = "Create many manual timeline scenes from pasted durations or start/end times.";
  sceneNoteButton.title = "Show editable note boxes under each base scene in the timeline.";
  videoNoteButton.title = "Show editable video motion/action notes under each base scene in the timeline.";
  lyricNoteButton.title = "Show editable line/lyric/dialogue boxes under each base scene in the timeline.";
  setInButton.title = "Set the selected timeline range start at the playhead.";
  setOutButton.title = "Set the selected timeline range end at the playhead.";
  clearRangeButton.title = "Clear the selected timeline range.";
  closeTimelineGapsButton.title = "Shift later base scenes left to remove empty gaps in the timeline.";
  snapSceneEdgeButton.title = "Move the selected scene start or end to its closest beat marker. A connected neighboring scene follows only at that shared boundary. Shortcuts: Ctrl+S snaps the start; Ctrl+E snaps the end.";
  splitSceneButton.title = "Click: split an unrendered base scene, or trim a rendered ID-LoRA / MiniMax built-in-audio scene before or after the playhead. Right-click also opens the rendered-scene trim chooser.";
  idLoraTrimModeButton.title = "Quiet scrub mode for finding left/right video trim points without autoplay. Available for ID-LoRA and MiniMax built-in-audio scenes.";
  overlayTrackToggleButton.title = "Turn the advanced overlay timeline on or off.";
  overlayTrackHintButton.title = "How does the overlay timeline work?";
  addTimelineMarkerButton.title = "Add a free timeline note/song marker at the playhead or selected In/Out range.";
  addSegmentButton.title = "Add a segment (S). If the playhead is at least 0.5 seconds past the final base scene, the new segment fills the gap to the playhead and its end snaps to the nearest beat when Snap beats is on.";
  addOverlaySegmentButton.title = "Add a new clip to the overlay track at the playhead without changing the base timeline.";
  undoButton.title = "Undo";
  redoButton.title = "Redo";
  playButton.title = "Play / Pause (Space)";
  stopButton.title = "Stop";
  multiSelectButton.title = "Select multiple scenes, or hold Ctrl/Cmd while clicking scenes in the timeline or scene list.";
  multiSelectHintButton.title = "What does Select Multi do? Ctrl/Cmd-click also toggles scene selection.";
  deleteSegmentButton.title = "Delete selected segment";
  deleteAllSegmentsButton.title = "Delete every base and insert/overlay segment from the timeline.";
  zoomOutButton.title = "Zoom out timeline";
  zoomInButton.title = "Zoom in timeline";
  bulkSegmentsButton.textContent = "Bulk Segments";
  addSegmentButton.textContent = "+ Segment";
  addOverlaySegmentButton.textContent = "+ Overlay Track";
  undoButton.textContent = "↶";
  redoButton.textContent = "↷";
  playButton.textContent = "▶";
  stopButton.textContent = "■";
  multiSelectButton.textContent = "Select Multi";
  multiSelectHintButton.textContent = "?";
  deleteSegmentButton.textContent = "×";
  deleteSegmentButton.style.borderColor = "#7f1d1d";
  deleteSegmentButton.style.color = "#fecaca";
  deleteAllSegmentsButton.style.borderColor = "#7f1d1d";
  deleteAllSegmentsButton.style.color = "#fecaca";
  for (const button of [bulkSegmentsButton, sceneNoteButton, videoNoteButton, lyricNoteButton, setInButton, setOutButton, clearRangeButton, closeTimelineGapsButton, snapSceneEdgeButton, splitSceneButton, idLoraTrimModeButton, overlayTrackToggleButton, overlayTrackHintButton, addTimelineMarkerButton, addSegmentButton, addOverlaySegmentButton, undoButton, redoButton, playButton, stopButton, multiSelectButton, multiSelectHintButton, deleteSegmentButton, deleteAllSegmentsButton, zoomOutButton, zoomInButton]) {
    button.style.padding = "7px 10px";
    button.style.minWidth = "0";
    button.style.flex = "0 0 auto";
  }
  bulkSegmentsButton.style.width = "max-content";
  sceneNoteButton.style.width = "max-content";
  videoNoteButton.style.width = "max-content";
  lyricNoteButton.style.width = "max-content";
  setInButton.style.width = "max-content";
  setOutButton.style.width = "max-content";
  clearRangeButton.style.width = "max-content";
  closeTimelineGapsButton.style.width = "max-content";
  snapSceneEdgeButton.style.width = "max-content";
  idLoraTrimModeButton.style.width = "max-content";
  addTimelineMarkerButton.style.width = "max-content";
  addSegmentButton.style.width = "max-content";
  addOverlaySegmentButton.style.width = "max-content";
  overlayTrackToggleButton.style.width = "max-content";
  overlayTrackHintButton.style.width = "34px";
  deleteAllSegmentsButton.style.width = "max-content";
  for (const button of [splitSceneButton, undoButton, redoButton, playButton, stopButton, deleteSegmentButton, zoomOutButton, zoomInButton]) {
    button.style.width = "34px";
  }
  const waveformModeSelect = makeSelect(Object.keys(WAVEFORM_MODES), "medium");
  waveformModeSelect.style.width = "max-content";
  waveformModeSelect.style.flex = "0 0 auto";
  for (const option of waveformModeSelect.options) {
    option.textContent = WAVEFORM_MODES[option.value]?.label || option.value;
  }
  const snapToBeatsControl = makeCheckbox("Snap beats", true);
  snapToBeatsControl.wrapper.style.margin = "0";
  snapToBeatsControl.wrapper.style.flex = "0 0 auto";
  const beatMarkersButton = makeButton("^");
  beatMarkersButton.title = "Show or hide beat markers";
  beatMarkersButton.style.width = "34px";
  beatMarkersButton.style.padding = "7px 10px";
  const locationThumbnailButton = makeButton("Locations: Off");
  locationThumbnailButton.title = "Show each scene's mapped location image or text title on the timeline.";
  locationThumbnailButton.style.padding = "7px 10px";
  const globalScrub = document.createElement("input");
  globalScrub.type = "range";
  globalScrub.min = "0";
  globalScrub.max = "0";
  globalScrub.step = "0.01";
  globalScrub.value = "0";
  globalScrub.style.cssText = "width:100%;accent-color:#22d3ee;";
  const globalScrubWrap = document.createElement("label");
  globalScrubWrap.style.cssText = "display:grid;grid-template-columns:auto auto minmax(160px,1fr) auto;align-items:center;gap:8px;color:#d4d4d8;font-size:12px;font-weight:800;background:#18181b;border-top:1px solid #27272a;padding:8px 10px;";
  const globalScrubLabel = document.createElement("span");
  globalScrubLabel.textContent = "Global audio scrub";
  globalScrubLabel.style.cssText = "white-space:nowrap;";
  const globalAudioMuteButton = makeButton("🔊");
  globalAudioMuteButton.type = "button";
  globalAudioMuteButton.title = "Mute global timeline audio. Useful when previewing a completed video that already has audio.";
  globalAudioMuteButton.style.cssText = "width:30px;height:28px;min-width:30px;padding:0;display:inline-flex;align-items:center;justify-content:center;font-size:14px;";
  const globalScrubTime = document.createElement("span");
  globalScrubTime.textContent = "00:00.00";
  globalScrubTime.style.cssText = "color:#67e8f9;font-variant-numeric:tabular-nums;white-space:nowrap;";
  globalScrubWrap.append(globalScrubLabel, globalAudioMuteButton, globalScrub, globalScrubTime);
  preview.append(previewStage, globalScrubWrap);
  const timelineInfo = document.createElement("div");
  timelineInfo.textContent = "No audio loaded";
  timelineInfo.style.cssText = "color:#67e8f9;font-variant-numeric:tabular-nums;white-space:nowrap;flex:0 0 auto;";
  const timelineRangeInfo = document.createElement("div");
  timelineRangeInfo.textContent = "Range: none";
  timelineRangeInfo.style.cssText = "color:#a5f3fc;font-variant-numeric:tabular-nums;white-space:nowrap;flex:0 0 auto;";
  const timelineStatusInfo = document.createElement("div");
  timelineStatusInfo.style.cssText = "display:flex;align-items:center;gap:12px;flex:0 0 auto;min-width:max-content;padding:0 4px;";
  timelineStatusInfo.append(timelineInfo, timelineRangeInfo);
  const deleteAllTimelineVideosButton = makeButton("Delete ALL Videos");
  deleteAllTimelineVideosButton.style.padding = "6px 10px";
  deleteAllTimelineVideosButton.style.borderColor = "#dc2626";
  deleteAllTimelineVideosButton.style.background = "#450a0a";
  deleteAllTimelineVideosButton.style.color = "#fee2e2";
  deleteAllTimelineVideosButton.title = "Permanently delete every generated video, video history entry, and thumbnail for every scene from the project folder, and clear the video latents. Cannot be undone.";
  const deleteAllTimelineImagesButton = makeButton("Delete ALL Images");
  deleteAllTimelineImagesButton.style.padding = "6px 10px";
  deleteAllTimelineImagesButton.style.borderColor = "#dc2626";
  deleteAllTimelineImagesButton.style.background = "#450a0a";
  deleteAllTimelineImagesButton.style.color = "#fee2e2";
  deleteAllTimelineImagesButton.title = "Delete every timeline image from every scene, including FLF first frames, last frames, and extracted chained start frames, and remove those files from the project folder.";
  const { toolsButton, deleteAllButton, refreshDeleteActions } = createTimelineToolWindows({
    overlay,
    toolButtons: [setInButton, setOutButton, clearRangeButton, closeTimelineGapsButton, snapSceneEdgeButton, overlayTrackToggleButton, overlayTrackHintButton, locationThumbnailButton],
    deleteButtons: [deleteAllTimelineImagesButton, deleteAllTimelineVideosButton, deleteAllSegmentsButton],
    getDeleteAvailability,
  });
  const audioMaskButton = makeButton("Audio Mask");
  const stemVisibilityButton = makeButton("Hide Stems");
  stemVisibilityButton.title = "Hide or show the stem tracks under the scenes. Hiding only changes what the timeline shows. The stems and masks stay as they are.";
  const stemMonitorButton = makeButton("Hear Stems");
  stemMonitorButton.title = "Play the masked stem mix of a scene from the timeline instead of the main audio, so you hear what the stem choices sound like. Scenes without a stem mix keep playing the main audio.";
  const zoomWrap = document.createElement("div");
  zoomWrap.style.cssText = "display:flex;gap:4px;align-items:center;";
  zoomWrap.append(zoomOutButton, zoomInButton);
  const timelineToolRail = document.createElement("div");
  timelineToolRail.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:12px 5px 12px 6px;border-right:1px solid #27272a;background:#09090b;overflow-y:auto;overflow-x:hidden;min-height:0;scrollbar-width:thin;";
  for (const button of [bulkSegmentsButton, sceneNoteButton, videoNoteButton, lyricNoteButton, addTimelineMarkerButton, addSegmentButton, addOverlaySegmentButton, audioMaskButton, stemMonitorButton, stemVisibilityButton]) {
    button.style.width = "96px";
    button.style.minHeight = "40px";
    button.style.padding = "6px 7px";
    button.style.whiteSpace = "normal";
    button.style.lineHeight = "1.18";
    button.style.textAlign = "center";
  }
  bulkSegmentsButton.textContent = "Bulk";
  addSegmentButton.textContent = "+ Segment";
  addOverlaySegmentButton.textContent = "+ Overlay Track";
  timelineToolRail.append(bulkSegmentsButton, sceneNoteButton, videoNoteButton, lyricNoteButton, addTimelineMarkerButton, addSegmentButton, addOverlaySegmentButton, audioMaskButton, stemMonitorButton, stemVisibilityButton);
  timelineHeader.append(toolsButton, splitSceneButton, idLoraTrimModeButton, undoButton, redoButton, playButton, stopButton, multiSelectButton, multiSelectHintButton, waveformModeSelect, snapToBeatsControl.wrapper, beatMarkersButton, zoomWrap, timelineStatusInfo, deleteSegmentButton, deleteAllButton);
  const timelineBody = document.createElement("div");
  timelineBody.style.cssText = "display:grid;grid-template-columns:auto minmax(0,1fr);min-height:0;overflow:hidden;";
  const timelineViewport = document.createElement("div");
  timelineViewport.style.cssText = "position:relative;overflow:auto;min-height:0;padding:12px;";
  const timelineCanvas = document.createElement("canvas");
  timelineCanvas.height = TIMELINE_HEIGHT;
  timelineCanvas.style.cssText = `display:block;height:${TIMELINE_HEIGHT}px;background:#09090b;border:1px solid #27272a;border-radius:6px;`;
  const segmentLayer = document.createElement("div");
  segmentLayer.style.cssText = `position:absolute;left:12px;top:12px;height:${TIMELINE_HEIGHT}px;pointer-events:none;`;
  const playhead = document.createElement("div");
  playhead.style.cssText = `position:absolute;left:12px;top:12px;height:${TIMELINE_HEIGHT}px;width:3px;background:#f4f4f5;box-shadow:0 0 10px rgba(103,232,249,.8);cursor:ew-resize;z-index:5;`;
  // Stem tracks live in their own layer. A timeline redraw (it happens at every scene change) leaves it alone.
  const stemLayer = document.createElement("div");
  stemLayer.style.cssText = `position:absolute;left:12px;top:12px;height:${TIMELINE_HEIGHT}px;pointer-events:none;contain:layout style;`;
  timelineViewport.append(timelineCanvas, segmentLayer, stemLayer, playhead);
  timelineBody.append(timelineToolRail, timelineViewport);
  timeline.append(timelineResizeHandle, timelineHeader, timelineBody);

  return {
    addOverlaySegmentButton, addSegmentButton, addTimelineMarkerButton, audioMaskButton, beatMarkersButton, bulkSegmentsButton, stemMonitorButton, stemVisibilityButton,
    clearRangeButton, closeTimelineGapsButton, deleteAllSegmentsButton, deleteAllTimelineImagesButton,
    deleteAllTimelineVideosButton, deleteSegmentButton, globalAudioMuteButton,
    globalScrub, globalScrubTime, idLoraTrimModeButton, lyricNoteButton, locationThumbnailButton, multiSelectButton,
    multiSelectHintButton, overlayTrackHintButton, overlayTrackToggleButton, playButton, playhead, redoButton,
    refreshDeleteActions, sceneNoteButton, segmentLayer, setInButton, setOutButton, snapSceneEdgeButton,
    snapToBeatsControl, splitSceneButton, stopButton, timeline, timelineCanvas, timelineInfo,
    stemLayer, timelineRangeInfo, timelineResizeHandle, timelineViewport, undoButton,
    videoNoteButton, waveformModeSelect, zoomInButton, zoomOutButton,
  };
}

export function createTimelineView({
  stemLayer, timelineViewport, activeSegment, appendTimelineFirstLastFrameThumbnail, autoSaveSessionQuiet, clampTimelineMarkerToNonOverlap,
  currentGlobalTime, currentProjectAudioPath, currentVideoMode, cycleSegmentImageHistory,
  cycleSegmentVideoHistory, enableImageDrop, enableLutDrop, enablePostEffectDrop,
  ensureAllSegmentRuntimeFields, handleSegmentPick, i2vNotesInput, isSegmentMultiSelected,
  loadedGlobalAudioDuration, lyricTextInput, makeDragHandle, markerVisualEnd, mediaThumbnailHtml,
  openAudioContextMenu, openDirectorNoteContextMenu, openSceneOptions, openSegmentContextMenu,
  openTimelineSceneCard, playhead, pushHistory, render, rtvReferenceBehaviorForSegment, sceneListPane,
  segmentImageSource, segmentLayer, segmentTrack, selectedSegmentImageThumbnailPath, miniMaxH3ModeForSegment,
  selectedTimelineRangeInfo, setActiveSegment, state, syncInspector, syncLyricMapperFromSegments,
  timelineCanvas, timelineDuration, timelineSegmentLabel, toggleSegmentPreviewMode,
  locationThumbnailButton, refreshDeleteActions,
}) {
  let showLocationThumbnails = false;
  locationThumbnailButton.onclick = () => {
    showLocationThumbnails = !showLocationThumbnails;
    locationThumbnailButton.textContent = `Locations: ${showLocationThumbnails ? "On" : "Off"}`;
    render();
  };
  // The scene card's quick button: lock this scene's MiniMax settings and make it a Latent Continuation Masked scene
  // with the Masked transition. Clicking it again puts the scene back the way it was before the first click.
  async function toggleSceneMaskedContinuation(segment) {
    if (!segment) return;
    if (state.segments.indexOf(segment) <= 0) {
      toast("The first scene has no previous scene to continue from.", true);
      return;
    }
    const before = segment.minimax_h3_masked_quick_prev;
    const maskedNow = Boolean(segment.use_scene_minimax_h3_settings)
      && segment.minimax_h3_settings?.continuity_mode === "latent_continuation_masked";
    if (maskedNow) {
      pushHistory();
      if (before && before.locked === false) {
        segment.use_scene_minimax_h3_settings = false;
        delete segment.minimax_h3_settings;
      } else {
        segment.minimax_h3_settings = cloneMiniMaxH3Settings({
          ...segment.minimax_h3_settings,
          continuity_mode: before?.continuity_mode || "off",
          location_transition_preset: before?.location_transition_preset || "normal",
        });
      }
      delete segment.minimax_h3_masked_quick_prev;
      if (segment.id === state.activeId) syncInspector();
      render();
      await autoSaveSessionQuiet("MiniMax H3 scene masked continuation turned off");
      toast(before && before.locked === false
        ? "Masked continuation off. This scene follows the project MiniMax settings again."
        : "Masked continuation off for this scene.");
      return;
    }
    const base = cloneMiniMaxH3Settings(segment.use_scene_minimax_h3_settings && segment.minimax_h3_settings
      ? segment.minimax_h3_settings
      : state.miniMaxH3Settings);
    if (!isMiniMaxH3ContinuityAllowedForMode("latent_continuation_masked", base.video_mode, base.render_pass)) {
      toast("Latent Continuation Masked is not available for this scene's MiniMax mode (Image + Reference and 2 Pass Advanced are excluded).", true);
      return;
    }
    pushHistory();
    segment.minimax_h3_masked_quick_prev = {
      locked: Boolean(segment.use_scene_minimax_h3_settings),
      continuity_mode: base.continuity_mode,
      location_transition_preset: base.location_transition_preset,
    };
    segment.use_scene_minimax_h3_settings = true;
    segment.minimax_h3_settings = cloneMiniMaxH3Settings({
      ...base,
      continuity_mode: "latent_continuation_masked",
      location_transition_preset: "masked",
    });
    segment.minimax_h3_mode = segment.minimax_h3_settings.video_mode;
    if (segment.id === state.activeId) syncInspector();
    render();
    await autoSaveSessionQuiet("MiniMax H3 scene set to masked continuation");
    toast("Scene locked and set to Latent Continuation Masked with the Masked transition.");
  }
  function normalizeSegments(changedSegment, changedIndex = null) {
    sortSegments(state.segments);
    const minDuration = 0.1;
    const active = changedSegment || activeSegment();
    const activeIndex = changedIndex ?? state.segments.findIndex((segment) => segment.id === active?.id);
    if (activeIndex < 0) return;

    active.start = Math.max(0, Number(active.start || 0));
    active.end = Math.max(active.start + minDuration, Number(active.end || active.start + 4));

    const prev = state.segments[activeIndex - 1] || null;
    const next = state.segments[activeIndex + 1] || null;

    if (!prev) {
      active.start = 0;
    } else {
      if (hasLockedVideo(prev)) {
        active.start = Math.max(active.start, Number(prev.end || 0));
      } else {
        const prevStart = Number(prev.start || 0);
        active.start = Math.max(active.start, prevStart + minDuration);
        prev.end = active.start;
      }
    }

    active.end = Math.max(active.start + minDuration, active.end);

    if (next) {
      if (hasLockedVideo(next)) {
        active.end = Math.min(active.end, Number(next.start || active.end));
      } else {
        const nextEnd = Number(next.end || active.end + minDuration);
        active.end = Math.min(active.end, nextEnd - minDuration);
      }
      active.end = Math.max(active.start + minDuration, active.end);
      if (!hasLockedVideo(next)) next.start = active.end;
    }
    state.duration = Math.max(Number(state.duration || 0), Number(active.end || 0));

    if (prev && !hasLockedVideo(prev)) prev.end = active.start;
    if (next) {
      if (!hasLockedVideo(next)) {
        next.start = active.end;
        if (next.end < next.start + minDuration) next.end = next.start + minDuration;
      }
    }
    sortSegments(state.segments);
  }

  // The band for stem lanes exists only while some scene has stems and the Audio Mask window's switch is on.
  function refreshStemBand() {
    const names = state.showTimelineStems !== false ? stemRowNames(state.segments.map((segment) => segment.id)) : [];
    stemRowList = names.length ? [...names, "mix"] : [];
    stemBandHeight = stemRowList.length ? stemRowList.length * (STEM_ROW_HEIGHT + STEM_ROW_GAP) + 6 : 0;
  }

  // Only tracks near the visible part of the timeline exist, and the layer is rebuilt only when what it shows changes
  // (zoom, scroll to a new scene, an edit, stems arriving). Drawn tracks are reused from the cache.
  const STEM_RENDER_MARGIN_PX = 900;
  let stemLayerKey = "";

  function createStemRowCanvas(item) {
    const { segment, rowName, stem, lanes } = item;
    const rowCanvas = document.createElement("canvas");
    rowCanvas.width = item.rowWidth;
    rowCanvas.height = STEM_ROW_HEIGHT;
    rowCanvas.style.cssText = `
      position:absolute;left:${item.left}px;top:${item.top}px;width:${rowCanvas.width}px;height:${STEM_ROW_HEIGHT}px;
      z-index:1;pointer-events:auto;cursor:crosshair;border:1px solid ${lanes.enabled ? "#22d3ee" : "rgba(103,232,249,.25)"};border-radius:4px;
    `;
    if (stem) {
      rowCanvas.title = `${STEM_LABELS[rowName]}: drag to keep a part of this stem, drag a region's edges or middle to change it, double-click a region to delete it. Double-click anywhere else on the track to mute the stem, and double-click a muted or silent stem to turn it back on. Use the Audio Mask button for exact times, levels and mute.`;
      drawStemRowWhenVisible(rowCanvas, attachStemEditing(rowCanvas, segment.id, rowName, lanes, stem));
    } else {
      rowCanvas.style.cursor = "pointer";
      rowCanvas.title = "Masked mix: the stems added together with every mask, level and mute applied. Click to open Audio Mask.";
      rowCanvas.onclick = (event) => {
        event.stopPropagation();
        setActiveSegment(segment);
        requestOpenAudioMask(segment.id);
      };
      drawStemRowWhenVisible(rowCanvas, () => drawStemRow(rowCanvas, "mix", lanes, null, []));
    }
    return rowCanvas;
  }

  function renderStemTracks() {
    const items = [];
    if (stemRowList.length) {
      const from = timelineViewport.scrollLeft - STEM_RENDER_MARGIN_PX;
      const to = timelineViewport.scrollLeft + timelineViewport.clientWidth + STEM_RENDER_MARGIN_PX;
      const bandTop = TIMELINE_SCENE_AUDIO_TOP + TIMELINE_SCENE_AUDIO_HEIGHT + 4;
      for (const segment of state.segments) {
        const left = segment.start * state.pxPerSecond;
        const rowWidth = Math.max(24, Math.floor((segment.end - segment.start) * state.pxPerSecond));
        if (left + rowWidth < from || left > to) continue;
        const lanes = getStemLanes(segment.id);
        // Stems made for a different scene length (merged, split or resized) would be drawn stretched.
        if (!lanes || Math.abs(Number(lanes.duration) - (segment.end - segment.start)) > 0.02) continue;
        stemRowList.forEach((rowName, rowIndex) => {
          const stem = rowName === "mix" ? null : lanes.stems.find((candidate) => candidate.name === rowName);
          if (rowName !== "mix" && !stem) return; // the model this scene was split with has no such stem
          items.push({
            segment, rowName, stem, lanes, rowWidth, left, top: bandTop + rowIndex * (STEM_ROW_HEIGHT + STEM_ROW_GAP),
            key: `${segment.id}|${rowName}`, peaks: stem ? stem.peaks : lanes.mix?.peaks,
            signature: stemRowSignature(rowName, lanes, stem, rowWidth),
          });
        });
      }
    }
    const layoutKey = items.map((item) => `${item.key}@${item.left}:${item.top}:${item.signature}`).join(";");
    if (layoutKey === stemLayerKey) return;
    stemLayerKey = layoutKey;
    const used = new Set();
    const fragment = document.createDocumentFragment();
    for (const item of items) {
      let entry = stemRowCache.get(item.key);
      if (!entry || entry.signature !== item.signature || entry.peaks !== item.peaks) {
        if (entry) stemDrawObserver?.unobserve(entry.canvas);
        entry = { signature: item.signature, peaks: item.peaks, canvas: createStemRowCanvas(item) };
        stemRowCache.set(item.key, entry);
      }
      entry.canvas.style.left = `${item.left}px`;
      entry.canvas.style.top = `${item.top}px`;
      fragment.append(entry.canvas);
      used.add(item.key);
    }
    stemLayer.replaceChildren(fragment);
    for (const [key, entry] of stemRowCache) {
      if (used.has(key)) continue;
      stemDrawObserver?.unobserve(entry.canvas);
      stemRowCache.delete(key);
    }
  }

  // Scrolling to scenes whose tracks do not exist yet creates them, a frame at a time.
  let stemScrollFrame = 0;
  timelineViewport.addEventListener("scroll", () => {
    if (stemScrollFrame || !stemRowList.length) return;
    stemScrollFrame = window.requestAnimationFrame(() => {
      stemScrollFrame = 0;
      renderStemTracks();
    });
  }, { passive: true });

  function timelineHeight() {
    refreshStemBand();
    const baseHeight = WAVEFORM_MODES[state.waveformMode]?.height || WAVEFORM_MODES.medium.height;
    const waveExtra = Math.max(48, baseHeight - 140);
    return timelineWaveTop() + waveExtra + 10;
  }

  function timelineMarkerTop() {
    return timelineVideoNoteTop() + (state.showTimelineVideoNotes ? TIMELINE_NOTE_HEIGHT + TIMELINE_NOTE_GAP : 0);
  }

  function timelineVideoNoteTop() {
    return timelineNoteTop() + (state.showTimelineSceneNotes ? TIMELINE_NOTE_HEIGHT + TIMELINE_NOTE_GAP : 0);
  }

  function timelineMarkerLaneVisible() {
    return normalizeTimelineMarkers(state.timelineMarkers).length > 0;
  }

  function timelineLyricNoteTop() {
    return timelineMarkerTop() + (timelineMarkerLaneVisible() ? TIMELINE_MARKER_HEIGHT + TIMELINE_NOTE_GAP : 0);
  }

  function timelineWaveTop() {
    if (state.showTimelineLyricNotes) return timelineLyricNoteTop() + TIMELINE_NOTE_HEIGHT + 14;
    if (timelineMarkerLaneVisible()) return timelineMarkerTop() + TIMELINE_MARKER_HEIGHT + 14;
    if (state.showTimelineVideoNotes) return timelineVideoNoteTop() + TIMELINE_NOTE_HEIGHT + 14;
    if (state.showTimelineSceneNotes) return timelineNoteTop() + TIMELINE_NOTE_HEIGHT + 14;
    return TIMELINE_SCENE_AUDIO_TOP + TIMELINE_SCENE_AUDIO_HEIGHT + stemBandHeight + 14;
  }

  function snapTimeToBeat(time) {
    if (!state.snapToBeats || !state.showBeatMarkers || !Array.isArray(state.beats) || !state.beats.length) return time;
    const value = Number(time || 0);
    let best = value;
    let bestDelta = Infinity;
    for (const beat of state.beats) {
      const beatTime = Number(beat || 0);
      const delta = Math.abs(beatTime - value);
      if (delta < bestDelta) {
        bestDelta = delta;
        best = beatTime;
      }
    }
    return bestDelta <= 0.14 ? best : value;
  }

  function snapAddedSegmentEndToNearestBeat(time, segmentStart) {
    const value = Math.max(0, Number(time || 0));
    if (!state.snapToBeats || !Array.isArray(state.beats) || !state.beats.length) return value;
    const minimumEnd = Number(segmentStart || 0) + 0.05;
    const audioEnd = loadedGlobalAudioDuration();
    const beatTimes = state.beats
      .map((beat) => Number(beat?.time ?? beat))
      .filter((beat) => Number.isFinite(beat) && beat >= minimumEnd && (!(audioEnd > 0) || beat <= audioEnd + 0.0001));
    if (!beatTimes.length) return value;
    return beatTimes.reduce((closest, beat) => (
      Math.abs(beat - value) < Math.abs(closest - value) ? beat : closest
    ), beatTimes[0]);
  }

  function renderBeatMarkersOverlay() {
    const visibleBeats = state.showBeatMarkers && Array.isArray(state.beats) ? state.beats : [];
    if (!visibleBeats.length) return;
    const top = Math.max(2, TIMELINE_SEGMENT_TOP - 18);
    for (const beatTime of visibleBeats) {
      const x = Number(beatTime || 0) * state.pxPerSecond;
      if (!Number.isFinite(x)) continue;
      const marker = document.createElement("div");
      marker.title = `Beat ${formatTime(beatTime)}`;
      marker.style.cssText = `
        position:absolute;left:${x}px;top:${top}px;width:2px;height:12px;
        z-index:4;pointer-events:none;background:rgba(244,244,245,.9);
        border-radius:2px;box-shadow:0 0 4px rgba(244,244,245,.35);
      `;
      segmentLayer.append(marker);
    }
  }

  function renderSelectedTimelineRangeOverlay() {
    const info = selectedTimelineRangeInfo();
    if (!info) return;
    const left = info.start * state.pxPerSecond;
    const width = Math.max(3, info.duration * state.pxPerSecond);
    const band = document.createElement("div");
    band.title = `Selected range ${formatTime(info.start)} - ${formatTime(info.end)} (${info.duration.toFixed(2)}s)`;
    band.style.cssText = `
      position:absolute;left:${left}px;top:0;width:${width}px;height:${timelineCanvas.height}px;
      z-index:0;pointer-events:none;background:rgba(34,211,238,.14);
      border-left:2px solid rgba(103,232,249,.9);border-right:2px solid rgba(103,232,249,.9);
      box-sizing:border-box;
    `;
    const inTag = document.createElement("div");
    inTag.textContent = "IN";
    inTag.style.cssText = "position:absolute;left:2px;top:2px;background:#0891b2;color:#ecfeff;font-size:10px;font-weight:900;padding:1px 4px;border-radius:3px;";
    const outTag = document.createElement("div");
    outTag.textContent = "OUT";
    outTag.style.cssText = "position:absolute;right:2px;top:2px;background:#0891b2;color:#ecfeff;font-size:10px;font-weight:900;padding:1px 4px;border-radius:3px;";
    band.append(inTag, outTag);
    segmentLayer.append(band);
  }

  function renderTimelineMarkersOverlay() {
    state.timelineMarkers = normalizeTimelineMarkers(state.timelineMarkers);
    const markers = state.timelineMarkers;
    if (!markers.length) return;
    const timelineTotal = timelineDuration();
    const markerTop = timelineMarkerTop();
    const laneLabel = document.createElement("div");
    laneLabel.textContent = "TIMELINE NOTES";
    laneLabel.style.cssText = `position:absolute;left:4px;top:${markerTop - 12}px;color:#a5f3fc;font-size:10px;font-weight:900;letter-spacing:.08em;pointer-events:none;text-shadow:0 1px 2px #020617;`;
    segmentLayer.append(laneLabel);
    for (const marker of markers) {
      const start = Number(marker.start || 0);
      const end = marker.end == null ? Math.min(start + 4, Math.max(start + 4, timelineTotal || start + 4)) : Number(marker.end);
      const left = start * state.pxPerSecond;
      const width = Math.max(TIMELINE_MARKER_MIN_WIDTH, (end - start) * state.pxPerSecond);
      const colors = timelineMarkerColor(marker.type);
      const item = document.createElement("div");
      item.title = `${marker.label || "Timeline note"}\n${formatTime(start)}${marker.end == null ? "" : ` - ${formatTime(end)}`}\n${marker.note || ""}`;
      const active = state.activeTimelineMarkerId === marker.id;
      item.style.cssText = `
        position:absolute;left:${left}px;top:${markerTop}px;width:${width}px;height:${TIMELINE_MARKER_HEIGHT}px;
        z-index:6;pointer-events:auto;cursor:grab;overflow:hidden;box-sizing:border-box;
        border:${active ? "2px" : "1px"} solid ${active ? "#f4f4f5" : colors.border};border-radius:4px;
        background:${colors.bg};color:${colors.text};font-size:11px;font-weight:800;
        box-shadow:${active ? "0 0 0 2px rgba(244,244,245,.18),0 0 10px rgba(103,232,249,.55)" : "none"};
      `;
      const header = document.createElement("div");
      header.textContent = `${marker.type || "note"} | ${formatTime(start)}${marker.end == null ? "" : ` - ${formatTime(end)}`}`;
      header.style.cssText = "height:18px;display:flex;align-items:center;padding:0 10px 0 12px;box-sizing:border-box;background:rgba(2,6,23,.45);font-size:10px;font-weight:900;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
      header.title = "Drag to move this timeline note. Double-click the note for exact timing and type.";
      const noteBox = document.createElement("textarea");
      noteBox.value = marker.note || marker.label || "";
      noteBox.placeholder = "Timeline note...";
      noteBox.style.cssText = `
        width:100%;height:${TIMELINE_MARKER_HEIGHT - 18}px;box-sizing:border-box;resize:none;display:block;
        border:0;border-top:1px solid rgba(255,255,255,.18);border-radius:0;background:rgba(2,6,23,.38);
        color:${colors.text};padding:5px 8px 5px 12px;font-size:11px;line-height:1.25;outline:none;
      `;
      noteBox.onpointerdown = (event) => event.stopPropagation();
      noteBox.onclick = (event) => event.stopPropagation();
      noteBox.onkeydown = (event) => event.stopPropagation();
      noteBox.onfocus = () => {
        noteBox.dataset.previousValue = noteBox.value;
        noteBox.dataset.historyPushed = "0";
        state.activeTimelineMarkerId = marker.id;
      };
      noteBox.oninput = () => {
        if (noteBox.dataset.historyPushed !== "1" && noteBox.dataset.previousValue !== noteBox.value) {
          pushHistory();
          noteBox.dataset.historyPushed = "1";
        }
        marker.note = noteBox.value || "";
        if (!String(marker.label || "").trim() || marker.label === "Timeline note" || marker.label === "Selected range") {
          marker.label = String(noteBox.value || "").trim().split(/\r?\n/)[0].slice(0, 44) || "Timeline note";
        }
      };
      noteBox.onchange = () => {
        marker.note = noteBox.value || "";
        autoSaveSessionQuiet("timeline note edited").catch(() => null);
      };
      const leftHandle = document.createElement("span");
      leftHandle.title = "Drag to adjust note start";
      leftHandle.style.cssText = "position:absolute;left:0;top:0;bottom:0;width:8px;background:rgba(255,255,255,.24);cursor:ew-resize;z-index:3;";
      const rightHandle = document.createElement("span");
      rightHandle.title = "Drag to adjust note end";
      rightHandle.style.cssText = "position:absolute;right:0;top:0;bottom:0;width:8px;background:rgba(255,255,255,.24);cursor:ew-resize;z-index:3;";
      item.append(header, noteBox, leftHandle, rightHandle);
      item.onclick = (event) => {
        event.stopPropagation();
        state.activeTimelineMarkerId = marker.id;
        render();
      };
      item.ondblclick = (event) => {
        event.stopPropagation();
        state.activeTimelineMarkerId = marker.id;
        openTimelineMarkerEditor(marker);
      };
      item.oncontextmenu = (event) => {
        event.preventDefault();
        event.stopPropagation();
        state.activeTimelineMarkerId = marker.id;
        openTimelineMarkerEditor(marker);
      };
      makeTimelineMarkerDragHandle(item, marker, "move", item, header);
      makeTimelineMarkerDragHandle(header, marker, "move", item, header);
      makeTimelineMarkerDragHandle(leftHandle, marker, "start", item, header);
      makeTimelineMarkerDragHandle(rightHandle, marker, "end", item, header);
      segmentLayer.append(item);
    }
  }

  function makeTimelineMarkerDragHandle(handle, marker, mode, item = null, header = null) {
    handle.onpointerdown = (event) => {
      if (event.button !== 0) return;
      if (event.target?.tagName === "TEXTAREA") return;
      event.preventDefault();
      event.stopPropagation();
      handle.setPointerCapture?.(event.pointerId);
      const markerId = marker.id;
      state.activeTimelineMarkerId = markerId;
      const startX = event.clientX;
      const startStart = Number(marker.start || 0);
      const startEnd = marker.end == null ? null : Number(marker.end);
      const minDuration = 0.15;
      const sorted = normalizeTimelineMarkers(state.timelineMarkers);
      const markerIndex = sorted.findIndex((item) => item.id === markerId);
      const previousMarker = markerIndex > 0 ? sorted[markerIndex - 1] : null;
      const nextMarker = markerIndex >= 0 && markerIndex < sorted.length - 1 ? sorted[markerIndex + 1] : null;
      const visualMinDuration = TIMELINE_MARKER_MIN_WIDTH / Math.max(1, Number(state.pxPerSecond || 1));
      const dragMinDuration = Math.max(minDuration, visualMinDuration);
      const minStart = previousMarker ? Math.max(0, markerVisualEnd(previousMarker) + 0.05) : 0;
      const maxEnd = nextMarker ? Math.max(minStart + minDuration, Number(nextMarker.start || 0) - 0.05) : Infinity;
      const startDuration = startEnd == null ? Math.max(4, dragMinDuration) : Math.max(dragMinDuration, startEnd - startStart);
      const visualDuration = Math.max(startDuration, visualMinDuration);
      const paintMarker = (liveMarker) => {
        if (!item) return;
        const liveStart = Number(liveMarker.start || 0);
        const liveEnd = markerEnd(liveMarker);
        item.style.left = `${liveStart * state.pxPerSecond}px`;
        item.style.width = `${Math.max(TIMELINE_MARKER_MIN_WIDTH, (liveEnd - liveStart) * state.pxPerSecond)}px`;
        if (header) header.textContent = `${liveMarker.type || "note"} | ${formatTime(liveStart)} - ${formatTime(liveEnd)}`;
      };
      let historySaved = false;
      const move = (moveEvent) => {
        const liveMarker = state.timelineMarkers.find((item) => item.id === markerId) || marker;
        if (!historySaved) {
          pushHistory();
          historySaved = true;
        }
        const delta = (moveEvent.clientX - startX) / state.pxPerSecond;
        if (mode === "start") {
          const desiredEnd = startEnd ?? Math.max(startStart + dragMinDuration, Number(liveMarker.end || startStart + 4));
          liveMarker.start = Math.max(minStart, Math.min(desiredEnd - dragMinDuration, startStart + delta));
          liveMarker.start = Number(liveMarker.start.toFixed(3));
        } else if (mode === "end") {
          liveMarker.end = Math.max(startStart + dragMinDuration, (startEnd ?? startStart + dragMinDuration) + delta);
          if (Number.isFinite(maxEnd)) liveMarker.end = Math.min(liveMarker.end, maxEnd);
          liveMarker.end = Number(liveMarker.end.toFixed(3));
        } else {
          let nextStart = Math.max(minStart, startStart + delta);
          if (Number.isFinite(maxEnd)) nextStart = Math.min(nextStart, maxEnd - visualDuration);
          nextStart = Math.max(minStart, nextStart);
          liveMarker.start = Number(nextStart.toFixed(3));
          liveMarker.end = Number((liveMarker.start + startDuration).toFixed(3));
        }
        paintMarker(liveMarker);
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        state.timelineMarkers = normalizeTimelineMarkers(state.timelineMarkers);
        render();
        autoSaveSessionQuiet("timeline marker moved").catch(() => null);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    };
  }

  function openTimelineMarkerEditor(marker) {
    const target = marker || newTimelineMarker(currentGlobalTime(), null);
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100008;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:16px;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(520px,calc(100vw - 32px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:14px;display:grid;gap:10px;";
    const title = document.createElement("div");
    title.textContent = target.id ? "Timeline Note" : "New Timeline Note";
    title.style.cssText = "font-size:15px;font-weight:900;color:#cffafe;";
    const typeInput = makeInput("note");
    typeInput.value = target.type || "note";
    const labelInput = makeInput("Timeline note");
    labelInput.value = target.label || "";
    const startInput = makeInput("0");
    startInput.value = formatTime(target.start || 0);
    const endInput = makeInput("optional");
    endInput.value = target.end == null ? "" : formatTime(target.end);
    const noteInput = document.createElement("textarea");
    noteInput.value = target.note || "";
    noteInput.placeholder = "What happens here? Performer, dialogue, chorus, camera idea, story beat...";
    noteInput.style.cssText = "min-height:96px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;line-height:1.45;";
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const row = (label, input) => {
      const wrap = document.createElement("label");
      wrap.style.cssText = "display:grid;gap:5px;color:#e4e4e7;font-size:12px;font-weight:800;";
      const text = document.createElement("span");
      text.textContent = label;
      wrap.append(text, input);
      return wrap;
    };
    grid.append(row("Type", typeInput), row("Label", labelInput), row("Start", startInput), row("End", endInput));
    const noteWrap = row("Note", noteInput);
    const buttons = document.createElement("div");
    buttons.style.cssText = "display:flex;gap:8px;justify-content:flex-end;";
    const save = makeButton("Save", "primary");
    const del = makeButton("Delete");
    const cancel = makeButton("Cancel");
    del.style.borderColor = "#7f1d1d";
    del.style.color = "#fecaca";
    buttons.append(del, cancel, save);
    box.append(title, grid, noteWrap, buttons);
    backdrop.append(box);
    document.body.append(backdrop);
    const close = () => backdrop.remove();
    cancel.onclick = close;
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) close();
    });
    del.onclick = () => {
      pushHistory();
      state.timelineMarkers = normalizeTimelineMarkers(state.timelineMarkers).filter((item) => item.id !== target.id);
      if (state.activeTimelineMarkerId === target.id) state.activeTimelineMarkerId = "";
      render();
      autoSaveSessionQuiet("timeline marker deleted").catch(() => null);
      close();
    };
    save.onclick = () => {
      const start = parseBulkTimeValue(startInput.value);
      const end = String(endInput.value || "").trim() ? parseBulkTimeValue(endInput.value) : NaN;
      if (!Number.isFinite(start) || start < 0) {
        toast("Timeline note start time is invalid.", true);
        return;
      }
      if (Number.isFinite(end) && end <= start) {
        toast("Timeline note end must be after start.", true);
        return;
      }
      pushHistory();
      target.start = start;
      target.end = Number.isFinite(end) ? end : null;
      target.type = String(typeInput.value || "note").trim() || "note";
      target.label = String(labelInput.value || target.type || "Timeline note").trim() || "Timeline note";
      target.note = String(noteInput.value || "").trim();
      if (!state.timelineMarkers.some((item) => item.id === target.id)) state.timelineMarkers.push(target);
      const clamped = clampTimelineMarkerToNonOverlap(target, target.start, target.end ?? target.start + 4);
      target.start = clamped.start;
      target.end = clamped.end;
      state.timelineMarkers = normalizeTimelineMarkers(state.timelineMarkers);
      state.activeTimelineMarkerId = target.id;
      render();
      autoSaveSessionQuiet("timeline marker edited").catch(() => null);
      close();
    };
  }

  // What the song waveform canvas was last drawn from. Drawing it again from the same things only repeats the work.
  let drawnWaveformKey = "";
  let drawnWaveformPeaks = null;

  function drawWaveform() {
    const height = timelineHeight();
    const width = Math.max(900, Math.ceil(Math.max(1, timelineDuration()) * state.pxPerSecond));
    timelineCanvas.style.height = `${height}px`;
    timelineCanvas.style.width = `${width}px`;
    segmentLayer.style.height = `${height}px`;
    segmentLayer.style.width = `${width}px`;
    stemLayer.style.height = `${height}px`;
    stemLayer.style.width = `${width}px`;
    playhead.style.height = `${height}px`;
    // The timeline redraws at every scene change. Resizing and repainting a canvas as wide as the song and as tall as
    // the stem tracks each time stalls playback, so it is only done when something it shows has changed.
    const waveformKey = [width, height, state.pxPerSecond, state.waveformMode, timelineDuration(), timelineWaveTop(),
      currentProjectAudioPath() ? 1 : 0, state.peaks.length].join("|");
    if (waveformKey === drawnWaveformKey && state.peaks === drawnWaveformPeaks) return;
    drawnWaveformKey = waveformKey;
    drawnWaveformPeaks = state.peaks;
    timelineCanvas.height = height;
    timelineCanvas.width = width;
    const ctx = timelineCanvas.getContext("2d");
    ctx.clearRect(0, 0, width, timelineCanvas.height);
    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, width, timelineCanvas.height);
    ctx.strokeStyle = "#164e63";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const waveTop = timelineWaveTop();
    const waveBottom = timelineCanvas.height - 10;
    const waveHeight = Math.max(24, waveBottom - waveTop);
    const mid = waveTop + waveHeight / 2;
    const peaks = currentProjectAudioPath() && state.peaks.length ? state.peaks : [0];
    const gain = WAVEFORM_MODES[state.waveformMode]?.gain || 1;
    for (let x = 0; x < width; x++) {
      const index = Math.floor((x / width) * peaks.length);
      const amp = Math.min(1, Math.max(0.02, (peaks[index] || 0) * gain));
      ctx.moveTo(x, mid - amp * (waveHeight / 2));
      ctx.lineTo(x, mid + amp * (waveHeight / 2));
    }
    ctx.stroke();
    ctx.fillStyle = "rgba(103,232,249,.16)";
    ctx.fillRect(0, waveTop - 1, width, 1);
    ctx.fillStyle = "#67e8f9";
    ctx.font = "11px sans-serif";
    for (let sec = 0; sec <= timelineDuration(); sec += 10) {
      const x = sec * state.pxPerSecond;
      ctx.fillRect(x, 0, 1, timelineCanvas.height);
      ctx.fillText(formatTime(sec), x + 3, 12);
    }
  }

  function renderSegments() {
    refreshDeleteActions();
    refreshStemBand();
    segmentLayer.textContent = "";
    ensureAllSegmentRuntimeFields();
    renderSelectedTimelineRangeOverlay();
    renderTimelineMarkersOverlay();
    const overlayLabel = document.createElement("div");
    overlayLabel.textContent = "OVERLAY";
    overlayLabel.style.cssText = `position:absolute;left:4px;top:${TIMELINE_OVERLAY_TOP - 11}px;color:#a5f3fc;font-size:10px;font-weight:900;letter-spacing:.08em;pointer-events:none;text-shadow:0 1px 2px #020617;`;
    const baseLabel = document.createElement("div");
    baseLabel.textContent = "BASE";
    baseLabel.style.cssText = `position:absolute;left:4px;top:${TIMELINE_SEGMENT_TOP - 14}px;color:#a5f3fc;font-size:10px;font-weight:900;letter-spacing:.08em;pointer-events:none;text-shadow:0 1px 2px #020617;`;
    overlayLabel.style.display = state.overlayTrack.enabled ? "" : "none";
    segmentLayer.append(overlayLabel, baseLabel);
    if (state.showTimelineSceneNotes) {
      const noteLabel = document.createElement("div");
      noteLabel.textContent = "DIRECTOR NOTES";
      noteLabel.style.cssText = `position:absolute;left:4px;top:${timelineNoteTop() - 12}px;color:#a5f3fc;font-size:10px;font-weight:900;letter-spacing:.08em;pointer-events:none;text-shadow:0 1px 2px #020617;`;
      segmentLayer.append(noteLabel);
    }
    if (state.showTimelineVideoNotes) {
      const videoLabel = document.createElement("div");
      videoLabel.textContent = "VIDEO NOTES";
      videoLabel.style.cssText = `position:absolute;left:4px;top:${timelineVideoNoteTop() - 12}px;color:#bae6fd;font-size:10px;font-weight:900;letter-spacing:.08em;pointer-events:none;text-shadow:0 1px 2px #020617;`;
      segmentLayer.append(videoLabel);
    }
    if (state.showTimelineLyricNotes) {
      const lyricLabel = document.createElement("div");
      lyricLabel.textContent = "LYRIC NOTES";
      lyricLabel.style.cssText = `position:absolute;left:4px;top:${timelineLyricNoteTop() - 12}px;color:#f0abfc;font-size:10px;font-weight:900;letter-spacing:.08em;pointer-events:none;text-shadow:0 1px 2px #020617;`;
      segmentLayer.append(lyricLabel);
    }
    const visibleTimelineOverlays = state.overlayTrack.enabled ? state.overlaySegments : [];
    for (const segment of [...visibleTimelineOverlays, ...state.segments]) {
      const isOverlay = segmentTrack(segment) === "overlay";
      const blockTop = isOverlay ? TIMELINE_OVERLAY_TOP : TIMELINE_SEGMENT_TOP;
      const blockHeight = isOverlay ? TIMELINE_OVERLAY_HEIGHT : TIMELINE_SEGMENT_HEIGHT;
      const block = document.createElement("button");
      block.type = "button";
      block.innerHTML = `<span style="position:relative;z-index:2;display:block;font-weight:900;">${escapeHtml(timelineSegmentLabel(segment))}</span><span style="position:relative;z-index:2;display:block;margin-top:3px;font-size:10px;color:#d4d4d8;">${formatTime(segment.start)} - ${formatTime(segment.end)} | ${formatDurationSeconds(segment.start, segment.end)}s</span>`;
      const left = segment.start * state.pxPerSecond;
      const width = Math.max(24, (segment.end - segment.start) * state.pxPerSecond);
      const videoMode = currentVideoMode();
      const engine = normalizeProjectVideoEngine(state.projectVideoEngine);
      const miniMaxFLF = engine === "minimax_h3" && miniMaxI2VFLFEnabled(segment, engine, miniMaxH3ModeForSegment(segment));
      const showFirstLastFrameThumb = !isOverlay && (
        miniMaxFLF || (engine !== "minimax_h3" && (
          videoMode === "flf"
          || (videoMode === "rtv" && rtvReferenceBehaviorForSegment(segment) === "first_last_frame")
        ))
      );
      const previewThumbPath = selectedSegmentImageThumbnailPath(segment);
      const location = !isOverlay && showLocationThumbnails
        ? mappedLocation(state.fluxReferenceBuilder, segment, state.segments.indexOf(segment)) : null;
      const locationImage = location?.image?.data || (location?.image?.path ? makeEditorImageUrl(location.image.path) : "");
      const thumb = showLocationThumbnails ? "" : !showFirstLastFrameThumb && previewThumbPath ? makeEditorThumbnailUrl(previewThumbPath) : "";
      const hasVideoPreview = Boolean(selectedSegmentVideoPath(segment));
      const inserted = !isOverlay && state.srtMode && segment.source !== "srt";
      const lockedByVideo = hasLockedVideo(segment);
      const isActive = Boolean(state.activeId) && segment.id === state.activeId;
      const isMultiSelected = isSegmentMultiSelected(segment);
      // Rendering now: blue glow. Rendering and selected: purple glow. Selected only: red glow.
      const isRendering = segment.video_status === "running";
      const isSelected = isActive || isMultiSelected;
      const borderColor = isRendering && isSelected ? "#c084fc" : isRendering ? "#3b82f6" : isSelected ? "#ef4444" : lockedByVideo ? "#a3e635" : isOverlay ? "#f97316" : inserted ? "#f59e0b" : "#0891b2";
      const borderWidth = isRendering || isSelected ? "3px" : "1px";
      const shadow = isRendering && isSelected
        ? "0 0 6px 1px rgba(192,132,252,.95), 0 0 12px 2px rgba(168,85,247,.55)"
        : isRendering
          ? "0 0 6px 1px rgba(96,165,250,.95), 0 0 12px 2px rgba(59,130,246,.55)"
          : isSelected
            ? "0 0 5px 1px rgba(239,68,68,.85)"
            : "none";
      block.style.cssText = `
        position:absolute;left:${left}px;top:${blockTop}px;width:${width}px;height:${blockHeight}px;
        border:${borderWidth} solid ${borderColor};
        border-radius:5px;background:${showFirstLastFrameThumb ? "#020617" : thumb ? `linear-gradient(rgba(0,0,0,.18),rgba(0,0,0,.18)), url("${thumb}") center / auto 100% repeat-x` : isOverlay ? "#7c2d12" : inserted ? "#92400e" : segment.image ? "#166534" : "#164e63"};
        color:#f4f4f5;font-size:11px;font-weight:800;overflow:hidden;cursor:pointer;pointer-events:auto;
        box-shadow:${shadow};
      `;
      if (showLocationThumbnails && !isOverlay) {
        if (locationImage) {
          const image = document.createElement("img");
          image.src = locationImage;
          image.alt = location?.name || "Mapped location";
          image.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;pointer-events:none;";
          block.prepend(image);
        }
        const label = document.createElement("span");
        label.textContent = location?.name || "No location mapped";
        label.style.cssText = "position:absolute;left:4px;right:4px;bottom:4px;z-index:3;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;background:rgba(2,6,23,.82);padding:2px 4px;border-radius:3px;font-size:10px;text-align:left;";
        block.append(label);
      } else if (showFirstLastFrameThumb) appendTimelineFirstLastFrameThumbnail(block, segment);
      else if (!thumb && hasVideoPreview) appendTimelineVideoThumbnail(block, segment);
      if (isOverlay) {
        normalizeOverlayClip(segment);
        const eye = document.createElement("span");
        eye.textContent = segment.overlay_enabled === false ? "◉̸" : "◉";
        eye.title = segment.overlay_enabled === false ? "Enable this overlay clip" : "Hide this overlay clip and show the base video";
        eye.style.cssText = "position:absolute;left:10px;top:5px;width:22px;height:20px;display:flex;align-items:center;justify-content:center;border:1px solid #fdba74;border-radius:4px;background:rgba(67,20,7,.92);color:#ffedd5;font-size:13px;font-weight:900;z-index:6;";
        eye.onpointerdown = (event) => event.stopPropagation();
        eye.onclick = (event) => {
          event.stopPropagation();
          pushHistory();
          segment.overlay_enabled = segment.overlay_enabled === false;
          render();
          autoSaveSessionQuiet("overlay clip visibility changed");
        };
        const lock = document.createElement("span");
        lock.textContent = segment.overlay_locked === false ? "🔓" : "🔒";
        lock.title = segment.overlay_locked === false ? "Lock this overlay clip" : "Unlock this overlay clip for moving and trimming";
        lock.style.cssText = "position:absolute;left:36px;top:5px;width:24px;height:20px;display:flex;align-items:center;justify-content:center;border:1px solid #fdba74;border-radius:4px;background:rgba(67,20,7,.92);color:#ffedd5;font-size:12px;z-index:6;";
        lock.onpointerdown = (event) => event.stopPropagation();
        lock.onclick = (event) => {
          event.stopPropagation();
          pushHistory();
          segment.overlay_locked = segment.overlay_locked === false;
          render();
          autoSaveSessionQuiet("overlay clip lock changed");
        };
        if (segment.overlay_enabled === false) block.style.opacity = ".48";
        block.append(eye, lock);
      }
      if (!isOverlay && state.segments.indexOf(segment) > 0) {
        // Quick button: lock this scene and make it a Latent Continuation Masked scene with the Masked transition.
        const maskedOn = Boolean(segment.use_scene_minimax_h3_settings)
          && segment.minimax_h3_settings?.continuity_mode === "latent_continuation_masked";
        const maskedButton = document.createElement("span");
        maskedButton.textContent = "⛓";
        maskedButton.title = maskedOn
          ? "Locked to Latent Continuation Masked (Masked transition). Click to turn it off."
          : "Lock this scene and set it to Latent Continuation Masked with the Masked transition (continue, then one smooth move). Click again to turn it off.";
        maskedButton.style.cssText = `position:absolute;left:12px;top:4px;z-index:5;width:18px;height:16px;display:flex;align-items:center;justify-content:center;border:1px solid ${maskedOn ? "#a3e635" : "#67e8f9"};border-radius:4px;background:${maskedOn ? "rgba(54,83,20,.95)" : "rgba(8,47,73,.92)"};color:${maskedOn ? "#ecfccb" : "#e0f2fe"};font-size:11px;line-height:1;cursor:pointer;`;
        maskedButton.onpointerdown = (event) => {
          event.stopPropagation();
          // The card is draggable, and a click that moves a pixel would start a drag and cancel the click.
          if (block.draggable) {
            block.draggable = false;
            const restore = () => {
              block.draggable = true;
              window.removeEventListener("pointerup", restore, true);
              window.removeEventListener("pointercancel", restore, true);
            };
            window.addEventListener("pointerup", restore, true);
            window.addEventListener("pointercancel", restore, true);
          }
        };
        maskedButton.ondblclick = (event) => event.stopPropagation();
        maskedButton.onclick = (event) => {
          event.stopPropagation();
          try {
            Promise.resolve(toggleSceneMaskedContinuation(segment)).catch((error) => {
              console.error("[VRGDG Timeline] Masked continuation button failed", error);
              toast(`Masked continuation button failed: ${String(error?.message || error)}`, true);
            });
          } catch (error) {
            console.error("[VRGDG Timeline] Masked continuation button failed", error);
            toast(`Masked continuation button failed: ${String(error?.message || error)}`, true);
          }
        };
        block.append(maskedButton);
      }
      const dblClickHint = !isOverlay ? "Double-click to review line & performer mapping. Shift + double-click to edit the Storyboard Card." : "";
      block.title = lockedByVideo ? "This scene has a generated video, so timing is locked." : dblClickHint;
      const dragImageSource = segmentImageSource(segment);
      if (dragImageSource) {
        block.draggable = true;
        block.ondragstart = (event) => {
          event.dataTransfer.setData("application/x-vrgdg-segment-id", segment.id);
          event.dataTransfer.setData("text/plain", segment.label || "Scene image");
          event.dataTransfer.effectAllowed = "copy";
        };
      }
      const hasPreviewImage = Boolean(segmentImageSource(segment));
      const hasPreviewVideo = Boolean(selectedSegmentVideoPath(segment));
      const imageHistory = Array.isArray(segment?.image_history) ? segment.image_history : [];
      const videoHistory = Array.isArray(segment?.video_history) ? segment.video_history : [];
      if ((segment.preview_mode === "video" && videoHistory.length) || (segment.preview_mode !== "video" && imageHistory.length)) {
        const historyButton = document.createElement("span");
        const isVideoMode = segment.preview_mode === "video";
        const count = isVideoMode ? videoHistory.length : imageHistory.length;
        const index = isVideoMode ? Number(segment.video_history_index || 0) : Number(segment.image_history_index || 0);
        historyButton.textContent = `${Math.max(1, index + 1)}/${count}`;
        historyButton.title = isVideoMode ? "Cycle generated video previews for this scene." : "Cycle generated image previews for this scene.";
        historyButton.style.cssText = `position:absolute;right:13px;top:5px;min-width:34px;height:20px;display:flex;align-items:center;justify-content:center;border:1px solid ${isVideoMode ? "#a78bfa" : "#67e8f9"};border-radius:4px;background:${isVideoMode ? "rgba(59,7,100,.92)" : "rgba(8,47,73,.92)"};color:${isVideoMode ? "#f3e8ff" : "#e0f2fe"};font-size:10px;font-weight:900;z-index:3;`;
        historyButton.onpointerdown = (event) => event.stopPropagation();
        historyButton.onclick = (event) => {
          event.stopPropagation();
          if (isVideoMode) cycleSegmentVideoHistory(segment);
          else cycleSegmentImageHistory(segment);
        };
        block.append(historyButton);
      }
      if (hasPreviewImage && hasPreviewVideo) {
        const modeButton = document.createElement("span");
        modeButton.textContent = segment.preview_mode === "image" ? "I" : "V";
        modeButton.title = "Switch preview between image and video.";
        modeButton.style.cssText = "position:absolute;right:13px;top:29px;min-width:34px;height:20px;display:flex;align-items:center;justify-content:center;border:1px solid #a3e635;border-radius:4px;background:rgba(20,83,45,.92);color:#ecfccb;font-size:10px;font-weight:900;z-index:3;";
        modeButton.onpointerdown = (event) => event.stopPropagation();
        modeButton.onclick = (event) => {
          event.stopPropagation();
          toggleSegmentPreviewMode(segment);
        };
        block.append(modeButton);
      }
      const sceneLut = normalizeSceneLut(segment?.lut || {});
      if (!isOverlay && sceneLut?.enabled !== false && sceneLut?.name) {
        const lutBadge = document.createElement("span");
        lutBadge.textContent = "LUT";
        lutBadge.title = `LUT: ${sceneLut.label || lutLabelFromName(sceneLut.name)}`;
        lutBadge.style.cssText = "position:absolute;left:10px;bottom:5px;min-width:30px;height:18px;display:flex;align-items:center;justify-content:center;border:1px solid #f0abfc;border-radius:4px;background:rgba(88,28,135,.9);color:#f5d0fe;font-size:10px;font-weight:900;z-index:3;";
        lutBadge.onpointerdown = (event) => event.stopPropagation();
        block.append(lutBadge);
      }
      const sceneGrain = normalizeSceneFilmGrain(segment?.film_grain || {});
      if (!isOverlay && sceneGrain && sceneGrain.enabled !== false) {
        const grainBadge = document.createElement("span");
        grainBadge.textContent = "GRAIN";
        grainBadge.title = `Film Grain: ${filmGrainLabel(sceneGrain)}`;
        grainBadge.style.cssText = "position:absolute;left:46px;bottom:5px;min-width:44px;height:18px;display:flex;align-items:center;justify-content:center;border:1px solid #fbbf24;border-radius:4px;background:rgba(113,63,18,.9);color:#fde68a;font-size:10px;font-weight:900;z-index:3;";
        grainBadge.onpointerdown = (event) => event.stopPropagation();
        block.append(grainBadge);
      }
      if (!isOverlay && segment._latentDirty) {
        const dirtyBadge = document.createElement("span");
        dirtyBadge.textContent = "LATENT DIRTY";
        dirtyBadge.title = "Predecessor scene was re-rendered or timeline was shifted since this latent was created. Re-rendering this scene is recommended.";
        dirtyBadge.style.cssText = "position:absolute;left:96px;bottom:5px;min-width:76px;height:18px;display:flex;align-items:center;justify-content:center;border:1px solid #f59e0b;border-radius:4px;background:rgba(120,53,15,.92);color:#fef3c7;font-size:9px;font-weight:900;z-index:3;";
        dirtyBadge.onpointerdown = (event) => event.stopPropagation();
        block.append(dirtyBadge);
      }
      const leftHandle = document.createElement("div");
      leftHandle.style.cssText = "position:absolute;left:0;top:0;bottom:0;width:8px;background:rgba(255,255,255,.25);cursor:ew-resize;z-index:4;";
      const rightHandle = document.createElement("div");
      rightHandle.style.cssText = "position:absolute;right:0;top:0;bottom:0;width:8px;background:rgba(255,255,255,.25);cursor:ew-resize;z-index:4;";
      block.append(leftHandle, rightHandle);
      block.onclick = (event) => handleSegmentPick(segment, event);
      block.oncontextmenu = (event) => openSegmentContextMenu(event, segment);
      if (!isOverlay) {
        block.ondblclick = (event) => {
          event.preventDefault();
          event.stopPropagation();
          openTimelineSceneCard(segment, event);
        };
      }
      enableImageDrop(block, segment);
      enableLutDrop(block, segment);
      enablePostEffectDrop(block, segment);
      makeDragHandle(block, segment, "move");
      makeDragHandle(leftHandle, segment, "start");
      makeDragHandle(rightHandle, segment, "end");
      segmentLayer.append(block);
      if (!isOverlay && state.showTimelineSceneNotes) {
        const noteField = "timeline_note";
        const noteBox = document.createElement("textarea");
        noteBox.dataset.sceneNoteSegmentId = segment.id || "";
        noteBox.value = String(segment[noteField] || "");
        noteBox.placeholder = "Director note...";
        noteBox.title = "Extra note saved on this scene. This does not replace Prompt Creator notes or main scene notes.";
        noteBox.style.cssText = `
          position:absolute;left:${left}px;top:${timelineNoteTop()}px;width:${Math.max(86, width)}px;height:${TIMELINE_NOTE_HEIGHT}px;
          box-sizing:border-box;resize:none;z-index:3;pointer-events:auto;
          border:1px solid ${isActive ? "#ef4444" : "#155e75"};border-radius:5px;
          background:rgba(8,47,73,.92);color:#ecfeff;padding:6px;font-size:11px;line-height:1.25;
          box-shadow:${isActive ? "0 0 0 2px rgba(239,68,68,.22)" : "none"};
        `;
        noteBox.onpointerdown = (event) => event.stopPropagation();
        noteBox.onclick = (event) => event.stopPropagation();
        noteBox.onkeydown = (event) => event.stopPropagation();
        noteBox.onfocus = () => {
          if (!state.multiSelectMode) {
            state.activeId = segment.id;
            state.activeTrack = "base";
          }
          noteBox.dataset.previousValue = noteBox.value;
          noteBox.dataset.historyPushed = "0";
        };
        noteBox.oninput = () => {
          if (noteBox.dataset.historyPushed !== "1" && noteBox.dataset.previousValue !== noteBox.value) {
            pushHistory();
            noteBox.dataset.historyPushed = "1";
          }
          segment[noteField] = noteBox.value || "";
        };
        noteBox.onchange = () => {
          if (noteBox.dataset.deleted === "1") return;
          segment[noteField] = noteBox.value || "";
          autoSaveSessionQuiet("timeline scene note edited");
        };
        noteBox.oncontextmenu = (event) => openDirectorNoteContextMenu(event, segment, noteBox);
        segmentLayer.append(noteBox);
      }
      if (!isOverlay && state.showTimelineVideoNotes) {
        const videoBox = document.createElement("textarea");
        videoBox.value = String(segment.i2v_notes || "");
        videoBox.placeholder = "Video motion/action notes...";
        videoBox.title = "Video notes for this scene. Gemma uses this for I2V/T2V camera motion, character movement, action, and performance direction.";
        videoBox.style.cssText = `
          position:absolute;left:${left}px;top:${timelineVideoNoteTop()}px;width:${Math.max(86, width)}px;height:${TIMELINE_NOTE_HEIGHT}px;
          box-sizing:border-box;resize:none;z-index:3;pointer-events:auto;
          border:1px solid ${isActive ? "#ef4444" : "#2563eb"};border-radius:5px;
          background:rgba(30,64,175,.82);color:#dbeafe;padding:6px;font-size:11px;line-height:1.25;
          box-shadow:${isActive ? "0 0 0 2px rgba(239,68,68,.22)" : "none"};
        `;
        videoBox.onpointerdown = (event) => event.stopPropagation();
        videoBox.onclick = (event) => event.stopPropagation();
        videoBox.onkeydown = (event) => event.stopPropagation();
        videoBox.onfocus = () => {
          videoBox.dataset.previousValue = videoBox.value;
          videoBox.dataset.historyPushed = "0";
        };
        videoBox.oninput = () => {
          if (videoBox.dataset.historyPushed !== "1" && videoBox.dataset.previousValue !== videoBox.value) {
            pushHistory();
            videoBox.dataset.historyPushed = "1";
          }
          segment.i2v_notes = videoBox.value || "";
          if (segment.id === state.activeId) i2vNotesInput.value = segment.i2v_notes;
        };
        videoBox.onchange = () => {
          segment.i2v_notes = videoBox.value || "";
          if (segment.id === state.activeId) syncInspector();
          autoSaveSessionQuiet("timeline video note edited");
        };
        segmentLayer.append(videoBox);
      }
      if (!isOverlay && state.showTimelineLyricNotes) {
        const lyricBox = document.createElement("textarea");
        lyricBox.value = String(segment.lyric_text || "");
        lyricBox.placeholder = "Line / lyric / dialogue...";
        lyricBox.title = "Line text for this scene. Gemma uses this for singing, speaking, or visual-only behavior based on Video Type.";
        lyricBox.style.cssText = `
          position:absolute;left:${left}px;top:${timelineLyricNoteTop()}px;width:${Math.max(24, width)}px;height:${TIMELINE_NOTE_HEIGHT}px;
          box-sizing:border-box;resize:none;z-index:3;pointer-events:auto;
          border:1px solid ${isActive ? "#ef4444" : "#7e22ce"};border-radius:5px;
          background:rgba(59,7,100,.86);color:#fae8ff;padding:6px;font-size:11px;line-height:1.25;
          box-shadow:${isActive ? "0 0 0 2px rgba(239,68,68,.22)" : "none"};
        `;
        lyricBox.onpointerdown = (event) => event.stopPropagation();
        lyricBox.onclick = (event) => event.stopPropagation();
        lyricBox.onkeydown = (event) => event.stopPropagation();
        lyricBox.onfocus = () => {
          if (!state.multiSelectMode) {
            state.activeId = segment.id;
            state.activeTrack = "base";
            lyricTextInput.value = lyricBox.value;
            lyricTextInput.dataset.vrgdgInspectorSegmentId = String(segment.id || "");
            lyricTextInput.dataset.vrgdgUserEdited = "0";
          }
          lyricBox.dataset.previousValue = lyricBox.value;
          lyricBox.dataset.savedValue = lyricBox.value;
          lyricBox.dataset.historyPushed = "0";
        };
        lyricBox.oninput = () => {
          if (lyricBox.dataset.historyPushed !== "1" && lyricBox.dataset.previousValue !== lyricBox.value) {
            pushHistory();
            lyricBox.dataset.historyPushed = "1";
          }
          // Resolve the live object by id. Timeline rerenders can replace the
          // segment array while an older textarea closure is still focused.
          const liveSegment = state.segments.find((item) => item.id === segment.id) || segment;
          liveSegment.lyric_text = lyricBox.value || "";
          liveSegment.lyric_no_lip_sync = isInstrumentalLyricText(liveSegment.lyric_text);
          if (liveSegment.id === state.activeId) {
            // Keep the inspector mirror explicitly in sync. Otherwise a stale
            // inspector edit can overwrite this timeline edit during autosave.
            lyricTextInput.value = liveSegment.lyric_text;
            lyricTextInput.dataset.vrgdgInspectorSegmentId = String(liveSegment.id || "");
            lyricTextInput.dataset.vrgdgUserEdited = "0";
          }
        };
        const saveTimelineLyricEdit = () => {
          const liveSegment = state.segments.find((item) => item.id === segment.id) || segment;
          liveSegment.lyric_text = lyricBox.value || "";
          liveSegment.lyric_no_lip_sync = isInstrumentalLyricText(liveSegment.lyric_text);
          if (liveSegment.id === state.activeId) {
            syncInspector();
            lyricTextInput.dataset.vrgdgInspectorSegmentId = String(liveSegment.id || "");
            lyricTextInput.dataset.vrgdgUserEdited = "0";
          }
          lyricBox.dataset.savedValue = lyricBox.value;
          applyLyricSectionsFromReferenceText(state.segments, state.lyricMapper?.source_text || "");
          syncLyricMapperFromSegments();
          autoSaveSessionQuiet("timeline line note edited");
        };
        lyricBox.onchange = saveTimelineLyricEdit;
        lyricBox.onblur = () => {
          // Some browser/layout interactions do not dispatch change before a
          // timeline rerender; blur is the final persistence safeguard.
          if (lyricBox.dataset.savedValue !== lyricBox.value) saveTimelineLyricEdit();
        };
        segmentLayer.append(lyricBox);
      }
      if (!isOverlay && segment.custom_audio_peaks?.length) {
        const audioStart = audioTimelineStart(segment);
        const audioDuration = Math.max(0.1, audioChunkDuration(segment));
        const audioLeft = audioStart * state.pxPerSecond;
        const audioWidth = Math.max(24, audioDuration * state.pxPerSecond);
        const audioWave = document.createElement("canvas");
        audioWave.width = Math.max(1, Math.floor(audioWidth));
        audioWave.height = TIMELINE_SCENE_AUDIO_HEIGHT;
        audioWave.title = "Custom scene audio waveform. Click for audio cut/delete options.";
        audioWave.style.cssText = `
          position:absolute;left:${audioLeft}px;top:${TIMELINE_SCENE_AUDIO_TOP}px;width:${audioWidth}px;height:${TIMELINE_SCENE_AUDIO_HEIGHT}px;
          z-index:1;pointer-events:auto;cursor:pointer;background:rgba(88,28,135,.36);border:1px solid rgba(216,180,254,.35);border-radius:4px;
        `;
        audioWave.onclick = (event) => openAudioContextMenu(event, segment);
        audioWave.oncontextmenu = (event) => openAudioContextMenu(event, segment);
        segmentLayer.append(audioWave);
        drawSegmentAudioWaveform(audioWave, segment.custom_audio_peaks);
      }
    }
    renderStemTracks();
    renderBeatMarkersOverlay();
  }

  // Scene statuses change through renderList(), so repaint the timeline when the set of rendering scenes changes.
  let renderingKey = "";
  function renderList() {
    const nextRenderingKey = [...state.segments, ...state.overlaySegments].filter((segment) => segment.video_status === "running").map((segment) => segment.id).join(",");
    if (nextRenderingKey !== renderingKey) {
      renderingKey = nextRenderingKey;
      renderSegments();
    }
    sceneListPane.textContent = "";
    ensureAllSegmentRuntimeFields();
    for (const [index, segment] of state.segments.entries()) {
      const row = document.createElement("div");
      row.role = "button";
      row.tabIndex = 0;
      const thumb = mediaThumbnailHtml(segment, 56);
      const inserted = state.srtMode && segment.source !== "srt";
      const t2iDone = Boolean(segmentImageSource(segment));
      const i2vDone = Boolean(String(segment.i2v_prompt || "").trim());
      const videoDone = Boolean(segment.video_path);
      const imageHistory = Array.isArray(segment?.image_history) ? segment.image_history : [];
      const videoHistory = Array.isArray(segment?.video_history) ? segment.video_history : [];
      const historyStatus = imageHistory.length ? `<span style="border:1px solid #67e8f9;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#bae6fd;">IMG ${imageHistory.length}</span>` : "";
      const videoHistoryStatus = videoHistory.length ? `<span style="border:1px solid #a78bfa;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#f3e8ff;">VID ${videoHistory.length}</span>` : "";
      const zStatus = segment.use_scene_zimage_settings ? `<span style="border:1px solid #f59e0b;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#fde68a;">Z custom</span>` : "";
      const audioStatus = segment.custom_audio_path ? `<span style="border:1px solid #a78bfa;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#ddd6fe;">AUD</span>` : "";
      const lutStatus = sceneLutBadgeHtml(segment);
      const grainStatus = sceneFilmGrainBadgeHtml(segment);
      const status = `
        <div style="display:flex;gap:6px;margin-top:6px;align-items:center;flex-wrap:wrap;">
          <span style="border:1px solid ${t2iDone ? "#22c55e" : "#52525b"};border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:${t2iDone ? "#bbf7d0" : "#a1a1aa"};">T2I ${t2iDone ? "OK" : "--"}</span>
          <span style="border:1px solid ${i2vDone ? "#22c55e" : "#52525b"};border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:${i2vDone ? "#bbf7d0" : "#a1a1aa"};">I2V ${i2vDone ? "OK" : "--"}</span>
          <span style="border:1px solid ${videoDone ? "#22c55e" : "#52525b"};border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:${videoDone ? "#bbf7d0" : "#a1a1aa"};">VID ${videoDone ? "OK" : "--"}</span>
          ${historyStatus}
          ${videoHistoryStatus}
          ${zStatus}
          ${audioStatus}
          ${lutStatus}
          ${grainStatus}
        </div>
      `;
      const isActive = Boolean(state.activeId) && segment.id === state.activeId;
      const isMultiSelected = isSegmentMultiSelected(segment);
      row.style.cssText = `width:100%;text-align:left;border:${isActive || isMultiSelected ? "3px" : "1px"} solid ${isActive || isMultiSelected ? "#ef4444" : inserted ? "#f59e0b" : "#3f3f46"};border-radius:7px;background:${isActive || isMultiSelected ? "#3f1d24" : inserted ? "#451a03" : "#27272a"};color:#fafafa;padding:8px;margin-bottom:8px;cursor:pointer;box-shadow:${isActive || isMultiSelected ? "0 0 0 2px rgba(239,68,68,.25), 0 0 18px rgba(239,68,68,.42)" : "none"};`;
      row.innerHTML = `<div style="font-weight:800;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${index + 1}. ${escapeHtml(segment.label || "Scene")}</div><div style="font-size:11px;color:#a1a1aa;margin-top:4px;">Duration in seconds: ${formatDurationSeconds(segment.start, segment.end)}</div><div style="font-size:11px;color:#71717a;margin-top:2px;">${formatTime(segment.start)} - ${formatTime(segment.end)}</div>${status}${thumb}`;
      row.onclick = (event) => handleSegmentPick(segment, event);
      row.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          handleSegmentPick(segment, event);
        }
      };
      const optionsButton = document.createElement("span");
      optionsButton.textContent = "Options";
      optionsButton.title = "Scene options";
      optionsButton.style.cssText = "display:inline-flex;margin-top:6px;border:1px solid #3f3f46;border-radius:4px;padding:3px 6px;font-size:10px;font-weight:900;color:#f4f4f5;background:#18181b;";
      optionsButton.onclick = (event) => {
        event.stopPropagation();
        setActiveSegment(segment);
        openSceneOptions(segment);
      };
      row.append(optionsButton);
      const dragImageSource = segmentImageSource(segment);
      if (dragImageSource) {
        row.draggable = true;
        row.ondragstart = (event) => {
          event.dataTransfer.setData("application/x-vrgdg-segment-id", segment.id);
          event.dataTransfer.setData("text/plain", segment.label || "Scene image");
          event.dataTransfer.effectAllowed = "copy";
        };
      }
      enableImageDrop(row, segment);
      enableLutDrop(row, segment);
      enablePostEffectDrop(row, segment);
      sceneListPane.append(row);
    }
    if (state.overlaySegments.length) {
      const header = document.createElement("div");
      header.textContent = "Insert timeline";
      header.style.cssText = "margin:10px 0 8px;color:#fdba74;font-size:12px;font-weight:900;text-transform:uppercase;letter-spacing:.08em;";
      sceneListPane.append(header);
    }
    for (const [index, segment] of state.overlaySegments.entries()) {
      const row = document.createElement("div");
      row.role = "button";
      row.tabIndex = 0;
      const thumb = mediaThumbnailHtml(segment, 50);
      const isActive = Boolean(state.activeId) && segment.id === state.activeId;
      const isMultiSelected = isSegmentMultiSelected(segment);
      row.style.cssText = `width:100%;text-align:left;border:${isActive || isMultiSelected ? "3px" : "1px"} solid ${isActive || isMultiSelected ? "#ef4444" : "#f97316"};border-radius:7px;background:${isActive || isMultiSelected ? "#3f1d24" : "#431407"};color:#fafafa;padding:8px;margin-bottom:8px;cursor:pointer;box-shadow:${isActive || isMultiSelected ? "0 0 0 2px rgba(239,68,68,.25), 0 0 18px rgba(239,68,68,.42)" : "none"};`;
      row.innerHTML = `<div style="font-weight:800;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">Insert ${index + 1}. ${escapeHtml(segment.label || "Insert")}</div><div style="font-size:11px;color:#fed7aa;margin-top:4px;">${formatTime(segment.start)} - ${formatTime(segment.end)} | ${formatDurationSeconds(segment.start, segment.end)}s</div>${thumb}`;
      row.onclick = (event) => handleSegmentPick(segment, event);
      row.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          handleSegmentPick(segment, event);
        }
      };
      enableImageDrop(row, segment);
      sceneListPane.append(row);
    }
  }

  return {
    drawWaveform, normalizeSegments, openTimelineMarkerEditor, renderList, renderSegments,
    snapAddedSegmentEndToNearestBeat, snapTimeToBeat,
  };
}
