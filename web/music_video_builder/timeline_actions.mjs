import { normalizeOverlayClip, normalizeOverlayTrackState, OVERLAY_TRACK_HELP_HTML } from "../VRGDG_OverlayTrack.js";
import { confirmDeleteMediaAction } from "./batch_actions.mjs";
import { confirmDestructiveAction } from "./confirm_dialog.mjs";
import { makeEditorVideoUrl, postJson } from "./comfy_api.mjs";
import { makeButton, makeField, makeSelect, toast } from "./controls.mjs";
import { showAddSegmentPositionModal, showLongSegmentConfirm } from "./dialogs.mjs";
import { formatTime } from "./format.mjs";
import { isInstrumentalLyricText } from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder } from "./reference_data.mjs";
import { createUniqueSegmentId, newSegment, newTimelineMarker, renumberGenericBaseSceneLabels, rewriteRenamedScenePaths, sortSegments } from "./segments.mjs";
import { selectedSegmentVideoPath, selectedSegmentVideoThumbnailPath } from "./selection_preview.mjs";
import {
  mediaPathKey,
  normalizeSegmentVideoHistory,
  normalizeTimelineMarkers,
  normalizeTimelineRange,
} from "./timeline_state.mjs";
import { shiftSegmentTiming } from "./timeline_view.mjs";

function waitForVideoEvent(video, eventName, timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    let done = false;
    const cleanup = () => {
      video.removeEventListener(eventName, onEvent);
      video.removeEventListener("error", onError);
      clearTimeout(timer);
    };
    const finish = (callback, value) => {
      if (done) return;
      done = true;
      cleanup();
      callback(value);
    };
    const onEvent = () => finish(resolve);
    const onError = () => finish(reject, new Error("Video frame could not be loaded."));
    const timer = setTimeout(() => finish(reject, new Error("Timed out while loading the selected video frame.")), timeoutMs);
    video.addEventListener(eventName, onEvent, { once: true });
    video.addEventListener("error", onError, { once: true });
  });
}

async function seekVideoForCapture(video, time) {
  if (video.readyState < 1) await waitForVideoEvent(video, "loadedmetadata");
  const duration = Number(video.duration || 0);
  const maxTime = Number.isFinite(duration) && duration > 0 ? Math.max(0, duration - 0.02) : Number.MAX_SAFE_INTEGER;
  const target = Math.max(0, Math.min(Number(time || 0), maxTime));
  if (Math.abs(Number(video.currentTime || 0) - target) > 0.02) {
    const seeked = waitForVideoEvent(video, "seeked");
    video.currentTime = target;
    await seeked;
  }
  if (video.readyState < 2) await waitForVideoEvent(video, "loadeddata");
}

export function captureVideoFrameDataUrl(video) {
  const width = Number(video.videoWidth || 0);
  const height = Number(video.videoHeight || 0);
  if (!width || !height) throw new Error("The selected video frame is not ready yet.");
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not create a frame capture canvas.");
  context.drawImage(video, 0, 0, width, height);
  return canvas.toDataURL("image/png");
}

export function parseBulkTimeValue(raw) {
  const text = String(raw || "").trim().replace(",", ".");
  if (!text) return NaN;
  const parts = text.split(":").map((part) => part.trim());
  if (parts.length === 1) return Number(parts[0]);
  if (parts.length === 2) return Number(parts[0]) * 60 + Number(parts[1]);
  if (parts.length === 3) return Number(parts[0]) * 3600 + Number(parts[1]) * 60 + Number(parts[2]);
  return NaN;
}

function cleanBulkTimingLines(text) {
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*(?:[-*]|\d+[.)])\s*/, "").trim())
    .filter((line) => line && !line.startsWith("#"));
}

function parseBulkSegmentTimings(text, mode, appendStart = 0) {
  const lines = cleanBulkTimingLines(text);
  const segments = [];
  if (mode === "durations") {
    let cursor = Number(appendStart || 0);
    for (const line of lines) {
      const duration = parseBulkTimeValue(line);
      if (!Number.isFinite(duration) || duration <= 0) {
        throw new Error(`Invalid duration: ${line}`);
      }
      segments.push({ start: cursor, end: cursor + duration });
      cursor += duration;
    }
  } else if (mode === "ranges") {
    for (const line of lines) {
      const match = line.match(/^(.+?)\s*(?:-->|-|to)\s*(.+)$/i);
      if (!match) throw new Error(`Invalid start/end row: ${line}`);
      const start = parseBulkTimeValue(match[1]);
      const end = parseBulkTimeValue(match[2]);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        throw new Error(`Invalid start/end row: ${line}`);
      }
      segments.push({ start, end });
    }
  } else {
    const markers = lines.map(parseBulkTimeValue);
    if (markers.some((value) => !Number.isFinite(value))) {
      throw new Error("One or more timestamp markers could not be read.");
    }
    for (let index = 0; index < markers.length - 1; index += 1) {
      const start = markers[index];
      const end = markers[index + 1];
      if (end <= start) throw new Error("Timestamp markers must go from earliest to latest.");
      segments.push({ start, end });
    }
  }
  if (!segments.length) throw new Error("No segments were found in the box.");
  return segments;
}

export function showOverlayTrackHelp() {
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100010;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:20px;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(720px,calc(100vw - 40px));max-height:calc(100vh - 60px);overflow:auto;border:1px solid #0891b2;border-radius:9px;background:#111827;color:#d4d4d8;box-shadow:0 24px 80px rgba(0,0,0,.65);padding:18px;display:flex;flex-direction:column;gap:14px;";
  const title = document.createElement("div");
  title.textContent = "Overlay Track Help";
  title.style.cssText = "font-size:18px;font-weight:900;color:#cffafe;";
  const content = document.createElement("div");
  content.innerHTML = OVERLAY_TRACK_HELP_HTML;
  const close = makeButton("Close", "primary");
  close.onclick = () => backdrop.remove();
  box.append(title, content, close);
  backdrop.append(box);
  backdrop.onpointerdown = (event) => {
    if (event.target === backdrop) backdrop.remove();
  };
  document.body.append(backdrop);
}

export function createTimelineActions({
  activeSegment, addOverlaySegmentButton, addSceneImageHistoryPath, allEditableSegments, audio,
  autoSaveSessionQuiet, baseSceneVideoTrimKind, chooseRenderedSceneTrimAtPlayhead, closeBaseTimelineGap,
  currentGlobalTime, customImageFileInput, enforceAudioTimelineEnd,
  ensureSegmentRuntimeFields, freezeTimingControl, loadDirtyLatentBadges, loadedGlobalAudioDuration,
  nextFreeTimelineMarkerRange, nextOverlaySlotNumber, openTimelineMarkerEditor, overlayTrackToggleButton,
  pauseTimelineForEditing, playbackSegmentAtTime, previewVideo, projectInput, pushHistory, render, renderList,
  requireActiveSegment, sceneAudio, sceneListPane, sceneSlotNumber, segmentImageSource, segmentIndexInfo,
  segmentLayer, segmentTrack, selectedSegmentImagePath, selectedTimelineRangeInfo, setActiveSegment,
  snapAddedSegmentEndToNearestBeat, snapTimeToBeat, state, syncI2VMotionJsonFromSegments, syncInspector,
  syncPreview, syncPromptJsonFromSegments, syncSegmentT2IPrompt, syncTimelineTrimModeButton,
  syncZEnhanceSettingsPanel, timelineDuration, updateHistoryButtons, updateSelectedMediaTools,
}) {
  let frameCaptureInFlight = false;
  let mediaDeleteInFlight = false;
  async function loadCustomImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    customImageFileInput.value = "";
    customImageFileInput.click();
  }

  function previewVideoFrameSegment() {
    const previewPath = String(previewVideo.dataset.path || "");
    if (previewPath) {
      const matched = allEditableSegments().find((segment) => selectedSegmentVideoPath(segment) === previewPath);
      if (matched) return matched;
    }
    const current = currentGlobalTime();
    return playbackSegmentAtTime(current) || activeSegment();
  }

  function frameImageTargetSegments() {
    return allEditableSegments()
      .slice()
      .sort((a, b) => {
        const startDelta = Number(a.start || 0) - Number(b.start || 0);
        if (Math.abs(startDelta) > 0.0001) return startDelta;
        const trackA = segmentTrack(a) === "overlay" ? 0 : 1;
        const trackB = segmentTrack(b) === "overlay" ? 0 : 1;
        return trackA - trackB;
      });
  }

  function chooseFrameImageTargetScene(defaultSegment = activeSegment()) {
    const segments = frameImageTargetSegments();
    if (!segments.length) {
      toast("No scenes found to receive the captured frame.", true);
      return Promise.resolve(null);
    }
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:18px;";
      const box = document.createElement("div");
      box.style.cssText = "width:min(480px,calc(100vw - 36px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);overflow:hidden;";
      const header = document.createElement("div");
      header.style.cssText = "padding:14px 16px;border-bottom:1px solid #1f2937;background:#083f4f;";
      header.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Use Frame as Image</div><div style="font-size:12px;color:#bae6fd;margin-top:4px;">Choose which scene should receive the captured video frame.</div>`;
      const body = document.createElement("div");
      body.style.cssText = "padding:14px 16px;display:flex;flex-direction:column;gap:10px;";
      const defaultTarget = segments.some((segment) => segment.id === defaultSegment?.id) ? defaultSegment : segments[0];
      const sceneSelect = makeSelect(segments.map((segment) => segment.id), defaultTarget?.id || segments[0]?.id || "");
      for (const option of sceneSelect.options) {
        const segment = segments.find((item) => item.id === option.value);
        const info = segmentIndexInfo(segment);
        const prefix = info.track === "overlay" ? `Insert ${info.index + 1}` : `Scene ${info.index + 1}`;
        const timing = `${formatTime(Number(segment?.start || 0))} - ${formatTime(Number(segment?.end || 0))}`;
        option.textContent = `${prefix}: ${segment?.label || prefix} (${timing})`;
      }
      const note = document.createElement("div");
      note.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;";
      note.textContent = "This captures the frame currently visible in the video preview, then saves it as the selected base scene or insert image.";
      body.append(makeField("Save captured frame to", sceneSelect), note);
      const actions = document.createElement("div");
      actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:12px 16px;border-top:1px solid #1f2937;";
      const cancel = makeButton("Cancel");
      const apply = makeButton("Use Frame", "primary");
      actions.append(cancel, apply);
      box.append(header, body, actions);
      backdrop.append(box);
      document.body.append(backdrop);
      const finish = (segment) => {
        backdrop.remove();
        resolve(segment || null);
      };
      cancel.onclick = () => finish(null);
      apply.onclick = () => finish(segments.find((segment) => segment.id === sceneSelect.value) || null);
      backdrop.addEventListener("pointerdown", (event) => {
        if (event.target === backdrop) finish(null);
      });
    });
  }

  async function captureSelectedVideoFrameAsImage(sourceSegment = previewVideoFrameSegment()) {
    if (frameCaptureInFlight) return;
    if (!sourceSegment) {
      toast("Select or scrub to a scene video frame first.", true);
      return;
    }
    const videoPath = selectedSegmentVideoPath(sourceSegment);
    if (!videoPath) {
      toast("The current preview does not have a selected video frame to use.", true);
      return;
    }
    const targetSegment = await chooseFrameImageTargetScene(sourceSegment);
    if (!targetSegment) return;
    frameCaptureInFlight = true;
    let tempVideo = null;
    try {
      const previewHasVideo = previewVideo.dataset.path === videoPath && previewVideo.readyState >= 1;
      const sourceVideo = previewHasVideo ? previewVideo : document.createElement("video");
      if (!previewHasVideo) {
        tempVideo = sourceVideo;
        sourceVideo.muted = true;
        sourceVideo.playsInline = true;
        sourceVideo.preload = "auto";
        sourceVideo.src = makeEditorVideoUrl(videoPath);
      }
      const localTime = previewHasVideo
        ? Number(sourceVideo.currentTime || 0)
        : Math.max(0, currentGlobalTime() - Number(sourceSegment.start || 0));
      await seekVideoForCapture(sourceVideo, localTime);
      const imageData = captureVideoFrameDataUrl(sourceVideo);
      const projectFolder = projectInput.value || state.projectFolder;
      pushHistory();
      if (projectFolder) {
        const sceneNumber = sceneSlotNumber(targetSegment);
        const saved = await postJson("/vrgdg/music_builder/archive_scene_image", {
          image_data: imageData,
          project_folder: projectFolder,
          scene_number: sceneNumber,
        });
        if (!saved.saved_path) throw new Error("The frame was captured, but no saved path was returned.");
        addSceneImageHistoryPath(targetSegment, saved.saved_path);
        targetSegment.approved_image_path = "";
        targetSegment.custom_image_path = "";
        targetSegment.custom_image_data = "";
        targetSegment.custom_image_name = "";
      } else {
        targetSegment.custom_image_data = imageData;
        targetSegment.custom_image_name = "video_frame.png";
        targetSegment.custom_image_path = "";
        targetSegment.approved_image_path = "";
        targetSegment.image = null;
        targetSegment.preview_mode = "image";
      }
      targetSegment.image = null;
      targetSegment.preview_mode = "image";
      setActiveSegment(targetSegment);
      syncPreview(targetSegment);
      render();
      await autoSaveSessionQuiet("video frame saved as scene image");
      toast(`Saved current video frame as image for ${targetSegment.label || "scene"}.`);
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      if (tempVideo) {
        tempVideo.pause();
        tempVideo.removeAttribute("src");
        tempVideo.load?.();
      }
      frameCaptureInFlight = false;
      updateSelectedMediaTools();
    }
  }

  // Scenes after an inserted base scene move down one position, so their numbered files and latents must follow.
  async function renumberScenesAfterInsert(insertedSceneNumber) {
    const projectFolder = String(state.projectFolder || projectInput?.value || "").trim();
    if (!projectFolder) return [];
    const { renamed } = await postJson("/vrgdg/music_builder/renumber_scenes_after_insert", {
      project_folder: projectFolder,
      inserted_scene_number: insertedSceneNumber,
    });
    return renamed;
  }

  function applyRenamedScenePaths(renamed) {
    if (!renamed.length) return;
    rewriteRenamedScenePaths(state.segments, renamed);
    rewriteRenamedScenePaths(state.overlaySegments, renamed);
    if (Array.isArray(state.audioClips)) rewriteRenamedScenePaths(state.audioClips, renamed);
    // Undo would restore the old numbering while the files keep the new names.
    state.undoStack = [];
    state.redoStack = [];
    updateHistoryButtons();
  }

  async function addSegment() {
    const lastSegmentEnd = state.segments.reduce(
      (latest, segment) => Math.max(latest, Number(segment.end || 0)),
      0,
    );
    const playheadTime = Math.max(0, Number(currentGlobalTime() || 0));
    const playheadAppendDuration = playheadTime - lastSegmentEnd;
    if (state.segments.length && playheadAppendDuration >= 0.5) {
      const segmentEnd = snapAddedSegmentEndToNearestBeat(playheadTime, lastSegmentEnd);
      const segmentDuration = segmentEnd - lastSegmentEnd;
      if (segmentDuration > 10 && !await showLongSegmentConfirm(segmentDuration)) return;
      pushHistory();
      const segment = newSegment(lastSegmentEnd, segmentEnd);
      segment.source = state.srtMode ? "inserted" : "manual";
      state.segments.push(segment);
      state.duration = Math.max(Number(state.duration || 0), segmentEnd);
      enforceAudioTimelineEnd();
      sortSegments(state.segments);
      setActiveSegment(segment);
      await syncPromptJsonFromSegments("segment added at playhead");
      await syncI2VMotionJsonFromSegments("segment added at playhead");
      autoSaveSessionQuiet("segment added at playhead");
      return;
    }

    const active = activeSegment();
    let duration = 4;
    let insertIndex = state.segments.length;
    let start = state.segments[state.segments.length - 1]?.end || 0;
    let shiftFromIndex = null;
    if (active) {
      const activeIndex = state.segments.findIndex((segment) => segment.id === active.id);
      if (activeIndex >= 0) {
        const choice = await showAddSegmentPositionModal(active.label || `Scene ${activeIndex + 1}`);
        if (!choice) return;
        if (choice === "before") {
          insertIndex = Math.max(0, activeIndex);
          start = active.start;
          shiftFromIndex = activeIndex;
        } else {
          insertIndex = activeIndex + 1;
          start = active.end;
          if (activeIndex + 1 < state.segments.length) shiftFromIndex = activeIndex + 1;
        }
      }
    }
    const audioEnd = loadedGlobalAudioDuration();
    if (audioEnd > 0) {
      duration = Math.min(duration, Math.max(0, audioEnd - Number(start || 0)));
      if (duration <= 0.001) {
        toast(`The base timeline already ends with the audio at ${formatTime(audioEnd)}.`, true);
        return;
      }
    }
    let renamed = [];
    if (insertIndex < state.segments.length) {
      try {
        renamed = await renumberScenesAfterInsert(insertIndex + 1);
      } catch (error) {
        toast(`Scene was not added: ${error?.message || error}`, true);
        return;
      }
    }
    pushHistory();
    if (shiftFromIndex != null) {
      for (let index = shiftFromIndex; index < state.segments.length; index += 1) {
        shiftSegmentTiming(state.segments[index], duration);
      }
    }
    const end = start + duration;
    const segment = newSegment(start, end);
    segment.source = state.srtMode ? "inserted" : "manual";
    state.segments.splice(insertIndex, 0, segment);
    applyRenamedScenePaths(renamed);
    state.duration = Math.max(Number(state.duration || 0), end, ...state.segments.map((item) => Number(item.end || 0)));
    enforceAudioTimelineEnd();
    sortSegments(state.segments);
    setActiveSegment(segment);
    await syncPromptJsonFromSegments("segment added");
    await syncI2VMotionJsonFromSegments("segment added");
    autoSaveSessionQuiet("segment added");
    if (renamed.length) {
      loadDirtyLatentBadges();
      toast("Later scene files were renumbered to match; undo history was cleared.");
    }
  }

  async function splitActiveSceneAtPlayhead() {
    const segment = activeSegment();
    if (!segment) {
      toast("Select a base timeline scene to split first.", true);
      return;
    }
    if (segmentTrack(segment) === "overlay") {
      toast("The scissors currently split base timeline scenes only.", true);
      return;
    }
    if (String(selectedSegmentVideoPath(segment) || "").trim() || (Array.isArray(segment.video_history) && segment.video_history.length)) {
      if (baseSceneVideoTrimKind(segment)) {
        await chooseRenderedSceneTrimAtPlayhead();
        return;
      }
      toast("This rendered scene cannot be split. MiniMax built-in-audio and ID-LoRA rendered scenes can be trimmed at the playhead instead.", true);
      return;
    }
    const index = state.segments.indexOf(segment);
    if (index < 0) {
      toast("The selected scene could not be found in the base timeline.", true);
      return;
    }
    const start = Number(segment.start || 0);
    const end = Number(segment.end || start);
    const splitTime = Math.max(start, Math.min(end, Number(currentGlobalTime() || 0)));
    if (splitTime - start < 0.05 || end - splitTime < 0.05) {
      toast("Move the playhead inside the selected scene, away from its edges, then click ✂ again.", true);
      return;
    }

    let renamed = [];
    if (index + 1 < state.segments.length) {
      try {
        renamed = await renumberScenesAfterInsert(index + 2);
      } catch (error) {
        toast(`Scene was not split: ${error?.message || error}`, true);
        return;
      }
    }
    pushHistory();
    const originalLabel = String(segment.label || `Scene ${index + 1}`);
    const originalLyric = String(segment.lyric_text || "");
    const right = typeof structuredClone === "function"
      ? structuredClone(segment)
      : JSON.parse(JSON.stringify(segment));
    right.id = createUniqueSegmentId();
    right.start = Number(splitTime.toFixed(3));
    right.end = end;
    segment.end = Number(splitTime.toFixed(3));

    if (segment.custom_audio_path) {
      const originalAudioStart = Number(segment.custom_audio_timeline_start ?? start);
      const originalSourceStart = Number(segment.custom_audio_source_start || 0);
      const leftDuration = Math.max(0, splitTime - originalAudioStart);
      const rightDuration = Math.max(0, Number(segment.custom_audio_duration || end - start) - leftDuration);
      segment.custom_audio_duration = leftDuration;
      right.custom_audio_timeline_start = splitTime;
      right.custom_audio_source_start = originalSourceStart + leftDuration;
      right.custom_audio_duration = rightDuration;
    }

    if (!isInstrumentalLyricText(originalLyric)) {
      right.lyric_text = "";
      right.lyric_no_lip_sync = false;
      right.timeline_note = [
        String(right.timeline_note || "").trim(),
        `Split from ${originalLabel} at ${formatTime(splitTime)}. Add the lyric/dialogue for this new right-hand scene.`,
      ].filter(Boolean).join("\n");
    }

    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    for (const key of ["subject_scene_map", "scene_map", "scene_trigger_map", "ingredients_scene_map"]) {
      const map = refs[key];
      if (map && typeof map === "object" && Object.prototype.hasOwnProperty.call(map, segment.id)) {
        const value = map[segment.id];
        map[right.id] = Array.isArray(value) ? [...value] : value && typeof value === "object" ? { ...value } : value;
      }
    }
    state.fluxReferenceBuilder = refs;
    state.segments.splice(index + 1, 0, ensureSegmentRuntimeFields(right));
    state.segments.forEach((item, itemIndex) => {
      if (/^scene\s+\d+$/i.test(String(item.label || "").trim())) item.label = `Scene ${itemIndex + 1}`;
    });
    applyRenamedScenePaths(renamed);
    state.activeId = right.id;
    state.activeTrack = "base";
    state.duration = timelineDuration();
    syncInspector();
    render();
    loadDirtyLatentBadges();
    await syncPromptJsonFromSegments("scene split at playhead");
    await syncI2VMotionJsonFromSegments("scene split at playhead");
    await autoSaveSessionQuiet("scene split at playhead");
    toast(renamed.length
      ? `Split ${originalLabel} at ${formatTime(splitTime)}. Later scene files were renumbered to match; undo history was cleared.`
      : `Split ${originalLabel} at ${formatTime(splitTime)}. Later scene timing did not move.`);
  }

  function setTimelineRangePoint(which) {
    const time = Math.max(0, currentGlobalTime());
    const range = normalizeTimelineRange(state.selectedTimelineRange);
    pushHistory();
    if (which === "in") range.in = time;
    else range.out = time;
    state.selectedTimelineRange = normalizeTimelineRange(range);
    render();
    autoSaveSessionQuiet(`timeline range ${which}`).catch(() => null);
  }

  function clearSelectedTimelineRange() {
    pushHistory();
    state.selectedTimelineRange = { in: null, out: null };
    render();
    autoSaveSessionQuiet("timeline range cleared").catch(() => null);
  }

  function addTimelineMarkerFromSelection() {
    const range = selectedTimelineRangeInfo();
    const start = currentGlobalTime();
    const freeRange = nextFreeTimelineMarkerRange(start, 4);
    const marker = range
      ? newTimelineMarker(range.start, range.end)
      : newTimelineMarker(freeRange.start, freeRange.end);
    marker.label = range ? "Selected range" : "Timeline note";
    marker.note = range ? `Range ${formatTime(range.start)} - ${formatTime(range.end)}` : "";
    if (range) {
      openTimelineMarkerEditor(marker);
      return;
    }
    pushHistory();
    state.timelineMarkers.push(marker);
    state.timelineMarkers = normalizeTimelineMarkers(state.timelineMarkers);
    state.activeTimelineMarkerId = marker.id;
    render();
    autoSaveSessionQuiet("timeline note added").catch(() => null);
  }

  async function applyBulkSegmentTimings(timings, action) {
    const replacing = action === "replace";
    const newSegments = timings.map((timing, index) => {
      const segment = newSegment(Number(timing.start || 0), Number(timing.end || 0));
      segment.label = `Scene ${replacing ? index + 1 : state.segments.length + index + 1}`;
      segment.source = "manual";
      return segment;
    });
    pushHistory();
    if (replacing) {
      state.segments = newSegments;
      state.srtMode = false;
      state.timingFrozen = false;
      state.activeId = newSegments[0]?.id || "";
    } else {
      state.segments.push(...newSegments);
      state.activeId = newSegments[0]?.id || state.activeId;
    }
    sortSegments(state.segments);
    state.duration = Math.max(
      Number(state.duration || 0),
      ...state.segments.map((segment) => Number(segment.end || 0)),
      ...state.overlaySegments.map((segment) => Number(segment.end || 0)),
    );
    state.activeTrack = "base";
    syncInspector();
    render();
    await syncPromptJsonFromSegments("bulk segments");
    await syncI2VMotionJsonFromSegments("bulk segments");
    await autoSaveSessionQuiet("bulk segments");
  }

  function openBulkSegmentsModal(options = {}) {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(760px,calc(100vw - 40px));max-height:calc(100vh - 60px);border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);display:flex;flex-direction:column;overflow:hidden;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 16px;border-bottom:1px solid #155e75;background:#083f4f;";
    const heading = document.createElement("div");
    heading.textContent = "Bulk Manual Segments";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const close = makeButton("Close");
    header.append(heading, close);
    const body = document.createElement("div");
    body.style.cssText = "padding:14px 16px;display:flex;flex-direction:column;gap:12px;overflow:auto;";
    const explanation = document.createElement("div");
    explanation.innerHTML = `
      <div><strong style="color:#e0f2fe;">What this does</strong></div>
      <div>Create many manual base timeline scenes from pasted timing text instead of clicking + Segment over and over.</div>
      <div style="margin-top:6px;"><strong style="color:#fecaca;">Replace current base timeline</strong> rebuilds the base scenes and clears generated scene outputs from those new scenes. Inserts stay in the insert track.</div>
      <div><strong style="color:#bbf7d0;">Append after last scene</strong> adds duration-based scenes after the current last base scene.</div>
    `;
    explanation.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;color:#d4d4d8;font-size:12px;line-height:1.45;";
    const modeSelect = makeSelect(["fixed", "fit", "durations", "ranges", "markers"], options.initialMode || "fixed");
    modeSelect.options[0].textContent = "Fixed length + scene count";
    modeSelect.options[1].textContent = "Fit to song duration";
    modeSelect.options[2].textContent = "Durations";
    modeSelect.options[3].textContent = "Start - End rows";
    modeSelect.options[4].textContent = "Timestamp markers";
    const actionSelect = makeSelect(["replace", "append"], "replace");
    actionSelect.options[0].textContent = "Replace current base timeline";
    actionSelect.options[1].textContent = "Append after last scene";
    const controls = document.createElement("div");
    controls.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    controls.append(makeField("Input format", modeSelect), makeField("Apply mode", actionSelect));
    const examples = document.createElement("div");
    examples.style.cssText = "border:1px solid #334155;border-radius:7px;background:#111827;padding:10px;color:#cbd5e1;font-size:12px;line-height:1.45;";
    const textarea = document.createElement("textarea");
    textarea.style.cssText = "min-height:220px;resize:vertical;border:1px solid #3f3f46;border-radius:7px;background:#09090b;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;";
    const fixedControls = document.createElement("div");
    fixedControls.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const fixedDuration = document.createElement("input");
    fixedDuration.type = "number";
    fixedDuration.min = "0.1";
    fixedDuration.step = "0.1";
    fixedDuration.value = "8";
    const fixedCount = document.createElement("input");
    fixedCount.type = "number";
    fixedCount.min = "1";
    fixedCount.step = "1";
    fixedCount.value = "24";
    fixedControls.append(makeField("Scene length (seconds)", fixedDuration), makeField("Scene count", fixedCount));
    const fixedTimings = (start) => {
      const duration = Number(fixedDuration.value);
      const count = Number(fixedCount.value);
      if (!Number.isFinite(duration) || duration <= 0) throw new Error("Scene length must be greater than zero.");
      if (!Number.isInteger(count) || count < 1) throw new Error("Scene count must be a whole number of at least 1.");
      return Array.from({ length: count }, (_, index) => ({ start: start + index * duration, end: start + (index + 1) * duration }));
    };
    const fitTimings = () => {
      const duration = Number(fixedDuration.value);
      const songDuration = Number(audio.duration || state.duration || 0);
      if (!Number.isFinite(duration) || duration <= 0) throw new Error("Scene length must be greater than zero.");
      if (!Number.isFinite(songDuration) || songDuration <= 0) throw new Error("Load a song first so its duration is available.");
      const count = Math.max(1, Math.ceil(songDuration / duration));
      return Array.from({ length: count }, (_, index) => ({
        start: index * duration,
        end: index === count - 1 ? songDuration : (index + 1) * duration,
      }));
    };
    const setExample = () => {
      fixedControls.style.display = ["fixed", "fit"].includes(modeSelect.value) ? "grid" : "none";
      fixedControls.style.gridTemplateColumns = modeSelect.value === "fit" ? "1fr" : "1fr 1fr";
      fixedCount.parentElement.style.display = modeSelect.value === "fit" ? "none" : "block";
      textarea.style.display = ["fixed", "fit"].includes(modeSelect.value) ? "none" : "block";
      if (modeSelect.value === "fixed") {
        examples.innerHTML = `<strong style="color:#e0f2fe;">Fixed length + scene count</strong><br>Create the requested number of consecutive scenes, all with the same duration.`;
        actionSelect.disabled = false;
      } else if (modeSelect.value === "fit") {
        examples.innerHTML = `<strong style="color:#e0f2fe;">Fit to song duration</strong><br>Fill the entire loaded song using the preferred scene length. Any remainder becomes its own shorter final scene so generated clips never lose the ending.`;
        actionSelect.value = "replace";
        actionSelect.disabled = true;
      } else if (modeSelect.value === "durations") {
        examples.innerHTML = `<strong style="color:#e0f2fe;">Durations</strong><br>One scene duration per line. Values can be seconds, <code>mm:ss</code>, or <code>hh:mm:ss</code>.`;
        if (!textarea.value.trim()) textarea.value = "4\n6.5\n3\n8\n5";
        actionSelect.disabled = false;
      } else if (modeSelect.value === "ranges") {
        examples.innerHTML = `<strong style="color:#e0f2fe;">Start - End rows</strong><br>One exact scene range per line, like <code>0 - 4</code> or <code>00:04.00 - 00:08.50</code>. These are absolute timeline times.`;
        if (!textarea.value.trim()) textarea.value = "0 - 4\n4 - 8.5\n8.5 - 13\n13 - 20";
        actionSelect.value = "replace";
        actionSelect.disabled = true;
      } else {
        examples.innerHTML = `<strong style="color:#e0f2fe;">Timestamp markers</strong><br>One marker per line. Scenes are created between each neighboring pair. Example: 0, 4, 8.5 creates two scenes.`;
        if (!textarea.value.trim()) textarea.value = "0\n4\n8.5\n13\n20";
        actionSelect.value = "replace";
        actionSelect.disabled = true;
      }
    };
    const preview = document.createElement("div");
    preview.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;color:#a5f3fc;font-size:12px;white-space:pre-wrap;min-height:42px;";
    const updatePreview = () => {
      try {
        const appendStart = actionSelect.value === "append" ? Number(state.segments[state.segments.length - 1]?.end || 0) : 0;
        const timings = modeSelect.value === "fixed" ? fixedTimings(appendStart) : modeSelect.value === "fit" ? fitTimings() : parseBulkSegmentTimings(textarea.value, modeSelect.value, appendStart);
        const first = timings[0];
        const last = timings[timings.length - 1];
        preview.textContent = `Ready: ${timings.length} scene${timings.length === 1 ? "" : "s"} | ${formatTime(first.start)} - ${formatTime(last.end)} | total ${(last.end - first.start).toFixed(2)}s`;
      } catch (error) {
        preview.textContent = `Preview: ${String(error?.message || error)}`;
      }
    };
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:12px 16px;border-top:1px solid #1f2937;";
    const cancel = makeButton("Cancel");
    const apply = makeButton("Apply Bulk Segments", "primary");
    actions.append(cancel, apply);
    body.append(explanation, controls, examples, fixedControls, textarea, preview);
    box.append(header, body, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    const closeModal = () => { backdrop.remove(); options.onClose?.(); };
    close.onclick = closeModal;
    cancel.onclick = closeModal;
    modeSelect.onchange = () => {
      textarea.value = "";
      setExample();
      updatePreview();
    };
    actionSelect.onchange = updatePreview;
    textarea.addEventListener("input", updatePreview);
    fixedDuration.addEventListener("input", updatePreview);
    fixedCount.addEventListener("input", updatePreview);
    apply.onclick = async () => {
      try {
        const appendStart = actionSelect.value === "append" ? Number(state.segments[state.segments.length - 1]?.end || 0) : 0;
        const timings = modeSelect.value === "fixed" ? fixedTimings(appendStart) : modeSelect.value === "fit" ? fitTimings() : parseBulkSegmentTimings(textarea.value, modeSelect.value, appendStart);
        await applyBulkSegmentTimings(timings, actionSelect.value);
        toast(`Created ${timings.length} manual scene${timings.length === 1 ? "" : "s"}.`);
        closeModal();
      } catch (error) {
        toast(String(error?.message || error), true);
        updatePreview();
      }
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) closeModal();
    });
    setExample();
    updatePreview();
  }

  async function addOverlaySegment() {
    if (!state.overlayTrack.enabled) {
      toast("Turn on Overlay Track before adding an overlay clip.", true);
      return;
    }
    const duration = 4;
    const start = Math.max(0, snapTimeToBeat(currentGlobalTime()));
    const end = Math.min(Math.max(start + duration, start + 0.1), Math.max(start + duration, timelineDuration() || start + duration));
    const segment = newSegment(start, end);
    segment.track = "overlay";
    segment.source = "overlay";
    segment.overlay_slot_number = nextOverlaySlotNumber();
    segment.label = `Insert ${state.overlaySegments.length + 1}`;
    normalizeOverlayClip(segment);
    pushHistory();
    state.overlaySegments.push(segment);
    sortSegments(state.overlaySegments);
    state.duration = Math.max(Number(state.duration || 0), end);
    setActiveSegment(segment);
    await autoSaveSessionQuiet("insert segment added");
  }

  function syncOverlayTrackControls() {
    const enabled = Boolean(state.overlayTrack?.enabled);
    overlayTrackToggleButton.textContent = `Overlay Track: ${enabled ? "On" : "Off"}`;
    overlayTrackToggleButton.style.background = enabled ? "#0e7490" : "#27272a";
    overlayTrackToggleButton.style.borderColor = enabled ? "#22d3ee" : "#3f3f46";
    addOverlaySegmentButton.style.display = enabled ? "" : "none";
  }

  async function toggleOverlayTrack() {
    pushHistory();
    state.overlayTrack = normalizeOverlayTrackState({ enabled: !state.overlayTrack?.enabled });
    syncOverlayTrackControls();
    render();
    await autoSaveSessionQuiet("overlay track toggled");
    toast(state.overlayTrack.enabled
      ? "Overlay Track enabled. Existing overlays will be used during playback and stitching."
      : "Overlay Track disabled. Overlay clips are preserved but ignored during playback and stitching.");
  }

  // A scene latent only matches the render that produced it, so it is removed with that video.
  // Never reindex here: the scene keeps its slot, only its stale latent goes.
  async function deleteStaleSceneLatents(segment = null) {
    const projectFolder = String(state.projectFolder || projectInput?.value || "").trim();
    if (!projectFolder) return;
    const payload = { project_folder: projectFolder };
    if (segment) {
      if (segmentTrack(segment) === "overlay") return;
      const slotNumber = sceneSlotNumber(segment);
      if (!slotNumber) return;
      payload.scene_number = slotNumber;
      payload.reindex = false;
    } else {
      payload.all = true;
    }
    const resp = await postJson("/vrgdg/music_builder/delete_scene_latent", payload, 10000).catch((error) => {
      console.warn("[VRGDG] Could not delete stale scene latent:", error);
      return null;
    });
    loadDirtyLatentBadges();
    return resp;
  }

  async function deleteAllSegments() {
    const baseCount = Array.isArray(state.segments) ? state.segments.length : 0;
    const overlayCount = Array.isArray(state.overlaySegments) ? state.overlaySegments.length : 0;
    const total = baseCount + overlayCount;
    if (!total) {
      toast("No segments to delete.", true);
      return;
    }
    const { confirmed } = await confirmDestructiveAction({
      title: `Delete ALL ${total} segment${total === 1 ? "" : "s"}?`,
      message: [
        "You are about to delete every segment on the timeline.",
        "The scenes and their timing are removed from this project. Rendered image and video files stay in the project folder. You can bring the segments back with Undo.",
      ],
      details: [
        `${baseCount} base segment${baseCount === 1 ? "" : "s"}`,
        `${overlayCount} insert/overlay segment${overlayCount === 1 ? "" : "s"}`,
      ],
    });
    if (!confirmed) return;
    pauseTimelineForEditing();
    pushHistory();
    state.segments = [];
    state.overlaySegments = [];
    state.selectedSegmentIds = [];
    state.activeId = "";
    state.activeTrack = "base";
    state.srtMode = false;
    state.timingFrozen = false;
    state.timelineTrimEditMode = false;
    state.selectedTimelineRange = { in: null, out: null };
    state.sceneAudioSegmentId = "";
    state.sceneAudioGlobalTime = 0;
    sceneAudio.removeAttribute("src");
    sceneAudio.load();
    freezeTimingControl.input.checked = false;
    // Clear stale visual blocks immediately. The normal render below rebuilds
    // labels/overlays, but deleted scene nodes must never survive a render error.
    segmentLayer.textContent = "";
    sceneListPane.textContent = "";
    syncTimelineTrimModeButton();
    syncInspector();
    render();
    await syncPromptJsonFromSegments("all segments deleted");
    await syncI2VMotionJsonFromSegments("all segments deleted");
    await autoSaveSessionQuiet("all segments deleted");
    toast(`Deleted ${total} segment${total === 1 ? "" : "s"}.`);
  }

  async function deleteSelectedMedia({ segment = activeSegment(), type = segment?.preview_mode === "video" ? "video" : "image" } = {}) {
    if (mediaDeleteInFlight) return;
    if (!segment) {
      toast("No selected image or video to delete.", true);
      return;
    }
    const path = type === "video" ? selectedSegmentVideoPath(segment) : selectedSegmentImagePath(segment);
    const inMemoryImage = type === "image" && !path && Boolean(segment.custom_image_data || segment.image);
    if (!path && !inMemoryImage) {
      toast("No selected image or video to delete.", true);
      return;
    }
    const media = { segment, type, path };
    const mediaThumbnailPath = type === "video" ? selectedSegmentVideoThumbnailPath(segment) : "";
    const ok = inMemoryImage
      ? (await confirmDestructiveAction({
        title: "Delete selected image?",
        message: "Remove this unsaved image from the scene?",
      })).confirmed
      : await confirmDeleteMediaAction(type, path);
    if (!ok) return;
    mediaDeleteInFlight = true;
    try {
      if (path) await postJson("/vrgdg/music_builder/delete_project_media", {
        project_folder: projectInput.value,
        path,
      });
      if (mediaThumbnailPath) {
        await postJson("/vrgdg/music_builder/delete_project_media", {
          project_folder: projectInput.value,
          path: mediaThumbnailPath,
        }).catch(() => null);
      }
      pushHistory();
      if (media.type === "video") {
        const removedIndex = (media.segment.video_history || []).findIndex((item) => mediaPathKey(item) === mediaPathKey(media.path));
        media.segment.video_history = (media.segment.video_history || []).filter((item) => mediaPathKey(item) !== mediaPathKey(media.path));
        if (removedIndex >= 0 && Array.isArray(media.segment.video_thumbnail_history)) {
          media.segment.video_thumbnail_history.splice(removedIndex, 1);
        }
        media.segment.video_backup_paths = (media.segment.video_backup_paths || []).filter((item) => mediaPathKey(item) !== mediaPathKey(media.path));
        if (mediaThumbnailPath) {
          media.segment.video_backup_thumbnail_paths = (media.segment.video_backup_thumbnail_paths || []).filter((item) => mediaPathKey(item) !== mediaPathKey(mediaThumbnailPath));
        }
        media.segment.video_history_index = Math.min(Math.max(0, Number(media.segment.video_history_index || 0)), media.segment.video_history.length - 1);
        if (media.segment.video_history_index < 0) media.segment.video_history_index = -1;
        if (media.segment.video_path === media.path) {
          media.segment.video_path = media.segment.video_history[media.segment.video_history_index] || "";
          media.segment.video_thumbnail_path = media.segment.video_thumbnail_history?.[media.segment.video_history_index] || "";
          media.segment.video_source_path = "";
          media.segment.video_output = null;
        }
        if (!media.segment.video_path) media.segment.video_thumbnail_path = "";
        if (!media.segment.video_path) media.segment.video_status = "none";
        normalizeSegmentVideoHistory(media.segment);
        media.segment.preview_mode = media.segment.image_history?.length ? "image" : "video";
      } else {
        media.segment.image_history = (media.segment.image_history || [])
          .filter((item) => mediaPathKey(item) !== mediaPathKey(media.path));
        media.segment.image_history_index = media.segment.image_history.length - 1;
        if (mediaPathKey(media.segment.approved_image_path) === mediaPathKey(media.path)) media.segment.approved_image_path = "";
        if (inMemoryImage || mediaPathKey(media.segment.custom_image_path) === mediaPathKey(media.path)) {
          media.segment.custom_image_path = "";
          media.segment.custom_image_data = "";
          media.segment.custom_image_name = "";
        }
        media.segment.image_assignment_cleared = !segmentImageSource(media.segment);
        if (media.segment.image_assignment_cleared) media.segment.image = null;
        media.segment.preview_mode = !media.segment.image_assignment_cleared || !selectedSegmentVideoPath(media.segment) ? "image" : "video";
      }
      ensureSegmentRuntimeFields(media.segment);
      // Deleting any video of this scene makes its saved latent stale, so it goes with it.
      let latentRemoved = false;
      if (media.type === "video") {
        const latentResp = await deleteStaleSceneLatents(media.segment);
        latentRemoved = Boolean(latentResp?.deleted);
        media.segment._latentDirty = false;
      }
      syncPreview(media.segment);
      syncInspector();
      renderList();
      render();
      await autoSaveSessionQuiet(`${media.type} deleted`);
      toast(`Deleted ${media.type} from project.${latentRemoved ? " Its saved latent was removed too." : ""}`);
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      mediaDeleteInFlight = false;
      updateSelectedMediaTools();
    }
  }

  function sendPromptToEnhance(sourceLabel, promptText) {
    const segment = requireActiveSegment();
    if (!segment) return;
    const prompt = String(promptText || "").trim();
    if (!prompt) {
      toast(`${sourceLabel} prompt is empty.`, true);
      return;
    }
    pushHistory();
    syncSegmentT2IPrompt(segment, prompt);
    syncZEnhanceSettingsPanel();
    render();
    toast(`${sourceLabel} prompt set as this scene's image/Enhance prompt.`);
  }

  async function deleteSegment() {
    const segment = activeSegment();
    if (!segment) return;
    const isBase = segmentTrack(segment) !== "overlay";
    const slotNumber = isBase ? sceneSlotNumber(segment) : null;
    const segmentName = String(segment.label || "").trim() || (isBase ? `Scene ${slotNumber}` : "Insert clip");
    const hasMedia = Boolean(selectedSegmentVideoPath(segment)) || Boolean(segment.image_history?.length) || Boolean(segment.image);
    const { confirmed } = await confirmDestructiveAction({
      title: isBase ? "Delete this scene?" : "Delete this insert clip?",
      message: isBase ? [
        "You are about to delete this scene from the timeline.",
        "Its image and video files are moved to the project's removed_scene_assets folder, and the files of every later scene are renumbered to match. Undo history is cleared when files are renumbered.",
      ] : [
        "You are about to delete this insert clip from the timeline.",
      ],
      details: [
        `${segmentName} (${formatTime(Number(segment.start || 0))} to ${formatTime(Number(segment.end || 0))})`,
        ...(hasMedia ? ["This scene has generated media."] : []),
      ],
    });
    if (!confirmed) return;
    const projectFolder = String(state.projectFolder || projectInput?.value || "").trim();
    // Later scenes move up one position, so their numbered files and latents must follow.
    let renamed = [];
    if (slotNumber && projectFolder) {
      try {
        ({ renamed } = await postJson("/vrgdg/music_builder/renumber_scenes_after_removal", {
          project_folder: projectFolder,
          removed_scene_number: slotNumber,
        }));
      } catch (error) {
        toast(`Scene was not deleted: ${error?.message || error}`, true);
        return;
      }
    }
    pushHistory();
    if (segmentTrack(segment) === "overlay") {
      state.overlaySegments = state.overlaySegments.filter((item) => item.id !== segment.id);
      state.activeId = state.overlaySegments[0]?.id || state.segments[0]?.id || "";
    } else {
      const removedStart = Number(segment.start || 0);
      const removedEnd = Math.max(removedStart, Number(segment.end || removedStart));
      state.segments = state.segments.filter((item) => item.id !== segment.id);
      if (renamed.length) {
        rewriteRenamedScenePaths(state.segments, renamed);
        rewriteRenamedScenePaths(state.overlaySegments, renamed);
        if (Array.isArray(state.audioClips)) rewriteRenamedScenePaths(state.audioClips, renamed);
        // Undo would restore the old numbering while the files keep the new names.
        state.undoStack = [];
        state.redoStack = [];
        updateHistoryButtons();
        toast("Later scene files were renumbered to match; undo history was cleared.");
      }
      const removedDuration = closeBaseTimelineGap(removedStart, removedEnd);
      renumberGenericBaseSceneLabels(state.segments);
      const next = state.segments.find((item) => Number(item.start || 0) >= removedStart - 0.001) || state.segments[state.segments.length - 1] || null;
      state.activeId = next?.id || state.overlaySegments[0]?.id || "";
      if (removedDuration > 0.001) state.sceneAudioGlobalTime = Math.max(0, Math.min(currentGlobalTime(), removedStart));
    }
    state.activeTrack = segmentTrack(activeSegment());
    syncInspector();
    render();
    loadDirtyLatentBadges();
    await syncPromptJsonFromSegments("segment deleted");
    await syncI2VMotionJsonFromSegments("segment deleted");
    autoSaveSessionQuiet("segment deleted");
  }

  return {
    addOverlaySegment, addSegment, addTimelineMarkerFromSelection, applyBulkSegmentTimings,
    captureSelectedVideoFrameAsImage, clearSelectedTimelineRange, deleteAllSegments, deleteSegment,
    deleteSelectedMedia, deleteStaleSceneLatents, loadCustomImage, openBulkSegmentsModal, sendPromptToEnhance,
    setTimelineRangePoint, splitActiveSceneAtPlayhead, syncOverlayTrackControls, toggleOverlayTrack,
  };
}
