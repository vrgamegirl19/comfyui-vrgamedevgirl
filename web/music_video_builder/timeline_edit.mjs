import { normalizeOverlayClip, normalizeOverlayTrackState } from "../VRGDG_OverlayTrack.js";
import { postJson } from "./comfy_api.mjs";
import { makeButton, makeField, makeInput, normalizeProjectVideoEngine, toast } from "./controls.mjs";
import { formatTime } from "./format.mjs";
import { mergeTimestampedLyricText } from "./lyric_transcription.mjs";
import { audioFileFromDrop } from "./media_import.mjs";
import { isMiniMaxH3LatentContinuationMode } from "./minimax_h3.mjs";
import { chooseBatchModeAction } from "./project_actions.mjs";
import { pickPath } from "./project_setup.mjs";
import { isInstrumentalLyricText } from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder, normalizeIdLoraReferenceBuilder } from "./reference_data.mjs";
import { newSegment, newTimelineMarker, renumberGenericBaseSceneLabels, rewriteRenamedScenePaths, sortSegments } from "./segments.mjs";
import { hasLockedVideo, selectedSegmentVideoPath } from "./selection_preview.mjs";
import {
  activateSegmentVideoPath,
  audioChunkDuration,
  audioSourceStart,
  audioTimelineStart,
  mediaPathKey,
  normalizeTimelineMarkers,
  timelineSegmentDuration,
} from "./timeline_state.mjs";
import { shiftSegmentTiming } from "./timeline_view.mjs";

function clearSegmentAudio(segment) {
  segment.custom_audio_path = "";
  segment.custom_audio_name = "";
  segment.custom_audio_duration = 0;
  segment.custom_audio_full_duration = 0;
  segment.custom_audio_timeline_start = Number(segment.start || 0);
  segment.custom_audio_source_start = 0;
  segment.custom_audio_peaks = [];
  segment.custom_audio_beats = [];
}

function splitAudioPeaks(peaks, ratio) {
  const values = Array.isArray(peaks) ? peaks : [];
  const index = Math.max(1, Math.min(values.length - 1, Math.round(values.length * ratio)));
  return [values.slice(0, index), values.slice(index)];
}

function splitAudioBeats(beats, cutLocal, totalDuration) {
  const values = Array.isArray(beats) ? beats : [];
  const left = [];
  const right = [];
  for (const beat of values) {
    const value = Number(beat || 0);
    if (value < cutLocal) left.push(value);
    else if (value <= totalDuration) right.push(Math.max(0, value - cutLocal));
  }
  return [left, right];
}

function mergeUniqueSceneText(firstValue, secondValue, separator = "\n\n") {
  const values = [firstValue, secondValue]
    .map((value) => String(value || "").trim())
    .filter(Boolean);
  if (!values.length) return "";
  if (values.length === 1 || values[0] === values[1]) return values[0];
  return values.join(separator);
}

function mergeUniqueStringArray(firstValue, secondValue) {
  return Array.from(new Set(
    [...(Array.isArray(firstValue) ? firstValue : []), ...(Array.isArray(secondValue) ? secondValue : [])]
      .map((value) => String(value || "").trim())
      .filter(Boolean),
  ));
}

export function createTimelineEdit({
  activeSegment, allEditableSegments, autoSaveSessionQuiet, clampTimelineMarkerToNonOverlap,
  collectedSceneVideoFolder, createProgressWindow, currentGlobalTime, currentVideoMode, deleteSegment,
  loadedGlobalAudioDuration, miniMaxH3ContinuityModeForSegment, miniMaxH3SettingsForSegment,
  nextOverlaySlotNumber, normalizeSegments, openLyricReviewModal, openStoryboardBuilderFromProject,
  pauseTimelineForEditing, projectInput, pushHistory, reloadBeatMarkersFromAudio, render, renderSegments,
  sceneDisplayName, sceneSlotNumber, segmentIndexInfo, segmentTrack, setActiveSegment, setBeatMarkersVisible,
  setGlobalPlaybackTime, state, syncI2VMotionJsonFromSegments, syncInspector, syncPreview,
  syncPromptJsonFromSegments, updateHistoryButtons,
}) {
  async function loadDirtyLatentBadges() {
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    if (!projectFolder) return;
    try {
      const resp = await postJson("/vrgdg/music_builder/list_dirty_latents", {
        project_folder: projectFolder,
      }, 5000);
      if (!resp?.ok || !Array.isArray(resp.dirty_scenes)) return;
      const dirtySet = new Set(resp.dirty_scenes);
      let changed = false;
      state.segments.forEach((seg) => {
        const slot = sceneSlotNumber(seg);
        const wasDirty = Boolean(seg._latentDirty);
        // Only scenes that continue from the previous scene's latent care whether it changed.
        const usesLatentContinuation = isMiniMaxH3LatentContinuationMode(miniMaxH3ContinuityModeForSegment(seg));
        const isDirty = dirtySet.has(slot) && usesLatentContinuation;
        if (wasDirty !== isDirty) {
          seg._latentDirty = isDirty;
          changed = true;
        }
      });
      if (changed) renderSegments();
    } catch (e) {
      // Quietly ignore background poll failures
    }
  }

  function openSceneOptions(segment) {
    if (!segment) return;
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:12px;display:flex;flex-direction:column;gap:10px;";
    const title = document.createElement("div");
    title.textContent = `${segment.label || "Scene"} options`;
    title.style.cssText = "font-size:14px;font-weight:900;color:#cffafe;";
    const customAudioStatus = document.createElement("div");
    customAudioStatus.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#d4d4d8;padding:9px;font-size:12px;overflow-wrap:anywhere;";
    const updateCustomAudioStatus = () => {
      customAudioStatus.textContent = segment.custom_audio_path
        ? `Custom audio: ${segment.custom_audio_name || segment.custom_audio_path}`
        : "No custom scene audio selected.";
    };
    updateCustomAudioStatus();
    const audioDrop = document.createElement("div");
    audioDrop.textContent = "Drag audio file here";
    audioDrop.style.cssText = "border:1px dashed #38bdf8;border-radius:6px;background:#082f49;color:#e0f2fe;padding:16px;text-align:center;font-size:12px;font-weight:900;";
    const sceneAudioFileInput = document.createElement("input");
    sceneAudioFileInput.type = "file";
    sceneAudioFileInput.accept = "audio/wav,audio/mpeg,audio/flac,audio/mp4,audio/ogg,.wav,.mp3,.flac,.m4a,.ogg";
    sceneAudioFileInput.style.display = "none";
    box.append(sceneAudioFileInput);
    const pickCustomAudio = makeButton("Load Audio File");
    const clearCustomAudio = makeButton("Clear");
    const saveOptions = makeButton("Save", "primary");
    const closeOptions = makeButton("Close");
    const sceneSilenceDurationInput = makeInput(String(Math.max(0.1, timelineSegmentDuration(segment) || 4).toFixed(2)), "number");
    sceneSilenceDurationInput.min = "0.1";
    sceneSilenceDurationInput.step = "0.1";
    const createSceneSilence = makeButton("Use Silence For This Scene", "primary");
    const sceneSilencePanel = document.createElement("div");
    sceneSilencePanel.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end;";
    sceneSilencePanel.append(makeField("Silence duration seconds", sceneSilenceDurationInput), createSceneSilence);
    const note = document.createElement("div");
    note.textContent = "Drop or load an audio file for this scene, or create silence. It will be copied into the project folder, sent to LTX for this scene, and used for final stitching when scene-audio mode is active.";
    note.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;";
    actions.append(pickCustomAudio, clearCustomAudio, saveOptions, closeOptions);
    box.append(title, customAudioStatus, audioDrop, sceneSilencePanel, note, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    const saveAudioFile = (file) => {
      if (!file) return;
      const projectFolder = projectInput.value || state.projectFolder;
      if (!projectFolder) {
        toast("Set a project folder first so the scene audio can be copied there.", true);
        return;
      }
      const sceneNumber = sceneSlotNumber(segment);
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const data = await postJson("/vrgdg/music_builder/save_scene_audio", {
            project_folder: projectFolder,
            scene_number: sceneNumber,
            audio_data: String(reader.result || ""),
            audio_name: file.name || "scene_audio.wav",
          }, 180000);
          pushHistory();
          segment.custom_audio_path = data.saved_path || "";
          segment.custom_audio_name = file.name || "";
          segment.custom_audio_duration = Number(data.duration || 0);
          segment.custom_audio_full_duration = Number(data.duration || 0);
          segment.custom_audio_timeline_start = Number(segment.start || 0);
          segment.custom_audio_source_start = 0;
          segment.custom_audio_peaks = Array.isArray(data.peaks) ? data.peaks : [];
          segment.custom_audio_beats = Array.isArray(data.beats) ? data.beats : [];
          if (segment.custom_audio_duration > 0 && !hasLockedVideo(segment)) {
            segment.end = Number(segment.start || 0) + segment.custom_audio_duration;
            normalizeSegments(segment);
          }
          updateCustomAudioStatus();
          render();
          toast(`Custom scene audio saved:\n${segment.custom_audio_path}`);
        } catch (error) {
          toast(String(error?.message || error), true);
        }
      };
      reader.onerror = () => toast("Failed to read the audio file.", true);
      reader.readAsDataURL(file);
    };
    const createSilentSceneAudio = async () => {
      const projectFolder = projectInput.value || state.projectFolder;
      if (!projectFolder) {
        toast("Set a project folder first so the silent scene audio can be created there.", true);
        return;
      }
      const duration = Math.max(0.1, Number(sceneSilenceDurationInput.value || timelineSegmentDuration(segment) || 4));
      try {
        createSceneSilence.disabled = true;
        createSceneSilence.textContent = "Creating...";
        const data = await postJson("/vrgdg/music_builder/create_silent_audio", {
          project_folder: projectFolder,
          scope: "scene",
          scene_number: sceneSlotNumber(segment),
          duration,
        }, 180000);
        pushHistory();
        segment.custom_audio_path = data.saved_path || data.audio_path || "";
        segment.custom_audio_name = data.audio_name || `Silence ${duration.toFixed(2)}s`;
        segment.custom_audio_duration = Number(data.duration || duration);
        segment.custom_audio_full_duration = Number(data.duration || duration);
        segment.custom_audio_timeline_start = Number(segment.start || 0);
        segment.custom_audio_source_start = 0;
        segment.custom_audio_peaks = Array.isArray(data.peaks) ? data.peaks : [];
        segment.custom_audio_beats = Array.isArray(data.beats) ? data.beats : [];
        if (segment.custom_audio_duration > 0 && !hasLockedVideo(segment)) {
          segment.end = Number(segment.start || 0) + segment.custom_audio_duration;
          normalizeSegments(segment);
        }
        updateCustomAudioStatus();
        render();
        toast(`Silent scene audio created:\n${segment.custom_audio_path}`);
      } catch (error) {
        toast(String(error?.message || error), true);
      } finally {
        createSceneSilence.disabled = false;
        createSceneSilence.textContent = "Use Silence For This Scene";
      }
    };
    pickCustomAudio.onclick = () => sceneAudioFileInput.click();
    createSceneSilence.onclick = createSilentSceneAudio;
    sceneAudioFileInput.onchange = () => {
      saveAudioFile(sceneAudioFileInput.files?.[0]);
      sceneAudioFileInput.value = "";
    };
    audioDrop.addEventListener("dragover", (event) => {
      if (!audioFileFromDrop(event)) return;
      event.preventDefault();
      event.stopPropagation();
      audioDrop.style.borderColor = "#a3e635";
    });
    audioDrop.addEventListener("dragleave", () => {
      audioDrop.style.borderColor = "#38bdf8";
    });
    audioDrop.addEventListener("drop", (event) => {
      const file = audioFileFromDrop(event);
      if (!file) return;
      event.preventDefault();
      event.stopPropagation();
      audioDrop.style.borderColor = "#38bdf8";
      saveAudioFile(file);
    });
    clearCustomAudio.onclick = () => {
      pushHistory();
      segment.custom_audio_path = "";
      segment.custom_audio_name = "";
      segment.custom_audio_duration = 0;
      segment.custom_audio_full_duration = 0;
      segment.custom_audio_timeline_start = Number(segment.start || 0);
      segment.custom_audio_source_start = 0;
      segment.custom_audio_peaks = [];
      segment.custom_audio_beats = [];
      updateCustomAudioStatus();
      render();
    };
    closeOptions.onclick = () => backdrop.remove();
    saveOptions.onclick = () => {
      pushHistory();
      render();
      backdrop.remove();
      toast(segment.custom_audio_path ? `Custom scene audio saved:\n${segment.custom_audio_path}` : "Custom scene audio cleared.");
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
  }

  function openAudioContextMenu(event, segment) {
    event.preventDefault();
    event.stopPropagation();
    setActiveSegment(segment);
    document.querySelector(".vrgdg-builder-context-menu")?.remove();
    if (!segment?.custom_audio_path) return;
    const menu = document.createElement("div");
    menu.className = "vrgdg-builder-context-menu";
    menu.style.cssText = "position:fixed;z-index:100010;min-width:190px;border:1px solid #7e22ce;border-radius:7px;background:#111827;color:#f8fafc;box-shadow:0 16px 50px rgba(0,0,0,.55);padding:6px;display:flex;flex-direction:column;gap:4px;";
    menu.style.left = `${Math.min(window.innerWidth - 200, event.clientX)}px`;
    menu.style.top = `${Math.min(window.innerHeight - 150, event.clientY)}px`;
    const addItem = (label, action, disabled = false) => {
      const button = makeButton(label);
      button.disabled = disabled;
      button.style.justifyContent = "flex-start";
      button.style.textAlign = "left";
      if (disabled) {
        button.style.opacity = ".45";
        button.style.cursor = "not-allowed";
      }
      button.onclick = () => {
        menu.remove();
        action();
      };
      menu.append(button);
    };
    const rect = event.currentTarget?.getBoundingClientRect?.();
    const ratio = rect ? Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width))) : 0.5;
    const duration = audioChunkDuration(segment);
    const cutLocal = ratio * duration;
    addItem("Cut audio here", () => {
      if (cutLocal < 0.25 || duration - cutLocal < 0.25) {
        toast("Cut point is too close to the edge of the audio chunk.", true);
        return;
      }
      pushHistory();
      const audioStart = audioTimelineStart(segment);
      const cutTime = audioStart + cutLocal;
      const rightDuration = duration - cutLocal;
      const right = newSegment(cutTime, cutTime + rightDuration);
      right.label = `${segment.label || "Scene"} audio`;
      right.source = state.srtMode ? "inserted" : "manual";
      right.custom_audio_path = segment.custom_audio_path;
      right.custom_audio_name = segment.custom_audio_name;
      right.custom_audio_full_duration = Number(segment.custom_audio_full_duration || duration);
      right.custom_audio_timeline_start = cutTime;
      right.custom_audio_source_start = audioSourceStart(segment) + cutLocal;
      right.custom_audio_duration = rightDuration;
      const [leftPeaks, rightPeaks] = splitAudioPeaks(segment.custom_audio_peaks, ratio);
      const [leftBeats, rightBeats] = splitAudioBeats(segment.custom_audio_beats, cutLocal, duration);
      segment.custom_audio_peaks = leftPeaks;
      segment.custom_audio_beats = leftBeats;
      right.custom_audio_peaks = rightPeaks;
      right.custom_audio_beats = rightBeats;
      segment.custom_audio_duration = cutLocal;
      segment.end = Math.min(Number(segment.end || cutTime), cutTime);
      const index = state.segments.findIndex((item) => item.id === segment.id);
      state.segments.splice(index + 1, 0, right);
      state.activeId = right.id;
      render();
      syncInspector();
    });
    addItem("Delete audio chunk", () => {
      pushHistory();
      clearSegmentAudio(segment);
      render();
      syncInspector();
    });
    addItem("Scene options", () => openSceneOptions(segment));
    document.body.append(menu);
    const close = (closeEvent) => {
      if (!menu.contains(closeEvent.target)) {
        menu.remove();
        window.removeEventListener("pointerdown", close);
      }
    };
    setTimeout(() => window.addEventListener("pointerdown", close), 0);
  }

  function shiftTimelineAfterTrim(segment, oldEnd, removedDuration) {
    const delta = -Math.max(0, Number(removedDuration || 0));
    if (!segment || !delta) return;
    const threshold = Number(oldEnd || 0) - 0.001;
    for (const item of state.segments) {
      if (item.id !== segment.id && Number(item.start || 0) >= threshold) shiftSegmentTiming(item, delta);
    }
    for (const item of state.overlaySegments) {
      if (Number(item.start || 0) >= threshold) shiftSegmentTiming(item, delta);
    }
    sortSegments(state.segments);
    sortSegments(state.overlaySegments);
  }

  function closeBaseTimelineGap(startTime, endTime) {
    const start = Number(startTime || 0);
    const end = Number(endTime || 0);
    const duration = Math.max(0, end - start);
    if (duration <= 0.0001) return 0;
    const delta = -duration;
    const threshold = end - 0.001;
    for (const item of state.segments) {
      if (Number(item.start || 0) >= threshold) shiftSegmentTiming(item, delta);
    }
    for (const item of state.overlaySegments) {
      if (Number(item.start || 0) >= threshold) shiftSegmentTiming(item, delta);
    }
    sortSegments(state.segments);
    sortSegments(state.overlaySegments);
    return duration;
  }

  function closeAllBaseTimelineGaps() {
    const sorted = [...state.segments].sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    let removed = 0;
    let cursor = 0;
    for (const segment of sorted) {
      const start = Number(segment.start || 0);
      const end = Math.max(start + 0.05, Number(segment.end || start + 0.05));
      if (start > cursor + 0.001) {
        const gap = start - cursor;
        shiftSegmentTiming(segment, -gap);
        removed += gap;
        for (const overlay of state.overlaySegments) {
          if (Number(overlay.start || 0) >= start - 0.001) shiftSegmentTiming(overlay, -gap);
        }
      } else if (Math.abs(start - cursor) <= 0.001 && start !== cursor) {
        shiftSegmentTiming(segment, cursor - start);
      }
      cursor = Math.max(cursor, Number(segment.end || 0));
    }
    sortSegments(state.segments);
    sortSegments(state.overlaySegments);
    return removed;
  }

  async function closeTimelineGapsFromMenu() {
    const undoLength = state.undoStack.length;
    pushHistory();
    const removed = closeAllBaseTimelineGaps();
    if (removed <= 0.001) {
      if (state.undoStack.length > undoLength) {
        state.undoStack.pop();
        updateHistoryButtons();
      }
      toast("No base timeline gaps found.", true);
      return;
    }
    syncInspector();
    render();
    await autoSaveSessionQuiet("timeline gaps closed");
    toast(`Closed ${removed.toFixed(2)}s of timeline gap.`);
  }

  async function snapSceneEdgeToNearestBeat(segment, side) {
    if (!segment || segmentTrack(segment) === "overlay") {
      toast("Select a base scene before snapping a scene edge.", true);
      return false;
    }
    if (state.timingFrozen) {
      toast("Scene timing is frozen. Unfreeze timing before snapping a scene edge.", true);
      return false;
    }
    if (hasLockedVideo(segment)) {
      toast("Clear this scene's rendered video before changing its timing.", true);
      return false;
    }
    if (!Array.isArray(state.beats) || !state.beats.length) {
      const loaded = await reloadBeatMarkersFromAudio();
      if (!loaded) return false;
    }

    const sorted = [...state.segments].sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    const index = sorted.findIndex((item) => item.id === segment.id);
    if (index < 0) {
      toast("The selected scene is no longer on the base timeline.", true);
      return false;
    }

    const minDuration = 0.1;
    const connectedTolerance = 0.03;
    const previous = sorted[index - 1] || null;
    const next = sorted[index + 1] || null;
    const current = side === "start" ? Number(segment.start || 0) : Number(segment.end || 0);
    const previousIsConnected = Boolean(previous && Math.abs(Number(previous.end || 0) - Number(segment.start || 0)) <= connectedTolerance);
    const nextIsConnected = Boolean(next && Math.abs(Number(next.start || 0) - Number(segment.end || 0)) <= connectedTolerance);
    const connectedNeighbor = side === "start" && previousIsConnected
      ? previous
      : side === "end" && nextIsConnected
        ? next
        : null;

    if (connectedNeighbor && hasLockedVideo(connectedNeighbor)) {
      const neighborIndex = sorted.findIndex((item) => item.id === connectedNeighbor.id);
      toast(`Clear ${sceneDisplayName(connectedNeighbor, neighborIndex)}'s rendered video before moving their shared boundary.`, true);
      return false;
    }

    let minimum;
    let maximum;
    if (side === "start") {
      minimum = previous
        ? previousIsConnected
          ? Number(previous.start || 0) + minDuration
          : Number(previous.end || 0)
        : 0;
      maximum = Number(segment.end || 0) - minDuration;
    } else {
      minimum = Number(segment.start || 0) + minDuration;
      maximum = next
        ? nextIsConnected
          ? Number(next.end || 0) - minDuration
          : Number(next.start || 0)
        : Number.POSITIVE_INFINITY;
      const audioEnd = loadedGlobalAudioDuration();
      if (audioEnd > 0) maximum = Math.min(maximum, audioEnd);
    }

    const beats = state.beats
      .map((beat) => Number(beat?.time ?? beat))
      .filter((beat) => Number.isFinite(beat) && beat >= minimum - 0.0001 && beat <= maximum + 0.0001)
      .sort((a, b) => a - b);
    if (!beats.length) {
      toast(`No beat marker can fit this scene's ${side} without overlapping another scene or making a clip shorter than ${minDuration.toFixed(1)}s.`, true);
      return false;
    }

    const target = beats.reduce((closest, beat) => (
      Math.abs(beat - current) < Math.abs(closest - current) ? beat : closest
    ), beats[0]);
    if (Math.abs(target - current) <= 0.0001) {
      setBeatMarkersVisible(true);
      render();
      toast(`The selected scene ${side} is already on its closest beat marker.`);
      return true;
    }

    pauseTimelineForEditing();
    pushHistory();
    if (side === "start") {
      segment.start = target;
      if (previousIsConnected) previous.end = target;
    } else {
      segment.end = target;
      if (nextIsConnected) next.start = target;
    }
    sortSegments(state.segments);
    setBeatMarkersVisible(true);
    state.sceneSelectionUsesGlobalAudio = true;
    setGlobalPlaybackTime(target);
    syncInspector();
    render();
    await autoSaveSessionQuiet(`scene ${side} snapped to beat`);

    const neighborText = connectedNeighbor
      ? ` ${sceneDisplayName(connectedNeighbor, sorted.findIndex((item) => item.id === connectedNeighbor.id))}'s connected ${side === "start" ? "end" : "start"} moved with it; its other edge stayed fixed.`
      : " No other scene timing changed.";
    toast(`Snapped ${sceneDisplayName(segment, index)} ${side} from ${formatTime(current)} to ${formatTime(target)}.${neighborText}`);
    return true;
  }

  async function snapAllSceneStartsToNearestBeats() {
    if (state.timingFrozen) {
      toast("Scene timing is frozen. Unfreeze timing before snapping scene starts.", true);
      return false;
    }
    const sorted = [...state.segments].sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    if (sorted.length < 3) {
      toast("At least three base scenes are required to snap scene starts beginning with Scene 3.", true);
      return false;
    }
    if (!Array.isArray(state.beats) || !state.beats.length) {
      const loaded = await reloadBeatMarkersFromAudio();
      if (!loaded) return false;
    }
    const beats = state.beats
      .map((beat) => Number(beat?.time ?? beat))
      .filter((beat) => Number.isFinite(beat) && beat >= 0)
      .sort((a, b) => a - b);
    if (!beats.length) {
      toast("No beat-marker grid is available for scene snapping.", true);
      return false;
    }

    const minimumDuration = 0.1;
    const connectedTolerance = 0.03;
    const simulated = sorted.map((segment) => ({
      segment,
      start: Number(segment.start || 0),
      end: Number(segment.end || 0),
    }));
    const changes = [];
    for (let index = 2; index < simulated.length; index += 1) {
      const previous = simulated[index - 1];
      const current = simulated[index];
      const connected = Math.abs(previous.end - current.start) <= connectedTolerance;
      const minimum = connected ? previous.start + minimumDuration : previous.end;
      const maximum = current.end - minimumDuration;
      const validBeats = beats.filter((beat) => beat >= minimum - 0.0001 && beat <= maximum + 0.0001);
      if (!validBeats.length) {
        toast(`Could not snap ${sceneDisplayName(current.segment, index)} because no beat fits between its neighboring scene boundaries. No timing was changed.`, true);
        return false;
      }
      const target = validBeats.reduce((nearest, beat) => (
        Math.abs(beat - current.start) < Math.abs(nearest - current.start) ? beat : nearest
      ), validBeats[0]);
      if (Math.abs(target - current.start) <= 0.0001) continue;
      const affected = [current.segment];
      if (connected) affected.push(previous.segment);
      const locked = affected.find((segment) => hasLockedVideo(segment));
      if (locked) {
        const lockedIndex = sorted.findIndex((segment) => segment.id === locked.id);
        toast(`Clear ${sceneDisplayName(locked, lockedIndex)}'s rendered video before moving its shared scene boundary. No timing was changed.`, true);
        return false;
      }
      changes.push({ index, from: current.start, to: target, connected });
      current.start = target;
      if (connected) previous.end = target;
    }

    if (!changes.length) {
      setBeatMarkersVisible(true);
      render();
      toast("Every scene start from Scene 3 onward is already on its nearest valid beat marker.");
      return true;
    }

    const connectedCount = changes.filter((change) => change.connected).length;
    const confirmed = window.confirm(
      `Snap ${changes.length} scene start${changes.length === 1 ? "" : "s"} to their nearest beat markers?\n\n` +
      "The Scene 1-to-2 boundary will stay fixed. " +
      `${connectedCount} connected previous scene end${connectedCount === 1 ? "" : "s"} will move with the shared cuts. ` +
      "The final scene's end will stay fixed.\n\nNo images or videos will be removed."
    );
    if (!confirmed) return false;

    pauseTimelineForEditing();
    pushHistory();
    for (const item of simulated) {
      item.segment.start = Number(item.start.toFixed(3));
      item.segment.end = Number(item.end.toFixed(3));
    }
    sortSegments(state.segments);
    setBeatMarkersVisible(true);
    syncInspector();
    render();
    await autoSaveSessionQuiet("scene 3 and later starts snapped to beats");
    toast(`Snapped ${changes.length} scene start${changes.length === 1 ? "" : "s"} to their nearest beat markers.`);
    return true;
  }

  async function openSnapSceneEdgeMenu() {
    const segment = activeSegment();
    if (!segment || segmentTrack(segment) === "overlay") {
      toast("Select a base scene before snapping a scene edge.", true);
      return;
    }
    const index = [...state.segments]
      .sort((a, b) => Number(a.start || 0) - Number(b.start || 0))
      .findIndex((item) => item.id === segment.id);
    const side = await chooseBatchModeAction({
      title: "Snap Scene Edge to Beat",
      intro: `Choose which edge of ${sceneDisplayName(segment, index)} should move to its closest beat marker. Only a directly connected neighbor's shared edge will follow.`,
      confirmLabel: "Snap to Closest Beat",
      choices: [
        {
          value: "start",
          label: "Start of scene",
          description: "Move this scene's start. If the previous scene touches it, only that previous scene's end moves with the cut.",
        },
        {
          value: "end",
          label: "End of scene",
          description: "Move this scene's end. If the next scene touches it, only that next scene's start moves with the cut.",
        },
      ],
    });
    if (!side) return;
    await snapSceneEdgeToNearestBeat(segment, side);
  }

  function baseSceneVideoTrimKind(segment) {
    if (!segment || segmentTrack(segment) === "overlay" || !String(selectedSegmentVideoPath(segment) || "").trim()) return "";
    if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
      return miniMaxH3SettingsForSegment(segment).audio_mode === "built_in_audio" ? "minimax_h3" : "";
    }
    return currentVideoMode() === "id_lora" ? "id_lora" : "";
  }

  async function chooseRenderedSceneTrimAtPlayhead() {
    const segment = activeSegment();
    if (!segment || segmentTrack(segment) === "overlay") {
      toast("Select a rendered base scene before right-clicking ✂ to trim it.", true);
      return;
    }
    if (!String(selectedSegmentVideoPath(segment) || "").trim()) {
      toast("The selected scene does not have a rendered video to trim.", true);
      return;
    }
    if (!baseSceneVideoTrimKind(segment)) {
      toast("Right-click trimming is available for MiniMax built-in-audio and ID-LoRA rendered scenes.", true);
      return;
    }

    const trimTime = Number(currentGlobalTime() || 0);
    const sceneStart = Number(segment.start || 0);
    const sceneEnd = Number(segment.end || sceneStart);
    if (trimTime <= sceneStart + 0.15 || trimTime >= sceneEnd - 0.15) {
      toast("Move the playhead inside the selected rendered scene, then right-click ✂ again.", true);
      return;
    }
    pauseTimelineForEditing();
    const choice = await chooseBatchModeAction({
      title: "Trim Rendered Scene at Playhead",
      intro: `${sceneDisplayName(segment, segmentIndexInfo(segment).index)} — playhead at ${formatTime(trimTime)}. Choose which portion to remove.`,
      confirmLabel: "Trim Scene",
      choices: [
        {
          value: "before",
          label: "Remove before playhead",
          description: "Discard the beginning of the rendered clip and keep everything from the playhead onward.",
        },
        {
          value: "after",
          label: "Remove after playhead",
          description: "Keep the beginning of the rendered clip and discard everything after the playhead.",
        },
      ],
    });
    if (!choice) return;
    await trimBaseSceneVideoAtPlayhead(segment, choice === "before" ? "left" : "right", trimTime);
  }

  async function trimBaseSceneVideoAtPlayhead(segment, side, trimTimeOverride = null) {
    const videoPath = String(selectedSegmentVideoPath(segment) || "").trim();
    const trimKind = baseSceneVideoTrimKind(segment);
    const miniMaxTrim = trimKind === "minimax_h3";
    const modeLabel = miniMaxTrim ? "MiniMax H3 built-in-audio" : "ID-LoRA";
    if (!trimKind) {
      const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
      toast(miniMaxProject
        ? "Playhead trimming is currently available for MiniMax built-in-audio scenes. Input-audio scenes stay protected so their separately stitched source audio cannot drift."
        : "Timeline video trim is only available in ID-LoRA I2V mode.", true);
      return;
    }
    if (!segment || segmentTrack(segment) === "overlay" || !videoPath) {
      toast(`Select a rendered ${modeLabel} scene video first.`, true);
      return;
    }
    const sceneStart = Number(segment.start || 0);
    const sceneEnd = Number(segment.end || 0);
    const sceneDuration = Math.max(0, sceneEnd - sceneStart);
    const playheadTime = Number.isFinite(Number(trimTimeOverride)) ? Number(trimTimeOverride) : currentGlobalTime();
    const local = Math.max(0, Math.min(sceneDuration, playheadTime - sceneStart));
    const minDuration = 0.15;
    const trimLeft = side === "left";
    if (playheadTime <= sceneStart + minDuration || playheadTime >= sceneEnd - minDuration) {
      toast("Move the playhead or right-click point inside the scene before trimming.", true);
      return;
    }
    const sourceStart = trimLeft ? local : 0;
    const nextDuration = trimLeft ? sceneDuration - local : local;
    const removedDuration = trimLeft ? local : sceneDuration - local;
    if (nextDuration < minDuration || removedDuration <= 0) {
      toast("Trim point is too close to the edge of the scene.", true);
      return;
    }
    const label = trimLeft ? "trim_left" : "trim_right";
    const progress = createProgressWindow(trimLeft ? "Trimming left side of scene video" : "Trimming right side of scene video");
    try {
      pauseTimelineForEditing();
      progress.set(`Creating trimmed ${modeLabel} scene video with synchronized embedded audio...\n${videoPath}`, 20);
      const data = await postJson("/vrgdg/workflow_runner/trim_scene_video", {
        project_folder: projectInput.value || state.projectFolder || "",
        source_path: videoPath,
        scene_number: sceneSlotNumber(segment),
        start: sourceStart,
        duration: nextDuration,
        label,
        mark_as_audio_video: miniMaxTrim,
      }, 180000);
      pushHistory();
      const oldEnd = sceneEnd;
      activateSegmentVideoPath(segment, data.video_path || videoPath, data.thumbnail_path || "");
      segment.video_cache_bust = Date.now();
      segment.video_status = "done";
      segment.end = sceneStart + nextDuration;
      shiftTimelineAfterTrim(segment, oldEnd, removedDuration);
      if (miniMaxTrim) {
        segment.minimax_h3_trimmed_video = true;
        segment.minimax_h3_trim_side = trimLeft ? "left" : "right";
        segment.minimax_h3_trim_removed_seconds = Number(removedDuration.toFixed(3));
        segment.minimax_h3_trim_source_start_seconds = Number(sourceStart.toFixed(3));
        segment.minimax_h3_trimmed_duration_seconds = Number(nextDuration.toFixed(3));
        if (!trimLeft) {
          const nextSegment = allEditableSegments()
            .filter((item) => segmentTrack(item) === segmentTrack(segment))
            .sort((left, right) => Number(left.start || 0) - Number(right.start || 0))
            .find((item) => Number(item.start || 0) >= Number(segment.end || 0) - 0.001 && item.id !== segment.id);
          if (nextSegment) {
            nextSegment.minimax_h3_continuity_frame_path = "";
            nextSegment.minimax_h3_continuity_source_video_path = "";
            nextSegment.minimax_h3_continuity_source_scene_id = "";
            nextSegment.minimax_h3_continuity_image_number = 0;
          }
        }
      }
      state.activeId = segment.id;
      state.sceneAudioGlobalTime = trimLeft ? Number(segment.start || 0) : Number(segment.end || 0);
      syncInspector();
      render();
      await autoSaveSessionQuiet(`${modeLabel} scene video trimmed ${trimLeft ? "left" : "right"}`);
      progress.set(`Trim complete.\n${data.video_path || ""}`, 100);
      progress.close(900);
      toast(`Trimmed ${trimLeft ? "left" : "right"} side of ${modeLabel} scene video.${miniMaxTrim && !trimLeft ? " Rerender the following scene if it was already created with previous-frame continuity." : ""}`);
    } catch (error) {
      progress.set(`Trim failed:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    }
  }

  async function trimOverlayVideoAtPlayhead(segment, side, trimTimeOverride = null) {
    const videoPath = String(selectedSegmentVideoPath(segment) || "").trim();
    if (!segment || segmentTrack(segment) !== "overlay" || !videoPath) return;
    if (segment.overlay_locked !== false) {
      toast("Unlock this overlay clip before trimming it.", true);
      return;
    }
    const start = Number(segment.start || 0);
    const end = Number(segment.end || 0);
    const cut = Number.isFinite(Number(trimTimeOverride)) ? Number(trimTimeOverride) : currentGlobalTime();
    if (cut <= start + 0.15 || cut >= end - 0.15) {
      toast("Move the playhead inside the overlay clip before trimming.", true);
      return;
    }
    const trimLeft = side === "left";
    const sourceStart = trimLeft ? cut - start : 0;
    const duration = trimLeft ? end - cut : cut - start;
    const progress = createProgressWindow(trimLeft ? "Trimming overlay left" : "Trimming overlay right");
    try {
      progress.set(`Creating trimmed overlay video...\n${videoPath}`, 20);
      const data = await postJson("/vrgdg/workflow_runner/trim_scene_video", {
        project_folder: projectInput.value || state.projectFolder || "",
        source_path: videoPath,
        scene_number: sceneSlotNumber(segment),
        start: sourceStart,
        duration,
        label: trimLeft ? "overlay_trim_left" : "overlay_trim_right",
      }, 180000);
      pushHistory();
      activateSegmentVideoPath(segment, data.video_path || videoPath, data.thumbnail_path || "");
      if (trimLeft) segment.start = cut;
      else segment.end = cut;
      segment.video_cache_bust = Date.now();
      sortSegments(state.overlaySegments);
      render();
      syncInspector();
      await autoSaveSessionQuiet("overlay video trimmed");
      progress.set(`Overlay trim complete.\n${data.video_path || ""}`, 100);
      progress.close(900);
    } catch (error) {
      progress.set(`Trim failed:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    }
  }

  function migrateSceneMappingsAfterMerge(targetId, removedId) {
    const builder = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const subjectMap = builder.subject_scene_map && typeof builder.subject_scene_map === "object"
      ? builder.subject_scene_map
      : {};
    const mergedSubjects = mergeUniqueStringArray(subjectMap[targetId], subjectMap[removedId]);
    if (mergedSubjects.length) subjectMap[targetId] = mergedSubjects;
    else delete subjectMap[targetId];
    delete subjectMap[removedId];
    builder.subject_scene_map = subjectMap;

    ["scene_map", "scene_trigger_map", "ingredients_scene_map"].forEach((mapName) => {
      const sceneMap = builder[mapName] && typeof builder[mapName] === "object"
        ? builder[mapName]
        : {};
      const targetHasValue = Object.prototype.hasOwnProperty.call(sceneMap, targetId)
        && sceneMap[targetId] !== null
        && sceneMap[targetId] !== "";
      if (!targetHasValue && Object.prototype.hasOwnProperty.call(sceneMap, removedId)) {
        sceneMap[targetId] = sceneMap[removedId];
      }
      delete sceneMap[removedId];
      builder[mapName] = sceneMap;
    });

    state.fluxReferenceBuilder = builder;

    const idLoraBuilder = normalizeIdLoraReferenceBuilder(state.idLoraReferenceBuilder);
    const targetEntry = idLoraBuilder.scene_map[targetId] || null;
    const removedEntry = idLoraBuilder.scene_map[removedId] || null;
    if (targetEntry || removedEntry) {
      const targetScene = state.segments.find((scene) => String(scene.id) === targetId);
      const dialogue = mergeUniqueSceneText(targetEntry?.dialogue, removedEntry?.dialogue, "\n");
      idLoraBuilder.scene_map[targetId] = {
        ...(removedEntry || {}),
        ...(targetEntry || {}),
        dialogue: dialogue || String(targetScene?.lyric_text || "").trim(),
        manual_duration: Math.max(
          0.25,
          Number(targetScene?.end || 0) - Number(targetScene?.start || 0),
        ),
      };
    }
    delete idLoraBuilder.scene_map[removedId];
    state.idLoraReferenceBuilder = normalizeIdLoraReferenceBuilder(idLoraBuilder);
  }

  async function mergeAdjacentBaseScene(selectedScene, direction) {
    if (!selectedScene || segmentTrack(selectedScene) === "overlay") return;

    const baseScenes = [...state.segments].sort(
      (a, b) => Number(a.start || 0) - Number(b.start || 0),
    );
    const selectedIndex = baseScenes.findIndex((scene) => String(scene.id) === String(selectedScene.id));
    const neighborIndex = direction === "left" ? selectedIndex - 1 : selectedIndex + 1;
    if (selectedIndex < 0 || neighborIndex < 0 || neighborIndex >= baseScenes.length) {
      toast(`There is no scene on the ${direction} to merge.`, true);
      return;
    }

    const selected = baseScenes[selectedIndex];
    const neighbor = baseScenes[neighborIndex];
    const leftScene = direction === "left" ? neighbor : selected;
    const rightScene = direction === "left" ? selected : neighbor;
    if (hasLockedVideo(leftScene) || hasLockedVideo(rightScene)) {
      toast("Remove the rendered video from both scenes before merging them.", true);
      return;
    }

    // Later scenes move up one position, so their numbered files (video_0017 -> video_0016, ...) must follow;
    // otherwise reloading the project pairs each scene with its neighbour's files.
    const projectFolder = String(state.projectFolder || projectInput.value || "").trim();
    const renamed = projectFolder
      ? (await postJson("/vrgdg/music_builder/renumber_scenes_after_removal", {
        project_folder: projectFolder,
        removed_scene_number: sceneSlotNumber(rightScene),
      })).renamed
      : [];

    pushHistory();
    const leftLyric = String(leftScene.lyric_text || "").trim();
    const rightLyric = String(rightScene.lyric_text || "").trim();
    leftScene.start = Math.min(Number(leftScene.start || 0), Number(rightScene.start || 0));
    leftScene.end = Math.max(
      Number(leftScene.end || leftScene.start),
      Number(rightScene.end || rightScene.start),
    );
    leftScene.timeline_note = mergeUniqueSceneText(leftScene.timeline_note, rightScene.timeline_note, "\n");
    leftScene.notes = mergeUniqueSceneText(leftScene.notes, rightScene.notes);
    leftScene.flux_notes = mergeUniqueSceneText(leftScene.flux_notes, rightScene.flux_notes);
    leftScene.nb_notes = mergeUniqueSceneText(leftScene.nb_notes, rightScene.nb_notes);
    leftScene.i2v_notes = mergeUniqueSceneText(leftScene.i2v_notes, rightScene.i2v_notes);
    leftScene.story_beat = mergeUniqueSceneText(leftScene.story_beat, rightScene.story_beat);
    leftScene.lyric_text = leftLyric || rightLyric
      ? mergeTimestampedLyricText(leftLyric, rightLyric, "[instrumental]")
      : "";
    leftScene.lyric_singers = mergeUniqueStringArray(leftScene.lyric_singers, rightScene.lyric_singers);
    leftScene.lyric_no_lip_sync = leftScene.lyric_text
      ? isInstrumentalLyricText(leftScene.lyric_text)
      : Boolean(leftScene.lyric_no_lip_sync && rightScene.lyric_no_lip_sync);

    state.segments = state.segments.filter((scene) => String(scene.id) !== String(rightScene.id));
    migrateSceneMappingsAfterMerge(String(leftScene.id), String(rightScene.id));
    renumberGenericBaseSceneLabels(state.segments);
    if (renamed.length) {
      rewriteRenamedScenePaths(state.segments, renamed);
      rewriteRenamedScenePaths(state.overlaySegments, renamed);
      // Undo would restore the old numbering while the files keep the new names.
      state.undoStack = [];
      state.redoStack = [];
      updateHistoryButtons();
    }

    state.activeTrack = "base";
    state.activeId = leftScene.id;
    state.selectedSegmentIds = state.multiSelectMode ? [leftScene.id] : [];
    syncInspector();
    render();
    await syncPromptJsonFromSegments("adjacent scenes merged");
    await syncI2VMotionJsonFromSegments("adjacent scenes merged");
    autoSaveSessionQuiet("adjacent scenes merged");
    loadDirtyLatentBadges();
    toast(renamed.length
      ? `Merged into ${leftScene.label || "scene"}. Later scene files were renumbered to match; undo history was cleared.`
      : `Merged into ${leftScene.label || "scene"}.`);
  }

  function openSegmentContextMenu(event, segment) {
    event.preventDefault();
    event.stopPropagation();
    state.timelineEditMenuOpen = true;
    pauseTimelineForEditing();
    setActiveSegment(segment);
    document.querySelector(".vrgdg-builder-context-menu")?.remove();
    const menu = document.createElement("div");
    menu.className = "vrgdg-builder-context-menu";
    menu.style.cssText = "position:fixed;z-index:100010;min-width:180px;border:1px solid #155e75;border-radius:7px;background:#111827;color:#f8fafc;box-shadow:0 16px 50px rgba(0,0,0,.55);padding:6px;display:flex;flex-direction:column;gap:4px;";
    menu.style.left = `${Math.min(window.innerWidth - 190, event.clientX)}px`;
    menu.style.top = `${Math.min(window.innerHeight - 150, event.clientY)}px`;
    const closeMenu = () => {
      state.timelineEditMenuOpen = false;
      menu.remove();
      window.removeEventListener("pointerdown", closeOnPointer, true);
      window.removeEventListener("contextmenu", closeOnPointer, true);
      window.removeEventListener("keydown", closeOnKey, true);
    };
    const closeOnPointer = (closeEvent) => {
      if (!menu.contains(closeEvent.target)) closeMenu();
    };
    const closeOnKey = (keyEvent) => {
      if (keyEvent.key === "Escape") closeMenu();
    };
    const addItem = (label, action, disabled = false) => {
      const button = makeButton(label);
      button.disabled = disabled;
      button.style.justifyContent = "flex-start";
      button.style.textAlign = "left";
      button.onclick = () => {
        closeMenu();
        action();
      };
      menu.append(button);
    };
    const isOverlay = segmentTrack(segment) === "overlay";
    const baseTrimKind = !isOverlay ? baseSceneVideoTrimKind(segment) : "";
    const playheadTime = currentGlobalTime();
    const sceneStart = Number(segment.start || 0);
    const sceneEnd = Number(segment.end || 0);
    const rect = event.currentTarget?.getBoundingClientRect?.();
    const ratio = rect ? Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width))) : 0.5;
    const clickedTime = sceneStart + ratio * Math.max(0.1, sceneEnd - sceneStart);
    const playheadInsideScene = playheadTime > sceneStart + 0.15 && playheadTime < sceneEnd - 0.15;
    const clickInsideScene = clickedTime > sceneStart + 0.15 && clickedTime < sceneEnd - 0.15;
    const trimTime = playheadInsideScene ? playheadTime : clickedTime;
    const trimLabel = playheadInsideScene ? "Playhead" : "Click";
    addItem("Restore Video...", () => restoreVideoForSegment(segment), !String(state.projectFolder || projectInput.value || "").trim());
    if (baseTrimKind) {
      addItem(`Trim Left at ${trimLabel}`, () => trimBaseSceneVideoAtPlayhead(segment, "left", trimTime), !(playheadInsideScene || clickInsideScene));
      addItem(`Trim Right at ${trimLabel}`, () => trimBaseSceneVideoAtPlayhead(segment, "right", trimTime), !(playheadInsideScene || clickInsideScene));
    }
    if (isOverlay && String(selectedSegmentVideoPath(segment) || "").trim()) {
      addItem(`Trim Left at ${trimLabel}`, () => trimOverlayVideoAtPlayhead(segment, "left", trimTime), segment.overlay_locked !== false || !(playheadInsideScene || clickInsideScene));
      addItem(`Trim Right at ${trimLabel}`, () => trimOverlayVideoAtPlayhead(segment, "right", trimTime), segment.overlay_locked !== false || !(playheadInsideScene || clickInsideScene));
    }
    if (!isOverlay) {
      const baseScenes = [...state.segments].sort(
        (a, b) => Number(a.start || 0) - Number(b.start || 0),
      );
      const sceneIndex = baseScenes.findIndex((scene) => String(scene.id) === String(segment.id));
      const leftScene = sceneIndex > 0 ? baseScenes[sceneIndex - 1] : null;
      const rightScene = sceneIndex >= 0 && sceneIndex < baseScenes.length - 1
        ? baseScenes[sceneIndex + 1]
        : null;
      addItem(
        "Merge with scene on left",
        () => mergeAdjacentBaseScene(segment, "left").catch((error) => toast(String(error?.message || error), true)),
        !leftScene || hasLockedVideo(segment) || hasLockedVideo(leftScene),
      );
      addItem(
        "Merge with scene on right",
        () => mergeAdjacentBaseScene(segment, "right").catch((error) => toast(String(error?.message || error), true)),
        !rightScene || hasLockedVideo(segment) || hasLockedVideo(rightScene),
      );
      addItem("Copy as insert track", () => copyBaseSceneAsOverlay(segment));
    }
    addItem("Close timeline gaps", closeTimelineGapsFromMenu);
    addItem("Scene options", () => openSceneOptions(segment));
    addItem("Delete scene", deleteSegment);
    document.body.append(menu);
    setTimeout(() => {
      window.addEventListener("pointerdown", closeOnPointer, true);
      window.addEventListener("contextmenu", closeOnPointer, true);
      window.addEventListener("keydown", closeOnKey, true);
    }, 0);
  }

  function clearDirectorNote(segment, noteBox = null) {
    const segmentId = segment?.id || "";
    const target = allEditableSegments().find((item) => item.id === segmentId) || segment;
    if (!target) return false;
    if (noteBox) {
      noteBox.dataset.deleted = "1";
      noteBox.value = "";
    }
    target.timeline_note = "";
    if (segment && segment !== target) segment.timeline_note = "";
    if (target.id === activeSegment()?.id) syncInspector();
    render();
    return true;
  }

  function openDirectorNoteContextMenu(event, segment, noteBox = null) {
    event.preventDefault();
    event.stopPropagation();
    if (!segment) return;
    if (noteBox) segment.timeline_note = noteBox.value || "";
    setActiveSegment(segment);
    document.querySelector(".vrgdg-builder-context-menu")?.remove();
    const menu = document.createElement("div");
    menu.className = "vrgdg-builder-context-menu";
    menu.style.cssText = "position:fixed;z-index:100010;min-width:220px;border:1px solid #155e75;border-radius:7px;background:#111827;color:#f8fafc;box-shadow:0 16px 50px rgba(0,0,0,.55);padding:6px;display:flex;flex-direction:column;gap:4px;";
    menu.style.left = `${Math.min(window.innerWidth - 230, event.clientX)}px`;
    menu.style.top = `${Math.min(window.innerHeight - 150, event.clientY)}px`;
    const addItem = (label, action, disabled = false) => {
      const button = makeButton(label);
      button.disabled = disabled;
      button.style.justifyContent = "flex-start";
      button.style.textAlign = "left";
      button.onpointerdown = (buttonEvent) => {
        buttonEvent.stopPropagation();
      };
      button.onclick = (buttonEvent) => {
        buttonEvent.preventDefault();
        buttonEvent.stopPropagation();
        closeMenu();
        action();
      };
      menu.append(button);
    };
    const noteText = String(noteBox?.value ?? segment.timeline_note ?? "").trim();
    addItem("Copy to Timeline Note", () => {
      const start = Number(segment.start || 0);
      const end = Math.max(start + 0.15, Number(segment.end || start + 4));
      pushHistory();
      const marker = newTimelineMarker(start, end);
      marker.type = "director note";
      marker.label = `${segment.label || sceneDisplayName(segment, segmentIndexInfo(segment).index)} note`;
      marker.note = noteText;
      const clamped = clampTimelineMarkerToNonOverlap(marker, marker.start, marker.end ?? marker.start + 4);
      marker.start = clamped.start;
      marker.end = clamped.end;
      state.timelineMarkers.push(marker);
      state.timelineMarkers = normalizeTimelineMarkers(state.timelineMarkers);
      state.activeTimelineMarkerId = marker.id;
      render();
      autoSaveSessionQuiet("director note copied to timeline note").catch(() => null);
      toast("Director note copied into a Timeline Note.");
    }, !noteText);
    addItem("Delete Director Note", () => {
      pushHistory();
      clearDirectorNote(segment, noteBox);
      autoSaveSessionQuiet("director note deleted").catch(() => null);
      toast("Director note deleted.");
    }, !noteText);
    addItem("Open Scene Options", () => openSceneOptions(segment));
    document.body.append(menu);
    const closeMenu = () => {
      menu.remove();
      window.removeEventListener("pointerdown", closeOnPointer, true);
      window.removeEventListener("click", closeOnPointer, true);
      window.removeEventListener("contextmenu", closeOnPointer, true);
      window.removeEventListener("keydown", closeOnKey, true);
    };
    const closeOnPointer = (closeEvent) => {
      if (menu.contains(closeEvent.target)) return;
      closeMenu();
    };
    const closeOnKey = (keyEvent) => {
      if (keyEvent.key === "Escape") closeMenu();
    };
    setTimeout(() => {
      window.addEventListener("pointerdown", closeOnPointer, true);
      window.addEventListener("click", closeOnPointer, true);
      window.addEventListener("contextmenu", closeOnPointer, true);
      window.addEventListener("keydown", closeOnKey, true);
    }, 0);
  }

  function openTimelineSceneCard(segment, event) {
    if (event?.shiftKey) {
      openStoryboardBuilderFromProject({ focusSceneId: segment.id });
    } else {
      openLyricReviewModal({ singleSceneId: segment.id });
    }
  }

  async function restoreVideoForSegment(segment, options = {}) {
    if (!segment) return;
    const projectFolder = String(state.projectFolder || projectInput.value || "").trim();
    if (!projectFolder) {
      toast("Create or load a Builder project before restoring a scene video.", true);
      return;
    }
    const sourcePath = await pickPath("video", { value: "" });
    if (!sourcePath) return;
    const sceneNumber = sceneSlotNumber(segment);
    const expectedDuration = Math.max(0.1, timelineSegmentDuration(segment) || 0);
    const sceneLabel = sceneDisplayName(segment, segmentIndexInfo(segment).index);
    const restorePayload = (confirmDurationMismatch = false) => ({
      project_folder: projectFolder,
      scene_number: sceneNumber,
      source_path: sourcePath,
      expected_duration: expectedDuration,
      duration_tolerance: 0.5,
      confirm_duration_mismatch: confirmDurationMismatch || options.askDurationMismatch === false,
    });
    let data;
    try {
      data = await postJson("/vrgdg/music_builder/restore_scene_video", restorePayload(false), 120000);
      if (data.needs_confirmation) {
        const selectedDuration = Number(data.duration || 0);
        const sceneDuration = Number(data.expected_duration || expectedDuration || 0);
        const proceed = options.askDurationMismatch === false ? true : window.confirm(
          `${sceneLabel}: the selected video duration does not match this timeline scene.\n\n`
          + `Scene length: ${sceneDuration.toFixed(2)}s\n`
          + `Selected video: ${selectedDuration ? selectedDuration.toFixed(2) : "unknown"}s\n\n`
          + "Restore it anyway?"
        );
        if (!proceed) return;
        data = await postJson("/vrgdg/music_builder/restore_scene_video", restorePayload(true), 120000);
      }
    } catch (error) {
      toast(`Could not restore scene video:\n${String(error?.message || error)}`, true);
      return;
    }
    if (!data.video_path) {
      toast("No video was restored.", true);
      return;
    }
    pushHistory();
    if (data.backup_path) {
      if (!Array.isArray(segment.video_backup_paths)) segment.video_backup_paths = [];
      if (!segment.video_backup_paths.some((item) => mediaPathKey(item) === mediaPathKey(data.backup_path))) {
        segment.video_backup_paths.push(data.backup_path);
      }
    }
    if (data.backup_thumbnail_path) {
      if (!Array.isArray(segment.video_backup_thumbnail_paths)) segment.video_backup_thumbnail_paths = [];
      if (!segment.video_backup_thumbnail_paths.some((item) => mediaPathKey(item) === mediaPathKey(data.backup_thumbnail_path))) {
        segment.video_backup_thumbnail_paths.push(data.backup_thumbnail_path);
      }
    }
    activateSegmentVideoPath(segment, data.video_path, data.thumbnail_path || "");
    segment.video_folder = data.video_folder || collectedSceneVideoFolder();
    segment.video_status = "done";
    segment.preview_mode = "video";
    segment.video_cache_bust = Date.now();
    syncPreview(segment);
    syncInspector();
    render();
    await autoSaveSessionQuiet("scene video restored manually");
    const durationLine = data.duration ? `\nDuration: ${Number(data.duration).toFixed(2)}s` : "";
    toast(`Restored video for ${sceneLabel}.\n${data.video_path}${durationLine}`);
  }

  async function copyBaseSceneAsOverlay(sourceSegment) {
    if (!sourceSegment || segmentTrack(sourceSegment) === "overlay") return;
    const segment = typeof structuredClone === "function"
      ? structuredClone(sourceSegment)
      : JSON.parse(JSON.stringify(sourceSegment));
    segment.id = `seg_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
    segment.track = "overlay";
    segment.source = "overlay_copy";
    segment.overlay_slot_number = nextOverlaySlotNumber();
    delete segment.scene_slot_number;
    delete segment.slot_number;
    segment.label = `${sourceSegment.label || "Scene"} - Insert`;
    segment.video_history = segment.video_path ? [segment.video_path] : [];
    segment.video_thumbnail_history = segment.video_path ? [segment.video_thumbnail_path || ""] : [];
    segment.video_backup_paths = [];
    segment.video_backup_thumbnail_paths = [];
    segment.video_history_index = segment.video_path ? 0 : -1;
    normalizeOverlayClip(segment);
    pushHistory();
    state.overlayTrack = normalizeOverlayTrackState({ enabled: true });
    state.overlaySegments.push(segment);
    sortSegments(state.overlaySegments);
    state.duration = Math.max(Number(state.duration || 0), Number(segment.end || 0));
    setActiveSegment(segment);
    await autoSaveSessionQuiet("base scene copied as insert track");
    toast(`${sourceSegment.label || "Scene"} copied to the Overlay Track.`);
  }

  return {
    baseSceneVideoTrimKind, chooseRenderedSceneTrimAtPlayhead, closeBaseTimelineGap,
    closeTimelineGapsFromMenu, loadDirtyLatentBadges, openAudioContextMenu, openDirectorNoteContextMenu,
    openSceneOptions, openSegmentContextMenu, openSnapSceneEdgeMenu, openTimelineSceneCard,
    snapAllSceneStartsToNearestBeats, snapSceneEdgeToNearestBeat,
  };
}
