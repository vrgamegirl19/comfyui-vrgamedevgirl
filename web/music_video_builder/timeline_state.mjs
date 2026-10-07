import { audioUrl, refreshEditorThumbnailUrl } from "./comfy_api.mjs";
import { TIMELINE_MARKER_MIN_WIDTH } from "./constants.mjs";
import { normalizeProjectVideoEngine, toast } from "./controls.mjs";
import {
  cloneMiniMaxH3Settings,
  normalizeMiniMaxH3ContinuityMode,
  normalizeMiniMaxH3LocationTransitionPreset,
  normalizeMiniMaxH3Mode,
  normalizeMiniMaxH3SceneImageUse,
  normalizeMiniMaxH3StartFrameCharacterInfluence,
  normalizeMiniMaxH3VideoPurpose,
  normalizeMiniMaxSpeakerAssignments,
} from "./minimax_h3.mjs";
import {
  cloneI2VVideoSettings,
  cloneKrea2TwoPassSettings,
  cloneNBImageSettings,
  cloneZImageSettings,
} from "./model_settings.mjs";
import { isInstrumentalLyricText } from "./prompt_text.mjs";
import {
  createUniqueSegmentId,
  issuedSegmentIds,
  newSegment,
  normalizeVideoPromptOrigin,
  sortSegments,
} from "./segments.mjs";
import { selectedSegmentVideoPath } from "./selection_preview.mjs";

function normalizedOverlaySlotNumber(value) {
  const number = Number.parseInt(value, 10);
  return Number.isFinite(number) && number >= 10001 ? number : 0;
}

export function normalizeBatchScope(sceneScope = "all") {
  return ["all", "selected", "from_selected"].includes(sceneScope) ? sceneScope : "all";
}

export function batchScopeLabel(sceneScope = "all") {
  const scope = normalizeBatchScope(sceneScope);
  if (scope === "selected") return "selected scenes";
  if (scope === "from_selected") return "from selected scene";
  return "all scenes";
}

export function batchEmptyMessage(sceneScope = "all") {
  const scope = normalizeBatchScope(sceneScope);
  if (scope === "selected") return "No selected scenes found. Turn on Select Multi and choose scenes first.";
  if (scope === "from_selected") return "No selected clip found. Select the clip you want to start from first.";
  return "No scenes found. Add or load scenes first.";
}

export function timelineAudioStartForSegment(segment) {
  if (!segment) return 0;
  return String(segment.custom_audio_path || "").trim() ? audioTimelineStart(segment) : Math.max(0, Number(segment.start || 0));
}

export function timelineAudioDurationForSegment(segment) {
  if (!segment) return 0;
  return String(segment.custom_audio_path || "").trim() ? audioChunkDuration(segment) : timelineSegmentDuration(segment);
}

export function timelineAudioEndForSegment(segment) {
  return timelineAudioStartForSegment(segment) + timelineAudioDurationForSegment(segment);
}

export function audioTimelineStart(segment) {
  if (!segment) return 0;
  const value = Number(segment.custom_audio_timeline_start);
  return Number.isFinite(value) ? value : Number(segment.start || 0);
}

export function audioSourceStart(segment) {
  if (!segment) return 0;
  const value = Number(segment.custom_audio_source_start);
  return Number.isFinite(value) ? Math.max(0, value) : 0;
}

export function audioChunkDuration(segment) {
  if (!segment) return 0;
  const value = Number(segment.custom_audio_duration);
  if (Number.isFinite(value) && value > 0) return value;
  return Math.max(0, Number(segment.end || 0) - Number(segment.start || 0));
}

export function timelineSegmentDuration(segment) {
  if (!segment) return 0;
  return Math.max(0, Number(segment.end || 0) - Number(segment.start || 0));
}

export function audioTimelineEnd(segment) {
  return audioTimelineStart(segment) + audioChunkDuration(segment);
}

export function mediaPathKey(path) {
  return String(path || "").trim().replace(/\\/g, "/").replace(/\/+/g, "/").toLowerCase();
}

export function isBackupSceneVideoPath(path) {
  return mediaPathKey(path).includes("/rendered_scene_videos_backup/");
}

export function normalizeSegmentVideoHistory(segment) {
  if (!segment) return [];
  const currentPath = String(segment.video_path || "").trim();
  const currentKey = mediaPathKey(currentPath);
  const previousHistory = Array.isArray(segment.video_history) ? segment.video_history : [];
  const previousThumbnailHistory = Array.isArray(segment.video_thumbnail_history) ? segment.video_thumbnail_history : [];
  const previousIndex = Number(segment.video_history_index);
  const previousSelectedPath = Number.isFinite(previousIndex) && previousIndex >= 0 ? String(previousHistory[previousIndex] || "").trim() : "";
  const previousSelectedKey = mediaPathKey(previousSelectedPath);
  const thumbnailByVideoKey = new Map();
  previousHistory.forEach((path, index) => {
    const key = mediaPathKey(path);
    const thumbnail = String(previousThumbnailHistory[index] || "").trim();
    if (key && thumbnail) thumbnailByVideoKey.set(key, thumbnail);
  });
  if (Array.isArray(segment.video_backup_paths) && Array.isArray(segment.video_backup_thumbnail_paths)) {
    segment.video_backup_paths.forEach((path, index) => {
      const key = mediaPathKey(path);
      const thumbnail = String(segment.video_backup_thumbnail_paths[index] || "").trim();
      if (key && thumbnail) thumbnailByVideoKey.set(key, thumbnail);
    });
  }
  if (currentKey && String(segment.video_thumbnail_path || "").trim()) {
    thumbnailByVideoKey.set(currentKey, String(segment.video_thumbnail_path || "").trim());
  }
  const seen = new Set();
  const cleaned = [];
  const candidates = [
    ...(Array.isArray(segment.video_backup_paths) ? segment.video_backup_paths : []),
    ...(Array.isArray(segment.video_history) ? segment.video_history : []),
    currentPath,
  ];
  for (const item of candidates) {
    const path = String(item || "").trim();
    const key = mediaPathKey(path);
    if (!path || !key || seen.has(key)) continue;
    // MiniMax H3 stage 1/2 source files are scratch outputs. The copied
    // rendered_scene_videos_backup versions are the review entries; keeping
    // both makes one render appear twice in timeline history.
    if (
      key.includes("/vrgdg_minimaxh3/")
      && /_stage[12][^/]*\.mp4$/i.test(key)
      && !isBackupSceneVideoPath(path)
    ) continue;
    seen.add(key);
    cleaned.push(path);
  }
  segment.video_backup_paths = cleaned.filter(isBackupSceneVideoPath);
  segment.video_history = cleaned;
  segment.video_thumbnail_history = cleaned.map((path) => thumbnailByVideoKey.get(mediaPathKey(path)) || "");
  if (!cleaned.length) {
    segment.video_history_index = -1;
  } else if (previousSelectedKey) {
    const selectedIndex = cleaned.findIndex((item) => mediaPathKey(item) === previousSelectedKey);
    segment.video_history_index = selectedIndex >= 0 ? selectedIndex : Math.max(0, Math.min(cleaned.length - 1, Number(segment.video_history_index || 0)));
  } else {
    const currentIndex = currentKey ? cleaned.findIndex((item) => mediaPathKey(item) === currentKey) : -1;
    segment.video_history_index = currentIndex >= 0 ? currentIndex : Math.max(0, Math.min(cleaned.length - 1, Number(segment.video_history_index || 0)));
  }
  segment.video_thumbnail_path = segment.video_thumbnail_history[segment.video_history_index] || "";
  return cleaned;
}

export function activateSegmentVideoPath(segment, videoPath, thumbnailPath = "") {
  if (!segment || !String(videoPath || "").trim()) return;
  segment.video_path = String(videoPath || "").trim();
  if (String(thumbnailPath || "").trim()) {
    segment.video_thumbnail_path = String(thumbnailPath || "").trim();
    refreshEditorThumbnailUrl(segment.video_thumbnail_path);
  }
  normalizeSegmentVideoHistory(segment);
  const selectedIndex = segment.video_history.findIndex((item) => mediaPathKey(item) === mediaPathKey(segment.video_path));
  if (selectedIndex >= 0) segment.video_history_index = selectedIndex;
  segment.video_thumbnail_path = segment.video_thumbnail_history[segment.video_history_index] || segment.video_thumbnail_path || "";
}

export function normalizeTimelineRange(range) {
  const input = range && typeof range === "object" ? range : {};
  const start = Number(input.in ?? input.start);
  const end = Number(input.out ?? input.end);
  const hasStart = Number.isFinite(start);
  const hasEnd = Number.isFinite(end);
  if (!hasStart && !hasEnd) return { in: null, out: null };
  if (hasStart && hasEnd && end < start) return { in: Math.max(0, end), out: Math.max(0, start) };
  return {
    in: hasStart ? Math.max(0, start) : null,
    out: hasEnd ? Math.max(0, end) : null,
  };
}

export function normalizeTimelineMarkers(markers) {
  return (Array.isArray(markers) ? markers : [])
    .map((marker) => {
      const start = Math.max(0, Number(marker?.start || 0));
      const rawEnd = Number(marker?.end);
      const end = Number.isFinite(rawEnd) && rawEnd > start ? rawEnd : null;
      return {
        id: String(marker?.id || `mark_${Date.now()}_${Math.floor(Math.random() * 10000)}`),
        start,
        end,
        type: String(marker?.type || "note").trim() || "note",
        label: String(marker?.label || "Timeline note").trim() || "Timeline note",
        note: String(marker?.note || "").trim(),
      };
    })
    .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
}

export function markerEnd(marker) {
  return Number.isFinite(Number(marker?.end)) ? Number(marker.end) : Number(marker?.start || 0);
}

export function markerOverlapsRange(marker, start, end) {
  const markerStart = Number(marker?.start || 0);
  const markerStop = Math.max(markerStart + 0.05, markerEnd(marker));
  return markerStop > start && markerStart < end;
}

export function markerContext(marker) {
  return {
    id: marker.id || "",
    start: Number(marker.start || 0),
    end: marker.end == null ? null : Number(marker.end),
    duration: marker.end == null ? 0 : Math.max(0, Number(marker.end || 0) - Number(marker.start || 0)),
    type: marker.type || "note",
    label: marker.label || "Timeline note",
    note: marker.note || "",
  };
}

function isInternalApprovedImagePath(path) {
  return /(^|[\\/])zimage_approved([\\/]|$)/i.test(String(path || ""));
}

function importedSrtTextFromSegment(segment) {
  if (!segment || typeof segment !== "object") return "";
  const candidates = [
    segment.lyric_text,
    segment.text,
    segment.caption,
    segment.subtitle,
    segment.srt_text,
    segment.prompt,
    segment.notes,
  ];
  return String(candidates.find((value) => String(value || "").trim()) || "").trim();
}

export function createTimelineState({
  audio, audioInput, autoSaveSessionQuiet, cancelPreviewPlayStart, currentVideoMode, drawWaveform,
  idLoraTrimModeButton, inspector, leftPanelToggle, leftResizeHandle, main, miniMaxH3SettingsForSegment,
  multiSelectButton, playButton, previewVideo, render, renderSegments, rightPanelToggle, rightResizeHandle, sceneAudio, sceneDisplayName, setActiveSegment, setGlobalPlaybackTime, shell,
  segmentList, silentTimeline, state, syncInspector, timelineViewport, updateAudioScrubbers, updateGlobalAudioMuteButton,
  wizardVideoSettings,
}) {
  function allEditableSegments() {
    const base = Array.isArray(state.segments) ? state.segments : [];
    const overlays = Array.isArray(state.overlaySegments) ? state.overlaySegments : [];
    return [...base, ...overlays].filter((segment) => segment && typeof segment === "object" && !Array.isArray(segment));
  }

  function segmentTrack(segment) {
    if (!segment) return "base";
    if (state.overlaySegments.some((item) => item.id === segment.id)) return "overlay";
    return "base";
  }

  function segmentIndexInfo(segment) {
    if (!segment) return { track: "base", index: -1 };
    const baseIndex = state.segments.findIndex((item) => item.id === segment.id);
    if (baseIndex >= 0) return { track: "base", index: baseIndex };
    const overlayIndex = state.overlaySegments.findIndex((item) => item.id === segment.id);
    if (overlayIndex >= 0) return { track: "overlay", index: overlayIndex };
    return { track: segment.track === "overlay" ? "overlay" : "base", index: -1 };
  }

  function nextOverlaySlotNumber() {
    const used = state.overlaySegments
      .map((segment) => normalizedOverlaySlotNumber(segment?.overlay_slot_number || segment?.scene_slot_number || segment?.slot_number))
      .filter(Boolean);
    return Math.max(10000, ...used) + 1;
  }

  function assignOverlaySlotNumbers() {
    const used = new Set();
    let nextSlot = Math.max(
      10000,
      ...state.overlaySegments
        .map((segment) => normalizedOverlaySlotNumber(segment?.overlay_slot_number || segment?.scene_slot_number || segment?.slot_number))
        .filter(Boolean)
    ) + 1;
    for (const [index, segment] of state.overlaySegments.entries()) {
      if (!segment) continue;
      let slot = normalizedOverlaySlotNumber(segment.overlay_slot_number || segment.scene_slot_number || segment.slot_number);
      if (!slot || used.has(slot)) {
        slot = Math.max(nextSlot, 10000 + index + 1);
        while (used.has(slot)) slot += 1;
        nextSlot = slot + 1;
      }
      segment.overlay_slot_number = slot;
      used.add(slot);
    }
  }

  function sceneSlotNumber(segment) {
    const info = segmentIndexInfo(segment);
    if (info.track === "overlay") {
      return normalizedOverlaySlotNumber(segment?.overlay_slot_number || segment?.scene_slot_number || segment?.slot_number)
        || (10000 + Math.max(0, info.index) + 1);
    }
    return Math.max(0, info.index) + 1;
  }

  function activeSegment() {
    return allEditableSegments().find((segment) => segment.id === state.activeId) || null;
  }

  function selectedSegmentsForBatch({ baseOnly = false } = {}) {
    const ids = new Set(Array.isArray(state.selectedSegmentIds) ? state.selectedSegmentIds : []);
    const items = allEditableSegments().filter((segment) => ids.has(segment.id));
    return baseOnly ? items.filter((segment) => segmentTrack(segment) !== "overlay") : items;
  }

  function isSegmentMultiSelected(segment) {
    return Boolean(segment?.id && Array.isArray(state.selectedSegmentIds) && state.selectedSegmentIds.includes(segment.id));
  }

  function updateMultiSelectButton() {
    const count = selectedSegmentsForBatch().length;
    multiSelectButton.textContent = state.multiSelectMode ? `Multi ${count}` : "Select Multi";
    multiSelectButton.style.background = state.multiSelectMode ? "#0e7490" : "#27272a";
    multiSelectButton.style.borderColor = state.multiSelectMode ? "#22d3ee" : "#3f3f46";
    multiSelectButton.style.color = state.multiSelectMode ? "#ecfeff" : "#f4f4f5";
  }

  function toggleMultiSegmentSelection(segment) {
    if (!segment?.id) return;
    const ids = new Set(Array.isArray(state.selectedSegmentIds) ? state.selectedSegmentIds : []);
    if (ids.has(segment.id)) ids.delete(segment.id);
    else ids.add(segment.id);
    state.selectedSegmentIds = Array.from(ids);
    if (ids.has(segment.id)) {
      state.activeId = segment.id;
      state.activeTrack = segmentTrack(segment);
    } else if (state.activeId === segment.id) {
      const next = allEditableSegments().find((item) => ids.has(item.id));
      state.activeId = next?.id || "";
      state.activeTrack = next ? segmentTrack(next) : state.activeTrack || "base";
    }
    syncInspector();
    render();
  }

  // Ctrl-clicking a scene after another ctrl-clicked scene selects every scene between them on the same track.
  // Returns false when there is no earlier ctrl-clicked scene to range from, so the click is a normal toggle.
  function selectSegmentRangeFromAnchor(segment) {
    const anchor = allEditableSegments().find((item) => item.id === state.rangeAnchorId);
    if (!anchor || !segment?.id || anchor.id === segment.id) return false;
    const anchorInfo = segmentIndexInfo(anchor);
    const targetInfo = segmentIndexInfo(segment);
    if (anchorInfo.track !== targetInfo.track || anchorInfo.index < 0 || targetInfo.index < 0) return false;
    const track = targetInfo.track === "overlay" ? state.overlaySegments : state.segments;
    const low = Math.min(anchorInfo.index, targetInfo.index);
    const high = Math.max(anchorInfo.index, targetInfo.index);
    const ids = new Set(Array.isArray(state.selectedSegmentIds) ? state.selectedSegmentIds : []);
    for (let index = low; index <= high; index += 1) if (track[index]?.id) ids.add(track[index].id);
    state.selectedSegmentIds = Array.from(ids);
    state.activeId = segment.id;
    state.activeTrack = targetInfo.track;
    syncInspector();
    render();
    return true;
  }

  function handleSegmentPick(segment, event = null) {
    const ctrlPressed = Boolean(event?.ctrlKey || event?.metaKey);
    if (ctrlPressed) {
      const enteringMultiSelect = !state.multiSelectMode;
      if (!enteringMultiSelect && !isSegmentMultiSelected(segment) && selectSegmentRangeFromAnchor(segment)) {
        state.rangeAnchorId = segment.id;
        return;
      }
      state.rangeAnchorId = segment?.id || "";
      if (enteringMultiSelect) {
        // The scene we are already on joins the selection, so ctrl-clicking another scene adds to it.
        const current = activeSegment();
        state.selectedSegmentIds = current?.id ? [current.id] : [];
      }
      state.multiSelectMode = true;
      state.modifierMultiSelectMode = true;
      if (enteringMultiSelect && segment?.id && segment.id === state.activeId) {
        // Ctrl-clicking the scene we are on keeps it selected instead of emptying the selection.
        syncInspector();
        render();
      } else {
        toggleMultiSegmentSelection(segment);
      }
    } else if (state.multiSelectMode) {
      if (state.modifierMultiSelectMode) {
        state.multiSelectMode = false;
        state.modifierMultiSelectMode = false;
        state.rangeAnchorId = "";
        state.selectedSegmentIds = [];
        setActiveSegment(segment);
        selectSegmentGlobalAudioStart(segment);
      } else {
        toggleMultiSegmentSelection(segment);
      }
    } else {
      setActiveSegment(segment);
      selectSegmentGlobalAudioStart(segment);
    }
  }

  function applyImageSettingsToMultiSelection(kind, settings) {
    if (!state.multiSelectMode) return 0;
    const targets = selectedSegmentsForBatch();
    if (targets.length <= 1) return 0;
    for (const segment of targets) {
      if (kind === "zimage") {
        segment.use_scene_zimage_settings = true;
        segment.zimage_settings = cloneZImageSettings(settings);
      } else if (kind === "ernie_image") {
        segment.use_scene_ernie_image_settings = true;
        segment.ernie_image_settings = { ...settings, loras: Array.isArray(settings.loras) ? settings.loras.map((item) => ({ ...item })) : [] };
      } else if (kind === "krea2_2pass") {
        segment.use_scene_krea2_2pass_settings = true;
        segment.krea2_2pass_settings = cloneKrea2TwoPassSettings(settings);
      } else if (kind === "flux_klein") {
        segment.use_scene_flux_klein_settings = true;
        segment.flux_klein_settings = { ...settings, loras: Array.isArray(settings.loras) ? settings.loras.map((item) => ({ ...item })) : [] };
      } else if (kind === "nano_banana") {
        segment.use_scene_nb_image_settings = true;
        segment.nb_image_settings = cloneNBImageSettings(settings);
      }
    }
    return targets.length;
  }

  function hasMultiSceneBatchSelection() {
    return state.multiSelectMode && selectedSegmentsForBatch().length > 1;
  }

  function batchScopeChoices() {
    const selectedCount = selectedSegmentsForBatch().length;
    const active = activeSegment();
    const choices = [
      {
        value: "all",
        label: "All scenes",
        description: "Run the normal batch across every scene.",
      },
    ];
    if (active) {
      choices.push({
        value: "from_selected",
        label: `From ${sceneDisplayName(active, segmentIndexInfo(active).index)}`,
        description: "Start at the currently selected clip and run every clip after it in timeline order.",
      });
    }
    if (selectedCount > 1) {
      choices.push({
        value: "selected",
        label: `Selected scenes only (${selectedCount})`,
        description: "Only run the scenes currently selected with Select Multi. Scenes do not need to be next to each other.",
      });
    }
    return choices.length > 1 ? choices : [];
  }

  function batchTargetItems(sceneScope = "all", { baseOnly = false } = {}) {
    const scope = normalizeBatchScope(sceneScope);
    let source = scope === "selected" ? selectedSegmentsForBatch({ baseOnly }) : allEditableSegments();
    if (baseOnly) source = source.filter((segment) => segmentTrack(segment) !== "overlay");
    if (scope === "from_selected") {
      const active = activeSegment();
      if (!active) return [];
      const activeStart = audioTimelineStart(active);
      const activeTrack = segmentTrack(active);
      const activeIndex = segmentIndexInfo(active).index;
      source = source.filter((segment) => {
        const start = audioTimelineStart(segment);
        if (start > activeStart + 0.001) return true;
        if (Math.abs(start - activeStart) > 0.001) return false;
        if (segment.id === active.id) return true;
        if (segmentTrack(segment) !== activeTrack) return true;
        return segmentIndexInfo(segment).index >= activeIndex;
      });
    }
    return source.map((segment) => ({ segment, index: segmentIndexInfo(segment).index }));
  }

  function applyVideoSettingsToMultiSelection(settings) {
    if (!state.multiSelectMode) return 0;
    const targets = selectedSegmentsForBatch();
    if (targets.length <= 1) return 0;
    for (const segment of targets) {
      segment.use_scene_i2v_video_settings = true;
      segment.i2v_video_settings = cloneI2VVideoSettings(settings);
    }
    return targets.length;
  }

  function usingSceneAudioMode() {
    return state.segments.some((segment) => String(segment.custom_audio_path || "").trim());
  }

  function segmentUsesRenderedTimelineAudio(segment) {
    if (!segment || !String(selectedSegmentVideoPath(segment) || "").trim()) return false;
    if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
      return miniMaxH3SettingsForSegment(segment).audio_mode === "built_in_audio";
    }
    return currentVideoMode() === "id_lora";
  }

  function usingRenderedSceneAudioMode() {
    return !currentProjectAudioPath() && state.segments.some((segment) => segmentUsesRenderedTimelineAudio(segment));
  }

  function timelineAudioPathForSegment(segment) {
    if (!segment) return "";
    const customAudioPath = String(segment.custom_audio_path || "").trim();
    if (customAudioPath) return customAudioPath;
    if (!currentProjectAudioPath() && segmentUsesRenderedTimelineAudio(segment)) return String(selectedSegmentVideoPath(segment) || "").trim();
    if (usingSceneAudioMode()) return currentProjectAudioPath();
    return "";
  }

  function timelineAudioSourceStartForSegment(segment) {
    if (!segment) return 0;
    if (String(segment.custom_audio_path || "").trim()) return audioSourceStart(segment);
    if (!currentProjectAudioPath() && segmentUsesRenderedTimelineAudio(segment)) return 0;
    return Math.max(0, Number(segment.start || 0));
  }

  function timelineAudioSegmentAtTime(time) {
    const current = Number(time || 0);
    return state.segments.find((segment, index) => {
      if (!timelineAudioPathForSegment(segment)) return false;
      const start = timelineAudioStartForSegment(segment);
      const end = timelineAudioEndForSegment(segment);
      const isLast = index === state.segments.length - 1;
      return current >= start && (current < end || (isLast && current <= end));
    }) || null;
  }

  function audioSegmentAtTime(time) {
    const current = Number(time || 0);
    return state.segments.find((segment, index) => {
      if (!String(segment.custom_audio_path || "").trim()) return false;
      const start = audioTimelineStart(segment);
      const end = audioTimelineEnd(segment);
      const isLast = index === state.segments.length - 1;
      return current >= start && (current < end || (isLast && current <= end));
    }) || null;
  }

  function timelineDuration() {
    const segmentEnd = state.segments.reduce((max, segment) => Math.max(max, Number(segment.end || 0)), 0);
    const overlayEnd = state.overlaySegments.reduce((max, segment) => Math.max(max, Number(segment.end || 0)), 0);
    const sceneAudioEnd = state.segments.reduce((max, segment) => Math.max(max, segment.custom_audio_path ? audioTimelineEnd(segment) : 0), 0);
    const markerEnd = normalizeTimelineMarkers(state.timelineMarkers).reduce((max, marker) => Math.max(max, Number(marker.end ?? marker.start ?? 0)), 0);
    const range = normalizeTimelineRange(state.selectedTimelineRange);
    const rangeEnd = Number.isFinite(Number(range.out)) ? Number(range.out) : 0;
    const audioEnd = loadedGlobalAudioDuration();
    if (audioEnd > 0) return audioEnd;
    return Math.max(segmentEnd, overlayEnd, sceneAudioEnd, markerEnd, rangeEnd, Number(state.duration || 0));
  }

  function loadedGlobalAudioDuration() {
    const analyzedDuration = Number(state.audioDuration);
    if (Number.isFinite(analyzedDuration) && analyzedDuration > 0) return analyzedDuration;
    const mediaDuration = Number(audio.duration);
    return Number.isFinite(mediaDuration) && mediaDuration > 0 ? mediaDuration : 0;
  }

  function playbackDuration() {
    if (currentProjectAudioPath()) {
      const audioEnd = loadedGlobalAudioDuration();
      if (audioEnd > 0) return audioEnd;
    }
    return timelineDuration();
  }

  function enforceAudioTimelineEnd() {
    const audioEnd = loadedGlobalAudioDuration();
    if (!(audioEnd > 0)) return { trimmed: 0, removed: 0 };
    let trimmed = 0;
    let removed = 0;
    const clampTrack = (segments) => {
      const kept = [];
      for (const segment of segments) {
        const start = Math.max(0, Number(segment.start || 0));
        if (start >= audioEnd - 0.001) {
          removed += 1;
          continue;
        }
        const originalEnd = Math.max(start, Number(segment.end || start));
        const end = Math.min(originalEnd, audioEnd);
        if (end <= start + 0.001) {
          removed += 1;
          continue;
        }
        if (end < originalEnd - 0.001) trimmed += 1;
        segment.start = start;
        segment.end = end;
        kept.push(segment);
      }
      return kept;
    };
    state.segments = clampTrack(state.segments);
    state.overlaySegments = clampTrack(state.overlaySegments);
    state.duration = audioEnd;
    if (!state.segments.some((segment) => segment.id === state.activeId) && !state.overlaySegments.some((segment) => segment.id === state.activeId)) {
      state.activeId = state.segments[state.segments.length - 1]?.id || state.overlaySegments[state.overlaySegments.length - 1]?.id || "";
      state.activeTrack = state.segments.some((segment) => segment.id === state.activeId) ? "base" : "overlay";
    }
    return { trimmed, removed };
  }

  function selectedTimelineRangeInfo() {
    const range = normalizeTimelineRange(state.selectedTimelineRange);
    if (!Number.isFinite(Number(range.in)) || !Number.isFinite(Number(range.out)) || Number(range.out) <= Number(range.in)) return null;
    const start = Number(range.in);
    const end = Number(range.out);
    return {
      start,
      end,
      duration: end - start,
      overlapping_scenes: allEditableSegments()
        .filter((segment) => Number(segment.end || 0) > start && Number(segment.start || 0) < end)
        .map((segment) => ({
          id: segment.id || "",
          number: segmentIndexInfo(segment).index + 1,
          label: sceneDisplayName(segment, segmentIndexInfo(segment).index),
          start: Number(segment.start || 0),
          end: Number(segment.end || 0),
        })),
      overlapping_markers: normalizeTimelineMarkers(state.timelineMarkers)
        .filter((marker) => markerOverlapsRange(marker, start, end))
        .map(markerContext),
    };
  }

  function markerVisualDuration(marker) {
    const start = Number(marker?.start || 0);
    const actual = Math.max(0.15, markerEnd(marker) - start);
    const visual = TIMELINE_MARKER_MIN_WIDTH / Math.max(1, Number(state.pxPerSecond || 1));
    return Math.max(actual, visual);
  }

  function markerVisualEnd(marker) {
    return Number(marker?.start || 0) + markerVisualDuration(marker);
  }

  function nextFreeTimelineMarkerRange(start, duration = 4) {
    const cleanDuration = Math.max(0.15, TIMELINE_MARKER_MIN_WIDTH / Math.max(1, Number(state.pxPerSecond || 1)), Number(duration || 4));
    let nextStart = Math.max(0, Number(start || 0));
    const markers = normalizeTimelineMarkers(state.timelineMarkers);
    let changed = true;
    let guard = 0;
    while (changed && guard < 200) {
      changed = false;
      guard += 1;
      const nextEnd = nextStart + cleanDuration;
      for (const marker of markers) {
        const markerStart = Number(marker.start || 0);
        const markerStop = Math.max(markerStart + 0.15, markerVisualEnd(marker));
        if (nextEnd > markerStart && nextStart < markerStop) {
          nextStart = markerStop + 0.05;
          changed = true;
        }
      }
    }
    return { start: Number(nextStart.toFixed(3)), end: Number((nextStart + cleanDuration).toFixed(3)) };
  }

  function timelineMarkerBounds(markerId) {
    const markers = normalizeTimelineMarkers(state.timelineMarkers).filter((item) => item.id !== markerId);
    const marker = state.timelineMarkers.find((item) => item.id === markerId);
    const currentStart = Number(marker?.start || 0);
    let minStart = 0;
    let maxEnd = Infinity;
    for (const item of markers) {
      const itemStart = Number(item.start || 0);
      const itemEnd = Math.max(itemStart + 0.15, markerVisualEnd(item));
      if (itemEnd <= currentStart) minStart = Math.max(minStart, itemEnd + 0.05);
      if (itemStart >= currentStart) maxEnd = Math.min(maxEnd, itemStart - 0.05);
    }
    return { minStart, maxEnd };
  }

  function clampTimelineMarkerToNonOverlap(marker, desiredStart, desiredEnd) {
    const minDuration = Math.max(0.15, TIMELINE_MARKER_MIN_WIDTH / Math.max(1, Number(state.pxPerSecond || 1)));
    const bounds = timelineMarkerBounds(marker.id);
    const duration = Math.max(minDuration, Number(desiredEnd || 0) - Number(desiredStart || 0));
    let start = Math.max(bounds.minStart, Number(desiredStart || 0));
    let end = start + duration;
    if (Number.isFinite(bounds.maxEnd) && end > bounds.maxEnd) {
      end = Math.max(bounds.minStart + minDuration, bounds.maxEnd);
      start = Math.max(bounds.minStart, end - duration);
    }
    return {
      start: Number(Math.max(0, start).toFixed(3)),
      end: Number(Math.max(start + minDuration, end).toFixed(3)),
    };
  }

  function currentProjectAudioPath() {
    return String(audioInput.value || state.audioPath || "").trim();
  }

  function usingSceneAudioPlaybackMode() {
    return !currentProjectAudioPath() && (usingSceneAudioMode() || usingRenderedSceneAudioMode());
  }

  function activateGlobalTimelineAudioPlayback(startTime = 0) {
    const start = Math.max(0, Number(startTime || 0));
    sceneAudio.pause();
    sceneAudio.onloadedmetadata = null;
    sceneAudio.removeAttribute("src");
    sceneAudio.load();
    state.sceneAudioSegmentId = "";
    state.sceneAudioGlobalTime = start;
    state.sceneSelectionUsesGlobalAudio = true;
    seekAudioWhenReady(start);
  }

  function audioSourceDurationForScene(segment) {
    if (!segment) return 0;
    if (String(segment.custom_audio_path || "").trim()) {
      const fullDuration = Number(segment.custom_audio_full_duration);
      if (Number.isFinite(fullDuration) && fullDuration > 0) return fullDuration;
      const chunkDuration = Number(segment.custom_audio_duration);
      return Number.isFinite(chunkDuration) && chunkDuration > 0 ? chunkDuration : 0;
    }
    const loadedDuration = loadedGlobalAudioDuration();
    return Number.isFinite(loadedDuration) && loadedDuration > 0 ? loadedDuration : 0;
  }

  function seekAudioWhenReady(targetTime) {
    const time = Math.max(0, Number(targetTime || 0));
    const apply = () => {
      try {
        audio.currentTime = time;
      } catch {
        // Some browsers need metadata first; the once listener below will retry.
      }
    };
    if (audio.readyState >= 1) {
      apply();
    } else {
      audio.addEventListener("loadedmetadata", apply, { once: true });
    }
  }

  function ensureGlobalTimelineAudioSource(targetTime = audio.currentTime) {
    const path = currentProjectAudioPath();
    if (!path) return false;
    if (audio.dataset.path !== path || !audio.src) {
      const wasMuted = audio.muted;
      audio.pause();
      audio.src = audioUrl(path);
      audio.dataset.path = path;
      audio.muted = wasMuted;
      audio.load();
    }
    seekAudioWhenReady(targetTime);
    return true;
  }

  function updatePlayPauseButton() {
    const playing = isTimelinePlaying();
    playButton.textContent = playing ? "Ⅱ" : "▶";
    playButton.title = playing ? "Pause (Space)" : "Play (Space)";
  }

  function setGlobalTimelineAudioMuted(muted) {
    const value = Boolean(muted);
    audio.muted = value;
    sceneAudio.muted = value;
    updateGlobalAudioMuteButton();
  }

  function pauseAllAudio() {
    cancelPreviewPlayStart();
    stopSilentTimelinePlayback();
    audio.pause();
    sceneAudio.pause();
    updatePlayPauseButton();
  }

  function selectSegmentGlobalAudioStart(segment) {
    if (!segment) return false;
    const start = Math.max(0, Number(segment.start || 0));
    audio.pause();
    sceneAudio.pause();
    sceneAudio.removeAttribute("src");
    state.sceneAudioSegmentId = "";
    state.sceneSelectionUsesGlobalAudio = true;
    state.sceneAudioGlobalTime = start;
    if (!ensureGlobalTimelineAudioSource(start)) {
      if (usingSceneAudioPlaybackMode()) {
        state.sceneSelectionUsesGlobalAudio = false;
        setGlobalPlaybackTime(start);
        updatePlayPauseButton();
        updateAudioScrubbers();
        return true;
      }
      state.sceneSelectionUsesGlobalAudio = false;
      state.sceneAudioGlobalTime = start;
      updatePlayPauseButton();
      updateAudioScrubbers();
      return true;
    }
    seekAudioWhenReady(start);
    updatePlayPauseButton();
    updateAudioScrubbers();
    return true;
  }

  function pauseTimelineForEditing() {
    pauseAllAudio();
    if (!previewVideo.paused) previewVideo.pause();
    previewVideo.muted = false;
    updateAudioScrubbers();
  }

  function syncTimelineTrimModeButton() {
    const miniMaxBuiltInAudio = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3"
      && miniMaxH3SettingsForSegment(activeSegment()).audio_mode === "built_in_audio";
    const canUseTrimMode = currentVideoMode() === "id_lora" || miniMaxBuiltInAudio;
    if (!canUseTrimMode && state.timelineTrimEditMode) state.timelineTrimEditMode = false;
    idLoraTrimModeButton.style.display = canUseTrimMode ? "" : "none";
    idLoraTrimModeButton.textContent = state.timelineTrimEditMode ? "Trim Mode On" : "Trim Mode";
    idLoraTrimModeButton.style.borderColor = state.timelineTrimEditMode ? "#22d3ee" : "#3f3f46";
    idLoraTrimModeButton.style.background = state.timelineTrimEditMode ? "#0e7490" : "#27272a";
    idLoraTrimModeButton.style.color = state.timelineTrimEditMode ? "#ecfeff" : "#f4f4f5";
  }

  function applyLayoutSizes() {
    const left = Math.max(180, Math.min(520, Number(state.leftPanelWidth || 260)));
    const right = Math.max(280, Math.min(720, Number(state.rightPanelWidth || 360)));
    // The timeline can grow until only a strip is left for the top bar and the video window.
    const timelineMax = Math.max(300, (shell.clientHeight || window.innerHeight) - 230);
    const timelineHeight = Math.max(190, Math.min(timelineMax, Number(state.timelinePanelHeight || 300)));
    const collapsed = Boolean(state.leftPanelCollapsed);
    const rightCollapsed = Boolean(state.rightPanelCollapsed);
    state.leftPanelWidth = left;
    state.rightPanelWidth = right;
    state.timelinePanelHeight = timelineHeight;
    main.style.gridTemplateColumns = `${collapsed ? 0 : left}px ${collapsed ? 0 : 7}px minmax(0,1fr) ${rightCollapsed ? 0 : 7}px ${rightCollapsed ? 0 : right}px`;
    shell.style.gridTemplateRows = `auto minmax(0,1fr) ${timelineHeight}px`;
    // These elements are flex columns, so showing them means display:flex, not clearing the value.
    segmentList.style.display = collapsed ? "none" : "flex";
    leftResizeHandle.style.display = collapsed ? "none" : "block";
    inspector.style.display = rightCollapsed ? "none" : "flex";
    rightResizeHandle.style.display = rightCollapsed ? "none" : "block";
    // The columns belong to this grid, so they are set here too. A browser that kept an older inspector.mjs
    // (which once used columns 6 and 7) would otherwise leave the panel outside the grid.
    rightResizeHandle.style.gridColumn = "4";
    inspector.style.gridColumn = "5";
    rightPanelToggle.textContent = rightCollapsed ? "\u25C0" : "\u25B6";
    rightPanelToggle.title = rightCollapsed ? "Show the settings panel" : "Hide the settings panel";
    rightPanelToggle.style.right = rightCollapsed ? "0px" : `${right}px`;
    leftPanelToggle.textContent = collapsed ? "\u25B6" : "\u25C0";
    leftPanelToggle.title = collapsed ? "Show the Scenes, Tools and Post Processing panel" : "Hide the Scenes, Tools and Post Processing panel";
    leftPanelToggle.style.left = collapsed ? "0px" : `${left + 7}px`;
    drawWaveform();
  }

  function setTimelineZoom(value, anchorTime = currentGlobalTime()) {
    const oldZoom = Math.max(1, Number(state.pxPerSecond || 45));
    const zoom = Math.max(8, Math.min(260, Number(value || 45)));
    state.timelineZoom = zoom;
    state.pxPerSecond = zoom;
    drawWaveform();
    renderSegments();
    const previousScroll = timelineViewport.scrollLeft;
    const anchorX = anchorTime * oldZoom;
    const viewportAnchor = anchorX - previousScroll;
    timelineViewport.scrollLeft = Math.max(0, anchorTime * zoom - viewportAnchor);
    autoSaveSessionQuiet("timeline zoom changed");
  }

  function makePanelResize(handle, mode) {
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      handle.setPointerCapture?.(event.pointerId);
      const startX = event.clientX;
      const startY = event.clientY;
      const startLeft = state.leftPanelWidth;
      const startRight = state.rightPanelWidth;
      const startTimeline = state.timelinePanelHeight;
      const move = (moveEvent) => {
        if (mode === "left") {
          state.leftPanelWidth = startLeft + (moveEvent.clientX - startX);
        } else if (mode === "right") {
          state.rightPanelWidth = startRight - (moveEvent.clientX - startX);
        } else if (mode === "timeline") {
          state.timelinePanelHeight = startTimeline - (moveEvent.clientY - startY);
        }
        applyLayoutSizes();
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        state.onLayoutChanged?.();
        autoSaveSessionQuiet("layout resized");
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
  }

  function ensureSegmentRuntimeFields(segment) {
    if (!segment) return segment;
    if (!Number.isFinite(Number(segment.start))) segment.start = 0;
    if (!Number.isFinite(Number(segment.end))) segment.end = Number(segment.start || 0) + 4;
    if (Number(segment.end) <= Number(segment.start)) segment.end = Number(segment.start || 0) + 0.1;
    if (segment.label == null) segment.label = "New scene";
    if (segment.lyric_text == null) segment.lyric_text = "";
    if (segment.lyric_section == null) segment.lyric_section = "";
    if (segment.story_beat == null) segment.story_beat = "";
    if (segment.flf_start_state == null) segment.flf_start_state = "";
    if (segment.flf_transformation == null) segment.flf_transformation = "";
    if (segment.flf_end_state == null) segment.flf_end_state = "";
    if (segment.flf_carry_forward == null) segment.flf_carry_forward = "";
    segment.flf_endpoint_mode = String(segment.flf_endpoint_mode || "").trim().toLowerCase() === "custom" ? "custom" : "auto";
    if (segment.flf_custom_end_direction == null) segment.flf_custom_end_direction = "";
    if (segment.flf_motion_plan == null) segment.flf_motion_plan = "";
    if (segment.flf_end_frame_prompt == null) segment.flf_end_frame_prompt = "";
    segment.flf_end_frame_stale = Boolean(segment.flf_end_frame_stale);
    segment.flf_final_prompt_ready = Boolean(segment.flf_final_prompt_ready);
    if (!Array.isArray(segment.lyric_singers)) segment.lyric_singers = [];
    segment.lyric_shot_word_timing_enabled = Boolean(segment.lyric_shot_word_timing_enabled);
    segment.lyric_performance_mode = ["together", "cue_map"].includes(String(segment.lyric_performance_mode || "").trim())
      ? String(segment.lyric_performance_mode || "").trim()
      : "together";
    segment.lyric_cue_map = Array.isArray(segment.lyric_cue_map)
      ? segment.lyric_cue_map.map((cue) => {
        const type = ["instrumental", "vocal"].includes(String(cue?.type || "").trim()) ? String(cue.type).trim() : "vocal";
        const cueTime = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value))
          ? Math.max(0, Number(value))
          : null;
        return {
          type,
          text: String(cue?.text || "").trim(),
          action_note: String(cue?.action_note || cue?.actionNote || cue?.note || "").trim(),
          singer_id: String(cue?.singer_id || cue?.singerId || cue?.subject_id || cue?.subjectId || "").trim(),
          singer_name: String(cue?.singer_name || cue?.singerName || cue?.name || "").trim(),
          start: cueTime(cue?.start),
          end: cueTime(cue?.end),
          vocal_start: type === "vocal" ? cueTime(cue?.vocal_start) : null,
          vocal_end: type === "vocal" ? cueTime(cue?.vocal_end) : null,
        };
      }).filter((cue) => cue.type === "instrumental" || cue.text || cue.action_note)
      : [];
    segment.minimax_speaker_assignments = normalizeMiniMaxSpeakerAssignments(
      segment.minimax_speaker_assignments || segment.speaker_assignments || segment.dialogue_cues || [],
    );
    if (segment.minimax_speaker_assignments.length) {
      const filledSpeakerCues = segment.minimax_speaker_assignments.filter((cue) => cue.text);
      // Speaker assignments can seed an empty scene, but they must not replace
      // lyric text the user has manually edited in the timeline.
      if (filledSpeakerCues.length && !String(segment.lyric_text || "").trim()) {
        segment.lyric_text = filledSpeakerCues.map((cue) => cue.text).join("\n");
        segment.lyric_singers = Array.from(new Set(filledSpeakerCues.map((cue) => cue.speaker_name).filter(Boolean)));
      }
    }
    if (segment.facial_performance == null) segment.facial_performance = "";
    if (segment.facial_performance_custom == null) segment.facial_performance_custom = "";
    segment.lyric_no_lip_sync = Boolean(segment.lyric_no_lip_sync);
    segment.no_character_present = Boolean(segment.no_character_present || segment.no_subject || segment.no_visible_subject);
    if (segment.timeline_note == null) segment.timeline_note = "";
    if (segment.i2v_notes == null) segment.i2v_notes = "";
    segment.i2v_prompt_origin = normalizeVideoPromptOrigin(segment.i2v_prompt_origin);
    segment.minimax_h3_mode = normalizeMiniMaxH3Mode(segment.minimax_h3_mode);
    if (segment.minimax_h3_video_style == null) segment.minimax_h3_video_style = "";
    if (segment.minimax_h3_video_style_custom == null) segment.minimax_h3_video_style_custom = "";
    if (segment.temporal_world_effect_override == null) segment.temporal_world_effect_override = "global";
    if (segment.temporal_world_effect_custom == null) segment.temporal_world_effect_custom = "";
    segment.use_scene_minimax_h3_settings = Boolean(segment.use_scene_minimax_h3_settings);
    segment.location_continuous_shot = Boolean(segment.location_continuous_shot);
    if (segment.minimax_h3_settings != null && typeof segment.minimax_h3_settings !== "object") {
      segment.minimax_h3_settings = null;
    }
    if (segment.use_scene_minimax_h3_settings) {
      const savedSceneSettings = segment.minimax_h3_settings || {};
      const hasSavedTransitionPreset = Object.prototype.hasOwnProperty.call(savedSceneSettings, "location_transition_preset");
      const legacyTransitionPreset = normalizeMiniMaxH3LocationTransitionPreset(segment.minimax_h3_location_transition_preset);
      const legacyTransitionCustom = String(segment.minimax_h3_location_transition_custom || "").trim();
      segment.minimax_h3_settings = cloneMiniMaxH3Settings({
        ...cloneMiniMaxH3Settings(state.miniMaxH3Settings),
        ...savedSceneSettings,
        video_mode: savedSceneSettings.video_mode || segment.minimax_h3_mode,
        render_pass: savedSceneSettings.render_pass ?? (normalizeMiniMaxH3Mode(savedSceneSettings.video_mode || segment.minimax_h3_mode) === "image_reference_to_video" ? "two_pass" : state.miniMaxH3Settings.render_pass),
        location_transition_preset: hasSavedTransitionPreset ? savedSceneSettings.location_transition_preset : legacyTransitionPreset,
        location_transition_custom: hasSavedTransitionPreset ? savedSceneSettings.location_transition_custom : legacyTransitionCustom,
      });
      segment.minimax_h3_mode = segment.minimax_h3_settings.video_mode;
      if (!hasSavedTransitionPreset && (legacyTransitionPreset !== "normal" || legacyTransitionCustom)) {
        segment.minimax_h3_location_transition_preset = "normal";
        segment.minimax_h3_location_transition_custom = "";
      }
    }
    if (segment.minimax_h3_prompt == null) segment.minimax_h3_prompt = "";
    if (segment.minimax_h3_pass2_prompt == null) segment.minimax_h3_pass2_prompt = "";
    segment.minimax_h3_prompt_origin = normalizeVideoPromptOrigin(segment.minimax_h3_prompt_origin);
    if (segment.minimax_h3_reference_keys != null) {
      segment.minimax_h3_reference_keys = Array.isArray(segment.minimax_h3_reference_keys)
        ? segment.minimax_h3_reference_keys.map((key) => String(key || "").trim()).filter(Boolean).slice(0, 9)
        : null;
    }
    segment.minimax_h3_scene_image_use = normalizeMiniMaxH3SceneImageUse(
      segment.minimax_h3_scene_image_use,
      Boolean(segment.minimax_h3_use_scene_image_as_start_frame),
    );
    segment.minimax_h3_use_scene_image_as_start_frame = segment.minimax_h3_scene_image_use === "exact_start_frame";
    segment.minimax_h3_start_frame_character_influence = normalizeMiniMaxH3StartFrameCharacterInfluence(
      segment.minimax_h3_start_frame_character_influence,
    );
    segment.minimax_h3_continuity_frame_path = String(segment.minimax_h3_continuity_frame_path || "");
    segment.minimax_h3_continuity_source_video_path = String(segment.minimax_h3_continuity_source_video_path || "");
    segment.minimax_h3_continuity_source_scene_id = String(segment.minimax_h3_continuity_source_scene_id || "");
    segment.minimax_h3_continuity_mode_used = normalizeMiniMaxH3ContinuityMode(segment.minimax_h3_continuity_mode_used);
    segment.minimax_h3_continuity_image_number = Math.max(0, Math.trunc(Number(segment.minimax_h3_continuity_image_number || 0)));
    segment.minimax_h3_location_transition_preset = normalizeMiniMaxH3LocationTransitionPreset(segment.minimax_h3_location_transition_preset);
    segment.minimax_h3_location_transition_custom = String(segment.minimax_h3_location_transition_custom || "");
    segment.minimax_h3_video_references = (Array.isArray(segment.minimax_h3_video_references) ? segment.minimax_h3_video_references : [])
      .slice(0, 3)
      .map((item) => ({
        path: String(item?.path || "").trim(),
        start_seconds: Math.max(0, Number(item?.start_seconds || 0)),
        duration: Math.max(0, Number(item?.duration || 0)),
        purpose: normalizeMiniMaxH3VideoPurpose(item?.purpose),
        use_audio: Boolean(item?.use_audio),
      }));
    if (!["global", "auto", "smooth", "morph"].includes(String(segment.flf_transition_type || ""))) segment.flf_transition_type = "global";
    if (segment.id == null || !String(segment.id).trim()) segment.id = createUniqueSegmentId();
    if (!Array.isArray(segment.image_history)) segment.image_history = [];
    const approvedImagePath = String(segment.approved_image_path || "");
    segment.image_history = segment.image_history.filter((item, index, list) => {
      const path = String(item || "");
      return path && path !== approvedImagePath && !isInternalApprovedImagePath(path) && list.indexOf(item) === index;
    });
    if (!Number.isFinite(Number(segment.image_history_index))) segment.image_history_index = segment.image_history.length ? segment.image_history.length - 1 : -1;
    segment.image_history_index = segment.image_history.length
      ? Math.max(0, Math.min(segment.image_history.length - 1, Number(segment.image_history_index || 0)))
      : -1;
    if (segment.enhance_notes == null) segment.enhance_notes = "";
    if (segment.enhance_prompt == null) segment.enhance_prompt = "";
    if (segment.custom_audio_path == null) segment.custom_audio_path = "";
    if (segment.custom_audio_name == null) segment.custom_audio_name = "";
    if (!Number.isFinite(Number(segment.custom_audio_duration))) segment.custom_audio_duration = 0;
    if (!Number.isFinite(Number(segment.custom_audio_full_duration))) segment.custom_audio_full_duration = Number(segment.custom_audio_duration || 0);
    if (!Number.isFinite(Number(segment.custom_audio_timeline_start))) segment.custom_audio_timeline_start = Number(segment.start || 0);
    if (!Number.isFinite(Number(segment.custom_audio_source_start))) segment.custom_audio_source_start = 0;
    if (!Array.isArray(segment.custom_audio_peaks)) segment.custom_audio_peaks = [];
    if (!Array.isArray(segment.custom_audio_beats)) segment.custom_audio_beats = [];
    if (!Array.isArray(segment.flux_image_ingredients)) {
      segment.flux_image_ingredients = [];
      if (segment.flux_subject_image_path || segment.flux_subject_image_data || segment.flux_subject_image_name) {
        segment.flux_image_ingredients.push({
          path: segment.flux_subject_image_path || "",
          data: segment.flux_subject_image_data || "",
          name: segment.flux_subject_image_name || "subject.png",
        });
      }
      if (segment.flux_location_image_path || segment.flux_location_image_data || segment.flux_location_image_name) {
        segment.flux_image_ingredients.push({
          path: segment.flux_location_image_path || "",
          data: segment.flux_location_image_data || "",
          name: segment.flux_location_image_name || "location.png",
        });
      }
    }
    if (segment.flux_notes == null) segment.flux_notes = "";
    if (segment.flux_prompt == null) segment.flux_prompt = "";
    if (segment.use_scene_zimage_settings == null) segment.use_scene_zimage_settings = false;
    if (segment.zimage_settings && typeof segment.zimage_settings !== "object") segment.zimage_settings = null;
    if (segment.use_scene_ernie_image_settings == null) segment.use_scene_ernie_image_settings = false;
    if (segment.ernie_image_settings && typeof segment.ernie_image_settings !== "object") segment.ernie_image_settings = null;
    if (segment.use_scene_krea2_2pass_settings == null) segment.use_scene_krea2_2pass_settings = false;
    if (segment.krea2_2pass_settings && typeof segment.krea2_2pass_settings !== "object") segment.krea2_2pass_settings = null;
    if (segment.use_scene_flux_klein_settings == null) segment.use_scene_flux_klein_settings = false;
    if (segment.flux_klein_settings && typeof segment.flux_klein_settings !== "object") segment.flux_klein_settings = null;
    if (segment.nb_notes == null) segment.nb_notes = "";
    if (segment.nb_prompt == null) segment.nb_prompt = "";
    if (segment.use_scene_nb_image_settings == null) segment.use_scene_nb_image_settings = false;
    if (segment.nb_image_settings && typeof segment.nb_image_settings !== "object") segment.nb_image_settings = null;
    if (segment.use_scene_i2v_video_settings == null) segment.use_scene_i2v_video_settings = false;
    if (segment.i2v_video_settings && typeof segment.i2v_video_settings !== "object") segment.i2v_video_settings = null;
    if (segment.use_scene_image_as_rtv_ref == null) segment.use_scene_image_as_rtv_ref = false;
    if (!["none", "character_anchor", "first_last_frame"].includes(String(segment.rtv_reference_behavior || ""))) {
      segment.rtv_reference_behavior = segment.use_scene_image_as_rtv_ref ? "character_anchor" : "none";
    }
    segment.use_scene_image_as_rtv_ref = segment.rtv_reference_behavior === "character_anchor";
    if (segment.first_last_frame_end_image_path == null) segment.first_last_frame_end_image_path = "";
    if (segment.first_last_frame_end_image_data == null) segment.first_last_frame_end_image_data = "";
    if (segment.first_last_frame_end_image_name == null) segment.first_last_frame_end_image_name = "";
    if (!["image", "video"].includes(segment.preview_mode)) segment.preview_mode = segment.video_path ? "video" : "image";
    if (segment.video_path == null) segment.video_path = "";
    if (segment.video_thumbnail_path == null) segment.video_thumbnail_path = "";
    if (segment.video_folder == null) segment.video_folder = "";
    if (!Array.isArray(segment.video_history)) segment.video_history = [];
    if (!Array.isArray(segment.video_thumbnail_history)) segment.video_thumbnail_history = [];
    if (!Array.isArray(segment.video_backup_paths)) segment.video_backup_paths = [];
    if (!Array.isArray(segment.video_backup_thumbnail_paths)) segment.video_backup_thumbnail_paths = [];
    if (segment.video_output == null) segment.video_output = null;
    if (segment.video_status == null) segment.video_status = segment.video_path ? "done" : "none";
    if (!segment.video_path && segment.video_output && typeof segment.video_output === "object") {
      segment.video_path = segment.video_output.path || segment.video_output.filename || "";
    }
    if (segment.video_path && !segment.video_history.includes(segment.video_path)) {
      segment.video_history.push(segment.video_path);
    }
    normalizeSegmentVideoHistory(segment);
    return segment;
  }

  function ensureAllSegmentRuntimeFields() {
    state.segments = (Array.isArray(state.segments) ? state.segments : [])
      .filter((segment) => segment && typeof segment === "object" && !Array.isArray(segment))
      .filter((segment, index) => {
        const recoveredId = String(segment.id || "").match(/^recovered_scene_(\d+)$/i);
        if (recoveredId && Number(recoveredId[1]) >= 10000) return false;
        if (segment.track === "overlay") return false;
        return index < 10000;
      })
      .map((segment) => {
        segment.track = "base";
        return ensureSegmentRuntimeFields(segment);
      });
    state.overlaySegments = (Array.isArray(state.overlaySegments) ? state.overlaySegments : [])
      .filter((segment) => segment && typeof segment === "object" && !Array.isArray(segment))
      .map((segment) => {
        segment.track = "overlay";
        return ensureSegmentRuntimeFields(segment);
      });
    const seenIds = new Set();
    const repairedIds = [];
    for (const segment of [...state.segments, ...state.overlaySegments]) {
      const originalId = String(segment.id || "").trim();
      if (!originalId || seenIds.has(originalId)) {
        const replacementId = createUniqueSegmentId();
        segment.id = replacementId;
        repairedIds.push({ originalId, replacementId, label: String(segment.label || "Scene") });
      } else {
        segment.id = originalId;
        issuedSegmentIds.add(originalId);
      }
      seenIds.add(segment.id);
    }
    if (repairedIds.length) {
      state.repairedSegmentIdCount = Number(state.repairedSegmentIdCount || 0) + repairedIds.length;
      const refs = state.fluxReferenceBuilder;
      const mapKeys = ["subject_scene_map", "performer_scene_map", "scene_map", "scene_trigger_map", "ingredients_scene_map"];
      if (refs && typeof refs === "object") {
        for (const { originalId, replacementId } of repairedIds) {
          if (!originalId) continue;
          for (const key of mapKeys) {
            const map = refs[key];
            if (map && typeof map === "object" && Object.prototype.hasOwnProperty.call(map, originalId)) {
              const value = map[originalId];
              map[replacementId] = Array.isArray(value) ? [...value] : value && typeof value === "object" ? { ...value } : value;
            }
          }
        }
      }
      console.warn("[VRGDG Music Builder] Repaired duplicate or missing timeline scene IDs:", repairedIds);
    }
    assignOverlaySlotNumbers();
    sortSegments(state.overlaySegments);
  }

  function sanitizedSessionSegments(segments = [], track = "base") {
    return (Array.isArray(segments) ? segments : [])
      .filter((segment) => segment && typeof segment === "object" && !Array.isArray(segment))
      .map((segment) => {
        segment.track = track;
        return ensureSegmentRuntimeFields(segment);
      });
  }

  function normalizeImportedSrtSegments(segments) {
    return (Array.isArray(segments) ? segments : []).map((segment, index) => {
      const normalized = ensureSegmentRuntimeFields(segment && typeof segment === "object" ? segment : newSegment(0, 4));
      const text = importedSrtTextFromSegment(normalized);
      if (text) {
        normalized.lyric_text = text;
        normalized.lyric_no_lip_sync = isInstrumentalLyricText(text);
      }
      if (!String(normalized.label || "").trim() || /^prompt\s+\d+$/i.test(String(normalized.label || ""))) {
        normalized.label = `Scene ${index + 1}`;
      }
      normalized.source = normalized.source || "srt_import";
      return normalized;
    });
  }

  function videoSettingsSegment() {
    return wizardVideoSettings.global ? null : activeSegment();
  }

  function stopSilentTimelinePlayback() {
    silentTimeline.playing = false;
    if (silentTimeline.raf) window.cancelAnimationFrame(silentTimeline.raf);
    silentTimeline.raf = 0;
  }

  function startSilentTimelinePlayback(startTime = currentGlobalTime()) {
    const maxTime = playbackDuration();
    if (!(maxTime > 0)) {
      toast("Add a scene with a positive duration before playing the timeline.", true);
      return false;
    }
    audio.pause();
    sceneAudio.pause();
    stopSilentTimelinePlayback();
    silentTimeline.startTime = Math.max(0, Math.min(maxTime, Number(startTime || 0)));
    silentTimeline.startedAt = performance.now();
    state.sceneAudioGlobalTime = silentTimeline.startTime;
    state.sceneSelectionUsesGlobalAudio = false;
    silentTimeline.playing = true;

    const tick = (now) => {
      if (!silentTimeline.playing) return;
      const elapsed = Math.max(0, (Number(now || 0) - silentTimeline.startedAt) / 1000);
      const current = Math.min(maxTime, silentTimeline.startTime + elapsed);
      state.sceneAudioGlobalTime = current;
      updateAudioScrubbers();
      if (current >= maxTime - 0.001) {
        stopSilentTimelinePlayback();
        if (!previewVideo.paused) previewVideo.pause();
        updatePlayPauseButton();
        updateAudioScrubbers();
        return;
      }
      silentTimeline.raf = window.requestAnimationFrame(tick);
    };
    silentTimeline.raf = window.requestAnimationFrame(tick);
    updatePlayPauseButton();
    updateAudioScrubbers();
    return true;
  }

  function currentGlobalTime() {
    if (silentTimeline.playing) return Number(state.sceneAudioGlobalTime || 0);
    if (usingSceneAudioPlaybackMode() && (!state.sceneSelectionUsesGlobalAudio || (sceneAudio.src && !sceneAudio.paused))) {
      return Number(state.sceneAudioGlobalTime || 0);
    }
    if (!currentProjectAudioPath()) return Number(state.sceneAudioGlobalTime || 0);
    return Number(audio.currentTime || 0);
  }

  function isTimelinePlaying() {
    return silentTimeline.playing || (audio.src && !audio.paused) || (sceneAudio.src && !sceneAudio.paused);
  }

  return {
    activateGlobalTimelineAudioPlayback, activeSegment, allEditableSegments,
    applyImageSettingsToMultiSelection, applyLayoutSizes, applyVideoSettingsToMultiSelection,
    audioSourceDurationForScene, batchScopeChoices, batchTargetItems, clampTimelineMarkerToNonOverlap,
    currentGlobalTime, currentProjectAudioPath, enforceAudioTimelineEnd, ensureAllSegmentRuntimeFields,
    ensureGlobalTimelineAudioSource, ensureSegmentRuntimeFields, handleSegmentPick,
    hasMultiSceneBatchSelection, isSegmentMultiSelected, isTimelinePlaying, loadedGlobalAudioDuration,
    makePanelResize, markerVisualEnd, nextFreeTimelineMarkerRange, nextOverlaySlotNumber,
    normalizeImportedSrtSegments, pauseAllAudio, pauseTimelineForEditing, playbackDuration,
    sanitizedSessionSegments, sceneSlotNumber, seekAudioWhenReady, segmentIndexInfo, segmentTrack,
    selectedSegmentsForBatch, selectedTimelineRangeInfo, setGlobalTimelineAudioMuted, setTimelineZoom,
    startSilentTimelinePlayback, stopSilentTimelinePlayback, syncTimelineTrimModeButton,
    timelineAudioPathForSegment, timelineAudioSegmentAtTime, timelineAudioSourceStartForSegment,
    timelineDuration, updateMultiSelectButton, updatePlayPauseButton, usingSceneAudioMode,
    usingSceneAudioPlaybackMode, videoSettingsSegment,
  };
}
