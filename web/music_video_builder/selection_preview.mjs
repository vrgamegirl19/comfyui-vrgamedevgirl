import { overlayClipIsEnabled } from "../VRGDG_OverlayTrack.js";
import {
  audioUrl,
  makeEditorImageUrl,
  makeEditorThumbnailUrl,
  makeEditorVideoUrl,
  makeImageViewUrl,
} from "./comfy_api.mjs";
import { escapeHtml, makeButton, makeField, makeSelect, normalizeProjectVideoEngine, toast } from "./controls.mjs";
import { formatDurationSeconds, formatTime } from "./format.mjs";
import {
  audioTimelineEnd,
  audioTimelineStart,
  mediaPathKey,
  normalizeSegmentVideoHistory,
  timelineAudioDurationForSegment,
  timelineAudioEndForSegment,
  timelineAudioStartForSegment,
} from "./timeline_state.mjs";

export function localPlaybackTime(segment, globalTime) {
  if (!segment) return 0;
  const duration = Math.max(0.1, Number(segment.end || 0) - Number(segment.start || 0));
  return Math.max(0, Math.min(duration, Number(globalTime || 0) - Number(segment.start || 0)));
}

export function hasLockedVideo(segment) {
  return Boolean(segment?.video_path);
}

export function selectedSegmentVideoPath(segment) {
  if (!segment) return "";
  if (!Array.isArray(segment.video_history)) normalizeSegmentVideoHistory(segment);
  const history = Array.isArray(segment?.video_history) ? segment.video_history : [];
  const index = Math.max(0, Math.min(history.length - 1, Number(segment?.video_history_index || 0)));
  const path = history[index] || segment?.video_path || "";
  return isLikelyVideoPath(path) ? path : "";
}

function selectedSegmentVideoCacheKey(segment, videoPath = "") {
  const path = String(videoPath || selectedSegmentVideoPath(segment) || "").trim();
  if (!path) return "";
  return `${path}|${String(segment?.video_cache_bust || "")}`;
}

export function isLikelyVideoPath(path) {
  const text = String(path || "").trim();
  if (!text) return false;
  return /\.(?:mp4|mov|mkv|webm|avi|m4v)$/i.test(text);
}

export function selectedSegmentVideoThumbnailPath(segment) {
  if (!segment) return "";
  if (!Array.isArray(segment.video_history)) normalizeSegmentVideoHistory(segment);
  const thumbnails = Array.isArray(segment?.video_thumbnail_history) ? segment.video_thumbnail_history : [];
  const videos = Array.isArray(segment?.video_history) ? segment.video_history : [];
  const index = Math.max(0, Math.min(videos.length - 1, Number(segment?.video_history_index || 0)));
  const selectedVideo = String(videos[index] || "").trim();
  const currentThumbnail = mediaPathKey(selectedVideo) === mediaPathKey(segment?.video_path) ? segment?.video_thumbnail_path : "";
  return String(thumbnails[index] || currentThumbnail || "").trim();
}

function clearSegmentTextFields(segment, fields = []) {
  if (!segment) return;
  for (const field of fields) {
    if (field in segment || segment[field] != null) segment[field] = "";
  }
}

function timelineImageSourceUrl(image = {}) {
  const source = image && typeof image === "object" ? image : {};
  if (source.data) return String(source.data);
  if (source.path) return makeEditorThumbnailUrl(source.path);
  return "";
}

export function appendTimelineVideoThumbnail(block, segment) {
  const videoPath = selectedSegmentVideoPath(segment);
  if (!videoPath) return;
  const thumbnailPath = selectedSegmentVideoThumbnailPath(segment);
  const visual = document.createElement("span");
  visual.title = videoPath;
  if (thumbnailPath) {
    const thumbnailUrl = makeEditorThumbnailUrl(thumbnailPath).replace(/"/g, "%22");
    visual.style.cssText = `position:absolute;inset:0;background:linear-gradient(rgba(0,0,0,.18),rgba(0,0,0,.18)),url("${thumbnailUrl}") center / auto 100% repeat-x;pointer-events:none;z-index:0;`;
  } else {
    visual.textContent = "VIDEO";
    visual.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#020617;color:#67e8f9;font-size:10px;font-weight:900;letter-spacing:0;pointer-events:none;z-index:0;opacity:.78;";
  }
  const shade = document.createElement("span");
  shade.style.cssText = "position:absolute;inset:0;background:rgba(0,0,0,.18);pointer-events:none;z-index:1;";
  block.append(visual, shade);
}

export function createFirstLastFramePreviewSlot(label, image = {}, emptyText = "Missing") {
  const slot = document.createElement("div");
  slot.style.cssText = "min-width:0;display:flex;flex-direction:column;gap:5px;";
  const title = document.createElement("div");
  title.textContent = label;
  title.style.cssText = "font-size:10px;font-weight:900;color:#a5f3fc;text-transform:uppercase;letter-spacing:0;";
  const frame = document.createElement("div");
  frame.style.cssText = "position:relative;height:78px;border:1px solid #155e75;border-radius:6px;background:#020617;overflow:hidden;display:flex;align-items:center;justify-content:center;";
  const src = timelineImageSourceUrl(image);
  if (src) {
    const img = document.createElement("img");
    img.src = src;
    img.alt = label;
    img.draggable = false;
    img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
    frame.append(img);
  } else {
    const text = document.createElement("div");
    text.textContent = emptyText;
    text.style.cssText = "font-size:11px;font-weight:900;color:#67e8f9;text-align:center;padding:8px;";
    frame.append(text);
  }
  slot.append(title, frame);
  return slot;
}

export function openFirstLastFrameImagePreview(image = {}, label = "First Last Frame image") {
  const src = timelineImageSourceUrl(image);
  if (!src) return;
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.84);display:flex;align-items:center;justify-content:center;padding:24px;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(1180px,calc(100vw - 48px));max-height:calc(100vh - 48px);border:1px solid #155e75;border-radius:8px;background:#07111f;color:#f8fafc;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 20px 70px rgba(0,0,0,.65);";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 12px;border-bottom:1px solid #155e75;background:#0f172a;";
  const title = document.createElement("div");
  title.textContent = String(image.name || image.path || label);
  title.style.cssText = "font-size:13px;font-weight:900;color:#cffafe;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
  const close = makeButton("Close");
  const stage = document.createElement("div");
  stage.style.cssText = "min-height:180px;max-height:calc(100vh - 118px);background:#020617;display:flex;align-items:center;justify-content:center;padding:12px;overflow:auto;";
  const img = document.createElement("img");
  img.src = src; img.alt = label; img.draggable = false;
  img.style.cssText = "display:block;max-width:100%;max-height:calc(100vh - 150px);object-fit:contain;border-radius:4px;";
  stage.append(img); header.append(title, close); box.append(header, stage); backdrop.append(box); document.body.append(backdrop);
  const dismiss = () => { backdrop.remove(); document.removeEventListener("keydown", keydown, true); };
  const keydown = (event) => { if (event.key === "Escape") dismiss(); };
  close.onclick = dismiss;
  backdrop.addEventListener("pointerdown", (event) => { if (event.target === backdrop) dismiss(); });
  document.addEventListener("keydown", keydown, true);
}

export function createSelectionPreview({
  activeSegment, allEditableSegments, audioInput, audioSummary, cancelPreviewPlayStart,
  clearSceneEndFrameButton, clearSegmentAdjustPreview, clearSegmentFilmGrainPreview, clearSegmentLutPreview,
  createFluxPromptButton, createI2VButton, createNBPromptButton, createSceneEndFrameButton,
  createSceneVideoButton, createT2IButton, currentGlobalTime,
  deleteSegmentButton, editErnieT2IInstructionsButton,
  editFlowGptT2IInstructionsButton, editFluxKleinT2IInstructionsButton, editI2VPromptButton,
  editIdLoraInstructionsButton, editImagePromptButtons, editKrea2T2IInstructionsButton,
  editNanoBT2IInstructionsButton, editZImageT2IInstructionsButton, endInput, ensureGlobalTimelineAudioSource,
  ensureSegmentRuntimeFields, ernieCreateButton, ernieCreateT2IButton, ernieGemmaModelSelect,
  ernieMmprojSelect, ernieNotesInput, ernieRefImagePanel, ernieT2IPrompt, ernieTextGemmaModelSelect,
  ernieUseVisionReference, firstLastFramePromptReferences, firstLastFrameResolvedEndImageSource,
  firstLastFrameStartImageSource, flfTransitionTypeSelect, flowGptCreatePromptButton, fluxPrompt,
  fluxUseDirectorNotes, fluxUseTextOnlyGemmaPrompt, freezeTimingControl, gemmaModelSelect, globalAudioSummary,
  globalScrub, globalScrubTime, i2vGemmaModelSelect, i2vMmprojSelect, i2vMotionJsonInput, i2vNotesInput,
  i2vPrompt, i2vTextGemmaModelSelect, i2vUseGgufModel, isSegmentMultiSelected, isTimelinePlaying,
  krea2TwoPassCreateT2IButton, krea2TwoPassNotesInput, krea2TwoPassRefImagePanel, krea2TwoPassT2IPrompt,
  krea2TwoPassUseVisionReference, labelInput, loadCustomImageButton, lyricSingersInput, lyricTextInput,
  miniMaxGemmaModelSelect, miniMaxMmprojSelect, miniMaxSceneVideoButtons, miniMaxTextGemmaModelSelect,
  mmprojSelect, nbApiKey, nbGemmaModelSelect, nbMmprojSelect, nbModelSelect, nbNotes, nbPrompt,
  nbUseDirectorNotes, nbUseTextOnlyGemmaPrompt, notesInput, openSceneAudioOptionsButton,
  pauseTimelineForEditing, playStart, playbackDuration, playhead, postProcessComparePreview, preloadVideo,
  previewButton, previewDecodeHint, previewEmpty, previewImage, previewNBButton, previewVideo,
  previewVideoState, promptJsonInput, refImageInput, refImagePanel, render, renderFilmGrainPostProcessPanel,
  renderSceneAdjustPanel, renderSceneToolsPanel, rtvReferenceBehaviorGlobalValue, rtvReferenceBehaviorSelect,
  saveI2VPromptButton, saveI2VVideoSettingsFromPanel, saveMiniMaxH3SettingsFromPanel, saveMiniMaxPromptButton,
  savedI2VPrompts, sceneAudio, segmentImageSource, segmentIndexInfo, segmentLayer, segmentTrack, refreshDeleteActions,
  selectedSegmentsForBatch, silentTimeline, srtInput, startInput,
  startSilentTimelinePlayback, state, storyIdeaInput, subjectSceneInput, syncErnieImagePanel,
  syncFluxKleinPanel, syncInspectorPanels, syncKrea2TwoPassPanel, syncMiniMaxH3Panel,
  syncRTVSceneImageAnchorPanel, syncVideoModePanel, syncZEnhanceSettingsPanel, syncZImageSettingsPanel,
  t2iPrompt, t2iTextGemmaModelSelect, t2vRefImagePanel, themeStyleInput, timelineAudioPathForSegment,
  timelineAudioSegmentAtTime, timelineAudioSourceStartForSegment, timelineCanvas, timelineDuration,
  updateActiveFromInputs, updateI2VPromptSaveButtonState, useI2VVisionReference,
  useSceneErnieImageSettings, useSceneFluxKleinSettings, useSceneI2VVideoSettings,
  useSceneKrea2TwoPassSettings, useSceneMiniMaxH3Settings, useSceneNBImageSettings, useSceneZImageSettings,
  useT2VVisionReference, useVisionReference, useVrgdgTextContext, usingSceneAudioPlaybackMode,
  zEnhanceGemmaButton, zEnhanceGemmaModelSelect, zEnhanceGemmaNotes, zEnhanceMmprojSelect,
  zEnhancePromptPreview,
}) {
  function moveActiveSceneSelection(direction) {
    const track = state.activeTrack === "overlay" ? "overlay" : "base";
    const list = track === "overlay" ? state.overlaySegments : state.segments;
    if (!list.length) return false;
    const currentIndex = list.findIndex((segment) => segment.id === state.activeId);
    const fallbackIndex = direction > 0 ? -1 : list.length;
    const nextIndex = Math.max(0, Math.min(list.length - 1, (currentIndex >= 0 ? currentIndex : fallbackIndex) + direction));
    const next = list[nextIndex];
    if (!next || next.id === state.activeId) return false;
    setActiveSegment(next);
    return true;
  }

  function clearActiveSegment() {
    if (!state.activeId) return;
    state.activeId = "";
    syncInspector();
    render();
  }

  function setMultiSelectMode(enabled) {
    state.multiSelectMode = Boolean(enabled);
    state.modifierMultiSelectMode = false;
    if (!state.multiSelectMode) {
      state.selectedSegmentIds = [];
    } else if (state.activeId && !state.selectedSegmentIds.length) {
      state.selectedSegmentIds = [state.activeId];
    }
    render();
  }

  function parseMultiSelectSceneList(value) {
    const sceneCount = state.segments.length;
    let normalized = String(value || "").trim().toLowerCase();
    if (!normalized) throw new Error("Enter at least one scene number, range, or shortcut.");
    normalized = normalized
      .replace(/\b(?:scenes?|clips?)\b/gi, " ")
      .replace(/(^|\n)\s*[-*•]\s*/g, "$1")
      .replace(/(\d+)\s*(?:\.\.|-|–|—|\bto\b|\bthrough\b)\s*(\d+)/gi, "$1-$2")
      .replace(/\b(?:and)\b|&/gi, " ")
      .replace(/(\d+)\.(?=\s|,|;|$)/g, "$1")
      .replace(/[#\[\]{}()'\":|]/g, " ");
    const tokens = normalized.split(/[\s,;]+/).filter(Boolean);
    const sceneNumbers = new Set();
    const invalid = [];
    const addNumber = (number) => {
      if (!Number.isInteger(number) || number < 1 || number > sceneCount) {
        invalid.push(String(number));
        return;
      }
      sceneNumbers.add(number);
    };
    for (const token of tokens) {
      if (token === "all") {
        for (let number = 1; number <= sceneCount; number += 1) sceneNumbers.add(number);
        continue;
      }
      if (token === "odd" || token === "even") {
        const parity = token === "odd" ? 1 : 0;
        for (let number = 1; number <= sceneCount; number += 1) {
          if (number % 2 === parity) sceneNumbers.add(number);
        }
        continue;
      }
      if (token === "none" || token === "clear") continue;
      const range = token.match(/^(\d+)-(\d+)$/);
      if (range) {
        const first = Number(range[1]);
        const last = Number(range[2]);
        if (first < 1 || first > sceneCount || last < 1 || last > sceneCount) {
          invalid.push(token);
          continue;
        }
        const step = first <= last ? 1 : -1;
        for (let number = first; ; number += step) {
          sceneNumbers.add(number);
          if (number === last) break;
        }
        continue;
      }
      if (/^\d+$/.test(token)) {
        addNumber(Number(token));
        continue;
      }
      invalid.push(token);
    }
    if (invalid.length) {
      const rangeText = sceneCount ? `Scene numbers must be between 1 and ${sceneCount}.` : "There are no base scenes yet.";
      throw new Error(`Could not read: ${invalid.join(", ")}. ${rangeText}`);
    }
    return Array.from(sceneNumbers).sort((a, b) => a - b);
  }

  function openMultiSelectChooser() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:20px;box-sizing:border-box;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(680px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "How do you want to select clips?";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const currentCount = selectedSegmentsForBatch().length;
    const summary = document.createElement("div");
    summary.textContent = currentCount
      ? `${currentCount} clip${currentCount === 1 ? " is" : "s are"} currently selected.`
      : "No clips are currently selected.";
    summary.style.cssText = "font-size:12px;color:#cbd5e1;";

    const clickCard = document.createElement("div");
    clickCard.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:11px;display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const clickCopy = document.createElement("div");
    clickCopy.innerHTML = `<strong style="color:#e0f2fe;">Click timeline clips</strong><br><span style="font-size:12px;color:#94a3b8;">Turn multi-select on, or hold Ctrl/Cmd while clicking any base scene or insert to add or remove it.</span>`;
    const clickSelect = makeButton("Start Clicking", "primary");
    clickSelect.style.flex = "0 0 auto";
    clickCard.append(clickCopy, clickSelect);

    const listCard = document.createElement("div");
    listCard.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:11px;display:flex;flex-direction:column;gap:9px;";
    const listHeading = document.createElement("div");
    listHeading.innerHTML = `<strong style="color:#e0f2fe;">Enter base scene numbers</strong><br><span style="font-size:12px;color:#94a3b8;">Use commas, spaces, new lines, ranges, or a normal pasted list. Examples: <code>1, 3, 7</code>, <code>4-8</code>, <code>2 to 6</code>, <code>[1, 5, 9]</code>, <code>all</code>, <code>odd</code>, or <code>even</code>.</span>`;
    const input = document.createElement("textarea");
    const selectedBaseNumbers = state.segments
      .map((segment, index) => isSegmentMultiSelected(segment) ? index + 1 : 0)
      .filter(Boolean);
    input.value = selectedBaseNumbers.join(", ");
    input.placeholder = "1, 3, 5-8\n12\n15";
    input.style.cssText = "width:100%;box-sizing:border-box;min-height:112px;resize:vertical;border:1px solid #374151;border-radius:7px;background:#09090b;color:#f8fafc;padding:10px;font-size:13px;line-height:1.45;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;";
    const quickRow = document.createElement("div");
    quickRow.style.cssText = "display:flex;flex-wrap:wrap;gap:7px;";
    for (const [label, value] of [["All", "all"], ["Odd", "odd"], ["Even", "even"], ["Clear", "none"]]) {
      const button = makeButton(label);
      button.onclick = () => {
        input.value = value;
        input.dispatchEvent(new Event("input"));
        input.focus();
      };
      quickRow.append(button);
    }
    const actionSelect = makeSelect(["replace", "add", "remove"], "replace");
    actionSelect.options[0].textContent = "Replace current selection";
    actionSelect.options[1].textContent = "Add to current selection";
    actionSelect.options[2].textContent = "Remove from current selection";
    const status = document.createElement("div");
    status.style.cssText = "min-height:18px;font-size:12px;color:#a5f3fc;";
    const updateStatus = () => {
      try {
        const numbers = parseMultiSelectSceneList(input.value);
        status.textContent = `${numbers.length} base scene${numbers.length === 1 ? "" : "s"} found${numbers.length ? `: ${numbers.join(", ")}` : "."}`;
        status.style.color = "#a5f3fc";
      } catch (error) {
        status.textContent = String(error?.message || error);
        status.style.color = "#fca5a5";
      }
    };
    input.addEventListener("input", updateStatus);
    listCard.append(listHeading, input, quickRow, makeField("How to apply this list", actionSelect), status);

    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px;";
    const cancel = makeButton("Cancel");
    const turnOff = makeButton("Exit Multi-select");
    turnOff.style.display = state.multiSelectMode ? "" : "none";
    const apply = makeButton("Apply Scene List", "primary");
    actions.append(cancel, turnOff, apply);
    box.append(heading, summary, clickCard, listCard, actions);
    backdrop.append(box);
    document.body.append(backdrop);

    const close = () => backdrop.remove();
    cancel.onclick = close;
    clickSelect.onclick = () => {
      setMultiSelectMode(true);
      close();
      toast("Multi-select is on. Click timeline clips to add or remove them.");
    };
    turnOff.onclick = () => {
      setMultiSelectMode(false);
      close();
      toast("Multi-select is off.");
    };
    apply.onclick = () => {
      try {
        const numbers = parseMultiSelectSceneList(input.value);
        const listedIds = new Set(numbers.map((number) => state.segments[number - 1]?.id).filter(Boolean));
        const nextIds = new Set(Array.isArray(state.selectedSegmentIds) ? state.selectedSegmentIds : []);
        if (actionSelect.value === "replace") nextIds.clear();
        for (const id of listedIds) {
          if (actionSelect.value === "remove") nextIds.delete(id);
          else nextIds.add(id);
        }
        state.selectedSegmentIds = Array.from(nextIds);
        state.multiSelectMode = true;
        if (state.selectedSegmentIds.length && !nextIds.has(state.activeId)) {
          state.activeId = state.selectedSegmentIds[0];
          state.activeTrack = "base";
          syncInspector();
        }
        render();
        close();
        const count = selectedSegmentsForBatch().length;
        toast(`${count} clip${count === 1 ? "" : "s"} selected. Multi-select is on.`);
      } catch (error) {
        status.textContent = String(error?.message || error);
        status.style.color = "#fca5a5";
        input.focus();
      }
    };
    input.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") apply.click();
      if (event.key === "Escape") close();
    });
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) close();
    });
    updateStatus();
    input.focus();
    input.select();
  }

  function segmentAtTime(time) {
    const current = Number(time || 0);
    return state.segments.find((segment, index) => {
      const start = Number(segment.start || 0);
      const end = Number(segment.end || 0);
      const isLast = index === state.segments.length - 1;
      return current >= start && (current < end || (isLast && current <= end));
    }) || null;
  }

  function playbackSegmentAtTime(time) {
    const current = Number(time || 0);
    const overlay = (state.overlayTrack.enabled ? state.overlaySegments : [])
      .filter((segment) => overlayClipIsEnabled(segment, state.overlayTrack))
      .slice()
      .sort((a, b) => Number(b.start || 0) - Number(a.start || 0))
      .find((segment) => {
        const start = Number(segment.start || 0);
        const end = Number(segment.end || 0);
        return current >= start && current < end;
      });
    return overlay || segmentAtTime(current);
  }

  const PRELOAD_NEXT_CLIP_LEAD_SECONDS = 1.5;

  // Warms the browser's HTTP cache for the upcoming scene's clip a little
  // before the playhead reaches it. Combined with the stable video_cache_bust
  // query param above, the real swap in setPreviewVideoSource then hits a
  // cache instead of a cold fetch, removing most of the stall at the cut.
  function maybePreloadNextClip(segment, current) {
    if (!segment) return;
    const end = Number(segment.end || 0);
    const remaining = end - Number(current || 0);
    if (!(remaining > 0) || remaining > PRELOAD_NEXT_CLIP_LEAD_SECONDS) return;
    const nextSegment = playbackSegmentAtTime(end + 0.05);
    if (!nextSegment || nextSegment.id === segment.id) return;
    const nextPath = selectedSegmentVideoPath(nextSegment);
    if (!nextPath) return;
    const cacheKey = selectedSegmentVideoCacheKey(nextSegment, nextPath);
    if (!cacheKey || preloadVideo.dataset.cacheKey === cacheKey) return;
    preloadVideo.src = makeEditorVideoUrl(nextPath, nextSegment.video_cache_bust);
    preloadVideo.dataset.cacheKey = cacheKey;
    preloadVideo.load();
  }

  function previewVideoLoadIssueText(reason, segment, videoPath) {
    const info = segmentIndexInfo(segment);
    const label = info.track === "overlay" ? `Insert ${info.index + 1}` : `Scene ${info.index + 1}`;
    const reasonText = reason === "timeout"
      ? "The player found a video file, but it could not read the video metadata."
      : "The player found a video file, but the browser rejected it.";
    const mediaError = previewVideo.error;
    const codeText = mediaError?.code ? ` Browser media error code: ${mediaError.code}.` : "";
    return `${label} has a video file, but it cannot be previewed in the browser. ${reasonText}${codeText}\n\nMost likely: the MP4 is incomplete, still being written, missing duration metadata, or encoded with a codec/pixel format the browser cannot play. The file may still exist on disk. Try opening the scene video directly; if it plays outside the browser, re-encode/remux it to H.264 MP4 with yuv420p and faststart, then use Restore Video.`;
  }

  function selectedSegmentImageThumbnailPath(segment) {
    if (!segment) return "";
    ensureSegmentRuntimeFields(segment);
    return segment.image_history?.[segment.image_history_index]
      || segment.image_history?.[segment.image_history.length - 1]
      || segment.custom_image_path
      || segment.approved_image_path
      || "";
  }

  function clearConceptPromptNotesFromSegments() {
    const conceptFields = [
      "notes",
      "flux_notes",
      "nb_notes",
      "story_beat",
      "prompt_summary",
      "scene_summary",
      "image_prompt",
      "text_to_image_prompt",
    ];
    for (const segment of allEditableSegments()) {
      clearSegmentTextFields(segment, conceptFields);
    }
  }

  function clearI2VMotionNotesFromSegments() {
    const motionFields = [
      "i2v_notes",
      "video_notes",
      "motion_summary",
      "motion_video_summary",
    ];
    for (const segment of allEditableSegments()) {
      clearSegmentTextFields(segment, motionFields);
    }
  }

  function mediaThumbnailHtml(segment, height = 56) {
    const imagePath = selectedSegmentImageThumbnailPath(segment);
    if (imagePath) {
      return `<img src="${escapeHtml(makeEditorThumbnailUrl(imagePath))}" style="width:100%;height:${height}px;object-fit:cover;border-radius:4px;margin-top:6px;background:#050505;">`;
    }
    const videoPath = selectedSegmentVideoPath(segment);
    if (!videoPath) return "";
    const thumbnailPath = selectedSegmentVideoThumbnailPath(segment);
    if (thumbnailPath) {
      return `<img src="${escapeHtml(makeEditorThumbnailUrl(thumbnailPath))}" title="${escapeHtml(videoPath)}" style="width:100%;height:${height}px;object-fit:cover;border-radius:4px;margin-top:6px;background:#050505;">`;
    }
    return `<div title="${escapeHtml(videoPath)}" style="width:100%;height:${height}px;box-sizing:border-box;border:1px solid #155e75;border-radius:4px;margin-top:6px;background:#020617;display:flex;align-items:center;justify-content:center;color:#67e8f9;font-size:11px;font-weight:900;letter-spacing:0;">VIDEO</div>`;
  }

  function appendTimelineFirstLastFrameThumbnail(block, segment) {
    const promptStart = firstLastFramePromptReferences(segment)[0] || {};
    const resolvedStart = firstLastFrameStartImageSource(segment) || {};
    const firstSource = (resolvedStart.path || resolvedStart.data)
      ? resolvedStart
      : (promptStart.path || promptStart.data)
        ? promptStart
        : segmentImageSource(segment) || {};
    const lastSource = firstLastFrameResolvedEndImageSource(segment) || {};
    const firstUrl = timelineImageSourceUrl(firstSource);
    const lastUrl = timelineImageSourceUrl(lastSource);
    const wrap = document.createElement("span");
    wrap.title = lastUrl
      ? "First Last Frame: first frame on the left, generated end frame on the right."
      : "First Last Frame: end frame is missing.";
    wrap.style.cssText = "position:absolute;inset:0;display:grid;grid-template-columns:minmax(0,1fr) 13px minmax(0,1fr);background:#020617;pointer-events:none;z-index:0;";
    const makeSlot = (url, label, missing = false) => {
      const slot = document.createElement("span");
      slot.style.cssText = `position:relative;min-width:0;overflow:hidden;background:${missing ? "rgba(15,23,42,.88)" : "#050505"};display:flex;align-items:center;justify-content:center;`;
      if (url) {
        const img = document.createElement("img");
        img.src = url;
        img.alt = label;
        img.draggable = false;
        img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;opacity:.9;";
        slot.append(img);
      } else {
        const text = document.createElement("span");
        text.textContent = missing ? "END?" : "START?";
        text.style.cssText = "font-size:9px;font-weight:900;color:#bae6fd;text-shadow:0 1px 2px #020617;";
        slot.append(text);
      }
      const badge = document.createElement("span");
      badge.textContent = label;
      badge.style.cssText = "position:absolute;left:4px;top:4px;min-width:13px;height:13px;border:1px solid rgba(255,255,255,.45);border-radius:3px;background:rgba(2,6,23,.72);color:#f8fafc;font-size:8px;font-weight:900;line-height:12px;text-align:center;";
      slot.append(badge);
      return slot;
    };
    const arrow = document.createElement("span");
    arrow.textContent = ">";
    arrow.style.cssText = "display:flex;align-items:center;justify-content:center;background:rgba(8,47,73,.78);color:#a5f3fc;font-size:10px;font-weight:900;";
    wrap.append(makeSlot(firstUrl, "A"), arrow, makeSlot(lastUrl, "B", !lastUrl));
    const shade = document.createElement("span");
    shade.style.cssText = "position:absolute;inset:0;background:rgba(0,0,0,.16);pointer-events:none;z-index:1;";
    block.append(wrap, shade);
  }

  function updateSelectedMediaTools() {
    refreshDeleteActions();
  }

  function previewVideoIsReadyAt(local) {
    return !previewVideo.seeking
      && previewVideo.readyState >= 2
      && Number.isFinite(local)
      && Math.abs(Number(previewVideo.currentTime || 0) - local) <= 0.2;
  }

  // Waits (briefly) for the preview video to actually be showing the given
  // local time before resolving. Used right before Play starts audio, so a
  // scene that's still loading/seeking from a recent scrub doesn't let the
  // audio clock run ahead of it - that's what causes the video to visibly
  // stutter and jump forward to catch up a moment after playback starts.
  function waitForPreviewVideoReady(local, timeoutMs = 700) {
    return new Promise((resolve) => {
      if (previewVideoIsReadyAt(local)) {
        resolve();
        return;
      }
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        previewVideo.removeEventListener("seeked", check);
        previewVideo.removeEventListener("canplay", check);
        previewVideo.removeEventListener("loadeddata", check);
        clearTimeout(timer);
        resolve();
      };
      const check = () => {
        if (previewVideoIsReadyAt(local)) finish();
      };
      previewVideo.addEventListener("seeked", check);
      previewVideo.addEventListener("canplay", check);
      previewVideo.addEventListener("loadeddata", check);
      const timer = setTimeout(finish, timeoutMs);
    });
  }

  function seekGlobalTimelineFromEvent(event) {
    const maxTime = playbackDuration();
    if (maxTime <= 0) return;
    const rect = timelineCanvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
    setGlobalPlaybackTime(Math.max(0, Math.min(maxTime, x / state.pxPerSecond)));
    updateAudioScrubbers();
  }

  function beginGlobalTimelineScrub(event) {
    if (event.button !== 0) return;
    if (event.target !== timelineCanvas && event.target !== playhead) return;
    event.preventDefault();
    event.stopPropagation();
    if (state.timelineTrimEditMode) pauseTimelineForEditing();
    state.isScrubbing = true;
    seekGlobalTimelineFromEvent(event);
    // Pointer events can fire far faster than the video can seek. Collapse
    // them to at most one processed position per animation frame, always the
    // latest one, instead of driving a seek off every raw mousemove - that
    // backlog of superseded seeks is why the preview used to lag behind
    // wherever the slider actually was.
    let pendingMoveEvent = null;
    let scrubRaf = 0;
    const flushPendingMove = () => {
      scrubRaf = 0;
      if (pendingMoveEvent) {
        seekGlobalTimelineFromEvent(pendingMoveEvent);
        pendingMoveEvent = null;
      }
    };
    const move = (moveEvent) => {
      pendingMoveEvent = moveEvent;
      if (!scrubRaf) scrubRaf = requestAnimationFrame(flushPendingMove);
    };
    const up = () => {
      if (scrubRaf) cancelAnimationFrame(scrubRaf);
      flushPendingMove();
      state.isScrubbing = false;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      updateAudioScrubbers();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  function setPreviewVideoSource(segment, videoPath) {
    const cacheKey = selectedSegmentVideoCacheKey(segment, videoPath);
    if (!videoPath || !cacheKey) return;
    if (previewVideo.dataset.cacheKey !== cacheKey) {
      if (previewVideoState.loadTimer) clearTimeout(previewVideoState.loadTimer);
      previewVideoState.loadTimer = null;
      previewDecodeHint.style.display = "none";
      previewVideo.pause();
      previewVideoState.pendingSeekTarget = null;
      previewVideo.src = makeEditorVideoUrl(videoPath, segment?.video_cache_bust);
      previewVideo.dataset.path = videoPath;
      previewVideo.dataset.cacheKey = cacheKey;
      previewVideo.dataset.segmentId = String(segment?.id || "");
      previewVideo.dataset.failureKey = "";
      previewVideo.load();
      previewVideoState.loadTimer = setTimeout(() => {
        if (previewVideo.dataset.cacheKey === cacheKey && previewVideo.readyState < 1) {
          handlePreviewVideoLoadIssue("timeout");
        }
      }, 8000);
    }
  }

  function clearPreviewVideoLoadState() {
    if (previewVideoState.loadTimer) clearTimeout(previewVideoState.loadTimer);
    previewVideoState.loadTimer = null;
    previewDecodeHint.style.display = "none";
    previewVideo.dataset.failureKey = "";
    previewVideo.dataset.segmentId = "";
  }

  function handlePreviewVideoLoadIssue(reason = "error") {
    if (previewVideoState.loadTimer) clearTimeout(previewVideoState.loadTimer);
    previewVideoState.loadTimer = null;
    const segmentId = String(previewVideo.dataset.segmentId || "");
    const segment = allEditableSegments().find((item) => String(item?.id || "") === segmentId) || activeSegment();
    const videoPath = String(previewVideo.dataset.path || selectedSegmentVideoPath(segment) || "");
    const failureKey = `${reason}|${selectedSegmentVideoCacheKey(segment, videoPath)}`;
    if (previewVideo.dataset.failureKey === failureKey) return;
    previewVideo.dataset.failureKey = failureKey;
    const message = previewVideoLoadIssueText(reason, segment, videoPath);
    const thumbnailPath = selectedSegmentVideoThumbnailPath(segment);
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.dataset.path = "";
    previewVideo.dataset.cacheKey = "";
    previewVideo.style.display = "none";
    if (thumbnailPath) {
      previewImage.src = makeEditorImageUrl(thumbnailPath);
      previewImage.title = "Video preview could not decode. Showing thumbnail.";
      previewImage.style.display = "block";
    }
    previewEmpty.style.display = "none";
    previewDecodeHint.textContent = message;
    previewDecodeHint.style.display = "block";
    toast("Video exists, but the browser preview cannot play it. Check the preview message for what to do.", true);
  }

  function syncPreviewPlayback(current) {
    const playing = isTimelinePlaying();
    const segment = playing ? playbackSegmentAtTime(current) : activeSegment();
    if (segment?.adjust_preview_image_path) {
      showAdjustPreviewImage(segment);
      return;
    }
    if (segment?.film_grain_preview_image_path) {
      showFilmGrainPreviewImage(segment);
      return;
    }
    if (segment?.lut_preview_image_path) {
      showLutPreviewImage(segment);
      return;
    }
    postProcessComparePreview.hide();
    if (playing) maybePreloadNextClip(segment, current);
    const videoPath = selectedSegmentVideoPath(segment);
    if (!segment || !videoPath) {
      if (!previewVideo.paused) previewVideo.pause();
      previewVideo.muted = false;
      return;
    }
    if (previewVideo.dataset.cacheKey !== selectedSegmentVideoCacheKey(segment, videoPath)) {
      setPreviewVideoSource(segment, videoPath);
      previewVideo.muted = false;
      previewVideo.style.display = "block";
      previewImage.style.display = "none";
      previewEmpty.style.display = "none";
    }
    // A cut is detected up to one tick late, so the correct in-scene offset
    // right after a swap isn't always ~0 - it can be off by up to a tick's
    // worth of time, which used to land just under the resync threshold and
    // never get corrected for the rest of that scene. Always request the
    // right position; a seek issued before metadata loads (readyState 0) is
    // simply deferred by the browser and applied once it's ready.
    const local = localPlaybackTime(segment, current);
    const seekThreshold = state.isScrubbing ? 0.05 : 0.2;
    previewVideoState.pendingSeekTarget = null;
    if (Number.isFinite(local) && Math.abs(Number(previewVideo.currentTime || 0) - local) > seekThreshold) {
      // While a seek is already in flight, issuing another currentTime write on
      // top of it doesn't jump ahead - it queues behind the one already
      // decoding. During a fast scrub drag that means the visible frame keeps
      // chasing a backlog of stale positions instead of the one the user is
      // actually pointing at. Remember only the latest desired target and let
      // the `seeked` handler below jump straight to it once the decoder is free.
      if (previewVideo.seeking) {
        previewVideoState.pendingSeekTarget = local;
      } else {
        try {
          previewVideo.currentTime = local;
        } catch {
          // Some browsers reject seeking until metadata is ready. The next timeupdate will retry.
        }
      }
    }
    if (isTimelinePlaying() && !state.timelineEditMenuOpen && !state.timelineTrimEditMode) {
      // The timeline uses the project/global audio as the audible source.
      // Keep the synced preview video silent so rendered MP4 audio cannot double or drift against it.
      previewVideo.muted = true;
      previewVideo.play().catch(() => {});
    } else if (!previewVideo.paused) {
      previewVideoState.syncPause = true;
      try {
        previewVideo.pause();
      } finally {
        previewVideoState.syncPause = false;
      }
      previewVideo.muted = false;
    }
  }

  function playSceneAudioFrom(time = currentGlobalTime()) {
    state.sceneSelectionUsesGlobalAudio = false;
    const maxTime = timelineDuration();
    const segment = timelineAudioSegmentAtTime(time) || playbackSegmentAtTime(time) || state.segments.find((item) => timelineAudioEndForSegment(item) > time && timelineAudioPathForSegment(item)) || state.segments[0] || null;
    if (!segment) return false;
    state.activeId = segment.id;
    state.sceneAudioGlobalTime = Math.max(timelineAudioStartForSegment(segment), Math.min(maxTime, Number(time || 0)));
    syncInspector();
    render();
    const sourcePath = timelineAudioPathForSegment(segment);
    if (!sourcePath) {
      sceneAudio.pause();
      sceneAudio.removeAttribute("src");
      state.sceneAudioSegmentId = "";
      const next = state.segments.find((item) => timelineAudioStartForSegment(item) > timelineAudioStartForSegment(segment) && timelineAudioPathForSegment(item));
      if (next) {
        return playSceneAudioFrom(timelineAudioStartForSegment(next));
      }
      updateAudioScrubbers();
      return false;
    }
    const local = Math.max(0, state.sceneAudioGlobalTime - timelineAudioStartForSegment(segment)) + timelineAudioSourceStartForSegment(segment);
    sceneAudio.src = audioUrl(sourcePath);
    sceneAudio.load();
    state.sceneAudioSegmentId = segment.id;
    const start = () => {
      try {
        sceneAudio.currentTime = local;
      } catch {
        // Metadata not ready yet.
      }
      sceneAudio.play().catch(() => {
        sceneAudio.pause();
        startSilentTimelinePlayback(state.sceneAudioGlobalTime);
      });
    };
    if (Number.isFinite(sceneAudio.duration)) start();
    else sceneAudio.onloadedmetadata = start;
    return true;
  }

  function setActiveSegment(segment) {
    const displayedSegment = activeSegment();
    if (displayedSegment && displayedSegment.id !== segment?.id) {
      clearSegmentLutPreview(displayedSegment);
      clearSegmentAdjustPreview(displayedSegment);
      clearSegmentFilmGrainPreview(displayedSegment);
      updateActiveFromInputs({ skipHistory: true });
      saveI2VVideoSettingsFromPanel();
      // Snapshot the visible MiniMax panel before changing scenes. Do not rely
      // on state.activeId here; some selection paths update it first.
      if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
        saveMiniMaxH3SettingsFromPanel();
      }
    }
    state.activeId = segment?.id || "";
    state.activeTrack = segment ? segmentTrack(segment) : state.activeTrack || "base";
    syncInspector();
    render();
  }

  function selectedSegmentImagePath(segment) {
    ensureSegmentRuntimeFields(segment);
    const history = Array.isArray(segment?.image_history) ? segment.image_history : [];
    const index = Math.max(0, Math.min(history.length - 1, Number(segment?.image_history_index || 0)));
    return history[index] || segment?.approved_image_path || segment?.custom_image_path || "";
  }

  function syncPreview(segment) {
    ensureSegmentRuntimeFields(segment);
    if (segment?.adjust_preview_image_path) {
      showAdjustPreviewImage(segment);
      return;
    }
    if (segment?.film_grain_preview_image_path) {
      showFilmGrainPreviewImage(segment);
      return;
    }
    if (segment?.lut_preview_image_path) {
      showLutPreviewImage(segment);
      return;
    }
    postProcessComparePreview.hide();
    const videoPath = selectedSegmentVideoPath(segment);
    if (segment?.preview_mode !== "image" && videoPath) {
      setPreviewVideoSource(segment, videoPath);
      previewVideo.muted = false;
      previewVideo.style.display = "block";
      previewImage.style.display = "none";
      previewEmpty.style.display = "none";
      return;
    }
    clearPreviewVideoLoadState();
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.dataset.path = "";
    previewVideo.dataset.cacheKey = "";
    previewVideo.style.display = "none";
    if (segment?.custom_image_data) {
      previewImage.src = segment.custom_image_data;
      previewImage.style.display = "block";
      previewEmpty.style.display = "none";
      return;
    }
    if (segment?.custom_image_path && !segment?.image?.filename) {
      previewImage.src = makeEditorImageUrl(segment.custom_image_path);
      previewImage.style.display = "block";
      previewEmpty.style.display = "none";
      return;
    }
    const selectedSource = segmentImageSource(segment);
    if (selectedSource?.path) {
      previewImage.src = makeEditorImageUrl(selectedSource.path);
      previewImage.style.display = "block";
      previewEmpty.style.display = "none";
      return;
    }
    if (selectedSource?.data) {
      previewImage.src = selectedSource.data;
      previewImage.style.display = "block";
      previewEmpty.style.display = "none";
      return;
    }
    const image = segment?.image || null;
    if (image?.filename) {
      previewImage.src = makeImageViewUrl(image);
      previewImage.style.display = "block";
      previewEmpty.style.display = "none";
      return;
    }
    previewImage.removeAttribute("src");
    previewImage.style.display = "none";
    previewEmpty.style.display = "block";
  }

  function showLutPreviewImage(segment) {
    const path = String(segment?.lut_preview_image_path || "").trim();
    if (!path) return false;
    const sourcePath = String(segment?.lut_preview_source_preview_path || "").trim();
    clearPreviewVideoLoadState();
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.load();
    previewVideo.dataset.path = "";
    previewVideo.dataset.cacheKey = "";
    previewVideo.style.display = "none";
    if (postProcessComparePreview.show({ beforePath: sourcePath, afterPath: path, title: "LUT preview" })) {
      previewImage.removeAttribute("src");
      previewImage.style.display = "none";
      previewEmpty.style.display = "none";
      return true;
    }
    postProcessComparePreview.hide();
    previewImage.src = makeEditorImageUrl(path);
    previewImage.title = "Temporary LUT preview";
    previewImage.style.display = "block";
    previewEmpty.style.display = "none";
    return true;
  }

  function showFilmGrainPreviewImage(segment) {
    const path = String(segment?.film_grain_preview_image_path || "").trim();
    if (!path) return false;
    const sourcePath = String(segment?.film_grain_preview_source_preview_path || "").trim();
    clearPreviewVideoLoadState();
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.load();
    previewVideo.dataset.path = "";
    previewVideo.dataset.cacheKey = "";
    previewVideo.style.display = "none";
    if (postProcessComparePreview.show({ beforePath: sourcePath, afterPath: path, title: "Film grain preview" })) {
      previewImage.removeAttribute("src");
      previewImage.style.display = "none";
      previewEmpty.style.display = "none";
      return true;
    }
    postProcessComparePreview.hide();
    previewImage.src = makeEditorImageUrl(path);
    previewImage.title = "Temporary film grain preview";
    previewImage.style.display = "block";
    previewEmpty.style.display = "none";
    return true;
  }

  function showAdjustPreviewImage(segment) {
    const path = String(segment?.adjust_preview_image_path || "").trim();
    if (!path) return false;
    const sourcePath = String(segment?.adjust_preview_source_preview_path || "").trim();
    clearPreviewVideoLoadState();
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.load();
    previewVideo.dataset.path = "";
    previewVideo.dataset.cacheKey = "";
    previewVideo.style.display = "none";
    if (postProcessComparePreview.show({ beforePath: sourcePath, afterPath: path, title: "Adjust preview" })) {
      previewImage.removeAttribute("src");
      previewImage.style.display = "none";
      previewEmpty.style.display = "none";
      return true;
    }
    postProcessComparePreview.hide();
    previewImage.src = makeEditorImageUrl(path);
    previewImage.title = "Temporary Adjust preview";
    previewImage.style.display = "block";
    previewEmpty.style.display = "none";
    return true;
  }

  function syncInspector() {
    const previousPanelSegmentId = String(state.miniMaxH3PanelSegmentId || "");
    const segment = activeSegment();
    if (previousPanelSegmentId && previousPanelSegmentId !== String(segment?.id || "")
      && normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
      const previousPanelSegment = allEditableSegments().find((item) => String(item?.id || "") === previousPanelSegmentId);
      if (previousPanelSegment) saveMiniMaxH3SettingsFromPanel(previousPanelSegment);
    }
    startInput.dataset.vrgdgInspectorSegmentId = String(segment?.id || "");
    lyricTextInput.dataset.vrgdgInspectorSegmentId = String(segment?.id || "");
    lyricTextInput.dataset.vrgdgUserEdited = "0";
    lyricSingersInput.dataset.vrgdgInspectorSegmentId = String(segment?.id || "");
    lyricSingersInput.dataset.vrgdgUserEdited = "0";
    const disabled = !segment;
    for (const control of [labelInput, startInput, endInput, notesInput, ernieNotesInput, krea2TwoPassNotesInput, nbNotes, lyricTextInput, lyricSingersInput, i2vNotesInput, t2iPrompt, ernieT2IPrompt, krea2TwoPassT2IPrompt, nbPrompt, i2vPrompt, zEnhanceGemmaNotes, zEnhancePromptPreview, previewButton, ernieCreateButton, previewNBButton, deleteSegmentButton, createSceneVideoButton, miniMaxSceneVideoButtons[0], saveI2VPromptButton, saveMiniMaxPromptButton]) {
      control.disabled = disabled;
    }
    loadCustomImageButton.disabled = disabled;
    openSceneAudioOptionsButton.disabled = disabled;
    for (const control of [t2iTextGemmaModelSelect, gemmaModelSelect, mmprojSelect, ernieTextGemmaModelSelect, ernieGemmaModelSelect, ernieMmprojSelect, zEnhanceGemmaModelSelect, zEnhanceMmprojSelect, i2vTextGemmaModelSelect, i2vGemmaModelSelect, i2vMmprojSelect, miniMaxTextGemmaModelSelect, miniMaxGemmaModelSelect, miniMaxMmprojSelect, nbApiKey, nbModelSelect, nbGemmaModelSelect, nbMmprojSelect, fluxUseTextOnlyGemmaPrompt.input, fluxUseDirectorNotes.input, nbUseTextOnlyGemmaPrompt.input, nbUseDirectorNotes.input, useVisionReference.input, ernieUseVisionReference.input, krea2TwoPassUseVisionReference.input, useI2VVisionReference.input, useT2VVisionReference.input, useSceneZImageSettings.input, useSceneErnieImageSettings.input, useSceneFluxKleinSettings.input, useSceneNBImageSettings.input, useSceneI2VVideoSettings.input, useSceneMiniMaxH3Settings.input, rtvReferenceBehaviorSelect, createSceneEndFrameButton, clearSceneEndFrameButton, i2vUseGgufModel.input, refImageInput, createT2IButton, editZImageT2IInstructionsButton, ernieCreateT2IButton, editErnieT2IInstructionsButton, krea2TwoPassCreateT2IButton, editKrea2T2IInstructionsButton, createNBPromptButton, editNanoBT2IInstructionsButton, flowGptCreatePromptButton, editFlowGptT2IInstructionsButton, createFluxPromptButton, editFluxKleinT2IInstructionsButton, createI2VButton, editI2VPromptButton, editIdLoraInstructionsButton, zEnhanceGemmaButton, ...editImagePromptButtons]) {
      control.disabled = disabled;
    }
    const lockedByVideo = hasLockedVideo(segment);
    const isOverlay = segmentTrack(segment) === "overlay";
    startInput.disabled = disabled || (!isOverlay && state.timingFrozen) || lockedByVideo;
    endInput.disabled = disabled || (!isOverlay && state.timingFrozen) || lockedByVideo;
    freezeTimingControl.input.checked = Boolean(state.timingFrozen);
    promptJsonInput.value = state.promptJsonPath || "";
    i2vMotionJsonInput.value = state.i2vMotionJsonPath || "";
    useSceneErnieImageSettings.input.checked = Boolean(segment?.use_scene_ernie_image_settings);
    useSceneKrea2TwoPassSettings.input.checked = Boolean(segment?.use_scene_krea2_2pass_settings);
    useSceneFluxKleinSettings.input.checked = Boolean(segment?.use_scene_flux_klein_settings);
    useSceneNBImageSettings.input.checked = Boolean(segment?.use_scene_nb_image_settings);
    useSceneI2VVideoSettings.input.checked = Boolean(segment?.use_scene_i2v_video_settings);
    rtvReferenceBehaviorSelect.value = rtvReferenceBehaviorGlobalValue();
    useVrgdgTextContext.input.checked = Boolean(state.useVrgdgTextContext);
    themeStyleInput.value = state.themeStylePath || "";
    storyIdeaInput.value = state.storyIdeaPath || "";
    subjectSceneInput.value = state.subjectScenePath || "";
    globalAudioSummary.innerHTML = `
      <div><strong>Global audio:</strong> ${escapeHtml(audioInput.value || "Not loaded")}</div>
      <div style="margin-top:6px;"><strong>SRT:</strong> ${escapeHtml(srtInput.value || "Not loaded")}</div>
      <div style="margin-top:6px;color:#a1a1aa;">Global audio drives the whole timeline. Use this for music videos, songs, visualizers, and beat/lyric timing.</div>
    `;
    syncInspectorPanels();
    syncRTVSceneImageAnchorPanel();
    renderSceneToolsPanel();
    renderSceneAdjustPanel();
    if (state.leftPanelTab === "luts" && state.postProcessTab === "film_grain") {
      renderFilmGrainPostProcessPanel();
    }
    if (!segment) {
      labelInput.value = "";
      startInput.value = "0";
      endInput.value = "4";
      notesInput.value = "";
      ernieNotesInput.value = "";
      krea2TwoPassNotesInput.value = "";
      nbNotes.value = "";
      lyricTextInput.value = "";
      lyricSingersInput.value = "";
      i2vNotesInput.value = "";
      t2iPrompt.value = "";
      ernieT2IPrompt.value = "";
      krea2TwoPassT2IPrompt.value = "";
      nbPrompt.value = "";
      i2vPrompt.value = "";
      editI2VPromptButton.style.display = "none";
      saveI2VPromptButton.disabled = true;
      saveI2VPromptButton.style.opacity = "0.5";
      saveI2VPromptButton.style.cursor = "not-allowed";
      saveMiniMaxPromptButton.disabled = true;
      saveMiniMaxPromptButton.style.opacity = "0.5";
      saveMiniMaxPromptButton.style.cursor = "not-allowed";
      editImagePromptButtons.forEach((button) => {
        button.style.display = "none";
      });
      useVisionReference.input.checked = false;
      ernieUseVisionReference.input.checked = false;
      krea2TwoPassUseVisionReference.input.checked = false;
      useI2VVisionReference.input.checked = true;
      useT2VVisionReference.input.checked = false;
      useSceneZImageSettings.input.checked = false;
      useSceneNBImageSettings.input.checked = false;
      refImageInput.value = "";
      refImagePanel.style.display = "none";
      ernieRefImagePanel.style.display = "none";
      krea2TwoPassRefImagePanel.style.display = "none";
      t2vRefImagePanel.style.display = "none";
      audioSummary.textContent = "Select a scene to view or edit scene audio.";
      syncZImageSettingsPanel();
      syncFluxKleinPanel();
      syncErnieImagePanel();
      syncKrea2TwoPassPanel();
      syncZEnhanceSettingsPanel();
      syncVideoModePanel();
      syncPreview(null);
      return;
    }
    labelInput.value = segment.label || "";
    startInput.value = segment.start;
    endInput.value = segment.end;
    notesInput.value = segment.notes || "";
    ernieNotesInput.value = segment.notes || "";
    krea2TwoPassNotesInput.value = segment.notes || "";
    nbNotes.value = segment.nb_notes || segment.flux_notes || segment.notes || "";
    lyricTextInput.value = segment.lyric_text || "";
    lyricSingersInput.value = Array.isArray(segment.lyric_singers) ? segment.lyric_singers.join(", ") : "";
    i2vNotesInput.value = segment.i2v_notes || "";
    flfTransitionTypeSelect.value = segment.flf_transition_type || "global";
    t2iPrompt.value = segment.t2i_prompt || "";
    ernieT2IPrompt.value = segment.t2i_prompt || "";
    krea2TwoPassT2IPrompt.value = segment.t2i_prompt || "";
    fluxPrompt.value = segment.t2i_prompt || segment.flux_prompt || "";
    nbPrompt.value = segment.t2i_prompt || segment.nb_prompt || "";
    if (!savedI2VPrompts.has(segment)) savedI2VPrompts.set(segment, String(segment.i2v_prompt || ""));
    i2vPrompt.value = segment.i2v_prompt || "";
    editI2VPromptButton.style.display = String(segment.i2v_prompt || "").trim() ? "" : "none";
    updateI2VPromptSaveButtonState();
    editImagePromptButtons.forEach((button) => {
      button.style.display = String(segment.t2i_prompt || segment.flux_prompt || segment.nb_prompt || "").trim() ? "" : "none";
    });
    useVisionReference.input.checked = Boolean(segment.use_vision_reference);
    ernieUseVisionReference.input.checked = Boolean(segment.use_vision_reference);
    krea2TwoPassUseVisionReference.input.checked = Boolean(segment.use_vision_reference);
    useI2VVisionReference.input.checked = segment.use_i2v_vision_reference !== false;
    useT2VVisionReference.input.checked = Boolean(segment.use_t2v_vision_reference);
    useSceneZImageSettings.input.checked = Boolean(segment.use_scene_zimage_settings);
    refImageInput.value = segment.ref_image_path || "";
    refImagePanel.style.display = useVisionReference.input.checked ? "flex" : "none";
    ernieRefImagePanel.style.display = ernieUseVisionReference.input.checked ? "flex" : "none";
    krea2TwoPassRefImagePanel.style.display = krea2TwoPassUseVisionReference.input.checked ? "flex" : "none";
    audioSummary.innerHTML = segment.custom_audio_path
      ? `
        <div><strong>Scene audio:</strong> ${escapeHtml(segment.custom_audio_name || segment.custom_audio_path)}</div>
        <div style="margin-top:6px;"><strong>Timeline:</strong> ${formatTime(audioTimelineStart(segment))} - ${formatTime(audioTimelineEnd(segment))}</div>
        <div style="margin-top:6px;"><strong>Duration:</strong> ${formatDurationSeconds(audioTimelineStart(segment), audioTimelineEnd(segment))} seconds</div>
        <div style="margin-top:6px;color:#a1a1aa;">Click the purple waveform on the timeline for cut/delete audio options.</div>
      `
      : "No custom scene audio selected. Use Scene Audio Options to drop or load audio for this scene.";
    syncZImageSettingsPanel();
    syncFluxKleinPanel();
    syncErnieImagePanel();
    syncKrea2TwoPassPanel();
    syncZEnhanceSettingsPanel();
    syncVideoModePanel();
    syncMiniMaxH3Panel();
    syncPreview(segment);
    updateAudioScrubbers();
  }

  function updateAudioScrubbers() {
    const current = currentGlobalTime();
    const maxTime = playbackDuration();
    const followPlayback = Boolean(isTimelinePlaying() || state.isScrubbing);
    if (followPlayback) {
      const playbackSegment = playbackSegmentAtTime(current);
      if (playbackSegment && playbackSegment.id !== state.activeId) {
        state.activeId = playbackSegment.id;
        // Sync the video first so the new clip starts loading immediately.
        // syncInspector() (dozens of form fields) and render() (rebuilds every
        // timeline block from scratch) are comparatively heavy, and running
        // them synchronously here was blocking that load on the same frame as
        // the cut, which is what produced the visible stutter at scene
        // boundaries (and, during a scrub drag, the preview lagging behind
        // wherever the slider actually was). Push them to the next frame
        // instead so they stay off the video swap.
        syncPreviewPlayback(current);
        if (isTimelinePlaying() || state.isScrubbing) {
          requestAnimationFrame(() => {
            syncInspector();
            render();
          });
        } else {
          syncInspector();
          render();
        }
        return;
      }
    }
    globalScrub.max = String(Math.max(0, maxTime));
    if (!state.isScrubbing) globalScrub.value = String(current);
    globalScrubTime.textContent = `${formatTime(current)} / ${formatTime(maxTime)}`;
    // Scene blocks are positioned inside segmentLayer, which starts after the
    // timeline viewport padding. Keep the playhead on that same visual origin.
    playhead.style.left = `${segmentLayer.offsetLeft + current * state.pxPerSecond}px`;
    syncPreviewPlayback(current);
  }

  function setGlobalPlaybackTime(value) {
    if (playStart.inFlight) cancelPreviewPlayStart();
    const maxTime = playbackDuration();
    const time = Math.max(0, Math.min(maxTime, Number(value || 0)));
    state.sceneAudioGlobalTime = time;
    if (silentTimeline.playing) {
      silentTimeline.startTime = time;
      silentTimeline.startedAt = performance.now();
    }
    if (usingSceneAudioPlaybackMode()) {
      state.sceneSelectionUsesGlobalAudio = false;
      const segment = timelineAudioSegmentAtTime(time) || playbackSegmentAtTime(time) || state.segments[state.segments.length - 1] || null;
      if (segment) state.activeId = segment.id;
      const sourcePath = timelineAudioPathForSegment(segment);
      if (segment && sourcePath) {
        const local = Math.max(0, Math.min(timelineAudioDurationForSegment(segment), time - timelineAudioStartForSegment(segment))) + timelineAudioSourceStartForSegment(segment);
        if (state.sceneAudioSegmentId !== segment.id) {
          sceneAudio.src = audioUrl(sourcePath);
          sceneAudio.load();
          state.sceneAudioSegmentId = segment.id;
        }
        try {
          sceneAudio.currentTime = local;
        } catch {
          // Browser may need metadata first; playback will retry on play.
        }
      } else {
        sceneAudio.pause();
        sceneAudio.removeAttribute("src");
        state.sceneAudioSegmentId = "";
      }
    } else {
      ensureGlobalTimelineAudioSource(time);
    }
  }

  return {
    appendTimelineFirstLastFrameThumbnail, beginGlobalTimelineScrub, clearActiveSegment,
    clearConceptPromptNotesFromSegments, clearI2VMotionNotesFromSegments, handlePreviewVideoLoadIssue,
    mediaThumbnailHtml, moveActiveSceneSelection, openMultiSelectChooser, playSceneAudioFrom,
    playbackSegmentAtTime, selectedSegmentImagePath,
    selectedSegmentImageThumbnailPath, setActiveSegment, setGlobalPlaybackTime, showAdjustPreviewImage,
    showFilmGrainPreviewImage, showLutPreviewImage, syncInspector, syncPreview, syncPreviewPlayback,
    updateAudioScrubbers, updateSelectedMediaTools, waitForPreviewVideoReady,
  };
}
