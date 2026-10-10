import { FACIAL_PERFORMANCE_PRESETS } from "../storyboard_builder/performance_presets.mjs";
import { audioUrl, makeEditorImageUrl } from "./comfy_api.mjs";
import { escapeHtml, makeButton, makeCheckbox, makeField, makeInput, makeSelect, toast } from "./controls.mjs";
import { showConfirmModal, showInfoModal } from "./dialogs.mjs";
import { formatDurationSeconds, formatTime } from "./format.mjs";
import { refmodPreviewUrl } from "./refmod_card.mjs";
import { isIdentityCard } from "./refmod_labels.mjs";
import { newSegment, sortSegments } from "./segments.mjs";

export function createLyricReview({
  activeSegment, applyIngredientsReferenceMappings, applyLyricSectionsFromReferenceText, audioInput,
  currentGlobalTime, currentVideoMode, ensureAllSegmentRuntimeFields, ensureSegmentRuntimeFields,
  hasLockedVideo, isInstrumentalLyricText, isNoLipSyncSingerChoice, normalizeFluxReferenceBuilder,
  openStoryboardBuilderFromProject, parseBulkTimeValue, pushHistory, referenceBuilderSubjectChoices, render, saveSession, segmentTrack, state,
  syncIngredientsSceneMapFromSubjectMappings, syncInspector, syncLyricMapperFromSegments, timelineDuration,
}) {
  let activeLyricReviewBackdrop = null;
  // What each scene looked like when its scene beat / LLM prompt were last replaced (or first seen): sceneId -> { beat, llm }
  // snapshots. A button is red only while the scene differs from its snapshot, so undoing an edit clears the red.
  // It outlives the window, so a scene stays red when the window is closed and opened again.
  const storyBaselines = new Map();

  function openLyricReviewModal(options = {}) {
    if (activeLyricReviewBackdrop?.isConnected) return;

    const focusSceneId = String(options?.focusSceneId || options?.focus_scene_id || "").trim();
    const singleSceneId = String(options?.singleSceneId || options?.single_scene_id || options?.sceneId || "").trim();
    ensureAllSegmentRuntimeFields();
    const allScenes = [...state.segments].sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    const targetScene = singleSceneId ? allScenes.find((item) => item.id === singleSceneId) : null;
    const isSingleScene = Boolean(targetScene);
    const targetSceneIndex = targetScene ? allScenes.findIndex((item) => item.id === targetScene.id) : -1;
    const scenes = isSingleScene ? [targetScene] : allScenes;
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = isSingleScene
      ? "width:min(calc(1680px + var(--vrg-identity-extra,0px)),calc(100vw - 24px));max-height:calc(100vh - 32px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;box-sizing:border-box;"
      : "width:min(calc(1840px + var(--vrg-identity-extra,0px)),calc(100vw - 16px));max-height:calc(100vh - 32px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;box-sizing:border-box;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    if (isSingleScene) {
      heading.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Review Lines + Map Performers — ${escapeHtml(targetScene.label || `Scene ${targetSceneIndex + 1}`)}</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Listen to this scene, correct lyrics/dialogue, assign performers or speakers, and set lip-sync or location details.</div>`;
    } else {
      heading.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Review Lines + Map Performers</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Listen scene by scene, correct transcribed lyrics/dialogue, assign performers or speakers, and mark instrumental or B-roll before running Gemma.</div>`;
    }
    const lyricReviewHint = makeButton("?");
    lyricReviewHint.title = "Explain lyric review controls";
    lyricReviewHint.style.width = "44px";
    const close = makeButton("Close");
    const openAllButton = isSingleScene ? makeButton("Open All Scenes") : null;
    if (openAllButton) {
      openAllButton.title = "Save this scene and open the full Review Lines + Map Performers window.";
      openAllButton.onclick = () => navigateReviewScene({ focusSceneId: targetScene.id });
      header.append(heading, openAllButton, lyricReviewHint, close);
    } else {
      header.append(heading, lyricReviewHint, close);
    }

    lyricReviewHint.onclick = () => showInfoModal({
      title: "Lyric Review Help",
      lines: [
        "Use this window to fix the line text, scene timing, performers/speakers, locations, and no-lip-sync flags before Gemma creates video prompts.",
        "Timing edit mode: Lock rest only changes the shared boundary between two neighboring scenes. Ripple following scenes shifts every later scene and is mainly for fixing timeline drift.",
        "Play Scene plays only that scene's current start/end range. Play From Here starts at that scene and keeps playing so you can stop where the next boundary should be.",
        "Set Start uses the audio playhead as this scene's start and moves the previous scene's end to match. Set End uses the audio playhead as this scene's end.",
        "Split At Playhead cuts that scene into two scenes at the current audio playhead. It copies line text, performer/speaker mapping, B-roll/instrumental state, motion notes, and location mapping into both pieces.",
        "Instrumental means nobody should sing or lip-sync in that scene. B-roll / no lip-sync means the scene can have visuals or movement, but visible people should not mouth the lyric.",
        "Performer/speaker choices tell Gemma who should sing, say, or carry the line depending on the global Video Type. Location connects the scene to a Reference Builder location image.",
        "Copy boundary words is optional. It appends the first word or words from the next vocal scene onto the current scene, which can help LTX warm-up and cooldown frames keep lyric context.",
        "Move end words to next is text-only. It removes the last word or words from each lyric scene and prepends them to the next vocal scene when the transcript split landed too early. If the next scene is instrumental or no-lip-sync, the words are only removed and are not added to that scene.",
        "Move last word to start of next scene changes only that row and its immediate neighbor. Press the main Save button afterward to update the timeline, cue maps, and every saved lyric note file.",
        "Save Lines + Timing + Performers + Locations applies the edited rows to the real timeline and saves the project.",
      ],
    });

    const note = document.createElement("div");
    note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:9px;";
    note.textContent = isSingleScene
      ? "These fields are the line notes Gemma uses for I2V/T2V prompting for this scene. Fix typos or timing mistakes here, then choose who performs or speaks. Use B-roll or Instrumental when nobody should lip-sync. Switching scenes saves your changes."
      : "These fields are the timeline line notes Gemma uses for I2V/T2V prompting. Fix typos or timing mistakes here, then choose who performs or speaks in each scene. Use B-roll or Instrumental when nobody should lip-sync.";

    const reviewReferenceBuilder = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    if (!reviewReferenceBuilder.subject_scene_map || typeof reviewReferenceBuilder.subject_scene_map !== "object") reviewReferenceBuilder.subject_scene_map = {};
    const singleReviewSubject = Array.isArray(reviewReferenceBuilder.subjects) && reviewReferenceBuilder.subjects.length === 1 ? reviewReferenceBuilder.subjects[0] : null;
    const performerLabelPanel = document.createElement("div");
    performerLabelPanel.style.cssText = `display:${singleReviewSubject ? "grid" : "none"};grid-template-columns:minmax(220px,360px) minmax(0,1fr);gap:10px;align-items:center;border:1px solid #334155;border-radius:7px;background:#0b1220;padding:9px;`;
    const performerLabelInput = makeInput(singleReviewSubject?.name && singleReviewSubject.name !== "Character 1" ? singleReviewSubject.name : "the performer");
    performerLabelInput.placeholder = "the woman, the man, the performer, lead character...";
    const performerLabelHelp = document.createElement("div");
    performerLabelHelp.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    performerLabelHelp.textContent = "This is the phrase Gemma/LTX sees for your single character. Use natural wording like the woman, the man, or the performer instead of Character 1.";
    performerLabelPanel.append(makeField("Single character performer label", performerLabelInput), performerLabelHelp);
    const performerLabelValue = () => String(performerLabelInput.value || singleReviewSubject?.name || "the performer").trim() || "the performer";
    const isSingleSubjectChoice = (choice) => Boolean(singleReviewSubject?.id && choice?.id === singleReviewSubject.id);
    const reviewChoiceLabel = (choice) => isSingleSubjectChoice(choice) ? performerLabelValue() : String(choice?.label || "");
    const syncSingleSubjectPerformerLabel = () => {
      if (!singleReviewSubject) return;
      const nextLabel = performerLabelValue();
      singleReviewSubject.name = nextLabel;
      if (reviewReferenceBuilder.subjects?.[0]) reviewReferenceBuilder.subjects[0].name = nextLabel;
      if (reviewReferenceBuilder.subject_count === 1 && reviewReferenceBuilder.subjects?.[0]) {
        reviewReferenceBuilder.subjects[0].name = nextLabel;
      }
      state.fluxReferenceBuilder = reviewReferenceBuilder;
    };
    performerLabelInput.addEventListener("input", () => {
      if (!singleReviewSubject?.id) return;
      const nextLabel = performerLabelValue();
      for (const input of box.querySelectorAll(`[data-review-subject-id="${CSS.escape(singleReviewSubject.id)}"]`)) {
        input.value = nextLabel;
      }
      for (const label of box.querySelectorAll(`[data-review-subject-label="${CSS.escape(singleReviewSubject.id)}"]`)) {
        label.textContent = nextLabel;
      }
    });

    const timingModePanel = document.createElement("div");
    timingModePanel.style.cssText = "display:grid;grid-template-columns:minmax(220px,280px) minmax(260px,1fr) minmax(360px,420px) minmax(280px,360px);gap:10px;align-items:center;border:1px solid #334155;border-radius:7px;background:#0b1220;padding:9px;";
    const timingModeSelect = makeSelect(["lock", "ripple"], "lock");
    timingModeSelect.options[0].textContent = "Lock rest of timeline";
    timingModeSelect.options[1].textContent = "Ripple following scenes";
    const timingModeHelp = document.createElement("div");
    timingModeHelp.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    const updateTimingModeHelp = () => {
      timingModeHelp.textContent = timingModeSelect.value === "ripple"
        ? "Ripple moves every following scene when you change an end time. Use this only when the timeline is globally drifting."
        : "Lock rest only moves the shared boundary between neighboring scenes. Later scenes keep their timing.";
    };
    timingModeSelect.onchange = updateTimingModeHelp;
    updateTimingModeHelp();
    const boundaryOverlapCheckbox = makeCheckbox("Copy boundary words", false);
    const boundaryOverlapCount = makeInput("1", "number");
    boundaryOverlapCount.min = "1";
    boundaryOverlapCount.max = "5";
    boundaryOverlapCount.step = "1";
    boundaryOverlapCount.disabled = true;
    boundaryOverlapCount.title = "How many words to copy from the start of the next vocal scene.";
    const boundaryOverlapApply = makeButton("Apply");
    boundaryOverlapApply.style.cssText = "padding:7px 8px;min-width:68px;";
    const boundaryOverlapWrap = document.createElement("div");
    boundaryOverlapWrap.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 70px auto;gap:6px;align-items:end;";
    boundaryOverlapWrap.append(boundaryOverlapCheckbox.wrapper, makeField("Words", boundaryOverlapCount), boundaryOverlapApply);
    const moveTailWordsCheckbox = makeCheckbox("Move end words to next", false);
    const moveTailWordsCount = makeInput("1", "number");
    moveTailWordsCount.min = "1";
    moveTailWordsCount.max = "8";
    moveTailWordsCount.step = "1";
    moveTailWordsCount.disabled = true;
    moveTailWordsCount.title = "How many words to move from each scene end to the next scene start.";
    const moveTailWordsApply = makeButton("Apply");
    moveTailWordsApply.style.cssText = "padding:7px 8px;min-width:68px;";
    const moveTailWordsWrap = document.createElement("div");
    moveTailWordsWrap.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 70px auto;gap:6px;align-items:end;";
    moveTailWordsWrap.append(moveTailWordsCheckbox.wrapper, makeField("Words", moveTailWordsCount), moveTailWordsApply);
    const lyricWordToolsWrap = document.createElement("div");
    lyricWordToolsWrap.style.cssText = "display:flex;flex-direction:column;gap:8px;";
    lyricWordToolsWrap.append(boundaryOverlapWrap, moveTailWordsWrap);
    const boundaryOverlapHelp = document.createElement("div");
    boundaryOverlapHelp.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    boundaryOverlapHelp.textContent = "Copy duplicates next-scene starter words for warm-up/cooldown context. Move removes end words from one scene and puts them at the start of the next vocal scene. Instrumental/no-lip-sync targets stay lyric-free.";
    boundaryOverlapCheckbox.input.onchange = () => {
      boundaryOverlapCount.disabled = !boundaryOverlapCheckbox.input.checked;
    };
    moveTailWordsCheckbox.input.onchange = () => {
      moveTailWordsCount.disabled = !moveTailWordsCheckbox.input.checked;
    };
    if (isSingleScene) {
      lyricWordToolsWrap.style.display = "none";
      boundaryOverlapHelp.style.display = "none";
      timingModePanel.style.gridTemplateColumns = "minmax(220px,280px) minmax(260px,1fr)";
    }
    timingModePanel.append(makeField("Timing edit mode", timingModeSelect), timingModeHelp, lyricWordToolsWrap, boundaryOverlapHelp);

    const audioPanel = document.createElement("div");
    audioPanel.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) repeat(3,auto);gap:8px;align-items:center;border:1px solid #334155;border-radius:7px;background:#0b1220;padding:9px;";
    const reviewAudio = document.createElement("audio");
    reviewAudio.controls = true;
    reviewAudio.preload = "metadata";
    reviewAudio.style.cssText = "width:100%;height:34px;";
    const audioPath = String(audioInput.value || state.audioPath || "").trim();
    if (audioPath) {
      reviewAudio.src = audioUrl(audioPath);
    } else {
      reviewAudio.style.display = "none";
      const noAudio = document.createElement("div");
      noAudio.textContent = "No project audio loaded yet.";
      noAudio.style.cssText = "font-size:12px;color:#94a3b8;";
      audioPanel.append(noAudio);
    }
    let reviewStopAt = null;
    let reviewStopTimer = null;
    let reviewStopRaf = null;
    let activeReviewPlayButton = null;
    let activeReviewPlayLabel = "";
    const clearReviewStopGuards = () => {
      if (reviewStopTimer != null) window.clearTimeout(reviewStopTimer);
      if (reviewStopRaf != null) window.cancelAnimationFrame(reviewStopRaf);
      reviewStopTimer = null;
      reviewStopRaf = null;
    };
    const resetReviewPlayButton = () => {
      if (activeReviewPlayButton) activeReviewPlayButton.textContent = activeReviewPlayLabel || "Play";
      activeReviewPlayButton = null;
      activeReviewPlayLabel = "";
    };
    const setReviewPlayButton = (button, label) => {
      resetReviewPlayButton();
      activeReviewPlayButton = button || null;
      activeReviewPlayLabel = label || "";
      if (activeReviewPlayButton) activeReviewPlayButton.textContent = "Pause";
    };
    const stopReviewAtBoundary = () => {
      const end = reviewStopAt;
      clearReviewStopGuards();
      reviewStopAt = null;
      reviewAudio.pause();
      if (Number.isFinite(end)) {
        try {
          reviewAudio.currentTime = end;
        } catch {
          // Some browsers reject seeks while media metadata is still settling.
        }
      }
    };
    const armReviewStopGuard = (start, end) => {
      clearReviewStopGuards();
      const safeStart = Math.max(0, Number(start) || 0);
      const safeEnd = Math.max(safeStart + 0.1, Number(end) || safeStart + 0.1);
      reviewStopTimer = window.setTimeout(stopReviewAtBoundary, Math.max(40, (safeEnd - safeStart) * 1000 + 30));
      const tick = () => {
        if (reviewStopAt == null) return;
        if (reviewAudio.currentTime >= reviewStopAt - 0.012) {
          stopReviewAtBoundary();
          return;
        }
        reviewStopRaf = window.requestAnimationFrame(tick);
      };
      reviewStopRaf = window.requestAnimationFrame(tick);
    };
    reviewAudio.addEventListener("timeupdate", () => {
      if (reviewStopAt == null) return;
      if (reviewAudio.currentTime >= reviewStopAt - 0.012) stopReviewAtBoundary();
    });
    reviewAudio.addEventListener("pause", () => {
      clearReviewStopGuards();
      resetReviewPlayButton();
    });
    reviewAudio.addEventListener("ended", () => {
      clearReviewStopGuards();
      resetReviewPlayButton();
    });
    const reviewTimingForSegment = (segment) => {
      const fallbackStart = Math.max(0, Number(segment?.start || 0));
      const fallbackEnd = Math.max(fallbackStart + 0.1, Number(segment?.end || fallbackStart + 4));
      if (!segment?.id || !rowList) return { start: fallbackStart, end: fallbackEnd };
      const row = rowList.querySelector(`[data-review-segment-id="${CSS.escape(segment.id)}"]`);
      if (!row) return { start: fallbackStart, end: fallbackEnd };
      const start = parseBulkTimeValue(row.querySelector("[data-review-start]")?.value);
      const end = parseBulkTimeValue(row.querySelector("[data-review-end]")?.value);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start + 0.05) {
        return { start: fallbackStart, end: fallbackEnd };
      }
      return { start: Math.max(0, start), end };
    };
    const playRange = (segment, button = null) => {
      if (!reviewAudio.src) {
        toast("Load audio first.", true);
        return;
      }
      if (button && activeReviewPlayButton === button && !reviewAudio.paused) {
        clearReviewStopGuards();
        reviewAudio.pause();
        reviewStopAt = null;
        resetReviewPlayButton();
        return;
      }
      const timing = reviewTimingForSegment(segment);
      const start = Math.max(0, timing.start);
      const end = Math.max(start + 0.1, timing.end);
      let started = false;
      const beginPlayback = () => {
        if (started) return;
        started = true;
        reviewStopAt = end;
        armReviewStopGuard(start, end);
        reviewAudio.play().catch((error) => toast(String(error?.message || error), true));
      };
      clearReviewStopGuards();
      reviewAudio.pause();
      reviewStopAt = end;
      reviewAudio.currentTime = start;
      if (button) setReviewPlayButton(button, button.dataset.playLabel || button.textContent || "Play Scene");
      reviewAudio.addEventListener("seeked", beginPlayback, { once: true });
      window.setTimeout(beginPlayback, 45);
    };
    const playFromSegment = (segment, button = null) => {
      if (!reviewAudio.src) {
        toast("Load audio first.", true);
        return;
      }
      if (button && activeReviewPlayButton === button && !reviewAudio.paused) {
        clearReviewStopGuards();
        reviewAudio.pause();
        reviewStopAt = null;
        resetReviewPlayButton();
        return;
      }
      const timing = reviewTimingForSegment(segment);
      clearReviewStopGuards();
      reviewAudio.pause();
      reviewStopAt = null;
      reviewAudio.currentTime = Math.max(0, timing.start);
      if (button) setReviewPlayButton(button, button.dataset.playLabel || button.textContent || "Play From Here");
      reviewAudio.play().catch((error) => toast(String(error?.message || error), true));
    };
    const jumpSelected = makeButton("Play Selected Scene");
    const prevScene = makeButton("Prev");
    const nextScene = makeButton("Next");
    const selectedIndex = () => Math.max(0, scenes.findIndex((segment) => segment.id === state.activeId));
    const setActiveReviewScene = (segment) => {
      if (!segment) return;
      state.activeId = segment.id;
      state.activeTrack = segmentTrack(segment);
      syncInspector();
      render();
      const row = rowList.querySelector(`[data-review-segment-id="${CSS.escape(segment.id)}"]`);
      row?.scrollIntoView({ block: "center", behavior: "smooth" });
    };
    jumpSelected.dataset.playLabel = "Play Selected Scene";
    jumpSelected.onclick = () => playRange(activeSegment() || scenes[0], jumpSelected);
    prevScene.onclick = () => {
      const index = selectedIndex();
      setActiveReviewScene(scenes[Math.max(0, index - 1)]);
    };
    nextScene.onclick = () => {
      const index = selectedIndex();
      setActiveReviewScene(scenes[Math.min(scenes.length - 1, index + 1)]);
    };
    if (isSingleScene) {
      jumpSelected.textContent = "Play Scene";
      jumpSelected.dataset.playLabel = "Play Scene";
      jumpSelected.onclick = () => playRange(targetScene, jumpSelected);
      prevScene.textContent = "Prev Scene";
      nextScene.textContent = "Next Scene";
      prevScene.title = "Save this scene and open the previous scene.";
      nextScene.title = "Save this scene and open the next scene.";
      prevScene.disabled = targetSceneIndex <= 0;
      nextScene.disabled = targetSceneIndex >= allScenes.length - 1;
      prevScene.onclick = () => {
        if (targetSceneIndex > 0) {
          return navigateReviewScene({ singleSceneId: allScenes[targetSceneIndex - 1].id });
        }
      };
      nextScene.onclick = () => {
        if (targetSceneIndex < allScenes.length - 1) {
          return navigateReviewScene({ singleSceneId: allScenes[targetSceneIndex + 1].id });
        }
      };
      if (audioPath) {
        const targetStart = Math.max(0, Number(targetScene.start || 0));
        const cueAudio = () => {
          try { reviewAudio.currentTime = targetStart; } catch {}
        };
        if (reviewAudio.readyState >= 1) cueAudio();
        else reviewAudio.addEventListener("loadedmetadata", cueAudio, { once: true });
      }
    }
    if (audioPath) audioPanel.append(reviewAudio);
    audioPanel.append(prevScene, jumpSelected, nextScene);

    const rowList = document.createElement("div");
    rowList.style.cssText = isSingleScene
      ? "display:flex;flex-direction:column;gap:8px;padding-right:4px;"
      : "display:flex;flex-direction:column;gap:8px;max-height:56vh;overflow-y:auto;overflow-x:hidden;padding-right:4px;";

    const reviewPlayheadTime = () => {
      if (Number.isFinite(Number(reviewAudio?.currentTime))) return Math.max(0, Number(reviewAudio.currentTime));
      return Math.max(0, Number(currentGlobalTime() || 0));
    };

    const reviewRows = () => [...rowList.querySelectorAll("[data-review-segment-id]")];
    const liveReviewSegmentForRow = (row) => state.segments.find((item) => item.id === row?.dataset?.reviewSegmentId) || null;

    const syncReviewRowFromSegment = (row) => {
      const segment = liveReviewSegmentForRow(row);
      if (!segment) return;
      const startInput = row.querySelector("[data-review-start]");
      const endInput = row.querySelector("[data-review-end]");
      if (startInput) startInput.value = formatTime(segment.start);
      if (endInput) endInput.value = formatTime(segment.end);
      updateReviewTimingDisplay(row);
      rememberReviewRowTiming(row);
    };

    const syncAllReviewRowsFromSegments = () => {
      for (const row of reviewRows()) syncReviewRowFromSegment(row);
    };

    const updateReviewTimingDisplay = (row) => {
      const start = parseBulkTimeValue(row.querySelector("[data-review-start]")?.value);
      const end = parseBulkTimeValue(row.querySelector("[data-review-end]")?.value);
      const display = row.querySelector("[data-review-time-display]");
      if (!display) return;
      display.textContent = Number.isFinite(start) && Number.isFinite(end) && end > start
        ? `${formatTime(start)} - ${formatTime(end)} | ${formatDurationSeconds(start, end)}s`
        : "Invalid timing";
    };

    const rememberReviewRowTiming = (row) => {
      const start = parseBulkTimeValue(row.querySelector("[data-review-start]")?.value);
      const end = parseBulkTimeValue(row.querySelector("[data-review-end]")?.value);
      if (Number.isFinite(start)) row.dataset.reviewLastStart = String(start);
      if (Number.isFinite(end)) row.dataset.reviewLastEnd = String(end);
    };

    const shiftReviewRowsAfter = (row, delta) => {
      if (!Number.isFinite(delta) || Math.abs(delta) < 0.0001) return;
      const rows = reviewRows();
      const rowIndex = rows.indexOf(row);
      for (const nextRow of rows.slice(rowIndex + 1)) {
        const nextSegment = liveReviewSegmentForRow(nextRow);
        if (nextSegment) {
          nextSegment.start = Math.max(0, Number(nextSegment.start || 0) + delta);
          nextSegment.end = Math.max(nextSegment.start + 0.05, Number(nextSegment.end || nextSegment.start + 0.05) + delta);
        }
        syncReviewRowFromSegment(nextRow);
      }
    };

    const showShortReviewSceneConfirm = (sceneName, seconds, mergeText) => new Promise((resolve) => {
      const warnBackdrop = document.createElement("div");
      warnBackdrop.style.cssText = "position:fixed;inset:0;z-index:100009;background:rgba(0,0,0,.64);display:flex;align-items:center;justify-content:center;";
      const warnBox = document.createElement("div");
      warnBox.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #991b1b;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const warnHeading = document.createElement("div");
      warnHeading.textContent = "Scene is shorter than 2 seconds";
      warnHeading.style.cssText = "font-size:16px;font-weight:900;color:#fecaca;";
      const warnBody = document.createElement("div");
      warnBody.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;display:flex;flex-direction:column;gap:8px;";
      const line1 = document.createElement("div");
      line1.textContent = `${sceneName} is now ${seconds.toFixed(2)} seconds. Very short scenes can cut off lyrics or make LTX timing feel jumpy.`;
      const line2 = document.createElement("div");
      line2.textContent = mergeText;
      warnBody.append(line1, line2);
      const warnActions = document.createElement("div");
      warnActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
      const keep = makeButton("Keep Short Scene");
      const merge = makeButton("Merge Scenes", "primary");
      keep.onclick = () => {
        warnBackdrop.remove();
        resolve(false);
      };
      merge.onclick = () => {
        warnBackdrop.remove();
        resolve(true);
      };
      warnActions.append(keep, merge);
      warnBox.append(warnHeading, warnBody, warnActions);
      warnBackdrop.append(warnBox);
      document.body.append(warnBackdrop);
    });

    const reviewSegmentForRow = (row) => scenes.find((item) => item.id === row?.dataset?.reviewSegmentId) || null;

    const reviewRowLyricText = (row) => {
      const instrumental = Boolean(row.querySelector("[data-review-instrumental='1']")?.checked);
      if (instrumental) return "[instrumental]";
      return String(row.querySelector("[data-review-lyric-text]")?.value || "").trim();
    };

    const reviewRowRawLyricText = (row) => {
      const instrumental = Boolean(row.querySelector("[data-review-instrumental='1']")?.checked);
      if (instrumental) return "[instrumental]";
      const text = row.querySelector("[data-review-lyric-text]");
      return String(text?.dataset?.reviewRawLyricText ?? text?.value ?? "");
    };

    const reviewRowBlocksLipSync = (row) => {
      return Boolean(row.querySelector("[data-review-instrumental='1']")?.checked || row.querySelector("[data-review-broll='1']")?.checked);
    };

    const lyricWords = (value) => String(value || "").trim().split(/\s+/).filter(Boolean);
    const normalizeLyricWord = (value) => String(value || "").toLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");
    const lyricEndsWithWords = (text, words) => {
      const textWords = lyricWords(text);
      if (!words.length || textWords.length < words.length) return false;
      const tail = textWords.slice(-words.length).map(normalizeLyricWord);
      const compare = words.map(normalizeLyricWord);
      return tail.every((word, index) => word && word === compare[index]);
    };
    const lyricStartsWithWords = (text, words) => {
      const textWords = lyricWords(text);
      if (!words.length || textWords.length < words.length) return false;
      const head = textWords.slice(0, words.length).map(normalizeLyricWord);
      const compare = words.map(normalizeLyricWord);
      return head.every((word, index) => word && word === compare[index]);
    };

    const boundaryWordCountValue = () => {
      const count = Math.floor(Number(boundaryOverlapCount.value || 1));
      return Math.max(1, Math.min(5, Number.isFinite(count) ? count : 1));
    };
    const moveTailWordCountValue = () => {
      const count = Math.floor(Number(moveTailWordsCount.value || 1));
      return Math.max(1, Math.min(8, Number.isFinite(count) ? count : 1));
    };

    const setReviewRowRawLyricText = (row, value) => {
      const text = row?.querySelector("[data-review-lyric-text]");
      if (!text) return;
      const nextValue = String(value || "");
      text.dataset.reviewRawLyricText = nextValue;
      text.value = nextValue;
    };

    const restoreReviewRowsToRawLyricText = () => {
      applyingBoundaryOverlapPreview = true;
      try {
        for (const row of reviewRows()) {
          const text = row.querySelector("[data-review-lyric-text]");
          if (!text) continue;
          text.value = reviewRowRawLyricText(row);
        }
      } finally {
        applyingBoundaryOverlapPreview = false;
      }
    };

    const collectBoundaryOverlapLyricOverrides = () => {
      const rows = reviewRows();
      const overrides = new Map();
      const entries = rows.map((row) => {
        const segment = reviewSegmentForRow(row);
        const text = reviewRowRawLyricText(row);
        const blocked = reviewRowBlocksLipSync(row) || isInstrumentalLyricText(text);
        if (segment) overrides.set(segment.id, text);
        return { row, segment, text, blocked };
      });
      if (!boundaryOverlapCheckbox.input.checked) return overrides;
      const count = boundaryWordCountValue();
      for (let index = 0; index < entries.length - 1; index++) {
        const current = entries[index];
        const next = entries[index + 1];
        if (!current.segment || current.blocked || !current.text || !next || next.blocked || !next.text) continue;
        const words = lyricWords(next.text).slice(0, count);
        if (!words.length || lyricEndsWithWords(current.text, words)) continue;
        overrides.set(current.segment.id, `${current.text.trim()} ${words.join(" ")}`.trim());
      }
      return overrides;
    };

    let applyingBoundaryOverlapPreview = false;
    const refreshBoundaryOverlapPreview = () => {
      const overrides = collectBoundaryOverlapLyricOverrides();
      applyingBoundaryOverlapPreview = true;
      try {
        for (const row of reviewRows()) {
          const segment = reviewSegmentForRow(row);
          const text = row.querySelector("[data-review-lyric-text]");
          if (!segment || !text) continue;
          const nextText = overrides.get(segment.id);
          text.value = nextText == null ? reviewRowRawLyricText(row) : nextText;
        }
      } finally {
        applyingBoundaryOverlapPreview = false;
      }
    };

    const applyMoveTailWordsToNext = () => {
      if (!moveTailWordsCheckbox.input.checked) {
        toast("Turn on Move end words to next, then press Apply.", true);
        return;
      }
      restoreReviewRowsToRawLyricText();
      const count = moveTailWordCountValue();
      const rows = reviewRows();
      let moved = 0;
      let removedOnly = 0;
      for (let index = 0; index < rows.length - 1; index++) {
        const current = rows[index];
        const next = rows[index + 1];
        const currentText = reviewRowRawLyricText(current);
        const nextText = reviewRowRawLyricText(next);
        const currentBlocked = reviewRowBlocksLipSync(current) || isInstrumentalLyricText(currentText);
        const nextBlocked = reviewRowBlocksLipSync(next) || isInstrumentalLyricText(nextText);
        if (currentBlocked) continue;
        const words = lyricWords(currentText);
        if (words.length <= count) continue;
        const movedWords = words.slice(-count);
        const remainingWords = words.slice(0, -count);
        setReviewRowRawLyricText(current, remainingWords.join(" "));
        if (nextBlocked) {
          removedOnly += 1;
        } else if (!lyricStartsWithWords(nextText, movedWords)) {
          setReviewRowRawLyricText(next, `${movedWords.join(" ")} ${nextText}`.trim());
          moved += 1;
        } else {
          moved += 1;
        }
      }
      refreshBoundaryOverlapPreview();
      const parts = [];
      if (moved) parts.push(`moved into ${moved} next scene${moved === 1 ? "" : "s"}`);
      if (removedOnly) parts.push(`removed before ${removedOnly} instrumental/no-lip-sync scene${removedOnly === 1 ? "" : "s"}`);
      toast(parts.length ? `End words ${parts.join(" and ")}. Press Save to keep it.` : "No eligible lyric rows needed word moves.", !parts.length);
    };

    const pendingReviewWordMoves = [];
    const moveLastWordToNextReviewRow = (currentRow) => {
      restoreReviewRowsToRawLyricText();
      const rows = reviewRows();
      const currentIndex = rows.indexOf(currentRow);
      const nextRow = rows[currentIndex + 1] || null;
      if (currentIndex < 0 || !nextRow) {
        toast("There is no next scene to receive the last word.", true);
        return;
      }
      const currentText = reviewRowRawLyricText(currentRow);
      const nextText = reviewRowRawLyricText(nextRow);
      if (reviewRowBlocksLipSync(currentRow) || isInstrumentalLyricText(currentText)) {
        toast("This scene is instrumental or no-lip-sync, so it has no movable lyric word.", true);
        return;
      }
      if (reviewRowBlocksLipSync(nextRow) || isInstrumentalLyricText(nextText)) {
        toast("The next scene is instrumental or no-lip-sync. Change that scene before moving a lyric word into it.", true);
        return;
      }
      const words = lyricWords(currentText);
      if (!words.length) {
        toast("This scene has no lyric word to move.", true);
        return;
      }
      const movedWord = words.pop();
      const nextWords = lyricWords(nextText);
      setReviewRowRawLyricText(currentRow, words.join(" "));
      if (!nextWords.length || normalizeLyricWord(nextWords[0]) !== normalizeLyricWord(movedWord)) {
        setReviewRowRawLyricText(nextRow, `${movedWord} ${nextText}`.trim());
      }
      pendingReviewWordMoves.push({
        sourceId: String(currentRow.dataset.reviewSegmentId || ""),
        targetId: String(nextRow.dataset.reviewSegmentId || ""),
        word: movedWord,
      });
      refreshBoundaryOverlapPreview();
      toast(`Moved “${movedWord}” to the start of the next scene. Press Save Lines to update every lyric note.`);
    };

    const moveWordAcrossStructuredLyricRows = (sourceRows, targetRows, movedWord) => {
      if (!Array.isArray(sourceRows) || !Array.isArray(targetRows) || !movedWord) return;
      const source = [...sourceRows].reverse().find((cue) => cue && String(cue.type || "vocal") !== "instrumental" && String(cue.text || "").trim());
      if (source) {
        const words = lyricWords(source.text);
        if (words.length && normalizeLyricWord(words[words.length - 1]) === normalizeLyricWord(movedWord)) {
          words.pop();
          source.text = words.join(" ");
        }
      }
      const target = targetRows.find((cue) => cue && String(cue.type || "vocal") !== "instrumental") || null;
      if (target) {
        const words = lyricWords(target.text);
        if (!words.length || normalizeLyricWord(words[0]) !== normalizeLyricWord(movedWord)) {
          target.text = `${movedWord} ${String(target.text || "")}`.trim();
        }
      }
    };

    const applyPendingReviewWordMoves = (segmentsById) => {
      for (const move of pendingReviewWordMoves) {
        const source = segmentsById.get(move.sourceId);
        const target = segmentsById.get(move.targetId);
        if (!source || !target) continue;
        moveWordAcrossStructuredLyricRows(source.lyric_cue_map, target.lyric_cue_map, move.word);
        moveWordAcrossStructuredLyricRows(source.minimax_speaker_assignments, target.minimax_speaker_assignments, move.word);
        moveWordAcrossStructuredLyricRows(source.speaker_assignments, target.speaker_assignments, move.word);
        moveWordAcrossStructuredLyricRows(source.dialogue_cues, target.dialogue_cues, move.word);
      }
    };

    boundaryOverlapCheckbox.input.onchange = () => {
      boundaryOverlapCount.disabled = !boundaryOverlapCheckbox.input.checked;
      if (!boundaryOverlapCheckbox.input.checked) refreshBoundaryOverlapPreview();
    };
    boundaryOverlapApply.onclick = () => {
      const overrides = collectBoundaryOverlapLyricOverrides();
      let applied = 0;
      for (const row of reviewRows()) {
        const segment = reviewSegmentForRow(row);
        if (!segment || !overrides.has(segment.id)) continue;
        const nextText = String(overrides.get(segment.id) || "");
        if (nextText !== reviewRowRawLyricText(row)) applied += 1;
        setReviewRowRawLyricText(row, nextText);
      }
      refreshBoundaryOverlapPreview();
      toast(boundaryOverlapCheckbox.input.checked
        ? (applied ? `Copied boundary words into ${applied} scene${applied === 1 ? "" : "s"}. Press Save to keep it.` : "No eligible boundary words to copy.")
        : "Boundary word copy is off. Raw lyric text restored.", boundaryOverlapCheckbox.input.checked && !applied);
    };
    moveTailWordsCheckbox.input.onchange = () => {
      moveTailWordsCount.disabled = !moveTailWordsCheckbox.input.checked;
    };
    moveTailWordsApply.onclick = applyMoveTailWordsToNext;

    const mergedReviewLyricText = (a, b) => {
      const values = [a, b].map((value) => String(value || "").trim()).filter(Boolean);
      const nonInstrumental = values.filter((value) => !isInstrumentalLyricText(value));
      if (!nonInstrumental.length) return values[0] || "[instrumental]";
      return nonInstrumental.join("\n");
    };

    const checkedReviewSingers = (row) => new Set([...row.querySelectorAll("[data-review-singer-choice='1']")]
      .filter((input) => input.checked)
      .map((input) => input.value)
      .filter(Boolean));

    const refreshReviewRowLabels = () => {
      const rows = reviewRows();
      rows.forEach((row, index) => {
        const segment = liveReviewSegmentForRow(row);
        const label = `Scene ${index + 1}`;
        if (segment) segment.label = label;
        const labelEl = row.querySelector("[data-review-scene-label]");
        if (labelEl) labelEl.textContent = segment?.label || label;
        const moveWordButton = row.querySelector("[data-review-move-last-word]");
        if (moveWordButton) moveWordButton.disabled = index >= rows.length - 1;
        syncReviewRowFromSegment(row);
      });
      state.segments.forEach((segment, index) => {
        segment.label = `Scene ${index + 1}`;
      });
    };

    const mergeReviewRows = (targetRow, absorbedRow) => {
      if (!targetRow || !absorbedRow || targetRow === absorbedRow) {
        toast("Choose two different lyric review scenes to merge.", true);
        return;
      }
      const targetSegment = reviewSegmentForRow(targetRow);
      const absorbedSegment = reviewSegmentForRow(absorbedRow);
      if (!targetSegment || !absorbedSegment) return;
      if (targetSegment.id === absorbedSegment.id) {
        toast("Choose two different lyric review scenes to merge.", true);
        return;
      }
      ensureSegmentRuntimeFields(targetSegment);
      ensureSegmentRuntimeFields(absorbedSegment);
      pushHistory();
      const targetStartInput = targetRow.querySelector("[data-review-start]");
      const targetEndInput = targetRow.querySelector("[data-review-end]");
      const absorbedStart = parseBulkTimeValue(absorbedRow.querySelector("[data-review-start]")?.value);
      const absorbedEnd = parseBulkTimeValue(absorbedRow.querySelector("[data-review-end]")?.value);
      const targetStart = parseBulkTimeValue(targetStartInput?.value);
      const targetEnd = parseBulkTimeValue(targetEndInput?.value);
      let mergedStart = Number.isFinite(targetStart) ? targetStart : Number(targetSegment.start || 0);
      let mergedEnd = Number.isFinite(targetEnd) ? targetEnd : Number(targetSegment.end || mergedStart + 0.05);
      if (targetStartInput && Number.isFinite(absorbedStart) && Number.isFinite(targetStart) && absorbedStart < targetStart) {
        mergedStart = absorbedStart;
        targetStartInput.value = formatTime(mergedStart);
      }
      if (targetEndInput && Number.isFinite(absorbedEnd) && Number.isFinite(targetEnd) && absorbedEnd > targetEnd) {
        mergedEnd = absorbedEnd;
        targetEndInput.value = formatTime(mergedEnd);
      }
      targetSegment.start = Math.max(0, mergedStart);
      targetSegment.end = Math.max(targetSegment.start + 0.05, mergedEnd);
      const targetText = targetRow.querySelector("[data-review-lyric-text]");
      if (targetText) {
        targetText.value = mergedReviewLyricText(reviewRowLyricText(targetRow), reviewRowLyricText(absorbedRow));
        targetText.dataset.reviewRawLyricText = targetText.value;
      }
      const absorbedInstrumental = Boolean(absorbedRow.querySelector("[data-review-instrumental='1']")?.checked);
      const targetInstrumental = targetRow.querySelector("[data-review-instrumental='1']");
      if (targetInstrumental && !isInstrumentalLyricText(targetText?.value || "") && absorbedInstrumental) targetInstrumental.checked = false;
      const targetBroll = targetRow.querySelector("[data-review-broll='1']");
      const absorbedBroll = absorbedRow.querySelector("[data-review-broll='1']");
      if (targetBroll && absorbedBroll?.checked) targetBroll.checked = true;
      const singerUnion = checkedReviewSingers(targetRow);
      for (const singer of checkedReviewSingers(absorbedRow)) singerUnion.add(singer);
      for (const input of targetRow.querySelectorAll("[data-review-singer-choice='1']")) {
        input.checked = singerUnion.has(input.value);
      }
      const targetLocation = targetRow.querySelector("[data-review-location]");
      const absorbedLocation = absorbedRow.querySelector("[data-review-location]");
      if (targetLocation && !targetLocation.value && absorbedLocation?.value) targetLocation.value = absorbedLocation.value;
      const stateIndex = state.segments.findIndex((item) => item.id === absorbedSegment.id);
      if (stateIndex >= 0) state.segments.splice(stateIndex, 1);
      const sceneIndex = scenes.findIndex((item) => item.id === absorbedSegment.id);
      if (sceneIndex >= 0) scenes.splice(sceneIndex, 1);
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      if (refs.scene_map) delete refs.scene_map[absorbedSegment.id];
      if (refs.subject_scene_map) delete refs.subject_scene_map[absorbedSegment.id];
      if (refs.extra_scene_map) delete refs.extra_scene_map[absorbedSegment.id];
      state.fluxReferenceBuilder = refs;
      ensureSegmentRuntimeFields(targetSegment);
      absorbedRow.remove();
      refreshReviewRowLabels();
      updateReviewTimingDisplay(targetRow);
      rememberReviewRowTiming(targetRow);
      toast("Merged the two lyric review scenes. Save to apply the updated timeline.");
    };

    const maybeWarnShortReviewScene = async (shortRow, mergeTargetRow, mergeAbsorbedRow, mergeText) => {
      if (!shortRow?.isConnected || !mergeTargetRow?.isConnected || !mergeAbsorbedRow?.isConnected) return;
      if (mergeTargetRow === mergeAbsorbedRow) return;
      const start = parseBulkTimeValue(shortRow.querySelector("[data-review-start]")?.value);
      const end = parseBulkTimeValue(shortRow.querySelector("[data-review-end]")?.value);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
      const seconds = end - start;
      if (seconds >= 2) return;
      const segment = reviewSegmentForRow(shortRow);
      const sceneName = segment?.label || "This scene";
      const shouldMerge = await showShortReviewSceneConfirm(sceneName, seconds, mergeText);
      if (shouldMerge && mergeTargetRow?.isConnected && mergeAbsorbedRow?.isConnected) mergeReviewRows(mergeTargetRow, mergeAbsorbedRow);
    };

    const showMergeWithNextConfirm = (currentLabel, nextLabel) => new Promise((resolve) => {
      const confirmBackdrop = document.createElement("div");
      confirmBackdrop.style.cssText = "position:fixed;inset:0;z-index:100009;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
      const confirmBox = document.createElement("div");
      confirmBox.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #7f1d1d;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const confirmHeading = document.createElement("div");
      confirmHeading.textContent = "Merge With Next Scene?";
      confirmHeading.style.cssText = "font-size:16px;font-weight:900;color:#fecaca;";
      const confirmBody = document.createElement("div");
      confirmBody.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;";
      confirmBody.textContent = `This will combine ${currentLabel || "this scene"} and ${nextLabel || "the next scene"} into one lyric review scene. The merged scene keeps the earliest start, latest end, combined lyric text, selected characters, and location mapping. Save the lyric review after merging to apply it to the timeline.`;
      const confirmActions = document.createElement("div");
      confirmActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
      const cancel = makeButton("Cancel");
      const merge = makeButton("Merge Scenes", "danger");
      cancel.onclick = () => {
        confirmBackdrop.remove();
        resolve(false);
      };
      merge.onclick = () => {
        confirmBackdrop.remove();
        resolve(true);
      };
      confirmBackdrop.addEventListener("pointerdown", (event) => {
        if (event.target === confirmBackdrop) cancel.click();
      });
      confirmActions.append(cancel, merge);
      confirmBox.append(confirmHeading, confirmBody, confirmActions);
      confirmBackdrop.append(confirmBox);
      document.body.append(confirmBackdrop);
      cancel.focus();
    });

    const mergeReviewRowWithNext = async (row) => {
      const rows = reviewRows();
      const rowIndex = rows.indexOf(row);
      const nextRow = rows[rowIndex + 1] || null;
      if (!nextRow) {
        toast("There is no next scene to merge with.", true);
        return;
      }
      const currentSegment = reviewSegmentForRow(row);
      const nextSegment = reviewSegmentForRow(nextRow);
      const confirmed = await showMergeWithNextConfirm(currentSegment?.label, nextSegment?.label);
      if (!confirmed) return;
      mergeReviewRows(row, nextRow);
    };

    const syncReviewSharedBoundaryPreview = (row, field) => {
      const rows = reviewRows();
      const rowIndex = rows.indexOf(row);
      if (rowIndex < 0) return;
      const start = parseBulkTimeValue(row.querySelector("[data-review-start]")?.value);
      const end = parseBulkTimeValue(row.querySelector("[data-review-end]")?.value);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start + 0.05) {
        updateReviewTimingDisplay(row);
        return;
      }
      if (field === "start") {
        const prevRow = rows[rowIndex - 1] || null;
        const prevEnd = prevRow?.querySelector("[data-review-end]");
        if (prevEnd) {
          prevEnd.value = formatTime(start);
          updateReviewTimingDisplay(prevRow);
        }
      } else if (field === "end" && timingModeSelect.value !== "ripple") {
        const nextRow = rows[rowIndex + 1] || null;
        const nextStart = nextRow?.querySelector("[data-review-start]");
        if (nextStart) {
          nextStart.value = formatTime(end);
          updateReviewTimingDisplay(nextRow);
        }
      }
      updateReviewTimingDisplay(row);
    };

    const singleReviewTimingNeighbors = (row) => {
      const ordered = [...state.segments].sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
      const index = ordered.findIndex((item) => item.id === row.dataset.reviewSegmentId);
      return { previous: ordered[index - 1], following: ordered.slice(index + 1) };
    };

    const canEditSingleReviewTiming = (row, affected) => {
      if (!isSingleScene) return true;
      if (state.timingFrozen || affected.some((segment) => segment && hasLockedVideo(segment))) {
        toast("Unfreeze timing and unlock affected scene videos before changing timing.", true);
        syncReviewRowFromSegment(row);
        return false;
      }
      return true;
    };

    const handleReviewStartEdited = async (row) => {
      const startInput = row.querySelector("[data-review-start]");
      const endInput = row.querySelector("[data-review-end]");
      const start = parseBulkTimeValue(startInput?.value);
      const end = parseBulkTimeValue(endInput?.value);
      if (!Number.isFinite(start) || !Number.isFinite(end) || start >= end - 0.05) {
        toast("Start must be before this scene end.", true);
        updateReviewTimingDisplay(row);
        return;
      }
      const rows = reviewRows();
      const rowIndex = rows.indexOf(row);
      const prevRow = rows[rowIndex - 1] || null;
      const segment = liveReviewSegmentForRow(row);
      const neighbors = isSingleScene ? singleReviewTimingNeighbors(row) : null;
      if (!canEditSingleReviewTiming(row, [segment, neighbors?.previous])) return;
      if (neighbors?.previous && start < Number(neighbors.previous.start || 0) + 0.05) {
        toast("Start must leave time for the previous scene. Use Open All Scenes to merge scenes.", true);
        syncReviewRowFromSegment(row);
        return;
      }
      if (segment) {
        segment.start = Math.max(0, start);
        segment.end = Math.max(segment.start + 0.05, end);
      }
      if (prevRow) {
        const prevSegment = liveReviewSegmentForRow(prevRow);
        if (prevSegment) prevSegment.end = Math.max(Number(prevSegment.start || 0) + 0.05, start);
        syncReviewRowFromSegment(prevRow);
        await maybeWarnShortReviewScene(prevRow, prevRow, row, "Do you want to merge that previous scene with this scene instead?");
      } else if (isSingleScene) {
        if (neighbors.previous) neighbors.previous.end = Math.max(0, start);
      }
      syncReviewRowFromSegment(row);
      state.duration = timelineDuration();
      syncInspector();
      render();
    };

    const handleReviewEndEdited = async (row, previousEndOverride = null) => {
      const startInput = row.querySelector("[data-review-start]");
      const endInput = row.querySelector("[data-review-end]");
      const start = parseBulkTimeValue(startInput?.value);
      const end = parseBulkTimeValue(endInput?.value);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start + 0.05) {
        toast("End must be after this scene start.", true);
        updateReviewTimingDisplay(row);
        return;
      }
      const previousEnd = Number.isFinite(previousEndOverride)
        ? previousEndOverride
        : Number(row.dataset.reviewLastEnd || start);
      const delta = end - previousEnd;
      const segment = liveReviewSegmentForRow(row);
      const neighbors = isSingleScene ? singleReviewTimingNeighbors(row) : null;
      const following = neighbors?.following || [];
      const affected = timingModeSelect.value === "ripple" ? following : following.slice(0, 1);
      if (!canEditSingleReviewTiming(row, [segment, ...affected])) return;
      if (isSingleScene && timingModeSelect.value !== "ripple" && following[0]
        && end > Number(following[0].end || 0) - 0.05) {
        toast("End must leave time for the next scene. Use Open All Scenes to merge scenes.", true);
        syncReviewRowFromSegment(row);
        return;
      }
      if (segment) {
        segment.start = Math.max(0, start);
        segment.end = Math.max(segment.start + 0.05, end);
      }
      syncReviewRowFromSegment(row);
      const rows = reviewRows();
      const rowIndex = rows.indexOf(row);
      const nextRow = rows[rowIndex + 1] || null;
      if (timingModeSelect.value === "ripple") {
        if (nextRow) {
          shiftReviewRowsAfter(row, delta);
        } else if (isSingleScene) {
          for (const nextSeg of following) {
            nextSeg.start = Math.max(0, Number(nextSeg.start || 0) + delta);
            nextSeg.end = Math.max(nextSeg.start + 0.05, Number(nextSeg.end || nextSeg.start + 0.05) + delta);
          }
        }
        state.duration = timelineDuration();
        syncInspector();
        render();
        return;
      }
      if (nextRow) {
        const nextSegment = liveReviewSegmentForRow(nextRow);
        if (nextSegment) {
          nextSegment.start = end;
          nextSegment.end = Math.max(nextSegment.start + 0.05, Number(nextSegment.end || nextSegment.start + 0.05));
        }
        syncReviewRowFromSegment(nextRow);
        await maybeWarnShortReviewScene(nextRow, row, nextRow, "Do you want to merge it with the scene you just extended?");
      } else if (isSingleScene) {
        if (following[0]) following[0].start = end;
      }
      state.duration = timelineDuration();
      syncInspector();
      render();
    };

    const normalizeEditedReviewTiming = async () => {
      for (const row of reviewRows()) {
        if (!row.isConnected) continue;
        const start = parseBulkTimeValue(row.querySelector("[data-review-start]")?.value);
        const end = parseBulkTimeValue(row.querySelector("[data-review-end]")?.value);
        const previousStart = Number(row.dataset.reviewLastStart || start);
        const previousEnd = Number(row.dataset.reviewLastEnd || end);
        if (Number.isFinite(start) && Number.isFinite(previousStart) && Math.abs(start - previousStart) > 0.0001) {
          await handleReviewStartEdited(row);
        }
        if (Number.isFinite(end) && Number.isFinite(previousEnd) && Math.abs(end - previousEnd) > 0.0001) {
          await handleReviewEndEdited(row, previousEnd);
        }
      }
    };

    const setReviewRowStartToPlayhead = async (row) => {
      const time = reviewPlayheadTime();
      const endInput = row.querySelector("[data-review-end]");
      const startInput = row.querySelector("[data-review-start]");
      const end = parseBulkTimeValue(endInput?.value);
      if (Number.isFinite(end) && time >= end - 0.05) {
        toast("Start must be before this scene end.", true);
        return;
      }
      startInput.value = formatTime(time);
      const rows = reviewRows();
      const rowIndex = rows.indexOf(row);
      const prevRow = rows[rowIndex - 1] || null;
      await handleReviewStartEdited(row);
    };

    const setReviewRowEndToPlayhead = async (row) => {
      const time = reviewPlayheadTime();
      const startInput = row.querySelector("[data-review-start]");
      const endInput = row.querySelector("[data-review-end]");
      const start = parseBulkTimeValue(startInput?.value);
      const previousEnd = parseBulkTimeValue(endInput?.value);
      if (Number.isFinite(start) && time <= start + 0.05) {
        toast("End must be after this scene start.", true);
        return;
      }
      endInput.value = formatTime(time);
      await handleReviewEndEdited(row, previousEnd);
    };

    const applyReviewRowValues = (row, segment, includeTiming = false, options = {}) => {
      if (!segment || typeof segment !== "object") return false;
      const instrumental = Boolean(row.querySelector("[data-review-instrumental='1']")?.checked);
      const broll = Boolean(row.querySelector("[data-review-broll='1']")?.checked);
      const noCharacterPresent = Boolean(row.querySelector("[data-review-no-character='1']")?.checked);
      const lyricText = Object.prototype.hasOwnProperty.call(options, "lyricTextOverride")
        ? options.lyricTextOverride
        : row.querySelector("[data-review-lyric-text]")?.value || "";
      segment.lyric_text = instrumental ? "[instrumental]" : lyricText;
      segment.lyric_no_lip_sync = instrumental || broll;
      segment.no_character_present = noCharacterPresent;
      segment.facial_performance = String(row.querySelector("[data-review-facial-performance='1']")?.value || "").trim();
      segment.facial_performance_custom = String(row.querySelector("[data-review-facial-performance-custom='1']")?.value || "").trim();
      const checkedSingerInputs = [...row.querySelectorAll("[data-review-singer-choice='1']")]
        .filter((input) => !instrumental && !broll && !noCharacterPresent && input.checked);
      segment.lyric_singers = checkedSingerInputs
        .map((input) => input.value)
        .filter(Boolean);
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      refs.subject_scene_map = refs.subject_scene_map && typeof refs.subject_scene_map === "object" ? refs.subject_scene_map : {};
      refs.performer_scene_map = refs.performer_scene_map && typeof refs.performer_scene_map === "object" ? refs.performer_scene_map : {};
      const presentSubjectIds = [...row.querySelectorAll("[data-review-present-subject='1']")]
        .filter((input) => !noCharacterPresent && input.checked)
        .map((input) => String(input.value || "").trim())
        .filter(Boolean);
      const uniqueSubjectIds = Array.from(new Set(presentSubjectIds));
      if (!noCharacterPresent && uniqueSubjectIds.length) refs.subject_scene_map[segment.id] = uniqueSubjectIds;
      else delete refs.subject_scene_map[segment.id];
      const validSubjectIds = new Set((refs.subjects || []).map((subject) => String(subject?.id || "").trim()).filter(Boolean));
      const performerSubjectIds = [];
      for (const input of checkedSingerInputs) {
        const rawId = String(input.dataset.reviewSubjectId || "").trim();
        if (rawId === "group") {
          performerSubjectIds.push(...uniqueSubjectIds);
          continue;
        }
        if (rawId && validSubjectIds.has(rawId)) {
          performerSubjectIds.push(rawId);
          continue;
        }
        const cleanName = String(input.value || "").trim().toLowerCase();
        const byName = (refs.subjects || []).find((subject) => String(subject?.name || "").trim().toLowerCase() === cleanName);
        if (byName?.id) performerSubjectIds.push(String(byName.id));
      }
      const uniquePerformerIds = Array.from(new Set(performerSubjectIds.filter((id) => validSubjectIds.has(String(id)))));
      if (!instrumental && !broll && !noCharacterPresent && uniquePerformerIds.length) {
        refs.performer_scene_map[segment.id] = uniquePerformerIds;
        refs.subject_scene_map[segment.id] = Array.from(new Set([...(refs.subject_scene_map[segment.id] || []), ...uniquePerformerIds]));
      } else {
        delete refs.performer_scene_map[segment.id];
      }
      state.fluxReferenceBuilder = refs;
      if (includeTiming) {
        const start = parseBulkTimeValue(row.querySelector("[data-review-start]")?.value);
        const end = parseBulkTimeValue(row.querySelector("[data-review-end]")?.value);
        if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
          throw new Error(`${segment.label || "Scene"} has invalid start/end timing.`);
        }
        segment.start = Math.max(0, start);
        segment.end = Math.max(segment.start + 0.05, end);
      }
      const locationSelect = row.querySelector("[data-review-location]");
      if (locationSelect) {
        const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
        const locationId = String(locationSelect.value || "").trim();
        if (!refs.scene_map || typeof refs.scene_map !== "object") refs.scene_map = {};
        if (locationId) refs.scene_map[segment.id] = locationId;
        else delete refs.scene_map[segment.id];
        state.fluxReferenceBuilder = refs;
      }
      return true;
    };

    const copyLyricReviewFields = (source, start, end) => {
      const target = newSegment(start, end);
      target.label = source.label || "";
      target.source = source.source || "lyric_review_split";
      target.timeline_note = source.timeline_note || "";
      target.lyric_text = source.lyric_text || "";
      target.lyric_singers = Array.isArray(source.lyric_singers) ? [...source.lyric_singers] : [];
      target.lyric_no_lip_sync = Boolean(source.lyric_no_lip_sync);
      target.no_character_present = Boolean(source.no_character_present);
      target.facial_performance = source.facial_performance || "";
      target.facial_performance_custom = source.facial_performance_custom || "";
      target.i2v_notes = source.i2v_notes || "";
      return target;
    };

    const splitReviewRowAtPlayhead = async (row, segment) => {
      if (!canEditSingleReviewTiming(row, [segment])) return;
      const segmentStart = Number(segment.start || 0);
      const segmentEnd = Number(segment.end || 0);
      const splitTime = Math.max(segmentStart, Math.min(segmentEnd, Number(reviewAudio?.currentTime || currentGlobalTime() || 0)));
      if (splitTime - segmentStart < 0.05 || segmentEnd - splitTime < 0.05) {
        toast("Move the review audio playhead inside this scene first.", true);
        return;
      }
      try {
        pushHistory();
        applyReviewRowValues(row, segment, false);
        const index = state.segments.findIndex((item) => item.id === segment.id);
        if (index < 0) throw new Error("Could not find the selected scene in the base timeline.");
        const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
        const oldLocationId = refs.scene_map?.[segment.id] || "";
        const oldSubjectIds = Array.isArray(refs.subject_scene_map?.[segment.id]) ? [...refs.subject_scene_map[segment.id]] : [];
        const oldExtraEntries = Array.isArray(refs.extra_scene_map?.[segment.id]) ? refs.extra_scene_map[segment.id].map((entry) => ({ ...entry })) : [];
        const pieces = [];
        const before = copyLyricReviewFields(segment, segmentStart, splitTime);
        const after = copyLyricReviewFields(segment, splitTime, segmentEnd);
        pieces.push(before, after);
        state.segments.splice(index, 1, ...pieces);
        if (oldLocationId) {
          refs.scene_map[before.id] = oldLocationId;
          refs.scene_map[after.id] = oldLocationId;
          delete refs.scene_map[segment.id];
        }
        if (oldSubjectIds.length) {
          refs.subject_scene_map[before.id] = [...oldSubjectIds];
          refs.subject_scene_map[after.id] = [...oldSubjectIds];
          delete refs.subject_scene_map[segment.id];
        }
        if (oldExtraEntries.length) {
          refs.extra_scene_map[before.id] = oldExtraEntries.map((entry) => ({ ...entry }));
          refs.extra_scene_map[after.id] = oldExtraEntries.map((entry) => ({ ...entry }));
          delete refs.extra_scene_map[segment.id];
        }
        state.fluxReferenceBuilder = refs;
        sortSegments(state.segments);
        state.segments.forEach((item, itemIndex) => {
          item.label = `Scene ${itemIndex + 1}`;
        });
        state.activeId = after.id;
        state.activeTrack = "base";
        state.duration = timelineDuration();
        syncInspector();
        render();
        await saveSession({ quiet: true, throwOnError: true });
        toast("Split scene at review playhead.");
        closeModal();
        openLyricReviewModal(isSingleScene ? { singleSceneId: after.id } : {});
      } catch (error) {
        toast(String(error?.message || error), true);
      }
    };

    const choices = referenceBuilderSubjectChoices();
    const reviewLocations = Array.isArray(reviewReferenceBuilder.locations) ? reviewReferenceBuilder.locations : [];
    const subjectIdsForReviewChoice = (choice) => {
      const subjects = Array.isArray(reviewReferenceBuilder.subjects) ? reviewReferenceBuilder.subjects : [];
      if (!choice || isNoLipSyncSingerChoice(choice.label)) return [];
      if (choice.id === "group") return subjects.filter(isIdentityCard).map((subject) => subject.id).filter(Boolean);
      const byId = subjects.find((subject) => subject.id === choice.id);
      if (byId?.id) return [byId.id];
      const cleanLabel = String(choice.label || "").trim().toLowerCase();
      const byName = subjects.find((subject) => String(subject.name || "").trim().toLowerCase() === cleanLabel);
      return byName?.id ? [byName.id] : [];
    };
    const reviewLocationForSegment = (segment, index) => {
      return reviewReferenceBuilder.scene_map?.[segment.id]
        || reviewReferenceBuilder.scene_map?.[String(index + 1)]
        || "";
    };
    const reviewSubjectIdsForSegment = (segment, index) => {
      const direct = reviewReferenceBuilder.subject_scene_map?.[segment.id];
      if (Array.isArray(direct)) return direct.map(String).filter(Boolean);
      const byNumber = reviewReferenceBuilder.subject_scene_map?.[String(index + 1)];
      if (Array.isArray(byNumber)) return byNumber.map(String).filter(Boolean);
      const subjects = Array.isArray(reviewReferenceBuilder.subjects) ? reviewReferenceBuilder.subjects : [];
      const singerNames = Array.isArray(segment.lyric_singers) ? segment.lyric_singers.map((name) => String(name || "").trim().toLowerCase()) : [];
      if (!singerNames.length) return [];
      return subjects
        .filter((subject) => singerNames.includes(String(subject.name || "").trim().toLowerCase()) || singerNames.includes(String(subject.id || "").trim().toLowerCase()))
        .map((subject) => subject.id)
        .filter(Boolean);
    };
    // Identities (people) and props (clothing, objects, vehicles and the like) are listed apart. Both are scene
    // subjects, so both use the same checkbox marker and are saved into the same scene map.
    const reviewAllSubjects = () => (Array.isArray(reviewReferenceBuilder.subjects) ? reviewReferenceBuilder.subjects : []);
    const reviewHasProps = () => reviewAllSubjects().some((subject) => !isIdentityCard(subject));
    const renderSubjectPresenceChoices = (segment, index, container, group = "identities") => {
      container.textContent = "";
      const subjects = reviewAllSubjects().filter((subject) => (group === "props" ? !isIdentityCard(subject) : isIdentityCard(subject)));
      const selected = new Set(reviewSubjectIdsForSegment(segment, index));
      if (!subjects.length) {
        const empty = document.createElement("div");
        empty.textContent = group === "props" ? "No props, clothing or vehicles yet." : "No Reference Builder subjects yet.";
        empty.style.cssText = "font-size:11px;color:#94a3b8;";
        container.append(empty);
        return;
      }
      for (const subject of subjects) {
        const subjectId = String(subject.id || "").trim();
        if (!subjectId) continue;
        const label = document.createElement("label");
        label.style.cssText = "display:inline-flex;align-items:center;gap:5px;border:1px solid #334155;border-radius:999px;background:#0f172a;color:#e2e8f0;padding:5px 8px;font-size:11px;line-height:1;cursor:pointer;user-select:none;";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.dataset.reviewPresentSubject = "1";
        input.dataset.reviewPresentGroup = group;
        input.dataset.reviewSubjectId = subjectId;
        input.value = subjectId;
        input.checked = selected.has(subjectId);
        input.style.cssText = "margin:0;";
        const name = document.createElement("span");
        name.textContent = subject.name || (group === "props" ? "Prop" : "Subject");
        label.append(input, name);
        container.append(label);
      }
    };
    const renderSingerChoices = (segment, container, instrumentalInput, brollInput) => {
      container.textContent = "";
      const selected = new Set(Array.isArray(segment.lyric_singers) ? segment.lyric_singers : []);
      for (const choice of choices) {
        if (isNoLipSyncSingerChoice(choice.label)) continue;
        const choiceLabel = reviewChoiceLabel(choice);
        const label = document.createElement("label");
        label.style.cssText = "display:inline-flex;align-items:center;gap:5px;border:1px solid #334155;border-radius:999px;background:#0f172a;color:#e2e8f0;padding:5px 8px;font-size:11px;line-height:1;cursor:pointer;user-select:none;";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.dataset.reviewSingerChoice = "1";
        input.dataset.reviewSubjectId = choice.id;
        input.value = choiceLabel;
        input.checked = selected.has(choiceLabel) || selected.has(choice.label) || selected.has(choice.id);
        input.style.cssText = "margin:0;";
        input.onchange = () => {
          if (!input.checked) return;
          const row = input.closest("[data-review-segment-id]");
          if (!row) return;
          const subjectIds = subjectIdsForReviewChoice(choice);
          for (const presentInput of row.querySelectorAll("[data-review-present-subject='1']")) {
            if (subjectIds.includes(presentInput.value)) presentInput.checked = true;
          }
        };
        const name = document.createElement("span");
        name.dataset.reviewSubjectLabel = choice.id;
        name.textContent = choiceLabel;
        label.append(input, name);
        const global = makeButton("All");
        global.title = `Use ${choiceLabel} as a visible scene character everywhere. Instrumental and B-roll scenes still stay no-lip-sync.`;
        global.style.cssText = "padding:4px 6px;min-width:0;font-size:10px;line-height:1;border-radius:999px;";
        global.onclick = async (event) => {
          event.preventDefault();
          event.stopPropagation();
          global.disabled = true;
          try {
            syncSingleSubjectPerformerLabel();
            const subjectIds = subjectIdsForReviewChoice(choice);
            if (subjectIds.length) {
              for (const scene of scenes) {
                const existing = Array.isArray(reviewReferenceBuilder.subject_scene_map?.[scene.id]) ? reviewReferenceBuilder.subject_scene_map[scene.id] : [];
                reviewReferenceBuilder.subject_scene_map[scene.id] = Array.from(new Set([...existing, ...subjectIds]));
              }
              state.fluxReferenceBuilder = reviewReferenceBuilder;
            }
            for (const row of reviewRows()) {
              if (row.querySelector("[data-review-no-character='1']")?.checked) continue;
              for (const presentInput of row.querySelectorAll("[data-review-present-subject='1']")) {
                if (subjectIds.includes(presentInput.value)) presentInput.checked = true;
              }
              for (const singerInput of row.querySelectorAll("[data-review-singer-choice='1']")) {
                if (singerInput.dataset.reviewSubjectId === choice.id || singerInput.value === choiceLabel) singerInput.checked = true;
              }
            }
            for (const row of reviewRows()) {
              const segment = scenes.find((item) => item.id === row.dataset.reviewSegmentId);
              if (segment) applyReviewRowValues(row, segment, false);
            }
            applyLyricSectionsFromReferenceText(state.segments, state.lyricMapper?.source_text || "");
            syncLyricMapperFromSegments();
            const syncedIngredients = syncIngredientsSceneMapFromSubjectMappings(state.fluxReferenceBuilder);
            state.fluxReferenceBuilder = syncedIngredients.refs;
            if (currentVideoMode() === "ingredients") applyIngredientsReferenceMappings(state.fluxReferenceBuilder);
            syncInspector();
            render();
            await saveSession({ quiet: true, throwOnError: true });
            showInfoModal({
              title: "Line Review Saved",
              lines: [subjectIds.length
                ? `${reviewChoiceLabel(choice)} was assigned as a visible scene character everywhere. Instrumental and B-roll scenes remain no-lip-sync. No-character scenes stay empty.`
                : `${reviewChoiceLabel(choice)} was added to every scene.`],
              confirmLabel: "OK",
            });
          } catch (error) {
            toast(String(error?.message || error), true);
          } finally {
            global.disabled = false;
          }
        };
        const choiceWrap = document.createElement("div");
        choiceWrap.style.cssText = "display:inline-flex;align-items:center;gap:4px;";
        choiceWrap.append(label, global);
        container.append(choiceWrap);
      }
    };

    for (const [index, segment] of scenes.entries()) {
      const sceneDisplayIndex = isSingleScene ? targetSceneIndex : index;
      const sceneNumber = sceneDisplayIndex + 1;
      const row = document.createElement("div");
      row.dataset.reviewSegmentId = segment.id;
      row.style.cssText = "display:grid;grid-template-columns:96px minmax(140px,160px) minmax(240px,1fr) minmax(280px,1.15fr) var(--vrg-identity-col,110px) minmax(210px,260px) minmax(190px,230px) minmax(150px,170px) 124px;gap:8px;align-items:start;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:8px;box-sizing:border-box;width:100%;min-width:0;";
      const meta = document.createElement("div");
      meta.style.minWidth = "0";
      meta.innerHTML = `<div data-review-scene-label style="font-weight:900;color:#cffafe;">${escapeHtml(segment.label || `Scene ${sceneNumber}`)}</div><div data-review-time-display style="font-size:11px;color:#cbd5e1;margin-top:4px;">${formatTime(segment.start)} - ${formatTime(segment.end)} | ${formatDurationSeconds(segment.start, segment.end)}s</div>`;
      const timing = document.createElement("div");
      timing.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:0;";
      const startInput = makeInput(formatTime(segment.start));
      const endInput = makeInput(formatTime(segment.end));
      startInput.dataset.reviewStart = "1";
      endInput.dataset.reviewEnd = "1";
      startInput.style.fontSize = "11px";
      endInput.style.fontSize = "11px";
      const splitButton = makeButton("Split At Playhead");
      const mergeNextButton = makeButton("Merge With Next");
      const setStartButton = makeButton("Set Start");
      const setEndButton = makeButton("Set End");
      setStartButton.title = "Set this scene start to the lyric review audio playhead. The previous scene end moves with it; later scenes stay locked.";
      setEndButton.title = "Set this scene end to the lyric review audio playhead. Lock mode only moves the next scene start; Ripple mode shifts later scenes.";
      mergeNextButton.title = "Merge this scene with the next lyric review scene. This removes the next row and combines timing, lyrics, selected characters, and location.";
      setStartButton.style.padding = "7px 8px";
      setEndButton.style.padding = "7px 8px";
      splitButton.style.padding = "7px 8px";
      mergeNextButton.style.padding = "7px 8px";
      if (isSingleScene) mergeNextButton.style.display = "none";
      setStartButton.onclick = () => setReviewRowStartToPlayhead(row);
      setEndButton.onclick = () => setReviewRowEndToPlayhead(row);
      splitButton.onclick = () => splitReviewRowAtPlayhead(row, segment);
      mergeNextButton.onclick = () => mergeReviewRowWithNext(row);
      const commitStartInput = () => {
        const start = parseBulkTimeValue(startInput.value);
        const previousStart = Number(row.dataset.reviewLastStart);
        if (!Number.isFinite(start) || !Number.isFinite(previousStart) || Math.abs(start - previousStart) > 0.0001) {
          handleReviewStartEdited(row);
        }
      };
      const commitEndInput = () => {
        const end = parseBulkTimeValue(endInput.value);
        const previousEnd = Number(row.dataset.reviewLastEnd);
        if (!Number.isFinite(end) || !Number.isFinite(previousEnd) || Math.abs(end - previousEnd) > 0.0001) {
          handleReviewEndEdited(row, previousEnd);
        }
      };
      startInput.addEventListener("input", () => syncReviewSharedBoundaryPreview(row, "start"));
      endInput.addEventListener("input", () => syncReviewSharedBoundaryPreview(row, "end"));
      startInput.addEventListener("change", commitStartInput);
      endInput.addEventListener("change", commitEndInput);
      startInput.addEventListener("blur", commitStartInput);
      endInput.addEventListener("blur", commitEndInput);
      startInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          startInput.blur();
        }
      });
      endInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          endInput.blur();
        }
      });
      timing.append(makeField("Start", startInput), makeField("End", endInput), setStartButton, setEndButton, splitButton, mergeNextButton);
      const text = document.createElement("textarea");
      text.dataset.reviewLyricText = "1";
      text.value = String(segment.lyric_text || "");
      text.dataset.reviewRawLyricText = text.value;
      text.placeholder = "Line / lyric / dialogue, or [instrumental]...";
      text.style.cssText = "width:100%;box-sizing:border-box;min-height:64px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;line-height:1.35;min-width:0;";
      text.addEventListener("input", () => {
        if (applyingBoundaryOverlapPreview) return;
        text.dataset.reviewRawLyricText = text.value;
        if (boundaryOverlapCheckbox.input.checked) refreshBoundaryOverlapPreview();
      });
      const subjectSingerPanel = document.createElement("div");
      subjectSingerPanel.style.cssText = "display:flex;flex-direction:column;gap:7px;min-width:0;";
      const presentPanel = document.createElement("div");
      presentPanel.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:7px;min-height:42px;box-sizing:border-box;";
      const propsPanel = document.createElement("div");
      propsPanel.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:7px;min-height:42px;box-sizing:border-box;";
      const singerPanel = document.createElement("div");
      singerPanel.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:7px;min-height:42px;box-sizing:border-box;";
      const flags = document.createElement("div");
      flags.style.cssText = "display:flex;flex-direction:column;gap:8px;padding-top:4px;min-width:0;";
      const instrumental = makeCheckbox("Instrumental", isInstrumentalLyricText(segment.lyric_text));
      const broll = makeCheckbox("B-roll / no lip-sync", Boolean(segment.lyric_no_lip_sync) && !isInstrumentalLyricText(segment.lyric_text));
      const noCharacter = makeCheckbox("No character present", Boolean(segment.no_character_present));
      instrumental.input.dataset.reviewInstrumental = "1";
      broll.input.dataset.reviewBroll = "1";
      noCharacter.input.dataset.reviewNoCharacter = "1";
      flags.append(instrumental.wrapper, broll.wrapper, noCharacter.wrapper);
      const updateNoCharacterState = () => {
        const disabled = Boolean(noCharacter.input.checked);
        for (const input of [...presentPanel.querySelectorAll("[data-review-present-subject='1']"), ...propsPanel.querySelectorAll("[data-review-present-subject='1']")]) {
          input.disabled = disabled;
          if (disabled) input.checked = false;
        }
        const noLipSync = Boolean(instrumental.input.checked || broll.input.checked);
        for (const input of singerPanel.querySelectorAll("[data-review-singer-choice='1']")) {
          input.disabled = disabled || noLipSync;
          if (disabled || noLipSync) input.checked = false;
        }
        presentPanel.style.opacity = disabled ? "0.55" : "1";
        propsPanel.style.opacity = disabled ? "0.55" : "1";
        singerPanel.style.opacity = (disabled || noLipSync) ? "0.55" : "1";
      };
      const updateDisabled = () => {
        if (instrumental.input.checked) {
          if (!isInstrumentalLyricText(text.value)) text.dataset.reviewPreInstrumentalText = text.value;
          text.value = "[instrumental]";
          text.dataset.reviewRawLyricText = "[instrumental]";
          broll.input.checked = false;
        } else if (isInstrumentalLyricText(text.value)) {
          const restored = String(text.dataset.reviewPreInstrumentalText || "");
          text.value = restored;
          text.dataset.reviewRawLyricText = restored;
        }
        if (broll.input.checked && isInstrumentalLyricText(text.value)) {
          text.value = text.dataset.reviewRawLyricText && !isInstrumentalLyricText(text.dataset.reviewRawLyricText)
            ? text.dataset.reviewRawLyricText
            : "";
        }
        updateNoCharacterState();
        refreshBoundaryOverlapPreview();
      };
      renderSubjectPresenceChoices(segment, sceneDisplayIndex, presentPanel, "identities");
      renderSubjectPresenceChoices(segment, sceneDisplayIndex, propsPanel, "props");
      renderSingerChoices(segment, singerPanel, instrumental.input, broll.input);
      subjectSingerPanel.append(makeField("Identities in scene", presentPanel));
      if (reviewHasProps()) subjectSingerPanel.append(makeField("Props, clothing and vehicles in scene", propsPanel));
      subjectSingerPanel.append(makeField("Performer / speaker / lip-sync (identities only)", singerPanel));
      const facialPanel = document.createElement("div");
      facialPanel.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:0;";
      const facialSelect = makeSelect(FACIAL_PERFORMANCE_PRESETS, segment.facial_performance || "");
      facialSelect.dataset.reviewFacialPerformance = "1";
      const facialCustom = document.createElement("textarea");
      facialCustom.dataset.reviewFacialPerformanceCustom = "1";
      facialCustom.value = String(segment.facial_performance_custom || "");
      facialCustom.placeholder = "Custom facial text...";
      facialCustom.style.cssText = "width:100%;box-sizing:border-box;min-height:42px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;line-height:1.35;min-width:0;";
      facialPanel.append(facialSelect, facialCustom);
      instrumental.input.onchange = updateDisabled;
      broll.input.onchange = updateDisabled;
      noCharacter.input.onchange = updateDisabled;
      updateDisabled();
      const locationWrap = document.createElement("div");
      locationWrap.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:0;";
      const locationSelect = document.createElement("select");
      locationSelect.dataset.reviewLocation = "1";
      locationSelect.style.cssText = "width:100%;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#f8fafc;padding:8px;font-size:12px;";
      locationSelect.append(new Option("Unassigned", ""));
      for (const location of reviewLocations) {
        locationSelect.append(new Option(location.name || "Location", location.id));
      }
      locationSelect.value = reviewLocationForSegment(segment, sceneDisplayIndex);
      if (!reviewLocations.length) {
        locationSelect.disabled = true;
        locationSelect.title = "Add locations in Reference Builder first.";
      }
      const locationHint = document.createElement("div");
      locationHint.textContent = reviewLocations.length ? "Optional Reference Builder location for this scene." : "No Reference Builder locations yet.";
      locationHint.style.cssText = "font-size:10px;color:#94a3b8;line-height:1.25;";
      locationWrap.append(makeField("Location", locationSelect), locationHint);
      const buttons = document.createElement("div");
      buttons.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:124px;";
      const play = makeButton("Play Scene");
      const playFrom = makeButton("Play From Here");
      const select = makeButton("Select");
      const moveLastWord = makeButton("Move last word to start of next scene");
      moveLastWord.dataset.reviewMoveLastWord = "1";
      moveLastWord.title = "Move only this scene's final lyric word to the beginning of the next scene. Use Save Lines afterward to synchronize the timeline, cue maps, and lyric note files.";
      moveLastWord.disabled = index >= scenes.length - 1;
      if (isSingleScene) moveLastWord.title = "Open All Scenes to move a word between neighboring scene cards.";
      play.style.padding = "7px 8px";
      playFrom.style.padding = "7px 8px";
      select.style.padding = "7px 8px";
      play.style.width = "100%";
      playFrom.style.width = "100%";
      select.style.width = "100%";
      moveLastWord.style.cssText = "padding:7px 8px;width:100%;font-size:10px;line-height:1.25;white-space:normal;";
      play.dataset.playLabel = "Play Scene";
      playFrom.dataset.playLabel = "Play From Here";
      play.onclick = () => playRange(segment, play);
      playFrom.onclick = () => playFromSegment(segment, playFrom);
      select.onclick = () => setActiveReviewScene(segment);
      moveLastWord.onclick = () => moveLastWordToNextReviewRow(row);
      // Replace this scene's scene beat and/or LLM prompt, the same as selecting only this scene in the Story Builder.
      const replaceBeat = makeButton("Replace Scene Beat");
      const replaceLlm = makeButton("Replace LLM Prompt");
      for (const [button, beat, prompt, hint] of [
        [replaceBeat, true, false, "Replace the scene beat of this scene only with the selected LLM."],
        [replaceLlm, false, true, "Replace the video prompt of this scene only with the selected LLM."],
      ]) {
        button.dataset.reviewStoryAction = "1";
        button.dataset.reviewStoryKind = beat ? "beat" : "llm";
        button.style.cssText = "padding:7px 8px;width:100%;font-size:11px;line-height:1.25;white-space:normal;";
        button.dataset.reviewBaseStyle = button.style.cssText;
        button.title = `${hint} Your edits in this window are saved first.`;
        button.onclick = () => runSceneStoryAction(row, segment, { beat, prompt });
      }
      buttons.append(play, playFrom, select, moveLastWord, replaceBeat, replaceLlm);
      const identityStrip = document.createElement("div");
      identityStrip.dataset.reviewIdentityStrip = "1";
      identityStrip.style.cssText = "display:flex;flex-direction:column;gap:8px;min-width:0;";
      row.append(meta, timing, text, subjectSingerPanel, identityStrip, facialPanel, locationWrap, flags, buttons);
      rowList.append(row);
      rememberReviewRowTiming(row);
    }

    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Close");
    const save = makeButton(isSingleScene ? "Save Scene" : "Save Lines + Timing + Performers + Locations", "primary");
    actions.append(cancel, save);
    // Pictures of what is checked in a scene (identities in one row, props, clothing and vehicles in another), between the
    // identity boxes and the facial performance choice. The column is as wide as the scene with the most pictures in a row,
    // and the window grows with it.
    const IDENTITY_THUMB = 64;
    const IDENTITY_GAP = 6;
    const IDENTITY_MIN_COLUMN = 110;
    const identityThumbUrl = (subject) => {
      const refmodName = String(subject?.refmod?.name || "").trim();
      if (subject?.source === "refmod" && refmodName) return refmodPreviewUrl(refmodName);
      const image = subject?.image || {};
      return String(image.data || "").trim() || (image.path ? makeEditorImageUrl(image.path) : "");
    };
    // One picture card for a subject: its preview or image, with its name underneath.
    const identityPictureCard = (subject, fallbackName) => {
      const name = String(subject.name || fallbackName);
      const card = document.createElement("div");
      card.title = name;
      card.style.cssText = `display:flex;flex-direction:column;gap:3px;align-items:center;width:${IDENTITY_THUMB}px;flex:0 0 ${IDENTITY_THUMB}px;`;
      const url = identityThumbUrl(subject);
      const frame = document.createElement("div");
      frame.style.cssText = `width:${IDENTITY_THUMB}px;height:${IDENTITY_THUMB}px;border:1px solid #334155;border-radius:6px;background:#0b1220;overflow:hidden;display:flex;align-items:center;justify-content:center;color:#64748b;font-size:10px;`;
      if (url) {
        const image = document.createElement("img");
        image.src = url;
        image.alt = name;
        image.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
        image.onerror = () => { image.remove(); frame.textContent = "No image"; };
        frame.append(image);
      } else {
        frame.textContent = "No image";
      }
      const caption = document.createElement("div");
      caption.textContent = name;
      caption.style.cssText = `width:${IDENTITY_THUMB}px;font-size:10px;color:#cbd5e1;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`;
      card.append(frame, caption);
      return card;
    };
    const refreshIdentityStrips = () => {
      const subjects = reviewAllSubjects();
      let most = 0;
      for (const row of reviewRows()) {
        const strip = row.querySelector("[data-review-identity-strip='1']");
        if (!strip) continue;
        strip.textContent = "";
        // Identities first, then the props, clothing and vehicles checked for the scene, each in its own row.
        for (const [group, fallbackName] of [["identities", "Identity"], ["props", "Prop"]]) {
          const ids = [...row.querySelectorAll(`[data-review-present-group='${group}']`)]
            .filter((input) => input.checked && !input.disabled)
            .map((input) => String(input.value || ""));
          most = Math.max(most, ids.length);
          if (!ids.length) continue;
          const line = document.createElement("div");
          line.dataset.reviewPictureGroup = group;
          line.style.cssText = `display:flex;flex-wrap:nowrap;gap:${IDENTITY_GAP}px;align-items:flex-start;`;
          for (const id of ids) {
            const subject = subjects.find((item) => String(item?.id || "") === id);
            if (subject) line.append(identityPictureCard(subject, fallbackName));
          }
          strip.append(line);
        }
      }
      const column = Math.max(IDENTITY_MIN_COLUMN, most * (IDENTITY_THUMB + IDENTITY_GAP) - IDENTITY_GAP);
      rowList.style.setProperty("--vrg-identity-col", `${column}px`);
      box.style.setProperty("--vrg-identity-extra", `${column - IDENTITY_MIN_COLUMN}px`);
    };
    // What a scene's beat and prompt are written from: its line, timing, who is in it and who performs, its flags, facial
    // performance and location. A change to any of them means the beat and the LLM prompt are out of date.
    const rowStorySnapshot = (row) => JSON.stringify([
      row.querySelector("[data-review-lyric-text]")?.value ?? "",
      row.querySelector("[data-review-start]")?.value ?? "",
      row.querySelector("[data-review-end]")?.value ?? "",
      [...row.querySelectorAll("[data-review-present-subject='1']")].filter((input) => input.checked).map((input) => input.value).sort(),
      [...row.querySelectorAll("[data-review-singer-choice='1']")].filter((input) => input.checked).map((input) => input.value).sort(),
      ["[data-review-instrumental]", "[data-review-broll]", "[data-review-no-character]"].map((selector) => Boolean(row.querySelector(selector)?.checked)),
      row.querySelector("[data-review-facial-performance='1']")?.value ?? "",
      row.querySelector("[data-review-facial-performance-custom='1']")?.value ?? "",
      row.querySelector("[data-review-location]")?.value ?? "",
    ]);
    const STORY_KINDS = ["beat", "llm"];
    const storyBaselineFor = (id) => {
      let entry = storyBaselines.get(id);
      if (!entry) {
        entry = {};
        storyBaselines.set(id, entry);
      }
      return entry;
    };
    const storyKindIsRed = (row, kind) => {
      const entry = storyBaselines.get(String(row.dataset.reviewSegmentId || ""));
      return Boolean(entry && entry[kind] !== undefined && entry[kind] !== rowStorySnapshot(row));
    };
    const refreshStoryActionButtons = () => {
      for (const row of reviewRows()) {
        for (const button of row.querySelectorAll("[data-review-story-action='1']")) {
          button.style.cssText = storyKindIsRed(row, button.dataset.reviewStoryKind)
            ? `${button.dataset.reviewBaseStyle}background:#b91c1c;border-color:#ef4444;color:#fff;font-weight:800;`
            : button.dataset.reviewBaseStyle;
        }
      }
    };
    // Scenes that are red and so need their scene beat / LLM prompt replaced.
    const scenesNeedingReplace = () => reviewRows()
      .filter((row) => STORY_KINDS.some((kind) => storyKindIsRed(row, kind)))
      .map((row) => {
        const id = String(row.dataset.reviewSegmentId || "");
        return (Array.isArray(state.segments) ? state.segments : []).find((item) => item?.id === id)?.label || "A scene";
      });
    // Forgets baselines that match the scene again, so the next open starts from what the scene is then.
    const releaseStoryBaselines = () => {
      for (const row of reviewRows()) {
        const id = String(row.dataset.reviewSegmentId || "");
        const entry = storyBaselines.get(id);
        if (!entry) continue;
        const now = rowStorySnapshot(row);
        for (const kind of STORY_KINDS) if (entry[kind] === now) delete entry[kind];
        if (!STORY_KINDS.some((kind) => entry[kind] !== undefined)) storyBaselines.delete(id);
      }
    };
    rowList.addEventListener("change", () => { refreshIdentityStrips(); refreshStoryActionButtons(); });
    rowList.addEventListener("input", refreshStoryActionButtons);
    box.append(header, note, performerLabelPanel, timingModePanel, audioPanel, rowList, actions);
    refreshIdentityStrips();
    for (const row of reviewRows()) {
      const id = String(row.dataset.reviewSegmentId || "");
      const entry = storyBaselineFor(id);
      const now = rowStorySnapshot(row);
      for (const kind of STORY_KINDS) if (entry[kind] === undefined) entry[kind] = now;
    }
    refreshStoryActionButtons();
    backdrop.append(box);
    document.body.append(backdrop);
    if (focusSceneId) {
      requestAnimationFrame(() => {
        const focusRow = rowList.querySelector(`[data-review-segment-id="${CSS.escape(focusSceneId)}"]`);
        if (!focusRow) return;
        focusRow.style.borderColor = "#22d3ee";
        focusRow.style.boxShadow = "0 0 0 2px rgba(34,211,238,.28)";
        focusRow.scrollIntoView({ behavior: "smooth", block: "center" });
        focusRow.querySelector("[data-review-lyric-text]")?.focus({ preventScroll: true });
      });
    }
    activeLyricReviewBackdrop = backdrop;
    const closeModal = () => {
      releaseStoryBaselines();
      clearReviewStopGuards();
      reviewAudio.pause();
      backdrop.remove();
      if (activeLyricReviewBackdrop === backdrop) activeLyricReviewBackdrop = null;
    };
    close.onclick = closeModal;
    cancel.onclick = closeModal;
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) closeModal();
    });
    // Edited scenes whose scene beat or LLM prompt were not replaced: the render keeps using the old ones. OK keeps the
    // edits and goes on, Cancel leaves the window as it is.
    const confirmUnreplacedStory = async () => {
      refreshStoryActionButtons();
      const names = scenesNeedingReplace();
      if (!names.length) return true;
      return showConfirmModal({
        title: "Unreplaced Changes",
        lines: [
          "Changes will not be applied to renders unless Replace Scene Beat and Replace LLM Prompt are run for the changed scenes.",
          `Changed: ${names.join(", ")}`,
          "OK saves your changes and closes. Cancel keeps this window open.",
        ],
      });
    };
    const navigateReviewScene = async (nextOptions) => {
      if (!(await confirmUnreplacedStory())) return;
      if (!(await saveReviewChanges(true)) || !backdrop.isConnected) return;
      closeModal();
      openLyricReviewModal(nextOptions);
    };
    const saveReviewChanges = async (quiet = false) => {
      if (save.disabled) return false;
      try {
        save.disabled = true;
        pushHistory();
        ensureAllSegmentRuntimeFields();
        syncSingleSubjectPerformerLabel();
        await normalizeEditedReviewTiming();
        const lyricOverrides = collectBoundaryOverlapLyricOverrides();
        const liveSegmentsById = new Map((Array.isArray(state.segments) ? state.segments : [])
          .filter((segment) => segment && typeof segment === "object" && !Array.isArray(segment) && segment.id)
          .map((segment) => [String(segment.id), segment]));
        for (const row of rowList.querySelectorAll("[data-review-segment-id]")) {
          const segmentId = String(row.dataset.reviewSegmentId || "");
          const segment = liveSegmentsById.get(segmentId);
          if (!segment) continue;
          applyReviewRowValues(row, segment, true, { lyricTextOverride: lyricOverrides.get(segment.id) });
        }
        applyPendingReviewWordMoves(liveSegmentsById);
        applyLyricSectionsFromReferenceText(state.segments, state.lyricMapper?.source_text || "");
        syncLyricMapperFromSegments();
        const syncedIngredients = syncIngredientsSceneMapFromSubjectMappings(state.fluxReferenceBuilder);
        state.fluxReferenceBuilder = syncedIngredients.refs;
        if (currentVideoMode() === "ingredients") applyIngredientsReferenceMappings(state.fluxReferenceBuilder);
        sortSegments(state.segments);
        ensureAllSegmentRuntimeFields();
        state.segments.forEach((segment, index) => {
          segment.label = `Scene ${index + 1}`;
        });
        state.duration = timelineDuration();
        syncInspector();
        render();
        await saveSession({ quiet: true, throwOnError: true });
        pendingReviewWordMoves.length = 0;
        if (!quiet) showInfoModal({
          title: isSingleScene ? `${targetScene.label || "Scene"} Saved` : "Line Review Saved",
          lines: isSingleScene
            ? ["Scene lines, timing, performer labels, lip-sync flags, and location mapping were saved to the timeline."]
            : ["Lines, scene timing, performer labels, no-lip-sync/no-character flags, and location mapping were saved to the timeline."],
          confirmLabel: "OK",
        });
        return true;
      } catch (error) {
        const message = String(error?.message || error);
        if (/image_history/i.test(message)) {
          try {
            ensureAllSegmentRuntimeFields();
            state.duration = timelineDuration();
            syncInspector();
            render();
            await saveSession({ quiet: true, throwOnError: true });
            pendingReviewWordMoves.length = 0;
            if (!quiet) showInfoModal({
              title: "Line Review Saved",
              lines: ["Lines, timing, performers, no-lip-sync/no-character flags, and locations were saved. A stale media history value was cleaned up automatically."],
              confirmLabel: "OK",
            });
            return true;
          } catch (retryError) {
            showInfoModal({
              title: "Line Review Save Error",
              lines: [String(retryError?.message || retryError)],
              confirmLabel: "OK",
            });
            return false;
          }
        }
        showInfoModal({
          title: "Line Review Save Error",
          lines: [message],
          confirmLabel: "OK",
        });
        return false;
      } finally {
        save.disabled = false;
      }
    };
    save.onclick = async () => {
      if (scenesNeedingReplace().length) {
        if (!(await confirmUnreplacedStory())) return;
        if (await saveReviewChanges(true) && backdrop.isConnected) closeModal();
        return;
      }
      return saveReviewChanges();
    };
    // Saves this window, then runs the Story Builder's Replace Scene Beat and/or LLM for this one scene.
    async function runSceneStoryAction(row, segment, { beat, prompt }) {
      if (typeof openStoryboardBuilderFromProject !== "function") {
        toast("The Story Builder is not available here.", true);
        return;
      }
      const sceneName = segment.label || "this scene";
      const what = beat && prompt ? "scene beat and LLM prompt" : beat ? "scene beat" : "LLM prompt";
      const go = await showConfirmModal({
        title: `Replace for ${sceneName}`,
        lines: [`Replace the ${what} for ${sceneName} only?`, "Your edits in this window are saved first. No other scene is changed."],
      });
      if (!go) return;
      if (!(await saveReviewChanges(true)) || !backdrop.isConnected) return;
      const live = (Array.isArray(state.segments) ? state.segments : []).find((item) => item?.id === segment.id);
      if (!live) {
        toast("Could not find this scene after saving.", true);
        return;
      }
      const actionButtons = [...row.querySelectorAll("[data-review-story-action='1']")];
      actionButtons.forEach((button) => { button.disabled = true; });
      const replacedSnapshot = rowStorySnapshot(row);
      let replaced = false;
      try {
        await new Promise((resolve) => {
          openStoryboardBuilderFromProject({
            sceneActions: { sceneId: live.id, beat, prompt },
            onSceneActionsDone: (ok) => { replaced = Boolean(ok); },
            onClose: resolve,
          });
        });
      } finally {
        actionButtons.forEach((button) => { button.disabled = false; });
      }
      if (replaced) {
        const entry = storyBaselineFor(String(live.id));
        if (beat) entry.beat = replacedSnapshot;
        if (prompt) entry.llm = replacedSnapshot;
        refreshStoryActionButtons();
      }
    }
  }

  return { openLyricReviewModal };
}
