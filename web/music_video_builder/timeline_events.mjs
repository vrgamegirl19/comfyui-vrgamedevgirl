import { toast } from "./controls.mjs";
import { localPlaybackTime, selectedSegmentVideoPath } from "./selection_preview.mjs";
import { timelineAudioDurationForSegment, timelineAudioStartForSegment } from "./timeline_state.mjs";
import { showMultiSelectHint } from "./video_render.mjs";

export function wireTimelineControls({
  activeSegment, applyAutoBpmCalibration, applyCapCutBeatImport, applyThreePointBeatCalibration, audio,
  autoSaveSessionQuiet, beatCalibration, beatCalibrationCancelButton, beatCalibrationCaptureButton,
  beatCalibrationGridType, beatCalibrationTimecodeInput, beatMarkersButton, beginGlobalTimelineScrub,
  calibrateFirstBeatButton, cancelPreviewPlayStart, captureBeatCalibrationAnchor,
  clearActiveSegment, closeBeatCalibrationWizard, currentGlobalTime,
  deleteAllSegments, deleteAllSegmentsButton, deleteAllTimelineImages, deleteAllTimelineImagesButton,
  deleteAllTimelineVideos, deleteAllTimelineVideosButton, deleteSegment, deleteSegmentButton,
  enforceAudioTimelineEnd, ensureAutoBpmForCalibration, freezeTimingControl,
  ensureCapCutBeatsForCalibration, ensureGlobalTimelineAudioSource, globalAudioMuteButton, globalScrub,
  isTimelinePlaying, lyricNoteButton, multiSelectButton, multiSelectHintButton, openBeatCalibrationWizard,
  openMultiSelectChooser, pauseAllAudio, playbackDuration, playbackSegmentAtTime, playButton, playhead,
  playSceneAudioFrom, playStart, previewEmpty, previewStage, previewVideo, pushHistory, reloadBeatMarkersFromAudio, render,
  renderBeatCalibrationWizard, sceneAudio, sceneListPane, sceneNoteButton, seekAudioWhenReady,
  setBeatMarkersVisible, setGlobalPlaybackTime, setGlobalTimelineAudioMuted, setTimelineZoom,
  snapAllSceneStartsButton, snapAllSceneStartsToNearestBeats, snapToBeatsControl, startSilentTimelinePlayback,
  state, stopButton, stopSilentTimelinePlayback, syncInspector, syncLyricNoteControls, syncPreviewPlayback,
  syncSceneNoteControls, syncTimelineTrimModeButton, syncVideoNoteControls, timelineAudioPathForSegment,
  timelineAudioSourceStartForSegment, timelineCanvas, timelineViewport, updateAudioScrubbers,
  updatePlayPauseButton, usingSceneAudioPlaybackMode, videoNoteButton,
  waitForPreviewVideoReady, waveformModeSelect, zoomInButton, zoomOutButton, prepareAudioEdits,
}) {
  freezeTimingControl.input.addEventListener("change", () => {
    pushHistory();
    state.timingFrozen = Boolean(freezeTimingControl.input.checked);
    syncInspector();
    render();
    toast(state.timingFrozen ? "All scene timing frozen." : "Scene timing unlocked for editing.");
    autoSaveSessionQuiet("global scene timing lock changed").catch((error) => toast(String(error), true));
  });
  deleteSegmentButton.onclick = deleteSegment;
  deleteAllSegmentsButton.onclick = deleteAllSegments;
  deleteAllTimelineVideosButton.onclick = deleteAllTimelineVideos;
  deleteAllTimelineImagesButton.onclick = deleteAllTimelineImages;
  globalAudioMuteButton.onclick = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setGlobalTimelineAudioMuted(!(audio.muted && sceneAudio.muted));
  };
  playButton.onclick = async () => {
    if (state.timelineTrimEditMode) {
      state.timelineTrimEditMode = false;
      syncTimelineTrimModeButton();
    }
    if (isTimelinePlaying()) {
      pauseAllAudio();
      updateAudioScrubbers();
      return;
    }
    if (playStart.inFlight) {
      cancelPreviewPlayStart();
      return;
    }
    const request = ++playStart.request;
    playStart.inFlight = true;
    try {
      if (prepareAudioEdits) {
        const position = currentGlobalTime();
        if (await prepareAudioEdits()) {
          if (request !== playStart.request) return;
          setGlobalPlaybackTime(position);
          render();
        }
      }
      // If the user just scrubbed here, the preview video may still be
      // loading/seeking to this position. Kick that off (in case it hasn't
      // already started) and wait briefly for it before starting audio, so
      // the audio clock doesn't get a head start on a video that then has to
      // visibly jump to catch up.
      let effectiveStart = currentGlobalTime();
      if (effectiveStart >= playbackDuration() - 0.025) {
        effectiveStart = Math.max(0, Number(activeSegment()?.start || 0));
      }
      const startSegment = playbackSegmentAtTime(effectiveStart);
      if (startSegment && selectedSegmentVideoPath(startSegment)) {
        syncPreviewPlayback(effectiveStart);
        await waitForPreviewVideoReady(localPlaybackTime(startSegment, effectiveStart));
      }
    } catch (error) {
      toast(String(error.message || error), true);
      return;
    } finally {
      if (request === playStart.request) playStart.inFlight = false;
    }
    if (request !== playStart.request || isTimelinePlaying()) return;
    if (!state.sceneSelectionUsesGlobalAudio && usingSceneAudioPlaybackMode()) {
      audio.pause();
      const started = playSceneAudioFrom(currentGlobalTime());
      if (!started) startSilentTimelinePlayback(currentGlobalTime());
      else updatePlayPauseButton();
      return;
    }
    let startTime = currentGlobalTime();
    if (startTime >= playbackDuration() - 0.025) {
      startTime = Math.max(0, Number(activeSegment()?.start || 0));
    }
    if (!ensureGlobalTimelineAudioSource(startTime)) {
      startSilentTimelinePlayback(startTime);
      return;
    }
    seekAudioWhenReady(startTime);
    audio.play().then(updatePlayPauseButton).catch(() => startSilentTimelinePlayback(startTime));
  };
  multiSelectButton.onclick = openMultiSelectChooser;
  multiSelectHintButton.onclick = showMultiSelectHint;
  stopButton.onclick = () => {
    pauseAllAudio();
    audio.currentTime = 0;
    sceneAudio.currentTime = 0;
    state.sceneAudioGlobalTime = 0;
    state.sceneAudioSegmentId = "";
    if (!previewVideo.paused) previewVideo.pause();
    updateAudioScrubbers();
  };
  timelineCanvas.addEventListener("pointerdown", beginGlobalTimelineScrub);
  playhead.addEventListener("pointerdown", beginGlobalTimelineScrub);
  timelineViewport.addEventListener("click", (event) => {
    if (event.target === timelineViewport || event.target === timelineCanvas) clearActiveSegment();
  });
  sceneListPane.addEventListener("click", (event) => {
    if (event.target === sceneListPane) clearActiveSegment();
  });
  previewStage.addEventListener("click", (event) => {
    if (event.target === previewStage || event.target === previewEmpty) clearActiveSegment();
  });
  globalScrub.addEventListener("pointerdown", () => {
    state.isScrubbing = true;
  });
  // The native range input can fire "input" faster than the preview video can
  // seek. Collapse to one processed value per animation frame, same as the
  // timeline-canvas drag, so the preview chases the latest thumb position
  // instead of a backlog of superseded ones.
  let globalScrubPendingValue = null;
  let globalScrubRaf = 0;
  globalScrub.addEventListener("input", () => {
    globalScrubPendingValue = Number(globalScrub.value || 0);
    if (!globalScrubRaf) {
      globalScrubRaf = requestAnimationFrame(() => {
        globalScrubRaf = 0;
        if (globalScrubPendingValue != null) {
          setGlobalPlaybackTime(globalScrubPendingValue);
          globalScrubPendingValue = null;
          updateAudioScrubbers();
        }
      });
    }
  });
  globalScrub.addEventListener("change", () => {
    state.isScrubbing = false;
    if (globalScrubRaf) {
      cancelAnimationFrame(globalScrubRaf);
      globalScrubRaf = 0;
    }
    if (globalScrubPendingValue != null) {
      setGlobalPlaybackTime(globalScrubPendingValue);
      globalScrubPendingValue = null;
    }
    updateAudioScrubbers();
  });
  waveformModeSelect.onchange = () => {
    state.waveformMode = waveformModeSelect.value || "medium";
    render();
  };
  sceneNoteButton.onclick = () => {
    state.showTimelineSceneNotes = !state.showTimelineSceneNotes;
    syncSceneNoteControls();
    render();
    autoSaveSessionQuiet(state.showTimelineSceneNotes ? "timeline scene notes shown" : "timeline scene notes hidden");
  };
  videoNoteButton.onclick = () => {
    state.showTimelineVideoNotes = !state.showTimelineVideoNotes;
    syncVideoNoteControls();
    render();
    autoSaveSessionQuiet(state.showTimelineVideoNotes ? "timeline video notes shown" : "timeline video notes hidden");
  };
  lyricNoteButton.onclick = () => {
    state.showTimelineLyricNotes = !state.showTimelineLyricNotes;
    syncLyricNoteControls();
    render();
    autoSaveSessionQuiet(state.showTimelineLyricNotes ? "timeline line notes shown" : "timeline line notes hidden");
  };
  zoomOutButton.onclick = () => setTimelineZoom(state.pxPerSecond / 1.25);
  zoomInButton.onclick = () => setTimelineZoom(state.pxPerSecond * 1.25);
  // Ctrl + mouse wheel over the timeline zooms the timeline only (not the page), around the time under the pointer.
  // Wheel events arrive many times a second, so they are combined into one zoom per frame and saved once after the last.
  let wheelFactor = 1;
  let wheelAnchorTime = 0;
  let wheelFrame = 0;
  let wheelSaveTimer = 0;
  timelineViewport.addEventListener("wheel", (event) => {
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    event.stopPropagation();
    const rect = timelineViewport.getBoundingClientRect();
    const zoom = Math.max(1, Number(state.pxPerSecond || 45));
    // 12 px is the timeline's left padding inside the viewport.
    wheelAnchorTime = Math.max(0, (timelineViewport.scrollLeft + event.clientX - rect.left - 12) / zoom);
    const pixels = event.deltaMode === 1 ? event.deltaY * 33 : event.deltaY;
    wheelFactor *= Math.exp(-pixels * 0.0015);
    if (wheelFrame) return;
    wheelFrame = window.requestAnimationFrame(() => {
      wheelFrame = 0;
      const factor = wheelFactor;
      wheelFactor = 1;
      setTimelineZoom(state.pxPerSecond * factor, wheelAnchorTime, { save: false });
      window.clearTimeout(wheelSaveTimer);
      wheelSaveTimer = window.setTimeout(() => autoSaveSessionQuiet("timeline zoom changed"), 500);
    });
  }, { passive: false });
  beatMarkersButton.onclick = async () => {
    if (state.showBeatMarkers && (!state.beats || !state.beats.length)) {
      const loaded = await reloadBeatMarkersFromAudio();
      if (!loaded) setBeatMarkersVisible(false);
      render();
      return;
    }
    const shouldShow = !state.showBeatMarkers;
    setBeatMarkersVisible(shouldShow);
    if (state.showBeatMarkers && (!state.beats || !state.beats.length)) {
      const loaded = await reloadBeatMarkersFromAudio();
      if (!loaded) setBeatMarkersVisible(false);
    }
    render();
  };
  calibrateFirstBeatButton.onclick = () => {
    openBeatCalibrationWizard().catch((error) => {
      toast(`Could not open beat calibration:\n${String(error?.message || error)}`, true);
    });
  };
  beatCalibrationCaptureButton.onclick = async () => {
    if (beatCalibrationGridType.value === "capcut_import") {
      const imported = await ensureCapCutBeatsForCalibration();
      if (!imported) return;
      applyCapCutBeatImport().catch((error) => {
        toast(`Could not import CapCut beat markers:\n${String(error?.message || error)}`, true);
      });
      return;
    }
    if (beatCalibrationGridType.value === "auto_bpm") {
      const bpm = await ensureAutoBpmForCalibration();
      if (!(bpm > 0)) return;
      if (beatCalibration.draft?.anchors?.length === 1) {
        applyAutoBpmCalibration().catch((error) => {
          toast(`Could not apply automatic BPM grid:\n${String(error?.message || error)}`, true);
        });
      } else {
        const captured = captureBeatCalibrationAnchor();
        if (captured) {
          applyAutoBpmCalibration().catch((error) => {
            toast(`Could not apply automatic BPM grid:\n${String(error?.message || error)}`, true);
          });
        }
      }
      return;
    }
    if (beatCalibration.draft?.anchors?.length === 3) {
      applyThreePointBeatCalibration().catch((error) => {
        toast(`Could not apply beat calibration:\n${String(error?.message || error)}`, true);
      });
      return;
    }
    captureBeatCalibrationAnchor();
  };
  beatCalibrationCancelButton.onclick = closeBeatCalibrationWizard;
  beatCalibrationGridType.onchange = () => {
    if (beatCalibration.draft) beatCalibration.draft.anchors = [];
    renderBeatCalibrationWizard();
    if (beatCalibrationGridType.value === "auto_bpm") {
      ensureAutoBpmForCalibration().catch((error) => {
        toast(`Could not analyze BPM:\n${String(error?.message || error)}`, true);
      });
    } else if (beatCalibrationGridType.value === "capcut_import") {
      ensureCapCutBeatsForCalibration().catch((error) => {
        toast(`Could not locate matching CapCut beats:\n${String(error?.message || error)}`, true);
      });
    }
  };
  beatCalibrationTimecodeInput.onkeydown = (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    beatCalibrationCaptureButton.click();
  };
  snapAllSceneStartsButton.onclick = () => {
    snapAllSceneStartsToNearestBeats().catch((error) => {
      toast(`Could not snap all scene starts:\n${String(error?.message || error)}`, true);
    });
  };
  snapToBeatsControl.input.onchange = () => {
    state.snapToBeats = Boolean(snapToBeatsControl.input.checked);
  };
  audio.addEventListener("timeupdate", updateAudioScrubbers);
  audio.addEventListener("loadedmetadata", () => {
    const mediaDuration = Number(audio.duration);
    if (Number.isFinite(mediaDuration) && mediaDuration > 0) {
      state.audioDuration = mediaDuration;
      const result = enforceAudioTimelineEnd();
      if (result.trimmed || result.removed) render();
    }
    updateAudioScrubbers();
  });
  audio.addEventListener("play", () => {
    stopSilentTimelinePlayback();
    updatePlayPauseButton();
  });
  audio.addEventListener("pause", () => {
    updatePlayPauseButton();
    updateAudioScrubbers();
  });
  audio.addEventListener("ended", () => {
    if (!previewVideo.paused) previewVideo.pause();
    updatePlayPauseButton();
    updateAudioScrubbers();
  });
  sceneAudio.addEventListener("play", () => {
    stopSilentTimelinePlayback();
    updatePlayPauseButton();
  });
  sceneAudio.addEventListener("pause", updatePlayPauseButton);
  sceneAudio.addEventListener("timeupdate", () => {
    const segment = state.segments.find((item) => item.id === state.sceneAudioSegmentId) || activeSegment();
    if (segment) {
      const sourceLocal = Math.max(0, Number(sceneAudio.currentTime || 0) - timelineAudioSourceStartForSegment(segment));
      state.sceneAudioGlobalTime = timelineAudioStartForSegment(segment) + sourceLocal;
      if (sourceLocal >= timelineAudioDurationForSegment(segment) - 0.03) {
        sceneAudio.pause();
        sceneAudio.dispatchEvent(new Event("ended"));
        return;
      }
    }
    updateAudioScrubbers();
  });
  sceneAudio.addEventListener("ended", () => {
    const current = currentGlobalTime();
    const next = state.segments.find((segment) => timelineAudioStartForSegment(segment) >= current - 0.02 && segment.id !== state.sceneAudioSegmentId && timelineAudioPathForSegment(segment)) ||
      state.segments.find((segment) => timelineAudioStartForSegment(segment) > current && timelineAudioPathForSegment(segment));
    if (next) {
      playSceneAudioFrom(timelineAudioStartForSegment(next));
    } else {
      if (!previewVideo.paused) previewVideo.pause();
      updateAudioScrubbers();
    }
  });
}
