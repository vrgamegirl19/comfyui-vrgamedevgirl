import { postJson } from "./comfy_api.mjs";
import { getWidget, setWidgetValue, toast } from "./controls.mjs";
import { formatTime } from "./format.mjs";

function parseCapCutTimecode(value, fpsValue) {
  const text = String(value || "").trim();
  const fps = Number(fpsValue);
  if (!Number.isFinite(fps) || fps <= 0 || fps > 120) {
    throw new Error("Enter a valid FPS from 1 through 120.");
  }
  const match = text.match(/^(\d+):([0-5]?\d):([0-5]?\d)(?::|\+)(\d+)$/);
  if (!match) {
    throw new Error("Use CapCut timecode HH:MM:SS:FF or HH:MM:SS+FF, for example 00:01:31:11.");
  }
  const hours = Number(match[1]);
  const minutes = Number(match[2]);
  const seconds = Number(match[3]);
  const frames = Number(match[4]);
  const maximumFrame = Math.ceil(fps) - 1;
  if (frames < 0 || frames > maximumFrame) {
    throw new Error(`At ${fps} FPS, the frame value must be between 00 and ${String(maximumFrame).padStart(2, "0")}.`);
  }
  return hours * 3600 + minutes * 60 + seconds + frames / fps;
}

export function createBeatCalibration({
  audioInput, autoSaveSessionQuiet, beatCalibration, beatCalibrationAnchors, beatCalibrationCaptureButton,
  beatCalibrationFpsInput, beatCalibrationGridType, beatCalibrationGridTypeHint, beatCalibrationInstruction,
  beatCalibrationTimecodeGrid, beatCalibrationTimecodeHint, beatCalibrationTimecodeInput,
  beatCalibrationWizard, beatMarkersButton, currentGlobalTime, enforceAudioTimelineEnd,
  loadedGlobalAudioDuration, lyricNoteButton, node, pauseTimelineForEditing, projectInput, pushHistory,
  render, sceneNoteButton, setGlobalPlaybackTime, state, videoNoteButton,
}) {
  function showBeatMarkersIfAvailable() {
    if (Array.isArray(state.beats) && state.beats.length) {
      setBeatMarkersVisible(true);
    }
  }

  async function reloadBeatMarkersFromAudio() {
    closeBeatCalibrationWizard();
    const audioPath = String(audioInput.value || getWidget(node, "audio_path")?.value || "").trim();
    if (!audioPath) {
      toast("No audio path is loaded, so beat markers cannot be analyzed.", true);
      return false;
    }
    try {
      const data = await postJson("/vrgdg/music_builder/analyze_audio", {
        audio_path: audioPath,
        project_folder: projectInput.value || state.projectFolder || "",
        target_peaks: 1800,
      }, 90000);
      audioInput.value = data.audio_path || audioInput.value;
      setWidgetValue(node, "audio_path", audioInput.value);
      state.duration = Math.max(Number(state.duration || 0), Number(data.duration || 0));
      state.audioDuration = Number(data.duration || 0);
      enforceAudioTimelineEnd();
      state.peaks = Array.isArray(data.peaks) ? data.peaks : [];
      state.beats = Array.isArray(data.beats) ? data.beats : [];
      state.detectedTempoBpm = Math.max(0, Number(data.tempo_bpm || 0));
      state.beatCalibration = null;
      setBeatMarkersVisible(Boolean(state.beats.length));
      if (!state.beats.length) {
        toast("Audio analysis finished, but no beat markers were detected.", true);
        return false;
      }
      render();
      toast(`Loaded ${state.beats.length} beat marker${state.beats.length === 1 ? "" : "s"}.`);
      await autoSaveSessionQuiet("beat markers refreshed");
      return true;
    } catch (error) {
      toast(`Could not reload beat markers:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  async function ensureAutoBpmForCalibration() {
    const draft = beatCalibration.draft;
    if (!draft) return 0;
    const cachedBpm = Math.max(0, Number(draft.tempoBpm || state.detectedTempoBpm || 0));
    if (cachedBpm > 0) return cachedBpm;
    if (draft.tempoLoading) return 0;
    const audioPath = String(audioInput.value || getWidget(node, "audio_path")?.value || "").trim();
    if (!audioPath) {
      toast("Load project audio before using automatic BPM detection.", true);
      return 0;
    }
    draft.tempoLoading = true;
    renderBeatCalibrationWizard();
    try {
      const data = await postJson("/vrgdg/music_builder/analyze_audio", {
        audio_path: audioPath,
        project_folder: projectInput.value || state.projectFolder || "",
        target_peaks: 1800,
      }, 90000);
      const bpm = Math.max(0, Number(data.tempo_bpm || 0));
      if (!(bpm > 0)) {
        toast("The audio analyzer could not determine a BPM for this song.", true);
        return 0;
      }
      state.peaks = Array.isArray(data.peaks) ? data.peaks : state.peaks;
      state.beats = Array.isArray(data.beats) ? data.beats : state.beats;
      state.detectedTempoBpm = bpm;
      draft.sourceBeats = state.beats
        .map((beat) => Number(beat?.time ?? beat))
        .filter((beat) => Number.isFinite(beat) && beat >= 0)
        .sort((a, b) => a - b);
      draft.tempoBpm = bpm;
      state.beatCalibration = null;
      setBeatMarkersVisible(Boolean(state.beats.length));
      await autoSaveSessionQuiet("detected audio BPM");
      return bpm;
    } catch (error) {
      toast(`Could not analyze BPM:\n${String(error?.message || error)}`, true);
      return 0;
    } finally {
      if (beatCalibration.draft === draft) {
        draft.tempoLoading = false;
        renderBeatCalibrationWizard();
        render();
      }
    }
  }

  async function ensureCapCutBeatsForCalibration() {
    const draft = beatCalibration.draft;
    if (!draft) return null;
    if (draft.capCutImport?.beats?.length) return draft.capCutImport;
    if (draft.capCutLoading) return null;
    draft.capCutLoading = true;
    renderBeatCalibrationWizard();
    try {
      const data = await postJson("/vrgdg/music_builder/import_capcut_beats", {
        audio_duration: loadedGlobalAudioDuration(),
      }, 90000);
      const beats = Array.isArray(data.beats)
        ? data.beats.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value >= 0).sort((a, b) => a - b)
        : [];
      if (beats.length < 2) throw new Error("The matching CapCut project did not contain a usable beat-marker sequence.");
      draft.capCutImport = { ...data, beats };
      return draft.capCutImport;
    } catch (error) {
      toast(`Could not locate matching CapCut beats:\n${String(error?.message || error)}`, true);
      return null;
    } finally {
      if (beatCalibration.draft === draft) {
        draft.capCutLoading = false;
        renderBeatCalibrationWizard();
      }
    }
  }

  function beatCalibrationProjectedTime(sourceIndex) {
    const draft = beatCalibration.draft;
    const sourceTime = Number(draft?.sourceBeats?.[sourceIndex]);
    if (!draft || !Number.isFinite(sourceTime) || !draft.anchors.length) return sourceTime;
    const first = draft.anchors[0];
    if (draft.anchors.length === 1) return sourceTime + (first.targetTime - first.sourceTime);
    const middle = draft.anchors[1];
    const sourceSpan = middle.sourceTime - first.sourceTime;
    if (sourceSpan <= 0) return sourceTime;
    const scale = (middle.targetTime - first.targetTime) / sourceSpan;
    return first.targetTime + (sourceTime - first.sourceTime) * scale;
  }

  function renderBeatCalibrationWizard() {
    const anchors = beatCalibration.draft?.anchors || [];
    const usesAutoBpm = beatCalibrationGridType.value === "auto_bpm";
    const usesCapCutImport = beatCalibrationGridType.value === "capcut_import";
    const labels = usesAutoBpm ? ["Chosen start"] : ["First", "Middle", "Last"];
    const usesEvenGrid = beatCalibrationGridType.value === "even_grid";
    const detectedBpm = Math.max(0, Number(beatCalibration.draft?.tempoBpm || state.detectedTempoBpm || 0));
    const capCutImport = beatCalibration.draft?.capCutImport || null;
    beatCalibrationTimecodeGrid.style.display = usesCapCutImport ? "none" : "grid";
    beatCalibrationTimecodeHint.style.display = usesCapCutImport ? "none" : "block";
    beatCalibrationCaptureButton.disabled = Boolean(usesCapCutImport && beatCalibration.draft?.capCutLoading);
    beatCalibrationGridTypeHint.textContent = usesCapCutImport
      ? beatCalibration.draft?.capCutLoading
        ? "Searching the local CapCut project index for the newest project matching this audio duration..."
        : capCutImport
          ? `Found ${capCutImport.project_name || "CapCut project"}: ${capCutImport.beats.length} markers from ${capCutImport.beat_source === "timeline_markers" ? "CapCut's frame-aligned timeline markers" : "the AI beat cache"}.`
          : "Imports exact beat timestamps from the newest local CapCut project matching this audio duration. CapCut files remain read-only."
      : usesAutoBpm
      ? `${beatCalibration.draft?.tempoLoading ? "Analyzing audio BPM..." : detectedBpm > 0 ? `Detected BPM: ${detectedBpm.toFixed(3)}.` : "BPM has not been analyzed yet."} Choose one exact starting beat; the grid will be generated evenly from there to the audio end.`
      : usesEvenGrid
        ? "Builds one constant-tempo grid from the first through last anchor and rounds every marker to the selected FPS. The middle anchor checks that the same beat count was matched."
        : "Preserves the detected marker pattern, then warps it separately before and after the middle anchor.";
    beatCalibrationInstruction.textContent = usesCapCutImport
      ? capCutImport
        ? `Audio: ${capCutImport.audio_name || "unknown"}\nFirst: ${formatTime(capCutImport.beats[0])}   Last: ${formatTime(capCutImport.beats[capCutImport.beats.length - 1])}\nCapCut FPS: ${Number(capCutImport.project_fps || 0).toFixed(3)}`
        : "Select this mode to locate and preview the matching CapCut beat data."
      : usesAutoBpm
      ? anchors.length === 0
        ? "Enter the exact CapCut timecode for the beat where the new grid should begin, or move the playhead there, then capture it."
        : "The start beat is captured. Apply the automatically detected BPM grid."
      : anchors.length === 0
      ? "Enter the first beat's CapCut timecode below, or move the playhead there, then capture it."
      : anchors.length === 1
        ? "Enter a clearly identifiable middle beat's CapCut timecode, or move the playhead there, then capture it."
        : anchors.length === 2
          ? "Enter the last real beat's CapCut timecode, or move the playhead there, then capture it."
          : usesEvenGrid
            ? "All three anchors are captured. Apply calibration to replace the detected pattern with one even, frame-aligned grid."
            : "All three anchors are captured. Apply calibration to correct offset and cumulative drift.";
    beatCalibrationAnchors.textContent = usesCapCutImport
      ? capCutImport?.draft_path ? `Project file: ${capCutImport.draft_path}\nBeat cache: ${capCutImport.beat_cache_path || "not needed"}` : ""
      : labels.map((label, index) => {
        const anchor = anchors[index];
        return `${label}: ${anchor ? `${formatTime(anchor.targetTime)}  (matched marker ${formatTime(anchor.sourceTime)})` : "not captured"}`;
      }).join("\n");
    beatCalibrationCaptureButton.textContent = usesCapCutImport
      ? beatCalibration.draft?.capCutLoading ? "Finding CapCut Markers..." : "Import CapCut Markers"
      : usesAutoBpm
      ? anchors.length === 0 ? "Capture Grid Start" : "Apply Auto BPM Grid"
      : anchors.length === 0
      ? "Capture First Beat"
      : anchors.length === 1
        ? "Capture Middle Beat"
        : anchors.length === 2
          ? "Capture Last Beat"
          : "Apply Calibration";
  }

  async function openBeatCalibrationWizard() {
    if (!Array.isArray(state.beats) || !state.beats.length) {
      const loaded = await reloadBeatMarkersFromAudio();
      if (!loaded) return false;
    }
    const sourceBeats = state.beats
      .map((beat) => Number(beat?.time ?? beat))
      .filter((beat) => Number.isFinite(beat) && beat >= 0)
      .sort((a, b) => a - b);
    if (sourceBeats.length < 3) {
      toast("At least three beat markers are required for three-point calibration.", true);
      return false;
    }
    beatCalibration.draft = {
      sourceBeats,
      anchors: [],
      tempoBpm: Math.max(0, Number(state.detectedTempoBpm || 0)),
      tempoLoading: false,
      capCutLoading: false,
      capCutImport: null,
    };
    const projectFps = Number(state.i2vVideoSettings?.fps || 24);
    beatCalibrationFpsInput.value = String(Number.isFinite(projectFps) && projectFps > 0 ? projectFps : 24);
    beatCalibrationTimecodeInput.value = "";
    beatCalibrationWizard.style.display = "flex";
    setBeatMarkersVisible(true);
    renderBeatCalibrationWizard();
    render();
    return true;
  }

  function captureBeatCalibrationAnchor() {
    const draft = beatCalibration.draft;
    const maximumAnchors = beatCalibrationGridType.value === "auto_bpm" ? 1 : 3;
    if (!draft || draft.anchors.length >= maximumAnchors) return false;
    const capCutTimecode = String(beatCalibrationTimecodeInput.value || "").trim();
    let targetTime = Number(Math.max(0, currentGlobalTime()).toFixed(3));
    if (capCutTimecode) {
      try {
        targetTime = Number(parseCapCutTimecode(capCutTimecode, beatCalibrationFpsInput.value).toFixed(3));
      } catch (error) {
        toast(String(error?.message || error), true);
        return false;
      }
      const audioEnd = loadedGlobalAudioDuration();
      if (audioEnd > 0 && targetTime > audioEnd + 0.0005) {
        toast(`That timecode is beyond the loaded audio duration of ${formatTime(audioEnd)}.`, true);
        return false;
      }
      state.sceneSelectionUsesGlobalAudio = true;
      setGlobalPlaybackTime(targetTime);
    }
    const previous = draft.anchors[draft.anchors.length - 1] || null;
    if (previous && targetTime <= previous.targetTime + 0.05) {
      toast("Move the playhead later than the previously captured beat.", true);
      return false;
    }
    const firstCandidate = previous ? previous.sourceIndex + 1 : 0;
    let sourceIndex = -1;
    let bestDelta = Number.POSITIVE_INFINITY;
    for (let index = firstCandidate; index < draft.sourceBeats.length; index += 1) {
      const projected = beatCalibrationProjectedTime(index);
      const delta = Math.abs(projected - targetTime);
      if (delta < bestDelta) {
        sourceIndex = index;
        bestDelta = delta;
      }
    }
    if (sourceIndex < 0) {
      toast("No later beat marker is available for this calibration point.", true);
      return false;
    }
    draft.anchors.push({
      sourceIndex,
      sourceTime: draft.sourceBeats[sourceIndex],
      targetTime,
    });
    beatCalibrationTimecodeInput.value = "";
    renderBeatCalibrationWizard();
    return true;
  }

  async function applyCapCutBeatImport() {
    const imported = beatCalibration.draft?.capCutImport || await ensureCapCutBeatsForCalibration();
    const beats = Array.isArray(imported?.beats) ? imported.beats : [];
    if (beats.length < 2) return false;
    const projectFps = Number(imported.project_fps || 0);
    const builderFps = Number(beatCalibrationFpsInput.value || state.i2vVideoSettings?.fps || 0);
    const fpsWarning = projectFps > 0 && builderFps > 0 && Math.abs(projectFps - builderFps) > 0.001
      ? `\nWARNING: CapCut uses ${projectFps.toFixed(3)} FPS while the Builder is set to ${builderFps.toFixed(3)} FPS.\n`
      : "";
    const confirmed = window.confirm(
      `Import exact CapCut beat markers?\n\n` +
      `Project: ${imported.project_name || "unknown"}\n` +
      `Audio: ${imported.audio_name || "unknown"}\n` +
      `Markers: ${beats.length}\n` +
      `First: ${formatTime(beats[0])}\n` +
      `Last: ${formatTime(beats[beats.length - 1])}\n` +
      `Source: ${imported.beat_source === "timeline_markers" ? "frame-aligned CapCut timeline markers" : "CapCut AI beat cache"}\n` +
      fpsWarning +
      "\nThe current Builder beat markers will be replaced. Scene timing will not change."
    );
    if (!confirmed) return false;

    pauseTimelineForEditing();
    pushHistory();
    state.beats = beats.map((value) => Number(Number(value).toFixed(6)));
    state.beatCalibration = {
      mode: "capcut_import",
      project_name: imported.project_name || "",
      draft_path: imported.draft_path || "",
      beat_cache_path: imported.beat_cache_path || "",
      beat_source: imported.beat_source || "",
      project_fps: projectFps,
      marker_count: state.beats.length,
      first_time: state.beats[0],
      last_time: state.beats[state.beats.length - 1],
      calibrated_at: new Date().toISOString(),
    };
    beatCalibration.draft = null;
    beatCalibrationWizard.style.display = "none";
    setBeatMarkersVisible(true);
    render();
    await autoSaveSessionQuiet("imported exact CapCut beat markers");
    toast(`Imported ${state.beats.length} exact beat markers from CapCut project ${imported.project_name || ""}. Scene timing was not changed.`);
    return true;
  }

  async function applyAutoBpmCalibration() {
    const draft = beatCalibration.draft;
    const startAnchor = draft?.anchors?.[0];
    if (!draft || !startAnchor) return false;
    const bpm = Math.max(0, Number(draft.tempoBpm || state.detectedTempoBpm || 0));
    if (!(bpm > 0)) {
      toast("No automatically detected BPM is available yet.", true);
      return false;
    }
    const fps = Number(beatCalibrationFpsInput.value);
    if (!Number.isFinite(fps) || fps <= 0 || fps > 120) {
      toast("Enter a valid FPS from 1 through 120.", true);
      return false;
    }
    const audioEnd = loadedGlobalAudioDuration();
    if (!(audioEnd > startAnchor.targetTime + 0.05)) {
      toast("The chosen grid start must be before the end of the loaded audio.", true);
      return false;
    }
    const beatInterval = 60 / bpm;
    const calibrated = [Number(startAnchor.targetTime.toFixed(3))];
    for (let offset = 1; ; offset += 1) {
      const idealTime = startAnchor.targetTime + offset * beatInterval;
      if (idealTime > audioEnd + 0.0005) break;
      const rounded = Number((Math.round(idealTime * fps) / fps).toFixed(3));
      if (rounded > audioEnd + 0.0005) break;
      if (rounded > calibrated[calibrated.length - 1] + 0.0005) calibrated.push(rounded);
    }
    if (calibrated.length < 2) {
      toast("The detected BPM did not produce a usable grid after the chosen start.", true);
      return false;
    }
    const confirmed = window.confirm(
      `Apply auto-detected BPM grid?\n\n` +
      `Detected tempo: ${bpm.toFixed(3)} BPM\n` +
      `Chosen start: ${formatTime(startAnchor.targetTime)}\n` +
      `Beat interval: ${beatInterval.toFixed(6)} seconds\n` +
      `Generated markers: ${calibrated.length}\n` +
      `Last marker: ${formatTime(calibrated[calibrated.length - 1])}\n\n` +
      "Markers before the chosen start will be removed. Scene timing will not change."
    );
    if (!confirmed) return false;

    pauseTimelineForEditing();
    pushHistory();
    state.beats = calibrated;
    state.beatCalibration = {
      mode: "auto_bpm",
      detected_bpm: Number(bpm.toFixed(6)),
      beat_interval: Number(beatInterval.toFixed(9)),
      fps: Number(fps.toFixed(6)),
      start_time: Number(startAnchor.targetTime.toFixed(3)),
      source_marker_index: startAnchor.sourceIndex,
      source_marker_time: Number(startAnchor.sourceTime.toFixed(3)),
      calibrated_at: new Date().toISOString(),
    };
    beatCalibration.draft = null;
    beatCalibrationWizard.style.display = "none";
    setBeatMarkersVisible(true);
    render();
    await autoSaveSessionQuiet("auto detected BPM beat grid");
    toast(`Auto BPM grid applied at ${bpm.toFixed(3)} BPM. Scene timing was not changed.`);
    return true;
  }

  async function applyThreePointBeatCalibration() {
    const draft = beatCalibration.draft;
    if (!draft || draft.anchors.length !== 3) return false;
    const [first, middle, last] = draft.anchors;
    const gridType = beatCalibrationGridType.value === "even_grid" ? "even_grid" : "detected_warp";
    const firstSourceSpan = middle.sourceTime - first.sourceTime;
    const secondSourceSpan = last.sourceTime - middle.sourceTime;
    const firstTargetSpan = middle.targetTime - first.targetTime;
    const secondTargetSpan = last.targetTime - middle.targetTime;
    if (firstSourceSpan <= 0 || secondSourceSpan <= 0 || firstTargetSpan <= 0 || secondTargetSpan <= 0) {
      toast("The calibration points are not in a valid chronological order.", true);
      return false;
    }
    const firstScale = firstTargetSpan / firstSourceSpan;
    const secondScale = secondTargetSpan / secondSourceSpan;
    if (firstScale < 0.5 || firstScale > 1.5 || secondScale < 0.5 || secondScale > 1.5) {
      toast("The selected beats imply an extreme tempo change. Capture the middle and last points again using corresponding beat markers.", true);
      return false;
    }

    const calibrated = [];
    let evenGridDetails = null;
    if (gridType === "even_grid") {
      const fps = Number(beatCalibrationFpsInput.value);
      if (!Number.isFinite(fps) || fps <= 0 || fps > 120) {
        toast("Enter a valid FPS from 1 through 120.", true);
        return false;
      }
      const totalIntervals = last.sourceIndex - first.sourceIndex;
      const middleIntervalIndex = middle.sourceIndex - first.sourceIndex;
      if (totalIntervals < 2 || middleIntervalIndex <= 0 || middleIntervalIndex >= totalIntervals) {
        toast("The anchors do not identify a usable beat count. Capture three corresponding beats again.", true);
        return false;
      }
      const beatInterval = (last.targetTime - first.targetTime) / totalIntervals;
      if (!Number.isFinite(beatInterval) || beatInterval <= 0.05) {
        toast("The selected anchors do not produce a usable even beat interval.", true);
        return false;
      }
      const projectedMiddle = first.targetTime + middleIntervalIndex * beatInterval;
      const middleErrorSeconds = projectedMiddle - middle.targetTime;
      const middleErrorFrames = middleErrorSeconds * fps;
      for (let offset = 0; offset <= totalIntervals; offset += 1) {
        let markerTime;
        if (offset === 0) markerTime = first.targetTime;
        else if (offset === totalIntervals) markerTime = last.targetTime;
        else markerTime = Math.round((first.targetTime + offset * beatInterval) * fps) / fps;
        const rounded = Number(markerTime.toFixed(3));
        if (calibrated.length && rounded <= calibrated[calibrated.length - 1] + 0.0005) {
          toast("The chosen FPS is too low to represent this beat interval without duplicate markers.", true);
          return false;
        }
        calibrated.push(rounded);
      }
      evenGridDetails = {
        fps,
        beatInterval,
        bpm: 60 / beatInterval,
        middleErrorFrames,
      };
    } else {
      for (let index = first.sourceIndex; index <= last.sourceIndex; index += 1) {
        const sourceTime = draft.sourceBeats[index];
        const mapped = sourceTime <= middle.sourceTime
          ? first.targetTime + (sourceTime - first.sourceTime) * firstScale
          : middle.targetTime + (sourceTime - middle.sourceTime) * secondScale;
        const rounded = Number(mapped.toFixed(3));
        if (!calibrated.length || rounded > calibrated[calibrated.length - 1] + 0.0005) calibrated.push(rounded);
      }
    }
    if (calibrated.length < 3) {
      toast("Calibration did not produce a usable beat grid.", true);
      return false;
    }

    const calibrationDetails = evenGridDetails
      ? `Grid type: Even BPM grid\nInterval: ${evenGridDetails.beatInterval.toFixed(6)} seconds\nTempo: ${evenGridDetails.bpm.toFixed(3)} BPM\nMiddle check: ${evenGridDetails.middleErrorFrames >= 0 ? "+" : ""}${evenGridDetails.middleErrorFrames.toFixed(2)} frames${Math.abs(evenGridDetails.middleErrorFrames) > 1 ? " (warning only)" : ""}\n\n`
      : `Grid type: Warp detected markers\nGrid scale before middle: ${firstScale.toFixed(6)}\nGrid scale after middle: ${secondScale.toFixed(6)}\n\n`;
    const confirmed = window.confirm(
      `${gridType === "even_grid" ? "Apply even beat grid" : "Apply detected-marker calibration"}?\n\n` +
      `First: ${formatTime(first.targetTime)}\nMiddle: ${formatTime(middle.targetTime)}\nLast: ${formatTime(last.targetTime)}\n\n` +
      calibrationDetails +
      "Markers before the first beat and after the last beat will be removed. Scene timing will not change."
    );
    if (!confirmed) return false;

    pauseTimelineForEditing();
    pushHistory();
    state.beats = calibrated;
    state.beatCalibration = {
      mode: gridType,
      anchors: draft.anchors.map((anchor) => ({
        source_index: anchor.sourceIndex,
        source_time: Number(anchor.sourceTime.toFixed(3)),
        target_time: Number(anchor.targetTime.toFixed(3)),
      })),
      first_scale: Number(firstScale.toFixed(9)),
      second_scale: Number(secondScale.toFixed(9)),
      ...(evenGridDetails ? {
        fps: Number(evenGridDetails.fps.toFixed(6)),
        beat_interval: Number(evenGridDetails.beatInterval.toFixed(9)),
        bpm: Number(evenGridDetails.bpm.toFixed(6)),
        middle_error_frames: Number(evenGridDetails.middleErrorFrames.toFixed(3)),
      } : {}),
      calibrated_at: new Date().toISOString(),
    };
    beatCalibration.draft = null;
    beatCalibrationWizard.style.display = "none";
    setBeatMarkersVisible(true);
    render();
    await autoSaveSessionQuiet(gridType === "even_grid" ? "even beat grid calibration" : "detected beat marker calibration");
    toast(`${gridType === "even_grid" ? "Even beat grid" : "Detected-marker calibration"} applied. Scene timing was not changed.`);
    return true;
  }

  function closeBeatCalibrationWizard() {
    beatCalibration.draft = null;
    beatCalibrationWizard.style.display = "none";
  }

  function setBeatMarkersVisible(visible) {
    state.showBeatMarkers = Boolean(visible);
    beatMarkersButton.style.background = state.showBeatMarkers ? "#164e63" : "#27272a";
    beatMarkersButton.style.borderColor = state.showBeatMarkers ? "#0891b2" : "#3f3f46";
    beatMarkersButton.style.color = state.showBeatMarkers ? "#cffafe" : "#fafafa";
  }

  function syncSceneNoteControls() {
    sceneNoteButton.style.background = state.showTimelineSceneNotes ? "#164e63" : "#27272a";
    sceneNoteButton.style.borderColor = state.showTimelineSceneNotes ? "#0891b2" : "#3f3f46";
    sceneNoteButton.style.color = state.showTimelineSceneNotes ? "#cffafe" : "#f4f4f5";
    sceneNoteButton.textContent = state.showTimelineSceneNotes ? "Hide Scene Notes" : "+ Scene Note";
  }

  function syncVideoNoteControls() {
    videoNoteButton.style.background = state.showTimelineVideoNotes ? "#164e63" : "#27272a";
    videoNoteButton.style.borderColor = state.showTimelineVideoNotes ? "#0891b2" : "#3f3f46";
    videoNoteButton.style.color = state.showTimelineVideoNotes ? "#cffafe" : "#f4f4f5";
    videoNoteButton.textContent = state.showTimelineVideoNotes ? "Hide Video Notes" : "+ Video Note";
  }

  function syncLyricNoteControls() {
    lyricNoteButton.style.background = state.showTimelineLyricNotes ? "#164e63" : "#27272a";
    lyricNoteButton.style.borderColor = state.showTimelineLyricNotes ? "#0891b2" : "#3f3f46";
    lyricNoteButton.style.color = state.showTimelineLyricNotes ? "#cffafe" : "#f4f4f5";
    lyricNoteButton.textContent = state.showTimelineLyricNotes ? "Hide Line Notes" : "+ Line Note";
  }

  return {
    applyAutoBpmCalibration, applyCapCutBeatImport, applyThreePointBeatCalibration,
    captureBeatCalibrationAnchor, closeBeatCalibrationWizard, ensureAutoBpmForCalibration,
    ensureCapCutBeatsForCalibration, openBeatCalibrationWizard, reloadBeatMarkersFromAudio,
    renderBeatCalibrationWizard, setBeatMarkersVisible, showBeatMarkersIfAvailable, syncLyricNoteControls,
    syncSceneNoteControls, syncVideoNoteControls,
  };
}
