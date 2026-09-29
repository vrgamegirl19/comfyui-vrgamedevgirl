import {
  applyCompactButtonLabel,
  makeButton,
  makeCheckbox,
  makeField,
  makeInput,
  makeSelect,
  normalizeProjectVideoEngine,
  normalizeVideoType,
  toast,
} from "./controls.mjs";
import { formatTime } from "./format.mjs";
import {
  formatCueTime,
  lyricCueTextParts,
  miniMaxEffectiveCueEnd,
  miniMaxNextCueStartTime,
  rebuildSpeakerCueMapFromTimestampedSegments,
  singerCuePlaybackRangeForCue,
  syncCueEndBoundariesFromNextStarts,
  timestampedCueSegments,
} from "./lyric_cues.mjs";
import { runTimestampedCueWorkflow } from "./lyric_transcription.mjs";
import { normalizeMiniMaxSpeakerAssignments } from "./minimax_h3.mjs";
import { syncLyricTextFromCueMap, syncMiniMaxSpeakerAssignmentLegacyFields } from "./minimax_speaker_cues.mjs";
import { flattenLyricForPrompt, isInstrumentalLyricText } from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder } from "./reference_data.mjs";
import { timelineAudioStartForSegment, timelineSegmentDuration } from "./timeline_state.mjs";

export function createMiniMaxSpeakerAssignments({
  activateGlobalTimelineAudioPlayback, activeSegment, audio, autoSaveSessionQuiet,
  autoTimeMiniMaxSingerCuesForSegment, createProgressWindow, miniMaxAddSpeakerCueButton,
  miniMaxAutoTimeBeforePrompt, miniMaxH3SettingsForSegment, miniMaxSpeakerAssignmentList,
  miniMaxSpeakerAssignmentNote, normalizeLyricCueMapForSegment, playSceneAudioFrom, playSingerCueRange,
  prepareSceneAudioClipForTimestamping, pushHistory, render, selectedPerformerSubjectsForSegment,
  singerCueRelativePlayheadTime, startSilentTimelinePlayback, state, storyboardReferenceDataForSegment,
  syncInspector, updatePlayPauseButton,
}) {
  function miniMaxMappedSpeakersForSegment(segment) {
    if (!segment || segment.no_character_present) return [];
    return (storyboardReferenceDataForSegment(segment).subject_refs || [])
      .filter((subject) => (subject.reference_type || "character") === "character")
      .map((subject) => ({ id: String(subject.id || ""), name: String(subject.name || "Character").trim() || "Character" }));
  }

  function ensureMiniMaxSpeakerAssignments(segment, speakers = miniMaxMappedSpeakersForSegment(segment)) {
    if (!segment) return [];
    let cues = normalizeMiniMaxSpeakerAssignments(segment.minimax_speaker_assignments || segment.speaker_assignments || []);
    if (!cues.length) {
      const existingLine = isInstrumentalLyricText(segment.lyric_text) ? "" : String(segment.lyric_text || "").trim();
      if (existingLine) {
        const preferredName = String((Array.isArray(segment.lyric_singers) ? segment.lyric_singers[0] : "") || "").trim();
        const speaker = speakers.find((item) => item.name.toLowerCase() === preferredName.toLowerCase()) || speakers[0] || { id: "", name: preferredName };
        cues = normalizeMiniMaxSpeakerAssignments([{ speaker_id: speaker.id, speaker_name: speaker.name, text: existingLine }]);
      } else if (speakers.length) {
        cues = normalizeMiniMaxSpeakerAssignments(speakers.map((speaker) => ({ speaker_id: speaker.id, speaker_name: speaker.name, text: "" })));
      }
      segment.minimax_speaker_assignments = cues;
    }
    return cues;
  }

  function isMiniMaxSingerAssignmentMode(segment = activeSegment()) {
    if (!segment) return false;
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    const settings = miniMaxH3SettingsForSegment(segment);
    const videoType = normalizeVideoType(segment?.performance_mode || state.videoType);
    return Boolean(miniMaxProject && videoType === "singing" && settings.audio_mode === "input_audio");
  }

  function isMiniMaxBuiltInSpeakerAssignmentMode(segment = activeSegment()) {
    if (!segment) return false;
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    const settings = miniMaxH3SettingsForSegment(segment);
    const videoType = normalizeVideoType(segment?.performance_mode || state.videoType);
    return Boolean(miniMaxProject && videoType === "speaking" && settings.audio_mode === "built_in_audio");
  }

  function renderMiniMaxSpeakerAssignmentPanel() {
    const segment = activeSegment();
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    const settings = miniMaxH3SettingsForSegment(segment);
    const shortFilm = normalizeVideoType(segment?.performance_mode || state.videoType) === "speaking";
    const singerMode = isMiniMaxSingerAssignmentMode(segment);
    const enabled = Boolean(miniMaxProject && segment && !segment.no_character_present && ((shortFilm && settings.audio_mode === "built_in_audio") || singerMode));
    const speakers = enabled ? miniMaxMappedSpeakersForSegment(segment) : [];
    miniMaxSpeakerAssignmentList.replaceChildren();
    miniMaxAutoTimeBeforePrompt.input.checked = Boolean(state.autoTimeSingerCuesBeforePrompt);
    miniMaxAutoTimeBeforePrompt.input.onchange = () => {
      state.autoTimeSingerCuesBeforePrompt = Boolean(miniMaxAutoTimeBeforePrompt.input.checked);
      autoSaveSessionQuiet("MiniMax auto-time-before-prompt setting changed").catch(() => null);
    };
    miniMaxAddSpeakerCueButton.disabled = !enabled || !speakers.length || (singerMode && segment?.lyric_performance_mode !== "cue_map");
    miniMaxAddSpeakerCueButton.textContent = singerMode ? "Add Lyric Cue" : "Add Dialogue Cue";
    miniMaxAddSpeakerCueButton.style.display = (singerMode || isMiniMaxBuiltInSpeakerAssignmentMode(segment)) ? "none" : "";
    if (!miniMaxProject || !segment) {
      miniMaxSpeakerAssignmentNote.textContent = "Choose an active MiniMax scene to assign vocal performers.";
      return;
    }
    if (singerMode) {
      if (segment.no_character_present) {
        miniMaxSpeakerAssignmentNote.textContent = "This scene is marked No character present, so it cannot contain assigned singing.";
        return;
      }
      miniMaxSpeakerAssignmentNote.textContent = "Singer Assignment uses the same performers mapped in Reference Builder. Choose Together when all selected singers perform the full line, or Split cue map when different singers take turns on lyric chunks.";
      if (!speakers.length) {
        const empty = document.createElement("div");
        empty.textContent = "No singers are mapped to this scene yet. Map performer subjects in Reference Builder first.";
        empty.style.cssText = "border:1px dashed #334155;border-radius:7px;padding:12px;color:#94a3b8;text-align:center;font-size:12px;";
        miniMaxSpeakerAssignmentList.append(empty);
        return;
      }
      let selectedPerformers = selectedPerformerSubjectsForSegment(segment);
      const selectedIds = new Set(selectedPerformers.map((subject) => String(subject.id)));
      if (!selectedIds.size && speakers[0]?.id) {
        segment.lyric_singers = [speakers[0].name];
        selectedIds.add(String(speakers[0].id));
        selectedPerformers = speakers.slice(0, 1);
      } else if (selectedPerformers.length) {
        segment.lyric_singers = selectedPerformers.map((subject) => subject.name || "Character");
      }
      const audioTools = document.createElement("div");
      audioTools.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;";
      const playSceneCueAudio = makeButton("Play Scene Audio", "primary");
      const jumpSceneCueAudio = makeButton("Jump To Scene Start");
      const autoTimeSceneCues = makeButton("Auto Time This Scene");
      applyCompactButtonLabel(playSceneCueAudio, "Play\nScene Audio", { noMap: true, padding: "7px 6px", title: "Play this scene's audio range." });
      applyCompactButtonLabel(jumpSceneCueAudio, "Jump To\nScene Start", { noMap: true, padding: "7px 6px", title: "Move the playhead to this scene start." });
      applyCompactButtonLabel(autoTimeSceneCues, "Auto Time\nThis Scene", { noMap: true, padding: "7px 6px", title: "Use the existing Stable-ts timestamp workflow to fill cue start/end times for this scene. Manual editing stays available." });
      playSceneCueAudio.onclick = () => {
        const start = timelineAudioStartForSegment(segment);
        if (!playSceneAudioFrom(start)) {
          activateGlobalTimelineAudioPlayback(Math.max(0, Number(segment.start || 0)));
          audio.play().then(updatePlayPauseButton).catch(() => startSilentTimelinePlayback(Math.max(0, Number(segment.start || 0))));
        }
      };
      jumpSceneCueAudio.onclick = () => {
        const start = Math.max(0, Number(segment.start || 0));
        activateGlobalTimelineAudioPlayback(start);
        toast(`Playhead moved to ${formatTime(start)}. Play the scene, then use Set Start / Set End on cue rows.`);
      };
      autoTimeSceneCues.onclick = () => autoTimeMiniMaxSingerCuesForSegment(segment);
      audioTools.append(playSceneCueAudio, jumpSceneCueAudio, autoTimeSceneCues);
      miniMaxSpeakerAssignmentList.append(audioTools);

      const modeWrap = document.createElement("div");
      modeWrap.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.2fr);gap:8px;align-items:end;border:1px solid #155e75;border-radius:7px;background:#07111f;padding:9px;";
      const performerSelect = document.createElement("select");
      performerSelect.multiple = true;
      performerSelect.size = Math.min(5, Math.max(2, speakers.length));
      performerSelect.style.cssText = "min-height:74px;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:6px;font-size:12px;";
      speakers.forEach((speaker) => {
        const option = new Option(speaker.name, speaker.id);
        option.selected = selectedIds.has(String(speaker.id));
        performerSelect.append(option);
      });
      const performanceMode = makeSelect(["together", "cue_map"], segment.lyric_performance_mode || "together");
      performanceMode.options[0].textContent = "Together / same full line";
      performanceMode.options[1].textContent = "Split lyric cues";
      performerSelect.onchange = () => {
        const chosen = Array.from(performerSelect.selectedOptions || []).map((option) => speakers.find((speaker) => speaker.id === option.value)).filter(Boolean);
        segment.lyric_singers = chosen.map((speaker) => speaker.name);
        const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
        if (!refs.performer_scene_map || typeof refs.performer_scene_map !== "object") refs.performer_scene_map = {};
        if (chosen.length) refs.performer_scene_map[segment.id] = chosen.map((speaker) => speaker.id);
        else delete refs.performer_scene_map[segment.id];
        state.fluxReferenceBuilder = refs;
        if (chosen.length < 2 && !segment.lyric_shot_word_timing_enabled) {
          segment.lyric_performance_mode = "together";
          segment.lyric_cue_map = [];
        }
        renderMiniMaxSpeakerAssignmentPanel();
        autoSaveSessionQuiet("MiniMax singer performers changed").catch(() => null);
      };
      performanceMode.onchange = () => {
        segment.lyric_performance_mode = performanceMode.value;
        if (performanceMode.value === "cue_map" && !normalizeLyricCueMapForSegment(segment, undefined, { preserveBlank: true }).length) {
          const performers = selectedPerformerSubjectsForSegment(segment);
          segment.lyric_cue_map = lyricCueTextParts(segment.lyric_text).map((text, index) => {
            const performer = performers[index % performers.length] || performers[0] || {};
            return { text, singer_id: performer.id || "", singer_name: performer.name || "" };
          });
        }
        renderMiniMaxSpeakerAssignmentPanel();
        autoSaveSessionQuiet("MiniMax singer assignment mode changed").catch(() => null);
      };
      modeWrap.append(makeField("Singers / performers", performerSelect), makeField("Performance mode", performanceMode));
      miniMaxSpeakerAssignmentList.append(modeWrap);
      const timedWords = makeCheckbox("Match exact sung words to each shot", Boolean(segment.lyric_shot_word_timing_enabled));
      timedWords.wrapper.style.cssText += "border:1px solid #155e75;border-radius:7px;background:#07111f;padding:10px;";
      timedWords.input.title = "Optional. Auto Time analyzes the scene audio, keeps silent shots instrumental, and assigns only the words actually sung during each shot.";
      timedWords.input.onchange = () => {
        segment.lyric_shot_word_timing_enabled = Boolean(timedWords.input.checked);
        if (segment.lyric_shot_word_timing_enabled) {
          segment.lyric_performance_mode = "cue_map";
          const performer = selectedPerformerSubjectsForSegment(segment)[0] || {};
          const fullLyric = isInstrumentalLyricText(segment.lyric_text) ? "" : flattenLyricForPrompt(segment.lyric_text);
          if (fullLyric) segment.lyric_cue_map = [{ type: "vocal", text: fullLyric, action_note: "", singer_id: performer.id || "", singer_name: performer.name || "", start: null, end: null }];
        } else if (selectedPerformerSubjectsForSegment(segment).length < 2) {
          segment.lyric_performance_mode = "together";
          segment.lyric_cue_map = [];
        }
        renderMiniMaxSpeakerAssignmentPanel();
        autoSaveSessionQuiet("MiniMax exact shot lyric timing changed").catch(() => null);
      };
      const timedWordsNote = document.createElement("div");
      timedWordsNote.textContent = "Optional and off by default. Turn it on, then click Auto Time This Scene. Silent shots get no lip-sync; vocal shots receive only the words heard inside that shot.";
      timedWordsNote.style.cssText = "font-size:11px;color:#94a3b8;line-height:1.45;margin:-3px 4px 2px;";
      miniMaxSpeakerAssignmentList.append(timedWords.wrapper, timedWordsNote);
      if ((selectedPerformers.length >= 2 || segment.lyric_shot_word_timing_enabled) && segment.lyric_performance_mode === "cue_map" && !normalizeLyricCueMapForSegment(segment, undefined, { preserveBlank: true }).length) {
        segment.lyric_cue_map = lyricCueTextParts(segment.lyric_text).map((text, index) => {
          const performer = selectedPerformers[index % selectedPerformers.length] || selectedPerformers[0] || {};
          return { text, singer_id: performer.id || "", singer_name: performer.name || "" };
        });
      }
      if ((selectedPerformers.length < 2 && !segment.lyric_shot_word_timing_enabled) || segment.lyric_performance_mode !== "cue_map") {
        const together = document.createElement("div");
        const performerNames = Array.from(performerSelect.selectedOptions || []).map((option) => option.textContent).filter(Boolean);
        together.textContent = performerNames.length > 1
          ? `${performerNames.join(" and ")} will sing/speak the complete scene lyric together.`
          : `${performerNames[0] || "The selected singer"} will sing/speak the complete scene lyric.`;
        together.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;color:#cbd5e1;font-size:12px;line-height:1.45;";
        miniMaxSpeakerAssignmentList.append(together);
        return;
      }
      const cues = normalizeLyricCueMapForSegment(segment, undefined, { preserveBlank: true });
      segment.lyric_cue_map = cues;
      const cueActions = document.createElement("div");
      cueActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
      const addLyricCue = makeButton("Add Lyric Cue", "primary");
      const addInstrumentalCue = makeButton("Add Instrumental Cue");
      applyCompactButtonLabel(addLyricCue, "Add\nLyric Cue", { noMap: true, padding: "7px 6px" });
      applyCompactButtonLabel(addInstrumentalCue, "Add\nInstrumental", { noMap: true, padding: "7px 6px", title: "Add Instrumental Cue" });
      addLyricCue.onclick = () => {
        const performer = selectedPerformers[cues.filter((cue) => cue.type !== "instrumental").length % selectedPerformers.length] || selectedPerformers[0] || speakers[0] || {};
        const start = miniMaxNextCueStartTime(segment, cues);
        cues.push({ type: "vocal", text: "", action_note: "", singer_id: performer.id || "", singer_name: performer.name || "", start, end: null });
        segment.lyric_cue_map = cues;
        renderMiniMaxSpeakerAssignmentPanel();
      };
      addInstrumentalCue.onclick = () => {
        const start = miniMaxNextCueStartTime(segment, cues);
        cues.push({ type: "instrumental", text: "", action_note: "", singer_id: "", singer_name: "", start, end: null });
        segment.lyric_cue_map = cues;
        renderMiniMaxSpeakerAssignmentPanel();
      };
      cueActions.append(addLyricCue, addInstrumentalCue);
      miniMaxSpeakerAssignmentList.append(cueActions);
      let draggedIndex = -1;
      cues.forEach((cue, index) => {
        const row = document.createElement("div");
        row.style.cssText = "display:flex;flex-direction:column;gap:8px;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:8px;";
        row.addEventListener("dragover", (event) => {
          if (draggedIndex < 0 || draggedIndex === index) return;
          event.preventDefault();
          row.style.borderColor = "#22d3ee";
        });
        row.addEventListener("dragleave", () => { row.style.borderColor = "#334155"; });
        row.addEventListener("drop", (event) => {
          event.preventDefault();
          row.style.borderColor = "#334155";
          if (draggedIndex < 0 || draggedIndex === index) return;
          const [moved] = cues.splice(draggedIndex, 1);
          cues.splice(index, 0, moved);
          segment.lyric_cue_map = cues;
          syncLyricTextFromCueMap(segment);
          renderMiniMaxSpeakerAssignmentPanel();
          autoSaveSessionQuiet("MiniMax singer cue reordered").catch(() => null);
        });
        const handle = document.createElement("button");
        handle.type = "button";
        handle.textContent = "::";
        handle.title = "Drag to change lyric cue order";
        handle.draggable = true;
        handle.style.cssText = "height:38px;border:1px solid #334155;border-radius:6px;background:#07111f;color:#67e8f9;font-weight:900;cursor:grab;";
        handle.addEventListener("dragstart", () => { draggedIndex = index; row.style.opacity = ".55"; });
        handle.addEventListener("dragend", () => { draggedIndex = -1; row.style.opacity = ""; });
        const number = document.createElement("div");
        number.textContent = String(index + 1);
        number.style.cssText = "font-size:14px;font-weight:900;color:#cffafe;text-align:center;";
        const mainRow = document.createElement("div");
        mainRow.style.cssText = "display:grid;grid-template-columns:32px 34px minmax(106px,.5fr) minmax(132px,.65fr) minmax(180px,1.4fr) 58px;gap:7px;align-items:center;";
        const typeSelect = makeSelect(["vocal", "instrumental"], cue.type || "vocal");
        typeSelect.options[0].textContent = "Lyric";
        typeSelect.options[1].textContent = "Instrumental";
        const singerSelect = makeSelect(speakers.map((speaker) => ({ value: speaker.id, label: speaker.name })), cue.singer_id);
        const line = makeInput(cue.type === "instrumental" ? cue.action_note || "" : cue.text || "");
        line.placeholder = cue.type === "instrumental" ? "Action note while nobody sings..." : "Exact lyric words for this singer...";
        const remove = makeButton("Remove");
        applyCompactButtonLabel(remove, "Remove", { minWidth: 0, padding: "7px 6px" });
        typeSelect.addEventListener("change", () => {
          cue.type = typeSelect.value === "instrumental" ? "instrumental" : "vocal";
          if (cue.type === "instrumental") {
            cue.action_note = cue.action_note || "";
            cue.text = "";
            cue.singer_id = "";
            cue.singer_name = "";
          } else {
            const singer = selectedPerformers[0] || speakers[0] || {};
            cue.text = cue.text || cue.action_note || "";
            cue.action_note = "";
            cue.singer_id = cue.singer_id || singer.id || "";
            cue.singer_name = cue.singer_name || singer.name || "";
          }
          segment.lyric_cue_map = cues;
          renderMiniMaxSpeakerAssignmentPanel();
        });
        singerSelect.addEventListener("change", () => {
          const singer = speakers.find((item) => item.id === singerSelect.value) || { id: singerSelect.value, name: singerSelect.selectedOptions[0]?.textContent || "" };
          cue.singer_id = singer.id;
          cue.singer_name = singer.name;
          segment.lyric_cue_map = cues;
          syncLyricTextFromCueMap(segment);
          autoSaveSessionQuiet("MiniMax singer cue changed").catch(() => null);
        });
        line.addEventListener("input", () => {
          if (cue.type === "instrumental") cue.action_note = line.value;
          else cue.text = line.value;
          segment.lyric_cue_map = cues;
          syncLyricTextFromCueMap(segment);
        });
        line.addEventListener("change", () => autoSaveSessionQuiet("MiniMax lyric cue changed").catch(() => null));
        remove.onclick = () => {
          pushHistory();
          cues.splice(index, 1);
          segment.lyric_cue_map = cues;
          syncLyricTextFromCueMap(segment);
          renderMiniMaxSpeakerAssignmentPanel();
          autoSaveSessionQuiet("MiniMax singer cue removed").catch(() => null);
        };
        singerSelect.disabled = cue.type === "instrumental";
        mainRow.append(handle, number, typeSelect, singerSelect, line, remove);
        const timingRow = document.createElement("div");
        timingRow.style.cssText = "display:grid;grid-template-columns:minmax(44px,.45fr) minmax(44px,.45fr) 58px 64px 58px 58px 60px;gap:7px;align-items:center;padding-left:74px;";
        const cueRange = singerCuePlaybackRangeForCue(segment, cues, index);
        const effectiveEnd = miniMaxEffectiveCueEnd(segment, cues, index);
        const startLabel = document.createElement("div");
        startLabel.textContent = `Start: ${formatCueTime(cue.start)}`;
        startLabel.style.cssText = "font-size:11px;color:#a5f3fc;";
        const endLabel = document.createElement("div");
        endLabel.textContent = `End: ${formatCueTime(effectiveEnd)}`;
        endLabel.style.cssText = "font-size:11px;color:#a5f3fc;";
        const playCue = makeButton("Play Cue", "primary");
        const playFromCue = makeButton("Play From Here");
        const setStart = makeButton("Set Start");
        const setEnd = makeButton("Set End");
        const clearTiming = makeButton("Clear Timing");
        applyCompactButtonLabel(playCue, "Play\nCue", { noMap: true, minWidth: 0, padding: "6px 5px", title: "Play only this cue's timed audio range." });
        applyCompactButtonLabel(playFromCue, "From\nHere", { noMap: true, minWidth: 0, padding: "6px 5px", title: "Play from this cue start." });
        applyCompactButtonLabel(setStart, "Set\nStart", { noMap: true, minWidth: 0, padding: "6px 5px" });
        applyCompactButtonLabel(setEnd, "Set\nEnd", { noMap: true, minWidth: 0, padding: "6px 5px" });
        applyCompactButtonLabel(clearTiming, "Clear\nTime", { noMap: true, minWidth: 0, padding: "6px 5px", title: "Clear Timing" });
        playCue.title = "Play from this cue start to its end, the next cue start, or the scene end.";
        playFromCue.title = "Play from this cue start, or from the scene start if no cue start is set.";
        playCue.onclick = () => {
          playSingerCueRange(segment, cueRange.start, cueRange.end, playCue, "Play Cue");
        };
        playFromCue.onclick = () => {
          playSingerCueRange(segment, cueRange.start, null, playFromCue, "Play From Here");
        };
        setStart.onclick = () => {
          cue.start = singerCueRelativePlayheadTime(segment);
          if (index > 0) cues[index - 1].end = cue.start;
          if (Number.isFinite(Number(cue.end)) && cue.end <= cue.start) cue.end = null;
          syncCueEndBoundariesFromNextStarts(cues);
          segment.lyric_cue_map = cues;
          renderMiniMaxSpeakerAssignmentPanel();
        };
        setEnd.onclick = () => {
          cue.end = singerCueRelativePlayheadTime(segment);
          if (Number.isFinite(Number(cue.start)) && cue.end <= cue.start) {
            toast("Cue end must be after cue start.", true);
            cue.end = null;
          } else if (index + 1 < cues.length) {
            cues[index + 1].start = cue.end;
            if (Number.isFinite(Number(cues[index + 1].end)) && cues[index + 1].end <= cues[index + 1].start) cues[index + 1].end = null;
          }
          syncCueEndBoundariesFromNextStarts(cues);
          segment.lyric_cue_map = cues;
          renderMiniMaxSpeakerAssignmentPanel();
        };
        clearTiming.onclick = () => {
          cue.start = null;
          cue.end = null;
          segment.lyric_cue_map = cues;
          renderMiniMaxSpeakerAssignmentPanel();
        };
        timingRow.append(startLabel, endLabel, playCue, playFromCue, setStart, setEnd, clearTiming);
        row.append(mainRow, timingRow);
        miniMaxSpeakerAssignmentList.append(row);
      });
      return;
    }
    if (!shortFilm) {
      miniMaxSpeakerAssignmentNote.textContent = "Speaker Assignment is available for Short Film projects. Music Video continues to use the scene lyric/vocal line.";
      return;
    }
    if (settings.audio_mode !== "built_in_audio") {
      miniMaxSpeakerAssignmentNote.textContent = "Speaker Assignment is disabled for Input Audio because changing the words would no longer match the supplied audio. Select Built-in MiniMax Audio in Video Settings to create dialogue here.";
      return;
    }
    if (segment.no_character_present) {
      miniMaxSpeakerAssignmentNote.textContent = "This scene is marked No character present, so it cannot contain assigned dialogue.";
      return;
    }
    miniMaxSpeakerAssignmentNote.textContent = "Speaker Assignment uses timed cue rows for MiniMax built-in audio. Add Dialogue cues for spoken lines, or Instrumental cues for action/silence where nobody speaks.";
    if (!speakers.length) {
      const empty = document.createElement("div");
      empty.textContent = "No characters are mapped to this scene yet. Map them in Reference Builder first.";
      empty.style.cssText = "border:1px dashed #334155;border-radius:7px;padding:12px;color:#94a3b8;text-align:center;font-size:12px;";
      miniMaxSpeakerAssignmentList.append(empty);
      return;
    }
    const cues = ensureMiniMaxSpeakerAssignments(segment, speakers);
    segment.minimax_speaker_assignments = cues;
    const audioTools = document.createElement("div");
    const hasCustomSpeakerTimingAudio = Boolean(String(segment.custom_audio_path || "").trim());
    audioTools.style.cssText = `display:grid;grid-template-columns:${hasCustomSpeakerTimingAudio ? "1fr 1fr 1fr" : "1fr 1fr"};gap:8px;`;
    const playSceneCueAudio = makeButton("Play Scene Clock", "primary");
    const jumpSceneCueAudio = makeButton("Jump To Scene Start");
    const autoTimeSceneCues = makeButton("Auto Time This Scene");
    applyCompactButtonLabel(playSceneCueAudio, "Play\nScene Clock", { noMap: true, padding: "7px 6px", title: "Play this scene range. If no generated audio exists yet, the silent timeline clock still lets you set cue times." });
    applyCompactButtonLabel(jumpSceneCueAudio, "Jump To\nScene Start", { noMap: true, padding: "7px 6px", title: "Move the playhead to this scene start." });
    applyCompactButtonLabel(autoTimeSceneCues, "Auto Time\nThis Scene", { noMap: true, padding: "7px 6px", title: "Use Stable-ts to fill dialogue/instrumental cue start/end times from this scene's custom audio." });
    playSceneCueAudio.onclick = () => {
      const start = timelineAudioStartForSegment(segment);
      if (!playSceneAudioFrom(start)) {
        activateGlobalTimelineAudioPlayback(Math.max(0, Number(segment.start || 0)));
        audio.play().then(updatePlayPauseButton).catch(() => startSilentTimelinePlayback(Math.max(0, Number(segment.start || 0))));
      }
    };
    jumpSceneCueAudio.onclick = () => {
      const start = Math.max(0, Number(segment.start || 0));
      activateGlobalTimelineAudioPlayback(start);
      toast(`Playhead moved to ${formatTime(start)}. Play the scene, then use Set Start / Set End on cue rows.`);
    };
    autoTimeSceneCues.onclick = () => autoTimeMiniMaxSpeakerCuesForSegment(segment);
    audioTools.append(playSceneCueAudio, jumpSceneCueAudio);
    if (hasCustomSpeakerTimingAudio) audioTools.append(autoTimeSceneCues);
    miniMaxSpeakerAssignmentList.append(audioTools);

    const cueActions = document.createElement("div");
    cueActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const addDialogueCue = makeButton("Add Dialogue Cue", "primary");
    const addInstrumentalCue = makeButton("Add Instrumental Cue");
    applyCompactButtonLabel(addDialogueCue, "Add\nDialogue", { noMap: true, padding: "7px 6px", title: "Add Dialogue Cue" });
    applyCompactButtonLabel(addInstrumentalCue, "Add\nInstrumental", { noMap: true, padding: "7px 6px", title: "Add Instrumental Cue" });
    addDialogueCue.onclick = () => {
      const speaker = speakers[cues.filter((cue) => cue.type !== "instrumental").length % speakers.length] || speakers[0] || {};
      const start = miniMaxNextCueStartTime(segment, cues);
      cues.push({ type: "dialogue", text: "", action_note: "", speaker_id: speaker.id || "", speaker_name: speaker.name || "", start, end: null });
      segment.minimax_speaker_assignments = cues;
      renderMiniMaxSpeakerAssignmentPanel();
      autoSaveSessionQuiet("MiniMax dialogue cue added").catch(() => null);
    };
    addInstrumentalCue.onclick = () => {
      const start = miniMaxNextCueStartTime(segment, cues);
      cues.push({ type: "instrumental", text: "", action_note: "", speaker_id: "", speaker_name: "", start, end: null });
      segment.minimax_speaker_assignments = cues;
      renderMiniMaxSpeakerAssignmentPanel();
      autoSaveSessionQuiet("MiniMax instrumental cue added").catch(() => null);
    };
    cueActions.append(addDialogueCue, addInstrumentalCue);
    miniMaxSpeakerAssignmentList.append(cueActions);

    let draggedIndex = -1;
    cues.forEach((cue, index) => {
      const row = document.createElement("div");
      row.style.cssText = "display:flex;flex-direction:column;gap:8px;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:8px;";
      row.addEventListener("dragover", (event) => {
        if (draggedIndex < 0 || draggedIndex === index) return;
        event.preventDefault();
        row.style.borderColor = "#22d3ee";
      });
      row.addEventListener("dragleave", () => { row.style.borderColor = "#334155"; });
      row.addEventListener("drop", (event) => {
        event.preventDefault();
        row.style.borderColor = "#334155";
        if (draggedIndex < 0 || draggedIndex === index) return;
        const [moved] = cues.splice(draggedIndex, 1);
        cues.splice(index, 0, moved);
        segment.minimax_speaker_assignments = cues;
        syncMiniMaxSpeakerAssignmentLegacyFields(segment);
        renderMiniMaxSpeakerAssignmentPanel();
        autoSaveSessionQuiet("MiniMax speaker cue reordered").catch(() => null);
      });
      const handle = document.createElement("button");
      handle.type = "button";
      handle.textContent = "::";
      handle.title = "Drag to change speaking order";
      handle.draggable = true;
      handle.style.cssText = "height:38px;border:1px solid #334155;border-radius:6px;background:#07111f;color:#67e8f9;font-weight:900;cursor:grab;";
      handle.addEventListener("dragstart", () => { draggedIndex = index; row.style.opacity = ".55"; });
      handle.addEventListener("dragend", () => { draggedIndex = -1; row.style.opacity = ""; });
      const number = document.createElement("div");
      number.textContent = String(index + 1);
      number.style.cssText = "font-size:14px;font-weight:900;color:#cffafe;text-align:center;";
      const mainRow = document.createElement("div");
      mainRow.style.cssText = "display:grid;grid-template-columns:32px 34px minmax(106px,.5fr) minmax(132px,.65fr) minmax(180px,1.4fr) 58px;gap:7px;align-items:center;";
      const typeSelect = makeSelect(["dialogue", "instrumental"], cue.type === "instrumental" ? "instrumental" : "dialogue");
      typeSelect.options[0].textContent = "Dialogue";
      typeSelect.options[1].textContent = "Instrumental";
      const speakerSelect = makeSelect(speakers.map((speaker) => ({ value: speaker.id, label: speaker.name })), cue.speaker_id);
      if (!speakers.some((speaker) => speaker.id === cue.speaker_id) && cue.speaker_name) {
        speakerSelect.prepend(new Option(`${cue.speaker_name} (not currently mapped)`, cue.speaker_id));
        speakerSelect.value = cue.speaker_id;
      }
      const line = makeInput(cue.type === "instrumental" ? cue.action_note || "" : cue.text || "");
      line.placeholder = cue.type === "instrumental" ? "Action note while nobody speaks..." : "Exact words this character says...";
      const remove = makeButton("Remove");
      applyCompactButtonLabel(remove, "Remove", { minWidth: 0, padding: "7px 6px" });
      typeSelect.addEventListener("change", () => {
        cue.type = typeSelect.value === "instrumental" ? "instrumental" : "dialogue";
        if (cue.type === "instrumental") {
          cue.action_note = cue.action_note || "";
          cue.text = "";
          cue.speaker_id = "";
          cue.speaker_name = "";
        } else {
          const speaker = speakers[0] || {};
          cue.text = cue.text || cue.action_note || "";
          cue.action_note = "";
          cue.speaker_id = cue.speaker_id || speaker.id || "";
          cue.speaker_name = cue.speaker_name || speaker.name || "";
        }
        segment.minimax_speaker_assignments = cues;
        syncMiniMaxSpeakerAssignmentLegacyFields(segment);
        renderMiniMaxSpeakerAssignmentPanel();
      });
      speakerSelect.addEventListener("change", () => {
        const speaker = speakers.find((item) => item.id === speakerSelect.value) || { id: speakerSelect.value, name: speakerSelect.selectedOptions[0]?.textContent || "" };
        cue.speaker_id = speaker.id;
        cue.speaker_name = speaker.name;
        segment.minimax_speaker_assignments = cues;
        syncMiniMaxSpeakerAssignmentLegacyFields(segment);
        autoSaveSessionQuiet("MiniMax speaker assignment changed").catch(() => null);
      });
      line.addEventListener("input", () => {
        if (cue.type === "instrumental") cue.action_note = line.value;
        else cue.text = line.value;
        segment.minimax_speaker_assignments = cues;
        syncMiniMaxSpeakerAssignmentLegacyFields(segment);
      });
      line.addEventListener("change", () => autoSaveSessionQuiet("MiniMax dialogue cue changed").catch(() => null));
      remove.onclick = () => {
        pushHistory();
        cues.splice(index, 1);
        segment.minimax_speaker_assignments = cues;
        syncMiniMaxSpeakerAssignmentLegacyFields(segment);
        renderMiniMaxSpeakerAssignmentPanel();
        autoSaveSessionQuiet("MiniMax dialogue cue removed").catch(() => null);
      };
      speakerSelect.disabled = cue.type === "instrumental";
      mainRow.append(handle, number, typeSelect, speakerSelect, line, remove);
      const timingRow = document.createElement("div");
      timingRow.style.cssText = "display:grid;grid-template-columns:minmax(44px,.45fr) minmax(44px,.45fr) 58px 64px 58px 58px 60px;gap:7px;align-items:center;padding-left:74px;";
      const cueRange = singerCuePlaybackRangeForCue(segment, cues, index);
      const effectiveEnd = miniMaxEffectiveCueEnd(segment, cues, index);
      const startLabel = document.createElement("div");
      startLabel.textContent = `Start: ${formatCueTime(cue.start)}`;
      startLabel.style.cssText = "font-size:11px;color:#a5f3fc;";
      const endLabel = document.createElement("div");
      endLabel.textContent = `End: ${formatCueTime(effectiveEnd)}`;
      endLabel.style.cssText = "font-size:11px;color:#a5f3fc;";
      const playCue = makeButton("Play Cue", "primary");
      const playFromCue = makeButton("Play From Here");
      const setStart = makeButton("Set Start");
      const setEnd = makeButton("Set End");
      const clearTiming = makeButton("Clear Timing");
      applyCompactButtonLabel(playCue, "Play\nCue", { noMap: true, minWidth: 0, padding: "6px 5px", title: "Play only this cue's timed scene range." });
      applyCompactButtonLabel(playFromCue, "From\nHere", { noMap: true, minWidth: 0, padding: "6px 5px", title: "Play from this cue start." });
      applyCompactButtonLabel(setStart, "Set\nStart", { noMap: true, minWidth: 0, padding: "6px 5px" });
      applyCompactButtonLabel(setEnd, "Set\nEnd", { noMap: true, minWidth: 0, padding: "6px 5px" });
      applyCompactButtonLabel(clearTiming, "Clear\nTime", { noMap: true, minWidth: 0, padding: "6px 5px", title: "Clear Timing" });
      playCue.onclick = () => playSingerCueRange(segment, cueRange.start, cueRange.end, playCue, "Play Cue");
      playFromCue.onclick = () => playSingerCueRange(segment, cueRange.start, null, playFromCue, "Play From Here");
      setStart.onclick = () => {
        cue.start = singerCueRelativePlayheadTime(segment);
        if (index > 0) cues[index - 1].end = cue.start;
        if (Number.isFinite(Number(cue.end)) && cue.end <= cue.start) cue.end = null;
        syncCueEndBoundariesFromNextStarts(cues);
        segment.minimax_speaker_assignments = cues;
        renderMiniMaxSpeakerAssignmentPanel();
        autoSaveSessionQuiet("MiniMax speaker cue timing changed").catch(() => null);
      };
      setEnd.onclick = () => {
        cue.end = singerCueRelativePlayheadTime(segment);
        if (Number.isFinite(Number(cue.start)) && cue.end <= cue.start) {
          toast("Cue end must be after cue start.", true);
          cue.end = null;
        } else if (index + 1 < cues.length) {
          cues[index + 1].start = cue.end;
          if (Number.isFinite(Number(cues[index + 1].end)) && cues[index + 1].end <= cues[index + 1].start) cues[index + 1].end = null;
        }
        syncCueEndBoundariesFromNextStarts(cues);
        segment.minimax_speaker_assignments = cues;
        renderMiniMaxSpeakerAssignmentPanel();
        autoSaveSessionQuiet("MiniMax speaker cue timing changed").catch(() => null);
      };
      clearTiming.onclick = () => {
        cue.start = null;
        cue.end = null;
        segment.minimax_speaker_assignments = cues;
        renderMiniMaxSpeakerAssignmentPanel();
        autoSaveSessionQuiet("MiniMax speaker cue timing cleared").catch(() => null);
      };
      timingRow.append(startLabel, endLabel, playCue, playFromCue, setStart, setEnd, clearTiming);
      row.append(mainRow, timingRow);
      miniMaxSpeakerAssignmentList.append(row);
    });
  }

  async function autoTimeMiniMaxSpeakerCuesForSegment(segment) {
    if (!segment || !isMiniMaxBuiltInSpeakerAssignmentMode(segment)) return;
    const progress = createProgressWindow("Auto Time Speaker Cues");
    try {
      const cues = ensureMiniMaxSpeakerAssignments(segment, miniMaxMappedSpeakersForSegment(segment));
      const dialogueCues = normalizeMiniMaxSpeakerAssignments(cues).filter((cue) => cue.type !== "instrumental" && flattenLyricForPrompt(cue.text));
      if (!dialogueCues.length) throw new Error("Add at least one dialogue cue before auto-timing this scene.");
      if (!String(segment.custom_audio_path || "").trim()) {
        throw new Error("Speaker Auto Time needs a custom audio file on this scene. MiniMax built-in audio does not exist yet before rendering, so enter dialogue timing manually or add custom audio for timing.");
      }
      progress.set("Preparing dialogue cue text...", 5);
      const referenceDialogue = dialogueCues.map((cue) => flattenLyricForPrompt(cue.text)).filter(Boolean).join("\n");
      const audioPath = await prepareSceneAudioClipForTimestamping(segment, progress);
      const payload = await runTimestampedCueWorkflow(audioPath, referenceDialogue, progress, timelineSegmentDuration(segment));
      const timestamped = timestampedCueSegments(payload);
      if (!timestamped.length) throw new Error("Stable-ts did not return usable cue timestamps for this scene.");
      progress.set("Applying timestamped speaker cue rows...", 88);
      const rebuilt = rebuildSpeakerCueMapFromTimestampedSegments(segment, cues, timestamped, { minInstrumentalGap: 0.5 });
      if (!rebuilt.length) throw new Error("No usable speaker cue rows were created from the timestamped result.");
      pushHistory();
      segment.minimax_speaker_assignments = rebuilt;
      syncMiniMaxSpeakerAssignmentLegacyFields(segment);
      renderMiniMaxSpeakerAssignmentPanel();
      syncInspector();
      render();
      await autoSaveSessionQuiet("MiniMax speaker cues auto-timed");
      const dialogueReturned = timestamped.filter((item) => item.type !== "instrumental").length;
      const instrumentalReturned = rebuilt.filter((item) => item.type === "instrumental").length;
      const warning = dialogueReturned !== dialogueCues.length
        ? `\nWarning: Stable-ts returned ${dialogueReturned} dialogue cue${dialogueReturned === 1 ? "" : "s"} for ${dialogueCues.length} mapped dialogue cue${dialogueCues.length === 1 ? "" : "s"}. Review before rendering.`
        : "";
      progress.set(`Auto timing complete.\nDialogue cues: ${dialogueReturned}\nInstrumental gaps: ${instrumentalReturned}${warning}`, 100);
      progress.close(2600);
      toast("Speaker cue timing filled. Review with Play Cue before rendering.");
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      progress.close(7000);
      toast(String(error?.message || error), true);
    }
  }

  return {
    ensureMiniMaxSpeakerAssignments, isMiniMaxBuiltInSpeakerAssignmentMode, isMiniMaxSingerAssignmentMode,
    miniMaxMappedSpeakersForSegment, renderMiniMaxSpeakerAssignmentPanel,
  };
}
