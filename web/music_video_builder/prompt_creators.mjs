import { postJson } from "./comfy_api.mjs";
import { makeButton, makeCheckbox, makeField, makeInput, makeSelect, toast } from "./controls.mjs";
import { loadContextTextQuiet } from "./project_files.mjs";
import { isInstrumentalLyricText, syncConceptPromptToStoryBeat } from "./prompt_text.mjs";
import { normalizeTimelineMarkers } from "./timeline_state.mjs";

export function createPromptCreators({
  activeSegment, autoSaveSessionQuiet, clearFinalPromptList, countPromptFindReplaceMatches,
  createProgressWindow, currentVideoMode, editFinalPromptList, ernieNotesInput,
  fluxReferenceContextForSegment, i2vNotesInput, krea2TwoPassNotesInput, nbNotes, notesInput, pushHistory,
  reloadFinalPromptList, render, replacePromptPhraseAcrossScenes, sceneDisplayName, selectedTimelineRangeInfo,
  state, storyIdeaInput, syncI2VMotionJsonFromSegments, syncPromptJsonFromSegments, t2iTextGemmaModelSelect,
  textGemmaRunnerPayload, themeStyleInput, transcribeLyricsForTimeline, updateActiveFromInputs,
}) {
  function conceptPromptTimelineNotesForSegment(segment) {
    const start = Number(segment?.start || 0);
    const end = Number(segment?.end || start);
    const duration = Math.max(0.01, end - start);
    return normalizeTimelineMarkers(state.timelineMarkers)
      .filter((marker) => {
        const markerStart = Number(marker.start || 0);
        const markerEnd = Number(marker.end || markerStart);
        const overlap = Math.max(0, Math.min(end, markerEnd) - Math.max(start, markerStart));
        return overlap >= 0.25 || overlap / duration >= 0.1;
      })
      .slice(0, 8)
      .map((marker) => ({
        type: String(marker.type || "note"),
        label: String(marker.label || ""),
        note: String(marker.note || ""),
        start: Number(marker.start || 0),
        end: Number(marker.end || marker.start || 0),
      }));
  }

  function conceptPromptReferenceModes(segment) {
    const context = fluxReferenceContextForSegment(segment);
    return {
      subject_reference_mode: context.has_subject_reference ? "subject reference available" : "no subject reference",
      location_reference_mode: context.has_location_reference ? "location reference available" : "no location reference",
    };
  }

  function conceptPromptScenePayload(segment, index, sourceMode = "all") {
    const includeDirector = ["director", "director_scene", "all"].includes(sourceMode);
    const includeScene = ["scene", "director_scene", "all"].includes(sourceMode);
    const includeTimeline = ["timeline", "all"].includes(sourceMode);
    const includeLyrics = ["lyrics", "lyrics_subjects", "all"].includes(sourceMode);
    const includeSubjects = ["subjects", "lyrics_subjects", "all"].includes(sourceMode);
    const refContext = conceptPromptReferenceModes(segment);
    const referenceContext = fluxReferenceContextForSegment(segment);
    const lyricSingers = segment.no_character_present ? [] : Array.isArray(segment.lyric_singers)
      ? segment.lyric_singers.map((item) => String(item || "").trim()).filter(Boolean)
      : [];
    const mappedSubjectText = includeSubjects ? String(referenceContext.subject_description || "").trim() : "";
    return {
      scene_number: index + 1,
      label: sceneDisplayName(segment, index),
      lyric_text: includeLyrics ? String(segment.lyric_text || "").trim() : "",
      lyric_singers: includeSubjects ? lyricSingers : [],
      lyric_instrumental: includeLyrics ? isInstrumentalLyricText(segment.lyric_text) : false,
      lyric_no_lip_sync: includeLyrics ? Boolean(segment.lyric_no_lip_sync) : false,
      no_character_present: Boolean(segment.no_character_present),
      mapped_subjects: mappedSubjectText,
      director_note: includeDirector ? String(segment.timeline_note || "").trim() : "",
      scene_note: includeScene ? String(segment.notes || "").trim() : "",
      timeline_notes: includeTimeline ? conceptPromptTimelineNotesForSegment(segment) : [],
      ...refContext,
    };
  }

  function conceptPromptTargets(scope = "all") {
    const baseScenes = state.segments.map((segment, index) => ({ segment, index }));
    if (scope === "selected") {
      const active = activeSegment();
      const activeIndex = state.segments.findIndex((segment) => segment.id === active?.id);
      return active && activeIndex >= 0 ? [{ segment: active, index: activeIndex }] : [];
    }
    if (scope === "range") {
      const range = selectedTimelineRangeInfo();
      if (!range) return [];
      return baseScenes.filter(({ segment }) => {
        const overlap = Math.max(0, Math.min(Number(segment.end || 0), range.end) - Math.max(Number(segment.start || 0), range.start));
        return overlap >= 0.25;
      });
    }
    return baseScenes;
  }

  async function runConceptPromptCreator(options = {}) {
    updateActiveFromInputs();
    const sourceMode = String(options.sourceMode || "all").trim() || "all";
    const scope = String(options.scope || "all").trim() || "all";
    const batchSize = Math.max(1, Math.min(10, Math.floor(Number(options.batchSize || 5))));
    const targets = conceptPromptTargets(scope);
    if (!targets.length) {
      toast(scope === "range" ? "No scenes overlap the selected timeline range." : "No scenes found for concept prompt creation.", true);
      return { updated: 0, skipped: ["no scenes"] };
    }
    const progress = options.progress || createProgressWindow("Concept Prompt Creator");
    const closeProgress = !options.progress;
    const useStory = options.useStory !== false;
    const useTheme = options.useTheme !== false;
    const storyIdea = useStory ? await loadContextTextQuiet(storyIdeaInput.value || state.storyIdeaPath) : "";
    const themeStyle = useTheme ? await loadContextTextQuiet(themeStyleInput.value || state.themeStylePath) : "";
    let previousSummary = "";
    let updated = 0;
    const appliedKeys = [];
    try {
      pushHistory();
      for (let offset = 0; offset < targets.length; offset += batchSize) {
        if (state.batchCancelled) throw new Error("Stopped by user.");
        const batch = targets.slice(offset, offset + batchSize);
        progress.set(`Creating concept prompts for scenes ${batch[0].index + 1}-${batch[batch.length - 1].index + 1}...`, 10 + Math.round((offset / targets.length) * 80));
        const data = await postJson("/vrgdg/music_builder/generate_concept_prompts", {
          ...textGemmaRunnerPayload(),
          model_file: t2iTextGemmaModelSelect.value,
          source_mode: sourceMode,
          story_idea: storyIdea,
          theme_style: themeStyle,
          previous_summary: previousSummary,
          scenes: batch.map(({ segment, index }) => conceptPromptScenePayload(segment, index, sourceMode)),
          temperature: options.temperature ?? 0.45,
          top_p: options.topP ?? 0.95,
          max_new_tokens: options.maxNewTokens ?? 1200,
        }, 180000);
        const prompts = data.prompts || {};
        for (const { segment, index } of batch) {
          const promptText = String(prompts[`Scene${index + 1}`] || prompts[`Scene ${index + 1}`] || "").trim();
          if (!promptText) continue;
          segment.notes = promptText;
          segment.flux_notes = promptText;
          segment.nb_notes = promptText;
          syncConceptPromptToStoryBeat(segment, promptText);
          if (segment.id === activeSegment()?.id) {
            notesInput.value = promptText;
            ernieNotesInput.value = promptText;
            krea2TwoPassNotesInput.value = promptText;
            nbNotes.value = promptText;
          }
          appliedKeys.push(`Scene${index + 1}`);
          updated += 1;
        }
        previousSummary = String(data.summary || previousSummary || "").slice(0, 1200);
        render();
      }
      await syncPromptJsonFromSegments("Concept Prompt Creator").catch(() => null);
      await autoSaveSessionQuiet("Concept Prompt Creator").catch(() => null);
      progress.set(`Updated ${updated} scene note${updated === 1 ? "" : "s"} and story beat${updated === 1 ? "" : "s"} with generated concept prompts.`, 100);
      if (closeProgress) progress.close(1200);
      toast(`Concept prompts updated: ${updated}`);
      return { updated, applied: appliedKeys };
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
      if (options.throwOnError) throw error;
      return { updated, error: String(error?.message || error) };
    }
  }

  function openConceptPromptCreatorModal() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100007;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:18px;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,calc(100vw - 36px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Concept Prompt Creator";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const close = makeButton("Close");
    header.append(heading, close);
    const note = document.createElement("div");
    note.textContent = "Creates batched concept prompts from notes and writes them into the scene notes fields.";
    note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    const source = makeSelect(["all", "director", "scene", "timeline", "lyrics", "subjects", "lyrics_subjects", "director_scene"], "all");
    source.options[0].textContent = "All available notes";
    source.options[1].textContent = "Director notes";
    source.options[2].textContent = "Scene notes";
    source.options[3].textContent = "Timeline notes";
    source.options[4].textContent = "Line notes";
    source.options[5].textContent = "Subject / performer mapping";
    source.options[6].textContent = "Line notes + subjects/performers";
    source.options[7].textContent = "Director + scene notes";
    const scope = makeSelect(["all", "selected", "range"], "all");
    scope.options[0].textContent = "All scenes";
    scope.options[1].textContent = "Selected scene";
    scope.options[2].textContent = "Selected timeline range";
    const batchSize = makeInput("5");
    batchSize.type = "number";
    batchSize.min = "1";
    batchSize.max = "10";
    const useStory = makeCheckbox("Use Story idea", true);
    const useTheme = makeCheckbox("Use Theme/style", true);
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    grid.append(makeField("Source", source), makeField("Scope", scope), makeField("Scenes per batch", batchSize), useStory.wrapper, useTheme.wrapper);
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const run = makeButton("Create Concept Prompts", "primary");
    actions.append(cancel, run);
    box.append(header, note, grid, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    const dismiss = () => backdrop.remove();
    close.onclick = dismiss;
    cancel.onclick = dismiss;
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) dismiss();
    });
    run.onclick = async () => {
      dismiss();
      await runConceptPromptCreator({
        sourceMode: source.value,
        scope: scope.value,
        batchSize: batchSize.value,
        useStory: useStory.input.checked,
        useTheme: useTheme.input.checked,
      });
    };
  }

  function motionNoteScenePayload(segment, index, sourceMode = "concept") {
    const includeDirector = ["concept_director", "all"].includes(sourceMode);
    const includeTimeline = ["concept_timeline", "all"].includes(sourceMode);
    return {
      scene_number: index + 1,
      label: sceneDisplayName(segment, index),
      concept_prompt: String(segment.notes || "").trim(),
      director_note: includeDirector ? String(segment.timeline_note || "").trim() : "",
      timeline_notes: includeTimeline ? conceptPromptTimelineNotesForSegment(segment) : [],
    };
  }

  function validateMotionNoteInputs(targets, sourceMode) {
    const needsConcept = true;
    const needsDirector = ["concept_director", "all"].includes(sourceMode);
    const needsTimeline = ["concept_timeline", "all"].includes(sourceMode);
    if (needsConcept && !targets.some(({ segment }) => String(segment.notes || "").trim())) {
      return "You do not have any concept prompts in the selected scenes yet. Create/update concept prompts first, then try again.";
    }
    if (needsDirector && !targets.some(({ segment }) => String(segment.timeline_note || "").trim())) {
      return "You chose director notes, but the selected scenes do not have director notes. Add/update director notes, then try again.";
    }
    if (needsTimeline && !targets.some(({ segment }) => conceptPromptTimelineNotesForSegment(segment).length)) {
      return "You chose timeline notes, but no timeline notes overlap the selected scenes. Add/update timeline notes, then try again.";
    }
    return "";
  }

  async function runMotionNoteCreator(options = {}) {
    updateActiveFromInputs();
    const sourceMode = String(options.sourceMode || "concept").trim() || "concept";
    const scope = String(options.scope || "all").trim() || "all";
    const batchSize = Math.max(1, Math.min(10, Math.floor(Number(options.batchSize || 5))));
    const targets = conceptPromptTargets(scope);
    if (!targets.length) {
      toast(scope === "range" ? "No scenes overlap the selected timeline range." : "No scenes found for motion note creation.", true);
      return { updated: 0, skipped: ["no scenes"] };
    }
    const validation = validateMotionNoteInputs(targets, sourceMode);
    if (validation) {
      toast(validation, true);
      return { updated: 0, error: validation };
    }
    const progress = options.progress || createProgressWindow("Motion Note Creator");
    const closeProgress = !options.progress;
    const useStory = options.useStory !== false;
    const useTheme = options.useTheme !== false;
    const storyIdea = useStory ? await loadContextTextQuiet(storyIdeaInput.value || state.storyIdeaPath) : "";
    const themeStyle = useTheme ? await loadContextTextQuiet(themeStyleInput.value || state.themeStylePath) : "";
    let previousSummary = "";
    let updated = 0;
    const appliedKeys = [];
    try {
      pushHistory();
      for (let offset = 0; offset < targets.length; offset += batchSize) {
        if (state.batchCancelled) throw new Error("Stopped by user.");
        const batch = targets.slice(offset, offset + batchSize);
        progress.set(`Creating motion notes for scenes ${batch[0].index + 1}-${batch[batch.length - 1].index + 1}...`, 10 + Math.round((offset / targets.length) * 80));
        const data = await postJson("/vrgdg/music_builder/generate_motion_notes", {
          ...textGemmaRunnerPayload(),
          model_file: t2iTextGemmaModelSelect.value,
          source_mode: sourceMode,
          story_idea: storyIdea,
          theme_style: themeStyle,
          previous_summary: previousSummary,
          video_mode: currentVideoMode(),
          scenes: batch.map(({ segment, index }) => motionNoteScenePayload(segment, index, sourceMode)),
          temperature: options.temperature ?? 0.45,
          top_p: options.topP ?? 0.95,
          max_new_tokens: options.maxNewTokens ?? 1200,
        }, 180000);
        const notes = data.notes || {};
        for (const { segment, index } of batch) {
          const noteText = String(notes[`Motion${index + 1}`] || notes[`Motion ${index + 1}`] || notes[`Scene${index + 1}`] || "").trim();
          if (!noteText) continue;
          segment.i2v_notes = noteText;
          if (segment.id === activeSegment()?.id) i2vNotesInput.value = noteText;
          appliedKeys.push(`Motion${index + 1}`);
          updated += 1;
        }
        previousSummary = String(data.summary || previousSummary || "").slice(0, 1200);
        render();
      }
      await syncI2VMotionJsonFromSegments("Motion Note Creator").catch(() => null);
      await autoSaveSessionQuiet("Motion Note Creator").catch(() => null);
      progress.set(`Updated ${updated} I2V/T2V motion note${updated === 1 ? "" : "s"}.`, 100);
      if (closeProgress) progress.close(1200);
      toast(`Motion notes updated: ${updated}`);
      return { updated, applied: appliedKeys };
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
      return { updated, error: String(error?.message || error) };
    }
  }

  function openMotionNoteCreatorModal() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100007;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:18px;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,calc(100vw - 36px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Motion Note Creator";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const close = makeButton("Close");
    header.append(heading, close);
    const note = document.createElement("div");
    note.textContent = "Creates batched I2V/T2V motion notes and writes them into the I2V motion notes fields/file.";
    note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    const source = makeSelect(["concept", "concept_director", "concept_timeline", "all"], "concept");
    source.options[0].textContent = "Concept prompts only";
    source.options[1].textContent = "Concept prompts + director notes";
    source.options[2].textContent = "Concept prompts + timeline notes";
    source.options[3].textContent = "All three";
    const scope = makeSelect(["all", "selected", "range"], "all");
    scope.options[0].textContent = "All scenes";
    scope.options[1].textContent = "Selected scene";
    scope.options[2].textContent = "Selected timeline range";
    const batchSize = makeInput("5");
    batchSize.type = "number";
    batchSize.min = "1";
    batchSize.max = "10";
    const useStory = makeCheckbox("Use Story idea", true);
    const useTheme = makeCheckbox("Use Theme/style", true);
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    grid.append(makeField("Source", source), makeField("Scope", scope), makeField("Scenes per batch", batchSize), useStory.wrapper, useTheme.wrapper);
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const run = makeButton("Create Motion Notes", "primary");
    actions.append(cancel, run);
    box.append(header, note, grid, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    const dismiss = () => backdrop.remove();
    close.onclick = dismiss;
    cancel.onclick = dismiss;
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) dismiss();
    });
    run.onclick = async () => {
      dismiss();
      await runMotionNoteCreator({
        sourceMode: source.value,
        scope: scope.value,
        batchSize: batchSize.value,
        useStory: useStory.input.checked,
        useTheme: useTheme.input.checked,
      });
    };
  }

  function openPromptOptionsModal() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Prompt Options";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const close = makeButton("Close");
    header.append(heading, close);
    const note = document.createElement("div");
    note.textContent = "Edit or reload the final generated prompt text files for this project. Reload updates the scene boxes immediately and saves the session.";
    note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    const transcribeLyrics = makeButton("Transcribe Lines For Timeline", "primary");
    transcribeLyrics.title = "Use the current audio and builder SRT timing to fill each scene's Line / lyric / dialogue field.";
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;";
    const imageGroup = document.createElement("div");
    imageGroup.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;display:flex;flex-direction:column;gap:8px;";
    const videoGroup = document.createElement("div");
    videoGroup.style.cssText = imageGroup.style.cssText;
    const imageHeading = document.createElement("div");
    imageHeading.textContent = "Image";
    imageHeading.style.cssText = "font-size:13px;font-weight:900;color:#cffafe;";
    const videoHeading = document.createElement("div");
    videoHeading.textContent = "Video";
    videoHeading.style.cssText = imageHeading.style.cssText;
    const editT2I = makeButton("Edit Text to Image Prompts", "primary");
    const createConceptPrompts = makeButton("Create Concept Prompts", "primary");
    const reloadT2I = makeButton("Reload Text to Image Prompts");
    const originalT2I = makeButton("Reload Original T2I Prompts");
    const clearT2I = makeButton("Clear All T2I Prompts");
    clearT2I.style.borderColor = "#7f1d1d";
    clearT2I.style.color = "#fecaca";
    const editI2V = makeButton("Edit Image to Video Prompts", "primary");
    const createMotionNotes = makeButton("Create Motion Notes", "primary");
    const reloadI2V = makeButton("Reload Image to Video Prompts");
    const originalI2V = makeButton("Reload Original I2V Prompts");
    const clearI2V = makeButton("Clear All I2V Prompts");
    const editMiniMax = makeButton("Edit MiniMax H3 Prompts", "primary");
    const reloadMiniMax = makeButton("Reload MiniMax H3 Prompts");
    const promptValidationGroup = document.createElement("div");
    promptValidationGroup.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;display:flex;flex-direction:column;gap:7px;";
    const promptValidationHeading = document.createElement("div");
    promptValidationHeading.textContent = "MiniMax H3 prompt validation";
    promptValidationHeading.style.cssText = "font-size:13px;font-weight:900;color:#cffafe;";
    const failOnInvalidPromptFormats = makeCheckbox("Fail on invalid prompt formats (testing)", Boolean(state.failOnInvalidPromptFormats));
    const promptValidationNote = document.createElement("div");
    promptValidationNote.textContent = "Off by default: externally authored MiniMax prompts may use any format, but prompts over 7,000 characters still fail. Turn this on to enforce the builder's strict format checks.";
    promptValidationNote.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    promptValidationGroup.append(promptValidationHeading, failOnInvalidPromptFormats.wrapper, promptValidationNote);
    clearI2V.style.borderColor = "#7f1d1d";
    clearI2V.style.color = "#fecaca";
    imageGroup.append(imageHeading, createConceptPrompts, editT2I, reloadT2I, originalT2I, clearT2I);
    videoGroup.append(videoHeading, createMotionNotes, editI2V, reloadI2V, originalI2V, clearI2V, editMiniMax, reloadMiniMax);
    grid.append(imageGroup, videoGroup);
    const replaceGroup = document.createElement("div");
    replaceGroup.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;display:flex;flex-direction:column;gap:9px;";
    const replaceHeading = document.createElement("div");
    replaceHeading.textContent = "Find / Replace Prompts";
    replaceHeading.style.cssText = "font-size:13px;font-weight:900;color:#cffafe;";
    const replaceNote = document.createElement("div");
    replaceNote.textContent = "Finds the full phrase only, such as replacing \"the woman\" with a character LoRA trigger word across saved prompts.";
    replaceNote.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    const replaceFields = document.createElement("div");
    replaceFields.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const findInput = makeInput("");
    findInput.placeholder = "Find exact phrase, e.g. the woman";
    const replaceInput = makeInput("");
    replaceInput.placeholder = "Replace with, e.g. f4nt4syw0m3n";
    replaceFields.append(makeField("Find", findInput), makeField("Replace with", replaceInput));
    const replaceChecks = document.createElement("div");
    replaceChecks.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:8px;align-items:center;";
    const includeT2I = makeCheckbox("Text-to-image prompts", true);
    const includeI2V = makeCheckbox("I2V / T2V video prompts", true);
    const includeMiniMax = makeCheckbox("MiniMax H3 prompts", true);
    const caseSensitive = makeCheckbox("Case sensitive", false);
    replaceChecks.append(includeT2I.wrapper, includeI2V.wrapper, includeMiniMax.wrapper, caseSensitive.wrapper);
    const replaceStatus = document.createElement("div");
    replaceStatus.style.cssText = "min-height:18px;font-size:12px;color:#94a3b8;";
    const replaceActions = document.createElement("div");
    replaceActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const previewReplace = makeButton("Preview Matches", "primary");
    const applyReplace = makeButton("Replace All");
    replaceActions.append(previewReplace, applyReplace);
    replaceGroup.append(replaceHeading, replaceNote, replaceFields, replaceChecks, replaceStatus, replaceActions);
    box.append(header, note, transcribeLyrics, grid, promptValidationGroup, replaceGroup);
    backdrop.append(box);
    document.body.append(backdrop);
    const run = (action) => {
      backdrop.remove();
      action();
    };
    close.onclick = () => backdrop.remove();
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
    if (state.imageModelMode === "nano_banana") {
      editT2I.textContent = "Edit Nano B Prompts";
      reloadT2I.textContent = "Reload Nano B Prompts";
      clearT2I.textContent = "Clear All Nano B Prompts";
    } else if (state.imageModelMode === "flux_klein") {
      editT2I.textContent = "Edit Flux/Klein Prompts";
      reloadT2I.textContent = "Reload Flux/Klein Prompts";
      clearT2I.textContent = "Clear All Flux/Klein Prompts";
    }
    createConceptPrompts.onclick = () => run(() => openConceptPromptCreatorModal());
    createMotionNotes.onclick = () => run(() => openMotionNoteCreatorModal());
    editT2I.onclick = () => run(() => editFinalPromptList("t2i"));
    transcribeLyrics.onclick = () => run(() => transcribeLyricsForTimeline());
    reloadT2I.onclick = () => run(() => reloadFinalPromptList("t2i", false));
    originalT2I.onclick = () => run(() => reloadFinalPromptList("t2i", true));
    clearT2I.onclick = () => run(() => clearFinalPromptList("t2i"));
    editI2V.onclick = () => run(() => editFinalPromptList("i2v"));
    reloadI2V.onclick = () => run(() => reloadFinalPromptList("i2v", false));
    originalI2V.onclick = () => run(() => reloadFinalPromptList("i2v", true));
    clearI2V.onclick = () => run(() => clearFinalPromptList("i2v"));
    editMiniMax.onclick = () => run(() => editFinalPromptList("minimax"));
    reloadMiniMax.onclick = () => run(() => reloadFinalPromptList("minimax", false));
    failOnInvalidPromptFormats.input.addEventListener("change", () => {
      state.failOnInvalidPromptFormats = Boolean(failOnInvalidPromptFormats.input.checked);
      void autoSaveSessionQuiet("MiniMax prompt validation setting changed");
    });
    const selectedReplaceKinds = () => [
      includeT2I.input.checked ? "t2i" : "",
      includeI2V.input.checked ? "i2v" : "",
      includeMiniMax.input.checked ? "minimax" : "",
    ].filter(Boolean);
    const previewPromptReplace = () => {
      const findText = String(findInput.value || "").trim();
      if (!findText) {
        replaceStatus.textContent = "Enter a phrase to find first.";
        replaceStatus.style.color = "#fbbf24";
        return null;
      }
      const targetKinds = selectedReplaceKinds();
      if (!targetKinds.length) {
        replaceStatus.textContent = "Choose at least one prompt type.";
        replaceStatus.style.color = "#fbbf24";
        return null;
      }
      const result = countPromptFindReplaceMatches(findText, targetKinds, { caseSensitive: caseSensitive.input.checked });
      replaceStatus.textContent = `Found ${result.matches} match${result.matches === 1 ? "" : "es"} in ${result.fields} prompt field${result.fields === 1 ? "" : "s"} across ${result.scenes} scene${result.scenes === 1 ? "" : "s"}.`;
      replaceStatus.style.color = result.matches ? "#67e8f9" : "#fbbf24";
      return { ...result, targetKinds };
    };
    previewReplace.onclick = previewPromptReplace;
    applyReplace.onclick = async () => {
      const preview = previewPromptReplace();
      if (!preview || !preview.matches) return;
      const findText = String(findInput.value || "").trim();
      const replaceText = String(replaceInput.value || "");
      if (!window.confirm(`Replace ${preview.matches} exact phrase match${preview.matches === 1 ? "" : "es"} for "${findText}"?`)) return;
      const result = await replacePromptPhraseAcrossScenes(findText, replaceText, preview.targetKinds, { caseSensitive: caseSensitive.input.checked });
      replaceStatus.textContent = `Replaced ${result.matches} match${result.matches === 1 ? "" : "es"} across ${result.scenes} scene${result.scenes === 1 ? "" : "s"}.`;
      replaceStatus.style.color = "#67e8f9";
      toast(`Prompt find/replace complete. Replaced ${result.matches} match${result.matches === 1 ? "" : "es"}.`);
    };
  }

  return { openPromptOptionsModal, runConceptPromptCreator, runMotionNoteCreator };
}
