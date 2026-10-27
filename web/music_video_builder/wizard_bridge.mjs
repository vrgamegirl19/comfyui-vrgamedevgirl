import { organizeWizardVideoModels } from "./wizard_models_layout.mjs";
import { storyboardGptPayload } from "../storyboard_builder/gpt_payload.mjs";

import { openWizard, wizardNeeds } from "../VRGDG_Wizard.js";

import { GEMMA_VIDEO_PROMPT_TIMEOUT_MS, makeEditorImageUrl, postJson } from "./comfy_api.mjs";
import {
  DEFAULT_I2V_DIFFUSION_MODEL,
  DEFAULT_I2V_UNET,
  REQUIRED_LTX_MSR_LORA,
  REQUIRED_LTX25_MSR_LORA,
} from "./constants.mjs";
import {
  makeButton,
  makeSubTabs,
  normalizeProjectVideoEngine,
  normalizeProjectVideoType,
  normalizeVideoType,
  toast,
  VIDEO_TYPE_OPTIONS,
} from "./controls.mjs";
import { gemmaBatchFailureStore, recordGemmaBatchFailure, showGemmaBatchFailures } from "./dialogs.mjs";

import {
  cloneMiniMaxH3Settings,
  MINIMAX_H3_AUDIO_MODE_OPTIONS,
  MINIMAX_H3_MODE_OPTIONS,
} from "./minimax_h3.mjs";
import { cloneI2VVideoSettings, normalizeBuilderStoryLayer } from "./model_settings.mjs";
import { chooseBatchModeAction } from "./project_actions.mjs";
import {
  flattenLyricForPrompt,
  isRecoverableBuildGemmaError,
  normalizeGemmaContextLimit,
  normalizeGemmaGpuDevice,
  normalizeGemmaGpuLayers,
} from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder, normalizeLyricMapper } from "./reference_data.mjs";

export function createWizardBridge({
  activeProjectFolderForSave, activeSegment, allEditableSegments, audioInput, autoSaveSessionQuiet,
  chooseProjectAudioFile, confirmAndRunZImageAll, createProgressWindow, createSceneVideoActions,
  createSceneVideoButtons, createScenesFromTimestampedLyrics, createSilentTimelineAudioForDuration,
  currentVideoMode, ensureAllSegmentRuntimeFields, ensureAutoTimedSingerCuesBeforePrompt,
  ernieImagePanel, finalizeVideoPromptDraftOnly, flowGptModePanel, fluxKleinPanel,
  gemmaModelSelect, i2vGemmaModelSelect, i2vMmprojSelect, i2vTextGemmaModelSelect,
  imageModeDisplayLabel, importTimelineImagesFromFolder, inspector, krea2TwoPassPanel,
  miniMaxModePanels, miniMaxPassChooser, miniMaxSubTabs, mmprojSelect, nbImagePanel, newProject,
  normalizeLyricCueMapForSegment, openBulkSegmentsModal, openFluxReferenceBuilderModal,
  openGemmaRunnerModal, openIdLoraReferenceBuilderModalSafely,
  openIngredientsReferenceBuilderModal, openLyricMappingWorkflowModal, openLyricReviewModal,
  openStoryboardBuilderFromProject, projectInput, promptRunnerActionName, pushHistory, render,
  renderAllScenes, rtvSceneImageAnchorSection, saveErnieImageSettingsFromPanel,
  saveFlowGptBrowserSettingsFromPanel, saveFluxKleinSettingsFromPanel, saveGemmaJunkDebug,
  saveI2VVideoSettingsFromPanel, saveKrea2TwoPassSettingsFromPanel, saveMiniMaxH3SettingsFromPanel,
  saveNBImageSettingsFromPanel, saveSession, saveZImageSettingsFromPanel, sceneDisplayName,
  segmentTrack, selectedPerformerSubjectsForSegment, setInspectorTab, setSegmentPromptForEdit,
  state, storyboardPipeline, storyboardScenePayload, syncErnieImagePanel, syncFlowGptBrowserPanel,
  syncFluxKleinPanel, syncI2VVideoSettingsPanel, syncInspector, syncKrea2TwoPassPanel,
  syncMiniMaxH3Panel, syncNBImagePanel, syncProjectVideoEngineUI, syncVideoModePanel,
  syncVideoTypeControl, syncZImageSettingsPanel, t2iTextGemmaModelSelect, textGemmaRunnerPayload,
  transcribeLyricsForTimeline, updateActiveFromInputs, useSceneI2VVideoSettings,
  useSceneI2VVideoSettingsNote, useSceneMiniMaxH3Settings, useSceneMiniMaxH3SettingsNote,
  videoModeDisplayLabel, videoSubTabs, wizardStoryboardState, wizardVideoSettings,
  zimageSettingsPanel,
}) {
  function setBuilderLtxVersion(selectedVersion) {
      const versionDefaults = selectedVersion === "2.5" ? {
        use_gguf_model: false,
        diffusion_model_name: "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        vae_name: "ltx-2.5-video-vae-conv-bf16.safetensors",
        audio_vae_name: "ltx-2.5-audio-vae-bf16.safetensors",
        clip_name1: "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        upscale_model_name: "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        msr_lora_name: REQUIRED_LTX25_MSR_LORA,
      } : {
        use_gguf_model: true,
        unet_name: DEFAULT_I2V_UNET,
        diffusion_model_name: DEFAULT_I2V_DIFFUSION_MODEL,
        vae_name: "LTX23_video_vae_bf16.safetensors",
        audio_vae_name: "LTX23_audio_vae_bf16.safetensors",
        clip_name1: "gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors",
        clip_name2: "ltx-2.3_text_projection_bf16.safetensors",
        upscale_model_name: "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        msr_lora_name: REQUIRED_LTX_MSR_LORA,
      };
      state.i2vVideoSettings = cloneI2VVideoSettings({
        ...(state.i2vVideoSettings || {}),
        ...versionDefaults,
        ltx_version: selectedVersion,
      });
      for (const segment of allEditableSegments()) {
        if (!segment?.i2v_video_settings) continue;
        segment.i2v_video_settings = cloneI2VVideoSettings({
          ...segment.i2v_video_settings,
          ...versionDefaults,
          ltx_version: selectedVersion,
        });
      }
      syncI2VVideoSettingsPanel();
  }

  // Scene beats and video prompts for every scene. Used by the wizard and the Story Builder.
  const enforceWizardStoryboardVideoFacialRequirements = (prompt, scene = {}) => {
    let text = String(prompt || "").trim();
    const promptMentionsFace = /\b(?:woman|man|girl|boy|person|subject|singer|rapper|performer|speaker|character|face|eyes?|brows?|gaze|mouth|jaw|cheeks?|expression|smile|frown|sings?|singing|says|speaks?)\b/i.test(text);
    const hasCharacter = !scene.no_character_present && !scene.noCharacterPresent && (
      (Array.isArray(scene.subject_refs) && scene.subject_refs.length)
      || (Array.isArray(scene.subjects) && scene.subjects.length)
      || (Array.isArray(scene.visible_subjects) && scene.visible_subjects.length)
      || promptMentionsFace
    );
    if (!text || !hasCharacter) return text;
    const vocalStatus = scene.vocal_status || {};
    const promptSaysSinging = /\b(?:sings?|singing|raps?|rapping)\b/i.test(text);
    const isSinging = promptSaysSinging || (String(scene.performance_mode || vocalStatus.performance_mode || normalizeVideoType(state.videoType)).trim() === "singing"
      && vocalStatus.should_lip_sync !== false
      && !vocalStatus.instrumental
      && !vocalStatus.no_lip_sync
      && !scene.lyric_no_lip_sync
      && Boolean(String(vocalStatus.lyric_text || scene.lyrics || scene.lyric_text || "").trim()));
    if (isSinging) {
      text = text
        .replace(/\bwith\s+a\s+quiet,\s*internal\s+intensity\b/gi, "with controlled internal intensity")
        .replace(/\bwith\s+quiet\s+internal\s+intensity\b/gi, "with controlled internal intensity")
        .replace(/\bquiet,\s*internal\s+intensity\b/gi, "controlled internal intensity")
        .replace(/\bquiet\s+internal\s+intensity\b/gi, "controlled internal intensity")
        .replace(/\bquiet\s+intensity\b/gi, "controlled intensity")
        .replace(/\bquiet\s+performance\b/gi, "controlled performance")
        .replace(/\bquiet\s+emotion\b/gi, "restrained emotion")
        .replace(/\bquiet\s+singing\b/gi, "focused singing");
    }
    const hasBlink = /\bblink\w*\b/i.test(text);
    const hasEyeMovement = /\beye\s+movement\b|\beyes?\s+(?:shift|move|track|glance|flick|dart)\b/i.test(text);
    const additions = [];
    if (!hasEyeMovement) additions.push("subtle natural eye movement");
    if (!hasBlink) additions.push("occasional natural blinking");
    if (additions.length) {
      const insert = `, ${additions.join(", ")}`;
      const faceSentence = text.match(/([^.]*(?:face|eyes?|brows?|gaze|expression)[^.]*)(\.)/i);
      if (faceSentence && typeof faceSentence.index === "number") {
        const nextSentence = `${faceSentence[1].trimEnd()}${insert}`;
        text = `${text.slice(0, faceSentence.index)}${nextSentence}${text.slice(faceSentence.index + faceSentence[1].length)}`;
      } else {
        text = `${text.replace(/\.+\s*$/, "")} with ${additions.join(", ")}.`;
      }
    }
    return text.replace(/\s{2,}/g, " ").trim();
  };
  const applyWizardStoryboardTriggerPhrases = (prompt, scene) => {
    let text = enforceWizardStoryboardVideoFacialRequirements(prompt, scene);
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const parts = { start: [], end: [] };
    const add = (trigger, position = "start") => {
      const value = String(trigger || "").trim();
      if (!value) return;
      const key = position === "end" ? "end" : "start";
      if (!parts[key].some((item) => item.toLowerCase() === value.toLowerCase())) parts[key].push(value);
    };
    const subjectPosition = refs.subject_trigger_position === "end" ? "end" : "start";
    const locationPosition = refs.location_trigger_position === "end" ? "end" : "start";
    (Array.isArray(scene.subject_refs) ? scene.subject_refs : []).forEach((subject) => {
      add(subject.trigger_phrase || subject.trigger || subject.Trigger, subjectPosition);
    });
    if (scene.location_ref) {
      add(scene.location_ref.trigger_phrase || scene.location_ref.trigger || scene.location_ref.Trigger, locationPosition);
    }
    add(scene.trigger_phrase || scene.trigger || scene.Trigger, scene.trigger_position === "end" ? "end" : "start");
    const stripBoundaryTrigger = (value, trigger) => {
      let current = String(value || "").trim();
      const escaped = String(trigger || "").trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      if (!escaped) return current;
      const leading = new RegExp(`^\\s*${escaped}\\s*(?:,\\s*)?`, "i");
      const trailing = new RegExp(`(?:,\\s*)?${escaped}\\s*$`, "i");
      let previous = "";
      while (current && current !== previous) {
        previous = current;
        current = current.replace(leading, "").replace(trailing, "").trim();
      }
      return current;
    };
    [...parts.start, ...parts.end]
      .sort((a, b) => b.length - a.length)
      .forEach((trigger) => {
        text = stripBoundaryTrigger(text, trigger);
      });
    if (parts.start.length) {
      const prefix = parts.start.join(", ");
      if (!text.toLowerCase().startsWith(prefix.toLowerCase())) text = text ? `${prefix}, ${text}` : prefix;
    }
    if (parts.end.length) {
      const suffix = parts.end.join(", ");
      if (!text.toLowerCase().endsWith(suffix.toLowerCase())) text = text ? `${text}, ${suffix}` : suffix;
    }
    return text;
  };
  // Retry entries for the scene a batch stopped on and the scenes after it that were never tried.
  const unfinishedSceneFailures = (scenes, error) => scenes.map((scene, index) => ({
    key: `unfinished:${scene.id}`,
    segmentId: String(scene.id || ""),
    sceneLabel: scene.label || `Scene ${scene.scene_number || ""}`.trim(),
    error: index === 0 ? String(error?.message || error) : "Not tried: the run stopped before this scene.",
    raw: index === 0 ? String(error?.rawGemmaPrompt || error?.message || error) : "",
  }));
  // sceneIds: run only these scenes (the Story Builder's picked scenes), replacing their beats.
  const createWizardSceneBeats = async ({ overwrite = false, failedIds = [], sceneIds = [] } = {}) => {
    const segments = allEditableSegments()
      .slice()
      .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    const scenes = storyboardScenePayload();
    const retryIds = new Set([...failedIds, ...sceneIds].map((value) => String(value)));
    const targets = scenes.filter((scene) => {
      const segment = segments.find((item) => item.id === scene.id);
      return retryIds.size
        ? retryIds.has(String(scene.id || ""))
        : overwrite || !String(segment?.story_beat || scene.story_beat || "").trim();
    });
    if (!targets.length) {
      toast(overwrite ? "No scenes found." : "No scene story beats are missing.");
      return { created: 0 };
    }
    const storyboardState = wizardStoryboardState(scenes, { promptMode: "image" });
    const progress = createProgressWindow(overwrite ? "Replace Scene Story Beats" : "Create Missing Scene Story Beats", { zIndex: 100012 });
    let created = 0;
    let stoppedAt = 0;
    const failures = [];
    try {
      progress.set(`${failedIds.length ? "Retrying failed" : "Creating"} ${targets.length} scene story beat${targets.length === 1 ? "" : "s"}...`, 5);
      for (let index = 0; index < targets.length; index += 1) {
        stoppedAt = index;
        const scene = targets[index];
        const segmentIndex = segments.findIndex((item) => item.id === scene.id);
        const previousBeat = segmentIndex > 0 ? String(segments[segmentIndex - 1]?.story_beat || "") : "";
      const nextLyrics = segmentIndex >= 0 && segmentIndex < segments.length - 1 ? String(segments[segmentIndex + 1]?.lyric_text || "") : "";
      const base = 8 + Math.round((index / Math.max(1, targets.length)) * 84);
      progress.set(`Scene Beat ${index + 1}/${targets.length}: ${scene.label || `Scene ${scene.scene_number || index + 1}`}`, base);
      try {
          const beatStoryboardPayload = storyboardGptPayload(storyboardState, [{ ...scene, story_beat: "" }]);
          const authoritativeMappedSingers = segmentIndex >= 0
            ? selectedPerformerSubjectsForSegment(segments[segmentIndex], state.fluxReferenceBuilder)
              .map((subject) => String(subject?.name || "").trim())
              .filter(Boolean)
            : [];
          const mappedSingers = authoritativeMappedSingers.length
            ? authoritativeMappedSingers
            : (Array.isArray(scene.lyric_singers) && scene.lyric_singers.length
              ? scene.lyric_singers
              : (segmentIndex >= 0 && Array.isArray(segments[segmentIndex]?.lyric_singers) ? segments[segmentIndex].lyric_singers : []));
          const visibleSubjects = Array.isArray(scene.subjects) ? scene.subjects : [];
          const singing = Array.from(new Set(mappedSingers.map((item) => String(item || "").trim()).filter(Boolean)));
          const silent = visibleSubjects
            .map((item) => String(item || "").trim())
            .filter((item) => item && !singing.some((singer) => singer.toLowerCase() === item.toLowerCase()));
          const beatScene = beatStoryboardPayload.scenes?.[0];
          if (beatScene) {
            // The Scenes tab override, which the beat must always follow.
            if (scene.scene_override) beatScene.scene_override = scene.scene_override;
            beatScene.lyric_singers = singing;
            beatScene.performer_assignment = {
              singing,
              silent,
              instruction: singing.length === 1
                ? `${singing[0]} is the only singing performer. Every other visible subject is silent.`
                : singing.length > 1
                  ? `Only ${singing.join(", ")} sing. Every other visible subject is silent.`
                  : "No visible subject is assigned to sing.",
            };
          }
          const data = await postJson("/vrgdg/storyboard/scene_story_beat", {
            ...textGemmaRunnerPayload(),
            // The project folder lets a Story Mode beat use the project's own Story Mode beat instructions.
            project_folder: activeProjectFolderForSave() || "",
            model_file: i2vTextGemmaModelSelect.value || t2iTextGemmaModelSelect.value || "",
            story_layer: normalizeBuilderStoryLayer(state.builderStoryLayer),
            storyboard_payload: beatStoryboardPayload,
            all_subjects: (Array.isArray(state.fluxReferenceBuilder?.subjects) ? state.fluxReferenceBuilder.subjects : [])
              .map((subject) => ({ name: String(subject?.name || ""), description: String(subject?.description || ""),
                reference_type: String(subject?.reference_type || "character") })),
            previous_beat: previousBeat,
            next_lyrics: nextLyrics,
            unload_after: index === targets.length - 1,
            max_new_tokens: 360,
          }, 240000);
          const segment = segments.find((item) => item.id === scene.id);
          if (segment) segment.story_beat = String(data.story_beat || "").trim();
          created += 1;
        } catch (error) {
          if (!isRecoverableBuildGemmaError(error)) throw error;
          failures.push({
            key: `storybeat:${scene.id}`,
            segmentId: String(scene.id || ""),
            sceneLabel: scene.label || `Scene ${scene.scene_number || index + 1}`,
            error: String(error?.message || error),
            raw: String(error?.message || error),
          });
          progress.set(`Scene Beat ${index + 1}/${targets.length} skipped. Continuing with the remaining scenes...`, base);
        }
      }
      syncInspector();
      render();
      await autoSaveSessionQuiet("wizard scene story beats");
      progress.set(`Scene story beats complete.\nCreated ${created}.${failures.length ? ` ${failures.length} scene${failures.length === 1 ? " was" : "s were"} skipped.` : ""}`, 100);
      progress.close(1400);
      toast(`Created ${created} scene story beat${created === 1 ? "" : "s"}${failures.length ? ` with ${failures.length} skipped scene${failures.length === 1 ? "" : "s"}` : ""}.`, Boolean(failures.length));
      if (failures.length) showGemmaBatchFailures(failures, {
        retryHandler: (items) => createWizardSceneBeats({ failedIds: items.map((item) => item.segmentId) }),
      });
      return { created, failures };
    } catch (error) {
      progress.set(`Scene story beats stopped after ${created}/${targets.length}:\n${String(error?.message || error)}`, 100);
      progress.close(1800);
      toast(String(error?.message || error), true);
      // Keep the beats made before the stop.
      syncInspector();
      render();
      await autoSaveSessionQuiet("wizard scene story beats stopped");
      showGemmaBatchFailures([...failures, ...unfinishedSceneFailures(targets.slice(stoppedAt), error)], {
        retryHandler: (items) => createWizardSceneBeats({ failedIds: items.map((item) => item.segmentId) }),
      });
      return { created, error: String(error?.message || error) };
    }
  };
  const runWizardStoryboardGemmaAll = async (options = {}) => {
    updateActiveFromInputs();
    saveI2VVideoSettingsFromPanel();
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    const videoMode = currentVideoMode();
    const segments = allEditableSegments()
      .slice()
      .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    const scenes = storyboardScenePayload();
    if (!scenes.length) {
      toast("No storyboard scenes found.", true);
      return;
    }
    const segmentById = new Map(segments.map((segment) => [String(segment.id || ""), segment]));
    // sceneIds: run only these scenes (the Story Builder's picked scenes).
    const failedIds = new Set([...(options.failedIds || []), ...(options.sceneIds || [])].map((value) => String(value)));
    const targetScenes = failedIds.size
      ? scenes.filter((scene) => failedIds.has(String(scene.id || "")))
      : scenes;
    const storyboardState = {
      ...wizardStoryboardState(scenes, { promptMode: "video", videoMode }),
      gemmaSettings: {
        ...textGemmaRunnerPayload(),
        model_file: i2vTextGemmaModelSelect.value || t2iTextGemmaModelSelect.value || "",
        vision_model_file: i2vGemmaModelSelect.value || gemmaModelSelect.value || "",
        mmproj_file: i2vMmprojSelect.value || mmprojSelect.value || "",
        n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
        n_gpu_layers: normalizeGemmaGpuLayers(state.gemmaGpuLayers),
        gpu_device: normalizeGemmaGpuDevice(state.gemmaGpuDevice),
        n_threads: 8,
        unload_after: true,
      },
    };
    const runnerName = promptRunnerActionName();
    const runnerGenericName = runnerName;
    const progress = createProgressWindow(`Storyboard ${runnerName} All (${miniMaxProject ? "MiniMax H3 project/locked modes" : videoModeDisplayLabel(videoMode, true)})`, { zIndex: 100012 });
    let created = 0;
    let stoppedAt = 0;
    const failurePrefix = `storyboard:${videoMode}`;
    try {
      progress.set(`${options.failedIds?.length ? "Retrying failed Storyboard scenes" : `Starting Storyboard ${runnerName} All`}...\nScenes: ${targetScenes.length}\nUsing the same prompt writer as Storyboard Builder.`, 5);
      for (let index = 0; index < targetScenes.length; index += 1) {
        stoppedAt = index;
        const scene = targetScenes[index];
        const segment = segmentById.get(String(scene.id || "")) || segments.find((item) => String(item.id || "") === String(scene.id || "")) || segments[index];
        const base = 8 + Math.round((index / Math.max(1, targetScenes.length)) * 84);
        const label = `Storyboard ${runnerName} All ${index + 1}/${targetScenes.length}: ${scene.label || `Scene ${scene.scene_number || index + 1}`}`;
        try {
          progress.set(`${label}\nCreating storyboard video prompt...`, base);
          if (segment && state.autoTimeSingerCuesBeforePrompt && !segment.no_character_present
            && normalizeVideoType(segment.performance_mode || state.videoType) === "singing") {
            const timed = await ensureAutoTimedSingerCuesBeforePrompt(segment);
            if (!timed) throw new Error(`${label}: Whisper timing did not complete; prompt generation was stopped.`);
          }
          const promptCueMap = segment && Array.isArray(segment.lyric_cue_map)
            ? normalizeLyricCueMapForSegment(segment, undefined, { preserveBlank: true })
            : [];
          const timedCueContract = promptCueMap.length
            ? [
              "AUTHORITATIVE TIMED LYRIC CUES — use these exact intervals; do not repeat the complete lyric in every shot:",
              ...promptCueMap.map((cue, cueIndex) => {
                const start = Number(cue?.start);
                const end = Number(cue?.end);
                const range = Number.isFinite(start) && Number.isFinite(end) && end > start
                  ? `${start.toFixed(3)}s-${end.toFixed(3)}s`
                  : `cue ${cueIndex + 1}`;
                return cue.type === "instrumental"
                  ? `[${range}] Use only the assigned visual action and camera direction.`
                  : `[${range}] ${cue.singer_name || "the assigned singer"} sings only: "${flattenLyricForPrompt(cue.text)}".`;
              }),
              "Each cue is authoritative. Assign only the words in that interval to that shot."
            ].join("\n")
            : "";
          const sceneForPrompt = segment
            ? {
              ...scene,
              lyrics: timedCueContract || scene.lyrics || scene.lyric_text || "",
              timed_lyric_cue_contract: timedCueContract,
              lyric_cue_map: promptCueMap,
              performer_assignment: {
                singing: Array.from(new Set(promptCueMap.filter((cue) => cue.type !== "instrumental").map((cue) => cue.singer_name).filter(Boolean))),
                cue_map: promptCueMap,
              },
              lyric_shot_word_timing_enabled: Boolean(segment.lyric_shot_word_timing_enabled),
              lyric_performance_mode: String(segment.lyric_performance_mode || ""),
            }
            : scene;
          const promptStoryboardPayload = storyboardGptPayload(storyboardState, [sceneForPrompt]);
          const data = miniMaxProject
            ? await storyboardPromptPipeline()(sceneForPrompt, {
              unloadAfter: index === targetScenes.length - 1,
              storyboardPayload: promptStoryboardPayload,
              progress,
              progressPercent: base,
              progressLabel: label,
              // The Story Builder writes prompts with Story Mode's own MiniMax instructions.
              storyMode: Boolean(options.storyMode),
            })
            : await postJson("/vrgdg/storyboard/gemma_video_prompt", {
              ...(storyboardState.gemmaSettings || {}),
              unload_after: index === targetScenes.length - 1,
              storyboard_payload: promptStoryboardPayload,
              max_new_tokens: 1400,
              temperature: 0.35,
              top_p: 0.90,
            }, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
          const finalizedPrompt = miniMaxProject
            ? String(data.prompt || "").trim()
            : segment ? finalizeVideoPromptDraftOnly(segment, data.prompt) : String(data.prompt || "").trim();
          const prompt = miniMaxProject
            ? finalizedPrompt
            : applyWizardStoryboardTriggerPhrases(finalizedPrompt, sceneForPrompt);
          if (!prompt) throw new Error(`${scene.label || `Scene ${index + 1}`}: ${runnerGenericName} returned an empty Storyboard video prompt.`);
          if (segment) {
            if (miniMaxProject) {
              segment.minimax_h3_prompt = prompt;
              segment.minimax_h3_prompt_origin = "gemma";
            } else {
              setSegmentPromptForEdit(segment, "i2v", prompt, { origin: "gemma" });
              segment.video_prompt_type = videoMode;
            }
          }
          scene.video_prompt = prompt;
          scene.video_prompt_origin = "gemma";
          gemmaBatchFailureStore()[`${failurePrefix}:${scene.id}`] = undefined;
          created += 1;
          // Autosave copies the inspector's prompt box into the selected scene, so show the new prompt there first or
          // the selected scene's old prompt is written back over it.
          syncInspector();
          await autoSaveSessionQuiet(`${runnerName} Storyboard ${scene.label || `Scene ${index + 1}`}`);
          progress.set(`${label}\nSaved prompt into the Video Builder scene.`, Math.min(96, base + 6));
        } catch (error) {
          if (!isRecoverableBuildGemmaError(error)) throw error;
          let debugPath = String(error?.gemmaDebugPath || "");
          if (!debugPath) {
            try {
              debugPath = await saveGemmaJunkDebug(error, { label, segment });
            } catch (debugError) {
              console.warn("[VRGDG Music Builder] Could not save recoverable Storyboard prompt debug output:", debugError);
            }
          }
          recordGemmaBatchFailure(`${failurePrefix}:${scene.id}`, segment, sceneDisplayName(segment), error, debugPath);
          progress.set(`${label} skipped. Continuing with the remaining scenes...`, base);
        }
      }
      ensureAllSegmentRuntimeFields();
      syncInspector();
      render();
      progress.set("Saving storyboard prompts into the project...", 96);
      await autoSaveSessionQuiet("wizard storyboard llm all");
      const failures = targetScenes.map((scene) => gemmaBatchFailureStore()[`${failurePrefix}:${scene.id}`]).filter(Boolean);
      progress.set(`Storyboard ${runnerName} All complete.\nCreated ${created} video prompt${created === 1 ? "" : "s"}.${failures.length ? ` ${failures.length} scene${failures.length === 1 ? " was" : "s were"} skipped.` : ""}`, 100);
      progress.close(1800);
      toast(`Storyboard ${runnerName} All created ${created} video prompt${created === 1 ? "" : "s"}${failures.length ? ` with ${failures.length} skipped scene${failures.length === 1 ? "" : "s"}` : ""}.`, Boolean(failures.length));
      if (failures.length) {
        showGemmaBatchFailures(failures, {
          retryHandler: (items) => runWizardStoryboardGemmaAll({ failedIds: items.map((item) => item.segmentId) }),
        });
      }
    } catch (error) {
      progress.set(`Storyboard ${runnerName} All stopped after ${created}/${targetScenes.length} scenes:\n${String(error?.message || error)}`, 100);
      progress.close(1800);
      toast(`Storyboard ${runnerName} All stopped after ${created}/${targetScenes.length} scenes:\n${String(error?.message || error)}`, true);
      const skipped = targetScenes.slice(0, stoppedAt)
        .map((scene) => gemmaBatchFailureStore()[`${failurePrefix}:${scene.id}`]).filter(Boolean);
      showGemmaBatchFailures([...skipped, ...unfinishedSceneFailures(targetScenes.slice(stoppedAt), error)], {
        retryHandler: (items) => runWizardStoryboardGemmaAll({ failedIds: items.map((item) => item.segmentId) }),
      });
    }
  };

  // The Storyboard Builder owns the Video Builder prompt pipeline; register it without opening its UI.
  function storyboardPromptPipeline() {
    if (typeof storyboardPipeline.runner !== "function") openStoryboardBuilderFromProject({ registerPromptPipelineOnly: true });
    if (typeof storyboardPipeline.runner !== "function") {
      throw new Error("The Storyboard prompt pipeline is not ready yet. Refresh ComfyUI and try again.");
    }
    return storyboardPipeline.runner;
  }

  function openWizardFromBuilder() {
    let ltxAudioChoice = state.wizardBetaDraft?.audioMode === "silent" ? "silent" : "input_audio";
    const modes = {
      ltx: [
        { value: "i2v", label: "Image to Video", description: "Generate or supply starting images, then animate them." },
        { value: "t2v", label: "Text to Video", description: "Describe scenes, characters and locations without requiring reference images." },
        { value: "rtv", label: "Reference to Video", description: "Guide the video with character and location reference images." },
        { value: "ingredients", label: "Ingredients to Video", description: "Combine the reference images assigned in your Ingredients Builder." },
        { value: "id_lora", label: "ID-LoRA", description: "Use an identity LoRA, starting image and reference voice." },
        { value: "flf", label: "First / Last Frame", description: "Guide each scene between starting and ending images." },
        { value: "import", label: "Import Custom Video — not implemented in Builder", disabled: true },
      ],
      minimax_h3: MINIMAX_H3_MODE_OPTIONS.map(item => ({ ...item, description: {
        text_to_video: "Create video from scene descriptions, with optional character and location context.",
        image_to_video: "Animate generated or uploaded starting images.",
        reference_to_video: "Use character and location reference images to guide scenes.",
        image_reference_to_video: "Combine a starting image with references using the two-pass workflow. Requires input audio.",
        video_to_video: "Use source videos for continuation, movement, camera or visual guidance.",
      }[item.value] })),
    };
    const sync = () => {
      syncProjectVideoEngineUI(); syncVideoModePanel(); syncI2VVideoSettingsPanel(); syncMiniMaxH3Panel();
      syncFluxKleinPanel(); syncZImageSettingsPanel(); syncErnieImagePanel();
      syncKrea2TwoPassPanel(); syncNBImagePanel(); syncFlowGptBrowserPanel(); syncInspector();
    };
    const flush = () => {
      if (wizardVideoSettings.global) {
        if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") saveMiniMaxH3SettingsFromPanel();
        else saveI2VVideoSettingsFromPanel();
      }
      if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
        state.videoModelMode = ({ text_to_video: "t2v", image_to_video: "i2v", reference_to_video: "rtv", image_reference_to_video: "i2v", video_to_video: "t2v" })[state.miniMaxH3Settings.video_mode];
      }
      const saveImage = { zimage: saveZImageSettingsFromPanel, flux_klein: saveFluxKleinSettingsFromPanel, ernie_image: saveErnieImageSettingsFromPanel, krea2_2pass: saveKrea2TwoPassSettingsFromPanel, nano_banana: saveNBImageSettingsFromPanel, flow_gpt: saveFlowGptBrowserSettingsFromPanel }[state.imageModelMode];
      saveImage?.();
    };
    const referenceDraft = (reference) => ({
      ...reference.image,
      name: reference.image?.name || reference.name || "Reference",
      referenceId: reference.id,
      title: reference.name || "",
      description: reference.description || "",
    });
    const snapshot = () => {
      const engine = normalizeProjectVideoEngine(state.projectVideoEngine);
      const mini = cloneMiniMaxH3Settings(state.miniMaxH3Settings);
      const mode = engine === "minimax_h3" ? mini.video_mode : currentVideoMode();
      const inputOnly = mode === "image_reference_to_video" || (mode === "reference_to_video" && ["two_pass", "advanced"].includes(mini.ref_pass_mode));
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      const subjectReferences = refs.subjects.filter(item => item.image?.path || item.image?.data);
      const locationReferences = refs.locations.filter(item => item.image?.path || item.image?.data);
      return {
        engine, mode, pipeline: mini.pipeline, modes, ltxVersion: state.i2vVideoSettings.ltx_version || "2.5", imageMode: state.imageModelMode || "zimage",
        performance: state.videoType, performances: VIDEO_TYPE_OPTIONS,
        promptRunnerLabel: promptRunnerActionName(),
        audioMode: engine === "minimax_h3" ? mini.audio_mode : mode === "id_lora" ? "reference_voice" : ltxAudioChoice,
        audioModes: engine === "minimax_h3" ? MINIMAX_H3_AUDIO_MODE_OPTIONS.map(item => ({ ...item, disabled: inputOnly && item.value === "built_in_audio" })) : mode === "id_lora" ? [{ value: "reference_voice", label: "ID-LoRA reference voice" }] : [{ value: "input_audio", label: "Use an audio file" }, { value: "silent", label: "Silent timeline" }],
        audioHelp: engine === "minimax_h3" ? inputOnly ? "This multi-pass mode requires input audio. Choose a single-pass mode for built-in audio." : "Built-in audio generates sound from the prompt and does not require a song." : mode === "id_lora" ? "Set the reference voice in Models & LoRAs. Scene dialogue guides speech." : "The current LTX Builder supports supplied audio or silence; native generated audio is not enabled in its render path.",
        audioPath: String(audioInput.value || state.audioPath || ""),
        lyrics: state.lyricMapper?.source_text || "", direction: state.builderStoryLayer?.overall_story_idea || "",
        subjects: subjectReferences.map(referenceDraft), locations: locationReferences.map(referenceDraft),
        draft: state.wizardBetaDraft ? {
          ...state.wizardBetaDraft,
          singer: subjectReferences[0] ? referenceDraft(subjectReferences[0]) : null,
          subjects: subjectReferences.map(referenceDraft),
          locations: locationReferences.map(referenceDraft),
        } : null, scenes: allEditableSegments(),
        referenceCount: [...refs.subjects, ...refs.locations].filter(item => item.image?.path || item.image?.data).length,
        refmodCount: [...refs.subjects, ...refs.locations].filter(item => item.source === "refmod" && item.refmod?.name).length,
        hasVideoReference: (activeSegment()?.minimax_h3_video_references || []).some(item => item.path),
        imageModes: ["zimage", "flux_klein", "ernie_image", "krea2_2pass", "nano_banana", "flow_gpt"].map(value => ({ value, label: imageModeDisplayLabel(value) })),
      };
    };
    const configure = values => {
      flush(); pushHistory();
      if (values.engine) state.projectVideoEngine = normalizeProjectVideoEngine(values.engine);
      if (values.ltxVersion) setBuilderLtxVersion(values.ltxVersion);
      if (values.mode) {
        if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
          state.miniMaxH3Settings = cloneMiniMaxH3Settings({ ...state.miniMaxH3Settings,
            pipeline: values.mode === "reference_to_video" ? state.miniMaxH3Settings.pipeline : "standard",
            video_mode: values.mode });
          state.miniMaxH3TwoPassEnabled = values.mode === "image_reference_to_video";
          state.miniMaxH3ThreePassEnabled = false;
        } else state.videoModelMode = values.mode;
      }
      if (values.pipeline && normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
        state.miniMaxH3Settings = cloneMiniMaxH3Settings({ ...state.miniMaxH3Settings, pipeline: values.pipeline });
        state.miniMaxH3TwoPassEnabled = state.miniMaxH3Settings.render_pass === "two_pass";
        state.miniMaxH3ThreePassEnabled = state.miniMaxH3Settings.render_pass === "three_pass";
      }
      if (values.imageMode) {
        state.imageModelMode = values.imageMode;
        state.fluxKleinSettings.image_model_mode = values.imageMode;
        state.fluxKleinSettings.enabled = values.imageMode === "flux_klein";
      }
      if (values.performance) { state.videoType = normalizeProjectVideoType(values.performance); syncVideoTypeControl(); }
      if (values.audioMode) {
        if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") state.miniMaxH3Settings = cloneMiniMaxH3Settings({ ...state.miniMaxH3Settings, audio_mode: values.audioMode });
        else ltxAudioChoice = values.audioMode;
      }
      if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
        const mode = state.miniMaxH3Settings.video_mode;
        state.videoModelMode = ({ text_to_video: "t2v", image_to_video: "i2v", reference_to_video: "rtv", image_reference_to_video: "i2v", video_to_video: "t2v" })[mode];
      }
      sync();
    };
    const movePanel = (holder, panel) => {
      const marker = document.createComment("Wizard panel position");
      panel.before(marker); holder.append(panel);
      return () => { if (marker.parentNode) marker.replaceWith(panel); };
    };
    const openScene = id => {
      flush(); state.activeId = id; state.activeTrack = segmentTrack(allEditableSegments().find(item => item.id === id)); sync(); setInspectorTab("video");
      const overlay = document.createElement("div"); overlay.style.cssText = "position:fixed;inset:0;background:#000b;z-index:100008;display:grid;place-items:center;padding:24px";
      const box = document.createElement("div"); box.style.cssText = "background:#17212c;color:white;width:min(900px,100%);max-height:90vh;overflow:auto;padding:18px;border:1px solid #64748b;border-radius:12px";
      const close = makeButton("Back to Wizard"); const content = document.createElement("div");
      box.append(close, content); overlay.append(box); document.body.append(overlay);
      const restore = movePanel(content, inspector);
      close.onclick = () => { flush(); restore(); overlay.remove(); sync(); };
    };
    const addReferences = draft => {
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      const upsert = (list, id, name, description, image, previousImage) => {
        let item = list.find(item => item.id === id);
        if (!item) { item = { id, name, description: "", reference_type: "character" }; list.push(item); }
        item.name = name;
        item.description = description ?? item.description;
        if (image && (previousImage?.data !== image.data || item.image?.name !== image.name || (!item.image?.path && item.image?.data !== image.data))) item.image = { path: image.path || "", data: image.data, name: image.name };
      };
      const removed = new Set(draft.removedReferenceIds || []);
      refs.subjects = refs.subjects.filter(item => !removed.has(item.id));
      refs.locations = refs.locations.filter(item => !removed.has(item.id));
      for (const map of [refs.subject_scene_map, refs.performer_scene_map]) {
        for (const [id, values] of Object.entries(map || {})) map[id] = values.filter(value => !removed.has(value));
      }
      for (const [id, value] of Object.entries(refs.scene_map || {})) if (removed.has(value)) delete refs.scene_map[id];
      const referenceMode = wizardNeeds(draft.engine, draft.mode).references;
      const subjects = draft.subjects || (draft.singer ? [draft.singer] : []);
      const saveImages = (images, list, prefix) => images.forEach((image, index) => {
        if (!image.referenceId) {
          let id = prefix === "character" && !index ? "wizard_beta_character" : `wizard_beta_${prefix}_${index}`;
          let suffix = index;
          while (list.some(item => item.id === id)) id = `wizard_beta_${prefix}_${++suffix}`;
          image.referenceId = id;
        }
        upsert(list, image.referenceId, image.title?.trim() || image.name.replace(/\.[^.]+$/, ""), image.description, image);
      });
      if (subjects.length) {
        refs.cleared = false;
        saveImages(subjects, refs.subjects, "character");
      } else if (!referenceMode && draft.characters.trim()) {
        refs.cleared = false;
        upsert(refs.subjects, "wizard_beta_character", "Wizard character", draft.characters);
      }
      if (draft.locations.length || (!referenceMode && draft.locationsText.trim())) { refs.locations_cleared = false; refs.cleared = false; }
      if (!referenceMode && draft.locationsText.trim()) upsert(refs.locations, "wizard_beta_location_text", "Wizard location", draft.locationsText);
      saveImages(draft.locations, refs.locations, "location");
      if (removed.size) refs.subject = refs.subjects[0] ? { ...refs.subjects[0] } : { name: "", description: "", image: { path: "", data: "", name: "" } };
      refs.subject_count = refs.subjects.length;
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
    };
    const generatePrompts = () => openStoryboardBuilderFromProject({
      focusedSection: "scenes", allowImagePrep: false, promptActionOnly: true,
    });
    openWizard({
      snapshot, configure, flush,
      imageUrl: makeEditorImageUrl,
      mountSettings: (holder, kind) => {
        if (kind !== "video") {
          const panel = ({ zimage: zimageSettingsPanel, flux_klein: fluxKleinPanel, ernie_image: ernieImagePanel, krea2_2pass: krea2TwoPassPanel, nano_banana: nbImagePanel, flow_gpt: flowGptModePanel })[state.imageModelMode];
          return movePanel(holder, panel);
        }
        wizardVideoSettings.global = true;
        syncI2VVideoSettingsPanel(); syncMiniMaxH3Panel();
        const miniMax = snapshot().engine === "minimax_h3";
        const source = (miniMax ? miniMaxSubTabs : videoSubTabs).wrapper.children[1];
        const contents = Array.from(source.children).slice(0, 2);
        const restores = [];
        const hidden = document.createElement("div");
        const sceneControls = miniMax
          ? [useSceneMiniMaxH3Settings.wrapper, useSceneMiniMaxH3SettingsNote, ...Object.values(miniMaxModePanels)]
          : [useSceneI2VVideoSettings.wrapper, useSceneI2VVideoSettingsNote, createSceneVideoActions, rtvSceneImageAnchorSection];
        for (const control of sceneControls) restores.push(movePanel(hidden, control));
        if (!miniMax) {
          for (const button of createSceneVideoButtons) {
            if (contents.some(content => content.contains(button))) restores.push(movePanel(hidden, button.parentElement));
          }
        }
        restores.push(...organizeWizardVideoModels({ holder, contents, miniMax, miniMaxPassChooser, movePanel }));
        const tabs = contents.map((content, index) => {
          const wrapper = document.createElement("div");
          wrapper.className = index ? "wb-settings-tab" : "wb-settings-tab wb-model-grid";
          restores.push(movePanel(wrapper, content));
          content.style.display = "flex";
          return { label: index ? "Video Settings" : "Models", value: index ? "settings" : "models", content: wrapper };
        });
        const modelTabs = makeSubTabs(tabs).wrapper;
        modelTabs.className = "wb-model-tabs";
        holder.append(modelTabs);
        return () => {
          flush();
          for (const restore of restores.reverse()) restore();
          wizardVideoSettings.global = false;
          (miniMax ? miniMaxSubTabs : videoSubTabs).setActive("models");
          syncI2VVideoSettingsPanel(); syncMiniMaxH3Panel();
        };
      },
      openRunner: openGemmaRunnerModal,
      openReferences: (focusedSection, onClose) => {
        const { engine, mode, imageMode } = snapshot();
        const needs = wizardNeeds(engine, mode);
        const textOnlyMode = ["t2v", "text_to_video"].includes(mode)
          || (needs.images && !needs.references && !["nano_banana", "flux_klein", "flow_gpt"].includes(imageMode));
        if (focusedSection) {
          openFluxReferenceBuilderModal({ wizardMode: true, focusedSection, onClose, textOnlyMode });
          return;
        }
        if (mode === "ingredients") openIngredientsReferenceBuilderModal();
        else if (mode === "id_lora") openIdLoraReferenceBuilderModalSafely();
        else openFluxReferenceBuilderModal({ wizardMode: true, textOnlyMode });
      },
      openLyrics: openLyricReviewModal,
      openStoryboard: (focusedSection, onClose) => {
        const { engine, mode } = snapshot();
        openStoryboardBuilderFromProject({ focusedSection, onClose,
          allowImagePrep: engine === "minimax_h3" ? mode === "image_to_video" : mode === "i2v" });
      },
      openScene,
      save: async (draft, edits) => {
        if (!String(projectInput.value || state.projectFolder || "").trim()) {
          // newProject resets settings; preserve the selections made in the wizard.
          const settings = { performance: state.videoType, engine: state.projectVideoEngine, mode: state.videoModelMode, mini: cloneMiniMaxH3Settings(state.miniMaxH3Settings), ltx: cloneI2VVideoSettings(state.i2vVideoSettings), imageMode: state.imageModelMode,
            z: state.zimageSettings, flux: state.fluxKleinSettings, ernie: state.ernieImageSettings, krea: state.krea2TwoPassSettings, nb: state.nbImageSettings, flow: state.flowGptBrowserSettings,
            runner: Object.fromEntries([
              "textGemmaRunner", "qwenModelFile", "qwenMmprojFile", "gemmaModelFile", "gemmaContextLimit", "gemmaOutputTokenLimit", "gemmaGpuLayers", "gemmaGpuDevice",
              "lmStudioBaseUrl", "lmStudioModel", "lmStudioContextLimit", "lmStudioOutputTokenLimit", "llmApiProvider", "llmApiModel",
              "ownServerUrl", "ownServerModel", "ownServerOutputTokenLimit", "ownServerTimeoutMinutes",
            ].map(key => [key, state[key]])) };
          if (!await newProject()) throw new Error("Choose a project folder to save.");
          Object.assign(state, settings.runner, { videoType: settings.performance, projectVideoEngine: settings.engine, videoModelMode: settings.mode, miniMaxH3Settings: settings.mini, i2vVideoSettings: settings.ltx, imageModelMode: settings.imageMode, zimageSettings: settings.z, fluxKleinSettings: settings.flux, ernieImageSettings: settings.ernie, krea2TwoPassSettings: settings.krea, nbImageSettings: settings.nb, flowGptBrowserSettings: settings.flow });
          syncVideoTypeControl();
          sync();
        }
        if (draft.song && !await chooseProjectAudioFile(draft.song)) throw new Error("The audio file could not be saved.");
        pushHistory(); addReferences(draft);
        state.lyricMapper = normalizeLyricMapper({ ...state.lyricMapper, source_text: draft.lyrics });
        state.builderStoryLayer = normalizeBuilderStoryLayer({ ...state.builderStoryLayer, overall_story_idea: draft.direction });
        for (const [id, fields] of edits) { const scene = allEditableSegments().find(item => item.id === id); if (scene) Object.assign(scene, fields); }
        const { song, ...savedDraft } = draft; state.wizardBetaDraft = savedDraft;
        sync(); await saveSession({ quiet: true, throwOnError: true }); render();
      },
      openTiming: async (kind) => {
        const inputAudio = snapshot().audioMode === "input_audio";
        const previousIds = new Set(allEditableSegments().map(scene => scene.id));
        if (kind !== "manual" && !inputAudio) throw new Error("Transcription requires input audio.");
        if (inputAudio && !snapshot().audioPath) throw new Error("Choose an audio file first.");
        if (kind === "existing") {
          if (!allEditableSegments().length) throw new Error("Create scenes first or choose No scenes yet.");
          await transcribeLyricsForTimeline();
        } else if (kind === "new") {
          await createScenesFromTimestampedLyrics();
        } else if (kind === "manual") {
          await new Promise(resolve => {
            if (inputAudio) openLyricMappingWorkflowModal({ manualOnly: true, onClose: resolve });
            else openBulkSegmentsModal({ initialMode: "ranges", onClose: resolve });
          });
        }
        const created = allEditableSegments().filter(scene => !previousIds.has(scene.id));
        if (created.length) {
          const draft = state.wizardBetaDraft;
          const needs = wizardNeeds(draft.engine, draft.mode);
          const context = [draft.direction, !needs.references && draft.characters && `Characters: ${draft.characters}`, !needs.references && draft.locationsText && `Locations: ${draft.locationsText}`, draft.sound && `Sound: ${draft.sound}`].filter(Boolean).join("\n");
          for (const scene of created) {
            scene.i2v_notes = [scene.i2v_notes, context].filter(Boolean).join("\n");
            if (needs.endFrame && draft.endImage) { scene.first_last_frame_end_image_data = draft.endImage.data; scene.first_last_frame_end_image_name = draft.endImage.name; }
            if (needs.video && draft.videoPath) scene.minimax_h3_video_references = [{ path: draft.videoPath, purpose: "continuation", start_seconds: 0, duration: 0, use_audio: false }];
          }
          if (draft.audioMode === "silent") await createSilentTimelineAudioForDuration(Math.max(...allEditableSegments().map(scene => Number(scene.end || 0))), { quiet: true });
          sync(); await saveSession({ quiet: true, throwOnError: true }); render();
        } else sync();
      },
      importImages: importTimelineImagesFromFolder,
      generateImages: confirmAndRunZImageAll,
      generatePrompts,
      render: async () => {
        flush();
        const action = await chooseBatchModeAction({
          title: "Render All?",
          intro: "Render all scenes using their prepared prompts, images and effective video settings, then stitch the final video. Missing requirements are shown by Render All; prepare them in Storyboard Scenes before trying again.",
          confirmLabel: "Render All",
          choices: [
            { value: "resume_missing", label: "Resume missing videos", description: "Keep existing videos and render only missing ones." },
            { value: "redo_videos", label: "Redo videos", description: "Render new video versions using the prepared prompts and images." },
          ],
        });
        if (!action) return;
        await renderAllScenes({ sceneScope: "all", forceVideos: action === "redo_videos" });
      },
    });
  }

  return {
    createSceneBeats: createWizardSceneBeats, createVideoPrompts: runWizardStoryboardGemmaAll,
    openWizardFromBuilder,
  };
}
