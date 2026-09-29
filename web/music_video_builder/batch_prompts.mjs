import { storyboardGptPayload } from "../storyboard_builder/gpt_payload.mjs";
import { GEMMA_VIDEO_PROMPT_TIMEOUT_MS, postJson } from "./comfy_api.mjs";
import { USE_STORYBOARD_PROMPT_PIPELINE_FOR_SIDE_PANEL } from "./constants.mjs";
import { escapeHtml, normalizeVideoType, toast } from "./controls.mjs";
import {
  gemmaBatchFailureStore,
  looksLikeUnfilledMiniMaxTemplate,
  recordGemmaBatchFailure,
  showGemmaBatchFailures,
} from "./dialogs.mjs";
import { t2iMissingReason } from "./image_prompts.mjs";
import { miniMaxH3InstructionKey, miniMaxH3ModeLabel } from "./minimax_h3.mjs";
import { miniMaxDialogueAssignmentsForSegment } from "./minimax_prompt.mjs";
import {
  flattenLyricForPrompt,
  isInstrumentalLyricText,
  isRecoverableBuildGemmaError,
  segmentUsesNoLipSyncPerformance,
} from "./prompt_text.mjs";
import { normalizeVideoPromptOrigin } from "./segments.mjs";
import { ACTIVE_STORYBOARD_PROMPT_PIPELINE } from "./storyboard_bridge.mjs";
import { batchEmptyMessage, batchScopeLabel, normalizeBatchScope } from "./timeline_state.mjs";



export function createBatchPrompts({
  activeProjectFolderForSave, activeSegment, allEditableSegments, assembleMiniMaxH3PromptFromCreative,
  assertBatchNotStopped, assertMiniMaxH3ReferenceDescriptionsReady, assertValidMiniMaxH3FinalPrompt,
  autoSaveSessionQuiet, batchTargetItems, buildI2VPromptRequestForSegment, convertLtxPromptsToMiniMaxButton,
  createFluxPromptButton, createI2VButton, createProgressWindow, createT2IButton, currentVideoMode,
  effectiveVideoPerformanceModeForSegment, ensureAllSegmentRuntimeFields,
  ensureAutoTimedSingerCuesBeforePrompt, ensureBuilderManagedFx, ernieCreateT2IButton,
  finalizeVideoPromptDraftOnly, finalizeVideoPromptForSegment, firstLastFramePromptReferences,
  flfGemmaContextMode, flfGemmaSceneConcept, flfGemmaVisualNotes, flfTransitionLoraActive, gemmaRunnerLabel,
  gemmaRunnerLine, gemmaT2IAllButton, gemmaVideoAllButton, generateT2IPromptForSegment, getI2VImageReference,
  hasFirstLastFrameEndImage, i2vAutoChainEnabled, i2vGemmaModelSelect, i2vMmprojSelect, i2vPrompt,
  i2vTextGemmaModelSelect, idLoraGemmaNotesForSegment, idLoraSceneContext, img2imgContinuityEnabled,
  krea2TwoPassCreateT2IButton, llmApiVisionModelSelected, miniMaxCreatePromptButton, miniMaxGemmaModelSelect,
  miniMaxH3CreativePromptContextForSegment, miniMaxH3FrameContinuityPromptEnabled, miniMaxH3ModeForSegment,
  miniMaxH3PromptCharacterBudget, miniMaxH3PromptVisionImages, miniMaxH3PromptVisionImagesForRunner,
  miniMaxH3SceneImageIsPromptInspiration, miniMaxH3SceneImageUseForSegment, miniMaxH3SettingsForSegment,
  miniMaxH3VocalCueMapText, miniMaxMmprojSelect, miniMaxOrderedImageReferenceItemsForSegment, miniMaxPrompt,
  miniMaxPromptReferenceMismatch, miniMaxPromptReferenceSignature, miniMaxRenderReferenceImagePaths,
  miniMaxTextGemmaModelSelect, normalizeLyricCueMapForSegment, previousAutoChainSourceSegment, projectInput,
  promptRunnerActionName, pushHistory, render, requireActiveSegment, rtvReferenceBehaviorForSegment,
  runClearMemoryWorkflowQuiet, runGemmaImagePromptPassWithRetry, runVideoPromptEnhancementBatch,
  saveGemmaJunkDebug, saveMiniMaxSceneInputsFromPanel, saveSessionForSceneVideo, sceneDisplayName,
  sceneVideoConceptPromptText, segmentImageSource, segmentIndexInfo, segmentMappedLocationText,
  segmentMappedSubjectText, setVideoVisionReferenceEnabled, state, storyboardPipeline, storyboardScenePayload,
  syncInspector, syncMiniMaxH3Panel, syncRTVSceneImageAnchorPanel, syncVideoModePanel, textGemmaRunnerPayload,
  updateActiveFromInputs, updateMiniMaxPromptCharacterStatus, videoGemmaNotesForSegment,
  videoModeDisplayLabel, videoVisionReferenceEnabled, wizardStoryboardState, zImageAllButton,
}) {
  function miniMaxBatchReferenceProblems(scenes) {
    return scenes.map(({ segment, index }) => {
      if (miniMaxH3FrameContinuityPromptEnabled(segment)) return "";
      const mode = miniMaxH3ModeForSegment(segment);
      const prompt = String(segment?.minimax_h3_prompt || segment?.i2v_prompt || "").trim();
      const paths = miniMaxRenderReferenceImagePaths(segment, mode);
      const mismatch = miniMaxPromptReferenceMismatch(segment, prompt, mode, paths);
      return mismatch ? `${sceneDisplayName(segment, index)}: ${mismatch} Regenerate this scene's MiniMax prompt.` : "";
    }).filter(Boolean);
  }

  function rememberMiniMaxPromptReferences(segment, prompt, signature) {
    if (!signature) return;
    const binding = { prompt: String(prompt).trim(), signature };
    segment.minimax_h3_prompt_reference_binding = binding;
    const timelineSegment = allEditableSegments().find((item) => item.id === segment.id);
    if (timelineSegment) timelineSegment.minimax_h3_prompt_reference_binding = binding;
  }

  async function runMiniMaxH3PromptGeneration(segment, mode, options = {}) {
    await ensureAutoTimedSingerCuesBeforePrompt(segment);
    const referenceSignature = miniMaxPromptReferenceSignature(segment, mode);
    const visionImages = Array.isArray(options.visionImages)
      ? options.visionImages.filter((item) => item && (String(item.path || "").trim() || String(item.data || "").trim()))
      : miniMaxH3PromptVisionImagesForRunner(segment, mode);
    const visualOnly = segmentUsesNoLipSyncPerformance(segment);
    const promptLyricText = visualOnly || isInstrumentalLyricText(segment.lyric_text) ? "" : flattenLyricForPrompt(segment.lyric_text);
    const promptSingerNames = visualOnly ? [] : (Array.isArray(segment.lyric_singers)
      ? segment.lyric_singers
      : String(segment.lyric_singers || "").split(/[,;\n]+/))
      .map((value) => String(value || "").trim())
      .filter(Boolean);
    const lyricCueMap = visualOnly ? [] : normalizeLyricCueMapForSegment(segment, undefined, { preserveBlank: true });
    const rawExplicitCueMap = !visualOnly && String(segment?.lyric_performance_mode || "") === "cue_map" && Array.isArray(segment?.lyric_cue_map)
      ? segment.lyric_cue_map
      : [];
    if (rawExplicitCueMap.length && !lyricCueMap.length) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: the explicit singer cue map was dropped before LLM prompting. Generation was stopped instead of sending incorrect vocal instructions.`);
    }
    const effectiveSingerNames = promptSingerNames.length
      ? promptSingerNames
      : Array.from(new Set(lyricCueMap.map((cue) => String(cue.singer_name || "").trim()).filter(Boolean)));
    const vocalCueContract = visualOnly ? "" : miniMaxH3VocalCueMapText(segment, mode, { compact: false });
    const assignmentNotes = vocalCueContract
      ? `AUTHORITATIVE PERFORMER / VOCAL CUE MAP — obey exactly; this is part of the user assignment, not optional scene flavor:\n${vocalCueContract}`
      : "";
    const requestedMaxNewTokens = Number(options.maxNewTokens ?? 4000);
    // Give the creative shot descriptions the full H3 limit on the first
    // attempt. Starting at 6300 made Gemma over-compress rich shot prose into
    // telegraphic summaries even when the final prompt could fit under 7000.
    let targetLimit = 7000;
    let lastOversizeError = null;
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      const characterBudget = miniMaxH3PromptCharacterBudget(segment, mode, targetLimit);
      if (characterBudget.fixedChars >= characterBudget.hardLimit) {
        throw new Error(`The required MiniMax H3 definitions and format use ${characterBudget.fixedChars} characters before any shot descriptions, already exceeding the 7,000-character maximum. Shorten mapped reference descriptions before generating this scene.`);
      }
      const budgetMaxNewTokens = Math.max(600, Math.ceil(characterBudget.shotDescriptionChars / 2.5) + 250);
      const contextOptions = {
        ...(options.contextOptions || {}),
        h3TargetLimit: targetLimit,
      };
      const data = await postJson("/vrgdg/music_builder/generate_t2v", {
        ...textGemmaRunnerPayload(),
        project_folder: options.projectFolder || activeProjectFolderForSave(),
        scene_id: options.sceneId || segment.id || "",
        builder_instruction_key: options.builderInstructionKey || miniMaxH3InstructionKey(mode),
        model_file: visionImages.length ? miniMaxGemmaModelSelect.value : miniMaxTextGemmaModelSelect.value,
        repair_model_file: miniMaxTextGemmaModelSelect.value,
        mmproj_file: visionImages.length ? miniMaxMmprojSelect.value : "",
        t2i_prompt: miniMaxH3CreativePromptContextForSegment(segment, mode, contextOptions),
        user_notes: [String(options.userNotes || "").trim(), assignmentNotes].filter(Boolean).join("\n\n"),
        subject_context: "",
        location_context: "",
        no_character_present: Boolean(segment.no_character_present),
        image_references: visionImages,
        prompt_only_scene_inspiration: options.promptOnlySceneInspiration ?? miniMaxH3SceneImageIsPromptInspiration(segment),
        frame_continuity_prompt: Boolean(options.frameContinuityPrompt),
        performance_mode: options.performanceMode || effectiveVideoPerformanceModeForSegment(segment),
        lyric_text: promptLyricText,
        singers: effectiveSingerNames,
        lyric_cue_map: lyricCueMap,
        performer_assignment: {
          singing: Array.from(new Set(effectiveSingerNames)),
          cue_map: lyricCueMap,
        },
        audio_mode: options.audioMode || miniMaxH3SettingsForSegment(segment).audio_mode,
        speaker_assignments: Array.isArray(options.speakerAssignments)
          ? options.speakerAssignments
          : visualOnly ? [] : miniMaxDialogueAssignmentsForSegment(segment),
        camera_motion_speed: segment.camera_motion_speed,
        camera_motion_speed_guidance: segment.camera_motion_speed_guidance,
        character_motion_speed: segment.character_motion_speed,
        character_motion_guidance: segment.character_motion_guidance,
        theme_style_path: "",
        story_idea_path: "",
        subject_scene_path: "",
        unload_after: options.unloadAfter !== false,
        temperature: Number(options.temperature ?? 0.45),
        top_p: Number(options.topP ?? 0.92),
        max_new_tokens: Math.min(requestedMaxNewTokens, budgetMaxNewTokens),
      }, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
      if (data.llm_request_audit_path) {
        segment.minimax_h3_llm_request_audit_path = String(data.llm_request_audit_path);
      }
      const generatedPrompt = String(data.prompt || "").trim();
      if (!generatedPrompt) {
        throw new Error(options.emptyPromptMessage || `The LLM returned an empty MiniMax ${miniMaxH3ModeLabel(mode)} prompt.`);
      }
      if (looksLikeUnfilledMiniMaxTemplate(generatedPrompt)) {
        const error = new Error(`Gemma returned an unfilled MiniMax ${miniMaxH3ModeLabel(mode)} template.`);
        error.rawGemmaPrompt = generatedPrompt;
        throw error;
      }
      try {
        const assembledPrompt = assembleMiniMaxH3PromptFromCreative(segment, mode, generatedPrompt);
        const prompt = typeof options.finalizePrompt === "function"
          ? String(options.finalizePrompt(assembledPrompt) || "").trim()
          : assembledPrompt;
        if (!prompt) throw new Error(options.emptyPromptMessage || `The LLM returned an empty MiniMax ${miniMaxH3ModeLabel(mode)} prompt.`);
        assertValidMiniMaxH3FinalPrompt(prompt, segment, mode);
        rememberMiniMaxPromptReferences(segment, prompt, referenceSignature);
        return {
          ...data,
          prompt,
          already_finalized: true,
          minimax_h3_mode: mode,
          used_minimax_h3_instructions: true,
          h3_prompt_characters: prompt.length,
          h3_prompt_generation_attempts: attempt,
        };
      } catch (error) {
        if (error?.code !== "MINIMAX_H3_PROMPT_TOO_LONG") throw error;
        lastOversizeError = error;
        if (attempt >= 3) break;
        const reduction = Math.max(350, Number(error.promptLength || 7000) - 7000 + 250);
        targetLimit = Math.max(characterBudget.fixedChars, targetLimit - reduction);
      }
    }
    throw new Error(`Gemma could not produce a complete MiniMax H3 prompt within 7,000 characters after 3 concise retries. Last result: ${lastOversizeError?.promptLength || "unknown"} characters. Shorten scene notes or mapped reference descriptions.`);
  }

  async function createMiniMaxH3PromptWithLLM() {
    if (USE_STORYBOARD_PROMPT_PIPELINE_FOR_SIDE_PANEL && storyboardSidePanelPromptPipelineReady()) {
      const segment = requireActiveSegment();
      if (!segment) return;
      updateActiveFromInputs();
      saveMiniMaxSceneInputsFromPanel();
      try {
        await createSidePanelPromptFromStoryboardPipeline(segment, miniMaxH3ModeForSegment(segment), "MiniMax");
        toast("Created the MiniMax prompt through the Storyboard pipeline.");
      } catch (error) {
        toast(`Storyboard side-panel prompt failed. Legacy path is preserved; set USE_STORYBOARD_PROMPT_PIPELINE_FOR_SIDE_PANEL to false to switch back.\n${String(error?.message || error)}`, true);
      }
      return;
    }
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    saveMiniMaxSceneInputsFromPanel();
    const mode = miniMaxH3ModeForSegment(segment);
    const modeLabel = miniMaxH3ModeLabel(mode);
    const duration = Number(segment.end || 0) - Number(segment.start || 0);
    if (!Number.isFinite(duration) || duration <= 0) {
      toast("This scene needs valid start and end times before creating its MiniMax prompt.", true);
      return;
    }
    const projectFolder = activeProjectFolderForSave();
    if (!projectFolder) {
      toast("Create or load a Builder project before creating MiniMax prompts.", true);
      return;
    }
    const rendererPromptImages = miniMaxH3PromptVisionImages(segment, mode);
    const visionImages = miniMaxH3PromptVisionImagesForRunner(segment, mode);
    const rendererReferenceImages = ["reference_to_video", "image_reference_to_video"].includes(mode)
      ? miniMaxOrderedImageReferenceItemsForSegment(segment, mode)
      : [];
    const sceneImageUse = miniMaxH3SceneImageUseForSegment(segment);
    const sceneImageSourceAvailable = Boolean(segmentImageSource(segment)?.path || segmentImageSource(segment)?.data);
    if (mode === "image_to_video" && !rendererPromptImages.length) {
      toast("Image to Video needs a selected scene image before the LLM can create its MiniMax prompt.", true);
      return;
    }
    if (mode === "image_reference_to_video" && !sceneImageSourceAvailable) {
      toast("Image to Video 2 Pass needs a selected scene image before the LLM can create its MiniMax prompt.", true);
      return;
    }
    if (mode === "reference_to_video" && sceneImageUse !== "off" && !sceneImageSourceAvailable) {
      toast("The selected scene-image mode needs a timeline image for this scene before the LLM can create its MiniMax prompt.", true);
      return;
    }
    if (mode === "reference_to_video" && !rendererReferenceImages.length) {
      toast("Reference to Video needs at least one ordered Reference Builder image.", true);
      return;
    }
    try {
      assertMiniMaxH3ReferenceDescriptionsReady(segment, mode);
    } catch (error) {
      toast(String(error?.message || error), true);
      return;
    }
    const videoReferences = (Array.isArray(segment.minimax_h3_video_references) ? segment.minimax_h3_video_references : [])
      .filter((item) => String(item?.path || "").trim());
    if (mode === "video_to_video" && !videoReferences.length) {
      toast("Video to Video needs at least one reference video path.", true);
      return;
    }
    if (visionImages.length && state.textGemmaRunner === "llm_api" && !llmApiVisionModelSelected()) {
      toast("MiniMax image/reference prompting needs a vision-capable API model selected in LLM Runner.", true);
      return;
    }
    if (mode === "text_to_video" && !sceneVideoConceptPromptText(segment) && !String(segment.i2v_notes || segment.story_beat || segment.lyric_text || "").trim()) {
      toast("Add a scene idea, notes, story beat, lyrics, or motion direction before creating a Text to Video prompt.", true);
      return;
    }

    let progress = null;
    try {
      miniMaxCreatePromptButton.disabled = true;
      miniMaxCreatePromptButton.textContent = "Creating...";
      progress = createProgressWindow(`Creating MiniMax ${modeLabel} prompt`);
      progress.set(`Autosaving this scene before LLM prompting...`, 8);
      await autoSaveSessionQuiet(`MiniMax ${modeLabel} prompt`);
      progress.set(`Running ${visionImages.length ? "vision-assisted" : "text-only"} MiniMax prompt direction...\n${gemmaRunnerLine({ vision: Boolean(visionImages.length) })}`, 42);
      const data = await runMiniMaxH3PromptGeneration(segment, mode, {
        projectFolder,
        unloadAfter: true,
        finalizePrompt: (prompt) => ensureBuilderManagedFx(prompt, segment),
        emptyPromptMessage: `The LLM returned an empty MiniMax ${modeLabel} prompt.`,
      });
      pushHistory();
      segment.minimax_h3_prompt = data.prompt;
      segment.minimax_h3_prompt_origin = "gemma";
      miniMaxPrompt.value = segment.minimax_h3_prompt;
      updateMiniMaxPromptCharacterStatus(segment);
      render();
      await autoSaveSessionQuiet(`MiniMax ${modeLabel} prompt complete`);
      const auditPath = String(data.llm_request_audit_path || "").trim();
      progress.set(`MiniMax ${modeLabel} prompt ready.${auditPath ? `\n\nExact LLM request and raw response:\n${auditPath}` : ""}`, 100);
      progress.close(auditPath ? 5000 : 900);
      if (auditPath) console.info("[VRGDG Music Builder] Exact MiniMax H3 LLM request audit:", auditPath);
      toast(`Created the MiniMax ${modeLabel} prompt with ${gemmaRunnerLabel({ vision: Boolean(visionImages.length) })}.${auditPath ? `\nLLM request audit: ${auditPath}` : ""}`);
    } catch (error) {
      const debugPath = error?.gemmaDebugPath || await saveGemmaJunkDebug(error, { label: `MiniMax ${modeLabel} prompt`, segment });
      if (isRecoverableBuildGemmaError(error)) {
        const failure = recordGemmaBatchFailure(`minimax:${mode}:${segment.id}`, segment, sceneDisplayName(segment), error, debugPath);
        showGemmaBatchFailures([failure], {
          retryHandler: () => {
            if (activeSegment()?.id !== segment.id) throw new Error(`Select ${failure.sceneLabel} to retry it.`);
            return createMiniMaxH3PromptWithLLM();
          },
        });
      }
      progress?.set(`Error:\n${String(error?.message || error)}${debugPath ? `\n\nRaw LLM output saved to:\n${debugPath}` : ""}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      miniMaxCreatePromptButton.disabled = false;
      syncMiniMaxH3Panel();
    }
  }

  async function convertAllLtxVideoPromptsToMiniMaxH3(options = {}) {
    updateActiveFromInputs();
    if (normalizeVideoType(state.videoType) === "speaking") {
      toast("This converter is for music-video projects, not speaking / short-film projects.", true);
      return;
    }
    const failedIds = new Set((options.failedIds || []).map((value) => String(value)));
    const targets = allEditableSegments()
      .map((segment) => ({ segment, ltxPrompt: String(segment?.i2v_prompt || "").trim() }))
      .filter((item) => item.ltxPrompt && (!failedIds.size || failedIds.has(String(item.segment?.id || ""))));
    if (!targets.length) {
      toast("No populated LTX video prompts were found to convert.", true);
      return;
    }

    const projectFolder = activeProjectFolderForSave();
    if (!projectFolder) {
      toast("Create or load a Builder project before converting its LTX prompts.", true);
      return;
    }

    let progress = null;
    let converted = 0;
    let historySaved = false;
    const failures = [];
    try {
      convertLtxPromptsToMiniMaxButton.disabled = true;
      convertLtxPromptsToMiniMaxButton.textContent = "Converting...";
      state.batchCancelled = false;
      progress = createProgressWindow("Convert LTX Video Prompts to MiniMax H3");

      for (let index = 0; index < targets.length; index += 1) {
        assertBatchNotStopped();
        const { segment, ltxPrompt } = targets[index];
        const mode = miniMaxH3ModeForSegment(segment);
        const modeLabel = miniMaxH3ModeLabel(mode);
        const duration = Math.max(0, Number(segment.end || 0) - Number(segment.start || 0));
        const label = `${sceneDisplayName(segment, segmentIndexInfo(segment).index)} (${index + 1}/${targets.length})`;
        const percent = 5 + Math.round((index / Math.max(1, targets.length)) * 88);
        progress.set(`${failedIds.size ? "Retrying failed" : "Converting"} ${label}: existing LTX prompt to detailed MiniMax H3 ${modeLabel} format...\n${gemmaRunnerLine()}`, percent);

        try {
          const data = await runMiniMaxH3PromptGeneration(segment, mode, {
          projectFolder,
          unloadAfter: index === targets.length - 1,
          audioMode: "input_audio",
          performanceMode: effectiveVideoPerformanceModeForSegment(segment),
          contextOptions: {
            storyboardContext: [
              "Existing LTX video prompt to convert:",
              ltxPrompt,
              "",
              `Target scene duration: ${duration.toFixed(3)} seconds.`,
            ].join("\n"),
          },
          userNotes: [
            "Rewrite the existing LTX video prompt as one substantially more detailed MiniMax H3 shot-description plan for the selected mode.",
            "Preserve the same scene, subjects, setting, wardrobe, action, camera intent, performance, and ending. Do not invent a different scene or change the creative intent.",
            "Return only the required JSON shot-description payload. The Builder will assemble the final MiniMax H3 prompt sections, reference definitions, audio reuse, continuity, and safety blocks.",
          ].join("\n"),
          temperature: 0.4,
          topP: 0.92,
          maxNewTokens: 4000,
          emptyPromptMessage: `${label}: the LLM returned an empty MiniMax H3 prompt.`,
          });

          const prompt = String(data.prompt || "").trim();
          if (!prompt) throw new Error(`${label}: the LLM returned an empty MiniMax H3 prompt.`);
          if (!historySaved) {
            pushHistory();
            historySaved = true;
          }
          segment.minimax_h3_prompt = prompt;
          segment.minimax_h3_prompt_origin = "gemma";
          converted += 1;
        } catch (error) {
          if (!isRecoverableBuildGemmaError(error)) throw error;
          const debugPath = error?.gemmaDebugPath || await saveGemmaJunkDebug(error, { label, segment });
          failures.push(recordGemmaBatchFailure(`minimax-convert:${segment.id}`, segment, sceneDisplayName(segment), error, debugPath));
          progress.set(`${label} skipped. Continuing with the remaining prompts...`, percent);
        }
      }

      if (activeSegment()) {
        miniMaxPrompt.value = String(activeSegment().minimax_h3_prompt || "");
        updateMiniMaxPromptCharacterStatus(activeSegment());
      }
      ensureAllSegmentRuntimeFields();
      syncInspector();
      render();
      await autoSaveSessionQuiet("converted LTX video prompts to MiniMax H3");
      progress.set(`Converted ${converted} LTX video prompt${converted === 1 ? "" : "s"} to MiniMax H3.${failures.length ? ` ${failures.length} prompt${failures.length === 1 ? " was" : "s were"} skipped.` : ""} Global audio and original LTX prompts were not changed.`, 100);
      progress.close(1600);
      toast(`Converted ${converted} LTX video prompt${converted === 1 ? "" : "s"} to MiniMax H3${failures.length ? ` with ${failures.length} skipped prompt${failures.length === 1 ? "" : "s"}` : ""}.`, Boolean(failures.length));
      if (failures.length) showGemmaBatchFailures(failures, {
        retryHandler: (items) => convertAllLtxVideoPromptsToMiniMaxH3({ failedIds: items.map((item) => item.segmentId) }),
      });
    } catch (error) {
      progress?.set(`Conversion stopped after ${converted}/${targets.length} prompts:\n${String(error?.message || error)}`, 100);
      toast(`LTX to MiniMax H3 conversion stopped after ${converted}/${targets.length} prompts:\n${String(error?.message || error)}`, true);
    } finally {
      convertLtxPromptsToMiniMaxButton.disabled = false;
      convertLtxPromptsToMiniMaxButton.textContent = "Convert LTX Video Prompts to MiniMax H3";
    }
  }

  async function createI2VPromptWithGemma() {
    if (USE_STORYBOARD_PROMPT_PIPELINE_FOR_SIDE_PANEL && storyboardSidePanelPromptPipelineReady()) {
      const segment = requireActiveSegment();
      if (!segment) return;
      updateActiveFromInputs();
      try {
        await createSidePanelPromptFromStoryboardPipeline(segment, currentVideoMode(), "video");
        toast("Created the video prompt through the Storyboard pipeline.");
      } catch (error) {
        toast(`Storyboard side-panel prompt failed. Legacy path is preserved; set USE_STORYBOARD_PROMPT_PIPELINE_FOR_SIDE_PANEL to false to switch back.\n${String(error?.message || error)}`, true);
      }
      return;
    }
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const videoMode = currentVideoMode();
    const isT2V = videoMode === "t2v";
    const isRTV = videoMode === "rtv";
    const isFLF = videoMode === "flf";
    const isIngredients = videoMode === "ingredients";
    const isIdLora = videoMode === "id_lora";
    const modeLabel = videoModeDisplayLabel(videoMode, true);
    const textScriptMode = isRTV || isFLF || isIngredients || isIdLora;
    const useFirstLastFrameVision = isFLF || (isRTV && rtvReferenceBehaviorForSegment(segment) === "first_last_frame");
    const firstLastFrameReferences = useFirstLastFrameVision ? firstLastFramePromptReferences(segment) : [];
    const useImageReference = useFirstLastFrameVision ? true : isIdLora ? true : textScriptMode ? false : isT2V ? Boolean(segment.use_t2v_vision_reference) : segment.use_i2v_vision_reference !== false;
    const imageReference = useFirstLastFrameVision ? (firstLastFrameReferences[0] || { path: "", data: "" }) : useImageReference ? getI2VImageReference(segment) : { path: "", data: "" };
    const conceptPrompt = isFLF ? flfGemmaSceneConcept(segment) : sceneVideoConceptPromptText(segment);
    const idLoraContext = isIdLora ? idLoraSceneContext(segment) : null;
    if (useFirstLastFrameVision && firstLastFrameReferences.length < 2) {
      toast("First Last Frame needs a resolved start and end image before Gemma can write the transition prompt. With prompt pre-generation enabled, later scenes may use their assigned timeline images.", true);
      return;
    }
    if (useImageReference && !imageReference.path && !imageReference.data) {
      toast(isT2V
        ? "Hey, T2V Gemma image reference is on, but no reference image is loaded. Drop/load a reference image or turn it off."
        : "Hey, you need a scene image first. Save/load an image, or turn off image reference to create I2V from the T2I prompt instead.", true);
      return;
    }
    if (useImageReference && state.textGemmaRunner === "llm_api" && !llmApiVisionModelSelected()) {
      toast("Hey, LLM API vision needs a vision-capable API model selected. Open LLM Runner, choose an API model that supports images, then try again.", true);
      return;
    }
    if (!useFirstLastFrameVision && (isT2V || textScriptMode || !useImageReference) && !conceptPrompt && !(isIdLora && idLoraContext?.contextText)) {
      toast(isT2V || textScriptMode
        ? "Hey, this scene needs some scene text, notes, lyrics, or mapped references before Gemma can make a text-to-video prompt."
        : "Hey, you need scene text or a T2I prompt first, or turn image reference back on and use a saved/custom image.", true);
      return;
    }
    let progress = null;
    try {
      createI2VButton.disabled = true;
      createI2VButton.textContent = "Gemma...";
      progress = createProgressWindow(`Creating ${modeLabel} prompt`);
      progress.set(`Autosaving session/SRT before Gemma ${modeLabel}...`, 8);
      await autoSaveSessionQuiet(`Gemma ${modeLabel}`);
      progress.set(useImageReference ? "Preparing image reference and motion notes..." : "Preparing T2I prompt and motion notes...", 20);
      progress.set(useImageReference
        ? `Running Gemma vision ${modeLabel} prompt generation...\n${gemmaRunnerLine({ vision: true })}`
        : `Running Gemma text-only ${modeLabel} prompt generation...\n${gemmaRunnerLine()}`, 50);
      const data = await postJson(isT2V || textScriptMode ? "/vrgdg/music_builder/generate_t2v" : "/vrgdg/music_builder/generate_i2v", {
        ...textGemmaRunnerPayload(),
        project_folder: activeProjectFolderForSave(),
        scene_id: segment.id || "",
        builder_instruction_key: isIdLora ? "id_lora" : isIngredients ? "ingredients" : isFLF ? "i2v" : isRTV ? "rtv" : isT2V ? "t2v" : "i2v",
        model_file: useImageReference ? i2vGemmaModelSelect.value : i2vTextGemmaModelSelect.value,
        mmproj_file: useImageReference ? i2vMmprojSelect.value : "",
        t2i_prompt: isIdLora ? [conceptPrompt, idLoraContext?.contextText || ""].filter(Boolean).join("\n\n") : isT2V || textScriptMode ? conceptPrompt : useImageReference ? "" : conceptPrompt,
        performance_mode: effectiveVideoPerformanceModeForSegment(segment),
        image_reference_path: imageReference.path,
        image_reference_data: imageReference.data,
        image_references: firstLastFrameReferences,
        first_last_frame_mode: useFirstLastFrameVision,
        flf_context_mode: isFLF ? flfGemmaContextMode(segment) : "full",
        transition_lora_active: isFLF && flfTransitionLoraActive(segment),
        flf_start_state: isFLF ? String(segment.flf_start_state || "").trim() : "",
        flf_transformation: isFLF ? String(segment.flf_transformation || "").trim() : "",
        flf_end_state: isFLF ? String(segment.flf_end_state || "").trim() : "",
        flf_carry_forward: isFLF ? String(segment.flf_carry_forward || "").trim() : "",
        repair_model_file: i2vTextGemmaModelSelect.value,
        user_notes: isIdLora ? idLoraGemmaNotesForSegment(segment) : isFLF ? flfGemmaVisualNotes(segment) : videoGemmaNotesForSegment(segment),
        subject_context: isIdLora ? [idLoraContext?.characterName || "", String(idLoraContext?.character?.description || "").trim()].filter(Boolean).join("\n") : isFLF && flfGemmaContextMode(segment) !== "full" ? "" : segment.no_character_present ? "" : (isT2V || textScriptMode || !useImageReference ? segmentMappedSubjectText(segment) : ""),
        location_context: isIdLora ? [idLoraContext?.locationName || "", String(idLoraContext?.location?.description || "").trim()].filter(Boolean).join("\n") : isFLF && flfGemmaContextMode(segment) !== "full" ? "" : isT2V || textScriptMode || !useImageReference ? segmentMappedLocationText(segment) : "",
        no_character_present: Boolean(segment.no_character_present),
        theme_style_path: useImageReference && !isT2V && !textScriptMode ? "" : state.useVrgdgTextContext ? state.themeStylePath || "" : "",
        story_idea_path: useImageReference && !isT2V && !textScriptMode ? "" : state.useVrgdgTextContext ? state.storyIdeaPath || "" : "",
        subject_scene_path: useImageReference && !isT2V && !textScriptMode ? "" : state.useVrgdgTextContext ? state.subjectScenePath || "" : "",
        unload_after: !state.useI2VPromptEnhancementPass || useImageReference,
      }, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
      pushHistory();
      segment.i2v_prompt = ensureBuilderManagedFx(
        await finalizeVideoPromptForSegment(segment, data.prompt, progress, 82, `${modeLabel} prompt enhancement`, { unloadAfter: true }),
        segment,
      );
      segment.i2v_prompt_origin = "gemma";
      i2vPrompt.value = segment.i2v_prompt;
      render();
      await autoSaveSessionQuiet(`Gemma ${modeLabel} complete`);
      progress.set(`${modeLabel} prompt ready.`, 100);
      progress.close(900);
      toast(data.used_image_reference ? "Gemma created I2V prompt from the image reference." : `Gemma created ${modeLabel} prompt from the T2I prompt.`);
    } catch (error) {
      const debugPath = error?.gemmaDebugPath || await saveGemmaJunkDebug(error, { label: `single ${modeLabel} prompt`, segment });
      if (isRecoverableBuildGemmaError(error)) {
        const failure = recordGemmaBatchFailure(`i2v:${videoMode}:${segment.id}`, segment, sceneDisplayName(segment), error, debugPath);
        showGemmaBatchFailures([failure], {
          retryHandler: () => {
            if (activeSegment()?.id !== segment.id) throw new Error(`Select ${failure.sceneLabel} to retry it.`);
            return createI2VPromptWithGemma();
          },
        });
      }
      progress?.set(`Error:\n${String(error?.message || error)}${debugPath ? `\n\nRaw Gemma output saved to:\n${debugPath}` : ""}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      createI2VButton.disabled = false;
      syncVideoModePanel();
    }
  }

  async function generateTextOnlyI2VPromptForSegment(segment, progress = null, percent = 50, label = "Gemma I2V", options = {}) {
    if (!segment) throw new Error("Scene is missing.");
    const t2iText = sceneVideoConceptPromptText(segment);
    if (!t2iText) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: scene text/concept is missing.`);
    progress?.set(`${label}: converting T2I prompt to I2V prompt without vision...\n${gemmaRunnerLine()}`, percent);
    const data = await postJson("/vrgdg/music_builder/generate_i2v", {
      ...textGemmaRunnerPayload(),
      model_file: i2vTextGemmaModelSelect.value,
      mmproj_file: "",
      t2i_prompt: t2iText,
      performance_mode: effectiveVideoPerformanceModeForSegment(segment),
      image_reference_path: "",
      image_reference_data: "",
      repair_model_file: i2vTextGemmaModelSelect.value,
      user_notes: videoGemmaNotesForSegment(segment),
      subject_context: segment.no_character_present ? "" : segmentMappedSubjectText(segment),
      location_context: segmentMappedLocationText(segment),
      no_character_present: Boolean(segment.no_character_present),
      theme_style_path: state.useVrgdgTextContext ? state.themeStylePath || "" : "",
      story_idea_path: state.useVrgdgTextContext ? state.storyIdeaPath || "" : "",
      subject_scene_path: state.useVrgdgTextContext ? state.subjectScenePath || "" : "",
      unload_after: options.deferEnhancement ? options.unloadAfter !== false : state.useI2VPromptEnhancementPass ? false : options.unloadAfter !== false,
    }, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
    pushHistory();
    segment.i2v_prompt = options.deferEnhancement
      ? finalizeVideoPromptDraftOnly(segment, data.prompt)
      : await finalizeVideoPromptForSegment(segment, data.prompt, progress, Math.min(98, percent + 20), `${label}: enhancement pass`, { unloadAfter: options.unloadAfter !== false });
    if (!segment.i2v_prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: Gemma returned an empty I2V prompt.`);
    segment.i2v_prompt_origin = "gemma";
    if (segment.id === state.activeId) i2vPrompt.value = segment.i2v_prompt;
    render();
    return data;
  }

  async function generateI2VPromptForSegment(segment, progress = null, percent = 50, label = "Gemma I2V", options = {}) {
    await ensureAutoTimedSingerCuesBeforePrompt(segment);
    const request = buildI2VPromptRequestForSegment(segment, options);
    progress?.set(request.useImageReference
      ? `${label}: creating ${request.modeLabel} prompt from reference image, concept, and motion notes...\n${gemmaRunnerLine({ vision: true })}`
      : `${label}: converting T2I prompt to ${request.modeLabel} prompt without vision...\n${gemmaRunnerLine()}`, percent);
    const data = await postJson(request.endpoint, request.payload, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
    pushHistory();
    try {
      segment.i2v_prompt = options.deferEnhancement
        ? finalizeVideoPromptDraftOnly(segment, data.prompt)
        : await finalizeVideoPromptForSegment(segment, data.prompt, progress, Math.min(98, percent + 20), `${label}: enhancement pass`, { unloadAfter: request.useImageReference ? true : options.unloadAfter !== false });
    } catch (error) {
      error.rawGemmaPrompt = String(data.prompt || "");
      throw error;
    }
    if (!segment.i2v_prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: Gemma returned an empty ${request.modeLabel} prompt.`);
    segment.i2v_prompt_origin = "gemma";
    if (segment.id === state.activeId) i2vPrompt.value = segment.i2v_prompt;
    render();
    return data;
  }

  async function generateI2VPromptForSegmentWithFLFRetry(segment, progress = null, percent = 50, label = "Gemma FLF", options = {}) {
    const maxAttempts = Math.max(1, Math.min(5, Number(options.maxFLFAttempts || 3)));
    let lastError = null;
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      assertBatchNotStopped();
      try {
        const attemptLabel = attempt === 1 ? label : `${label} — retry ${attempt}/${maxAttempts}`;
        return await generateI2VPromptForSegment(segment, progress, percent, attemptLabel, options);
      } catch (error) {
        lastError = error;
        const message = String(error?.message || error || "");
        const incompleteObservation = /FLF Gemma vision observation was incomplete|both START and END descriptions are required/i.test(message);
        if (!incompleteObservation || attempt >= maxAttempts) throw error;
        progress?.set(
          `${label}: Gemma returned an incomplete START/END observation.\nRetrying this same scene from scratch (${attempt + 1}/${maxAttempts})...`,
          percent,
        );
        try {
          await runClearMemoryWorkflowQuiet(progress, `${label} incomplete observation attempt ${attempt}`, percent);
        } catch (cleanupError) {
          console.warn("[VRGDG Music Builder] FLF observation retry cleanup failed:", cleanupError);
        }
        await new Promise((resolve) => setTimeout(resolve, 750));
      }
    }
    throw lastError || new Error(`${label}: FLF vision prompt failed after ${maxAttempts} attempts.`);
  }

  async function generateIndependentFLFMotionPlanForSegment(segment, progress = null, percent = 50, label = "Independent FLF Motion Plan") {
    if (!segment) throw new Error("Scene is missing.");
    const startImage = segmentImageSource(segment);
    if (!startImage?.path && !startImage?.data) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: a start image is required before creating its motion plan.`);
    }
    const previousVideoPrompt = String(segment.i2v_prompt || "");
    const previousVideoPromptOrigin = normalizeVideoPromptOrigin(segment.i2v_prompt_origin);
    try {
      await generateI2VPromptForSegment(segment, progress, percent, label, {
        provisionalMotionPlan: true,
        forceVision: true,
        deferEnhancement: true,
        unloadAfter: true,
      });
      const plan = String(segment.i2v_prompt || "").trim();
      if (!plan) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: Gemma returned an empty motion plan.`);
      segment.flf_motion_plan = plan;
      segment.flf_end_frame_stale = hasFirstLastFrameEndImage(segment);
      segment.flf_final_prompt_ready = false;
      return plan;
    } finally {
      segment.i2v_prompt = previousVideoPrompt;
      segment.i2v_prompt_origin = previousVideoPromptOrigin;
      if (segment.id === state.activeId) i2vPrompt.value = previousVideoPrompt;
      syncRTVSceneImageAnchorPanel();
      render();
    }
  }

  async function generateFinalIndependentFLFPromptForSegment(segment, progress = null, percent = 82, label = "Final Independent FLF Prompt") {
    if (!segment) throw new Error("Scene is missing.");
    const refs = firstLastFramePromptReferences(segment);
    if (refs.length < 2) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: both the scene's own start image and end image are required before creating the final FLF prompt.`);
    }
    if (segment.flf_end_frame_stale) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: the ending direction or motion plan changed after this end image was created. Recreate the end frame before creating the final FLF prompt.`);
    }
    const motionPlan = String(segment.flf_motion_plan || "").trim();
    const customDirection = segment.flf_endpoint_mode === "custom" ? String(segment.flf_custom_end_direction || "").trim() : "";
    await generateI2VPromptForSegmentWithFLFRetry(segment, progress, percent, label, {
      unloadAfter: true,
      forceVision: true,
      maxFLFAttempts: 3,
      extraUserNotes: [
        motionPlan ? `ORIGINAL PROVISIONAL MOTION PLAN — preserve its intended action and camera path while adapting it to the two actual images:\n${motionPlan}` : "",
        customDirection ? `REQUIRED USER ENDING DIRECTION:\n${customDirection}` : "",
      ].filter(Boolean).join("\n\n"),
    });
    segment.flf_final_prompt_ready = true;
    if (segment.id === state.activeId) i2vPrompt.value = segment.i2v_prompt || "";
    syncRTVSceneImageAnchorPanel();
    render();
    return String(segment.i2v_prompt || "").trim();
  }

  async function i2vAllScenes(options = {}) {
    const videoMode = currentVideoMode();
    const isT2V = videoMode === "t2v";
    const isRTV = videoMode === "rtv";
    const isIngredients = videoMode === "ingredients";
    const isIdLora = videoMode === "id_lora";
    const modeLabel = videoModeDisplayLabel(videoMode, true);
    const runnerName = promptRunnerActionName();
    const progress = options.progress || createProgressWindow(`${runnerName} ${modeLabel} All Scenes`);
    const closeProgress = !options.progress;
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const allScenes = batchTargetItems(sceneScope).map(({ segment }) => segment);
    const redoPrompts = options.i2vRunMode === "redo_prompts";
    const failedIds = new Set((options.failedIds || []).map((value) => String(value)));
    const forceTextOnly = Boolean(options.forceTextOnly);
    const forceVision = Boolean(options.forceVision);
    const deferEnhancement = Boolean(state.useI2VPromptEnhancementPass);
    if (redoPrompts) {
      allScenes.forEach((segment) => {
        segment.i2v_prompt = "";
        segment.i2v_prompt_origin = "manual";
      });
    }
    const scenes = options.i2vRunMode === "failed_only"
      ? allScenes.filter((segment) => failedIds.has(String(segment?.id || "")))
      : allScenes.filter((segment) => redoPrompts || !String(segment?.i2v_prompt || "").trim());
    const missing = [];
    const continuityCanCreateImageLater = (segment, requestedUseImageReference) => {
      if (forceVision || forceTextOnly || !requestedUseImageReference || videoMode !== "i2v") return false;
      if (!i2vAutoChainEnabled() && !img2imgContinuityEnabled()) return false;
      if (!previousAutoChainSourceSegment(segment)) return false;
      const imageReference = getI2VImageReference(segment);
      return !imageReference.path && !imageReference.data;
    };
    if (!allScenes.length) missing.push(batchEmptyMessage(sceneScope));
    scenes.forEach((segment) => {
      const index = segmentIndexInfo(segment).index;
      const firstLastFrameVision = videoMode === "flf" || (isRTV && rtvReferenceBehaviorForSegment(segment) === "first_last_frame");
      const requestedUseImageReference = firstLastFrameVision ? true : isIdLora ? false : forceVision ? true : forceTextOnly ? false : videoVisionReferenceEnabled(segment);
      const forceTextForContinuity = continuityCanCreateImageLater(segment, requestedUseImageReference);
      const useImageReference = requestedUseImageReference && !forceTextForContinuity;
      const firstLastFrameReferences = firstLastFrameVision ? firstLastFramePromptReferences(segment) : [];
      const imageReference = firstLastFrameVision ? (firstLastFrameReferences[0] || { path: "", data: "" }) : useImageReference ? getI2VImageReference(segment) : { path: "", data: "" };
      if (firstLastFrameVision && firstLastFrameReferences.length < 2) {
        missing.push(`${sceneDisplayName(segment, index)}: First Last Frame needs a resolved start and end image; assign a timeline image or dedicated end frame.`);
      } else if (useImageReference && !imageReference.path && !imageReference.data) {
        missing.push(`${sceneDisplayName(segment, index)}: ${modeLabel} image reference is enabled, but no reference image was found.`);
      }
      if (useImageReference && state.textGemmaRunner === "llm_api" && !llmApiVisionModelSelected()) {
        missing.push(`${sceneDisplayName(segment, index)}: LLM API vision needs a vision-capable API model selected in LLM Runner.`);
      }
      if ((isRTV || isIngredients || isIdLora) && !sceneVideoConceptPromptText(segment)) {
        missing.push(`${sceneDisplayName(segment, index)}: ${modeLabel} prompt is missing. Export prompts from Storyboard Builder, or add scene notes/concept text before running ${runnerName}.`);
      } else if ((isT2V || !useImageReference) && !sceneVideoConceptPromptText(segment)) {
        missing.push(`${sceneDisplayName(segment, index)}: scene text/concept is missing.`);
      }
    });
    if (missing.length) {
      progress.setHtml(`
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:900;color:#fecaca;">${escapeHtml(runnerName)} ${escapeHtml(modeLabel)} All cannot start yet.</div>
          <div>Fix these first:</div>
          <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
        </div>
      `, 100);
      if (options.throwOnError) throw new Error(missing.join("\n"));
      return;
    }
    try {
      createI2VButton.disabled = true;
      progress.set(`Autosaving session/SRT before ${runnerName} ${modeLabel} All (${batchScopeLabel(sceneScope)})...`, 3);
      await saveSessionForSceneVideo();
      if (!scenes.length) {
        progress.set(`All scenes already have ${modeLabel} prompts. Skipping ${runnerName} ${modeLabel} All.`, 100);
        if (closeProgress) progress.close(1800);
        toast(`All scenes already have ${modeLabel} prompts. ${runnerName} ${modeLabel} skipped.`);
        return;
      }
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const segment = scenes[index];
        state.activeId = segment.id;
        syncInspector();
        render();
        const base = Math.floor((index / scenes.length) * (deferEnhancement ? 68 : 100));
        const firstLastFrameVision = videoMode === "flf" || (isRTV && rtvReferenceBehaviorForSegment(segment) === "first_last_frame");
        const requestedUseImageReference = firstLastFrameVision ? true : isIdLora ? false : forceVision ? true : forceTextOnly ? false : videoVisionReferenceEnabled(segment);
        const forceTextForContinuity = continuityCanCreateImageLater(segment, requestedUseImageReference);
        const useImageReference = requestedUseImageReference && !forceTextForContinuity;
        const displayIndex = segmentIndexInfo(segment).index;
        progress.set(`${runnerName} ${modeLabel} All ${index + 1}/${scenes.length}: ${sceneDisplayName(segment, displayIndex)}\nScope: ${batchScopeLabel(sceneScope)}\nBatch mode: ${firstLastFrameVision ? "first/last vision" : forceVision ? "vision" : forceTextOnly || forceTextForContinuity ? "text only" : "scene checkbox"}\n${firstLastFrameVision ? "Using first frame + end frame images plus scene/pacing notes." : useImageReference ? "Using image reference plus T2I prompt/motion notes." : forceTextForContinuity ? "Using T2I prompt text only; continuity will create this scene image during Render All." : "Using T2I prompt text only."}`, base);
        const generatePrompt = firstLastFrameVision
          ? generateI2VPromptForSegmentWithFLFRetry
          : generateI2VPromptForSegment;
        try {
          await generatePrompt(segment, progress, Math.min(deferEnhancement ? 70 : 98, base + 30), `${runnerName} ${modeLabel} All ${index + 1}/${scenes.length}`, {
            unloadAfter: false,
            forceTextOnly: forceTextOnly || forceTextForContinuity,
            forceVision,
            deferEnhancement,
            maxFLFAttempts: 3,
          });
          gemmaBatchFailureStore()[`i2v:${videoMode}:${segment.id}`] = undefined;
          await autoSaveSessionQuiet(`${runnerName} ${modeLabel} All ${sceneDisplayName(segment, displayIndex)}`);
        } catch (error) {
          if (!isRecoverableBuildGemmaError(error)) throw error;
          const debugPath = error?.gemmaDebugPath || await saveGemmaJunkDebug(error, { label: `${runnerName} ${modeLabel} All`, segment });
          recordGemmaBatchFailure(`i2v:${videoMode}:${segment.id}`, segment, sceneDisplayName(segment, displayIndex), error, debugPath);
          progress.set(`${runnerName} ${modeLabel} ${sceneDisplayName(segment, displayIndex)} skipped. Continuing with the remaining scenes...`, base);
        }
      }
      if (deferEnhancement) {
        await runClearMemoryWorkflowQuiet(progress, `${runnerName} ${modeLabel} draft prompt pass`, 72);
        await runVideoPromptEnhancementBatch(scenes, progress, {
          sceneScope,
          percentBase: 72,
          percentSpan: 23,
        });
      }
      await runClearMemoryWorkflowQuiet(progress, `${runnerName} ${modeLabel} prompt pass`, 96);
      await autoSaveSessionQuiet(`${runnerName} ${modeLabel} All complete`);
      const failures = scenes.map((segment) => gemmaBatchFailureStore()[`i2v:${videoMode}:${segment.id}`]).filter(Boolean);
      progress.set(`${runnerName} ${modeLabel} All complete. ${failures.length ? `${failures.length} scene${failures.length === 1 ? " was" : "s were"} skipped.` : "All scenes succeeded."}`, 100);
      if (closeProgress) progress.close(1800);
      if (failures.length) showGemmaBatchFailures(failures, { retryHandler: retryOnlyGemmaBatchFailures });
    } catch (error) {
      const debugPath = error?.gemmaDebugPath || await saveGemmaJunkDebug(error, { label: `${runnerName} ${modeLabel} All` });
      progress.set(`Stopped/Error:\n${String(error?.message || error)}${debugPath ? `\n\nRaw Gemma output saved to:\n${debugPath}` : ""}`, 100);
      if (options.throwOnError) throw error;
      toast(String(error?.message || error), true);
    } finally {
      createI2VButton.disabled = false;
    }
  }

  function promptAllModeTargets(promptRunMode = "missing_only", imageMode = state.imageModelMode || "zimage", sceneScope = "all") {
    const scenes = batchTargetItems(sceneScope);
    if (promptRunMode === "redo_all") return scenes;
    return scenes.filter(({ segment }) => {
      if (imageMode === "flux_klein") return !String(segment.flux_prompt || segment.t2i_prompt || "").trim();
      if (imageMode === "flow_gpt") return !String(segment.flow_gpt_prompt || segment.nb_prompt || segment.t2i_prompt || "").trim();
      if (imageMode === "nano_banana") return !String(segment.nb_prompt || segment.t2i_prompt || "").trim();
      return !String(segment.t2i_prompt || "").trim();
    });
  }

  async function gemmaT2IAllScenes(options = {}) {
    updateActiveFromInputs();
    const imageMode = state.imageModelMode || "zimage";
    const modelLabel = imageMode === "flux_klein" ? "Flux/Klein" : imageMode === "flow_gpt" ? "Flow/GPT" : imageMode === "nano_banana" ? "NanoBanana" : imageMode === "ernie_image" ? "Ernie" : "ZImage";
    const runnerName = promptRunnerActionName();
    const promptRunMode = options.promptRunMode || "redo_all";
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const redoPrompts = promptRunMode === "redo_all";
    const failedIds = new Set((options.failedIds || []).map((value) => String(value)));
    const allScenes = batchTargetItems(sceneScope);
    const targetScenes = promptRunMode === "failed_only"
      ? allScenes.filter(({ segment }) => failedIds.has(String(segment?.id || "")))
      : promptAllModeTargets(promptRunMode, imageMode, sceneScope);
    const progress = createProgressWindow(`${runnerName} T2I All (${modelLabel})`);
    const missing = [];
    if (!allScenes.length) missing.push(batchEmptyMessage(sceneScope));
    if (!String(projectInput.value || "").trim()) missing.push("Project folder is missing.");
    targetScenes.forEach(({ segment, index }) => {
      if (imageMode === "flux_klein") {
        return;
      } else if (imageMode === "nano_banana" || imageMode === "flow_gpt") {
        return;
      } else {
        const reason = t2iMissingReason(segment);
        if (reason) missing.push(`${sceneDisplayName(segment, index)}: ${reason}`);
      }
    });
    if (missing.length) {
      progress.setHtml(`
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:900;color:#fecaca;">${escapeHtml(runnerName)} T2I All cannot start yet.</div>
          <div>Fix these first:</div>
          <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
        </div>
      `, 100);
      toast(`${runnerName} T2I All needs scene inputs first.`, true);
      return;
    }
    try {
      state.batchCancelled = false;
      gemmaT2IAllButton.disabled = true;
      zImageAllButton.disabled = true;
      createT2IButton.disabled = true;
      ernieCreateT2IButton.disabled = true;
      krea2TwoPassCreateT2IButton.disabled = true;
      createFluxPromptButton.disabled = true;
      progress.set(`Autosaving session/SRT before ${runnerName} T2I All (${batchScopeLabel(sceneScope)})...`, 3);
      await saveSessionForSceneVideo();
      if (redoPrompts) {
        targetScenes.forEach(({ segment }) => {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
          segment.flow_gpt_prompt = "";
          segment.enhance_prompt = "";
        });
      }
      const promptScenes = redoPrompts ? targetScenes : promptAllModeTargets("missing_only", imageMode, sceneScope);
      if (!promptScenes.length) {
        progress.set("All scenes already have T2I prompts. Nothing to do.", 100);
        progress.close(1800);
        toast("All scenes already have T2I prompts.");
        return;
      }
      for (let index = 0; index < promptScenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = promptScenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 5 + Math.floor((index / promptScenes.length) * 88);
        state.activeId = segment.id;
        syncInspector();
        render();
        const promptLabel = `${runnerName} T2I All ${index + 1}/${promptScenes.length}: ${sceneLabel}`;
        try {
          await runGemmaImagePromptPassWithRetry(segment, progress, base, promptLabel, generateT2IPromptForSegment, {
          unloadAfter: false,
          imageMode,
          generatorOptions: { imageMode },
            textFallback: false,
          });
          gemmaBatchFailureStore()[`t2i:${imageMode}:${segment.id}`] = undefined;
          assertBatchNotStopped();
          await autoSaveSessionQuiet(`${runnerName} T2I All ${sceneLabel}`);
        } catch (error) {
          if (!isRecoverableBuildGemmaError(error)) throw error;
          const debugPath = error?.gemmaDebugPath || await saveGemmaJunkDebug(error, { label: promptLabel, segment });
          recordGemmaBatchFailure(`t2i:${imageMode}:${segment.id}`, segment, sceneLabel, error, debugPath);
          progress.set(`${promptLabel} skipped. Continuing with the remaining scenes...`, base);
        }
      }
      await runClearMemoryWorkflowQuiet(progress, `${runnerName} T2I prompt pass`, 96);
      await autoSaveSessionQuiet(`${runnerName} T2I All complete`);
      const failures = promptScenes.map(({ segment }) => gemmaBatchFailureStore()[`t2i:${imageMode}:${segment.id}`]).filter(Boolean);
      progress.set(`${runnerName} T2I All complete. ${failures.length ? `${failures.length} scene${failures.length === 1 ? " was" : "s were"} skipped.` : "All scenes succeeded."}\nReview or edit the image prompts before creating images.`, 100);
      progress.close(2500);
      toast(`${runnerName} T2I All complete${failures.length ? ` with ${failures.length} skipped scene${failures.length === 1 ? "" : "s"}` : ""}.`, Boolean(failures.length));
      if (failures.length) showGemmaBatchFailures(failures, { retryHandler: retryOnlyGemmaBatchFailures });
    } catch (error) {
      const message = String(error?.message || error);
      const stopped = /stopped by user/i.test(message);
      progress.set(`${stopped ? "Stopped" : "Error"}:\n${message}\n\nRunning memory cleanup...`, 100);
      toast(message, !stopped);
      try {
        const cleanupOutput = await runClearMemoryWorkflowQuiet(progress, stopped ? `stopped ${runnerName} T2I All` : `${runnerName} T2I All error`, 100);
        progress.set(`${stopped ? "Stopped" : "Error"}:\n${message}\n\n${cleanupOutput}`, 100);
      } catch (cleanupError) {
        console.warn(`[VRGDG Music Builder] Cleanup after ${runnerName} T2I All failed:`, cleanupError);
      }
    } finally {
      gemmaT2IAllButton.disabled = false;
      zImageAllButton.disabled = false;
      createT2IButton.disabled = false;
      ernieCreateT2IButton.disabled = false;
      krea2TwoPassCreateT2IButton.disabled = false;
      createFluxPromptButton.disabled = false;
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function retryOnlyGemmaBatchFailures(failures) {
    const items = Array.isArray(failures) ? failures : [];
    const t2i = items.filter((item) => String(item.key || "").startsWith("t2i:"));
    const i2v = items.filter((item) => String(item.key || "").startsWith("i2v:"));
    if (t2i.length) {
      await gemmaT2IAllScenes({ promptRunMode: "failed_only", failedIds: t2i.map((item) => item.segmentId), sceneScope: "all" });
      return;
    }
    if (i2v.length) {
      await i2vAllScenes({ i2vRunMode: "failed_only", failedIds: i2v.map((item) => item.segmentId), sceneScope: "all" });
    }
  }

  async function gemmaVideoAllTextOnly(options = {}) {
    updateActiveFromInputs();
    gemmaVideoAllButton.disabled = true;
    const changedReferenceFlags = [];
    const videoLabel = videoModeDisplayLabel(currentVideoMode(), true);
    const runnerName = promptRunnerActionName();
    const sceneScope = normalizeBatchScope(options.sceneScope);
    try {
      if (options.promptRunMode === "enhance_existing") {
        const progress = createProgressWindow(`${runnerName} ${videoLabel} Enhancement All`);
        try {
          const targets = batchTargetItems(sceneScope).map(({ segment }) => segment).filter((segment) => String(segment?.i2v_prompt || "").trim());
          if (!targets.length) {
            progress.set(`No existing ${videoLabel} prompts found to enhance.`, 100);
            progress.close(1800);
            toast(`No existing ${videoLabel} prompts found to enhance.`, true);
            return;
          }
          await saveSessionForSceneVideo();
          await runVideoPromptEnhancementBatch(targets, progress, {
            force: true,
            sceneScope,
            percentBase: 5,
            percentSpan: 88,
          });
          await runClearMemoryWorkflowQuiet(progress, `${runnerName} ${videoLabel} enhancement pass`, 96);
          await autoSaveSessionQuiet(`${runnerName} ${videoLabel} enhancement complete`);
          progress.set(`${runnerName} ${videoLabel} enhancement pass complete.`, 100);
          progress.close(1800);
          toast(`${runnerName} ${videoLabel} enhancement pass complete.`);
        } catch (error) {
          const debugPath = error?.gemmaDebugPath || await saveGemmaJunkDebug(error, { label: `${runnerName} ${videoLabel} Enhancement All` });
          progress.set(`Stopped/Error:\n${String(error?.message || error)}${debugPath ? `\n\nRaw Gemma output saved to:\n${debugPath}` : ""}`, 100);
          toast(String(error?.message || error), true);
        }
        return;
      }
      if (options.gemmaInputMode === "vision") {
        batchTargetItems(sceneScope).forEach(({ segment }) => {
          const previous = videoVisionReferenceEnabled(segment);
          if (!previous) {
            changedReferenceFlags.push({ segment, previous });
            setVideoVisionReferenceEnabled(segment, true);
          }
        });
      }
      await i2vAllScenes({
        i2vRunMode: options.promptRunMode === "missing_only" ? "missing_only" : "redo_prompts",
        forceTextOnly: options.gemmaInputMode !== "vision",
        forceVision: options.gemmaInputMode === "vision",
        sceneScope,
      }, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
    } finally {
      changedReferenceFlags.forEach(({ segment, previous }) => setVideoVisionReferenceEnabled(segment, previous));
      gemmaVideoAllButton.disabled = false;
    }
  }

  function storyboardSidePanelPromptPipelineReady() {
    return typeof state.onCreateVideoPrompt === "function"
      || typeof storyboardPipeline.runner === "function"
      || typeof ACTIVE_STORYBOARD_PROMPT_PIPELINE === "function";
  }
  async function createSidePanelPromptFromStoryboardPipeline(segment, mode, label = "Storyboard") {
    const storyboardScenes = storyboardScenePayload();
    const scene = storyboardScenes.find((item) => item.id === segment.id) || storyboardScenes.find((item) => Number(item.scene_number) === Number(segmentIndexInfo(segment).index + 1));
    if (!scene) throw new Error("The active scene could not be mapped into the Storyboard prompt pipeline.");
    const storyboardState = wizardStoryboardState(storyboardScenes, {
      promptMode: "video",
      videoMode: currentVideoMode(),
      miniMaxH3Mode: mode,
    });
    const storyboardPayload = storyboardGptPayload(storyboardState, [scene]);
    const progress = createProgressWindow(`Creating ${label} prompt`);
    try {
      const pipeline = typeof state.onCreateVideoPrompt === "function"
        ? state.onCreateVideoPrompt
        : (storyboardPipeline.runner || ACTIVE_STORYBOARD_PROMPT_PIPELINE);
      if (typeof pipeline !== "function") {
        throw new Error("The Storyboard prompt pipeline is not ready yet. Reload the Video Builder once, then try again.");
      }
      const data = await pipeline(scene, {
        storyboardPayload,
        progress,
      progressLabel: `Side panel → Storyboard pipeline (${label})`,
      progressPercent: 35,
      unloadAfter: true,
      });
      if (mode === "text_to_video" || mode === "image_to_video" || mode === "reference_to_video" || mode === "image_reference_to_video" || mode === "video_to_video") {
        segment.minimax_h3_prompt = data.prompt;
        segment.minimax_h3_prompt_origin = "gemma";
        miniMaxPrompt.value = data.prompt;
        updateMiniMaxPromptCharacterStatus(segment);
      } else {
        segment.i2v_prompt = data.prompt;
        segment.i2v_prompt_origin = "gemma";
        i2vPrompt.value = data.prompt;
      }
      render();
      await autoSaveSessionQuiet(`Storyboard pipeline ${label} prompt complete`);
      progress.set(`${label} prompt ready.`, 100);
      progress.close(900);
      return data.prompt;
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      progress.close(1800);
      throw error;
    }
  }

  return {
    convertAllLtxVideoPromptsToMiniMaxH3, createI2VPromptWithGemma, createMiniMaxH3PromptWithLLM,
    gemmaT2IAllScenes, gemmaVideoAllTextOnly, generateFinalIndependentFLFPromptForSegment,
    generateI2VPromptForSegment, generateIndependentFLFMotionPlanForSegment, i2vAllScenes,
    miniMaxBatchReferenceProblems, runMiniMaxH3PromptGeneration,
  };
}
