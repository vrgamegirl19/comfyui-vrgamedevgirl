import { speakingAudioEditsActive } from "./audio_clip_editor.mjs";
import { GEMMA_VIDEO_PROMPT_TIMEOUT_MS, postJson } from "./comfy_api.mjs";
import { escapeHtml, normalizeProjectVideoEngine, setWidgetValue, toast } from "./controls.mjs";
import { formatTime } from "./format.mjs";
import { sceneConceptPromptText, t2iMissingReason } from "./image_prompts.mjs";
import { miniMaxH3ModeLabel, normalizeMiniMaxH3Voice } from "./minimax_h3.mjs";
import { miniMaxDialogueAssignmentsForSegment } from "./minimax_prompt.mjs";
import { referenceBuilderSubjectHasImage } from "./minimax_references.mjs";
import {
  cloneErnieImageSettings,
  cloneI2VVideoSettings,
  cloneKrea2TwoPassSettings,
  cloneZImageSettings,
  defaultErnieImageSettings,
  defaultKrea2TwoPassSettings,
} from "./model_settings.mjs";
import { filmGrainLabel, lutLabelFromName, normalizeSceneFilmGrain, normalizeSceneLut } from "./post_process.mjs";
import {
  normalizeAutoImg2ImgCreativity,
  normalizeAutoImg2ImgStartStep,
  normalizeGemmaContextLimit,
} from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder } from "./reference_data.mjs";
import { selectedSegmentVideoPath } from "./selection_preview.mjs";
import {
  activateSegmentVideoPath,
  batchEmptyMessage,
  mediaPathKey,
  normalizeBatchScope,
  timelineSegmentDuration,
} from "./timeline_state.mjs";

function snapshotSceneImagePromptState(segment) {
  return {
    image: segment.image || null,
    image_history: Array.isArray(segment.image_history) ? [...segment.image_history] : [],
    image_history_index: Number(segment.image_history_index ?? -1),
    custom_image_path: segment.custom_image_path || "",
    custom_image_data: segment.custom_image_data || "",
    custom_image_name: segment.custom_image_name || "",
    approved_image_path: segment.approved_image_path || "",
    preview_mode: segment.preview_mode || "image",
    t2i_prompt: segment.t2i_prompt || "",
    flux_prompt: segment.flux_prompt || "",
    nb_prompt: segment.nb_prompt || "",
    flow_gpt_prompt: segment.flow_gpt_prompt || "",
    enhance_prompt: segment.enhance_prompt || "",
  };
}

function postProcessPathMatches(pathA = "", pathB = "") {
  return Boolean(String(pathA || "").trim() && String(pathB || "").trim() && mediaPathKey(pathA) === mediaPathKey(pathB));
}

export function renderMissingListHtml(missing) {
  return `
      <div style="display:flex;flex-direction:column;gap:10px;">
        <div style="font-weight:900;color:#fecaca;">Render All cannot start yet.</div>
        <div>Fix these first, then press Render All again:</div>
        <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
      </div>
    `;
}

export function createSceneRenderPrep({
  activeI2VVideoSettings, activeProjectFolderForSave, activeSegment, activeZImageSettings,
  addSceneImageHistoryPath, allEditableSegments, applyIngredientsSheetForSceneIfMapped, audioInput,
  autoSaveSessionQuiet, batchTargetItems, createErnieImageForSegment, createFlowGptImageForSegment,
  createFluxKleinImageForSegment, createKrea2TwoPassImageForSegment, createNBImageForSegmentWithRetry,
  createSilentTimelineAudioForDuration, createZImageForSegment, currentProjectAudioPath, currentVideoMode,
  effectiveVideoPerformanceModeForSegment, ensureSegmentRuntimeFields, finalizeVideoPromptForSegment,
  firstLastFrameEndImageSource, firstLastFrameResolvedEndImageSource, firstLastFrameStartImageSource,
  flfChainingEnabled, flfRenderChainStartSource, flfSameLocationCameraDiversityDirection, gemmaRunnerLine,
  generateFluxKleinPromptForSegment, generateI2VPromptForSegment, generateNBPromptForSegment,
  generateT2IPromptForSegment, hasFirstLastFrameEndImage, i2vAutoChainEnabled, i2vGemmaModelSelect,
  i2vMmprojSelect, i2vPrompt, i2vTextGemmaModelSelect, idLoraSceneContext, imageModeDisplayLabel,
  imageModeImg2ImgContinuityLabel, imageModeSupportsImg2ImgContinuity, img2imgContinuityEnabled,
  miniMaxBatchReferenceProblems, miniMaxH3ContinuityModeForSegment, miniMaxH3ContinuityReferenceReserved,
  miniMaxH3FrameContinuityPromptEnabled, miniMaxH3ModeForSegment, miniMaxH3SettingsForSegment,
  miniMaxOrderedImageReferenceItemsForSegment, node, normalizeSceneAdjust, projectInput, pushHistory,
  referenceBuilderSubjectItemsForSegment, render, rtvReferenceBehaviorForSegment,
  rtvReferenceBehaviorGlobalValue, rtvReferencesForSegment, sceneAdjustHasRenderableChanges,
  sceneAdjustSignature, sceneDisplayName, sceneSlotNumber, segmentImageSource, segmentIndexInfo,
  segmentMappedLocationText, segmentMappedSubjectText, segmentTrack, selectedSegmentImagePath, srtInput,
  state, storyboardReferenceDataForSegment, syncErnieImagePanel, syncI2VVideoSettingsPanel, syncInspector,
  syncKrea2TwoPassPanel, syncPreview, syncRTVSceneImageAnchorPanel, syncSegmentFlowGptPrompt,
  syncSegmentT2IPrompt, syncZImageSettingsPanel, textGemmaRunnerPayload, timelineDuration,
  usingSceneAudioMode, videoGemmaNotesForSegment,
}) {
  function validateSceneReadyForVideo(segment, sceneIndex) {
    const name = sceneDisplayName(segment, sceneIndex);
    const missing = [];
    const mode = currentVideoMode();
    const selectedLtxVersion = activeI2VVideoSettings()?.ltx_version || "2.5";
    if (selectedLtxVersion === "2.5" && !["i2v", "id_lora", "t2v", "rtv", "ingredients", "flf"].includes(mode)) {
      missing.push(`${name}: ${mode.toUpperCase()} is not available with LTX 2.5. Select LTX 2.3 (legacy) in Builder Settings for this mode.`);
    }
    const promptLabel = mode === "id_lora" ? "ID-LoRA I2V" : mode === "ingredients" ? "Ingredients to Video" : mode === "flf" ? "First Last Frame" : mode === "rtv" ? "Reference to Video" : mode === "t2v" ? "T2V" : "I2V";
    if (mode === "ingredients") applyIngredientsSheetForSceneIfMapped(segment);
    if ((mode === "i2v" || mode === "ingredients" || mode === "id_lora") && !segmentImageSource(segment)) missing.push(`${name}: selected scene image is missing.`);
    if (mode === "id_lora") {
      const idContext = idLoraSceneContext(segment);
      if (!idContext.dialogue) missing.push(`${name}: ID-LoRA dialogue line is missing in the ID-LoRA Ref Builder.`);
      if (!idContext.voicePath) missing.push(`${name}: ID-LoRA character reference voice sample is missing.`);
    }
    if (mode === "rtv") {
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      const rtvReferences = rtvReferencesForSegment(segment);
      if (rtvReferenceBehaviorForSegment(segment) === "first_last_frame") {
        const firstFrame = segmentImageSource(segment);
        const lastFrame = firstLastFrameEndImageSource(segment);
        if (!firstFrame?.path && !firstFrame?.data) missing.push(`${name}: First Last Frame needs a selected scene image for the first frame.`);
        if (!lastFrame?.path && !lastFrame?.data) missing.push(`${name}: First Last Frame needs a generated/stored end frame.`);
      } else {
        const hasSubjectReference = Boolean(rtvReferences.use_subject_placeholder)
          || referenceBuilderSubjectItemsForSegment(refs, segment).some(referenceBuilderSubjectHasImage);
        if (!hasSubjectReference) missing.push(`${name}: Reference to Video needs a subject image in Reference Builder.`);
      }
    }
    if (mode === "flf") {
      const previous = previousAutoChainSourceSegment(segment);
      const firstFrame = firstLastFrameStartImageSource(segment);
      const lastFrame = firstLastFrameResolvedEndImageSource(segment);
      if (!firstFrame?.path && !firstFrame?.data) {
        if (!(previous && flfChainingEnabled(segment) && flfRenderChainStartSource(segment) === "rendered_frame")) {
          missing.push(previous && flfChainingEnabled(segment)
            ? `${name}: the previous scene needs an assigned end image to use as this scene's first frame.`
            : `${name}: First Last Frame needs a selected first-frame image.`);
        }
      }
      if (!lastFrame?.path && !lastFrame?.data) missing.push(`${name}: First Last Frame needs its own end image or assigned scene image.`);
    }
    if (!String(segment?.i2v_prompt || "").trim()) missing.push(`${name}: ${promptLabel} prompt is missing.`);
    return missing;
  }

  function validateMiniMaxSceneReadyForVideo(segment, sceneIndex) {
    const name = sceneDisplayName(segment, sceneIndex);
    const missing = [];
    const mode = miniMaxH3ModeForSegment(segment);
    const savedPrompt = String(segment?.minimax_h3_prompt || segment?.i2v_prompt || "").trim();
    const frameContinuityPromptEnabled = miniMaxH3FrameContinuityPromptEnabled(segment);
    if (!frameContinuityPromptEnabled) {
      if (!savedPrompt) {
        missing.push(`${name}: MiniMax ${miniMaxH3ModeLabel(mode)} prompt is missing.`);
      } else if (savedPrompt.length > 7000) {
        missing.push(`${name}: MiniMax prompt is ${savedPrompt.length.toLocaleString()} characters and exceeds the 7,000-character maximum. Regenerate or shorten it before rendering.`);
      }
    }
    if (!Number.isFinite(Number(segment?.start)) || !Number.isFinite(Number(segment?.end)) || Number(segment.end) <= Number(segment.start)) {
      missing.push(`${name}: timeline start and end times are invalid.`);
    }
    if (mode === "image_to_video" && !String(selectedSegmentImagePath(segment) || "").trim()) {
      missing.push(`${name}: MiniMax Image to Video needs a selected scene image.`);
    }
    if (["reference_to_video", "image_reference_to_video"].includes(mode) && segment?.minimax_h3_use_scene_image_as_start_frame && !String(selectedSegmentImagePath(segment) || "").trim()) {
      missing.push(`${name}: MiniMax Reference to Video is set to use the scene image as its start frame, but no scene image is saved.`);
    }
    const continuityMode = miniMaxH3ContinuityModeForSegment(segment);
    const continuityReserved = miniMaxH3ContinuityReferenceReserved(segment);
    if (continuityMode === "exact_start_frame" && segment?.minimax_h3_use_scene_image_as_start_frame) {
      missing.push(`${name}: choose either the scene image or the previous rendered final frame as the exact start frame, not both.`);
    }
    const orderedReferenceCount = miniMaxOrderedImageReferenceItemsForSegment(segment, mode).length;
    if (orderedReferenceCount + (continuityReserved ? 1 : 0) > 9) {
      missing.push(`${name}: MiniMax continuity needs one image slot. Remove one Reference Builder image so the total stays at nine.`);
    }
    if (mode === "reference_to_video" && !orderedReferenceCount && !continuityReserved) {
      missing.push(`${name}: MiniMax Reference to Video needs a start frame or at least one ordered Reference Builder image.`);
    }
    if (mode === "video_to_video" && !(segment?.minimax_h3_video_references || []).some((item) => String(item?.path || "").trim())) {
      missing.push(`${name}: MiniMax Video to Video needs at least one reference-video path.`);
    }
    const settings = miniMaxH3SettingsForSegment(segment);
    if (settings.audio_mode === "built_in_audio" && effectiveVideoPerformanceModeForSegment(segment) === "speaking") {
      const mappedSubjects = storyboardReferenceDataForSegment(segment).subject_refs || [];
      const mappedIds = new Set(mappedSubjects.map((subject) => String(subject.id || "")).filter(Boolean));
      const dialogueCues = miniMaxDialogueAssignmentsForSegment(segment);
      const speakingIds = new Set(dialogueCues.map((cue) => String(cue.speaker_id || "")).filter(Boolean));
      for (const cue of dialogueCues) {
        if (!cue.speaker_id || !mappedIds.has(String(cue.speaker_id))) {
          missing.push(`${name}: dialogue cue “${cue.text.slice(0, 80)}${cue.text.length > 80 ? "..." : ""}” needs a speaker currently mapped to this scene.`);
        }
      }
      for (const subject of mappedSubjects) {
        if (dialogueCues.length && !speakingIds.has(String(subject.id || ""))) continue;
        const voice = normalizeMiniMaxH3Voice(subject.minimax_voice);
        if (voice.preset_id.endsWith("_custom") && (!voice.preset_name || !voice.description)) {
          missing.push(`${name}: ${subject.name || "a mapped character"}'s custom MiniMax voice needs both an exact preset name and an exact voice description in Reference Builder.`);
        }
      }
    }
    return missing;
  }

  async function validateSrtTimingForSceneVideo({ segment, sceneIndex, srtPath, promptNumber, expectedDuration }) {
    const promptIndex = Math.max(0, Number(promptNumber || 1) - 1);
    const uiDuration = Number(expectedDuration ?? timelineSegmentDuration(segment));
    const data = await postJson("/vrgdg/music_builder/load_srt", { srt_path: srtPath }, 60000);
    const srtSegment = Array.isArray(data.segments) ? data.segments[promptIndex] : null;
    if (!srtSegment) {
      throw new Error(
        `SRT timing check failed before creating video.\n\n` +
        `${sceneDisplayName(segment, sceneIndex)} is using prompt #${promptNumber}, but that prompt was not found in:\n${srtPath}`
      );
    }
    const srtDuration = Math.max(0, Number(srtSegment.end || 0) - Number(srtSegment.start || 0));
    const diff = Math.abs(srtDuration - uiDuration);
    console.log("[VRGDG Music Builder] SRT timing check", {
      scene: sceneIndex + 1,
      promptNumber,
      srtPath,
      uiStart: Number(segment.start || 0),
      uiEnd: Number(segment.end || 0),
      uiDuration,
      srtStart: Number(srtSegment.start || 0),
      srtEnd: Number(srtSegment.end || 0),
      srtDuration,
      diff,
    });
    if (!Number.isFinite(uiDuration) || uiDuration <= 0) {
      throw new Error(`${sceneDisplayName(segment, sceneIndex)} has an invalid UI duration: ${uiDuration}`);
    }
    if (!Number.isFinite(srtDuration) || srtDuration <= 0) {
      throw new Error(`SRT prompt #${promptNumber} has an invalid duration in:\n${srtPath}`);
    }
    if (diff > 0.25) {
      throw new Error(
        `SRT timing mismatch before creating video.\n\n` +
        `${sceneDisplayName(segment, sceneIndex)} UI duration: ${uiDuration.toFixed(3)}s\n` +
        `SRT prompt #${promptNumber} duration: ${srtDuration.toFixed(3)}s\n\n` +
        `UI timing: ${Number(segment.start || 0).toFixed(3)} -> ${Number(segment.end || 0).toFixed(3)}\n` +
        `SRT timing: ${Number(srtSegment.start || 0).toFixed(3)} -> ${Number(srtSegment.end || 0).toFixed(3)}\n\n` +
        `SRT path being sent to hidden workflow:\n${srtPath}\n\n` +
        `Stopping before LTX runs so it cannot accidentally render the wrong/huge clip.`
      );
    }
    return { srt_segment: srtSegment, srt_duration: srtDuration, ui_duration: uiDuration, srt_path: data.srt_path || srtPath };
  }

  function forceAutoChainWarmupFrames(segment) {
    if (!segment) return;
    const settings = segment.use_scene_i2v_video_settings
      ? cloneI2VVideoSettings(segment.i2v_video_settings || state.i2vVideoSettings)
      : cloneI2VVideoSettings(state.i2vVideoSettings);
    settings.pre_frames = 1;
    segment.use_scene_i2v_video_settings = true;
    segment.i2v_video_settings = settings;
    segment.auto_chain_pre_frames = 1;
    if (segment.id === activeSegment()?.id) syncI2VVideoSettingsPanel();
  }

  function autoChainReferenceContextForSegment(segment) {
    const references = storyboardReferenceDataForSegment(segment);
    const subjectLines = Array.isArray(references.subject_refs)
      ? references.subject_refs.map((subject) => {
        const name = String(subject?.name || "").trim();
        const description = String(subject?.description || "").trim();
        const trigger = String(subject?.trigger_phrase || "").trim();
        return [name, description, trigger ? `trigger: ${trigger}` : ""].filter(Boolean).join(" - ");
      }).filter(Boolean)
      : [];
    const location = references.location_ref || null;
    const locationText = location
      ? [
        String(location.name || "").trim(),
        String(location.description || "").trim(),
        String(location.trigger_phrase || "").trim() ? `trigger: ${String(location.trigger_phrase || "").trim()}` : "",
      ].filter(Boolean).join(" - ")
      : "";
    return {
      ...references,
      subject_context: subjectLines.join("\n"),
      location_context: locationText,
    };
  }

  function autoChainSceneContextForSegment(segment) {
    const referenceContext = autoChainReferenceContextForSegment(segment);
    return {
      scene_concept: sceneConceptPromptText(segment),
      motion_notes: videoGemmaNotesForSegment(segment),
      scene_notes: String(segment?.notes || "").trim(),
      director_note: String(segment?.timeline_note || segment?.director_note || "").trim(),
      story_beat: String(segment?.story_beat || segment?.beat || "").trim(),
      lyric_text: String(segment?.lyric_text || segment?.lyrics || "").trim(),
      lyric_section: String(segment?.lyric_section || segment?.section || "").trim(),
      mapped_subject_context: segmentMappedSubjectText(segment),
      mapped_location_context: segmentMappedLocationText(segment),
      no_character_present: Boolean(segment?.no_character_present),
      reference_context: referenceContext,
    };
  }

  async function prepareAutoChainedNextScene(previousSegment, nextSegment, progress = null, percent = 50, label = "Auto Chain") {
    if (!previousSegment || !nextSegment) return null;
    const projectFolder = projectInput.value || state.projectFolder;
    if (!projectFolder) throw new Error("Project folder is missing.");
    const previousVideoPath = selectedSegmentVideoPath(previousSegment);
    if (!previousVideoPath) throw new Error(`${sceneDisplayName(previousSegment, segmentIndexInfo(previousSegment).index)} has no rendered video for auto-chain.`);
    const nextIndex = segmentIndexInfo(nextSegment).index;
    progress?.set(`${label}: extracting final frame for ${sceneDisplayName(nextSegment, nextIndex)}...`, percent);
    const extracted = await postJson("/vrgdg/music_builder/extract_video_final_frame", {
      project_folder: projectFolder,
      source_path: previousVideoPath,
      scene_number: sceneSlotNumber(nextSegment),
    }, 120000);
    const framePath = String(extracted.saved_path || "").trim();
    if (!framePath) throw new Error("Final frame extraction did not return an image path.");
    pushHistory();
    addSceneImageHistoryPath(nextSegment, framePath);
    nextSegment.approved_image_path = "";
    nextSegment.custom_image_path = "";
    nextSegment.custom_image_data = "";
    nextSegment.custom_image_name = "";
    nextSegment.image = null;
    forceAutoChainWarmupFrames(nextSegment);
    const chainContext = autoChainSceneContextForSegment(nextSegment);
    progress?.set(`${label}: creating chained I2V prompt for ${sceneDisplayName(nextSegment, nextIndex)}...\n${gemmaRunnerLine({ vision: true })}`, Math.min(98, percent + 8));
    const data = await postJson("/vrgdg/music_builder/generate_chained_i2v", {
      ...textGemmaRunnerPayload(),
      project_folder: activeProjectFolderForSave(),
      scene_id: nextSegment.id || "",
      model_file: i2vGemmaModelSelect.value,
      mmproj_file: i2vMmprojSelect.value,
      image_reference_path: framePath,
      image_reference_data: "",
      scene_context: chainContext.scene_concept,
      user_notes: chainContext.motion_notes,
      scene_notes: chainContext.scene_notes,
      director_note: chainContext.director_note,
      story_beat: chainContext.story_beat,
      lyric_text: chainContext.lyric_text,
      lyric_section: chainContext.lyric_section,
      subject_context: chainContext.mapped_subject_context,
      location_context: chainContext.mapped_location_context,
      no_character_present: chainContext.no_character_present,
      reference_context: chainContext.reference_context,
      chain_style: state.autoChainStyle || "continuous",
      chain_direction: state.autoChainDirection || "",
      transition_lora_prompt: Boolean(state.autoChainTransitionLoraPrompt),
      transition_lora_trigger: state.autoChainTransitionTrigger || "zhuanchang",
      performance_mode: effectiveVideoPerformanceModeForSegment(nextSegment),
      repair_model_file: i2vTextGemmaModelSelect.value,
      unload_after: true,
      n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
      temperature: 0.25,
      top_p: 0.9,
      max_new_tokens: 1200,
    }, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
    const prompt = String(data.prompt || "").trim();
    if (!prompt) throw new Error("Gemma returned an empty chained I2V prompt.");
    nextSegment.i2v_prompt = await finalizeVideoPromptForSegment(nextSegment, prompt, progress, Math.min(99, percent + 14), `${label}: prompt cleanup`, {
      unloadAfter: true,
      suppressVocalPrefix: Boolean(state.autoChainTransitionLoraPrompt),
    });
    nextSegment.i2v_prompt_origin = "gemma";
    nextSegment.auto_chain_source_video_path = previousVideoPath;
    nextSegment.auto_chain_source_frame_path = framePath;
    nextSegment.auto_chain_style = state.autoChainStyle || "continuous";
    nextSegment.auto_chain_direction = state.autoChainDirection || "";
    if (nextSegment.id === state.activeId) {
      i2vPrompt.value = nextSegment.i2v_prompt;
      syncPreview(nextSegment);
    }
    render();
    return { framePath, prompt: nextSegment.i2v_prompt };
  }

  function setSegmentImg2ImgContinuitySource(segment, imageMode, framePath) {
    if (!segment || !framePath) return null;
    const imageName = String(framePath || "").split(/[\\/]/).pop() || "previous_final_frame.png";
    if (imageMode === "ernie_image") {
      const settings = cloneErnieImageSettings(segment.use_scene_ernie_image_settings ? segment.ernie_image_settings : state.ernieImageSettings);
      settings.use_image_to_image = true;
      settings.image_to_image_start_at_step = normalizeAutoImg2ImgStartStep(state.autoImg2ImgStartStep);
      settings.image_to_image_path = framePath;
      settings.image_to_image_data = "";
      settings.image_to_image_name = imageName;
      segment.use_scene_ernie_image_settings = true;
      segment.ernie_image_settings = settings;
      return settings;
    }
    if (imageMode === "krea2_2pass") {
      const settings = cloneKrea2TwoPassSettings(segment.use_scene_krea2_2pass_settings ? segment.krea2_2pass_settings : state.krea2TwoPassSettings);
      settings.use_image_to_image = true;
      settings.image_to_image_creativity = normalizeAutoImg2ImgCreativity(state.autoImg2ImgCreativity);
      settings.image_to_image_path = framePath;
      settings.image_to_image_data = "";
      settings.image_to_image_name = imageName;
      segment.use_scene_krea2_2pass_settings = true;
      segment.krea2_2pass_settings = settings;
      return settings;
    }
    const settings = cloneZImageSettings(segment.use_scene_zimage_settings ? segment.zimage_settings : state.zimageSettings);
    settings.use_image_to_image = true;
    settings.image_to_image_start_at_step = normalizeAutoImg2ImgStartStep(state.autoImg2ImgStartStep);
    settings.image_to_image_path = framePath;
    settings.image_to_image_data = "";
    settings.image_to_image_name = imageName;
    segment.use_scene_zimage_settings = true;
    segment.zimage_settings = settings;
    return settings;
  }

  async function prepareAutoImg2ImgContinuityForScene(previousSegment, nextSegment, imageMode = state.imageModelMode || "zimage", progress = null, percent = 50, label = "Img2Img Continuity") {
    if (!previousSegment || !nextSegment) return null;
    if (!imageModeSupportsImg2ImgContinuity(imageMode)) {
      throw new Error(`Img2Img continuity is not available for ${imageModeImg2ImgContinuityLabel(imageMode)} yet. Choose ZImage, Ernie, or Krea 2 for the image model, or turn continuity Off.`);
    }
    const projectFolder = projectInput.value || state.projectFolder;
    if (!projectFolder) throw new Error("Project folder is missing.");
    const previousVideoPath = selectedSegmentVideoPath(previousSegment);
    if (!previousVideoPath) throw new Error(`${sceneDisplayName(previousSegment, segmentIndexInfo(previousSegment).index)} has no rendered video for Img2Img continuity.`);
    const nextIndex = segmentIndexInfo(nextSegment).index;
    progress?.set(`${label}: extracting final frame for ${sceneDisplayName(nextSegment, nextIndex)}...`, percent);
    const extracted = await postJson("/vrgdg/music_builder/extract_video_final_frame", {
      project_folder: projectFolder,
      source_path: previousVideoPath,
      scene_number: sceneSlotNumber(nextSegment),
    }, 120000);
    const framePath = String(extracted.saved_path || "").trim();
    if (!framePath) throw new Error("Final frame extraction did not return an image path.");
    pushHistory();
    setSegmentImg2ImgContinuitySource(nextSegment, imageMode, framePath);
    nextSegment.auto_img2img_source_video_path = previousVideoPath;
    nextSegment.auto_img2img_source_frame_path = framePath;
    nextSegment.auto_img2img_image_mode = imageMode;
    if (nextSegment.id === state.activeId) {
      syncZImageSettingsPanel();
      syncErnieImagePanel();
      syncKrea2TwoPassPanel();
      syncInspector();
    }
    render();
    return { framePath };
  }

  function canAutoChainFromPreviousRenderedScene(segment) {
    const previousSegment = previousAutoChainSourceSegment(segment);
    return Boolean(previousSegment && String(selectedSegmentVideoPath(previousSegment) || "").trim());
  }

  function canImg2ImgContinuityFromPreviousRenderedScene(segment) {
    const previousSegment = previousAutoChainSourceSegment(segment);
    return Boolean(previousSegment && String(selectedSegmentVideoPath(previousSegment) || "").trim());
  }

  function validateSceneReadyForAutoImg2ImgContinuity(segment, sceneIndex, imageMode = state.imageModelMode || "zimage") {
    const missing = [];
    if (!imageModeSupportsImg2ImgContinuity(imageMode)) {
      missing.push(`${sceneDisplayName(segment, sceneIndex)}: Img2Img continuity needs ZImage, Ernie, or Krea 2 image mode.`);
    }
    const prompt = imageMode === "krea2_2pass" || imageMode === "ernie_image" || imageMode === "zimage"
      ? String(segment?.t2i_prompt || "").trim()
      : "";
    if (!prompt) missing.push(`${sceneDisplayName(segment, sceneIndex)}: image prompt is missing.`);
    if (!canImg2ImgContinuityFromPreviousRenderedScene(segment)) {
      missing.push(`${sceneDisplayName(segment, sceneIndex)}: previous scene has no rendered video for Img2Img continuity.`);
    }
    if (!String(segment?.i2v_prompt || "").trim()) missing.push(`${sceneDisplayName(segment, sceneIndex)}: I2V prompt is missing.`);
    return missing;
  }

  async function createImageForSegmentInCurrentMode(segment, imageMode, progress, percentBase, percentSpan, label, options = {}) {
    if (imageMode === "ernie_image") {
      await createErnieImageForSegment(segment, progress, percentBase, percentSpan, `${label}: Ernie image`, options);
    } else if (imageMode === "krea2_2pass") {
      await createKrea2TwoPassImageForSegment(segment, progress, percentBase, percentSpan, `${label}: Krea 2 image`, options);
    } else if (imageMode === "flux_klein") {
      await createFluxKleinImageForSegment(segment, progress, percentBase, percentSpan, `${label}: Flux/Klein image`);
    } else if (imageMode === "nano_banana") {
      await createNBImageForSegmentWithRetry(segment, progress, percentBase, percentSpan, `${label}: NanoBanana image`, { maxRetries: 10 });
    } else if (imageMode === "flow_gpt") {
      await createFlowGptImageForSegment(segment, progress, percentBase, percentSpan, `${label}: Flow/GPT image`, options);
    } else {
      await createZImageForSegment(segment, progress, percentBase, percentSpan, `${label}: ZImage`, options);
    }
  }

  async function prepareFLFRenderedFrameNextScene(previousSegment, nextSegment, progress = null, percent = 50, label = "FLF Render Chain", options = {}) {
    if (!previousSegment || !nextSegment) return null;
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    const previousVideoPath = String(selectedSegmentVideoPath(previousSegment) || "").trim();
    if (!projectFolder) throw new Error("Project folder is missing.");
    if (!previousVideoPath) throw new Error(`${sceneDisplayName(previousSegment, segmentIndexInfo(previousSegment).index)} has no rendered video to extract its end frame.`);
    progress?.set(`${label}: extracting the actual final video frame...`, percent);
    const extracted = await postJson("/vrgdg/music_builder/extract_video_final_frame", {
      project_folder: projectFolder,
      source_path: previousVideoPath,
      scene_number: sceneSlotNumber(nextSegment),
    }, 120000);
    const framePath = String(extracted.saved_path || "").trim();
    if (!framePath) throw new Error("Rendered final-frame extraction did not return an image path.");
    pushHistory();
    nextSegment.flf_rendered_start_frame_path = framePath;
    nextSegment.flf_rendered_start_frame_data = "";
    nextSegment.flf_rendered_start_frame_name = String(framePath).split(/[\\/]/).pop() || "rendered_previous_end.png";
    nextSegment.flf_rendered_source_video_path = previousVideoPath;
    const shouldRefreshPrompt = options.refreshPrompt === true || !String(nextSegment.i2v_prompt || "").trim();
    if (shouldRefreshPrompt) {
      progress?.set(`${label}: the video prompt is missing, so Gemma is viewing the extracted frame and target end frame...`, Math.min(98, percent + 3));
      await generateI2VPromptForSegment(nextSegment, progress, Math.min(98, percent + 5), `${label}: Gemma FLF`, { unloadAfter: true, forceVision: true });
    } else {
      progress?.set(`${label}: extracted final frame assigned. Keeping the existing saved FLF video prompt.`, Math.min(98, percent + 5));
    }
    if (nextSegment.id === state.activeId) {
      i2vPrompt.value = nextSegment.i2v_prompt || "";
      syncRTVSceneImageAnchorPanel();
    }
    render();
    await autoSaveSessionQuiet("FLF rendered-frame chain prepared");
    return { framePath, prompt: nextSegment.i2v_prompt || "", prompt_refreshed: shouldRefreshPrompt };
  }

  function restoreSceneImagePromptState(segment, snapshot) {
    if (!segment || !snapshot) return;
    segment.image = snapshot.image || null;
    segment.image_history = Array.isArray(snapshot.image_history) ? [...snapshot.image_history] : [];
    segment.image_history_index = Number(snapshot.image_history_index ?? -1);
    segment.custom_image_path = snapshot.custom_image_path || "";
    segment.custom_image_data = snapshot.custom_image_data || "";
    segment.custom_image_name = snapshot.custom_image_name || "";
    segment.approved_image_path = snapshot.approved_image_path || "";
    segment.preview_mode = snapshot.preview_mode || "image";
    segment.t2i_prompt = snapshot.t2i_prompt || "";
    segment.flux_prompt = snapshot.flux_prompt || "";
    segment.nb_prompt = snapshot.nb_prompt || "";
    segment.flow_gpt_prompt = snapshot.flow_gpt_prompt || "";
    segment.enhance_prompt = snapshot.enhance_prompt || "";
    if (segment.id === activeSegment()?.id) {
      syncInspector();
      syncPreview(segment);
    }
  }

  function firstLastFrameEndPromptForSegment(segment, firstFrame, imageMode = state.imageModelMode || "zimage") {
    const sceneName = sceneDisplayName(segment, segmentIndexInfo(segment).index);
    const motion = [
      segment.i2v_notes,
      segment.motion_notes,
      segment.story_beat,
      segment.notes,
      segment.flux_notes,
      segment.nb_notes,
      segment.lyric_text,
    ].map((value) => String(value || "").trim()).filter(Boolean).join("\n");
    const modeHint = imageMode === "flow_gpt"
      ? "Write as a browser image prompt."
      : imageMode === "nano_banana"
        ? "Write as a NanoBanana image prompt."
        : imageMode === "flux_klein"
          ? "Write as a Flux/Klein image prompt."
          : "Write as a cinematic text-to-image prompt.";
    const sourceName = firstFrame?.name || firstFrame?.path || "the first frame";
    const startState = String(segment.flf_start_state || "").trim();
    const transformation = String(segment.flf_transformation || "").trim();
    const endState = String(segment.flf_end_state || "").trim();
    const carryForward = String(segment.flf_carry_forward || "").trim();
    const motionPlan = String(segment.flf_motion_plan || segment.i2v_prompt || "").trim();
    const customDirection = segment.flf_endpoint_mode === "custom" ? String(segment.flf_custom_end_direction || "").trim() : "";
    const subjectContinuity = segment.no_character_present ? "" : String(segmentMappedSubjectText(segment) || "").trim();
    const locationContinuity = String(segmentMappedLocationText(segment) || "").trim();
    const sameLocationDiversity = flfSameLocationCameraDiversityDirection(segment, "end");
    return [
      `${modeHint} Create the LAST FRAME for a First Last Frame video shot.`,
      `Scene: ${sceneName}`,
      `Use ${sourceName} as the FIRST FRAME identity, composition, style, lighting, wardrobe, environment, and color reference.`,
      "The new image must feel like the natural end point of the same shot, not a different scene.",
      "Preserve character identity and the same visual world. Change pose, camera endpoint, expression, staging, or revealed action only as needed to create a clear A-to-B video transition.",
      startState ? `Storyboard opening state:\n${startState}` : "",
      transformation ? `Required continuous transformation:\n${transformation}` : "",
      endState ? `REQUIRED ENDPOINT — make the generated image visibly match this destination:\n${endState}` : "",
      carryForward ? `Continuity details that must remain usable by the following scene:\n${carryForward}` : "",
      customDirection ? `REQUIRED USER ENDING DIRECTION — this overrides an automatically inferred endpoint:\n${customDirection}` : "",
      motionPlan ? `PROVISIONAL I2V MOTION PLAN — freeze the generated image at the final visual moment described here:\n${motionPlan}` : "",
      subjectContinuity ? `CHARACTER AND WARDROBE CONTINUITY — preserve these mapped Reference Builder details, including details revealed outside the starting crop:\n${subjectContinuity}` : "",
      locationContinuity ? `LOCATION CONTINUITY:\n${locationContinuity}` : "",
      sameLocationDiversity,
      motion ? `Scene motion/story direction:\n${motion}` : "Scene motion/story direction:\nCreate a clear cinematic endpoint with meaningful visual change from the first frame.",
      endState || customDirection ? "Treat the required endpoint as authoritative. Do not substitute a generic pose or invent a different destination." : "",
      "Final answer should be only the image prompt for the end frame. Do not mention files, inputs, first frame, last frame, references, MSR, LoRA, or workflow nodes.",
    ].filter(Boolean).join("\n\n");
  }

  function addFirstFrameIngredientToSegment(segment, firstFrame, imageMode = state.imageModelMode || "zimage") {
    if (!firstFrame?.path && !firstFrame?.data) return () => {};
    const ingredient = {
      path: firstFrame.path || "",
      data: firstFrame.data || "",
      name: firstFrame.name || "first_frame.png",
    };
    if (imageMode === "flux_klein" || imageMode === "nano_banana" || imageMode === "flow_gpt") {
      const previous = Array.isArray(segment.flux_image_ingredients) ? [...segment.flux_image_ingredients] : [];
      segment.flux_image_ingredients = [ingredient, ...previous];
      return () => { segment.flux_image_ingredients = previous; };
    }
    return () => {};
  }

  function setFirstFrameAsImageToImageSourceForMode(firstFrame, imageMode = state.imageModelMode || "zimage") {
    if (!firstFrame?.path && !firstFrame?.data) return () => {};
    if (!["zimage", "ernie_image", "krea2_2pass"].includes(imageMode)) return () => {};
    const path = firstFrame.path || "";
    const data = firstFrame.data || "";
    const name = firstFrame.name || "first_frame.png";
    if (imageMode === "zimage") {
      const settings = activeZImageSettings();
      if (!settings.use_image_to_image) return () => {};
      const previous = {
        use_image_to_image: Boolean(settings.use_image_to_image),
        image_to_image_path: settings.image_to_image_path || "",
        image_to_image_data: settings.image_to_image_data || "",
        image_to_image_name: settings.image_to_image_name || "",
      };
      settings.use_image_to_image = true;
      settings.image_to_image_path = path;
      settings.image_to_image_data = data;
      settings.image_to_image_name = name;
      syncZImageSettingsPanel();
      return () => {
        Object.assign(settings, previous);
        syncZImageSettingsPanel();
      };
    }
    if (imageMode === "ernie_image") {
      const settings = activeSegment()?.use_scene_ernie_image_settings
        ? (activeSegment().ernie_image_settings || cloneErnieImageSettings(state.ernieImageSettings))
        : (state.ernieImageSettings || defaultErnieImageSettings());
      if (!settings.use_image_to_image) return () => {};
      const previous = {
        use_image_to_image: Boolean(settings.use_image_to_image),
        image_to_image_path: settings.image_to_image_path || "",
        image_to_image_data: settings.image_to_image_data || "",
        image_to_image_name: settings.image_to_image_name || "",
      };
      settings.use_image_to_image = true;
      settings.image_to_image_path = path;
      settings.image_to_image_data = data;
      settings.image_to_image_name = name;
      if (activeSegment()?.use_scene_ernie_image_settings) activeSegment().ernie_image_settings = settings;
      else state.ernieImageSettings = settings;
      syncErnieImagePanel();
      return () => {
        Object.assign(settings, previous);
        if (activeSegment()?.use_scene_ernie_image_settings) activeSegment().ernie_image_settings = settings;
        else state.ernieImageSettings = settings;
        syncErnieImagePanel();
      };
    }
    const settings = activeSegment()?.use_scene_krea2_2pass_settings
      ? (activeSegment().krea2_2pass_settings || cloneKrea2TwoPassSettings(state.krea2TwoPassSettings))
      : (state.krea2TwoPassSettings || defaultKrea2TwoPassSettings());
    if (!settings.use_image_to_image) return () => {};
    const previous = {
      use_image_to_image: Boolean(settings.use_image_to_image),
      image_to_image_path: settings.image_to_image_path || "",
      image_to_image_data: settings.image_to_image_data || "",
      image_to_image_name: settings.image_to_image_name || "",
    };
    settings.use_image_to_image = true;
    settings.image_to_image_path = path;
    settings.image_to_image_data = data;
    settings.image_to_image_name = name;
    if (activeSegment()?.use_scene_krea2_2pass_settings) activeSegment().krea2_2pass_settings = settings;
    else state.krea2TwoPassSettings = settings;
    syncKrea2TwoPassPanel();
    return () => {
      Object.assign(settings, previous);
      if (activeSegment()?.use_scene_krea2_2pass_settings) activeSegment().krea2_2pass_settings = settings;
      else state.krea2TwoPassSettings = settings;
      syncKrea2TwoPassPanel();
    };
  }

  function temporarilyDisableImageToImageForOpeningFrame(imageMode = state.imageModelMode || "zimage") {
    if (imageMode === "zimage") {
      const settings = activeZImageSettings();
      const previous = Boolean(settings.use_image_to_image);
      settings.use_image_to_image = false;
      syncZImageSettingsPanel();
      return () => {
        settings.use_image_to_image = previous;
        syncZImageSettingsPanel();
      };
    }
    if (imageMode === "ernie_image") {
      const segment = activeSegment();
      const useSceneSettings = Boolean(segment?.use_scene_ernie_image_settings);
      const settings = useSceneSettings
        ? (segment.ernie_image_settings || cloneErnieImageSettings(state.ernieImageSettings))
        : (state.ernieImageSettings || defaultErnieImageSettings());
      const previous = Boolean(settings.use_image_to_image);
      settings.use_image_to_image = false;
      if (useSceneSettings) segment.ernie_image_settings = settings;
      else state.ernieImageSettings = settings;
      syncErnieImagePanel();
      return () => {
        settings.use_image_to_image = previous;
        if (useSceneSettings) segment.ernie_image_settings = settings;
        else state.ernieImageSettings = settings;
        syncErnieImagePanel();
      };
    }
    if (imageMode === "krea2_2pass") {
      const segment = activeSegment();
      const useSceneSettings = Boolean(segment?.use_scene_krea2_2pass_settings);
      const settings = useSceneSettings
        ? (segment.krea2_2pass_settings || cloneKrea2TwoPassSettings(state.krea2TwoPassSettings))
        : (state.krea2TwoPassSettings || defaultKrea2TwoPassSettings());
      const previous = Boolean(settings.use_image_to_image);
      settings.use_image_to_image = false;
      if (useSceneSettings) segment.krea2_2pass_settings = settings;
      else state.krea2TwoPassSettings = settings;
      syncKrea2TwoPassPanel();
      return () => {
        settings.use_image_to_image = previous;
        if (useSceneSettings) segment.krea2_2pass_settings = settings;
        else state.krea2TwoPassSettings = settings;
        syncKrea2TwoPassPanel();
      };
    }
    return () => {};
  }

  function savedImagePromptForMode(segment, imageMode = state.imageModelMode || "zimage") {
    if (!segment) return "";
    if (imageMode === "flow_gpt") return String(segment.flow_gpt_prompt || segment.nb_prompt || segment.t2i_prompt || segment.flux_prompt || "").trim();
    if (imageMode === "nano_banana") return String(segment.nb_prompt || segment.t2i_prompt || segment.flux_prompt || "").trim();
    if (imageMode === "flux_klein") return String(segment.flux_prompt || segment.t2i_prompt || segment.nb_prompt || "").trim();
    return String(segment.t2i_prompt || segment.flux_prompt || segment.nb_prompt || segment.flow_gpt_prompt || "").trim();
  }

  async function createEndFrameForSegment(segment, imageMode, progress = null, percentBase = 20, percentSpan = 70, label = "Create End Frame", options = {}) {
    if (!segment) throw new Error("Scene is missing.");
    const suppliedFirstFrame = options.firstFrame || null;
    const firstFrame = suppliedFirstFrame?.path || suppliedFirstFrame?.data
      ? suppliedFirstFrame
      : firstLastFrameStartImageSource(segment);
    if (!firstFrame?.path && !firstFrame?.data) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: first-frame image is missing${flfChainingEnabled(segment) && previousAutoChainSourceSegment(segment) ? "; the previous scene needs an end frame" : ""}.`);
    }
    const snapshot = snapshotSceneImagePromptState(segment);
    const restoreIngredient = addFirstFrameIngredientToSegment(segment, firstFrame, imageMode);
    state.activeId = segment.id;
    syncInspector();
    const restoreI2I = setFirstFrameAsImageToImageSourceForMode(firstFrame, imageMode);
    const originalPromptContext = {
      notes: segment.notes || "",
      flux_notes: segment.flux_notes || "",
      nb_notes: segment.nb_notes || "",
      ref_image_path: segment.ref_image_path || "",
      ref_image_data: segment.ref_image_data || "",
      ref_image_name: segment.ref_image_name || "",
      use_vision_reference: segment.use_vision_reference,
    };
    let generatedEndpointPrompt = "";
    try {
      if (options.useExistingPrompt === true) {
        const existingPrompt = String(segment.flf_end_frame_prompt || "").trim() || savedImagePromptForMode(segment, imageMode);
        if (!existingPrompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: saved image prompt is missing.`);
        syncSegmentT2IPrompt(segment, existingPrompt);
        if (imageMode === "flow_gpt") syncSegmentFlowGptPrompt(segment, existingPrompt);
        progress?.set(`${label}: using the existing saved image prompt; Gemma is not needed.`, percentBase);
      } else {
        const endDirection = firstLastFrameEndPromptForSegment(segment, firstFrame, imageMode);
        segment.notes = endDirection;
        segment.flux_notes = endDirection;
        segment.nb_notes = endDirection;
        segment.ref_image_path = firstFrame.path || "";
        segment.ref_image_data = firstFrame.data || "";
        segment.ref_image_name = firstFrame.name || "first_frame.png";
        segment.use_vision_reference = true;
        progress?.set(`${label}: Gemma Vision is inspecting the first frame and writing the endpoint prompt...`, percentBase);
        if (imageMode === "flux_klein") {
          await generateFluxKleinPromptForSegment(segment, progress, percentBase + percentSpan * 0.12, `${label}: Gemma Flux/Klein`, { unloadAfter: true });
        } else if (imageMode === "nano_banana" || imageMode === "flow_gpt") {
          await generateNBPromptForSegment(segment, progress, percentBase + percentSpan * 0.12, `${label}: Gemma ${imageMode === "flow_gpt" ? "Browser AI" : "NanoBanana"}`, { imageMode, unloadAfter: true, flfImageTarget: "end" });
          if (imageMode === "flow_gpt") syncSegmentFlowGptPrompt(segment, segment.nb_prompt || segment.t2i_prompt || "");
        } else {
          await generateT2IPromptForSegment(segment, progress, percentBase + percentSpan * 0.12, `${label}: Gemma Vision`, { unloadAfter: true, forceVision: true, flfImageTarget: "end" });
        }
      }
      const sameLocationDiversity = flfSameLocationCameraDiversityDirection(segment, "end");
      if (sameLocationDiversity) {
        const endpointPrompt = savedImagePromptForMode(segment, imageMode);
        if (endpointPrompt && !endpointPrompt.includes("REQUIRED SAME-LOCATION CAMERA CHANGE:")) {
          const strengthenedPrompt = `${endpointPrompt}\n\n${sameLocationDiversity}`.trim();
          if (imageMode === "flow_gpt") syncSegmentFlowGptPrompt(segment, strengthenedPrompt);
          else syncSegmentT2IPrompt(segment, strengthenedPrompt);
        }
      }
      generatedEndpointPrompt = savedImagePromptForMode(segment, imageMode);
      progress?.set(`${label}: creating end-frame image with ${imageModeDisplayLabel(imageMode)}...`, percentBase + percentSpan * 0.3);
      await createImageForSegmentInCurrentMode(
        segment,
        imageMode,
        progress,
        percentBase + percentSpan * 0.38,
        percentSpan * 0.5,
        label,
        { includePreviousSceneImage: false },
      );
      const generated = segmentImageSource(segment);
      if (!generated?.path && !generated?.data) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: end frame image did not return a saved image.`);
      segment.first_last_frame_end_image_path = generated.path || "";
      segment.first_last_frame_end_image_data = generated.data || "";
      segment.first_last_frame_end_image_name = generated.name || "last_frame.png";
      segment.flf_end_frame_stale = false;
      segment.flf_final_prompt_ready = false;
    } finally {
      Object.assign(segment, originalPromptContext);
      restoreI2I();
      restoreIngredient();
      restoreSceneImagePromptState(segment, snapshot);
    }
    if (generatedEndpointPrompt) segment.flf_end_frame_prompt = generatedEndpointPrompt;
    render();
    return firstLastFrameEndImageSource(segment);
  }

  async function ensureSelectedImageForSceneVideo(segment, sceneIndex) {
    const source = segmentImageSource(segment);
    if (!source) throw new Error(`${sceneDisplayName(segment, sceneIndex)}: selected scene image is missing.`);
    const projectFolder = projectInput.value || state.projectFolder;
    if (!projectFolder) throw new Error("Project folder is missing.");
    const data = await postJson("/vrgdg/music_builder/save_scene_image", {
      source_path: source.path || "",
      image_data: source.data || "",
      project_folder: projectFolder,
      scene_number: sceneSlotNumber(segment),
    });
    segment.approved_image_path = data.saved_path || "";
    ensureSegmentRuntimeFields(segment);
    return segment.approved_image_path;
  }

  const POST_PROCESS_RENDER_CRF = 23;
  const POST_PROCESS_FILM_GRAIN_RENDER_CRF = 26;
  const POST_PROCESS_STITCH_CRF = 28;

  function normalizePostProcessEncodeCrf(value, fallback = POST_PROCESS_RENDER_CRF) {
    const crf = Math.round(Number(value));
    if (!Number.isFinite(crf)) return fallback;
    return Math.max(16, Math.min(35, crf));
  }

  function postProcessEncodeCrf(options = {}, effect = "") {
    if (options.encodeCrf != null) {
      return normalizePostProcessEncodeCrf(options.encodeCrf, POST_PROCESS_RENDER_CRF);
    }
    return effect === "film_grain" ? POST_PROCESS_FILM_GRAIN_RENDER_CRF : POST_PROCESS_RENDER_CRF;
  }

  function sceneVideoHasCurrentLut(segment, videoPath = "", options = {}) {
    const lut = normalizeSceneLut(segment?.lut || {});
    const applied = segment?.lut_last_applied || {};
    return Boolean(
      lut
      && lut.enabled !== false
      && String(applied.name || "") === lut.name
      && Math.abs(Number(applied.strength ?? -1) - Number(lut.strength ?? 10)) < 0.001
      && (options.ignorePreserveAudio || options.preserveAudio == null || (options.preserveAudio === false ? applied.preserve_audio === false : applied.preserve_audio !== false))
      && (options.ignoreEncodeCrf || options.encodeCrf == null || Number(applied.encode_crf || 0) === normalizePostProcessEncodeCrf(options.encodeCrf))
      && (!videoPath || !applied.video_path || mediaPathKey(applied.video_path) === mediaPathKey(videoPath))
    );
  }

  function sceneVideoHasCurrentFilmGrain(segment, videoPath = "", options = {}) {
    const grain = normalizeSceneFilmGrain(segment?.film_grain || {});
    const applied = segment?.film_grain_last_applied || {};
    return Boolean(
      grain
      && grain.enabled !== false
      && Math.abs(Number(applied.grain_intensity ?? -1) - Number(grain.grain_intensity ?? 0.04)) < 0.0001
      && Math.abs(Number(applied.saturation_mix ?? -1) - Number(grain.saturation_mix ?? 0.5)) < 0.0001
      && (options.ignorePreserveAudio || options.preserveAudio == null || (options.preserveAudio === false ? applied.preserve_audio === false : applied.preserve_audio !== false))
      && (options.ignoreEncodeCrf || options.encodeCrf == null || Number(applied.encode_crf || 0) === normalizePostProcessEncodeCrf(options.encodeCrf))
      && (!videoPath || !applied.video_path || mediaPathKey(applied.video_path) === mediaPathKey(videoPath))
    );
  }

  function sceneVideoHasCurrentAdjust(segment, videoPath = "", options = {}) {
    const adjust = normalizeSceneAdjust(segment?.adjust || {}, { keepEmpty: true });
    const applied = segment?.adjust_last_applied || {};
    return Boolean(
      adjust
      && sceneAdjustHasRenderableChanges(adjust)
      && String(applied.signature || "") === sceneAdjustSignature(adjust)
      && (options.ignorePreserveAudio || options.preserveAudio == null || (options.preserveAudio === false ? applied.preserve_audio === false : applied.preserve_audio !== false))
      && (options.ignoreEncodeCrf || options.encodeCrf == null || Number(applied.encode_crf || 0) === normalizePostProcessEncodeCrf(options.encodeCrf))
      && (!videoPath || !applied.video_path || mediaPathKey(applied.video_path) === mediaPathKey(videoPath))
    );
  }

  function sceneVideoIncludesCurrentLut(segment, videoPath = "") {
    const path = String(videoPath || selectedSegmentVideoPath(segment) || "").trim();
    if (!path) return false;
    if (sceneVideoHasCurrentLut(segment, path, { ignoreEncodeCrf: true, ignorePreserveAudio: true })) return true;
    const lutPath = String(segment?.lut_last_applied?.video_path || "").trim();
    if (!sceneVideoHasCurrentLut(segment, lutPath, { ignoreEncodeCrf: true, ignorePreserveAudio: true })) return false;
    const adjustApplied = segment?.adjust_last_applied || {};
    const grainApplied = segment?.film_grain_last_applied || {};
    if (postProcessPathMatches(adjustApplied.source_video_path, lutPath) && postProcessPathMatches(adjustApplied.video_path, path)) return true;
    if (postProcessPathMatches(grainApplied.source_video_path, lutPath) && postProcessPathMatches(grainApplied.video_path, path)) return true;
    if (postProcessPathMatches(adjustApplied.source_video_path, lutPath) && postProcessPathMatches(grainApplied.source_video_path, adjustApplied.video_path) && postProcessPathMatches(grainApplied.video_path, path)) return true;
    return false;
  }

  function sceneVideoIncludesCurrentAdjust(segment, videoPath = "") {
    const path = String(videoPath || selectedSegmentVideoPath(segment) || "").trim();
    if (!path) return false;
    if (sceneVideoHasCurrentAdjust(segment, path, { ignoreEncodeCrf: true, ignorePreserveAudio: true })) return true;
    const adjustPath = String(segment?.adjust_last_applied?.video_path || "").trim();
    if (!sceneVideoHasCurrentAdjust(segment, adjustPath, { ignoreEncodeCrf: true, ignorePreserveAudio: true })) return false;
    const grainApplied = segment?.film_grain_last_applied || {};
    return Boolean(postProcessPathMatches(grainApplied.source_video_path, adjustPath) && postProcessPathMatches(grainApplied.video_path, path));
  }

  function sceneVideoIncludesCurrentFilmGrain(segment, videoPath = "") {
    const path = String(videoPath || selectedSegmentVideoPath(segment) || "").trim();
    return Boolean(path && sceneVideoHasCurrentFilmGrain(segment, path, { ignoreEncodeCrf: true, ignorePreserveAudio: true }));
  }

  async function ensureSceneLutsAppliedBeforeStitch(baseSegments, progress, options = {}) {
    const segments = Array.isArray(baseSegments) ? baseSegments : [];
    const targets = segments
      .map((segment, index) => ({ segment, index, videoPath: String(selectedSegmentVideoPath(segment) || "").trim() }))
      .filter(({ segment, videoPath }) => {
        const lut = normalizeSceneLut(segment?.lut || {});
        return lut && lut.enabled !== false && videoPath && !sceneVideoIncludesCurrentLut(segment, videoPath);
      });
    if (!targets.length) return;
    for (let index = 0; index < targets.length; index += 1) {
      const { segment, videoPath } = targets[index];
      const sceneIndex = segmentIndexInfo(segment).index;
      const base = 88 + Math.floor((index / Math.max(1, targets.length)) * 5);
      progress?.set(`Applying LUT ${index + 1}/${targets.length} before stitching...\n${sceneDisplayName(segment, sceneIndex)}\n${normalizeSceneLut(segment.lut)?.label || ""}`, base);
      const result = await applySceneLutToRenderedVideo(
        segment,
        sceneIndex,
        videoPath,
        segment.video_thumbnail_path || "",
        progress,
        (value) => Math.min(93, base + (value - 90) * 0.5),
        "",
        { preserveAudio: normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3" && miniMaxH3SettingsForSegment(segment).audio_mode === "built_in_audio", encodeCrf: POST_PROCESS_STITCH_CRF },
      );
      activateSegmentVideoPath(segment, result.video_path || videoPath, result.thumbnail_path || segment.video_thumbnail_path || "");
      segment.video_cache_bust = Date.now();
    }
    if (options.autoSaveAfter !== false) {
      await autoSaveSessionQuiet("scene LUTs applied before stitch");
    }
    render();
  }

  async function ensureSceneAdjustsAppliedBeforeStitch(baseSegments, progress, options = {}) {
    const segments = Array.isArray(baseSegments) ? baseSegments : [];
    const targets = segments
      .map((segment, index) => ({ segment, index, videoPath: String(selectedSegmentVideoPath(segment) || "").trim() }))
      .filter(({ segment, videoPath }) => {
        if (!segment?.adjust || segment.adjust.enabled !== true) return false;
        const adjust = normalizeSceneAdjust(segment?.adjust || {}, { keepEmpty: true });
        return adjust && sceneAdjustHasRenderableChanges(adjust) && videoPath && !sceneVideoIncludesCurrentAdjust(segment, videoPath);
      });
    if (!targets.length) return;
    for (let index = 0; index < targets.length; index += 1) {
      const { segment, videoPath } = targets[index];
      const sceneIndex = segmentIndexInfo(segment).index;
      const base = 92 + Math.floor((index / Math.max(1, targets.length)) * 3);
      progress?.set(`Applying Adjust ${index + 1}/${targets.length} before stitching...\n${sceneDisplayName(segment, sceneIndex)}`, base);
      const result = await applySceneAdjustToRenderedVideo(
        segment,
        sceneIndex,
        videoPath,
        segment.video_thumbnail_path || "",
        progress,
        (value) => Math.min(95, base + (value - 90) * 0.3),
        "",
        { preserveAudio: normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3" && miniMaxH3SettingsForSegment(segment).audio_mode === "built_in_audio", encodeCrf: POST_PROCESS_STITCH_CRF },
      );
      activateSegmentVideoPath(segment, result.video_path || videoPath, result.thumbnail_path || segment.video_thumbnail_path || "");
      segment.video_cache_bust = Date.now();
    }
    if (options.autoSaveAfter !== false) {
      await autoSaveSessionQuiet("scene Adjust applied before stitch");
    }
    render();
  }

  async function ensureSceneFilmGrainAppliedBeforeStitch(baseSegments, progress, options = {}) {
    const segments = Array.isArray(baseSegments) ? baseSegments : [];
    const targets = segments
      .map((segment, index) => ({ segment, index, videoPath: String(selectedSegmentVideoPath(segment) || "").trim() }))
      .filter(({ segment, videoPath }) => {
        const grain = normalizeSceneFilmGrain(segment?.film_grain || {});
        return grain && grain.enabled !== false && videoPath && !sceneVideoIncludesCurrentFilmGrain(segment, videoPath);
      });
    if (!targets.length) return;
    for (let index = 0; index < targets.length; index += 1) {
      const { segment, videoPath } = targets[index];
      const sceneIndex = segmentIndexInfo(segment).index;
      const base = 93 + Math.floor((index / Math.max(1, targets.length)) * 4);
      progress?.set(`Applying Film Grain ${index + 1}/${targets.length} before stitching...\n${sceneDisplayName(segment, sceneIndex)}\n${filmGrainLabel(segment.film_grain)}`, base);
      const result = await applySceneFilmGrainToRenderedVideo(
        segment,
        sceneIndex,
        videoPath,
        segment.video_thumbnail_path || "",
        progress,
        (value) => Math.min(97, base + (value - 90) * 0.4),
        "",
        { preserveAudio: normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3" && miniMaxH3SettingsForSegment(segment).audio_mode === "built_in_audio", encodeCrf: POST_PROCESS_STITCH_CRF },
      );
      activateSegmentVideoPath(segment, result.video_path || videoPath, result.thumbnail_path || segment.video_thumbnail_path || "");
      segment.video_cache_bust = Date.now();
    }
    if (options.autoSaveAfter !== false) {
      await autoSaveSessionQuiet("scene film grain applied before stitch");
    }
    render();
  }

  function validateRenderAllReady(options = {}) {
    const missing = [];
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const allScenes = batchTargetItems(sceneScope).map(({ segment }) => segment);
    if (!allScenes.length) missing.push(batchEmptyMessage(sceneScope));
    const scenesToRender = allScenes
      .map((segment) => ({ segment, index: segmentIndexInfo(segment).index }))
      .filter(({ segment }) => options.forceVideos || !String(selectedSegmentVideoPath(segment) || "").trim());
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    if (miniMaxProject) {
      if (!String(projectInput.value || state.projectFolder || "").trim()) missing.push("Project folder is missing.");
      scenesToRender.forEach(({ segment, index }) => {
        if (miniMaxH3SettingsForSegment(segment).audio_mode !== "built_in_audio"
          && !String(segment.custom_audio_path || currentProjectAudioPath() || audioInput.value || "").trim()) {
          missing.push(`${sceneDisplayName(segment, index)}: MiniMax H3 needs project audio or custom scene audio.`);
        }
        missing.push(...validateMiniMaxSceneReadyForVideo(segment, index));
      });
      missing.push(...miniMaxBatchReferenceProblems(scenesToRender));
      return missing;
    }
    const idLoraMode = currentVideoMode() === "id_lora";
    const embeddedSceneAudioMode = currentVideoMode() === "id_lora" || !!options.useEmbeddedSceneAudio;
    const sceneAudioMode = !embeddedSceneAudioMode && usingSceneAudioMode();
    if (!idLoraMode && !sceneAudioMode && !String(audioInput.value || "").trim()) missing.push("Audio file path is missing.");
    if (!idLoraMode && sceneAudioMode && !currentProjectAudioPath()) {
      const audioCheckScenes = sceneScope === "all" ? state.segments.map((segment, index) => ({ segment, index })) : batchTargetItems(sceneScope, { baseOnly: true });
      audioCheckScenes.forEach(({ segment, index }) => {
        if (!String(segment.custom_audio_path || "").trim()) {
          missing.push(`${sceneDisplayName(segment, index)}: scene audio is missing and no global audio is available as a fallback.`);
        }
      });
    }
    if (!String(projectInput.value || "").trim()) missing.push("Project folder is missing.");
    const canAutoChain = Boolean(i2vAutoChainEnabled() && currentVideoMode() === "i2v");
    const canAutoImg2Img = Boolean(img2imgContinuityEnabled() && currentVideoMode() === "i2v");
    const autoImg2ImgImageMode = state.imageModelMode || "zimage";
    scenesToRender.forEach(({ segment }, renderIndex) => {
      if (canAutoChain && renderIndex > 0) return;
      if (canAutoChain && renderIndex === 0 && canAutoChainFromPreviousRenderedScene(segment)) return;
      if (canAutoImg2Img && renderIndex > 0) {
        const readiness = validateSceneReadyForAutoImg2ImgContinuity(segment, segmentIndexInfo(segment).index, autoImg2ImgImageMode)
          .filter((message) => !/previous scene has no rendered video/i.test(message));
        missing.push(...readiness);
        return;
      }
      if (canAutoImg2Img && renderIndex === 0 && canImg2ImgContinuityFromPreviousRenderedScene(segment)) {
        missing.push(...validateSceneReadyForAutoImg2ImgContinuity(segment, segmentIndexInfo(segment).index, autoImg2ImgImageMode));
        return;
      }
      const sceneMissing = validateSceneReadyForVideo(segment, segmentIndexInfo(segment).index);
      missing.push(...(currentVideoMode() === "flf" ? sceneMissing.filter((message) => !/First Last Frame prompt is missing/i.test(message)) : sceneMissing));
    });
    return missing;
  }

  function audioFallbackTargetScenes(options = {}) {
    if (options.segment) return [options.segment].filter(Boolean);
    return batchTargetItems(normalizeBatchScope(options.sceneScope), { baseOnly: true })
      .map(({ segment }) => segment)
      .filter(Boolean);
  }

  function silentAudioDurationForRender(options = {}) {
    const scenes = audioFallbackTargetScenes(options);
    const sceneEnd = scenes.reduce((max, segment) => Math.max(max, Number(segment.end || 0)), 0);
    const sceneDuration = scenes.reduce((max, segment) => Math.max(max, timelineSegmentDuration(segment)), 0);
    return Math.max(0.1, sceneEnd, sceneDuration, timelineDuration(), Number(state.duration || 0));
  }

  async function ensureAudioOrOfferSilentTimeline(options = {}) {
    if (speakingAudioEditsActive(state)) return true;
    if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
      const targetScenes = audioFallbackTargetScenes(options);
      if (targetScenes.length && targetScenes.every((segment) => miniMaxH3SettingsForSegment(segment).audio_mode === "built_in_audio")) return true;
    }
    if (normalizeProjectVideoEngine(state.projectVideoEngine) !== "minimax_h3" && currentVideoMode() === "id_lora") return true;
    if (currentProjectAudioPath()) return true;
    const scenes = audioFallbackTargetScenes(options);
    if (!scenes.length) return true;
    if (scenes.some((segment) => String(segment.custom_audio_path || "").trim())) return true;
    if (!(activeProjectFolderForSave() || String(projectInput.value || "").trim())) return true;
    const duration = silentAudioDurationForRender(options);
    const targetLabel = options.segment
      ? `${options.segment.label || "this scene"}`
      : normalizeBatchScope(options.sceneScope) === "selected"
        ? "the selected scenes"
        : normalizeBatchScope(options.sceneScope) === "from_selected"
          ? "the scenes being rendered"
          : "the timeline";
    const ok = window.confirm(`No global audio file found, and no custom scene audio is set for ${targetLabel}.\n\nUse silence instead?\n\nOK will create a silent audio clip ${formatTime(duration)} long and continue. Cancel returns to the UI.`);
    if (!ok) return false;
    const data = await createSilentTimelineAudioForDuration(duration, { quiet: true });
    if (!data?.audio_path && !data?.saved_path) return false;
    toast(`Silent timeline audio created for this render:\n${audioInput.value}`);
    return true;
  }

  function imageAllSegmentsForMode(mode = "resume_missing", imageMode = state.imageModelMode || "zimage", sceneScope = "all") {
    const scenes = batchTargetItems(sceneScope);
    if (mode === "redo_prompts_images" || mode === "keep_prompts_redo_images") return scenes;
    return scenes.filter(({ segment }) => !segmentImageSource(segment));
  }

  function endFrameSegmentsForMode(mode = "resume_missing", sceneScope = "all") {
    const scenes = batchTargetItems(sceneScope);
    if (mode === "redo_end_frames") return scenes;
    return scenes.filter(({ segment }) => !hasFirstLastFrameEndImage(segment));
  }

  function validateCreateEndFramesReady(options = {}) {
    const mode = options.endFrameRunMode || "resume_missing";
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const missing = [];
    if (rtvReferenceBehaviorGlobalValue() !== "first_last_frame") {
      missing.push("Reference to Video > Reference Behavior must be set to First Last Frame.");
    }
    if (!batchTargetItems(sceneScope).length) missing.push(batchEmptyMessage(sceneScope));
    if (!String(projectInput.value || "").trim()) missing.push("Project folder is missing.");
    endFrameSegmentsForMode(mode, sceneScope).forEach(({ segment, index }) => {
      const firstFrame = segmentImageSource(segment);
      if (!firstFrame?.path && !firstFrame?.data) {
        missing.push(`${sceneDisplayName(segment, index)}: selected scene image is missing for the first frame.`);
      }
    });
    return missing;
  }

  function validateZImageAllReady(options = {}) {
    const mode = options.imageRunMode || "resume_missing";
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const missing = [];
    if (!batchTargetItems(sceneScope).length) missing.push(batchEmptyMessage(sceneScope));
    if (!String(projectInput.value || "").trim()) missing.push("Project folder is missing.");
    imageAllSegmentsForMode(mode, state.imageModelMode || "zimage", sceneScope).forEach(({ segment }) => {
      if (mode !== "redo_prompts_images" && String(segment.t2i_prompt || "").trim()) return;
      const reason = t2iMissingReason(segment);
      if (reason) missing.push(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: ${reason}`);
    });
    return missing;
  }

  async function prepareSceneAudioMix(progress, label = "Preparing scene audio mix", options = {}) {
    if (speakingAudioEditsActive(state)) {
      const { prepareEditedAudio } = await import("./audio_clip_editor.mjs");
      const edited = await prepareEditedAudio(state, projectInput.value || state.projectFolder, { generation: true });
      return { audioPath: edited.audio_path, srtPath: state.srtPath || srtInput.value, usedSceneAudio: true };
    }
    const sceneAudioMode = usingSceneAudioMode();
    if (!sceneAudioMode) {
      return {
        audioPath: audioInput.value,
        srtPath: state.srtPath || srtInput.value,
        usedSceneAudio: false,
      };
    }
    progress?.set(`${label}...`, 6);
    const data = await postJson("/vrgdg/music_builder/prepare_scene_audio_mix", {
      project_folder: projectInput.value,
      segments: state.segments,
      global_audio_path: currentProjectAudioPath(),
      allow_missing_scene_audio: !!options.allowMissingSceneAudio,
      srt_text_field: options.rtvLtx25 ? "lyric_text" : "label",
    }, 180000);
    if (data.srt_path) {
      state.srtPath = data.srt_path;
      srtInput.value = data.srt_path;
      setWidgetValue(node, "srt_path", data.srt_path);
    }
    render();
    return {
      audioPath: data.audio_path || audioInput.value,
      srtPath: data.srt_path || state.srtPath || srtInput.value,
      usedSceneAudio: true,
    };
  }

  function ltx25RtvSceneOrdinal(segment) {
    const ordered = (Array.isArray(state.segments) ? state.segments : [])
      .map((item, index) => ({ item, index }))
      .sort((left, right) => {
        const delta = Number(left.item?.start || 0) - Number(right.item?.start || 0);
        return Math.abs(delta) > 0.001 ? delta : left.index - right.index;
      });
    const ordinal = ordered.findIndex(({ item }) => item?.id === segment?.id);
    return ordinal >= 0 ? ordinal + 1 : 1;
  }

  function previousAutoChainSourceSegment(segment) {
    if (!segment) return null;
    const track = segmentTrack(segment);
    const timelineSegments = allEditableSegments()
      .filter((item) => segmentTrack(item) === track)
      .sort((a, b) => {
        // Visual continuity follows scene order, independent of moved or stale audio positions.
        const startDiff = Number(a.start || 0) - Number(b.start || 0);
        if (Math.abs(startDiff) > 0.001) return startDiff;
        return segmentIndexInfo(a).index - segmentIndexInfo(b).index;
      });
    const index = timelineSegments.findIndex((item) => item.id === segment.id);
    return index > 0 ? timelineSegments[index - 1] : null;
  }

  async function applySceneAdjustToRenderedVideo(segment, sceneIndex, videoPath, thumbnailPath = "", progress = null, pct = (value) => value, batchLabel = "", options = {}) {
    if (!segment?.adjust || segment.adjust.enabled !== true) {
      return { video_path: videoPath, thumbnail_path: thumbnailPath };
    }
    const adjust = normalizeSceneAdjust(segment?.adjust || {}, { keepEmpty: true });
    if (!adjust || !sceneAdjustHasRenderableChanges(adjust) || !String(videoPath || "").trim()) {
      return { video_path: videoPath, thumbnail_path: thumbnailPath };
    }
    const currentVideoPath = String(videoPath || "").trim();
    const currentThumbnailPath = String(thumbnailPath || "").trim();
    const lastRenderedPath = String(segment.adjust_rendered_path || segment.adjust_last_applied?.video_path || "").trim();
    const lastSourcePath = String(segment.adjust_last_applied?.source_video_path || "").trim();
    const lastSourceThumbnail = String(segment.adjust_last_applied?.source_thumbnail_path || "").trim();
    const hasStoredOriginal = String(segment.video_original_path || "").trim();
    if (!hasStoredOriginal) {
      segment.video_original_path = currentVideoPath;
      segment.video_original_thumbnail_path = currentThumbnailPath;
    }
    const sourcePath = lastRenderedPath && mediaPathKey(currentVideoPath) === mediaPathKey(lastRenderedPath) && lastSourcePath
      ? lastSourcePath
      : currentVideoPath;
    const sourceThumbnailPath = lastRenderedPath && mediaPathKey(currentVideoPath) === mediaPathKey(lastRenderedPath) && lastSourceThumbnail
      ? lastSourceThumbnail
      : currentThumbnailPath;
    const encodeCrf = postProcessEncodeCrf(options, "adjust");
    progress?.set(`${batchLabel}Applying Adjust to ${sceneDisplayName(segment, sceneIndex)}...`, pct(94));
    const result = await postJson("/vrgdg/music_builder/post_process/adjust/apply_video", {
      input_path: sourcePath,
      settings: adjust,
      device: "auto",
      batch_size: 8,
      replace_source: false,
      thumbnail_path: currentThumbnailPath,
      preserve_audio: options.preserveAudio !== false,
      encode_crf: encodeCrf,
    }, 20 * 60 * 1000);
    segment.adjust_rendered_path = result.output || currentVideoPath;
    segment.adjust_rendered_thumbnail_path = result.thumbnail_path || currentThumbnailPath;
    segment.adjust_last_applied = {
      enabled: adjust.enabled === true,
      signature: sceneAdjustSignature(adjust),
      settings: { ...adjust },
      applied_at: Date.now(),
      elapsed_seconds: Number(result.elapsed_seconds || 0),
      processed_frames: Number(result.processed_frames || 0),
      audio_preserved: Boolean(result.audio_preserved),
      preserve_audio: result.preserve_audio !== false,
      source_video_path: sourcePath,
      source_thumbnail_path: sourceThumbnailPath,
      video_path: result.output || currentVideoPath,
      encode_crf: Number(result.encode_crf || encodeCrf),
      encoder: result.encoder || "",
      browser_friendly: Boolean(result.browser_friendly),
      source_had_audio: result.source_had_audio,
    };
    return {
      video_path: result.output || currentVideoPath,
      thumbnail_path: result.thumbnail_path || currentThumbnailPath,
      encoder: result.encoder || "",
      browser_friendly: Boolean(result.browser_friendly),
      source_had_audio: result.source_had_audio,
    };
  }

  async function applySceneLutToRenderedVideo(segment, sceneIndex, videoPath, thumbnailPath = "", progress = null, pct = (value) => value, batchLabel = "", options = {}) {
    const lut = normalizeSceneLut(segment?.lut || {});
    if (!lut || lut.enabled === false || !String(videoPath || "").trim()) {
      return { video_path: videoPath, thumbnail_path: thumbnailPath };
    }
    const currentVideoPath = String(videoPath || "").trim();
    const currentThumbnailPath = String(thumbnailPath || "").trim();
    const lastRenderedPath = String(segment.lut_rendered_path || segment.lut_last_applied?.video_path || "").trim();
    const hasStoredOriginal = String(segment.video_original_path || "").trim();
    if (!hasStoredOriginal) {
      segment.video_original_path = currentVideoPath;
      segment.video_original_thumbnail_path = currentThumbnailPath;
    } else if (lastRenderedPath && mediaPathKey(currentVideoPath) !== mediaPathKey(lastRenderedPath)) {
      segment.video_original_path = currentVideoPath;
      segment.video_original_thumbnail_path = currentThumbnailPath;
    }
    const sourcePath = String(segment.video_original_path || currentVideoPath).trim();
    const encodeCrf = postProcessEncodeCrf(options, "lut");
    progress?.set(`${batchLabel}Applying LUT to ${sceneDisplayName(segment, sceneIndex)}...\n${lut.label || lutLabelFromName(lut.name)}`, pct(94));
    const result = await postJson("/vrgdg/music_builder/luts/apply_video", {
      input_path: sourcePath,
      lut_name: lut.name,
      strength: lut.strength,
      replace_source: false,
      thumbnail_path: currentThumbnailPath,
      preserve_audio: options.preserveAudio !== false,
      encode_crf: encodeCrf,
    }, 20 * 60 * 1000);
    segment.lut_rendered_path = result.output || currentVideoPath;
    segment.lut_rendered_thumbnail_path = result.thumbnail_path || currentThumbnailPath;
    segment.lut_last_applied = {
      name: lut.name,
      label: lut.label || lutLabelFromName(lut.name),
      strength: lut.strength,
      applied_at: Date.now(),
      elapsed_seconds: Number(result.elapsed_seconds || 0),
      processed_frames: Number(result.processed_frames || 0),
      audio_preserved: Boolean(result.audio_preserved),
      preserve_audio: result.preserve_audio !== false,
      source_video_path: sourcePath,
      video_path: result.output || currentVideoPath,
      encode_crf: Number(result.encode_crf || encodeCrf),
      encoder: result.encoder || "",
      browser_friendly: Boolean(result.browser_friendly),
      source_had_audio: result.source_had_audio,
    };
    return {
      video_path: result.output || currentVideoPath,
      thumbnail_path: result.thumbnail_path || currentThumbnailPath,
    };
  }

  async function applySceneFilmGrainToRenderedVideo(segment, sceneIndex, videoPath, thumbnailPath = "", progress = null, pct = (value) => value, batchLabel = "", options = {}) {
    const grain = normalizeSceneFilmGrain(segment?.film_grain || {});
    if (!grain || grain.enabled === false || !String(videoPath || "").trim()) {
      return { video_path: videoPath, thumbnail_path: thumbnailPath };
    }
    const currentVideoPath = String(videoPath || "").trim();
    const currentThumbnailPath = String(thumbnailPath || "").trim();
    const lastRenderedPath = String(segment.film_grain_rendered_path || segment.film_grain_last_applied?.video_path || "").trim();
    const lastSourcePath = String(segment.film_grain_last_applied?.source_video_path || "").trim();
    const lastSourceThumbnail = String(segment.film_grain_last_applied?.source_thumbnail_path || "").trim();
    const hasStoredOriginal = String(segment.video_original_path || "").trim();
    if (!hasStoredOriginal) {
      segment.video_original_path = currentVideoPath;
      segment.video_original_thumbnail_path = currentThumbnailPath;
    }
    const sourcePath = lastRenderedPath && mediaPathKey(currentVideoPath) === mediaPathKey(lastRenderedPath) && lastSourcePath
      ? lastSourcePath
      : currentVideoPath;
    const sourceThumbnailPath = lastRenderedPath && mediaPathKey(currentVideoPath) === mediaPathKey(lastRenderedPath) && lastSourceThumbnail
      ? lastSourceThumbnail
      : currentThumbnailPath;
    const encodeCrf = postProcessEncodeCrf(options, "film_grain");
    progress?.set(`${batchLabel}Applying film grain to ${sceneDisplayName(segment, sceneIndex)}...\n${filmGrainLabel(grain)}`, pct(94));
    const result = await postJson("/vrgdg/music_builder/post_process/film_grain/apply_video", {
      input_path: sourcePath,
      grain_intensity: grain.grain_intensity,
      saturation_mix: grain.saturation_mix,
      device: "auto",
      batch_size: 8,
      replace_source: false,
      thumbnail_path: currentThumbnailPath,
      preserve_audio: options.preserveAudio !== false,
      encode_crf: encodeCrf,
    }, 20 * 60 * 1000);
    segment.film_grain_rendered_path = result.output || currentVideoPath;
    segment.film_grain_rendered_thumbnail_path = result.thumbnail_path || currentThumbnailPath;
    segment.film_grain_last_applied = {
      enabled: grain.enabled !== false,
      grain_intensity: grain.grain_intensity,
      saturation_mix: grain.saturation_mix,
      applied_at: Date.now(),
      elapsed_seconds: Number(result.elapsed_seconds || 0),
      processed_frames: Number(result.processed_frames || 0),
      audio_preserved: Boolean(result.audio_preserved),
      preserve_audio: result.preserve_audio !== false,
      source_video_path: sourcePath,
      source_thumbnail_path: sourceThumbnailPath,
      video_path: result.output || currentVideoPath,
      encode_crf: Number(result.encode_crf || encodeCrf),
      encoder: result.encoder || "",
      browser_friendly: Boolean(result.browser_friendly),
      source_had_audio: result.source_had_audio,
    };
    return {
      video_path: result.output || currentVideoPath,
      thumbnail_path: result.thumbnail_path || currentThumbnailPath,
      encoder: result.encoder || "",
      browser_friendly: Boolean(result.browser_friendly),
      source_had_audio: result.source_had_audio,
    };
  }

  return {
    applySceneAdjustToRenderedVideo, applySceneFilmGrainToRenderedVideo, applySceneLutToRenderedVideo,
    canImg2ImgContinuityFromPreviousRenderedScene, createEndFrameForSegment,
    createImageForSegmentInCurrentMode, endFrameSegmentsForMode, ensureAudioOrOfferSilentTimeline,
    ensureSceneAdjustsAppliedBeforeStitch, ensureSceneFilmGrainAppliedBeforeStitch,
    ensureSceneLutsAppliedBeforeStitch, ensureSelectedImageForSceneVideo, imageAllSegmentsForMode,
    ltx25RtvSceneOrdinal, prepareAutoChainedNextScene, prepareAutoImg2ImgContinuityForScene,
    prepareFLFRenderedFrameNextScene, prepareSceneAudioMix, previousAutoChainSourceSegment,
    savedImagePromptForMode, validateCreateEndFramesReady, validateMiniMaxSceneReadyForVideo,
    validateRenderAllReady, validateSceneReadyForVideo, validateSrtTimingForSceneVideo,
    validateZImageAllReady,
  };
}
