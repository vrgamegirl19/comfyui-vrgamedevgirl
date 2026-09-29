import { normalizeOverlayClip, normalizeOverlayTrackState } from "../VRGDG_OverlayTrack.js";
import {
  normalizeOwnServerTimeoutMinutes,
  normalizeSceneRenderWaitHours,
  setBuilderAutomaticMemoryCleanupEnabled,
} from "./comfy_api.mjs";
import {
  cloneFlowGptBrowserSettings,
  cloneI2VVideoSettings,
  cloneKrea2TwoPassSettings,
  normalizeAutoBuildPreparation,
  normalizeBuilderStoryboardDefaults,
  normalizeBuilderStoryLayer,
} from "./model_settings.mjs";
import { cloneKrea2ReferenceSettings } from "./models.mjs";
import { normalizeNotificationSettings } from "./notifications.mjs";
import {
  normalizeAutoImg2ImgCreativity,
  normalizeAutoImg2ImgStartStep,
  normalizeContinuityMode,
  normalizeGemmaContextLimit,
  normalizeGemmaGpuLayers,
  normalizeLmStudioContextLimit,
  normalizeOutputTokenLimit,
} from "./prompt_text.mjs";
import { normalizeLyricMapper } from "./reference_data.mjs";
import { normalizeTimelineMarkers, normalizeTimelineRange } from "./timeline_state.mjs";

export function createHistory({
  activeSegment, applyLayoutSizes, autoSaveControl, ensureAllSegmentRuntimeFields, imageContinuityEnabled,
  imageContinuityStrength, redoButton, render, segmentTrack, setBeatMarkersVisible, snapToBeatsControl, state,
  syncBuilderLlmModelSelectsFromRunner, syncErnieImagePanel, syncFluxKleinPanel,
  syncI2VMotionJsonFromSegments, syncI2VVideoSettingsPanel, syncInspector, syncKrea2TwoPassPanel,
  syncLyricNoteControls, syncPromptJsonFromSegments, syncSceneNoteControls, syncVideoModePanel,
  syncVideoNoteControls, syncZEnhanceSettingsPanel, syncZImageSettingsPanel, undoButton, waveformModeSelect,
}) {
  const HISTORY_BLOB_TOKEN_PREFIX = "__VRGDG_HISTORY_BLOB__:";
  let nextHistoryBlobToken = 1;
  const historyBlobByToken = new Map();
  const historyTokenByBlob = new Map();

  function historySnapshotReplacer(key, value) {
    if (typeof value !== "string") return value;
    const isMediaDataUrl = /^data:(?:image|video|audio)\//i.test(value);
    const isLargeDataField = value.length >= 65536 && (key === "data" || /(?:_data|Data)$/.test(key));
    if (!isMediaDataUrl && !isLargeDataField) return value;
    let token = historyTokenByBlob.get(value);
    if (!token) {
      token = `${HISTORY_BLOB_TOKEN_PREFIX}${nextHistoryBlobToken++}`;
      historyTokenByBlob.set(value, token);
      historyBlobByToken.set(token, value);
    }
    return token;
  }

  function historySnapshotReviver(_key, value) {
    if (typeof value !== "string" || !value.startsWith(HISTORY_BLOB_TOKEN_PREFIX)) return value;
    return historyBlobByToken.get(value) ?? value;
  }

  function pruneHistoryBlobCache() {
    const snapshots = [...state.undoStack, ...state.redoStack];
    for (const [token, blob] of historyBlobByToken.entries()) {
      if (snapshots.some((snapshot) => snapshot.includes(token))) continue;
      historyBlobByToken.delete(token);
      historyTokenByBlob.delete(blob);
    }
  }

  function clearHistoryBlobCache() {
    historyBlobByToken.clear();
    historyTokenByBlob.clear();
    nextHistoryBlobToken = 1;
  }

  function historySnapshot() {
    return JSON.stringify({
      segments: state.segments,
      overlaySegments: state.overlaySegments,
      overlayTrack: normalizeOverlayTrackState(state.overlayTrack),
      activeId: state.activeId,
      activeTrack: state.activeTrack,
      timingFrozen: state.timingFrozen,
      srtMode: state.srtMode,
      promptJsonPath: state.promptJsonPath,
      i2vMotionJsonPath: state.i2vMotionJsonPath,
      imageTriggerPhrase: state.imageTriggerPhrase,
      videoTriggerPhrase: state.videoTriggerPhrase,
      useI2VPromptEnhancementPass: state.useI2VPromptEnhancementPass,
      failOnInvalidPromptFormats: Boolean(state.failOnInvalidPromptFormats),
      autoChainLastFrame: state.autoChainLastFrame,
      autoChainStyle: state.autoChainStyle,
      autoChainDirection: state.autoChainDirection,
      autoChainTransitionLoraPrompt: state.autoChainTransitionLoraPrompt,
      autoChainTransitionTrigger: state.autoChainTransitionTrigger,
      useVrgdgTextContext: state.useVrgdgTextContext,
      themeStylePath: state.themeStylePath,
      storyIdeaPath: state.storyIdeaPath,
      subjectScenePath: state.subjectScenePath,
      textGemmaRunner: state.textGemmaRunner,
      qwenModelFile: state.qwenModelFile,
      qwenMmprojFile: state.qwenMmprojFile,
      gemmaModelFile: state.gemmaModelFile,
      gemmaContextLimit: normalizeGemmaContextLimit(state.gemmaContextLimit),
      gemmaOutputTokenLimit: normalizeOutputTokenLimit(state.gemmaOutputTokenLimit),
      gemmaGpuLayers: normalizeGemmaGpuLayers(state.gemmaGpuLayers),
      lmStudioBaseUrl: state.lmStudioBaseUrl,
      lmStudioModel: state.lmStudioModel,
      lmStudioApiKey: state.lmStudioApiKey,
      lmStudioContextLimit: normalizeLmStudioContextLimit(state.lmStudioContextLimit),
      lmStudioOutputTokenLimit: normalizeOutputTokenLimit(state.lmStudioOutputTokenLimit),
      llmApiProvider: state.llmApiProvider,
      llmApiModel: state.llmApiModel,
      llmApiKeyProject: state.llmApiKeyProject,
      ownServerUrl: state.ownServerUrl,
      ownServerModel: state.ownServerModel,
      ownServerApiKeyProject: state.ownServerApiKeyProject,
      ownServerOutputTokenLimit: normalizeOutputTokenLimit(state.ownServerOutputTokenLimit),
      ownServerTimeoutMinutes: normalizeOwnServerTimeoutMinutes(state.ownServerTimeoutMinutes),
      notificationSettings: normalizeNotificationSettings(state.notificationSettings),
      automaticMemoryCleanup: Boolean(state.automaticMemoryCleanup),
      waveformMode: state.waveformMode,
      snapToBeats: state.snapToBeats,
      beats: state.beats,
      detectedTempoBpm: state.detectedTempoBpm,
      beatCalibration: state.beatCalibration,
      showBeatMarkers: state.showBeatMarkers,
      showTimelineSceneNotes: state.showTimelineSceneNotes,
      showTimelineVideoNotes: state.showTimelineVideoNotes,
      showTimelineLyricNotes: state.showTimelineLyricNotes,
      selectedTimelineRange: state.selectedTimelineRange,
      timelineMarkers: state.timelineMarkers,
      activeTimelineMarkerId: state.activeTimelineMarkerId,
      leftPanelWidth: state.leftPanelWidth,
      rightPanelWidth: state.rightPanelWidth,
      timelinePanelHeight: state.timelinePanelHeight,
      timelineZoom: state.timelineZoom,
      autoSaveEnabled: state.autoSaveEnabled,
      imageModelMode: state.imageModelMode,
      zimageSettings: state.zimageSettings,
      referenceKrea2Settings: state.referenceKrea2Settings,
      fluxKleinSettings: state.fluxKleinSettings,
      nbImageSettings: state.nbImageSettings,
      ernieImageSettings: state.ernieImageSettings,
      krea2TwoPassSettings: state.krea2TwoPassSettings,
      useFluxGlobalImageIngredients: state.useFluxGlobalImageIngredients,
      fluxGlobalImageIngredients: state.fluxGlobalImageIngredients,
      builderStoryLayer: normalizeBuilderStoryLayer(state.builderStoryLayer),
      builderStoryboardDefaults: normalizeBuilderStoryboardDefaults(state.builderStoryboardDefaults),
      autoBuildPreparation: normalizeAutoBuildPreparation(state.autoBuildPreparation),
      wizardBetaDraft: state.wizardBetaDraft,
      lyricMapper: normalizeLyricMapper(state.lyricMapper),
      zEnhanceSettings: state.zEnhanceSettings,
      videoModelMode: state.videoModelMode,
      i2vVideoSettings: state.i2vVideoSettings,
      continuityMode: normalizeContinuityMode(state.continuityMode, state.autoChainLastFrame),
      autoImg2ImgStartStep: normalizeAutoImg2ImgStartStep(state.autoImg2ImgStartStep),
      autoImg2ImgCreativity: normalizeAutoImg2ImgCreativity(state.autoImg2ImgCreativity),
      promptToolsHintPrefs: state.promptToolsHintPrefs,
    }, historySnapshotReplacer);
  }

  function restoreHistorySnapshot(snapshot) {
    const data = JSON.parse(snapshot, historySnapshotReviver);
    const legacyLlmMaxTokens = data.llmMaxTokens ?? data.llm_max_tokens;
    state.isRestoringHistory = true;
    state.segments = data.segments || [];
    state.overlaySegments = data.overlaySegments || data.overlay_segments || [];
    state.overlaySegments.forEach(normalizeOverlayClip);
    state.overlayTrack = normalizeOverlayTrackState(data.overlayTrack || data.overlay_track || state.overlayTrack);
    ensureAllSegmentRuntimeFields();
    state.activeId = data.activeId || state.segments[0]?.id || "";
    state.activeTrack = data.activeTrack || data.active_track || segmentTrack(activeSegment()) || "base";
    state.timingFrozen = Boolean(data.timingFrozen);
    state.srtMode = Boolean(data.srtMode);
    state.promptJsonPath = data.promptJsonPath || "";
    state.i2vMotionJsonPath = data.i2vMotionJsonPath || "";
    state.imageTriggerPhrase = data.imageTriggerPhrase || "";
    state.videoTriggerPhrase = data.videoTriggerPhrase || "";
    state.useI2VPromptEnhancementPass = data.useI2VPromptEnhancementPass ?? data.use_i2v_prompt_enhancement_pass ?? state.useI2VPromptEnhancementPass ?? false;
    state.failOnInvalidPromptFormats = data.failOnInvalidPromptFormats ?? data.fail_on_invalid_prompt_formats ?? state.failOnInvalidPromptFormats ?? false;
    state.autoChainLastFrame = data.autoChainLastFrame ?? data.auto_chain_last_frame ?? state.autoChainLastFrame ?? false;
    state.imageContinuityEnabled = data.image_continuity_enabled ?? state.imageContinuityEnabled ?? false;
    state.imageContinuityStrength = ["close", "balanced", "creative"].includes(data.image_continuity_strength) ? data.image_continuity_strength : (state.imageContinuityStrength || "balanced");
    imageContinuityEnabled.input.checked = Boolean(state.imageContinuityEnabled); imageContinuityStrength.value = state.imageContinuityStrength;
    state.continuityMode = normalizeContinuityMode(data.continuityMode || data.continuity_mode || state.continuityMode, state.autoChainLastFrame);
    state.autoChainLastFrame = state.continuityMode === "i2v_chain";
    state.autoImg2ImgStartStep = normalizeAutoImg2ImgStartStep(data.autoImg2ImgStartStep ?? data.auto_img2img_start_step ?? state.autoImg2ImgStartStep);
    state.autoImg2ImgCreativity = normalizeAutoImg2ImgCreativity(data.autoImg2ImgCreativity ?? data.auto_img2img_creativity ?? state.autoImg2ImgCreativity);
    state.autoChainStyle = data.autoChainStyle || data.auto_chain_style || state.autoChainStyle || "continuous";
    state.autoChainDirection = data.autoChainDirection || data.auto_chain_direction || state.autoChainDirection || "";
    state.autoChainTransitionLoraPrompt = data.autoChainTransitionLoraPrompt ?? data.auto_chain_transition_lora_prompt ?? state.autoChainTransitionLoraPrompt ?? false;
    state.autoChainTransitionTrigger = data.autoChainTransitionTrigger || data.auto_chain_transition_trigger || state.autoChainTransitionTrigger || "zhuanchang";
    state.useVrgdgTextContext = data.useVrgdgTextContext ?? true;
    state.themeStylePath = data.themeStylePath || "";
    state.storyIdeaPath = data.storyIdeaPath || "";
    state.subjectScenePath = data.subjectScenePath || "";
    state.textGemmaRunner = data.textGemmaRunner || data.text_gemma_runner || state.textGemmaRunner || "builtin";
    state.qwenModelFile = data.qwenModelFile || data.qwen_model_file || state.qwenModelFile || "";
    state.qwenMmprojFile = data.qwenMmprojFile || data.qwen_mmproj_file || state.qwenMmprojFile || "";
    state.gemmaModelFile = data.gemmaModelFile || data.gemma_model_file || state.gemmaModelFile || "";
    syncBuilderLlmModelSelectsFromRunner();
    state.gemmaContextLimit = normalizeGemmaContextLimit(data.gemmaContextLimit ?? data.gemma_context_limit ?? data.n_ctx ?? legacyLlmMaxTokens ?? state.gemmaContextLimit);
    state.gemmaOutputTokenLimit = normalizeOutputTokenLimit(data.gemmaOutputTokenLimit ?? data.gemma_output_token_limit ?? legacyLlmMaxTokens ?? state.gemmaOutputTokenLimit);
    state.gemmaGpuLayers = normalizeGemmaGpuLayers(data.gemmaGpuLayers ?? data.gemma_gpu_layers ?? data.n_gpu_layers ?? state.gemmaGpuLayers);
    state.lmStudioBaseUrl = data.lmStudioBaseUrl || data.lm_studio_base_url || state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1";
    state.lmStudioModel = data.lmStudioModel || data.lm_studio_model || state.lmStudioModel || "";
    state.lmStudioApiKey = data.lmStudioApiKey || data.lm_studio_api_key || state.lmStudioApiKey || "";
    state.lmStudioContextLimit = normalizeLmStudioContextLimit(data.lmStudioContextLimit ?? data.lm_studio_context_limit ?? state.lmStudioContextLimit);
    state.lmStudioOutputTokenLimit = normalizeOutputTokenLimit(data.lmStudioOutputTokenLimit ?? data.lm_studio_output_token_limit ?? legacyLlmMaxTokens ?? state.lmStudioOutputTokenLimit);
    state.llmApiProvider = data.llmApiProvider || data.llm_api_provider || state.llmApiProvider || "openai";
    state.llmApiModel = data.llmApiModel || data.llm_api_model || state.llmApiModel || "";
    state.llmApiKeyProject = data.llmApiKeyProject || data.llm_api_key_project || state.llmApiKeyProject || "";
    if (state.llmApiKeyProject) state.llmApiKey = state.llmApiKeyProject;
    state.notificationSettings = normalizeNotificationSettings(data.notificationSettings || data.notification_settings || state.notificationSettings);
    state.automaticMemoryCleanup = setBuilderAutomaticMemoryCleanupEnabled(data.automaticMemoryCleanup ?? data.automatic_memory_cleanup ?? state.automaticMemoryCleanup ?? false);
    state.sceneRenderWaitHours = normalizeSceneRenderWaitHours(data.sceneRenderWaitHours ?? data.scene_render_wait_hours ?? state.sceneRenderWaitHours);
    state.waveformMode = data.waveformMode || state.waveformMode || "medium";
    state.snapToBeats = data.snapToBeats ?? state.snapToBeats ?? true;
    state.showTimelineSceneNotes = data.showTimelineSceneNotes ?? state.showTimelineSceneNotes ?? false;
    state.showTimelineVideoNotes = data.showTimelineVideoNotes ?? state.showTimelineVideoNotes ?? false;
    state.showTimelineLyricNotes = data.showTimelineLyricNotes ?? state.showTimelineLyricNotes ?? false;
    state.selectedTimelineRange = normalizeTimelineRange(data.selectedTimelineRange || data.selected_timeline_range || state.selectedTimelineRange);
    state.timelineMarkers = normalizeTimelineMarkers(data.timelineMarkers || data.timeline_markers || state.timelineMarkers);
    state.activeTimelineMarkerId = data.activeTimelineMarkerId || data.active_timeline_marker_id || "";
    state.peaks = Array.isArray(data.peaks) ? data.peaks : state.peaks;
    state.beats = Array.isArray(data.beats) ? data.beats : state.beats;
    state.detectedTempoBpm = Math.max(0, Number(data.detectedTempoBpm ?? data.detected_tempo_bpm ?? state.detectedTempoBpm ?? 0));
    state.beatCalibration = data.beatCalibration || data.beat_calibration || null;
    setBeatMarkersVisible(data.showBeatMarkers ?? state.showBeatMarkers ?? false);
    state.leftPanelWidth = data.leftPanelWidth || state.leftPanelWidth || 260;
    state.rightPanelWidth = data.rightPanelWidth || state.rightPanelWidth || 360;
    state.timelinePanelHeight = data.timelinePanelHeight || state.timelinePanelHeight || 300;
    state.timelineZoom = data.timelineZoom || state.timelineZoom || 45;
    state.autoSaveEnabled = data.autoSaveEnabled ?? state.autoSaveEnabled ?? true;
    state.imageModelMode = data.imageModelMode || data.fluxKleinSettings?.image_model_mode || state.imageModelMode || "zimage";
    state.pxPerSecond = state.timelineZoom;
    waveformModeSelect.value = state.waveformMode;
    snapToBeatsControl.input.checked = Boolean(state.snapToBeats);
    syncSceneNoteControls();
    syncVideoNoteControls();
    syncLyricNoteControls();
    autoSaveControl.input.checked = Boolean(state.autoSaveEnabled);
    applyLayoutSizes();
    state.zimageSettings = data.zimageSettings || state.zimageSettings;
    state.referenceKrea2Settings = cloneKrea2ReferenceSettings(data.referenceKrea2Settings || data.reference_krea2_settings || state.referenceKrea2Settings);
    state.fluxKleinSettings = data.fluxKleinSettings || state.fluxKleinSettings;
    state.flowGptBrowserSettings = cloneFlowGptBrowserSettings(data.flowGptBrowserSettings || data.flow_gpt_browser_settings || state.flowGptBrowserSettings);
    state.nbImageSettings = data.nbImageSettings || data.nb_image_settings || state.nbImageSettings;
    state.ernieImageSettings = data.ernieImageSettings || state.ernieImageSettings;
    state.krea2TwoPassSettings = cloneKrea2TwoPassSettings(data.krea2TwoPassSettings || data.krea2_2pass_settings || state.krea2TwoPassSettings);
    state.useFluxGlobalImageIngredients = Boolean(data.useFluxGlobalImageIngredients);
    state.fluxGlobalImageIngredients = Array.isArray(data.fluxGlobalImageIngredients) ? data.fluxGlobalImageIngredients : [];
    state.builderStoryLayer = normalizeBuilderStoryLayer(data.builderStoryLayer || data.builder_story_layer || {});
    state.builderStoryboardDefaults = normalizeBuilderStoryboardDefaults(data.builderStoryboardDefaults || data.builder_storyboard_defaults || {});
    state.autoBuildPreparation = normalizeAutoBuildPreparation(data.autoBuildPreparation || data.auto_build_preparation || {});
    state.wizardBetaDraft = data.wizardBetaDraft || null;
    state.lyricMapper = normalizeLyricMapper(data.lyricMapper || data.lyric_mapper || {});
    state.zEnhanceSettings = data.zEnhanceSettings || state.zEnhanceSettings;
    state.videoModelMode = data.videoModelMode || data.video_model_mode || state.videoModelMode || "i2v";
    state.i2vVideoSettings = cloneI2VVideoSettings(data.i2vVideoSettings || state.i2vVideoSettings);
    state.promptToolsHintPrefs = data.promptToolsHintPrefs || data.prompt_tools_hint_prefs || state.promptToolsHintPrefs || {};
    syncZImageSettingsPanel();
    syncFluxKleinPanel();
    syncErnieImagePanel();
    syncKrea2TwoPassPanel();
    syncZEnhanceSettingsPanel();
    syncI2VVideoSettingsPanel();
    syncVideoModePanel();
    syncInspector();
    render();
    updateHistoryButtons();
    state.isRestoringHistory = false;
  }

  function pushHistory() {
    if (state.isRestoringHistory) return;
    const snapshot = historySnapshot();
    if (state.undoStack[state.undoStack.length - 1] === snapshot) return;
    state.undoStack.push(snapshot);
    if (state.undoStack.length > 50) state.undoStack.shift();
    state.redoStack = [];
    pruneHistoryBlobCache();
    updateHistoryButtons();
  }

  function undo() {
    if (!state.undoStack.length) return;
    const current = historySnapshot();
    const previous = state.undoStack.pop();
    state.redoStack.push(current);
    if (state.redoStack.length > 50) state.redoStack.shift();
    restoreHistorySnapshot(previous);
    pruneHistoryBlobCache();
    syncPromptJsonFromSegments("undo");
    syncI2VMotionJsonFromSegments("undo");
  }

  function redo() {
    if (!state.redoStack.length) return;
    const current = historySnapshot();
    const next = state.redoStack.pop();
    state.undoStack.push(current);
    if (state.undoStack.length > 50) state.undoStack.shift();
    restoreHistorySnapshot(next);
    pruneHistoryBlobCache();
    syncPromptJsonFromSegments("redo");
    syncI2VMotionJsonFromSegments("redo");
  }

  function updateHistoryButtons() {
    undoButton.disabled = !state.undoStack.length;
    redoButton.disabled = !state.redoStack.length;
    undoButton.style.opacity = undoButton.disabled ? ".55" : "1";
    redoButton.style.opacity = redoButton.disabled ? ".55" : "1";
  }

  return { clearHistoryBlobCache, pushHistory, redo, undo, updateHistoryButtons };
}
