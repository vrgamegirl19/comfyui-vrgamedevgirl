import { createToast, makeButton, setButtonDisabled, setButtonVariant } from "./controls.mjs";
import {
  FACIAL_PERFORMANCE_PRESETS,
  ID_LORA_FACIAL_PERFORMANCE_PRESETS,
  ID_LORA_PERFORMANCE_STYLE_PRESETS,
  PERFORMANCE_STYLE_PRESETS,
} from "./performance_presets.mjs";
import { createStoryboardReferences, normalizeReferenceBuilderCatalog } from "./references.mjs";
import {
  normalizeStoryArcDetail,
  normalizeStoryboardMiniMaxH3AudioMode,
  normalizeStoryboardMiniMaxH3Mode,
  normalizeStoryboardPerformanceMode,
  normalizeStoryboardProjectVideoEngine,
  normalizeStoryboardShortFilmPlanningMode,
  normalizeStoryLayer,
  scenesFromBuilderPayload,
  storyboardCutFrequencyValue,
  storyboardSpeedGuidance,
  storyboardSpeedValue,
} from "./scenes.mjs";
import { normalizeStoryboardScriptImportState } from "./script_import.mjs";
import {
  ID_LORA_IMAGE_AESTHETIC_PRESETS,
  ID_LORA_IMAGE_SHOT_FLOW_PRESETS,
  normalizeStoryboardCustomCameraFlowSequence,
  STORYBOARD_IMAGE_AESTHETIC_PRESETS,
  STORYBOARD_IMAGE_SHOT_FLOW_PRESETS,
} from "./shot_presets.mjs";
import {
  MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS,
  MINIMAX_VIDEO_STYLE_PRESETS,
  STORYBOARD_FX_PRESETS,
  storyboardSceneSupportsVideoStyle,
  storyboardTemporalIntensity,
  storyboardTemporalProtectedMode,
} from "./video_style.mjs";
import { createPromptGeneration } from "./prompt_generation.mjs";
import { createSceneTable } from "./scene_table.mjs";
import { createStoryLayer, lyricStoryStrengthText } from "./story_layer.mjs";
import { createSceneBeats, sceneStoryBeatMissing } from "./scene_beats.mjs";
import { createScriptMapper } from "./script_mapper.mjs";
import { createSettingsPanel } from "./settings_panel.mjs";
import { createStoryboardPersistence } from "./persistence.mjs";
import { createSceneEditor } from "./scene_editor.mjs";
import { wireStoryboardEvents } from "./storyboard_events.mjs";
import { buildStoryLayerPanel } from "./story_layer_layout.mjs";
import { buildSceneDefaultsPanel } from "./defaults_panel_layout.mjs";
import { buildStoryboardShell } from "./storyboard_layout.mjs";
import { hasMappedStoryboardLocation, NO_MAPPED_LOCATIONS_MESSAGE, runStoryGenerationSequence } from "./story_workflow.mjs";

export function openStoryboardBuilder(payload = {}) {
  const promptActionOnly = payload.promptActionOnly === true;
  const focusedSection = ["defaults", "story", "scenes"].includes(payload.focusedSection) ? payload.focusedSection : "";
  const focusedTitle = { defaults: "Scene Defaults", story: "Story Layer", scenes: "Scenes" }[focusedSection];
  const allowImagePrep = !focusedSection || payload.allowImagePrep === true;
  const focusSceneId = String(payload.focusSceneId || payload.focus_scene_id || "").trim();
  if (focusSceneId && document.querySelector("[data-vrgdg-focused-storyboard]")) return;
  const sceneFocus = { only: Boolean(focusSceneId) };
  const projectFolder = String(payload.projectFolder || payload.project_folder || "").trim();
  const incomingProjectVideoEngine = String(payload.projectVideoEngine || payload.project_video_engine || "").trim();
  const projectVideoEngine = normalizeStoryboardProjectVideoEngine(incomingProjectVideoEngine);
  const payloadMiniMaxH3Mode = projectVideoEngine === "minimax_h3"
    ? normalizeStoryboardMiniMaxH3Mode(payload.miniMaxH3Mode || payload.minimax_h3_mode || payload.videoPromptType || payload.video_prompt_type)
    : "";
  const payloadMiniMaxH3AudioMode = projectVideoEngine === "minimax_h3"
    ? normalizeStoryboardMiniMaxH3AudioMode(payload.miniMaxH3AudioMode || payload.minimax_h3_audio_mode)
    : "input_audio";
  const payloadVideoPromptType = ["i2v", "id_lora", "t2v", "rtv", "ingredients", "flf"].includes(String(payload.videoPromptType || payload.video_prompt_type || "").trim())
    ? String(payload.videoPromptType || payload.video_prompt_type || "").trim()
    : "";
  const isIdLoraMode = payloadVideoPromptType === "id_lora";
  const payloadPerformanceMode = normalizeStoryboardPerformanceMode(payload.performanceMode || payload.performance_mode || payload.videoType || payload.video_type);
  const isMiniMaxShortFilmMode = projectVideoEngine === "minimax_h3" && payloadPerformanceMode === "speaking";
  const usesFilmPlanningProfile = isIdLoraMode || isMiniMaxShortFilmMode;
  const openingMode = projectVideoEngine === "minimax_h3"
    ? (payloadMiniMaxH3Mode === "image_to_video" ? "storyboard_prompts" : "image_to_video_prep")
    : (payloadVideoPromptType === "i2v" ? "storyboard_prompts" : "image_to_video_prep");
  const state = {
    projectFolder,
    projectVideoEngine,
    // The Video Builder project renders from saved RefMods, so scene cards show RefMod labels and the GPT payload names them.
    refmodPipeline: Boolean(payload.refmodPipeline || payload.refmod_pipeline),
    miniMaxH3AudioMode: payloadMiniMaxH3AudioMode,
    lineMappingLyrics: String(payload.lineMappingLyrics || payload.line_mapping_lyrics || payload.lyricMapper?.source_text || payload.lyric_mapper?.source_text || ""),
    timelineMarkers: payload.timelineMarkers || payload.timeline_markers || [],
    getTimelineMarkers: typeof payload.getTimelineMarkers === "function" ? payload.getTimelineMarkers : null,
    mode: openingMode,
    scenes: scenesFromBuilderPayload(payload).map((scene) => ({
      ...scene,
      video_prompt_type: payloadVideoPromptType || scene.video_prompt_type,
      performance_mode: scene.performance_mode || payloadPerformanceMode,
    })),
    referenceBuilder: normalizeReferenceBuilderCatalog(payload.referenceBuilder || payload.reference_builder || {}),
    storyLayer: normalizeStoryLayer(payload.storyLayer || payload.story_layer || {}),
    scriptImport: normalizeStoryboardScriptImportState(payload.scriptImport || payload.script_import || {}),
    onReferenceMappingsChanged: typeof payload.onReferenceMappingsChanged === "function" ? payload.onReferenceMappingsChanged : null,
    onStoryLayerChanged: typeof payload.onStoryLayerChanged === "function" ? payload.onStoryLayerChanged : null,
    onPrepareStoryContext: typeof payload.onPrepareStoryContext === "function" ? payload.onPrepareStoryContext : null,
    onPromptsExported: typeof payload.onPromptsExported === "function" ? payload.onPromptsExported : null,
    onApplyIdLoraDialoguePlan: typeof payload.onApplyIdLoraDialoguePlan === "function" ? payload.onApplyIdLoraDialoguePlan : null,
    onApplyMiniMaxDialoguePlan: typeof payload.onApplyMiniMaxDialoguePlan === "function" ? payload.onApplyMiniMaxDialoguePlan : null,
    onCreateVideoPrompt: typeof payload.onCreateVideoPrompt === "function" ? payload.onCreateVideoPrompt : null,
    onBeforeCreateVideoPrompt: typeof payload.onBeforeCreateVideoPrompt === "function" ? payload.onBeforeCreateVideoPrompt : null,
    onSceneChanged: typeof payload.onSceneChanged === "function" ? payload.onSceneChanged : null,
    query: "",
    selected: new Set(),
    saving: false,
    gemmaSettings: payload.gemmaSettings || payload.gemma_settings || {},
    sendAdjacentLyricContext: Boolean(payload.sendAdjacentLyricContext ?? payload.send_adjacent_lyric_context ?? payload.builderStoryboardDefaults?.send_adjacent_lyric_context ?? payload.builder_storyboard_defaults?.send_adjacent_lyric_context),
    cameraFlow: String(payload.cameraFlow || payload.camera_flow || "balanced"),
    customCameraFlowSequence: normalizeStoryboardCustomCameraFlowSequence(payload.customCameraFlowSequence || payload.custom_camera_flow_sequence || payload.builderStoryboardDefaults?.custom_camera_flow_sequence || payload.builder_storyboard_defaults?.custom_camera_flow_sequence),
    imageShotFlow: String(payload.imageShotFlow || payload.image_shot_flow || (usesFilmPlanningProfile ? "film_dialogue_coverage" : "intimate")),
    imageAesthetic: String(payload.imageAesthetic || payload.image_aesthetic || (usesFilmPlanningProfile ? "film_default" : "")),
    videoStyle: String(payload.videoStyle || payload.video_style || ""),
    videoStyleCustom: String(payload.videoStyleCustom || payload.video_style_custom || ""),
    temporalWorldEffect: String(payload.temporalWorldEffect || payload.temporal_world_effect || ""),
    temporalWorldEffectCustom: String(payload.temporalWorldEffectCustom || payload.temporal_world_effect_custom || ""),
    temporalAllowBackgroundExtras: (payload.temporalAllowBackgroundExtras ?? payload.temporal_allow_background_extras) !== false,
    temporalBackgroundIntensity: storyboardTemporalIntensity(payload.temporalBackgroundIntensity ?? payload.temporal_background_intensity ?? 8),
    temporalEnvironmentTimePassage: (payload.temporalEnvironmentTimePassage ?? payload.temporal_environment_time_passage) !== false,
    temporalProtectedCharacters: storyboardTemporalProtectedMode(payload.temporalProtectedCharacters || payload.temporal_protected_characters),
    temporalProtectedCustom: String(payload.temporalProtectedCustom || payload.temporal_protected_custom || ""),
    fxPreset: String(payload.fxPreset || payload.fx_preset || payload.builderStoryboardDefaults?.fx_preset || payload.builder_storyboard_defaults?.fx_preset || ""),
    fxCustomJson: String(payload.fxCustomJson || payload.fx_custom_json || payload.builderStoryboardDefaults?.fx_custom_json || payload.builder_storyboard_defaults?.fx_custom_json || ""),
    globalConsistencyPhrase: String(payload.globalConsistencyPhrase || payload.global_consistency_phrase || ""),
    performanceStyle: String(payload.performanceStyle || payload.performance_style || payload.performance_style_default || (usesFilmPlanningProfile ? "dialogue_naturalism" : "")),
    facialPerformance: String(payload.facialPerformance || payload.facial_performance || payload.facial_performance_default || ""),
    facialPerformanceCustom: String(payload.facialPerformanceCustom || payload.facial_performance_custom || payload.facial_performance_custom_default || ""),
    cameraMotionSpeed: storyboardSpeedValue(payload.cameraMotionSpeed ?? payload.camera_motion_speed ?? payload.motion_defaults?.camera_motion_speed, 4),
    characterMotionSpeed: storyboardSpeedValue(payload.characterMotionSpeed ?? payload.character_motion_speed ?? payload.motion_defaults?.character_motion_speed, 4),
    storyArcDetail: normalizeStoryArcDetail(payload.storyArcDetail ?? payload.story_arc_detail),
    cutFrequency: storyboardCutFrequencyValue(payload.cutFrequency ?? payload.minimax_h3_cut_frequency ?? payload.builderStoryboardDefaults?.minimax_h3_cut_frequency ?? payload.builder_storyboard_defaults?.minimax_h3_cut_frequency),
    performanceMode: payloadPerformanceMode,
    shortFilmPlanningMode: normalizeStoryboardShortFilmPlanningMode(
      payload.shortFilmPlanningMode
      || payload.short_film_planning_mode
      || payload.builderStoryboardDefaults?.short_film_planning_mode
      || payload.builder_storyboard_defaults?.short_film_planning_mode,
    ),
    videoPromptType: payloadVideoPromptType,
    miniMaxH3Mode: payloadMiniMaxH3Mode,
    imageMode: String(payload.imageMode || payload.image_mode || "zimage").trim() || "zimage",
    imageModeLabel: String(payload.imageModeLabel || payload.image_mode_label || "").trim(),
    renderedSceneCount: Number(payload.renderedSceneCount || 0),
  };

  function promptRunnerName() {
    const runner = String(state.gemmaSettings?.text_runner || state.gemmaSettings?.gemma_runner || "builtin").trim().toLowerCase();
    if (runner === "lm_studio" || runner === "lmstudio" || runner === "lm-studio") return "LM Studio";
    if (runner === "llm_api" || runner === "llmapi" || runner === "llm-api" || runner === "api") return "LLM API";
    if (["qwen", "qwen_local", "qwen-local", "qwen_gguf", "qwen_gguf_local"].includes(runner)) return "Qwen Local";
    if (["ownserver", "own-server", "own_server", "custom_openai", "openai_compatible", "custom_server", "my_server"].includes(runner)) return "Custom Server";
    return "Gemma Local";
  }
  const imageShotFlowPresets = usesFilmPlanningProfile ? ID_LORA_IMAGE_SHOT_FLOW_PRESETS : STORYBOARD_IMAGE_SHOT_FLOW_PRESETS;
  const imageAestheticPresets = usesFilmPlanningProfile ? ID_LORA_IMAGE_AESTHETIC_PRESETS : STORYBOARD_IMAGE_AESTHETIC_PRESETS;
  const performanceStylePresets = usesFilmPlanningProfile ? ID_LORA_PERFORMANCE_STYLE_PRESETS : PERFORMANCE_STYLE_PRESETS;
  const facialPerformancePresets = usesFilmPlanningProfile ? ID_LORA_FACIAL_PERFORMANCE_PRESETS : FACIAL_PERFORMANCE_PRESETS;
  if (!imageShotFlowPresets[state.imageShotFlow]) state.imageShotFlow = Object.keys(imageShotFlowPresets)[0] || "off";
  if (!imageAestheticPresets.some((item) => item.value === state.imageAesthetic)) state.imageAesthetic = imageAestheticPresets[0]?.value || "";
  if (!MINIMAX_VIDEO_STYLE_PRESETS.some((item) => item.value === state.videoStyle)) state.videoStyle = "";
  if (!MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS.some((item) => item.value === state.temporalWorldEffect)) state.temporalWorldEffect = "";
  if (!STORYBOARD_FX_PRESETS.some((item) => item.value === state.fxPreset)) state.fxPreset = "";
  if (!performanceStylePresets.some((item) => item.value === state.performanceStyle)) state.performanceStyle = performanceStylePresets[0]?.value || "";
  if (!facialPerformancePresets.some((item) => item.value === state.facialPerformance)) state.facialPerformance = facialPerformancePresets[0]?.value || "";
  function storyboardDefaultsPayload() {
    return {
    builder_storyboard_defaults: {
      global_consistency_phrase: String(state.globalConsistencyPhrase || "").trim(),
      camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
      character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
      minimax_h3_cut_frequency: storyboardCutFrequencyValue(state.cutFrequency),
      camera_guidance: storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
      character_guidance: storyboardSpeedGuidance(state.characterMotionSpeed, "character"),
      send_adjacent_lyric_context: Boolean(state.sendAdjacentLyricContext),
      performance_style: String(state.performanceStyle || ""),
      short_film_planning_mode: normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode),
      camera_flow: String(state.cameraFlow || ""),
      custom_camera_flow_sequence: normalizeStoryboardCustomCameraFlowSequence(state.customCameraFlowSequence),
      image_shot_flow: String(state.imageShotFlow || ""),
      image_aesthetic: String(state.imageAesthetic || ""),
      video_style: String(state.videoStyle || ""),
      video_style_custom: String(state.videoStyleCustom || "").trim(),
      temporal_world_effect: String(state.temporalWorldEffect || ""),
      temporal_world_effect_custom: String(state.temporalWorldEffectCustom || "").trim(),
      temporal_allow_background_extras: state.temporalAllowBackgroundExtras !== false,
      temporal_background_intensity: storyboardTemporalIntensity(state.temporalBackgroundIntensity),
      temporal_environment_time_passage: state.temporalEnvironmentTimePassage !== false,
      temporal_protected_characters: storyboardTemporalProtectedMode(state.temporalProtectedCharacters),
      temporal_protected_custom: String(state.temporalProtectedCustom || "").trim(),
      fx_preset: String(state.fxPreset || ""),
      fx_custom_json: String(state.fxCustomJson || "").trim(),
    },
    global_consistency_phrase: String(state.globalConsistencyPhrase || "").trim(),
    performance_style_default: String(state.performanceStyle || ""),
    short_film_planning_mode: normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode),
    video_style: String(state.videoStyle || ""),
    video_style_custom: String(state.videoStyleCustom || "").trim(),
    temporal_world_effect: String(state.temporalWorldEffect || ""),
    temporal_world_effect_custom: String(state.temporalWorldEffectCustom || "").trim(),
    temporal_allow_background_extras: state.temporalAllowBackgroundExtras !== false,
    temporal_background_intensity: storyboardTemporalIntensity(state.temporalBackgroundIntensity),
    temporal_environment_time_passage: state.temporalEnvironmentTimePassage !== false,
    temporal_protected_characters: storyboardTemporalProtectedMode(state.temporalProtectedCharacters),
    temporal_protected_custom: String(state.temporalProtectedCustom || "").trim(),
    fx_preset: String(state.fxPreset || ""),
    fx_custom_json: String(state.fxCustomJson || "").trim(),
    camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
    character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
    story_arc_detail: normalizeStoryArcDetail(state.storyArcDetail),
    minimax_h3_cut_frequency: storyboardCutFrequencyValue(state.cutFrequency),
    custom_camera_flow_sequence: normalizeStoryboardCustomCameraFlowSequence(state.customCameraFlowSequence),
    motion_defaults: {
      camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
      character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
      camera_guidance: storyboardSpeedGuidance(state.cameraMotionSpeed, "camera"),
      character_guidance: storyboardSpeedGuidance(state.characterMotionSpeed, "character"),
    },
  };
  }
  function getSelectedScenes() {
    if (!state.selected || !state.selected.size) return [];
    return state.scenes.filter((scene) => state.selected.has(scene.id));
  }
  function getSelectedScene() {
    const scenes = getSelectedScenes();
    return scenes.length === 1 ? scenes[0] : null;
  }
  function promptRunnerGenericName() {
    return promptRunnerName();
  }
  function promptAllButtonText() {
    const kind = state.mode === "image_to_video_prep" ? "Video" : "Image";
    const selectedScenes = getSelectedScenes();
    if (selectedScenes.length === 1) {
      return `${promptRunnerName()} ${kind}`;
    }
    if (selectedScenes.length > 1) {
      return `${promptRunnerName()} ${kind} (${selectedScenes.length})`;
    }
    return `${promptRunnerName()} ${kind} All`;
  }

  const {
    add, backdrop, clearPromptsButton, clearStoryBeatsButton, close, gemmaAllButton, gptButton, header,
    headerActions, importImagePromptsButton, keepGemmaLoadedInput, search, shell, stepPrep, stepPrompts,
    steps, titleBlock,
  } = buildStoryboardShell({
    focusedTitle, promptAllButtonText, promptRunnerName, state,
  });
  header.append(titleBlock, steps, headerActions);

  const {
    cameraFlowApply, cameraFlowControls, cameraFlowInfo, cameraFlowReplace, cameraFlowSelect,
    cameraSpeedControls, cameraSpeedHint, cameraSpeedInfo, cameraSpeedInput, cameraSpeedValue,
    characterSpeedControls, characterSpeedHint, characterSpeedInfo, characterSpeedInput, characterSpeedValue,
    storyArcDetailInfo, storyArcDetailSelect,
    consistencyInfo, consistencyInput, cutFrequencyControls, cutFrequencyHint, cutFrequencyInfo,
    cutFrequencyInput, cutFrequencyValue, facialApply, facialCustomInfo, facialCustomInput, facialInfo,
    facialReplace, facialSelect, fxControls, fxCustomControls, fxCustomInput, fxInfo, fxSelect,
    imageAestheticApply, imageAestheticControls, imageAestheticInfo, imageAestheticReplace,
    imageAestheticSelect, imageCustomStyleControls, imageCustomStyleInfo, imageCustomStyleInput,
    imageShotApply, imageShotControls, imageShotInfo, imageShotReplace, imageShotSelect,
    imageWorldStyleControls, imageWorldStyleInfo, imageWorldStyleSelect, middleContent, note,
    performanceApply, performanceInfo, performanceReplace, performanceSelect, sceneDefaultsPanel,
    temporalEffectControls, temporalEffectCustomControls, temporalEffectCustomInput, temporalEffectInfo,
    temporalEffectOptions, temporalEffectSelect, temporalEnvironmentInput, temporalExtrasInput,
    temporalIntensityInput, temporalIntensityValue, temporalProtectedCustomControls,
    temporalProtectedCustomInput, temporalProtectedSelect, videoStyleApply, videoStyleControls,
    videoStyleCustomControls, videoStyleCustomInput, videoStyleInfo, videoStyleReplace, videoStyleSelect,
  } = buildSceneDefaultsPanel({
    facialPerformancePresets, focusedSection, imageAestheticPresets, imageShotFlowPresets,
    performanceStylePresets, state, usesFilmPlanningProfile,
  });
  sceneDefaultsPanel.classList.add("vrgdg-storyboard-panel");

  const {
    adjacentLyricContextInput, applyDialoguePlanButton, createMissingBeatsButton, createStoryArcButton,
    createStorySequenceButton,
    createStoryBriefButton, detectSectionsButton, gptStoryButton, idLoraDialoguePlanner,
    idLoraDialoguePlannerText, idLoraDialogueSceneCount, importStoryJsonButton, lyricStoryStrengthHintButton,
    lyricStoryStrengthInput, lyricStoryStrengthValue, miniMaxGuidedWorkflowSteps, miniMaxScriptImporter,
    miniMaxScriptImporterText, openMiniMaxScriptMapperButton, overallStoryIdeaInput, planDialogueScenesButton,
    replaceBeatsButton, shortFilmPlanningModeInfo, shortFilmPlanningModeSelect, shortFilmPlanningModeWrap,
    songStoryBriefInput, storyActions, storyLayerEnabledInput, storyLayerPanel, userStoryArcInput,
  } = buildStoryLayerPanel({
    focusedSection, isIdLoraMode, isMiniMaxShortFilmMode, promptRunnerName, state, usesFilmPlanningProfile,
  });
  syncLyricStoryStrengthLabel();

  const tableWrap = document.createElement("div");
  tableWrap.style.cssText = "margin:10px 24px 18px;overflow:auto;border:1px solid #334155;border-radius:10px;background:#0b1220;min-height:0;";

  const footer = document.createElement("div");
  footer.className = "vrgdg-storyboard-footer";
  footer.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:14px;padding:16px 24px;border-top:1px solid #334155;background:#111827;min-width:0;";
  const stats = document.createElement("div");

  const save = makeButton(focusedTitle ? `Save ${focusedTitle}` : "Save Storyboard");
  const exportPrompts = makeButton(state.onPromptsExported ? "Save Prompts to Timeline + Files" : "Export Prompt Files Only", "purple");
  exportPrompts.title = state.onPromptsExported
    ? "Copy prompts into matching Video Builder timeline segments and write TXT and JSON prompt files. This does not create or replace timeline segments."
    : "Write TXT and JSON prompt files only. This does not create or replace Video Builder timeline segments.";
  const { renderTable } = createSceneTable({
    currentRows, promptRunnerName, refreshActionButtons, state, stats, tableWrap,
    createScenePromptForActiveMode: (...args) => createScenePromptForActiveMode(...args),
    addStoryboardReferenceFromFile: (...args) => addStoryboardReferenceFromFile(...args),
    syncReferenceMappingsToVideoCreator: (...args) => syncReferenceMappingsToVideoCreator(...args),
    refreshSetupPanelSummaries: (...args) => refreshSetupPanelSummaries(...args),
    openSceneEditor: (...args) => openSceneEditor(...args),
  });

  const {
    applyCameraFlow, applyFacialPerformance, applyImageAesthetic, applyImageShotFlow, applyPerformanceStyle,
    openCustomCameraFlowDialog, refreshCameraFlowInfo, refreshCameraSpeedInfo, refreshCharacterSpeedInfo,
    refreshConsistencyInfo, refreshCutFrequencyInfo, refreshFacialInfo, refreshFxInfo,
    refreshImageAestheticInfo, refreshImageShotInfo, refreshImageWorldStyleInfo, refreshPerformanceInfo,
    refreshSetupPanelSummaries,
  } = createSettingsPanel({
    applyDialoguePlanButton, cameraFlowInfo, cameraSpeedInfo, cameraSpeedValue, characterSpeedInfo,
    characterSpeedValue, consistencyInfo, createMissingBeatsButton, createStoryArcButton,
    createStoryBriefButton, cutFrequencyInfo, cutFrequencyValue, detectSectionsButton, facialCustomInfo,
    facialInfo, facialPerformancePresets, fxCustomControls, fxInfo, idLoraDialoguePlanner,
    idLoraDialoguePlannerText, idLoraDialogueSceneCount, imageAestheticInfo, imageAestheticPresets,
    imageCustomStyleInfo, imageCustomStyleInput, imageShotFlowPresets, imageShotInfo, imageWorldStyleInfo,
    imageWorldStyleSelect, isFullyCustomShortFilm, isIdLoraMode, isMiniMaxShortFilmMode,
    miniMaxGuidedWorkflowSteps, miniMaxScriptImporter, miniMaxScriptImporterText,
    openMiniMaxScriptMapperButton, performanceInfo, performanceStylePresets, planDialogueScenesButton,
    promptRunnerName, refreshActionButtons, renderTable, sceneDefaultsPanel,
    shortFilmPlanningModeInfo, shortFilmPlanningModeWrap, state, storyActions, storyLayerPanel,
    usesFilmPlanningProfile,
  });

  const {
    copyStoryLayerForGpt, createStoryArcWithGemma, createStoryBriefWithGemma, detectLyricSections,
    notifyStoryboardDefaultsChanged, openImportStoryJsonModal, refreshTemporalEffectInfo,
    refreshVideoStyleInfo, syncStoryLayerFromInputs,
  } = createStoryLayer({
    imageCustomStyleInput, imageWorldStyleSelect, lyricStoryStrengthInput, overallStoryIdeaInput,
    promptRunnerName, refreshSetupPanelSummaries, renderTable, songStoryBriefInput, state,
    storyLayerEnabledInput, storyboardDefaultsPayload, temporalEffectCustomControls, temporalEffectInfo,
    temporalEffectOptions, temporalIntensityInput, temporalIntensityValue, temporalProtectedCustomControls,
    userStoryArcInput, videoStyleCustomControls, videoStyleInfo,
  });

  const {
    absorbSceneReferencesIntoCatalog, addStoryboardReferenceFromFile, applyVideoStyle,
    syncReferenceMappingsToVideoCreator,
  } = createStoryboardReferences({
    renderTable, state,
  });

  const {
    clearAllStoryboardPrompts, createScenePromptForActiveMode, enforceStoryboardVideoFacialRequirements,
    openImportImagePromptsFromGptModal, startAllPromptsWithGemma,
  } = createPromptGeneration({
    currentRows, gemmaAllButton, getSelectedScenes, keepGemmaLoadedInput, promptRunnerGenericName,
    promptRunnerName, renderTable, state, syncReferenceMappingsToVideoCreator,
    saveStoryboard: (...args) => saveStoryboard(...args),
  });

  const { copyStoryboardForGpt, exportPromptFiles, loadExisting, saveStoryboard } = createStoryboardPersistence({
    absorbSceneReferencesIntoCatalog, adjacentLyricContextInput, cameraFlowSelect, cameraSpeedInput, characterSpeedInput, storyArcDetailSelect,
    consistencyInput, cutFrequencyInput, enforceStoryboardVideoFacialRequirements, exportPrompts,
    facialCustomInput, facialPerformancePresets, facialSelect, fxCustomInput, fxSelect,
    getSelectedScenes, imageAestheticPresets, imageAestheticSelect, imageCustomStyleInput, imageShotFlowPresets, imageShotSelect,
    imageWorldStyleSelect,
    incomingProjectVideoEngine, keepGemmaLoadedInput, lyricStoryStrengthInput, openImportImagePromptsFromGptModal, openingMode, overallStoryIdeaInput, payload,
    payloadVideoPromptType, performanceSelect, performanceStylePresets, refreshCameraFlowInfo,
    refreshCameraSpeedInfo, refreshCharacterSpeedInfo, refreshConsistencyInfo, refreshCutFrequencyInfo,
    refreshFacialInfo, refreshFxInfo, refreshImageAestheticInfo, refreshImageShotInfo, refreshImageWorldStyleInfo, refreshPerformanceInfo,
    refreshTemporalEffectInfo, refreshVideoStyleInfo, renderTable, save, setMode, shortFilmPlanningModeSelect,
    songStoryBriefInput, state, storyLayerEnabledInput, storyboardDefaultsPayload,
    syncLyricStoryStrengthLabel, syncReferenceMappingsToVideoCreator, syncStoryLayerFromInputs,
    temporalEffectCustomInput, temporalEffectSelect, temporalEnvironmentInput, temporalExtrasInput,
    temporalIntensityInput, temporalProtectedCustomInput, temporalProtectedSelect, userStoryArcInput,
    videoStyleCustomInput, videoStyleSelect,
  });

  const {
    clearAllStoryboardStoryBeats, createAllSceneBeatsWithGemma, createSceneBeatWithGemma,
    handleReplaceSceneBeats, propagateFlfEndStateToNextScene,
  } = createSceneBeats({
    currentRows, getSelectedScenes, promptRunnerName, renderTable, saveStoryboard, state,
    syncStoryLayerFromInputs,
  });

  const { applyFilmDialoguePlanToVideoBuilder, openMiniMaxScriptMapper, planFilmDialogueScenesWithLlm } = createScriptMapper({
    applyDialoguePlanButton, idLoraDialogueSceneCount, isFullyCustomShortFilm, isIdLoraMode,
    isMiniMaxShortFilmMode, notifyStoryboardDefaultsChanged, promptRunnerName, refreshSetupPanelSummaries,
    renderTable, saveStoryboard, setMode, songStoryBriefInput, state, syncStoryLayerFromInputs,
    userStoryArcInput,
  });

  async function createStorySequenceWithGemma() {
    if (!hasMappedStoryboardLocation(state)) {
      createToast(NO_MAPPED_LOCATIONS_MESSAGE, true);
      return;
    }
    const buttons = [createStorySequenceButton, createStoryArcButton, createStoryBriefButton, createMissingBeatsButton];
    buttons.forEach((button) => { button.disabled = true; });
    try {
      const result = await runStoryGenerationSequence({
        createArc: createStoryArcWithGemma,
        createBrief: createStoryBriefWithGemma,
        createBeats: createAllSceneBeatsWithGemma,
        save: () => saveStoryboard({ throwOnError: true }),
      });
      if (result.completed) createToast("Story arc, brief, and missing scene beats are ready.");
    } catch (error) {
      createToast(`Story creation stopped: ${String(error?.message || error)}`, true);
    } finally {
      buttons.forEach((button) => { button.disabled = false; });
    }
  }

  const { openSceneEditor } = createSceneEditor({
    absorbSceneReferencesIntoCatalog, addStoryboardReferenceFromFile, backdrop, createSceneBeatWithGemma,
    createScenePromptForActiveMode, facialPerformancePresets, isFullyCustomShortFilm, isMiniMaxShortFilmMode,
    performanceStylePresets, promptRunnerGenericName, promptRunnerName, propagateFlfEndStateToNextScene,
    renderTable, saveStoryboard, sceneFocus, state, syncReferenceMappingsToVideoCreator,
    syncStoryLayerFromInputs,
  });

  stats.style.cssText = "flex:1 1 260px;min-width:0;color:#cbd5e1;font-size:13px;overflow-wrap:anywhere;";
  const footerActions = document.createElement("div");
  footerActions.style.cssText = "display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:flex-end;min-width:0;max-width:100%;";
  footerActions.append(save, exportPrompts);
  footer.append(stats, footerActions);

  if (!focusedSection || focusedSection === "defaults") middleContent.append(sceneDefaultsPanel);
  if (!focusedSection || focusedSection === "story") middleContent.append(storyLayerPanel);
  if (!focusedSection || focusedSection === "scenes") middleContent.append(tableWrap);
  if (focusedSection) {
    if (focusedSection !== "scenes") {
      headerActions.replaceChildren(close);
      footerActions.replaceChildren(save);
    }
    if (focusedSection === "story" || !allowImagePrep) steps.remove();
  }
  shell.append(header, note, middleContent, footer);
  backdrop.append(shell);
  document.body.append(backdrop);
  if (promptActionOnly) backdrop.style.display = "none";
  if (sceneFocus.only) {
    backdrop.dataset.vrgdgFocusedStoryboard = focusSceneId;
    backdrop.style.display = "none";
  }

  function syncLyricStoryStrengthLabel() {
    lyricStoryStrengthValue.textContent = lyricStoryStrengthText(lyricStoryStrengthInput.value);
  }

  function refreshActionButtons() {
    const selectedScenes = getSelectedScenes();
    // Red marks a story step that already ran, or any story step once the project has rendered scenes.
    const hasRenderedScenes = state.renderedSceneCount > 0;
    setButtonVariant(createStoryArcButton, hasRenderedScenes || String(state.storyLayer.user_story_arc || "").trim() ? "danger" : "primary");
    setButtonVariant(createStoryBriefButton, hasRenderedScenes || String(state.storyLayer.song_story_brief || "").trim() ? "danger" : "primary");
    const beatsMissing = currentRows().some((scene) => sceneStoryBeatMissing(scene, state.videoPromptType === "flf"));
    setButtonVariant(createMissingBeatsButton, beatsMissing ? "purple" : "danger");
    setButtonDisabled(createMissingBeatsButton, !beatsMissing);
    createMissingBeatsButton.title = beatsMissing ? "" : "Every scene has a story beat. Check scenes below and use Replace Scene Beats to redo them.";
    setButtonDisabled(replaceBeatsButton, !selectedScenes.length);
    const isVideoPrepMode = state.mode === "image_to_video_prep";
    const kind = isVideoPrepMode ? "Video" : "Image";
    const runnerName = promptRunnerName();
    const count = selectedScenes.length;
    if (count > 0) {
      if (count === 1) {
        const sceneLabel = selectedScenes[0].label || `Scene ${selectedScenes[0].scene_number || ""}`.trim();
        replaceBeatsButton.textContent = "Replace Scene Beat";
        replaceBeatsButton.title = `Replace the scene beat in selected ${sceneLabel} only.`;
        gptButton.textContent = `GPT ${kind}`;
        gptButton.title = `Copy GPT JSON for selected ${sceneLabel} only and open GPT.`;
        gemmaAllButton.textContent = `${runnerName} ${kind}`;
        gemmaAllButton.title = `Create ${kind.toLowerCase()} prompt for selected ${sceneLabel} only with ${runnerName}.`;
      } else {
        replaceBeatsButton.textContent = `Replace Scene Beats (${count})`;
        replaceBeatsButton.title = `Replace the scene beats in ${count} selected scenes only.`;
        gptButton.textContent = `GPT ${kind} (${count})`;
        gptButton.title = `Copy GPT JSON for ${count} selected scenes only and open GPT.`;
        gemmaAllButton.textContent = `${runnerName} ${kind} (${count})`;
        gemmaAllButton.title = `Create ${kind.toLowerCase()} prompts for ${count} selected scenes only with ${runnerName}.`;
      }
    } else {
      replaceBeatsButton.textContent = "Replace Scene Beats";
      replaceBeatsButton.title = "Check the scenes below whose story beats you want to replace.";
      gptButton.textContent = isVideoPrepMode ? "GPT Video All" : "GPT Image All";
      gptButton.title = isVideoPrepMode
        ? "Copy all Storyboard scene-card inputs as JSON and open the video prompt GPT."
        : "Copy all Image Prep scene-card inputs as JSON and open the Krea 2 text-to-image prompt GPT.";
      gemmaAllButton.textContent = `${runnerName} ${kind} All`;
      gemmaAllButton.title = isVideoPrepMode
        ? "Choose whether to create only missing video prompts or redo all visible scenes. If a scene has an image path, local vision uses it as guidance."
        : "Create text-to-image prompts for the visible scenes with the selected LLM runner.";
    }
  }

  function setMode(mode) {
    if (focusedSection && (!allowImagePrep || focusedSection === "story")) mode = "image_to_video_prep";
    state.mode = mode;
    const isVideoPrepMode = mode === "image_to_video_prep";
    const videoStyleEligible = state.projectVideoEngine === "ltx"
      || (state.projectVideoEngine === "minimax_h3" && ["text_to_video", "reference_to_video"].includes(state.miniMaxH3Mode))
      || state.scenes.some((scene) => storyboardSceneSupportsVideoStyle(scene));
    stepPrompts.style.background = mode === "storyboard_prompts" ? "#0e7490" : "#2b2b30";
    stepPrompts.style.borderColor = mode === "storyboard_prompts" ? "#06b6d4" : "#3f3f46";
    stepPrep.style.background = mode === "image_to_video_prep" ? "#0e7490" : "#2b2b30";
    stepPrep.style.borderColor = mode === "image_to_video_prep" ? "#06b6d4" : "#3f3f46";
    shell.querySelector("#vrgdg-storyboard-mode-pill").textContent = mode === "image_to_video_prep" ? "Video Prep" : "Planning";
    shell.querySelector("#vrgdg-storyboard-subtitle").textContent = mode === "image_to_video_prep"
      ? "Use scene images with vision guidance to create video prompts before rendering."
      : "Create text-to-image prompts for each scene before image generation.";
    note.textContent = mode === "image_to_video_prep"
      ? "Video Prep uses existing scene images when available, plus subjects, locations, lyrics, story beats, and motion notes to create video prompts."
      : "Image Prep creates text-to-image prompts from subjects, locations, lyrics, story beats, shot direction, and the story layer.";
    if (focusedSection === "story") {
      shell.querySelector("#vrgdg-storyboard-mode-pill").textContent = "Story";
      shell.querySelector("#vrgdg-storyboard-subtitle").textContent = "Plan the story shared by your scenes.";
      note.textContent = "Develop the overall idea, story arc and scene beats. Save to update the project.";
    } else if (focusedSection === "defaults") {
      shell.querySelector("#vrgdg-storyboard-subtitle").textContent = "Set project defaults. Use Fill Missing or Replace All to update existing scenes.";
    }
    refreshActionButtons();
    importImagePromptsButton.style.display = "";
    importImagePromptsButton.title = isVideoPrepMode
      ? "Paste JSON from the video prompt GPT and update Video Prep prompts."
      : "Paste JSON from the Text to Image Prompt Builder GPT and update Image Prep prompts.";
    imageShotControls.style.display = isVideoPrepMode ? "none" : "flex";
    imageShotInfo.style.display = isVideoPrepMode ? "none" : "";
    imageAestheticControls.style.display = isVideoPrepMode ? "none" : "flex";
    imageAestheticInfo.style.display = isVideoPrepMode ? "none" : "";
    videoStyleControls.style.display = isVideoPrepMode && videoStyleEligible ? "flex" : "none";
    videoStyleCustomControls.style.display = isVideoPrepMode && videoStyleEligible && state.videoStyle === "custom" ? "flex" : "none";
    videoStyleInfo.style.display = isVideoPrepMode && videoStyleEligible ? "" : "none";
    const temporalEffectEligible = isVideoPrepMode;
    temporalEffectControls.style.display = temporalEffectEligible ? "flex" : "none";
    temporalEffectCustomControls.style.display = temporalEffectEligible && state.temporalWorldEffect === "custom" ? "flex" : "none";
    temporalEffectOptions.style.display = temporalEffectEligible && Boolean(state.temporalWorldEffect) ? "flex" : "none";
    temporalProtectedCustomControls.style.display = temporalEffectEligible && Boolean(state.temporalWorldEffect) && state.temporalProtectedCharacters === "custom" ? "flex" : "none";
    temporalEffectInfo.style.display = temporalEffectEligible ? "" : "none";
    fxControls.style.display = isVideoPrepMode ? "flex" : "none";
    fxInfo.style.display = isVideoPrepMode ? "" : "none";
    fxCustomControls.style.display = isVideoPrepMode && state.fxPreset === "custom" ? "flex" : "none";
    imageWorldStyleControls.style.display = isVideoPrepMode ? "none" : "flex";
    imageWorldStyleInfo.style.display = isVideoPrepMode ? "none" : "";
    imageCustomStyleControls.style.display = isVideoPrepMode ? "none" : "flex";
    imageCustomStyleInfo.style.display = isVideoPrepMode ? "none" : "";
    cameraFlowControls.style.display = isVideoPrepMode ? "flex" : "none";
    cameraFlowInfo.style.display = isVideoPrepMode ? "" : "none";
    cameraSpeedControls.style.display = isVideoPrepMode ? "flex" : "none";
    cameraSpeedInfo.style.display = isVideoPrepMode ? "" : "none";
    const cutFrequencyEligible = isVideoPrepMode;
    cutFrequencyControls.style.display = cutFrequencyEligible ? "flex" : "none";
    cutFrequencyInfo.style.display = cutFrequencyEligible ? "" : "none";
    characterSpeedControls.style.display = isVideoPrepMode ? "flex" : "none";
    characterSpeedInfo.style.display = isVideoPrepMode ? "" : "none";
    refreshConsistencyInfo();
    refreshSetupPanelSummaries();
    renderTable();
  }

  function isFullyCustomShortFilm() {
    return isMiniMaxShortFilmMode
    && normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode) === "fully_custom";
  }

  function currentRows() {
    const q = state.query.trim().toLowerCase();
    if (!q) return state.scenes;
    return state.scenes.filter((scene) => [
      scene.label,
      scene.lyrics,
      scene.lyric_section,
      scene.story_beat,
      scene.prompt_summary,
      scene.motion_summary,
      scene.setting,
      scene.shot_type,
      ...(scene.subjects || []),
    ].join(" ").toLowerCase().includes(q));
  }

  wireStoryboardEvents({
    add, adjacentLyricContextInput, applyCameraFlow, applyDialoguePlanButton, applyFacialPerformance,
    applyFilmDialoguePlanToVideoBuilder, applyImageAesthetic, applyImageShotFlow, applyPerformanceStyle,
    applyVideoStyle, cameraFlowApply, cameraFlowReplace, cameraFlowSelect, cameraSpeedHint, cameraSpeedInput,
    characterSpeedHint, characterSpeedInput, storyArcDetailSelect, clearAllStoryboardPrompts, clearAllStoryboardStoryBeats,
    clearPromptsButton, clearStoryBeatsButton, consistencyInput, copyStoryboardForGpt, copyStoryLayerForGpt,
    createAllSceneBeatsWithGemma, createMissingBeatsButton, createStoryArcButton, createStoryArcWithGemma,
    createStorySequenceButton, createStorySequenceWithGemma,
    createStoryBriefButton, createStoryBriefWithGemma, cutFrequencyHint, cutFrequencyInput,
    detectLyricSections, detectSectionsButton, exportPromptFiles, exportPrompts, facialApply,
    facialCustomInput, facialReplace, facialSelect, fxCustomInput, fxSelect, gemmaAllButton, gptButton,
    gptStoryButton, handleReplaceSceneBeats, imageAestheticApply, imageAestheticPresets,
    imageAestheticReplace, imageAestheticSelect, imageCustomStyleInput, imageShotApply, imageShotFlowPresets,
    imageShotReplace, imageShotSelect, imageWorldStyleSelect, importImagePromptsButton, importStoryJsonButton,
    keepGemmaLoadedInput, lyricStoryStrengthHintButton, lyricStoryStrengthInput,
    notifyStoryboardDefaultsChanged, openCustomCameraFlowDialog, openImportImagePromptsFromGptModal,
    openImportStoryJsonModal, openMiniMaxScriptMapper, openMiniMaxScriptMapperButton, openSceneEditor,
    overallStoryIdeaInput, performanceApply, performanceReplace, performanceSelect, planDialogueScenesButton,
    planFilmDialogueScenesWithLlm, promptRunnerName, refreshCameraFlowInfo, refreshCameraSpeedInfo,
    refreshCharacterSpeedInfo, refreshConsistencyInfo, refreshCutFrequencyInfo, refreshFacialInfo,
    refreshFxInfo, refreshImageAestheticInfo, refreshImageShotInfo, refreshImageWorldStyleInfo,
    refreshPerformanceInfo, refreshSetupPanelSummaries, refreshTemporalEffectInfo, refreshVideoStyleInfo,
    renderTable, replaceBeatsButton, save, saveStoryboard, search, setMode, shortFilmPlanningModeSelect,
    songStoryBriefInput, startAllPromptsWithGemma, state, stepPrep, stepPrompts, storyLayerEnabledInput,
    syncLyricStoryStrengthLabel, syncStoryLayerFromInputs, temporalEffectCustomInput, temporalEffectSelect,
    temporalEnvironmentInput, temporalExtrasInput, temporalIntensityInput, temporalProtectedCustomInput,
    temporalProtectedSelect, userStoryArcInput, videoStyleApply, videoStyleCustomInput, videoStyleReplace,
    videoStyleSelect,
  });
  function closeStoryboard() { if (state.saving) return; backdrop.remove(); payload.onClose?.(); }
  close.onclick = closeStoryboard;
  backdrop.addEventListener("pointerdown", (event) => {
    if (event.target === backdrop) closeStoryboard();
  });
  refreshCameraFlowInfo();
  refreshImageShotInfo();
  refreshImageAestheticInfo();
  refreshVideoStyleInfo();
  refreshTemporalEffectInfo();
  refreshFxInfo();
  refreshImageWorldStyleInfo();
  refreshConsistencyInfo();
  refreshCameraSpeedInfo();
  refreshCutFrequencyInfo();
  refreshPerformanceInfo();
  refreshCharacterSpeedInfo();
  refreshFacialInfo();
  setMode(state.mode || "storyboard_prompts");
  loadExisting().then(async (loaded) => {
    if (promptActionOnly) {
      try {
        setMode("image_to_video_prep");
        await startAllPromptsWithGemma();
      } finally {
        closeStoryboard();
      }
      return;
    }
    if (!focusSceneId) return;
    const target = loaded && state.scenes.find((scene) => scene.id === focusSceneId);
    if (target) {
      openSceneEditor(target);
    } else {
      sceneFocus.only = false;
      delete backdrop.dataset.vrgdgFocusedStoryboard;
      backdrop.style.display = "";
    }
  });
}
