import {
  normalizeScene,
  normalizeStoryboardShortFilmPlanningMode,
  storyboardCutFrequencyValue,
  storyboardSpeedValue,
} from "./scenes.mjs";
import { normalizeStoryboardCustomCameraFlowSequence, STORYBOARD_CAMERA_FLOW_PRESETS } from "./shot_presets.mjs";
import {
  MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS,
  MINIMAX_VIDEO_STYLE_PRESETS,
  STORYBOARD_FX_PRESETS,
  storyboardTemporalIntensity,
  storyboardTemporalProtectedMode,
} from "./video_style.mjs";

export function wireStoryboardEvents({
  add, adjacentLyricContextInput, applyCameraFlow, applyDialoguePlanButton, applyFacialPerformance,
  applyFilmDialoguePlanToVideoBuilder, applyImageAesthetic, applyImageShotFlow, applyPerformanceStyle,
  applyVideoStyle, cameraFlowApply, cameraFlowReplace, cameraFlowSelect, cameraSpeedHint, cameraSpeedInput,
  characterSpeedHint, characterSpeedInput, clearAllStoryboardPrompts, clearAllStoryboardStoryBeats,
  clearPromptsButton, clearStoryBeatsButton, consistencyInput, copyStoryboardForGpt, copyStoryLayerForGpt,
  createAllSceneBeatsWithGemma, createMissingBeatsButton, createStoryArcButton, createStoryArcWithGemma,
  createStoryBriefButton, createStoryBriefWithGemma, cutFrequencyHint, cutFrequencyInput, detectLyricSections,
  detectSectionsButton, exportPromptFiles, exportPrompts, facialApply, facialCustomInput, facialReplace,
  facialSelect, fxCustomInput, fxSelect, gemmaAllButton, gptButton, gptStoryButton, handleReplaceSceneBeats,
  imageAestheticApply, imageAestheticPresets, imageAestheticReplace, imageAestheticSelect,
  imageCustomStyleInput, imageShotApply, imageShotFlowPresets, imageShotReplace, imageShotSelect,
  imageWorldStyleSelect, importImagePromptsButton, importStoryJsonButton, keepGemmaLoadedInput,
  lyricStoryStrengthHintButton, lyricStoryStrengthInput, notifyStoryboardDefaultsChanged,
  openCustomCameraFlowDialog, openImportImagePromptsFromGptModal, openImportStoryJsonModal,
  openMiniMaxScriptMapper, openMiniMaxScriptMapperButton, openSceneEditor, overallStoryIdeaInput,
  performanceApply, performanceReplace, performanceSelect, planDialogueScenesButton,
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
}) {
  stepPrompts.onclick = () => setMode("storyboard_prompts");
  stepPrep.onclick = () => setMode("image_to_video_prep");
  search.oninput = () => {
    state.query = search.value || "";
    renderTable();
  };
  cameraFlowSelect.onchange = async () => {
    const previous = state.cameraFlow;
    const next = STORYBOARD_CAMERA_FLOW_PRESETS[cameraFlowSelect.value] ? cameraFlowSelect.value : "balanced";
    if (next === "custom") {
      state.cameraFlow = "custom";
      const imported = await openCustomCameraFlowDialog();
      if (imported) {
        state.customCameraFlowSequence = normalizeStoryboardCustomCameraFlowSequence(imported);
        state.cameraFlow = "custom";
      } else {
        state.cameraFlow = previous === "custom" && state.customCameraFlowSequence.length ? "custom" : (STORYBOARD_CAMERA_FLOW_PRESETS[previous] ? previous : "balanced");
      }
    } else {
      state.cameraFlow = next;
    }
    cameraFlowSelect.value = state.cameraFlow;
    refreshCameraFlowInfo();
    notifyStoryboardDefaultsChanged();
  };
  imageShotSelect.onchange = () => {
    state.imageShotFlow = imageShotFlowPresets[imageShotSelect.value] ? imageShotSelect.value : Object.keys(imageShotFlowPresets)[0] || "off";
    imageShotSelect.value = state.imageShotFlow;
    refreshImageShotInfo();
    notifyStoryboardDefaultsChanged();
  };
  imageAestheticSelect.onchange = () => {
    state.imageAesthetic = imageAestheticPresets.some((preset) => preset.value === imageAestheticSelect.value) ? imageAestheticSelect.value : imageAestheticPresets[0]?.value || "";
    imageAestheticSelect.value = state.imageAesthetic;
    refreshImageAestheticInfo();
    notifyStoryboardDefaultsChanged();
  };
  videoStyleSelect.onchange = () => {
    state.videoStyle = MINIMAX_VIDEO_STYLE_PRESETS.some((preset) => preset.value === videoStyleSelect.value) ? videoStyleSelect.value : "";
    videoStyleSelect.value = state.videoStyle;
    refreshVideoStyleInfo();
    notifyStoryboardDefaultsChanged();
  };
  videoStyleCustomInput.addEventListener("input", () => {
    state.videoStyleCustom = videoStyleCustomInput.value;
    refreshVideoStyleInfo();
  });
  videoStyleCustomInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  temporalEffectSelect.onchange = () => {
    const previous = state.temporalWorldEffect;
    state.temporalWorldEffect = MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS.some((preset) => preset.value === temporalEffectSelect.value)
      ? temporalEffectSelect.value
      : "";
    temporalEffectSelect.value = state.temporalWorldEffect;
    if (!previous && state.temporalWorldEffect) {
      state.temporalAllowBackgroundExtras = true;
      state.temporalEnvironmentTimePassage = true;
      temporalExtrasInput.checked = true;
      temporalEnvironmentInput.checked = true;
    }
    refreshTemporalEffectInfo();
    notifyStoryboardDefaultsChanged();
  };
  temporalEffectCustomInput.addEventListener("input", () => {
    state.temporalWorldEffectCustom = temporalEffectCustomInput.value;
    refreshTemporalEffectInfo();
  });
  temporalEffectCustomInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  fxSelect.onchange = () => {
    state.fxPreset = STORYBOARD_FX_PRESETS.some((preset) => preset.value === fxSelect.value) ? fxSelect.value : "";
    fxSelect.value = state.fxPreset;
    refreshFxInfo();
    notifyStoryboardDefaultsChanged();
  };
  fxCustomInput.addEventListener("input", () => {
    state.fxCustomJson = fxCustomInput.value;
    refreshFxInfo();
  });
  fxCustomInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  temporalExtrasInput.onchange = () => {
    state.temporalAllowBackgroundExtras = temporalExtrasInput.checked;
    refreshTemporalEffectInfo();
    notifyStoryboardDefaultsChanged();
  };
  temporalEnvironmentInput.onchange = () => {
    state.temporalEnvironmentTimePassage = temporalEnvironmentInput.checked;
    refreshTemporalEffectInfo();
    notifyStoryboardDefaultsChanged();
  };
  temporalIntensityInput.addEventListener("input", () => {
    state.temporalBackgroundIntensity = storyboardTemporalIntensity(temporalIntensityInput.value);
    refreshTemporalEffectInfo();
  });
  temporalIntensityInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  temporalProtectedSelect.onchange = () => {
    state.temporalProtectedCharacters = storyboardTemporalProtectedMode(temporalProtectedSelect.value);
    temporalProtectedSelect.value = state.temporalProtectedCharacters;
    refreshTemporalEffectInfo();
    notifyStoryboardDefaultsChanged();
  };
  temporalProtectedCustomInput.addEventListener("input", () => {
    state.temporalProtectedCustom = temporalProtectedCustomInput.value;
    refreshTemporalEffectInfo();
  });
  temporalProtectedCustomInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  consistencyInput.addEventListener("input", () => {
    state.globalConsistencyPhrase = consistencyInput.value.trim();
    refreshConsistencyInfo();
  });
  consistencyInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  cameraSpeedInput.addEventListener("input", () => {
    state.cameraMotionSpeed = storyboardSpeedValue(cameraSpeedInput.value, 4);
    cameraSpeedInput.value = String(state.cameraMotionSpeed);
    refreshCameraSpeedInfo();
  });
  cameraSpeedInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  cameraSpeedHint.onclick = () => {
    window.alert([
      `Camera Motion Speed controls how much movement ${promptRunnerName()}/GPT should put into the camera plan for Video Prep.`,
      "",
      "0: locked-off static camera.",
      "1-3: slow, gentle camera motion; one simple move at most.",
      "4-6: controlled cinematic movement like tracking, pan, dolly, crane, reveal, or orbit.",
      "7-8: energetic movement with stronger tracking, orbit, whip pan, rise, reveal, or compound motion.",
      "9-10: fast action camera language; multiple coordinated moves can happen in one scene while keeping the subject readable.",
    ].join("\n"));
  };
  cutFrequencyInput.addEventListener("input", () => {
    state.cutFrequency = storyboardCutFrequencyValue(cutFrequencyInput.value);
    cutFrequencyInput.value = String(state.cutFrequency);
    refreshCutFrequencyInfo();
  });
  cutFrequencyInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  cutFrequencyHint.onclick = () => {
    window.alert([
      "Cut Frequency controls editing inside each timeline segment.",
      "",
      "0: one smooth continuous take with no cuts.",
      "1-3: occasional cuts, scaled to the segment's exact duration.",
      "4-6: a moderate number of evenly spaced cuts.",
      "7-9: frequent cuts with short coherent coverage shots.",
      "10: maximum frequency — request a new continuity-preserving angle every second.",
      "",
      "Example: a 5-second segment at 10 starts with shot 1, then cuts at 1s, 2s, 3s, and 4s.",
      "LTX writes cuts in ordinary language such as 'then cut to'; MiniMax keeps its structured CUT TO format.",
      "Changing this setting affects newly generated prompts; it does not rewrite existing prompts automatically.",
    ].join("\n"));
  };
  cameraFlowApply.onclick = () => applyCameraFlow({ overwrite: false });
  cameraFlowReplace.onclick = () => applyCameraFlow({ overwrite: true });
  imageShotApply.onclick = () => applyImageShotFlow({ overwrite: false });
  imageShotReplace.onclick = () => applyImageShotFlow({ overwrite: true });
  imageAestheticApply.onclick = () => applyImageAesthetic({ overwrite: false });
  imageAestheticReplace.onclick = () => applyImageAesthetic({ overwrite: true });
  videoStyleApply.onclick = () => applyVideoStyle({ overwrite: false });
  videoStyleReplace.onclick = () => applyVideoStyle({ overwrite: true });
  performanceSelect.onchange = () => {
    state.performanceStyle = String(performanceSelect.value || "");
    refreshPerformanceInfo();
    notifyStoryboardDefaultsChanged();
  };
  characterSpeedInput.addEventListener("input", () => {
    state.characterMotionSpeed = storyboardSpeedValue(characterSpeedInput.value, 4);
    characterSpeedInput.value = String(state.characterMotionSpeed);
    refreshCharacterSpeedInfo();
  });
  characterSpeedInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  characterSpeedHint.onclick = () => {
    window.alert([
      "Character Motion Speed controls how active the subject's body movement should be.",
      "",
      "0: subject stays still or holds a pose.",
      "1-3: subtle motion like gestures, turns, swaying, reaching, or small steps.",
      "4-6: active performance like walking, dancing, interacting with objects, or using the set.",
      "7-8: energetic action like running, hard dancing, climbing, struggling, spinning, or crossing the space.",
      "9-10: fast action movement like sprinting, explosive dance, chase beats, rapid direction changes, or intense physical performance.",
    ].join("\n"));
  };
  facialSelect.onchange = () => {
    state.facialPerformance = String(facialSelect.value || "");
    refreshFacialInfo();
    notifyStoryboardDefaultsChanged();
  };
  facialCustomInput.oninput = () => {
    state.facialPerformanceCustom = String(facialCustomInput.value || "");
    refreshFacialInfo();
  };
  facialCustomInput.addEventListener("change", notifyStoryboardDefaultsChanged);
  performanceApply.onclick = () => applyPerformanceStyle({ overwrite: false });
  performanceReplace.onclick = () => applyPerformanceStyle({ overwrite: true });
  facialApply.onclick = () => applyFacialPerformance({ overwrite: false });
  facialReplace.onclick = () => applyFacialPerformance({ overwrite: true });
  add.onclick = () => {
    const next = normalizeScene({ scene_number: state.scenes.length + 1, label: `Scene ${state.scenes.length + 1}` }, state.scenes.length);
    state.scenes.push(next);
    openSceneEditor(next);
    renderTable();
  };
  gptButton.onclick = copyStoryboardForGpt;
  importImagePromptsButton.onclick = openImportImagePromptsFromGptModal;
  gptStoryButton.onclick = copyStoryLayerForGpt;
  importStoryJsonButton.onclick = openImportStoryJsonModal;
  gemmaAllButton.onclick = startAllPromptsWithGemma;
  clearPromptsButton.onclick = clearAllStoryboardPrompts;
  clearStoryBeatsButton.onclick = clearAllStoryboardStoryBeats;
  adjacentLyricContextInput.onchange = () => {
    state.sendAdjacentLyricContext = adjacentLyricContextInput.checked;
    syncStoryLayerFromInputs({ notify: true });
  };
  storyLayerEnabledInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  shortFilmPlanningModeSelect.addEventListener("change", () => {
    state.shortFilmPlanningMode = normalizeStoryboardShortFilmPlanningMode(shortFilmPlanningModeSelect.value);
    shortFilmPlanningModeSelect.value = state.shortFilmPlanningMode;
    refreshSetupPanelSummaries();
    notifyStoryboardDefaultsChanged();
    renderTable();
  });
  imageWorldStyleSelect.addEventListener("change", () => {
    refreshImageWorldStyleInfo();
    syncStoryLayerFromInputs({ notify: true });
  });
  imageCustomStyleInput.addEventListener("input", () => {
    refreshImageWorldStyleInfo();
    syncStoryLayerFromInputs();
  });
  imageCustomStyleInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  lyricStoryStrengthInput.addEventListener("input", () => {
    syncLyricStoryStrengthLabel();
    syncStoryLayerFromInputs();
  });
  lyricStoryStrengthInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  lyricStoryStrengthHintButton.onclick = () => {
    window.alert([
      `Lyric Story Strength controls how literally ${promptRunnerName()} should follow the lyrics when creating the story arc, story brief, scene beats, and prompt context.`,
      "",
      "0: do not use lyrics as story source.",
      "1-3: use lyrics as mood and emotional timing only.",
      "4-6: balance lyrics with the story arc, subjects, and locations.",
      "7-8: lyrics strongly shape the scene story; include recognizable lyric anchors when possible.",
      "9-10: use lyrics as literally as possible; non-instrumental scenes should include a concrete object, action, emotion, or situation from the exact lyric line whenever possible.",
    ].join("\n"));
  };
  overallStoryIdeaInput.addEventListener("input", syncStoryLayerFromInputs);
  overallStoryIdeaInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  userStoryArcInput.addEventListener("input", syncStoryLayerFromInputs);
  userStoryArcInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  songStoryBriefInput.addEventListener("input", syncStoryLayerFromInputs);
  songStoryBriefInput.addEventListener("change", () => syncStoryLayerFromInputs({ notify: true }));
  createStoryArcButton.onclick = createStoryArcWithGemma;
  createStoryBriefButton.onclick = createStoryBriefWithGemma;
  createMissingBeatsButton.onclick = () => createAllSceneBeatsWithGemma();
  replaceBeatsButton.onclick = handleReplaceSceneBeats;
  detectSectionsButton.onclick = detectLyricSections;
  openMiniMaxScriptMapperButton.onclick = openMiniMaxScriptMapper;
  planDialogueScenesButton.onclick = planFilmDialogueScenesWithLlm;
  applyDialoguePlanButton.onclick = applyFilmDialoguePlanToVideoBuilder;
  keepGemmaLoadedInput.onchange = () => {
    state.gemmaSettings = {
      ...(state.gemmaSettings || {}),
      keep_loaded_for_storyboard_all: Boolean(keepGemmaLoadedInput.checked),
    };
  };
  save.onclick = saveStoryboard;
  exportPrompts.onclick = exportPromptFiles;
}
