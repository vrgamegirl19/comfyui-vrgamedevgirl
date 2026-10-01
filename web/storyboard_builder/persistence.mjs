import { postJson } from "./api.mjs";
import { copyTextToClipboard, createToast } from "./controls.mjs";
import { openStoryboardGptUrl, storyboardGptPayload } from "./gpt_payload.mjs";
import {
  mergeReferenceBuilderCatalog,
  normalizeReferenceBuilderCatalog,
  storyboardSubjectNamesFromRefs,
} from "./references.mjs";
import {
  normalizeStoryArcDetail,
  ensureStoryboardReferenceOpening,
  normalizeScene,
  normalizeStoryboardMiniMaxH3Mode,
  normalizeStoryboardPerformanceMode,
  normalizeStoryboardProjectVideoEngine,
  normalizeStoryboardShortFilmPlanningMode,
  normalizeStoryLayer,
  normalizeVideoPromptOrigin,
  slimSceneForRequest,
  slimStoryboardForRequest,
  storyboardCutFrequencyValue,
  storyboardSpeedValue,
} from "./scenes.mjs";
import { normalizeStoryboardScriptImportState } from "./script_import.mjs";
import { normalizeStoryboardCustomCameraFlowSequence, STORYBOARD_CAMERA_FLOW_PRESETS } from "./shot_presets.mjs";
import {
  MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS,
  MINIMAX_VIDEO_STYLE_PRESETS,
  STORYBOARD_FX_PRESETS,
  storyboardTemporalIntensity,
  storyboardTemporalProtectedMode,
} from "./video_style.mjs";



export function createStoryboardPersistence({
  absorbSceneReferencesIntoCatalog, cameraFlowSelect, cameraSpeedInput, characterSpeedInput, storyArcDetailSelect, consistencyInput,
  cutFrequencyInput, enforceStoryboardVideoFacialRequirements, exportPrompts, facialCustomInput,
  facialPerformancePresets, facialSelect, focusedSection, fxCustomInput, fxSelect, getSelectedScenes,
  imageAestheticPresets, imageAestheticSelect, imageShotFlowPresets, imageShotSelect,
  incomingProjectVideoEngine, lyricStoryStrengthInput, openingMode, overallStoryIdeaInput, payload,
  payloadVideoPromptType, performanceSelect, performanceStylePresets, refreshCameraFlowInfo,
  refreshCameraSpeedInfo, refreshCharacterSpeedInfo, refreshConsistencyInfo, refreshCutFrequencyInfo,
  refreshFacialInfo, refreshFxInfo, refreshImageAestheticInfo, refreshImageShotInfo, refreshPerformanceInfo,
  refreshTemporalEffectInfo, refreshVideoStyleInfo, renderTable, save, setMode, shortFilmPlanningModeSelect,
  songStoryBriefInput, state, storyLayerEnabledInput, storyboardDefaultsPayload, syncLyricStoryStrengthLabel,
  syncReferenceMappingsToVideoCreator, syncStoryLayerFromInputs, temporalEffectCustomInput,
  temporalEffectSelect, temporalEnvironmentInput, temporalExtrasInput, temporalIntensityInput,
  temporalProtectedCustomInput, temporalProtectedSelect, userStoryArcInput, videoStyleCustomInput,
  videoStyleSelect,
}) {
  const hasIncomingProjectVideoEngine = Boolean(incomingProjectVideoEngine);

  async function loadExisting() {
    if (!state.projectFolder) {
      renderTable();
      return true;
    }
    try {
      const incomingScenes = state.scenes.map((scene) => normalizeScene(scene));
      const data = await postJson("/vrgdg/storyboard/load", { project_folder: state.projectFolder });
      const saved = data.storyboard || {};
      if (!hasIncomingProjectVideoEngine) {
        state.projectVideoEngine = normalizeStoryboardProjectVideoEngine(saved.project_video_engine || saved.projectVideoEngine || state.projectVideoEngine);
      }
      const savedReferences = normalizeReferenceBuilderCatalog(saved.reference_builder || saved.referenceBuilder || {});
      const currentHasSubjects = Array.isArray(state.referenceBuilder?.subjects) && state.referenceBuilder.subjects.length > 0;
      const currentHasLocations = Array.isArray(state.referenceBuilder?.locations) && state.referenceBuilder.locations.length > 0;
      const currentLocationsCleared = Boolean(state.referenceBuilder?.locations_cleared);
      if ((!currentHasSubjects && savedReferences.subjects.length) || (!currentHasLocations && !currentLocationsCleared && savedReferences.locations.length)) {
        const nextReferences = {
          subjects: currentHasSubjects ? state.referenceBuilder.subjects : savedReferences.subjects,
          locations: currentLocationsCleared ? [] : (currentHasLocations ? state.referenceBuilder.locations : savedReferences.locations),
          locations_cleared: currentLocationsCleared,
        };
        state.referenceBuilder = normalizeReferenceBuilderCatalog(nextReferences);
      } else if (!currentHasSubjects && !currentHasLocations && !currentLocationsCleared && (savedReferences.subjects.length || savedReferences.locations.length)) {
        state.referenceBuilder = mergeReferenceBuilderCatalog(state.referenceBuilder, savedReferences);
      }
      if (Array.isArray(saved.scenes) && saved.scenes.length) {
        const savedScenes = saved.scenes.map((scene, index) => normalizeScene(scene, index));
        const scenesToShow = incomingScenes.length ? incomingScenes : savedScenes;
        state.scenes = scenesToShow.map((fresh, index) => {
          const normalized = savedScenes.find((item) => item.id === fresh.id)
            || savedScenes.find((item) => Number(item.scene_number) === Number(fresh.scene_number))
            || null;
          if (!normalized) return normalizeScene(fresh, index);
          // Live scenes from the Video Builder own the lyric-review flags (B-roll / no lip-sync, instrumental,
          // no character, singers) and the Wizard / Scene Defaults fields (shot, camera, performance, facial).
          // The saved storyboard copy can be older, so it must not overwrite them.
          const liveOwned = incomingScenes.length
            ? {
              lyric_singers: fresh.lyric_singers,
              lyric_no_lip_sync: fresh.lyric_no_lip_sync,
              lyric_instrumental: fresh.lyric_instrumental,
              performance_style: fresh.performance_style,
              facial_performance: fresh.facial_performance,
              facial_performance_custom: fresh.facial_performance_custom,
              shot_type: fresh.shot_type || normalized.shot_type,
              camera_motion: fresh.camera_motion || normalized.camera_motion,
              character_motion: fresh.character_motion || normalized.character_motion,
            }
            : {};
          const noCharacterPresent = incomingScenes.length
            ? Boolean(fresh.no_character_present)
            : Boolean(fresh.no_character_present || normalized.no_character_present);
          const subjectRefs = incomingScenes.length ? (fresh.subject_refs || []) : (fresh.subject_refs?.length ? fresh.subject_refs : normalized.subject_refs);
          const subjects = subjectRefs?.length
            ? storyboardSubjectNamesFromRefs(subjectRefs)
            : Array.from(new Set([
              ...(fresh.subjects || []),
              ...(normalized.subjects || []),
            ].map((item) => String(item || "").trim()).filter(Boolean)));
          return {
            ...normalized,
            ...liveOwned,
            id: fresh.id || normalized.id,
            scene_number: fresh.scene_number || normalized.scene_number,
            label: fresh.label || normalized.label,
            video_prompt_type: payloadVideoPromptType || fresh.video_prompt_type || normalized.video_prompt_type,
            project_video_engine: state.projectVideoEngine,
            minimax_h3_mode: normalizeStoryboardMiniMaxH3Mode(fresh.minimax_h3_mode || normalized.minimax_h3_mode),
            timeline_start: Number(fresh.timeline_start ?? normalized.timeline_start ?? 0),
            timeline_end: Number(fresh.timeline_end ?? normalized.timeline_end ?? 0),
            exact_duration: Math.max(0, Number(fresh.exact_duration ?? normalized.exact_duration ?? 0)),
            lyrics: fresh.lyrics || normalized.lyrics,
            lyric_section: fresh.lyric_section || normalized.lyric_section,
            story_beat: fresh.story_beat || normalized.story_beat,
            audio_direction: fresh.audio_direction || normalized.audio_direction,
            continuity: fresh.continuity || normalized.continuity,
            flf_start_state: fresh.flf_start_state || normalized.flf_start_state,
            flf_transformation: fresh.flf_transformation || normalized.flf_transformation,
            flf_end_state: fresh.flf_end_state || normalized.flf_end_state,
            flf_carry_forward: fresh.flf_carry_forward || normalized.flf_carry_forward,
            performance_mode: fresh.performance_mode || normalized.performance_mode || state.performanceMode,
            prompt_summary: state.mode === "image_to_video_prep" ? (fresh.prompt_summary || normalized.prompt_summary) : "",
            motion_summary: fresh.motion_summary || normalized.motion_summary,
            temporal_world_effect_override: fresh.temporal_world_effect_override || normalized.temporal_world_effect_override || "global",
            temporal_world_effect_custom: fresh.temporal_world_effect_custom || normalized.temporal_world_effect_custom || "",
            image_path: fresh.image_path || normalized.image_path,
            no_character_present: noCharacterPresent,
            subjects,
            subject_refs: noCharacterPresent ? [] : subjectRefs,
            setting: currentLocationsCleared ? "" : (fresh.location_ref?.name || normalized.setting || fresh.setting),
            location_ref: currentLocationsCleared ? null : (incomingScenes.length ? fresh.location_ref : (fresh.location_ref || normalized.location_ref)),
          };
        });
        if (currentLocationsCleared) {
          state.scenes.forEach((scene) => {
            scene.location_ref = null;
            scene.setting = "";
          });
        }
        absorbSceneReferencesIntoCatalog(state.scenes);
      }
      // The current video mode decides the opening workspace every time. Do not
      // restore a stale Image Prep/Video Prep tab from an earlier visit.
      state.mode = openingMode;
      state.performanceMode = normalizeStoryboardPerformanceMode(saved.performance_mode || saved.performanceMode || state.performanceMode);
      state.shortFilmPlanningMode = normalizeStoryboardShortFilmPlanningMode(saved.short_film_planning_mode || saved.shortFilmPlanningMode || state.shortFilmPlanningMode);
      shortFilmPlanningModeSelect.value = state.shortFilmPlanningMode;
      state.customCameraFlowSequence = normalizeStoryboardCustomCameraFlowSequence(
        saved.custom_camera_flow_sequence
        || saved.customCameraFlowSequence
        || saved.builder_storyboard_defaults?.custom_camera_flow_sequence
        || saved.builderStoryboardDefaults?.custom_camera_flow_sequence
        || state.customCameraFlowSequence,
      );
      if (saved.camera_flow && STORYBOARD_CAMERA_FLOW_PRESETS[saved.camera_flow]) {
        state.cameraFlow = saved.camera_flow;
        cameraFlowSelect.value = state.cameraFlow;
      }
      if (saved.image_shot_flow && imageShotFlowPresets[saved.image_shot_flow]) {
        state.imageShotFlow = saved.image_shot_flow;
        imageShotSelect.value = state.imageShotFlow;
      }
      state.imageAesthetic = String(saved.image_aesthetic || saved.imageAesthetic || state.imageAesthetic || "");
      if (!imageAestheticPresets.some((preset) => preset.value === state.imageAesthetic)) state.imageAesthetic = imageAestheticPresets[0]?.value || "";
      imageAestheticSelect.value = state.imageAesthetic;
      state.videoStyle = String(saved.video_style || saved.videoStyle || state.videoStyle || "");
      if (!MINIMAX_VIDEO_STYLE_PRESETS.some((preset) => preset.value === state.videoStyle)) state.videoStyle = "";
      videoStyleSelect.value = state.videoStyle;
      state.videoStyleCustom = String(saved.video_style_custom || saved.videoStyleCustom || state.videoStyleCustom || "");
      videoStyleCustomInput.value = state.videoStyleCustom;
      state.temporalWorldEffect = String(saved.temporal_world_effect || saved.temporalWorldEffect || state.temporalWorldEffect || "");
      if (!MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS.some((preset) => preset.value === state.temporalWorldEffect)) state.temporalWorldEffect = "";
      temporalEffectSelect.value = state.temporalWorldEffect;
      state.temporalWorldEffectCustom = String(saved.temporal_world_effect_custom || saved.temporalWorldEffectCustom || state.temporalWorldEffectCustom || "");
      temporalEffectCustomInput.value = state.temporalWorldEffectCustom;
      state.temporalAllowBackgroundExtras = (saved.temporal_allow_background_extras ?? saved.temporalAllowBackgroundExtras ?? state.temporalAllowBackgroundExtras) !== false;
      temporalExtrasInput.checked = state.temporalAllowBackgroundExtras;
      state.temporalBackgroundIntensity = storyboardTemporalIntensity(saved.temporal_background_intensity ?? saved.temporalBackgroundIntensity ?? state.temporalBackgroundIntensity);
      temporalIntensityInput.value = String(state.temporalBackgroundIntensity);
      state.temporalEnvironmentTimePassage = (saved.temporal_environment_time_passage ?? saved.temporalEnvironmentTimePassage ?? state.temporalEnvironmentTimePassage) !== false;
      temporalEnvironmentInput.checked = state.temporalEnvironmentTimePassage;
      state.temporalProtectedCharacters = storyboardTemporalProtectedMode(saved.temporal_protected_characters || saved.temporalProtectedCharacters || state.temporalProtectedCharacters);
      temporalProtectedSelect.value = state.temporalProtectedCharacters;
      state.temporalProtectedCustom = String(saved.temporal_protected_custom || saved.temporalProtectedCustom || state.temporalProtectedCustom || "");
      temporalProtectedCustomInput.value = state.temporalProtectedCustom;
      state.fxPreset = String(saved.fx_preset || saved.fxPreset || saved.builder_storyboard_defaults?.fx_preset || saved.builderStoryboardDefaults?.fx_preset || state.fxPreset || "");
      if (!STORYBOARD_FX_PRESETS.some((preset) => preset.value === state.fxPreset)) state.fxPreset = "";
      fxSelect.value = state.fxPreset;
      state.fxCustomJson = String(saved.fx_custom_json || saved.fxCustomJson || saved.builder_storyboard_defaults?.fx_custom_json || saved.builderStoryboardDefaults?.fx_custom_json || state.fxCustomJson || "");
      fxCustomInput.value = state.fxCustomJson;
      state.globalConsistencyPhrase = String(saved.global_consistency_phrase || saved.globalConsistencyPhrase || state.globalConsistencyPhrase || "");
      consistencyInput.value = state.globalConsistencyPhrase;
      state.performanceStyle = String(saved.performance_style_default || saved.performance_style || state.performanceStyle || "");
      if (!performanceStylePresets.some((preset) => preset.value === state.performanceStyle)) state.performanceStyle = performanceStylePresets[0]?.value || "";
      performanceSelect.value = state.performanceStyle;
      state.facialPerformance = String(saved.facial_performance_default || saved.facial_performance || state.facialPerformance || "");
      if (!facialPerformancePresets.some((preset) => preset.value === state.facialPerformance)) state.facialPerformance = facialPerformancePresets[0]?.value || "";
      state.facialPerformanceCustom = String(saved.facial_performance_custom_default || saved.facial_performance_custom || state.facialPerformanceCustom || "");
      facialSelect.value = state.facialPerformance;
      facialCustomInput.value = state.facialPerformanceCustom;
      state.cameraMotionSpeed = storyboardSpeedValue(saved.camera_motion_speed ?? saved.motion_defaults?.camera_motion_speed ?? state.cameraMotionSpeed, 4);
      state.characterMotionSpeed = storyboardSpeedValue(saved.character_motion_speed ?? saved.motion_defaults?.character_motion_speed ?? state.characterMotionSpeed, 4);
      state.cutFrequency = storyboardCutFrequencyValue(saved.minimax_h3_cut_frequency ?? saved.cut_frequency ?? state.cutFrequency);
      cameraSpeedInput.value = String(state.cameraMotionSpeed);
      characterSpeedInput.value = String(state.characterMotionSpeed);
      state.storyArcDetail = normalizeStoryArcDetail(saved.story_arc_detail ?? state.storyArcDetail);
      storyArcDetailSelect.value = state.storyArcDetail;
      storyArcDetailSelect.dispatchEvent(new Event("input"));
      cutFrequencyInput.value = String(state.cutFrequency);
      state.storyLayer = normalizeStoryLayer(saved.story_layer || saved.storyLayer || {});
      state.scriptImport = normalizeStoryboardScriptImportState(saved.script_import || saved.scriptImport || state.scriptImport || {});
      storyLayerEnabledInput.checked = state.storyLayer.enabled !== false;
      overallStoryIdeaInput.value = state.storyLayer.overall_story_idea || "";
      userStoryArcInput.value = state.storyLayer.user_story_arc || "";
      songStoryBriefInput.value = state.storyLayer.song_story_brief || "";
      lyricStoryStrengthInput.value = String(state.storyLayer.lyric_story_strength ?? 7);
      syncLyricStoryStrengthLabel();
      refreshCameraFlowInfo();
      refreshImageShotInfo();
      refreshImageAestheticInfo();
      refreshVideoStyleInfo();
      refreshTemporalEffectInfo();
      refreshFxInfo();
      refreshConsistencyInfo();
      refreshCameraSpeedInfo();
      refreshCutFrequencyInfo();
      refreshPerformanceInfo();
      refreshCharacterSpeedInfo();
      refreshFacialInfo();
      setMode(state.mode);
      syncReferenceMappingsToVideoCreator();
      return true;
    } catch (error) {
      createToast(String(error?.message || error), true);
      renderTable();
      return false;
    }
  }

  async function copyStoryboardForGpt() {
    const selectedScenes = getSelectedScenes();
    if (selectedScenes.length > 0) {
      try {
        const payload = storyboardGptPayload(state, selectedScenes);
        const text = JSON.stringify(payload, null, 2);
        await copyTextToClipboard(text);
        openStoryboardGptUrl(payload);
        createToast(selectedScenes.length === 1
          ? `Copied GPT JSON for ${selectedScenes[0].label || `Scene ${selectedScenes[0].scene_number}`} and opened GPT.`
          : `Copied GPT JSON for ${selectedScenes.length} selected scenes and opened GPT.`);
      } catch (error) {
        createToast(`Could not copy GPT JSON:\n${String(error?.message || error)}`, true);
      }
      return;
    }
    try {
      const payload = storyboardGptPayload(state);
      const text = JSON.stringify(payload, null, 2);
      await copyTextToClipboard(text);
      openStoryboardGptUrl(payload);
      createToast(`Copied Storyboard GPT JSON for ${payload.scenes.length} scenes and opened GPT.`);
    } catch (error) {
      createToast(`Could not copy Storyboard GPT JSON:\n${String(error?.message || error)}`, true);
    }
  }

  async function saveStoryboard({ throwOnError = false } = {}) {
    if (!state.projectFolder) {
      const message = "Save the AI Video Builder project first so Storyboard Builder knows where to write files.";
      if (throwOnError) throw new Error(message);
      createToast(message, true);
      return;
    }
    state.saving = true;
    save.disabled = true;
    try {
      syncStoryLayerFromInputs();
      state.scenes.forEach((scene) => {
        if (state.projectVideoEngine !== "minimax_h3" && String(scene.video_prompt || "").trim() && normalizeVideoPromptOrigin(scene.video_prompt_origin) === "gemma") {
          scene.video_prompt = enforceStoryboardVideoFacialRequirements(scene.video_prompt, scene);
        }
      });
      const data = await postJson("/vrgdg/storyboard/save", {
        project_folder: state.projectFolder,
        storyboard: slimStoryboardForRequest(state),
      });
      // The Storyboard is already saved. Do not trigger a redundant parent
      // Video Builder session save from this completed save action.
      syncStoryLayerFromInputs({ notify: false });
      if (focusedSection && payload.onFocusedSave) await payload.onFocusedSave({
        ...storyboardDefaultsPayload(),
        story_layer: normalizeStoryLayer(state.storyLayer),
        script_import: normalizeStoryboardScriptImportState(state.scriptImport),
        facial_performance_default: state.facialPerformance || "",
        facial_performance_custom_default: state.facialPerformanceCustom || "",
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
      });
      createToast(`Storyboard saved:\n${data.storyboard?.path || ""}`);
    } catch (error) {
      if (throwOnError) throw error;
      createToast(String(error?.message || error), true);
    } finally {
      save.disabled = false;
      state.saving = false;
    }
  }

  async function exportPromptFiles() {
    if (!state.projectFolder) {
      createToast("Save the AI Video Builder project first so Storyboard Builder knows where to export prompt files.", true);
      return;
    }
    exportPrompts.disabled = true;
    try {
      state.scenes.forEach((scene) => {
        if (String(scene.image_prompt || "").trim()) scene.image_prompt = ensureStoryboardReferenceOpening(scene.image_prompt, scene, state.imageMode);
        if (state.projectVideoEngine !== "minimax_h3" && String(scene.video_prompt || "").trim() && normalizeVideoPromptOrigin(scene.video_prompt_origin) === "gemma") {
          scene.video_prompt = enforceStoryboardVideoFacialRequirements(scene.video_prompt, scene);
        }
      });
      const data = await postJson("/vrgdg/storyboard/export_prompts", {
        project_folder: state.projectFolder,
        storyboard: slimStoryboardForRequest(state),
      });
      if (state.onPromptsExported) {
        state.onPromptsExported({
          ...storyboardDefaultsPayload(),
          story_layer: normalizeStoryLayer(state.storyLayer),
          scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
        });
      }
      const destination = state.onPromptsExported
        ? " and copied them into matching Video Builder timeline segments. Timeline segments were not created or replaced."
        : " to files only. The Video Builder timeline was not created or replaced.";
      createToast(`Exported ${data.scene_count || 0} scene prompt rows${destination}\n\nText:\n${data.t2i_prompts_path}\n${data.i2v_prompts_path}\nJSON:\n${data.t2i_prompts_json_path || ""}\n${data.video_prompts_json_path || ""}`);
    } catch (error) {
      createToast(String(error?.message || error), true);
    } finally {
      exportPrompts.disabled = false;
    }
  }

  return { copyStoryboardForGpt, exportPromptFiles, loadExisting, saveStoryboard };
}
