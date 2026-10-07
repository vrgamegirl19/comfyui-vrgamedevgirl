import {
  BROWSER_IMAGE_PROVIDERS,
  getBrowserImageStatus,
  openBrowserImageLogin,
  setupBrowserImageAutomation,
} from "../VRGDG_BrowserImageBridge.js";
import { openMusicVideoWizard } from "./wizard.mjs";
import { storyboardGptPayload } from "../storyboard_builder/gpt_payload.mjs";
import {
  FACIAL_PERFORMANCE_PRESETS,
  PERFORMANCE_STYLE_PRESETS,
  storyboardFacialPerformancePreset,
  storyboardPerformancePreset,
} from "../storyboard_builder/performance_presets.mjs";
import {
  STORYBOARD_CAMERA_FLOW_PRESETS,
  STORYBOARD_IMAGE_AESTHETIC_PRESETS,
  STORYBOARD_IMAGE_SHOT_FLOW_PRESETS,
  storyboardCameraFlowEntry,
  storyboardImageAestheticPreset,
  storyboardImageShotFlowEntry,
} from "../storyboard_builder/shot_presets.mjs";
import { openWizardBeta, wizardBetaNeeds } from "../VRGDG_WizardBeta.js";
import { formatBrowserImageStatus } from "./browser_ai.mjs";
import { GEMMA_VIDEO_PROMPT_TIMEOUT_MS, makeEditorImageUrl, postJson } from "./comfy_api.mjs";
import {
  DEFAULT_I2V_DIFFUSION_MODEL,
  DEFAULT_I2V_UNET,
  NB_IMAGE_MODELS,
  REQUIRED_LTX_MSR_LORA,
  REQUIRED_LTX25_MSR_LORA,
} from "./constants.mjs";
import {
  makeButton,
  makeSubTabs,
  normalizeProjectVideoEngine,
  normalizeVideoType,
  toast,
  VIDEO_TYPE_OPTIONS,
} from "./controls.mjs";
import { gemmaBatchFailureStore, recordGemmaBatchFailure, showGemmaBatchFailures } from "./dialogs.mjs";
import { applyLyricSectionsFromReferenceText } from "./lyric_transcription.mjs";
import { cloneMiniMaxH3Settings, MINIMAX_H3_AUDIO_MODE_OPTIONS, MINIMAX_H3_MODE_OPTIONS } from "./minimax_h3.mjs";
import {
  browserImageLoginStatus,
  browserImageProviderDebugPort,
  browserImageProviderLabel,
  builderMotionSpeedGuidance,
  cloneErnieImageSettings,
  cloneFlowGptBrowserSettings,
  cloneI2VVideoSettings,
  cloneKrea2TwoPassSettings,
  cloneNBImageSettings,
  cloneZImageSettings,
  defaultErnieImageSettings,
  defaultFlowGptBrowserSettings,
  defaultFluxKleinSettings,
  defaultKrea2TwoPassSettings,
  defaultZImageSettings,
  normalizeBuilderStoryboardDefaults,
  normalizeBuilderStoryLayer,
  normalizeFlowGptBrowserProvider,
  repairI2VVideoSettingDimensions,
} from "./model_settings.mjs";
import { chooseBatchModeAction } from "./project_actions.mjs";
import {
  flattenLyricForPrompt,
  isRecoverableBuildGemmaError,
  normalizeGemmaContextLimit,
  normalizeGemmaGpuLayers,
} from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder, normalizeLyricMapper } from "./reference_data.mjs";



export function createWizardBridge({
  activeI2VVideoSettings, activeProjectFolderForSave, activeSegment, allEditableSegments, audioInput,
  autoSaveSessionQuiet, chooseProjectAudioFile, confirmAndRunFullBuild, confirmAndRunGemmaT2IAll,
  confirmAndRunGemmaVideoAll, confirmAndRunZImageAll, createProgressWindow, createSceneVideoActions,
  createSceneVideoButtons, createScenesFromTimestampedLyrics, createSilentTimelineAudioForDuration,
  currentVideoMode, ensureAllSegmentRuntimeFields, ensureAutoTimedSingerCuesBeforePrompt, ernieClipPicker,
  ernieImagePanel, ernieUnetPicker, ernieVaePicker, exportManualFlowGptRefs, finalizeVideoPromptDraftOnly,
  flowGptImageAllScenes, flowGptManualAutoAdvance, flowGptManualMode, flowGptManualStatus, flowGptModePanel,
  flowGptStatusText, fluxClipPicker, fluxKleinPanel, fluxUnetPicker, fluxVaePicker, gemmaModelSelect,
  gemmaRunnerLabel, gemmaRunnerLine, i2vAudioVaePicker, i2vClip1Picker, i2vClip2Picker,
  i2vDiffusionModelPicker, i2vFpsInput, i2vGemmaModelSelect, i2vHeightInput, i2vLoraCount, i2vLoraSlots,
  i2vMmprojSelect, i2vSeedInput, i2vTextGemmaModelSelect, i2vUnetPicker, i2vUpscalePicker, i2vUseGgufModel,
  i2vUseLora, i2vVaePicker, i2vWidthInput, imageModeDisplayLabel, importLatestManualFlowGptDownload,
  importTimelineImagesFromFolder, inspector, krea2TwoPassClipPicker, krea2TwoPassPanel,
  krea2TwoPassUnetPicker, krea2TwoPassVaePicker, ltxMsrFirstPassStrength, ltxMsrLoraPicker,
  miniMaxGemmaModelSelect, miniMaxMmprojSelect, miniMaxModePanels, miniMaxPassChooser, miniMaxSubTabs,
  miniMaxTextGemmaModelSelect, mmprojSelect, nbImagePanel, newProject, normalizeLyricCueMapForSegment,
  openBulkSegmentsModal, openFluxReferenceBuilderModal, openGemmaRunnerModal,
  openIdLoraReferenceBuilderModalSafely, openIngredientsReferenceBuilderModal, openLyricMappingWorkflowModal,
  openLyricReviewModal, openManualFlowGptBrowser, openStoryboardBuilderFromProject, projectInput,
  promptRunnerActionName, pushHistory, render, renderAllScenes, rtvSceneImageAnchorSection,
  saveErnieImageSettingsFromPanel, saveFlowGptBrowserSettingsFromPanel, saveFluxKleinSettingsFromPanel,
  saveGemmaJunkDebug, saveI2VVideoSettingsFromPanel, saveKrea2TwoPassSettingsFromPanel,
  saveMiniMaxH3SettingsFromPanel, saveNBImageSettingsFromPanel, saveSession, saveZImageSettingsFromPanel,
  sceneDisplayName, segmentTrack, selectedPerformerSubjectsForSegment, setInspectorTab,
  setSegmentPromptForEdit, state, storyboardPipeline, storyboardReferenceBuilderWithIdLoraRefs,
  storyboardScenePayload, subjectSceneInput, syncErnieImagePanel, syncFlowGptBrowserPanel,
  syncFlowGptManualPanel, syncFluxKleinPanel, syncI2VVideoModelPickerVisibility, syncI2VVideoSettingsPanel,
  syncInspector, syncKrea2TwoPassLlmSelectsFromShared, syncKrea2TwoPassPanel, syncLyricAndSubjectNoteFiles,
  syncMiniMaxH3Panel, syncNBImagePanel, syncProjectVideoEngineUI, syncVideoModePanel, syncVideoTypeControl,
  syncZImageSettingsPanel, t2iTextGemmaModelSelect, textGemmaRunnerPayload, transcribeLyricsForTimeline,
  updateActiveFromInputs, useSceneI2VVideoSettings, useSceneI2VVideoSettingsNote, useSceneMiniMaxH3Settings,
  useSceneMiniMaxH3SettingsNote, videoModeDisplayLabel, videoSubTabs, wizardStoryboardState,
  wizardVideoSettings, zClipPicker, zUnetPicker, zVaePicker, zimageSettingsPanel,
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

  function openWizardFromBuilder() {
    const setWizardVideoMode = (mode) => {
      const normalized = String(mode || "").trim().toLowerCase();
      const allowed = ["i2v", "id_lora", "rtv", "t2v", "ingredients"];
      if (!allowed.includes(normalized)) return;
      pushHistory();
      state.videoModelMode = normalized;
      syncVideoModePanel();
      syncI2VVideoSettingsPanel();
      autoSaveSessionQuiet("wizard video mode").catch(() => null);
    };
    const normalizeWizardImageMode = (mode) => {
      const normalized = String(mode || "").trim().toLowerCase();
      return ["zimage", "flux_klein", "nano_banana", "ernie_image", "krea2_2pass", "flow_gpt"].includes(normalized)
        ? normalized
        : "zimage";
    };
    const setWizardImageMode = async (mode) => {
      const normalized = normalizeWizardImageMode(mode);
      pushHistory();
      state.imageModelMode = normalized;
      state.fluxKleinSettings.image_model_mode = normalized;
      state.fluxKleinSettings.enabled = normalized === "flux_klein";
      syncFluxKleinPanel();
      syncZImageSettingsPanel();
      syncErnieImagePanel();
      syncKrea2TwoPassPanel();
      syncInspector();
      render();
      await autoSaveSessionQuiet("wizard image mode");
      return normalized;
    };
    const compactWizardText = (value, limit = 700) => {
      const text = String(value || "").replace(/\s+/g, " ").trim();
      return text.length > limit ? `${text.slice(0, Math.max(0, limit - 3)).trim()}...` : text;
    };
    const wizardLocationKey = (value) => String(value || "").trim().toLowerCase().replace(/\s+/g, " ");
    const wizardLocationByName = (refs, name) => {
      const key = wizardLocationKey(name);
      return (refs.locations || []).find((location) => wizardLocationKey(location.name) === key) || null;
    };
    const wizardOptionsFromSelect = (select) => Array.from(select?.options || [])
      .map((option) => String(option?.value || option?.textContent || "").trim())
      .filter(Boolean);
    const wizardOptionsFromPicker = (picker) => Array.isArray(picker?.options)
      ? picker.options.map((item) => String(item || "").trim()).filter(Boolean)
      : [];
    const applyWizardSettings = async (settings = {}) => {
      if (settings.image_model_mode) {
        const imageMode = normalizeWizardImageMode(settings.image_model_mode);
        state.imageModelMode = imageMode;
        state.fluxKleinSettings.image_model_mode = imageMode;
        state.fluxKleinSettings.enabled = imageMode === "flux_klein";
        const imageSettings = settings.image_settings || {};
        if (imageSettings && typeof imageSettings === "object") {
          if (imageMode === "flux_klein") {
            state.fluxKleinSettings = {
              ...state.fluxKleinSettings,
              ...imageSettings,
              image_model_mode: imageMode,
              enabled: true,
            };
          } else if (imageMode === "ernie_image") {
            state.ernieImageSettings = cloneErnieImageSettings({
              ...state.ernieImageSettings,
              ...imageSettings,
            });
          } else if (imageMode === "krea2_2pass") {
            state.krea2TwoPassSettings = cloneKrea2TwoPassSettings({
              ...state.krea2TwoPassSettings,
              ...imageSettings,
            });
          } else if (imageMode === "nano_banana") {
            state.nbImageSettings = cloneNBImageSettings({
              ...state.nbImageSettings,
              ...imageSettings,
            });
          } else if (imageMode === "flow_gpt") {
            state.flowGptBrowserSettings = cloneFlowGptBrowserSettings({
              ...state.flowGptBrowserSettings,
              ...imageSettings,
            });
            flowGptManualMode.input.checked = Boolean(imageSettings.manual_mode);
            flowGptManualAutoAdvance.input.checked = Boolean(imageSettings.manual_auto_advance);
          } else if (imageMode === "zimage") {
            state.zimageSettings = cloneZImageSettings({
              ...state.zimageSettings,
              ...imageSettings,
            });
          }
        }
      }
      if (settings.use_gguf_model != null) i2vUseGgufModel.input.checked = settings.use_gguf_model !== false;
      i2vUnetPicker.input.value = String(settings.unet_name || i2vUnetPicker.input.value || "");
      i2vDiffusionModelPicker.input.value = String(settings.diffusion_model_name || i2vDiffusionModelPicker.input.value || DEFAULT_I2V_DIFFUSION_MODEL);
      i2vVaePicker.input.value = String(settings.vae_name || i2vVaePicker.input.value || "");
      i2vClip1Picker.input.value = String(settings.clip_name1 || i2vClip1Picker.input.value || "");
      i2vClip2Picker.input.value = String(settings.clip_name2 || i2vClip2Picker.input.value || "");
      i2vUpscalePicker.input.value = String(settings.upscale_model_name || i2vUpscalePicker.input.value || "");
      i2vAudioVaePicker.input.value = String(settings.audio_vae_name || i2vAudioVaePicker.input.value || "");
      i2vFpsInput.value = Number(settings.fps || i2vFpsInput.value || 24);
      i2vWidthInput.value = Number(settings.width || i2vWidthInput.value || 1920);
      i2vHeightInput.value = Number(settings.height || i2vHeightInput.value || 1080);
      i2vSeedInput.value = Number(settings.seed || i2vSeedInput.value || 69);
      if (settings.msr_lora_name != null) ltxMsrLoraPicker.input.value = String(settings.msr_lora_name || ltxMsrLoraPicker.input.value || REQUIRED_LTX_MSR_LORA);
      if (settings.msr_first_pass_strength != null) ltxMsrFirstPassStrength.value = Number(settings.msr_first_pass_strength || ltxMsrFirstPassStrength.value || 1);
      i2vUseLora.input.checked = Boolean(settings.use_loras);
      i2vLoraCount.value = Math.max(0, Math.min(4, Number(settings.lora_count || 0)));
      const incomingLoras = Array.isArray(settings.loras) ? settings.loras : [];
      i2vLoraSlots.forEach((slot, index) => {
        const lora = incomingLoras[index] || {};
        slot.picker.input.value = String(lora.name || slot.picker.input.value || "[none]");
        slot.firstPassStrength.value = Number(lora.first_pass_strength ?? lora.strength ?? slot.firstPassStrength.value ?? 1);
        slot.secondPassStrength.value = 0;
      });
      if (String(settings.text_gemma_model || "").trim()) {
        t2iTextGemmaModelSelect.value = settings.text_gemma_model;
        i2vTextGemmaModelSelect.value = settings.text_gemma_model;
        miniMaxTextGemmaModelSelect.value = settings.text_gemma_model;
      }
      if (String(settings.vision_gemma_model || "").trim()) {
        gemmaModelSelect.value = settings.vision_gemma_model;
        i2vGemmaModelSelect.value = settings.vision_gemma_model;
        miniMaxGemmaModelSelect.value = settings.vision_gemma_model;
      }
      if (String(settings.mmproj_file || "").trim()) {
        mmprojSelect.value = settings.mmproj_file;
        i2vMmprojSelect.value = settings.mmproj_file;
        miniMaxMmprojSelect.value = settings.mmproj_file;
      }
      syncKrea2TwoPassLlmSelectsFromShared();
      syncI2VVideoModelPickerVisibility();
      saveI2VVideoSettingsFromPanel();
      syncFluxKleinPanel();
      syncZImageSettingsPanel();
      syncErnieImagePanel();
      syncKrea2TwoPassPanel();
      syncFlowGptBrowserPanel();
      syncVideoModePanel();
      syncI2VVideoSettingsPanel();
      render();
      await autoSaveSessionQuiet("wizard settings applied");
      toast("Wizard settings applied.");
      return true;
    };
    const upsertWizardLocations = (refs, locations = []) => {
      refs.locations = Array.isArray(refs.locations) ? refs.locations : [];
      let added = 0;
      let updated = 0;
      for (const item of locations) {
        const name = String(item?.name || "").trim();
        if (!name) continue;
        const description = String(item?.description || "").trim();
        const existing = wizardLocationByName(refs, name);
        if (existing) {
          if (description && description !== existing.description) {
            existing.description = description;
            updated += 1;
          }
        } else {
          refs.locations_cleared = false;
          refs.locations.push({
            id: `loc_wizard_${Date.now()}_${refs.locations.length}_${Math.floor(Math.random() * 10000)}`,
            name,
            description,
            image: { path: "", data: "", name: "" },
          });
          added += 1;
        }
      }
      return { added, updated };
    };
    const wizardLyricsFromScenes = () => allEditableSegments()
      .map((segment, index) => {
        const lyric = String(segment?.lyric_text || "").trim();
        if (!lyric) return "";
        return `Scene ${index + 1}: ${lyric}`;
      })
      .filter(Boolean)
      .join("\n");
    const wizardSubjectContextForLocations = (refs) => (refs.subjects || [])
      .map((subject) => {
        const type = String(subject.reference_type || "character").trim();
        const name = String(subject.name || "").trim();
        const description = String(subject.description || "").trim();
        if (!name && !description) return "";
        return `${name || "Reference"}${type ? ` (${type})` : ""}${description ? `: ${description}` : ""}`;
      })
      .filter(Boolean)
      .join("\n");
    const createWizardLocationsFromLyrics = async (options = {}) => {
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      const lyricsText = String(options.lyrics || "").trim() || wizardLyricsFromScenes();
      if (!lyricsText) {
        toast("Paste lyrics or create lyric scenes before creating locations from lyrics.", true);
        return { added: 0, updated: 0 };
      }
      const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
      if (!modelFile && !["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner)) {
        toast("Choose a non-vision Gemma model first, or use LM Studio, LLM API, or your own server in LLM Runner.", true);
        return { added: 0, updated: 0 };
      }
      const progress = createProgressWindow("Wizard Location Scout", { zIndex: 100012 });
      try {
        progress.set(`Sending lyrics to Gemma location scout...\n${gemmaRunnerLine()}`, 15);
        const data = await postJson("/vrgdg/music_builder/wizard_locations_from_lyrics", {
          ...textGemmaRunnerPayload(),
          model_file: modelFile,
          lyrics_text: lyricsText,
          style_theme: options.styleTheme || options.style_theme || refs.location_style_theme || "",
          subject_context: wizardSubjectContextForLocations(refs),
          existing_locations: refs.locations.map((item) => ({
            name: compactWizardText(item.name, 90),
            description: compactWizardText(item.description, 700),
          })),
          max_locations: refs.max_generated_locations || 8,
          unload_after: true,
          n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
          max_new_tokens: 2200,
        }, 10 * 60 * 1000);
        const { added, updated } = upsertWizardLocations(refs, data.locations || []);
        refs.use_location_references = Boolean(refs.locations.length || Object.keys(refs.scene_map || {}).length);
        refs.cleared = false;
        state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
        syncInspector();
        render();
        await syncLyricAndSubjectNoteFiles("wizard locations from lyrics");
        await autoSaveSessionQuiet("wizard locations from lyrics");
        progress.set(`Location list ready.\nAdded: ${added}\nUpdated: ${updated}`, 100);
        progress.close(1800);
        toast(`Created ${added} new location${added === 1 ? "" : "s"} from lyrics.`);
        return { added, updated };
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
        return { added: 0, updated: 0, error: String(error?.message || error) };
      }
    };
    const autoMapWizardLocations = async () => {
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      if (!refs.locations.length) {
        toast("Add or import locations in Reference Builder before using wizard Auto Map.", true);
        return { mapped: 0 };
      }
      const scenes = allEditableSegments().map((segment, index) => ({
        id: segment.id,
        label: segment.label || `Scene ${index + 1}`,
        concept: "",
        notes: compactWizardText(segment.lyric_text || "", 850),
      })).filter((scene) => String(scene.concept || scene.notes || "").trim());
      if (!scenes.length) {
        toast("Create timeline scenes with lyric text before using wizard location mapping.", true);
        return { mapped: 0 };
      }
      const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
      if (!modelFile && !["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner)) {
        toast("Choose a non-vision Gemma model first, or use LM Studio, LLM API, or your own server in LLM Runner.", true);
        return { mapped: 0 };
      }
      const progress = createProgressWindow("Wizard Location Mapping", { zIndex: 100012 });
      try {
        progress.set(`Sending scene lyric lines and existing locations to Gemma...\n${gemmaRunnerLine()}`, 15);
        const data = await postJson("/vrgdg/music_builder/flux_reference_location_map", {
          ...textGemmaRunnerPayload(),
          model_file: modelFile,
          scenes,
          subject_scene_text: compactWizardText(subjectSceneInput.value || "", 1600),
          existing_locations: refs.locations.map((item) => ({
            name: compactWizardText(item.name, 90),
            description: compactWizardText(item.description, 700),
          })),
          unload_after: true,
          n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
          max_new_tokens: 1800,
        }, 10 * 60 * 1000);
        upsertWizardLocations(refs, data.locations || []);
        refs.scene_map = refs.scene_map && typeof refs.scene_map === "object" ? refs.scene_map : {};
        let mapped = 0;
        for (const [sceneId, locationName] of Object.entries(data.scene_map || {})) {
          const location = wizardLocationByName(refs, locationName);
          if (!location?.id) continue;
          refs.scene_map[sceneId] = location.id;
          mapped += 1;
        }
        refs.use_location_references = Boolean(refs.locations.length || Object.keys(refs.scene_map || {}).length);
        refs.cleared = false;
        state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
        syncInspector();
        render();
        await syncLyricAndSubjectNoteFiles("wizard location mapping");
        await autoSaveSessionQuiet("wizard location mapping");
        progress.set(`Location mapping complete.\nMapped scenes: ${mapped}`, 100);
        progress.close(1800);
        toast(`Wizard auto-mapped ${mapped} scene${mapped === 1 ? "" : "s"} to locations.`);
        return { mapped };
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
        return { mapped: 0, error: String(error?.message || error) };
      }
    };
    const applyWizardSceneDefaultSettingsToState = (settings = {}) => {
      const cameraSpeed = Math.max(0, Math.min(10, Number(settings.cameraMotionSpeed ?? settings.camera_motion_speed ?? state.builderStoryboardDefaults?.camera_motion_speed ?? 4)));
      const characterSpeed = Math.max(0, Math.min(10, Number(settings.characterMotionSpeed ?? settings.character_motion_speed ?? state.builderStoryboardDefaults?.character_motion_speed ?? 4)));
      state.builderStoryboardDefaults = normalizeBuilderStoryboardDefaults({
        ...state.builderStoryboardDefaults,
        camera_flow: settings.cameraFlow ?? settings.camera_flow ?? state.builderStoryboardDefaults?.camera_flow,
        image_shot_flow: settings.imageShotFlow ?? settings.image_shot_flow ?? state.builderStoryboardDefaults?.image_shot_flow,
        image_aesthetic: settings.imageAesthetic ?? settings.image_aesthetic ?? state.builderStoryboardDefaults?.image_aesthetic,
        global_consistency_phrase: settings.globalConsistencyPhrase ?? settings.global_consistency_phrase ?? state.builderStoryboardDefaults?.global_consistency_phrase,
        performance_style: settings.performanceStyle ?? settings.performance_style ?? state.builderStoryboardDefaults?.performance_style,
        camera_motion_speed: cameraSpeed,
        character_motion_speed: characterSpeed,
        camera_guidance: builderMotionSpeedGuidance(cameraSpeed, "camera"),
        character_guidance: builderMotionSpeedGuidance(characterSpeed, "character"),
      });
      if (Object.prototype.hasOwnProperty.call(settings, "facialPerformance") || Object.prototype.hasOwnProperty.call(settings, "facial_performance")) {
        state.defaultFacialPerformance = String(settings.facialPerformance ?? settings.facial_performance ?? "");
      }
      if (Object.prototype.hasOwnProperty.call(settings, "facialPerformanceCustom") || Object.prototype.hasOwnProperty.call(settings, "facial_performance_custom")) {
        state.defaultFacialPerformanceCustom = String(settings.facialPerformanceCustom ?? settings.facial_performance_custom ?? "");
      }
      if (settings.storyLayer || settings.story_layer) {
        state.builderStoryLayer = normalizeBuilderStoryLayer(settings.storyLayer || settings.story_layer);
      }
      return {
        sceneDefaults: normalizeBuilderStoryboardDefaults(state.builderStoryboardDefaults),
        storyLayer: normalizeBuilderStoryLayer(state.builderStoryLayer),
      };
    };
    const updateWizardSceneDefaultSettings = async (settings = {}, label = "wizard scene settings") => {
      const updated = applyWizardSceneDefaultSettingsToState(settings);
      await autoSaveSessionQuiet(label);
      return updated;
    };
    const applyWizardSceneDefaults = async (settings = {}) => {
      const cameraFlow = STORYBOARD_CAMERA_FLOW_PRESETS[settings.cameraFlow] ? settings.cameraFlow : "balanced";
      const imageShotFlow = STORYBOARD_IMAGE_SHOT_FLOW_PRESETS[settings.imageShotFlow] ? settings.imageShotFlow : "intimate";
      const imageAesthetic = STORYBOARD_IMAGE_AESTHETIC_PRESETS.some((preset) => preset.value === settings.imageAesthetic)
        ? String(settings.imageAesthetic || "")
        : "";
      const performanceStyle = String(settings.performanceStyle || "");
      const facialPerformance = String(settings.facialPerformance || "");
      const facialPerformanceCustom = String(settings.facialPerformanceCustom || "");
      const shouldApplyImageShot = Boolean(settings.applyImageShotFlow);
      const shouldApplyImageAesthetic = Boolean(settings.applyImageAesthetic);
      const shouldApplyPerformance = Boolean(settings.applyPerformance);
      const shouldApplyFacial = Boolean(settings.applyFacialPerformance);
      const shouldApplyCamera = settings.applyCamera == null
        ? !shouldApplyImageShot && !shouldApplyImageAesthetic && !shouldApplyPerformance && !settings.applyFacialPerformance
        : Boolean(settings.applyCamera);
      const overwriteImageShot = Boolean(settings.overwriteImageShotFlow);
      const overwriteImageAesthetic = Boolean(settings.overwriteImageAesthetic);
      const overwriteCamera = Boolean(settings.overwriteCamera);
      const overwritePerformance = Boolean(settings.overwritePerformance);
      const overwriteFacial = Boolean(settings.overwriteFacialPerformance);
      let previousMotion = "";
      let imageShotChanged = 0;
      let imageAestheticChanged = 0;
      let cameraChanged = 0;
      let performanceChanged = 0;
      let facialChanged = 0;
      const scenes = allEditableSegments();
      pushHistory();
      applyWizardSceneDefaultSettingsToState(settings);
      const nextStoryboardDefaults = {
        ...state.builderStoryboardDefaults,
      };
      if (shouldApplyCamera) nextStoryboardDefaults.camera_flow = cameraFlow;
      if (shouldApplyImageShot) nextStoryboardDefaults.image_shot_flow = imageShotFlow;
      if (shouldApplyImageAesthetic) nextStoryboardDefaults.image_aesthetic = imageAesthetic;
      if (shouldApplyPerformance && performanceStyle) nextStoryboardDefaults.performance_style = performanceStyle;
      state.builderStoryboardDefaults = normalizeBuilderStoryboardDefaults(nextStoryboardDefaults);
      if (facialPerformance || facialPerformanceCustom) {
        state.defaultFacialPerformance = facialPerformance;
        state.defaultFacialPerformanceCustom = facialPerformanceCustom;
      }
      scenes.forEach((segment, index) => {
        if (shouldApplyImageShot && imageShotFlow !== "off") {
          const shot = storyboardImageShotFlowEntry(imageShotFlow, index);
          if (shot && (overwriteImageShot || !String(segment.shot_type || "").trim())) {
            segment.shot_type = shot;
            imageShotChanged += 1;
          }
        }
        if (shouldApplyImageAesthetic) {
          const aestheticDescription = String(storyboardImageAestheticPreset(imageAesthetic).description || "").trim();
          const existing = String(segment.motion_summary || "");
          const hasAesthetic = existing
            .split(/\r?\n/)
            .some((line) => line.trim().toLowerCase().startsWith("image aesthetic:"));
          if (aestheticDescription && (overwriteImageAesthetic || !hasAesthetic)) {
            const prefix = "Image aesthetic:";
            const lines = existing
              .replace(/\r\n/g, "\n")
              .split("\n")
              .filter((line) => !line.trim().toLowerCase().startsWith(prefix.toLowerCase()));
            lines.push(`${prefix} ${aestheticDescription}.`);
            segment.motion_summary = lines.map((line) => line.trim()).filter(Boolean).join("\n");
            imageAestheticChanged += 1;
          }
        }
      const entry = shouldApplyCamera
        ? storyboardCameraFlowEntry(cameraFlow, index, previousMotion, state.builderStoryboardDefaults?.custom_camera_flow_sequence)
        : null;
        if (entry && cameraFlow !== "off") {
          const hadShot = Boolean(String(segment.shot_type || "").trim());
          const hadCamera = Boolean(String(segment.camera_motion || segment.motion_preset || "").trim());
          if ((overwriteCamera || !hadShot) && entry.shot) {
            segment.shot_type = entry.shot;
            cameraChanged += 1;
          }
          if ((overwriteCamera || !hadCamera) && entry.camera) {
            segment.camera_motion = entry.camera;
            cameraChanged += 1;
          }
          previousMotion = String(segment.camera_motion || entry.camera || previousMotion);
        }
        if (shouldApplyPerformance) {
          const hadPerformance = Boolean(String(segment.performance_style || "").trim());
          if (overwritePerformance || !hadPerformance) {
            segment.performance_style = performanceStyle;
            performanceChanged += 1;
          }
        }
        if (shouldApplyFacial) {
          const hadFacial = Boolean(String(segment.facial_performance || "").trim() || String(segment.facial_performance_custom || "").trim());
          if (overwriteFacial || !hadFacial) {
            segment.facial_performance = facialPerformance;
            segment.facial_performance_custom = facialPerformanceCustom;
            facialChanged += 1;
          }
        }
      });
      syncInspector();
      render();
      await autoSaveSessionQuiet("wizard scene defaults");
      toast(`Wizard scene defaults applied.\nImage shots: ${imageShotChanged}\nImage aesthetics: ${imageAestheticChanged}\nCamera fields: ${cameraChanged}\nPerformance scenes: ${performanceChanged}\nFacial scenes: ${facialChanged}`);
      return { imageShotChanged, imageAestheticChanged, cameraChanged, performanceChanged, facialChanged };
    };
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
    const createWizardStoryBrief = async (draft = {}) => {
      const layer = normalizeBuilderStoryLayer({
        ...state.builderStoryLayer,
        ...(draft?.storyLayer || draft?.story_layer || {}),
        user_story_arc: draft?.userStoryArc ?? draft?.user_story_arc ?? state.builderStoryLayer?.user_story_arc,
      });
      state.builderStoryLayer = layer;
      const scenes = storyboardScenePayload();
      const progress = createProgressWindow("Wizard Story Brief", { zIndex: 100012 });
      try {
        progress.set("Creating compact story brief from lyrics, sections, and story arc...", 18);
        const data = await postJson("/vrgdg/storyboard/story_brief", {
          ...textGemmaRunnerPayload(),
          model_file: i2vTextGemmaModelSelect.value || t2iTextGemmaModelSelect.value || "",
          story_layer: layer,
          lyrics: scenes.map((scene) => `${scene.lyric_section ? `[${scene.lyric_section}]\n` : ""}${scene.lyrics || ""}`).filter(Boolean).join("\n\n"),
          scenes,
          unload_after: true,
          max_new_tokens: 800,
        }, 240000);
        state.builderStoryLayer = normalizeBuilderStoryLayer({
          ...layer,
          song_story_brief: data.story_brief || "",
        });
        await autoSaveSessionQuiet("wizard story brief");
        progress.set("Story brief saved.", 100);
        progress.close(1400);
        toast("Wizard story brief created.");
        return normalizeBuilderStoryLayer(state.builderStoryLayer);
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
        return normalizeBuilderStoryLayer(state.builderStoryLayer);
      }
    };
    const createWizardStoryArc = async (draft = {}) => {
      const layer = normalizeBuilderStoryLayer({
        ...state.builderStoryLayer,
        ...(draft?.storyLayer || draft?.story_layer || {}),
        user_story_arc: draft?.userStoryArc ?? draft?.user_story_arc ?? state.builderStoryLayer?.user_story_arc,
      });
      state.builderStoryLayer = layer;
      const scenes = storyboardScenePayload();
      const progress = createProgressWindow(`Wizard Story Arc — ${gemmaRunnerLabel()}`, { zIndex: 100012 });
      try {
        const storyArcModeLabel = videoModeDisplayLabel(currentVideoMode(), true);
        progress.set(`Creating a short song-structure story arc from lyrics, subjects, and locations...\nVideo mode: ${storyArcModeLabel}`, 18);
        const data = await postJson("/vrgdg/storyboard/story_arc", {
          ...textGemmaRunnerPayload(),
          model_file: i2vTextGemmaModelSelect.value || t2iTextGemmaModelSelect.value || "",
          story_layer: layer,
          story_idea: draft?.storyIdea ?? draft?.story_idea ?? layer.user_story_arc,
          character_motion: draft?.characterMotion ?? draft?.character_motion ?? draft?.storyCharacterMotion ?? 7,
          lyrics: scenes.map((scene) => `${scene.lyric_section ? `[${scene.lyric_section}]\n` : ""}${scene.lyrics || ""}`).filter(Boolean).join("\n\n"),
          scenes,
          reference_builder: storyboardReferenceBuilderWithIdLoraRefs(state.fluxReferenceBuilder),
          unload_after: true,
          max_new_tokens: 900,
        }, 240000);
        state.builderStoryLayer = normalizeBuilderStoryLayer({
          ...layer,
          user_story_arc: data.story_arc || "",
        });
        await autoSaveSessionQuiet("wizard story arc");
        progress.set("Story arc saved.", 100);
        progress.close(1400);
        toast("Wizard story arc created.");
        return normalizeBuilderStoryLayer(state.builderStoryLayer);
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
        return normalizeBuilderStoryLayer(state.builderStoryLayer);
      }
    };
    const detectWizardLyricSections = async (referenceLyrics = "") => {
      const applied = applyLyricSectionsFromReferenceText(allEditableSegments(), referenceLyrics);
      if (applied) {
        syncInspector();
        render();
        await autoSaveSessionQuiet("wizard lyric sections detected");
      }
      toast(applied ? `Detected lyric sections for ${applied} scene${applied === 1 ? "" : "s"}.` : "No missing lyric sections were detected.");
      return applied;
    };
    const createWizardSceneBeats = async ({ overwrite = false, failedIds = [] } = {}) => {
      const segments = allEditableSegments()
        .slice()
        .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
      const scenes = storyboardScenePayload();
      const retryIds = new Set(failedIds.map((value) => String(value)));
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
      const failures = [];
      try {
        progress.set(`${retryIds.size ? "Retrying failed" : "Creating"} ${targets.length} scene story beat${targets.length === 1 ? "" : "s"}...`, 5);
        for (let index = 0; index < targets.length; index += 1) {
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
              model_file: i2vTextGemmaModelSelect.value || t2iTextGemmaModelSelect.value || "",
              story_layer: normalizeBuilderStoryLayer(state.builderStoryLayer),
              storyboard_payload: beatStoryboardPayload,
              all_subjects: (Array.isArray(state.fluxReferenceBuilder?.subjects) ? state.fluxReferenceBuilder.subjects : [])
                .map((subject) => ({ name: String(subject?.name || ""), description: String(subject?.description || "") })),
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
        toast(String(error?.message || error), true);
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
      const failedIds = new Set((options.failedIds || []).map((value) => String(value)));
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
          n_threads: 8,
          unload_after: true,
        },
      };
      const runnerName = promptRunnerActionName();
      const runnerGenericName = runnerName;
      const progress = createProgressWindow(`Storyboard ${runnerName} All (${miniMaxProject ? "MiniMax H3 project/locked modes" : videoModeDisplayLabel(videoMode, true)})`, { zIndex: 100012 });
      let created = 0;
      const failurePrefix = `storyboard:${videoMode}`;
      try {
        progress.set(`${failedIds.size ? "Retrying failed Storyboard scenes" : `Starting Storyboard ${runnerName} All`}...\nScenes: ${targetScenes.length}\nUsing the same prompt writer as Storyboard Builder.`, 5);
        for (let index = 0; index < targetScenes.length; index += 1) {
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
        toast(`Storyboard ${runnerName} All stopped after ${created}/${targetScenes.length} scenes:\n${String(error?.message || error)}`, true);
      }
    };
    const wizardSnapshot = () => {
      const refs = storyboardReferenceBuilderWithIdLoraRefs(state.fluxReferenceBuilder);
      const videoSettings = repairI2VVideoSettingDimensions(activeI2VVideoSettings() || state.i2vVideoSettings || {});
      const llmOptions = Array.from(new Set([
        ...wizardOptionsFromSelect(t2iTextGemmaModelSelect),
        ...wizardOptionsFromSelect(i2vTextGemmaModelSelect),
        ...wizardOptionsFromSelect(gemmaModelSelect),
        ...wizardOptionsFromSelect(i2vGemmaModelSelect),
      ]));
      return {
        projectFolder: String(projectInput.value || state.projectFolder || "").trim(),
        projectVideoEngine: normalizeProjectVideoEngine(state.projectVideoEngine),
        audioPath: String(audioInput.value || state.audioPath || "").trim(),
        wizardFolder: String(projectInput.value || state.projectFolder || "").trim() ? `${String(projectInput.value || state.projectFolder || "").trim()}\\wizard` : "",
        sceneCount: allEditableSegments().length,
        videoMode: currentVideoMode(),
        videoModeLabel: videoModeDisplayLabel(currentVideoMode()),
        imageMode: state.imageModelMode || "zimage",
        imageModeLabel: imageModeDisplayLabel(state.imageModelMode || "zimage"),
        imageModeOptions: [
          { value: "zimage", label: imageModeDisplayLabel("zimage") },
          { value: "flux_klein", label: imageModeDisplayLabel("flux_klein") },
          { value: "ernie_image", label: imageModeDisplayLabel("ernie_image") },
          { value: "krea2_2pass", label: imageModeDisplayLabel("krea2_2pass") },
          { value: "flow_gpt", label: imageModeDisplayLabel("flow_gpt") },
          { value: "nano_banana", label: imageModeDisplayLabel("nano_banana") },
        ],
        subjectCount: Array.isArray(refs.subjects) ? refs.subjects.length : 0,
        locationCount: Array.isArray(refs.locations) ? refs.locations.length : 0,
        storyLayer: normalizeBuilderStoryLayer(state.builderStoryLayer),
        storyBeatCount: allEditableSegments().filter((segment) => String(segment.story_beat || "").trim()).length,
        lyricSectionCount: allEditableSegments().filter((segment) => String(segment.lyric_section || "").trim()).length,
        videoSettings: {
          fps: Number(videoSettings.fps || 24),
          width: Number(videoSettings.width || 1920),
          height: Number(videoSettings.height || 1080),
          seed: Number(videoSettings.seed || 69),
          use_gguf_model: videoSettings.use_gguf_model !== false,
          unet_name: String(videoSettings.unet_name || ""),
          diffusion_model_name: String(videoSettings.diffusion_model_name || ""),
          vae_name: String(videoSettings.vae_name || ""),
          clip_name1: String(videoSettings.clip_name1 || ""),
          clip_name2: String(videoSettings.clip_name2 || ""),
          upscale_model_name: String(videoSettings.upscale_model_name || ""),
          audio_vae_name: String(videoSettings.audio_vae_name || ""),
          msr_lora_name: String(videoSettings.msr_lora_name || REQUIRED_LTX_MSR_LORA),
          msr_first_pass_strength: Number(videoSettings.msr_first_pass_strength ?? 1),
          use_loras: Boolean(videoSettings.use_loras),
          lora_count: Number(videoSettings.lora_count || 0),
          loras: Array.isArray(videoSettings.loras) ? videoSettings.loras.map((lora) => ({
            name: String(lora?.name || "[none]"),
            first_pass_strength: Number(lora?.first_pass_strength ?? lora?.strength ?? 1),
            second_pass_strength: Number(lora?.second_pass_strength ?? 0),
          })) : [],
        },
        gemmaSettings: {
          text_model: String(i2vTextGemmaModelSelect.value || t2iTextGemmaModelSelect.value || ""),
          vision_model: String(i2vGemmaModelSelect.value || gemmaModelSelect.value || ""),
          mmproj: String(i2vMmprojSelect.value || mmprojSelect.value || ""),
        },
        imageSettings: {
          zimage: cloneZImageSettings(state.zimageSettings || defaultZImageSettings()),
          flux_klein: { ...(state.fluxKleinSettings || defaultFluxKleinSettings()) },
          ernie_image: cloneErnieImageSettings(state.ernieImageSettings || defaultErnieImageSettings()),
          krea2_2pass: cloneKrea2TwoPassSettings(state.krea2TwoPassSettings || defaultKrea2TwoPassSettings()),
          flow_gpt: {
            ...cloneFlowGptBrowserSettings(state.flowGptBrowserSettings || defaultFlowGptBrowserSettings()),
            manual_mode: Boolean(flowGptManualMode.input.checked),
            manual_auto_advance: Boolean(flowGptManualAutoAdvance.input.checked),
          },
          nano_banana: { ...(state.nbImageSettings || {}) },
        },
        imageModelOptions: {
          zimage: {
            unets: wizardOptionsFromPicker(zUnetPicker),
            clip: wizardOptionsFromPicker(zClipPicker),
            vae: wizardOptionsFromPicker(zVaePicker),
          },
          flux_klein: {
            unets: wizardOptionsFromPicker(fluxUnetPicker),
            clip: wizardOptionsFromPicker(fluxClipPicker),
            vae: wizardOptionsFromPicker(fluxVaePicker),
          },
          ernie_image: {
            unets: wizardOptionsFromPicker(ernieUnetPicker),
            clip: wizardOptionsFromPicker(ernieClipPicker),
            vae: wizardOptionsFromPicker(ernieVaePicker),
          },
          krea2_2pass: {
            unets: wizardOptionsFromPicker(krea2TwoPassUnetPicker),
            clip: wizardOptionsFromPicker(krea2TwoPassClipPicker),
            vae: wizardOptionsFromPicker(krea2TwoPassVaePicker),
          },
          nano_banana: {
            models: NB_IMAGE_MODELS,
          },
        },
        modelOptions: {
          unets: wizardOptionsFromPicker(i2vUnetPicker),
          diffusion_models: wizardOptionsFromPicker(i2vDiffusionModelPicker),
          vae: wizardOptionsFromPicker(i2vVaePicker),
          clip: Array.from(new Set([...wizardOptionsFromPicker(i2vClip1Picker), ...wizardOptionsFromPicker(i2vClip2Picker)])),
          upscale_models: wizardOptionsFromPicker(i2vUpscalePicker),
          loras: wizardOptionsFromPicker(ltxMsrLoraPicker),
          llm: llmOptions,
          mmproj: Array.from(new Set([...wizardOptionsFromSelect(i2vMmprojSelect), ...wizardOptionsFromSelect(mmprojSelect)])),
        },
        sceneDefaults: {
          cameraFlow: state.builderStoryboardDefaults?.camera_flow || "balanced",
          imageShotFlow: state.builderStoryboardDefaults?.image_shot_flow || "intimate",
          imageAesthetic: state.builderStoryboardDefaults?.image_aesthetic || "",
          videoStyle: state.builderStoryboardDefaults?.video_style || "",
          videoStyleCustom: state.builderStoryboardDefaults?.video_style_custom || "",
          temporalWorldEffect: state.builderStoryboardDefaults?.temporal_world_effect || "",
          temporalWorldEffectCustom: state.builderStoryboardDefaults?.temporal_world_effect_custom || "",
          temporalAllowBackgroundExtras: state.builderStoryboardDefaults?.temporal_allow_background_extras !== false,
          temporalBackgroundIntensity: state.builderStoryboardDefaults?.temporal_background_intensity ?? 8,
          temporalEnvironmentTimePassage: state.builderStoryboardDefaults?.temporal_environment_time_passage !== false,
          temporalProtectedCharacters: state.builderStoryboardDefaults?.temporal_protected_characters || "all_referenced",
          temporalProtectedCustom: state.builderStoryboardDefaults?.temporal_protected_custom || "",
          globalConsistencyPhrase: state.builderStoryboardDefaults?.global_consistency_phrase || "",
          performanceStyle: state.builderStoryboardDefaults?.performance_style || "",
          facialPerformance: state.defaultFacialPerformance || "",
          facialPerformanceCustom: state.defaultFacialPerformanceCustom || "",
          cameraMotionSpeed: state.builderStoryboardDefaults?.camera_motion_speed ?? 4,
          characterMotionSpeed: state.builderStoryboardDefaults?.character_motion_speed ?? 4,
          cameraFlowOptions: Object.entries(STORYBOARD_CAMERA_FLOW_PRESETS).map(([value, preset]) => ({
            value,
            label: preset.label || value,
            description: preset.description || "",
            count: Array.isArray(preset.sequence) ? preset.sequence.length : 0,
          })),
          imageShotFlowOptions: Object.entries(STORYBOARD_IMAGE_SHOT_FLOW_PRESETS).map(([value, preset]) => ({
            value,
            label: preset.label || value,
            description: preset.description || "",
            count: Array.isArray(preset.sequence) ? preset.sequence.length : 0,
          })),
          imageAestheticOptions: STORYBOARD_IMAGE_AESTHETIC_PRESETS.map((preset) => ({
            value: preset.value,
            label: preset.label,
            description: preset.description || "",
          })),
          performanceStyleOptions: PERFORMANCE_STYLE_PRESETS.map((preset) => ({
            value: preset.value,
            label: preset.label,
            description: storyboardPerformancePreset(preset.value).direction || preset.description || "",
          })),
          facialPerformanceOptions: FACIAL_PERFORMANCE_PRESETS.map((preset) => ({
            value: preset.value,
            label: preset.label,
            description: storyboardFacialPerformancePreset(preset.value).direction || preset.description || "",
          })),
        },
      };
    };
    openMusicVideoWizard({
      snapshot: wizardSnapshot,
      setVideoMode: setWizardVideoMode,
      setImageMode: setWizardImageMode,
      chooseAudioFile: chooseProjectAudioFile,
      createScenesFromLyrics: createScenesFromTimestampedLyrics,
      applySettings: applyWizardSettings,
      updateFlowGptSettings: async (settings = {}) => {
        state.flowGptBrowserSettings = cloneFlowGptBrowserSettings({
          ...state.flowGptBrowserSettings,
          ...settings,
        });
        if (settings.manual_mode != null) flowGptManualMode.input.checked = Boolean(settings.manual_mode);
        if (settings.manual_auto_advance != null) flowGptManualAutoAdvance.input.checked = Boolean(settings.manual_auto_advance);
        syncFlowGptBrowserPanel();
        await autoSaveSessionQuiet("wizard Flow/GPT settings updated");
        return cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
      },
      openGemmaRunner: openGemmaRunnerModal,
      setupFlowGptBrowser: async () => {
        const data = await setupBrowserImageAutomation({ install_portable_node: true, install_if_missing: true, strict_ssl: false });
        const status = data.status || formatBrowserImageStatus(data);
        flowGptStatusText.textContent = status;
        toast("Browser automation setup finished.");
        return status;
      },
      checkFlowGptBrowser: async () => {
        const data = await getBrowserImageStatus();
        const status = formatBrowserImageStatus(data);
        flowGptStatusText.textContent = status;
        return status;
      },
      openFlowGptLogin: async (provider = BROWSER_IMAGE_PROVIDERS.FLOW_NANO_BANANA) => {
        const normalized = normalizeFlowGptBrowserProvider(provider);
        const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
        await openBrowserImageLogin(normalized, {
          debug_port: browserImageProviderDebugPort(normalized),
          timeoutMs: 60000,
        });
        const status = browserImageLoginStatus(normalized, settings);
        flowGptStatusText.textContent = status;
        toast(`${browserImageProviderLabel(normalized)} login browser opened.`);
        return status;
      },
      openFlowGptManualBrowser: async () => {
        flowGptManualMode.input.checked = true;
        syncFlowGptManualPanel();
        await openManualFlowGptBrowser();
        return flowGptManualStatus.textContent || "Manual browser opened.";
      },
      exportFlowGptManualRefs: async () => {
        flowGptManualMode.input.checked = true;
        syncFlowGptManualPanel();
        await exportManualFlowGptRefs();
        return flowGptManualStatus.textContent || "Scene refs exported.";
      },
      importLatestFlowGptManualDownload: async () => {
        flowGptManualMode.input.checked = true;
        syncFlowGptManualPanel();
        await importLatestManualFlowGptDownload();
        return flowGptManualStatus.textContent || "Latest manual download imported.";
      },
      openReferenceBuilder: (mode = currentVideoMode()) => {
        const normalized = String(mode || currentVideoMode() || "i2v").trim().toLowerCase();
        setWizardVideoMode(normalized);
        if (normalized === "ingredients") {
          openIngredientsReferenceBuilderModal();
        } else if (normalized === "rtv") {
          openFluxReferenceBuilderModal({ wizardMode: true });
        } else if (normalized === "id_lora") {
          openIdLoraReferenceBuilderModalSafely();
        } else {
          openFluxReferenceBuilderModal({ wizardMode: true, textOnlyMode: true });
        }
      },
      openLyricMapping: openLyricMappingWorkflowModal,
      openLyricReview: openLyricReviewModal,
      openStoryboard: openStoryboardBuilderFromProject,
      createLocationsFromLyrics: createWizardLocationsFromLyrics,
      autoMapLocations: autoMapWizardLocations,
      detectLyricSections: detectWizardLyricSections,
      createStoryArc: createWizardStoryArc,
      createStoryBrief: createWizardStoryBrief,
      createSceneBeats: createWizardSceneBeats,
      updateStoryLayer: async (storyLayer = {}) => {
        state.builderStoryLayer = normalizeBuilderStoryLayer(storyLayer);
        await autoSaveSessionQuiet("wizard story layer updated");
        return normalizeBuilderStoryLayer(state.builderStoryLayer);
      },
      updateSceneDefaultSettings: updateWizardSceneDefaultSettings,
      applySceneDefaults: applyWizardSceneDefaults,
      saveWizardDraft: async (draft = {}) => {
        const projectFolder = activeProjectFolderForSave();
        if (!projectFolder) return null;
        return postJson("/vrgdg/music_builder/save_wizard_draft", {
          project_folder: projectFolder,
          lyrics: String(draft?.lyrics || ""),
          draft,
        }, 60000);
      },
      loadWizardDraft: async () => {
        const projectFolder = activeProjectFolderForSave();
        if (!projectFolder) return null;
        return postJson("/vrgdg/music_builder/load_wizard_draft", {
          project_folder: projectFolder,
        }, 60000);
      },
      runGemmaImageAll: async () => {
        await confirmAndRunGemmaT2IAll();
      },
      runFlowGptImageAll: async () => {
        state.imageModelMode = "flow_gpt";
        state.fluxKleinSettings.image_model_mode = "flow_gpt";
        state.fluxKleinSettings.enabled = false;
        syncFluxKleinPanel();
        syncFlowGptBrowserPanel();
        syncInspector();
        render();
        await flowGptImageAllScenes({ imageRunMode: "resume_missing" });
      },
      runGemmaVideoAll: async () => {
        if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3" || ["i2v", "rtv"].includes(currentVideoMode())) await runWizardStoryboardGemmaAll();
        else await confirmAndRunGemmaVideoAll();
      },
      buildFullVideo: async () => {
        await confirmAndRunFullBuild();
      },
      saveProject: (options = {}) => saveSession({ quiet: options?.quiet !== false, throwOnError: false }),
    });
  }

  // The Storyboard Builder owns the Video Builder prompt pipeline; register it without opening its UI.
  function storyboardPromptPipeline() {
    if (typeof storyboardPipeline.runner !== "function") openStoryboardBuilderFromProject({ registerPromptPipelineOnly: true });
    if (typeof storyboardPipeline.runner !== "function") {
      throw new Error("The Storyboard prompt pipeline is not ready yet. Refresh ComfyUI and try again.");
    }
    return storyboardPipeline.runner;
  }

  function openWizardBetaFromBuilder() {
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
      syncProjectVideoEngineUI(); syncVideoModePanel(); syncI2VVideoSettingsPanel();
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
        engine, mode, modes, ltxVersion: state.i2vVideoSettings.ltx_version || "2.5", imageMode: state.imageModelMode || "zimage",
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
          state.miniMaxH3Settings = cloneMiniMaxH3Settings({ ...state.miniMaxH3Settings, video_mode: values.mode });
          state.miniMaxH3TwoPassEnabled = values.mode === "image_reference_to_video";
          state.miniMaxH3ThreePassEnabled = false;
        } else state.videoModelMode = values.mode;
      }
      if (values.imageMode) {
        state.imageModelMode = values.imageMode;
        state.fluxKleinSettings.image_model_mode = values.imageMode;
        state.fluxKleinSettings.enabled = values.imageMode === "flux_klein";
      }
      if (values.performance) { state.videoType = normalizeVideoType(values.performance); syncVideoTypeControl(); }
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
      const marker = document.createComment("Wizard Beta panel position");
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
      const referenceMode = wizardBetaNeeds(draft.engine, draft.mode).references;
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
    openWizardBeta({
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
        if (miniMax) restores.push(movePanel(holder, miniMaxPassChooser));
        const tabs = contents.map((content, index) => {
          const wrapper = document.createElement("div");
          restores.push(movePanel(wrapper, content));
          content.style.display = "flex";
          return { label: index ? "Video Settings" : "Models", value: index ? "settings" : "models", content: wrapper };
        });
        holder.append(makeSubTabs(tabs).wrapper);
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
        const needs = wizardBetaNeeds(engine, mode);
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
              "textGemmaRunner", "qwenModelFile", "qwenMmprojFile", "gemmaModelFile", "gemmaContextLimit", "gemmaOutputTokenLimit", "gemmaGpuLayers",
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
          const needs = wizardBetaNeeds(draft.engine, draft.mode);
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

  return { openWizardBetaFromBuilder, openWizardFromBuilder };
}
