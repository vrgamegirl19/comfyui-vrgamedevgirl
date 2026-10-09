import {
  audioUrl,
  getJson,
  MAX_SCENE_RENDER_WAIT_HOURS,
  normalizeSceneRenderWaitHours,
  postJson,
  setBuilderAutomaticMemoryCleanupEnabled,
} from "./comfy_api.mjs";
import {
  DEFAULT_I2V_DIFFUSION_MODEL,
  DEFAULT_I2V_UNET,
  REQUIRED_LTX_MSR_LORA,
  REQUIRED_LTX25_MSR_LORA,
} from "./constants.mjs";
import {
  makeButton,
  makeCheckbox,
  makeField,
  makeInput,
  makePickerField,
  makeSelect,
  makeSettingsSection,
  normalizeProjectVideoEngine,
  setWidgetValue,
  toast,
} from "./controls.mjs";
import { formatTime } from "./format.mjs";
import { cloneI2VVideoSettings } from "./model_settings.mjs";
import { normalizeNotificationSettings } from "./notifications.mjs";
import {
  normalizeAutoImg2ImgCreativity,
  normalizeAutoImg2ImgStartStep,
  normalizeContinuityMode,
} from "./prompt_text.mjs";
import { newSegment } from "./segments.mjs";

export async function pickPath(kind, input) {
  try {
    const data = await postJson("/vrgdg/music_builder/pick_path", { kind });
    if (data.path) {
      input.value = data.path;
      return data.path;
    }
    return "";
  } catch (error) {
    toast(String(error?.message || error), true);
    return "";
  }
}

export function createProjectSetup({
  activateGlobalTimelineAudioPlayback, activeProjectFolderForSave, allEditableSegments, audio, audioInput,
  autoSaveSessionQuiet, createSilentTimelineAudioButton, drawWaveform, enforceAudioTimelineEnd,
  freezeTimingControl, getPreferredProjectRoot, globalScrub, loadButton, loadSrtButton,
  loadedGlobalAudioDuration, lutsTools, node, normalizeImportedSrtSegments, pickAudioButton, pickSrtButton,
  playBuilderNotification, projectInput, projectLyricNotesPath, pushHistory, refreshGemmaChoices,
  refreshLoraChoices, refreshModelChoices, renderList, renderSegments, saveSession, scenePanel, selectedTimelineRangeInfo,
  setPreferredProjectRoot, settingsModalControls, showBeatMarkersIfAvailable, silentAudioDurationInput,
  srtInput, state, syncI2VVideoSettingsPanel, syncInspector, syncLyricAndSubjectNoteFiles,
  syncLyricNoteControls, syncOverlayTrackControls, syncProjectVideoEngineUI, timelineDuration, timelineInfo,
  timelineRangeInfo, updateMultiSelectButton, updateSelectedMediaTools,
}) {
  async function loadAudio() {
    try {
      loadButton.disabled = true;
      loadButton.textContent = "Loading...";
      if (!audioInput.value.trim()) {
        const paths = await getJson("/vrgdg/music_builder/default_audio_srt_paths");
        audioInput.value = paths.audio_path || "";
        if (!audioInput.value.trim()) {
          throw new Error(`No temp audio file found.\nAudio folder: ${paths.audio_folder}`);
        }
      }
      const data = await postJson("/vrgdg/music_builder/analyze_audio", {
        audio_path: audioInput.value,
        project_folder: projectInput.value || state.projectFolder || "",
        target_peaks: 1800,
      }, 90000);
      audioInput.value = data.audio_path || audioInput.value;
      state.audioPath = audioInput.value;
      state.duration = Number(data.duration || 0);
      state.audioDuration = Number(data.duration || 0);
      enforceAudioTimelineEnd();
      state.peaks = data.peaks || [];
      state.beats = data.beats || [];
      state.detectedTempoBpm = Math.max(0, Number(data.tempo_bpm || 0));
      state.beatCalibration = null;
      showBeatMarkersIfAvailable();
      audio.dataset.path = data.audio_path || audioInput.value;
      audio.src = audioUrl(audio.dataset.path);
      audio.load();
      activateGlobalTimelineAudioPlayback(0);
      globalScrub.max = String(Math.max(0, state.duration));
      setWidgetValue(node, "audio_path", data.audio_path || audioInput.value);
      if (!state.segments.length) {
        state.segments.push(newSegment(0, Math.min(4, Math.max(0.05, state.duration || 4))));
        state.activeId = state.segments[0].id;
      }
      syncInspector();
      render();
      toast(`Loaded audio: ${formatTime(state.duration)}`);
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      loadButton.disabled = false;
      loadButton.textContent = "Load Audio";
    }
  }

  async function loadSrt(options = {}) {
    try {
      if (!srtInput.value.trim()) {
        const paths = await getJson("/vrgdg/music_builder/default_audio_srt_paths");
        srtInput.value = paths.srt_path || "";
        if (!srtInput.value.trim()) {
          throw new Error(`No temp SRT file found.\nSRT folder: ${paths.srt_folder}`);
        }
      }
      const data = await postJson("/vrgdg/music_builder/load_srt", {
        srt_path: srtInput.value,
      });
      pushHistory();
      state.segments = normalizeImportedSrtSegments(data.segments || []);
      state.overlaySegments = [];
      state.srtPath = data.srt_path || "";
      state.activeId = state.segments[0]?.id || "";
      state.timingFrozen = true;
      state.srtMode = true;
      state.showTimelineLyricNotes = true;
      freezeTimingControl.input.checked = true;
      syncLyricNoteControls();
      const lyricPath = projectLyricNotesPath();
      if (lyricPath) await syncLyricAndSubjectNoteFiles("SRT path load");
      syncInspector();
      render();
      if (!options.skipSessionSave && activeProjectFolderForSave()) await saveSession({ quiet: true, throwOnError: true });
      toast(`Loaded ${state.segments.length} SRT segment${state.segments.length === 1 ? "" : "s"}.\nTiming is frozen.`);
    } catch (error) {
      if (options.throwOnError) throw error;
      toast(String(error?.message || error), true);
    }
  }

  function chooseProjectAudioFile(file) {
    if (!file) return Promise.resolve(null);
    const projectFolder = projectInput.value || state.projectFolder;
    if (!projectFolder) {
      toast("Set the project folder first so the audio can be copied there.", true);
      return Promise.resolve(null);
    }
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const data = await postJson("/vrgdg/music_builder/save_project_audio", {
            project_folder: projectFolder,
            audio_data: String(reader.result || ""),
            audio_name: file.name || "project_audio.wav",
          }, 180000);
          audioInput.value = data.saved_path || "";
          state.audioPath = audioInput.value;
          state.duration = Number(data.duration || 0);
          state.audioDuration = Number(data.duration || 0);
          enforceAudioTimelineEnd();
          state.peaks = data.peaks || [];
          state.beats = data.beats || [];
          state.detectedTempoBpm = Math.max(0, Number(data.tempo_bpm || 0));
          state.beatCalibration = null;
          state.sceneAudioGlobalTime = 0;
          audio.dataset.path = audioInput.value;
          audio.src = audioUrl(audioInput.value);
          audio.load();
          activateGlobalTimelineAudioPlayback(0);
          setWidgetValue(node, "audio_path", audioInput.value);
          render();
          toast(`Loaded audio:\n${audioInput.value}`);
          resolve(data);
        } catch (error) {
          toast(String(error?.message || error), true);
          resolve(null);
        }
      };
      reader.onerror = () => {
        toast("Failed to read the audio file.", true);
        resolve(null);
      };
      reader.readAsDataURL(file);
    });
  }

  async function createSilentTimelineAudio() {
    const duration = Math.max(0.1, Number(silentAudioDurationInput.value || 60));
    return createSilentTimelineAudioForDuration(duration);
  }

  function chooseProjectSrtFile(file) {
    if (!file) return;
    const projectFolder = projectInput.value || state.projectFolder;
    if (!projectFolder) {
      toast("Set the project folder first so the SRT can be copied there.", true);
      return;
    }
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const data = await postJson("/vrgdg/music_builder/save_project_srt", {
          project_folder: projectFolder,
          srt_text: String(reader.result || ""),
        }, 60000);
        pushHistory();
        srtInput.value = data.srt_path || "";
        state.srtPath = srtInput.value;
        state.segments = normalizeImportedSrtSegments(data.segments || []);
        state.overlaySegments = [];
        state.activeId = state.segments[0]?.id || "";
        state.timingFrozen = true;
        state.srtMode = true;
        state.showTimelineLyricNotes = true;
        setWidgetValue(node, "srt_path", state.srtPath);
        syncLyricNoteControls();
        const lyricPath = projectLyricNotesPath();
        if (lyricPath) await syncLyricAndSubjectNoteFiles("SRT file import");
        syncInspector();
        render();
        if (activeProjectFolderForSave()) await saveSession({ quiet: true, throwOnError: true });
        toast(`Loaded ${state.segments.length} SRT segment${state.segments.length === 1 ? "" : "s"}.\n${state.srtPath}`);
      } catch (error) {
        toast(String(error?.message || error), true);
      }
    };
    reader.onerror = () => toast("Failed to read the SRT file.", true);
    reader.readAsText(file);
  }

  function openSettingsModal() {
    syncInspector();
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.58);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(980px,calc(100vw - 40px));max-height:calc(100vh - 48px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:14px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const title = document.createElement("div");
    title.textContent = "Builder Settings";
    title.style.cssText = "font-size:15px;font-weight:900;color:#cffafe;";
    const modalClose = makeButton("Close");
    header.append(title, modalClose);
    const pathGrid = document.createElement("div");
    pathGrid.style.cssText = "display:grid;grid-template-columns:1fr;gap:10px;";
    const customModelsRootInput = makeInput(state.customModelsRoot || "");
    customModelsRootInput.placeholder = "Optional custom models root, e.g. H:\\AIStuff\\models";
    const saveCustomModelsRootButton = makeButton("Save Models Root", "primary");
    const customModelsRootRow = document.createElement("div");
    customModelsRootRow.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:end;";
    customModelsRootRow.append(makeField("Custom ComfyUI models root", customModelsRootInput, "Optional. Point this at the folder that contains diffusion_models, text_encoders, vae, loras, LLM, and other Comfy model folders."), saveCustomModelsRootButton);
    const projectRootInput = makeInput(getPreferredProjectRoot());
    projectRootInput.placeholder = "Optional projects root, e.g. D:\\VRGDG Projects";
    const chooseProjectRootButton = makeButton("Choose Folder");
    const saveProjectRootButton = makeButton("Save Projects Root", "primary");
    const clearProjectRootButton = makeButton("Use ComfyUI Output");
    const projectRootActions = document.createElement("div");
    projectRootActions.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(120px,auto));gap:8px;";
    projectRootActions.append(chooseProjectRootButton, saveProjectRootButton, clearProjectRootButton);
    const projectRootNote = document.createElement("div");
    projectRootNote.textContent = "Optional and safe: only newly created projects use this parent folder. Existing projects and ComfyUI temporary render outputs are not moved. Leave blank to keep using the ComfyUI output folder. A full folder path entered in New Project still overrides this preference for that one project.";
    projectRootNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
    const projectStoragePanel = makeSettingsSection("Project Storage", [
      makeField("Default folder for new projects", projectRootInput),
      projectRootActions,
      projectRootNote,
    ], true);
    pathGrid.append(
      makePickerField("Audio file path", audioInput, pickAudioButton),
      makeField("Project folder", projectInput),
      makePickerField("SRT path", srtInput, pickSrtButton),
      customModelsRootRow,
    );
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(120px,1fr));gap:8px;";
    actions.append(loadButton, loadSrtButton);
    const note = document.createElement("div");
    note.textContent = "These are workflow setup paths and one-time loading actions. Keeping them here leaves more room for scene editing.";
    note.style.cssText = "font-size:12px;color:#a1a1aa;line-height:1.45;";
    const themeControlMount = document.createElement("div");
    themeControlMount.style.cssText = "display:flex;align-items:center;gap:8px;flex-wrap:wrap;";
    window.VRGDG_UIThemes?.mountControl?.(themeControlMount);
    if (!themeControlMount.children.length) {
      const themeUnavailable = document.createElement("div");
      themeUnavailable.textContent = "Theme controls are unavailable until the UI theme helper is loaded.";
      themeUnavailable.style.cssText = "font-size:12px;color:#a1a1aa;line-height:1.45;";
      themeControlMount.append(themeUnavailable);
    }
    const themePanel = makeSettingsSection("Builder UI Theme", [
      themeControlMount,
    ], false);
    const projectVideoEngineSelect = makeSelect([
      { value: "ltx", label: "LTX (current Builder)" },
      { value: "minimax_h3", label: "MiniMax H3" },
    ], normalizeProjectVideoEngine(state.projectVideoEngine));
    settingsModalControls.projectVideoEngineSelect = projectVideoEngineSelect;
    const projectVideoEngineNote = document.createElement("div");
    projectVideoEngineNote.textContent = "This choice belongs to the whole project. LTX projects keep the existing scene renderer unchanged. MiniMax H3 projects use the separate MiniMax scene action and exact-timeline adapter.";
    projectVideoEngineNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
    const projectVideoEnginePanel = makeSettingsSection("Project Video Engine", [
      makeField("Video engine for this project", projectVideoEngineSelect),
      projectVideoEngineNote,
    ], true);
    const ltxVersionSelect = makeSelect([
      { value: "2.5", label: "LTX 2.5 (default)" },
      { value: "2.3", label: "LTX 2.3 (legacy)" },
    ], state.i2vVideoSettings?.ltx_version || "2.5");
    const ltxVersionNote = document.createElement("div");
    ltxVersionNote.textContent = "LTX 2.5 is the default for all LTX video modes. Choose LTX 2.3 to use the legacy GGUF and dual-CLIP workflows.";
    ltxVersionNote.style.cssText = projectVideoEngineNote.style.cssText;
    const ltxVersionPanel = makeSettingsSection("LTX Version", [
      makeField("Builder LTX version", ltxVersionSelect),
      ltxVersionNote,
    ], true);
    ltxVersionSelect.addEventListener("change", async () => {
      const selectedVersion = ltxVersionSelect.value;
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
      await autoSaveSessionQuiet(`selected LTX ${ltxVersionSelect.value}`);
    });
    const sceneRenderWaitHoursInput = makeInput(
      String(normalizeSceneRenderWaitHours(state.sceneRenderWaitHours)),
      "number",
    );
    sceneRenderWaitHoursInput.min = "1";
    sceneRenderWaitHoursInput.max = String(MAX_SCENE_RENDER_WAIT_HOURS);
    sceneRenderWaitHoursInput.step = "1";
    const renderWaitingNote = document.createElement("div");
    renderWaitingNote.textContent = "How long the Builder follows each queued LTX or MiniMax scene before reporting a wait-limit error. This does not cancel ComfyUI's render. Existing projects default to 2 hours; choose 1–24 hours for future scene waits.";
    renderWaitingNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
    const renderWaitingPanel = makeSettingsSection("Render Waiting", [
      makeField("Scene render wait limit (hours)", sceneRenderWaitHoursInput),
      renderWaitingNote,
    ], true);
    const automaticMemoryCleanupControl = makeCheckbox(
      "Run automatic RAM/VRAM cleanup",
      Boolean(state.automaticMemoryCleanup),
    );
    const automaticMemoryCleanupNote = document.createElement("div");
    automaticMemoryCleanupNote.textContent = "Off is recommended when ComfyUI DynamicVRAM is enabled. When off, the Builder bypasses embedded RAMCleanup/VRAMCleanup nodes in hidden workflows and will not directly clear Comfy/Gemma caches between tasks, retries, errors, or stops. The manual Clear Memory button still works.";
    automaticMemoryCleanupNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
    const memoryManagementPanel = makeSettingsSection("Memory Management", [
      automaticMemoryCleanupControl.wrapper,
      automaticMemoryCleanupNote,
    ], false);
    const notificationSettings = normalizeNotificationSettings(state.notificationSettings);
    const notificationMode = makeSelect(["off", "errors", "batch", "complete", "all"], notificationSettings.mode);
    const successSound = makeSelect(["chime", "bell", "double_beep", "soft"], notificationSettings.success_sound);
    const errorSound = makeSelect(["warning", "double_beep", "soft"], notificationSettings.error_sound);
    const notificationVolume = makeInput(String(notificationSettings.volume), "number");
    notificationVolume.min = "0";
    notificationVolume.max = "1";
    notificationVolume.step = "0.05";
    const successCustomFile = document.createElement("input");
    successCustomFile.type = "file";
    successCustomFile.accept = "audio/*,.mp3,.wav,.ogg,.m4a";
    successCustomFile.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-size:12px;";
    const errorCustomFile = successCustomFile.cloneNode();
    const successCustomName = document.createElement("div");
    const errorCustomName = document.createElement("div");
    for (const item of [successCustomName, errorCustomName]) {
      item.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;overflow-wrap:anywhere;";
    }
    const syncCustomAudioLabels = () => {
      const settings = normalizeNotificationSettings(state.notificationSettings);
      successCustomName.textContent = settings.success_custom_name ? `Custom success sound: ${settings.success_custom_name}` : "No custom success sound. Built-in sound will be used.";
      errorCustomName.textContent = settings.error_custom_name ? `Custom error sound: ${settings.error_custom_name}` : "No custom error sound. Built-in sound will be used.";
    };
    const saveNotificationSettings = async () => {
      state.notificationSettings = normalizeNotificationSettings({
        ...state.notificationSettings,
        mode: notificationMode.value,
        success_sound: successSound.value,
        error_sound: errorSound.value,
        volume: notificationVolume.value,
      });
      syncCustomAudioLabels();
      await autoSaveSessionQuiet("notification settings");
    };
    const readCustomSound = (file, kind) => {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async () => {
        const dataUrl = String(reader.result || "");
        if (kind === "error") {
          state.notificationSettings.error_custom_audio = dataUrl;
          state.notificationSettings.error_custom_name = file.name || "custom error audio";
        } else {
          state.notificationSettings.success_custom_audio = dataUrl;
          state.notificationSettings.success_custom_name = file.name || "custom success audio";
        }
        syncCustomAudioLabels();
        await autoSaveSessionQuiet("custom notification sound");
        playBuilderNotification(kind, true);
      };
      reader.onerror = () => toast("Could not read custom notification audio.", true);
      reader.readAsDataURL(file);
    };
    const notificationGrid = document.createElement("div");
    notificationGrid.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;";
    notificationGrid.append(
      makeField("Notify me when", notificationMode, "Off: no sounds. Errors: failures only. Batch: full runs only. Complete: prompts/images/videos plus errors. All: every toast."),
      makeField("Volume", notificationVolume, "0 is silent, 1 is full volume."),
      makeField("Success sound", successSound),
      makeField("Failure sound", errorSound),
    );
    const customSoundGrid = document.createElement("div");
    customSoundGrid.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;";
    const successCustomWrap = document.createElement("div");
    const errorCustomWrap = document.createElement("div");
    const clearSuccessSound = makeButton("Clear Success Custom");
    const clearErrorSound = makeButton("Clear Error Custom");
    const testSuccessSound = makeButton("Test Success", "primary");
    const testErrorSound = makeButton("Test Failure", "primary");
    successCustomWrap.style.cssText = "display:flex;flex-direction:column;gap:7px;";
    errorCustomWrap.style.cssText = successCustomWrap.style.cssText;
    successCustomWrap.append(makeField("Custom success audio", successCustomFile), successCustomName, clearSuccessSound, testSuccessSound);
    errorCustomWrap.append(makeField("Custom failure audio", errorCustomFile), errorCustomName, clearErrorSound, testErrorSound);
    customSoundGrid.append(successCustomWrap, errorCustomWrap);
    const notificationPanel = makeSettingsSection("Audio Notifications", [
      notificationGrid,
      customSoundGrid,
    ], false);
    state.continuityMode = normalizeContinuityMode(state.continuityMode, state.autoChainLastFrame);
    const continuityModeSelect = makeSelect(["off", "i2v_chain", "img2img"], state.continuityMode || "off");
    for (const option of continuityModeSelect.options) {
      option.textContent = {
        off: "Off",
        i2v_chain: "Chain previous final frame into I2V",
        img2img: "Use previous final frame for next Img2Img",
      }[option.value] || option.value;
    }
    const autoChainStyleSelect = makeSelect(["continuous", "surreal", "transformation", "environment_shift"], state.autoChainStyle || "continuous");
    for (const option of autoChainStyleSelect.options) {
      option.textContent = {
        continuous: "Continuous",
        surreal: "Surreal",
        transformation: "Transformation",
        environment_shift: "Environment shift",
      }[option.value] || option.value;
    }
    const autoChainDirectionInput = makeInput(state.autoChainDirection || "");
    autoChainDirectionInput.placeholder = "Optional direction for chained scenes...";
    const autoChainTransitionLoraControl = makeCheckbox("Use Transition LoRA prompt style", Boolean(state.autoChainTransitionLoraPrompt));
    const autoChainTransitionTriggerInput = makeInput(state.autoChainTransitionTrigger || "zhuanchang");
    autoChainTransitionTriggerInput.placeholder = "zhuanchang";
    const autoImg2ImgStartStepSlider = document.createElement("input");
    autoImg2ImgStartStepSlider.type = "range";
    autoImg2ImgStartStepSlider.min = "1";
    autoImg2ImgStartStepSlider.max = "8";
    autoImg2ImgStartStepSlider.step = "1";
    autoImg2ImgStartStepSlider.value = String(normalizeAutoImg2ImgStartStep(state.autoImg2ImgStartStep));
    autoImg2ImgStartStepSlider.style.cssText = "width:100%;accent-color:#22d3ee;";
    const autoImg2ImgStartStepInput = makeInput(String(normalizeAutoImg2ImgStartStep(state.autoImg2ImgStartStep)), "number");
    autoImg2ImgStartStepInput.min = "1";
    autoImg2ImgStartStepInput.max = "8";
    autoImg2ImgStartStepInput.step = "1";
    const autoImg2ImgStartStepHint = document.createElement("div");
    autoImg2ImgStartStepHint.textContent = "ZImage/Ernie: 1 = more creative, 8 = more like the previous final frame.";
    autoImg2ImgStartStepHint.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;";
    const autoImg2ImgCreativitySlider = document.createElement("input");
    autoImg2ImgCreativitySlider.type = "range";
    autoImg2ImgCreativitySlider.min = "0";
    autoImg2ImgCreativitySlider.max = "10";
    autoImg2ImgCreativitySlider.step = "1";
    autoImg2ImgCreativitySlider.value = String(normalizeAutoImg2ImgCreativity(state.autoImg2ImgCreativity));
    autoImg2ImgCreativitySlider.style.cssText = "width:100%;accent-color:#22d3ee;";
    const autoImg2ImgCreativityInput = makeInput(String(normalizeAutoImg2ImgCreativity(state.autoImg2ImgCreativity)), "number");
    autoImg2ImgCreativityInput.min = "0";
    autoImg2ImgCreativityInput.max = "10";
    autoImg2ImgCreativityInput.step = "1";
    const autoImg2ImgCreativityHint = document.createElement("div");
    autoImg2ImgCreativityHint.textContent = "Krea 2: 0 ignores the image, 10 keeps the previous final frame most intact.";
    autoImg2ImgCreativityHint.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;";
    const autoChainGrid = document.createElement("div");
    autoChainGrid.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;";
    const continuityModeField = makeField("Continuity mode", continuityModeSelect);
    const autoChainStyleField = makeField("Chain style", autoChainStyleSelect);
    const autoChainDirectionField = makeField("Chain direction", autoChainDirectionInput);
    const autoChainTriggerField = makeField("Transition trigger", autoChainTransitionTriggerInput);
    const autoChainOnlyControls = document.createElement("div");
    autoChainOnlyControls.style.cssText = "display:contents;";
    const autoImg2ImgOnlyControls = document.createElement("div");
    autoImg2ImgOnlyControls.style.cssText = "display:contents;";
    const autoImg2ImgStartStepWrap = document.createElement("div");
    autoImg2ImgStartStepWrap.style.cssText = "display:flex;flex-direction:column;gap:6px;";
    autoImg2ImgStartStepWrap.append(makeField("Img2Img similarity", autoImg2ImgStartStepSlider), autoImg2ImgStartStepHint, makeField("Similarity value", autoImg2ImgStartStepInput));
    const autoImg2ImgCreativityWrap = document.createElement("div");
    autoImg2ImgCreativityWrap.style.cssText = "display:flex;flex-direction:column;gap:6px;";
    autoImg2ImgCreativityWrap.append(makeField("Krea Img2Img strength", autoImg2ImgCreativitySlider), autoImg2ImgCreativityHint, makeField("Strength value", autoImg2ImgCreativityInput));
    autoChainGrid.append(
      continuityModeField,
      autoChainOnlyControls,
      autoImg2ImgOnlyControls,
    );
    autoChainOnlyControls.append(
      autoChainStyleField,
      autoChainDirectionField,
      autoChainTransitionLoraControl.wrapper,
      autoChainTriggerField,
    );
    autoImg2ImgOnlyControls.append(autoImg2ImgStartStepWrap, autoImg2ImgCreativityWrap);
    const autoChainNote = document.createElement("div");
    autoChainNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;";
    autoChainNote.textContent = "Render All / Build Full Video only. I2V chain feeds each final frame plus a new Gemma Vision prompt into the next video scene. Img2Img continuity feeds each final frame into the next scene's image-to-image source, using the existing Storyboard/Wizard image prompt and the Img2Img strength shown here. Pick one mode or Off.";
    const autoChainPanel = makeSettingsSection("Scene Continuity", [
      autoChainGrid,
      autoChainNote,
    ], false);
    const saveAutoChainSettings = async () => {
      state.continuityMode = normalizeContinuityMode(continuityModeSelect.value || "off");
      state.autoChainLastFrame = state.continuityMode === "i2v_chain";
      state.autoChainStyle = autoChainStyleSelect.value || "continuous";
      state.autoChainDirection = autoChainDirectionInput.value || "";
      state.autoChainTransitionLoraPrompt = Boolean(autoChainTransitionLoraControl.input.checked);
      state.autoChainTransitionTrigger = autoChainTransitionTriggerInput.value || "zhuanchang";
      state.autoImg2ImgStartStep = normalizeAutoImg2ImgStartStep(autoImg2ImgStartStepInput.value || autoImg2ImgStartStepSlider.value);
      state.autoImg2ImgCreativity = normalizeAutoImg2ImgCreativity(autoImg2ImgCreativityInput.value || autoImg2ImgCreativitySlider.value);
      await autoSaveSessionQuiet("auto chain settings");
    };
    const syncContinuityModeVisibility = () => {
      const mode = normalizeContinuityMode(continuityModeSelect.value || "off");
      const imageMode = state.imageModelMode || "zimage";
      autoChainOnlyControls.style.display = mode === "i2v_chain" ? "contents" : "none";
      autoImg2ImgOnlyControls.style.display = mode === "img2img" ? "contents" : "none";
      autoImg2ImgStartStepWrap.style.display = mode === "img2img" && imageMode !== "krea2_2pass" ? "flex" : "none";
      autoImg2ImgCreativityWrap.style.display = mode === "img2img" && imageMode === "krea2_2pass" ? "flex" : "none";
    };
    const syncAutoImg2ImgStartStepInputs = () => {
      const value = normalizeAutoImg2ImgStartStep(autoImg2ImgStartStepInput.value || autoImg2ImgStartStepSlider.value);
      autoImg2ImgStartStepSlider.value = String(value);
      autoImg2ImgStartStepInput.value = String(value);
    };
    const syncAutoImg2ImgCreativityInputs = () => {
      const value = normalizeAutoImg2ImgCreativity(autoImg2ImgCreativityInput.value || autoImg2ImgCreativitySlider.value);
      autoImg2ImgCreativitySlider.value = String(value);
      autoImg2ImgCreativityInput.value = String(value);
    };
    continuityModeSelect.addEventListener("change", saveAutoChainSettings);
    continuityModeSelect.addEventListener("change", syncContinuityModeVisibility);
    autoChainStyleSelect.addEventListener("change", saveAutoChainSettings);
    autoChainDirectionInput.addEventListener("input", saveAutoChainSettings);
    autoChainTransitionLoraControl.input.addEventListener("change", saveAutoChainSettings);
    autoChainTransitionTriggerInput.addEventListener("input", saveAutoChainSettings);
    autoImg2ImgStartStepSlider.addEventListener("input", () => {
      autoImg2ImgStartStepInput.value = autoImg2ImgStartStepSlider.value;
      saveAutoChainSettings();
    });
    autoImg2ImgStartStepInput.addEventListener("input", () => {
      syncAutoImg2ImgStartStepInputs();
      saveAutoChainSettings();
    });
    autoImg2ImgCreativitySlider.addEventListener("input", () => {
      autoImg2ImgCreativityInput.value = autoImg2ImgCreativitySlider.value;
      saveAutoChainSettings();
    });
    autoImg2ImgCreativityInput.addEventListener("input", () => {
      syncAutoImg2ImgCreativityInputs();
      saveAutoChainSettings();
    });
    notificationMode.addEventListener("change", saveNotificationSettings);
    successSound.addEventListener("change", saveNotificationSettings);
    errorSound.addEventListener("change", saveNotificationSettings);
    notificationVolume.addEventListener("input", saveNotificationSettings);
    successCustomFile.addEventListener("change", () => readCustomSound(successCustomFile.files?.[0], "success"));
    errorCustomFile.addEventListener("change", () => readCustomSound(errorCustomFile.files?.[0], "error"));
    automaticMemoryCleanupControl.input.addEventListener("change", async () => {
      state.automaticMemoryCleanup = setBuilderAutomaticMemoryCleanupEnabled(automaticMemoryCleanupControl.input.checked);
      await autoSaveSessionQuiet("automatic memory cleanup setting");
      toast(state.automaticMemoryCleanup
        ? "Automatic RAM/VRAM cleanup enabled."
        : "Automatic RAM/VRAM cleanup disabled. DynamicVRAM will manage memory pressure.");
    });
    sceneRenderWaitHoursInput.addEventListener("change", async () => {
      state.sceneRenderWaitHours = normalizeSceneRenderWaitHours(sceneRenderWaitHoursInput.value);
      sceneRenderWaitHoursInput.value = String(state.sceneRenderWaitHours);
      await autoSaveSessionQuiet("scene render wait limit");
      toast(`Scene renders will now wait up to ${state.sceneRenderWaitHours} hour${state.sceneRenderWaitHours === 1 ? "" : "s"}.`);
    });
    projectVideoEngineSelect.addEventListener("change", async () => {
      state.projectVideoEngine = normalizeProjectVideoEngine(projectVideoEngineSelect.value);
      syncProjectVideoEngineUI();
      await autoSaveSessionQuiet("project video engine");
      toast(state.projectVideoEngine === "minimax_h3"
        ? "This project now uses the separate MiniMax H3 scene renderer."
        : "This project now uses the existing LTX scene renderer.");
    });
    clearSuccessSound.onclick = async () => {
      state.notificationSettings.success_custom_audio = "";
      state.notificationSettings.success_custom_name = "";
      syncCustomAudioLabels();
      await autoSaveSessionQuiet("clear success notification sound");
    };
    clearErrorSound.onclick = async () => {
      state.notificationSettings.error_custom_audio = "";
      state.notificationSettings.error_custom_name = "";
      syncCustomAudioLabels();
      await autoSaveSessionQuiet("clear failure notification sound");
    };
    testSuccessSound.onclick = () => playBuilderNotification("success", true);
    testErrorSound.onclick = () => playBuilderNotification("error", true);
    syncCustomAudioLabels();
    syncContinuityModeVisibility();
    const sceneOptionsNote = document.createElement("div");
    sceneOptionsNote.textContent = "Scene details, tools, and adjustments apply to the currently selected scene. Select a scene on the timeline before opening these options.";
    sceneOptionsNote.style.cssText = "font-size:12px;color:#a1a1aa;line-height:1.45;";
    const sceneOptionsPanel = makeSettingsSection("Scene Options", [sceneOptionsNote, scenePanel], false);
    box.append(header, pathGrid, actions, note, sceneOptionsPanel, projectStoragePanel, projectVideoEnginePanel, ltxVersionPanel, renderWaitingPanel, themePanel, memoryManagementPanel, autoChainPanel, notificationPanel);
    backdrop.append(box);
    document.body.append(backdrop);
    modalClose.onclick = () => backdrop.remove();
    chooseProjectRootButton.onclick = async () => {
      const path = await pickPath("project_root", projectRootInput);
      if (path) projectRootInput.value = path;
    };
    saveProjectRootButton.onclick = () => {
      const root = setPreferredProjectRoot(projectRootInput.value);
      projectRootInput.value = root;
      toast(root
        ? `New projects will be created under:\n${root}`
        : "New projects will use the ComfyUI output folder.");
    };
    clearProjectRootButton.onclick = () => {
      projectRootInput.value = "";
      setPreferredProjectRoot("");
      toast("New projects will use the ComfyUI output folder.");
    };
    saveCustomModelsRootButton.onclick = async () => {
      try {
        saveCustomModelsRootButton.disabled = true;
        saveCustomModelsRootButton.textContent = "Saving...";
        const data = await postJson("/vrgdg/workflow_runner/model_root", {
          models_root: customModelsRootInput.value,
        });
        state.customModelsRoot = data.models_root || "";
        customModelsRootInput.value = state.customModelsRoot;
        await Promise.all([refreshGemmaChoices(), refreshLoraChoices(), refreshModelChoices()]);
        const registeredCount = Array.isArray(data.registered) ? data.registered.length : 0;
        toast(`Custom models root saved.${registeredCount ? `\nRegistered ${registeredCount} model folder mapping${registeredCount === 1 ? "" : "s"}.` : ""}`);
      } catch (error) {
        toast(String(error?.message || error), true);
      } finally {
        saveCustomModelsRootButton.disabled = false;
        saveCustomModelsRootButton.textContent = "Save Models Root";
      }
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
  }

  async function createSilentTimelineAudioForDuration(duration, options = {}) {
    const projectFolder = activeProjectFolderForSave() || String(projectInput.value || "").trim();
    if (!projectFolder) {
      toast("Set the project folder first so the silent timeline audio can be created there.", true);
      return null;
    }
    state.projectFolder = projectFolder;
    projectInput.value = projectFolder;
    setWidgetValue(node, "project_folder", projectFolder);
    const targetDuration = Math.max(0.1, Number(duration || 60));
    try {
      createSilentTimelineAudioButton.disabled = true;
      createSilentTimelineAudioButton.textContent = "Creating...";
      const data = await postJson("/vrgdg/music_builder/create_silent_audio", {
        project_folder: projectFolder,
        scope: "project",
        duration: targetDuration,
      }, 180000);
      audioInput.value = data.audio_path || data.saved_path || "";
      state.audioPath = audioInput.value;
      state.duration = Number(data.duration || targetDuration);
      state.audioDuration = Number(data.duration || targetDuration);
      enforceAudioTimelineEnd();
      state.peaks = Array.isArray(data.peaks) ? data.peaks : [];
      state.beats = Array.isArray(data.beats) ? data.beats : [];
      state.detectedTempoBpm = Math.max(0, Number(data.tempo_bpm || 0));
      state.beatCalibration = null;
      state.sceneAudioGlobalTime = 0;
      audio.dataset.path = audioInput.value;
      audio.src = audioUrl(audioInput.value);
      audio.load();
      activateGlobalTimelineAudioPlayback(0);
      globalScrub.max = String(Math.max(0, state.duration));
      setWidgetValue(node, "audio_path", audioInput.value);
      if (!state.segments.length) {
        state.segments.push(newSegment(0, Math.min(4, Math.max(0.05, state.duration || 4))));
        state.activeId = state.segments[0].id;
      }
      syncInspector();
      render();
      try {
        await saveSession({ quiet: true, throwOnError: true });
      } catch (saveError) {
        console.warn("[VRGDG Music Builder] Silent timeline audio was created, but autosave failed:", saveError);
        toast(`Silent audio was created, but autosave failed:\n${String(saveError?.message || saveError)}`, true);
      }
      if (!options.quiet) toast(`Silent timeline audio created:\n${audioInput.value}`);
      return data;
    } catch (error) {
      toast(String(error?.message || error), true);
      return null;
    } finally {
      createSilentTimelineAudioButton.disabled = false;
      createSilentTimelineAudioButton.textContent = "Create Silent Audio";
    }
  }

  function render() {
    freezeTimingControl.input.checked = Boolean(state.timingFrozen);
    enforceAudioTimelineEnd();
    syncOverlayTrackControls();
    drawWaveform();
    renderSegments();
    renderList();
    if (state.leftPanelTab === "luts") {
      lutsTools.render();
    }
    updateSelectedMediaTools();
    updateMultiSelectButton();
    const timelineEnd = timelineDuration();
    const audioEnd = loadedGlobalAudioDuration();
    timelineInfo.textContent = `${state.segments.length} base / ${state.overlaySegments.length} overlay${state.overlaySegments.length === 1 ? "" : "s"} | Timeline ${formatTime(timelineEnd)}${audioEnd > 0 ? ` | Audio ${formatTime(audioEnd)}` : ""}`;
    const rangeInfo = selectedTimelineRangeInfo();
    timelineRangeInfo.textContent = rangeInfo
      ? `Range: ${formatTime(rangeInfo.start)} -> ${formatTime(rangeInfo.end)}  ${rangeInfo.duration.toFixed(2)}s`
      : "Range: none";
  }

  return {
    chooseProjectAudioFile, chooseProjectSrtFile, createSilentTimelineAudio,
    createSilentTimelineAudioForDuration, loadAudio, loadSrt, openSettingsModal, render,
  };
}
