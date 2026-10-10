import { api } from "../../../scripts/api.js";
import { postJson } from "./comfy_api.mjs";
import { makeButton, setWidgetValue, toast } from "./controls.mjs";
import { showSaveProjectAsModal, showTextInputModal } from "./dialogs.mjs";
import { cloneMiniMaxH3Settings } from "./minimax_h3.mjs";
import {
  defaultErnieImageSettings,
  defaultFlowGptBrowserSettings,
  defaultFluxKleinSettings,
  defaultI2VVideoSettings,
  defaultKrea2TwoPassSettings,
  defaultZEnhanceSettings,
  defaultZImageSettings,
  normalizeBuilderStoryLayer,
} from "./model_settings.mjs";
import { DEFAULT_KREA2_REFERENCE_SETTINGS } from "./models.mjs";
import {
  buildPromptCreatorLyricSegments,
  buildPromptCreatorSrtText,
  buildPromptCreatorWhisperPreview,
  loadContextTextQuiet,
  lyricsFromSegmentsForPromptCreator,
} from "./project_files.mjs";
import {
  defaultFluxReferenceBuilder,
  normalizeGemmaContextLimit,
  normalizeGemmaGpuLayers,
  normalizeLmStudioContextLimit,
  normalizeOutputTokenLimit,
} from "./prompt_text.mjs";
import { defaultIdLoraReferenceBuilder, defaultLyricMapper, normalizeLyricMapper } from "./reference_data.mjs";
import { newSegment } from "./segments.mjs";

export function timestampForProjectName() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}_${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;
}

function confirmLongBatchAction({ title, lines = [], confirmLabel = "Continue" } = {}) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = title || "Start Batch Process?";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const body = document.createElement("div");
    body.style.cssText = "display:flex;flex-direction:column;gap:8px;font-size:13px;color:#d4d4d8;line-height:1.45;";
    for (const line of lines) {
      const item = document.createElement("div");
      item.textContent = line;
      body.append(item);
    }
    const note = document.createElement("div");
    note.textContent = "This can take a long time. You can use Stop if you need to interrupt it.";
    note.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:9px;color:#fde68a;font-size:12px;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const confirm = makeButton(confirmLabel, "primary");
    cancel.onclick = () => {
      backdrop.remove();
      resolve(false);
    };
    confirm.onclick = () => {
      backdrop.remove();
      resolve(true);
    };
    actions.append(cancel, confirm);
    box.append(heading, body, note, actions);
    backdrop.append(box);
    document.body.append(backdrop);
  });
}

export function chooseBatchModeAction({ title, intro = "", choices = [], confirmLabel = "Continue", extraGroups = [], returnAll = false, defaultValue = "" } = {}) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(680px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = title || "Choose Batch Mode";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const text = document.createElement("div");
    text.textContent = intro || "";
    text.style.cssText = `display:${intro ? "block" : "none"};font-size:13px;color:#d4d4d8;line-height:1.45;`;
    let selected = choices.some((choice) => choice.value === defaultValue) ? defaultValue : choices[0]?.value || "";
    const extraSelected = {};
    const list = document.createElement("div");
    list.style.cssText = "display:flex;flex-direction:column;gap:8px;max-height:420px;overflow:auto;";
    choices.forEach((choice, index) => {
      const id = `vrgdg_batch_choice_${Date.now()}_${index}`;
      const label = document.createElement("label");
      label.htmlFor = id;
      label.style.cssText = "display:grid;grid-template-columns:auto minmax(0,1fr);gap:10px;align-items:start;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;cursor:pointer;";
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "vrgdg_batch_choice";
      input.id = id;
      input.value = choice.value;
      input.checked = choice.value === selected;
      input.style.marginTop = "3px";
      input.onchange = () => {
        if (input.checked) selected = choice.value;
      };
      const copy = document.createElement("div");
      const name = document.createElement("div");
      name.textContent = choice.label || choice.value;
      name.style.cssText = "font-weight:900;color:#f8fafc;font-size:13px;";
      const desc = document.createElement("div");
      desc.textContent = choice.description || "";
      desc.style.cssText = "margin-top:4px;color:#cbd5e1;font-size:12px;line-height:1.45;";
      copy.append(name, desc);
      label.append(input, copy);
      label.onclick = () => {
        input.checked = true;
        selected = choice.value;
      };
      list.append(label);
    });
    for (const group of extraGroups) {
      const groupBox = document.createElement("div");
      groupBox.style.cssText = "display:flex;flex-direction:column;gap:8px;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;";
      const groupTitle = document.createElement("div");
      groupTitle.textContent = group.label || "Options";
      groupTitle.style.cssText = "font-weight:900;color:#cffafe;font-size:13px;";
      const groupNote = document.createElement("div");
      groupNote.textContent = group.description || "";
      groupNote.style.cssText = `display:${group.description ? "block" : "none"};color:#cbd5e1;font-size:12px;line-height:1.45;`;
      const groupChoices = document.createElement("div");
      groupChoices.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;";
      const key = group.key || `extra_${Object.keys(extraSelected).length}`;
      extraSelected[key] = group.choices?.[0]?.value || "";
      (group.choices || []).forEach((choice, index) => {
        const id = `vrgdg_batch_extra_${key}_${Date.now()}_${index}`;
        const label = document.createElement("label");
        label.htmlFor = id;
        label.style.cssText = "display:grid;grid-template-columns:auto minmax(0,1fr);gap:8px;align-items:start;border:1px solid #334155;border-radius:7px;background:#111827;padding:9px;cursor:pointer;";
        const input = document.createElement("input");
        input.type = "radio";
        input.name = `vrgdg_batch_extra_${key}`;
        input.id = id;
        input.value = choice.value;
        input.checked = index === 0;
        input.style.marginTop = "3px";
        input.onchange = () => {
          if (input.checked) extraSelected[key] = choice.value;
        };
        const copy = document.createElement("div");
        const name = document.createElement("div");
        name.textContent = choice.label || choice.value;
        name.style.cssText = "font-weight:900;color:#f8fafc;font-size:12px;";
        const desc = document.createElement("div");
        desc.textContent = choice.description || "";
        desc.style.cssText = `display:${choice.description ? "block" : "none"};margin-top:3px;color:#cbd5e1;font-size:11px;line-height:1.35;`;
        copy.append(name, desc);
        label.append(input, copy);
        label.onclick = () => {
          input.checked = true;
          extraSelected[key] = choice.value;
        };
        groupChoices.append(label);
      });
      groupBox.append(groupTitle, groupNote, groupChoices);
      list.append(groupBox);
    }
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const confirm = makeButton(confirmLabel, "primary");
    cancel.onclick = () => {
      backdrop.remove();
      resolve("");
    };
    confirm.onclick = () => {
      backdrop.remove();
      resolve(returnAll ? { mode: selected, ...extraSelected } : selected);
    };
    actions.append(cancel, confirm);
    box.append(heading, text, list, actions);
    backdrop.append(box);
    document.body.append(backdrop);
  });
}

export function createProjectActions({
  audio, audioInput, autoLoadAll, clearHistoryBlobCache, clearTriggerPhrasesForFreshProject,
  closeBeatCalibrationWizard, createProgressWindow, currentSessionData, ernieImageTriggerInput,
  exportProjectButton, faceFixTool, fluxImageTriggerInput, getPreferredProjectRoot, i2vMotionJsonInput,
  imageTriggerInput, importProjectButton, isBlankStarterProject, loadGlobalModelDefaultsQuiet,
  loadSessionFromProject, node, pauseAllAudio, projectInput, promptJsonInput,
  referenceBuilderSubjectLocationText, rememberLastProject, render, resetBuilderETA,
  restoreBrowserAiDownloadsQuietly, saveI2VVideoSettingsFromPanel, saveSession, sceneAudio,
  sendToPromptCreatorButton, setBeatMarkersVisible, srtInput, state, storyIdeaInput, subjectSceneInput,
  syncErnieImagePanel, syncFluxKleinPanel, syncI2VVideoSettingsPanel, syncInspector, syncKrea2TwoPassPanel,
  syncProjectVideoEngineUI, syncVideoModePanel, syncVideoTypeControl, syncZEnhanceSettingsPanel,
  syncZImageSettingsPanel, themeStyleInput, updateActiveFromInputs, useVrgdgTextContext, videoTriggerInput,
}) {
  function resetProjectState(projectFolder, sessionPath = "", srtPath = "") {
    restoreBrowserAiDownloadsQuietly().catch(() => null);
    faceFixTool.reset?.();
    closeBeatCalibrationWizard();
    pauseAllAudio();
    audio.removeAttribute("src");
    sceneAudio.removeAttribute("src");
    const cleanProjectFolder = String(projectFolder || "").trim();
    const contextPath = (filename) => {
      const folder = cleanProjectFolder.replace(/[\\/]+$/, "");
      if (!folder) return "";
      const separator = folder.includes("\\") ? "\\" : "/";
      return `${folder}${separator}project_context${separator}${filename}`;
    };
    state.duration = 0;
    state.audioDuration = 0;
    state.audioPath = "";
    state.peaks = [];
    state.beats = [];
    state.beatCalibration = null;
    setBeatMarkersVisible(false);
    state.srtMode = false;
    state.timingFrozen = false;
    state.audioClips = null;
    state.speakingAudioDefaults = {};
    state.audioClipMixPath = "";
    state.audioClipMixKey = "";
    state.audioClipGenerationMixPath = "";
    state.audioClipGenerationMixKey = "";
    state.selectedTimelineRange = { in: null, out: null };
    state.timelineMarkers = [];
    state.activeTimelineMarkerId = "";
    state.promptJsonPath = contextPath("ConceptPrompts.txt");
    state.i2vMotionJsonPath = contextPath("I2VMotionNotes.txt");
    state.imageTriggerPhrase = "";
    state.videoTriggerPhrase = "";
    state.themeStylePath = contextPath("themestyle.txt");
    state.storyIdeaPath = contextPath("storyconcept.txt");
    state.subjectScenePath = contextPath("subjectsandscenes.txt");
    state.builderAgentMessages = [];
    state.builderAgentAutoApply = false;
    state.builderAgentPurpose = "scene_work";
    state.builderAgentReferenceImages = [];
    state.builderStorySourcePath = "";
    state.builderStorySourcePreview = "";
    state.builderStoryReferenceImages = [];
    state.builderStoryReferenceNotes = "";
    state.builderStoryLayer = normalizeBuilderStoryLayer({});
    state.renderLogs = [];
    resetBuilderETA();
    state.activeRenderLogId = "";
    state.useVrgdgTextContext = true;
    state.projectFolder = cleanProjectFolder;
    state.sessionPath = sessionPath || "";
    state.srtPath = srtPath || "";
    state.llmApiKey = "";
    state.llmApiKeyProject = "";
    state.elevenLabsApiKey = "";
    state.elevenLabsApiKeyProject = "";
    state.segments = [newSegment(0, 4)];
    state.overlaySegments = [];
    state.activeTrack = "base";
    state.activeId = state.segments[0]?.id || "";
    state.sceneAudioMode = false;
    state.sceneAudioSegmentId = "";
    state.sceneAudioGlobalTime = 0;
    state.sceneSelectionUsesGlobalAudio = false;
    state.zimageSettings = defaultZImageSettings();
    state.referenceKrea2Settings = { ...DEFAULT_KREA2_REFERENCE_SETTINGS };
    state.fluxKleinSettings = defaultFluxKleinSettings();
    state.flowGptBrowserSettings = defaultFlowGptBrowserSettings();
    state.ernieImageSettings = defaultErnieImageSettings();
    state.krea2TwoPassSettings = defaultKrea2TwoPassSettings();
    state.lyricMapper = defaultLyricMapper();
    state.useFluxGlobalImageIngredients = false;
    state.fluxGlobalImageIngredients = [];
    state.fluxReferenceBuilder = defaultFluxReferenceBuilder();
    state.idLoraReferenceBuilder = defaultIdLoraReferenceBuilder();
    state.zEnhanceSettings = defaultZEnhanceSettings();
    state.videoType = "singing";
    state.projectVideoEngine = "ltx";
    state.miniMaxH3Settings = cloneMiniMaxH3Settings();
    state.wizardBetaDraft = null;
    state.miniMaxH3TwoPassEnabled = false;
    state.miniMaxH3ThreePassEnabled = false;
    state.videoModelMode = "i2v";
    state.i2vVideoSettings = defaultI2VVideoSettings();
    state.promptToolsHintPrefs = {};
    projectInput.value = state.projectFolder;
    srtInput.value = state.srtPath;
    setWidgetValue(node, "audio_path", "");
    setWidgetValue(node, "project_folder", state.projectFolder);
    setWidgetValue(node, "session_path", state.sessionPath);
    setWidgetValue(node, "srt_path", state.srtPath);
    audioInput.value = "";
    audio.dataset.path = "";
    audio.removeAttribute("src");
    audio.load();
    promptJsonInput.value = state.promptJsonPath;
    i2vMotionJsonInput.value = state.i2vMotionJsonPath;
    imageTriggerInput.value = "";
    ernieImageTriggerInput.value = "";
    fluxImageTriggerInput.value = "";
    videoTriggerInput.value = "";
    themeStyleInput.value = state.themeStylePath;
    storyIdeaInput.value = state.storyIdeaPath;
    subjectSceneInput.value = state.subjectScenePath;
    useVrgdgTextContext.input.checked = true;
    state.undoStack = [];
    state.redoStack = [];
    clearHistoryBlobCache();
    syncZImageSettingsPanel();
    syncFluxKleinPanel();
    syncErnieImagePanel();
    syncKrea2TwoPassPanel();
    syncZEnhanceSettingsPanel();
    syncVideoTypeControl();
    syncProjectVideoEngineUI();
    syncI2VVideoSettingsPanel();
    syncVideoModePanel();
    syncInspector();
    render();
  }

  async function newProject() {
    const defaultName = `VRGDG_Project_${timestampForProjectName()}`;
    const projectName = await showTextInputModal({
      title: "New Project",
      label: "Project name or full project folder path",
      value: defaultName,
      confirmLabel: "Create Project",
    });
    if (projectName === null) return false;
    try {
      const data = await postJson("/vrgdg/music_builder/new_project", {
        project_folder: projectName,
        project_root: getPreferredProjectRoot(),
      }, 60000);
      resetProjectState(data.project_folder || "", data.session_path || "", data.srt_path || "");
      await loadGlobalModelDefaultsQuiet();
      clearTriggerPhrasesForFreshProject();
      syncZImageSettingsPanel();
      syncFluxKleinPanel();
      syncErnieImagePanel();
      syncI2VVideoSettingsPanel();
      if (data.concept_prompts_path) {
        promptJsonInput.value = data.concept_prompts_path;
        state.promptJsonPath = data.concept_prompts_path;
      }
      if (data.i2v_motion_notes_path) {
        i2vMotionJsonInput.value = data.i2v_motion_notes_path;
        state.i2vMotionJsonPath = data.i2v_motion_notes_path;
      }
      if (data.theme_style_path) {
        themeStyleInput.value = data.theme_style_path;
        state.themeStylePath = data.theme_style_path;
      }
      if (data.story_idea_path) {
        storyIdeaInput.value = data.story_idea_path;
        state.storyIdeaPath = data.story_idea_path;
      }
      if (data.subject_scene_path) {
        subjectSceneInput.value = data.subject_scene_path;
        state.subjectScenePath = data.subject_scene_path;
      }
      await state.applyVideoProfileToNewProject?.();
      rememberLastProject(state.projectFolder);
      await saveSession({ quiet: true });
      toast(`New project created.\n${state.projectFolder}`);
      return true;
    } catch (error) {
      toast(String(error?.message || error), true);
      return false;
    }
  }

  function openPromptCreatorPanel() {
    const creator = window.VRGDGMusicVideoPromptCreator;
    if (!creator?.open) {
      toast("Prompt Creator UI is not loaded yet. Refresh ComfyUI and try again.", true);
      return;
    }
    creator.open({
      projectFolder: projectInput.value || state.projectFolder || "",
      onSaved: (result) => {
        if (result?.project_folder) {
          projectInput.value = result.project_folder;
          state.projectFolder = result.project_folder;
          setWidgetValue(node, "project_folder", state.projectFolder);
        }
        if (result?.srt_path) {
          srtInput.value = result.srt_path;
          state.srtPath = result.srt_path;
          setWidgetValue(node, "srt_path", state.srtPath);
        }
        if (result?.files) {
          promptJsonInput.value = result.files["ConceptPrompts.txt"] || promptJsonInput.value;
          i2vMotionJsonInput.value = result.files["I2VMotionNotes.txt"] || i2vMotionJsonInput.value;
          themeStyleInput.value = result.files["themestyle.txt"] || themeStyleInput.value;
          storyIdeaInput.value = result.files["storyconcept.txt"] || storyIdeaInput.value;
          subjectSceneInput.value = result.files["subjectsandscenes.txt"] || subjectSceneInput.value;
          state.promptJsonPath = promptJsonInput.value;
          state.i2vMotionJsonPath = i2vMotionJsonInput.value;
          state.themeStylePath = themeStyleInput.value;
          state.storyIdeaPath = storyIdeaInput.value;
          state.subjectScenePath = subjectSceneInput.value;
        }
      },
      onSendToVideoCreator: async (result) => {
        const sourceProject = String(result?.project_folder || "").trim();
        const currentProject = String(projectInput.value || state.projectFolder || "").trim();
        if (sourceProject && (!currentProject || isBlankStarterProject())) {
          projectInput.value = sourceProject;
          state.projectFolder = sourceProject;
          setWidgetValue(node, "project_folder", state.projectFolder);
        }
        await autoLoadAll({ sourceProjectFolder: result?.project_folder || "", throwOnError: true });
      },
    });
  }

  async function sendCurrentProjectToPromptCreator() {
    let progress = null;
    try {
      if (!window.VRGDGMusicVideoPromptCreator?.open) {
        throw new Error("Prompt Creator UI is not loaded yet. Refresh ComfyUI and try again.");
      }
      sendToPromptCreatorButton.disabled = true;
      sendToPromptCreatorButton.textContent = "Sending...";
      progress = createProgressWindow("Sending To Prompt Creator");
      progress.set("Saving current Video Creator project...", 18);
      await saveSession({ quiet: true, throwOnError: true });
      const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
      if (!projectFolder) throw new Error("Create or load a project before sending it to Prompt Creator.");
      if (!state.segments.length) throw new Error("Create timeline scenes before sending them to Prompt Creator.");

      progress.set("Building Prompt Creator SRT and lyric segment draft...", 42);
      const lyricSegments = buildPromptCreatorLyricSegments(state.segments);
      const whisperPreview = buildPromptCreatorWhisperPreview(lyricSegments);
      const srtText = buildPromptCreatorSrtText(state.segments);
      const mapper = normalizeLyricMapper(state.lyricMapper);
      const fallbackLyrics = lyricsFromSegmentsForPromptCreator(state.segments);
      const referenceLyrics = String(mapper.source_text || "").trim() || fallbackLyrics;

      const existingSubjectLocations = await loadContextTextQuiet(subjectSceneInput.value || state.subjectScenePath);
      const refBuilderText = referenceBuilderSubjectLocationText();
      const subjectLocations = [existingSubjectLocations, refBuilderText]
        .map((value) => String(value || "").trim())
        .filter(Boolean)
        .filter((value, index, values) => values.indexOf(value) === index)
        .join("\n\n");

      const payload = {
        project_folder: projectFolder,
        audio_path: audioInput.value || "",
        full_lyrics: referenceLyrics,
        style_theme: await loadContextTextQuiet(themeStyleInput.value || state.themeStylePath),
        story_idea: await loadContextTextQuiet(storyIdeaInput.value || state.storyIdeaPath),
        subject_locations: subjectLocations,
        whisper_segments: whisperPreview,
        corrected_segments_text: JSON.stringify(lyricSegments, null, 2),
        concept_prompts_text: "",
        i2v_motion_notes_text: "",
        srt_text: srtText,
        use_srt_durations: true,
        fixed_scene_duration: 4,
        min_duration: 4,
        max_duration: 10,
        bias: 0.7,
        duration_preset: "varied_no_repeat",
        empty_segment_text: "[instrumental]",
        concept_match_mode: "medium",
        append_subject_to_prompts: true,
        repair_lyric_segments: false,
      text_gemma_runner: state.textGemmaRunner || "builtin",
      qwen_model_file: state.qwenModelFile || "",
      qwen_mmproj_file: state.qwenMmprojFile || "",
      gemma_model_file: state.gemmaModelFile || "",
        gemma_context_limit: normalizeGemmaContextLimit(state.gemmaContextLimit),
        gemma_output_token_limit: normalizeOutputTokenLimit(state.gemmaOutputTokenLimit),
        gemma_gpu_layers: normalizeGemmaGpuLayers(state.gemmaGpuLayers),
        lm_studio_base_url: state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1",
        lm_studio_model: state.lmStudioModel || "",
        lm_studio_api_key: state.lmStudioApiKey || "",
        lm_studio_context_limit: normalizeLmStudioContextLimit(state.lmStudioContextLimit),
        lm_studio_output_token_limit: normalizeOutputTokenLimit(state.lmStudioOutputTokenLimit),
      };

      progress.set("Saving Prompt Creator draft from this timeline...", 70);
      let result = null;
      try {
        result = await postJson("/vrgdg/music_prompt_creator/save_draft", payload, 90000);
      } catch (error) {
        if (/\b405\b/.test(String(error?.message || error))) {
          throw new Error("The Prompt Creator draft backend route is not loaded yet. Fully restart ComfyUI, refresh the browser, then try Send To Prompt Creator again.");
        }
        throw error;
      }
      if (result?.files?.["builder_segments.srt"]) {
        srtInput.value = result.files["builder_segments.srt"];
        state.srtPath = srtInput.value;
        setWidgetValue(node, "srt_path", state.srtPath);
      }
      progress.set("Opening Prompt Creator with this draft loaded...", 92);
      openPromptCreatorPanel();
      progress.set("Sent to Prompt Creator.", 100);
      progress.close(900);
      toast(`Sent current timeline lyrics and SRT to Prompt Creator.\n${result?.draft_path || projectFolder}`);
    } catch (error) {
      const message = String(error?.message || error);
      progress?.set(`Error:\n${message}`, 100);
      toast(message, true);
    } finally {
      sendToPromptCreatorButton.disabled = false;
      sendToPromptCreatorButton.textContent = "Send To Prompt Creator";
    }
  }

  async function saveProjectAs() {
    const currentProject = String(projectInput.value || state.projectFolder || "").trim();
    if (!currentProject) {
      toast("Create or load a project before using Save Project As.", true);
      return;
    }
    const currentName = currentProject.split(/[\\/]/).filter(Boolean).pop() || "VRGDG_Project";
    const targetProject = await showSaveProjectAsModal(`${currentName}_${timestampForProjectName()}`);
    if (targetProject === null) return;
    try {
      updateActiveFromInputs();
      saveI2VVideoSettingsFromPanel();
      const data = await postJson("/vrgdg/music_builder/save_project_as", {
        source_project_folder: currentProject,
        target_project_folder: targetProject,
        project_root: getPreferredProjectRoot(),
        audio_path: audioInput.value,
        session: currentSessionData(),
      }, 120000);
      await loadSessionFromProject(data.project_folder || targetProject);
      toast(`Project saved as.\n${state.projectFolder}`);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  async function exportShareableProject() {
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    if (!projectFolder) {
      toast("Create or load a project before exporting it.", true);
      return;
    }
    if (String(state.llmApiKeyProject || state.ownServerApiKeyProject || state.elevenLabsApiKeyProject || "").trim()) {
      const proceed = window.confirm(
        "This project contains a saved API key. Exporting a shareable ZIP may include that key in the project session. Continue exporting?",
      );
      if (!proceed) return;
    }
    try {
      exportProjectButton.disabled = true;
      exportProjectButton.textContent = "Preparing Project ZIP...";
      await saveSession({ quiet: true, throwOnError: true });
      const link = document.createElement("a");
      link.href = `/vrgdg/music_builder/export_project?project_folder=${encodeURIComponent(projectFolder)}`;
      link.download = "";
      link.style.display = "none";
      document.body.append(link);
      link.click();
      link.remove();
      toast("Project ZIP export started. Keep ComfyUI running until the browser download finishes.");
    } catch (error) {
      toast(`Could not export project:\n${String(error?.message || error)}`, true);
    } finally {
      exportProjectButton.disabled = false;
      exportProjectButton.textContent = "Export Shareable Project ZIP";
    }
  }

  async function importShareableProject() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".zip,.vrgdg.zip,application/zip";
    input.style.display = "none";
    document.body.append(input);
    input.onchange = async () => {
      const file = input.files?.[0];
      input.remove();
      if (!file) return;
      const progress = createProgressWindow("Importing Video Builder Project");
      try {
        importProjectButton.disabled = true;
        importProjectButton.textContent = "Importing Project...";
        progress.set(`Uploading and extracting ${file.name}...\nLarge video projects can take several minutes.`, 20);
        const form = new FormData();
        form.append("project_zip", file, file.name);
        const response = await api.fetchApi("/vrgdg/music_builder/import_project", {
          method: "POST",
          body: form,
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data?.ok) throw new Error(String(data?.error || `Import failed (${response.status})`));
        progress.set("Project extracted. Loading and rebasing project media paths...", 82);
        await loadSessionFromProject(data.project_folder);
        progress.set("Project imported and loaded.", 100);
        progress.close(1200);
        toast(`Imported portable project.\n${data.project_folder}`);
      } catch (error) {
        progress.set(`Import failed:\n${String(error?.message || error)}`, 100);
        toast(`Could not import project:\n${String(error?.message || error)}`, true);
      } finally {
        importProjectButton.disabled = false;
        importProjectButton.textContent = "Import Project ZIP";
      }
    };
    input.oncancel = () => input.remove();
    input.click();
  }

  return {
    exportShareableProject, importShareableProject, newProject, openPromptCreatorPanel, resetProjectState,
    saveProjectAs, sendCurrentProjectToPromptCreator,
  };
}
