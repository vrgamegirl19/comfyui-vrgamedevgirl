import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { BUILDER_FONT_STACK } from "./constants.mjs";
import {
  getWidget,
  makeButton,
  makeBuyMeACoffeeButton,
  makeCheckbox,
  makeField,
  makeInput,
  makeVideoTypeSelect,
  normalizeProjectVideoEngine,
  normalizeVideoType,
  styleCompactToolbarButton,
  toast,
  VIDEO_TYPE_OPTIONS,
} from "./controls.mjs";
import { showModelDownloadModal } from "./dialogs.mjs";
import { pickPath } from "./project_setup.mjs";
import { openRefModsStudio } from "./refmods_studio.mjs";
import { showOverlayTrackHelp } from "./timeline_actions.mjs";

export function wireToolbar({
  activeSegment, addOverlaySegment, addOverlaySegmentButton, addSegment, addSegmentButton,
  addTimelineMarkerButton, addTimelineMarkerFromSelection, applyBuilderFullscreen, autoBuildButton,
  autoLoadAll, autoLoadAllButton, autoSaveControl, autoSaveSessionQuiet, branchProject, branchProjectButton,
  builderAgentButton, bulkSegmentsButton, chooseGlobalAudioButton, chooseProjectAudioFile,
  chooseProjectSrtFile, chooseRenderedSceneTrimAtPlayhead, clearMemoryButton, clearRangeButton,
  clearSelectedTimelineRange, closeTimelineGapsButton, closeTimelineGapsFromMenu, confirmAndRunFullBuild,
  confirmAndRunFullFLFBuild, confirmAndRunGemmaT2IAll, confirmAndRunGemmaVideoAll, confirmAndRunRenderAll,
  confirmAndRunZEnhanceAll, confirmAndRunZImageAll, confirmOpenLegacyPromptCreator,
  convertAllLtxVideoPromptsToMiniMaxH3, convertLtxPromptsToMiniMaxButton, createSilentTimelineAudio,
  createSilentTimelineAudioButton, currentVideoMode, downloadModelsButton, editCurrentVideoPromptWithGemma,
  editI2VPromptButton, exportProjectButton, exportShareableProject, fluxReferenceBuilderButton,
  fullBuildButton, fullFLFBuildButton, fullscreen, fullscreenButton, gemmaRunnerButton, gemmaT2IAllButton,
  gemmaVideoAllButton, globalAudioDrop, globalAudioModeSelect, idLoraReferenceAudioInput,
  idLoraTrimModeButton, importI2VMotionJson, importI2VMotionJsonButton, importProjectButton, importPromptJson,
  importPromptJsonButton, importSceneNotesButton, importSceneNotesJson, importShareableProject, loadAudio,
  loadButton, loadLastProject, loadLastProjectButton, loadSession, loadSessionButton, loadSrt, loadSrtButton,
  lyricMapperButton, menuButton, menuDropdown, miniMaxH3SettingsForSegment, newProject, newProjectButton,
  openAutoBuildModal, openBuilderAgentModal, openBulkSegmentsModal, openGemmaRunnerModal,
  openLyricMappingWorkflowModal, openPromptOptionsModal, openReferenceBuilderTargetChooser,
  openRenderLogModal, openSceneAudioOptionsButton, openSceneOptions, openSettingsModal, openSnapSceneEdgeMenu,
  openStitchPreviewModal, openStoryboardBuilderFromProject, openWhatsNewModal, openWizardBetaFromBuilder,
  openWizardFromBuilder, overlayTrackHintButton, overlayTrackToggleButton, pauseTimelineForEditing,
  pickAudioButton, pickIdLoraReferenceAudioButton, pickSrtButton, projectAudioFileInput, projectSrtFileInput,
  promptCreatorButton, promptOptionsButton, redo, redoButton, renderAllButton, renderImageSlideshowPreview,
  renderLogButton, requireActiveSegment, reviewGuideButton, runClearMemoryWorkflow, saveButton,
  saveI2VVideoSettingsFromPanel, saveProjectAs, saveProjectAsButton, saveSession,
  sendCurrentProjectToPromptCreator, sendToPromptCreatorButton, setInButton, setOutButton,
  setTimelineRangePoint, settingsButton, silentAudioDurationInput, slideshowPreviewButton,
  snapSceneEdgeButton, splitActiveSceneAtPlayhead, splitSceneButton, state, stitchPreviewButton,
  stopCurrentWorkflow, stopWorkflowButton, storyboardBuilderButton, syncGlobalAudioModeControls,
  syncTimelineTrimModeButton, syncVideoTypeControl, toggleOverlayTrack, undo, undoButton,
  updateAudioScrubbers, updatePromptRunnerButtonLabels, updateStatus, updateStatusAction, updateV10Button,
  updateV10HintButton, updateWhatsNewAction, videoTypeSelect, whatsNewMenuButton, wizardBetaButton,
  wizardButton, zEnhanceAllButton, zEnhanceAllToolButton, zImageAllButton,
}) {
  menuButton.onclick = (event) => {
    event.stopPropagation();
    menuDropdown.style.display = menuDropdown.style.display === "flex" ? "none" : "flex";
  };
  updateV10HintButton.onclick = () => {
    window.alert(
      "What Update to Latest does:\n\n" +
      "1. Finds this custom node's installed folder automatically.\n" +
      "2. Runs: git fetch origin main\n" +
      "3. Runs: git switch main\n" +
      "4. Runs: git pull --ff-only origin main\n" +
      "5. If requirements.txt changed, installs it with the Python running ComfyUI.\n\n" +
      "It does not run git reset or git clean, and it does not delete files you created. Git will stop and show an error if switching or pulling would overwrite conflicting work. Nothing outside the comfyui-vrgamedevgirl folder is touched."
    );
  };
  updateV10Button.onclick = async () => {
    const confirmed = window.confirm(
      "Update these custom nodes to the latest production version from main now?\n\n" +
      "Git will stop if your code changes conflict. Files you created are not deleted. If requirements.txt changed, its packages will be installed with ComfyUI's Python.\n\nContinue?"
    );
    if (!confirmed) return;

    const originalText = updateV10Button.textContent;
    updateV10Button.disabled = true;
    updateV10Button.textContent = "Updating...";
    try {
      const response = await api.fetchApi("/vrgdg/update/v10", { method: "POST" });
      let payload = {};
      try {
        payload = await response.json();
      } catch (_) {
        payload = {};
      }
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || `Update failed (HTTP ${response.status}).`);
      }
      const requirementsNote = payload.requirements_error
        ? `The code update completed, but updated Python requirements could not be installed automatically:\n\n${payload.requirements_error}`
        : payload.requirements_changed
          ? "Updated Python requirements were installed successfully."
          : "No requirements.txt changes were detected, so dependency installation was skipped.";
      window.alert(
        `${requirementsNote}\n\nRESTART REQUIRED: Fully stop and restart ComfyUI, then hard-refresh the browser page so the new Python and JavaScript files load.\n\nAfter restarting, you can optionally open What's New from the status banner or Builder menu.`
      );
    } catch (error) {
      window.alert(`Update did not complete:\n\n${error?.message || error}`);
    } finally {
      updateV10Button.disabled = false;
      updateV10Button.textContent = originalText;
    }
  };
  updateStatusAction.onclick = () => updateV10Button.click();
  updateWhatsNewAction.onclick = () => {
    openWhatsNewModal({ mode: updateStatus.payload?.outdated ? "available" : "current" });
  };
  menuDropdown.addEventListener("click", (event) => {
    if (event.target === autoSaveControl.input || autoSaveControl.wrapper.contains(event.target)) return;
    menuDropdown.style.display = "none";
  });
  window.addEventListener("pointerdown", (event) => {
    if (menuDropdown.style.display !== "flex") return;
    if (menuDropdown.contains(event.target) || menuButton.contains(event.target)) return;
    menuDropdown.style.display = "none";
  });
  videoTypeSelect.onchange = async () => {
    state.videoType = normalizeVideoType(videoTypeSelect.value);
    syncVideoTypeControl();
    await autoSaveSessionQuiet("video type changed");
    toast(`Video Type set to ${VIDEO_TYPE_OPTIONS.find((item) => item.value === state.videoType)?.label || "Singing (music video)"}.`);
  };
  loadButton.onclick = () => {
    if (loadButton.disabled) return;
    loadAudio();
  };
  settingsButton.onclick = openSettingsModal;
  renderLogButton.onclick = openRenderLogModal;
  reviewGuideButton.onclick = () => {
    menuDropdown.style.display = "none";
    window.open(
      "https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/blob/main/Workflows/LTX-2_Workflows/Video_Builder/readme.md",
      "_blank",
      "noopener,noreferrer"
    );
  };
  whatsNewMenuButton.onclick = () => openWhatsNewModal({ mode: "history" });
  newProjectButton.onclick = async () => {
    await newProject();
  };
  saveProjectAsButton.onclick = saveProjectAs;
  branchProjectButton.onclick = branchProject;
  exportProjectButton.onclick = exportShareableProject;
  importProjectButton.onclick = importShareableProject;
  autoSaveControl.input.addEventListener("change", () => {
    state.autoSaveEnabled = Boolean(autoSaveControl.input.checked);
    if (state.projectFolder) {
      saveSession({ quiet: true });
    }
  });
  undoButton.onclick = undo;
  redoButton.onclick = redo;
  overlayTrackToggleButton.onclick = toggleOverlayTrack;
  overlayTrackHintButton.onclick = showOverlayTrackHelp;
  splitSceneButton.onclick = splitActiveSceneAtPlayhead;
  splitSceneButton.oncontextmenu = (event) => {
    event.preventDefault();
    event.stopPropagation();
    chooseRenderedSceneTrimAtPlayhead().catch((error) => toast(String(error?.message || error), true));
  };
  loadSrtButton.onclick = loadSrt;
  loadSessionButton.onclick = loadSession;
  loadLastProjectButton.onclick = loadLastProject;
  promptCreatorButton.onclick = confirmOpenLegacyPromptCreator;
  wizardButton.onclick = () => {
    try {
      openWizardFromBuilder();
    } catch (error) {
      console.error("VRGDG Video Wizard failed to open", error);
      toast(`Video Wizard failed to open:\n${String(error?.message || error)}`, true);
    }
  };
  wizardBetaButton.onclick = () => {
    try { openWizardBetaFromBuilder(); }
    catch (error) {
      console.error("VRGDG Wizard Beta failed to open", error);
      toast(`Wizard Beta failed to open:\n${String(error?.message || error)}`, true);
    }
  };
  autoBuildButton.onclick = () => {
    try {
      openAutoBuildModal();
    } catch (error) {
      console.error("VRGDG Auto Build failed to open", error);
      toast(`Auto Build failed to open:\n${String(error?.message || error)}`, true);
    }
  };
  storyboardBuilderButton.onclick = () => {
    try {
      openStoryboardBuilderFromProject();
    } catch (error) {
      console.error("VRGDG Storyboard Builder failed to open", error);
      toast(`Storyboard Builder failed to open:\n${String(error?.message || error)}`, true);
    }
  };
  fluxReferenceBuilderButton.onclick = openReferenceBuilderTargetChooser;
  lyricMapperButton.onclick = openLyricMappingWorkflowModal;
  sendToPromptCreatorButton.onclick = sendCurrentProjectToPromptCreator;
  promptOptionsButton.onclick = openPromptOptionsModal;
  gemmaRunnerButton.onclick = openGemmaRunnerModal;
  builderAgentButton.onclick = openBuilderAgentModal;
  autoLoadAllButton.onclick = autoLoadAll;
  importSceneNotesButton.onclick = importSceneNotesJson;
  clearMemoryButton.onclick = runClearMemoryWorkflow;
  renderAllButton.onclick = confirmAndRunRenderAll;
  stitchPreviewButton.onclick = openStitchPreviewModal;
  slideshowPreviewButton.onclick = renderImageSlideshowPreview;
  gemmaT2IAllButton.onclick = confirmAndRunGemmaT2IAll;
  gemmaVideoAllButton.onclick = confirmAndRunGemmaVideoAll;
  updatePromptRunnerButtonLabels();
  zImageAllButton.onclick = confirmAndRunZImageAll;
  zEnhanceAllButton.onclick = confirmAndRunZEnhanceAll;
  zEnhanceAllToolButton.onclick = confirmAndRunZEnhanceAll;
  convertLtxPromptsToMiniMaxButton.onclick = convertAllLtxVideoPromptsToMiniMaxH3;
  editI2VPromptButton.onclick = editCurrentVideoPromptWithGemma;
  fullBuildButton.onclick = confirmAndRunFullBuild;
  fullFLFBuildButton.onclick = confirmAndRunFullFLFBuild;
  stopWorkflowButton.onclick = stopCurrentWorkflow;
  downloadModelsButton.onclick = showModelDownloadModal;
  fullscreenButton.onclick = () => applyBuilderFullscreen(!fullscreen.enabled);
  openSceneAudioOptionsButton.onclick = () => {
    const segment = requireActiveSegment();
    if (segment) openSceneOptions(segment);
  };
  chooseGlobalAudioButton.onclick = () => projectAudioFileInput.click();
  globalAudioModeSelect.onchange = () => {
    syncGlobalAudioModeControls();
    if (globalAudioModeSelect.value === "silent") silentAudioDurationInput.focus();
  };
  createSilentTimelineAudioButton.onclick = createSilentTimelineAudio;
  globalAudioDrop.addEventListener("dragover", (event) => {
    event.preventDefault();
    globalAudioDrop.style.borderColor = "#a3e635";
    globalAudioDrop.style.background = "#064e3b";
  });
  globalAudioDrop.addEventListener("dragleave", () => {
    globalAudioDrop.style.borderColor = "#06b6d4";
    globalAudioDrop.style.background = "#082f49";
  });
  globalAudioDrop.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    globalAudioDrop.style.borderColor = "#06b6d4";
    globalAudioDrop.style.background = "#082f49";
    const file = Array.from(event.dataTransfer?.files || []).find((item) => item.type?.startsWith?.("audio/") || /\.(wav|mp3|flac|m4a|ogg)$/i.test(item.name || ""));
    if (file) chooseProjectAudioFile(file);
    else toast("Drop a WAV, MP3, FLAC, M4A, or OGG audio file for global timeline audio.", true);
  });
  pickAudioButton.onclick = () => projectAudioFileInput.click();
  pickIdLoraReferenceAudioButton.onclick = async () => {
    const path = await pickPath("audio", idLoraReferenceAudioInput);
    if (path) {
      saveI2VVideoSettingsFromPanel();
      toast("ID-LoRA reference voice sample selected.");
    }
  };
  pickSrtButton.onclick = () => projectSrtFileInput.click();
  projectAudioFileInput.onchange = () => {
    chooseProjectAudioFile(projectAudioFileInput.files?.[0]);
    projectAudioFileInput.value = "";
  };
  projectSrtFileInput.onchange = () => {
    chooseProjectSrtFile(projectSrtFileInput.files?.[0]);
    projectSrtFileInput.value = "";
  };
  saveButton.onclick = saveSession;
  importPromptJsonButton.onclick = importPromptJson;
  importI2VMotionJsonButton.onclick = importI2VMotionJson;
  bulkSegmentsButton.onclick = openBulkSegmentsModal;
  setInButton.onclick = () => setTimelineRangePoint("in");
  setOutButton.onclick = () => setTimelineRangePoint("out");
  clearRangeButton.onclick = clearSelectedTimelineRange;
  closeTimelineGapsButton.onclick = closeTimelineGapsFromMenu;
  snapSceneEdgeButton.onclick = openSnapSceneEdgeMenu;
  idLoraTrimModeButton.onclick = () => {
    const miniMaxBuiltInAudio = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3"
      && miniMaxH3SettingsForSegment(activeSegment()).audio_mode === "built_in_audio";
    if (currentVideoMode() !== "id_lora" && !miniMaxBuiltInAudio) return;
    state.timelineTrimEditMode = !state.timelineTrimEditMode;
    if (state.timelineTrimEditMode) pauseTimelineForEditing();
    syncTimelineTrimModeButton();
    updateAudioScrubbers();
    toast(state.timelineTrimEditMode ? "Trim Mode is on. Scrub to find the cut frame, then right-click the scene to trim." : "Trim Mode is off.");
  };
  addTimelineMarkerButton.onclick = addTimelineMarkerFromSelection;
  addSegmentButton.onclick = addSegment;
  addOverlaySegmentButton.onclick = addOverlaySegment;
}

export function buildProjectControls({ node }) {
  const topbar = document.createElement("div");
  topbar.style.cssText = `position:relative;display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:8px;align-items:center;padding:8px 10px;border-bottom:1px solid #27272a;background:#202024;min-width:0;font-family:${BUILDER_FONT_STACK};font-weight:500;letter-spacing:0;`;
  const audioInput = makeInput(String(getWidget(node, "audio_path")?.value || ""));
  const projectInput = makeInput(String(getWidget(node, "project_folder")?.value || ""));
  const srtInput = makeInput("");
  const pickAudioButton = makeButton("Pick");
  const pickSrtButton = makeButton("Pick");
  pickAudioButton.textContent = "Choose Audio";
  pickSrtButton.textContent = "Choose SRT";
  const settingsButton = makeButton("Settings");
  const reviewGuideButton = makeButton("Review Guide");
  reviewGuideButton.title = "Open the AI Video Builder guide on GitHub.";
  const whatsNewMenuButton = makeButton("What's New");
  whatsNewMenuButton.title = "Review recent AI Video Builder features, improvements, and fixes.";
  const loadButton = makeButton("Load Audio", "primary");
  const loadSrtButton = makeButton("Load SRT", "primary");
  const menuButton = makeButton("Menu");
  const loadSessionButton = makeButton("Load Project");
  const loadLastProjectButton = makeButton("Load Last Project");
  const newProjectButton = makeButton("New Project");
  const saveProjectAsButton = makeButton("Save Project As");
  const branchProjectButton = makeButton("Branch Project...");
  const exportProjectButton = makeButton("Export Shareable Project ZIP");
  const importProjectButton = makeButton("Import Project ZIP");
  const saveButton = makeButton("Quick Save", "primary");
  const videoTypeSelect = makeVideoTypeSelect("singing");
  const videoTypeField = makeField("Video Type", videoTypeSelect);
  videoTypeField.style.minWidth = "180px";
  videoTypeField.style.fontFamily = BUILDER_FONT_STACK;
  videoTypeField.style.fontWeight = "500";
  videoTypeSelect.style.fontFamily = BUILDER_FONT_STACK;
  videoTypeSelect.style.fontWeight = "400";
  const autoSaveControl = makeCheckbox("Auto save", true);
  autoSaveControl.wrapper.style.cssText += "border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:7px 10px;";
  const fullscreenButton = makeButton("Fullscreen");
  fullscreenButton.title = "Expand the Video Creator to fill the browser window without closing or resetting anything.";
  const closeButton = makeButton("Close");

  return {
    audioInput, autoSaveControl, branchProjectButton, closeButton, exportProjectButton, fullscreenButton,
    importProjectButton, loadButton, loadLastProjectButton, loadSessionButton, loadSrtButton, menuButton,
    newProjectButton, pickAudioButton, pickSrtButton, projectInput, reviewGuideButton, saveButton,
    saveProjectAsButton, settingsButton, srtInput, topbar, videoTypeField, videoTypeSelect,
    whatsNewMenuButton,
  };
}

export function buildTopbar({
  autoSaveControl, branchProjectButton, builderETAState, builderLifecycle, closeBuilderNow, closeButton,
  exportProjectButton, fullscreenButton, importProjectButton, loadLastProjectButton, loadSessionButton,
  menuButton, newProjectButton, overlay, reviewGuideButton, saveButton, saveProjectAsButton, saveSession,
  settingsButton, topbar, videoTypeField, whatsNewMenuButton,
}) {
  const confirmCloseBuilder = () => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100050;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:20px;box-sizing:border-box;";
    const box = document.createElement("div");
    box.style.cssText = `width:min(520px,calc(100vw - 40px));border:1px solid #155e75;border-radius:10px;background:#0f172a;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.65);padding:16px;display:flex;flex-direction:column;gap:12px;font-family:${BUILDER_FONT_STACK};`;
    const title = document.createElement("div");
    title.textContent = "Close Video Builder?";
    title.style.cssText = "font-size:17px;font-weight:600;color:#cffafe;";
    const note = document.createElement("div");
    note.textContent = "Are you sure you want to exit? Any unsaved changes will be lost unless you save first.";
    note.style.cssText = "font-size:13px;line-height:1.45;color:#cbd5e1;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;";
    const saveAndClose = makeButton("Save + Close", "primary");
    const closeWithoutSaving = makeButton("Close Without Saving");
    const goBack = makeButton("Go Back");
    closeWithoutSaving.style.borderColor = "#991b1b";
    closeWithoutSaving.style.color = "#fecaca";
    closeWithoutSaving.style.background = "#3f1518";
    goBack.onclick = () => backdrop.remove();
    closeWithoutSaving.onclick = () => {
      backdrop.remove();
      closeBuilderNow();
    };
    saveAndClose.onclick = async () => {
      saveAndClose.disabled = true;
      saveAndClose.textContent = "Saving...";
      try {
        await saveSession({ quiet: true, throwOnError: true });
        backdrop.remove();
        closeBuilderNow();
      } catch (error) {
        saveAndClose.disabled = false;
        saveAndClose.textContent = "Save + Close";
        toast(`Could not save before closing: ${error?.message || error}`, true);
      }
    };
    actions.append(saveAndClose, closeWithoutSaving, goBack);
    box.append(title, note, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
  };
  closeButton.onclick = confirmCloseBuilder;
  const promptCreatorButton = makeButton("Prompt Creator (Legacy)");
  const autoLoadAllButton = makeButton("Import Data From Prompt Creator");
  const importSceneNotesButton = makeButton("Import Scene Notes JSON");
  const wizardButton = makeButton("Wizard Legacy", "primary");
  const wizardBetaButton = makeButton("Wizard Beta", "primary");
  const autoBuildButton = makeButton("Auto Build", "primary");
  const storyboardBuilderButton = makeButton("Storyboard Builder");
  const fluxReferenceBuilderButton = makeButton("Reference Builder");
  const refModsStudioButton = makeButton("RefMods Studio");
  refModsStudioButton.onclick = openRefModsStudio;
  const lyricMapperButton = makeButton("Line Mapping");
  const sendToPromptCreatorButton = makeButton("Send To Prompt Creator");
  const promptOptionsButton = makeButton("Prompt Options");
  const gemmaRunnerButton = makeButton("LLM Runner");
  const builderAgentButton = makeButton("Agent");
  const clearMemoryButton = makeButton("Clear Memory");
  const renderAllButton = makeButton("Render All");
  const renderLogButton = makeButton("Render Log");
  renderLogButton.title = "View live and previous Render All timing reports.";
  const stitchPreviewButton = makeButton("Stitch Preview");
  const slideshowPreviewButton = makeButton("Image Slideshow Preview");
  const gemmaT2IAllButton = makeButton("LLM T2I All");
  const gemmaVideoAllButton = makeButton("LLM Video All");
  const zImageAllButton = makeButton("Image All");
  const zEnhanceAllButton = makeButton("Enhance All");
  const zEnhanceAllToolButton = makeButton("Enhance All");
  const convertLtxPromptsToMiniMaxButton = makeButton("Convert LTX Video Prompts to MiniMax H3", "primary");
  convertLtxPromptsToMiniMaxButton.title = "Use the selected LLM Runner to convert every populated LTX video prompt into a MiniMax H3 prompt. Global audio and all other project data stay unchanged.";
  const importImageFolderButton = makeButton("Fill Timeline Images From Folder", "primary");
  const projectBatchButton = makeButton("Project Batch", "primary");
  const fullBuildButton = makeButton("Build Full Video");
  const fullFLFBuildButton = makeButton("Build Full FLF Video");
  const stopWorkflowButton = makeButton("Stop");
  const downloadModelsButton = makeButton("Download Models");
  const buyMeACoffeeButton = makeBuyMeACoffeeButton();
  const updateV10Button = makeButton("Update to Latest");
  updateV10Button.style.background = "#9a3412";
  updateV10Button.style.borderColor = "#ea580c";
  updateV10Button.style.color = "#fff7ed";
  updateV10Button.style.fontWeight = "800";
  updateV10Button.title = "Fetch and fast-forward this installation to the latest production version on main.";
  const updateV10HintButton = makeButton("?");
  updateV10HintButton.title = "What does the updater do?";
  updateV10HintButton.style.cssText += "flex:0 0 36px;width:36px;text-align:center;justify-content:center;background:#431407;border-color:#c2410c;color:#ffedd5;font-weight:900;";
  const updateV10Row = document.createElement("div");
  updateV10Row.style.cssText = "display:flex;gap:6px;margin-top:6px;padding-top:8px;border-top:1px solid #3f3f46;";
  stopWorkflowButton.style.background = "#b91c1c";
  stopWorkflowButton.style.borderColor = "#7f1d1d";
  stopWorkflowButton.style.color = "#fee2e2";
  autoBuildButton.style.background = "linear-gradient(180deg,#22d3ee,#0891b2)";
  autoBuildButton.style.borderColor = "#67e8f9";
  autoBuildButton.style.color = "#082f49";
  styleCompactToolbarButton(saveButton, {
    lines: ["Quick", "Save"],
    icon: "save",
    width: 54,
    title: "Save the current project immediately.",
  });
  styleCompactToolbarButton(wizardButton, {
    lines: ["Wizard", "Legacy"],
    icon: "wizard",
    width: 58,
    title: "Open the legacy guided Video Builder setup workflow.",
  });
  styleCompactToolbarButton(wizardBetaButton, {
    lines: ["Wizard", "Beta"], icon: "wizard", width: 58,
    title: "Open Wizard Beta: save setup, prepare scenes, review, and render.",
  });
  styleCompactToolbarButton(autoBuildButton, {
    lines: ["Auto", "Build"],
    icon: "auto",
    width: 56,
    title: "Automatically build a complete music-video timeline from a song, lyrics, a singer image, and optional locations.",
  });
  styleCompactToolbarButton(storyboardBuilderButton, {
    lines: ["Story", "Builder"],
    icon: "story",
    width: 58,
    title: "Open Storyboard Builder to plan and edit scene prompts.",
  });
  styleCompactToolbarButton(fluxReferenceBuilderButton, {
    lines: ["Ref", "Builder"],
    icon: "reference",
    width: 54,
    title: "Open Reference Builder to manage characters and locations.",
  });
  styleCompactToolbarButton(refModsStudioButton, {
    lines: ["RefMods", "Studio"],
    icon: "reference",
    width: 58,
    title: "Create a RefMod from your images and save it by type under models/refmods.",
  });
  styleCompactToolbarButton(lyricMapperButton, {
    lines: ["Line", "Mapping"],
    icon: "mapping",
    width: 56,
    title: "Transcribe lyrics or dialogue and map performers to scenes.",
  });
  styleCompactToolbarButton(gemmaRunnerButton, {
    lines: ["LLM", "Runner"],
    icon: "brain",
    width: 54,
    title: "Choose the language-model runner used for prompt writing.",
  });
  styleCompactToolbarButton(promptOptionsButton, {
    lines: ["Prompt", "Options"],
    icon: "prompt",
    width: 58,
    title: "Open prompt editing, reload, clear, and prompt-file tools.",
  });
  styleCompactToolbarButton(downloadModelsButton, {
    lines: ["Models"],
    icon: "download",
    width: 54,
    title: "Download or review the models used by the builder.",
  });
  styleCompactToolbarButton(clearMemoryButton, {
    lines: ["Clear", "RAM"],
    icon: "memory",
    width: 52,
    title: "Clear Builder, ComfyUI, and model memory caches.",
  });
  styleCompactToolbarButton(fullscreenButton, {
    icon: "fullscreen",
    iconOnly: true,
    width: 40,
    ariaLabel: "Enter fullscreen",
    title: "Expand the Video Creator to fill the browser window without closing or resetting anything.",
  });
  styleCompactToolbarButton(closeButton, {
    icon: "close",
    iconOnly: true,
    width: 40,
    ariaLabel: "Close Video Builder",
    title: "Close the Video Builder.",
  });
  closeButton.style.borderColor = "#991b1b";
  closeButton.style.color = "#fecaca";
  closeButton.style.background = "#3f1518";
  stopWorkflowButton.style.width = "52px";
  stopWorkflowButton.style.minWidth = "52px";
  stopWorkflowButton.style.height = "42px";
  stopWorkflowButton.style.padding = "6px";
  stopWorkflowButton.style.fontFamily = "Inter,Segoe UI,Roboto,Arial,sans-serif";
  stopWorkflowButton.style.fontWeight = "600";
  stopWorkflowButton.style.letterSpacing = "0";
  menuButton.style.fontFamily = "Inter,Segoe UI,Roboto,Arial,sans-serif";
  menuButton.style.fontWeight = "500";
  menuButton.style.letterSpacing = "0";
  const menuDropdown = document.createElement("div");
  menuDropdown.style.cssText = "display:none;position:absolute;left:10px;top:calc(100% + 1px);z-index:20;min-width:260px;max-height:min(760px,calc(100vh - 88px));overflow-y:auto;overflow-x:hidden;overscroll-behavior:contain;scrollbar-gutter:stable;box-sizing:border-box;border:1px solid #3f3f46;border-radius:8px;background:#18181b;box-shadow:0 18px 60px rgba(0,0,0,.55);padding:8px;gap:6px;flex-direction:column;";
  const styleMenuItem = (button) => {
    button.style.width = "100%";
    button.style.textAlign = "left";
    button.style.justifyContent = "flex-start";
  };
  menuDropdown.append(buyMeACoffeeButton);
  for (const button of [newProjectButton, loadSessionButton, loadLastProjectButton, saveProjectAsButton, branchProjectButton, exportProjectButton, importProjectButton, settingsButton, reviewGuideButton, whatsNewMenuButton, gemmaT2IAllButton, gemmaVideoAllButton, zImageAllButton, zEnhanceAllButton, renderAllButton, renderLogButton, stitchPreviewButton, slideshowPreviewButton, fullBuildButton, fullFLFBuildButton]) {
    styleMenuItem(button);
    menuDropdown.append(button);
  }
  autoSaveControl.wrapper.style.marginTop = "4px";
  menuDropdown.append(autoSaveControl.wrapper);
  styleMenuItem(updateV10Button);
  updateV10Button.style.flex = "1 1 auto";
  updateV10Row.append(updateV10Button, updateV10HintButton);
  menuDropdown.append(updateV10Row);
  const projectActions = document.createElement("div");
  projectActions.style.cssText = "display:flex;gap:6px;align-items:center;flex-wrap:nowrap;min-width:max-content;";
  projectActions.append(menuButton, videoTypeField, saveButton);
  const batchActions = document.createElement("div");
  batchActions.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:nowrap;border-left:1px solid #3f3f46;border-right:1px solid #3f3f46;padding:0 10px;flex:0 0 auto;";
  batchActions.style.display = "none";
  const importActions = document.createElement("div");
  importActions.style.cssText = "display:flex;gap:5px;align-items:center;justify-content:center;flex-wrap:nowrap;min-width:0;overflow:visible;";
  importActions.append(wizardButton, wizardBetaButton, autoBuildButton, storyboardBuilderButton, fluxReferenceBuilderButton, refModsStudioButton, lyricMapperButton, gemmaRunnerButton, promptOptionsButton);
  const centerActions = document.createElement("div");
  centerActions.style.cssText = "position:relative;display:flex;gap:8px;align-items:center;justify-content:center;min-width:0;overflow:visible;";
  centerActions.append(importActions, batchActions);
  const builderETA = document.createElement("div");
  builderETA.setAttribute("aria-label", "Estimated render time remaining");
  builderETA.style.cssText = "display:none;position:absolute;left:0;top:50%;transform:translateY(-50%);width:190px;box-sizing:border-box;padding:6px 8px;border:1px solid #334155;border-radius:7px;background:#1e293b;color:#e2e8f0;font-size:12px;line-height:1.5;text-align:center;font-variant-numeric:tabular-nums;white-space:nowrap;";
  const builderSceneETA = document.createElement("div");
  const builderFullETA = document.createElement("div");
  builderETA.append(builderSceneETA, builderFullETA);
  centerActions.append(builderETA);
  const positionBuilderETA = () => {
    if (!builderETAState.log) return;
    const room = importActions.getBoundingClientRect().left - centerActions.getBoundingClientRect().left;
    const fits = room >= 200;
    const parent = fits ? centerActions : topbar;
    if (builderETA.parentElement !== parent) parent.append(builderETA);
    builderETA.style.position = fits ? "absolute" : "static";
    builderETA.style.transform = fits ? "translateY(-50%)" : "none";
    builderETA.style.gridColumn = fits ? "" : "1 / -1";
    builderETA.style.justifySelf = "center";
  };
  const builderETAResizeObserver = new ResizeObserver(positionBuilderETA);
  builderETAResizeObserver.observe(centerActions);
  builderETAResizeObserver.observe(importActions);

  const builderResourceMonitor = document.createElement("div");
  builderResourceMonitor.setAttribute("aria-label", "Video Builder RAM and VRAM usage");
  builderResourceMonitor.style.cssText = `position:absolute;right:0;top:50%;transform:translateY(-50%);display:none;align-items:center;gap:10px;width:250px;height:42px;box-sizing:border-box;padding:5px 9px;border:1px solid #3f3f46;border-radius:7px;background:#18181b;color:#d4d4d8;font-family:${BUILDER_FONT_STACK};font-size:10px;line-height:1.25;pointer-events:auto;`;
  builderResourceMonitor.innerHTML = `
    <div style="flex:1 1 0;min-width:0;">
      <div style="display:flex;justify-content:space-between;gap:6px;"><span>VRAM</span><strong data-builder-resource-value="vram" style="color:#67e8f9;font-weight:600;white-space:nowrap;">—</strong></div>
      <div style="height:3px;margin-top:4px;overflow:hidden;border-radius:3px;background:#3f3f46;"><i data-builder-resource-bar="vram" style="display:block;width:0;height:100%;background:#06b6d4;"></i></div>
    </div>
    <div style="flex:1 1 0;min-width:0;">
      <div style="display:flex;justify-content:space-between;gap:6px;"><span>RAM</span><strong data-builder-resource-value="ram" style="color:#c4b5fd;font-weight:600;white-space:nowrap;">—</strong></div>
      <div style="height:3px;margin-top:4px;overflow:hidden;border-radius:3px;background:#3f3f46;"><i data-builder-resource-bar="ram" style="display:block;width:0;height:100%;background:#8b5cf6;"></i></div>
    </div>
  `;
  builderResourceMonitor.title = "RAM and VRAM on the machine running ComfyUI. Updated every two seconds.";
  centerActions.append(builderResourceMonitor);

  const renderBuilderResourceMetric = (key, memory) => {
    const value = builderResourceMonitor.querySelector(`[data-builder-resource-value="${key}"]`);
    const bar = builderResourceMonitor.querySelector(`[data-builder-resource-bar="${key}"]`);
    const used = Number(memory?.used);
    const total = Number(memory?.total);
    const valid = Number.isFinite(used) && Number.isFinite(total) && total > 0;
    const usage = valid ? Math.max(0, Math.min(100, used / total * 100)) : 0;
    value.textContent = valid ? `${(used / 2 ** 30).toFixed(1)}/${(total / 2 ** 30).toFixed(1)} GB` : "—";
    value.title = valid ? `${usage.toFixed(1)}% used` : "Reading unavailable";
    bar.style.width = `${usage}%`;
  };
  const positionBuilderResourceMonitor = () => {
    const centerBounds = centerActions.getBoundingClientRect();
    const actionBounds = importActions.getBoundingClientRect();
    const availableRight = centerBounds.right - actionBounds.right;
    builderResourceMonitor.style.display = availableRight >= 265 ? "flex" : "none";
  };
  const pollBuilderResources = async () => {
    if (!overlay.isConnected || builderLifecycle.resourceController) return;
    builderLifecycle.resourceController = new AbortController();
    const timeout = setTimeout(() => builderLifecycle.resourceController?.abort(), 5000);
    try {
      const response = await api.fetchApi("/vrgdg/resource-monitor", {
        signal: builderLifecycle.resourceController.signal,
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const selectedGpu = String(app.extensionManager.setting.get("VRGDG.ResourceMonitor.GPU") ?? 0);
      const gpu = Array.isArray(data?.gpus)
        ? (data.gpus.find((item) => String(item?.index) === selectedGpu) || data.gpus[0])
        : null;
      renderBuilderResourceMetric("vram", gpu);
      renderBuilderResourceMetric("ram", data?.ram);
      builderResourceMonitor.title = gpu
        ? `GPU ${gpu.index} · ${gpu.name} · RAM and VRAM on the machine running ComfyUI`
        : "RAM is available. VRAM requires NVIDIA nvidia-smi.";
    } catch {
      if (overlay.isConnected) {
        renderBuilderResourceMetric("vram", null);
        renderBuilderResourceMetric("ram", null);
        builderResourceMonitor.title = "Resource monitor unavailable. Restart ComfyUI after installing the monitor.";
      }
    } finally {
      clearTimeout(timeout);
      builderLifecycle.resourceController = null;
      if (overlay.isConnected) builderLifecycle.resourceTimer = setTimeout(pollBuilderResources, 2000);
    }
  };
  const utilityActions = document.createElement("div");
  utilityActions.style.cssText = "display:flex;gap:5px;align-items:center;justify-content:flex-end;flex-wrap:nowrap;min-width:max-content;";
  const projectVideoEngineBadge = document.createElement("div");
  projectVideoEngineBadge.textContent = "◈ LTX";
  projectVideoEngineBadge.title = "Switch project video engine: LTX";
  projectVideoEngineBadge.setAttribute("role", "button");
  projectVideoEngineBadge.setAttribute("tabindex", "0");
  projectVideoEngineBadge.setAttribute("aria-label", "Switch project video engine");
  projectVideoEngineBadge.setAttribute("aria-live", "polite");
  projectVideoEngineBadge.style.cssText = `height:26px;box-sizing:border-box;display:inline-flex;align-items:center;padding:0 9px;border:1px solid #60a5fa;border-radius:999px;background:#172554;color:#bfdbfe;font-family:${BUILDER_FONT_STACK};font-size:11px;font-weight:600;letter-spacing:0;white-space:nowrap;cursor:pointer;user-select:none;`;
  utilityActions.append(projectVideoEngineBadge, stopWorkflowButton, downloadModelsButton, clearMemoryButton, fullscreenButton, closeButton);
  topbar.append(projectActions, centerActions, utilityActions, menuDropdown);

  return {
    autoBuildButton, autoLoadAllButton, builderAgentButton, builderETA, builderETAResizeObserver,
    builderFullETA, builderSceneETA, centerActions, clearMemoryButton, convertLtxPromptsToMiniMaxButton,
    downloadModelsButton, fluxReferenceBuilderButton, fullBuildButton, fullFLFBuildButton, gemmaRunnerButton,
    gemmaT2IAllButton, gemmaVideoAllButton, importActions, importImageFolderButton, importSceneNotesButton,
    lyricMapperButton, menuDropdown, pollBuilderResources, positionBuilderETA, positionBuilderResourceMonitor,
    projectBatchButton, projectVideoEngineBadge, promptCreatorButton, promptOptionsButton, renderAllButton,
    renderLogButton, sendToPromptCreatorButton, slideshowPreviewButton, stitchPreviewButton,
    stopWorkflowButton, storyboardBuilderButton, updateV10Button, updateV10HintButton, wizardBetaButton,
    wizardButton, zEnhanceAllButton, zEnhanceAllToolButton, zImageAllButton,
  };
}

export function createFullscreen({
  applyLayoutSizes, fullscreen, fullscreenButton, normalShellStyle, overlay, render, shell,
}) {
  const fullscreenShellStyle = `
    width: 100vw;
    height: 100vh;
    display: grid;
    grid-template-rows: auto minmax(0,1fr) minmax(230px, 34vh);
    background: #18181b;
    color: #fafafa;
    border: 0;
    border-radius: 0;
    overflow: hidden;
    box-shadow: none;
  `;

  function applyBuilderFullscreen(enabled) {
    fullscreen.enabled = Boolean(enabled);
    shell.style.cssText = fullscreen.enabled ? fullscreenShellStyle : normalShellStyle;
    overlay.style.alignItems = fullscreen.enabled ? "stretch" : "center";
    overlay.style.justifyContent = fullscreen.enabled ? "stretch" : "center";
    overlay.style.background = fullscreen.enabled ? "#09090b" : "rgba(0,0,0,.72)";
    styleCompactToolbarButton(fullscreenButton, {
      icon: fullscreen.enabled ? "restore" : "fullscreen",
      iconOnly: true,
      width: 40,
      ariaLabel: fullscreen.enabled ? "Exit fullscreen" : "Enter fullscreen",
      title: fullscreen.enabled
        ? "Return the Video Creator to the normal floating panel size."
        : "Expand the Video Creator to fill the browser window without closing or resetting anything.",
    });
    applyLayoutSizes();
    render();
  }

  return { applyBuilderFullscreen };
}
