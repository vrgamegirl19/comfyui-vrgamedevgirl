import { api } from "../../../scripts/api.js";
import "./prompt_creator.mjs";
import { createMusicVideoBuilderLuts } from "../VRGDG_MusicVideoBuilderLUTs.js";
import { createPostProcessComparePreview } from "../VRGDG_PostProcessComparePreview.js";
import { createFaceFixTool } from "../VRGDG_FaceFixUI.js?v=20260716-1";

import { normalizeOverlayTrackState } from "../VRGDG_OverlayTrack.js";
import "./styles.mjs";
import {
  DEFAULT_SCENE_RENDER_WAIT_HOURS,
  getJson,
  makeEditorImageUrl,
  makeImageViewUrl,
  postJson,
  queueWorkflowPrompt,
  saveBuilderSessionJson,
  setBuilderAutomaticMemoryCleanupEnabled,
  waitForImages,
} from "./comfy_api.mjs";
import { BUILDER_FONT_STACK, BUILDER_UI_VERSION } from "./constants.mjs";
import {
  makeButton,
  makeField,
  makeSettingsPanel,
  makeSettingsSection,
  makeSubTabs,
  normalizeProjectVideoEngine,
  setWidgetValue,
  toast,
} from "./controls.mjs";
import { pickProjectSessionFile, showInfoModal, showLastProgressWindow, showLoadProjectModal, showTextInputModal } from "./dialogs.mjs";
import { cloneMiniMaxH3Settings } from "./minimax_h3.mjs";
import { DEFAULT_KREA2_REFERENCE_SETTINGS } from "./models.mjs";
import { defaultNotificationSettings } from "./notifications.mjs";
import { createLyricReview } from "./lyric_review.mjs";
import {
  buildInspectorControls,
  buildInspectorInputs,
  buildInspectorPanels,
  buildInspectorTabs,
  createInspector,
} from "./inspector.mjs";
import {
  createModelSettings,
  defaultErnieImageSettings,
  defaultFlowGptBrowserSettings,
  defaultFluxKleinSettings,
  defaultI2VVideoSettings,
  defaultKrea2TwoPassSettings,
  defaultZEnhanceSettings,
  defaultZImageSettings,
  normalizeAutoBuildPreparation,
  normalizeBuilderStoryboardDefaults,
  normalizeBuilderStoryLayer,
} from "./model_settings.mjs";
import { createMiniMaxPanel } from "./minimax_panel.mjs";
import { wireMiniMaxPanel } from "./minimax_panel_events.mjs";
import { buildMiniMaxPanel } from "./minimax_panel_layout.mjs";
import { createRenderLog } from "./render_log.mjs";
import { buildPostProcessPane, createPostProcess } from "./post_process.mjs";
import { createLlmRunner, defaultNBImageSettings } from "./llm_runner.mjs";
import {
  createPromptText,
  defaultFluxReferenceBuilder,
  isInstrumentalLyricText,
  isNoLipSyncSingerChoice,
} from "./prompt_text.mjs";
import {
  createReferenceData,
  defaultIdLoraReferenceBuilder,
  defaultLyricMapper,
  normalizeFluxReferenceBuilder,
} from "./reference_data.mjs";
import { createSelectionPreview, hasLockedVideo, selectedSegmentVideoPath } from "./selection_preview.mjs";
import { createImageReferences } from "./image_references.mjs";
import {
  buildImageModeCards,
  buildImageSettingsPanels,
  createImagePanels,
  wireFluxKleinControls,
  wireImageModelControls,
  wireImageSettingsInputs,
  wireZEnhanceControls,
} from "./image_panels.mjs";
import { createBrowserAi, wireBrowserAiPanel } from "./browser_ai.mjs";
import { buildTimelineView, createTimelineView } from "./timeline_view.mjs";
import { createMediaImport, installFileDropNavigationGuard } from "./media_import.mjs";
import { createProjectFiles, wireContextFileInputs } from "./project_files.mjs";
import { createLyricCues } from "./lyric_cues.mjs";
import { applyLyricSectionsFromReferenceText, createLyricTranscription } from "./lyric_transcription.mjs";
import { createImageGeneration, zEnhancePayloadFromSettings } from "./image_generation.mjs";
import { captureVideoFrameDataUrl, createTimelineActions, parseBulkTimeValue } from "./timeline_actions.mjs";
import { createProjectActions } from "./project_actions.mjs";
import { createHistory } from "./history.mjs";
import { createNotificationSounds } from "./notification_sounds.mjs";
import {
  buildVideoSettingsPanels,
  createVideoSettingsPanel,
  wireVideoSettingsControls,
} from "./video_settings_panel.mjs";
import { createTimelineEdit } from "./timeline_edit.mjs";
import { createLyricMapping } from "./lyric_mapping.mjs";
import { createPromptEditing } from "./prompt_editing.mjs";
import { createBuilderAgent } from "./builder_agent.mjs";
import { createGemmaRunner } from "./gemma_runner.mjs";
import { createBatchActions } from "./batch_actions.mjs";
import { createAutoBuild } from "./auto_build.mjs";
import { createReferenceBuilder } from "./reference_builder.mjs";
import { createIngredientsBuilder } from "./ingredients_builder.mjs";
import { createPromptCreators } from "./prompt_creators.mjs";
import { createWizardBridge } from "./wizard_bridge.mjs";
import { createStoryboardBridge } from "./storyboard_bridge.mjs";
import { wirePanelButtons, wireReferenceControls, wireSceneInputs } from "./inspector_events.mjs";
import { buildProjectControls, buildTopbar, createFullscreen, wireToolbar } from "./toolbar.mjs";
import { wireTimelineControls } from "./timeline_events.mjs";
import { wireKeyboardShortcuts } from "./keyboard_shortcuts.mjs";
import { buildUpdateBanner, createWhatsNewModal } from "./update_banner.mjs";
import { buildLeftPanel } from "./left_panel.mjs";
import { createSession } from "./session.mjs";
import { createSceneOutput } from "./scene_output.mjs";
import { createSceneRenderPrep } from "./scene_render_prep.mjs";
import { createImagePrompts } from "./image_prompts.mjs";
import { createMiniMaxPrompt } from "./minimax_prompt.mjs";
import { createBatchPrompts } from "./batch_prompts.mjs";
import { createProjectSetup } from "./project_setup.mjs";
import { createBeatCalibration } from "./beat_calibration.mjs";
import { createLlmPopout } from "./llm_popout.mjs";
import { createTimelineState } from "./timeline_state.mjs";
import { createUiProfileActions } from "./ui_profiles.mjs";
import { createMiniMaxReferences } from "./minimax_references.mjs";
import { createIdLoraBuilder } from "./id_lora_builder.mjs";
import { createVideoRender } from "./video_render.mjs";
import { createBatchRender } from "./batch_render.mjs";
import { createModelPickers } from "./model_pickers.mjs";

export function openBuilder(node) {
  // Mutable runtime state shared across the builder features for this open builder.
  const wizardVideoSettings = { global: false };
  const builderETAState = { timer: 0, log: null };
  const updateStatus = { payload: null };
  const playStart = { inFlight: false, request: 0 };
  const settingsModalControls = { projectVideoEngineSelect: null };
  const beatCalibration = { draft: null };
  const previewVideoState = { loadTimer: null, syncPause: false, pendingSeekTarget: null };
  const timelinePromptSave = { saving: false };
  const silentTimeline = { playing: false, raf: 0, startedAt: 0, startTime: 0 };
  const projectBatch = { running: false, stopRequested: false };
  const storyboardPipeline = { runner: null };
  const browserAiSend = { lastStartedAt: 0 };
  const fullscreen = { enabled: false };
  const builderLifecycle = { keydownHandler: null, resourceTimer: 0, resourceController: null, resourceResizeObserver: null };
  const gemmaThenCreateVideoButtons = [];
  const editImagePromptButtons = [];
  const savedI2VPrompts = new WeakMap();
  const savedMiniMaxPrompts = new WeakMap();

  setBuilderAutomaticMemoryCleanupEnabled(false);
  console.log(`[VRGDG Music Builder] UI version ${BUILDER_UI_VERSION}`);
  const overlay = document.createElement("div");
  overlay.dataset.vrgdgThemeRoot = "true";
  overlay.style.cssText = `position:fixed;inset:0;z-index:100000;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;font-family:${BUILDER_FONT_STACK};`;
  const shell = document.createElement("div");
  const normalShellStyle = `
    width: min(1800px, calc(100vw - 24px));
    height: min(920px, calc(100vh - 24px));
    display: grid;
    grid-template-rows: auto minmax(0,1fr) minmax(230px, 34vh);
    background: #18181b;
    color: #fafafa;
    border: 1px solid #3f3f46;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 24px 90px rgba(0,0,0,.55);
  `;
  shell.style.cssText = normalShellStyle;

  const {
    refreshV10UpdateStatus, updateReleaseNotes, updateStatusAction, updateStatusBanner, updateWhatsNewAction,
  } = buildUpdateBanner({
    updateStatus,
  });

  const {
    audioInput, autoSaveControl, branchProjectButton, closeButton, exportProjectButton, fullscreenButton,
    importProjectButton, loadButton, loadLastProjectButton, loadSessionButton, loadSrtButton, menuButton,
    newProjectButton, pickAudioButton, pickSrtButton, projectInput, reviewGuideButton, saveButton,
    saveProjectAsButton, settingsButton, srtInput, topbar, uiProfileControls, uiProfileField, videoTypeField,
    videoTypeSelect, whatsNewMenuButton,
  } = buildProjectControls({
    node,
  });

  const closeBuilderNow = () => {
    pauseAllAudio();
    if (!previewVideo.paused) previewVideo.pause();
    clearTimeout(builderLifecycle.resourceTimer);
    clearInterval(builderETAState.timer);
    builderETAResizeObserver.disconnect();
    builderLifecycle.resourceController?.abort();
    builderLifecycle.resourceResizeObserver?.disconnect();
    window.removeEventListener("vrgdg:builder-toast", toastNotificationHandler);
    if (builderLifecycle.keydownHandler) document.removeEventListener("keydown", builderLifecycle.keydownHandler, true);
    restoreBrowserAiDownloadsQuietly().catch(() => null);
    overlay.remove();
  };
  const {
    autoBuildButton, autoLoadAllButton, builderAgentButton, builderETA, builderETAResizeObserver,
    builderFullETA, builderSceneETA, centerActions, clearMemoryButton, convertLtxPromptsToMiniMaxButton,
    downloadModelsButton, fluxReferenceBuilderButton, fullBuildButton, fullFLFBuildButton, gemmaRunnerButton,
    gemmaT2IAllButton, gemmaVideoAllButton, importActions, importImageFolderButton, importSceneNotesButton,
    lyricMapperButton, menuDropdown, pollBuilderResources, positionBuilderETA, positionBuilderResourceMonitor,
    projectBatchButton, projectVideoEngineBadge, promptCreatorButton, promptOptionsButton, renderAllButton,
    renderLogButton, sendToPromptCreatorButton, slideshowPreviewButton, stitchPreviewButton,
    stopWorkflowButton, storyboardBuilderButton, updateV10Button, updateV10HintButton, wizardBetaButton,
    wizardButton, zEnhanceAllButton, zEnhanceAllToolButton, zImageAllButton,
  } = buildTopbar({
    autoSaveControl, branchProjectButton, builderETAState, builderLifecycle, closeBuilderNow, closeButton,
    exportProjectButton, fullscreenButton, importProjectButton, loadLastProjectButton, loadSessionButton,
    menuButton, newProjectButton, overlay, reviewGuideButton, saveButton, saveProjectAsButton, settingsButton,
    topbar, uiProfileField, videoTypeField, whatsNewMenuButton,
    saveSession: (...args) => saveSession(...args),
  });

  const {
    beatCalibrationAnchors, beatCalibrationCancelButton, beatCalibrationCaptureButton,
    beatCalibrationFpsInput, beatCalibrationGridType, beatCalibrationGridTypeHint, beatCalibrationInstruction,
    beatCalibrationTimecodeGrid, beatCalibrationTimecodeHint, beatCalibrationTimecodeInput,
    beatCalibrationWizard, calibrateFirstBeatButton, leftTabBar, lutsTabButton, main, makeToolRow,
    projectBatchAddCurrent, projectBatchAddCustom, projectBatchAddRecent, projectBatchAddSession,
    projectBatchClearMemory, projectBatchContinueOnError, projectBatchForceVideos, projectBatchPanel,
    projectBatchQueue, projectBatchRun, projectBatchStatus, projectBatchStop, sceneListPane, scenesTabButton,
    segmentList, snapAllSceneStartsButton, toolsPane, toolsTabButton,
  } = buildLeftPanel({
    autoLoadAllButton, builderAgentButton, convertLtxPromptsToMiniMaxButton, importImageFolderButton,
    importSceneNotesButton, projectBatchButton, promptCreatorButton, sendToPromptCreatorButton,
    zEnhanceAllToolButton,
  });
  const lutsTools = createMusicVideoBuilderLuts({
    api,
    toast,
    getSelectedScene: () => activeSegment(),
    applyLutToScene: (lut, segment) => applyLutToSegment(lut, segment),
    updateScene: (segment) => {
      if (segment) setActiveSegment(segment);
    },
    refresh: () => render(),
    autoSave: autoSaveSessionQuiet,
  });
  const {
    filmGrainPane, fxOverlaysPane, fxOverlaysTab, fxPane, postProcessFxTab, postProcessGrainTab,
    postProcessLutsTab, postProcessPane,
  } = buildPostProcessPane({
    lutsTools,
  });
  leftTabBar.append(scenesTabButton, toolsTabButton, lutsTabButton);
  segmentList.append(leftTabBar, sceneListPane, toolsPane, postProcessPane);
  // A tab on the edge of the left panel hides it so the video window and timeline get the space, and shows it again.
  const leftPanelToggle = document.createElement("button");
  leftPanelToggle.type = "button";
  leftPanelToggle.style.cssText = "position:absolute;top:50%;transform:translateY(-50%);z-index:6;width:16px;height:64px;padding:0;border:1px solid #155e75;border-left:0;border-radius:0 8px 8px 0;background:#083344;color:#cffafe;font-size:11px;font-weight:900;cursor:pointer;display:flex;align-items:center;justify-content:center;";
  // Starting text and spots. applyLayoutSizes moves them, but a browser holding an older timeline_state.mjs would
  // otherwise leave both tabs blank and stacked at the left edge.
  leftPanelToggle.textContent = "◀";
  leftPanelToggle.style.left = "267px";
  const rightPanelToggle = document.createElement("button");
  rightPanelToggle.type = "button";
  rightPanelToggle.textContent = "▶";
  rightPanelToggle.style.cssText = "position:absolute;top:50%;transform:translateY(-50%);z-index:6;width:16px;height:64px;padding:0;border:1px solid #155e75;border-right:0;border-radius:8px 0 0 8px;background:#083344;color:#cffafe;font-size:11px;font-weight:900;cursor:pointer;display:flex;align-items:center;justify-content:center;";
  rightPanelToggle.style.right = "360px";
  const leftResizeHandle = document.createElement("div");
  leftResizeHandle.title = "Drag to resize scene list";
  leftResizeHandle.style.cssText = "cursor:col-resize;background:#18181b;border-left:1px solid #27272a;border-right:1px solid #27272a;";
  leftResizeHandle.style.gridColumn = "2";
  const preview = document.createElement("div");
  preview.style.cssText = "display:grid;grid-template-rows:minmax(0,1fr) auto;min-height:0;background:#09090b;";
  preview.style.gridColumn = "3";
  const previewStage = document.createElement("div");
  previewStage.style.cssText = "position:relative;width:100%;height:100%;display:flex;align-items:center;justify-content:center;min-height:0;overflow:hidden;";
  const previewEmpty = document.createElement("div");
  previewEmpty.textContent = "Create a segment, add a T2I prompt, then preview ZImage.";
  previewEmpty.style.cssText = "color:#71717a;font-size:13px;text-align:center;";
  const previewImage = document.createElement("img");
  previewImage.alt = "";
  previewImage.style.cssText = "display:none;max-width:100%;max-height:100%;object-fit:contain;background:#050505;";
  const postProcessComparePreview = createPostProcessComparePreview({ makeImageUrl: makeEditorImageUrl });
  const previewVideo = document.createElement("video");
  previewVideo.controls = true;
  previewVideo.playsInline = true;
  previewVideo.muted = false;
  previewVideo.style.cssText = "display:none;max-width:100%;max-height:100%;object-fit:contain;background:#050505;";
  const previewDecodeHint = document.createElement("div");
  previewDecodeHint.style.cssText = "display:none;position:absolute;left:14px;right:14px;bottom:14px;z-index:2;padding:10px 12px;border:1px solid #ef4444;border-radius:6px;background:rgba(15,23,42,.92);color:#fee2e2;font-size:12px;line-height:1.35;box-shadow:0 12px 36px rgba(0,0,0,.45);";
  previewVideo.addEventListener("error", () => {
    handlePreviewVideoLoadIssue("error");
  });
  previewVideo.addEventListener("loadedmetadata", () => {
    if (previewVideoState.loadTimer) clearTimeout(previewVideoState.loadTimer);
    previewVideoState.loadTimer = null;
    previewDecodeHint.style.display = "none";
    previewVideo.dataset.failureKey = "";
  });
  previewVideo.addEventListener("pause", () => {
    // The native video control pauses only the preview element. If the
    // timeline audio remains active, syncPreviewPlayback() immediately calls
    // play() again on its next timeupdate. Treat a user pause as a request to
    // pause the master timeline as well. Programmatic sync pauses are marked
    // so they do not recursively stop timeline playback.
    if (previewVideoState.syncPause || previewVideo.ended || !isTimelinePlaying()) return;
    pauseAllAudio();
    updateAudioScrubbers();
  });
  previewVideo.addEventListener("seeked", () => {
    // A newer scrub position arrived while this seek was still decoding.
    // Jump straight to it instead of the queue of stale positions in between.
    if (previewVideoState.pendingSeekTarget == null) return;
    const target = previewVideoState.pendingSeekTarget;
    previewVideoState.pendingSeekTarget = null;
    try {
      previewVideo.currentTime = target;
    } catch {
      // Ignore; the next scrub/timeupdate tick will retry if still needed.
    }
  });
  // Hidden element used only to warm the browser's cache for the next scene's
  // clip a moment before the playhead reaches it, so the visible swap at the
  // cut doesn't stall on a cold fetch from the local server. It never plays.
  const preloadVideo = document.createElement("video");
  preloadVideo.muted = true;
  preloadVideo.preload = "auto";
  preloadVideo.style.cssText = "display:none;width:0;height:0;";
  const renderStatusButton = document.createElement("button");
  renderStatusButton.type = "button";
  renderStatusButton.textContent = "Render status";
  renderStatusButton.title = "Show the render status window again after closing it";
  renderStatusButton.style.cssText = "position:absolute;left:8px;top:8px;z-index:3;padding:3px 8px;border:1px solid #155e75;border-radius:5px;background:rgba(8,51,68,.82);color:#cffafe;font-size:11px;font-weight:700;cursor:pointer;opacity:.75;";
  renderStatusButton.onmouseenter = () => { renderStatusButton.style.opacity = "1"; };
  renderStatusButton.onmouseleave = () => { renderStatusButton.style.opacity = ".75"; };
  renderStatusButton.onclick = () => {
    if (!showLastProgressWindow()) toast("No render status window is open. Start a render to see one.");
  };
  previewStage.append(previewEmpty, previewImage, previewVideo, preloadVideo, postProcessComparePreview.element, previewDecodeHint, renderStatusButton);
  const customImageFileInput = document.createElement("input");
  customImageFileInput.type = "file";
  customImageFileInput.accept = "image/png,image/jpeg,image/webp";
  customImageFileInput.style.display = "none";
  shell.append(customImageFileInput);
  const faceFixTool = createFaceFixTool({
    toast,
    getVideoPath: () => String(previewVideo.dataset.path || selectedSegmentVideoPath(activeSegment()) || ""),
    getProjectFolder: () => String(projectInput.value || ""),
    getPlayheadContext: () => {
      const previewSegmentId = String(previewVideo.dataset.segmentId || "");
      const selected = activeSegment();
      const previewMatchesSelection = Boolean(selected && previewSegmentId && String(selected.id || "") === previewSegmentId);
      const segment = selected || allEditableSegments().find((item) => String(item?.id || "") === previewSegmentId);
      return {
        time: previewMatchesSelection ? Number(previewVideo.currentTime || 0) : 0,
        videoPath: String(selectedSegmentVideoPath(segment) || (previewMatchesSelection ? previewVideo.dataset.path : "") || ""),
        segmentId: String(segment?.id || ""),
        sceneLabel: segment ? sceneDisplayName(segment, segmentIndexInfo(segment).index) : "scene",
      };
    },
    captureCurrentFrame: () => captureVideoFrameDataUrl(previewVideo),
    generateFacePrompt: async ({ referenceImage }) => {
      const modelFile = referenceDescriptionVisionModel();
      const mmprojFile = referenceDescriptionMmproj();
      if (!["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner) && (!modelFile || !mmprojFile)) {
        throw new Error("Choose a vision model and Vision mmproj in LLM Runner first.");
      }
      const data = await postJson("/vrgdg/music_builder/describe_reference_image", {
        ...textGemmaRunnerPayload(),
        model_file: modelFile,
        mmproj_file: mmprojFile,
        reference_type: "face",
        image_data: referenceImage,
        unload_after: true,
        clear_before_load: false,
        temperature: 0.15,
        top_p: 0.85,
        max_new_tokens: 180,
      }, 4 * 60 * 1000);
      const description = String(data.description || "").trim();
      if (!description) throw new Error("The selected LLM Runner returned an empty face description.");
      return description;
    },
    calculateAnchors: async (payload) => postJson("/vrgdg/face_fix/estimate_anchors", payload, 120000),
    startJob: async (payload, mode, onStatus) => {
      onStatus?.(mode === "frame" ? "Detecting and preparing the playhead face..." : "Extracting frames and tracking the face across the selected range...");
      const prepared = await postJson("/vrgdg/face_fix/prepare", { ...payload, mode }, 30 * 60 * 1000);
      const anchors = Array.isArray(prepared.anchors) ? prepared.anchors : [];
      if (!anchors.length) throw new Error("Face Fix prepared no anchors for enhancement.");
      const baseSettings = saveZEnhanceSettingsFromPanel();
      const settings = {
        ...baseSettings,
        width: 512,
        height: 512,
        seed: Number.isFinite(Number(payload.seed)) ? Number(payload.seed) : 42,
        seed_mode: "fixed",
        enhance_amount: Number(payload.enhance_amount || 8),
      };
      let enhancedCount = 0;
      let firstEnhancedPreview = "";
      for (let index = 0; index < anchors.length; index += 1) {
        const anchor = anchors[index];
        const label = `Face Fix anchor ${index + 1}/${anchors.length}`;
        let cleanupFinished = false;
        try {
          onStatus?.(`${label}: building hidden Z-Enhance workflow...\nVideo index ${anchor.index}, source frame ${anchor.frame_number}`);
          const enhancePayload = zEnhancePayloadFromSettings(
            settings,
            payload.prompt,
            { path: anchor.source_path, name: `face_fix_anchor_${String(anchor.index).padStart(6, "0")}.png` },
          );
          const built = await postJson("/vrgdg/workflow_runner/build_z_upscale_enhance_prompt", enhancePayload, 120000);
          onStatus?.(`${label}: queueing 512×512 face enhancement...`);
          const queued = await queueWorkflowPrompt(built.prompt, {
            onStatus: (message) => onStatus?.(`${label}: ${message}`),
          });
          const promptId = queued?.prompt_id;
          if (!promptId) throw new Error(`${label}: ComfyUI did not return a prompt ID.`);
          const images = await waitForImages(promptId, (message) => {
            onStatus?.(`${label}: ${message}\nPrompt ID: ${promptId}`);
          });
          const image = images[images.length - 1];
          if (!image) throw new Error(`${label}: Z-Enhance returned no image.`);
          const accepted = await postJson("/vrgdg/face_fix/accept_enhanced_anchor", {
            manifest_path: prepared.manifest_path,
            run_index: anchor.run_index,
            order: anchor.order,
            image,
          }, 120000);
          if (!firstEnhancedPreview) {
            firstEnhancedPreview = accepted?.enhanced_preview_data || makeImageViewUrl(image);
          }
          enhancedCount += 1;
          onStatus?.(`${label}: enhanced face saved.\nClearing RAM/VRAM before the next crop...`);
          await runImageMemoryCleanupQuiet({ set: (message) => onStatus?.(`${label}: ${message}`) }, label, 95);
          cleanupFinished = true;
        } finally {
          if (!cleanupFinished) {
            onStatus?.(`${label}: cleaning memory after an interrupted or failed enhancement...`);
            await runImageMemoryCleanupQuiet({ set: (message) => onStatus?.(`${label}: ${message}`) }, `${label} recovery`, 100).catch(() => "");
          }
        }
      }
      onStatus?.(`Enhanced ${enhancedCount}/${anchors.length} anchors. Building the hidden LTX 2.3 face-video workflow...`);
      const runs = Array.isArray(prepared.runs) ? prepared.runs : [];
      let totalLtxFrames = 0;
      let firstLtxPreview = "";
      for (let runIndex = 0; runIndex < runs.length; runIndex += 1) {
        const run = runs[runIndex];
        const runLabel = `LTX face run ${runIndex + 1}/${runs.length}`;
        let ltxCleanupFinished = false;
        try {
          const builtLtx = await postJson("/vrgdg/face_fix/build_ltx_prompt", {
            manifest_path: prepared.manifest_path,
            run_index: run.run_index,
          }, 120000);
          onStatus?.(`${runLabel}: queueing...\nFrames: ${builtLtx.frame_count}\nAnchors: ${builtLtx.anchor_indices_text}`);
          const queuedLtx = await queueWorkflowPrompt(builtLtx.prompt, {
            onStatus: (message) => onStatus?.(`${runLabel}: ${message}`),
          });
          const ltxPromptId = queuedLtx?.prompt_id;
          if (!ltxPromptId) throw new Error(`${runLabel}: ComfyUI did not return a prompt ID.`);
          const ltxImages = await waitForImages(ltxPromptId, (message) => {
            onStatus?.(`${runLabel}: ${message}\nPrompt ID: ${ltxPromptId}`);
          });
          onStatus?.(`${runLabel}: returned ${ltxImages.length} frame(s). Validating the Preview Image batch...`);
          const acceptedLtx = await postJson("/vrgdg/face_fix/accept_ltx_frames", {
            manifest_path: prepared.manifest_path,
            run_index: run.run_index,
            images: ltxImages,
          }, 10 * 60 * 1000);
          totalLtxFrames += Number(acceptedLtx.ltx_frame_count || 0);
          if (!firstLtxPreview) firstLtxPreview = acceptedLtx.ltx_preview_data || "";
          onStatus?.(`${runLabel}: validated ${acceptedLtx.ltx_frame_count}/${acceptedLtx.frame_count} frames.\nClearing RAM/VRAM...`);
          await runImageMemoryCleanupQuiet({ set: (message) => onStatus?.(`${runLabel} cleanup: ${message}`) }, runLabel, 98);
          ltxCleanupFinished = true;
        } finally {
          if (!ltxCleanupFinished) {
            onStatus?.(`${runLabel}: cleaning memory after an interrupted or failed pass...`);
            await runImageMemoryCleanupQuiet({ set: (message) => onStatus?.(`${runLabel} recovery: ${message}`) }, `${runLabel} recovery`, 100).catch(() => "");
          }
        }
      }
      let finalized = {};
      if (mode === "range") {
        onStatus?.(`All ${totalLtxFrames} LTX face frames are validated.\nSafely feathering visible faces into the source frames and rebuilding the scene...`);
        finalized = await postJson("/vrgdg/face_fix/finalize", {
          manifest_path: prepared.manifest_path,
          feather: payload.feather,
          color_match: payload.color_match,
        }, 30 * 60 * 1000);
        const segment = allEditableSegments().find((item) => String(item?.id || "") === String(payload.segment_id || ""));
        if (segment && finalized?.output_video_path) {
          pushHistory();
          addSegmentVideoHistoryPath(segment, finalized.output_video_path);
          segment.preview_mode = "video";
          segment.video_status = "done";
          setActiveSegment(segment);
          syncPreview(segment);
          render();
          await autoSaveSessionQuiet("LTX Face Fix repaired scene video");
        }
      }
      return {
        ...prepared,
        ...finalized,
        execution_stage: mode === "range" ? "complete" : "ltx_frames_ready",
        enhanced_count: enhancedCount,
        ltx_frame_count: totalLtxFrames,
        enhanced_preview_data: firstLtxPreview || firstEnhancedPreview,
      };
    },
  });
  toolsPane.append(makeToolRow(faceFixTool.button, "Repair blurry distant faces with Z-Enhanced anchors and temporally consistent LTX face-video processing."));
  const {
    editI2VMotionJsonButton, editPromptJsonButton, editStoryIdeaButton, editSubjectSceneButton,
    editThemeStyleButton, ernieImageTriggerInput, fluxImageTriggerInput, freezeTimingControl,
    i2iImageFileInput, i2vMotionJsonInput, imageFolderFileInput, imageTriggerInput, importI2VMotionJsonButton,
    importPromptJsonButton, inspector, krea2TwoPassImageTriggerInput, labelInput, loadVrgdgContextButton,
    projectAudioFileInput, projectSrtFileInput, promptJsonInput, rightResizeHandle, storyIdeaInput,
    subjectSceneInput, themeStyleInput, useSceneErnieImageSettings, useSceneFluxKleinSettings,
    useSceneI2VVideoSettings, useSceneI2VVideoSettingsNote, useSceneKrea2TwoPassSettings,
    useSceneNBImageSettings, useVrgdgTextContext, videoSettingsScopeNote, videoTriggerInput,
    visionRefFileInput,
  } = buildInspectorInputs({
    leftResizeHandle, main, preview, segmentList, shell,
  });
  const {
    browserAiAddExtrasButton, browserAiAddGroupImagesButton, browserAiAddLocationsButton,
    browserAiAddMembersButton, browserAiAddSingerButton, browserAiAutoAdvanceGroup, browserAiBandSequenceMode,
    browserAiBandSequencePanel, browserAiChooseLocationButton, browserAiClearExtrasButton,
    browserAiClearGroupImagesButton, browserAiClearLocationButton, browserAiClearLocationsButton,
    browserAiClearMembersButton, browserAiClearSingerButton, browserAiCustomGroupsPanel,
    browserAiDeleteGroupButton, browserAiDownloadOverrideProviders, browserAiDuplicateGroupButton,
    browserAiExtrasDrop, browserAiExtrasList, browserAiFinishButton, browserAiGroupDrop, browserAiGroupList,
    browserAiGroupPrompt, browserAiGroupSelect, browserAiGroupsNote, browserAiGroupStatus,
    browserAiLocationDrop, browserAiLocationList, browserAiLocationsDrop, browserAiLocationsList,
    browserAiMembersDrop, browserAiMembersList, browserAiNewGroupButton, browserAiRenameGroupButton,
    browserAiSendButton, browserAiSequenceLocationSelect, browserAiSequenceProgress,
    browserAiSequenceSetSelect, browserAiSessionActions, browserAiSingerDrop, browserAiSingerList,
    createFluxPromptButton, createNBPromptButton, editFlowGptT2IInstructionsButton,
    editFluxKleinT2IInstructionsButton, editFluxPromptButton, editNanoBT2IInstructionsButton,
    editNBPromptButton, ernieBatchSize, ernieClipPicker, ernieGrid, ernieHeight, ernieI2IDrop,
    ernieI2ILoadButton, ernieI2IPanel, ernieI2IPath, ernieI2ISlider, ernieI2IStartStep, ernieImagePanel,
    ernieLoraCount, ernieLoraPanel, ernieLoraRows, ernieLoraSlots, ernieSeed, ernieSeedMode, ernieUnetPicker,
    ernieUseImageToImage, ernieUseLora, ernieVaePicker, ernieWidth, flowGptAskPreviousImage,
    flowGptAspectRatio, flowGptAspectRatioField, flowGptCreateImageButton, flowGptCreatePromptButton,
    flowGptFailureMode, flowGptLoginButton, flowGptManualActions, flowGptManualAutoAdvance,
    flowGptManualChatPrompt, flowGptManualExportRefsButton, flowGptManualImportLatestButton,
    flowGptManualMode, flowGptManualOpenButton, flowGptManualStatus, flowGptModePanel, flowGptPrompt,
    flowGptProviderRow, flowGptRetries, flowGptSetupActions, flowGptSetupButton, flowGptSetupNote,
    flowGptStatusButton, flowGptStatusText, flowGptTimeout, flowNanoProviderButton, fluxClipPicker,
    fluxGemmaModelSelect, fluxGlobalIngredientButton, fluxGlobalIngredientClearButton,
    fluxGlobalIngredientDrop, fluxGlobalIngredientFileInput, fluxGlobalIngredientList,
    fluxGlobalIngredientPanel, fluxGrid, fluxHeight, fluxImageRefsPanel, fluxIngredientButton,
    fluxIngredientClearButton, fluxIngredientDrop, fluxIngredientFileInput, fluxIngredientList,
    fluxKleinPanel, fluxLoraCount, fluxLoraPanel, fluxLoraRows, fluxLoraSlots, fluxMmprojSelect, fluxNotes,
    fluxPrompt, fluxSeed, fluxUnetPicker, fluxUseDirectorNotes, fluxUseLora, fluxUseTextOnlyGemmaPrompt,
    fluxVaePicker, fluxWidth, gptImageProviderButton, krea2TwoPassAspectRatio, krea2TwoPassBatchSize,
    krea2TwoPassCfg, krea2TwoPassClipPicker, krea2TwoPassCreativity, krea2TwoPassCreativityInput,
    krea2TwoPassI2IDrop, krea2TwoPassI2ILoadButton, krea2TwoPassI2IPanel, krea2TwoPassI2IPath,
    krea2TwoPassLoraCount, krea2TwoPassLoraPanel, krea2TwoPassLoraRows, krea2TwoPassLoraSlots,
    krea2TwoPassPanel, krea2TwoPassSampler, krea2TwoPassSeed, krea2TwoPassSeedMode, krea2TwoPassSettingsGrid,
    krea2TwoPassUnetPicker, krea2TwoPassUseImageToImage, krea2TwoPassUseLora, krea2TwoPassVaePicker,
    metaImageProviderButton, nbApiKey, nbGemmaModelSelect, nbGlobalIngredientButton,
    nbGlobalIngredientClearButton, nbGlobalIngredientDrop, nbGlobalIngredientList, nbGlobalIngredientPanel,
    nbImagePanel, nbIngredientActions, nbIngredientButton, nbIngredientClearButton, nbIngredientDrop,
    nbIngredientList, nbMmprojSelect, nbModelSelect, nbNotes, nbPrompt, nbUseDirectorNotes,
    nbUseGlobalIngredients, nbUseTextOnlyGemmaPrompt, previewFluxButton, previewNBButton,
    sendFluxPromptToEnhanceButton, sendNBPromptToEnhanceButton, useFluxGlobalIngredients, useFluxKlein,
    useSceneZImageSettings, zBatchSize, zClipPicker, zEnhanceAmount, zEnhanceAmountValue, zEnhanceButton,
    zEnhanceClipPicker, zEnhanceGemmaButton, zEnhanceGemmaModelSelect, zEnhanceGemmaNotes, zEnhanceGrid,
    zEnhanceHeight, zEnhanceHint, zEnhanceLoraCount, zEnhanceLoraPanel, zEnhanceLoraRows, zEnhanceLoraSlots,
    zEnhanceMmprojSelect, zEnhancePanel, zEnhancePromptPreview, zEnhanceSeed, zEnhanceSeedMode, zEnhanceTitle,
    zEnhanceUnetPicker, zEnhanceUseLora, zEnhanceVaePicker, zEnhanceWidth, zFirstGrid, zFirstHeight,
    zFirstTitle, zFirstWidth, zI2IDrop, zI2ILoadButton, zI2IPanel, zI2IPath, zI2ISlider, zI2IStartStep,
    zimageSettingsPanel, zLoraCount, zLoraPanel, zLoraRows, zLoraSlots, zSecondGrid, zSecondHeight,
    zSecondTitle, zSecondWidth, zSeed, zSeedGrid, zSeedMode, zUnetPicker, zUseImageToImage, zUseLora,
    zVaePicker,
  } = buildImageSettingsPanels({
    makeEditImagePromptButton, shell,
  });
  const {
    createI2VButton, createT2IButton, editErnieT2IInstructionsButton, editErnieT2IPromptButton,
    editI2VInstructionsButton, editI2VPromptButton, editIdLoraInstructionsButton,
    editIngredientsInstructionsButton, editKrea2T2IInstructionsButton, editKrea2TwoPassT2IPromptButton,
    editRTVInstructionsButton, editT2IPromptButton, editT2VInstructionsButton,
    editZImageT2IInstructionsButton, endInput, ernieCreateT2IButton, ernieGemmaModelSelect, ernieImageCard,
    ernieImageModePanel, ernieMmprojSelect, ernieNotesInput, ernieRefImageDrop, ernieRefImageLoadButton,
    ernieRefImagePanel, ernieSendT2IPromptToEnhanceButton, ernieT2IPrompt, ernieTextGemmaModelSelect,
    ernieUseVisionReference, flfTransitionTypeField, flfTransitionTypeSelect, flowGptCard, fluxKleinCard,
    fluxKleinModePanel, gemmaModelSelect, i2vGemmaModelSelect, i2vMmprojSelect, i2vNotesInput, i2vPrompt,
    i2vPromptEnhancementNote, i2vReferenceNote, i2vTextGemmaModelSelect, imageModelChooserWrap,
    krea2TwoPassCard, krea2TwoPassCreateT2IButton, krea2TwoPassGemmaModelSelect, krea2TwoPassMmprojSelect,
    krea2TwoPassModePanel, krea2TwoPassNotesInput, krea2TwoPassRefImageDrop, krea2TwoPassRefImageLoadButton,
    krea2TwoPassRefImagePanel, krea2TwoPassSendT2IPromptToEnhanceButton, krea2TwoPassT2IPrompt,
    krea2TwoPassTextGemmaModelSelect, krea2TwoPassUseVisionReference, loadCustomImageButton,
    lyricSingersInput, lyricTextInput, miniMaxGemmaModelSelect, miniMaxMmprojSelect,
    miniMaxTextGemmaModelSelect, mmprojSelect, nbImageCard, notesInput, refImageDrop, refImageInput,
    refImageLoadButton, refImagePanel, saveI2VPromptButton, sendT2IPromptToEnhanceButton, startInput,
    syncMiniMaxLlmSelectsFromShared, t2iPrompt, t2iTextGemmaModelSelect, t2vLocationNote, t2vReferenceNote,
    t2vRefImageDrop, t2vRefImageLoadButton, t2vRefImagePanel, useI2VPromptEnhancementPass,
    useI2VVisionReference, useT2VVisionReference, useVisionReference, zEnhanceCard, zImageCard,
    zImageModePanel,
  } = buildImageModeCards({
    makeEditImagePromptButton, zI2IDrop,
    syncKrea2TwoPassLlmSelectsFromShared: (...args) => syncKrea2TwoPassLlmSelectsFromShared(...args),
    syncKrea2TwoPassLlmSelectsToShared: (...args) => syncKrea2TwoPassLlmSelectsToShared(...args),
    updateI2VPromptSaveButtonState: (...args) => updateI2VPromptSaveButtonState(...args),
  });
  const {
    clearSceneEndFrameButton, createSceneEndFrameButton, createSceneVideoActions, createSceneVideoButton,
    createSceneVideoButtons, ernieCreateButton, ernieCreateButtons, finalizeSceneFLFPromptButton,
    firstLastFramePreviewGrid, firstLastFramePreviewPanel, firstLastFrameStatus, firstLastFrameVideoCard,
    flfChainedSettingsPanel, flfChainPreviousEndFrame, flfColorMatchFadeInput, flfColorMatchStrengthInput,
    flfCustomEndDirection, flfEndpointModeSelect, flfEndpointPromptPreview, flfFirstAttentionStrength,
    flfFirstGuideBlurInput, flfFirstGuideCrfInput, flfFirstGuideCrop, flfFirstGuideFrameIndexInput,
    flfFirstGuideInterpolation, flfFirstGuideStrengthInput, flfGemmaContextModeSelect,
    flfGlobalTransitionTypeSelect, flfGuideSettingsSection, flfLastAttentionStrength, flfLastGuideBlurInput,
    flfLastGuideCrfInput, flfLastGuideCrop, flfLastGuideFrameIndexInput, flfLastGuideInterpolation,
    flfLastGuideStrengthInput, flfMatchPreviousClipColor, flfMotionPlanPreview, flfPerScenePlanner,
    flfPreGeneratePromptsFromSceneImages, flfRenderChainSourceSelect, flfRestoreWorkflowDefaultsButton,
    flfStructureModeSelect, flfTransitionLoraNote, fluxCreateButtons, i2vAdvancedNodeSettingsPanel,
    i2vAdvancedNodeSettingsSection, i2vAudioVaePicker, i2vClip1Picker, i2vClip2Picker,
    i2vDiffusionLoaderAdvanced, i2vDiffusionModelField, i2vDiffusionModelPicker, i2vEnableFp16Accumulation,
    i2vFpsInput, i2vHeightInput, i2vLoraCount, i2vLoraHintButton, i2vLoraPanel, i2vLoraRows, i2vLoraSlots,
    i2vPass1Bypass, i2vPass1NodePanel, i2vPass1SamplerSelect, i2vPass1SigmasInput, i2vPass1StrengthInput,
    i2vPass1StrengthSlider, i2vPass2Bypass, i2vPass2NodePanel, i2vPass2SamplerSelect, i2vPass2SigmasInput,
    i2vPass2StrengthInput, i2vPass2StrengthSlider, i2vPreFramesInput, i2vSeedInput, i2vSettingsGrid,
    i2vTailLossFramesInput, i2vUnetModelField, i2vUnetPicker, i2vUpscalePicker, i2vUseGgufModel, i2vUseLora,
    i2vUseSageAttention, i2vVaePicker, i2vWarmCooldownSection, i2vWidthInput, idLoraIdentityGrid,
    idLoraIdentityScaleInput, idLoraReferenceAudioField, idLoraReferenceAudioInput, idLoraReferenceAudioNote,
    idLoraVideoCard, imageToVideoCard, importCustomVideoCard, importCustomVideoPanel, ingredientsToVideoCard,
    krea2TwoPassCreateButton, krea2TwoPassCreateButtons, loadSceneEndFrameButton, ltx25AspectRatioSelect,
    ltx25MegapixelsInput, ltx25ResolutionGrid, ltxIdLoraFirstPassStrength, ltxIdLoraPicker,
    ltxIdLoraRequiredPanel, ltxIdLoraSecondPassStrength, ltxIngredientsFirstPassStrength,
    ltxIngredientsLoraPicker, ltxIngredientsRequiredPanel, ltxIngredientsResolutionWarning,
    ltxMsrBackgroundMode, ltxMsrFirstPassStrength, ltxMsrLoraPicker, ltxMsrReferenceStrength,
    ltxMsrRequiredPanel, ltxMsrSecondPassStrength, miniMaxReferenceButtons, miniMaxReferencesButton,
    miniMaxSceneVideoButton, miniMaxSceneVideoButtons, miniMaxVideoReferencesButton, nbCreateButtons,
    pickIdLoraReferenceAudioButton, planSceneEndMotionButton, previewButton, referenceToVideoCard,
    rtvReferenceBehaviorField, rtvReferenceBehaviorNote, rtvReferenceBehaviorSelect,
    rtvSceneImageAnchorSection, sceneEndFrameFileInput, textToVideoCard, videoModeChooser, zCreateButtons,
  } = buildVideoSettingsPanels({
    previewFluxButton, previewNBButton, wrapCreateSceneVideoActions,
  });
  const {
    audioSummary, chooseGlobalAudioButton, createSilentTimelineAudioButton, globalAudioDrop, globalAudioGuide,
    globalAudioModeSelect, globalAudioSummary, inspectorActions, openSceneAudioOptionsButton,
    silentAudioDurationInput, silentAudioPanel, timingGrid,
  } = buildInspectorControls({
    endInput, previewButton, startInput,
  });
  syncGlobalAudioModeControls();
  const {
    audioPanel, audioTabButton, imagePanel, imageTabButton, inspectorTabs, noSceneNotice, sceneAdjustPanel,
    sceneDetailsPanel, scenePanel, sceneTabButton, sceneToolsPanel, videoPanel, videoTabButton,
  } = buildInspectorTabs();
  sceneTabButton.onclick = () => setInspectorTab("scene");
  imageTabButton.onclick = () => setInspectorTab("image");
  videoTabButton.onclick = () => setInspectorTab("video");
  audioTabButton.onclick = () => setInspectorTab("audio");
  const { idLoraVoiceSettingsSection, imageContinuityEnabled, imageContinuityStrength } = buildInspectorPanels({
    browserAiAutoAdvanceGroup, browserAiBandSequenceMode, browserAiBandSequencePanel,
    browserAiCustomGroupsPanel, browserAiGroupPrompt, browserAiGroupsNote, browserAiGroupStatus,
    browserAiSessionActions, createFluxPromptButton, createNBPromptButton, createT2IButton,
    editErnieT2IInstructionsButton, editErnieT2IPromptButton, editFlowGptT2IInstructionsButton,
    editFluxKleinT2IInstructionsButton, editFluxPromptButton, editI2VMotionJsonButton,
    editKrea2T2IInstructionsButton, editKrea2TwoPassT2IPromptButton, editNanoBT2IInstructionsButton,
    editNBPromptButton, editPromptJsonButton, editStoryIdeaButton, editSubjectSceneButton,
    editT2IPromptButton, editThemeStyleButton, editZImageT2IInstructionsButton, ernieBatchSize,
    ernieClipPicker, ernieCreateButton, ernieCreateT2IButton, ernieGemmaModelSelect, ernieGrid, ernieI2IPanel,
    ernieImageModePanel, ernieImagePanel, ernieImageTriggerInput, ernieLoraPanel, ernieMmprojSelect,
    ernieNotesInput, ernieRefImagePanel, ernieSendT2IPromptToEnhanceButton, ernieT2IPrompt,
    ernieTextGemmaModelSelect, ernieUnetPicker, ernieUseImageToImage, ernieUseLora, ernieUseVisionReference,
    ernieVaePicker, flowGptAskPreviousImage, flowGptAspectRatioField, flowGptCreateImageButton,
    flowGptCreatePromptButton, flowGptFailureMode, flowGptManualActions, flowGptManualAutoAdvance,
    flowGptManualChatPrompt, flowGptManualMode, flowGptManualStatus, flowGptModePanel, flowGptPrompt,
    flowGptProviderRow, flowGptRetries, flowGptSetupActions, flowGptSetupNote, flowGptStatusText,
    flowGptTimeout, fluxClipPicker, fluxGemmaModelSelect, fluxGrid, fluxImageRefsPanel, fluxImageTriggerInput,
    fluxKleinModePanel, fluxKleinPanel, fluxLoraPanel, fluxMmprojSelect, fluxNotes, fluxPrompt,
    fluxUnetPicker, fluxUseDirectorNotes, fluxUseLora, fluxUseTextOnlyGemmaPrompt, fluxVaePicker,
    freezeTimingControl, gemmaModelSelect, i2vMotionJsonInput, idLoraIdentityGrid, idLoraReferenceAudioField,
    idLoraReferenceAudioNote, imageModelChooserWrap, imagePanel, imageTriggerInput, importI2VMotionJsonButton,
    importPromptJsonButton, inspectorActions, krea2TwoPassClipPicker, krea2TwoPassCreateButton,
    krea2TwoPassCreateT2IButton, krea2TwoPassGemmaModelSelect, krea2TwoPassI2IPanel,
    krea2TwoPassImageTriggerInput, krea2TwoPassLoraPanel, krea2TwoPassMmprojSelect, krea2TwoPassModePanel,
    krea2TwoPassNotesInput, krea2TwoPassPanel, krea2TwoPassRefImagePanel,
    krea2TwoPassSendT2IPromptToEnhanceButton, krea2TwoPassSettingsGrid, krea2TwoPassT2IPrompt,
    krea2TwoPassTextGemmaModelSelect, krea2TwoPassUnetPicker, krea2TwoPassUseImageToImage,
    krea2TwoPassUseLora, krea2TwoPassUseVisionReference, krea2TwoPassVaePicker, labelInput,
    loadVrgdgContextButton, makeErnieCreateButton, makeFluxCreateButton, makeKrea2TwoPassCreateButton,
    makeNBCreateButton, makeZCreateButton, mmprojSelect, nbApiKey, nbGemmaModelSelect,
    nbGlobalIngredientPanel, nbImagePanel, nbIngredientActions, nbIngredientDrop, nbIngredientList,
    nbMmprojSelect, nbModelSelect, nbNotes, nbPrompt, nbUseDirectorNotes, nbUseGlobalIngredients,
    nbUseTextOnlyGemmaPrompt, notesInput, previewFluxButton, previewNBButton, promptJsonInput, refImagePanel,
    sceneAdjustPanel, sceneDetailsPanel, scenePanel, sceneToolsPanel, sendFluxPromptToEnhanceButton,
    sendNBPromptToEnhanceButton, sendT2IPromptToEnhanceButton, storyIdeaInput, subjectSceneInput, t2iPrompt,
    t2iTextGemmaModelSelect, themeStyleInput, timingGrid, useSceneErnieImageSettings,
    useSceneFluxKleinSettings, useSceneKrea2TwoPassSettings, useSceneNBImageSettings, useSceneZImageSettings,
    useVisionReference, useVrgdgTextContext, zBatchSize, zClipPicker, zEnhanceAmount, zEnhanceAmountValue,
    zEnhanceButton, zEnhanceClipPicker, zEnhanceGemmaButton, zEnhanceGemmaModelSelect, zEnhanceGemmaNotes,
    zEnhanceGrid, zEnhanceHint, zEnhanceLoraPanel, zEnhanceMmprojSelect, zEnhancePanel, zEnhancePromptPreview,
    zEnhanceTitle, zEnhanceUnetPicker, zEnhanceUseLora, zEnhanceVaePicker, zFirstGrid, zFirstTitle, zI2IPanel,
    zImageModePanel, zimageSettingsPanel, zLoraPanel, zSecondGrid, zSecondTitle, zSeedGrid, zUnetPicker,
    zUseImageToImage, zUseLora, zVaePicker,
  });

  const {
    advancedTwoPassControls, miniMaxAccelerationControls, miniMaxAddSpeakerCueButton,
    miniMaxAdvancedLatentUpscalerPicker, miniMaxAdvancedSettings, miniMaxAdvancedVramPreset, miniMaxAspectRatio, miniMaxAudioMode,
    miniMaxAudioNote, miniMaxAudioVaePicker, miniMaxAutoTimeAllScenesButton, miniMaxAutoTimeBeforePrompt,
    miniMaxClipPicker, miniMaxContinuityMode, miniMaxContinuityNote, miniMaxContinuityPromptFromLastFrame,
    miniMaxCooldownFrames, miniMaxCreatePromptButton, miniMaxDenoise, miniMaxDiffusionModelPicker,
    miniMaxEasyCacheBypass, miniMaxEasyCacheEndPercent, miniMaxEasyCacheReuseThreshold,
    miniMaxEasyCacheSettings, miniMaxEasyCacheStartPercent, miniMaxEasyCacheVerbose,
    miniMaxEditContinuityPromptInstructionsButton, miniMaxEditInstructionsButton, miniMaxEnginePanel,
    miniMaxFp16Accumulation, miniMaxImageModeSource, miniMaxContinuationDirection, miniMaxContinuationDirectionField, miniMaxPromptAutoNote,
    miniMaxContinuationStart, miniMaxContinuationStartField, miniMaxContinuationStartValue,
    miniMaxLatentContextFrames, miniMaxLatentContinuationRow,
    miniMaxLatentStatusPill, miniMaxLocationTransitionControls, miniMaxLocationTransitionCustom,
    miniMaxLocationTransitionCustomField, miniMaxLocationTransitionPreset, miniMaxLoraCount, miniMaxLoraNote,
    miniMaxLoraRows, miniMaxLoraSection, miniMaxLoraSlots, miniMaxMegapixels, miniMaxMegapixelsField, miniMaxResolutionPreset,
    miniMaxMemoryEfficientSageAttention, miniMaxModeButtons, miniMaxModelLoaderSettings, miniMaxModePanels,
    miniMaxPass2Prompt, miniMaxPass2PromptField, miniMaxPassButtons, miniMaxPassChooser, miniMaxPrompt, miniMaxVideoProfileControls,
    miniMaxPromptCharacterStatus, miniMaxPromptRunnerNote, miniMaxReferenceConditioningSettings,
    miniMaxRefImageSize, miniMaxSageAttention, miniMaxSamplerName, miniMaxSamplerSettings,
    miniMaxSceneImageUse, miniMaxSceneImageUseField, miniMaxScheduler, miniMaxSeed, miniMaxSeedField,
    miniMaxSettingsScopeNote, miniMaxSpeakerAssignmentList, miniMaxSpeakerAssignmentNote,
    miniMaxStartFrameCharacterInfluence, miniMaxStartFrameCharacterInfluenceField,
    miniMaxStartFrameReferenceNote, miniMaxSteps, miniMaxSubTabs, miniMaxThreePassLoraPicker,
    miniMaxThreePassLoraSection, miniMaxThreePassLoraStrength, miniMaxThreePassRefImageSize,
    miniMaxThreePassSettings, miniMaxTurboLoraField, miniMaxTurboLoraPicker, miniMaxTurboLoraStrength,
    miniMaxTurboLoraStrengthField, miniMaxTurboNote, miniMaxTurboSection, miniMaxTwoPassLatentScale, miniMaxTwoPassLatentUpscalerPicker,
    miniMaxTwoPassLoraLayout, miniMaxTwoPassLoraPicker, miniMaxTwoPassLoraPreset,
    miniMaxTwoPassLoraPresetButtons, miniMaxTwoPassLoraPresetField, miniMaxTwoPassLoraSection,
    miniMaxTwoPassLoraStatus, miniMaxTwoPassLoraStrength, miniMaxTwoPassOutputCrf, miniMaxTwoPassRefImageSize,
    miniMaxTwoPassResizeMethod, miniMaxTwoPassSettings, miniMaxTwoPassTeCacheDepth, miniMaxTwoPassTeDevice,
    miniMaxTwoPassTeEnd, miniMaxTwoPassTeMcs, miniMaxTwoPassTeProcessingControl, miniMaxTwoPassTeStart,
    miniMaxTwoPassUseFastVaeDecode, miniMaxUseCurrentSceneVideoButton, miniMaxUseLoras, miniMaxUseTurboLora,
    miniMaxVideoReferenceRows, miniMaxVideoVaePicker, miniMaxWarmupFrames, saveMiniMaxPromptButton,
    twoPassControls, useSceneMiniMaxH3Settings, useSceneMiniMaxH3SettingsNote,
  } = buildMiniMaxPanel({
    miniMaxGemmaModelSelect, miniMaxMmprojSelect, miniMaxReferencesButton, miniMaxSceneVideoButton,
    miniMaxTextGemmaModelSelect, miniMaxVideoReferencesButton,
    updateMiniMaxPromptSaveButtonState: (...args) => updateMiniMaxPromptSaveButtonState(...args),
  });

  const videoSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        useSceneI2VVideoSettings.wrapper,
        useSceneI2VVideoSettingsNote,
        makeSettingsSection("Video Models", [
          i2vUseGgufModel.wrapper,
          i2vUnetModelField,
          i2vDiffusionModelField,
          makeField("Video VAE", i2vVaePicker.wrapper),
          makeField("Clip model 1", i2vClip1Picker.wrapper),
          makeField("Clip model 2", i2vClip2Picker.wrapper),
          makeField("Latent upscaler", i2vUpscalePicker.wrapper),
          makeField("Audio VAE", i2vAudioVaePicker.wrapper),
          i2vDiffusionLoaderAdvanced,
        ]),
        makeSettingsSection("Non-Vision LLM Models", [
          makeField("Non-Vision text LLM model", i2vTextGemmaModelSelect),
        ]),
        makeSettingsSection("Vision LLM Models", [
          makeField("Vision LLM model", i2vGemmaModelSelect),
          makeField("Vision mmproj", i2vMmprojSelect),
        ]),
        ltxMsrRequiredPanel,
        ltxIngredientsRequiredPanel,
        ltxIdLoraRequiredPanel,
        i2vUseLora.wrapper,
        i2vLoraPanel,
        createSceneVideoActions,
      ]),
    },
    {
      label: "Video Settings",
      value: "settings",
      content: makeSettingsPanel([
        videoSettingsScopeNote,
        makeField("Video trigger phrase", videoTriggerInput),
        makeSettingsSection("Render basics", [
          i2vSettingsGrid,
          ltx25ResolutionGrid,
          ltxIngredientsResolutionWarning,
        ]),
        idLoraVoiceSettingsSection,
        flfGuideSettingsSection,
        makeSettingsSection("Advanced Settings", [
          i2vWarmCooldownSection,
          i2vAdvancedNodeSettingsSection,
        ]),
        rtvSceneImageAnchorSection,
        makeCreateSceneVideoButton(),
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        useI2VPromptEnhancementPass.wrapper,
        i2vPromptEnhancementNote,
        makeField("Line / lyric / dialogue", lyricTextInput),
        makeField("Performer(s) / speaker(s)", lyricSingersInput),
        makeField("Video motion notes", i2vNotesInput),
        flfTransitionTypeField,
        useI2VVisionReference.wrapper,
        i2vReferenceNote,
        useT2VVisionReference.wrapper,
        t2vReferenceNote,
        t2vLocationNote,
        t2vRefImagePanel,
        createI2VButton,
        editI2VPromptButton,
        makeField("Video prompt", i2vPrompt),
        saveI2VPromptButton,
        makeSettingsSection("Advanced", [
          editI2VInstructionsButton,
          editIdLoraInstructionsButton,
          editRTVInstructionsButton,
          editIngredientsInstructionsButton,
          editT2VInstructionsButton,
        ], false),
        makeCreateSceneVideoButton(),
      ]),
    },
  ]);
  const ltxVideoPanel = document.createElement("div");
  ltxVideoPanel.style.cssText = "display:flex;flex-direction:column;gap:10px;";
  ltxVideoPanel.append(videoModeChooser, importCustomVideoPanel, videoSubTabs.wrapper);
  videoPanel.append(ltxVideoPanel, miniMaxEnginePanel);
  audioPanel.append(
    makeSettingsSection("Scene Audio", [
      audioSummary,
      openSceneAudioOptionsButton,
    ]),
    makeSettingsSection("Timeline Audio", [
      globalAudioSummary,
      makeField("Audio source", globalAudioModeSelect),
      globalAudioDrop,
      chooseGlobalAudioButton,
      silentAudioPanel,
      globalAudioGuide,
    ], false),
  );
  inspector.append(inspectorTabs, noSceneNotice, scenePanel, imagePanel, videoPanel, audioPanel);

  const {
    addOverlaySegmentButton, addSegmentButton, addTimelineMarkerButton, beatMarkersButton, bulkSegmentsButton,
    clearRangeButton, closeTimelineGapsButton, deleteAllSegmentsButton, deleteAllTimelineImagesButton,
    deleteAllTimelineVideosButton, deleteSegmentButton, deleteSelectedMediaButton, globalAudioMuteButton,
    globalScrub, globalScrubTime, idLoraTrimModeButton, lyricNoteButton, locationThumbnailButton, multiSelectButton,
    multiSelectHintButton, overlayTrackHintButton, overlayTrackToggleButton, playButton, playhead, redoButton,
    sceneNoteButton, segmentLayer, selectedMediaLabel, setInButton, setOutButton, snapSceneEdgeButton,
    snapToBeatsControl, splitSceneButton, stopButton, timeline, timelineCanvas, timelineInfo,
    timelineRangeInfo, timelineResizeHandle, timelineViewport, undoButton, useFrameAsImageButton,
    videoNoteButton, waveformModeSelect, zoomInButton, zoomOutButton,
  } = buildTimelineView({
    preview, previewStage,
  });
  const audio = document.createElement("audio");
  audio.preload = "metadata";
  const sceneAudio = document.createElement("audio");
  sceneAudio.preload = "metadata";
  updateGlobalAudioMuteButton();

  const shellHeader = document.createElement("div");
  shellHeader.style.cssText = "display:flex;flex-direction:column;min-width:0;min-height:0;";
  shellHeader.append(updateStatusBanner, topbar);
  shell.append(shellHeader, main, timeline);
  overlay.append(shell);
  document.body.append(overlay);
  builderLifecycle.resourceResizeObserver = new ResizeObserver(positionBuilderResourceMonitor);
  builderLifecycle.resourceResizeObserver.observe(centerActions);
  builderLifecycle.resourceResizeObserver.observe(importActions);
  positionBuilderResourceMonitor();
  void pollBuilderResources();
  refreshV10UpdateStatus();
  installFileDropNavigationGuard(shell);
  window.VRGDG_UIThemes?.registerRoot?.(overlay);

  setTimeout(() => {
    showStartupWelcome().catch((error) => {
      console.warn("[VRGDG Music Builder] Startup welcome failed:", error);
      toast(`Video Creator startup failed:\n${String(error?.message || error)}`, true);
    });
  }, 250);
  for (const eventName of ["dragenter", "dragover", "dragleave", "drop"]) {
    overlay.addEventListener(eventName, (event) => {
      if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
      if (event.target?.closest?.("[data-vrgdg-file-drop-zone='true']")) return;
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation?.();
    }, true);
  }

  const state = {
    duration: 0,
    audioDuration: 0,
    audioPath: String(audioInput.value || ""),
    peaks: [],
    beats: [],
    detectedTempoBpm: 0,
    beatCalibration: null,
    segments: [],
    overlaySegments: [],
    overlayTrack: normalizeOverlayTrackState(),
    activeId: "",
    miniMaxH3PanelSegmentId: "",
    activeTrack: "base",
    multiSelectMode: false,
    modifierMultiSelectMode: false,
    selectedSegmentIds: [],
    inspectorTab: "scene",
    leftPanelTab: "scenes",
    pxPerSecond: 45,
    timelineZoom: 45,
    selectedTimelineRange: { in: null, out: null },
    timelineMarkers: [],
    activeTimelineMarkerId: "",
    waveformMode: "medium",
    snapToBeats: true,
    showBeatMarkers: false,
    showTimelineSceneNotes: false,
    showTimelineVideoNotes: false,
    showTimelineLyricNotes: false,
    imageContinuityEnabled: false,
    imageContinuityStrength: "balanced",
    leftPanelWidth: 260,
    leftPanelCollapsed: false,
    rightPanelCollapsed: false,
    llmPopoutOpen: false,
    llmPopoutWidth: 460,
    llmPopoutHeight: 460,
    llmPopoutX: null,
    llmPopoutY: null,
    uiProfile: "",
    rightPanelWidth: 360,
    timelinePanelHeight: 300,
    projectFolder: projectInput.value,
    sessionPath: "",
    srtPath: "",
    autoSaveEnabled: true,
    // Testing option: relaxed mode accepts externally authored MiniMax prompts
    // but still enforces the provider's hard 7,000-character limit.
    failOnInvalidPromptFormats: false,
    automaticMemoryCleanup: false,
    sceneRenderWaitHours: DEFAULT_SCENE_RENDER_WAIT_HOURS,
    isScrubbing: false,
    isClipScrubbing: false,
    sceneAudioMode: false,
    sceneAudioSegmentId: "",
    sceneAudioGlobalTime: 0,
    sceneSelectionUsesGlobalAudio: false,
    timingFrozen: false,
    srtMode: false,
    promptJsonPath: "",
    i2vMotionJsonPath: "",
    lyricSegmentsPath: "",
    imageTriggerPhrase: "",
    videoTriggerPhrase: "",
    defaultFacialPerformance: "",
    defaultFacialPerformanceCustom: "",
    useI2VPromptEnhancementPass: false,
    autoChainLastFrame: false,
    autoChainStyle: "continuous",
    autoChainDirection: "",
    autoChainTransitionLoraPrompt: false,
    autoChainTransitionTrigger: "zhuanchang",
    useVrgdgTextContext: true,
    themeStylePath: "",
    storyIdeaPath: "",
    subjectScenePath: "",
    textGemmaRunner: "builtin",
    qwenModelFile: "",
    qwenMmprojFile: "",
    gemmaModelFile: "",
    gemmaContextLimit: 8000,
    gemmaOutputTokenLimit: 8192,
    gemmaGpuLayers: 99,
    lmStudioBaseUrl: "http://127.0.0.1:1234/v1",
    lmStudioModel: "",
    lmStudioApiKey: "",
    lmStudioContextLimit: 32768,
    lmStudioOutputTokenLimit: 8192,
    llmApiProvider: "openai",
    llmApiModel: "",
    llmApiKey: "",
    llmApiKeyProject: "",
    llmApiChoices: null,
    ownServerUrl: "http://127.0.0.1:8000/v1",
    ownServerModel: "",
    ownServerApiKey: "",
    ownServerApiKeyProject: "",
    ownServerOutputTokenLimit: 8192,
    ownServerTimeoutMinutes: 6,
    customModelsRoot: "",
    notificationSettings: defaultNotificationSettings(),
    videoType: "singing",
    projectVideoEngine: "ltx",
    miniMaxH3Settings: cloneMiniMaxH3Settings(),
    autoTimeSingerCuesBeforePrompt: false,
    miniMaxH3TwoPassEnabled: false,
    miniMaxH3ThreePassEnabled: false,
    imageModelMode: "zimage",
    zimageSettings: defaultZImageSettings(),
    referenceKrea2Settings: { ...DEFAULT_KREA2_REFERENCE_SETTINGS },
    fluxKleinSettings: defaultFluxKleinSettings(),
    flowGptBrowserSettings: defaultFlowGptBrowserSettings(),
    ernieImageSettings: defaultErnieImageSettings(),
    krea2TwoPassSettings: defaultKrea2TwoPassSettings(),
    nbImageSettings: defaultNBImageSettings(),
    useFluxGlobalImageIngredients: false,
    fluxGlobalImageIngredients: [],
    fluxReferenceBuilder: defaultFluxReferenceBuilder(),
    idLoraReferenceBuilder: defaultIdLoraReferenceBuilder(),
    lyricMapper: defaultLyricMapper(),
    zEnhanceSettings: defaultZEnhanceSettings(),
    videoModelMode: "i2v",
    timelineTrimEditMode: false,
    timelineEditMenuOpen: false,
    i2vVideoSettings: defaultI2VVideoSettings(),
    continuityMode: "off",
    autoImg2ImgStartStep: 5,
    autoImg2ImgCreativity: 5,
    promptToolsHintPrefs: {},
    builderAgentMessages: [],
    builderAgentAutoApply: false,
    builderAgentPurpose: "scene_work",
    builderAgentReferenceImages: [],
    builderStorySourcePath: "",
    builderStorySourcePreview: "",
    builderStoryReferenceImages: [],
    builderStoryReferenceNotes: "",
    builderStoryLayer: normalizeBuilderStoryLayer({}),
    builderStoryboardDefaults: normalizeBuilderStoryboardDefaults({}),
    autoBuildPreparation: normalizeAutoBuildPreparation({}),
    wizardBetaDraft: null,
    postProcessTab: "luts",
    adjustLivePreview: false,
    adjustLivePreviewTimer: null,
    adjustLivePreviewToken: 0,
    adjustLivePreviewBusy: false,
    adjustLivePreviewPending: false,
    adjustLivePreviewStatus: "",
    builderAgentFloating: null,
    renderLogs: [],
    activeRenderLogId: "",
    renderLogModalRefresh: null,
    undoStack: [],
    redoStack: [],
    isRestoringHistory: false,
    batchCancelled: false,
  };
  // A floating window with the MiniMax prompt fields, opened by a checkbox at the top of the right panel.
  const llmPopout = createLlmPopout({
    state, inspector, overlay,
    fields: {
      prompt: miniMaxPrompt,
      saveButton: saveMiniMaxPromptButton,
      status: miniMaxPromptCharacterStatus,
      pass2Prompt: miniMaxPass2Prompt,
      pass2Field: miniMaxPass2PromptField,
      direction: miniMaxContinuationDirection,
      promptAutoNote: miniMaxPromptAutoNote,
      start: miniMaxContinuationStart,
      startValue: miniMaxContinuationStartValue,
      startField: miniMaxContinuationStartField,
    },
    autoSaveSessionQuiet: (...args) => autoSaveSessionQuiet(...args),
    activeSegment: (...args) => activeSegment(...args),
    sceneDisplayName: (...args) => sceneDisplayName(...args),
    segmentIndexInfo: (...args) => segmentIndexInfo(...args),
  });

  const {
    createDetailedLocationDescriptionWithGemma, createProgressWindow, describeReferenceImageWithGemma,
    gemmaRunnerLabel, gemmaRunnerLine, llmApiVisionModelSelected, promptRunnerActionName, referenceDescriptionMmproj,
    referenceDescriptionVisionModel, textGemmaRunnerPayload, updatePromptRunnerButtonLabels,
  } = createLlmRunner({
    autoSaveSessionQuiet, createFluxPromptButton, createI2VButton, createNBPromptButton, createT2IButton,
    ernieCreateT2IButton, ernieGemmaModelSelect, ernieMmprojSelect, flowGptCreatePromptButton,
    fluxGemmaModelSelect, fluxMmprojSelect, gemmaModelSelect, gemmaT2IAllButton, gemmaVideoAllButton,
    i2vGemmaModelSelect, i2vMmprojSelect, i2vTextGemmaModelSelect, krea2TwoPassCreateT2IButton, mmprojSelect,
    nbGemmaModelSelect, nbMmprojSelect, state, t2iTextGemmaModelSelect, zEnhanceGemmaButton,
    currentVideoMode: (...args) => currentVideoMode(...args),
  });

  const { clearHistoryBlobCache, pushHistory, redo, undo, updateHistoryButtons } = createHistory({
    autoSaveControl, imageContinuityEnabled, imageContinuityStrength, redoButton, snapToBeatsControl, state,
    undoButton, waveformModeSelect,
    syncInspector: (...args) => syncInspector(...args),
    syncI2VVideoSettingsPanel: (...args) => syncI2VVideoSettingsPanel(...args),
    syncVideoModePanel: (...args) => syncVideoModePanel(...args),
    syncZEnhanceSettingsPanel: (...args) => syncZEnhanceSettingsPanel(...args),
    syncErnieImagePanel: (...args) => syncErnieImagePanel(...args),
    syncFluxKleinPanel: (...args) => syncFluxKleinPanel(...args),
    syncKrea2TwoPassPanel: (...args) => syncKrea2TwoPassPanel(...args),
    syncZImageSettingsPanel: (...args) => syncZImageSettingsPanel(...args),
    render: (...args) => render(...args),
    setBeatMarkersVisible: (...args) => setBeatMarkersVisible(...args),
    syncLyricNoteControls: (...args) => syncLyricNoteControls(...args),
    syncSceneNoteControls: (...args) => syncSceneNoteControls(...args),
    syncVideoNoteControls: (...args) => syncVideoNoteControls(...args),
    activeSegment: (...args) => activeSegment(...args),
    applyLayoutSizes: (...args) => applyLayoutSizes(...args),
    ensureAllSegmentRuntimeFields: (...args) => ensureAllSegmentRuntimeFields(...args),
    segmentTrack: (...args) => segmentTrack(...args),
    syncI2VMotionJsonFromSegments: (...args) => syncI2VMotionJsonFromSegments(...args),
    syncPromptJsonFromSegments: (...args) => syncPromptJsonFromSegments(...args),
    syncBuilderLlmModelSelectsFromRunner: (...args) => syncBuilderLlmModelSelectsFromRunner(...args),
  });

  const {
    applyLutToSegment, clearSegmentAdjustPreview, clearSegmentFilmGrainPreview, clearSegmentLutPreview,
    enableLutDrop, enablePostEffectDrop, normalizeSceneAdjust, renderFilmGrainPostProcessPanel,
    renderSceneAdjustPanel, renderSceneToolsPanel, sceneAdjustHasRenderableChanges, sceneAdjustSignature,
    syncLeftPanelTabs, syncPostProcessTabs,
  } = createPostProcess({
    autoSaveSessionQuiet, createProgressWindow, filmGrainPane, fxOverlaysPane, fxOverlaysTab, fxPane,
    lutsTabButton, lutsTools, postProcessFxTab, postProcessGrainTab, postProcessLutsTab, postProcessPane,
    projectInput, pushHistory, sceneAdjustPanel, sceneListPane, scenesTabButton, sceneToolsPanel, state,
    toolsPane, toolsTabButton,
    selectedSegmentImagePath: (...args) => selectedSegmentImagePath(...args),
    showAdjustPreviewImage: (...args) => showAdjustPreviewImage(...args),
    showFilmGrainPreviewImage: (...args) => showFilmGrainPreviewImage(...args),
    showLutPreviewImage: (...args) => showLutPreviewImage(...args),
    syncInspector: (...args) => syncInspector(...args),
    syncPreview: (...args) => syncPreview(...args),
    applySceneAdjustToRenderedVideo: (...args) => applySceneAdjustToRenderedVideo(...args),
    applySceneFilmGrainToRenderedVideo: (...args) => applySceneFilmGrainToRenderedVideo(...args),
    applySceneLutToRenderedVideo: (...args) => applySceneLutToRenderedVideo(...args),
    render: (...args) => render(...args),
    activeSegment: (...args) => activeSegment(...args),
    allEditableSegments: (...args) => allEditableSegments(...args),
    segmentIndexInfo: (...args) => segmentIndexInfo(...args),
    segmentTrack: (...args) => segmentTrack(...args),
  });

  const { playBuilderNotification, shouldNotifyForToast, syncVideoTypeControl } = createNotificationSounds({
    state, videoTypeSelect,
  });

  const {
    firstLastFrameEndImageSource, firstLastFramePromptReferences, firstLastFrameResolvedEndImageSource,
    firstLastFrameStartImageSource, flfChainingEnabled, flfGemmaContextMode, flfGemmaSceneConcept,
    flfGemmaVisualNotes, flfPreGeneratePromptsEnabled, flfRenderChainStartSource, flfTransitionLoraActive,
    hasFirstLastFrameEndImage, loadFirstLastFrameEndFile, mergedFluxImageIngredients,
    promoteChainedFLFSceneImageToEndFrame, renderFluxGlobalIngredientList, renderFluxIngredientList,
    renderNBIngredientList, rtvReferenceBehaviorForSegment, rtvSceneImageAnchorPayload,
    setImageSeedForCurrentMode, setMiniMaxH3SeedRandom, setVideoSeedRandom, syncFluxGlobalIngredientPanel,
  } = createImageReferences({
    autoSaveSessionQuiet, ernieSeed, fluxGlobalIngredientList, fluxGlobalIngredientPanel, fluxIngredientList,
    fluxSeed, i2vSeedInput, krea2TwoPassSeed, nbGlobalIngredientList, nbGlobalIngredientPanel,
    nbIngredientList, nbUseGlobalIngredients, projectInput, pushHistory, state, useFluxGlobalIngredients,
    zSeed,
    renderList: (...args) => renderList(...args),
    miniMaxH3SettingsForSegment: (...args) => miniMaxH3SettingsForSegment(...args),
    syncMiniMaxH3Panel: (...args) => syncMiniMaxH3Panel(...args),
    currentVideoMode: (...args) => currentVideoMode(...args),
    syncRTVSceneImageAnchorPanel: (...args) => syncRTVSceneImageAnchorPanel(...args),
    i2vVideoSettingsForSegment: (...args) => i2vVideoSettingsForSegment(...args),
    previousAutoChainSourceSegment: (...args) => previousAutoChainSourceSegment(...args),
    sceneVideoConceptPromptText: (...args) => sceneVideoConceptPromptText(...args),
    render: (...args) => render(...args),
    activeSegment: (...args) => activeSegment(...args),
    sceneSlotNumber: (...args) => sceneSlotNumber(...args),
    logicalSubjectIdsForScene: (...args) => logicalSubjectIdsForScene(...args),
    sceneReferenceMapValue: (...args) => sceneReferenceMapValue(...args),
    segmentImageSource: (...args) => segmentImageSource(...args),
    activeErnieImageSettings: (...args) => activeErnieImageSettings(...args),
    activeFluxKleinSettings: (...args) => activeFluxKleinSettings(...args),
    activeKrea2TwoPassSettings: (...args) => activeKrea2TwoPassSettings(...args),
    activeZImageSettings: (...args) => activeZImageSettings(...args),
    storyboardVideoExtraNotesForSegment: (...args) => storyboardVideoExtraNotesForSegment(...args),
    videoGemmaNotesForSegment: (...args) => videoGemmaNotesForSegment(...args),
  });

  const {
    appendTimelineFirstLastFrameThumbnail, beginGlobalTimelineScrub, clearActiveSegment,
    clearConceptPromptNotesFromSegments, clearI2VMotionNotesFromSegments, handlePreviewVideoLoadIssue,
    mediaThumbnailHtml, moveActiveSceneSelection, openMultiSelectChooser, playSceneAudioFrom,
    playbackSegmentAtTime, selectedMediaForDelete, selectedSegmentImagePath,
    selectedSegmentImageThumbnailPath, setActiveSegment, setGlobalPlaybackTime, showAdjustPreviewImage,
    showFilmGrainPreviewImage, showLutPreviewImage, syncInspector, syncPreview, syncPreviewPlayback,
    updateAudioScrubbers, updateSelectedMediaTools, waitForPreviewVideoReady,
  } = createSelectionPreview({
    audioInput, audioSummary, cancelPreviewPlayStart, clearSceneEndFrameButton, clearSegmentAdjustPreview,
    clearSegmentFilmGrainPreview, clearSegmentLutPreview, createFluxPromptButton, createI2VButton,
    createNBPromptButton, createSceneEndFrameButton, createSceneVideoButton, createT2IButton,
    deleteAllTimelineVideosButton, deleteSegmentButton, deleteSelectedMediaButton,
    editErnieT2IInstructionsButton, editFlowGptT2IInstructionsButton, editFluxKleinT2IInstructionsButton,
    editI2VPromptButton, editIdLoraInstructionsButton, editImagePromptButtons, editKrea2T2IInstructionsButton,
    editNanoBT2IInstructionsButton, editZImageT2IInstructionsButton, endInput, ernieCreateButton,
    ernieCreateT2IButton, ernieGemmaModelSelect, ernieMmprojSelect, ernieNotesInput, ernieRefImagePanel,
    ernieT2IPrompt, ernieTextGemmaModelSelect, ernieUseVisionReference, firstLastFramePromptReferences,
    firstLastFrameResolvedEndImageSource, firstLastFrameStartImageSource, flfTransitionTypeSelect,
    flowGptCreatePromptButton, fluxPrompt, fluxUseDirectorNotes, fluxUseTextOnlyGemmaPrompt,
    freezeTimingControl, gemmaModelSelect, globalAudioSummary, globalScrub, globalScrubTime,
    i2vGemmaModelSelect, i2vMmprojSelect, i2vMotionJsonInput, i2vNotesInput, i2vPrompt,
    i2vTextGemmaModelSelect, i2vUseGgufModel, krea2TwoPassCreateT2IButton, krea2TwoPassNotesInput,
    krea2TwoPassRefImagePanel, krea2TwoPassT2IPrompt, krea2TwoPassUseVisionReference, labelInput,
    loadCustomImageButton, lyricSingersInput, lyricTextInput, miniMaxGemmaModelSelect, miniMaxMmprojSelect,
    miniMaxSceneVideoButtons, miniMaxTextGemmaModelSelect, mmprojSelect, nbApiKey, nbGemmaModelSelect,
    nbMmprojSelect, nbModelSelect, nbNotes, nbPrompt, nbUseDirectorNotes, nbUseTextOnlyGemmaPrompt,
    notesInput, openSceneAudioOptionsButton, playhead, playStart, postProcessComparePreview, preloadVideo,
    previewButton, previewDecodeHint, previewEmpty, previewImage, previewNBButton, previewVideo,
    previewVideoState, promptJsonInput, refImageInput, refImagePanel, renderFilmGrainPostProcessPanel,
    renderSceneAdjustPanel, renderSceneToolsPanel, rtvReferenceBehaviorSelect, savedI2VPrompts,
    saveI2VPromptButton, saveMiniMaxPromptButton, sceneAudio, segmentLayer, selectedMediaLabel,
    silentTimeline, srtInput, startInput, state, storyIdeaInput, subjectSceneInput, t2iPrompt,
    t2iTextGemmaModelSelect, t2vRefImagePanel, themeStyleInput, timelineCanvas, useFrameAsImageButton,
    useI2VVisionReference, useSceneErnieImageSettings, useSceneFluxKleinSettings, useSceneI2VVideoSettings,
    useSceneKrea2TwoPassSettings, useSceneMiniMaxH3Settings, useSceneNBImageSettings, useSceneZImageSettings,
    useT2VVisionReference, useVisionReference, useVrgdgTextContext, zEnhanceGemmaButton,
    zEnhanceGemmaModelSelect, zEnhanceGemmaNotes, zEnhanceMmprojSelect, zEnhancePromptPreview,
    saveMiniMaxH3SettingsFromPanel: (...args) => saveMiniMaxH3SettingsFromPanel(...args),
    syncMiniMaxH3Panel: (...args) => syncMiniMaxH3Panel(...args),
    rtvReferenceBehaviorGlobalValue: (...args) => rtvReferenceBehaviorGlobalValue(...args),
    saveI2VVideoSettingsFromPanel: (...args) => saveI2VVideoSettingsFromPanel(...args),
    syncRTVSceneImageAnchorPanel: (...args) => syncRTVSceneImageAnchorPanel(...args),
    syncVideoModePanel: (...args) => syncVideoModePanel(...args),
    syncZEnhanceSettingsPanel: (...args) => syncZEnhanceSettingsPanel(...args),
    updateActiveFromInputs: (...args) => updateActiveFromInputs(...args),
    syncErnieImagePanel: (...args) => syncErnieImagePanel(...args),
    syncFluxKleinPanel: (...args) => syncFluxKleinPanel(...args),
    syncKrea2TwoPassPanel: (...args) => syncKrea2TwoPassPanel(...args),
    syncZImageSettingsPanel: (...args) => syncZImageSettingsPanel(...args),
    render: (...args) => render(...args),
    activeSegment: (...args) => activeSegment(...args),
    allEditableSegments: (...args) => allEditableSegments(...args),
    currentGlobalTime: (...args) => currentGlobalTime(...args),
    ensureGlobalTimelineAudioSource: (...args) => ensureGlobalTimelineAudioSource(...args),
    ensureSegmentRuntimeFields: (...args) => ensureSegmentRuntimeFields(...args),
    isSegmentMultiSelected: (...args) => isSegmentMultiSelected(...args),
    isTimelinePlaying: (...args) => isTimelinePlaying(...args),
    pauseTimelineForEditing: (...args) => pauseTimelineForEditing(...args),
    playbackDuration: (...args) => playbackDuration(...args),
    segmentIndexInfo: (...args) => segmentIndexInfo(...args),
    segmentTrack: (...args) => segmentTrack(...args),
    selectedSegmentsForBatch: (...args) => selectedSegmentsForBatch(...args),
    startSilentTimelinePlayback: (...args) => startSilentTimelinePlayback(...args),
    timelineAudioPathForSegment: (...args) => timelineAudioPathForSegment(...args),
    timelineAudioSegmentAtTime: (...args) => timelineAudioSegmentAtTime(...args),
    timelineAudioSourceStartForSegment: (...args) => timelineAudioSourceStartForSegment(...args),
    timelineDuration: (...args) => timelineDuration(...args),
    usingSceneAudioPlaybackMode: (...args) => usingSceneAudioPlaybackMode(...args),
    segmentImageSource: (...args) => segmentImageSource(...args),
    syncInspectorPanels: (...args) => syncInspectorPanels(...args),
    updateI2VPromptSaveButtonState: (...args) => updateI2VPromptSaveButtonState(...args),
  });

  const { openLyricReviewModal } = createLyricReview({
    applyLyricSectionsFromReferenceText, audioInput, hasLockedVideo, isInstrumentalLyricText,
    isNoLipSyncSingerChoice, normalizeFluxReferenceBuilder, parseBulkTimeValue, pushHistory, state,
    syncInspector,
    saveSession: (...args) => saveSession(...args),
    currentVideoMode: (...args) => currentVideoMode(...args),
    render: (...args) => render(...args),
    activeSegment: (...args) => activeSegment(...args),
    currentGlobalTime: (...args) => currentGlobalTime(...args),
    ensureAllSegmentRuntimeFields: (...args) => ensureAllSegmentRuntimeFields(...args),
    ensureSegmentRuntimeFields: (...args) => ensureSegmentRuntimeFields(...args),
    segmentTrack: (...args) => segmentTrack(...args),
    timelineDuration: (...args) => timelineDuration(...args),
    applyIngredientsReferenceMappings: (...args) => applyIngredientsReferenceMappings(...args),
    referenceBuilderSubjectChoices: (...args) => referenceBuilderSubjectChoices(...args),
    syncIngredientsSceneMapFromSubjectMappings: (...args) => syncIngredientsSceneMapFromSubjectMappings(...args),
    syncLyricMapperFromSegments: (...args) => syncLyricMapperFromSegments(...args),
  });

  const {
    clearMiniMaxImageReferenceStartFrameOnModeSwitch, ensureMiniMaxSpeakerAssignments,
    isMiniMaxBuiltInSpeakerAssignmentMode, isMiniMaxSingerAssignmentMode, miniMaxH3ContinuityModeForSegment,
    miniMaxH3ContinuityReferenceReserved, miniMaxH3ModeForSegment, miniMaxH3SceneImageIsPromptInspiration,
    miniMaxH3SceneImageUseForSegment, miniMaxH3SettingsForSegment,
    miniMaxH3StartFrameCharacterInfluenceForSegment, miniMaxMappedSpeakersForSegment,
    renderMiniMaxSpeakerAssignmentPanel, saveMiniMaxH3SettingsFromPanel, saveMiniMaxSceneInputsFromPanel,
    setMiniMaxH3ModeForSegment, setMiniMaxH3RenderPassForSegment, syncMiniMaxH3Panel,
    syncMiniMaxReferenceButtons, toggleProjectVideoEngineFromBadge, updateMiniMaxPromptCharacterStatus,
    updateMiniMaxPromptSaveButtonState,
  } = createMiniMaxPanel({
    advancedTwoPassControls, audio, autoSaveSessionQuiet, createProgressWindow, gemmaRunnerLabel,
    miniMaxAccelerationControls, miniMaxAddSpeakerCueButton, miniMaxAdvancedLatentUpscalerPicker, miniMaxAdvancedSettings,
    miniMaxAdvancedVramPreset,
    miniMaxAspectRatio, miniMaxAudioMode, miniMaxAudioNote, miniMaxAudioVaePicker,
    miniMaxAutoTimeBeforePrompt, miniMaxClipPicker, miniMaxContinuityMode, miniMaxContinuityNote,
    miniMaxContinuityPromptFromLastFrame, miniMaxCooldownFrames, miniMaxCreatePromptButton, miniMaxDenoise,
    miniMaxDiffusionModelPicker, miniMaxEasyCacheBypass, miniMaxEasyCacheEndPercent,
    miniMaxEasyCacheReuseThreshold, miniMaxEasyCacheSettings, miniMaxEasyCacheStartPercent,
    miniMaxEasyCacheVerbose, miniMaxEditInstructionsButton, miniMaxFp16Accumulation, miniMaxImageModeSource,
    miniMaxContinuationDirection, miniMaxContinuationDirectionField, miniMaxPromptAutoNote, miniMaxContinuationStart,
    miniMaxContinuationStartField, miniMaxContinuationStartValue, miniMaxLatentContextFrames,
    miniMaxLatentContinuationRow, miniMaxLatentStatusPill,
    miniMaxH3FrameContinuityPromptEnabled: (...args) => miniMaxH3FrameContinuityPromptEnabled(...args),
    miniMaxLocationTransitionControls, miniMaxLocationTransitionCustom, miniMaxLocationTransitionCustomField,
    miniMaxLocationTransitionPreset, miniMaxLoraCount, miniMaxLoraNote, miniMaxLoraRows, miniMaxLoraSection,
    miniMaxLoraSlots, miniMaxMegapixels, miniMaxMegapixelsField, miniMaxResolutionPreset, miniMaxMemoryEfficientSageAttention,
    miniMaxModeButtons, miniMaxModelLoaderSettings, miniMaxModePanels, miniMaxPass2Prompt,
    miniMaxPass2PromptField, miniMaxPassButtons, miniMaxPassChooser, miniMaxPrompt,
    miniMaxPromptCharacterStatus, miniMaxPromptRunnerNote, miniMaxReferenceButtons,
    miniMaxReferenceConditioningSettings, miniMaxRefImageSize, miniMaxSageAttention, miniMaxSamplerName,
    miniMaxSamplerSettings, miniMaxSceneImageUse, miniMaxSceneImageUseField, miniMaxSceneVideoButton,
    miniMaxScheduler, miniMaxSeed, miniMaxSeedField, miniMaxSettingsScopeNote, miniMaxSpeakerAssignmentList,
    miniMaxSpeakerAssignmentNote, miniMaxStartFrameCharacterInfluence,
    miniMaxStartFrameCharacterInfluenceField, miniMaxStartFrameReferenceNote, miniMaxSteps, miniMaxSubTabs,
    miniMaxThreePassLoraPicker, miniMaxThreePassLoraSection, miniMaxThreePassLoraStrength,
    miniMaxThreePassRefImageSize, miniMaxThreePassSettings, miniMaxTurboLoraField, miniMaxTurboLoraPicker,
    miniMaxTurboLoraStrength, miniMaxTurboLoraStrengthField, miniMaxTurboNote, miniMaxTurboSection,
    miniMaxTwoPassLatentScale,
    miniMaxTwoPassLatentUpscalerPicker, miniMaxTwoPassLoraLayout, miniMaxTwoPassLoraPicker,
    miniMaxTwoPassLoraPreset, miniMaxTwoPassLoraPresetButtons, miniMaxTwoPassLoraPresetField,
    miniMaxTwoPassLoraSection, miniMaxTwoPassLoraStatus, miniMaxTwoPassLoraStrength, miniMaxTwoPassOutputCrf,
    miniMaxTwoPassRefImageSize, miniMaxTwoPassResizeMethod, miniMaxTwoPassSettings,
    miniMaxTwoPassTeCacheDepth, miniMaxTwoPassTeDevice, miniMaxTwoPassTeEnd, miniMaxTwoPassTeMcs,
    miniMaxTwoPassTeProcessingControl, miniMaxTwoPassTeStart, miniMaxTwoPassUseFastVaeDecode, miniMaxUseLoras,
    miniMaxUseTurboLora, miniMaxVideoReferenceRows, miniMaxVideoReferencesButton, miniMaxVideoVaePicker,
    miniMaxWarmupFrames, playSceneAudioFrom, projectInput, pushHistory, savedMiniMaxPrompts,
    saveMiniMaxPromptButton, selectedSegmentImagePath, state, syncInspector, timelinePromptSave,
    twoPassControls, useSceneMiniMaxH3Settings, wizardVideoSettings,
    storyboardReferenceDataForSegment: (...args) => storyboardReferenceDataForSegment(...args),
    previousAutoChainSourceSegment: (...args) => previousAutoChainSourceSegment(...args),
    normalizeLyricCueMapForSegment: (...args) => normalizeLyricCueMapForSegment(...args),
    selectedPerformerSubjectsForSegment: (...args) => selectedPerformerSubjectsForSegment(...args),
    singerCueRelativePlayheadTime: (...args) => singerCueRelativePlayheadTime(...args),
    autoTimeMiniMaxSingerCuesForSegment: (...args) => autoTimeMiniMaxSingerCuesForSegment(...args),
    playSingerCueRange: (...args) => playSingerCueRange(...args),
    prepareSceneAudioClipForTimestamping: (...args) => prepareSceneAudioClipForTimestamping(...args),
    miniMaxH3PromptCharacterBudget: (...args) => miniMaxH3PromptCharacterBudget(...args),
    miniMaxPromptReferenceMismatch: (...args) => miniMaxPromptReferenceMismatch(...args),
    render: (...args) => render(...args),
    activateGlobalTimelineAudioPlayback: (...args) => activateGlobalTimelineAudioPlayback(...args),
    activeSegment: (...args) => activeSegment(...args),
    allEditableSegments: (...args) => allEditableSegments(...args),
    sceneSlotNumber: (...args) => sceneSlotNumber(...args),
    segmentTrack: (...args) => segmentTrack(...args),
    startSilentTimelinePlayback: (...args) => startSilentTimelinePlayback(...args),
    syncTimelineTrimModeButton: (...args) => syncTimelineTrimModeButton(...args),
    updatePlayPauseButton: (...args) => updatePlayPauseButton(...args),
    videoSettingsSegment: (...args) => videoSettingsSegment(...args),
    miniMaxDesiredReferenceKeysForSegment: (...args) => miniMaxDesiredReferenceKeysForSegment(...args),
    miniMaxH3ReferenceCapacityStatus: (...args) => miniMaxH3ReferenceCapacityStatus(...args),
    miniMaxOrderedImageReferenceItemsForSegment: (...args) => miniMaxOrderedImageReferenceItemsForSegment(...args),
    segmentImageSource: (...args) => segmentImageSource(...args),
    syncProjectVideoEngineUI: (...args) => syncProjectVideoEngineUI(...args),
  });

  const {
    builderStorySourcePath, clearFinalPromptList, countPromptFindReplaceMatches, createStoryScenesFromSource,
    editContextTextFile, editFinalPromptList, loadBuilderStorySource, locationExtractionStyleTheme,
    projectContextPath, projectPromptsPath, projectReferenceBuilderLocationsPath, projectSceneNotesPath,
    referenceBuilderSubjectLocationText, reloadFinalPromptList, replacePromptPhraseAcrossScenes,
    saveBuilderStorySource, saveGemmaJunkDebug, segmentPromptForEdit, setSegmentPromptForEdit,
  } = createProjectFiles({
    autoSaveSessionQuiet, createProgressWindow, ernieGemmaModelSelect, ernieTextGemmaModelSelect,
    fluxGemmaModelSelect, gemmaModelSelect, gemmaRunnerLine, i2vGemmaModelSelect, i2vTextGemmaModelSelect,
    projectInput, pushHistory, state, storyIdeaInput, syncInspector, t2iTextGemmaModelSelect, themeStyleInput,
    zEnhanceGemmaModelSelect,
    saveSession: (...args) => saveSession(...args),
    currentVideoMode: (...args) => currentVideoMode(...args),
    sceneDisplayName: (...args) => sceneDisplayName(...args),
    videoModeDisplayLabel: (...args) => videoModeDisplayLabel(...args),
    sceneVideoConceptPromptText: (...args) => sceneVideoConceptPromptText(...args),
    render: (...args) => render(...args),
    activeSegment: (...args) => activeSegment(...args),
    allEditableSegments: (...args) => allEditableSegments(...args),
    segmentIndexInfo: (...args) => segmentIndexInfo(...args),
    syncI2VMotionJsonFromSegments: (...args) => syncI2VMotionJsonFromSegments(...args),
    syncPromptJsonFromSegments: (...args) => syncPromptJsonFromSegments(...args),
  });

  const {
    buildI2VPromptRequestForSegment, createT2IPromptWithGemma, editCurrentImagePromptWithGemma,
    editCurrentVideoPromptWithGemma, finalizeVideoPromptDraftOnly, finalizeVideoPromptForSegment,
    generateEnhancePromptWithGemma, getI2VImageReference, openBuilderInstructionEditor,
    runVideoPromptEnhancementBatch, upscaleEnhanceImage,
  } = createPromptEditing({
    autoSaveSessionQuiet, createProgressWindow, createT2IButton, editI2VPromptButton, editImagePromptButtons,
    ernieGemmaModelSelect, ernieMmprojSelect, ernieTextGemmaModelSelect, firstLastFramePromptReferences,
    flfGemmaContextMode, flfGemmaSceneConcept, flfGemmaVisualNotes, flfTransitionLoraActive,
    fluxGemmaModelSelect, fluxMmprojSelect, gemmaModelSelect, gemmaRunnerLine, i2vGemmaModelSelect,
    i2vMmprojSelect, i2vPrompt, i2vTextGemmaModelSelect, llmApiVisionModelSelected, mmprojSelect,
    nbGemmaModelSelect, nbMmprojSelect, pushHistory, rtvReferenceBehaviorForSegment, saveGemmaJunkDebug,
    segmentPromptForEdit, state, syncInspector, syncPreview, t2iTextGemmaModelSelect, textGemmaRunnerPayload,
    zEnhanceButton, zEnhanceGemmaButton, zEnhanceGemmaModelSelect, zEnhanceGemmaNotes, zEnhanceMmprojSelect,
    zEnhancePromptPreview,
    activeProjectFolderForSave: (...args) => activeProjectFolderForSave(...args),
    currentVideoMode: (...args) => currentVideoMode(...args),
    saveZEnhanceSettingsFromPanel: (...args) => saveZEnhanceSettingsFromPanel(...args),
    syncVideoModePanel: (...args) => syncVideoModePanel(...args),
    updateActiveFromInputs: (...args) => updateActiveFromInputs(...args),
    activeScenePromptForEnhance: (...args) => activeScenePromptForEnhance(...args),
    imageModeDisplayLabel: (...args) => imageModeDisplayLabel(...args),
    sceneDisplayName: (...args) => sceneDisplayName(...args),
    videoModeDisplayLabel: (...args) => videoModeDisplayLabel(...args),
    ltx25SelectedCastCoverageContract: (...args) => ltx25SelectedCastCoverageContract(...args),
    segmentMappedLocationText: (...args) => segmentMappedLocationText(...args),
    segmentMappedSubjectText: (...args) => segmentMappedSubjectText(...args),
    applyMappedTriggerPhrases: (...args) => applyMappedTriggerPhrases(...args),
    archiveGeneratedSceneImage: (...args) => archiveGeneratedSceneImage(...args),
    assertBatchNotStopped: (...args) => assertBatchNotStopped(...args),
    generateT2IPromptForSegment: (...args) => generateT2IPromptForSegment(...args),
    requireActiveSegment: (...args) => requireActiveSegment(...args),
    sceneVideoConceptPromptText: (...args) => sceneVideoConceptPromptText(...args),
    currentEnhanceSource: (...args) => currentEnhanceSource(...args),
    enhanceImageForSegment: (...args) => enhanceImageForSegment(...args),
    nbImageSettingsForSegment: (...args) => nbImageSettingsForSegment(...args),
    render: (...args) => render(...args),
    activeSegment: (...args) => activeSegment(...args),
    sceneSlotNumber: (...args) => sceneSlotNumber(...args),
    segmentIndexInfo: (...args) => segmentIndexInfo(...args),
    idLoraSceneContext: (...args) => idLoraSceneContext(...args),
    fluxReferenceContextForSegment: (...args) => fluxReferenceContextForSegment(...args),
    segmentImageSource: (...args) => segmentImageSource(...args),
    applyImageTriggerToPrompt: (...args) => applyImageTriggerToPrompt(...args),
    applyVocalDirectiveToVideoPrompt: (...args) => applyVocalDirectiveToVideoPrompt(...args),
    effectiveVideoPerformanceModeForSegment: (...args) => effectiveVideoPerformanceModeForSegment(...args),
    facialPerformanceNoteForSegment: (...args) => facialPerformanceNoteForSegment(...args),
    idLoraGemmaNotesForSegment: (...args) => idLoraGemmaNotesForSegment(...args),
    idLoraSpeechTextForSegment: (...args) => idLoraSpeechTextForSegment(...args),
    storyboardVideoExtraNotesForSegment: (...args) => storyboardVideoExtraNotesForSegment(...args),
    syncSegmentT2IPrompt: (...args) => syncSegmentT2IPrompt(...args),
    videoGemmaNotesForSegment: (...args) => videoGemmaNotesForSegment(...args),
    videoTriggerPhraseForSegment: (...args) => videoTriggerPhraseForSegment(...args),
  });

  const { openStoryboardBuilderFromProject } = createStoryboardBridge({
    autoSaveSessionQuiet, buildI2VPromptRequestForSegment, finalizeVideoPromptForSegment, gemmaRunnerLine,
    getI2VImageReference, i2vMmprojSelect, llmApiVisionModelSelected, miniMaxH3ModeForSegment,
    miniMaxH3SceneImageUseForSegment, miniMaxH3SettingsForSegment, mmprojSelect, projectInput, pushHistory,
    selectedSegmentImagePath, setSegmentPromptForEdit, state, storyboardPipeline, syncInspector,
    textGemmaRunnerPayload,
    activeProjectFolderForSave: (...args) => activeProjectFolderForSave(...args),
    saveSession: (...args) => saveSession(...args),
    currentVideoMode: (...args) => currentVideoMode(...args),
    saveI2VVideoSettingsFromPanel: (...args) => saveI2VVideoSettingsFromPanel(...args),
    syncI2VVideoSettingsPanel: (...args) => syncI2VVideoSettingsPanel(...args),
    syncVideoModePanel: (...args) => syncVideoModePanel(...args),
    updateActiveFromInputs: (...args) => updateActiveFromInputs(...args),
    imageModeDisplayLabel: (...args) => imageModeDisplayLabel(...args),
    sceneDisplayName: (...args) => sceneDisplayName(...args),
    storyboardScenePayload: (...args) => storyboardScenePayload(...args),
    miniMaxH3CutPlanForSegment: (...args) => miniMaxH3CutPlanForSegment(...args),
    applyMappedTriggerPhrases: (...args) => applyMappedTriggerPhrases(...args),
    applyMiniMaxH3NativeVoiceBlock: (...args) => applyMiniMaxH3NativeVoiceBlock(...args),
    ensureAutoTimedSingerCuesBeforePrompt: (...args) => ensureAutoTimedSingerCuesBeforePrompt(...args),
    ensureBuilderManagedFx: (...args) => ensureBuilderManagedFx(...args),
    miniMaxH3PromptVisionImages: (...args) => miniMaxH3PromptVisionImages(...args),
    miniMaxH3PromptVisionImagesForRunner: (...args) => miniMaxH3PromptVisionImagesForRunner(...args),
    runMiniMaxH3PromptGeneration: (...args) => runMiniMaxH3PromptGeneration(...args),
    render: (...args) => render(...args),
    activeSegment: (...args) => activeSegment(...args),
    allEditableSegments: (...args) => allEditableSegments(...args),
    ensureAllSegmentRuntimeFields: (...args) => ensureAllSegmentRuntimeFields(...args),
    ensureSegmentRuntimeFields: (...args) => ensureSegmentRuntimeFields(...args),
    segmentIndexInfo: (...args) => segmentIndexInfo(...args),
    timelineDuration: (...args) => timelineDuration(...args),
    storyboardReferenceBuilderWithIdLoraRefs: (...args) => storyboardReferenceBuilderWithIdLoraRefs(...args),
    assertMiniMaxH3ReferenceDescriptionsReady: (...args) => assertMiniMaxH3ReferenceDescriptionsReady(...args),
    miniMaxOrderedImageReferenceItemsForSegment: (...args) => miniMaxOrderedImageReferenceItemsForSegment(...args),
    segmentImageSource: (...args) => segmentImageSource(...args),
    effectiveVideoPerformanceModeForSegment: (...args) => effectiveVideoPerformanceModeForSegment(...args),
    videoTriggerPhraseForSegment: (...args) => videoTriggerPhraseForSegment(...args),
  });

  const {
    baseSceneVideoTrimKind, chooseRenderedSceneTrimAtPlayhead, closeBaseTimelineGap,
    closeTimelineGapsFromMenu, loadDirtyLatentBadges, openAudioContextMenu, openDirectorNoteContextMenu,
    openSceneOptions, openSegmentContextMenu, openSnapSceneEdgeMenu, openTimelineSceneCard,
    snapAllSceneStartsToNearestBeats, snapSceneEdgeToNearestBeat,
  } = createTimelineEdit({
    autoSaveSessionQuiet, createProgressWindow, miniMaxH3ContinuityModeForSegment,
    miniMaxH3SettingsForSegment, openLyricReviewModal, openStoryboardBuilderFromProject, projectInput,
    pushHistory, setActiveSegment, setGlobalPlaybackTime, state, syncInspector, syncPreview,
    updateHistoryButtons,
    normalizeSegments: (...args) => normalizeSegments(...args),
    renderSegments: (...args) => renderSegments(...args),
    currentVideoMode: (...args) => currentVideoMode(...args),
    collectedSceneVideoFolder: (...args) => collectedSceneVideoFolder(...args),
    sceneDisplayName: (...args) => sceneDisplayName(...args),
    render: (...args) => render(...args),
    renderAllScenes: (...args) => renderAllScenes(...args),
    stitchPreviewFromSegments: (...args) => stitchPreviewFromSegments(...args),
    isSegmentMultiSelected: (...args) => isSegmentMultiSelected(...args),
    selectedSegmentsForBatch: (...args) => selectedSegmentsForBatch(...args),
    reloadBeatMarkersFromAudio: (...args) => reloadBeatMarkersFromAudio(...args),
    setBeatMarkersVisible: (...args) => setBeatMarkersVisible(...args),
    activeSegment: (...args) => activeSegment(...args),
    allEditableSegments: (...args) => allEditableSegments(...args),
    clampTimelineMarkerToNonOverlap: (...args) => clampTimelineMarkerToNonOverlap(...args),
    currentGlobalTime: (...args) => currentGlobalTime(...args),
    loadedGlobalAudioDuration: (...args) => loadedGlobalAudioDuration(...args),
    nextOverlaySlotNumber: (...args) => nextOverlaySlotNumber(...args),
    pauseTimelineForEditing: (...args) => pauseTimelineForEditing(...args),
    sceneSlotNumber: (...args) => sceneSlotNumber(...args),
    segmentIndexInfo: (...args) => segmentIndexInfo(...args),
    segmentTrack: (...args) => segmentTrack(...args),
    syncI2VMotionJsonFromSegments: (...args) => syncI2VMotionJsonFromSegments(...args),
    syncPromptJsonFromSegments: (...args) => syncPromptJsonFromSegments(...args),
    deleteSegment: (...args) => deleteSegment(...args),
  });

  const {
    drawWaveform, normalizeSegments, openTimelineMarkerEditor, renderList, renderSegments,
    snapAddedSegmentEndToNearestBeat, snapTimeToBeat,
  } = createTimelineView({
    appendTimelineFirstLastFrameThumbnail, autoSaveSessionQuiet, enableLutDrop, enablePostEffectDrop,
    i2vNotesInput, lyricTextInput, mediaThumbnailHtml, openAudioContextMenu, openDirectorNoteContextMenu,
    openSceneOptions, openSegmentContextMenu, openTimelineSceneCard, playhead, pushHistory,
    rtvReferenceBehaviorForSegment, sceneListPane, segmentLayer, selectedSegmentImageThumbnailPath,
    setActiveSegment, state, syncInspector, timelineCanvas, locationThumbnailButton,
    currentVideoMode: (...args) => currentVideoMode(...args),
    timelineSegmentLabel: (...args) => timelineSegmentLabel(...args),
    cycleSegmentImageHistory: (...args) => cycleSegmentImageHistory(...args),
    cycleSegmentVideoHistory: (...args) => cycleSegmentVideoHistory(...args),
    toggleSegmentPreviewMode: (...args) => toggleSegmentPreviewMode(...args),
    render: (...args) => render(...args),
    activeSegment: (...args) => activeSegment(...args),
    clampTimelineMarkerToNonOverlap: (...args) => clampTimelineMarkerToNonOverlap(...args),
    currentGlobalTime: (...args) => currentGlobalTime(...args),
    currentProjectAudioPath: (...args) => currentProjectAudioPath(...args),
    ensureAllSegmentRuntimeFields: (...args) => ensureAllSegmentRuntimeFields(...args),
    handleSegmentPick: (...args) => handleSegmentPick(...args),
    isSegmentMultiSelected: (...args) => isSegmentMultiSelected(...args),
    loadedGlobalAudioDuration: (...args) => loadedGlobalAudioDuration(...args),
    markerVisualEnd: (...args) => markerVisualEnd(...args),
    segmentTrack: (...args) => segmentTrack(...args),
    selectedTimelineRangeInfo: (...args) => selectedTimelineRangeInfo(...args),
    timelineDuration: (...args) => timelineDuration(...args),
    syncLyricMapperFromSegments: (...args) => syncLyricMapperFromSegments(...args),
    enableImageDrop: (...args) => enableImageDrop(...args),
    makeDragHandle: (...args) => makeDragHandle(...args),
    segmentImageSource: (...args) => segmentImageSource(...args),
  });

  const {
    applyRTVReferenceBehaviorToAll, currentVideoMode, rtvReferenceBehaviorGlobalValue,
    saveI2VVideoSettingsFromPanel, saveZEnhanceSettingsFromPanel, syncI2VVideoModelPickerVisibility,
    syncI2VVideoSettingsPanel, syncRTVSceneImageAnchorPanel, syncVideoModePanel, syncZEnhanceSettingsPanel,
    updateActiveFromInputs,
  } = createVideoSettingsPanel({
    clearSceneEndFrameButton, createI2VButton, createSceneEndFrameButton, editI2VInstructionsButton,
    editI2VPromptButton, editIdLoraInstructionsButton, editImagePromptButtons,
    editIngredientsInstructionsButton, editRTVInstructionsButton, editT2VInstructionsButton, endInput,
    ernieNotesInput, ernieRefImagePanel, ernieT2IPrompt, ernieUseVisionReference,
    finalizeSceneFLFPromptButton, firstLastFrameEndImageSource, firstLastFramePreviewGrid,
    firstLastFramePreviewPanel, firstLastFrameStartImageSource, firstLastFrameStatus, firstLastFrameVideoCard,
    flfChainedSettingsPanel, flfChainingEnabled, flfChainPreviousEndFrame, flfColorMatchFadeInput,
    flfColorMatchStrengthInput, flfCustomEndDirection, flfEndpointModeSelect, flfEndpointPromptPreview,
    flfFirstAttentionStrength, flfFirstGuideBlurInput, flfFirstGuideCrfInput, flfFirstGuideCrop,
    flfFirstGuideFrameIndexInput, flfFirstGuideInterpolation, flfFirstGuideStrengthInput,
    flfGemmaContextModeSelect, flfGlobalTransitionTypeSelect, flfGuideSettingsSection,
    flfLastAttentionStrength, flfLastGuideBlurInput, flfLastGuideCrfInput, flfLastGuideCrop,
    flfLastGuideFrameIndexInput, flfLastGuideInterpolation, flfLastGuideStrengthInput,
    flfMatchPreviousClipColor, flfMotionPlanPreview, flfPerScenePlanner, flfPreGeneratePromptsFromSceneImages,
    flfRenderChainSourceSelect, flfStructureModeSelect, flfTransitionTypeField, flfTransitionTypeSelect,
    flowGptPrompt, fluxPrompt, i2vAudioVaePicker, i2vClip1Picker, i2vClip2Picker, i2vDiffusionLoaderAdvanced,
    i2vDiffusionModelField, i2vDiffusionModelPicker, i2vEnableFp16Accumulation, i2vFpsInput, i2vHeightInput,
    i2vLoraCount, i2vLoraSlots, i2vNotesInput, i2vPass1Bypass, i2vPass1SamplerSelect, i2vPass1SigmasInput,
    i2vPass1StrengthInput, i2vPass2Bypass, i2vPass2NodePanel, i2vPass2SamplerSelect, i2vPass2SigmasInput,
    i2vPass2StrengthInput, i2vPreFramesInput, i2vPrompt, i2vReferenceNote, i2vSeedInput, i2vSettingsGrid,
    i2vTailLossFramesInput, i2vUnetModelField, i2vUnetPicker, i2vUpscalePicker, i2vUseGgufModel, i2vUseLora,
    i2vUseSageAttention, i2vVaePicker, i2vWarmCooldownSection, i2vWidthInput, idLoraIdentityScaleInput,
    idLoraReferenceAudioInput, idLoraVideoCard, idLoraVoiceSettingsSection, imageToVideoCard,
    importCustomVideoCard, importCustomVideoPanel, ingredientsToVideoCard, krea2TwoPassNotesInput,
    krea2TwoPassRefImagePanel, krea2TwoPassT2IPrompt, krea2TwoPassUseVisionReference, labelInput,
    loadFirstLastFrameEndFile, ltx25AspectRatioSelect, ltx25MegapixelsInput, ltx25ResolutionGrid,
    ltxIdLoraFirstPassStrength, ltxIdLoraPicker, ltxIdLoraRequiredPanel, ltxIdLoraSecondPassStrength,
    ltxIngredientsFirstPassStrength, ltxIngredientsLoraPicker, ltxIngredientsRequiredPanel,
    ltxIngredientsResolutionWarning, ltxMsrBackgroundMode, ltxMsrFirstPassStrength, ltxMsrLoraPicker,
    ltxMsrReferenceStrength, ltxMsrRequiredPanel, ltxMsrSecondPassStrength, lyricSingersInput, lyricTextInput,
    nbNotes, nbPrompt, normalizeSegments, notesInput, planSceneEndMotionButton,
    promoteChainedFLFSceneImageToEndFrame, promptRunnerActionName, pushHistory, referenceToVideoCard,
    refImageInput, refImagePanel, rtvReferenceBehaviorField, rtvReferenceBehaviorForSegment,
    rtvReferenceBehaviorNote, rtvReferenceBehaviorSelect, rtvSceneImageAnchorSection, sceneEndFrameFileInput,
    selectedSegmentImageThumbnailPath, startInput, state, syncInspector, t2iPrompt, t2vLocationNote,
    t2vReferenceNote, t2vRefImagePanel, textToVideoCard, useI2VPromptEnhancementPass, useI2VVisionReference,
    useSceneI2VVideoSettings, useT2VVisionReference, useVisionReference, videoSettingsScopeNote, videoSubTabs,
    videoTriggerInput, wizardVideoSettings, zEnhanceAmount, zEnhanceAmountValue, zEnhanceClipPicker,
    zEnhanceGemmaNotes, zEnhanceHeight, zEnhanceLoraCount, zEnhanceLoraPanel, zEnhanceLoraRows,
    zEnhanceLoraSlots, zEnhancePromptPreview, zEnhanceSeed, zEnhanceSeedMode, zEnhanceUnetPicker,
    zEnhanceUseLora, zEnhanceVaePicker, zEnhanceWidth,
    syncI2VAdvancedNodeControls: (...args) => syncI2VAdvancedNodeControls(...args),
    activeScenePromptForEnhance: (...args) => activeScenePromptForEnhance(...args),
    render: (...args) => render(...args),
    activeSegment: (...args) => activeSegment(...args),
    applyVideoSettingsToMultiSelection: (...args) => applyVideoSettingsToMultiSelection(...args),
    hasMultiSceneBatchSelection: (...args) => hasMultiSceneBatchSelection(...args),
    segmentTrack: (...args) => segmentTrack(...args),
    syncTimelineTrimModeButton: (...args) => syncTimelineTrimModeButton(...args),
    videoSettingsSegment: (...args) => videoSettingsSegment(...args),
    segmentImageSource: (...args) => segmentImageSource(...args),
    updateI2VLoraVisibility: (...args) => updateI2VLoraVisibility(...args),
    updateI2VPromptSaveButtonState: (...args) => updateI2VPromptSaveButtonState(...args),
    activeI2VVideoSettings: (...args) => activeI2VVideoSettings(...args),
  });

  const {
    finishSingleSceneETA, openRenderLogModal, persistRenderLog, renderETAScene, resetBuilderETA,
    startBuilderETA, startSingleSceneETA, upsertRenderLog,
  } = createRenderLog({
    builderETA, builderETAState, builderFullETA, builderSceneETA, currentVideoMode, miniMaxH3ModeForSegment,
    miniMaxH3SettingsForSegment, overlay, positionBuilderETA, projectInput, state,
  });

  const {
    advanceErnieSeedAfterRun, advanceKrea2TwoPassSeedAfterRun, advanceZEnhanceSeedAfterRun,
    advanceZImageSeedAfterRun, saveErnieImageSettingsFromPanel, saveFlowGptBrowserSettingsFromPanel,
    saveFluxKleinSettingsFromPanel, saveKrea2TwoPassSettingsFromPanel, saveNBImageSettingsFromPanel,
    saveZImageSettingsFromPanel, setFlowGptProvider, syncErnieImagePanel, syncFlowGptBrowserPanel,
    syncFluxKleinPanel, syncI2VAdvancedNodeControls, syncKrea2TwoPassPanel, syncNBImagePanel,
    syncZImageSettingsPanel,
  } = createImagePanels({
    autoSaveSessionQuiet, browserAiGroupPrompt, currentVideoMode, ernieBatchSize, ernieClipPicker,
    ernieCreateButton, ernieHeight, ernieI2IPanel, ernieI2IPath, ernieI2ISlider, ernieI2IStartStep,
    ernieImageCard, ernieImageModePanel, ernieImagePanel, ernieImageTriggerInput, ernieLoraCount,
    ernieLoraPanel, ernieLoraRows, ernieLoraSlots, ernieSeed, ernieSeedMode, ernieUnetPicker,
    ernieUseImageToImage, ernieUseLora, ernieVaePicker, ernieWidth, flowGptAskPreviousImage,
    flowGptAspectRatio, flowGptAspectRatioField, flowGptCard, flowGptFailureMode, flowGptLoginButton,
    flowGptManualChatPrompt, flowGptModePanel, flowGptPrompt, flowGptRetries, flowGptTimeout,
    flowNanoProviderButton, fluxClipPicker, fluxHeight, fluxImageTriggerInput, fluxKleinCard,
    fluxKleinModePanel, fluxKleinPanel, fluxLoraCount, fluxLoraPanel, fluxLoraRows, fluxLoraSlots, fluxNotes,
    fluxPrompt, fluxSeed, fluxUnetPicker, fluxUseDirectorNotes, fluxUseLora, fluxUseTextOnlyGemmaPrompt,
    fluxVaePicker, fluxWidth, gptImageProviderButton, i2vAdvancedNodeSettingsPanel,
    i2vAdvancedNodeSettingsSection, i2vPass1Bypass, i2vPass1NodePanel, i2vPass1SamplerSelect,
    i2vPass1SigmasInput, i2vPass1StrengthInput, i2vPass1StrengthSlider, i2vPass2Bypass, i2vPass2NodePanel,
    i2vPass2SamplerSelect, i2vPass2SigmasInput, i2vPass2StrengthInput, i2vPass2StrengthSlider,
    imageTriggerInput, krea2TwoPassAspectRatio, krea2TwoPassBatchSize, krea2TwoPassCard, krea2TwoPassCfg,
    krea2TwoPassClipPicker, krea2TwoPassCreateButton, krea2TwoPassCreativity, krea2TwoPassCreativityInput,
    krea2TwoPassI2IPanel, krea2TwoPassI2IPath, krea2TwoPassImageTriggerInput, krea2TwoPassLoraCount,
    krea2TwoPassLoraPanel, krea2TwoPassLoraRows, krea2TwoPassLoraSlots, krea2TwoPassModePanel,
    krea2TwoPassPanel, krea2TwoPassSampler, krea2TwoPassSeed, krea2TwoPassSeedMode, krea2TwoPassUnetPicker,
    krea2TwoPassUseImageToImage, krea2TwoPassUseLora, krea2TwoPassVaePicker, loadCustomImageButton,
    mergedFluxImageIngredients, metaImageProviderButton, nbApiKey, nbImageCard, nbImagePanel, nbModelSelect,
    nbNotes, nbPrompt, nbUseDirectorNotes, nbUseTextOnlyGemmaPrompt, previewButton, pushHistory,
    renderFluxGlobalIngredientList, renderFluxIngredientList, renderList, renderNBIngredientList, state,
    syncFluxGlobalIngredientPanel, useFluxKlein, useSceneErnieImageSettings, useSceneFluxKleinSettings,
    useSceneKrea2TwoPassSettings, useSceneNBImageSettings, useSceneZImageSettings, zBatchSize, zClipPicker,
    zEnhanceCard, zEnhancePanel, zEnhanceSeed, zFirstHeight, zFirstWidth, zI2IPanel, zI2IPath, zI2ISlider,
    zI2IStartStep, zImageCard, zImageModePanel, zLoraCount, zLoraPanel, zLoraRows, zLoraSlots, zSecondHeight,
    zSecondWidth, zSeed, zSeedMode, zUnetPicker, zUseImageToImage, zUseLora, zVaePicker,
    renderBrowserAiReferenceGroups: (...args) => renderBrowserAiReferenceGroups(...args),
    syncFlowGptManualPanel: (...args) => syncFlowGptManualPanel(...args),
    activeSegment: (...args) => activeSegment(...args),
    applyImageSettingsToMultiSelection: (...args) => applyImageSettingsToMultiSelection(...args),
    hasMultiSceneBatchSelection: (...args) => hasMultiSceneBatchSelection(...args),
    fluxReferenceContextForSegment: (...args) => fluxReferenceContextForSegment(...args),
    nbReferenceContextForSegment: (...args) => nbReferenceContextForSegment(...args),
    activeErnieImageSettings: (...args) => activeErnieImageSettings(...args),
    activeFluxKleinSettings: (...args) => activeFluxKleinSettings(...args),
    activeKrea2TwoPassSettings: (...args) => activeKrea2TwoPassSettings(...args),
    activeNBImageSettings: (...args) => activeNBImageSettings(...args),
    activeZImageSettings: (...args) => activeZImageSettings(...args),
  });

  const {
    activeBrowserAiReferenceGroup, activeScenePromptForEnhance, addBrowserAiBandSequenceFiles,
    addBrowserAiGroupFiles, browserAiBandSequenceSelection, chooseBrowserAiImageFiles,
    clearBrowserAiBandSequenceFiles, exportManualFlowGptRefs, finishBrowserAiDownloadSession,
    flfStillEndpointNotes, imagePromptNotesWithDirector, importLatestManualFlowGptDownload,
    openManualFlowGptBrowser, refreshFlowGptBrowserStatus, renderBrowserAiReferenceGroups,
    restoreBrowserAiDownloadsQuietly, sendBrowserAiReferenceGroup, setBrowserAiLocationFile,
    syncFlowGptManualPanel,
  } = createBrowserAi({
    autoSaveSessionQuiet, browserAiAutoAdvanceGroup, browserAiBandSequenceMode, browserAiBandSequencePanel,
    browserAiCustomGroupsPanel, browserAiDeleteGroupButton, browserAiDownloadOverrideProviders,
    browserAiExtrasList, browserAiFinishButton, browserAiGroupList, browserAiGroupPrompt,
    browserAiGroupSelect, browserAiGroupStatus, browserAiLocationList, browserAiLocationsList,
    browserAiMembersList, browserAiSendButton, browserAiSequenceLocationSelect, browserAiSequenceProgress,
    browserAiSequenceSetSelect, browserAiSingerList, flowGptManualAutoAdvance, flowGptManualChatPrompt,
    flowGptManualExportRefsButton, flowGptManualImportLatestButton, flowGptManualMode,
    flowGptManualOpenButton, flowGptManualStatus, flowGptStatusText, moveActiveSceneSelection, projectInput,
    pushHistory, saveFlowGptBrowserSettingsFromPanel, setActiveSegment, state, syncPreview,
    zEnhancePromptPreview,
    sceneDisplayName: (...args) => sceneDisplayName(...args),
    addSceneImageHistoryPath: (...args) => addSceneImageHistoryPath(...args),
    requireActiveSegment: (...args) => requireActiveSegment(...args),
    flowGptBrowserSettingsForSegment: (...args) => flowGptBrowserSettingsForSegment(...args),
    render: (...args) => render(...args),
    activeSegment: (...args) => activeSegment(...args),
    sceneSlotNumber: (...args) => sceneSlotNumber(...args),
    segmentIndexInfo: (...args) => segmentIndexInfo(...args),
  });

  const {
    exportShareableProject, importShareableProject, newProject, openPromptCreatorPanel, resetProjectState,
    saveProjectAs, sendCurrentProjectToPromptCreator,
  } = createProjectActions({
    audio, audioInput, clearHistoryBlobCache, createProgressWindow, ernieImageTriggerInput,
    exportProjectButton, faceFixTool, fluxImageTriggerInput, i2vMotionJsonInput, imageTriggerInput,
    importProjectButton, node, projectInput, promptJsonInput, referenceBuilderSubjectLocationText,
    resetBuilderETA, restoreBrowserAiDownloadsQuietly, saveI2VVideoSettingsFromPanel, sceneAudio,
    sendToPromptCreatorButton, srtInput, state, storyIdeaInput, subjectSceneInput, syncErnieImagePanel,
    syncFluxKleinPanel, syncI2VVideoSettingsPanel, syncInspector, syncKrea2TwoPassPanel, syncVideoModePanel,
    syncVideoTypeControl, syncZEnhanceSettingsPanel, syncZImageSettingsPanel, themeStyleInput,
    updateActiveFromInputs, useVrgdgTextContext, videoTriggerInput,
    autoLoadAll: (...args) => autoLoadAll(...args),
    currentSessionData: (...args) => currentSessionData(...args),
    getPreferredProjectRoot: (...args) => getPreferredProjectRoot(...args),
    loadSessionFromProject: (...args) => loadSessionFromProject(...args),
    rememberLastProject: (...args) => rememberLastProject(...args),
    saveSession: (...args) => saveSession(...args),
    isBlankStarterProject: (...args) => isBlankStarterProject(...args),
    render: (...args) => render(...args),
    closeBeatCalibrationWizard: (...args) => closeBeatCalibrationWizard(...args),
    setBeatMarkersVisible: (...args) => setBeatMarkersVisible(...args),
    pauseAllAudio: (...args) => pauseAllAudio(...args),
    clearTriggerPhrasesForFreshProject: (...args) => clearTriggerPhrasesForFreshProject(...args),
    loadGlobalModelDefaultsQuiet: (...args) => loadGlobalModelDefaultsQuiet(...args),
    syncProjectVideoEngineUI: (...args) => syncProjectVideoEngineUI(...args),
  });

  const {
    activeProjectFolderForSave, addProjectToBatch, autoLoadAll, currentSessionData, getPreferredProjectRoot,
    importI2VMotionJson, importPromptJson, importSceneNotesJson, loadDefaultContextPaths, loadLastProject,
    loadSession, loadSessionFromProject, persistIngredientsSheetImages, projectContextFilesForSessionSave,
    projectListUrl, recoverSceneVideosFromProject, rememberLastProject, renderProjectBatchQueue,
    runClearMemoryWorkflow, runClearMemoryWorkflowQuiet, runImageMemoryCleanupQuiet, runProjectBatchQueue,
    saveSession, saveSessionForSceneVideo, setPreferredProjectRoot, showStartupWelcome, stopCurrentWorkflow,
  } = createSession({
    audio, audioInput, autoLoadAllButton, autoSaveControl, autoSaveSessionQuiet, clearMemoryButton,
    createProgressWindow, faceFixTool, i2vMotionJsonInput, imageContinuityEnabled, imageContinuityStrength,
    loadDirtyLatentBadges, newProject, node, previewVideo, projectBatch, projectBatchAddCurrent,
    projectBatchAddCustom, projectBatchAddRecent, projectBatchAddSession, projectBatchClearMemory,
    projectBatchContinueOnError, projectBatchForceVideos, projectBatchQueue, projectBatchRun,
    projectBatchStatus, projectBatchStop, projectContextPath, projectInput, projectSceneNotesPath,
    promptJsonInput, pushHistory, referenceBuilderSubjectLocationText, resetBuilderETA, resetProjectState,
    restoreBrowserAiDownloadsQuietly, saveI2VVideoSettingsFromPanel, saveMiniMaxH3SettingsFromPanel,
    saveMiniMaxSceneInputsFromPanel, segmentLayer, snapToBeatsControl, srtInput, state, stopWorkflowButton,
    storyIdeaInput, subjectSceneInput, syncErnieImagePanel, syncFluxKleinPanel, syncI2VVideoSettingsPanel,
    syncInspector, syncKrea2TwoPassPanel, syncLeftPanelTabs, syncVideoModePanel, syncVideoTypeControl,
    syncZEnhanceSettingsPanel, syncZImageSettingsPanel, themeStyleInput, updateActiveFromInputs,
    useVrgdgTextContext, waveformModeSelect,
    syncLyricAndSubjectNoteFiles: (...args) => syncLyricAndSubjectNoteFiles(...args),
    loadAudio: (...args) => loadAudio(...args),
    loadSrt: (...args) => loadSrt(...args),
    render: (...args) => render(...args),
    closeBeatCalibrationWizard: (...args) => closeBeatCalibrationWizard(...args),
    setBeatMarkersVisible: (...args) => setBeatMarkersVisible(...args),
    showBeatMarkersIfAvailable: (...args) => showBeatMarkersIfAvailable(...args),
    syncLyricNoteControls: (...args) => syncLyricNoteControls(...args),
    syncSceneNoteControls: (...args) => syncSceneNoteControls(...args),
    syncVideoNoteControls: (...args) => syncVideoNoteControls(...args),
    activateGlobalTimelineAudioPlayback: (...args) => activateGlobalTimelineAudioPlayback(...args),
    allEditableSegments: (...args) => allEditableSegments(...args),
    applyLayoutSizes: (...args) => applyLayoutSizes(...args),
    enforceAudioTimelineEnd: (...args) => enforceAudioTimelineEnd(...args),
    ensureAllSegmentRuntimeFields: (...args) => ensureAllSegmentRuntimeFields(...args),
    ensureSegmentRuntimeFields: (...args) => ensureSegmentRuntimeFields(...args),
    loadedGlobalAudioDuration: (...args) => loadedGlobalAudioDuration(...args),
    pauseAllAudio: (...args) => pauseAllAudio(...args),
    sanitizedSessionSegments: (...args) => sanitizedSessionSegments(...args),
    sceneSlotNumber: (...args) => sceneSlotNumber(...args),
    ingredientsSheetForSegment: (...args) => ingredientsSheetForSegment(...args),
    syncI2VMotionJsonFromSegments: (...args) => syncI2VMotionJsonFromSegments(...args),
    syncPromptJsonFromSegments: (...args) => syncPromptJsonFromSegments(...args),
    renderAllScenes: (...args) => renderAllScenes(...args),
    cloneFlowGptBrowserSettingsForLoadedProject: (...args) => cloneFlowGptBrowserSettingsForLoadedProject(...args),
    syncProjectVideoEngineUI: (...args) => syncProjectVideoEngineUI(...args),
    hasAnyI2VMotionNotes: (...args) => hasAnyI2VMotionNotes(...args),
  });

  const {
    activeVideoOutputFolder, collectedSceneVideoFolder, i2vImagesFolder, i2vVideoSettingsForSegment,
    i2vVideoSettingsPayload, imageModeDisplayLabel, saveTimelinePrompt, sceneDisplayName,
    sceneVideoDetailsHtml, storyboardReferenceDataForSegment, storyboardScenePayload, timelineSegmentLabel,
    videoModeDisplayLabel, wizardStoryboardState,
  } = createSceneOutput({
    activeProjectFolderForSave, audioInput, currentVideoMode, firstLastFrameStartImageSource,
    getI2VImageReference, i2vPrompt, miniMaxH3ModeForSegment, miniMaxH3SettingsForSegment, miniMaxPrompt,
    projectInput, savedI2VPrompts, savedMiniMaxPrompts, saveI2VPromptButton, saveI2VVideoSettingsFromPanel,
    saveMiniMaxPromptButton, saveSession, selectedSegmentImagePath, state, timelinePromptSave,
    updateMiniMaxPromptSaveButtonState,
    miniMaxH3VocalCueMapText: (...args) => miniMaxH3VocalCueMapText(...args),
    normalizeLyricCueMapForSegment: (...args) => normalizeLyricCueMapForSegment(...args),
    selectedPerformerSubjectsForSegment: (...args) => selectedPerformerSubjectsForSegment(...args),
    activeSegment: (...args) => activeSegment(...args),
    allEditableSegments: (...args) => allEditableSegments(...args),
    sceneSlotNumber: (...args) => sceneSlotNumber(...args),
    segmentIndexInfo: (...args) => segmentIndexInfo(...args),
    logicalExtraSubjectsForScene: (...args) => logicalExtraSubjectsForScene(...args),
    logicalReferenceSubjects: (...args) => logicalReferenceSubjects(...args),
    logicalSubjectIdsForScene: (...args) => logicalSubjectIdsForScene(...args),
    sceneReferenceMapValue: (...args) => sceneReferenceMapValue(...args),
    storyboardReferenceBuilderWithIdLoraRefs: (...args) => storyboardReferenceBuilderWithIdLoraRefs(...args),
    segmentImageSource: (...args) => segmentImageSource(...args),
    updateI2VPromptSaveButtonState: (...args) => updateI2VPromptSaveButtonState(...args),
    effectiveVideoPerformanceModeForSegment: (...args) => effectiveVideoPerformanceModeForSegment(...args),
  });

  const {
    buildShotAlignedSingerCueMap, formatLyricSegmentText, ltx25SelectedCastCoverageContract,
    miniMaxH3CueShotContractText, miniMaxH3CutPlanForSegment, miniMaxH3SubjectLabelMapForSegment,
    miniMaxH3VocalCueMapText, normalizeLyricCueMapForSegment, segmentMappedLocationReference,
    segmentMappedLocationText, segmentMappedSubjectText, sceneCastGuardForSegment, selectedCastCoverageContract,
    selectedPerformerSubjectsForSegment, singerCueRelativePlayheadTime, syncPerformerInspectorForSegment,
  } = createLyricCues({
    i2vVideoSettingsForSegment, isMiniMaxBuiltInSpeakerAssignmentMode, isMiniMaxSingerAssignmentMode,
    lyricSingersInput, miniMaxH3ModeForSegment, state,
    activeSegment: (...args) => activeSegment(...args),
    allEditableSegments: (...args) => allEditableSegments(...args),
    currentGlobalTime: (...args) => currentGlobalTime(...args),
    segmentIndexInfo: (...args) => segmentIndexInfo(...args),
    logicalExtraSubjectsForScene: (...args) => logicalExtraSubjectsForScene(...args),
    logicalReferenceSubjects: (...args) => logicalReferenceSubjects(...args),
    logicalSubjectIdsForScene: (...args) => logicalSubjectIdsForScene(...args),
    sceneReferenceMapArray: (...args) => sceneReferenceMapArray(...args),
    sceneReferenceMapValue: (...args) => sceneReferenceMapValue(...args),
    miniMaxOrderedImageReferenceItemsForSegment: (...args) => miniMaxOrderedImageReferenceItemsForSegment(...args),
    miniMaxH3FrameContinuityPromptEnabled: (...args) => miniMaxH3FrameContinuityPromptEnabled(...args),
  });

  const {
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
  } = createSceneRenderPrep({
    activeProjectFolderForSave, audioInput, autoSaveSessionQuiet, currentVideoMode,
    finalizeVideoPromptForSegment, firstLastFrameEndImageSource, firstLastFrameResolvedEndImageSource,
    firstLastFrameStartImageSource, flfChainingEnabled, flfRenderChainStartSource, gemmaRunnerLine,
    hasFirstLastFrameEndImage, i2vGemmaModelSelect, i2vMmprojSelect, i2vPrompt, i2vTextGemmaModelSelect,
    imageModeDisplayLabel, miniMaxH3ContinuityModeForSegment, miniMaxH3ContinuityReferenceReserved,
    miniMaxH3ModeForSegment, miniMaxH3SettingsForSegment, node, normalizeSceneAdjust, projectInput,
    pushHistory, rtvReferenceBehaviorForSegment, rtvReferenceBehaviorGlobalValue,
    sceneAdjustHasRenderableChanges, sceneAdjustSignature, sceneDisplayName, segmentMappedLocationText,
    segmentMappedSubjectText, selectedSegmentImagePath, srtInput, state, storyboardReferenceDataForSegment,
    syncErnieImagePanel, syncI2VVideoSettingsPanel, syncInspector, syncKrea2TwoPassPanel, syncPreview,
    syncRTVSceneImageAnchorPanel, syncZImageSettingsPanel, textGemmaRunnerPayload,
    flfSameLocationCameraDiversityDirection: (...args) => flfSameLocationCameraDiversityDirection(...args),
    addSceneImageHistoryPath: (...args) => addSceneImageHistoryPath(...args),
    generateT2IPromptForSegment: (...args) => generateT2IPromptForSegment(...args),
    createErnieImageForSegment: (...args) => createErnieImageForSegment(...args),
    createFlowGptImageForSegment: (...args) => createFlowGptImageForSegment(...args),
    createFluxKleinImageForSegment: (...args) => createFluxKleinImageForSegment(...args),
    createKrea2TwoPassImageForSegment: (...args) => createKrea2TwoPassImageForSegment(...args),
    createZImageForSegment: (...args) => createZImageForSegment(...args),
    generateFluxKleinPromptForSegment: (...args) => generateFluxKleinPromptForSegment(...args),
    generateNBPromptForSegment: (...args) => generateNBPromptForSegment(...args),
    generateI2VPromptForSegment: (...args) => generateI2VPromptForSegment(...args),
    miniMaxBatchReferenceProblems: (...args) => miniMaxBatchReferenceProblems(...args),
    createSilentTimelineAudioForDuration: (...args) => createSilentTimelineAudioForDuration(...args),
    render: (...args) => render(...args),
    activeSegment: (...args) => activeSegment(...args),
    allEditableSegments: (...args) => allEditableSegments(...args),
    batchTargetItems: (...args) => batchTargetItems(...args),
    currentProjectAudioPath: (...args) => currentProjectAudioPath(...args),
    ensureSegmentRuntimeFields: (...args) => ensureSegmentRuntimeFields(...args),
    sceneSlotNumber: (...args) => sceneSlotNumber(...args),
    segmentIndexInfo: (...args) => segmentIndexInfo(...args),
    segmentTrack: (...args) => segmentTrack(...args),
    timelineDuration: (...args) => timelineDuration(...args),
    usingSceneAudioMode: (...args) => usingSceneAudioMode(...args),
    applyIngredientsSheetForSceneIfMapped: (...args) => applyIngredientsSheetForSceneIfMapped(...args),
    idLoraSceneContext: (...args) => idLoraSceneContext(...args),
    miniMaxOrderedImageReferenceItemsForSegment: (...args) => miniMaxOrderedImageReferenceItemsForSegment(...args),
    referenceBuilderSubjectItemsForSegment: (...args) => referenceBuilderSubjectItemsForSegment(...args),
    rtvReferencesForSegment: (...args) => rtvReferencesForSegment(...args),
    segmentImageSource: (...args) => segmentImageSource(...args),
    miniMaxH3FrameContinuityPromptEnabled: (...args) => miniMaxH3FrameContinuityPromptEnabled(...args),
    createNBImageForSegmentWithRetry: (...args) => createNBImageForSegmentWithRetry(...args),
    activeI2VVideoSettings: (...args) => activeI2VVideoSettings(...args),
    activeZImageSettings: (...args) => activeZImageSettings(...args),
    effectiveVideoPerformanceModeForSegment: (...args) => effectiveVideoPerformanceModeForSegment(...args),
    i2vAutoChainEnabled: (...args) => i2vAutoChainEnabled(...args),
    imageModeImg2ImgContinuityLabel: (...args) => imageModeImg2ImgContinuityLabel(...args),
    imageModeSupportsImg2ImgContinuity: (...args) => imageModeSupportsImg2ImgContinuity(...args),
    img2imgContinuityEnabled: (...args) => img2imgContinuityEnabled(...args),
    syncSegmentFlowGptPrompt: (...args) => syncSegmentFlowGptPrompt(...args),
    syncSegmentT2IPrompt: (...args) => syncSegmentT2IPrompt(...args),
    videoGemmaNotesForSegment: (...args) => videoGemmaNotesForSegment(...args),
  });

  const {
    applyMappedTriggerPhrases, autoTimeAllMiniMaxSingerScenes, autoTimeMiniMaxSingerCuesForSegment,
    createScenesFromTimestampedLyrics, flfSameLocationCameraDiversityDirection, playSingerCueRange,
    prepareSceneAudioClipForTimestamping, projectLyricNotesPath, syncLyricAndSubjectNoteFiles,
    transcribeExistingScenesWithOptions, transcribeLyricsForTimeline,
  } = createLyricTranscription({
    activeProjectFolderForSave, audio, audioInput, autoSaveSessionQuiet, buildShotAlignedSingerCueMap,
    createProgressWindow, currentVideoMode, flfTransitionLoraActive, formatLyricSegmentText,
    miniMaxAutoTimeAllScenesButton, normalizeLyricCueMapForSegment, previousAutoChainSourceSegment,
    projectInput, projectPromptsPath, pushHistory, renderMiniMaxSpeakerAssignmentPanel, saveSession,
    sceneDisplayName, segmentMappedLocationReference, segmentMappedLocationText,
    selectedPerformerSubjectsForSegment, state, syncInspector, updateAudioScrubbers,
    ensureAutoTimedSingerCuesBeforePrompt: (...args) => ensureAutoTimedSingerCuesBeforePrompt(...args),
    render: (...args) => render(...args),
    syncLyricNoteControls: (...args) => syncLyricNoteControls(...args),
    activateGlobalTimelineAudioPlayback: (...args) => activateGlobalTimelineAudioPlayback(...args),
    allEditableSegments: (...args) => allEditableSegments(...args),
    currentProjectAudioPath: (...args) => currentProjectAudioPath(...args),
    ensureSegmentRuntimeFields: (...args) => ensureSegmentRuntimeFields(...args),
    pauseAllAudio: (...args) => pauseAllAudio(...args),
    sceneSlotNumber: (...args) => sceneSlotNumber(...args),
    segmentIndexInfo: (...args) => segmentIndexInfo(...args),
    startSilentTimelinePlayback: (...args) => startSilentTimelinePlayback(...args),
    updatePlayPauseButton: (...args) => updatePlayPauseButton(...args),
    applyLyricMapperToSegments: (...args) => applyLyricMapperToSegments(...args),
    logicalSubjectIdsForScene: (...args) => logicalSubjectIdsForScene(...args),
  });

  const {
    chooseProjectAudioFile, chooseProjectSrtFile, createSilentTimelineAudio,
    createSilentTimelineAudioForDuration, loadAudio, loadSrt, openSettingsModal, render,
  } = createProjectSetup({
    activeProjectFolderForSave, audio, audioInput, autoSaveSessionQuiet, createSilentTimelineAudioButton,
    drawWaveform, freezeTimingControl, getPreferredProjectRoot, globalScrub, loadButton, loadSrtButton,
    lutsTools, node, pickAudioButton, pickSrtButton, playBuilderNotification, projectInput,
    projectLyricNotesPath, pushHistory, renderList, renderSegments, saveSession, setPreferredProjectRoot,
    settingsModalControls, silentAudioDurationInput, srtInput, state, syncI2VVideoSettingsPanel,
    syncInspector, syncLyricAndSubjectNoteFiles, timelineInfo, timelineRangeInfo, updateSelectedMediaTools,
    showBeatMarkersIfAvailable: (...args) => showBeatMarkersIfAvailable(...args),
    syncLyricNoteControls: (...args) => syncLyricNoteControls(...args),
    activateGlobalTimelineAudioPlayback: (...args) => activateGlobalTimelineAudioPlayback(...args),
    allEditableSegments: (...args) => allEditableSegments(...args),
    enforceAudioTimelineEnd: (...args) => enforceAudioTimelineEnd(...args),
    loadedGlobalAudioDuration: (...args) => loadedGlobalAudioDuration(...args),
    normalizeImportedSrtSegments: (...args) => normalizeImportedSrtSegments(...args),
    selectedTimelineRangeInfo: (...args) => selectedTimelineRangeInfo(...args),
    timelineDuration: (...args) => timelineDuration(...args),
    updateMultiSelectButton: (...args) => updateMultiSelectButton(...args),
    syncOverlayTrackControls: (...args) => syncOverlayTrackControls(...args),
    refreshGemmaChoices: (...args) => refreshGemmaChoices(...args),
    refreshLoraChoices: (...args) => refreshLoraChoices(...args),
    refreshModelChoices: (...args) => refreshModelChoices(...args),
    syncProjectVideoEngineUI: (...args) => syncProjectVideoEngineUI(...args),
  });

  const {
    activateGlobalTimelineAudioPlayback, activeSegment, allEditableSegments,
    applyImageSettingsToMultiSelection, applyLayoutSizes, applyVideoSettingsToMultiSelection,
    audioSourceDurationForScene, batchScopeChoices, batchTargetItems, clampTimelineMarkerToNonOverlap,
    currentGlobalTime, currentProjectAudioPath, enforceAudioTimelineEnd, ensureAllSegmentRuntimeFields,
    ensureGlobalTimelineAudioSource, ensureSegmentRuntimeFields, handleSegmentPick,
    hasMultiSceneBatchSelection, isSegmentMultiSelected, isTimelinePlaying, loadedGlobalAudioDuration,
    makePanelResize, markerVisualEnd, nextFreeTimelineMarkerRange, nextOverlaySlotNumber,
    normalizeImportedSrtSegments, pauseAllAudio, pauseTimelineForEditing, playbackDuration,
    sanitizedSessionSegments, sceneSlotNumber, seekAudioWhenReady, segmentIndexInfo, segmentTrack,
    selectedSegmentsForBatch, selectedTimelineRangeInfo, setGlobalTimelineAudioMuted, setTimelineZoom,
    startSilentTimelinePlayback, stopSilentTimelinePlayback, syncTimelineTrimModeButton,
    timelineAudioPathForSegment, timelineAudioSegmentAtTime, timelineAudioSourceStartForSegment,
    timelineDuration, updateMultiSelectButton, updatePlayPauseButton, usingSceneAudioMode,
    usingSceneAudioPlaybackMode, videoSettingsSegment,
  } = createTimelineState({
    audio, audioInput, autoSaveSessionQuiet, cancelPreviewPlayStart, currentVideoMode, drawWaveform,
    idLoraTrimModeButton, inspector, leftPanelToggle, leftResizeHandle, main, miniMaxH3SettingsForSegment,
    multiSelectButton, playButton, previewVideo, render, renderSegments, rightPanelToggle, rightResizeHandle, sceneAudio, sceneDisplayName, setActiveSegment, setGlobalPlaybackTime, shell,
    segmentList, silentTimeline, state, syncInspector, timelineViewport, updateAudioScrubbers, updateGlobalAudioMuteButton,
    wizardVideoSettings,
  });

  const {
    addSceneImageHistoryPath, addSegmentVideoHistoryPath, archiveGeneratedSceneImage, assertBatchNotStopped,
    cycleSegmentImageHistory, cycleSegmentVideoHistory, generateT2IPromptForSegment,
    generateTextOnlyImagePromptFallbackForSegment, isBlankStarterProject, requireActiveSegment,
    sceneVideoConceptPromptText, toggleSegmentPreviewMode,
  } = createImagePrompts({
    activeProjectFolderForSave, activeSegment, allEditableSegments, applyMappedTriggerPhrases,
    ensureSegmentRuntimeFields, ernieTextGemmaModelSelect, flfSameLocationCameraDiversityDirection,
    flfStillEndpointNotes, gemmaRunnerLine, i2vTextGemmaModelSelect, projectInput,
    promoteChainedFLFSceneImageToEndFrame, promptRunnerActionName, pushHistory, render, sceneDisplayName,
    sceneSlotNumber, segmentIndexInfo, segmentMappedLocationText, segmentMappedSubjectText, setActiveSegment,
    state, storyboardScenePayload, syncInspector, syncPreview, t2iTextGemmaModelSelect,
    textGemmaRunnerPayload, wizardStoryboardState,
    applyImageContinuityToPromptSettings: (...args) => applyImageContinuityToPromptSettings(...args),
    nbImageSettingsForSegment: (...args) => nbImageSettingsForSegment(...args),
    fluxReferenceContextForSegment: (...args) => fluxReferenceContextForSegment(...args),
    segmentImageSource: (...args) => segmentImageSource(...args),
    applyImageTriggerToPrompt: (...args) => applyImageTriggerToPrompt(...args),
    syncSegmentFlowGptPrompt: (...args) => syncSegmentFlowGptPrompt(...args),
    syncSegmentT2IPrompt: (...args) => syncSegmentT2IPrompt(...args),
  });

  const {
    applyAutoBpmCalibration, applyCapCutBeatImport, applyThreePointBeatCalibration,
    captureBeatCalibrationAnchor, closeBeatCalibrationWizard, ensureAutoBpmForCalibration,
    ensureCapCutBeatsForCalibration, openBeatCalibrationWizard, reloadBeatMarkersFromAudio,
    renderBeatCalibrationWizard, setBeatMarkersVisible, showBeatMarkersIfAvailable, syncLyricNoteControls,
    syncSceneNoteControls, syncVideoNoteControls,
  } = createBeatCalibration({
    audioInput, autoSaveSessionQuiet, beatCalibration, beatCalibrationAnchors, beatCalibrationCaptureButton,
    beatCalibrationFpsInput, beatCalibrationGridType, beatCalibrationGridTypeHint, beatCalibrationInstruction,
    beatCalibrationTimecodeGrid, beatCalibrationTimecodeHint, beatCalibrationTimecodeInput,
    beatCalibrationWizard, beatMarkersButton, currentGlobalTime, enforceAudioTimelineEnd,
    loadedGlobalAudioDuration, lyricNoteButton, node, pauseTimelineForEditing, projectInput, pushHistory,
    render, sceneNoteButton, setGlobalPlaybackTime, state, videoNoteButton,
  });

  const {
    applyIngredientsReferenceMappings, applyIngredientsSheetForSceneIfMapped, applyLyricMapperToSegments,
    autoMapIngredientsSheets, buildIdLoraPromptForScene, idLoraSceneContext, ingredientsSheetForSegment,
    locationScoutCharacterPayloadForGpt, locationScoutLyricsPayloadForGpt, logicalExtraSubjectsForScene,
    logicalReferenceSubjects, logicalSubjectIdsForScene, openAdvancedLocationScoutGptForRefs,
    openLocationScoutGptForRefs, referenceBuilderSubjectChoices, sceneReferenceMapArray,
    sceneReferenceMapValue, setBaseSegmentDurationRipple, storyboardReferenceBuilderWithIdLoraRefs,
    syncI2VMotionJsonFromSegments, syncIngredientsSceneMapFromSubjectMappings, syncLyricMapperFromSegments,
    syncPromptJsonFromSegments,
  } = createReferenceData({
    addSceneImageHistoryPath, allEditableSegments, currentVideoMode, i2vMotionJsonInput,
    i2vVideoSettingsForSegment, promptJsonInput, segmentIndexInfo, state, timelineDuration,
    conceptPromptsTextFromSegments: (...args) => conceptPromptsTextFromSegments(...args),
    hasAnyI2VMotionNotes: (...args) => hasAnyI2VMotionNotes(...args),
    i2vMotionNotesTextFromSegments: (...args) => i2vMotionNotesTextFromSegments(...args),
  });

  const { openLyricMappingWorkflowModal } = createLyricMapping({
    activeSegment, allEditableSegments, applyIngredientsReferenceMappings, applyLyricMapperToSegments,
    audioInput, autoSaveSessionQuiet, createScenesFromTimestampedLyrics, currentVideoMode,
    openLyricReviewModal, projectSrtFileInput, pushHistory, referenceBuilderSubjectChoices, render,
    saveSession, sceneDisplayName, state, syncIngredientsSceneMapFromSubjectMappings, syncInspector,
    syncLyricMapperFromSegments, syncLyricNoteControls, timelineDuration, transcribeLyricsForTimeline,
  });

  const {
    addFluxIngredient, droppedSceneImageSource, enableFluxIngredientDrop, enableImageDrop,
    importTimelineImagesFromFolder, loadCustomImageFile, loadFluxIngredientFile, loadImageToImageFile,
    loadVisionReferenceFile, makeDragHandle, segmentImageSource, setImageToImageSource,
    setVisionReferenceSource,
  } = createMediaImport({
    activeSegment, addSceneImageHistoryPath, audio, audioInput, autoSaveSessionQuiet, createProgressWindow,
    currentVideoMode, ensureSegmentRuntimeFields, ernieI2ISlider, ernieI2IStartStep, ernieRefImagePanel,
    ernieUseVisionReference, flfChainingEnabled, globalAudioModeSelect, handleSegmentPick,
    importImageFolderButton, krea2TwoPassCreativity, krea2TwoPassCreativityInput, krea2TwoPassRefImagePanel,
    krea2TwoPassUseVisionReference, loadFirstLastFrameEndFile, normalizeSegments, openTimelineSceneCard,
    previousAutoChainSourceSegment, projectInput, pushHistory, refImageInput, refImagePanel, render,
    renderFluxGlobalIngredientList, renderFluxIngredientList, renderList, renderNBIngredientList,
    sceneDisplayName, sceneSlotNumber, segmentIndexInfo, segmentTrack, setActiveSegment,
    silentAudioDurationInput, snapTimeToBeat, state, syncErnieImagePanel, syncFluxKleinPanel,
    syncGlobalAudioModeControls, syncInspector, syncKrea2TwoPassPanel, syncPreview, syncZImageSettingsPanel,
    t2vRefImagePanel, timelineViewport, useT2VVisionReference, useVisionReference, zI2ISlider, zI2IStartStep,
    activeKrea2TwoPassSettings: (...args) => activeKrea2TwoPassSettings(...args),
    activeZImageSettings: (...args) => activeZImageSettings(...args),
  });

  const {
    applyMiniMaxH3NativeVoiceBlock, assembleMiniMaxH3PromptFromCreative, assertValidMiniMaxH3FinalPrompt,
    ensureAutoTimedSingerCuesBeforePrompt, ensureBuilderManagedFx, miniMaxH3CreativePromptContextForSegment,
    miniMaxH3ImageReferencePromptItems, miniMaxH3PromptCharacterBudget, miniMaxH3PromptVisionImages,
    miniMaxH3PromptVisionImagesForRunner, miniMaxPromptReferenceMismatch, miniMaxPromptReferenceSignature,
    miniMaxRenderReferenceImagePaths,
  } = createMiniMaxPrompt({
    autoTimeMiniMaxSingerCuesForSegment, firstLastFrameEndImageSource, isMiniMaxBuiltInSpeakerAssignmentMode,
    isMiniMaxSingerAssignmentMode, logicalExtraSubjectsForScene, logicalSubjectIdsForScene,
    miniMaxH3CueShotContractText, miniMaxH3CutPlanForSegment, miniMaxH3ModeForSegment,
    miniMaxH3SceneImageIsPromptInspiration, miniMaxH3SceneImageUseForSegment, miniMaxH3SettingsForSegment,
    miniMaxH3StartFrameCharacterInfluenceForSegment, miniMaxH3SubjectLabelMapForSegment,
    miniMaxH3VocalCueMapText, normalizeLyricCueMapForSegment, previousAutoChainSourceSegment,
    sceneDisplayName, sceneVideoConceptPromptText, segmentImageSource, segmentIndexInfo,
    segmentMappedLocationText, segmentMappedSubjectText, sceneCastGuardForSegment, selectedCastCoverageContract,
    selectedPerformerSubjectsForSegment, selectedSegmentImagePath, state, storyboardReferenceDataForSegment,
    assertMiniMaxH3ReferenceCapacity: (...args) => assertMiniMaxH3ReferenceCapacity(...args),
    miniMaxOrderedImageReferenceItemsForSegment: (...args) => miniMaxOrderedImageReferenceItemsForSegment(...args),
    miniMaxReferenceBuilderImagePathsForSegment: (...args) => miniMaxReferenceBuilderImagePathsForSegment(...args),
    miniMaxReferencePurposeText: (...args) => miniMaxReferencePurposeText(...args),
    miniMaxH3FrameContinuityPromptEnabled: (...args) => miniMaxH3FrameContinuityPromptEnabled(...args),
  });

  const { openFluxReferenceBuilderModal } = createReferenceBuilder({
    activeSegment, advanceZImageSeedAfterRun, allEditableSegments, autoSaveSessionQuiet,
    createDetailedLocationDescriptionWithGemma, createProgressWindow, currentVideoMode,
    describeReferenceImageWithGemma, droppedSceneImageSource, gemmaModelSelect, gemmaRunnerLine,
    i2vGemmaModelSelect, i2vTextGemmaModelSelect, locationExtractionStyleTheme,
    locationScoutLyricsPayloadForGpt, logicalReferenceSubjects, logicalSubjectIdsForScene,
    miniMaxH3ModeForSegment, miniMaxH3SettingsForSegment, normalizeLyricCueMapForSegment,
    openAdvancedLocationScoutGptForRefs, openLocationScoutGptForRefs, openLyricReviewModal,
    playSingerCueRange, projectContextPath, projectInput, projectReferenceBuilderLocationsPath,
    projectSceneNotesPath, pushHistory, render, renderFluxIngredientList, renderNBIngredientList,
    runImageMemoryCleanupQuiet, saveSession, saveZImageSettingsFromPanel, sceneDisplayName,
    sceneReferenceMapArray, sceneSlotNumber, selectedSegmentsForBatch, singerCueRelativePlayheadTime, state,
    subjectSceneInput, syncInspector, syncMiniMaxH3Panel, syncMiniMaxReferenceButtons,
    syncPerformerInspectorForSegment, syncZImageSettingsPanel, t2iTextGemmaModelSelect,
    textGemmaRunnerPayload, themeStyleInput, zClipPicker, zSeed, zUnetPicker, zVaePicker,
  });

  const { openIngredientsReferenceBuilderModal } = createIngredientsBuilder({
    activeSegment, allEditableSegments, applyIngredientsReferenceMappings, autoMapIngredientsSheets,
    autoSaveSessionQuiet, createDetailedLocationDescriptionWithGemma, createProgressWindow,
    describeReferenceImageWithGemma, gemmaRunnerLine, i2vTextGemmaModelSelect, locationExtractionStyleTheme,
    locationScoutCharacterPayloadForGpt, openLocationScoutGptForRefs, pushHistory, render, sceneDisplayName,
    state, syncIngredientsSceneMapFromSubjectMappings, syncInspector, syncPreview, t2iTextGemmaModelSelect,
    textGemmaRunnerPayload,
  });

  const { openIdLoraReferenceBuilderModalSafely, openReferenceBuilderTargetChooser } = createIdLoraBuilder({
    activeSegment, autoSaveSessionQuiet, drawWaveform, openFluxReferenceBuilderModal,
    openIngredientsReferenceBuilderModal, pushHistory, render, renderSegments, sceneDisplayName,
    setBaseSegmentDurationRipple, setMiniMaxH3ModeForSegment, state, syncInspector, syncMiniMaxH3Panel,
    syncVideoModePanel,
  });

  const {
    assertMiniMaxH3ReferenceCapacity, assertMiniMaxH3ReferenceDescriptionsReady,
    fluxReferenceContextForSegment, miniMaxDesiredReferenceKeysForSegment, miniMaxH3ReferenceCapacityStatus,
    miniMaxOrderedImageReferenceItemsForSegment, miniMaxReferenceBuilderImagePathsForSegment,
    miniMaxReferenceKeysForSegment, miniMaxReferencePurposeText, nbReferenceContextForSegment,
    openMiniMaxReferenceSelector, referenceBuilderSubjectItemsForSegment, rtvReferencesForSegment,
  } = createMiniMaxReferences({
    activeSegment, autoSaveSessionQuiet, currentVideoMode, firstLastFrameResolvedEndImageSource,
    logicalExtraSubjectsForScene, logicalReferenceSubjects, logicalSubjectIdsForScene,
    miniMaxH3ContinuityReferenceReserved, miniMaxH3ImageReferencePromptItems, miniMaxH3ModeForSegment,
    miniMaxH3SceneImageIsPromptInspiration, miniMaxH3StartFrameCharacterInfluenceForSegment,
    openReferenceBuilderTargetChooser, pushHistory, requireActiveSegment, rtvReferenceBehaviorForSegment,
    rtvSceneImageAnchorPayload, sceneDisplayName, sceneReferenceMapValue, segmentImageSource,
    segmentIndexInfo, state, syncMiniMaxReferenceButtons,
  });

  const { openPromptOptionsModal, runConceptPromptCreator, runMotionNoteCreator } = createPromptCreators({
    activeSegment, autoSaveSessionQuiet, clearFinalPromptList, countPromptFindReplaceMatches,
    createProgressWindow, currentVideoMode, editFinalPromptList, ernieNotesInput,
    fluxReferenceContextForSegment, i2vNotesInput, krea2TwoPassNotesInput, nbNotes, notesInput, pushHistory,
    reloadFinalPromptList, render, replacePromptPhraseAcrossScenes, sceneDisplayName,
    selectedTimelineRangeInfo, state, storyIdeaInput, syncI2VMotionJsonFromSegments,
    syncPromptJsonFromSegments, t2iTextGemmaModelSelect, textGemmaRunnerPayload, themeStyleInput,
    transcribeLyricsForTimeline, updateActiveFromInputs,
  });

  const {
    applyImageContinuityToPromptSettings, createErnieImageForSegment, createFlowGptImageForSegment,
    createFlowGptPromptWithGemma, createFluxKleinImageForSegment, createFluxKleinPromptWithGemma,
    createKrea2TwoPassImageForSegment, createNBImageForSegment, createNBPromptWithGemma,
    createZImageForSegment, currentEnhanceSource, enhanceImageForSegment, flowGptBrowserSettingsForSegment,
    fluxKleinSettingsForSegment, generateFluxKleinPromptForSegment, generateNBPromptForSegment,
    nbImageSettingsForSegment, previewErnieImage, previewFlowGptImage, previewFluxKleinImage,
    previewKrea2TwoPassImage, previewNBImage, previewZImage, previousSceneStartImageIngredient,
  } = createImageGeneration({
    activeProjectFolderForSave, activeSegment, addSceneImageHistoryPath, advanceErnieSeedAfterRun,
    advanceKrea2TwoPassSeedAfterRun, advanceZEnhanceSeedAfterRun, advanceZImageSeedAfterRun,
    archiveGeneratedSceneImage, autoSaveSessionQuiet, createFluxPromptButton, createNBPromptButton,
    createProgressWindow, currentVideoMode, ernieCreateButtons, ernieLoraSlots, ernieSeed,
    firstLastFrameEndImageSource, flfSameLocationCameraDiversityDirection, flowGptCreateImageButton,
    flowGptCreatePromptButton, flowGptPrompt, fluxCreateButtons, fluxGemmaModelSelect, fluxMmprojSelect,
    fluxPrompt, fluxReferenceContextForSegment, gemmaRunnerLine,
    generateTextOnlyImagePromptFallbackForSegment, imagePromptNotesWithDirector, krea2TwoPassCreateButtons,
    krea2TwoPassSeed, mergedFluxImageIngredients, nbCreateButtons, nbGemmaModelSelect, nbMmprojSelect,
    nbPrompt, nbReferenceContextForSegment, previousAutoChainSourceSegment, pushHistory, render,
    requireActiveSegment, runClearMemoryWorkflowQuiet, runImageMemoryCleanupQuiet,
    saveErnieImageSettingsFromPanel, saveFlowGptBrowserSettingsFromPanel, saveFluxKleinSettingsFromPanel,
    saveKrea2TwoPassSettingsFromPanel, saveNBImageSettingsFromPanel, saveZEnhanceSettingsFromPanel,
    saveZImageSettingsFromPanel, sceneDisplayName, segmentImageSource, segmentIndexInfo, state, syncInspector,
    syncPreview, t2iTextGemmaModelSelect, textGemmaRunnerPayload, updateActiveFromInputs, zCreateButtons,
    zEnhancePromptPreview, zEnhanceSeed, zLoraSlots, zSeed,
    applyImageTriggerToPrompt: (...args) => applyImageTriggerToPrompt(...args),
    ensureSegmentT2IPromptHasTrigger: (...args) => ensureSegmentT2IPromptHasTrigger(...args),
    syncSegmentFlowGptPrompt: (...args) => syncSegmentFlowGptPrompt(...args),
    syncSegmentT2IPrompt: (...args) => syncSegmentT2IPrompt(...args),
  });

  const {
    activeErnieImageSettings, activeFluxKleinSettings, activeI2VVideoSettings, activeKrea2TwoPassSettings,
    activeNBImageSettings, activeZImageSettings, clearTriggerPhrasesForFreshProject,
    cloneFlowGptBrowserSettingsForLoadedProject, loadGlobalModelDefaultsQuiet, setVideoVisionReferenceEnabled,
    syncProjectVideoEngineUI, videoVisionReferenceEnabled,
  } = createModelSettings({
    activeSegment, currentVideoMode, ensureAllSegmentRuntimeFields, ernieImageTriggerInput,
    fluxImageTriggerInput, imageTriggerInput, krea2TwoPassImageTriggerInput, ltxVideoPanel,
    miniMaxEnginePanel, projectVideoEngineBadge, segmentImageSource, settingsModalControls, state,
    syncErnieImagePanel, syncFluxKleinPanel, syncI2VVideoSettingsPanel, syncKrea2TwoPassPanel,
    syncMiniMaxH3Panel, syncNBImagePanel, syncVideoModePanel, syncVideoTypeControl, syncZEnhanceSettingsPanel,
    syncZImageSettingsPanel, videoSettingsSegment, videoTriggerInput,
  });

  const {
    applyImageTriggerToPrompt, applyVocalDirectiveToVideoPrompt, conceptPromptsTextFromSegments,
    effectiveVideoPerformanceModeForSegment, ensureSegmentT2IPromptHasTrigger,
    facialPerformanceNoteForSegment, hasAnyI2VMotionNotes, i2vAutoChainEnabled,
    i2vMotionNotesTextFromSegments, idLoraGemmaNotesForSegment, idLoraSpeechTextForSegment,
    imageModeImg2ImgContinuityLabel, imageModeSupportsImg2ImgContinuity, img2imgContinuityEnabled,
    recoverFromBuildGemmaError, runGemmaImagePromptPassWithRetry, storyboardVideoExtraNotesForSegment,
    syncSegmentFlowGptPrompt, syncSegmentT2IPrompt, videoGemmaNotesForSegment, videoTriggerPhraseForSegment,
  } = createPromptText({
    activeSegment, assertBatchNotStopped, autoSaveSessionQuiet, createProgressWindow, ernieT2IPrompt,
    flowGptPrompt, fluxKleinSettingsForSegment, fluxPrompt, fluxReferenceContextForSegment,
    generateTextOnlyImagePromptFallbackForSegment, idLoraSceneContext, krea2TwoPassT2IPrompt,
    nbImageSettingsForSegment, nbPrompt, render, runClearMemoryWorkflowQuiet, runImageMemoryCleanupQuiet,
    sceneDisplayName, segmentIndexInfo, segmentMappedLocationText, segmentMappedSubjectText, sceneCastGuardForSegment, state,
    storyboardScenePayload, t2iPrompt, zEnhancePromptPreview,
  });

  const {
    convertAllLtxVideoPromptsToMiniMaxH3, createI2VPromptWithGemma, createMiniMaxH3PromptWithLLM,
    gemmaT2IAllScenes, gemmaVideoAllTextOnly, generateFinalIndependentFLFPromptForSegment,
    generateI2VPromptForSegment, generateIndependentFLFMotionPlanForSegment, i2vAllScenes,
    miniMaxBatchReferenceProblems, runMiniMaxH3PromptGeneration,
  } = createBatchPrompts({
    activeProjectFolderForSave, activeSegment, allEditableSegments, assembleMiniMaxH3PromptFromCreative,
    assertBatchNotStopped, assertMiniMaxH3ReferenceDescriptionsReady, assertValidMiniMaxH3FinalPrompt,
    autoSaveSessionQuiet, batchTargetItems, buildI2VPromptRequestForSegment, convertLtxPromptsToMiniMaxButton,
    createFluxPromptButton, createI2VButton, createProgressWindow, createT2IButton, currentVideoMode,
    effectiveVideoPerformanceModeForSegment, ensureAllSegmentRuntimeFields,
    ensureAutoTimedSingerCuesBeforePrompt, ensureBuilderManagedFx, ernieCreateT2IButton,
    finalizeVideoPromptDraftOnly, finalizeVideoPromptForSegment, firstLastFramePromptReferences,
    flfGemmaContextMode, flfGemmaSceneConcept, flfGemmaVisualNotes, flfTransitionLoraActive, gemmaRunnerLabel,
    gemmaRunnerLine, gemmaT2IAllButton, gemmaVideoAllButton, generateT2IPromptForSegment,
    getI2VImageReference, hasFirstLastFrameEndImage, i2vAutoChainEnabled, i2vGemmaModelSelect,
    i2vMmprojSelect, i2vPrompt, i2vTextGemmaModelSelect, idLoraGemmaNotesForSegment, idLoraSceneContext,
    img2imgContinuityEnabled, krea2TwoPassCreateT2IButton, llmApiVisionModelSelected,
    miniMaxCreatePromptButton, miniMaxGemmaModelSelect, miniMaxH3CreativePromptContextForSegment,
    miniMaxH3ModeForSegment, miniMaxH3PromptCharacterBudget, miniMaxH3PromptVisionImages,
    miniMaxH3PromptVisionImagesForRunner, miniMaxH3SceneImageIsPromptInspiration,
    miniMaxH3SceneImageUseForSegment, miniMaxH3SettingsForSegment, miniMaxH3VocalCueMapText,
    miniMaxMmprojSelect, miniMaxOrderedImageReferenceItemsForSegment, miniMaxPrompt,
    miniMaxPromptReferenceMismatch, miniMaxPromptReferenceSignature, miniMaxRenderReferenceImagePaths,
    miniMaxTextGemmaModelSelect, normalizeLyricCueMapForSegment, previousAutoChainSourceSegment, projectInput,
    promptRunnerActionName, pushHistory, render, requireActiveSegment, rtvReferenceBehaviorForSegment,
    runClearMemoryWorkflowQuiet, runGemmaImagePromptPassWithRetry, runVideoPromptEnhancementBatch,
    saveGemmaJunkDebug, saveMiniMaxSceneInputsFromPanel, saveSessionForSceneVideo, sceneDisplayName,
    sceneVideoConceptPromptText, segmentImageSource, segmentIndexInfo, segmentMappedLocationText,
    segmentMappedSubjectText, sceneCastGuardForSegment, setVideoVisionReferenceEnabled, state, storyboardPipeline,
    storyboardScenePayload, syncInspector, syncMiniMaxH3Panel, syncRTVSceneImageAnchorPanel,
    syncVideoModePanel, textGemmaRunnerPayload, updateActiveFromInputs, updateMiniMaxPromptCharacterStatus,
    videoGemmaNotesForSegment, videoModeDisplayLabel, videoVisionReferenceEnabled, wizardStoryboardState,
    zImageAllButton,
    miniMaxH3FrameContinuityPromptEnabled: (...args) => miniMaxH3FrameContinuityPromptEnabled(...args),
  });

  const {
    createMiniMaxSceneVideo, createSceneVideo, miniMaxH3FrameContinuityPromptEnabled, openStitchPreviewModal, stitchPreviewFromSegments,
    renderImageSlideshowPreview, renderMiniMaxSceneVideoWithProgress, renderSceneVideoWithProgress,
    runGemmaThenCreateSceneVideo, stitchRenderedScenes,
  } = createVideoRender({
    activeSegment, activeVideoOutputFolder, advancedTwoPassControls, applyMappedTriggerPhrases,
    applySceneAdjustToRenderedVideo, applySceneFilmGrainToRenderedVideo, applySceneLutToRenderedVideo,
    applyVocalDirectiveToVideoPrompt, assertValidMiniMaxH3FinalPrompt, audioInput,
    audioSourceDurationForScene, autoSaveSessionQuiet, buildIdLoraPromptForScene, collectedSceneVideoFolder,
    createProgressWindow, createSceneVideoButtons, currentProjectAudioPath, currentVideoMode,
    enforceAudioTimelineEnd, ensureAudioOrOfferSilentTimeline, ensureBuilderManagedFx,
    ensureSceneAdjustsAppliedBeforeStitch, ensureSceneFilmGrainAppliedBeforeStitch,
    ensureSceneLutsAppliedBeforeStitch, ensureSelectedImageForSceneVideo, finishSingleSceneETA,
    firstLastFrameEndImageSource, firstLastFrameResolvedEndImageSource, firstLastFrameStartImageSource,
    flfChainingEnabled, flfRenderChainStartSource, gemmaThenCreateVideoButtons, generateI2VPromptForSegment,
    i2vAutoChainEnabled, i2vImagesFolder, i2vPrompt, i2vVideoSettingsForSegment, i2vVideoSettingsPayload,
    idLoraSceneContext, loadDirtyLatentBadges, ltx25RtvSceneOrdinal, miniMaxH3ContinuityModeForSegment,
    miniMaxH3ModeForSegment, miniMaxH3PromptVisionImages, miniMaxH3SceneImageIsPromptInspiration,
    miniMaxH3SettingsForSegment, miniMaxOrderedImageReferenceItemsForSegment, miniMaxPrompt,
    miniMaxPromptReferenceMismatch, miniMaxReferenceKeysForSegment, miniMaxRenderReferenceImagePaths,
    miniMaxSceneVideoButtons, nextOverlaySlotNumber, persistIngredientsSheetImages,
    prepareFLFRenderedFrameNextScene, previousAutoChainSourceSegment, projectInput, promptRunnerActionName,
    pushHistory, render, renderList, requireActiveSegment, rtvReferencesForSegment,
    runClearMemoryWorkflowQuiet, runMiniMaxH3PromptGeneration, saveMiniMaxH3SettingsFromPanel,
    saveMiniMaxSceneInputsFromPanel, saveSessionForSceneVideo, sceneDisplayName, sceneSlotNumber,
    sceneVideoDetailsHtml, segmentImageSource, segmentIndexInfo, segmentTrack, selectedSegmentImagePath,
    selectedSegmentsForBatch, setActiveSegment, startSingleSceneETA, state, syncInspector, syncPreview,
    twoPassControls, updateActiveFromInputs, updateMiniMaxPromptCharacterStatus, usingSceneAudioMode,
    validateMiniMaxSceneReadyForVideo, validateSceneReadyForVideo, validateSrtTimingForSceneVideo,
    videoModeDisplayLabel, videoTriggerPhraseForSegment,
  });

  const {
    setInspectorTab, syncBuilderLlmModelSelectsFromRunner, syncInspectorPanels,
    syncKrea2TwoPassLlmSelectsFromShared, syncKrea2TwoPassLlmSelectsToShared, updateI2VPromptSaveButtonState,
  } = createInspector({
    activeSegment, applyLayoutSizes, audioPanel, audioTabButton, ernieGemmaModelSelect, ernieMmprojSelect,
    ernieTextGemmaModelSelect, fluxGemmaModelSelect, fluxMmprojSelect, gemmaModelSelect, i2vGemmaModelSelect,
    i2vMmprojSelect, i2vPrompt, i2vTextGemmaModelSelect, imagePanel, imageTabButton, inspectorTabs,
    krea2TwoPassGemmaModelSelect, krea2TwoPassMmprojSelect, krea2TwoPassTextGemmaModelSelect,
    miniMaxGemmaModelSelect, miniMaxMmprojSelect, miniMaxTextGemmaModelSelect, mmprojSelect,
    nbGemmaModelSelect, nbMmprojSelect, noSceneNotice, savedI2VPrompts, saveI2VPromptButton, scenePanel,
    sceneTabButton, state, t2iTextGemmaModelSelect, timelinePromptSave, videoPanel, videoTabButton,
    zEnhanceGemmaModelSelect, zEnhanceMmprojSelect,
  });

  const { openBuilderAgentModal } = createBuilderAgent({
    activeSegment, addFluxIngredient, allEditableSegments, audioInput, autoSaveSessionQuiet,
    builderStorySourcePath, chooseProjectAudioFile, createErnieImageForSegment,
    createFluxKleinImageForSegment, createKrea2TwoPassImageForSegment, createNBImageForSegment,
    createProgressWindow, createSceneVideo, createStoryScenesFromSource, createZImageForSegment,
    currentVideoMode, droppedSceneImageSource, editContextTextFile, ernieNotesInput, fluxGemmaModelSelect,
    fluxMmprojSelect, fluxNotes, gemmaModelSelect, gemmaRunnerLabel, generateFluxKleinPromptForSegment,
    generateI2VPromptForSegment, generateNBPromptForSegment, generateT2IPromptForSegment, i2vGemmaModelSelect,
    i2vMmprojSelect, i2vNotesInput, loadBuilderStorySource, mergedFluxImageIngredients, mmprojSelect,
    nbGemmaModelSelect, nbMmprojSelect, nbNotes, notesInput, projectAudioFileInput, projectContextPath,
    projectInput, promptRunnerActionName, pushHistory, render, renderFluxIngredientList,
    renderNBIngredientList, runConceptPromptCreator, runImageMemoryCleanupQuiet, runMotionNoteCreator,
    saveBuilderStorySource, sceneDisplayName, sceneSlotNumber, segmentImageSource, segmentIndexInfo,
    segmentTrack, selectedTimelineRangeInfo, setInspectorTab, shell, state, storyIdeaInput, subjectSceneInput,
    syncFluxKleinPanel, syncI2VMotionJsonFromSegments, syncInspector, syncPromptJsonFromSegments,
    syncVideoModePanel, t2iTextGemmaModelSelect, textGemmaRunnerPayload, themeStyleInput,
    videoModeDisplayLabel,
  });

  const { openGemmaRunnerModal } = createGemmaRunner({
    autoSaveSessionQuiet, createProgressWindow, saveSession, state, syncBuilderLlmModelSelectsFromRunner,
    t2iTextGemmaModelSelect, textGemmaRunnerPayload, updatePromptRunnerButtonLabels,
  });

  const {
    buildFullFLFVideoPipeline, buildFullVideoPipeline, buildIndependentFLFPairs, createEndFramesAllScenes,
    createFLFImageChain, createNBImageForSegmentWithRetry, ernieImageAllScenes, flowGptImageAllScenes,
    fluxKleinAllScenes, krea2TwoPassImageAllScenes, nbImageAllScenes, renderAllScenes, zEnhanceAllScenes,
    zEnhanceBatchTargets, zImageAllScenes,
  } = createBatchRender({
    loadSessionFromProject,
    activeSegment, allEditableSegments, assertBatchNotStopped, autoSaveSessionQuiet, batchTargetItems,
    canImg2ImgContinuityFromPreviousRenderedScene, createEndFrameForSegment, createErnieImageForSegment,
    createFlowGptImageForSegment, createFluxKleinImageForSegment, createFluxPromptButton,
    createImageForSegmentInCurrentMode, createKrea2TwoPassImageForSegment, createNBImageForSegment,
    createNBPromptButton, createProgressWindow, createSceneVideoButtons, createT2IButton,
    createZImageForSegment, currentEnhanceSource, currentVideoMode, endFrameSegmentsForMode,
    enhanceImageForSegment, ensureAudioOrOfferSilentTimeline, ernieCreateButtons, ernieCreateT2IButton,
    firstLastFrameEndImageSource, firstLastFramePromptReferences, firstLastFrameStartImageSource,
    flfChainedSettingsPanel, flfChainingEnabled, flfChainPreviousEndFrame, flfPreGeneratePromptsEnabled,
    flfRenderChainStartSource, flfSameLocationCameraDiversityDirection, flfStructureModeSelect,
    flowGptCreateImageButton, flowGptCreatePromptButton, fluxCreateButtons, fullBuildButton,
    fullFLFBuildButton, gemmaThenCreateVideoButtons, generateFinalIndependentFLFPromptForSegment,
    generateFluxKleinPromptForSegment, generateI2VPromptForSegment,
    generateIndependentFLFMotionPlanForSegment, generateNBPromptForSegment, generateT2IPromptForSegment,
    hasFirstLastFrameEndImage, i2vAllScenes, i2vAutoChainEnabled, i2vTextGemmaModelSelect,
    imageAllSegmentsForMode, imageModeDisplayLabel, imageModeImg2ImgContinuityLabel,
    imageModeSupportsImg2ImgContinuity, img2imgContinuityEnabled, krea2TwoPassCreateButtons,
    krea2TwoPassCreateT2IButton, miniMaxBatchReferenceProblems, miniMaxH3ModeForSegment,
    miniMaxSceneVideoButtons, nbCreateButtons, persistRenderLog, prepareAutoChainedNextScene,
    prepareAutoImg2ImgContinuityForScene, prepareFLFRenderedFrameNextScene, prepareSceneAudioMix,
    previousAutoChainSourceSegment, previousSceneStartImageIngredient, projectInput, pushHistory,
    recoverFromBuildGemmaError, recoverSceneVideosFromProject, render, renderAllButton, renderETAScene,
    renderMiniMaxSceneVideoWithProgress, renderSceneVideoWithProgress, runClearMemoryWorkflowQuiet,
    runGemmaImagePromptPassWithRetry, runImageMemoryCleanupQuiet, savedImagePromptForMode,
    saveFlowGptBrowserSettingsFromPanel, saveI2VVideoSettingsFromPanel, saveMiniMaxH3SettingsFromPanel,
    saveSessionForSceneVideo, saveZEnhanceSettingsFromPanel, sceneDisplayName, sceneSlotNumber,
    segmentImageSource, segmentIndexInfo, segmentTrack, setImageSeedForCurrentMode, setMiniMaxH3SeedRandom,
    setVideoSeedRandom, startBuilderETA, state, stitchRenderedScenes, storyboardScenePayload,
    syncI2VVideoSettingsPanel, syncInspector, syncRTVSceneImageAnchorPanel, syncSegmentFlowGptPrompt,
    syncSegmentT2IPrompt, t2iTextGemmaModelSelect, textGemmaRunnerPayload, updateActiveFromInputs,
    upsertRenderLog, validateCreateEndFramesReady, validateRenderAllReady, validateZImageAllReady,
    videoModeDisplayLabel, wizardStoryboardState, zCreateButtons, zEnhanceAllButton, zEnhanceAllToolButton,
    zEnhanceButton, zImageAllButton,
  });

  const {
    addOverlaySegment, addSegment, addTimelineMarkerFromSelection, applyBulkSegmentTimings,
    captureSelectedVideoFrameAsImage, clearSelectedTimelineRange, deleteAllSegments, deleteSegment,
    deleteSelectedMedia, deleteStaleSceneLatents, loadCustomImage, openBulkSegmentsModal, sendPromptToEnhance,
    setTimelineRangePoint, splitActiveSceneAtPlayhead, syncOverlayTrackControls, toggleOverlayTrack,
  } = createTimelineActions({
    activeSegment, addOverlaySegmentButton, addSceneImageHistoryPath, allEditableSegments, audio,
    autoSaveSessionQuiet, baseSceneVideoTrimKind, chooseRenderedSceneTrimAtPlayhead, closeBaseTimelineGap,
    currentGlobalTime, customImageFileInput, deleteSelectedMediaButton, enforceAudioTimelineEnd,
    ensureSegmentRuntimeFields, freezeTimingControl, loadDirtyLatentBadges, loadedGlobalAudioDuration,
    nextFreeTimelineMarkerRange, nextOverlaySlotNumber, openTimelineMarkerEditor, overlayTrackToggleButton,
    pauseTimelineForEditing, playbackSegmentAtTime, previewVideo, projectInput, pushHistory, render,
    renderList, requireActiveSegment, sceneAudio, sceneListPane, sceneSlotNumber, segmentImageSource,
    segmentIndexInfo, segmentLayer, segmentTrack, selectedMediaForDelete, selectedTimelineRangeInfo,
    setActiveSegment, snapAddedSegmentEndToNearestBeat, snapTimeToBeat, state, syncI2VMotionJsonFromSegments,
    syncInspector, syncPreview, syncPromptJsonFromSegments, syncSegmentT2IPrompt, syncTimelineTrimModeButton,
    syncZEnhanceSettingsPanel, timelineDuration, updateHistoryButtons, updateSelectedMediaTools, useFrameAsImageButton,
  });

  const {
    branchProject, confirmAndRunFullBuild, confirmAndRunGemmaT2IAll, confirmAndRunGemmaVideoAll,
    confirmAndRunRenderAll, confirmAndRunZEnhanceAll, confirmAndRunZImageAll, deleteAllTimelineImages,
    deleteAllTimelineVideos,
  } = createBatchActions({
    activeSegment, allEditableSegments, audioInput, autoSaveSessionQuiet, batchScopeChoices,
    buildFullVideoPipeline, buildIndependentFLFPairs, createEndFramesAllScenes, createFLFImageChain,
    currentSessionData, currentVideoMode, deleteAllTimelineImagesButton, deleteAllTimelineVideosButton,
    deleteStaleSceneLatents, ensureSegmentRuntimeFields, ernieImageAllScenes, flowGptImageAllScenes,
    fluxKleinAllScenes, gemmaT2IAllScenes, gemmaVideoAllTextOnly, getPreferredProjectRoot,
    hasFirstLastFrameEndImage, krea2TwoPassImageAllScenes, loadSessionFromProject, nbImageAllScenes,
    pauseTimelineForEditing, previewImage, previewVideo, projectInput, promptRunnerActionName, pushHistory,
    render, renderAllScenes, renderList, rtvReferenceBehaviorGlobalValue, saveI2VVideoSettingsFromPanel,
    sceneAudio, sceneListPane, segmentImageSource, segmentLayer, state, syncFluxKleinPanel, syncInspector,
    syncPreview, updateActiveFromInputs, updateSelectedMediaTools, videoModeDisplayLabel,
    videoVisionReferenceEnabled, zEnhanceAllScenes, zEnhanceBatchTargets, zImageAllScenes,
  });

  const { openAutoBuildModal } = createAutoBuild({
    allEditableSegments, applyBulkSegmentTimings, assertBatchNotStopped,
    assertMiniMaxH3ReferenceDescriptionsReady, audio, audioInput, autoSaveSessionQuiet,
    chooseProjectAudioFile, createProgressWindow, describeReferenceImageWithGemma, gemmaRunnerLabel,
    gemmaRunnerLine, gemmaVideoAllTextOnly, llmApiVisionModelSelected, miniMaxH3PromptVisionImagesForRunner,
    miniMaxOrderedImageReferenceItemsForSegment, newProject, openGemmaRunnerModal,
    openStoryboardBuilderFromProject, projectInput, pushHistory, render, runMiniMaxH3PromptGeneration,
    saveGemmaJunkDebug, saveSession, sceneDisplayName, sceneReferenceMapArray, state,
    syncI2VVideoSettingsPanel, syncInspector, syncProjectVideoEngineUI, syncVideoModePanel,
    syncVideoTypeControl, transcribeExistingScenesWithOptions,
  });

  const { openWizardBetaFromBuilder, openWizardFromBuilder } = createWizardBridge({
    activeI2VVideoSettings, activeProjectFolderForSave, activeSegment, allEditableSegments, audioInput,
    autoSaveSessionQuiet, chooseProjectAudioFile, confirmAndRunFullBuild, confirmAndRunGemmaT2IAll,
    confirmAndRunGemmaVideoAll, confirmAndRunZImageAll, createProgressWindow,
    createScenesFromTimestampedLyrics, createSceneVideoActions, createSceneVideoButtons,
    createSilentTimelineAudioForDuration, currentVideoMode, ensureAllSegmentRuntimeFields,
    ensureAutoTimedSingerCuesBeforePrompt, ernieClipPicker, ernieImagePanel, ernieUnetPicker, ernieVaePicker,
    exportManualFlowGptRefs, finalizeVideoPromptDraftOnly, flowGptImageAllScenes, flowGptManualAutoAdvance,
    flowGptManualMode, flowGptManualStatus, flowGptModePanel, flowGptStatusText, fluxClipPicker,
    fluxKleinPanel, fluxUnetPicker, fluxVaePicker, gemmaModelSelect, gemmaRunnerLabel, gemmaRunnerLine,
    i2vAudioVaePicker, i2vClip1Picker, i2vClip2Picker, i2vDiffusionModelPicker, i2vFpsInput,
    i2vGemmaModelSelect, i2vHeightInput, i2vLoraCount, i2vLoraSlots, i2vMmprojSelect, i2vSeedInput,
    i2vTextGemmaModelSelect, i2vUnetPicker, i2vUpscalePicker, i2vUseGgufModel, i2vUseLora, i2vVaePicker,
    i2vWidthInput, imageModeDisplayLabel, importLatestManualFlowGptDownload, importTimelineImagesFromFolder,
    inspector, krea2TwoPassClipPicker, krea2TwoPassPanel, krea2TwoPassUnetPicker, krea2TwoPassVaePicker,
    ltxMsrFirstPassStrength, ltxMsrLoraPicker, miniMaxGemmaModelSelect, miniMaxMmprojSelect,
    miniMaxModePanels, miniMaxPassChooser, miniMaxSubTabs, miniMaxTextGemmaModelSelect, mmprojSelect,
    nbImagePanel, newProject, normalizeLyricCueMapForSegment, openBulkSegmentsModal,
    openFluxReferenceBuilderModal, openGemmaRunnerModal, openIdLoraReferenceBuilderModalSafely,
    openIngredientsReferenceBuilderModal, openLyricMappingWorkflowModal, openLyricReviewModal,
    openManualFlowGptBrowser, openStoryboardBuilderFromProject, projectInput, promptRunnerActionName,
    pushHistory, render, renderAllScenes, rtvSceneImageAnchorSection, saveErnieImageSettingsFromPanel,
    saveFlowGptBrowserSettingsFromPanel, saveFluxKleinSettingsFromPanel, saveGemmaJunkDebug,
    saveI2VVideoSettingsFromPanel, saveKrea2TwoPassSettingsFromPanel, saveMiniMaxH3SettingsFromPanel,
    saveNBImageSettingsFromPanel, saveSession, saveZImageSettingsFromPanel, sceneDisplayName, segmentTrack,
    selectedPerformerSubjectsForSegment, setInspectorTab, setSegmentPromptForEdit, state, storyboardPipeline,
    storyboardReferenceBuilderWithIdLoraRefs, storyboardScenePayload, subjectSceneInput, syncErnieImagePanel,
    syncFlowGptBrowserPanel, syncFlowGptManualPanel, syncFluxKleinPanel, syncI2VVideoModelPickerVisibility,
    syncI2VVideoSettingsPanel, syncInspector, syncKrea2TwoPassLlmSelectsFromShared, syncKrea2TwoPassPanel,
    syncLyricAndSubjectNoteFiles, syncMiniMaxH3Panel, syncNBImagePanel, syncProjectVideoEngineUI,
    syncVideoModePanel, syncVideoTypeControl, syncZImageSettingsPanel, t2iTextGemmaModelSelect,
    textGemmaRunnerPayload, transcribeLyricsForTimeline, updateActiveFromInputs, useSceneI2VVideoSettings,
    useSceneI2VVideoSettingsNote, useSceneMiniMaxH3Settings, useSceneMiniMaxH3SettingsNote,
    videoModeDisplayLabel, videoSubTabs, wizardStoryboardState, wizardVideoSettings, zClipPicker,
    zimageSettingsPanel, zUnetPicker, zVaePicker,
  });

  const {
    confirmAndRunFullFLFBuild, confirmOpenLegacyPromptCreator, loadCustomModelRootSetting,
    refreshGemmaChoices, refreshLoraChoices, refreshModelChoices, setSceneI2VVideoSettingsEnabled,
    setSceneMiniMaxH3SettingsEnabled, updateI2VLoraVisibility, wireI2VStrengthPair, wireVisionReferenceDrop,
  } = createModelPickers({
    activeI2VVideoSettings, activeSegment, autoSaveSessionQuiet, batchScopeChoices, buildFullFLFVideoPipeline,
    currentVideoMode, droppedSceneImageSource, ernieClipPicker, ernieGemmaModelSelect, ernieLoraSlots,
    ernieMmprojSelect, ernieTextGemmaModelSelect, ernieUnetPicker, ernieVaePicker, flfTransitionLoraNote,
    fluxClipPicker, fluxGemmaModelSelect, fluxLoraSlots, fluxMmprojSelect, fluxUnetPicker, fluxVaePicker,
    gemmaModelSelect, i2vAudioVaePicker, i2vClip1Picker, i2vClip2Picker, i2vDiffusionModelPicker,
    i2vGemmaModelSelect, i2vLoraCount, i2vLoraPanel, i2vLoraRows, i2vLoraSlots, i2vMmprojSelect,
    i2vTextGemmaModelSelect, i2vUnetPicker, i2vUpscalePicker, i2vUseLora, i2vVaePicker,
    krea2TwoPassClipPicker, krea2TwoPassLoraSlots, krea2TwoPassUnetPicker, krea2TwoPassVaePicker,
    loadVisionReferenceFile, ltxIdLoraPicker, ltxIngredientsLoraPicker, ltxMsrLoraPicker,
    miniMaxAdvancedLatentUpscalerPicker, miniMaxAudioVaePicker, miniMaxClipPicker,
    miniMaxDiffusionModelPicker, miniMaxGemmaModelSelect, miniMaxLoraSlots, miniMaxMmprojSelect,
    miniMaxTextGemmaModelSelect, miniMaxThreePassLoraPicker, miniMaxTurboLoraPicker,
    miniMaxTwoPassLatentUpscalerPicker, miniMaxTwoPassLoraPicker, miniMaxVideoVaePicker, mmprojSelect,
    nbGemmaModelSelect, nbMmprojSelect, openPromptCreatorPanel, pushHistory, renderList,
    saveI2VVideoSettingsFromPanel, saveMiniMaxH3SettingsFromPanel, setVisionReferenceSource, state,
    syncBuilderLlmModelSelectsFromRunner, syncI2VVideoSettingsPanel, syncKrea2TwoPassLlmSelectsFromShared,
    syncMiniMaxH3Panel, syncMiniMaxLlmSelectsFromShared, t2iTextGemmaModelSelect, zClipPicker,
    zEnhanceClipPicker, zEnhanceGemmaModelSelect, zEnhanceLoraSlots, zEnhanceMmprojSelect, zEnhanceUnetPicker,
    zEnhanceVaePicker, zLoraSlots, zUnetPicker, zVaePicker,
  });

  const { openWhatsNewModal } = createWhatsNewModal({
    updateReleaseNotes, updateStatus, updateV10Button,
  });

  const { applyBuilderFullscreen } = createFullscreen({
    applyLayoutSizes, fullscreen, fullscreenButton, normalShellStyle, overlay, render, shell,
  });

  function cancelPreviewPlayStart() {
    playStart.request += 1;
    playStart.inFlight = false;
  }

  function makeEditImagePromptButton() {
    const button = makeButton("Edit Prompt");
  button.title = "Ask the selected LLM runner to make a focused edit to the current image prompt.";
    button.style.display = "none";
    editImagePromptButtons.push(button);
    return button;
  }

  function wrapCreateSceneVideoActions(button) {
    const quickButton = makeButton("✨▶", "primary");
    quickButton.title = "Run Gemma for the current video mode, then immediately create the scene video without stopping for prompt review.";
    quickButton.setAttribute("aria-label", "Run Gemma then create scene video");
    quickButton.style.cssText = "flex:0 0 42px;min-width:42px;padding-left:8px;padding-right:8px;";
    button.style.flex = "1 1 auto";
    gemmaThenCreateVideoButtons.push(quickButton);
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:stretch;gap:6px;width:100%;";
    row.append(button, quickButton);
    return row;
  }

  function makeCreateSceneVideoButton() {
    const button = makeButton("Create Scene Video", "primary");
    createSceneVideoButtons.push(button);
    return wrapCreateSceneVideoActions(button);
  }

  function makeZCreateButton() {
    const button = makeButton("Create Z-Image", "primary");
    zCreateButtons.push(button);
    return button;
  }

  function makeErnieCreateButton() {
    const button = makeButton("Create with Ernie", "primary");
    ernieCreateButtons.push(button);
    return button;
  }

  function makeKrea2TwoPassCreateButton() {
    const button = makeButton("Create with Krea 2", "primary");
    krea2TwoPassCreateButtons.push(button);
    return button;
  }

  function makeFluxCreateButton() {
    const button = makeButton("Create with Flux/Klein", "primary");
    fluxCreateButtons.push(button);
    return button;
  }

  function makeNBCreateButton() {
    const button = makeButton("Create with NanoBanana", "primary");
    nbCreateButtons.push(button);
    return button;
  }

  function syncGlobalAudioModeControls() {
    const silent = globalAudioModeSelect.value === "silent";
    globalAudioDrop.style.display = silent ? "none" : "block";
    chooseGlobalAudioButton.style.display = silent ? "none" : "";
    silentAudioPanel.style.display = silent ? "grid" : "none";
  }



  projectVideoEngineBadge.addEventListener("click", () => { void toggleProjectVideoEngineFromBadge(); });
  projectVideoEngineBadge.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    void toggleProjectVideoEngineFromBadge();
  });

  scenesTabButton.onclick = () => {
    state.leftPanelTab = "scenes";
    syncLeftPanelTabs();
    autoSaveSessionQuiet("left panel scenes tab").catch(() => null);
  };
  toolsTabButton.onclick = () => {
    state.leftPanelTab = "tools";
    syncLeftPanelTabs();
    autoSaveSessionQuiet("left panel tools tab").catch(() => null);
  };
  lutsTabButton.onclick = () => {
    state.leftPanelTab = "luts";
    syncLeftPanelTabs();
    autoSaveSessionQuiet("left panel post process tab").catch(() => null);
  };
  postProcessLutsTab.onclick = () => {
    state.postProcessTab = "luts";
    syncPostProcessTabs();
  };
  postProcessGrainTab.onclick = () => {
    state.postProcessTab = "film_grain";
    syncPostProcessTabs();
  };
  postProcessFxTab.onclick = () => {
    state.postProcessTab = "fx";
    syncPostProcessTabs();
  };
  syncLeftPanelTabs();

  window.addEventListener("beforeunload", () => {
    restoreBrowserAiDownloadsQuietly().catch(() => null);
    const segments = [
      ...(Array.isArray(state.segments) ? state.segments : []),
      ...(Array.isArray(state.overlaySegments) ? state.overlaySegments : []),
    ];
    for (const segment of segments) {
      const paths = [
        String(segment?.lut_preview_image_path || "").trim(),
        String(segment?.film_grain_preview_image_path || "").trim(),
        String(segment?.adjust_preview_image_path || "").trim(),
        segment?.lut_preview_source_temporary ? String(segment?.lut_preview_source_preview_path || "").trim() : "",
        segment?.film_grain_preview_source_temporary ? String(segment?.film_grain_preview_source_preview_path || "").trim() : "",
        segment?.adjust_preview_source_temporary ? String(segment?.adjust_preview_source_preview_path || "").trim() : "",
      ].filter(Boolean);
      for (const path of paths) {
        try {
          const payload = new Blob([JSON.stringify({ path, project_folder: projectInput.value || state.projectFolder || "" })], { type: "application/json" });
          navigator.sendBeacon?.("/vrgdg/music_builder/luts/delete_preview", payload);
        } catch {
          // Best-effort cleanup only.
        }
      }
    }
  });

  const toastNotificationHandler = (event) => {
    const message = String(event?.detail?.message || "");
    const isError = Boolean(event?.detail?.isError);
    if (!shouldNotifyForToast(message, isError)) return;
    playBuilderNotification(isError ? "error" : "success");
  };
  window.addEventListener("vrgdg:builder-toast", toastNotificationHandler);

  function updateGlobalAudioMuteButton() {
    const muted = Boolean(audio.muted && sceneAudio.muted);
    globalAudioMuteButton.textContent = muted ? "🔇" : "🔊";
    globalAudioMuteButton.title = muted
      ? "Unmute global timeline audio."
      : "Mute global timeline audio. Useful when previewing a completed video that already has audio.";
  }

  miniMaxAutoTimeAllScenesButton.onclick = () => autoTimeAllMiniMaxSingerScenes();

  async function autoSaveSessionQuiet(reason = "") {
    if (!state.autoSaveEnabled) return false;
    try {
      updateActiveFromInputs();
      saveI2VVideoSettingsFromPanel();
      if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
        saveMiniMaxH3SettingsFromPanel();
        saveMiniMaxSceneInputsFromPanel();
      }
      const projectFolder = activeProjectFolderForSave();
      if (!projectFolder) {
        console.warn(`[VRGDG Music Builder] Autosave skipped before ${reason || "action"} because no active project is set.`);
        return false;
      }
      await persistIngredientsSheetImages(projectFolder);
      const data = await saveBuilderSessionJson({
        audio_path: audioInput.value,
        project_folder: projectFolder,
        session: currentSessionData(),
        project_context_files: await projectContextFilesForSessionSave(),
      }, 60000);
      if (data?.stale) {
        console.warn(`[VRGDG Music Builder] Autosave skipped before ${reason || "action"}: stale snapshot rejected by server.`);
        toast("Project was modified elsewhere. Snapshot not overwritten to protect newer edits. Please reload the project.", true);
        return false;
      }
      state.projectFolder = data.project_folder || state.projectFolder;
      state.sessionPath = data.session_path || state.sessionPath;
      state.srtPath = data.srt_path || state.srtPath;
      projectInput.value = state.projectFolder || projectInput.value;
      srtInput.value = state.srtPath || srtInput.value;
      setWidgetValue(node, "project_folder", state.projectFolder);
      setWidgetValue(node, "session_path", state.sessionPath);
      setWidgetValue(node, "srt_path", state.srtPath);
      rememberLastProject(state.projectFolder);
      await syncPromptJsonFromSegments(`autosave ${reason || "session"}`);
      await syncI2VMotionJsonFromSegments(`autosave ${reason || "session"}`);
      await syncLyricAndSubjectNoteFiles(`autosave ${reason || "session"}`);
      if (reason) console.log(`[VRGDG Music Builder] Autosaved session/SRT: ${reason}`, state.sessionPath || "");
      return true;
    } catch (error) {
      console.warn("[VRGDG Music Builder] Autosave failed:", error);
      toast(`Autosave failed before ${reason || "this action"}:\n${String(error?.message || error)}`, true);
      return false;
    }
  }


  projectBatchButton.onclick = () => {
    projectBatchPanel.style.display = projectBatchPanel.style.display === "none" ? "flex" : "none";
  };
  projectBatchAddCurrent.onclick = () => {
    const folder = String(projectInput.value || state.projectFolder || "").trim();
    if (!folder) {
      toast("Create or load a project before adding the current project to Project Batch.", true);
      return;
    }
    addProjectToBatch(folder);
  };
  projectBatchAddRecent.onclick = async () => {
    try {
      const data = await getJson(projectListUrl());
      const choice = await showLoadProjectModal(Array.isArray(data.projects) ? data.projects : []);
      if (choice?.project_folder) addProjectToBatch(choice.project_folder);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  };
  projectBatchAddSession.onclick = async () => {
    try {
      const folder = await pickProjectSessionFile();
      if (folder) addProjectToBatch(folder);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  };
  projectBatchAddCustom.onclick = async () => {
    const folder = await showTextInputModal({
      title: "Add Project To Batch",
      label: "Project folder path",
      value: "",
      placeholder: "Full project folder path...",
      confirmLabel: "Add Project",
    });
    if (folder) addProjectToBatch(folder);
  };
  projectBatchRun.onclick = runProjectBatchQueue;
  projectBatchStop.onclick = () => {
    projectBatch.stopRequested = true;
    state.batchCancelled = true;
    projectBatchStatus.textContent = "Stop requested. The current render will stop at the next safe checkpoint.";
    toast("Project Batch stop requested.");
  };
  renderProjectBatchQueue();

  saveI2VPromptButton.addEventListener("click", () => saveTimelinePrompt("i2v"));
  saveMiniMaxPromptButton.addEventListener("click", () => saveTimelinePrompt("minimax"));

  wireSceneInputs({
    endInput, ernieNotesInput, ernieT2IPrompt, freezeTimingControl, i2vMotionJsonInput, i2vNotesInput,
    i2vPrompt, krea2TwoPassNotesInput, krea2TwoPassT2IPrompt, labelInput, lyricSingersInput, lyricTextInput,
    notesInput, promptJsonInput, pushHistory, render, startInput, state, syncInspector, t2iPrompt,
    updateActiveFromInputs, zEnhanceGemmaNotes, zEnhancePromptPreview,
  });
  wireImageSettingsInputs({
    activeSegment, ernieImageTriggerInput, flowGptAskPreviousImage, flowGptAspectRatio, flowGptFailureMode,
    flowGptManualAutoAdvance, flowGptManualMode, flowGptPrompt, flowGptRetries, flowGptTimeout,
    fluxImageTriggerInput, fluxUseDirectorNotes, fluxUseTextOnlyGemmaPrompt, imageTriggerInput,
    krea2TwoPassImageTriggerInput, nbApiKey, nbModelSelect, nbNotes, nbPrompt, nbUseDirectorNotes,
    nbUseTextOnlyGemmaPrompt, pushHistory, saveErnieImageSettingsFromPanel,
    saveFlowGptBrowserSettingsFromPanel, saveFluxKleinSettingsFromPanel, saveKrea2TwoPassSettingsFromPanel,
    saveNBImageSettingsFromPanel, saveZImageSettingsFromPanel, syncFlowGptManualPanel,
    syncSegmentFlowGptPrompt,
  });
  wireBrowserAiPanel({
    activeBrowserAiReferenceGroup, addBrowserAiBandSequenceFiles, addBrowserAiGroupFiles,
    autoSaveSessionQuiet, browserAiAddExtrasButton, browserAiAddGroupImagesButton,
    browserAiAddLocationsButton, browserAiAddMembersButton, browserAiAddSingerButton,
    browserAiAutoAdvanceGroup, browserAiBandSequenceMode, browserAiBandSequenceSelection,
    browserAiChooseLocationButton, browserAiClearExtrasButton, browserAiClearGroupImagesButton,
    browserAiClearLocationButton, browserAiClearLocationsButton, browserAiClearMembersButton,
    browserAiClearSingerButton, browserAiDeleteGroupButton, browserAiDuplicateGroupButton,
    browserAiExtrasDrop, browserAiFinishButton, browserAiGroupDrop, browserAiGroupPrompt,
    browserAiGroupSelect, browserAiGroupStatus, browserAiLocationDrop, browserAiLocationsDrop,
    browserAiMembersDrop, browserAiNewGroupButton, browserAiRenameGroupButton, browserAiSend,
    browserAiSendButton, browserAiSequenceLocationSelect, browserAiSequenceSetSelect, browserAiSingerDrop,
    chooseBrowserAiImageFiles, clearBrowserAiBandSequenceFiles, exportManualFlowGptRefs,
    finishBrowserAiDownloadSession, flowGptCreateImageButton, flowGptLoginButton, flowGptManualChatPrompt,
    flowGptManualExportRefsButton, flowGptManualImportLatestButton, flowGptManualOpenButton,
    flowGptManualStatus, flowGptSetupButton, flowGptStatusButton, flowGptStatusText, flowNanoProviderButton,
    gptImageProviderButton, importLatestManualFlowGptDownload, metaImageProviderButton,
    openManualFlowGptBrowser, previewFlowGptImage, refreshFlowGptBrowserStatus,
    renderBrowserAiReferenceGroups, saveFlowGptBrowserSettingsFromPanel, sendBrowserAiReferenceGroup,
    setBrowserAiLocationFile, setFlowGptProvider, state,
  });
  videoTriggerInput.addEventListener("input", saveI2VVideoSettingsFromPanel);
  wireContextFileInputs({
    autoSaveSessionQuiet, clearConceptPromptNotesFromSegments, clearI2VMotionNotesFromSegments,
    editContextTextFile, editI2VMotionJsonButton, editPromptJsonButton, editStoryIdeaButton,
    editSubjectSceneButton, editThemeStyleButton, i2vMotionJsonInput, importI2VMotionJson, importPromptJson,
    loadDefaultContextPaths, loadVrgdgContextButton, promptJsonInput, pushHistory, render, state,
    storyIdeaInput, subjectSceneInput, syncInspector, themeStyleInput, useVrgdgTextContext,
  });
  wireReferenceControls({
    activeSegment, applyRTVReferenceBehaviorToAll, autoSaveSessionQuiet, clearSceneEndFrameButton,
    createEndFrameForSegment, createProgressWindow, createSceneEndFrameButton, currentVideoMode,
    ernieUseVisionReference, finalizeSceneFLFPromptButton, firstLastFrameStartImageSource, flfChainingEnabled,
    flfCustomEndDirection, flfEndpointModeSelect, flfTransitionTypeSelect,
    generateFinalIndependentFLFPromptForSegment, generateIndependentFLFMotionPlanForSegment,
    hasFirstLastFrameEndImage, krea2TwoPassUseVisionReference, loadFirstLastFrameEndFile,
    loadSceneEndFrameButton, planSceneEndMotionButton, pushHistory, refImageInput, render, renderList,
    rtvReferenceBehaviorForSegment, rtvReferenceBehaviorSelect, saveSessionForSceneVideo, sceneDisplayName,
    sceneEndFrameFileInput, segmentImageSource, segmentIndexInfo, setSceneI2VVideoSettingsEnabled,
    setSceneMiniMaxH3SettingsEnabled, state, syncErnieImagePanel, syncFluxKleinPanel, syncInspector,
    syncKrea2TwoPassPanel, syncNBImagePanel, syncRTVSceneImageAnchorPanel, syncZImageSettingsPanel,
    updateActiveFromInputs, useI2VPromptEnhancementPass, useI2VVisionReference, useSceneErnieImageSettings,
    useSceneFluxKleinSettings, useSceneI2VVideoSettings, useSceneKrea2TwoPassSettings,
    useSceneMiniMaxH3Settings, useSceneNBImageSettings, useSceneZImageSettings, useT2VVisionReference,
    useVisionReference,
  });
  wireToolbar({
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
    idLoraTrimModeButton, importI2VMotionJson, importI2VMotionJsonButton, importProjectButton,
    importPromptJson, importPromptJsonButton, importSceneNotesButton, importSceneNotesJson,
    importShareableProject, loadAudio, loadButton, loadLastProject, loadLastProjectButton, loadSession,
    loadSessionButton, loadSrt, loadSrtButton, lyricMapperButton, menuButton, menuDropdown,
    miniMaxH3SettingsForSegment, newProject, newProjectButton, openAutoBuildModal, openBuilderAgentModal,
    openBulkSegmentsModal, openGemmaRunnerModal, openLyricMappingWorkflowModal, openPromptOptionsModal,
    openReferenceBuilderTargetChooser, openRenderLogModal, openSceneAudioOptionsButton, openSceneOptions,
    openSettingsModal, openSnapSceneEdgeMenu, openStitchPreviewModal, openStoryboardBuilderFromProject,
    openWhatsNewModal, openWizardBetaFromBuilder, openWizardFromBuilder, overlayTrackHintButton,
    overlayTrackToggleButton, pauseTimelineForEditing, pickAudioButton, pickIdLoraReferenceAudioButton,
    pickSrtButton, projectAudioFileInput, projectSrtFileInput, promptCreatorButton, promptOptionsButton, redo,
    redoButton, renderAllButton, renderImageSlideshowPreview, renderLogButton, requireActiveSegment,
    reviewGuideButton, runClearMemoryWorkflow, saveButton, saveI2VVideoSettingsFromPanel, saveProjectAs,
    saveProjectAsButton, saveSession, sendCurrentProjectToPromptCreator, sendToPromptCreatorButton,
    setInButton, setOutButton, setTimelineRangePoint, settingsButton, silentAudioDurationInput,
    slideshowPreviewButton, snapSceneEdgeButton, splitActiveSceneAtPlayhead, splitSceneButton, state,
    stitchPreviewButton, stopCurrentWorkflow, stopWorkflowButton, storyboardBuilderButton,
    syncGlobalAudioModeControls, syncTimelineTrimModeButton, syncVideoTypeControl, toggleOverlayTrack, undo,
    undoButton, updateAudioScrubbers, updatePromptRunnerButtonLabels, updateStatus, updateStatusAction,
    updateV10Button, updateV10HintButton, updateWhatsNewAction, videoTypeSelect, whatsNewMenuButton,
    wizardBetaButton, wizardButton, zEnhanceAllButton, zEnhanceAllToolButton, zImageAllButton,
  });
  wirePanelButtons({
    applyRTVReferenceBehaviorToAll, autoSaveSessionQuiet, createFlowGptPromptWithGemma,
    createFluxKleinPromptWithGemma, createFluxPromptButton, createI2VButton, createI2VPromptWithGemma,
    createMiniMaxH3PromptWithLLM, createMiniMaxSceneVideo, createNBPromptButton, createNBPromptWithGemma,
    createSceneVideo, createSceneVideoButtons, createT2IButton, createT2IPromptWithGemma,
    customImageFileInput, droppedSceneImageSource, editCurrentImagePromptWithGemma,
    editErnieT2IInstructionsButton, editFlowGptT2IInstructionsButton, editFluxKleinT2IInstructionsButton,
    editI2VInstructionsButton, editIdLoraInstructionsButton, editImagePromptButtons,
    editIngredientsInstructionsButton, editKrea2T2IInstructionsButton, editNanoBT2IInstructionsButton,
    editRTVInstructionsButton, editT2VInstructionsButton, editZImageT2IInstructionsButton,
    enableFluxIngredientDrop, ernieCreateButtons, ernieCreateT2IButton, ernieI2IDrop, ernieI2ILoadButton,
    ernieImageCard, ernieRefImageDrop, ernieRefImageLoadButton, ernieSendT2IPromptToEnhanceButton,
    ernieT2IPrompt, firstLastFrameVideoCard, flowGptCard, flowGptCreatePromptButton, fluxCreateButtons,
    fluxGlobalIngredientButton, fluxGlobalIngredientClearButton, fluxGlobalIngredientDrop,
    fluxGlobalIngredientFileInput, fluxIngredientButton, fluxIngredientClearButton, fluxIngredientDrop,
    fluxIngredientFileInput, fluxKleinCard, fluxPrompt, gemmaThenCreateVideoButtons,
    generateEnhancePromptWithGemma, i2iImageFileInput, idLoraVideoCard, imageFolderFileInput,
    imageToVideoCard, importCustomVideoCard, importImageFolderButton, importTimelineImagesFromFolder,
    ingredientsToVideoCard, krea2TwoPassCard, krea2TwoPassCreateButtons, krea2TwoPassCreateT2IButton,
    krea2TwoPassI2IDrop, krea2TwoPassI2ILoadButton, krea2TwoPassRefImageDrop, krea2TwoPassRefImageLoadButton,
    krea2TwoPassSendT2IPromptToEnhanceButton, krea2TwoPassT2IPrompt, loadCustomImage, loadCustomImageButton,
    loadCustomImageFile, loadFluxIngredientFile, loadImageToImageFile, loadVisionReferenceFile,
    miniMaxCreatePromptButton, miniMaxEditContinuityPromptInstructionsButton, miniMaxEditInstructionsButton,
    miniMaxH3ModeForSegment, miniMaxReferenceButtons, miniMaxSceneVideoButtons, nbCreateButtons,
    nbGlobalIngredientButton, nbGlobalIngredientClearButton, nbGlobalIngredientDrop, nbImageCard,
    nbIngredientButton, nbIngredientClearButton, nbIngredientDrop, nbUseGlobalIngredients,
    openBuilderInstructionEditor, openMiniMaxReferenceSelector, previewErnieImage, previewFluxKleinImage,
    previewKrea2TwoPassImage, previewNBImage, previewZImage, pushHistory, referenceToVideoCard, refImageDrop,
    refImageLoadButton, render, renderFluxGlobalIngredientList, renderFluxIngredientList,
    renderNBIngredientList, renderSegments, requireActiveSegment, runGemmaThenCreateSceneVideo,
    sendFluxPromptToEnhanceButton, sendPromptToEnhance, sendT2IPromptToEnhanceButton, setImageToImageSource,
    state, syncFluxGlobalIngredientPanel, syncFluxKleinPanel, syncI2VVideoSettingsPanel, syncVideoModePanel,
    t2iPrompt, t2vRefImageDrop, t2vRefImageLoadButton, textToVideoCard, useFluxGlobalIngredients,
    visionRefFileInput, wireVisionReferenceDrop, zCreateButtons, zEnhanceCard, zEnhanceGemmaButton, zI2IDrop,
    zI2ILoadButton, zImageCard,
  });
  wireTimelineControls({
    activeSegment, applyAutoBpmCalibration, applyCapCutBeatImport, applyThreePointBeatCalibration, audio,
    autoSaveSessionQuiet, beatCalibration, beatCalibrationCancelButton, beatCalibrationCaptureButton,
    beatCalibrationGridType, beatCalibrationTimecodeInput, beatMarkersButton, beginGlobalTimelineScrub,
    calibrateFirstBeatButton, cancelPreviewPlayStart, captureBeatCalibrationAnchor,
    captureSelectedVideoFrameAsImage, clearActiveSegment, closeBeatCalibrationWizard, currentGlobalTime,
    deleteAllSegments, deleteAllSegmentsButton, deleteAllTimelineImages, deleteAllTimelineImagesButton,
    deleteAllTimelineVideos, deleteAllTimelineVideosButton, deleteSegment, deleteSegmentButton,
    deleteSelectedMedia, deleteSelectedMediaButton, enforceAudioTimelineEnd, ensureAutoBpmForCalibration,
    ensureCapCutBeatsForCalibration, ensureGlobalTimelineAudioSource, globalAudioMuteButton, globalScrub,
    isTimelinePlaying, lyricNoteButton, multiSelectButton, multiSelectHintButton, openBeatCalibrationWizard,
    openMultiSelectChooser, pauseAllAudio, playbackDuration, playbackSegmentAtTime, playButton, playhead,
    playSceneAudioFrom, playStart, previewEmpty, previewStage, previewVideo, reloadBeatMarkersFromAudio,
    render, renderBeatCalibrationWizard, sceneAudio, sceneListPane, sceneNoteButton, seekAudioWhenReady,
    setBeatMarkersVisible, setGlobalPlaybackTime, setGlobalTimelineAudioMuted, setTimelineZoom,
    snapAllSceneStartsButton, snapAllSceneStartsToNearestBeats, snapToBeatsControl,
    startSilentTimelinePlayback, state, stopButton, stopSilentTimelinePlayback, syncLyricNoteControls,
    syncPreviewPlayback, syncSceneNoteControls, syncTimelineTrimModeButton, syncVideoNoteControls,
    timelineAudioPathForSegment, timelineAudioSourceStartForSegment, timelineCanvas, timelineViewport,
    updateAudioScrubbers, updatePlayPauseButton, useFrameAsImageButton, usingSceneAudioPlaybackMode,
    videoNoteButton, waitForPreviewVideoReady, waveformModeSelect, zoomInButton, zoomOutButton,
  });
  wireKeyboardShortcuts({
    activeSegment, addSegmentButton, builderLifecycle, moveActiveSceneSelection, overlay, playButton, redo,
    segmentTrack, snapSceneEdgeToNearestBeat, undo,
  });

  loadCustomModelRootSetting().finally(() => {
    refreshGemmaChoices().catch((error) => {
      toast(`Could not load Gemma model choices:\n${String(error?.message || error)}`, true);
    });
    refreshLoraChoices().catch((error) => {
      toast(`Could not load LoRA choices:\n${String(error?.message || error)}`, true);
    });
    refreshModelChoices().catch((error) => {
      toast(`Could not load I2V model choices:\n${String(error?.message || error)}`, true);
    });
  });

  wireImageModelControls({
    ernieBatchSize, ernieClipPicker, ernieHeight, ernieI2IPath, ernieI2ISlider, ernieI2IStartStep,
    ernieLoraCount, ernieLoraSlots, ernieSeed, ernieSeedMode, ernieUnetPicker, ernieUseImageToImage,
    ernieUseLora, ernieVaePicker, ernieWidth, i2vAudioVaePicker, i2vClip1Picker, i2vClip2Picker,
    i2vDiffusionModelPicker, i2vEnableFp16Accumulation, i2vUnetPicker, i2vUpscalePicker, i2vUseGgufModel,
    i2vUseSageAttention, i2vVaePicker, krea2TwoPassAspectRatio, krea2TwoPassBatchSize, krea2TwoPassCfg,
    krea2TwoPassClipPicker, krea2TwoPassCreativity, krea2TwoPassCreativityInput, krea2TwoPassI2IPath,
    krea2TwoPassLoraCount, krea2TwoPassLoraSlots, krea2TwoPassSampler, krea2TwoPassSeed, krea2TwoPassSeedMode,
    krea2TwoPassUnetPicker, krea2TwoPassUseImageToImage, krea2TwoPassUseLora, krea2TwoPassVaePicker,
    saveErnieImageSettingsFromPanel, saveI2VVideoSettingsFromPanel, saveKrea2TwoPassSettingsFromPanel,
    saveZImageSettingsFromPanel, syncI2VVideoModelPickerVisibility, zBatchSize, zClipPicker, zFirstHeight,
    zFirstWidth, zI2IPath, zI2ISlider, zI2IStartStep, zLoraCount, zLoraSlots, zSecondHeight, zSecondWidth,
    zSeed, zSeedMode, zUnetPicker, zUseImageToImage, zUseLora, zVaePicker,
  });
  wireMiniMaxPanel({
    activeSegment, advancedTwoPassControls, allEditableSegments, autoSaveSessionQuiet,
    clearMiniMaxImageReferenceStartFrameOnModeSwitch, ensureMiniMaxSpeakerAssignments,
    isMiniMaxSingerAssignmentMode, loadDirtyLatentBadges, miniMaxAccelerationControls,
    miniMaxAddSpeakerCueButton, miniMaxAdvancedLatentUpscalerPicker, miniMaxAdvancedVramPreset, miniMaxAspectRatio, miniMaxAudioMode, miniMaxAudioVaePicker, miniMaxClipPicker,
    miniMaxContinuityMode, miniMaxContinuityPromptFromLastFrame, miniMaxCooldownFrames, miniMaxDenoise,
    miniMaxDiffusionModelPicker, miniMaxEasyCacheBypass, miniMaxEasyCacheEndPercent,
    miniMaxEasyCacheReuseThreshold, miniMaxEasyCacheStartPercent, miniMaxEasyCacheVerbose,
    miniMaxFp16Accumulation, miniMaxH3ContinuityModeForSegment, miniMaxH3ModeForSegment,
    miniMaxH3SceneImageUseForSegment, miniMaxH3SettingsForSegment, miniMaxContinuationDirection, miniMaxContinuationStart,
    miniMaxContinuationStartValue, miniMaxLatentContextFrames,
    miniMaxLocationTransitionCustom, miniMaxLocationTransitionPreset, miniMaxLoraCount, miniMaxLoraSlots,
    miniMaxMappedSpeakersForSegment, miniMaxMegapixels, miniMaxMemoryEfficientSageAttention, miniMaxResolutionPreset, miniMaxVideoProfileControls,
    miniMaxModeButtons, miniMaxPass2Prompt, miniMaxPassButtons, miniMaxPrompt, miniMaxSageAttention,
    miniMaxSamplerName, miniMaxSceneImageUse, miniMaxScheduler, miniMaxSeed,
    miniMaxStartFrameCharacterInfluence, miniMaxSteps, miniMaxThreePassLoraPicker,
    miniMaxThreePassLoraStrength, miniMaxThreePassRefImageSize, miniMaxTurboLoraPicker,
    miniMaxTurboLoraStrength, miniMaxTwoPassLatentScale,
    miniMaxTwoPassLatentUpscalerPicker, miniMaxTwoPassLoraPicker, miniMaxTwoPassLoraPreset,
    miniMaxTwoPassLoraPresetButtons, miniMaxTwoPassLoraStatus, miniMaxTwoPassLoraStrength,
    miniMaxTwoPassOutputCrf, miniMaxTwoPassRefImageSize, miniMaxTwoPassResizeMethod,
    miniMaxTwoPassTeCacheDepth, miniMaxTwoPassTeDevice, miniMaxTwoPassTeEnd, miniMaxTwoPassTeMcs,
    miniMaxTwoPassTeProcessingControl, miniMaxTwoPassTeStart, miniMaxTwoPassUseFastVaeDecode,
    miniMaxUseCurrentSceneVideoButton, miniMaxUseLoras, miniMaxUseTurboLora, miniMaxVideoReferenceRows,
    miniMaxVideoVaePicker, miniMaxWarmupFrames, pushHistory, renderMiniMaxSpeakerAssignmentPanel,
    requireActiveSegment, saveMiniMaxH3SettingsFromPanel, saveMiniMaxSceneInputsFromPanel, segmentImageSource,
    segmentTrack, selectedPerformerSubjectsForSegment, setMiniMaxH3ModeForSegment,
    setMiniMaxH3RenderPassForSegment, state, syncMiniMaxH3Panel, syncMiniMaxReferenceButtons, twoPassControls,
    videoSettingsSegment, wizardVideoSettings,
  });
  main.style.position = "relative";
  main.append(leftPanelToggle, rightPanelToggle);
  rightPanelToggle.onclick = () => {
    state.rightPanelCollapsed = !state.rightPanelCollapsed;
    applyLayoutSizes();
    state.onLayoutChanged?.();
    autoSaveSessionQuiet("right panel toggled");
  };
  leftPanelToggle.onclick = () => {
    state.leftPanelCollapsed = !state.leftPanelCollapsed;
    applyLayoutSizes();
    state.onLayoutChanged?.();
    autoSaveSessionQuiet("left panel toggled");
  };
  makePanelResize(leftResizeHandle, "left");
  makePanelResize(rightResizeHandle, "right");
  makePanelResize(timelineResizeHandle, "timeline");
  llmPopout.activate();
  // UI layout profiles. The one chosen last loads now and its layout wins over the project's saved sizes.
  const uiProfiles = createUiProfileActions({ controls: uiProfileControls, state, toast, applyLayoutSizes, autoSaveSessionQuiet });
  uiProfiles.wire();
  uiProfiles.refresh().catch((error) => console.warn("[VRGDG Music Builder] Could not load UI layouts:", error));
  setInspectorTab("scene");
  wireFluxKleinControls({
    fluxClipPicker, fluxHeight, fluxLoraCount, fluxLoraSlots, fluxNotes, fluxPrompt, fluxSeed, fluxUnetPicker,
    fluxUseLora, fluxVaePicker, fluxWidth, saveFluxKleinSettingsFromPanel, useFluxKlein,
  });
  i2vLoraHintButton.addEventListener("click", () => {
    showInfoModal({
      title: "Video LoRA Strengths",
      lines: [
        "Image to Video, Text to Video, Ingredients to Video, and LTX 2.5 Reference to Video use separate Pass 1 and Pass 2 strengths.",
        "Legacy LTX 2.3 Reference to Video and First Last Frame use one LoRA strength only; Pass 2 is hidden and ignored in those modes.",
        "If a LoRA hurts I2V/T2V motion, lower Pass 1. If you want more LoRA detail in the final result, raise Pass 2.",
      ],
    });
  });

  wireZEnhanceControls({
    saveZEnhanceSettingsFromPanel, upscaleEnhanceImage, zEnhanceAmount, zEnhanceButton, zEnhanceClipPicker,
    zEnhanceHeight, zEnhanceLoraCount, zEnhanceLoraSlots, zEnhanceSeed, zEnhanceSeedMode, zEnhanceUnetPicker,
    zEnhanceUseLora, zEnhanceVaePicker, zEnhanceWidth,
  });

  wireVideoSettingsControls({
    autoSaveSessionQuiet, flfChainedSettingsPanel, flfChainPreviousEndFrame, flfColorMatchFadeInput,
    flfColorMatchStrengthInput, flfFirstAttentionStrength, flfFirstGuideBlurInput, flfFirstGuideCrfInput,
    flfFirstGuideCrop, flfFirstGuideFrameIndexInput, flfFirstGuideInterpolation, flfFirstGuideStrengthInput,
    flfGemmaContextModeSelect, flfGlobalTransitionTypeSelect, flfLastAttentionStrength, flfLastGuideBlurInput,
    flfLastGuideCrfInput, flfLastGuideCrop, flfLastGuideFrameIndexInput, flfLastGuideInterpolation,
    flfLastGuideStrengthInput, flfMatchPreviousClipColor, flfPreGeneratePromptsFromSceneImages,
    flfRenderChainSourceSelect, flfRestoreWorkflowDefaultsButton, flfStructureModeSelect, i2vFpsInput,
    i2vHeightInput, i2vLoraCount, i2vLoraSlots, i2vPass1Bypass, i2vPass1SamplerSelect, i2vPass1SigmasInput,
    i2vPass1StrengthInput, i2vPass1StrengthSlider, i2vPass2Bypass, i2vPass2SamplerSelect, i2vPass2SigmasInput,
    i2vPass2StrengthInput, i2vPass2StrengthSlider, i2vPreFramesInput, i2vSeedInput, i2vTailLossFramesInput,
    i2vUseLora, i2vWidthInput, idLoraIdentityScaleInput, idLoraReferenceAudioInput, imageContinuityEnabled,
    imageContinuityStrength, ltx25AspectRatioSelect, ltx25MegapixelsInput, ltxIdLoraFirstPassStrength,
    ltxIdLoraPicker, ltxIdLoraSecondPassStrength, ltxIngredientsFirstPassStrength, ltxIngredientsLoraPicker,
    ltxMsrBackgroundMode, ltxMsrFirstPassStrength, ltxMsrLoraPicker, ltxMsrReferenceStrength,
    ltxMsrSecondPassStrength, render, renderList, saveI2VVideoSettingsFromPanel, state,
    syncRTVSceneImageAnchorPanel, updateI2VLoraVisibility, wireI2VStrengthPair,
  });
  updateI2VLoraVisibility();

  syncInspector();
  syncZImageSettingsPanel();
  syncFluxKleinPanel();
  syncZEnhanceSettingsPanel();
  syncI2VVideoSettingsPanel();
  syncVideoModePanel();
  syncSceneNoteControls();
  syncVideoNoteControls();
  syncLyricNoteControls();
  updateHistoryButtons();
  // The Builder opens filling the browser window. The fullscreen button still returns it to the floating panel.
  applyBuilderFullscreen(true);
}
