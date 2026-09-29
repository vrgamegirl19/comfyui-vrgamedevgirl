import { normalizeOverlayClip, normalizeOverlayTrackState } from "../VRGDG_OverlayTrack.js";
import {
  audioUrl,
  cancelComfyExecutionAndWaitIdle,
  getJson,
  normalizeOwnServerTimeoutMinutes,
  normalizeSceneRenderWaitHours,
  postJson,
  queueWorkflowPrompt,
  saveBuilderSessionJson,
  setBuilderAutomaticMemoryCleanupEnabled,
  syncBuilderSessionSaveRevision,
  waitForText,
} from "./comfy_api.mjs";
import {
  makeButton,
  normalizeProjectVideoEngine,
  normalizeVideoType,
  setWidgetValue,
  styleCompactToolbarButton,
  toast,
} from "./controls.mjs";
import { showLoadProjectModal, showWelcomeProjectModal } from "./dialogs.mjs";
import { cloneMiniMaxH3Settings } from "./minimax_h3.mjs";
import {
  cloneErnieImageSettings,
  cloneFlowGptBrowserSettings,
  cloneI2VVideoSettings,
  cloneKrea2TwoPassSettings,
  cloneZImageSettings,
  normalizeAutoBuildPreparation,
  normalizeBuilderStoryboardDefaults,
  normalizeBuilderStoryLayer,
  scrubGlobalImageToImageSourceForProject,
} from "./model_settings.mjs";
import { cloneKrea2ReferenceSettings } from "./models.mjs";
import { normalizeNotificationSettings } from "./notifications.mjs";
import { loadContextTextQuiet } from "./project_files.mjs";
import {
  loadI2VMotionNotesFromPath,
  loadLyricSegmentsFromPath,
  loadPromptJsonFromPath,
  normalizeAutoImg2ImgCreativity,
  normalizeAutoImg2ImgStartStep,
  normalizeContinuityMode,
  normalizeGemmaContextLimit,
  normalizeGemmaGpuLayers,
  normalizeLmStudioContextLimit,
  normalizeOutputTokenLimit,
  syncConceptPromptToStoryBeat,
} from "./prompt_text.mjs";
import {
  normalizeFluxReferenceBuilder,
  normalizeIdLoraReferenceBuilder,
  normalizeLyricMapper,
} from "./reference_data.mjs";
import { normalizeRenderLogs, renderLogDuration } from "./render_log.mjs";
import { newSegment } from "./segments.mjs";
import {
  mediaPathKey,
  normalizeSegmentVideoHistory,
  normalizeTimelineMarkers,
  normalizeTimelineRange,
} from "./timeline_state.mjs";

async function runClearMemoryNodeWorkflow(progress, label, percent = 95) {
  progress?.set(`Running RAM/VRAM cleanup workflow after ${label}...`, percent);
  const built = await postJson("/vrgdg/workflow_runner/build_clear_memory_prompt", {}, 120000);
  const queued = await queueWorkflowPrompt(built.prompt, {
    allowMemoryCleanup: true,
    onStatus: (status) => progress?.set(`${status}\n\nWaiting to run RAM/VRAM cleanup workflow...`, percent),
  });
  const promptId = queued?.prompt_id;
  if (!promptId) throw new Error("ComfyUI queued the memory cleanup workflow but did not return a prompt_id.");
  const text = await waitForText(promptId, (message) => {
    progress?.set(`${message}\nPrompt ID: ${promptId}`, percent);
  }, null, 3 * 60 * 1000);
  return text.join("\n").trim();
}

async function runFullMemoryCleanup(progress, label, percent = 95) {
  let workflowOutput = "";
  let workflowError = null;
  try {
    workflowOutput = await runClearMemoryNodeWorkflow(progress, label, percent);
  } catch (error) {
    workflowError = error;
    console.warn(`[VRGDG Music Builder] RAM/VRAM cleanup workflow after ${label} failed:`, error);
  }
  progress?.set(`Running direct Comfy/Gemma cache cleanup after ${label}...`, percent);
  const direct = await postJson("/vrgdg/music_builder/clear_memory_direct", {}, 120000);
  const directOutput = direct.message || `Direct memory cleanup finished after ${label}.`;
  return [
    workflowOutput || (workflowError ? `RAM/VRAM cleanup workflow failed: ${String(workflowError?.message || workflowError)}` : ""),
    directOutput,
  ].filter(Boolean).join("\n\n");
}

function projectBatchName(folder) {
  return String(folder || "").split(/[\\/]/).filter(Boolean).pop() || String(folder || "Project");
}

function showProjectBatchResultsModal(results = []) {
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(820px,calc(100vw - 48px));max-height:calc(100vh - 48px);overflow:hidden;border:1px solid #155e75;border-radius:8px;background:#0f172a;color:#cffafe;box-shadow:0 22px 70px rgba(0,0,0,.6);display:flex;flex-direction:column;";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 14px;border-bottom:1px solid #155e75;background:#083344;";
  const completed = results.filter((item) => item.status === "complete");
  const failed = results.filter((item) => item.status !== "complete");
  const title = document.createElement("div");
  title.textContent = `Project Batch Complete: ${completed.length} completed, ${failed.length} failed`;
  title.style.cssText = "font-size:14px;font-weight:900;";
  const close = makeButton("Close");
  close.style.padding = "6px 10px";
  header.append(title, close);
  const body = document.createElement("div");
  body.style.cssText = "display:flex;flex-direction:column;gap:9px;padding:14px;overflow:auto;";
  results.forEach((item, index) => {
    const row = document.createElement("div");
    row.style.cssText = `display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:center;border:1px solid ${item.status === "complete" ? "#155e75" : "#7f1d1d"};border-radius:7px;background:#020617;padding:10px;`;
    const info = document.createElement("div");
    info.style.cssText = "min-width:0;display:flex;flex-direction:column;gap:5px;";
    const name = document.createElement("div");
    name.textContent = `${index + 1}. ${item.name || projectBatchName(item.folder)} — ${item.status === "complete" ? "Complete" : "Failed"}`;
    name.style.cssText = `font-size:12px;font-weight:900;color:${item.status === "complete" ? "#a5f3fc" : "#fecaca"};`;
    const path = document.createElement("div");
    path.textContent = item.status === "complete" ? (item.finalVideoPath || "Completed, but no final video path was reported.") : item.error;
    path.title = path.textContent;
    path.style.cssText = "font-size:11px;color:#e2e8f0;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.35;";
    const folder = document.createElement("div");
    folder.textContent = item.folder;
    folder.style.cssText = "font-size:10px;color:#67e8f9;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    info.append(name, path, folder);
    const open = makeButton(item.status === "complete" ? "Open Video" : "Open Project");
    open.disabled = item.status === "complete" && !item.finalVideoPath;
    open.onclick = async () => {
      const target = item.status === "complete" ? item.finalVideoPath : item.folder;
      if (!target) return;
      open.disabled = true;
      open.textContent = "Opening...";
      try {
        await postJson("/vrgdg/music_builder/open_local_file", { path: target }, 30000);
      } catch (error) {
        toast(String(error?.message || error), true);
      } finally {
        open.disabled = false;
        open.textContent = item.status === "complete" ? "Open Video" : "Open Project";
      }
    };
    row.append(info, open);
    body.append(row);
  });
  if (!results.length) {
    const empty = document.createElement("div");
    empty.textContent = "No projects were run.";
    empty.style.cssText = "font-size:12px;color:#a1a1aa;";
    body.append(empty);
  }
  box.append(header, body);
  backdrop.append(box);
  document.body.append(backdrop);
  const finish = () => backdrop.remove();
  close.onclick = finish;
  backdrop.addEventListener("pointerdown", (event) => {
    if (event.target === backdrop) finish();
  });
}

export function createSession({
  activateGlobalTimelineAudioPlayback, allEditableSegments, applyLayoutSizes, audio, audioInput,
  autoLoadAllButton, autoSaveControl, autoSaveSessionQuiet, clearMemoryButton,
  cloneFlowGptBrowserSettingsForLoadedProject, closeBeatCalibrationWizard, createProgressWindow,
  enforceAudioTimelineEnd, ensureAllSegmentRuntimeFields, ensureSegmentRuntimeFields, faceFixTool,
  hasAnyI2VMotionNotes, i2vMotionJsonInput, imageContinuityEnabled, imageContinuityStrength,
  ingredientsSheetForSegment, loadAudio, loadDirtyLatentBadges, loadSrt, loadedGlobalAudioDuration,
  newProject, node, pauseAllAudio, previewVideo, projectBatch, projectBatchAddCurrent, projectBatchAddCustom,
  projectBatchAddRecent, projectBatchAddSession, projectBatchClearMemory, projectBatchContinueOnError,
  projectBatchForceVideos, projectBatchQueue, projectBatchRun, projectBatchStatus, projectBatchStop,
  projectContextPath, projectInput, projectSceneNotesPath, promptJsonInput, pushHistory,
  referenceBuilderSubjectLocationText, render, renderAllScenes, resetBuilderETA, resetProjectState,
  restoreBrowserAiDownloadsQuietly, sanitizedSessionSegments, saveI2VVideoSettingsFromPanel,
  saveMiniMaxH3SettingsFromPanel, saveMiniMaxSceneInputsFromPanel, sceneSlotNumber, segmentLayer,
  setBeatMarkersVisible, showBeatMarkersIfAvailable, snapToBeatsControl, srtInput, state, stopWorkflowButton,
  storyIdeaInput, subjectSceneInput, syncErnieImagePanel, syncFluxKleinPanel, syncI2VMotionJsonFromSegments,
  syncI2VVideoSettingsPanel, syncInspector, syncKrea2TwoPassPanel, syncLeftPanelTabs,
  syncLyricAndSubjectNoteFiles, syncLyricNoteControls, syncProjectVideoEngineUI, syncPromptJsonFromSegments,
  syncSceneNoteControls, syncVideoModePanel, syncVideoNoteControls, syncVideoTypeControl,
  syncZEnhanceSettingsPanel, syncZImageSettingsPanel, themeStyleInput, updateActiveFromInputs,
  useVrgdgTextContext, waveformModeSelect,
}) {
  const projectBatchQueueItems = [];

  function projectListUrl() {
    const root = getPreferredProjectRoot();
    return root
      ? `/vrgdg/music_builder/list_projects?project_root=${encodeURIComponent(root)}`
      : "/vrgdg/music_builder/list_projects";
  }

  async function importPromptJson(options = {}) {
    try {
      if (!promptJsonInput.value.trim()) {
        const paths = await getJson("/vrgdg/music_builder/default_context_paths");
        promptJsonInput.value = paths.concept_prompts_path || "";
        state.promptJsonPath = promptJsonInput.value;
      }
      const data = await postJson("/vrgdg/music_builder/load_prompt_json", {
        prompt_json_path: promptJsonInput.value,
      });
      const prompts = data.prompts || [];
      if (options.pushHistory !== false) pushHistory();
      state.promptJsonPath = data.prompt_json_path || promptJsonInput.value;
      for (let index = 0; index < state.segments.length && index < prompts.length; index++) {
        const segment = state.segments[index];
        segment.notes = prompts[index];
        segment.flux_notes = prompts[index];
        segment.nb_notes = prompts[index];
        syncConceptPromptToStoryBeat(segment, prompts[index]);
        if (!state.segments[index].label || /^Prompt\s+\d+$/i.test(state.segments[index].label)) {
          state.segments[index].label = `Scene ${index + 1}`;
        }
      }
      syncInspector();
      render();
      if (!options.quiet) toast(`Imported ${prompts.length} prompt${prompts.length === 1 ? "" : "s"} into segment notes and scene story beats.`);
      return prompts;
    } catch (error) {
      if (options.throwOnError) throw error;
      if (!options.quiet) toast(String(error?.message || error), true);
      return [];
    }
  }

  async function loadDefaultContextPaths() {
    try {
      const data = await getJson("/vrgdg/music_builder/default_context_paths");
      promptJsonInput.value = data.concept_prompts_path || "";
      i2vMotionJsonInput.value = data.i2v_motion_notes_path || "";
      themeStyleInput.value = data.theme_style_path || "";
      storyIdeaInput.value = data.story_idea_path || "";
      subjectSceneInput.value = data.subject_scene_path || "";
      state.promptJsonPath = promptJsonInput.value;
      state.i2vMotionJsonPath = i2vMotionJsonInput.value;
      state.themeStylePath = themeStyleInput.value;
      state.storyIdeaPath = storyIdeaInput.value;
      state.subjectScenePath = subjectSceneInput.value;
      state.useVrgdgTextContext = true;
      pushHistory();
      useVrgdgTextContext.input.checked = true;
      toast("Using VRGDG_TEMP TextFiles paths as Gemma context.");
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  function clearGeneratedSceneOutputsForImport() {
    for (const segment of allEditableSegments()) {
      ensureSegmentRuntimeFields(segment);
      segment.t2i_prompt = "";
      segment.i2v_prompt = "";
      segment.i2v_prompt_origin = "manual";
      segment.enhance_prompt = "";
      segment.approved_image_path = "";
      segment.custom_image_path = "";
      segment.custom_image_data = "";
      segment.custom_image_name = "";
      segment.image = null;
      segment.image_history = [];
      segment.image_history_index = -1;
      segment.video_path = "";
      segment.video_history = [];
      segment.video_history_index = -1;
      segment.video_source_path = "";
      segment.video_folder = "";
      segment.video_output = null;
      segment.video_status = "none";
      segment.preview_mode = "image";
      segment.flux_prompt = "";
      segment.flux_image_ingredients = [];
    }
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.dataset.path = "";
    previewVideo.dataset.cacheKey = "";
    previewVideo.style.display = "none";
  }

  async function runClearMemoryWorkflow() {
    let progress = null;
    try {
      clearMemoryButton.disabled = true;
      clearMemoryButton.textContent = "Clearing...";
      progress = createProgressWindow("Clearing memory");
      const output = await runFullMemoryCleanup(progress, "manual clear memory", 35);
      progress.set(output || "Memory cleanup finished.", 100);
      progress.close(4500);
      toast("Memory cleanup workflow finished.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      clearMemoryButton.disabled = false;
      styleCompactToolbarButton(clearMemoryButton, {
        lines: ["Clear", "RAM"],
        icon: "memory",
        width: 52,
        title: "Clear Builder, ComfyUI, and model memory caches.",
      });
    }
  }

  async function importI2VMotionJson(options = {}) {
    try {
      if (!i2vMotionJsonInput.value.trim()) {
        const paths = await getJson("/vrgdg/music_builder/default_context_paths");
        i2vMotionJsonInput.value = paths.i2v_motion_notes_path || "";
        state.i2vMotionJsonPath = i2vMotionJsonInput.value;
      }
      const notes = await loadI2VMotionNotesFromPath(i2vMotionJsonInput.value);
      if (options.pushHistory !== false) pushHistory();
      state.i2vMotionJsonPath = i2vMotionJsonInput.value;
      for (let index = 0; index < state.segments.length && index < notes.length; index++) {
        state.segments[index].i2v_notes = notes[index];
      }
      syncInspector();
      render();
      if (!options.quiet) toast(`Imported ${notes.length} I2V motion note${notes.length === 1 ? "" : "s"} into scenes.`);
      return notes;
    } catch (error) {
      if (options.throwOnError) throw error;
      if (!options.quiet) toast(String(error?.message || error), true);
      return [];
    }
  }

  async function importSceneNotesJson(options = {}) {
    try {
      const sceneNotesPath = projectSceneNotesPath();
      if (!sceneNotesPath) throw new Error("Create or load a project before importing SceneNotes.json.");
      const notes = await loadPromptJsonFromPath(sceneNotesPath);
      if (options.pushHistory !== false) pushHistory();
      let applied = 0;
      let nonEmpty = 0;
      for (let index = 0; index < state.segments.length && index < notes.length; index += 1) {
        state.segments[index].timeline_note = String(notes[index] || "");
        applied += 1;
        if (String(notes[index] || "").trim()) nonEmpty += 1;
      }
      state.showTimelineSceneNotes = true;
      syncSceneNoteControls();
      syncInspector();
      render();
      for (const noteBox of segmentLayer.querySelectorAll("[data-scene-note-segment-id]")) {
        const segment = allEditableSegments().find((item) => item.id === noteBox.dataset.sceneNoteSegmentId);
        if (segment) noteBox.value = String(segment.timeline_note || "");
      }
      if (!options.skipAutoSave) await autoSaveSessionQuiet("SceneNotes.json import");
      if (!options.quiet) toast(`Imported ${nonEmpty} non-empty scene note${nonEmpty === 1 ? "" : "s"} into ${applied} scene${applied === 1 ? "" : "s"} from:\n${sceneNotesPath}`);
      return notes;
    } catch (error) {
      if (options.throwOnError) throw error;
      if (!options.quiet) toast(String(error?.message || error), true);
      return [];
    }
  }

  async function stopCurrentWorkflow() {
    state.batchCancelled = true;
    pauseAllAudio();
    let progress = null;
    try {
      stopWorkflowButton.disabled = true;
      stopWorkflowButton.textContent = "Stopping...";
      progress = createProgressWindow("Stopping workflow");
      progress.set("Interrupting ComfyUI and clearing pending queue...", 20);
      await cancelComfyExecutionAndWaitIdle((status) => {
        progress.set(`${status}`, 45);
      }, { shouldCancel: () => false });
      progress.set(state.automaticMemoryCleanup ? "Clearing memory after stop..." : "Automatic memory cleanup is disabled; stopping without clearing models or caches...", 45);
      const cleanupOutput = await runClearMemoryWorkflowQuiet(progress, "stop request", 85);
      progress.set(state.automaticMemoryCleanup ? "Stop requested and memory cleanup finished." : `Stop requested.\n${cleanupOutput}`, 100);
      progress.close(3000);
      toast(state.automaticMemoryCleanup ? "Stop requested. Memory cleanup ran." : "Stop requested. Automatic memory cleanup was skipped.");
    } catch (error) {
      progress?.set(`Error while stopping:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      stopWorkflowButton.disabled = false;
      stopWorkflowButton.textContent = "Stop";
    }
  }

  async function showStartupWelcome() {
    try {
      resetProjectState("", "", "");
    } catch (error) {
      console.warn("[VRGDG Music Builder] Fresh startup reset failed:", error);
      state.projectFolder = "";
      state.sessionPath = "";
      state.srtPath = "";
      state.segments = [newSegment(0, 4)];
      state.overlaySegments = [];
      state.activeId = state.segments[0]?.id || "";
    }
    const projectsLoader = getJson(projectListUrl())
      .then((data) => Array.isArray(data.projects) ? data.projects : []);
    const choice = await showWelcomeProjectModal([], projectsLoader);
    if (!choice) return false;
    if (choice.action === "new") {
      return Boolean(await newProject());
    }
    if (choice.action === "load" && choice.project_folder) {
      return await loadSessionFromProject(choice.project_folder);
    }
    return false;
  }

  async function loadSession() {
    try {
      let projects = [];
      try {
        const data = await getJson(projectListUrl());
        projects = Array.isArray(data.projects) ? data.projects : [];
      } catch (error) {
        console.warn("[VRGDG Music Builder] Could not list existing projects:", error);
      }
      const choice = await showLoadProjectModal(projects);
      if (!choice?.project_folder) return;
      await loadSessionFromProject(choice.project_folder);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  async function loadLastProject() {
    const folder = getLastProject() || projectInput.value || state.projectFolder;
    if (!String(folder || "").trim()) {
      toast("No last project has been saved yet. Use Load Project first.", true);
      return;
    }
    await loadSessionFromProject(folder);
  }

  function renderProjectBatchQueue() {
    projectBatchQueue.replaceChildren();
    if (!projectBatchQueueItems.length) {
      const empty = document.createElement("div");
      empty.textContent = "No projects queued yet.";
      empty.style.cssText = "border:1px dashed #155e75;border-radius:6px;padding:10px;color:#7dd3fc;font-size:11px;text-align:center;";
      projectBatchQueue.append(empty);
      projectBatchRun.disabled = true;
      return;
    }
    projectBatchQueueItems.forEach((item, index) => {
      const row = document.createElement("div");
      row.style.cssText = "display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:7px;align-items:center;border:1px solid #155e75;border-radius:6px;background:#0b1220;padding:7px;";
      const number = document.createElement("div");
      number.textContent = String(index + 1);
      number.style.cssText = "width:22px;height:22px;border-radius:999px;background:#083344;color:#a5f3fc;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:900;";
      const info = document.createElement("div");
      info.style.cssText = "min-width:0;";
      const name = document.createElement("div");
      name.textContent = item.name || projectBatchName(item.folder);
      name.style.cssText = "font-size:12px;font-weight:900;color:#f8fafc;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      const path = document.createElement("div");
      path.textContent = item.folder;
      path.title = item.folder;
      path.style.cssText = "font-size:10px;color:#67e8f9;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      info.append(name, path);
      const remove = makeButton("Remove");
      remove.style.padding = "5px 7px";
      remove.disabled = projectBatch.running;
      remove.onclick = () => {
        projectBatchQueueItems.splice(index, 1);
        renderProjectBatchQueue();
      };
      row.append(number, info, remove);
      projectBatchQueue.append(row);
    });
    projectBatchRun.disabled = projectBatch.running || !projectBatchQueueItems.length;
  }

  function addProjectToBatch(folder) {
    const clean = String(folder || "").trim();
    if (!clean) return false;
    if (projectBatchQueueItems.some((item) => item.folder.toLowerCase() === clean.toLowerCase())) {
      toast("That project is already in the batch queue.", true);
      return false;
    }
    projectBatchQueueItems.push({ folder: clean, name: projectBatchName(clean) });
    projectBatchStatus.textContent = `${projectBatchQueueItems.length} project${projectBatchQueueItems.length === 1 ? "" : "s"} queued.`;
    renderProjectBatchQueue();
    return true;
  }

  function setProjectBatchRunning(running) {
    projectBatch.running = Boolean(running);
    projectBatchRun.disabled = projectBatch.running || !projectBatchQueueItems.length;
    projectBatchStop.disabled = !projectBatch.running;
    for (const button of [projectBatchAddCurrent, projectBatchAddRecent, projectBatchAddSession, projectBatchAddCustom]) button.disabled = projectBatch.running;
    projectBatchForceVideos.input.disabled = projectBatch.running;
    projectBatchContinueOnError.input.disabled = projectBatch.running;
    projectBatchClearMemory.input.disabled = projectBatch.running;
    renderProjectBatchQueue();
  }

  async function runProjectBatchQueue() {
    if (projectBatch.running) return;
    if (!projectBatchQueueItems.length) {
      toast("Add at least one saved project to the Project Batch queue.", true);
      return;
    }
    projectBatch.stopRequested = false;
    setProjectBatchRunning(true);
    const queue = projectBatchQueueItems.map((item) => ({ ...item }));
    const results = [];
    const progress = createProgressWindow("Project Batch");
    const startedAt = Date.now();
    try {
      for (let index = 0; index < queue.length; index += 1) {
        if (projectBatch.stopRequested) break;
        const item = queue[index];
        const name = item.name || projectBatchName(item.folder);
        const prefix = `Project ${index + 1}/${queue.length}: ${name}`;
        projectBatchStatus.textContent = `${prefix}\nLoading project...`;
        progress.set(`${prefix}\nLoading project...`, Math.round((index / queue.length) * 100));
        let result = { ...item, status: "failed", finalVideoPath: "", error: "" };
        try {
          const loaded = await loadSessionFromProject(item.folder);
          if (!loaded) throw new Error("Project could not be loaded.");
          await saveSession({ quiet: true, throwOnError: true });
          projectBatchStatus.textContent = `${prefix}\nRunning Render All...`;
          progress.set(`${prefix}\nRunning Render All. The normal Render All progress window will show scene details.`, Math.round((index / queue.length) * 100));
          state.batchCancelled = false;
          const renderResult = await renderAllScenes({
            sceneScope: "all",
            forceVideos: Boolean(projectBatchForceVideos.input.checked),
            suppressFinalModal: true,
          });
          if (projectBatch.stopRequested || state.batchCancelled) throw new Error("Stopped by user.");
          await saveSession({ quiet: true, throwOnError: true });
          const finalVideoPath = String(renderResult?.final_video_path || state.finalVideoPath || "");
          if (!finalVideoPath) throw new Error("Render All finished without reporting a final video path.");
          result.status = "complete";
          result.finalVideoPath = finalVideoPath;
          result.error = "";
          results.push(result);
          projectBatchStatus.textContent = `${prefix}\nComplete.\n${result.finalVideoPath || ""}`;
          if (projectBatchClearMemory.input.checked && index < queue.length - 1) {
            progress.set(`${prefix}\nClearing memory before the next project...`, Math.round(((index + 0.9) / queue.length) * 100));
            await runClearMemoryWorkflowQuiet(progress, `Project Batch ${name}`, Math.round(((index + 0.95) / queue.length) * 100));
          }
        } catch (error) {
          result.error = String(error?.message || error);
          results.push(result);
          projectBatchStatus.textContent = `${prefix}\nFailed:\n${result.error}`;
          if (!projectBatchContinueOnError.input.checked || projectBatch.stopRequested || state.batchCancelled) break;
        }
      }
      const completed = results.filter((item) => item.status === "complete").length;
      const failed = results.filter((item) => item.status !== "complete").length;
      const elapsed = renderLogDuration(Date.now() - startedAt);
      progress.set(`Project Batch finished.\nCompleted: ${completed}\nFailed: ${failed}\nElapsed: ${elapsed}`, 100);
      progress.close(4500);
      projectBatchStatus.textContent = `Project Batch finished.\nCompleted: ${completed}\nFailed: ${failed}\nElapsed: ${elapsed}`;
      showProjectBatchResultsModal(results);
    } finally {
      state.batchCancelled = false;
      setProjectBatchRunning(false);
    }
  }

  const LAST_PROJECT_KEY = "vrgdg_music_builder_last_project_folder";
  const PROJECT_ROOT_KEY = "vrgdg_music_builder_project_root";

  function rememberLastProject(projectFolder = "") {
    const folder = String(projectFolder || "").trim();
    if (!folder) return;
    try {
      localStorage.setItem(LAST_PROJECT_KEY, folder);
    } catch (error) {
      console.warn("[VRGDG Music Builder] Could not save last project folder:", error);
    }
  }

  function getLastProject() {
    try {
      return localStorage.getItem(LAST_PROJECT_KEY) || "";
    } catch {
      return "";
    }
  }

  function getPreferredProjectRoot() {
    try {
      return String(localStorage.getItem(PROJECT_ROOT_KEY) || "").trim();
    } catch {
      return "";
    }
  }

  function setPreferredProjectRoot(projectRoot = "") {
    const root = String(projectRoot || "").trim();
    try {
      if (root) localStorage.setItem(PROJECT_ROOT_KEY, root);
      else localStorage.removeItem(PROJECT_ROOT_KEY);
    } catch (error) {
      console.warn("[VRGDG Music Builder] Could not save the preferred projects root:", error);
    }
    return root;
  }

  async function autoLoadAll(options = {}) {
    try {
      autoLoadAllButton.disabled = true;
      autoLoadAllButton.textContent = "Importing...";
      pushHistory();
      let paths = await postJson("/vrgdg/music_builder/project_prompt_creator_paths", {
        project_folder: projectInput.value || state.projectFolder || "",
      });
      let exists = paths.exists || {};
      let sourceLabel = "this project";
      const sourceProjectFolder = String(options.sourceProjectFolder || "").trim();
      let currentPrompts = [];
      let currentMotionNotes = [];
      if (exists.concept_prompts_path) {
        try {
          currentPrompts = await loadPromptJsonFromPath(paths.concept_prompts_path);
        } catch (_error) {
          currentPrompts = [];
        }
      }
      if (exists.i2v_motion_notes_path) {
        try {
          currentMotionNotes = await loadI2VMotionNotesFromPath(paths.i2v_motion_notes_path);
        } catch (_error) {
          currentMotionNotes = [];
        }
      }
      const hasCurrentPrompts = currentPrompts.some((item) => String(item || "").trim());
      const hasCurrentMotionNotes = currentMotionNotes.some((item) => String(item || "").trim());
      if (sourceProjectFolder) {
        const targetProjectFolder = String(projectInput.value || state.projectFolder || "").trim();
        if (targetProjectFolder && targetProjectFolder.replace(/[\\/]+$/, "").toLowerCase() === sourceProjectFolder.replace(/[\\/]+$/, "").toLowerCase()) {
          paths = await postJson("/vrgdg/music_builder/project_prompt_creator_paths", {
            project_folder: sourceProjectFolder,
          });
        } else {
          try {
            paths = await postJson("/vrgdg/music_builder/copy_prompt_creator_outputs", {
              project_folder: targetProjectFolder,
              source_project_folder: sourceProjectFolder,
            }, 90000);
          } catch (error) {
            if (/\b405\b/.test(String(error?.message || error))) {
              throw new Error("The Prompt Creator handoff backend route is not loaded yet. Fully restart ComfyUI, refresh the browser, then try Send To Video Creator again.");
            }
            throw error;
          }
        }
        exists = paths.exists || {};
        sourceLabel = paths.source_project_folder ? `selected Prompt Creator project:\n${paths.source_project_folder}` : "selected Prompt Creator project";
        if (!exists.srt_path || !exists.concept_prompts_path) {
          throw new Error("The selected Prompt Creator project does not have saved SRT and concept prompt outputs yet. Run or save Prompt Creator outputs first, then send it to Video Creator.");
        }
      } else if (!exists.srt_path || !exists.concept_prompts_path || !hasCurrentPrompts || !hasCurrentMotionNotes) {
        paths = await postJson("/vrgdg/music_builder/import_latest_prompt_creator_outputs", {
          project_folder: projectInput.value || state.projectFolder || "",
        }, 90000);
        exists = paths.exists || {};
        sourceLabel = paths.source_project_folder ? `latest Prompt Creator project:\n${paths.source_project_folder}` : "latest Prompt Creator project";
        if (!exists.srt_path || !exists.concept_prompts_path) {
          throw new Error("No previous Prompt Creator output was found. Run Prompt Creator first, then import it into this project.");
        }
      }
      if (paths.audio_path) audioInput.value = paths.audio_path;
      if (paths.srt_path) srtInput.value = paths.srt_path;
      promptJsonInput.value = paths.concept_prompts_path || "";
      i2vMotionJsonInput.value = exists.i2v_motion_notes_path ? paths.i2v_motion_notes_path || "" : "";
      state.lyricSegmentsPath = exists.lyric_segments_path ? paths.lyric_segments_path || "" : "";
      themeStyleInput.value = exists.theme_style_path ? paths.theme_style_path || "" : "";
      storyIdeaInput.value = exists.story_idea_path ? paths.story_idea_path || "" : "";
      subjectSceneInput.value = exists.subject_scene_path ? paths.subject_scene_path || "" : "";
      state.promptJsonPath = promptJsonInput.value;
      state.i2vMotionJsonPath = i2vMotionJsonInput.value;
      state.themeStylePath = themeStyleInput.value;
      state.storyIdeaPath = storyIdeaInput.value;
      state.subjectScenePath = subjectSceneInput.value;
      state.useVrgdgTextContext = true;
      useVrgdgTextContext.input.checked = true;
      if (paths.audio_path) await loadAudio();
      await loadSrt({ throwOnError: true, skipSessionSave: true });
      let importedPrompts = [];
      let importedMotionNotes = [];
      let importedLyrics = [];
      if (promptJsonInput.value) {
        importedPrompts = await importPromptJson({ quiet: true, pushHistory: false });
      }
      if (i2vMotionJsonInput.value) {
        importedMotionNotes = await importI2VMotionJson({ quiet: true, pushHistory: false });
      }
      if (state.lyricSegmentsPath) {
        try {
          importedLyrics = await loadLyricSegmentsFromPath(state.lyricSegmentsPath);
        } catch (error) {
          console.warn("[VRGDG Music Builder] Could not import lyric segment status:", error);
          importedLyrics = [];
        }
      }
      for (let index = 0; index < state.segments.length && index < importedLyrics.length; index += 1) {
        state.segments[index].lyric_text = String(importedLyrics[index] || "").trim();
      }
      const nonEmptyPrompts = importedPrompts.filter((item) => String(item || "").trim()).length;
      const nonEmptyMotionNotes = importedMotionNotes.filter((item) => String(item || "").trim()).length;
      if (!nonEmptyPrompts) {
        throw new Error(
          `Prompt Creator import found ${promptJsonInput.value || "ConceptPrompts.txt"}, but it contains no usable prompts. Run Prompt Creator first, then import it into this project.`
        );
      }
      clearGeneratedSceneOutputsForImport();
      syncInspector();
      render();
      await autoSaveSessionQuiet("prompt creator import");
      const parts = [
        `Imported ${nonEmptyPrompts} concept prompt${nonEmptyPrompts === 1 ? "" : "s"}`,
        nonEmptyMotionNotes
          ? `${nonEmptyMotionNotes} I2V motion note${nonEmptyMotionNotes === 1 ? "" : "s"}`
          : "no I2V motion notes found",
      ];
      toast(`${parts.join(" and ")} from ${sourceLabel}. Previous generated images/videos were cleared from this project session.`);
    } catch (error) {
      toast(String(error?.message || error), true);
      if (options.throwOnError) throw error;
    } finally {
      autoLoadAllButton.disabled = false;
      autoLoadAllButton.textContent = "Import Data From Prompt Creator";
    }
  }

  async function runClearMemoryWorkflowQuiet(progress, label, percent = 95) {
    if (!state.automaticMemoryCleanup) {
      const output = `Automatic RAM/VRAM cleanup skipped after ${label} (disabled in Builder Settings).`;
      progress?.set(output, percent);
      return output;
    }
    const output = await runFullMemoryCleanup(progress, label, percent);
    progress?.set(output, percent);
    return output;
  }

  async function runImageMemoryCleanupQuiet(progress, label, percent = 95) {
    try {
      return await runClearMemoryWorkflowQuiet(progress, label, percent);
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Memory cleanup after ${label} failed:`, error);
      return "";
    }
  }

  function currentSessionData() {
    enforceAudioTimelineEnd();
    return {
      segments: sanitizedSessionSegments(state.segments, "base"),
      overlay_segments: sanitizedSessionSegments(state.overlaySegments, "overlay"),
      overlay_track: normalizeOverlayTrackState(state.overlayTrack),
      active_track: state.activeTrack,
      timing_frozen: state.timingFrozen,
      srt_mode: state.srtMode,
      prompt_json_path: state.promptJsonPath,
      i2v_motion_json_path: state.i2vMotionJsonPath,
      image_trigger_phrase: state.imageTriggerPhrase,
      video_trigger_phrase: state.videoTriggerPhrase,
      default_facial_performance: state.defaultFacialPerformance || "",
      default_facial_performance_custom: state.defaultFacialPerformanceCustom || "",
      use_i2v_prompt_enhancement_pass: Boolean(state.useI2VPromptEnhancementPass),
      fail_on_invalid_prompt_formats: Boolean(state.failOnInvalidPromptFormats),
      continuity_mode: normalizeContinuityMode(state.continuityMode, state.autoChainLastFrame),
      auto_img2img_start_step: normalizeAutoImg2ImgStartStep(state.autoImg2ImgStartStep),
      auto_img2img_creativity: normalizeAutoImg2ImgCreativity(state.autoImg2ImgCreativity),
      auto_chain_last_frame: Boolean(state.autoChainLastFrame),
      image_continuity_enabled: Boolean(state.imageContinuityEnabled),
      image_continuity_strength: state.imageContinuityStrength || "balanced",
      auto_chain_style: state.autoChainStyle || "continuous",
      auto_chain_direction: state.autoChainDirection || "",
      auto_chain_transition_lora_prompt: Boolean(state.autoChainTransitionLoraPrompt),
      auto_chain_transition_trigger: state.autoChainTransitionTrigger || "zhuanchang",
      use_vrgdg_text_context: state.useVrgdgTextContext,
      theme_style_path: state.themeStylePath,
      story_idea_path: state.storyIdeaPath,
      subject_scene_path: state.subjectScenePath,
      text_gemma_runner: state.textGemmaRunner || "builtin",
      gemma_context_limit: normalizeGemmaContextLimit(state.gemmaContextLimit),
      gemma_output_token_limit: normalizeOutputTokenLimit(state.gemmaOutputTokenLimit),
      gemma_gpu_layers: normalizeGemmaGpuLayers(state.gemmaGpuLayers),
      lm_studio_base_url: state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1",
      lm_studio_model: state.lmStudioModel || "",
      lm_studio_api_key: state.lmStudioApiKey || "",
      lm_studio_context_limit: normalizeLmStudioContextLimit(state.lmStudioContextLimit),
      lm_studio_output_token_limit: normalizeOutputTokenLimit(state.lmStudioOutputTokenLimit),
      llm_api_provider: state.llmApiProvider || "openai",
      llm_api_model: state.llmApiModel || "",
      llm_api_key_project: state.llmApiKeyProject || "",
      own_server_url: state.ownServerUrl || "http://127.0.0.1:8000/v1",
      own_server_model: state.ownServerModel || "",
      own_server_api_key_project: state.ownServerApiKeyProject || "",
      own_server_output_token_limit: normalizeOutputTokenLimit(state.ownServerOutputTokenLimit),
      own_server_timeout: normalizeOwnServerTimeoutMinutes(state.ownServerTimeoutMinutes) * 60,
      notification_settings: normalizeNotificationSettings(state.notificationSettings),
      automatic_memory_cleanup: Boolean(state.automaticMemoryCleanup),
      scene_render_wait_hours: normalizeSceneRenderWaitHours(state.sceneRenderWaitHours),
      waveform_mode: state.waveformMode,
      snap_to_beats: state.snapToBeats,
      show_beat_markers: state.showBeatMarkers,
      show_timeline_scene_notes: state.showTimelineSceneNotes,
      show_timeline_video_notes: state.showTimelineVideoNotes,
      show_timeline_lyric_notes: state.showTimelineLyricNotes,
      selected_timeline_range: normalizeTimelineRange(state.selectedTimelineRange),
      timeline_markers: normalizeTimelineMarkers(state.timelineMarkers),
      active_timeline_marker_id: state.activeTimelineMarkerId || "",
      audio_duration: loadedGlobalAudioDuration(),
      audio_peaks: Array.isArray(state.peaks) ? state.peaks : [],
      beat_markers: Array.isArray(state.beats) ? state.beats : [],
      detected_tempo_bpm: Math.max(0, Number(state.detectedTempoBpm || 0)),
      beat_calibration: state.beatCalibration,
      left_panel_width: state.leftPanelWidth,
      left_panel_tab: state.leftPanelTab === "tools" || state.leftPanelTab === "luts" ? state.leftPanelTab : "scenes",
      right_panel_width: state.rightPanelWidth,
      timeline_panel_height: state.timelinePanelHeight,
      timeline_zoom: state.timelineZoom,
      auto_save_enabled: state.autoSaveEnabled,
      video_type: normalizeVideoType(state.videoType),
      video_engine: normalizeProjectVideoEngine(state.projectVideoEngine),
      minimax_h3_settings: cloneMiniMaxH3Settings(state.miniMaxH3Settings),
        minimax_h3_two_pass: state.miniMaxH3Settings.render_pass === "two_pass",
        minimax_h3_three_pass: state.miniMaxH3Settings.render_pass === "three_pass",
        minimax_h3_advanced_two_pass: state.miniMaxH3Settings.render_pass === "three_pass",
      image_model_mode: state.imageModelMode,
      zimage_settings: state.zimageSettings,
      reference_krea2_settings: cloneKrea2ReferenceSettings(state.referenceKrea2Settings),
      flux_klein_settings: state.fluxKleinSettings,
      flow_gpt_browser_settings: cloneFlowGptBrowserSettings(state.flowGptBrowserSettings),
      nb_image_settings: state.nbImageSettings,
      ernie_image_settings: state.ernieImageSettings,
      krea2_2pass_settings: cloneKrea2TwoPassSettings(state.krea2TwoPassSettings),
      use_flux_global_image_ingredients: Boolean(state.useFluxGlobalImageIngredients),
      flux_global_image_ingredients: Array.isArray(state.fluxGlobalImageIngredients) ? state.fluxGlobalImageIngredients : [],
      flux_reference_builder: normalizeFluxReferenceBuilder(state.fluxReferenceBuilder),
      id_lora_reference_builder: normalizeIdLoraReferenceBuilder(state.idLoraReferenceBuilder),
      lyric_mapper: normalizeLyricMapper(state.lyricMapper),
      z_enhance_settings: state.zEnhanceSettings,
      video_model_mode: state.videoModelMode || "i2v",
      i2v_video_settings: state.i2vVideoSettings,
      prompt_tools_hint_prefs: state.promptToolsHintPrefs || {},
      builder_agent_messages: Array.isArray(state.builderAgentMessages) ? state.builderAgentMessages : [],
      builder_agent_auto_apply: Boolean(state.builderAgentAutoApply),
      builder_agent_purpose: state.builderAgentPurpose || "scene_work",
      builder_agent_reference_images: Array.isArray(state.builderAgentReferenceImages) ? state.builderAgentReferenceImages : [],
      builder_story_source_path: state.builderStorySourcePath || "",
      builder_story_reference_images: Array.isArray(state.builderStoryReferenceImages) ? state.builderStoryReferenceImages : [],
      builder_story_reference_notes: state.builderStoryReferenceNotes || "",
      builder_story_layer: normalizeBuilderStoryLayer(state.builderStoryLayer),
      builder_storyboard_defaults: normalizeBuilderStoryboardDefaults(state.builderStoryboardDefaults),
      auto_build_preparation: normalizeAutoBuildPreparation(state.autoBuildPreparation),
      wizard_beta_draft: state.wizardBetaDraft,
      render_logs: normalizeRenderLogs(state.renderLogs),
      active_render_log_id: state.activeRenderLogId || "",
    };
  }

  async function persistIngredientsSheetImages(projectFolder) {
    const folder = String(projectFolder || "").trim();
    if (!folder) return 0;
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    let saved = 0;
    const dataUrlSize = (value) => String(value || "").length;
    const compactImageDataUrl = (dataUrl) => new Promise((resolve) => {
      const raw = String(dataUrl || "").trim();
      if (!/^data:image\//i.test(raw) || dataUrlSize(raw) < 780000) {
        resolve(raw);
        return;
      }
      const img = new Image();
      img.onload = () => {
        const attempts = [
          [1400, 0.82],
          [1100, 0.74],
          [900, 0.66],
          [720, 0.58],
          [560, 0.5],
        ];
        let best = raw;
        for (const [maxDim, quality] of attempts) {
          const scale = Math.min(1, maxDim / Math.max(img.naturalWidth || img.width || 1, img.naturalHeight || img.height || 1));
          const width = Math.max(1, Math.round((img.naturalWidth || img.width || 1) * scale));
          const height = Math.max(1, Math.round((img.naturalHeight || img.height || 1) * scale));
          const canvas = document.createElement("canvas");
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext("2d");
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(0, 0, width, height);
          ctx.drawImage(img, 0, 0, width, height);
          const candidate = canvas.toDataURL("image/jpeg", quality);
          if (candidate.length < best.length) best = candidate;
          if (candidate.length < 720000) {
            resolve(candidate);
            return;
          }
        }
        resolve(best);
      };
      img.onerror = () => resolve(raw);
      img.src = raw;
    });
    const saveImageObject = async (image, referenceType, name) => {
      const imageData = String(image?.data || "").trim();
      if (!/^data:image\//i.test(imageData)) return false;
      const uploadData = await compactImageDataUrl(imageData);
      const data = await postJson("/vrgdg/music_builder/save_flux_reference_image", {
        project_folder: folder,
        reference_type: referenceType,
        name,
        image_data: uploadData,
      }, 60000);
      image.path = data.saved_path || "";
      image.data = "";
      image.name = image.name || name || "reference.png";
      image.preview_url = "";
      return true;
    };
    for (const [index, sheet] of (refs.ingredients_sheets || []).entries()) {
      const image = sheet?.image || {};
      if (await saveImageObject(image, "ingredients_sheet", sheet.name || image.name || `Ingredients Sheet ${index + 1}`)) saved += 1;
    }
    const sheetByName = new Map((refs.ingredients_sheets || []).map((sheet) => [String(sheet.name || "").trim().toLowerCase(), sheet]));
    for (const subject of (refs.subjects || [])) {
      const matchingSheet = sheetByName.get(String(subject.name || "").trim().toLowerCase());
      if (matchingSheet?.image?.path && (subject.image?.data || !subject.image?.path)) {
        subject.image = { ...(matchingSheet.image || {}) };
      } else if (subject.image && await saveImageObject(subject.image, "subject", subject.name || subject.image.name || "subject")) {
        saved += 1;
      }
    }
    if (refs.subject?.image?.data) {
      const firstSubjectImage = refs.subjects?.[0]?.image || null;
      if (firstSubjectImage?.path) refs.subject.image = { ...firstSubjectImage };
      else if (await saveImageObject(refs.subject.image, "subject", refs.subjects?.[0]?.name || "subject")) saved += 1;
    }
    for (const location of (refs.locations || [])) {
      if (location.image && await saveImageObject(location.image, "location", location.name || location.image.name || "location")) saved += 1;
    }
    for (const segment of allEditableSegments()) {
      const sheet = ingredientsSheetForSegment(segment, refs);
      if (sheet?.image?.path && segment.custom_image_data) {
        segment.custom_image_path = sheet.image.path;
        segment.custom_image_data = "";
        segment.custom_image_name = sheet.image.name || sheet.name || segment.custom_image_name || "ingredients_reference.png";
      }
    }
    state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
    return saved;
  }

  function activeProjectFolderForSave() {
    const folder = String(state.projectFolder || "").trim();
    if (folder) {
      projectInput.value = folder;
      setWidgetValue(node, "project_folder", folder);
    }
    return folder;
  }

  async function projectContextFilesForSessionSave() {
    const files = {};
    const entries = [
      ["storyconcept.txt", storyIdeaInput.value || state.storyIdeaPath],
      ["subjectsandscenes.txt", subjectSceneInput.value || state.subjectScenePath],
      ["themestyle.txt", themeStyleInput.value || state.themeStylePath],
    ];
    for (const [filename, path] of entries) {
      let content = await loadContextTextQuiet(path);
      // Subject/scene context is also represented by the reference builder.
      // Keep that canonical data available even when the legacy text file was
      // never populated or points at an old shared TextFiles location.
      if (filename === "subjectsandscenes.txt" && !content) content = referenceBuilderSubjectLocationText();
      files[filename] = String(content || "").trim();
    }
    return files;
  }

  async function saveSession(options = {}) {
    try {
      updateActiveFromInputs();
      saveI2VVideoSettingsFromPanel();
      // Quick Save must capture the live builder panels before serializing the
      // session, including MiniMax's scene/project-scoped controls.
      if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
        saveMiniMaxH3SettingsFromPanel();
        saveMiniMaxSceneInputsFromPanel();
      }
      ensureAllSegmentRuntimeFields();
      const projectFolder = activeProjectFolderForSave();
      if (!projectFolder) {
        const message = "Create a new project, load a project, or use Save Project As before Quick Save.";
        if (options.throwOnError) throw new Error(message);
        if (!options.quiet) toast(message, true);
        return null;
      }
      await persistIngredientsSheetImages(projectFolder);
      const data = await saveBuilderSessionJson({
        audio_path: audioInput.value,
        project_folder: projectFolder,
        session: currentSessionData(),
        project_context_files: await projectContextFilesForSessionSave(),
      }, 60000);
      await syncPromptJsonFromSegments("session save");
      await syncI2VMotionJsonFromSegments("session save");
      await syncLyricAndSubjectNoteFiles("session save");
      state.projectFolder = data.project_folder || "";
      state.sessionPath = data.session_path || "";
      state.srtPath = data.srt_path || "";
      // The live builder state is already the source of the session we just
      // saved. Rehydrating that same payload here is redundant and can make a
      // completed save appear to fail if an unrelated panel normalizer rejects
      // one of its values. Loading a project still performs the full hydrate;
      // an explicit caller may request it here when that behavior is needed.
      if (data.session && options.refreshFromSavedSession === true) {
        state.miniMaxH3Settings = cloneMiniMaxH3Settings(data.session.minimax_h3_settings || state.miniMaxH3Settings);
        if (data.session.minimax_h3_settings?.render_pass == null && data.session.minimax_h3_settings?.ref_pass_mode == null) {
          state.miniMaxH3Settings.render_pass = (data.session.minimax_h3_advanced_two_pass ?? data.session.minimax_h3_three_pass)
            ? "three_pass"
            : data.session.minimax_h3_two_pass ? "two_pass" : state.miniMaxH3Settings.render_pass;
        }
        state.miniMaxH3TwoPassEnabled = state.miniMaxH3Settings.render_pass === "two_pass";
        state.miniMaxH3ThreePassEnabled = state.miniMaxH3Settings.render_pass === "three_pass";
        state.overlaySegments = Array.isArray(data.session.overlay_segments) ? data.session.overlay_segments : state.overlaySegments;
        state.overlaySegments.forEach(normalizeOverlayClip);
        state.overlayTrack = normalizeOverlayTrackState(data.session.overlay_track || state.overlayTrack);
        ensureAllSegmentRuntimeFields();
        state.activeTrack = data.session.active_track || state.activeTrack || "base";
        state.timingFrozen = Boolean(data.session.timing_frozen);
        state.srtMode = Boolean(data.session.srt_mode);
        state.promptJsonPath = data.session.prompt_json_path || state.promptJsonPath;
        state.i2vMotionJsonPath = data.session.i2v_motion_json_path || state.i2vMotionJsonPath;
        if (Object.prototype.hasOwnProperty.call(data.session, "image_trigger_phrase")) state.imageTriggerPhrase = data.session.image_trigger_phrase || "";
        if (Object.prototype.hasOwnProperty.call(data.session, "video_trigger_phrase")) state.videoTriggerPhrase = data.session.video_trigger_phrase || "";
        state.defaultFacialPerformance = data.session.default_facial_performance || data.session.defaultFacialPerformance || state.defaultFacialPerformance || "";
        state.defaultFacialPerformanceCustom = data.session.default_facial_performance_custom || data.session.defaultFacialPerformanceCustom || state.defaultFacialPerformanceCustom || "";
        state.useI2VPromptEnhancementPass = data.session.use_i2v_prompt_enhancement_pass ?? state.useI2VPromptEnhancementPass ?? false;
        state.failOnInvalidPromptFormats = data.session.fail_on_invalid_prompt_formats ?? state.failOnInvalidPromptFormats ?? false;
        state.autoChainLastFrame = data.session.auto_chain_last_frame ?? state.autoChainLastFrame ?? false;
        state.imageContinuityEnabled = data.session.image_continuity_enabled ?? state.imageContinuityEnabled ?? false;
        state.imageContinuityStrength = data.session.image_continuity_strength || state.imageContinuityStrength || "balanced";
        imageContinuityEnabled.input.checked = Boolean(state.imageContinuityEnabled); imageContinuityStrength.value = state.imageContinuityStrength;
        state.continuityMode = normalizeContinuityMode(data.session.continuity_mode || state.continuityMode, state.autoChainLastFrame);
        state.autoChainLastFrame = state.continuityMode === "i2v_chain";
        state.autoImg2ImgStartStep = normalizeAutoImg2ImgStartStep(data.session.auto_img2img_start_step ?? state.autoImg2ImgStartStep);
        state.autoImg2ImgCreativity = normalizeAutoImg2ImgCreativity(data.session.auto_img2img_creativity ?? state.autoImg2ImgCreativity);
        state.autoChainStyle = data.session.auto_chain_style || state.autoChainStyle || "continuous";
        state.autoChainDirection = data.session.auto_chain_direction || state.autoChainDirection || "";
        state.autoChainTransitionLoraPrompt = data.session.auto_chain_transition_lora_prompt ?? state.autoChainTransitionLoraPrompt ?? false;
        state.autoChainTransitionTrigger = data.session.auto_chain_transition_trigger || state.autoChainTransitionTrigger || "zhuanchang";
        state.useVrgdgTextContext = data.session.use_vrgdg_text_context ?? state.useVrgdgTextContext;
        state.themeStylePath = data.session.theme_style_path || state.themeStylePath;
        state.storyIdeaPath = data.session.story_idea_path || state.storyIdeaPath;
        state.subjectScenePath = data.session.subject_scene_path || state.subjectScenePath;
        state.llmApiProvider = data.session.llm_api_provider || state.llmApiProvider || "openai";
        state.llmApiModel = data.session.llm_api_model || state.llmApiModel || "";
        state.llmApiKeyProject = data.session.llm_api_key_project || "";
        if (state.llmApiKeyProject) state.llmApiKey = state.llmApiKeyProject;
        state.ownServerUrl = data.session.own_server_url || state.ownServerUrl || "http://127.0.0.1:8000/v1";
        state.ownServerModel = data.session.own_server_model || state.ownServerModel || "";
        state.ownServerApiKeyProject = data.session.own_server_api_key_project || "";
        if (state.ownServerApiKeyProject) state.ownServerApiKey = state.ownServerApiKeyProject;
        state.ownServerOutputTokenLimit = normalizeOutputTokenLimit(data.session.own_server_output_token_limit ?? state.ownServerOutputTokenLimit);
        state.ownServerTimeoutMinutes = normalizeOwnServerTimeoutMinutes(
          data.session.own_server_timeout_minutes
          ?? (Number(data.session.own_server_timeout) > 15 ? Number(data.session.own_server_timeout) / 60 : data.session.own_server_timeout)
          ?? state.ownServerTimeoutMinutes,
        );
        state.builderAgentMessages = Array.isArray(data.session.builder_agent_messages) ? data.session.builder_agent_messages : state.builderAgentMessages || [];
        state.builderAgentAutoApply = data.session.builder_agent_auto_apply ?? state.builderAgentAutoApply ?? false;
        state.builderAgentPurpose = data.session.builder_agent_purpose || state.builderAgentPurpose || "scene_work";
        state.builderAgentReferenceImages = Array.isArray(data.session.builder_agent_reference_images) ? data.session.builder_agent_reference_images : state.builderAgentReferenceImages || [];
        state.builderStorySourcePath = data.session.builder_story_source_path || state.builderStorySourcePath || "";
        state.builderStoryReferenceImages = Array.isArray(data.session.builder_story_reference_images) ? data.session.builder_story_reference_images : state.builderStoryReferenceImages || [];
        state.builderStoryReferenceNotes = data.session.builder_story_reference_notes || state.builderStoryReferenceNotes || "";
        state.builderStoryLayer = normalizeBuilderStoryLayer(data.session.builder_story_layer || {});
        state.builderStoryboardDefaults = normalizeBuilderStoryboardDefaults(data.session.builder_storyboard_defaults || data.session.builderStoryboardDefaults || {});
        state.autoBuildPreparation = normalizeAutoBuildPreparation(data.session.auto_build_preparation || data.session.autoBuildPreparation || {});
        state.wizardBetaDraft = data.session.wizard_beta_draft || null;
        state.renderLogs = normalizeRenderLogs(data.session.render_logs || state.renderLogs);
        state.activeRenderLogId = data.session.active_render_log_id || state.renderLogs[state.renderLogs.length - 1]?.id || "";
        state.textGemmaRunner = data.session.text_gemma_runner || state.textGemmaRunner || "builtin";
        state.gemmaContextLimit = normalizeGemmaContextLimit(data.session.gemma_context_limit ?? data.session.n_ctx ?? data.session.llm_max_tokens ?? data.session.llmMaxTokens ?? state.gemmaContextLimit);
        state.gemmaOutputTokenLimit = normalizeOutputTokenLimit(data.session.gemma_output_token_limit ?? data.session.llm_max_tokens ?? data.session.llmMaxTokens ?? state.gemmaOutputTokenLimit);
        state.gemmaGpuLayers = normalizeGemmaGpuLayers(data.session.gemma_gpu_layers ?? data.session.n_gpu_layers ?? state.gemmaGpuLayers);
        state.lmStudioBaseUrl = data.session.lm_studio_base_url || state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1";
        state.lmStudioModel = data.session.lm_studio_model || state.lmStudioModel || "";
        state.lmStudioApiKey = data.session.lm_studio_api_key || state.lmStudioApiKey || "";
        state.lmStudioContextLimit = normalizeLmStudioContextLimit(data.session.lm_studio_context_limit ?? state.lmStudioContextLimit);
        state.lmStudioOutputTokenLimit = normalizeOutputTokenLimit(data.session.lm_studio_output_token_limit ?? data.session.llm_max_tokens ?? data.session.llmMaxTokens ?? state.lmStudioOutputTokenLimit);
        state.notificationSettings = normalizeNotificationSettings(data.session.notification_settings || state.notificationSettings);
        state.automaticMemoryCleanup = setBuilderAutomaticMemoryCleanupEnabled(data.session.automatic_memory_cleanup ?? state.automaticMemoryCleanup ?? false);
        state.sceneRenderWaitHours = normalizeSceneRenderWaitHours(data.session.scene_render_wait_hours ?? state.sceneRenderWaitHours);
        state.waveformMode = data.session.waveform_mode || state.waveformMode;
        state.snapToBeats = data.session.snap_to_beats ?? state.snapToBeats;
        state.showTimelineSceneNotes = data.session.show_timeline_scene_notes ?? state.showTimelineSceneNotes ?? false;
        state.showTimelineVideoNotes = data.session.show_timeline_video_notes ?? state.showTimelineVideoNotes ?? false;
        state.showTimelineLyricNotes = data.session.show_timeline_lyric_notes ?? state.showTimelineLyricNotes ?? false;
        state.selectedTimelineRange = normalizeTimelineRange(data.session.selected_timeline_range || state.selectedTimelineRange);
        state.timelineMarkers = normalizeTimelineMarkers(data.session.timeline_markers || state.timelineMarkers);
        state.activeTimelineMarkerId = data.session.active_timeline_marker_id || state.activeTimelineMarkerId || "";
        state.peaks = Array.isArray(data.session.audio_peaks) ? data.session.audio_peaks : state.peaks;
        state.beats = Array.isArray(data.session.beat_markers) ? data.session.beat_markers : state.beats;
        state.detectedTempoBpm = Math.max(0, Number(data.session.detected_tempo_bpm ?? state.detectedTempoBpm ?? 0));
        state.beatCalibration = data.session.beat_calibration || null;
        setBeatMarkersVisible(data.session.show_beat_markers ?? state.showBeatMarkers);
        state.leftPanelWidth = data.session.left_panel_width || state.leftPanelWidth;
        state.leftPanelTab = data.session.left_panel_tab === "tools" || data.session.left_panel_tab === "luts" ? data.session.left_panel_tab : "scenes";
        state.rightPanelWidth = data.session.right_panel_width || state.rightPanelWidth;
        state.timelinePanelHeight = data.session.timeline_panel_height || state.timelinePanelHeight;
        state.timelineZoom = data.session.timeline_zoom || state.timelineZoom;
        state.autoSaveEnabled = data.session.auto_save_enabled ?? state.autoSaveEnabled;
        state.videoType = normalizeVideoType(data.session.video_type || data.session.videoType || state.videoType);
        state.projectVideoEngine = normalizeProjectVideoEngine(data.session.video_engine ?? state.projectVideoEngine);
        state.imageModelMode = data.session.image_model_mode || data.session.flux_klein_settings?.image_model_mode || state.imageModelMode || "zimage";
        state.pxPerSecond = state.timelineZoom;
        waveformModeSelect.value = state.waveformMode;
        snapToBeatsControl.input.checked = Boolean(state.snapToBeats);
        syncSceneNoteControls();
        syncVideoNoteControls();
        syncLyricNoteControls();
        autoSaveControl.input.checked = Boolean(state.autoSaveEnabled);
        syncVideoTypeControl();
        syncProjectVideoEngineUI();
        syncLeftPanelTabs();
        applyLayoutSizes();
        state.zimageSettings = scrubGlobalImageToImageSourceForProject(cloneZImageSettings(data.session.zimage_settings || state.zimageSettings), state.projectFolder);
        state.referenceKrea2Settings = cloneKrea2ReferenceSettings(data.session.reference_krea2_settings || state.referenceKrea2Settings);
        state.fluxKleinSettings = data.session.flux_klein_settings || state.fluxKleinSettings;
        state.flowGptBrowserSettings = cloneFlowGptBrowserSettings(data.session.flow_gpt_browser_settings || state.flowGptBrowserSettings);
        state.nbImageSettings = data.session.nb_image_settings || state.nbImageSettings;
        state.ernieImageSettings = scrubGlobalImageToImageSourceForProject(cloneErnieImageSettings(data.session.ernie_image_settings || state.ernieImageSettings), state.projectFolder);
        state.krea2TwoPassSettings = scrubGlobalImageToImageSourceForProject(cloneKrea2TwoPassSettings(data.session.krea2_2pass_settings || state.krea2TwoPassSettings), state.projectFolder);
        state.useFluxGlobalImageIngredients = Boolean(data.session.use_flux_global_image_ingredients);
        state.fluxGlobalImageIngredients = Array.isArray(data.session.flux_global_image_ingredients) ? data.session.flux_global_image_ingredients : [];
        state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(data.session.flux_reference_builder);
        state.idLoraReferenceBuilder = normalizeIdLoraReferenceBuilder(data.session.id_lora_reference_builder || data.session.idLoraReferenceBuilder);
        state.lyricMapper = normalizeLyricMapper(data.session.lyric_mapper);
        state.zEnhanceSettings = data.session.z_enhance_settings || state.zEnhanceSettings;
        state.videoModelMode = data.session.video_model_mode || state.videoModelMode || "i2v";
        state.i2vVideoSettings = cloneI2VVideoSettings(data.session.i2v_video_settings || state.i2vVideoSettings);
        state.promptToolsHintPrefs = data.session.prompt_tools_hint_prefs || state.promptToolsHintPrefs || {};
        syncZImageSettingsPanel();
        syncFluxKleinPanel();
        syncErnieImagePanel();
        syncKrea2TwoPassPanel();
        syncI2VVideoSettingsPanel();
        syncVideoModePanel();
        syncInspector();
      }
      projectInput.value = state.projectFolder;
      setWidgetValue(node, "project_folder", state.projectFolder);
      setWidgetValue(node, "session_path", state.sessionPath);
      setWidgetValue(node, "srt_path", state.srtPath);
      rememberLastProject(state.projectFolder);
      if (!options.quiet) toast(`Saved builder session and SRT.\n${state.srtPath}`);
      return data;
    } catch (error) {
      if (options.throwOnError) throw error;
      toast(String(error?.message || error), true);
      return null;
    }
  }

  async function saveSessionForSceneVideo() {
    updateActiveFromInputs();
    saveI2VVideoSettingsFromPanel();
    if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") saveMiniMaxH3SettingsFromPanel();
    const projectFolder = activeProjectFolderForSave();
    if (!projectFolder) throw new Error("Create or load a project before rendering scene videos.");
    await persistIngredientsSheetImages(projectFolder);
    const data = await saveBuilderSessionJson({
      audio_path: audioInput.value,
      project_folder: projectFolder,
      session: currentSessionData(),
      project_context_files: await projectContextFilesForSessionSave(),
    }, 60000);
    await syncPromptJsonFromSegments("scene video save");
    await syncI2VMotionJsonFromSegments("scene video save");
    await syncLyricAndSubjectNoteFiles("scene video save");
    state.projectFolder = data.project_folder || state.projectFolder;
    state.sessionPath = data.session_path || state.sessionPath;
    state.srtPath = data.srt_path || state.srtPath;
    projectInput.value = state.projectFolder;
    srtInput.value = state.srtPath;
    setWidgetValue(node, "project_folder", state.projectFolder);
    setWidgetValue(node, "session_path", state.sessionPath);
    setWidgetValue(node, "srt_path", state.srtPath);
    rememberLastProject(state.projectFolder);
    return state.srtPath;
  }

  async function recoverSceneVideosFromProject(options = {}) {
    const projectFolder = String(options.projectFolder || state.projectFolder || projectInput.value || "").trim();
    if (!projectFolder) return 0;
    const scan = await postJson("/vrgdg/music_builder/scan_scene_videos", {
      project_folder: projectFolder,
    });
    const videos = scan.videos || {};
    const videoThumbnails = scan.video_thumbnails || {};
    const videoBackups = scan.video_backups || {};
    const videoBackupThumbnails = scan.video_backup_thumbnails || {};
    const recoveredFromScratch = scan.recovered_from_scratch || {};
    let restored = 0;
    const applyFoundVideo = (segment, sceneKey) => {
      if (!segment) return;
      const videoPath = videos[sceneKey] || "";
      segment.video_backup_paths = Array.isArray(videoBackups[sceneKey]) ? videoBackups[sceneKey] : [];
      segment.video_backup_thumbnail_paths = Array.isArray(videoBackupThumbnails[sceneKey]) ? videoBackupThumbnails[sceneKey] : [];
      segment.video_thumbnail_path = videoThumbnails[sceneKey] || segment.video_thumbnail_path || "";
      if (!videoPath) return;
      const currentPath = String(segment.video_path || "").trim();
      const changed = mediaPathKey(currentPath) !== mediaPathKey(videoPath);
      segment.video_path = videoPath;
      segment.video_folder = scan.video_folder || segment.video_folder || "";
      segment.video_status = "done";
      segment.preview_mode = "video";
      normalizeSegmentVideoHistory(segment);
      if (changed) restored += 1;
    };
    state.segments.forEach((segment, index) => applyFoundVideo(segment, String(index + 1)));
    state.overlaySegments.forEach((segment) => applyFoundVideo(segment, String(sceneSlotNumber(segment))));
    if (restored) {
      console.log(`[VRGDG Music Builder] Recovered ${restored} scene video path(s) from project folder.`);
      if (options.renderAfter !== false) {
        syncInspector();
        render();
      }
      if (options.autoSave) {
        await autoSaveSessionQuiet(options.autoSaveReason || "recovered scene videos from project folder");
      }
      if (options.toast) {
        const scratchCount = Object.keys(recoveredFromScratch).length;
        const scratchLine = scratchCount ? `\nCopied ${scratchCount} completed render${scratchCount === 1 ? "" : "s"} out of temporary video output folders.` : "";
        toast(`Recovered ${restored} scene video${restored === 1 ? "" : "s"} from the project folder.${scratchLine}`);
      }
    }
    return restored;
  }

  async function loadSessionFromProject(projectFolder) {
    try {
      await restoreBrowserAiDownloadsQuietly();
      closeBeatCalibrationWizard();
      const folder = String(projectFolder || "").trim();
      if (!folder) {
        toast("Choose a project folder that contains vrgdg_builder_session.json.", true);
        return false;
      }
      const data = await postJson("/vrgdg/music_builder/load_session", {
        project_folder: folder,
      });
      const session = data.session || {};
      syncBuilderSessionSaveRevision(session.builder_save_revision);
      faceFixTool.reset?.();
      pushHistory();
      state.miniMaxH3Settings = cloneMiniMaxH3Settings(session.minimax_h3_settings || {});
      if (session.minimax_h3_settings?.render_pass == null && session.minimax_h3_settings?.ref_pass_mode == null) {
        state.miniMaxH3Settings.render_pass = (session.minimax_h3_advanced_two_pass ?? session.minimax_h3_three_pass)
          ? "three_pass"
          : session.minimax_h3_two_pass ? "two_pass" : state.miniMaxH3Settings.render_pass;
      }
      state.miniMaxH3TwoPassEnabled = state.miniMaxH3Settings.render_pass === "two_pass";
      state.miniMaxH3ThreePassEnabled = state.miniMaxH3Settings.render_pass === "three_pass";
      state.segments = Array.isArray(session.segments) ? session.segments : [];
      state.overlaySegments = Array.isArray(session.overlay_segments) ? session.overlay_segments : [];
      state.overlaySegments.forEach(normalizeOverlayClip);
      state.overlayTrack = normalizeOverlayTrackState(session.overlay_track || {});
      state.repairedSegmentIdCount = 0;
      // Load scene-keyed mappings before ID repair so a repaired duplicate keeps
      // the same subject, location, trigger, and ingredients assignments.
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(session.flux_reference_builder || {});
      ensureAllSegmentRuntimeFields();
      state.projectFolder = data.project_folder || folder;
      state.sessionPath = data.session_path || "";
      state.srtPath = data.srt_path || session.srt_path || state.srtPath;
      state.timingFrozen = Boolean(session.timing_frozen);
      state.srtMode = Boolean(session.srt_mode);
      state.promptJsonPath = session.prompt_json_path || "";
      state.i2vMotionJsonPath = session.i2v_motion_json_path || "";
      state.imageTriggerPhrase = session.image_trigger_phrase || "";
      state.videoTriggerPhrase = session.video_trigger_phrase || "";
      state.defaultFacialPerformance = session.default_facial_performance || session.defaultFacialPerformance || "";
      state.defaultFacialPerformanceCustom = session.default_facial_performance_custom || session.defaultFacialPerformanceCustom || "";
      state.useI2VPromptEnhancementPass = session.use_i2v_prompt_enhancement_pass ?? state.useI2VPromptEnhancementPass ?? false;
      state.failOnInvalidPromptFormats = session.fail_on_invalid_prompt_formats ?? state.failOnInvalidPromptFormats ?? false;
      state.autoChainLastFrame = session.auto_chain_last_frame ?? state.autoChainLastFrame ?? false;
      state.imageContinuityEnabled = session.image_continuity_enabled ?? state.imageContinuityEnabled ?? false;
      state.imageContinuityStrength = session.image_continuity_strength || state.imageContinuityStrength || "balanced";
      imageContinuityEnabled.input.checked = Boolean(state.imageContinuityEnabled); imageContinuityStrength.value = state.imageContinuityStrength;
      state.continuityMode = normalizeContinuityMode(session.continuity_mode || state.continuityMode, state.autoChainLastFrame);
      state.autoChainLastFrame = state.continuityMode === "i2v_chain";
      state.autoImg2ImgStartStep = normalizeAutoImg2ImgStartStep(session.auto_img2img_start_step ?? state.autoImg2ImgStartStep);
      state.autoImg2ImgCreativity = normalizeAutoImg2ImgCreativity(session.auto_img2img_creativity ?? state.autoImg2ImgCreativity);
      state.autoChainStyle = session.auto_chain_style || state.autoChainStyle || "continuous";
      state.autoChainDirection = session.auto_chain_direction || state.autoChainDirection || "";
      state.autoChainTransitionLoraPrompt = session.auto_chain_transition_lora_prompt ?? state.autoChainTransitionLoraPrompt ?? false;
      state.autoChainTransitionTrigger = session.auto_chain_transition_trigger || state.autoChainTransitionTrigger || "zhuanchang";
      state.useVrgdgTextContext = session.use_vrgdg_text_context ?? true;
      state.themeStylePath = session.theme_style_path || "";
      state.storyIdeaPath = session.story_idea_path || "";
      state.subjectScenePath = session.subject_scene_path || "";
      state.llmApiProvider = session.llm_api_provider || state.llmApiProvider || "openai";
      state.llmApiModel = session.llm_api_model || state.llmApiModel || "";
      state.llmApiKeyProject = session.llm_api_key_project || "";
      if (state.llmApiKeyProject) state.llmApiKey = state.llmApiKeyProject;
      state.ownServerUrl = session.own_server_url || state.ownServerUrl || "http://127.0.0.1:8000/v1";
      state.ownServerModel = session.own_server_model || state.ownServerModel || "";
      state.ownServerApiKeyProject = session.own_server_api_key_project || "";
      // Never carry a session-only credential into a newly loaded project's
      // arbitrary server URL. A project key is restored only when explicitly saved.
      state.ownServerApiKey = state.ownServerApiKeyProject;
      state.ownServerOutputTokenLimit = normalizeOutputTokenLimit(session.own_server_output_token_limit ?? state.ownServerOutputTokenLimit);
      state.ownServerTimeoutMinutes = normalizeOwnServerTimeoutMinutes(
        session.own_server_timeout_minutes
        ?? (Number(session.own_server_timeout) > 15 ? Number(session.own_server_timeout) / 60 : session.own_server_timeout)
        ?? state.ownServerTimeoutMinutes,
      );
      state.builderAgentMessages = Array.isArray(session.builder_agent_messages) ? session.builder_agent_messages : [];
      state.builderAgentAutoApply = Boolean(session.builder_agent_auto_apply);
      state.builderAgentPurpose = session.builder_agent_purpose || "scene_work";
      state.builderAgentReferenceImages = Array.isArray(session.builder_agent_reference_images) ? session.builder_agent_reference_images : [];
      state.builderStorySourcePath = session.builder_story_source_path || projectContextPath("AgentStorySource.txt") || "";
      state.builderStoryReferenceImages = Array.isArray(session.builder_story_reference_images) ? session.builder_story_reference_images : [];
      state.builderStoryReferenceNotes = session.builder_story_reference_notes || "";
      state.builderStoryLayer = normalizeBuilderStoryLayer(session.builder_story_layer || {});
      state.builderStoryboardDefaults = normalizeBuilderStoryboardDefaults(session.builder_storyboard_defaults || session.builderStoryboardDefaults || {});
      state.autoBuildPreparation = normalizeAutoBuildPreparation(session.auto_build_preparation || session.autoBuildPreparation || {});
      state.wizardBetaDraft = session.wizard_beta_draft || null;
      resetBuilderETA();
      state.renderLogs = normalizeRenderLogs(session.render_logs);
      state.activeRenderLogId = session.active_render_log_id || state.renderLogs[state.renderLogs.length - 1]?.id || "";
      state.textGemmaRunner = session.text_gemma_runner || state.textGemmaRunner || "builtin";
      state.gemmaContextLimit = normalizeGemmaContextLimit(session.gemma_context_limit ?? session.n_ctx ?? session.llm_max_tokens ?? session.llmMaxTokens ?? state.gemmaContextLimit);
      state.gemmaOutputTokenLimit = normalizeOutputTokenLimit(session.gemma_output_token_limit ?? session.llm_max_tokens ?? session.llmMaxTokens ?? state.gemmaOutputTokenLimit);
      state.gemmaGpuLayers = normalizeGemmaGpuLayers(session.gemma_gpu_layers ?? session.n_gpu_layers ?? state.gemmaGpuLayers);
      state.lmStudioBaseUrl = session.lm_studio_base_url || state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1";
      state.lmStudioModel = session.lm_studio_model || state.lmStudioModel || "";
      state.lmStudioApiKey = session.lm_studio_api_key || state.lmStudioApiKey || "";
      state.lmStudioContextLimit = normalizeLmStudioContextLimit(session.lm_studio_context_limit ?? state.lmStudioContextLimit);
      state.lmStudioOutputTokenLimit = normalizeOutputTokenLimit(session.lm_studio_output_token_limit ?? session.llm_max_tokens ?? session.llmMaxTokens ?? state.lmStudioOutputTokenLimit);
      state.notificationSettings = normalizeNotificationSettings(session.notification_settings || state.notificationSettings);
      state.automaticMemoryCleanup = setBuilderAutomaticMemoryCleanupEnabled(session.automatic_memory_cleanup ?? false);
      state.sceneRenderWaitHours = normalizeSceneRenderWaitHours(session.scene_render_wait_hours ?? state.sceneRenderWaitHours);
      state.waveformMode = session.waveform_mode || state.waveformMode || "medium";
      state.snapToBeats = session.snap_to_beats ?? state.snapToBeats ?? true;
      state.showTimelineSceneNotes = session.show_timeline_scene_notes ?? state.showTimelineSceneNotes ?? false;
      state.showTimelineVideoNotes = session.show_timeline_video_notes ?? state.showTimelineVideoNotes ?? false;
      state.showTimelineLyricNotes = session.show_timeline_lyric_notes ?? state.showTimelineLyricNotes ?? false;
      state.selectedTimelineRange = normalizeTimelineRange(session.selected_timeline_range || {});
      state.timelineMarkers = normalizeTimelineMarkers(session.timeline_markers || []);
      state.activeTimelineMarkerId = session.active_timeline_marker_id || "";
      state.audioDuration = Math.max(0, Number(session.audio_duration || 0));
      if (state.audioDuration > 0) enforceAudioTimelineEnd();
      state.peaks = Array.isArray(session.audio_peaks) ? session.audio_peaks : state.peaks;
      state.beats = Array.isArray(session.beat_markers) ? session.beat_markers : state.beats;
      state.detectedTempoBpm = Math.max(0, Number(session.detected_tempo_bpm ?? state.detectedTempoBpm ?? 0));
      state.beatCalibration = session.beat_calibration || null;
      setBeatMarkersVisible(session.show_beat_markers ?? state.showBeatMarkers ?? false);
      state.leftPanelWidth = session.left_panel_width || state.leftPanelWidth || 260;
      state.leftPanelTab = session.left_panel_tab === "tools" || session.left_panel_tab === "luts" ? session.left_panel_tab : "scenes";
      state.rightPanelWidth = session.right_panel_width || state.rightPanelWidth || 360;
      state.timelinePanelHeight = session.timeline_panel_height || state.timelinePanelHeight || 300;
      state.timelineZoom = session.timeline_zoom || state.timelineZoom || 45;
      state.autoSaveEnabled = session.auto_save_enabled ?? state.autoSaveEnabled ?? true;
      state.videoType = normalizeVideoType(session.video_type || session.videoType || state.videoType);
      state.projectVideoEngine = normalizeProjectVideoEngine(session.video_engine);
      state.imageModelMode = session.image_model_mode || session.flux_klein_settings?.image_model_mode || state.imageModelMode || "zimage";
      state.pxPerSecond = state.timelineZoom;
      waveformModeSelect.value = state.waveformMode;
      snapToBeatsControl.input.checked = Boolean(state.snapToBeats);
      syncSceneNoteControls();
      syncVideoNoteControls();
      syncLyricNoteControls();
      autoSaveControl.input.checked = Boolean(state.autoSaveEnabled);
      syncVideoTypeControl();
      syncProjectVideoEngineUI();
      syncLeftPanelTabs();
      applyLayoutSizes();
      state.zimageSettings = scrubGlobalImageToImageSourceForProject(cloneZImageSettings(session.zimage_settings || state.zimageSettings), state.projectFolder);
      state.referenceKrea2Settings = cloneKrea2ReferenceSettings(session.reference_krea2_settings || state.referenceKrea2Settings);
      state.fluxKleinSettings = session.flux_klein_settings || state.fluxKleinSettings;
      state.flowGptBrowserSettings = cloneFlowGptBrowserSettingsForLoadedProject(session.flow_gpt_browser_settings);
      state.nbImageSettings = session.nb_image_settings || state.nbImageSettings;
      state.ernieImageSettings = scrubGlobalImageToImageSourceForProject(cloneErnieImageSettings(session.ernie_image_settings || state.ernieImageSettings), state.projectFolder);
      state.krea2TwoPassSettings = scrubGlobalImageToImageSourceForProject(cloneKrea2TwoPassSettings(session.krea2_2pass_settings || state.krea2TwoPassSettings), state.projectFolder);
      state.useFluxGlobalImageIngredients = Boolean(session.use_flux_global_image_ingredients);
      state.fluxGlobalImageIngredients = Array.isArray(session.flux_global_image_ingredients) ? session.flux_global_image_ingredients : [];
      state.idLoraReferenceBuilder = normalizeIdLoraReferenceBuilder(session.id_lora_reference_builder || session.idLoraReferenceBuilder);
      state.lyricMapper = normalizeLyricMapper(session.lyric_mapper);
      state.zEnhanceSettings = session.z_enhance_settings || state.zEnhanceSettings;
      state.videoModelMode = session.video_model_mode || state.videoModelMode || "i2v";
      state.i2vVideoSettings = cloneI2VVideoSettings(session.i2v_video_settings || state.i2vVideoSettings);
      state.promptToolsHintPrefs = session.prompt_tools_hint_prefs || state.promptToolsHintPrefs || {};
      state.audioPath = String(session.audio_path || "");
      if (session.audio_path) {
        audioInput.value = session.audio_path;
        setWidgetValue(node, "audio_path", session.audio_path);
        try {
          const audioData = await postJson("/vrgdg/music_builder/analyze_audio", {
            audio_path: session.audio_path,
            project_folder: projectInput.value || state.projectFolder || "",
            target_peaks: 1800,
          });
          audioInput.value = audioData.audio_path || session.audio_path;
          setWidgetValue(node, "audio_path", audioInput.value);
          state.duration = Number(audioData.duration || 0);
          state.audioDuration = Number(audioData.duration || 0);
          enforceAudioTimelineEnd();
          state.peaks = Array.isArray(audioData.peaks) && audioData.peaks.length ? audioData.peaks : state.peaks;
          if (!state.beatCalibration) {
            state.beats = Array.isArray(audioData.beats) && audioData.beats.length ? audioData.beats : state.beats;
          }
          showBeatMarkersIfAvailable();
          audio.dataset.path = audioInput.value;
          audio.src = audioUrl(audioInput.value);
          audio.load();
          activateGlobalTimelineAudioPlayback(0);
        } catch (error) {
          toast(`Loaded session, but audio waveform failed:\n${String(error?.message || error)}`, true);
        }
      } else {
        audioInput.value = "";
        setWidgetValue(node, "audio_path", "");
        audio.dataset.path = "";
        audio.removeAttribute("src");
        audio.load();
      }
      try {
        await recoverSceneVideosFromProject({ projectFolder: state.projectFolder, renderAfter: false });
      } catch (error) {
        console.warn("[VRGDG Music Builder] Scene video scan failed:", error);
      }
      projectInput.value = state.projectFolder;
      srtInput.value = state.srtPath;
      setWidgetValue(node, "project_folder", state.projectFolder);
      setWidgetValue(node, "session_path", state.sessionPath);
      setWidgetValue(node, "srt_path", state.srtPath);
      rememberLastProject(state.projectFolder);
      if (!state.i2vMotionJsonPath) {
        try {
          const paths = await postJson("/vrgdg/music_builder/project_prompt_creator_paths", {
            project_folder: state.projectFolder,
          });
          if (paths?.exists?.i2v_motion_notes_path && paths.i2v_motion_notes_path) {
            state.i2vMotionJsonPath = paths.i2v_motion_notes_path;
          }
        } catch (error) {
          console.warn("[VRGDG Music Builder] Could not find project I2V motion notes path during load:", error);
        }
      }
      i2vMotionJsonInput.value = state.i2vMotionJsonPath || i2vMotionJsonInput.value || "";
      if (state.i2vMotionJsonPath && !hasAnyI2VMotionNotes(state.segments)) {
        const restoredNotes = await importI2VMotionJson({ quiet: true, pushHistory: false });
        if (restoredNotes.some((note) => String(note || "").trim())) {
          console.log(`[VRGDG Music Builder] Restored ${restoredNotes.length} I2V motion note(s) from project file.`);
        }
      }
      state.activeTrack = session.active_track || "base";
      state.activeId = state.segments[0]?.id || state.overlaySegments[0]?.id || "";
      syncZImageSettingsPanel();
      syncFluxKleinPanel();
      syncErnieImagePanel();
      syncKrea2TwoPassPanel();
      syncZEnhanceSettingsPanel();
      syncI2VVideoSettingsPanel();
      syncVideoModePanel();
      syncInspector();
      render();
      loadDirtyLatentBadges();
      const repairedSegmentIdCount = Number(state.repairedSegmentIdCount || 0);
      if (repairedSegmentIdCount) {
        await saveSession({ quiet: true, throwOnError: true });
        state.repairedSegmentIdCount = 0;
      }
      toast(repairedSegmentIdCount
        ? `Loaded builder session and repaired ${repairedSegmentIdCount} duplicate scene ID${repairedSegmentIdCount === 1 ? "" : "s"}.\n${state.sessionPath}`
        : `Loaded builder session.\n${state.sessionPath}`);
      return true;
    } catch (error) {
      toast(String(error?.message || error), true);
      return false;
    }
  }

  return {
    activeProjectFolderForSave, addProjectToBatch, autoLoadAll, currentSessionData, getPreferredProjectRoot,
    importI2VMotionJson, importPromptJson, importSceneNotesJson, loadDefaultContextPaths, loadLastProject,
    loadSession, loadSessionFromProject, persistIngredientsSheetImages, projectContextFilesForSessionSave,
    projectListUrl, recoverSceneVideosFromProject, rememberLastProject, renderProjectBatchQueue,
    runClearMemoryWorkflow, runClearMemoryWorkflowQuiet, runImageMemoryCleanupQuiet, runProjectBatchQueue,
    saveSession, saveSessionForSceneVideo, setPreferredProjectRoot, showStartupWelcome, stopCurrentWorkflow,
  };
}
