import { makeButton, makeCheckbox, makeInput, makeSelect } from "./controls.mjs";

export function buildLeftPanel({
  autoLoadAllButton, builderAgentButton, convertLtxPromptsToMiniMaxButton, importImageFolderButton,
  importSceneNotesButton, projectBatchButton, promptCreatorButton, sendToPromptCreatorButton,
  zEnhanceAllToolButton,
}) {
  const main = document.createElement("div");
  main.style.cssText = "display:grid;grid-template-columns:260px 7px minmax(0,1fr) 7px 360px;min-height:0;overflow:hidden;";
  const segmentList = document.createElement("div");
  segmentList.style.cssText = "display:flex;flex-direction:column;min-height:0;border-right:1px solid #27272a;background:#202024;";
  segmentList.style.gridColumn = "1";
  const leftTabBar = document.createElement("div");
  leftTabBar.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;padding:8px 8px 0;background:#202024;flex:0 0 auto;";
  const scenesTabButton = makeButton("Scenes");
  const toolsTabButton = makeButton("Tools");
  const lutsTabButton = makeButton("Post Process");
  const calibrateFirstBeatButton = makeButton("Beat Calibration...");
  const snapAllSceneStartsButton = makeButton("Snap Scene 3+ Starts to Beats");
  calibrateFirstBeatButton.title = "Open three-point beat calibration. Capture the first, middle, and last real beats with the playhead to correct cumulative grid drift.";
  snapAllSceneStartsButton.title = "Snap Scene 3 and every later base scene start to its nearest beat. The Scene 1-to-2 boundary stays fixed; connected previous scene ends follow later shared boundaries.";
  scenesTabButton.title = "Show the vertical scene list.";
  toolsTabButton.title = "Show project tools and prompt handoff actions.";
  lutsTabButton.title = "Browse post-process effects and apply them to the selected scene.";
  const sceneListPane = document.createElement("div");
  sceneListPane.style.cssText = "overflow:auto;padding:10px;min-height:0;flex:1 1 auto;";
  const toolsPane = document.createElement("div");
  toolsPane.style.cssText = "display:none;overflow:auto;padding:10px;min-height:0;flex:1 1 auto;";
  toolsPane.className = "vrgdg-builder-tools-pane";
  const toolIntro = document.createElement("div");
  toolIntro.textContent = "Project tools";
  toolIntro.style.cssText = "font-size:12px;font-weight:900;color:#cffafe;margin:2px 0 8px;";
  const makeToolRow = (button, hint) => {
    const row = document.createElement("div");
    row.className = "vrgdg-builder-tool-row";
    const note = document.createElement("div");
    note.textContent = hint;
    note.className = "vrgdg-builder-tool-hint";
    row.append(button, note);
    return row;
  };
  const beatCalibrationToolRow = makeToolRow(calibrateFirstBeatButton, "Capture a first, middle, and last beat with the playhead, then correct both beat-grid offset and cumulative drift. Scene timing stays unchanged.");
  const projectBatchPanel = document.createElement("div");
  projectBatchPanel.style.cssText = "display:none;margin:0 0 10px;border:1px solid #0891b2;border-radius:7px;background:#082f49;padding:9px;gap:8px;flex-direction:column;";
  const projectBatchTitle = document.createElement("div");
  projectBatchTitle.textContent = "Project Batch";
  projectBatchTitle.style.cssText = "font-size:13px;font-weight:900;color:#cffafe;";
  const projectBatchHelp = document.createElement("div");
  projectBatchHelp.textContent = "Queue saved projects and run Render All on each one in order. The next project opens only after the previous project finishes and saves.";
  projectBatchHelp.style.cssText = "font-size:11px;line-height:1.45;color:#e0f2fe;";
  const projectBatchActions = document.createElement("div");
  projectBatchActions.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;";
  const projectBatchAddCurrent = makeButton("Add Current", "primary");
  const projectBatchAddRecent = makeButton("Add Project");
  const projectBatchAddSession = makeButton("Add Session JSON");
  const projectBatchAddCustom = makeButton("Paste Folder");
  projectBatchActions.append(projectBatchAddCurrent, projectBatchAddRecent, projectBatchAddSession, projectBatchAddCustom);
  const projectBatchOptions = document.createElement("div");
  projectBatchOptions.style.cssText = "display:flex;flex-direction:column;gap:6px;border:1px solid #155e75;border-radius:6px;background:#0f172a;padding:7px;";
  const projectBatchForceVideos = makeCheckbox("Redo scene videos", false);
  const projectBatchContinueOnError = makeCheckbox("Continue if a project fails", true);
  const projectBatchClearMemory = makeCheckbox("Clear memory between projects", true);
  for (const option of [projectBatchForceVideos, projectBatchContinueOnError, projectBatchClearMemory]) {
    option.wrapper.style.fontSize = "11px";
    option.wrapper.style.fontWeight = "600";
  }
  projectBatchOptions.append(projectBatchForceVideos.wrapper, projectBatchContinueOnError.wrapper, projectBatchClearMemory.wrapper);
  const projectBatchQueue = document.createElement("div");
  projectBatchQueue.style.cssText = "display:flex;flex-direction:column;gap:6px;max-height:220px;overflow:auto;border:1px solid #155e75;border-radius:6px;background:#071422;padding:7px;";
  const projectBatchFooter = document.createElement("div");
  projectBatchFooter.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;";
  const projectBatchRun = makeButton("Run Batch", "primary");
  const projectBatchStop = makeButton("Stop Batch");
  projectBatchStop.disabled = true;
  projectBatchFooter.append(projectBatchRun, projectBatchStop);
  const projectBatchStatus = document.createElement("div");
  projectBatchStatus.style.cssText = "font-size:11px;color:#bae6fd;white-space:pre-wrap;line-height:1.35;";
  projectBatchPanel.append(projectBatchTitle, projectBatchHelp, projectBatchActions, projectBatchOptions, projectBatchQueue, projectBatchFooter, projectBatchStatus);
  const beatCalibrationWizard = document.createElement("div");
  beatCalibrationWizard.style.cssText = "display:none;margin:0 0 10px;border:1px solid #0891b2;border-radius:7px;background:#082f49;padding:9px;gap:8px;flex-direction:column;";
  const beatCalibrationTitle = document.createElement("div");
  beatCalibrationTitle.textContent = "Beat Calibration";
  beatCalibrationTitle.style.cssText = "font-size:13px;font-weight:900;color:#cffafe;";
  const beatCalibrationGridType = makeSelect([
    { value: "detected_warp", label: "Warp detected markers" },
    { value: "even_grid", label: "Even BPM grid (CapCut-style test)" },
    { value: "auto_bpm", label: "Auto-detect BPM + chosen start" },
    { value: "capcut_import", label: "Import exact CapCut markers" },
  ], "detected_warp");
  const beatCalibrationGridTypeField = document.createElement("label");
  beatCalibrationGridTypeField.style.cssText = "display:flex;flex-direction:column;gap:3px;font-size:10px;color:#bae6fd;";
  beatCalibrationGridTypeField.append(document.createTextNode("Grid type"), beatCalibrationGridType);
  const beatCalibrationGridTypeHint = document.createElement("div");
  beatCalibrationGridTypeHint.style.cssText = "font-size:9px;line-height:1.35;color:#7dd3fc;";
  const beatCalibrationInstruction = document.createElement("div");
  beatCalibrationInstruction.style.cssText = "font-size:11px;line-height:1.45;color:#e0f2fe;white-space:pre-line;";
  const beatCalibrationAnchors = document.createElement("div");
  beatCalibrationAnchors.style.cssText = "font-size:10px;line-height:1.5;color:#bae6fd;font-variant-numeric:tabular-nums;white-space:pre-line;";
  const beatCalibrationTimecodeInput = makeInput("");
  beatCalibrationTimecodeInput.placeholder = "00:00:00:00";
  beatCalibrationTimecodeInput.title = "CapCut timecode: HH:MM:SS:FF or HH:MM:SS+FF";
  const beatCalibrationFpsInput = makeInput("24");
  beatCalibrationFpsInput.type = "number";
  beatCalibrationFpsInput.min = "1";
  beatCalibrationFpsInput.max = "120";
  beatCalibrationFpsInput.step = "0.001";
  const beatCalibrationTimecodeGrid = document.createElement("div");
  beatCalibrationTimecodeGrid.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 64px;gap:6px;";
  const beatCalibrationTimecodeField = document.createElement("label");
  beatCalibrationTimecodeField.style.cssText = "display:flex;flex-direction:column;gap:3px;font-size:10px;color:#bae6fd;";
  beatCalibrationTimecodeField.append(document.createTextNode("CapCut timecode"), beatCalibrationTimecodeInput);
  const beatCalibrationFpsField = document.createElement("label");
  beatCalibrationFpsField.style.cssText = "display:flex;flex-direction:column;gap:3px;font-size:10px;color:#bae6fd;";
  beatCalibrationFpsField.append(document.createTextNode("FPS"), beatCalibrationFpsInput);
  beatCalibrationTimecodeGrid.append(beatCalibrationTimecodeField, beatCalibrationFpsField);
  const beatCalibrationTimecodeHint = document.createElement("div");
  beatCalibrationTimecodeHint.textContent = "Enter HH:MM:SS:FF or HH:MM:SS+FF. If filled, Capture uses this exact timecode and moves the playhead automatically.";
  beatCalibrationTimecodeHint.style.cssText = "font-size:9px;line-height:1.35;color:#7dd3fc;";
  const beatCalibrationCaptureButton = makeButton("Capture First Beat");
  const beatCalibrationCancelButton = makeButton("Cancel");
  beatCalibrationCaptureButton.style.width = "100%";
  beatCalibrationCancelButton.style.width = "100%";
  const beatCalibrationActions = document.createElement("div");
  beatCalibrationActions.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;";
  beatCalibrationActions.append(beatCalibrationCaptureButton, beatCalibrationCancelButton);
  beatCalibrationWizard.append(beatCalibrationTitle, beatCalibrationGridTypeField, beatCalibrationGridTypeHint, beatCalibrationInstruction, beatCalibrationTimecodeGrid, beatCalibrationTimecodeHint, beatCalibrationAnchors, beatCalibrationActions);
  toolsPane.append(
    toolIntro,
    makeToolRow(promptCreatorButton, "Legacy workflow. Storyboard Builder is the newer, recommended way to plan story beats, references, defaults, and scene prompts."),
    makeToolRow(sendToPromptCreatorButton, "Send this Video Builder timeline back into Prompt Creator as an editable draft."),
    makeToolRow(autoLoadAllButton, "Import the latest Prompt Creator outputs into this project, including timing and prompt data."),
    makeToolRow(importSceneNotesButton, "Load a scene-notes JSON file and map its notes onto the current scenes."),
    makeToolRow(importImageFolderButton, "Choose a folder of numbered images and fill base timeline scenes in numeric order. Requires project audio and existing scenes."),
    makeToolRow(projectBatchButton, "Run Render All across multiple saved projects one after another for overnight batches."),
    projectBatchPanel,
    beatCalibrationToolRow,
    beatCalibrationWizard,
    makeToolRow(snapAllSceneStartsButton, "Snap Scene 3 onward to the nearest valid beat. The Scene 1-to-2 intro boundary stays fixed; connected previous scene ends move with later shared cuts."),
    makeToolRow(convertLtxPromptsToMiniMaxButton, "Convert every populated LTX video prompt into a detailed MiniMax H3 prompt. The global audio file, scene timing, and original LTX prompts stay unchanged."),
    makeToolRow(zEnhanceAllToolButton, "Upscale/enhance every timeline scene that already has an image, using each scene's current T2I/image prompt."),
    makeToolRow(builderAgentButton, "Open Builder Agent for scene help, prompt edits, image references, and project guidance.")
  );

  return {
    beatCalibrationAnchors, beatCalibrationCancelButton, beatCalibrationCaptureButton,
    beatCalibrationFpsInput, beatCalibrationGridType, beatCalibrationGridTypeHint, beatCalibrationInstruction,
    beatCalibrationTimecodeGrid, beatCalibrationTimecodeHint, beatCalibrationTimecodeInput,
    beatCalibrationWizard, calibrateFirstBeatButton, leftTabBar, lutsTabButton, main, makeToolRow,
    projectBatchAddCurrent, projectBatchAddCustom, projectBatchAddRecent, projectBatchAddSession,
    projectBatchClearMemory, projectBatchContinueOnError, projectBatchForceVideos, projectBatchPanel,
    projectBatchQueue, projectBatchRun, projectBatchStatus, projectBatchStop, sceneListPane, scenesTabButton,
    segmentList, snapAllSceneStartsButton, toolsPane, toolsTabButton,
  };
}
