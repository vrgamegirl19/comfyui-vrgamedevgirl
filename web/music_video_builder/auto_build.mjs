import { storyboardCameraFlowEntry } from "../storyboard_builder/shot_presets.mjs";
import {
  COMPACT_TOOLBAR_ICONS,
  escapeHtml,
  makeButton,
  makeCheckbox,
  normalizeProjectVideoEngine,
  toast,
} from "./controls.mjs";
import { gemmaBatchFailureStore, recordGemmaBatchFailure, showGemmaBatchFailures } from "./dialogs.mjs";
import { readFileAsDataUrl } from "./media_import.mjs";
import { cloneMiniMaxH3Settings, normalizeMiniMaxH3Voice, normalizeMiniMaxSpeakerAssignments } from "./minimax_h3.mjs";
import {
  autoBuildFingerprint,
  builderMotionSpeedGuidance,
  normalizeAutoBuildPreparation,
  normalizeBuilderStoryboardDefaults,
  normalizeBuilderStoryLayer,
} from "./model_settings.mjs";
import { isInstrumentalLyricText, isRecoverableBuildGemmaError } from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder, normalizeLyricMapper } from "./reference_data.mjs";

export function createAutoBuild({
  allEditableSegments, applyBulkSegmentTimings, assertBatchNotStopped,
  assertMiniMaxH3ReferenceDescriptionsReady, audio, audioInput, autoSaveSessionQuiet, chooseProjectAudioFile,
  createProgressWindow, describeReferenceImageWithGemma, gemmaRunnerLabel, gemmaRunnerLine,
  gemmaVideoAllTextOnly, llmApiVisionModelSelected, miniMaxH3PromptVisionImagesForRunner,
  miniMaxOrderedImageReferenceItemsForSegment, newProject, openGemmaRunnerModal,
  openStoryboardBuilderFromProject, projectInput, pushHistory, render, runMiniMaxH3PromptGeneration,
  saveGemmaJunkDebug, saveSession, sceneDisplayName, sceneReferenceMapArray, state, syncI2VVideoSettingsPanel,
  syncInspector, syncProjectVideoEngineUI, syncVideoModePanel, syncVideoTypeControl,
  transcribeExistingScenesWithOptions,
}) {
  function openAutoBuildModal() {
    const backdrop = document.createElement("div");
    // Keep nested builders (Reference Builder, LLM Runner, and their import dialogs)
    // above Auto Build when they are opened from this modal.
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100005;background:rgba(0,0,0,.74);display:flex;align-items:center;justify-content:center;padding:18px;box-sizing:border-box;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(720px,calc(100vw - 36px));max-height:calc(100vh - 36px);overflow:hidden;border:1px solid #0891b2;border-radius:14px;background:linear-gradient(180deg,#0f172a,#0b1220);color:#f8fafc;box-shadow:0 28px 90px rgba(0,0,0,.72);display:flex;flex-direction:column;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:14px;padding:16px 18px;border-bottom:1px solid #164e63;background:linear-gradient(135deg,#083344,#0f172a);";
    const headingWrap = document.createElement("div");
    headingWrap.style.cssText = "display:flex;align-items:center;gap:11px;min-width:0;";
    const headingIcon = document.createElement("div");
    headingIcon.innerHTML = `<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${COMPACT_TOOLBAR_ICONS.auto}</svg>`;
    headingIcon.style.cssText = "width:42px;height:42px;border:1px solid #22d3ee;border-radius:10px;background:#063b4a;color:#67e8f9;display:flex;align-items:center;justify-content:center;flex:0 0 auto;";
    const heading = document.createElement("div");
    heading.innerHTML = '<div style="font-size:19px;font-weight:950;color:#ecfeff;">Auto Build Music Video</div><div style="font-size:12px;color:#a5f3fc;margin-top:3px;">Add four things. Auto Build prepares everything else for you.</div>';
    const close = makeButton("Close");
    close.style.padding = "7px 10px";
    headingWrap.append(headingIcon, heading);
    header.append(headingWrap, close);

    const body = document.createElement("div");
    body.style.cssText = "padding:16px 18px;display:flex;flex-direction:column;gap:12px;overflow:auto;";
    const engineCard = document.createElement("div");
    engineCard.style.cssText = "border:1px solid #334155;border-radius:10px;background:#0f172a;padding:11px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:center;";
    const engineCopy = document.createElement("div");
    engineCopy.innerHTML = '<div style="font-size:12px;font-weight:900;color:#e0f2fe;">Video model</div><div style="font-size:11px;color:#94a3b8;margin-top:2px;">Auto Build applies the correct prompt format for the selected engine.</div>';
    const engineToggle = document.createElement("div");
    engineToggle.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:5px;";
    const ltxEngine = makeButton("LTX");
    const miniMaxEngine = makeButton("MiniMax");
    ltxEngine.style.minWidth = "76px";
    miniMaxEngine.style.minWidth = "76px";
    let selectedEngine = normalizeProjectVideoEngine(state.projectVideoEngine);
    const syncEngineButtons = () => {
      for (const [button, value] of [[ltxEngine, "ltx"], [miniMaxEngine, "minimax_h3"]]) {
        const active = selectedEngine === value;
        button.style.background = active ? "#06b6d4" : "#1e293b";
        button.style.borderColor = active ? "#67e8f9" : "#475569";
        button.style.color = active ? "#082f49" : "#e2e8f0";
        button.setAttribute("aria-pressed", active ? "true" : "false");
      }
    };
    let syncAutoBuildRunnerStatus = () => {};
    const selectAutoBuildEngine = async (engine) => {
      selectedEngine = normalizeProjectVideoEngine(engine);
      state.projectVideoEngine = selectedEngine;
      if (typeof projectVideoEngineSelect !== "undefined" && projectVideoEngineSelect) {
        projectVideoEngineSelect.value = selectedEngine;
      }
      syncProjectVideoEngineUI();
      syncVideoModePanel();
      syncI2VVideoSettingsPanel();
      syncEngineButtons();
      syncAutoBuildRunnerStatus();
      await autoSaveSessionQuiet("Auto Build video engine");
    };
    ltxEngine.onclick = () => {
      selectAutoBuildEngine("ltx").catch((error) => toast(`Could not switch the Builder to LTX:\n${String(error?.message || error)}`, true));
    };
    miniMaxEngine.onclick = () => {
      selectAutoBuildEngine("minimax_h3").catch((error) => toast(`Could not switch the Builder to MiniMax H3:\n${String(error?.message || error)}`, true));
    };
    syncEngineButtons();
    engineToggle.append(ltxEngine, miniMaxEngine);
    engineCard.append(engineCopy, engineToggle);

    const runnerCard = document.createElement("div");
    runnerCard.style.cssText = "border:1px solid #334155;border-radius:10px;background:#0f172a;padding:11px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:center;";
    const runnerCopy = document.createElement("div");
    const runnerTitle = document.createElement("div");
    runnerTitle.textContent = "LLM runner";
    runnerTitle.style.cssText = "font-size:12px;font-weight:900;color:#e0f2fe;";
    const runnerStatus = document.createElement("div");
    runnerStatus.style.cssText = "font-size:11px;color:#94a3b8;margin-top:2px;";
    const openRunner = makeButton("LLM Runner", "primary");
    openRunner.style.minWidth = "116px";
    syncAutoBuildRunnerStatus = () => {
      const runnerLabel = gemmaRunnerLabel({ vision: selectedEngine === "minimax_h3" });
      const needsKey = state.textGemmaRunner === "llm_api" && !String(state.llmApiKey || "").trim();
      const needsVisionModel = selectedEngine === "minimax_h3" && state.textGemmaRunner === "llm_api" && !llmApiVisionModelSelected();
      const needsOwnServer = state.textGemmaRunner === "own_server" && (!String(state.ownServerUrl || "").trim() || !String(state.ownServerModel || "").trim());
      runnerStatus.textContent = needsKey
        ? `${runnerLabel} selected, but no API key is set.`
        : needsVisionModel
        ? `${runnerLabel} selected, but choose a vision-capable API model.`
        : needsOwnServer
        ? `${runnerLabel} selected, but the server URL or model name is missing.`
        : `Current: ${runnerLabel}`;
      runnerStatus.style.color = needsKey || needsVisionModel || needsOwnServer ? "#fdba74" : "#94a3b8";
    };
    openRunner.onclick = () => {
      const runnerBackdrop = openGemmaRunnerModal();
      const timer = setInterval(() => {
        syncAutoBuildRunnerStatus();
        if (!runnerBackdrop?.isConnected) clearInterval(timer);
      }, 500);
    };
    runnerCopy.append(runnerTitle, runnerStatus);
    runnerCard.append(runnerCopy, openRunner);
    syncAutoBuildRunnerStatus();

    const songInput = document.createElement("input");
    songInput.type = "file";
    songInput.accept = "audio/*,.mp3,.wav,.flac,.m4a,.ogg";
    songInput.style.display = "none";
    const singerInput = document.createElement("input");
    singerInput.type = "file";
    singerInput.accept = "image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp";
    singerInput.style.display = "none";
    const locationInput = document.createElement("input");
    locationInput.type = "file";
    locationInput.multiple = true;
    locationInput.accept = "image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp";
    locationInput.style.display = "none";
    body.append(songInput, singerInput, locationInput);

    const selected = { song: null, singer: null, locations: [], locationSource: "images" };
    const uploadCard = (number, title, caption, buttonLabel, optional = false) => {
      const card = document.createElement("div");
      card.style.cssText = "border:1px solid #334155;border-radius:10px;background:#111827;padding:11px;display:grid;grid-template-columns:34px minmax(0,1fr) auto;gap:10px;align-items:center;";
      const badge = document.createElement("div");
      badge.textContent = String(number);
      badge.style.cssText = "width:30px;height:30px;border:1px solid #0891b2;border-radius:999px;background:#083344;color:#a5f3fc;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:950;";
      const copy = document.createElement("div");
      const label = document.createElement("div");
      label.innerHTML = `${escapeHtml(title)}${optional ? ' <span style="font-size:10px;color:#94a3b8;font-weight:700;">(optional)</span>' : ""}`;
      label.style.cssText = "font-size:13px;font-weight:900;color:#e2e8f0;";
      const status = document.createElement("div");
      status.textContent = caption;
      status.style.cssText = "font-size:11px;color:#94a3b8;margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      copy.append(label, status);
      const choose = makeButton(buttonLabel, "primary");
      choose.style.minWidth = "116px";
      card.append(badge, copy, choose);
      return { card, choose, status };
    };

    const currentAudioPath = String(audioInput.value || state.audioPath || "").trim();
    const songRow = uploadCard(1, "Song", currentAudioPath ? `Using current audio: ${currentAudioPath.split(/[\\/]/).pop()}` : "MP3, WAV, FLAC, M4A, or OGG", "Choose Song");
    const singerRow = uploadCard(3, "Singer image", "One clear reference image of the singer", "Choose Singer");
    const locationRow = uploadCard(4, "Location images", "One image repeats; multiple images rotate every two scenes", "Add Locations", true);
    songRow.choose.onclick = () => songInput.click();
    singerRow.choose.onclick = () => singerInput.click();
    locationRow.choose.onclick = () => {
      selected.locationSource = "images";
      locationInput.click();
    };
    songInput.onchange = () => {
      selected.song = songInput.files?.[0] || null;
      songRow.status.textContent = selected.song ? selected.song.name : (currentAudioPath ? `Using current audio: ${currentAudioPath.split(/[\\/]/).pop()}` : "Choose a song file");
    };
    singerInput.onchange = () => {
      selected.singer = singerInput.files?.[0] || null;
      singerRow.status.textContent = selected.singer ? selected.singer.name : "Choose one singer image";
    };
    locationInput.onchange = () => {
      selected.locationSource = "images";
      selected.locations = Array.from(locationInput.files || []);
      locationRow.status.textContent = selected.locations.length
        ? `${selected.locations.length} location image${selected.locations.length === 1 ? "" : "s"}: ${selected.locations.map((file) => file.name).join(", ")}`
        : "No locations selected; Auto Build will continue without them";
    };

    const lyricsCard = document.createElement("div");
    lyricsCard.style.cssText = "border:1px solid #334155;border-radius:10px;background:#111827;padding:11px;display:grid;grid-template-columns:34px minmax(0,1fr);gap:10px;align-items:start;";
    const lyricsBadge = document.createElement("div");
    lyricsBadge.textContent = "2";
    lyricsBadge.style.cssText = "width:30px;height:30px;border:1px solid #0891b2;border-radius:999px;background:#083344;color:#a5f3fc;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:950;";
    const lyricsWrap = document.createElement("div");
    const lyricsLabel = document.createElement("div");
    lyricsLabel.textContent = "Lyrics";
    lyricsLabel.style.cssText = "font-size:13px;font-weight:900;color:#e2e8f0;margin-bottom:7px;";
    const lyricsInput = document.createElement("textarea");
    lyricsInput.placeholder = "Paste the complete lyrics here...";
    lyricsInput.spellcheck = true;
    lyricsInput.style.cssText = "width:100%;box-sizing:border-box;min-height:150px;resize:vertical;border:1px solid #475569;border-radius:8px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;line-height:1.45;";
    lyricsWrap.append(lyricsLabel, lyricsInput);
    lyricsCard.append(lyricsBadge, lyricsWrap);

    const advanced = document.createElement("details");
    advanced.style.cssText = "border:1px solid #334155;border-radius:10px;background:#0f172a;padding:0 11px;";
    const advancedSummary = document.createElement("summary");
    advancedSummary.textContent = "Advanced options";
    advancedSummary.style.cssText = "cursor:pointer;padding:11px 0;font-size:12px;font-weight:900;color:#cbd5e1;";
    const advancedBody = document.createElement("div");
    advancedBody.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;padding:0 0 11px;";
    const durationLabel = document.createElement("label");
    durationLabel.textContent = "Segment duration";
    durationLabel.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:11px;color:#cbd5e1;font-weight:800;";
    const durationSelect = document.createElement("select");
    durationSelect.style.cssText = "border:1px solid #475569;border-radius:7px;background:#020617;color:#f8fafc;padding:8px;font-size:12px;";
    for (const [value, label] of [["8", "8 seconds (recommended)"], ["5", "5 seconds (lower VRAM)"], ["6", "6 seconds"], ["custom", "Custom"]]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      durationSelect.append(option);
    }
    const customDuration = document.createElement("input");
    customDuration.type = "number";
    customDuration.min = "1";
    customDuration.max = "60";
    customDuration.step = "0.5";
    customDuration.value = "8";
    customDuration.placeholder = "Seconds";
    customDuration.style.cssText = "display:none;border:1px solid #475569;border-radius:7px;background:#020617;color:#f8fafc;padding:8px;font-size:12px;";
    durationSelect.onchange = () => { customDuration.style.display = durationSelect.value === "custom" ? "block" : "none"; };
    durationLabel.append(durationSelect, customDuration);
    const advancedHint = document.createElement("div");
    advancedHint.textContent = "Shorter segments reduce per-clip VRAM use. Auto Build still clamps the final segment to the exact audio end.";
    advancedHint.style.cssText = "align-self:end;color:#94a3b8;font-size:11px;line-height:1.4;padding-bottom:2px;";
    const useCurrentBuilderSettings = makeCheckbox("Use current Scene Defaults and story settings", false);
    useCurrentBuilderSettings.wrapper.style.cssText += "grid-column:1 / -1;border:1px solid #334155;border-radius:7px;background:#111827;padding:8px;";
    useCurrentBuilderSettings.wrapper.title = "Preserve the current Scene Defaults, story layer, story beats, and per-scene performance settings instead of applying Auto Build recommendations.";
    const storyBoardRow = document.createElement("div");
    storyBoardRow.style.cssText = "grid-column:1 / -1;display:flex;align-items:center;justify-content:space-between;gap:12px;border:1px solid #164e63;border-radius:7px;background:#082f49;padding:9px 10px;";
    const storyBoardCopy = document.createElement("div");
    storyBoardCopy.innerHTML = '<div style="font-size:12px;font-weight:900;color:#e0f2fe;">Scene defaults and story</div><div style="font-size:11px;color:#bae6fd;margin-top:3px;line-height:1.35;">Open the Story Board to edit the defaults, story layer, and scene story beats before Auto Build.</div>';
    const openStoryBoard = makeButton("Go to Story Board", "primary");
    openStoryBoard.style.cssText += "white-space:nowrap;padding:8px 11px;font-size:12px;";
    openStoryBoard.title = "Open Storyboard Builder without losing the Auto Build inputs.";
    const prepareAutoBuildStoryContext = async ({ setProgress } = {}) => {
      const update = (message, percent) => setProgress?.(message, percent);
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      refs.cleared = false;
      refs.locations_cleared = false;
      update("Collecting Auto Mode singer and location references...", 20);
      if (selected.singer) {
        const singerImageName = selected.singer.name || "singer.png";
        const singerData = await readFileAsDataUrl(selected.singer);
        const singer = refs.subjects.find((item) =>
          item?.auto_build_role === "singer"
          && String(item?.image?.name || "") === singerImageName,
        ) || {
          id: `auto_build_singer_${Date.now()}`,
          name: "Singer",
          description: "",
          reference_type: "character",
          auto_build_role: "singer",
          trigger_phrase: "",
          trigger_position: "start",
          minimax_voice: normalizeMiniMaxH3Voice(),
          image: { path: "", data: singerData, name: singerImageName },
        };
        if (!refs.subjects.includes(singer)) refs.subjects.unshift(singer);
        if (!singer.image?.data) singer.image = { path: "", data: singerData, name: singerImageName };
        if (!String(singer.description || "").trim()) {
          update("Creating singer description with the LLM...", 35);
          await describeReferenceImageWithGemma(singer, "subject", { unloadAfter: false, clearBeforeLoad: false });
        }
        if (!String(singer.description || "").trim()) singer.description = "The lead singer and visible performer in every scene.";
        refs.subject = { ...(refs.subject || {}), description: singer.description, image: { ...(singer.image || {}) } };
        refs.use_subject_reference = true;
      }
      const locations = [];
      const locationFiles = selected.locationSource === "images" ? selected.locations : [];
      for (let index = 0; index < locationFiles.length; index += 1) {
        const file = locationFiles[index];
        const imageName = file.name || `location_${index + 1}.png`;
        const imageData = await readFileAsDataUrl(file);
        const location = refs.locations.find((item) =>
          item?.auto_build_role === "location"
          && String(item?.image?.name || "") === imageName,
        ) || {
          id: `auto_build_location_${Date.now()}_${index}`,
          name: imageBaseName(file, `Location ${index + 1}`),
          description: "",
          auto_build_role: "location",
          trigger_phrase: "",
          trigger_position: "start",
          image: { path: "", data: imageData, name: imageName },
        };
        if (!refs.locations.includes(location)) refs.locations.push(location);
        if (!location.image?.data) location.image = { path: "", data: imageData, name: imageName };
        if (!String(location.description || "").trim()) {
          update(`Creating location description ${index + 1}/${locationFiles.length} with the LLM...`, 45 + Math.round((index / Math.max(1, locationFiles.length)) * 35));
          await describeReferenceImageWithGemma(location, "location", { unloadAfter: index === locationFiles.length - 1, clearBeforeLoad: false });
        }
        locations.push(location);
      }
      refs.use_location_references = refs.use_location_references || locations.length > 0;
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
      state.lyricMapper = normalizeLyricMapper({ ...state.lyricMapper, source_text: String(lyricsInput.value || "").trim() });
      await autoSaveSessionQuiet("Auto Build story context preparation");
      update("Descriptions saved. Packaging story context...", 88);
      return { reference_builder: state.fluxReferenceBuilder, source_lyrics: String(lyricsInput.value || "").trim() };
    };
    openStoryBoard.onclick = () => {
      try {
        openStoryboardBuilderFromProject({
          sourceLyrics: lyricsInput.value,
          onPrepareStoryContext: prepareAutoBuildStoryContext,
        });
      } catch (error) {
        console.error("VRGDG Storyboard Builder failed to open from Auto Build", error);
        toast(`Storyboard Builder failed to open:\n${String(error?.message || error)}`, true);
      }
    };
    storyBoardRow.append(storyBoardCopy, openStoryBoard);
    const skipPromptCreation = makeCheckbox("Skip prompt creation (review timeline first)", false);
    skipPromptCreation.wrapper.style.cssText += "grid-column:1 / -1;border:1px solid #334155;border-radius:7px;background:#111827;padding:8px;";
    skipPromptCreation.wrapper.title = "Return to the timeline after Auto Build saves the audio, scenes, references, mappings, and lyric notes. You can generate prompts later from the normal builder tools.";
    advancedBody.append(durationLabel, advancedHint, storyBoardRow, useCurrentBuilderSettings.wrapper, skipPromptCreation.wrapper);
    advanced.append(advancedSummary, advancedBody);

    const note = document.createElement("div");
    note.style.cssText = "border:1px solid #155e75;border-radius:9px;background:#062b36;padding:10px 12px;color:#bae6fd;font-size:11px;line-height:1.45;";
    const selectedDurationLabel = () => durationSelect.value === "custom"
      ? `${Math.max(1, Math.min(60, Number(customDuration.value) || 8))} seconds`
      : `${durationSelect.value} seconds`;
    const syncAutoBuildNote = () => {
      note.textContent = skipPromptCreation.input.checked
        ? `Auto Build creates complete ${selectedDurationLabel()} coverage, describes the singer and locations, transcribes the existing scenes, maps everything, and saves the timeline for review. Prompt creation is skipped; generate prompts later from the normal builder tools.`
        : `Auto Build creates complete ${selectedDurationLabel()} coverage, uses the LLM to describe the singer and locations, transcribes the existing scenes, maps everything, applies the recommended defaults, and generates prompts. It returns you to the normal timeline so you can review every lyric note.`;
    };
    durationSelect.addEventListener("change", syncAutoBuildNote);
    customDuration.addEventListener("input", syncAutoBuildNote);
    skipPromptCreation.input.addEventListener("change", syncAutoBuildNote);
    syncAutoBuildNote();
    const projectWarning = document.createElement("div");
    const getAutoBuildProjectFolder = () => String(projectInput.value || state.projectFolder || "").trim();
    let projectFolder = getAutoBuildProjectFolder();
    projectWarning.style.cssText = `border:1px solid #92400e;border-radius:9px;background:#451a03;padding:10px 12px;color:#fed7aa;font-size:11px;line-height:1.45;${projectFolder ? "display:none;" : ""}`;
    const projectWarningText = document.createElement("div");
    projectWarningText.textContent = "Auto Build needs a project folder to save the song and reference images.";
    const createProjectNow = makeButton("Create Project Now", "primary");
    createProjectNow.style.cssText += "margin-top:8px;padding:7px 10px;background:#f97316;border-color:#fdba74;color:#431407;font-size:12px;font-weight:800;";
    projectWarning.append(projectWarningText, createProjectNow);
    const status = document.createElement("div");
    status.style.cssText = "display:none;border:1px solid #0891b2;border-radius:9px;background:#083344;padding:11px 12px;color:#cffafe;font-size:12px;font-weight:800;white-space:pre-wrap;line-height:1.45;";
    body.append(engineCard, runnerCard, songRow.card, lyricsCard, singerRow.card, locationRow.card, advanced, note, projectWarning, status);

    const footer = document.createElement("div");
    footer.style.cssText = "display:grid;grid-template-columns:1fr minmax(210px,auto);gap:10px;padding:13px 18px;border-top:1px solid #164e63;background:#0f172a;";
    const cancel = makeButton("Cancel");
    const build = makeButton("Build My Music Video", "primary");
    build.style.cssText += "font-size:13px;padding:11px 18px;background:linear-gradient(180deg,#22d3ee,#0891b2);border-color:#67e8f9;";
    const refreshProjectReady = () => {
      projectFolder = getAutoBuildProjectFolder();
      const ready = Boolean(projectFolder);
      projectWarning.style.display = ready ? "none" : "block";
      build.textContent = ready ? "Build My Music Video" : "Create Project & Build";
      build.title = ready
        ? "Build the music-video timeline from the inputs above."
        : "Create a project first, then continue Auto Build without losing the inputs above.";
      return ready;
    };
    refreshProjectReady();
    footer.append(cancel, build);
    box.append(header, body, footer);
    backdrop.append(box);
    document.body.append(backdrop);

    const closeModal = () => backdrop.remove();
    close.onclick = closeModal;
    cancel.onclick = closeModal;
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) closeModal();
    });
    const ensureAutoBuildProject = async () => {
      if (refreshProjectReady()) return true;
      createProjectNow.disabled = true;
      build.disabled = true;
      try {
        const created = await newProject();
        refreshProjectReady();
        return Boolean(created && getAutoBuildProjectFolder());
      } finally {
        createProjectNow.disabled = false;
        build.disabled = false;
      }
    };
    createProjectNow.onclick = async () => {
      await ensureAutoBuildProject();
    };

    const imageBaseName = (file, fallback) => String(file?.name || fallback || "Reference")
      .replace(/\.[^.]+$/, "")
      .replace(/[_-]+/g, " ")
      .trim() || fallback || "Reference";
    const hasRealTimeline = () => {
      const scenes = allEditableSegments();
      if (scenes.length > 1) return true;
      const scene = scenes[0];
      if (!scene) return false;
      return Boolean(
        String(scene.lyric_text || scene.notes || scene.t2i_prompt || scene.i2v_prompt || scene.minimax_h3_prompt || "").trim()
        || scene.image
        || scene.video_path
        || !/^New scene$/i.test(String(scene.label || "New scene"))
      );
    };

    const generateAutoBuildMiniMaxPrompts = async (options = {}) => {
      const failedIds = new Set((options.failedIds || []).map((value) => String(value)));
      const allScenes = [...state.segments];
      const scenes = options.failedOnly
        ? allScenes.filter((segment) => failedIds.has(String(segment?.id || "")))
        : options.missingOnly
          ? allScenes.filter((segment) => !String(segment?.minimax_h3_prompt || "").trim())
        : allScenes;
      const progress = createProgressWindow("Auto Build MiniMax Prompts");
      let created = 0;
      const failures = [];
      let historySaved = false;
      try {
        state.batchCancelled = false;
        for (let index = 0; index < scenes.length; index += 1) {
          assertBatchNotStopped();
          const segment = scenes[index];
          const mode = "reference_to_video";
          const visionImages = miniMaxH3PromptVisionImagesForRunner(segment, mode);
          const rendererReferences = miniMaxOrderedImageReferenceItemsForSegment(segment, mode);
          if (!rendererReferences.length) throw new Error(`${sceneDisplayName(segment, index)} has no mapped Reference Builder image.`);
          assertMiniMaxH3ReferenceDescriptionsReady(segment, mode);
          if (visionImages.length && state.textGemmaRunner === "llm_api" && !llmApiVisionModelSelected()) {
            throw new Error("MiniMax reference prompting needs a vision-capable API model selected in LLM Runner.");
          }
          const percent = 5 + Math.round((index / Math.max(1, scenes.length)) * 90);
          progress.set(`Creating MiniMax prompt ${index + 1}/${scenes.length}: ${sceneDisplayName(segment, index)}\n${gemmaRunnerLine({ vision: Boolean(visionImages.length) })}`, percent);
          try {
            const data = await runMiniMaxH3PromptGeneration(segment, mode, {
              projectFolder,
              sceneId: segment.id || "",
              unloadAfter: index === scenes.length - 1,
              promptOnlySceneInspiration: false,
              performanceMode: "singing",
              audioMode: "input_audio",
              speakerAssignments: [],
              emptyPromptMessage: `${sceneDisplayName(segment, index)} returned an empty MiniMax prompt.`,
            });
            gemmaBatchFailureStore()[`minimax:${mode}:${segment.id}`] = undefined;
            if (!historySaved) {
              pushHistory();
              historySaved = true;
            }
            segment.minimax_h3_prompt = data.prompt;
            segment.minimax_h3_prompt_origin = "gemma";
            created += 1;
            await autoSaveSessionQuiet(`Auto Build MiniMax prompt ${index + 1}`);
          } catch (error) {
            if (!isRecoverableBuildGemmaError(error)) throw error;
            const debugPath = error?.gemmaDebugPath || await saveGemmaJunkDebug(error, { label: `Auto Build MiniMax prompt ${index + 1}`, segment });
            failures.push(recordGemmaBatchFailure(`minimax:${mode}:${segment.id}`, segment, sceneDisplayName(segment, index), error, debugPath));
            progress.set(`${sceneDisplayName(segment, index)} skipped. Continuing with the remaining MiniMax scenes...`, percent);
          }
        }
        progress.set(`Created ${created} MiniMax video prompt${created === 1 ? "" : "s"}.${failures.length ? ` Skipped ${failures.length} scene${failures.length === 1 ? "" : "s"}.` : ""}`, 100);
        progress.close(1600);
        if (failures.length) showGemmaBatchFailures(failures, {
          retryHandler: (items) => generateAutoBuildMiniMaxPrompts({ failedOnly: true, failedIds: items.map((item) => item.segmentId) }),
        });
        return created;
      } catch (error) {
        progress.set(`Auto Build MiniMax prompts stopped after ${created}/${scenes.length}:\n${String(error?.message || error)}`, 100);
        progress.close(5000);
        throw error;
      }
    };

    const validateAutoBuildPromptRunner = () => {
      if (state.textGemmaRunner === "llm_api") {
        if (!String(state.llmApiKey || "").trim()) {
          throw new Error("LLM API is selected, but no API key is set. Open LLM Runner in Auto Build, paste/test the key, then start Auto Build again.");
        }
        if (selectedEngine === "minimax_h3" && !llmApiVisionModelSelected()) {
          throw new Error("MiniMax Auto Build uses reference images for prompt writing. Open LLM Runner and choose a vision-capable API model.");
        }
      }
      if (state.textGemmaRunner === "lm_studio" && !String(state.lmStudioModel || "").trim()) {
        throw new Error("LM Studio is selected, but no model name is set. Open LLM Runner, load/select a model, then start Auto Build again.");
      }
      if (state.textGemmaRunner === "own_server") {
        if (!String(state.ownServerUrl || "").trim()) {
          throw new Error("Custom Server is selected, but no URL is set. Open LLM Runner, paste the server URL, then start Auto Build again.");
        }
        if (!String(state.ownServerModel || "").trim()) {
          throw new Error("Custom Server is selected, but no model name is set. Open LLM Runner, enter or load the served model, then start Auto Build again.");
        }
      }
    };

    const applyAutoBuildSingerPerformerMapping = (singer) => {
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      if (!refs.subject_scene_map || typeof refs.subject_scene_map !== "object") refs.subject_scene_map = {};
      state.segments.forEach((segment, index) => {
        refs.subject_scene_map[segment.id] = Array.from(new Set([singer.id, ...sceneReferenceMapArray(refs.subject_scene_map, segment, index)].filter(Boolean)));
        segment.lyric_singers = [singer.name];
        segment.lyric_no_lip_sync = false;
        segment.no_character_present = false;
        const cueText = String(segment.lyric_text || "").trim();
        segment.minimax_speaker_assignments = cueText && !isInstrumentalLyricText(cueText)
          ? normalizeMiniMaxSpeakerAssignments([{
            speaker_id: singer.id,
            speaker_name: singer.name,
            text: cueText,
          }])
          : [];
      });
      refs.use_subject_reference = true;
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
    };

    build.onclick = async () => {
      const lyrics = String(lyricsInput.value || "").trim();
      if (!lyrics) {
        toast("Paste the complete lyrics before starting Auto Build.", true);
        lyricsInput.focus();
        return;
      }
      // Preserve exactly what the user pasted as the project's canonical song
      // text before timestamping adds instrumental gaps to scene-level notes.
      state.lyricMapper = normalizeLyricMapper({
        ...state.lyricMapper,
        source_text: lyrics,
      });
      if (!selected.singer) {
        toast("Choose one singer reference image before starting Auto Build.", true);
        return;
      }
      const requestedSegmentDuration = durationSelect.value === "custom"
        ? Number(customDuration.value)
        : Number(durationSelect.value);
      const segmentDuration = Math.max(1, Math.min(60, Number.isFinite(requestedSegmentDuration) ? requestedSegmentDuration : 8));
      const autoBuildInputFingerprint = autoBuildFingerprint([
        lyrics,
        segmentDuration,
        selectedEngine,
        selected.singer ? `${selected.singer.name || ""}:${selected.singer.size || 0}:${selected.singer.lastModified || 0}` : "",
        ...(selected.locationSource === "images" ? selected.locations.map((file) => `${file?.name || ""}:${file?.size || 0}:${file?.lastModified || 0}`) : []),
      ]);
      const resumeExistingAutoBuild = Boolean(
        state.segments.length
        && state.autoBuildPreparation?.input_fingerprint === autoBuildInputFingerprint,
      );
      try {
        validateAutoBuildPromptRunner();
      } catch (error) {
        syncAutoBuildRunnerStatus();
        toast(String(error?.message || error), true);
        openGemmaRunnerModal();
        return;
      }
      if (hasRealTimeline() && !window.confirm("Auto Build will replace the current base timeline and its insert scenes. Continue?")) return;
      if (!await ensureAutoBuildProject()) {
        toast("Create a project before starting Auto Build.", true);
        return;
      }
      const audioAvailable = Boolean(selected.song || String(audioInput.value || state.audioPath || "").trim());
      if (!audioAvailable) {
        toast("Choose a song before starting Auto Build.", true);
        return;
      }

      build.disabled = true;
      cancel.disabled = true;
      close.disabled = true;
      createProjectNow.disabled = true;
      for (const control of [ltxEngine, miniMaxEngine, openRunner, songRow.choose, singerRow.choose, locationRow.choose, lyricsInput, durationSelect, customDuration, useCurrentBuilderSettings.input, skipPromptCreation.input]) control.disabled = true;
      status.style.display = "block";
      const setStatus = (message) => {
        status.textContent = message;
        status.scrollIntoView({ block: "nearest", behavior: "smooth" });
      };

      try {
        setStatus("1/7  Loading the song...");
        if (selected.song) {
          const loaded = await chooseProjectAudioFile(selected.song);
          if (!loaded) throw new Error("The selected song could not be loaded into the project.");
        }
        const songDuration = Number(state.audioDuration || audio.duration || state.duration || 0);
        if (!Number.isFinite(songDuration) || songDuration <= 0) throw new Error("The song duration could not be read.");

        state.projectVideoEngine = normalizeProjectVideoEngine(selectedEngine);
        state.videoType = "singing";
        state.videoModelMode = "rtv";
        state.miniMaxH3Settings = cloneMiniMaxH3Settings({
          ...state.miniMaxH3Settings,
          video_mode: "reference_to_video",
          audio_mode: "input_audio",
        });
        syncVideoTypeControl();
        syncProjectVideoEngineUI();
        syncVideoModePanel();
        syncI2VVideoSettingsPanel();

        setStatus(resumeExistingAutoBuild
          ? "2/7  Reusing the existing Auto Build timeline..."
          : `2/7  Creating complete ${segmentDuration}-second timeline coverage...`);
        if (!resumeExistingAutoBuild) {
          const sceneCount = Math.max(1, Math.ceil(songDuration / segmentDuration));
          const timings = Array.from({ length: sceneCount }, (_, index) => ({
            start: index * segmentDuration,
            end: Math.min(songDuration, (index + 1) * segmentDuration),
          }));
          await applyBulkSegmentTimings(timings, "replace");
        }
        state.overlaySegments = [];
        state.activeTrack = "base";
        state.duration = songDuration;

        setStatus("3/7  Adding and mapping singer and location references...");
        const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
        refs.cleared = false;
        refs.locations_cleared = false;
        const importedLocations = refs.locations.slice();
        const singerData = await readFileAsDataUrl(selected.singer);
        const singerImageName = selected.singer.name || "singer.png";
        const singer = refs.subjects.find((item) =>
          item?.auto_build_role === "singer"
          && String(item?.image?.name || "") === singerImageName,
        ) || {
          id: `auto_build_singer_${Date.now()}`,
          name: "Singer",
          description: "",
          reference_type: "character",
          auto_build_role: "singer",
          trigger_phrase: "",
          trigger_position: "start",
          extra_reference_for: "",
          extra_reference_note: "",
          minimax_voice: normalizeMiniMaxH3Voice(),
          image: { path: "", data: singerData, name: singerImageName },
        };
        if (!refs.subjects.includes(singer)) refs.subjects.unshift(singer);
        if (!singer.image?.data) singer.image = { path: "", data: singerData, name: singerImageName };
        refs.subject_count = refs.subjects.length;
        refs.subject = {
          description: singer.description,
          reference_type: "character",
          minimax_voice: normalizeMiniMaxH3Voice(),
          image: { ...singer.image },
        };
        refs.use_subject_reference = true;
        const autoLocations = [];
        const locationData = await Promise.all(selected.locationSource === "images" ? selected.locations.map((file) => readFileAsDataUrl(file)) : []);
        (selected.locationSource === "images" ? selected.locations : []).forEach((file, index) => {
          const imageName = file.name || `location_${index + 1}.png`;
          const location = refs.locations.find((item) =>
            item?.auto_build_role === "location"
            && String(item?.image?.name || "") === imageName,
          ) || {
            id: `auto_build_location_${Date.now()}_${index}`,
            name: imageBaseName(file, `Location ${index + 1}`),
            description: "",
            auto_build_role: "location",
            trigger_phrase: "",
            trigger_position: "start",
            image: { path: "", data: locationData[index], name: imageName },
          };
          if (!refs.locations.includes(location)) refs.locations.push(location);
          if (!location.image?.data) location.image = { path: "", data: locationData[index], name: imageName };
          autoLocations.push(location);
        });

        setStatus("3/7  Describing singer and location references with the LLM...");
        if (!String(singer.description || "").trim()) {
          await describeReferenceImageWithGemma(singer, "subject", {
            unloadAfter: autoLocations.length === 0,
            clearBeforeLoad: false,
          });
        }
        if (!String(singer.description || "").trim()) {
          singer.description = "The lead singer and visible performer in every scene.";
        }
        refs.subject.description = singer.description;
        for (let index = 0; index < autoLocations.length; index += 1) {
          const location = autoLocations[index];
          setStatus(`3/7  Describing location ${index + 1}/${autoLocations.length} with the LLM...`);
          if (!String(location.description || "").trim()) {
            await describeReferenceImageWithGemma(location, "location", {
              unloadAfter: index === autoLocations.length - 1,
              clearBeforeLoad: false,
            });
          }
        }
        refs.use_location_references = importedLocations.length > 0 || autoLocations.length > 0;
        refs.subject_scene_map = refs.subject_scene_map || {};
        refs.scene_map = refs.scene_map || {};
        state.segments.forEach((segment, index) => {
          if (!sceneReferenceMapArray(refs.subject_scene_map, segment, index).length) refs.subject_scene_map[segment.id] = [singer.id];
          const availableLocations = autoLocations.length ? autoLocations : importedLocations;
          if (!refs.scene_map[segment.id] && availableLocations.length) {
            refs.scene_map[segment.id] = availableLocations[Math.floor(index / 2) % availableLocations.length].id;
          }
          segment.lyric_singers = [singer.name];
          segment.lyric_no_lip_sync = false;
          segment.no_character_present = false;
          segment.minimax_speaker_assignments = [];
          if (!resumeExistingAutoBuild) segment.story_beat = "";
          segment.minimax_h3_mode = "reference_to_video";
          segment.minimax_h3_scene_image_use = "off";
        });
        state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);

        if (useCurrentBuilderSettings.input.checked) {
          setStatus("4/7  Preserving current Scene Defaults and story settings...");
        } else {
          setStatus("4/7  Applying recommended scene defaults...");
          state.builderStoryLayer = normalizeBuilderStoryLayer({
            enabled: true,
            overall_story_idea: "",
            user_story_arc: "",
            song_story_brief: "",
            lyric_story_strength: 3,
            image_world_style: "natural",
            image_custom_style_direction: "",
          });
          state.builderStoryboardDefaults = normalizeBuilderStoryboardDefaults({
            ...state.builderStoryboardDefaults,
            video_style: "Cinematic realism",
            video_style_custom: "",
            temporal_world_effect: "",
            temporal_world_effect_custom: "",
            global_consistency_phrase: "",
            camera_flow: "intimate_closeups",
            camera_motion_speed: 6,
            minimax_h3_cut_frequency: 3,
            performance_style: "",
            character_motion_speed: 4,
            camera_guidance: builderMotionSpeedGuidance(6, "camera"),
            character_guidance: builderMotionSpeedGuidance(4, "character"),
          });
          state.defaultFacialPerformance = "";
          state.defaultFacialPerformanceCustom = "";
          let previousMotion = "";
          state.segments.forEach((segment, index) => {
            const camera = storyboardCameraFlowEntry("intimate_closeups", index, previousMotion);
            if (camera?.shot) segment.shot_type = camera.shot;
            if (camera?.camera) segment.camera_motion = camera.camera;
            previousMotion = String(segment.camera_motion || previousMotion);
            segment.performance_style = "";
            segment.facial_performance = "";
            segment.facial_performance_custom = "";
            segment.character_motion = builderMotionSpeedGuidance(4, "character");
          });
        }

        const needsLyricTranscription = state.segments.some((segment) => !String(segment.lyric_text || "").trim());
        setStatus(needsLyricTranscription
          ? "5/7  Filling missing lyric notes..."
          : "5/7  Reusing existing lyric notes...");
        if (needsLyricTranscription) {
          await transcribeExistingScenesWithOptions({
            referenceLyrics: lyrics,
            language: "english",
            replaceAll: false,
            instrumentalText: "[instrumental]",
          }, {
            recordHistory: false,
          });
        }
        applyAutoBuildSingerPerformerMapping(singer);

        setStatus("6/7  Saving references and timeline mappings...");
        state.autoBuildPreparation = normalizeAutoBuildPreparation({
          input_fingerprint: autoBuildInputFingerprint,
          lyrics_fingerprint: autoBuildFingerprint([lyrics]),
          segment_duration: segmentDuration,
          engine: selectedEngine,
          prepared_at: new Date().toISOString(),
        });
        await saveSession({ quiet: true, throwOnError: true });
        syncInspector();
        render();

        if (skipPromptCreation.input.checked) {
          closeModal();
          state.activeId = state.segments[0]?.id || state.activeId;
          state.activeTrack = "base";
          syncInspector();
          render();
          toast(`Auto Build complete.\nCreated ${state.segments.length} scenes with full song coverage.\nPrompt creation was skipped. Review the timeline and lyric notes, then create prompts when ready.`);
          return;
        }
        setStatus("7/7  Preparing video prompts. The normal prompt progress window will open next...");
        closeModal();
        if (selectedEngine === "minimax_h3") {
          await generateAutoBuildMiniMaxPrompts({ missingOnly: true });
        } else {
          await gemmaVideoAllTextOnly({
            promptRunMode: "missing_only",
            gemmaInputMode: "text",
            sceneScope: "all",
          });
        }
        state.activeId = state.segments[0]?.id || state.activeId;
        state.activeTrack = "base";
        syncInspector();
        render();
        const promptCount = selectedEngine === "minimax_h3"
          ? state.segments.filter((segment) => String(segment.minimax_h3_prompt || "").trim()).length
          : state.segments.filter((segment) => String(segment.i2v_prompt || "").trim()).length;
        toast(`Auto Build complete.\nCreated ${state.segments.length} scenes with full song coverage.\nGenerated ${promptCount} video prompt${promptCount === 1 ? "" : "s"}.\nReview the timeline and lyric notes before rendering.`);
      } catch (error) {
        setStatus(`Auto Build paused:\n${String(error?.message || error)}\n\nAny completed timeline work has been preserved for review.`);
        build.disabled = false;
        cancel.disabled = false;
        close.disabled = false;
        createProjectNow.disabled = false;
        for (const control of [ltxEngine, miniMaxEngine, openRunner, songRow.choose, singerRow.choose, locationRow.choose, lyricsInput, durationSelect, customDuration, useCurrentBuilderSettings.input, skipPromptCreation.input]) control.disabled = false;
        toast(`Auto Build paused:\n${String(error?.message || error)}`, true);
      }
    };
  }

  return { openAutoBuildModal };
}
