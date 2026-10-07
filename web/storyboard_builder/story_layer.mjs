import { postJson } from "./api.mjs";
import { copyTextToClipboard, createStoryboardProgressWindow, createToast, makeButton } from "./controls.mjs";
import { STORY_LAYER_CHATGPT_URL, storyLayerGptPayload } from "./gpt_payload.mjs";
import { imagePromptImportJsonText } from "./prompt_generation.mjs";
import { normalizeReferenceBuilderCatalog } from "./references.mjs";
import {
  normalizeStoryArcDetail,
  normalizeScene,
  normalizeStoryLayer,
  slimSceneForRequest,
  slimStoryboardForRequest,
  storyboardSpeedValue,
} from "./scenes.mjs";
import { normalizeStoryboardScriptImportState } from "./script_import.mjs";
import { hasMappedStoryboardLocation, NO_MAPPED_LOCATIONS_MESSAGE } from "./story_workflow.mjs";
import {
  storyboardMiniMaxVideoStylePreset,
  storyboardMiniMaxVideoStyleVerbiage,
  storyboardSceneSupportsVideoStyle,
  storyboardTemporalIntensity,
  storyboardTemporalWorldEffectForScene,
} from "./video_style.mjs";

export function showStoryLayerGptHandoff(payloadJson, chatWindow = null) {
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100014;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(900px,calc(100vw - 48px));max-height:calc(100vh - 48px);border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 22px 80px rgba(0,0,0,.62);display:flex;flex-direction:column;overflow:hidden;";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;background:#083f4f;border-bottom:1px solid #155e75;padding:13px 15px;";
  const title = document.createElement("div");
  title.innerHTML = "<div style=\"font-size:17px;font-weight:900;color:#cffafe;\">GPT Story JSON</div><div style=\"font-size:12px;color:#cbd5e1;margin-top:3px;\">Attach or paste the JSON, then send an explicit request to process it.</div>";
  const close = makeButton("Close");
  header.append(title, close);
  const body = document.createElement("div");
  body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:12px;overflow:auto;";
  const status = document.createElement("div");
  status.style.cssText = "font-size:12px;color:#94a3b8;min-height:18px;";
  const text = document.createElement("textarea");
  text.value = payloadJson;
  text.spellcheck = false;
  text.style.cssText = "min-height:360px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;white-space:pre;overflow:auto;";
  const actions = document.createElement("div");
  actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;";
  const copy = makeButton("Copy JSON", "primary");
  const copyRequest = makeButton("Copy Request");
  const openChat = makeButton("Open ChatGPT", "primary");
  actions.append(copy, copyRequest, openChat, close);
  body.append(status, text, actions);
  box.append(header, body);
  backdrop.append(box);
  document.body.append(backdrop);
  const closeModal = () => backdrop.remove();
  const copyJson = async () => {
    try {
      await copyTextToClipboard(text.value);
      status.textContent = "Copied JSON to clipboard. Paste it into ChatGPT.";
      status.style.color = "#67e8f9";
    } catch (error) {
      status.textContent = "Clipboard copy was blocked. Select the JSON above and copy it manually.";
      status.style.color = "#fbbf24";
    }
  };
  const copyRequestText = async () => {
    try {
      await copyTextToClipboard("Process the attached story-layer planning JSON now. Do not ask what I want done. Use its task_instruction and project_inputs, then return only the final JSON with overall_story_idea, user_story_arc, and song_story_brief.");
      status.textContent = "Copied the request text. Paste it into ChatGPT after attaching the JSON.";
      status.style.color = "#67e8f9";
    } catch (error) {
      status.textContent = "Clipboard copy was blocked. Manually type: Process the attached story-layer planning JSON now and return the final JSON.";
      status.style.color = "#fbbf24";
    }
  };
  close.onclick = closeModal;
  copy.onclick = copyJson;
  copyRequest.onclick = copyRequestText;
  openChat.onclick = () => {
    if (chatWindow && !chatWindow.closed) chatWindow.focus();
    else window.open(STORY_LAYER_CHATGPT_URL, "_blank", "noopener,noreferrer");
  };
  backdrop.addEventListener("pointerdown", (event) => {
    if (event.target === backdrop) closeModal();
  });
  text.focus();
  text.select();
}

export function parseStoryLayerImportJson(rawText) {
  const data = JSON.parse(imagePromptImportJsonText(rawText));
  const source = data?.story_layer && typeof data.story_layer === "object"
    ? { ...data, ...data.story_layer }
    : data;
  if (!source || typeof source !== "object" || Array.isArray(source)) {
    throw new Error("Story JSON must be an object.");
  }
  const stringifyStoryValue = (raw) => {
    if (raw === undefined || raw === null) return "";
    if (typeof raw === "string" || typeof raw === "number" || typeof raw === "boolean") return String(raw).trim();
    if (Array.isArray(raw)) return raw.map(stringifyStoryValue).filter(Boolean).join("\n");
    if (typeof raw === "object") {
      return Object.entries(raw)
        .map(([key, value]) => {
          const text = stringifyStoryValue(value);
          return text ? `${key}:\n${text}` : "";
        })
        .filter(Boolean)
        .join("\n\n");
    }
    return "";
  };
  const value = (...keys) => {
    for (const key of keys) {
      if (source[key] !== undefined && source[key] !== null) return stringifyStoryValue(source[key]);
    }
    return "";
  };
  const result = {
    overall_story_idea: value("overall_story_idea", "overallStoryIdea", "story_idea", "storyIdea"),
    user_story_arc: value("user_story_arc", "userStoryArc", "story_arc", "storyArc"),
    song_story_brief: value("song_story_brief", "songStoryBrief", "story_brief", "storyBrief", "brief"),
  };
  if (!result.overall_story_idea && !result.user_story_arc && !result.song_story_brief) {
    throw new Error("No overall_story_idea, user_story_arc/story_arc, or song_story_brief/story_brief was found.");
  }
  return result;
}

export function createStoryLayer({
  imageCustomStyleInput, imageWorldStyleSelect, lyricStoryStrengthInput, overallStoryIdeaInput,
  promptRunnerName, refreshSetupPanelSummaries, renderTable, songStoryBriefInput, state,
  storyLayerEnabledInput, storyboardDefaultsPayload, temporalEffectCustomControls, temporalEffectInfo,
  temporalEffectOptions, temporalIntensityInput, temporalIntensityValue, temporalProtectedCustomControls,
  userStoryArcInput, videoStyleCustomControls, videoStyleInfo,
}) {
  function syncStoryLayerFromInputs({ notify = false } = {}) {
    state.storyLayer = normalizeStoryLayer({
      enabled: storyLayerEnabledInput.checked,
      overall_story_idea: overallStoryIdeaInput.value,
      user_story_arc: userStoryArcInput.value,
      song_story_brief: songStoryBriefInput.value,
      lyric_story_strength: lyricStoryStrengthInput.value,
      image_world_style: imageWorldStyleSelect.value,
      image_custom_style_direction: imageCustomStyleInput.value,
    });
    if (notify && state.onStoryLayerChanged) {
      state.onStoryLayerChanged({
        ...storyboardDefaultsPayload(),
        story_layer: normalizeStoryLayer(state.storyLayer),
        script_import: normalizeStoryboardScriptImportState(state.scriptImport),
        facial_performance_default: state.facialPerformance || "",
        facial_performance_custom_default: state.facialPerformanceCustom || "",
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
      });
    }
    refreshSetupPanelSummaries();
  }
  function notifyStoryboardDefaultsChanged() {
    if (!state.onStoryLayerChanged) return;
    state.onStoryLayerChanged({
      ...storyboardDefaultsPayload(),
      story_layer: normalizeStoryLayer(state.storyLayer),
      script_import: normalizeStoryboardScriptImportState(state.scriptImport),
      facial_performance_default: state.facialPerformance || "",
      facial_performance_custom_default: state.facialPerformanceCustom || "",
      scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
    });
  }

  function lyricsForStoryBrief() {
    const blocks = [];
    state.scenes.forEach((scene, index) => {
      const normalized = normalizeScene(scene, index);
      const section = String(normalized.lyric_section || "").trim();
      const lyric = String(normalized.lyrics || "").trim();
      if (!section && !lyric) return;
      const previous = blocks[blocks.length - 1];
      if (section && previous?.section?.toLowerCase() === section.toLowerCase()) {
        if (lyric) previous.lyrics.push(lyric);
        return;
      }
      blocks.push({ section, lyrics: lyric ? [lyric] : [] });
    });
    return blocks
      .map(({ section, lyrics }) => `${section ? `[${section}]\n` : ""}${lyrics.join("\n")}`.trim())
      .filter(Boolean)
      .join("\n\n");
  }

  function refreshVideoStyleInfo() {
    const preset = storyboardMiniMaxVideoStylePreset(state.videoStyle);
    const exactVerbiage = storyboardMiniMaxVideoStyleVerbiage(state.videoStyle, state.videoStyleCustom);
    videoStyleCustomControls.style.display = state.mode === "image_to_video_prep"
      && (state.projectVideoEngine === "ltx"
        || (state.projectVideoEngine === "minimax_h3" && ["text_to_video", "reference_to_video"].includes(state.miniMaxH3Mode))
        || state.scenes.some((scene) => storyboardSceneSupportsVideoStyle(scene)))
      && state.videoStyle === "custom"
      ? "flex"
      : "none";
    videoStyleInfo.textContent = exactVerbiage
      ? `Required exact wording in every eligible prompt: ${exactVerbiage}`
      : "Optional. Choose the governing visual aesthetic for eligible video scenes.";
    refreshSetupPanelSummaries();
  }

  function refreshTemporalEffectInfo() {
    state.temporalBackgroundIntensity = storyboardTemporalIntensity(temporalIntensityInput.value);
    temporalIntensityValue.textContent = `${state.temporalBackgroundIntensity}/10`;
    temporalEffectCustomControls.style.display = state.mode === "image_to_video_prep"
      && state.temporalWorldEffect === "custom" ? "flex" : "none";
    temporalEffectOptions.style.display = state.mode === "image_to_video_prep"
      && Boolean(state.temporalWorldEffect) ? "flex" : "none";
    temporalProtectedCustomControls.style.display = state.mode === "image_to_video_prep"
      && Boolean(state.temporalWorldEffect)
      && state.temporalProtectedCharacters === "custom" ? "flex" : "none";
    const effect = storyboardTemporalWorldEffectForScene({}, state);
    temporalEffectInfo.textContent = effect
      ? `Required in every video prompt unless a scene overrides it.\n${effect.exact_verbiage}`
      : "Optional. Choose a global temporal/world effect. Existing projects and scenes remain natural-time while this is Off.";
    refreshSetupPanelSummaries();
  }

  function sectionMapFromLyrics() {
    const map = new Map();
    let current = "";
    state.scenes.forEach((scene, index) => {
      const lyric = String(scene.lyrics || "").trim();
      const explicit = String(scene.lyric_section || "").trim();
      const header = lyric.match(/^\s*\[([^\]]{2,80})\]\s*$/);
      if (explicit) current = explicit;
      else if (header) current = header[1].trim();
      else if (current) map.set(scene.id || `scene_${index + 1}`, current);
    });
    return map;
  }

  function detectLyricSections() {
    const map = sectionMapFromLyrics();
    let changed = 0;
    state.scenes.forEach((scene, index) => {
      const key = scene.id || `scene_${index + 1}`;
      const section = map.get(key);
      const lyric = String(scene.lyrics || "").trim();
      const header = lyric.match(/^\s*\[([^\]]{2,80})\]\s*$/);
      if (header && !String(scene.lyric_section || "").trim()) {
        scene.lyric_section = header[1].trim();
        changed += 1;
      } else if (section && !String(scene.lyric_section || "").trim()) {
        scene.lyric_section = section;
        changed += 1;
      }
    });
    renderTable();
    syncStoryLayerFromInputs();
    createToast(changed ? `Detected lyric sections for ${changed} scene${changed === 1 ? "" : "s"}.` : "No missing lyric sections were detected.");
  }

  function confirmStoryStepRerun(stepName, existingText) {
    const rendered = state.renderedSceneCount;
    if (rendered > 0) {
      return window.confirm(`This project already has ${rendered} rendered scene${rendered === 1 ? "" : "s"}. A new ${stepName} can change scene beats and prompts so they no longer match the rendered scenes. Continue?`);
    }
    return !String(existingText || "").trim() || window.confirm(`This replaces the current ${stepName}. Continue?`);
  }

  async function createStoryBriefWithGemma() {
    syncStoryLayerFromInputs();
    if (!hasMappedStoryboardLocation(state)) {
      createToast(NO_MAPPED_LOCATIONS_MESSAGE, true);
      return null;
    }
    if (!confirmStoryStepRerun("story brief", state.storyLayer.song_story_brief)) return null;
    const authoritativeScript = normalizeStoryboardScriptImportState(state.scriptImport);
    const progress = createStoryboardProgressWindow(`Story Brief — ${promptRunnerName()}`);
    try {
      progress.set(authoritativeScript.enabled
        ? "Creating a short-film production brief around the exact imported script..."
        : "Creating compact song story brief from lyrics, sections, and your story arc...", 18);
      const data = await postJson("/vrgdg/storyboard/story_brief", {
        ...(state.gemmaSettings || {}),
        story_layer: normalizeStoryLayer(state.storyLayer),
        script_import: normalizeStoryboardScriptImportState(state.scriptImport),
        performance_mode: state.performanceMode,
        reference_builder: state.referenceBuilder || {},
        storyboard: slimStoryboardForRequest(state),
        lyrics: lyricsForStoryBrief(),
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
        unload_after: true,
        max_new_tokens: authoritativeScript.enabled ? 1200 : 800,
      }, 240000);
      if (!String(data.story_brief || "").trim()) throw new Error("The LLM returned an empty story brief.");
      state.storyLayer.song_story_brief = String(data.story_brief || "").trim();
      songStoryBriefInput.value = state.storyLayer.song_story_brief;
      syncStoryLayerFromInputs({ notify: true });
      progress.set("Story brief saved into the Story Layer.", 100);
      progress.close(1600);
      createToast("Story brief created.");
      return state.storyLayer.song_story_brief;
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      createToast(`Story brief failed:\n${String(error?.message || error)}`, true);
      return null;
    }
  }

  async function createStoryArcWithGemma() {
    syncStoryLayerFromInputs();
    if (!hasMappedStoryboardLocation(state)) {
      createToast(NO_MAPPED_LOCATIONS_MESSAGE, true);
      return null;
    }
    if (!confirmStoryStepRerun("story arc", state.storyLayer.user_story_arc)) return null;
    const authoritativeScript = normalizeStoryboardScriptImportState(state.scriptImport);
    const progress = createStoryboardProgressWindow(`${authoritativeScript.enabled ? "Short Film Premise" : "Story Arc"} — ${promptRunnerName()}`);
    const storyArcSeed = Math.floor(Math.random() * 2147483647);
    const existingStoryArcText = String(userStoryArcInput.value || "").trim();
    const overallStoryIdea = String(overallStoryIdeaInput.value || "").trim();
    const storyLayerForRequest = normalizeStoryLayer({
      ...state.storyLayer,
      overall_story_idea: overallStoryIdea,
      user_story_arc: "",
    });
    try {
      progress.set(authoritativeScript.enabled
        ? `Creating a visual short-film premise around the exact imported dialogue...\nReroll seed: ${storyArcSeed}`
        : `Creating a short song-structure story arc from lyrics, subjects, and locations...\nReroll seed: ${storyArcSeed}`, 18);
      const data = await postJson("/vrgdg/storyboard/story_arc", {
        ...(state.gemmaSettings || {}),
        n_ctx: Math.max(16384, Number(state.gemmaSettings?.n_ctx) || 0),
        seed: storyArcSeed,
        story_arc_seed: storyArcSeed,
        story_layer: storyLayerForRequest,
        script_import: normalizeStoryboardScriptImportState(state.scriptImport),
        performance_mode: state.performanceMode,
        storyboard: slimStoryboardForRequest(state),
        story_idea: overallStoryIdea,
        previous_story_arc: existingStoryArcText,
        lyrics: lyricsForStoryBrief(),
        project_folder: state.projectFolder,
        line_mapping_lyrics: state.lineMappingLyrics,
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
        reference_builder: state.referenceBuilder || {},
        camera_flow: state.cameraFlow || "",
        camera_motion_speed: storyboardSpeedValue(state.cameraMotionSpeed, 4),
        character_motion: storyboardSpeedValue(state.characterMotionSpeed, 4),
        character_motion_speed: storyboardSpeedValue(state.characterMotionSpeed, 4),
        story_arc_detail: normalizeStoryArcDetail(state.storyArcDetail),
        performance_style: state.performanceStyle || "",
        facial_performance: state.facialPerformance || "",
        facial_performance_custom: state.facialPerformanceCustom || "",
        unload_after: true,
        max_new_tokens: 2400,
      }, 240000);
      if (!String(data.story_arc || "").trim()) throw new Error("The LLM returned an empty story arc.");
      state.storyLayer.user_story_arc = String(data.story_arc || "").trim();
      userStoryArcInput.value = state.storyLayer.user_story_arc;
      syncStoryLayerFromInputs({ notify: true });
      progress.set(`${authoritativeScript.enabled ? "Short-film premise" : "Story arc"} saved into the Story Layer.\nSeed: ${storyArcSeed}`, 100);
      progress.close(1600);
      createToast(`${authoritativeScript.enabled ? "Short-film premise" : "Story arc"} created. Seed: ${storyArcSeed}`);
      return state.storyLayer.user_story_arc;
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      progress.showDiagnostics?.(error?.diagnostics);
      createToast(`Story arc failed:\n${String(error?.message || error)}`, true);
      return null;
    }
  }

  async function copyStoryLayerForGpt() {
    const progress = createStoryboardProgressWindow("GPT Story Preparation");
    try {
      progress.set("Preparing Auto Mode story context...", 12);
      syncStoryLayerFromInputs();
      if (state.onPrepareStoryContext) {
        const prepared = await state.onPrepareStoryContext({
          setProgress: (message, percent) => progress.set(String(message || "Preparing story context..."), Number(percent || 50)),
        });
        if (prepared?.reference_builder || prepared?.referenceBuilder) {
          state.referenceBuilder = normalizeReferenceBuilderCatalog(prepared.reference_builder || prepared.referenceBuilder);
        }
        if (prepared?.source_lyrics || prepared?.sourceLyrics) {
          state.lineMappingLyrics = String(prepared.source_lyrics || prepared.sourceLyrics || "");
        }
      }
      progress.set("Packaging lyrics, descriptions, presets, and story settings...", 82);
      const payload = storyLayerGptPayload(state);
      const payloadText = JSON.stringify(payload, null, 2);
      let clipboardCopied = true;
      try {
        await copyTextToClipboard(payloadText);
      } catch (error) {
        clipboardCopied = false;
      }
      const lyricCount = payload.project_inputs.ordered_lyrics.length;
      const sceneCount = payload.project_inputs.scenes.length;
      const sourceLyricsPresent = Boolean(payload.project_inputs.source_lyrics);
      const lyricStatus = lyricCount ? `${lyricCount} lyric entries` : (sourceLyricsPresent ? "full pasted lyrics" : "no lyrics");
      progress.set("Story context ready. JSON review window opened.", 100);
      progress.close(1200);
      showStoryLayerGptHandoff(payloadText);
      createToast(`${clipboardCopied ? "Copied" : "Prepared"} Story Layer JSON (${sceneCount} scenes, ${lyricStatus}). Use Open ChatGPT in the JSON window.`);
    } catch (error) {
      progress.set(`Story preparation failed:\n${String(error?.message || error)}`, 100);
      progress.close(5000);
      createToast(`Could not copy Story Layer GPT JSON:\n${String(error?.message || error)}`, true);
    }
  }

  function openImportStoryJsonModal() {
    const importBackdrop = document.createElement("div");
    importBackdrop.style.cssText = "position:fixed;inset:0;z-index:100013;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box;";
    const importBox = document.createElement("div");
    importBox.style.cssText = "width:min(840px,calc(100vw - 48px));max-height:calc(100vh - 48px);border:1px solid #155e75;border-radius:10px;background:#111827;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.62);display:flex;flex-direction:column;overflow:hidden;";
    const importHeader = document.createElement("div");
    importHeader.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;background:#083f4f;border-bottom:1px solid #155e75;padding:13px 15px;";
    const importTitle = document.createElement("div");
    importTitle.innerHTML = "<div style=\"font-size:17px;font-weight:900;color:#cffafe;\">Import Story JSON</div><div style=\"font-size:12px;color:#cbd5e1;margin-top:3px;\">Paste the GPT response or load a .json file. This fills the overall idea, story arc, and story brief.</div>";
    const importClose = makeButton("Close");
    importHeader.append(importTitle, importClose);
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".json,application/json,text/plain";
    const input = document.createElement("textarea");
    input.placeholder = '{\n  "overall_story_idea": "...",\n  "user_story_arc": "...",\n  "song_story_brief": "..."\n}';
    input.spellcheck = false;
    input.style.cssText = "min-height:300px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;";
    const status = document.createElement("div");
    status.style.cssText = "min-height:18px;font-size:12px;color:#94a3b8;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const cancel = makeButton("Cancel");
    const apply = makeButton("Import Story", "purple");
    actions.append(cancel, apply);
    const body = document.createElement("div");
    body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:10px;overflow:auto;";
    body.append(fileInput, input, status, actions);
    importBox.append(importHeader, body);
    importBackdrop.append(importBox);
    document.body.append(importBackdrop);
    const closeImport = () => importBackdrop.remove();
    importClose.onclick = closeImport;
    cancel.onclick = closeImport;
    importBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === importBackdrop) closeImport();
    });
    fileInput.onchange = async () => {
      const file = fileInput.files?.[0];
      if (!file) return;
      input.value = await file.text();
      status.textContent = `Loaded ${file.name}. Review it, then click Import Story.`;
    };
    apply.onclick = () => {
      try {
        const imported = parseStoryLayerImportJson(input.value);
        if (imported.overall_story_idea) {
          state.storyLayer.overall_story_idea = imported.overall_story_idea;
          overallStoryIdeaInput.value = imported.overall_story_idea;
        }
        if (imported.user_story_arc) {
          state.storyLayer.user_story_arc = imported.user_story_arc;
          userStoryArcInput.value = imported.user_story_arc;
        }
        if (imported.song_story_brief) {
          state.storyLayer.song_story_brief = imported.song_story_brief;
          songStoryBriefInput.value = imported.song_story_brief;
        }
        syncStoryLayerFromInputs({ notify: true });
        status.textContent = "Story Layer fields updated.";
        status.style.color = "#67e8f9";
        createToast("Imported story idea, story arc, and story brief.");
        closeImport();
      } catch (error) {
        status.textContent = String(error?.message || error);
        status.style.color = "#fca5a5";
      }
    };
    input.focus();
  }

  return {
    copyStoryLayerForGpt, createStoryArcWithGemma, createStoryBriefWithGemma, detectLyricSections,
    notifyStoryboardDefaultsChanged, openImportStoryJsonModal, refreshTemporalEffectInfo,
    refreshVideoStyleInfo, syncStoryLayerFromInputs,
  };
}

export function lyricStoryStrengthText(value) {
  const strength = Math.max(0, Math.min(10, Number(value || 7)));
  if (strength <= 0) return "0 / ignore lyrics";
  if (strength <= 3) return `${strength} / mood only`;
  if (strength <= 6) return `${strength} / balanced`;
  if (strength <= 8) return `${strength} / strong lyric story`;
  return `${strength} / literal lyric anchors`;
}
