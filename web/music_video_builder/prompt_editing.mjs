import { GEMMA_VIDEO_ENHANCE_TIMEOUT_MS, postJson } from "./comfy_api.mjs";
import { escapeHtml, makeButton, makeCheckbox, makeField, toast } from "./controls.mjs";
import { recordGemmaBatchFailure, showGemmaBatchFailures } from "./dialogs.mjs";
import { sceneConceptPromptText, t2iMissingReason } from "./image_prompts.mjs";
import {
  applyTriggerPhrase,
  isInstrumentalLyricText,
  isRecoverableBuildGemmaError,
  normalizeGemmaContextLimit,
  quoteOrderedLyricCues,
  segmentUsesNoLipSyncPerformance,
} from "./prompt_text.mjs";
import { batchScopeLabel, normalizeBatchScope } from "./timeline_state.mjs";

function builderInstructionSourceLabel(source) {
  if (source === "scene") return "Scene custom instructions";
  if (source === "all_scenes") return "All-scenes custom instructions";
  return "Built-in default";
}

async function chooseBuilderInstructionPreset(key) {
  const data = await postJson("/vrgdg/music_builder/list_instruction_presets", { key });
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100008;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,calc(100vw - 36px));max-height:calc(100vh - 42px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:10px;box-sizing:border-box;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const title = document.createElement("div");
    const groupLabel = data.preset_group_label || data.label || key;
    title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Load ${escapeHtml(groupLabel)} Preset</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Showing presets shared by compatible instruction editors. Loading a preset only fills the editor. Save it after reviewing.</div>`;
    const close = makeButton("Close");
    header.append(title, close);
    const list = document.createElement("div");
    list.style.cssText = "display:flex;flex-direction:column;gap:8px;";
    const finish = (value) => {
      backdrop.remove();
      resolve(value);
    };
    if (!data.presets?.length) {
      const empty = document.createElement("div");
      empty.textContent = `No presets saved for this shared group yet: ${groupLabel}.`;
      empty.style.cssText = "border:1px solid #334155;border-radius:7px;background:#020617;color:#cbd5e1;padding:10px;font-size:12px;";
      list.append(empty);
    } else {
      for (const preset of data.presets) {
        const row = document.createElement("button");
        row.type = "button";
        row.textContent = preset.legacy ? `${preset.name} (legacy ${data.label || key})` : preset.name;
        row.title = preset.path || "";
        row.style.cssText = "text-align:left;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;font-weight:800;cursor:pointer;";
        row.onclick = () => finish(preset);
        list.append(row);
      }
    }
    close.onclick = () => finish(null);
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) finish(null);
    });
    box.append(header, list);
    backdrop.append(box);
    document.body.append(backdrop);
  });
}

function showVideoPromptEditModal(currentPrompt, modeLabel, options = {}) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100007;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(760px,calc(100vw - 36px));max-height:calc(100vh - 42px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;box-sizing:border-box;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Edit ${escapeHtml(modeLabel)} Prompt</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Text-only edit: sends only the current prompt and your requested change.</div>`;
    const close = makeButton("Close");
    header.append(title, close);
    const request = document.createElement("textarea");
    request.placeholder = "What would you like to change?";
    request.style.cssText = "width:100%;box-sizing:border-box;min-height:120px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;line-height:1.45;font-family:monospace;";
    const useSceneContext = makeCheckbox("Use full scene context", false);
    const sceneContextHint = document.createElement("div");
    sceneContextHint.textContent = "Off: sends only this prompt and your change request for a small edit. On: also sends scene notes, lyrics, motion notes, subject/location context, and the current image prompt for a fuller rewrite.";
    sceneContextHint.style.cssText = "font-size:11px;color:#94a3b8;line-height:1.35;margin-top:-6px;";
    const useStartingImage = makeCheckbox("Use starting image", false);
    const imageHint = document.createElement("div");
    imageHint.textContent = "Off: text-only prompt edit. On: also sends the current scene image so Gemma can make visual-aware motion or camera changes.";
    imageHint.style.cssText = sceneContextHint.style.cssText;
    const showStartingImage = Boolean(options.canUseStartingImage);
    useStartingImage.wrapper.style.display = showStartingImage ? "flex" : "none";
    imageHint.style.display = showStartingImage ? "" : "none";
    const promptPreview = document.createElement("pre");
    promptPreview.textContent = currentPrompt;
    promptPreview.style.cssText = "margin:0;max-height:220px;overflow:auto;white-space:pre-wrap;border:1px solid #334155;border-radius:7px;background:#020617;color:#cbd5e1;padding:10px;font-size:11px;line-height:1.4;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const update = makeButton("Update Prompt", "primary");
    actions.append(cancel, update);
    box.append(header, makeField("What would you like to change?", request), useSceneContext.wrapper, sceneContextHint, useStartingImage.wrapper, imageHint, makeField("Current prompt", promptPreview), actions);
    backdrop.append(box);
    document.body.append(backdrop);
    const finish = (value) => {
      backdrop.remove();
      resolve(value);
    };
    close.onclick = () => finish(null);
    cancel.onclick = () => finish(null);
    update.onclick = () => {
      const editRequest = String(request.value || "").trim();
      if (!editRequest) {
        toast("Tell Gemma what you want changed first.", true);
        request.focus();
        return;
      }
      finish({
        editRequest,
        useFullSceneContext: Boolean(useSceneContext.input.checked),
        useStartingImage: showStartingImage && Boolean(useStartingImage.input.checked),
      });
    };
    request.focus();
  });
}

function showImagePromptEditModal(currentPrompt, modeLabel, options = {}) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100007;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(760px,calc(100vw - 36px));max-height:calc(100vh - 42px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;box-sizing:border-box;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Edit ${escapeHtml(modeLabel)} Image Prompt</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Focused edit: sends the current image prompt and your requested change.</div>`;
    const close = makeButton("Close");
    header.append(title, close);
    const request = document.createElement("textarea");
    request.placeholder = "What would you like to change?";
    request.style.cssText = "width:100%;box-sizing:border-box;min-height:120px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;line-height:1.45;font-family:monospace;";
    const useSceneContext = makeCheckbox("Use full scene context", false);
    const sceneContextHint = document.createElement("div");
    sceneContextHint.textContent = "Off: sends only this prompt and your change request. On: also sends scene notes, lyrics, subject/location context, and reference descriptions for a fuller rewrite.";
    sceneContextHint.style.cssText = "font-size:11px;color:#94a3b8;line-height:1.35;margin-top:-6px;";
    const useReferenceImage = makeCheckbox("Use current scene image as visual reference", false);
    const imageHint = document.createElement("div");
    imageHint.textContent = "Off: text-only prompt edit. On: also sends the selected scene image so Gemma can make visual-aware prompt changes.";
    imageHint.style.cssText = sceneContextHint.style.cssText;
    const showReferenceImage = Boolean(options.canUseReferenceImage);
    useReferenceImage.wrapper.style.display = showReferenceImage ? "flex" : "none";
    imageHint.style.display = showReferenceImage ? "" : "none";
    const promptPreview = document.createElement("pre");
    promptPreview.textContent = currentPrompt;
    promptPreview.style.cssText = "margin:0;max-height:220px;overflow:auto;white-space:pre-wrap;border:1px solid #334155;border-radius:7px;background:#020617;color:#cbd5e1;padding:10px;font-size:11px;line-height:1.4;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const update = makeButton("Update Prompt", "primary");
    actions.append(cancel, update);
    box.append(header, makeField("What would you like to change?", request), useSceneContext.wrapper, sceneContextHint, useReferenceImage.wrapper, imageHint, makeField("Current prompt", promptPreview), actions);
    backdrop.append(box);
    document.body.append(backdrop);
    const finish = (value) => {
      backdrop.remove();
      resolve(value);
    };
    close.onclick = () => finish(null);
    cancel.onclick = () => finish(null);
    update.onclick = () => {
      const editRequest = String(request.value || "").trim();
      if (!editRequest) {
        toast("Tell Gemma what you want changed first.", true);
        request.focus();
        return;
      }
      finish({
        editRequest,
        useFullSceneContext: Boolean(useSceneContext.input.checked),
        useReferenceImage: showReferenceImage && Boolean(useReferenceImage.input.checked),
      });
    };
    request.focus();
  });
}

export function createPromptEditing({
  activeProjectFolderForSave, activeScenePromptForEnhance, activeSegment, applyImageTriggerToPrompt,
  applyMappedTriggerPhrases, applyVocalDirectiveToVideoPrompt, archiveGeneratedSceneImage,
  assertBatchNotStopped, autoSaveSessionQuiet, createProgressWindow, createT2IButton, currentEnhanceSource,
  currentVideoMode, editI2VPromptButton, editImagePromptButtons, effectiveVideoPerformanceModeForSegment,
  enhanceImageForSegment, ernieGemmaModelSelect, ernieMmprojSelect, ernieTextGemmaModelSelect,
  facialPerformanceNoteForSegment, firstLastFramePromptReferences, flfGemmaContextMode, flfGemmaSceneConcept,
  flfGemmaVisualNotes, flfTransitionLoraActive, fluxGemmaModelSelect, fluxMmprojSelect,
  fluxReferenceContextForSegment, gemmaModelSelect, gemmaRunnerLine, generateT2IPromptForSegment,
  i2vGemmaModelSelect, i2vMmprojSelect, i2vPrompt, i2vTextGemmaModelSelect, idLoraGemmaNotesForSegment,
  idLoraSceneContext, idLoraSpeechTextForSegment, imageModeDisplayLabel, llmApiVisionModelSelected,
  ltx25SelectedCastCoverageContract, mmprojSelect, nbGemmaModelSelect, nbImageSettingsForSegment,
  nbMmprojSelect, pushHistory, render, requireActiveSegment, rtvReferenceBehaviorForSegment,
  saveGemmaJunkDebug, saveZEnhanceSettingsFromPanel, sceneDisplayName, sceneSlotNumber,
  sceneVideoConceptPromptText, segmentImageSource, segmentIndexInfo, segmentMappedLocationText,
  segmentMappedSubjectText, segmentPromptForEdit, state, storyboardVideoExtraNotesForSegment, syncInspector,
  syncPreview, syncSegmentT2IPrompt, syncVideoModePanel, t2iTextGemmaModelSelect, textGemmaRunnerPayload,
  updateActiveFromInputs, videoGemmaNotesForSegment, videoModeDisplayLabel, videoTriggerPhraseForSegment,
  zEnhanceButton, zEnhanceGemmaButton, zEnhanceGemmaModelSelect, zEnhanceGemmaNotes, zEnhanceMmprojSelect,
  zEnhancePromptPreview,
}) {
  async function generateEnhancePromptWithGemma() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    let source = currentEnhanceSource(segment);
    if (!source?.path && !source?.data && segment.image?.filename) {
      const archived = await archiveGeneratedSceneImage(segment, segment.image);
      if (archived) source = { path: archived, name: "scene_image.png" };
    }
    if (!source?.path && !source?.data) {
      toast("Hey, load or create an image first so Gemma can see what to enhance.", true);
      return;
    }
    const modelFile = String(zEnhanceGemmaModelSelect.value || "").trim();
    const mmprojFile = String(zEnhanceMmprojSelect.value || "").trim();
    if (!modelFile || !mmprojFile) {
      toast("Choose the Enhance vision Gemma model and mmproj in the Enhance Models tab first.", true);
      return;
    }
    const notes = String(zEnhanceGemmaNotes.value || "").trim();
    let progress = null;
    try {
      zEnhanceGemmaButton.disabled = true;
      zEnhanceGemmaButton.textContent = "Gemma...";
      progress = createProgressWindow("Creating Enhance prompt");
      progress.set(`Reading selected image with Gemma vision...\n${gemmaRunnerLine({ vision: true })}`, 20);
      const userNotes = [
        "Create a text-to-image prompt for image-to-image upscale/enhance.",
        "Describe the visible subject, setting, lighting, clothing, pose, composition, and mood from the image.",
        "Preserve the important identity and scene details from the image.",
        notes ? `User enhancement notes:\n${notes}` : "User enhancement notes:\nKeep the image identity, improve cinematic detail, clarity, lighting, and polish.",
      ].join("\n\n");
      const data = await postJson("/vrgdg/music_builder/generate_t2i", {
        model_file: modelFile,
        mmproj_file: mmprojFile,
        use_vision: true,
        ref_image_path: source.path || "",
        ref_image_data: source.data || "",
        scene_number: sceneSlotNumber(segment),
        user_notes: userNotes,
        unload_after: true,
        max_new_tokens: 1000,
      }, 10 * 60 * 1000);
      pushHistory();
      segment.enhance_prompt = applyTriggerPhrase(data.prompt, state.imageTriggerPhrase);
      zEnhancePromptPreview.value = segment.enhance_prompt;
      progress.set("Enhance prompt ready.", 100);
      progress.close(900);
      render();
      await autoSaveSessionQuiet("Gemma enhance prompt");
      toast("Gemma created the Enhance prompt.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      zEnhanceGemmaButton.disabled = false;
      zEnhanceGemmaButton.textContent = "Gemma Enhance Prompt";
    }
  }

  async function upscaleEnhanceImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const source = currentEnhanceSource(segment);
    if (!source?.path && !source?.data && !segment.image?.filename) {
      toast("Hey, you need an image first. Create, save, load, or choose a scene image before using upscale/enhance.", true);
      return;
    }
    const enhancePromptInfo = activeScenePromptForEnhance({ copyFallback: true });
    const enhancePrompt = enhancePromptInfo.prompt;
    if (!enhancePrompt) {
      toast("Hey, this scene needs a T2I prompt first. Create one with Gemma T2I, type one into the T2I prompt box, or create a Flux/Klein prompt.", true);
      return;
    }
    const settings = saveZEnhanceSettingsFromPanel();
    let progress = null;
    try {
      zEnhanceButton.disabled = true;
      zEnhanceButton.textContent = "Enhancing...";
      progress = createProgressWindow("Upscale / Enhance image");
      progress.set("Autosaving session/SRT before Enhance...", 8);
      await autoSaveSessionQuiet("upscale/enhance");
      pushHistory();
      await enhanceImageForSegment(segment, progress, 20, 70, "Upscale / Enhance image", {
        settings,
        source,
        promptSource: "scene_image_prompt",
      });
      syncPreview(segment);
      render();
      await autoSaveSessionQuiet("upscale/enhance complete");
      progress.set("Enhanced image ready.", 100);
      progress.close(900);
      toast("Upscale/enhance image ready.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      zEnhanceButton.disabled = false;
      zEnhanceButton.textContent = "Upscale / Enhance Image";
    }
  }

  async function createT2IPromptWithGemma() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const missing = t2iMissingReason(segment);
    if (missing) {
      toast(`Hey, ${missing}`, true);
      return;
    }
    let progress = null;
    try {
      createT2IButton.disabled = true;
      createT2IButton.textContent = "Gemma...";
      progress = createProgressWindow("Creating T2I prompt");
      progress.set("Autosaving session/SRT before Gemma T2I...", 8);
      await autoSaveSessionQuiet("Gemma T2I");
      const data = await generateT2IPromptForSegment(segment, progress, 45, segment.use_vision_reference ? "Gemma with reference image" : "Gemma from notes");
      await autoSaveSessionQuiet("Gemma T2I complete");
      progress.set("T2I prompt ready.", 100);
      progress.close(900);
      toast(data.used_storyboard_prompt_writer
        ? "Gemma created T2I from the storyboard scene card."
        : data.used_reference_image
          ? "Gemma created T2I from reference image."
          : "Gemma created T2I from notes.");
    } catch (error) {
      if (isRecoverableBuildGemmaError(error)) {
        const debugPath = error?.gemmaDebugPath || await saveGemmaJunkDebug(error, { label: "single T2I prompt", segment });
        const failure = recordGemmaBatchFailure(`t2i:${state.imageModelMode || "zimage"}:${segment.id}`, segment, sceneDisplayName(segment), error, debugPath);
        showGemmaBatchFailures([failure], {
          retryHandler: () => {
            if (activeSegment()?.id !== segment.id) throw new Error(`Select ${failure.sceneLabel} to retry it.`);
            return createT2IPromptWithGemma();
          },
        });
      }
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      createT2IButton.disabled = false;
      createT2IButton.textContent = "Gemma T2I";
    }
  }

  function getI2VImageReference(segment) {
    if (!segment) return { path: "", data: "" };
    const source = segmentImageSource(segment);
    if (source?.data) return { path: "", data: source.data };
    if (source?.path) return { path: source.path, data: "" };
    return { path: "", data: "" };
  }

  async function openBuilderInstructionEditor(key = "i2v") {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const projectFolder = activeProjectFolderForSave();
    if (!projectFolder) {
      toast("Create or load a Builder project before editing Gemma instructions.", true);
      return;
    }
    let data;
    try {
      data = await postJson("/vrgdg/music_builder/get_instruction", {
        project_folder: projectFolder,
        key,
        scene_id: segment.id || "",
      });
    } catch (error) {
      toast(`Could not load Builder instructions:\n${String(error?.message || error)}`, true);
      return;
    }

    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100007;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(920px,calc(100vw - 36px));max-height:calc(100vh - 42px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;box-sizing:border-box;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;";
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Edit ${escapeHtml(data.label || "I2V")} Gemma Instructions</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">${escapeHtml(sceneDisplayName(segment, segmentIndexInfo(segment).index))}</div>`;
    const close = makeButton("Close");
    header.append(title, close);
    const status = document.createElement("div");
    status.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;color:#dbeafe;padding:10px;font-size:12px;line-height:1.45;";
    const setStatus = (nextData, message = "") => {
      data = nextData || data;
      const lines = [
        `Currently using: ${builderInstructionSourceLabel(data.source)}`,
        data.path ? `Path: ${data.path}` : "",
        message,
      ].filter(Boolean);
      status.textContent = lines.join("\n");
    };
    const editor = document.createElement("textarea");
    editor.value = data.text || data.default_text || "";
    editor.spellcheck = false;
    editor.style.cssText = "width:100%;box-sizing:border-box;min-height:420px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;line-height:1.45;font-family:monospace;";
    let appliedInstructionText = String(editor.value || "").trim();
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;";
    const instructionLabel = data.label || key;
    const saveScene = makeButton("Save for This Scene", "primary");
    const saveAll = makeButton("Save for All Scenes", "primary");
    const resetScene = makeButton("Reset Scene");
    const resetAll = makeButton("Reset All Scenes");
    const savePreset = makeButton("Save Preset");
    const loadPreset = makeButton("Load Preset");
    actions.append(saveScene, saveAll, resetScene, resetAll, savePreset, loadPreset);
    const hasUnsavedInstructionChanges = () => String(editor.value || "").trim() !== appliedInstructionText;
    const closeEditor = () => {
      if (hasUnsavedInstructionChanges() && !window.confirm("The custom instructions in this editor have not been saved, so they will not be used yet.\n\nUse Save for This Scene or Save for All Scenes if you want these instructions to apply.\n\nClose without saving?")) {
        return;
      }
      backdrop.remove();
    };
    close.onclick = closeEditor;
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) closeEditor();
    });
    const saveScope = async (scope) => {
      const text = String(editor.value || "").trim();
      if (!text) {
        toast("Instruction text is empty.", true);
        return;
      }
      try {
        const saved = await postJson("/vrgdg/music_builder/save_instruction", {
          project_folder: projectFolder,
          key,
          scene_id: segment.id || "",
          scope,
          text,
        });
        appliedInstructionText = String(saved.text || text || "").trim();
        setStatus(saved, scope === "all_scenes" ? "Saved for all scenes in this project." : "Saved for this scene.");
        toast(scope === "all_scenes" ? `Saved ${instructionLabel} instructions for all scenes.` : `Saved ${instructionLabel} instructions for this scene.`);
      } catch (error) {
        toast(`Could not save instructions:\n${String(error?.message || error)}`, true);
      }
    };
    saveScene.onclick = () => saveScope("scene");
    saveAll.onclick = () => saveScope("all_scenes");
    resetScene.onclick = async () => {
      if (!window.confirm(`Reset this scene's ${instructionLabel} instructions?`)) return;
      try {
        const reset = await postJson("/vrgdg/music_builder/reset_instruction", {
          project_folder: projectFolder,
          key,
          scene_id: segment.id || "",
          scope: "scene",
        });
        editor.value = reset.text || reset.default_text || "";
        appliedInstructionText = String(editor.value || "").trim();
        setStatus(reset, "Scene override reset.");
      } catch (error) {
        toast(`Could not reset scene instructions:\n${String(error?.message || error)}`, true);
      }
    };
    resetAll.onclick = async () => {
      if (!window.confirm(`Reset all-scenes ${instructionLabel} instructions for this project?`)) return;
      try {
        const reset = await postJson("/vrgdg/music_builder/reset_instruction", {
          project_folder: projectFolder,
          key,
          scene_id: segment.id || "",
          scope: "all_scenes",
        });
        editor.value = reset.text || reset.default_text || "";
        appliedInstructionText = String(editor.value || "").trim();
        setStatus(reset, "All-scenes override reset.");
      } catch (error) {
        toast(`Could not reset all-scenes instructions:\n${String(error?.message || error)}`, true);
      }
    };
    savePreset.onclick = async () => {
      const name = window.prompt("Preset name:");
      if (!name) return;
      try {
        const preset = await postJson("/vrgdg/music_builder/save_instruction_preset", {
          key,
          name,
          text: editor.value,
        });
        setStatus(data, `Saved shared preset: ${preset.name}\nGroup: ${preset.preset_group_label || instructionLabel}\nPreset path: ${preset.path || ""}\nPreset folder: ${preset.preset_folder || ""}`);
        toast(`Saved ${preset.preset_group_label || instructionLabel} preset: ${preset.name}`);
      } catch (error) {
        toast(`Could not save preset:\n${String(error?.message || error)}`, true);
      }
    };
    loadPreset.onclick = async () => {
      try {
        const preset = await chooseBuilderInstructionPreset(key);
        if (!preset) return;
        const loaded = await postJson("/vrgdg/music_builder/load_instruction_preset", {
          key,
          name: preset.name,
        });
        editor.value = loaded.text || "";
        setStatus(data, `Loaded shared preset: ${loaded.name}\nGroup: ${loaded.preset_group_label || instructionLabel}\nReview it, then save for this scene or all scenes.\nPreset path: ${loaded.path || ""}`);
      } catch (error) {
        toast(`Could not load preset:\n${String(error?.message || error)}`, true);
      }
    };
    setStatus(data);
    box.append(header, status, makeField("Instructions", editor), actions);
    backdrop.append(box);
    document.body.append(backdrop);
    editor.focus();
  }

  async function enhanceVideoPromptForSegment(segment, draftPrompt, progress = null, percent = 80, label = "I2V prompt enhancement", options = {}) {
    const base = String(draftPrompt || "").trim();
    if ((!state.useI2VPromptEnhancementPass && !options.force) || !base) return base;
    const videoMode = currentVideoMode();
    const isT2V = videoMode === "t2v";
    const isRTV = videoMode === "rtv";
    const isFLF = videoMode === "flf";
    const isIngredients = videoMode === "ingredients";
    const modeLabel = videoModeDisplayLabel(videoMode, true);
    if (isFLF && flfGemmaContextMode(segment) !== "full") return base;
    const lyricText = quoteOrderedLyricCues(String(segment?.lyric_text || "").trim()).trim();
    const noVocal = Boolean(segmentUsesNoLipSyncPerformance(segment) || isInstrumentalLyricText(lyricText));
    const singers = Array.isArray(segment?.lyric_singers) ? segment.lyric_singers.map((value) => String(value || "").trim()).filter(Boolean) : [];
    progress?.set(`${label}: improving ${modeLabel} prompt shape...\n${gemmaRunnerLine()}`, percent);
    const data = await postJson("/vrgdg/music_builder/enhance_video_prompt", {
      ...textGemmaRunnerPayload(),
      model_file: i2vTextGemmaModelSelect.value,
      repair_model_file: i2vTextGemmaModelSelect.value,
      draft_prompt: base,
      mode_label: modeLabel,
      performance_mode: effectiveVideoPerformanceModeForSegment(segment),
      t2i_prompt: isFLF ? flfGemmaSceneConcept(segment) : sceneVideoConceptPromptText(segment),
      user_notes: isFLF ? flfGemmaVisualNotes(segment) : [facialPerformanceNoteForSegment(segment), String(segment?.i2v_notes || "").trim()].filter(Boolean).join("\n\n"),
      lyric_text: noVocal ? "" : lyricText.replace(/^["'“”‘’]+|["'“”‘’]+$/g, ""),
      singers: segment?.no_character_present ? [] : singers,
      no_vocal: noVocal,
      no_character_present: Boolean(segment?.no_character_present),
      unload_after: options.unloadAfter !== false,
      n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
      max_new_tokens: 1200,
    }, GEMMA_VIDEO_ENHANCE_TIMEOUT_MS);
    return String(data.prompt || "").trim() || base;
  }

  function normalizeIdLoraScriptPrompt(segment, rawPrompt) {
    const text = String(rawPrompt || "").replace(/\r\n/g, "\n").trim();
    const sectionValue = (name) => {
      const pattern = new RegExp(`\\[${name}\\]\\s*:?\\s*([\\s\\S]*?)(?=\\n\\s*\\[(?:VISUAL|SPEECH|SOUNDS)\\]\\s*:?|$)`, "i");
      return String(text.match(pattern)?.[1] || "").trim();
    };
    const visual = sectionValue("VISUAL") || text.replace(/\[(?:VISUAL|SPEECH|SOUNDS)\]\s*:?\s*/gi, "").trim();
    const speech = sectionValue("SPEECH") || idLoraSpeechTextForSegment(segment) || "I can feel this moment changing.";
    const sounds = sectionValue("SOUNDS") || "Soft room tone, subtle breath, and quiet ambient space.";
    return [
      `[VISUAL]: ${visual || "A cinematic close-up of the subject holding a focused expression while the camera moves gently."}`,
      `[SPEECH]: ${speech}`,
      `[SOUNDS]: ${sounds}`,
    ].join("\n");
  }

  async function finalizeVideoPromptForSegment(segment, rawPrompt, progress = null, percent = 82, label = "I2V prompt enhancement", options = {}) {
    if (currentVideoMode() === "id_lora") {
      return normalizeIdLoraScriptPrompt(segment, rawPrompt);
    }
    const directiveOptions = { suppressPrefix: Boolean(options.suppressVocalPrefix) };
    const draft = applyVocalDirectiveToVideoPrompt(rawPrompt, segment, directiveOptions);
    const enhanced = await enhanceVideoPromptForSegment(segment, draft, progress, percent, label, options);
    return applyMappedTriggerPhrases(applyVocalDirectiveToVideoPrompt(applyTriggerPhrase(enhanced, videoTriggerPhraseForSegment(segment)), segment, directiveOptions), segment, { ensureTransitionLast: true });
  }

  function finalizeVideoPromptDraftOnly(segment, rawPrompt) {
    if (currentVideoMode() === "id_lora") {
      return normalizeIdLoraScriptPrompt(segment, rawPrompt);
    }
    const draft = applyVocalDirectiveToVideoPrompt(rawPrompt, segment);
    return applyMappedTriggerPhrases(applyVocalDirectiveToVideoPrompt(applyTriggerPhrase(draft, videoTriggerPhraseForSegment(segment)), segment), segment, { ensureTransitionLast: true });
  }

  function buildI2VPromptRequestForSegment(segment, options = {}) {
    if (!segment) throw new Error("Scene is missing.");
    const videoMode = currentVideoMode();
    const isT2V = videoMode === "t2v";
    const isRTV = videoMode === "rtv";
    const isFLF = videoMode === "flf";
    const isIngredients = videoMode === "ingredients";
    const isIdLora = videoMode === "id_lora";
    const modeLabel = videoModeDisplayLabel(videoMode, false);
    const provisionalMotionPlan = Boolean(options.provisionalMotionPlan);
    const forceTextOnly = Boolean(options.forceTextOnly);
    const forceVision = Boolean(options.forceVision);
    const textScriptMode = isRTV || isFLF || isIngredients || isIdLora;
    const useFirstLastFrameVision = !provisionalMotionPlan && (isFLF || (isRTV && rtvReferenceBehaviorForSegment(segment) === "first_last_frame"));
    const firstLastFrameReferences = useFirstLastFrameVision ? firstLastFramePromptReferences(segment) : [];
    const useImageReference = provisionalMotionPlan ? true : useFirstLastFrameVision ? true : isIdLora ? true : textScriptMode ? false : forceVision ? true : forceTextOnly ? false : (isT2V ? Boolean(segment.use_t2v_vision_reference) : segment.use_i2v_vision_reference !== false);
    const imageReference = provisionalMotionPlan
      ? (segmentImageSource(segment) || { path: "", data: "" })
      : useFirstLastFrameVision
        ? (firstLastFrameReferences[0] || { path: "", data: "" })
        : useImageReference ? getI2VImageReference(segment) : { path: "", data: "" };
    const t2iText = isFLF ? flfGemmaSceneConcept(segment) : sceneVideoConceptPromptText(segment);
    if (useFirstLastFrameVision && firstLastFrameReferences.length < 2) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: First Last Frame prompt needs both a first frame image and an end frame image.`);
    }
    if (useImageReference && !imageReference.path && !imageReference.data) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: ${modeLabel} image reference is enabled, but no reference image was found.`);
    }
    if (useImageReference && state.textGemmaRunner === "llm_api" && !llmApiVisionModelSelected()) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: LLM API vision needs a vision-capable API model selected. Open LLM Runner and choose an API model that supports images.`);
    }
    const idLoraContext = isIdLora ? idLoraSceneContext(segment) : null;
    if (!useFirstLastFrameVision && (isT2V || textScriptMode || !useImageReference) && !t2iText && !(isIdLora && idLoraContext?.contextText)) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: scene text/concept is missing.`);
    }
    const baseNotes = videoGemmaNotesForSegment(segment);
    const modeNotes = isIdLora ? idLoraGemmaNotesForSegment(segment, baseNotes) : isFLF
      ? flfGemmaVisualNotes(segment)
      : baseNotes;
    const storyboardNotes = String(options.skipStoryboardExtraNotes || (isFLF && flfGemmaContextMode(segment) !== "full") ? "" : storyboardVideoExtraNotesForSegment(segment, options.storyboardScene)).trim();
    const provisionalInstructions = provisionalMotionPlan ? [
      "INDEPENDENT FIRST/LAST FRAME MOTION PLAN:",
      "Study the supplied start image and write a complete I2V motion plan for this scene.",
      "Describe continuous camera motion, subject action, environment motion, and a visually precise final moment.",
      "The final sentence must clearly state what should be visible in the final frozen frame, including shot scale, camera angle, subject pose/action endpoint, expression, visible wardrobe, important props, and environment.",
      "Use shot-aware planning: for a close-up, consider a modest pullback, arc, pan, tilt, expression change, or upper-body reveal; for a medium shot, consider a push toward the face, hands, or a story-relevant prop; for a wide shot, consider pushing toward the subject, upper body, face, or an important environment detail.",
      "Do not choose an arbitrary hand, foot, or body-detail endpoint unless the lyric, story beat, action, or user direction makes that detail meaningful.",
      "Do not mention an end-frame image because it has not been created yet. Do not reuse another scene's ending.",
      segment.flf_endpoint_mode === "custom" && String(segment.flf_custom_end_direction || "").trim()
        ? `REQUIRED USER ENDING DIRECTION:\n${String(segment.flf_custom_end_direction || "").trim()}`
        : "Choose an ending that follows naturally from the story beat and start image. Avoid revealing unseen wardrobe or anatomy unless the supplied character description specifies those details.",
    ].filter(Boolean).join("\n") : "";
    const selectedCastContract = ltx25SelectedCastCoverageContract(segment);
    const extraNotes = [storyboardNotes, selectedCastContract, provisionalInstructions, String(options.extraUserNotes || "").trim()].filter(Boolean).join("\n\n");
    const mappedSubjectContext = segment.no_character_present ? "" : segmentMappedSubjectText(segment);
    const mappedLocationContext = segmentMappedLocationText(segment);
    return {
      endpoint: provisionalMotionPlan ? "/vrgdg/music_builder/generate_i2v" : isT2V || textScriptMode ? "/vrgdg/music_builder/generate_t2v" : "/vrgdg/music_builder/generate_i2v",
      modeLabel: provisionalMotionPlan ? "independent FLF motion-plan" : modeLabel,
      useImageReference,
      payload: {
        ...textGemmaRunnerPayload(),
        project_folder: activeProjectFolderForSave(),
        scene_id: segment.id || "",
        builder_instruction_key: provisionalMotionPlan ? "i2v" : isIdLora ? "id_lora" : isIngredients ? "ingredients" : isRTV ? "rtv" : isT2V ? "t2v" : "i2v",
        model_file: useImageReference ? i2vGemmaModelSelect.value : i2vTextGemmaModelSelect.value,
        mmproj_file: useImageReference ? i2vMmprojSelect.value : "",
        t2i_prompt: isIdLora ? [t2iText, idLoraContext?.contextText || ""].filter(Boolean).join("\n\n") : isT2V || textScriptMode ? t2iText : useImageReference ? "" : t2iText,
        performance_mode: effectiveVideoPerformanceModeForSegment(segment),
        image_reference_path: imageReference.path,
        image_reference_data: imageReference.data,
        image_references: firstLastFrameReferences,
        first_last_frame_mode: useFirstLastFrameVision,
        flf_context_mode: isFLF ? flfGemmaContextMode(segment) : "full",
        transition_lora_active: isFLF && flfTransitionLoraActive(segment),
        flf_start_state: isFLF ? String(segment.flf_start_state || "").trim() : "",
        flf_transformation: isFLF ? String(segment.flf_transformation || "").trim() : "",
        flf_end_state: isFLF ? String(segment.flf_end_state || "").trim() : "",
        flf_carry_forward: isFLF ? String(segment.flf_carry_forward || "").trim() : "",
        repair_model_file: i2vTextGemmaModelSelect.value,
        user_notes: (options.skipStoryboardExtraNotes ? [extraNotes, modeNotes] : [modeNotes, extraNotes]).filter(Boolean).join("\n\n"),
        subject_context: provisionalMotionPlan ? mappedSubjectContext : isIdLora ? [idLoraContext?.characterName || "", String(idLoraContext?.character?.description || "").trim()].filter(Boolean).join("\n") : isFLF && flfGemmaContextMode(segment) !== "full" ? "" : segment.no_character_present ? "" : (isT2V || textScriptMode || !useImageReference ? mappedSubjectContext : ""),
        location_context: provisionalMotionPlan ? mappedLocationContext : isIdLora ? [idLoraContext?.locationName || "", String(idLoraContext?.location?.description || "").trim()].filter(Boolean).join("\n") : isFLF && flfGemmaContextMode(segment) !== "full" ? "" : isT2V || textScriptMode || !useImageReference ? mappedLocationContext : "",
        no_character_present: Boolean(segment.no_character_present),
        theme_style_path: useImageReference && !isT2V && !textScriptMode ? "" : state.useVrgdgTextContext ? state.themeStylePath || "" : "",
        story_idea_path: useImageReference && !isT2V && !textScriptMode ? "" : state.useVrgdgTextContext ? state.storyIdeaPath || "" : "",
        subject_scene_path: useImageReference && !isT2V && !textScriptMode ? "" : state.useVrgdgTextContext ? state.subjectScenePath || "" : "",
        unload_after: options.deferEnhancement ? options.unloadAfter !== false : state.useI2VPromptEnhancementPass ? useImageReference : options.unloadAfter !== false,
      },
    };
  }

  async function runVideoPromptEnhancementBatch(segments, progress = null, options = {}) {
    const modeLabel = videoModeDisplayLabel(currentVideoMode(), true);
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const force = Boolean(options.force);
    const targets = (segments || []).filter((segment) => String(segment?.i2v_prompt || "").trim());
    if (!targets.length) {
      progress?.set(`No existing ${modeLabel} prompts found to enhance.`, 100);
      return 0;
    }
    for (let index = 0; index < targets.length; index += 1) {
      assertBatchNotStopped();
      const segment = targets[index];
      const displayIndex = segmentIndexInfo(segment).index;
      state.activeId = segment.id;
      syncInspector();
      render();
      const percent = Math.min(98, Number(options.percentBase || 0) + Math.floor(((index + 1) / targets.length) * Number(options.percentSpan || 90)));
      progress?.set(`Gemma ${modeLabel} enhancement ${index + 1}/${targets.length}: ${sceneDisplayName(segment, displayIndex)}\nScope: ${batchScopeLabel(sceneScope)}\n${gemmaRunnerLine()}`, percent);
      pushHistory();
      try {
        segment.i2v_prompt = await finalizeVideoPromptForSegment(
          segment,
          segment.i2v_prompt,
          progress,
          percent,
          `Gemma ${modeLabel} enhancement ${index + 1}/${targets.length}`,
          { force, unloadAfter: index === targets.length - 1 },
        );
      } catch (error) {
        await saveGemmaJunkDebug(error, {
          label: `Gemma ${modeLabel} enhancement ${index + 1}/${targets.length}`,
          segment,
        });
        throw error;
      }
      if (segment.id === state.activeId) i2vPrompt.value = segment.i2v_prompt;
      render();
      await autoSaveSessionQuiet(`Gemma ${modeLabel} enhancement ${sceneDisplayName(segment, displayIndex)}`);
    }
    return targets.length;
  }

  function imagePromptEditModelPayload(imageMode, useReferenceImage) {
    if (imageMode === "ernie_image") {
      return {
        model_file: useReferenceImage ? ernieGemmaModelSelect.value : ernieTextGemmaModelSelect.value,
        repair_model_file: ernieTextGemmaModelSelect.value,
        mmproj_file: useReferenceImage ? ernieMmprojSelect.value : "",
      };
    }
    if (imageMode === "flux_klein") {
      return {
        model_file: useReferenceImage ? fluxGemmaModelSelect.value : (t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || ""),
        repair_model_file: t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "",
        mmproj_file: useReferenceImage ? fluxMmprojSelect.value : "",
      };
    }
    if (imageMode === "nano_banana" || imageMode === "flow_gpt") {
      return {
        model_file: useReferenceImage ? nbGemmaModelSelect.value : (t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || ""),
        repair_model_file: t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "",
        mmproj_file: useReferenceImage ? nbMmprojSelect.value : "",
      };
    }
    return {
      model_file: useReferenceImage ? gemmaModelSelect.value : t2iTextGemmaModelSelect.value,
      repair_model_file: t2iTextGemmaModelSelect.value,
      mmproj_file: useReferenceImage ? mmprojSelect.value : "",
    };
  }

  async function editCurrentImagePromptWithGemma() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const imageMode = state.imageModelMode || "zimage";
    if (imageMode === "z_enhance") {
      toast("Enhance has its own Gemma Enhance Prompt tool. Switch to an image model prompt to edit scene image prompts.", true);
      return;
    }
    const currentPrompt = String(segmentPromptForEdit(segment, "t2i") || "").trim();
    if (!currentPrompt) {
      toast("Create or type an image prompt first, then Gemma can edit it.", true);
      return;
    }
    const modeLabel = imageModeDisplayLabel(imageMode, true);
    const imageReference = segmentImageSource(segment) || (segment.ref_image_path ? { path: segment.ref_image_path } : null);
    const canUseReferenceImage = Boolean(imageReference?.path || imageReference?.data);
    const editOptions = await showImagePromptEditModal(currentPrompt, modeLabel, { canUseReferenceImage });
    if (!editOptions) return;
    const editRequest = String(editOptions.editRequest || "").trim();
    const useFullSceneContext = Boolean(editOptions.useFullSceneContext);
    const useReferenceImage = canUseReferenceImage && Boolean(editOptions.useReferenceImage);
    if (useReferenceImage && state.textGemmaRunner === "llm_api" && !llmApiVisionModelSelected()) {
      toast("Hey, LLM API vision needs a vision-capable API model selected. Open LLM Runner, choose an API model that supports images, then try again.", true);
      return;
    }
    const buttons = editImagePromptButtons;
    let progress = null;
    try {
      buttons.forEach((button) => {
        button.disabled = true;
        button.textContent = "Editing...";
      });
      progress = createProgressWindow(`Editing ${modeLabel} image prompt`);
      progress.set(`Running ${useReferenceImage ? "vision-assisted" : "text-only"} image prompt edit${useFullSceneContext ? " with scene context" : ""}...\n${gemmaRunnerLine({ vision: useReferenceImage })}`, 35);
      const data = await postJson("/vrgdg/music_builder/edit_image_prompt", {
        ...textGemmaRunnerPayload(),
        ...imagePromptEditModelPayload(imageMode, useReferenceImage),
        current_prompt: currentPrompt,
        edit_request: editRequest,
        mode_label: modeLabel,
        prompt_mode: imageMode,
        use_vision_reference: useReferenceImage,
        ref_image_path: useReferenceImage ? (imageReference.path || "") : "",
        ref_image_data: useReferenceImage ? (imageReference.data || "") : "",
        use_full_scene_context: useFullSceneContext,
        reference_context: imageMode === "nano_banana" || imageMode === "flow_gpt" ? nbImageSettingsForSegment(segment).reference_context : imageMode === "flux_klein" ? fluxReferenceContextForSegment(segment) : {},
        scene_context: useFullSceneContext ? {
          label: sceneDisplayName(segment, segmentIndexInfo(segment).index),
          scene_notes: String(segment.notes || segment.nb_notes || segment.flux_notes || "").trim(),
          director_note: String(segment.timeline_note || "").trim(),
          lyric_text: String(segment.lyric_text || "").trim(),
          lyric_section: String(segment.lyric_section || "").trim(),
          subject_context: segment.no_character_present ? "" : segmentMappedSubjectText(segment),
          location_context: segmentMappedLocationText(segment),
          no_character_present: Boolean(segment.no_character_present),
        } : {},
        unload_after: true,
        n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
        temperature: 0.25,
        top_p: 0.9,
        max_new_tokens: 1200,
      }, 180000);
      const nextPrompt = String(data.prompt || "").trim();
      if (!nextPrompt) throw new Error("Gemma returned an empty edited image prompt.");
      pushHistory();
      syncSegmentT2IPrompt(segment, applyMappedTriggerPhrases(applyImageTriggerToPrompt(nextPrompt, segment, imageMode, { validateJunk: true }), segment));
      editImagePromptButtons.forEach((button) => {
        button.style.display = "";
      });
      render();
      await autoSaveSessionQuiet(`Gemma edited ${modeLabel} image prompt`);
      progress.set("Edited image prompt ready.", 100);
      progress.close(900);
      toast(`Gemma edited the ${modeLabel} image prompt.`);
    } catch (error) {
      const debugPath = error?.gemmaDebugPath || await saveGemmaJunkDebug(error, { label: `edit ${modeLabel} image prompt`, segment });
      progress?.set(`Error:\n${String(error?.message || error)}${debugPath ? `\n\nRaw Gemma output saved to:\n${debugPath}` : ""}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      buttons.forEach((button) => {
        button.disabled = false;
        button.textContent = "Edit Prompt";
      });
      syncInspector();
    }
  }

  async function editCurrentVideoPromptWithGemma() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const currentPrompt = String(segment.i2v_prompt || i2vPrompt.value || "").trim();
    if (!currentPrompt) {
      toast("Create or type a video prompt first, then Gemma can edit it.", true);
      return;
    }
    const modeLabel = videoModeDisplayLabel(currentVideoMode(), true);
    const videoMode = currentVideoMode();
    const imageReference = videoMode === "i2v" ? getI2VImageReference(segment) : { path: "", data: "" };
    const canUseStartingImage = videoMode === "i2v" && Boolean(imageReference.path || imageReference.data);
    const editOptions = await showVideoPromptEditModal(currentPrompt, modeLabel, { canUseStartingImage });
    if (!editOptions) return;
    const editRequest = String(editOptions.editRequest || "").trim();
    const useFullSceneContext = Boolean(editOptions.useFullSceneContext);
    const useStartingImage = canUseStartingImage && Boolean(editOptions.useStartingImage);
    if (useStartingImage && state.textGemmaRunner === "llm_api" && !llmApiVisionModelSelected()) {
      toast("Hey, LLM API vision needs a vision-capable API model selected. Open LLM Runner, choose an API model that supports images, then try again.", true);
      return;
    }
    const singers = Array.isArray(segment.lyric_singers) ? segment.lyric_singers.map((value) => String(value || "").trim()).filter(Boolean) : [];
    let progress = null;
    try {
      editI2VPromptButton.disabled = true;
      editI2VPromptButton.textContent = "Editing...";
      progress = createProgressWindow(`Editing ${modeLabel} prompt`);
      progress.set(`Running ${useStartingImage ? "vision-assisted" : "text-only"} prompt edit${useFullSceneContext ? " with scene context" : ""}...\n${gemmaRunnerLine({ vision: useStartingImage })}`, 35);
      const data = await postJson("/vrgdg/music_builder/edit_video_prompt", {
        ...textGemmaRunnerPayload(),
        model_file: useStartingImage ? i2vGemmaModelSelect.value : i2vTextGemmaModelSelect.value,
        repair_model_file: i2vTextGemmaModelSelect.value,
        mmproj_file: useStartingImage ? i2vMmprojSelect.value : "",
        current_prompt: currentPrompt,
        edit_request: editRequest,
        mode_label: modeLabel,
        performance_mode: effectiveVideoPerformanceModeForSegment(segment),
        use_vision_reference: useStartingImage,
        image_reference_path: useStartingImage ? imageReference.path : "",
        image_reference_data: useStartingImage ? imageReference.data : "",
        use_full_scene_context: useFullSceneContext,
        scene_context: useFullSceneContext ? {
          label: sceneDisplayName(segment, segmentIndexInfo(segment).index),
          image_prompt: sceneConceptPromptText(segment),
          scene_notes: String(segment.notes || "").trim(),
          director_note: String(segment.timeline_note || "").trim(),
          motion_notes: String(segment.i2v_notes || "").trim(),
          lyric_text: String(segment.lyric_text || "").trim(),
          lyric_section: String(segment.lyric_section || "").trim(),
          performance_mode: effectiveVideoPerformanceModeForSegment(segment),
          singers: segment.no_character_present ? [] : singers,
          subject_context: segment.no_character_present ? "" : segmentMappedSubjectText(segment),
          location_context: segmentMappedLocationText(segment),
          no_character_present: Boolean(segment.no_character_present),
        } : {},
        unload_after: true,
        n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
        temperature: 0.25,
        top_p: 0.9,
        max_new_tokens: 1200,
      }, GEMMA_VIDEO_ENHANCE_TIMEOUT_MS);
      const nextPrompt = String(data.prompt || "").trim();
      if (!nextPrompt) throw new Error("Gemma returned an empty edited prompt.");
      pushHistory();
      segment.i2v_prompt = nextPrompt;
      i2vPrompt.value = nextPrompt;
      editI2VPromptButton.style.display = "";
      render();
      await autoSaveSessionQuiet(`Gemma edited ${modeLabel} prompt`);
      progress.set("Edited prompt ready.", 100);
      progress.close(900);
      toast(`Gemma edited the ${modeLabel} prompt.`);
    } catch (error) {
      const debugPath = error?.gemmaDebugPath || await saveGemmaJunkDebug(error, { label: `edit ${modeLabel} prompt`, segment });
      progress?.set(`Error:\n${String(error?.message || error)}${debugPath ? `\n\nRaw Gemma output saved to:\n${debugPath}` : ""}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      editI2VPromptButton.disabled = false;
      editI2VPromptButton.textContent = "Edit Prompt";
      syncVideoModePanel();
      syncInspector();
    }
  }

  return {
    buildI2VPromptRequestForSegment, createT2IPromptWithGemma, editCurrentImagePromptWithGemma,
    editCurrentVideoPromptWithGemma, finalizeVideoPromptDraftOnly, finalizeVideoPromptForSegment,
    generateEnhancePromptWithGemma, getI2VImageReference, openBuilderInstructionEditor,
    runVideoPromptEnhancementBatch, upscaleEnhanceImage,
  };
}
