import { buildBrowserImagePrompt } from "../VRGDG_BrowserImageBridge.js";
import { postJson, queueWorkflowPrompt, waitForImages } from "./comfy_api.mjs";
import { DEFAULT_NB_IMAGE_MODEL, FLUX_GEMMA_TIMEOUT_MS } from "./constants.mjs";
import { toast } from "./controls.mjs";
import { builderImageInstructionKey, sceneLyricTextForPromptValidation } from "./image_prompts.mjs";
import { setButtonGroupState } from "./inspector.mjs";
import {
  browserImageProviderLabel,
  browserImageProviderTimeout,
  cloneFlowGptBrowserSettings,
  cloneFluxKleinSettings,
  cloneNBImageSettings,
} from "./model_settings.mjs";
import { normalizeGemmaContextLimit } from "./prompt_text.mjs";

function fluxKleinLoraPayload(settings = {}) {
  const count = Math.max(0, Math.min(4, Number(settings.lora_count || 0)));
  const useLoras = Boolean(settings.use_loras && count > 0);
  const payload = {
    use_custom_loras: useLoras,
    lora_count: useLoras ? count : 0,
  };
  for (let slot = 1; slot <= 4; slot++) {
    const config = settings.loras?.[slot - 1] || {};
    payload[`lora_${slot}`] = config.name || "[none]";
    payload[`strength_${slot}`] = Number(config.strength ?? 1);
  }
  return payload;
}

function addUniqueBrowserImageIngredient(ingredients, item) {
  if (!Array.isArray(ingredients) || !item) return ingredients;
  const path = String(item.path || "");
  const data = String(item.data || "");
  const name = String(item.name || "");
  if (!path && !data) return ingredients;
  const key = path || data || name;
  if (!ingredients.some((existing) => (existing.path || existing.data || existing.name) === key)) {
    ingredients.push({ path, data, name: name || "reference.png" });
  }
  return ingredients;
}

function browserImageReferencePrompt(prompt, settings = {}) {
  const text = String(prompt || "").trim();
  if (!text) return text;
  const context = settings.reference_context || {};
  const labels = [];
  if (context.has_subject_reference) labels.push("character reference");
  if (context.has_location_reference) labels.push("location reference");
  if (settings.previous_scene_image_attached) labels.push("last scene image reference");
  if (!labels.length && Array.isArray(settings.image_ingredients) && settings.image_ingredients.length) {
    labels.push("reference images");
  }
  if (!labels.length) return text;
  const joined = labels.length === 1
    ? labels[0]
    : labels.length === 2
      ? `${labels[0]} and ${labels[1]}`
      : `${labels.slice(0, -1).join(", ")}, and ${labels[labels.length - 1]}`;
  const continuity = settings.previous_scene_image_attached
    ? settings.previous_scene_image_purpose === "same_location_composition_diversity"
      ? " Treat the previous scene image as identity/environment continuity evidence and as a negative composition reference. The mapped location is a navigable 3D space: move the camera and character into a clearly different sub-area, change the viewing direction, shot size, subject placement, and foreground/background arrangement, and do not recreate or closely imitate the previous composition. Preserve realistic perspective, scale, floor contact, depth, shadows, reflections, color spill, and occlusion so the character is physically inside the environment rather than pasted onto it."
      : " Treat the last scene image as continuity context for what happened immediately before this scene; continue the story without simply copying the previous image unless requested."
    : "";
  const instruction = `Using the provided ${joined} as visual context, create the requested scene image.${continuity}`;
  if (text.toLowerCase().startsWith(instruction.toLowerCase())) return text;
  const opening = text.split(/[.!?\n]/, 1)[0];
  const hasReferenceOpening = /^(?:using|use|based on)\b/i.test(opening);
  const referencesAlreadyExplained = hasReferenceOpening && labels.every((label) => {
    if (label === "character reference") return /\b(?:character|subject)\s+reference\b/i.test(opening);
    if (label === "location reference") return /\blocation\s+reference\b/i.test(opening);
    // The separate continuity guidance identifies the previous scene image.
    if (label === "last scene image reference") return Boolean(continuity);
    return /\breference\s+images?\b/i.test(opening);
  });
  if (referencesAlreadyExplained) {
    const guidance = continuity.trim();
    return guidance && !text.toLowerCase().includes(guidance.toLowerCase())
      ? `${text}\n\n${guidance}`
      : text;
  }
  return `${instruction}\n\n${text}`.trim();
}

function enhancePromptForSegment(segment, { copyFallback = false } = {}) {
  const explicitEnhancePrompt = String(segment?.enhance_prompt || "").trim();
  if (explicitEnhancePrompt) return { prompt: explicitEnhancePrompt, source: "enhance" };
  const scenePrompt = String(segment?.t2i_prompt || segment?.flux_prompt || segment?.nb_prompt || segment?.flow_gpt_prompt || segment?.notes || "").trim();
  if (scenePrompt) {
    if (copyFallback && segment) segment.enhance_prompt = scenePrompt;
    return { prompt: scenePrompt, source: "scene" };
  }
  return { prompt: "", source: "" };
}

export function sceneImagePromptForEnhanceAll(segment) {
  const prompt = String(segment?.t2i_prompt || segment?.flux_prompt || segment?.nb_prompt || segment?.flow_gpt_prompt || "").trim();
  return { prompt, source: prompt ? "image_prompt" : "" };
}

export function zEnhancePayloadFromSettings(settings = {}, prompt = "", source = {}) {
  const loraCount = Math.max(0, Math.min(4, Number(settings.lora_count || 0)));
  const useLoras = Boolean(settings.use_loras && loraCount > 0);
  const payload = {
    prompt,
    source_image_path: source.path || "",
    source_image_data: source.data || "",
    source_image_name: source.name || "source.png",
    unet_name: settings.unet_name || "",
    clip_name: settings.clip_name || "",
    vae_name: settings.vae_name || "",
    width: settings.width || 1920,
    height: settings.height || 1080,
    seed: settings.seed || 1,
    seed_mode: settings.seed_mode || "fixed",
    enhance_amount: settings.enhance_amount || 8,
    use_custom_loras: useLoras,
    lora_count: useLoras ? loraCount : 0,
  };
  for (let index = 0; index < 4; index += 1) {
    const lora = settings.loras?.[index] || {};
    payload[`lora_${index + 1}`] = useLoras && index < loraCount ? (lora.name || "[none]") : "[none]";
    payload[`strength_${index + 1}`] = Number(lora.strength ?? 1);
  }
  return payload;
}

export function createImageGeneration({
  activeProjectFolderForSave, activeSegment, addSceneImageHistoryPath, advanceErnieSeedAfterRun,
  advanceKrea2TwoPassSeedAfterRun, advanceZEnhanceSeedAfterRun, advanceZImageSeedAfterRun,
  applyImageTriggerToPrompt, archiveGeneratedSceneImage, autoSaveSessionQuiet, createFluxPromptButton,
  createNBPromptButton, createProgressWindow, currentVideoMode, ensureSegmentT2IPromptHasTrigger,
  ernieCreateButtons, ernieLoraSlots, ernieSeed, firstLastFrameEndImageSource,
  flfSameLocationCameraDiversityDirection, flowGptCreateImageButton, flowGptCreatePromptButton, flowGptPrompt,
  fluxCreateButtons, fluxGemmaModelSelect, fluxMmprojSelect, fluxPrompt, fluxReferenceContextForSegment,
  gemmaRunnerLine, generateTextOnlyImagePromptFallbackForSegment, imagePromptNotesWithDirector,
  krea2TwoPassCreateButtons, krea2TwoPassSeed, mergedFluxImageIngredients, nbCreateButtons,
  nbGemmaModelSelect, nbMmprojSelect, nbPrompt, nbReferenceContextForSegment, previousAutoChainSourceSegment,
  pushHistory, render, requireActiveSegment, runClearMemoryWorkflowQuiet, runImageMemoryCleanupQuiet,
  saveErnieImageSettingsFromPanel, saveFlowGptBrowserSettingsFromPanel, saveFluxKleinSettingsFromPanel,
  saveKrea2TwoPassSettingsFromPanel, saveNBImageSettingsFromPanel, saveZEnhanceSettingsFromPanel,
  saveZImageSettingsFromPanel, sceneDisplayName, segmentImageSource, segmentIndexInfo, state, syncInspector,
  syncPreview, syncSegmentFlowGptPrompt, syncSegmentT2IPrompt, t2iTextGemmaModelSelect,
  textGemmaRunnerPayload, updateActiveFromInputs, zCreateButtons, zEnhancePromptPreview, zEnhanceSeed,
  zLoraSlots, zSeed,
}) {
  async function createZImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "ZImage", options = {}) {
    state.activeId = segment.id;
    syncInspector();
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "zimage", segment.notes || "");
    if (!prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: T2I prompt is missing.`);
    progress?.set(`${label}: preparing ZImage settings...`, percentBase);
    const zSettings = saveZImageSettingsFromPanel();
    const useLoras = Boolean(zSettings.use_loras && zSettings.lora_count > 0);
    const payload = {
      prompt,
      unet_name: zSettings.unet_name || "",
      clip_name: zSettings.clip_name || "",
      vae_name: zSettings.vae_name || "",
      first_pass_width: zSettings.first_pass_width,
      first_pass_height: zSettings.first_pass_height,
      second_pass_width: zSettings.second_pass_width,
      second_pass_height: zSettings.second_pass_height,
      seed: zSettings.seed,
      seed_mode: zSettings.seed_mode || "fixed",
      batch_size: zSettings.batch_size || 1,
      use_custom_loras: useLoras,
      lora_count: useLoras ? zSettings.lora_count : 0,
      ltx_two_pass_mode: false,
      use_image_to_image: options.bypassImageToImage === true ? false : Boolean(zSettings.use_image_to_image),
      image_to_image_start_at_step: zSettings.image_to_image_start_at_step || 5,
      image_to_image_path: zSettings.image_to_image_path || "",
      image_to_image_data: zSettings.image_to_image_data || "",
      image_to_image_name: zSettings.image_to_image_name || "",
    };
    zLoraSlots.forEach((slot, index) => {
      payload[`lora_${index + 1}`] = useLoras && index < zSettings.lora_count ? slot.picker.input.value : "[none]";
      payload[`first_pass_strength_${index + 1}`] = Number(slot.firstPassStrength.value || 0.5);
      payload[`second_pass_strength_${index + 1}`] = Number(slot.secondPassStrength.value || 1);
      payload[`strength_${index + 1}`] = Number(slot.secondPassStrength.value || 1);
    });
    progress?.set(`${label}: building hidden ZImage workflow...`, percentBase + percentSpan * 0.25);
    const built = await postJson("/vrgdg/workflow_runner/build_zimage_prompt", payload);
    if (Number.isFinite(Number(built.used_seed))) {
      zSettings.seed = Number(built.used_seed);
      zSeed.value = String(zSettings.seed);
    }
    progress?.set(`${label}: queueing ZImage workflow...`, percentBase + percentSpan * 0.45);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the preview but did not return a prompt_id.");
    const images = await waitForImages(promptId, (message) => {
      progress?.set(`${label}: ${message}\nPrompt ID: ${promptId}`, percentBase + percentSpan * 0.72);
    });
    for (const image of images) {
      await archiveGeneratedSceneImage(segment, image);
    }
    syncSegmentT2IPrompt(segment, prompt);
    segment.image = images[images.length - 1] || null;
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    syncPreview(segment);
    render();
    advanceZImageSeedAfterRun(zSettings);
    return images;
  }

  async function previewZImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const prompt = String(segment.t2i_prompt || segment.notes || "").trim();
    if (!prompt) {
      toast("Hey, you need a T2I prompt first. Create one with Gemma T2I, type one into the T2I prompt box, or add scene notes.", true);
      return;
    }
    let progress = null;
    let ranZImage = false;
    try {
      setButtonGroupState(zCreateButtons, { disabled: true, text: "Creating..." });
      progress = createProgressWindow("Creating ZImage preview");
      progress.set("Autosaving session/SRT before ZImage...", 8);
      await autoSaveSessionQuiet("ZImage preview");
      ranZImage = true;
      await createZImageForSegment(segment, progress, 15, 75, "ZImage preview");
      await autoSaveSessionQuiet("ZImage preview complete");
      await runImageMemoryCleanupQuiet(progress, "ZImage preview", 94);
      progress.set("ZImage preview ready.", 100);
      progress.close(900);
      toast("ZImage preview ready.");
    } catch (error) {
      if (ranZImage) {
        await runImageMemoryCleanupQuiet(progress, "failed ZImage preview", 100);
      }
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      setButtonGroupState(zCreateButtons, { disabled: false, text: "Create Z-Image" });
    }
  }

  async function createErnieImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "Ernie", options = {}) {
    state.activeId = segment.id;
    syncInspector();
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "ernie_image", segment.notes || "");
    if (!prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: T2I prompt is missing.`);
    progress?.set(`${label}: preparing Ernie settings...`, percentBase);
    const settings = saveErnieImageSettingsFromPanel();
    const useLoras = Boolean(settings.use_loras && settings.lora_count > 0);
    const payload = {
      prompt,
      unet_name: settings.unet_name || "",
      clip_name: settings.clip_name || "",
      vae_name: settings.vae_name || "",
      width: settings.width,
      height: settings.height,
      seed: settings.seed,
      seed_mode: settings.seed_mode || "fixed",
      batch_size: settings.batch_size || 1,
      use_custom_loras: useLoras,
      lora_count: useLoras ? settings.lora_count : 0,
      use_image_to_image: options.bypassImageToImage === true ? false : Boolean(settings.use_image_to_image),
      image_to_image_start_at_step: settings.image_to_image_start_at_step || 5,
      image_to_image_path: settings.image_to_image_path || "",
      image_to_image_data: settings.image_to_image_data || "",
      image_to_image_name: settings.image_to_image_name || "",
    };
    ernieLoraSlots.forEach((slot, index) => {
      payload[`lora_${index + 1}`] = useLoras && index < settings.lora_count ? slot.picker.input.value : "[none]";
      payload[`strength_${index + 1}`] = Number(slot.strength.value || 1);
    });
    progress?.set(`${label}: building hidden Ernie workflow...`, percentBase + percentSpan * 0.25);
    let built;
    try {
      built = await postJson("/vrgdg/workflow_runner/build_ernie_image_prompt", payload);
    } catch (error) {
      if (/\b405\b/.test(String(error?.message || error))) {
        throw new Error("Ernie backend route is not loaded yet. Fully restart ComfyUI so the new Ernie workflow route is registered, then refresh the browser.");
      }
      throw error;
    }
    if (Number.isFinite(Number(built.used_seed))) {
      settings.seed = Number(built.used_seed);
      ernieSeed.value = String(settings.seed);
    }
    progress?.set(`${label}: queueing Ernie workflow...`, percentBase + percentSpan * 0.45);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the Ernie image but did not return a prompt_id.");
    const images = await waitForImages(promptId, (message) => {
      progress?.set(`${label}: ${message}\nPrompt ID: ${promptId}`, percentBase + percentSpan * 0.72);
    });
    for (const image of images) {
      await archiveGeneratedSceneImage(segment, image);
    }
    segment.enhance_prompt = prompt;
    zEnhancePromptPreview.value = prompt;
    segment.image = images[images.length - 1] || null;
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    syncPreview(segment);
    render();
    advanceErnieSeedAfterRun(settings);
    return images;
  }

  async function previewErnieImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const prompt = String(segment.t2i_prompt || segment.notes || "").trim();
    if (!prompt) {
      toast("Hey, you need a T2I prompt first. Create one with Gemma T2I, type one into the T2I prompt box, or add scene notes.", true);
      return;
    }
    let progress = null;
    try {
      setButtonGroupState(ernieCreateButtons, { disabled: true, text: "Creating..." });
      progress = createProgressWindow("Creating Ernie image");
      progress.set("Autosaving session/SRT before Ernie...", 8);
      await autoSaveSessionQuiet("Ernie image");
      await createErnieImageForSegment(segment, progress, 15, 75, "Ernie image");
      await autoSaveSessionQuiet("Ernie image complete");
      progress.set("Ernie image ready.", 100);
      progress.close(900);
      toast("Ernie image ready.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      setButtonGroupState(ernieCreateButtons, { disabled: false, text: "Create with Ernie" });
    }
  }

  async function createKrea2TwoPassImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "Krea 2", options = {}) {
    state.activeId = segment.id;
    syncInspector();
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "krea2_2pass", segment.notes || "");
    if (!prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: T2I prompt is missing.`);
    progress?.set(`${label}: preparing Krea 2 settings...`, percentBase);
    const settings = saveKrea2TwoPassSettingsFromPanel();
    const loraCount = Math.max(0, Math.min(4, Number(settings.lora_count || 0)));
    const useLoras = Boolean(settings.use_loras && loraCount > 0);
    const payload = {
      prompt,
      unet_name: settings.unet_name || "",
      clip_name: settings.clip_name || "",
      vae_name: settings.vae_name || "",
      use_custom_loras: useLoras,
      lora_count: useLoras ? loraCount : 0,
      aspect_ratio: settings.aspect_ratio || "16:9 (Widescreen)",
      sampler_name: settings.sampler_name || "euler_ancestral_cfg_pp",
      cfg: Math.max(1, Math.min(1.2, Number(settings.cfg ?? 1.2))),
      seed: settings.seed,
      seed_mode: settings.seed_mode || "fixed",
      batch_size: settings.batch_size || 1,
      use_image_to_image: options.bypassImageToImage === true ? false : Boolean(settings.use_image_to_image),
      image_to_image_creativity: Math.max(0, Math.min(10, Number(settings.image_to_image_creativity ?? 5))),
      image_to_image_path: settings.image_to_image_path || "",
      image_to_image_data: settings.image_to_image_data || "",
      image_to_image_name: settings.image_to_image_name || "",
    };
    for (let index = 0; index < 4; index += 1) {
      const lora = settings.loras?.[index] || {};
      payload[`lora_${index + 1}`] = useLoras && index < loraCount ? (lora.name || "[none]") : "[none]";
      payload[`first_pass_strength_${index + 1}`] = Number(lora.first_pass_strength ?? lora.strength ?? 0.5);
      payload[`second_pass_strength_${index + 1}`] = Number(lora.second_pass_strength ?? lora.strength ?? 0);
      payload[`strength_${index + 1}`] = Number(lora.second_pass_strength ?? lora.strength ?? 0);
    }
    progress?.set(`${label}: building hidden Krea 2 workflow...`, percentBase + percentSpan * 0.25);
    let built;
    try {
      built = await postJson("/vrgdg/workflow_runner/build_krea2_2pass_prompt", payload);
    } catch (error) {
      if (/\b405\b/.test(String(error?.message || error))) {
        throw new Error("Krea 2 backend route is not loaded yet. Fully restart ComfyUI so the new workflow route is registered, then refresh the browser.");
      }
      throw error;
    }
    if (Number.isFinite(Number(built.used_seed))) {
      settings.seed = Number(built.used_seed);
      krea2TwoPassSeed.value = String(settings.seed);
    }
    progress?.set(`${label}: queueing Krea 2 workflow...`, percentBase + percentSpan * 0.45);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the Krea 2 image but did not return a prompt_id.");
    const images = await waitForImages(promptId, (message) => {
      progress?.set(`${label}: ${message}\nPrompt ID: ${promptId}`, percentBase + percentSpan * 0.72);
    });
    for (const image of images) {
      await archiveGeneratedSceneImage(segment, image);
    }
    segment.enhance_prompt = prompt;
    zEnhancePromptPreview.value = prompt;
    segment.image = images[images.length - 1] || null;
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    syncPreview(segment);
    render();
    advanceKrea2TwoPassSeedAfterRun(settings);
    return images;
  }

  async function previewKrea2TwoPassImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const prompt = String(segment.t2i_prompt || segment.notes || "").trim();
    if (!prompt) {
      toast("Hey, you need a T2I prompt first. Create one with Gemma T2I, type one into the T2I prompt box, or add scene notes.", true);
      return;
    }
    let progress = null;
    try {
      setButtonGroupState(krea2TwoPassCreateButtons, { disabled: true, text: "Creating..." });
      progress = createProgressWindow("Creating Krea 2 image");
      progress.set("Autosaving session/SRT before Krea 2...", 8);
      await autoSaveSessionQuiet("Krea 2 image");
      await createKrea2TwoPassImageForSegment(segment, progress, 15, 75, "Krea 2 image");
      await runClearMemoryWorkflowQuiet(progress, "Krea 2 image", 94);
      await autoSaveSessionQuiet("Krea 2 image complete");
      progress.set("Krea 2 image ready.", 100);
      progress.close(900);
      toast("Krea 2 image ready.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      setButtonGroupState(krea2TwoPassCreateButtons, { disabled: false, text: "Create with Krea 2" });
    }
  }

  async function createFluxKleinPromptWithGemma() {
    const segment = requireActiveSegment();
    if (!segment) return;
    saveFluxKleinSettingsFromPanel();
    let progress = null;
    try {
      createFluxPromptButton.disabled = true;
      createFluxPromptButton.textContent = "Gemma...";
      progress = createProgressWindow("Creating Flux/Klein prompt");
      progress.set("Autosaving session/SRT before Gemma Flux/Klein...", 8);
      await autoSaveSessionQuiet("Gemma Flux/Klein prompt");
      const data = await generateFluxKleinPromptForSegment(segment, progress, 25, "Gemma Flux/Klein", { unloadAfter: true });
      progress.set("Flux/Klein prompt ready.", 100);
      await autoSaveSessionQuiet("Gemma Flux/Klein prompt complete");
      progress.close(900);
      render();
      toast("Gemma created the Flux/Klein prompt.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      createFluxPromptButton.disabled = false;
      createFluxPromptButton.textContent = "Gemma Flux Prompt";
    }
  }

  async function generateFluxKleinPromptForSegment(segment, progress = null, percent = 25, label = "Flux/Klein Gemma", options = {}) {
    state.activeId = segment.id;
    syncInspector();
    render();
    let settings = fluxKleinSettingsForSegment(segment);
    let userNotes = imagePromptNotesWithDirector(segment, settings.notes || segment.notes || "", settings.use_director_notes);
    ({ settings, userNotes } = applyImageContinuityToPromptSettings(segment, settings, userNotes));
    if (settings.use_text_only_gemma_prompt || !Array.isArray(settings.image_ingredients) || !settings.image_ingredients.length) {
      return await generateTextOnlyImagePromptFallbackForSegment(segment, progress, percent, `${label}: text-only Gemma`, { imageMode: "flux_klein", userNotes });
    }
    progress?.set(`${label}: combining global and scene image ingredients for Gemma vision...\n${gemmaRunnerLine({ vision: true })}`, percent);
    const data = await postJson("/vrgdg/music_builder/generate_flux_klein_prompt", {
      ...textGemmaRunnerPayload(),
      model_file: fluxGemmaModelSelect.value,
      mmproj_file: fluxMmprojSelect.value,
      project_folder: activeProjectFolderForSave(),
      scene_id: segment.id || "",
      lyric_text: sceneLyricTextForPromptValidation(segment),
      builder_instruction_key: "flux_klein_t2i",
      image_ingredients: settings.image_ingredients || [],
      reference_context: settings.reference_context || {},
      repair_model_file: t2iTextGemmaModelSelect.value,
      user_notes: userNotes,
      clear_before_load: options.clearBeforeLoad !== false,
      unload_after: options.unloadAfter !== false,
      seed: options.seed,
      temperature: options.temperature,
      top_p: options.topP,
    }, FLUX_GEMMA_TIMEOUT_MS);
    pushHistory();
    syncSegmentT2IPrompt(segment, applyImageTriggerToPrompt(data.prompt, segment, "flux_klein", { validateJunk: true }));
    render();
    return data;
  }

  async function createFluxKleinImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "Flux/Klein") {
    state.activeId = segment.id;
    syncInspector();
    render();
    let settings = fluxKleinSettingsForSegment(segment);
    ({ settings } = applyImageContinuityToPromptSettings(segment, settings, ""));
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "flux_klein", settings.prompt || "");
    if (!prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: Flux/Klein prompt is missing.`);
    progress?.set(`${label}: building hidden Flux/Klein workflow...`, percentBase + percentSpan * 0.25);
    const built = await postJson("/vrgdg/workflow_runner/build_flux_klein_prompt", {
      prompt,
      image_ingredients: settings.image_ingredients || [],
      unet_name: settings.unet_name || "",
      clip_name: settings.clip_name || "",
      vae_name: settings.vae_name || "",
      width: settings.width || 1024,
      height: settings.height || 576,
      seed: settings.seed || 100,
      ...fluxKleinLoraPayload(settings),
    });
    progress?.set(`${label}: queueing Flux/Klein workflow...`, percentBase + percentSpan * 0.45);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the Flux/Klein image but did not return a prompt_id.");
    const images = await waitForImages(promptId, (message) => {
      progress?.set(`${label}: ${message}\nPrompt ID: ${promptId}`, percentBase + percentSpan * 0.72);
    });
    pushHistory();
    segment.image = images[images.length - 1] || null;
    await archiveGeneratedSceneImage(segment, segment.image);
    syncSegmentT2IPrompt(segment, prompt);
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    if (segment.id === activeSegment()?.id) {
      syncPreview(segment);
    }
    render();
    return images;
  }

  function previousBaseSceneWithImage(segment) {
    const info = segmentIndexInfo(segment);
    if (!segment || info.track === "overlay") return null;
    const base = Array.isArray(state.segments) ? state.segments : [];
    for (let index = Math.min(info.index, base.length) - 1; index >= 0; index -= 1) {
      const previous = base[index];
      if (segmentImageSource(previous)) return previous;
    }
    return null;
  }

  function previousSceneImageIngredient(segment) {
    const previous = currentVideoMode() === "flf"
      ? previousAutoChainSourceSegment(segment)
      : previousBaseSceneWithImage(segment);
    if (!previous) return null;
    const image = currentVideoMode() === "flf"
      ? firstLastFrameEndImageSource(previous)
      : segmentImageSource(previous);
    if (!image?.path && !image?.data) return null;
    const previousInfo = segmentIndexInfo(previous);
    return {
      path: String(image.path || ""),
      data: String(image.data || ""),
      name: `previous_scene_${previousInfo.index + 1}.png`,
      source_scene_label: sceneDisplayName(previous, previousInfo.index),
    };
  }

  function previousSceneStartImageIngredient(segment) {
    const previous = previousAutoChainSourceSegment(segment);
    if (!previous) return null;
    const image = segmentImageSource(previous);
    if (!image?.path && !image?.data) return null;
    const previousInfo = segmentIndexInfo(previous);
    return {
      path: String(image.path || ""),
      data: String(image.data || ""),
      name: `previous_scene_${previousInfo.index + 1}_start.png`,
      source_scene_label: sceneDisplayName(previous, previousInfo.index),
    };
  }

  async function maybeAttachPreviousSceneImage(settings, segment, options = {}) {
    const next = {
      ...settings,
      image_ingredients: Array.isArray(settings.image_ingredients) ? settings.image_ingredients.map((item) => ({ ...item })) : [],
    };
    const shouldInclude = options.includePreviousSceneImage === true || next.ask_previous_scene_image || state.imageContinuityEnabled;
    if (!shouldInclude || options.includePreviousSceneImage === false) return next;
    const suppliedPreviousImage = options.previousSceneImageIngredient && typeof options.previousSceneImageIngredient === "object"
      ? options.previousSceneImageIngredient
      : null;
    const previousImage = suppliedPreviousImage || previousSceneImageIngredient(segment);
    if (!previousImage) {
      if (currentVideoMode() === "flf" && (next.ask_previous_scene_image || options.includePreviousSceneImage === true) && previousAutoChainSourceSegment(segment)) {
        throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: previous-scene continuity is enabled, but the immediately previous scene has no assigned FLF last-frame image.`);
      }
      return next;
    }
    if (currentVideoMode() === "flf") {
      const previous = previousAutoChainSourceSegment(segment);
      const previousStart = segmentImageSource(previous) || {};
      const previousEnd = firstLastFrameEndImageSource(previous) || {};
      const endpointKeys = new Set([
        previousStart.path, previousStart.data,
        previousEnd.path, previousEnd.data,
      ].map((value) => String(value || "")).filter(Boolean));
      next.image_ingredients = next.image_ingredients.filter((item) => {
        const role = String(item?.role || item?.first_last_frame_role || "").toLowerCase();
        if (role === "first_frame" || role === "last_frame" || role === "first" || role === "last") return false;
        return ![item?.path, item?.data].some((value) => endpointKeys.has(String(value || "")));
      });
    }
    addUniqueBrowserImageIngredient(next.image_ingredients, previousImage);
    next.previous_scene_image_attached = true;
    next.previous_scene_image_label = previousImage.source_scene_label || "";
    next.previous_scene_image_purpose = String(options.previousSceneImagePurpose || "continuity");
    return next;
  }

  async function createFlowGptImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "Flow/GPT", options = {}) {
    state.activeId = segment.id;
    syncInspector();
    render();
    const settings = await maybeAttachPreviousSceneImage(flowGptBrowserSettingsForSegment(segment), segment, options);
    const savedPrompt = syncSegmentFlowGptPrompt(segment, settings.prompt || "");
    if (!savedPrompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: Flow/GPT prompt is missing.`);
    const prompt = browserImageReferencePrompt(savedPrompt, settings);
    const priorAttempt = Math.max(0, Number(segment.flow_gpt_generation_attempt || 0));
    const generationAttempt = priorAttempt + 1;
    segment.flow_gpt_generation_attempt = generationAttempt;
    const configuredTimeout = Math.max(60, Math.min(2400, Number(settings.timeout_seconds || 600)));
    const cacheBustTimeout = configuredTimeout >= 2400
      ? configuredTimeout - (generationAttempt % 2)
      : configuredTimeout + (generationAttempt % 2);
    progress?.set(`${label}: building hidden browser image workflow${settings.previous_scene_image_attached ? ` with previous scene reference (${settings.previous_scene_image_label})` : ""}...`, percentBase + percentSpan * 0.25);
    const built = await buildBrowserImagePrompt({
      provider: settings.provider,
      prompt,
      aspect_ratio: settings.aspect_ratio || "16:9",
      image_ingredients: settings.image_ingredients || [],
      timeout_seconds: cacheBustTimeout,
      reuse_open_project: options.reuseOpenProject === true,
    });
    if (settings.previous_scene_image_attached && Number(built.image_count || 0) < 1) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: previous-scene continuity was requested, but Browser AI received zero reference images.`);
    }
    progress?.set(`${label}: queueing ${built.provider_label || "browser image"} workflow...`, percentBase + percentSpan * 0.45);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the Flow/GPT image but did not return a prompt_id.");
    const images = await waitForImages(promptId, (message) => {
      progress?.set(`${label}: ${message}\nPrompt ID: ${promptId}`, percentBase + percentSpan * 0.72);
    });
    pushHistory();
    segment.image = images[images.length - 1] || null;
    await archiveGeneratedSceneImage(segment, segment.image);
    syncSegmentFlowGptPrompt(segment, savedPrompt);
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    if (segment.id === activeSegment()?.id) {
      syncPreview(segment);
    }
    render();
    return images;
  }

  async function previewFlowGptImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    const settings = saveFlowGptBrowserSettingsFromPanel();
    const prompt = syncSegmentFlowGptPrompt(segment, flowGptPrompt.value || segment.flow_gpt_prompt || segment.t2i_prompt || segment.flux_prompt || segment.nb_prompt || "");
    if (!prompt) {
      toast("Hey, you need a Flow/GPT browser prompt first. Type one into the Browser prompt box or create a normal image prompt first.", true);
      return;
    }
    let progress = null;
    try {
      flowGptCreateImageButton.disabled = true;
      flowGptCreateImageButton.textContent = "Creating...";
      progress = createProgressWindow(`Creating ${browserImageProviderLabel(settings.provider)} image`);
      progress.set("Autosaving session/SRT before Flow/GPT image...", 8);
      await autoSaveSessionQuiet("Flow/GPT image");
      await createFlowGptImageForSegment(segment, progress, 15, 75, "Flow/GPT image", {
        includePreviousSceneImage: Boolean(settings.ask_previous_scene_image),
      });
      await autoSaveSessionQuiet("Flow/GPT image complete");
      progress.set("Flow/GPT image ready.", 100);
      progress.close(900);
      toast("Flow/GPT image preview ready.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      flowGptCreateImageButton.disabled = false;
      flowGptCreateImageButton.textContent = "Create with Browser AI";
    }
  }

  async function createNBPromptWithGemma() {
    const segment = requireActiveSegment();
    if (!segment) return;
    saveNBImageSettingsFromPanel();
    let progress = null;
    try {
      createNBPromptButton.disabled = true;
      createNBPromptButton.textContent = "Gemma...";
      progress = createProgressWindow("Creating NanoBanana prompt");
      progress.set("Autosaving session/SRT before Gemma NanoBanana...", 8);
      await autoSaveSessionQuiet("Gemma NanoBanana prompt");
      const data = await generateNBPromptForSegment(segment, progress, 25, "Gemma NanoBanana", { unloadAfter: true });
      progress.set("NanoBanana prompt ready.", 100);
      await autoSaveSessionQuiet("Gemma NanoBanana prompt complete");
      progress.close(900);
      render();
      toast("Gemma created the NanoBanana prompt.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      createNBPromptButton.disabled = false;
      createNBPromptButton.textContent = "Gemma NB Prompt";
    }
  }

  async function createFlowGptPromptWithGemma() {
    const segment = requireActiveSegment();
    if (!segment) return;
    saveNBImageSettingsFromPanel();
    saveFlowGptBrowserSettingsFromPanel();
    let progress = null;
    try {
      flowGptCreatePromptButton.disabled = true;
      flowGptCreatePromptButton.textContent = "Gemma...";
      progress = createProgressWindow("Creating Browser AI prompt");
      progress.set("Autosaving session/SRT before Gemma Browser AI...", 8);
      await autoSaveSessionQuiet("Gemma Browser AI prompt");
      await generateNBPromptForSegment(segment, progress, 25, "Gemma Browser AI", { unloadAfter: true });
      const prompt = syncSegmentFlowGptPrompt(segment, segment.nb_prompt || segment.t2i_prompt || "");
      flowGptPrompt.value = prompt;
      progress.set("Browser AI prompt ready.", 100);
      await autoSaveSessionQuiet("Gemma Browser AI prompt complete");
      progress.close(900);
      render();
      toast("Gemma created the Browser AI prompt.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      flowGptCreatePromptButton.disabled = false;
      flowGptCreatePromptButton.textContent = "Gemma Browser Prompt";
    }
  }

  async function generateNBPromptForSegment(segment, progress = null, percent = 25, label = "NanoBanana Gemma", options = {}) {
    state.activeId = segment.id;
    syncInspector();
    render();
    const imageMode = options.imageMode || state.imageModelMode || "nano_banana";
    const isFlowGpt = imageMode === "flow_gpt";
    const flfImageTarget = state.videoModelMode === "flf" ? String(options.flfImageTarget || "").trim().toLowerCase() : "";
    let settings = nbImageSettingsForSegment(segment);
    let userNotes = imagePromptNotesWithDirector(segment, settings.notes || segment.notes || "", settings.use_director_notes);
    ({ settings, userNotes } = applyImageContinuityToPromptSettings(segment, settings, userNotes));
    const diversityDirection = flfImageTarget === "start" ? flfSameLocationCameraDiversityDirection(segment, "start") : "";
    if (diversityDirection) userNotes = [userNotes, diversityDirection].filter(Boolean).join("\n\n");
    if (settings.use_text_only_gemma_prompt || !Array.isArray(settings.image_ingredients) || !settings.image_ingredients.length) {
      return await generateTextOnlyImagePromptFallbackForSegment(segment, progress, percent, `${label}: text-only Gemma`, { imageMode, userNotes });
    }
    progress?.set(`${label}: creating structured ${isFlowGpt ? "Flow/GPT" : "NanoBanana"} prompt from reference images...\n${gemmaRunnerLine({ vision: true })}`, percent);
    const data = await postJson("/vrgdg/music_builder/generate_nb_image_prompt", {
      ...textGemmaRunnerPayload(),
      model_file: nbGemmaModelSelect.value || fluxGemmaModelSelect.value,
      mmproj_file: nbMmprojSelect.value || fluxMmprojSelect.value,
      lmstudio_base_url: state.lmStudioBaseUrl || "",
      lmstudio_model: state.lmStudioModel || "",
      lmstudio_api_key: state.lmStudioApiKey || "",
      project_folder: activeProjectFolderForSave(),
      scene_id: segment.id || "",
      lyric_text: sceneLyricTextForPromptValidation(segment),
      prompt_mode: imageMode,
      builder_instruction_key: builderImageInstructionKey(imageMode),
      flf_image_target: flfImageTarget,
      image_ingredients: settings.image_ingredients || [],
      reference_context: settings.reference_context || {},
      repair_model_file: t2iTextGemmaModelSelect.value,
      user_notes: userNotes,
      clear_before_load: options.clearBeforeLoad !== false,
      unload_after: options.unloadAfter !== false,
      n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
      max_new_tokens: 900,
      seed: options.seed,
      temperature: options.temperature,
      top_p: options.topP,
    }, 180000);
    pushHistory();
    if (isFlowGpt) {
      syncSegmentFlowGptPrompt(segment, data.prompt || "");
    } else {
      syncSegmentT2IPrompt(segment, applyImageTriggerToPrompt(data.prompt, segment, "nano_banana", { validateJunk: true }));
    }
    render();
    return data;
  }

  async function createNBImageForSegment(segment, progress = null, percentBase = 45, percentSpan = 35, label = "NanoBanana") {
    state.activeId = segment.id;
    syncInspector();
    render();
    let settings = nbImageSettingsForSegment(segment);
    ({ settings } = applyImageContinuityToPromptSettings(segment, settings, ""));
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "nano_banana", settings.prompt || "");
    if (!prompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: NanoBanana prompt is missing.`);
    if (!String(settings.api_key || "").trim()) throw new Error("NanoBanana API key is missing.");
    progress?.set(`${label}: building hidden NanoBanana workflow...`, percentBase + percentSpan * 0.25);
    const built = await postJson("/vrgdg/workflow_runner/build_nb_image_prompt", {
      api_key: settings.api_key || "",
      model: settings.model || DEFAULT_NB_IMAGE_MODEL,
      prompt,
      image_ingredients: settings.image_ingredients || [],
    });
    progress?.set(`${label}: queueing NanoBanana workflow...`, percentBase + percentSpan * 0.45);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the NanoBanana image but did not return a prompt_id.");
    const images = await waitForImages(promptId, (message) => {
      progress?.set(`${label}: ${message}\nPrompt ID: ${promptId}`, percentBase + percentSpan * 0.72);
    });
    pushHistory();
    segment.image = images[images.length - 1] || null;
    await archiveGeneratedSceneImage(segment, segment.image);
    syncSegmentT2IPrompt(segment, prompt);
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    if (segment.id === activeSegment()?.id) {
      syncPreview(segment);
    }
    render();
    return images;
  }

  async function previewNBImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    const settings = saveNBImageSettingsFromPanel();
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "nano_banana", settings.prompt || nbPrompt.value || "");
    if (!prompt) {
      toast("Hey, you need a NanoBanana prompt first. Click Gemma NB Prompt or type one into the NanoBanana prompt box.", true);
      return;
    }
    if (!String(settings.api_key || "").trim()) {
      toast("NanoBanana API key is missing.", true);
      return;
    }
    let progress = null;
    try {
      setButtonGroupState(nbCreateButtons, { disabled: true, text: "Creating..." });
      progress = createProgressWindow("Creating NanoBanana image");
      progress.set("Autosaving session/SRT before NanoBanana image...", 8);
      await autoSaveSessionQuiet("NanoBanana image");
      await createNBImageForSegment(segment, progress, 15, 75, "NanoBanana image");
      await autoSaveSessionQuiet("NanoBanana image complete");
      progress.set("NanoBanana image ready.", 100);
      progress.close(900);
      toast("NanoBanana image preview ready.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      setButtonGroupState(nbCreateButtons, { disabled: false, text: "Create with NanoBanana" });
    }
  }

  async function previewFluxKleinImage() {
    const segment = requireActiveSegment();
    if (!segment) return;
    const settings = saveFluxKleinSettingsFromPanel();
    const prompt = ensureSegmentT2IPromptHasTrigger(segment, "flux_klein", settings.prompt || fluxPrompt.value || "");
    if (!prompt) {
      toast("Hey, you need a Flux/Klein prompt first. Click Gemma Flux Prompt or type one into the Flux/Klein prompt box.", true);
      return;
    }
    let progress = null;
    try {
      setButtonGroupState(fluxCreateButtons, { disabled: true, text: "Creating..." });
      progress = createProgressWindow("Creating Flux/Klein image");
      progress.set("Autosaving session/SRT before Flux/Klein image...", 8);
      await autoSaveSessionQuiet("Flux/Klein image");
      progress.set("Building hidden Flux/Klein workflow...", 30);
      const built = await postJson("/vrgdg/workflow_runner/build_flux_klein_prompt", {
        prompt,
        image_ingredients: settings.image_ingredients || [],
        unet_name: settings.unet_name || "",
        clip_name: settings.clip_name || "",
        vae_name: settings.vae_name || "",
        width: settings.width || 1024,
        height: settings.height || 576,
        seed: settings.seed || 100,
        ...fluxKleinLoraPayload(settings),
      });
      progress.set("Queueing Flux/Klein workflow...", 50);
      const queued = await queueWorkflowPrompt(built.prompt);
      const promptId = queued?.prompt_id;
      if (!promptId) throw new Error("ComfyUI queued the Flux/Klein image but did not return a prompt_id.");
      progress.set(`Queued prompt ID:\n${promptId}\n\nWaiting for image...`, 65);
      const images = await waitForImages(promptId, (message) => progress.set(`${message}\nPrompt ID: ${promptId}`, 80));
      pushHistory();
      segment.image = images[images.length - 1] || null;
      await archiveGeneratedSceneImage(segment, segment.image);
      syncSegmentT2IPrompt(segment, prompt);
      segment.custom_image_path = "";
      segment.custom_image_data = "";
      segment.custom_image_name = "";
      segment.approved_image_path = "";
      segment.preview_mode = "image";
      syncPreview(segment);
      render();
      await autoSaveSessionQuiet("Flux/Klein image complete");
      progress.set("Flux/Klein preview ready.", 100);
      progress.close(900);
      toast("Flux/Klein image preview ready.");
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      setButtonGroupState(fluxCreateButtons, { disabled: false, text: "Create with Flux/Klein" });
    }
  }

  function currentEnhanceSource(segment) {
    const source = segmentImageSource(segment);
    if (source?.path || source?.data) return source;
    return null;
  }

  async function enhanceImageForSegment(segment, progress = null, percentBase = 20, percentSpan = 70, label = "Enhance", options = {}) {
    state.activeId = segment.id;
    syncInspector();
    render();
    let source = options.source || currentEnhanceSource(segment);
    if (!source?.path && !source?.data && segment.image?.filename) {
      const archived = await archiveGeneratedSceneImage(segment, segment.image);
      if (archived) source = { path: archived, name: "scene_image.png" };
    }
    if (!source?.path && !source?.data) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: selected scene image is missing.`);
    }
    const promptInfo = options.promptSource === "saved_enhance_prompt"
      ? enhancePromptForSegment(segment, { copyFallback: true })
      : sceneImagePromptForEnhanceAll(segment);
    const enhancePrompt = promptInfo.prompt;
    if (!enhancePrompt) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: Enhance/T2I prompt is missing.`);
    }
    if (source.path) addSceneImageHistoryPath(segment, source.path);
    const settings = options.settings || saveZEnhanceSettingsFromPanel();
    progress?.set(`${label}: building hidden upscale/enhance workflow...`, percentBase + percentSpan * 0.18);
    const built = await postJson("/vrgdg/workflow_runner/build_z_upscale_enhance_prompt", zEnhancePayloadFromSettings(settings, enhancePrompt, source));
    if (Number.isFinite(Number(built.used_seed))) {
      settings.seed = Number(built.used_seed);
      zEnhanceSeed.value = String(settings.seed);
    }
    progress?.set(`${label}: queueing upscale/enhance workflow...`, percentBase + percentSpan * 0.38);
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the upscale/enhance workflow but did not return a prompt_id.");
    progress?.set(`${label}: queued prompt ID:\n${promptId}\n\nWaiting for enhanced image...`, percentBase + percentSpan * 0.58);
    const images = await waitForImages(promptId, (message) => {
      progress?.set(`${label}: ${message}\nPrompt ID: ${promptId}`, percentBase + percentSpan * 0.78);
    });
    for (const image of images) {
      await archiveGeneratedSceneImage(segment, image);
    }
    segment.image = images[images.length - 1] || null;
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    if (segment.id === activeSegment()?.id) syncPreview(segment);
    render();
    advanceZEnhanceSeedAfterRun(settings);
    return images;
  }

  function fluxKleinSettingsForSegment(segment = activeSegment()) {
    const current = segment?.use_scene_flux_klein_settings ? (segment.flux_klein_settings || cloneFluxKleinSettings(state.fluxKleinSettings)) : (state.fluxKleinSettings || {});
    return {
      ...current,
      image_ingredients: mergedFluxImageIngredients(segment),
      notes: segment?.flux_notes || "",
      prompt: segment?.flux_prompt || "",
    };
  }

  function applyImageContinuityToPromptSettings(segment, settings, baseNotes) {
    if (!state.imageContinuityEnabled) return { settings, userNotes: baseNotes };
    const previous = previousAutoChainSourceSegment(segment);
    if (!previous) return { settings, userNotes: baseNotes };
    const image = segmentImageSource(previous);
    const previousPrompt = String(previous.t2i_prompt || previous.flux_prompt || previous.nb_prompt || "").trim();
    const strength = ["close", "creative"].includes(state.imageContinuityStrength) ? state.imageContinuityStrength : "balanced";
    const direction = strength === "close"
      ? "Create the immediate next visual moment with very close continuity. Preserve identity, wardrobe, location, lighting, palette, lens language, and composition; allow only modest natural progression."
      : strength === "creative"
        ? "Create a clearly connected next visual moment with imaginative progression while preserving recognizable identity, story world, style, palette, and continuity anchors."
        : "Create the next connected visual moment with balanced progression. Preserve identity, wardrobe, setting, lighting logic, palette, and style while advancing pose, action, expression, or camera framing.";
    const next = { ...settings, image_ingredients: [...(settings.image_ingredients || [])] };
    if (image?.path || image?.data) next.image_ingredients.unshift({ path: image.path || "", data: image.data || "", name: image.name || "previous_scene.png", label: "Previous scene continuity image" });
    return { settings: next, userNotes: [baseNotes, direction, previousPrompt ? `Previous scene prompt:\n${previousPrompt}` : ""].filter(Boolean).join("\n\n") };
  }

  function nbImageSettingsForSegment(segment = activeSegment()) {
    const current = segment?.use_scene_nb_image_settings
      ? (segment.nb_image_settings || cloneNBImageSettings(state.nbImageSettings))
      : (state.nbImageSettings || {});
    return {
      ...cloneNBImageSettings(current),
      image_ingredients: mergedFluxImageIngredients(segment),
      notes: segment?.nb_notes || segment?.flux_notes || segment?.notes || "",
      prompt: segment?.nb_prompt || segment?.t2i_prompt || "",
      reference_context: nbReferenceContextForSegment(segment),
    };
  }

  function flowGptBrowserSettingsForSegment(segment = activeSegment()) {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    const timeout = browserImageProviderTimeout(settings);
    return {
      ...settings,
      timeout_seconds: timeout || settings.timeout_seconds || 600,
      image_ingredients: mergedFluxImageIngredients(segment),
      prompt: segment?.flow_gpt_prompt || segment?.t2i_prompt || segment?.flux_prompt || segment?.nb_prompt || "",
      reference_context: fluxReferenceContextForSegment(segment),
    };
  }

  return {
    applyImageContinuityToPromptSettings, createErnieImageForSegment, createFlowGptImageForSegment,
    createFlowGptPromptWithGemma, createFluxKleinImageForSegment, createFluxKleinPromptWithGemma,
    createKrea2TwoPassImageForSegment, createNBImageForSegment, createNBPromptWithGemma,
    createZImageForSegment, currentEnhanceSource, enhanceImageForSegment, flowGptBrowserSettingsForSegment,
    fluxKleinSettingsForSegment, generateFluxKleinPromptForSegment, generateNBPromptForSegment,
    nbImageSettingsForSegment, previewErnieImage, previewFlowGptImage, previewFluxKleinImage,
    previewKrea2TwoPassImage, previewNBImage, previewZImage, previousSceneStartImageIngredient,
  };
}
