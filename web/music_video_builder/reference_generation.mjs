import { buildBrowserImagePrompt } from "../VRGDG_BrowserImageBridge.js";
import { postJson, queueWorkflowPrompt, waitForImages } from "./comfy_api.mjs";
import { makeButton, makeField, makeInput, makeSearchableLoraPicker, makeSelect, toast } from "./controls.mjs";
import { showGemmaBatchFailures } from "./dialogs.mjs";
import { hasReferenceImage } from "./llm_runner.mjs";
import { chooseModelValue, wireSearchablePicker } from "./model_pickers.mjs";
import {
  browserImageProviderLabel,
  browserImageProviderTimeout,
  cloneFlowGptBrowserSettings,
  cloneZImageSettings,
} from "./model_settings.mjs";
import { cloneKrea2ReferenceSettings, DEFAULT_KREA2_REFERENCE_SETTINGS } from "./models.mjs";
import { loadContextTextQuiet, lyricsFromSegmentsForPromptCreator } from "./project_files.mjs";
import { applyTriggerPhrase, isRecoverableBuildGemmaError } from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder } from "./reference_data.mjs";

function zimageReferencePayload(prompt, settings) {
  const zSettings = cloneZImageSettings(settings);
  const useLoras = Boolean(zSettings.use_loras && zSettings.lora_count > 0);
  const trigger = zSettings.image_trigger_phrase || "";
  const payload = {
    prompt: applyTriggerPhrase(prompt, trigger, { validateJunk: true }),
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
    use_image_to_image: false,
    image_to_image_start_at_step: zSettings.image_to_image_start_at_step || 5,
    image_to_image_path: "",
    image_to_image_data: "",
    image_to_image_name: "",
  };
  for (let index = 0; index < 4; index += 1) {
    const lora = zSettings.loras?.[index] || {};
    payload[`lora_${index + 1}`] = useLoras && index < zSettings.lora_count ? (lora.name || "[none]") : "[none]";
    payload[`first_pass_strength_${index + 1}`] = Number(lora.first_pass_strength ?? lora.strength ?? 0.5);
    payload[`second_pass_strength_${index + 1}`] = Number(lora.second_pass_strength ?? lora.strength ?? 1);
    payload[`strength_${index + 1}`] = Number(lora.second_pass_strength ?? lora.strength ?? 1);
  }
  return payload;
}

function referenceGeneratorPickerOptions(options = [], defaults = []) {
  const merged = [...(Array.isArray(defaults) ? defaults : [defaults]), ...(options || [])];
  return Array.from(new Set(merged.map((item) => String(item || "").trim()).filter(Boolean)));
}

function setReferenceGeneratorOptions(picker, options = [], preferred = "") {
  const preferredList = Array.isArray(preferred) ? preferred : [preferred];
  picker.options = referenceGeneratorPickerOptions(options, preferredList);
  picker.input.value = chooseModelValue(picker.options, picker.input.value, preferredList) || picker.input.value || preferredList[0] || "";
  wireSearchablePicker(picker);
}

export function createReferenceGeneration({
  activeSegment, advanceZImageSeedAfterRun, autoMapLocations, autoSaveSessionQuiet,
  createAllMissingLocationZImages, createAllMissingSubjectImages, createProgressWindow,
  describeReferenceImageWithGemma, ensureSubjectCount, extractLocations, extractSubjects, gemmaRunnerLine,
  i2vTextGemmaModelSelect, inlineProgress, inlineProgressBar, inlineProgressText, keepGemmaLoadedForLocations,
  locationScoutLyricsPayloadForGpt, projectInput, refs, renderAll, renderFluxIngredientList,
  renderNBIngredientList, runImageMemoryCleanupQuiet, saveSession, saveZImageSettingsFromPanel, state,
  subjectDescription, subjectNameInput, subjectTypeSelect, syncZImageSettingsPanel, t2iTextGemmaModelSelect,
  textGemmaRunnerPayload, themeStyleInput, useLocations, useSubject, zClipPicker, zSeed, zUnetPicker,
  zVaePicker,
}) {
  function setInlineProgress(message, percent = 8) {
    inlineProgress.style.display = "flex";
    inlineProgressText.textContent = message || "Working...";
    inlineProgressBar.style.width = `${Math.max(0, Math.min(100, Number(percent) || 0))}%`;
  }
  function hideInlineProgress(delay = 1200) {
    setTimeout(() => {
      inlineProgress.style.display = "none";
      inlineProgressBar.style.width = "0%";
    }, delay);
  }

  function currentZImageReferenceSettings() {
    const settings = cloneZImageSettings(saveZImageSettingsFromPanel());
    settings.use_image_to_image = false;
    settings.image_to_image_path = "";
    settings.image_to_image_data = "";
    settings.image_to_image_name = "";
    return settings;
  }

  const krea2ReferencePayload = (prompt, settings = {}) => ({
    prompt: String(prompt || "").trim(),
    krea_unet_name: settings.krea_unet_name || "krea2_turbo_fp8_scaled.safetensors",
    krea_clip_name: settings.krea_clip_name || "qwen3vl_4b_fp8_scaled.safetensors",
    krea_vae_name: settings.krea_vae_name || "qwen_image_vae.safetensors",
    z_unet_name: settings.z_unet_name || "z_image_turbo_bf16.safetensors",
    z_clip_name: settings.z_clip_name || "qwen_3_4b.safetensors",
    z_vae_name: settings.z_vae_name || "ae.safetensors",
    first_pass_width: Number(settings.first_pass_width || 1024),
    first_pass_height: Number(settings.first_pass_height || 576),
    width: Number(settings.width || 1920),
    height: Number(settings.height || 1080),
    seed: Number(settings.seed || 1),
    seed_mode: settings.seed_mode || "fixed",
    batch_size: 1,
  });

  function currentKrea2ReferenceSettings() {
    state.referenceKrea2Settings = cloneKrea2ReferenceSettings(state.referenceKrea2Settings);
    return state.referenceKrea2Settings;
  }

  function rememberKrea2ReferenceSettings(settings = {}) {
    state.referenceKrea2Settings = cloneKrea2ReferenceSettings(settings);
    return state.referenceKrea2Settings;
  }

  function buildReferenceGeneratorControls() {
    const zSettings = currentZImageReferenceSettings();
    const kreaSettings = currentKrea2ReferenceSettings();
    const zModel = makeSearchableLoraPicker(kreaSettings.z_unet_name || zSettings.unet_name || DEFAULT_KREA2_REFERENCE_SETTINGS.z_unet_name);
    const zClip = makeSearchableLoraPicker(kreaSettings.z_clip_name || zSettings.clip_name || DEFAULT_KREA2_REFERENCE_SETTINGS.z_clip_name);
    const zVae = makeSearchableLoraPicker(kreaSettings.z_vae_name || zSettings.vae_name || DEFAULT_KREA2_REFERENCE_SETTINGS.z_vae_name);
    const kreaModel = makeSearchableLoraPicker(kreaSettings.krea_unet_name || DEFAULT_KREA2_REFERENCE_SETTINGS.krea_unet_name);
    const kreaClip = makeSearchableLoraPicker(kreaSettings.krea_clip_name || DEFAULT_KREA2_REFERENCE_SETTINGS.krea_clip_name);
    const kreaVae = makeSearchableLoraPicker(kreaSettings.krea_vae_name || DEFAULT_KREA2_REFERENCE_SETTINGS.krea_vae_name);

    setReferenceGeneratorOptions(zModel, zUnetPicker.options || [], DEFAULT_KREA2_REFERENCE_SETTINGS.z_unet_name);
    setReferenceGeneratorOptions(zClip, zClipPicker.options || [], DEFAULT_KREA2_REFERENCE_SETTINGS.z_clip_name);
    setReferenceGeneratorOptions(zVae, zVaePicker.options || [], DEFAULT_KREA2_REFERENCE_SETTINGS.z_vae_name);
    setReferenceGeneratorOptions(kreaModel, zUnetPicker.options || [], DEFAULT_KREA2_REFERENCE_SETTINGS.krea_unet_name);
    setReferenceGeneratorOptions(kreaClip, zClipPicker.options || [], DEFAULT_KREA2_REFERENCE_SETTINGS.krea_clip_name);
    setReferenceGeneratorOptions(kreaVae, zVaePicker.options || [], DEFAULT_KREA2_REFERENCE_SETTINGS.krea_vae_name);

    const seed = makeInput(String(kreaSettings.seed || zSettings.seed || DEFAULT_KREA2_REFERENCE_SETTINGS.seed), "number");
    seed.min = "0";
    seed.step = "1";
    const seedMode = makeSelect(["fixed", "random"], kreaSettings.seed_mode || zSettings.seed_mode || DEFAULT_KREA2_REFERENCE_SETTINGS.seed_mode);
    const firstWidth = makeInput(String(kreaSettings.first_pass_width || DEFAULT_KREA2_REFERENCE_SETTINGS.first_pass_width), "number");
    const firstHeight = makeInput(String(kreaSettings.first_pass_height || DEFAULT_KREA2_REFERENCE_SETTINGS.first_pass_height), "number");
    const width = makeInput(String(kreaSettings.width || zSettings.second_pass_width || DEFAULT_KREA2_REFERENCE_SETTINGS.width), "number");
    const height = makeInput(String(kreaSettings.height || zSettings.second_pass_height || DEFAULT_KREA2_REFERENCE_SETTINGS.height), "number");
    for (const input of [firstWidth, firstHeight, width, height]) {
      input.min = "64";
      input.step = "8";
    }

    const readZImage = () => ({
      unet_name: zModel.input.value,
      clip_name: zClip.input.value,
      vae_name: zVae.input.value,
    });
    const readKrea2 = () => rememberKrea2ReferenceSettings({
      krea_unet_name: kreaModel.input.value,
      krea_clip_name: kreaClip.input.value,
      krea_vae_name: kreaVae.input.value,
      z_unet_name: zModel.input.value,
      z_clip_name: zClip.input.value,
      z_vae_name: zVae.input.value,
      first_pass_width: Number(firstWidth.value || DEFAULT_KREA2_REFERENCE_SETTINGS.first_pass_width),
      first_pass_height: Number(firstHeight.value || DEFAULT_KREA2_REFERENCE_SETTINGS.first_pass_height),
      width: Number(width.value || DEFAULT_KREA2_REFERENCE_SETTINGS.width),
      height: Number(height.value || DEFAULT_KREA2_REFERENCE_SETTINGS.height),
      seed: Number(seed.value || DEFAULT_KREA2_REFERENCE_SETTINGS.seed),
      seed_mode: seedMode.value || DEFAULT_KREA2_REFERENCE_SETTINGS.seed_mode,
    });

    return { zModel, zClip, zVae, kreaModel, kreaClip, kreaVae, seed, seedMode, firstWidth, firstHeight, width, height, readZImage, readKrea2 };
  }

  function applyReferenceDescription(referenceType, target, description = "") {
    description = String(description || "").trim();
    if (!description || !target || typeof target !== "object") return;
    target.description = description;
    if (referenceType === "subject") {
      const matchingSubject = refs.subjects.find((subject) => subject === target || (subject.id && subject.id === target.id));
      if (matchingSubject) matchingSubject.description = description;
      if (refs.subject_count === 1 && (!matchingSubject || refs.subjects[0]?.id === matchingSubject.id || target === refs.subject)) {
        refs.subject.description = description;
        if (refs.subjects[0]) refs.subjects[0].description = description;
        subjectDescription.value = description;
      }
    }
  }

  async function describeGeneratedReference(referenceType, target, prompt = "") {
    if (referenceType === "subject") {
      setInlineProgress(`Vision Gemma describing the generated subject image...\n${gemmaRunnerLine({ vision: true })}`, 91);
      const description = await describeReferenceImageWithGemma(target, "subject", {
        unloadAfter: true,
        clearBeforeLoad: false,
      });
      applyReferenceDescription("subject", target, description);
      return description;
    }
    applyReferenceDescription(referenceType, target, prompt);
    return String(prompt || "").trim();
  }

  async function persistGeneratedReferenceImage(referenceType, target) {
    state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
    renderFluxIngredientList(activeSegment());
    renderNBIngredientList(activeSegment());
    try {
      await saveSession({ quiet: true, throwOnError: true });
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Generated ${referenceType} reference image was saved, but the Reference Builder session autosave failed:`, error);
      toast(`The generated image file was saved, but the Reference Builder list could not be autosaved.\nPlease click Save Reference Builder before closing.\n${String(error?.message || error)}`, true);
    }
    return target;
  }

  async function runFluxReferenceWithZImage(referenceType, target, sourceText, name = "", generatorSettings = null) {
    const text = String(sourceText || "").trim();
    if (!text) {
      toast(referenceType === "subject" ? "Enter a subject description first." : "Enter a location description first.", true);
      return;
    }
    const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
    if (!modelFile && !["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner)) {
      toast("Choose a non-vision Gemma model first.", true);
      return;
    }
    let progress = null;
    let ranZImage = false;
    try {
      setInlineProgress(referenceType === "subject" ? "Creating subject prompt with Gemma..." : "Creating location prompt with Gemma...", 8);
      progress = createProgressWindow(referenceType === "subject" ? "Creating subject reference" : "Creating location reference", { zIndex: 100008 });
      progress.set(`Creating ZImage prompt with Gemma...\n${gemmaRunnerLine()}`, 8);
      const styleTheme = state.useVrgdgTextContext ? await loadContextTextQuiet(themeStyleInput.value) : "";
      const promptData = await postJson("/vrgdg/music_builder/flux_reference_zimage_prompt", {
        ...textGemmaRunnerPayload(),
        model_file: modelFile,
        reference_type: referenceType,
        source_text: text,
        style_theme: styleTheme,
        unload_after: true,
      }, 3 * 60 * 1000);
      const zSettings = generatorSettings?.zimage ? cloneZImageSettings({ ...currentZImageReferenceSettings(), ...generatorSettings.zimage }) : currentZImageReferenceSettings();
      setInlineProgress("Building ZImage reference workflow...", 28);
      progress.set("Building ZImage reference workflow...", 28);
      const built = await postJson("/vrgdg/workflow_runner/build_zimage_prompt", zimageReferencePayload(promptData.prompt, zSettings));
      if (Number.isFinite(Number(built.used_seed))) {
        zSettings.seed = Number(built.used_seed);
        state.zimageSettings = zSettings;
        zSeed.value = String(zSettings.seed);
      }
      setInlineProgress("Queueing ZImage reference workflow...", 42);
      progress.set("Queueing ZImage reference workflow...", 42);
      const queued = await queueWorkflowPrompt(built.prompt);
      const promptId = queued?.prompt_id;
      if (!promptId) throw new Error("ComfyUI queued the ZImage reference but did not return a prompt_id.");
      ranZImage = true;
      const images = await waitForImages(promptId, (message) => {
        setInlineProgress(`${message}\nPrompt ID: ${promptId}`, 66);
        progress?.set(`${message}\nPrompt ID: ${promptId}`, 66);
      });
      const image = images[images.length - 1];
      if (!image) throw new Error("ZImage did not return a reference image.");
      setInlineProgress("Saving reference image into the project...", 88);
      progress.set("Saving reference image into the project...", 88);
      const saved = await postJson("/vrgdg/music_builder/save_flux_reference_image", {
        project_folder: projectInput.value || state.projectFolder,
        reference_type: referenceType,
        name: name || text.slice(0, 48) || referenceType,
        image,
      });
      target.image.path = saved.saved_path || "";
      target.image.data = "";
      target.image.name = `${name || referenceType}.png`;
      await describeGeneratedReference(referenceType, target, promptData.prompt);
      await persistGeneratedReferenceImage(referenceType, target);
      advanceZImageSeedAfterRun(zSettings);
      syncZImageSettingsPanel();
      renderAll();
      setInlineProgress("Cleaning memory after ZImage reference...", 94);
      await runImageMemoryCleanupQuiet(progress, "ZImage reference", 94);
      setInlineProgress("Reference image ready.", 100);
      progress.set("Reference image ready.", 100);
      progress.close(1300);
      hideInlineProgress();
      toast(referenceType === "subject" ? "Subject reference created with ZImage." : "Location reference created with ZImage.");
    } catch (error) {
      if (ranZImage) {
        setInlineProgress("Cleaning memory after failed ZImage reference...", 100);
        await runImageMemoryCleanupQuiet(progress, "failed ZImage reference", 100);
      }
      setInlineProgress(`Error:\n${String(error?.message || error)}`, 100);
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    }
  }

  async function runFluxReferenceWithFlowGpt(referenceType, target, sourceText, name = "") {
    const text = String(sourceText || "").trim();
    if (!text) {
      toast(referenceType === "subject" ? "Enter a subject description first." : "Enter a location description first.", true);
      return;
    }
    const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
    if (!modelFile && !["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner)) {
      toast("Choose a non-vision Gemma model first.", true);
      return;
    }
    let progress = null;
    try {
      const browserSettings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
      const providerLabel = browserImageProviderLabel(browserSettings.provider);
      setInlineProgress(referenceType === "subject" ? "Creating subject prompt with Gemma..." : "Creating location prompt with Gemma...", 8);
      progress = createProgressWindow(referenceType === "subject" ? "Creating subject reference" : "Creating location reference", { zIndex: 100008 });
      progress.set(`Creating ${providerLabel} reference prompt with Gemma...\n${gemmaRunnerLine()}`, 8);
      const styleTheme = state.useVrgdgTextContext ? await loadContextTextQuiet(themeStyleInput.value) : "";
      const promptData = await postJson("/vrgdg/music_builder/flux_reference_zimage_prompt", {
        ...textGemmaRunnerPayload(),
        model_file: modelFile,
        reference_type: referenceType,
        source_text: text,
        style_theme: styleTheme,
        unload_after: true,
      }, 3 * 60 * 1000);
      const timeout = browserImageProviderTimeout(browserSettings);
      setInlineProgress(`Building ${providerLabel} browser workflow...`, 28);
      progress.set(`Building ${providerLabel} browser workflow...`, 28);
      const built = await buildBrowserImagePrompt({
        provider: browserSettings.provider,
        prompt: promptData.prompt,
        aspect_ratio: browserSettings.aspect_ratio || "16:9",
        image_ingredients: [],
        timeout_seconds: timeout || browserSettings.timeout_seconds || 600,
      });
      setInlineProgress(`Queueing ${built.provider_label || providerLabel} reference workflow...`, 42);
      progress.set(`Queueing ${built.provider_label || providerLabel} reference workflow...`, 42);
      const queued = await queueWorkflowPrompt(built.prompt);
      const promptId = queued?.prompt_id;
      if (!promptId) throw new Error(`ComfyUI queued the ${providerLabel} reference but did not return a prompt_id.`);
      const images = await waitForImages(promptId, (message) => {
        setInlineProgress(`${message}\nPrompt ID: ${promptId}`, 66);
        progress?.set(`${message}\nPrompt ID: ${promptId}`, 66);
      });
      const image = images[images.length - 1];
      if (!image) throw new Error(`${providerLabel} did not return a reference image.`);
      setInlineProgress("Saving reference image into the project...", 88);
      progress.set("Saving reference image into the project...", 88);
      const saved = await postJson("/vrgdg/music_builder/save_flux_reference_image", {
        project_folder: projectInput.value || state.projectFolder,
        reference_type: referenceType,
        name: name || text.slice(0, 48) || referenceType,
        image,
      });
      if (!target.image) target.image = { path: "", data: "", name: "" };
      target.image.path = saved.saved_path || "";
      target.image.data = "";
      target.image.name = `${name || referenceType}.png`;
      await describeGeneratedReference(referenceType, target, promptData.prompt);
      await persistGeneratedReferenceImage(referenceType, target);
      renderAll();
      setInlineProgress("Reference image ready.", 100);
      progress.set("Reference image ready.", 100);
      progress.close(1300);
      hideInlineProgress();
      toast(referenceType === "subject" ? `Subject reference created with ${providerLabel}.` : `Location reference created with ${providerLabel}.`);
    } catch (error) {
      setInlineProgress(`Error:\n${String(error?.message || error)}`, 100);
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    }
  }

  async function createMissingLocationReferencesWithZImage(workflow = "zimage", generatorSettings = {}) {
    const useKrea2 = workflow === "krea2";
    const useFlowGpt = workflow === "flow_gpt";
    const browserSettings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    const workflowLabel = useFlowGpt
      ? browserImageProviderLabel(browserSettings.provider)
      : useKrea2 ? "Krea2 + ZImage enhancer" : "ZImage";
    const missingLocations = refs.locations
      .map((location, index) => ({ location, index }))
      .filter(({ location }) => {
        const image = location?.image || {};
        return String(location?.name || location?.description || "").trim()
          && !String(image.path || image.data || "").trim();
      });
    if (!missingLocations.length) {
      toast("All listed locations already have images, or no usable locations were found.");
      return;
    }
    const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
    if (!modelFile && !["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner)) {
      toast("Choose a non-vision Gemma model first.", true);
      return;
    }
    const progress = createProgressWindow("Creating location references", { zIndex: 100008 });
    createAllMissingLocationZImages.disabled = true;
    extractLocations.disabled = true;
    autoMapLocations.disabled = true;
    createAllMissingLocationZImages.textContent = "Creating...";
    const keepGemmaLoaded = Boolean(keepGemmaLoadedForLocations.input.checked);
    let ranWorkflow = false;
    try {
      const styleTheme = state.useVrgdgTextContext ? await loadContextTextQuiet(themeStyleInput.value) : "";
      const promptJobs = [];
      for (let index = 0; index < missingLocations.length; index += 1) {
        const { location } = missingLocations[index];
        const sourceText = `${location.name || ""}\n${location.description || ""}`.trim();
        const isLastPrompt = index === missingLocations.length - 1;
        const percent = 6 + Math.round((index / Math.max(1, missingLocations.length)) * 34);
        const message = `Creating location prompts with Gemma...\n${index + 1}/${missingLocations.length}: ${location.name || `Location ${index + 1}`}\n${gemmaRunnerLine()}`;
        setInlineProgress(message, percent);
        progress.set(message, percent);
        const promptData = await postJson("/vrgdg/music_builder/flux_reference_zimage_prompt", {
          ...textGemmaRunnerPayload(),
          model_file: modelFile,
          reference_type: "location",
          source_text: sourceText,
          style_theme: styleTheme,
          unload_after: keepGemmaLoaded ? isLastPrompt : true,
        }, 3 * 60 * 1000);
        promptJobs.push({ location, prompt: promptData.prompt });
      }

      let zSettings = generatorSettings?.zimage ? cloneZImageSettings({ ...currentZImageReferenceSettings(), ...generatorSettings.zimage }) : currentZImageReferenceSettings();
      const kreaSettings = generatorSettings?.krea2 || {};
      for (let index = 0; index < promptJobs.length; index += 1) {
        const { location, prompt } = promptJobs[index];
        const label = location.name || `Location ${index + 1}`;
        const basePercent = 42 + Math.round((index / Math.max(1, promptJobs.length)) * 50);
        setInlineProgress(`Building ${workflowLabel} location workflow...\n${index + 1}/${promptJobs.length}: ${label}`, basePercent);
        progress.set(`Building ${workflowLabel} location workflow...\n${index + 1}/${promptJobs.length}: ${label}`, basePercent);
        const built = useFlowGpt
          ? await buildBrowserImagePrompt({
            provider: browserSettings.provider,
            prompt,
            aspect_ratio: browserSettings.aspect_ratio || "16:9",
            image_ingredients: [],
            timeout_seconds: browserImageProviderTimeout(browserSettings),
          })
          : useKrea2
            ? await postJson("/vrgdg/workflow_runner/build_krea2_prompt", krea2ReferencePayload(prompt, kreaSettings))
            : await postJson("/vrgdg/workflow_runner/build_zimage_prompt", zimageReferencePayload(prompt, zSettings));
        if (!useFlowGpt && !useKrea2 && Number.isFinite(Number(built.used_seed))) {
          zSettings.seed = Number(built.used_seed);
          state.zimageSettings = zSettings;
          zSeed.value = String(zSettings.seed);
        }
        setInlineProgress(`Queueing ${workflowLabel} location workflow...\n${index + 1}/${promptJobs.length}: ${label}`, basePercent + 3);
        progress.set(`Queueing ${workflowLabel} location workflow...\n${index + 1}/${promptJobs.length}: ${label}`, basePercent + 3);
        const queued = await queueWorkflowPrompt(built.prompt);
        const promptId = queued?.prompt_id;
        if (!promptId) throw new Error(`ComfyUI queued ${label} but did not return a prompt_id.`);
        ranWorkflow = true;
        const images = await waitForImages(promptId, (message) => {
          setInlineProgress(`${index + 1}/${promptJobs.length}: ${label}\n${message}\nPrompt ID: ${promptId}`, basePercent + 6);
          progress.set(`${index + 1}/${promptJobs.length}: ${label}\n${message}\nPrompt ID: ${promptId}`, basePercent + 6);
        });
        const image = images[images.length - 1];
        if (!image) throw new Error(`${workflowLabel} did not return a reference image for ${label}.`);
        const saved = await postJson("/vrgdg/music_builder/save_flux_reference_image", {
          project_folder: projectInput.value || state.projectFolder,
          reference_type: "location",
          name: location.name || `location_${index + 1}`,
          image,
        });
        if (!location.image) location.image = { path: "", data: "", name: "" };
        location.image.path = saved.saved_path || "";
        location.image.data = "";
        location.image.name = `${location.name || `location_${index + 1}`}.png`;
        await describeGeneratedReference("location", location, prompt);
        await persistGeneratedReferenceImage("location", location);
        if (!useFlowGpt && !useKrea2) {
          advanceZImageSeedAfterRun(zSettings);
          zSettings = cloneZImageSettings(state.zimageSettings);
          syncZImageSettingsPanel();
        }
        renderAll();
      }
      refs.use_location_references = true;
      useLocations.input.checked = true;
      if (!useFlowGpt) {
        setInlineProgress("Cleaning memory after location references...", 94);
        await runImageMemoryCleanupQuiet(progress, "location references", 94);
      }
      setInlineProgress("Location reference images ready.", 100);
      progress.set(`Created ${promptJobs.length} location reference image${promptJobs.length === 1 ? "" : "s"}.`, 100);
      progress.close(1800);
      hideInlineProgress();
      toast(`Created ${promptJobs.length} location reference image${promptJobs.length === 1 ? "" : "s"} with ${workflowLabel}.`);
    } catch (error) {
      if (ranWorkflow && !useFlowGpt) {
        setInlineProgress("Cleaning memory after failed location reference batch...", 100);
        await runImageMemoryCleanupQuiet(progress, "failed location references", 100);
      }
      setInlineProgress(`Error:\n${String(error?.message || error)}`, 100);
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      createAllMissingLocationZImages.disabled = false;
      extractLocations.disabled = false;
      autoMapLocations.disabled = false;
      createAllMissingLocationZImages.textContent = "Generate Missing Images";
    }
  }

  async function createMissingSubjectReferencesWithImageWorkflow(workflow = "zimage", generatorSettings = {}) {
    ensureSubjectCount();
    if (refs.subject_count === 1 && refs.subjects[0]) {
      refs.subjects[0].name = subjectNameInput.value || refs.subjects[0].name || refs.subject.name || "Subject";
      refs.subjects[0].reference_type = subjectTypeSelect.value || refs.subjects[0].reference_type || refs.subject.reference_type || "character";
      refs.subjects[0].description = subjectDescription.value || refs.subjects[0].description || refs.subject.description || "";
      refs.subject = { ...refs.subject, ...refs.subjects[0], image: refs.subjects[0].image || refs.subject.image || { path: "", data: "", name: "" } };
    }
    const useKrea2 = workflow === "krea2";
    const useFlowGpt = workflow === "flow_gpt";
    const browserSettings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    const workflowLabel = useFlowGpt
      ? browserImageProviderLabel(browserSettings.provider)
      : useKrea2 ? "Krea2 + ZImage enhancer" : "ZImage";
    const missingSubjects = refs.subjects
      .map((subject, index) => ({ subject, index }))
      .filter(({ subject }) => {
        const image = subject?.image || {};
        return String(subject?.name || subject?.description || "").trim()
          && !String(image.path || image.data || "").trim();
      });
    if (!missingSubjects.length) {
      toast("All listed subjects already have images, or no usable subjects were found.");
      return;
    }
    const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
    if (!modelFile && !["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner)) {
      toast("Choose a non-vision Gemma model first.", true);
      return;
    }
    const progress = createProgressWindow("Creating subject references", { zIndex: 100008 });
    createAllMissingSubjectImages.disabled = true;
    extractSubjects.disabled = true;
    createAllMissingSubjectImages.textContent = "Creating...";
    let ranWorkflow = false;
    try {
      const styleTheme = state.useVrgdgTextContext ? await loadContextTextQuiet(themeStyleInput.value) : "";
      const promptJobs = [];
      for (let index = 0; index < missingSubjects.length; index += 1) {
        const { subject } = missingSubjects[index];
        const sourceText = `${subject.name || ""}\n${subject.reference_type ? `Reference type: ${subject.reference_type}` : ""}\n${subject.description || ""}`.trim();
        const percent = 6 + Math.round((index / Math.max(1, missingSubjects.length)) * 34);
        const message = `Creating subject prompts with Gemma...\n${index + 1}/${missingSubjects.length}: ${subject.name || `Subject ${index + 1}`}\n${gemmaRunnerLine()}`;
        setInlineProgress(message, percent);
        progress.set(message, percent);
        const promptData = await postJson("/vrgdg/music_builder/flux_reference_zimage_prompt", {
          ...textGemmaRunnerPayload(),
          model_file: modelFile,
          reference_type: "subject",
          source_text: sourceText,
          style_theme: styleTheme,
          unload_after: index === missingSubjects.length - 1,
        }, 3 * 60 * 1000);
        promptJobs.push({ subject, prompt: promptData.prompt });
      }

      let zSettings = generatorSettings?.zimage ? cloneZImageSettings({ ...currentZImageReferenceSettings(), ...generatorSettings.zimage }) : currentZImageReferenceSettings();
      const kreaSettings = generatorSettings?.krea2 || {};
      for (let index = 0; index < promptJobs.length; index += 1) {
        const { subject, prompt } = promptJobs[index];
        const label = subject.name || `Subject ${index + 1}`;
        const basePercent = 42 + Math.round((index / Math.max(1, promptJobs.length)) * 50);
        setInlineProgress(`Building ${workflowLabel} subject workflow...\n${index + 1}/${promptJobs.length}: ${label}`, basePercent);
        progress.set(`Building ${workflowLabel} subject workflow...\n${index + 1}/${promptJobs.length}: ${label}`, basePercent);
        const built = useFlowGpt
          ? await buildBrowserImagePrompt({
            provider: browserSettings.provider,
            prompt,
            aspect_ratio: browserSettings.aspect_ratio || "16:9",
            image_ingredients: [],
            timeout_seconds: browserImageProviderTimeout(browserSettings),
          })
          : useKrea2
            ? await postJson("/vrgdg/workflow_runner/build_krea2_prompt", krea2ReferencePayload(prompt, kreaSettings))
            : await postJson("/vrgdg/workflow_runner/build_zimage_prompt", zimageReferencePayload(prompt, zSettings));
        if (!useFlowGpt && !useKrea2 && Number.isFinite(Number(built.used_seed))) {
          zSettings.seed = Number(built.used_seed);
          state.zimageSettings = zSettings;
          zSeed.value = String(zSettings.seed);
        }
        setInlineProgress(`Queueing ${workflowLabel} subject workflow...\n${index + 1}/${promptJobs.length}: ${label}`, basePercent + 3);
        progress.set(`Queueing ${workflowLabel} subject workflow...\n${index + 1}/${promptJobs.length}: ${label}`, basePercent + 3);
        const queued = await queueWorkflowPrompt(built.prompt);
        const promptId = queued?.prompt_id;
        if (!promptId) throw new Error(`ComfyUI queued ${label} but did not return a prompt_id.`);
        ranWorkflow = true;
        const images = await waitForImages(promptId, (message) => {
          setInlineProgress(`${index + 1}/${promptJobs.length}: ${label}\n${message}\nPrompt ID: ${promptId}`, basePercent + 6);
          progress.set(`${index + 1}/${promptJobs.length}: ${label}\n${message}\nPrompt ID: ${promptId}`, basePercent + 6);
        });
        const image = images[images.length - 1];
        if (!image) throw new Error(`${workflowLabel} did not return a reference image for ${label}.`);
        const saved = await postJson("/vrgdg/music_builder/save_flux_reference_image", {
          project_folder: projectInput.value || state.projectFolder,
          reference_type: "subject",
          name: subject.name || `subject_${index + 1}`,
          image,
        });
        if (!subject.image) subject.image = { path: "", data: "", name: "" };
        subject.image.path = saved.saved_path || "";
        subject.image.data = "";
        subject.image.name = `${subject.name || `subject_${index + 1}`}.png`;
        await describeGeneratedReference("subject", subject, prompt);
        await persistGeneratedReferenceImage("subject", subject);
        if (refs.subject_count === 1 && refs.subjects[0]?.id === subject.id) {
          refs.subject.image = subject.image;
        }
        if (!useFlowGpt && !useKrea2) {
          advanceZImageSeedAfterRun(zSettings);
          zSettings = cloneZImageSettings(state.zimageSettings);
          syncZImageSettingsPanel();
        }
        renderAll();
      }
      refs.use_subject_reference = true;
      useSubject.input.checked = true;
      if (!useFlowGpt) {
        setInlineProgress("Cleaning memory after subject references...", 94);
        await runImageMemoryCleanupQuiet(progress, "subject references", 94);
      }
      setInlineProgress("Subject reference images ready.", 100);
      progress.set(`Created ${promptJobs.length} subject reference image${promptJobs.length === 1 ? "" : "s"}.`, 100);
      progress.close(1800);
      hideInlineProgress();
      toast(`Created ${promptJobs.length} subject reference image${promptJobs.length === 1 ? "" : "s"} with ${workflowLabel}.`);
    } catch (error) {
      if (ranWorkflow && !useFlowGpt) {
        setInlineProgress("Cleaning memory after failed subject reference batch...", 100);
        await runImageMemoryCleanupQuiet(progress, "failed subject references", 100);
      }
      setInlineProgress(`Error:\n${String(error?.message || error)}`, 100);
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      createAllMissingSubjectImages.disabled = false;
      extractSubjects.disabled = false;
      createAllMissingSubjectImages.textContent = "Generate Missing Images";
    }
  }

  async function describeReferenceItemsWithGemma(items, referenceType, options = {}) {
    const failedIds = new Set((options.failedIds || []).map((value) => String(value)));
    const jobs = items
      .filter((item) => item?.target && hasReferenceImage(item.target.image || {})
        && (failedIds.size ? failedIds.has(String(item.target.id || "")) : !String(item.target.description || "").trim()));
    if (!jobs.length) {
      toast(referenceType === "location" ? "No location images are missing descriptions." : "No character images are missing descriptions.");
      return 0;
    }
    const progress = createProgressWindow(referenceType === "location" ? "Describing locations" : "Describing characters", { zIndex: 100008 });
    let completed = 0;
    const failures = [];
    try {
      for (let index = 0; index < jobs.length; index += 1) {
        const { target, label } = jobs[index];
        const isLast = index === jobs.length - 1;
        try {
          progress.set(`Vision Gemma describing ${referenceType} image...\n${index + 1}/${jobs.length}: ${label || target.name || referenceType}\n${gemmaRunnerLine({ vision: true })}`, 8 + Math.round((index / Math.max(1, jobs.length)) * 84));
          await describeReferenceImageWithGemma(target, referenceType, {
            unloadAfter: options.keepLoaded ? isLast : true,
            clearBeforeLoad: index === 0 && Boolean(options.clearBeforeLoad),
          });
          if (referenceType === "subject" && refs.subject_count === 1) {
            subjectDescription.value = target.description || "";
            refs.subject.description = target.description || "";
          }
          completed += 1;
          renderAll();
          await autoSaveSessionQuiet(`Gemma described ${referenceType} ${label || target.name || index + 1}`);
        } catch (error) {
          if (!isRecoverableBuildGemmaError(error)) throw error;
          failures.push({
            key: `reference-description:${referenceType}:${target.id}`,
            segmentId: String(target.id || ""),
            sceneLabel: label || target.name || referenceType,
            error: String(error?.message || error),
            raw: String(error?.message || error),
          });
          progress.set(`${label || target.name || referenceType} skipped. Continuing with the remaining descriptions...`, 8 + Math.round((index / Math.max(1, jobs.length)) * 84));
        }
      }
      progress.set(`Gemma descriptions complete.\nUpdated ${completed} ${referenceType}${completed === 1 ? "" : "s"}.${failures.length ? ` ${failures.length} skipped.` : ""}`, 100);
      progress.close(1800);
      toast(`Gemma described ${completed} ${referenceType}${completed === 1 ? "" : "s"}${failures.length ? ` with ${failures.length} skipped` : ""}.`, Boolean(failures.length));
      if (failures.length) showGemmaBatchFailures(failures, {
        retryHandler: (failed) => describeReferenceItemsWithGemma(items, referenceType, {
          ...options,
          failedIds: failed.map((item) => item.segmentId),
        }),
      });
      return completed;
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
      return completed;
    }
  }

  async function describeSingleReferenceWithGemma(target, referenceType, label = "") {
    const effectiveType = referenceType === "subject" ? (target?.reference_type || "character") : referenceType;
    const progress = createProgressWindow(referenceType === "location" ? "Describing location" : "Describing reference", { zIndex: 100008 });
    try {
      progress.set(`Vision Gemma describing ${effectiveType} image...\n${label || target?.name || effectiveType}\n${gemmaRunnerLine({ vision: true })}`, 18);
      const description = await describeReferenceImageWithGemma(target, referenceType, { unloadAfter: true, clearBeforeLoad: false });
      applyReferenceDescription(referenceType, target, description);
      renderAll();
      await autoSaveSessionQuiet(`Gemma described ${effectiveType}`);
      progress.set("Description updated.", 100);
      progress.close(1200);
      toast(referenceType === "location" ? "Location description updated." : "Reference description updated.");
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    }
  }

  async function runFluxReferenceWithKrea2(referenceType, target, sourceText, name = "", generatorSettings = {}) {
    const text = String(sourceText || "").trim();
    if (!text) {
      toast(referenceType === "subject" ? "Enter a subject description first." : "Enter a location description first.", true);
      return;
    }
    const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
    if (!modelFile && !["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner)) {
      toast("Choose a non-vision Gemma model first.", true);
      return;
    }
    let progress = null;
    let ranKrea2 = false;
    try {
      setInlineProgress(referenceType === "subject" ? "Creating subject prompt with Gemma..." : "Creating location prompt with Gemma...", 8);
      progress = createProgressWindow(referenceType === "subject" ? "Creating subject reference" : "Creating location reference", { zIndex: 100008 });
      progress.set(`Creating reference prompt with Gemma...\n${gemmaRunnerLine()}`, 8);
      const styleTheme = state.useVrgdgTextContext ? await loadContextTextQuiet(themeStyleInput.value) : "";
      const promptData = await postJson("/vrgdg/music_builder/flux_reference_zimage_prompt", {
        ...textGemmaRunnerPayload(),
        model_file: modelFile,
        reference_type: referenceType,
        source_text: text,
        style_theme: styleTheme,
        unload_after: true,
      }, 3 * 60 * 1000);
      const kreaSettings = generatorSettings.krea2 || {};
      setInlineProgress("Building Krea2 + ZImage enhancer workflow...", 28);
      progress.set("Building Krea2 + ZImage enhancer workflow...", 28);
      const built = await postJson("/vrgdg/workflow_runner/build_krea2_prompt", krea2ReferencePayload(promptData.prompt, kreaSettings));
      setInlineProgress("Queueing Krea2 reference workflow...", 42);
      progress.set("Queueing Krea2 reference workflow...", 42);
      const queued = await queueWorkflowPrompt(built.prompt);
      const promptId = queued?.prompt_id;
      if (!promptId) throw new Error("ComfyUI queued the Krea2 reference but did not return a prompt_id.");
      ranKrea2 = true;
      const images = await waitForImages(promptId, (message) => {
        setInlineProgress(`${message}\nPrompt ID: ${promptId}`, 66);
        progress?.set(`${message}\nPrompt ID: ${promptId}`, 66);
      });
      const image = images[images.length - 1];
      if (!image) throw new Error("Krea2 did not return a reference image.");
      setInlineProgress("Saving reference image into the project...", 88);
      progress.set("Saving reference image into the project...", 88);
      const saved = await postJson("/vrgdg/music_builder/save_flux_reference_image", {
        project_folder: projectInput.value || state.projectFolder,
        reference_type: referenceType,
        name: name || text.slice(0, 48) || referenceType,
        image,
      });
      target.image.path = saved.saved_path || "";
      target.image.data = "";
      target.image.name = `${name || referenceType}.png`;
      await describeGeneratedReference(referenceType, target, promptData.prompt);
      await persistGeneratedReferenceImage(referenceType, target);
      renderAll();
      setInlineProgress("Cleaning memory after Krea2 reference...", 94);
      await runImageMemoryCleanupQuiet(progress, "Krea2 reference", 94);
      setInlineProgress("Reference image ready.", 100);
      progress.set("Reference image ready.", 100);
      progress.close(1300);
      hideInlineProgress();
      toast(referenceType === "subject" ? "Subject reference created with Krea2." : "Location reference created with Krea2.");
    } catch (error) {
      if (ranKrea2) {
        setInlineProgress("Cleaning memory after failed Krea2 reference...", 100);
        await runImageMemoryCleanupQuiet(progress, "failed Krea2 reference", 100);
      }
      setInlineProgress(`Error:\n${String(error?.message || error)}`, 100);
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    }
  }

  function createFluxReferenceWithZImage(referenceType, target, sourceText, name = "") {
    const text = String(sourceText || "").trim();
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100009;background:rgba(0,0,0,.65);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(920px,calc(100vw - 36px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.6);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Generate ${referenceType === "subject" ? "Subject" : "Location"} Reference</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">${referenceType === "subject" ? "Choose how Gemma should create this subject, then choose the image workflow." : "Choose the image workflow for this Reference Builder image."}</div>`;
    const close = makeButton("Close");
    header.append(title, close);
    const note = document.createElement("div");
    note.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;color:#dbeafe;padding:10px;font-size:12px;line-height:1.45;";
    note.textContent = referenceType === "subject"
      ? "Gemma will turn your selected source into one character reference-sheet prompt. If you choose lyrics/style, it can invent a character that fits the song. If you do not like the result, run it again."
      : "ZImage uses the existing reference-image workflow. Krea2 uses the hidden Krea2 text-to-image workflow, then runs the ZImage enhancer pass. Browser AI uses the selected browser provider and login/settings from the Browser AI image panel. A location description is required before generation can run.";

    const generatorControls = buildReferenceGeneratorControls();
    const { zModel, zClip, zVae, kreaModel, kreaClip, kreaVae, seed, seedMode, firstWidth, firstHeight, width, height, readZImage, readKrea2 } = generatorControls;
    const subjectOptionsCard = document.createElement("div");
    subjectOptionsCard.style.cssText = "border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;display:flex;flex-direction:column;gap:10px;";
    const subjectMode = makeSelect([
      { value: "description_only", label: "Use description only" },
      { value: "invent_from_lyrics", label: "Invent character from lyrics/style" },
      { value: "description_lyrics", label: "Use description + lyrics/style" },
      { value: "image_description", label: "Use image/description identity" },
      { value: "image_lyrics", label: "Use image identity + lyrics/style" },
      { value: "all", label: "Use everything" },
    ], "description_only");
    const savedDraft = target?.reference_generation_draft && typeof target.reference_generation_draft === "object" ? target.reference_generation_draft : {};
    if (referenceType === "subject" && savedDraft.subject_mode && Array.from(subjectMode.options).some((option) => option.value === savedDraft.subject_mode)) {
      subjectMode.value = savedDraft.subject_mode;
    }
    const subjectLabel = makeInput(savedDraft.subject_label || target?.name || name || "");
    subjectLabel.placeholder = "lead performer, the woman, villain, dancer...";
    const genderRole = makeInput(savedDraft.gender_role || "");
    genderRole.placeholder = "optional: female lead singer, older male narrator, nonbinary dancer...";
    const songStyle = document.createElement("textarea");
    songStyle.value = savedDraft.song_style || "";
    songStyle.placeholder = "Optional song / visual style: dark synthpop, southern gothic, glossy K-pop, VHS horror...";
    songStyle.style.cssText = "min-height:58px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;line-height:1.35;";
    const lyricsInput = document.createElement("textarea");
    lyricsInput.value = savedDraft.lyrics_context || locationScoutLyricsPayloadForGpt() || lyricsFromSegmentsForPromptCreator(state.segments || []);
    lyricsInput.placeholder = "Optional lyrics/dialogue context...";
    lyricsInput.style.cssText = "min-height:100px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;line-height:1.35;font-family:monospace;";
    const extraDirection = document.createElement("textarea");
    extraDirection.value = savedDraft.extra_direction || "";
    extraDirection.placeholder = "Optional extra direction: age, wardrobe, hair, era, attitude, what to avoid...";
    extraDirection.style.cssText = "min-height:70px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;line-height:1.35;";
    const subjectSourceGrid = document.createElement("div");
    subjectSourceGrid.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;";
    subjectSourceGrid.append(
      makeField("Generate subject from", subjectMode),
      makeField("Subject label", subjectLabel),
      makeField("Gender / role", genderRole),
      makeField("Song / visual style", songStyle),
    );
    subjectOptionsCard.append(subjectSourceGrid, makeField("Lyrics / dialogue context", lyricsInput), makeField("Extra user direction", extraDirection));
    const hasSubjectImage = () => hasReferenceImage(target?.image || {});
    const subjectImageNote = document.createElement("div");
    subjectImageNote.style.cssText = "font-size:11px;color:#94a3b8;line-height:1.35;";
    const syncSubjectModeHint = () => {
      const needsImage = /^image_|^all$/.test(subjectMode.value || "");
      subjectImageNote.textContent = needsImage
        ? hasSubjectImage()
          ? "This subject already has an image. Gemma will use the row description/image identity notes as identity guidance while creating the new reference sheet prompt."
          : "No subject image is loaded yet. This mode will still use text, but upload/drop an image first if you want image identity guidance."
        : "Description and lyrics/style modes do not require an uploaded image.";
    };
    subjectMode.addEventListener("change", syncSubjectModeHint);
    subjectOptionsCard.append(subjectImageNote);
    syncSubjectModeHint();
    subjectOptionsCard.style.display = referenceType === "subject" ? "flex" : "none";
    const persistReferenceGenerationDraft = () => {
      if (!target || typeof target !== "object") return;
      target.reference_generation_draft = {
        subject_mode: subjectMode.value || "description_only",
        subject_label: String(subjectLabel.value || ""),
        gender_role: String(genderRole.value || ""),
        song_style: String(songStyle.value || ""),
        lyrics_context: String(lyricsInput.value || ""),
        extra_direction: String(extraDirection.value || ""),
      };
      if (referenceType === "subject") {
        const matchingSubject = refs.subjects.find((subject) => subject === target || (subject.id && subject.id === target.id));
        if (matchingSubject) matchingSubject.reference_generation_draft = target.reference_generation_draft;
        if (refs.subject_count === 1 && (!matchingSubject || refs.subjects[0]?.id === matchingSubject.id || target === refs.subject)) {
          refs.subject.reference_generation_draft = target.reference_generation_draft;
          if (refs.subjects[0]) refs.subjects[0].reference_generation_draft = target.reference_generation_draft;
        }
      }
    };
    [subjectMode, subjectLabel, genderRole, songStyle, lyricsInput, extraDirection].forEach((control) => {
      control.addEventListener("input", persistReferenceGenerationDraft);
      control.addEventListener("change", persistReferenceGenerationDraft);
    });
    const subjectSourceTextForRun = () => {
      persistReferenceGenerationDraft();
      if (referenceType !== "subject") return text;
      const mode = subjectMode.value || "description_only";
      const label = String(subjectLabel.value || target?.name || name || "Subject").trim();
      const description = String(target?.description || text || "").trim();
      const type = String(target?.reference_type || "character").trim() || "character";
      const includeDescription = ["description_only", "description_lyrics", "image_description", "all"].includes(mode);
      const includeLyrics = ["invent_from_lyrics", "description_lyrics", "image_lyrics", "all"].includes(mode);
      const includeImage = ["image_description", "image_lyrics", "all"].includes(mode);
      const parts = [
        `Generation mode: ${mode.replace(/_/g, " ")}`,
        `Subject label: ${label || "Subject"}`,
        `Reference type: ${type}`,
      ];
      if (includeDescription) parts.push(`Existing description:\n${description || "(none provided)"}`);
      if (includeImage) {
        const image = target?.image || {};
        parts.push(`Reference image guidance:\n${hasReferenceImage(image) ? `Use the already loaded subject image identity as guidance. Image name/path: ${image.name || image.path || "loaded image"}.` : "No image is currently loaded; rely on text context only."}`);
      }
      if (includeLyrics) parts.push(`Lyrics / dialogue context:\n${String(lyricsInput.value || "").trim() || "(none provided)"}`);
      if (String(songStyle.value || "").trim()) parts.push(`Song / visual style:\n${String(songStyle.value || "").trim()}`);
      if (String(genderRole.value || "").trim()) parts.push(`Gender / role:\n${String(genderRole.value || "").trim()}`);
      if (String(extraDirection.value || "").trim()) parts.push(`Extra user direction:\n${String(extraDirection.value || "").trim()}`);
      if (!includeDescription && !includeLyrics && !String(extraDirection.value || "").trim()) {
        parts.push(`Existing description:\n${description || label || "Create a visually distinctive character that fits the project."}`);
      }
      return parts.join("\n\n");
    };

    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;align-items:start;";
    const modalHelp = (text) => {
      const help = document.createElement("div");
      help.textContent = text;
      help.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.4;";
      return help;
    };
    const sharedZImageCard = document.createElement("div");
    sharedZImageCard.style.cssText = "border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;";
    const sharedTitle = document.createElement("div");
    sharedTitle.style.cssText = "grid-column:1/-1;font-size:13px;font-weight:900;color:#cffafe;";
    sharedTitle.textContent = "Shared ZImage / Enhancer Models";
    const sharedNote = document.createElement("div");
    sharedNote.style.cssText = "grid-column:1/-1;font-size:12px;color:#94a3b8;line-height:1.35;";
    sharedNote.textContent = "Used directly by ZImage, and reused as the enhancer pass when Krea2 is selected.";
    sharedZImageCard.append(
      sharedTitle,
      sharedNote,
      makeField("Diffusion model", zModel.wrapper),
      makeField("Text encoder", zClip.wrapper),
      makeField("VAE", zVae.wrapper)
    );
    const optionCard = (heading, body, actionLabel, action) => {
      const card = document.createElement("div");
      card.style.cssText = "border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;display:flex;flex-direction:column;gap:10px;";
      const h = document.createElement("div");
      h.style.cssText = "font-size:14px;font-weight:900;color:#cffafe;";
      h.textContent = heading;
      const button = makeButton(actionLabel, "primary");
      button.onclick = async () => {
        const runText = subjectSourceTextForRun();
        if (!String(runText || "").trim()) {
          toast(referenceType === "subject" ? "Enter a subject description, lyrics, or extra direction first." : "Enter a location description first.", true);
          return;
        }
        closeModal();
        await action(runText);
      };
      card.append(h, ...body, button);
      return card;
    };
    grid.append(
      optionCard("ZImage", [
        modalHelp("Use the shared ZImage model settings above for a direct ZImage reference image."),
      ], "Use ZImage", (runText) => runFluxReferenceWithZImage(referenceType, target, runText, name, {
        zimage: readZImage(),
      })),
      optionCard("Krea2 + ZImage Enhancer", [
        makeField("Krea2 diffusion model", kreaModel.wrapper),
        makeField("Krea2 text encoder", kreaClip.wrapper),
        makeField("Krea2 VAE", kreaVae.wrapper),
        makeField("Seed", seed),
        makeField("Seed mode", seedMode),
        makeField("Krea first width", firstWidth),
        makeField("Krea first height", firstHeight),
        makeField("Final width", width),
        makeField("Final height", height),
      ], "Use Krea2 + Enhancer", (runText) => runFluxReferenceWithKrea2(referenceType, target, runText, name, {
        krea2: readKrea2(),
      })),
      optionCard("Flow/GPT", [
        modalHelp("Use the selected browser image provider. Flow requires its browser profile and manual aspect setup; GPT Image appends the selected aspect ratio; Meta AI uses the meta.ai profile and login."),
      ], "Use Flow/GPT", (runText) => runFluxReferenceWithFlowGpt(referenceType, target, runText, name)),
    );
    const footer = document.createElement("div");
    footer.style.cssText = "display:flex;justify-content:flex-end;";
    const cancel = makeButton("Cancel");
    footer.append(cancel);
    box.append(header, note, subjectOptionsCard, sharedZImageCard, grid, footer);
    backdrop.append(box);
    document.body.append(backdrop);
    function closeModal() {
      persistReferenceGenerationDraft();
      autoSaveSessionQuiet("reference generation draft saved").catch(() => null);
      backdrop.remove();
    }
    close.onclick = closeModal;
    cancel.onclick = closeModal;
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) closeModal();
    });
  }

  function openMissingSubjectImageGeneratorDialog() {
    ensureSubjectCount();
    if (refs.subject_count === 1 && refs.subjects[0]) {
      refs.subjects[0].name = subjectNameInput.value || refs.subjects[0].name || refs.subject.name || "Subject";
      refs.subjects[0].reference_type = subjectTypeSelect.value || refs.subjects[0].reference_type || refs.subject.reference_type || "character";
      refs.subjects[0].description = subjectDescription.value || refs.subjects[0].description || refs.subject.description || "";
      refs.subject = { ...refs.subject, ...refs.subjects[0], image: refs.subjects[0].image || refs.subject.image || { path: "", data: "", name: "" } };
    }
    const missingCount = refs.subjects
      .filter((subject) => {
        const image = subject?.image || {};
        return String(subject?.name || subject?.description || "").trim()
          && !String(image.path || image.data || "").trim();
      }).length;
    if (!missingCount) {
      toast("All listed subjects already have images, or no usable subjects were found.");
      return;
    }
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100009;background:rgba(0,0,0,.65);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(820px,calc(100vw - 36px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.6);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Generate Missing Subject Images</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Choose one workflow for ${missingCount} missing subject/reference image${missingCount === 1 ? "" : "s"}.</div>`;
    const close = makeButton("Close");
    header.append(title, close);
    const note = document.createElement("div");
    note.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;color:#dbeafe;padding:10px;font-size:12px;line-height:1.45;";
    note.textContent = "This runs Gemma prompts for each missing subject/reference, then generates each image with the workflow you choose here. Browser AI uses the selected browser provider and login/settings from the Browser AI image panel.";

    const generatorControls = buildReferenceGeneratorControls();
    const { zModel, zClip, zVae, kreaModel, kreaClip, kreaVae, seed, seedMode, firstWidth, firstHeight, width, height, readZImage, readKrea2 } = generatorControls;

    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;align-items:start;";
    const modalHelp = (text) => {
      const help = document.createElement("div");
      help.textContent = text;
      help.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.4;";
      return help;
    };
    const sharedZImageCard = document.createElement("div");
    sharedZImageCard.style.cssText = "border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;";
    const sharedTitle = document.createElement("div");
    sharedTitle.style.cssText = "grid-column:1/-1;font-size:13px;font-weight:900;color:#cffafe;";
    sharedTitle.textContent = "Shared ZImage / Enhancer Models";
    const sharedNote = document.createElement("div");
    sharedNote.style.cssText = "grid-column:1/-1;font-size:12px;color:#94a3b8;line-height:1.35;";
    sharedNote.textContent = "Used directly by ZImage, and reused as the enhancer pass when Krea2 is selected.";
    sharedZImageCard.append(
      sharedTitle,
      sharedNote,
      makeField("Diffusion model", zModel.wrapper),
      makeField("Text encoder", zClip.wrapper),
      makeField("VAE", zVae.wrapper)
    );
    const optionCard = (heading, body, actionLabel, action) => {
      const card = document.createElement("div");
      card.style.cssText = "border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;display:flex;flex-direction:column;gap:10px;";
      const h = document.createElement("div");
      h.style.cssText = "font-size:14px;font-weight:900;color:#cffafe;";
      h.textContent = heading;
      const button = makeButton(actionLabel, "primary");
      button.onclick = async () => {
        closeModal();
        await action();
      };
      card.append(h, ...body, button);
      return card;
    };
    grid.append(
      optionCard("ZImage", [
        modalHelp("Use the shared ZImage model settings above for every missing subject/reference image."),
      ], "Use ZImage For All Missing", () => createMissingSubjectReferencesWithImageWorkflow("zimage", {
        zimage: readZImage(),
      })),
      optionCard("Krea2 + ZImage Enhancer", [
        makeField("Krea2 diffusion model", kreaModel.wrapper),
        makeField("Krea2 text encoder", kreaClip.wrapper),
        makeField("Krea2 VAE", kreaVae.wrapper),
        makeField("Seed", seed),
        makeField("Seed mode", seedMode),
        makeField("Krea first width", firstWidth),
        makeField("Krea first height", firstHeight),
        makeField("Final width", width),
        makeField("Final height", height),
      ], "Use Krea2 + Enhancer For All Missing", () => createMissingSubjectReferencesWithImageWorkflow("krea2", {
        krea2: readKrea2(),
      })),
      optionCard("Flow/GPT", [
        modalHelp("Use the selected browser image provider for every missing subject/reference image. Log into the chosen browser profile first."),
      ], "Use Flow/GPT For All Missing", () => createMissingSubjectReferencesWithImageWorkflow("flow_gpt")),
    );
    const footer = document.createElement("div");
    footer.style.cssText = "display:flex;justify-content:flex-end;";
    const cancel = makeButton("Cancel");
    footer.append(cancel);
    box.append(header, note, sharedZImageCard, grid, footer);
    backdrop.append(box);
    document.body.append(backdrop);
    function closeModal() {
      backdrop.remove();
    }
    close.onclick = closeModal;
    cancel.onclick = closeModal;
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) closeModal();
    });
  }

  function openMissingLocationImageGeneratorDialog() {
    const missingCount = refs.locations
      .filter((location) => {
        const image = location?.image || {};
        return String(location?.name || location?.description || "").trim()
          && !String(image.path || image.data || "").trim();
      }).length;
    if (!missingCount) {
      toast("All listed locations already have images, or no usable locations were found.");
      return;
    }
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100009;background:rgba(0,0,0,.65);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(820px,calc(100vw - 36px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.6);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Generate Missing Location Images</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Choose one workflow for ${missingCount} missing location image${missingCount === 1 ? "" : "s"}.</div>`;
    const close = makeButton("Close");
    header.append(title, close);
    const note = document.createElement("div");
    note.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;color:#dbeafe;padding:10px;font-size:12px;line-height:1.45;";
    note.textContent = "This runs Gemma prompts for each missing location, then generates each image with the workflow you choose here. Browser AI uses the selected browser provider and login/settings from the Browser AI image panel.";

    const generatorControls = buildReferenceGeneratorControls();
    const { zModel, zClip, zVae, kreaModel, kreaClip, kreaVae, seed, seedMode, firstWidth, firstHeight, width, height, readZImage, readKrea2 } = generatorControls;

    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;align-items:start;";
    const modalHelp = (text) => {
      const help = document.createElement("div");
      help.textContent = text;
      help.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.4;";
      return help;
    };
    const sharedZImageCard = document.createElement("div");
    sharedZImageCard.style.cssText = "border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;";
    const sharedTitle = document.createElement("div");
    sharedTitle.style.cssText = "grid-column:1/-1;font-size:13px;font-weight:900;color:#cffafe;";
    sharedTitle.textContent = "Shared ZImage / Enhancer Models";
    const sharedNote = document.createElement("div");
    sharedNote.style.cssText = "grid-column:1/-1;font-size:12px;color:#94a3b8;line-height:1.35;";
    sharedNote.textContent = "Used directly by ZImage, and reused as the enhancer pass when Krea2 is selected.";
    sharedZImageCard.append(
      sharedTitle,
      sharedNote,
      makeField("Diffusion model", zModel.wrapper),
      makeField("Text encoder", zClip.wrapper),
      makeField("VAE", zVae.wrapper)
    );
    const optionCard = (heading, body, actionLabel, action) => {
      const card = document.createElement("div");
      card.style.cssText = "border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;display:flex;flex-direction:column;gap:10px;";
      const h = document.createElement("div");
      h.style.cssText = "font-size:14px;font-weight:900;color:#cffafe;";
      h.textContent = heading;
      const button = makeButton(actionLabel, "primary");
      button.onclick = async () => {
        closeModal();
        await action();
      };
      card.append(h, ...body, button);
      return card;
    };
    grid.append(
      optionCard("ZImage", [
        modalHelp("Use the shared ZImage model settings above for every missing location image."),
      ], "Use ZImage For All Missing", () => createMissingLocationReferencesWithZImage("zimage", {
        zimage: readZImage(),
      })),
      optionCard("Krea2 + ZImage Enhancer", [
        makeField("Krea2 diffusion model", kreaModel.wrapper),
        makeField("Krea2 text encoder", kreaClip.wrapper),
        makeField("Krea2 VAE", kreaVae.wrapper),
        makeField("Seed", seed),
        makeField("Seed mode", seedMode),
        makeField("Krea first width", firstWidth),
        makeField("Krea first height", firstHeight),
        makeField("Final width", width),
        makeField("Final height", height),
      ], "Use Krea2 + Enhancer For All Missing", () => createMissingLocationReferencesWithZImage("krea2", {
        krea2: readKrea2(),
      })),
      optionCard("Flow/GPT", [
        modalHelp("Use the selected browser image provider for every missing location image. Log into the chosen browser profile first."),
      ], "Use Flow/GPT For All Missing", () => createMissingLocationReferencesWithZImage("flow_gpt")),
    );
    const footer = document.createElement("div");
    footer.style.cssText = "display:flex;justify-content:flex-end;";
    const cancel = makeButton("Cancel");
    footer.append(cancel);
    box.append(header, note, sharedZImageCard, grid, footer);
    backdrop.append(box);
    document.body.append(backdrop);
    function closeModal() {
      backdrop.remove();
    }
    close.onclick = closeModal;
    cancel.onclick = closeModal;
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) closeModal();
    });
  }

  return {
    applyReferenceDescription, createFluxReferenceWithZImage, describeReferenceItemsWithGemma,
    describeSingleReferenceWithGemma, openMissingLocationImageGeneratorDialog,
    openMissingSubjectImageGeneratorDialog,
  };
}
