import { BROWSER_IMAGE_PROVIDERS } from "../VRGDG_BrowserImageBridge.js";
import {
  DEFAULT_I2V_PASS1_SIGMAS,
  DEFAULT_I2V_PASS2_SIGMAS,
  DEFAULT_INGREDIENTS_SAMPLER,
  DEFAULT_NB_IMAGE_MODEL,
  NB_IMAGE_MODELS,
} from "./constants.mjs";
import {
  makeButton,
  makeCheckbox,
  makeField,
  makeInput,
  makeMiniButton,
  makeSearchableLoraPicker,
  makeSelect,
  toast,
} from "./controls.mjs";
import { makeImageModelCard } from "./inspector.mjs";
import { chooseModelValue, wireSearchablePicker } from "./model_pickers.mjs";
import {
  browserImageProviderLabel,
  browserImageProviderShortLabel,
  browserImageProviderTimeout,
  cloneFlowGptBrowserSettings,
  cloneKrea2TwoPassSettings,
  defaultFlowGptBrowserSettings,
  normalizeFlowGptBrowserProvider,
} from "./model_settings.mjs";

export function normalizeI2VSigmasText(value, fallback) {
  const text = String(value || "").trim();
  if (!text) return fallback;
  const parts = text.split(",").map((item) => item.trim()).filter(Boolean);
  if (!parts.length || parts.some((item) => !Number.isFinite(Number(item)))) return fallback;
  return parts.join(", ");
}

export function setI2VStrengthPair(slider, input, value) {
  const next = Math.max(0, Math.min(1, Number(value ?? 1)));
  slider.value = String(next);
  input.value = String(next);
}

export function wireImageSettingsInputs({
  activeSegment, ernieImageTriggerInput, flowGptAskPreviousImage, flowGptAspectRatio, flowGptFailureMode,
  flowGptManualAutoAdvance, flowGptManualMode, flowGptPrompt, flowGptRetries, flowGptTimeout,
  fluxImageTriggerInput, fluxUseDirectorNotes, fluxUseTextOnlyGemmaPrompt, imageTriggerInput,
  krea2TwoPassImageTriggerInput, nbApiKey, nbModelSelect, nbNotes, nbPrompt, nbUseDirectorNotes,
  nbUseTextOnlyGemmaPrompt, pushHistory, saveErnieImageSettingsFromPanel, saveFlowGptBrowserSettingsFromPanel,
  saveFluxKleinSettingsFromPanel, saveKrea2TwoPassSettingsFromPanel, saveNBImageSettingsFromPanel,
  saveZImageSettingsFromPanel, syncFlowGptManualPanel, syncSegmentFlowGptPrompt,
}) {
  imageTriggerInput.addEventListener("input", saveZImageSettingsFromPanel);
  ernieImageTriggerInput.addEventListener("input", saveErnieImageSettingsFromPanel);
  krea2TwoPassImageTriggerInput.addEventListener("input", saveKrea2TwoPassSettingsFromPanel);
  fluxImageTriggerInput.addEventListener("input", saveFluxKleinSettingsFromPanel);
  fluxUseTextOnlyGemmaPrompt.input.addEventListener("change", saveFluxKleinSettingsFromPanel);
  fluxUseDirectorNotes.input.addEventListener("change", saveFluxKleinSettingsFromPanel);
  nbApiKey.addEventListener("input", saveNBImageSettingsFromPanel);
  nbModelSelect.addEventListener("change", saveNBImageSettingsFromPanel);
  nbUseTextOnlyGemmaPrompt.input.addEventListener("change", saveNBImageSettingsFromPanel);
  nbUseDirectorNotes.input.addEventListener("change", saveNBImageSettingsFromPanel);
  nbNotes.addEventListener("input", saveNBImageSettingsFromPanel);
  nbPrompt.addEventListener("input", saveNBImageSettingsFromPanel);
  flowGptAspectRatio.addEventListener("change", saveFlowGptBrowserSettingsFromPanel);
  flowGptTimeout.addEventListener("input", saveFlowGptBrowserSettingsFromPanel);
  flowGptRetries.addEventListener("input", saveFlowGptBrowserSettingsFromPanel);
  flowGptFailureMode.addEventListener("change", saveFlowGptBrowserSettingsFromPanel);
  flowGptAskPreviousImage.input.addEventListener("change", saveFlowGptBrowserSettingsFromPanel);
  flowGptPrompt.addEventListener("input", () => {
    pushHistory();
    const segment = activeSegment();
    if (segment) syncSegmentFlowGptPrompt(segment, flowGptPrompt.value || "", { preserveTypingWhitespace: true, skipInputSync: true });
  });
  flowGptManualMode.input.addEventListener("change", () => {
    syncFlowGptManualPanel();
    toast(flowGptManualMode.input.checked ? "Flow/GPT Manual Mode is on." : "Flow/GPT Manual Mode is off.");
  });
  flowGptManualAutoAdvance.input.addEventListener("change", syncFlowGptManualPanel);
}

export function wireImageModelControls({
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
}) {
  for (const control of [zFirstWidth, zFirstHeight, zSecondWidth, zSecondHeight, zSeed, zSeedMode, zBatchSize, zLoraCount, zI2IStartStep, zI2IPath]) {
    control.addEventListener("input", saveZImageSettingsFromPanel);
    control.addEventListener("change", saveZImageSettingsFromPanel);
  }
  for (const control of [ernieWidth, ernieHeight, ernieSeed, ernieSeedMode, ernieBatchSize, ernieLoraCount, ernieI2IStartStep, ernieI2IPath]) {
    control.addEventListener("input", saveErnieImageSettingsFromPanel);
    control.addEventListener("change", saveErnieImageSettingsFromPanel);
  }
  for (const control of [krea2TwoPassLoraCount, krea2TwoPassAspectRatio, krea2TwoPassSampler, krea2TwoPassSeed, krea2TwoPassSeedMode, krea2TwoPassCfg, krea2TwoPassBatchSize, krea2TwoPassCreativity, krea2TwoPassCreativityInput, krea2TwoPassI2IPath]) {
    control.addEventListener("input", saveKrea2TwoPassSettingsFromPanel);
    control.addEventListener("change", saveKrea2TwoPassSettingsFromPanel);
  }
  for (const picker of [zUnetPicker, zClipPicker, zVaePicker]) {
    wireSearchablePicker(picker, saveZImageSettingsFromPanel);
    picker.input.addEventListener("change", saveZImageSettingsFromPanel);
  }
  for (const picker of [ernieUnetPicker, ernieClipPicker, ernieVaePicker]) {
    wireSearchablePicker(picker, saveErnieImageSettingsFromPanel);
    picker.input.addEventListener("change", saveErnieImageSettingsFromPanel);
  }
  for (const picker of [krea2TwoPassUnetPicker, krea2TwoPassClipPicker, krea2TwoPassVaePicker]) {
    wireSearchablePicker(picker, saveKrea2TwoPassSettingsFromPanel);
    picker.input.addEventListener("change", saveKrea2TwoPassSettingsFromPanel);
  }
  for (const slot of krea2TwoPassLoraSlots) {
    wireSearchablePicker(slot.picker, saveKrea2TwoPassSettingsFromPanel);
    slot.picker.input.addEventListener("change", saveKrea2TwoPassSettingsFromPanel);
    slot.firstPassStrength.addEventListener("input", saveKrea2TwoPassSettingsFromPanel);
    slot.firstPassStrength.addEventListener("change", saveKrea2TwoPassSettingsFromPanel);
    slot.secondPassStrength.addEventListener("input", saveKrea2TwoPassSettingsFromPanel);
    slot.secondPassStrength.addEventListener("change", saveKrea2TwoPassSettingsFromPanel);
  }
  zI2ISlider.addEventListener("input", () => {
    zI2IStartStep.value = zI2ISlider.value;
    saveZImageSettingsFromPanel();
  });
  zI2IStartStep.addEventListener("input", () => {
    const value = Math.max(1, Math.min(8, Number(zI2IStartStep.value || 5)));
    zI2ISlider.value = String(value);
  });
  zUseLora.input.addEventListener("change", saveZImageSettingsFromPanel);
  zUseImageToImage.input.addEventListener("change", saveZImageSettingsFromPanel);
  ernieI2ISlider.addEventListener("input", () => {
    ernieI2IStartStep.value = ernieI2ISlider.value;
    saveErnieImageSettingsFromPanel();
  });
  ernieI2IStartStep.addEventListener("input", () => {
    const value = Math.max(1, Math.min(8, Number(ernieI2IStartStep.value || 5)));
    ernieI2ISlider.value = String(value);
  });
  ernieUseLora.input.addEventListener("change", saveErnieImageSettingsFromPanel);
  ernieUseImageToImage.input.addEventListener("change", saveErnieImageSettingsFromPanel);
  krea2TwoPassCreativity.addEventListener("input", () => {
    krea2TwoPassCreativityInput.value = krea2TwoPassCreativity.value;
  });
  krea2TwoPassCreativityInput.addEventListener("input", () => {
    krea2TwoPassCreativity.value = String(Math.max(0, Math.min(10, Number(krea2TwoPassCreativityInput.value || 0))));
  });
  krea2TwoPassUseLora.input.addEventListener("change", saveKrea2TwoPassSettingsFromPanel);
  krea2TwoPassUseImageToImage.input.addEventListener("change", saveKrea2TwoPassSettingsFromPanel);
  for (const slot of zLoraSlots) {
    wireSearchablePicker(slot.picker, saveZImageSettingsFromPanel);
    slot.firstPassStrength.addEventListener("input", saveZImageSettingsFromPanel);
    slot.firstPassStrength.addEventListener("change", saveZImageSettingsFromPanel);
    slot.secondPassStrength.addEventListener("input", saveZImageSettingsFromPanel);
    slot.secondPassStrength.addEventListener("change", saveZImageSettingsFromPanel);
  }
  for (const slot of ernieLoraSlots) {
    wireSearchablePicker(slot.picker, saveErnieImageSettingsFromPanel);
    slot.strength.addEventListener("input", saveErnieImageSettingsFromPanel);
    slot.strength.addEventListener("change", saveErnieImageSettingsFromPanel);
  }

  i2vUseGgufModel.input.addEventListener("change", () => {
    syncI2VVideoModelPickerVisibility();
    saveI2VVideoSettingsFromPanel();
  });
  for (const control of [i2vUseSageAttention, i2vEnableFp16Accumulation]) {
    control.input.addEventListener("change", saveI2VVideoSettingsFromPanel);
  }
  for (const picker of [i2vUnetPicker, i2vDiffusionModelPicker, i2vVaePicker, i2vClip1Picker, i2vClip2Picker, i2vUpscalePicker, i2vAudioVaePicker]) {
    wireSearchablePicker(picker, saveI2VVideoSettingsFromPanel);
    picker.input.addEventListener("change", saveI2VVideoSettingsFromPanel);
  }
}

export function wireFluxKleinControls({
  fluxClipPicker, fluxHeight, fluxLoraCount, fluxLoraSlots, fluxNotes, fluxPrompt, fluxSeed, fluxUnetPicker,
  fluxUseLora, fluxVaePicker, fluxWidth, saveFluxKleinSettingsFromPanel, useFluxKlein,
}) {
  for (const picker of [fluxUnetPicker, fluxClipPicker, fluxVaePicker]) {
    wireSearchablePicker(picker, saveFluxKleinSettingsFromPanel);
    picker.input.addEventListener("change", saveFluxKleinSettingsFromPanel);
  }
  for (const control of [fluxNotes, fluxPrompt, fluxWidth, fluxHeight, fluxSeed, fluxLoraCount]) {
    control.addEventListener("input", saveFluxKleinSettingsFromPanel);
    control.addEventListener("change", saveFluxKleinSettingsFromPanel);
  }
  useFluxKlein.input.addEventListener("change", saveFluxKleinSettingsFromPanel);
  fluxUseLora.input.addEventListener("change", saveFluxKleinSettingsFromPanel);
  for (const slot of fluxLoraSlots) {
    wireSearchablePicker(slot.picker, saveFluxKleinSettingsFromPanel);
    slot.strength.addEventListener("input", saveFluxKleinSettingsFromPanel);
    slot.strength.addEventListener("change", saveFluxKleinSettingsFromPanel);
  }
}

export function wireZEnhanceControls({
  saveZEnhanceSettingsFromPanel, upscaleEnhanceImage, zEnhanceAmount, zEnhanceButton, zEnhanceClipPicker,
  zEnhanceHeight, zEnhanceLoraCount, zEnhanceLoraSlots, zEnhanceSeed, zEnhanceSeedMode, zEnhanceUnetPicker,
  zEnhanceUseLora, zEnhanceVaePicker, zEnhanceWidth,
}) {
  zEnhanceButton.onclick = upscaleEnhanceImage;
  zEnhanceUseLora.input.addEventListener("change", saveZEnhanceSettingsFromPanel);
  for (const control of [zEnhanceWidth, zEnhanceHeight, zEnhanceSeed, zEnhanceSeedMode, zEnhanceAmount, zEnhanceLoraCount]) {
    control.addEventListener("input", saveZEnhanceSettingsFromPanel);
    control.addEventListener("change", saveZEnhanceSettingsFromPanel);
  }
  for (const picker of [zEnhanceUnetPicker, zEnhanceClipPicker, zEnhanceVaePicker]) {
    wireSearchablePicker(picker, saveZEnhanceSettingsFromPanel);
  }
  for (const slot of zEnhanceLoraSlots) {
    wireSearchablePicker(slot.picker, saveZEnhanceSettingsFromPanel);
    slot.strength.addEventListener("input", saveZEnhanceSettingsFromPanel);
    slot.strength.addEventListener("change", saveZEnhanceSettingsFromPanel);
  }
}

export function buildImageSettingsPanels({ makeEditImagePromptButton, shell }) {
  const zimageSettingsPanel = document.createElement("div");
  zimageSettingsPanel.style.cssText = "display:flex;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const zUnetPicker = makeSearchableLoraPicker("z_image_turbo_bf16.safetensors");
  const zClipPicker = makeSearchableLoraPicker("qwen_3_4b.safetensors");
  const zVaePicker = makeSearchableLoraPicker("ae.safetensors");
  const useSceneZImageSettings = makeCheckbox("Use custom ZImage settings for this scene", false);
  const zFirstTitle = document.createElement("div");
  zFirstTitle.textContent = "First pass (low res)";
  zFirstTitle.style.cssText = "font-size:12px;color:#f4f4f5;font-weight:900;";
  const zFirstGrid = document.createElement("div");
  zFirstGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  const zFirstWidth = makeInput("1280", "number");
  const zFirstHeight = makeInput("720", "number");
  zFirstGrid.append(makeField("Width", zFirstWidth), makeField("Height", zFirstHeight));
  const zSecondTitle = document.createElement("div");
  zSecondTitle.textContent = "2nd pass (upscale enhance)";
  zSecondTitle.style.cssText = "font-size:12px;color:#f4f4f5;font-weight:900;";
  const zSecondGrid = document.createElement("div");
  zSecondGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  const zSecondWidth = makeInput("1920", "number");
  const zSecondHeight = makeInput("1080", "number");
  zSecondGrid.append(makeField("Width", zSecondWidth), makeField("Height", zSecondHeight));
  const zSeed = makeInput("1", "number");
  const zSeedMode = makeSelect(["fixed", "randomize", "increment", "decrement"], "fixed");
  const zBatchSize = makeInput("1", "number");
  zBatchSize.min = "1";
  zBatchSize.max = "16";
  zBatchSize.step = "1";
  const zSeedGrid = document.createElement("div");
  zSeedGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  zSeedGrid.append(makeField("Seed", zSeed), makeField("Seed mode", zSeedMode));
  const zLoraCount = makeInput("0", "number");
  zLoraCount.min = "0";
  zLoraCount.max = "4";
  const zUseLora = makeCheckbox("Use LoRAs?", false);
  const zLoraPanel = document.createElement("div");
  zLoraPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const zLoraRows = document.createElement("div");
  zLoraRows.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const zLoraSlots = [];
  for (let slot = 1; slot <= 4; slot++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 76px 76px;gap:8px;";
    const picker = makeSearchableLoraPicker("[none]");
    const firstPassStrength = makeInput("0.5", "number");
    firstPassStrength.step = "0.01";
    const secondPassStrength = makeInput("1", "number");
    secondPassStrength.step = "0.01";
    row.append(makeField(`LoRA ${slot}`, picker.wrapper), makeField("Pass 1", firstPassStrength), makeField("Pass 2", secondPassStrength));
    zLoraRows.append(row);
    zLoraSlots.push({ row, picker, firstPassStrength, secondPassStrength });
  }
  const zUseImageToImage = makeCheckbox("Use image-to-image?", false);
  const zI2IPanel = document.createElement("div");
  zI2IPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const zI2ISlider = document.createElement("input");
  zI2ISlider.type = "range";
  zI2ISlider.min = "1";
  zI2ISlider.max = "8";
  zI2ISlider.step = "1";
  zI2ISlider.value = "5";
  zI2ISlider.style.cssText = "width:100%;accent-color:#22d3ee;";
  const zI2IHint = document.createElement("div");
  zI2IHint.textContent = "1 = more creative, 8 = more like original";
  zI2IHint.style.cssText = "font-size:11px;color:#a1a1aa;";
  const zI2IStartStep = makeInput("5", "number");
  zI2IStartStep.min = "1";
  zI2IStartStep.max = "8";
  zI2IStartStep.step = "1";
  const zI2IPath = makeInput("");
  zI2IPath.placeholder = "Image-to-image source path...";
  zI2IPath.style.display = "none";
  const zI2IDrop = document.createElement("div");
  zI2IDrop.textContent = "Drop an image here, or drag a scene image from the timeline.";
  zI2IDrop.style.cssText = "border:1px dashed #155e75;border-radius:6px;background:#020617;color:#bae6fd;padding:10px;text-align:center;font-size:12px;";
  const zI2IActions = document.createElement("div");
  zI2IActions.style.cssText = "display:grid;grid-template-columns:1fr;gap:8px;";
  const zI2ILoadButton = makeButton("Load I2I Image", "primary");
  zI2IActions.append(zI2ILoadButton);
  zLoraPanel.append(makeField("LoRA count", zLoraCount), zLoraRows);
  zI2IPanel.append(makeField("I2I similarity", zI2ISlider), zI2IHint, makeField("I2I start step", zI2IStartStep), zI2IPath, zI2IDrop, zI2IActions);
  const ernieImagePanel = document.createElement("div");
  ernieImagePanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const ernieUnetPicker = makeSearchableLoraPicker("ernie\\ernie-image-turbo.safetensors");
  const ernieClipPicker = makeSearchableLoraPicker("ministral-3-3b.safetensors");
  const ernieVaePicker = makeSearchableLoraPicker("flux\\flux2-vae.safetensors");
  const ernieWidth = makeInput("1280", "number");
  const ernieHeight = makeInput("720", "number");
  const ernieSeed = makeInput("1", "number");
  const ernieSeedMode = makeSelect(["fixed", "randomize", "increment", "decrement"], "fixed");
  const ernieBatchSize = makeInput("1", "number");
  ernieBatchSize.min = "1";
  ernieBatchSize.max = "16";
  ernieBatchSize.step = "1";
  const ernieGrid = document.createElement("div");
  ernieGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  ernieGrid.append(makeField("Width", ernieWidth), makeField("Height", ernieHeight), makeField("Seed", ernieSeed), makeField("Seed mode", ernieSeedMode));
  const ernieUseLora = makeCheckbox("Use LoRAs?", false);
  const ernieLoraPanel = document.createElement("div");
  ernieLoraPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const ernieLoraCount = makeInput("0", "number");
  ernieLoraCount.min = "0";
  ernieLoraCount.max = "4";
  const ernieLoraRows = document.createElement("div");
  ernieLoraRows.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const ernieLoraSlots = [];
  for (let slot = 1; slot <= 4; slot++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:1fr 84px;gap:8px;";
    const picker = makeSearchableLoraPicker("[none]");
    const strength = makeInput("1", "number");
    strength.step = "0.01";
    row.append(makeField(`LoRA ${slot}`, picker.wrapper), makeField("Strength", strength));
    ernieLoraRows.append(row);
    ernieLoraSlots.push({ row, picker, strength });
  }
  ernieLoraPanel.append(makeField("LoRA count", ernieLoraCount), ernieLoraRows);
  const ernieUseImageToImage = makeCheckbox("Use image-to-image?", false);
  const ernieI2IPanel = document.createElement("div");
  ernieI2IPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const ernieI2ISlider = document.createElement("input");
  ernieI2ISlider.type = "range";
  ernieI2ISlider.min = "1";
  ernieI2ISlider.max = "8";
  ernieI2ISlider.step = "1";
  ernieI2ISlider.value = "5";
  ernieI2ISlider.style.cssText = "width:100%;accent-color:#22d3ee;";
  const ernieI2IHint = document.createElement("div");
  ernieI2IHint.textContent = "1 = more creative, 8 = more like original";
  ernieI2IHint.style.cssText = "font-size:11px;color:#a1a1aa;";
  const ernieI2IStartStep = makeInput("5", "number");
  ernieI2IStartStep.min = "1";
  ernieI2IStartStep.max = "8";
  ernieI2IStartStep.step = "1";
  const ernieI2IPath = makeInput("");
  ernieI2IPath.placeholder = "Image-to-image source path...";
  ernieI2IPath.style.display = "none";
  const ernieI2IDrop = document.createElement("div");
  ernieI2IDrop.textContent = "Drop an image here, or drag a scene image from the timeline.";
  ernieI2IDrop.style.cssText = zI2IDrop.style.cssText;
  const ernieI2IActions = document.createElement("div");
  ernieI2IActions.style.cssText = "display:grid;grid-template-columns:1fr;gap:8px;";
  const ernieI2ILoadButton = makeButton("Load I2I Image", "primary");
  ernieI2IActions.append(ernieI2ILoadButton);
  ernieI2IPanel.append(makeField("I2I similarity", ernieI2ISlider), ernieI2IHint, makeField("I2I start step", ernieI2IStartStep), ernieI2IPath, ernieI2IDrop, ernieI2IActions);
  const krea2TwoPassPanel = document.createElement("div");
  krea2TwoPassPanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const krea2TwoPassUnetPicker = makeSearchableLoraPicker("krea2_turbo_fp8_scaled.safetensors");
  const krea2TwoPassClipPicker = makeSearchableLoraPicker("qwen3vl_4b_fp8_scaled.safetensors");
  const krea2TwoPassVaePicker = makeSearchableLoraPicker("qwen_image_vae.safetensors");
  const krea2TwoPassUseLora = makeCheckbox("Use LoRAs?", false);
  const krea2TwoPassLoraCount = makeInput("0", "number");
  krea2TwoPassLoraCount.min = "0";
  krea2TwoPassLoraCount.max = "4";
  const krea2TwoPassLoraPanel = document.createElement("div");
  krea2TwoPassLoraPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const krea2TwoPassLoraRows = document.createElement("div");
  krea2TwoPassLoraRows.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const krea2TwoPassLoraSlots = [];
  for (let slot = 1; slot <= 4; slot++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 76px 76px;gap:8px;";
    const picker = makeSearchableLoraPicker("[none]");
    const firstPassStrength = makeInput("0.5", "number");
    firstPassStrength.step = "0.01";
    const secondPassStrength = makeInput("0", "number");
    secondPassStrength.step = "0.01";
    row.append(makeField(`LoRA ${slot}`, picker.wrapper), makeField("Pass 1", firstPassStrength), makeField("Pass 2", secondPassStrength));
    krea2TwoPassLoraRows.append(row);
    krea2TwoPassLoraSlots.push({ row, picker, firstPassStrength, secondPassStrength });
  }
  krea2TwoPassLoraPanel.append(makeField("LoRA count", krea2TwoPassLoraCount), krea2TwoPassLoraRows);
  const krea2TwoPassAspectRatio = makeSelect([
    "16:9 (Widescreen)",
    "9:16 (Portrait)",
    "1:1 (Square)",
    "4:3 (Landscape)",
    "3:4 (Portrait)",
    "3:2 (Landscape)",
    "2:3 (Portrait)",
    "21:9 (Cinematic)",
  ], "16:9 (Widescreen)");
  const krea2TwoPassSampler = makeSelect([
    "euler_ancestral_cfg_pp",
    "euler",
    "euler_ancestral",
    "heun",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_2m",
    "ddim",
    "uni_pc",
  ], "euler_ancestral_cfg_pp");
  const krea2TwoPassSeed = makeInput("1", "number");
  const krea2TwoPassSeedMode = makeSelect(["fixed", "randomize", "increment", "decrement"], "fixed");
  const krea2TwoPassCfg = makeInput("1.2", "number");
  krea2TwoPassCfg.min = "1";
  krea2TwoPassCfg.max = "1.2";
  krea2TwoPassCfg.step = "0.01";
  const krea2TwoPassCfgField = makeField("CFG", krea2TwoPassCfg);
  const krea2TwoPassCfgNote = document.createElement("div");
  krea2TwoPassCfgNote.textContent = "Use 1-1.2. Recommended: 1.2.";
  krea2TwoPassCfgNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;margin-top:3px;";
  krea2TwoPassCfgField.append(krea2TwoPassCfgNote);
  const krea2TwoPassBatchSize = makeInput("1", "number");
  krea2TwoPassBatchSize.min = "1";
  krea2TwoPassBatchSize.max = "16";
  krea2TwoPassBatchSize.step = "1";
  const krea2TwoPassSettingsGrid = document.createElement("div");
  krea2TwoPassSettingsGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  krea2TwoPassSettingsGrid.append(
    makeField("Aspect ratio", krea2TwoPassAspectRatio),
    makeField("Sampler", krea2TwoPassSampler),
    makeField("Seed", krea2TwoPassSeed),
    makeField("Seed mode", krea2TwoPassSeedMode),
    krea2TwoPassCfgField,
    makeField("Batch size", krea2TwoPassBatchSize)
  );
  const krea2TwoPassUseImageToImage = makeCheckbox("Use image-to-image?", false);
  const krea2TwoPassI2IPanel = document.createElement("div");
  krea2TwoPassI2IPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const krea2TwoPassCreativity = document.createElement("input");
  krea2TwoPassCreativity.type = "range";
  krea2TwoPassCreativity.min = "0";
  krea2TwoPassCreativity.max = "10";
  krea2TwoPassCreativity.step = "1";
  krea2TwoPassCreativity.value = "5";
  krea2TwoPassCreativity.style.cssText = "width:100%;accent-color:#22d3ee;";
  const krea2TwoPassCreativityInput = makeInput("5", "number");
  krea2TwoPassCreativityInput.min = "0";
  krea2TwoPassCreativityInput.max = "10";
  krea2TwoPassCreativityInput.step = "1";
  const krea2TwoPassI2IHint = document.createElement("div");
  krea2TwoPassI2IHint.textContent = "0 ignores the image. 10 keeps the original image most intact while still changing it based on the prompt.";
  krea2TwoPassI2IHint.style.cssText = "font-size:11px;color:#a1a1aa;";
  const krea2TwoPassI2IPath = makeInput("");
  krea2TwoPassI2IPath.placeholder = "Image-to-image source path...";
  krea2TwoPassI2IPath.style.display = "none";
  const krea2TwoPassI2IDrop = document.createElement("div");
  krea2TwoPassI2IDrop.textContent = "Drop an image here, or drag a scene image from the timeline.";
  krea2TwoPassI2IDrop.style.cssText = zI2IDrop.style.cssText;
  const krea2TwoPassI2IActions = document.createElement("div");
  krea2TwoPassI2IActions.style.cssText = "display:grid;grid-template-columns:1fr;gap:8px;";
  const krea2TwoPassI2ILoadButton = makeButton("Load I2I Image", "primary");
  krea2TwoPassI2IActions.append(krea2TwoPassI2ILoadButton);
  krea2TwoPassI2IPanel.append(makeField("I2I creativity", krea2TwoPassCreativity), krea2TwoPassI2IHint, makeField("Creativity value", krea2TwoPassCreativityInput), krea2TwoPassI2IPath, krea2TwoPassI2IDrop, krea2TwoPassI2IActions);
  const fluxKleinPanel = document.createElement("div");
  fluxKleinPanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const useFluxKlein = makeCheckbox("Build image using Flux/Klein?", false);
  const fluxIngredientFileInput = document.createElement("input");
  fluxIngredientFileInput.type = "file";
  fluxIngredientFileInput.accept = "image/png,image/jpeg,image/webp";
  fluxIngredientFileInput.multiple = true;
  fluxIngredientFileInput.style.display = "none";
  shell.append(fluxIngredientFileInput);
  const fluxGlobalIngredientFileInput = document.createElement("input");
  fluxGlobalIngredientFileInput.type = "file";
  fluxGlobalIngredientFileInput.accept = "image/png,image/jpeg,image/webp";
  fluxGlobalIngredientFileInput.multiple = true;
  fluxGlobalIngredientFileInput.style.display = "none";
  shell.append(fluxGlobalIngredientFileInput);
  const useFluxGlobalIngredients = makeCheckbox("Use global image ingredients", false);
  const fluxGlobalIngredientPanel = document.createElement("div");
  fluxGlobalIngredientPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const fluxGlobalIngredientDrop = document.createElement("div");
  fluxGlobalIngredientDrop.innerHTML = `<b>Global image ingredients</b><br><span>Drop character, face, costume, or style references here to use them in every Flux/Klein scene.</span>`;
  fluxGlobalIngredientDrop.style.cssText = "border:1px dashed #7c3aed;border-radius:6px;background:#12091f;color:#ddd6fe;padding:12px;text-align:center;font-size:12px;line-height:1.45;";
  const fluxGlobalIngredientList = document.createElement("div");
  fluxGlobalIngredientList.style.cssText = "display:flex;flex-direction:column;gap:6px;";
  const fluxGlobalIngredientActions = document.createElement("div");
  fluxGlobalIngredientActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  const fluxGlobalIngredientButton = makeButton("Upload Global Images", "primary");
  const fluxGlobalIngredientClearButton = makeButton("Clear Globals");
  fluxGlobalIngredientActions.append(fluxGlobalIngredientButton, fluxGlobalIngredientClearButton);
  fluxGlobalIngredientPanel.append(
    fluxGlobalIngredientDrop,
    fluxGlobalIngredientActions,
    fluxGlobalIngredientList,
  );
  const nbUseGlobalIngredients = makeCheckbox("Use global Nano B reference images", false);
  const nbGlobalIngredientPanel = document.createElement("div");
  nbGlobalIngredientPanel.style.cssText = fluxGlobalIngredientPanel.style.cssText;
  const nbGlobalIngredientDrop = document.createElement("div");
  nbGlobalIngredientDrop.innerHTML = `<b>Global Nano B reference images</b><br><span>Drop character, face, costume, or style references here to use them in every Nano B scene.</span>`;
  nbGlobalIngredientDrop.style.cssText = fluxGlobalIngredientDrop.style.cssText;
  const nbGlobalIngredientList = document.createElement("div");
  nbGlobalIngredientList.style.cssText = fluxGlobalIngredientList.style.cssText;
  const nbGlobalIngredientActions = document.createElement("div");
  nbGlobalIngredientActions.style.cssText = fluxGlobalIngredientActions.style.cssText;
  const nbGlobalIngredientButton = makeButton("Upload Global References", "primary");
  const nbGlobalIngredientClearButton = makeButton("Clear Globals");
  nbGlobalIngredientActions.append(nbGlobalIngredientButton, nbGlobalIngredientClearButton);
  nbGlobalIngredientPanel.append(
    nbGlobalIngredientDrop,
    nbGlobalIngredientActions,
    nbGlobalIngredientList,
  );
  const fluxIngredientDrop = document.createElement("div");
  fluxIngredientDrop.innerHTML = `<b>Image ingredients</b><br><span>Drop images here: character, background, props, style references, or anything else Flux/Klein should use.</span>`;
  fluxIngredientDrop.style.cssText = "border:1px dashed #155e75;border-radius:6px;background:#020617;color:#bae6fd;padding:12px;text-align:center;font-size:12px;line-height:1.45;";
  const fluxIngredientList = document.createElement("div");
  fluxIngredientList.style.cssText = "display:flex;flex-direction:column;gap:6px;";
  const fluxIngredientActions = document.createElement("div");
  fluxIngredientActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  const fluxIngredientButton = makeButton("Upload Images", "primary");
  const fluxIngredientClearButton = makeButton("Clear Images");
  fluxIngredientActions.append(fluxIngredientButton, fluxIngredientClearButton);
  const fluxNotes = document.createElement("textarea");
  fluxNotes.placeholder = "Optional pose, camera, wardrobe, lighting, or mood notes...";
  fluxNotes.style.cssText = "width:100%;box-sizing:border-box;min-height:72px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:9px;font-size:12px;line-height:1.45;";
  const fluxUseTextOnlyGemmaPrompt = makeCheckbox("Use text-only LLM for Flux prompts", false);
  const fluxUseDirectorNotes = makeCheckbox("Use Director Notes in Flux prompt", false);
  const fluxGemmaModelSelect = makeSelect([""], "");
  const fluxMmprojSelect = makeSelect([""], "");
  const fluxPrompt = document.createElement("textarea");
  fluxPrompt.placeholder = "Flux/Klein prompt...";
  fluxPrompt.style.cssText = fluxNotes.style.cssText;
  const fluxUnetPicker = makeSearchableLoraPicker("flux\\flux-2-klein-4b-fp8.safetensors");
  const fluxClipPicker = makeSearchableLoraPicker("qwen_3_4b.safetensors");
  const fluxVaePicker = makeSearchableLoraPicker("flux\\flux2-vae.safetensors");
  const fluxUseLora = makeCheckbox("Use Flux/Klein LoRAs?", false);
  const fluxLoraPanel = document.createElement("div");
  fluxLoraPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const fluxLoraCount = makeInput("0", "number");
  fluxLoraCount.min = "0";
  fluxLoraCount.max = "4";
  const fluxLoraRows = document.createElement("div");
  fluxLoraRows.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const fluxLoraSlots = [];
  for (let slot = 1; slot <= 4; slot++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 84px;gap:8px;";
    const picker = makeSearchableLoraPicker("[none]");
    const strength = makeInput("1", "number");
    strength.step = "0.01";
    row.append(makeField(`LoRA ${slot}`, picker.wrapper), makeField("Strength", strength));
    fluxLoraRows.append(row);
    fluxLoraSlots.push({ row, picker, strength });
  }
  fluxLoraPanel.append(makeField("LoRA count", fluxLoraCount), fluxLoraRows);
  const fluxWidth = makeInput("1024", "number");
  const fluxHeight = makeInput("576", "number");
  const fluxSeed = makeInput("100", "number");
  const fluxGrid = document.createElement("div");
  fluxGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  fluxGrid.append(makeField("Width", fluxWidth), makeField("Height", fluxHeight), makeField("Seed", fluxSeed));
  const createFluxPromptButton = makeButton("Gemma Flux Prompt", "primary");
  const editFluxKleinT2IInstructionsButton = makeButton("Edit Flux/Klein T2I Instructions");
  const editFluxPromptButton = makeEditImagePromptButton();
  const previewFluxButton = makeButton("Create with Flux/Klein", "primary");
  const sendFluxPromptToEnhanceButton = makeMiniButton("Send to Enhance");
  const nbImagePanel = document.createElement("div");
  nbImagePanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const nbApiKey = makeInput("");
  nbApiKey.type = "password";
  nbApiKey.placeholder = "NanoBanana API key...";
  const nbModelSelect = makeSelect(NB_IMAGE_MODELS, DEFAULT_NB_IMAGE_MODEL);
  const nbGemmaModelSelect = makeSelect([""], "");
  const nbMmprojSelect = makeSelect([""], "");
  const nbUseTextOnlyGemmaPrompt = makeCheckbox("Use text-only LLM for Nano B prompts", false);
  const nbUseDirectorNotes = makeCheckbox("Use Director Notes in Nano B prompt", false);
  const nbNotes = document.createElement("textarea");
  nbNotes.placeholder = "Optional camera, framing, pose, scene, or edit notes for NanoBanana...";
  nbNotes.style.cssText = fluxNotes.style.cssText;
  const nbPrompt = document.createElement("textarea");
  nbPrompt.placeholder = "NanoBanana prompt...";
  nbPrompt.style.cssText = fluxPrompt.style.cssText;
  const nbIngredientDrop = document.createElement("div");
  nbIngredientDrop.innerHTML = `<b>NanoBanana reference images</b><br><span>Drop character and scene references here. Reference Builder images are also included when enabled for this scene.</span>`;
  nbIngredientDrop.style.cssText = fluxIngredientDrop.style.cssText;
  const nbIngredientList = document.createElement("div");
  nbIngredientList.style.cssText = fluxIngredientList.style.cssText;
  const nbIngredientActions = document.createElement("div");
  nbIngredientActions.style.cssText = fluxIngredientActions.style.cssText;
  const nbIngredientButton = makeButton("Upload References", "primary");
  const nbIngredientClearButton = makeButton("Clear References");
  nbIngredientActions.append(nbIngredientButton, nbIngredientClearButton);
  const createNBPromptButton = makeButton("Gemma NB Prompt", "primary");
  const editNanoBT2IInstructionsButton = makeButton("Edit Nano B T2I Instructions");
  const editNBPromptButton = makeEditImagePromptButton();
  const previewNBButton = makeButton("Create with NanoBanana", "primary");
  const flowGptModePanel = document.createElement("div");
  flowGptModePanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const flowGptProviderRow = document.createElement("div");
  flowGptProviderRow.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;";
  const flowNanoProviderButton = makeButton("Flow Nano Banana", "primary");
  const gptImageProviderButton = makeButton("GPT Image", "primary");
  const metaImageProviderButton = makeButton("Meta AI", "primary");
  flowGptProviderRow.append(flowNanoProviderButton, gptImageProviderButton, metaImageProviderButton);
  const flowGptSetupNote = document.createElement("div");
  flowGptSetupNote.style.cssText = "font-size:11px;color:#d4d4d8;line-height:1.45;border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:9px;";
  flowGptSetupNote.textContent = "Browser image providers use real Chrome/Chromium profiles. Install automation first, choose a provider, then open that provider login before long runs. Linux/macOS require system Node.js/npm; Windows can use portable Node. Flow prompts are sent without appended aspect-ratio text and Flow must be manually set to 1 image/aspect ratio. GPT Image appends the selected aspect ratio. Meta AI uses meta.ai, supports reference uploads, and requires login before generation/download. If you use fallback mode, log into every provider you plan to allow.";
  const flowGptStatusText = document.createElement("div");
  flowGptStatusText.style.cssText = "font-size:11px;color:#a1a1aa;white-space:pre-wrap;line-height:1.35;";
  flowGptStatusText.textContent = "Browser automation status has not been checked yet.";
  const flowGptSetupButton = makeButton("Install Browser Automation", "primary");
  const flowGptStatusButton = makeButton("Check Browser Setup");
  const flowGptLoginButton = makeButton("Open Selected Login");
  const flowGptSetupActions = document.createElement("div");
  flowGptSetupActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  flowGptSetupActions.append(flowGptSetupButton, flowGptStatusButton, flowGptLoginButton);
  const flowGptAspectRatio = makeSelect(["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], "16:9");
  const flowGptTimeout = makeInput("600", "number");
  flowGptTimeout.min = "60";
  flowGptTimeout.max = "2400";
  flowGptTimeout.step = "10";
  const flowGptRetries = makeInput("10", "number");
  flowGptRetries.min = "1";
  flowGptRetries.max = "20";
  flowGptRetries.step = "1";
  const flowGptFailureMode = makeSelect(["last_successful_image", "try_other_provider", "stop"], "last_successful_image");
  for (const option of flowGptFailureMode.options) {
    if (option.value === "last_successful_image") option.textContent = "Use last successful image";
    else if (option.value === "try_other_provider") option.textContent = "Try other provider first";
    else if (option.value === "stop") option.textContent = "Stop";
  }
  const flowGptAskPreviousImage = makeCheckbox("Send previous scene's FLF end frame as context", false);
  const flowGptPrompt = document.createElement("textarea");
  flowGptPrompt.placeholder = "Browser AI image prompt...";
  flowGptPrompt.style.cssText = "width:100%;box-sizing:border-box;min-height:92px;resize:vertical;border:1px solid #27272a;border-radius:6px;background:#18181b;color:#d4d4d8;padding:8px;font-size:11px;line-height:1.35;";
  ["keydown", "keypress", "keyup"].forEach((eventName) => {
    flowGptPrompt.addEventListener(eventName, (event) => {
      event.stopPropagation();
    });
  });
  const flowGptAspectRatioField = makeField("GPT aspect ratio", flowGptAspectRatio);
  const flowGptCreatePromptButton = makeButton("Gemma Browser Prompt", "primary");
  const editFlowGptT2IInstructionsButton = makeButton("Edit Browser T2I Instructions");
  const flowGptCreateImageButton = makeButton("Create with Browser AI", "primary");
  const flowGptManualMode = makeCheckbox("Manual Mode", false);
  const flowGptManualAutoAdvance = makeCheckbox("Auto-advance after import", false);
  const flowGptManualChatPrompt = document.createElement("textarea");
  flowGptManualChatPrompt.placeholder = "Prompt to copy into the browser after exporting reference images...";
  flowGptManualChatPrompt.style.cssText = "width:100%;box-sizing:border-box;min-height:120px;resize:vertical;border:1px solid #27272a;border-radius:6px;background:#18181b;color:#d4d4d8;padding:8px;font-size:11px;line-height:1.35;";
  ["keydown", "keypress", "keyup"].forEach((eventName) => {
    flowGptManualChatPrompt.addEventListener(eventName, (event) => event.stopPropagation());
  });
  const flowGptManualOpenButton = makeButton("Open Manual Browser", "primary");
  const flowGptManualExportRefsButton = makeButton("Export Scene Refs");
  const flowGptManualImportLatestButton = makeButton("Import Latest Download");
  const flowGptManualStatus = document.createElement("div");
  flowGptManualStatus.style.cssText = "font-size:11px;color:#a1a1aa;white-space:pre-wrap;line-height:1.35;";
  flowGptManualStatus.textContent = "Manual Mode is off.";
  const flowGptManualActions = document.createElement("div");
  flowGptManualActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  flowGptManualActions.append(flowGptManualOpenButton, flowGptManualExportRefsButton, flowGptManualImportLatestButton);
  const browserAiGroupsNote = document.createElement("div");
  browserAiGroupsNote.style.cssText = "font-size:11px;color:#d4d4d8;line-height:1.45;border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:9px;";
  browserAiGroupsNote.textContent = "Band Sequence mode lets you add the singer, optional extras, the other band members, and every location once. It derives Singer Only, optional Singer + Extras, Other Members Only, and Full Band sets for each location. Requests reuse one provider tab, and you decide when to send each next set after reviewing and downloading. Turn Band Sequence off to use custom groups instead. This is project-wide and is not tied to the active scene.";
  const browserAiBandSequenceMode = makeCheckbox("Band Sequence mode", false);
  const browserAiGroupSelect = makeSelect([], "");
  const browserAiAutoAdvanceGroup = makeCheckbox("Auto-select next group/set after sending", true);
  const browserAiNewGroupButton = makeButton("New Group");
  const browserAiDuplicateGroupButton = makeButton("Duplicate");
  const browserAiRenameGroupButton = makeButton("Rename");
  const browserAiDeleteGroupButton = makeButton("Delete");
  const browserAiGroupActions = document.createElement("div");
  browserAiGroupActions.style.cssText = "display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;";
  browserAiGroupActions.append(browserAiNewGroupButton, browserAiDuplicateGroupButton, browserAiRenameGroupButton, browserAiDeleteGroupButton);
  const browserAiGroupDrop = document.createElement("div");
  browserAiGroupDrop.dataset.vrgdgFileDropZone = "true";
  browserAiGroupDrop.innerHTML = "<b>Character / band references</b><br><span>Drop any number of images for the selected group.</span>";
  browserAiGroupDrop.style.cssText = "border:1px dashed #0891b2;border-radius:6px;background:#082f49;color:#cffafe;padding:14px;text-align:center;font-size:11px;line-height:1.45;cursor:pointer;";
  const browserAiGroupList = document.createElement("div");
  browserAiGroupList.style.cssText = "display:flex;flex-direction:column;gap:5px;";
  const browserAiAddGroupImagesButton = makeButton("Add Group Images", "primary");
  const browserAiClearGroupImagesButton = makeButton("Clear Group Images");
  const browserAiGroupImageActions = document.createElement("div");
  browserAiGroupImageActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:6px;";
  browserAiGroupImageActions.append(browserAiAddGroupImagesButton, browserAiClearGroupImagesButton);
  const browserAiLocationDrop = document.createElement("div");
  browserAiLocationDrop.dataset.vrgdgFileDropZone = "true";
  browserAiLocationDrop.innerHTML = "<b>Location reference</b><br><span>Drop one location image. Replace it whenever the location changes.</span>";
  browserAiLocationDrop.style.cssText = "border:1px dashed #7c3aed;border-radius:6px;background:#2e1065;color:#ede9fe;padding:14px;text-align:center;font-size:11px;line-height:1.45;cursor:pointer;";
  const browserAiLocationList = document.createElement("div");
  browserAiLocationList.style.cssText = browserAiGroupList.style.cssText;
  const browserAiChooseLocationButton = makeButton("Choose Location", "primary");
  const browserAiClearLocationButton = makeButton("Clear Location");
  const browserAiLocationActions = document.createElement("div");
  browserAiLocationActions.style.cssText = browserAiGroupImageActions.style.cssText;
  browserAiLocationActions.append(browserAiChooseLocationButton, browserAiClearLocationButton);
  const browserAiSingerDrop = document.createElement("div");
  browserAiSingerDrop.dataset.vrgdgFileDropZone = "true";
  browserAiSingerDrop.innerHTML = "<b>Singer references</b><br><span>Drop the singer's reference sheets or images.</span>";
  browserAiSingerDrop.style.cssText = browserAiGroupDrop.style.cssText;
  const browserAiSingerList = document.createElement("div");
  browserAiSingerList.style.cssText = browserAiGroupList.style.cssText;
  const browserAiAddSingerButton = makeButton("Add Singer References", "primary");
  const browserAiClearSingerButton = makeButton("Clear Singer");
  const browserAiSingerActions = document.createElement("div");
  browserAiSingerActions.style.cssText = browserAiGroupImageActions.style.cssText;
  browserAiSingerActions.append(browserAiAddSingerButton, browserAiClearSingerButton);
  const browserAiExtrasDrop = document.createElement("div");
  browserAiExtrasDrop.dataset.vrgdgFileDropZone = "true";
  browserAiExtrasDrop.innerHTML = "<b>Extras (optional)</b><br><span>Drop non-band characters who should appear with the singer.</span>";
  browserAiExtrasDrop.style.cssText = browserAiGroupDrop.style.cssText;
  const browserAiExtrasList = document.createElement("div");
  browserAiExtrasList.style.cssText = browserAiGroupList.style.cssText;
  const browserAiAddExtrasButton = makeButton("Add Extras", "primary");
  const browserAiClearExtrasButton = makeButton("Clear Extras");
  const browserAiExtrasActions = document.createElement("div");
  browserAiExtrasActions.style.cssText = browserAiGroupImageActions.style.cssText;
  browserAiExtrasActions.append(browserAiAddExtrasButton, browserAiClearExtrasButton);
  const browserAiSingerPanel = document.createElement("div");
  browserAiSingerPanel.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:0;";
  browserAiSingerPanel.append(browserAiSingerDrop, browserAiSingerActions, browserAiSingerList);
  const browserAiExtrasPanel = document.createElement("div");
  browserAiExtrasPanel.style.cssText = browserAiSingerPanel.style.cssText;
  browserAiExtrasPanel.append(browserAiExtrasDrop, browserAiExtrasActions, browserAiExtrasList);
  const browserAiSingerExtrasGrid = document.createElement("div");
  browserAiSingerExtrasGrid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;align-items:start;";
  browserAiSingerExtrasGrid.append(browserAiSingerPanel, browserAiExtrasPanel);
  const browserAiMembersDrop = document.createElement("div");
  browserAiMembersDrop.dataset.vrgdgFileDropZone = "true";
  browserAiMembersDrop.innerHTML = "<b>Other band member references</b><br><span>Drop everyone except the singer.</span>";
  browserAiMembersDrop.style.cssText = browserAiGroupDrop.style.cssText;
  const browserAiMembersList = document.createElement("div");
  browserAiMembersList.style.cssText = browserAiGroupList.style.cssText;
  const browserAiAddMembersButton = makeButton("Add Other Members", "primary");
  const browserAiClearMembersButton = makeButton("Clear Other Members");
  const browserAiMembersActions = document.createElement("div");
  browserAiMembersActions.style.cssText = browserAiGroupImageActions.style.cssText;
  browserAiMembersActions.append(browserAiAddMembersButton, browserAiClearMembersButton);
  const browserAiLocationsDrop = document.createElement("div");
  browserAiLocationsDrop.dataset.vrgdgFileDropZone = "true";
  browserAiLocationsDrop.innerHTML = "<b>All location references</b><br><span>Drop every location once. Each location will run through every available subject set.</span>";
  browserAiLocationsDrop.style.cssText = browserAiLocationDrop.style.cssText;
  const browserAiLocationsList = document.createElement("div");
  browserAiLocationsList.style.cssText = browserAiGroupList.style.cssText;
  const browserAiAddLocationsButton = makeButton("Add Locations", "primary");
  const browserAiClearLocationsButton = makeButton("Clear Locations");
  const browserAiLocationsActions = document.createElement("div");
  browserAiLocationsActions.style.cssText = browserAiGroupImageActions.style.cssText;
  browserAiLocationsActions.append(browserAiAddLocationsButton, browserAiClearLocationsButton);
  const browserAiSequenceLocationSelect = makeSelect([], "");
  const browserAiSequenceSetSelect = makeSelect([
    { value: "0", label: "1. Singer only" },
  ], "0");
  const browserAiSequenceProgress = document.createElement("div");
  browserAiSequenceProgress.style.cssText = "font-size:11px;color:#bae6fd;border:1px solid #155e75;border-radius:6px;background:#082f49;padding:8px;line-height:1.4;";
  const browserAiBandSequencePanel = document.createElement("div");
  browserAiBandSequencePanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  browserAiBandSequencePanel.append(
    browserAiSingerExtrasGrid,
    browserAiMembersDrop,
    browserAiMembersActions,
    browserAiMembersList,
    browserAiLocationsDrop,
    browserAiLocationsActions,
    browserAiLocationsList,
    makeField("Current location", browserAiSequenceLocationSelect),
    makeField("Current subject set", browserAiSequenceSetSelect),
    browserAiSequenceProgress,
  );
  const browserAiCustomGroupsPanel = document.createElement("div");
  browserAiCustomGroupsPanel.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  browserAiCustomGroupsPanel.append(
    makeField("Reference group", browserAiGroupSelect),
    browserAiGroupActions,
    browserAiGroupDrop,
    browserAiGroupImageActions,
    browserAiGroupList,
    browserAiLocationDrop,
    browserAiLocationActions,
    browserAiLocationList,
  );
  const browserAiGroupPrompt = document.createElement("textarea");
  browserAiGroupPrompt.placeholder = "Prompt sent with the selected reference group and location...";
  browserAiGroupPrompt.style.cssText = flowGptManualChatPrompt.style.cssText;
  ["keydown", "keypress", "keyup"].forEach((eventName) => {
    browserAiGroupPrompt.addEventListener(eventName, (event) => event.stopPropagation());
  });
  const browserAiSendButton = makeButton("Send Selected Group", "primary");
  const browserAiFinishButton = makeButton("Finish Session + Restore Downloads");
  browserAiFinishButton.disabled = true;
  const browserAiSessionActions = document.createElement("div");
  browserAiSessionActions.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;";
  browserAiSessionActions.append(browserAiSendButton, browserAiFinishButton);
  const browserAiGroupStatus = document.createElement("div");
  browserAiGroupStatus.style.cssText = "font-size:11px;color:#a1a1aa;white-space:pre-wrap;line-height:1.4;";
  browserAiGroupStatus.textContent = "Choose a group, add references, and send when ready.";
  const browserAiDownloadOverrideProviders = new Set();
  const sendNBPromptToEnhanceButton = makeMiniButton("Send to Enhance");
  const fluxImageRefsPanel = document.createElement("div");
  fluxImageRefsPanel.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  fluxImageRefsPanel.append(
    useFluxGlobalIngredients.wrapper,
    fluxGlobalIngredientPanel,
    fluxIngredientDrop,
    fluxIngredientActions,
    fluxIngredientList,
  );
  const zEnhancePanel = document.createElement("div");
  zEnhancePanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const zEnhanceTitle = document.createElement("div");
  zEnhanceTitle.textContent = "Upscale / Enhance selected image";
  zEnhanceTitle.style.cssText = "font-size:12px;color:#f4f4f5;font-weight:900;";
  const zEnhancePromptPreview = document.createElement("textarea");
  zEnhancePromptPreview.placeholder = "Enhance prompt copied from the selected scene...";
  zEnhancePromptPreview.style.cssText = "width:100%;box-sizing:border-box;min-height:92px;resize:vertical;border:1px solid #27272a;border-radius:6px;background:#18181b;color:#d4d4d8;padding:8px;font-size:11px;line-height:1.35;";
  const zEnhanceGemmaNotes = document.createElement("textarea");
  zEnhanceGemmaNotes.placeholder = "Optional notes for Gemma: what to preserve, improve, restyle, or emphasize from the selected image...";
  zEnhanceGemmaNotes.style.cssText = zEnhancePromptPreview.style.cssText;
  const zEnhanceGemmaModelSelect = makeSelect([""], "");
  const zEnhanceMmprojSelect = makeSelect([""], "");
  const zEnhanceGemmaButton = makeButton("Gemma Enhance Prompt", "primary");
  const zEnhanceUnetPicker = makeSearchableLoraPicker("z_image_turbo_bf16.safetensors");
  const zEnhanceClipPicker = makeSearchableLoraPicker("qwen_3_4b.safetensors");
  const zEnhanceVaePicker = makeSearchableLoraPicker("ae.safetensors");
  const zEnhanceWidth = makeInput("1920", "number");
  const zEnhanceHeight = makeInput("1080", "number");
  const zEnhanceSeed = makeInput("1", "number");
  const zEnhanceSeedMode = makeSelect(["fixed", "randomize", "increment", "decrement"], "randomize");
  const zEnhanceAmount = document.createElement("input");
  zEnhanceAmount.type = "range";
  zEnhanceAmount.min = "1";
  zEnhanceAmount.max = "20";
  zEnhanceAmount.step = "1";
  zEnhanceAmount.value = "8";
  zEnhanceAmount.style.cssText = "width:100%;accent-color:#22d3ee;";
  const zEnhanceAmountValue = document.createElement("div");
  zEnhanceAmountValue.style.cssText = "font-size:11px;color:#a1a1aa;";
  const zEnhanceHint = document.createElement("div");
  zEnhanceHint.textContent = "Higher values keep closer to the original. Lower values are more creative. Adding a ZImage character LoRA can act like a face swap plus enhancement.";
  zEnhanceHint.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;";
  const zEnhanceGrid = document.createElement("div");
  zEnhanceGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  zEnhanceGrid.append(makeField("Width", zEnhanceWidth), makeField("Height", zEnhanceHeight), makeField("Seed", zEnhanceSeed), makeField("Seed mode", zEnhanceSeedMode));
  const zEnhanceUseLora = makeCheckbox("Use LoRAs?", false);
  const zEnhanceLoraPanel = document.createElement("div");
  zEnhanceLoraPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const zEnhanceLoraCount = makeInput("0", "number");
  zEnhanceLoraCount.min = "0";
  zEnhanceLoraCount.max = "4";
  const zEnhanceLoraRows = document.createElement("div");
  zEnhanceLoraRows.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const zEnhanceLoraSlots = [];
  for (let slot = 1; slot <= 4; slot++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:1fr 84px;gap:8px;";
    const picker = makeSearchableLoraPicker("[none]");
    const strength = makeInput("1", "number");
    strength.step = "0.01";
    row.append(makeField(`Enhance LoRA ${slot}`, picker.wrapper), makeField("Strength", strength));
    zEnhanceLoraRows.append(row);
    zEnhanceLoraSlots.push({ row, picker, strength });
  }
  zEnhanceLoraPanel.append(makeField("LoRA count", zEnhanceLoraCount), zEnhanceLoraRows);
  const zEnhanceButton = makeButton("Upscale / Enhance Image", "primary");

  return {
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
  };
}

export function buildImageModeCards({
  makeEditImagePromptButton, syncKrea2TwoPassLlmSelectsFromShared, syncKrea2TwoPassLlmSelectsToShared,
  updateI2VPromptSaveButtonState, zI2IDrop,
}) {
  const imageModelChooserWrap = document.createElement("div");
  imageModelChooserWrap.style.cssText = "display:flex;flex-direction:column;gap:5px;min-width:0;";
  const imageModelChooserLabel = document.createElement("div");
  imageModelChooserLabel.textContent = "Model";
  imageModelChooserLabel.style.cssText = "font-size:11px;font-weight:900;color:#bae6fd;letter-spacing:0;";
  const imageModelChooser = document.createElement("div");
  imageModelChooser.style.cssText = "display:flex;gap:6px;overflow-x:auto;overflow-y:hidden;max-width:100%;padding:0 0 3px;scrollbar-width:thin;";
  const zImageCard = makeImageModelCard("ZImage", "zimage");
  const fluxKleinCard = makeImageModelCard("Flux Klein", "flux_klein");
  const nbImageCard = makeImageModelCard("Nano B", "nano_banana");
  const ernieImageCard = makeImageModelCard("Ernie", "ernie_image");
  const krea2TwoPassCard = makeImageModelCard("Krea 2", "krea2_2pass");
  const flowGptCard = makeImageModelCard("Browser AI", "flow_gpt");
  const zEnhanceCard = makeImageModelCard("Enhance", "z_enhance");
  const loadCustomImageButton = makeImageModelCard("+ Custom", "custom_image");
  loadCustomImageButton.title = "Load a custom image for the selected scene";
  imageModelChooser.append(zImageCard, fluxKleinCard, nbImageCard, ernieImageCard, krea2TwoPassCard, flowGptCard, zEnhanceCard, loadCustomImageButton);
  imageModelChooserWrap.append(imageModelChooserLabel, imageModelChooser);
  const zImageModePanel = document.createElement("div");
  zImageModePanel.style.cssText = "display:flex;flex-direction:column;gap:10px;";
  const fluxKleinModePanel = document.createElement("div");
  fluxKleinModePanel.style.cssText = "display:none;flex-direction:column;gap:10px;";
  const ernieImageModePanel = document.createElement("div");
  ernieImageModePanel.style.cssText = "display:none;flex-direction:column;gap:10px;";
  const krea2TwoPassModePanel = document.createElement("div");
  krea2TwoPassModePanel.style.cssText = "display:none;flex-direction:column;gap:10px;";
  const startInput = makeInput("0", "number");
  startInput.step = "0.01";
  const endInput = makeInput("4", "number");
  endInput.step = "0.01";
  const notesInput = document.createElement("textarea");
  notesInput.placeholder = "Scene notes for Gemma / prompt direction...";
  notesInput.style.cssText = "width:100%;box-sizing:border-box;min-height:82px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:9px;font-size:12px;line-height:1.45;";
  const i2vNotesInput = document.createElement("textarea");
  i2vNotesInput.placeholder = "Extra video motion notes, camera movement, character movement...";
  i2vNotesInput.style.cssText = notesInput.style.cssText;
  const flfTransitionTypeSelect = makeSelect([
    { value: "global", label: "Use Global Setting" },
    { value: "auto", label: "Auto: Gemma decides" },
    { value: "smooth", label: "Smooth Transition" },
    { value: "morph", label: "Surreal Morph" },
  ], "global");
  const flfTransitionTypeField = makeField("First Last Frame transition type", flfTransitionTypeSelect);
  flfTransitionTypeField.style.display = "none";
  const lyricTextInput = document.createElement("textarea");
  lyricTextInput.placeholder = "Lyric, dialogue, or timing line for this scene...";
  lyricTextInput.style.cssText = notesInput.style.cssText;
  const lyricSingersInput = makeInput("");
  lyricSingersInput.placeholder = "Optional performer/speaker names, comma separated...";
  const t2iTextGemmaModelSelect = makeSelect([""], "");
  const gemmaModelSelect = makeSelect([""], "");
  const mmprojSelect = makeSelect([""], "");
  const krea2TwoPassTextGemmaModelSelect = makeSelect([""], "");
  const krea2TwoPassGemmaModelSelect = makeSelect([""], "");
  const krea2TwoPassMmprojSelect = makeSelect([""], "");
  const ernieTextGemmaModelSelect = makeSelect([""], "");
  const ernieGemmaModelSelect = makeSelect([""], "");
  const ernieMmprojSelect = makeSelect([""], "");
  const i2vTextGemmaModelSelect = makeSelect([""], "");
  const i2vGemmaModelSelect = makeSelect([""], "");
  const i2vMmprojSelect = makeSelect([""], "");
  const miniMaxTextGemmaModelSelect = makeSelect([""], "");
  const miniMaxGemmaModelSelect = makeSelect([""], "");
  const miniMaxMmprojSelect = makeSelect([""], "");
  for (const select of [krea2TwoPassTextGemmaModelSelect, krea2TwoPassGemmaModelSelect, krea2TwoPassMmprojSelect]) {
    select.addEventListener("change", syncKrea2TwoPassLlmSelectsToShared);
  }
  for (const select of [t2iTextGemmaModelSelect, gemmaModelSelect, mmprojSelect]) {
    select.addEventListener("change", syncKrea2TwoPassLlmSelectsFromShared);
  }
  const syncMiniMaxLlmSelectsFromShared = () => {
    miniMaxTextGemmaModelSelect.value = i2vTextGemmaModelSelect.value || "";
    miniMaxGemmaModelSelect.value = i2vGemmaModelSelect.value || "";
    miniMaxMmprojSelect.value = i2vMmprojSelect.value || "";
  };
  const syncMiniMaxLlmSelectsToShared = () => {
    i2vTextGemmaModelSelect.value = miniMaxTextGemmaModelSelect.value || "";
    i2vGemmaModelSelect.value = miniMaxGemmaModelSelect.value || "";
    i2vMmprojSelect.value = miniMaxMmprojSelect.value || "";
  };
  for (const select of [miniMaxTextGemmaModelSelect, miniMaxGemmaModelSelect, miniMaxMmprojSelect]) {
    select.addEventListener("change", syncMiniMaxLlmSelectsToShared);
  }
  for (const select of [i2vTextGemmaModelSelect, i2vGemmaModelSelect, i2vMmprojSelect]) {
    select.addEventListener("change", syncMiniMaxLlmSelectsFromShared);
  }
  const useVisionReference = makeCheckbox("Use vision reference image?", false);
  const useI2VVisionReference = makeCheckbox("Use image reference for I2V prompt?", true);
  const useI2VPromptEnhancementPass = makeCheckbox("I2V prompt enhancement pass", false);
  const i2vPromptEnhancementNote = document.createElement("div");
  i2vPromptEnhancementNote.textContent = "Optional second Gemma pass that rewrites I2V/T2V drafts into a stronger LTX-ready paragraph shape.";
  i2vPromptEnhancementNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;margin-top:-4px;";
  const i2vReferenceNote = document.createElement("div");
  i2vReferenceNote.textContent = "When checked, Gemma looks at the scene image and your video notes to create the I2V prompt. When unchecked, it uses the T2I prompt text and your video notes instead.";
  i2vReferenceNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;margin-top:-4px;";
  const useT2VVisionReference = makeCheckbox("Use image reference for T2V Gemma prompt?", false);
  const t2vReferenceNote = document.createElement("div");
  t2vReferenceNote.textContent = "Optional: Gemma looks at a reference image for pose, framing, mood, or visual direction while still creating a text-to-video prompt.";
  t2vReferenceNote.style.cssText = i2vReferenceNote.style.cssText;
  const t2vLocationNote = document.createElement("div");
  t2vLocationNote.textContent = "T2V uses mapped character and location descriptions as text context for Gemma. It does not pass those reference images into the video render.";
  t2vLocationNote.style.cssText = i2vReferenceNote.style.cssText;
  const refImageInput = makeInput("");
  refImageInput.style.display = "none";
  const refImagePanel = document.createElement("div");
  refImagePanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
  const refImageNote = document.createElement("div");
  refImageNote.textContent = "Optional: give Gemma a visual reference for the direction you want.";
  refImageNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;";
  const refImageDrop = document.createElement("div");
  refImageDrop.textContent = "Drop a reference image here, or drag a scene image from the timeline.";
  refImageDrop.style.cssText = zI2IDrop.style.cssText;
  const refImageLoadButton = makeButton("Load Reference Image", "primary");
  refImagePanel.append(refImageNote, refImageDrop, refImageLoadButton, refImageInput);
  const createT2IButton = makeButton("Gemma T2I", "primary");
  const editZImageT2IInstructionsButton = makeButton("Edit ZImage T2I Instructions");
  const editT2IPromptButton = makeEditImagePromptButton();
  const createI2VButton = makeButton("Gemma I2V", "primary");
  const sendT2IPromptToEnhanceButton = makeMiniButton("Send to Enhance");
  const t2iPrompt = document.createElement("textarea");
  t2iPrompt.placeholder = "Text-to-image prompt...";
  t2iPrompt.style.cssText = notesInput.style.cssText;
  const ernieNotesInput = document.createElement("textarea");
  ernieNotesInput.placeholder = notesInput.placeholder;
  ernieNotesInput.style.cssText = notesInput.style.cssText;
  const ernieT2IPrompt = document.createElement("textarea");
  ernieT2IPrompt.placeholder = t2iPrompt.placeholder;
  ernieT2IPrompt.style.cssText = t2iPrompt.style.cssText;
  const ernieUseVisionReference = makeCheckbox("Use vision reference image?", false);
  const ernieRefImagePanel = document.createElement("div");
  ernieRefImagePanel.style.cssText = refImagePanel.style.cssText;
  const ernieRefImageNote = document.createElement("div");
  ernieRefImageNote.textContent = refImageNote.textContent;
  ernieRefImageNote.style.cssText = refImageNote.style.cssText;
  const ernieRefImageDrop = document.createElement("div");
  ernieRefImageDrop.textContent = refImageDrop.textContent;
  ernieRefImageDrop.style.cssText = refImageDrop.style.cssText;
  const ernieRefImageLoadButton = makeButton("Load Reference Image", "primary");
  ernieRefImagePanel.append(ernieRefImageNote, ernieRefImageDrop, ernieRefImageLoadButton);
  const krea2TwoPassNotesInput = document.createElement("textarea");
  krea2TwoPassNotesInput.placeholder = notesInput.placeholder;
  krea2TwoPassNotesInput.style.cssText = notesInput.style.cssText;
  const krea2TwoPassT2IPrompt = document.createElement("textarea");
  krea2TwoPassT2IPrompt.placeholder = t2iPrompt.placeholder;
  krea2TwoPassT2IPrompt.style.cssText = t2iPrompt.style.cssText;
  const krea2TwoPassUseVisionReference = makeCheckbox("Use vision reference image?", false);
  const krea2TwoPassRefImagePanel = document.createElement("div");
  krea2TwoPassRefImagePanel.style.cssText = refImagePanel.style.cssText;
  const krea2TwoPassRefImageNote = document.createElement("div");
  krea2TwoPassRefImageNote.textContent = refImageNote.textContent;
  krea2TwoPassRefImageNote.style.cssText = refImageNote.style.cssText;
  const krea2TwoPassRefImageDrop = document.createElement("div");
  krea2TwoPassRefImageDrop.textContent = refImageDrop.textContent;
  krea2TwoPassRefImageDrop.style.cssText = refImageDrop.style.cssText;
  const krea2TwoPassRefImageLoadButton = makeButton("Load Reference Image", "primary");
  krea2TwoPassRefImagePanel.append(krea2TwoPassRefImageNote, krea2TwoPassRefImageDrop, krea2TwoPassRefImageLoadButton);
  const t2vRefImagePanel = document.createElement("div");
  t2vRefImagePanel.style.cssText = refImagePanel.style.cssText;
  const t2vRefImageNote = document.createElement("div");
  t2vRefImageNote.textContent = "Drop/load the image Gemma should look at while writing the T2V prompt.";
  t2vRefImageNote.style.cssText = refImageNote.style.cssText;
  const t2vRefImageDrop = document.createElement("div");
  t2vRefImageDrop.textContent = refImageDrop.textContent;
  t2vRefImageDrop.style.cssText = refImageDrop.style.cssText;
  const t2vRefImageLoadButton = makeButton("Load Reference Image", "primary");
  t2vRefImagePanel.append(t2vRefImageNote, t2vRefImageDrop, t2vRefImageLoadButton);
  const ernieCreateT2IButton = makeButton("Gemma T2I", "primary");
  const editErnieT2IInstructionsButton = makeButton("Edit Ernie T2I Instructions");
  const editErnieT2IPromptButton = makeEditImagePromptButton();
  const ernieSendT2IPromptToEnhanceButton = makeMiniButton("Send to Enhance");
  const krea2TwoPassCreateT2IButton = makeButton("Gemma T2I", "primary");
  const editKrea2TwoPassT2IPromptButton = makeEditImagePromptButton();
  const krea2TwoPassSendT2IPromptToEnhanceButton = makeMiniButton("Send to Enhance");
  const editKrea2T2IInstructionsButton = makeButton("Edit Krea 2 T2I Instructions");
  const editI2VPromptButton = makeButton("Edit Prompt");
  editI2VPromptButton.title = "Ask Gemma to make a focused text-only edit to the current video prompt.";
  editI2VPromptButton.style.display = "none";
  const editI2VInstructionsButton = makeButton("Edit I2V Instructions");
  editI2VInstructionsButton.title = "Advanced: customize the Gemma instructions used when writing Image-to-Video prompts.";
  const editIdLoraInstructionsButton = makeButton("Edit ID-LoRA Instructions");
  editIdLoraInstructionsButton.title = "Advanced: customize the Gemma instructions used when writing ID-LoRA I2V scripts.";
  const editRTVInstructionsButton = makeButton("Edit Reference Video Instructions");
  editRTVInstructionsButton.title = "Advanced: customize the Gemma instructions used when writing Reference-to-Video prompts.";
  const editIngredientsInstructionsButton = makeButton("Edit Ingredients Video Instructions");
  editIngredientsInstructionsButton.title = "Advanced: customize the Gemma instructions used when writing Ingredients-to-Video prompts.";
  const editT2VInstructionsButton = makeButton("Edit T2V Instructions");
  editT2VInstructionsButton.title = "Advanced: customize the Gemma instructions used when writing Text-to-Video prompts.";
  const i2vPrompt = document.createElement("textarea");
  i2vPrompt.placeholder = "Image-to-video prompt...";
  i2vPrompt.style.cssText = notesInput.style.cssText;
  const saveI2VPromptButton = makeButton("Save Updated Prompt", "primary");
  saveI2VPromptButton.title = "Save the updated prompt to this scene and sync to the project storyboard.";
  saveI2VPromptButton.style.width = "100%";
  saveI2VPromptButton.style.marginTop = "4px";
  saveI2VPromptButton.disabled = true;
  saveI2VPromptButton.style.opacity = "0.5";
  saveI2VPromptButton.style.cursor = "not-allowed";

  i2vPrompt.addEventListener("input", updateI2VPromptSaveButtonState);
  i2vPrompt.addEventListener("change", updateI2VPromptSaveButtonState);

  return {
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
  };
}

export function createImagePanels({
  activeErnieImageSettings, activeFluxKleinSettings, activeKrea2TwoPassSettings, activeNBImageSettings,
  activeSegment, activeZImageSettings, applyImageSettingsToMultiSelection, autoSaveSessionQuiet,
  browserAiGroupPrompt, currentVideoMode, ernieBatchSize, ernieClipPicker, ernieCreateButton, ernieHeight,
  ernieI2IPanel, ernieI2IPath, ernieI2ISlider, ernieI2IStartStep, ernieImageCard, ernieImageModePanel,
  ernieImagePanel, ernieImageTriggerInput, ernieLoraCount, ernieLoraPanel, ernieLoraRows, ernieLoraSlots,
  ernieSeed, ernieSeedMode, ernieUnetPicker, ernieUseImageToImage, ernieUseLora, ernieVaePicker, ernieWidth,
  flowGptAskPreviousImage, flowGptAspectRatio, flowGptAspectRatioField, flowGptCard, flowGptFailureMode,
  flowGptLoginButton, flowGptManualChatPrompt, flowGptModePanel, flowGptPrompt, flowGptRetries,
  flowGptTimeout, flowNanoProviderButton, fluxClipPicker, fluxHeight, fluxImageTriggerInput, fluxKleinCard,
  fluxKleinModePanel, fluxKleinPanel, fluxLoraCount, fluxLoraPanel, fluxLoraRows, fluxLoraSlots, fluxNotes,
  fluxPrompt, fluxReferenceContextForSegment, fluxSeed, fluxUnetPicker, fluxUseDirectorNotes, fluxUseLora,
  fluxUseTextOnlyGemmaPrompt, fluxVaePicker, fluxWidth, gptImageProviderButton, hasMultiSceneBatchSelection,
  i2vAdvancedNodeSettingsPanel, i2vAdvancedNodeSettingsSection, i2vPass1Bypass, i2vPass1NodePanel,
  i2vPass1SamplerSelect, i2vPass1SigmasInput, i2vPass1StrengthInput, i2vPass1StrengthSlider, i2vPass2Bypass,
  i2vPass2NodePanel, i2vPass2SamplerSelect, i2vPass2SigmasInput, i2vPass2StrengthInput,
  i2vPass2StrengthSlider, imageTriggerInput, krea2TwoPassAspectRatio, krea2TwoPassBatchSize, krea2TwoPassCard,
  krea2TwoPassCfg, krea2TwoPassClipPicker, krea2TwoPassCreateButton, krea2TwoPassCreativity,
  krea2TwoPassCreativityInput, krea2TwoPassI2IPanel, krea2TwoPassI2IPath, krea2TwoPassImageTriggerInput,
  krea2TwoPassLoraCount, krea2TwoPassLoraPanel, krea2TwoPassLoraRows, krea2TwoPassLoraSlots,
  krea2TwoPassModePanel, krea2TwoPassPanel, krea2TwoPassSampler, krea2TwoPassSeed, krea2TwoPassSeedMode,
  krea2TwoPassUnetPicker, krea2TwoPassUseImageToImage, krea2TwoPassUseLora, krea2TwoPassVaePicker,
  loadCustomImageButton, mergedFluxImageIngredients, metaImageProviderButton, nbApiKey, nbImageCard,
  nbImagePanel, nbModelSelect, nbNotes, nbPrompt, nbReferenceContextForSegment, nbUseDirectorNotes,
  nbUseTextOnlyGemmaPrompt, previewButton, pushHistory, renderBrowserAiReferenceGroups,
  renderFluxGlobalIngredientList, renderFluxIngredientList, renderList, renderNBIngredientList, state,
  syncFlowGptManualPanel, syncFluxGlobalIngredientPanel, useFluxKlein, useSceneErnieImageSettings,
  useSceneFluxKleinSettings, useSceneKrea2TwoPassSettings, useSceneNBImageSettings, useSceneZImageSettings,
  zBatchSize, zClipPicker, zEnhanceCard, zEnhancePanel, zEnhanceSeed, zFirstHeight, zFirstWidth, zI2IPanel,
  zI2IPath, zI2ISlider, zI2IStartStep, zImageCard, zImageModePanel, zLoraCount, zLoraPanel, zLoraRows,
  zLoraSlots, zSecondHeight, zSecondWidth, zSeed, zSeedMode, zUnetPicker, zUseImageToImage, zUseLora,
  zVaePicker,
}) {
  function updateZLoraVisibility() {
    const count = Math.max(0, Math.min(4, Number(zLoraCount.value || 0)));
    zLoraPanel.style.display = zUseLora.input.checked ? "flex" : "none";
    zLoraRows.style.display = zUseLora.input.checked && count > 0 ? "flex" : "none";
    zLoraSlots.forEach((slot, index) => {
      slot.row.style.display = index < count ? "grid" : "none";
    });
  }

  function updateZImageToImageVisibility() {
    zI2IPanel.style.display = zUseImageToImage.input.checked ? "flex" : "none";
  }

  function saveZImageSettingsFromPanel() {
    pushHistory();
    const count = Math.max(0, Math.min(4, Number(zLoraCount.value || 0)));
    const currentSettings = activeZImageSettings() || {};
    const i2iPathValue = zI2IPath.value || "";
    const keepDataSource = Boolean(currentSettings.image_to_image_data && i2iPathValue === currentSettings.image_to_image_name);
    const settings = {
      unet_name: zUnetPicker.input.value || "z_image_turbo_bf16.safetensors",
      clip_name: zClipPicker.input.value || "qwen_3_4b.safetensors",
      vae_name: zVaePicker.input.value || "ae.safetensors",
      first_pass_width: Number(zFirstWidth.value || 1280),
      first_pass_height: Number(zFirstHeight.value || 720),
      second_pass_width: Number(zSecondWidth.value || 1920),
      second_pass_height: Number(zSecondHeight.value || 1080),
      seed: Number(zSeed.value || 1),
      seed_mode: zSeedMode.value || "fixed",
      batch_size: Math.max(1, Math.min(16, Number(zBatchSize.value || 1))),
      image_trigger_phrase: imageTriggerInput.value || "",
      use_loras: Boolean(zUseLora.input.checked),
      lora_count: count,
      loras: zLoraSlots.map((slot) => ({
        name: slot.picker.input.value || "[none]",
        first_pass_strength: Number(slot.firstPassStrength.value || 0.5),
        second_pass_strength: Number(slot.secondPassStrength.value || 1),
        strength: Number(slot.secondPassStrength.value || 1),
      })),
      use_image_to_image: Boolean(zUseImageToImage.input.checked),
      image_to_image_start_at_step: Math.max(1, Math.min(8, Number(zI2IStartStep.value || zI2ISlider.value || 5))),
      image_to_image_path: keepDataSource ? "" : i2iPathValue,
      image_to_image_data: keepDataSource ? currentSettings.image_to_image_data || "" : "",
      image_to_image_name: keepDataSource ? currentSettings.image_to_image_name || "" : "",
    };
    const segment = activeSegment();
    if (segment?.use_scene_zimage_settings || hasMultiSceneBatchSelection()) {
      segment.zimage_settings = settings;
      if (segment) segment.use_scene_zimage_settings = true;
    } else {
      state.imageTriggerPhrase = settings.image_trigger_phrase || "";
      state.zimageSettings = settings;
    }
    applyImageSettingsToMultiSelection("zimage", settings);
    updateZLoraVisibility();
    updateZImageToImageVisibility();
    renderList();
    return settings;
  }

  function advanceZImageSeedAfterRun(settings) {
    const mode = String(settings?.seed_mode || "fixed").toLowerCase();
    if (mode === "increment") {
      settings.seed = Math.min(Number.MAX_SAFE_INTEGER, Number(settings.seed || 0) + 1);
    } else if (mode === "decrement") {
      settings.seed = Math.max(0, Number(settings.seed || 0) - 1);
    } else {
      return;
    }
    zSeed.value = String(settings.seed);
    if (activeSegment()?.use_scene_zimage_settings) {
      activeSegment().zimage_settings = settings;
    } else {
      state.zimageSettings = settings;
    }
  }

  function updateErnieLoraVisibility() {
    const count = Math.max(0, Math.min(4, Number(ernieLoraCount.value || 0)));
    ernieLoraPanel.style.display = ernieUseLora.input.checked ? "flex" : "none";
    ernieLoraRows.style.display = ernieUseLora.input.checked && count > 0 ? "flex" : "none";
    ernieLoraSlots.forEach((slot, index) => {
      slot.row.style.display = index < count ? "grid" : "none";
    });
  }

  function updateErnieImageToImageVisibility() {
    ernieI2IPanel.style.display = ernieUseImageToImage.input.checked ? "flex" : "none";
  }

  function saveErnieImageSettingsFromPanel() {
    pushHistory();
    const count = Math.max(0, Math.min(4, Number(ernieLoraCount.value || 0)));
    const segment = activeSegment();
    const currentSettings = activeErnieImageSettings() || {};
    const i2iPathValue = ernieI2IPath.value || "";
    const keepDataSource = Boolean(currentSettings.image_to_image_data && i2iPathValue === currentSettings.image_to_image_name);
    const settings = {
      unet_name: ernieUnetPicker.input.value || "ernie\\ernie-image-turbo.safetensors",
      clip_name: ernieClipPicker.input.value || "ministral-3-3b.safetensors",
      vae_name: ernieVaePicker.input.value || "flux\\flux2-vae.safetensors",
      width: Number(ernieWidth.value || 1280),
      height: Number(ernieHeight.value || 720),
      seed: Number(ernieSeed.value || 1),
      seed_mode: ernieSeedMode.value || "fixed",
      batch_size: Math.max(1, Math.min(16, Number(ernieBatchSize.value || 1))),
      image_trigger_phrase: ernieImageTriggerInput.value || "",
      use_loras: Boolean(ernieUseLora.input.checked),
      lora_count: count,
      loras: ernieLoraSlots.map((slot) => ({ name: slot.picker.input.value || "[none]", strength: Number(slot.strength.value || 1) })),
      use_image_to_image: Boolean(ernieUseImageToImage.input.checked),
      image_to_image_start_at_step: Math.max(1, Math.min(8, Number(ernieI2IStartStep.value || ernieI2ISlider.value || 5))),
      image_to_image_path: keepDataSource ? "" : i2iPathValue,
      image_to_image_data: keepDataSource ? currentSettings.image_to_image_data || "" : "",
      image_to_image_name: keepDataSource ? currentSettings.image_to_image_name || "" : "",
    };
    if (segment?.use_scene_ernie_image_settings || hasMultiSceneBatchSelection()) {
      if (segment) {
        segment.use_scene_ernie_image_settings = true;
        segment.ernie_image_settings = settings;
      }
    }
    else {
      state.imageTriggerPhrase = settings.image_trigger_phrase || "";
      state.ernieImageSettings = settings;
    }
    applyImageSettingsToMultiSelection("ernie_image", settings);
    updateErnieLoraVisibility();
    updateErnieImageToImageVisibility();
    return settings;
  }

  function advanceErnieSeedAfterRun(settings) {
    const mode = String(settings?.seed_mode || "fixed").toLowerCase();
    if (mode === "increment") {
      settings.seed = Math.min(Number.MAX_SAFE_INTEGER, Number(settings.seed || 0) + 1);
    } else if (mode === "decrement") {
      settings.seed = Math.max(0, Number(settings.seed || 0) - 1);
    } else {
      return;
    }
    ernieSeed.value = String(settings.seed);
    if (activeSegment()?.use_scene_ernie_image_settings) activeSegment().ernie_image_settings = settings;
    else state.ernieImageSettings = settings;
  }

  function updateKrea2TwoPassImageToImageVisibility() {
    krea2TwoPassI2IPanel.style.display = krea2TwoPassUseImageToImage.input.checked ? "flex" : "none";
  }

  function updateKrea2TwoPassLoraVisibility() {
    const count = Math.max(0, Math.min(4, Number(krea2TwoPassLoraCount.value || 0)));
    krea2TwoPassLoraPanel.style.display = krea2TwoPassUseLora.input.checked ? "flex" : "none";
    krea2TwoPassLoraRows.style.display = krea2TwoPassUseLora.input.checked && count > 0 ? "flex" : "none";
    krea2TwoPassLoraSlots.forEach((slot, index) => {
      slot.row.style.display = index < count ? "grid" : "none";
    });
  }

  function saveKrea2TwoPassSettingsFromPanel() {
    pushHistory();
    const segment = activeSegment();
    const currentSettings = activeKrea2TwoPassSettings() || {};
    const i2iPathValue = krea2TwoPassI2IPath.value || "";
    const keepDataSource = Boolean(currentSettings.image_to_image_data && i2iPathValue === currentSettings.image_to_image_name);
    const loraCount = Math.max(0, Math.min(4, Number(krea2TwoPassLoraCount.value || 0)));
    const settings = cloneKrea2TwoPassSettings({
      unet_name: krea2TwoPassUnetPicker.input.value || "krea2_turbo_fp8_scaled.safetensors",
      clip_name: krea2TwoPassClipPicker.input.value || "qwen3vl_4b_fp8_scaled.safetensors",
      vae_name: krea2TwoPassVaePicker.input.value || "qwen_image_vae.safetensors",
      use_loras: Boolean(krea2TwoPassUseLora.input.checked),
      lora_count: loraCount,
      loras: krea2TwoPassLoraSlots.map((slot) => ({
        name: slot.picker.input.value || "[none]",
        first_pass_strength: Number(slot.firstPassStrength.value || 0.5),
        second_pass_strength: Number(slot.secondPassStrength.value || 0),
        strength: Number(slot.secondPassStrength.value || 0),
      })),
      aspect_ratio: krea2TwoPassAspectRatio.value || "16:9 (Widescreen)",
      sampler_name: krea2TwoPassSampler.value || "euler_ancestral_cfg_pp",
      cfg: Number(krea2TwoPassCfg.value || 1.2),
      seed: Number(krea2TwoPassSeed.value || 1),
      seed_mode: krea2TwoPassSeedMode.value || "fixed",
      batch_size: Math.max(1, Math.min(16, Number(krea2TwoPassBatchSize.value || 1))),
      image_trigger_phrase: krea2TwoPassImageTriggerInput.value || "",
      use_image_to_image: Boolean(krea2TwoPassUseImageToImage.input.checked),
      image_to_image_creativity: Math.max(0, Math.min(10, Number(krea2TwoPassCreativityInput.value || krea2TwoPassCreativity.value || 5))),
      image_to_image_path: keepDataSource ? "" : i2iPathValue,
      image_to_image_data: keepDataSource ? currentSettings.image_to_image_data || "" : "",
      image_to_image_name: keepDataSource ? currentSettings.image_to_image_name || "" : "",
    });
    if (segment?.use_scene_krea2_2pass_settings || hasMultiSceneBatchSelection()) {
      if (segment) {
        segment.use_scene_krea2_2pass_settings = true;
        segment.krea2_2pass_settings = settings;
      }
    } else {
      state.imageTriggerPhrase = settings.image_trigger_phrase || "";
      state.krea2TwoPassSettings = settings;
    }
    applyImageSettingsToMultiSelection("krea2_2pass", settings);
    updateKrea2TwoPassLoraVisibility();
    updateKrea2TwoPassImageToImageVisibility();
    return settings;
  }

  function advanceKrea2TwoPassSeedAfterRun(settings) {
    const mode = String(settings?.seed_mode || "fixed").toLowerCase();
    if (mode === "increment") {
      settings.seed = Math.min(Number.MAX_SAFE_INTEGER, Number(settings.seed || 0) + 1);
    } else if (mode === "decrement") {
      settings.seed = Math.max(0, Number(settings.seed || 0) - 1);
    } else {
      return;
    }
    krea2TwoPassSeed.value = String(settings.seed);
    if (activeSegment()?.use_scene_krea2_2pass_settings) activeSegment().krea2_2pass_settings = settings;
    else state.krea2TwoPassSettings = settings;
  }

  function advanceZEnhanceSeedAfterRun(settings) {
    const mode = String(settings?.seed_mode || "fixed").toLowerCase();
    if (mode === "increment") {
      settings.seed = Math.min(Number.MAX_SAFE_INTEGER, Number(settings.seed || 0) + 1);
    } else if (mode === "decrement") {
      settings.seed = Math.max(0, Number(settings.seed || 0) - 1);
    } else {
      return;
    }
    zEnhanceSeed.value = String(settings.seed);
    state.zEnhanceSettings = settings;
  }

  function updateFluxLoraVisibility() {
    const count = Math.max(0, Math.min(4, Number(fluxLoraCount.value || 0)));
    fluxLoraPanel.style.display = fluxUseLora.input.checked ? "flex" : "none";
    fluxLoraRows.style.display = fluxUseLora.input.checked && count > 0 ? "flex" : "none";
    fluxLoraSlots.forEach((slot, index) => {
      slot.row.style.display = index < count ? "grid" : "none";
    });
  }

  function saveFluxKleinSettingsFromPanel() {
    pushHistory();
    const count = Math.max(0, Math.min(4, Number(fluxLoraCount.value || 0)));
    const current = activeFluxKleinSettings() || {};
    const segment = activeSegment();
    if (segment) {
      if (!Array.isArray(segment.flux_image_ingredients)) segment.flux_image_ingredients = [];
      segment.flux_notes = fluxNotes.value || "";
      segment.flux_prompt = fluxPrompt.value || "";
      segment.t2i_prompt = segment.flux_prompt;
    }
    const settings = {
      enabled: Boolean(useFluxKlein.input.checked),
      image_model_mode: state.imageModelMode || current.image_model_mode || (useFluxKlein.input.checked ? "flux_klein" : "zimage"),
      unet_name: fluxUnetPicker.input.value || "",
      clip_name: fluxClipPicker.input.value || "",
      vae_name: fluxVaePicker.input.value || "",
      width: Number(fluxWidth.value || 1024),
      height: Number(fluxHeight.value || 576),
      seed: Number(fluxSeed.value || 100),
      use_text_only_gemma_prompt: Boolean(fluxUseTextOnlyGemmaPrompt.input.checked),
      use_director_notes: Boolean(fluxUseDirectorNotes.input.checked),
      use_loras: Boolean(fluxUseLora.input.checked),
      lora_count: count,
      loras: fluxLoraSlots.map((slot) => ({
        name: slot.picker.input.value || "[none]",
        strength: Number(slot.strength.value || 1),
      })),
      image_trigger_phrase: fluxImageTriggerInput.value || "",
    };
    if (segment?.use_scene_flux_klein_settings || hasMultiSceneBatchSelection()) {
      if (segment) {
        segment.use_scene_flux_klein_settings = true;
        segment.flux_klein_settings = settings;
      }
    }
    else {
      state.imageTriggerPhrase = settings.image_trigger_phrase || "";
      state.fluxKleinSettings = settings;
    }
    applyImageSettingsToMultiSelection("flux_klein", settings);
    updateFluxLoraVisibility();
    return {
      ...settings,
      image_ingredients: mergedFluxImageIngredients(segment),
      use_global_image_ingredients: Boolean(state.useFluxGlobalImageIngredients),
      global_image_ingredients: Array.isArray(state.fluxGlobalImageIngredients) ? state.fluxGlobalImageIngredients : [],
      scene_image_ingredients: Array.isArray(segment?.flux_image_ingredients) ? segment.flux_image_ingredients : [],
      notes: segment?.flux_notes || "",
      prompt: segment?.flux_prompt || "",
      reference_context: fluxReferenceContextForSegment(segment),
    };
  }

  function saveNBImageSettingsFromPanel() {
    pushHistory();
    const current = activeNBImageSettings() || {};
    const segment = activeSegment();
    if (segment) {
      if (!Array.isArray(segment.flux_image_ingredients)) segment.flux_image_ingredients = [];
      segment.nb_notes = nbNotes.value || "";
      segment.nb_prompt = nbPrompt.value || "";
      segment.t2i_prompt = segment.nb_prompt;
    }
    const settings = {
      ...current,
      api_key: nbApiKey.value || "",
      model: nbModelSelect.value || DEFAULT_NB_IMAGE_MODEL,
      use_text_only_gemma_prompt: Boolean(nbUseTextOnlyGemmaPrompt.input.checked),
      use_director_notes: Boolean(nbUseDirectorNotes.input.checked),
    };
    if (segment?.use_scene_nb_image_settings || hasMultiSceneBatchSelection()) {
      if (segment) {
        segment.use_scene_nb_image_settings = true;
        segment.nb_image_settings = settings;
      }
    } else {
      state.nbImageSettings = settings;
    }
    applyImageSettingsToMultiSelection("nano_banana", settings);
    return {
      ...settings,
      image_ingredients: mergedFluxImageIngredients(segment),
      notes: segment?.nb_notes || segment?.flux_notes || segment?.notes || "",
      prompt: segment?.nb_prompt || "",
      reference_context: nbReferenceContextForSegment(segment),
    };
  }

  function syncFlowGptBrowserPanel() {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    state.flowGptBrowserSettings = settings;
    const isGpt = settings.provider === BROWSER_IMAGE_PROVIDERS.GPT_IMAGE;
    flowGptAspectRatio.value = settings.aspect_ratio || "16:9";
    flowGptTimeout.value = browserImageProviderTimeout(settings);
    flowGptRetries.value = settings.max_retries || 10;
    flowGptFailureMode.value = settings.failure_mode || "last_successful_image";
    flowGptAskPreviousImage.input.checked = Boolean(settings.ask_previous_scene_image);
    flowGptManualChatPrompt.value = settings.manual_chat_prompt || defaultFlowGptBrowserSettings().manual_chat_prompt;
    browserAiGroupPrompt.value = settings.manual_chat_prompt || defaultFlowGptBrowserSettings().manual_chat_prompt;
    const segment = activeSegment();
    flowGptPrompt.value = segment?.flow_gpt_prompt || segment?.t2i_prompt || segment?.flux_prompt || segment?.nb_prompt || "";
    flowGptAspectRatioField.style.display = isGpt ? "flex" : "none";
    flowGptAspectRatio.disabled = !isGpt;
    flowGptAspectRatio.title = isGpt ? "GPT Image gets this aspect ratio appended to the prompt." : "This provider does not use the appended GPT aspect-ratio setting.";
    flowGptLoginButton.textContent = `Open ${browserImageProviderShortLabel(settings.provider)} Login`;
    for (const [button, provider] of [
      [flowNanoProviderButton, BROWSER_IMAGE_PROVIDERS.FLOW_NANO_BANANA],
      [gptImageProviderButton, BROWSER_IMAGE_PROVIDERS.GPT_IMAGE],
      [metaImageProviderButton, BROWSER_IMAGE_PROVIDERS.META_AI],
    ]) {
      const active = settings.provider === provider;
      button.style.background = active ? "#06b6d4" : "#27272a";
      button.style.borderColor = active ? "#0891b2" : "#3f3f46";
      button.style.color = active ? "#082f49" : "#f4f4f5";
    }
    renderBrowserAiReferenceGroups();
    syncFlowGptManualPanel();
  }

  function saveFlowGptBrowserSettingsFromPanel() {
    pushHistory();
    const current = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    const timeout = Math.max(60, Math.min(2400, Number(flowGptTimeout.value || current.timeout_seconds || 600)));
    state.flowGptBrowserSettings = cloneFlowGptBrowserSettings({
      ...current,
      aspect_ratio: flowGptAspectRatio.value || "16:9",
      timeout_seconds: timeout,
      flow_timeout_seconds: current.provider === BROWSER_IMAGE_PROVIDERS.FLOW_NANO_BANANA ? timeout : current.flow_timeout_seconds,
      gpt_timeout_seconds: current.provider === BROWSER_IMAGE_PROVIDERS.GPT_IMAGE ? timeout : current.gpt_timeout_seconds,
      meta_timeout_seconds: current.provider === BROWSER_IMAGE_PROVIDERS.META_AI ? timeout : current.meta_timeout_seconds,
      max_retries: Math.max(1, Math.min(20, Number(flowGptRetries.value || 10))),
      failure_mode: flowGptFailureMode.value || "last_successful_image",
      ask_previous_scene_image: Boolean(flowGptAskPreviousImage.input.checked),
      manual_chat_prompt: String(flowGptManualChatPrompt.value || "").trim() || defaultFlowGptBrowserSettings().manual_chat_prompt,
    });
    syncFlowGptBrowserPanel();
    return state.flowGptBrowserSettings;
  }

  function syncI2VAdvancedNodeControls(settings = {}) {
    const mode = currentVideoMode();
    let pass1SamplerName = settings.pass1_sampler_name || "euler_ancestral";
    let pass1Sigmas = settings.pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS;
    let pass2SamplerName = settings.pass2_sampler_name || "euler_ancestral";
    let pass2Sigmas = settings.pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS;
    if (mode === "t2v") {
      pass1SamplerName = settings.t2v_pass1_sampler_name || "euler_ancestral";
      pass1Sigmas = settings.t2v_pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS;
      pass2SamplerName = settings.t2v_pass2_sampler_name || "euler_ancestral";
      pass2Sigmas = settings.t2v_pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS;
    } else if (mode === "rtv") {
      pass1SamplerName = settings.rtv_pass1_sampler_name || "euler_ancestral";
      pass1Sigmas = settings.rtv_pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS;
      pass2SamplerName = settings.rtv_pass2_sampler_name || "euler_ancestral";
      pass2Sigmas = settings.rtv_pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS;
    } else if (mode === "ingredients") {
      pass1SamplerName = settings.ingredients_pass1_sampler_name || DEFAULT_INGREDIENTS_SAMPLER;
      pass1Sigmas = settings.ingredients_pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS;
      pass2SamplerName = settings.ingredients_pass2_sampler_name || DEFAULT_INGREDIENTS_SAMPLER;
      pass2Sigmas = settings.ingredients_pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS;
    }
    i2vPass1SamplerSelect.value = pass1SamplerName;
    i2vPass1SigmasInput.value = pass1Sigmas;
    setI2VStrengthPair(i2vPass1StrengthSlider, i2vPass1StrengthInput, settings.pass1_inplace_strength ?? 1);
    i2vPass1Bypass.input.checked = Boolean(settings.pass1_inplace_bypass);
    i2vPass2SamplerSelect.value = pass2SamplerName;
    i2vPass2SigmasInput.value = pass2Sigmas;
    setI2VStrengthPair(i2vPass2StrengthSlider, i2vPass2StrengthInput, settings.pass2_inplace_strength ?? 1);
    i2vPass2Bypass.input.checked = Boolean(settings.pass2_inplace_bypass);
    const showNodeSettings = mode === "i2v" || mode === "id_lora" || mode === "t2v" || mode === "rtv" || mode === "ingredients";
    const showInplaceSettings = mode === "i2v" || mode === "id_lora";
    const showSecondPass = mode !== "rtv" || settings.ltx_version !== "2.3";
    i2vAdvancedNodeSettingsPanel.style.display = showNodeSettings ? "flex" : "none";
    i2vAdvancedNodeSettingsSection.style.display = showNodeSettings ? "" : "none";
    i2vPass1NodePanel.inplaceCard.style.display = showInplaceSettings ? "flex" : "none";
    i2vPass2NodePanel.style.display = showSecondPass ? "flex" : "none";
    i2vPass2NodePanel.inplaceCard.style.display = showInplaceSettings ? "flex" : "none";
  }

  function setFlowGptProvider(provider) {
    pushHistory();
    const normalized = normalizeFlowGptBrowserProvider(provider);
    state.flowGptBrowserSettings = cloneFlowGptBrowserSettings({
      ...state.flowGptBrowserSettings,
      provider: normalized,
    });
    syncFlowGptBrowserPanel();
    autoSaveSessionQuiet(`Browser image provider changed to ${browserImageProviderLabel(normalized)}`).catch(() => null);
  }

  function syncZImageSettingsPanel() {
    const segment = activeSegment();
    useSceneZImageSettings.input.checked = Boolean(segment?.use_scene_zimage_settings);
    const settings = activeZImageSettings() || {};
    imageTriggerInput.value = settings.image_trigger_phrase || "";
    zUnetPicker.input.value = settings.unet_name || "z_image_turbo_bf16.safetensors";
    zClipPicker.input.value = settings.clip_name || "qwen_3_4b.safetensors";
    zVaePicker.input.value = settings.vae_name || "ae.safetensors";
    zFirstWidth.value = settings.first_pass_width || 1280;
    zFirstHeight.value = settings.first_pass_height || 720;
    zSecondWidth.value = settings.second_pass_width || 1920;
    zSecondHeight.value = settings.second_pass_height || 1080;
    zSeed.value = settings.seed || 1;
    zSeedMode.value = settings.seed_mode || "fixed";
    zBatchSize.value = Math.max(1, Math.min(16, Number(settings.batch_size || 1)));
    zUseLora.input.checked = Boolean(settings.use_loras);
    zLoraCount.value = Number(settings.lora_count || 0);
    zLoraPanel.style.display = zUseLora.input.checked ? "flex" : "none";
    zLoraRows.style.display = zUseLora.input.checked && Number(zLoraCount.value || 0) > 0 ? "flex" : "none";
    zLoraSlots.forEach((slot, index) => {
      const config = settings.loras?.[index] || {};
      const legacyStrength = config.strength;
      slot.row.style.display = index < Number(zLoraCount.value || 0) ? "grid" : "none";
      slot.picker.input.value = config.name || "[none]";
      slot.firstPassStrength.value = config.first_pass_strength ?? legacyStrength ?? 0.5;
      slot.secondPassStrength.value = config.second_pass_strength ?? legacyStrength ?? 1;
    });
    zUseImageToImage.input.checked = Boolean(settings.use_image_to_image);
    zI2IPanel.style.display = zUseImageToImage.input.checked ? "flex" : "none";
    const startStep = Math.max(1, Math.min(8, Number(settings.image_to_image_start_at_step || 5)));
    zI2ISlider.value = String(startStep);
    zI2IStartStep.value = String(startStep);
    zI2IPath.value = settings.image_to_image_path || settings.image_to_image_name || "";
  }

  function syncErnieImagePanel() {
    const segment = activeSegment();
    useSceneErnieImageSettings.input.checked = Boolean(segment?.use_scene_ernie_image_settings);
    const settings = activeErnieImageSettings() || {};
    ernieImageTriggerInput.value = settings.image_trigger_phrase || "";
    ernieUnetPicker.input.value = chooseModelValue(
      ernieUnetPicker.options || [],
      settings.unet_name || "ernie\\ernie-image-turbo.safetensors",
      ["ernie\\ernie-image-turbo.safetensors", "ernie-image-turbo.safetensors"],
    ) || settings.unet_name || "";
    ernieClipPicker.input.value = settings.clip_name || "ministral-3-3b.safetensors";
    ernieVaePicker.input.value = chooseModelValue(
      ernieVaePicker.options || [],
      settings.vae_name || "flux\\flux2-vae.safetensors",
      ["flux\\flux2-vae.safetensors", "flux2-vae.safetensors"],
    ) || settings.vae_name || "";
    ernieWidth.value = settings.width || 1280;
    ernieHeight.value = settings.height || 720;
    ernieSeed.value = settings.seed || 1;
    ernieSeedMode.value = settings.seed_mode || "fixed";
    ernieBatchSize.value = Math.max(1, Math.min(16, Number(settings.batch_size || 1)));
    ernieUseLora.input.checked = Boolean(settings.use_loras);
    ernieLoraCount.value = Number(settings.lora_count || 0);
    updateErnieLoraVisibility();
    ernieLoraSlots.forEach((slot, index) => {
      const config = settings.loras?.[index] || {};
      slot.picker.input.value = config.name || "[none]";
      slot.strength.value = config.strength ?? 1;
    });
    ernieUseImageToImage.input.checked = Boolean(settings.use_image_to_image);
    updateErnieImageToImageVisibility();
    const startStep = Math.max(1, Math.min(8, Number(settings.image_to_image_start_at_step || 5)));
    ernieI2ISlider.value = String(startStep);
    ernieI2IStartStep.value = String(startStep);
    ernieI2IPath.value = settings.image_to_image_path || settings.image_to_image_name || "";
  }

  function syncKrea2TwoPassPanel() {
    const segment = activeSegment();
    useSceneKrea2TwoPassSettings.input.checked = Boolean(segment?.use_scene_krea2_2pass_settings);
    const settings = activeKrea2TwoPassSettings() || {};
    krea2TwoPassImageTriggerInput.value = settings.image_trigger_phrase || "";
    krea2TwoPassUnetPicker.input.value = chooseModelValue(
      krea2TwoPassUnetPicker.options || [],
      settings.unet_name || "krea2_turbo_fp8_scaled.safetensors",
      "krea2_turbo_fp8_scaled.safetensors",
    ) || settings.unet_name || "";
    krea2TwoPassClipPicker.input.value = chooseModelValue(
      krea2TwoPassClipPicker.options || [],
      settings.clip_name || "qwen3vl_4b_fp8_scaled.safetensors",
      "qwen3vl_4b_fp8_scaled.safetensors",
    ) || settings.clip_name || "";
    krea2TwoPassVaePicker.input.value = chooseModelValue(
      krea2TwoPassVaePicker.options || [],
      settings.vae_name || "qwen_image_vae.safetensors",
      "qwen_image_vae.safetensors",
    ) || settings.vae_name || "";
    krea2TwoPassUseLora.input.checked = Boolean(settings.use_loras);
    krea2TwoPassLoraCount.value = Math.max(0, Math.min(4, Number(settings.lora_count || 0)));
    krea2TwoPassLoraPanel.style.display = krea2TwoPassUseLora.input.checked ? "flex" : "none";
    krea2TwoPassLoraRows.style.display = krea2TwoPassUseLora.input.checked && Number(krea2TwoPassLoraCount.value || 0) > 0 ? "flex" : "none";
    krea2TwoPassLoraSlots.forEach((slot, index) => {
      const config = settings.loras?.[index] || {};
      const legacyStrength = config.strength;
      slot.row.style.display = index < Number(krea2TwoPassLoraCount.value || 0) ? "grid" : "none";
      slot.picker.input.value = chooseModelValue(slot.picker.options || [], config.name || "[none]") || config.name || "[none]";
      slot.firstPassStrength.value = config.first_pass_strength ?? legacyStrength ?? 0.5;
      slot.secondPassStrength.value = config.second_pass_strength ?? legacyStrength ?? 0;
    });
    krea2TwoPassAspectRatio.value = settings.aspect_ratio || "16:9 (Widescreen)";
    krea2TwoPassSampler.value = settings.sampler_name || "euler_ancestral_cfg_pp";
    krea2TwoPassSeed.value = Number(settings.seed || 1);
    krea2TwoPassSeedMode.value = settings.seed_mode || "fixed";
    krea2TwoPassCfg.value = Math.max(1, Math.min(1.2, Number(settings.cfg ?? 1.2)));
    krea2TwoPassBatchSize.value = Math.max(1, Math.min(16, Number(settings.batch_size || 1)));
    krea2TwoPassUseImageToImage.input.checked = Boolean(settings.use_image_to_image);
    updateKrea2TwoPassImageToImageVisibility();
    const creativity = Math.max(0, Math.min(10, Number(settings.image_to_image_creativity ?? 5)));
    krea2TwoPassCreativity.value = String(creativity);
    krea2TwoPassCreativityInput.value = String(creativity);
    krea2TwoPassI2IPath.value = settings.image_to_image_path || settings.image_to_image_name || "";
  }

  function syncFluxKleinPanel() {
    const segment = activeSegment();
    const settings = activeFluxKleinSettings() || {};
    useSceneFluxKleinSettings.input.checked = Boolean(segment?.use_scene_flux_klein_settings);
    fluxImageTriggerInput.value = settings.image_trigger_phrase || "";
    const mode = state.imageModelMode || settings.image_model_mode || "zimage";
    state.imageModelMode = mode;
    settings.image_model_mode = mode;
    settings.enabled = mode === "flux_klein";
    zImageModePanel.style.display = mode === "zimage" ? "flex" : "none";
    fluxKleinModePanel.style.display = mode === "flux_klein" || mode === "nano_banana" ? "flex" : "none";
    ernieImageModePanel.style.display = mode === "ernie_image" ? "flex" : "none";
    krea2TwoPassModePanel.style.display = mode === "krea2_2pass" ? "flex" : "none";
    flowGptModePanel.style.display = mode === "flow_gpt" ? "flex" : "none";
    zEnhancePanel.style.display = mode === "z_enhance" ? "flex" : "none";
    previewButton.style.display = mode === "zimage" ? "" : "none";
    ernieCreateButton.style.display = mode === "ernie_image" ? "" : "none";
    krea2TwoPassCreateButton.style.display = mode === "krea2_2pass" ? "" : "none";
    useFluxKlein.input.checked = mode === "flux_klein";
    fluxKleinPanel.style.display = mode === "flux_klein" ? "flex" : "none";
    nbImagePanel.style.display = mode === "nano_banana" ? "flex" : "none";
    ernieImagePanel.style.display = mode === "ernie_image" ? "flex" : "none";
    krea2TwoPassPanel.style.display = mode === "krea2_2pass" ? "flex" : "none";
    for (const card of [zImageCard, fluxKleinCard, nbImageCard, ernieImageCard, krea2TwoPassCard, flowGptCard, zEnhanceCard]) {
      const active = card.dataset.model === mode;
      card.style.borderColor = active ? "#0891b2" : "#3f3f46";
      card.style.background = active ? "#06b6d4" : "#27272a";
      card.style.color = active ? "#082f49" : "#f4f4f5";
      card.style.boxShadow = active ? "inset 0 0 0 1px rgba(8,47,73,.22)" : "none";
    }
    loadCustomImageButton.style.borderColor = "#3f3f46";
    loadCustomImageButton.style.background = "#27272a";
    loadCustomImageButton.style.color = "#f4f4f5";
    loadCustomImageButton.style.boxShadow = "none";
    syncFluxGlobalIngredientPanel();
    renderFluxGlobalIngredientList();
    renderFluxIngredientList(segment);
    renderNBIngredientList(segment);
    fluxNotes.value = segment?.flux_notes || "";
    fluxPrompt.value = segment?.t2i_prompt || segment?.flux_prompt || "";
    fluxUnetPicker.input.value = chooseModelValue(
      fluxUnetPicker.options || [],
      settings.unet_name || "flux\\flux-2-klein-4b-fp8.safetensors",
      ["flux\\flux-2-klein-4b-fp8.safetensors", "flux-2-klein-4b-fp8.safetensors"],
    ) || settings.unet_name || "";
    fluxClipPicker.input.value = settings.clip_name || "qwen_3_4b.safetensors";
    fluxVaePicker.input.value = chooseModelValue(
      fluxVaePicker.options || [],
      settings.vae_name || "flux\\flux2-vae.safetensors",
      ["flux\\flux2-vae.safetensors", "flux2-vae.safetensors"],
    ) || settings.vae_name || "";
    fluxWidth.value = settings.width || 1024;
    fluxHeight.value = settings.height || 576;
    fluxSeed.value = settings.seed || 100;
    fluxUseTextOnlyGemmaPrompt.input.checked = Boolean(settings.use_text_only_gemma_prompt);
    fluxUseDirectorNotes.input.checked = Boolean(settings.use_director_notes);
    fluxUseLora.input.checked = Boolean(settings.use_loras);
    fluxLoraCount.value = Number(settings.lora_count || 0);
    fluxLoraSlots.forEach((slot, index) => {
      const config = settings.loras?.[index] || {};
      slot.row.style.display = index < Number(fluxLoraCount.value || 0) ? "grid" : "none";
      slot.picker.input.value = config.name || "[none]";
      slot.strength.value = config.strength ?? 1;
    });
    updateFluxLoraVisibility();
    syncErnieImagePanel();
    syncNBImagePanel();
    syncFlowGptBrowserPanel();
  }

  function syncNBImagePanel() {
    const segment = activeSegment();
    const settings = activeNBImageSettings() || {};
    useSceneNBImageSettings.input.checked = Boolean(segment?.use_scene_nb_image_settings);
    nbApiKey.value = settings.api_key || "";
    nbModelSelect.value = NB_IMAGE_MODELS.includes(settings.model) ? settings.model : DEFAULT_NB_IMAGE_MODEL;
    nbUseTextOnlyGemmaPrompt.input.checked = Boolean(settings.use_text_only_gemma_prompt);
    nbUseDirectorNotes.input.checked = Boolean(settings.use_director_notes);
    nbNotes.value = segment?.nb_notes || segment?.flux_notes || segment?.notes || "";
    nbPrompt.value = segment?.nb_prompt || segment?.t2i_prompt || "";
    renderNBIngredientList(segment);
  }

  return {
    advanceErnieSeedAfterRun, advanceKrea2TwoPassSeedAfterRun, advanceZEnhanceSeedAfterRun,
    advanceZImageSeedAfterRun, saveErnieImageSettingsFromPanel, saveFlowGptBrowserSettingsFromPanel,
    saveFluxKleinSettingsFromPanel, saveKrea2TwoPassSettingsFromPanel, saveNBImageSettingsFromPanel,
    saveZImageSettingsFromPanel, setFlowGptProvider, syncErnieImagePanel, syncFlowGptBrowserPanel,
    syncFluxKleinPanel, syncI2VAdvancedNodeControls, syncKrea2TwoPassPanel, syncNBImagePanel,
    syncZImageSettingsPanel,
  };
}
