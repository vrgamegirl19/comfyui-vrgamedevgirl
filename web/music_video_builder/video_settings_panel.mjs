import {
  BAD_I2V_UNET_ALIASES,
  DEFAULT_I2V_DIFFUSION_MODEL,
  DEFAULT_I2V_PASS1_SIGMAS,
  DEFAULT_I2V_PASS2_SIGMAS,
  DEFAULT_I2V_UNET,
  DEFAULT_INGREDIENTS_SAMPLER,
  DEFAULT_LTX_INGREDIENTS_HEIGHT,
  DEFAULT_LTX_INGREDIENTS_WIDTH,
  I2V_SAMPLER_OPTIONS,
  REQUIRED_LTX_ID_LORA,
  REQUIRED_LTX_ID_LORA_URL,
  REQUIRED_LTX_INGREDIENTS_LORA,
  REQUIRED_LTX_MSR_LORA,
} from "./constants.mjs";
import {
  applyCompactButtonLabel,
  makeButton,
  makeCheckbox,
  makeEditField,
  makeField,
  makeGptLinkButton,
  makeI2VNodeOverridePassPanel,
  makeInput,
  makeMiniButton,
  makeSearchableLoraPicker,
  makeSelect,
  makeSettingsPanel,
  makeSettingsSection,
  toast,
} from "./controls.mjs";
import { sceneImagePromptForEnhanceAll } from "./image_generation.mjs";
import { normalizeI2VSigmasText } from "./image_panels.mjs";
import { normalizeRTVReferenceBehavior } from "./image_references.mjs";
import { makeImageModelCard } from "./inspector.mjs";
import { wireSearchablePicker } from "./model_pickers.mjs";
import { cloneI2VVideoSettings, defaultI2VVideoSettings, repairI2VVideoSettingDimensions } from "./model_settings.mjs";
import { isInstrumentalLyricText } from "./prompt_text.mjs";
import { sortSegments } from "./segments.mjs";
import {
  createFirstLastFramePreviewSlot,
  hasLockedVideo,
  openFirstLastFrameImagePreview,
} from "./selection_preview.mjs";

export function wireVideoSettingsControls({
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
}) {
  i2vUseLora.input.addEventListener("change", updateI2VLoraVisibility);
  i2vUseLora.input.addEventListener("change", saveI2VVideoSettingsFromPanel);
  i2vLoraCount.addEventListener("input", saveI2VVideoSettingsFromPanel);
  i2vLoraCount.addEventListener("change", saveI2VVideoSettingsFromPanel);
  wireSearchablePicker(ltxMsrLoraPicker, saveI2VVideoSettingsFromPanel);
  wireSearchablePicker(ltxIngredientsLoraPicker, saveI2VVideoSettingsFromPanel);
  wireSearchablePicker(ltxIdLoraPicker, saveI2VVideoSettingsFromPanel);
  for (const control of [ltxMsrFirstPassStrength, ltxMsrSecondPassStrength, ltxMsrReferenceStrength, ltxMsrBackgroundMode]) {
    control.addEventListener("input", saveI2VVideoSettingsFromPanel);
    control.addEventListener("change", saveI2VVideoSettingsFromPanel);
  }
  ltxIngredientsFirstPassStrength.addEventListener("input", saveI2VVideoSettingsFromPanel);
  ltxIngredientsFirstPassStrength.addEventListener("change", saveI2VVideoSettingsFromPanel);
  for (const control of [ltxIdLoraFirstPassStrength, ltxIdLoraSecondPassStrength, idLoraReferenceAudioInput, idLoraIdentityScaleInput]) {
    control.addEventListener("input", saveI2VVideoSettingsFromPanel);
    control.addEventListener("change", saveI2VVideoSettingsFromPanel);
  }
  wireI2VStrengthPair(i2vPass1StrengthSlider, i2vPass1StrengthInput);
  wireI2VStrengthPair(i2vPass2StrengthSlider, i2vPass2StrengthInput);
  for (const control of [i2vPass1SamplerSelect, i2vPass1SigmasInput, i2vPass1Bypass.input, i2vPass2SamplerSelect, i2vPass2SigmasInput, i2vPass2Bypass.input]) {
    control.addEventListener("input", saveI2VVideoSettingsFromPanel);
    control.addEventListener("change", saveI2VVideoSettingsFromPanel);
  }
  for (const slot of i2vLoraSlots) {
    wireSearchablePicker(slot.picker, saveI2VVideoSettingsFromPanel);
    slot.firstPassStrength.addEventListener("input", saveI2VVideoSettingsFromPanel);
    slot.firstPassStrength.addEventListener("change", saveI2VVideoSettingsFromPanel);
    slot.secondPassStrength.addEventListener("input", saveI2VVideoSettingsFromPanel);
    slot.secondPassStrength.addEventListener("change", saveI2VVideoSettingsFromPanel);
  }
  for (const control of [i2vFpsInput, i2vWidthInput, i2vHeightInput, ltx25AspectRatioSelect, ltx25MegapixelsInput, i2vSeedInput, i2vTailLossFramesInput, i2vPreFramesInput, flfFirstGuideStrengthInput, flfLastGuideStrengthInput, flfFirstGuideFrameIndexInput, flfLastGuideFrameIndexInput, flfFirstGuideCrfInput, flfLastGuideCrfInput, flfFirstGuideBlurInput, flfLastGuideBlurInput, flfFirstAttentionStrength, flfLastAttentionStrength]) {
    control.addEventListener("input", saveI2VVideoSettingsFromPanel);
    control.addEventListener("change", saveI2VVideoSettingsFromPanel);
  }
  for (const control of [flfFirstGuideInterpolation, flfLastGuideInterpolation, flfFirstGuideCrop, flfLastGuideCrop]) {
    control.addEventListener("change", saveI2VVideoSettingsFromPanel);
  }
  flfChainPreviousEndFrame.input.addEventListener("change", () => {
    state.i2vVideoSettings = cloneI2VVideoSettings(state.i2vVideoSettings || defaultI2VVideoSettings());
    state.i2vVideoSettings.flf_chain_previous_end_frame = Boolean(flfChainPreviousEndFrame.input.checked);
    flfStructureModeSelect.value = flfChainPreviousEndFrame.input.checked ? "chained" : "independent";
    flfChainedSettingsPanel.style.display = flfChainPreviousEndFrame.input.checked ? "flex" : "none";
    syncRTVSceneImageAnchorPanel();
    renderList();
    autoSaveSessionQuiet("Global FLF chaining changed");
  });
  flfStructureModeSelect.addEventListener("change", async () => {
    const chained = flfStructureModeSelect.value !== "independent";
    state.i2vVideoSettings = cloneI2VVideoSettings(state.i2vVideoSettings || defaultI2VVideoSettings());
    state.i2vVideoSettings.flf_chain_previous_end_frame = chained;
    flfChainPreviousEndFrame.input.checked = chained;
    flfChainedSettingsPanel.style.display = chained ? "flex" : "none";
    state.segments.forEach((segment) => {
      if (segment.use_scene_i2v_video_settings && segment.i2v_video_settings) {
        segment.i2v_video_settings.flf_chain_previous_end_frame = chained;
      }
      if (!chained) {
        segment.flf_rendered_start_frame_path = "";
        segment.flf_rendered_start_frame_data = "";
        segment.flf_rendered_start_frame_name = "";
        segment.flf_rendered_source_video_path = "";
      }
    });
    syncRTVSceneImageAnchorPanel();
    render();
    await autoSaveSessionQuiet(`FLF structure changed to ${chained ? "chained" : "independent pairs"}`);
    toast(chained
      ? "FLF Chained mode: each scene can inherit the previous scene's end."
      : "FLF Independent pairs: every scene will use its own start and end image.");
  });
  flfPreGeneratePromptsFromSceneImages.input.addEventListener("change", async () => {
    state.i2vVideoSettings = cloneI2VVideoSettings(state.i2vVideoSettings || defaultI2VVideoSettings());
    state.i2vVideoSettings.flf_pregenerate_prompts_from_scene_images = Boolean(flfPreGeneratePromptsFromSceneImages.input.checked);
    await autoSaveSessionQuiet("FLF prompt pre-generation mode changed");
  });
  flfMatchPreviousClipColor.input.addEventListener("change", async () => {
    state.i2vVideoSettings = cloneI2VVideoSettings(state.i2vVideoSettings || defaultI2VVideoSettings());
    state.i2vVideoSettings.flf_match_previous_clip_color = Boolean(flfMatchPreviousClipColor.input.checked);
    await autoSaveSessionQuiet("FLF opening color match changed");
  });
  for (const control of [flfColorMatchStrengthInput, flfColorMatchFadeInput]) {
    const saveGlobalColorMatch = () => {
      state.i2vVideoSettings = cloneI2VVideoSettings(state.i2vVideoSettings || defaultI2VVideoSettings());
      state.i2vVideoSettings.flf_color_match_strength = Math.max(0, Math.min(1, Number(flfColorMatchStrengthInput.value || 0)));
      state.i2vVideoSettings.flf_color_match_fade_seconds = Math.max(0.05, Math.min(30, Number(flfColorMatchFadeInput.value || 1)));
    };
    control.addEventListener("input", saveGlobalColorMatch);
    control.addEventListener("change", async () => {
      saveGlobalColorMatch();
      await autoSaveSessionQuiet("FLF opening color match settings changed");
    });
  }
  flfRenderChainSourceSelect.addEventListener("change", async () => {
    state.i2vVideoSettings = cloneI2VVideoSettings(state.i2vVideoSettings || defaultI2VVideoSettings());
    state.i2vVideoSettings.flf_render_chain_start_source = flfRenderChainSourceSelect.value === "previous_image" ? "previous_image" : "rendered_frame";
    syncRTVSceneImageAnchorPanel();
    renderList();
    await autoSaveSessionQuiet("FLF actual render-chain source changed");
  });
  flfGlobalTransitionTypeSelect.addEventListener("change", async () => {
    saveI2VVideoSettingsFromPanel();
    await autoSaveSessionQuiet("Global First Last Frame transition type changed");
  });
  flfGemmaContextModeSelect.addEventListener("change", async () => {
    saveI2VVideoSettingsFromPanel();
    await autoSaveSessionQuiet("FLF Gemma visual context changed");
    toast(`FLF Gemma visual context: ${flfGemmaContextModeSelect.options[flfGemmaContextModeSelect.selectedIndex]?.textContent || flfGemmaContextModeSelect.value}`);
  });
  flfRestoreWorkflowDefaultsButton.addEventListener("click", async () => {
    flfGemmaContextModeSelect.value = "images_story";
    flfFirstGuideStrengthInput.value = "0.7";
    flfLastGuideStrengthInput.value = "0.7";
    flfFirstGuideFrameIndexInput.value = "0";
    flfLastGuideFrameIndexInput.value = "-1";
    flfFirstGuideCrfInput.value = "29";
    flfLastGuideCrfInput.value = "29";
    flfFirstGuideBlurInput.value = "1";
    flfLastGuideBlurInput.value = "1";
    flfFirstGuideInterpolation.value = "lanczos";
    flfLastGuideInterpolation.value = "lanczos";
    flfFirstGuideCrop.value = "center";
    flfLastGuideCrop.value = "center";
    flfFirstAttentionStrength.value = "0.9";
    flfLastAttentionStrength.value = "1";
    saveI2VVideoSettingsFromPanel();
    await autoSaveSessionQuiet("FLF workflow defaults restored");
    toast("First Last Frame guide settings restored to the hidden workflow defaults.");
  });
  imageContinuityEnabled.input.checked = Boolean(state.imageContinuityEnabled);
  imageContinuityStrength.value = state.imageContinuityStrength || "balanced";
  imageContinuityEnabled.input.addEventListener("change", async () => {
    state.imageContinuityEnabled = Boolean(imageContinuityEnabled.input.checked);
    await autoSaveSessionQuiet("Image All continuity changed");
  });
  imageContinuityStrength.addEventListener("change", async () => {
    state.imageContinuityStrength = ["close", "creative"].includes(imageContinuityStrength.value) ? imageContinuityStrength.value : "balanced";
    await autoSaveSessionQuiet("Image All continuity strength changed");
  });
}

export function buildVideoSettingsPanels({ previewFluxButton, previewNBButton, wrapCreateSceneVideoActions }) {
  const videoModeChooser = document.createElement("div");
  videoModeChooser.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:repeat(4,auto);gap:6px;";
  const styleVideoModeCard = (card, column, row) => {
    card.style.gridColumn = String(column);
    card.style.gridRow = String(row);
    card.style.width = "100%";
    card.style.minWidth = "0";
    card.style.flex = "1 1 auto";
    card.style.padding = "0 8px";
    card.style.justifySelf = "stretch";
  };
  const imageToVideoCard = makeImageModelCard("Image to Video", "i2v");
  styleVideoModeCard(imageToVideoCard, 1, 1);
  const idLoraVideoCard = makeImageModelCard("ID-LoRA I2V", "id_lora");
  styleVideoModeCard(idLoraVideoCard, 2, 1);
  const textToVideoCard = makeImageModelCard("Text to Video", "t2v");
  styleVideoModeCard(textToVideoCard, 1, 2);
  const referenceToVideoCard = makeImageModelCard("Reference to Video", "rtv");
  styleVideoModeCard(referenceToVideoCard, 2, 2);
  const ingredientsToVideoCard = makeImageModelCard("Ingredients to Video", "ingredients");
  styleVideoModeCard(ingredientsToVideoCard, 1, 3);
  const firstLastFrameVideoCard = makeImageModelCard("First Last Frame", "flf");
  styleVideoModeCard(firstLastFrameVideoCard, 2, 3);
  const importCustomVideoCard = makeImageModelCard("Import Custom Video", "import");
  styleVideoModeCard(importCustomVideoCard, 1, 4);
  videoModeChooser.append(imageToVideoCard, idLoraVideoCard, textToVideoCard, referenceToVideoCard, ingredientsToVideoCard, firstLastFrameVideoCard, importCustomVideoCard);
  const importCustomVideoComingSoon = document.createElement("div");
  importCustomVideoComingSoon.textContent = "Import Custom Video is coming soon. Use Restore Video from a scene's right-click menu for manual scene repair.";
  importCustomVideoComingSoon.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;color:#dbeafe;padding:10px;font-size:12px;line-height:1.45;";
  const importCustomVideoPanel = makeSettingsPanel([
    importCustomVideoComingSoon,
  ]);
  const i2vUseGgufModel = makeCheckbox("Use GGUF model?", true);
  const i2vUnetPicker = makeSearchableLoraPicker("");
  const i2vDiffusionModelPicker = makeSearchableLoraPicker(DEFAULT_I2V_DIFFUSION_MODEL);
  const i2vUnetModelField = makeField("Unet model", i2vUnetPicker.wrapper);
  const i2vDiffusionModelField = makeField("Diffusion model", i2vDiffusionModelPicker.wrapper);
  const i2vUseSageAttention = makeCheckbox("Use Sage Attention", false);
  const i2vEnableFp16Accumulation = makeCheckbox("Enable fp16 accumulation", false);
  const i2vDiffusionLoaderAdvanced = makeSettingsSection("Advanced Diffusion Loader Settings", [
    i2vUseSageAttention.wrapper,
    i2vEnableFp16Accumulation.wrapper,
  ], false);
  const i2vVaePicker = makeSearchableLoraPicker("");
  const i2vClip1Picker = makeSearchableLoraPicker("");
  const i2vClip2Picker = makeSearchableLoraPicker("");
  const i2vUpscalePicker = makeSearchableLoraPicker("");
  const i2vAudioVaePicker = makeSearchableLoraPicker("");
  const i2vFpsInput = makeInput("24", "number");
  const i2vWidthInput = makeInput("1920", "number");
  const i2vHeightInput = makeInput("1080", "number");
  const ltx25AspectRatioSelect = makeSelect([
    "1:1 (Square)", "2:3 (Portrait Photo)", "3:2 (Photo)", "3:4 (Portrait Standard)",
    "4:3 (Standard)", "9:16 (Portrait Widescreen)", "16:9 (Widescreen)", "21:9 (Ultrawide)",
  ], "16:9 (Widescreen)");
  const ltx25MegapixelsInput = makeInput("1.2", "number");
  ltx25MegapixelsInput.min = "0.1";
  ltx25MegapixelsInput.max = "16";
  ltx25MegapixelsInput.step = "0.1";
  const i2vSeedInput = makeInput("69", "number");
  const i2vTailLossFramesInput = makeInput("25", "number");
  i2vTailLossFramesInput.min = "0";
  i2vTailLossFramesInput.step = "1";
  const i2vPreFramesInput = makeInput("50", "number");
  i2vPreFramesInput.min = "0";
  i2vPreFramesInput.step = "1";
  const i2vSrtSplitAdvancedNote = document.createElement("div");
  i2vSrtSplitAdvancedNote.textContent = "Advanced";
  i2vSrtSplitAdvancedNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;margin-top:-4px;";
  const i2vUseLora = makeCheckbox("Use video LoRAs?", false);
  const i2vLoraPanel = document.createElement("div");
  i2vLoraPanel.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const i2vLoraHintRow = document.createElement("div");
  i2vLoraHintRow.style.cssText = "display:flex;justify-content:flex-end;";
  const i2vLoraHintButton = makeButton("?", "neutral");
  i2vLoraHintButton.title = "Optional extra video LoRAs use one strength for Reference to Video, or separate pass strengths for I2V/T2V.";
  i2vLoraHintButton.style.cssText += "width:34px;padding:7px 0;";
  i2vLoraHintRow.append(i2vLoraHintButton);
  const flfTransitionLoraNote = document.createElement("div");
  flfTransitionLoraNote.textContent = "FLF recommendation: use an LTX 2.3 transition LoRA when the endpoints differ substantially in subject, framing, material, style, or scene content. It helps LTX create a continuous semantic transformation instead of a late cut or dissolve.";
  flfTransitionLoraNote.style.cssText = "display:none;border:1px solid #7c3aed;border-radius:7px;background:#2e1065;color:#ddd6fe;padding:8px 10px;font-size:11px;line-height:1.4;";
  const i2vLoraCount = makeInput("0", "number");
  i2vLoraCount.min = "0";
  i2vLoraCount.max = "4";
  const i2vLoraRows = document.createElement("div");
  i2vLoraRows.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const i2vLoraSlots = [];
  for (let slot = 1; slot <= 4; slot++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:1fr 84px 84px;gap:8px;";
    const picker = makeSearchableLoraPicker("[none]");
    const firstPassStrength = makeInput("1", "number");
    firstPassStrength.step = "0.01";
    const secondPassStrength = makeInput("1", "number");
    secondPassStrength.step = "0.01";
    const firstPassField = makeField("Pass 1", firstPassStrength);
    const firstPassLabel = firstPassField.querySelector("span");
    const secondPassField = makeField("Pass 2", secondPassStrength);
    row.append(
      makeField(`Video LoRA ${slot}`, picker.wrapper),
      firstPassField,
      secondPassField
    );
    i2vLoraRows.append(row);
    i2vLoraSlots.push({ row, picker, firstPassStrength, secondPassStrength, firstPassLabel, secondPassField });
  }
  const ltxMsrRequiredPanel = document.createElement("div");
  ltxMsrRequiredPanel.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #155e75;border-radius:8px;background:#082f49;padding:10px;";
  const ltxMsrRequiredNote = document.createElement("div");
  ltxMsrRequiredNote.textContent = "Required for Reference to Video. This LoRA is always applied before optional video LoRAs; tune its strength here.";
  ltxMsrRequiredNote.style.cssText = "font-size:11px;color:#bae6fd;line-height:1.35;";
  const ltxMsrLoraPicker = makeSearchableLoraPicker(REQUIRED_LTX_MSR_LORA);
  const ltxMsrFirstPassStrength = makeInput("1", "number");
  ltxMsrFirstPassStrength.step = "0.01";
  const ltxMsrSecondPassStrength = makeInput("1", "number");
  ltxMsrSecondPassStrength.step = "0.01";
  const ltxMsrReferenceStrength = makeSelect([
    "auto - based on subject count",
    "17 - light",
    "25 - balanced",
    "33 - strong",
    "41 - strongest",
  ], "auto - based on subject count");
  const ltxMsrBackgroundMode = makeSelect([
    "no background reference",
    "use location/background reference",
  ], "no background reference");
  const ltxMsrStrengthGrid = document.createElement("div");
  ltxMsrStrengthGrid.style.cssText = "display:grid;grid-template-columns:1fr 84px;gap:8px;";
  ltxMsrStrengthGrid.append(
    makeField("Required MSR LoRA", ltxMsrLoraPicker.wrapper),
    makeField("Strength", ltxMsrFirstPassStrength)
  );
  const ltxMsrReferenceGrid = document.createElement("div");
  ltxMsrReferenceGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  ltxMsrReferenceGrid.append(makeField("Reference strength", ltxMsrReferenceStrength), makeField("Background", ltxMsrBackgroundMode));
  ltxMsrRequiredPanel.append(ltxMsrRequiredNote, ltxMsrStrengthGrid, ltxMsrReferenceGrid);
  const rtvReferenceBehaviorSelect = makeSelect([
    { value: "none", label: "None" },
    { value: "character_anchor", label: "Character Anchor: use scene image as 2nd ref" },
    { value: "first_last_frame", label: "First Last Frame: first image to end image" },
  ], "none");
  rtvReferenceBehaviorSelect.title = "Choose how Reference to Video should pack MSR reference images. Only one behavior can be active at a time.";
  const rtvReferenceBehaviorField = makeField("Mode", rtvReferenceBehaviorSelect);
  const rtvReferenceBehaviorNote = document.createElement("div");
  rtvReferenceBehaviorNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;margin-top:-4px;";
  const firstLastFramePreviewPanel = document.createElement("div");
  firstLastFramePreviewPanel.style.cssText = "display:none;border:1px solid #155e75;border-radius:7px;background:#071922;padding:9px;gap:8px;flex-direction:column;";
  const firstLastFramePreviewGrid = document.createElement("div");
  firstLastFramePreviewGrid.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 22px minmax(0,1fr);gap:8px;align-items:stretch;";
  const firstLastFrameStatus = document.createElement("div");
  firstLastFrameStatus.style.cssText = "font-size:11px;line-height:1.35;color:#bae6fd;overflow-wrap:anywhere;";
  const createSceneEndFrameButton = makeMiniButton("Create End Frame");
  createSceneEndFrameButton.title = "Generate or replace the active scene's First Last Frame end image using the current image model.";
  const planSceneEndMotionButton = makeMiniButton("Create Motion Plan");
  planSceneEndMotionButton.title = "Inspect this scene's start image and create the provisional I2V motion plan that will be used to design its end frame.";
  const finalizeSceneFLFPromptButton = makeMiniButton("Create Final FLF Prompt");
  finalizeSceneFLFPromptButton.title = "Inspect the actual start and end images and rewrite the saved video prompt so it connects those exact endpoints.";
  const clearSceneEndFrameButton = makeMiniButton("Clear End Frame");
  const loadSceneEndFrameButton = makeMiniButton("Load End Frame");
  loadSceneEndFrameButton.title = "Load an existing image as this scene's end frame. No image provider is required.";
  const sceneEndFrameFileInput = document.createElement("input");
  sceneEndFrameFileInput.type = "file";
  sceneEndFrameFileInput.accept = "image/png,image/jpeg,image/webp";
  sceneEndFrameFileInput.style.display = "none";
  clearSceneEndFrameButton.title = "Remove the active scene's stored First Last Frame end image.";
  const flfEndpointModeSelect = makeSelect([
    { value: "auto", label: "Auto — derive the ending from the motion prompt" },
    { value: "custom", label: "Custom — follow my ending direction" },
  ], "auto");
  const flfCustomEndDirection = document.createElement("textarea");
  flfCustomEndDirection.placeholder = "Optional custom ending, e.g. Pull back to a full-body view as she reaches the rainy window.";
  flfCustomEndDirection.style.cssText = "width:100%;box-sizing:border-box;min-height:74px;resize:vertical;border:1px solid #334155;border-radius:6px;background:#020617;color:#e2e8f0;padding:8px;font-size:11px;line-height:1.4;";
  ["keydown", "keypress", "keyup"].forEach((eventName) => flfCustomEndDirection.addEventListener(eventName, (event) => event.stopPropagation()));
  const flfPerScenePlanner = document.createElement("div");
  flfPerScenePlanner.style.cssText = "display:none;border:1px solid #164e63;border-radius:7px;background:#06151d;padding:9px;gap:8px;flex-direction:column;";
  const flfPerScenePlannerTitle = document.createElement("div");
  flfPerScenePlannerTitle.innerHTML = '<strong style="color:#cffafe;">Independent-pair endpoint planner</strong><div style="margin-top:3px;color:#94a3b8;font-size:11px;line-height:1.4;">Auto reads the start image and scene context to write a provisional motion plan. Custom adds your required ending to that plan. The end image is generated afterward; the final FLF prompt is created only after both real images exist.</div>';
  const flfMotionPlanDetails = document.createElement("details");
  const flfMotionPlanSummary = document.createElement("summary");
  flfMotionPlanSummary.textContent = "Saved provisional motion plan";
  flfMotionPlanSummary.style.cssText = "cursor:pointer;color:#bae6fd;font-weight:800;font-size:11px;";
  const flfMotionPlanPreview = document.createElement("pre");
  flfMotionPlanPreview.style.cssText = "margin:7px 0 0;max-height:180px;overflow:auto;white-space:pre-wrap;border:1px solid #1e3a5f;border-radius:5px;background:#020617;color:#cbd5e1;padding:8px;font-size:10px;line-height:1.4;";
  flfMotionPlanDetails.append(flfMotionPlanSummary, flfMotionPlanPreview);
  const flfEndpointPromptDetails = document.createElement("details");
  const flfEndpointPromptSummary = document.createElement("summary");
  flfEndpointPromptSummary.textContent = "Saved end-frame image prompt";
  flfEndpointPromptSummary.style.cssText = flfMotionPlanSummary.style.cssText;
  const flfEndpointPromptPreview = document.createElement("pre");
  flfEndpointPromptPreview.style.cssText = flfMotionPlanPreview.style.cssText;
  flfEndpointPromptDetails.append(flfEndpointPromptSummary, flfEndpointPromptPreview);
  flfPerScenePlanner.append(
    flfPerScenePlannerTitle,
    makeField("Endpoint planning", flfEndpointModeSelect),
    makeField("Custom ending direction", flfCustomEndDirection),
    flfMotionPlanDetails,
    flfEndpointPromptDetails,
  );
  const firstLastFrameActions = document.createElement("div");
  firstLastFrameActions.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;";
  firstLastFrameActions.append(loadSceneEndFrameButton, planSceneEndMotionButton, createSceneEndFrameButton, finalizeSceneFLFPromptButton, clearSceneEndFrameButton, sceneEndFrameFileInput);
  firstLastFramePreviewPanel.append(firstLastFramePreviewGrid, firstLastFrameStatus, flfPerScenePlanner, firstLastFrameActions);
  const rtvSceneImageAnchorSection = makeSettingsSection("Reference Behavior", [
    rtvReferenceBehaviorField,
    rtvReferenceBehaviorNote,
    firstLastFramePreviewPanel,
  ]);
  rtvSceneImageAnchorSection.style.display = "none";
  const ltxIngredientsRequiredPanel = document.createElement("div");
  ltxIngredientsRequiredPanel.style.cssText = ltxMsrRequiredPanel.style.cssText;
  const ltxIngredientsRequiredNote = document.createElement("div");
  ltxIngredientsRequiredNote.textContent = "Required for Ingredients to Video. This LoRA is always applied on pass 1; pass 2 is kept at 0 for the required LoRA.";
  ltxIngredientsRequiredNote.style.cssText = ltxMsrRequiredNote.style.cssText;
  const ltxIngredientsLoraPicker = makeSearchableLoraPicker(REQUIRED_LTX_INGREDIENTS_LORA);
  const ltxIngredientsFirstPassStrength = makeInput("1", "number");
  ltxIngredientsFirstPassStrength.step = "0.01";
  const ltxIngredientsStrengthGrid = document.createElement("div");
  ltxIngredientsStrengthGrid.style.cssText = "display:grid;grid-template-columns:1fr 84px;gap:8px;";
  ltxIngredientsStrengthGrid.append(
    makeField("Required Ingredients LoRA", ltxIngredientsLoraPicker.wrapper),
    makeField("Pass 1", ltxIngredientsFirstPassStrength)
  );
  ltxIngredientsRequiredPanel.append(ltxIngredientsRequiredNote, ltxIngredientsStrengthGrid);
  const ltxIdLoraRequiredPanel = document.createElement("div");
  ltxIdLoraRequiredPanel.style.cssText = ltxMsrRequiredPanel.style.cssText;
  const ltxIdLoraRequiredNote = document.createElement("div");
  ltxIdLoraRequiredNote.textContent = "Required for ID-LoRA I2V. This LoRA is always applied in slot 1; optional video LoRAs start after it.";
  ltxIdLoraRequiredNote.style.cssText = ltxMsrRequiredNote.style.cssText;
  const ltxIdLoraHelpRow = document.createElement("div");
  ltxIdLoraHelpRow.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;";
  const ltxIdLoraLinkButton = makeGptLinkButton("Open LoRA Page", REQUIRED_LTX_ID_LORA_URL);
  ltxIdLoraLinkButton.style.cssText += "padding:6px 9px;font-size:11px;";
  ltxIdLoraHelpRow.append(ltxIdLoraRequiredNote, ltxIdLoraLinkButton);
  const ltxIdLoraPicker = makeSearchableLoraPicker(REQUIRED_LTX_ID_LORA);
  const ltxIdLoraFirstPassStrength = makeInput("1", "number");
  ltxIdLoraFirstPassStrength.step = "0.01";
  const ltxIdLoraSecondPassStrength = makeInput("1", "number");
  ltxIdLoraSecondPassStrength.step = "0.01";
  const ltxIdLoraStrengthGrid = document.createElement("div");
  ltxIdLoraStrengthGrid.style.cssText = "display:grid;grid-template-columns:1fr 84px 84px;gap:8px;";
  ltxIdLoraStrengthGrid.append(
    makeField("Required ID-LoRA", ltxIdLoraPicker.wrapper),
    makeField("Pass 1", ltxIdLoraFirstPassStrength),
    makeField("Pass 2", ltxIdLoraSecondPassStrength)
  );
  ltxIdLoraRequiredPanel.append(ltxIdLoraHelpRow, ltxIdLoraStrengthGrid);
  const idLoraReferenceAudioInput = makeInput("");
  const pickIdLoraReferenceAudioButton = makeButton("Pick");
  const idLoraReferenceAudioField = makeEditField("Fallback voice sample", idLoraReferenceAudioInput, pickIdLoraReferenceAudioButton);
  const idLoraReferenceAudioNote = document.createElement("div");
  idLoraReferenceAudioNote.textContent = "Use ID-LoRA Ref Builder for normal per-character voices. This fallback is only used when a scene's selected character has no voice sample.";
  idLoraReferenceAudioNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;margin-top:-4px;";
  const idLoraIdentityScaleInput = makeInput("3", "number");
  idLoraIdentityScaleInput.step = "0.1";
  const idLoraIdentityGrid = document.createElement("div");
  idLoraIdentityGrid.style.cssText = "display:grid;grid-template-columns:1fr;gap:8px;";
  idLoraIdentityGrid.append(
    makeField("Identity scale", idLoraIdentityScaleInput)
  );
  i2vLoraPanel.append(i2vLoraHintRow, flfTransitionLoraNote, makeField("Video LoRA count", i2vLoraCount), i2vLoraRows);
  const i2vSettingsGrid = document.createElement("div");
  i2vSettingsGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  i2vSettingsGrid.append(makeField("FPS", i2vFpsInput), makeField("Seed", i2vSeedInput), makeField("Width", i2vWidthInput), makeField("Height", i2vHeightInput));
  const ltx25ResolutionGrid = document.createElement("div");
  ltx25ResolutionGrid.style.cssText = i2vSettingsGrid.style.cssText;
  ltx25ResolutionGrid.append(
    makeField("Aspect ratio", ltx25AspectRatioSelect),
    makeField("Megapixels", ltx25MegapixelsInput, "LTX 2.5 resolves width and height from this target and rounds them to a multiple of 32."),
  );
  const ltxIngredientsResolutionWarning = document.createElement("div");
  ltxIngredientsResolutionWarning.textContent = "Ingredients LoRA was trained at 768x448. Other resolutions, including portrait, can break quality or composition. Final output is 2x after the second pass.";
  ltxIngredientsResolutionWarning.style.cssText = "display:none;border:1px solid #991b1b;border-radius:7px;background:#450a0a;color:#fecaca;padding:8px 10px;font-size:11px;line-height:1.35;font-weight:800;";
  const i2vSrtSplitGrid = document.createElement("div");
  i2vSrtSplitGrid.style.cssText = i2vSettingsGrid.style.cssText;
  i2vSrtSplitGrid.append(makeField("Cool Down Frames", i2vTailLossFramesInput), makeField("Warm Up Frames", i2vPreFramesInput));
  const i2vWarmCooldownSection = makeSettingsSection("Warm/Cool Frames", [
    i2vSrtSplitAdvancedNote,
    i2vSrtSplitGrid,
  ]);
  const flfGuideSettingsGrid = document.createElement("div");
  flfGuideSettingsGrid.style.cssText = i2vSettingsGrid.style.cssText;
  const flfFirstGuideStrengthInput = makeInput("0.7", "number");
  const flfLastGuideStrengthInput = makeInput("0.7", "number");
  const flfFirstGuideFrameIndexInput = makeInput("0", "number");
  const flfLastGuideFrameIndexInput = makeInput("-1", "number");
  for (const control of [flfFirstGuideStrengthInput, flfLastGuideStrengthInput]) {
    control.min = "0"; control.max = "1"; control.step = "0.01";
  }
  flfGuideSettingsGrid.append(makeField("First Frame Guide", flfFirstGuideStrengthInput), makeField("Last Frame Guide", flfLastGuideStrengthInput));
  flfGuideSettingsGrid.append(makeField("First Guide Frame Index", flfFirstGuideFrameIndexInput), makeField("Last Guide Frame Index", flfLastGuideFrameIndexInput));
  const flfFirstGuideCrfInput = makeInput("29", "number");
  const flfLastGuideCrfInput = makeInput("29", "number");
  const flfFirstGuideBlurInput = makeInput("1", "number");
  const flfLastGuideBlurInput = makeInput("1", "number");
  for (const control of [flfFirstGuideCrfInput, flfLastGuideCrfInput]) { control.min = "0"; control.max = "51"; control.step = "1"; }
  for (const control of [flfFirstGuideBlurInput, flfLastGuideBlurInput]) { control.min = "0"; control.max = "7"; control.step = "1"; }
  const flfFirstGuideInterpolation = makeSelect(["lanczos", "bislerp", "nearest", "bilinear", "bicubic", "area", "nearest-exact"], "lanczos");
  const flfLastGuideInterpolation = makeSelect(["lanczos", "bislerp", "nearest", "bilinear", "bicubic", "area", "nearest-exact"], "lanczos");
  const flfFirstGuideCrop = makeSelect(["center", "disabled"], "center");
  const flfLastGuideCrop = makeSelect(["center", "disabled"], "center");
  const flfFirstAttentionStrength = makeInput("0.90", "number");
  const flfLastAttentionStrength = makeInput("1.00", "number");
  for (const control of [flfFirstAttentionStrength, flfLastAttentionStrength]) { control.min = "0"; control.max = "1"; control.step = "0.01"; }
  const flfAdvancedDetails = document.createElement("details");
  flfAdvancedDetails.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:8px;";
  const flfAdvancedSummary = document.createElement("summary");
  flfAdvancedSummary.textContent = "Advanced guide preprocessing and attention";
  flfAdvancedSummary.style.cssText = "cursor:pointer;color:#bae6fd;font-weight:800;";
  const flfAdvancedGrid = document.createElement("div");
  flfAdvancedGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;";
  flfAdvancedGrid.append(
    makeField("First CRF", flfFirstGuideCrfInput), makeField("Last CRF", flfLastGuideCrfInput),
    makeField("First blur radius", flfFirstGuideBlurInput), makeField("Last blur radius", flfLastGuideBlurInput),
    makeField("First interpolation", flfFirstGuideInterpolation), makeField("Last interpolation", flfLastGuideInterpolation),
    makeField("First crop", flfFirstGuideCrop), makeField("Last crop", flfLastGuideCrop),
    makeField("First attention strength", flfFirstAttentionStrength), makeField("Last attention strength", flfLastAttentionStrength)
  );
  const flfAdvancedNote = document.createElement("div");
  flfAdvancedNote.style.cssText = "font-size:11px;color:#cbd5e1;line-height:1.45;margin-top:10px;white-space:pre-line;";
  flfAdvancedNote.textContent = "CRF: preprocessing compression; higher values can encourage motion, lower values preserve more image detail.\nBlur radius: softens guide detail; more blur can release motion but reduces exactness.\nInterpolation: resize method; Lanczos is the recommended quality default.\nCrop: center crops to the target frame; Disabled resizes without the node's center-crop step.\nAttention strength: how strongly each image influences LTX self-attention, separately from latent guide strength.\nFrame index: 0 anchors the opening; negative values count backward from the end.\nStrength: how exactly the latent follows that guide image.";
  flfAdvancedDetails.append(flfAdvancedSummary, flfAdvancedGrid, flfAdvancedNote);
  const flfRestoreWorkflowDefaultsButton = makeButton("Restore Workflow Defaults", "neutral");
  flfRestoreWorkflowDefaultsButton.title = "Reset all First Last Frame guide controls to the values currently stored in the hidden workflow.";
  const flfGuideSettingsNote = document.createElement("div");
  flfGuideSettingsNote.textContent = "The current workflow defaults are preserved: strengths 0.70, indexes 0 and -1, CRF 29, blur 1, Lanczos, center crop, and attention strengths 0.90 and 1.00.";
  flfGuideSettingsNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;margin-top:-4px;";
  const flfDurationGuidanceNote = document.createElement("div");
  flfDurationGuidanceNote.textContent = "Duration guidance: similar endpoints or simple camera changes may work in 4–5 seconds. If the images differ substantially in anatomy, identity, material, framing, style, or scene content, use at least 8 seconds so LTX has enough time to complete and stabilize the transition. Keep 24 FPS; increasing FPS does not add transition time.";
  flfDurationGuidanceNote.style.cssText = "border:1px solid #0e7490;border-radius:7px;background:#083344;color:#cffafe;padding:8px 10px;font-size:11px;line-height:1.45;";
  const flfStructureModeSelect = makeSelect([
    { value: "chained", label: "Chained — previous end becomes next start" },
    { value: "independent", label: "Independent pairs — every scene owns a start + end" },
  ], "chained");
  const flfChainPreviousEndFrame = makeCheckbox("Global: reuse previous scene's end frame as the next first frame", true);
  const flfRenderChainSourceSelect = makeSelect([
    { value: "rendered_frame", label: "Previous video's extracted final frame" },
    { value: "previous_image", label: "Previous scene's assigned end image" },
  ], "rendered_frame");
  const flfPreGeneratePromptsFromSceneImages = makeCheckbox("Global: pre-generate prompts from scene images", false);
  const flfMatchPreviousClipColor = makeCheckbox("Global: match previous clip color at the start", false);
  const flfColorMatchStrengthInput = makeInput("0.85", "number");
  flfColorMatchStrengthInput.min = "0";
  flfColorMatchStrengthInput.max = "1";
  flfColorMatchStrengthInput.step = "0.05";
  const flfColorMatchFadeInput = makeInput("1.0", "number");
  flfColorMatchFadeInput.min = "0.05";
  flfColorMatchFadeInput.max = "30";
  flfColorMatchFadeInput.step = "0.05";
  const flfColorMatchGrid = document.createElement("div");
  flfColorMatchGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  flfColorMatchGrid.append(makeField("Color-match strength", flfColorMatchStrengthInput), makeField("Fade out (seconds)", flfColorMatchFadeInput));
  const flfGlobalTransitionTypeSelect = makeSelect([
    { value: "auto", label: "Auto: Gemma decides" },
    { value: "smooth", label: "Smooth Transition" },
    { value: "morph", label: "Surreal Morph" },
  ], "auto");
  const flfGemmaContextModeSelect = makeSelect([
    { value: "images_story", label: "Images + story beat (recommended)" },
    { value: "images_only", label: "Images only" },
    { value: "full", label: "Full scene context" },
  ], "images_story");
  const flfGemmaContextNote = document.createElement("div");
  flfGemmaContextNote.textContent = "Controls only how Gemma designs the visual transition. Images always remain endpoint truth. The exact lyric/speaking line and selected facial-performance direction are added afterward, even in Images only mode.";
  flfGemmaContextNote.style.cssText = flfGuideSettingsNote.style.cssText;
  const flfChainNote = document.createElement("div");
  flfChainNote.textContent = "Scene 1 always uses its own selected image. For Scene 2 onward, choose whether the actual video workflow starts from the prior rendered video's extracted final frame or the prior scene's assigned end image.";
  flfChainNote.style.cssText = flfGuideSettingsNote.style.cssText;
  const flfPreGenerateNote = document.createElement("div");
  flfPreGenerateNote.textContent = "Prompt-only shortcut: when all scene/end images already exist, Gemma uses the previous scene's assigned end image as a provisional start reference and can prepare every prompt before rendering. This does not choose the workflow's actual render input; the Actual chained render start selector above controls whether rendering uses the extracted video frame or the previous assigned image. If no provisional image exists, prompt generation automatically waits for the rendered frame.";
  flfPreGenerateNote.style.cssText = flfGuideSettingsNote.style.cssText;
  const flfColorMatchNote = document.createElement("div");
  flfColorMatchNote.textContent = "Optional post-process for chained FLF clips. It samples the previous video's actual final frame, color-matches the beginning of the new clip, and smoothly fades the correction away. It does not use the extracted frame as an LTX guide or alter the previous video. Best paired with Previous scene's assigned end image above.";
  flfColorMatchNote.style.cssText = flfGuideSettingsNote.style.cssText;
  const flfWorkflowHint = document.createElement("details");
  flfWorkflowHint.open = true;
  flfWorkflowHint.style.cssText = "border:1px solid #0891b2;border-radius:8px;background:#06202a;color:#e0f2fe;padding:10px;font-size:11px;line-height:1.48;";
  const flfWorkflowHintSummary = document.createElement("summary");
  flfWorkflowHintSummary.textContent = "How Chained and Independent First/Last Frame workflows work";
  flfWorkflowHintSummary.style.cssText = "cursor:pointer;font-weight:900;color:#cffafe;font-size:12px;";
  const flfWorkflowHintBody = document.createElement("div");
  flfWorkflowHintBody.innerHTML = `
    <div style="margin-top:10px;border:1px solid #155e75;border-radius:6px;background:#04151d;padding:8px;"><strong>Where to run it:</strong> choose the FLF structure here, then click <strong>Image All</strong>. Choose <strong>Build Independent Start + End Pairs</strong> for a safe resume, one of the independent redo choices to replace work, or <strong>Resume FLF Image Chain</strong> for the shared-endpoint workflow. Image All prepares images and prompts only; use Render All afterward.</div>
    <div style="margin-top:10px;"><strong style="color:#fef3c7;">Chained</strong></div>
    <div>Creates one opening image for the first scene and one destination image per scene. Scene 1 end becomes Scene 2 start, Scene 2 end becomes Scene 3 start, and so on. During rendering you can start the next clip from the previous assigned end image or from the previous video's extracted final frame. Use this when you want continuous visual handoffs between scenes.</div>
    <div style="margin-top:10px;"><strong style="color:#bbf7d0;">Independent pairs</strong></div>
    <div>Every scene owns two separate images. The batch builder always finishes <em>all start images first</em>, then makes provisional I2V motion plans for all scenes, then returns to Scene 1 and creates every end image, then inspects each real start/end pair to create the final FLF render prompts. No image is shared with another scene.</div>
    <ol style="margin:8px 0 0 20px;padding:0;">
      <li><strong>Pass 1 — starts:</strong> normal Image All logic creates or keeps one start image for every target scene.</li>
      <li><strong>Pass 2 — motion plans:</strong> Gemma views each completed start image and writes how the shot moves and exactly where it finishes. A custom per-scene ending direction is followed when supplied.</li>
      <li><strong>Pass 3 — ends:</strong> the selected image model creates each end frame from the start image, motion plan, mapped character/location descriptions, and supported reference sheets.</li>
      <li><strong>Pass 4 — final prompts:</strong> Gemma views the actual start and actual end together and writes the final FLF prompt that connects those real images.</li>
    </ol>
    <div style="margin-top:9px;"><strong>Reference behavior:</strong> Flux/Klein, NanoBanana, and Browser AI modes can receive the start image alongside their configured reference ingredients. ZImage, Ernie, and Krea use their supported image-to-image path when enabled and always receive mapped character/clothing and location descriptions in the endpoint instructions. If an endpoint reveals clothing that was outside the start image, make the Reference Builder description explicit.</div>
    <div style="margin-top:9px;"><strong>Resume safety:</strong> “Build Independent Start + End Pairs” keeps completed starts, motion plans, ends, and final prompts. It resumes only missing stages. The redo option keeps starts but rebuilds the motion plans, end images, and final prompts.</div>`;
  flfWorkflowHint.append(flfWorkflowHintSummary, flfWorkflowHintBody);
  const flfChainedSettingsPanel = document.createElement("div");
  flfChainedSettingsPanel.style.cssText = "display:flex;flex-direction:column;gap:8px;border:1px solid #334155;border-radius:7px;background:#0b1220;padding:9px;";
  flfChainedSettingsPanel.append(makeField("Actual chained render start", flfRenderChainSourceSelect), flfChainNote, flfPreGeneratePromptsFromSceneImages.wrapper, flfPreGenerateNote, flfMatchPreviousClipColor.wrapper, flfColorMatchGrid, flfColorMatchNote);
  const flfGuideSettingsSection = makeSettingsSection("First / Last Frame Settings", [makeField("FLF structure", flfStructureModeSelect), flfWorkflowHint, makeField("Global transition type", flfGlobalTransitionTypeSelect), makeField("Gemma visual context", flfGemmaContextModeSelect), flfGemmaContextNote, flfDurationGuidanceNote, flfGuideSettingsGrid, flfGuideSettingsNote, flfRestoreWorkflowDefaultsButton, flfAdvancedDetails, flfChainedSettingsPanel]);
  flfGuideSettingsSection.style.display = "none";
  const i2vPass1SamplerSelect = makeSelect(I2V_SAMPLER_OPTIONS, "euler_ancestral");
  const i2vPass1SigmasInput = makeInput(DEFAULT_I2V_PASS1_SIGMAS);
  const i2vPass1StrengthSlider = makeInput("1", "range");
  i2vPass1StrengthSlider.min = "0";
  i2vPass1StrengthSlider.max = "1";
  i2vPass1StrengthSlider.step = "0.01";
  i2vPass1StrengthSlider.style.accentColor = "#a855f7";
  const i2vPass1StrengthInput = makeInput("1", "number");
  i2vPass1StrengthInput.min = "0";
  i2vPass1StrengthInput.max = "1";
  i2vPass1StrengthInput.step = "0.01";
  const i2vPass1Bypass = makeCheckbox("", false);
  const i2vPass2SamplerSelect = makeSelect(I2V_SAMPLER_OPTIONS, "euler_ancestral");
  const i2vPass2SigmasInput = makeInput(DEFAULT_I2V_PASS2_SIGMAS);
  const i2vPass2StrengthSlider = makeInput("1", "range");
  i2vPass2StrengthSlider.min = "0";
  i2vPass2StrengthSlider.max = "1";
  i2vPass2StrengthSlider.step = "0.01";
  i2vPass2StrengthSlider.style.accentColor = "#a855f7";
  const i2vPass2StrengthInput = makeInput("1", "number");
  i2vPass2StrengthInput.min = "0";
  i2vPass2StrengthInput.max = "1";
  i2vPass2StrengthInput.step = "0.01";
  const i2vPass2Bypass = makeCheckbox("", false);
  const i2vAdvancedNodeSettingsPanel = document.createElement("div");
  i2vAdvancedNodeSettingsPanel.style.cssText = "display:flex;flex-direction:column;gap:10px;";
  const i2vPass1NodePanel = makeI2VNodeOverridePassPanel("Pass 1", {
    sampler: i2vPass1SamplerSelect,
    sigmas: i2vPass1SigmasInput,
    strength: i2vPass1StrengthSlider,
    strengthNumber: i2vPass1StrengthInput,
    bypass: i2vPass1Bypass,
  });
  const i2vPass2NodePanel = makeI2VNodeOverridePassPanel("Pass 2", {
    sampler: i2vPass2SamplerSelect,
    sigmas: i2vPass2SigmasInput,
    strength: i2vPass2StrengthSlider,
    strengthNumber: i2vPass2StrengthInput,
    bypass: i2vPass2Bypass,
  });
  i2vAdvancedNodeSettingsPanel.append(i2vPass1NodePanel, i2vPass2NodePanel);
  const i2vAdvancedNodeSettingsSection = i2vAdvancedNodeSettingsPanel;
  const createSceneVideoButton = makeButton("Create Scene Video", "primary");
  const createSceneVideoButtons = [createSceneVideoButton];
  const miniMaxSceneVideoButton = makeButton("Create MiniMax H3 Scene Video", "primary");
  applyCompactButtonLabel(miniMaxSceneVideoButton, "Create MiniMax H3\nScene Video", { noMap: true, padding: "8px 8px", title: "Create MiniMax H3 Scene Video" });
  miniMaxSceneVideoButton.title = "Render this scene with the MiniMax H3 hidden workflow and preserve the exact timeline duration.";
  const miniMaxSceneVideoButtons = [miniMaxSceneVideoButton];
  const miniMaxReferencesButton = makeButton("Choose MiniMax References (0/9)");
  miniMaxReferencesButton.title = "Choose and order up to nine images from the existing Reference Builder for this scene.";
  const miniMaxVideoReferencesButton = makeButton("Choose MiniMax Edit References (0/9)");
  miniMaxVideoReferencesButton.title = "Choose character, location, background, prop, or style images to use alongside the Video to Video source.";
  const miniMaxReferenceButtons = [miniMaxReferencesButton, miniMaxVideoReferencesButton];
  const createSceneVideoActions = wrapCreateSceneVideoActions(createSceneVideoButton);
  const previewButton = makeButton("Create Z-Image", "primary");
  const zCreateButtons = [previewButton];
  const ernieCreateButton = makeButton("Create with Ernie", "primary");
  const ernieCreateButtons = [ernieCreateButton];
  const krea2TwoPassCreateButton = makeButton("Create with Krea 2", "primary");
  const krea2TwoPassCreateButtons = [krea2TwoPassCreateButton];
  const fluxCreateButtons = [previewFluxButton];
  const nbCreateButtons = [previewNBButton];

  return {
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
  };
}

export function createVideoSettingsPanel({
  activeI2VVideoSettings, activeScenePromptForEnhance, activeSegment, applyVideoSettingsToMultiSelection,
  clearSceneEndFrameButton, createI2VButton, createSceneEndFrameButton, editI2VInstructionsButton,
  editI2VPromptButton, editIdLoraInstructionsButton, editImagePromptButtons,
  editIngredientsInstructionsButton, editRTVInstructionsButton, editT2VInstructionsButton, endInput,
  ernieNotesInput, ernieRefImagePanel, ernieT2IPrompt, ernieUseVisionReference, finalizeSceneFLFPromptButton,
  firstLastFrameEndImageSource, firstLastFramePreviewGrid, firstLastFramePreviewPanel,
  firstLastFrameStartImageSource, firstLastFrameStatus, firstLastFrameVideoCard, flfChainPreviousEndFrame,
  flfChainedSettingsPanel, flfChainingEnabled, flfColorMatchFadeInput, flfColorMatchStrengthInput,
  flfCustomEndDirection, flfEndpointModeSelect, flfEndpointPromptPreview, flfFirstAttentionStrength,
  flfFirstGuideBlurInput, flfFirstGuideCrfInput, flfFirstGuideCrop, flfFirstGuideFrameIndexInput,
  flfFirstGuideInterpolation, flfFirstGuideStrengthInput, flfGemmaContextModeSelect,
  flfGlobalTransitionTypeSelect, flfGuideSettingsSection, flfLastAttentionStrength, flfLastGuideBlurInput,
  flfLastGuideCrfInput, flfLastGuideCrop, flfLastGuideFrameIndexInput, flfLastGuideInterpolation,
  flfLastGuideStrengthInput, flfMatchPreviousClipColor, flfMotionPlanPreview, flfPerScenePlanner,
  flfPreGeneratePromptsFromSceneImages, flfRenderChainSourceSelect, flfStructureModeSelect,
  flfTransitionTypeField, flfTransitionTypeSelect, flowGptPrompt, fluxPrompt, hasMultiSceneBatchSelection,
  i2vAudioVaePicker, i2vClip1Picker, i2vClip2Picker, i2vDiffusionLoaderAdvanced, i2vDiffusionModelField,
  i2vDiffusionModelPicker, i2vEnableFp16Accumulation, i2vFpsInput, i2vHeightInput, i2vLoraCount, i2vLoraSlots,
  i2vNotesInput, i2vPass1Bypass, i2vPass1SamplerSelect, i2vPass1SigmasInput, i2vPass1StrengthInput,
  i2vPass2Bypass, i2vPass2NodePanel, i2vPass2SamplerSelect, i2vPass2SigmasInput, i2vPass2StrengthInput,
  i2vPreFramesInput, i2vPrompt, i2vReferenceNote, i2vSeedInput, i2vSettingsGrid, i2vTailLossFramesInput,
  i2vUnetModelField, i2vUnetPicker, i2vUpscalePicker, i2vUseGgufModel, i2vUseLora, i2vUseSageAttention,
  i2vVaePicker, i2vWarmCooldownSection, i2vWidthInput, idLoraIdentityScaleInput, idLoraReferenceAudioInput,
  idLoraVideoCard, idLoraVoiceSettingsSection, imageToVideoCard, importCustomVideoCard,
  importCustomVideoPanel, ingredientsToVideoCard, krea2TwoPassNotesInput, krea2TwoPassRefImagePanel,
  krea2TwoPassT2IPrompt, krea2TwoPassUseVisionReference, labelInput, loadFirstLastFrameEndFile,
  ltx25AspectRatioSelect, ltx25MegapixelsInput, ltx25ResolutionGrid, ltxIdLoraFirstPassStrength,
  ltxIdLoraPicker, ltxIdLoraRequiredPanel, ltxIdLoraSecondPassStrength, ltxIngredientsFirstPassStrength,
  ltxIngredientsLoraPicker, ltxIngredientsRequiredPanel, ltxIngredientsResolutionWarning,
  ltxMsrBackgroundMode, ltxMsrFirstPassStrength, ltxMsrLoraPicker, ltxMsrReferenceStrength,
  ltxMsrRequiredPanel, ltxMsrSecondPassStrength, lyricSingersInput, lyricTextInput, nbNotes, nbPrompt,
  normalizeSegments, notesInput, planSceneEndMotionButton, promoteChainedFLFSceneImageToEndFrame,
  promptRunnerActionName, pushHistory, refImageInput, refImagePanel, referenceToVideoCard, render,
  rtvReferenceBehaviorField, rtvReferenceBehaviorForSegment, rtvReferenceBehaviorNote,
  rtvReferenceBehaviorSelect, rtvSceneImageAnchorSection, sceneEndFrameFileInput, segmentImageSource,
  segmentTrack, selectedSegmentImageThumbnailPath, startInput, state, syncI2VAdvancedNodeControls,
  syncInspector, syncTimelineTrimModeButton, t2iPrompt, t2vLocationNote, t2vRefImagePanel, t2vReferenceNote,
  textToVideoCard, updateI2VLoraVisibility, updateI2VPromptSaveButtonState, useI2VPromptEnhancementPass,
  useI2VVisionReference, useSceneI2VVideoSettings, useT2VVisionReference, useVisionReference,
  videoSettingsScopeNote, videoSettingsSegment, videoSubTabs, videoTriggerInput, wizardVideoSettings,
  zEnhanceAmount, zEnhanceAmountValue, zEnhanceClipPicker, zEnhanceGemmaNotes, zEnhanceHeight,
  zEnhanceLoraCount, zEnhanceLoraPanel, zEnhanceLoraRows, zEnhanceLoraSlots, zEnhancePromptPreview,
  zEnhanceSeed, zEnhanceSeedMode, zEnhanceUnetPicker, zEnhanceUseLora, zEnhanceVaePicker, zEnhanceWidth,
}) {
  function saveZEnhanceSettingsFromPanel() {
    const segment = activeSegment();
    if (segment) {
      segment.enhance_notes = zEnhanceGemmaNotes.value || "";
      segment.enhance_prompt = sceneImagePromptForEnhanceAll(segment).prompt || "";
    }
    const count = Math.max(0, Math.min(4, Number(zEnhanceLoraCount.value || 0)));
    state.zEnhanceSettings = {
      unet_name: zEnhanceUnetPicker.input.value || "",
      clip_name: zEnhanceClipPicker.input.value || "",
      vae_name: zEnhanceVaePicker.input.value || "",
      width: Number(zEnhanceWidth.value || 1920),
      height: Number(zEnhanceHeight.value || 1080),
      seed: Number(zEnhanceSeed.value || 1),
      seed_mode: zEnhanceSeedMode.value || "fixed",
      enhance_amount: Math.max(1, Math.min(20, Number(zEnhanceAmount.value || 8))),
      use_loras: Boolean(zEnhanceUseLora.input.checked),
      lora_count: count,
      loras: zEnhanceLoraSlots.map((slot) => ({ name: slot.picker.input.value || "[none]", strength: Number(slot.strength.value || 1) })),
    };
    updateZEnhanceLoraVisibility();
    zEnhanceAmountValue.textContent = `Enhance amount: ${state.zEnhanceSettings.enhance_amount}`;
    return state.zEnhanceSettings;
  }

  function syncI2VVideoModelPickerVisibility() {
    const isLtx25 = (activeI2VVideoSettings()?.ltx_version || "2.5") !== "2.3";
    if (isLtx25) {
      i2vUnetModelField.style.display = "none";
      i2vDiffusionModelField.style.display = "flex";
      i2vDiffusionLoaderAdvanced.style.display = "";
      return;
    }
    const useGguf = Boolean(i2vUseGgufModel.input.checked);
    i2vUnetModelField.style.display = useGguf ? "flex" : "none";
    i2vDiffusionModelField.style.display = useGguf ? "none" : "flex";
    i2vDiffusionLoaderAdvanced.style.display = useGguf ? "none" : "";
  }

  function syncVideoInstructionEditorButtons() {
    const mode = currentVideoMode();
    editI2VInstructionsButton.style.display = mode === "i2v" ? "" : "none";
    editIdLoraInstructionsButton.style.display = mode === "id_lora" ? "" : "none";
    editRTVInstructionsButton.style.display = mode === "rtv" ? "" : "none";
    editIngredientsInstructionsButton.style.display = mode === "ingredients" ? "" : "none";
    editT2VInstructionsButton.style.display = mode === "t2v" ? "" : "none";
  }

  function rtvReferenceBehaviorGlobalValue() {
    if (state.segments.some((item) => rtvReferenceBehaviorForSegment(item) === "first_last_frame")) return "first_last_frame";
    if (state.segments.some((item) => rtvReferenceBehaviorForSegment(item) === "character_anchor")) return "character_anchor";
    return "none";
  }

  function applyRTVReferenceBehaviorToAll(behavior) {
    const value = normalizeRTVReferenceBehavior(behavior);
    state.segments.forEach((item) => {
      item.rtv_reference_behavior = value;
      item.use_scene_image_as_rtv_ref = value === "character_anchor";
    });
  }

  function updateActiveFromInputs(options = {}) {
    const segment = activeSegment();
    if (!segment) return;
    const inspectorSegmentId = String(startInput.dataset.vrgdgInspectorSegmentId || "");
    if (inspectorSegmentId !== String(segment.id || "")) {
      console.warn(
        "[VRGDG Music Builder] Ignored stale inspector values for a different scene.",
        { inspectorSegmentId, activeSegmentId: String(segment.id || "") },
      );
      syncInspector();
      return;
    }
    if (!options.skipHistory) pushHistory();
    segment.label = labelInput.value || "Scene";
    const isOverlay = segmentTrack(segment) === "overlay";
    if ((!state.timingFrozen || isOverlay) && (isOverlay ? segment.overlay_locked === false : !hasLockedVideo(segment))) {
      segment.start = Math.max(0, Number(startInput.value || 0));
      segment.end = Math.max(segment.start + 0.1, Number(endInput.value || segment.start + 4));
    }
    segment.notes = notesInput.value || "";
    if (state.imageModelMode === "ernie_image") {
      segment.notes = ernieNotesInput.value || "";
    } else if (state.imageModelMode === "krea2_2pass") {
      segment.notes = krea2TwoPassNotesInput.value || "";
    } else if (state.imageModelMode === "nano_banana") {
      segment.nb_notes = nbNotes.value || "";
    }
    segment.i2v_notes = i2vNotesInput.value || "";
    segment.flf_transition_type = ["auto", "smooth", "morph"].includes(flfTransitionTypeSelect.value) ? flfTransitionTypeSelect.value : "global";
    // The inspector can temporarily contain an old/blank value while scenes are
    // replaced by transcription, SRT import, project load, or another batch
    // operation. Only let this field overwrite the scene after a real user edit.
    // Timeline lyric boxes update their scene objects directly.
    const lyricInspectorSegmentId = String(lyricTextInput.dataset.vrgdgInspectorSegmentId || "");
    if (
      lyricTextInput.dataset.vrgdgUserEdited === "1"
      && lyricInspectorSegmentId === String(segment.id || "")
    ) {
      segment.lyric_text = lyricTextInput.value || "";
      segment.lyric_no_lip_sync = isInstrumentalLyricText(segment.lyric_text);
    }
    const singerInspectorSegmentId = String(lyricSingersInput.dataset.vrgdgInspectorSegmentId || "");
    if (
      lyricSingersInput.dataset.vrgdgUserEdited === "1"
      && singerInspectorSegmentId === String(segment.id || "")
    ) {
      segment.lyric_singers = lyricSingersInput.value.split(",").map((item) => item.trim()).filter(Boolean);
      if (segment.lyric_singers.length < 2 && !segment.lyric_shot_word_timing_enabled) {
        segment.lyric_performance_mode = "together";
        segment.lyric_cue_map = [];
      }
    }
    let editedT2IPrompt = t2iPrompt.value || "";
    if (state.imageModelMode === "ernie_image") editedT2IPrompt = ernieT2IPrompt.value || "";
    else if (state.imageModelMode === "krea2_2pass") editedT2IPrompt = krea2TwoPassT2IPrompt.value || "";
    else if (state.imageModelMode === "flux_klein") editedT2IPrompt = fluxPrompt.value || "";
    else if (state.imageModelMode === "nano_banana") editedT2IPrompt = nbPrompt.value || "";
    else if (state.imageModelMode === "flow_gpt") editedT2IPrompt = flowGptPrompt.value || "";
    else if (state.imageModelMode === "z_enhance") editedT2IPrompt = zEnhancePromptPreview.value || "";
    segment.t2i_prompt = editedT2IPrompt;
    segment.flux_prompt = editedT2IPrompt;
    segment.nb_prompt = editedT2IPrompt;
    segment.flow_gpt_prompt = editedT2IPrompt;
    if (t2iPrompt.value !== editedT2IPrompt) t2iPrompt.value = editedT2IPrompt;
    if (ernieT2IPrompt.value !== editedT2IPrompt) ernieT2IPrompt.value = editedT2IPrompt;
    if (krea2TwoPassT2IPrompt.value !== editedT2IPrompt) krea2TwoPassT2IPrompt.value = editedT2IPrompt;
    if (fluxPrompt.value !== editedT2IPrompt) fluxPrompt.value = editedT2IPrompt;
    if (nbPrompt.value !== editedT2IPrompt) nbPrompt.value = editedT2IPrompt;
    if (flowGptPrompt.value !== editedT2IPrompt) flowGptPrompt.value = editedT2IPrompt;
    if (zEnhancePromptPreview.value !== editedT2IPrompt) zEnhancePromptPreview.value = editedT2IPrompt;
    const previousI2VPrompt = String(segment.i2v_prompt || "");
    const nextI2VPrompt = i2vPrompt.value || "";
    segment.i2v_prompt = nextI2VPrompt;
    if (nextI2VPrompt !== previousI2VPrompt) segment.i2v_prompt_origin = "manual";
    editI2VPromptButton.style.display = String(segment.i2v_prompt || "").trim() ? "" : "none";
    updateI2VPromptSaveButtonState();
    editImagePromptButtons.forEach((button) => {
      button.style.display = String(editedT2IPrompt || "").trim() ? "" : "none";
    });
    segment.enhance_notes = zEnhanceGemmaNotes.value || "";
    segment.enhance_prompt = sceneImagePromptForEnhanceAll(segment).prompt || "";
    segment.use_vision_reference = Boolean(useVisionReference.input.checked);
    if (state.imageModelMode === "ernie_image") {
      segment.use_vision_reference = Boolean(ernieUseVisionReference.input.checked);
    } else if (state.imageModelMode === "krea2_2pass") {
      segment.use_vision_reference = Boolean(krea2TwoPassUseVisionReference.input.checked);
    }
    segment.use_i2v_vision_reference = Boolean(useI2VVisionReference.input.checked);
    segment.use_t2v_vision_reference = Boolean(useT2VVisionReference.input.checked);
    applyRTVReferenceBehaviorToAll(rtvReferenceBehaviorSelect.value);
    segment.ref_image_path = refImageInput.value || "";
    refImagePanel.style.display = segment.use_vision_reference ? "flex" : "none";
    ernieRefImagePanel.style.display = segment.use_vision_reference ? "flex" : "none";
    krea2TwoPassRefImagePanel.style.display = segment.use_vision_reference ? "flex" : "none";
    t2vRefImagePanel.style.display = currentVideoMode() === "t2v" && segment.use_t2v_vision_reference ? "flex" : "none";
    syncRTVSceneImageAnchorPanel();
    if (!state.timingFrozen && !hasLockedVideo(segment) && !isOverlay) normalizeSegments(segment);
    if (isOverlay) sortSegments(state.overlaySegments);
    render();
  }

  function syncZEnhanceSettingsPanel() {
    const settings = state.zEnhanceSettings || {};
    const promptInfo = activeScenePromptForEnhance();
    const segment = activeSegment();
    zEnhanceGemmaNotes.value = segment?.enhance_notes || "";
    zEnhancePromptPreview.value = promptInfo.prompt || "";
    zEnhanceUnetPicker.input.value = settings.unet_name || "z_image_turbo_bf16.safetensors";
    zEnhanceClipPicker.input.value = settings.clip_name || "qwen_3_4b.safetensors";
    zEnhanceVaePicker.input.value = settings.vae_name || "ae.safetensors";
    zEnhanceWidth.value = settings.width || 1920;
    zEnhanceHeight.value = settings.height || 1080;
    zEnhanceSeed.value = settings.seed || 1;
    zEnhanceSeedMode.value = settings.seed_mode || "randomize";
    zEnhanceAmount.value = Math.max(1, Math.min(20, Number(settings.enhance_amount || 8)));
    zEnhanceAmountValue.textContent = `Enhance amount: ${zEnhanceAmount.value}`;
    zEnhanceUseLora.input.checked = Boolean(settings.use_loras);
    zEnhanceLoraCount.value = Number(settings.lora_count || 0);
    zEnhanceLoraSlots.forEach((slot, index) => {
      const config = settings.loras?.[index] || {};
      slot.picker.input.value = config.name || "[none]";
      slot.strength.value = config.strength ?? 1;
    });
    updateZEnhanceLoraVisibility();
  }

  function syncI2VVideoSettingsPanel() {
    const segment = videoSettingsSegment();
    useSceneI2VVideoSettings.input.checked = Boolean(segment?.use_scene_i2v_video_settings);
    videoSettingsScopeNote.textContent = wizardVideoSettings.global ? "Editing project-wide LTX models, LoRAs and video settings." : segment?.use_scene_i2v_video_settings
      ? "This scene is using custom video models, settings, and LoRAs from the Models tab."
      : "This scene is using global video models, settings, and LoRAs. Enable custom scene video settings in the Models tab.";
    const settings = repairI2VVideoSettingDimensions(activeI2VVideoSettings() || {});
    videoTriggerInput.value = settings.video_trigger_phrase || "";
    i2vUseGgufModel.input.checked = settings.use_gguf_model !== false;
    i2vUnetPicker.input.value = BAD_I2V_UNET_ALIASES.has(settings.unet_name) ? DEFAULT_I2V_UNET : settings.unet_name || "";
    i2vDiffusionModelPicker.input.value = settings.diffusion_model_name || DEFAULT_I2V_DIFFUSION_MODEL;
    i2vUseSageAttention.input.checked = Boolean(settings.use_sage_attention);
    i2vEnableFp16Accumulation.input.checked = Boolean(settings.enable_fp16_accumulation);
    i2vVaePicker.input.value = settings.vae_name || "";
    i2vClip1Picker.input.value = settings.clip_name1 || "";
    i2vClip2Picker.input.value = settings.clip_name2 || "";
    i2vUpscalePicker.input.value = settings.upscale_model_name || "";
    i2vAudioVaePicker.input.value = settings.audio_vae_name || "";
    const isIngredientsMode = currentVideoMode() === "ingredients";
    const regularWidth = Number(settings.width || 1920);
    const regularHeight = Number(settings.height || 1080);
    const repairedRegularWidth = (!isIngredientsMode && regularWidth === DEFAULT_LTX_INGREDIENTS_WIDTH && regularHeight === DEFAULT_LTX_INGREDIENTS_HEIGHT) ? 1920 : regularWidth;
    const repairedRegularHeight = (!isIngredientsMode && regularWidth === DEFAULT_LTX_INGREDIENTS_WIDTH && regularHeight === DEFAULT_LTX_INGREDIENTS_HEIGHT) ? 1080 : regularHeight;
    const rawIngredientsWidth = Number(settings.ingredients_width || DEFAULT_LTX_INGREDIENTS_WIDTH);
    const rawIngredientsHeight = Number(settings.ingredients_height || DEFAULT_LTX_INGREDIENTS_HEIGHT);
    const ingredientsWidth = rawIngredientsWidth === 1920 && rawIngredientsHeight === 1080 ? DEFAULT_LTX_INGREDIENTS_WIDTH : rawIngredientsWidth;
    const ingredientsHeight = rawIngredientsWidth === 1920 && rawIngredientsHeight === 1080 ? DEFAULT_LTX_INGREDIENTS_HEIGHT : rawIngredientsHeight;
    i2vFpsInput.value = settings.fps || 24;
    i2vWidthInput.value = isIngredientsMode ? ingredientsWidth : repairedRegularWidth;
    i2vHeightInput.value = isIngredientsMode ? ingredientsHeight : repairedRegularHeight;
    ltx25AspectRatioSelect.value = settings.resolution_aspect_ratio || "16:9 (Widescreen)";
    ltx25MegapixelsInput.value = Number(settings.resolution_megapixels || 1.2);
    const isLtx25 = settings.ltx_version !== "2.3";
    const isLtx25T2V = isLtx25 && currentVideoMode() === "t2v";
    i2vSettingsGrid.style.display = "grid";
    ltx25ResolutionGrid.style.display = "none";
    i2vUseGgufModel.wrapper.style.display = isLtx25 ? "none" : "flex";
    i2vClip2Picker.wrapper.parentElement.style.display = isLtx25 ? "none" : "flex";
    i2vSeedInput.value = settings.seed || 69;
    i2vTailLossFramesInput.value = Math.max(0, Number(settings.tail_loss_frames ?? 25));
    const isFLFMode = currentVideoMode() === "flf";
    i2vPreFramesInput.value = Math.max(0, Number(isFLFMode ? (settings.flf_pre_frames ?? 0) : (settings.pre_frames ?? 50)));
    flfFirstGuideStrengthInput.value = Number(settings.flf_first_guide_strength ?? 0.7);
    flfLastGuideStrengthInput.value = Number(settings.flf_last_guide_strength ?? 0.7);
    flfFirstGuideFrameIndexInput.value = Number(settings.flf_first_guide_frame_idx ?? 0);
    flfLastGuideFrameIndexInput.value = Number(settings.flf_last_guide_frame_idx ?? -1);
    flfFirstGuideCrfInput.value = Number(settings.flf_first_guide_crf ?? 29);
    flfLastGuideCrfInput.value = Number(settings.flf_last_guide_crf ?? 29);
    flfFirstGuideBlurInput.value = Number(settings.flf_first_guide_blur_radius ?? 1);
    flfLastGuideBlurInput.value = Number(settings.flf_last_guide_blur_radius ?? 1);
    flfFirstGuideInterpolation.value = settings.flf_first_guide_interpolation || "lanczos";
    flfLastGuideInterpolation.value = settings.flf_last_guide_interpolation || "lanczos";
    flfFirstGuideCrop.value = settings.flf_first_guide_crop || "center";
    flfLastGuideCrop.value = settings.flf_last_guide_crop || "center";
    flfFirstAttentionStrength.value = Number(settings.flf_first_attention_strength ?? 0.9);
    flfLastAttentionStrength.value = Number(settings.flf_last_attention_strength ?? 1);
    const flfUsesChain = state.i2vVideoSettings?.flf_chain_previous_end_frame !== false;
    flfChainPreviousEndFrame.input.checked = flfUsesChain;
    flfStructureModeSelect.value = flfUsesChain ? "chained" : "independent";
    flfChainedSettingsPanel.style.display = flfUsesChain ? "flex" : "none";
    flfRenderChainSourceSelect.value = state.i2vVideoSettings?.flf_render_chain_start_source === "previous_image" ? "previous_image" : "rendered_frame";
    flfPreGeneratePromptsFromSceneImages.input.checked = state.i2vVideoSettings?.flf_pregenerate_prompts_from_scene_images === true;
    flfMatchPreviousClipColor.input.checked = state.i2vVideoSettings?.flf_match_previous_clip_color === true;
    flfColorMatchStrengthInput.value = Number(state.i2vVideoSettings?.flf_color_match_strength ?? 0.85);
    flfColorMatchFadeInput.value = Number(state.i2vVideoSettings?.flf_color_match_fade_seconds ?? 1.0);
    flfGlobalTransitionTypeSelect.value = ["smooth", "morph"].includes(settings.flf_global_transition_type) ? settings.flf_global_transition_type : "auto";
    flfGemmaContextModeSelect.value = ["images_only", "images_story", "full"].includes(settings.flf_gemma_context_mode) ? settings.flf_gemma_context_mode : "images_story";
    ltxMsrLoraPicker.input.value = settings.msr_lora_name || REQUIRED_LTX_MSR_LORA;
    ltxMsrFirstPassStrength.value = Number(settings.msr_first_pass_strength ?? 1);
    ltxMsrSecondPassStrength.value = 0;
    ltxMsrReferenceStrength.value = settings.msr_reference_strength || "auto - based on subject count";
    const isLtx25Rtv = settings.ltx_version !== "2.3";
    const backgroundChoices = isLtx25Rtv
      ? ["no background reference", "use location/background reference"]
      : ["neutral placeholder (WIP/testing)", "use location/background reference"];
    const selectedBackground = String(settings.msr_background_mode || "").includes("location")
      ? "use location/background reference"
      : backgroundChoices[0];
    ltxMsrBackgroundMode.replaceChildren(...backgroundChoices.map((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      return option;
    }));
    ltxMsrBackgroundMode.value = selectedBackground;
    ltxIngredientsLoraPicker.input.value = settings.ingredients_lora_name || REQUIRED_LTX_INGREDIENTS_LORA;
    ltxIngredientsFirstPassStrength.value = Number(settings.ingredients_first_pass_strength ?? 1);
    ltxIdLoraPicker.input.value = settings.id_lora_name || REQUIRED_LTX_ID_LORA;
    ltxIdLoraFirstPassStrength.value = Number(settings.id_lora_first_pass_strength ?? 1);
    ltxIdLoraSecondPassStrength.value = Number(settings.id_lora_second_pass_strength ?? 1);
    idLoraReferenceAudioInput.value = settings.id_lora_reference_audio_path || "";
    idLoraIdentityScaleInput.value = Number(settings.identity_guidance_scale ?? 3);
    ltxIngredientsResolutionWarning.style.display = isIngredientsMode ? "block" : "none";
    i2vUseLora.input.checked = Boolean(settings.use_loras);
    i2vLoraCount.value = Number(settings.lora_count || 0);
    i2vLoraSlots.forEach((slot, index) => {
      const config = settings.loras?.[index] || {};
      slot.picker.input.value = config.name || "[none]";
      const legacyStrength = config.strength ?? 1;
      slot.firstPassStrength.value = config.first_pass_strength ?? legacyStrength;
      slot.secondPassStrength.value = config.second_pass_strength ?? legacyStrength;
    });
    syncI2VAdvancedNodeControls(settings);
    updateI2VLoraVisibility();
    syncI2VVideoModelPickerVisibility();
  }

  function saveI2VVideoSettingsFromPanel() {
    const count = Math.max(0, Math.min(4, Number(i2vLoraCount.value || 0)));
    const segment = videoSettingsSegment();
    const previous = activeI2VVideoSettings() || {};
    const mode = currentVideoMode();
    const isI2VMode = mode === "i2v";
    const isT2VMode = mode === "t2v";
    const isRTVMode = mode === "rtv";
    const isIngredientsMode = mode === "ingredients";
    const isIdLoraMode = mode === "id_lora";
    const isFLFMode = mode === "flf";
    const previousRegularWidth = Number(previous.width || 1920);
    const previousRegularHeight = Number(previous.height || 1080);
    const repairedPreviousWidth = previousRegularWidth === DEFAULT_LTX_INGREDIENTS_WIDTH && previousRegularHeight === DEFAULT_LTX_INGREDIENTS_HEIGHT ? 1920 : previousRegularWidth;
    const repairedPreviousHeight = previousRegularWidth === DEFAULT_LTX_INGREDIENTS_WIDTH && previousRegularHeight === DEFAULT_LTX_INGREDIENTS_HEIGHT ? 1080 : previousRegularHeight;
    const regularWidth = isIngredientsMode ? repairedPreviousWidth : Number(i2vWidthInput.value || 1920);
    const regularHeight = isIngredientsMode ? repairedPreviousHeight : Number(i2vHeightInput.value || 1080);
    const previousIngredientsWidth = Number(previous.ingredients_width || DEFAULT_LTX_INGREDIENTS_WIDTH);
    const previousIngredientsHeight = Number(previous.ingredients_height || DEFAULT_LTX_INGREDIENTS_HEIGHT);
    const repairedPreviousIngredientsWidth = previousIngredientsWidth === 1920 && previousIngredientsHeight === 1080 ? DEFAULT_LTX_INGREDIENTS_WIDTH : previousIngredientsWidth;
    const repairedPreviousIngredientsHeight = previousIngredientsWidth === 1920 && previousIngredientsHeight === 1080 ? DEFAULT_LTX_INGREDIENTS_HEIGHT : previousIngredientsHeight;
    const ingredientsWidth = isIngredientsMode ? Number(i2vWidthInput.value || DEFAULT_LTX_INGREDIENTS_WIDTH) : repairedPreviousIngredientsWidth;
    const ingredientsHeight = isIngredientsMode ? Number(i2vHeightInput.value || DEFAULT_LTX_INGREDIENTS_HEIGHT) : repairedPreviousIngredientsHeight;
    const defaultSamplerForMode = isIngredientsMode ? DEFAULT_INGREDIENTS_SAMPLER : "euler_ancestral";
    const pass1SamplerName = i2vPass1SamplerSelect.value || defaultSamplerForMode;
    const pass1Sigmas = normalizeI2VSigmasText(i2vPass1SigmasInput.value, DEFAULT_I2V_PASS1_SIGMAS);
    const pass2SamplerName = i2vPass2SamplerSelect.value || defaultSamplerForMode;
    const pass2Sigmas = normalizeI2VSigmasText(i2vPass2SigmasInput.value, DEFAULT_I2V_PASS2_SIGMAS);
    const settings = {
      ltx_version: previous.ltx_version === "2.3" ? "2.3" : "2.5",
      use_gguf_model: Boolean(i2vUseGgufModel.input.checked),
      unet_name: BAD_I2V_UNET_ALIASES.has(i2vUnetPicker.input.value) ? DEFAULT_I2V_UNET : i2vUnetPicker.input.value || "",
      diffusion_model_name: i2vDiffusionModelPicker.input.value || DEFAULT_I2V_DIFFUSION_MODEL,
      use_sage_attention: Boolean(i2vUseSageAttention.input.checked),
      enable_fp16_accumulation: Boolean(i2vEnableFp16Accumulation.input.checked),
      vae_name: i2vVaePicker.input.value || "",
      clip_name1: i2vClip1Picker.input.value || "",
      clip_name2: i2vClip2Picker.input.value || "",
      upscale_model_name: i2vUpscalePicker.input.value || "",
      audio_vae_name: i2vAudioVaePicker.input.value || "",
      fps: Number(i2vFpsInput.value || 24),
      width: regularWidth,
      height: regularHeight,
      resolution_aspect_ratio: ltx25AspectRatioSelect.value || "16:9 (Widescreen)",
      resolution_megapixels: Math.max(0.1, Number(ltx25MegapixelsInput.value || 1.2)),
      seed: Number(i2vSeedInput.value || 69),
      tail_loss_frames: isIdLoraMode ? 0 : Math.max(0, Number(i2vTailLossFramesInput.value || 0)),
      pre_frames: isIdLoraMode ? 0 : isFLFMode ? Math.max(0, Number(previous.pre_frames ?? 50)) : Math.max(0, Number(i2vPreFramesInput.value || 0)),
      flf_pre_frames: isFLFMode ? Math.max(0, Number(i2vPreFramesInput.value || 0)) : Math.max(0, Number(previous.flf_pre_frames ?? 0)),
      flf_first_guide_strength: isFLFMode ? Math.max(0, Math.min(1, Number(flfFirstGuideStrengthInput.value || 0))) : Number(previous.flf_first_guide_strength ?? 0.7),
      flf_last_guide_strength: isFLFMode ? Math.max(0, Math.min(1, Number(flfLastGuideStrengthInput.value || 0))) : Number(previous.flf_last_guide_strength ?? 0.7),
      flf_first_guide_frame_idx: isFLFMode ? Math.trunc(Number(flfFirstGuideFrameIndexInput.value || 0)) : Math.trunc(Number(previous.flf_first_guide_frame_idx ?? 0)),
      flf_last_guide_frame_idx: isFLFMode ? Math.trunc(Number(flfLastGuideFrameIndexInput.value || 0)) : Math.trunc(Number(previous.flf_last_guide_frame_idx ?? -1)),
      flf_first_guide_crf: isFLFMode ? Math.max(0, Math.min(51, Math.trunc(Number(flfFirstGuideCrfInput.value || 0)))) : Number(previous.flf_first_guide_crf ?? 29),
      flf_last_guide_crf: isFLFMode ? Math.max(0, Math.min(51, Math.trunc(Number(flfLastGuideCrfInput.value || 0)))) : Number(previous.flf_last_guide_crf ?? 29),
      flf_first_guide_blur_radius: isFLFMode ? Math.max(0, Math.min(7, Math.trunc(Number(flfFirstGuideBlurInput.value || 0)))) : Number(previous.flf_first_guide_blur_radius ?? 1),
      flf_last_guide_blur_radius: isFLFMode ? Math.max(0, Math.min(7, Math.trunc(Number(flfLastGuideBlurInput.value || 0)))) : Number(previous.flf_last_guide_blur_radius ?? 1),
      flf_first_guide_interpolation: isFLFMode ? flfFirstGuideInterpolation.value : (previous.flf_first_guide_interpolation || "lanczos"),
      flf_last_guide_interpolation: isFLFMode ? flfLastGuideInterpolation.value : (previous.flf_last_guide_interpolation || "lanczos"),
      flf_first_guide_crop: isFLFMode ? flfFirstGuideCrop.value : (previous.flf_first_guide_crop || "center"),
      flf_last_guide_crop: isFLFMode ? flfLastGuideCrop.value : (previous.flf_last_guide_crop || "center"),
      flf_first_attention_strength: isFLFMode ? Math.max(0, Math.min(1, Number(flfFirstAttentionStrength.value || 0))) : Number(previous.flf_first_attention_strength ?? 0.9),
      flf_last_attention_strength: isFLFMode ? Math.max(0, Math.min(1, Number(flfLastAttentionStrength.value || 0))) : Number(previous.flf_last_attention_strength ?? 1),
      flf_chain_previous_end_frame: isFLFMode ? Boolean(flfChainPreviousEndFrame.input.checked) : previous.flf_chain_previous_end_frame !== false,
      flf_render_chain_start_source: isFLFMode ? (flfRenderChainSourceSelect.value === "previous_image" ? "previous_image" : "rendered_frame") : (previous.flf_render_chain_start_source || "rendered_frame"),
      flf_pregenerate_prompts_from_scene_images: isFLFMode ? Boolean(flfPreGeneratePromptsFromSceneImages.input.checked) : previous.flf_pregenerate_prompts_from_scene_images === true,
      flf_match_previous_clip_color: isFLFMode ? Boolean(flfMatchPreviousClipColor.input.checked) : previous.flf_match_previous_clip_color === true,
      flf_color_match_strength: isFLFMode ? Math.max(0, Math.min(1, Number(flfColorMatchStrengthInput.value || 0))) : Number(previous.flf_color_match_strength ?? 0.85),
      flf_color_match_fade_seconds: isFLFMode ? Math.max(0.05, Math.min(30, Number(flfColorMatchFadeInput.value || 1))) : Number(previous.flf_color_match_fade_seconds ?? 1.0),
      flf_global_transition_type: isFLFMode && ["smooth", "morph"].includes(flfGlobalTransitionTypeSelect.value) ? flfGlobalTransitionTypeSelect.value : isFLFMode ? "auto" : (previous.flf_global_transition_type || "auto"),
      flf_gemma_context_mode: isFLFMode && ["images_only", "images_story", "full"].includes(flfGemmaContextModeSelect.value) ? flfGemmaContextModeSelect.value : (previous.flf_gemma_context_mode || "images_story"),
      video_trigger_phrase: videoTriggerInput.value || "",
      msr_lora_name: ltxMsrLoraPicker.input.value || REQUIRED_LTX_MSR_LORA,
      msr_first_pass_strength: Number(ltxMsrFirstPassStrength.value || 1),
      msr_second_pass_strength: 0,
      msr_reference_strength: ltxMsrReferenceStrength.value || "auto - based on subject count",
      msr_background_mode: ltxMsrBackgroundMode.value || (previous.ltx_version === "2.3" ? "neutral placeholder (WIP/testing)" : "no background reference"),
      ingredients_lora_name: ltxIngredientsLoraPicker.input.value || REQUIRED_LTX_INGREDIENTS_LORA,
      ingredients_first_pass_strength: Number(ltxIngredientsFirstPassStrength.value || 1),
      ingredients_second_pass_strength: 0,
      ingredients_width: ingredientsWidth,
      ingredients_height: ingredientsHeight,
      id_lora_name: ltxIdLoraPicker.input.value || REQUIRED_LTX_ID_LORA,
      id_lora_first_pass_strength: Number(ltxIdLoraFirstPassStrength.value || 1),
      id_lora_second_pass_strength: Number(ltxIdLoraSecondPassStrength.value || 1),
      id_lora_reference_audio_path: idLoraReferenceAudioInput.value || "",
      identity_guidance_scale: Number(idLoraIdentityScaleInput.value || 3),
      identity_start_percent: 0,
      identity_end_percent: 1,
      id_lora_duration: Number(previous.id_lora_duration || 5),
      pass1_sampler_name: (isI2VMode || isIdLoraMode) ? pass1SamplerName : (previous.pass1_sampler_name || "euler_ancestral"),
      pass1_sigmas: (isI2VMode || isIdLoraMode) ? pass1Sigmas : (previous.pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS),
      pass1_inplace_strength: (isI2VMode || isIdLoraMode) ? Math.max(0, Math.min(1, Number(i2vPass1StrengthInput.value || 1))) : Number(previous.pass1_inplace_strength ?? 1),
      pass1_inplace_bypass: (isI2VMode || isIdLoraMode) ? Boolean(i2vPass1Bypass.input.checked) : Boolean(previous.pass1_inplace_bypass),
      pass2_sampler_name: (isI2VMode || isIdLoraMode) ? pass2SamplerName : (previous.pass2_sampler_name || "euler_ancestral"),
      pass2_sigmas: (isI2VMode || isIdLoraMode) ? pass2Sigmas : (previous.pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS),
      pass2_inplace_strength: (isI2VMode || isIdLoraMode) ? Math.max(0, Math.min(1, Number(i2vPass2StrengthInput.value || 1))) : Number(previous.pass2_inplace_strength ?? 1),
      pass2_inplace_bypass: (isI2VMode || isIdLoraMode) ? Boolean(i2vPass2Bypass.input.checked) : Boolean(previous.pass2_inplace_bypass),
      t2v_pass1_sampler_name: isT2VMode ? pass1SamplerName : (previous.t2v_pass1_sampler_name || "euler_ancestral"),
      t2v_pass1_sigmas: isT2VMode ? pass1Sigmas : (previous.t2v_pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS),
      t2v_pass2_sampler_name: isT2VMode ? pass2SamplerName : (previous.t2v_pass2_sampler_name || "euler_ancestral"),
      t2v_pass2_sigmas: isT2VMode ? pass2Sigmas : (previous.t2v_pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS),
      rtv_pass1_sampler_name: isRTVMode ? pass1SamplerName : (previous.rtv_pass1_sampler_name || "euler_ancestral"),
      rtv_pass1_sigmas: isRTVMode ? pass1Sigmas : (previous.rtv_pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS),
      rtv_pass2_sampler_name: isRTVMode ? pass2SamplerName : (previous.rtv_pass2_sampler_name || "euler_ancestral"),
      rtv_pass2_sigmas: isRTVMode ? pass2Sigmas : (previous.rtv_pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS),
      ingredients_pass1_sampler_name: isIngredientsMode ? (pass1SamplerName || DEFAULT_INGREDIENTS_SAMPLER) : (previous.ingredients_pass1_sampler_name || DEFAULT_INGREDIENTS_SAMPLER),
      ingredients_pass1_sigmas: isIngredientsMode ? pass1Sigmas : (previous.ingredients_pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS),
      ingredients_pass2_sampler_name: isIngredientsMode ? (pass2SamplerName || DEFAULT_INGREDIENTS_SAMPLER) : (previous.ingredients_pass2_sampler_name || DEFAULT_INGREDIENTS_SAMPLER),
      ingredients_pass2_sigmas: isIngredientsMode ? pass2Sigmas : (previous.ingredients_pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS),
      use_loras: Boolean(i2vUseLora.input.checked),
      lora_count: count,
      loras: i2vLoraSlots.map((slot) => ({
        name: slot.picker.input.value || "[none]",
        first_pass_strength: Number(slot.firstPassStrength.value || 1),
        second_pass_strength: Number(slot.secondPassStrength.value || 1),
      })),
    };
    repairI2VVideoSettingDimensions(settings);
    if (segment?.use_scene_i2v_video_settings || (!wizardVideoSettings.global && hasMultiSceneBatchSelection())) {
      if (segment) {
        segment.use_scene_i2v_video_settings = true;
        segment.i2v_video_settings = settings;
      }
    }
    else {
      state.videoTriggerPhrase = settings.video_trigger_phrase || "";
      state.i2vVideoSettings = settings;
    }
    if (!wizardVideoSettings.global) applyVideoSettingsToMultiSelection(settings);
    updateI2VLoraVisibility();
    return settings;
  }

  function currentVideoMode() {
    if (state.videoModelMode === "import") return "import";
    if (state.videoModelMode === "id_lora") return "id_lora";
    if (state.videoModelMode === "t2v") return "t2v";
    if (state.videoModelMode === "rtv") return "rtv";
    if (state.videoModelMode === "ingredients") return "ingredients";
    if (state.videoModelMode === "flf") return "flf";
    return "i2v";
  }

  function syncRTVSceneImageAnchorPanel() {
    const segment = activeSegment();
    const isRTV = currentVideoMode() === "rtv";
    const isFLF = currentVideoMode() === "flf";
    if (isFLF && segment) promoteChainedFLFSceneImageToEndFrame(segment);
    const image = segment ? (isFLF ? firstLastFrameStartImageSource(segment) : segmentImageSource(segment)) : null;
    const hasImage = Boolean(image?.path || image?.data);
    const behavior = rtvReferenceBehaviorGlobalValue();
    const lastImage = segment ? firstLastFrameEndImageSource(segment) : null;
    const hasLastImage = Boolean(lastImage?.path || lastImage?.data);
    const independentFLF = isFLF && !flfChainingEnabled(segment);
    flfPerScenePlanner.style.display = independentFLF ? "flex" : "none";
    flfEndpointModeSelect.value = segment?.flf_endpoint_mode === "custom" ? "custom" : "auto";
    flfCustomEndDirection.value = String(segment?.flf_custom_end_direction || "");
    flfCustomEndDirection.parentElement.style.display = independentFLF && flfEndpointModeSelect.value === "custom" ? "flex" : "none";
    flfMotionPlanPreview.textContent = String(segment?.flf_motion_plan || "").trim() || "No provisional motion plan has been created yet.";
    flfEndpointPromptPreview.textContent = String(segment?.flf_end_frame_prompt || "").trim() || "No saved end-frame image prompt yet.";
    rtvSceneImageAnchorSection.style.display = isRTV || isFLF ? "" : "none";
    if (rtvSceneImageAnchorSection.firstElementChild) rtvSceneImageAnchorSection.firstElementChild.textContent = isFLF ? "First / End Frames" : "Reference Behavior";
    rtvReferenceBehaviorField.style.display = isFLF ? "none" : "flex";
    rtvReferenceBehaviorNote.style.display = isFLF ? "none" : "block";
    rtvReferenceBehaviorSelect.value = isFLF ? "first_last_frame" : behavior;
    rtvReferenceBehaviorSelect.disabled = isFLF || !isRTV;
    firstLastFramePreviewPanel.style.display = isFLF || (isRTV && behavior === "first_last_frame") ? "flex" : "none";
    firstLastFramePreviewGrid.textContent = "";
    if (isFLF || (isRTV && behavior === "first_last_frame")) {
      const chainedFirst = Boolean(image?.chained_from_scene_id || image?.chained_from_rendered_video);
      const firstThumbPath = segment && !chainedFirst ? selectedSegmentImageThumbnailPath(segment) : "";
      const firstImage = firstThumbPath ? { path: firstThumbPath } : image;
      const arrow = document.createElement("div");
      arrow.textContent = ">";
      arrow.style.cssText = "display:flex;align-items:center;justify-content:center;color:#67e8f9;font-size:16px;font-weight:900;";
      const firstSlot = createFirstLastFramePreviewSlot(image?.chained_from_rendered_video ? "First Frame (Rendered Video End)" : chainedFirst ? "First Frame (Previous End Guide)" : "First Frame", firstImage || {}, segment ? "Missing" : "No scene");
      const endSlot = createFirstLastFramePreviewSlot("End Frame — Drop Image", lastImage || {}, "Drop or load image");
      endSlot.style.cursor = "copy";
      const holdEndFrameDrop = (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
        endSlot.style.opacity = ".72";
        endSlot.style.outline = "2px solid #22d3ee";
      };
      endSlot.addEventListener("dragenter", holdEndFrameDrop, true);
      endSlot.addEventListener("dragover", holdEndFrameDrop, true);
      endSlot.addEventListener("dragleave", (event) => {
        event.preventDefault(); event.stopPropagation();
        endSlot.style.opacity = "1"; endSlot.style.outline = "none";
      }, true);
      endSlot.addEventListener("drop", async (event) => {
        event.preventDefault(); event.stopPropagation(); event.stopImmediatePropagation();
        endSlot.style.opacity = "1"; endSlot.style.outline = "none";
        try {
          const file = Array.from(event.dataTransfer?.files || []).find((item) => String(item.type || "").startsWith("image/") || /\.(?:png|jpe?g|webp)$/i.test(String(item.name || "")));
          if (!file) throw new Error("No supported image file reached the End Frame drop target. Drop a PNG, JPG, JPEG, or WEBP file from Explorer.");
          await loadFirstLastFrameEndFile(file, segment);
        }
        catch (error) { toast(String(error?.message || error), true); }
      }, true);
      endSlot.title = hasLastImage ? "Click to open the end frame. Drag an image here to replace it." : "Click to load an end frame, or drag an image here.";
      endSlot.addEventListener("click", () => {
        if (hasLastImage) openFirstLastFrameImagePreview(lastImage, "End Frame");
        else sceneEndFrameFileInput.click();
      });
      firstLastFramePreviewGrid.append(firstSlot, arrow, endSlot);
      firstLastFrameStatus.textContent = !segment
        ? "Select a scene to inspect its First Last Frame refs."
        : hasImage && hasLastImage
          ? (isFLF
            ? independentFLF && segment.flf_end_frame_stale
              ? "The ending direction or motion plan changed. Recreate this end frame before rendering."
              : independentFLF && !segment.flf_final_prompt_ready
              ? "Start and end images are ready. Create the final FLF prompt to finish this independent pair."
              : "Ready: this scene will guide the video from its first frame to its last frame."
            : "Ready: this scene will send the first frame and end frame as the two references.")
          : hasImage
            ? `Missing end frame: create one before generating ${isFLF ? "First Last Frame" : "reference-video"} prompts or rendering.`
            : "Missing first frame: run normal Image All first, then create the end frame.";
      firstLastFrameStatus.style.color = hasImage && hasLastImage ? "#bbf7d0" : "#fde68a";
    }
    createSceneEndFrameButton.textContent = hasLastImage ? "Replace End Frame" : "Create End Frame";
    createSceneEndFrameButton.disabled = !(isFLF || (isRTV && behavior === "first_last_frame")) || !segment || !hasImage;
    planSceneEndMotionButton.style.display = independentFLF ? "" : "none";
    finalizeSceneFLFPromptButton.style.display = independentFLF ? "" : "none";
    planSceneEndMotionButton.disabled = !independentFLF || !segment || !hasImage;
    finalizeSceneFLFPromptButton.disabled = !independentFLF || !segment || !hasImage || !hasLastImage;
    planSceneEndMotionButton.style.opacity = planSceneEndMotionButton.disabled ? ".55" : "1";
    finalizeSceneFLFPromptButton.style.opacity = finalizeSceneFLFPromptButton.disabled ? ".55" : "1";
    clearSceneEndFrameButton.disabled = !(isFLF || (isRTV && behavior === "first_last_frame")) || !segment || !hasLastImage;
    createSceneEndFrameButton.style.opacity = createSceneEndFrameButton.disabled ? ".55" : "1";
    clearSceneEndFrameButton.style.opacity = clearSceneEndFrameButton.disabled ? ".55" : "1";
    if (behavior === "first_last_frame") {
      rtvReferenceBehaviorNote.textContent = hasImage
        ? (isFLF ? "Each scene's selected image is its first frame. Create or replace the separate end frame below." : "Global: each scene's selected image is the first frame. The next step will generate/store a separate last frame.")
        : "Global: run normal Image All first. Those images become first frames, then Create End Frames will make last frames.";
    } else if (behavior === "character_anchor") {
      rtvReferenceBehaviorNote.textContent = hasImage
        ? "Global: every scene will use its own scene image as a second MSR character reference when available. The video prompt is unchanged."
        : "Global: this can be enabled before Image All. Scenes without images will use their scene image as soon as one exists.";
    } else {
      rtvReferenceBehaviorNote.textContent = "Global: Reference to Video uses the normal Reference Builder subject/background refs.";
    }
  }

  function syncVideoModePanel() {
    const mode = currentVideoMode();
    state.videoModelMode = mode;
    for (const card of [imageToVideoCard, idLoraVideoCard, textToVideoCard, referenceToVideoCard, ingredientsToVideoCard, firstLastFrameVideoCard, importCustomVideoCard]) {
      const active = card.dataset.model === mode;
      card.style.borderColor = active ? "#71717a" : "#3f3f46";
      card.style.background = active ? "#52525b" : "#27272a";
      card.style.color = "#f4f4f5";
      card.style.boxShadow = active ? "inset 0 0 0 1px rgba(244,244,245,.12)" : "none";
    }
    const isImport = mode === "import";
    const isIdLora = mode === "id_lora";
    const isT2V = mode === "t2v";
    const isRTV = mode === "rtv";
    const isIngredients = mode === "ingredients";
    const isFLF = mode === "flf";
    const isT2VLike = isT2V || isRTV || isFLF || isIngredients || isIdLora;
    importCustomVideoPanel.style.display = isImport ? "flex" : "none";
    videoSubTabs.wrapper.style.display = isImport ? "none" : "";
    useI2VPromptEnhancementPass.input.checked = Boolean(state.useI2VPromptEnhancementPass);
    useI2VVisionReference.wrapper.style.display = isT2VLike ? "none" : "flex";
    i2vReferenceNote.style.display = isT2VLike ? "none" : "";
    useT2VVisionReference.wrapper.style.display = isT2V ? "flex" : "none";
    t2vReferenceNote.style.display = isT2V ? "" : "none";
    t2vLocationNote.style.display = isT2V ? "" : "none";
    t2vRefImagePanel.style.display = isT2V && useT2VVisionReference.input.checked ? "flex" : "none";
    ltxMsrRequiredPanel.style.display = isRTV ? "flex" : "none";
    ltxIngredientsRequiredPanel.style.display = isIngredients ? "flex" : "none";
    ltxIdLoraRequiredPanel.style.display = isIdLora ? "flex" : "none";
    idLoraVoiceSettingsSection.style.display = isIdLora ? "" : "none";
    flfGuideSettingsSection.style.display = isFLF ? "" : "none";
    flfTransitionTypeField.style.display = isFLF ? "flex" : "none";
    const isLegacySinglePassRTV = isRTV && (activeI2VVideoSettings()?.ltx_version || "2.5") === "2.3";
    i2vPass2NodePanel.style.display = isFLF || isLegacySinglePassRTV ? "none" : "";
    syncTimelineTrimModeButton();
    i2vWarmCooldownSection.style.display = isIdLora ? "none" : "";
    const runnerName = promptRunnerActionName();
    createI2VButton.textContent = isIdLora ? `${runnerName} ID Script` : isIngredients ? `${runnerName} Ingredients Video` : isFLF ? `${runnerName} First/Last Prompt` : isRTV ? `${runnerName} Reference Video` : isT2V ? `${runnerName} T2V` : `${runnerName} I2V`;
    syncVideoInstructionEditorButtons();
    i2vNotesInput.placeholder = isT2V
      ? "Extra text-to-video motion notes, camera movement, character movement..."
      : isIngredients
        ? "Extra ingredients-to-video motion notes, camera movement, subject actions..."
      : isIdLora
        ? "Short-film direction, dialogue intent, voice tone, camera movement, and sound cues..."
      : isRTV
        ? "Extra reference-to-video motion notes, camera movement, subject actions..."
      : isFLF
        ? "Describe the desired motion between the first and last frame..."
      : "Extra video motion notes, camera movement, character movement...";
    i2vPrompt.placeholder = isIdLora ? "[VISUAL]: ...\n[SPEECH]: ...\n[SOUNDS]: ..." : isIngredients ? "Ingredients-to-video prompt..." : isFLF ? "First-to-last-frame motion prompt..." : isRTV ? "Reference-to-video prompt..." : isT2V ? "Text-to-video prompt..." : "Image-to-video prompt...";
    syncRTVSceneImageAnchorPanel();
    updateI2VLoraVisibility();
  }

  function updateZEnhanceLoraVisibility() {
    const count = Math.max(0, Math.min(4, Number(zEnhanceLoraCount.value || 0)));
    zEnhanceLoraPanel.style.display = zEnhanceUseLora.input.checked ? "flex" : "none";
    zEnhanceLoraRows.style.display = zEnhanceUseLora.input.checked && count > 0 ? "flex" : "none";
    zEnhanceLoraSlots.forEach((slot, index) => {
      slot.row.style.display = index < count ? "grid" : "none";
    });
  }

  return {
    applyRTVReferenceBehaviorToAll, currentVideoMode, rtvReferenceBehaviorGlobalValue,
    saveI2VVideoSettingsFromPanel, saveZEnhanceSettingsFromPanel, syncI2VVideoModelPickerVisibility,
    syncI2VVideoSettingsPanel, syncRTVSceneImageAnchorPanel, syncVideoModePanel, syncZEnhanceSettingsPanel,
    updateActiveFromInputs,
  };
}
