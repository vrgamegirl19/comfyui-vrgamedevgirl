import {
  applyCompactButtonLabel,
  makeButton,
  makeCheckbox,
  makeEditField,
  makeField,
  makeInput,
  makeSelect,
  makeSettingsPanel,
  makeSettingsSection,
  makeSubTabs,
} from "./controls.mjs";

export function makeImageModelCard(label, value) {
  const card = document.createElement("button");
  card.type = "button";
  card.dataset.model = value;
  card.textContent = label;
  card.style.cssText = "height:34px;min-width:72px;flex:0 0 auto;border:1px solid #3f3f46;border-radius:999px;background:#27272a;color:#f4f4f5;font-size:12px;font-weight:900;cursor:pointer;padding:0 10px;white-space:nowrap;";
  return card;
}

function selectBuilderLlmValue(select, value) {
  const selectedValue = String(value || "").trim();
  if (!selectedValue) return;
  if (!Array.from(select.options).some((option) => option.value === selectedValue)) {
    const option = document.createElement("option");
    option.value = selectedValue;
    option.textContent = selectedValue;
    select.append(option);
  }
  select.value = selectedValue;
}

function copySelectOptions(sourceSelect, targetSelect) {
  targetSelect.textContent = "";
  for (const option of sourceSelect.options) {
    targetSelect.append(option.cloneNode(true));
  }
  targetSelect.value = sourceSelect.value || "";
}

export function setButtonGroupState(buttons, { disabled = false, text = "" } = {}) {
  for (const button of buttons) {
    button.disabled = disabled;
    if (text === "Create MiniMax H3 Scene Video") applyCompactButtonLabel(button, "Create MiniMax H3\nScene Video", { noMap: true, padding: "8px 8px", title: text });
    else if (text) button.textContent = text;
  }
}

export function buildInspectorPanels({
  browserAiAutoAdvanceGroup, browserAiBandSequenceMode, browserAiBandSequencePanel,
  browserAiCustomGroupsPanel, browserAiGroupPrompt, browserAiGroupsNote, browserAiGroupStatus,
  browserAiSessionActions, createFluxPromptButton, createNBPromptButton, createT2IButton,
  editErnieT2IInstructionsButton, editErnieT2IPromptButton, editFlowGptT2IInstructionsButton,
  editFluxKleinT2IInstructionsButton, editFluxPromptButton, editI2VMotionJsonButton,
  editKrea2T2IInstructionsButton, editKrea2TwoPassT2IPromptButton, editNanoBT2IInstructionsButton,
  editNBPromptButton, editPromptJsonButton, editStoryIdeaButton, editSubjectSceneButton, editT2IPromptButton,
  editThemeStyleButton, editZImageT2IInstructionsButton, ernieBatchSize, ernieClipPicker, ernieCreateButton,
  ernieCreateT2IButton, ernieGemmaModelSelect, ernieGrid, ernieI2IPanel, ernieImageModePanel, ernieImagePanel,
  ernieImageTriggerInput, ernieLoraPanel, ernieMmprojSelect, ernieNotesInput, ernieRefImagePanel,
  ernieSendT2IPromptToEnhanceButton, ernieT2IPrompt, ernieTextGemmaModelSelect, ernieUnetPicker,
  ernieUseImageToImage, ernieUseLora, ernieUseVisionReference, ernieVaePicker, flowGptAskPreviousImage,
  flowGptAspectRatioField, flowGptCreateImageButton, flowGptCreatePromptButton, flowGptFailureMode,
  flowGptManualActions, flowGptManualAutoAdvance, flowGptManualChatPrompt, flowGptManualMode,
  flowGptManualStatus, flowGptModePanel, flowGptPrompt, flowGptProviderRow, flowGptRetries,
  flowGptSetupActions, flowGptSetupNote, flowGptStatusText, flowGptTimeout, fluxClipPicker,
  fluxGemmaModelSelect, fluxGrid, fluxImageRefsPanel, fluxImageTriggerInput, fluxKleinModePanel,
  fluxKleinPanel, fluxLoraPanel, fluxMmprojSelect, fluxNotes, fluxPrompt, fluxUnetPicker,
  fluxUseDirectorNotes, fluxUseLora, fluxUseTextOnlyGemmaPrompt, fluxVaePicker, freezeTimingControl,
  gemmaModelSelect, i2vMotionJsonInput, idLoraIdentityGrid, idLoraReferenceAudioField,
  idLoraReferenceAudioNote, imageModelChooserWrap, imagePanel, imageTriggerInput, importI2VMotionJsonButton,
  importPromptJsonButton, inspectorActions, krea2TwoPassClipPicker, krea2TwoPassCreateButton,
  krea2TwoPassCreateT2IButton, krea2TwoPassGemmaModelSelect, krea2TwoPassI2IPanel,
  krea2TwoPassImageTriggerInput, krea2TwoPassLoraPanel, krea2TwoPassMmprojSelect, krea2TwoPassModePanel,
  krea2TwoPassNotesInput, krea2TwoPassPanel, krea2TwoPassRefImagePanel,
  krea2TwoPassSendT2IPromptToEnhanceButton, krea2TwoPassSettingsGrid, krea2TwoPassT2IPrompt,
  krea2TwoPassTextGemmaModelSelect, krea2TwoPassUnetPicker, krea2TwoPassUseImageToImage, krea2TwoPassUseLora,
  krea2TwoPassUseVisionReference, krea2TwoPassVaePicker, labelInput, loadVrgdgContextButton,
  makeErnieCreateButton, makeFluxCreateButton, makeKrea2TwoPassCreateButton, makeNBCreateButton,
  makeZCreateButton, mmprojSelect, nbApiKey, nbGemmaModelSelect, nbGlobalIngredientPanel, nbImagePanel,
  nbIngredientActions, nbIngredientDrop, nbIngredientList, nbMmprojSelect, nbModelSelect, nbNotes, nbPrompt,
  nbUseDirectorNotes, nbUseGlobalIngredients, nbUseTextOnlyGemmaPrompt, notesInput, previewFluxButton,
  previewNBButton, promptJsonInput, refImagePanel, sceneAdjustPanel, sceneDetailsPanel, scenePanel,
  sceneToolsPanel, sendFluxPromptToEnhanceButton, sendNBPromptToEnhanceButton, sendT2IPromptToEnhanceButton,
  storyIdeaInput, subjectSceneInput, t2iPrompt, t2iTextGemmaModelSelect, themeStyleInput, timingGrid,
  useSceneErnieImageSettings, useSceneFluxKleinSettings, useSceneKrea2TwoPassSettings,
  useSceneNBImageSettings, useSceneZImageSettings, useVisionReference, useVrgdgTextContext, zBatchSize,
  zClipPicker, zEnhanceAmount, zEnhanceAmountValue, zEnhanceButton, zEnhanceClipPicker, zEnhanceGemmaButton,
  zEnhanceGemmaModelSelect, zEnhanceGemmaNotes, zEnhanceGrid, zEnhanceHint, zEnhanceLoraPanel,
  zEnhanceMmprojSelect, zEnhancePanel, zEnhancePromptPreview, zEnhanceTitle, zEnhanceUnetPicker,
  zEnhanceUseLora, zEnhanceVaePicker, zFirstGrid, zFirstTitle, zI2IPanel, zImageModePanel,
  zimageSettingsPanel, zLoraPanel, zSecondGrid, zSecondTitle, zSeedGrid, zUnetPicker, zUseImageToImage,
  zUseLora, zVaePicker,
}) {
  const zImageModelsSection = makeSettingsPanel([
    makeSettingsSection("ZImage Models", [
      makeField("ZImage model", zUnetPicker.wrapper),
      makeField("CLIP", zClipPicker.wrapper),
      makeField("VAE", zVaePicker.wrapper),
    ]),
    makeSettingsSection("LLM Models", [
      makeField("Non-Vision text LLM model", t2iTextGemmaModelSelect),
      makeField("Vision LLM model", gemmaModelSelect),
      makeField("Vision mmproj", mmprojSelect),
    ]),
    zUseLora.wrapper,
    zLoraPanel,
    makeZCreateButton(),
  ]);
  const zImageSettingsSection = makeSettingsPanel([
    makeField("Image trigger phrase", imageTriggerInput),
    useSceneZImageSettings.wrapper,
    zFirstTitle,
    zFirstGrid,
    zSecondTitle,
    zSecondGrid,
    zSeedGrid,
    makeField("Batch size", zBatchSize),
    zUseImageToImage.wrapper,
    zI2IPanel,
    inspectorActions,
  ]);
  const zImagePromptSection = makeSettingsPanel([
    makeField("Notes", notesInput),
    useVisionReference.wrapper,
    refImagePanel,
    createT2IButton,
    editT2IPromptButton,
    makeField("T2I prompt", t2iPrompt),
    sendT2IPromptToEnhanceButton,
    makeSettingsSection("Advanced", [editZImageT2IInstructionsButton], false),
    makeZCreateButton(),
  ]);
  const zImageSubTabs = makeSubTabs([
    { label: "Models", value: "models", content: zImageModelsSection },
    { label: "Image Settings", value: "settings", content: zImageSettingsSection },
    { label: "LLM Prompting", value: "prompting", content: zImagePromptSection },
  ]);
  zimageSettingsPanel.append(zImageSubTabs.wrapper);
  zImageModePanel.append(zimageSettingsPanel);
  fluxKleinModePanel.append(fluxKleinPanel);
  ernieImageModePanel.append(ernieImagePanel);
  krea2TwoPassModePanel.append(krea2TwoPassPanel);
  const ernieImageSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        makeSettingsSection("Ernie Models", [
          makeField("Ernie model", ernieUnetPicker.wrapper),
          makeField("CLIP", ernieClipPicker.wrapper),
          makeField("VAE", ernieVaePicker.wrapper),
        ]),
        makeSettingsSection("LLM Models", [
          makeField("Non-Vision text LLM model", ernieTextGemmaModelSelect),
          makeField("Vision LLM model", ernieGemmaModelSelect),
          makeField("Vision mmproj", ernieMmprojSelect),
        ]),
        ernieUseLora.wrapper,
        ernieLoraPanel,
        makeErnieCreateButton(),
      ]),
    },
    {
      label: "Image Settings",
      value: "settings",
      content: makeSettingsPanel([
        useSceneErnieImageSettings.wrapper,
        makeField("Image trigger phrase", ernieImageTriggerInput),
        ernieGrid,
        makeField("Batch size", ernieBatchSize),
        ernieUseImageToImage.wrapper,
        ernieI2IPanel,
        ernieCreateButton,
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        makeField("Notes", ernieNotesInput),
        ernieUseVisionReference.wrapper,
        ernieRefImagePanel,
        ernieCreateT2IButton,
        editErnieT2IPromptButton,
        makeField("T2I prompt", ernieT2IPrompt),
        ernieSendT2IPromptToEnhanceButton,
        makeSettingsSection("Advanced", [editErnieT2IInstructionsButton], false),
        makeErnieCreateButton(),
      ]),
    },
  ]);
  ernieImagePanel.append(ernieImageSubTabs.wrapper);
  const krea2TwoPassSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        makeSettingsSection("Krea 2 Models", [
          makeField("Krea2 model", krea2TwoPassUnetPicker.wrapper),
          makeField("CLIP", krea2TwoPassClipPicker.wrapper),
          makeField("VAE", krea2TwoPassVaePicker.wrapper),
          krea2TwoPassUseLora.wrapper,
          krea2TwoPassLoraPanel,
        ]),
        makeSettingsSection("LLM Models", [
          makeField("Non-Vision text LLM model", krea2TwoPassTextGemmaModelSelect),
          makeField("Vision LLM model", krea2TwoPassGemmaModelSelect),
          makeField("Vision mmproj", krea2TwoPassMmprojSelect),
        ]),
        makeKrea2TwoPassCreateButton(),
      ]),
    },
    {
      label: "Image Settings",
      value: "settings",
      content: makeSettingsPanel([
        useSceneKrea2TwoPassSettings.wrapper,
        makeField("Image trigger phrase", krea2TwoPassImageTriggerInput),
        krea2TwoPassSettingsGrid,
        krea2TwoPassUseImageToImage.wrapper,
        krea2TwoPassI2IPanel,
        krea2TwoPassCreateButton,
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        makeField("Notes", krea2TwoPassNotesInput),
        krea2TwoPassUseVisionReference.wrapper,
        krea2TwoPassRefImagePanel,
        krea2TwoPassCreateT2IButton,
        editKrea2TwoPassT2IPromptButton,
        makeField("T2I prompt", krea2TwoPassT2IPrompt),
        krea2TwoPassSendT2IPromptToEnhanceButton,
        makeSettingsSection("Advanced", [editKrea2T2IInstructionsButton], false),
        makeKrea2TwoPassCreateButton(),
      ]),
    },
  ]);
  krea2TwoPassPanel.append(krea2TwoPassSubTabs.wrapper);
  const fluxKleinSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        makeSettingsSection("Flux/Klein Models", [
          makeField("Flux model", fluxUnetPicker.wrapper),
          makeField("Flux CLIP", fluxClipPicker.wrapper),
          makeField("Flux VAE", fluxVaePicker.wrapper),
        ]),
        makeSettingsSection("Vision LLM Models", [
        makeField("Vision LLM model", fluxGemmaModelSelect),
          makeField("Vision mmproj", fluxMmprojSelect),
        ]),
        fluxUseLora.wrapper,
        fluxLoraPanel,
        makeFluxCreateButton(),
      ]),
    },
    {
      label: "Image Settings",
      value: "settings",
      content: makeSettingsPanel([
        useSceneFluxKleinSettings.wrapper,
        fluxUseTextOnlyGemmaPrompt.wrapper,
        fluxUseDirectorNotes.wrapper,
        makeField("Image trigger phrase", fluxImageTriggerInput),
        fluxImageRefsPanel,
        fluxGrid,
        previewFluxButton,
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        makeField("Flux/Klein notes", fluxNotes),
        createFluxPromptButton,
        editFluxPromptButton,
        makeField("Flux/Klein prompt", fluxPrompt),
        sendFluxPromptToEnhanceButton,
        makeSettingsSection("Advanced", [editFluxKleinT2IInstructionsButton], false),
        makeFluxCreateButton(),
      ]),
    },
  ]);
  fluxKleinPanel.append(
    fluxKleinSubTabs.wrapper,
  );
  const nbImageSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        useSceneNBImageSettings.wrapper,
        makeSettingsSection("NanoBanana", [
          makeField("Google Cloud API key", nbApiKey),
          makeField("Model", nbModelSelect),
        ]),
        makeSettingsSection("Vision LLM Models", [
        makeField("Vision LLM model", nbGemmaModelSelect),
          makeField("Vision mmproj", nbMmprojSelect),
        ]),
        makeNBCreateButton(),
      ]),
    },
    {
      label: "Image Settings",
      value: "settings",
      content: makeSettingsPanel([
        nbUseGlobalIngredients.wrapper,
        nbUseTextOnlyGemmaPrompt.wrapper,
        nbUseDirectorNotes.wrapper,
        nbGlobalIngredientPanel,
        nbIngredientDrop,
        nbIngredientActions,
        nbIngredientList,
        makeNBCreateButton(),
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        makeField("NanoBanana notes", nbNotes),
        createNBPromptButton,
        editNBPromptButton,
        makeField("NanoBanana prompt", nbPrompt),
        sendNBPromptToEnhanceButton,
        makeSettingsSection("Advanced", [editNanoBT2IInstructionsButton], false),
        previewNBButton,
      ]),
    },
  ]);
  nbImagePanel.append(nbImageSubTabs.wrapper);
  fluxKleinModePanel.append(nbImagePanel);
  const flowGptSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        flowGptProviderRow,
        flowGptSetupNote,
        flowGptSetupActions,
        flowGptStatusText,
      ]),
    },
    {
      label: "Image Settings",
      value: "settings",
      content: makeSettingsPanel([
        flowGptAspectRatioField,
        makeField("Timeout seconds", flowGptTimeout),
        makeField("Retries", flowGptRetries),
        makeField("After final failure", flowGptFailureMode),
        flowGptAskPreviousImage.wrapper,
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        makeField("Browser prompt", flowGptPrompt),
        flowGptCreatePromptButton,
        makeSettingsSection("Advanced", [editFlowGptT2IInstructionsButton], false),
        flowGptCreateImageButton,
      ]),
    },
    {
      label: "Groups",
      value: "groups",
      content: makeSettingsPanel([
        browserAiGroupsNote,
        browserAiBandSequenceMode.wrapper,
        browserAiAutoAdvanceGroup.wrapper,
        browserAiBandSequencePanel,
        browserAiCustomGroupsPanel,
        makeField("Editable generation prompt", browserAiGroupPrompt),
        browserAiSessionActions,
        browserAiGroupStatus,
      ]),
    },
    {
      label: "Manual",
      value: "manual",
      content: makeSettingsPanel([
        flowGptManualMode.wrapper,
        flowGptManualAutoAdvance.wrapper,
        makeField("Prompt sent with reference images", flowGptManualChatPrompt),
        flowGptManualActions,
        flowGptManualStatus,
      ]),
    },
  ]);
  flowGptModePanel.append(flowGptSubTabs.wrapper);
  const zEnhanceSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        makeField("ZImage model", zEnhanceUnetPicker.wrapper),
        makeField("CLIP", zEnhanceClipPicker.wrapper),
        makeField("VAE", zEnhanceVaePicker.wrapper),
        makeSettingsSection("Vision LLM Models", [
        makeField("Vision LLM model", zEnhanceGemmaModelSelect),
          makeField("Vision mmproj", zEnhanceMmprojSelect),
        ]),
        zEnhanceUseLora.wrapper,
        zEnhanceLoraPanel,
      ]),
    },
    {
      label: "Image Settings",
      value: "settings",
      content: makeSettingsPanel([
        zEnhanceGrid,
        makeField("Enhance amount", zEnhanceAmount),
        zEnhanceAmountValue,
        zEnhanceHint,
        zEnhanceButton,
      ]),
    },
    {
      label: "LLM Prompting",
      value: "prompting",
      content: makeSettingsPanel([
        makeField("LLM notes", zEnhanceGemmaNotes),
        zEnhanceGemmaButton,
        makeField("Enhance prompt", zEnhancePromptPreview),
      ]),
    },
  ]);
  zEnhancePanel.append(
    zEnhanceTitle,
    zEnhanceSubTabs.wrapper,
  );
  sceneDetailsPanel.append(
    makeField("Scene label", labelInput),
    freezeTimingControl.wrapper,
    timingGrid,
    makeEditField("Prompt JSON path", promptJsonInput, editPromptJsonButton),
    importPromptJsonButton,
    makeEditField("I2V motion notes JSON path", i2vMotionJsonInput, editI2VMotionJsonButton),
    importI2VMotionJsonButton,
    useVrgdgTextContext.wrapper,
    loadVrgdgContextButton,
    makeEditField("Global theme/style text file", themeStyleInput, editThemeStyleButton),
    makeEditField("Global story idea text file", storyIdeaInput, editStoryIdeaButton),
    makeEditField("Global subject/scene text file", subjectSceneInput, editSubjectSceneButton),
  );
  const sceneSubTabs = makeSubTabs([
    { label: "Scene details", value: "details", content: sceneDetailsPanel },
    { label: "Scene Tools", value: "tools", content: sceneToolsPanel },
    { label: "Adjust", value: "adjust", content: sceneAdjustPanel },
  ]);
  scenePanel.append(sceneSubTabs.wrapper);
  const imageContinuityEnabled = makeCheckbox("Continue Image All from the previous scene", false);
  const imageContinuityStrength = makeSelect([
    { value: "close", label: "Close Continuation" },
    { value: "balanced", label: "Balanced Progression" },
    { value: "creative", label: "Creative Progression" },
  ], "balanced");
  const imageContinuityPanel = makeSettingsPanel([
    imageContinuityEnabled.wrapper,
    makeField("Continuity strength", imageContinuityStrength),
  ]);
  imagePanel.append(
    imageModelChooserWrap,
    imageContinuityPanel,
    zImageModePanel,
    fluxKleinModePanel,
    ernieImageModePanel,
    krea2TwoPassModePanel,
    flowGptModePanel,
    zEnhancePanel,
    inspectorActions,
  );
  const idLoraVoiceSettingsSection = makeSettingsSection("ID-LoRA Voice + Identity", [
    idLoraReferenceAudioField,
    idLoraReferenceAudioNote,
    idLoraIdentityGrid,
  ]);
  idLoraVoiceSettingsSection.style.display = "none";

  return { idLoraVoiceSettingsSection, imageContinuityEnabled, imageContinuityStrength };
}

export function buildInspectorTabs() {
  const inspectorTabs = document.createElement("div");
  inspectorTabs.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:6px;position:sticky;top:0;z-index:3;background:#202024;padding-bottom:2px;";
  const sceneTabButton = makeButton("Scene");
  const imageTabButton = makeButton("Image");
  const videoTabButton = makeButton("Video");
  const audioTabButton = makeButton("Audio");
  inspectorTabs.append(sceneTabButton, imageTabButton, videoTabButton, audioTabButton);
  const scenePanel = document.createElement("div");
  const sceneDetailsPanel = document.createElement("div");
  const sceneToolsPanel = document.createElement("div");
  const sceneAdjustPanel = document.createElement("div");
  const imagePanel = document.createElement("div");
  const videoPanel = document.createElement("div");
  const audioPanel = document.createElement("div");
  const noSceneNotice = document.createElement("div");
  noSceneNotice.style.cssText = "display:none;min-height:220px;align-items:center;justify-content:center;text-align:center;border:1px dashed #3f3f46;border-radius:8px;background:#18181b;color:#a1a1aa;padding:24px;font-size:13px;line-height:1.5;";
  noSceneNotice.innerHTML = `<div><div style="color:#e4e4e7;font-weight:900;font-size:15px;margin-bottom:6px;">Select a scene</div><div>Choose a scene from the list or timeline to edit its prompts, images, video, audio, and timing.</div></div>`;
  for (const panel of [scenePanel, imagePanel, videoPanel, audioPanel]) {
    panel.style.cssText = "display:flex;flex-direction:column;gap:10px;";
  }
  for (const panel of [sceneDetailsPanel, sceneToolsPanel, sceneAdjustPanel]) {
    panel.style.cssText = "display:flex;flex-direction:column;gap:10px;";
  }

  return {
    audioPanel, audioTabButton, imagePanel, imageTabButton, inspectorTabs, noSceneNotice, sceneAdjustPanel,
    sceneDetailsPanel, scenePanel, sceneTabButton, sceneToolsPanel, videoPanel, videoTabButton,
  };
}

export function buildInspectorInputs({ leftResizeHandle, main, preview, segmentList, shell }) {
  const imageFolderFileInput = document.createElement("input");
  imageFolderFileInput.type = "file";
  imageFolderFileInput.accept = "image/png,image/jpeg,image/webp";
  imageFolderFileInput.multiple = true;
  imageFolderFileInput.setAttribute("webkitdirectory", "");
  imageFolderFileInput.setAttribute("directory", "");
  imageFolderFileInput.style.display = "none";
  shell.append(imageFolderFileInput);
  const visionRefFileInput = document.createElement("input");
  visionRefFileInput.type = "file";
  visionRefFileInput.accept = "image/png,image/jpeg,image/webp";
  visionRefFileInput.style.display = "none";
  shell.append(visionRefFileInput);
  const i2iImageFileInput = document.createElement("input");
  i2iImageFileInput.type = "file";
  i2iImageFileInput.accept = "image/png,image/jpeg,image/webp";
  i2iImageFileInput.style.display = "none";
  shell.append(i2iImageFileInput);
  const projectAudioFileInput = document.createElement("input");
  projectAudioFileInput.type = "file";
  projectAudioFileInput.accept = "audio/wav,audio/mpeg,audio/flac,audio/mp4,audio/ogg,.wav,.mp3,.flac,.m4a,.ogg";
  projectAudioFileInput.style.display = "none";
  shell.append(projectAudioFileInput);
  const projectSrtFileInput = document.createElement("input");
  projectSrtFileInput.type = "file";
  projectSrtFileInput.accept = ".srt,text/plain";
  projectSrtFileInput.style.display = "none";
  shell.append(projectSrtFileInput);
  const inspector = document.createElement("div");
  inspector.style.cssText = "display:flex;flex-direction:column;gap:10px;padding:10px;border-left:1px solid #27272a;background:#202024;overflow-y:auto;overflow-x:hidden;min-height:0;box-sizing:border-box;scrollbar-width:thin;";
  inspector.style.gridColumn = "5";
  const rightResizeHandle = document.createElement("div");
  rightResizeHandle.title = "Drag to resize settings panel";
  rightResizeHandle.style.cssText = "cursor:col-resize;background:#18181b;border-left:1px solid #27272a;border-right:1px solid #27272a;";
  rightResizeHandle.style.gridColumn = "4";
  main.append(segmentList, leftResizeHandle, preview, rightResizeHandle, inspector);

  const labelInput = makeInput("");
  const freezeTimingControl = makeCheckbox("Freeze SRT timing", false);
  const promptJsonInput = makeInput("");
  const importPromptJsonButton = makeButton("Import Prompt JSON", "primary");
  const editPromptJsonButton = makeButton("Edit");
  const i2vMotionJsonInput = makeInput("");
  const importI2VMotionJsonButton = makeButton("Import I2V Motion Notes", "primary");
  const editI2VMotionJsonButton = makeButton("Edit");
  const imageTriggerInput = makeInput("");
  imageTriggerInput.placeholder = "Optional image trigger word or phrase...";
  const useSceneErnieImageSettings = makeCheckbox("Use custom Ernie settings for this scene", false);
  const ernieImageTriggerInput = makeInput("");
  ernieImageTriggerInput.placeholder = imageTriggerInput.placeholder;
  const useSceneKrea2TwoPassSettings = makeCheckbox("Use custom Krea 2 settings for this scene", false);
  const krea2TwoPassImageTriggerInput = makeInput("");
  krea2TwoPassImageTriggerInput.placeholder = imageTriggerInput.placeholder;
  const useSceneFluxKleinSettings = makeCheckbox("Use custom Flux/Klein settings for this scene", false);
  const fluxImageTriggerInput = makeInput("");
  fluxImageTriggerInput.placeholder = imageTriggerInput.placeholder;
  const useSceneNBImageSettings = makeCheckbox("Use custom NanoBanana settings for this scene", false);
  const useSceneI2VVideoSettings = makeCheckbox("Use custom video models/settings/LoRAs for this scene", false);
  const useSceneI2VVideoSettingsNote = document.createElement("div");
  useSceneI2VVideoSettingsNote.textContent = "Applies video model files, LoRAs, LoRA count, pass strengths, FPS, size, trigger phrase, and seed to this scene only.";
  useSceneI2VVideoSettingsNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;margin-top:-4px;";
  const videoSettingsScopeNote = document.createElement("div");
  videoSettingsScopeNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.35;";
  const videoTriggerInput = makeInput("");
  videoTriggerInput.placeholder = "Optional video trigger word or phrase...";
  const useVrgdgTextContext = makeCheckbox("Use VRGDG text context files", true);
  const loadVrgdgContextButton = makeButton("Use Default TextFiles Paths", "primary");
  const themeStyleInput = makeInput("");
  const storyIdeaInput = makeInput("");
  const subjectSceneInput = makeInput("");
  const editThemeStyleButton = makeButton("Edit");
  const editStoryIdeaButton = makeButton("Edit");
  const editSubjectSceneButton = makeButton("Edit");

  return {
    editI2VMotionJsonButton, editPromptJsonButton, editStoryIdeaButton, editSubjectSceneButton,
    editThemeStyleButton, ernieImageTriggerInput, fluxImageTriggerInput, freezeTimingControl,
    i2iImageFileInput, i2vMotionJsonInput, imageFolderFileInput, imageTriggerInput, importI2VMotionJsonButton,
    importPromptJsonButton, inspector, krea2TwoPassImageTriggerInput, labelInput, loadVrgdgContextButton,
    projectAudioFileInput, projectSrtFileInput, promptJsonInput, rightResizeHandle, storyIdeaInput,
    subjectSceneInput, themeStyleInput, useSceneErnieImageSettings, useSceneFluxKleinSettings,
    useSceneI2VVideoSettings, useSceneI2VVideoSettingsNote, useSceneKrea2TwoPassSettings,
    useSceneNBImageSettings, useVrgdgTextContext, videoSettingsScopeNote, videoTriggerInput,
    visionRefFileInput,
  };
}

export function buildInspectorControls({ endInput, previewButton, startInput }) {
  const inspectorActions = document.createElement("div");
  inspectorActions.style.cssText = "display:grid;grid-template-columns:1fr;gap:6px;";
  for (const button of [previewButton]) {
    button.style.padding = "7px 8px";
    button.style.fontSize = "11px";
  }
  const timingGrid = document.createElement("div");
  timingGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
  timingGrid.append(makeField("Start", startInput), makeField("End", endInput));
  const audioSummary = document.createElement("div");
  audioSummary.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#d4d4d8;padding:9px;font-size:12px;line-height:1.45;overflow-wrap:anywhere;";
  const globalAudioSummary = document.createElement("div");
  globalAudioSummary.style.cssText = audioSummary.style.cssText;
  const openSceneAudioOptionsButton = makeButton("Open Scene Audio Options", "primary");
  const globalAudioDrop = document.createElement("div");
  globalAudioDrop.dataset.vrgdgFileDropZone = "true";
  globalAudioDrop.textContent = "Drop global/timeline audio here";
  globalAudioDrop.style.cssText = "border:1px dashed #06b6d4;border-radius:7px;background:#082f49;color:#cffafe;padding:14px;text-align:center;font-size:12px;font-weight:900;";
  const chooseGlobalAudioButton = makeButton("Choose Global Audio", "primary");
  const globalAudioGuide = document.createElement("div");
  globalAudioGuide.style.cssText = "font-size:12px;color:#a1a1aa;line-height:1.45;";
  globalAudioGuide.innerHTML = "<strong>Pick one audio style:</strong> use Scene Audio for per-scene dialogue/short films/ads, or Global Audio for music videos, songs, visualizers, and other timeline-driven projects.";
  const globalAudioModeSelect = makeSelect([
    { value: "file", label: "Use audio file" },
    { value: "silent", label: "No audio / silent timeline" },
  ], "file");
  const silentAudioDurationInput = makeInput("60", "number");
  silentAudioDurationInput.min = "0.1";
  silentAudioDurationInput.step = "0.1";
  const createSilentTimelineAudioButton = makeButton("Create Silent Audio", "primary");
  const silentAudioPanel = document.createElement("div");
  silentAudioPanel.style.cssText = "display:grid;grid-template-columns:1fr;gap:8px;";
  const silentAudioNote = document.createElement("div");
  silentAudioNote.textContent = "Creates a silent WAV in the project folder and loads it as the timeline audio, so final render can continue without music or dialogue.";
  silentAudioNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  silentAudioPanel.append(makeField("Silent duration seconds", silentAudioDurationInput), createSilentTimelineAudioButton, silentAudioNote);

  return {
    audioSummary, chooseGlobalAudioButton, createSilentTimelineAudioButton, globalAudioDrop, globalAudioGuide,
    globalAudioModeSelect, globalAudioSummary, inspectorActions, openSceneAudioOptionsButton,
    silentAudioDurationInput, silentAudioPanel, timingGrid,
  };
}

export function createInspector({
  activeSegment, applyLayoutSizes, audioPanel, audioTabButton, ernieGemmaModelSelect, ernieMmprojSelect,
  ernieTextGemmaModelSelect, fluxGemmaModelSelect, fluxMmprojSelect, gemmaModelSelect, i2vGemmaModelSelect,
  i2vMmprojSelect, i2vPrompt, i2vTextGemmaModelSelect, imagePanel, imageTabButton, inspectorTabs,
  krea2TwoPassGemmaModelSelect, krea2TwoPassMmprojSelect, krea2TwoPassTextGemmaModelSelect,
  miniMaxGemmaModelSelect, miniMaxMmprojSelect, miniMaxTextGemmaModelSelect, mmprojSelect, nbGemmaModelSelect,
  nbMmprojSelect, noSceneNotice, saveI2VPromptButton, savedI2VPrompts, scenePanel, sceneTabButton, state,
  t2iTextGemmaModelSelect, timelinePromptSave, videoPanel, videoTabButton, zEnhanceGemmaModelSelect,
  zEnhanceMmprojSelect,
}) {
  const builderTextLlmModelSelects = [
    t2iTextGemmaModelSelect,
    ernieTextGemmaModelSelect,
    krea2TwoPassTextGemmaModelSelect,
    i2vTextGemmaModelSelect,
    miniMaxTextGemmaModelSelect,
  ];
  const builderVisionLlmModelSelects = [
    gemmaModelSelect,
    ernieGemmaModelSelect,
    krea2TwoPassGemmaModelSelect,
    zEnhanceGemmaModelSelect,
    i2vGemmaModelSelect,
    miniMaxGemmaModelSelect,
    fluxGemmaModelSelect,
    nbGemmaModelSelect,
  ];
  const builderVisionMmprojSelects = [
    mmprojSelect,
    ernieMmprojSelect,
    krea2TwoPassMmprojSelect,
    zEnhanceMmprojSelect,
    i2vMmprojSelect,
    miniMaxMmprojSelect,
    fluxMmprojSelect,
    nbMmprojSelect,
  ];
  function syncBuilderLlmModelSelectsFromRunner() {
    const runner = String(state.textGemmaRunner || "builtin").toLowerCase();
    if (!["builtin", "qwen_local"].includes(runner)) return;
    const modelFile = runner === "qwen_local" ? state.qwenModelFile : state.gemmaModelFile;
    for (const select of [...builderTextLlmModelSelects, ...builderVisionLlmModelSelects]) {
      selectBuilderLlmValue(select, modelFile);
    }
    if (runner === "qwen_local") {
      for (const select of builderVisionMmprojSelects) {
        selectBuilderLlmValue(select, state.qwenMmprojFile);
      }
    }
  }
  function syncInspectorPanels() {
    const hasScene = Boolean(activeSegment());
    const tabName = state.inspectorTab || "scene";
    noSceneNotice.style.display = hasScene ? "none" : "flex";
    inspectorTabs.style.opacity = hasScene ? "1" : ".45";
    inspectorTabs.style.pointerEvents = hasScene ? "auto" : "none";
    scenePanel.style.display = hasScene && tabName === "scene" ? "flex" : "none";
    imagePanel.style.display = hasScene && tabName === "image" ? "flex" : "none";
    videoPanel.style.display = hasScene && tabName === "video" ? "flex" : "none";
    audioPanel.style.display = hasScene && tabName === "audio" ? "flex" : "none";
  }
  function setInspectorTab(tabName) {
    const activeColor = "#06b6d4";
    const inactiveColor = "#27272a";
    state.inspectorTab = tabName;
    syncInspectorPanels();
    if (tabName === "image" || tabName === "video" || tabName === "audio") {
      state.rightPanelWidth = Math.max(state.rightPanelWidth || 360, 460);
    }
    applyLayoutSizes();
    for (const [button, name] of [[sceneTabButton, "scene"], [imageTabButton, "image"], [videoTabButton, "video"], [audioTabButton, "audio"]]) {
      const active = name === tabName;
      button.style.background = active ? activeColor : inactiveColor;
      button.style.borderColor = active ? "#0891b2" : "#3f3f46";
      button.style.color = active ? "#082f49" : "#f4f4f5";
    }
  }

  function syncKrea2TwoPassLlmSelectsFromShared() {
    copySelectOptions(t2iTextGemmaModelSelect, krea2TwoPassTextGemmaModelSelect);
    copySelectOptions(gemmaModelSelect, krea2TwoPassGemmaModelSelect);
    copySelectOptions(mmprojSelect, krea2TwoPassMmprojSelect);
  }

  function syncKrea2TwoPassLlmSelectsToShared() {
    t2iTextGemmaModelSelect.value = krea2TwoPassTextGemmaModelSelect.value || "";
    gemmaModelSelect.value = krea2TwoPassGemmaModelSelect.value || "";
    mmprojSelect.value = krea2TwoPassMmprojSelect.value || "";
  }

  function updateI2VPromptSaveButtonState() {
    const segment = activeSegment();
    if (!segment) {
      saveI2VPromptButton.disabled = true;
      saveI2VPromptButton.style.opacity = "0.5";
      saveI2VPromptButton.style.cursor = "not-allowed";
      return;
    }
    const saved = String(savedI2VPrompts.get(segment) ?? segment.i2v_prompt ?? "");
    const current = String(i2vPrompt.value || "");
    const isDirty = !timelinePromptSave.saving && current !== saved;
    saveI2VPromptButton.disabled = !isDirty;
    saveI2VPromptButton.style.opacity = isDirty ? "1" : "0.5";
    saveI2VPromptButton.style.cursor = isDirty ? "pointer" : "not-allowed";
  }

  return {
    setInspectorTab, syncBuilderLlmModelSelectsFromRunner, syncInspectorPanels,
    syncKrea2TwoPassLlmSelectsFromShared, syncKrea2TwoPassLlmSelectsToShared, updateI2VPromptSaveButtonState,
  };
}
