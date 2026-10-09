import { toast } from "./controls.mjs";
import { imageFileFromDrop } from "./media_import.mjs";
import { miniMaxH3InstructionKey } from "./minimax_h3.mjs";
import {
  cloneErnieImageSettings,
  cloneFluxKleinSettings,
  cloneKrea2TwoPassSettings,
  cloneNBImageSettings,
  cloneZImageSettings,
} from "./model_settings.mjs";

export function wireSceneInputs({
  endInput, ernieNotesInput, ernieT2IPrompt, i2vMotionJsonInput, i2vNotesInput,
  i2vPrompt, krea2TwoPassNotesInput, krea2TwoPassT2IPrompt, labelInput, lyricSingersInput, lyricTextInput,
  notesInput, promptJsonInput, pushHistory, render, startInput, state, syncInspector, t2iPrompt,
  updateActiveFromInputs, zEnhanceGemmaNotes, zEnhancePromptPreview,
}) {
  for (const control of [labelInput, startInput, endInput, notesInput, ernieNotesInput, krea2TwoPassNotesInput, i2vNotesInput, t2iPrompt, ernieT2IPrompt, krea2TwoPassT2IPrompt, i2vPrompt, zEnhanceGemmaNotes, zEnhancePromptPreview]) {
    control.addEventListener("focus", pushHistory);
    control.addEventListener("input", () => updateActiveFromInputs({ skipHistory: true }));
    control.addEventListener("change", () => updateActiveFromInputs({ skipHistory: true }));
  }
  lyricTextInput.addEventListener("focus", pushHistory);
  lyricTextInput.addEventListener("input", () => {
    lyricTextInput.dataset.vrgdgUserEdited = "1";
    updateActiveFromInputs({ skipHistory: true });
  });
  lyricTextInput.addEventListener("change", () => {
    lyricTextInput.dataset.vrgdgUserEdited = "1";
    updateActiveFromInputs({ skipHistory: true });
  });
  lyricSingersInput.addEventListener("focus", pushHistory);
  lyricSingersInput.addEventListener("input", () => {
    lyricSingersInput.dataset.vrgdgUserEdited = "1";
    updateActiveFromInputs({ skipHistory: true });
  });
  lyricSingersInput.addEventListener("change", () => {
    lyricSingersInput.dataset.vrgdgUserEdited = "1";
    updateActiveFromInputs({ skipHistory: true });
  });
  promptJsonInput.addEventListener("input", () => {
    pushHistory();
    state.promptJsonPath = promptJsonInput.value || "";
  });
  i2vMotionJsonInput.addEventListener("input", () => {
    pushHistory();
    state.i2vMotionJsonPath = i2vMotionJsonInput.value || "";
  });
}

export function wireReferenceControls({
  activeSegment, applyRTVReferenceBehaviorToAll, autoSaveSessionQuiet, clearSceneEndFrameButton,
  createEndFrameForSegment, createProgressWindow, createSceneEndFrameButton, currentVideoMode,
  ernieUseVisionReference, finalizeSceneFLFPromptButton, firstLastFrameStartImageSource, flfChainingEnabled,
  flfCustomEndDirection, flfEndpointModeSelect, flfTransitionTypeSelect,
  generateFinalIndependentFLFPromptForSegment, generateIndependentFLFMotionPlanForSegment,
  hasFirstLastFrameEndImage, krea2TwoPassUseVisionReference, loadFirstLastFrameEndFile,
  loadSceneEndFrameButton, planSceneEndMotionButton, pushHistory, refImageInput, render, renderList,
  rtvReferenceBehaviorForSegment, rtvReferenceBehaviorSelect, saveSessionForSceneVideo, sceneDisplayName,
  sceneEndFrameFileInput, segmentImageSource, segmentIndexInfo, setSceneI2VVideoSettingsEnabled,
  setSceneMiniMaxH3SettingsEnabled, state, syncErnieImagePanel, syncFluxKleinPanel, syncInspector,
  syncKrea2TwoPassPanel, syncNBImagePanel, syncRTVSceneImageAnchorPanel, syncZImageSettingsPanel,
  updateActiveFromInputs, useI2VPromptEnhancementPass, useI2VVisionReference, useSceneErnieImageSettings,
  useSceneFluxKleinSettings, useSceneI2VVideoSettings, useSceneKrea2TwoPassSettings,
  useSceneMiniMaxH3Settings, useSceneNBImageSettings, useSceneZImageSettings, useT2VVisionReference,
  useVisionReference,
}) {
  useVisionReference.input.addEventListener("change", updateActiveFromInputs);
  ernieUseVisionReference.input.addEventListener("change", updateActiveFromInputs);
  krea2TwoPassUseVisionReference.input.addEventListener("change", updateActiveFromInputs);
  useI2VVisionReference.input.addEventListener("change", updateActiveFromInputs);
  flfTransitionTypeSelect.addEventListener("change", async () => {
    updateActiveFromInputs();
    await autoSaveSessionQuiet("First Last Frame transition type changed");
  });
  useT2VVisionReference.input.addEventListener("change", updateActiveFromInputs);
  useI2VPromptEnhancementPass.input.addEventListener("change", async () => {
    state.useI2VPromptEnhancementPass = Boolean(useI2VPromptEnhancementPass.input.checked);
    await autoSaveSessionQuiet("I2V prompt enhancement setting");
    toast(state.useI2VPromptEnhancementPass ? "I2V prompt enhancement pass is on." : "I2V prompt enhancement pass is off.");
  });
  useSceneZImageSettings.input.addEventListener("change", () => {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    segment.use_scene_zimage_settings = Boolean(useSceneZImageSettings.input.checked);
    if (segment.use_scene_zimage_settings && !segment.zimage_settings) {
      segment.zimage_settings = cloneZImageSettings(state.zimageSettings);
    }
    syncZImageSettingsPanel();
    renderList();
    toast(segment.use_scene_zimage_settings ? "This scene now has custom ZImage settings." : "This scene is using global ZImage settings again.");
  });
  useSceneErnieImageSettings.input.addEventListener("change", () => {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    segment.use_scene_ernie_image_settings = Boolean(useSceneErnieImageSettings.input.checked);
    if (segment.use_scene_ernie_image_settings && !segment.ernie_image_settings) {
      segment.ernie_image_settings = cloneErnieImageSettings(state.ernieImageSettings);
    }
    syncErnieImagePanel();
    renderList();
    toast(segment.use_scene_ernie_image_settings ? "This scene now has custom Ernie settings." : "This scene is using global Ernie settings again.");
  });
  useSceneKrea2TwoPassSettings.input.addEventListener("change", () => {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    segment.use_scene_krea2_2pass_settings = Boolean(useSceneKrea2TwoPassSettings.input.checked);
    if (segment.use_scene_krea2_2pass_settings && !segment.krea2_2pass_settings) {
      segment.krea2_2pass_settings = cloneKrea2TwoPassSettings(state.krea2TwoPassSettings);
    }
    syncKrea2TwoPassPanel();
    renderList();
    toast(segment.use_scene_krea2_2pass_settings ? "This scene now has custom Krea 2 settings." : "This scene is using global Krea 2 settings again.");
  });
  useSceneFluxKleinSettings.input.addEventListener("change", () => {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    segment.use_scene_flux_klein_settings = Boolean(useSceneFluxKleinSettings.input.checked);
    if (segment.use_scene_flux_klein_settings && !segment.flux_klein_settings) {
      segment.flux_klein_settings = cloneFluxKleinSettings(state.fluxKleinSettings);
    }
    syncFluxKleinPanel();
    renderList();
    toast(segment.use_scene_flux_klein_settings ? "This scene now has custom Flux/Klein settings." : "This scene is using global Flux/Klein settings again.");
  });
  useSceneNBImageSettings.input.addEventListener("change", () => {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    segment.use_scene_nb_image_settings = Boolean(useSceneNBImageSettings.input.checked);
    if (segment.use_scene_nb_image_settings && !segment.nb_image_settings) {
      segment.nb_image_settings = cloneNBImageSettings(state.nbImageSettings);
    }
    syncNBImagePanel();
    renderList();
    toast(segment.use_scene_nb_image_settings ? "This scene now has custom NanoBanana settings." : "This scene is using global NanoBanana settings again.");
  });
  useSceneI2VVideoSettings.input.addEventListener("change", () => {
    setSceneI2VVideoSettingsEnabled(Boolean(useSceneI2VVideoSettings.input.checked));
  });
  useSceneMiniMaxH3Settings.input.addEventListener("change", () => {
    setSceneMiniMaxH3SettingsEnabled(Boolean(useSceneMiniMaxH3Settings.input.checked)).catch((error) => toast(String(error?.message || error), true));
  });
  rtvReferenceBehaviorSelect.addEventListener("change", () => {
    pushHistory();
    applyRTVReferenceBehaviorToAll(rtvReferenceBehaviorSelect.value);
    syncRTVSceneImageAnchorPanel();
    render();
    autoSaveSessionQuiet("RTV reference behavior changed globally").catch(() => null);
  });
  flfEndpointModeSelect.addEventListener("change", () => {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    segment.flf_endpoint_mode = flfEndpointModeSelect.value === "custom" ? "custom" : "auto";
    segment.flf_motion_plan = "";
    segment.flf_end_frame_prompt = "";
    segment.flf_end_frame_stale = hasFirstLastFrameEndImage(segment);
    segment.flf_final_prompt_ready = false;
    flfCustomEndDirection.parentElement.style.display = segment.flf_endpoint_mode === "custom" ? "flex" : "none";
    syncRTVSceneImageAnchorPanel();
    autoSaveSessionQuiet("Independent FLF endpoint mode changed").catch(() => null);
  });
  flfCustomEndDirection.addEventListener("input", () => {
    const segment = activeSegment();
    if (!segment) return;
    segment.flf_custom_end_direction = flfCustomEndDirection.value || "";
    segment.flf_motion_plan = "";
    segment.flf_end_frame_prompt = "";
    segment.flf_end_frame_stale = hasFirstLastFrameEndImage(segment);
    segment.flf_final_prompt_ready = false;
  });
  flfCustomEndDirection.addEventListener("change", () => {
    autoSaveSessionQuiet("Independent FLF custom ending changed").catch(() => null);
  });
  planSceneEndMotionButton.addEventListener("click", async () => {
    updateActiveFromInputs();
    const segment = activeSegment();
    if (!segment) return;
    const progress = createProgressWindow("Create Independent FLF Motion Plan");
    try {
      planSceneEndMotionButton.disabled = true;
      progress.set("Gemma is inspecting this scene's start image and planning its camera motion, subject action, and final visual moment...", 12);
      await generateIndependentFLFMotionPlanForSegment(segment, progress, 30, `${sceneDisplayName(segment, segmentIndexInfo(segment).index)} motion plan`);
      await autoSaveSessionQuiet("Independent FLF scene motion plan created");
      progress.set("Motion plan saved. Review it in the scene panel, then create the end frame.", 100);
      progress.close(2200);
      toast("Independent FLF motion plan created.");
    } catch (error) {
      const message = String(error?.message || error);
      progress.set(`Error:\n${message}`, 100);
      toast(message, true);
    } finally {
      syncRTVSceneImageAnchorPanel();
      syncInspector();
      render();
    }
  });
  finalizeSceneFLFPromptButton.addEventListener("click", async () => {
    updateActiveFromInputs();
    const segment = activeSegment();
    if (!segment) return;
    const progress = createProgressWindow("Create Final Independent FLF Prompt");
    try {
      finalizeSceneFLFPromptButton.disabled = true;
      progress.set("Gemma is inspecting this scene's actual start and end images and adapting the saved motion plan to those exact endpoints...", 12);
      await generateFinalIndependentFLFPromptForSegment(segment, progress, 35, `${sceneDisplayName(segment, segmentIndexInfo(segment).index)} final FLF prompt`);
      await autoSaveSessionQuiet("Independent FLF final scene prompt created");
      progress.set("Final FLF prompt saved. This scene is ready to render.", 100);
      progress.close(2200);
      toast("Final independent FLF prompt created.");
    } catch (error) {
      const message = String(error?.message || error);
      progress.set(`Error:\n${message}`, 100);
      toast(message, true);
    } finally {
      syncRTVSceneImageAnchorPanel();
      syncInspector();
      render();
    }
  });
  createSceneEndFrameButton.addEventListener("click", async () => {
    updateActiveFromInputs();
    const segment = activeSegment();
    if (!segment) {
      toast("Select a scene before creating an end frame.", true);
      return;
    }
    if (rtvReferenceBehaviorForSegment(segment) !== "first_last_frame") {
      toast("Set Reference Behavior to First Last Frame first.", true);
      return;
    }
    const firstFrame = currentVideoMode() === "flf"
      ? firstLastFrameStartImageSource(segment)
      : segmentImageSource(segment);
    if (!firstFrame?.path && !firstFrame?.data) {
      toast("This scene needs a first-frame image before creating the end frame.", true);
      return;
    }
    const progress = createProgressWindow("Create Scene End Frame");
    const previousLabel = createSceneEndFrameButton.textContent;
    try {
      createSceneEndFrameButton.disabled = true;
      clearSceneEndFrameButton.disabled = true;
      createSceneEndFrameButton.textContent = "Creating...";
      progress.set("Autosaving before creating the scene end frame...", 4);
      await saveSessionForSceneVideo();
      const imageMode = state.imageModelMode || "zimage";
      const independentFLF = currentVideoMode() === "flf" && !flfChainingEnabled(segment);
      if (independentFLF && !String(segment.flf_motion_plan || "").trim()) {
        progress.set("No saved motion plan exists, so Gemma is creating it from this scene's start image first...", 10);
        await generateIndependentFLFMotionPlanForSegment(segment, progress, 18, `${sceneDisplayName(segment, segmentIndexInfo(segment).index)} motion plan`);
      }
      await createEndFrameForSegment(
        segment,
        imageMode,
        progress,
        independentFLF ? 38 : 12,
        independentFLF ? 42 : 76,
        `${sceneDisplayName(segment, segmentIndexInfo(segment).index)} end frame`
      );
      if (independentFLF) {
        progress.set("End image created. Gemma is now viewing the actual start/end pair and writing the final FLF prompt...", 82);
        await generateFinalIndependentFLFPromptForSegment(segment, progress, 86, `${sceneDisplayName(segment, segmentIndexInfo(segment).index)} final FLF prompt`);
      }
      await autoSaveSessionQuiet("First Last Frame end frame created");
      progress.set(independentFLF
        ? "Scene end frame and final two-image FLF prompt created. This independent pair is ready to render."
        : "Scene end frame created. This scene is ready for First Last Frame video.", 100);
      progress.close(2200);
      toast("Scene end frame created.");
    } catch (error) {
      const message = String(error?.message || error);
      progress.set(`Error:\n${message}`, 100);
      toast(message, true);
    } finally {
      createSceneEndFrameButton.textContent = previousLabel;
      syncRTVSceneImageAnchorPanel();
      syncInspector();
      render();
    }
  });
  clearSceneEndFrameButton.addEventListener("click", () => {
    const segment = activeSegment();
    if (!segment) return;
    if (!hasFirstLastFrameEndImage(segment)) {
      syncRTVSceneImageAnchorPanel();
      return;
    }
    pushHistory();
    segment.first_last_frame_end_image_path = "";
    segment.first_last_frame_end_image_data = "";
    segment.first_last_frame_end_image_name = "";
    segment.flf_end_frame_prompt = "";
    segment.flf_end_frame_stale = false;
    segment.flf_final_prompt_ready = false;
    syncRTVSceneImageAnchorPanel();
    syncInspector();
    render();
    autoSaveSessionQuiet("First Last Frame end frame cleared").catch(() => null);
    toast("Scene end frame cleared.");
  });
  loadSceneEndFrameButton.addEventListener("click", () => sceneEndFrameFileInput.click());
  sceneEndFrameFileInput.addEventListener("change", async () => {
    try { await loadFirstLastFrameEndFile(sceneEndFrameFileInput.files?.[0]); }
    catch (error) { toast(String(error?.message || error), true); }
    finally { sceneEndFrameFileInput.value = ""; }
  });
  refImageInput.addEventListener("input", updateActiveFromInputs);
  refImageInput.addEventListener("change", updateActiveFromInputs);
}

export function wirePanelButtons({
  applyRTVReferenceBehaviorToAll, autoSaveSessionQuiet, createFlowGptPromptWithGemma,
  createFluxKleinPromptWithGemma, createFluxPromptButton, createI2VButton, createI2VPromptWithGemma,
  createMiniMaxH3PromptWithLLM, createMiniMaxSceneVideo, createNBPromptButton, createNBPromptWithGemma,
  createSceneVideo, createSceneVideoButtons, createT2IButton, createT2IPromptWithGemma, customImageFileInput,
  droppedSceneImageSource, editCurrentImagePromptWithGemma, editErnieT2IInstructionsButton,
  editFlowGptT2IInstructionsButton, editFluxKleinT2IInstructionsButton, editI2VInstructionsButton,
  editIdLoraInstructionsButton, editImagePromptButtons, editIngredientsInstructionsButton,
  editKrea2T2IInstructionsButton, editNanoBT2IInstructionsButton, editRTVInstructionsButton,
  editT2VInstructionsButton, editZImageT2IInstructionsButton, enableFluxIngredientDrop, ernieCreateButtons,
  ernieCreateT2IButton, ernieI2IDrop, ernieI2ILoadButton, ernieImageCard, ernieRefImageDrop,
  ernieRefImageLoadButton, ernieSendT2IPromptToEnhanceButton, ernieT2IPrompt, firstLastFrameVideoCard,
  flowGptCard, flowGptCreatePromptButton, fluxCreateButtons, fluxGlobalIngredientButton,
  fluxGlobalIngredientClearButton, fluxGlobalIngredientDrop, fluxGlobalIngredientFileInput,
  fluxIngredientButton, fluxIngredientClearButton, fluxIngredientDrop, fluxIngredientFileInput, fluxKleinCard,
  fluxPrompt, gemmaThenCreateVideoButtons, generateEnhancePromptWithGemma, i2iImageFileInput, idLoraVideoCard,
  imageFolderFileInput, imageToVideoCard, importCustomVideoCard, importImageFolderButton,
  importTimelineImagesFromFolder, ingredientsToVideoCard, krea2TwoPassCard, krea2TwoPassCreateButtons,
  krea2TwoPassCreateT2IButton, krea2TwoPassI2IDrop, krea2TwoPassI2ILoadButton, krea2TwoPassRefImageDrop,
  krea2TwoPassRefImageLoadButton, krea2TwoPassSendT2IPromptToEnhanceButton, krea2TwoPassT2IPrompt,
  loadCustomImage, loadCustomImageButton, loadCustomImageFile, loadFluxIngredientFile, loadImageToImageFile,
  loadVisionReferenceFile, miniMaxCreatePromptButton, miniMaxEditContinuityPromptInstructionsButton,
  miniMaxEditInstructionsButton, miniMaxH3ModeForSegment, miniMaxReferenceButtons, miniMaxSceneVideoButtons,
  nbCreateButtons, nbGlobalIngredientButton, nbGlobalIngredientClearButton, nbGlobalIngredientDrop,
  nbImageCard, nbIngredientButton, nbIngredientClearButton, nbIngredientDrop, nbUseGlobalIngredients,
  openBuilderInstructionEditor, openMiniMaxReferenceSelector, previewErnieImage, previewFluxKleinImage,
  previewKrea2TwoPassImage, previewNBImage, previewZImage, pushHistory, referenceToVideoCard, refImageDrop,
  refImageLoadButton, render, renderFluxGlobalIngredientList, renderFluxIngredientList,
  renderNBIngredientList, renderSegments, requireActiveSegment, runGemmaThenCreateSceneVideo,
  sendFluxPromptToEnhanceButton, sendPromptToEnhance, sendT2IPromptToEnhanceButton, setImageToImageSource,
  state, syncFluxGlobalIngredientPanel, syncFluxKleinPanel, syncI2VVideoSettingsPanel, syncVideoModePanel,
  t2iPrompt, t2vRefImageDrop, t2vRefImageLoadButton, textToVideoCard, useFluxGlobalIngredients,
  visionRefFileInput, wireVisionReferenceDrop, zCreateButtons, zEnhanceCard, zEnhanceGemmaButton, zI2IDrop,
  zI2ILoadButton, zImageCard,
}) {
  createT2IButton.onclick = createT2IPromptWithGemma;
  ernieCreateT2IButton.onclick = createT2IPromptWithGemma;
  editErnieT2IInstructionsButton.onclick = () => openBuilderInstructionEditor("ernie_t2i");
  krea2TwoPassCreateT2IButton.onclick = createT2IPromptWithGemma;
  editKrea2T2IInstructionsButton.onclick = () => openBuilderInstructionEditor("krea2_t2i");
  editImagePromptButtons.forEach((button) => {
    button.onclick = editCurrentImagePromptWithGemma;
  });
  createI2VButton.onclick = createI2VPromptWithGemma;
  editZImageT2IInstructionsButton.onclick = () => openBuilderInstructionEditor("zimage_t2i");
  editI2VInstructionsButton.onclick = () => openBuilderInstructionEditor("i2v");
  editIdLoraInstructionsButton.onclick = () => openBuilderInstructionEditor("id_lora");
  editRTVInstructionsButton.onclick = () => openBuilderInstructionEditor("rtv");
  editIngredientsInstructionsButton.onclick = () => openBuilderInstructionEditor("ingredients");
  editT2VInstructionsButton.onclick = () => openBuilderInstructionEditor("t2v");
  miniMaxCreatePromptButton.onclick = createMiniMaxH3PromptWithLLM;
  miniMaxEditInstructionsButton.onclick = () => {
    const segment = requireActiveSegment();
    if (!segment) return;
    openBuilderInstructionEditor(miniMaxH3InstructionKey(miniMaxH3ModeForSegment(segment)));
  };
  miniMaxEditContinuityPromptInstructionsButton.onclick = () => openBuilderInstructionEditor("minimax_h3_frame_continuity");
  sendT2IPromptToEnhanceButton.onclick = () => sendPromptToEnhance("T2I", t2iPrompt.value);
  ernieSendT2IPromptToEnhanceButton.onclick = () => sendPromptToEnhance("T2I", ernieT2IPrompt.value);
  krea2TwoPassSendT2IPromptToEnhanceButton.onclick = () => sendPromptToEnhance("T2I", krea2TwoPassT2IPrompt.value);
  sendFluxPromptToEnhanceButton.onclick = () => sendPromptToEnhance("Flux/Klein", fluxPrompt.value);
  zEnhanceGemmaButton.onclick = generateEnhancePromptWithGemma;
  createFluxPromptButton.onclick = createFluxKleinPromptWithGemma;
  editFluxKleinT2IInstructionsButton.onclick = () => openBuilderInstructionEditor("flux_klein_t2i");
  createNBPromptButton.onclick = createNBPromptWithGemma;
  editNanoBT2IInstructionsButton.onclick = () => openBuilderInstructionEditor("nano_b_t2i");
  flowGptCreatePromptButton.onclick = createFlowGptPromptWithGemma;
  editFlowGptT2IInstructionsButton.onclick = () => openBuilderInstructionEditor("flow_gpt_t2i");
  for (const button of createSceneVideoButtons) button.onclick = createSceneVideo;
  for (const button of miniMaxSceneVideoButtons) button.onclick = createMiniMaxSceneVideo;
  for (const button of miniMaxReferenceButtons) button.onclick = openMiniMaxReferenceSelector;
  for (const button of gemmaThenCreateVideoButtons) button.onclick = runGemmaThenCreateSceneVideo;
  loadCustomImageButton.onclick = loadCustomImage;
  importImageFolderButton.onclick = () => {
    imageFolderFileInput.value = "";
    imageFolderFileInput.click();
  };
  for (const button of zCreateButtons) button.onclick = previewZImage;
  for (const button of ernieCreateButtons) button.onclick = previewErnieImage;
  for (const button of krea2TwoPassCreateButtons) button.onclick = previewKrea2TwoPassImage;
  for (const button of fluxCreateButtons) button.onclick = previewFluxKleinImage;
  for (const button of nbCreateButtons) button.onclick = previewNBImage;
  customImageFileInput.addEventListener("change", () => {
    const file = customImageFileInput.files?.[0];
    if (file) loadCustomImageFile(file);
  });
  imageFolderFileInput.addEventListener("change", () => {
    const files = Array.from(imageFolderFileInput.files || []);
    if (files.length) importTimelineImagesFromFolder(files);
    imageFolderFileInput.value = "";
  });
  i2iImageFileInput.addEventListener("change", () => {
    const file = i2iImageFileInput.files?.[0];
    if (file) loadImageToImageFile(file);
    i2iImageFileInput.value = "";
  });
  zI2ILoadButton.onclick = () => i2iImageFileInput.click();
  ernieI2ILoadButton.onclick = () => i2iImageFileInput.click();
  krea2TwoPassI2ILoadButton.onclick = () => i2iImageFileInput.click();
  zImageCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "zimage";
    state.fluxKleinSettings.image_model_mode = "zimage";
    state.fluxKleinSettings.enabled = false;
    syncFluxKleinPanel();
    autoSaveSessionQuiet("image mode changed to ZImage").catch(() => null);
  };
  fluxKleinCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "flux_klein";
    state.fluxKleinSettings.image_model_mode = "flux_klein";
    state.fluxKleinSettings.enabled = true;
    syncFluxKleinPanel();
    autoSaveSessionQuiet("image mode changed to Flux/Klein").catch(() => null);
  };
  ernieImageCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "ernie_image";
    state.fluxKleinSettings.image_model_mode = "ernie_image";
    state.fluxKleinSettings.enabled = false;
    syncFluxKleinPanel();
    autoSaveSessionQuiet("image mode changed to Ernie").catch(() => null);
  };
  krea2TwoPassCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "krea2_2pass";
    state.fluxKleinSettings.image_model_mode = "krea2_2pass";
    state.fluxKleinSettings.enabled = false;
    syncFluxKleinPanel();
    autoSaveSessionQuiet("image mode changed to Krea 2").catch(() => null);
  };
  flowGptCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "flow_gpt";
    state.fluxKleinSettings.image_model_mode = "flow_gpt";
    state.fluxKleinSettings.enabled = false;
    syncFluxKleinPanel();
    autoSaveSessionQuiet("image mode changed to Flow/GPT").catch(() => null);
  };
  zEnhanceCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "z_enhance";
    state.fluxKleinSettings.image_model_mode = "z_enhance";
    state.fluxKleinSettings.enabled = false;
    syncFluxKleinPanel();
    autoSaveSessionQuiet("image mode changed to Enhance").catch(() => null);
  };
  nbImageCard.onclick = () => {
    pushHistory();
    state.imageModelMode = "nano_banana";
    state.fluxKleinSettings.image_model_mode = "nano_banana";
    state.fluxKleinSettings.enabled = false;
    syncFluxKleinPanel();
    autoSaveSessionQuiet("image mode changed to NanoBanana").catch(() => null);
  };
  imageToVideoCard.onclick = () => {
    pushHistory();
    state.videoModelMode = "i2v";
    syncVideoModePanel();
    syncI2VVideoSettingsPanel();
    renderSegments();
  };
  idLoraVideoCard.onclick = () => {
    pushHistory();
    state.videoModelMode = "id_lora";
    syncVideoModePanel();
    syncI2VVideoSettingsPanel();
    renderSegments();
  };
  textToVideoCard.onclick = () => {
    pushHistory();
    state.videoModelMode = "t2v";
    syncVideoModePanel();
    syncI2VVideoSettingsPanel();
    renderSegments();
  };
  referenceToVideoCard.onclick = () => {
    pushHistory();
    state.videoModelMode = "rtv";
    syncVideoModePanel();
    syncI2VVideoSettingsPanel();
    renderSegments();
  };
  ingredientsToVideoCard.onclick = () => {
    pushHistory();
    state.videoModelMode = "ingredients";
    syncVideoModePanel();
    syncI2VVideoSettingsPanel();
    renderSegments();
  };
  firstLastFrameVideoCard.onclick = () => {
    pushHistory();
    state.videoModelMode = "flf";
    applyRTVReferenceBehaviorToAll("first_last_frame");
    syncVideoModePanel();
    syncI2VVideoSettingsPanel();
    renderSegments();
  };
  importCustomVideoCard.onclick = () => {
    toast("Import Custom Video is coming soon.", true);
  };
  fluxIngredientFileInput.addEventListener("change", () => {
    const files = Array.from(fluxIngredientFileInput.files || []);
    for (const file of files) loadFluxIngredientFile(file);
    fluxIngredientFileInput.value = "";
  });
  fluxGlobalIngredientFileInput.addEventListener("change", () => {
    const files = Array.from(fluxGlobalIngredientFileInput.files || []);
    for (const file of files) loadFluxIngredientFile(file, { global: true });
    fluxGlobalIngredientFileInput.value = "";
  });
  fluxIngredientButton.onclick = () => fluxIngredientFileInput.click();
  fluxGlobalIngredientButton.onclick = () => fluxGlobalIngredientFileInput.click();
  nbGlobalIngredientButton.onclick = () => fluxGlobalIngredientFileInput.click();
  fluxGlobalIngredientClearButton.onclick = () => {
    pushHistory();
    state.fluxGlobalImageIngredients = [];
    renderFluxGlobalIngredientList();
    render();
    toast("Global Flux/Klein image ingredients cleared.");
  };
  nbGlobalIngredientClearButton.onclick = () => {
    pushHistory();
    state.fluxGlobalImageIngredients = [];
    renderFluxGlobalIngredientList();
    render();
    toast("Global Nano B reference images cleared.");
  };
  useFluxGlobalIngredients.input.addEventListener("change", () => {
    pushHistory();
    state.useFluxGlobalImageIngredients = Boolean(useFluxGlobalIngredients.input.checked);
    syncFluxGlobalIngredientPanel();
    render();
  });
  nbUseGlobalIngredients.input.addEventListener("change", () => {
    pushHistory();
    state.useFluxGlobalImageIngredients = Boolean(nbUseGlobalIngredients.input.checked);
    syncFluxGlobalIngredientPanel();
    render();
  });
  fluxIngredientClearButton.onclick = () => {
    const segment = requireActiveSegment();
    if (!segment) return;
    pushHistory();
    segment.flux_image_ingredients = [];
    renderFluxIngredientList(segment);
    renderNBIngredientList(segment);
    render();
    toast("Flux/Klein image ingredients cleared for this scene.");
  };
  enableFluxIngredientDrop(fluxGlobalIngredientDrop, { global: true });
  enableFluxIngredientDrop(nbGlobalIngredientDrop, { global: true });
  enableFluxIngredientDrop(fluxIngredientDrop);
  nbIngredientButton.onclick = () => fluxIngredientFileInput.click();
  nbIngredientClearButton.onclick = () => {
    const segment = requireActiveSegment();
    if (!segment) return;
    pushHistory();
    segment.flux_image_ingredients = [];
    renderFluxIngredientList(segment);
    renderNBIngredientList(segment);
    render();
    toast("NanoBanana reference images cleared for this scene.");
  };
  enableFluxIngredientDrop(nbIngredientDrop);
  zI2IDrop.addEventListener("dragover", (event) => {
    const types = Array.from(event.dataTransfer?.types || []);
    if (!types.includes("Files") && !types.includes("application/x-vrgdg-segment-id")) return;
    event.preventDefault();
    event.stopPropagation();
    zI2IDrop.style.borderColor = "#a3e635";
  });
  zI2IDrop.addEventListener("dragleave", () => {
    zI2IDrop.style.borderColor = "#155e75";
  });
  zI2IDrop.addEventListener("drop", (event) => {
    const sceneSource = droppedSceneImageSource(event);
    if (sceneSource) {
      event.preventDefault();
      event.stopPropagation();
      zI2IDrop.style.borderColor = "#155e75";
      setImageToImageSource(sceneSource);
      return;
    }
    const file = imageFileFromDrop(event);
    if (!file) return;
    event.preventDefault();
    event.stopPropagation();
    zI2IDrop.style.borderColor = "#155e75";
    loadImageToImageFile(file);
  });
  ernieI2IDrop.addEventListener("dragover", (event) => {
    const types = Array.from(event.dataTransfer?.types || []);
    if (!types.includes("Files") && !types.includes("application/x-vrgdg-segment-id")) return;
    event.preventDefault();
    event.stopPropagation();
    ernieI2IDrop.style.borderColor = "#a3e635";
  });
  ernieI2IDrop.addEventListener("dragleave", () => {
    ernieI2IDrop.style.borderColor = "#155e75";
  });
  ernieI2IDrop.addEventListener("drop", (event) => {
    const sceneSource = droppedSceneImageSource(event);
    if (sceneSource) {
      event.preventDefault();
      event.stopPropagation();
      ernieI2IDrop.style.borderColor = "#155e75";
      setImageToImageSource(sceneSource);
      return;
    }
    const file = imageFileFromDrop(event);
    if (!file) return;
    event.preventDefault();
    event.stopPropagation();
    ernieI2IDrop.style.borderColor = "#155e75";
    loadImageToImageFile(file);
  });
  krea2TwoPassI2IDrop.addEventListener("dragover", (event) => {
    const types = Array.from(event.dataTransfer?.types || []);
    if (!types.includes("Files") && !types.includes("application/x-vrgdg-segment-id")) return;
    event.preventDefault();
    event.stopPropagation();
    krea2TwoPassI2IDrop.style.borderColor = "#a3e635";
  });
  krea2TwoPassI2IDrop.addEventListener("dragleave", () => {
    krea2TwoPassI2IDrop.style.borderColor = "#155e75";
  });
  krea2TwoPassI2IDrop.addEventListener("drop", (event) => {
    const sceneSource = droppedSceneImageSource(event);
    if (sceneSource) {
      event.preventDefault();
      event.stopPropagation();
      krea2TwoPassI2IDrop.style.borderColor = "#155e75";
      setImageToImageSource(sceneSource);
      return;
    }
    const file = imageFileFromDrop(event);
    if (!file) return;
    event.preventDefault();
    event.stopPropagation();
    krea2TwoPassI2IDrop.style.borderColor = "#155e75";
    loadImageToImageFile(file);
  });
  const t2vVisionRefFileInput = document.createElement("input");
  t2vVisionRefFileInput.type = "file";
  t2vVisionRefFileInput.accept = "image/*";
  t2vVisionRefFileInput.style.display = "none";
  document.body.append(t2vVisionRefFileInput);
  refImageLoadButton.onclick = () => visionRefFileInput.click();
  ernieRefImageLoadButton.onclick = () => visionRefFileInput.click();
  krea2TwoPassRefImageLoadButton.onclick = () => visionRefFileInput.click();
  t2vRefImageLoadButton.onclick = () => t2vVisionRefFileInput.click();
  visionRefFileInput.addEventListener("change", () => {
    loadVisionReferenceFile(visionRefFileInput.files?.[0]);
    visionRefFileInput.value = "";
  });
  t2vVisionRefFileInput.addEventListener("change", () => {
    loadVisionReferenceFile(t2vVisionRefFileInput.files?.[0], { forT2V: true });
    t2vVisionRefFileInput.value = "";
  });
  wireVisionReferenceDrop(refImageDrop);
  wireVisionReferenceDrop(ernieRefImageDrop);
  wireVisionReferenceDrop(krea2TwoPassRefImageDrop);
  wireVisionReferenceDrop(t2vRefImageDrop, { forT2V: true });
}
