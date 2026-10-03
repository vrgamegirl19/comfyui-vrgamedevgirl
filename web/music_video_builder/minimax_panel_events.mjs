import { getJson } from "./comfy_api.mjs";
import { toast } from "./controls.mjs";
import { miniMaxNextCueStartTime } from "./lyric_cues.mjs";
import {
  cloneMiniMaxH3Settings,
  DEFAULT_MINIMAX_H3_SETTINGS,
  miniMaxInstalledPass2Lora,
  normalizeMiniMaxH3ContinuityMode,
  normalizeMiniMaxH3SceneImageUse,
  normalizeMiniMaxH3StartFrameCharacterInfluence,
  normalizeMiniMaxSpeakerAssignments,
  selectMiniMaxH3PassSettings,
} from "./minimax_h3.mjs";
import { syncMiniMaxSpeakerAssignmentLegacyFields } from "./minimax_speaker_cues.mjs";
import { wireSearchablePicker } from "./model_pickers.mjs";
import { selectedSegmentVideoPath } from "./selection_preview.mjs";

export function wireMiniMaxPanel({
  activeSegment, advancedTwoPassControls, allEditableSegments, autoSaveSessionQuiet,
  clearMiniMaxImageReferenceStartFrameOnModeSwitch, ensureMiniMaxSpeakerAssignments,
  isMiniMaxSingerAssignmentMode, loadDirtyLatentBadges, miniMaxAccelerationControls,
  miniMaxAddSpeakerCueButton, miniMaxAdvancedLatentUpscalerPicker, miniMaxAdvancedVramPreset, miniMaxAspectRatio, miniMaxAudioMode, miniMaxAudioVaePicker, miniMaxClipPicker,
  miniMaxContinuityMode, miniMaxContinuityPromptFromLastFrame, miniMaxCooldownFrames, miniMaxDenoise,
  miniMaxDiffusionModelPicker, miniMaxEasyCacheBypass, miniMaxEasyCacheEndPercent,
  miniMaxEasyCacheReuseThreshold, miniMaxEasyCacheStartPercent, miniMaxEasyCacheVerbose,
  miniMaxFp16Accumulation, miniMaxH3ContinuityModeForSegment, miniMaxH3ModeForSegment,
  miniMaxH3SceneImageUseForSegment, miniMaxH3SettingsForSegment, miniMaxLatentContextFrames,
  miniMaxLocationTransitionCustom, miniMaxLocationTransitionPreset, miniMaxLoraCount, miniMaxLoraSlots,
  miniMaxMappedSpeakersForSegment, miniMaxMegapixels, miniMaxMemoryEfficientSageAttention, miniMaxResolutionPreset, miniMaxModeButtons,
  miniMaxPass2Prompt, miniMaxPassButtons, miniMaxPrompt, miniMaxSageAttention, miniMaxSamplerName,
  miniMaxSceneImageUse, miniMaxScheduler, miniMaxSeed, miniMaxStartFrameCharacterInfluence, miniMaxSteps,
  miniMaxThreePassLoraPicker, miniMaxThreePassLoraStrength, miniMaxThreePassRefImageSize,
  miniMaxTurboLoraPicker, miniMaxTurboLoraStrength, miniMaxTwoPassLatentScale, miniMaxTwoPassLatentUpscalerPicker, miniMaxTwoPassLoraPicker,
  miniMaxTwoPassLoraPreset, miniMaxTwoPassLoraPresetButtons, miniMaxTwoPassLoraStatus,
  miniMaxTwoPassLoraStrength, miniMaxTwoPassOutputCrf, miniMaxTwoPassRefImageSize, miniMaxTwoPassResizeMethod,
  miniMaxTwoPassTeCacheDepth, miniMaxTwoPassTeDevice, miniMaxTwoPassTeEnd, miniMaxTwoPassTeMcs,
  miniMaxTwoPassTeProcessingControl, miniMaxTwoPassTeStart, miniMaxTwoPassUseFastVaeDecode,
  miniMaxUseCurrentSceneVideoButton, miniMaxUseLoras, miniMaxUseTurboLora, miniMaxVideoReferenceRows,
  miniMaxVideoVaePicker, miniMaxWarmupFrames, pushHistory, renderMiniMaxSpeakerAssignmentPanel,
  requireActiveSegment, saveMiniMaxH3SettingsFromPanel, saveMiniMaxSceneInputsFromPanel, segmentImageSource,
  segmentTrack, selectedPerformerSubjectsForSegment, setMiniMaxH3ModeForSegment,
  setMiniMaxH3RenderPassForSegment, state, syncMiniMaxH3Panel, syncMiniMaxReferenceButtons, twoPassControls,
  videoSettingsSegment, wizardVideoSettings,
}) {
  const persistMiniMaxSettings = () => {
    saveMiniMaxH3SettingsFromPanel();
    autoSaveSessionQuiet("MiniMax H3 project settings").catch(() => null);
  };
  for (const picker of [miniMaxDiffusionModelPicker, miniMaxClipPicker, miniMaxVideoVaePicker, miniMaxAudioVaePicker, miniMaxTwoPassLatentUpscalerPicker, miniMaxAdvancedLatentUpscalerPicker]) {
    wireSearchablePicker(picker, saveMiniMaxH3SettingsFromPanel);
    picker.input.addEventListener("change", persistMiniMaxSettings);
  }
  wireSearchablePicker(miniMaxTurboLoraPicker, saveMiniMaxH3SettingsFromPanel);
  miniMaxTurboLoraPicker.input.addEventListener("change", persistMiniMaxSettings);
  for (const button of miniMaxTwoPassLoraPresetButtons) {
    button.onclick = async () => {
      if (!state.miniMaxH3TwoPassEnabled || state.miniMaxH3ThreePassEnabled) return;
      for (const presetButton of miniMaxTwoPassLoraPresetButtons) presetButton.disabled = true;
      try {
        const data = await getJson("/vrgdg/workflow_runner/lora_list");
        if (!state.miniMaxH3TwoPassEnabled || state.miniMaxH3ThreePassEnabled) return;
        miniMaxTwoPassLoraPicker.options = data.loras || [];
        const lora = miniMaxInstalledPass2Lora(button.dataset.preset, miniMaxTwoPassLoraPicker.options);
        if (!lora) {
          miniMaxTwoPassLoraStatus.textContent = `No matching ${button.dataset.preset}-step LoRA installed. Download one below, then click the preset again.`;
          return;
        }
        pushHistory();
        miniMaxTwoPassLoraPicker.input.value = lora;
        miniMaxTwoPassLoraPreset.dataset.preset = button.dataset.preset;
        twoPassControls[1].steps.value = String(Number(button.dataset.preset) / 2);
        miniMaxTwoPassLoraStatus.textContent = "";
        saveMiniMaxH3SettingsFromPanel();
        syncMiniMaxH3Panel();
        await autoSaveSessionQuiet("MiniMax H3 Pass 2 LoRA preset");
      } catch (error) {
        miniMaxTwoPassLoraStatus.textContent = `Could not apply LoRA preset: ${String(error?.message || error)}`;
      } finally {
        for (const presetButton of miniMaxTwoPassLoraPresetButtons) presetButton.disabled = false;
      }
    };
  }
  wireSearchablePicker(miniMaxTwoPassLoraPicker, saveMiniMaxH3SettingsFromPanel);
  miniMaxTwoPassLoraPicker.input.addEventListener("change", persistMiniMaxSettings);
  wireSearchablePicker(miniMaxThreePassLoraPicker, saveMiniMaxH3SettingsFromPanel);
  miniMaxThreePassLoraPicker.input.addEventListener("change", persistMiniMaxSettings);
  for (const slot of miniMaxLoraSlots) {
    wireSearchablePicker(slot.picker, saveMiniMaxH3SettingsFromPanel);
    slot.picker.input.addEventListener("change", persistMiniMaxSettings);
    slot.strength.addEventListener("input", saveMiniMaxH3SettingsFromPanel);
    slot.strength.addEventListener("change", persistMiniMaxSettings);
    slot.applyTo.addEventListener("change", persistMiniMaxSettings);
  }
  for (const control of [
    miniMaxAspectRatio,
    miniMaxResolutionPreset,
    miniMaxAudioMode,
    miniMaxContinuityMode,
    miniMaxContinuityPromptFromLastFrame.input,
    miniMaxLatentContextFrames,
    miniMaxMegapixels,
    miniMaxSeed,
    miniMaxWarmupFrames,
    miniMaxCooldownFrames,
    miniMaxSamplerName,
    miniMaxScheduler,
    miniMaxSteps,
    miniMaxDenoise,
    miniMaxEasyCacheBypass.input,
    miniMaxEasyCacheReuseThreshold,
    miniMaxEasyCacheStartPercent,
    miniMaxEasyCacheEndPercent,
    miniMaxEasyCacheVerbose.input,
    miniMaxSageAttention,
    miniMaxMemoryEfficientSageAttention.input,
    miniMaxFp16Accumulation.input,
    miniMaxUseLoras.input,
    miniMaxLoraCount,
    miniMaxTurboLoraStrength,
    miniMaxTwoPassLoraStrength,
    miniMaxThreePassLoraStrength,
    miniMaxTwoPassRefImageSize,
    miniMaxThreePassRefImageSize,
    miniMaxAdvancedVramPreset,
    miniMaxTwoPassLatentScale,
    ...miniMaxAccelerationControls.flatMap(({ single, pass1, pass2 }) => [single.input, pass1.input, pass2.input]),
    miniMaxTwoPassUseFastVaeDecode.input,
    miniMaxTwoPassTeProcessingControl,
    miniMaxTwoPassTeStart,
    miniMaxTwoPassTeEnd,
    miniMaxTwoPassTeMcs,
    miniMaxTwoPassTeCacheDepth,
    miniMaxTwoPassTeDevice,
    miniMaxTwoPassResizeMethod,
    miniMaxTwoPassOutputCrf,
    ...twoPassControls.flatMap((pass) => [
      pass.steps,
      pass.denoise,
      pass.sampler,
      pass.scheduler,
      pass.seed,
    ]),
    ...advancedTwoPassControls.flatMap((pass) => [
      ...(pass.megapixels ? [pass.resolutionPreset, pass.megapixels] : []),
      pass.steps,
      pass.denoise,
      pass.sampler,
      pass.scheduler,
      pass.seed,
    ]),
  ]) {
    control.addEventListener("input", saveMiniMaxH3SettingsFromPanel);
    control.addEventListener("change", persistMiniMaxSettings);
  }
  miniMaxUseLoras.input.addEventListener("change", () => {
    const segment = videoSettingsSegment();
    if (miniMaxUseLoras.input.checked) {
      miniMaxUseTurboLora.input.checked = false;
      if (Math.max(0, Math.trunc(Number(miniMaxLoraCount.value) || 0)) < 1) miniMaxLoraCount.value = "1";
    }
    const settings = saveMiniMaxH3SettingsFromPanel();
    if (segment?.use_scene_minimax_h3_settings) segment.minimax_h3_settings = settings;
    else state.miniMaxH3Settings = settings;
    syncMiniMaxH3Panel();
    autoSaveSessionQuiet("MiniMax H3 LoRA setting").catch(() => null);
  });
  miniMaxLoraCount.addEventListener("change", () => {
    miniMaxLoraCount.value = String(Math.max(0, Math.min(4, Math.trunc(Number(miniMaxLoraCount.value) || 0))));
    if (Number(miniMaxLoraCount.value || 0) > 0) miniMaxUseLoras.input.checked = true;
    persistMiniMaxSettings();
    syncMiniMaxH3Panel();
  });
  miniMaxUseTurboLora.input.addEventListener("change", () => {
    const segment = videoSettingsSegment();
    const currentSettings = miniMaxH3SettingsForSegment(segment);
    if (miniMaxUseTurboLora.input.checked) {
      miniMaxUseLoras.input.checked = false;
      currentSettings.steps_before_turbo = Math.max(1, Math.trunc(Number(miniMaxSteps.value) || DEFAULT_MINIMAX_H3_SETTINGS.steps));
      currentSettings.easy_cache_bypass_before_turbo = miniMaxEasyCacheBypass.input.checked;
      miniMaxSteps.value = "4";
      miniMaxEasyCacheBypass.input.checked = true;
    } else {
      miniMaxSteps.value = String(currentSettings.steps_before_turbo || DEFAULT_MINIMAX_H3_SETTINGS.steps);
      miniMaxEasyCacheBypass.input.checked = Boolean(currentSettings.easy_cache_bypass_before_turbo);
    }
    const settings = saveMiniMaxH3SettingsFromPanel();
    settings.steps_before_turbo = currentSettings.steps_before_turbo;
    settings.easy_cache_bypass_before_turbo = currentSettings.easy_cache_bypass_before_turbo;
    if (segment?.use_scene_minimax_h3_settings) {
      segment.minimax_h3_settings = settings;
    } else {
      state.miniMaxH3Settings = settings;
    }
    syncMiniMaxH3Panel();
    autoSaveSessionQuiet("MiniMax H3 Turbo setting").catch(() => null);
  });
  for (const button of miniMaxModeButtons) {
    button.onclick = async () => {
      const segment = wizardVideoSettings.global ? null : requireActiveSegment();
      if (!segment && !wizardVideoSettings.global) return;
      pushHistory();
      clearMiniMaxImageReferenceStartFrameOnModeSwitch(segment, button.dataset.minimaxH3Mode);
      setMiniMaxH3RenderPassForSegment(segment, button.dataset.minimaxH3Mode === "image_reference_to_video" ? "two_pass" : "single");
      setMiniMaxH3ModeForSegment(segment, button.dataset.minimaxH3Mode);
      syncMiniMaxH3Panel();
      await autoSaveSessionQuiet(segment.use_scene_minimax_h3_settings ? "MiniMax H3 locked scene mode" : "MiniMax H3 project mode");
    };
  }
  for (const button of miniMaxPassButtons) {
    button.onclick = async () => {
      const segment = wizardVideoSettings.global ? null : requireActiveSegment();
      if (!segment && !wizardVideoSettings.global) return;
      pushHistory();
      clearMiniMaxImageReferenceStartFrameOnModeSwitch(segment, "reference_to_video");
      const current = saveMiniMaxH3SettingsFromPanel(segment);
      const settings = selectMiniMaxH3PassSettings(current, button.dataset.passMode);
      if (segment?.use_scene_minimax_h3_settings) segment.minimax_h3_settings = settings;
      else state.miniMaxH3Settings = settings;
      if (segment) setMiniMaxH3RenderPassForSegment(segment, button.dataset.passMode);
      if (segment) setMiniMaxH3ModeForSegment(segment, "reference_to_video");
      else state.miniMaxH3Settings = cloneMiniMaxH3Settings({ ...settings, video_mode: "reference_to_video" });
      syncMiniMaxH3Panel();
      await autoSaveSessionQuiet("MiniMax H3 reference pass settings");
    };
  }
  miniMaxAudioMode.addEventListener("change", syncMiniMaxH3Panel);
  miniMaxContinuityPromptFromLastFrame.input.addEventListener("change", syncMiniMaxH3Panel);
  miniMaxLocationTransitionPreset.addEventListener("change", () => {
    pushHistory();
    saveMiniMaxH3SettingsFromPanel();
    syncMiniMaxH3Panel();
    autoSaveSessionQuiet("MiniMax H3 location transition preset").catch(() => null);
  });
  miniMaxLocationTransitionCustom.addEventListener("input", saveMiniMaxH3SettingsFromPanel);
  miniMaxLocationTransitionCustom.addEventListener("change", () => {
    saveMiniMaxH3SettingsFromPanel();
    autoSaveSessionQuiet("MiniMax H3 custom location transition").catch(() => null);
  });
  miniMaxContinuityMode.addEventListener("change", () => {
    const segment = activeSegment();
    const continuityMode = normalizeMiniMaxH3ContinuityMode(miniMaxContinuityMode.value);
    const affectedSegments = segment?.use_scene_minimax_h3_settings
      ? [segment]
      : allEditableSegments().filter((item) => (
        !item?.use_scene_minimax_h3_settings
        && miniMaxH3ContinuityModeForSegment(item) === "exact_start_frame"
      ));
    const clearedStartFrames = continuityMode === "exact_start_frame"
      ? affectedSegments.filter((item) => item?.minimax_h3_use_scene_image_as_start_frame)
      : [];
    for (const item of clearedStartFrames) {
      item.minimax_h3_use_scene_image_as_start_frame = false;
      item.minimax_h3_scene_image_use = "off";
    }
    if (clearedStartFrames.length) {
      miniMaxSceneImageUse.value = "off";
      toast(`Previous-frame exact continuation is now the sole exact start frame. Turned off the scene-image exact start-frame option for ${clearedStartFrames.length} scene${clearedStartFrames.length === 1 ? "" : "s"}.`);
    }
    syncMiniMaxH3Panel();
    syncMiniMaxReferenceButtons();
    loadDirtyLatentBadges();
    autoSaveSessionQuiet("MiniMax H3 continuity mode changed").catch(() => null);
  });
  miniMaxAddSpeakerCueButton.onclick = () => {
    const segment = requireActiveSegment();
    if (!segment) return;
    const speakers = miniMaxMappedSpeakersForSegment(segment);
    if (!speakers.length) {
      toast("Map at least one character to this scene in Reference Builder first.", true);
      return;
    }
    pushHistory();
    if (isMiniMaxSingerAssignmentMode(segment)) {
      const selectedPerformers = selectedPerformerSubjectsForSegment(segment);
      const performer = selectedPerformers[0] || speakers[0];
      segment.lyric_performance_mode = selectedPerformers.length >= 2 ? "cue_map" : "together";
      segment.lyric_singers = selectedPerformers.length
        ? selectedPerformers.map((subject) => subject.name || "Character")
        : [performer.name || "Character"];
      const cues = Array.isArray(segment.lyric_cue_map) ? segment.lyric_cue_map : [];
      const start = miniMaxNextCueStartTime(segment, cues);
      cues.push({ type: "vocal", text: "", action_note: "", singer_id: performer.id || "", singer_name: performer.name || "", start, end: null });
      segment.lyric_cue_map = cues;
      renderMiniMaxSpeakerAssignmentPanel();
      autoSaveSessionQuiet("MiniMax lyric cue added").catch(() => null);
      return;
    }
    const cues = ensureMiniMaxSpeakerAssignments(segment, speakers);
    const speaker = speakers[0];
    cues.push(...normalizeMiniMaxSpeakerAssignments([{ speaker_id: speaker.id, speaker_name: speaker.name, text: "" }]));
    segment.minimax_speaker_assignments = cues;
    syncMiniMaxSpeakerAssignmentLegacyFields(segment);
    renderMiniMaxSpeakerAssignmentPanel();
    autoSaveSessionQuiet("MiniMax dialogue cue added").catch(() => null);
  };
  miniMaxPrompt.addEventListener("input", saveMiniMaxSceneInputsFromPanel);
  miniMaxPrompt.addEventListener("change", () => autoSaveSessionQuiet("MiniMax H3 scene prompt").catch(() => null));
  miniMaxPass2Prompt.addEventListener("input", saveMiniMaxSceneInputsFromPanel);
  miniMaxPass2Prompt.addEventListener("change", () => autoSaveSessionQuiet("MiniMax 2nd Pass Prompt").catch(() => null));
  miniMaxSceneImageUse.addEventListener("change", async () => {
    const segment = requireActiveSegment();
    if (!segment) return;
    const sceneImageUse = normalizeMiniMaxH3SceneImageUse(miniMaxSceneImageUse.value);
    const exactStartFrame = sceneImageUse === "exact_start_frame";
    const usesSceneImage = sceneImageUse !== "off";
    const characterInfluence = normalizeMiniMaxH3StartFrameCharacterInfluence(
      miniMaxStartFrameCharacterInfluence.value,
    );
    const sceneLocked = Boolean(segment.use_scene_minimax_h3_settings);
    const candidates = (sceneLocked ? [segment] : allEditableSegments()).filter((item) => (
      segmentTrack(item) !== "overlay"
      && ["reference_to_video", "image_reference_to_video"].includes(miniMaxH3ModeForSegment(item))
      && (sceneLocked || !item.use_scene_minimax_h3_settings)
    ));
    const blocked = exactStartFrame
      ? candidates.filter((item) => miniMaxH3ContinuityModeForSegment(item) === "exact_start_frame")
      : [];
    const targets = exactStartFrame
      ? candidates.filter((item) => miniMaxH3ContinuityModeForSegment(item) !== "exact_start_frame")
      : candidates;
    if (!targets.length) {
      miniMaxSceneImageUse.value = miniMaxH3SceneImageUseForSegment(segment);
      toast(blocked.length
        ? "Every Reference-to-Video scene is using the previous rendered frame as its exact start, so the scene-image start frame could not be enabled."
        : "There are no eligible Reference-to-Video scenes to update.", true);
      return;
    }
    pushHistory();
    for (const item of targets) {
      item.minimax_h3_scene_image_use = sceneImageUse;
      item.minimax_h3_use_scene_image_as_start_frame = exactStartFrame;
      item.minimax_h3_start_frame_character_influence = characterInfluence;
    }
    const missingSceneImages = usesSceneImage
      ? targets.filter((item) => !Boolean(segmentImageSource(item)?.path || segmentImageSource(item)?.data))
      : [];
    syncMiniMaxH3Panel();
    syncMiniMaxReferenceButtons();
    await autoSaveSessionQuiet("MiniMax H3 scene image use");
    if (usesSceneImage) {
      const scopeLabel = sceneLocked
        ? "this locked Reference-to-Video scene"
        : `${targets.length} unlocked Reference-to-Video scene${targets.length === 1 ? "" : "s"}`;
      const action = exactStartFrame
        ? "Uses each scene image as MiniMax Image 1 and the exact start frame"
        : sceneImageUse === "environment_inspiration"
          ? "Uses each scene image only as environment inspiration for the prompt-writing LLM; framing and all character details are ignored, and MiniMax never receives the image"
          : "Uses each scene image only as environment and framing inspiration for the prompt-writing LLM; all character details are ignored, and MiniMax never receives the image";
      const details = [
        `${action} for ${scopeLabel}.`,
        blocked.length ? `Skipped ${blocked.length} scene${blocked.length === 1 ? "" : "s"} using previous-frame exact continuity.` : "",
        missingSceneImages.length ? `${missingSceneImages.length} enabled scene${missingSceneImages.length === 1 ? " still needs" : "s still need"} a timeline image before prompting or rendering.` : "",
        "Regenerate existing MiniMax prompts to apply this setup.",
      ].filter(Boolean);
      toast(details.join("\n"), Boolean(missingSceneImages.length));
    } else {
      toast(sceneLocked
        ? "The scene image will not be used for this locked Reference-to-Video scene."
        : `Turned off scene-image use for all ${targets.length} unlocked Reference-to-Video scene${targets.length === 1 ? "" : "s"}.`);
    }
  });
  miniMaxStartFrameCharacterInfluence.addEventListener("change", async () => {
    const segment = requireActiveSegment();
    if (!segment) return;
    const characterInfluence = normalizeMiniMaxH3StartFrameCharacterInfluence(
      miniMaxStartFrameCharacterInfluence.value,
    );
    const sceneLocked = Boolean(segment.use_scene_minimax_h3_settings);
    const targets = (sceneLocked ? [segment] : allEditableSegments()).filter((item) => (
      segmentTrack(item) !== "overlay"
      && ["reference_to_video", "image_reference_to_video"].includes(miniMaxH3ModeForSegment(item))
      && (sceneLocked || !item.use_scene_minimax_h3_settings)
    ));
    if (!targets.length) {
      toast("There are no eligible Reference-to-Video scenes to update.", true);
      return;
    }
    pushHistory();
    for (const item of targets) {
      item.minimax_h3_start_frame_character_influence = characterInfluence;
    }
    syncMiniMaxH3Panel();
    await autoSaveSessionQuiet("MiniMax H3 global start-frame character influence");
    const scopeLabel = sceneLocked
      ? "this locked Reference-to-Video scene"
      : `all ${targets.length} unlocked Reference-to-Video scene${targets.length === 1 ? "" : "s"}`;
    toast(miniMaxStartFrameCharacterInfluence.value === "face_hair_only"
      ? `Character references will supply only face and hair for ${scopeLabel}. Regenerate existing MiniMax prompts to apply the new priority.`
      : `Character references may supply the full character identity for ${scopeLabel}. Regenerate existing MiniMax prompts to apply the new priority.`);
  });
  for (const row of miniMaxVideoReferenceRows) {
    for (const control of [row.path, row.start, row.duration, row.purpose, row.useAudio.input]) {
      control.addEventListener("input", saveMiniMaxSceneInputsFromPanel);
      control.addEventListener("change", () => {
        saveMiniMaxSceneInputsFromPanel();
        autoSaveSessionQuiet("MiniMax H3 video references").catch(() => null);
      });
    }
  }
  miniMaxUseCurrentSceneVideoButton.onclick = async () => {
    const segment = requireActiveSegment();
    if (!segment) return;
    const path = String(selectedSegmentVideoPath(segment) || "").trim();
    if (!path) {
      toast("This scene does not have a selected video to use as a MiniMax reference.", true);
      return;
    }
    pushHistory();
    miniMaxVideoReferenceRows[0].path.value = path;
    saveMiniMaxSceneInputsFromPanel();
    syncMiniMaxH3Panel();
    await autoSaveSessionQuiet("MiniMax H3 current scene video reference");
  };
}
