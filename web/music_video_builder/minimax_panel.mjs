import { postJson } from "./comfy_api.mjs";
import { applyCompactButtonLabel, normalizeProjectVideoEngine, toast } from "./controls.mjs";
import {
  cloneMiniMaxH3Settings,
  DEFAULT_MINIMAX_H3_SETTINGS,
  isMiniMaxH3ContinuityAllowedForMode,
  miniMaxH3ContinuationStartLimits,
  miniMaxH3ContinuationStartSeconds,
  isMiniMaxH3LatentContinuationMode,
  miniMaxH3ModeLabel,
  normalizeMiniMaxH3ContinuityMode,
  normalizeMiniMaxH3LocationTransitionPreset,
  normalizeMiniMaxH3Mode,
  normalizeMiniMaxH3SceneImageUse,
  normalizeMiniMaxH3StartFrameCharacterInfluence,
  normalizeMiniMaxH3VideoPurpose,
} from "./minimax_h3.mjs";
import { createMiniMaxSpeakerAssignments } from "./minimax_speaker_assignments.mjs";

export function createMiniMaxPanel({
  activateGlobalTimelineAudioPlayback, activeSegment, advancedTwoPassControls, allEditableSegments, audio,
  autoSaveSessionQuiet, autoTimeMiniMaxSingerCuesForSegment, createProgressWindow, gemmaRunnerLabel,
  miniMaxAccelerationControls, miniMaxAddSpeakerCueButton, miniMaxAdvancedLatentUpscalerPicker, miniMaxAdvancedSettings,
  miniMaxAdvancedVramPreset,
  miniMaxAspectRatio, miniMaxAudioMode, miniMaxAudioNote, miniMaxAudioVaePicker, miniMaxAutoTimeBeforePrompt,
  miniMaxClipPicker, miniMaxContinuityMode, miniMaxContinuityNote, miniMaxContinuitySection, miniMaxContinuityPromptFromLastFrame,
  miniMaxCooldownFrames, miniMaxCreatePromptButton, miniMaxDenoise, miniMaxDesiredReferenceKeysForSegment,
  miniMaxDiffusionModelPicker, miniMaxEasyCacheBypass, miniMaxEasyCacheEndPercent,
  miniMaxEasyCacheReuseThreshold, miniMaxEasyCacheSettings, miniMaxEasyCacheStartPercent,
  miniMaxEasyCacheVerbose, miniMaxEditInstructionsButton, miniMaxFp16Accumulation,
  miniMaxH3PromptCharacterBudget, miniMaxH3ReferenceCapacityStatus, miniMaxImageModeSource, miniMaxKeyframes,
  miniMaxContinuationDirection, miniMaxContinuationDirectionField, miniMaxPromptAutoNote, miniMaxH3FrameContinuityPromptEnabled,
  miniMaxContinuationStart, miniMaxContinuationStartField, miniMaxContinuationStartValue,
  miniMaxLatentContextFrames, miniMaxLatentContinuationRow, miniMaxLatentStatusPill,
  miniMaxLocationTransitionControls, miniMaxLocationTransitionCustom, miniMaxLocationTransitionCustomField,
  miniMaxLocationTransitionPreset, miniMaxLoraCount, miniMaxLoraNote, miniMaxLoraRows, miniMaxLoraSection,
  miniMaxLoraSlots, miniMaxMegapixels, miniMaxMegapixelsField, miniMaxResolutionPreset, miniMaxMemoryEfficientSageAttention,
  miniMaxModeButtons, miniMaxModePanels, miniMaxModelLoaderSettings,
  miniMaxOrderedImageReferenceItemsForSegment, miniMaxPass2Prompt, miniMaxPass2PromptField,
  miniMaxPassButtons, miniMaxPassChooser, miniMaxPrompt, miniMaxPromptCharacterStatus,
  miniMaxPromptReferenceMismatch, miniMaxPromptRunnerNote, miniMaxRefImageSize, miniMaxReferenceButtons,
  miniMaxReferenceConditioningSettings, miniMaxSageAttention, miniMaxSamplerName, miniMaxSamplerSettings,
  miniMaxSceneImageUse, miniMaxSceneImageUseField, miniMaxSceneVideoButton, miniMaxScheduler, miniMaxSeed,
  miniMaxSeedField, miniMaxSettingsScopeNote, miniMaxSpeakerAssignmentList, miniMaxSpeakerAssignmentNote,
  miniMaxStartFrameCharacterInfluence, miniMaxStartFrameCharacterInfluenceField,
  miniMaxStartFrameReferenceNote, miniMaxSteps, miniMaxSubTabs, miniMaxThreePassLoraPicker,
  miniMaxThreePassLoraSection, miniMaxThreePassLoraStrength, miniMaxThreePassRefImageSize,
  miniMaxThreePassSettings, miniMaxTurboLoraField, miniMaxTurboLoraPicker, miniMaxTurboLoraStrength,
  miniMaxTurboLoraStrengthField, miniMaxTurboNote, miniMaxTurboSection, miniMaxTwoPassLatentScale, miniMaxTwoPassLatentUpscalerPicker,
  miniMaxTwoPassLoraLayout, miniMaxTwoPassLoraPicker, miniMaxTwoPassLoraPreset,
  miniMaxTwoPassLoraPresetButtons, miniMaxTwoPassLoraPresetField, miniMaxTwoPassLoraSection,
  miniMaxTwoPassLoraStatus, miniMaxTwoPassLoraStrength, miniMaxTwoPassOutputCrf, miniMaxTwoPassRefImageSize,
  miniMaxTwoPassResizeMethod, miniMaxTwoPassSettings, miniMaxTwoPassTeCacheDepth, miniMaxTwoPassTeDevice,
  miniMaxTwoPassTeEnd, miniMaxTwoPassTeMcs, miniMaxTwoPassTeProcessingControl, miniMaxTwoPassTeStart,
  miniMaxTwoPassUseFastVaeDecode, miniMaxUseLoras, miniMaxUseTurboLora, miniMaxVideoReferenceRows,
  miniMaxVideoReferencesButton, miniMaxVideoVaePicker, miniMaxWarmupFrames, normalizeLyricCueMapForSegment,
  playSceneAudioFrom, playSingerCueRange, prepareSceneAudioClipForTimestamping,
  previousAutoChainSourceSegment, projectInput, pushHistory, render, saveMiniMaxPromptButton,
  savedMiniMaxPrompts, sceneSlotNumber, segmentImageSource, segmentTrack, selectedPerformerSubjectsForSegment,
  selectedSegmentImagePath, singerCueRelativePlayheadTime, startSilentTimelinePlayback, state,
  storyboardReferenceDataForSegment, syncInspector, syncProjectVideoEngineUI, syncTimelineTrimModeButton,
  timelinePromptSave, twoPassControls, updatePlayPauseButton, useSceneMiniMaxH3Settings, videoSettingsSegment,
  wizardVideoSettings,
}) {
  const {
    ensureMiniMaxSpeakerAssignments, isMiniMaxBuiltInSpeakerAssignmentMode, isMiniMaxSingerAssignmentMode,
    miniMaxMappedSpeakersForSegment, renderMiniMaxSpeakerAssignmentPanel,
  } = createMiniMaxSpeakerAssignments({
    activateGlobalTimelineAudioPlayback, activeSegment, audio, autoSaveSessionQuiet,
    autoTimeMiniMaxSingerCuesForSegment, createProgressWindow, miniMaxAddSpeakerCueButton,
    miniMaxAutoTimeBeforePrompt, miniMaxH3SettingsForSegment, miniMaxSpeakerAssignmentList,
    miniMaxSpeakerAssignmentNote, normalizeLyricCueMapForSegment, playSceneAudioFrom, playSingerCueRange,
    prepareSceneAudioClipForTimestamping, pushHistory, render, selectedPerformerSubjectsForSegment,
    singerCueRelativePlayheadTime, startSilentTimelinePlayback, state, storyboardReferenceDataForSegment,
    syncInspector, updatePlayPauseButton,
  });

  function updateMiniMaxPromptCharacterStatus(segment = activeSegment()) {
    const length = String(miniMaxPrompt.value || "").length;
    const mode = segment ? miniMaxH3ModeForSegment(segment) : "text_to_video";
    const budget = segment ? miniMaxH3PromptCharacterBudget(segment, mode) : { fixedChars: 0, shotDescriptionChars: 6500 };
    const color = length > 7000 ? "#fca5a5" : length > 6500 ? "#fde68a" : "#86efac";
    const border = length > 7000 ? "#7f1d1d" : length > 6500 ? "#854d0e" : "#166534";
    miniMaxPromptCharacterStatus.style.color = color;
    miniMaxPromptCharacterStatus.style.borderColor = border;
    miniMaxPromptCharacterStatus.textContent = length
      ? `H3 prompt: ${length.toLocaleString()} / 7,000 characters. Fixed format: ${budget.fixedChars.toLocaleString()}${budget.refmodReserve ? ` (includes ${budget.refmodReserve.toLocaleString()} reserved for RefMod labels)` : ""}; planned shot allowance: ${budget.shotDescriptionChars.toLocaleString()}.`
      : `H3 prompt: 0 / 7,000 characters. Fixed format: ${budget.fixedChars.toLocaleString()}${budget.refmodReserve ? ` (includes ${budget.refmodReserve.toLocaleString()} reserved for RefMod labels)` : ""}; planned shot allowance: ${budget.shotDescriptionChars.toLocaleString()}.`;
    const tokenText = segment && typeof miniMaxH3ReferenceCapacityStatus === "function" ? miniMaxH3ReferenceCapacityStatus(segment, mode).tokenText : "";
    if (tokenText) {
      miniMaxPromptCharacterStatus.textContent += ` ${tokenText}`;
      if (/Over |lighter/.test(tokenText)) miniMaxPromptCharacterStatus.style.color = "#fde68a";
    }
    const referenceMismatch = segment ? miniMaxPromptReferenceMismatch(segment, miniMaxPrompt.value, mode) : "";
    if (referenceMismatch) {
      miniMaxPromptCharacterStatus.textContent += ` ${referenceMismatch} Regenerate this scene's MiniMax prompt before rendering.`;
      miniMaxPromptCharacterStatus.style.color = "#fde68a";
    }
  }

  async function toggleProjectVideoEngineFromBadge() {
    state.projectVideoEngine = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3"
      ? "ltx"
      : "minimax_h3";
    syncProjectVideoEngineUI();
    await autoSaveSessionQuiet("project video engine badge toggle");
    toast(state.projectVideoEngine === "minimax_h3"
      ? "Switched this project to MiniMax H3."
      : "Switched this project to LTX.");
  }

  function miniMaxH3ContinuityModeForSegment(segment = activeSegment()) {
    const mode = miniMaxH3ModeForSegment(segment);
    const continuityMode = normalizeMiniMaxH3ContinuityMode(miniMaxH3SettingsForSegment(segment).continuity_mode);
    return isMiniMaxH3ContinuityAllowedForMode(continuityMode, mode, miniMaxH3SettingsForSegment(segment).render_pass) ? continuityMode : "off";
  }

  function miniMaxH3SceneImageUseForSegment(segment = activeSegment()) {
    return normalizeMiniMaxH3SceneImageUse(
      segment?.minimax_h3_scene_image_use,
      Boolean(segment?.minimax_h3_use_scene_image_as_start_frame),
    );
  }

  function setMiniMaxH3ModeForSegment(segment, value) {
    const mode = normalizeMiniMaxH3Mode(value);
    const current = miniMaxH3SettingsForSegment(segment);
    const modeDefaults = mode === "reference_to_video" && current.video_mode === "image_to_video"
      ? {
        diffusion_model_name: current.diffusion_model_name === "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
          ? DEFAULT_MINIMAX_H3_SETTINGS.diffusion_model_name : current.diffusion_model_name,
        two_pass_lora_name: current.two_pass_lora_name === "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors"
          ? DEFAULT_MINIMAX_H3_SETTINGS.two_pass_lora_name : current.two_pass_lora_name,
      } : {};
    if (segment?.use_scene_minimax_h3_settings) {
      segment.minimax_h3_settings = cloneMiniMaxH3Settings({
        ...miniMaxH3SettingsForSegment(segment),
        video_mode: mode,
        ...modeDefaults,
      });
      segment.minimax_h3_mode = mode;
    } else {
      state.miniMaxH3Settings = cloneMiniMaxH3Settings({
        ...state.miniMaxH3Settings,
        video_mode: mode,
        ...modeDefaults,
      });
      if (segment) segment.minimax_h3_mode = mode;
    }
    return mode;
  }

  function setMiniMaxH3RenderPassForSegment(segment, value) {
    const renderPass = ["two_pass", "three_pass"].includes(value) ? value : "single";
    if (segment?.use_scene_minimax_h3_settings) {
      segment.minimax_h3_settings = cloneMiniMaxH3Settings({
        ...miniMaxH3SettingsForSegment(segment),
        render_pass: renderPass,
      });
    } else {
      state.miniMaxH3Settings = cloneMiniMaxH3Settings({
        ...state.miniMaxH3Settings,
        render_pass: renderPass,
      });
    }
    state.miniMaxH3TwoPassEnabled = renderPass === "two_pass";
    state.miniMaxH3ThreePassEnabled = renderPass === "three_pass";
    return renderPass;
  }

  function clearMiniMaxImageReferenceStartFrameOnModeSwitch(segment, targetMode) {
    if (normalizeMiniMaxH3Mode(targetMode) !== "reference_to_video") return;
    if (miniMaxH3ModeForSegment(segment) !== "image_reference_to_video") return;
    const sceneLocked = Boolean(segment?.use_scene_minimax_h3_settings);
    const targets = (sceneLocked ? [segment] : allEditableSegments()).filter((item) => (
      item
      && segmentTrack(item) !== "overlay"
      && !(!sceneLocked && item.use_scene_minimax_h3_settings)
      && miniMaxH3ModeForSegment(item) === "image_reference_to_video"
    ));
    for (const item of targets) {
      item.minimax_h3_scene_image_use = "off";
      item.minimax_h3_use_scene_image_as_start_frame = false;
    }
  }

  function saveMiniMaxSceneInputsFromPanel() {
    const segment = activeSegment();
    if (!segment) return;
    segment.minimax_h3_prompt = miniMaxPrompt.value || "";
    segment.minimax_h3_pass2_prompt = miniMaxPass2Prompt.value || "";
    segment.minimax_h3_continuation_direction = String(miniMaxContinuationDirection.value || "").trim();
    updateMiniMaxPromptCharacterStatus(segment);
    updateMiniMaxPromptSaveButtonState();
    segment.minimax_h3_scene_image_use = normalizeMiniMaxH3SceneImageUse(miniMaxSceneImageUse.value);
    segment.minimax_h3_use_scene_image_as_start_frame = segment.minimax_h3_scene_image_use === "exact_start_frame";
    segment.minimax_h3_start_frame_character_influence = normalizeMiniMaxH3StartFrameCharacterInfluence(
      miniMaxStartFrameCharacterInfluence.value,
    );
    segment.minimax_h3_video_references = miniMaxVideoReferenceRows
      .map((row) => ({
        path: String(row.path.value || "").trim(),
        start_seconds: Math.max(0, Number(row.start.value || 0)),
        duration: Math.max(0, Number(row.duration.value || 0)),
        purpose: normalizeMiniMaxH3VideoPurpose(row.purpose.value),
        use_audio: Boolean(row.useAudio.input.checked),
      }))
      .filter((item) => item.path)
      .slice(0, 3);
  }

  let miniMaxLatentCheckCounter = 0;
  async function updateMiniMaxLatentPredecessorStatus(segment) {
    const checkId = ++miniMaxLatentCheckCounter;
    if (!segment) {
      miniMaxLatentStatusPill.style.display = "none";
      return;
    }
    const continuityMode = normalizeMiniMaxH3ContinuityMode(miniMaxContinuityMode.value);
    if (!isMiniMaxH3LatentContinuationMode(continuityMode)) {
      miniMaxLatentStatusPill.style.display = "none";
      return;
    }
    miniMaxLatentStatusPill.style.display = "inline-flex";
    const slotNumber = sceneSlotNumber(segment);
    if (slotNumber <= 1) {
      miniMaxLatentStatusPill.textContent = "Scene 1 has no predecessor — starts fresh and saves latent on render";
      miniMaxLatentStatusPill.style.borderColor = "#0284c7";
      miniMaxLatentStatusPill.style.background = "#0c4a6e";
      miniMaxLatentStatusPill.style.color = "#38bdf8";
      return;
    }
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    if (!projectFolder) {
      miniMaxLatentStatusPill.textContent = "Set project folder to check predecessor latent";
      miniMaxLatentStatusPill.style.borderColor = "#334155";
      miniMaxLatentStatusPill.style.background = "#0f172a";
      miniMaxLatentStatusPill.style.color = "#94a3b8";
      return;
    }
    miniMaxLatentStatusPill.textContent = `Checking Scene ${slotNumber - 1} latent status...`;
    miniMaxLatentStatusPill.style.borderColor = "#334155";
    miniMaxLatentStatusPill.style.background = "#0f172a";
    miniMaxLatentStatusPill.style.color = "#94a3b8";
    try {
      const resp = await postJson("/vrgdg/music_builder/check_latent_predecessor", {
        project_folder: projectFolder,
        scene_number: slotNumber,
      }, 5000);
      if (checkId !== miniMaxLatentCheckCounter) return;
      if (resp?.predecessor_exists) {
        const dirtyNote = resp.dirty ? " (marked dirty)" : "";
        // Masked continuation needs the predecessor's tail-padding info to cut its window before any padding.
        const needsRerender = !resp.tail_padding_known;
        const rerenderNote = needsRerender ? " — no tail info, re-render it for an exact seam" : "";
        miniMaxLatentStatusPill.textContent = `Predecessor Scene ${resp.predecessor_scene} latent ready (${resp.frame_count} frames, ${resp.token_count} tokens)${dirtyNote}${rerenderNote}`;
        const warn = resp.dirty || needsRerender;
        miniMaxLatentStatusPill.style.borderColor = warn ? "#b45309" : "#166534";
        miniMaxLatentStatusPill.style.background = warn ? "#451a03" : "#052e16";
        miniMaxLatentStatusPill.style.color = warn ? "#fbbf24" : "#4ade80";
      } else {
        miniMaxLatentStatusPill.textContent = `Scene ${resp.predecessor_scene} latent missing — render Scene ${resp.predecessor_scene} first`;
        miniMaxLatentStatusPill.style.borderColor = "#991b1b";
        miniMaxLatentStatusPill.style.background = "#450a0a";
        miniMaxLatentStatusPill.style.color = "#f87171";
      }
    } catch (e) {
      if (checkId !== miniMaxLatentCheckCounter) return;
      miniMaxLatentStatusPill.textContent = `Could not verify Scene ${slotNumber - 1} latent`;
      miniMaxLatentStatusPill.style.borderColor = "#475569";
      miniMaxLatentStatusPill.style.background = "#1e293b";
      miniMaxLatentStatusPill.style.color = "#cbd5e1";
    }
  }

  function saveMiniMaxH3SettingsFromPanel(targetSegment = null) {
    // DOM event listeners pass their Event as the first argument. Only treat
    // an actual scene object as an explicit target.
    const segment = wizardVideoSettings.global ? null : (targetSegment?.id ? targetSegment : activeSegment());
    const currentSettings = miniMaxH3SettingsForSegment(segment);
    const loraEnabled = Boolean(miniMaxUseLoras.input.checked);
    const turboEnabled = currentSettings.video_mode !== "reference_to_video" && Boolean(miniMaxUseTurboLora.input.checked) && !loraEnabled;
    const loraCount = loraEnabled ? Math.max(0, Math.min(4, Math.trunc(Number(miniMaxLoraCount.value || 0)))) : 0;
    const loras = miniMaxLoraSlots
      .slice(0, loraCount)
      .map((slot) => ({
        name: String(slot.picker.input.value || "").trim(),
        strength: Math.max(-10, Math.min(10, Number(slot.strength.value || 1))),
        apply_to: slot.applyTo.value,
      }))
      .filter((item) => item.name && item.name !== "[none]");
    const settings = cloneMiniMaxH3Settings({
      ...currentSettings,
      video_mode: currentSettings.video_mode,
      render_pass: currentSettings.render_pass,
      audio_mode: miniMaxAudioMode.value,
      continuity_mode: miniMaxContinuityMode.value,
      continuity_prompt_from_last_frame: miniMaxContinuityPromptFromLastFrame.input.checked,
      i2v_transition_style: miniMaxKeyframes.transitionStyle.value,
      i2v_transition_direction: miniMaxKeyframes.transitionDirection.value,
      location_transition_preset: miniMaxLocationTransitionPreset.value,
      location_transition_custom: miniMaxLocationTransitionCustom.value,
      latent_context_frames: Number(miniMaxLatentContextFrames.value || 39),
      diffusion_model_name: miniMaxDiffusionModelPicker.input.value,
      clip_name: miniMaxClipPicker.input.value,
      video_vae_name: miniMaxVideoVaePicker.input.value,
      audio_vae_name: miniMaxAudioVaePicker.input.value,
      aspect_ratio: miniMaxAspectRatio.value,
      resolution_preset: miniMaxResolutionPreset.value,
      megapixels: miniMaxMegapixels.value,
      seed: miniMaxSeed.value,
      warmup_frames: miniMaxWarmupFrames.value,
      cooldown_frames: miniMaxCooldownFrames.value,
      sampler_name: miniMaxSamplerName.value,
      scheduler: miniMaxScheduler.value,
      steps: miniMaxSteps.value,
      steps_before_turbo: turboEnabled ? currentSettings.steps_before_turbo : miniMaxSteps.value,
      denoise: miniMaxDenoise.value,
      ref_image_size: currentSettings.render_pass === "three_pass"
        ? miniMaxThreePassRefImageSize.value
        : currentSettings.render_pass === "two_pass"
          ? miniMaxTwoPassRefImageSize.value
          : miniMaxRefImageSize.value,
      two_pass_lora_name: miniMaxTwoPassLoraPicker.input.value,
      two_pass_lora_strength: miniMaxTwoPassLoraStrength.value,
      two_pass_lora_preset: Number(miniMaxTwoPassLoraPreset.dataset.preset || 4),
      two_pass_defaults_version: DEFAULT_MINIMAX_H3_SETTINGS.two_pass_defaults_version,
      two_pass_latent_upscale_scale: miniMaxTwoPassLatentScale.value,
      two_pass_latent_upscaler_name: currentSettings.render_pass === "three_pass"
        ? miniMaxAdvancedLatentUpscalerPicker.input.value
        : miniMaxTwoPassLatentUpscalerPicker.input.value,
      ...Object.fromEntries(miniMaxAccelerationControls.flatMap(({ key, single, pass1, pass2 }) => [
        [`use_${key}`, single.input.checked],
        [`pass1_use_${key}`, pass1.input.checked],
        [`pass2_use_${key}`, pass2.input.checked],
      ])),
      use_fast_vae_decode: currentSettings.render_pass !== "single"
        ? currentSettings.use_fast_vae_decode : miniMaxTwoPassUseFastVaeDecode.input.checked,
      two_pass_use_fast_vae_decode: currentSettings.render_pass !== "single"
        ? miniMaxTwoPassUseFastVaeDecode.input.checked : currentSettings.two_pass_use_fast_vae_decode,
      two_pass_te_speed_processing_control: miniMaxTwoPassTeProcessingControl.value,
      two_pass_te_speed_start_percent: miniMaxTwoPassTeStart.value,
      two_pass_te_speed_end_percent: miniMaxTwoPassTeEnd.value,
      two_pass_te_speed_mcs: miniMaxTwoPassTeMcs.value,
      two_pass_te_speed_cache_depth: miniMaxTwoPassTeCacheDepth.value,
      two_pass_te_speed_device: miniMaxTwoPassTeDevice.value,
      two_pass_final_resize_method: miniMaxTwoPassResizeMethod.value,
      two_pass_output_crf: miniMaxTwoPassOutputCrf.value,
      three_pass_lightx_lora_name: miniMaxThreePassLoraPicker.input.value,
      three_pass_lightx_lora_strength: miniMaxThreePassLoraStrength.value,
      advanced_two_pass_vram_preset: miniMaxAdvancedVramPreset.value,
      advanced_two_pass_defaults_version: DEFAULT_MINIMAX_H3_SETTINGS.advanced_two_pass_defaults_version,
      ...Object.fromEntries(twoPassControls.flatMap((control) => [
        [`${control.prefix}steps`, control.steps.value],
        [`${control.prefix}denoise`, control.denoise.value],
        [`${control.prefix}sampler`, control.sampler.value],
        [`${control.prefix}scheduler`, control.scheduler.value],
        [`${control.prefix}seed`, control.seed.value],
      ])),
      ...Object.fromEntries(advancedTwoPassControls.flatMap((control) => [
        // Only Pass 1 has its own resolution; Pass 2 uses the shared output resolution.
        ...(control.megapixels ? [
          [`${control.prefix}resolution_preset`, control.resolutionPreset.value],
          [`${control.prefix}megapixels`, control.megapixels.value],
        ] : []),
        [`${control.prefix}steps`, control.steps.value],
        [`${control.prefix}denoise`, control.denoise.value],
        [`${control.prefix}sampler`, control.sampler.value],
        [`${control.prefix}scheduler`, control.scheduler.value],
        [`${control.prefix}seed`, control.seed.value],
      ])),
      easy_cache_bypass: miniMaxEasyCacheBypass.input.checked,
      easy_cache_bypass_before_turbo: turboEnabled
        ? currentSettings.easy_cache_bypass_before_turbo
        : miniMaxEasyCacheBypass.input.checked,
      easy_cache_reuse_threshold: miniMaxEasyCacheReuseThreshold.value,
      easy_cache_start_percent: miniMaxEasyCacheStartPercent.value,
      easy_cache_end_percent: miniMaxEasyCacheEndPercent.value,
      easy_cache_verbose: miniMaxEasyCacheVerbose.input.checked,
      sage_attention: miniMaxSageAttention.value,
      use_memory_efficient_sage_attention: miniMaxMemoryEfficientSageAttention.input.checked,
      enable_fp16_accumulation: miniMaxFp16Accumulation.input.checked,
      use_loras: loraEnabled,
      lora_count: loraCount,
      loras,
      use_turbo_lora: turboEnabled,
      turbo_lora_name: miniMaxTurboLoraPicker.input.value,
      turbo_lora_strength: miniMaxTurboLoraStrength.value,
    });
    if (segment?.use_scene_minimax_h3_settings) {
      segment.minimax_h3_settings = settings;
      segment.minimax_h3_mode = settings.video_mode;
    } else {
      state.miniMaxH3Settings = settings;
      if (segment) segment.minimax_h3_mode = settings.video_mode;
    }
    return settings;
  }

  function updateMiniMaxPromptSaveButtonState() {
    const segment = activeSegment();
    if (!segment) {
      saveMiniMaxPromptButton.disabled = true;
      saveMiniMaxPromptButton.style.opacity = "0.5";
      saveMiniMaxPromptButton.style.cursor = "not-allowed";
      return;
    }
    const saved = String(savedMiniMaxPrompts.get(segment) ?? segment.minimax_h3_prompt ?? "");
    const current = String(miniMaxPrompt.value || "");
    const isDirty = !timelinePromptSave.saving && current !== saved;
    saveMiniMaxPromptButton.disabled = !isDirty;
    saveMiniMaxPromptButton.style.opacity = isDirty ? "1" : "0.5";
    saveMiniMaxPromptButton.style.cursor = isDirty ? "pointer" : "not-allowed";
  }

  function miniMaxH3SettingsForSegment(segment = videoSettingsSegment()) {
    const globalSettings = cloneMiniMaxH3Settings(state.miniMaxH3Settings);
    if (!segment?.use_scene_minimax_h3_settings) return globalSettings;
    const sceneSettings = segment.minimax_h3_settings && typeof segment.minimax_h3_settings === "object"
      ? segment.minimax_h3_settings
      : {};
    const legacyPreset = normalizeMiniMaxH3LocationTransitionPreset(segment.minimax_h3_location_transition_preset);
    const legacyCustom = String(segment.minimax_h3_location_transition_custom || "").trim();
    const hasSceneTransitionPreset = Object.prototype.hasOwnProperty.call(sceneSettings, "location_transition_preset");
    const settings = cloneMiniMaxH3Settings({
      ...globalSettings,
      ...sceneSettings,
      // The pipeline belongs to the whole project, so a scene with its own settings follows it.
      pipeline: globalSettings.pipeline,
      video_mode: sceneSettings.video_mode || segment.minimax_h3_mode || globalSettings.video_mode,
      render_pass: sceneSettings.render_pass ?? (normalizeMiniMaxH3Mode(sceneSettings.video_mode || segment.minimax_h3_mode) === "image_reference_to_video" ? "two_pass" : globalSettings.render_pass),
      location_transition_preset: hasSceneTransitionPreset ? sceneSettings.location_transition_preset : legacyPreset,
      location_transition_custom: hasSceneTransitionPreset ? sceneSettings.location_transition_custom : legacyCustom,
    });
    const sceneHasPreTurboEasyCache = sceneSettings.easy_cache_bypass_before_turbo != null
      || sceneSettings.easyCacheBypassBeforeTurbo != null;
    if (settings.use_turbo_lora && !sceneHasPreTurboEasyCache) {
      settings.easy_cache_bypass_before_turbo = Boolean(sceneSettings.easy_cache_bypass ?? globalSettings.easy_cache_bypass);
      settings.easy_cache_bypass = true;
    }
    segment.minimax_h3_settings = settings;
    segment.minimax_h3_mode = settings.video_mode;
    if (!hasSceneTransitionPreset && (legacyPreset !== "normal" || legacyCustom)) {
      segment.minimax_h3_location_transition_preset = "normal";
      segment.minimax_h3_location_transition_custom = "";
    }
    return settings;
  }

  function miniMaxH3ModeForSegment(segment = activeSegment()) {
    return miniMaxH3SettingsForSegment(segment).video_mode;
  }

  // No continuity mode adds a reference image any more: Latent Continuation Masked works on the latent, and the
  // previous-final-frame modes were retired. So no reference slot is held back for a continuity frame.
  function miniMaxH3ContinuityReferenceReserved() {
    return false;
  }

  function miniMaxH3StartFrameCharacterInfluenceForSegment(segment = activeSegment()) {
    return normalizeMiniMaxH3StartFrameCharacterInfluence(
      segment?.minimax_h3_start_frame_character_influence,
    );
  }

  function miniMaxH3SceneImageIsPromptInspiration(segment = activeSegment()) {
    return ["environment_inspiration", "environment_framing_inspiration"].includes(
      miniMaxH3SceneImageUseForSegment(segment),
    );
  }

  function syncMiniMaxReferenceButtons() {
    const segment = activeSegment();
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    if (!miniMaxProject) {
      for (const button of miniMaxReferenceButtons) {
        button.textContent = button === miniMaxVideoReferencesButton
          ? "Choose MiniMax Edit References (0/9)"
          : "Choose MiniMax References (0/9)";
        button.disabled = true;
      }
      return;
    }
    const libraryCount = segment && typeof miniMaxDesiredReferenceKeysForSegment === "function"
      ? miniMaxDesiredReferenceKeysForSegment(segment).length
      : 0;
    const effectiveMode = miniMaxH3ModeForSegment(segment);
    const orderedCount = segment && typeof miniMaxOrderedImageReferenceItemsForSegment === "function"
      ? miniMaxOrderedImageReferenceItemsForSegment(segment, effectiveMode).length
      : libraryCount;
    const hasStartFrame = Boolean(
      segment?.minimax_h3_use_scene_image_as_start_frame
      && effectiveMode === "reference_to_video"
      && (segmentImageSource(segment)?.path || segmentImageSource(segment)?.data)
    );
    const hasContinuityReservation = miniMaxH3ContinuityReferenceReserved(segment);
    const capacity = segment ? miniMaxH3ReferenceCapacityStatus(segment, effectiveMode) : { count: 0, overflow: 0 };
    for (const button of miniMaxReferenceButtons) {
      button.textContent = button === miniMaxVideoReferencesButton
        ? `Choose MiniMax Edit References (${capacity.count}/9${capacity.overflow ? " — TOO MANY" : ""})`
        : hasStartFrame
          ? `Choose MiniMax References (${capacity.count}/9${capacity.overflow ? " — TOO MANY" : ` — start + ${Math.max(0, orderedCount - 1)}`})`
          : `Choose MiniMax References (${capacity.count || libraryCount}/9${capacity.overflow ? " — TOO MANY" : ""})`;
      button.style.borderColor = capacity.overflow ? "#ef4444" : "";
      button.disabled = !segment;
    }
  }

  // Scene clothing: pick what each character wears in this scene, or go back to the card defaults.
  function renderRefmodClothing(container, segment) {
    if (!container) return;
    const rows = segment && typeof miniMaxH3ReferenceCapacityStatus === "function"
      ? miniMaxH3ReferenceCapacityStatus(segment, "reference_to_video").clothing || []
      : [];
    container.replaceChildren();
    container.style.display = rows.length ? "flex" : "none";
    if (!rows.length) return;
    const title = document.createElement("div");
    title.textContent = "Clothing in this scene";
    title.style.cssText = "font-size:11px;font-weight:700;color:#a5f3fc;";
    container.append(title);
    for (const row of rows) {
      const label = document.createElement("label");
      label.style.cssText = "display:grid;grid-template-columns:minmax(80px,1fr) minmax(120px,2fr);gap:8px;align-items:center;font-size:11px;color:#e2e8f0;";
      label.append(document.createTextNode(row.character));
      const select = document.createElement("select");
      select.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:5px;font-size:11px;min-width:0;";
      select.append(new Option("Card default (follows the character)", "__default__"));
      select.append(new Option("No clothing RefMod", ""));
      for (const option of row.options) select.append(new Option(option.name, option.id));
      select.value = row.overridden ? row.current : "__default__";
      select.onchange = async () => {
        const next = { ...(segment.refmod_clothing_override || {}) };
        if (select.value === "__default__") delete next[row.character_id];
        else next[row.character_id] = select.value;
        pushHistory?.();
        if (Object.keys(next).length) segment.refmod_clothing_override = next;
        else delete segment.refmod_clothing_override;
        await autoSaveSessionQuiet("scene RefMod clothing");
        syncMiniMaxH3Panel();
        updateMiniMaxPromptCharacterStatus(segment);
        toast("Clothing changed. Regenerate this scene's prompt so it names the new clothing.");
      };
      label.append(select);
      container.append(label);
    }
  }

  // The "Direction starts at" slider: from 0.5 s to half of this scene, on while the direction box is on.
  function syncMiniMaxContinuationStart(segment, enabled) {
    const sceneSeconds = segment ? Math.max(0, Number(segment.end || 0) - Number(segment.start || 0)) : 0;
    const { low, high } = miniMaxH3ContinuationStartLimits(sceneSeconds);
    const start = miniMaxH3ContinuationStartSeconds(sceneSeconds, segment?.minimax_h3_continuation_start_seconds);
    miniMaxContinuationStart.min = String(low);
    miniMaxContinuationStart.max = String(high);
    miniMaxContinuationStart.value = String(start);
    const fixed = high <= low;
    miniMaxContinuationStart.disabled = !enabled || fixed;
    miniMaxContinuationStartField.style.opacity = enabled ? "1" : ".55";
    miniMaxContinuationStartValue.textContent = !segment
      ? ""
      : `${start.toFixed(1)} s into the scene. ${fixed ? "This scene is too short to move it." : `Up to ${high.toFixed(1)} s (half of this ${sceneSeconds.toFixed(2)} s scene).`} ${Math.max(0, sceneSeconds - start).toFixed(1)} s are left for the direction.`;
  }

  function syncMiniMaxH3Panel() {
    state.syncLlmPopout?.();
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    const segment = videoSettingsSegment();
    state.miniMaxH3PanelSegmentId = String(segment?.id || "");
    state.miniMaxH3Settings = cloneMiniMaxH3Settings(state.miniMaxH3Settings);
    if (state.miniMaxH3Settings.location_transition_preset === "normal") {
      const legacyTransitionSegment = [segment, ...allEditableSegments()]
        .filter((item, index, items) => item && items.indexOf(item) === index && !item.use_scene_minimax_h3_settings)
        .find((item) => normalizeMiniMaxH3LocationTransitionPreset(item.minimax_h3_location_transition_preset) !== "normal");
      if (legacyTransitionSegment) {
        state.miniMaxH3Settings = cloneMiniMaxH3Settings({
          ...state.miniMaxH3Settings,
          location_transition_preset: legacyTransitionSegment.minimax_h3_location_transition_preset,
          location_transition_custom: legacyTransitionSegment.minimax_h3_location_transition_custom,
        });
        for (const item of allEditableSegments()) {
          if (item.use_scene_minimax_h3_settings) continue;
          item.minimax_h3_location_transition_preset = "normal";
          item.minimax_h3_location_transition_custom = "";
        }
      }
    }
    const settings = miniMaxH3SettingsForSegment(segment);
    state.miniMaxH3TwoPassEnabled = settings.render_pass === "two_pass";
    state.miniMaxH3ThreePassEnabled = settings.render_pass === "three_pass";
    miniMaxDiffusionModelPicker.input.value = settings.diffusion_model_name;
    miniMaxClipPicker.input.value = settings.clip_name;
    miniMaxVideoVaePicker.input.value = settings.video_vae_name;
    miniMaxAudioVaePicker.input.value = settings.audio_vae_name;
    miniMaxAudioMode.value = settings.audio_mode;
    miniMaxContinuityMode.value = settings.continuity_mode;
    miniMaxContinuityPromptFromLastFrame.input.checked = Boolean(settings.continuity_prompt_from_last_frame);
    miniMaxLocationTransitionPreset.value = settings.location_transition_preset;
    miniMaxLocationTransitionCustom.value = settings.location_transition_custom;
    miniMaxLatentContextFrames.value = String(settings.latent_context_frames);
    miniMaxAspectRatio.value = settings.aspect_ratio;
    miniMaxResolutionPreset.value = settings.resolution_preset;
    miniMaxMegapixels.value = String(settings.megapixels);
    miniMaxResolutionPreset.syncResolution();
    miniMaxSeed.value = String(settings.seed);
    miniMaxWarmupFrames.value = String(settings.warmup_frames);
    miniMaxCooldownFrames.value = String(settings.cooldown_frames);
    miniMaxSamplerName.value = settings.sampler_name;
    miniMaxScheduler.value = settings.scheduler;
    miniMaxSteps.value = String(settings.steps);
    miniMaxDenoise.value = String(settings.denoise);
    miniMaxRefImageSize.value = settings.ref_image_size;
    miniMaxTwoPassRefImageSize.value = settings.ref_image_size;
    miniMaxThreePassRefImageSize.value = settings.ref_image_size;
    miniMaxTwoPassLoraPicker.input.value = settings.two_pass_lora_name;
    miniMaxTwoPassLoraStrength.value = String(settings.two_pass_lora_strength);
    miniMaxTwoPassLoraStatus.textContent = "";
    miniMaxTwoPassLoraPreset.dataset.preset = String(settings.two_pass_lora_preset);
    const showTwoPassLoraPreset = state.miniMaxH3TwoPassEnabled && !state.miniMaxH3ThreePassEnabled;
    miniMaxTwoPassLoraPresetField.style.display = showTwoPassLoraPreset ? "" : "none";
    miniMaxTwoPassLoraLayout.style.gridTemplateColumns = showTwoPassLoraPreset ? "minmax(0,1fr) auto" : "minmax(0,1fr)";
    for (const button of miniMaxTwoPassLoraPresetButtons) {
      const active = Number(button.dataset.preset) === settings.two_pass_lora_preset;
      button.setAttribute("aria-pressed", String(active));
      button.style.background = active ? "#06b6d4" : "#27272a";
      button.style.color = active ? "#082f49" : "#f4f4f5";
    }
    miniMaxTwoPassLatentScale.value = String(settings.two_pass_latent_upscale_scale);
    miniMaxTwoPassLatentUpscalerPicker.input.value = settings.two_pass_latent_upscaler_name;
    miniMaxAdvancedLatentUpscalerPicker.input.value = settings.two_pass_latent_upscaler_name;
    const accelerationMultiPass = state.miniMaxH3TwoPassEnabled || state.miniMaxH3ThreePassEnabled;
    for (const { key, single, pass1, pass2 } of miniMaxAccelerationControls) {
      single.input.checked = Boolean(settings[`use_${key}`]);
      single.input.style.display = accelerationMultiPass ? "none" : "";
      single.input.disabled = accelerationMultiPass;
      pass1.wrapper.style.display = accelerationMultiPass ? "" : "none";
      pass2.wrapper.style.display = accelerationMultiPass ? "" : "none";
      pass1.input.checked = Boolean(settings[`pass1_use_${key}`] ?? settings[`two_pass_use_${key}`]);
      pass2.input.checked = Boolean(settings[`pass2_use_${key}`] ?? settings[`two_pass_use_${key}`]);
    }
    miniMaxTwoPassUseFastVaeDecode.input.checked = Boolean(accelerationMultiPass
      ? settings.two_pass_use_fast_vae_decode : settings.use_fast_vae_decode);
    miniMaxTwoPassTeProcessingControl.value = String(settings.two_pass_te_speed_processing_control);
    miniMaxTwoPassTeStart.value = String(settings.two_pass_te_speed_start_percent);
    miniMaxTwoPassTeEnd.value = String(settings.two_pass_te_speed_end_percent);
    miniMaxTwoPassTeMcs.value = String(settings.two_pass_te_speed_mcs);
    miniMaxTwoPassTeCacheDepth.value = String(settings.two_pass_te_speed_cache_depth);
    miniMaxTwoPassTeDevice.value = settings.two_pass_te_speed_device;
    miniMaxTwoPassResizeMethod.value = settings.two_pass_final_resize_method;
    miniMaxTwoPassOutputCrf.value = String(settings.two_pass_output_crf);
    miniMaxThreePassLoraPicker.input.value = settings.three_pass_lightx_lora_name;
    miniMaxThreePassLoraStrength.value = String(settings.three_pass_lightx_lora_strength);
    miniMaxAdvancedVramPreset.value = settings.advanced_two_pass_vram_preset;
    twoPassControls.forEach((control) => {
      control.steps.value = String(settings[`${control.prefix}steps`]);
      control.denoise.value = String(settings[`${control.prefix}denoise`]);
      control.sampler.value = settings[`${control.prefix}sampler`];
      control.scheduler.value = settings[`${control.prefix}scheduler`];
      control.seed.value = String(settings[`${control.prefix}seed`]);
    });
    advancedTwoPassControls.forEach((control) => {
      if (control.megapixels) {
        control.resolutionPreset.value = settings[`${control.prefix}resolution_preset`] || "custom";
        control.megapixels.value = String(settings[`${control.prefix}megapixels`]);
        control.syncResolutionPreset();
      }
      control.steps.value = String(settings[`${control.prefix}steps`]);
      control.denoise.value = String(settings[`${control.prefix}denoise`]);
      control.sampler.value = settings[`${control.prefix}sampler`];
      control.scheduler.value = settings[`${control.prefix}scheduler`];
      control.seed.value = String(settings[`${control.prefix}seed`]);
    });
    const multiPassMode = state.miniMaxH3TwoPassEnabled || state.miniMaxH3ThreePassEnabled;
    miniMaxSeedField.style.display = multiPassMode ? "none" : "";
    // Advanced Settings stays visible in multi-pass so Sage Attention and fp16 accumulation can be changed. The sampler,
    // EasyCache and reference parts of it hide themselves for multi-pass in the mode update below.
    miniMaxAdvancedSettings.style.display = "";
    miniMaxTwoPassSettings.style.display = state.miniMaxH3TwoPassEnabled ? "" : "none";
    miniMaxThreePassSettings.style.display = state.miniMaxH3ThreePassEnabled ? "" : "none";
    // Pass 1 resolution lives in Render Settings and applies to 2 Pass Advanced only.
    advancedTwoPassControls[0].resolutionField.style.display = state.miniMaxH3ThreePassEnabled ? "contents" : "none";
    miniMaxTwoPassLoraSection.style.display = (state.miniMaxH3TwoPassEnabled || state.miniMaxH3ThreePassEnabled) ? "" : "none";
    miniMaxThreePassLoraSection.style.display = "none";
    miniMaxEasyCacheBypass.input.checked = settings.easy_cache_bypass;
    miniMaxEasyCacheReuseThreshold.value = String(settings.easy_cache_reuse_threshold);
    miniMaxEasyCacheStartPercent.value = String(settings.easy_cache_start_percent);
    miniMaxEasyCacheEndPercent.value = String(settings.easy_cache_end_percent);
    miniMaxEasyCacheVerbose.input.checked = settings.easy_cache_verbose;
    miniMaxSageAttention.value = settings.sage_attention;
    miniMaxMemoryEfficientSageAttention.input.checked = settings.use_memory_efficient_sage_attention;
    miniMaxFp16Accumulation.input.checked = settings.enable_fp16_accumulation;
    miniMaxUseLoras.input.checked = settings.use_loras;
    miniMaxLoraCount.value = String(settings.lora_count || 0);
    miniMaxLoraRows.style.display = settings.use_loras && Number(settings.lora_count || 0) > 0 ? "flex" : "none";
    miniMaxLoraSlots.forEach((slot, index) => {
      const item = settings.loras?.[index] || {};
      slot.row.style.display = settings.use_loras && index < Number(settings.lora_count || 0) ? "grid" : "none";
      slot.picker.input.value = item.name || slot.picker.input.value || "[none]";
      slot.strength.value = String(item.strength ?? slot.strength.value ?? 1);
      slot.applyTo.value = ["both", "pass1", "pass2"].includes(item.apply_to) ? item.apply_to : "pass1";
    });
    miniMaxUseTurboLora.input.checked = settings.use_turbo_lora;
    miniMaxTurboLoraPicker.input.value = settings.turbo_lora_name;
    miniMaxTurboLoraStrength.value = String(settings.turbo_lora_strength);
    miniMaxUseTurboLora.input.disabled = settings.use_loras;
    miniMaxTurboSection.style.opacity = settings.use_loras ? ".55" : "";
    miniMaxUseLoras.input.disabled = settings.use_turbo_lora;
    miniMaxLoraSection.style.opacity = settings.use_turbo_lora ? ".55" : "";
    miniMaxLoraCount.disabled = !settings.use_loras || settings.use_turbo_lora;
    miniMaxLoraSlots.forEach((slot) => {
      slot.picker.input.disabled = !settings.use_loras || settings.use_turbo_lora;
      slot.strength.disabled = !settings.use_loras || settings.use_turbo_lora;
    });
    miniMaxTurboLoraField.style.display = settings.use_turbo_lora ? "flex" : "none";
    miniMaxTurboLoraStrengthField.style.display = settings.use_turbo_lora ? "flex" : "none";
    miniMaxLoraNote.textContent = settings.use_loras
      ? "Normal MiniMax LoRAs are ON. Turbo acceleration is disabled while this stack is active."
      : settings.use_turbo_lora
        ? "Normal MiniMax LoRAs are disabled while Turbo acceleration is active."
        : "Optional normal MiniMax LoRAs are OFF.";
    miniMaxTurboNote.textContent = settings.use_turbo_lora
        ? "Turbo LoRA is ON. The selected LoRA is applied with ComfyUI's built-in LoRA loader. No separate Turbo custom-node repository is required."
      : settings.use_loras
        ? "Turbo is unavailable while normal MiniMax LoRAs are enabled."
        : "Turbo is OFF. The normal MiniMax sampler, scheduler, and step settings are used.";
    miniMaxSamplerName.disabled = settings.use_turbo_lora;
    miniMaxScheduler.disabled = settings.use_turbo_lora;
    miniMaxSteps.disabled = false;
    miniMaxSteps.min = "1";
    useSceneMiniMaxH3Settings.input.checked = Boolean(segment?.use_scene_minimax_h3_settings);
    useSceneMiniMaxH3Settings.input.disabled = !segment;
    const sceneMiniMaxSettingsLocked = Boolean(segment?.use_scene_minimax_h3_settings);
    miniMaxSettingsScopeNote.textContent = wizardVideoSettings.global ? "Editing project-wide MiniMax models, LoRAs and video settings." : sceneMiniMaxSettingsLocked
      ? "Scene lock is ON — this scene uses its own MiniMax mode, models, and video settings."
      : "Scene lock is OFF — this scene follows the project-global MiniMax mode, models, and video settings.";
    miniMaxSceneImageUseField.firstElementChild.textContent = sceneMiniMaxSettingsLocked
      ? "Scene image use — this locked scene"
      : "Scene image use — all unlocked Image/Reference-to-Video scenes";
    miniMaxSceneImageUseField.title = sceneMiniMaxSettingsLocked
      ? "Changes only this locked scene. Prompt-only inspiration is sent to the vision LLM but not to MiniMax."
      : "Applies across every eligible unlocked base scene. Locked scenes remain unchanged; prompt-only inspiration is never sent to MiniMax.";
    miniMaxStartFrameCharacterInfluenceField.firstElementChild.textContent = sceneMiniMaxSettingsLocked
      ? "Character reference influence — this locked scene"
      : "Character reference influence — all unlocked Image/Reference-to-Video scenes";
    miniMaxStartFrameCharacterInfluenceField.title = sceneMiniMaxSettingsLocked
      ? "Changes only this locked scene's character-reference priority."
      : "Applies this priority across eligible unlocked base scenes; locked scenes remain unchanged.";
    const mode = settings.video_mode;
    const modeLabel = miniMaxH3ModeLabel(mode);
    const imageReferenceTwoPass = mode === "image_reference_to_video";
    const twoPass = Boolean(state.miniMaxH3TwoPassEnabled)
      && ["reference_to_video", "image_reference_to_video", "image_to_video"].includes(mode);
    const threePass = Boolean(state.miniMaxH3ThreePassEnabled)
      && ["reference_to_video", "image_reference_to_video"].includes(mode);
    const hideMultiPassIgnoredSettings = twoPass || threePass;
    miniMaxLoraSlots.forEach((slot) => {
      slot.applyToField.style.display = (twoPass || threePass) ? "flex" : "none";
      slot.row.style.gridTemplateColumns = (twoPass || threePass)
        ? "minmax(0,1fr) 92px minmax(120px,0.45fr)"
        : "minmax(0,1fr) 92px";
    });
    if (mode === "reference_to_video" && !twoPass && !threePass) {
      miniMaxLoraNote.textContent = "Choose optional MiniMax LoRAs here, including Turbo LoRAs. Set the sampler and steps in Advanced Settings.";
    }
    if (twoPass || threePass) {
      miniMaxLoraNote.textContent = settings.use_loras
        ? "Extra LoRAs are ON. Each can target pass 1, pass 2, or both. The required Turbo LoRA remains separate and pass-2-only."
        : "Optional extra LoRAs are OFF. Enable them, choose a count, then select the target pass for each LoRA.";
    }
    miniMaxSeedField.style.display = hideMultiPassIgnoredSettings ? "none" : "";
    miniMaxSamplerSettings.style.display = hideMultiPassIgnoredSettings ? "none" : "";
    miniMaxEasyCacheSettings.style.display = hideMultiPassIgnoredSettings || mode === "reference_to_video" ? "none" : "";
    // Sage Attention and fp16 accumulation apply to every pass layout. The memory-efficient patch is single pass only.
    miniMaxModelLoaderSettings.style.display = "";
    miniMaxMemoryEfficientSageAttention.wrapper.style.display = hideMultiPassIgnoredSettings ? "none" : "";
    miniMaxReferenceConditioningSettings.style.display = !hideMultiPassIgnoredSettings
      && ["reference_to_video", "video_to_video"].includes(mode)
      ? ""
      : "none";
    // Multi-pass workflows have their own optional normal-LoRA controls.
    // Turbo is a separate single-pass acceleration path and must not be offered here.
    miniMaxLoraSection.style.display = "";
    miniMaxTurboSection.style.display = hideMultiPassIgnoredSettings || mode === "reference_to_video" ? "none" : "";
    miniMaxUseTurboLora.input.checked = hideMultiPassIgnoredSettings ? false : settings.use_turbo_lora;
    miniMaxUseTurboLora.input.disabled = hideMultiPassIgnoredSettings || settings.use_loras;
    for (const button of miniMaxModeButtons) {
      const active = button.dataset.minimaxH3Mode === mode;
      button.style.background = active ? "#06b6d4" : "#27272a";
      button.style.borderColor = active ? "#0891b2" : "#3f3f46";
      button.style.color = active ? "#082f49" : "#f4f4f5";
    }
    const refmodUi = miniMaxPassButtons.refmod;
    const refmodPipeline = settings.pipeline === "refmod";
    if (refmodUi) {
      refmodUi.modeChooser.style.display = refmodPipeline ? "none" : "grid";
      refmodUi.note.style.display = refmodPipeline ? "block" : "none";
      renderRefmodClothing(refmodUi.clothing, refmodPipeline ? segment : null);
      for (const button of refmodUi.pipelineButtons) {
        const active = button.dataset.minimaxH3Pipeline === (refmodPipeline ? "refmod" : "standard");
        button.setAttribute("aria-pressed", String(active));
        button.style.background = active ? "#06b6d4" : "#27272a";
        button.style.borderColor = active ? "#0891b2" : "#3f3f46";
        button.style.color = active ? "#082f49" : "#f4f4f5";
      }
    }
    // 2 Pass Advanced is not offered with RefMods.
    const singleOrTwoPassOnly = refmodPipeline || mode === "image_to_video";
    miniMaxPassButtons[2].style.display = singleOrTwoPassOnly ? "none" : "";
    miniMaxPassChooser.style.gridTemplateColumns = singleOrTwoPassOnly ? "repeat(2,minmax(0,1fr))" : "repeat(3,minmax(0,1fr))";
    miniMaxPassChooser.style.display = ["reference_to_video", "image_to_video"].includes(mode) ? "grid" : "none";
    for (const button of miniMaxPassButtons) {
      const active = button.dataset.passMode === settings.render_pass;
      button.setAttribute("aria-pressed", String(active));
      button.style.background = active ? "#06b6d4" : "#27272a";
      button.style.borderColor = active ? "#0891b2" : "#3f3f46";
      button.style.color = active ? "#082f49" : "#f4f4f5";
    }
    for (const [panelMode, panel] of Object.entries(miniMaxModePanels)) {
      panel.style.display = panelMode === mode ? "flex" : "none";
    }
    const sceneImageSource = segmentImageSource(segment);
    miniMaxKeyframes.sync(segment, mode, settings);
    const hasSceneImage = Boolean(sceneImageSource?.path || sceneImageSource?.data);
    // RefMods have no start frame, so the scene-image controls are hidden in the RefMod pipeline.
    miniMaxSceneImageUseField.style.display = hasSceneImage && !refmodPipeline ? "flex" : "none";
    miniMaxStartFrameCharacterInfluenceField.style.display = hasSceneImage && !refmodPipeline && miniMaxSceneImageUse.value === "exact_start_frame" ? "flex" : "none";
    miniMaxStartFrameReferenceNote.style.display = hasSceneImage && !refmodPipeline ? "block" : "none";
    if (segment) {
      if (!savedMiniMaxPrompts.has(segment)) savedMiniMaxPrompts.set(segment, String(segment.minimax_h3_prompt || segment.i2v_prompt || ""));
    }
    miniMaxPrompt.value = String(segment?.minimax_h3_prompt || segment?.i2v_prompt || "");
    miniMaxContinuationDirection.value = String(segment?.minimax_h3_continuation_direction || "");
    // When this scene's prompt is written from the previous scene's final frame, the prompt box is not used and the
    // continuation direction is what steers it. Otherwise it is the other way round. The render uses the same rule.
    const promptFromLastFrame = Boolean(segment) && miniMaxH3FrameContinuityPromptEnabled(segment);
    miniMaxContinuationDirectionField.style.display = promptFromLastFrame ? "flex" : "none";
    miniMaxContinuationStartField.style.display = promptFromLastFrame ? "flex" : "none";
    miniMaxContinuitySection.style.display = ["image_to_video", "image_reference_to_video"].includes(mode) ? "none" : "";
    miniMaxPrompt.disabled = promptFromLastFrame;
    miniMaxPrompt.style.opacity = promptFromLastFrame ? ".55" : "1";
    miniMaxPromptAutoNote.style.display = promptFromLastFrame ? "block" : "none";
    miniMaxContinuationDirection.disabled = !promptFromLastFrame;
    miniMaxContinuationDirectionField.style.opacity = promptFromLastFrame ? "1" : ".55";
    syncMiniMaxContinuationStart(segment, promptFromLastFrame);
    miniMaxPass2Prompt.value = String(segment?.minimax_h3_pass2_prompt || "");
    miniMaxPass2PromptField.style.display = threePass ? "flex" : "none";
    updateMiniMaxPromptCharacterStatus(segment);
    updateMiniMaxPromptSaveButtonState();
    const compactMiniMaxModeLabel = mode === "reference_to_video"
      ? "Ref2V"
      : mode === "image_to_video"
        ? "I2V"
        : mode === "video_to_video"
          ? "V2V"
          : "T2V";
    miniMaxCreatePromptButton.textContent = `Create ${compactMiniMaxModeLabel} Prompt`;
    miniMaxCreatePromptButton.title = `Generate the MiniMax H3 ${modeLabel} prompt with the selected LLM. The prompt is saved to this scene; it does not render video.`;
    miniMaxEditInstructionsButton.textContent = `Edit ${compactMiniMaxModeLabel} Instructions`;
    miniMaxEditInstructionsButton.title = `Edit the LLM instructions used to create MiniMax H3 ${modeLabel} prompts.`;
    const assignmentTabButton = miniMaxSubTabs.wrapper.querySelector('[data-value="speakers"]');
    if (assignmentTabButton) applyCompactButtonLabel(assignmentTabButton, isMiniMaxSingerAssignmentMode(segment) ? "Singer Assignment" : "Speaker Assignment", { minWidth: 0, padding: "7px 6px" });
    miniMaxPromptRunnerNote.textContent = `Uses the existing ${gemmaRunnerLabel()} selection from LLM Runner. The result is saved only as this scene's MiniMax H3 prompt.`;
    miniMaxAudioNote.textContent = settings.audio_mode === "built_in_audio"
      ? "MiniMax generates the scene audio from the prompt. In Short Film mode, character voice presets are configured in Reference Builder and copied exactly into every matching scene prompt."
      : "Uses custom scene audio or project audio unchanged for exact timing and lip sync. Native voice presets are hidden while Input Audio is selected.";
    // Image modes use per-scene frames and do not support between-scene continuation.
    const continuitySupported = isMiniMaxH3ContinuityAllowedForMode(settings.continuity_mode, mode, settings.render_pass);
    for (const option of miniMaxContinuityMode.options) {
      option.disabled = !isMiniMaxH3ContinuityAllowedForMode(option.value, mode, settings.render_pass);
    }
    miniMaxContinuityMode.disabled = false;
    // Masked continuation plans its own warm-up and tail, so the Render settings frames do not apply.
    const maskedActive = continuitySupported && settings.continuity_mode === "latent_continuation_masked";
    miniMaxWarmupFrames.disabled = maskedActive;
    miniMaxCooldownFrames.disabled = maskedActive;
    const maskedFramesTitle = maskedActive ? "Not used while Latent Continuation Masked is selected. It plans its own warm-up and tail." : "";
    miniMaxWarmupFrames.title = maskedFramesTitle;
    miniMaxCooldownFrames.title = maskedFramesTitle;
    const isLatentContinuation = isMiniMaxH3LatentContinuationMode(settings.continuity_mode);
    miniMaxLatentContinuationRow.style.display = (continuitySupported && isLatentContinuation) ? "flex" : "none";
    miniMaxLatentContextFrames.disabled = !continuitySupported || !isLatentContinuation;
    miniMaxContinuityPromptFromLastFrame.input.disabled = !continuitySupported || !isLatentContinuation;
    const showLocationTransitionControls = continuitySupported
      && isLatentContinuation
      && Boolean(settings.continuity_prompt_from_last_frame);
    miniMaxLocationTransitionControls.style.display = showLocationTransitionControls ? "flex" : "none";
    miniMaxLocationTransitionPreset.disabled = !showLocationTransitionControls;
    miniMaxLocationTransitionCustomField.style.display = showLocationTransitionControls
      && miniMaxLocationTransitionPreset.value === "custom" ? "flex" : "none";
    miniMaxContinuityNote.textContent = !continuitySupported
      ? "Latent Continuation Masked is not available in Image + Reference 2 Pass or 2 Pass Advanced. This scene renders without continuity."
      : isLatentContinuation
        ? `Latent Continuation Masked: copies a phase-aligned run of the predecessor's saved latent (39 frames recommended, also 90/141/192) into the head of this scene's latent and protects it with a denoise mask, so the model keeps those frames and generates the rest. Needs ComfyUI 0.34.0 or newer. Works in Single pass and 2 Pass (the exact head is applied again in pass 2). Not available in 2 Pass Advanced. Other context sizes fall back to 39. The head is trimmed from the video and audio together. The predecessor must have a saved latent.`
      : "Off: every scene starts independently from its normal MiniMax references.";
    if (continuitySupported && isLatentContinuation && settings.continuity_prompt_from_last_frame) {
      miniMaxContinuityNote.textContent += " Automatic prompt loop is ON: Scene 1 keeps your prompt. Before every later scene renders, the vision LLM uses the predecessor's actual final frame as its highest-priority opening truth, adds this scene's story/audio/reference context, saves a complete one-take prompt, and retries up to 10 times if prompting fails.";
    }
    if (continuitySupported && isLatentContinuation) {
      updateMiniMaxLatentPredecessorStatus(segment);
    } else {
      miniMaxLatentStatusPill.style.display = "none";
    }
    const sceneImageUse = miniMaxH3SceneImageUseForSegment(segment);
    miniMaxSceneImageUse.value = imageReferenceTwoPass ? "exact_start_frame" : sceneImageUse;
    miniMaxSceneImageUse.disabled = !segment || imageReferenceTwoPass;
    const exactStartFrameOption = Array.from(miniMaxSceneImageUse.options).find((option) => option.value === "exact_start_frame");
    if (exactStartFrameOption) exactStartFrameOption.disabled = settings.continuity_mode === "exact_start_frame" || isLatentContinuation;
    const startFrameCharacterInfluence = miniMaxH3StartFrameCharacterInfluenceForSegment(segment);
    miniMaxStartFrameCharacterInfluence.value = startFrameCharacterInfluence;
    miniMaxStartFrameCharacterInfluence.disabled = !segment
      || sceneImageUse !== "exact_start_frame"
      || settings.continuity_mode === "exact_start_frame"
      || isLatentContinuation;
    miniMaxStartFrameCharacterInfluenceField.style.display = hasSceneImage && sceneImageUse === "exact_start_frame" ? "flex" : "none";
    miniMaxStartFrameReferenceNote.textContent = (settings.continuity_mode === "exact_start_frame" || isLatentContinuation)
      && sceneImageUse === "exact_start_frame"
      ? "The previous rendered final frame is the sole exact opening frame. Choose an LLM-only inspiration mode or Do not use instead."
      : sceneImageUse === "environment_inspiration"
        ? "The scene image is shown only to the prompt-writing LLM for location, environment, atmosphere, mood, lighting, weather, colors, materials, background details, relevant objects, and scene activity. It must ignore framing, shot distance, camera angle, lens, composition, and every character detail, pose, and placement. MiniMax never receives this image."
        : sceneImageUse === "environment_framing_inspiration"
          ? "The scene image is shown only to the prompt-writing LLM for environment plus optional shot framing, distance, angle, lens, and composition inspiration. It must ignore every character's identity, appearance, clothing, pose, and placement. MiniMax never receives this image."
          : startFrameCharacterInfluence === "face_hair_only" && sceneImageUse === "exact_start_frame"
        ? "Image 1 keeps its pose, body, wardrobe, accessories, composition, camera, lighting, props, and environment. Character references supply only face identity and hair from the first generated frame—there is no visible swap or transformation."
        : sceneImageUse === "exact_start_frame"
          ? "The scene image becomes MiniMax Image 1 and locks the exact opening composition. Character and location references follow it."
          : "The scene image is not used by the prompt-writing LLM or MiniMax.";
    const imagePath = String(segment ? selectedSegmentImagePath(segment) || "" : "").trim();
    miniMaxImageModeSource.textContent = imagePath
      ? `Scene image input:\n${imagePath}`
      : "This scene has no selected timeline image. Add or generate an image before using Image to Video.";
    const videoReferences = Array.isArray(segment?.minimax_h3_video_references) ? segment.minimax_h3_video_references : [];
    miniMaxVideoReferenceRows.forEach((row, index) => {
      const item = videoReferences[index] || {};
      row.path.value = item.path || "";
      row.start.value = String(Math.max(0, Number(item.start_seconds || 0)));
      row.duration.value = String(Math.max(0, Number(item.duration || 0)));
      row.purpose.value = normalizeMiniMaxH3VideoPurpose(item.purpose || (index === 0 ? "continuation" : "movement"));
      row.useAudio.input.checked = Boolean(item.use_audio);
    });
    miniMaxSceneVideoButton.disabled = !miniMaxProject || !segment;
    syncMiniMaxReferenceButtons();
    renderMiniMaxSpeakerAssignmentPanel();
    syncTimelineTrimModeButton();
  }

  return {
    clearMiniMaxImageReferenceStartFrameOnModeSwitch, ensureMiniMaxSpeakerAssignments,
    isMiniMaxBuiltInSpeakerAssignmentMode, isMiniMaxSingerAssignmentMode, miniMaxH3ContinuityModeForSegment,
    miniMaxH3ContinuityReferenceReserved, miniMaxH3ModeForSegment, miniMaxH3SceneImageIsPromptInspiration,
    miniMaxH3SceneImageUseForSegment, miniMaxH3SettingsForSegment,
    miniMaxH3StartFrameCharacterInfluenceForSegment, miniMaxMappedSpeakersForSegment,
    renderMiniMaxSpeakerAssignmentPanel, saveMiniMaxH3SettingsFromPanel, saveMiniMaxSceneInputsFromPanel,
    setMiniMaxH3ModeForSegment, setMiniMaxH3RenderPassForSegment, syncMiniMaxH3Panel,
    syncMiniMaxReferenceButtons, toggleProjectVideoEngineFromBadge, updateMiniMaxPromptCharacterStatus,
    updateMiniMaxPromptSaveButtonState,
  };
}
