import {
  applyCompactButtonLabel,
  makeButton,
  makeCheckbox,
  makeField,
  makeInput,
  makeSearchableLoraPicker,
  makeSelect,
  makeSettingsPanel,
  makeSettingsSection,
  makeSubTabs,
} from "./controls.mjs";
import {
  DEFAULT_MINIMAX_H3_SETTINGS,
  MINIMAX_H3_AUDIO_MODE_OPTIONS,
  MINIMAX_H3_CONTINUITY_OPTIONS,
  MINIMAX_H3_LOCATION_TRANSITION_OPTIONS,
  MINIMAX_H3_MODE_OPTIONS,
  MINIMAX_H3_RESOLUTION_PRESETS,
  MINIMAX_H3_SAGE_ATTENTION_OPTIONS,
  MINIMAX_H3_SCENE_IMAGE_USE_OPTIONS,
  MINIMAX_H3_START_FRAME_CHARACTER_INFLUENCE_OPTIONS,
  MINIMAX_H3_VIDEO_REFERENCE_PURPOSES,
  miniMaxH3FrameSize,
  miniMaxH3PresetMegapixels,
  miniMaxH3TilePlan,
} from "./minimax_h3.mjs";

export function buildMiniMaxPanel({
  miniMaxGemmaModelSelect, miniMaxMmprojSelect, miniMaxReferencesButton, miniMaxSceneVideoButton,
  miniMaxTextGemmaModelSelect, miniMaxVideoReferencesButton, updateMiniMaxPromptSaveButtonState,
}) {
  const miniMaxEnginePanel = document.createElement("div");
  miniMaxEnginePanel.style.cssText = "display:none;flex-direction:column;gap:10px;";
  const miniMaxBanner = document.createElement("div");
  miniMaxBanner.innerHTML = `<div style="font-size:14px;font-weight:900;color:#cffafe;">MiniMax H3 Project</div><div style="font-size:11px;color:#a5f3fc;margin-top:3px;">Mode, models, and video settings apply to the whole project unless the active scene is locked to custom settings.</div>`;
  miniMaxBanner.style.cssText = "border:1px solid #0891b2;border-radius:7px;background:#083344;padding:10px;line-height:1.35;";
  const miniMaxModeChooser = document.createElement("div");
  miniMaxModeChooser.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;";
  const miniMaxModeButtons = MINIMAX_H3_MODE_OPTIONS.map((item) => {
    const button = item.value === "image_reference_to_video"
      ? makeButton("Image to Video\n2 Pass")
      : makeButton(item.buttonLabel || item.label);
    applyCompactButtonLabel(button, item.buttonLabel || item.label, { noMap: true, minWidth: 0, padding: "7px 6px", title: item.label });
    button.dataset.minimaxH3Mode = item.value;
    miniMaxModeChooser.append(button);
    return button;
  });
  const miniMaxPassChooser = document.createElement("div");
  miniMaxPassChooser.setAttribute("role", "group");
  miniMaxPassChooser.setAttribute("aria-label", "Reference to video passes");
  miniMaxPassChooser.style.cssText = "display:none;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;";
  const miniMaxSinglePassButton = makeButton("Single pass");
  const miniMaxTwoPassButton = makeButton("2 pass");
  const miniMaxThreePassButton = makeButton("2 pass advanced");
  const miniMaxPassButtons = [miniMaxSinglePassButton, miniMaxTwoPassButton, miniMaxThreePassButton];
  miniMaxPassButtons.forEach((button, index) => {
    button.dataset.passMode = ["single", "two_pass", "three_pass"][index];
    miniMaxPassChooser.append(button);
  });
  const miniMaxDiffusionModelPicker = makeSearchableLoraPicker(DEFAULT_MINIMAX_H3_SETTINGS.diffusion_model_name);
  const miniMaxClipPicker = makeSearchableLoraPicker(DEFAULT_MINIMAX_H3_SETTINGS.clip_name);
  const miniMaxVideoVaePicker = makeSearchableLoraPicker(DEFAULT_MINIMAX_H3_SETTINGS.video_vae_name);
  const miniMaxAudioVaePicker = makeSearchableLoraPicker(DEFAULT_MINIMAX_H3_SETTINGS.audio_vae_name);
  const miniMaxAudioMode = makeSelect(MINIMAX_H3_AUDIO_MODE_OPTIONS, DEFAULT_MINIMAX_H3_SETTINGS.audio_mode);
  const miniMaxContinuityMode = makeSelect(MINIMAX_H3_CONTINUITY_OPTIONS, DEFAULT_MINIMAX_H3_SETTINGS.continuity_mode);
  const MINIMAX_H3_LATENT_CONTEXT_OPTIONS = [
    { value: "16", label: "16 frames (5 tokens)" },
    { value: "22", label: "22 frames (7 tokens — recommended)" },
    { value: "39", label: "39 frames (12 tokens)" },
    { value: "56", label: "56 frames (17 tokens)" },
  ];
  const miniMaxLatentContextFrames = makeSelect(MINIMAX_H3_LATENT_CONTEXT_OPTIONS, String(DEFAULT_MINIMAX_H3_SETTINGS.latent_context_frames || 22));
  miniMaxLatentContextFrames.title = "Number of trailing context frames loaded directly from the predecessor scene's saved latent.";
  const miniMaxLatentContextField = makeField("Latent context frames", miniMaxLatentContextFrames);
  const miniMaxContinuityPromptFromLastFrame = makeCheckbox("Create each next scene prompt from the previous rendered final frame", false);
  miniMaxContinuityPromptFromLastFrame.wrapper.title = "Scene 1 keeps its authored prompt. Before rendering Scene 2 and later, the Builder extracts the predecessor's actual final frame and asks the vision LLM to create and save a complete continuous-shot prompt from it plus the scene's story, audio timing, and references.";
  const miniMaxLocationTransitionPreset = makeSelect(MINIMAX_H3_LOCATION_TRANSITION_OPTIONS, "normal");
  const miniMaxLocationTransitionPresetField = makeField("Location change transition", miniMaxLocationTransitionPreset);
  miniMaxLocationTransitionPresetField.title = "Applies globally to every unlocked MiniMax scene. Lock MiniMax settings on a scene to give that scene its own transition preset.";
  const miniMaxLocationTransitionCustom = document.createElement("textarea");
  miniMaxLocationTransitionCustom.placeholder = "Describe the transition you want, such as: combine surreal transformation with a cinematic eye transition.";
  miniMaxLocationTransitionCustom.style.cssText = "width:100%;min-height:64px;box-sizing:border-box;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;line-height:1.4;";
  const miniMaxLocationTransitionCustomField = makeField("Custom location transition", miniMaxLocationTransitionCustom);
  const miniMaxLocationTransitionNote = document.createElement("div");
  miniMaxLocationTransitionNote.textContent = "Global for all unlocked scenes; the MiniMax scene lock creates an override. The preset activates only at a mapped location boundary.";
  miniMaxLocationTransitionNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  const miniMaxLocationTransitionControls = document.createElement("div");
  miniMaxLocationTransitionControls.style.cssText = "display:none;flex-direction:column;gap:6px;padding:8px;border:1px solid #334155;border-radius:7px;background:#0f172a;";
  miniMaxLocationTransitionControls.append(miniMaxLocationTransitionPresetField, miniMaxLocationTransitionCustomField, miniMaxLocationTransitionNote);
  const miniMaxEditContinuityPromptInstructionsButton = makeButton("Edit frame-continuity LLM instructions");
  miniMaxEditContinuityPromptInstructionsButton.title = "Edit the dedicated vision-LLM instructions used for automatic frame-to-frame continuation prompts.";
  const miniMaxLatentStatusPill = document.createElement("div");
  miniMaxLatentStatusPill.style.cssText = "display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:700;border-radius:12px;padding:4px 10px;border:1px solid #334155;background:#0f172a;color:#94a3b8;margin-top:2px;";
  miniMaxLatentStatusPill.textContent = "Checking predecessor latent...";
  const miniMaxLatentContinuationRow = document.createElement("div");
  miniMaxLatentContinuationRow.style.cssText = "display:flex;flex-direction:column;gap:6px;margin-top:6px;";
  miniMaxLatentContinuationRow.append(miniMaxContinuityPromptFromLastFrame.wrapper, miniMaxLocationTransitionControls, miniMaxEditContinuityPromptInstructionsButton, miniMaxLatentContextField, miniMaxLatentStatusPill);
  const miniMaxContinuityNote = document.createElement("div");
  miniMaxContinuityNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  const miniMaxNoGgufNote = document.createElement("div");
  miniMaxNoGgufNote.textContent = "MiniMax H3 currently uses the standard diffusion-model loader. GGUF is not enabled yet.";
  miniMaxNoGgufNote.style.cssText = "font-size:11px;color:#fcd34d;line-height:1.4;";
  const miniMaxUseTurboLora = makeCheckbox("Use legacy MiniMax-H3 Turbo LoRA", false);
  miniMaxUseTurboLora.wrapper.title = "Applies the selected legacy Turbo LoRA through ComfyUI's built-in LoRA loader. It does not require a separate Turbo custom-node repository.";
  const miniMaxTurboLoraPicker = makeSearchableLoraPicker(DEFAULT_MINIMAX_H3_SETTINGS.turbo_lora_name);
  const miniMaxTurboLoraField = makeField("Turbo LoRA", miniMaxTurboLoraPicker.wrapper);
  const miniMaxTurboLoraStrength = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.turbo_lora_strength), "number");
  miniMaxTurboLoraStrength.min = "-10";
  miniMaxTurboLoraStrength.max = "10";
  miniMaxTurboLoraStrength.step = "0.01";
  const miniMaxTurboLoraStrengthField = makeField("Turbo LoRA strength", miniMaxTurboLoraStrength);
  const miniMaxTurboNote = document.createElement("div");
  miniMaxTurboNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  const miniMaxTurboSection = makeSettingsSection("Turbo acceleration", [
    miniMaxUseTurboLora.wrapper,
    miniMaxTurboLoraField,
    miniMaxTurboLoraStrengthField,
    miniMaxTurboNote,
  ]);
  const miniMaxUseLoras = makeCheckbox("Use MiniMax LoRAs?", false);
  const miniMaxLoraCount = makeInput("0", "number");
  miniMaxLoraCount.min = "0";
  miniMaxLoraCount.max = "4";
  miniMaxLoraCount.step = "1";
  const miniMaxLoraRows = document.createElement("div");
  miniMaxLoraRows.style.cssText = "display:none;flex-direction:column;gap:8px;";
  const miniMaxLoraSlots = [];
  for (let index = 0; index < 4; index += 1) {
    const slot = index + 1;
    const picker = makeSearchableLoraPicker("[none]");
    const strength = makeInput("1", "number");
    strength.min = "-10";
    strength.max = "10";
    strength.step = "0.01";
    const applyTo = makeSelect([
      { value: "both", label: "Both passes" },
      { value: "pass1", label: "Pass 1 only" },
      { value: "pass2", label: "Pass 2 only" },
    ], "both");
    const applyToField = makeField("2-pass target", applyTo);
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 92px minmax(120px,0.45fr);gap:8px;";
    row.append(makeField(`MiniMax LoRA ${slot}`, picker.wrapper), makeField("Strength", strength), applyToField);
    miniMaxLoraRows.append(row);
    miniMaxLoraSlots.push({ row, picker, strength, applyTo, applyToField });
  }
  const miniMaxLoraNote = document.createElement("div");
  miniMaxLoraNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  const miniMaxLoraSection = makeSettingsSection("Optional MiniMax LoRAs", [
    miniMaxUseLoras.wrapper,
    makeField("LoRA count", miniMaxLoraCount),
    miniMaxLoraRows,
    miniMaxLoraNote,
  ]);
  const miniMaxAspectRatio = makeSelect([
    "16:9 (Widescreen)",
    "9:16 (Portrait Widescreen)",
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "4:3 (Standard)",
    "3:4 (Portrait Standard)",
    "21:9 (Ultrawide)",
  ], DEFAULT_MINIMAX_H3_SETTINGS.aspect_ratio);
  const miniMaxMegapixels = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.megapixels), "number");
  miniMaxMegapixels.min = "0.1";
  miniMaxMegapixels.max = "16";
  miniMaxMegapixels.step = "0.1";
  const miniMaxSeed = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.seed), "number");
  miniMaxSeed.step = "1";
  const miniMaxWarmupFrames = makeInput("0", "number");
  miniMaxWarmupFrames.min = "0";
  miniMaxWarmupFrames.step = "1";
  const miniMaxCooldownFrames = makeInput("0", "number");
  miniMaxCooldownFrames.min = "0";
  miniMaxCooldownFrames.step = "1";
  const miniMaxRenderSettingsGrid = document.createElement("div");
  miniMaxRenderSettingsGrid.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;";
  const miniMaxAspectRatioField = makeField("Aspect ratio", miniMaxAspectRatio);
  // One output resolution for single pass, 2 Pass and 2 Pass Advanced (Pass 2 size).
  const miniMaxResolutionPreset = makeSelect(MINIMAX_H3_RESOLUTION_PRESETS, DEFAULT_MINIMAX_H3_SETTINGS.resolution_preset);
  const miniMaxResolutionPresetField = makeField("Output resolution", miniMaxResolutionPreset, "Shared by single pass, 2 Pass and 2 Pass Advanced. 2 Pass Advanced uses it as the Pass 2 size.");
  const miniMaxResolutionSummary = document.createElement("div");
  miniMaxResolutionSummary.style.cssText = "grid-column:1 / -1;font-size:11px;color:#a1a1aa;line-height:1.45;";
  const miniMaxMegapixelsField = makeField("Custom megapixels", miniMaxMegapixels);
  const miniMaxSeedField = makeField("Seed", miniMaxSeed);
  const miniMaxWarmupFramesField = makeField("Warmup frames", miniMaxWarmupFrames);
  const miniMaxCooldownFramesField = makeField("Cooldown frames", miniMaxCooldownFrames);
  miniMaxRenderSettingsGrid.append(
    miniMaxAspectRatioField,
    miniMaxResolutionPresetField,
    miniMaxMegapixelsField,
    miniMaxSeedField,
    miniMaxWarmupFramesField,
    miniMaxCooldownFramesField,
    miniMaxResolutionSummary,
  );
  const miniMaxSamplerName = makeSelect([
    "res_multistep",
    "euler",
    "euler_ancestral",
    "heun",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
    "uni_pc",
    "deis",
  ], DEFAULT_MINIMAX_H3_SETTINGS.sampler_name);
  const miniMaxScheduler = makeSelect([
    "simple",
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "ddim_uniform",
    "beta",
    "linear_quadratic",
    "kl_optimal",
  ], DEFAULT_MINIMAX_H3_SETTINGS.scheduler);
  const miniMaxSteps = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.steps), "number");
  miniMaxSteps.min = "1";
  miniMaxSteps.max = "1000";
  miniMaxSteps.step = "1";
  const miniMaxDenoise = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.denoise), "number");
  miniMaxDenoise.min = "0";
  miniMaxDenoise.max = "1";
  miniMaxDenoise.step = "0.01";
  const miniMaxRefImageSize = makeSelect([
    { value: "match", label: "Match generation size" },
    { value: "max", label: "Max identity fidelity (2048px short edge)" },
  ], DEFAULT_MINIMAX_H3_SETTINGS.ref_image_size);
  const threePassSamplerOptions = [
    "euler",
    "euler_ancestral",
    "res_multistep",
    "heun",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
    "uni_pc",
    "deis",
    "sa_solver",
  ];
  const threePassSchedulerOptions = [
    "beta",
    "simple",
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "ddim_uniform",
    "linear_quadratic",
    "kl_optimal",
  ];
  const advancedTwoPassControls = [1, 2].map((pass) => {
    const prefix = `advanced_two_pass_pass${pass}_`;
    // Only Pass 1 has its own resolution. Pass 2 is the shared output resolution above.
    const hasResolution = pass === 1;
    const megapixels = hasResolution ? makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS[`${prefix}megapixels`]), "number") : null;
    if (megapixels) {
      megapixels.min = "0.1";
      megapixels.max = "16";
      megapixels.step = "0.1";
    }
    const resolutionPreset = hasResolution
      ? makeSelect(MINIMAX_H3_RESOLUTION_PRESETS, DEFAULT_MINIMAX_H3_SETTINGS[`${prefix}resolution_preset`] || "custom")
      : null;
    const steps = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS[`${prefix}steps`]), "number");
    steps.min = "1";
    steps.max = "1000";
    steps.step = "1";
    const denoise = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS[`${prefix}denoise`]), "number");
    denoise.min = "0";
    denoise.max = "1";
    denoise.step = "0.01";
    const sampler = makeSelect(threePassSamplerOptions, DEFAULT_MINIMAX_H3_SETTINGS[`${prefix}sampler`]);
    const scheduler = makeSelect(threePassSchedulerOptions, DEFAULT_MINIMAX_H3_SETTINGS[`${prefix}scheduler`]);
    const seed = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS[`${prefix}seed`]), "number");
    seed.step = "1";
    // Shown in the main Render Settings grid, only for 2 Pass Advanced (the panel toggles it).
    const resolutionField = hasResolution ? document.createElement("div") : null;
    if (resolutionField) {
      resolutionField.style.display = "none";
      resolutionField.append(
        makeField("Pass 1 resolution", resolutionPreset, "Base-generation resolution before the MMH3 upscale to the output resolution (2 Pass Advanced only)."),
        makeField("Pass 1 custom megapixels", megapixels),
      );
    }
    const syncResolutionPreset = () => {
      if (!hasResolution) return;
      const resolved = miniMaxH3PresetMegapixels(resolutionPreset.value, miniMaxAspectRatio.value);
      megapixels.disabled = Boolean(resolved);
      if (resolved !== null) megapixels.value = String(resolved);
    };
    if (hasResolution) {
      resolutionPreset.addEventListener("change", syncResolutionPreset);
      miniMaxAspectRatio.addEventListener("change", syncResolutionPreset);
      syncResolutionPreset();
    }
    const samplingFields = [
      makeField("Steps", steps),
      makeField("Sampler", sampler),
      makeField("Scheduler", scheduler),
      makeField("Denoise", denoise),
      makeField("Seed", seed),
    ];
    return { prefix, megapixels, resolutionPreset, syncResolutionPreset, steps, denoise, sampler, scheduler, seed, resolutionField, samplingFields };
  });
  const miniMaxAdvancedSamplingFields = makeSettingsSection("Pass Sampling (Advanced)", [
    ...advancedTwoPassControls.map((control, index) => makeSettingsSection(
      index === 0 ? "Pass 1 — Base Generation" : "Pass 2 — MMH3 Tiled Refinement",
      control.samplingFields,
      false,
    )),
  ], false);
  // display:contents keeps the wrapper out of the grid so its two fields sit in the grid cells.
  const miniMaxPass1ResolutionFields = advancedTwoPassControls[0].resolutionField;
  miniMaxRenderSettingsGrid.insertBefore(miniMaxPass1ResolutionFields, miniMaxSeedField);
  const twoPassControls = [1, 2].map((pass) => {
    const prefix = `two_pass_pass${pass}_`;
    const steps = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS[`${prefix}steps`]), "number");
    steps.min = "1";
    steps.max = "1000";
    steps.step = "1";
    const denoise = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS[`${prefix}denoise`]), "number");
    denoise.min = "0";
    denoise.max = "1";
    denoise.step = "0.01";
    const sampler = makeSelect(threePassSamplerOptions, DEFAULT_MINIMAX_H3_SETTINGS[`${prefix}sampler`]);
    const scheduler = makeSelect(threePassSchedulerOptions, DEFAULT_MINIMAX_H3_SETTINGS[`${prefix}scheduler`]);
    const seed = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS[`${prefix}seed`]), "number");
    seed.step = "1";
    const section = makeSettingsSection(`Pass ${pass}`, [
      makeField("Steps", steps),
      makeField("Sampler", sampler),
      makeField("Scheduler", scheduler),
      makeField("Denoise", denoise),
      makeField("Seed", seed),
    ], pass === 1);
    return { prefix, steps, denoise, sampler, scheduler, seed, section };
  });
  const miniMaxTwoPassLatentScale = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_latent_upscale_scale), "number");
  miniMaxTwoPassLatentScale.min = "1";
  miniMaxTwoPassLatentScale.max = "8";
  miniMaxTwoPassLatentScale.step = "0.1";
  const miniMaxTwoPassRefImageSize = makeSelect([
    { value: "max", label: "max" },
    { value: "match", label: "match" },
  ], DEFAULT_MINIMAX_H3_SETTINGS.ref_image_size);
  const miniMaxThreePassRefImageSize = makeSelect([
    { value: "max", label: "max" },
    { value: "match", label: "match" },
  ], DEFAULT_MINIMAX_H3_SETTINGS.ref_image_size);
  const miniMaxTwoPassLatentUpscalerPicker = makeSearchableLoraPicker(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_latent_upscaler_name);
  const miniMaxAccelerationControls = [
    ["te_speed", "Use TE-Speed-MiniMaxH3 (OSS)"],
    ["feedforward", "Use FeedForward (lower VRAM for longer scenes)"],
    ["block_sparse_attention", "Use Block Sparse Attention (faster)"],
  ].map(([key, label]) => {
    const single = makeCheckbox(label, false);
    const pass1 = makeCheckbox("Pass 1", false);
    const pass2 = makeCheckbox("Pass 2", false);
    pass1.input.setAttribute("aria-label", `${label}, Pass 1`);
    pass2.input.setAttribute("aria-label", `${label}, Pass 2`);
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:12px;";
    single.wrapper.style.flex = "1";
    row.append(single.wrapper, pass1.wrapper, pass2.wrapper);
    return { key, single, pass1, pass2, row };
  });
  const miniMaxTwoPassUseFastVaeDecode = makeCheckbox("Use Fast Batched VAE Decode (batch size 8)", false);
  const miniMaxTwoPassTeProcessingControl = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_te_speed_processing_control), "number");
  miniMaxTwoPassTeProcessingControl.min = "0"; miniMaxTwoPassTeProcessingControl.max = "1"; miniMaxTwoPassTeProcessingControl.step = "0.01";
  const miniMaxTwoPassTeStart = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_te_speed_start_percent), "number");
  miniMaxTwoPassTeStart.min = "0"; miniMaxTwoPassTeStart.max = "1"; miniMaxTwoPassTeStart.step = "0.01";
  const miniMaxTwoPassTeEnd = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_te_speed_end_percent), "number");
  miniMaxTwoPassTeEnd.min = "0"; miniMaxTwoPassTeEnd.max = "1"; miniMaxTwoPassTeEnd.step = "0.01";
  const miniMaxTwoPassTeMcs = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_te_speed_mcs), "number");
  miniMaxTwoPassTeMcs.min = "1"; miniMaxTwoPassTeMcs.max = "64"; miniMaxTwoPassTeMcs.step = "1";
  const miniMaxTwoPassTeCacheDepth = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_te_speed_cache_depth), "number");
  miniMaxTwoPassTeCacheDepth.min = "0"; miniMaxTwoPassTeCacheDepth.max = "1"; miniMaxTwoPassTeCacheDepth.step = "0.01";
  const miniMaxTwoPassTeDevice = makeSelect(["auto", "cuda", "cpu"], DEFAULT_MINIMAX_H3_SETTINGS.two_pass_te_speed_device);
  const miniMaxTwoPassResizeMethod = makeSelect(["nvidia_rtx_vsr", "lanczos", "bicubic", "bilinear", "nearest-exact"], DEFAULT_MINIMAX_H3_SETTINGS.two_pass_final_resize_method);
  const miniMaxTwoPassOutputCrf = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_output_crf), "number");
  miniMaxTwoPassOutputCrf.min = "0"; miniMaxTwoPassOutputCrf.max = "100"; miniMaxTwoPassOutputCrf.step = "1";
  const miniMaxTwoPassSpeedNote = document.createElement("div");
  miniMaxTwoPassSpeedNote.textContent = "The selected Turbo LoRA is applied ONLY to pass 2 and makes the refinement pass MUCH faster. TE-Speed is separate optional acceleration for the model path; uncheck it to bypass the OSS speed node completely.";
  miniMaxTwoPassSpeedNote.style.cssText = "font-size:11px;color:#facc15;line-height:1.45;";
  const miniMaxTwoPassLatentScaleNote = document.createElement("div");
  miniMaxTwoPassLatentScaleNote.textContent = "Learned latent scale controls how much the pass-one video latent is enlarged before pass two. A value of 2 doubles its latent width and height. Keep this at 2 for the supplied upscaler; final width and height remain the actual requested output size.";
  miniMaxTwoPassLatentScaleNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.45;";
  const miniMaxAccelerationSettings = makeSettingsSection("Acceleration", [
    ...miniMaxAccelerationControls.map((control) => control.row),
    miniMaxTwoPassUseFastVaeDecode.wrapper,
    makeSettingsSection("TE-Speed Advanced", [
      makeField("Processing control", miniMaxTwoPassTeProcessingControl),
      makeField("Start percent", miniMaxTwoPassTeStart),
      makeField("End percent", miniMaxTwoPassTeEnd),
      makeField("MCS", miniMaxTwoPassTeMcs),
      makeField("Cache depth", miniMaxTwoPassTeCacheDepth),
      makeField("Device", miniMaxTwoPassTeDevice),
    ], false),
  ]);
  const miniMaxTwoPassSettings = makeSettingsSection("Two-Pass Settings", [
    miniMaxTwoPassSpeedNote,
    makeField("Reference image sizing", miniMaxTwoPassRefImageSize, "Controls the MiniMax H3 reference-conditioning image-size mode. Default: max."),
    makeField("Latent upscaler model", miniMaxTwoPassLatentUpscalerPicker.wrapper),
    ...twoPassControls.map((item) => item.section),
    makeSettingsSection("Latent Upscale Advanced", [
      miniMaxTwoPassLatentScaleNote,
      makeField("Learned latent scale", miniMaxTwoPassLatentScale),
    ], false),
    makeSettingsSection("Output Advanced", [
      makeField("Final resize method", miniMaxTwoPassResizeMethod),
      makeField("Output CRF", miniMaxTwoPassOutputCrf),
    ], false),
  ], false);
  miniMaxTwoPassSettings.style.display = "none";
  // Tiles, chunks, fades and the upscaler device come from the output resolution at run time
  // (minimax/tile_plan.py). The VRAM preset is the only tiling choice; bigger cards should use 2 Pass.
  const miniMaxAdvancedVramPreset = makeSelect([
    { value: "8gb", label: "8 GB — smallest tiles, shortest chunks" },
    { value: "12gb", label: "12 GB" },
    { value: "16gb", label: "16 GB" },
    { value: "24gb", label: "24 GB — largest tiles" },
  ], DEFAULT_MINIMAX_H3_SETTINGS.advanced_two_pass_vram_preset);
  const miniMaxAdvancedLatentUpscalerPicker = makeSearchableLoraPicker(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_latent_upscaler_name);
  const miniMaxAdvancedDependencyNote = document.createElement("div");
  miniMaxAdvancedDependencyNote.textContent = "Requires the latest Comfyui-MMH3-UltimateUpscale. Pass 2 uses temporal chunks and spatial tiles, planned automatically from the output resolution and your VRAM preset, and becomes the final timeline clip.";
  miniMaxAdvancedDependencyNote.style.cssText = "font-size:11px;color:#facc15;line-height:1.45;";
  const syncMiniMaxResolution = () => {
    const resolved = miniMaxH3PresetMegapixels(miniMaxResolutionPreset.value, miniMaxAspectRatio.value);
    miniMaxMegapixels.disabled = resolved !== null;
    if (resolved !== null) miniMaxMegapixels.value = String(resolved);
    const size = miniMaxH3FrameSize(miniMaxMegapixels.value, miniMaxAspectRatio.value);
    const plan = miniMaxH3TilePlan(miniMaxAdvancedVramPreset.value, miniMaxMegapixels.value, miniMaxAspectRatio.value);
    miniMaxResolutionSummary.textContent = `Output size: ${size.width}×${size.height}px. `
      + `2 Pass Advanced tiling (${miniMaxAdvancedVramPreset.value.replace("gb", " GB")}): ${plan.rows}×${plan.cols} tiles, ${plan.chunk}-frame chunks.`;
  };
  // The panel calls this after loading saved settings into the controls.
  miniMaxResolutionPreset.syncResolution = syncMiniMaxResolution;
  for (const control of [miniMaxResolutionPreset, miniMaxAspectRatio, miniMaxMegapixels, miniMaxAdvancedVramPreset]) {
    control.addEventListener("change", syncMiniMaxResolution);
  }
  miniMaxMegapixels.addEventListener("input", syncMiniMaxResolution);
  syncMiniMaxResolution();
  const miniMaxThreePassSettings = makeSettingsSection("2 Pass Advanced — MMH3 Ultimate Upscale", [
    miniMaxAdvancedDependencyNote,
    makeField("VRAM preset", miniMaxAdvancedVramPreset, "Sizes an equal tile grid for the output resolution and sets chunk length and overlap. 2 Pass Advanced is for cards up to 24 GB; larger cards should use 2 Pass."),
    makeField("Reference image sizing", miniMaxThreePassRefImageSize, "Controls the MiniMax H3 reference-conditioning image-size mode. Default: max."),
    makeField("Latent upscaler model", miniMaxAdvancedLatentUpscalerPicker.wrapper),
    miniMaxAdvancedSamplingFields,
  ], false);
  miniMaxThreePassSettings.style.display = "none";
  const miniMaxEasyCacheBypass = makeCheckbox("Bypass EasyCache", DEFAULT_MINIMAX_H3_SETTINGS.easy_cache_bypass);
  const miniMaxEasyCacheReuseThreshold = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.easy_cache_reuse_threshold), "number");
  miniMaxEasyCacheReuseThreshold.min = "0";
  miniMaxEasyCacheReuseThreshold.max = "1";
  miniMaxEasyCacheReuseThreshold.step = "0.01";
  const miniMaxEasyCacheStartPercent = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.easy_cache_start_percent), "number");
  miniMaxEasyCacheStartPercent.min = "0";
  miniMaxEasyCacheStartPercent.max = "1";
  miniMaxEasyCacheStartPercent.step = "0.01";
  const miniMaxEasyCacheEndPercent = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.easy_cache_end_percent), "number");
  miniMaxEasyCacheEndPercent.min = "0";
  miniMaxEasyCacheEndPercent.max = "1";
  miniMaxEasyCacheEndPercent.step = "0.01";
  const miniMaxEasyCacheVerbose = makeCheckbox("Verbose EasyCache logging", DEFAULT_MINIMAX_H3_SETTINGS.easy_cache_verbose);
  const miniMaxSageAttention = makeSelect(MINIMAX_H3_SAGE_ATTENTION_OPTIONS, DEFAULT_MINIMAX_H3_SETTINGS.sage_attention);
  const miniMaxMemoryEfficientSageAttention = makeCheckbox("Use memory-efficient MiniMax H3 Sage Attention patch", DEFAULT_MINIMAX_H3_SETTINGS.use_memory_efficient_sage_attention);
  const miniMaxFp16Accumulation = makeCheckbox("Enable fp16 accumulation", DEFAULT_MINIMAX_H3_SETTINGS.enable_fp16_accumulation);
  const miniMaxEasyCacheNote = document.createElement("div");
  miniMaxEasyCacheNote.textContent = "Bypass sends the diffusion model directly to the scheduler and removes EasyCache from the queued workflow copy.";
  miniMaxEasyCacheNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  const miniMaxSamplerSettings = makeSettingsSection("Sampler", [
    makeField("Sampler", miniMaxSamplerName),
    makeField("Scheduler", miniMaxScheduler),
    makeField("Steps", miniMaxSteps),
    makeField("Denoise", miniMaxDenoise),
  ]);
  const miniMaxTwoPassLoraPicker = makeSearchableLoraPicker(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_lora_name);
  const miniMaxTwoPassLoraStrength = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_lora_strength), "number");
  miniMaxTwoPassLoraStrength.min = "-10";
  miniMaxTwoPassLoraStrength.max = "10";
  miniMaxTwoPassLoraStrength.step = "0.01";
  const miniMaxTwoPassLoraPreset = document.createElement("div");
  miniMaxTwoPassLoraPreset.style.cssText = "display:flex;gap:8px;";
  miniMaxTwoPassLoraPreset.setAttribute("role", "group");
  miniMaxTwoPassLoraPreset.setAttribute("aria-label", "Pass 2 LoRA preset");
  const miniMaxTwoPassLoraPresetButtons = [4, 8].map((preset) => {
    const button = makeButton(`${preset} step`);
    button.dataset.preset = String(preset);
    button.title = `Select an installed ${preset}-step Turbo LoRA and set Pass 2 to ${preset / 2} steps.`;
    miniMaxTwoPassLoraPreset.append(button);
    return button;
  });
  const miniMaxTwoPassLoraPresetField = makeField("Pass 2 LoRA preset", miniMaxTwoPassLoraPreset);
  const miniMaxTwoPassLoraFields = document.createElement("div");
  miniMaxTwoPassLoraFields.style.cssText = "display:flex;flex-direction:column;gap:8px;min-width:0;";
  miniMaxTwoPassLoraFields.append(
    makeField("Pass 2 Turbo LoRA", miniMaxTwoPassLoraPicker.wrapper),
    makeField("Strength", miniMaxTwoPassLoraStrength),
  );
  const miniMaxTwoPassLoraLayout = document.createElement("div");
  miniMaxTwoPassLoraLayout.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:center;";
  miniMaxTwoPassLoraLayout.append(miniMaxTwoPassLoraFields, miniMaxTwoPassLoraPresetField);
  const miniMaxTwoPassLoraStatus = document.createElement("div");
  miniMaxTwoPassLoraStatus.setAttribute("role", "status");
  miniMaxTwoPassLoraStatus.style.cssText = "font-size:11px;color:#facc15;";
  const miniMaxTwoPassLoraDownload = document.createElement("a");
  miniMaxTwoPassLoraDownload.href = "https://huggingface.co/Kijai/MiniMax-H3_comfy/tree/main/loras";
  miniMaxTwoPassLoraDownload.target = "_blank";
  miniMaxTwoPassLoraDownload.rel = "noopener noreferrer";
  miniMaxTwoPassLoraDownload.textContent = "Download MiniMax Turbo LoRAs";
  miniMaxTwoPassLoraDownload.style.cssText = "font-size:11px;color:#67e8f9;";
  const miniMaxTwoPassLoraSection = makeSettingsSection("Pass 2 Turbo LoRA", [
    miniMaxTwoPassLoraLayout, miniMaxTwoPassLoraStatus, miniMaxTwoPassLoraDownload,
  ]);
  const miniMaxThreePassLoraPicker = makeSearchableLoraPicker(DEFAULT_MINIMAX_H3_SETTINGS.three_pass_lightx_lora_name);
  const miniMaxThreePassLoraStrength = makeInput(String(DEFAULT_MINIMAX_H3_SETTINGS.three_pass_lightx_lora_strength), "number");
  miniMaxThreePassLoraStrength.min = "-10";
  miniMaxThreePassLoraStrength.max = "10";
  miniMaxThreePassLoraStrength.step = "0.01";
  const miniMaxThreePassLoraSection = makeSettingsSection("3-Pass LightX2V LoRA", [
    makeField("LightX2V LoRA", miniMaxThreePassLoraPicker.wrapper),
    makeField("Strength", miniMaxThreePassLoraStrength),
  ]);
  const miniMaxReferenceConditioningSettings = makeSettingsSection("Reference Conditioning", [
    makeField("Reference image sizing", miniMaxRefImageSize),
  ]);
  const miniMaxEasyCacheSettings = makeSettingsSection("EasyCache", [
    miniMaxEasyCacheBypass.wrapper,
    miniMaxEasyCacheNote,
    makeField("Reuse threshold", miniMaxEasyCacheReuseThreshold),
    makeField("Start percent", miniMaxEasyCacheStartPercent),
    makeField("End percent", miniMaxEasyCacheEndPercent),
    miniMaxEasyCacheVerbose.wrapper,
  ]);
  const miniMaxModelLoaderSettings = makeSettingsSection("Model Loader", [
    makeField("Sage Attention", miniMaxSageAttention),
    miniMaxMemoryEfficientSageAttention.wrapper,
    miniMaxFp16Accumulation.wrapper,
  ]);
  const miniMaxAdvancedSettings = makeSettingsSection("Advanced Settings", [
    miniMaxSamplerSettings,
    miniMaxReferenceConditioningSettings,
    miniMaxEasyCacheSettings,
    miniMaxModelLoaderSettings,
  ], true);
  const miniMaxTextModePanel = makeSettingsPanel([]);
  const miniMaxTextModeNote = document.createElement("div");
  miniMaxTextModeNote.textContent = "Text to Video sends no image or video references. The scene prompt and custom/project audio drive the shot.";
  miniMaxTextModeNote.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
  miniMaxTextModePanel.append(miniMaxTextModeNote);
  const miniMaxImageModePanel = makeSettingsPanel([]);
  const miniMaxImageModeSource = document.createElement("div");
  miniMaxImageModeSource.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;overflow-wrap:anywhere;";
  miniMaxImageModePanel.append(miniMaxImageModeSource);
  const miniMaxReferenceModePanel = makeSettingsPanel([]);
  const miniMaxReferenceModeNote = document.createElement("div");
  miniMaxReferenceModeNote.textContent = "Uses the selected scene image as the exact start frame. Reference Builder images are optional and can provide character, location, ingredient-sheet, or storyboard-grid guidance.";
  miniMaxReferenceModeNote.style.cssText = miniMaxTextModeNote.style.cssText;
  const miniMaxSceneImageUse = makeSelect(MINIMAX_H3_SCENE_IMAGE_USE_OPTIONS, "off");
  const miniMaxSceneImageUseField = makeField("Scene image use — all unlocked Image/Reference-to-Video scenes", miniMaxSceneImageUse);
  miniMaxSceneImageUseField.title = "Choose whether each scene image is unused, becomes the exact MiniMax start frame, or is shown only to the prompt-writing vision LLM as inspiration.";
  const miniMaxStartFrameCharacterInfluence = makeSelect(MINIMAX_H3_START_FRAME_CHARACTER_INFLUENCE_OPTIONS, "full_character");
  const miniMaxStartFrameCharacterInfluenceField = makeField("Character reference influence — all Image/Reference-to-Video scenes", miniMaxStartFrameCharacterInfluence);
  miniMaxStartFrameCharacterInfluenceField.title = "Applies this character-reference priority across every eligible base scene that uses the scene image as its exact start frame.";
  const miniMaxStartFrameReferenceNote = document.createElement("div");
  miniMaxStartFrameReferenceNote.textContent = "Choose whether the scene image is unused, becomes MiniMax's exact start frame, or is shown only to the prompt-writing vision LLM as environment inspiration.";
  miniMaxStartFrameReferenceNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  miniMaxReferenceModePanel.append(miniMaxReferenceModeNote, miniMaxSceneImageUseField, miniMaxStartFrameCharacterInfluenceField, miniMaxStartFrameReferenceNote, miniMaxReferencesButton);
  const miniMaxVideoModePanel = makeSettingsPanel([]);
  const miniMaxVideoModeNote = document.createElement("div");
  miniMaxVideoModeNote.textContent = "Add up to three source/reference videos. You can also choose ordered Reference Builder images to replace or preserve people, backgrounds, locations, props, or visual style during the edit.";
  miniMaxVideoModeNote.style.cssText = miniMaxTextModeNote.style.cssText;
  const miniMaxVideoReferenceRows = [];
  for (let index = 0; index < 3; index += 1) {
    const path = makeInput("");
    path.placeholder = `Reference video ${index + 1} full path...`;
    const start = makeInput("0", "number");
    start.min = "0";
    start.step = "0.01";
    const duration = makeInput("0", "number");
    duration.min = "0";
    duration.step = "0.01";
    const purpose = makeSelect(MINIMAX_H3_VIDEO_REFERENCE_PURPOSES, index === 0 ? "continuation" : "movement");
    const useAudio = makeCheckbox("Use video audio", false);
    const timing = document.createElement("div");
    timing.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:7px;";
    timing.append(makeField("Start seconds", start), makeField("Duration (0 = auto)", duration));
    const row = makeSettingsSection(`Reference Video ${index + 1}`, [
      makeField("Video path", path),
      makeField("Reference purpose", purpose),
      timing,
      useAudio.wrapper,
    ], index === 0);
    miniMaxVideoReferenceRows.push({ path, start, duration, purpose, useAudio });
    miniMaxVideoModePanel.append(row);
  }
  const miniMaxUseCurrentSceneVideoButton = makeButton("Use Current Scene Video as Reference 1");
  miniMaxVideoModePanel.prepend(miniMaxVideoModeNote, miniMaxVideoReferencesButton, miniMaxUseCurrentSceneVideoButton);
  const miniMaxModePanels = {
    text_to_video: miniMaxTextModePanel,
    image_to_video: miniMaxImageModePanel,
    reference_to_video: miniMaxReferenceModePanel,
    video_to_video: miniMaxVideoModePanel,
  };
  const miniMaxPrompt = document.createElement("textarea");
  miniMaxPrompt.placeholder = "MiniMax H3 scene prompt...";
  miniMaxPrompt.style.cssText = "min-height:190px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:9px;font-size:12px;line-height:1.4;";
  ["keydown", "keypress", "keyup"].forEach((eventName) => miniMaxPrompt.addEventListener(eventName, (event) => event.stopPropagation()));
  const saveMiniMaxPromptButton = makeButton("Save Updated Prompt", "primary");
  saveMiniMaxPromptButton.title = "Save the updated MiniMax prompt to this scene and sync to the project storyboard.";
  saveMiniMaxPromptButton.style.width = "100%";
  saveMiniMaxPromptButton.style.marginTop = "4px";
  saveMiniMaxPromptButton.disabled = true;
  saveMiniMaxPromptButton.style.opacity = "0.5";
  saveMiniMaxPromptButton.style.cursor = "not-allowed";

  miniMaxPrompt.addEventListener("input", updateMiniMaxPromptSaveButtonState);
  miniMaxPrompt.addEventListener("change", updateMiniMaxPromptSaveButtonState);
  const miniMaxPromptCharacterStatus = document.createElement("div");
  miniMaxPromptCharacterStatus.style.cssText = "font-size:11px;line-height:1.4;border:1px solid #334155;border-radius:6px;background:#0f172a;padding:7px 9px;color:#86efac;";
  const miniMaxPass2Prompt = document.createElement("textarea");
  miniMaxPass2Prompt.placeholder = "Optional prompt used only by Ref to Video 2 Pass Advanced...";
  miniMaxPass2Prompt.style.cssText = "min-height:110px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:9px;font-size:12px;line-height:1.4;";
  ["keydown", "keypress", "keyup"].forEach((eventName) => miniMaxPass2Prompt.addEventListener(eventName, (event) => event.stopPropagation()));
  const miniMaxPass2PromptField = makeField("2nd Pass Prompt", miniMaxPass2Prompt);
  miniMaxPass2PromptField.style.display = "none";
  const miniMaxPromptActions = document.createElement("div");
  miniMaxPromptActions.style.cssText = "display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:8px;";
  const miniMaxCreatePromptButton = makeButton("Create MiniMax H3 Prompt", "primary");
  const miniMaxEditInstructionsButton = makeButton("Edit Text to Video Instructions");
  miniMaxPromptActions.append(miniMaxCreatePromptButton, miniMaxEditInstructionsButton);
  const miniMaxPromptRunnerNote = document.createElement("div");
  miniMaxPromptRunnerNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  const miniMaxSpeakerAssignmentPanel = makeSettingsPanel([]);
  const miniMaxSpeakerAssignmentNote = document.createElement("div");
  miniMaxSpeakerAssignmentNote.style.cssText = "font-size:11px;color:#cbd5e1;line-height:1.45;border:1px solid #155e75;border-radius:7px;background:#07111f;padding:9px;";
  const miniMaxSpeakerAssignmentList = document.createElement("div");
  miniMaxSpeakerAssignmentList.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  const miniMaxAddSpeakerCueButton = makeButton("Add Dialogue Cue", "primary");
  const miniMaxAutoTimeBeforePrompt = makeCheckbox("Auto-time lyric cues before creating prompts", false);
  miniMaxAutoTimeBeforePrompt.wrapper.style.cssText += "border:1px solid #155e75;border-radius:7px;background:#07111f;padding:10px;";
  miniMaxAutoTimeBeforePrompt.input.title = "Before each MiniMax H3 prompt, enable exact lyric-to-shot timing and run Stable-ts on the scene audio.";
  const miniMaxAutoTimeAllScenesButton = makeButton("Auto Time All Scenes", "primary");
  miniMaxAutoTimeAllScenesButton.title = "Run Stable-ts lyric timing for every eligible singing scene now, so the cue timing can be reviewed before prompts are created.";
  const miniMaxAutoTimeBeforePromptNote = document.createElement("div");
  miniMaxAutoTimeBeforePromptNote.textContent = "Auto Time All Scenes fills every eligible scene now for review. The checkbox still makes each MiniMax prompt wait for that scene's timing first. Both require scene lyric text, a mapped singer, and usable scene/project audio.";
  miniMaxAutoTimeBeforePromptNote.style.cssText = "font-size:11px;color:#94a3b8;line-height:1.45;margin:-3px 4px 2px;";
  miniMaxSpeakerAssignmentPanel.append(miniMaxSpeakerAssignmentNote, miniMaxAutoTimeBeforePrompt.wrapper, miniMaxAutoTimeAllScenesButton, miniMaxAutoTimeBeforePromptNote, miniMaxSpeakerAssignmentList, miniMaxAddSpeakerCueButton);
  const miniMaxAudioNote = document.createElement("div");
  miniMaxAudioNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;";
  const useSceneMiniMaxH3Settings = makeCheckbox("Lock MiniMax mode, models, and video settings for this scene", false);
  const useSceneMiniMaxH3SettingsNote = document.createElement("div");
  useSceneMiniMaxH3SettingsNote.textContent = "Off: this scene follows the project's current MiniMax mode, models, render settings, and advanced settings. On: the current values are copied to and locked for this scene only.";
  useSceneMiniMaxH3SettingsNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.4;margin-top:-4px;";
  const miniMaxSettingsScopeNote = document.createElement("div");
  miniMaxSettingsScopeNote.style.cssText = "font-size:11px;font-weight:800;color:#67e8f9;line-height:1.4;";
  const miniMaxSubTabs = makeSubTabs([
    {
      label: "Models",
      value: "models",
      content: makeSettingsPanel([
        miniMaxNoGgufNote,
        makeField("Diffusion model", miniMaxDiffusionModelPicker.wrapper),
        makeField("Text encoder / CLIP", miniMaxClipPicker.wrapper),
        makeField("Video VAE", miniMaxVideoVaePicker.wrapper),
        makeField("Audio VAE", miniMaxAudioVaePicker.wrapper),
        makeSettingsSection("Non-Vision LLM Models", [
          makeField("Non-Vision text LLM model", miniMaxTextGemmaModelSelect),
        ]),
        makeSettingsSection("Vision LLM Models", [
          makeField("Vision LLM model", miniMaxGemmaModelSelect),
          makeField("Vision mmproj", miniMaxMmprojSelect),
        ]),
      ]),
    },
    {
      label: "Video Settings",
      value: "input",
      content: makeSettingsPanel([
        useSceneMiniMaxH3Settings.wrapper,
        useSceneMiniMaxH3SettingsNote,
        miniMaxSettingsScopeNote,
        makeSettingsSection("Audio", [
          makeField("Audio mode", miniMaxAudioMode),
          miniMaxAudioNote,
        ]),
        makeSettingsSection("Between-scene continuity", [
          makeField("Previous rendered final frame", miniMaxContinuityMode),
          miniMaxLatentContinuationRow,
          miniMaxContinuityNote,
        ]),
        miniMaxLoraSection,
        miniMaxTwoPassLoraSection,
        miniMaxThreePassLoraSection,
        miniMaxTurboSection,
        ...Object.values(miniMaxModePanels),
        makeSettingsSection("Render Settings", [miniMaxRenderSettingsGrid]),
        miniMaxAdvancedSettings,
        miniMaxAccelerationSettings,
        miniMaxTwoPassSettings,
        miniMaxThreePassSettings,
      ]),
    },
    {
      label: "Speaker Assignment",
      value: "speakers",
      content: miniMaxSpeakerAssignmentPanel,
    },
    {
      label: "LLM Prompting",
      value: "prompt",
      content: makeSettingsPanel([
        miniMaxPromptRunnerNote,
        miniMaxPromptActions,
        makeField("MiniMax H3 prompt", miniMaxPrompt),
        saveMiniMaxPromptButton,
        miniMaxPromptCharacterStatus,
        miniMaxPass2PromptField,
      ]),
    },
  ]);
  miniMaxEnginePanel.append(miniMaxBanner, miniMaxModeChooser, miniMaxPassChooser, miniMaxSubTabs.wrapper, miniMaxSceneVideoButton);

  return {
    advancedTwoPassControls, miniMaxAccelerationControls, miniMaxAddSpeakerCueButton,
    miniMaxAdvancedLatentUpscalerPicker, miniMaxAdvancedSettings, miniMaxAdvancedVramPreset, miniMaxAspectRatio, miniMaxAudioMode,
    miniMaxAudioNote, miniMaxAudioVaePicker, miniMaxAutoTimeAllScenesButton, miniMaxAutoTimeBeforePrompt,
    miniMaxClipPicker, miniMaxContinuityMode, miniMaxContinuityNote, miniMaxContinuityPromptFromLastFrame,
    miniMaxCooldownFrames, miniMaxCreatePromptButton, miniMaxDenoise, miniMaxDiffusionModelPicker,
    miniMaxEasyCacheBypass, miniMaxEasyCacheEndPercent, miniMaxEasyCacheReuseThreshold,
    miniMaxEasyCacheSettings, miniMaxEasyCacheStartPercent, miniMaxEasyCacheVerbose,
    miniMaxEditContinuityPromptInstructionsButton, miniMaxEditInstructionsButton, miniMaxEnginePanel,
    miniMaxFp16Accumulation, miniMaxImageModeSource, miniMaxLatentContextFrames, miniMaxLatentContinuationRow,
    miniMaxLatentStatusPill, miniMaxLocationTransitionControls, miniMaxLocationTransitionCustom,
    miniMaxLocationTransitionCustomField, miniMaxLocationTransitionPreset, miniMaxLoraCount, miniMaxLoraNote,
    miniMaxLoraRows, miniMaxLoraSection, miniMaxLoraSlots, miniMaxMegapixels, miniMaxMegapixelsField, miniMaxResolutionPreset,
    miniMaxMemoryEfficientSageAttention, miniMaxModeButtons, miniMaxModelLoaderSettings, miniMaxModePanels,
    miniMaxPass2Prompt, miniMaxPass2PromptField, miniMaxPassButtons, miniMaxPassChooser, miniMaxPrompt,
    miniMaxPromptCharacterStatus, miniMaxPromptRunnerNote, miniMaxReferenceConditioningSettings,
    miniMaxRefImageSize, miniMaxSageAttention, miniMaxSamplerName, miniMaxSamplerSettings,
    miniMaxSceneImageUse, miniMaxSceneImageUseField, miniMaxScheduler, miniMaxSeed, miniMaxSeedField,
    miniMaxSettingsScopeNote, miniMaxSpeakerAssignmentList, miniMaxSpeakerAssignmentNote,
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
    miniMaxTwoPassUseFastVaeDecode, miniMaxUseCurrentSceneVideoButton, miniMaxUseLoras, miniMaxUseTurboLora,
    miniMaxVideoReferenceRows, miniMaxVideoVaePicker, miniMaxWarmupFrames, saveMiniMaxPromptButton,
    twoPassControls, useSceneMiniMaxH3Settings, useSceneMiniMaxH3SettingsNote,
  };
}
