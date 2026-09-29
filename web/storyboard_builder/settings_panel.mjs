import { createToast, makeButton, replaceLabeledPlanningLine } from "./controls.mjs";
import { FACIAL_PERFORMANCE_PRESETS, PERFORMANCE_STYLE_PRESETS } from "./performance_presets.mjs";
import {
  normalizeStoryboardProjectVideoEngine,
  normalizeStoryLayer,
  storyboardCutFrequencyLabel,
  storyboardCutFrequencyValue,
  storyboardSpeedGuidance,
  storyboardSpeedLabel,
  storyboardSpeedValue,
} from "./scenes.mjs";
import { normalizeStoryboardScriptImportState } from "./script_import.mjs";
import {
  normalizeStoryboardCustomCameraFlowSequence,
  STORYBOARD_CAMERA_FLOW_PRESETS,
  STORYBOARD_IMAGE_AESTHETIC_PRESETS,
  STORYBOARD_IMAGE_SHOT_FLOW_PRESETS,
  storyboardCameraFlowEntry,
} from "./shot_presets.mjs";
import {
  normalizeStoryboardCustomFxJson,
  storyboardFxPreset,
  storyboardMiniMaxVideoStylePreset,
  storyboardTemporalWorldEffectPreset,
} from "./video_style.mjs";

export function createSettingsPanel({
  applyDialoguePlanButton, cameraFlowInfo, cameraSpeedInfo, cameraSpeedValue, characterSpeedInfo,
  characterSpeedValue, consistencyInfo, createMissingBeatsButton, createStoryArcButton,
  createStoryBriefButton, cutFrequencyInfo, cutFrequencyValue, detectSectionsButton, facialCustomInfo,
  facialInfo, facialPerformancePresets, fxCustomControls, fxInfo, idLoraDialoguePlanner,
  idLoraDialoguePlannerText, idLoraDialogueSceneCount, imageAestheticInfo, imageAestheticPresets,
  imageCustomStyleInfo, imageCustomStyleInput, imageShotFlowPresets, imageShotInfo, imageWorldStyleInfo,
  imageWorldStyleSelect, isFullyCustomShortFilm, isIdLoraMode, isMiniMaxShortFilmMode,
  miniMaxGuidedWorkflowSteps, miniMaxScriptImporter, miniMaxScriptImporterText, openMiniMaxScriptMapperButton,
  performanceInfo, performanceStylePresets, planDialogueScenesButton, promptRunnerName, refreshActionButtons,
  renderTable, sceneDefaultsPanel, shortFilmPlanningModeInfo, shortFilmPlanningModeWrap,
  state, storyActions, storyLayerPanel, usesFilmPlanningProfile,
}) {
  function imageShotFlowPresetForMode(value = "") {
    return imageShotFlowPresets[value] || imageShotFlowPresets[Object.keys(imageShotFlowPresets)[0]] || STORYBOARD_IMAGE_SHOT_FLOW_PRESETS.intimate;
  }
  function imageAestheticPresetForMode(value = "") {
    return imageAestheticPresets.find((item) => item.value === value) || imageAestheticPresets[0] || STORYBOARD_IMAGE_AESTHETIC_PRESETS[0];
  }
  function performancePresetForMode(value = "") {
    return performanceStylePresets.find((item) => item.value === value) || performanceStylePresets[0] || PERFORMANCE_STYLE_PRESETS[0];
  }
  function facialPresetForMode(value = "") {
    return facialPerformancePresets.find((item) => item.value === value) || facialPerformancePresets[0] || FACIAL_PERFORMANCE_PRESETS[0];
  }
  function openCustomCameraFlowDialog() {
    return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100070;background:rgba(0,0,0,.74);display:flex;align-items:center;justify-content:center;padding:22px;box-sizing:border-box;";
    const panel = document.createElement("div");
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.style.cssText = "width:min(900px,calc(100vw - 44px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:11px;background:#0f172a;color:#e5e7eb;box-shadow:0 24px 90px rgba(0,0,0,.7);";
    const header = document.createElement("div");
    header.style.cssText = "padding:16px 18px;background:#083f4f;border-bottom:1px solid #155e75;";
    const title = document.createElement("div");
    title.style.cssText = "font-size:18px;font-weight:900;color:#cffafe;";
    title.textContent = "Import Custom Camera Shot List";
    const subtitle = document.createElement("div");
    subtitle.style.cssText = "margin-top:5px;color:#bae6fd;font-size:12px;line-height:1.45;";
    subtitle.textContent = "Paste a JSON list, a JSON object containing shots/sequence/candidates, or one shot per line. Optional camera movement can follow a pipe, em dash, arrow, or =>.";
    header.append(title, subtitle);
    const body = document.createElement("div");
    body.style.cssText = "padding:16px 18px;display:grid;gap:12px;";
    const examples = document.createElement("pre");
    examples.style.cssText = "margin:0;padding:10px;border:1px solid #334155;border-radius:8px;background:#07111f;color:#cbd5e1;font:11px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;";
    examples.textContent = `Accepted examples:\n\nJSON array:\n[{"shot":"wide performance shot","camera":"slow push-in"},{"shot":"close-up of the eyes","camera":"pan left"}]\n\nJSON object:\n{"shots":[{"shot":"tracking shot","camera":"side follow"}]}\n\nPlain list:\n1. Wide shot — slow pull-back\n2. Close-up of the hands | slow tilt down\n3. Overhead shot -> slow drift`;
    const input = document.createElement("textarea");
    input.value = state.customCameraFlowSequence.length ? JSON.stringify(state.customCameraFlowSequence, null, 2) : "";
    input.placeholder = "Paste or enter your custom camera-shot list here...";
    input.style.cssText = "width:100%;min-height:260px;resize:vertical;box-sizing:border-box;border:1px solid #475569;border-radius:8px;background:#020617;color:#e2e8f0;padding:11px;font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;outline:none;";
    const status = document.createElement("div");
    status.style.cssText = "min-height:18px;color:#94a3b8;font-size:12px;line-height:1.4;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;justify-content:flex-end;gap:9px;";
    const cancel = makeButton("Cancel");
    const importButton = makeButton("Import Custom List", "primary");
    actions.append(cancel, importButton);
    body.append(examples, input, status, actions);
    panel.append(header, body);
    backdrop.append(panel);
    document.body.append(backdrop);
    const finish = (value) => {
      document.removeEventListener("keydown", onKeyDown, true);
      backdrop.remove();
      resolve(value);
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        finish(null);
      }
    };
    document.addEventListener("keydown", onKeyDown, true);
    cancel.onclick = () => finish(null);
    importButton.onclick = () => {
      const sequence = normalizeStoryboardCustomCameraFlowSequence(input.value);
      if (!sequence.length) {
        status.textContent = "No valid shot entries were found. Add a shot description, then try Import again.";
        status.style.color = "#fca5a5";
        return;
      }
      finish(sequence);
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) finish(null);
    });
    input.focus();
  });
  }

  function cameraFlowEntryForScene(profileKey, sceneIndex, previousMotion = "") {
    return storyboardCameraFlowEntry(profileKey, sceneIndex, previousMotion, state.customCameraFlowSequence);
  }

  function sceneLooksLikeStarterPlaceholder(scene = {}) {
    const text = [
      scene.lyrics,
      scene.story_beat,
      scene.prompt_summary,
      scene.motion_summary,
      scene.image_prompt,
      scene.video_prompt,
      scene.image_path,
      scene.setting,
    ].map((item) => String(item || "").trim()).join("");
    return !text;
  }
  function shouldShowFilmDialoguePlanner() {
    return (isIdLoraMode || (isMiniMaxShortFilmMode && !isFullyCustomShortFilm()))
      && state.scenes.length > 0
      && state.scenes.length <= 2
      && state.scenes.every(sceneLooksLikeStarterPlaceholder);
  }
  function hasFilmDialoguePlan() {
    return (isIdLoraMode || isMiniMaxShortFilmMode)
      && state.scenes.some((scene) => String(scene.lyrics || scene.story_beat || scene.image_prompt || "").trim())
      && (isIdLoraMode
        ? state.scenes.some((scene) => String(scene.video_prompt_type || "") === "id_lora")
        : state.scenes.some((scene) => normalizeStoryboardProjectVideoEngine(scene.project_video_engine || state.projectVideoEngine) === "minimax_h3"));
  }

  function refreshSetupPanelSummaries() {
    const cameraPreset = STORYBOARD_CAMERA_FLOW_PRESETS[state.cameraFlow] || STORYBOARD_CAMERA_FLOW_PRESETS.balanced;
    const imageShotPreset = imageShotFlowPresetForMode(state.imageShotFlow);
    const imageAestheticPreset = imageAestheticPresetForMode(state.imageAesthetic);
    const videoStylePreset = storyboardMiniMaxVideoStylePreset(state.videoStyle);
    const temporalEffectPreset = storyboardTemporalWorldEffectPreset(state.temporalWorldEffect);
    const performancePreset = performancePresetForMode(state.performanceStyle);
    const facialPreset = facialPresetForMode(state.facialPerformance);
    sceneDefaultsPanel.setSummary(state.mode === "image_to_video_prep"
      ? `${cameraPreset.label || "Camera flow"}${state.videoStyle ? ` · ${videoStylePreset.label}` : ""}${state.temporalWorldEffect ? ` · ${temporalEffectPreset.label}` : ""}${state.fxPreset ? ` · ${storyboardFxPreset(state.fxPreset).label}` : ""} · camera ${storyboardSpeedValue(state.cameraMotionSpeed, 4)}/10 · cuts ${storyboardCutFrequencyValue(state.cutFrequency)}/10 · character ${storyboardSpeedValue(state.characterMotionSpeed, 4)}/10 · ${performancePreset.label || "Performance style"} · ${facialPreset.label || "Facial performance"}${state.globalConsistencyPhrase ? " · consistency phrase" : ""}`
      : `${imageShotPreset.label || "Still shot flow"} · ${imageAestheticPreset.label || "Image aesthetic"} · ${performancePreset.label || "Performance style"} · ${facialPreset.label || "Facial performance"}${state.globalConsistencyPhrase ? " · consistency phrase" : ""}`);
    const beatCount = state.scenes.filter((scene) => String(scene.story_beat || "").trim()).length;
    const sectionCount = state.scenes.filter((scene) => String(scene.lyric_section || "").trim()).length;
    const hasBrief = Boolean(String(state.storyLayer.song_story_brief || "").trim());
    const hasArc = Boolean(String(state.storyLayer.user_story_arc || "").trim());
    const lyricStrength = normalizeStoryLayer(state.storyLayer).lyric_story_strength;
    const filmPlannerVisible = shouldShowFilmDialoguePlanner();
    const fullyCustom = isFullyCustomShortFilm();
    const activeScriptImport = normalizeStoryboardScriptImportState(state.scriptImport);
    shortFilmPlanningModeWrap.style.display = isMiniMaxShortFilmMode ? "grid" : "none";
    miniMaxGuidedWorkflowSteps.style.display = isMiniMaxShortFilmMode && state.miniMaxH3AudioMode === "built_in_audio" && !fullyCustom ? "grid" : "none";
    miniMaxScriptImporter.style.display = isMiniMaxShortFilmMode && state.miniMaxH3AudioMode === "built_in_audio" ? "grid" : "none";
    miniMaxScriptImporterText.innerHTML = activeScriptImport.enabled
      ? `<div style="font-weight:900;color:#cffafe;">Authoritative Script Active</div><div style="color:#bae6fd;line-height:1.4;margin-top:3px;"><strong>${activeScriptImport.cues.length}</strong> exact dialogue cues are mapped into <strong>${activeScriptImport.scene_plan.scene_count}</strong> planned MiniMax segments at a ${activeScriptImport.maximum_scene_seconds}-second maximum. Guided Film may develop the visual story but cannot rewrite the dialogue.</div>`
      : `<div style="font-weight:900;color:#cffafe;">Import Script / Script Mapper</div><div style="color:#bae6fd;line-height:1.4;margin-top:3px;">Paste a <strong>speaker: dialogue</strong> script or load a .txt/.json file. Validate exact cues, match speakers, and preview automatically timed MiniMax segments without changing the timeline.</div>`;
    openMiniMaxScriptMapperButton.textContent = activeScriptImport.enabled ? "Step 1 — Review / Replace Script" : "Step 1 — Import / Activate Script";
    if (activeScriptImport.enabled) {
      idLoraDialogueSceneCount.value = String(activeScriptImport.scene_plan.scene_count || 1);
      idLoraDialogueSceneCount.disabled = true;
      idLoraDialogueSceneCount.title = "The authoritative Script Mapper plan controls the required segment count.";
      idLoraDialoguePlannerText.innerHTML = `<div style="font-weight:900;color:#cffafe;">Step 2 — Develop Imported Script Storyboard</div><div style="color:#bae6fd;line-height:1.35;margin-top:3px;">The LLM will create editable storyboard scene cards with visual story beats, actions, reactions, shots, camera direction, locations, ambience, and continuity for all ${activeScriptImport.scene_plan.scene_count} locked script sections. Exact dialogue, speakers, and order are enforced. The timeline remains unchanged. Afterward, complete <strong>Step 3</strong> by reviewing the cards below.</div>`;
      planDialogueScenesButton.textContent = `Step 2 — Develop ${activeScriptImport.scene_plan.scene_count} Scenes`;
      planDialogueScenesButton.title = `Develop ${activeScriptImport.scene_plan.scene_count} editable storyboard scene cards. After reviewing them, use Create ${activeScriptImport.scene_plan.scene_count} Timeline Segments.`;
    } else {
      idLoraDialogueSceneCount.disabled = false;
      idLoraDialogueSceneCount.title = "Number of guided dialogue scenes to create.";
      idLoraDialoguePlannerText.innerHTML = isMiniMaxShortFilmMode
        ? `<div style="font-weight:900;color:#cffafe;">Step 2 — Plan Storyboard Scenes</div><div style="color:#bae6fd;line-height:1.35;margin-top:3px;">Enter a story idea, outline, or pasted script above. If left blank, the selected LLM invents editable short-film storyboard scenes from your MiniMax H3 characters and locations. The timeline remains unchanged until Step 4.</div>`
        : `<div style="font-weight:900;color:#cffafe;">Plan Storyboard Scenes</div><div style="color:#bae6fd;line-height:1.35;margin-top:3px;">Enter a story idea, outline, or pasted script above. If left blank, the selected LLM invents editable short-film storyboard scenes from your ID-LoRA characters and locations. This does not alter the timeline until you choose Create Timeline Segments.</div>`;
      planDialogueScenesButton.textContent = isMiniMaxShortFilmMode ? "Step 2 — Plan Storyboard Scenes" : "Plan Storyboard Scenes";
      planDialogueScenesButton.title = "Develop editable storyboard scene cards. This does not create Video Builder timeline segments.";
    }
    shortFilmPlanningModeInfo.innerHTML = fullyCustom
      ? `<strong style="color:#cffafe;">Manual scene cards are authoritative.</strong><br>Enter dialogue in Speaker Assignment and fill the scene beat, action, shot, camera, setting, references, audio direction, and continuity yourself. Prompt generation formats your entries but may not invent or rewrite them.`
      : `<strong style="color:#cffafe;">The LLM can help plan the film.</strong><br>Use the premise/script, reference characters, film shot coverage, story beats, and dialogue planner to create scene cards. You can still edit every result manually before prompting.`;
    idLoraDialoguePlanner.style.display = !fullyCustom && (activeScriptImport.enabled || filmPlannerVisible || hasFilmDialoguePlan()) ? "grid" : "none";
    const hasApplyDialogueCallback = isIdLoraMode ? Boolean(state.onApplyIdLoraDialoguePlan) : Boolean(state.onApplyMiniMaxDialoguePlan);
    applyDialoguePlanButton.style.display = hasFilmDialoguePlan() && hasApplyDialogueCallback ? "" : "none";
    const plannedTimelineSceneCount = state.scenes.filter((scene) => String(scene.lyrics || scene.story_beat || scene.image_prompt || "").trim()).length;
    applyDialoguePlanButton.textContent = plannedTimelineSceneCount
      ? `${isMiniMaxShortFilmMode ? "Step 4 — " : ""}Create ${plannedTimelineSceneCount} Timeline Segment${plannedTimelineSceneCount === 1 ? "" : "s"}`
      : `${isMiniMaxShortFilmMode ? "Step 4 — " : ""}Create Timeline Segments`;
    applyDialoguePlanButton.title = plannedTimelineSceneCount
      ? `Create ${plannedTimelineSceneCount} real Video Builder timeline segment${plannedTimelineSceneCount === 1 ? "" : "s"} from these storyboard scenes. This may replace existing base timeline scenes.`
      : "Create real Video Builder timeline segments from the reviewed storyboard scenes.";
    storyActions.style.display = fullyCustom ? "none" : "flex";
    createStoryArcButton.textContent = usesFilmPlanningProfile ? "Create Story Premise" : "Create User Story Arc";
    createStoryBriefButton.textContent = usesFilmPlanningProfile ? "Create Short Film Brief" : "Create Story Brief";
    createMissingBeatsButton.textContent = isIdLoraMode ? "Create Missing Scene Beats" : "Create Missing Scene Beats";
    detectSectionsButton.style.display = usesFilmPlanningProfile ? "none" : "";
    storyLayerPanel.setSummary(usesFilmPlanningProfile
      ? `${state.storyLayer.enabled === false ? "Off" : "On"} · ${isIdLoraMode ? "ID-LoRA dialogue story" : (fullyCustom ? "MiniMax fully custom film" : "MiniMax guided film")} · ${beatCount}/${state.scenes.length} beats${hasBrief ? " · brief" : ""}${hasArc ? " · premise" : ""}${filmPlannerVisible ? " · starter scenes" : ""}`
      : `${state.storyLayer.enabled === false ? "Off" : "On"} · lyric ${lyricStrength}/10 · ${beatCount}/${state.scenes.length} beats · ${sectionCount}/${state.scenes.length} sections${hasBrief ? " · brief" : ""}${hasArc ? " · user arc" : ""}`);
    refreshActionButtons();
  }

  function refreshCameraFlowInfo() {
    const preset = STORYBOARD_CAMERA_FLOW_PRESETS[state.cameraFlow] || STORYBOARD_CAMERA_FLOW_PRESETS.balanced;
    const count = state.cameraFlow === "custom"
      ? normalizeStoryboardCustomCameraFlowSequence(state.customCameraFlowSequence).length
      : (preset.sequence?.length || 0);
    cameraFlowInfo.textContent = state.cameraFlow === "off"
      ? preset.description
      : `${preset.description} ${state.cameraFlow === "custom" ? (count ? `The project list contains ${count} shot${count === 1 ? "" : "s"}.` : "Import a custom shot list to activate this flow.") : `For any scene count, it cycles through ${count} camera beats and only fills blank fields.`}`;
    refreshSetupPanelSummaries();
  }

  function refreshCameraSpeedInfo() {
    cameraSpeedValue.textContent = storyboardSpeedLabel(state.cameraMotionSpeed, "camera");
    cameraSpeedInfo.textContent = storyboardSpeedGuidance(state.cameraMotionSpeed, "camera");
    refreshSetupPanelSummaries();
  }

  function refreshCutFrequencyInfo() {
    const frequency = storyboardCutFrequencyValue(state.cutFrequency);
    const engineLabel = state.projectVideoEngine === "minimax_h3" ? "MiniMax" : "LTX";
    cutFrequencyValue.textContent = storyboardCutFrequencyLabel(frequency);
    cutFrequencyInfo.textContent = frequency <= 0
      ? `${engineLabel} prompts use one smooth, continuous shot for every segment. Existing prompts are not changed until regenerated.`
      : frequency >= 10
        ? `Maximum ${engineLabel} editing: each segment requests a new continuity-preserving shot every second. A 5-second segment gets four cuts.`
        : `${engineLabel} scales this ${frequency}/10 editing intensity to each segment's exact duration. LTX writes the cuts in ordinary language; MiniMax uses its structured CUT TO format. Existing prompts are not changed until regenerated.`;
    refreshSetupPanelSummaries();
  }

  function refreshImageShotInfo() {
    const preset = imageShotFlowPresetForMode(state.imageShotFlow);
    const count = preset.sequence?.length || 0;
    imageShotInfo.textContent = state.imageShotFlow === "off"
      ? preset.description
      : `${preset.description} Cycles through ${count} still compositions and only fills blank shot fields.`;
    refreshSetupPanelSummaries();
  }

  function refreshImageAestheticInfo() {
    const preset = imageAestheticPresetForMode(state.imageAesthetic);
    imageAestheticInfo.textContent = `${preset.description} Used as still-image aesthetic guidance for Image Prep.`;
    refreshSetupPanelSummaries();
  }

  function refreshImageWorldStyleInfo() {
    const labels = {
      natural: "Naturalistic world; surreal details appear only when the scene requires them.",
      surreal_subject: "The setting stays believable while the subject receives the strongest surreal treatment.",
      balanced_surreal: "Subject and environment are both visibly surreal while remaining spatially readable.",
      full_surreal: "Every visible layer follows dream logic—including environment, background, architecture, lighting, perspective, props, subject, and materials.",
      abstract: "A strongly nonliteral world built from symbolic form, impossible space, expressive material, color, and light.",
      custom: "Your custom direction is the primary whole-frame visual contract.",
    };
    imageWorldStyleInfo.textContent = labels[imageWorldStyleSelect.value] || labels.natural;
    imageCustomStyleInfo.textContent = imageCustomStyleInput.value.trim()
      ? "This custom direction is added to the selected preset and applies to the entire frame."
      : "Optional. Enter your complete style idea; select Fully custom when it should be the primary direction.";
    refreshSetupPanelSummaries();
  }

  function refreshConsistencyInfo() {
    consistencyInfo.textContent = state.globalConsistencyPhrase
      ? `${promptRunnerName()} will incorporate this phrase into every generated prompt while keeping the wording as intact as the scene allows.`
      : `Optional phrase ${promptRunnerName()} should preserve across every prompt, such as makeup, styling, texture, wardrobe detail, or visual motif.`;
    refreshSetupPanelSummaries();
  }

  function refreshPerformanceInfo() {
    const preset = performancePresetForMode(state.performanceStyle);
    const presetDescription = preset.description || preset.direction || preset.label || "Performance guidance";
    performanceInfo.textContent = state.performanceStyle
      ? `${presetDescription} Used by ${promptRunnerName()}/GPT for scenes without a per-scene ${isIdLoraMode ? "acting" : "performance"} style.`
      : `${presetDescription} Pick a style here to use it as the default for blank scenes.`;
    refreshSetupPanelSummaries();
  }

  function refreshCharacterSpeedInfo() {
    characterSpeedValue.textContent = storyboardSpeedLabel(state.characterMotionSpeed, "character");
    characterSpeedInfo.textContent = storyboardSpeedGuidance(state.characterMotionSpeed, "character");
    refreshSetupPanelSummaries();
  }

  function refreshFacialInfo() {
    const preset = facialPresetForMode(state.facialPerformance);
    facialInfo.textContent = state.facialPerformance
      ? `${preset.description} Used by ${promptRunnerName()}/GPT for scenes without a per-scene facial performance preset.`
      : `${preset.description} Pick a preset here to use it as the default for blank scenes.`;
    facialCustomInfo.textContent = state.facialPerformanceCustom
      ? "Custom facial text is appended to the selected preset, or used directly when Custom is selected."
      : "Optional custom wording for eyes, brows, cheeks, jaw, mouth behavior, emotion, and blinking.";
    refreshSetupPanelSummaries();
  }

  function applyCameraFlow({ overwrite = false } = {}) {
    if (state.mode !== "image_to_video_prep") {
      createToast("Auto camera flow is only available in Video Prep.");
      return;
    }
    const profileKey = state.cameraFlow || "balanced";
    if (profileKey === "off") {
      createToast("Auto camera flow is off.");
      return;
    }
    let previousMotion = "";
    let changed = 0;
    state.scenes.forEach((scene, index) => {
      const entry = cameraFlowEntryForScene(profileKey, index, previousMotion);
      if (!entry) return;
      const hadShot = Boolean(String(scene.shot_type || "").trim());
      const hadCamera = Boolean(String(scene.camera_motion || "").trim());
      if ((overwrite || !hadShot) && entry.shot) {
        scene.shot_type = entry.shot;
        changed += 1;
      }
      if ((overwrite || !hadCamera) && entry.camera) {
        scene.camera_motion = entry.camera;
        changed += 1;
      }
      previousMotion = String(scene.camera_motion || entry.camera || previousMotion);
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Auto camera flow replaced ${changed} field${changed === 1 ? "" : "s"}.` : "No camera fields were changed.");
    } else {
      createToast(changed ? `Auto camera flow filled ${changed} blank field${changed === 1 ? "" : "s"}.` : "No blank shot or camera fields needed filling.");
    }
  }

  function applyImageShotFlow({ overwrite = false } = {}) {
    if (state.mode === "image_to_video_prep") {
      createToast("Still shot flow is only available in Image Prep.");
      return;
    }
    const profileKey = state.imageShotFlow || "intimate";
    if (profileKey === "off") {
      createToast("Still shot flow is off.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene, index) => {
      const sequence = imageShotFlowPresetForMode(profileKey).sequence || [];
      const shot = sequence[index % sequence.length] || "";
      if (!shot) return;
      if (!overwrite && String(scene.shot_type || "").trim()) return;
      scene.shot_type = shot;
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Still shot flow replaced ${changed} scene${changed === 1 ? "" : "s"}.` : "No shot fields were changed.");
    } else {
      createToast(changed ? `Still shot flow filled ${changed} blank scene${changed === 1 ? "" : "s"}.` : "No blank shot fields needed filling.");
    }
  }

  function applyImageAesthetic({ overwrite = false } = {}) {
    if (state.mode === "image_to_video_prep") {
      createToast("Image aesthetic is only available in Image Prep.");
      return;
    }
    const preset = imageAestheticPresetForMode(state.imageAesthetic);
    const value = String(preset.description || "").trim();
    if (!value) {
      createToast("Choose an image aesthetic first.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene) => {
      const existing = String(scene.motion_summary || "");
      const hasAesthetic = existing.split(/\r?\n/).some((line) => line.trim().toLowerCase().startsWith("image aesthetic:"));
      if (!overwrite && hasAesthetic) return;
      scene.motion_summary = replaceLabeledPlanningLine(existing, "Image aesthetic", value);
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Image aesthetic replaced ${changed} scene${changed === 1 ? "" : "s"}.` : "No image aesthetic notes were changed.");
    } else {
      createToast(changed ? `Image aesthetic filled ${changed} scene${changed === 1 ? "" : "s"}.` : "No blank image aesthetic notes needed filling.");
    }
  }

  function applyPerformanceStyle({ overwrite = false } = {}) {
    const value = String(state.performanceStyle || "").trim();
    if (!value) {
      createToast(isIdLoraMode ? "Choose a global acting style first." : "Choose a global performance style first.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene) => {
      if (!overwrite && String(scene.performance_style || "").trim()) return;
      scene.performance_style = value;
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `${isIdLoraMode ? "Acting" : "Performance"} style replaced ${changed} scene${changed === 1 ? "" : "s"}.` : `No ${isIdLoraMode ? "acting" : "performance"} style fields were changed.`);
    } else {
      createToast(changed ? `${isIdLoraMode ? "Acting" : "Performance"} style filled ${changed} blank scene${changed === 1 ? "" : "s"}.` : `No blank ${isIdLoraMode ? "acting" : "performance"} style fields needed filling.`);
    }
  }

  function applyFacialPerformance({ overwrite = false } = {}) {
    const value = String(state.facialPerformance || "").trim();
    const custom = String(state.facialPerformanceCustom || "").trim();
    if (!value && !custom) {
      createToast("Choose a global facial performance preset or enter custom facial text first.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene) => {
      const hasPreset = String(scene.facial_performance || "").trim();
      const hasCustom = String(scene.facial_performance_custom || "").trim();
      if (!overwrite && (hasPreset || hasCustom)) return;
      scene.facial_performance = value;
      scene.facial_performance_custom = custom;
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Facial performance replaced ${changed} scene${changed === 1 ? "" : "s"}.` : "No facial performance fields were changed.");
    } else {
      createToast(changed ? `Facial performance filled ${changed} blank scene${changed === 1 ? "" : "s"}.` : "No blank facial performance fields needed filling.");
    }
  }
  function refreshFxInfo() {
    const preset = storyboardFxPreset(state.fxPreset);
    const custom = state.fxPreset === "custom" ? normalizeStoryboardCustomFxJson(state.fxCustomJson) : null;
    fxInfo.textContent = state.fxPreset === ""
      ? preset.description
      : custom
        ? `${custom.label}: ${custom.cues.length} custom cue${custom.cues.length === 1 ? "" : "s"}. The Builder injects one cue into each finished timestamped shot after ${promptRunnerName()} returns.`
        : `${preset.description} The Builder injects one cue into each finished timestamped shot after ${promptRunnerName()} returns.`;
    fxCustomControls.style.display = state.fxPreset === "custom" ? "flex" : "none";
    refreshSetupPanelSummaries();
  }

  return {
    applyCameraFlow, applyFacialPerformance, applyImageAesthetic, applyImageShotFlow, applyPerformanceStyle,
    openCustomCameraFlowDialog, refreshCameraFlowInfo, refreshCameraSpeedInfo, refreshCharacterSpeedInfo,
    refreshConsistencyInfo, refreshCutFrequencyInfo, refreshFacialInfo, refreshFxInfo,
    refreshImageAestheticInfo, refreshImageShotInfo, refreshImageWorldStyleInfo, refreshPerformanceInfo,
    refreshSetupPanelSummaries,
  };
}
