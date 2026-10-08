import {
  createStoryboardProgressWindow,
  createToast,
  makeButton,
  makeGroupedSelect,
  makeInput,
  makeMultiSelect,
  makeSelect,
  makeTextarea,
  replaceLabeledPlanningLine,
  sortGroupsAlphabetically,
  sortOptionsAlphabetically,
} from "./controls.mjs";
import { videoPromptTypeHint } from "./prompt_generation.mjs";
import {
  chooseStoryboardImageFile,
  makeStoryboardImageUrl,
  readStoryboardImageFile,
  referenceChipHtml,
  storyboardSubjectNamesFromRefs,
} from "./references.mjs";
import {
  normalizeStoryboardMiniMaxH3AudioMode,
  normalizeStoryboardMiniMaxH3Mode,
  normalizeStoryboardPerformanceMode,
  normalizeStoryboardSpeakerAssignments,
  normalizeVideoPromptOrigin,
  slimSceneForRequest,
} from "./scenes.mjs";
import {
  CAMERA_MOTION_GROUPS,
  CHARACTER_MOTION_GROUPS,
  IMAGE_SHOT_TYPES,
  STILL_CAMERA_STYLE_GROUPS,
  VIDEO_SHOT_TYPES,
} from "./shot_presets.mjs";
import { MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS, MINIMAX_VIDEO_STYLE_PRESETS } from "./video_style.mjs";

export function createSceneEditor({
  absorbSceneReferencesIntoCatalog, addStoryboardReferenceFromFile, backdrop, createSceneBeatWithGemma,
  createScenePromptForActiveMode, facialPerformancePresets, isFullyCustomShortFilm, isMiniMaxShortFilmMode,
  performanceStylePresets, promptRunnerGenericName, promptRunnerName, propagateFlfEndStateToNextScene,
  renderTable, saveStoryboard, sceneFocus, state, syncReferenceMappingsToVideoCreator,
  syncStoryLayerFromInputs,
}) {
  function openSceneEditor(scene) {
    const isVideoPrepMode = state.mode === "image_to_video_prep";
    const isImagePrepMode = !isVideoPrepMode;
    const editorSceneIndex = state.scenes.findIndex((item) => item.id === scene.id);
    const inheritedFlfStart = editorSceneIndex > 0 ? String(state.scenes[editorSceneIndex - 1]?.flf_end_state || "").trim() : "";
    if ((state.videoPromptType === "flf" || scene.video_prompt_type === "flf") && inheritedFlfStart) scene.flf_start_state = inheritedFlfStart;
    absorbSceneReferencesIntoCatalog([scene]);
    const editorBackdrop = document.createElement("div");
    editorBackdrop.style.cssText = "position:fixed;inset:0;z-index:100012;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:18px;";
    const editor = document.createElement("div");
    editor.style.cssText = "width:min(1420px,calc(100vw - 42px));max-height:calc(100vh - 42px);overflow:auto;border:1px solid #0e7490;border-radius:16px;background:linear-gradient(135deg,#07111f,#0f172a 46%,#071827);color:#f8fafc;box-shadow:0 28px 90px rgba(0,0,0,.68);padding:18px;display:flex;flex-direction:column;gap:12px;";
    const label = makeInput(scene.label, "Scene label");
    const lyricSection = makeInput(scene.lyric_section || "", "Verse 1, Chorus, Bridge, Outro...");
    const lyrics = makeTextarea(scene.lyrics, "Lyrics, script, or beat for this scene...", 4);
    const storyBeat = makeTextarea(scene.story_beat || "", "Scene story beat for this scene...", 4);
    const flfStartState = makeTextarea(scene.flf_start_state || "", "What must be visible in this scene's first frame...", 3);
    const flfTransformation = makeTextarea(scene.flf_transformation || "", "What changes continuously between the two frames...", 3);
    const flfEndState = makeTextarea(scene.flf_end_state || "", "What must be visible in this scene's last frame...", 3);
    const flfCarryForward = makeTextarea(scene.flf_carry_forward || "", "Continuity details the next scene should inherit...", 3);
    if ((state.videoPromptType === "flf" || scene.video_prompt_type === "flf") && editorSceneIndex > 0) {
      flfStartState.readOnly = true;
      flfStartState.title = "Automatically inherited from the previous scene's end-frame state.";
      flfStartState.style.opacity = "0.78";
    }
    const summary = makeTextarea(scene.prompt_summary, "Image prompt summary...", 3);
    const motion = makeTextarea(
      scene.motion_summary,
      isImagePrepMode ? "Still photography notes..." : "Custom motion, camera, action, or LLM direction...",
      3,
    );
    const cameraGroups = isImagePrepMode ? STILL_CAMERA_STYLE_GROUPS : CAMERA_MOTION_GROUPS;
    const cameraMotionOptions = cameraGroups.flatMap((group) => group.options || []);
    const cameraMotionValue = scene.camera_motion || cameraMotionOptions.find((item) => String(scene.motion_summary || "").toLowerCase().includes(item.toLowerCase())) || "";
    const cameraMotionPreset = makeGroupedSelect(sortGroupsAlphabetically(cameraGroups), cameraMotionValue);
    const customCameraMotion = makeInput(scene.camera_motion || "", isImagePrepMode ? "Custom still camera style" : "Custom camera motion");
    const characterMotionOptions = CHARACTER_MOTION_GROUPS.flatMap((group) => group.options || []);
    const characterMotionValue = scene.character_motion || characterMotionOptions.find((item) => String(scene.motion_summary || "").toLowerCase().includes(item.toLowerCase())) || "";
    const characterMotionPreset = makeGroupedSelect(sortGroupsAlphabetically(CHARACTER_MOTION_GROUPS), characterMotionValue);
    const customCharacterMotion = makeInput(scene.character_motion || "", "Custom character motion");
    const performanceStyle = makeSelect(sortOptionsAlphabetically(performanceStylePresets), scene.performance_style || "");
    const videoStyle = makeSelect(sortOptionsAlphabetically(MINIMAX_VIDEO_STYLE_PRESETS), state.videoStyle || scene.video_style || "");
    videoStyle.disabled = Boolean(state.videoStyle);
    videoStyle.title = state.videoStyle ? "The global Video style is required for every eligible scene." : "Choose a style for this scene.";
    const videoStyleCustom = makeTextarea(
      state.videoStyle === "custom" ? state.videoStyleCustom : (scene.video_style_custom || state.videoStyleCustom || ""),
      "Type the exact style wording that must appear unchanged in this scene's prompt...",
      3,
    );
    videoStyleCustom.disabled = Boolean(state.videoStyle);
    const temporalEffectOverride = makeSelect([
      { value: "global", label: "Use global temporal effect" },
      { value: "off", label: "Off for this scene" },
      ...sortOptionsAlphabetically([{ value: "", label: "" }, ...MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS.filter((item) => item.value)]).slice(1).map((item) => ({ value: item.value, label: item.label })),
    ], scene.temporal_world_effect_override || "global");
    const temporalEffectCustom = makeTextarea(scene.temporal_world_effect_custom || "", "Exact custom temporal behavior for only this scene...", 3);
    const facialPerformance = makeSelect(sortOptionsAlphabetically(facialPerformancePresets), scene.facial_performance || "");
    const facialPerformanceCustom = makeTextarea(scene.facial_performance_custom || "", "Optional custom facial expression/movement text for this scene...", 3);
    const includeMicLabel = document.createElement("label");
    includeMicLabel.style.cssText = "display:flex;align-items:center;gap:8px;border:1px solid #334155;border-radius:8px;background:#0f172a;color:#cbd5e1;padding:9px 10px;font-size:12px;font-weight:900;";
    const includeMic = document.createElement("input");
    includeMic.type = "checkbox";
    includeMic.checked = Boolean(scene.include_microphone);
    includeMicLabel.append(includeMic, document.createTextNode("Include microphone in prompt"));
    const noCharacterLabel = document.createElement("label");
    noCharacterLabel.style.cssText = includeMicLabel.style.cssText;
    const noCharacterInput = document.createElement("input");
    noCharacterInput.type = "checkbox";
    noCharacterInput.checked = Boolean(scene.no_character_present);
    noCharacterLabel.append(noCharacterInput, document.createTextNode("No character present"));
    const miniMaxProject = state.projectVideoEngine === "minimax_h3";
    const videoPromptType = makeSelect(miniMaxProject ? [
      { value: "text_to_video", label: "MiniMax H3 — Text to Video" },
      { value: "image_to_video", label: "MiniMax H3 — Image to Video" },
      { value: "reference_to_video", label: "MiniMax H3 — Reference to Video" },
      { value: "video_to_video", label: "MiniMax H3 — Video to Video" },
    ] : [
      { value: "i2v", label: "Image to Video" },
      { value: "id_lora", label: "ID-LoRA I2V" },
      { value: "t2v", label: "Text to Video" },
      { value: "rtv", label: "Reference to Video" },
      { value: "ingredients", label: "Ingredients to Video" },
    ], miniMaxProject ? normalizeStoryboardMiniMaxH3Mode(scene.minimax_h3_mode) : (scene.video_prompt_type || "i2v"));
    const subjects = makeInput((scene.subjects || []).join(", "), "Subjects, comma separated");
    const subjectDetails = makeTextarea(
      (Array.isArray(scene.subject_refs) ? scene.subject_refs : [])
        .map((subject) => `${subject.name || "Subject"}: ${subject.description || ""}`.trim())
        .filter(Boolean)
        .join("\n\n"),
      "Character descriptions from Reference Builder...",
      4,
    );
    const setting = makeInput(scene.setting || scene.location_ref?.description || scene.location_ref?.name || "", "Location / setting");
    const locationDetails = makeTextarea(
      scene.location_ref
        ? `${scene.location_ref.name || "Location"}: ${scene.location_ref.description || ""}`.trim()
        : "",
      "Location description from Reference Builder...",
      4,
    );
    const shot = makeInput(scene.shot_type, "Shot type");
    const shotPreset = makeSelect([{ value: "", label: "Choose a preset..." }, { value: "__custom__", label: "Custom / keep typed value" }], "__custom__");
    const imagePrompt = makeTextarea(scene.image_prompt, "Full text-to-image prompt...", 7);
    const videoPrompt = makeTextarea(scene.video_prompt, "Full video prompt...", 7);
    let editorVideoPromptOrigin = normalizeVideoPromptOrigin(scene.video_prompt_origin);
    videoPrompt.addEventListener("input", () => {
      editorVideoPromptOrigin = "manual";
    });
    const imagePath = makeInput(scene.image_path, "Image path");
    imagePath.type = "hidden";
    let sceneImageData = String(scene.image_data || scene.image_reference_data || "").trim();
    let sceneImageName = String(scene.image_name || scene.image_reference_name || "").trim();
    const startingImageControl = document.createElement("div");
    startingImageControl.dataset.vrgdgFileDropZone = "true";
    startingImageControl.style.cssText = "display:grid;grid-template-columns:112px 1fr;gap:12px;align-items:center;border:1px dashed #155e75;border-radius:9px;background:#07111f;padding:10px;cursor:pointer;";
    const startingImagePreview = document.createElement("div");
    startingImagePreview.style.cssText = "width:112px;height:82px;border:1px solid #334155;border-radius:7px;background:#020617 center/contain no-repeat;display:grid;place-items:center;color:#64748b;font-size:11px;text-align:center;overflow:hidden;";
    const startingImageDetails = document.createElement("div");
    startingImageDetails.style.cssText = "display:flex;flex-direction:column;gap:7px;min-width:0;";
    const startingImageButton = makeButton("Upload Image", "primary");
    startingImageButton.type = "button";
    const startingImageStatus = document.createElement("div");
    startingImageStatus.style.cssText = "font-size:11px;color:#94a3b8;overflow-wrap:anywhere;";
    const startingImageNote = document.createElement("div");
    startingImageNote.style.cssText = "font-size:11px;line-height:1.4;color:#cbd5e1;";
    startingImageNote.textContent = "This is the finished starting frame for this I2V scene. Storyboard Builder analyzes it when writing the video prompt so the motion matches what is actually visible.";
    startingImageDetails.append(startingImageButton, startingImageStatus, startingImageNote);
    startingImageControl.append(startingImagePreview, startingImageDetails, imagePath);
    const refreshStartingImage = () => {
      const source = sceneImageData
        ? (sceneImageData.startsWith("data:") ? sceneImageData : `data:image/png;base64,${sceneImageData}`)
        : (imagePath.value.trim() ? makeStoryboardImageUrl(imagePath.value.trim()) : "");
      startingImagePreview.style.backgroundImage = source ? `url("${source.replace(/"/g, "%22")}")` : "none";
      startingImagePreview.textContent = source ? "" : "Drop image here";
      startingImageStatus.textContent = source
        ? `Selected: ${sceneImageName || imagePath.value.trim().split(/[\\/]/).pop() || "uploaded image"}`
        : "No starting image selected. Drop a PNG, JPG, or WEBP here, or click Upload Image.";
    };
    const useStartingImageFile = async (file) => {
      if (!file) return;
      sceneImageData = await readStoryboardImageFile(file);
      sceneImageName = file.name || "starting_frame.png";
      imagePath.value = "";
      refreshStartingImage();
    };
    startingImageButton.onclick = async (event) => {
      event.preventDefault();
      event.stopPropagation();
      await useStartingImageFile(await chooseStoryboardImageFile());
    };
    startingImageControl.onclick = async (event) => {
      if (event.target === startingImageButton) return;
      await useStartingImageFile(await chooseStoryboardImageFile());
    };
    startingImageControl.addEventListener("dragover", (event) => {
      if (!event.dataTransfer?.files?.length && !Array.from(event.dataTransfer?.types || []).includes("Files")) return;
      event.preventDefault();
      event.stopPropagation();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      startingImageControl.style.borderColor = "#22d3ee";
    });
    startingImageControl.addEventListener("dragleave", () => { startingImageControl.style.borderColor = "#155e75"; });
    startingImageControl.addEventListener("drop", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      startingImageControl.style.borderColor = "#155e75";
      await useStartingImageFile(Array.from(event.dataTransfer?.files || []).find((file) => String(file.type || "").startsWith("image/")) || null);
    });
    refreshStartingImage();
    const triggerPhrase = makeInput(scene.trigger_phrase || "", "Optional scene trigger phrase");
    const triggerPosition = makeSelect([
      { value: "start", label: "Add trigger to start" },
      { value: "end", label: "Add trigger to end" },
    ], scene.trigger_position || "start");
    const notes = makeTextarea(scene.notes, "Extra planning notes...", 3);
    const timelineNote = makeTextarea(scene.timeline_note || "", "Director note shown on the timeline...", 3);
    const audioDirection = makeTextarea(scene.audio_direction || "", "Exact ambience, sound effects, silence, breathing, or audio behavior for this scene...", 4);
    const continuityDirection = makeTextarea(scene.continuity || "", "Exact identity, wardrobe, prop, location, screen-direction, and spatial continuity requirements...", 4);
    const selectedSubjectIds = scene.no_character_present ? [] : (Array.isArray(scene.subject_refs) ? scene.subject_refs : [])
      .map((ref) => String(ref?.id || ""))
      .filter(Boolean);
    const subjectSelect = makeMultiSelect(
      state.referenceBuilder.subjects.map((subject) => ({ value: subject.id, label: subject.name })),
      selectedSubjectIds,
    );
    const savedLocationId = String(scene.location_ref?.id || "");
    const locationOptions = [
      { value: "", label: "Unassigned" },
      ...state.referenceBuilder.locations.map((location) => ({ value: location.id, label: location.name })),
    ];
    const locationSelect = makeSelect(locationOptions, savedLocationId);
    const field = (name, control) => {
      const wrap = document.createElement("label");
      wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:800;color:#cbd5e1;";
      wrap.textContent = name;
      wrap.append(control);
      return wrap;
    };
    const section = (number, title, content, { collapsible = false, open = false } = {}) => {
      const wrap = collapsible ? document.createElement("details") : document.createElement("section");
      if (collapsible) wrap.open = open;
      wrap.style.cssText = "border:1px solid #1f3b46;border-radius:10px;background:linear-gradient(135deg,rgba(8,51,68,.34),rgba(15,23,42,.9));padding:14px;box-shadow:inset 0 1px 0 rgba(255,255,255,.03);";
      const heading = collapsible ? document.createElement("summary") : document.createElement("div");
      heading.style.cssText = "display:flex;align-items:center;gap:12px;color:#e2e8f0;font-size:20px;font-weight:900;cursor:pointer;list-style:none;";
      const badge = document.createElement("span");
      badge.textContent = String(number);
      badge.style.cssText = "width:30px;height:30px;border-radius:999px;background:#155e75;color:#cffafe;display:grid;place-items:center;font-size:15px;flex:0 0 auto;";
      const text = document.createElement("span");
      text.textContent = title;
      heading.append(badge, text);
      if (collapsible) {
        const chevron = document.createElement("span");
        chevron.textContent = "⌄";
        chevron.style.cssText = "margin-left:auto;color:#cbd5e1;font-size:22px;";
        heading.append(chevron);
      }
      const body = document.createElement("div");
      body.style.cssText = "margin-top:12px;";
      body.append(content);
      wrap.append(heading, body);
      return wrap;
    };
    const twoCol = () => {
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:16px 28px;";
      return grid;
    };
    const threeCol = () => {
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px 28px;";
      return grid;
    };
    const iconField = (icon, control) => {
      const row = document.createElement("div");
      row.style.cssText = "display:grid;grid-template-columns:44px 1fr;gap:8px;align-items:center;";
      const ico = document.createElement("div");
      ico.textContent = icon;
      ico.style.cssText = "width:42px;height:42px;border:1px solid #155e75;border-radius:8px;background:#083344;color:#22d3ee;display:grid;place-items:center;font-size:20px;";
      row.append(ico, control);
      return row;
    };
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const videoTypeHint = document.createElement("div");
    videoTypeHint.style.cssText = "grid-column:1/-1;border:1px solid #334155;border-radius:8px;background:#0f172a;color:#cbd5e1;font-size:12px;line-height:1.45;padding:9px 10px;";
    const shotPresetField = field("Shot type preset", shotPreset);
    const shotCustomField = field("Custom shot type", shot);
    const cameraMotionField = field(isImagePrepMode ? "Still camera style preset" : "Camera motion preset", cameraMotionPreset);
    const characterMotionField = field("Character motion preset", characterMotionPreset);
    const customCharacterMotionField = field("Custom character motion", customCharacterMotion);
    const performanceStyleField = field("Performance / song style", performanceStyle);
    const videoStyleField = field(state.videoStyle ? "Video aesthetic — global and required" : "Video aesthetic", videoStyle);
    const videoStyleCustomField = field("Custom style wording — copied exactly", videoStyleCustom);
    const temporalEffectField = field("Temporal / world effect", temporalEffectOverride);
    const temporalEffectCustomField = field("Custom temporal wording", temporalEffectCustom);
    const facialPerformanceField = field("Facial performance", facialPerformance);
    const facialPerformanceCustomField = field("Custom facial performance", facialPerformanceCustom);
    const imagePathField = field("Starting image", startingImageControl);
    const motionField = field(isImagePrepMode ? "Still photography notes" : "Motion Notes / LLM Direction", motion);
    const t2iPromptField = field("T2I prompt", imagePrompt);
    if (isVideoPrepMode) {
      grid.append(field("Video prompt type", videoPromptType), videoStyleField, videoStyleCustomField, field("Setting", setting), videoTypeHint, field("Subjects", subjects), performanceStyleField, facialPerformanceField, facialPerformanceCustomField, includeMicLabel, noCharacterLabel, shotPresetField, shotCustomField, cameraMotionField, characterMotionField, customCharacterMotionField, imagePathField, field("Scene trigger phrase", triggerPhrase), field("Trigger placement", triggerPosition));
    } else {
      grid.append(field("Setting", setting), field("Subjects", subjects), performanceStyleField, facialPerformanceField, facialPerformanceCustomField, includeMicLabel, noCharacterLabel, shotPresetField, shotCustomField, cameraMotionField, field("Scene trigger phrase", triggerPhrase), field("Trigger placement", triggerPosition));
    }
    const referenceGrid = document.createElement("div");
    referenceGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:16px 28px;";
    if (state.referenceBuilder.subjects.length || state.referenceBuilder.locations.length) {
      referenceGrid.append(
        field("Reference Builder characters", subjectSelect),
        field("Reference Builder location", locationSelect),
      );
    } else {
      referenceGrid.innerHTML = `<div style="grid-column:1/-1;color:#94a3b8;font-size:12px;">No Reference Builder subjects or locations are available yet. Add them in Reference Builder first, then reopen Storyboard Builder.</div>`;
    }
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;";
    const gemmaBeat = makeButton(`${promptRunnerGenericName()} Story Beat`, "primary");
    const gemma = makeButton("Generate Prompt", "purple");
    const cancel = makeButton("Cancel");
    const apply = makeButton("Save Scene Card", "primary");
    actions.append(cancel, gemmaBeat, gemma, apply);
    if (isFullyCustomShortFilm()) {
      gemmaBeat.style.display = "none";
      gemma.title = "Formats the manually entered scene card into a MiniMax H3 prompt without inventing or rewriting scene content.";
    }
    const closeEditor = makeButton("×");
    closeEditor.style.cssText += "font-size:26px;line-height:1;width:44px;height:44px;padding:0;border-radius:8px;";
    const header = document.createElement("div");
    header.style.cssText = "display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center;";
    const headerIcon = document.createElement("div");
    headerIcon.textContent = "▣";
    headerIcon.style.cssText = "width:54px;height:54px;border-radius:14px;background:#164e63;color:#67e8f9;display:grid;place-items:center;font-size:28px;";
    const headerText = document.createElement("div");
    headerText.innerHTML = `<div style="font-size:28px;font-weight:900;color:#f8fafc;">Edit Scene Card</div><div style="color:#cbd5e1;margin-top:3px;">${isVideoPrepMode ? "Define the details for this scene to generate a rich video prompt." : "Define the details for this scene to generate a rich text-to-image prompt."}</div>`;
    header.append(headerIcon, headerText, closeEditor);

    const basicsGrid = twoCol();
    basicsGrid.append(field("Scene label", label), field("Lyric section", lyricSection), field("Scene / lyrics", lyrics), field("Scene story beat", storyBeat));
    if (isVideoPrepMode) {
      basicsGrid.append(field("Prompt mode", iconField("▣", videoPromptType)), videoStyleField, videoStyleCustomField, temporalEffectField, temporalEffectCustomField, field("Performance / song style", performanceStyle), field("Facial performance", facialPerformance), field("Custom facial performance", facialPerformanceCustom), includeMicLabel, noCharacterLabel, videoTypeHint);
    } else {
      const imagePromptType = makeInput("Text to Image", "Text to Image");
      imagePromptType.readOnly = true;
      basicsGrid.append(field("Image prompt type", iconField("▣", imagePromptType)), field("Performance / song style", performanceStyle), field("Facial performance", facialPerformance), field("Custom facial performance", facialPerformanceCustom), includeMicLabel, noCharacterLabel);
    }

    const addSubject = makeButton("+ Add subject");
    addSubject.style.background = "#0f172a";
    addSubject.style.borderStyle = "dashed";
    const addLocation = makeButton("+ Add location");
    addLocation.style.background = "#0f172a";
    addLocation.style.borderStyle = "dashed";
    const subjectChip = document.createElement("div");
    const locationChip = document.createElement("div");
    const refreshReferenceChips = () => {
      const selectedSubjects = Array.from(subjectSelect.selectedOptions).map((option) => state.referenceBuilder.subjects.find((subject) => subject.id === option.value)).filter(Boolean);
      const selectedLocation = state.referenceBuilder.locations.find((location) => location.id === locationSelect.value) || (locationSelect.value && scene.location_ref?.id === locationSelect.value ? scene.location_ref : null);
      subjectChip.innerHTML = noCharacterInput.checked
        ? `<span style="color:#fca5a5;">No character present</span>`
        : selectedSubjects.length
        ? selectedSubjects.map((ref) => referenceChipHtml(ref, "Subject")).join("")
        : `<span style="color:#94a3b8;">No subject selected</span>`;
      locationChip.innerHTML = selectedLocation
        ? referenceChipHtml(selectedLocation, "Location")
        : `<span style="color:#94a3b8;">No location selected</span>`;
    };
    const refreshNoCharacterState = () => {
      subjectSelect.disabled = Boolean(noCharacterInput.checked);
      subjects.disabled = Boolean(noCharacterInput.checked);
      subjectDetails.disabled = Boolean(noCharacterInput.checked);
      if (noCharacterInput.checked) {
        for (const option of subjectSelect.options) option.selected = false;
        subjects.value = "";
        subjectDetails.value = "";
      }
      refreshReferenceChips();
    };
    const referencesGrid = twoCol();
    const subjectPick = document.createElement("div");
    subjectPick.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:12px;align-items:end;";
    subjectPick.append(field("Subject(s)", subjectChip), addSubject);
    const locationPick = document.createElement("div");
    locationPick.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:12px;align-items:end;";
    locationPick.append(field("Setting / Location", locationChip), addLocation);
    referencesGrid.append(subjectPick, locationPick, ...Array.from(referenceGrid.children));
    refreshReferenceChips();

    const speakerAssignmentEnabled = isVideoPrepMode
      && miniMaxProject
      && normalizeStoryboardPerformanceMode(scene.performance_mode || state.performanceMode) === "speaking"
      && normalizeStoryboardMiniMaxH3AudioMode(scene.minimax_h3_audio_mode || state.miniMaxH3AudioMode) === "built_in_audio";
    let speakerAssignments = normalizeStoryboardSpeakerAssignments(scene.speaker_assignments);
    const speakerAssignmentWrap = document.createElement("div");
    speakerAssignmentWrap.style.cssText = "display:flex;flex-direction:column;gap:9px;";
    const speakerAssignmentNote = document.createElement("div");
    speakerAssignmentNote.style.cssText = "border:1px solid #155e75;border-radius:8px;background:#07111f;color:#cbd5e1;padding:9px 10px;font-size:12px;line-height:1.45;";
    speakerAssignmentNote.textContent = "Drag cues into the exact speaking order. The same character can have multiple turns. Speaker choices come only from this scene’s mapped Reference Builder characters.";
    const speakerAssignmentRows = document.createElement("div");
    speakerAssignmentRows.style.cssText = "display:flex;flex-direction:column;gap:8px;";
    const addSpeakerAssignment = makeButton("Add Dialogue Cue", "primary");
    const mappedSpeakerOptions = () => {
      const selectedIds = Array.from(subjectSelect.selectedOptions).map((option) => String(option.value || "")).filter(Boolean);
      const selected = selectedIds
        .map((id) => state.referenceBuilder.subjects.find((subject) => String(subject.id || "") === id))
        .filter(Boolean);
      const fallback = Array.isArray(scene.subject_refs) ? scene.subject_refs : [];
      return (selected.length ? selected : fallback)
        .filter((subject) => subject && typeof subject === "object")
        .map((subject) => ({ id: String(subject.id || ""), name: String(subject.name || "Character").trim() || "Character" }));
    };
    const syncSpeakerAssignmentLegacy = () => {
      speakerAssignments = normalizeStoryboardSpeakerAssignments(speakerAssignments);
      scene.speaker_assignments = speakerAssignments;
      const filled = speakerAssignments.filter((cue) => cue.text);
      const combined = filled.map((cue) => cue.text).join("\n");
      lyrics.value = combined;
      scene.lyrics = combined;
      scene.lyric_singers = Array.from(new Set(filled.map((cue) => cue.speaker_name).filter(Boolean)));
    };
    const ensureSpeakerAssignments = () => {
      if (speakerAssignments.length) return;
      const speakers = mappedSpeakerOptions();
      const existingLine = String(scene.lyrics || lyrics.value || "").trim();
      if (existingLine) {
        const preferred = String((Array.isArray(scene.lyric_singers) ? scene.lyric_singers[0] : "") || "").trim();
        const speaker = speakers.find((item) => item.name.toLowerCase() === preferred.toLowerCase()) || speakers[0] || { id: "", name: preferred };
        speakerAssignments = normalizeStoryboardSpeakerAssignments([{ speaker_id: speaker.id, speaker_name: speaker.name, text: existingLine }]);
      } else if (speakers.length) {
        speakerAssignments = normalizeStoryboardSpeakerAssignments(speakers.map((speaker) => ({ speaker_id: speaker.id, speaker_name: speaker.name, text: "" })));
      }
      syncSpeakerAssignmentLegacy();
    };
    const renderSpeakerAssignments = () => {
      speakerAssignmentRows.replaceChildren();
      const speakers = mappedSpeakerOptions();
      addSpeakerAssignment.disabled = !speakers.length || Boolean(noCharacterInput.checked);
      ensureSpeakerAssignments();
      if (!speakers.length || noCharacterInput.checked) {
        const empty = document.createElement("div");
        empty.textContent = noCharacterInput.checked
          ? "This scene is marked No character present."
          : "Map one or more Reference Builder characters to this scene first.";
        empty.style.cssText = "border:1px dashed #334155;border-radius:8px;padding:12px;color:#94a3b8;text-align:center;font-size:12px;";
        speakerAssignmentRows.append(empty);
        return;
      }
      let draggedIndex = -1;
      speakerAssignments.forEach((cue, index) => {
        const row = document.createElement("div");
        row.style.cssText = "display:grid;grid-template-columns:34px 34px minmax(160px,.65fr) minmax(300px,1.5fr) 76px;gap:8px;align-items:center;border:1px solid #334155;border-radius:8px;background:#0f172a;padding:8px;";
        row.addEventListener("dragover", (event) => {
          if (draggedIndex < 0 || draggedIndex === index) return;
          event.preventDefault();
          row.style.borderColor = "#22d3ee";
        });
        row.addEventListener("dragleave", () => { row.style.borderColor = "#334155"; });
        row.addEventListener("drop", (event) => {
          event.preventDefault();
          row.style.borderColor = "#334155";
          if (draggedIndex < 0 || draggedIndex === index) return;
          const [moved] = speakerAssignments.splice(draggedIndex, 1);
          speakerAssignments.splice(index, 0, moved);
          syncSpeakerAssignmentLegacy();
          renderSpeakerAssignments();
        });
        const handle = document.createElement("button");
        handle.type = "button";
        handle.textContent = "::";
        handle.title = "Drag to change speaking order";
        handle.draggable = true;
        handle.style.cssText = "height:38px;border:1px solid #334155;border-radius:6px;background:#07111f;color:#67e8f9;font-weight:900;cursor:grab;";
        handle.addEventListener("dragstart", () => { draggedIndex = index; row.style.opacity = ".55"; });
        handle.addEventListener("dragend", () => { draggedIndex = -1; row.style.opacity = ""; });
        const number = document.createElement("div");
        number.textContent = String(index + 1);
        number.style.cssText = "font-weight:900;color:#cffafe;text-align:center;";
        const speakerSelect = makeSelect(speakers.map((speaker) => ({ value: speaker.id, label: speaker.name })), cue.speaker_id);
        if (!speakers.some((speaker) => speaker.id === cue.speaker_id) && cue.speaker_name) {
          speakerSelect.prepend(new Option(`${cue.speaker_name} (not currently mapped)`, cue.speaker_id));
          speakerSelect.value = cue.speaker_id;
        }
        const line = makeInput(cue.text || "", "Exact words this character says...");
        const remove = makeButton("Remove");
        speakerSelect.addEventListener("change", () => {
          const speaker = speakers.find((item) => item.id === speakerSelect.value) || { id: speakerSelect.value, name: speakerSelect.selectedOptions[0]?.textContent || "" };
          cue.speaker_id = speaker.id;
          cue.speaker_name = speaker.name;
          syncSpeakerAssignmentLegacy();
        });
        line.addEventListener("input", () => {
          cue.text = line.value;
          syncSpeakerAssignmentLegacy();
        });
        remove.onclick = () => {
          speakerAssignments.splice(index, 1);
          syncSpeakerAssignmentLegacy();
          renderSpeakerAssignments();
        };
        row.append(handle, number, speakerSelect, line, remove);
        speakerAssignmentRows.append(row);
      });
    };
    addSpeakerAssignment.onclick = () => {
      const speaker = mappedSpeakerOptions()[0];
      if (!speaker) return;
      speakerAssignments.push(...normalizeStoryboardSpeakerAssignments([{ speaker_id: speaker.id, speaker_name: speaker.name, text: "" }]));
      syncSpeakerAssignmentLegacy();
      renderSpeakerAssignments();
    };
    speakerAssignmentWrap.append(speakerAssignmentNote, speakerAssignmentRows, addSpeakerAssignment);
    if (speakerAssignmentEnabled) {
      lyrics.readOnly = true;
      lyrics.title = "This value is built automatically from the ordered Speaker Assignment cues below.";
      lyrics.style.opacity = "0.78";
      renderSpeakerAssignments();
    }

    const motionGrid = isVideoPrepMode ? threeCol() : twoCol();
    if (isVideoPrepMode) {
      motionGrid.append(
        field("Starting shot preset", iconField("▣", shotPreset)),
        field("Camera motion preset", iconField("▣", cameraMotionPreset)),
        field("Character motion preset", iconField("♟", characterMotionPreset)),
        field("Custom starting shot (optional)", shot),
        field("Custom camera motion (optional)", customCameraMotion),
        field("Custom character motion (optional)", customCharacterMotion),
      );
    } else {
      motionGrid.append(
        field("Shot / composition preset", iconField("▣", shotPreset)),
        field("Still camera / photography preset", iconField("▣", cameraMotionPreset)),
        field("Custom shot / composition (optional)", shot),
        field("Custom still camera style (optional)", customCameraMotion),
      );
    }

    const advancedGrid = twoCol();
    if (isVideoPrepMode) {
      advancedGrid.append(field("Prompt summary", summary), motionField, field("Character details", subjectDetails), field("Location details", locationDetails), imagePathField, t2iPromptField, field("Video prompt", videoPrompt));
      if (isMiniMaxShortFilmMode) {
        advancedGrid.append(field("Manual audio / sound direction", audioDirection), field("Manual continuity requirements", continuityDirection));
      }
    } else {
      advancedGrid.append(t2iPromptField, field("Character details", subjectDetails), field("Location details", locationDetails), field("Still photography notes", motion));
    }
    const notesWrap = document.createElement("div");
    notesWrap.append(field("Scene Note (timeline)", timelineNote), field("Planning Notes", notes));
    const flfBeatGrid = twoCol();
    flfBeatGrid.append(
      field(editorSceneIndex > 0 ? "Start-frame state (inherited from previous end)" : "Start-frame state", flfStartState),
      field("Transformation during scene", flfTransformation),
      field("End-frame state", flfEndState),
      field("Carry-forward state", flfCarryForward),
    );
    const flfBeatSection = section(2, "First / Last Frame Endpoint Beat", flfBeatGrid);
    const editorSections = [
      header,
      section(1, "Scene Basics", basicsGrid),
    ];
    if (speakerAssignmentEnabled) editorSections.push(section("2", "Speaker Assignment", speakerAssignmentWrap));
    if (state.videoPromptType === "flf" || scene.video_prompt_type === "flf") editorSections.push(flfBeatSection);
    editorSections.push(
      section(3, "References", referencesGrid),
      section(4, isVideoPrepMode ? "Camera & Motion" : "Shot & Still Camera", motionGrid),
      section(5, "Advanced Options", advancedGrid, { collapsible: true, open: false }),
      section(6, "Notes", notesWrap),
      actions,
    );
    editor.replaceChildren(
      ...editorSections,
    );
    editorBackdrop.append(editor);
    document.body.append(editorBackdrop);
    closeEditor.onclick = () => {
      editorBackdrop.remove();
      if (sceneFocus.only) backdrop.remove();
    };
    const refreshShotPresetForVideoType = () => {
      const type = videoPromptType.value || "i2v";
      const imageToVideoType = type === "i2v" || type === "image_to_video";
      const textToVideoType = type === "t2v" || type === "text_to_video";
      const referenceToVideoType = type === "rtv" || type === "reference_to_video";
      const videoStyleType = !miniMaxProject || textToVideoType || referenceToVideoType;
      const options = isImagePrepMode ? IMAGE_SHOT_TYPES : (imageToVideoType ? VIDEO_SHOT_TYPES : Array.from(new Set([...IMAGE_SHOT_TYPES, ...VIDEO_SHOT_TYPES])));
      const current = shot.value || scene.shot_type || "";
      shotPreset.replaceChildren();
      for (const option of [
        { value: "", label: isImagePrepMode ? "Choose shot / composition preset..." : (imageToVideoType ? "Choose camera/motion preset..." : "Choose starting shot preset...") },
        { value: "__custom__", label: "Custom / keep typed value" },
        ...[...options].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base", numeric: true })).map((item) => ({ value: item, label: item })),
      ]) {
        const item = document.createElement("option");
        item.value = option.value;
        item.textContent = option.label;
        shotPreset.append(item);
      }
      shotPreset.value = options.includes(current) ? current : "__custom__";
      shotPresetField.firstChild.textContent = isImagePrepMode ? "Shot / composition preset" : (imageToVideoType ? "Camera / motion preset" : "Starting shot preset");
      shotCustomField.firstChild.textContent = isImagePrepMode ? "Custom shot / composition" : (imageToVideoType ? "Custom camera / motion" : "Custom starting shot");
      videoTypeHint.textContent = videoPromptTypeHint(type);
      motionField.firstChild.textContent = isImagePrepMode
        ? "Still photography notes"
        : imageToVideoType
          ? "Motion Notes / LLM Direction"
          : referenceToVideoType
            ? "Motion Notes / LLM Direction (with references)"
            : "Motion Notes / LLM Direction";
      t2iPromptField.style.display = isImagePrepMode || (!textToVideoType && !referenceToVideoType) ? "flex" : "none";
      imagePathField.style.display = isVideoPrepMode && !textToVideoType && !referenceToVideoType ? "flex" : "none";
      videoStyleField.style.display = isVideoPrepMode && videoStyleType ? "flex" : "none";
      videoStyleCustomField.style.display = isVideoPrepMode && videoStyleType && videoStyle.value === "custom" ? "flex" : "none";
      temporalEffectField.style.display = isVideoPrepMode ? "flex" : "none";
      temporalEffectCustomField.style.display = isVideoPrepMode && temporalEffectOverride.value === "custom" ? "flex" : "none";
      videoPrompt.style.display = isVideoPrepMode ? "" : "none";
      videoPrompt.placeholder = textToVideoType
        ? "Full text-to-video prompt..."
        : referenceToVideoType
          ? "Full reference-to-video prompt..."
          : type === "video_to_video"
            ? "Full video-to-video prompt..."
          : "Full image-to-video prompt...";
    };
    refreshShotPresetForVideoType();
    videoPromptType.addEventListener("change", refreshShotPresetForVideoType);
    videoStyle.addEventListener("change", refreshShotPresetForVideoType);
    temporalEffectOverride.addEventListener("change", refreshShotPresetForVideoType);
    const refreshSubjectDetailsFromSelection = () => {
      const selectedIds = Array.from(subjectSelect.selectedOptions).map((option) => option.value).filter(Boolean);
      const selectedSubjects = selectedIds
        .map((id) => state.referenceBuilder.subjects.find((subject) => subject.id === id))
        .filter(Boolean);
      subjectDetails.value = selectedSubjects
        .map((subject) => `${subject.name || "Subject"}: ${subject.description || ""}`.trim())
        .filter(Boolean)
        .join("\n\n");
    };
    subjectSelect.addEventListener("change", refreshSubjectDetailsFromSelection);
    subjectSelect.addEventListener("change", refreshReferenceChips);
    noCharacterInput.addEventListener("change", refreshNoCharacterState);
    if (speakerAssignmentEnabled) {
      subjectSelect.addEventListener("change", renderSpeakerAssignments);
      noCharacterInput.addEventListener("change", renderSpeakerAssignments);
    }
    refreshNoCharacterState();
    shotPreset.addEventListener("change", () => {
      if (shotPreset.value && shotPreset.value !== "__custom__") shot.value = shotPreset.value;
    });
    cameraMotionPreset.addEventListener("change", () => {
      const selectedMotion = String(cameraMotionPreset.value || "").trim();
      if (!selectedMotion) return;
      customCameraMotion.value = selectedMotion;
      const currentMotion = String(motion.value || "").trim();
      motion.value = replaceLabeledPlanningLine(currentMotion, isImagePrepMode ? "Still camera style" : "Camera motion", selectedMotion);
    });
    characterMotionPreset.addEventListener("change", () => {
      const selectedMotion = String(characterMotionPreset.value || "").trim();
      if (!selectedMotion) return;
      customCharacterMotion.value = selectedMotion;
      const currentMotion = String(motion.value || "").trim();
      motion.value = replaceLabeledPlanningLine(currentMotion, "Character motion", selectedMotion);
    });
    locationSelect.addEventListener("change", () => {
      const selectedLocation = state.referenceBuilder.locations.find((location) => location.id === locationSelect.value) || (locationSelect.value && scene.location_ref?.id === locationSelect.value ? scene.location_ref : null);
      if (selectedLocation) {
        setting.value = selectedLocation.description || selectedLocation.name || "";
        locationDetails.value = `${selectedLocation.name || "Location"}: ${selectedLocation.description || ""}`.trim();
      } else {
        locationDetails.value = "";
      }
      refreshReferenceChips();
    });
    addSubject.onclick = async () => {
      saveEditorFieldsToScene();
      const ref = await addStoryboardReferenceFromFile("subject", scene);
      if (!ref) return;
      let option = Array.from(subjectSelect.options).find((item) => item.value === ref.id);
      if (!option) {
        option = document.createElement("option");
        option.value = ref.id;
        option.textContent = ref.name;
        subjectSelect.append(option);
      }
      option.selected = true;
      refreshSubjectDetailsFromSelection();
      refreshReferenceChips();
    };
    addLocation.onclick = async () => {
      saveEditorFieldsToScene();
      const ref = await addStoryboardReferenceFromFile("location", scene);
      if (!ref) return;
      let option = Array.from(locationSelect.options).find((item) => item.value === ref.id);
      if (!option) {
        option = document.createElement("option");
        option.value = ref.id;
        option.textContent = ref.name;
        locationSelect.append(option);
      }
      locationSelect.value = ref.id;
      setting.value = ref.description || ref.name || "";
      refreshReferenceChips();
    };
    const saveEditorFieldsToScene = () => {
      scene.label = label.value.trim() || scene.label;
      scene.lyric_section = lyricSection.value.trim();
      scene.lyrics = lyrics.value.trim();
      scene.story_beat = storyBeat.value.trim();
      scene.flf_start_state = flfStartState.value.trim();
      scene.flf_transformation = flfTransformation.value.trim();
      scene.flf_end_state = flfEndState.value.trim();
      scene.flf_carry_forward = flfCarryForward.value.trim();
      propagateFlfEndStateToNextScene(scene);
      if (isVideoPrepMode) scene.prompt_summary = summary.value.trim();
      scene.motion_summary = motion.value.trim();
      if (isVideoPrepMode && miniMaxProject) {
        scene.minimax_h3_mode = normalizeStoryboardMiniMaxH3Mode(videoPromptType.value);
        scene.project_video_engine = "minimax_h3";
      } else if (isVideoPrepMode) {
        scene.video_prompt_type = videoPromptType.value || "i2v";
      }
      scene.no_character_present = Boolean(noCharacterInput.checked);
      scene.subjects = scene.no_character_present ? [] : subjects.value.split(/[,;\n]+/).map((item) => item.trim()).filter(Boolean);
      scene.setting = setting.value.trim();
      if (state.referenceBuilder.subjects.length && !scene.no_character_present) {
        const selectedIds = Array.from(subjectSelect.selectedOptions).map((option) => option.value).filter(Boolean);
        scene.subject_refs = selectedIds
          .map((id) => state.referenceBuilder.subjects.find((subject) => subject.id === id))
          .filter(Boolean);
        const detailsByName = new Map(
          subjectDetails.value
            .split(/\n{2,}/)
            .map((block) => {
              const parts = block.split(":");
              const name = String(parts.shift() || "").trim();
              const description = parts.join(":").trim();
              return name ? [name.toLowerCase(), description] : null;
            })
            .filter(Boolean)
        );
        scene.subject_refs = scene.subject_refs.map((subject) => ({
          ...subject,
          description: detailsByName.get(String(subject.name || "").toLowerCase()) ?? subject.description,
        }));
        if (scene.subject_refs.length) {
          scene.subjects = storyboardSubjectNamesFromRefs(scene.subject_refs);
        }
      } else if (scene.no_character_present) {
        scene.subject_refs = [];
      }
      if (state.referenceBuilder.locations.length) {
        const selectedLocation = state.referenceBuilder.locations.find((location) => location.id === locationSelect.value) || (locationSelect.value && scene.location_ref?.id === locationSelect.value ? scene.location_ref : null);
        const locationParts = String(locationDetails.value || "").split(":");
        const locationName = String(locationParts.shift() || "").trim();
        const locationDescription = locationParts.join(":").trim();
        scene.location_ref = selectedLocation
          ? {
              ...selectedLocation,
              name: locationName || selectedLocation.name,
              description: locationDescription || selectedLocation.description || "",
            }
          : null;
        if (selectedLocation) scene.setting = selectedLocation.description || selectedLocation.name || scene.setting;
        if (scene.location_ref) scene.setting = scene.location_ref.description || scene.location_ref.name || scene.setting;
      }
      scene.shot_type = shot.value.trim();
      scene.camera_motion = customCameraMotion.value.trim() || cameraMotionPreset.value.trim();
      if (isVideoPrepMode) scene.character_motion = customCharacterMotion.value.trim() || characterMotionPreset.value.trim();
      scene.performance_style = performanceStyle.value || "";
      scene.video_style = videoStyle.value || "";
      scene.video_style_custom = videoStyle.value === "custom" ? videoStyleCustom.value.trim() : "";
      scene.temporal_world_effect_override = temporalEffectOverride.value || "global";
      scene.temporal_world_effect_custom = temporalEffectOverride.value === "custom" ? temporalEffectCustom.value.trim() : "";
      scene.facial_performance = facialPerformance.value || "";
      scene.facial_performance_custom = facialPerformanceCustom.value.trim();
      scene.include_microphone = Boolean(includeMic.checked);
      scene.trigger_phrase = triggerPhrase.value.trim();
      scene.trigger_position = triggerPosition.value === "end" ? "end" : "start";
      scene.image_prompt = imagePrompt.value.trim();
      if (isVideoPrepMode) {
        scene.video_prompt = videoPrompt.value.trim();
        scene.video_prompt_origin = editorVideoPromptOrigin;
      }
      if (isVideoPrepMode) {
        scene.image_path = imagePath.value.trim();
        scene.image_data = sceneImageData;
        scene.image_name = sceneImageName;
      }
      if (speakerAssignmentEnabled) {
        scene.minimax_h3_audio_mode = "built_in_audio";
        syncSpeakerAssignmentLegacy();
      }
      scene.notes = notes.value.trim();
      scene.timeline_note = timelineNote.value.trim();
      scene.audio_direction = audioDirection.value.trim();
      scene.continuity = continuityDirection.value.trim();
    };
    cancel.onclick = () => {
      editorBackdrop.remove();
      if (sceneFocus.only) backdrop.remove();
    };
    gemma.onclick = async () => {
      const previous = gemma.textContent;
      gemma.disabled = true;
      const runnerName = promptRunnerName();
      gemma.textContent = `Running ${runnerName}...`;
      const progress = createStoryboardProgressWindow(`Storyboard ${runnerName}`);
      try {
        saveEditorFieldsToScene();
        progress.set(`Preparing ${scene.label || "scene"} for ${runnerName}...`, 12);
        await createScenePromptForActiveMode(scene, { progress, progressPercent: 32 });
        progress.set(state.mode === "image_to_video_prep" ? "Storyboard video prompt ready." : "Storyboard image prompt ready.", 100);
        progress.close(1200);
        imagePrompt.value = scene.image_prompt || "";
        videoPrompt.value = scene.video_prompt || "";
        editorVideoPromptOrigin = normalizeVideoPromptOrigin(scene.video_prompt_origin);
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
      } finally {
        gemma.disabled = false;
        gemma.textContent = previous;
      }
    };
    gemmaBeat.onclick = async () => {
      const previous = gemmaBeat.textContent;
      gemmaBeat.disabled = true;
      gemmaBeat.textContent = "Creating...";
      const progress = createStoryboardProgressWindow("Scene Story Beat");
      try {
        saveEditorFieldsToScene();
        await createSceneBeatWithGemma(scene, { progress, progressPercent: 35 });
        storyBeat.value = scene.story_beat || "";
        flfStartState.value = scene.flf_start_state || "";
        flfTransformation.value = scene.flf_transformation || "";
        flfEndState.value = scene.flf_end_state || "";
        flfCarryForward.value = scene.flf_carry_forward || "";
        progress.set("Scene story beat ready.", 100);
        progress.close(1200);
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
      } finally {
        gemmaBeat.disabled = false;
        gemmaBeat.textContent = previous;
      }
    };
    apply.onclick = async () => {
      if (apply.disabled) return;
      apply.disabled = true;
      closeEditor.disabled = true;
      cancel.disabled = true;
      try {
        saveEditorFieldsToScene();
        if (sceneFocus.only) await saveStoryboard({ throwOnError: true });
        else if (state.onSceneChanged) await state.onSceneChanged(slimSceneForRequest(scene, editorSceneIndex));
        syncReferenceMappingsToVideoCreator();
        syncStoryLayerFromInputs({ notify: true });
        editorBackdrop.remove();
        if (sceneFocus.only) backdrop.remove();
        else renderTable();
      } catch (error) {
        createToast(String(error?.message || error), true);
      } finally {
        apply.disabled = false;
        closeEditor.disabled = false;
        cancel.disabled = false;
      }
    };
  }

  return { openSceneEditor };
}
