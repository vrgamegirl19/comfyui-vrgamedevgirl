import { makeEditorImageUrl, postJson } from "./comfy_api.mjs";
import {
  makeButton,
  makeCheckbox,
  makeField,
  makeInput,
  makeSelect,
  normalizeProjectVideoEngine,
  toast,
} from "./controls.mjs";
import { normalizeMiniMaxH3Mode, normalizeMiniMaxH3Voice } from "./minimax_h3.mjs";
import { normalizeFluxReferenceBuilder, subjectExtraTargetId } from "./reference_data.mjs";
import { createReferenceSceneMapping } from "./reference_scene_mapping.mjs";
import { createReferenceGeneration } from "./reference_generation.mjs";
import { createReferenceImages } from "./reference_images.mjs";
import { createReferenceLocations } from "./reference_locations.mjs";
import { createReferenceSubjects } from "./reference_subjects.mjs";
import { createSceneAssignment } from "./reference_scene_assignment.mjs";

export function createReferenceBuilder({
  activeSegment, advanceZImageSeedAfterRun, allEditableSegments, autoSaveSessionQuiet,
  createDetailedLocationDescriptionWithGemma, createProgressWindow, currentVideoMode,
  describeReferenceImageWithGemma, droppedSceneImageSource, gemmaModelSelect, gemmaRunnerLine,
  i2vGemmaModelSelect, i2vTextGemmaModelSelect, locationExtractionStyleTheme,
  locationScoutLyricsPayloadForGpt, logicalReferenceSubjects, logicalSubjectIdsForScene,
  miniMaxH3ModeForSegment, miniMaxH3SettingsForSegment, normalizeLyricCueMapForSegment,
  openAdvancedLocationScoutGptForRefs, openLocationScoutGptForRefs, openLyricReviewModal, playSingerCueRange,
  projectContextPath, projectInput, projectReferenceBuilderLocationsPath, projectSceneNotesPath, pushHistory,
  render, renderFluxIngredientList, renderNBIngredientList, runImageMemoryCleanupQuiet, saveSession,
  saveZImageSettingsFromPanel, sceneDisplayName, sceneReferenceMapArray, sceneSlotNumber,
  selectedSegmentsForBatch, singerCueRelativePlayheadTime, state, subjectSceneInput, syncInspector,
  syncMiniMaxH3Panel, syncMiniMaxReferenceButtons, syncPerformerInspectorForSegment, syncZImageSettingsPanel,
  t2iTextGemmaModelSelect, textGemmaRunnerPayload, themeStyleInput, zClipPicker, zSeed, zUnetPicker,
  zVaePicker,
}) {
  function openFluxReferenceBuilderModal(options = {}) {
    const focusedSection = ["subjects", "locations", "mapping"].includes(options.focusedSection) ? options.focusedSection : "";
    const sectionTitle = { subjects: "Subjects", locations: "Locations", mapping: "Mappings" }[focusedSection];
    const wizardLocationMode = Boolean(options?.wizardMode || options?.wizard_location_mode);
    const referenceImagesEnabled = options?.textOnlyMode !== true;
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    const miniMaxTargetMode = normalizeMiniMaxH3Mode(options?.miniMaxTargetMode || miniMaxH3ModeForSegment(activeSegment()));
    const referenceBuilderTargetLabel = !referenceImagesEnabled
      ? "Gemma scene text mapping"
      : miniMaxProject && miniMaxTargetMode === "video_to_video"
        ? "MiniMax Video to Video"
        : miniMaxProject && miniMaxTargetMode === "reference_to_video"
          ? "MiniMax Reference to Video"
          : currentVideoMode() === "rtv"
            ? "LTX Reference to Video"
            : currentVideoMode() === "ingredients"
              ? "LTX Ingredients to Video"
              : "Gemma scene text mapping";
    state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const refs = state.fluxReferenceBuilder;
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(1380px,calc(100vw - 42px));height:calc(100vh - 44px);max-height:calc(100vh - 44px);box-sizing:border-box;overflow:hidden;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    const referenceBuilderDescription = !referenceImagesEnabled
      ? "Map character and location descriptions to scenes for LLM prompt writing. This text-only mode does not send reference images."
      : miniMaxProject && miniMaxTargetMode === "video_to_video"
        ? "Build and map character, background, location, prop, and style images that MiniMax can use alongside the source video for replacements and edits."
        : miniMaxProject && miniMaxTargetMode === "reference_to_video"
          ? "Build and map ordered character, location, prop, style, and storyboard images for MiniMax Reference to Video."
          : "Map character and location descriptions to scenes for Gemma prompt writing. Flux/Nano can also use attached images when those image modes are active.";
    heading.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">${sectionTitle ? `Edit ${sectionTitle}` : "Scene Reference Builder"}</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">${referenceBuilderDescription}</div>`;
    const close = makeButton("Close");
    header.append(heading, close);

    const useSubject = { input: { checked: true } };
    const useLocations = { input: { checked: true } };
    const includeManual = { input: { checked: refs.include_manual_ingredients !== false } };
    const inlineProgress = document.createElement("div");
    inlineProgress.style.cssText = "display:none;border:1px solid #155e75;border-radius:7px;background:#07111f;padding:9px 10px;gap:7px;flex-direction:column;";
    const inlineProgressText = document.createElement("div");
    inlineProgressText.style.cssText = "font-size:12px;color:#e0f2fe;white-space:pre-wrap;line-height:1.35;";
    const inlineProgressTrack = document.createElement("div");
    inlineProgressTrack.style.cssText = "height:8px;border-radius:999px;background:#164e63;overflow:hidden;";
    const inlineProgressBar = document.createElement("div");
    inlineProgressBar.style.cssText = "height:100%;width:0%;background:#22d3ee;border-radius:999px;transition:width .18s ease;";
    inlineProgressTrack.append(inlineProgressBar);
    inlineProgress.append(inlineProgressText, inlineProgressTrack);

    const tabShell = document.createElement("div");
    tabShell.style.cssText = "border:1px solid #1e3a5f;border-radius:8px;background:#0b1220;overflow:hidden;display:flex;flex:1 1 auto;flex-direction:column;min-height:0;";
    const tabBar = document.createElement("div");
    tabBar.style.cssText = `display:grid;grid-template-columns:repeat(${miniMaxProject ? 4 : 3},minmax(0,1fr));border-bottom:1px solid #1e3a5f;background:#0f172a;`;
    const tabContent = document.createElement("div");
    tabContent.style.cssText = "padding:12px;min-height:0;overflow:auto;box-sizing:border-box;";
    const cardStyle = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:12px;display:flex;flex-direction:column;gap:10px;";

    function modalDragGuard(event) {
      const path = typeof event.composedPath === "function" ? event.composedPath() : [];
      const inDropZone = path.some((item) => item?.dataset?.vrgdgFileDropZone === "true")
        || event.target?.closest?.("[data-vrgdg-file-drop-zone='true']");
      if (!inDropZone) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    }
    for (const eventName of ["dragenter", "dragover", "drop"]) {
      backdrop.addEventListener(eventName, modalDragGuard, true);
      box.addEventListener(eventName, modalDragGuard, true);
    }

    const subjectCard = document.createElement("div");
    subjectCard.style.cssText = cardStyle;
    const subjectHeader = document.createElement("div");
    subjectHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;";
    const subjectTitle = document.createElement("div");
    subjectTitle.textContent = referenceImagesEnabled ? "Subject Image References" : "Subject Text References";
    subjectTitle.style.cssText = "font-size:14px;font-weight:900;color:#cffafe;";
    const extractSubjects = makeButton("Extract Subjects", "primary");
    const describeMissingSubjects = makeButton("Gemma - Create subject Description if missing", "primary");
    const createAllMissingSubjectImages = makeButton("Generate Missing Images", "primary");
    const importSubjects = makeButton("Import Subjects", "primary");
    const addSubjectButton = makeButton("Add Subject", "primary");
    const arrangeSubjects = makeButton("Arrange");
    const removeAllSubjects = makeButton("Remove All Subjects");
    const subjectActions = document.createElement("div");
    subjectActions.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;";
    describeMissingSubjects.title = referenceImagesEnabled
      ? "Use vision Gemma to describe character reference images that do not already have descriptions."
      : "Use Gemma to create missing subject descriptions from the current prompts, lyrics, and scene notes.";
    createAllMissingSubjectImages.title = "Batch-create missing subject/reference images. You will choose ZImage, Krea2 + ZImage enhancer, or Flow/GPT before generation starts.";
    importSubjects.title = "Import subject images and matching .txt descriptions from this project's subject_location/subject folder.";
    addSubjectButton.title = "Add one blank subject/reference row.";
    arrangeSubjects.title = "Open a reorder window for subject/reference image cards.";
    subjectActions.append(extractSubjects, describeMissingSubjects, createAllMissingSubjectImages, importSubjects, addSubjectButton, arrangeSubjects, removeAllSubjects);
    createAllMissingSubjectImages.style.display = referenceImagesEnabled ? "" : "none";
    importSubjects.style.display = referenceImagesEnabled ? "" : "none";
    subjectHeader.append(subjectTitle, subjectActions);
    const subjectCountInput = makeInput(String(refs.subject_count || 0), "number");
    subjectCountInput.min = "0";
    subjectCountInput.max = "12";
    subjectCountInput.step = "1";
    const subjectSourceSelect = makeSelect(["prompts_and_director_notes", "director_notes_only", "prompts_only"], "prompts_and_director_notes");
    for (const option of subjectSourceSelect.options) {
      option.textContent = {
        prompts_and_director_notes: "Prompts + Director Notes",
        director_notes_only: "Director Notes only",
        prompts_only: "Prompts / scene notes only",
      }[option.value] || option.value;
    }
    const subjectNameInput = makeInput(refs.subjects?.[0]?.name && refs.subjects[0].name !== "Character 1" ? refs.subjects[0].name : "the performer");
    subjectNameInput.placeholder = "the woman, the man, the performer, lead character...";
    const referenceTypeOptions = [
      ["character", "Character / person"],
      ["prop", "Prop"],
      ["object", "Object"],
      ["vehicle", "Vehicle"],
      ["creature", "Creature"],
      ["outfit", "Outfit / clothing"],
      ["style", "Style reference"],
      ["environment", "Environment detail"],
      ["other", "Other"],
    ];
    function makeReferenceTypeSelect(value = "character") {
      const select = makeSelect(referenceTypeOptions.map(([type]) => type), value || "character");
      for (const option of select.options) {
        option.textContent = referenceTypeOptions.find(([type]) => type === option.value)?.[1] || option.value;
      }
      return select;
    }
    const subjectTypeSelect = makeReferenceTypeSelect(refs.subject.reference_type || refs.subjects?.[0]?.reference_type || "character");
    const subjectDescription = document.createElement("textarea");
    subjectDescription.value = refs.subject.description || "";
    subjectDescription.placeholder = "Subject/character description...";
    subjectDescription.style.cssText = "min-height:92px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:9px;font-size:12px;";
    const subjectDrop = document.createElement("div");
    subjectDrop.style.cssText = "min-height:118px;border:1px dashed #0891b2;border-radius:7px;background:#061620;color:#cffafe;display:flex;align-items:center;justify-content:center;text-align:center;padding:10px;overflow:hidden;";
    const subjectButtons = document.createElement("div");
    subjectButtons.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const createSubjectZImage = makeButton("Generate Subject", "primary");
    const describeSubjectImage = makeButton("Gemma Describe", "primary");
    const uploadSubject = makeButton("Upload Subject Image", "primary");
    const clearSubject = makeButton("Clear Subject");
    createSubjectZImage.title = "Choose ZImage, Krea2 + ZImage enhancer, or Flow/GPT for this subject reference.";
    describeSubjectImage.title = "Use vision Gemma to write the reference description from this image.";
    subjectButtons.append(createSubjectZImage, describeSubjectImage, uploadSubject, clearSubject);
    const subjectsList = document.createElement("div");
    subjectsList.style.cssText = "display:none;flex-direction:column;gap:10px;max-height:560px;overflow:auto;padding-right:4px;";
    subjectCard.append(subjectHeader, makeField("Reference count", subjectCountInput), makeField("Extract subjects from", subjectSourceSelect), makeField("Reference label", subjectNameInput), makeField("Reference type", subjectTypeSelect), makeField("Reference description", subjectDescription), subjectDrop, subjectButtons, subjectsList);

    const extrasToggle = makeCheckbox("Add non-singing/speaking background characters", refs.extras_enabled);
    extrasToggle.wrapper.style.cssText += "border:1px solid #334155;border-radius:7px;background:#111827;padding:9px;";
    if (miniMaxProject) subjectCard.insertBefore(extrasToggle.wrapper, subjectsList);
    const extrasCard = document.createElement("div");
    extrasCard.style.cssText = cardStyle;
    const extrasHeader = document.createElement("div");
    extrasHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;";
    const extrasTitle = document.createElement("div");
    extrasTitle.innerHTML = `<div style="font-size:14px;font-weight:900;color:#cffafe;">Extra Subjects</div><div style="font-size:11px;color:#94a3b8;margin-top:3px;">Reusable background performers. Their photos help the description AI only and are never sent to the MiniMax video model.</div>`;
    const extrasActions = document.createElement("div");
    extrasActions.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;";
    const addExtra = makeButton("Add Extra", "primary");
    const describeMissingExtras = makeButton("Generate Missing Descriptions", "primary");
    extrasActions.append(addExtra, describeMissingExtras);
    extrasHeader.append(extrasTitle, extrasActions);
    const extrasDisabledNote = document.createElement("div");
    extrasDisabledNote.style.cssText = "font-size:12px;color:#fbbf24;border:1px solid #78350f;border-radius:7px;background:#1c1207;padding:10px;";
    extrasDisabledNote.textContent = "Enable “Add non-singing/speaking background characters” on the Subjects tab to use extras.";
    const extrasList = document.createElement("div");
    extrasList.style.cssText = "display:flex;flex-direction:column;gap:10px;";
    extrasCard.append(extrasHeader, extrasDisabledNote, extrasList);

    const locationsCard = document.createElement("div");
    locationsCard.style.cssText = cardStyle;
    const locationsHeader = document.createElement("div");
    locationsHeader.style.cssText = "display:flex;flex-direction:column;gap:10px;";
    const locationsTitle = document.createElement("div");
    locationsTitle.textContent = referenceImagesEnabled ? "Location Image References" : "Location Text References";
    locationsTitle.style.cssText = subjectTitle.style.cssText;
    const extractLocations = makeButton("Gemma Extract", "primary");
    const gptLocationScout = makeButton("GPT Scout", "primary");
    const gptLocationScoutAdvanced = makeButton("GPT Scout Advanced", "primary");
    const autoMapLocations = makeButton("Auto Map Locations with Gemma", "primary");
    const describeMissingLocations = makeButton("Gemma - Create Location Description if missing", "primary");
    const importLocations = makeButton("Import Location List", "primary");
    const exportLocations = makeButton("Export Locations", "primary");
    const uploadLocationImages = makeButton("Upload Images", "primary");
    const createAllMissingLocationZImages = makeButton("Generate Missing Images", "primary");
    const addLocation = makeButton("Add Location", "primary");
    const arrangeLocations = makeButton("Arrange");
    const removeAllLocations = makeButton("Remove All Locations");
    extractLocations.textContent = "Gemma Extract";
    gptLocationScout.textContent = "GPT Scout";
    gptLocationScoutAdvanced.title = "Use AFTER setting up the Storyboard story arc and brief. Then use lyrics, scene beats, subjects, style, and current mappings to assign narrative locations.";
    autoMapLocations.textContent = "Auto Map";
    importLocations.textContent = "Import List";
    exportLocations.textContent = "Export";
    uploadLocationImages.textContent = "Upload Images";
    createAllMissingLocationZImages.textContent = "Generate Missing Images";
    addLocation.textContent = "Add";
    arrangeLocations.textContent = "Arrange";
    removeAllLocations.textContent = "Remove";
    extractLocations.title = "Ask Gemma to extract a reusable location list from your scenes.";
    gptLocationScout.title = "Copy lyrics/dialogue, style/theme, and character descriptions as JSON, then open the Music Video Location Scout GPT.";
    autoMapLocations.title = "Ask Gemma to assign saved locations to each scene.";
    describeMissingLocations.title = referenceImagesEnabled
      ? "Use vision Gemma to describe location reference images that do not already have descriptions."
      : "Use Gemma to create missing location descriptions from the current prompts, lyrics, and scene notes.";
    importLocations.title = "Paste a location list, scene map, or combined location + scene JSON.";
    exportLocations.title = "Save the current location list and scene-location map as JSON in this project folder.";
    uploadLocationImages.title = "Select one or more location reference images and create location cards from them.";
    createAllMissingLocationZImages.title = "Batch-create missing location images. You will choose ZImage, Krea2 + ZImage enhancer, or Flow/GPT before generation starts.";
    addLocation.title = "Add one location card manually.";
    arrangeLocations.title = "Open a reorder window for location cards.";
    removeAllLocations.title = "Remove every location card and clear location mappings.";
    uploadLocationImages.style.display = referenceImagesEnabled ? "" : "none";
    createAllMissingLocationZImages.style.display = referenceImagesEnabled ? "" : "none";
    const locationActions = document.createElement("div");
    locationActions.style.cssText = "display:grid;grid-template-columns:1fr;gap:10px;";
    const locationStyleTheme = makeInput(refs.location_style_theme || "");
    locationStyleTheme.placeholder = "Optional style/theme for location extraction...";
    locationStyleTheme.title = "Optional. Helps Gemma choose locations that match your video style, theme, era, or mood.";
    locationStyleTheme.addEventListener("input", () => {
      refs.location_style_theme = locationStyleTheme.value;
    });
    function locationActionGroup(label, buttons, extra = null) {
      const group = document.createElement("div");
      group.style.cssText = "border:1px solid #334155;border-radius:8px;background:#0b1220;padding:10px 12px;display:flex;flex-direction:column;gap:9px;";
      const rowLabel = document.createElement("div");
      rowLabel.textContent = label;
      rowLabel.style.cssText = "border-bottom:1px solid #273449;padding-bottom:8px;font-size:12px;font-weight:900;color:#cbd5e1;text-transform:uppercase;letter-spacing:.08em;";
      const buttonWrap = document.createElement("div");
      buttonWrap.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;";
      for (const button of buttons) {
        button.style.minHeight = "32px";
        button.style.padding = "6px 11px";
        button.style.flex = "0 0 auto";
        button.style.minWidth = "82px";
        buttonWrap.append(button);
      }
      group.append(rowLabel, buttonWrap);
      if (extra) group.append(extra);
      return group;
    }
    locationActions.append(
      locationActionGroup("Gemma / GPT", [extractLocations, gptLocationScout, gptLocationScoutAdvanced, autoMapLocations, describeMissingLocations], makeField("Optional style/theme for location extraction", locationStyleTheme)),
      locationActionGroup("Manage", [importLocations, exportLocations, uploadLocationImages, createAllMissingLocationZImages, addLocation, arrangeLocations, removeAllLocations])
    );
    locationsHeader.append(locationsTitle, locationActions);
    const keepGemmaLoadedForLocations = makeCheckbox("Keep Gemma loaded while creating location prompts", true);
    keepGemmaLoadedForLocations.wrapper.style.cssText += "border:1px solid #334155;border-radius:7px;background:#111827;padding:8px;";
    const locationsList = document.createElement("div");
    locationsList.style.cssText = "display:flex;flex-direction:column;gap:10px;max-height:560px;overflow:auto;padding-right:4px;";
    locationsCard.append(locationsHeader, keepGemmaLoadedForLocations.wrapper, locationsList);

    const mappingCard = document.createElement("div");
    mappingCard.style.cssText = cardStyle;
    const mappingHeader = document.createElement("div");
    mappingHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;";
    const mappingTitle = document.createElement("div");
    mappingTitle.textContent = "Scene Mapping";
    mappingTitle.style.cssText = subjectTitle.style.cssText;
    const mapSubjectsFromLyrics = makeButton("Map Subjects From Lyrics", "primary");
    const mapSubjectsFromSceneNotes = makeButton("Map Subjects From Scene Notes", "primary");
    const exportGptSceneContext = makeButton("Export GPT Context", "primary");
    const importGptSceneMap = makeButton("Import GPT Map", "primary");
    const assignScenes = makeButton("Assign Scenes", "primary");
    const openAdvancedLineMapping = makeButton("Open Advanced Line Mapping");
    mapSubjectsFromLyrics.title = "Use saved Line Review performer choices to assign character references per scene.";
    mapSubjectsFromSceneNotes.title = "Read SceneNotes.json and assign character references when scene notes mention saved character names.";
    exportGptSceneContext.title = "Copy subjects, locations, lyric lines, and current scene mappings as JSON, then open the Scene Mapping Assistant GPT.";
    importGptSceneMap.title = "Paste GPT scene mapping JSON to assign saved subjects and locations back to scenes.";
    assignScenes.title = "Bulk-assign saved characters and locations using random, rotating, or repeating block patterns.";
    openAdvancedLineMapping.title = "Open the existing Review Lines + Map Performers window for audio, timing, transcription, and detailed performer review.";
    const mappingActions = document.createElement("div");
    mappingActions.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;";
    mappingActions.append(openAdvancedLineMapping, mapSubjectsFromLyrics, mapSubjectsFromSceneNotes, assignScenes, exportGptSceneContext, importGptSceneMap);
    mappingHeader.append(mappingTitle, mappingActions);
    const mappingNote = document.createElement("div");
    mappingNote.textContent = "Choose which character and location text each scene should send to Gemma. Images are only used by image/reference-image workflows that support them.";
    mappingNote.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    const globalPerformanceControls = document.createElement("div");
    globalPerformanceControls.style.cssText = "border:1px solid #155e75;border-radius:7px;background:#071422;padding:9px;display:flex;flex-direction:column;gap:7px;";
    const mappingList = document.createElement("div");
    mappingList.style.cssText = "display:flex;flex-direction:column;gap:8px;max-height:560px;overflow:auto;padding-right:4px;";
    mappingCard.append(mappingHeader, mappingNote, globalPerformanceControls, mappingList);
    openAdvancedLineMapping.onclick = () => launchAdvancedLineMapping();

    const referenceTabs = [
      { id: "subjects", label: "Subjects", node: subjectCard },
      ...(miniMaxProject ? [{ id: "extras", label: "Extra Subjects", node: extrasCard }] : []),
      { id: "locations", label: "Locations", node: locationsCard },
      { id: "mapping", label: "Mapping", node: mappingCard },
    ];
    const referenceTabButtons = new Map();
    function setReferenceTab(id) {
      const active = focusedSection || (referenceTabs.some((tab) => tab.id === id) ? id : "subjects");
      tabContent.textContent = "";
      for (const tab of referenceTabs) {
        const button = referenceTabButtons.get(tab.id);
        if (button) {
          button.style.background = tab.id === active ? "#10243a" : "transparent";
          button.style.color = tab.id === active ? "#cffafe" : "#94a3b8";
          button.style.borderBottom = tab.id === active ? "3px solid #06b6d4" : "3px solid transparent";
        }
        tab.node.style.display = tab.id === active ? "flex" : "none";
        if (tab.id === active) tabContent.append(tab.node);
      }
    }
    for (const tab of referenceTabs) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = tab.label;
      button.style.cssText = "border:0;border-right:1px solid #1e3a5f;background:transparent;color:#94a3b8;padding:15px 14px;font-size:14px;font-weight:900;cursor:pointer;letter-spacing:0;";
      button.onclick = () => setReferenceTab(tab.id);
      referenceTabButtons.set(tab.id, button);
      tabBar.append(button);
    }
    if (!focusedSection) tabShell.append(tabBar);
    tabShell.append(tabContent);
    setReferenceTab(options?.initialTab || (wizardLocationMode ? "locations" : "subjects"));
    const footer = document.createElement("div");
    footer.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;flex:0 0 auto;";
    const cancel = makeButton("Cancel");
    const save = makeButton(sectionTitle ? `Save ${sectionTitle}` : "Save Reference Builder", "primary");
    footer.append(cancel, save);
    box.append(header, inlineProgress, tabShell, footer);
    backdrop.append(box);
    document.body.append(backdrop);

    if (options?.openLocationImport) {
      setTimeout(() => openImportLocationsDialog(), 0);
    }

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.multiple = true;
    fileInput.style.display = "none";
    box.append(fileInput);
    const pendingImage = { target: null };

    function imageLabel(image) {
      return String(image?.name || image?.path || (image?.data ? "Custom image loaded" : ""));
    }
    function imageSrc(image) {
      return image?.data || (image?.path ? makeEditorImageUrl(image.path) : "");
    }

    const {
      openArrangeReferenceDialog, openReferenceImagePreview, renderDrop, renderSubjectThumbnailStrip,
      setImageTargetFromFiles, subjectPreviewImages, syncPrimarySubjectImage, uploadFor, wireDrop,
    } = createReferenceImages({
      droppedSceneImageSource, fileInput, imageLabel, imageSrc, pendingImage, refs, renderAll,
      subjectCountInput, useLocations, useSubject,
      createLocation: (...args) => createLocation(...args),
      createSubject: (...args) => createSubject(...args),
      ensureSubjectCount: (...args) => ensureSubjectCount(...args),
      syncSingleSubjectInputsFromFirstSubject: (...args) => syncSingleSubjectInputsFromFirstSubject(...args),
    });

    const { launchAdvancedLineMapping, openSceneAssignmentDialog } = createSceneAssignment({
      allEditableSegments, backdrop, logicalReferenceSubjects, logicalSubjectIdsForScene, openLyricReviewModal,
      pushHistory, refs, renderAll, selectedSegmentsForBatch, state,
    });

    const { exportSceneMappingContextForGpt, openImportGptSceneMapDialog, renderMapping } = createReferenceSceneMapping({
      allEditableSegments, autoSaveSessionQuiet, exportGptSceneContext, globalPerformanceControls, imageLabel,
      imageSrc, launchAdvancedLineMapping, logicalReferenceSubjects, logicalSubjectIdsForScene, mappingList,
      mappingNote, miniMaxH3ModeForSegment, miniMaxProject, normalizeLyricCueMapForSegment,
      openReferenceImagePreview, playSingerCueRange, projectContextPath, pushHistory, referenceImagesEnabled,
      refs, renderAll, sceneDisplayName, sceneReferenceMapArray, sceneSlotNumber, singerCueRelativePlayheadTime,
      subjectPreviewImages, syncInspector, syncPerformerInspectorForSegment, useLocations, useSubject,
      renderLocations: (...args) => renderLocations(...args),
      renderSubjects: (...args) => renderSubjects(...args),
    });

    const {
      applyReferenceDescription, createFluxReferenceWithZImage, describeReferenceItemsWithGemma,
      describeSingleReferenceWithGemma, openMissingLocationImageGeneratorDialog,
      openMissingSubjectImageGeneratorDialog,
    } = createReferenceGeneration({
      activeSegment, advanceZImageSeedAfterRun, autoMapLocations, autoSaveSessionQuiet,
      createAllMissingLocationZImages, createAllMissingSubjectImages, createProgressWindow,
      describeReferenceImageWithGemma, extractLocations, extractSubjects, gemmaRunnerLine,
      i2vTextGemmaModelSelect, inlineProgress, inlineProgressBar, inlineProgressText,
      keepGemmaLoadedForLocations, locationScoutLyricsPayloadForGpt, projectInput, refs, renderAll,
      renderFluxIngredientList, renderNBIngredientList, runImageMemoryCleanupQuiet, saveSession,
      saveZImageSettingsFromPanel, state, subjectDescription, subjectNameInput, subjectTypeSelect,
      syncZImageSettingsPanel, t2iTextGemmaModelSelect, textGemmaRunnerPayload, themeStyleInput, useLocations,
      useSubject, zClipPicker, zSeed, zUnetPicker, zVaePicker,
      ensureSubjectCount: (...args) => ensureSubjectCount(...args),
    });

    const {
      autoMapLocationsWithGemma, createLocation, exportReferenceBuilderLocations, extractLocationsWithGemma,
      openImportLocationSourceDialog, openImportLocationsDialog, renderLocations,
    } = createReferenceLocations({
      activeSegment, allEditableSegments, autoMapLocations, autoSaveSessionQuiet,
      createDetailedLocationDescriptionWithGemma, createFluxReferenceWithZImage, createProgressWindow,
      describeSingleReferenceWithGemma, exportLocations, extractLocations, gemmaRunnerLine,
      i2vTextGemmaModelSelect, imageTargetFor, locationExtractionStyleTheme, locationKey, locationStyleTheme,
      locationsList, projectReferenceBuilderLocationsPath, referenceImagesEnabled, refs, render, renderAll,
      renderDrop, renderFluxIngredientList, renderMapping, renderNBIngredientList, sceneSlotNumber, state,
      subjectDrop, subjectSceneInput, t2iTextGemmaModelSelect, textGemmaRunnerPayload, uploadFor, useLocations,
      wireDrop, wizardLocationMode,
    });

    const {
      autoMapSubjectsFromLyrics, autoMapSubjectsFromSceneNotesJson, createSubject, ensureSubjectCount,
      extractSubjectsWithGemma, renderExtras, renderSubjects, syncSingleSubjectInputsFromFirstSubject,
    } = createReferenceSubjects({
      activeSegment, allEditableSegments, createFluxReferenceWithZImage, createProgressWindow, currentVideoMode,
      describeSingleReferenceWithGemma, extractSubjects, extrasDisabledNote, extrasList, gemmaRunnerLine,
      i2vTextGemmaModelSelect, imageTargetFor, locationKey, logicalReferenceSubjects, logicalSubjectIdsForScene,
      makeReferenceTypeSelect, miniMaxH3SettingsForSegment, miniMaxProject, projectInput, projectSceneNotesPath,
      referenceImagesEnabled, refs, renderAll, renderDrop, renderMapping, renderSubjectThumbnailStrip, state,
      subjectButtons, subjectCountInput, subjectDescription, subjectDrop, subjectNameInput, subjectSceneInput,
      subjectSourceSelect, subjectTypeSelect, subjectsList, syncMiniMaxReferenceButtons,
      syncPrimarySubjectImage, t2iTextGemmaModelSelect, textGemmaRunnerPayload, uploadFor, useSubject, wireDrop,
    });

    assignScenes.onclick = openSceneAssignmentDialog;

    function imageTargetFor(owner, key = "image", kind = "") {
      return { owner, key, kind };
    }
    fileInput.onchange = () => {
      setImageTargetFromFiles(pendingImage.target, Array.from(fileInput.files || []));
      fileInput.value = "";
      pendingImage.target = null;
    };
    function locationKey(name) {
      return String(name || "").trim().toLowerCase().replace(/\s+/g, " ");
    }

    extrasToggle.input.onchange = () => {
      refs.extras_enabled = Boolean(extrasToggle.input.checked);
      renderExtras();
      renderMapping();
    };
    addExtra.onclick = () => {
      refs.extras_enabled = true;
      extrasToggle.input.checked = true;
      refs.extra_subjects.push({ id: `extra_${Date.now()}_${Math.floor(Math.random() * 10000)}`, title: `Extra ${refs.extra_subjects.length + 1}`, description: "", count: 1, style: "", send_to_minimax: false, reference_image_type: "single", image: { path: "", data: "", name: "", preview_url: "" } });
      renderExtras();
    };
    describeMissingExtras.onclick = () => describeReferenceItemsWithGemma(
      refs.extra_subjects.map((target, index) => ({ target, label: target.title || `Extra ${index + 1}` })),
      "extra",
      { keepLoaded: true, clearBeforeLoad: false }
    );

    function renderAll() {
      if (refs.subjects.length || Number(refs.subject_count || 0) > 0) ensureSubjectCount();
      else refs.subject_count = 0;
      if (refs.subject_count === 1 && refs.subjects[0]) syncSingleSubjectInputsFromFirstSubject();
      renderSubjects();
      renderExtras();
      renderLocations();
      renderMapping();
    }

    uploadSubject.onclick = () => uploadFor(imageTargetFor(refs.subject, "image", "subject"));
    addSubjectButton.onclick = () => {
      createSubject();
      refs.use_subject_reference = true;
      useSubject.input.checked = true;
      renderAll();
    };
    arrangeSubjects.onclick = () => openArrangeReferenceDialog("subject");
    createSubjectZImage.onclick = () => createFluxReferenceWithZImage("subject", refs.subject, refs.subject.description || subjectDescription.value || "", "subject_reference");
    describeSubjectImage.onclick = async () => {
      ensureSubjectCount();
      const primarySubject = refs.subjects[0] || refs.subject;
      const progress = createProgressWindow("Describing character", { zIndex: 100008 });
      try {
        progress.set(`Vision Gemma describing character image...\n${subjectNameInput.value || refs.subjects[0]?.name || "Subject"}\n${gemmaRunnerLine({ vision: true })}`, 18);
        const description = await describeReferenceImageWithGemma(primarySubject, "subject", { unloadAfter: true });
        applyReferenceDescription("subject", primarySubject, description);
        renderAll();
        await autoSaveSessionQuiet("Gemma described subject");
        progress.set("Description updated.", 100);
        progress.close(1200);
        toast("Character description updated.");
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
      }
    };
    describeMissingSubjects.onclick = () => {
      ensureSubjectCount();
      const subjectItems = refs.subjects.map((subject, index) => ({
        target: subject,
        label: subject.name || `Character ${index + 1}`,
      }));
      describeReferenceItemsWithGemma(subjectItems, "subject", { keepLoaded: true, clearBeforeLoad: false });
    };
    createAllMissingSubjectImages.onclick = openMissingSubjectImageGeneratorDialog;
    describeMissingLocations.onclick = () => {
      const locationItems = refs.locations.map((location, index) => ({
        target: location,
        label: location.name || `Location ${index + 1}`,
      }));
      describeReferenceItemsWithGemma(locationItems, "location", { keepLoaded: true, clearBeforeLoad: false });
    };
    clearSubject.onclick = () => {
      refs.subject.image = { path: "", data: "", name: "" };
      renderAll();
    };
    removeAllSubjects.onclick = () => {
      if (refs.subjects.length && !window.confirm("Remove all subject references and clear subject scene mappings?")) return;
      refs.subject_count = 0;
      refs.subject = {
        name: "",
        description: "",
        reference_type: "character",
        image: { path: "", data: "", name: "" },
      };
      refs.subjects = [];
      refs.subject_scene_map = {};
      refs.use_subject_reference = false;
      useSubject.input.checked = false;
      subjectCountInput.value = "0";
      subjectNameInput.value = "";
      subjectTypeSelect.value = "character";
      subjectDescription.value = "";
      renderAll();
      toast("Removed all subject references.");
    };
    importSubjects.onclick = async () => {
      const projectFolder = String(state.projectFolder || "").trim();
      const progress = createProgressWindow("Importing Subjects", { zIndex: 100008 });
      try {
        if (!projectFolder) throw new Error("Create or load a project first so the subject folder can be found.");
        progress.set(`Looking for subject images and descriptions...\n${projectFolder}\\subject_location\\subject`, 20);
        const data = await postJson("/vrgdg/music_builder/import_reference_subjects", {
          project_folder: projectFolder,
        }, 60000);
        const imported = Array.isArray(data.subjects) ? data.subjects.map((subject, index) => {
          const name = String(subject?.name || `Subject ${index + 1}`).trim() || `Subject ${index + 1}`;
          const image = subject?.image && typeof subject.image === "object" ? subject.image : {};
          return {
            id: String(subject?.id || `subj_import_${Date.now()}_${index}_${Math.floor(Math.random() * 10000)}`),
            name,
            description: String(subject?.description || "").trim(),
            reference_type: String(subject?.reference_type || subject?.referenceType || subject?.type || "character").trim() || "character",
            extra_reference_for: String(subject?.extra_reference_for || subject?.extraReferenceFor || subject?.same_subject_as || subject?.sameSubjectAs || "").trim(),
            extra_reference_note: String(subject?.extra_reference_note || subject?.extraReferenceNote || "").trim(),
            image: {
              path: String(image.path || ""),
              data: String(image.data || ""),
              name: String(image.name || ""),
            },
          };
        }).filter((subject) => subject.name) : [];
        if (!imported.length) throw new Error("No subject images were returned from the import.");
        refs.subjects = imported;
        refs.subject_count = imported.length;
        refs.subject_scene_map = {};
        refs.use_subject_reference = true;
        refs.subject = {
          name: imported[0].name || "the performer",
          description: imported[0].description || "",
          reference_type: imported[0].reference_type || "character",
          image: imported[0].image || { path: "", data: "", name: "" },
        };
        useSubject.input.checked = true;
        subjectCountInput.value = String(refs.subject_count);
        subjectNameInput.value = refs.subject.name || "the performer";
        subjectTypeSelect.value = refs.subject.reference_type || "character";
        subjectDescription.value = refs.subject.description || "";
        state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
        renderAll();
        renderFluxIngredientList(activeSegment());
        renderNBIngredientList(activeSegment());
        render();
        await autoSaveSessionQuiet("imported reference subjects");
        const missingCount = Array.isArray(data.missing_descriptions) ? data.missing_descriptions.length : 0;
        const message = `Imported ${imported.length} subject${imported.length === 1 ? "" : "s"} from subject_location/subject.${missingCount ? `\n${missingCount} subject${missingCount === 1 ? " is" : "s are"} missing matching .txt descriptions.` : ""}`;
        progress.set(message, 100);
        toast(message);
        progress.close(2200);
      } catch (error) {
        const message = String(error?.message || error || "Could not import subjects.");
        progress.set(`Error:\n${message}`, 100);
        toast(message, true);
      }
    };
    addLocation.onclick = () => {
      createLocation();
      refs.locations_cleared = false;
      refs.use_location_references = true;
      useLocations.input.checked = true;
      renderAll();
    };
    uploadLocationImages.onclick = () => uploadFor({ kind: "location", bulk: true });
    arrangeLocations.onclick = () => openArrangeReferenceDialog("location");
    removeAllLocations.onclick = () => {
      if (refs.locations.length && !window.confirm("Remove all location references and clear location scene mappings?")) return;
      refs.locations = [];
      refs.scene_map = {};
      refs.scene_trigger_map = {};
      refs.use_location_references = false;
      refs.locations_cleared = true;
      useLocations.input.checked = false;
      renderAll();
      toast("Removed all location references.");
    };
    subjectCountInput.addEventListener("change", () => {
      ensureSubjectCount({ allowTrim: true });
      renderAll();
    });
    subjectCountInput.addEventListener("input", () => {
      extractSubjects.style.display = Number(subjectCountInput.value || 0) > 1 ? "" : "none";
    });
    extractSubjects.onclick = extractSubjectsWithGemma;
    extractLocations.onclick = extractLocationsWithGemma;
    gptLocationScout.onclick = () => openLocationScoutGptForRefs(refs, locationStyleTheme.value || "", {
      onImportList: openImportLocationSourceDialog,
    });
    gptLocationScoutAdvanced.onclick = () => openAdvancedLocationScoutGptForRefs(refs, locationStyleTheme.value || "", {
      onImportList: openImportLocationSourceDialog,
    });
    autoMapLocations.onclick = autoMapLocationsWithGemma;
    createAllMissingLocationZImages.onclick = openMissingLocationImageGeneratorDialog;
    importLocations.onclick = openImportLocationSourceDialog;
    exportLocations.onclick = exportReferenceBuilderLocations;
    exportGptSceneContext.onclick = exportSceneMappingContextForGpt;
    importGptSceneMap.onclick = openImportGptSceneMapDialog;
    mapSubjectsFromLyrics.onclick = () => {
      const mapped = autoMapSubjectsFromLyrics();
      refs.use_subject_reference = true;
      useSubject.input.checked = true;
      renderAll();
      toast(mapped
        ? `Mapped character references from line performer choices for ${mapped} scene${mapped === 1 ? "" : "s"}.`
        : "No line performer assignments matched Reference Builder subjects yet.");
    };
    mapSubjectsFromSceneNotes.onclick = async () => {
      try {
        mapSubjectsFromSceneNotes.disabled = true;
        mapSubjectsFromSceneNotes.textContent = "Mapping...";
        const result = await autoMapSubjectsFromSceneNotesJson();
        refs.use_subject_reference = true;
        useSubject.input.checked = true;
        renderAll();
        toast(result.mapped
          ? `Mapped ${result.matches} character reference${result.matches === 1 ? "" : "s"} from SceneNotes.json across ${result.mapped} scene${result.mapped === 1 ? "" : "s"}.`
          : `No Reference Builder character names were found in SceneNotes.json.\n${result.path}`);
      } catch (error) {
        toast(String(error?.message || error), true);
      } finally {
        mapSubjectsFromSceneNotes.disabled = false;
        mapSubjectsFromSceneNotes.textContent = "Map Subjects From Scene Notes";
      }
    };
    subjectNameInput.addEventListener("input", () => {
      if (refs.subject_count === 1 && refs.subjects[0]) {
        refs.subjects[0].name = subjectNameInput.value || "Character 1";
        renderMapping();
      }
    });
    subjectTypeSelect.addEventListener("change", () => {
      refs.subject.reference_type = subjectTypeSelect.value || "character";
      if (refs.subject_count === 1 && refs.subjects[0]) refs.subjects[0].reference_type = refs.subject.reference_type;
      renderMapping();
    });
    subjectDescription.addEventListener("input", () => {
      if (refs.subject_count === 1) refs.subject.description = subjectDescription.value;
    });
    wireDrop(subjectDrop, imageTargetFor(refs.subject, "image", "subject"));
    function closeReferenceBuilder() { backdrop.remove(); options.onClose?.(); }
    close.onclick = closeReferenceBuilder;
    cancel.onclick = closeReferenceBuilder;
    save.onclick = async () => {
      refs.subjects = Array.isArray(refs.subjects)
        ? refs.subjects.filter((subject) => subject && typeof subject === "object" && String(subject.id || subject.name || subject.description || subject.image?.path || subject.image?.data).trim())
        : [];
      refs.subject_count = refs.subjects.length;
      subjectCountInput.value = String(refs.subject_count || 0);
      if (refs.subject_count === 1 && refs.subjects[0]) {
        const primarySubject = refs.subjects[0];
        refs.subject = {
          name: primarySubject.name || "Character 1",
          description: primarySubject.description || "",
          reference_type: primarySubject.reference_type || "character",
          minimax_voice: normalizeMiniMaxH3Voice(primarySubject.minimax_voice),
          reference_generation_draft: primarySubject.reference_generation_draft || refs.subject.reference_generation_draft || {},
          image: { ...(primarySubject.image || { path: "", data: "", name: "" }) },
        };
        syncSingleSubjectInputsFromFirstSubject();
      } else if (refs.subjects[0]) {
        refs.subject = {
          name: refs.subjects[0].name || "Character 1",
          description: refs.subjects[0].description || "",
          reference_type: refs.subjects[0].reference_type || "character",
          minimax_voice: normalizeMiniMaxH3Voice(refs.subjects[0].minimax_voice),
          image: { ...(refs.subjects[0].image || { path: "", data: "", name: "" }) },
        };
      } else {
        refs.subject = {
          name: "",
          description: "",
          reference_type: "character",
          minimax_voice: normalizeMiniMaxH3Voice(),
          image: { path: "", data: "", name: "" },
        };
      }
      const validSubjectIds = new Set(logicalReferenceSubjects(refs).map((subject) => String(subject.id || "").trim()).filter(Boolean));
      if (!refs.subject_scene_map || typeof refs.subject_scene_map !== "object") refs.subject_scene_map = {};
      if (!refs.performer_scene_map || typeof refs.performer_scene_map !== "object") refs.performer_scene_map = {};
      for (const [sceneId, ids] of Object.entries(refs.subject_scene_map || {})) {
        const pruned = Array.from(new Set((Array.isArray(ids) ? ids : []).map(String).map((id) => {
          const subject = refs.subjects.find((item) => item.id === id);
          return subject ? (subjectExtraTargetId(subject) || subject.id) : "";
        }).filter((id) => validSubjectIds.has(id))));
        if (pruned.length) refs.subject_scene_map[sceneId] = pruned;
        else delete refs.subject_scene_map[sceneId];
      }
      for (const [sceneId, ids] of Object.entries(refs.performer_scene_map || {})) {
        const pruned = Array.from(new Set((Array.isArray(ids) ? ids : []).map(String).map((id) => {
          const subject = refs.subjects.find((item) => item.id === id);
          return subject ? (subjectExtraTargetId(subject) || subject.id) : "";
        }).filter((id) => validSubjectIds.has(id))));
        if (pruned.length) refs.performer_scene_map[sceneId] = pruned;
        else delete refs.performer_scene_map[sceneId];
      }
      const validExtraIds = new Set((refs.extra_subjects || []).map((extra) => String(extra.id || "").trim()).filter(Boolean));
      if (!refs.extra_scene_map || typeof refs.extra_scene_map !== "object") refs.extra_scene_map = {};
      for (const [sceneId, entries] of Object.entries(refs.extra_scene_map)) {
        const seenExtraIds = new Set();
        const pruned = (Array.isArray(entries) ? entries : []).map((entry) => {
          const extraId = String(entry?.extra_id || entry?.extraId || "").trim();
          if (!validExtraIds.has(extraId) || seenExtraIds.has(extraId)) return null;
          seenExtraIds.add(extraId);
          const interaction = ["background", "background_dancing", "alongside", "dancing_with", "direct"].includes(String(entry?.interaction || "").trim())
            ? String(entry.interaction).trim()
            : "background";
          return { extra_id: extraId, interaction };
        }).filter(Boolean);
        if (pruned.length) refs.extra_scene_map[sceneId] = pruned;
        else delete refs.extra_scene_map[sceneId];
      }
      for (const select of mappingList.querySelectorAll("[data-subject-map-segment-id]")) {
        const segmentId = select.dataset.subjectMapSegmentId || "";
        if (!segmentId) continue;
        const subjectIds = Array.from(select.selectedOptions || []).map((option) => option.value).filter((id) => validSubjectIds.has(id));
        if (subjectIds.length) refs.subject_scene_map[segmentId] = subjectIds;
        else delete refs.subject_scene_map[segmentId];
      }
      for (const segment of allEditableSegments()) {
        const performerIds = sceneReferenceMapArray(refs.performer_scene_map, segment).filter((id) => validSubjectIds.has(id));
        if (performerIds.length) {
          refs.performer_scene_map[segment.id] = performerIds;
          const performerNames = logicalReferenceSubjects(refs)
            .filter((subject) => performerIds.includes(String(subject.id)))
            .map((subject) => subject.name || "Character")
            .filter(Boolean);
          if (performerNames.length) segment.lyric_singers = performerNames;
        } else {
          delete refs.performer_scene_map[segment.id];
        }
      }
      const validLocationIds = new Set((refs.locations || []).map((location) => String(location.id || "").trim()).filter(Boolean));
      if (!validLocationIds.size) {
        refs.scene_map = {};
        refs.scene_trigger_map = {};
        refs.use_location_references = false;
        refs.locations_cleared = true;
        useLocations.input.checked = false;
      } else {
        refs.locations_cleared = false;
        if (!refs.scene_map || typeof refs.scene_map !== "object") refs.scene_map = {};
        for (const select of mappingList.querySelectorAll("[data-location-map-segment-id]")) {
          const segmentId = select.dataset.locationMapSegmentId || "";
          if (!segmentId) continue;
          const locationId = String(select.value || "").trim();
          if (locationId && validLocationIds.has(locationId)) refs.scene_map[segmentId] = locationId;
          else delete refs.scene_map[segmentId];
        }
        for (const [sceneId, locationId] of Object.entries(refs.scene_map || {})) {
          if (!validLocationIds.has(String(locationId || "").trim())) delete refs.scene_map[sceneId];
        }
      }
      refs.use_subject_reference = Boolean(refs.subjects.length || Object.keys(refs.subject_scene_map || {}).length);
      refs.use_location_references = Boolean(validLocationIds.size || Object.keys(refs.scene_map || {}).length);
      refs.include_manual_ingredients = Boolean(includeManual.input.checked);
      refs.location_style_theme = String(locationStyleTheme.value || "");
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
      renderFluxIngredientList(activeSegment());
      renderNBIngredientList(activeSegment());
      syncPerformerInspectorForSegment(activeSegment());
      syncMiniMaxH3Panel();
      render();
      // This is an explicit Save action and must work even when optional
      // autosave is disabled. It also needs the backend's manifest/context
      // validation before the modal can report success.
      await saveSession({ quiet: true, throwOnError: true });
      toast(`${referenceBuilderTargetLabel} reference builder saved.`);
      closeReferenceBuilder();
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) closeReferenceBuilder();
    });
    renderAll();
  }

  return { openFluxReferenceBuilderModal };
}
