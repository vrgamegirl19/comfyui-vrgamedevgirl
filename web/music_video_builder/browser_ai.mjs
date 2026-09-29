import {
  BROWSER_IMAGE_PROVIDERS,
  finishManualBrowserImageSession,
  getBrowserImageStatus,
  importLatestManualBrowserImageDownload,
  openBrowserImageLogin,
  openManualBrowserImageProvider,
  setupBrowserImageAutomation,
  storeBrowserImageReference,
  submitManualBrowserImageRequest,
  uploadManualBrowserImageRefs,
} from "../VRGDG_BrowserImageBridge.js";
import { makeEditorImageUrl } from "./comfy_api.mjs";
import { makeButton, makeInput, toast } from "./controls.mjs";
import { sceneImagePromptForEnhanceAll } from "./image_generation.mjs";
import { readFileAsDataUrl } from "./media_import.mjs";
import {
  browserImageLoginStatus,
  browserImageProviderDebugPort,
  browserImageProviderLabel,
  browserImageProviderShortLabel,
  browserImageProviderTimeout,
  cloneFlowGptBrowserSettings,
  defaultFlowGptBrowserSettings,
} from "./model_settings.mjs";

function browserAiCharacterCountPrompt(characterCount, subjectInstruction = "", includesNamedExtras = false) {
  const count = Math.max(1, Number(characterCount || 1));
  const singular = count === 1;
  const characterLabel = singular ? "character" : "characters";
  const placementSubject = singular ? "the character" : `all ${count} characters`;
  const lines = [
    "Using the character and location reference images, create 5 new images. "
      + `Place ${placementSubject} naturally within the location in different areas, poses, and compositions. `
      + "Vary the camera position and angle for each image. "
      + `Integrate ${placementSubject} into the environment instead of simply pasting ${singular ? "the character" : "the characters"} onto the scene.`,
    "The images should look like cinematic video movie stills for a music video.",
    "16:9 aspect ratio.",
    `I need a close up wide angle image of the ${count} ${characterLabel}.`,
    "DO NOT SEND A GRID OF IMAGES!!! SEND EACH IMAGE AS ITS OWN IMAGE",
    includesNamedExtras
      ? `ONLY INCLUDE THE ${count} ${characterLabel.toUpperCase()}. NO ADDITIONAL CHARACTERS.`
      : `ONLY INCLUDE THE ${count} ${characterLabel.toUpperCase()}. NO EXTRAS.`,
  ];
  if (subjectInstruction) lines.push(subjectInstruction);
  return lines.join("\n");
}

function browserAiReferenceRow(item, onRemove) {
  const row = document.createElement("div");
  row.style.cssText = "display:grid;grid-template-columns:42px minmax(0,1fr) auto;gap:7px;align-items:center;border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:5px;";
  const image = document.createElement("img");
  image.alt = item?.name || "Reference";
  image.src = item?.path ? makeEditorImageUrl(item.path) : String(item?.data || "");
  image.style.cssText = "width:42px;height:42px;object-fit:cover;border-radius:4px;background:#09090b;";
  const label = document.createElement("div");
  label.textContent = item?.name || String(item?.path || "reference image").split(/[\\/]/).pop();
  label.title = item?.path || item?.name || "";
  label.style.cssText = "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;color:#e4e4e7;";
  const remove = makeButton("Remove");
  remove.style.padding = "5px 7px";
  remove.onclick = onRemove;
  row.append(image, label, remove);
  return row;
}

function browserAiLocationDisplayName(item, index = 0) {
  const customLabel = String(item?.location_label || "").trim();
  if (customLabel) return customLabel;
  const filename = String(item?.name || item?.path || "").split(/[\\/]/).pop();
  const filenameWithoutExtension = filename.replace(/\.(?:png|jpe?g|webp)$/i, "").trim();
  return filenameWithoutExtension || `Location ${index + 1}`;
}

function browserAiLocationReferenceRow(item, index, onRemove, onLabelChange) {
  const row = document.createElement("div");
  row.style.cssText = "display:grid;grid-template-columns:42px minmax(0,1fr) auto;gap:7px;align-items:center;border:1px solid #7c3aed;border-radius:6px;background:#18181b;padding:5px;";
  const image = document.createElement("img");
  image.alt = browserAiLocationDisplayName(item, index);
  image.src = item?.path ? makeEditorImageUrl(item.path) : String(item?.data || "");
  image.style.cssText = "width:42px;height:42px;object-fit:cover;border-radius:4px;background:#09090b;";
  const details = document.createElement("div");
  details.style.cssText = "display:flex;flex-direction:column;gap:3px;min-width:0;";
  const locationName = makeInput(String(item?.location_label || ""));
  locationName.placeholder = `Type location name, e.g. ${index === 0 ? "Water" : `Location ${index + 1}`}`;
  locationName.maxLength = 90;
  locationName.title = "This name is used in the location selector and as the Browser AI download folder.";
  ["keydown", "keypress", "keyup"].forEach((eventName) => {
    locationName.addEventListener(eventName, (event) => event.stopPropagation());
  });
  locationName.addEventListener("keydown", (event) => {
    if (event.key === "Enter") locationName.blur();
  });
  locationName.addEventListener("change", () => onLabelChange(String(locationName.value || "").trim()));
  const filename = document.createElement("div");
  filename.textContent = item?.name || String(item?.path || "location image").split(/[\\/]/).pop();
  filename.title = item?.path || item?.name || "";
  filename.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:9px;color:#71717a;";
  details.append(locationName, filename);
  const remove = makeButton("Remove");
  remove.style.padding = "5px 7px";
  remove.onclick = onRemove;
  row.append(image, details, remove);
  return row;
}

export function formatBrowserImageStatus(data) {
  return [
    `Chrome: ${data.chrome_ready ? "ready" : "missing"}`,
    `Node: ${data.node_ready ? "ready" : "missing"}`,
    `Playwright: ${data.playwright_ready ? "installed" : "missing"}`,
    data.chrome_error ? `Chrome error: ${data.chrome_error}` : "",
  ].filter(Boolean).join("\n");
}

export function wireBrowserAiPanel({
  activeBrowserAiReferenceGroup, addBrowserAiBandSequenceFiles, addBrowserAiGroupFiles, autoSaveSessionQuiet,
  browserAiAddExtrasButton, browserAiAddGroupImagesButton, browserAiAddLocationsButton,
  browserAiAddMembersButton, browserAiAddSingerButton, browserAiAutoAdvanceGroup, browserAiBandSequenceMode,
  browserAiBandSequenceSelection, browserAiChooseLocationButton, browserAiClearExtrasButton,
  browserAiClearGroupImagesButton, browserAiClearLocationButton, browserAiClearLocationsButton,
  browserAiClearMembersButton, browserAiClearSingerButton, browserAiDeleteGroupButton,
  browserAiDuplicateGroupButton, browserAiExtrasDrop, browserAiFinishButton, browserAiGroupDrop,
  browserAiGroupPrompt, browserAiGroupSelect, browserAiGroupStatus, browserAiLocationDrop,
  browserAiLocationsDrop, browserAiMembersDrop, browserAiNewGroupButton, browserAiRenameGroupButton,
  browserAiSend, browserAiSendButton, browserAiSequenceLocationSelect, browserAiSequenceSetSelect,
  browserAiSingerDrop, chooseBrowserAiImageFiles, clearBrowserAiBandSequenceFiles, exportManualFlowGptRefs,
  finishBrowserAiDownloadSession, flowGptCreateImageButton, flowGptLoginButton, flowGptManualChatPrompt,
  flowGptManualExportRefsButton, flowGptManualImportLatestButton, flowGptManualOpenButton,
  flowGptManualStatus, flowGptSetupButton, flowGptStatusButton, flowGptStatusText, flowNanoProviderButton,
  gptImageProviderButton, importLatestManualFlowGptDownload, metaImageProviderButton,
  openManualFlowGptBrowser, previewFlowGptImage, refreshFlowGptBrowserStatus, renderBrowserAiReferenceGroups,
  saveFlowGptBrowserSettingsFromPanel, sendBrowserAiReferenceGroup, setBrowserAiLocationFile,
  setFlowGptProvider, state,
}) {
  browserAiGroupPrompt.addEventListener("input", () => {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    settings.manual_chat_prompt = browserAiGroupPrompt.value || "";
    state.flowGptBrowserSettings = settings;
    flowGptManualChatPrompt.value = browserAiGroupPrompt.value || "";
  });
  browserAiGroupPrompt.addEventListener("change", () => autoSaveSessionQuiet("Browser AI group prompt changed").catch(() => null));
  browserAiBandSequenceMode.input.addEventListener("change", () => {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    settings.band_sequence_enabled = Boolean(browserAiBandSequenceMode.input.checked);
    if (settings.band_sequence_enabled) {
      settings.band_sequence_location_index = 0;
      settings.band_sequence_set_index = 0;
    }
    settings.browser_ai_prompt_sequence_key = "";
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    browserAiGroupStatus.textContent = settings.band_sequence_enabled
      ? "Band Sequence mode is on. Add singer references, other members, and all locations once."
      : "Custom Groups mode is on.";
    autoSaveSessionQuiet("Browser AI Band Sequence mode changed").catch(() => null);
  });
  browserAiAutoAdvanceGroup.input.addEventListener("change", () => {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    settings.auto_advance_reference_group = Boolean(browserAiAutoAdvanceGroup.input.checked);
    state.flowGptBrowserSettings = settings;
    browserAiGroupStatus.textContent = settings.auto_advance_reference_group
      ? "After each submission, the next group or Band Sequence set will be selected. You still decide when to send it."
      : "Automatic selection is off. The current group or set will remain selected after sending.";
    autoSaveSessionQuiet("Browser AI group auto-advance changed").catch(() => null);
  });
  browserAiSequenceLocationSelect.addEventListener("change", () => {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    settings.band_sequence_location_index = Math.max(0, Number(browserAiSequenceLocationSelect.value || 0));
    settings.band_sequence_set_index = 0;
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    autoSaveSessionQuiet("Browser AI sequence location selected").catch(() => null);
  });
  browserAiSequenceSetSelect.addEventListener("change", () => {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    settings.band_sequence_set_index = Math.max(0, Math.min(
      browserAiBandSequenceSelection(settings).setCount - 1,
      Number(browserAiSequenceSetSelect.value || 0),
    ));
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    autoSaveSessionQuiet("Browser AI sequence subject set selected").catch(() => null);
  });
  browserAiGroupSelect.addEventListener("change", () => {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    settings.active_reference_group_id = browserAiGroupSelect.value || settings.reference_groups[0].id;
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    autoSaveSessionQuiet("Browser AI reference group selected").catch(() => null);
  });
  browserAiAddSingerButton.onclick = () => chooseBrowserAiImageFiles({
    multiple: true,
    onFiles: (files) => addBrowserAiBandSequenceFiles(files, "band_sequence_singer_references", "group", "Singer", "Singer references"),
  });
  browserAiSingerDrop.onclick = () => browserAiAddSingerButton.click();
  browserAiClearSingerButton.onclick = () => clearBrowserAiBandSequenceFiles("band_sequence_singer_references", "Singer references");
  browserAiAddExtrasButton.onclick = () => chooseBrowserAiImageFiles({
    multiple: true,
    onFiles: (files) => addBrowserAiBandSequenceFiles(files, "band_sequence_extra_references", "group", "Extras", "Extras"),
  });
  browserAiExtrasDrop.onclick = () => browserAiAddExtrasButton.click();
  browserAiClearExtrasButton.onclick = () => clearBrowserAiBandSequenceFiles("band_sequence_extra_references", "Extras");
  browserAiAddMembersButton.onclick = () => chooseBrowserAiImageFiles({
    multiple: true,
    onFiles: (files) => addBrowserAiBandSequenceFiles(files, "band_sequence_member_references", "group", "Other Band Members", "Other band member references"),
  });
  browserAiMembersDrop.onclick = () => browserAiAddMembersButton.click();
  browserAiClearMembersButton.onclick = () => clearBrowserAiBandSequenceFiles("band_sequence_member_references", "Other band member references");
  browserAiAddLocationsButton.onclick = () => chooseBrowserAiImageFiles({
    multiple: true,
    onFiles: (files) => addBrowserAiBandSequenceFiles(files, "band_sequence_location_references", "location", "Locations", "Locations"),
  });
  browserAiLocationsDrop.onclick = () => browserAiAddLocationsButton.click();
  browserAiClearLocationsButton.onclick = () => clearBrowserAiBandSequenceFiles("band_sequence_location_references", "Locations");
  browserAiNewGroupButton.onclick = () => {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    const id = `group_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
    const group = { id, name: `Group ${settings.reference_groups.length + 1}`, images: [] };
    settings.reference_groups.push(group);
    settings.active_reference_group_id = id;
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    browserAiGroupStatus.textContent = `${group.name} created.`;
    autoSaveSessionQuiet("Browser AI reference group created").catch(() => null);
  };
  browserAiDuplicateGroupButton.onclick = () => {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    const source = activeBrowserAiReferenceGroup(settings);
    const id = `group_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
    const group = { id, name: `${source.name} Copy`, images: source.images.map((item) => ({ ...item })) };
    settings.reference_groups.push(group);
    settings.active_reference_group_id = id;
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    browserAiGroupStatus.textContent = `${source.name} duplicated. Swap the location separately whenever you want.`;
    autoSaveSessionQuiet("Browser AI reference group duplicated").catch(() => null);
  };
  browserAiRenameGroupButton.onclick = () => {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    const group = activeBrowserAiReferenceGroup(settings);
    const name = window.prompt("Reference group name", group.name);
    if (name == null || !String(name).trim()) return;
    group.name = String(name).trim().slice(0, 90);
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    browserAiGroupStatus.textContent = `Group renamed to ${group.name}. Existing reference files were left in place.`;
    autoSaveSessionQuiet("Browser AI reference group renamed").catch(() => null);
  };
  browserAiDeleteGroupButton.onclick = () => {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    if (settings.reference_groups.length <= 1) return;
    const group = activeBrowserAiReferenceGroup(settings);
    if (!window.confirm(`Delete ${group.name} from this project? The stored image files will be kept.`)) return;
    settings.reference_groups = settings.reference_groups.filter((item) => item.id !== group.id);
    settings.active_reference_group_id = settings.reference_groups[0].id;
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    browserAiGroupStatus.textContent = `${group.name} removed. Its image files were not deleted.`;
    autoSaveSessionQuiet("Browser AI reference group deleted").catch(() => null);
  };
  browserAiAddGroupImagesButton.onclick = () => chooseBrowserAiImageFiles({ multiple: true, onFiles: addBrowserAiGroupFiles });
  browserAiGroupDrop.onclick = () => browserAiAddGroupImagesButton.click();
  browserAiClearGroupImagesButton.onclick = () => {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    const group = activeBrowserAiReferenceGroup(settings);
    if (group.images.length && !window.confirm(`Remove all references from ${group.name}? Stored image files will be kept.`)) return;
    group.images = [];
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    autoSaveSessionQuiet("Browser AI group references cleared").catch(() => null);
  };
  browserAiChooseLocationButton.onclick = () => chooseBrowserAiImageFiles({ multiple: false, onFiles: (files) => setBrowserAiLocationFile(files[0]) });
  browserAiLocationDrop.onclick = () => browserAiChooseLocationButton.click();
  browserAiClearLocationButton.onclick = () => {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    settings.location_reference = null;
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    autoSaveSessionQuiet("Browser AI location cleared").catch(() => null);
  };
  for (const [dropZone, handler] of [
    [browserAiGroupDrop, (files) => addBrowserAiGroupFiles(files)],
    [browserAiLocationDrop, (files) => setBrowserAiLocationFile(Array.from(files || [])[0])],
    [browserAiSingerDrop, (files) => addBrowserAiBandSequenceFiles(files, "band_sequence_singer_references", "group", "Singer", "Singer references")],
    [browserAiExtrasDrop, (files) => addBrowserAiBandSequenceFiles(files, "band_sequence_extra_references", "group", "Extras", "Extras")],
    [browserAiMembersDrop, (files) => addBrowserAiBandSequenceFiles(files, "band_sequence_member_references", "group", "Other Band Members", "Other band member references")],
    [browserAiLocationsDrop, (files) => addBrowserAiBandSequenceFiles(files, "band_sequence_location_references", "location", "Locations", "Locations")],
  ]) {
    dropZone.addEventListener("dragover", (event) => {
      if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
      event.preventDefault();
      event.stopPropagation();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    });
    dropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      event.stopPropagation();
      Promise.resolve(handler(event.dataTransfer?.files || [])).catch((error) => {
        browserAiGroupStatus.textContent = String(error?.message || error);
        toast(String(error?.message || error), true);
      });
    });
  }

  browserAiSendButton.onclick = () => {
    const now = Date.now();
    if (browserAiSendButton.disabled || now - browserAiSend.lastStartedAt < 750) return;
    browserAiSend.lastStartedAt = now;
    browserAiSendButton.disabled = true;
    browserAiGroupStatus.textContent = "Send received. Starting the Browser AI request...";
    sendBrowserAiReferenceGroup().catch((error) => {
      browserAiGroupStatus.textContent = `Browser AI send failed:\n${String(error?.message || error)}`;
      toast(String(error?.message || error), true);
    }).finally(() => {
      browserAiSendButton.disabled = false;
    });
  };
  browserAiFinishButton.onclick = () => finishBrowserAiDownloadSession().catch((error) => {
    browserAiGroupStatus.textContent = `Could not restore browser downloads:\n${String(error?.message || error)}`;
    toast(String(error?.message || error), true);
  });
  flowNanoProviderButton.onclick = () => setFlowGptProvider(BROWSER_IMAGE_PROVIDERS.FLOW_NANO_BANANA);
  gptImageProviderButton.onclick = () => setFlowGptProvider(BROWSER_IMAGE_PROVIDERS.GPT_IMAGE);
  metaImageProviderButton.onclick = () => setFlowGptProvider(BROWSER_IMAGE_PROVIDERS.META_AI);
  flowGptStatusButton.onclick = () => refreshFlowGptBrowserStatus().catch((error) => toast(String(error?.message || error), true));
  flowGptSetupButton.onclick = async () => {
    flowGptSetupButton.disabled = true;
    flowGptStatusText.textContent = "Installing browser automation dependencies...";
    try {
      const data = await setupBrowserImageAutomation({ install_portable_node: true, install_if_missing: true, strict_ssl: false });
      flowGptStatusText.textContent = data.status || formatBrowserImageStatus(data);
      toast("Browser automation setup finished.");
    } catch (error) {
      flowGptStatusText.textContent = `Setup failed:\n${String(error?.message || error)}`;
      toast(String(error?.message || error), true);
    } finally {
      flowGptSetupButton.disabled = false;
    }
  };
  flowGptLoginButton.onclick = async () => {
    try {
      const settings = saveFlowGptBrowserSettingsFromPanel();
      await openBrowserImageLogin(settings.provider, {
        debug_port: browserImageProviderDebugPort(settings.provider),
        timeoutMs: 60000,
      });
      flowGptStatusText.textContent = browserImageLoginStatus(settings.provider, settings);
      toast(`${browserImageProviderLabel(settings.provider)} login browser opened.`);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  };
  flowGptCreateImageButton.onclick = previewFlowGptImage;
  flowGptManualOpenButton.onclick = () => openManualFlowGptBrowser().catch((error) => {
    flowGptManualStatus.textContent = `Manual browser open failed:\n${String(error?.message || error)}`;
    toast(String(error?.message || error), true);
  });
  flowGptManualExportRefsButton.onclick = () => exportManualFlowGptRefs().catch((error) => {
    flowGptManualStatus.textContent = `Manual ref export failed:\n${String(error?.message || error)}`;
    toast(String(error?.message || error), true);
  });
  flowGptManualImportLatestButton.onclick = () => importLatestManualFlowGptDownload().catch((error) => {
    flowGptManualStatus.textContent = `Latest manual download import failed:\n${String(error?.message || error)}`;
    toast(String(error?.message || error), true);
  });
}

export function createBrowserAi({
  activeSegment, addSceneImageHistoryPath, autoSaveSessionQuiet, browserAiAutoAdvanceGroup,
  browserAiBandSequenceMode, browserAiBandSequencePanel, browserAiCustomGroupsPanel,
  browserAiDeleteGroupButton, browserAiDownloadOverrideProviders, browserAiExtrasList, browserAiFinishButton,
  browserAiGroupList, browserAiGroupPrompt, browserAiGroupSelect, browserAiGroupStatus, browserAiLocationList,
  browserAiLocationsList, browserAiMembersList, browserAiSendButton, browserAiSequenceLocationSelect,
  browserAiSequenceProgress, browserAiSequenceSetSelect, browserAiSingerList,
  flowGptBrowserSettingsForSegment, flowGptManualAutoAdvance, flowGptManualChatPrompt,
  flowGptManualExportRefsButton, flowGptManualImportLatestButton, flowGptManualMode, flowGptManualOpenButton,
  flowGptManualStatus, flowGptStatusText, moveActiveSceneSelection, projectInput, pushHistory, render,
  requireActiveSegment, saveFlowGptBrowserSettingsFromPanel, sceneDisplayName, sceneSlotNumber,
  segmentIndexInfo, setActiveSegment, state, syncPreview, zEnhancePromptPreview,
}) {
  function activeBrowserAiReferenceGroup(settings = state.flowGptBrowserSettings) {
    const normalized = Array.isArray(settings?.reference_groups) && settings.reference_groups.length
      ? settings
      : cloneFlowGptBrowserSettings(settings);
    return normalized.reference_groups.find((group) => group.id === normalized.active_reference_group_id)
      || normalized.reference_groups[0];
  }

  function browserAiBandSequenceSelection(settings = state.flowGptBrowserSettings) {
    const singerReferences = Array.isArray(settings?.band_sequence_singer_references) ? settings.band_sequence_singer_references : [];
    const extraReferences = Array.isArray(settings?.band_sequence_extra_references) ? settings.band_sequence_extra_references : [];
    const memberReferences = Array.isArray(settings?.band_sequence_member_references) ? settings.band_sequence_member_references : [];
    const locations = Array.isArray(settings?.band_sequence_location_references) ? settings.band_sequence_location_references : [];
    const locationIndex = locations.length
      ? Math.max(0, Math.min(locations.length - 1, Number(settings?.band_sequence_location_index || 0)))
      : 0;
    const singerCount = singerReferences.length ? 1 : 0;
    const extraCount = extraReferences.length;
    const memberCount = memberReferences.length;
    const definitions = [
      {
        key: "singer",
        label: "Singer only",
        references: singerReferences,
        characterCount: singerCount,
        instruction: "SUBJECT SET: SINGER ONLY. INCLUDE THE SINGER AND DO NOT INCLUDE ANY OTHER BAND MEMBERS.",
      },
      ...(extraCount ? [{
        key: "singer_extras",
        label: "Singer + Extras",
        references: [...singerReferences, ...extraReferences],
        characterCount: singerCount + extraCount,
        instruction: `SUBJECT SET: SINGER WITH EXTRAS. TREAT THE SINGER AND THE ${extraCount} EXTRA CHARACTER${extraCount === 1 ? "" : "S"} AS ONE GROUP AND INCLUDE THEM TOGETHER IN EVERY IMAGE. DO NOT INCLUDE ANY BAND MEMBERS. DO NOT ADD ANY OTHER PEOPLE.`,
      }] : []),
      {
        key: "members",
        label: "Other band members only",
        references: memberReferences,
        characterCount: memberCount,
        instruction: `SUBJECT SET: OTHER BAND MEMBERS ONLY. INCLUDE THE ${memberCount} OTHER BAND MEMBER${memberCount === 1 ? "" : "S"} AND DO NOT INCLUDE THE SINGER.`,
      },
      {
        key: "full_band",
        label: "Full band with singer",
        references: [...singerReferences, ...memberReferences],
        characterCount: singerCount + memberCount,
        instruction: `SUBJECT SET: FULL BAND. INCLUDE THE SINGER AND ALL ${memberCount} OTHER BAND MEMBER${memberCount === 1 ? "" : "S"}.`,
      },
    ];
    const setIndex = Math.max(0, Math.min(definitions.length - 1, Number(settings?.band_sequence_set_index || 0)));
    if (settings) {
      settings.band_sequence_location_index = locationIndex;
      settings.band_sequence_set_index = setIndex;
    }
    const definition = definitions[setIndex];
    return {
      ...definition,
      setIndex,
      locationIndex,
      location: locations[locationIndex] || null,
      locationCount: locations.length,
      setCount: definitions.length,
      singerCount,
      extraCount,
      memberCount,
    };
  }

  function updateBrowserAiPromptCharacterCount(settings) {
    const bandSelection = settings.band_sequence_enabled ? browserAiBandSequenceSelection(settings) : null;
    const group = bandSelection ? null : activeBrowserAiReferenceGroup(settings);
    const characterCount = bandSelection ? bandSelection.characterCount : group.images.length;
    const sequenceKey = bandSelection
      ? `band:${bandSelection.key}:${characterCount}`
      : `group:${group.id}:${characterCount}`;
    const previousCount = Math.max(0, Number(settings.browser_ai_prompt_character_count || 0));
    const previousSequenceKey = String(settings.browser_ai_prompt_sequence_key || "");
    if (!characterCount || (previousCount === characterCount && previousSequenceKey === sequenceKey)) return false;

    const legacyDefault = defaultFlowGptBrowserSettings().manual_chat_prompt.trim();
    const currentPrompt = String(settings.manual_chat_prompt || "").trim();
    if (!currentPrompt || currentPrompt === legacyDefault) {
      settings.manual_chat_prompt = browserAiCharacterCountPrompt(
        characterCount,
        bandSelection?.instruction || "",
        bandSelection?.key === "singer_extras",
      );
    } else {
      const singular = characterCount === 1;
      const characterLabel = singular ? "character" : "characters";
      const placementSubject = singular ? "the character" : `all ${characterCount} characters`;
      let nextPrompt = currentPrompt
        .replace(/Place (?:the character|all \d+ characters) naturally/gi, `Place ${placementSubject} naturally`)
        .replace(/Integrate (?:the character|all \d+ characters) into/gi, `Integrate ${placementSubject} into`)
        .replace(/pasting (?:the character|the characters) onto/gi, `pasting ${singular ? "the character" : "the characters"} onto`);
      const closeUpLine = `I need a close up wide angle image of the ${characterCount} ${characterLabel}.`;
      const exactCountLine = bandSelection?.key === "singer_extras"
        ? `ONLY INCLUDE THE ${characterCount} ${characterLabel.toUpperCase()}. NO ADDITIONAL CHARACTERS.`
        : `ONLY INCLUDE THE ${characterCount} ${characterLabel.toUpperCase()}. NO EXTRAS.`;
      if (/I need a close up wide angle image of the \d+ characters?\.?/i.test(nextPrompt)) {
        nextPrompt = nextPrompt.replace(/I need a close up wide angle image of the \d+ characters?\.?/gi, closeUpLine);
      } else {
        nextPrompt += `\n${closeUpLine}`;
      }
      if (!/DO NOT SEND A GRID OF IMAGES/i.test(nextPrompt)) {
        nextPrompt += "\nDO NOT SEND A GRID OF IMAGES!!! SEND EACH IMAGE AS ITS OWN IMAGE";
      }
      if (/ONLY INCLUDE THE \d+ CHARACTERS?(?:\. (?:NO EXTRAS|NO ADDITIONAL CHARACTERS)\.| (?:NO EXTRAS|NO ADDITIONAL CHARACTERS))?/i.test(nextPrompt)) {
        nextPrompt = nextPrompt.replace(/ONLY INCLUDE THE \d+ CHARACTERS?(?:\. (?:NO EXTRAS|NO ADDITIONAL CHARACTERS)\.| (?:NO EXTRAS|NO ADDITIONAL CHARACTERS))?/gi, exactCountLine);
      } else {
        nextPrompt += `\n${exactCountLine}`;
      }
      nextPrompt = nextPrompt.replace(/\n?SUBJECT SET:[^\n]*/gi, "").trim();
      if (bandSelection?.instruction) nextPrompt += `\n${bandSelection.instruction}`;
      settings.manual_chat_prompt = nextPrompt;
    }
    settings.browser_ai_prompt_character_count = characterCount;
    settings.browser_ai_prompt_sequence_key = sequenceKey;
    return true;
  }

  async function storeBrowserAiReferenceFile(file, referenceType, groupName) {
    const isImage = file && (/^image\//i.test(file.type || "") || /\.(png|jpe?g|webp)$/i.test(file.name || ""));
    if (!isImage) throw new Error("Use a PNG, JPG, JPEG, or WEBP reference image.");
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    if (!projectFolder) throw new Error("Create or load a project before adding Browser AI references.");
    const imageData = await readFileAsDataUrl(file);
    const saved = await storeBrowserImageReference({
      project_folder: projectFolder,
      group_name: groupName,
      reference_type: referenceType,
      image_data: imageData,
      name: file.name || "reference.png",
      timeoutMs: 120000,
    });
    return { path: saved.saved_path || "", data: "", name: file.name || saved.name || "reference.png" };
  }

  async function addBrowserAiGroupFiles(files) {
    const imageFiles = Array.from(files || []).filter((file) => /^image\//i.test(file.type || "") || /\.(png|jpe?g|webp)$/i.test(file.name || ""));
    if (!imageFiles.length) return;
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    const group = activeBrowserAiReferenceGroup(settings);
    const remaining = Math.max(0, 50 - group.images.length - (settings.location_reference ? 1 : 0));
    if (!remaining) throw new Error("Browser AI supports up to 50 reference images per request.");
    const accepted = imageFiles.slice(0, remaining);
    browserAiGroupStatus.textContent = `Saving ${accepted.length} reference image${accepted.length === 1 ? "" : "s"} into the project...`;
    for (const file of accepted) {
      group.images.push(await storeBrowserAiReferenceFile(file, "group", group.name));
    }
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    await autoSaveSessionQuiet("Browser AI group references added");
    browserAiGroupStatus.textContent = `${group.name} now contains ${group.images.length} reference image${group.images.length === 1 ? "" : "s"}.`;
  }

  async function addBrowserAiBandSequenceFiles(files, settingKey, referenceType, folderName, label) {
    const imageFiles = Array.from(files || []).filter((file) => /^image\//i.test(file.type || "") || /\.(png|jpe?g|webp)$/i.test(file.name || ""));
    if (!imageFiles.length) return;
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    const target = settings[settingKey];
    const isLocationList = settingKey === "band_sequence_location_references";
    const subjectReferenceCount = settings.band_sequence_singer_references.length
      + settings.band_sequence_extra_references.length
      + settings.band_sequence_member_references.length;
    const remaining = isLocationList ? Math.max(0, 200 - target.length) : Math.max(0, 49 - subjectReferenceCount);
    if (!remaining) {
      throw new Error(isLocationList
        ? "Band Sequence supports up to 200 saved locations."
        : "Singer, extra, and other-member references can use up to 49 images so one location can be included in each request.");
    }
    const accepted = imageFiles.slice(0, remaining);
    browserAiGroupStatus.textContent = `Saving ${accepted.length} ${label.toLowerCase()} image${accepted.length === 1 ? "" : "s"} into the project...`;
    for (const file of accepted) {
      target.push(await storeBrowserAiReferenceFile(file, referenceType, folderName));
    }
    if (settingKey === "band_sequence_extra_references") {
      settings.band_sequence_set_index = 0;
      settings.browser_ai_prompt_sequence_key = "";
    }
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    await autoSaveSessionQuiet(`Browser AI ${label.toLowerCase()} added`);
    browserAiGroupStatus.textContent = `${label}: ${target.length} saved image${target.length === 1 ? "" : "s"}.`;
  }

  function clearBrowserAiBandSequenceFiles(settingKey, label) {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    if (settings[settingKey].length && !window.confirm(`Remove all ${label.toLowerCase()} from Band Sequence? Stored image files will be kept.`)) return;
    settings[settingKey] = [];
    if (settingKey === "band_sequence_location_references") {
      settings.band_sequence_location_index = 0;
      settings.band_sequence_set_index = 0;
    } else if (settingKey === "band_sequence_extra_references") {
      settings.band_sequence_set_index = 0;
    }
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    browserAiGroupStatus.textContent = `${label} cleared. Stored files were not deleted.`;
    autoSaveSessionQuiet(`Browser AI ${label.toLowerCase()} cleared`).catch(() => null);
  }

  async function setBrowserAiLocationFile(file) {
    if (!file) return;
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    const group = activeBrowserAiReferenceGroup(settings);
    if (group.images.length >= 50 && !settings.location_reference) {
      throw new Error("Remove one group image before adding a location; Browser AI supports 50 references per request.");
    }
    browserAiGroupStatus.textContent = "Saving the location reference into the project...";
    settings.location_reference = await storeBrowserAiReferenceFile(file, "location", group.name);
    if (settings.auto_advance_reference_group !== false && settings.reference_groups.length) {
      settings.active_reference_group_id = settings.reference_groups[0].id;
    }
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    await autoSaveSessionQuiet("Browser AI location reference changed");
    const selectedGroup = activeBrowserAiReferenceGroup(state.flowGptBrowserSettings);
    browserAiGroupStatus.textContent = `Location ready: ${browserAiLocationDisplayName(settings.location_reference)}. Type a location name above if the filename is not descriptive. Starting group: ${selectedGroup.name}.`;
  }

  function chooseBrowserAiImageFiles({ multiple = false, onFiles }) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp";
    input.multiple = multiple;
    input.style.display = "none";
    document.body.append(input);
    input.onchange = () => {
      const files = Array.from(input.files || []);
      input.remove();
      Promise.resolve(onFiles(files)).catch((error) => {
        browserAiGroupStatus.textContent = String(error?.message || error);
        toast(String(error?.message || error), true);
      });
    };
    input.click();
  }

  function advanceBrowserAiSelectionAfterSubmit(completed) {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    if (settings.auto_advance_reference_group === false) {
      return { nextLabel: "", cycleComplete: false };
    }
    if (completed.mode === "band_sequence") {
      const locations = settings.band_sequence_location_references;
      const setCount = browserAiBandSequenceSelection(settings).setCount;
      if (completed.setIndex < setCount - 1) {
        settings.band_sequence_location_index = completed.locationIndex;
        settings.band_sequence_set_index = completed.setIndex + 1;
      } else if (completed.locationIndex < locations.length - 1) {
        settings.band_sequence_location_index = completed.locationIndex + 1;
        settings.band_sequence_set_index = 0;
      } else {
        return { nextLabel: "", cycleComplete: true };
      }
      state.flowGptBrowserSettings = settings;
      renderBrowserAiReferenceGroups();
      const next = browserAiBandSequenceSelection(state.flowGptBrowserSettings);
      return {
        nextLabel: `${next.label} at ${browserAiLocationDisplayName(next.location, next.locationIndex)}`,
        cycleComplete: false,
      };
    }
    const completedIndex = settings.reference_groups.findIndex((item) => item.id === completed.groupId);
    if (completedIndex < 0 || completedIndex >= settings.reference_groups.length - 1) {
      return { nextLabel: "", cycleComplete: completedIndex >= 0 };
    }
    const nextGroup = settings.reference_groups[completedIndex + 1];
    settings.active_reference_group_id = nextGroup.id;
    state.flowGptBrowserSettings = settings;
    renderBrowserAiReferenceGroups();
    return { nextLabel: activeBrowserAiReferenceGroup(state.flowGptBrowserSettings).name, cycleComplete: false };
  }

  async function sendBrowserAiReferenceGroup() {
    const settings = saveFlowGptBrowserSettingsFromPanel();
    const bandSelection = settings.band_sequence_enabled ? browserAiBandSequenceSelection(settings) : null;
    const group = bandSelection
      ? { id: `band_sequence_${bandSelection.locationIndex}_${bandSelection.setIndex}`, name: bandSelection.label, images: bandSelection.references }
      : activeBrowserAiReferenceGroup(settings);
    const location = bandSelection ? bandSelection.location : settings.location_reference;
    const references = [...group.images, ...(location ? [location] : [])];
    const prompt = String(browserAiGroupPrompt.value || settings.manual_chat_prompt || "").trim();
    if (bandSelection) {
      if (!settings.band_sequence_singer_references.length) throw new Error("Add at least one singer reference before using Band Sequence.");
      if (!settings.band_sequence_member_references.length) throw new Error("Add the other band member references before using Band Sequence.");
      if (!location) throw new Error("Add at least one location before using Band Sequence.");
      if (!group.images.length) throw new Error(`${bandSelection.label} does not have any subject references.`);
    } else if (!references.length) {
      throw new Error("Add at least one group or location reference image before sending.");
    }
    if (!prompt) throw new Error("Enter the Browser AI generation prompt before sending.");
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    const existingProjectDownloadSession = browserAiDownloadOverrideProviders.has(settings.provider);
    const redirectDownloads = existingProjectDownloadSession || (Boolean(projectFolder) && window.confirm(
      `Save manual downloads from this Browser AI session into the project?\n\n${projectFolder}\\Browser AI Images\n\nChoose Cancel to keep the controlled browser's normal Downloads folder.`
    ));
    const providerLabel = browserImageProviderShortLabel(settings.provider);
    const locationName = location
      ? browserAiLocationDisplayName(location, bandSelection?.locationIndex || 0)
      : "No Location";
    const selectionName = bandSelection
      ? `${bandSelection.label} at ${locationName}`
      : group.name;
    browserAiGroupStatus.textContent = `Preparing ${providerLabel}, sending ${selectionName} with ${group.images.length} subject reference${group.images.length === 1 ? "" : "s"}${location ? " plus the location" : ""}, entering the prompt, and submitting it...`;
    const data = await submitManualBrowserImageRequest(settings.provider, {
        provider: settings.provider,
        debug_port: browserImageProviderDebugPort(settings.provider),
        timeout_seconds: browserImageProviderTimeout(settings),
        image_ingredients: references,
        prompt,
        project_folder: projectFolder,
        group_name: group.name,
        download_location_name: locationName,
        download_set_name: bandSelection?.label || group.name,
        redirect_downloads_to_project: redirectDownloads,
        open_new_tab: false,
        fresh_request: true,
        timeoutMs: Math.max(300000, (browserImageProviderTimeout(settings) + 30) * 1000),
    });
    if (data.redirect_downloads_to_project) browserAiDownloadOverrideProviders.add(settings.provider);
    else browserAiDownloadOverrideProviders.delete(settings.provider);
    browserAiFinishButton.disabled = browserAiDownloadOverrideProviders.size === 0;
    const completed = bandSelection
      ? { mode: "band_sequence", locationIndex: bandSelection.locationIndex, setIndex: bandSelection.setIndex }
      : { mode: "custom_group", groupId: group.id };
    const { nextLabel, cycleComplete } = advanceBrowserAiSelectionAfterSubmit(completed);
    const submissionLocation = `Submitted ${selectionName} in the existing ${data.provider_label || providerLabel} tab.`;
    const sequenceStatus = nextLabel
      ? ` Next selection ready: ${nextLabel}. Download the current results, then click ${settings.band_sequence_enabled ? "Send Selected Set" : "Send Selected Group"} whenever you are ready.`
      : cycleComplete
        ? " Sequence complete. The last selection remains active."
        : " The current selection remains active.";
    browserAiGroupStatus.textContent = data.redirect_downloads_to_project
      ? `${submissionLocation}${sequenceStatus}\nDownloads from this controlled browser are temporarily going to:\n${data.download_path}\n\nClick Finish Session + Restore Downloads when you are done.`
      : `${submissionLocation}${sequenceStatus}\nReview and download the images manually using the browser's normal Downloads folder.`;
    toast(`${selectionName} submitted to ${data.provider_label || providerLabel}.`);
    void autoSaveSessionQuiet("Browser AI reference selection submitted");
  }

  async function finishBrowserAiDownloadSession({ quiet = false } = {}) {
    const providers = [...browserAiDownloadOverrideProviders];
    if (!providers.length) return;
    for (const provider of providers) {
      await finishManualBrowserImageSession(provider, {
        debug_port: browserImageProviderDebugPort(provider),
        timeout_seconds: 60,
        timeoutMs: 60000,
      });
      browserAiDownloadOverrideProviders.delete(provider);
    }
    browserAiFinishButton.disabled = true;
    if (!quiet) {
      browserAiGroupStatus.textContent = "Browser AI session finished. Downloads were restored to the browser default.";
      toast("Browser downloads restored to the normal folder.");
    }
  }

  async function restoreBrowserAiDownloadsQuietly() {
    try {
      await finishBrowserAiDownloadSession({ quiet: true });
    } catch (error) {
      console.warn("[VRGDG Music Builder] Could not restore Browser AI downloads while closing:", error);
    }
  }

  function manualFlowGptPayload(segment = activeSegment()) {
    const settings = flowGptBrowserSettingsForSegment(segment);
    return {
      provider: settings.provider,
      debug_port: browserImageProviderDebugPort(settings.provider),
      timeout_seconds: browserImageProviderTimeout(settings),
    };
  }

  function manualSceneReferenceImages(segment = activeSegment()) {
    const settings = flowGptBrowserSettingsForSegment(segment);
    const seen = new Set();
    return (settings.image_ingredients || [])
      .map((item) => ({
        path: String(item?.path || "").trim(),
        data: String(item?.data || "").trim(),
        name: String(item?.name || "").trim(),
      }))
      .filter((item) => item.path || item.data)
      .filter((item) => {
        const key = item.path || item.data || item.name;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
  }

  async function openManualFlowGptBrowser() {
    const settings = saveFlowGptBrowserSettingsFromPanel();
    const providerLabel = browserImageProviderShortLabel(settings.provider);
    flowGptManualStatus.textContent = `Opening ${providerLabel} manual browser...`;
    const data = await openManualBrowserImageProvider(settings.provider, {
      ...manualFlowGptPayload(),
      timeoutMs: 60000,
    });
    browserAiDownloadOverrideProviders.delete(settings.provider);
    browserAiFinishButton.disabled = browserAiDownloadOverrideProviders.size === 0;
    flowGptManualStatus.textContent = `${data.provider_label || providerLabel} manual browser opened.\nDownload import target: selected scene when you arm it.`;
    toast(`${data.provider_label || providerLabel} manual browser opened.`);
  }

  async function exportManualFlowGptRefs() {
    const segment = requireActiveSegment();
    if (!segment) return;
    const refs = manualSceneReferenceImages(segment);
    if (!refs.length) {
      toast("No Flow/GPT reference images found for the selected scene.", true);
      flowGptManualStatus.textContent = "No reference images found for this scene.";
      return;
    }
    const settings = saveFlowGptBrowserSettingsFromPanel();
    const providerLabel = browserImageProviderShortLabel(settings.provider);
    flowGptManualExportRefsButton.disabled = true;
    flowGptManualStatus.textContent = `Exporting ${refs.length} reference image${refs.length === 1 ? "" : "s"} to ${providerLabel}...`;
    try {
      await uploadManualBrowserImageRefs(settings.provider, {
        ...manualFlowGptPayload(segment),
        image_ingredients: refs,
        prompt: settings.manual_chat_prompt,
        timeoutMs: 300000,
      });
      browserAiDownloadOverrideProviders.delete(settings.provider);
      browserAiFinishButton.disabled = browserAiDownloadOverrideProviders.size === 0;
      flowGptManualStatus.textContent = `Exported ${refs.length} reference image${refs.length === 1 ? "" : "s"} and copied the editable prompt to ${providerLabel}.`;
      toast(`Exported refs and copied the prompt to ${providerLabel}.`);
    } finally {
      flowGptManualExportRefsButton.disabled = !flowGptManualMode.input.checked;
    }
  }

  async function importLatestManualFlowGptDownload() {
    const segment = requireActiveSegment();
    if (!segment) return;
    const projectFolder = projectInput.value || state.projectFolder;
    if (!projectFolder) {
      toast("Create or load a project before importing a manual browser download.", true);
      return;
    }
    const settings = saveFlowGptBrowserSettingsFromPanel();
    const providerLabel = browserImageProviderShortLabel(settings.provider);
    const sceneLabel = sceneDisplayName(segment, segmentIndexInfo(segment).index);
    flowGptManualImportLatestButton.disabled = true;
    flowGptManualStatus.textContent = `Importing latest ${providerLabel} download -> ${sceneLabel}...`;
    try {
      const data = await importLatestManualBrowserImageDownload(settings.provider, {
        ...manualFlowGptPayload(segment),
        project_folder: projectFolder,
        scene_number: sceneSlotNumber(segment),
        timeoutMs: 120000,
      });
      await applyManualFlowGptImportResult(segment, data, providerLabel, sceneLabel);
    } finally {
      flowGptManualImportLatestButton.disabled = !flowGptManualMode.input.checked;
    }
  }

  async function applyManualFlowGptImportResult(segment, data, providerLabel, sceneLabel) {
    const savedPath = data?.scene_image?.saved_path || data?.saved_path || "";
    if (!savedPath) throw new Error("Manual browser import did not return a saved image path.");
    pushHistory();
    addSceneImageHistoryPath(segment, savedPath);
    segment.approved_image_path = savedPath;
    segment.custom_image_path = "";
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.image = null;
    segment.preview_mode = "image";
    setActiveSegment(segment);
    syncPreview(segment);
    render();
    await autoSaveSessionQuiet("manual Flow/GPT image import");
    flowGptManualStatus.textContent = `Imported ${providerLabel} download into ${sceneLabel}.\n${savedPath}`;
    toast(`Imported ${providerLabel} download into ${sceneLabel}.`);
    if (flowGptManualAutoAdvance.input.checked) moveActiveSceneSelection(1);
  }

  async function refreshFlowGptBrowserStatus() {
    flowGptStatusText.textContent = "Checking browser automation setup...";
    try {
      const data = await getBrowserImageStatus();
      flowGptStatusText.textContent = formatBrowserImageStatus(data);
      return data;
    } catch (error) {
      flowGptStatusText.textContent = `Status check failed:\n${String(error?.message || error)}`;
      throw error;
    }
  }

  function flfStillEndpointNotes(segment, target = "start") {
    if (state.videoModelMode !== "flf" || !segment) return "";
    const endpoint = target === "end" ? String(segment.flf_end_state || "").trim() : String(segment.flf_start_state || "").trim();
    const transformation = String(segment.flf_transformation || "").trim();
    const carryForward = String(segment.flf_carry_forward || "").trim();
    return [
      `First / Last Frame storyboard target: ${target === "end" ? "END FRAME" : "START FRAME"} still image.`,
      endpoint ? `Required visible endpoint state:\n${endpoint}` : "",
      target === "end" && transformation ? `Transformation context (use only to choose the correct frozen endpoint):\n${transformation}` : "",
      target === "end" && carryForward ? `Continuity constraints:\n${carryForward}` : "",
      target === "start" ? "This is strictly the untouched opening condition before the scene action begins. Do not include, foreshadow, partially show, or imply anything from the later transformation or destination state." : "",
      "Create one frozen still image matching the required endpoint. Do not describe animation, morphing over time, or workflow terminology in the final prompt.",
    ].filter(Boolean).join("\n\n");
  }

  function imagePromptNotesWithDirector(segment, notes, useDirectorNotes = false) {
    const parts = [];
    const baseNotes = String(notes || "").trim();
    const directorNote = String(segment?.timeline_note || "").trim();
    if (baseNotes) parts.push(baseNotes);
    const flfTarget = /create the LAST FRAME/i.test(baseNotes) ? "end" : "start";
    const flfEndpoint = flfStillEndpointNotes(segment, flfTarget);
    if (flfEndpoint) parts.push(flfEndpoint);
    if (useDirectorNotes && directorNote) {
      parts.push(`Director note for this scene:\n${directorNote}`);
    }
    return parts.join("\n\n");
  }

  function activeScenePromptForEnhance({ copyFallback = false } = {}) {
    const segment = activeSegment();
    const promptInfo = sceneImagePromptForEnhanceAll(segment);
    if (copyFallback && segment && promptInfo.prompt) {
      segment.enhance_prompt = promptInfo.prompt;
      zEnhancePromptPreview.value = promptInfo.prompt;
    }
    return promptInfo;
  }

  function renderBrowserAiReferenceGroups() {
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    updateBrowserAiPromptCharacterCount(settings);
    state.flowGptBrowserSettings = settings;
    const group = activeBrowserAiReferenceGroup(settings);
    browserAiGroupSelect.innerHTML = "";
    settings.reference_groups.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = `${item.name} (${item.images.length})`;
      browserAiGroupSelect.append(option);
    });
    browserAiGroupSelect.value = group.id;
    browserAiGroupList.innerHTML = "";
    if (!group.images.length) {
      const empty = document.createElement("div");
      empty.textContent = "No character or band references in this group yet.";
      empty.style.cssText = "font-size:10px;color:#71717a;padding:2px 1px;";
      browserAiGroupList.append(empty);
    } else {
      group.images.forEach((item, index) => {
        browserAiGroupList.append(browserAiReferenceRow(item, () => {
          const current = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
          const currentGroup = activeBrowserAiReferenceGroup(current);
          currentGroup.images.splice(index, 1);
          state.flowGptBrowserSettings = current;
          renderBrowserAiReferenceGroups();
          autoSaveSessionQuiet("Browser AI group reference removed").catch(() => null);
        }));
      });
    }
    browserAiLocationList.innerHTML = "";
    if (settings.location_reference) {
      browserAiLocationList.append(browserAiLocationReferenceRow(
        settings.location_reference,
        0,
        () => {
          const current = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
          current.location_reference = null;
          state.flowGptBrowserSettings = current;
          renderBrowserAiReferenceGroups();
          autoSaveSessionQuiet("Browser AI location cleared").catch(() => null);
        },
        (locationLabel) => {
          const current = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
          if (!current.location_reference) return;
          current.location_reference.location_label = locationLabel;
          state.flowGptBrowserSettings = current;
          renderBrowserAiReferenceGroups();
          autoSaveSessionQuiet("Browser AI location named").catch(() => null);
        },
      ));
    } else {
      const empty = document.createElement("div");
      empty.textContent = "No location selected. A group can still be sent without one.";
      empty.style.cssText = "font-size:10px;color:#71717a;padding:2px 1px;";
      browserAiLocationList.append(empty);
    }
    const renderSequenceReferences = (container, items, settingKey, emptyText, saveReason) => {
      container.innerHTML = "";
      if (!items.length) {
        const empty = document.createElement("div");
        empty.textContent = emptyText;
        empty.style.cssText = "font-size:10px;color:#71717a;padding:2px 1px;";
        container.append(empty);
        return;
      }
      items.forEach((item, index) => {
        container.append(browserAiReferenceRow(item, () => {
          const current = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
          current[settingKey].splice(index, 1);
          if (settingKey === "band_sequence_location_references") {
            current.band_sequence_location_index = Math.max(0, Math.min(
              current.band_sequence_location_index,
              current.band_sequence_location_references.length - 1,
            ));
          }
          state.flowGptBrowserSettings = current;
          renderBrowserAiReferenceGroups();
          autoSaveSessionQuiet(saveReason).catch(() => null);
        }));
      });
    };
    renderSequenceReferences(
      browserAiSingerList,
      settings.band_sequence_singer_references,
      "band_sequence_singer_references",
      "No singer references added yet.",
      "Browser AI singer reference removed",
    );
    renderSequenceReferences(
      browserAiExtrasList,
      settings.band_sequence_extra_references,
      "band_sequence_extra_references",
      "No optional extras added.",
      "Browser AI extra reference removed",
    );
    renderSequenceReferences(
      browserAiMembersList,
      settings.band_sequence_member_references,
      "band_sequence_member_references",
      "No other band member references added yet.",
      "Browser AI band member reference removed",
    );
    browserAiLocationsList.innerHTML = "";
    if (!settings.band_sequence_location_references.length) {
      const empty = document.createElement("div");
      empty.textContent = "No sequence locations added yet.";
      empty.style.cssText = "font-size:10px;color:#71717a;padding:2px 1px;";
      browserAiLocationsList.append(empty);
    } else {
      settings.band_sequence_location_references.forEach((item, index) => {
        browserAiLocationsList.append(browserAiLocationReferenceRow(
          item,
          index,
          () => {
            const current = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
            current.band_sequence_location_references.splice(index, 1);
            current.band_sequence_location_index = Math.max(0, Math.min(
              current.band_sequence_location_index,
              current.band_sequence_location_references.length - 1,
            ));
            state.flowGptBrowserSettings = current;
            renderBrowserAiReferenceGroups();
            autoSaveSessionQuiet("Browser AI sequence location removed").catch(() => null);
          },
          (locationLabel) => {
            const current = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
            const location = current.band_sequence_location_references[index];
            if (!location) return;
            location.location_label = locationLabel;
            state.flowGptBrowserSettings = current;
            renderBrowserAiReferenceGroups();
            autoSaveSessionQuiet("Browser AI sequence location named").catch(() => null);
          },
        ));
      });
    }
    browserAiSequenceLocationSelect.innerHTML = "";
    settings.band_sequence_location_references.forEach((item, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${index + 1}. ${String(item.location_label || "").trim() || "[Type a location name below]"}`;
      browserAiSequenceLocationSelect.append(option);
    });
    if (!settings.band_sequence_location_references.length) {
      const option = document.createElement("option");
      option.value = "0";
      option.textContent = "Add at least one location";
      browserAiSequenceLocationSelect.append(option);
    }
    const sequenceSelection = browserAiBandSequenceSelection(settings);
    browserAiSequenceSetSelect.innerHTML = "";
    const sequenceDefinitions = [
      "Singer only",
      ...(settings.band_sequence_extra_references.length ? ["Singer + Extras"] : []),
      "Other band members only",
      "Full band with singer",
    ];
    sequenceDefinitions.forEach((label, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${index + 1}. ${label}`;
      browserAiSequenceSetSelect.append(option);
    });
    browserAiSequenceLocationSelect.value = String(sequenceSelection.locationIndex);
    browserAiSequenceSetSelect.value = String(sequenceSelection.setIndex);
    browserAiSequenceProgress.textContent = sequenceSelection.location
      ? `Location ${sequenceSelection.locationIndex + 1} of ${sequenceSelection.locationCount}: ${browserAiLocationDisplayName(sequenceSelection.location, sequenceSelection.locationIndex)} | Set ${sequenceSelection.setIndex + 1} of ${sequenceSelection.setCount}: ${sequenceSelection.label}`
      : "Add singer references, optional extras, other band members, and all locations to begin.";
    browserAiBandSequenceMode.input.checked = Boolean(settings.band_sequence_enabled);
    browserAiBandSequencePanel.style.display = settings.band_sequence_enabled ? "flex" : "none";
    browserAiCustomGroupsPanel.style.display = settings.band_sequence_enabled ? "none" : "flex";
    browserAiDeleteGroupButton.disabled = settings.reference_groups.length <= 1;
    browserAiAutoAdvanceGroup.input.checked = settings.auto_advance_reference_group !== false;
    browserAiSendButton.textContent = settings.band_sequence_enabled ? "Send Selected Set" : "Send Selected Group";
    browserAiGroupPrompt.value = settings.manual_chat_prompt || defaultFlowGptBrowserSettings().manual_chat_prompt;
    flowGptManualChatPrompt.value = browserAiGroupPrompt.value;
    browserAiFinishButton.disabled = browserAiDownloadOverrideProviders.size === 0;
  }

  function syncFlowGptManualPanel() {
    const enabled = Boolean(flowGptManualMode.input.checked);
    for (const control of [flowGptManualAutoAdvance.input, flowGptManualOpenButton, flowGptManualExportRefsButton, flowGptManualImportLatestButton]) {
      control.disabled = !enabled;
      control.style.opacity = enabled ? "1" : "0.62";
    }
    if (!enabled) {
      flowGptManualStatus.textContent = "Manual Mode is off.";
      return;
    }
    const segment = activeSegment();
    const settings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
    const providerLabel = browserImageProviderShortLabel(settings.provider);
    flowGptManualStatus.textContent = segment
      ? `Manual Mode: ${providerLabel} ready for ${sceneDisplayName(segment, segmentIndexInfo(segment).index)}.`
      : `Manual Mode: ${providerLabel} ready. Select a scene before importing downloads.`;
  }

  return {
    activeBrowserAiReferenceGroup, activeScenePromptForEnhance, addBrowserAiBandSequenceFiles,
    addBrowserAiGroupFiles, browserAiBandSequenceSelection, chooseBrowserAiImageFiles,
    clearBrowserAiBandSequenceFiles, exportManualFlowGptRefs, finishBrowserAiDownloadSession,
    flfStillEndpointNotes, imagePromptNotesWithDirector, importLatestManualFlowGptDownload,
    openManualFlowGptBrowser, refreshFlowGptBrowserStatus, renderBrowserAiReferenceGroups,
    restoreBrowserAiDownloadsQuietly, sendBrowserAiReferenceGroup, setBrowserAiLocationFile,
    syncFlowGptManualPanel,
  };
}
