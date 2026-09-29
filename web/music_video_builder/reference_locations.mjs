import { postJson } from "./comfy_api.mjs";
import { LOCATION_MAPPER_GPT_URL } from "./constants.mjs";
import { makeButton, makeField, makeGptLinkButton, makeInput, toast } from "./controls.mjs";
import { sceneConceptPromptText } from "./image_prompts.mjs";
import { normalizeGemmaContextLimit } from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder } from "./reference_data.mjs";

function sceneNumberFromImportKey(key) {
  const text = String(key || "").trim();
  const match = text.match(/(?:scene|segment|prompt|motion|location)?\s*#?\s*(\d+)/i);
  return match ? Number(match[1]) : 0;
}

function parseLocationSceneMapImportText(rawText) {
  const text = String(rawText || "").trim();
  if (!text) return [];
  const normalizedEntry = (entry, fallbackSceneNumber = 0, fallbackKey = "") => {
    if (typeof entry === "string") {
      const location = entry.trim();
      return location ? { sceneNumber: fallbackSceneNumber, key: fallbackKey, location, description: "" } : null;
    }
    if (!entry || typeof entry !== "object") return null;
    const rawScene =
      entry.scene_number
      ?? entry.sceneNumber
      ?? entry.scene
      ?? entry.segment
      ?? entry.segment_number
      ?? entry.segmentNumber
      ?? entry.number
      ?? fallbackSceneNumber
      ?? 0;
    const sceneNumber = typeof rawScene === "number"
      ? rawScene
      : (sceneNumberFromImportKey(rawScene) || Number(rawScene) || fallbackSceneNumber || 0);
    const location = String(
      entry.location
      ?? entry.location_name
      ?? entry.locationName
      ?? entry.name
      ?? entry.setting
      ?? ""
    ).trim();
    const description = String(
      entry.description
      ?? entry.location_description
      ?? entry.locationDescription
      ?? entry.prompt
      ?? entry.details
      ?? ""
    ).trim();
    return location ? { sceneNumber, key: fallbackKey, location, description } : null;
  };
  const fromJsonValue = (value) => {
    const mappings = [];
    if (Array.isArray(value)) {
      value.forEach((entry, index) => {
        const normalized = normalizedEntry(entry, index + 1, String(index + 1));
        if (normalized) mappings.push(normalized);
      });
      return mappings;
    }
    if (value && typeof value === "object") {
      const explicitMap = value.scene_map || value.sceneMap || value.location_map || value.locationMap || value.locations_by_scene || value.locationsByScene;
      if (explicitMap && typeof explicitMap === "object") return fromJsonValue(explicitMap);
      for (const [key, entry] of Object.entries(value)) {
        const sceneNumber = sceneNumberFromImportKey(key);
        const normalized = normalizedEntry(entry, sceneNumber, key);
        if (normalized) mappings.push(normalized);
      }
    }
    return mappings;
  };
  try {
    return fromJsonValue(JSON.parse(text)).filter((item) => item.location);
  } catch {
    return [];
  }
}

export function createReferenceLocations({
  activeSegment, allEditableSegments, autoMapLocations, autoSaveSessionQuiet,
  createDetailedLocationDescriptionWithGemma, createFluxReferenceWithZImage, createProgressWindow,
  describeSingleReferenceWithGemma, exportLocations, extractLocations, gemmaRunnerLine,
  i2vTextGemmaModelSelect, imageTargetFor, locationExtractionStyleTheme, locationKey, locationStyleTheme,
  locationsList, projectReferenceBuilderLocationsPath, referenceImagesEnabled, refs, render, renderAll,
  renderDrop, renderFluxIngredientList, renderMapping, renderNBIngredientList, sceneSlotNumber, state,
  subjectDrop, subjectSceneInput, t2iTextGemmaModelSelect, textGemmaRunnerPayload, uploadFor, useLocations,
  wireDrop, wizardLocationMode,
}) {
  function createLocation(name = "", description = "") {
    refs.locations_cleared = false;
    const location = {
      id: `loc_${Date.now()}_${Math.floor(Math.random() * 10000)}`,
      name: String(name || `Location ${refs.locations.length + 1}`).trim(),
      description: String(description || "").trim(),
      image: { path: "", data: "", name: "" },
    };
    refs.locations.push(location);
    return location;
  }
  function normalizeImportedLocation(item) {
    if (!item || typeof item !== "object") return null;
    const name = String(item.location ?? item.name ?? item.title ?? item.label ?? "").trim();
    const description = String(item.description ?? item.prompt ?? item.details ?? item.notes ?? "").trim();
    if (!name && !description) return null;
    return { name: name || `Location ${refs.locations.length + 1}`, description };
  }
  function parseLocationImportText(rawText) {
    const text = String(rawText || "").trim();
    if (!text) return [];
    const fromJsonValue = (value) => {
      const items = [];
      if (Array.isArray(value)) {
        for (const item of value) {
          const normalized = normalizeImportedLocation(item);
          if (normalized) items.push(normalized);
        }
        return items;
      }
      if (value && typeof value === "object") {
        const list = value.locations || value.LocationList || value.location_list;
        if (Array.isArray(list)) return fromJsonValue(list);
        for (const [key, entry] of Object.entries(value)) {
          if (entry && typeof entry === "object") {
            const normalized = normalizeImportedLocation({ name: key, ...entry });
            if (normalized) items.push(normalized);
          } else if (typeof entry === "string") {
            const sceneLikeKey = sceneNumberFromImportKey(key) > 0;
            items.push(sceneLikeKey
              ? { name: entry.trim(), description: "" }
              : { name: String(key || "").trim(), description: entry.trim() });
          }
        }
      }
      return items.filter((item) => item.name);
    };
    try {
      const parsed = JSON.parse(text);
      const jsonItems = fromJsonValue(parsed);
      if (jsonItems.length) return jsonItems;
    } catch {
      // Fall through to plain-text formats.
    }
    const blockItems = text.split(/\n\s*\n+/)
      .map((block) => block.split(/\r?\n/).map((line) => line.trim()).filter(Boolean))
      .filter((lines) => lines.length >= 2)
      .map((lines) => ({ name: lines[0].replace(/^[-*]\s*/, "").trim(), description: lines.slice(1).join(" ").trim() }))
      .filter((item) => item.name && item.description);
    if (blockItems.length) return blockItems;
    return text.split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const clean = line.replace(/^[-*]\s*/, "");
        const match = clean.match(/^(.+?)\s*(?:=>|=|:|\s+-\s+)\s*(.+)$/);
        return match ? { name: match[1].trim(), description: match[2].trim() } : null;
      })
      .filter((item) => item?.name && item.description);
  }
  function openImportLocationsDialog() {
    const importBackdrop = document.createElement("div");
    importBackdrop.style.cssText = "position:fixed;inset:0;z-index:100009;background:rgba(0,0,0,.65);display:flex;align-items:center;justify-content:center;";
    const importBox = document.createElement("div");
    importBox.style.cssText = "width:min(760px,calc(100vw - 34px));max-height:calc(100vh - 40px);border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.58);display:flex;flex-direction:column;overflow:hidden;";
    const importHeader = document.createElement("div");
    importHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;background:#083f4f;border-bottom:1px solid #155e75;padding:12px 14px;";
    const importTitle = document.createElement("div");
    importTitle.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Import Location List / Scene Map</div><div style="font-size:12px;color:#cbd5e1;margin-top:3px;">Paste locations to create cards, or scene-to-location JSON to auto-map scenes.</div>`;
    const gptTools = document.createElement("div");
    gptTools.style.cssText = "display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px;align-items:center;margin-left:auto;";
    const mapperGpt = makeGptLinkButton("GPT: Locations Only", LOCATION_MAPPER_GPT_URL);
    mapperGpt.style.cssText += "white-space:nowrap;";
    const gptNote = document.createElement("div");
    gptNote.textContent = "Creates scene-location JSON without trigger words.";
    gptNote.style.cssText = "flex-basis:100%;text-align:right;font-size:11px;color:#bae6fd;line-height:1.25;";
    gptTools.append(mapperGpt, gptNote);
    const importClose = makeButton("Close");
    importHeader.append(importTitle, gptTools, importClose);
    const importBody = document.createElement("div");
    importBody.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:10px;overflow:auto;";
    const help = document.createElement("div");
    help.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;font-size:12px;color:#dbeafe;line-height:1.45;";
    help.innerHTML = `
        <div style="font-weight:900;color:#cffafe;margin-bottom:6px;">Accepted formats</div>
        <div>Location list JSON:</div>
        <pre style="white-space:pre-wrap;margin:6px 0 10px;color:#e2e8f0;">[
  { "location": "Glass hallway", "description": "A long mirrored corridor..." },
  { "name": "Chrome vault corridor", "description": "A sealed industrial passage..." }
]</pre>
        <div>Combined location + scene map JSON:</div>
        <pre style="white-space:pre-wrap;margin:6px 0 10px;color:#e2e8f0;">[
  {
    "scene": "scene1",
    "location": "Glass hallway",
    "description": "A long mirrored corridor with black glass walls..."
  },
  {
    "scene": "scene2",
    "location": "Chrome vault corridor",
    "description": "A sealed industrial passage lined with circular vault doors..."
  }
]</pre>
        <div>Scene map JSON:</div>
        <pre style="white-space:pre-wrap;margin:6px 0 10px;color:#e2e8f0;">{
  "scene1": { "location": "Glass hallway" },
  "scene2": { "location": "Chrome vault corridor" }
}</pre>
        <div>Array scene map:</div>
        <pre style="white-space:pre-wrap;margin:6px 0 10px;color:#e2e8f0;">[
  { "scene": 1, "location": "Glass hallway" },
  { "lyricSegment": "[instrumental]", "location": "Chrome vault corridor" }
]</pre>
        <div>Quick text:</div>
        <pre style="white-space:pre-wrap;margin:6px 0 0;color:#e2e8f0;">Glass hallway = A long mirrored corridor...
Chrome vault corridor: A sealed industrial passage...</pre>
        <div style="margin-top:8px;color:#94a3b8;">Scene map imports use scene numbers first. If an array item has no scene number, its order maps to Scene 1, Scene 2, and so on. Missing location cards are created automatically.</div>`;
    const input = document.createElement("textarea");
    input.spellcheck = false;
    input.placeholder = "Paste JSON, scene map JSON, or Name = Description lines here...";
    input.style.cssText = "min-height:280px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;";
    const importActions = document.createElement("div");
    importActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const importCancel = makeButton("Cancel");
    const importApply = makeButton("Import Locations / Map", "primary");
    importActions.append(importCancel, importApply);
    importBody.append(help, input, importActions);
    importBox.append(importHeader, importBody);
    importBackdrop.append(importBox);
    document.body.append(importBackdrop);
    importClose.onclick = () => importBackdrop.remove();
    importCancel.onclick = () => importBackdrop.remove();
    importBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === importBackdrop) importBackdrop.remove();
    });
    importApply.onclick = () => {
      const items = parseLocationImportText(input.value);
      const mappings = parseLocationSceneMapImportText(input.value);
      if (!items.length && !mappings.length) {
        toast("No locations found. Paste location JSON, scene map JSON, or Name = Description lines.", true);
        return;
      }
      const { added, updated } = importLocationItems(items);
      const { mapped, created } = applyImportedLocationSceneMap(mappings);
      refs.use_location_references = true;
      useLocations.input.checked = true;
      renderAll();
      toast(`Imported locations. Added: ${added + created}. Updated: ${updated}. Scene mappings: ${mapped}. Review, then Save Reference Builder.`);
      importBackdrop.remove();
    };
    input.focus();
  }

  function openImportLocationSourceDialog() {
    const choiceBackdrop = document.createElement("div");
    choiceBackdrop.style.cssText = "position:fixed;inset:0;z-index:100009;background:rgba(0,0,0,.65);display:flex;align-items:center;justify-content:center;";
    const choiceBox = document.createElement("div");
    choiceBox.style.cssText = "width:min(620px,calc(100vw - 34px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.58);display:flex;flex-direction:column;overflow:hidden;";
    const choiceHeader = document.createElement("div");
    choiceHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;background:#083f4f;border-bottom:1px solid #155e75;padding:12px 14px;";
    const choiceTitle = document.createElement("div");
    choiceTitle.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Import Locations</div><div style="font-size:12px;color:#cbd5e1;margin-top:3px;">Choose how you want to add location references.</div>`;
    const choiceClose = makeButton("Close");
    choiceHeader.append(choiceTitle, choiceClose);
    const choiceBody = document.createElement("div");
    choiceBody.style.cssText = "padding:14px;display:grid;grid-template-columns:1fr 1fr;gap:12px;";
    const jsonButton = makeButton("Paste JSON / Text", "primary");
    const folderButton = makeButton("Import Folder", "primary");
    const jsonCard = document.createElement("div");
    jsonCard.style.cssText = "border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;display:flex;flex-direction:column;gap:10px;";
    jsonCard.innerHTML = `<div style="font-weight:900;color:#cffafe;">JSON or text list</div><div style="font-size:12px;color:#cbd5e1;line-height:1.45;">Paste location cards, scene maps, or combined scene + location JSON. This is the current importer.</div>`;
    jsonCard.append(jsonButton);
    const folderCard = document.createElement("div");
    folderCard.style.cssText = "border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;display:flex;flex-direction:column;gap:10px;";
    folderCard.innerHTML = `<div style="font-weight:900;color:#cffafe;">Project folder</div><div style="font-size:12px;color:#cbd5e1;line-height:1.45;">Imports image files and matching .txt descriptions from:<br><code>subject_location/location</code></div>`;
    folderCard.append(folderButton);
    choiceBody.append(jsonCard, folderCard);
    choiceBox.append(choiceHeader, choiceBody);
    choiceBackdrop.append(choiceBox);
    document.body.append(choiceBackdrop);
    const closeChoice = () => choiceBackdrop.remove();
    choiceClose.onclick = closeChoice;
    choiceBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === choiceBackdrop) closeChoice();
    });
    jsonButton.onclick = () => {
      closeChoice();
      openImportLocationsDialog();
    };
    folderButton.onclick = () => {
      closeChoice();
      importLocationsFromProjectFolder();
    };
  }

  async function exportReferenceBuilderLocations() {
    const path = projectReferenceBuilderLocationsPath();
    if (!path) {
      toast("Create or load a project before exporting locations.", true);
      return;
    }
    const locationById = new Map((refs.locations || []).map((location) => [String(location.id || ""), location]));
    const exportLocationJson = (location = {}) => {
      const item = { name: String(location.name || "").trim() };
      const description = String(location.description || "").trim();
      const trigger = String(location.trigger_phrase || "").trim();
      if (description) item.description = description;
      if (trigger) item.trigger = trigger;
      return item;
    };
    const exportSceneLocationJson = (location = null) => {
      const item = { location: String(location?.name || "").trim() };
      const description = String(location?.description || "").trim();
      const trigger = String(location?.trigger_phrase || "").trim();
      if (description) item.description = description;
      if (trigger) item.trigger = trigger;
      return item;
    };
    const exportData = {
      locations: (refs.locations || []).map(exportLocationJson).filter((location) => location.name),
      scene_map: {},
    };
    allEditableSegments().forEach((segment, index) => {
      const sceneNumber = index + 1;
      const locationId = String(refs.scene_map?.[segment.id] || refs.scene_map?.[String(sceneNumber)] || "").trim();
      const location = locationById.get(locationId) || null;
      const name = String(location?.name || "").trim();
      exportData.scene_map[`scene${sceneNumber}`] = exportSceneLocationJson(location);
    });
    try {
      exportLocations.disabled = true;
      exportLocations.textContent = "Exporting...";
      const result = await postJson("/vrgdg/music_builder/save_text_file", {
        path,
        content: JSON.stringify(exportData, null, 2),
      }, 60000);
      const mappedCount = Object.values(exportData.scene_map).filter((value) => String(value?.location || "").trim()).length;
      toast(`Exported ${exportData.locations.length} location${exportData.locations.length === 1 ? "" : "s"} and ${mappedCount} scene map${mappedCount === 1 ? "" : "s"} to:\n${result.path || path}`);
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      exportLocations.disabled = false;
      exportLocations.textContent = "Export";
    }
  }
  function locationByName(name) {
    return refs.locations.find((item) => locationKey(item.name) === locationKey(name));
  }
  function importLocationItems(items) {
    let added = 0;
    let updated = 0;
    const seen = new Set();
    for (const item of items) {
      const name = String(item?.name || "").trim();
      const description = String(item?.description || "").trim();
      const key = locationKey(name);
      if (!key || seen.has(key)) continue;
      seen.add(key);
      const existing = locationByName(name);
      if (existing) {
        existing.name = name;
        if (description) existing.description = description;
        updated += 1;
      } else {
        createLocation(name, description);
        added += 1;
      }
    }
    return { added, updated };
  }
  function applyImportedLocationSceneMap(mappings) {
    if (!refs.scene_map || typeof refs.scene_map !== "object") refs.scene_map = {};
    let mapped = 0;
    let created = 0;
    const segments = allEditableSegments();
    const segmentForMapping = (mapping, index) => {
      const sceneNumber = Number(mapping.sceneNumber || 0);
      if (sceneNumber > 0) {
        const byIndex = segments[sceneNumber - 1];
        if (byIndex) return byIndex;
        const bySlot = segments.find((segment) => sceneSlotNumber(segment) === sceneNumber);
        if (bySlot) return bySlot;
      }
      if (mapping.key) {
        const byId = segments.find((segment) => String(segment.id) === String(mapping.key));
        if (byId) return byId;
      }
      return segments[index] || null;
    };
    mappings.forEach((mapping, index) => {
      const locationName = String(mapping?.location || "").trim();
      if (!locationName) return;
      let location = locationByName(locationName);
      if (!location) {
        location = createLocation(locationName, mapping.description || "");
        created += 1;
      } else if (!String(location.description || "").trim() && mapping.description) {
        location.description = String(mapping.description || "").trim();
      }
      const segment = segmentForMapping(mapping, index);
      if (!segment) return;
      refs.scene_map[segment.id] = location.id;
      mapped += 1;
    });
    return { mapped, created };
  }

  async function importLocationsFromProjectFolder() {
    const projectFolder = String(state.projectFolder || "").trim();
    const progress = createProgressWindow("Importing Locations", { zIndex: 100008 });
    try {
      if (!projectFolder) throw new Error("Create or load a project first so the location folder can be found.");
      progress.set(`Looking for location images and descriptions...\n${projectFolder}\\subject_location\\location`, 20);
      const data = await postJson("/vrgdg/music_builder/import_reference_locations", {
        project_folder: projectFolder,
      }, 60000);
      const imported = Array.isArray(data.locations) ? data.locations.map((location, index) => {
        const name = String(location?.name || `Location ${index + 1}`).trim() || `Location ${index + 1}`;
        const image = location?.image && typeof location.image === "object" ? location.image : {};
        return {
          id: String(location?.id || `loc_import_${Date.now()}_${index}_${Math.floor(Math.random() * 10000)}`),
          name,
          description: String(location?.description || "").trim(),
          image: {
            path: String(image.path || ""),
            data: String(image.data || ""),
            name: String(image.name || ""),
          },
        };
      }).filter((location) => location.name) : [];
      if (!imported.length) throw new Error("No location images were returned from the import.");
      let added = 0;
      let updated = 0;
      for (const location of imported) {
        const existing = locationByName(location.name);
        if (existing) {
          existing.description = location.description || existing.description || "";
          existing.image = location.image || existing.image || { path: "", data: "", name: "" };
          updated += 1;
        } else {
          refs.locations.push(location);
          added += 1;
        }
      }
      refs.use_location_references = true;
      useLocations.input.checked = true;
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
      renderAll();
      renderFluxIngredientList(activeSegment());
      renderNBIngredientList(activeSegment());
      render();
      await autoSaveSessionQuiet("imported reference locations");
      const missingCount = Array.isArray(data.missing_descriptions) ? data.missing_descriptions.length : 0;
      const message = `Imported locations from subject_location/location. Added: ${added}. Updated: ${updated}.${missingCount ? `\n${missingCount} location${missingCount === 1 ? " is" : "s are"} missing matching .txt descriptions.` : ""}`;
      progress.set(message, 100);
      toast(message);
      progress.close(2400);
    } catch (error) {
      const message = String(error?.message || error || "Could not import locations.");
      progress.set(`Error:\n${message}`, 100);
      toast(message, true);
    }
  }

  function referenceSubjectContextForLocations() {
    return (refs.subjects || [])
    .map((subject) => {
      const type = String(subject.reference_type || "character").trim();
      const name = String(subject.name || "").trim();
      const description = String(subject.description || "").trim();
      if (!name && !description) return "";
      return `${name || "Reference"}${type ? ` (${type})` : ""}${description ? `: ${description}` : ""}`;
    })
    .filter(Boolean)
    .join("\n");
  }

  async function extractLocationsWithGemma() {
    let progress = null;
    extractLocations.disabled = true;
    autoMapLocations.disabled = true;
    extractLocations.textContent = "Extracting...";
    progress = createProgressWindow("Extracting locations", { zIndex: 100008 });
    progress.set("Checking lyrics, scene notes, concept prompts, and location context...", 5);
    const rawSegments = allEditableSegments();
    const scenes = rawSegments.map((segment, index) => {
      const planningNotes = [
        segment.notes || "",
        segment.timeline_note || "",
        segment.i2v_notes || "",
      ].filter(Boolean).join("\n");
      return {
        id: segment.id,
        label: segment.label || `Scene ${index + 1}`,
        concept: sceneConceptPromptText(segment),
        notes: planningNotes,
        lyric: segment.lyric_text || "",
      };
    });
    const planningScenes = scenes
      .map((scene) => ({
        id: scene.id,
        label: scene.label,
        concept: scene.concept,
        notes: scene.notes,
      }))
      .filter((scene) => String(scene.concept || scene.notes || "").trim());
    const lyricScenes = scenes.filter((scene) => String(scene.lyric || "").trim());
    if (!planningScenes.length && !lyricScenes.length) {
      const message = "Extract Locations needs lyrics, scene notes, concept prompts, or timeline notes first.";
      progress.set(`Error:\n${message}`, 100);
      toast(message, true);
      extractLocations.disabled = false;
      autoMapLocations.disabled = false;
      extractLocations.textContent = "Gemma Extract";
      return;
    }
    const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
    if (!modelFile && !["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner)) {
      const message = "Choose a non-vision Gemma model first, or use LM Studio, LLM API, or your own server in LLM Runner.";
      progress.set(`Error:\n${message}`, 100);
      toast(message, true);
      extractLocations.disabled = false;
      autoMapLocations.disabled = false;
      extractLocations.textContent = "Gemma Extract";
      return;
    }
    try {
      const useLyricsScout = Boolean((wizardLocationMode && lyricScenes.length) || (!planningScenes.length && lyricScenes.length));
      const styleTheme = await locationExtractionStyleTheme(locationStyleTheme.value);
      progress.set(`${useLyricsScout ? "Asking Gemma location scout to create locations from lyrics" : "Asking Gemma for a reusable location list"}...\n${gemmaRunnerLine()}`, 15);
      const data = useLyricsScout
        ? await postJson("/vrgdg/music_builder/wizard_locations_from_lyrics", {
          ...textGemmaRunnerPayload(),
          model_file: modelFile,
          lyrics_text: lyricScenes.map((scene, index) => `Scene ${index + 1}: ${scene.lyric}`).join("\n"),
          style_theme: styleTheme,
          subject_context: referenceSubjectContextForLocations(),
          existing_locations: refs.locations.map((item) => ({ name: item.name || "", description: item.description || "" })),
          max_locations: refs.max_generated_locations || 8,
          n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
          max_new_tokens: 2200,
          unload_after: true,
        }, 10 * 60 * 1000)
        : await postJson("/vrgdg/music_builder/flux_reference_extract_locations", {
          ...textGemmaRunnerPayload(),
          model_file: modelFile,
          scenes: planningScenes,
          subject_scene_text: subjectSceneInput.value || "",
          style_theme: styleTheme,
          subject_context: referenceSubjectContextForLocations(),
          existing_locations: refs.locations.map((item) => ({ name: item.name || "", description: item.description || "" })),
          max_locations: refs.max_generated_locations || 8,
          n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
          unload_after: true,
        }, 10 * 60 * 1000);
      progress.set("Adding extracted locations to the Reference Builder...", 78);
      let added = 0;
      let updated = 0;
      for (const item of data.locations || []) {
        const name = String(item.name || "").trim();
        if (!name) continue;
        let location = locationByName(name);
        if (!location) {
          location = createLocation(name, item.description || "");
          added += 1;
        } else if (!String(location.description || "").trim() && item.description) {
          location.description = String(item.description || "");
          updated += 1;
        }
      }
      renderAll();
      progress.set(`Extracted locations ready.\nAdded: ${added}\nUpdated: ${updated}\n\nReview/edit the locations, add or create images, then run Auto Map.`, 100);
      progress.close(2600);
      toast(`Extracted ${added} new location${added === 1 ? "" : "s"}. Review them before Auto Map.`);
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      extractLocations.disabled = false;
      autoMapLocations.disabled = false;
      extractLocations.textContent = "Gemma Extract";
    }
  }

  async function autoMapLocationsWithGemma() {
    let progress = null;
    autoMapLocations.disabled = true;
    autoMapLocations.textContent = "Mapping...";
    progress = createProgressWindow("Auto mapping locations", { zIndex: 100008 });
    progress.set("Checking lyrics, scene notes, concept prompts, location list, and Gemma settings...\nNo reference images are required for Auto Map.", 5);
    const scenes = allEditableSegments().map((segment, index) => {
      const planningNotes = [
        segment.notes || "",
        segment.timeline_note || "",
        segment.i2v_notes || "",
      ].filter(Boolean).join("\n");
      return {
        id: segment.id,
        label: segment.label || `Scene ${index + 1}`,
        concept: sceneConceptPromptText(segment),
        notes: planningNotes,
        lyric: segment.lyric_text || "",
      };
    });
    const planningScenes = scenes
      .map((scene) => ({
        id: scene.id,
        label: scene.label,
        concept: scene.concept,
        notes: scene.notes,
      }))
      .filter((scene) => String(scene.concept || scene.notes || "").trim());
    const lyricScenes = scenes
      .map((scene) => ({
        id: scene.id,
        label: scene.label,
        concept: "",
        notes: scene.lyric,
      }))
      .filter((scene) => String(scene.notes || "").trim());
    const usableScenes = scenes
      .map((scene) => ({
        id: scene.id,
        label: scene.label,
        concept: scene.concept,
        notes: String(scene.notes || "").trim() || String(scene.lyric || "").trim(),
      }))
      .filter((scene) => String(scene.concept || scene.notes || "").trim());
    if (!usableScenes.length) {
      const message = "Auto Map needs lyrics, scene notes, concept prompts, or timeline notes first. It does not need images yet, but it does need scene text to choose locations.";
      progress.set(`Error:\n${message}`, 100);
      toast(message, true);
      autoMapLocations.disabled = false;
      autoMapLocations.textContent = "Auto Map Locations with Gemma";
      return;
    }
    if (!refs.locations.length) {
      const message = "Auto Map needs at least one location first. Click Extract Locations or Add Location, then run Auto Map.";
      progress.set(`Error:\n${message}`, 100);
      toast(message, true);
      autoMapLocations.disabled = false;
      autoMapLocations.textContent = "Auto Map Locations with Gemma";
      return;
    }
    const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
    if (!modelFile && !["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner)) {
      const message = "Choose a non-vision Gemma model first, or use LM Studio, LLM API, or your own server in LLM Runner.";
      progress.set(`Error:\n${message}`, 100);
      toast(message, true);
      autoMapLocations.disabled = false;
      autoMapLocations.textContent = "Auto Map Locations with Gemma";
      return;
    }
    try {
      const compactText = (value, limit) => {
        const text = String(value || "").replace(/\s+/g, " ").trim();
        return text.length > limit ? `${text.slice(0, Math.max(0, limit - 3)).trim()}...` : text;
      };
      const compactScenes = usableScenes.map((scene) => ({
        ...scene,
        concept: compactText(scene.concept, 900),
        notes: compactText(scene.notes, 450),
      }));
      const compactLocations = refs.locations.map((item) => ({
        name: compactText(item.name, 80),
        description: compactText(item.description, 650),
      }));
      const locationMapBatchSize = 8;
      const batches = [];
      for (let index = 0; index < compactScenes.length; index += locationMapBatchSize) {
        batches.push(compactScenes.slice(index, index + locationMapBatchSize));
      }
      const mergedData = { locations: [], scene_map: {} };
      const usedLocationCounts = {};
      const previousAssignments = [];
      for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
        progress.set(`Sending scene text and lyric fallbacks to Gemma...\nBatch ${batchIndex + 1}/${batches.length}\n${gemmaRunnerLine()}`, 10 + Math.round((batchIndex / Math.max(1, batches.length)) * 55));
        const batchData = await postJson("/vrgdg/music_builder/flux_reference_location_map", {
          ...textGemmaRunnerPayload(),
          model_file: modelFile,
          scenes: batches[batchIndex],
          subject_scene_text: compactText(subjectSceneInput.value || "", 1600),
          existing_locations: compactLocations,
          used_location_counts: usedLocationCounts,
          previous_assignments: previousAssignments.slice(-24),
          unload_after: batchIndex === batches.length - 1,
          n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
          max_new_tokens: 1200,
        }, 10 * 60 * 1000);
        mergedData.locations = batchData.locations || mergedData.locations || [];
        Object.assign(mergedData.scene_map, batchData.scene_map || {});
        for (const scene of batches[batchIndex]) {
          const locationName = String(batchData.scene_map?.[scene.id] || "").trim();
          if (!locationName) continue;
          usedLocationCounts[locationName] = (Number(usedLocationCounts[locationName] || 0) || 0) + 1;
          previousAssignments.push({
            scene: scene.label || scene.id,
            location: locationName,
          });
        }
      }
      const data = mergedData;
      progress.set("Applying location map...", 80);
      const nameToId = new Map();
      for (const item of data.locations || []) {
        const name = String(item.name || "").trim();
        if (!name) continue;
        let location = locationByName(name);
        if (!location) location = createLocation(name, item.description || "");
        else if (!String(location.description || "").trim() && item.description) location.description = String(item.description || "");
        nameToId.set(locationKey(location.name), location.id);
      }
      for (const [sceneId, locationName] of Object.entries(data.scene_map || {})) {
        const id = nameToId.get(locationKey(locationName)) || locationByName(locationName)?.id || "";
        if (id) refs.scene_map[sceneId] = id;
      }
      refs.use_location_references = true;
      useLocations.input.checked = true;
      renderAll();
      progress.set("Location mapping ready. Review/edit the mappings before saving.", 100);
      progress.close(1600);
      toast(`Auto mapped ${(data.locations || []).length} location${(data.locations || []).length === 1 ? "" : "s"}. Review, add images, then save.`);
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      autoMapLocations.disabled = false;
      autoMapLocations.textContent = "Auto Map Locations with Gemma";
    }
  }

  function renderLocations() {
    locationsList.innerHTML = "";
    locationsList.style.maxHeight = "none";
    locationsList.style.overflow = "visible";
    if (!refs.locations.length) {
      const empty = document.createElement("div");
      empty.innerHTML = referenceImagesEnabled
        ? `<strong style="color:#cffafe;">No locations yet.</strong><br><span>Drop one or more location images here, upload images, or add locations/scenes.</span>`
        : `<strong style="color:#cffafe;">No locations yet.</strong><br><span>Add, import, or extract reusable location descriptions for scene mapping.</span>`;
      empty.style.cssText = "font-size:12px;color:#94a3b8;border:1px dashed #0891b2;border-radius:6px;padding:14px;text-align:center;background:#061620;";
      if (referenceImagesEnabled) wireDrop(empty, { kind: "location", bulk: true });
      locationsList.append(empty);
      return;
    }
    let draggedLocationIndex = -1;
    const moveLocationRow = (fromIndex, toIndex) => {
      if (fromIndex === toIndex || fromIndex < 0 || toIndex < 0 || fromIndex >= refs.locations.length || toIndex >= refs.locations.length) return;
      const [moved] = refs.locations.splice(fromIndex, 1);
      refs.locations.splice(toIndex, 0, moved);
      renderAll();
    };
    refs.locations.forEach((location, index) => {
      const row = document.createElement("div");
      row.draggable = false;
      row.dataset.locationIndex = String(index);
      row.style.cssText = "border:1px solid #334155;border-radius:7px;background:linear-gradient(135deg,#0f172a,#111827);padding:10px;display:flex;flex-direction:column;gap:8px;";
      row.addEventListener("dragstart", (event) => {
        draggedLocationIndex = index;
        row.style.opacity = "0.55";
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", String(index));
      });
      row.addEventListener("dragend", () => {
        draggedLocationIndex = -1;
        row.style.opacity = "";
      });
      row.addEventListener("dragover", (event) => {
        if (draggedLocationIndex < 0 || draggedLocationIndex === index) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        row.style.borderColor = "#22d3ee";
      });
      row.addEventListener("dragleave", () => {
        row.style.borderColor = "#334155";
      });
      row.addEventListener("drop", (event) => {
        event.preventDefault();
        row.style.borderColor = "#334155";
        const fromIndex = Number(event.dataTransfer.getData("text/plain") || draggedLocationIndex);
        moveLocationRow(fromIndex, index);
      });
      const name = makeInput(location.name || `Location ${index + 1}`);
      const description = document.createElement("textarea");
      description.value = location.description || "";
      description.placeholder = "Location description / generation prompt...";
      description.style.cssText = "min-height:76px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;";
      const usedBy = allEditableSegments()
        .map((segment, sceneIndex) => refs.scene_map?.[segment.id] === location.id ? `Scene ${sceneIndex + 1}` : "")
        .filter(Boolean)
        .join(", ") || "Not mapped yet";
      const used = document.createElement("div");
      used.textContent = `Used by: ${usedBy}`;
      used.style.cssText = "font-size:11px;color:#a5f3fc;";
      const drop = document.createElement("div");
      drop.style.cssText = `${subjectDrop.style.cssText};min-height:132px;height:132px;`;
      drop.style.display = "flex";
      const target = imageTargetFor(location, "image", "location");
      renderDrop(drop, location.image, "Drop location image here");
      wireDrop(drop, target);
      const buttons = document.createElement("div");
      buttons.style.cssText = "display:grid;grid-template-columns:1fr;gap:6px;align-self:stretch;";
      const createZImage = makeButton("Generate Location", "primary");
      createZImage.title = "Choose ZImage or Krea2 + ZImage enhancer for this location reference.";
      const describeImage = makeButton("Gemma Describe", "primary");
      const upload = makeButton("Upload", "primary");
      const detailedDescription = makeButton("Detailed Description", "primary");
      detailedDescription.title = "Expand the existing location label and short description into a detailed standalone environment description.";
      const clear = makeButton("Clear");
      const remove = makeButton("Remove");
      createZImage.onclick = () => createFluxReferenceWithZImage("location", location, `${location.name || ""}\n${location.description || ""}`, location.name || `location_${index + 1}`);
      describeImage.onclick = () => describeSingleReferenceWithGemma(location, "location", location.name || `Location ${index + 1}`);
      detailedDescription.onclick = () => createDetailedLocationDescriptionWithGemma(location, detailedDescription, renderAll);
      upload.onclick = () => uploadFor(target);
      clear.onclick = () => {
        location.image = { path: "", data: "", name: "" };
        renderAll();
      };
      remove.onclick = () => {
        refs.locations.splice(index, 1);
        for (const segment of allEditableSegments()) {
          if (refs.scene_map?.[segment.id] === location.id) refs.scene_map[segment.id] = "";
        }
        if (!refs.locations.length) {
          refs.scene_map = {};
          refs.scene_trigger_map = {};
          refs.use_location_references = false;
          refs.locations_cleared = true;
          useLocations.input.checked = false;
        }
        renderAll();
      };
      name.addEventListener("input", () => {
        location.name = name.value;
        renderMapping();
      });
      description.addEventListener("input", () => {
        location.description = description.value;
      });
      if (referenceImagesEnabled) buttons.append(createZImage, describeImage, detailedDescription, upload, clear);
      else buttons.append(detailedDescription);
      buttons.append(remove);
      const handle = document.createElement("button");
      handle.type = "button";
      handle.textContent = "::";
      handle.title = "Drag to reorder";
      handle.draggable = true;
      handle.style.cssText = "width:28px;height:42px;border:1px solid #334155;border-radius:6px;background:#0b1220;color:#67e8f9;font-weight:900;cursor:grab;";
      const number = document.createElement("div");
      number.textContent = String(index + 1);
      number.style.cssText = "font-size:18px;font-weight:900;color:#f8fafc;text-align:center;";
      const nameField = makeField("Location label", name);
      const descField = makeField("Description", description);
      const imageWrap = document.createElement("div");
      imageWrap.style.cssText = "min-width:230px;";
      imageWrap.append(drop);
      const mainRow = document.createElement("div");
      mainRow.style.cssText = referenceImagesEnabled
        ? "display:grid;grid-template-columns:28px 34px minmax(180px,0.9fr) minmax(340px,1.5fr) minmax(230px,1fr) 148px;gap:10px;align-items:center;min-width:970px;"
        : "display:grid;grid-template-columns:28px 34px minmax(200px,.8fr) minmax(420px,1.5fr) 108px;gap:10px;align-items:center;min-width:790px;";
      mainRow.append(handle, number, nameField, descField);
      if (referenceImagesEnabled) mainRow.append(imageWrap);
      mainRow.append(buttons);
      const metaRow = document.createElement("div");
      metaRow.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between;";
      metaRow.append(used);
      row.append(mainRow, metaRow);
      locationsList.append(row);
    });
  }

  return {
    autoMapLocationsWithGemma, createLocation, exportReferenceBuilderLocations, extractLocationsWithGemma,
    openImportLocationSourceDialog, openImportLocationsDialog, renderLocations,
  };
}
