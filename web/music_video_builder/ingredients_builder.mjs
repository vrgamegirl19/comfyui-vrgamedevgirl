import { makeEditorImageUrl, postJson } from "./comfy_api.mjs";
import { LOCATION_MAPPER_GPT_URL } from "./constants.mjs";
import { escapeHtml, makeButton, makeField, makeGptLinkButton, makeInput, makeSubTabs, toast } from "./controls.mjs";
import { sceneConceptPromptText } from "./image_prompts.mjs";
import { hasReferenceImage } from "./llm_runner.mjs";
import { imageFileFromDrop } from "./media_import.mjs";
import { normalizeGemmaContextLimit } from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder } from "./reference_data.mjs";

export function createIngredientsBuilder({
  activeSegment, allEditableSegments, applyIngredientsReferenceMappings, autoMapIngredientsSheets,
  autoSaveSessionQuiet, createDetailedLocationDescriptionWithGemma, createProgressWindow,
  describeReferenceImageWithGemma, gemmaRunnerLine, i2vTextGemmaModelSelect, locationExtractionStyleTheme,
  locationScoutCharacterPayloadForGpt, openLocationScoutGptForRefs, pushHistory, render, sceneDisplayName,
  state, syncIngredientsSceneMapFromSubjectMappings, syncInspector, syncPreview, t2iTextGemmaModelSelect,
  textGemmaRunnerPayload,
}) {
  function openIngredientsReferenceBuilderModal() {
    let refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const syncedIngredients = syncIngredientsSceneMapFromSubjectMappings(refs);
    refs = syncedIngredients.refs;
    state.fluxReferenceBuilder = refs;
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(1380px,calc(100vw - 42px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;";
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Ingredients Reference Builder</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Upload complete Ingredients ref sheets, map them to scenes, then Ingredients to Video will use the mapped sheet as that scene image.</div>`;
    const close = makeButton("Close");
    header.append(title, close);

    const content = document.createElement("div");
    content.style.cssText = "display:flex;flex-direction:column;gap:0;min-height:0;";

    const sheetPanel = document.createElement("div");
    sheetPanel.style.cssText = "border:1px solid #334155;border-radius:0 7px 7px 7px;background:#0b1220;padding:10px;display:flex;flex-direction:column;gap:10px;";
    const sheetHeader = document.createElement("div");
    sheetHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;";
    sheetHeader.innerHTML = `<div style="font-weight:900;color:#e0f2fe;">Ingredients Sheets</div>`;
    const describeSheets = makeButton("Gemma - Describe Missing", "primary");
    const addSheet = makeButton("Add Sheet", "primary");
    describeSheets.title = "Use vision Gemma to write character appearance descriptions for sheet images that do not already have descriptions.";
    const sheetHeaderActions = document.createElement("div");
    sheetHeaderActions.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;";
    sheetHeaderActions.append(describeSheets, addSheet);
    sheetHeader.append(sheetHeaderActions);
    const sheetList = document.createElement("div");
    sheetList.style.cssText = "display:flex;flex-direction:column;gap:10px;max-height:min(62vh,720px);overflow:auto;padding-right:4px;";
    sheetPanel.append(sheetHeader, sheetList);

    const mappingPanel = document.createElement("div");
    mappingPanel.style.cssText = "border:1px solid #334155;border-radius:0 7px 7px 7px;background:#0b1220;padding:10px;display:flex;flex-direction:column;gap:10px;";
    const mapHeader = document.createElement("div");
    mapHeader.innerHTML = `<div style="font-weight:900;color:#e0f2fe;">Scene Mapping</div><div style="font-size:12px;color:#94a3b8;margin-top:2px;">Choose the Ingredients sheet image and optional location text for each scene.</div>`;
    const autoBox = document.createElement("div");
    autoBox.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;border:1px solid #1f2937;border-radius:7px;background:#020617;padding:9px;";
    const sourceLabels = {
      director_notes: "Director Notes",
      concept_prompt: "Concept Prompt",
      scene_notes: "Scene Notes",
      lyric_text: "Lyric Text",
    };
    const sourceInputs = {};
    for (const [key, label] of Object.entries(sourceLabels)) {
      const wrap = document.createElement("label");
      wrap.style.cssText = "display:flex;align-items:center;gap:7px;font-size:12px;color:#dbeafe;";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = refs.ingredients_auto_map_sources?.[key] !== false;
      sourceInputs[key] = input;
      wrap.append(input, document.createTextNode(label));
      autoBox.append(wrap);
    }
    const mapActions = document.createElement("div");
    mapActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const autoMap = makeButton("Auto Map By Sheet Name", "primary");
    const clearMappings = makeButton("Clear Mappings");
    mapActions.append(autoMap, clearMappings);
    const sceneList = document.createElement("div");
    sceneList.style.cssText = "display:flex;flex-direction:column;gap:10px;max-height:min(62vh,720px);overflow:auto;padding-right:4px;";
    mappingPanel.append(mapHeader, autoBox, mapActions, sceneList);

    const locationPanel = document.createElement("div");
    locationPanel.style.cssText = "border:1px solid #334155;border-radius:0 7px 7px 7px;background:#0b1220;padding:10px;display:flex;flex-direction:column;gap:8px;";
    const locationHeader = document.createElement("div");
    locationHeader.innerHTML = `<div style="font-weight:900;color:#e0f2fe;">Location Text</div><div style="font-size:12px;color:#94a3b8;margin-top:2px;">Optional text-only locations for Gemma/prompt planning. No location images or triggers are used here.</div>`;
    const locationToolActions = document.createElement("div");
    locationToolActions.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;";
    const extractLocations = makeButton("Gemma Extract", "primary");
    const gptLocationScout = makeButton("GPT Scout", "primary");
    const importLocations = makeButton("Import JSON", "primary");
    const addLocation = makeButton("Add Location", "primary");
    const autoMapLocations = makeButton("Auto Map", "primary");
    const clearLocations = makeButton("Clear Locations");
    extractLocations.title = "Ask Gemma to extract a reusable text-only location list from the source text and scene text.";
    gptLocationScout.title = "Copy lyrics/dialogue, style/theme, and character descriptions as JSON, then open the Music Video Location Scout GPT.";
    locationToolActions.append(extractLocations, gptLocationScout, importLocations, addLocation, autoMapLocations, clearLocations);
    const locationStyleTheme = makeInput(refs.location_style_theme || "");
    locationStyleTheme.placeholder = "Optional style/theme for extracted locations...";
    locationStyleTheme.title = "Optional. Helps Gemma choose locations that match your character/style/theme.";
    const maxGeneratedLocations = makeInput(String(refs.max_generated_locations || 8), "number");
    maxGeneratedLocations.min = "1";
    maxGeneratedLocations.max = "50";
    maxGeneratedLocations.step = "1";
    maxGeneratedLocations.title = "Maximum number of locations Gemma may add during one automatic extraction.";
    maxGeneratedLocations.addEventListener("change", () => {
      refs.max_generated_locations = Math.max(1, Math.min(50, Number(maxGeneratedLocations.value || 8) || 8));
      maxGeneratedLocations.value = String(refs.max_generated_locations);
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
      autoSaveSessionQuiet("maximum generated locations").catch(() => null);
    });
    locationStyleTheme.addEventListener("input", () => {
      refs.location_style_theme = locationStyleTheme.value;
    });
    const locationHint = document.createElement("div");
    locationHint.textContent = "Import location lists or scene-to-location JSON, then choose those locations in the scene dropdowns above.";
    locationHint.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    const locationList = document.createElement("div");
    locationList.style.cssText = "display:flex;flex-direction:column;gap:8px;max-height:260px;overflow:auto;padding-right:4px;";
    locationPanel.append(locationHeader, locationToolActions, makeField("Optional style/theme for location extraction", locationStyleTheme), locationHint, locationList);

    const ingredientsTabs = makeSubTabs([
      { value: "sheets", label: "Sheets", content: sheetPanel },
      { value: "mapping", label: "Mapping", content: mappingPanel },
      { value: "locations", label: "Locations", content: locationPanel },
    ]);
    content.append(ingredientsTabs.wrapper);

    const footer = document.createElement("div");
    footer.style.cssText = "display:flex;justify-content:flex-end;gap:8px;";
    const cancel = makeButton("Cancel");
    const save = makeButton("Save And Apply", "primary");
    footer.append(cancel, save);
    box.append(header, content, footer);
    backdrop.append(box);
    document.body.append(backdrop);

    const normalizeLocalRefs = () => {
      refs.ingredients_auto_map_sources = Object.fromEntries(Object.entries(sourceInputs).map(([key, input]) => [key, input.checked]));
      refs = normalizeFluxReferenceBuilder(refs);
      return refs;
    };

    const syncMappingInputs = () => {
      for (const select of sceneList.querySelectorAll("[data-ingredients-scene-map='1']")) {
        const sceneId = select.dataset.sceneId || "";
        select.value = refs.ingredients_scene_map?.[sceneId] || "";
      }
      for (const select of sceneList.querySelectorAll("[data-ingredients-location-map='1']")) {
        const sceneId = select.dataset.sceneId || "";
        select.value = refs.scene_map?.[sceneId] || "";
      }
    };

    const ingredientsSheetById = (sheetId) => {
      const id = String(sheetId || "");
      return (refs.ingredients_sheets || []).find((item) => String(item.id || "") === id) || null;
    };

    const setSheetImageFromFile = (sheet, file) => {
      if (!sheet || !file) return;
      const sheetId = String(sheet.id || "");
      const fileName = file.name || `${sheet.name || "ingredients"}_sheet.png`;
      const previewUrl = URL.createObjectURL(file);
      const previewImage = {
        path: "",
        data: "",
        name: fileName,
        preview_url: previewUrl,
      };
      const currentSheet = ingredientsSheetById(sheetId);
      if (currentSheet) currentSheet.image = previewImage;
      else sheet.image = previewImage;
      renderSheets();
      syncMappingInputs();
      const reader = new FileReader();
      reader.onload = () => {
        const loadedImage = {
          path: "",
          data: String(reader.result || ""),
          name: fileName,
          preview_url: previewUrl,
        };
        const latestSheet = ingredientsSheetById(sheetId);
        if (latestSheet) latestSheet.image = loadedImage;
        else sheet.image = loadedImage;
        renderSheets();
        syncMappingInputs();
        toast(`Loaded Ingredients sheet image:\n${fileName || "uploaded image"}`);
      };
      reader.onerror = () => toast("Failed to read Ingredients sheet image.", true);
      reader.readAsDataURL(file);
    };

    const setSheetImageFromText = (sheet, text) => {
      const value = String(text || "").trim();
      if (!sheet || !value) return;
      const sheetId = String(sheet.id || "");
      const currentSheet = ingredientsSheetById(sheetId) || sheet;
      if (/^data:image\//i.test(value)) {
        currentSheet.image = { path: "", data: value, name: currentSheet.image?.name || `${currentSheet.name || "ingredients"}_sheet.png`, preview_url: "" };
      } else {
        currentSheet.image = { path: value, data: "", name: currentSheet.image?.name || value.split(/[\\/]/).pop() || `${currentSheet.name || "ingredients"}_sheet.png`, preview_url: "" };
      }
      renderSheets();
      syncMappingInputs();
    };

    const describeIngredientSheetWithGemma = async (sheet) => {
      const currentSheet = ingredientsSheetById(sheet?.id) || sheet;
      if (!currentSheet) return;
      const progress = createProgressWindow("Describing Ingredients sheet", { zIndex: 100008 });
      try {
        progress.set(`Vision Gemma describing sheet image...\n${currentSheet.name || "Ingredients sheet"}\n${gemmaRunnerLine({ vision: true })}`, 18);
        await describeReferenceImageWithGemma(currentSheet, "subject", { unloadAfter: true });
        renderSheets();
        await autoSaveSessionQuiet("Gemma described ingredients sheet");
        progress.set("Sheet description updated.", 100);
        progress.close(1200);
        toast("Ingredients sheet description updated.");
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
      }
    };

    const describeMissingIngredientSheetsWithGemma = async () => {
      const jobs = (refs.ingredients_sheets || [])
        .filter((sheet) => !String(sheet.description || "").trim() && hasReferenceImage(sheet.image || {}));
      if (!jobs.length) {
        toast("No Ingredients sheet images are missing descriptions.");
        return;
      }
      const progress = createProgressWindow("Describing Ingredients sheets", { zIndex: 100008 });
      try {
        for (let index = 0; index < jobs.length; index += 1) {
          const sheet = jobs[index];
          progress.set(`Vision Gemma describing sheet image...\n${index + 1}/${jobs.length}: ${sheet.name || `Ingredients Sheet ${index + 1}`}\n${gemmaRunnerLine({ vision: true })}`, 8 + Math.round((index / Math.max(1, jobs.length)) * 84));
          await describeReferenceImageWithGemma(sheet, "subject", { unloadAfter: index === jobs.length - 1 });
          renderSheets();
          await autoSaveSessionQuiet(`Gemma described ingredients sheet ${index + 1}`);
        }
        progress.set(`Gemma sheet descriptions complete.\nUpdated ${jobs.length} sheet${jobs.length === 1 ? "" : "s"}.`, 100);
        progress.close(1800);
        toast(`Gemma described ${jobs.length} Ingredients sheet${jobs.length === 1 ? "" : "s"}.`);
      } catch (error) {
        progress.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
      }
    };

    const collectMappingsFromDom = () => {
      refs.ingredients_scene_map = {};
      for (const select of sceneList.querySelectorAll("[data-ingredients-scene-map='1']")) {
        const sceneId = select.dataset.sceneId || "";
        const sheetId = String(select.value || "").trim();
        if (sceneId && sheetId) refs.ingredients_scene_map[sceneId] = sheetId;
      }
      const validLocationIds = new Set((refs.locations || []).map((location) => String(location.id || "").trim()).filter(Boolean));
      if (!validLocationIds.size) {
        refs.scene_map = {};
        refs.use_location_references = false;
        refs.locations_cleared = true;
      } else {
        refs.locations_cleared = false;
        if (!refs.scene_map || typeof refs.scene_map !== "object") refs.scene_map = {};
        for (const select of sceneList.querySelectorAll("[data-ingredients-location-map='1']")) {
          const sceneId = select.dataset.sceneId || "";
          const locationId = String(select.value || "").trim();
          if (sceneId && locationId && validLocationIds.has(locationId)) refs.scene_map[sceneId] = locationId;
          else if (sceneId) delete refs.scene_map[sceneId];
        }
      }
      refs.use_location_references = Boolean(validLocationIds.size && ((refs.locations || []).length || Object.keys(refs.scene_map || {}).length));
      normalizeLocalRefs();
    };

    const syncNamedIngredientsSheetsToSubjects = () => {
      const namedSheets = (refs.ingredients_sheets || [])
        .filter((sheet) => {
          const name = String(sheet.name || "").trim();
          return name && !/^ingredients\s+sheet\s+\d*$/i.test(name);
        });
      if (!namedSheets.length) return 0;
      const imageIsEmpty = (image = {}) => !String(image.path || image.data || image.name || image.preview_url || "").trim();
      refs.subjects = Array.isArray(refs.subjects) ? refs.subjects : [];
      refs.subjects = refs.subjects.filter((subject, index) => {
        const name = String(subject?.name || "").trim();
        const description = String(subject?.description || "").trim();
        const image = subject?.image || {};
        return !(index === 0 && /^Character\s+1$/i.test(name) && !description && imageIsEmpty(image));
      });
      const subjectKey = (value) => String(value || "").trim().toLowerCase().replace(/\s+/g, " ");
      const byName = new Map(refs.subjects.map((subject) => [subjectKey(subject.name), subject]));
      let synced = 0;
      for (const sheet of namedSheets) {
        const name = String(sheet.name || "").trim();
        const key = subjectKey(name);
        if (!key) continue;
        const existing = byName.get(key);
        if (existing) {
          if (!existing.image || imageIsEmpty(existing.image)) existing.image = { ...(sheet.image || { path: "", data: "", name: "" }) };
          const sheetDescription = String(sheet.description || "").trim();
          if (sheetDescription) existing.description = sheetDescription;
          synced += 1;
          continue;
        }
        const subject = {
          id: `subj_${Date.now()}_${refs.subjects.length}_${Math.floor(Math.random() * 10000)}`,
          name,
          description: String(sheet.description || ""),
          trigger_phrase: "",
          trigger_position: "start",
          image: { ...(sheet.image || { path: "", data: "", name: "" }) },
        };
        refs.subjects.push(subject);
        byName.set(key, subject);
        synced += 1;
      }
          refs.subject_count = refs.subjects.length;
      refs.use_subject_reference = refs.subjects.length > 0;
      return synced;
    };

    const locationKey = (value) => String(value || "").trim().toLowerCase().replace(/\s+/g, " ");
    const locationByName = (name) => (refs.locations || []).find((location) => locationKey(location.name) === locationKey(name)) || null;
    const locationById = (id) => (refs.locations || []).find((location) => String(location.id || "") === String(id || "")) || null;
    const ingredientSubjectContextForLocations = () => locationScoutCharacterPayloadForGpt(refs)
      .map((item) => {
        const type = String(item.type || "character").trim();
        const name = String(item.name || "").trim();
        const description = String(item.description || "").trim();
        if (!name && !description) return "";
        return `${name || "Reference"}${type ? ` (${type})` : ""}${description ? `: ${description}` : ""}`;
      })
      .filter(Boolean)
      .join("\n");
    const upsertLocationText = (items = []) => {
      refs.locations = Array.isArray(refs.locations) ? refs.locations : [];
      let added = 0;
      let updated = 0;
      for (const item of items) {
        const name = String(item?.name || item?.location || item?.setting || "").trim();
        if (!name) continue;
        const description = String(item?.description || item?.location_description || item?.prompt || item?.details || "").trim();
        const existing = locationByName(name);
        if (existing) {
          if (description) existing.description = description;
          updated += 1;
        } else {
          refs.locations.push({
            id: `loc_${Date.now()}_${refs.locations.length}_${Math.floor(Math.random() * 10000)}`,
            name,
            description,
            image: { path: "", data: "", name: "" },
          });
          added += 1;
        }
      }
      refs.use_location_references = Boolean(refs.locations.length || Object.keys(refs.scene_map || {}).length);
      return { added, updated };
    };
    const parseIngredientLocationImport = (rawText) => {
      const text = String(rawText || "").trim();
      if (!text) return { locations: [], sceneMap: [] };
      const normalizeLocation = (item) => {
        if (!item || typeof item !== "object") return null;
        const name = String(item.location || item.location_name || item.name || item.setting || "").trim();
        const description = String(item.description || item.location_description || item.prompt || item.details || "").trim();
        return name ? { name, description } : null;
      };
      const normalizeSceneMap = (item, fallbackIndex = 0) => {
        if (!item || typeof item !== "object") return null;
        const sceneRaw = item.scene_number ?? item.sceneNumber ?? item.scene ?? item.segment ?? item.number ?? fallbackIndex + 1;
        const sceneNumber = Number(String(sceneRaw).match(/\d+/)?.[0] || sceneRaw || fallbackIndex + 1);
        const location = normalizeLocation(item);
        return location ? { sceneNumber, ...location } : null;
      };
      try {
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed)) {
          return {
            locations: parsed.map(normalizeLocation).filter(Boolean),
            sceneMap: parsed.map(normalizeSceneMap).filter(Boolean),
          };
        }
        if (parsed && typeof parsed === "object") {
          const source = parsed.locations || parsed.location_list || parsed.scenes || parsed.scene_map || parsed;
          if (Array.isArray(source)) {
            return {
              locations: source.map(normalizeLocation).filter(Boolean),
              sceneMap: source.map(normalizeSceneMap).filter(Boolean),
            };
          }
          return {
            locations: [],
            sceneMap: Object.entries(source)
              .filter(([key]) => !["trigger_position", "triggerPosition", "trigger_placement", "subject_trigger_position", "subjectTriggerPosition", "location_trigger_position", "locationTriggerPosition"].includes(key))
              .map(([key, value], index) => {
                const objectValue = typeof value === "string" ? { location: value } : value;
                return normalizeSceneMap({ scene: key, ...(objectValue || {}) }, index);
              })
              .filter(Boolean),
          };
        }
      } catch (_error) {
        // Fall through to simple line formats.
      }
      return {
        locations: text.split(/\r?\n/)
          .map((line) => line.trim())
          .filter(Boolean)
          .map((line) => {
            const clean = line.replace(/^[-*]\s*/, "");
            const match = clean.match(/^(.+?)\s*(?:=>|=|:|\s+-\s+)\s*(.+)$/);
            return match ? { name: match[1].trim(), description: match[2].trim() } : { name: clean, description: "" };
          })
          .filter((item) => item.name),
        sceneMap: [],
      };
    };
    const applyIngredientLocationSceneMap = (sceneMap = []) => {
      const segments = allEditableSegments();
      refs.scene_map = refs.scene_map && typeof refs.scene_map === "object" ? refs.scene_map : {};
      let mapped = 0;
      for (const item of sceneMap) {
        const segment = segments[Math.max(0, Number(item.sceneNumber || 1) - 1)];
        if (!segment) continue;
        const location = locationByName(item.name);
        if (!location) continue;
        refs.scene_map[segment.id] = location.id;
        mapped += 1;
      }
      refs.use_location_references = Boolean(refs.locations.length || Object.keys(refs.scene_map || {}).length);
      return mapped;
    };
    const repairIngredientLocationMappings = () => {
      refs.locations = Array.isArray(refs.locations) ? refs.locations : [];
      refs.scene_map = refs.scene_map && typeof refs.scene_map === "object" ? refs.scene_map : {};
      const segments = allEditableSegments();
      let repaired = 0;
      for (const [index, segment] of segments.entries()) {
        const keys = [segment.id, String(index + 1), `scene${index + 1}`, `Scene ${index + 1}`].filter(Boolean);
        const existingKey = keys.find((key) => String(refs.scene_map?.[key] || "").trim());
        if (!existingKey) continue;
        const locationId = String(refs.scene_map[existingKey] || "").trim();
        if (!locationId || locationById(locationId)) {
          if (existingKey !== segment.id && locationId) {
            refs.scene_map[segment.id] = locationId;
            delete refs.scene_map[existingKey];
          }
          continue;
        }
        const locationRef = segment.location_ref && typeof segment.location_ref === "object" ? segment.location_ref : null;
        const locationName = String(locationRef?.name || segment.mapped_location || segment.location || segment.setting || "").trim();
        const locationDescription = String(locationRef?.description || segment.location_description || "").trim();
        if (locationName || locationDescription) {
          let location = locationName ? locationByName(locationName) : null;
          if (!location) {
            location = {
              id: locationId,
              name: locationName || `Location ${index + 1}`,
              description: locationDescription,
              image: { path: "", data: "", name: "" },
            };
            refs.locations.push(location);
          } else {
            refs.scene_map[segment.id] = location.id;
          }
          if (locationDescription && !String(location.description || "").trim()) location.description = locationDescription;
          if (existingKey !== segment.id) delete refs.scene_map[existingKey];
          repaired += 1;
        } else {
          delete refs.scene_map[existingKey];
        }
      }
      refs.use_location_references = Boolean(refs.locations.length || Object.keys(refs.scene_map || {}).length);
      return repaired;
    };

    function renderLocations() {
      refs = normalizeFluxReferenceBuilder(refs);
      locationList.innerHTML = "";
      if (!refs.locations.length) {
        const empty = document.createElement("div");
        empty.style.cssText = "border:1px dashed #334155;border-radius:7px;padding:12px;text-align:center;color:#94a3b8;font-size:12px;";
        empty.textContent = "No location text yet. Import JSON or add a location.";
        locationList.append(empty);
        return;
      }
      refs.locations.forEach((location, index) => {
        const row = document.createElement("div");
        row.style.cssText = "border:1px solid #334155;border-radius:7px;background:linear-gradient(135deg,#0f172a,#111827);padding:10px;display:grid;grid-template-columns:34px minmax(180px,.85fr) minmax(320px,1.4fr) 150px 110px;gap:10px;align-items:center;min-width:920px;";
        const number = document.createElement("div");
        number.textContent = String(index + 1);
        number.style.cssText = "font-size:18px;font-weight:900;color:#f8fafc;text-align:center;";
        const name = document.createElement("input");
        name.value = location.name || `Location ${index + 1}`;
        name.placeholder = "Location name";
        name.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#111827;color:#f8fafc;padding:7px 8px;font-size:12px;";
        name.oninput = () => {
          const currentLocation = locationById(location.id) || location;
          currentLocation.name = name.value;
          renderScenes();
        };
        const description = document.createElement("textarea");
        description.value = location.description || "";
        description.placeholder = "Location description text...";
        description.style.cssText = "width:100%;box-sizing:border-box;min-height:58px;resize:vertical;border:1px solid #334155;border-radius:6px;background:#111827;color:#f8fafc;padding:7px 8px;font-size:12px;line-height:1.4;";
        description.oninput = () => {
          const currentLocation = locationById(location.id) || location;
          currentLocation.description = description.value;
        };
        const detailedDescription = makeButton("Detailed Description", "primary");
        detailedDescription.title = "Expand the existing location label and short description into a detailed standalone environment description.";
        detailedDescription.onclick = () => createDetailedLocationDescriptionWithGemma(locationById(location.id) || location, detailedDescription, renderAll);
        const remove = makeButton("Remove");
        remove.onclick = () => {
          refs.locations = refs.locations.filter((item) => item.id !== location.id);
          for (const [sceneId, locationId] of Object.entries(refs.scene_map || {})) {
            if (locationId === location.id) delete refs.scene_map[sceneId];
          }
          renderAll();
        };
        row.append(number, makeField("Location label", name), makeField("Description", description), detailedDescription, remove);
        locationList.append(row);
      });
    }

    function renderSheets() {
      refs = normalizeFluxReferenceBuilder(refs);
      sheetList.innerHTML = "";
      if (!refs.ingredients_sheets.length) {
        const empty = document.createElement("div");
        empty.style.cssText = "border:1px dashed #334155;border-radius:7px;padding:14px;text-align:center;color:#94a3b8;font-size:12px;";
        empty.textContent = "No Ingredients sheets yet.";
        sheetList.append(empty);
        return;
      }
      refs.ingredients_sheets.forEach((sheet, index) => {
        const row = document.createElement("div");
        row.style.cssText = "border:1px solid #334155;border-radius:7px;background:linear-gradient(135deg,#0f172a,#111827);padding:10px;display:flex;flex-direction:column;gap:8px;";
        row.dataset.sheetId = sheet.id;
        const image = sheet.image || {};
        const persistedSrc = image.data || (image.path ? makeEditorImageUrl(image.path) : "");
        const previewUrl = String(image.preview_url || "");
        const transientPreviewSrc = /^blob:/i.test(previewUrl) && !persistedSrc ? "" : previewUrl;
        const previewSrc = persistedSrc || transientPreviewSrc;
        const imageLoaded = Boolean(previewSrc);
        const top = document.createElement("div");
        top.style.cssText = "display:grid;grid-template-columns:34px 76px minmax(180px,.8fr) minmax(300px,1.2fr) minmax(260px,1fr) 150px;gap:10px;align-items:center;min-width:960px;";
        const number = document.createElement("div");
        number.textContent = String(index + 1);
        number.style.cssText = "font-size:18px;font-weight:900;color:#f8fafc;text-align:center;";
        const thumb = document.createElement("div");
        thumb.style.cssText = `width:76px;height:58px;border:1px solid ${imageLoaded ? "#155e75" : "#334155"};border-radius:6px;background:#111827;display:flex;align-items:center;justify-content:center;overflow:hidden;color:#64748b;font-size:10px;`;
        if (imageLoaded) {
          const thumbImg = document.createElement("img");
          thumbImg.src = previewSrc;
          thumbImg.alt = sheet.name || "Ingredients sheet";
          thumbImg.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
          thumbImg.onerror = () => {
            thumb.textContent = "Loaded";
            thumb.style.color = "#67e8f9";
          };
          thumb.append(thumbImg);
        } else {
          thumb.textContent = "No image";
        }
        const name = document.createElement("input");
        name.value = sheet.name || `Ingredients Sheet ${index + 1}`;
        name.placeholder = "Sheet name, used for auto-map";
        name.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#111827;color:#f8fafc;padding:7px 8px;font-size:12px;";
        name.oninput = () => {
          const currentSheet = ingredientsSheetById(sheet.id) || sheet;
          currentSheet.name = name.value;
          renderScenes();
        };
        const notes = document.createElement("textarea");
        notes.value = sheet.description || "";
        notes.placeholder = "Optional character/subject description sent to Storyboard and Gemma...";
        notes.style.cssText = "width:100%;box-sizing:border-box;min-height:78px;resize:vertical;border:1px solid #334155;border-radius:6px;background:#111827;color:#f8fafc;padding:7px 8px;font-size:12px;";
        notes.oninput = () => {
          const currentSheet = ingredientsSheetById(sheet.id) || sheet;
          currentSheet.description = notes.value;
        };
        const preview = document.createElement("div");
        preview.style.cssText = "height:96px;border:1px dashed #334155;border-radius:7px;background:#111827;display:flex;align-items:center;justify-content:center;overflow:hidden;color:#94a3b8;font-size:12px;text-align:center;";
        if (previewSrc) {
          const img = document.createElement("img");
          img.src = previewSrc;
          img.alt = sheet.name || "Ingredients sheet";
          img.style.cssText = "width:100%;height:100%;object-fit:contain;display:block;";
          img.onerror = () => {
            preview.innerHTML = "";
            const error = document.createElement("div");
            error.style.cssText = "padding:12px;color:#fca5a5;line-height:1.35;overflow-wrap:anywhere;";
            error.textContent = `Image is loaded, but the preview could not render: ${image.name || image.path || "uploaded sheet"}`;
            preview.append(error);
          };
          preview.append(img);
        } else {
          preview.textContent = "Drop image here, upload, or paste a path below.";
        }
        const status = document.createElement("div");
        status.style.cssText = `font-size:11px;line-height:1.35;overflow-wrap:anywhere;color:${imageLoaded ? "#67e8f9" : "#94a3b8"};`;
        status.textContent = imageLoaded
          ? `Loaded: ${image.name || image.path || "uploaded image"}`
          : "No sheet image loaded yet.";
        preview.ondragover = (event) => {
          event.preventDefault();
          preview.style.borderColor = "#22d3ee";
        };
        preview.ondragleave = () => { preview.style.borderColor = "#334155"; };
        preview.ondrop = (event) => {
          event.preventDefault();
          preview.style.borderColor = "#334155";
          const file = imageFileFromDrop(event);
          if (file) {
            setSheetImageFromFile(sheet, file);
            return;
          }
          const text = event.dataTransfer?.getData("text/plain") || event.dataTransfer?.getData("text/uri-list") || "";
          setSheetImageFromText(sheet, text);
        };
        const controls = document.createElement("div");
        controls.style.cssText = "display:grid;grid-template-columns:1fr;gap:6px;align-self:stretch;";
        const upload = makeButton("Upload");
        const describe = makeButton("Gemma Describe", "primary");
        describe.title = "Use vision Gemma to write the character appearance description from this sheet image.";
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "image/*";
        fileInput.style.display = "none";
        upload.onclick = () => fileInput.click();
        describe.onclick = () => describeIngredientSheetWithGemma(sheet);
        fileInput.onchange = () => {
          const file = fileInput.files?.[0];
          if (file) setSheetImageFromFile(sheet, file);
          fileInput.value = "";
        };
        const pathInput = document.createElement("input");
        pathInput.value = image.path || "";
        pathInput.placeholder = "Optional existing image path";
        pathInput.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#111827;color:#f8fafc;padding:7px 8px;font-size:12px;";
        pathInput.onchange = () => setSheetImageFromText(sheet, pathInput.value);
        const clear = makeButton("Clear Image");
        clear.onclick = () => {
          const currentSheet = ingredientsSheetById(sheet.id) || sheet;
          currentSheet.image = { path: "", data: "", name: "" };
          renderSheets();
        };
        const remove = makeButton("Remove");
        remove.onclick = () => {
          refs.ingredients_sheets = refs.ingredients_sheets.filter((item) => item.id !== sheet.id);
          for (const [sceneId, sheetId] of Object.entries(refs.ingredients_scene_map || {})) {
            if (sheetId === sheet.id) delete refs.ingredients_scene_map[sceneId];
          }
          renderAll();
        };
        controls.append(upload, describe, clear, remove, fileInput);
        const nameField = makeField("Sheet label", name);
        const notesField = makeField("Character / subject description", notes);
        const imageWrap = document.createElement("div");
        imageWrap.style.cssText = "display:flex;flex-direction:column;gap:5px;min-width:0;";
        imageWrap.append(preview, status);
        top.append(number, thumb, nameField, notesField, imageWrap, controls);
        row.append(top, makeField("Optional existing image path", pathInput));
        sheetList.append(row);
      });
    }

    function renderScenes() {
      refs = normalizeFluxReferenceBuilder(refs);
      sceneList.innerHTML = "";
      const sheets = refs.ingredients_sheets || [];
      const locations = refs.locations || [];
      const segments = allEditableSegments();
      const sceneMappedValue = (map, segment, index) => {
        const keys = [segment.id, String(index + 1), `scene${index + 1}`, `Scene ${index + 1}`].filter(Boolean);
        for (const key of keys) {
          const value = String(map?.[key] || "").trim();
          if (value) return value;
        }
        return "";
      };
      const makeSheetPreview = (sheetId) => {
        const sheet = sheets.find((item) => String(item.id || "") === String(sheetId || ""));
        const wrap = document.createElement("div");
        wrap.style.cssText = "display:flex;gap:7px;align-items:center;min-height:58px;overflow:hidden;padding:5px;border:1px solid #1e3a5f;border-radius:7px;background:#071422;";
        if (!sheet) {
          const empty = document.createElement("div");
          empty.textContent = "No sheet";
          empty.style.cssText = "font-size:11px;color:#94a3b8;padding:0 6px;";
          wrap.append(empty);
          return wrap;
        }
        const image = sheet.image || {};
        const src = image.data || (image.path ? makeEditorImageUrl(image.path) : "") || String(image.preview_url || "");
        const thumb = document.createElement("div");
        thumb.style.cssText = "width:54px;height:54px;border:1px solid #155e75;border-radius:6px;background:#061620;overflow:hidden;display:flex;align-items:center;justify-content:center;flex:0 0 auto;";
        if (src) {
          const img = document.createElement("img");
          img.src = src;
          img.alt = sheet.name || "Ingredients sheet";
          img.draggable = false;
          img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
          thumb.append(img);
        } else {
          const noImg = document.createElement("span");
          noImg.textContent = "No img";
          noImg.style.cssText = "font-size:10px;font-weight:900;color:#67e8f9;text-align:center;padding:4px;";
          thumb.append(noImg);
        }
        const label = document.createElement("div");
        label.textContent = sheet.name || "Ingredients Sheet";
        label.style.cssText = "font-size:11px;color:#cbd5e1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;";
        wrap.append(thumb, label);
        return wrap;
      };
      if (!segments.length) {
        const empty = document.createElement("div");
        empty.style.cssText = "border:1px dashed #334155;border-radius:7px;padding:14px;text-align:center;color:#94a3b8;font-size:12px;";
        empty.textContent = "No scenes are loaded yet.";
        sceneList.append(empty);
        return;
      }
      segments.forEach((segment, index) => {
        const row = document.createElement("div");
        row.style.cssText = "display:grid;grid-template-columns:minmax(170px,.7fr) minmax(180px,.85fr) minmax(200px,1fr) minmax(180px,.85fr);gap:10px;align-items:stretch;border:1px solid #334155;border-radius:7px;background:linear-gradient(135deg,#0f172a,#111827);padding:10px;";
        const label = document.createElement("div");
        const lyric = String(segment.lyric_text || "").trim();
        label.innerHTML = `<div style="font-size:11px;color:#67e8f9;font-weight:900;text-transform:uppercase;">Scene ${index + 1}</div><div style="font-size:13px;font-weight:900;color:#f8fafc;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(sceneDisplayName(segment, index))}</div><div style="font-size:11px;color:#94a3b8;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(lyric || sceneConceptPromptText(segment) || segment.notes || "No notes yet")}</div>`;
        label.style.cssText = "min-width:0;border:1px solid #1e3a5f;border-radius:7px;background:#071422;padding:10px;display:flex;flex-direction:column;justify-content:center;";
        const select = document.createElement("select");
        select.dataset.ingredientsSceneMap = "1";
        select.dataset.sceneId = segment.id || "";
        select.style.cssText = "width:100%;border:1px solid #334155;border-radius:6px;background:#111827;color:#f8fafc;padding:7px 8px;font-size:12px;";
        select.innerHTML = `<option value="">No Ingredients sheet</option>${sheets.map((sheet) => `<option value="${escapeHtml(sheet.id)}">${escapeHtml(sheet.name || "Ingredients Sheet")}</option>`).join("")}`;
        const sheetValue = sceneMappedValue(refs.ingredients_scene_map, segment, index);
        select.value = sheets.some((sheet) => sheet.id === sheetValue) ? sheetValue : "";
        select.onchange = () => {
          collectMappingsFromDom();
          renderScenes();
        };
        const locationSelect = document.createElement("select");
        locationSelect.dataset.ingredientsLocationMap = "1";
        locationSelect.dataset.sceneId = segment.id || "";
        locationSelect.style.cssText = select.style.cssText;
        locationSelect.innerHTML = `<option value="">No location text</option>${locations.map((location) => `<option value="${escapeHtml(location.id)}">${escapeHtml(location.name || "Location")}</option>`).join("")}`;
        const rawLocationValue = sceneMappedValue(refs.scene_map, segment, index);
        const locationMatch = locations.find((location) => String(location.id || "") === rawLocationValue)
          || locations.find((location) => locationKey(location.name) === locationKey(rawLocationValue));
        const locationValue = locationMatch?.id || rawLocationValue;
        if (locationMatch?.id && rawLocationValue !== locationMatch.id) refs.scene_map[segment.id] = locationMatch.id;
        const hasLocationOption = Boolean(locationMatch);
        if (locationValue && !hasLocationOption) {
          const missingOption = new Option("Missing saved location", locationValue);
          missingOption.dataset.missingLocation = "1";
          locationSelect.append(missingOption);
        }
        locationSelect.value = hasLocationOption || locationValue ? locationValue : "";
        locationSelect.disabled = !locations.length;
        locationSelect.title = locations.length ? "Location text sent to Gemma/prompt planning for this scene." : "Add or import locations with Location Text Tools below.";
        locationSelect.onchange = () => collectMappingsFromDom();
        row.append(label, select, makeSheetPreview(select.value), locationSelect);
        sceneList.append(row);
      });
    }

    function renderAll() {
      repairIngredientLocationMappings();
      renderSheets();
      renderLocations();
      renderScenes();
    }

    addSheet.onclick = () => {
      collectMappingsFromDom();
      refs.ingredients_sheets.push({
        id: `ingredients_${Date.now()}_${Math.floor(Math.random() * 10000)}`,
        name: `Ingredients Sheet ${refs.ingredients_sheets.length + 1}`,
        description: "",
        image: { path: "", data: "", name: "" },
      });
      renderAll();
    };
    describeSheets.onclick = () => describeMissingIngredientSheetsWithGemma();
    autoMap.onclick = () => {
      collectMappingsFromDom();
      const sources = Object.fromEntries(Object.entries(sourceInputs).map(([key, input]) => [key, input.checked]));
      const result = autoMapIngredientsSheets(refs, sources);
      refs = result.refs;
      renderScenes();
      toast(`Auto-mapped ${result.mapped} scene${result.mapped === 1 ? "" : "s"} by Ingredients sheet name.`);
    };
    clearMappings.onclick = () => {
      refs.ingredients_scene_map = {};
      renderScenes();
    };
    extractLocations.onclick = async () => {
      let progress = null;
      try {
        collectMappingsFromDom();
        extractLocations.disabled = true;
        autoMapLocations.disabled = true;
        extractLocations.textContent = "Extracting...";
        progress = createProgressWindow("Extracting Ingredients locations", { zIndex: 100008 });
        const sceneInputs = allEditableSegments().map((segment, index) => {
          const planningNotes = [segment.notes || "", segment.timeline_note || "", segment.i2v_notes || ""].filter(Boolean).join("\n");
          return {
            id: segment.id,
            label: segment.label || `Scene ${index + 1}`,
            concept: sceneConceptPromptText(segment),
            notes: planningNotes,
            lyric: segment.lyric_text || "",
          };
        });
        const planningScenes = sceneInputs
          .map((scene) => ({ id: scene.id, label: scene.label, concept: scene.concept, notes: scene.notes }))
          .filter((scene) => String(scene.concept || scene.notes || "").trim());
        const lyricScenes = sceneInputs.filter((scene) => String(scene.lyric || "").trim());
        if (!planningScenes.length && !lyricScenes.length) throw new Error("Gemma Extract needs scene prompts, notes, lyrics, or timeline notes first.");
        const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
        if (!modelFile && !["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner)) throw new Error("Choose a non-vision Gemma model first, or use LM Studio, LLM API, or your own server in LLM Runner.");
        const useLyricsScout = Boolean(!planningScenes.length && lyricScenes.length);
        const styleTheme = await locationExtractionStyleTheme(locationStyleTheme.value);
        progress.set(`${useLyricsScout ? "Asking Gemma location scout to create locations from lyrics" : "Asking Gemma for reusable location descriptions"}...\n${gemmaRunnerLine()}`, 15);
        const data = useLyricsScout
          ? await postJson("/vrgdg/music_builder/wizard_locations_from_lyrics", {
            ...textGemmaRunnerPayload(),
            model_file: modelFile,
            lyrics_text: lyricScenes.map((scene, index) => `Scene ${index + 1}: ${scene.lyric}`).join("\n"),
            style_theme: styleTheme,
            subject_context: ingredientSubjectContextForLocations(),
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
            subject_scene_text: "",
            style_theme: styleTheme,
            subject_context: ingredientSubjectContextForLocations(),
            existing_locations: refs.locations.map((item) => ({ name: item.name || "", description: item.description || "" })),
            max_locations: refs.max_generated_locations || 8,
            n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
            unload_after: true,
          }, 10 * 60 * 1000);
        const { added, updated } = upsertLocationText((data.locations || []).map((item) => ({ name: item.name, description: item.description })));
        if (added || updated) refs.locations_cleared = false;
        refs.use_location_references = true;
        renderAll();
        progress.set(`Extracted Ingredients location text.\nAdded: ${added}\nUpdated: ${updated}`, 100);
        progress.close(1800);
        toast(`Extracted ${added} location${added === 1 ? "" : "s"}.`);
      } catch (error) {
        progress?.set(`Error:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
      } finally {
        extractLocations.disabled = false;
        autoMapLocations.disabled = false;
        extractLocations.textContent = "Gemma Extract";
      }
    };
    gptLocationScout.onclick = () => openLocationScoutGptForRefs(refs, locationStyleTheme.value || "", {
      onImportList: () => importLocations.click(),
    });
    addLocation.onclick = () => {
      collectMappingsFromDom();
      refs.locations = Array.isArray(refs.locations) ? refs.locations : [];
      refs.locations_cleared = false;
      refs.locations.push({
        id: `loc_${Date.now()}_${refs.locations.length}_${Math.floor(Math.random() * 10000)}`,
        name: `Location ${refs.locations.length + 1}`,
        description: "",
        image: { path: "", data: "", name: "" },
      });
      refs.use_location_references = true;
      renderAll();
    };
    clearLocations.onclick = () => {
      if (!window.confirm("Clear Ingredients location text and scene location mappings?")) return;
      refs.locations = [];
      refs.scene_map = {};
      refs.use_location_references = false;
      refs.locations_cleared = true;
      renderAll();
    };
    autoMapLocations.onclick = () => {
      collectMappingsFromDom();
      const locations = refs.locations || [];
      if (!locations.length) {
        toast("Add or import locations before auto-mapping.", true);
        return;
      }
      refs.scene_map = refs.scene_map && typeof refs.scene_map === "object" ? refs.scene_map : {};
      let mapped = 0;
      for (const segment of allEditableSegments()) {
        const text = [
          sceneConceptPromptText(segment),
          segment.notes || "",
          segment.timeline_note || "",
          segment.i2v_notes || "",
          segment.lyric_text || "",
        ].join(" ").toLowerCase();
        const match = locations.find((location) => {
          const name = String(location.name || "").trim().toLowerCase();
          return name && text.includes(name);
        });
        if (match?.id) {
          refs.scene_map[segment.id] = match.id;
          mapped += 1;
        }
      }
      refs.use_location_references = Boolean(refs.locations.length || Object.keys(refs.scene_map || {}).length);
      renderScenes();
      toast(`Auto-mapped ${mapped} scene${mapped === 1 ? "" : "s"} by location name.`);
    };
    importLocations.onclick = () => {
      const importBackdrop = document.createElement("div");
      importBackdrop.style.cssText = "position:fixed;inset:0;z-index:100009;background:rgba(0,0,0,.65);display:flex;align-items:center;justify-content:center;";
      const importBox = document.createElement("div");
      importBox.style.cssText = "width:min(760px,calc(100vw - 34px));max-height:calc(100vh - 40px);border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.58);padding:14px;display:flex;flex-direction:column;gap:10px;";
      const importHeader = document.createElement("div");
      importHeader.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;";
      const importTitle = document.createElement("div");
      importTitle.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Import Ingredients Location Text / Scene Map</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Paste location text or scene-location JSON. Trigger fields are ignored here.</div>`;
      const gptTools = document.createElement("div");
      gptTools.style.cssText = "display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px;align-items:center;min-width:190px;";
      const mapperGpt = makeGptLinkButton("GPT: Locations Only", LOCATION_MAPPER_GPT_URL);
      mapperGpt.style.cssText += "white-space:nowrap;";
      gptTools.append(mapperGpt);
      importHeader.append(importTitle, gptTools);
      const help = document.createElement("div");
      help.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;font-size:12px;color:#dbeafe;line-height:1.45;max-height:230px;overflow:auto;";
      help.innerHTML = `
        <strong>Accepted formats</strong>
        <div style="margin-top:8px;">Location list JSON:</div>
        <pre style="white-space:pre-wrap;margin:6px 0 0;color:#e2e8f0;">[
  { "location": "Glass hallway", "description": "A long mirrored corridor..." },
  { "name": "Chrome vault corridor", "description": "A sealed industrial passage..." }
]</pre>
        <div style="margin-top:8px;">Combined location + scene map JSON:</div>
        <pre style="white-space:pre-wrap;margin:6px 0 0;color:#e2e8f0;">[
  { "scene": "scene1", "location": "Glass hallway", "description": "A long mirrored corridor..." },
  { "scene": "scene2", "location": "Chrome vault corridor", "description": "A sealed industrial passage..." }
]</pre>
        <div style="margin-top:8px;">Scene map JSON:</div>
        <pre style="white-space:pre-wrap;margin:6px 0 0;color:#e2e8f0;">{
  "scene1": { "location": "Glass hallway" },
  "scene2": { "location": "Chrome vault corridor" }
}</pre>
        <div style="margin-top:8px;">Quick text:</div>
        <pre style="white-space:pre-wrap;margin:6px 0 0;color:#e2e8f0;">Glass hallway = A long mirrored corridor...
Chrome vault corridor = A sealed industrial passage...</pre>`;
      const input = document.createElement("textarea");
      input.placeholder = "Paste location text or scene-location JSON...";
      input.style.cssText = "min-height:250px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;";
      const actions = document.createElement("div");
      actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
      const cancelImport = makeButton("Cancel");
      const applyImport = makeButton("Import", "primary");
      actions.append(cancelImport, applyImport);
      importBox.append(importHeader, help, input, actions);
      importBackdrop.append(importBox);
      document.body.append(importBackdrop);
      const closeImport = () => importBackdrop.remove();
      cancelImport.onclick = closeImport;
      importBackdrop.addEventListener("pointerdown", (event) => {
        if (event.target === importBackdrop) closeImport();
      });
      applyImport.onclick = () => {
        try {
          const parsed = parseIngredientLocationImport(input.value);
          const { added, updated } = upsertLocationText(parsed.locations);
          const mapped = applyIngredientLocationSceneMap(parsed.sceneMap);
          renderAll();
          closeImport();
          toast(`Imported location text. Added: ${added}. Updated: ${updated}. Scene mappings: ${mapped}.`);
        } catch (error) {
          toast(String(error?.message || error), true);
        }
      };
      input.focus();
    };
    cancel.onclick = () => backdrop.remove();
    close.onclick = () => backdrop.remove();
    save.onclick = async () => {
      pushHistory();
      collectMappingsFromDom();
      syncNamedIngredientsSheetsToSubjects();
      refs.location_style_theme = String(locationStyleTheme.value || "");
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
      const applied = applyIngredientsReferenceMappings(state.fluxReferenceBuilder);
      syncInspector();
      syncPreview(activeSegment());
      render();
      await autoSaveSessionQuiet("ingredients reference builder");
      toast(`Ingredients Reference Builder saved. Applied ${applied} mapped sheet${applied === 1 ? "" : "s"} to scenes.`);
      backdrop.remove();
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
    renderAll();
  }

  return { openIngredientsReferenceBuilderModal };
}
