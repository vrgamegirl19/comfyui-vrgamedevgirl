import { postJson } from "./comfy_api.mjs";
import { SCENE_MAPPING_GPT_URL } from "./constants.mjs";
import {
  applyCompactButtonLabel,
  copyTextToClipboard,
  escapeHtml,
  makeButton,
  makeInput,
  makeSelect,
  toast,
} from "./controls.mjs";
import { hasReferenceImage } from "./llm_runner.mjs";
import {
  formatCueTime,
  lyricCueTextParts,
  miniMaxEffectiveCueEnd,
  miniMaxNextCueStartTime,
  singerCuePlaybackRangeForCue,
  syncCueEndBoundariesFromNextStarts,
} from "./lyric_cues.mjs";
import { normalizeMiniMaxH3Mode } from "./minimax_h3.mjs";
import { flattenLyricForPrompt } from "./prompt_text.mjs";
import { subjectExtraTargetId } from "./reference_data.mjs";

function cleanSceneMapName(value = "") {
  return String(value || "").trim().toLowerCase().replace(/\s+/g, " ");
}

function sceneMappingImportSceneNumber(value) {
  const text = String(value || "").trim();
  const match = text.match(/(?:scene|segment|clip)?\s*#?\s*(\d+)/i);
  return match ? Number(match[1]) : Number(text) || 0;
}

function normalizeSceneMappingSubjectNames(value) {
  if (Array.isArray(value)) {
    return value.map((item) => {
      if (item && typeof item === "object") return item.name || item.subject || item.character || item.id || "";
      return item;
    }).map((item) => String(item || "").trim()).filter(Boolean);
  }
  if (value && typeof value === "object") {
    return normalizeSceneMappingSubjectNames(value.subjects || value.characters || value.names || value.name || value.subject || value.character);
  }
  return String(value || "")
    .split(/[,;|]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseGptSceneMappingJson(rawText) {
  const text = String(rawText || "").trim();
  if (!text) return [];
  const data = JSON.parse(text);
  const source = Array.isArray(data)
    ? data
    : (data.scenes || data.scene_mappings || data.sceneMappings || data.mappings || data.scene_map || data.sceneMap || []);
  if (Array.isArray(source)) return source;
  if (source && typeof source === "object") {
    return Object.entries(source).map(([key, value]) => {
      if (value && typeof value === "object") return { scene: sceneMappingImportSceneNumber(key), ...value };
      return { scene: sceneMappingImportSceneNumber(key), location: value };
    });
  }
  return [];
}

export function createReferenceSceneMapping({
  allEditableSegments, autoSaveSessionQuiet, exportGptSceneContext, globalPerformanceControls, imageLabel,
  imageSrc, launchAdvancedLineMapping, logicalReferenceSubjects, logicalSubjectIdsForScene, mappingList,
  mappingNote, miniMaxH3ModeForSegment, miniMaxProject, normalizeLyricCueMapForSegment,
  openReferenceImagePreview, playSingerCueRange, projectContextPath, pushHistory, referenceImagesEnabled,
  refs, renderAll, renderLocations, renderSubjects, sceneDisplayName, sceneReferenceMapArray, sceneSlotNumber,
  singerCueRelativePlayheadTime, subjectPreviewImages, syncInspector, syncPerformerInspectorForSegment,
  useLocations, useSubject,
}) {
  function performerIdsForMappingSegment(segment, logicalSubjects = logicalReferenceSubjects(refs)) {
    const savedIds = sceneReferenceMapArray(refs.performer_scene_map, segment);
    if (savedIds.length) return savedIds.map(String).filter(Boolean);
    const selected = new Set((Array.isArray(segment?.lyric_singers) ? segment.lyric_singers : [])
      .map((value) => String(value || "").trim().toLowerCase())
      .filter(Boolean));
    if (!selected.size) return [];
    return logicalSubjects
      .filter((subject) => selected.has(String(subject?.id || "").trim().toLowerCase()) || selected.has(String(subject?.name || "").trim().toLowerCase()))
      .map((subject) => String(subject.id || "").trim())
      .filter(Boolean);
  }

  function ensureCueMapForMappedScene(segment, performerIds = []) {
    if (!segment || performerIds.length < 2) return;
    if (normalizeLyricCueMapForSegment(segment, refs, { preserveBlank: true }).length) return;
    const performers = logicalReferenceSubjects(refs).filter((subject) => performerIds.includes(String(subject.id)));
    segment.lyric_cue_map = lyricCueTextParts(segment.lyric_text).map((text, cueIndex) => {
      const performer = performers[cueIndex % performers.length] || performers[0] || {};
      return { type: "vocal", text, action_note: "", singer_id: performer.id || "", singer_name: performer.name || "", start: null, end: null };
    });
  }

  function renderGlobalPerformanceControls() {
    globalPerformanceControls.replaceChildren();
    const logicalSubjects = logicalReferenceSubjects(refs);
    const targets = allEditableSegments()
      .map((segment) => ({ segment, performerIds: performerIdsForMappingSegment(segment, logicalSubjects) }))
      .filter((item) => item.performerIds.length >= 2);
    const modes = new Set(targets.map((item) => String(item.segment.lyric_performance_mode || "together") === "cue_map" ? "cue_map" : "together"));
    const currentMode = !targets.length ? "together" : (modes.size === 1 ? Array.from(modes)[0] : "mixed");
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-weight:800;color:#cffafe;">Global performance mode</div><div style="font-size:11px;color:#94a3b8;margin-top:2px;">Applies to scenes with 2+ selected performers. Individual scene dropdowns still override this.</div>`;
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:minmax(180px,260px) minmax(110px,150px) minmax(0,1fr);gap:8px;align-items:center;";
    const modeSelect = makeSelect([
      { value: "mixed", label: "Mixed / keep current" },
      { value: "together", label: "Together / same full line" },
      { value: "cue_map", label: "Split cue map" },
    ], currentMode);
    const apply = makeButton("Apply To All", "primary");
    apply.disabled = !targets.length || modeSelect.value === "mixed";
    const status = document.createElement("div");
    status.style.cssText = "font-size:11px;color:#a5f3fc;line-height:1.35;";
    status.textContent = targets.length
      ? `${targets.length} mapped scene${targets.length === 1 ? "" : "s"} with 2+ performers. Current: ${currentMode === "mixed" ? "mixed per-scene modes" : modeSelect.options[modeSelect.selectedIndex]?.textContent || currentMode}.`
      : "No scenes currently have 2+ selected performers.";
    modeSelect.onchange = () => {
      apply.disabled = !targets.length || modeSelect.value === "mixed";
    };
    apply.onclick = () => {
      const mode = String(modeSelect.value || "");
      if (!["together", "cue_map"].includes(mode)) return;
      pushHistory();
      for (const item of targets) {
        item.segment.lyric_performance_mode = mode;
        if (mode === "cue_map") ensureCueMapForMappedScene(item.segment, item.performerIds);
      }
      syncInspector();
      renderMapping();
      autoSaveSessionQuiet("global MiniMax performance mode changed").catch(() => null);
      toast(`Set ${targets.length} mapped scene${targets.length === 1 ? "" : "s"} to ${mode === "cue_map" ? "Split cue map" : "Together"}.`);
    };
    row.append(modeSelect, apply, status);
    globalPerformanceControls.append(title, row);
  }

  function sceneMappingContextPath() {
    return projectContextPath("ReferenceBuilderSceneMappingContext.json");
  }

  function sceneMappingSegments() {
    return allEditableSegments().slice().sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
  }

  function sceneMappingSubjectByNameOrId(value) {
    const key = cleanSceneMapName(value);
    if (!key) return null;
    return (refs.subjects || []).find((subject) =>
      cleanSceneMapName(subject.id) === key
      || cleanSceneMapName(subject.name) === key
      || cleanSceneMapName(subject.label) === key
    ) || null;
  }

  function sceneMappingLocationByNameOrId(value) {
    const key = cleanSceneMapName(value);
    if (!key) return null;
    return (refs.locations || []).find((location) =>
      cleanSceneMapName(location.id) === key
      || cleanSceneMapName(location.name) === key
      || cleanSceneMapName(location.label) === key
    ) || null;
  }

  function buildSceneMappingContextJson() {
    const subjects = (refs.subjects || []).map((subject) => ({
      id: String(subject.id || ""),
      name: String(subject.name || "").trim(),
      type: String(subject.reference_type || "character"),
      description: String(subject.description || "").trim(),
    })).filter((subject) => subject.name);
    const locations = (refs.locations || []).map((location) => ({
      id: String(location.id || ""),
      name: String(location.name || "").trim(),
      description: String(location.description || "").trim(),
    })).filter((location) => location.name);
    const subjectById = new Map(subjects.map((subject) => [subject.id, subject]));
    const locationById = new Map(locations.map((location) => [location.id, location]));
    const scenes = sceneMappingSegments().map((segment, index) => {
      const sceneNumber = index + 1;
      const subjectIds = Array.isArray(refs.subject_scene_map?.[segment.id])
        ? refs.subject_scene_map[segment.id]
        : (Array.isArray(refs.subject_scene_map?.[String(sceneNumber)]) ? refs.subject_scene_map[String(sceneNumber)] : []);
      const locationId = String(refs.scene_map?.[segment.id] || refs.scene_map?.[String(sceneNumber)] || "");
      return {
        scene: sceneNumber,
        scene_id: String(segment.id || ""),
        label: String(segment.label || `Scene ${sceneNumber}`),
        lyric_section: String(segment.lyric_section || ""),
        lyric_line: String(segment.lyric_text || ""),
        story_beat: String(segment.story_beat || ""),
        no_character_present: Boolean(segment.no_character_present),
        current_subjects: subjectIds.map((id) => subjectById.get(String(id || ""))?.name || "").filter(Boolean),
        current_location: locationById.get(locationId)?.name || "",
      };
    });
    return {
      instructions: "Return a JSON object with a scenes array. Use only the subject and location names from this file unless the user explicitly says a scene should be unassigned. Match by scene number.",
      user_input: "Optional: tell the GPT who is singing, when no subject should appear, whether instrumentals should include subjects, or any story/location rules.",
      summary: {
        subjects: subjects.map(({ name, type, description }) => ({ name, type, description })),
        locations: locations.map(({ name, description }) => ({ name, description })),
      },
      subjects,
      locations,
      scenes,
      expected_output_example: {
        scenes: [
          {
            scene: 1,
            subjects: ["subject name or none"],
            location: "location name or Unassigned",
            no_character_present: false,
            reason: "short optional reason",
          },
        ],
      },
    };
  }

  async function exportSceneMappingContextForGpt() {
    const path = sceneMappingContextPath();
    if (!path) {
      toast("Create or load a project before exporting GPT scene mapping context.", true);
      return;
    }
    try {
      exportGptSceneContext.disabled = true;
      const data = buildSceneMappingContextJson();
      const content = JSON.stringify(data, null, 2);
      exportGptSceneContext.textContent = "Copying...";
      let copied = false;
      try {
        const handoff = [
          "Process the following pasted scene-mapping JSON as the complete input for this task.",
          "The JSON is pasted inline, not attached as a file. Do not say that an attachment is missing or ask me to re-upload it.",
          "Return only the requested scene-mapping JSON output after using the subjects, locations, lyric lines, and scene numbers from the data.",
          "```json",
          content,
          "```",
        ].join("\n\n");
        copied = await copyTextToClipboard(handoff);
      } catch (copyError) {
        console.warn("[VRGDG Music Builder] Could not copy GPT scene mapping context:", copyError);
      }
      const gptWindow = window.open(SCENE_MAPPING_GPT_URL, "_blank", "noopener,noreferrer");
      exportGptSceneContext.textContent = "Saving...";
      const result = await postJson("/vrgdg/music_builder/save_text_file", {
        path,
        content,
      }, 60000);
      toast(`${copied ? "Copied the GPT scene-mapping request and JSON to clipboard" : "Clipboard copy was blocked by the browser"}${gptWindow ? " and opened the GPT." : ", but the GPT popup was blocked."}\nSaved backup for ${data.scenes.length} scene${data.scenes.length === 1 ? "" : "s"} to:\n${result.path || path}`);
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      exportGptSceneContext.disabled = false;
      exportGptSceneContext.textContent = "Export GPT Context";
    }
  }

  function applyGptSceneMappings(items) {
    refs.subject_scene_map = refs.subject_scene_map && typeof refs.subject_scene_map === "object" ? refs.subject_scene_map : {};
    refs.scene_map = refs.scene_map && typeof refs.scene_map === "object" ? refs.scene_map : {};
    const segments = sceneMappingSegments();
    let subjectMapped = 0;
    let locationMapped = 0;
    let clearedSubjects = 0;
    let clearedLocations = 0;
    const unknownSubjects = new Set();
    const unknownLocations = new Set();
    const targetSegmentForItem = (item, fallbackIndex) => {
      const sceneId = String(item.scene_id || item.sceneId || item.id || "").trim();
      if (sceneId) {
        const byId = segments.find((segment) => String(segment.id || "") === sceneId);
        if (byId) return byId;
      }
      const sceneNumber = sceneMappingImportSceneNumber(item.scene_number ?? item.sceneNumber ?? item.scene ?? item.segment ?? item.clip ?? fallbackIndex + 1);
      if (sceneNumber > 0) return segments[sceneNumber - 1] || segments.find((segment) => sceneSlotNumber(segment) === sceneNumber) || null;
      return segments[fallbackIndex] || null;
    };
    items.forEach((item, index) => {
      if (!item || typeof item !== "object") return;
      const segment = targetSegmentForItem(item, index);
      if (!segment) return;
      const explicitNoCharacter = Boolean(item.no_character_present || item.noCharacterPresent || item.no_subject || item.noSubject);
      const subjectNames = normalizeSceneMappingSubjectNames(
        item.subjects ?? item.characters ?? item.character_refs ?? item.characterRefs ?? item.subject ?? item.character ?? item.singer
      );
      if (explicitNoCharacter || subjectNames.some((name) => /^(none|no character|no subject|unassigned|empty)$/i.test(name))) {
        delete refs.subject_scene_map[segment.id];
        segment.no_character_present = true;
        clearedSubjects += 1;
      } else if (subjectNames.length) {
        const ids = [];
        for (const name of subjectNames) {
          const subject = sceneMappingSubjectByNameOrId(name);
          if (subject?.id) ids.push(subject.id);
          else unknownSubjects.add(name);
        }
        const uniqueIds = Array.from(new Set(ids));
        if (uniqueIds.length) {
          refs.subject_scene_map[segment.id] = uniqueIds;
          segment.no_character_present = false;
          subjectMapped += 1;
        }
      }
      const locationName = String(item.location ?? item.location_name ?? item.locationName ?? item.setting ?? "").trim();
      if (/^(none|unassigned|empty|no location)$/i.test(locationName)) {
        delete refs.scene_map[segment.id];
        clearedLocations += 1;
      } else if (locationName) {
        const location = sceneMappingLocationByNameOrId(locationName);
        if (location?.id) {
          refs.scene_map[segment.id] = location.id;
          locationMapped += 1;
        } else {
          unknownLocations.add(locationName);
        }
      }
    });
    refs.use_subject_reference = Boolean((refs.subjects || []).length || Object.keys(refs.subject_scene_map || {}).length);
    refs.use_location_references = Boolean((refs.locations || []).length || Object.keys(refs.scene_map || {}).length);
    useSubject.input.checked = refs.use_subject_reference;
    useLocations.input.checked = refs.use_location_references;
    return {
      subjectMapped,
      locationMapped,
      clearedSubjects,
      clearedLocations,
      unknownSubjects: Array.from(unknownSubjects),
      unknownLocations: Array.from(unknownLocations),
    };
  }

  function openImportGptSceneMapDialog() {
    const importBackdrop = document.createElement("div");
    importBackdrop.style.cssText = "position:fixed;inset:0;z-index:100009;background:rgba(0,0,0,.65);display:flex;align-items:center;justify-content:center;";
    const importBox = document.createElement("div");
    importBox.style.cssText = "width:min(780px,calc(100vw - 34px));max-height:calc(100vh - 40px);border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.58);display:flex;flex-direction:column;overflow:hidden;";
    const importHeader = document.createElement("div");
    importHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;background:#083f4f;border-bottom:1px solid #155e75;padding:12px 14px;";
    const importTitle = document.createElement("div");
    importTitle.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Import GPT Scene Mapping</div><div style="font-size:12px;color:#cbd5e1;margin-top:3px;">Paste the GPT output JSON to map saved subjects and locations back to scenes.</div>`;
    const closeImport = makeButton("Close");
    importHeader.append(importTitle, closeImport);
    const note = document.createElement("div");
    note.style.cssText = "margin:12px 14px 0;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;font-size:12px;color:#cbd5e1;line-height:1.45;";
    note.innerHTML = `<b style="color:#e0f2fe;">Accepted JSON:</b><pre style="white-space:pre-wrap;margin:8px 0 0;color:#dbeafe;">{
  "scenes": [
    { "scene": 1, "subjects": ["the woman"], "location": "Neon motel pool" },
    { "scene": 2, "subjects": [], "location": "Foggy pine road", "no_character_present": true }
  ]
}</pre>`;
    const input = document.createElement("textarea");
    input.placeholder = "Paste GPT scene mapping JSON here...";
    input.spellcheck = false;
    input.style.cssText = "margin:12px 14px;min-height:300px;resize:vertical;border:1px solid #cbd5e1;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 14px 14px;";
    const cancelImport = makeButton("Cancel");
    const applyImport = makeButton("Import Scene Map", "primary");
    actions.append(cancelImport, applyImport);
    importBox.append(importHeader, note, input, actions);
    importBackdrop.append(importBox);
    document.body.append(importBackdrop);
    const closeDialog = () => importBackdrop.remove();
    closeImport.onclick = closeDialog;
    cancelImport.onclick = closeDialog;
    applyImport.onclick = () => {
      try {
        const items = parseGptSceneMappingJson(input.value);
        if (!items.length) throw new Error("No scene mappings were found in the pasted JSON.");
        const result = applyGptSceneMappings(items);
        renderAll();
        const warnings = [];
        if (result.unknownSubjects.length) warnings.push(`Unknown subjects: ${result.unknownSubjects.slice(0, 8).join(", ")}${result.unknownSubjects.length > 8 ? "..." : ""}`);
        if (result.unknownLocations.length) warnings.push(`Unknown locations: ${result.unknownLocations.slice(0, 8).join(", ")}${result.unknownLocations.length > 8 ? "..." : ""}`);
        toast(`Imported GPT scene map. Subjects mapped: ${result.subjectMapped}. Locations mapped: ${result.locationMapped}. Cleared: ${result.clearedSubjects + result.clearedLocations}.${warnings.length ? `\n${warnings.join("\n")}` : ""}`);
        closeDialog();
      } catch (error) {
        toast(String(error?.message || error), true);
      }
    };
    importBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === importBackdrop) closeDialog();
    });
    input.focus();
  }

  function renderMapping() {
    mappingList.innerHTML = "";
    renderGlobalPerformanceControls();
    const logicalSubjects = logicalReferenceSubjects(refs);
    const showSubjects = logicalSubjects.length > 0;
    const singleGlobalSubject = refs.use_subject_reference && logicalSubjects.length === 1;
    const makeMappingThumb = (image = {}, label = "Reference") => {
      const thumb = document.createElement("div");
      thumb.title = imageLabel(image) || label;
      thumb.style.cssText = "width:96px;height:72px;border:1px solid #155e75;border-radius:6px;background:#061620;overflow:hidden;display:flex;align-items:center;justify-content:center;flex:0 0 auto;";
      const src = imageSrc(image);
      if (src) {
        const img = document.createElement("img");
        img.src = src;
        img.alt = label;
        img.draggable = false;
        img.title = "Click to preview larger";
        img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;cursor:zoom-in;";
        img.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          openReferenceImagePreview(image, label);
        });
        thumb.append(img);
      } else {
        const empty = document.createElement("span");
        empty.textContent = "No img";
        empty.style.cssText = "font-size:11px;font-weight:900;color:#67e8f9;text-align:center;padding:6px;";
        thumb.append(empty);
      }
      return thumb;
    };
    const makeMappingPreview = (items = [], emptyText = "Unassigned") => {
      const wrap = document.createElement("div");
      wrap.style.cssText = "display:flex;gap:8px;align-items:center;min-height:92px;overflow-x:auto;padding:7px;border:1px solid #1e3a5f;border-radius:7px;background:#071422;scrollbar-width:thin;";
      if (!items.length) {
        const empty = document.createElement("div");
        empty.textContent = emptyText;
        empty.style.cssText = "font-size:11px;color:#94a3b8;padding:0 6px;";
        wrap.append(empty);
        return wrap;
      }
      items.slice(0, 8).forEach((item) => {
        const cell = document.createElement("div");
        cell.style.cssText = "display:flex;flex-direction:column;gap:4px;align-items:center;min-width:100px;max-width:118px;";
        cell.append(makeMappingThumb(item.image || {}, item.label || "Reference"));
        const caption = document.createElement("div");
        caption.textContent = item.label || "Reference";
        caption.style.cssText = "width:100%;font-size:11px;color:#cbd5e1;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        cell.append(caption);
        wrap.append(cell);
      });
      if (items.length > 8) {
        const more = document.createElement("div");
        more.textContent = `+${items.length - 8}`;
        more.style.cssText = "font-size:12px;font-weight:900;color:#67e8f9;padding:0 6px;";
        wrap.append(more);
      }
      return wrap;
    };
    const markVisualPicker = (preview, text = "Click to choose / change") => {
      preview.style.position = "relative";
      preview.style.paddingTop = "27px";
      const hint = document.createElement("div");
      hint.textContent = text;
      hint.style.cssText = "position:absolute;left:8px;top:6px;font-size:10px;font-weight:900;color:#22d3ee;text-transform:uppercase;letter-spacing:.03em;pointer-events:none;";
      preview.append(hint);
      return preview;
    };
    const openVisualMappingPicker = ({ title = "Choose", items = [], selectedIds = [], multiple = false, onApply }) => {
      const pickerBackdrop = document.createElement("div");
      pickerBackdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:22px;";
      const picker = document.createElement("div");
      picker.style.cssText = "width:min(980px,calc(100vw - 44px));max-height:calc(100vh - 48px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#0b1220;color:#f8fafc;padding:14px;display:flex;flex-direction:column;gap:12px;";
      const head = document.createElement("div");
      head.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;";
      const heading = document.createElement("div");
      heading.textContent = title;
      heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
      const closePicker = makeButton("Cancel");
      head.append(heading, closePicker);
      const choices = document.createElement("div");
      choices.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;";
      const selected = new Set(selectedIds.map(String));
      const renderChoices = () => {
        choices.replaceChildren();
        const unassigned = document.createElement("button");
        unassigned.type = "button";
        unassigned.textContent = multiple ? "Clear selection" : "Unassigned";
        unassigned.style.cssText = "min-height:132px;border:1px dashed #475569;border-radius:7px;background:#111827;color:#94a3b8;cursor:pointer;font-weight:900;";
        unassigned.onclick = () => { selected.clear(); if (!multiple) applySelection(); else renderChoices(); };
        choices.append(unassigned);
        items.forEach((item) => {
          const card = document.createElement("button");
          card.type = "button";
          const active = selected.has(String(item.id));
          card.style.cssText = `min-height:132px;border:2px solid ${active ? "#22d3ee" : "#334155"};border-radius:7px;background:${active ? "#083344" : "#111827"};color:#f8fafc;padding:8px;display:flex;flex-direction:column;gap:7px;align-items:center;cursor:pointer;`;
          card.append(makeMappingThumb(item.image || {}, item.label || "Reference"));
          const name = document.createElement("div");
          name.textContent = item.label || "Reference";
          name.style.cssText = "font-size:12px;font-weight:900;text-align:center;";
          card.append(name);
          card.onclick = () => {
            if (multiple) {
              if (selected.has(String(item.id))) selected.delete(String(item.id));
              else selected.add(String(item.id));
              renderChoices();
            } else {
              selected.clear();
              selected.add(String(item.id));
              applySelection();
            }
          };
          choices.append(card);
        });
      };
      const applySelection = () => { pickerBackdrop.remove(); onApply?.(Array.from(selected)); };
      const footer = document.createElement("div");
      footer.style.cssText = `display:${multiple ? "grid" : "none"};grid-template-columns:1fr 1fr;gap:8px;`;
      const cancel = makeButton("Cancel");
      const apply = makeButton("Apply Selection", "primary");
      cancel.onclick = () => pickerBackdrop.remove();
      apply.onclick = applySelection;
      closePicker.onclick = () => pickerBackdrop.remove();
      footer.append(cancel, apply);
      picker.append(head, choices, footer);
      pickerBackdrop.append(picker);
      document.body.append(pickerBackdrop);
      renderChoices();
    };
    const subjectPreviewItems = (selectedIds) => {
      const ids = new Set(selectedIds);
      const items = [];
      logicalSubjects.filter((subject) => ids.has(subject.id)).forEach((subject) => {
        const previews = subjectPreviewImages(subject);
        if (previews.length) {
          previews.forEach(({ image, label }) => items.push({ image, label: label || subject.name || "Subject" }));
        } else {
          items.push({ image: subject.image || {}, label: subject.name || "Subject" });
        }
      });
      return items;
    };
    const openCueMapEditor = (segment, performerIds = []) => {
      const performers = logicalSubjects.filter((subject) => performerIds.includes(String(subject.id)));
      if (performers.length < 2) {
        toast("Choose at least two performers before splitting lyric cues.", true);
        return;
      }
      const parts = lyricCueTextParts(segment.lyric_text);
      if (!parts.length) {
        toast("This scene needs lyric/dialogue text before cue mapping.", true);
        return;
      }
      const existing = normalizeLyricCueMapForSegment(segment, refs, { preserveBlank: true });
      const rows = (existing.length ? existing : parts.map((text, index) => {
        const performer = performers[index % performers.length] || performers[0];
        return { type: "vocal", text, action_note: "", singer_id: performer.id, singer_name: performer.name || "", start: null, end: null };
      })).map((cue, index) => {
        const performer = performers.find((subject) => subject.id === cue.singer_id) || performers[index % performers.length] || performers[0];
        return {
          type: cue.type === "instrumental" ? "instrumental" : "vocal",
          text: cue.text || parts[index] || "",
          action_note: cue.action_note || "",
          singer_id: cue.type === "instrumental" ? "" : performer?.id || "",
          singer_name: cue.type === "instrumental" ? "" : performer?.name || "",
          start: Number.isFinite(Number(cue.start)) ? Number(cue.start) : null,
          end: Number.isFinite(Number(cue.end)) ? Number(cue.end) : null,
        };
      });
      const cueBackdrop = document.createElement("div");
      cueBackdrop.style.cssText = "position:fixed;inset:0;z-index:100030;background:rgba(0,0,0,.76);display:flex;align-items:center;justify-content:center;padding:20px;box-sizing:border-box;";
      const cuePanel = document.createElement("div");
      cuePanel.style.cssText = "width:min(760px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow:auto;border:1px solid #155e75;border-radius:9px;background:#0b1220;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.72);padding:15px;display:flex;flex-direction:column;gap:12px;";
      const cueTitle = document.createElement("div");
      cueTitle.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Split Line Between Performers</div><div style="font-size:12px;color:#cbd5e1;margin-top:4px;">Assign each lyric/dialogue cue to exactly one performer. Everyone else stays silent during that cue.</div>`;
      const cueList = document.createElement("div");
      cueList.style.cssText = "display:flex;flex-direction:column;gap:8px;";
      const renderCueRows = () => {
        cueList.replaceChildren();
        rows.forEach((row, index) => {
          const line = document.createElement("div");
          line.style.cssText = "display:flex;flex-direction:column;gap:7px;border:1px solid #1e3a5f;border-radius:7px;background:#071422;padding:8px;";
          const top = document.createElement("div");
          top.style.cssText = "display:grid;grid-template-columns:minmax(104px,.35fr) minmax(132px,.55fr) minmax(0,1fr) 34px;gap:7px;align-items:center;";
          const type = makeSelect(["vocal", "instrumental"], row.type || "vocal");
          type.options[0].textContent = "Lyric";
          type.options[1].textContent = "Instrumental";
          const text = makeInput(row.type === "instrumental" ? row.action_note || "" : row.text || "");
          text.placeholder = row.type === "instrumental" ? "action note while nobody sings..." : "lyric cue...";
          text.oninput = () => { row.text = text.value; };
          const performer = makeSelect(performers.map((subject) => subject.id), row.singer_id || performers[index % performers.length]?.id || "");
          Array.from(performer.options).forEach((option) => {
            const subject = performers.find((item) => item.id === option.value);
            option.textContent = subject?.name || "Character";
          });
          performer.disabled = row.type === "instrumental";
          type.onchange = () => {
            row.type = type.value === "instrumental" ? "instrumental" : "vocal";
            if (row.type === "instrumental") {
              row.action_note = row.action_note || "";
              row.text = "";
              row.singer_id = "";
              row.singer_name = "";
            } else {
              const nextPerformer = performers[index % performers.length] || performers[0] || {};
              row.text = row.text || row.action_note || "";
              row.action_note = "";
              row.singer_id = row.singer_id || nextPerformer.id || "";
              row.singer_name = row.singer_name || nextPerformer.name || "";
            }
            renderCueRows();
          };
          performer.onchange = () => {
            row.singer_id = performer.value;
            row.singer_name = performers.find((subject) => subject.id === performer.value)?.name || "";
          };
          const remove = makeButton("×");
          remove.title = "Remove cue";
          remove.style.cssText += "min-width:0;width:34px;padding:6px 0;font-size:18px;line-height:1;";
          remove.onclick = () => {
            rows.splice(index, 1);
            renderCueRows();
          };
          text.oninput = () => {
            if (row.type === "instrumental") row.action_note = text.value;
            else row.text = text.value;
          };
          top.append(type, performer, text, remove);
          const timing = document.createElement("div");
          timing.style.cssText = "display:grid;grid-template-columns:minmax(44px,.45fr) minmax(44px,.45fr) 58px 64px 58px 58px 60px;gap:7px;align-items:center;";
          const rowRange = singerCuePlaybackRangeForCue(segment, rows, index);
          const effectiveEnd = miniMaxEffectiveCueEnd(segment, rows, index);
          const startLabel = document.createElement("div");
          startLabel.textContent = `Start: ${formatCueTime(row.start)}`;
          startLabel.style.cssText = "font-size:11px;color:#a5f3fc;";
          const endLabel = document.createElement("div");
          endLabel.textContent = `End: ${formatCueTime(effectiveEnd)}`;
          endLabel.style.cssText = "font-size:11px;color:#a5f3fc;";
          const playCue = makeButton("Play Cue", "primary");
          const playFromCue = makeButton("Play From Here");
          const setStart = makeButton("Set Start");
          const setEnd = makeButton("Set End");
          const clearTime = makeButton("Clear Timing");
          applyCompactButtonLabel(playCue, "Play\nCue", { noMap: true, minWidth: 0, padding: "6px 5px", title: "Play only this cue's timed audio range." });
          applyCompactButtonLabel(playFromCue, "From\nHere", { noMap: true, minWidth: 0, padding: "6px 5px", title: "Play from this cue start." });
          applyCompactButtonLabel(setStart, "Set\nStart", { noMap: true, minWidth: 0, padding: "6px 5px" });
          applyCompactButtonLabel(setEnd, "Set\nEnd", { noMap: true, minWidth: 0, padding: "6px 5px" });
          applyCompactButtonLabel(clearTime, "Clear\nTime", { noMap: true, minWidth: 0, padding: "6px 5px", title: "Clear Timing" });
          playCue.title = "Play from this cue start to its end, the next cue start, or the scene end.";
          playFromCue.title = "Play from this cue start, or from the scene start if no cue start is set.";
          playCue.onclick = () => {
            playSingerCueRange(segment, rowRange.start, rowRange.end, playCue, "Play Cue");
          };
          playFromCue.onclick = () => {
            playSingerCueRange(segment, rowRange.start, null, playFromCue, "Play From Here");
          };
          setStart.onclick = () => {
            row.start = singerCueRelativePlayheadTime(segment);
            if (index > 0) rows[index - 1].end = row.start;
            if (Number.isFinite(Number(row.end)) && row.end <= row.start) row.end = null;
            syncCueEndBoundariesFromNextStarts(rows);
            renderCueRows();
          };
          setEnd.onclick = () => {
            row.end = singerCueRelativePlayheadTime(segment);
            if (Number.isFinite(Number(row.start)) && row.end <= row.start) {
              toast("Cue end must be after cue start.", true);
              row.end = null;
            } else if (index + 1 < rows.length) {
              rows[index + 1].start = row.end;
              if (Number.isFinite(Number(rows[index + 1].end)) && rows[index + 1].end <= rows[index + 1].start) rows[index + 1].end = null;
            }
            syncCueEndBoundariesFromNextStarts(rows);
            renderCueRows();
          };
          clearTime.onclick = () => { row.start = null; row.end = null; renderCueRows(); };
          timing.append(startLabel, endLabel, playCue, playFromCue, setStart, setEnd, clearTime);
          line.append(top, timing);
          cueList.append(line);
        });
      };
      const cueActions = document.createElement("div");
      cueActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;";
      const addCue = makeButton("Add Lyric Cue");
      const addInstrumental = makeButton("Add Instrumental");
      const cancelCue = makeButton("Cancel");
      const saveCue = makeButton("Save Cue Map", "primary");
      applyCompactButtonLabel(addCue, "Add\nLyric Cue", { noMap: true, padding: "7px 6px" });
      applyCompactButtonLabel(addInstrumental, "Add\nInstrumental", { noMap: true, padding: "7px 6px" });
      applyCompactButtonLabel(saveCue, "Save\nCue Map", { noMap: true, padding: "7px 6px" });
      addCue.onclick = () => {
        const performer = performers[rows.length % performers.length] || performers[0];
        const start = miniMaxNextCueStartTime(segment, rows);
        rows.push({ type: "vocal", text: "", action_note: "", singer_id: performer?.id || "", singer_name: performer?.name || "", start, end: null });
        renderCueRows();
      };
      addInstrumental.onclick = () => {
        const start = miniMaxNextCueStartTime(segment, rows);
        rows.push({ type: "instrumental", text: "", action_note: "", singer_id: "", singer_name: "", start, end: null });
        renderCueRows();
      };
      cancelCue.onclick = () => cueBackdrop.remove();
      saveCue.onclick = () => {
        const saved = rows.map((row, index) => {
          const performer = performers.find((subject) => subject.id === row.singer_id) || performers[index % performers.length] || performers[0];
          return {
            type: row.type === "instrumental" ? "instrumental" : "vocal",
            text: row.type === "instrumental" ? "" : flattenLyricForPrompt(row.text),
            action_note: row.type === "instrumental" ? String(row.action_note || "").trim() : "",
            singer_id: row.type === "instrumental" ? "" : performer?.id || "",
            singer_name: row.type === "instrumental" ? "" : performer?.name || "",
            start: Number.isFinite(Number(row.start)) ? Math.max(0, Number(row.start)) : null,
            end: Number.isFinite(Number(row.end)) ? Math.max(0, Number(row.end)) : null,
          };
        }).filter((row) => row.type === "instrumental" || (row.text && row.singer_id));
        if (!saved.length) {
          toast("Add at least one lyric cue before saving.", true);
          return;
        }
        segment.lyric_performance_mode = "cue_map";
        segment.lyric_cue_map = saved;
        syncPerformerInspectorForSegment(segment);
        cueBackdrop.remove();
        renderMapping();
      };
      cueActions.append(addCue, addInstrumental, cancelCue, saveCue);
      cuePanel.append(cueTitle, cueList, cueActions);
      cueBackdrop.append(cuePanel);
      document.body.append(cueBackdrop);
      cueBackdrop.addEventListener("pointerdown", (event) => {
        if (event.target === cueBackdrop) cueBackdrop.remove();
      });
      renderCueRows();
    };
    mappingNote.textContent = singleGlobalSubject
      ? `Each row shows its current lyric or dialogue line. Click the visual cards to choose who is present, who performs the line, and the scene location.`
      : showSubjects
      ? `Each row shows its current lyric or dialogue line. Click any Present, Performs Line, or Location image card to choose or change its visual mapping.`
      : `Each row shows its current lyric or dialogue line. Click the Location image card to choose or change it; choose Unassigned for no location reference.`;
    allEditableSegments().forEach((segment, index) => {
      const row = document.createElement("div");
      row.style.cssText = `border:1px solid #334155;border-radius:7px;background:linear-gradient(135deg,#0f172a,#111827);padding:10px;display:grid;grid-template-columns:minmax(180px,.8fr) ${showSubjects ? "minmax(230px,1fr) minmax(190px,.8fr) " : ""}minmax(240px,1fr) 88px;gap:10px;align-items:stretch;`;
      const label = document.createElement("div");
      const lyricLine = String(segment.lyric_text || "").trim();
      label.innerHTML = `<div style="font-size:11px;color:#67e8f9;font-weight:900;text-transform:uppercase;">Scene ${index + 1}</div><div style="font-size:13px;color:#f8fafc;font-weight:900;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(segment.label || `Scene ${index + 1}`)}</div><div style="font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;margin-top:9px;">Lyric / dialogue</div><div title="${escapeHtml(lyricLine || "No lyric or dialogue for this scene")}" style="font-size:12px;color:${lyricLine ? "#fde68a" : "#71717a"};line-height:1.35;margin-top:3px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;overflow-wrap:anywhere;">${escapeHtml(lyricLine || "No lyric — instrumental / visual scene")}</div>`;
      label.style.cssText = "min-width:0;border:1px solid #1e3a5f;border-radius:7px;background:#071422;padding:10px;display:flex;flex-direction:column;justify-content:center;";
      const subjectSelect = document.createElement("select");
      subjectSelect.multiple = true;
      subjectSelect.dataset.subjectMapSegmentId = segment.id;
      subjectSelect.size = Math.min(4, Math.max(2, logicalSubjects.length || 2));
      subjectSelect.style.display = "none";
      logicalSubjects.forEach((subject) => {
        const extraCount = refs.subjects.filter((item) => subjectExtraTargetId(item) === subject.id).length;
        subjectSelect.append(new Option(`${subject.name || "Character"}${extraCount ? ` (${extraCount + 1} ref images)` : ""}`, subject.id));
      });
      const selectedSubjectIds = new Set(logicalSubjectIdsForScene(refs, segment));
      if (!selectedSubjectIds.size && singleGlobalSubject && logicalSubjects[0]?.id) selectedSubjectIds.add(logicalSubjects[0].id);
      for (const option of subjectSelect.options) option.selected = selectedSubjectIds.has(option.value);
      subjectSelect.onchange = () => {
        refs.subject_scene_map[segment.id] = Array.from(subjectSelect.selectedOptions).map((option) => option.value);
        renderSubjects();
        renderMapping();
      };
      const select = document.createElement("select");
      select.dataset.locationMapSegmentId = segment.id;
      select.style.display = "none";
      select.append(new Option("Unassigned", ""));
      refs.locations.forEach((location) => select.append(new Option(location.name || "Location", location.id)));
      select.value = refs.scene_map?.[segment.id] || "";
      select.onchange = () => {
        refs.scene_map[segment.id] = select.value;
        renderLocations();
        renderMapping();
      };
      const subjectPanel = document.createElement("div");
      subjectPanel.style.cssText = referenceImagesEnabled
        ? "display:grid;grid-template-columns:minmax(120px,.75fr) minmax(150px,1fr);gap:8px;align-items:stretch;min-width:0;"
        : "display:grid;grid-template-columns:minmax(160px,1fr);gap:8px;align-items:stretch;min-width:0;";
      if (showSubjects) {
        const presentTitle = document.createElement("div");
        presentTitle.textContent = "Present";
        presentTitle.style.cssText = "font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;grid-column:1/-1;";
        const presentPreview = markVisualPicker(makeMappingPreview(subjectPreviewItems(selectedSubjectIds), "Choose characters present"));
        presentPreview.style.cursor = "pointer";
        presentPreview.style.gridColumn = "1 / -1";
        presentPreview.onclick = () => openVisualMappingPicker({
          title: `${sceneDisplayName(segment, index)} — Characters Present`,
          items: logicalSubjects.map((subject) => ({ id: subject.id, label: subject.name || "Character", image: subjectPreviewImages(subject)[0]?.image || subject.image || {} })),
          selectedIds: Array.from(selectedSubjectIds), multiple: true,
          onApply: (ids) => { if (ids.length) refs.subject_scene_map[segment.id] = ids; else delete refs.subject_scene_map[segment.id]; renderMapping(); },
        });
        subjectPanel.append(presentTitle, subjectSelect, presentPreview);
      }
      const performerPanel = document.createElement("div");
      performerPanel.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:0;border:1px solid #1e3a5f;border-radius:7px;background:#071422;padding:8px;";
      const performerTitle = document.createElement("div");
      performerTitle.textContent = "Performs line";
      performerTitle.style.cssText = "font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;";
      const performerSelect = document.createElement("select");
      performerSelect.multiple = true;
      performerSelect.size = Math.min(4, Math.max(2, logicalSubjects.length || 2));
      performerSelect.style.display = "none";
      const savedPerformerIds = sceneReferenceMapArray(refs.performer_scene_map, segment);
      const selectedPerformers = new Set(savedPerformerIds.length
        ? savedPerformerIds.map((value) => String(value || "").trim().toLowerCase())
        : (Array.isArray(segment.lyric_singers) ? segment.lyric_singers : []).map((value) => String(value || "").trim().toLowerCase()));
      logicalSubjects.forEach((subject) => {
        const label = String(subject.name || "Character").trim();
        const option = new Option(label, subject.id);
        option.selected = selectedPerformers.has(label.toLowerCase()) || selectedPerformers.has(String(subject.id || "").toLowerCase());
        performerSelect.append(option);
      });
      performerSelect.onchange = () => {
        const selectedOptions = Array.from(performerSelect.selectedOptions || []);
        segment.lyric_singers = selectedOptions.map((option) => option.textContent || option.value).filter(Boolean);
        if (!refs.performer_scene_map || typeof refs.performer_scene_map !== "object") refs.performer_scene_map = {};
        const performerIds = selectedOptions.map((option) => option.value).filter(Boolean);
        if (performerIds.length) refs.performer_scene_map[segment.id] = performerIds;
        else delete refs.performer_scene_map[segment.id];
        const present = new Set(Array.isArray(refs.subject_scene_map?.[segment.id]) ? refs.subject_scene_map[segment.id] : []);
        selectedOptions.forEach((option) => present.add(option.value));
        if (present.size) refs.subject_scene_map[segment.id] = Array.from(present);
        syncPerformerInspectorForSegment(segment);
        renderMapping();
        autoSaveSessionQuiet("performer assignment changed").catch(() => null);
      };
      const selectedPerformerIds = Array.from(performerSelect.selectedOptions || []).map((option) => option.value);
      const performerPreview = markVisualPicker(makeMappingPreview(subjectPreviewItems(new Set(selectedPerformerIds)), "Choose singer / speaker"));
      performerPreview.style.cursor = "pointer";
      performerPreview.onclick = () => openVisualMappingPicker({
        title: `${sceneDisplayName(segment, index)} — Performs This Line`,
        items: logicalSubjects.map((subject) => ({ id: subject.id, label: subject.name || "Character", image: subjectPreviewImages(subject)[0]?.image || subject.image || {} })),
        selectedIds: selectedPerformerIds, multiple: true,
        onApply: (ids) => {
          segment.lyric_singers = logicalSubjects.filter((subject) => ids.includes(String(subject.id))).map((subject) => subject.name || "Character");
          if (!refs.performer_scene_map || typeof refs.performer_scene_map !== "object") refs.performer_scene_map = {};
          if (ids.length) refs.performer_scene_map[segment.id] = ids.map(String);
          else delete refs.performer_scene_map[segment.id];
          const present = new Set(Array.isArray(refs.subject_scene_map?.[segment.id]) ? refs.subject_scene_map[segment.id] : []);
          ids.forEach((id) => present.add(id));
          if (present.size) refs.subject_scene_map[segment.id] = Array.from(present);
          syncPerformerInspectorForSegment(segment);
          renderMapping();
          autoSaveSessionQuiet("performer assignment changed").catch(() => null);
        },
      });
      performerPanel.append(performerTitle, performerSelect, performerPreview);
      if (selectedPerformerIds.length >= 2) {
        const modeRow = document.createElement("div");
        modeRow.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;align-items:center;";
        const modeSelect = makeSelect(["together", "cue_map"], segment.lyric_performance_mode || "together");
        modeSelect.options[0].textContent = "Together";
        modeSelect.options[1].textContent = "Split cue map";
        modeSelect.title = "Together means all selected performers sing/speak the same full line. Split cue map assigns lyric chunks to specific performers.";
        const editCueMap = makeButton("Cues", "primary");
        editCueMap.title = "Assign lyric chunks to performers for this scene.";
        editCueMap.style.cssText += "min-width:58px;padding:7px 8px;";
        modeSelect.onchange = () => {
          segment.lyric_performance_mode = modeSelect.value;
          if (modeSelect.value === "cue_map" && !normalizeLyricCueMapForSegment(segment, refs, { preserveBlank: true }).length) {
            const performers = logicalSubjects.filter((subject) => selectedPerformerIds.includes(String(subject.id)));
            segment.lyric_cue_map = lyricCueTextParts(segment.lyric_text).map((text, cueIndex) => {
              const performer = performers[cueIndex % performers.length] || performers[0] || {};
              return { text, singer_id: performer.id || "", singer_name: performer.name || "" };
            });
          }
          syncPerformerInspectorForSegment(segment);
          renderMapping();
        };
        editCueMap.onclick = () => openCueMapEditor(segment, selectedPerformerIds);
        modeRow.append(modeSelect, editCueMap);
        const summary = document.createElement("div");
        summary.style.cssText = "font-size:10px;color:#a5f3fc;line-height:1.35;";
        const cueCount = normalizeLyricCueMapForSegment(segment, refs, { preserveBlank: true }).length;
        summary.textContent = segment.lyric_performance_mode === "cue_map"
          ? `${cueCount || 0} mapped cue${cueCount === 1 ? "" : "s"}; only assigned performer sings each cue.`
          : "All selected performers sing/speak the same full line together.";
        performerPanel.append(modeRow, summary);
      } else if (segment.lyric_performance_mode !== "together" && !segment.lyric_shot_word_timing_enabled) {
        segment.lyric_performance_mode = "together";
        segment.lyric_cue_map = [];
        syncPerformerInspectorForSegment(segment);
      }
      const locationPanel = document.createElement("div");
      locationPanel.style.cssText = referenceImagesEnabled
        ? "display:grid;grid-template-columns:minmax(120px,.78fr) minmax(150px,1fr);gap:8px;align-items:stretch;min-width:0;"
        : "display:grid;grid-template-columns:minmax(160px,1fr);gap:8px;align-items:stretch;min-width:0;";
      const selectedLocation = refs.locations.find((location) => location.id === select.value);
      const locationPreview = markVisualPicker(makeMappingPreview(selectedLocation ? [{ image: selectedLocation.image || {}, label: selectedLocation.name || "Location" }] : [], "Choose location"));
      locationPreview.style.cursor = "pointer";
      locationPreview.style.gridColumn = "1 / -1";
      locationPreview.onclick = () => openVisualMappingPicker({
        title: `${sceneDisplayName(segment, index)} — Location`,
        items: refs.locations.map((location) => ({ id: location.id, label: location.name || "Location", image: location.image || {} })),
        selectedIds: select.value ? [select.value] : [], multiple: false,
        onApply: (ids) => { if (ids[0]) refs.scene_map[segment.id] = ids[0]; else delete refs.scene_map[segment.id]; renderMapping(); },
      });
      locationPanel.append(select, locationPreview);
      row.append(label);
      if (showSubjects) row.append(subjectPanel, performerPanel);
      row.append(locationPanel);
      const extraMode = normalizeMiniMaxH3Mode(miniMaxH3ModeForSegment(segment));
      const availableExtras = refs.extra_subjects || [];
      if (miniMaxProject && refs.extras_enabled && availableExtras.length && ["reference_to_video", "video_to_video"].includes(extraMode)) {
        const extraPanel = document.createElement("div");
        extraPanel.style.cssText = "grid-column:1/-1;border:1px solid #164e63;border-radius:7px;background:#071a24;padding:9px;display:flex;flex-direction:column;gap:7px;";
        const extraTitle = document.createElement("div");
        extraTitle.textContent = "Extra Subjects — continuously present across scene cuts; visible whenever framing permits";
        extraTitle.style.cssText = "font-size:10px;color:#67e8f9;font-weight:900;text-transform:uppercase;";
        extraPanel.append(extraTitle);
        const mappedEntries = Array.isArray(refs.extra_scene_map?.[segment.id]) ? refs.extra_scene_map[segment.id] : [];
        const mappedById = new Map(mappedEntries.map((entry) => [String(entry?.extra_id || ""), entry]));
        const mappedExtras = availableExtras.filter((extra) => mappedById.has(extra.id));
        const representedPeople = mappedExtras.reduce((total, extra) => total + Math.max(1, Math.round(Number(extra.count) || 1)), 0);
        if (mappedExtras.length >= 6 || representedPeople >= 10) {
          const populationWarning = document.createElement("div");
          populationWarning.style.cssText = "border:1px solid #a16207;border-radius:6px;background:#2a1d06;color:#fde68a;padding:7px 9px;font-size:11px;line-height:1.4;";
          populationWarning.textContent = `Large ensemble: ${mappedExtras.length} defined extra${mappedExtras.length === 1 ? "" : "s"}, representing ${representedPeople} person${representedPeople === 1 ? "" : "s"}. This is allowed, but every defined extra must be visible by label in at least one appropriate shot. Verify wardrobe/location fit and include wide or medium coverage.`;
          extraPanel.append(populationWarning);
        }
        const imageExtras = availableExtras.filter((extra) => hasReferenceImage(extra.image || {}) && String(extra.description || "").trim());
        const textExtras = availableExtras.filter((extra) => !hasReferenceImage(extra.image || {}) || !String(extra.description || "").trim());
        if (imageExtras.length) {
          const selectedImageIds = imageExtras.filter((extra) => mappedById.has(extra.id)).map((extra) => extra.id);
          const imagePickerTitle = document.createElement("div");
          imagePickerTitle.textContent = "Choose image-backed extras";
          imagePickerTitle.style.cssText = "font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;margin-top:2px;";
          const imagePreview = markVisualPicker(makeMappingPreview(
            imageExtras.filter((extra) => mappedById.has(extra.id)).map((extra) => ({ image: extra.image || {}, label: extra.title || "Extra" })),
            "Choose extras by image"
          ));
          imagePreview.style.cursor = "pointer";
          imagePreview.onclick = () => openVisualMappingPicker({
            title: `${sceneDisplayName(segment, index)} — Extra Subjects`,
            items: imageExtras.map((extra) => ({ id: extra.id, label: `${extra.title || "Extra"}${Number(extra.count || 1) > 1 ? ` (${extra.count})` : ""}`, image: extra.image || {} })),
            selectedIds: selectedImageIds,
            multiple: true,
            onApply: (ids) => {
              const selected = new Set(ids.map(String));
              const current = new Map((Array.isArray(refs.extra_scene_map?.[segment.id]) ? refs.extra_scene_map[segment.id] : []).map((entry) => [String(entry?.extra_id || ""), entry]));
              for (const extra of imageExtras) {
                if (selected.has(extra.id)) {
                  const existing = current.get(extra.id);
                  const existingInteraction = existing?.interaction || "background";
                  current.set(extra.id, {
                    extra_id: extra.id,
                    interaction: segment.no_character_present && !["background", "background_dancing"].includes(existingInteraction)
                      ? "background"
                      : existingInteraction,
                  });
                } else {
                  current.delete(extra.id);
                }
              }
              const next = Array.from(current.values());
              if (next.length) refs.extra_scene_map[segment.id] = next;
              else delete refs.extra_scene_map[segment.id];
              renderMapping();
            },
          });
          extraPanel.append(imagePickerTitle, imagePreview);
        }
        const extrasToRender = [
          ...imageExtras.filter((extra) => mappedById.has(extra.id)),
          ...textExtras,
        ];
        if (textExtras.length) {
          const fallbackTitle = document.createElement("div");
          fallbackTitle.textContent = imageExtras.length ? "Extras without images" : "Choose extras";
          fallbackTitle.style.cssText = "font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;margin-top:4px;";
          extraPanel.append(fallbackTitle);
        }
        for (const extra of extrasToRender) {
          const imageBacked = hasReferenceImage(extra.image || {});
          const line = document.createElement("div");
          line.style.cssText = `display:grid;grid-template-columns:${imageBacked ? "54px" : "auto"} minmax(150px,1fr) minmax(150px,220px);gap:8px;align-items:center;`;
          const checked = document.createElement("input");
          checked.type = "checkbox";
          checked.checked = mappedById.has(extra.id);
          const hasDescription = Boolean(String(extra.description || "").trim());
          checked.disabled = !hasDescription && !checked.checked;
          checked.title = hasDescription ? "" : "Add a locked appearance description before mapping this extra.";
          let selector = checked;
          if (imageBacked) {
            const thumbnail = document.createElement("img");
            thumbnail.src = imageSrc(extra.image || {});
            thumbnail.alt = extra.title || "Extra";
            thumbnail.title = "Selected image-backed extra";
            thumbnail.style.cssText = "width:50px;height:50px;object-fit:cover;border:1px solid #0891b2;border-radius:6px;background:#020617;";
            selector = thumbnail;
          }
          const name = document.createElement("div");
          name.textContent = `${extra.title || "Extra"}${Number(extra.count || 1) > 1 ? ` (${extra.count})` : ""}${hasDescription ? "" : " — description required"}`;
          name.style.cssText = "font-size:12px;color:#e0f2fe;";
          const interaction = makeSelect(["background", "background_dancing", "alongside", "dancing_with", "direct"], mappedById.get(extra.id)?.interaction || "background");
          interaction.options[0].textContent = "Background only";
          interaction.options[1].textContent = "Background dancing";
          interaction.options[2].textContent = "Dancing alongside";
          interaction.options[3].textContent = "Dancing with";
          interaction.options[4].textContent = "Direct interaction";
          if (segment.no_character_present) {
            if (!["background", "background_dancing"].includes(interaction.value)) interaction.value = "background";
            interaction.options[2].disabled = true;
            interaction.options[3].disabled = true;
            interaction.options[4].disabled = true;
            interaction.title = "This scene has no main subject, so extras may remain ambient or dance in the background, but cannot dance alongside, dance with, or directly interact with a main subject.";
          }
          interaction.disabled = !checked.checked;
          const saveExtraEntry = () => {
            const current = new Map((Array.isArray(refs.extra_scene_map?.[segment.id]) ? refs.extra_scene_map[segment.id] : []).map((entry) => [String(entry?.extra_id || ""), entry]));
            if (checked.checked) current.set(extra.id, { extra_id: extra.id, interaction: segment.no_character_present && !["background", "background_dancing"].includes(interaction.value) ? "background" : interaction.value });
            else current.delete(extra.id);
            const next = Array.from(current.values());
            if (next.length) refs.extra_scene_map[segment.id] = next;
            else delete refs.extra_scene_map[segment.id];
          };
          checked.onchange = () => { interaction.disabled = !checked.checked; saveExtraEntry(); };
          interaction.onchange = saveExtraEntry;
          line.append(selector, name, interaction);
          extraPanel.append(line);
        }
        row.append(extraPanel);
      }
      const advanced = makeButton("Advanced");
      advanced.title = "Open Review Lines + Map Performers focused on this scene.";
      advanced.style.cssText = "align-self:center;padding:8px 7px;min-width:0;color:#67e8f9;";
      advanced.onclick = () => launchAdvancedLineMapping(segment);
      row.append(advanced);
      mappingList.append(row);
    });
  }

  return { exportSceneMappingContextForGpt, openImportGptSceneMapDialog, renderMapping };
}
