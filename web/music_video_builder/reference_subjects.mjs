import { postJson } from "./comfy_api.mjs";
import { makeButton, makeCheckbox, makeField, makeInput, makeSelect, normalizeVideoType, toast } from "./controls.mjs";
import { renameSubjectInDescription } from "./format.mjs";
import { sceneConceptPromptText } from "./image_prompts.mjs";
import { hasReferenceImage } from "./llm_runner.mjs";
import { MINIMAX_H3_VOICE_PRESETS, normalizeMiniMaxH3Pipeline, normalizeMiniMaxH3Voice } from "./minimax_h3.mjs";
import { isNoLipSyncSingerChoice, loadPromptJsonFromPath } from "./prompt_text.mjs";
import { isExtraSubjectReference, subjectExtraTargetId } from "./reference_data.mjs";
import { buildRefmodPicker } from "./refmod_card.mjs";

function subjectGenderHint(subject) {
  const text = `${subject?.name || ""} ${subject?.description || ""}`.toLowerCase();
  if (/\b(female|woman|girl|lady)\b/.test(text)) return "female";
  if (/\b(male|man|boy|guy)\b/.test(text)) return "male";
  return "";
}

export function createReferenceSubjects({
  activeSegment, allEditableSegments, createFluxReferenceWithZImage, createProgressWindow, currentVideoMode,
  describeSingleReferenceWithGemma, extractSubjects, extrasDisabledNote, extrasList, gemmaRunnerLine,
  i2vTextGemmaModelSelect, imageTargetFor, locationKey, logicalReferenceSubjects, logicalSubjectIdsForScene,
  makeReferenceTypeSelect, miniMaxH3SettingsForSegment, miniMaxProject, projectInput, projectSceneNotesPath,
  referenceImagesEnabled, refs, renderAll, renderDrop, renderMapping, renderSubjectThumbnailStrip, state,
  subjectButtons, subjectCountInput, subjectDescription, subjectDrop, subjectNameInput, subjectSceneInput,
  subjectSourceSelect, subjectTypeSelect, subjectsList, syncMiniMaxReferenceButtons, syncPrimarySubjectImage,
  t2iTextGemmaModelSelect, textGemmaRunnerPayload, uploadFor, useSubject, wireDrop,
}) {
  const showMiniMaxNativeVoiceControls = miniMaxProject
    && normalizeVideoType(state.videoType) === "speaking"
    && miniMaxH3SettingsForSegment(activeSegment()).audio_mode === "built_in_audio";
  const allowExtraSubjectReferences = referenceImagesEnabled && !miniMaxProject && currentVideoMode() === "rtv";
  // In the RefMod pipeline each card picks a saved RefMod instead of an image.
  const refmodPipeline = miniMaxProject && normalizeMiniMaxH3Pipeline(state.miniMaxH3Settings?.pipeline) === "refmod";

  function ensureSubjectCount(options = {}) {
    let desiredCount = Math.max(0, Math.min(12, Number(subjectCountInput.value || refs.subject_count || refs.subjects.length || 0)));
    if (!options.allowTrim && refs.subjects.length > desiredCount) {
      desiredCount = Math.min(12, refs.subjects.length);
      subjectCountInput.value = String(desiredCount);
    }
    refs.subject_count = desiredCount;
    while (refs.subjects.length < refs.subject_count) {
      refs.subjects.push({
        id: `subj_${Date.now()}_${refs.subjects.length}_${Math.floor(Math.random() * 10000)}`,
        name: `Character ${refs.subjects.length + 1}`,
        description: "",
        reference_type: "character",
        extra_reference_for: "",
        extra_reference_note: "",
        minimax_voice: normalizeMiniMaxH3Voice(),
        image: { path: "", data: "", name: "" },
      });
    }
    refs.subjects = refs.subjects.slice(0, refs.subject_count);
    if (refs.subject_count === 1 && refs.subjects[0]) {
      refs.subjects[0].name = subjectNameInput.value || refs.subjects[0].name || "Character 1";
      refs.subjects[0].reference_type = subjectTypeSelect.value || refs.subjects[0].reference_type || "character";
      refs.subjects[0].description = subjectDescription.value || refs.subjects[0].description || "";
      refs.subjects[0].image = hasReferenceImage(refs.subject.image || {}) ? refs.subject.image : (refs.subjects[0].image || { path: "", data: "", name: "" });
    }
  }
  function createSubject(name = "", description = "") {
    const subject = {
      id: `subj_${Date.now()}_${Math.floor(Math.random() * 10000)}`,
      name: String(name || `Character ${refs.subjects.length + 1}`).trim(),
      description: String(description || "").trim(),
      reference_type: "character",
      extra_reference_for: "",
      extra_reference_note: "",
      minimax_voice: normalizeMiniMaxH3Voice(),
      source: refmodPipeline ? "refmod" : "image",
      image: { path: "", data: "", name: "" },
    };
    refs.subjects.push(subject);
    refs.subject_count = refs.subjects.length;
    subjectCountInput.value = String(refs.subject_count);
    return subject;
  }
  function syncSingleSubjectInputsFromFirstSubject() {
    const first = refs.subjects?.[0] || null;
    if (!first) {
      subjectNameInput.value = "";
      subjectTypeSelect.value = "character";
      subjectDescription.value = "";
      refs.subject = {
        name: "",
        description: "",
        reference_type: "character",
        minimax_voice: normalizeMiniMaxH3Voice(),
        image: { path: "", data: "", name: "" },
      };
      return;
    }
    subjectNameInput.value = first.name || "Character 1";
    subjectTypeSelect.value = first.reference_type || "character";
    subjectDescription.value = first.description || "";
    refs.subject = {
      name: first.name || "Character 1",
      description: first.description || "",
      reference_type: first.reference_type || "character",
      minimax_voice: normalizeMiniMaxH3Voice(first.minimax_voice),
      image: { ...(first.image || { path: "", data: "", name: "" }) },
    };
  }
  function noteSubjectHints(note) {
    const text = String(note || "").toLowerCase();
    const hints = [];
    if (/\b(female|woman|girl|lady)\b/.test(text)) hints.push("female");
    if (/\b(male|man|boy|guy)\b/.test(text)) hints.push("male");
    for (const subject of logicalReferenceSubjects(refs)) {
      const name = String(subject?.name || "").trim().toLowerCase();
      if (name && new RegExp(`\\b${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(text)) {
        hints.push(subject.id);
      }
    }
    return Array.from(new Set(hints));
  }
  function autoMapSubjectsFromDirectorNotes() {
    if (!refs.subject_scene_map || typeof refs.subject_scene_map !== "object") refs.subject_scene_map = {};
    const subjectByGender = {};
    for (const subject of logicalReferenceSubjects(refs)) {
      const gender = subjectGenderHint(subject);
      if (gender && !subjectByGender[gender]) subjectByGender[gender] = subject.id;
    }
    let mapped = 0;
    for (const segment of allEditableSegments()) {
      const hints = noteSubjectHints(segment.timeline_note);
      const ids = hints
        .map((hint) => subjectByGender[hint] || hint)
        .filter((id) => logicalReferenceSubjects(refs).some((subject) => subject.id === id));
      const uniqueIds = Array.from(new Set(ids));
      if (uniqueIds.length) {
        refs.subject_scene_map[segment.id] = uniqueIds;
        mapped += 1;
      }
    }
    return mapped;
  }
  function subjectIdsFromLyricSingers(segment) {
    if (!segment) return [];
    const singers = Array.isArray(segment.lyric_singers) ? segment.lyric_singers.map((value) => String(value || "").trim()).filter(Boolean) : [];
    if (!singers.length) return [];
    const selected = [];
    const addSubject = (subject) => {
      if (subject?.id && !selected.includes(subject.id)) selected.push(subject.id);
    };
    const normalized = (value) => String(value || "").toLowerCase().replace(/[^\p{L}\p{N}\s]+/gu, " ").replace(/\s+/g, " ").trim();
    const subjectByGender = {};
    for (const subject of logicalReferenceSubjects(refs)) {
      const gender = subjectGenderHint(subject);
      if (gender && !subjectByGender[gender]) subjectByGender[gender] = subject;
    }
    for (const singer of singers) {
      const cleanSinger = normalized(singer);
      if (!cleanSinger || isNoLipSyncSingerChoice(singer)) continue;
      if (/\b(group|all visible|all singers|duet|both)\b/i.test(singer)) {
        logicalReferenceSubjects(refs).forEach(addSubject);
        continue;
      }
      const byId = logicalReferenceSubjects(refs).find((subject) => subject.id === singer);
      if (byId) {
        addSubject(byId);
        continue;
      }
      const byName = logicalReferenceSubjects(refs).find((subject) => {
        const name = normalized(subject.name);
        return name && (cleanSinger === name || cleanSinger.includes(name) || name.includes(cleanSinger));
      });
      if (byName) {
        addSubject(byName);
        continue;
      }
      if (/\b(female|woman|girl|lady)\b/.test(cleanSinger) && subjectByGender.female) {
        addSubject(subjectByGender.female);
        continue;
      }
      if (/\b(male|man|boy|guy)\b/.test(cleanSinger) && subjectByGender.male) {
        addSubject(subjectByGender.male);
      }
    }
    return selected;
  }
  function autoMapSubjectsFromLyrics() {
    ensureSubjectCount();
    if (!refs.subjects.length) return 0;
    if (!refs.subject_scene_map || typeof refs.subject_scene_map !== "object") refs.subject_scene_map = {};
    let mapped = 0;
    for (const segment of allEditableSegments()) {
      const subjectIds = subjectIdsFromLyricSingers(segment);
      if (!subjectIds.length) continue;
      refs.subject_scene_map[segment.id] = subjectIds;
      mapped += 1;
    }
    return mapped;
  }
  const normalizeSceneNoteSubjectText = (value) => String(value || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
  function sceneNoteMentionsSubject(note, subject) {
    const needle = normalizeSceneNoteSubjectText(subject?.name || "");
    if (!needle) return false;
    const haystack = ` ${normalizeSceneNoteSubjectText(note)} `;
    return haystack.includes(` ${needle} `);
  }
  async function autoMapSubjectsFromSceneNotesJson() {
    ensureSubjectCount();
    if (!refs.subjects.length) return { mapped: 0, matches: 0, path: "" };
    const sceneNotesPath = projectSceneNotesPath();
    if (!sceneNotesPath) throw new Error("Create or load a project before mapping subjects from SceneNotes.json.");
    const notes = await loadPromptJsonFromPath(sceneNotesPath);
    if (!refs.subject_scene_map || typeof refs.subject_scene_map !== "object") refs.subject_scene_map = {};
    let mapped = 0;
    let matches = 0;
    const segments = allEditableSegments();
    for (let index = 0; index < segments.length && index < notes.length; index += 1) {
      const segment = segments[index];
      const note = String(notes[index] || "");
      const subjectIds = [];
      for (const subject of logicalReferenceSubjects(refs)) {
        if (sceneNoteMentionsSubject(note, subject) && subject.id && !subjectIds.includes(subject.id)) {
          subjectIds.push(subject.id);
        }
      }
      if (!subjectIds.length) continue;
      refs.subject_scene_map[segment.id] = subjectIds;
      mapped += 1;
      matches += subjectIds.length;
    }
    return { mapped, matches, path: sceneNotesPath };
  }

  function updateSubjectDrop() {
    return renderDrop(subjectDrop, refs.subject.image, "Drop subject reference image here");
  }
  const subjectKey = locationKey;
  function subjectByName(name) {
    return refs.subjects.find((item) => subjectKey(item.name) === subjectKey(name));
  }

  async function extractSubjectsWithGemma() {
    let progress = null;
    extractSubjects.disabled = true;
    extractSubjects.textContent = "Extracting...";
    progress = createProgressWindow("Extracting subjects", { zIndex: 100008 });
    const subjectSource = subjectSourceSelect.value || "prompts_and_director_notes";
    progress.set("Checking scene prompts, Director Notes, and character context...", 5);
    const scenes = allEditableSegments().map((segment, index) => ({
      id: segment.id,
      label: segment.label || `Scene ${index + 1}`,
      concept: subjectSource === "director_notes_only" ? "" : sceneConceptPromptText(segment),
      notes: subjectSource === "director_notes_only" ? "" : (segment.notes || ""),
      director_note: subjectSource === "prompts_only" ? "" : (segment.timeline_note || ""),
    }));
    const usableScenes = scenes.filter((scene) => String(scene.concept || scene.notes || scene.director_note || "").trim());
    if (!usableScenes.length) {
      const message = subjectSource === "director_notes_only"
        ? "Extract Subjects needs Director Notes first."
        : "Extract Subjects needs scene concept prompts, scene notes, or Director Notes first.";
      progress.set(`Error:\n${message}`, 100);
      toast(message, true);
      extractSubjects.disabled = false;
      extractSubjects.textContent = "Extract Subjects";
      return;
    }
    const modelFile = String(t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "").trim();
    if (!modelFile && !["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner)) {
      const message = "Choose a non-vision Gemma model first, or use LM Studio, LLM API, or your own server in LLM Runner.";
      progress.set(`Error:\n${message}`, 100);
      toast(message, true);
      extractSubjects.disabled = false;
      extractSubjects.textContent = "Extract Subjects";
      return;
    }
    try {
      progress.set(`Asking Gemma for reusable character references...\n${gemmaRunnerLine()}`, 15);
      const data = await postJson("/vrgdg/music_builder/flux_reference_extract_subjects", {
        ...textGemmaRunnerPayload(),
        model_file: modelFile,
        project_folder: projectInput.value || state.projectFolder || "",
        scenes: usableScenes,
        subject_source: subjectSource,
        subject_scene_text: subjectSceneInput.value || "",
        requested_count: Math.max(2, Number(subjectCountInput.value || refs.subject_count || 2)),
        existing_subjects: refs.subjects.map((item) => ({ name: item.name || "", description: item.description || "" })),
        unload_after: true,
      }, 10 * 60 * 1000);
      progress.set("Adding extracted subjects to the Reference Builder...", 78);
      let added = 0;
      let updated = 0;
      for (const item of data.subjects || []) {
        const name = String(item.name || "").trim();
        if (!name) continue;
        let subject = subjectByName(name);
        if (!subject) {
          subject = createSubject(name, item.description || "");
          added += 1;
        } else if (!String(subject.description || "").trim() && item.description) {
          subject.description = String(item.description || "");
          updated += 1;
        }
      }
      refs.subject_count = Math.max(2, refs.subjects.length, Number(subjectCountInput.value || 2));
      subjectCountInput.value = String(refs.subject_count);
      refs.use_subject_reference = true;
      useSubject.input.checked = true;
      const mapped = autoMapSubjectsFromDirectorNotes();
      renderAll();
      progress.set(`Extracted subjects ready.\nAdded: ${added}\nUpdated: ${updated}\nMapped from Director Notes: ${mapped}\n\nReview/edit characters, add images, then save.`, 100);
      progress.close(2600);
      toast(`Extracted subjects. Mapped ${mapped} scene${mapped === 1 ? "" : "s"} from Director Notes.`);
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      extractSubjects.disabled = false;
      extractSubjects.textContent = "Extract Subjects";
    }
  }

  function renderExtras() {
    if (!miniMaxProject) return;
    extrasDisabledNote.style.display = refs.extras_enabled ? "none" : "";
    extrasList.style.display = refs.extras_enabled ? "flex" : "none";
    extrasList.replaceChildren();
    if (!refs.extra_subjects.length) {
      const empty = document.createElement("div");
      empty.textContent = "No extra subjects yet. Add one, enter a locked appearance description, and map it to the scenes where it remains present across cuts.";
      empty.style.cssText = "font-size:12px;color:#94a3b8;border:1px dashed #0891b2;border-radius:7px;padding:16px;text-align:center;background:#061620;";
      extrasList.append(empty);
      return;
    }
    refs.extra_subjects.forEach((extra, index) => {
      const row = document.createElement("div");
      row.style.cssText = "border:1px solid #334155;border-radius:7px;background:linear-gradient(135deg,#0f172a,#111827);padding:10px;display:grid;grid-template-columns:minmax(170px,.7fr) minmax(240px,1.4fr);gap:8px;";
      const fields = document.createElement("div");
      fields.style.cssText = "display:flex;flex-direction:column;gap:8px;";
      const title = makeInput(extra.title || `Extra ${index + 1}`);
      title.placeholder = "Backup dancers, man in black hoodie...";
      const style = makeInput(extra.style || "");
      style.placeholder = "Pop, rock, jazz, custom...";
      const count = makeInput(String(extra.count || 1), "number");
      count.min = "1";
      count.max = "100";
      count.step = "1";
      const sendToMiniMax = makeCheckbox("Send this image to MiniMax when mapped", Boolean(extra.send_to_minimax));
      const referenceImageType = makeSelect([
        { value: "single", label: "Single character image" },
        { value: "multi_view", label: "Character reference sheet" },
      ], extra.reference_image_type || "single");
      fields.append(makeField("Title", title), makeField("Style / genre hint", style), makeField("How many", count), sendToMiniMax.wrapper, makeField("Reference image type", referenceImageType));
      const detail = document.createElement("div");
      detail.style.cssText = "display:flex;flex-direction:column;gap:8px;";
      const description = document.createElement("textarea");
      description.value = extra.description || "";
      description.placeholder = "Required locked hair, wardrobe, colors, shoes, and accessories...";
      description.style.cssText = "min-height:92px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;";
      const drop = document.createElement("div");
      drop.style.cssText = `${subjectDrop.style.cssText};min-height:110px;height:110px;`;
      const target = imageTargetFor(extra, "image", "extra");
      wireDrop(drop, target);
      renderDrop(drop, extra.image || {}, extra.send_to_minimax
        ? "MiniMax identity reference when this extra is mapped"
        : "Optional photo — description AI only; not sent to MiniMax");
      const referenceWarning = document.createElement("div");
      const missingCheckedImage = extra.send_to_minimax && !hasReferenceImage(extra.image || {});
      referenceWarning.textContent = missingCheckedImage
        ? "MiniMax image use is checked, but no image is loaded. This extra will fall back to a text-only subject until an image is added."
        : extra.send_to_minimax && extra.reference_image_type === "multi_view"
          ? "Reference-sheet mode treats every panel as the same person. Panels may use any framing or pose and are not separate people or scenes."
          : extra.send_to_minimax
            ? "This image will use one of the scene's nine MiniMax reference slots whenever this extra is mapped."
            : "The image is used only to help describe the extra unless MiniMax image use is checked.";
      referenceWarning.style.cssText = `font-size:11px;line-height:1.4;color:${missingCheckedImage ? "#fca5a5" : "#94a3b8"};`;
      const buttons = document.createElement("div");
      buttons.style.cssText = "display:flex;gap:7px;flex-wrap:wrap;";
      const describe = makeButton("Gemma Describe", "primary");
      const upload = makeButton("Upload Photo", "primary");
      const clear = makeButton("Clear Photo");
      const remove = makeButton("Remove");
      describe.disabled = !hasReferenceImage(extra.image || {});
      describe.onclick = async () => { await describeSingleReferenceWithGemma(extra, "extra", extra.title || `Extra ${index + 1}`); renderExtras(); };
      upload.onclick = () => uploadFor(target);
      clear.onclick = () => { extra.image = { path: "", data: "", name: "", preview_url: "" }; renderExtras(); };
      remove.onclick = () => {
        refs.extra_subjects.splice(index, 1);
        for (const [sceneId, entries] of Object.entries(refs.extra_scene_map || {})) {
          const remaining = (Array.isArray(entries) ? entries : []).filter((entry) => String(entry?.extra_id || "") !== extra.id);
          if (remaining.length) refs.extra_scene_map[sceneId] = remaining;
          else delete refs.extra_scene_map[sceneId];
        }
        renderAll();
      };
      buttons.append(describe, upload, clear, remove);
      title.oninput = () => { extra.title = title.value; };
      style.oninput = () => { extra.style = style.value; };
      count.onchange = () => { extra.count = Math.max(1, Math.min(100, Math.round(Number(count.value) || 1))); count.value = String(extra.count); };
      sendToMiniMax.input.onchange = () => {
        extra.send_to_minimax = Boolean(sendToMiniMax.input.checked);
        renderExtras();
        syncMiniMaxReferenceButtons();
      };
      referenceImageType.onchange = () => { extra.reference_image_type = referenceImageType.value; renderExtras(); };
      description.oninput = () => { extra.description = description.value; };
      detail.append(makeField("Locked appearance description (required before mapping)", description), drop, referenceWarning, buttons);
      row.append(fields, detail);
      extrasList.append(row);
    });
  }

  function renderSubjects() {
    if (refs.subjects.length || Number(refs.subject_count || 0) > 0) ensureSubjectCount();
    else refs.subject_count = 0;
    syncPrimarySubjectImage();
    const multi = true;
    extractSubjects.style.display = "";
    subjectNameInput.parentElement.style.display = "none";
    subjectTypeSelect.parentElement.style.display = "none";
    subjectDescription.parentElement.style.display = "none";
    subjectDrop.style.display = "none";
    subjectButtons.style.display = "none";
    subjectsList.style.display = "flex";
    subjectsList.style.maxHeight = "min(62vh, 720px)";
    if (!multi) {
      syncSingleSubjectInputsFromFirstSubject();
      updateSubjectDrop();
      return;
    }
    subjectsList.innerHTML = "";
    if (!refs.subjects.length) {
      const empty = document.createElement("div");
      empty.innerHTML = referenceImagesEnabled
        ? `<strong style="color:#cffafe;">No subject references yet.</strong><br><span>Drop one or more subject images here, upload images, import subjects, or add a subject row.</span>`
        : `<strong style="color:#cffafe;">No subject references yet.</strong><br><span>Add, import, or extract reusable subject descriptions for scene mapping.</span>`;
      empty.style.cssText = "font-size:12px;color:#94a3b8;border:1px dashed #0891b2;border-radius:7px;padding:16px;text-align:center;background:#061620;";
      if (referenceImagesEnabled) wireDrop(empty, { kind: "subject", bulk: true });
      subjectsList.append(empty);
      return;
    }
    let draggedSubjectIndex = -1;
    const moveSubjectRow = (fromIndex, toIndex) => {
      if (fromIndex === toIndex || fromIndex < 0 || toIndex < 0 || fromIndex >= refs.subjects.length || toIndex >= refs.subjects.length) return;
      const [moved] = refs.subjects.splice(fromIndex, 1);
      refs.subjects.splice(toIndex, 0, moved);
      syncSingleSubjectInputsFromFirstSubject();
      renderAll();
    };
    refs.subjects.forEach((subject, index) => {
      const row = document.createElement("div");
      row.draggable = false;
      row.dataset.subjectIndex = String(index);
      row.style.cssText = "border:1px solid #334155;border-radius:7px;background:linear-gradient(135deg,#0f172a,#111827);padding:10px;border-radius:7px;display:flex;flex-direction:column;gap:8px;";
      row.addEventListener("dragstart", (event) => {
        draggedSubjectIndex = index;
        row.style.opacity = "0.55";
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", String(index));
      });
      row.addEventListener("dragend", () => {
        draggedSubjectIndex = -1;
        row.style.opacity = "";
      });
      row.addEventListener("dragover", (event) => {
        if (draggedSubjectIndex < 0 || draggedSubjectIndex === index) return;
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
        const fromIndex = Number(event.dataTransfer.getData("text/plain") || draggedSubjectIndex);
        moveSubjectRow(fromIndex, index);
      });
      const refmodCard = refmodPipeline && subject.source === "refmod";
      const showImages = referenceImagesEnabled && !refmodCard;
      const name = makeInput(subject.name || `Character ${index + 1}`);
      const typeSelect = makeReferenceTypeSelect(subject.reference_type || "character");
      const description = document.createElement("textarea");
      description.value = subject.description || "";
      description.placeholder = "Reference details. For props/objects, describe shape, material, color, markings, and important visual features.";
      description.style.cssText = "min-height:76px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;";
      const sameSubjectWrap = document.createElement("div");
      sameSubjectWrap.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:9px;display:grid;grid-template-columns:auto 1fr;gap:8px 10px;align-items:center;";
      const sameSubjectCheck = document.createElement("input");
      sameSubjectCheck.type = "checkbox";
      sameSubjectCheck.checked = Boolean(subjectExtraTargetId(subject));
      const sameSubjectLabel = document.createElement("div");
      sameSubjectLabel.innerHTML = `<div style="font-size:12px;font-weight:900;color:#e0f2fe;">This is another reference image for</div><div style="font-size:11px;color:#94a3b8;margin-top:2px;">Uses this RTV slot as extra visual guidance for one subject. It will not be prompted as a second character.</div>`;
      const sameSubjectSelect = document.createElement("select");
      sameSubjectSelect.style.cssText = "grid-column:2;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:7px;font-size:12px;";
      sameSubjectSelect.append(new Option("Choose subject...", ""));
      refs.subjects
        .filter((candidate) => candidate.id !== subject.id && !isExtraSubjectReference(candidate))
        .forEach((candidate) => sameSubjectSelect.append(new Option(candidate.name || "Character", candidate.id)));
      sameSubjectSelect.value = subjectExtraTargetId(subject);
      sameSubjectSelect.disabled = !sameSubjectCheck.checked;
      sameSubjectWrap.append(sameSubjectCheck, sameSubjectLabel, sameSubjectSelect);
      const usedBy = allEditableSegments()
        .map((segment, sceneIndex) => logicalSubjectIdsForScene(refs, segment).includes(subjectExtraTargetId(subject) || subject.id) ? `Scene ${sceneIndex + 1}` : "")
        .filter(Boolean)
        .join(", ") || "Not mapped yet";
      const used = document.createElement("div");
      const targetSubject = refs.subjects.find((item) => item.id === subjectExtraTargetId(subject));
      const extraCount = refs.subjects.filter((item) => subjectExtraTargetId(item) === subject.id).length;
      used.textContent = targetSubject
        ? `Extra ref for: ${targetSubject.name || "Subject"} | Used by: ${usedBy}`
        : `Used by: ${usedBy}${referenceImagesEnabled && extraCount ? ` | Includes ${extraCount} extra ref image${extraCount === 1 ? "" : "s"}` : ""}`;
      used.style.cssText = "font-size:11px;color:#a5f3fc;";
      const drop = document.createElement("div");
      drop.style.cssText = `${subjectDrop.style.cssText};min-height:132px;height:132px;`;
      drop.style.display = "flex";
      const target = imageTargetFor(subject, "image", "subject");
      wireDrop(drop, target);
      const buttons = document.createElement("div");
      buttons.style.cssText = "display:grid;grid-template-columns:1fr;gap:6px;align-self:stretch;";
      const createZImage = makeButton("Generate Subject", "primary");
      createZImage.title = "Choose ZImage or Krea2 + ZImage enhancer for this subject reference.";
      const describeImage = makeButton("Gemma Describe", "primary");
      const upload = makeButton("Upload", "primary");
      const clear = makeButton("Clear");
      const remove = makeButton("Remove");
      createZImage.onclick = () => createFluxReferenceWithZImage("subject", subject, `${subject.name || ""}\n${subject.description || ""}`, subject.name || `character_${index + 1}`);
      describeImage.onclick = () => describeSingleReferenceWithGemma(subject, "subject", subject.name || `Character ${index + 1}`);
      upload.onclick = () => uploadFor(target);
      clear.onclick = () => {
        subject.image = { path: "", data: "", name: "" };
        renderAll();
      };
      remove.onclick = () => {
        refs.subjects.splice(index, 1);
        refs.subject_count = refs.subjects.length;
        subjectCountInput.value = String(refs.subject_count);
        refs.subjects.forEach((item) => {
          if (subjectExtraTargetId(item) === subject.id) {
            item.extra_reference_for = "";
            item.extra_reference_note = "";
          }
        });
        for (const segment of allEditableSegments()) {
          refs.subject_scene_map[segment.id] = (refs.subject_scene_map?.[segment.id] || []).filter((id) => id !== subject.id);
        }
        if (refs.subjects.length <= 1) syncSingleSubjectInputsFromFirstSubject();
        renderAll();
      };
      let previousSubjectName = String(subject.name || "");
      name.addEventListener("change", () => {
        const nextDescription = renameSubjectInDescription(subject.description, previousSubjectName, subject.name);
        if (nextDescription !== String(subject.description || "") && window.confirm(`Update "${previousSubjectName}" to "${subject.name}" in this subject's description? Existing scene prompts and lyrics will stay as written.`)) {
          subject.description = nextDescription;
          description.value = nextDescription;
          if (index === 0) {
            refs.subject.description = nextDescription;
            subjectDescription.value = nextDescription;
          }
        }
        previousSubjectName = String(subject.name || "");
      });
      name.addEventListener("input", () => {
        subject.name = name.value;
        if (index === 0) {
          refs.subject.name = name.value;
          subjectNameInput.value = name.value;
        }
        renderMapping();
      });
      typeSelect.addEventListener("change", () => {
        subject.reference_type = typeSelect.value || "character";
        if (index === 0) {
          refs.subject.reference_type = subject.reference_type;
          subjectTypeSelect.value = subject.reference_type;
        }
        if (refmodPipeline) renderAll();
        else renderMapping();
      });
      sameSubjectCheck.addEventListener("change", () => {
        if (sameSubjectCheck.checked) {
          sameSubjectSelect.disabled = false;
          subject.extra_reference_for = sameSubjectSelect.value || sameSubjectSelect.options[1]?.value || "";
          sameSubjectSelect.value = subject.extra_reference_for;
          refs.subjects.forEach((item) => {
            if (subjectExtraTargetId(item) === subject.id) {
              item.extra_reference_for = "";
              item.extra_reference_note = "";
            }
          });
        } else {
          subject.extra_reference_for = "";
          subject.extra_reference_note = "";
          sameSubjectSelect.value = "";
          sameSubjectSelect.disabled = true;
        }
        for (const segment of allEditableSegments()) {
          refs.subject_scene_map[segment.id] = logicalSubjectIdsForScene(refs, segment);
        }
        renderAll();
      });
      sameSubjectSelect.addEventListener("change", () => {
        subject.extra_reference_for = sameSubjectSelect.value || "";
        refs.subjects.forEach((item) => {
          if (subjectExtraTargetId(item) === subject.id) {
            item.extra_reference_for = "";
            item.extra_reference_note = "";
          }
        });
        for (const segment of allEditableSegments()) {
          refs.subject_scene_map[segment.id] = logicalSubjectIdsForScene(refs, segment);
        }
        renderAll();
      });
      description.addEventListener("input", () => {
        subject.description = description.value;
        if (index === 0) {
          refs.subject.description = description.value;
          subjectDescription.value = description.value;
        }
      });
      if (showImages) buttons.append(createZImage, describeImage, upload, clear);
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
      const nameField = makeField("Reference label", name);
      const typeField = makeField("Reference type", typeSelect);
      const descField = makeField("Description", description);
      const imageWrap = document.createElement("div");
      imageWrap.style.cssText = "display:grid;grid-template-columns:122px minmax(110px,1fr);gap:8px;align-items:stretch;";
      const thumbnailStrip = renderSubjectThumbnailStrip(subject, target);
      renderDrop(drop, thumbnailStrip ? { path: "", data: "", name: "" } : subject.image, thumbnailStrip ? "Drop image to replace" : "Drop subject reference image here");
      if (thumbnailStrip) {
        thumbnailStrip.style.margin = "0";
        thumbnailStrip.style.maxHeight = "132px";
        thumbnailStrip.style.overflow = "hidden";
        imageWrap.append(thumbnailStrip, drop);
      } else {
        imageWrap.style.gridTemplateColumns = "1fr";
        imageWrap.append(drop);
      }
      const mainRow = document.createElement("div");
      mainRow.style.cssText = showImages
        ? "display:grid;grid-template-columns:28px 34px minmax(150px,0.9fr) minmax(145px,0.75fr) minmax(240px,1.2fr) minmax(230px,1fr) 148px;gap:10px;align-items:center;min-width:1060px;"
        : "display:grid;grid-template-columns:28px 34px minmax(170px,.85fr) minmax(150px,.7fr) minmax(320px,1.5fr) 108px;gap:10px;align-items:center;min-width:820px;";
      mainRow.append(handle, number, nameField, typeField, descField);
      if (showImages) mainRow.append(imageWrap);
      mainRow.append(buttons);
      const metaRow = document.createElement("div");
      metaRow.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between;";
      metaRow.append(used);
      if (allowExtraSubjectReferences && refs.subjects.length > 1) metaRow.append(sameSubjectWrap);
      row.append(mainRow, metaRow);
      if (refmodPipeline) {
        row.append(buildRefmodPicker({
          item: subject,
          kind: "subject",
          characters: refs.subjects.filter((card) => (card.reference_type || "character") === "character"),
          onChange: (rebuild) => { if (rebuild) renderAll(); else renderMapping(); },
        }));
      }
      if (showMiniMaxNativeVoiceControls && subject.reference_type === "character" && !isExtraSubjectReference(subject)) {
        subject.minimax_voice = normalizeMiniMaxH3Voice(subject.minimax_voice);
        const voiceWrap = document.createElement("div");
        voiceWrap.style.cssText = "border:1px solid #155e75;border-radius:7px;background:#061620;padding:10px;display:flex;flex-direction:column;gap:8px;";
        const voiceHeading = document.createElement("div");
        voiceHeading.innerHTML = `<div style="font-size:12px;font-weight:900;color:#cffafe;">MiniMax built-in voice</div><div style="font-size:11px;color:#94a3b8;margin-top:2px;">Short Film + Built-in Audio only. The saved preset name and voice description are copied verbatim into every prompt where this character speaks.</div>`;
        const voicePreset = makeSelect(MINIMAX_H3_VOICE_PRESETS, subject.minimax_voice.preset_id);
        const voiceName = makeInput(subject.minimax_voice.preset_name || "");
        voiceName.placeholder = "Exact reusable preset name, e.g. MIDNIGHT ROSE";
        const voiceDescription = document.createElement("textarea");
        voiceDescription.placeholder = "Exact reusable voice description: adult voice range, accent, timbre, cadence, consonants, warmth, and microphone delivery.";
        voiceDescription.style.cssText = "min-height:76px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;line-height:1.4;";
        const voiceFields = document.createElement("div");
        voiceFields.style.cssText = "display:grid;grid-template-columns:minmax(180px,.7fr) minmax(320px,1.8fr);gap:8px;";
        const voiceNameField = makeField("Exact preset name", voiceName);
        const voiceDescriptionField = makeField("Exact voice description", voiceDescription);
        voiceFields.append(voiceNameField, voiceDescriptionField);
        const syncVoiceFields = () => {
          const preset = MINIMAX_H3_VOICE_PRESETS.find((item) => item.value === voicePreset.value) || MINIMAX_H3_VOICE_PRESETS[0];
          const custom = preset.value.endsWith("_custom");
          const normalizedVoice = normalizeMiniMaxH3Voice({
            preset_id: preset.value,
            preset_name: custom ? voiceName.value : preset.name,
            description: custom ? voiceDescription.value : preset.description,
          });
          subject.minimax_voice = normalizedVoice;
          voiceName.value = normalizedVoice.preset_name;
          voiceDescription.value = normalizedVoice.description;
          voiceName.disabled = !custom;
          voiceDescription.disabled = !custom;
          voiceFields.style.display = preset.value === "none" ? "none" : "grid";
        };
        voicePreset.addEventListener("change", () => {
          if (voicePreset.value.endsWith("_custom") && subject.minimax_voice.preset_id !== voicePreset.value) {
            voiceName.value = "";
            voiceDescription.value = "";
          }
          syncVoiceFields();
        });
        voiceName.addEventListener("input", () => {
          subject.minimax_voice = normalizeMiniMaxH3Voice({
            ...subject.minimax_voice,
            preset_id: voicePreset.value,
            preset_name: voiceName.value,
            description: voiceDescription.value,
          });
        });
        voiceDescription.addEventListener("input", () => {
          subject.minimax_voice = normalizeMiniMaxH3Voice({
            ...subject.minimax_voice,
            preset_id: voicePreset.value,
            preset_name: voiceName.value,
            description: voiceDescription.value,
          });
        });
        syncVoiceFields();
        voiceWrap.append(voiceHeading, makeField("Voice preset", voicePreset), voiceFields);
        row.append(voiceWrap);
      }
      subjectsList.append(row);
    });
  }

  return {
    autoMapSubjectsFromLyrics, autoMapSubjectsFromSceneNotesJson, createSubject, ensureSubjectCount,
    extractSubjectsWithGemma, renderExtras, renderSubjects, syncSingleSubjectInputsFromFirstSubject,
  };
}
