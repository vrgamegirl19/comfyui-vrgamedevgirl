import { postJson } from "./api.mjs";
import { createToast, escapeHtml, makeButton, makeInput, makeTextarea, tagsHtml } from "./controls.mjs";
import { normalizeStoryboardSpeakerAssignments } from "./scenes.mjs";
import { storyboardSceneSupportsVideoStyle } from "./video_style.mjs";

export function makeStoryboardImageUrl(path) {
  return `/vrgdg/video_editor/image?path=${encodeURIComponent(path)}&rand=${Date.now()}`;
}

export function storyboardReferenceImageSrc(image) {
  if (!image || typeof image !== "object") return "";
  const data = String(image.data || "").trim();
  if (data) return data.startsWith("data:") ? data : `data:image/png;base64,${data}`;
  const path = String(image.path || "").trim();
  return path ? makeStoryboardImageUrl(path) : "";
}

export function normalizeReferenceImage(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const image = source.image && typeof source.image === "object" ? source.image : source;
  const hasTopLevelImage = Boolean(source.path || source.data || source.image_path || source.imagePath || source.image_data || source.imageData);
  return {
    path: String(image.path || source.image_path || source.imagePath || source.path || "").trim(),
    data: String(image.data || source.image_data || source.imageData || source.data || "").trim(),
    name: String(image.name || source.image_name || source.imageName || (hasTopLevelImage ? source.name : "") || "").trim(),
  };
}

function mergeReferenceImages(existing = {}, incoming = {}) {
  const left = normalizeReferenceImage(existing);
  const right = normalizeReferenceImage(incoming);
  return {
    path: right.path || left.path,
    data: right.data || left.data,
    name: right.name || left.name,
  };
}

export function storyboardSubjectNamesFromRefs(subjectRefs = []) {
  return Array.from(new Set(
    (Array.isArray(subjectRefs) ? subjectRefs : [])
      .map((subject) => String(subject?.name || "").trim())
      .filter(Boolean)
  ));
}

export function storyboardReferenceId(prefix, name = "") {
  const slug = String(name || prefix || "reference")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48) || prefix;
  return `${prefix}_story_${Date.now()}_${slug}`;
}

export function readStoryboardImageFile(file) {
  return new Promise((resolve, reject) => {
    if (!file || !String(file.type || "").startsWith("image/")) {
      reject(new Error("Choose an image file."));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Could not read that image file."));
    reader.readAsDataURL(file);
  });
}

export function referenceChipHtml(ref, fallbackLabel = "Reference") {
  const image = storyboardReferenceImageSrc(ref?.image);
  const label = String(ref?.name || fallbackLabel || "Reference").trim();
  const thumb = image
    ? `<span style="width:34px;height:34px;border-radius:6px;border:1px solid #334155;background:#0f172a url('${escapeHtml(image)}') center/cover no-repeat;flex:0 0 auto;"></span>`
    : `<span style="width:34px;height:34px;border-radius:6px;border:1px dashed #334155;background:#07111f;color:#67e8f9;display:grid;place-items:center;font-size:12px;flex:0 0 auto;">▣</span>`;
  return `<span title="${escapeHtml(label)}" style="display:inline-flex;align-items:center;gap:7px;max-width:190px;border:1px solid #334155;border-radius:7px;background:#0f172a;color:#e5e7eb;padding:4px 7px;margin:3px 3px 3px 0;vertical-align:middle;">${thumb}<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;font-weight:800;">${escapeHtml(label)}</span></span>`;
}

export function subjectRefsHtml(scene) {
  const refs = Array.isArray(scene.subject_refs) ? scene.subject_refs : [];
  if (refs.length) return refs.map((ref, index) => referenceChipHtml(ref, `Subject ${index + 1}`)).join("");
  return tagsHtml(scene.subjects);
}

export function settingRefHtml(scene) {
  if (scene.location_ref && typeof scene.location_ref === "object" && String(scene.location_ref.name || scene.location_ref.image?.path || scene.location_ref.image?.data || "").trim()) {
    return referenceChipHtml(scene.location_ref, scene.setting || "Location");
  }
  return escapeHtml(scene.setting || "-");
}

export function normalizeReferenceBuilderCatalog(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const mergeReferenceList = (items = []) => {
    const byKey = new Map();
    const keyFor = (item) => {
      const name = String(item.name || "").trim().toLowerCase().replace(/\s+/g, " ");
      return name || String(item.id || "").trim().toLowerCase();
    };
    for (const item of items) {
      const key = keyFor(item);
      if (!key) continue;
      const existing = byKey.get(key) || {};
      byKey.set(key, {
        ...existing,
        ...item,
        id: existing.id || item.id,
        name: existing.name || item.name,
        description: existing.description || item.description,
        trigger_phrase: existing.trigger_phrase || item.trigger_phrase,
        image: mergeReferenceImages(existing.image, item.image),
      });
    }
    return Array.from(byKey.values());
  };
  const subjects = mergeReferenceList(Array.isArray(source.subjects) ? source.subjects
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `subject_${index + 1}`),
      name: String(item.name || `Character ${index + 1}`),
      description: String(item.description || ""),
      minimax_voice: item.minimax_voice && typeof item.minimax_voice === "object" ? { ...item.minimax_voice } : {},
      trigger_phrase: String(item.trigger_phrase || item.trigger || item.Trigger || ""),
      trigger_position: String(item.trigger_position || item.triggerPosition || item.trigger_placement || "start") === "end" ? "end" : "start",
      extra_reference_for: String(item.extra_reference_for || item.extraReferenceFor || item.same_subject_as || item.sameSubjectAs || ""),
      image: normalizeReferenceImage(item),
    })).filter((item) => !item.extra_reference_for) : []);
  const locations = mergeReferenceList(Array.isArray(source.locations) ? source.locations
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `location_${index + 1}`),
      name: String(item.name || `Location ${index + 1}`),
      description: String(item.description || ""),
      trigger_phrase: String(item.trigger_phrase || item.trigger || item.Trigger || ""),
      trigger_position: String(item.trigger_position || item.triggerPosition || item.trigger_placement || "start") === "end" ? "end" : "start",
      image: normalizeReferenceImage(item),
    })) : []);
  return {
    subjects,
    locations: Boolean(source.locations_cleared || source.locationsCleared || source.clear_locations || source.clearLocations) ? [] : locations,
    locations_cleared: Boolean(source.locations_cleared || source.locationsCleared || source.clear_locations || source.clearLocations),
    trigger_position: String(source.trigger_position || source.triggerPosition || source.trigger_placement || "start") === "end" ? "end" : "start",
    subject_trigger_position: String(source.subject_trigger_position || source.subjectTriggerPosition || source.trigger_position || "start") === "end" ? "end" : "start",
    location_trigger_position: String(source.location_trigger_position || source.locationTriggerPosition || source.trigger_position || "start") === "end" ? "end" : "start",
  };
}

export function mergeReferenceBuilderCatalog(base = {}, incoming = {}) {
  const normalizedBase = normalizeReferenceBuilderCatalog(base);
  const normalizedIncoming = normalizeReferenceBuilderCatalog(incoming);
  const mergeList = (left, right) => {
    const byKey = new Map();
    const keyFor = (item) => {
      const name = String(item.name || "").trim().toLowerCase().replace(/\s+/g, " ");
      return name || String(item.id || "").trim().toLowerCase();
    };
    for (const item of left) {
      const key = keyFor(item);
      if (key) byKey.set(key, { ...item, image: { ...(item.image || {}) } });
    }
    for (const item of right) {
      const key = keyFor(item);
      if (!key) continue;
      const existing = byKey.get(key) || {};
      byKey.set(key, {
        ...existing,
        ...item,
        image: mergeReferenceImages(existing.image, item.image),
      });
    }
    return Array.from(byKey.values());
  };
  return {
    subjects: mergeList(normalizedBase.subjects, normalizedIncoming.subjects),
    locations: normalizedBase.locations_cleared ? [] : mergeList(normalizedBase.locations, normalizedIncoming.locations),
    locations_cleared: Boolean(normalizedBase.locations_cleared || normalizedIncoming.locations_cleared),
  };
}

export function chooseStoryboardImageFile() {
  return new Promise((resolve) => {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.style.display = "none";
  document.body.append(input);
  input.onchange = () => {
    const file = input.files?.[0] || null;
    input.remove();
    resolve(file);
  };
  input.click();
});
}

export function promptStoryboardReferenceDetails({ kind, file, defaultName = "", defaultDescription = "" } = {}) {
  return new Promise((resolve) => {
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100050;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:18px;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(620px,calc(100vw - 40px));border:1px solid #155e75;border-radius:12px;background:#0f172a;color:#e5e7eb;box-shadow:0 24px 80px rgba(0,0,0,.58);overflow:hidden;";
  const title = kind === "location" ? "Add Location Reference" : "Add Subject Reference";
  box.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 16px;background:#083344;border-bottom:1px solid #155e75;">
        <div>
          <div style="font-size:18px;font-weight:900;color:#cffafe;">${escapeHtml(title)}</div>
          <div style="font-size:12px;color:#cbd5e1;margin-top:3px;">Name and describe this image so Storyboard Builder and Reference Builder can both use it.</div>
        </div>
      </div>
    `;
  const body = document.createElement("div");
  body.style.cssText = "padding:16px;display:flex;flex-direction:column;gap:12px;";
  const name = makeInput(defaultName || String(file?.name || "").replace(/\.[^.]+$/, ""), kind === "location" ? "Location name" : "Subject name");
  const description = makeTextarea(defaultDescription, kind === "location" ? "Location description..." : "Subject description...", 5);
  const actions = document.createElement("div");
  actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
  const cancel = makeButton("Cancel");
  const save = makeButton("Use Image", "primary");
  actions.append(cancel, save);
  body.append(
    (() => {
      const preview = document.createElement("div");
      preview.style.cssText = "height:150px;border:1px dashed #155e75;border-radius:10px;background:#07111f center/contain no-repeat;";
      return preview;
    })(),
    (() => {
      const wrap = document.createElement("label");
      wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:900;color:#cbd5e1;";
      wrap.append("Name", name);
      return wrap;
    })(),
    (() => {
      const wrap = document.createElement("label");
      wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:900;color:#cbd5e1;";
      wrap.append("Description", description);
      return wrap;
    })(),
    actions,
  );
  box.append(body);
  backdrop.append(box);
  document.body.append(backdrop);
  const preview = body.firstChild;
  readStoryboardImageFile(file)
    .then((dataUrl) => {
      preview.style.backgroundImage = `url("${dataUrl}")`;
      save.onclick = () => {
        const cleanName = String(name.value || "").trim();
        if (!cleanName) {
          createToast("Give this reference a name first.", true);
          return;
        }
        backdrop.remove();
        resolve({
          name: cleanName,
          description: String(description.value || "").trim(),
          image: { path: "", data: dataUrl, name: String(file?.name || cleanName) },
        });
      };
    })
    .catch((error) => {
      backdrop.remove();
      createToast(String(error?.message || error), true);
      resolve(null);
    });
  cancel.onclick = () => {
    backdrop.remove();
    resolve(null);
  };
});
}

export function createStoryboardReferences({ renderTable, state }) {
  function absorbSceneReferencesIntoCatalog(scenes = []) {
    const refs = normalizeReferenceBuilderCatalog(state.referenceBuilder || {});
    const locationIds = new Set(refs.locations.map((location) => String(location.id || "")).filter(Boolean));
    const locationByName = new Map(
      refs.locations
        .map((location) => [String(location.name || "").trim().toLowerCase().replace(/\s+/g, " "), location])
        .filter(([name]) => Boolean(name)),
    );
    const subjectIds = new Set(refs.subjects.map((subject) => String(subject.id || "")).filter(Boolean));
    for (const scene of scenes || []) {
      let location = scene?.location_ref;
      if ((!location || typeof location !== "object") && String(scene?.setting || "").trim()) {
        location = {
          id: "",
          name: String(scene.setting || "").trim(),
          description: String(scene.setting || "").trim(),
          image: { path: "", data: "", name: "" },
        };
      }
    if (location && typeof location === "object" && String(location.id || location.name || location.description || "").trim()) {
      const locationNameKey = String(location.name || scene.setting || "").trim().toLowerCase().replace(/\s+/g, " ");
      const existingLocation = locationNameKey ? locationByName.get(locationNameKey) : null;
      const id = String(existingLocation?.id || location.id || `location_from_scene_${scene.scene_number || refs.locations.length + 1}`).trim();
      location.id = id;
      scene.location_ref = location;
      if (!locationIds.has(id)) {
        const addedLocation = {
            id,
            name: String(location.name || scene.setting || "Saved location"),
            description: String(location.description || ""),
            trigger_phrase: String(location.trigger_phrase || ""),
            trigger_position: String(location.trigger_position || "start") === "end" ? "end" : "start",
            image: normalizeReferenceImage(location),
          };
          refs.locations.push(addedLocation);
          locationIds.add(id);
          const addedNameKey = String(addedLocation.name || "").trim().toLowerCase().replace(/\s+/g, " ");
          if (addedNameKey) locationByName.set(addedNameKey, addedLocation);
        }
      }
      for (const subject of Array.isArray(scene?.subject_refs) ? scene.subject_refs : []) {
        if (!subject || typeof subject !== "object") continue;
        const id = String(subject.id || subject.name || "").trim();
        if (!id || subjectIds.has(id)) continue;
        refs.subjects.push({
          id,
          name: String(subject.name || "Saved subject"),
          description: String(subject.description || ""),
          trigger_phrase: String(subject.trigger_phrase || ""),
          trigger_position: String(subject.trigger_position || "start") === "end" ? "end" : "start",
          image: normalizeReferenceImage(subject),
        });
        subjectIds.add(id);
      }
    }
    state.referenceBuilder = normalizeReferenceBuilderCatalog(refs);
  }

  function syncReferenceMappingsToVideoCreator() {
    if (!state.onReferenceMappingsChanged) return;
    state.onReferenceMappingsChanged({
      reference_builder: normalizeReferenceBuilderCatalog(state.referenceBuilder),
      scenes: state.scenes.map((scene) => ({
        id: scene.id,
        scene_number: scene.scene_number,
        no_character_present: Boolean(scene.no_character_present),
        subject_ids: scene.no_character_present ? [] : (Array.isArray(scene.subject_refs) ? scene.subject_refs : []).map((ref) => String(ref?.id || "")).filter(Boolean),
        location_id: String(scene.location_ref?.id || ""),
        trigger: String(scene.trigger_phrase || ""),
        trigger_position: String(scene.trigger_position || "start") === "end" ? "end" : "start",
        speaker_assignments: normalizeStoryboardSpeakerAssignments(scene.speaker_assignments),
        lyric_text: String(scene.lyrics || ""),
        lyric_singers: Array.isArray(scene.lyric_singers) ? [...scene.lyric_singers] : [],
      })),
    });
  }

  function upsertStoryboardReference(kind, reference) {
    if (!reference) return null;
    const list = kind === "location" ? state.referenceBuilder.locations : state.referenceBuilder.subjects;
    const name = String(reference.name || "").trim();
    const existing = list.find((item) => String(item.name || "").trim().toLowerCase() === name.toLowerCase());
    const merged = {
      ...(existing || {}),
      ...reference,
      id: existing?.id || reference.id || storyboardReferenceId(kind === "location" ? "loc" : "subj", name),
      name,
      description: String(reference.description || existing?.description || ""),
      image: reference.image || existing?.image || { path: "", data: "", name: "" },
    };
    if (existing) {
      Object.assign(existing, merged);
      return existing;
    }
    list.push(merged);
    return merged;
  }

  async function addStoryboardReferenceFromFile(kind, scene) {
    const file = await chooseStoryboardImageFile();
    if (!file) return null;
    const details = await promptStoryboardReferenceDetails({ kind, file });
    if (!details) return null;
    let reference = details;
    if (state.projectFolder) {
      try {
        const saved = await postJson("/vrgdg/storyboard/import_reference_image", {
          project_folder: state.projectFolder,
          kind,
          name: details.name,
          description: details.description,
          image_data: details.image?.data || "",
          file_name: details.image?.name || file.name || details.name,
        }, 120000);
        reference = saved.reference || reference;
      } catch (error) {
        createToast(`Could not save this reference image into the project folder. It will stay in this session only.\n${String(error?.message || error)}`, true);
      }
    } else {
      createToast("Save the AI Video Builder project first if you want imported Storyboard references to persist.", true);
    }
    const ref = upsertStoryboardReference(kind, reference);
    if (!ref || !scene) return ref;
    if (kind === "location") {
      scene.location_ref = ref;
      scene.setting = ref.description || ref.name || scene.setting || "";
    } else {
      const refs = Array.isArray(scene.subject_refs) ? scene.subject_refs.slice() : [];
      if (!refs.some((item) => String(item.id || "") === String(ref.id || ""))) refs.push(ref);
      scene.subject_refs = refs;
      scene.subjects = storyboardSubjectNamesFromRefs(refs);
    }
    syncReferenceMappingsToVideoCreator();
    renderTable();
    createToast(`${kind === "location" ? "Location" : "Subject"} reference added to ${scene.label || `Scene ${scene.scene_number}`}.`);
    return ref;
  }

  function applyVideoStyle({ overwrite = false } = {}) {
    if (state.mode !== "image_to_video_prep") {
      createToast("Video style is only available in Video Prep.");
      return;
    }
    const value = String(state.videoStyle || "").trim();
    if (!value) {
      createToast("Choose a video style first.");
      return;
    }
    if (value === "custom" && !String(state.videoStyleCustom || "").trim()) {
      createToast("Type the exact custom style wording first.");
      return;
    }
    let changed = 0;
    state.scenes.forEach((scene) => {
      if (!storyboardSceneSupportsVideoStyle(scene)) return;
      if (!overwrite && String(scene.video_style || "").trim()) return;
      scene.video_style = value;
      scene.video_style_custom = value === "custom" ? String(state.videoStyleCustom || "").trim() : "";
      changed += 1;
    });
    renderTable();
    if (overwrite) {
      createToast(changed ? `Video style replaced ${changed} eligible scene${changed === 1 ? "" : "s"}.` : "No eligible video scene styles were changed.");
    } else {
      createToast(changed ? `Video style filled ${changed} eligible scene${changed === 1 ? "" : "s"}.` : "No blank eligible video scene styles needed filling.");
    }
  }

  return {
    absorbSceneReferencesIntoCatalog, addStoryboardReferenceFromFile, applyVideoStyle,
    syncReferenceMappingsToVideoCreator,
  };
}
