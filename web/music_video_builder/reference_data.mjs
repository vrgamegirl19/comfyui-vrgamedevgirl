import { postJson } from "./comfy_api.mjs";
import { LOCATION_SCOUT_ADVANCED_CHATGPT_URL, LOCATION_SCOUT_GPT_URL } from "./constants.mjs";
import { copyTextToClipboard, makeButton, toast } from "./controls.mjs";
import { sceneConceptPromptText } from "./image_prompts.mjs";
import { normalizeMiniMaxH3Voice } from "./minimax_h3.mjs";
import { normalizeBuilderStoryLayer } from "./model_settings.mjs";
import {
  defaultFluxReferenceBuilder,
  isInstrumentalLyricText,
  isNoLipSyncSingerChoice,
  loadI2VMotionNotesFromPath,
} from "./prompt_text.mjs";
import { storyboardSubjectsForSegment } from "./scene_output.mjs";
import { hasLockedVideo } from "./selection_preview.mjs";
import { timelineSegmentDuration } from "./timeline_state.mjs";
import { shiftSegmentTiming } from "./timeline_view.mjs";

export function defaultIdLoraReferenceBuilder() {
  return {
    characters: [],
    locations: [],
    scene_map: {},
  };
}

export function estimateIdLoraDialogueDuration(text) {
  const clean = String(text || "").trim();
  const words = clean.match(/[\p{L}\p{N}'’-]+/gu) || [];
  const spokenSeconds = (words.length / 145) * 60;
  const estimated = spokenSeconds + 1.0;
  const clamped = Math.max(2, Math.min(12, estimated || 2));
  return Math.round(clamped * 4) / 4;
}

export function normalizeIdLoraReferenceBuilder(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const normalized = defaultIdLoraReferenceBuilder();
  const imageObject = (item = {}) => {
    const image = item?.image && typeof item.image === "object" ? item.image : item;
    return {
      path: String(image.path || item.image_path || item.imagePath || item.path || ""),
      data: String(image.data || item.image_data || item.imageData || item.data || ""),
      name: String(image.name || item.image_name || item.imageName || ""),
    };
  };
  normalized.characters = Array.isArray(source.characters) ? source.characters
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `id_lora_char_${Date.now()}_${index}_${Math.floor(Math.random() * 10000)}`),
      name: String(item.name || `Character ${index + 1}`),
      description: String(item.description || ""),
      image: imageObject(item.image || {}),
      voice_audio_path: String(item.voice_audio_path || item.voiceAudioPath || item.reference_audio_path || ""),
      voice_trim_start: Math.max(0, Number(item.voice_trim_start || item.voiceTrimStart || 0)),
      voice_trim_duration: Math.max(0, Number(item.voice_trim_duration || item.voiceTrimDuration || 0)),
      identity_guidance_scale: Number(item.identity_guidance_scale ?? item.identityGuidanceScale ?? 3),
      speech_style: String(item.speech_style || item.speechStyle || ""),
    }))
    : [];
  normalized.locations = Array.isArray(source.locations) ? source.locations
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `id_lora_loc_${Date.now()}_${index}_${Math.floor(Math.random() * 10000)}`),
      name: String(item.name || `Location ${index + 1}`),
      description: String(item.description || ""),
      image: imageObject(item.image || {}),
    }))
    : [];
  const rawMap = source.scene_map && typeof source.scene_map === "object" ? source.scene_map : {};
  normalized.scene_map = {};
  for (const [sceneId, rawEntry] of Object.entries(rawMap)) {
    const entry = rawEntry && typeof rawEntry === "object" ? rawEntry : {};
    const dialogue = String(entry.dialogue || entry.speech || entry.line || "");
    const autoDuration = entry.auto_duration !== false && entry.autoDuration !== false;
    const estimated = estimateIdLoraDialogueDuration(dialogue);
    normalized.scene_map[String(sceneId)] = {
      character_id: String(entry.character_id || entry.characterId || ""),
      location_id: String(entry.location_id || entry.locationId || ""),
      dialogue,
      auto_duration: autoDuration,
      manual_duration: Math.max(0.25, Number(entry.manual_duration || entry.manualDuration || estimated || 2)),
      estimated_duration: estimated,
    };
  }
  return normalized;
}

export function idLoraSceneEntry(refs, segment) {
  const normalizedRefs = normalizeIdLoraReferenceBuilder(refs);
  const sceneId = String(segment?.id || "");
  const existing = normalizedRefs.scene_map[sceneId] || {};
  const dialogue = String(existing.dialogue || segment?.lyric_text || "").trim();
  const estimated = estimateIdLoraDialogueDuration(dialogue);
  return {
    character_id: String(existing.character_id || ""),
    location_id: String(existing.location_id || ""),
    dialogue,
    auto_duration: existing.auto_duration !== false,
    manual_duration: Math.max(0.25, Number(existing.manual_duration || estimated || timelineSegmentDuration(segment) || 2)),
    estimated_duration: estimated,
  };
}

export function defaultLyricMapper() {
  return {
    source_text: "",
    lines: [],
  };
}

export function normalizeLyricMapper(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const normalized = defaultLyricMapper();
  normalized.source_text = String(source.source_text || source.sourceText || "");
  normalized.lines = Array.isArray(source.lines) ? source.lines
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `lyric_line_${index + 1}`),
      text: String(item.text || item.line || ""),
      singers: Array.isArray(item.singers) ? item.singers.map((value) => String(value || "").trim()).filter(Boolean) : [],
      instrumental: Boolean(item.instrumental),
      no_lip_sync: Boolean(item.no_lip_sync || item.noLipSync),
    }))
    : [];
  return normalized;
}

export function isExtraSubjectReference(subject = {}) {
  return Boolean(String(subject?.extra_reference_for || subject?.extraReferenceFor || subject?.same_subject_as || subject?.sameSubjectAs || "").trim());
}

export function subjectExtraTargetId(subject = {}) {
  return String(subject?.extra_reference_for || subject?.extraReferenceFor || subject?.same_subject_as || subject?.sameSubjectAs || "").trim();
}

export function expandSubjectReferencesForRender(refs, subjects = []) {
  const normalizedRefs = normalizeFluxReferenceBuilder(refs);
  const selectedIds = new Set((subjects || []).map((subject) => String(subject?.id || "").trim()).filter(Boolean));
  const expanded = [...subjects];
  for (const subject of normalizedRefs.subjects || []) {
    const targetId = subjectExtraTargetId(subject);
    if (targetId && selectedIds.has(targetId)) expanded.push(subject);
  }
  const seen = new Set();
  return expanded.filter((subject) => {
    const id = String(subject?.id || "").trim();
    if (!id || seen.has(id)) return false;
    seen.add(id);
    return true;
  });
}

export function splitLyricsToMapperLines(text) {
  return String(text || "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !/^\[[^\]]+\]$/.test(line))
    .map((line, index) => {
      const singerMatch = line.match(/^\s*\[([^\]]+)\]\s*(.*)$/);
      const labelText = singerMatch ? singerMatch[1].trim() : "";
      const lyricText = singerMatch ? singerMatch[2].trim() : line;
      const instrumental = isInstrumentalLyricText(labelText) || isInstrumentalLyricText(lyricText);
      const singers = instrumental || !labelText ? [] : labelText.split(/[,+/|;&]+/).map((item) => item.trim()).filter(Boolean);
      return {
        id: `lyric_line_${Date.now()}_${index}_${Math.floor(Math.random() * 10000)}`,
        text: instrumental ? "" : lyricText,
        singers,
        instrumental,
      };
    });
}

export function normalizedLyricMatchText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function lyricMatchScore(sceneText, lineText) {
  const scene = normalizedLyricMatchText(sceneText);
  const line = normalizedLyricMatchText(lineText);
  if (!scene || !line) return 0;
  if (scene === line) return 1000 + scene.length;
  if (line.includes(scene) || scene.includes(line)) return 700 + Math.min(scene.length, line.length);
  const sceneWords = scene.split(" ").filter((word) => word.length > 1);
  const lineWords = new Set(line.split(" ").filter((word) => word.length > 1));
  if (!sceneWords.length || !lineWords.size) return 0;
  const hits = sceneWords.filter((word) => lineWords.has(word)).length;
  return hits / Math.max(sceneWords.length, lineWords.size);
}

export function normalizeFluxReferenceBuilder(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const normalized = defaultFluxReferenceBuilder();
  const normalizeReferenceType = (value = "") => {
    const text = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
    if (["character", "person", "human", "subject", "singer"].includes(text)) return "character";
    if (["prop", "object", "vehicle", "creature", "animal", "outfit", "style", "environment", "other"].includes(text)) return text;
    return "character";
  };
  const normalizeRefImage = (item = {}) => {
    const sourceItem = item && typeof item === "object" ? item : {};
    const image = sourceItem.image && typeof sourceItem.image === "object" ? sourceItem.image : sourceItem;
    const hasTopLevelImage = Boolean(sourceItem.path || sourceItem.data || sourceItem.image_path || sourceItem.imagePath || sourceItem.image_data || sourceItem.imageData);
    return {
      path: String(image.path || sourceItem.image_path || sourceItem.imagePath || sourceItem.path || ""),
      data: String(image.data || sourceItem.image_data || sourceItem.imageData || sourceItem.data || ""),
      name: String(image.name || sourceItem.image_name || sourceItem.imageName || (hasTopLevelImage ? sourceItem.name : "") || ""),
      preview_url: String(image.preview_url || sourceItem.preview_url || ""),
    };
  };
  const refKeyForDedupe = (item = {}) => {
    const name = String(item.name || "").trim().toLowerCase().replace(/\s+/g, " ");
    const extraTarget = String(item.extra_reference_for || item.extraReferenceFor || item.same_subject_as || item.sameSubjectAs || "").trim();
    if (extraTarget) {
      const image = item.image && typeof item.image === "object" ? item.image : item;
      return String(item.id || image.path || image.data || `${name}:${extraTarget}`).trim().toLowerCase();
    }
    return name || String(item.id || "").trim().toLowerCase();
  };
  const dedupeRefsByName = (items = []) => {
    const byKey = new Map();
    for (const item of items) {
      const key = refKeyForDedupe(item);
      if (!key) continue;
      const existing = byKey.get(key) || {};
      byKey.set(key, {
        ...existing,
        ...item,
        id: existing.id || item.id,
        name: existing.name || item.name,
        description: existing.description || item.description,
        face_description: existing.face_description || item.face_description || item.faceDescription,
        face_description_source_key: existing.face_description_source_key || item.face_description_source_key || item.faceDescriptionSourceKey,
        reference_type: existing.reference_type || item.reference_type || item.referenceType || item.type,
        trigger_phrase: existing.trigger_phrase || item.trigger_phrase,
        trigger_position: existing.trigger_position || item.trigger_position,
        extra_reference_for: existing.extra_reference_for || item.extra_reference_for || item.extraReferenceFor || item.same_subject_as || item.sameSubjectAs,
        extra_reference_note: existing.extra_reference_note || item.extra_reference_note || item.extraReferenceNote,
        minimax_voice: existing.minimax_voice || item.minimax_voice || item.miniMaxVoice,
        image: {
          ...(item.image || {}),
          ...(existing.image || {}),
        },
      });
    }
    return Array.from(byKey.values());
  };
  const normalizeReferenceGenerationDraft = (value = {}) => {
    const draft = value && typeof value === "object" ? value : {};
    return {
      subject_mode: String(draft.subject_mode || draft.subjectMode || "description_only"),
      subject_label: String(draft.subject_label || draft.subjectLabel || ""),
      gender_role: String(draft.gender_role || draft.genderRole || ""),
      song_style: String(draft.song_style || draft.songStyle || ""),
      lyrics_context: String(draft.lyrics_context || draft.lyricsContext || ""),
      extra_direction: String(draft.extra_direction || draft.extraDirection || ""),
    };
  };
  normalized.use_subject_reference = Boolean(source.use_subject_reference);
  normalized.extras_enabled = Boolean(source.extras_enabled || source.extrasEnabled);
  normalized.use_location_references = Boolean(source.use_location_references);
  normalized.include_manual_ingredients = source.include_manual_ingredients !== false;
  normalized.cleared = Boolean(source.cleared || source.clear_all || source.empty);
  normalized.locations_cleared = Boolean(source.locations_cleared || source.locationsCleared || source.clear_locations || source.clearLocations);
  normalized.trigger_position = String(source.trigger_position || source.triggerPosition || source.trigger_placement || "start") === "end" ? "end" : "start";
  normalized.subject_trigger_position = String(source.subject_trigger_position || source.subjectTriggerPosition || source.trigger_position || "start") === "end" ? "end" : "start";
  normalized.location_trigger_position = String(source.location_trigger_position || source.locationTriggerPosition || source.trigger_position || "start") === "end" ? "end" : "start";
  normalized.location_style_theme = String(source.location_style_theme || source.locationStyleTheme || "");
  normalized.max_generated_locations = Math.max(1, Math.min(50, Number(source.max_generated_locations || source.maxGeneratedLocations || 8) || 8));
  const hasExplicitSubjectsArray = Array.isArray(source.subjects);
  const rawSubjects = dedupeRefsByName(hasExplicitSubjectsArray ? source.subjects : []);
  normalized.subject_count = normalized.cleared
    ? 0
    : hasExplicitSubjectsArray
      ? Math.min(12, rawSubjects.length)
      : Math.max(0, Math.min(12, rawSubjects.length || Number(source.subject_count || 0) || 0));
  const subject = source.subject && typeof source.subject === "object" ? source.subject : {};
  normalized.subject = {
    description: String(subject.description || ""),
    face_description: String(subject.face_description || subject.faceDescription || ""),
    face_description_source_key: String(subject.face_description_source_key || subject.faceDescriptionSourceKey || ""),
    reference_type: normalizeReferenceType(subject.reference_type || subject.referenceType || subject.type || "character"),
    minimax_voice: normalizeMiniMaxH3Voice(subject.minimax_voice || subject.miniMaxVoice),
    reference_generation_draft: normalizeReferenceGenerationDraft(subject.reference_generation_draft || subject.referenceGenerationDraft),
    image: normalizeRefImage(subject),
  };
  normalized.subjects = rawSubjects.length ? rawSubjects
    .filter((item) => item && typeof item === "object")
    .map((item, index) => {
      return {
        id: String(item.id || `subj_${Date.now()}_${index}_${Math.floor(Math.random() * 10000)}`),
        name: String(item.name || `Character ${index + 1}`),
        description: String(item.description || ""),
        face_description: String(item.face_description || item.faceDescription || ""),
        face_description_source_key: String(item.face_description_source_key || item.faceDescriptionSourceKey || ""),
        auto_build_role: String(item.auto_build_role || item.autoBuildRole || ""),
        reference_type: normalizeReferenceType(item.reference_type || item.referenceType || item.type || "character"),
        trigger_phrase: String(item.trigger_phrase || item.trigger || item.Trigger || ""),
        trigger_position: String(item.trigger_position || item.triggerPosition || item.trigger_placement || "start") === "end" ? "end" : "start",
        extra_reference_for: String(item.extra_reference_for || item.extraReferenceFor || item.same_subject_as || item.sameSubjectAs || ""),
        extra_reference_note: String(item.extra_reference_note || item.extraReferenceNote || ""),
        minimax_voice: normalizeMiniMaxH3Voice(item.minimax_voice || item.miniMaxVoice),
        reference_generation_draft: normalizeReferenceGenerationDraft(item.reference_generation_draft || item.referenceGenerationDraft),
        image: normalizeRefImage(item),
      };
    }) : [];
  if (!hasExplicitSubjectsArray && !normalized.subjects.length && (normalized.subject.description || normalized.subject.image.path || normalized.subject.image.data || normalized.subject.image.name)) {
    normalized.subjects.push({
      id: "subject_1",
      name: "Character 1",
      description: normalized.subject.description,
      face_description: normalized.subject.face_description,
      face_description_source_key: normalized.subject.face_description_source_key,
      reference_type: normalized.subject.reference_type || "character",
      trigger_phrase: String(subject.trigger_phrase || subject.trigger || subject.Trigger || ""),
      trigger_position: String(subject.trigger_position || subject.triggerPosition || subject.trigger_placement || "start") === "end" ? "end" : "start",
      extra_reference_for: "",
      extra_reference_note: "",
      minimax_voice: normalizeMiniMaxH3Voice(subject.minimax_voice || subject.miniMaxVoice),
      reference_generation_draft: normalizeReferenceGenerationDraft(subject.reference_generation_draft || subject.referenceGenerationDraft),
      image: { ...normalized.subject.image },
    });
  }
  while (!hasExplicitSubjectsArray && !normalized.cleared && normalized.subjects.length < normalized.subject_count) {
    normalized.subjects.push({
      id: `subj_${Date.now()}_${normalized.subjects.length}_${Math.floor(Math.random() * 10000)}`,
      name: `Character ${normalized.subjects.length + 1}`,
      description: "",
      reference_type: "character",
      trigger_phrase: "",
      trigger_position: "start",
      extra_reference_for: "",
      extra_reference_note: "",
      minimax_voice: normalizeMiniMaxH3Voice(),
      reference_generation_draft: normalizeReferenceGenerationDraft(),
      image: { path: "", data: "", name: "" },
    });
  }
  const subjectIdSet = new Set(normalized.subjects.map((subject) => String(subject.id || "").trim()).filter(Boolean));
  normalized.subjects.forEach((subject) => {
    const targetId = String(subject.extra_reference_for || "").trim();
    if (!targetId || targetId === subject.id || !subjectIdSet.has(targetId)) {
      subject.extra_reference_for = "";
      subject.extra_reference_note = subject.extra_reference_for ? subject.extra_reference_note : "";
    }
  });
  normalized.subjects = normalized.cleared ? [] : normalized.subjects.slice(0, normalized.subject_count);
  if (!normalized.subjects.length) {
    normalized.subject_count = 0;
    normalized.subject = {
      name: "",
      description: "",
      reference_type: normalized.subject.reference_type || "character",
      minimax_voice: normalized.subject.minimax_voice || normalizeMiniMaxH3Voice(),
      reference_generation_draft: normalized.subject.reference_generation_draft || normalizeReferenceGenerationDraft(),
      image: { path: "", data: "", name: "" },
    };
  }
  if (normalized.subjects.length) {
    const firstSubject = normalized.subjects[0];
    normalized.subject = {
      description: firstSubject.description || normalized.subject.description || "",
      face_description: firstSubject.face_description || normalized.subject.face_description || "",
      face_description_source_key: firstSubject.face_description_source_key || normalized.subject.face_description_source_key || "",
      reference_type: firstSubject.reference_type || normalized.subject.reference_type || "character",
      minimax_voice: normalizeMiniMaxH3Voice(firstSubject.minimax_voice || normalized.subject.minimax_voice),
      reference_generation_draft: firstSubject.reference_generation_draft || normalized.subject.reference_generation_draft || normalizeReferenceGenerationDraft(),
      image: (firstSubject.image?.path || firstSubject.image?.data || firstSubject.image?.name)
        ? { ...(firstSubject.image || { path: "", data: "", name: "" }) }
        : normalized.subject.image,
    };
  }
  normalized.subject_scene_map = {};
  if (source.subject_scene_map && typeof source.subject_scene_map === "object") {
    for (const [sceneId, value] of Object.entries(source.subject_scene_map)) {
      normalized.subject_scene_map[sceneId] = Array.isArray(value) ? value.map(String).filter(Boolean) : String(value || "").split(",").map((item) => item.trim()).filter(Boolean);
    }
  }
  normalized.extra_subjects = Array.isArray(source.extra_subjects || source.extraSubjects)
    ? (source.extra_subjects || source.extraSubjects)
      .filter((item) => item && typeof item === "object")
      .map((item, index) => ({
        id: String(item.id || `extra_${Date.now()}_${index}_${Math.floor(Math.random() * 10000)}`),
        title: String(item.title || item.name || `Extra ${index + 1}`),
        description: String(item.description || ""),
        count: Math.max(1, Math.min(100, Math.round(Number(item.count) || 1))),
        style: String(item.style || ""),
        send_to_minimax: Boolean(item.send_to_minimax ?? item.sendToMiniMax),
        reference_image_type: ["single", "multi_view"].includes(String(item.reference_image_type || item.referenceImageType || "single"))
          ? String(item.reference_image_type || item.referenceImageType || "single")
          : "single",
        image: normalizeRefImage(item),
      }))
    : [];
  const validExtraIds = new Set(normalized.extra_subjects.map((item) => item.id));
  normalized.extra_scene_map = {};
  const rawExtraSceneMap = source.extra_scene_map || source.extraSceneMap;
  if (rawExtraSceneMap && typeof rawExtraSceneMap === "object") {
    for (const [sceneId, value] of Object.entries(rawExtraSceneMap)) {
      const seenExtraIds = new Set();
      const entries = (Array.isArray(value) ? value : []).map((entry) => {
        const extraId = String(entry?.extra_id || entry?.extraId || "").trim();
        if (!validExtraIds.has(extraId) || seenExtraIds.has(extraId)) return null;
        seenExtraIds.add(extraId);
        const interaction = ["background", "background_dancing", "alongside", "dancing_with", "direct"].includes(String(entry?.interaction || "").trim())
          ? String(entry.interaction).trim()
          : "background";
        return { extra_id: extraId, interaction };
      }).filter(Boolean);
      if (entries.length) normalized.extra_scene_map[String(sceneId)] = entries;
    }
  }
  normalized.performer_scene_map = {};
  const rawPerformerSceneMap = source.performer_scene_map || source.performerSceneMap || source.lyric_performer_scene_map || source.lyricPerformerSceneMap;
  if (rawPerformerSceneMap && typeof rawPerformerSceneMap === "object") {
    for (const [sceneId, value] of Object.entries(rawPerformerSceneMap)) {
      normalized.performer_scene_map[sceneId] = Array.isArray(value) ? value.map(String).filter(Boolean) : String(value || "").split(",").map((item) => item.trim()).filter(Boolean);
    }
  }
  normalized.locations = (normalized.cleared || normalized.locations_cleared) ? [] : dedupeRefsByName(Array.isArray(source.locations) ? source.locations : [])
    .filter((item) => item && typeof item === "object")
    .map((item, index) => {
      return {
        id: String(item.id || `loc_${Date.now()}_${index}_${Math.floor(Math.random() * 10000)}`),
        name: String(item.name || `Location ${index + 1}`),
        description: String(item.description || ""),
        auto_build_role: String(item.auto_build_role || item.autoBuildRole || ""),
        trigger_phrase: String(item.trigger_phrase || item.trigger || item.Trigger || item.phrase || ""),
        trigger_position: String(item.trigger_position || item.triggerPosition || item.trigger_placement || "start") === "end" ? "end" : "start",
        reference_generation_draft: normalizeReferenceGenerationDraft(item.reference_generation_draft || item.referenceGenerationDraft),
        image: normalizeRefImage(item),
      };
    });
  normalized.scene_map = (normalized.cleared || normalized.locations_cleared) ? {} : (source.scene_map && typeof source.scene_map === "object" ? { ...source.scene_map } : {});
  normalized.scene_trigger_map = (normalized.cleared || normalized.locations_cleared) ? {} : (source.scene_trigger_map && typeof source.scene_trigger_map === "object" ? { ...source.scene_trigger_map } : {});
  const hasMeaningfulSubject = !normalized.cleared && (
    Boolean(normalized.subject?.description || normalized.subject?.image?.path || normalized.subject?.image?.data || normalized.subject?.image?.name)
    || Object.keys(normalized.subject_scene_map || {}).length > 0
    || normalized.subjects.some((subjectItem) => {
      const name = String(subjectItem?.name || "").trim();
      const placeholderName = /^Character\s+\d+$/i.test(name);
      return Boolean(
        (name && !placeholderName)
        || subjectItem?.description
        || subjectItem?.trigger_phrase
        || subjectItem?.image?.path
        || subjectItem?.image?.data
        || subjectItem?.image?.name
      );
    })
  );
  const validLocationIds = new Set(normalized.locations.map((location) => String(location.id || "").trim()).filter(Boolean));
  if (!validLocationIds.size) {
    normalized.scene_map = {};
  } else {
    for (const [sceneId, locationId] of Object.entries(normalized.scene_map || {})) {
      if (!validLocationIds.has(String(locationId || "").trim())) delete normalized.scene_map[sceneId];
    }
  }
  const hasMeaningfulLocation = !normalized.cleared && !normalized.locations_cleared && (
    normalized.locations.length > 0
    || Object.keys(normalized.scene_map || {}).length > 0
    || Object.keys(normalized.scene_trigger_map || {}).length > 0
  );
  normalized.use_subject_reference = Boolean(normalized.use_subject_reference || hasMeaningfulSubject);
  normalized.use_location_references = Boolean(normalized.use_location_references || hasMeaningfulLocation);
  normalized.ingredients_sheets = Array.isArray(source.ingredients_sheets) ? source.ingredients_sheets
    .filter((item) => item && typeof item === "object")
    .map((item, index) => ({
      id: String(item.id || `ingredients_${Date.now()}_${index}_${Math.floor(Math.random() * 10000)}`),
      name: String(item.name || `Ingredients Sheet ${index + 1}`),
      description: String(item.description || item.notes || ""),
      image: normalizeRefImage(item),
    })) : [];
  normalized.ingredients_scene_map = {};
  if (source.ingredients_scene_map && typeof source.ingredients_scene_map === "object") {
    for (const [sceneId, value] of Object.entries(source.ingredients_scene_map)) {
      const sheetId = String(value || "").trim();
      if (sheetId) normalized.ingredients_scene_map[sceneId] = sheetId;
    }
  }
  const defaultSources = defaultFluxReferenceBuilder().ingredients_auto_map_sources;
  const rawSources = source.ingredients_auto_map_sources && typeof source.ingredients_auto_map_sources === "object"
    ? source.ingredients_auto_map_sources
    : {};
  normalized.ingredients_auto_map_sources = {
    director_notes: rawSources.director_notes !== false,
    concept_prompt: rawSources.concept_prompt !== false,
    scene_notes: rawSources.scene_notes !== false,
    lyric_text: rawSources.lyric_text !== false,
  };
  for (const key of Object.keys(defaultSources)) {
    if (!(key in normalized.ingredients_auto_map_sources)) normalized.ingredients_auto_map_sources[key] = defaultSources[key];
  }
  return normalized;
}

function normalizedIngredientsMatchText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function ingredientsAutoMapTextForSegment(segment, sources = {}) {
  const parts = [];
  if (sources.director_notes !== false) parts.push(segment?.timeline_note || "");
  if (sources.concept_prompt !== false) parts.push(sceneConceptPromptText(segment));
  if (sources.scene_notes !== false) parts.push(segment?.notes || "", segment?.i2v_notes || "");
  if (sources.lyric_text !== false) parts.push(segment?.lyric_text || "");
  return normalizedIngredientsMatchText(parts.join(" "));
}

async function showLocationScoutGptHandoff(payloadJson, options = {}) {
  const gptLabel = String(options.gptLabel || "Location Scout GPT");
  const gptUrl = String(options.gptUrl || LOCATION_SCOUT_GPT_URL);
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100010;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(860px,calc(100vw - 48px));max-height:calc(100vh - 48px);border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 22px 80px rgba(0,0,0,.62);display:flex;flex-direction:column;overflow:hidden;";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;background:#083f4f;border-bottom:1px solid #155e75;padding:13px 15px;";
  const title = document.createElement("div");
  title.innerHTML = `<div style="font-size:17px;font-weight:900;color:#cffafe;">Open ${gptLabel}</div><div style="font-size:12px;color:#cbd5e1;margin-top:3px;">Copy this JSON, paste it into the GPT, then paste the GPT output back into Import List.</div>`;
  const close = makeButton("Close");
  header.append(title, close);
  const body = document.createElement("div");
  body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:12px;overflow:auto;";
  const note = document.createElement("div");
  note.style.cssText = "border:1px solid #155e75;border-radius:7px;background:#083344;color:#e0f2fe;padding:10px 12px;font-size:12px;line-height:1.5;";
  note.textContent = "We will copy the JSON below to your clipboard. In the GPT, paste it as the input. When the GPT gives you locations, come back here and use Import List / Import JSON to paste that output into the builder.";
  const status = document.createElement("div");
  status.style.cssText = "font-size:12px;color:#94a3b8;min-height:18px;";
  const text = document.createElement("textarea");
  text.value = payloadJson;
  text.spellcheck = false;
  text.style.cssText = "min-height:300px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;font-family:monospace;line-height:1.45;white-space:pre;overflow:auto;";
  const actions = document.createElement("div");
  actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;";
  const cancel = makeButton("Cancel");
  const copy = makeButton("Copy JSON", "primary");
  const continueButton = makeButton("Continue to GPT", "primary");
  const importButton = makeButton("Continue to Import List", "primary");
  importButton.style.display = "none";
  actions.append(cancel, copy, continueButton, importButton);
  body.append(note, status, text, actions);
  box.append(header, body);
  backdrop.append(box);
  document.body.append(backdrop);

  const closeModal = () => backdrop.remove();
  const copyJson = async () => {
    const copied = await copyTextToClipboard(text.value).catch(() => false);
    status.textContent = copied
      ? "Copied JSON to clipboard. Paste it into the GPT when it opens."
      : "Clipboard copy was blocked. You can manually select and copy the JSON from the box.";
    status.style.color = copied ? "#67e8f9" : "#fbbf24";
    return copied;
  };
  close.onclick = closeModal;
  cancel.onclick = closeModal;
  copy.onclick = copyJson;
  continueButton.onclick = async () => {
    await copyJson();
    const gptWindow = window.open(gptUrl, "_blank", "noopener,noreferrer");
    toast(gptWindow ? `Opened ${gptLabel}. Paste the copied JSON there.` : `The ${gptLabel} popup was blocked. Copy the JSON, then open ChatGPT manually.`, !gptWindow);
    if (typeof options.onImportList === "function") {
      importButton.style.display = "";
      status.textContent = "After the GPT gives you locations, click Continue to Import List and paste the GPT output there.";
      status.style.color = "#67e8f9";
    }
  };
  importButton.onclick = () => {
    closeModal();
    if (typeof options.onImportList === "function") {
      setTimeout(() => options.onImportList(), 40);
    }
  };
  backdrop.addEventListener("pointerdown", (event) => {
    if (event.target === backdrop) closeModal();
  });
  await copyJson();
  text.focus();
  text.select();
}

export function createReferenceData({
  addSceneImageHistoryPath, allEditableSegments, conceptPromptsTextFromSegments, currentVideoMode,
  hasAnyI2VMotionNotes, i2vMotionJsonInput, i2vMotionNotesTextFromSegments, i2vVideoSettingsForSegment,
  promptJsonInput, segmentIndexInfo, state, timelineDuration,
}) {
  function buildIdLoraPromptForScene(basePrompt, segment) {
    const context = idLoraSceneContext(segment);
    const prompt = String(basePrompt || "").trim();
    if (!context.contextText) return prompt;
    return `${prompt}\n\n[ID-LORA SCENE CASTING]\n${context.contextText}`.trim();
  }

  function storyboardReferenceBuilderWithIdLoraRefs(baseRefs = state.fluxReferenceBuilder) {
    const refs = normalizeFluxReferenceBuilder(baseRefs);
    if (currentVideoMode() !== "id_lora") return refs;
    const idRefs = normalizeIdLoraReferenceBuilder(state.idLoraReferenceBuilder);
    if (!idRefs.characters.length && !idRefs.locations.length) return refs;

    refs.subjects = Array.isArray(refs.subjects) ? refs.subjects.map((item) => ({ ...item, image: { ...(item.image || {}) } })) : [];
    refs.locations = Array.isArray(refs.locations) ? refs.locations.map((item) => ({ ...item, image: { ...(item.image || {}) } })) : [];
    refs.subject_scene_map = refs.subject_scene_map && typeof refs.subject_scene_map === "object" ? { ...refs.subject_scene_map } : {};
    refs.scene_map = refs.scene_map && typeof refs.scene_map === "object" ? { ...refs.scene_map } : {};

    const existingSubjectIds = new Set(refs.subjects.map((item) => String(item.id || "")));
    const existingLocationIds = new Set(refs.locations.map((item) => String(item.id || "")));
    const subjectIdFor = (id) => `id_lora_character_${String(id || "").replace(/[^a-z0-9_-]+/gi, "_")}`;
    const locationIdFor = (id) => `id_lora_location_${String(id || "").replace(/[^a-z0-9_-]+/gi, "_")}`;

    for (const character of idRefs.characters) {
      const id = subjectIdFor(character.id);
      if (existingSubjectIds.has(id)) continue;
      refs.subjects.push({
        id,
        name: String(character.name || "").trim() || "ID-LoRA Character",
        description: String(character.description || "").trim(),
        reference_type: "character",
        trigger_phrase: "",
        trigger_position: refs.subject_trigger_position || "start",
        image: { ...(character.image || {}) },
        source: "id_lora",
      });
      existingSubjectIds.add(id);
    }

    for (const location of idRefs.locations) {
      const id = locationIdFor(location.id);
      if (existingLocationIds.has(id)) continue;
      refs.locations.push({
        id,
        name: String(location.name || "").trim() || "ID-LoRA Location",
        description: String(location.description || "").trim(),
        trigger_phrase: "",
        trigger_position: refs.location_trigger_position || "start",
        image: { ...(location.image || {}) },
        source: "id_lora",
      });
      existingLocationIds.add(id);
    }

    for (const segment of allEditableSegments()) {
      const entry = idLoraSceneEntry(idRefs, segment);
      const sceneId = String(segment?.id || "");
      if (!sceneId) continue;
      if (entry.character_id) {
        const mappedId = subjectIdFor(entry.character_id);
        const existing = sceneReferenceMapArray(refs.subject_scene_map, segment);
        refs.subject_scene_map[sceneId] = Array.from(new Set([...existing, mappedId].filter(Boolean)));
      }
      if (entry.location_id) refs.scene_map[sceneId] = locationIdFor(entry.location_id);
    }

    refs.use_subject_reference = refs.use_subject_reference || refs.subjects.some((item) => String(item.description || item.image?.path || item.image?.data || "").trim());
    refs.use_location_references = refs.use_location_references || refs.locations.some((item) => String(item.description || item.image?.path || item.image?.data || "").trim());
    refs.subject_count = refs.subjects.length || refs.subject_count || 1;
    return normalizeFluxReferenceBuilder(refs);
  }

  function setBaseSegmentDurationRipple(segment, duration) {
    if (!segment || hasLockedVideo(segment)) return;
    const nextDuration = Math.max(0.25, Number(duration || 0));
    const sorted = [...state.segments].sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    const index = sorted.findIndex((item) => item.id === segment.id);
    if (index < 0) return;
    const oldEnd = Number(segment.end || segment.start || 0);
    const nextEnd = Number(segment.start || 0) + nextDuration;
    const delta = nextEnd - oldEnd;
    segment.end = nextEnd;
    if (Math.abs(delta) > 0.0001) {
      for (let i = index + 1; i < sorted.length; i += 1) {
        if (!hasLockedVideo(sorted[i])) shiftSegmentTiming(sorted[i], delta);
      }
    }
    state.duration = timelineDuration();
  }

  function referenceBuilderSubjectChoices() {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const choices = [];
    const seen = new Set();
    const add = (id, label) => {
      const cleanLabel = String(label || "").trim();
      if (!cleanLabel) return;
      const cleanId = String(id || cleanLabel).trim();
      if (!cleanId || seen.has(cleanId)) return;
      seen.add(cleanId);
      choices.push({ id: cleanId, label: cleanLabel });
    };
    for (const subject of logicalReferenceSubjects(refs)) {
      const subjectName = String(subject.name || "").trim();
      add(subject.id, subjectName && subjectName !== "Character 1" ? subjectName : "the performer");
    }
    if (!choices.length && (refs.subject?.description || refs.subject?.image?.path || refs.subject?.image?.data)) {
      add("subject", "Subject");
    }
    const hasReferenceSubjects = choices.length > 0;
    if (!hasReferenceSubjects) {
      add("female", "Female");
      add("male", "Male");
      add("other", "Other performer");
    }
    add("group", "Group / all visible performers");
    add("b_roll", "B-roll / no lip-sync");
    choices.hasReferenceSubjects = hasReferenceSubjects;
    return choices;
  }

  function sceneReferenceMapValue(map, segment, index = null) {
    const source = map && typeof map === "object" ? map : {};
    const sceneId = String(segment?.id || "").trim();
    if (sceneId && Object.prototype.hasOwnProperty.call(source, sceneId)) return source[sceneId];
    const hasSceneIdKeys = Object.keys(source).some((key) => !/^\d+$/.test(String(key || "").trim()));
    if (hasSceneIdKeys) return undefined;
    const sceneNumber = Number.isFinite(Number(index)) ? Number(index) + 1 : segmentIndexInfo(segment).index + 1;
    const numberKey = String(sceneNumber);
    return Object.prototype.hasOwnProperty.call(source, numberKey) ? source[numberKey] : undefined;
  }

  function sceneReferenceMapArray(map, segment, index = null) {
    const value = sceneReferenceMapValue(map, segment, index);
    return Array.isArray(value)
      ? value.map((item) => String(item || "").trim()).filter(Boolean)
      : String(value || "").split(",").map((item) => item.trim()).filter(Boolean);
  }

  function logicalReferenceSubjects(refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder)) {
    const normalizedRefs = normalizeFluxReferenceBuilder(refs);
    return (normalizedRefs.subjects || []).filter((subject) => !isExtraSubjectReference(subject));
  }

  function logicalSubjectIdsForScene(refs, segment, index = null) {
    const normalizedRefs = normalizeFluxReferenceBuilder(refs);
    const ids = sceneReferenceMapArray(normalizedRefs.subject_scene_map, segment, index);
    const validLogicalIds = new Set(logicalReferenceSubjects(normalizedRefs).map((subject) => subject.id));
    const mappedToLogical = ids
      .map((id) => {
        const subject = normalizedRefs.subjects.find((item) => item.id === id);
        if (!subject) return "";
        return subjectExtraTargetId(subject) || subject.id;
      })
      .filter((id) => id && validLogicalIds.has(id));
    return Array.from(new Set(mappedToLogical));
  }

  function logicalExtraSubjectsForScene(refs, segment, index = null) {
    const normalizedRefs = normalizeFluxReferenceBuilder(refs);
    if (!normalizedRefs.extras_enabled) return [];
    const rawEntries = sceneReferenceMapValue(normalizedRefs.extra_scene_map, segment, index);
    const entries = Array.isArray(rawEntries) ? rawEntries : [];
    const extrasById = new Map((normalizedRefs.extra_subjects || []).map((extra) => [String(extra.id || ""), extra]));
    const seen = new Set();
    return entries.map((entry) => {
      const extraId = String(entry?.extra_id || entry?.extraId || "").trim();
      const extra = extrasById.get(extraId);
      if (!extra || !String(extra.description || "").trim() || seen.has(extraId)) return null;
      seen.add(extraId);
      const interaction = ["background", "background_dancing", "alongside", "dancing_with", "direct"].includes(String(entry?.interaction || "").trim())
        ? String(entry.interaction).trim()
        : "background";
      return { extra, interaction };
    }).filter(Boolean);
  }

  function applyLyricMapperToSegments({ overwriteSingers = true } = {}) {
    state.lyricMapper = normalizeLyricMapper(state.lyricMapper);
    const mapperLines = state.lyricMapper.lines || [];
    if (!mapperLines.length) return 0;
    // Keep an explicit relationship between a mapper row and its timeline
    // scene. Text matching alone cannot find a scene that is currently marked
    // instrumental, which made correcting an instrumental row a no-op.
    const segments = state.segments.filter((segment) => segment && typeof segment === "object");
    const byMapperId = new Map(segments
      .filter((segment) => segment.lyric_mapper_line_id)
      .map((segment) => [String(segment.lyric_mapper_line_id), segment]));
    const assignments = new Map();
    const usedSegments = new Set();
    const assign = (line, segment) => {
      if (!line || !segment || usedSegments.has(segment.id)) return false;
      assignments.set(line.id, segment);
      usedSegments.add(segment.id);
      return true;
    };
    for (const line of mapperLines) assign(line, byMapperId.get(String(line.id || "")));
    if (mapperLines.length === segments.length) {
      mapperLines.forEach((line, index) => assign(line, segments[index]));
    }
    for (const line of mapperLines) {
      if (assignments.has(line.id)) continue;
      let best = null;
      let bestScore = 0;
      for (const segment of segments) {
        if (usedSegments.has(segment.id)) continue;
        const current = String(segment.lyric_text || "").trim();
        const score = line.instrumental && isInstrumentalLyricText(current)
          ? 1000
          : lyricMatchScore(current, line.text);
        if (score > bestScore) { best = segment; bestScore = score; }
      }
      if (best && bestScore >= 0.32) assign(line, best);
    }
    // If a mapper row was added while its corresponding scene is still
    // instrumental, preserve the user's line order as the final fallback.
    mapperLines.forEach((line, index) => assign(line, segments[index]));
    let applied = 0;
    for (const line of mapperLines) {
      const segment = assignments.get(line.id);
      if (!segment) continue;
      segment.lyric_mapper_line_id = line.id;
      if (line.instrumental) {
        segment.lyric_text = "[instrumental]";
        segment.lyric_section = "instrumental";
        if (overwriteSingers) segment.lyric_singers = [];
        segment.lyric_no_lip_sync = true;
      } else {
        segment.lyric_text = String(line.text || "").trim();
        if (String(segment.lyric_section || "").trim().toLowerCase() === "instrumental") segment.lyric_section = "";
        if (overwriteSingers || !Array.isArray(segment.lyric_singers) || !segment.lyric_singers.length) {
        const singers = Array.isArray(line.singers) ? [...line.singers] : [];
        if (line.no_lip_sync || singers.some(isNoLipSyncSingerChoice)) {
          segment.lyric_singers = singers.filter((value) => !isNoLipSyncSingerChoice(value));
          segment.lyric_no_lip_sync = true;
        } else {
          segment.lyric_singers = singers;
          segment.lyric_no_lip_sync = false;
        }
        }
      }
      applied += 1;
    }
    return applied;
  }

  function syncLyricMapperFromSegments() {
    const segments = state.segments.filter((segment) => segment && typeof segment === "object");
    if (!segments.length) return 0;
    const mapper = normalizeLyricMapper(state.lyricMapper);
    const lines = Array.isArray(mapper.lines) ? mapper.lines : [];
    const byId = new Map(lines.map((line) => [String(line.id || ""), line]));
    const used = new Set();
    const pairs = [];
    const pair = (segment, line) => {
      if (!segment || !line || used.has(line.id)) return false;
      used.add(line.id); pairs.push([segment, line]); return true;
    };
    for (const segment of segments) pair(segment, byId.get(String(segment.lyric_mapper_line_id || "")));
    if (lines.length === segments.length) segments.forEach((segment, index) => pair(segment, lines[index]));
    for (const segment of segments) {
      if (pairs.some(([item]) => item === segment)) continue;
      let best = null; let score = 0;
      for (const line of lines) {
        if (used.has(line.id)) continue;
        const candidate = segment.lyric_no_lip_sync && isInstrumentalLyricText(segment.lyric_text) ? 1 : lyricMatchScore(segment.lyric_text, line.text);
        if (candidate > score) { best = line; score = candidate; }
      }
      if (best && score >= 0.32) pair(segment, best);
    }
    if (!lines.length) {
      for (const segment of segments) lines.push({ id: `lyric_line_${Date.now()}_${lines.length}`, text: "", singers: [], instrumental: false });
      segments.forEach((segment, index) => pair(segment, lines[index]));
    }
    for (const [segment, line] of pairs) {
      const instrumental = Boolean(segment.lyric_no_lip_sync && isInstrumentalLyricText(segment.lyric_text));
      line.instrumental = instrumental;
      line.text = instrumental ? "" : String(segment.lyric_text || "").trim();
      line.singers = instrumental ? [] : (Array.isArray(segment.lyric_singers) ? [...segment.lyric_singers] : []);
      line.no_lip_sync = Boolean(segment.lyric_no_lip_sync && !instrumental);
      segment.lyric_mapper_line_id = line.id;
    }
    state.lyricMapper = normalizeLyricMapper({ ...mapper, lines });
    return pairs.length;
  }

  function ingredientsSheetForSegment(segment, refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder)) {
    if (!segment?.id) return null;
    const sheetId = String(refs.ingredients_scene_map?.[segment.id] || "").trim();
    if (!sheetId) return null;
    return (refs.ingredients_sheets || []).find((sheet) => String(sheet.id || "") === sheetId) || null;
  }

  function autoMapIngredientsSheets(refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder), sources = refs.ingredients_auto_map_sources) {
    const normalizedRefs = normalizeFluxReferenceBuilder(refs);
    const sheets = (normalizedRefs.ingredients_sheets || [])
      .map((sheet) => ({
        ...sheet,
        matchName: normalizedIngredientsMatchText(sheet.name),
      }))
      .filter((sheet) => sheet.matchName)
      .sort((a, b) => b.matchName.length - a.matchName.length);
    let mapped = 0;
    for (const segment of allEditableSegments()) {
      const haystack = ingredientsAutoMapTextForSegment(segment, sources);
      if (!haystack) continue;
      const match = sheets.find((sheet) => haystack.includes(sheet.matchName));
      if (!match) continue;
      normalizedRefs.ingredients_scene_map[segment.id] = match.id;
      mapped += 1;
    }
    normalizedRefs.ingredients_auto_map_sources = {
      director_notes: sources?.director_notes !== false,
      concept_prompt: sources?.concept_prompt !== false,
      scene_notes: sources?.scene_notes !== false,
      lyric_text: sources?.lyric_text !== false,
    };
    return { refs: normalizedRefs, mapped };
  }

  function applyIngredientsSheetToSegment(segment, sheet) {
    if (!segment || !sheet) return false;
    const image = sheet.image || {};
    const path = String(image.path || "").trim();
    const data = String(image.data || "").trim();
    if (!path && !data) return false;
    if (path) {
      segment.custom_image_path = path;
      segment.custom_image_data = "";
      if (typeof addSceneImageHistoryPath === "function") addSceneImageHistoryPath(segment, path);
    } else {
      segment.custom_image_path = "";
      segment.custom_image_data = data;
    }
    segment.custom_image_name = image.name || sheet.name || "ingredients_reference.png";
    segment.approved_image_path = "";
    segment.image = null;
    segment.preview_mode = "image";
    return true;
  }

  function applyIngredientsReferenceMappings(refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder)) {
    const normalizedRefs = normalizeFluxReferenceBuilder(refs);
    let applied = 0;
    for (const segment of allEditableSegments()) {
      if (applyIngredientsSheetToSegment(segment, ingredientsSheetForSegment(segment, normalizedRefs))) applied += 1;
    }
    return applied;
  }

  function locationScoutLyricsPayloadForGpt() {
    return allEditableSegments()
      .map((segment, index) => {
        const lyric = String(segment?.lyric_text || "").trim();
        if (!lyric) return "";
        const section = String(segment?.lyric_section || "").trim();
        return `${section ? `[${section}] ` : ""}Scene ${index + 1}: ${lyric}`;
      })
      .filter(Boolean)
      .join("\n");
  }

  function locationScoutCharacterPayloadForGpt(refsInput = state.fluxReferenceBuilder) {
    const normalizedRefs = normalizeFluxReferenceBuilder(refsInput);
    const rows = [];
    const seen = new Set();
    const addRow = (item, fallbackType = "character") => {
      const name = String(item?.name || "").trim();
      const description = String(item?.description || "").trim();
      const type = String(item?.reference_type || item?.referenceType || item?.type || fallbackType).trim() || fallbackType;
      if (!name && !description) return;
      const key = `${name.toLowerCase()}|${description.toLowerCase()}`;
      if (seen.has(key)) return;
      seen.add(key);
      rows.push({ name, type, description });
    };
    for (const subject of normalizedRefs.subjects || []) addRow(subject, "character");
    for (const sheet of normalizedRefs.ingredients_sheets || []) addRow(sheet, "ingredients_sheet");
    return rows;
  }

  async function openLocationScoutGptForRefs(refsInput = state.fluxReferenceBuilder, styleTheme = "", options = {}) {
    const payload = {
      lyrics: locationScoutLyricsPayloadForGpt(),
      style_theme: String(styleTheme || "").trim(),
      character_descriptions: locationScoutCharacterPayloadForGpt(refsInput),
    };
    if (!payload.lyrics) {
      toast("Create or paste lyric scenes before opening the Location Scout GPT.", true);
      return;
    }
    await showLocationScoutGptHandoff(JSON.stringify(payload, null, 2), options);
  }

  function locationScoutAdvancedPayloadForGpt(refsInput = state.fluxReferenceBuilder, styleTheme = "") {
    const refs = normalizeFluxReferenceBuilder(refsInput);
    const sourceLyrics = String(state.lyricMapper?.source_text || "").trim();
    const scenes = allEditableSegments().map((segment, index) => {
      const locationId = String(refs.scene_map?.[segment.id] || refs.scene_map?.[String(index + 1)] || "").trim();
      const location = refs.locations.find((item) => String(item.id || "") === locationId);
      return {
        scene_number: index + 1,
        id: String(segment.id || `scene_${index + 1}`),
        label: String(segment.label || `Scene ${index + 1}`),
        lyric_section: String(segment.lyric_section || "").trim(),
        lyrics: String(segment.lyric_text || segment.lyrics || "").trim(),
        story_beat: String(segment.story_beat || "").trim(),
        scene_notes: String(segment.notes || segment.director_note || "").trim(),
        subjects: storyboardSubjectsForSegment(segment),
        current_location: location ? { name: location.name || "", description: location.description || "" } : null,
      };
    });
    return {
      payload_type: "advanced_location_scout_planning",
      execution_mode: "execute_immediately",
      user_request: "Create a coherent location catalog and assign one fitting location to every scene now. Do not ask what I want done. Return only valid JSON matching output_format.",
      task_instruction: "Use the story layer as the narrative authority and the lyrics, lyric sections, scene story beats, subject descriptions, style, and existing mappings as constraints. Design locations that support the emotional progression and visual continuity. Reuse a location when the story calls for a recurring motif or chorus return. Do not assign a random place merely because it appears literally in a lyric. Every scene must receive one location assignment.",
      output_format: {
        locations: [
          { name: "Reusable location name", description: "Detailed physical setting, visual identity, lighting potential, props, and continuity notes." },
        ],
        scene_map: {
          scene1: { location: "Exact location name from locations", description: "Why this location fits this scene's lyric and story beat." },
        },
      },
      story_layer: normalizeBuilderStoryLayer(state.builderStoryLayer),
      source_lyrics: sourceLyrics,
      lyrics_instruction: "Use source_lyrics when scene lyrics are empty. Preserve lyric order and infer section structure from headings when possible.",
      style_context: {
        location_style_theme: String(styleTheme || refs.location_style_theme || "").trim(),
        video_style: String(state.builderStoryboardDefaults?.video_style || "").trim(),
        video_style_custom: String(state.builderStoryboardDefaults?.video_style_custom || "").trim(),
        camera_flow: String(state.builderStoryboardDefaults?.camera_flow || "").trim(),
        image_world_style: String(state.builderStoryLayer?.image_world_style || "").trim(),
        global_consistency_phrase: String(state.builderStoryboardDefaults?.global_consistency_phrase || "").trim(),
      },
      subject_descriptions: locationScoutCharacterPayloadForGpt(refs),
      existing_locations: (refs.locations || []).map((item) => ({ name: item.name || "", description: item.description || "" })),
      scenes,
    };
  }

  async function openAdvancedLocationScoutGptForRefs(refsInput = state.fluxReferenceBuilder, styleTheme = "", options = {}) {
    const payload = locationScoutAdvancedPayloadForGpt(refsInput, styleTheme);
    const storyLayer = payload.story_layer || {};
    if (!String(storyLayer.overall_story_idea || storyLayer.user_story_arc || storyLayer.song_story_brief || "").trim()) {
      toast("Set up or import the Storyboard story idea, story arc, and story brief before using GPT Scout Advanced.", true);
      return;
    }
    if (!payload.source_lyrics && !payload.scenes.some((scene) => scene.lyrics || scene.story_beat || scene.scene_notes)) {
      toast("Add lyrics or scene story context before opening Advanced Location Scout.", true);
      return;
    }
    await showLocationScoutGptHandoff(JSON.stringify(payload, null, 2), {
      ...options,
      gptUrl: LOCATION_SCOUT_ADVANCED_CHATGPT_URL,
      gptLabel: "Advanced Location Scout",
    });
  }

  function syncIngredientsSceneMapFromSubjectMappings(refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder)) {
    const normalizedRefs = normalizeFluxReferenceBuilder(refs);
    const sheets = normalizedRefs.ingredients_sheets || [];
    const subjects = normalizedRefs.subjects || [];
    if (!sheets.length || !subjects.length) return { refs: normalizedRefs, mapped: 0 };
    const key = (value) => String(value || "").trim().toLowerCase().replace(/\s+/g, " ");
    const sheetByName = new Map();
    for (const sheet of sheets) {
      const sheetKey = key(sheet.name);
      if (sheetKey && !sheetByName.has(sheetKey)) sheetByName.set(sheetKey, sheet);
    }
    if (!sheetByName.size) return { refs: normalizedRefs, mapped: 0 };
    const subjectById = new Map(subjects.map((subject) => [String(subject.id || ""), subject]));
    normalizedRefs.ingredients_scene_map = normalizedRefs.ingredients_scene_map || {};
    let mapped = 0;
    for (const segment of allEditableSegments()) {
      const subjectIds = Array.isArray(normalizedRefs.subject_scene_map?.[segment.id]) ? normalizedRefs.subject_scene_map[segment.id] : [];
      const singerNames = Array.isArray(segment.lyric_singers) ? segment.lyric_singers : [];
      const names = [
        ...subjectIds.map((id) => subjectById.get(String(id || ""))?.name || ""),
        ...singerNames,
      ];
      const match = names.map((name) => sheetByName.get(key(name))).find(Boolean);
      if (match?.id) {
        normalizedRefs.ingredients_scene_map[segment.id] = match.id;
        mapped += 1;
      }
    }
    return { refs: normalizedRefs, mapped };
  }

  function applyIngredientsSheetForSceneIfMapped(segment) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    return applyIngredientsSheetToSegment(segment, ingredientsSheetForSegment(segment, refs));
  }

  async function syncPromptJsonFromSegments(reason = "") {
    const path = String(promptJsonInput.value || state.promptJsonPath || "").trim();
    if (!path) return false;
    try {
      const result = await postJson("/vrgdg/music_builder/save_text_file", {
        path,
        content: conceptPromptsTextFromSegments(),
      });
      promptJsonInput.value = result.path || path;
      state.promptJsonPath = promptJsonInput.value;
      return true;
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Could not sync ConceptPrompts after ${reason || "segment change"}:`, error);
      toast(`Could not update ConceptPrompts.txt:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  async function syncI2VMotionJsonFromSegments(reason = "") {
    const path = String(i2vMotionJsonInput.value || state.i2vMotionJsonPath || "").trim();
    if (!path) return false;
    try {
      if (!hasAnyI2VMotionNotes()) {
        try {
          const existingNotes = await loadI2VMotionNotesFromPath(path);
          if (existingNotes.some((note) => String(note || "").trim())) {
            console.warn(`[VRGDG Music Builder] Skipped blank I2VMotionNotes sync after ${reason || "segment change"} because the existing file has motion notes.`);
            return false;
          }
        } catch (_error) {
          // If the file does not exist yet, allow the normal save path below.
        }
      }
      const result = await postJson("/vrgdg/music_builder/save_text_file", {
        path,
        content: i2vMotionNotesTextFromSegments(),
      });
      i2vMotionJsonInput.value = result.path || path;
      state.i2vMotionJsonPath = i2vMotionJsonInput.value;
      return true;
    } catch (error) {
      console.warn(`[VRGDG Music Builder] Could not sync I2VMotionNotes after ${reason || "segment change"}:`, error);
      toast(`Could not update I2VMotionNotes.txt:\n${String(error?.message || error)}`, true);
      return false;
    }
  }

  function idLoraSceneContext(segment) {
    const refs = normalizeIdLoraReferenceBuilder(state.idLoraReferenceBuilder);
    const entry = idLoraSceneEntry(refs, segment);
    const character = refs.characters.find((item) => item.id === entry.character_id) || null;
    const location = refs.locations.find((item) => item.id === entry.location_id) || null;
    const settings = i2vVideoSettingsForSegment(segment);
    const fallbackVoicePath = String(settings.id_lora_reference_audio_path || settings.reference_audio_path || "").trim();
    const dialogue = String(entry.dialogue || segment?.lyric_text || "").trim();
    const characterName = String(character?.name || "").trim();
    const locationName = String(location?.name || "").trim();
    const characterDescription = String(character?.description || "").trim();
    const locationDescription = String(location?.description || "").trim();
    const speechStyle = String(character?.speech_style || "").trim();
    const voicePath = String(character?.voice_audio_path || fallbackVoicePath || "").trim();
    const identityScale = Number(character?.identity_guidance_scale ?? settings.identity_guidance_scale ?? 3);
    const contextLines = [];
    if (characterName) contextLines.push(`Speaking character: ${characterName}`);
    if (characterDescription) contextLines.push(`Character identity: ${characterDescription}`);
    if (locationName) contextLines.push(`Location: ${locationName}`);
    if (locationDescription) contextLines.push(`Location details: ${locationDescription}`);
    if (speechStyle) contextLines.push(`Voice style: ${speechStyle}`);
    if (dialogue) contextLines.push(`Exact dialogue to generate: "${dialogue}"`);
    return {
      refs,
      entry,
      character,
      location,
      dialogue,
      characterName,
      locationName,
      voicePath,
      voiceTrimStart: 0,
      voiceTrimDuration: 0,
      identityScale: Number.isFinite(identityScale) ? identityScale : 3,
      contextText: contextLines.join("\n"),
      usesFallbackVoice: !String(character?.voice_audio_path || "").trim() && !!fallbackVoicePath,
    };
  }

  return {
    applyIngredientsReferenceMappings, applyIngredientsSheetForSceneIfMapped, applyLyricMapperToSegments,
    autoMapIngredientsSheets, buildIdLoraPromptForScene, idLoraSceneContext, ingredientsSheetForSegment,
    locationScoutCharacterPayloadForGpt, locationScoutLyricsPayloadForGpt, logicalExtraSubjectsForScene,
    logicalReferenceSubjects, logicalSubjectIdsForScene, openAdvancedLocationScoutGptForRefs,
    openLocationScoutGptForRefs, referenceBuilderSubjectChoices, sceneReferenceMapArray,
    sceneReferenceMapValue, setBaseSegmentDurationRipple, storyboardReferenceBuilderWithIdLoraRefs,
    syncI2VMotionJsonFromSegments, syncIngredientsSceneMapFromSubjectMappings, syncLyricMapperFromSegments,
    syncPromptJsonFromSegments,
  };
}
