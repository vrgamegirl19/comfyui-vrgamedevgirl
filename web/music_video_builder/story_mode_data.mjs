// Story Mode data kept in builder_story_layer: the story book, chapter settings, acts and scene plan.
// Pure functions only. Python twin: storyboard/story_mode.py. Both are checked against tests/story_mode_cases.json.

export const STORY_FLOW_VERSION = 2;
export const STORY_MODE_KEYS = ["story_flow_version", "story_book", "story_book_text", "story_settings", "story_acts", "story_scene_plan",
  "story_spot_min_scenes", "story_cast_ids", "story_allow_new_cast", "story_video_style", "story_continuation_mode",
  "story_stamps", "story_main_cast_id", "story_cast_roles", "story_relationships"];
// A character's part in the story, and how two characters are connected.
export const STORY_CAST_ROLES = ["main", "love_interest", "supporting", "antagonist"];
export const STORY_RELATIONSHIP_TYPES = ["love_interest", "partners", "exes", "family", "friends", "rivals", "strangers", "other"];
// A story layer without the Story Mode fields. The Story Builder owns those, so a layer coming back from the
// Storyboard or the wizard (which can hold an old copy) is merged through this and never replaces them.
export function withoutStoryModeKeys(layer) {
  return Object.fromEntries(Object.entries(layer && typeof layer === "object" ? layer : {})
    .filter(([key]) => !STORY_MODE_KEYS.includes(key)));
}
export const STORY_VIDEO_STYLES =["music_video", "spoken", "story_only"];
export const STORY_CONTINUATION_MODES = ["off", "location", "llm"];
// Story Builder steps whose output is stamped with prints of what it was made from (see story_staleness.mjs).
export const STORY_STAMP_SECTIONS = ["acts", "plan", "places", "beats", "prompts"];
const CONTINUATION_SOURCES = ["auto", "llm", "manual"];
export const DEFAULT_SPOT_MIN_SCENES = 3;
export const PLAN_PERFORMANCE_MODES = ["singing", "speaking", "no_lip_sync", "instrumental"];

const STORY_MAX_CHAPTERS = 40;
const STORY_MAX_PARAGRAPHS = 200;
const STORY_CHAPTER_HEADING = /^\s*(?:#+\s*)?chapter\s+([0-9]+|[a-z]+(?:[\s-][a-z]+)?)\s*(?:[:.\-–—]\s*(.*?))?\s*$/i;
const STORY_META_LINE = /^\s*(title|premise|setting)\s*:\s*(.*?)\s*$/i;

function storyText(value, limit = 4000) {
  return String(value ?? "").replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim().slice(0, limit);
}

function storyLine(value, limit = 200) {
  return storyText(value, limit * 2).replace(/\s+/g, " ").trim().slice(0, limit);
}

function storyIdList(value, limit = 400) {
  return (Array.isArray(value) ? value : []).map((item) => storyLine(item, 160)).filter(Boolean).slice(0, limit);
}

export function storySlug(value) {
  return storyLine(value).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 60) || "item";
}

export function storySettingId(name) {
  return `setting_${storySlug(name)}`;
}

export function normalizeStoryBook(value) {
  const source = value && typeof value === "object" ? value : {};
  const chapters = [];
  (Array.isArray(source.chapters) ? source.chapters : []).forEach((chapter, index) => {
    if (!chapter || typeof chapter !== "object" || chapters.length >= STORY_MAX_CHAPTERS) return;
    const paragraphs = (Array.isArray(chapter.paragraphs) ? chapter.paragraphs : [])
      .map((item) => storyText(item, 6000)).filter(Boolean).slice(0, STORY_MAX_PARAGRAPHS);
    chapters.push({
      id: storyLine(chapter.id, 80) || `ch_${index + 1}`,
      title: storyLine(chapter.title, 200),
      setting_id: storyLine(chapter.setting_id, 120),
      paragraphs,
    });
  });
  return { title: storyLine(source.title, 200), premise: storyText(source.premise, 2000), chapters };
}

export function normalizeStorySettings(value) {
  const settings = [];
  const seen = new Set();
  for (const item of Array.isArray(value) ? value : []) {
    if (!item || typeof item !== "object") continue;
    const name = storyLine(item.name, 160);
    const id = storyLine(item.id, 120) || (name ? storySettingId(name) : "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    settings.push({ id, name: name || id, look: storyText(item.look, 2000) });
  }
  return settings;
}

export function normalizeStoryActs(value) {
  const acts = [];
  (Array.isArray(value) ? value : []).forEach((item, index) => {
    if (!item || typeof item !== "object") return;
    acts.push({
      id: storyLine(item.id, 80) || `act_${index + 1}`,
      chapter_id: storyLine(item.chapter_id, 80),
      title: storyLine(item.title, 200),
      summary: storyText(item.summary, 2000),
      cast_ids: storyIdList(item.cast_ids, 40),
      scene_ids: storyIdList(item.scene_ids),
    });
  });
  return acts;
}

export function normalizePlanPerformance(value) {
  const key = storyLine(value).toLowerCase().replace(/[\s-]+/g, "_");
  if (["speaking", "short_film", "dialogue", "dialog", "spoken"].includes(key)) return "speaking";
  if (["no_lip_sync", "nolipsync", "no_lipsync", "story", "story_only", "visual_only", "silent"].includes(key)) return "no_lip_sync";
  if (["instrumental", "music_only", "no_vocals"].includes(key)) return "instrumental";
  return "singing";
}

export function normalizeStoryScenePlan(value) {
  const plan = [];
  const seen = new Set();
  for (const item of Array.isArray(value) ? value : []) {
    if (!item || typeof item !== "object") continue;
    const sceneId = storyLine(item.scene_id, 160);
    if (!sceneId || seen.has(sceneId)) continue;
    seen.add(sceneId);
    plan.push({
      scene_id: sceneId,
      act_id: storyLine(item.act_id, 80),
      story_slice: storyText(item.story_slice, 1200),
      spot: storyLine(item.spot, 160),
      mood: storyLine(item.mood, 200),
      performance_mode: normalizePlanPerformance(item.performance_mode),
      note_sources: storyIdList(item.note_sources, 12),
      continues_previous: item.continues_previous === true,
      continuation_source: CONTINUATION_SOURCES.includes(storyLine(item.continuation_source, 20).toLowerCase())
        ? storyLine(item.continuation_source, 20).toLowerCase() : "",
      llm_continues: item.llm_continues === true,
      continuation_direction: storyText(item.continuation_direction, 600),
      // Who is in the scene, who sings or speaks, what they do with each other, and who set the cast
      // ("story" or "manual"). user_note is the user's own direction for the scene.
      cast_ids: storyIdList(item.cast_ids, 12),
      performer_id: storyLine(item.performer_id, 160),
      interaction: storyText(item.interaction, 600),
      cast_source: ["story", "manual"].includes(storyLine(item.cast_source, 20).toLowerCase())
        ? storyLine(item.cast_source, 20).toLowerCase() : "",
      user_note: storyText(item.user_note, 1200),
    });
  }
  return plan;
}

// The Story Mode keys of a story layer, normalized. Keys the source does not have are left out.
export function normalizeStoryModeFields(source) {
  const value = source && typeof source === "object" ? source : {};
  const result = {};
  if ("story_flow_version" in value) {
    const version = Math.trunc(Number(value.story_flow_version) || 0);
    result.story_flow_version = version >= STORY_FLOW_VERSION ? STORY_FLOW_VERSION : 0;
  }
  if ("story_book" in value) result.story_book = normalizeStoryBook(value.story_book);
  if ("story_book_text" in value) result.story_book_text = storyText(value.story_book_text, 120000);
  if ("story_settings" in value) result.story_settings = normalizeStorySettings(value.story_settings);
  if ("story_acts" in value) result.story_acts = normalizeStoryActs(value.story_acts);
  if ("story_scene_plan" in value) result.story_scene_plan = normalizeStoryScenePlan(value.story_scene_plan);
  if ("story_spot_min_scenes" in value) result.story_spot_min_scenes = normalizeSpotMinScenes(value.story_spot_min_scenes);
  if ("story_cast_ids" in value) result.story_cast_ids = storyIdList(value.story_cast_ids, 40);
  if ("story_allow_new_cast" in value) {
    result.story_allow_new_cast = value.story_allow_new_cast === true || String(value.story_allow_new_cast).trim().toLowerCase() === "true";
  }
  if ("story_video_style" in value) result.story_video_style = normalizeStoryVideoStyle(value.story_video_style);
  if ("story_continuation_mode" in value) result.story_continuation_mode = normalizeStoryContinuationMode(value.story_continuation_mode);
  if ("story_stamps" in value) result.story_stamps = normalizeStoryStamps(value.story_stamps);
  if ("story_main_cast_id" in value) result.story_main_cast_id = storyLine(value.story_main_cast_id, 160);
  if ("story_cast_roles" in value) result.story_cast_roles = normalizeStoryCastRoles(value.story_cast_roles);
  if ("story_relationships" in value) result.story_relationships = normalizeStoryRelationships(value.story_relationships);
  return result;
}

// {cast id: role} for the story's characters. Unknown roles are dropped.
export function normalizeStoryCastRoles(value) {
  const roles = {};
  for (const [id, role] of Object.entries(plainObject(value)).slice(0, 40)) {
    const key = storyLine(id, 160);
    const name = storyLine(role, 40).toLowerCase();
    if (key && STORY_CAST_ROLES.includes(name)) roles[key] = name;
  }
  return roles;
}

// [{a, b, type, note}]: how two characters are connected. A pair with one person or an unknown type is dropped.
export function normalizeStoryRelationships(value) {
  const seen = new Set();
  const result = [];
  for (const item of (Array.isArray(value) ? value : []).slice(0, 40)) {
    const a = storyLine(item?.a, 160);
    const b = storyLine(item?.b, 160);
    const type = storyLine(item?.type, 40).toLowerCase();
    const pair = [a, b].sort().join("|");
    if (!a || !b || a === b || !STORY_RELATIONSHIP_TYPES.includes(type) || seen.has(pair)) continue;
    seen.add(pair);
    result.push({ a, b, type, note: storyLine(item?.note, 300) });
  }
  return result;
}

function plainObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function stampParts(value) {
  const parts = {};
  for (const [key, print] of Object.entries(plainObject(value)).slice(0, 12)) {
    const name = storyLine(key, 40);
    const text = storyLine(print, 64);
    if (name && text) parts[name] = text;
  }
  return parts;
}

// {book: {part: print}, acts|plan|places|beats|prompts: {id: {part: print}}}: what each step's output was made from.
export function normalizeStoryStamps(value) {
  const source = plainObject(value);
  const result = { book: stampParts(source.book) };
  for (const section of STORY_STAMP_SECTIONS) {
    const items = {};
    for (const [id, item] of Object.entries(plainObject(source[section])).slice(0, 2000)) {
      const key = storyLine(id, 160);
      const parts = stampParts(item);
      if (key && Object.keys(parts).length) items[key] = parts;
    }
    result[section] = items;
  }
  return result;
}

// "off", "location" or "llm"; anything else is "off".
export function normalizeStoryContinuationMode(value) {
  const text = storyLine(value).toLowerCase().replace(/[\s-]+/g, "_");
  return STORY_CONTINUATION_MODES.includes(text) ? text : "off";
}

// The scene plan with continues_previous set for every scene. `scenes` is the timeline in order ({id, start, end}).
// A manual choice is kept. Otherwise "location" continues a scene that touches the scene before at the same spot in
// the same chapter, "llm" uses the LLM's answer and "off" continues nothing. The first scene, and a scene with a gap
// before it, never continue.
export function resolveStoryContinuations(plan, scenes, acts, mode) {
  const resolvedMode = normalizeStoryContinuationMode(mode);
  const normalized = normalizeStoryScenePlan(plan);
  const entries = Object.fromEntries(normalized.map((entry) => [entry.scene_id, { ...entry }]));
  const chapterOf = {};
  for (const act of normalizeStoryActs(acts)) for (const id of act.scene_ids) chapterOf[id] = act.chapter_id;
  const seconds = (value) => (Number.isFinite(Number(value)) && value !== null && value !== "" ? Number(value) : 0);
  let previous = null;
  for (const scene of Array.isArray(scenes) ? scenes : []) {
    const id = String(scene?.id || "");
    const entry = entries[id];
    if (entry) {
      const before = previous ? entries[String(previous.id || "")] : null;
      const touches = previous !== null && Math.abs(seconds(scene.start) - seconds(previous.end)) <= 0.05;
      if (!touches) {
        entry.continues_previous = false;
        entry.continuation_source = "";
      } else if (entry.continuation_source !== "manual") {
        const sameSpot = Boolean(before && entry.spot && entry.spot.toLowerCase() === before.spot.toLowerCase()
          && chapterOf[id] && chapterOf[id] === chapterOf[before.scene_id]);
        if (resolvedMode === "location" && sameSpot) Object.assign(entry, { continues_previous: true, continuation_source: "auto" });
        else if (resolvedMode === "llm" && entry.llm_continues) Object.assign(entry, { continues_previous: true, continuation_source: "llm" });
        else Object.assign(entry, { continues_previous: false, continuation_source: "" });
      }
    }
    previous = scene || {};
  }
  return normalized.map((entry) => entries[entry.scene_id]);
}

// "music_video", "spoken" or "story_only"; anything else is "music_video".
export function normalizeStoryVideoStyle(value) {
  const text = storyLine(value).toLowerCase().replace(/[\s-]+/g, "_");
  return STORY_VIDEO_STYLES.includes(text) ? text : "music_video";
}

export function normalizeSpotMinScenes(value) {
  const number = Math.trunc(Number(value));
  return Number.isFinite(number) && value !== "" && value !== null ? Math.max(1, Math.min(20, number)) : DEFAULT_SPOT_MIN_SCENES;
}

// Timeline segment fields for a scene plan performance. Story-only and instrumental scenes are no lip sync.
export function planSegmentFields(performanceMode) {
  const mode = normalizePlanPerformance(performanceMode);
  if (mode === "no_lip_sync" || mode === "instrumental") return { performance_mode: "no_lip_sync", lyric_no_lip_sync: true };
  return { performance_mode: mode, lyric_no_lip_sync: false };
}

export function isStoryModeLayer(layer) {
  return Number(layer?.story_flow_version || 0) >= STORY_FLOW_VERSION;
}

export function splitParagraphs(value) {
  const body = storyText(value, 200000);
  if (!body) return [];
  const blocks = /\n\s*\n/.test(body) ? body.split(/\n\s*\n/) : body.split("\n");
  return blocks.map((block) => block.trim()).filter(Boolean);
}

export function parseStoryBookText(value) {
  const lines = storyText(value, 200000).split("\n");
  let title = "";
  let premise = "";
  const chapters = [];
  const settingNames = {};
  let current = null;
  let buffer = [];
  const preamble = [];
  const flush = () => {
    if (current) {
      current.paragraphs = splitParagraphs(buffer.join("\n"));
      chapters.push(current);
    }
  };
  for (const raw of lines) {
    const heading = STORY_CHAPTER_HEADING.exec(raw);
    if (heading) {
      flush();
      buffer = [];
      current = { id: `ch_${chapters.length + 1}`, title: storyLine(heading[2], 200) || `Chapter ${heading[1]}`, setting_id: "", paragraphs: [] };
      continue;
    }
    const meta = STORY_META_LINE.exec(raw);
    if (meta) {
      const key = meta[1].toLowerCase();
      const metaValue = meta[2];
      if (key === "setting" && current && !current.setting_id && !buffer.join("").trim()) {
        if (storyLine(metaValue)) {
          current.setting_id = storySettingId(metaValue);
          if (!(current.setting_id in settingNames)) settingNames[current.setting_id] = storyLine(metaValue, 160);
        }
        continue;
      }
      if (!current && key === "title" && !title) { title = storyLine(metaValue, 200); continue; }
      if (!current && key === "premise" && !premise) { premise = storyText(metaValue, 2000); continue; }
    }
    if (current) buffer.push(raw);
    else preamble.push(raw);
  }
  flush();
  if (!chapters.length) {
    const paragraphs = splitParagraphs(preamble.join("\n"));
    if (paragraphs.length) chapters.push({ id: "ch_1", title: "Chapter 1", setting_id: "", paragraphs });
  }
  return { book: normalizeStoryBook({ title, premise, chapters }), settingNames };
}

export function storyBookToText(book, settings = []) {
  const normalized = normalizeStoryBook(book);
  const names = Object.fromEntries(normalizeStorySettings(settings).map((item) => [item.id, item.name]));
  const head = [
    normalized.title ? `Title: ${normalized.title}` : "",
    normalized.premise ? `Premise: ${normalized.premise}` : "",
  ].filter(Boolean).join("\n");
  const blocks = head ? [head] : [];
  normalized.chapters.forEach((chapter, index) => {
    let heading = `Chapter ${index + 1}: ${chapter.title || `Chapter ${index + 1}`}`;
    const settingName = names[chapter.setting_id] || "";
    if (settingName) heading += `\nSetting: ${settingName}`;
    blocks.push([heading, ...chapter.paragraphs].join("\n\n"));
  });
  return blocks.join("\n\n");
}

export function syncStorySettings(book, settingNames = {}, existing = []) {
  const current = Object.fromEntries(normalizeStorySettings(existing).map((item) => [item.id, item]));
  const used = [];
  for (const chapter of normalizeStoryBook(book).chapters) {
    if (chapter.setting_id && !used.includes(chapter.setting_id)) used.push(chapter.setting_id);
  }
  return used.map((id) => {
    const item = { ...(current[id] || { id, name: "", look: "" }) };
    item.name = settingNames[id] || item.name || id;
    return item;
  });
}

// Story layer fields after the user edits the book text: book, text and settings updated together.
export function applyStoryBookText(layer, value) {
  const { book, settingNames } = parseStoryBookText(value);
  const settings = syncStorySettings(book, settingNames, layer?.story_settings);
  return {
    story_flow_version: STORY_FLOW_VERSION,
    story_book: book,
    story_book_text: storyBookToText(book, settings),
    story_settings: settings,
  };
}

// Characters for scenes that have none: each act's cast goes onto its scenes. User mappings are kept.
export function fillSceneCast(subjectSceneMap, acts, subjectIds) {
  const result = {};
  for (const [key, value] of Object.entries(subjectSceneMap || {})) result[key] = Array.isArray(value) ? [...value] : value;
  const valid = new Set(subjectIds);
  const changed = [];
  for (const act of normalizeStoryActs(acts)) {
    const cast = act.cast_ids.filter((id) => valid.has(id));
    if (!cast.length) continue;
    for (const sceneId of act.scene_ids) {
      const current = result[sceneId];
      if ((Array.isArray(current) && current.length) || (typeof current === "string" && current.trim())) continue;
      result[sceneId] = [...cast];
      changed.push(sceneId);
    }
  }
  return { map: result, changed };
}

// Each planned scene's characters (cast_ids) become its mapping; scenes the plan gives no cast keep theirs, or get
// their act's cast when they have none (fillSceneCast). Python twin: apply_scene_cast.
export function applySceneCast(subjectSceneMap, plan, acts, subjectIds) {
  const result = {};
  for (const [key, value] of Object.entries(subjectSceneMap || {})) result[key] = Array.isArray(value) ? [...value] : value;
  const valid = new Set(subjectIds);
  const changed = [];
  for (const entry of normalizeStoryScenePlan(plan)) {
    const cast = entry.cast_ids.filter((id) => valid.has(id));
    if (cast.length && JSON.stringify(result[entry.scene_id]) !== JSON.stringify(cast)) {
      result[entry.scene_id] = cast;
      changed.push(entry.scene_id);
    }
  }
  const filled = fillSceneCast(result, acts, subjectIds);
  return { map: filled.map, changed: [...changed, ...filled.changed.filter((id) => !changed.includes(id))] };
}

// One colour per chapter, used by the timeline stripe and the Story Builder act band.
export const STORY_CHAPTER_COLORS = ["#0e7490", "#7c3aed", "#b45309", "#15803d", "#be123c", "#1d4ed8", "#a16207"];

// Scene id -> { chapterIndex, chapterTitle, actTitle, spot } for the scenes that are in an act.
export function storySceneIndex(layer) {
  const chapters = normalizeStoryBook(layer?.story_book).chapters;
  const chapterIndex = Object.fromEntries(chapters.map((chapter, index) => [chapter.id, index]));
  const spots = Object.fromEntries(normalizeStoryScenePlan(layer?.story_scene_plan).map((entry) => [entry.scene_id, entry.spot]));
  const index = {};
  for (const act of normalizeStoryActs(layer?.story_acts)) {
    if (!(act.chapter_id in chapterIndex)) continue;
    const chapter = chapters[chapterIndex[act.chapter_id]];
    for (const sceneId of act.scene_ids) {
      index[sceneId] = { chapterIndex: chapterIndex[act.chapter_id], chapterTitle: chapter.title, actTitle: act.title, spot: spots[sceneId] || "" };
    }
  }
  return index;
}
