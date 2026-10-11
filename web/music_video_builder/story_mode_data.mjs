// Story Mode data kept in builder_story_layer: the story book, chapter settings, acts and scene plan.
// Pure functions only. Python twin: storyboard/story_mode.py. Both are checked against tests/story_mode_cases.json.

export const STORY_FLOW_VERSION = 2;
export const STORY_MODE_KEYS = ["story_flow_version", "story_book", "story_book_text", "story_settings", "story_acts", "story_scene_plan"];
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
  return result;
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
