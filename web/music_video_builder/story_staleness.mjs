// Story Builder change tracking. Each step's output is stamped with prints of what it was made from (story_stamps in
// the story layer). When one of those inputs changes, that output is out of date, and so is everything made after it:
// story -> acts -> scene plan -> places -> beats -> prompts. It only runs down that chain: redoing a later step never
// marks an earlier one. Pure functions only.
import {
  normalizeStoryActs, normalizeStoryBook, normalizeStoryScenePlan, normalizeStorySettings, normalizeStoryStamps,
  STORY_STAMP_SECTIONS,
} from "./story_mode_data.mjs";

const PART_REASONS = {
  premise: "premise changed",
  style: "telling style changed",
  cast: "cast changed",
  roles: "roles or relationships changed",
  chapter: "chapter text changed",
  act: "act changed",
  spot: "spot changed",
  setting: "setting look changed",
  plan: "scene plan changed",
  location: "location card changed",
  beat: "beat changed",
};

// A short print of any JSON value (32-bit FNV-1a, hex).
export function storyPrint(value) {
  const text = JSON.stringify(value ?? null);
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

function text(value) {
  return String(value ?? "").trim();
}

function byId(items, key = "id") {
  return Object.fromEntries(items.map((item) => [item[key], item]));
}

// Everything the stamps are compared against, worked out from the project as it is now.
function storyState({ layer, refs, segments, promptKey }) {
  const subjects = (refs?.subjects || []).filter((subject) => subject && !subject.extra_reference_for);
  const subjectById = byId(subjects);
  const castPrint = (ids) => storyPrint([...new Set(ids)].sort().map((id) => {
    const subject = subjectById[id];
    return subject ? [id, text(subject.name), text(subject.description), text(subject.face_description)] : [id];
  }));
  const storyCastIds = layer.story_cast_ids?.length ? layer.story_cast_ids : subjects.map((subject) => subject.id);
  const chapters = normalizeStoryBook(layer.story_book).chapters;
  const chapterById = byId(chapters);
  const settingById = byId(normalizeStorySettings(layer.story_settings));
  const acts = normalizeStoryActs(layer.story_acts);
  const actOfScene = Object.fromEntries(acts.flatMap((act) => act.scene_ids.map((id) => [id, act])));
  const plan = byId(normalizeStoryScenePlan(layer.story_scene_plan), "scene_id");
  const locationById = byId((refs?.locations || []).filter(Boolean));

  const prints = {
    book: {
      premise: storyPrint(text(layer.overall_story_idea)),
      style: storyPrint(layer.story_video_style || "music_video"),
      cast: castPrint(storyCastIds),
      roles: storyPrint([layer.story_main_cast_id || "", layer.story_cast_roles || {}, layer.story_relationships || []]),
    },
    acts: {}, plan: {}, places: {}, beats: {}, prompts: {},
  };
  for (const act of acts) {
    const chapter = chapterById[act.chapter_id];
    prints.acts[act.id] = {
      chapter: storyPrint(chapter ? [chapter.title, chapter.setting_id, chapter.paragraphs] : null),
      cast: castPrint(act.cast_ids.length ? act.cast_ids : storyCastIds),
    };
  }
  for (const segment of segments) {
    const id = segment.id;
    const act = actOfScene[id];
    const entry = plan[id];
    const setting = settingById[chapterById[act?.chapter_id]?.setting_id];
    const location = locationById[refs?.scene_map?.[id]];
    const locationPrint = storyPrint(location ? [text(location.name), text(location.description)] : null);
    const castIds = Array.isArray(refs?.subject_scene_map?.[id]) ? refs.subject_scene_map[id] : [];
    prints.plan[id] = { act: storyPrint(act ? [act.chapter_id, act.title, act.summary, act.cast_ids, act.scene_ids] : null) };
    prints.places[id] = {
      spot: storyPrint(text(entry?.spot).toLowerCase()),
      setting: storyPrint(setting ? [setting.name, setting.look] : null),
    };
    prints.beats[id] = {
      plan: storyPrint(entry ? [entry.story_slice, entry.spot, entry.mood, entry.performance_mode, entry.continues_previous,
        entry.continuation_direction] : null),
      location: locationPrint,
      cast: castPrint(castIds),
    };
    prints.prompts[id] = { beat: storyPrint(text(segment.story_beat)), location: locationPrint, cast: castPrint(castIds) };
  }
  // A print of the beat and prompt themselves (stamp part "out"), so a rewrite made outside the Story Builder (the
  // Storyboard, the inspector, a manual edit) is seen and counts as redone.
  const outputs = {
    beats: Object.fromEntries(segments.map((segment) => [segment.id, storyPrint(text(segment.story_beat))])),
    prompts: Object.fromEntries(segments.map((segment) => [segment.id, storyPrint(text(segment[promptKey]))])),
  };

  // What exists and so can be out of date. Missing outputs are shown as missing, not out of date.
  const exists = {
    book: chapters.length > 0,
    acts: new Set(acts.map((act) => act.id)),
    plan: new Set(segments.filter((segment) => plan[segment.id]?.act_id).map((segment) => segment.id)),
    places: new Set(segments.filter((segment) => plan[segment.id]?.spot && refs?.scene_map?.[segment.id]).map((segment) => segment.id)),
    beats: new Set(segments.filter((segment) => text(segment.story_beat)).map((segment) => segment.id)),
    prompts: new Set(segments.filter((segment) => text(segment[promptKey])).map((segment) => segment.id)),
  };
  return { prints, exists, actOfScene, outputs };
}

// The stamp for one output: the prints of its inputs, plus the print of the output itself for beats and prompts.
function stampFor(state, section, id) {
  const out = state.outputs[section]?.[id];
  return out ? { ...state.prints[section][id], out } : state.prints[section][id];
}

// The parts of a stamp that no longer match. A stamp without a part counts that part as unchanged.
function changedParts(stamp, current) {
  if (!stamp) return [];
  return Object.keys(current).filter((part) => part in stamp && stamp[part] !== current[part]).map((part) => PART_REASONS[part] || part);
}

// What is out of date and why: {book: [reasons], acts|plan|places|beats|prompts: {id: [reasons]}}. Each step is also
// out of date when the step before it is.
export function storyStaleness(input) {
  const { prints, exists, actOfScene } = storyState(input);
  const stamps = normalizeStoryStamps(input.layer.story_stamps);
  const result = { book: exists.book ? changedParts(stamps.book, prints.book) : [] };
  const own = (section, id) => changedParts(stamps[section][id], prints[section][id]);
  result.acts = {};
  for (const id of exists.acts) {
    const reasons = [...own("acts", id), ...(result.book.length ? ["the story is out of date"] : [])];
    if (reasons.length) result.acts[id] = reasons;
  }
  result.plan = {};
  for (const id of exists.plan) {
    const act = actOfScene[id];
    const reasons = [...own("plan", id), ...(act && result.acts[act.id] ? ["its act is out of date"] : [])];
    if (reasons.length) result.plan[id] = reasons;
  }
  result.places = {};
  for (const id of exists.places) {
    const reasons = [...own("places", id), ...(result.plan[id] ? ["its scene plan is out of date"] : [])];
    if (reasons.length) result.places[id] = reasons;
  }
  result.beats = {};
  for (const id of exists.beats) {
    const reasons = [...own("beats", id), ...(result.plan[id] ? ["its scene plan is out of date"] : []),
      ...(result.places[id] ? ["its location is out of date"] : [])];
    if (reasons.length) result.beats[id] = reasons;
  }
  result.prompts = {};
  for (const id of exists.prompts) {
    const reasons = [...own("prompts", id), ...(result.beats[id] ? ["its beat is out of date"] : [])];
    if (reasons.length) result.prompts[id] = reasons;
  }
  return result;
}

// The story layer's stamps with the given steps marked up to date. `marks` is {book: true, acts|...: true | [ids]};
// true means every existing output of that step.
export function stampStory(input, marks) {
  const state = storyState(input);
  const { prints, exists } = state;
  const stamps = normalizeStoryStamps(input.layer.story_stamps);
  if (marks.book && exists.book) stamps.book = prints.book;
  for (const section of STORY_STAMP_SECTIONS) {
    const mark = marks[section];
    if (!mark) continue;
    for (const id of mark === true ? exists[section] : mark) {
      if (exists[section].has(id)) stamps[section][id] = stampFor(state, section, id);
    }
  }
  return stamps;
}

// Stamps every existing output that has no stamp yet (a project from before change tracking, or a new act), drops
// stamps of outputs that are gone, and restamps a beat or prompt that was rewritten since its stamp, wherever that
// happened. Returns null when nothing changed.
export function fillStoryStamps(input) {
  const state = storyState(input);
  const { prints, exists, outputs } = state;
  const stamps = normalizeStoryStamps(input.layer.story_stamps);
  let changed = false;
  if (exists.book && !Object.keys(stamps.book).length) {
    stamps.book = prints.book;
    changed = true;
  }
  for (const section of STORY_STAMP_SECTIONS) {
    for (const id of Object.keys(stamps[section])) {
      if (!exists[section].has(id)) {
        delete stamps[section][id];
        changed = true;
      }
    }
    for (const id of exists[section]) {
      const stamp = stamps[section][id];
      const out = outputs[section]?.[id];
      if (!stamp || (out && stamp.out && stamp.out !== out)) {
        stamps[section][id] = stampFor(state, section, id);
        changed = true;
      } else if (out && !stamp.out) {
        // A stamp from before outputs were printed: remember the output as it is now, without marking it redone.
        stamps[section][id] = { ...stamp, out };
        changed = true;
      }
    }
  }
  return changed ? stamps : null;
}
