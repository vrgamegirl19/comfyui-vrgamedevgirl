// Story Builder (Story Mode) tab order, progress and readiness. Pure functions only; tested by tests/story_builder_data.mjs.
import { validateStoryActs } from "./story_acts_data.mjs";
import { normalizeStoryBook, normalizeStoryScenePlan } from "./story_mode_data.mjs";

export const STORY_BUILDER_TABS = [
  { id: "song", label: "1. Song", caption: "Lyrics, scenes, timed notes" },
  { id: "cast", label: "2. Cast", caption: "Characters and extras" },
  { id: "story", label: "3. Story", caption: "The book" },
  { id: "places", label: "4. Places", caption: "Your locations and settings" },
  { id: "acts", label: "5. Acts", caption: "Chapters into acts" },
  { id: "scenes", label: "6. Scenes", caption: "Each scene's part" },
  { id: "prompts", label: "7. Beats & Prompts", caption: "Storyboard" },
];

function hasImage(card) {
  return Boolean(card?.image?.path || card?.image?.data || card?.refmod?.name || card?.source === "refmod");
}

function characters(refs) {
  return (refs?.subjects || []).filter((subject) => subject && !subject.extra_reference_for
    && (subject.reference_type || "character") === "character");
}

// "empty", "progress" or "done" for every tab.
export function storyBuilderProgress({ sceneIds = [], refs = {}, layer = {}, segments = [] } = {}) {
  const chapters = normalizeStoryBook(layer.story_book).chapters;
  const acts = Array.isArray(layer.story_acts) ? layer.story_acts : [];
  const plan = normalizeStoryScenePlan(layer.story_scene_plan);
  const planned = new Set(plan.map((entry) => entry.scene_id));
  const cast = characters(refs);
  const sceneMap = refs?.scene_map || {};
  // A story location card is linked to a chapter setting, or is one of the places the user listed.
  const userPlaces = new Set(String(layer?.story_user_locations || "").split("\n")
    .map((line) => line.split(":")[0].replace(/\s+/g, " ").replace(/^[\s*-]+|[\s*-]+$/g, "").toLowerCase()).filter(Boolean));
  const storyCards = new Set((refs?.locations || [])
    .filter((card) => card?.story_setting_id || userPlaces.has(String(card?.name || "").trim().toLowerCase()))
    .map((card) => card.id));
  const mappedToStory = sceneIds.filter((id) => storyCards.has(sceneMap[id])).length;
  const withBeat = segments.filter((segment) => String(segment?.story_beat || "").trim()).length;
  const level = (count, total) => (!count ? "empty" : count >= total ? "done" : "progress");
  return {
    song: sceneIds.length ? "done" : "empty",
    cast: !cast.length ? "empty" : cast.every(hasImage) ? "done" : "progress",
    story: chapters.some((chapter) => chapter.paragraphs.length) ? "done" : "empty",
    acts: !acts.length ? "empty" : validateStoryActs(acts, chapters, sceneIds).length ? "progress" : "done",
    scenes: level(sceneIds.filter((id) => planned.has(id)).length, sceneIds.length || 1),
    // Location images are optional. Cards made from the user's list come before the plan, so they count as done
    // until scenes are planned; after that every scene must be mapped to one.
    places: !storyCards.size ? "empty" : !mappedToStory && !planned.size ? "done" : mappedToStory >= (sceneIds.length || 1) ? "done" : "progress",
    prompts: level(withBeat, segments.length || 1),
  };
}

// Why a tab's main action cannot run yet, or "" when it can.
export function storyBuilderBlocker(tabId, progress) {
  const needs = {
    cast: [["song", "Create the timeline scenes first (Song tab)."]],
    story: [["song", "Create the timeline scenes first (Song tab)."]],
    places: [["story", "Write the story first (Story tab)."]],
    acts: [["story", "Write the story first (Story tab)."], ["places", "Build the places first (Places tab); the acts happen inside them."]],
    scenes: [["acts", "Create valid acts first (Acts tab)."]],
    prompts: [["scenes", "Plan the scenes first (Scenes tab)."]],
  }[tabId] || [];
  for (const [tab, message] of needs) {
    // A project that already has acts is not stopped from rebuilding them because it has no place cards.
    if (tabId === "acts" && tab === "places" && progress.acts !== "empty") continue;
    if (tab === "acts" ? progress[tab] !== "done" : progress[tab] === "empty") return message;
  }
  return "";
}

// Characters that still need an image or RefMod before rendering.
export function castWithoutImages(refs) {
  return characters(refs).filter((subject) => !hasImage(subject)).map((subject) => subject.name || subject.id);
}

// Cards used by the scenes that still have no image, and the scenes (1-based) with no image at all to send.
export function referenceImageGaps(refs, sceneIds = []) {
  const subjects = new Map((refs?.subjects || []).filter((card) => card && !card.extra_reference_for).map((card) => [card.id, card]));
  const locations = new Map((refs?.locations || []).filter(Boolean).map((card) => [card.id, card]));
  const subjectMap = refs?.subject_scene_map || {};
  const sceneMap = refs?.scene_map || {};
  const missingCharacters = new Set();
  const missingLocations = new Set();
  const scenes = [];
  sceneIds.forEach((id, index) => {
    const cast = (Array.isArray(subjectMap[id]) ? subjectMap[id] : [subjectMap[id]]).map((key) => subjects.get(key)).filter(Boolean);
    const place = locations.get(sceneMap[id]);
    for (const card of cast) if (!hasImage(card)) missingCharacters.add(card.name || card.id);
    if (place && !hasImage(place)) missingLocations.add(place.name || place.id);
    if (!cast.some(hasImage) && !(place && hasImage(place))) scenes.push(index + 1);
  });
  return { characters: [...missingCharacters], locations: [...missingLocations], scenes };
}

// Location cards used by scenes that lost their story link (story_setting_id) are linked again to the chapter setting
// they are named after ("Home - Kitchen" or "Home"). Returns the repaired Reference Builder data, or null when nothing
// needed it.
export function relinkStoryLocations(refs, settings = []) {
  const used = new Set(Object.values(refs?.scene_map || {}).map(String));
  const named = (settings || []).filter((setting) => setting?.id && String(setting.name || "").trim())
    .map((setting) => ({ id: setting.id, name: String(setting.name).trim().toLowerCase() }))
    .sort((a, b) => b.name.length - a.name.length);
  let changed = false;
  const locations = (refs?.locations || []).map((card) => {
    if (!card || card.story_setting_id || !used.has(String(card.id))) return card;
    const name = String(card.name || "").trim().toLowerCase();
    const setting = named.find((item) => name === item.name || name.startsWith(`${item.name} - `));
    if (!setting) return card;
    changed = true;
    return { ...card, story_setting_id: setting.id, story_act_ids: Array.isArray(card.story_act_ids) ? card.story_act_ids : [] };
  });
  return changed ? { ...refs, locations } : null;
}

// "Kitchen" from the card "Home - Kitchen": the card name without the "Setting - " prefix of the setting it was built
// for. This is how a scene's plan spot is written. Python twin: made_place_cards (storyboard/story_book.py).
export function spotNameOfCard(card, settings = []) {
  const name = String(card?.name || "").replace(/\s+/g, " ").trim();
  const setting = (settings || []).find((item) => item?.id && item.id === card?.story_setting_id);
  const prefix = setting?.name ? `${String(setting.name).trim()} - ` : "";
  return prefix && name.toLowerCase().startsWith(prefix.toLowerCase()) ? name.slice(prefix.length) : name;
}

const spotWords = (value) => new Set(String(value || "").toLowerCase().replace(/[^a-z0-9' ]+/g, " ").split(/\s+/)
  .filter((word) => word && word !== "the"));
const wordsWithin = (small, big) => [...small].every((word) => big.has(word));

// The location card a plan spot names: the card name or spot name without case, else every word of the shorter name is
// a word of the longer one ("Kitchen" is "Grandma's kitchen"). Python twin: match_user_place (storyboard/story_book.py).
export function cardForSpot(cards, spot, settings = []) {
  const wanted = String(spot || "").replace(/\s+/g, " ").trim().toLowerCase();
  if (!wanted) return null;
  const list = (cards || []).filter((card) => card?.id && String(card.name || "").trim());
  const exact = list.find((card) => String(card.name).replace(/\s+/g, " ").trim().toLowerCase() === wanted
    || spotNameOfCard(card, settings).toLowerCase() === wanted);
  if (exact) return exact;
  const target = spotWords(wanted);
  if (!target.size) return null;
  return list.find((card) => {
    const own = spotWords(spotNameOfCard(card, settings));
    return own.size && (wordsWithin(own, target) || wordsWithin(target, own));
  }) || null;
}

// The scene plan with each scene's spot following the location card it is mapped to (Auto Map and the mapping boxes
// change the map, not the plan). Returns null when no spot changes.
export function planSpotsFromMapping(plan, refs, settings = []) {
  const cards = new Map((refs?.locations || []).filter((card) => card?.id).map((card) => [card.id, card]));
  let changed = false;
  const next = (plan || []).map((entry) => {
    const card = cards.get(refs?.scene_map?.[entry.scene_id]);
    const spot = card ? spotNameOfCard(card, settings) : "";
    if (!spot || spot === entry.spot) return entry;
    changed = true;
    return { ...entry, spot };
  });
  return changed ? next : null;
}

// "Scenes 48-50, 52" from [48, 49, 50, 52].
export function sceneRangeText(numbers) {
  const parts = [];
  for (const number of numbers) {
    const last = parts[parts.length - 1];
    if (last && number === last[1] + 1) last[1] = number;
    else parts.push([number, number]);
  }
  const text = parts.map(([start, end]) => (start === end ? `${start}` : `${start}-${end}`)).join(", ");
  return `${numbers.length === 1 ? "Scene" : "Scenes"} ${text}`;
}

// A Reference Builder subject card for a character found in the story.
export function storySubjectCard(character, id) {
  return {
    id,
    name: String(character?.name || "Character").trim(),
    description: String(character?.description || "").trim(),
    reference_type: "character",
    image: { path: "", data: "", name: "" },
  };
}
