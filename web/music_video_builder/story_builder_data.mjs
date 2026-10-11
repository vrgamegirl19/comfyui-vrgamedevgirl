// Story Builder (Story Mode) tab order, progress and readiness. Pure functions only; tested by tests/story_builder_data.mjs.
import { validateStoryActs } from "./story_acts_data.mjs";
import { normalizeStoryBook, normalizeStoryScenePlan } from "./story_mode_data.mjs";

export const STORY_BUILDER_TABS = [
  { id: "song", label: "1. Song", caption: "Lyrics, scenes, timed notes" },
  { id: "cast", label: "2. Cast", caption: "Characters and extras" },
  { id: "story", label: "3. Story", caption: "The book" },
  { id: "acts", label: "4. Acts", caption: "Chapters into acts" },
  { id: "scenes", label: "5. Scenes", caption: "Each scene's part" },
  { id: "places", label: "6. Places", caption: "Settings and spots" },
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
  const storyCards = new Set((refs?.locations || []).filter((card) => card?.story_setting_id).map((card) => card.id));
  const mappedToStory = sceneIds.filter((id) => storyCards.has(sceneMap[id])).length;
  const withBeat = segments.filter((segment) => String(segment?.story_beat || "").trim()).length;
  const level = (count, total) => (!count ? "empty" : count >= total ? "done" : "progress");
  return {
    song: sceneIds.length ? "done" : "empty",
    cast: !cast.length ? "empty" : cast.every(hasImage) ? "done" : "progress",
    story: chapters.some((chapter) => chapter.paragraphs.length) ? "done" : "empty",
    acts: !acts.length ? "empty" : validateStoryActs(acts, chapters, sceneIds).length ? "progress" : "done",
    scenes: level(sceneIds.filter((id) => planned.has(id)).length, sceneIds.length || 1),
    places: level(mappedToStory, sceneIds.length || 1),
    prompts: level(withBeat, segments.length || 1),
  };
}

// Why a tab's main action cannot run yet, or "" when it can.
export function storyBuilderBlocker(tabId, progress) {
  const needs = {
    cast: [["song", "Create the timeline scenes first (Song tab)."]],
    story: [["song", "Create the timeline scenes first (Song tab)."]],
    acts: [["story", "Write the story first (Story tab)."]],
    scenes: [["acts", "Create valid acts first (Acts tab)."]],
    places: [["scenes", "Plan the scenes first (Scenes tab)."]],
    prompts: [["scenes", "Plan the scenes first (Scenes tab)."]],
  }[tabId] || [];
  for (const [tab, message] of needs) {
    if (tab === "acts" ? progress[tab] !== "done" : progress[tab] === "empty") return message;
  }
  return "";
}

// Characters that still need an image or RefMod before rendering.
export function castWithoutImages(refs) {
  return characters(refs).filter((subject) => !hasImage(subject)).map((subject) => subject.name || subject.id);
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
