// Story Mode locations: merge the story's spot cards into the Reference Builder and map their scenes.
// Pure functions only. Python twin: storyboard/story_locations.py. Both are checked against tests/story_locations_cases.json.

// "Home - Kitchen"; just the setting when the spot is the setting itself.
export function storyCardName(settingName, spot) {
  const setting = String(settingName || "").trim();
  const place = String(spot || "").trim();
  if (!place || place.toLowerCase() === setting.toLowerCase()) return setting || place;
  return setting ? `${setting} - ${place}` : place;
}

// Reference Builder data with the story's cards merged in. A card with the same name is reused: its description is
// replaced only when empty (or with replaceDescriptions) and its image is never touched.
export function applyStoryLocations(refs, locations, newId, replaceDescriptions = false) {
  const result = JSON.parse(JSON.stringify(refs && typeof refs === "object" ? refs : {}));
  if (!Array.isArray(result.locations)) result.locations = [];
  const cards = result.locations;
  const byName = new Map(cards.filter((card) => card && typeof card === "object")
    .map((card) => [String(card.name || "").trim().toLowerCase(), card]));
  const taken = new Set(cards.map((card) => card?.id));
  const sceneMap = result.scene_map && typeof result.scene_map === "object" && !Array.isArray(result.scene_map) ? result.scene_map : {};
  result.scene_map = sceneMap;
  let added = 0;
  let updated = 0;
  for (const item of locations || []) {
    const name = String(item?.name || "").trim();
    if (!name) continue;
    let card = byName.get(name.toLowerCase());
    if (!card) {
      card = { id: newId(taken), name, description: item.description || "", image: { path: "", data: "", name: "" } };
      cards.push(card);
      byName.set(name.toLowerCase(), card);
      added += 1;
    } else if (item.description && (replaceDescriptions || !String(card.description || "").trim())) {
      card.description = item.description;
      updated += 1;
    }
    card.story_setting_id = item.story_setting_id || "";
    card.story_act_ids = [...(item.story_act_ids || [])];
    for (const sceneId of item.scene_ids || []) sceneMap[sceneId] = card.id;
  }
  result.locations_cleared = false;
  result.use_location_references = cards.length > 0;
  return { refs: result, added, updated };
}
