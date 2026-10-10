import { prettyRefmodName } from "./refmods_viewer_data.mjs";

// Pure helpers for Quick Add RefMod in the Reference Builder: which RefMods belong on the Subjects or Locations tab, and
// the card fields a RefMod fills in. The reference types match REFMOD_FOLDERS_BY_REFERENCE_TYPE in refmod_card.mjs.

export const REFERENCE_TYPE_BY_FOLDER = {
  identity: "character",
  clothing_men: "outfit",
  clothing_women: "outfit",
  background: "environment",
  style: "style",
  object: "object",
  prop: "prop",
  vehicle: "vehicle",
  creature: "creature",
  generic: "other",
  pose_motion: "other",
};

const folderOf = (entry) => String(entry?.folder || entry?.type || "generic");

// Backgrounds become location cards, everything else becomes a subject card.
export function quickAddScopeOf(entry) {
  return folderOf(entry) === "background" ? "location" : "subject";
}

export function entriesForScope(library, scope) {
  return (library || []).filter((entry) => quickAddScopeOf(entry) === scope);
}

// The fields a new card takes from a RefMod, matching what picking the RefMod on a blank card does.
export function cardFieldsFromRefmod(entry) {
  const folder = folderOf(entry);
  const fields = {
    name: prettyRefmodName(entry.name),
    description: String(entry.description || "").trim(),
    source: "refmod",
    refmod: {
      name: entry.name,
      folder: entry.folder,
      type: entry.type,
      kind: entry.kind === "image" ? "image" : "video",
      tokens: entry.tokens,
      frames: entry.frames,
      strength: 1,
    },
  };
  if (quickAddScopeOf(entry) === "subject") {
    fields.reference_type = REFERENCE_TYPE_BY_FOLDER[folder] || "other";
    if (fields.reference_type === "outfit") fields.clothing_set = folder === "clothing_men" ? "men" : "women";
  }
  return fields;
}

// The names of the RefMods the cards already use, so they can be marked as added.
export function usedRefmodNames(refs) {
  const names = new Set();
  for (const list of [refs?.subjects, refs?.extra_subjects, refs?.locations]) {
    for (const card of Array.isArray(list) ? list : []) {
      if (card?.refmod?.name) names.add(card.refmod.name);
    }
  }
  return names;
}
