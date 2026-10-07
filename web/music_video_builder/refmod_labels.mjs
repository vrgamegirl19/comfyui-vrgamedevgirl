// Which RefMods a scene uses, their order, and the labels the text encoder gives them.
// Pure functions with no imports, so they can be tested with Node.
// The Python twin is minimax/refmod_scene.py. Both are checked against tests/refmod_scene_cases.json.
//
// A RefMod reference is a Reference Builder card with source "refmod" and card.refmod = { name, kind, tokens, strength }.
// The Text Encode with RefMods node numbers visual mods per kind in loader order: <Picture n> for single-frame mods,
// <Video n> for mods with several frames, and <Audio n> for audio. Mods with strength 0 are skipped.

export const KIND_LABELS = { image: "Picture", video: "Video", audio: "Audio" };
export const CATEGORY_RANK = { character: 0, extra: 1, clothing: 2, object: 3, background: 4, style: 5 };

const text = (value) => String(value ?? "").trim();

export function isRefmodCard(card) {
  return Boolean(card && typeof card === "object" && text(card.source) === "refmod"
    && card.refmod && typeof card.refmod === "object" && text(card.refmod.name));
}

// The RefMod part of a card, cleaned, for places that rebuild cards field by field (the storyboard catalog). Returns {}
// for cards that are not RefMods. Twin of _refmod_card_fields in storyboard/scene_helpers.py.
export function refmodCardFields(card) {
  if (!isRefmodCard(card)) return {};
  const refmod = card.refmod;
  const number = (value, fallback) => (Number.isFinite(Number(value)) ? Number(value) : fallback);
  const fields = {
    source: "refmod",
    refmod: {
      name: text(refmod.name), folder: text(refmod.folder), type: text(refmod.type),
      kind: text(refmod.kind) === "image" ? "image" : "video",
      tokens: Math.trunc(number(refmod.tokens, 0)), frames: Math.trunc(number(refmod.frames, 1)),
      strength: Math.min(1, Math.max(0, number(refmod.strength ?? 1, 1))),
    },
  };
  if (text(card.reference_type)) fields.reference_type = text(card.reference_type);
  if (text(card.wears)) fields.wears = text(card.wears);
  if (text(card.clothing_set)) fields.clothing_set = text(card.clothing_set);
  if (card.follow === false) fields.follow = false;
  return fields;
}

export function cardCategory(card, kind = "subject") {
  if (kind === "extra") return "extra";
  if (kind === "location") return "background";
  const referenceType = text(card?.reference_type) || "character";
  if (referenceType === "character") return "character";
  if (referenceType === "outfit") return "clothing";
  if (referenceType === "environment") return "background";
  if (referenceType === "style") return "style";
  return "object";
}

function strengthOf(card) {
  const value = Number(card.refmod.strength ?? 1);
  if (!Number.isFinite(value)) return 1;
  return Math.min(1, Math.max(0, value));
}

function itemFor(card, kind) {
  const refmod = card.refmod;
  return {
    key: `${kind}:${text(card.id)}`,
    card_id: text(card.id),
    name: text(card.name) || text(refmod.name),
    category: cardCategory(card, kind),
    reference_type: text(card.reference_type) || (kind === "location" ? "environment" : "character"),
    mod_name: text(refmod.name),
    kind: text(refmod.kind) || "video",
    strength: strengthOf(card),
    tokens: Math.trunc(Number(refmod.tokens) || 0),
    description: text(card.description),
    wears: text(card.wears),
  };
}

function splitIds(value) {
  if (Array.isArray(value)) return value.map(text).filter(Boolean);
  return String(value ?? "").split(",").map((item) => item.trim()).filter(Boolean);
}

// Stable order: category rank, then (for clothing) the position of the character who wears it.
export function orderItems(items) {
  const characters = items.filter((item) => item.category === "character").map((item) => item.card_id);
  const keyed = items.map((item, position) => {
    const wearer = item.category === "clothing" && characters.includes(item.wears) ? characters.indexOf(item.wears) : characters.length;
    return { item, position, rank: CATEGORY_RANK[item.category] ?? 3, wearer: item.category === "clothing" ? wearer : 0 };
  });
  keyed.sort((a, b) => (a.rank - b.rank) || (a.wearer - b.wearer) || (a.position - b.position));
  return keyed.map((entry) => entry.item);
}

// Turn the cards a scene uses into ordered RefMod items.
//   subjectCards: the subject cards the scene maps to, in map order
//   extraCards:   extras sent to MiniMax
//   locationCard: the scene's location card or null
//   allSubjects:  every subject card (clothing cards are found here)
//   override:     { characterId: clothing card id(s) } chosen for this scene ("" means none)
// A clothing card tied to a character (wears) follows that character into the scene unless it has follow: false or the
// override chose other clothing. An override names its cards whatever their follow setting.
export function composeRefmodItems(subjectCards, extraCards, locationCard, allSubjects, override = {}) {
  const chosenFor = override && typeof override === "object" ? override : {};
  let items = [];
  const seen = new Set();
  const subjects = (allSubjects || []).filter((card) => card && typeof card === "object");
  const add = (card, kind) => {
    if (!isRefmodCard(card)) return;
    const key = `${kind}:${text(card.id)}`;
    if (seen.has(key)) return;
    seen.add(key);
    items.push(itemFor(card, kind));
  };
  for (const card of subjectCards || []) add(card, "subject");
  for (const card of extraCards || []) add(card, "extra");
  const characterIds = items.filter((item) => item.category === "character").map((item) => item.card_id);
  for (const characterId of characterIds) {
    if (Object.prototype.hasOwnProperty.call(chosenFor, characterId)) {
      const chosen = new Set(splitIds(chosenFor[characterId]));
      items = items.filter((item) => !(item.category === "clothing" && item.wears === characterId && !chosen.has(item.card_id)));
      for (const card of subjects) {
        if (chosen.has(text(card.id))) add(card, "subject");
      }
    } else {
      for (const card of subjects) {
        if (text(card.wears) === characterId && cardCategory(card) === "clothing" && card.follow !== false) add(card, "subject");
      }
    }
  }
  add(locationCard, "location");
  return orderItems(items);
}

// Add a label (for example "<Video 2>") to each item the way Text Encode with RefMods numbers them. Items with strength 0
// get an empty label. With includeAudio an entry for the scene audio mod ("<Audio 1>") is appended.
export function assignLabels(items, includeAudio = false) {
  const counters = { image: 0, video: 0, audio: 0 };
  const labelled = items.map((item) => {
    const kind = item.kind === "image" || item.kind === "video" ? item.kind : "video";
    if (item.strength <= 0) return { ...item, label: "" };
    counters[kind] += 1;
    return { ...item, label: `<${KIND_LABELS[kind]} ${counters[kind]}>` };
  });
  if (includeAudio) {
    counters.audio += 1;
    labelled.push({
      key: "audio:scene", card_id: "", name: "Scene audio", category: "audio", reference_type: "audio", mod_name: "",
      kind: "audio", strength: 1, tokens: 0, description: "", wears: "", label: `<Audio ${counters.audio}>`,
    });
  }
  return labelled;
}

export function totalTokens(items) {
  return items.filter((item) => item.strength > 0).reduce((sum, item) => sum + (Number(item.tokens) || 0), 0);
}

// What the scene's clothing control shows: one row per character in the scene, with every saved-RefMod clothing card as
// a choice. "current" is the clothing the scene uses for that character now ("" = none), "overridden" says whether the
// scene overrides the card defaults. items: the scene's composed items. override: segment.refmod_clothing_override.
export function clothingChoices(items, allSubjects, override) {
  const chosenFor = override && typeof override === "object" ? override : {};
  const clothing = (allSubjects || []).filter((card) => isRefmodCard(card) && cardCategory(card) === "clothing");
  if (!clothing.length) return [];
  return (items || []).filter((item) => item.category === "character").map((character) => {
    const overridden = Object.prototype.hasOwnProperty.call(chosenFor, character.card_id);
    const chosen = new Set(overridden ? splitIds(chosenFor[character.card_id]) : []);
    const worn = items.find((item) => item.category === "clothing" && (overridden ? chosen.has(item.card_id) : item.wears === character.card_id));
    return {
      character_id: character.card_id,
      character: character.name,
      current: worn ? worn.card_id : "",
      overridden,
      options: clothing.map((card) => ({ id: text(card.id), name: text(card.name) || "Clothing" })),
    };
  });
}

export const TOKEN_WARN_LIMIT = 6000;
export const IMBALANCE_RATIO = 2;

// Scene token total and character balance. Twin of token_report in minimax/refmod_scene.py. A character weighing less
// than half of the strongest one (tokens x strength) tends to be duplicated or replaced by the stronger ones.
export function tokenReport(items) {
  const active = (items || []).filter((item) => item.strength > 0);
  const total = active.reduce((sum, item) => sum + (Number(item.tokens) || 0), 0);
  const people = active
    .filter((item) => (item.category === "character" || item.category === "extra") && Number(item.tokens) > 0)
    .map((item) => ({ weight: Number(item.tokens) * Number(item.strength), item }));
  let imbalance = null;
  if (people.length >= 2) {
    const weak = people.reduce((best, entry) => (entry.weight < best.weight ? entry : best));
    const strong = people.reduce((best, entry) => (entry.weight > best.weight ? entry : best));
    if (weak.weight > 0 && strong.weight / weak.weight > IMBALANCE_RATIO) {
      imbalance = { weak: weak.item.name, strong: strong.item.name, ratio: Math.round((strong.weight / weak.weight) * 10) / 10 };
    }
  }
  return { total, limit: TOKEN_WARN_LIMIT, over_limit: total > TOKEN_WARN_LIMIT, imbalance };
}

// One short line for the scene status: the token total plus any warning. Empty when there is nothing to show.
export function tokenStatusText(items) {
  if (!(items || []).some((item) => item.strength > 0)) return "";
  const report = tokenReport(items);
  let line = `RefMods: ${report.total.toLocaleString()} tokens.`;
  if (report.over_limit) line += ` Over ${report.limit.toLocaleString()}: render time and memory rise, lower a strength or use lighter RefMods.`;
  if (report.imbalance) {
    line += ` ${report.imbalance.weak} is ${report.imbalance.ratio}x lighter than ${report.imbalance.strong} and may be duplicated. Add images to ${report.imbalance.weak} or lower ${report.imbalance.strong}'s strength.`;
  }
  return line;
}

// The prompt writer names cast members as <Subject n> (name), numbered in scene order, so <Subject n> is the n-th
// RefMod item. Text Encode with RefMods only binds a RefMod to the words around its real label (<Video n>, <Picture n>),
// so each label is put right after the first mention of its <Subject n> in every shot: "<Subject 1> (The man) <Video 1>".
// A RefMod the shots never name gets one short sentence in the first shot instead (clothing goes after its wearer).
// Call this once, on freshly assembled prompt text. Running it again changes nothing.
export function attachRefmodLabels(promptText, items) {
  let text = String(promptText ?? "");
  const labelled = (items || []).filter((item) => item.label);
  if (!labelled.length) return text;
  const starts = [...text.matchAll(/\[Shot \d+\]/g)].map((match) => match.index);
  const blocks = starts.length
    ? [text.slice(0, starts[0]), ...starts.map((start, index) => text.slice(start, index + 1 < starts.length ? starts[index + 1] : text.length))]
    : [text];
  const firstShot = starts.length ? 1 : 0;
  const attach = (block, tag, label) => {
    let from = 0;
    while (true) {
      const at = block.indexOf(tag, from);
      if (at < 0) return block;
      const after = at + tag.length;
      const named = block.slice(after).match(/^\s*\([^)]*\)/);
      const end = named ? after + named[0].length : after;
      if (block.slice(end).trimStart().startsWith(label)) return block;
      return `${block.slice(0, end)} ${label}${block.slice(end)}`;
    }
  };
  const subjectTag = (item) => `<Subject ${(items.indexOf(item) + 1)}>`;
  for (let index = firstShot; index < blocks.length; index += 1) {
    for (const item of labelled) blocks[index] = attach(blocks[index], subjectTag(item), item.label);
  }
  text = blocks.join("");
  const sentenceFor = (item) => {
    if (item.category === "background") return `The setting is ${item.label}.`;
    if (item.category === "style") return `The visual style follows ${item.label}.`;
    return `${item.label} (${item.name}) is in the scene.`;
  };
  const missing = labelled.filter((item) => !text.includes(item.label));
  for (const item of missing) {
    const wearer = item.category === "clothing" ? labelled.find((candidate) => candidate.card_id === item.wears && text.includes(candidate.label)) : null;
    if (wearer) {
      text = text.replace(wearer.label, `${wearer.label}, wearing ${item.label},`);
      continue;
    }
    const extra = sentenceFor(item);
    const shots = [...text.matchAll(/\[Shot \d+\]/g)].map((match) => match.index);
    if (!shots.length) {
      text = `${text.trimEnd()} ${extra}`;
      continue;
    }
    const end = shots.length > 1 ? shots[1] : text.length;
    const block = text.slice(shots[0], end);
    const trailing = block.match(/\s*$/)[0];
    text = `${text.slice(0, shots[0])}${block.slice(0, block.length - trailing.length)} ${extra}${trailing}${text.slice(end)}`;
  }
  return text;
}

// The ordered list the render payload carries (visual mods only).
export function referencePayload(items) {
  return items.filter((item) => item.mod_name).map((item) => ({
    name: item.mod_name, strength: item.strength, kind: item.kind, label: item.label || "",
    card_id: item.card_id, display_name: item.name, category: item.category,
  }));
}
