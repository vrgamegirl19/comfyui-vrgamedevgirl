// Hard wall that keeps people who are not selected for a scene out of that scene's text.
// Mirrors storyboard/cast_guard.py: a matcher for the characters left out of the scene (names,
// generic nouns, and gendered pronouns when that cannot be confused with a cast member), a second
// matcher for invented unnamed people ("a woman", "a stranger", "a crowd"), and helpers that flag
// or remove any sentence that refers to them. Names come from the Reference Builder and can be
// anything, so the invented-person matcher allows every word found in a selected name.

const MALE_PRONOUNS = ["he", "him", "his", "himself"];
const FEMALE_PRONOUNS = ["she", "her", "hers", "herself"];
const MALE_NOUNS = ["man", "men", "boy", "boys", "gentleman", "guy", "husband", "father", "brother", "son", "groom", "king", "male"];
const FEMALE_NOUNS = ["woman", "women", "girl", "girls", "lady", "wife", "mother", "sister", "daughter", "bride", "queen", "female"];
// Plural wording that implies a second person when the scene has a single cast member.
const PLURAL_TERMS = ["they", "them", "their", "theirs", "themselves", "both", "couple", "pair", "duo", "side by side"];
// People an LLM tends to invent. Allowed only when a selected cast member or extra is named with the word.
const INVENTED_SINGLE_TERMS = [
  "man", "men", "woman", "women", "girl", "girls", "boy", "boys", "lady", "gentleman", "guy", "stranger", "strangers",
  "figure", "figures", "silhouette", "silhouettes", "someone", "somebody", "companion", "companions", "lover", "lovers",
  "partner", "partners", "child", "children", "kid", "kids", "friend", "friends", "person", "people",
];
// Group wording. Also allowed when the scene has mapped extras, since extras are groups of people.
const INVENTED_GROUP_TERMS = ["crowd", "crowds", "onlookers", "bystanders", "passersby", "passers-by", "audience", "others", "dancers", "fans", "spectators"];
const GENERIC_TERMS = new Set(["the", "a", "an", "and", "of", "subject", "character", "person", "singer", "performer"]);

const tokens = (text) => String(text || "").toLowerCase().match(/[a-z']+/g) || [];
const subjectName = (subject) => String(typeof subject === "object" && subject ? subject.name || "" : subject || "").trim();
const escapeRegExp = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export function subjectGender(subject) {
  const words = tokens(`${subject?.name || ""} ${subject?.description || ""}`);
  const male = words.filter((word) => MALE_PRONOUNS.includes(word) || MALE_NOUNS.includes(word)).length;
  const female = words.filter((word) => FEMALE_PRONOUNS.includes(word) || FEMALE_NOUNS.includes(word)).length;
  if (male > female) return "m";
  if (female > male) return "f";
  return "";
}

function nameTerms(name) {
  const text = String(name || "").trim().replace(/\s+/g, " ").toLowerCase();
  if (!text) return new Set();
  const base = text.replace(/^(?:the|a|an)\s+/, "");
  const terms = new Set([text, base]);
  if (text === base) {
    const first = base.split(" ")[0];
    if (first.length >= 3 && !GENERIC_TERMS.has(first)) terms.add(first);
  }
  return new Set([...terms].filter((term) => term && !GENERIC_TERMS.has(term)));
}

const sourceOf = (terms) => [...terms]
  .sort((a, b) => b.length - a.length)
  .map((term) => escapeRegExp(term).replace(/ /g, "\\s+"))
  .join("|");

// allSubjects / castSubjects: arrays of { name, description } (plain name strings are accepted).
// extraNames: mapped extras, which are allowed in the scene.
export function buildCastGuard(allSubjects, castSubjects, extraNames = []) {
  const asObject = (item) => (item && typeof item === "object" ? item : { name: String(item || "") });
  const everyone = (allSubjects || []).map(asObject);
  const cast = (castSubjects || []).map(asObject);
  const extras = (extraNames || []).map((name) => String(name || "").trim()).filter(Boolean);
  const castKeys = new Set(cast.map((item) => subjectName(item).toLowerCase()).filter(Boolean));
  const excluded = everyone.filter((item) => subjectName(item) && !castKeys.has(subjectName(item).toLowerCase()));
  const castTerms = new Set();
  for (const item of cast) for (const term of nameTerms(subjectName(item))) castTerms.add(term);
  for (const name of extras) for (const term of nameTerms(name)) castTerms.add(term);
  const castWords = new Set([...castTerms].flatMap((term) => term.split(" ")));
  const terms = new Set();
  for (const item of excluded) {
    for (const term of nameTerms(subjectName(item))) {
      if (!castTerms.has(term) && !castWords.has(term)) terms.add(term);
    }
  }
  const castGenders = new Set(cast.map(subjectGender));
  if (cast.length && !castGenders.has("")) {
    for (const item of excluded) {
      const gender = subjectGender(item);
      if (gender && !castGenders.has(gender)) {
        for (const word of gender === "f" ? [...FEMALE_PRONOUNS, ...FEMALE_NOUNS] : [...MALE_PRONOUNS, ...MALE_NOUNS]) terms.add(word);
      }
    }
  }
  if (cast.length <= 1) for (const word of PLURAL_TERMS) terms.add(word);
  if (!cast.length && !extras.length) for (const word of [...MALE_PRONOUNS, ...FEMALE_PRONOUNS]) terms.add(word);
  const invented = new Set(INVENTED_SINGLE_TERMS);
  if (!extras.length) for (const word of INVENTED_GROUP_TERMS) invented.add(word);
  for (const word of castWords) invented.delete(word);
  return {
    // Fresh RegExp per use; the shared sources keep the guard serializable.
    source: sourceOf(terms),
    inventedSource: sourceOf(invented),
    castNames: cast.map(subjectName).filter(Boolean),
    excludedNames: excluded.map(subjectName),
    extraNames: extras,
  };
}

const matcher = (source, flags = "gi") => new RegExp(`(?<![A-Za-z'])(?:${source})(?![A-Za-z])`, flags);
const sources = (guard, invented) => [guard?.source, invented ? guard?.inventedSource : ""].filter(Boolean);

export function castLeaks(text, guard, invented = true) {
  if (!guard || !text) return [];
  const found = new Set();
  for (const source of sources(guard, invented)) {
    for (const match of String(text).matchAll(matcher(source))) found.add(match[0].toLowerCase());
  }
  return [...found].sort();
}

export function stripCastLeaks(text, guard, invented = true) {
  const tests = sources(guard, invented).map((source) => matcher(source, "i"));
  if (!tests.length || !text) return String(text || "").trim();
  const lines = String(text).split("\n").map((line) => {
    if (!line.trim()) return "";
    return line.trim().split(/(?<=[.!?])\s+/).filter((sentence) => !tests.some((test) => test.test(sentence))).join(" ").trim();
  });
  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

const FEELING_WORDS = [
  "feel", "feels", "feeling", "feelings", "grief", "longing", "yearning", "memory", "memories",
  "soul", "emotion", "emotions", "emotional", "nostalgia", "nostalgic", "heartbreak", "heartbroken",
];

// Distinct feeling words in text when it uses at least `minimum` of them, else []. Mirrors _feeling_word_hits
// in storyboard/story_layer.py. Shot text should describe what the camera sees.
export function feelingWordHits(text, minimum = 2) {
  const hits = [...String(text || "").matchAll(new RegExp(`\\b(?:${FEELING_WORDS.join("|")})\\b`, "gi"))].map((match) => match[0].toLowerCase());
  return hits.length >= minimum ? [...new Set(hits)].sort() : [];
}

// Positive cast statement that tells the model exactly who is in the scene.
export function castWallText(guard) {
  if (!guard) return "";
  const extras = guard.extraNames || [];
  if (!guard.castNames.length && !extras.length) {
    return "PEOPLE IN THIS SCENE: none. No people appear in this scene. Show only the location, objects, and light. Do not show any person, hand, arm, shadow, reflection, or silhouette.";
  }
  const people = [...guard.castNames, ...extras.map((name) => `${name} (mapped extras)`)].join(", ");
  let text = `PEOPLE IN THIS SCENE: ${people}. No other person exists in this scene. Do not add an unnamed man, woman, girl, boy, stranger, crowd, extra hand or arm, shadow, reflection, or silhouette of anyone else. Refer to each person only by name or label, never by pronoun.`;
  if (guard.excludedNames.length) text += ` Not in this scene: ${guard.excludedNames.join(", ")}.`;
  return text;
}
