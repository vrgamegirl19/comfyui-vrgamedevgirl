// Hard wall that keeps characters who are not selected for a scene out of that scene's text.
// Mirrors storyboard/cast_guard.py: a matcher for the characters left out of the scene (names,
// generic nouns, and gendered pronouns when that cannot be confused with a cast member), plus
// helpers that flag or remove any sentence that refers to them.

const MALE_PRONOUNS = ["he", "him", "his", "himself"];
const FEMALE_PRONOUNS = ["she", "her", "hers", "herself"];
const MALE_NOUNS = ["man", "men", "boy", "boys", "gentleman", "guy", "husband", "father", "brother", "son", "groom", "king", "male"];
const FEMALE_NOUNS = ["woman", "women", "girl", "girls", "lady", "wife", "mother", "sister", "daughter", "bride", "queen", "female"];
// Plural wording that implies a second person when the scene has a single cast member.
const PLURAL_TERMS = ["they", "them", "their", "theirs", "themselves", "both", "couple", "pair", "duo", "side by side"];
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

// allSubjects / castSubjects: arrays of { name, description } (plain name strings are accepted).
// Returns null when nobody is left out.
export function buildCastGuard(allSubjects, castSubjects) {
  const asObject = (item) => (item && typeof item === "object" ? item : { name: String(item || "") });
  const everyone = (allSubjects || []).map(asObject);
  const cast = (castSubjects || []).map(asObject);
  const castKeys = new Set(cast.map((item) => subjectName(item).toLowerCase()).filter(Boolean));
  const excluded = everyone.filter((item) => subjectName(item) && !castKeys.has(subjectName(item).toLowerCase()));
  if (!excluded.length) return null;
  const castTerms = new Set();
  for (const item of cast) for (const term of nameTerms(subjectName(item))) castTerms.add(term);
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
  if (!terms.size) return null;
  const ordered = [...terms].sort((a, b) => b.length - a.length);
  const source = ordered.map((term) => escapeRegExp(term).replace(/ /g, "\\s+")).join("|");
  return {
    // Fresh RegExp per use; the shared source keeps the guard serializable.
    source,
    castNames: cast.map(subjectName).filter(Boolean),
    excludedNames: excluded.map(subjectName),
  };
}

const matcher = (guard, flags = "gi") => new RegExp(`(?<![A-Za-z'])(?:${guard.source})(?![A-Za-z])`, flags);

export function castLeaks(text, guard) {
  if (!guard || !text) return [];
  return [...new Set([...String(text).matchAll(matcher(guard))].map((match) => match[0].toLowerCase()))].sort();
}

export function stripCastLeaks(text, guard) {
  if (!guard || !text) return String(text || "").trim();
  const test = matcher(guard, "i");
  const lines = String(text).split("\n").map((line) => {
    if (!line.trim()) return "";
    return line.trim().split(/(?<=[.!?])\s+/).filter((sentence) => !test.test(sentence)).join(" ").trim();
  });
  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

export function castWallText(guard) {
  if (!guard) return "";
  const cast = guard.castNames.join(", ") || "no one";
  return `CAST WALL (MANDATORY): the only characters in this scene are: ${cast}. Do not mention, show, imply, or hint at ${guard.excludedNames.join(", ")} in any way, including by name, pronoun, a hand, a shadow, a reflection, a voice, an off-screen presence, or a companion. Write the scene as if those characters do not exist.`;
}
