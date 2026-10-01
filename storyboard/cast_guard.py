"""Hard wall that keeps characters who are not selected for a scene out of that scene's text.

The story arc, story brief, and neighbouring beats describe the whole song, so an LLM can
carry a character into a scene that does not include them, or invent an unnamed person
("a woman", "a stranger", "a crowd"). These helpers build a matcher for the characters left
out of a scene (names, generic nouns, and gendered pronouns when that cannot be confused with
a cast member) plus a separate matcher for invented people, and remove or flag any sentence
that refers to them.

Subject names come from the Reference Builder and can be anything, so nothing here depends on
a hardcoded name. The invented-person matcher allows every word that appears in the name of a
selected cast member or mapped extra.

The same rules are mirrored in web/music_video_builder/cast_guard.mjs.
"""

import re

_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ENTRY_RE = re.compile(r"^(\s*(?:[-*•][ \t]*)?Scene[ \t]+(\d+)\b[^\n]*?)(?:[ \t]+[—–-][ \t]+)(.*)$")

_MALE_PRONOUNS = ("he", "him", "his", "himself")
_FEMALE_PRONOUNS = ("she", "her", "hers", "herself")
_MALE_NOUNS = ("man", "men", "boy", "boys", "gentleman", "guy", "husband", "father", "brother", "son", "groom", "king", "male")
_FEMALE_NOUNS = ("woman", "women", "girl", "girls", "lady", "wife", "mother", "sister", "daughter", "bride", "queen", "female")
# Plural wording that implies a second person when the scene has a single cast member.
_PLURAL_TERMS = ("they", "them", "their", "theirs", "themselves", "both", "couple", "pair", "duo", "side by side")
# People an LLM tends to invent. Allowed only when a selected cast member or extra is named with the word.
_INVENTED_SINGLE_TERMS = (
    "man", "men", "woman", "women", "girl", "girls", "boy", "boys", "lady", "gentleman", "guy", "stranger", "strangers",
    "figure", "figures", "silhouette", "silhouettes", "someone", "somebody", "companion", "companions", "lover", "lovers",
    "partner", "partners", "child", "children", "kid", "kids", "friend", "friends", "person", "people",
)
# Group wording. Also allowed when the scene has mapped extras, since extras are groups of people.
_INVENTED_GROUP_TERMS = (
    "crowd", "crowds", "onlookers", "bystanders", "passersby", "passers-by", "audience", "others", "dancers", "fans", "spectators",
)
_GENERIC_TERMS = {"the", "a", "an", "and", "of", "subject", "character", "person", "singer", "performer"}


def _tokens(text):
    return re.findall(r"[a-z']+", str(text or "").lower())


def subject_gender(subject):
    """Return "m", "f", or "" from the subject's name and description wording."""
    words = _tokens(f"{(subject or {}).get('name', '')} {(subject or {}).get('description', '')}")
    male = sum(1 for word in words if word in _MALE_PRONOUNS or word in _MALE_NOUNS)
    female = sum(1 for word in words if word in _FEMALE_PRONOUNS or word in _FEMALE_NOUNS)
    if male > female:
        return "m"
    if female > male:
        return "f"
    return ""


def _name_terms(name):
    text = re.sub(r"\s+", " ", str(name or "").strip()).lower()
    if not text:
        return set()
    base = _ARTICLE_RE.sub("", text)
    terms = {text, base}
    if text == base:
        first = base.split(" ")[0]
        if len(first) >= 3 and first not in _GENERIC_TERMS:
            terms.add(first)
    return {term for term in terms if term and term not in _GENERIC_TERMS}


def _subject_name(subject):
    if isinstance(subject, dict):
        return str(subject.get("name") or "").strip()
    return str(subject or "").strip()


def _compile(terms):
    ordered = sorted(terms, key=len, reverse=True)
    return re.compile(
        r"(?<![A-Za-z'])(?:" + "|".join(re.escape(term).replace(r"\ ", r"\s+") for term in ordered) + r")(?![A-Za-z])",
        re.IGNORECASE,
    )


def build_cast_guard(all_subjects, cast_subjects, extra_names=()):
    """Build a guard for the people who must not appear in a scene.

    all_subjects and cast_subjects are lists of dicts with name/description (plain name strings
    are accepted). extra_names lists mapped extras, which are allowed in the scene.

    Returns {"pattern", "invented_pattern", "cast_names", "excluded_names", "extra_names"}.
    "pattern" matches characters left out of the scene. "invented_pattern" matches unnamed people
    the scene does not contain.
    """
    everyone = [item if isinstance(item, dict) else {"name": str(item or "")} for item in (all_subjects or [])]
    cast = [item if isinstance(item, dict) else {"name": str(item or "")} for item in (cast_subjects or [])]
    extras = [str(item or "").strip() for item in (extra_names or []) if str(item or "").strip()]
    cast_keys = {_subject_name(item).casefold() for item in cast if _subject_name(item)}
    excluded = [item for item in everyone if _subject_name(item) and _subject_name(item).casefold() not in cast_keys]
    cast_terms = set()
    for item in cast:
        cast_terms |= _name_terms(_subject_name(item))
    for name in extras:
        cast_terms |= _name_terms(name)
    cast_words = {word for term in cast_terms for word in term.split(" ")}
    terms = set()
    for item in excluded:
        terms |= {term for term in _name_terms(_subject_name(item)) if term not in cast_terms and term not in cast_words}
    cast_genders = {subject_gender(item) for item in cast}
    if cast and "" not in cast_genders:
        for item in excluded:
            gender = subject_gender(item)
            if gender and gender not in cast_genders:
                terms.update(_FEMALE_PRONOUNS + _FEMALE_NOUNS if gender == "f" else _MALE_PRONOUNS + _MALE_NOUNS)
    if len(cast) <= 1:
        terms.update(_PLURAL_TERMS)
    if not cast and not extras:
        terms.update(_MALE_PRONOUNS + _FEMALE_PRONOUNS)
    invented = set(_INVENTED_SINGLE_TERMS)
    if not extras:
        invented.update(_INVENTED_GROUP_TERMS)
    invented -= cast_words
    return {
        "pattern": _compile(terms) if terms else None,
        "invented_pattern": _compile(invented) if invented else None,
        "cast_names": [_subject_name(item) for item in cast if _subject_name(item)],
        "excluded_names": [_subject_name(item) for item in excluded],
        "extra_names": extras,
    }


def _patterns(guard, invented):
    if not guard:
        return []
    found = [guard.get("pattern")]
    if invented:
        found.append(guard.get("invented_pattern"))
    return [pattern for pattern in found if pattern]


def cast_leaks(text, guard, invented=True):
    """Distinct lowercase terms in text that refer to a person outside the scene cast."""
    if not guard or not text:
        return []
    found = set()
    for pattern in _patterns(guard, invented):
        found.update(match.group(0).lower() for match in pattern.finditer(str(text)))
    return sorted(found)


def strip_cast_leaks(text, guard, invented=True):
    """Drop every sentence that refers to a person outside the scene cast."""
    patterns = _patterns(guard, invented)
    if not patterns or not text:
        return str(text or "").strip()
    kept_lines = []
    for line in str(text).split("\n"):
        sentences = _SENTENCE_SPLIT_RE.split(line.strip()) if line.strip() else [""]
        kept = [sentence for sentence in sentences if not any(pattern.search(sentence) for pattern in patterns)]
        kept_lines.append(" ".join(item for item in kept if item).strip() if line.strip() else "")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()


def cast_wall_text(guard):
    """Positive cast statement that tells the model exactly who is in the scene."""
    if not guard:
        return ""
    cast = guard["cast_names"]
    extras = guard.get("extra_names") or []
    if not cast and not extras:
        return (
            "PEOPLE IN THIS SCENE: none. No people appear in this scene. "
            "Show only the location, objects, and light. Do not show any person, hand, arm, shadow, reflection, or silhouette."
        )
    people = ", ".join(cast + [f"{name} (mapped extras)" for name in extras])
    banned = ", ".join(guard["excluded_names"])
    text = (
        f"PEOPLE IN THIS SCENE: {people}. No other person exists in this scene. "
        "Do not add an unnamed man, woman, girl, boy, stranger, crowd, extra hand or arm, shadow, reflection, or silhouette of anyone else. "
        "Refer to each person only by name, never by pronoun."
    )
    if banned:
        text += f" Not in this scene: {banned}."
    return text


def strip_story_arc_entry_leaks(arc_text, guards_by_scene, cast_names_by_scene=None):
    """Apply per-scene guards to "Scene N (...) — text" entries in a scene-by-scene story arc."""
    out = []
    for line in str(arc_text or "").split("\n"):
        match = _ENTRY_RE.match(line)
        guard = guards_by_scene.get(int(match.group(2))) if match else None
        if not match or not guard:
            out.append(line)
            continue
        body = strip_cast_leaks(match.group(3), guard)
        if not body:
            names = ", ".join((cast_names_by_scene or {}).get(int(match.group(2)), []) or guard["cast_names"]) or "the scene"
            body = f"The scene stays focused on {names}."
        out.append(f"{match.group(1)} — {body}")
    return "\n".join(out)
