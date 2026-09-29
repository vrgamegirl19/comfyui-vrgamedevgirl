"""Hard wall that keeps characters who are not selected for a scene out of that scene's text.

The story arc, story brief, and neighbouring beats describe the whole song, so an LLM can
carry a character into a scene that does not include them. These helpers build a matcher for
the characters left out of a scene (names, generic nouns, and gendered pronouns when that
cannot be confused with a cast member) and remove or flag any sentence that refers to them.

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


def build_cast_guard(all_subjects, cast_subjects):
    """Build a guard for the subjects in all_subjects that are not in cast_subjects.

    Both arguments are lists of dicts with name/description (plain name strings are accepted).
    Returns None when nobody is left out, otherwise {"pattern", "cast_names", "excluded_names"}.
    """
    everyone = [item if isinstance(item, dict) else {"name": str(item or "")} for item in (all_subjects or [])]
    cast = [item if isinstance(item, dict) else {"name": str(item or "")} for item in (cast_subjects or [])]
    cast_keys = {_subject_name(item).casefold() for item in cast if _subject_name(item)}
    excluded = [item for item in everyone if _subject_name(item) and _subject_name(item).casefold() not in cast_keys]
    if not excluded:
        return None
    cast_terms = set()
    for item in cast:
        cast_terms |= _name_terms(_subject_name(item))
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
    if not terms:
        return None
    ordered = sorted(terms, key=len, reverse=True)
    pattern = re.compile(r"(?<![A-Za-z'])(?:" + "|".join(re.escape(term).replace(r"\ ", r"\s+") for term in ordered) + r")(?![A-Za-z])", re.IGNORECASE)
    return {
        "pattern": pattern,
        "cast_names": [_subject_name(item) for item in cast if _subject_name(item)],
        "excluded_names": [_subject_name(item) for item in excluded],
    }


def cast_leaks(text, guard):
    """Distinct lowercase terms in text that refer to a character outside the scene cast."""
    if not guard or not text:
        return []
    return sorted({match.group(0).lower() for match in guard["pattern"].finditer(str(text))})


def strip_cast_leaks(text, guard):
    """Drop every sentence that refers to a character outside the scene cast."""
    if not guard or not text:
        return str(text or "").strip()
    kept_lines = []
    for line in str(text).split("\n"):
        sentences = _SENTENCE_SPLIT_RE.split(line.strip()) if line.strip() else [""]
        kept = [sentence for sentence in sentences if not guard["pattern"].search(sentence)]
        kept_lines.append(" ".join(item for item in kept if item).strip() if line.strip() else "")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()


def cast_wall_text(guard):
    """Prompt text that tells the model exactly who may and may not appear."""
    if not guard:
        return ""
    cast = ", ".join(guard["cast_names"]) or "no one"
    banned = ", ".join(guard["excluded_names"])
    return (
        f"CAST WALL (MANDATORY): the only characters in this scene are: {cast}. "
        f"Do not mention, show, imply, or hint at {banned} in any way, including by name, pronoun, "
        "a hand, a shadow, a reflection, a voice, an off-screen presence, or a companion. "
        "Write the scene as if those characters do not exist."
    )


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
