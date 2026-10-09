"""Which RefMods a scene uses, in what order, and the labels the text encoder will give them.

Python twin of ``refmod_labels.mjs`` in the Video Builder. The two must agree, because the prompt is written with
the browser's labels and the render must produce the same ones. ``tests/refmod_scene_cases.json`` holds the shared
test cases.

A RefMod reference is a Reference Builder card with ``source == "refmod"``. Its saved RefMod is named in
``card["refmod"]["name"]``. The Text Encode with RefMods node numbers visual mods per kind in loader order:
``<Picture n>`` for single-frame mods, ``<Video n>`` for mods with several frames, and ``<Audio n>`` for audio.
Mods with strength 0 are skipped and take no number.

Pure Python: no ComfyUI, torch or aiohttp imports.
"""

from typing import Any, Dict, List, Optional

from .scene_inputs import (
    _builder,
    _extra_target_id,
    _forced_extra_keys,
    _split_ids,
    _subject_items_for_segment,
    _text,
    scene_map_value,
)

KIND_LABELS = {"image": "Picture", "video": "Video", "audio": "Audio"}

# Order in a scene: characters, extras, clothing, objects, background, style.
CATEGORY_RANK = {"character": 0, "extra": 1, "clothing": 2, "object": 3, "background": 4, "style": 5}
_OBJECT_TYPES = ("object", "prop", "vehicle", "creature", "other")


def is_refmod_card(card: Any) -> bool:
    return (
        isinstance(card, dict)
        and _text(card.get("source")) == "refmod"
        and isinstance(card.get("refmod"), dict)
        and bool(_text(card["refmod"].get("name")))
    )


def card_category(card: Dict[str, Any], kind: str = "subject") -> str:
    """The ordering category of a card: character, extra, clothing, object, background or style."""
    if kind == "extra":
        return "extra"
    if kind == "location":
        return "background"
    reference_type = _text(card.get("reference_type")) or "character"
    if reference_type == "character":
        return "character"
    if reference_type == "outfit":
        return "clothing"
    if reference_type == "environment":
        return "background"
    if reference_type == "style":
        return "style"
    return "object"


def _strength(card: Dict[str, Any]) -> float:
    try:
        value = float(card["refmod"].get("strength", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return min(1.0, max(0.0, value))


def _item(card: Dict[str, Any], kind: str) -> Dict[str, Any]:
    refmod = card["refmod"]
    return {
        "key": f"{kind}:{_text(card.get('id'))}",
        "card_id": _text(card.get("id")),
        "name": _text(card.get("name")) or _text(refmod.get("name")),
        "category": card_category(card, kind),
        "reference_type": _text(card.get("reference_type")) or ("environment" if kind == "location" else "character"),
        "mod_name": _text(refmod.get("name")),
        "kind": _text(refmod.get("kind")) or "video",
        "strength": _strength(card),
        "tokens": int(refmod.get("tokens") or 0),
        "description": _text(card.get("description")),
        "wears": _text(card.get("wears")),
    }


def scene_subject_cards(segment: Dict[str, Any], subject_cards: List[Any]) -> List[Any]:
    """The subject cards a scene sends: none for a "no character present" scene. Twin of ``sceneSubjectCards``.

    ``_subject_items_for_segment`` (shared with the standard pipeline) still returns the project's first character
    for such a scene, so RefMod selection drops it here, like the browser does.
    """
    return [] if isinstance(segment, dict) and segment.get("no_character_present") else list(subject_cards or [])


def compose_items(
    subject_cards: List[Any],
    extra_cards: List[Any],
    location_card: Any,
    all_subjects: List[Any],
    override: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Turn the cards a scene uses into ordered RefMod items. Mirrors ``composeRefmodItems`` in refmod_labels.mjs.

    ``subject_cards``: the subject cards the scene maps to, in map order. ``extra_cards``: extras sent to MiniMax.
    ``location_card``: the scene's location card or None. ``all_subjects``: every subject card (clothing cards are
    found here). A clothing card tied to a character (``wears``) follows that character into the scene unless the
    card has ``follow: false`` or ``override`` (``{character_id: clothing card id(s)}``, an empty string for none)
    chose other clothing. An override names its cards whatever their ``follow`` setting.
    Cards that are not RefMods are ignored.
    """
    override = override if isinstance(override, dict) else {}
    items: List[Dict[str, Any]] = []
    seen = set()
    subjects = [s for s in all_subjects if isinstance(s, dict)]

    def add(card: Any, kind: str) -> None:
        if not is_refmod_card(card):
            return
        key = f"{kind}:{_text(card.get('id'))}"
        if key in seen:
            return
        seen.add(key)
        items.append(_item(card, kind))

    for card in subject_cards:
        add(card, "subject")
    for card in extra_cards:
        add(card, "extra")

    for character_id in [i["card_id"] for i in items if i["category"] == "character"]:
        if character_id in override:
            chosen = set(_split_ids(override[character_id]))
            items = [i for i in items if not (i["category"] == "clothing" and i["wears"] == character_id and i["card_id"] not in chosen)]
            for card in subjects:
                if _text(card.get("id")) in chosen:
                    add(card, "subject")
        else:
            for card in subjects:
                if _text(card.get("wears")) == character_id and card_category(card) == "clothing" and card.get("follow") is not False:
                    add(card, "subject")

    add(location_card, "location")
    return order_items(items)


def refmod_items_for_scene(session: Dict[str, Any], segment: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
    """The scene's RefMod references in render order, with no labels yet.

    Selects cards with the same scene maps as the standard pipeline (``subject_scene_map``, ``extra_scene_map``,
    ``scene_map``), leaves out subjects when the scene has no character (:func:`scene_subject_cards`), then orders
    them with :func:`compose_items`.
    """
    refs = _builder(session)
    subjects = [s for s in (refs.get("subjects") or []) if isinstance(s, dict)]
    extras = {_text(e.get("id")): e for e in (refs.get("extra_subjects") or []) if isinstance(e, dict)}
    extra_cards = [extras[key.split(":", 1)[1]] for key in _forced_extra_keys(refs, segment, index) if key.split(":", 1)[1] in extras]
    location_id = _text(scene_map_value(refs.get("scene_map"), segment, index))
    location = next((l for l in (refs.get("locations") or []) if isinstance(l, dict) and _text(l.get("id")) == location_id), None) if location_id else None
    # A scene's clothing comes from its scene mapping and each clothing card's "worn by" link. The old per-scene
    # clothing choice (``refmod_clothing_override``) has no control any more, so it is not applied.
    return compose_items(
        scene_subject_cards(segment, _subject_items_for_segment(refs, segment, index)), extra_cards, location, subjects,
    )


def order_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable order: category rank, then (for clothing) the position of the character who wears it."""
    characters = [item["card_id"] for item in items if item["category"] == "character"]

    def sort_key(pair):
        position, item = pair
        wearer = characters.index(item["wears"]) if item["category"] == "clothing" and item["wears"] in characters else len(characters)
        return (CATEGORY_RANK.get(item["category"], 3), wearer if item["category"] == "clothing" else 0, position)

    return [item for _, item in sorted(enumerate(items), key=sort_key)]


def assign_labels(items: List[Dict[str, Any]], include_audio: bool = False) -> List[Dict[str, Any]]:
    """Add ``label`` (for example ``<Video 2>``) to each item the way Text Encode with RefMods numbers them.

    Items with strength 0 get an empty label. When ``include_audio`` is true an ``audio`` entry (``<Audio 1>``) is
    appended, matching the scene audio mod the render adds after the visual mods.
    """
    counters = {"image": 0, "video": 0, "audio": 0}
    labelled: List[Dict[str, Any]] = []
    for item in items:
        entry = dict(item)
        kind = entry.get("kind") if entry.get("kind") in ("image", "video") else "video"
        if entry["strength"] <= 0:
            entry["label"] = ""
        else:
            counters[kind] += 1
            entry["label"] = f"<{KIND_LABELS[kind]} {counters[kind]}>"
        labelled.append(entry)
    if include_audio:
        counters["audio"] += 1
        labelled.append({
            "key": "audio:scene", "card_id": "", "name": "Scene audio", "category": "audio", "reference_type": "audio",
            "mod_name": "", "kind": "audio", "strength": 1.0, "tokens": 0, "description": "", "wears": "",
            "label": f"<Audio {counters['audio']}>",
        })
    return labelled


def total_tokens(items: List[Dict[str, Any]]) -> int:
    return sum(int(item.get("tokens") or 0) for item in items if item.get("strength", 0) > 0)


TOKEN_WARN_LIMIT = 6000
IMBALANCE_RATIO = 2.0


def token_report(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Scene token total and the character balance. Twin of ``tokenReport`` in refmod_labels.mjs.

    A character whose weight (tokens x strength) is less than half of the strongest character's tends to be duplicated
    or replaced by the stronger ones, so the weakest and strongest are named when the ratio passes ``IMBALANCE_RATIO``.
    """
    active = [item for item in items if item.get("strength", 0) > 0]
    total = sum(int(item.get("tokens") or 0) for item in active)
    people = [
        (int(item.get("tokens") or 0) * float(item["strength"]), item) for item in active
        if item.get("category") in ("character", "extra") and int(item.get("tokens") or 0) > 0
    ]
    imbalance = None
    if len(people) >= 2:
        weak = min(people, key=lambda pair: pair[0])
        strong = max(people, key=lambda pair: pair[0])
        if weak[0] > 0 and strong[0] / weak[0] > IMBALANCE_RATIO:
            imbalance = {"weak": weak[1]["name"], "strong": strong[1]["name"], "ratio": round(strong[0] / weak[0], 1)}
    return {"total": total, "limit": TOKEN_WARN_LIMIT, "over_limit": total > TOKEN_WARN_LIMIT, "imbalance": imbalance}


def reference_payload(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The ``refmod_references`` list the render payload carries: visual items only, already ordered."""
    return [
        {"name": item["mod_name"], "strength": item["strength"], "kind": item["kind"], "label": item.get("label", ""),
         "card_id": item["card_id"], "display_name": item["name"], "category": item["category"]}
        for item in items if item.get("mod_name")
    ]
