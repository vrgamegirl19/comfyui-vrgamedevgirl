"""Parsing and validation of LLM output for the Video Builder: JSON repair, location and subject parsing, and prompt checks."""

import json
import re
from .text_cleaning import _clean_visual_gemma_text


def _extract_json_object_from_text(text):
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    candidates = [cleaned, _repair_builder_json_like_text(cleaned)]
    last_error = None
    for candidate in candidates:
        if not str(candidate or "").strip():
            continue
        try:
            return json.loads(candidate)
        except Exception as error:
            last_error = error
    if last_error:
        raise last_error
    raise ValueError("Gemma did not return a JSON object.")


def _repair_builder_json_like_text(text):
    repaired = str(text or "").strip()
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    repaired = re.sub(r"//.*?$", "", repaired, flags=re.MULTILINE)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    json_keys = (
        "locations|scene_map|name|description|id|label|prompt|runner|used_model|"
        "title|premise|scenes|character_id|subject_id|speaker_id|location_id|"
        "dialogue|line|lyrics|story_beat|beat|visual_direction|summary|image_prompt|visual_prompt|"
        "shot_type|camera_motion|facial_performance|facial_performance_custom|emotion|delivery|"
        "motion_summary|video_notes|setting|location_name|character_name|speaker"
    )
    repaired = re.sub(
        rf'([{{\[,]\s*)({json_keys})\s*:',
        r'\1"\2":',
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(r'([{\[,]\s*)(Scene\s*\d+|Scene\d+)\s*:', lambda m: f'{m.group(1)}"{m.group(2).replace(" ", "")}":', repaired, flags=re.IGNORECASE)
    repaired = re.sub(
        rf'(^\s*)({json_keys})\s*:',
        r'\1"\2":',
        repaired,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    repaired = re.sub(r'(^\s*)(Scene\s*\d+|Scene\d+)\s*:', lambda m: f'{m.group(1)}"{m.group(2).replace(" ", "")}":', repaired, flags=re.IGNORECASE | re.MULTILINE)
    # Gemma sometimes omits commas between array objects or object properties.
    repaired = re.sub(r'(["}\]])\s*\n\s*(")', r'\1,\n\2', repaired)
    repaired = re.sub(r'(})\s*\n\s*({)', r'\1,\n\2', repaired)
    repaired = re.sub(r'(})\s*({)', r'\1,\2', repaired)
    repaired = re.sub(r'(\])\s*("scene_map"\s*:)', r'\1,\2', repaired, flags=re.IGNORECASE)
    return repaired


def _parse_flux_location_map_fallback(text, cleaned_scenes, existing_locations=None):
    cleaned = _clean_visual_gemma_text(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]

    locations = []
    seen_names = set()
    location_block_match = re.search(
        r'"?locations"?\s*:\s*\[(.*?)]\s*,?\s*"?scene_map"?\s*:',
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    location_block = location_block_match.group(1) if location_block_match else ""
    for object_text in re.findall(r"\{(.*?)\}", location_block, flags=re.DOTALL):
        name_match = re.search(r'"?name"?\s*:\s*"([^"]+)"', object_text, flags=re.IGNORECASE | re.DOTALL)
        description_match = re.search(r'"?description"?\s*:\s*"([^"]*)"', object_text, flags=re.IGNORECASE | re.DOTALL)
        name = re.sub(r"\s+", " ", (name_match.group(1) if name_match else "").strip())
        description = re.sub(r"\s+", " ", (description_match.group(1) if description_match else "").strip())
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        locations.append({"name": name, "description": description})

    if not locations and isinstance(existing_locations, list):
        for item in existing_locations:
            if not isinstance(item, dict):
                continue
            name = re.sub(r"\s+", " ", str(item.get("name", "") or "").strip())
            description = re.sub(r"\s+", " ", str(item.get("description", "") or "").strip())
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            locations.append({"name": name, "description": description})

    scene_lookup = {}
    for index, scene in enumerate(cleaned_scenes, start=1):
        scene_lookup[str(scene["id"]).strip().lower()] = scene["id"]
        scene_lookup[str(scene["label"]).strip().lower()] = scene["id"]
        scene_lookup[f"scene {index}"] = scene["id"]
        scene_lookup[f"scene{index}"] = scene["id"]
        scene_lookup[str(index)] = scene["id"]

    scene_map = {}
    scene_block_match = re.search(r'"?scene_map"?\s*:\s*\{(.*?)\}\s*$', cleaned, flags=re.IGNORECASE | re.DOTALL)
    scene_block = scene_block_match.group(1) if scene_block_match else ""
    for raw_key, raw_location in re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', scene_block, flags=re.DOTALL):
        lookup_key = re.sub(r"\s+", " ", raw_key.strip().lower())
        scene_id = scene_lookup.get(lookup_key) or scene_lookup.get(lookup_key.replace(" ", ""))
        location_name = re.sub(r"\s+", " ", raw_location.strip())
        if scene_id and location_name:
            scene_map[scene_id] = location_name

    if locations:
        if not scene_map:
            scene_map = _fallback_location_map_by_overlap(cleaned_scenes, locations)
        else:
            valid_location_names = {item["name"].lower(): item["name"] for item in locations}
            for scene in cleaned_scenes:
                raw_name = re.sub(r"\s+", " ", str(scene_map.get(scene["id"], "") or "").strip())
                if raw_name.lower() not in valid_location_names:
                    scene_map[scene["id"]] = _best_location_for_scene(scene, locations)["name"]
        return {"locations": locations, "scene_map": scene_map}

    raise ValueError("Gemma location map could not be parsed as JSON or recovered from text.")


def _fallback_location_map_by_overlap(cleaned_scenes, locations):
    mapped = {}
    for scene in cleaned_scenes:
        mapped[scene["id"]] = _best_location_for_scene(scene, locations)["name"]
    return mapped


def _best_location_for_scene(scene, locations):
    if not locations:
        return {"name": "Location 1", "description": ""}
    scene_text = f"{scene.get('concept', '')} {scene.get('notes', '')}"
    best_location = locations[0]
    best_score = -1
    for location in locations:
        score = _location_text_overlap_score(
            scene_text,
            f"{location.get('name', '')} {location.get('description', '')}",
        )
        if score > best_score:
            best_location = location
            best_score = score
    return best_location


def _canonical_location_name(name, locations):
    raw = re.sub(r"\s+", " ", str(name or "").strip()).lower()
    for location in locations or []:
        loc_name = re.sub(r"\s+", " ", str(location.get("name", "") or "").strip())
        if loc_name.lower() == raw:
            return loc_name
    return ""


def _location_usage_counts_from_payload(payload, locations):
    names = [re.sub(r"\s+", " ", str(item.get("name", "") or "").strip()) for item in locations or []]
    counts = {name: 0 for name in names if name}
    raw_counts = payload.get("used_location_counts")
    if isinstance(raw_counts, dict):
        for raw_name, raw_count in raw_counts.items():
            name = _canonical_location_name(raw_name, locations)
            if name:
                try:
                    counts[name] = max(0, int(raw_count or 0))
                except Exception:
                    counts[name] = counts.get(name, 0)
    raw_assignments = payload.get("previous_assignments")
    if isinstance(raw_assignments, list):
        for item in raw_assignments:
            if isinstance(item, dict):
                name = _canonical_location_name(item.get("location") or item.get("location_name"), locations)
            else:
                name = _canonical_location_name(item, locations)
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _balance_location_map_by_usage(scene_map, cleaned_scenes, locations, previous_counts=None):
    if not scene_map or not cleaned_scenes or not locations:
        return scene_map
    location_by_name = {}
    for item in locations:
        name = re.sub(r"\s+", " ", str(item.get("name", "") or "").strip())
        if name:
            location_by_name[name] = item
    location_names = list(location_by_name.keys())
    if len(location_names) <= 1:
        return scene_map
    balanced = {}
    fallback = _fallback_location_map_by_overlap(cleaned_scenes, locations)
    for scene in cleaned_scenes:
        name = _canonical_location_name(scene_map.get(scene["id"], ""), locations) or fallback.get(scene["id"], "")
        balanced[scene["id"]] = name

    previous_counts = previous_counts or {}
    current_counts = {name: 0 for name in location_names}
    for name in balanced.values():
        if name in current_counts:
            current_counts[name] += 1

    target_count = min(len(cleaned_scenes), len(location_names))
    desired_locations = sorted(
        location_names,
        key=lambda name: (int(previous_counts.get(name, 0) or 0), current_counts.get(name, 0), location_names.index(name)),
    )[:target_count]

    for desired_name in desired_locations:
        if current_counts.get(desired_name, 0) > 0:
            continue
        desired_location = location_by_name.get(desired_name) or {"name": desired_name, "description": ""}
        best_scene = None
        best_score = None
        for scene in cleaned_scenes:
            current_name = balanced.get(scene["id"], "")
            if current_name == desired_name:
                continue
            if current_counts.get(current_name, 0) <= 1 and any(current_counts.get(name, 0) == 0 for name in desired_locations if name != desired_name):
                continue
            scene_text = f"{scene.get('concept', '')} {scene.get('notes', '')}"
            desired_score = _location_text_overlap_score(scene_text, f"{desired_location.get('name', '')} {desired_location.get('description', '')}")
            current_location = location_by_name.get(current_name, {"name": current_name, "description": ""})
            current_score = _location_text_overlap_score(scene_text, f"{current_location.get('name', '')} {current_location.get('description', '')}")
            repeat_penalty = current_counts.get(current_name, 0) + int(previous_counts.get(current_name, 0) or 0)
            score = (desired_score - current_score) + repeat_penalty
            if best_score is None or score > best_score:
                best_score = score
                best_scene = scene
        if best_scene:
            old_name = balanced.get(best_scene["id"], "")
            if old_name in current_counts:
                current_counts[old_name] = max(0, current_counts[old_name] - 1)
            balanced[best_scene["id"]] = desired_name
            current_counts[desired_name] = current_counts.get(desired_name, 0) + 1
    return balanced


def _location_text_overlap_score(scene_text, location_text):
    stop_words = {
        "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into", "is", "it",
        "of", "on", "or", "the", "to", "with", "scene", "shot", "cinematic", "woman", "man",
        "girl", "boy", "subject", "character", "wearing", "light", "lighting",
    }
    scene_tokens = {
        token for token in re.findall(r"[a-z0-9]+", str(scene_text or "").lower())
        if len(token) > 2 and token not in stop_words
    }
    location_tokens = [
        token for token in re.findall(r"[a-z0-9]+", str(location_text or "").lower())
        if len(token) > 2 and token not in stop_words
    ]
    if not scene_tokens or not location_tokens:
        return 0
    score = 0
    for token in location_tokens:
        if token in scene_tokens:
            score += 3
        elif any(scene_token.startswith(token) or token.startswith(scene_token) for scene_token in scene_tokens if len(scene_token) > 4):
            score += 1
    return score


def _parse_location_lines(text):
    locations = []
    seen_names = set()
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip().strip("-").strip()
        if not line or line in {"{", "}", "[", "]"}:
            continue
        match = re.match(r"^\s*(?:Location\s*)?(\d+)\s*(?:[|:=\).-])\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if not match:
            continue
        rest = match.group(2).strip().strip('"').rstrip(",")
        parts = [part.strip().strip('"') for part in rest.split("|")]
        if len(parts) >= 2:
            name = parts[0]
            description = " | ".join(parts[1:])
        else:
            name = rest
            description = rest
        name = re.sub(r"^\s*name\s*[:=]\s*", "", name, flags=re.IGNORECASE)
        description = re.sub(r"^\s*description\s*[:=]\s*", "", description, flags=re.IGNORECASE)
        raw_name = name
        name, description = _clean_location_card(name, description, raw_name)
        if not name or _looks_like_location_meta_text(name) or _looks_like_location_meta_text(description) or not _valid_location_card(name, description) or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        locations.append({"name": name, "description": description})
    return locations


def _looks_like_location_meta_text(value):
    text = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    if not text:
        return True
    if len(text) > 140 and not re.search(r"\b(?:room|hall|hallway|corridor|street|road|forest|temple|pool|motel|stage|club|warehouse|desert|beach|shore|city|rooftop|alley|kitchen|bedroom|bathroom|church|chapel|station|train|car|bus|field|garden|vault|cave|lake|river|bridge|tunnel|apartment|house|mansion|hotel|bar|lounge|studio|parking|garage)\b", text):
        return True
    meta_patterns = (
        r"\bsince the provided\b",
        r"\bprovided .*content was not visible\b",
        r"\bprovided .*prompt\b",
        r"\bsubjectsandscenes\.txt\b",
        r"\bhere is (?:a|the)\b",
        r"\breusable location list\b",
        r"\bbased on the structural context\b",
        r"\bscene descriptions imply\b",
        r"\bcohesive project\b",
        r"\bi can(?:not|'t)\b",
        r"\bi(?:'|’)m sorry\b",
        r"\bas an ai\b",
        r"\boutput format\b",
        r"\buser input\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in meta_patterns)


_LOCATION_PLACE_WORDS = {
    "alley", "apartment", "arena", "attic", "ballroom", "bar", "barn", "bathroom", "beach", "bedroom",
    "bridge", "building", "cabin", "cafe", "casino", "cathedral", "cave", "chapel", "church", "city",
    "club", "corridor", "courtyard", "desert", "diner", "dock", "factory", "field", "forest", "foyer",
    "garage", "garden", "greenhouse", "hall", "hallway", "harbor", "hotel", "house", "kitchen", "lab",
    "lake", "lounge", "mansion", "market", "motel", "museum", "office", "palace", "parking", "pier",
    "pool", "quarry", "railway", "road", "rooftop", "room", "school", "set", "shore", "stage", "station",
    "street", "studio", "subway", "temple", "theater", "tower", "train", "tunnel", "vault", "warehouse",
    "workshop",
}


_NON_LOCATION_OBJECT_WORDS = {
    "bottle", "bowl", "bracelet", "camera", "candle", "chair", "collar", "comb", "crown", "dress",
    "frame", "glass", "kit", "knife", "locket", "mirror", "necklace", "perfume", "phone", "pocket",
    "razor", "rice", "ring", "shaving", "shoe", "sink", "sugar", "table", "tooth", "vanity", "veil", "window",
}


_LOCATION_SURFACE_WORDS = {
    "ceiling", "corner", "floor", "partition", "partitions", "wall", "walls",
}


_CHARACTER_WORDS = {
    "bride", "character", "face", "ghost", "girl", "hand", "human", "man", "person", "silhouette",
    "subject", "woman",
}


def _location_place_pattern():
    return r"(?:%s)" % "|".join(sorted((re.escape(word) for word in _LOCATION_PLACE_WORDS), key=len, reverse=True))


def _nice_location_name(value):
    text = re.sub(r"\s+", " ", str(value or "").strip(" ,.-"))
    if not text:
        return ""
    return text[:1].upper() + text[1:]


def _location_text_has_non_place_focus(value):
    tokens = set(re.findall(r"[a-z0-9]+", str(value or "").lower()))
    return bool(tokens & (_NON_LOCATION_OBJECT_WORDS | _LOCATION_SURFACE_WORDS | _CHARACTER_WORDS))


def _normalize_location_name(name):
    text = re.sub(r"\s+", " ", str(name or "").strip(" ,.-"))
    if not text:
        return ""
    lowered = text.lower()
    place_re = _location_place_pattern()
    tokens = set(re.findall(r"[a-z0-9]+", lowered))

    if re.search(r"\b(?:a|an|the|with|without|under|beneath|behind|beside|inside|outside|near|through|over|heavy|long|thick|deep|wide|large|small|empty|covered)\s*$", lowered):
        place_match = re.search(rf"^(.{{3,80}}?\b{place_re}\b)", text, flags=re.IGNORECASE)
        if place_match:
            return _nice_location_name(place_match.group(1))

    if tokens & (_NON_LOCATION_OBJECT_WORDS | _LOCATION_SURFACE_WORDS | _CHARACTER_WORDS):
        prep_match = re.search(
            rf"\b(?:in|inside|within|at|on|near|beside|behind|beneath|under)\s+(?:a|an|the)?\s*([^,.;]*?\b{place_re}\b(?:\s+\w+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if prep_match:
            return _nice_location_name(prep_match.group(1))

        surface_match = re.search(
            rf"^(.{{3,80}}?\b{place_re}\b)\s+(?:{'|'.join(sorted(_LOCATION_SURFACE_WORDS | _NON_LOCATION_OBJECT_WORDS, key=len, reverse=True))})\b",
            text,
            flags=re.IGNORECASE,
        )
        if surface_match:
            return _nice_location_name(surface_match.group(1))

        trailing_object_match = re.search(
            rf"\b({place_re})\s+(?:{'|'.join(sorted(_LOCATION_SURFACE_WORDS | _NON_LOCATION_OBJECT_WORDS, key=len, reverse=True))})\b",
            text,
            flags=re.IGNORECASE,
        )
        if trailing_object_match:
            before = text[:trailing_object_match.end(1)].strip()
            return _nice_location_name(before)

    return _nice_location_name(text)


def _clean_location_card(name, description, raw_name=None):
    normalized_name = _normalize_location_name(name)
    cleaned_description = _clean_location_description(normalized_name, description)
    raw_text = str(raw_name if raw_name is not None else name or "")
    if normalized_name.lower() != re.sub(r"\s+", " ", raw_text.strip(" ,.-")).lower() and _location_text_has_non_place_focus(raw_text):
        cleaned_description = normalized_name
    return normalized_name, cleaned_description


def _location_name_fragment_is_complete(fragment):
    text = re.sub(r"\s+", " ", str(fragment or "").strip().lower())
    if not text:
        return False
    if re.search(r"\b(?:a|an|the|with|without|under|beneath|behind|beside|inside|outside|near|through|over|heavy|long|thick|deep|wide|large|small|empty|covered)\s*$", text):
        return False
    return bool(re.search(rf"\b{_location_place_pattern()}\b", text, flags=re.IGNORECASE))


def _location_name_is_place(name):
    text = re.sub(r"\s+", " ", str(name or "").strip().lower())
    if not text:
        return False
    tokens = set(re.findall(r"[a-z0-9]+", text))
    if tokens & _LOCATION_PLACE_WORDS:
        if tokens & (_NON_LOCATION_OBJECT_WORDS | _LOCATION_SURFACE_WORDS | _CHARACTER_WORDS):
            normalized = _normalize_location_name(name).lower()
            normalized_tokens = set(re.findall(r"[a-z0-9]+", normalized))
            if normalized_tokens & (_NON_LOCATION_OBJECT_WORDS | _LOCATION_SURFACE_WORDS | _CHARACTER_WORDS):
                return False
        return True
    if tokens & (_NON_LOCATION_OBJECT_WORDS | _LOCATION_SURFACE_WORDS | _CHARACTER_WORDS):
        return False
    if re.search(r"\b(?:inside|outside|interior|exterior|room|space|area|zone)\b", text):
        return True
    return False


def _clean_location_description(name, description):
    name_text = re.sub(r"\s+", " ", str(name or "").strip())
    desc = re.sub(r"\s+", " ", str(description or "").strip())
    if not desc:
        return name_text
    fragments = [part.strip(" .") for part in re.split(r"\s*,\s*", desc) if part.strip(" .")]
    kept = []
    for index, fragment in enumerate(fragments):
        fragment_lower = fragment.lower()
        if index > 0 and re.search(
            r"\b(?:bride|woman|man|girl|boy|face|hand|human|silhouette|wearing|watching|pressed|resting|trailing|caught|hidden|dissolving|reflecting|smelling)\b",
            fragment_lower,
        ):
            continue
        if index > 0 and re.search(r"\b(?:razor|tooth|dress|veil|bottle|locket|collar|frame|kit|rice|sugar bowl)\b", fragment_lower):
            continue
        kept.append(fragment)
    cleaned = ", ".join(kept).strip()
    if not cleaned:
        cleaned = name_text
    if name_text and not cleaned.lower().startswith(name_text.lower()):
        cleaned = f"{name_text}, {cleaned}"
    return cleaned


def _valid_location_card(name, description):
    if not _location_name_is_place(name):
        return False
    combined = f"{name} {description}".lower()
    if re.search(r"\b(?:character|subject|woman|man|girl|boy|bride|ghostly face|human tooth|hand pressed)\b", combined):
        desc_without_name = str(description or "").lower().replace(str(name or "").lower(), "")
        if not re.search(r"\b(?:room|hall|hallway|corridor|street|road|forest|garden|stage|kitchen|bathroom|bedroom|ballroom|pool|motel|warehouse|studio|city)\b", desc_without_name):
            return False
    return True


def _clean_location_context_text(value):
    text = str(value or "").strip()
    if not text:
        return ""
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if re.search(r"\bsubjectsandscenes\.txt\b", line, flags=re.IGNORECASE):
            continue
        if re.search(r"\.(?:txt|json|srt)\b", line, flags=re.IGNORECASE) and re.search(r"[A-Za-z]:\\|/|\\", line):
            continue
        if _looks_like_location_meta_text(line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    if _looks_like_location_meta_text(cleaned):
        return ""
    return cleaned


def _parse_location_idea_lines(text):
    locations = []
    seen_names = set()
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^\s*music\s+video\s+locations\s*:?\s*$", line, flags=re.IGNORECASE):
            continue
        line = re.sub(r"^\s*(?:[-*\u2022]+|\d+[\).:-])\s*", "", line).strip()
        line = line.strip('"').strip()
        if not line or line in {"{", "}", "[", "]"} or line.startswith("{") or line.startswith("["):
            continue
        if re.match(r"^(?:location ideas?|output|user input)\s*:?\s*$", line, flags=re.IGNORECASE):
            continue
        name = line
        description = line
        split_match = re.match(r"^(.{4,80}?)(?:\s+-\s+|\s+--\s+|:\s+)(.{4,})$", line)
        if split_match:
            name = split_match.group(1).strip()
            description = split_match.group(2).strip()
        else:
            comma_parts = line.split(",", 1)
            if len(comma_parts) == 2 and 4 <= len(comma_parts[0].strip()) <= 60 and _location_name_fragment_is_complete(comma_parts[0]):
                name = comma_parts[0].strip()
        raw_name = name
        name, description = _clean_location_card(name, description, raw_name)
        if _looks_like_location_meta_text(name) or _looks_like_location_meta_text(description) or not _valid_location_card(name, description) or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        locations.append({"name": name, "description": description})
    return locations


def _parse_location_ideas_flexible(text):
    locations = _parse_location_idea_lines(text)
    if locations:
        return locations
    cleaned = _clean_visual_gemma_text(text)
    try:
        data = _extract_json_object_from_text(cleaned)
        if isinstance(data, dict):
            raw_locations = data.get("locations") or data.get("location_ideas") or data.get("music_video_locations")
        else:
            raw_locations = data
        if isinstance(raw_locations, list):
            parsed = []
            for item in raw_locations:
                if isinstance(item, dict):
                    raw_name = item.get("name") or item.get("location") or item.get("idea") or ""
                    name = _normalize_location_name(raw_name)
                    description = re.sub(r"\s+", " ", str(item.get("description") or item.get("detail") or item.get("visual_detail") or name).strip())
                else:
                    raw_name = item
                    name = _normalize_location_name(raw_name)
                    description = name
                name, description = _clean_location_card(name, description or name, raw_name)
                if name and not _looks_like_location_meta_text(name) and not _looks_like_location_meta_text(description) and _valid_location_card(name, description):
                    parsed.append({"name": name, "description": description or name})
            if parsed:
                return parsed
    except Exception:
        pass
    bulletish = re.split(r"(?:\n+|(?<=\.)\s+(?=[A-Z][A-Za-z ]{4,70}(?:\s+-|:)))", cleaned)
    parsed = _parse_location_idea_lines("\n".join(part.strip() for part in bulletish if part.strip()))
    if parsed:
        return parsed
    candidates = []
    for part in re.split(r";|\n", cleaned):
        part = re.sub(r"^\s*(?:Music Video Locations|Locations|Location ideas)\s*:?\s*", "", part.strip(), flags=re.I)
        if 4 <= len(part) <= 180 and not _looks_like_location_meta_text(part) and not re.search(r"\b(?:sorry|cannot|unable|lyrics|song meaning|summary)\b", part, flags=re.I):
            candidates.append(part)
    return _parse_location_idea_lines("\n".join(f"- {item}" for item in candidates))


def _parse_subject_lines(text):
    subjects = []
    for raw_line in str(text or "").splitlines():
      line = raw_line.strip().strip("-* ")
      if not line:
          continue
      match = re.match(r"^\s*(?:\d+[\).:\-|]\s*)?([^|:]+?)\s*\|\s*(.+)$", line)
      if match:
          name = re.sub(r"\s+", " ", match.group(1).strip())
          description = re.sub(r"\s+", " ", match.group(2).strip())
      else:
          match = re.match(r"^\s*(?:\d+[\).:\-|]\s*)?([^:]+?)\s*:\s*(.+)$", line)
          if not match:
              continue
          name = re.sub(r"\s+", " ", match.group(1).strip())
          description = re.sub(r"\s+", " ", match.group(2).strip())
      if name and description:
          subjects.append({"name": name, "description": description})
    return subjects


def _parse_scene_location_number_map(text, cleaned_scenes, locations):
    scene_map = {}
    scene_by_number = {str(index): scene["id"] for index, scene in enumerate(cleaned_scenes, start=1)}
    scene_by_label = {}
    for index, scene in enumerate(cleaned_scenes, start=1):
        scene_by_label[scene["id"].lower()] = scene["id"]
        scene_by_label[scene["label"].lower()] = scene["id"]
        scene_by_label[f"scene {index}"] = scene["id"]
        scene_by_label[f"scene{index}"] = scene["id"]
    location_by_number = {str(index): item["name"] for index, item in enumerate(locations, start=1)}
    location_by_name = {item["name"].lower(): item["name"] for item in locations}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip().strip(",")
        if not line:
            continue
        match = re.match(
            r'^\s*(?:Scene\s*)?("?[^"=:\|]+"?)\s*(?:=|:|\|)\s*(?:Location\s*)?("?[^",\s]+"?)',
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        raw_scene = match.group(1).strip().strip('"').lower()
        raw_location = match.group(2).strip().strip('"').lower()
        scene_id = scene_by_number.get(raw_scene) or scene_by_label.get(raw_scene) or scene_by_label.get(raw_scene.replace(" ", ""))
        location_name = location_by_number.get(raw_location) or location_by_name.get(raw_location)
        if scene_id and location_name:
            scene_map[scene_id] = location_name
    return scene_map


def _looks_like_gemma_repeat_failure(text):
    sample = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if not sample:
        return False

    compact = re.sub(r"[^a-z0-9_<>\-|]+", "", sample)
    for marker in (
        "completion-completion-completion",
        "thought-thought-thought",
        "de-facto-de-facto-de-facto",
        "de-fleshed",
        "cast-cast-cast",
        "prompt-cast-cast",
        "thoughtthoughtthought",
        "ownnessownnessownness",
        "nessnessnessness",
        "model_model_model",
        "modelmodelmodel",
        "end_anow",
        "thought_turn",
        "turn_turn",
        "<|channel>",
        "<channel|>",
    ):
        if marker in compact or marker in sample:
            return True

    if re.search(r"([a-z]{2,16})\1{5,}", compact):
        return True

    if re.search(r"(?:^|_)([a-z]{2,24})(?:_\1){5,}(?:_|$)", compact):
        return True

    if re.search(r"\b([a-zA-Z_]{3,})(?:[-\s]+\1){5,}\b", sample):
        return True

    unicode_tokens = re.findall(r"[\w']+", sample, flags=re.UNICODE)
    unicode_tokens = [token.strip("_'") for token in unicode_tokens if token.strip("_'")]
    if len(unicode_tokens) >= 16:
        token_counts = {}
        for token in unicode_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        if token_counts and max(token_counts.values()) >= 10 and max(token_counts.values()) / float(len(unicode_tokens)) >= 0.20:
            return True
        for size in (2, 3, 4):
            if len(unicode_tokens) < size * 4:
                continue
            phrase_counts = {}
            for index in range(len(unicode_tokens) - size + 1):
                phrase = " ".join(unicode_tokens[index:index + size])
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            if phrase_counts and max(phrase_counts.values()) >= 8:
                return True

    words = re.findall(r"[a-zA-Z_][a-zA-Z_']{2,}", sample)
    if len(words) < 18:
        return False

    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    common_words = {"the", "and", "with", "that", "this", "from", "into", "while", "during"}
    repeated_words = [
        count
        for word, count in counts.items()
        if word not in common_words
    ]
    if repeated_words and max(repeated_words) >= 10 and max(repeated_words) / float(len(words)) >= 0.25:
        return True

    phrases = [" ".join(words[index:index + 2]) for index in range(len(words) - 1)]
    if len(phrases) >= 12:
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        if max(phrase_counts.values()) >= 6:
            return True

    return False


def _looks_like_unfilled_prompt_template(text):
    sample = str(text or "").strip()
    if not sample:
        return False
    bracketed = re.findall(r"\[([^\]]{2,80})\]", sample)
    if len(bracketed) >= 2:
        return True
    placeholder_terms = (
        "subject",
        "setting",
        "environment",
        "time",
        "weather",
        "camera motion",
        "dynamic performance",
        "subject visibility",
        "framing",
        "reacts dynamically",
        "clothing",
        "hair",
    )
    lowered = sample.lower()
    if any(f"[{term}]" in lowered for term in placeholder_terms):
        return True
    if re.search(r"\[[^\]]*(?:subject|setting|environment|camera|motion|weather|lighting|dynamic|framing)[^\]]*\]", lowered):
        return True
    return False


def _looks_like_bad_reference_description(text):
    sample = re.sub(r"\s+", " ", str(text or "").strip())
    if not sample:
        return True
    if _looks_like_gemma_repeat_failure(sample) or _looks_like_unfilled_prompt_template(sample):
        return True
    tokens = re.findall(r"[\w']+", sample.lower(), flags=re.UNICODE)
    tokens = [token.strip("_'") for token in tokens if token.strip("_'")]
    if not tokens:
        return True
    if len(tokens) < 6 and len(set(tokens)) <= 2:
        return True
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    if counts:
        max_count = max(counts.values())
        if max_count >= 3 and max_count / float(len(tokens)) >= 0.45:
            return True
    for size in (1, 2, 3):
        if len(tokens) < size * 3:
            continue
        for index in range(0, len(tokens) - (size * 3) + 1):
            chunk = tokens[index:index + size]
            if all(tokens[index + offset:index + offset + size] == chunk for offset in range(size, size * 3, size)):
                return True
    alpha_chars = re.findall(r"[a-zA-Z]", sample)
    if len(alpha_chars) < 18:
        return True
    return False


def _looks_like_id_lora_script_prompt(text):
    sample = str(text or "").strip().lower()
    return all(label in sample for label in ("[visual]", "[speech]", "[sounds]"))


def _normalized_prompt_similarity_text(text):
    return " ".join(re.findall(r"[\w']+", str(text or "").lower(), flags=re.UNICODE))


def _looks_like_source_lyric_echo(text, payload):
    source = str((payload or {}).get("lyric_text") or (payload or {}).get("lyrics") or "").strip()
    candidate = str(text or "").strip()
    if not source or not candidate:
        return False
    source_norm = _normalized_prompt_similarity_text(source)
    candidate_norm = _normalized_prompt_similarity_text(candidate)
    if not source_norm or not candidate_norm:
        return False
    if candidate_norm == source_norm:
        return True
    source_tokens = source_norm.split()
    candidate_tokens = candidate_norm.split()
    if len(source_tokens) < 4 or len(candidate_tokens) < 4:
        return False
    source_set = set(source_tokens)
    candidate_set = set(candidate_tokens)
    overlap = len(source_set & candidate_set) / float(max(1, len(candidate_set)))
    length_ratio = min(len(source_tokens), len(candidate_tokens)) / float(max(len(source_tokens), len(candidate_tokens)))
    visual_terms = {
        "cinematic", "shot", "still", "frame", "portrait", "wide", "closeup", "close", "medium",
        "camera", "lens", "lighting", "background", "foreground", "composition", "scene",
        "environment", "location", "color", "palette", "texture", "detail", "reference",
    }
    has_visual_terms = bool(candidate_set & visual_terms)
    return overlap >= 0.90 and length_ratio >= 0.75 and not has_visual_terms


def _validate_reference_description(text, label):
    if _looks_like_bad_reference_description(text):
        raise ValueError(
            f"Gemma returned unusable repeated text for the {label} description. "
            "Try again, or use a clearer reference image."
        )


def _validate_builder_gemma_prompt(text, label, payload=None):
    if not str(text or "").strip():
        raise ValueError(f"Gemma returned an empty {label} prompt.")
    if _looks_like_source_lyric_echo(text, payload):
        raise ValueError(
            f"Gemma returned the scene lyrics instead of a usable {label} image prompt. "
            "Try again, add a short visual scene beat, or reduce Lyric Story Strength."
        )
    if _looks_like_gemma_repeat_failure(text):
        hint = "Try again or shorten the notes."
        if str(label or "").lower() != "flux/klein":
            hint = "Try again, shorten the notes, or turn off image reference."
        raise ValueError(
            f"Gemma returned repeated/thought text for the {label} prompt. "
            f"{hint}"
        )
    label_key = str(label or "").strip().lower().replace(" ", "_")
    if _looks_like_unfilled_prompt_template(text) and not (label_key in {"id-lora", "id_lora", "id-lora_i2v"} and _looks_like_id_lora_script_prompt(text)):
        raise ValueError(
            f"Gemma returned an unfilled template for the {label} prompt. "
            "Try again or add more specific scene/motion notes."
        )
