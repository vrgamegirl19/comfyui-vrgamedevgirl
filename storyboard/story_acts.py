"""Story Mode acts and scene plan rules (pure code, no LLM).

A chapter owns a contiguous run of timeline scenes. Its acts split that run into contiguous pieces, and every
scene belongs to exactly one act. Scene numbers here are 1-based timeline positions.

JS twin of the edit operations: ``web/music_video_builder/story_acts_data.mjs`` (checked against
``tests/story_acts_cases.json``).
"""

import re

from .story_mode import DEFAULT_SPOT_MIN_SCENES  # noqa: F401  (re-exported for the plan builder)


def allocate_chapter_scenes(paragraph_counts, scene_count):
    """Scenes per chapter, proportional to each chapter's paragraphs (largest remainder), at least 1 each.

    With more chapters than scenes the last chapters get 0 scenes.
    """
    weights = [max(1, int(count or 0)) for count in paragraph_counts]
    if not weights or scene_count <= 0:
        return [0] * len(weights)
    if len(weights) >= scene_count:
        return [1] * scene_count + [0] * (len(weights) - scene_count)
    spare = scene_count - len(weights)
    total = sum(weights)
    exact = [spare * weight / total for weight in weights]
    counts = [1 + int(value) for value in exact]
    left = scene_count - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - int(exact[i])), i))
    for index in order[:left]:
        counts[index] += 1
    return counts


def chapter_ranges(counts):
    """``[(first, last)]`` 1-based scene ranges for chapter scene counts; ``None`` for a chapter with no scenes."""
    ranges, start = [], 1
    for count in counts:
        if count <= 0:
            ranges.append(None)
            continue
        ranges.append((start, start + count - 1))
        start += count
    return ranges


def repair_act_ranges(proposed, first, last):
    """Contiguous act ranges covering ``first..last`` built from the LLM's ``[(first, last)]`` proposals.

    Proposals that start outside the range are dropped; the rest are sorted. Each act starts where its proposal
    starts, the first act starts at ``first`` and every act ends where the next begins. Nothing usable gives one act.
    """
    starts = []
    for item in proposed or []:
        try:
            start = int(item[0])
        except (TypeError, ValueError, IndexError):
            continue
        if first <= start <= last and start not in starts:
            starts.append(start)
    starts.sort()
    if not starts or starts[0] != first:
        starts.insert(0, first)
    return [(start, (starts[i + 1] - 1) if i + 1 < len(starts) else last) for i, start in enumerate(starts)]


def keep_notes_inside_acts(ranges, note_ranges):
    """Move act borders so no timed note range is cut in two. Borders move to the nearer edge of the note.

    ``note_ranges`` are ``(first, last)`` scene numbers. A note longer than the whole chapter is left alone.
    """
    if len(ranges) < 2:
        return list(ranges)
    first, last = ranges[0][0], ranges[-1][1]
    starts = [start for start, _end in ranges[1:]]
    for note_first, note_last in note_ranges or []:
        if note_first <= first and note_last >= last:
            continue
        moved = []
        for start in starts:
            if note_first < start <= note_last:
                before, after = note_first, note_last + 1
                start = before if (start - before) <= (after - start) else after
            moved.append(start)
        starts = moved
    starts = sorted({start for start in starts if first < start <= last})
    bounds = [first, *starts]
    return [(start, (bounds[i + 1] - 1) if i + 1 < len(bounds) else last) for i, start in enumerate(bounds)]


def fallback_act_ranges(paragraph_count, first, last):
    """One act per paragraph when the LLM gives nothing usable, scenes shared out evenly."""
    scene_total = last - first + 1
    count = max(1, min(int(paragraph_count or 1), scene_total))
    sizes = allocate_chapter_scenes([1] * count, scene_total)
    return [(first + a - 1, first + b - 1) for a, b in chapter_ranges(sizes)]


def _spot_runs(spots):
    runs, start = [], 0
    for index in range(1, len(spots) + 1):
        if index == len(spots) or spots[index].lower() != spots[start].lower():
            runs.append((start, index - 1))
            start = index
    return runs


def enforce_spot_runs(spots, min_run, locked=()):
    """Spots with every run at least ``min_run`` scenes long.

    Blank spots take the spot before them. The shortest short run is merged into its longer neighbour (the one before
    it on a tie), again until no short run is left. Runs holding a scene in ``locked`` (indexes whose spot came from a
    user note) are kept.
    """
    result = [str(spot or "").strip() for spot in spots]
    if not result:
        return result
    min_run = max(1, int(min_run or 1))
    locked = set(locked)
    if not result[0]:
        result[0] = next((spot for spot in result if spot), "")
    for index in range(1, len(result)):
        if not result[index]:
            result[index] = result[index - 1]
    while True:
        runs = _spot_runs(result)
        short = [
            (b - a + 1, position) for position, (a, b) in enumerate(runs)
            if b - a + 1 < min_run and not any(i in locked for i in range(a, b + 1))
        ]
        if len(runs) < 2 or not short:
            return result
        _length, position = min(short)
        a, b = runs[position]
        before = runs[position - 1] if position > 0 else None
        after = runs[position + 1] if position + 1 < len(runs) else None
        size = lambda run: -1 if run is None else run[1] - run[0] + 1  # noqa: E731
        target = before if size(before) >= size(after) else after
        for i in range(a, b + 1):
            result[i] = result[target[0]]


def act_scene_numbers(act, scene_number_by_id):
    numbers = [scene_number_by_id[sid] for sid in act.get("scene_ids") or [] if sid in scene_number_by_id]
    return sorted(numbers)


def validate_story_acts(acts, chapters, scene_ids):
    """Problems with an edited act list, as plain sentences. Empty means the acts are valid.

    Rules: every scene is in exactly one act, each act's scenes are contiguous, acts are in timeline order, every act
    belongs to a chapter of the book, and chapters keep book order along the timeline.
    """
    problems = []
    number_by_id = {sid: index + 1 for index, sid in enumerate(scene_ids)}
    chapter_order = {chapter["id"]: index for index, chapter in enumerate(chapters)}
    seen = {}
    previous_last, previous_chapter = 0, -1
    for act in acts:
        label = act.get("title") or act.get("id") or "An act"
        if act.get("chapter_id") not in chapter_order:
            problems.append(f"{label} belongs to no chapter of the story.")
            continue
        unknown = [sid for sid in act.get("scene_ids") or [] if sid not in number_by_id]
        if unknown:
            problems.append(f"{label} lists scenes that are not on the timeline: {', '.join(unknown)}.")
        numbers = act_scene_numbers(act, number_by_id)
        if not numbers:
            problems.append(f"{label} has no scenes.")
            continue
        for sid in act.get("scene_ids") or []:
            if sid in seen:
                problems.append(f"Scene {number_by_id.get(sid, sid)} is in both {seen[sid]} and {label}.")
            seen[sid] = label
        if numbers != list(range(numbers[0], numbers[-1] + 1)):
            problems.append(f"{label} skips scenes; an act's scenes must be next to each other.")
        if numbers[0] <= previous_last:
            problems.append(f"{label} is out of timeline order.")
        order = chapter_order[act["chapter_id"]]
        if order < previous_chapter:
            problems.append(f"{label} puts its chapter before an earlier chapter on the timeline.")
        previous_last, previous_chapter = numbers[-1], order
    missing = [str(number_by_id[sid]) for sid in scene_ids if sid not in seen]
    if missing:
        problems.append(f"Scenes not in any act: {', '.join(missing)}.")
    return problems


# ---------------------------------------------------------------------------
# Edit operations (the Acts tab)
# ---------------------------------------------------------------------------

def _renumber(acts):
    return [{**act, "id": f"act_{index + 1}"} for index, act in enumerate(acts)]


def move_act_border(acts, act_index, new_first_scene_id, scene_ids):
    """Move the border between act ``act_index - 1`` and ``act_index`` so act ``act_index`` starts at the scene.

    Both acts keep at least one scene and must be in the same chapter; otherwise the acts are returned unchanged.
    """
    if act_index <= 0 or act_index >= len(acts) or new_first_scene_id not in scene_ids:
        return [dict(act) for act in acts]
    before, after = acts[act_index - 1], acts[act_index]
    if before.get("chapter_id") != after.get("chapter_id"):
        return [dict(act) for act in acts]
    joined = list(before.get("scene_ids") or []) + list(after.get("scene_ids") or [])
    if new_first_scene_id not in joined:
        return [dict(act) for act in acts]
    cut = joined.index(new_first_scene_id)
    if cut <= 0 or cut >= len(joined):
        return [dict(act) for act in acts]
    result = [dict(act) for act in acts]
    result[act_index - 1] = {**before, "scene_ids": joined[:cut]}
    result[act_index] = {**after, "scene_ids": joined[cut:]}
    return result


def merge_acts(acts, act_index):
    """Merge act ``act_index`` into the act before it when both are in the same chapter."""
    if act_index <= 0 or act_index >= len(acts):
        return [dict(act) for act in acts]
    before, after = acts[act_index - 1], acts[act_index]
    if before.get("chapter_id") != after.get("chapter_id"):
        return [dict(act) for act in acts]
    merged = {
        **before,
        "summary": " ".join(part for part in (before.get("summary"), after.get("summary")) if part),
        "cast_ids": list(dict.fromkeys([*(before.get("cast_ids") or []), *(after.get("cast_ids") or [])])),
        "scene_ids": [*(before.get("scene_ids") or []), *(after.get("scene_ids") or [])],
    }
    return _renumber([*acts[:act_index - 1], merged, *acts[act_index + 1:]])


def split_act(acts, act_index, first_scene_id_of_new_act):
    """Split an act in two; the new act starts at the given scene. Both parts keep at least one scene."""
    if act_index < 0 or act_index >= len(acts):
        return [dict(act) for act in acts]
    act = acts[act_index]
    scene_ids = list(act.get("scene_ids") or [])
    if first_scene_id_of_new_act not in scene_ids or scene_ids.index(first_scene_id_of_new_act) == 0:
        return [dict(item) for item in acts]
    cut = scene_ids.index(first_scene_id_of_new_act)
    first = {**act, "scene_ids": scene_ids[:cut]}
    second = {**act, "title": f"{act.get('title') or 'Act'} (continued)", "scene_ids": scene_ids[cut:]}
    return _renumber([*acts[:act_index], first, second, *acts[act_index + 1:]])


def clean_name(value, limit=160):
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]
