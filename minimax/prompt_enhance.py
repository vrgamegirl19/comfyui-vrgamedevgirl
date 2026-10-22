"""MiniMax H3 "Enhance Prompts": put an LLM's rewritten [Shot N] blocks back into the original prompt.

The LLM sees the whole finished prompt, but only its shot blocks are taken from the reply. Everything before the
first shot and after the last one stays as it was, except for the scene-wide continuity paragraph and duplicate
music-video wording in the style line. A rewrite that loses a label, a cue time, a sung word or a "from X s to Y s" timing is refused, so a bad reply
never replaces a working prompt.
The separate reference-image task may also replace mapped visual definitions before this same shot validation.
"""

import re
from typing import List, Tuple

from .shot_prompt import HARD_LIMIT

# Definitions/retention may mention [Shot N] inline; only a line-opening label begins a shot block.
_SHOT_LABEL = re.compile(r"^[ \t]*\[Shot\s+(\d+)\]", re.M)
_SHOT_SPLIT = re.compile(r"(?=^[ \t]*\[Shot\s+\d+\])", re.M)
# A structured section header such as "overall_soundscape:" on its own line ends the shot blocks.
_SECTION = re.compile(r"\n\s*\n(?=[a-z][a-z_]*:[ \t]*(?:\n|$))")
_LEAD = re.compile(r"^\[Shot\s+\d+\]\s*(At\s+\d{1,2}:\d{2}(?:\.\d+)?,)?\s*")
_REFERENCE = re.compile(r"<(?:Subject|Picture|Audio|Video)\s+\d+>")
_TIMING = re.compile(r"\bfrom\s+\d+(?:\.\d+)?\s*s\s+to\s+\d+(?:\.\d+)?\s*s\b", re.I)
_CUE = re.compile(r"<d>([\s\S]*?)</d>", re.I)
_QUOTE = re.compile(r"\"([^\"\n]+)\"|“([^”\n]+)”")
CONTINUOUS_PERFORMANCE = (
    "All shots follow one continuous performance, preserving every person's exact position, body orientation, "
    "posture, and ongoing movement across every camera cut; each new shot continues directly from the final "
    "physical moment of the previous shot, with only the camera angle and framing changing."
)
CONTINUATION = "continuing directly from the previous shot's final moment,"
_CONTINUATION_START = re.compile(
    r"^continuing\s+directly\s+from\s+the\s+previous\s+shot['’]s\s+final\s+moment\s*,?\s*", re.I
)


class PromptEnhanceError(ValueError):
    """The enhanced prompt broke the prompt's fixed parts; the original prompt is kept."""


def split_prompt(text: str) -> Tuple[str, List[str], str]:
    """Split a prompt into the text before the shots, the [Shot N] blocks, and the text after them."""
    source = str(text or "").replace("\r\n", "\n")
    first = _SHOT_LABEL.search(source)
    if not first:
        raise PromptEnhanceError("The prompt has no [Shot N] blocks to enhance.")
    rest = source[first.start():]
    end = _SECTION.search(rest)
    body, suffix = (rest[:end.start()], rest[end.start():]) if end else (rest, "")
    blocks = [block.strip() for block in _SHOT_SPLIT.split(body) if block.strip()]
    return source[:first.start()], blocks, suffix


def _clean_reply(text: str) -> str:
    reply = re.sub(r"```[a-z]*", "", str(text or ""), flags=re.I)
    return re.sub(r"\*\*(\[Shot\s+\d+\])\*\*", r"\1", reply).replace("\r\n", "\n").strip()


def _words(text: str) -> List[str]:
    """Sung words only: tags, emotion headers and punctuation removed."""
    plain = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", str(text or ""))
    return re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)*", plain.lower())


def _quotes(text: str) -> List[List[str]]:
    return [_words(a or b) for a, b in _QUOTE.findall(text)]


def _shot_number(block: str) -> int:
    return int(_SHOT_LABEL.match(block).group(1))


def _keep_lead(original: str, enhanced: str, *, multi_shot: bool = False, first: bool = False) -> str:
    """Give the rewritten shot the original "[Shot N] At 00:00.000," opening."""
    lead = _LEAD.match(original).group(0).strip()
    body = _LEAD.sub('', enhanced, count=1).strip()
    if multi_shot:
        if first:
            # Migrate older replies that placed the global rule inside Shot 1.
            while body.startswith(CONTINUOUS_PERFORMANCE):
                body = body[len(CONTINUOUS_PERFORMANCE):].strip()
        else:
            body = f"{CONTINUATION} {_CONTINUATION_START.sub('', body, count=1)}"
    return f"{lead} {body}"


def _scene_opening(prefix: str, multi_shot: bool) -> str:
    """Keep the global rule after the detailed-description opening, before Shot 1."""
    prefix = re.sub(
        r"(?m)^(The target video is in a [^\n]*)$",
        lambda match: re.sub(r"\bmusic[ -]video(?:\s+music[ -]video)+(?=\s+style\b)",
                             "music-video", match.group(0), flags=re.I),
        prefix,
    )
    if multi_shot:
        prefix = prefix.replace(CONTINUOUS_PERFORMANCE, "").rstrip()
        prefix = (prefix + "\n\n" if prefix else "") + CONTINUOUS_PERFORMANCE + "\n\n"
    return prefix


def _check_shot(number: int, original: str, enhanced: str, allowed_labels: set) -> None:
    where = f"[Shot {number}]"
    missing = sorted(set(_REFERENCE.findall(original)) - set(_REFERENCE.findall(enhanced)))
    if missing:
        raise PromptEnhanceError(f"{where} lost {', '.join(missing)}.")
    unknown = sorted(set(_REFERENCE.findall(enhanced)) - allowed_labels)
    if unknown:
        raise PromptEnhanceError(f"{where} added {', '.join(unknown)}, which the prompt does not define.")
    if [t.lower() for t in _TIMING.findall(original)] != [t.lower() for t in _TIMING.findall(enhanced)]:
        raise PromptEnhanceError(f"{where} changed a 'from X s to Y s' timing.")
    if [_words(cue) for cue in _CUE.findall(original)] != [_words(cue) for cue in _CUE.findall(enhanced)]:
        raise PromptEnhanceError(f"{where} changed the sung or spoken words inside <d>...</d>.")
    enhanced_quotes = _quotes(enhanced)
    if any(quote not in enhanced_quotes for quote in _quotes(original)):
        raise PromptEnhanceError(f"{where} changed a quoted lyric.")


def merge_enhanced_prompt(original: str, reply: str) -> str:
    """The original prompt with its shot blocks replaced by the reply's. Raises ``PromptEnhanceError``."""
    prefix, old_blocks, suffix = split_prompt(original)
    cleaned = _clean_reply(reply)
    first = _SHOT_LABEL.search(cleaned)
    if not first:
        raise PromptEnhanceError("The LLM reply has no [Shot N] blocks.")
    _, new_blocks, _ = split_prompt(cleaned[first.start():])
    old_numbers = [_shot_number(block) for block in old_blocks]
    new_numbers = [_shot_number(block) for block in new_blocks]
    if new_numbers != old_numbers:
        raise PromptEnhanceError(f"The LLM returned shots {new_numbers}; the prompt has shots {old_numbers}.")
    allowed = set(_REFERENCE.findall(original))
    merged = []
    for index, (number, old, new) in enumerate(zip(old_numbers, old_blocks, new_blocks)):
        # Check the actual rewrite before adding boilerplate so an empty reply still fails.
        if len(_LEAD.sub("", new, count=1).strip()) < 24:
            raise PromptEnhanceError(f"[Shot {number}] came back empty.")
        new = _keep_lead(old, new, multi_shot=len(old_blocks) > 1, first=index == 0)
        _check_shot(number, old, new, allowed)
        merged.append(new)
    if not re.search(r"[.!?…\"”]\s*$", merged[-1]):
        raise PromptEnhanceError(f"[Shot {old_numbers[-1]}] is cut off.")
    body = "\n\n".join(merged)
    prefix = _scene_opening(prefix, len(old_blocks) > 1)
    prompt = f"{prefix}{body}{suffix}".strip()
    if len(prompt) > HARD_LIMIT:
        raise PromptEnhanceError(f"The enhanced prompt is {len(prompt)} characters, over the {HARD_LIMIT} maximum.")
    return prompt


def merge_reference_enhanced_prompt(original: str, reply: dict, references: list) -> str:
    """Allow only mapped visual definitions to change outside the usual validated shot rewrite."""
    if not isinstance(reply, dict) or not isinstance(reply.get("prompt"), str):
        raise PromptEnhanceError("Reference enhancement must return JSON with prompt and reference_descriptions.")
    descriptions = reply.get("reference_descriptions")
    expected = {item["label"]: item for item in references if item.get("describe")}
    if not isinstance(descriptions, list) or len(descriptions) != len(expected):
        raise PromptEnhanceError("Reference enhancement did not describe every mapped subject/location exactly once.")
    definitions = []
    seen = set()
    for entry in descriptions:
        if not isinstance(entry, dict):
            raise PromptEnhanceError("Invalid reference description.")
        label = entry.get("label")
        if label not in expected or label in seen:
            raise PromptEnhanceError(f"Unexpected or repeated reference description: {label}.")
        seen.add(label)
        description = entry.get("description")
        if (not isinstance(description, str) or not 12 <= len(description.strip()) <= 450
                or re.search(r"[<>\[\]\r\n]", description)):
            raise PromptEnhanceError(f"{label} needs one brief plain-text visual description.")
        item = expected[label]
        name = re.sub(r"[<>\[\]\r\n]", "", str(item.get("name") or item.get("kind") or "reference"))
        purpose = re.sub(r"[<>\[\]\r\n]", "", str(item.get("purpose") or "visual reference"))
        definitions.append((label, f"{label} is {name} in {item['picture_label']}: {description.strip().rstrip('.')}. Reference role: {purpose}."))
    prefix, blocks, suffix = split_prompt(original)
    # Compact image+reference prompts may not yet declare their start/end frame anchors.
    # Cover the exact connected picture order when adding subject definitions.
    for item in references:
        if item.get("describe") or item.get("kind") not in {"start_frame", "end_frame"}:
            continue
        label = item["picture_label"]
        if re.search(r"(?m)^" + re.escape(label) + r"\s+is\b", prefix):
            continue
        first_frame = item["kind"] == "start_frame"
        shot = _shot_number(blocks[0] if first_frame else blocks[-1])
        position = "first" if first_frame else "last"
        definitions.append((label, f"{label} is the {position} frame of [Shot {shot}], used as the exact {position}-frame visual-state anchor."))
    section = re.search(r"(?m)^subject_definitions:[ \t]*\n", prefix)
    if section:
        end = re.search(r"(?m)^[a-z][a-z_]*:[ \t]*(?:\n|$)", prefix[section.end():])
        stop = section.end() + end.start() if end else len(prefix)
        content = prefix[section.end():stop].rstrip()
        for label, definition in definitions:
            # Replace the mapped definition only; preserve audio, other definitions and fixed sections.
            pattern = r"(?m)^" + re.escape(label) + r"[^\n]*$"
            if re.search(pattern, content):
                content = re.sub(pattern, lambda _: definition, content)
            else:
                content += "\n" + definition
        prefix = prefix[:section.end()] + content + "\n\n" + prefix[stop:]
    else:
        prefix = "subject_definitions:\n" + "\n".join(line for _, line in definitions) + "\n\n" + prefix
    enriched_original = prefix + "\n\n".join(blocks) + suffix
    return merge_enhanced_prompt(enriched_original, reply["prompt"])
