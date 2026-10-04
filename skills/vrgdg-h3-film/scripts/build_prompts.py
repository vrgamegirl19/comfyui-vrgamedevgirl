"""screenplay.json -> MiniMax H3 Ref2VA prompts (six sections) + storyboard/scenes.json, with a lint that stops on mistakes.

  python build_prompts.py <project>

Shot text tokens:
  {key}                 the scene's <Subject N> label for a cast member
  {say:KEY|how|line}    a spoken line: stable (Sx) id, the speaker's fixed voice description on first use, <d>[English] line</d>
Automatic fixes: <pause>/<long pause> -> <breath> (H3 speaks "pause" aloud); <softer> removed (also spoken aloud);
on-screen speakers get a lip-sync sentence; voice-only speakers get a lips-closed sentence for everyone in frame.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_settings, project_dir, save_json  # noqa: E402

NEGATIVE = re.compile(r"\b(no|not|never|without|avoid|don't|doesn't|isn't|aren't|none|nobody|nothing|free of)\b|n't\b", re.I)
SPOKEN_TAGS = re.compile(r"<(catches breath|pants|/whisper)>", re.I)
SOFT_MALE = re.compile(r"\b(soft|softly|gentle|gently|sweet|sweetly|lullaby|whisper\w*|tender\w*)\b", re.I)
WARMUP_S_PER_FRAME = 1 / 24
WORDS_PER_SECOND = 2.6    # H3's natural dialogue pace; each <breath> adds about 0.4 s


def stamp(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def speech_seconds(line):
    words = len(re.findall(r"[A-Za-z0-9']+", re.sub(r"<[^>]+>", " ", line)))
    return words / WORDS_PER_SECOND + 0.4 * len(re.findall(r"<(?:deep )?breath>", line))


def lint(prompt):
    outside = re.sub(r"<d>.*?</d>", "", prompt, flags=re.S).replace("non_diegetic_music", "")
    return sorted({m.group(0) for m in NEGATIVE.finditer(outside)})


def build_scene(sp, settings, num, scene, problems, warnings):
    cast_keys, loc_key = scene["cast"], scene["loc"]
    cast, voices, sex = sp["cast"], sp["voices"], sp["sex"]
    voice_only = sp.get("voice_only", {})
    labels = {k: f"<Subject {i}>" for i, k in enumerate(cast_keys, start=1)}
    speakers = {}
    warmup_s = settings["warmup_frames"] * WARMUP_S_PER_FRAME

    def say(m):
        key, how, line = m.group(1).split("|", 2)
        if key not in voices:
            problems.append(f"scene {num}: speaker '{key}' has no voice description")
            return m.group(0)
        first = key not in speakers
        sid = speakers.setdefault(key, f"S{len(speakers) + 1}")
        line = re.sub(r"([.,?!])?\s*<(?:long )?pause>\s*", lambda x: (x.group(1) or ".") + " <breath> ", line)
        line = re.sub(r"\s*<softer>\s*", " ", line).strip()
        if SPOKEN_TAGS.search(line):
            problems.append(f"scene {num}: tag {SPOKEN_TAGS.search(line).group(0)} is spoken aloud by H3; use <breath>")
        if sex.get(key) == "M" and SOFT_MALE.search(how):
            warnings.append(f"scene {num}: '{SOFT_MALE.search(how).group(0)}' in a male delivery can flip the voice female")
        cue = f"<d>[English] {line}</d>"
        if key in voice_only:
            name = voice_only[key]["name"]
            word = {"M": "adult male", "F": "adult female"}.get(sex.get(key), "")
            voice = f", {voices[key]}," if first else ""
            return (f"{name[0].upper() + name[1:]}'s {word} voice ({sid}){voice} {how}: {cue} Every person in the frame keeps "
                    f"their lips closed while {name}'s voice speaks.")
        if key not in labels:
            problems.append(f"scene {num}: '{key}' speaks but is neither in the scene cast nor voice_only")
            return m.group(0)
        pos = cast[key]["pos"]
        voice = f" in {voices[key]}" if first else ""
        return f"{labels[key]} ({sid}) {how}{voice}: {cue} {pos.capitalize()} lips, mouth and jaw visibly form every word."

    body, speech_end = [], 0.0
    for n, (at, text) in enumerate(scene["shots"], start=1):
        lines = [m.split("|", 2)[2] for m in re.findall(r"\{say:([^}]*)\}", text)]
        if lines:
            speech_end = max(speech_end, at + sum(speech_seconds(x) for x in lines))
        text = re.sub(r"\{say:([^}]*)\}", say, text)
        for key, label in labels.items():
            text = text.replace("{" + key + "}", label)
        if re.search(r"\{[a-z_]+\}", text):
            problems.append(f"scene {num}: unknown token {re.search(r'{[a-z_]+}', text).group(0)}")
        head = "[Shot 1] " if n == 1 else f"[Shot {n}] At {stamp(at + warmup_s)}, the shot cuts to a new angle. "
        body.append(head + text)
    hold = settings.get("tail_hold_seconds", 1.0)
    for i in range(len(body) - 1, -1, -1):   # a breath after the scene's last line pads it so its last word lands early
        cut = body[i].rfind("</d>")
        if cut >= 0:
            if not body[i][:cut].rstrip().endswith(">"):
                body[i] = body[i][:cut].rstrip() + " <breath></d>" + body[i][cut + 4:]
                speech_end += 0.4
            break
    if speakers:   # H3 stretches a line to fill the clip, so give it an end point and a silent beat after it
        body[-1] += (f" The last spoken word ends by {stamp(scene['dur'] - hold + warmup_s)}, and the shot then holds in a "
                     f"silent beat until the end of the clip, every mouth closed, with only breathing and room sound.")
        if speech_end > scene["dur"] - 0.3:
            problems.append(f"scene {num}: the lines need about {speech_end:.1f} s but the scene is {scene['dur']} s; "
                            f"shorten a line, cut to it earlier, or lengthen the scene")
        elif speech_end > scene["dur"] - hold:
            warnings.append(f"scene {num}: the last line should end around {speech_end:.1f} s, inside the {hold} s silent hold "
                            f"before {scene['dur']} s; shorten it or start it earlier so the last word is not cut")
    shot_list = ", ".join(f"[Shot {n}]" for n in range(1, len(scene["shots"]) + 1))

    loc = sp["locations"][loc_key]
    defs, keep, images = [], [], []
    for key in cast_keys:
        c, i = cast[key], labels[key][9:-1]
        defs.append(f"{labels[key]} is {c['name']}, the person in <Picture {i}>, {c['desc']}. <Picture {i}> is a studio "
                    f"reference photo that defines {c['pos']} identity, face, hair and wardrobe."
                    + (f" When {c['name']} speaks, {c['pos']} voice is {voices[key]}." if key in voices else ""))
        keep.append(f"{labels[key]} (appears in {shot_list}): fully_preserved - {c['name']}'s face, hair and wardrobe stay "
                    f"consistent with <Picture {i}>.")
        images.append(c["ref"])
    if loc.get("ref"):
        n_loc = len(cast_keys) + 1
        defs.append(f"<Subject {n_loc}> is the environment in <Picture {n_loc}>, used as the location, lighting and atmosphere "
                    f"reference: {loc['desc']}.")
        keep.append(f"<Subject {n_loc}> (appears in {shot_list}): fully_preserved - {loc['name']}, its layout, palette and "
                    f"light stay consistent with <Picture {n_loc}>.")
        images.append(loc["ref"])
    else:   # text-only location: used when a plate's layout would fight the shot
        body[0] = body[0].replace("[Shot 1] ", f"[Shot 1] The scene takes place in {loc['desc']}. ", 1)
    for key in speakers:
        if key in voice_only:
            defs.append(f"{voice_only[key]['def']} Its voice is {voices[key]}; it is always the same voice.")

    style = sp["styles"].get(scene.get("style", "default"), sp["styles"]["default"])
    prompt = "\n\n".join([
        "subject_definitions:\n" + "\n".join(defs),
        f"summary:\n[reference generation] The target video is one scene of a {sp.get('genre', 'photorealistic')} short film "
        f"set in {loc['name']}: {scene['summary']}",
        "retention_analysis:\n" + "\n".join(keep),
        f"detailed_description:\n{style}\n" + "\n".join(body),
        f"overall_soundscape:\n{scene.get('sound', 'Natural room tone.')}",
        f"non_diegetic_music:\n{scene.get('music', 'N/A')}",
    ])
    return prompt, images, speakers


def main():
    project, _ = project_dir()
    settings = load_settings(project)
    sp = json.load(open(os.path.join(project, "screenplay.json"), encoding="utf-8"))
    scenes, problems, warnings, rows = [], [], [], []
    t = settings["film_offset"]
    for num, scene in enumerate(sp["scenes"], start=1):
        if scene["dur"] > settings["max_scene_seconds"]:
            problems.append(f"scene {num}: {scene['dur']} s is longer than {settings['max_scene_seconds']} s")
        if len(scene["cast"]) > settings["max_cast_per_shot"]:
            problems.append(f"scene {num}: {len(scene['cast'])} cast members (max {settings['max_cast_per_shot']})")
        prompt, images, speakers = build_scene(sp, settings, num, scene, problems, warnings)
        for s in ("M", "F"):
            same = [k for k in speakers if sp["sex"].get(k) == s]
            if len(same) > 1:
                problems.append(f"scene {num}: two {s} speakers {same} share one scene; H3 mixes same-sex voices")
        silent = [k for k in scene["cast"] if k not in speakers]
        all_text = " ".join(x for _, x in scene["shots"])
        for k in silent:
            if speakers and "{" + k + "}'s lips" not in all_text and (len(scene["cast"]) > 1 or "lips" not in all_text):
                warnings.append(f"scene {num}: silent on-screen '{k}' has no \"{{{k}}}'s lips stay pressed together\" line")
        if scene.get("continuity") and not re.search(r"\|\s*<(deep )?breath>", all_text) and speakers:
            warnings.append(f"scene {num}: continuation scenes should open their first line with <breath>")
        bad = lint(prompt)
        if bad:
            problems.append(f"scene {num}: negative wording {bad} (H3 renders what is named; describe what IS on screen)")
        if len(prompt) > settings["prompt_limit"]:
            problems.append(f"scene {num}: prompt is {len(prompt)} chars (limit {settings['prompt_limit']})")
        missing = [p for p in images if not os.path.isfile(os.path.join(project, p))]
        if missing:
            warnings.append(f"scene {num}: reference images not made yet {missing}")
        mode = {"exact_frame": "latent_continuation_exact_frame", "latent": "latent_continuation"}.get(scene.get("continuity", ""), "off")
        frame = os.path.join(project, "builder_project", "continuity_frames", f"scene_{num - 1:03d}_last.png")
        scenes.append({"scene_number": num, "start": round(t, 3), "end": round(t + scene["dur"], 3), "location": scene["loc"],
                       "cast": scene["cast"], "speakers": speakers, "continuity_mode": mode,
                       "latent_exact_frame_path": frame if mode.endswith("exact_frame") else "",
                       "prompt": prompt, "image_paths": [os.path.join(project, p) for p in images],
                       "dialogue": re.findall(r"<d>\[English\] (.*?)</d>", prompt)})
        with open(os.path.join(project, "prompts", f"scene_{num:03d}.txt"), "w", encoding="utf-8") as fh:
            fh.write(prompt)
        rows.append(f"| {num} | {stamp(t - settings['film_offset'])} | {scene['dur']:.1f}s | {sp['locations'][scene['loc']]['name']} "
                    f"| {', '.join(scene['cast'])} | {scene['summary']} | {len(prompt)} |")
        t += scene["dur"]
    save_json(os.path.join(project, "storyboard", "scenes.json"), {"film_offset": settings["film_offset"], "scenes": scenes})
    with open(os.path.join(project, "storyboard", "STORYBOARD.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# {sp['title']} storyboard\n\n| # | At | Len | Location | Cast | Beat | Prompt chars |\n"
                 "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    print(f"{len(scenes)} scenes, {t - settings['film_offset']:.1f}s")
    for w in warnings:
        print("warning:", w)
    for p in problems:
        print("PROBLEM:", p)
    print("lint ok" if not problems else "lint FAILED")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
