"""Build, check and apply MiniMax H3 reference-to-video prompts from the scene spec (spec.py in the work folder).

usage:
  build_prompts.py --work W                    build + check every scene; write prompts/scene_NN.txt, render_plan.json
  build_prompts.py --work W --apply 3 7 12     ...and save those scenes to the Builder (prompt, references, settings)
  build_prompts.py --work W --apply all
  build_prompts.py --work W --spec spec_alt.py --project "<copy id>" --apply 8 9    continuation alternates

The spec format is documented in assets/spec_template.py and the writing rules in references/prompt-writing.md.
A scene prompt has the six sections MiniMax H3 expects: subject_definitions, summary, retention_analysis,
detailed_description ([Shot N] At MM:SS.mmm,), overall_soundscape, non_diegetic_music.

Checks (PROBLEM blocks --apply for that scene; WARN does not):
  PROBLEM  over 7000 characters, negative words outside quotes and <d> cues (H3 renders what is named),
           wet/tear words, forbidden words outside their variant (instruments/mic off stage), a required word
           missing in a variant (mic on stage), an unused <Subject N>, a leftover {token}, shot times out of order.
  WARN     face-colour words (purple/red-faced/blush read as skin colour or romance), a <d> cue whose words or
           window do not match the words actually sung in that shot (lyrics_timed.json + timeline.json),
           a shot with vocals but no singing direction, "low-angle full-body" framing while the singer sings,
           a first shot that opens on someone other than the scene's lead (warm-up trims leave a one-frame flash).
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ORDINALS, Api, load_config, load_spec, parser, read_json, wpath, write_json  # noqa: E402

NEGATIVE = re.compile(r"\b(no|not|never|without|nothing|none|don't|doesn't|isn't|aren't|avoid|neither|nor|cannot|can't|won't)\b", re.I)
WET = re.compile(r"\b(wet|moist|glisten\w*|dewy|sweat(?!er)\w*|damp|tears?|tearful|crying|cries|cry)\b", re.I)
FACE_COLOUR = re.compile(r"\b(purple with|purple-faced|tomato[- ]red|beet[- ]red|red-faced|turning red|blush\w*|flush(?:ed|ing)? (?:pink|red))\b", re.I)
CUE = re.compile(r"from (\d+(?:\.\d+)?) s to (\d+(?:\.\d+)?) s, <d>\[([^\]]*)\]([\s\S]*?)</d>")

DEFAULT_TEXT = {
    "loc_composition": "The reference establishes the space's layout, atmosphere, lighting, and the characters' approximate positions. The people in the location composition are represented by their corresponding character references.",
    "loc_environment": "This reference defines the environment, lighting, and atmosphere, while the camera uses its own area of the space and places the characters in new positions within it.",
    "loc_no_cast": "This reference defines the environment, lighting, and atmosphere of the scene.",
    "audio_full": "<Audio 1> is the complete synchronized song and vocal track for the target video, reused as its complete final soundtrack and timing reference.",
    "audio_inst": "<Audio 1> is the complete instrumental music track for the target video, reused as its complete final soundtrack and timing reference.",
    "audio_dialogue": "<Audio 1> is the complete dialogue and ambience track for the target video, reused as its complete final soundtrack and timing reference.",
    "summary_composition": "The scene uses the location reference as its composition guide while preserving each character's individual appearance from their character references.",
    "summary_environment": "The scene keeps the location's environment and art style while staging the action in its own area of the space, preserving each character's appearance from their character references.",
    "summary_no_cast": "The scene keeps the location's environment and art style.",
    "retain_loc_composition": "{keep}. The people in the location composition are represented by their corresponding character references.",
    "retain_loc_environment": "{keep} are retained, while the camera uses its own area of the space and new positions for the characters.",
    "retain_loc_no_cast": "{keep} are retained.",
    "summary_text_loc": "The scene is staged in {where}, preserving each character's appearance from their character references.",
    "retain_audio": "<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.",
    "sound_full": "<Audio 1> remains the sole complete soundtrack with its original mix, timing, vocals, music, and dynamics intact.",
    "sound_inst": "<Audio 1> remains the sole complete soundtrack with its original mix, timing, and dynamics intact.",
    "sound_dialogue": "<Audio 1> carries the scene's dialogue and ambience.",
    "music_full": "<Audio 1> is reused as the complete song.",
    "music_inst": "<Audio 1> is reused as the complete music track.",
    "music_dialogue": "Silence; this part has diegetic ambience and dialogue only.",
}
DEFAULT_SING = "sings with passion in sync with the vocals in <Audio 1>"
DEFAULT_SUNG = "sings in sync with the vocals in <Audio 1>"


def timecode(t):
    m, s = divmod(float(t), 60)
    return f"{int(m):02d}:{s:06.3f}"


def strip_quotes(text):
    text = re.sub(r"<d>[\s\S]*?</d>", "<d></d>", text)
    return re.sub(r'"[^"]*"', '""', text)


def loc_is_text(loc):
    """A location written as prompt text only (no reference picture): its spec entry has "card": None."""
    return "card" in loc and not loc["card"]


class Builder:
    def __init__(self, spec):
        self.spec = spec
        self.text = {**DEFAULT_TEXT, **getattr(spec, "TEXT", {})}
        self.sing = getattr(spec, "SING", DEFAULT_SING)
        self.sung = getattr(spec, "SUNG", DEFAULT_SUNG)
        self.lint = getattr(spec, "LINT", {})

    def entity(self, table, key, variant):
        base = dict(getattr(self.spec, table)[key])
        over = (base.pop("variants", None) or {}).get(variant) or {}
        return {**base, **over}

    def entities(self, scene):
        variant = scene.get("variant")
        ents = [("char", k, self.entity("CHARACTERS", k, variant)) for k in scene.get("cast", [])]
        # per-scene description overrides, e.g. a group card that names only the members present in this scene
        for _, k, e in ents:
            if k in (scene.get("descs") or {}):
                e["desc"] = scene["descs"][k]
            if k in (scene.get("retains") or {}):
                e["retain"] = scene["retains"][k]
        ents += [("prop", k, self.entity("PROPS", k, variant)) for k in scene.get("props", [])]
        # extra locations for multi-shot scenes that change place (shot 1 kitchen, shot 2 staircase); the scene's
        # main "loc" stays the last picture. Their desc is used as-is plus the environment clause.
        for k in scene.get("extra_locs", []):
            e = dict(self.spec.LOCATIONS[k])
            e["desc"] = e["desc"] + ". " + self.text["loc_environment"].rstrip(".")
            e["retain"] = e["retain"] + " are retained."
            ents.append(("xloc", k, e))
        return ents

    def build(self, scene):
        ents = self.entities(scene)
        loc = self.spec.LOCATIONS[scene["loc"]]
        text_loc = loc_is_text(loc)
        has_cast = any(kind == "char" for kind, _, _ in ents)
        loc_n = len(ents) + 1
        mode = "no_cast" if not has_cast else ("composition" if scene.get("use") == "composition" else "environment")
        tokens = {key: f"<Subject {i + 1}>" for i, (_, key, _) in enumerate(ents)}
        # A prompt-only location (card None) has no picture: {loc} becomes its short name, and its full
        # description goes into the summary instead of a <Subject>.
        tokens.update(loc=(loc.get("name") or loc["desc"]) if text_loc else f"<Subject {loc_n}>",
                      sing=self.sing, sung=self.sung)

        defs = [f"<Subject {i + 1}> is {e['desc'].format(o=ORDINALS[i])}." for i, (_, _, e) in enumerate(ents)]
        if not text_loc:
            defs.append(f"<Subject {loc_n}> is {loc['desc'].format(o=ORDINALS[loc_n - 1])}. "
                        f"{scene.get('loc_clause') or self.text['loc_' + mode]}")
        defs.append(scene.get("audio_def") or self.text["audio_" + scene["audio"]])

        if text_loc:
            tail = scene.get("summary_tail") or self.text["summary_text_loc"].format(where=loc["desc"])
        else:
            tail = scene.get("summary_tail") or self.text["summary_" + mode]
        trigger = getattr(self.spec, "TRIGGER", "")
        summary = f"[reference generation + audio reuse] {trigger + ' ' if trigger else ''}{scene['summary']} {tail}"

        retention = [f"<Subject {i + 1}>: fully_preserved - {e['retain'][0].upper() + e['retain'][1:]}"
                     for i, (_, _, e) in enumerate(ents)]
        if not text_loc:
            keep = loc["retain"][0].upper() + loc["retain"][1:]
            retention.append(f"<Subject {loc_n}>: fully_preserved - " + self.text["retain_loc_" + mode].format(keep=keep))
        retention.append(self.text["retain_audio"])

        shots = []
        for k, (t, text) in enumerate(scene["shots"], start=1):
            body = text.format(**tokens)
            shots.append(f"[Shot {k}] {body}" if k == 1 else f"[Shot {k}] At {timecode(t)}, {body}")
        detailed = scene.get("style", self.spec.STYLE).format(light=scene["light"]) + "\n\n" + "\n\n".join(shots)

        variant = scene.get("variant")
        sound = scene.get("sound") or self.text["sound_" + scene["audio"]]
        music = scene.get("music") or self.text.get(f"music_{variant}") or self.text["music_" + scene["audio"]]
        return ("subject_definitions:\n" + "\n\n".join(defs)
                + "\n\nsummary:\n" + summary
                + "\n\nretention_analysis:\n" + "\n".join(retention)
                + "\n\ndetailed_description:\n" + detailed
                + "\n\noverall_soundscape:\n" + sound
                + "\n\nnon_diegetic_music:\n" + music)

    def check(self, scene, prompt, scene_window=None, words=None):
        problems, warns = [], []
        if len(prompt) > 7000:
            problems.append(f"too long: {len(prompt)} characters (limit 7000)")
        elif len(prompt) > 6500:
            warns.append(f"{len(prompt)} characters: close to the 7000 limit")
        bare = strip_quotes(prompt)
        for m in NEGATIVE.finditer(bare):
            problems.append(f"negative word '{m.group(0)}': ...{bare[max(0, m.start() - 50):m.end() + 40]}...")
        for m in WET.finditer(bare):
            problems.append(f"wet/tear word '{m.group(0)}'")
        for m in FACE_COLOUR.finditer(bare):
            warns.append(f"face-colour wording '{m.group(0)}': reads as skin colour or a romantic blush (fine for a flirt; for anger write actions)")
        variant = scene.get("variant")
        forbid = self.lint.get("forbid_outside_variants")
        if forbid and variant not in forbid.get("variants", []):
            for m in re.finditer(forbid["pattern"], bare, re.I):
                problems.append(f"'{m.group(0)}' outside {forbid.get('variants')}: ...{bare[max(0, m.start() - 50):m.end() + 40]}...")
        need = (self.lint.get("require_in_variant") or {}).get(variant)
        if need and not re.search(need, bare, re.I):
            problems.append(f"variant '{variant}' needs /{need}/ (for example the microphone on stage)")
        shots_text = prompt.split("detailed_description:")[1]
        n_subjects = len(self.entities(scene)) + (0 if loc_is_text(self.spec.LOCATIONS[scene["loc"]]) else 1)
        for i in range(1, n_subjects + 1):
            if f"<Subject {i}>" not in shots_text:
                problems.append(f"<Subject {i}> is defined but never used in the shots")
        if re.search(r"\{[a-z_]+\}", prompt):
            problems.append(f"unfilled token: {re.findall(r'{[a-z_]+}', prompt)[:3]}")
        times = [t for t, _ in scene["shots"]]
        if times != sorted(times) or times[0] != 0:
            problems.append(f"shot times must start at 0 and increase: {times}")
        if scene_window:
            start, end = scene_window
            if times[-1] >= end - start:
                problems.append(f"last shot starts at {times[-1]} s but the scene is {end - start:.2f} s")
        if words is not None and scene_window and scene["audio"] == "full":
            warns += self.check_cues(scene, scene_window, words)
        first = scene["shots"][0][1]
        lead = (scene.get("cast") or [None])[0]
        anchor = re.search(r"(?:looking past|over the shoulder of|from behind|from the [\w ]+ of|past) \{(\w+)\}", first)
        if anchor and anchor.group(1) not in (lead, "loc"):
            warns.append(f"shot 1 anchors the camera on {{{anchor.group(1)}}}: that opening tends to survive the warm-up trim "
                         "as a one-frame flash; open on the scene's main subject")
        return problems, warns

    def check_cues(self, scene, window, words):
        """Compare each shot's <d> cue with the words actually sung in that shot."""
        warns = []
        start, end = window
        cuts = [t for t, _ in scene["shots"]] + [end - start]
        for k, (t, text) in enumerate(scene["shots"]):
            s0, s1 = start + cuts[k], start + cuts[k + 1]
            sung = [w for w in words if s0 - 0.05 <= w["start"] < s1]
            cue = CUE.search(text.format(**{key: "" for key in re.findall(r"\{(\w+)\}", text)}))
            has_dir = "{sung}" in text or "{sing}" in text or "says in sync" in text
            if sung and not has_dir:
                warns.append(f"shot {k + 1} has vocals '{' '.join(w['word'] for w in sung)[:60]}' but no singing direction")
            if not cue:
                continue
            got = re.sub(r"<[^>]+>", " ", cue.group(4))
            got_n = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in got.split() if re.sub(r"[^a-z0-9]", "", w.lower())]
            want_n = [re.sub(r"[^a-z0-9]", "", w["word"].lower()) for w in sung]
            if got_n != want_n:
                warns.append(f"shot {k + 1} cue says '{' '.join(got_n)}' but '{' '.join(want_n)}' is sung in that shot")
            if sung:
                a, b = float(cue.group(1)), float(cue.group(2))
                real_a, real_b = sung[0]["start"] - start, sung[-1]["end"] - start
                if abs(a - real_a) > 0.4 or abs(b - real_b) > 0.6:
                    warns.append(f"shot {k + 1} cue window {a}-{b} s, sung {real_a:.1f}-{real_b:.1f} s")
            if re.search(r"low-angle full-body|full-body shot from (the floor|below)", text, re.I):
                warns.append(f"shot {k + 1}: low-angle full-body framing while she sings makes her face small")
        return warns


MATCH_OVER_PICTURES = 5   # user rule (corrected 2026-11-09): scenes with more than 5 reference pictures use ref_image_size "match"


def scene_settings(project_h3, duration, scene, ff_over, n_pictures=0):
    settings = dict(project_h3)
    ff = duration > ff_over
    for key in ("pass1_use_feedforward", "pass2_use_feedforward", "two_pass_use_feedforward"):
        settings[key] = ff
    if n_pictures > MATCH_OVER_PICTURES:
        settings["ref_image_size"] = "match"   # 6 pictures at "max" ran pass 1 at ~1.5 % per minute (Dead of the Party)
    hold = scene.get("continue_hold")
    if hold is not None:
        settings.update({"continuity_mode": "latent_continuation_masked", "latent_context_frames": 39,
                         "location_transition_preset": "masked"})
    return settings, ff, hold


def apply(api, builder, scene, prompt, project_h3, duration, ff_over, mic_variants):
    n = scene["n"]
    ents = builder.entities(scene)
    cards = [e["card"] for kind, _, e in ents if kind != "xloc"]
    loc_entry = builder.spec.LOCATIONS[scene["loc"]]
    loc_card = "" if loc_is_text(loc_entry) else loc_entry.get("card", scene["loc"])
    patch = {"subject_ids": cards, "location_id": loc_card,
             "performance_mode": "no_lip_sync" if scene["audio"] == "inst" else "singing",
             "lyric_no_lip_sync": scene["audio"] == "inst",
             "include_microphone": scene.get("variant") in mic_variants,
             "story_beat": scene["summary"]}
    out = []
    st, env = api.mutate("PATCH", f"/projects/{{pid}}/scenes/{n}", patch)
    out.append(("scene", st, env))
    keys = [(f"location:{e.get('card', k)}" if kind == "xloc" else f"subject:{e['card']}") for kind, k, e in ents]
    if loc_card:
        keys.append(f"location:{loc_card}")
    st, env = api.mutate("PUT", f"/projects/{{pid}}/scenes/{n}/minimax-references", {"keys": keys})
    out.append(("references", st, env))
    st, env = api.mutate("PUT", f"/projects/{{pid}}/scenes/{n}/prompts/minimax_h3_prompt", {"prompt": prompt, "origin": "manual"})
    out.append(("prompt", st, env))
    settings, ff, hold = scene_settings(project_h3, duration, scene, ff_over, len(keys))
    match = settings.get("ref_image_size") == "match" and project_h3.get("ref_image_size") != "match"
    if ff or hold is not None or match:
        body = {"use_scene_minimax_h3_settings": True, "minimax_h3_settings": settings}
        if hold is not None:
            body["minimax_h3_continuation_start_seconds"] = hold
        st, env = api.mutate("PATCH", f"/projects/{{pid}}/scenes/{n}", body)
        out.append(("scene settings", st, env))
    for what, st, env in out:
        if not env.get("ok"):
            print(f"   {what} FAILED {st}: {(env.get('error') or {}).get('message', env)}")
    if match:
        print(f"   {len(keys)} pictures -> ref_image_size match")
    return ff, hold


def main():
    p = parser(__doc__)
    p.add_argument("--spec", default="spec.py")
    p.add_argument("--apply", nargs="*", default=None, help="scene numbers, or 'all'")
    p.add_argument("--prefix", default=None, help="prompt file prefix (default scene_, or alt_scene_ for a non-default spec)")
    args = p.parse_args()
    cfg = load_config(args.work)
    spec = load_spec(cfg, args.spec)
    builder = Builder(spec)
    prefix = args.prefix or ("scene_" if args.spec == "spec.py" else "alt_scene_")
    timeline = (read_json(wpath(cfg, "timeline.json"), {}) or {}).get("scenes") or []
    words = (read_json(wpath(cfg, "lyrics_timed.json"), {}) or {}).get("words")
    wanted = None
    if args.apply is not None:
        wanted = {s["n"] for s in spec.SCENES} if args.apply in ([], ["all"]) else {int(a) for a in args.apply}
    api = Api(cfg, args.project) if wanted else None
    project_h3, durations = {}, {}
    if api:
        _, env = api.call("GET", "/projects/{pid}/settings")
        project_h3 = (env.get("data") or {}).get("minimax_h3") or {}
        durations = {int(s["number"]): float(s["end"]) - float(s["start"]) for s in api.scenes()}
    ff_over = float(cfg.get("feedforward_over_seconds", 8.0))
    mic_variants = getattr(spec, "MIC_VARIANTS", ["stage"])
    os.makedirs(wpath(cfg, "prompts"), exist_ok=True)

    plan_path = wpath(cfg, "render_plan.json")
    plan = read_json(plan_path, {}) or {}
    plan_scenes = plan.get("scenes") or {}
    bad = 0
    for scene in spec.SCENES:
        n = scene["n"]
        window = tuple(timeline[n - 1][:2]) if len(timeline) >= n else None
        prompt = builder.build(scene)
        problems, warns = builder.check(scene, prompt, window, words)
        with open(wpath(cfg, f"prompts/{prefix}{n:02d}.txt"), "w", encoding="utf-8") as handle:
            handle.write(prompt)
        print(f"scene {n:2d}: {len(prompt):5d} chars {'PROBLEM' if problems else 'OK'}")
        for msg in problems:
            print("   PROBLEM", msg)
        for msg in warns:
            print("   WARN   ", msg)
        bad += bool(problems)
        cards = [e["card"] for _, _, e in builder.entities(scene)]
        entry = {"audio": "no_vocals" if scene["audio"] == "inst" else "full", "cards": cards, "loc": scene["loc"]}
        if args.spec == "spec.py":
            plan_scenes[str(n)] = entry
        if wanted and n in wanted:
            if problems:
                print("   not applied (fix the problems first)")
                continue
            ff, hold = apply(api, builder, scene, prompt, project_h3, durations.get(n, 0.0), ff_over, mic_variants)
            print(f"   applied to '{api.pid}' (feedforward={ff}{', continuation hold ' + str(hold) + ' s' if hold is not None else ''})")
    write_json(plan_path, {"no_vocals_path": os.path.abspath(wpath(cfg, "audio/full_track_no_vocals.wav")), "scenes": plan_scenes})
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
