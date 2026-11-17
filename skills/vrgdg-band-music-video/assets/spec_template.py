"""Scene spec template for build_prompts.py. Copy it to <work>/spec.py and fill it in.

How build_prompts.py turns one scene into a MiniMax H3 prompt:
  subject_definitions  <Subject 1..k> = the scene's cast in order, then its props, then the location (always last).
                       Each "desc" holds {o}, which becomes the ordinal of that picture ("first", "second", ...),
                       so write "from the {o} character reference" / "from the {o} reference image".
                       The Builder sends the pictures in exactly this order (minimax-references keys).
  summary              "[reference generation + audio reuse] <summary> <tail for composition/environment/no cast>"
  retention_analysis   one fully_preserved line per subject + the location + <Audio 1> fully_copy
  detailed_description STYLE (the acting line, or the scene's style=) + "[Shot 1] ..." + "[Shot N] At MM:SS.mmm, ..." for each cut
  overall_soundscape / non_diegetic_music   from TEXT by audio kind (or the scene's own sound/music)

Tokens inside shot text: {<cast key>} {<prop key>} {loc} -> <Subject N>; {sing} / {sung} -> the singing verbs below.
Read references/prompt-writing.md before writing shots; assets/example_spec.py is a complete real example.
"""

# The acting line: ONE sentence about faces, used by every scene (a scene can set its own with style=...).
# No art-style, lighting or blanket camera sentences: the references carry the look and light, and those sentences
# brought overlay and fake-text glitches (PRETTY LITTLE MALFUNCTION). Camera moves go inside individual shots.
# {light} is still accepted by the builder but unused. Match the tone the user asked for (cute vs. big mugging).
STYLE = ("Every face performs with big, playful, cheeky expressions that change from beat to beat: brows shooting up, "
         "sly sideways glances, wide grins, winks, mock pouts and eyes flaring with glee.")

SING = "sings with passion in sync with the vocals in <Audio 1>"   # {sing}: no timed words (wide stage shots, big motion)
SUNG = "sings in sync with the vocals in <Audio 1>"                # {sung}: followed by "from A s to B s, <d>[...] words</d>"

# Section texts. Defaults exist for all of these in build_prompts.py; override to name the genre and the cast type.
TEXT = {
    "audio_inst": "<Audio 1> is the complete instrumental <genre> music track for the target video, reused as its complete final soundtrack and timing reference.",
    "sound_full": "<Audio 1> remains the sole complete soundtrack with its original mix, timing, vocals, music, and dynamics intact: <one-line description of the song>.",
    "sound_inst": "<Audio 1> remains the sole complete soundtrack with its original mix, timing, and dynamics intact: <one-line description of the instrumental>.",
    "music_full": "<Audio 1> is reused as the complete song: <genre, energy, vocal>.",
    "music_inst": "<Audio 1> is reused as the complete music track: <genre, energy>.",
    "music_stage": "<Audio 1> is reused as the complete song: <genre with the instruments named>.",   # used when variant == "stage"
    "music_dialogue": "Silence; this intro has diegetic ambience and dialogue only.",
}

# Words that may only appear in some variants (instruments / microphone only on stage), and words a variant needs.
LINT = {
    "forbid_outside_variants": {
        "pattern": r"\b(microphones?|mic|guitars?|bass|drums?|drumsticks?|sticks|cymbals?|instruments?|amplifiers?|amps?)\b",
        "variants": ["stage"],
    },
    "require_in_variant": {"stage": r"\b(microphones?|mic)\b"},
}
MIC_VARIANTS = ["stage"]
POSTER_CARDS = ["singer", "drummer"]   # cards for the social-kit poster refs, in attach order

# card    = the Reference Builder subject id (setup_refs.py / refs.json) whose image is sent for this character
# desc    = the look, pasted identically in every scene (identity drifts when it is paraphrased)
# retain  = the retention line
# variants= per-variant overrides; "stage" uses the sheet WITH the instrument and role words like "the drummer"
CHARACTERS = {
    "singer": {
        "card": "singer",
        "desc": "the blonde punk girl from the {o} character reference, the band's lead singer, with <hair>, <outfit>, <shoes>, and <accessories>",
        "retain": "Blonde hair, facial identity, complete outfit, and accessories.",
        "variants": {"stage": {"desc": "the female lead singer from the {o} character reference, with <hair>, <outfit>, <shoes>, and <accessories>"}},
    },
    "drummer": {
        "card": "drummer_off",            # the user's instrument-free sheet, used everywhere off stage
        "desc": "the red-haired punk from the {o} character reference, with <hair>, <outfit>, and <shoes>",
        "retain": "Red spiky hair, facial identity, complete outfit, and accessories.",
        "variants": {"stage": {"card": "drummer", "desc": "the drummer from the {o} character reference, with <hair>, <outfit>, and <shoes>"}},
    },
}

# Props that need their own picture (they appear on a character sheet, or must look exactly right).
PROPS = {
    "jacket": {"card": "studded_jacket",
               "desc": "the black leather biker jacket from the {o} reference image, with <details>; it is a loose garment carried and gripped in the hands throughout the scene",
               "retain": "Black leather, studs, zips, and patches of the jacket."},
}

LOCATIONS = {
    "kitchen": {"card": "kitchen",
                "desc": "the messy party kitchen from the {o} reference image, featuring <5-7 concrete features that are in the plate>",
                "retain": "cream cupboards, cluttered counter, fridge, red cups, bottles, and party lighting"},
    "stage": {"card": "stage", "desc": "the punk club stage from the {o} reference image, featuring <features>",
              "retain": "stage lighting, backdrop, amplifiers, drum kit, and crowd"},
}

SCENES = [
    # Singing scene, several shots, one timed cue per shot (words from shot_lyrics.py).
    dict(n=3, cast=["singer", "drummer"], loc="kitchen", use="environment", audio="full",
         light="pink and purple party lights and a harsh overhead kitchen light",
         summary="One sentence: who does what, where.",
         shots=[
             (0.0, "A waist-up medium shot in the crowded kitchen of {loc}: {singer} <action>, and she {sung} from 0.6 s to 4.8 s, <d>[English, scornful, defiant, singing] Your precious <i>consensus</i> can get in the <i>bin</i>!</d>, her eyes narrowed to scornful slits, her nose wrinkled in a sneer and her chin jutting out."),
             (4.9, "the camera cuts to a medium shot along the counter: {drummer} <action>, his eyebrows bouncing and a maniac's grin on his face, his lips pressed together."),
         ]),
    # Stage performance: the variant switches to the instrument sheets, role words and stage music text.
    dict(n=21, cast=["singer", "drummer"], loc="stage", use="composition", audio="full", variant="stage",
         light="blazing red and white stage lights",
         summary="The band's explosive stage performance, staged as in the stage reference.",
         shots=[
             (0.0, "A medium-wide shot matching the stage layout of {loc}: {singer} grips a corded microphone and {sung} from 0.0 s to 2.0 s, <d>[English, furious, triumphant, singing] <i>Fuck</i> the static!</d>, her brows slammed down; {drummer} pounds his drum kit, sticks blurring, his lips pressed together."),
             (2.5, "the camera cuts to a close-up of {singer} singing into the microphone as she {sung} from 2.5 s to 4.6 s, <d>[English, fierce, singing] next words</d>, her eyes flaring wide, sparks raining down behind her."),
         ]),
    # Scene without the singer: render against the vocal-free track (audio "inst"); acting only, no singing words.
    dict(n=10, cast=["drummer"], loc="kitchen", use="environment", audio="inst",
         light="pink party lights", summary="The drummer wins a drinking contest and crashes through a table.",
         shots=[(0.0, "A medium shot at a cheap wooden table in {loc}: {drummer} <action with a big facial reaction>.")]),
    # No main cast at all (crowd chaos): only the location; describe at most 3 extras precisely.
    dict(n=17, cast=[], loc="kitchen", use="environment", audio="inst", light="flickering party lights",
         summary="The party gets out of hand.", shots=[(0.0, "A handheld medium-wide shot of {loc}: <extras and action>.")]),
    # Spoken intro/outro scene: audio "dialogue", line windows measured from the ElevenLabs takes.
    dict(n=1, cast=["singer", "drummer"], loc="kitchen", use="environment", audio="dialogue", light="warm window light",
         summary="The band arrives.",
         shots=[(0.0, "A medium two-shot of {singer} and {drummer} in {loc}, {singer} on the left of the frame and {drummer} on the right: {drummer} turns toward {singer}, toward the left edge of the frame, and says in sync with <Audio 1> from 1.1 s to 2.6 s, <d>[English, skeptical, amused] Whose party is this, <i>anyway</i>?</d> {singer}'s lips stay pressed together in a smirk.")],
         sound="<Audio 1> carries the scene's dialogue and ambience: <what is audible>.",
         music="Silence; this intro has diegetic ambience and dialogue only."),
    # Held prop with its own card (the character uses a sheet without it).
    dict(n=13, cast=["drummer"], props=["jacket"], loc="kitchen", use="environment", audio="inst", light="hazy lamplight",
         summary="The drummer fights over his jacket.",
         summary_tail="The scene preserves the character's appearance from his character reference and the jacket's design from its reference.",
         shots=[(0.0, "A medium shot in {loc}: a burly punk with a tall green mohawk grabs one sleeve of {jacket} while {drummer} clutches the rest of it to his chest, giggling helplessly.")]),
    # Continuation alternate (only in a spec_alt.py used with a duplicate project): add continue_hold=<seconds>,
    # and write shot 1 as "For the first N seconds, carrying on from the previous moment ... At about N seconds ...".
]
