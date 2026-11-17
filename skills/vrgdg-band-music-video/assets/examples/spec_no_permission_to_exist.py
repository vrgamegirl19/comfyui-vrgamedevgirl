"""NO PERMISSION TO EXIST: British post-punk music video (MiniMax H3, 27 scenes, 2:44), written 2026-11-12/13.

Brief (rules.txt): free rein except the song. Characters and locations made with Krea 2 (two pass) on Reference
Builder cards. More energy and more fun than NO SAINT OF YOURS. Same settings and workflow as that video.

Idea: the band has taken over the waiting hall of a grey 1970s permit office and plays on its open floor. Between
band scenes the singer walks through the building breaking its small rules, with a prim young clerk as her foil:
his office (verse 1 and 2), the archive corridor, the entrance lobby, the lift (spoken bridge), the rooftop at dusk
(final chorus). By the end the clerk is in the hall with his tie loose, dancing to the band.

Band map in the hall (no raised stage): singer front centre, guitarist left of the frame, bassist right of the frame,
drummer back centre in front of the counter windows.
Stems: bass + guitar alone 0-8 s, drums in at 8 s; verse 1 sparse (lines with 2 s gaps); backing vocals only at
61.1-62.4 and 64.4-65.4 s; bridge 100-113 s is spoken with the band almost silent, drums back at 112 s;
"No permission" 142.5-144.1 s, "to exist" 148.9-150.5 s; the band plays to about 163 s.
Variants: "stage" = the singer with her microphone in the hall; "band" = band members only.

Lessons applied from NO SAINT OF YOURS: microphone wording without the grille; every wide shot shows the whole band
and moves in at once; the one drummer is named as the only drummer; the last hit keeps everyone playing; nothing
streaky (rain) behind a face; scenes 2-8 s, cuts in vocal gaps, continuity off.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_TL = json.load(open(os.path.join(_HERE, "timeline.json"), encoding="utf-8"))["scenes"]
_WORDS = json.load(open(os.path.join(_HERE, "lyrics_timed.json"), encoding="utf-8"))["words"]


def _cue(n, a, b, emo, text, verb, mode):
    start = _TL[n - 1][0]
    sung = [w for w in _WORDS if start + a - 0.05 <= w["start"] < start + b]
    if not sung:
        raise ValueError(f"scene {n}: no words between {a} and {b} s")
    t0, t1 = max(0.0, sung[0]["start"] - start), min(_TL[n - 1][1] - start, sung[-1]["end"] - start)
    return f"{verb} from {t0:.2f} s to {t1:.2f} s, <d>[English, {emo}, {mode}] {text}</d>"


def cue(n, a, b, emo, text):
    """A timed lyric cue for the words that start inside shot window a..b (scene-relative seconds) of scene n."""
    return _cue(n, a, b, emo, text, "{sung}", "singing")


def say(n, a, b, emo, text):
    """The same for the spoken bridge."""
    return _cue(n, a, b, emo, text, "speaks in sync with the voice in <Audio 1>", "speaking")


STYLE = ("Her face performs with cool, dry punk attitude that shifts from beat to beat: a level unimpressed stare, one "
         "brow lifting, a slow crooked smirk, the chin tipping up.")
STYLE_RUN = ("Her face performs with bright, cheeky punk energy: sharp steady eyes, a crooked grin breaking through "
             "between lines, the chin up, her whole body moving with the beat.")
STYLE_BIG = ("Every face performs with fierce, joyful punk energy: sharp steady eyes, a curled lip, a crooked grin "
             "breaking through between lines; the guitarist and the bassist bounce and stomp at their own spots, and "
             "the drummer hits hard, seated at his kit.")
STYLE_BAND = ("Every face is locked into the music with cool punk attitude: narrowed eyes, a crooked half-smile, the "
              "head snapping on the beat, the lips pressed together.")

SING = "sings with raw energy in sync with the vocals in <Audio 1>"
SUNG = "sings in sync with the vocals in <Audio 1>"

_SONG = "a tense British post-punk song at a fast 162 BPM with a smoky, raspy low female lead vocal"
TEXT = {
    "audio_inst": "<Audio 1> is the complete instrumental post-punk music track for the target video, reused as its complete final soundtrack and timing reference.",
    "sound_full": f"<Audio 1> remains the sole complete soundtrack with its original mix, timing, vocals, music, and dynamics intact: {_SONG}.",
    "sound_inst": "<Audio 1> remains the sole complete soundtrack with its original mix, timing, and dynamics intact: a tense British post-punk instrumental at a fast 162 BPM.",
    "music_full": "<Audio 1> is reused as the complete song: tense, driving British post-punk at 162 BPM with a smoky, raspy low female lead vocal.",
    "music_inst": "<Audio 1> is reused as the complete music track: tense, driving British post-punk at 162 BPM.",
    "music_stage": "<Audio 1> is reused as the complete song: British post-punk played live by the band, with a clipped distorted electric guitar, a descending electric bass line, dry punchy drums and a smoky, raspy low female lead vocal.",
    "music_band": "<Audio 1> is reused as the complete music track: British post-punk played live by the band, with a clipped distorted electric guitar, a descending electric bass line and dry punchy drums.",
}

LINT = {
    "forbid_outside_variants": {
        "pattern": r"\b(microphones?|mic|guitars?|bass|drums?|drumsticks?|sticks|cymbals?|instruments?|amplifiers?|amps?)\b",
        "variants": ["stage", "band"],
    },
    "require_in_variant": {"stage": r"\b(microphones?|mic)\b"},
}
MIC_VARIANTS = ["stage"]
POSTER_CARDS = ["singer", "guitarist", "bassist", "drummer"]

MIC = "a standard all-black handheld vocal microphone"
MIC2 = "the same all-black handheld microphone"

_SINGER = ("the band's lead singer from the {o} character reference, a young woman with a choppy platinum-blonde shag "
           "haircut with dark roots, winged eyeliner and deep red lipstick, wearing a black spiked choker, a studded "
           "black leather biker jacket over a ripped white t-shirt, a short red tartan skirt, ripped black fishnet "
           "tights and black combat boots")
_SINGER_KEEP = ("Platinum-blonde shag hair with dark roots, facial identity, red lipstick, spiked choker, studded "
                "black leather jacket, ripped white t-shirt, red tartan skirt, fishnet tights and combat boots")

CHARACTERS = {
    "singer": {
        "card": "singer",
        "desc": _SINGER + ", her hands empty and free",
        "retain": _SINGER_KEEP + ".",
        "variants": {"stage": {"desc": _SINGER + ", holding " + MIC + " in her right hand",
                               "retain": _SINGER_KEEP + ", and the all-black handheld microphone."}},
    },
    "guitarist": {
        "card": "guitarist",
        "desc": ("the band's guitarist from the {o} character reference, a tall wiry young man with a tall "
                 "scarlet-red mohawk, wearing a studded sleeveless black denim vest over a black-and-white striped "
                 "t-shirt, ripped black jeans and black boots, playing his white electric guitar"),
        "retain": "Scarlet-red mohawk, facial identity, studded black denim vest, striped t-shirt, ripped black jeans, and the white electric guitar.",
    },
    "bassist": {
        "card": "bassist",
        "desc": ("the band's bassist from the {o} character reference, a young woman with warm brown skin and a "
                 "black bob with a hot-pink fringe, wearing a cropped leopard-print faux-fur jacket over a black slip "
                 "dress, black fishnet tights and black platform shoes, playing her cherry-red electric bass guitar"),
        "retain": "Black bob with a hot-pink fringe, facial identity, leopard-print jacket, black slip dress, and the cherry-red electric bass guitar.",
    },
    "drummer": {
        "card": "drummer",
        "desc": ("the band's drummer from the {o} character reference, a stocky young man with a bleached-blond buzz "
                 "cut, wearing a white sleeveless vest top with thin red braces, pale bleach-splattered jeans and "
                 "oxblood-red boots, seated behind his black drum kit, one wooden drumstick in each hand"),
        "retain": "Bleached-blond buzz cut, facial identity, white vest top with red braces, pale jeans, and the black drum kit.",
    },
    "clerk": {
        "card": "clerk",
        "desc": ("the office clerk from the {o} character reference, a slim young man with neatly side-parted slick "
                 "brown hair and round thin wire-rimmed glasses, wearing a slightly too large pale grey three-piece "
                 "suit, a white shirt, a mustard-yellow tie and polished brown shoes"),
        "retain": "Side-parted brown hair, round wire-rimmed glasses, facial identity, pale grey three-piece suit and mustard-yellow tie.",
    },
}
PROPS = {}

LOCATIONS = {
    "hall": {"card": "hall",
             "desc": ("the vast waiting hall of a 1970s permit office at night from the {o} reference image, featuring "
                      "concrete pillars, beige wood-panelled walls, a long row of service counter windows along the back "
                      "wall, orange plastic chairs pushed to both sides, a wide open speckled linoleum floor in the "
                      "middle with loose paper forms on it, and long fluorescent tube lights overhead"),
             "retain": "concrete pillars, wood-panelled walls, the row of counter windows, orange plastic chairs at the sides, the open linoleum floor, and the fluorescent tube lights"},
    "office": {"card": "office",
               "desc": ("the cramped clerk's office from the {o} reference image, featuring a heavy wooden desk "
                        "crowded with a rack of rubber stamps, ink pads, a green glass banker's lamp, a beige "
                        "telephone and tall stacks of paper forms, a worn brown leather swivel chair behind the desk, "
                        "grey metal filing cabinets and shelves of ring binders along beige walls, venetian blinds and "
                        "brown carpet tiles"),
               "retain": "wooden desk with the rubber stamps, green banker's lamp, telephone and stacks of forms, brown leather swivel chair, grey filing cabinets, shelves of ring binders, and venetian blinds"},
    "archive": {"card": "archive",
                "desc": ("the endless narrow archive corridor from the {o} reference image, featuring both walls lined "
                         "from floor to ceiling with grey metal filing cabinets and open shelves of bulging manila "
                         "folders, a row of bare fluorescent tube lights receding into the far distance, a worn green "
                         "linoleum floor and pipes overhead"),
                "retain": "grey metal filing cabinets and shelves of manila folders on both sides, the receding row of fluorescent lights, green linoleum floor, and overhead pipes"},
    "lobby": {"card": "lobby",
              "desc": ("the entrance lobby of the 1970s office block in the evening from the {o} reference image, "
                       "featuring a row of waist-high chrome turnstiles across the middle of the room, a queue lane of "
                       "thick red rope barriers on brass posts, a polished speckled terrazzo floor, a long wood-veneer "
                       "reception desk, concrete pillars, potted palms and tall glass entrance doors with orange "
                       "street light beyond"),
              "retain": "chrome turnstiles, red rope barriers on brass posts, terrazzo floor, wood-veneer reception desk, concrete pillars, potted palms, and the glass entrance doors"},
    "lift": {"card": "lift",
             "desc": ("the inside of the old office lift from the {o} reference image, featuring brown wood-effect "
                      "laminate walls, a brass handrail at waist height, a brushed steel panel with two columns of "
                      "round amber buttons on the right-hand wall, a worn burgundy carpet and a square frosted "
                      "ceiling light"),
             "retain": "wood-effect laminate walls, brass handrail, steel panel of round amber buttons, burgundy carpet, and the frosted ceiling light"},
    "roof": {"card": "roof",
             "desc": ("the flat rooftop of the concrete office block at dusk from the {o} reference image, featuring a "
                      "wide gravel and tar surface, squat air vents, a rusty water tank and aerials, a low concrete "
                      "parapet, and beyond it a wide skyline of London rooftops and tower blocks under a glowing orange "
                      "and magenta sunset sky"),
             "retain": "gravel rooftop, air vents, rusty water tank, aerials, low concrete parapet, the London skyline, and the orange and magenta sunset sky"},
}

# ---------------------------------------------------------------- constants
STAGE_MAP = ("The band is set up on the open linoleum floor in the middle of the hall, in front of the counter windows. "
             "Fixed positions: the drummer seated behind the black drum kit at the back centre, the lead singer at the "
             "front centre, the guitarist on the left of the frame and the bassist on the right of the frame. The hall "
             "holds exactly one drum kit, one drummer, one guitarist, one bassist and one singer, and every shot faces "
             "the band from the open floor in front of them or from the side.")
STAGE_MAP_CLERK = STAGE_MAP + (" The clerk stands apart from the band among the orange plastic chairs at the right "
                               "side of the hall, the only other person there.")
ALL4 = "all four band members in view at their own spots"
ROAM = ("This reference defines the environment, lighting, and atmosphere, while the camera uses its own area of the "
        "space and places her in new positions within it. She is the only person in the place.")
ROAM2 = ("This reference defines the environment, lighting, and atmosphere, while the camera uses its own area of the "
         "space and places the two of them in new positions within it. They are the only two people in the place.")
TAIL_STAGE = ("The scene preserves each band member's appearance and instrument from their character references. "
              "Everything moves at normal real-time speed.")
TAIL_SOLO = ("The scene keeps the location's environment and art style and preserves her appearance from her "
             "character reference. She is alone, and everything moves at normal real-time speed.")
TAIL_DUO = ("The scene keeps the location's environment and art style and preserves both characters' appearance from "
            "their character references. Everything moves at normal real-time speed.")
CONT_OPEN = ("All shots are one continuous performance: every person keeps their exact position, posture and ongoing "
             "movement across each camera cut, and only the camera angle and framing change. ")
C = "continuing directly from the previous shot's final moment, "
BAND = ["singer", "guitarist", "bassist", "drummer"]
G_PLAYS = "{guitarist} at his own spot on the left of the frame slashes clipped chords on his white electric guitar in time with the guitar in <Audio 1>"
B_PLAYS = "{bassist} at her own spot on the right of the frame drives her cherry-red electric bass guitar in time with the bass"
D_PLAYS = "{drummer}, the band's only drummer, seated behind his black drum kit at the back centre, pounds a dry, fast beat with one drumstick in each hand"
BAND_PLAYS = G_PLAYS + ", " + B_PLAYS + ", and " + D_PLAYS
LIPS3 = "{guitarist}, {bassist} and {drummer} keep their lips pressed together"
CLERK_QUIET = "his lips pressed together"

L_HALL = "cold fluorescent tube light with warm amber patches"
L_IN = "flat fluorescent office light"
L_ROOF = "glowing orange and magenta dusk light"


def stage(n, summary, shots, cast=None, style=STYLE_BIG, audio="full", loc_clause=STAGE_MAP, variant="stage"):
    return dict(n=n, cast=cast or BAND, loc="hall", use="environment", audio=audio, variant=variant, light=L_HALL,
                style=style, summary=summary, summary_tail=TAIL_STAGE, loc_clause=loc_clause, shots=shots)


def solo(n, loc, summary, shots, style=STYLE, audio="full", light=L_IN):
    return dict(n=n, cast=["singer"], loc=loc, use="environment", audio=audio, light=light, style=style,
                summary=summary, summary_tail=TAIL_SOLO, loc_clause=ROAM, shots=shots)


def duo(n, loc, summary, shots, style=STYLE, audio="full", light=L_IN):
    return dict(n=n, cast=["singer", "clerk"], loc=loc, use="environment", audio=audio, light=light,
                style=style + " The clerk stays stiff and flustered, " + CLERK_QUIET + ".",
                summary=summary, summary_tail=TAIL_DUO, loc_clause=ROAM2, shots=shots)


SCENES = [
    # ------------------------------------------------------------ intro (stems: bass + guitar alone, drums at 8 s)
    stage(1, "In the grey waiting hall the bassist starts a descending line and the guitarist chops clipped chords over it.",
          cast=["bassist", "guitarist"], variant="band", style=STYLE_BAND, audio="inst",
          loc_clause="Two band members appear in this scene, each completely alone in a shot of their own on the open linoleum floor of the hall in front of the counter windows: first the bassist, then the guitarist. Each shot shows exactly one person.",
          shots=[
              (0.0, "The first frame is already a close shot of a cherry-red electric bass guitar in {loc}: the camera sits close on the hands of {bassist} as her fingers walk a slow descending line down the thick strings in time with the bass in <Audio 1>, then the camera rises along the neck to her face, her eyes down on the strings and a small smirk on her closed lips, the hot-pink fringe swinging as she nods."),
              (2.3, "Hard cut, the same moment in the same hall: a low-angle medium shot with exactly one person in the frame, {guitarist} alone, filling the frame from his knees to his red mohawk with the empty counter windows behind him, standing with his boots apart, chopping short, clipped chords on his white electric guitar in time with the guitar in <Audio 1>, his head jerking on every stroke and his lips pressed together, the fluorescent tubes of {loc} bright above his red mohawk as the camera pushes in toward him."),
          ]),
    stage(2, "The whole band waits in the hall as the intro tightens, then the drummer crashes in.",
          style=STYLE_BAND, audio="inst",
          shots=[
              (0.0, CONT_OPEN + "A wide low shot from the open floor in {loc} with " + ALL4 + ", the camera gliding fast along the linoleum toward the singer past scattered paper forms: " + G_PLAYS + " and " + B_PLAYS + "; {drummer}, the band's only drummer, sits behind his black drum kit at the back centre with both drumsticks raised and still; {singer} stands at the front centre with her head bowed, " + MIC + " held against her chest in her right hand, her lips closed. " + LIPS3 + "."),
              (2.3, C + "the band still at their spots, the camera cuts to a medium shot from the side of the drum kit: {drummer} holds both sticks high for a beat, then at about three and a half seconds brings them down hard on the snare and a cymbal and launches into a dry, driving beat in time with the drums in <Audio 1>, his shoulders pumping and his lips pressed together, the camera pushing in fast."),
          ]),

    # ------------------------------------------------------------ verse 1: the clerk's office, then the archive
    duo(3, "office", "In the clerk's cramped office she stands in front of his desk and sings, unimpressed, while he looks her up and down and slides a tall stack of forms toward her.",
        shots=[
            (0.0, CONT_OPEN + "The first frame is already a medium close-up at her eye level: {singer} stands in front of the heavy wooden desk in {loc} with her arms folded, looks straight into the camera and " + cue(3, 0.0, 6.5, "low, dry, restrained", "They measured my worth in a language of <i>doubt</i>") + ", one brow lifting on the last word; behind her across the desk, soft in the background, {clerk} sits in the brown leather swivel chair peering at her over his round glasses, " + CLERK_QUIET + "."),
            (5.0, C + "{singer} still standing in front of the desk with her arms folded, the camera cuts to a medium two-shot from the side of the desk: {clerk} pushes a tall stack of paper forms across the desk toward her with both hands and taps the top sheet twice with his pen, his eyes on her and " + CLERK_QUIET + "; she looks down at the stack, then turns her eyes to the camera with a slow crooked smirk, her lips closed."),
        ]),
    duo(4, "office", "She picks up the clerk's big rubber stamp, thumps it down on his forms and then stamps the back of his hand, and his eyebrows shoot up.",
        shots=[
            (0.0, CONT_OPEN + "The first frame is already a medium shot from the side of the heavy wooden desk in {loc}: {singer} leans over the desk with her left hand planted on the stack of paper forms, lifts one big rubber stamp with a round wooden handle from the rack in her right hand and keeps hold of that same stamp, turns her face to the camera and " + cue(4, 0.0, 6.6, "low, wry, amused", "Then acted surprised when I <i>figured</i> it out") + ", and on the last word she thumps the stamp down hard on the top form; across the desk {clerk} sits stiff in his chair watching the stamp, " + CLERK_QUIET + "."),
            (4.95, C + "{singer} still leaning over the desk with the wooden-handled rubber stamp in her right hand, the camera cuts to a close shot of the desktop: the hand of {clerk} rests flat on the desk, she presses the rubber stamp down onto the back of his hand for one beat and then lifts the stamp up and away out of the frame, and a round red ink mark is left on the back of his hand; the camera tilts up to his face as he lifts the hand and stares at the red mark, his eyebrows shooting up above his round glasses, " + CLERK_QUIET + "."),
        ]),
    solo(5, "archive", "Alone she strides down the endless archive corridor, trailing her fingers along the filing cabinets and yanking a drawer open as she passes.",
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a waist-up medium shot tracking backward in front of her: {singer} strides toward the camera down the narrow corridor of {loc}, both arms out wide so her fingertips trail along the grey filing cabinets on either side, the row of fluorescent tubes passing overhead, and she " + cue(5, 0.0, 6.4, "low, hushed, dangerous", "The rule is a <i>whisper</i>, the threat is a <i>grin</i>") + ", a slow sharp smile spreading on the last word."),
             (4.9, C + "{singer} still striding down the corridor, the camera cuts to a medium shot from the side that tracks with her: she hooks two fingers through the handle of a long filing drawer and yanks it all the way out as she walks past, the drawer sliding open full of manila folders behind her, and she keeps walking with her chin up and a crooked closed-lip smirk."),
         ]),
    solo(6, "archive", "In the archive corridor she pulls a fat folder from an open drawer, flips through it and tosses the pages over her shoulder.",
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a medium shot at her eye level: {singer} stands at an open filing drawer in the corridor of {loc}, pulls out a fat manila folder with both hands and flips it open, glances at the pages, then lifts her eyes to the camera and " + cue(6, 0.0, 6.6, "dry, mocking, cool", "The system says <i>thank you</i>, the damage gets <i>in</i>") + ", her brows lifting in mock politeness on thank you."),
             (5.1, C + "{singer} still at the open drawer with the folder in her hands, the camera cuts to a low-angle medium shot in front of her: she tosses the whole sheaf of pages up over her shoulder and walks toward the camera as the loose white pages flutter down through the corridor behind her, a wide crooked grin on her closed lips."),
         ]),

    # ------------------------------------------------------------ pre-chorus 1: the lobby
    solo(7, "lobby", "In the entrance lobby she marches down the roped queue lane, unhooks the red rope at the end and walks straight through.",
         style=STYLE_RUN,
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a waist-up medium shot tracking backward in front of her: {singer} marches toward the camera along the queue lane between the thick red rope barriers in {loc}, the chrome turnstiles and the glass doors behind her, jabbing her finger down at the floor on the beat as she " + cue(7, 0.0, 3.5, "firm, rising, defiant", "I do not <i>kneel</i> to a borrowed command") + ", her chin up."),
             (3.5, C + "{singer} now at the end of the queue lane, the camera cuts to a medium shot from the side: she unhooks the brass clip of the red rope with a flick of her right hand, lets the rope drop to the terrazzo floor and steps over it toward the camera as she " + cue(7, 3.5, 6.38, "firm, louder, proud", "I do not <i>shrink</i> for a stranger's demand") + ", her shoulders squared and her eyes on the lens."),
         ]),
    duo(8, "lobby", "She pushes through the chrome turnstile with her hip and strides on with her arms wide, belting, while the clerk hurries after her waving a sheaf of forms.",
        style=STYLE_RUN,
        shots=[
            (0.0, CONT_OPEN + "The first frame is already a medium close-up at her eye level: {singer} bumps the bar of a chrome turnstile in {loc} with her hip and pushes through it, the bar spinning behind her, shrugs with one open palm turned up toward the camera and " + cue(8, 0.0, 3.2, "bold, sarcastic, rising", "If existence is costly, then send me the <i>bill</i>") + ", one brow arched high."),
            (3.2, C + "{singer} now past the turnstiles, the camera cuts to a low-angle medium shot tracking backward in front of her: she strides toward the camera with both arms thrown wide and " + cue(8, 3.2, 6.82, "belting, loud, triumphant", "I am already here and I am <i>louder</i> still") + ", her head tipping back on the last word; a few steps behind her {clerk} hurries after her through the turnstile, waving a sheaf of paper forms above his head, " + CLERK_QUIET + "."),
        ]),

    # ------------------------------------------------------------ chorus 1: the band in the hall
    stage(9, "The whole band launches into the chorus on the open floor of the waiting hall and the singer belts it at the front.",
          shots=[
              (0.0, CONT_OPEN + "A wide low shot from the open floor in {loc} with " + ALL4 + ", the camera rushing forward in a fast arc from the left toward the singer: " + BAND_PLAYS + "; {singer} at the front centre holds " + MIC + " at her lips in her right hand, punches her left fist into the air and " + cue(9, 0.0, 3.0, "belting, defiant, raw", "I need no <i>permission</i> to exist") + ". " + LIPS3 + "."),
              (3.0, C + "{singer} still at the front centre with the band playing at their spots behind her, the camera cuts to a Dutch-tilted medium close-up from slightly below her eye level, pushing in: she grips " + MIC2 + " at her lips in her right hand, her left hand flicking away to the side, and " + cue(9, 3.0, 6.72, "scornful, biting, loud", "No holy approval, no ritual <i>insistence</i>") + ", a curled lip between the lines; behind her {drummer} pounds on, seated at his kit."),
          ]),
    stage(10, "The singer sneers the next lines into the camera at the front of the band while the guitarist and the bassist stomp at their spots.",
          shots=[
              (0.0, CONT_OPEN + "The first frame is already a waist-up medium shot from the open floor in {loc}, the camera tracking sideways along the front of the band from the right: {singer} at the front centre leans toward the lens with " + MIC + " at her lips in her right hand, waves her left hand away in a little mock blessing and " + cue(10, 0.0, 3.1, "mocking, sweet, sarcastic", "You can keep your blessing, your <i>patronising</i> tone") + "; behind her " + BAND_PLAYS + ". " + LIPS3 + "."),
              (3.1, C + "{singer} still at the front centre, the camera cuts to a low-angle medium shot that arcs around her from her right side to her front: she jabs her left thumb at her own chest twice, " + MIC2 + " at her lips in her right hand, and " + cue(10, 3.1, 6.23, "fierce, proud, loud", "I am not a <i>product</i>, I am not your <i>own</i>") + ", while {guitarist} behind her on the left stomps and slashes his white electric guitar and {bassist} on the right bounces with her bass, both with their lips pressed together."),
          ]),
    stage(11, "The guitarist and the bassist lean in and shout the refrain with the singer, then the drummer rips a fill.",
          shots=[
              (0.0, CONT_OPEN + "The first frame is already a medium shot from the open floor in {loc}, the camera pushing in fast on three people: {singer} in the middle with " + MIC + " at her lips in her right hand " + cue(11, 0.0, 5.0, "shouted, chanting, raw", "No <i>permission</i>, <breath> no <i>permission</i>") + ", stamping her boot on each word; {guitarist} on the left and {bassist} on the right lean in toward her from their spots and shout the same two words together with her each time, their mouths opening wide on the words and closing between them, all three grinning; behind them {drummer}, the band's only drummer, pounds his black drum kit with his lips pressed together."),
              (5.0, C + "the band still playing at their spots, the camera cuts to a medium shot from the side of the black drum kit: {drummer} rips a fast fill around the toms with both sticks and smashes a cymbal in time with the drums in <Audio 1>, his teeth set behind closed lips, the camera pushing in toward him."),
          ]),
    stage(12, "The singer plants her boots and belts the last line of the chorus as the camera sweeps around the band, then stares down the lens.",
          shots=[
              (0.0, CONT_OPEN + "A low shot from the open floor in {loc} with " + ALL4 + ", the camera orbiting from the left side around to the front while pushing in steadily toward the singer: " + BAND_PLAYS + "; {singer} at the front centre plants her boots wide, throws her left arm out to the side, " + MIC + " at her lips in her right hand, and " + cue(12, 0.0, 5.5, "belting, soaring, defiant", "I exist in open <i>opposition</i>") + ", her mouth open wide on the long last word. " + LIPS3 + "."),
              (5.5, C + "{singer} still at the front centre, the camera cuts to a close-up at her eye level: she lowers " + MIC2 + " to her chest in her right hand, closes her lips, breathes hard through her nose and stares straight down the lens with a slow crooked closed-lip grin, the band a soft blur playing on behind her."),
          ]),

    # ------------------------------------------------------------ verse 2: back in the clerk's office
    duo(13, "office", "She has taken the clerk's swivel chair with her boots up on his desk and sings at the camera while he stands beside the desk clutching his forms.",
        shots=[
            (0.0, CONT_OPEN + "The first frame is already a medium shot across the heavy wooden desk in {loc}: {singer} lounges in the brown leather swivel chair behind the desk with her black combat boots up on the desktop, crossed at the ankles, her hands behind her head, and she looks into the camera and " + cue(13, 0.0, 3.78, "dry, knowing, quick", "I have seen your control call itself <i>concern</i>") + ", one brow lifting; beside the desk {clerk} stands stiffly clutching a stack of paper forms to his chest, staring at her boots, " + CLERK_QUIET + "."),
            (3.78, C + "{singer} still in the swivel chair with her boots on the desk, the camera cuts to a close-up at her eye level: she swings her boots down, leans forward over the desk on her elbows with her chin on her fist and " + cue(13, 3.78, 6.9, "deadpan, cutting, quick", "I have watched your compassion conveniently <i>burn</i>") + ", her stare flat and steady, the green lamp glowing beside her."),
        ]),
    duo(14, "office", "She rips a form in half in front of the clerk, straightens his tie, pats his cheek and walks out.",
        shots=[
            (0.0, CONT_OPEN + "The first frame is already a waist-up medium shot: {singer} stands in front of the heavy wooden desk in {loc} holding one sheet of paper up in both hands at chest height, looks into the camera and " + cue(14, 0.0, 3.13, "flat, disgusted, quick", "The bargain was <i>rotten</i> before it was signed") + ", and as she sings she rips the sheet slowly down the middle so that she holds one ripped half in each hand."),
            (3.13, C + "{singer} still in front of the desk with a ripped half sheet in each hand, the camera cuts to a medium two-shot from the side: she lets the two halves drop to the floor, steps up to {clerk}, straightens his mustard-yellow tie with both hands and pats his cheek twice as she " + cue(14, 3.13, 6.26, "sweet, final, mocking", "I am done pretending obedience is <i>kind</i>") + ", with a bright false smile, then turns and walks away past the camera; he stands frozen with his eyes wide behind his round glasses, " + CLERK_QUIET + "."),
        ]),

    # ------------------------------------------------------------ pre-chorus 2: the archive, then the band
    solo(15, "archive", "She runs down the archive corridor toward the camera, then kicks a filing drawer shut and points down the lens.",
         style=STYLE_RUN,
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a waist-up medium shot tracking backward fast in front of her: {singer} runs toward the camera down the narrow corridor of {loc}, her jacket flapping and loose sheets of paper swirling up from the green floor in her wake, and she " + cue(15, 0.0, 2.75, "driving, fierce, loud", "I do not <i>kneel</i> to a borrowed command") + ", her eyes locked on the lens."),
             (2.75, C + "{singer} now slowing beside an open filing drawer, the camera cuts to a low-angle medium shot: she kicks the long drawer shut with the sole of her combat boot so it slams into the cabinet, then points straight down the lens and " + cue(15, 2.75, 5.62, "driving, proud, louder", "I do not <i>shrink</i> for a stranger's demand") + ", a crooked grin breaking through at the end."),
         ]),
    stage(16, "Back in the hall the singer throws the last lines of the pre-chorus at the camera and the band slams the build behind her.",
          shots=[
              (0.0, CONT_OPEN + "The first frame is already a medium close-up at her eye level at the front centre of the band in {loc}: {singer} holds " + MIC + " at her lips in her right hand, holds her left palm out flat toward the camera and " + cue(16, 0.0, 3.15, "bold, sarcastic, loud", "If existence is costly, then send me the <i>bill</i>") + ", one brow arched; behind her, soft in the background, {drummer}, the band's only drummer, pounds his black drum kit."),
              (3.15, C + "{singer} still at the front centre, the camera cuts to a low shot from the open floor with " + ALL4 + " that pushes in fast toward her: she stamps her boot, throws her head back, " + MIC2 + " at her lips in her right hand, and " + cue(16, 3.15, 7.02, "belting, huge, triumphant", "I am already here and I am <i>louder</i> still") + ", her mouth open wide on the last word, while {guitarist} on the left and {bassist} on the right jump on the spot with their instruments and {drummer} hammers the kit at the back centre, all three with their lips pressed together."),
          ]),

    # ------------------------------------------------------------ bridge (spoken, the band almost silent): the lift
    duo(17, "lift", "In the quiet lift she stands beside the clerk and speaks the bridge straight to the camera, deadpan, while he stares ahead.",
        style="Her face stays deadpan and very still, the words dry and quiet, with only the eyes and one brow moving. The clerk stays stiff and flustered, " + CLERK_QUIET + ".",
        shots=[
            (0.0, CONT_OPEN + "The first frame is already a waist-up medium two-shot looking into the cabin of {loc}: {singer} stands on the right beside the steel panel of amber buttons with her hands in her jacket pockets, and {clerk} stands on the left clutching a brown clipboard to his chest and staring straight ahead; she turns her eyes to the lens and " + say(17, 0.0, 3.55, "deadpan, dry, quiet", "Let the silence <i>accuse</i> me, let the righteous complain") + "; he glances sideways at her and quickly back to the front, " + CLERK_QUIET + "."),
            (3.55, C + "{singer} still standing in the same spot with her hands in her pockets, the camera cuts to a close-up at her eye level with the shoulder of the clerk's grey suit at the left edge of the frame: she gives a tiny shake of her head and " + say(17, 3.55, 6.71, "deadpan, low, certain", "I will not translate my refusal to <i>shame</i>") + ", then holds a still, level stare into the lens."),
        ]),
    duo(18, "lift", "Still in the lift she sweeps her palm down the button panel so every button lights, leans to the camera for the last line and walks out, leaving the clerk alone.",
        style="Her face starts deadpan and breaks into a wide wolfish grin on the last line. The clerk stays stiff and flustered, " + CLERK_QUIET + ".",
        shots=[
            (0.0, CONT_OPEN + "The first frame is already a waist-up medium two-shot looking into the cabin of {loc}: {singer} on the right and {clerk} on the left stand side by side facing the camera; she shrugs one shoulder and " + say(18, 0.0, 3.3, "dry, matter-of-fact, low", "There is no clean version of breaking the <i>chain</i>") + ", and as she speaks she lifts her right hand to the steel panel beside her and sweeps her palm slowly down both columns of round buttons so that every button glows bright amber; {clerk} stares at the glowing buttons, " + CLERK_QUIET + "."),
            (3.3, C + "{singer} with her hand just leaving the glowing button panel, the camera cuts to a close-up slightly below her eye level: she leans in toward the lens and " + say(18, 3.3, 6.89, "gleeful, sharp, defiant", "So <i>fuck</i> your permission, I will not explain") + ", a wide wolfish grin spreading as she speaks, then she walks forward past the camera and out of the frame, leaving {clerk} alone in the lift behind her, blinking at the glowing buttons with " + CLERK_QUIET + "."),
        ]),

    # ------------------------------------------------------------ final chorus: the band and the rooftop
    stage(19, "The band explodes into the final chorus in the hall and the singer attacks it with a raw, feral edge.",
          shots=[
              (0.0, CONT_OPEN + "A high shot in {loc} with " + ALL4 + ": the camera starts up near the fluorescent tubes and drops fast on a crane down toward the singer: " + BAND_PLAYS + "; {singer} at the front centre bounces on her toes with " + MIC + " at her lips in her right hand, her left fist pumping, and " + cue(19, 0.0, 3.0, "feral, gravelly, belting", "I need no <i>permission</i> to exist") + ". " + LIPS3 + "."),
              (3.0, C + "{singer} still at the front centre with the band playing at their spots behind her, the camera cuts to a close-up at her eye level that shakes slightly with the beat: she grips " + MIC2 + " at her lips in both hands, her hair whipping as she snaps her head, and " + cue(19, 3.0, 6.7, "feral, snarling, loud", "No holy approval, no ritual <i>insistence</i>") + ", her lip curling on the last word."),
          ]),
    solo(20, "roof", "On the rooftop at dusk she strides toward the camera with the London skyline behind her and thumps her chest as she sings.",
         style=STYLE_RUN, light=L_ROOF,
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a waist-up medium shot tracking backward in front of her: {singer} strides toward the camera across the gravel of {loc}, the wind lifting her hair and the orange and magenta sky over the London skyline behind her, flicking her right hand away over her shoulder as she " + cue(20, 0.0, 3.2, "mocking, loud, free", "You can keep your blessing, your <i>patronising</i> tone") + ", a crooked grin on her face."),
             (3.2, C + "{singer} now standing still near the low concrete parapet, the camera cuts to a low-angle close-up against the glowing sky: she thumps her right fist against her chest twice and " + cue(20, 3.2, 6.23, "fierce, proud, gravelly", "I am not a <i>product</i>, I am not your <i>own</i>") + ", her chin lifted and her eyes on the lens."),
         ]),
    stage(21, "The band shouts the refrain together in the hall, and at the side of the hall the clerk, his tie pulled loose, jumps up and down to the music.",
          cast=BAND + ["clerk"], loc_clause=STAGE_MAP_CLERK,
          shots=[
              (0.0, CONT_OPEN + "The first frame is already a medium shot from the open floor in {loc}, the camera arcing from the right around three people: {singer} in the middle with " + MIC + " at her lips in her right hand " + cue(21, 0.0, 4.9, "shouted, feral, chanting", "No <i>permission</i>, <breath> no <i>permission</i>") + ", punching her left fist up on each word; {bassist} on the right and {guitarist} on the left lean in toward her from their spots and shout the same two words together with her each time, their mouths opening wide on the words and closing between them; behind them {drummer}, the band's only drummer, pounds his black drum kit with his lips pressed together."),
              (4.9, "Hard cut, the same moment in the same hall: a medium shot of {clerk} standing alone among the orange plastic chairs at the right side of {loc}, his mustard-yellow tie pulled loose and his glasses slightly crooked, jumping up and down on the spot in time with the beat and punching a sheaf of paper forms into the air, a tight delighted grin on his closed lips, the camera pushing in on him."),
          ]),
    solo(22, "roof", "At the rooftop parapet she sings the last line of the chorus and flings two handfuls of torn paper into the dusk sky.",
         style=STYLE_RUN, light=L_ROOF,
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a medium shot at her eye level: {singer} stands on the gravel of {loc} beside the low concrete parapet with the London skyline and the sunset sky behind her, both fists full of ripped white paper scraps held at her chest, and she " + cue(22, 0.0, 5.2, "belting, soaring, triumphant", "I exist in open <i>opposition</i>") + ", and on the long last word she flings both handfuls of paper scraps high up into the air above her head."),
             (5.2, C + "{singer} with both arms still raised from the throw, the camera cuts to a low-angle medium shot looking up at her: the white paper scraps swirl and flutter down around her in the wind against the orange and magenta sky as she spreads her arms wide and tips her face up with a huge closed-lip grin."),
         ]),

    # ------------------------------------------------------------ ending: break, last words, outro
    stage(23, "In the instrumental break the camera whips across the band as they hammer the riff.",
          style=STYLE_BAND, audio="inst",
          shots=[
              (0.0, "The first frame is already a low-angle medium shot of {guitarist} at his own spot on the left of the band in {loc}, slashing clipped chords on his white electric guitar in time with the guitar in <Audio 1>, his red mohawk shaking; at about one and a half seconds the camera whip-pans to the right across {singer} at the front centre, who headbangs with " + MIC + " held against her chest in her right hand and her lips closed, and on past {drummer}, the band's only drummer, pounding his black drum kit at the back centre, and lands on {bassist} at her own spot on the right as she drives her cherry-red electric bass guitar in time with the bass, stomping one platform shoe on the beat; all four keep their lips pressed together."),
          ]),
    stage(24, "The singer snarls the first words of the last refrain in a tight close-up, then the whole band slams back in.",
          shots=[
              (0.0, CONT_OPEN + "The first frame is already a tight close-up at her eye level at the front centre of the band in {loc}: {singer} holds " + MIC + " at her lips in her right hand, glares into the lens and " + cue(24, 0.0, 2.3, "feral, gravelly, snarled", "No <i>permission</i>") + ", then closes her lips in a hard crooked grin and holds the stare."),
              (2.3, C + "{singer} still at the front centre, the camera cuts to a low shot from the open floor with " + ALL4 + " that pushes in fast toward her as the band slams the riff: " + BAND_PLAYS + "; {singer} whips her head down and up with her hair flying, " + MIC2 + " held at her chest in her right hand, her lips closed. " + LIPS3 + "."),
          ]),
    stage(25, "The singer growls the very last words of the song, and at the side of the hall the clerk stands on a chair conducting the band with his pen.",
          cast=BAND + ["clerk"], loc_clause=STAGE_MAP_CLERK,
          shots=[
              (0.0, CONT_OPEN + "The first frame is already a low-angle medium shot at the front centre of the band in {loc}: {singer} grips " + MIC + " at her lips in both hands, bends forward toward the lens and " + cue(25, 0.0, 2.4, "feral, growled, final", "to <i>exist</i>") + ", the last word long and rough, then straightens up with her lips closed and a fierce grin; behind her {guitarist} on the left of the frame slashes his white electric guitar, {bassist} on the right of the frame drives her cherry-red electric bass guitar and {drummer}, the band's only drummer, pounds his black drum kit at the back centre, all three with their lips pressed together, exactly four band members in the frame."),
              (2.4, "Hard cut, the same moment in the same hall: a low-angle medium shot with exactly one person in the frame, {clerk} alone, standing up on an orange plastic chair at the right side of {loc}, his mustard-yellow tie pulled loose, conducting the band with a pen in his raised right hand in big wild strokes on the beat and nodding his head hard, a delighted grin on his closed lips, the camera rising toward him."),
          ]),
    stage(26, "The band plays out the outro in the hall as sheets of paper drift down from above and the camera pulls up and back.",
          style=STYLE_BAND, audio="inst",
          shots=[
              (0.0, "The first frame is already a medium shot at the front centre of the band in {loc}: {singer} headbangs on the beat with " + MIC + " held against her chest in her right hand and her lips closed, while white sheets of paper drift down from above all around her; the camera pulls back and cranes up from her until the frame holds " + ALL4 + " under the falling paper: " + BAND_PLAYS + ". " + LIPS3 + ", and the camera keeps rising to the last frame."),
          ]),
    stage(27, "The band drives the last bars to the final crashing chord in the hall and plays right through it.",
          style=("Every face is fierce and grinning with the lips pressed together; the guitarist and the bassist "
                 "stomp at their own spots and the drummer hits hard, seated at his kit."),
          audio="inst",
          shots=[
              (0.0, CONT_OPEN + "A low shot from the open floor in {loc} with " + ALL4 + ", the camera pushing in steadily toward the singer as white sheets of paper drift down: " + BAND_PLAYS + "; {singer} at the front centre stamps on the beat with " + MIC + " held at her chest in her right hand, her lips closed. At about three seconds the band lands the final crashing chord together and keeps playing through it: {guitarist} sweeps his pick hand down across the strings in one huge last chord and lets it ring with his fretting hand on the neck, {bassist} hits a last deep note with her fingers on the strings, {drummer} brings both sticks down together on the cymbals and the snare, seated at his kit, and {singer} throws her left fist straight up with a wide closed-lip grin; all four hold their places with their instruments still sounding as the last chord rings out. " + LIPS3 + "."),
          ]),
]
