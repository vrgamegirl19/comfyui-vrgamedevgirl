"""NO SAINT OF YOURS: elegant gothic rock music video (MiniMax H3, 30 scenes, 2:59.6), written 2026-11-12.

Brief (rules.txt): the band plays ONLY on the stage (glass conservatory); the singer alone takes a solo journey through
every other location with her hands free; on stage she always holds the same black handheld microphone.
Stage map: singer front centre, guitarist left of the frame, bassist right of the frame, drummer back centre.
Stems: clean guitar alone 0-6.5 s, drums and bass enter ~7.4 s, guitar solo 103-121.6 s, voice alone 122-130 s,
drums-only break 138-140 s, guitar quiet 157-168 s, last full-band hit ends 177.1 s.
Journey (never a literal re-enactment of the lyrics): study -> courtyard -> dressing room -> hallway -> front door ->
courtyard -> garden gate at dawn.
Variants: "stage" = singer on stage with the microphone; "band" = stage shots with band members only.
Scenes are 2-8 s (user rule), continuity off, several shots where they fit, cuts in vocal gaps.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_TL = json.load(open(os.path.join(_HERE, "timeline.json"), encoding="utf-8"))["scenes"]
_WORDS = json.load(open(os.path.join(_HERE, "lyrics_timed.json"), encoding="utf-8"))["words"]


def cue(n, a, b, emo, text):
    """A timed lyric cue for the words that start inside shot window a..b (scene-relative seconds) of scene n."""
    start = _TL[n - 1][0]
    sung = [w for w in _WORDS if start + a - 0.05 <= w["start"] < start + b]
    if not sung:
        raise ValueError(f"scene {n}: no words between {a} and {b} s")
    t0, t1 = max(0.0, sung[0]["start"] - start), min(_TL[n - 1][1] - start, sung[-1]["end"] - start)
    return f"{{sung}} from {t0:.2f} s to {t1:.2f} s, <d>[English, {emo}, singing] {text}</d>"


STYLE = ("Every face performs with restrained, smouldering intensity that shifts from beat to beat: a steady level "
         "gaze, one brow lifting, a slow knowing half-smile, the jaw setting and the chin lifting in quiet defiance.")
STYLE_BIG = ("Every face performs with proud, passionate intensity that stays elegant and controlled: the eyes bright "
             "and steady, the brows lifting on the big notes, the chin lifted high and a defiant smile breaking through "
             "between lines; the guitarist and the "
             "bassist move with big live-show energy at their own spots, and the drummer plays hard, seated at his kit.")
STYLE_BAND = ("Every face is locked into the music with cool, confident intensity: narrowed eyes, a slow proud "
              "half-smile, the head nodding on the beat, the lips pressed together.")
STYLE_DAWN = ("Her face is calm, open and free: a slow breath, the brows soft, the chin lifted and a quiet, certain "
              "smile that grows from beat to beat.")

SING = "sings with passion in sync with the vocals in <Audio 1>"
SUNG = "sings in sync with the vocals in <Audio 1>"

_SONG = "an elegant gothic rock song at a steady 120 BPM with a rich, expressive female alto lead vocal"
TEXT = {
    "audio_inst": "<Audio 1> is the complete instrumental gothic rock music track for the target video, reused as its complete final soundtrack and timing reference.",
    "sound_full": f"<Audio 1> remains the sole complete soundtrack with its original mix, timing, vocals, music, and dynamics intact: {_SONG}.",
    "sound_inst": "<Audio 1> remains the sole complete soundtrack with its original mix, timing, and dynamics intact: an elegant gothic rock instrumental at a steady 120 BPM.",
    "music_full": "<Audio 1> is reused as the complete song: elegant, romantic gothic rock at a steady 120 BPM with a rich female alto lead vocal.",
    "music_inst": "<Audio 1> is reused as the complete music track: elegant, romantic gothic rock at a steady 120 BPM.",
    "music_stage": "<Audio 1> is reused as the complete song: elegant gothic rock played live by the band, with a shimmering clean and soaring distorted electric guitar, a dark melodic electric bass, crisp acoustic drums and a rich female alto lead vocal.",
    "music_band": "<Audio 1> is reused as the complete music track: elegant gothic rock played live by the band, with a shimmering clean and soaring distorted electric guitar, a dark melodic electric bass and crisp acoustic drums.",
}

LINT = {
    "forbid_outside_variants": {
        "pattern": r"\b(microphones?|mic|guitars?|bass|drums?|drumsticks?|sticks|cymbals?|instruments?|amplifiers?|amps?)\b",
        "variants": ["stage", "stage_ng", "band"],
    },
    "require_in_variant": {"stage": r"\b(microphones?|mic)\b", "stage_ng": r"\b(microphones?|mic)\b"},
}
MIC_VARIANTS = ["stage", "stage_ng"]
POSTER_CARDS = ["singer", "guitarist", "bassist", "drummer"]

MIC = "a standard all-black handheld vocal microphone with a black metal mesh grille"
MIC2 = "the same all-black handheld microphone"
MIC_NG = "a standard all-black handheld vocal microphone"

_SINGER = ("the band's lead singer from the {o} character reference, a woman with long wavy dark brown hair with "
           "caramel-blonde ends and wispy bangs, brown eyes, garnet drop earrings and a small garnet pendant on a fine "
           "silver chain, wearing a long black velvet gown with a draped neckline, sheer black floral-lace long "
           "sleeves and deep burgundy satin panels in the skirt")

CHARACTERS = {
    "singer": {
        "card": "singer",
        "desc": _SINGER + ", her hands empty and free",
        "retain": "Long wavy dark brown hair with caramel ends and bangs, facial identity, garnet earrings and pendant, and the black velvet gown with lace sleeves and burgundy satin panels.",
        "variants": {"stage_ng": {"desc": _SINGER + ", holding " + MIC_NG + " in her right hand",
                               "retain": "Long wavy brown hair with caramel ends and bangs, facial identity, garnet jewellery, black velvet and lace gown, and the all-black handheld microphone."},
                     "stage": {"desc": _SINGER + ", holding " + MIC + " in her right hand",
                               "retain": "Long wavy brown hair with caramel ends and bangs, facial identity, garnet jewellery, black velvet and lace gown, and the all-black handheld microphone."}},
    },
    "guitarist": {
        "card": "guitarist",
        "desc": "the band's lead guitarist from the {o} character reference, a lean man with shoulder-length wavy dark auburn hair with a grey streak at the temple, wearing a long black velvet frock coat with silver vine embroidery on the lapels and cuffs over a burgundy brocade waistcoat and a black shirt, and black trousers, playing his dark red electric guitar",
        "retain": "Dark auburn wavy hair with a grey streak, facial identity, black velvet frock coat, burgundy waistcoat, and the dark red electric guitar.",
    },
    "bassist": {
        "card": "bassist",
        "desc": "the band's bassist from the {o} character reference, a woman with short black tightly curled hair and a silver nose stud, wearing a sleeveless high-collared black tunic vest with a raised thorn-vine pattern and silver filigree clasps over a burgundy satin cravat, a wide silver cuff on her right wrist, and wide-legged black trousers, playing her glossy black electric bass guitar",
        "retain": "Short black curly hair, facial identity, sleeveless black tunic vest, burgundy cravat, and the glossy black electric bass guitar.",
    },
    "drummer": {
        "card": "drummer",
        "desc": "the band's drummer from the {o} character reference, a muscular man with black hair swept to one side over an undercut and a small silver hoop earring, wearing a sleeveless high-collared black velvet vest with burgundy baroque brocade panels and silver filigree clasps, and black trousers, seated behind his dark red drum kit with brass cymbals, one wooden drumstick in each hand",
        "retain": "Black side-swept undercut hair, facial identity, sleeveless black and burgundy velvet vest, and the dark red drum kit.",
    },
}
PROPS = {}

LOCATIONS = {
    "stage": {"card": "stage",
              "desc": "the gothic glass conservatory at night from the {o} reference image, featuring a soaring wrought-iron and glass dome with rain running down the panes, a full moon behind drifting clouds, climbing red roses and ivy on the iron pillars, glowing brass lanterns on the floor, a black and white checkered marble floor, and a low dark wooden stage platform in front of tall gothic glass doors",
              "retain": "glass dome, full moon, climbing red roses, brass lanterns, checkered marble floor, and the low wooden stage platform"},
    "office": {"card": "office",
               "desc": "the gothic study at night from the {o} reference image, featuring a large polished mahogany desk with an opened envelope, a handwritten letter and a fountain pen, a glowing amber glass desk lamp, a carved stone fireplace with a burning fire, tall bookshelves of leather-bound books, deep burgundy velvet curtains, and a tall leaded window streaked with rain",
               "retain": "mahogany desk with the letter and envelope, amber desk lamp, stone fireplace with a burning fire, bookshelves, burgundy velvet curtains, and the rain-streaked leaded window"},
    "courtyard": {"card": "courtyard",
                  "desc": "the gothic stone courtyard at night from the {o} reference image, featuring a cloister of pointed stone arches with hanging iron lanterns glowing amber, two long burgundy velvet banners, ivy climbing the old stone walls, stone urns of dark red flowers, broad rain-slick flagstones mirroring the lantern light, bare trees in mist beyond the arches, and a full moon behind dark clouds",
                  "retain": "pointed stone arches, glowing hanging lanterns, burgundy banners, ivy, rain-slick flagstones, misty trees, and the moon"},
    "vanity": {"card": "vanity",
               "desc": "the candlelit dressing room at night from the {o} reference image, featuring a carved dark wood vanity table with a tall oval silver-framed mirror, a silver candelabra with three burning candles, a silver tray holding ornate silver hairpins, a silver vase of dark red roses, a burgundy velvet stool, a black damask folding screen, burgundy velvet curtains, dark polished floorboards, and a tall window streaked with rain",
               "retain": "vanity table with the oval silver mirror, candelabra, silver tray of hairpins, red roses, burgundy velvet stool, black folding screen, and the rain-streaked window"},
    "hallway": {"card": "hallway",
                "desc": "the long gothic hallway at night from the {o} reference image, featuring tall leaded windows streaked with rain along the left wall, dark carved wood panelling, a long burgundy patterned runner on dark polished floorboards, gilt-framed oil paintings and brass candle sconces along the right wall, a burgundy velvet wingback armchair with a long black wool coat draped over it beside a small round table with a single candle, and a pointed stone archway at the far end",
                "retain": "tall rain-streaked windows, dark wood panelling, burgundy runner, gilt-framed paintings, candle sconces, the burgundy velvet armchair with the black coat, and the far archway"},
    "door": {"card": "door",
             "desc": "the gothic entrance hall at first light from the {o} reference image, featuring a heavy carved dark oak door standing open in a pointed stone archway onto a misty garden path with bare trees under a pale lilac dawn sky, a glowing iron wall lantern, a row of iron coat hooks, a brass stand of umbrellas, a deep red runner on a black and white checkered marble floor, and a carved dark wood console table holding an ornate silver key in a small silver dish, a silver vase of dark red hydrangeas and old leather books",
             "retain": "open carved oak door in the stone archway, misty dawn garden beyond, iron lantern, red runner on the checkered floor, and the console table with the silver key in its dish"},
    "outside": {"card": "outside",
                "desc": "the garden path at dawn from the {o} reference image, featuring tall black wrought-iron gates standing open between two mossy stone pillars topped with stone spheres, a broad path of rain-slick flagstones scattered with fallen autumn leaves, clipped dark hedges, a mossy stone balustrade with a single dark red rose blooming beside it, huge bare oak trees in drifting mist, and a glowing peach and gold sunrise beyond the gates",
                "retain": "open wrought-iron gates between mossy stone pillars, rain-slick flagstone path with autumn leaves, hedges, stone balustrade with the red rose, bare oaks in mist, and the sunrise"},
}

# ---------------------------------------------------------------- stage constants
STAGE_MAP = ("The band plays on the low dark wooden stage platform in front of the tall gothic glass doors. Fixed "
             "positions: the drummer seated behind the dark red drum kit at the back centre of the platform, the lead "
             "singer at the front centre, the guitarist at his own spot on the left of the frame and the bassist at "
             "her own spot on the right of the frame. The stage holds "
             "exactly one drum kit, one drummer, one guitarist, one bassist and one singer, and every shot faces the "
             "stage from the marble floor in front of it or from its side.")
STAGE_MAP2 = ("The band plays up on the raised low dark wooden stage platform in front of the tall gothic glass doors, "
              "all four on its boards a step above the marble floor. Fixed positions: the drummer seated behind the dark red "
              "drum kit at the back centre, the lead singer at the front centre, the guitarist at his own spot at the left "
              "end and the bassist at her own spot at the right end. The stage holds exactly one drum kit, one drummer, "
              "one guitarist, one bassist and one singer, and every shot faces the stage from the marble floor in front of "
              "it or from its side.")
ON_PLAT = "all four band members up on the raised wooden platform with the empty marble floor below them"
ROAM = ("This reference defines the environment, lighting, and atmosphere, while the camera uses its own area of the "
        "space and places her in new positions within it. She is the only person in the place.")
TAIL_STAGE = ("The scene preserves each band member's appearance and instrument from their character references. "
              "Everything moves at normal real-time speed.")
TAIL_SOLO = ("The scene keeps the location's environment and art style and preserves her appearance from her "
             "character reference. She is alone, and everything moves at normal real-time speed.")
CONT_OPEN = ("All shots are one continuous performance: every person keeps their exact position, posture and ongoing "
             "movement across each camera cut, and only the camera angle and framing change. ")
C = "continuing directly from the previous shot's final moment, "
BAND = ["singer", "guitarist", "bassist", "drummer"]
G_PLAYS = "{guitarist} at his own spot on the left of the frame plays his dark red electric guitar in time with the guitar in <Audio 1>"
B_PLAYS = "{bassist} at her own spot on the right of the frame plucks her glossy black electric bass guitar in time with the bass"
D_PLAYS = "{drummer}, the band's only drummer, seated behind his dark red drum kit at the back centre, drives the beat with one drumstick in each hand"
BAND_PLAYS = G_PLAYS + ", " + B_PLAYS + ", and " + D_PLAYS
LIPS3 = "{guitarist}, {bassist} and {drummer} keep their lips pressed together"

L_STAGE = "cold blue moonlight through the glass dome and warm amber lantern glow"
L_IN = "warm candle and firelight against cold blue rain light"
L_DAWN = "soft peach and lilac dawn light"


def stage(n, summary, shots, cast=None, style=STYLE_BIG, audio="full", loc_clause=STAGE_MAP, variant="stage"):
    return dict(n=n, cast=cast or BAND, loc="stage", use="environment", audio=audio, variant=variant, light=L_STAGE,
                style=style, summary=summary, summary_tail=TAIL_STAGE, loc_clause=loc_clause, shots=shots)


def solo(n, loc, summary, shots, style=STYLE, audio="full", light=L_IN):
    return dict(n=n, cast=["singer"], loc=loc, use="environment", audio=audio, light=light, style=style,
                summary=summary, summary_tail=TAIL_SOLO, loc_clause=ROAM, shots=shots)


SCENES = [
    # ------------------------------------------------------------ intro (stems: clean guitar alone to 6.5 s)
    stage(1, "Alone on the conservatory stage under the moonlit glass dome, the guitarist plays the clean opening motif.",
          cast=["guitarist"], variant="band", style=STYLE_BAND, audio="inst",
          loc_clause="Only one person is in this scene: the guitarist, standing alone at the left side of the low dark wooden stage platform in front of the tall gothic glass doors.",
          shots=[
              (0.0, CONT_OPEN + "The first frame is already a close shot of a dark red electric guitar: the camera sits close on the hands of {guitarist} as his fingers pick a slow, shimmering clean motif on the strings in time with the guitar in <Audio 1>, the lantern light of {loc} soft behind him, then the camera glides slowly up along the guitar neck to his face, his eyes lowered to the strings, his lips pressed together."),
              (3.4, C + "{guitarist} still playing the same motif at the same spot, the camera cuts to a medium shot from the marble floor in front of the platform, drifting slowly sideways: he stands alone on the low wooden platform of {loc} with the rain-streaked glass dome and the full moon high behind him, sways gently with the motif, then lifts his eyes to the camera with a slow, proud half-smile, his lips closed, and holds the look."),
          ]),
    stage(2, "On the conservatory stage the drummer crashes in and the bassist joins with a dark melodic line.",
          cast=["drummer", "bassist"], variant="band", style=STYLE_BAND, audio="inst",
          loc_clause="Two band members appear in this scene: the drummer seated behind the dark red drum kit at the back centre of the raised low dark wooden stage platform, and the bassist standing at her own spot at the right end of the platform. The stage holds exactly one drum kit and one drummer.",
          shots=[
              (0.0, "The first frame is already a medium shot from the marble floor at the front right of the platform in {loc}, looking up at the raised platform: {drummer}, the band's only drummer, sits behind his dark red drum kit at the back centre with one drumstick in each hand, raises both sticks, then at one second brings them down hard on the snare and a brass cymbal and settles into a crisp, steady beat in time with the drums in <Audio 1>, his shoulders rolling and his eyes narrowed in focus, while {bassist} stands at her own spot at the right end of the platform in the foreground of the frame and plucks her glossy black electric bass guitar in time with the bass in <Audio 1>, nodding on the beat; both keep their lips pressed together, and the camera pushes in slowly toward her."),
              (2.6, "Hard cut, the same moment on the same stage: a close medium shot of {bassist} at her own spot at the right end of the platform in {loc}, the camera rising slowly from her plucking hand on the strings of her glossy black electric bass guitar up to her face as she plays a dark, melodic line in time with the bass in <Audio 1>, her head nodding on the beat, a cool half-smile on her closed lips, her eyes lifting to the camera at the end."),
          ]),
    stage(3, "The full band plays the intro together on the conservatory stage while the singer waits at the front centre with her microphone lowered, then raises it.",
          style=STYLE_BAND,
          loc_clause=STAGE_MAP2,
          shots=[
              (0.0, CONT_OPEN + "A wide moving shot of {loc}: the camera starts high under the glass dome beside the hanging roses and cranes smoothly down and forward toward the low wooden platform, with " + ON_PLAT + ", as the full band plays: " + BAND_PLAYS + "; {singer} stands at the front centre with " + MIC + " lowered at her side in her right hand, her other hand resting on the skirt of her gown, her eyes lowered, swaying slowly with the beat, her lips closed. " + LIPS3 + "."),
              (3.3, C + "{singer} still at the front centre with the band playing behind her, the camera cuts to a medium close-up at her eye level, pushing in slowly: she lifts her eyes to the camera with a steady, level gaze, takes a slow breath through her nose with her lips closed, and raises " + MIC2 + " in her right hand up to just below her chin, her left hand at her side, holding the look into the lens."),
          ]),

    # ------------------------------------------------------------ verse 1: the study
    solo(4, "office", "Alone in the firelit study, she sings at the desk by the amber lamp and walks slowly around it toward the fireplace.",
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a medium close-up at her eye level: {singer} stands beside the mahogany desk in {loc}, the amber lamp glowing at her side and the tall leaded window behind her glowing a soft cold blue, her fingertips resting lightly on the garnet pendant at her collarbone as she looks straight into the camera and " + cue(4, 0.0, 3.3, "intimate, guarded", "You fasten <i>silver</i> at my throat,") + ", one brow lifting slightly, then lets her hand fall away from the pendant."),
             (3.3, C + "{singer} still beside the desk, the camera cuts to a waist-up medium shot that tracks sideways with her as she walks slowly along the edge of the desk toward the stone fireplace, smoothing the lace sleeve at her left wrist with her right hand, her eyes returning to the camera as she " + cue(4, 3.3, 7.4, "cool, restrained", "Then smooth the <i>creases</i> from my sleeve.") + ", her chin lifting a little, and she stops with the firelight on one side of her face."),
         ]),
    solo(5, "office", "In the study she picks up the handwritten letter from the desk, carries it to the fireplace and lets it fall into the fire.",
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a waist-up medium shot: {singer} stands at the mahogany desk in {loc} and lifts the handwritten letter from the desk in her right hand, glances down at the page, then raises her eyes to the camera and " + cue(5, 0.0, 3.35, "quiet, wounded", "You choose the <i>words</i> inside my note") + ", her jaw setting, the letter held at chest height and her left hand at her side."),
             (3.35, C + "{singer} still holding the letter in her right hand, the camera cuts to a medium close-up that moves with her as she turns and takes two slow steps to the stone fireplace, the flames bright beside her, and " + cue(5, 3.35, 7.4, "defiant, calm", "Before you let the <i>letter</i> leave.") + ", then she opens her fingers and lets the letter drop into the fire, the paper curling in the flames as she looks back into the camera with a slow, knowing half-smile."),
         ]),

    # ------------------------------------------------------------ pre-chorus 1: the courtyard
    solo(6, "courtyard", "Alone at night she walks along the lantern-lit stone cloister of the courtyard, trailing her hand over the old columns.",
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a waist-up medium shot tracking sideways beside her: {singer} walks slowly along the arcade of pointed stone arches in {loc}, the amber lanterns passing above her, her left hand gliding over a cold stone column as she turns her face to the camera and " + cue(6, 0.0, 3.95, "measured, bitter", "You call it <i>care</i>;") + ", her eyes narrowing."),
             (3.95, C + "{singer} still walking along the arcade, the camera cuts to a close-up at her eye level moving backward in front of her as she steps out from under the arch and " + cue(6, 3.95, 5.9, "low, certain", "I count the <i>cost</i>.") + ", her chin lifting, the lantern light warm on her face and the misty trees dark behind her."),
         ]),
    solo(7, "courtyard", "She stands in the open courtyard on the rain-slick flagstones under the moon as the camera circles her.",
         shots=[
             (0.0, "The first frame is already a medium close-up at her eye level: {singer} stands still in the middle of the rain-slick flagstones of {loc}, the glowing arches and burgundy banners behind her, her right hand pressed flat over her heart and her left hand at her side, and the camera glides in a slow arc around her from her front to her left side while her eyes follow the lens and she " + cue(7, 0.0, 3.8, "aching, proud", "I miss the <i>woman</i> you forgot.") + ", her brows drawing together, then her chin lifts and she holds a steady, defiant look into the camera."),
         ]),

    # ------------------------------------------------------------ chorus 1: the stage
    stage(8, "The full band launches into the chorus on the conservatory stage and the singer belts it at the front centre with her microphone.",
          shots=[
              (0.0, CONT_OPEN + "A wide low shot from the marble floor in front of the platform in {loc}, the camera sweeping in a fast arc from the left toward the centre with the whole band filling the frame under the moonlit glass dome: " + BAND_PLAYS + "; {singer} at the front centre holds " + MIC + " close to her lips in her right hand, her left arm sweeping out to her side, and " + cue(8, 0.0, 3.73, "belting, defiant", "I'm no <i>saint</i> of yours to save,") + ". " + LIPS3 + "."),
              (3.73, C + "{singer} still at the front centre with the band playing at their spots behind her, the camera cuts to a Dutch-tilted medium close-up from slightly below her eye level, pushing in: she grips " + MIC2 + " at her lips in her right hand, her left hand opening toward the camera, and " + cue(8, 3.73, 7.65, "fierce, proud", "No marble <i>angel</i> at your door.") + ", her gaze proud and steady; behind her the band plays on at their spots, {drummer} seated at his kit."),
          ]),
    stage(9, "On the conservatory stage the guitarist drives the chorus chords and the bassist digs into her line while the full band plays on.",
          style=STYLE_BAND, audio="inst", loc_clause=STAGE_MAP2,
          shots=[
              (0.0, CONT_OPEN + "A wide low shot from the marble floor in front of the platform in {loc} with " + ON_PLAT + ", the band driving the chorus: " + BAND_PLAYS + "; {singer} at the front centre sways with the beat, " + MIC + " lowered at her side in her right hand, her lips closed; the camera pushes in fast toward the left end of the platform and settles by one second on a low-angle medium shot of {guitarist}, who plants one boot forward and strikes big ringing chords on his dark red electric guitar in time with the guitar in <Audio 1>, his strumming arm swinging wide, his hair falling across his face as he nods hard on the beat, his lips pressed together in a fierce half-smile. " + LIPS3 + "."),
              (1.9, "Hard cut, the same moment on the same stage: a low-angle medium shot at the right end of the platform in {loc} of {bassist} leaning back with her glossy black electric bass guitar, her fingers walking the strings in time with the bass in <Audio 1>, her shoulders rocking with the groove, and she looks down into the camera with a proud, closed-lip smile as the camera pushes in."),
          ]),
    stage(10, "The singer belts the last line of the chorus at the front of the stage and holds the long final note as the camera pulls back to the full band, then the drummer drives the band on.",
          loc_clause=STAGE_MAP2,
          style=("Her face performs with proud, passionate intensity that stays elegant: the eyes bright and steady, the "
                 "brows lifting on the big note, the chin lifted high; the band plays hard at their own spots with their lips pressed together."),
          shots=[
              (0.0, "The first frame is already a medium close-up at her eye level at the front centre of the stage in {loc}: {singer} holds " + MIC + " close to her lips in her right hand, her left fist closing at her side, and " + cue(10, 0.0, 4.95, "belting, unbroken", "I won't <i>kneel</i> for you anymore.") + ", her mouth open wide and round on the long last note and held open to the end of the note, while the camera pulls back and cranes up and back from in front of the platform until, by the end of the shot, the frame holds " + ON_PLAT + ": " + BAND_PLAYS + ". " + LIPS3 + "."),
              (4.95, "Hard cut, the same moment on the same stage: a medium shot from the side of the platform in {loc} of {drummer}, the band's only drummer, seated behind his dark red drum kit with one drumstick in each hand, rolling a fast fill around the toms and crashing a brass cymbal in time with the drums in <Audio 1>, then locking back into the steady beat, his jaw set and his lips pressed together, the camera pushing in toward him."),
          ]),

    # ------------------------------------------------------------ verse 2: the dressing room
    solo(11, "vanity", "Alone in the candlelit dressing room she sits at the vanity table, runs her fingers through her hair and trails her hand across the silver tray.",
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a medium close-up at her eye level: {singer} sits on the burgundy velvet stool at the vanity table in {loc}, turned toward the camera with the candle flames and the oval silver mirror beside her, and slowly draws the fingers of her right hand through the long waves of her hair at her temple as she looks into the lens and " + cue(11, 0.0, 3.0, "hushed, deliberate", "I take the <i>pins</i> out of my hair,") + ", then lowers her hand to the table."),
             (3.0, C + "{singer} still seated on the stool at the vanity table, the camera cuts to a waist-up medium shot from the side as her right hand trails slowly across the silver tray of hairpins on the table and comes to rest beside it, and she turns her face back to the camera and " + cue(11, 3.0, 7.1, "cool, final", "Your silver <i>settles</i> on the tray.") + ", one brow lifting, the candlelight warm on her cheek."),
         ]),
    solo(12, "vanity", "She rises from the vanity and crosses the candlelit dressing room to the rain-streaked window.",
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a waist-up medium shot: {singer} rises from the burgundy velvet stool in {loc} and walks slowly across the dark floorboards toward the tall rain-streaked window, the camera tracking backward in front of her, and she glances back over her shoulder at the empty velvet stool, then faces the camera and " + cue(12, 0.0, 3.9, "wry, unafraid", "You watch me from the <i>velvet</i> chair;") + ", a slow knowing half-smile on her face."),
             (3.9, C + "{singer} now standing beside the tall window, the camera cuts to a close-up at her eye level with rain running down the glass behind her: she rests her left hand on the burgundy velvet curtain, looks straight into the lens and " + cue(12, 3.9, 7.45, "soft, resolved", "For once, there's <i>nothing</i> left to say.") + ", then closes her lips, lifts her chin and holds a calm, steady gaze."),
         ]),

    # ------------------------------------------------------------ pre-chorus 2: the hallway
    solo(13, "hallway", "She strides alone down the long gothic hallway past the rain-streaked windows and stops at one of them.",
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a waist-up medium shot tracking backward in front of her: {singer} strides down the burgundy runner of {loc} toward the camera, the tall rain-streaked windows passing on one side and the candle sconces and gilt-framed paintings on the other, her skirt swinging, her right hand brushing along the stone windowsill as she " + cue(13, 0.0, 4.1, "measured, burning", "You call it <i>care</i>; <breath> I count the <i>cost</i>.") + ", her eyes locked on the lens."),
             (4.1, C + "{singer} slowing to a stop beside a tall window, the camera cuts to a close-up at her eye level from the side: she lays her left palm flat on the cold rain-streaked glass, looks out for a beat, then turns her face to the camera and " + cue(13, 4.1, 7.8, "aching, proud", "I miss the <i>woman</i> you forgot.") + ", her brows drawing together and her chin lifting on the last word."),
         ]),

    # ------------------------------------------------------------ chorus 2: the stage
    stage(14, "Back on the conservatory stage the singer drives the second chorus with the full band at their spots.",
          loc_clause=STAGE_MAP2,
          shots=[
              (0.0, CONT_OPEN + "The first frame is already a waist-up medium shot from the marble floor in front of the platform in {loc}, the camera tracking sideways along the stage front from the left: {singer} at the front centre leans toward the lens with " + MIC + " at her lips in her right hand, her left hand pointing straight at the camera, and " + cue(14, 0.0, 3.85, "belting, scornful", "I'm no <i>saint</i> of yours to save,") + "; behind her " + BAND_PLAYS + ". " + LIPS3 + "."),
              (3.85, C + "{singer} still at the front centre, the camera cuts to a wide low shot from the marble floor that pushes in toward her while rising on a crane, with " + ON_PLAT + " under the glass dome: she throws her left arm high above her head, " + MIC2 + " at her lips in her right hand, and " + cue(14, 3.85, 7.78, "soaring, defiant", "No marble <i>angel</i> at your door.") + ", while {guitarist} on the left swings the neck of his guitar upward on the beat, {bassist} on the right rocks back with her bass, and {drummer} pounds his kit, seated at the back centre."),
          ]),
    stage(15, "On the conservatory stage the singer sings straight into the camera as the camera pushes in from the full band to her, with the bassist grooving behind her.",
          loc_clause=STAGE_MAP2,
          style=("Her face performs with proud, elegant intensity: the eyes bright and steady, one brow arching, a scornful little "
                 "smile; the band plays at their own spots with their lips pressed together."),
          shots=[
              (0.0, "The first frame is already a wide low shot from the marble floor in front of the platform in {loc} with " + ON_PLAT + ": " + BAND_PLAYS + "; {singer} at the front centre holds " + MIC + " at her lips in her right hand and " + cue(15, 0.0, 3.87, "mocking, sharp", "You want a <i>love</i> that will behave;") + ", and the camera pushes in fast, arcing toward her right side, to a medium close-up at her eye level by about one and a half seconds as she tilts her head, her left hand flicking outward and one brow arching high, with {bassist} at her own spot on the right of the frame soft in the background behind her, nodding on the beat. " + LIPS3 + "."),
          ]),
    stage(16, "The singer belts the last line of the second chorus and holds the long final note as the camera rises to show the whole band under the glass dome.",
          shots=[
              (0.0, "The first frame is already a medium close-up at her eye level at the front centre of the stage in {loc}: {singer} grips " + MIC + " at her lips in her right hand, her left fist pressed to her chest, and " + cue(16, 0.0, 7.8, "belting, triumphant", "I won't <i>kneel</i> for you <i>anymore</i>.") + ", and on the long last note she throws her left arm out wide and tilts her head back, her mouth open wide and round and held open to the end of the note, while the camera pulls back and cranes smoothly up from in front of the platform until the frame holds the whole band under the moonlit glass dome: " + BAND_PLAYS + ". " + LIPS3 + ", and the camera keeps rising to the last frame."),
          ]),

    # ------------------------------------------------------------ guitar solo (stems: lead guitar 103-121.6 s)
    stage(17, "On the conservatory stage the guitarist steps to the front and plays a soaring, melodic guitar solo while the band drives behind him.",
          style=STYLE_BAND, audio="inst", loc_clause=STAGE_MAP2,
          shots=[
              (0.0, CONT_OPEN + "A wide low shot from the marble floor in front of the platform in {loc} with " + ON_PLAT + ": {bassist} at her own spot at the right end plucks her glossy black electric bass guitar in time with the bass in <Audio 1>, {drummer}, the band's only drummer, drives the beat seated behind his dark red drum kit at the back centre, {singer} at the front centre sways with the beat, " + MIC + " lowered at her side in her right hand, her lips closed, and {guitarist} steps forward from the left end to the front edge of the platform, still on its boards, and bends a long, soaring high note on his dark red electric guitar in time with the lead guitar in <Audio 1>; the camera circles in toward him to a low-angle medium shot by two seconds as his fretting hand shakes the string high on the neck, his head tips back with his eyes closed and his hair falls from his face, the full moon above him. {bassist} and {drummer} keep their lips pressed together."),
              (3.0, C + "{guitarist} still at the front edge of the platform, the camera cuts to a close shot of his hands on the dark red electric guitar, his fingers running quickly up the fretboard in time with the lead guitar in <Audio 1>, then the camera tilts up to his face as he opens his eyes, looks into the lens with a slow, proud half-smile, his lips closed, and leans into the next note."),
          ]),
    stage(18, "During the guitar solo the drummer powers the beat, and the guitarist and the bassist lean back to back at the front of the stage.",
          style=STYLE_BAND, audio="inst", loc_clause=STAGE_MAP2,
          shots=[
              (0.0, "The first frame is already a wide shot from the side of the platform in {loc} with " + ON_PLAT + ": {guitarist} at the left end and {bassist} at the right end play on, {singer} at the front centre sways with the beat, " + MIC + " lowered at her side in her right hand, her lips closed, and {drummer}, the band's only drummer, seated behind his dark red drum kit at the back centre, hammers a hard, driving beat on the snare and the brass cymbals in time with the drums in <Audio 1>; the camera sweeps in a fast arc around the kit and settles by one and a half seconds on a medium shot of {drummer} with one drumstick in each hand, his arms swinging in big strokes, his hair shaking on every hit, his teeth set behind closed lips."),
              (3.0, "Hard cut, the same moment on the same stage: a waist-up medium shot from the marble floor in front of the platform in {loc}: {guitarist} on the left and {bassist} on the right lean back to back at the front of the platform, shoulders touching, he playing a climbing, singing lead line high on the neck of his dark red electric guitar in time with the lead guitar in <Audio 1> and she driving a dark pulsing line on her glossy black electric bass guitar in time with the bass in <Audio 1>, both rocking on the beat, {singer} soft behind them at the front centre of the platform and {drummer} soft at his kit at the back, then they turn their heads and trade a quick, proud closed-lip grin, the camera pushing in slowly."),
          ]),
    stage(19, "The whole band builds the end of the guitar solo on the conservatory stage while the singer moves with the music, her microphone lowered, then lifts it for the next line.",
          style=STYLE_BAND,
          loc_clause=STAGE_MAP2,
          shots=[
              (0.0, CONT_OPEN + "A wide low shot from the marble floor in front of the platform in {loc} with " + ON_PLAT + ", the camera gliding in a smooth arc from the right side toward the centre and pushing in slowly: {guitarist} at his own spot on the left of the frame climbs toward the last high note of his solo on his dark red electric guitar in time with the lead guitar in <Audio 1>, the guitar neck raised; " + B_PLAYS + "; {drummer}, the band's only drummer, drives the beat seated behind his dark red drum kit at the back centre; {singer} at the front centre sways with the rhythm, " + MIC + " lowered at her side in her right hand, her left hand lifting through her hair, her eyes closed and her lips closed. " + LIPS3 + "."),
              (3.4, C + "{singer} still at the front centre with the band playing behind her, the camera cuts to a medium close-up at her eye level, pushing in slowly: she opens her eyes to the camera, breathes in slowly through her nose with her lips closed, and raises " + MIC2 + " in her right hand to just below her chin, her left hand at her side, as the last guitar note rings out and she holds a still, steady gaze into the lens."),
          ]),

    # ------------------------------------------------------------ bridge (stems: voice alone 122-130 s)
    solo(20, "hallway", "In the quiet hallway she stands alone at the velvet armchair with one hand resting on the black wool coat that hangs over its back.",
         shots=[
             (0.0, "The first frame is already a close-up at her eye level with exactly one person in the frame: {singer} stands alone beside the burgundy velvet wingback armchair in {loc}, the candle on the small round table glowing beside her and rain on the tall windows behind; the long black wool coat hangs over the back of the armchair as an empty garment, and her right hand rests on its collar while her left hand stays at her side; she looks down at the coat, then lifts her eyes to the camera and " + cue(20, 0.0, 3.85, "hushed, tender", "My coat still smells of <i>autumn</i> rain.") + ", a small, private smile warming her face, the camera pushing in very slowly and staying close to the last frame."),
         ]),
    solo(21, "door", "In the entrance hall by the open door she lifts the silver key from its dish on the console table.",
         shots=[
             (0.0, "The first frame is already a medium close-up at her eye level: {singer} stands at the carved console table in {loc}, the open oak door and the pale misty dawn behind her, and her right hand lifts the ornate silver key from the small silver dish and holds it up at chest height between her fingers, her left hand resting on the table; she looks from the key into the camera and " + cue(21, 0.0, 3.65, "quiet, certain", "My fingers find the <i>waiting</i> key.") + ", then closes her fingers around the key and lifts her chin, the camera staying close to the last frame."),
         ]),
    solo(22, "door", "At the open front door she looks back one last time, then walks out through the stone archway into the misty dawn.",
         style=STYLE_DAWN, light=L_DAWN,
         shots=[
             (0.0, CONT_OPEN + "The first frame is already a waist-up medium shot inside {loc}: {singer} stands on the deep red runner in front of the open carved oak door with her right hand on the edge of the door, her body turned toward the misty dawn garden, looking back over her shoulder into the camera, and she " + cue(22, 0.0, 3.25, "level, unhurried", "You say my <i>name</i>, <breath> then say it again;") + ", her gaze steady."),
             (3.25, C + "{singer} still at the open door, the camera cuts to a close-up at her eye level as she turns fully to face the lens in the stone archway, the pale lilac dawn light glowing behind her hair, and " + cue(22, 3.25, 7.65, "warm, liberated", "This time, it sounds like <i>mine</i> to me.") + ", a quiet, certain smile spreading across her face; then she turns away and walks out through the archway onto the garden path, her skirt swaying, the camera holding on her as she goes."),
         ]),

    # ------------------------------------------------------------ final chorus
    stage(23, "On the conservatory stage the singer starts the final chorus almost alone over the drums, then the full band crashes back in around her.",
          loc_clause=STAGE_MAP2,
          shots=[
              (0.0, CONT_OPEN + "The first frame is already a close-up at her eye level at the front centre of the stage in {loc}: {singer} holds " + MIC + " at her lips in her right hand, her left hand open at her side, and " + cue(23, 0.0, 3.67, "low, burning", "I'm no <i>saint</i> of yours to save,") + ", her eyes fixed on the lens, her face and shoulders filling the frame and the dark glass doors soft and out of focus behind her."),
              (3.67, C + "{singer} still at the front centre, the camera cuts to a wide low shot from the marble floor on the centre line in front of the platform that rushes forward toward it as the band crashes back in under the glass dome, " + ON_PLAT + ", exactly four people on the stage: " + BAND_PLAYS + ", the one drum kit with its single drummer seated at its middle, directly behind {singer}; {singer} throws her left arm high, " + MIC2 + " at her lips in her right hand, and " + cue(23, 3.67, 7.6, "belting, euphoric", "No marble <i>angel</i> at your door.") + "."),
          ]),
    solo(24, "courtyard", "She strides alone across the rain-slick courtyard toward the camera, her arms sweeping wide.",
         style=STYLE_BIG.split(";")[0] + ".",
         shots=[
             (0.0, "The first frame is already a waist-up medium shot tracking backward in front of her: {singer} strides fast across the rain-slick flagstones of {loc} straight toward the camera, the glowing stone arches and burgundy banners behind her, her skirt whipping around her legs and her hair flying, and she sweeps both arms out wide and " + cue(24, 0.0, 3.85, "mocking, fearless", "You want a <i>love</i> that will behave;") + ", her gaze proud and steady and a defiant smile breaking through as she keeps coming toward the lens."),
         ]),
    stage(25, "On the conservatory stage the singer belts the last line of the final chorus with the whole band at full power.",
          shots=[
              (0.0, CONT_OPEN + "The first frame is already a medium close-up at her eye level at the front centre of the stage in {loc}: {singer} grips " + MIC + " at her lips in her right hand, shakes her head slowly, her left fist pressed to her chest, and " + cue(25, 0.0, 2.7, "belting, unbreakable", "I won't <i>kneel</i>") + ", her gaze proud and steady; behind her, soft in the background, {drummer}, the band's only drummer, plays seated behind his dark red drum kit."),
              (2.7, C + "{singer} still at the front centre, the camera cuts to a waist-up low shot from the marble floor that arcs around her from her left to her front with the band at their spots behind her: she flings her left arm straight up, " + MIC2 + " at her lips in her right hand, and " + cue(25, 2.7, 5.8, "soaring, triumphant", "for you <i>anymore</i>.") + ", her mouth open wide and round on the long last note and held open to the end of the shot, while {guitarist} on the left and {bassist} on the right play hard at their own spots and {drummer} pounds his kit, seated at the back centre, all three with their lips pressed together."),
          ]),
    stage(26, "The singer holds the long final note of the chorus at the front of the stage while the camera pushes in from the full band to her and then rises high over the band under the glass dome.",
          loc_clause=STAGE_MAP2,
          shots=[
              (0.0, "A wide low shot from the marble floor in front of the platform in {loc} with " + ON_PLAT + ": " + BAND_PLAYS + "; {singer} at the front centre stands with her left arm stretched out wide and her head tilted back, " + MIC + " at her lips in her right hand, and {sing}, one long, soaring held note [English, soaring, euphoric, singing], her mouth open wide and round, while the camera pushes in toward her. " + LIPS3 + "."),
              (2.6, C + "{singer} still holding the same long note at the front centre, the camera cuts to a medium close-up at her eye level, her mouth open wide and round and held open until five seconds, then she slowly lowers " + MIC2 + " to her chest and closes her lips, breathing hard through her nose with her eyes shut, while the camera pulls back and cranes up and back until the frame holds the whole band up on the raised platform under the glass dome, the lanterns and the checkered marble floor spreading out below, and keeps rising to the last frame; {guitarist}, {bassist} and {drummer} play on at their spots behind her with their lips pressed together."),
          ]),

    # ------------------------------------------------------------ ending: the garden gate at dawn
    solo(27, "outside", "At dawn she stands alone on the garden path by the mossy stone balustrade, with the open iron gates and the sunrise behind her.",
         style=STYLE_DAWN, light=L_DAWN,
         shots=[
             (0.0, "The first frame is already a medium close-up at her eye level: {singer} stands on the rain-slick flagstone path of {loc} beside the mossy stone balustrade, the single dark red rose beside her and the open wrought-iron gates and the glowing sunrise behind her, her left hand resting on the stone; she looks straight into the camera and " + cue(27, 0.0, 4.2, "calm, final", "No <i>saint</i> of yours.") + ", then she lifts her hand from the balustrade, turns and starts walking toward the open gates, the camera drifting after her."),
         ]),
    solo(28, "outside", "She walks out through the open iron gates into the sunrise and holds the last long note with her arms opening wide.",
         style=STYLE_DAWN, light=L_DAWN,
         shots=[
             (0.0, "The first frame is already a waist-up medium shot tracking backward in front of her: {singer} walks toward the camera through the open wrought-iron gates of {loc}, the mossy stone pillars passing on either side of her and the misty garden path behind her, the peach and gold sunrise lighting her face, and she " + cue(28, 0.0, 7.6, "free, soaring", "Not <i>yours</i> <i>anymore</i>.") + ", and on the long last note she slows to a stop, opens both arms wide at her sides with her palms turned up and tilts her face up to the light, her mouth open wide and round and held open to the end of the note, while the camera rises slowly in front of her; then she closes her lips in a quiet, certain smile."),
         ]),
    stage(29, "On the conservatory stage the whole band drives the final bars to one last crashing hit and holds the final pose.", variant="stage_ng",
          style=("Every face is fierce and proud with the lips pressed together; the guitarist and the bassist move "
                 "with big live-show energy at their own spots, and the drummer plays hard, seated at his kit."),
          loc_clause=STAGE_MAP2,
          shots=[
              (0.0, CONT_OPEN + "The first frame is already a medium shot from the side of the platform in {loc}: {drummer}, the band's only drummer, seated behind his dark red drum kit with one drumstick in each hand, thunders around the toms and the brass cymbals in time with the drums in <Audio 1>, his arms swinging in huge strokes, his lips pressed together, the camera pushing in fast."),
              (2.4, C + "{drummer} still pounding at his kit, the camera cuts to a wide low shot from the marble floor in front of the platform, pushing in toward the singer with " + ON_PLAT + " under the glass dome, exactly four people on the stage and one drum kit: {guitarist} on the left of the frame and {bassist} on the right of the frame play hard at their own spots, and {drummer} plays on, seated at the back centre; {singer} at the front centre rocks with the beat, " + MIC_NG + " at her lips in her right hand, her lips closed. At about five seconds the band lands the final crashing hit together and keeps playing through it: {guitarist} sweeps his pick hand down across the strings in one huge final chord and holds it with his fretting hand on the neck, {bassist} plays a last deep note with her fingers on the strings and her head thrown back, {drummer} brings both sticks down together on the cymbals and the snare in a final crash, seated at his kit, and {singer} flings her left arm straight up above her head with her fist open, " + MIC2 + " at her lips in her right hand, her chin high; all four hold their places with their instruments still sounding, breathing hard, as the last chord rings out."),
          ]),
    solo(30, "outside", "Beyond the open iron gates she walks away alone down the misty lane into the sunrise.",
         style="Her walk is slow, light and unhurried.", light=L_DAWN, audio="inst",
         shots=[
             (0.0, "The first frame is already a full-body shot from behind her, the camera holding still at the open wrought-iron gates of {loc}: {singer} walks away from the camera down the middle of the misty lane beyond the gates, seen from behind, her long hair and the skirt of her gown swaying, fallen autumn leaves on the flagstones, the huge bare oaks arching overhead and the peach and gold sunrise glowing in front of her; she keeps walking steadily away to the last frame."),
         ]),
]

# scene 4 only: the rain streaks on the window glass were being drawn onto her cheek, so this study entry (same
# reference picture) describes the window with cold blue night light instead of rain
LOCATIONS["office_dry"] = {**LOCATIONS["office"],
    "desc": LOCATIONS["office"]["desc"].replace("a tall leaded window streaked with rain",
                                                "a tall leaded window glowing with cold blue night light"),
    "retain": LOCATIONS["office"]["retain"].replace("rain-streaked leaded window",
                                                    "leaded window with cold blue night light")}
next(sc for sc in SCENES if sc["n"] == 4)["loc"] = "office_dry"
