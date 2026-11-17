# Writing the spec for a band video (the pattern both videos used)

`build_prompts.py` turns `spec.py` into six-section H3 prompts. The format is documented in
`assets/spec_template.py`; the deeper method is in `prompt-writing.md`. This file shows the pattern of the two
reference specs, which are the best starting point: copy
`assets/examples/spec_no_permission_to_exist.py` (band on an open floor, singer + foil character, spoken bridge)
or `assets/examples/spec_no_saint_of_yours.py` (band on a raised stage, singer alone in six rooms), then rewrite
every scene for the new song.

## Layout of the spec

1. **Docstring**: the brief, the idea, the band map, what the stems say (when each instrument enters, backing
   vocal windows, the bridge, the held last words), and the lessons applied. A later session reads this first.
2. **Timing helpers** reading `timeline.json` and `lyrics_timed.json`:
   - `cue(n, a, b, emotions, text)`: the sung words starting in shot window a..b (scene seconds) of scene n, as
     `{sung} from t0 s to t1 s, <d>[English, emotions, singing] text</d>`. It raises an error when no word starts
     there, which catches a wrong window before rendering.
   - `say(n, a, b, emotions, text)`: the same for spoken lines.
3. **Facial-acting lines per scene type**, one sentence each, nothing about style or lighting:
   - `STYLE` (solo, cool attitude), `STYLE_RUN` (solo, high energy), `STYLE_BIG` (full band, every face and how the
     players move at their own spots), `STYLE_BAND` (band without a sung line, lips pressed together).
   - A quiet scene gets a calm line of its own (`style=` on the scene): one style line applies to every face.
4. **`TEXT`**: the audio and music sentences (`audio_inst`, `sound_full`, `music_stage`, `music_band` ...), naming
   the genre, tempo and voice, and for band scenes the instruments as they sound.
5. **`LINT`**: microphones and instruments are forbidden outside the band variants; the `stage` variant must
   mention the microphone. `MIC_VARIANTS = ["stage"]`.
6. **`CHARACTERS`**: per card a `desc` ("the band's guitarist from the {o} character reference, a tall wiry young
   man with a tall scarlet-red mohawk, wearing ..., playing his white electric guitar") and a `retain` line. The
   singer has a `stage` variant holding the microphone; elsewhere "her hands empty and free". Describe wardrobe from
   the sheet, with the same words every time.
7. **`LOCATIONS`**: per card a `desc` naming its features ("featuring ...") and a `retain` list.
8. **Constants** (copy and adapt):

```python
MIC  = "a standard all-black handheld vocal microphone"   # no grille / mesh / head words: they make the head huge
MIC2 = "the same all-black handheld microphone"
STAGE_MAP = ("The band is set up on the open linoleum floor in the middle of the hall, in front of the counter windows. "
             "Fixed positions: the drummer seated behind the black drum kit at the back centre, the lead singer at the "
             "front centre, the guitarist on the left of the frame and the bassist on the right of the frame. The hall "
             "holds exactly one drum kit, one drummer, one guitarist, one bassist and one singer, and every shot faces "
             "the band from the open floor in front of them or from the side.")
ALL4 = "all four band members in view at their own spots"
ROAM = ("This reference defines the environment, lighting, and atmosphere, while the camera uses its own area of the "
        "space and places her in new positions within it. She is the only person in the place.")
CONT_OPEN = ("All shots are one continuous performance: every person keeps their exact position, posture and ongoing "
             "movement across each camera cut, and only the camera angle and framing change. ")
C = "continuing directly from the previous shot's final moment, "
D_PLAYS = ("{drummer}, the band's only drummer, seated behind his black drum kit at the back centre, pounds a dry, "
           "fast beat with one drumstick in each hand")
LIPS3 = "{guitarist}, {bassist} and {drummer} keep their lips pressed together"
```

   For a **raised stage** (NO SAINT OF YOURS) use its `STAGE_MAP2` and
   `ON_PLAT = "all four band members up on the raised wooden platform with the empty marble floor below them"`.
   Stage left / stage right were read as left / right of the frame; say "of the frame".
9. **Scene builders**: `stage(n, summary, shots, cast=BAND, variant="stage", audio="full", style=STYLE_BIG)`,
   `solo(n, loc, summary, shots)`, `duo(n, loc, summary, shots)` (singer + foil). Band scenes always get the full
   `BAND` cast (all four cards), even when a shot shows two people.
10. **`SCENES`**, grouped by song section with a comment line each.

## Shot patterns that landed

Band location, chorus (two shots, the second at the end of the first sung phrase):
```
(0.0, CONT_OPEN + "A wide low shot from the open floor in {loc} with " + ALL4 + ", the camera rushing forward in a fast
 arc from the left toward the singer: " + BAND_PLAYS + "; {singer} at the front centre holds " + MIC + " at her lips in
 her right hand, punches her left fist into the air and " + cue(9, 0.0, 3.0, "belting, defiant, raw", "I need no
 <i>permission</i> to exist") + ". " + LIPS3 + "."),
(3.0, C + "{singer} still at the front centre with the band playing at their spots behind her, the camera cuts to a
 Dutch-tilted medium close-up from slightly below her eye level, pushing in: ..."),
```

Single member (intro, solo, fill):
```
"Hard cut, the same moment in the same hall: a low-angle medium shot with exactly one person in the frame, {guitarist}
 alone, filling the frame from his knees to his red mohawk ..., chopping short, clipped chords on his white electric
 guitar in time with the guitar in <Audio 1>, his lips pressed together, ... as the camera pushes in toward him."
```

Bandmates shouting the refrain (only where the backing stem is active):
```
"{guitarist} on the left and {bassist} on the right lean in toward her from their spots and shout the same two words
 together with her each time, their mouths opening wide on the words and closing between them"
```

Solo journey, energy (camera tracks backward in front of her):
```
"The first frame is already a waist-up medium shot tracking backward in front of her: {singer} marches toward the
 camera along the queue lane between the thick red rope barriers in {loc}, ... jabbing her finger down at the floor
 on the beat as she " + cue(...) + ", her chin up."
```

Prop gag in steps (singer + foil):
```
"she presses the rubber stamp down onto the back of his hand for one beat and then lifts the stamp up and away out of
 the frame, and a round red ink mark is left on the back of his hand; the camera tilts up to his face as he lifts the
 hand and stares at the red mark, his eyebrows shooting up above his round glasses, his lips pressed together."
```

Final hit (everyone keeps playing): see scene 29 of the NO SAINT OF YOURS spec.

Moves that worked in band scenes: fast low glide across the floor toward the singer, orbit from the side to the
front while pushing in, arc along the front of the band, Dutch-tilted push-in, low angle from the floor, side of
the drum kit for fills. Every band scene has at least one shot with all four in view, and that shot moves.

## Checklist before `build_prompts.py --apply`

- `build_prompts.py --work W` shows no PROBLEM; every WARN read.
- Every prompt file in `prompts/` read top to bottom.
- Band scenes: all four cards in the cast; head counts and "only drummer" present; mic wording without a grille.
- Every sung shot has its singer on screen and a cue; every scene without the singer singing is on vocal-free
  audio (`audio="inst"`).
- Solo scenes: no mic, no instruments; the foil's lips closed.
- No rain/wet/tears/torn near faces; no fixed object moving with a walking person; exits early in a shot.
