# Writing the MiniMax H3 prompts

**The two reference projects.** The user rates the prompts of these two music videos as the ones that work best.
Start every new video from them:

| Project | What it is | Spec | Finished prompts |
|---|---|---|---|
| **DON'T LOOK NOW** (2026-10-23..25, 30 scenes, 3:52) | A narrator singer at a gala: one gag per scene, built from 2x2 storyboard cards; chorus scenes on a stage with a fixed band map; a 5 s ending after the song | `assets/example_spec_dont_look_now.py` (+ `_ending`, `_single_shot_continuation`) | `assets/example_prompts/dln_*.txt` |
| **PRETTY LITTLE MALFUNCTION** (2026-10-19, 26 scenes, 3:50) | A performance video with no story: the band only on stage, the singer roaming the theatre with a handheld mic, three recurring text-only extras | `assets/example_spec_plm.py` | `assets/example_prompts/plm_*.txt` |

The older FUCK THE STATIC files (`example_spec.py`, `scene_*.txt`) still show the anime party style, spoken intros and
held props. Every rule below came out of a real take; the scene numbers say which.

## Contents
1. [The six sections](#1-the-six-sections)
2. [Pictures and subjects](#2-pictures-and-subjects)
3. [Character definitions](#3-character-definitions)
4. [Locations](#4-locations)
5. [Summary, retention, sound and music](#5-summary-retention-sound-and-music)
6. [The acting line (no style, lighting or camera boilerplate)](#6-the-acting-line)
7. [Shots, cuts, framing and the first frame](#7-shots-cuts-framing-and-the-first-frame)
8. [Lyric cues](#8-lyric-cues)
9. [People who are not singing](#9-people-who-are-not-singing)
10. [Spoken lines](#10-spoken-lines)
11. [Acting](#11-acting)
12. [Motion, gags and thrown objects](#12-motion-gags-and-thrown-objects)
13. [Props, wardrobe and continuity](#13-props-wardrobe-and-continuity)
14. [Stage scenes: the band, instruments and the microphone](#14-stage-scenes)
15. [Crowds, guests and extras](#15-crowds-guests-and-extras)
16. [Storyboard scenes](#16-storyboard-scenes)
17. [Endings and scenes after the song](#17-endings-and-scenes-after-the-song)
18. [What prompts cannot fix](#18-what-prompts-cannot-fix)
19. [Wording traps](#19-wording-traps)
20. [Before every render: read the whole prompt](#20-before-every-render-read-the-whole-prompt)

---

## 1. The six sections

H3 reference-to-video prompts follow MiniMax's official format. build_prompts.py assembles the sections; you write
the scene fields and the shot text.

```
subject_definitions:
<Subject 1> is <character look> from the first character reference.        one paragraph per subject
<Subject N> is <location> from the Nth reference image, featuring .... <location clause>
<Audio 1> is the complete synchronized song and vocal track ... timing reference.

summary:
[reference generation + audio reuse] <one-sentence scene summary> <tail>

retention_analysis:
<Subject 1>: fully_preserved - <what must stay>
<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.

detailed_description:
<the acting line>

[Shot 1] <first shot>

[Shot 2] At 00:03.200, Hard cut, the same moment on the same stage, <second shot>

overall_soundscape:
<Audio 1> remains the sole complete soundtrack ...

non_diegetic_music:
<Audio 1> is reused as the complete song: ...
```

- **Length:** under 7,000 characters (the checker blocks more). Five-card stage scenes run 6,500–6,800, so keep the
  stage descriptions short (DLN's `stage` variants).
- **Rendered as written:** the Builder renders the saved prompt text exactly, so what build_prompts.py writes is
  what H3 reads. Read it (section 20).

## 2. Pictures and subjects

- **Picture order:** characters in cast order, then prop and storyboard cards, then the location, always last.
  Ordinals follow that order.
- **Every picture influences the shot.** A sheet that shows a guitar puts a guitar in the scene. A two-person sheet
  (DLN's drummer + pianist card) can put either person anywhere. Choose cards per scene: only who is on screen.
- **At most 9 pictures;** five band cards plus the stage is the most DLN used.
- **Storyboard cards** (section 16): `reference_type: "other"`, picture order singer, storyboard, location.
- **Never crop or edit the user's reference sheets or storyboards.** Show the frames and ask for a corrected image;
  the user makes one in minutes (DLN scene 17: a smaller necklace).

## 3. Character definitions

- **Paste the same look text in every scene.** Paraphrasing lets the identity drift. Hair, face, top, bottom, shoes,
  accessories, in the words the sheet shows.
- **Name wardrobe as what it is on the sheet,** never from a lyric. PLM: her wrist item is a red cuff bracelet; a
  lyric about tying something made a ribbon appear, and the user banned the word.
- **Variants per scene type** (`variants` in the spec): DLN `stage` = the singer with "the rose-gold wireless
  handheld microphone in her hand"; `roam` = no mic. Player variants say "playing his guitar" on stage, and "with his
  guitar" in a scene where they rest (DLN scene 30). A description is part of every shot it is in: "playing" in it
  makes them play.
- **Describe only what is in view** (section 19): an off-screen detail is drawn anyway. Boots and trousers in a
  waist-up scene, or a voice-only character's look, pull a wider or different shot.
- **Recurring text-only extras** get one fixed description each, a constant pasted verbatim every time (PLM `CLIP`,
  `DOWAGER`, `WAITER`; DLN `WIFE`, `HUSBAND`, `GRAN`, `WAITER`, `PEARL_SHORT`, `POSER`). The short form ("the man in
  the pink polo") renders a different person.

## 4. Locations

- **Treat the plate as a 3D space:** say where each person is inside it ("at the banquet table", "near the tall
  carved wooden doors", "on the front edge of the stage"), never "in front of the background".
- **Location clause per scene** (`loc_clause`):
  - roaming scenes: "This reference defines the environment, lighting, and atmosphere, while the camera uses its own
    area of the space and places the characters in new positions within it."
  - storyboard scenes: "...while the framing, camera angles and positions of the people in every shot follow the
    storyboard key frames exactly."
  - stage scenes: a **fixed stage map with exact counts** (section 14).
- **The clause lists people only if they are in the scene.** DLN scene 1 cast only the drummer and pianist, but the
  stage clause listed "five backing singers, guitarist, bassist and the singer", so H3 opened on an invented band in
  black. Scene 2 invented backing singers and a bassist the same way. Write a scene's own clause naming only who is
  there ("Only two people are on the stage: ...").
- **The plate's own features act.** The foyer plate had a grand staircase with people on it; with a walking verb in
  the summary, the singer started up on the stairs and walked down (DLN scene 3). Keep walking verbs out when she
  should simply be there, and give her a named spot ("near the tall carved wooden doors").
- **Plates that show people** (an audience, a band) can bring them into the shot. Stage plates with an audience:
  the audience appears only in a pull-back at the end of a shot (section 14).

## 5. Summary, retention, sound and music

- **Summary:** one plain sentence of what happens. **Its verbs act like a shot:** "the singer *passes* a couple"
  made every take walk her in (DLN scene 3); "stands still beside a couple" fixed it. "The full band hits the final
  pose" pulled the drummer off his kit (scene 29).
- **Name only who is there.** A summary about "the guests" put guests into a room meant to be empty (PLM).
- **Retention:** one line per subject naming what must stay.
- **Sound and music** per audio kind (`full`, `inst`, `dialogue`) as before; a stage `music_stage` names the live
  instruments. A scene after the song (section 17) overrides `audio_def`, `sound` and `music`: "<Audio 1> is quiet
  room tone; the song has ended and the stage is quiet." / "Silence; the song has ended."
- **Silent scenes:** keep voice words (murmur, whisper, chatter) out of the sound line, or H3 invents speech.

## 6. The acting line

`detailed_description` opens with **one sentence about faces, nothing else.** The user removed everything else
during PLM, and the overlay and fake-text glitches went away with it:

- **No style sentence** ("photorealistic, glossy music video look"): the references carry the look.
- **No lighting text** ("lit with warm spotlights", "glowing"): the plates carry the light.
- **No blanket camera list** ("whip-pans, Dutch tilts, spins"): put a camera move inside the one shot that needs it.

**The sentence sets the faces for the whole scene, so write it per tone:**
- PLM (cheeky performance): "Every face performs with big, playful, cheeky expressions that change from beat to
  beat: brows shooting up, sly sideways glances, wide grins, winks, mock pouts and eyes flaring with glee."
- DLN gag scenes (cute narrator, lively guests): "The singer's face stays charming and cute from beat to beat: soft
  smiles, playful little eyebrow raises, coy sideways glances, small winks and bright giggles. Every guest is lively
  and animated, chatting, laughing, sipping champagne, gesturing and looking around, clearly enjoying the party."
- DLN stage scenes: "The singer's face stays cute and charming: soft smiles, coy glances, small winks. The singer,
  the guitarist and the bassist perform with big live-show energy, moving, grinning and playing to the camera; the
  drummer and the pianist play at their instruments; the camera is always moving."
- quiet scenes: a calm sentence. A loud sentence ("brows slamming, sneers") makes whispered scenes scream.

**The line applies to every face and body it names.** "The whole band performs with big live-show energy, moving"
sent the drummer into the group pose while the kit kept a second one (DLN scene 29). Keep fixed players "at their
instruments" in the line itself.

## 7. Shots, cuts, framing and the first frame

**Count and length.**
- **Scenes:** 4–12 s, never over 12 s (pass 2 runs out of memory). DLN and PLM mostly ran 6.5–11.5 s.
- **Shots:** 2 for 4–6 s, 3–4 for 7–12 s; a single continuous shot when the moment is one move (PLM scene 21's
  push-in, DLN scene 30's crane). Every cut in a vocal gap (shot_lyrics.py).

**Continuity across cuts (PLM).** A multi-shot scene is one moment seen from several angles:
- shot 1 starts with `CONT_OPEN`: "All shots follow one continuous performance, preserving every person's exact
  position, body orientation, posture, and ongoing movement across every camera cut; each new shot continues directly
  from the final physical moment of the previous shot, with only the camera angle and framing changing."
- every later shot starts "continuing directly from the previous shot's final moment, <who is still where>, the
  camera cuts to ...". DLN stage scenes used `C2` = "Hard cut, the same moment on the same stage, ".
- a change of position is an action inside a shot ("she hops down from the top of the piano"), never a new pose
  after a cut.

**Framing the singer.** Medium, waist-up, medium close-up or close-up while she sings; never wider than full body.
Low-angle full-body shots made her face tiny. Wide shots are for entrances, instrumental moments and pull-backs.

**Camera moves that worked** (one per shot, written into the shot): handheld push-in, whip-pan to a player, a
Dutch-tilted low-angle medium shot, the camera circling behind him as he spins, tracking backward in front of her as
she struts, a crane rising from low in front of the stage, a slow pull-back revealing the audience, a camera rising
straight up as they wave at it.

**The first frame.**
- **Describe the opening framing as the first frame itself:** "the first frame is already a close medium shot at
  eye level: {singer} in the left foreground ..." (DLN scene 3). H3 otherwise opens on its own wide shot.
- **Storyboard scenes often still flash 2 wide frames (0.08 s) first** (DLN 17 and 3). The user edits those out; it
  is not worth a re-roll.
- **Open on the scene's main subject,** never "looking past the drummer" (a one-frame drummer flash).
- **Front-facing sheets look into the lens at the start.** For someone who must not look at the camera, plan a
  profile, over-the-shoulder or from-behind shot and give them a target ("glaring at him"). A brief lens glance in a
  comedy is fine.
- **End the last shot on a clear, held state:** a pose, a wink, a frozen reaction.

## 8. Lyric cues

H3 has the real song (`<Audio 1>`); the text tells it which words land in which shot and how to act them.

```
she {sung} from 0.28 s to 3.44 s, <d>[English, gossipy, sly, singing] Gold dress, short <i>fuse</i>, by the foyer <i>door</i></d>, with a sly sideways glance at them.
```

- **`{sung}`** = "sings in sync with the vocals in <Audio 1>". The window is scene-relative, first word start to
  last word end (shot_lyrics.py).
- **The cue holds exactly the words sung in that shot,** official spelling, split at the cut. Backing-vocal
  answers go in brackets inside it: `Don't look <i>now</i> (don't look!)`.
- **Two specific emotions** in the header: [gossipy, sly], [teasing, playful], [belting, euphoric]. Plain [happy] is
  weak.
- **Tags:** `<i>word</i>` on 1–3 words, `<breath>` between lines. Keep `<pause>` and `<softer>` out (spoken aloud).
- **Never put lyrics in quotation marks** anywhere in the prompt; they show up as text.
- **The word-free cue** for one long move or a scene that keeps producing scribbled text: "{singer} sings with passion
  in sync with the vocals in <Audio 1> [close, fond, tender, singing], her brows lifting softly..." (PLM scene 21).
- **No mouth words** for supplied-audio singing; the audio drives the mouth. On held loud notes: "mouth open wide and
  round, held open to the end of the note", and don't end that shot on a grin.
- **Shots without vocals** get no singing verb.

## 9. People who are not singing

- **Close every visible non-singer's lips by name,** with acting: "{guitarist}, {bassist} and {drummer} keep their
  lips pressed together, grinning".
- **Never give a visible face a talking verb while a lyric plays.** DLN scene 3: "the man leans in, murmuring soft
  compliments to her, his lips moving gently" lip-synced the singer's line "He called her darling" in every take,
  even with the vocal removed from the render audio. The fix: "a smooth, charming closed-lip smile ... silent, his
  lips pressed together and his mouth closed and still for the whole shot, only his eyebrows lifting playfully".
- **Remove the lead vocal for that window** when the camera is on someone else during a sung line:
  `vocal_gate.py --work W 3 3.6-5.9` writes `audio/gated/scene_03.wav`; renders use it automatically and the final
  keeps the real song. Use it together with the silent mouth: the gate alone did not stop the talking verb.
- **Give background faces a target, never the camera.** "she faces the camera with her lips pressed together" made
  the woman behind the singer stare into the lens all shot (DLN scene 3). "stands turned toward the man in profile,
  glaring at him with narrowed eyes, her eyes on him the whole shot" fixed it.
- **Scenes without the singer** render against the vocal-free track (`audio="inst"`).

## 10. Spoken lines

One speaker per shot, framed in profile or toward the listener; window from the track plan plus the line's own first
word offset; match tags to the ElevenLabs take; the listener gets a lips-pressed line.

## 11. Acting

- **Plan the expression per project with the user.** PLM: big cheeky mugging. DLN: the user asked for cute and
  charming, *not* PLM's eye-rolls, mock horror and snarls. Ask; offer a test render comparing two wordings.
- **Concrete faces, not adjectives:** brows shooting up, a sly sideways glance, a coy little wink, a cute uh-oh face
  with eyebrows raised, fingertips pressed to her lips with a sweet little wince that breaks into a giggle.
- **Write emotions as actions,** especially anger: "glaring with narrowed eyes, her brows knotted, her jaw tight".
  No face-colour words (they turn into skin colour or a blush).
- **No tears or wet words on faces, ever.**
- **Guests are always alive:** looking around, smiling, chatting, sipping, laughing. PLM's guests froze because the
  prompts literally said "frozen", "stiffly" and "stiff queue"; never use those words for people.
- **Let the reaction sell the gag:** the singer's giggle, wince or shrug in the last shot.

## 12. Motion, gags and thrown objects

- **One gag per scene, the action as a chain of visible steps ending on a clear state.** DLN gave each gag its own
  scene(s), at most one per scene.
- **Thrown objects fly fast:** "flies fast ... at full speed", plus "Everything moves at normal real-time speed." in
  the tail. "Slow motion" with people moving normally made the glass float.
- **Aim at the catcher's hand or the general direction,** never "straight toward her hair": H3 completes the hit.
- **Breaking things:** a dedicated close-up of the break, then describe only the visible aftermath.
- **No tricks with small objects in hands.** A drummer's stick flip or toss gave him three sticks (DLN scenes 1 and
  10). "one drumstick in each hand, pounds the toms and crashes the cymbal" fixed it. A count-in needs big, readable
  moves: "lifts both drumsticks high above his head, crossed in an X, and clicks them together four times ... his
  arms staying raised high in the air for the whole shot while the drums sit still in front of him". A small "clicks
  his sticks" became normal drumming.
- **Liquid on clothes** (champagne soaking a white shirt) gave blocky pass-2 artifacts; keep liquid on face and hair
  and say "his shirt and tuxedo clean and dry".
- **Drinking and hands near the mouth** stay out of singing windows.

## 13. Props, wardrobe and continuity

- **A prop is named in every shot after it appears, with where it is,** and nowhere it doesn't belong (PLM's crown
  only in scene 3).
- **Size and fit are part of the description.** "a pearl necklace of small pea-sized pearls hanging to her chest"
  became a huge rope she held and shook (DLN 17). "a short single strand of small white pearls that sits snugly
  around the base of her throat" plus a corrected storyboard fixed it.
- **Hands:** say what each hand holds when a prop is near ("a champagne glass in one hand and her other hand at her
  side"). "twisting the strand with her fingers" made her hold the necklace all shot.
- **Text-only objects** (a portrait, a painting) need their look every time.
- **Clothes that change in the story** revert to the sheet unless every later shot restates them.

## 14. Stage scenes

These rules made DLN's stage scenes work after three rounds of user notes.

**The stage map (in the stage `loc_clause`, every stage scene):**
"Fixed positions: the drum kit with the drummer seated behind it in the back-left corner; the ivory upright piano
with the pianist seated at its keyboard in the far back-right corner; the five backing singers in one row across the
back centre. The guitarist, the bassist and the singer move freely across the front of the stage. The stage holds
exactly one drum kit, one piano, one drummer, one pianist, one guitarist, one bassist, five backing singers and the
singer."
- **Exact counts stop clones.** PLM: "{drummer}, the band's only drummer, sits behind his cream drum kit at the
  back of the stage".
- **Name group members one by one when the count matters.** DLN 11 had six backing singers with two duplicates;
  "exactly five women standing in one row, from left to right: the woman with the big curly black afro, the blonde
  with a low bun, the woman with a long black braid, the woman with a black bob and bangs, and the woman with wavy
  auburn shoulder-length hair" fixed it.
- **Fixed players stay at their instruments in every line,** including the final pose: "the drummer seated behind
  his kit lifts one drumstick high". "The whole band ... freezes in a pose" made two drummers (scene 29).

**Framing (user rules, binding):**
- never a distant band shot (faces too small); full-band shots are tight, the band filling the frame;
- the audience only in an occasional pull-back at the end of a shot, seated, seen from behind, facing the stage,
  hands empty, no drinks;
- a tight shot that also has to show the drummer back-left and the pianist far back-right cannot hold eight people
  in front: put the fixed players in the pull-back ("then the camera pulls back slowly, revealing the drummer still
  seated behind his kit ...").

**Energy (PLM style):** every stage scene has its own staged moment and camera move: whip-pans, Dutch tilts, low
angles, orbits, a kneeling guitar solo, leaning in back to back, beckoning the camera, the backing singers covering
their eyes and peeking. The featured member plays to the camera (looks into the lens, winks, grins). Never "standing
there singing".

**Who plays when (stems):**
- only show a player playing where the stem says that instrument plays; cut to whoever leads (stem levels per
  second). DLN scene 2: the guitar stem sat 8 dB under the piano, so a front-and-centre guitarist looked wrong; small
  quiet stabs behind the piano lead fixed it.
- backing singers sing only where the Backing Vocals stem is active (`BS_SING`), otherwise they dance with their lips
  pressed together (`BS_DANCE`).
- mistimed fingers are normal for H3; don't re-roll for them.

**The microphone:** on stage, one rose-gold wireless handheld mic in her hand, never a mic stand. Off stage: none
(DLN), or the same handheld everywhere when the user wants it (PLM `roam`). The lint enforces whichever the spec
sets (`LINT`, `MIC_VARIANTS`).

## 15. Crowds, guests and extras

- **At most 3 described extras,** each clearly different; everyone else soft, out-of-focus, varied ("of every age,
  height and style, with dark, blonde, grey or silver hair").
- **Make the lead unique:** "{singer} the only red-haired person in the room".
- **Generic crowds clone faces.** Check crowd shots for twins.
- **A seated audience:** "the seated audience's backs fill the bottom of the frame, clapping with empty hands, all
  facing the stage". Drinks in hands while clapping looked wrong.

## 16. Storyboard scenes

For any gag that plain text staged wrong (DLN scene 6 failed three times as text), the user makes a 2x2 storyboard
in ChatGPT from a prompt you write (attach the singer sheet and the location plate; no text or numbers on it).
- **Card:** `reference_type: "other"`, picture order singer, storyboard, location; describe the sheet in its
  subject line (DLN `_sb()` builds it).
- **Shot 1** opens "The video opens exactly on the first storyboard panel of {storyboard_sNN}, and every shot
  recreates its storyboard panel's framing, camera angle and poses, filling the whole frame, with hard cuts between
  shots." Every shot: "Hard cut, exactly like the Nth storyboard panel (top left/top right/bottom left/bottom
  right): ...".
- **Location clause:** framing and positions "follow the storyboard key frames exactly".
- **Match the panels exactly,** even where another rule says otherwise (user: "just match the storyboard").
- **Describe the storyboard's extras** in the text with their full look.
- **When a panel itself is wrong** (a necklace too big), ask the user for a corrected storyboard; never edit it.

## 17. Endings and scenes after the song

The song often ends while the band is still playing in the last shot. DLN added scene 30, 5 s after the song, made in
a project copy (`assets/example_spec_dont_look_now_ending.py`, prompt `dln_30_ending_after_the_song.txt`):
- one continuous shot: a close medium shot of the singer and the front line, breathless and beaming, hands raised,
  instruments held still; the camera rises slowly up and back while they tilt their heads back and wave up at it,
  and the drummer and pianist come into view at their instruments, waving too;
- **render audio:** the song plus 5 s of silence (append it, pass it as the render's `audio_path`); the sound lines
  say the song has ended;
- **positive wording only:** "the band rests after the final note", not "the band has stopped playing" (a negative,
  and it puts "playing" in front of the model);
- appended to the film with `splice_scene.py --append`.

## 18. What prompts cannot fix

- **A painted face next to a real face animates.** DLN scene 23: a portrait beside the posing man moved in every
  take, whatever the wording (flat oil painting, brushstrokes, cracked varnish, gilt frame between them). What worked:
  the user made two still images (first and last frame, the painting identical in both); an Image to Video 2 Pass
  render in a project copy (FL2V 8-step LoRA, I2V model), 8 s long and slow; then the user cut a usable chunk, sped
  it up to the shot's length and swapped it in (builder-api.md, I2V first/last frame).
- **The 2 wide frames at the start of storyboard scenes** are cut in the edit.
- **Duplicates H3 adds in a crowd's back rows** are accepted when soft; clones of the main cast are not.

## 19. Wording traps

| Wrote | Got | Write instead |
|---|---|---|
| any no / not / never / without / stopped | the named thing appears | the positive state ("rests after the final note") |
| a stage clause listing the whole band in a scene that casts two | an invented band | a scene clause naming only who is there |
| "the singer passes a couple" (summary) | she walks in, down the plate's stairs | "stands still beside", plus a named spot |
| "both of her feet on the marble floor" | a full-body wide opening | describe nothing below what is in view; "the first frame is already a close medium shot" |
| "she faces the camera" (background extra) | she stares into the lens | "turned toward the man in profile, glaring at him" |
| "murmuring soft compliments, his lips moving" during a lyric | he lip-syncs the singer's line | "silent, lips pressed together, mouth closed and still for the whole shot" (+ vocal_gate.py) |
| "the whole band ... freezes in a pose" / "the whole band performs ... moving" | a second drummer in the group | fixed players "seated at their instruments" by name, in the acting line and the shot |
| "a tightly framed shot of the whole band" plus a back-corner drummer and pianist | people squeezed, cloned | tight on the front line; the fixed players in the pull-back |
| "five backing singers" | six, two duplicates | "exactly five women, from left to right: ..." |
| "twirls a drumstick, flips it up and catches it" | three sticks | "one drumstick in each hand" |
| "clicks his sticks together to count in" (small move) | normal drumming | big raised-arms count-in, "the drums sit still in front of him", no "play" in the acting line |
| a front-and-centre guitarist where the guitar stem is quiet | wrong lead | follow the stems; small quiet stabs behind the lead |
| "pearls hanging to her chest", "twisting the strand" | a huge necklace held all shot | "a short strand that sits snugly around the base of her throat", "her other hand at her side" |
| "playing his guitar" in the description of a resting scene | they play | a scene variant "with his guitar" |
| "slow motion" for a thrown glass | the glass floats | "flies fast at full speed"; normal real-time speed |
| "straight toward her hair" | the glass hits her | aim at the catcher's hand |
| champagne soaking a white shirt | pass-2 blocky artifacts | liquid on face and hair only, "shirt clean and dry" |
| "frozen", "stiffly", "stiff queue" for guests | frozen guests | lively actions |
| style, lighting or blanket camera sentences | overlays, fake text, nothing gained | the acting line only |
| lyrics in quotation marks | text in the video | words only inside the `<d>` cue |
| a ribbon from a lyric | a ribbon in every shot | wardrobe from the sheet |
| "low-angle full-body shot" while she sings | tiny face | medium close-up at her eye level |
| "the posh man in the pink polo" (short form) | a different man | the full fixed description every time |
| tears, wet, glistening, sweat | crying faces | other emotions |

## 20. Before every render: read the whole prompt

**User rule (2026-10-25): read every built prompt in full before it renders, and if anything is wrong, stop (even a
running render) and fix it without asking.** Changing one line is not enough: the bug often sits in a shared constant
(`TIGHT`, `STAGE_STYLE`, `SB_OPEN`, a character description) that the changed line didn't touch.

Read `prompts/scene_NN.txt` top to bottom and check:
- **Checks pass:** build_prompts.py shows no PROBLEM (it also crashes on a missing `{placeholder}` and then leaves
  the OLD prompt file in place, so confirm the file changed). Read every WARN.
- **Every section** (definitions, summary, acting line, shots, sound): nothing that isn't in view, no walking verb
  where she stands, no "whole band" where players are fixed, no "faces the camera" for extras, no talking verbs on
  silent faces, no negative words, no quotation marks, no typos ("a a").
- **Cast and pictures:** only who is on screen; the location clause names only them.
- **Counts:** exact numbers for groups, one of each instrument and player.
- **Singing:** every sung shot has its cue with the right words; silent faces closed by name.
- **First and last frame:** the opening framing stated as the first frame; a held final state.
- **Props:** named where they are, nowhere else; sizes and hands stated.
