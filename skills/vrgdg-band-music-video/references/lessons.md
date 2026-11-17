# Lessons from NO SAINT OF YOURS and NO PERMISSION TO EXIST

Both videos were made unattended overnight (2026-11-12 and 2026-11-13), then fixed with the user the next day. The
user called both "really good". This file holds everything that made them work, every failure and its fix, and
the things not to do again. Read it before planning, and again before writing prompts.

Contents: 1 Results · 2 Settings · 3 Audio and word timing · 4 Reference pictures · 5 Idea and scene plan ·
6 Prompt rules that fixed real failures · 7 Rendering unattended · 8 Repeated frames and RIFE · 9 Review ·
10 Delivery · 11 Fixes the user asked for afterwards · 12 Do-not list

---

## 1. Results (what "good" looked like)

| | NO SAINT OF YOURS | NO PERMISSION TO EXIST |
|---|---|---|
| Song | elegant gothic rock, 179.6 s | British post-punk, 162 BPM, 164.0 s |
| Scenes | 30 (2-8 s) | 27 (2-8 s) |
| Look | candle-lit gothic manor; band on a conservatory stage; singer alone through six rooms | grey 1970s permit office; band on the open floor of the waiting hall; singer breaking the building's rules with a prim clerk as her foil |
| Refs | user-supplied band sheets and locations | all made by Claude with Krea 2 (5 characters, 6 locations) |
| Word timing | ElevenLabs key rejected (401) -> local Stable-ts large-v3 on the lead stem | ElevenLabs Scribe on the lead stem |
| Renders | 40, every one finished first try | 41, every one finished first try |
| Time | about 14 min per scene (draft + finalize) on an RTX 5090, 2-8 s scenes; 27 scenes + retakes ran 23:20-07:25 | same |

The shape both used: **a band performance in one location** (choruses, instrumental breaks, the intro and
ending) alternating with **the singer's solo journey through four to six other locations** (verses,
pre-choruses, bridge). The second video added **one extra character as a foil** (the clerk), which made the solo
scenes funnier and gave a payoff (in the last chorus he is in the hall, tie loose, conducting the band).

## 2. Settings (the user's, used for both)

- Reference-to-video, **2 Pass**, standard pipeline, no DMAD.
- Model `Minimax-h3_Singularity_ref2va_v2_int8`; pass 1: 25 steps, res_multistep / simple, denoise 1.
- Pass 2: LoRA `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16` at 1.0, 4 steps, denoise 0.30.
- Resolution 2k. **No latent continuation.** Scenes 2-8 s. Feedforward only for scenes over 8 s (none were).
- `ref_image_size` max; a scene with **6 or more pictures** renders at `match` (the Builder now does this itself).
- Every scene runs both passes (draft, then finalize from the saved latent). Drafts are **not** reviewed in an
  unattended run.
- Out of memory: clear RAM/VRAM, retry 3 times, then `ref_image_size: match` for that scene (night_run.py).
- The user once wrote "8 step 768 V1.3"; only the 8-step **v1.1** ref2v LoRA was installed and it was used. Ask if
  a new video names a LoRA that is not installed (check_inputs.py shows what is there).
- `config_example.json` holds all of it.

## 3. Audio and word timing

- **Key or no key.** If the user gives an ElevenLabs key, test it first (`elevenlabs.py --work W test`). Use
  Scribe on the lead-vocal stem (`elevenlabs.py stt`). If there is no key, or it is rejected, use the Video
  Builder: `builder_transcribe.py` splits stems with the Builder's Demucs (`POST audio/stems`) when the user gave
  none, then times the lyrics with the Builder's Stable-ts / Whisper large-v3 (`POST lyrics/align`), and saves the
  words in Scribe's format. `align_lyrics.py` then works the same either way. Never stop the video because a key is
  missing.
- **Time on the lead-vocal stem, not the mix.** The user's own stems are cleaner than a Demucs split; use them when
  given (`lead_stem_path`).
- **The end of the song needs a hand check.** In both videos the aligner squeezed the last, long held words
  ("No permission ... to exist": the real times were 142.6-144.1 s and 148.9-150.5 s). Compare the last lines and
  every held note against `vocal_activity.py <lead stem> --detail A B`, and fix `lyrics_timed.json` by hand.
- **Stems decide who plays** (`stem_scenes.py`). The intro of NO PERMISSION TO EXIST had bass and guitar alone,
  then drums: scene 1 showed only the bassist and guitarist, scene 2 the drummer crashing in on his cue. Backing
  vocals only where the backing stem is active (the guitarist and bassist shout "No permission" with her there).
- **Vocal-free audio** for every scene where the singer is not on screen or not singing (intro, instrumental
  breaks, band-only shots), so no face lip-syncs a voice that is not theirs. `build_track.py` writes
  `full_track_no_vocals.wav`; with the user's stems, mix every stem except the lead vocal (pad 1 s).
- **A spoken bridge** is cued as speech (`say()` in the spec: "speaks in sync with the voice in <Audio 1>",
  `[English, deadpan, speaking]`), not singing.

## 4. Reference pictures (Krea 2 two-pass, through the API)

- Route: `POST /projects/{pid}/references/{subjects|locations}/{id}/image/generate` with `prompt` and
  `settings: {seed, seed_mode: "fixed"}`. The job result has `saved_path`. `make_refs.py gen ID SEED...` does it
  and copies each candidate to `refs/`; `make_refs.py pick ID SEED` puts the chosen one on the card.
- Generate **2-3 seeds per card** and pick. Use seeds far apart (211, 212, 221, 231...).
- **Three-panel sheet wording that works** (in make_refs.py `SHEET`): "one single continuous image ... the same
  person three times, side by side ... On the left third: an extreme close-up of the face only ... In the middle
  third: an upper-body view ... On the right third: a full-body view from head to shoes ... no borders, no frames,
  no dividing lines and no text." The older "LEFT PHOTO / CENTER PHOTO / RIGHT PHOTO" wording drew thin divider
  lines between the panels.
- **Krea ages people.** "In his mid-20s" gave a guitarist of about 40. "A handsome young man of 24 with a fresh,
  smooth youthful face and bright eyes" and "a beautiful young woman of 25 with a soft heart-shaped face" worked.
  The first singer came out hard-faced until "soft heart-shaped face, full cheeks, big eyes, delicate features"
  was added.
- **Spell out the genre look item by item.** "Punk drummer" alone gave a plain man; naming the buzz cut, braces,
  splattered jeans and oxblood boots gave a punk.
- **Empty hands on the sheet**, plain garments (no logos, so no fake text), solid mid-grey backdrop.
- **Sonnet checks every character sheet** (the user asks for this): an Agent with `model: "sonnet"` reads each
  candidate and answers per sheet: panel order face close-up / upper body / full body; solid grey with no
  borders, lines or text; the same person, hair and outfit in all three; looks the right age; hands empty; no
  extra people. Re-check any replacement too (the second bassist sheet was used without a re-check; check it next
  time).
- **Locations:** "A clear cinematic environment reference photograph of ... Wide eye-level framing that shows the
  layout of the whole space, photorealistic, empty and still, a place with nobody in it." Name every feature a
  scene will use (the stamp rack, the turnstiles, the button panel). The band's location needs **a wide open floor
  area** for the band.
- **A flat floor for the band is easier than a raised stage.** In NO SAINT OF YOURS, H3 often put the band on the
  floor beside or in front of the low platform; it needed the stronger STAGE_MAP2 / ON_PLAT wording ("up on the
  raised platform, all four on its boards a step above the marble floor ... with the empty marble floor below
  them") and 12 re-renders. NO PERMISSION TO EXIST put the band on the open floor of the hall and never had the
  problem.
- When the user supplies the pictures instead: never crop or edit them; build cards with `setup_refs.py`
  (`refs.json`). If a sheet leaks an instrument or a second outfit, show the user and ask for a fixed sheet.

## 5. Idea and scene plan

- **Pick one strong, simple idea that the singer can act out with her hands free**, with the band location as
  home base. NO PERMISSION TO EXIST: the band has taken over a permit office; between band scenes she breaks its
  rules (stamps the clerk's hand, boots on his desk, rips a form, unhooks the queue rope, hips through a turnstile,
  lights every lift button). One gag per scene, each a physical action the camera can see.
- **Fun and energy** (the user asked for more in the second video): moving camera in every scene, the singer
  physically active (striding, running, marching toward a backward-tracking camera), bandmates leaning in to shout
  the refrain, a foil character who reacts.
- **Lyrics set the mood, not the action.** Don't act lyrics out literally.
- **Use every location**, spread across the song; return to the band for every chorus.
- **Scene lengths 2-8 s, cut in vocal gaps, never mid-word** (`setup_timeline.py --dry-run` flags boundaries that
  cut a word). Most scenes are two shots: the second starts where the first sung phrase ends.
- **An ending that plays through:** the last scene lands the final hit with everyone still playing.
- Write the plan into `timeline.json` (start, end, label) and the idea + band map into the spec's docstring.

## 6. Prompt rules that fixed real failures

Each rule below came from a failed take. They are built into the two example specs.

| Failure | Fix (wording) |
|---|---|
| Two strangers with guitars behind the singer (NPtE 25) | **Every band member's card is in the cast of every band scene**, even when they are only in the background. A missing card makes H3 invent a stranger. |
| Two guitarists / a cloned bandmate (NPtE 1, NSoY 20) | A single-member shot says "**exactly one person in the frame, {guitarist} alone**". |
| Two drummers (NSoY 23, twice) | "**{drummer}, the band's only drummer**" every time; the stage map says "**exactly one drum kit, one drummer, one guitarist, one bassist and one singer**"; wide shots add "exactly four people on the stage and one drum kit". |
| Only some members on the stage; band beside the platform (NSoY, 12 scenes) | A fixed **stage map** with each member's spot; every wide shot names "all four band members in view at their own spots"; for a raised stage use the ON_PLAT wording (section 4). |
| Long static wide shots (user: "we never want the whole shot to be a long wide shot") | **Start wide and move in at once** (push-in, orbit, arc, fast glide), or cut to a medium/close shot. Opening a scene on a medium shot is fine too (NSoY 2). |
| Huge microphone head (NSoY 29, 26) | Mic wording "**a standard all-black handheld vocal microphone**", later "**the same all-black handheld microphone**". **No grille / mesh / head words**: "with a black metal mesh grille" made the head huge. Microphone only in band-location scenes (`stage` variant); the lint forbids mics and instruments anywhere else. |
| Band lifts arms and stops before the last note ends (NSoY 29) | The last hit **keeps playing through**: "the band lands the final crashing hit together and keeps playing through it ... all four hold their places with their instruments still sounding as the last chord rings out". |
| Streaky lines on her face (NSoY 4) | No rain, wet or streak words near a face (a "rain-streaked window" behind her put streaks on her skin). Also never "tears/torn" (wet-word lint): write "rips/ripped". |
| Bug-eyed face (NSoY 8) | Never "eyes flashing wide"; use "sharp steady eyes", "one brow lifting". |
| The stamp changed shape and never left his hand (NPtE 4) | **Prop gags as separate visible steps**: "presses the rubber stamp down onto the back of his hand for one beat, then lifts the stamp up and away out of the frame, and a round red ink mark is left". Name the prop the same way each time ("the wooden-handled rubber stamp"). |
| The drawer slid along with her (NPtE 5 reword, rejected by the user: "looks dumb ... not realistic") | Fixed furniture stays fixed: if she walks, she does not pull a drawer at the same time. Write "she stops at the cabinet, pulls the drawer open, then walks on". |
| She grabbed his face instead of two pats (NPtE 14); she never left the lift (NPtE 18) | Touch gestures on another face come out bigger than written; keep them light ("taps his cheek once with two fingers") or leave them out. Exits written at the end of a shot often don't happen; put the exit early in the shot, or end the scene on her in frame. |
| Singer left at the bottom of the frame as the camera rose (NPtE 26) | Pair any crane-up with the framing: "the camera rises while keeping her framed from the waist up in the centre". |
| Choppy repeated frames that follow a shot, not a seed (NPtE 5, 6) | See section 8. |

Standing prompt rules (from earlier projects, applied in both videos):
- **Positive-only wording**, no negatives. Describe only what is in view.
- **No style, lighting or blanket camera boilerplate.** The references carry the look. One facial-acting sentence
  per scene type (the spec's `STYLE`, `STYLE_RUN`, `STYLE_BIG`, `STYLE_BAND`).
- **State the first frame**: "The first frame is already a medium close-up at her eye level: ...".
- **Continuity across cuts**: open with `CONT_OPEN` ("All shots are one continuous performance: every person keeps
  their exact position ..."), and start each later shot with `C` ("continuing directly from the previous shot's
  final moment, ...") plus where people still are.
- **Lyric cues** from the word times, never quoted: `cue(n, a, b, emotions, text)` builds
  `{sung} from 0.00 s to 6.40 s, <d>[English, low, hushed, dangerous, singing] The rule is a <i>whisper</i> ...</d>`.
  Two or three emotion words, `<i>` on stressed words.
- **Silent people**: "lips pressed together" for every bandmate in every shot (`LIPS3`), and for the foil
  (`CLERK_QUIET`). The singer has closed lips in shots with no sung words.
- **Players play in time with their stem**: "slashes clipped chords on his white electric guitar in time with the
  guitar in <Audio 1>". Mistimed fingers are normal for H3; don't re-roll for them.
- **Medium and close shots while she sings**; never wider than full body for the singer.
- **Read every built prompt top to bottom before rendering.** Shared constants carry bugs into every scene (the
  mic grille wording did).
- Prompt length under the lint's 7000 characters: shorten character descriptions, not the action.

## 7. Rendering unattended

- `night_run.py --work W parent <scenes in order>` started with the Bash tool's `run_in_background` (never a bare
  `&`), then a Monitor on `logs/night.log`. It drafts and finalizes each scene, clears memory before and after
  every render, retries out-of-memory errors, re-rolls a choppy take once, and stitches at the end.
- `--recover` lets it restart ComfyUI with the user's launcher when it stops answering. Use it only when the user
  allowed it (NO SAINT OF YOURS rules: "always check ComfyUI every few minutes to make sure it did not crash or get
  stuck"). Without it, the runner waits for ComfyUI instead.
- After a PC restart ComfyUI may not be running. Ask the user to start it, or with their permission start it
  with their launcher, before the run.
- Render order: scenes that test the hardest things first (a full-band wide, a single-member shot, a gag with a
  prop), then the rest in timeline order.
- Retakes while it runs: append scene numbers to `logs/extra_jobs.txt`. Forced retakes later:
  `night_run.py --work W retake 5 6` (copy the current take to `takes/keep/` first).
- Keep every old take before re-rendering (`takes/keep/sceneNN_<why>.mp4`).

## 8. Repeated frames and RIFE

- Singularity V2 pass 1 sometimes repeats every 2nd or 4th frame (17 of 30 takes in NO SAINT OF YOURS). It reads
  as choppy motion and is invisible on still sheets. `held_frames.py` counts them: 0-6 clean, 7-11 mild, 12+
  choppy.
- night_run.py re-rolls a take with 12+ once and keeps the smoother of the two.
- **If a scene stays choppy across 2-3 seeds, the shot is the cause, not the seed.** Don't reword the action to fix
  it: the reworded scene 5 was rejected by the user. Smooth the existing take instead (the user's idea):
  `rife_fix.py IN.mp4 OUT.mp4` keeps every 2nd frame (12 fps) and interpolates back to 24 fps with the `FL_RIFE`
  node (rife49), puts the real frame back at each hard cut (RIFE blends the two shots there), keeps the frame count
  and copies the audio. Then `POST /scenes/{n}/video/select {"source_path": OUT}` and re-stitch. Scenes 5 and 6:
  47 -> 0 and 29 -> 0 repeated frames, still 0 ms sync. Check that the RIFE node exists first
  (`GET :8188/object_info/FL_RIFE`).

## 9. Review

- Review **every finished take at zoom**. Small review frames hid two drummers in NSoY 23. Count the people and
  drum kits in every band shot; check the mic size, everyone at their spot (and on the platform), who is singing,
  the foil's lips closed, the first frames, and the last frames of each scene against the next scene's first
  (`cut_sheet.py`).
- `held_frames.py` on every take.
- Then an **Opus frame review of every scene** (the user asked for this in NO SAINT OF YOURS: "pass the task to
  Claude Opus 5.5 to review frames of each scene to make sure they line up with the prompt and create a summary of
  issues"). Use an Agent with `model: "opus"`, giving it each scene's prompt file and frame sheets. It writes
  `REVIEW_ISSUES.md` (OK / minor / re-render per scene). Check yourself any scene re-rendered while it ran.
- Unattended: re-roll clear failures (wrong head count, strangers, a broken gag) within the budget, then put the
  rest on the watch list. With the user present: list the issues, propose fixes and **wait for their OK**.

## 10. Delivery

- `stitch.py --work W --prefix TITLE`. The checks must show duration = song, 0 ms sync in every scene, no black
  or frozen stretches, and no fade-in at the start.
- **Never overwrite a film**: number new stitches `_2`, `_3`, ... and keep the first stitch (`_auto`).
- `REPORT.md` (assets/report_template.md): the final's path and length, the settings, what happened overnight,
  a retake table, the watch list worst first, what is clean across the film, how to change a scene.
- `HANDOFF.md` written before the run starts, so another session (or Opus) can take over.

## 11. Fixes the user asked for afterwards (expect these)

- The full band in their spots in every band shot, never a long wide shot (NSoY: 12 scenes).
- A medium opening instead of a wide (NSoY 2).
- The mic looked huge (NSoY 29, 26): the user found the fix was the wording ("its the mic verbiage"), not the camera.
- A doubled drummer (NSoY 23).
- The band stopping before the last note (NSoY 29).
- Lines on her face that got worse when upscaled (NSoY 4: the rain wording, plus a new seed).
- Choppy scenes (NPtE 5, 6): RIFE, not a reworded re-render.
- When the user names a problem, look at the frames for it before theorising (the mic: "go look at the frames.. it
  was huge the whole time"). When they reject a theory, drop it.
- They may ask for film-look filter tests (grain, softening) on copies in a new folder for them to pick. Never
  apply a look to the final unasked.

## 12. Do-not list

- Don't edit `vrgdg_builder_session.json`; only the API writes project files.
- Don't change, load or unload the LM Studio model (this workflow doesn't need the LLM at all).
- Don't print or log the ElevenLabs key. Don't send the user's name, email or files to outside services (beyond the
  ElevenLabs transcription they asked for).
- Don't crop or edit the user's reference pictures.
- Don't queue renders the user didn't ask for; one GPU job at a time.
- Don't review drafts in an unattended run, and don't wait for an answer when the user is away.
- Don't re-roll a scene more than the budget; don't reword a scene the user accepted.
- Don't fade the film in from black at the start.
- Don't describe the mic's grille or head; don't write rain, wet, tears or torn near faces; don't write "eyes
  flashing wide".
- Don't leave a band member's card out of a band scene; don't let a wide shot stay wide.
- Don't make a fixed object move with a walking person.
- Don't trust small review frames: zoom and count.
