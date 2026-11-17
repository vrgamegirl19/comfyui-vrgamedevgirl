# Reviewing takes and fixing them

## 0. Two ways to run (the user picks)

**Review mode** (the user is here and wants to check takes; DON'T LOOK NOW's fix rounds):
1. **Draft** (pass 1, `render_guarded.py --phase draft`). About 6–10 minutes; every attempt's prompt saved as
   `prompts/scene_NN_dK.txt`.
2. **You review the QA sheet, then show the user** and wait for their word before moving on ("do ONLY a draft ...
   let me review the draft before moving on"). If they step out and say keep going, queue more drafts back to back
   (`drive_queue.py --phase draft`); still nothing is finalized.
3. **Finalize only approved drafts** (`--phase finalize`), the scene's latest draft (builder-api.md quirk 11).
4. Side renders in a copy (a single shot to splice, an I2V clip, an extra scene) follow the same loop.

**Unattended mode** (overnight, "run them all", "don't stop and don't ask"; DON'T LOOK NOW's overnight runs):
1. **No draft reviews.** `drive_queue.py --phase both` drafts and finalizes each scene before the next, with memory
   clearing and the out-of-memory ladder. Start it with run_in_background.
2. **When it ends, review every finished take yourself,** write the problem list, and queue one re-roll queue for
   the clear failures within the budget the user set (`--phase both` again).
3. **Stitch the final** and leave `logs/review_*.md` for the morning: what you re-rolled, what you kept, what's on
   the watch list.
4. Never wait on a reply; the GPU must never sit idle until the morning.

**In both modes:** after a user note, propose the fix and wait for their OK unless they told you to go ahead; act
only on problems they confirmed. **When they say "stop", stop the render at once** (kill the queue process, clear
ComfyUI's queue, interrupt) and start over with the fix.

## 1. Review every take, as it lands

- **The QA sheet:** render_queue.py writes `qa/scene_NN_<tag>.jpg` after each take, with 4 frames from the first
  second, 7 across the clip and the last frame. Open it with the Read tool every time.
- **For lip-sync or fast action,** pull a denser strip:
  `ffmpeg -ss 4.0 -t 1.2 -i <video> -vf "fps=10,scale=320:-2,tile=6x2" -frames:v 1 strip.jpg`.
  Tonight that showed the fence fall frame by frame.
- **Zoom on a face** when a speaker's mouth matters:
  `-vf "fps=5,crop=iw*0.5:ih*0.55:iw*0.3:ih*0.05,scale=360:-2,tile=4x2"`.

**Check in this order:**
1. **Wrong objects:** instruments or a mic off stage, props in the wrong scene, a second copy of a prop.
2. **People:** the right cast, nobody extra from the location image, no clones of the main cast in the crowd, and
   text-only extras looking the same as in their other scenes.
3. **Singing:** her face is big while she sings, the mouth moves on the right person, nothing covers her mouth, and
   no drink in her mouth.
4. **Acting:** strong, readable expressions on every face, and anger that reads as anger.
5. **Framing:** the singer never wider than full body, and no up-skirt or underneath angles.
6. **The first second:** no flash of another shot, and no lens stare. A short look at the camera is fine in a comedy.
7. **The end:** a clean final state, ready for the next cut.

Log a one-line verdict per take in `logs/review.txt` (ACCEPTED or REROLL plus the reason) and tell the user in one or
two sentences.

## 2. Re-roll rules

- **The seed is fixed,** so the same prompt renders the same take. **A re-roll needs a prompt change:** framing,
  action, blocking, or acting.
- **Budget: the user sets it per run** (FUCK THE STATIC: at most 2 re-rolls per scene; DLN overnight: at most 1). Then keep the best of the 3
  takes and put the scene on `logs/watch_list.txt` with what's wrong and where the other takes are. The user redoes
  watch-list scenes themselves later.
- **Back up every take before it's replaced:** render_queue.py copies it to `takes/<project>/video_NNNN_takeK.mp4`.
  To switch takes, `POST /projects/{pid}/scenes/{n}/video/select {"source_path"}`, or use the Builder.
- **Fix the prompt for scenes not yet rendered too:** when one take reveals a wording problem, fix every other scene
  that uses that wording before it renders. Tonight "teeth-gritted" and the sunglasses rule were applied ahead.
- **Log every re-roll** with its reason in `logs/reroll_list.txt`. The social kit and the report quote it.
- **Small flaws that aren't worth a re-roll:**
  - soft background lookalikes;
  - a brief hand gesture;
  - a stylistic drift in a no-cast scene;
  - a short glance at the lens in a comedy.
  
  Note them on the watch list and move on.

## 3. Failures seen on FUCK THE STATIC and what fixed them

| Scene | What went wrong | Fix that worked |
|---|---|---|
| 3 | guitarist and bassist carried their instruments at a party | the user made instrument-free sheets (`*_off` cards); never crop the sheets yourself |
| 3 (v1) | flat expressions; close-ups stared into the lens | version 2 prompts: per-shot lyric cues, the facial-acting style sentence, acting for every face |
| 13, 24 | one jacket worn and a second held | a no-jacket character sheet plus a separate jacket card (`outfit`), defined as carried |
| 24 | sat on top of the fence too long | camera on the garden side; struggle, teeter, topple headfirst, drop out of view, empty fence for the last second |
| 22 | singer's lip-sync wrong during the stage dive | that shot uses `{sing} [euphoric, liberated, singing]` without the words; the rest unchanged |
| 25 | sunglasses appeared on stage | props only in their own story world; removed from the stage scenes |
| 25 | one-frame drummer flash at the start | open shot 1 on the lead, not "from the drum riser, looking past the drummer" |
| 14 | posh partygoer turned black-haired | his full look restated in every shot, in every scene he's in |
| 9 | posh guy smiled instead of being angry | anger written as actions (stamps his foot, jabs a finger, scowls); "flushing pink" removed |
| 13 | "face purple with rage" rendered purple skin | "teeth gritted and eyes bulging with rage" |
| 5 | small face while singing; swigging from the bottle during the line | waist-up framing; she slams the bottle down and wipes her mouth before the cue |
| 8 | low angle under her skirt with a tiny face; red-haired clone of the drummer in the crowd | "medium close-up of her face and shoulders, camera at her eye level"; "the only red-haired person in the crowd", the crowd described as dark hair and beanies |
| 21 | rendered before the expression update, so no cues | re-rendered with per-shot cues; every scene rendered before a prompt-style change needs checking |
| 26 | wide composition made her face small while singing | a fresh take framed closer; or accept it for a composition hero shot |
| 20 | bandmates from the location image ran past early | accepted as general panic; use a plate without the band to avoid it |

## 3b. Failures seen on DON'T LOOK NOW and PRETTY LITTLE MALFUNCTION

| Scene | What went wrong | Fix that worked |
|---|---|---|
| DLN 1 | an invented band in black opened the scene | the stage clause listed the whole band; a scene clause naming only the drummer and pianist |
| DLN 1, 10 | stick flip/toss gave three sticks; a small count-in became normal drumming | "one drumstick in each hand"; the big raised-arms count-in; finally the user dropped the drummer shot |
| DLN 2 | invented backing singers and bassist; a front-and-centre guitarist where the guitar is quiet | scene clause with only its cast; piano lead, small quiet guitar stabs (stems) |
| DLN 3 | she started up on the stairs and walked down; the man lip-synced "He called her darling"; the woman behind stared into the lens | "stands still" (not "passes"), the first frame stated as a close medium shot, nothing below the chest; silent closed mouth + vocal_gate.py; "glaring at him in profile" |
| DLN 4 | the soaked man's mouth chewed during the lyric | a napkin pressed over his mouth for the whole close-up |
| DLN 4 | blocky orange artifacts on a soaked white shirt in pass 2 only | liquid on face and hair, "shirt and tuxedo clean and dry" |
| DLN 6 | the glass flew the wrong way / bounced off Gran (3 text tries) | a 2x2 storyboard card anchored "exactly like the Nth panel"; fast throw, aimed at the waiter's hand |
| DLN 6 | the cut from scene 5 re-threw the glass | a single-shot masked continuation in a copy; its first 1.25 s spliced over shot 1 (continuation-alternates.md) |
| DLN 11 | six backing singers, two duplicates | the five named one by one, left to right, in every shot |
| DLN 17 | singer pasted in, raised up; necklace huge and held | close foreground framing; "short strand at the base of her throat", hands stated; a corrected storyboard from the user |
| DLN 23 | the portrait beside the man animated in every take | I2V first/last frame clip from the user's two images, sped up and swapped in by hand (prompt-writing.md 18) |
| DLN 29 | two drummers in the final pose | no "whole band" in the acting line or the shot; tight front line; drummer and pianist revealed seated in the pull-back |
| DLN stage | guests on stage, band moving between shots, drinks while clapping, distant band shots | fixed stage map with exact counts; seated audience from behind with empty hands only in end pull-backs; tight band shots |
| PLM | frozen guests | the prompts said "frozen", "stiffly", "stiff queue"; lively actions instead |
| PLM 9, 10, 18 | drummer clones | "the band's only drummer, sits behind his kit" |
| PLM 15, 17, 21 | translucent overlays, scribbled fake text | no style, lighting or blanket camera sentences; a single continuous push-in; the word-free singing cue |
| PLM 6, 7, 8 | a ribbon appeared from a lyric | wardrobe from the sheet (a red cuff bracelet); lyrics not acted out literally |

**Not worth a re-roll** (DLN): 2 wide frames at the start of a storyboard scene (cut in the edit), a brief glance at
the lens in a comedy, mistimed instrument fingers, soft lookalikes in the back rows.

## 4. Across cuts

- **Cut sheets:** after the batch, `cut_sheet.py` puts the end of scene N beside the start of N+1. Check positions,
  props (who has the sunglasses), wardrobe, who is present, and the extras' looks.
- **Same-room runs** (kitchen 5–9, rooftop 26–28) are the candidates for continuation alternates
  (continuation-alternates.md).

- **Compare the last frame of scene N with the first of N+1** for scenes that should flow (DLN 5 → 6: she threw the
  glass at the end of 5, and 6 opened with her throwing it again). Fix with masked continuation
  (continuation-alternates.md).

## 4b. Edits after the stitch (splice_scene.py)

- **Swap a scene** for an edited clip (a user's hand edit, a spliced shot): `splice_scene.py --work W --film
  FINAL.mp4 --out FINAL_v2.mp4 23=edited.mp4`. Frame-exact, film audio copied, a PSNR check per swap.
- **Append an ending after the song:** `--append ending.mp4` (the audio is padded with silence).
- **Never use the stitch route's `overlays` for this:** each insert adds a frame and desyncs the rest
  (builder-api.md quirk 13). Rename such files `_OUT_OF_SYNC` so nobody uses them.
- **Splice shots inside a scene** with ffmpeg trim/concat at the hard-cut frame (find cuts with
  `select='gt(scene,0.2)',showinfo`), keep the scene's own audio, and check the frame count and audio checksum.
- **Speed up a slow clip** to fit a shot: `setpts=PTS*<slot frames>/<clip frames>,fps=24`, then `-frames:v <slot>`.
- **Mirror a shot** with `hflip` on its frame range only, when the user asks.

## 5. Delivery checks (stitch.py)

- **Duration:** within 0.1 s of the project audio. The Builder's stitch is frame-accurate: 172.25 s, 4134 frames at
  24 fps for a 172.24 s track.
- **Black and frozen stretches:** ffmpeg blackdetect and freezedetect. Expect none.
- **Frame 0 is a real picture.** Never fade the film in from black at the start, unless the user asks.
- **Overview sheet:** one frame every 4 s. Read it to see the whole video at a glance.
- **sync_check.py:** every scene 0 ms. Over 40 ms or a match under 0.35 means a clip's audio is wrong.

## 6. Report and watch list

At the end, write `_claude_work/REPORT.md` (assets/report_template.md):
- the final's path and length, and the checks that passed;
- the re-rolls and their reasons;
- the watch list, worst first;
- the alternates;
- where everything is;
- any API problems found, offered as fixes for the user to approve (this user maintains the node pack).

Keep `_claude_work/HANDOFF.md` (assets/handoff_template.md) current during long runs, so work can continue after a
context reset.
