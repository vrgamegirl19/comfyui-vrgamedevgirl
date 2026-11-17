# NO PERMISSION TO EXIST: report (2026-11-13)

## The film
- **Final:** `ComfyUI\output\No Permission to Exist\NO_PERMISSION_TO_EXIST.mp4`
- 1920x1088, 164.00 s (the song is 164.00 s), 27 scenes.
- Checks: every scene at 0 ms audio offset, no black or frozen stretches, no fade-in at the start.
- `NO_PERMISSION_TO_EXIST_auto.mp4` is the first stitch, before the six retakes below.

## The idea
The band has taken over the waiting hall of a grey 1970s permit office and plays on its open floor. Between band
scenes the singer breaks the building's small rules, with a prim young clerk as her foil: his office (the stamp on
his hand, her boots on his desk, the ripped form and the tie pat), the archive corridor, the entrance lobby (the
queue rope, the turnstile), the lift for the spoken bridge (every button lit), and the rooftop at dusk for the final
chorus. In the last chorus the clerk is in the hall with his tie loose, jumping and conducting the band.

## What was made
- **Characters (Krea 2 two-pass on Reference Builder cards, three-panel sheets on plain grey):** Vex the singer,
  Rook the guitarist, Jinx the bassist, Moz the drummer, and the Clerk. Prompts are in `tools\make_refs.py`,
  every candidate is in `refs\`.
- **Sonnet's panel check:** passed singer, guitarist, drummer and clerk; failed the first bassist sheet (uneven grey
  background), so her other candidate is on the card. Sonnet did not check that replacement.
- **Regenerated before the check:** the first singer sheets had divider lines between the panels, the first
  guitarist looked about 40, and the first drummer had no punk look.
- **Locations (Krea 2):** waiting hall, clerk's office, archive corridor, entrance lobby, lift, rooftop at dusk.
- **Lyrics:** timed with ElevenLabs Scribe (the new key works). The last line was fixed by hand from the vocal stem:
  "No permission" at 142.5 s, "to exist" at 148.9 s.

## Settings (same as NO SAINT OF YOURS)
Reference-to-video, 2 Pass, standard pipeline. Singularity V2, pass 1 25 steps. Pass 2: LoRA
`minimax_h3_ref2v_turbo_8step_v1.1_768p`, 4 steps, denoise 0.30. 2k, no latent continuation, no scene over 8 s.
Scenes 21 and 25 have six reference pictures, so they render with reference size "match".

## What happened
- ComfyUI was not running after the PC restart (23:17); it was started with `Run Video Builder ComfyUI.bat`.
- 41 renders. Every one finished on its first try: no out-of-memory errors, no crashes.

## Retakes (older takes are in `takes\keep`)
| Scene | Why | Result |
|---|---|---|
| 5, 6, 8, 12 | repeated frames, automatic re-roll overnight | 12 fixed (0); 5, 6, 8 unchanged |
| 1 | two guitarists in the second shot | fixed, he is alone |
| 25 | two strangers with guitars behind the singer (my mistake: the guitarist and bassist cards were left out) | fixed |
| 4 | the stamp changed shape and never left the clerk's hand | fixed: pressed, lifted, red mark |
| 5, 6, 8 | second re-roll for repeated frames | 8 better (38 -> 18); 5 and 6 unchanged |

## Watch list
- **Scenes 5 and 6 (archive corridor, 0:22-0:35):** still choppy after three takes each (47 and 29 repeated frames).
  It follows these shots, not the seed, so another re-roll will not fix it; a different shot or frame interpolation
  would.
- **Scene 8 (0:41):** 18 repeated frames. **Scenes 1, 17, 24:** 10 to 12 each.
- From the Opus 5.5 frame review of 22 scenes (`REVIEW_ISSUES.md`: 12 OK, 9 minor, 1 re-render, which was scene 4):
  - **Scene 14:** she grabs the clerk's face with both hands (the prompt asked for two pats on the cheek) and does
    not walk off.
  - **Scene 18:** she never walks out of the lift; it ends on her face pushing into the lens.
  - **Scene 26:** as the camera rises she stays a chest-up figure at the bottom of the frame.
  - Small: scene 10 mohawk cut off by the frame for a second; scene 12 a white globe floats behind the drummer;
    scene 21 the bass is orange for the first second; scene 22 the paper scraps look black against the sunset;
    scene 23 gibberish text on a wall sign; scene 24 the singer looks older in the opening close-up.
- Scenes 1, 5, 6, 8 and 25 were re-rendered while the Opus review ran, so I checked those myself, not Opus.

## Update (2026-11-13, morning): scenes 5 and 6 smoothed with RIFE
- **Newest film:** `NO_PERMISSION_TO_EXIST_2.mp4` (164.00 s, 0 ms sync). The first final is kept.
- A reworded re-render of scene 5 was rejected (the drawer slid along with her), so the earlier takes of 5 and 6
  were kept and smoothed instead: every second frame dropped (12 fps), then interpolated back to 24 fps with the
  RIFE node (`FL_RIFE`, rife49), with the real frame put back at each cut. Repeated frames: 47 -> 0 and 29 -> 0.
  Audio is copied unchanged. Tool: `tools\rife_fix.py IN.mp4 OUT.mp4`. The unsmoothed takes are in `takes\keep`.
- The same tool can smooth scene 8 (18 repeated frames) and scenes 1, 17, 24 (10 to 12).

## Clean across the film
One singer, guitarist, bassist, drummer and drum kit in every band shot of the final; a normal-sized microphone in
the hall and none anywhere else; no wide shot that stays wide; no distorted faces or hands.

## To change a scene
Edit its prompt in `spec.py`, run `build_prompts.py --work . --apply N`, then `tools\retake_run.py N`, then
`stitch.py --work . --prefix NO_PERMISSION_TO_EXIST`. `HANDOFF.md` has the rest.
