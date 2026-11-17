# NO SAINT OF YOURS: report (2026-11-12)

## The film
- **Final:** `ComfyUI\output\No Saint of Yours\NO_SAINT_OF_YOURS.mp4`
- 1920x1088, 179.60 s (the song is 179.60 s), 30 scenes.
- Checks: every scene at 0 ms audio offset, no black or frozen stretches, no fade-in at the start.
- `NO_SAINT_OF_YOURS_v1.mp4` (same as `_auto.mp4`) is the first stitch, before the five smoothness retakes.

## Settings every scene was rendered with
- Reference-to-video, 2 Pass, standard pipeline. Each scene ran pass 1 and then pass 2; no drafts were reviewed.
- Model `Minimax-h3_Singularity_ref2va_v2_int8`. Pass 1: 25 steps.
- Pass 2: LoRA `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16` at 1.0, 4 steps, denoise 0.30. No DMAD, no
  other LoRAs.
- **To confirm:** you wrote "8 step 768 V1.3". No 8-step v1.3 is installed (only `fl2v_turbo_4step_v1.3_768p`), so
  the 8-step ref2v v1.1 was used.
- No latent continuation. No scene is over 8 s, so feedforward stayed off. Reference image size stayed at max.

## What happened overnight
- ComfyUI was not running after the PC restart; it was started with `Run Video Builder ComfyUI.bat`.
- The ElevenLabs key in the folder was rejected (HTTP 401, invalid key). Lyrics were timed locally with Stable-ts
  large-v3 on the lead-vocal stem instead, and held notes were fixed from the stem level.
- After your settings change at 00:40 everything was restarted from scratch with the new pass 2.
- 40 renders in total. Every one finished on its first try: no out-of-memory errors, no crashes, no stuck jobs.

## Retakes (first takes are in `_claude_work\takes\keep`)
| Scene | Why | Result |
|---|---|---|
| 8 | bug-eyed, bared-teeth face | calm face, kept take 2 |
| 20 | two singers in frame | one singer, kept take 2 |
| 23 | two drummers | one drummer, kept take 2 |
| 1, 7, 15, 25, 27 | repeated frames (choppy motion) | all smoother, kept take 2 |

## Choppy motion (repeated frames)
Pass 1 sometimes repeats every 2nd or every 4th frame. It is already in the pass-1 draft, so only a new draft
fixes it. The five worst scenes were re-rolled. Still present, milder (counts are repeated frames per take):
scene 11 (41 of 170, slow shot), 15 (8), 25 (13), 4 (21), 6 (18), 21 (13), 24 (13), 22 (13), 30 (10), 2 (8),
9, 10, 26, 28 (6 each). Check with `tools\dup_runs.py <clip>`.

## Watch list (from the Opus 5.5 frame review; full detail in `REVIEW_ISSUES.md`)
Nothing below was re-rendered; these are for you to decide. The review covered 25 scenes (not the five that were
being re-rolled; I checked those myself).

Re-render suggested:
1. **Scene 26 (2:34):** the singer stands on the marble floor in front of the stage while the others play on the
   platform; her mouth is still wide open on the last frame.
2. **Scene 19 (1:55):** singer, guitarist and bassist on the floor in front of the platform; only the drummer on it.
3. **Scene 14 (1:23):** the same, plainest in the wide crane shot.

Borderline:
- **Scene 17:** the guitarist solos in front of an empty platform (no drum kit) while the drums play.
- **Scene 12:** she starts already standing, a jump from scene 11 where she sits at the vanity.
- **Scene 4:** she stands still at the fireplace; the walk along the desk is missing.

Recurring: the band stands beside or in front of the platform instead of on it (scenes 3, 9, 14, 17, 18, 19, 23,
29). Only scenes 8 and 16 have all four on the platform.

Smaller notes: scene 3, the mic is only visible in the close-up; scene 6 and scene 20 end on a broad smile;
scene 13 is a pained tight close-up; scene 22 opens full-body while she sings; scene 29 has a large mic head;
scene 9 has a sung line with no singer on screen (my plan, not the render).

Clean across the film: no distorted faces or hands, no duplicate people left, the black mic is always in her hand
on stage and never off it, and all four characters stay consistent.

## Choices I made
- "Stage left / stage right" was read as left / right of the frame.
- Scenes without the singer (1, 2, 9, 17, 18, 30) were rendered against the vocal-free stems.
- The skill's rule for matching reference size was changed from "over 4 pictures" to "over 5".

## Update (afternoon, 2026-11-12): stage fix
- **Newest film:** `NO_SAINT_OF_YOURS_5.mp4` (scene 23 re-rolled again: the wide shot had two drummers; scene 29 ending
  now has everyone still playing through the last chord). Previous: `NO_SAINT_OF_YOURS_4.mp4` (179.60 s, 0 ms sync, no black or frozen stretches). Older stitches
  (`_v1`, `NO_SAINT_OF_YOURS.mp4`, `_2`, `_3`) are kept.
- **Scene 29:** the mic head was huge; fixed by removing the "black metal mesh grille" wording for that scene
  (variant `stage_ng`).
- **Stage scenes redone** with the full band up on the platform, opening wide (scene 2: medium) and then moving in:
  2, 3, 9, 10, 14, 15, 17, 18, 19, 23, 26, 29. Scenes 1, 8, 16 and 25 were left as they were. The old takes are in
  `takes\keep` (`*_before_stagefix.mp4`, `scene02_stagefix_wide_big_mic.mp4`, scene 29 takes 1-3).
- **Open:** scene 26's close-up mic head looks large (same grille wording); about 10 takes still have mild repeated
  frames; the wide shots in 14, 23 and 29 are fairly long (you said they are fine).

## To change a scene
Edit its prompt in `spec.py`, run `build_prompts.py --work . --apply N`, render N, then
`stitch.py --work . --prefix NO_SAINT_OF_YOURS`. `HANDOFF.md` has the rest.
