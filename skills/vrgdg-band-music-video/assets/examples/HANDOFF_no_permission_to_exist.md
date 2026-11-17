# NO PERMISSION TO EXIST: handoff (started 2026-11-12 23:20, unattended overnight run)

## What this is
Music video for the user's song "No Permission to Exist" (British post-punk, 164.0 s, 162 BPM). Inputs:
`<song folder>` (rules.txt, lyrics, song style, song, 8 stems, ElevenLabs key).
Copies are in `source/`. Builder project `No Permission to Exist`, work folder `_claude_work/`.

## User rules (rules.txt)
- Free rein over everything except the song. More energy and more fun than NO SAINT OF YOURS.
- Characters: Krea 2 two-pass on Reference Builder cards, three-panel layout (face close-up, upper body, full body),
  solid grey background, no text or borders, everyone mid-20s, punk look. Sonnet reviews the sheets.
- Location images for the band and for the singer's solo scenes; extra characters allowed.
- Same settings and workflow as NO SAINT OF YOURS, and everything learned there.

## What was made
- Cards (`tools/make_refs.py` holds every prompt; pictures in `refs/`): singer 212, guitarist 231, bassist 222,
  drummer 232, clerk 221; hall 301, office 301, archive 301, lobby 301, lift 301, roof 302.
  Sonnet passed singer, guitarist, drummer, clerk; it failed bassist 221 (uneven grey), so 222 is used (not re-checked).
- Idea: the band has taken over a grey permit office's waiting hall; the singer breaks the building's rules with a
  prim clerk as her foil (office, archive, lobby, lift for the spoken bridge, rooftop); the clerk ends up dancing.
- 27 scenes (`timeline.json`), prompts in `spec.py`. Scenes 1, 2, 23, 26, 27 render against the vocal-free stems.
- Settings: Singularity V2, pass 1 25 steps; pass 2 turbo 8-step 768 v1.1 LoRA, 4 steps, denoise 0.30; 2k;
  no continuation; ref size max (scene 21 has 6 pictures -> match).
- Word timing: ElevenLabs Scribe on the lead-vocal stem; the last line fixed by hand from the stem
  ("No permission" 142.5-144.1 s, "to exist" 148.9-150.5 s).
- ComfyUI was not running after the PC restart (23:17); started with `Run Video Builder ComfyUI.bat`.

## Runner
`tools/night_run.py parent <scenes>`: draft + finalize per scene, health watchdog, out-of-memory ladder, one
automatic re-roll when a finished take has 12 or more repeated frames (first take kept in `takes/keep`), then the
stitch (`NO_PERMISSION_TO_EXIST_auto.mp4`). Log: `logs/night.log`. Forced retakes: `tools/retake_run.py N`.

## After the queue
Review every take at zoom (count drummers, mic size, band at their spots, clerk silent), `frame_check.py`,
cut sheets, final stitch `--prefix NO_PERMISSION_TO_EXIST`, Opus frame review -> REVIEW_ISSUES.md, REPORT.md.
