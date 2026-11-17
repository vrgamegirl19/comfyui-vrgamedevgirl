---
name: vrgdg-band-music-video
description: Make or revise a complete MiniMax H3 band music video in the VRGDG Video Builder, combining band performance with the lead singer's journey. Includes character and location references, word-timed lyrics, stem-driven playing, H3 prompts, draft and final renders, overnight queues, quality review, smoothing, stitching, and reports. Use for band, rock, punk, or metal music videos from a song folder with stems, lyrics, and rules, including unattended runs and fixes to existing scenes.
---

# VRGDG band music video (performance + singer's journey)

This skill reproduces the two videos the user rated best: **NO SAINT OF YOURS** (gothic rock, 30 scenes, 3:00) and
**NO PERMISSION TO EXIST** (British post-punk, 27 scenes, 2:44). Both were made overnight with no one watching,
then polished with the user the next day. Everything here comes from them: the settings, the scripts, the prompt
wording that fixed real failures, and the mistakes not to repeat.

**Read before starting:** `references/lessons.md` (what worked, every failure and its fix, the do-not list).
**Read before writing prompts:** `references/band-video-prompts.md` and the two example specs in
`assets/examples/`. The deeper prompt method is `references/prompt-writing.md`.

Use other skills instead for: a short film with H3 voices (vrgdg-h3-film); small Builder chores
(vrgdg-builder-api); story/narrator music videos with storyboard gags (vrgdg-h3-music-video, which this skill's
scripts come from).

## Ground rules

- **The user's rules file wins.** Read `rules.txt` and every other text file in their folder first (lyrics, song
  style, model settings). This skill's defaults fill the gaps (`assets/default_brief.md`).
- **Only the API writes Builder project files.** Never edit `vrgdg_builder_session.json`. Your files go in
  `<project>/_claude_work/`.
- **Never change the LLM** loaded in LM Studio (this workflow doesn't use it).
- **One GPU job at a time**, only when ComfyUI is idle. Never queue renders the user didn't ask for. Starting
  ComfyUI is the user's call, unless their rules allow you to recover it (`night_run.py --recover`).
- **Never print the ElevenLabs key**, and never send the user's personal details anywhere.
- **Never crop or edit the user's reference pictures.** Ask for a fixed sheet instead.
- **Positive-only prompts; describe only what is in view; no fade-in from black at the start.**
- **Read every built prompt in full before rendering.** If something is wrong, stop and fix it (even mid-run).
- **Two modes:** *unattended* ("I'm going to bed", "do your thing, GO!"): make every creative decision yourself,
  never wait, don't review drafts, re-roll clear failures within budget, report in the morning. *With the user
  present*: after reviewing a take, propose fixes and wait for their OK. A question gets an answer, not an edit.
- **Keep every old take and every film.** Copy takes to `takes/keep/` before re-rendering; number new films
  `_2`, `_3`.

## Scripts

Run with ComfyUI's embedded Python (`<ComfyUI portable>/python_embeded/python.exe`); every script takes
`--work <work folder>`. The scripts that came from vrgdg-h3-music-video are documented in their headers and in
`references/builder-api.md`; the ones made for these two videos:

| Script | Does |
|---|---|
| `builder_transcribe.py` | **no-key word timing:** the Builder's Demucs stems (`POST audio/stems`, skipped when the user gave stems) + the Builder's Stable-ts/Whisper large-v3 (`POST lyrics/align`) on the vocal stem -> words in Scribe's JSON shape |
| `set_singers.py` | the Builder's singer assignment (`lyric_singers`) on every sung scene, so the Builder's multi-shot cue timing works on the project too |
| `mix_stems.py` | the user's stems minus the vocals -> the vocal-free song |
| `stem_scenes.py` | which stem plays in which scene (who is shown playing, when backing vocals come in) |
| `make_refs.py` | Reference Builder cards + Krea 2 pictures by seed from `refs_spec.py`; `gen`, `pick`, `show` |
| `night_run.py` | unattended runner: draft + finalize per scene, ComfyUI watchdog, out-of-memory ladder, one re-roll for choppy takes, retakes, stitch |
| `held_frames.py` | counts repeated frames in a take (0-6 clean, 12+ choppy) |
| `rife_fix.py` | smooths a choppy take with the RIFE node (12 fps -> 24 fps), real frames kept at cuts, audio copied |

The main ones from the parent skill: `check_inputs.py`, `elevenlabs.py`, `align_lyrics.py`, `vocal_activity.py`,
`build_track.py`, `setup_project.py`, `setup_refs.py`, `setup_timeline.py`, `shot_lyrics.py`,
`build_prompts.py`, `render_guarded.py`, `cut_sheet.py`, `qa_sheet.py`, `stitch.py`, `sync_check.py`,
`splice_scene.py`, `social_kit.py`.

## Workflow

### 1. Read the inputs (minutes)
- List the user's folder: song, full mix, stems, lyrics, song style, rules, key file, reference folders.
- Note what the rules decide: settings, scene lengths, who makes the characters, locations, ages and look, tone,
  key or no key, whether you may restart ComfyUI, whether they want an Opus review at the end.
- `ffprobe` the song; `check_inputs.py` (model and LoRA installed, ComfyUI and the API up, ffmpeg).
- If the user is about to leave, reply in a few lines (what's there, the plan, the defaults you'll use) and start.

### 2. Project and work folder
- Create the Builder project (`POST /projects`), then `_claude_work/` with `source/` (copies of every input),
  `audio/`, `prompts/`, `qa/`, `logs/`, `refs/`, `takes/keep/`.
- `config.json` from `assets/config_example.json`: project id and folder, title, the user's settings (theirs win),
  `lead_stem_path`, `elevenlabs_key_file` (empty when there is no key), `comfy_launcher`, `singer_card`.

### 3. Audio and word timing
1. **Vocal-free song:** with the user's stems, `mix_stems.py --work W <stems dir>`. Without stems,
   `builder_transcribe.py` (step 3) writes it from the Builder's split.
2. `build_track.py --work W`, then `setup_project.py --work W` (applies settings, attaches `audio/full_track.wav`,
   prints the effective settings: check them).
3. **Word timing, one of:**
   - **ElevenLabs key given:** `elevenlabs.py --work W test`; if it passes,
     `elevenlabs.py --work W stt <lead vocal stem> audio/stt/vocals_scribe.json`.
   - **No key, or the key is rejected:** `builder_transcribe.py --work W [--vocals <lead vocal stem>]`. With no
     stems it splits them through the Builder first and fills `song_no_vocals`; then run `build_track.py` again
     so `full_track_no_vocals.wav` exists (keep `intro_seconds` 0 in that case: the Builder split already covers
     the whole track).
4. `align_lyrics.py --work W --words <the JSON from step 3>` -> `lyrics_timed.json`. Add repeats the lyrics file
   lacks and run again.
5. **Hand-check the end and the held notes** with `vocal_activity.py <lead stem> --detail A B` and fix
   `lyrics_timed.json` (both videos needed this for the last words).
6. `stem_scenes.py --work W <stems dir> --grid 1` for the song map: when each instrument enters, backing vocal
   windows, breaks, the bridge.

### 4. The idea and the timeline
- One idea (`lessons.md` section 5): the band in one location for choruses, breaks, intro and ending; the singer's
  journey through four to six locations for verses, pre-choruses and the bridge; one gag or action per scene; an
  optional foil character for a fun song; a payoff in the last chorus; the last hit played through.
- `timeline.json`: scenes 2-8 s (or the user's lengths), cuts in vocal gaps. `setup_timeline.py --work W
  --dry-run`, read the table and the word-cut warnings, then run it for real. Then `stem_scenes.py --work W
  <stems dir>` per scene.

### 5. Characters and locations
- **Claude makes them** (the NO PERMISSION TO EXIST way): write `refs_spec.py` from
  `assets/refs_spec_example.py` (exact young ages and fresh faces, the genre look item by item, empty hands,
  empty locations that name every feature you'll use, a wide open floor for the band). `make_refs.py --work W
  cards`, then `make_refs.py --work W gen <id> <seed> <seed> [<seed>]` for each card (one job at a time).
- **Sonnet checks each character sheet:** an Agent with `model: "sonnet"` reads the candidates and reports per
  sheet: panel order (face close-up, upper body, full body), solid grey with no borders/lines/text, the same
  person and outfit in all three, the right age, empty hands. Regenerate failures and have Sonnet check the
  replacement too. View the locations yourself. `make_refs.py --work W pick <id> <seed>`.
- **The user supplies them** (the NO SAINT OF YOURS way): `setup_refs.py` with a `refs.json`. Look at every
  sheet; note instruments or props on them (they leak into scenes).

### 6. Prompts
- Copy the closer example spec to `spec.py` (`assets/examples/`): a band on an open floor with a foil, or a
  raised stage with the singer alone. Rewrite the docstring, `TEXT`, `CHARACTERS`, `LOCATIONS`, constants and every
  scene for the new song (`references/band-video-prompts.md`).
- `shot_lyrics.py --work W --draft` gives each shot's words and windows.
- `build_prompts.py --work W` -> no PROBLEM, read every WARN, then **read every prompt file in `prompts/`** against
  the checklist in band-video-prompts.md. Fix and rebuild.
- `build_prompts.py --work W --apply all`, then `set_singers.py --work W`.
- Write `HANDOFF.md` (example in `assets/examples/`) so another session or Opus can take over.

### 7. Render
- Unattended: `night_run.py --work W [--recover] parent <scene list>` with `run_in_background`; Monitor
  `logs/night.log` for `FINALIZED|FAILED|choppy|QUEUE DONE|NIGHT RUN END` (re-arm it when it expires). Put the
  hardest scenes first (a full-band wide, a single-member shot, a prop gag), then timeline order. About 14 minutes
  per 2-8 s scene on an RTX 5090: 27 scenes plus retakes take about 8 hours.
- With the user present: drafts for review (`render_guarded.py --phase draft`), finalize what they approve.
- Retakes while it runs: append scene numbers to `logs/extra_jobs.txt`; later `night_run.py --work W retake N...`.

### 8. Review
- Every finished take at zoom: count people and drum kits, check the mic size, everyone at their spot (on the
  platform), the singer on screen for every sung line, silent people with closed lips, the gag readable, the
  first and last frames (`qa_sheet.py`, `cut_sheet.py`).
- `held_frames.py` on every take. A scene choppy across 2-3 seeds: `rife_fix.py`, then
  `POST /scenes/{n}/video/select`. Don't reword it.
- Re-roll clear failures (head counts, strangers, a broken gag) within the budget (one re-roll per scene
  overnight unless the user set one). Log verdicts in `logs/review.txt`.
- **Opus frame review** when the user asks for it (or at the end of an unattended run): an Agent with
  `model: "opus"` gets each scene's prompt and frame sheets and writes `REVIEW_ISSUES.md` (OK / minor / re-render
  per scene). Check yourself the scenes re-rendered while it ran.

### 9. Deliver
- `stitch.py --work W --prefix <TITLE>`: duration = song, 0 ms sync in every scene, no black or frozen stretches,
  no fade-in.
- `REPORT.md` from `assets/report_template.md` (examples in `assets/examples/`). Tell the user the film's path and
  length, the retakes, and the watch list worst first.

### 10. Fixes afterwards
The user reviews the film and names scenes. For each: look at the frames first, find the cause in the prompt
(`lessons.md` sections 6 and 11 list the ones seen so far), save the old take, fix the wording or use RIFE, re-render
only what they named, re-stitch to a new numbered film, and update REPORT.md. Fix the same wording in scenes not
yet rendered.

## Files

| Where | What |
|---|---|
| `references/lessons.md` | **the core:** results, settings, audio, reference pictures, idea, prompt fixes, rendering, repeated frames, review, delivery, the user's later fixes, the do-not list |
| `references/band-video-prompts.md` | the spec pattern for band videos, constants, shot patterns that landed, the pre-apply checklist |
| `references/prompt-writing.md` | the full H3 prompt method (sections, cues, first frames, wording traps) |
| `references/qa-and-fixes.md`, `builder-api.md`, `audio-and-lyrics.md`, `social-kit.md` | from the parent skill: QA, every endpoint and API quirk, audio details, the social kit |
| `assets/config_example.json` | the settings both videos used |
| `assets/default_brief.md` | the user's creative direction, for gaps in a new brief |
| `assets/refs_spec_example.py` | the Krea 2 cards of NO PERMISSION TO EXIST |
| `assets/spec_template.py` | the spec format |
| `assets/examples/` | both specs, both reports, the handoff, the timeline, and the user's rules files from both videos |
| `assets/report_template.md` | the end report |
