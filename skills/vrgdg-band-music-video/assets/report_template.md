# <TITLE>: report (<date>)

## The film
- **Final:** `<path>\<TITLE>.mp4`
- 1920x1088, <seconds> s (the song is <seconds> s), <N> scenes.
- Checks: every scene at 0 ms audio offset, no black or frozen stretches, no fade-in at the start.
- `<TITLE>_auto.mp4` is the first stitch, before the retakes below.

## The idea
<two to four sentences: the band location, the singer's journey, any extra character, the payoff>

## What was made
- **Characters:** <cards; Krea 2 seeds picked, or the user's sheets>. Prompts in `refs_spec.py`, candidates in `refs\`.
- **Sonnet's sheet check:** <passed / failed and replaced (re-checked?)>.
- **Locations:** <list>.
- **Lyrics:** timed with <ElevenLabs Scribe | the Video Builder's transcribe (Whisper large-v3 on the vocal stem)>;
  <hand fixes, e.g. the held last words>.

## Settings
<pass 1 / pass 2 / LoRA / resolution / continuation / scenes using reference size "match">

## What happened
- <ComfyUI restarts, key problems, out-of-memory retries>
- <N> renders; <how many finished first try>.

## Retakes (older takes are in `takes\keep`)
| Scene | Why | Result |
|---|---|---|

## Watch list (worst first)
- **Scene <n> (<m:ss>):** <what to check>.
- Repeated frames still in: <scenes and counts> (`held_frames.py`; `rife_fix.py` can smooth them).
- From the Opus frame review (`REVIEW_ISSUES.md`: <x> OK, <y> minor, <z> re-render): <the items>.

## Clean across the film
<head counts, mic size, faces and hands, characters consistent>

## To change a scene
Edit its prompt in `spec.py`, run `build_prompts.py --work . --apply N`, then `night_run.py --work . retake N`,
then `stitch.py --work . --prefix <TITLE>_2` (never overwrite an earlier film). `HANDOFF.md` has the rest.
