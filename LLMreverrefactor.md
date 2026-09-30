# LLM Revert + Enhance Proposal

## Implementation status (2026-09-30)

All phases A1 to A9 are implemented. 317 unit tests pass, `tests/test_node_registration.py` passes, and every touched `.mjs` file passes `node --check`. Nothing has been run against LM Studio yet.

| Phase | Result | Where |
|---|---|---|
| A1 | Baseline Story Arc prompt, motion tiers, and retry text restored. Scene-map prompt removed. | `llm/prompts/storyboard.py` |
| A2 | Five-item section slot list added (setting, action, blocking, small movement, carry). | `llm/prompts/storyboard.py` |
| A3 | Pass 1 is one paragraph per section again. Code writes `Scene N (Location) —`. No lower word bound. | `storyboard/story_layer.py` |
| A4 | Pass 2 writes one short note per scene. A failed scene is skipped and the section keeps its paragraph. It runs for Detailed and Rich levels, or when the payload sets `story_arc_scene_entries` (true or false). | `storyboard/story_layer.py`, `llm/prompts/storyboard.py` |
| A5 | Scene Beat: emotional wording replaced with shot-note wording, positive people/location/previous-state block added at the top and bottom, neutral fallback text. | `llm/prompts/storyboard.py`, `storyboard/story_layer.py` |
| A6 | MiniMax style essay replaced with a four-line shot format. "Background life" removed. | `llm/prompts/minimax.py` |
| A7 | Positive cast lock, invented-person matcher, mapped-extras allowance, name-based allow list. Mirrored in JS. Strip keeps the text when removal would gut it. | `storyboard/cast_guard.py`, `web/music_video_builder/cast_guard.mjs`, `lyric_cues.mjs`, `minimax_prompt.mjs`, `batch_prompts.mjs` |
| A8 | Feeling-word check with one rewrite, for beats (Python), scene notes (Python), and shots (JS). Life-movement bank and slider-driven energy text in the shot prompt. | `story_layer.py`, `batch_prompts.mjs`, `minimax_prompt.mjs` |
| A9 | LM Studio was already a JSON-schema runner and context limits are already handled. The first Story Arc call now uses the schema on those runners, with the plain-text call as fallback. | `storyboard/story_layer.py` |

Not done:
- R-c, unselected Reference Builder objects treated as off-cast: only people are filtered. Objects selected or not are not checked.
- The second MiniMax prompt builder near `minimax_prompt.mjs` line 1880 (Scene idea / image prompt context) keeps its old wording. Only the main shot-description builder changed.

Test checklist:
1. Solo scene: no woman, girl, stranger, extra hand, or crowd in the arc note, beat, or shot text.
2. Two-subject scene: both appear, labelled `<Subject 1>` and `<Subject 2>`.
3. Character motion 9 and camera 8 in a 10 s scene: several quick actions, no slowdown.
4. Character motion 2: small movements only.
5. Story Arc at Standard (Pass 1 only) and at Detailed (Pass 1 and Pass 2). Compare time and quality.
6. Console lines starting `[VRGDG Story Layer]` show skipped scenes or structured-output fallbacks.

Baseline: `8e595af` ("Reorganize pack into feature packages for 2.0"). The hash given, `8e595af5948abfaa6b03a8c335f349f081d66968`.
Compared against: `HEAD` (`b91981c`, branch `Beta2.0`).

Findings below come from reading the diff. I have not run the LLM against either version, so F1 to F5 are likely causes, not confirmed ones. F7 and F8 match the failures you reported.

Reported failures: (1) output refers to women and other people who are not in the scene, (2) output is emotional instead of music-video specific. Runner: LM Studio with bonsai-27b.

---

## 1. What changed since the baseline

Prompt and generation changes only. No provider, runner, cache, or output-cleaning code changed under `llm/`.

| File | Change |
|---|---|
| `llm/prompts/storyboard.py` | Story Arc prompt rewritten. Scene Beat prompt extended. Motion guidance, retry prompt, and JSON schema rewritten. |
| `llm/prompts/minimax.py` | Added "WHAT MAKES A GREAT SHOT DESCRIPTION" block. Mode text changed. |
| `llm/prompts/image.py`, `llm/image_prompt_generation.py`, `llm/video_prompt_generation.py` | Shot-scale rules (fewer close-ups). |
| `storyboard/story_layer.py` | Scene-by-scene arc, scene map, per-entry word budgets, cast repair pass, output normalizer changes. |
| `storyboard/cast_guard.py` (new) | Deterministic cast filter. |
| `storyboard/scene_helpers.py`, `persistence.py` | `story_arc_detail` setting (compact / standard / detailed / rich). |
| `web/music_video_builder/minimax_prompt.mjs` | Cast labels, per-subject vocal roles, name-to-label replacement, character budget change. |
| `web/storyboard_builder/shot_presets.mjs` | Shot preset lists rewritten. |

---

## 2. Findings

**F1. The Story Arc prompt asks for too many hard constraints at once.**
Baseline: one short role, 12 or so plain rules, one paragraph per section, max 100 words.
Now: scene-by-scene entries in a fixed `Scene N (Location) — text` shape, a word window per entry (0.7x to 1x of a limit), a story-spine mandate, a cast wall, a somatic-emotion rule, a single-action rule, a depth-plane rule, a diffusion guard per motion tier, and a no-colon rule. Local GGUF and Gemma runners break under this many simultaneous requirements. The result is a heading or format failure, then a retry, then `StoryArcFormatError`.

**F2. Word windows are too tight for the model to hit.**
`_story_arc_entry_word_limit` gives `1500 // scene_count`, with a floor of 22. A 40-scene song gets about 37 words per entry, with a minimum of about 25. Models cannot count to a 25-to-37 window reliably, and the normalizer then clips text mid-thought with an ellipsis (`_cap_story_arc_words`).

**F3. The new prompt tells the model what not to write, not what to write.**
Most new lines are "never / do not / avoid". The diffusion guards ("keep limbs anchored", "avoid multi-limb acrobatics") push output toward safe, generic motion. This is the main reason the arc is less descriptive to the scene, even when it parses.

**F4. Scene Beat has 20+ rules and two extra post-processing stages.**
The cast repair LLM call and `strip_cast_leaks` delete whole sentences. If a beat mentions a cast name in a way the filter reads as a leak, the beat can be gutted. The final fallback is the generic line "The scene stays focused on X…", which is the opposite of descriptive.

**F5. Prompt behaviour is split across Python and JS.**
`minimax_prompt.mjs` now renames characters to labels and adds a per-subject role list. `minimax.py` adds a second set of style rules. Two places now shape the same output, so a bad result is hard to trace.

**F7. Invented and off-cast people are not caught (reported failure 1).**
`storyboard/cast_guard.py` only flags characters that exist in the project but are not in the scene. It matches their names, gendered nouns, and pronouns. It does not catch people the model invents ("a woman", "a stranger", "a crowd", "a figure") when no left-out project subject has that gender. Two causes feed this:
- The arc and beat prompts list every project character and the whole-song arc, so the model has other people to borrow from.
- The prompt states the cast only negatively ("never mention"), and never states the positive fact "this scene contains exactly these people".

**F8. The prompts push emotion, not music-video craft (reported failure 2).**
"Emotional stakes", "emotional turn", "symbolism", "memory", and "how the scene should feel" appear in the Scene Beat rules, the arc prompt, and the MiniMax block. The model follows them. Nothing tells it what a music video scene is made of: performance vs narrative cutaways, lip-sync moments, choreography, lighting changes on the beat, set dressing, camera moves that follow the rhythm.

**F6. What worked at the baseline.**
A short role statement, a fixed heading skeleton, one paragraph per section, and one word cap. The output was rigid and predictable, and the parse rarely failed. It lacked scene-level detail: location texture, lighting, blocking, and physical emotion were not asked for in a structured way.

---

## 3. Proposal

Keep the baseline's rigid skeleton and short prompt. Move every "must be exact" requirement out of the prompt and into code. Then add scene detail through a fixed-slot template, not through more rules.

### D1. Restore the baseline prompt text for the Story Arc

Restore from `8e595af` in `llm/prompts/storyboard.py`:
- `_storyboard_story_arc_instruction`
- `_storyboard_story_arc_structure_instruction`
- `_storyboard_story_arc_motion_guidance`
- `_storyboard_story_arc_format_retry_instruction`
- `_storyboard_story_arc_schema`
- `_storyboard_story_arc_json_retry_instruction`

Command sketch: `git show 8e595af:llm/prompts/storyboard.py`, then copy only those functions. Do not check out the whole file, because `_storyboard_scene_beat_cast_repair_instruction` and the shot-scale text in the dialogue planner must stay.

### D2. Enhance with a fixed-slot section template (still one paragraph per section)

Add to the baseline prompt a short "What each section paragraph contains" list. Each item is positive and ordered:

1. Setting: the mapped location, with one lighting or weather detail taken from its description.
2. Action: one continuous physical action by the main character.
3. Blocking: where the character sits in the space (foreground, midground, background).
4. Physical emotion: one visible cue (breath, hands, gaze, posture).
5. Carry: the object or state that passes into the next section.

Word cap uses the existing `story_arc_detail` profile (compact / standard / detailed / rich). Keep that setting, but give the model a target and a hard cap only in code (D4), not a min/max window in the prompt.

### D3. Two-pass arc: sections first, scene detail second

- **Pass 1** (baseline behaviour): one call, exact headings, one paragraph per section. This is the call that parsed reliably.
- **Pass 2** (new, optional per detail level): one short call per scene. Input: that section's paragraph, the scene's location name and description, and the scene's cast. Output: one to two sentences following the D2 slots.

Small calls succeed more often on local models than one call that must produce every scene. If a Pass 2 call fails, keep the Pass 1 paragraph for that scene and continue. No exception.

### D4. Code writes the structure, the model writes only prose

- Code writes `Scene <n> (<location>) —` and assigns each scene to its section using the existing `_story_arc_scene_map`. The model never has to produce scene numbers, location names, or the em dash. This removes the colon problem and the exact-format retry path for scene entries.
- Code applies the word cap after generation (already exists). Remove the lower bound.
- Keep `cast_guard.py` as a post-filter for the arc and beats. Remove the cast wall text and the cast repair LLM call from the prompt path unless F4 is confirmed to be about cast leaks.

### D5. Scene Beat: baseline rules plus a Scene Facts block

- Restore the baseline rule list for `_storyboard_scene_beat_instruction` (drop the added cast rule, the arc-line rule, and keep only what the FLF path needs).
- Add one block, built by code, above the rules:
  ```
  Scene facts (use these; do not invent others):
  - Location: <name> — <description, trimmed>
  - Cast: <names>
  - Previous end state: <text or none>
  ```
- Add a fixed output template: one paragraph, in this order: Setting, Action, Emotion cue, Carry-forward. Keep the existing word limit (80, or 100 with extras).
- Keep `strip_cast_leaks`, but when it removes a sentence, keep the original beat if the result would be under 30 words. Log it with `[VRGDG Story Layer]`.

### D6. MiniMax shot prompt: keep plumbing, trim style rules

Keep:
- Cast labels and per-subject vocal roles in `minimax_prompt.mjs` (needed for lip-sync correctness).
- The "no appearance restatement" rule.

Review:
- The "WHAT MAKES A GREAT SHOT DESCRIPTION" block in `llm/prompts/minimax.py`. Reduce to Camera, Action, Environment. Drop Emotion, Realism, and the Subjects paragraph, because `minimax_prompt.mjs` already injects the cast block.

### D7. Cast lock built by code (fixes reported failure 1)

Target behaviour from you: if the scene card shows only the man, only he is in the prompt. No woman, girl, extra hands, or unnamed person. The baseline did this. Now it does not.

Extra cause found in the current shot prompt: `llm/prompts/minimax.py` line 29 asks for "background life around them", which invites unnamed people. Remove it. The INDEPENDENT SUBJECT ACTION text in `minimax_prompt.mjs` only applies with 2 or more subjects, so it is not a cause for solo scenes.

**Naming rules (Reference Builder names are user-defined):**
- **R-a.** Every cast line, label, and check is built from the Reference Builder name of each selected item, never from hardcoded words like "man" or "woman". `miniMaxH3SubjectLabelMapForSegment` already maps items to `<Subject N>` and keeps the name, so the cast lock reads from that.
- **R-b.** The invented-person check (D7 below) allows any word that appears in a selected item's name. A subject named "The Man" makes "man" legal in that scene. A subject named "Kai" makes "man" flagged unless the text is about Kai.
- **R-c.** Reference Builder objects and locations selected for the scene are allowed props and settings. They are not people and are never flagged. Objects not selected for the scene are treated like off-cast characters: no mention.
- **R-d.** `cast_guard.py` `subject_gender` currently guesses gender to decide which nouns and pronouns to flag. Names can be anything, so for the invented-person check, use the flagged-word list above and the name-based allow list from R-b. Do not depend on a gender guess.

Rules added to the shot prompt, from code, per scene (`<name>` is the Reference Builder name):
- Solo scene: `Only <Subject 1> (<name>) is in this shot. No other person, hand, arm, shadow, reflection, silhouette, or crowd appears. Refer to them only by label, never by pronoun.`
- Multi-subject scene: `Only <Subject 1> (<name 1>) and <Subject 2> (<name 2>) are in this shot. No other people.`
- No-character scene: `No people appear. Show only the location, objects, and light.`

- Per scene, code builds a positive line placed directly above the task: `People in this scene: <names>. No other people exist in this scene.` For a scene with no characters: `No people appear in this scene. Show only the location, objects, and light.`
- Send each per-scene call only that scene's cast and location. Do not send the whole-song character list or the whole-song arc. Whole-song context goes in as a 2-line summary.
- Extend `cast_guard.py` with an invented-person check. After generation, flag `woman, women, man, men, girl, boy, stranger, figure, silhouette, crowd, onlookers, couple, lover, someone` unless it refers to a named cast member.
- On a flag: one rewrite retry listing the flagged words (existing repair path). Strip the sentence only if the result stays over 30 words.
- Tell the model to use cast names, not "he/she", in arc entries and beats. This removes pronoun drift toward absent people.

### D8. Strict short shot format with life-like movement (fixes reported failure 2)

Direction from you: prompts must be short and strict, use `<Subject 1> The Man` / `<Subject 2> The Woman` labels, say what they do, how they move, and how the camera moves. Camera and action must fit the scene and follow the scene builder motion sliders (pop, fast motion, high energy), without hard limits from duration. Characters need more emotion, shown as life-like movement.

**Shot description format (one per shot, fixed order, 2 to 4 sentences):**
1. Camera: opening frame, one named move (direction, speed), ending frame.
2. Subject action: `<Subject 1> The Man` + one continuous physical action, then `<Subject 2> The Woman` + their own action (only the labels in this scene's cast).
3. Life detail: one or two small human movements from the list below, placed inside the action.
4. Light/set: one line, the location's own light and props only.

**Intensity comes from the scene builder settings, not from seconds.** No duration table and no per-second limits. The existing settings already reach the shot prompt (`camera_motion_speed`, `character_motion_speed`, camera flow, performance style, in `minimax_prompt.mjs` around lines 851 and 984). The prompt passes them through unchanged and the model matches them:

| Setting | Effect on the shot text |
|---|---|
| Character motion speed 1 to 3 | small movements, held poses with life detail |
| 4 to 6 | walking, turning, reaching, steady physical action |
| 7 to 10 | pop, fast, high-energy action: jumps, spins, runs, dance hits, quick reactions |
| Camera motion speed | same scale for the camera: slow drift at low values, fast whips, orbits, and push-ins at high values |

Rules:
- The slider value sets how energetic the action is. The prompt never caps it.
- Scene length is passed only as information ("this shot is N seconds"). It is not a rule. The model may fit a fast 4 s scene with several quick actions or a slow 10 s scene with one long move.
- High motion in a long scene is supported. Duration and slider combine as beat density: the slider sets how energetic each action is, and the scene length sets how many actions are chained. A 10 s scene at motion 9 gets a run of quick actions (for example spin, step, reach, snap turn, lean into camera), not one action stretched. A 4 s scene at motion 9 gets two or three fast beats. The prompt says: "Fill the full shot length with continuous action at this energy. Do not slow down or hold still."
- Long scenes are also split by the existing cut plan (`miniMaxH3OfficialShotPlan`, cut frequency setting). More shots means each shot carries its own camera move and action, so long high-energy scenes stay varied.
- The life-like movement bank (below) scales with the slider: subtle at low values, exaggerated and quick at high values.
- The restored baseline motion text in the Story Arc (D1) keeps the slider tiers. Its "diffusion guard" lines that limit motion are removed so high settings are not held back.

**Life-like movement bank (replaces abstract emotion; code injects 4 to 6 relevant ones, the model picks):**
breath catching on a lyric, weight shifting foot to foot, fingers tightening then loosening on an object, a glance away and back, a half step forward then a stop, shoulders dropping after a hold, a slow blink, head tilt on a held note, a hand brushing hair or collar, a small smile that fades, jaw tension then release, leaning toward or away, an unfinished gesture.
Rule: emotion is written only as one of these visible movements. Feeling words are not allowed.

**Prompt length:** target 60 to 110 words per shot. This is a length target for the prompt text, not a limit on how much motion happens in the shot. Remove the "WHAT MAKES A GREAT SHOT DESCRIPTION" six-point essay and replace it with the four-line format above. The existing character budget stays as a hard cap.

**Feeling-word lint (code):** if the output has 2 or more of `feel, feeling, grief, longing, yearning, memory, soul, emotion, emotional, nostalgia, heartbreak`, run one rewrite retry with the list. Keep the original if the retry is worse.

**Arc and beat wording:** remove "emotional stakes", "emotional turn", "symbolism", "how the scene should feel", "somatic emotion". Beats become: `Cast doing what, where, with what camera idea, for how long`.

### D9. LM Studio and bonsai-27b settings

- LM Studio supports JSON-schema structured output. Use it for the arc and per-scene calls so the model cannot add extra headings or commentary. `_runner_supports_json_schema` exists in `llm/builder_runner.py`. Confirm it returns true for LM Studio, and enable it if not.
- Use temperature 0.3 to 0.45 for arc and beat calls. Higher values increase invented people.
- Keep prompts short. A 27B model follows a short prompt with a template better than a long rule list.
- Confirm LM Studio context length is at least 8192. The current arc prompt with a scene map is long.

### D10. Keep untouched

- `story_arc_detail` setting and UI.
- `cast_guard.py` and `cast_guard.mjs` (as filters).
- Shot-scale changes in `image.py`, `image_prompt_generation.py`, `video_prompt_generation.py`, and `shot_presets.mjs`.
- All web UI changes not listed above.

---

## 4. Phases

| Phase | Work | Files | Check |
|---|---|---|---|
| A1 | Restore baseline Story Arc functions (D1) | `llm/prompts/storyboard.py` | Story Arc generates on the same project that fails now |
| A2 | Add slot template to the arc prompt (D2) | `llm/prompts/storyboard.py` | Output has location, action, blocking, emotion, carry in each section |
| A3 | Code-built scene entries, remove lower word bound (D4) | `storyboard/story_layer.py` | No `StoryArcFormatError` on a 40-scene project |
| A4 | Per-scene Pass 2 with fallback (D3) | `storyboard/story_layer.py`, `llm/prompts/storyboard.py` | A forced Pass 2 failure keeps the Pass 1 text |
| A5 | Scene Beat baseline rules and Scene Facts block (D5) | `llm/prompts/storyboard.py`, `storyboard/story_layer.py` | Beats name the location detail and are not the generic fallback line |
| A6 | Trim MiniMax style block (D6) | `llm/prompts/minimax.py` | Shot text still uses `<Subject N>` labels |
| A7 | Cast lock and invented-person check (D7) | `storyboard/cast_guard.py`, `storyboard/story_layer.py`, `tests/test_cast_guard.py` | Scene with one named subject never gets "a woman", "a stranger", or an unlisted name |
| A8 | Music-video vocabulary and emotion lint (D8) | `llm/prompts/storyboard.py`, `llm/prompts/minimax.py`, `storyboard/story_layer.py` | Beats read like a shot list note with no feeling words |
| A9 | LM Studio structured output check (D9) | `llm/builder_runner.py` | Arc returns valid JSON with only the required headings |

Run after each phase, from the pack root:
```
..\..\..\python_embeded\python.exe -m unittest discover -s tests -p "test_*.py"
```
`tests/test_story_arc_sections.py` and `tests/test_cast_guard.py` will need updates in A1 to A3, since they were written for the scene-map format.

---

## 5. Risks

- **R1.** Restoring the baseline arc drops the scene-by-scene map that the Scene Beat prompt reads ("If the User Story Arc contains a line that starts with this scene's number…"). D4 keeps that format, written by code, so the Scene Beat rule still works.
- **R2.** Pass 2 adds N small LLM calls per arc. On a 40-scene song on a local model, this adds time. Mitigation: make Pass 2 depend on `story_arc_detail` (off for compact and standard, on for detailed and rich).
- **R3.** Loosening the cast wall in the arc prompt may let non-cast characters back in. Mitigation: `strip_story_arc_entry_leaks` stays.
- **R4.** Existing saved storyboards were generated in the new format. They still load, because the parser accepts both.

---

## 6. Answers so far and open items

- **Q1 (answered).** Failures are off-cast people and emotional tone. Addressed by D7 and D8. Format errors were not reported, so F1 and F2 are lower priority than first stated.
- **Q2 (answered: unsure).** Test Pass 2 (D3) after A1, A2, A7, A8. Keep it only if beats are clearly better and a 40-scene arc finishes in a time you accept.
- **Q3 (answered).** LM Studio, bonsai-27b. See D9.
- **Q4 (answered).** The chorus/verse/bridge split is dropped. Scenes are shaped by camera move, duration, and life-like movement instead (D8).
- **Q5 (answered).** Names come from the Reference Builder and can be anything. "The Man" was only an example. Label format is `<Subject N>` plus the Reference Builder name, and the builder also holds objects and other kinds. See D7 rules R-a to R-d.

---

## 7. Recommendation

Order: A1, A2, A7, A8, A9, then test on one project. These restore the behaviour that worked, add scene detail, remove off-cast people, switch the tone to music-video craft, and check LM Studio structured output. Do A3 to A6 only if format errors or generic beats remain after the test.
