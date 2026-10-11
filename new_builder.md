# New Builder Flow: Story First, Locations From the Story

Goal: the user writes a real story, and the story is broken down level by level until every timeline
scene knows its part:

```
Story      the whole tale: title, premise, cast, world
 └ Chapter   a major part of the story, set in one general setting      (Home)
    └ Act      one event inside that chapter, a run of scenes             (The argument)
       └ Scene   one timeline scene at one specific spot in the setting  (Home - Kitchen)
```

Places follow the same tree: the chapter owns the **setting**, each scene stands at a **spot** inside it.

Lyrics set the timing, the scene count, and the mood and emotion. They never decide the story events or
any place. Lyrics can be literal or metaphorical, so nothing in them is read as a setting or a spot.

---

## 1. Current flow and what is wrong with it

Wizard order today (`web/music_video_builder/wizard.mjs:725`):

1. Settings
2. Audio
3. Lyrics + Scenes
4. Mode
5. References (subjects **and locations**)
6. Story Direction (story idea, arc, brief, beats)
7. Gemma + Render

Findings:

- **F1. Locations are made before any story exists.** LM Extract (`reference_locations.mjs:488` →
  `/vrgdg/music_builder/lm_scout_locations` → `_generate_lm_scout_locations`) sends the lyrics per scene,
  scene notes and an optional style/theme. Its prompt (`llm/prompts/location_scout.py`) calls the lyrics the
  "primary narrative and tonal anchor" and asks for 18-22 generic shootable places. The bodega, parking
  and laundromat lists in that prompt are why unrelated places show up.
  Same problem in `/vrgdg/music_builder/wizard_locations_from_lyrics` and Auto Map
  (`_generate_flux_reference_location_map`), which maps locations to scenes by lyric/concept text.
- **F2. The story arc is shaped by the song structure.** `_storyboard_story_arc_instruction`
  (`llm/prompts/storyboard.py:458`) forces one paragraph per lyric section (`Verse 1:`, `Chorus 1:`) and
  says "use only those locations as the physical settings". So the arc is written inside locations the
  lyrics picked, and reads as a shot list, not a story.
- **F3. Arc and brief depend on each other in a loop.** The brief prompt takes `user_story_arc`, and the
  arc prompt takes `song_story_brief` when it exists. Order is arc then brief (playbook steps 12-13), so the
  brief is a summary of the arc, not a plan for the scenes.
- **F4. The brief ignores the scene count.** `_storyboard_story_brief_instruction` returns 4 headings in
  under 250 words. With 20 or 46 scenes, no scene gets its own part of the story.
- **F5. Scene beats are locked to the mapped location.** Scene beat and prompt rules treat `location_ref` as
  the required set and tell the LLM to translate any story place into it. This rule is correct, but only
  once the mapped locations come from the story.
- **F6. Scene assignment is positional.** "Scenes per location block"
  (`reference_scene_assignment.mjs:57`, `location_block_size` in `reference_orchestrator.py:218`) cycles
  locations every N scenes regardless of what happens in those scenes.

---

## 2. New flow

| # | Step | What happens | Changed |
|---|------|--------------|---------|
| 1 | Settings | Models and render setup | No |
| 2 | Audio | Load the song | No |
| 3 | Lyrics + Scenes | Transcribe/align lyrics, build N timeline scenes | No |
| 4 | Mode | i2v or reference video | No |
| 5 | Cast | Main characters and extras, describe them | Split out of References |
| 6 | Story | Write the book: title, premise, chapters, each chapter's setting. New cast can be added here | New |
| 7 | Acts | Each chapter is broken into acts, each act a run of scenes | New |
| 8 | Scene Plan | Each scene gets its slice of its act and a spot in the chapter's setting, merged with the user's notes | Replaces the brief |
| 9 | Locations | Settings and their spots built from the story, already mapped to scenes | Rewritten |
| 10 | Scene Beats | One beat per scene from its slice, notes and location | Inputs change |
| 11 | Prompts + Render | Unchanged | No |

### Step 5: Cast

Same subject and extras cards as today, just before the story so the book uses their names and looks.

### Step 6: Story

- Input: premise from the user (a sentence up to a full draft), cast, optional tone/genre/world notes.
  Lyrics are sent only as mood and emotion context, never as events or places.
- Output: a title, a premise line, and 3-7 chapters of plain prose. Each chapter names its **setting**
  (Home, The Night Train, Grandmother's Farm). No `Intro`, `Verse`, `Chorus` or `Scene` labels and no
  camera language. It reads like a book.
- The user can set or change a chapter's setting directly. Two chapters can share a setting (the story
  returns home in chapter 5).
- Size follows the song: the prompt gets the scene count and the song length.
- Actions: write it by hand, paste a finished story, draft it with the LLM, expand or rewrite one chapter,
  reroll with a new seed.
- **Adding cast while writing:**
  - The user types a new name in the story (or presses **Add Cast Member**) and a new subject card is made
    in the Reference Builder with a name and a description drafted from the story.
  - After a draft, the LLM lists people in the story who have no card (the conductor, her brother). The
    user ticks the ones to keep and they become cards. Unticked ones stay unnamed background.
  - New cards need an image or RefMod before render, the same as cards made in step 5. The Story step
    shows a warning until they have one.

### Step 7: Acts

- An act is one event inside a chapter (the argument, the phone call, leaving the house). It is a
  contiguous run of scenes and stays inside its chapter's setting. It can move between spots in that
  setting (kitchen to hallway to front door).
- The LLM splits each chapter into acts and proposes a scene count per act. The total always equals N.
- **Timed timeline notes** pin events. A note at 42-58 s ("she finds the letter") must fall inside the
  act that covers those scenes, so act borders move to fit it.
- Each act stores: chapter, title, a 1-3 sentence summary, cast, first and last scene.
- The user can drag act borders on the timeline, change the scene count, merge or split acts, and
  regenerate one chapter's acts.
- Song energy is a hint: choruses and big drops should land on bigger story moments. It never reorders
  the story.

### Step 8: Scene Plan

- Each scene gets a `story_slice`: one or two sentences of what happens in that scene, taken from its act.
- Each scene gets a `spot`: the specific place inside the chapter's setting where the slice happens
  (Kitchen, Bedroom, Front Porch). The spot comes from the story action, never from the lyric line.
- **Scenes per location** becomes the minimum run at one spot. With a value of 3 the camera stays in the
  kitchen for at least 3 scenes before moving, unless the story or a user note moves it sooner.
- The scene's mood and emotion come from its lyric line and the song's energy at that point, weighted by
  Lyric Story Strength.
- The user's existing scene inputs are read first and win over generated text:
  - **Director Notes** (`timeline_note`): how the user wants the scene made and what happens.
  - **Video Notes** (`i2v_notes`): motion and action.
  - **Planning Notes** (`notes`).
  - **Timed Timeline Notes** (`timeline_markers`) that overlap the scene.
- When notes exist, the slice is written to agree with them. When a note conflicts with the story, the note
  wins and the slice records it, so scene beats and prompts never fight the user.
- Code checks: every scene is covered once, acts and chapters stay in order, no empty slice. Missing
  scenes get a retry for just those scenes. For 46+ scenes the plan is written one chapter at a time.
- The old brief header (premise, world, tone, motifs) is kept in `song_story_brief` as a short summary
  for prompts that need the whole picture.

### Step 9: Locations

- Input: the chapters' settings, the scene plan's spots, the book, the style/theme field (look and banned
  places), cast. **The lyrics are not sent to this step at all.**
- Two levels:
  - **Setting** (Home): a shared look for every spot inside it. Era, region, architecture, materials,
    palette, light, weather. Written once per setting.
  - **Spot** (Home - Kitchen): one location card per distinct spot. Its description starts from the
    setting's look so every room of the same house matches, then adds what is specific to the spot
    (20-38 words, physical only, as today).
- Spots are merged by name inside a setting, so returning to the kitchen in chapter 5 reuses the same card.
- Each spot card stores its setting, acts and scene ids, and the scene mapping is written directly. Auto
  Map and the block assignment are no longer needed for this, but stay available for manual changes.
- The user can rename, merge, split or regenerate a setting or a spot. Changing a setting's look offers to
  rewrite its spots. Images or RefMods are added per spot as today.

### Step 10: Scene beats

- Input per scene: its `story_slice`, the neighbouring slices, its act summary, the user's notes, the
  mapped location, cast, and the lyric line as timing and mood only.
- The location lock from F5 stays.

---

## 2b. Story Mode and the Story Builder

### Story Mode video type

- A fourth video type, **Story Mode** (`video_type: "story"`), next to Singing, Speaking and No Lip Sync.
  Selecting it turns on everything in this document (flow v2). The other three types keep today's flow.
- Today the video type also sets the performance (lip sync or not). In Story Mode the performance is set
  **per scene** with the existing `performance_mode` field (`singing`, `no_lip_sync`). The scene plan
  proposes it: the singer performs to camera in some scenes, other scenes are pure story. The user can
  flip any scene. Default per act, editable per scene.
- About 40 files branch on the video type, and `normalizeVideoType` (`controls.mjs:270`) turns any unknown
  value into `singing`. To avoid breaking them:
  - `normalizeVideoType` stays the performance normalizer and returns `singing` for `story`. Every render,
    audio and prompt check already goes through it (often as `segment.performance_mode || state.videoType`),
    so they need no change.
  - New `normalizeProjectVideoType` keeps `story`. It is used only where the project's type is stored:
    session save/load, project defaults, the Video Type select and the wizard.
  - The Python twins (`storyboard/scene_helpers.py`, `storyboard/session_sync.py`,
    `agent_api/video_prompt_inputs.py`) already read any unknown type as `singing`, and the Agent API saves
    `video_type` as sent. No Python change was needed.
  - Audio handling, lyric alignment and render pipelines are the same as Singing.
  - Done in S1. Tests: `tests/test_story_mode_video_type.py` and `tests/story_mode_video_type.mjs`.

### The Story Builder

In Story Mode the Storyboard Builder opens as the **Story Builder**: one window with tabs in the order the
video is made. Each tab brings in the tools it needs, wherever they live today.

| Tab | What you do there | Tools moved in from |
|-----|-------------------|---------------------|
| 1. Song | Check lyrics, line mapping, scene lengths (min/max), merge or split scenes, timed timeline notes | Lyrics + Scenes step, Line Mapping, timeline notes |
| 2. Cast | Characters and extras, images or RefMods, describe | Reference Builder Subjects tab, RefMods Quick Add |
| 3. Story | Write or draft the book, chapter settings, add cast found in the story | New |
| 4. Acts | Chapters and acts drawn as bands over a mini timeline, drag borders, merge/split | New |
| 5. Scenes | Scene plan table: act, slice, spot, performance, Director / Video / Planning Notes | New, plus scene card note fields |
| 6. Places | Settings with their look, spot cards with images or RefMods, scene mapping | Reference Builder Locations tab |
| 7. Beats & Prompts | Scene beats, image prep, video prep, prompt all | Current Storyboard Builder |

Layout rules:

- A tab shows a short "what's missing" line and stays usable, but the next tab's main action is disabled
  with a reason until its input exists (no Acts without a story, no Places without a scene plan).
- A thin progress bar across the top shows each tab as empty, in progress, or done.
- A right-hand side panel shows the selected scene everywhere: time, lyric, act, slice, spot, notes. Edits
  there save to the same scene fields.
- The header keeps only the actions of the open tab. Today's header holds 9 actions at once
  (`storyboard_layout.mjs:82`).
- The Video Builder timeline gets an act/chapter band above the scenes in Story Mode, so the story shows
  where you render.

Reuse, not copies:

- The Song, Cast and Places tabs mount the existing panels (line mapping, Reference Builder subject and
  location panels). They are not rewritten. Where a panel is tied to the Reference Builder window, it is
  split into a mountable part and the Reference Builder keeps using the same part.
- All tabs read and write the same session keys and go through the existing save, revision and undo path,
  so the Video Builder and the Agent API see the same project.

---

## 3. Data (session keys)

Everything stays inside existing session objects. No new top-level keys.

- `builder_story_layer` (existing) gains:
  - `story_flow_version`: `2` for projects on this flow
  - `story_book`: `{title, premise, chapters: [{id, title, setting_id, paragraphs: [str]}]}`
  - `story_book_text`: the editable plain-text form, parsed into `story_book` on save
  - `story_settings`: `[{id, name, look}]`, the general setting shared by a chapter's spots
  - `story_acts`: `[{id, chapter_id, title, summary, cast_ids, scene_ids}]`
  - `story_scene_plan`: `[{scene_id, act_id, story_slice, spot, mood, note_sources}]`
- `video_type` (existing): new value `story`. Per-scene `performance_mode` (existing) holds singing or
  story-only for each scene.
  - `overall_story_idea`, `song_story_brief`, `lyric_story_strength` keep their meaning
  - `user_story_arc` stays for old projects
- `flux_reference_builder.subjects[]` (existing): cards added from the story get `story_origin: true`.
- `flux_reference_builder.locations[]` (existing): one card per spot, gaining `story_setting_id` and
  `story_act_ids`. The mapping goes to the existing `location_scene_map`. Cards are named
  `Setting - Spot` so the Reference Builder groups them.
- Scene notes stay where they are today (`timeline_note`, `i2v_notes`, `notes`, `timeline_markers`).
- Storyboard card: `story_beat` stays. The act title and `story_slice` show read-only on the card.

---

## 4. Todo list

### Phase 1: Prompts and services (Python)

- **A1.** `llm/prompts/storyboard.py`: `_storyboard_story_book_instruction` (premise, cast, scene count,
  song length, seed) and `_storyboard_story_book_chapter_rewrite_instruction`.
- **A1b.** `llm/prompts/storyboard.py`: `_storyboard_story_split_instruction` plus JSON schema for a pasted
  story. It returns chapter borders and a setting per chapter without changing the user's text. Code
  checks the chapters joined back together equal the pasted text.
- **A2.** `llm/prompts/storyboard.py`: `_storyboard_story_cast_scan_instruction` that lists people in the
  book who have no cast card, with a drafted description each.
- **A3.** `llm/prompts/storyboard.py`: `_storyboard_story_acts_instruction` plus JSON schema. Inputs: one
  chapter, its setting, its scene range, timed notes in that range, song energy per scene.
- **A4.** `llm/prompts/storyboard.py`: `_storyboard_story_scene_plan_instruction` plus JSON schema. Inputs:
  one act, its setting, its scenes, minimum scenes per spot, each scene's Director / Video / Planning Notes
  and overlapping timed notes, each scene's lyric line labelled "mood and emotion only".
- **A5.** `llm/prompts/location_scout.py`: `_STORY_SETTING_INSTRUCTIONS` (one look per setting) and
  `_STORY_SPOT_INSTRUCTIONS` (spot cards that inherit the setting's look), with schemas. No generic place
  lists, and no lyrics in the input.
- **A5b.** `storyboard/story_layer.py`: rewrite `_lyric_story_strength_guidance` for v2 so every level
  talks about mood and emotion only, and add a hard rule to the story, act and plan prompts: lyrics are
  literal or metaphorical and never name a place or an event.
- **A6.** `storyboard/story_layer.py`: `_build_story_book`, `_scan_story_cast`, `_build_story_acts`,
  `_build_story_scene_plan`, and `parse_story_book_text`.
- **A7.** New `storyboard/story_acts.py`: pure code for act validation (scene count sums to N, contiguous,
  ordered, inside one chapter, timed notes inside their act), spot run validation (minimum scenes per spot
  unless a note moves it), act border moves, merge and split. Fallback split by paragraph count when the
  LLM fails twice.
- **A8.** `llm/image_prompt_generation.py`: `_generate_story_settings` and `_generate_story_spots`, which
  return spot cards with setting, act and scene ids.
- **A9.** `storyboard/story_layer.py` scene beat builder: on v2 feed slice, act summary and notes instead of
  `user_story_arc`. On v2 nothing reads the arc ↔ brief loop (F3).
- **A10.** `storyboard/story_layer.py`: `convert_story_arc_to_book` for old projects (D2).

### Phase 2: Routes and Agent API

- **A11.** `storyboard/nodes.py`: `/vrgdg/storyboard/story_book`, `story_book_chapter`, `story_cast_scan`,
  `story_acts`, `story_scene_plan`, `story_convert`. LLM calls through `asyncio.to_thread`.
- **A12.** `builder/routes.py`: `/vrgdg/music_builder/story_locations`.
- **A13.** `agent_api/orchestrator/storyboard_orchestrator.py`: steps `book`, `acts`, `plan` for
  `POST /projects/{pid}/story/{step}`, plus act edit routes (move border, merge, split).
- **A14.** `agent_api/orchestrator/reference_orchestrator.py` and router: `reference.story_locations` job
  that writes the scene mapping, and cast-from-story upsert.
- **A15.** Descriptions in `scripts/export_api_endpoints.py`, run `export_openapi.py` and
  `export_api_endpoints.py`, check `mcp_server/tools.py` still calls real routes.

### Phase 3: Frontend

- **A16a.** Done. Story Mode video type: `story` option, `normalizeProjectVideoType` where the project type
  is stored, `normalizeVideoType` returns `singing` for it. Contract test fails if a storing site uses
  `normalizeVideoType`.
- **A16b.** Per-scene performance in Story Mode: the scene plan proposes `performance_mode` (`singing`,
  `speaking`, `no_lip_sync`) and marks instrumental scenes with the existing `lyric_instrumental`. The
  Scenes tab and timeline let the user change it. A `speaking` scene needs its spoken line and speaker,
  using the existing `scene_dialogue` / speaker fields. Check that prompt and audio code handle a
  `speaking` scene inside a Singing-path project.
- **A16c.** New `web/storyboard_builder/story_builder_layout.mjs`: the tabbed Story Builder shell, progress
  bar, per-tab header actions, selected-scene side panel. Opens instead of the Storyboard shell in Story
  Mode.
- **A16d.** Make the line mapping, Reference Builder subject panel and location panel mountable inside the
  Story Builder, without changing how they work in the Reference Builder.
- **A16e.** Act/chapter band on the Video Builder timeline in Story Mode (`timeline_view.mjs`).
- **A16.** `wizard.mjs`: in Story Mode the wizard runs Settings → Audio → Lyrics + Scenes → Mode → opens
  the Story Builder → Gemma + Render. Other video types keep today's steps.
- **A17.** New `web/music_video_builder/story_book.mjs`: book editor, Draft / Expand chapter / Reroll,
  Add Cast Member, and the "people without a card" checklist. JS twin of `parse_story_book_text`.
- **A18.** New `web/music_video_builder/story_acts.mjs`: act bands drawn over the timeline (one colour per
  chapter), drag borders, scene count field, merge/split, regenerate one chapter's acts.
- **A19.** New `web/storyboard_builder/story_plan.mjs`: scene plan table (scene, time, lyric, act, slice,
  notes used) with inline edit.
- **A20.** `web/storyboard_builder/story_layer.mjs` and `story_layer_layout.mjs`: on v2 show Book → Acts →
  Plan → Locations → Beats in place of the section arc box. On v1 show **Convert to Story Book**.
- **A21.** `reference_builder.mjs` / `reference_locations.mjs`: **Build Locations From Story** as the main
  location action on v2, disabled with a message until the scene plan exists. Location cards are grouped
  under their setting, with an editable setting look above each group. LM Extract moves under "Other".
- **A22.** `wizard_bridge.mjs`, `storyboard_bridge.mjs`, `gpt_payload.mjs`: carry the new story layer keys.

### Phase 4: Docs and tests

- **A23.** Tests: book text parsing (Python and JS twin cases), act validation (sum to N, timed notes kept
  inside their act, merge/split), 46-scene per-chapter plan, notes overriding slices, story locations
  writing `location_scene_map`, cast added from story, v1 project loads unchanged, v1 → v2 conversion.
- **A24.** `MUSIC_VIDEO_AGENT_PROMPT.md`: cast → book → acts → plan → story locations → beats, and
  `story_idea` becomes required.
- **A25.** `AGENT_GUIDE.md`: storyboard section and new story layer keys.
- **A26.** Run the node registration test and the full unit test suite.

---

## 5. Decisions

- **D1. Cast before Story, more cast during Story.** Confirmed.
- **D2. Old projects keep the current flow; Convert to Story Book moves them to v2.** Confirmed.
- **D3. Story → Chapter → Act → Scene.** The LLM proposes acts and slices, code validates them, and the
  user's notes override generated text. Confirmed.
- **D4. Places follow the tree.** A chapter has one general setting (Home). Each scene stands at one
  specific spot inside it (Home - Kitchen). Spots in a setting share its look. Scenes per location
  becomes the minimum number of scenes at one spot. Confirmed.
- **D5. Lyrics drive mood and emotion only.** They never decide a place or a story event, and they are not
  sent to the location steps at all. Lyric Story Strength now sets only how strongly a scene's emotion
  follows its lyric line (0 ignore, 5 balanced, 10 close). Default 5. Confirmed.
- **D6. Story Mode is a video type.** It uses the Singing audio and render path, with performance chosen
  per scene: singing, speaking, story only, or instrumental. Confirmed.
- **D7. Story Builder is a tabbed workspace in Story Mode.** It brings line mapping, cast and places tools
  into one flow. Existing panels are mounted, not copied. Other video types keep the current Storyboard
  Builder. Confirmed.

## 6. Open questions

- **Q1.** Answered: yes. A pasted story is kept word for word. The LLM only splits it into chapters,
  proposes each chapter's setting, then builds acts and scenes from it as usual.
- **Q2.** Answered: no for now. One setting per chapter, and a change of setting starts a new chapter.
- **Q3.** Answered: Speaking mode keeps its own script flow and is not changed.
- **Q4.** Answered: a Story Mode scene can be `singing`, `speaking` (a spoken line over the music),
  `no_lip_sync` (story only) or instrumental.
- **Q5.** Answered: only Story Mode gets the tabbed layout for now.

## 7. Risks

- **R1.** Unknown video types fall back to `singing` silently. A save that used `normalizeVideoType` would
  turn a Story Mode project back into Singing. A16a's contract test covers this.
- **R2.** Reference Builder and line mapping panels are built inside closures with many dependencies
  (`reference_builder.mjs` passes about 40 functions between modules). Making them mountable (A16d) is the
  largest frontend task and should be done before the tabs.
- **R3.** Two windows can edit the same scene (Story Builder and Video Builder). Both already use the
  revision check, so the second save asks before overwriting.

## 8. Implementation stages

Work happens on the `newbuilder` branch. Each stage is its own commits, passes the full unit test suite,
and leaves Singing, Speaking and No Lip Sync projects unchanged.

| Stage | Tasks | Done when |
|-------|-------|-----------|
| S1. Story Mode type | A16a | A Story Mode project renders exactly like Singing. The video type contract test passes. |
| S2. The book | A1, A1b, A2, A6 (book part), A11/A13 (book routes) | Through the API: draft a book from a premise, split a pasted story with the text unchanged, list story characters without a card. |
| S3. Acts and scene plan | A3, A4, A5b, A7, A16b, rest of A6/A11/A13 | On a 20-scene and a 46-scene project every scene is covered in order, timed notes and Director/Video Notes are respected, each scene has a spot and a performance, no place comes from the lyrics. |
| S4. Settings and spots | A5, A8, A12, A14, A15 | Locations grouped by setting, mapped to scenes in `location_scene_map`, and they open correctly in the current Reference Builder. |
| S5. End to end without new UI | A9, A24 | A full Story Mode video made through MCP: book, acts, plan, locations, beats, prompts, render, stitch. Main checkpoint. |
| S6. Mountable panels | A16d | Line mapping, cast and location panels work in a test container and in the Reference Builder as before. |
| S7. Story Builder UI | A16c, A16e, A16, A17-A22 | The S5 video can be made entirely from the Story Builder tabs. |
| S8. Old projects and docs | A10, A23, A25, A26 | Convert to Story Book works on an old project, docs updated, full suite passes. |

S5 proves the story logic before the UI work starts, which is the largest risk (R2).
