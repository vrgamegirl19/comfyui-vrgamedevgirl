# Story Mode v3: Story, Acts, Chapters, Scenes

Follow-up to `new_builder.md`. Fixes two reported problems and reworks the story levels.

- Characters drop out of the story and the shots before the end (multiple user reports).
- With a shot frequency set, the cuts inside a scene do not hold one moment together.

## Findings

- **F1. Scenes never get their own cast.** The scene plan decides the slice, spot, mood and performance, not
  who is in the scene (`storyboard/story_plan.py:328`). `fillSceneCast` copies an act's whole cast onto its
  scenes, and only onto scenes with no mapping yet, so a singer mapped by Line Mapping blocks everyone else.
- **F2. Nothing keeps every character to the end.** The book prompt only lists names. The music video rule
  makes the first cast card "the performer" (`llm/prompts/story_mode.py:76`). Nothing checks later acts.
- **F3. No relationships.** Cards have a name and a description. "Love interest" cannot reach any prompt.
- **F4. Shot cuts are not anchored.** The cut plan only asks for a "continuity-preserving camera/shot cut"
  (`minimax/shot_prompt.py:73`, `web/music_video_builder/minimax_prompt.mjs:185`). Later shots are not told
  to start from the previous shot's last moment, so each shot becomes its own small scene.
- **F5. The levels are upside down.** Today: Story → Chapter (one setting) → Act (event) → Scene.
  Wanted: Story → Act → Chapter → Scene.

## Decisions

| | Decision |
|---|---|
| D1 | Levels become Story → Act → Chapter → Scene. Old projects convert. |
| D2 | The user picks the number of acts. The Acts tab explains: an act is a big part of the story and holds several chapters; a chapter is one place where several scenes happen. |
| D3 | Character presence per act and per chapter: absent, present or featured. |
| D4 | Settings live on chapters, so an act can move between places. |
| D5 | One main character, in most or every scene. The story uses only the cast the user picked: no new people, no scan for missing characters, no "allow new people" option. |
| D6 | Keep today's shot frequency and cut behaviour, and add a choice for one continuous moment across the cuts. |
| D7 | Phase 1 ships first. |
| D8 | Story Mode gets its own LLM instructions, separate from every other mode, editable per project. Changing them never changes another mode. |

## Phase 1: fixes on the current levels

### P1.1 Continuous cuts (F4, D6)

- New MiniMax setting next to the cut frequency: **Shot cuts: Free (as now) / One continuous moment**.
  Default Free, so existing projects do not change.
- One continuous moment:
  - The prompt opens with the continuity paragraph: one continuous performance; the same position, body
    orientation and posture across every cut; each shot continues from the last physical moment of the
    shot before; only the camera angle and framing change.
  - The LLM is told that each later shot starts from the previous shot's end state (same spot, wardrobe, light,
    props). Action may progress (a step, a gesture) but never jumps in time or place.
  - Later shots are written as "At 00:02.000, the camera cuts to …, continuing directly from …".
- Python (`minimax/shot_prompt.py`) and browser (`minimax_prompt.mjs`) versions change together, with tests.

### P1.2 Main character and strict cast (F2, D5)

- Cast tab:
  - **Main character**: one pick. They lead the story and appear in most or every scene. Replaces "first card".
  - **Role** per character: main, love interest, supporting, antagonist.
  - **Relationships**: pairs with a type and a short note ("Mara ↔ Joe: love interest, estranged").
- Removed: Scan For Missing Characters, the "allow new people" option and its prompt branch.
- Every story prompt gets the roles and relationships, and the rule that each character's arc runs to the
  ending. Only the picked cast may appear.

### P1.3 Cast per scene (F1, D5)

- The scene plan returns, per scene: `cast` (names from the cast), `performer`, and `interaction` (one line of
  what the characters do with each other).
- The main character is in the scene unless the story clearly puts them elsewhere.
- The scene's characters are written to `subject_scene_map`, replacing a mapping the story set before or a
  main-character-only mapping (Line Mapping). A mapping the user changed by hand is kept.
- Scenes tab: cast chips per scene to edit, and the interaction line.
- Beats and video prompts receive the interaction and the relationships.

## Phase 2: separate Story Mode instructions (D8)

- Story Mode instruction keys in the existing instruction system (`llm/builder_instructions.py`): book, acts,
  chapters, scene plan, setting looks, locations, beats, and one per MiniMax video mode.
- Defaults live in their own module (`llm/prompts/story_mode_*.py`), copied from today's text, so other modes
  keep theirs.
- Each instruction is split into an editable part (rules, style) and a fixed part the code fills (scenes,
  cast, timings, JSON format), so edits cannot break the parsing.
- Story Builder gets an Instructions panel: view, edit for this project, reset, save and load presets.
- Story Mode runs (including Beats & Prompts from the Story Builder) pass only these keys.

## Phase 3: Story → Acts → Chapters → Scenes (D1–D4)

Tabs (8):

1. **Song**: unchanged.
2. **Cast**: main character, roles, relationships (P1.2).
3. **Story**: premise, synopsis, and one line per character: who they are, what changes, how it ends.
4. **Acts**: the user picks how many. Each act: title, purpose (setup, turn, climax, resolution), story text,
   scene range, and a presence grid that the LLM proposes and the user clicks to change.

   ```
               Act 1      Act 2      Act 3
               sc 1-10    sc 11-22   sc 23-31
   Mara        ★ featured ★          ★
   Joe         · present  ★          ★
   Lena        – absent   ·          ·
   ```

   "Expand Act" rewrites an act's text to match its grid.
5. **Chapters** (new): each act split into chapters. A suggested count from the act's scenes (about one chapter
   per 3-5 scenes, never shorter than the spot run), which the user can change. Each chapter: title, one
   setting, its events, its cast (from the act's present characters), scene range, and a grid per act.
6. **Scenes**: slice, spot, mood, performance, cast chips, performer, interaction, continuation, scene audio.
7. **Places**: settings come from chapters.
8. **Beats & Prompts**: uses the scene cast, interaction and relationships.

Coverage check after every step (code, not the LLM), each warning with a "Fix with LLM" button:

- A character marked present in an act has no chapter in it.
- A character in a chapter has no scene in it.
- The last act leaves out the main character or a main role.

Data (story flow version 3):

- `story_book`: title, premise, synopsis, character arcs.
- `story_acts`: id, title, purpose, text, scene ids, presence `{cast_id: absent|present|featured}`.
- `story_chapters`: id, act id, title, setting id, text, cast ids, scene ids, presence.
- `story_scene_plan`: adds cast ids, performer id, interaction.
- `story_main_cast_id`, `story_cast_roles`, `story_relationships`.

Conversion from version 2, no loss: each old chapter (one setting, its prose) becomes an act holding one
chapter per old act; each old act becomes a chapter with the old chapter's setting.

Out-of-date tracking runs Story → Acts → Chapters → Scenes → Places → Beats → Prompts.

## Phase 4: agent API, MCP and docs

Routes and tools for the new steps, the Python versions of the data rules, `Api_Endpoints.md` and
`agent_api/endpoints.json` regenerated, `AGENT_GUIDE.md` and `new_builder.md` updated.
