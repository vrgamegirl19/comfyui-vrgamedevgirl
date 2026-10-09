# RefMod Pipeline: Integration Spec (draft for review)

Status: scoping only. No code has been changed for this spec. Round 4: your answers to Q1 to Q9 are folded in (section 10 lists them). Section 11 holds the open token-optimization design.
Goal: a separate RefMod pipeline inside the Video Builder, switched on per project, that renders scenes from saved RefMods (and imported images) instead of reference images, while leaving the standard pipeline untouched.

## 1. Scope

In scope
- A project-level switch: Standard pipeline or RefMod pipeline.
- RefMod render path: RefMod to Video, single pass or 2 pass.
- Reference Builder, Storyboard Builder, prompt writing, singer assignment and video settings (LoRAs included) working with RefMods.
- RefMods Studio changes needed to feed all of the above (more types, thumbnails).
- Agent API and MCP parity.

Out of scope for this spec
- I2V, Image + Ref 2 Pass, V2V and 2 Pass Advanced in the RefMod pipeline (removed on purpose).
- LTX. The RefMod pipeline is a MiniMax H3 pipeline.
- Voice, singing, music_style, sound_fx, ambience RefMod types (stay greyed out in the Studio).

## 2. What exists today (verified in the code)

| Area | How it works now | Where |
| --- | --- | --- |
| Engine switch | `session.video_engine` is `ltx` or `minimax_h3`. Badge in the top bar, select in Project Settings. 69 JS and 24 Python places test `=== "minimax_h3"`. | `controls.mjs` `normalizeProjectVideoEngine`, `session.mjs`, `project_setup.mjs`, `model_settings.mjs`, `agent_api/mutations.py` |
| MiniMax modes | `text_to_video`, `image_to_video`, `image_reference_to_video`, `reference_to_video`, `video_to_video` in `session.minimax_h3_settings.video_mode`. | `minimax_h3.mjs` `MINIMAX_H3_MODE_OPTIONS` |
| Passes | `render_pass`: `single`, `two_pass`, `three_pass` (three_pass is the button labelled 2 Pass Advanced). | `minimax_panel_layout.mjs` (`passMode`), `minimax/settings_payload.py` |
| Panel rules | `minimax_panel.mjs` hides and shows sampler, Easy Cache, Turbo, LoRA targets and reference settings per mode and pass. | `minimax_panel.mjs` around line 655 |
| Graph building | The runner loads an API template JSON and patches it by node id (115 resolution, 119/120 VAEs, 126 guider, 128 clip, 129 noise, 138 prompt, 141 model, 142 save, 171 audio, 172 audio drive, 180 reference media). | `runner/minimax_workflows.py`, `runner/minimax_inputs.py`, templates in `Workflows/UsedForUIDoNotTouch/minimax_*_api.json` |
| Patches | LoRAs, Turbo, TE-Speed, feed-forward chunking, block-sparse attention, fast VAE decode, latent save and continuation are patch functions keyed on node class or id. | `runner/minimax_patches.py` |
| Render payload | JS builds it in `video_render.mjs`. Python twin: `minimax/settings_payload.py` (settings) and `minimax/scene_inputs.py` (reference images, last frame, video refs). Both are checked by parity tests. | listed |
| References | `session.flux_reference_builder`: `subjects`, `extra_subjects`, `locations`, `ingredients_sheets` plus scene maps `subject_scene_map`, `extra_scene_map`, `scene_map`, `ingredients_scene_map`. Reference types today: character, prop, object, vehicle, creature, outfit, style, environment, other. | `reference_data.mjs`, `reference_builder.mjs`, `minimax_references.mjs` (`miniMaxReferenceBuilderCatalog`), `prompt_text.mjs` |
| Descriptions | "Gemma Describe" reads an image with the vision LLM and fills `description`. | `reference_subjects.mjs`, `llm_runner.mjs` |
| Prompt labels | The LLM writes `<Subject n> (Sn)` and `<Picture n>` with definition lines built from the reference list. Prompt budget is 7,000 characters. | `minimax_prompt.mjs`, `llm/prompts/minimax.py`, `minimax/prompt_assembly.py` |
| Singer assignment | Only `reference_type == character` items can be speakers. Cues are stored in `segment.minimax_speaker_assignments`. | `minimax_speaker_assignments.mjs` |
| Storyboard | Scenes carry `subject_refs` by name. GPT payloads use subject names and a trigger phrase. | `web/storyboard_builder/references.mjs`, `gpt_payload.mjs`, `storyboard/scene_prompts.py` |
| RefMod library route | `GET /refmods/library` already returns name (with subfolder), kind, concept, description, tokens and shape for every saved mod. No tensors are loaded. | `ComfyUI-MiniMaxH3Mod/library.py` |

## 3. What we learned about RefMods (drives the design)

1. A RefMod is stored as `image` (1 latent frame) or `video` (2+ frames). Stacked photos become `video`. Label type follows `kind`, not what the source was.
2. References only bind to prompt text when the text encoder is shown them with labels. The Text Encode with RefMods node gives `<Picture n>` or `<Video n>` per mod, counted per kind, in loader order, skipping strength 0. Apply H3 RefMod alone adds the latents with no labels.
3. Without labels, three characters collapsed into two Brads and no Darrel. The labeled path is untested end to end and is a gating check (phase 0).
4. Mods differ in size (Brad 22 frames and 5,632 tokens, Darrel 6 frames). Big mods dominate. A token readout per scene is needed.
5. The Text Encode node takes mods only. It replaces the reference conditioning, so reference audio and raw reference images do not pass through it.
6. The RefMod description lives in the file, so it can replace Gemma Describe for any saved mod.

## 4. Decisions

Each decision has a recommendation. Items that need your call are in section 10.

### D1. How the switch works (decided: whole project)
Add `pipeline` (`standard` or `refmod`) inside `session.minimax_h3_settings`. Keep `video_engine = minimax_h3`.
- Why not a third engine value: 93 existing engine checks would need an audit, and the Agent API, orchestrators and wizard all branch on `minimax_h3`.
- It is a whole-project switch. When it is `refmod`, every scene renders through the RefMod pipeline. No per-scene pipeline choice and the scene override for `pipeline` is hidden.
- Set from the MiniMax panel header (two buttons: Standard, RefMod) and from Project Settings. The top-bar badge shows `MiniMax` or `MiniMax RefMod`.
- Switching never deletes data. Each reference item keeps both its image and its RefMod fields, so switching back is safe.

### D2. Modes and passes in the RefMod pipeline
Right-hand panel, RefMod pipeline:
- Mode row: one mode, `RefMod to Video`. T2V, I2V, Image + Ref 2 Pass and V2V are hidden. Stored as `video_mode = reference_to_video` so existing code paths keep working.
- Pass row: Single, 2 Pass. 2 Pass Advanced is hidden.
- Reused sections: Output resolution, Seed, Audio mode, Continuity, LoRAs, Advanced model patches, Singer Assignment, Prompt.
- Hidden sections: Scene image use, start frame character influence, last frame, video references.

Settings visibility (RefMod pipeline):

| Setting group | Single | 2 Pass |
| --- | --- | --- |
| Resolution, aspect ratio, seed | shown | shown |
| Sampler, scheduler, steps | shown | per-pass fields (existing) |
| Easy Cache | hidden (reference_to_video already bypasses it) | hidden |
| Turbo LoRA (single) | hidden | n/a. The 2 pass Turbo LoRA stays required and pass 2 only |
| Extra LoRAs (use_loras, count, target pass) | shown, no target column | shown with target pass 1, 2 or both |
| Latent upscale settings | n/a | shown |
| TE-Speed, feed-forward, block-sparse, fast decode | shown | shown |
| Continuity (off, latent continuation) | shown | shown |
| `ref_image_size` | hidden | hidden (no images go to the reference node) |

### D3. Everything becomes a labeled mod
One conditioning path: Text Encode with RefMods. Label for each mod is `<Video n>` or `<Picture n>` by `kind`, numbered per kind in loader order.
- The Builder computes labels before it writes the prompt, from the scene's ordered mods and their saved `kind`. The same function runs on the server so the render never disagrees with the prompt.
- The run shows the node's `reference_map` in the render log, and the Builder warns if the prompt uses a label the render does not produce (the existing `miniMaxPromptReferenceMismatch` check, extended).

### D4. Images in the RefMod pipeline (decided: convert once, save permanently)
Each reference item is either `source: refmod` or `source: image`. An image card is converted into a real, saved RefMod the first time it is needed.
- The card needs a name and a type. Type comes from the card: character goes to `identity`, environment and location to `background`, outfit to `clothing_men` or `clothing_women` (the card's clothing set), style to `style`, object, prop, vehicle and creature to their own folders.
- The mod is saved to `models/refmods/<type>/<name>.safetensors` using the card's name (for example a location named "Alpine Meadow" becomes `models/refmods/background/Alpine Meadow.safetensors`). The same preview PNG is saved beside it (the source image itself, scaled, so no decode is needed).
- After saving, the card flips to `source: refmod` and points at the new file. Every later scene and every later project uses the saved file. Nothing is re-encoded.
- Name clash: if a mod with that name exists in that folder, the Builder asks before replacing it, or lets the user pick another name. It never overwrites silently.
- Time: one VAE encode per image, once, at conversion. Saved RefMods cost no encode at render time. The conversion can run when the user presses "Save as RefMod" on the card, or automatically before the first render that needs it (with a progress note).
- Description: image cards keep Gemma Describe. The text is written into the new mod's description field at conversion, so the file carries it from then on.
- Single-image identity: the Studio asks for front, left and right views. A character converted from one image has one frame and is image-kind. The Builder shows a note that three views give a stronger identity.
- Scene start images are not used in this pipeline (no I2V).
- Separate cost to measure in phase 0: the Text Encode node decodes every mod back to pictures for the text encoder on every render. With 3 characters plus a background this adds a decode per mod per scene. Mitigation if it is slow: cache the decoded presentation frames per mod file.

### D5. Audio and video type (decided: full music video support in version 1)
Music videos must work from the first release, so the audio path is not dropped. Version 1 here means the first release of the RefMod pipeline we ship.
- Input audio (singing and no lip sync): the scene audio still drives the audio latent through the existing audio drive node (`VRGDG_MiniMaxH3AudioDrive`, node 172). In addition, the scene audio is turned into an audio-kind mod on the fly and passed to Text Encode, so the text encoder sees `<Audio 1>` and the prompt rules that name it keep working (single performer, multi-performer cues, `<d>[English] line</d>`).
  - Needs the `VRGDG RefMod Combine` node to merge the visual mods with the audio mod, and the pack's audio extract node (audio plus audio VAE, unsaved).
  - The audio mod is built per scene from the trimmed scene clip. It is small and is cached per clip.
- Built-in audio (speaking, short film): the existing native-audio conversion (`_use_minimax_h3_native_audio`) is applied to the RefMod templates. No scene audio is sent. Character voice presets from the Reference Builder keep working because character cards keep `minimax_voice`.
- Video type is respected exactly as today: Singing (music video) uses singer assignment and lyric cues, Speaking uses speaker assignment and built-in audio, No lip sync sends no cues and no speaker rules. All three are checked in phase 0 and phase 4.
- Phase 0 must compare lip-sync on one singing scene between the standard pipeline and the RefMod pipeline. If the audio mod does not give parity, we revisit this decision before building UI.

### D6. Types, folders and how they map (decided: one folder per type)
Folder under `models/refmods` is always the type (already built). Final type list:

| RefMods Studio type (folder) | Reference Builder type it fills | Builder tab | Describe prompt |
| --- | --- | --- | --- |
| identity | character | Subjects | identity (done) |
| clothing_men | outfit | Subjects | clothing (done) |
| clothing_women | outfit | Subjects | clothing (done) |
| background | environment, and Locations tab items | Locations | background (done) |
| style | style | Subjects (style row) | style (done) |
| pose_motion | other | Subjects | pose_motion (done) |
| generic | other | Subjects | generic (done) |
| object (new) | object | Subjects | new: object |
| prop (new) | prop | Subjects | new: prop |
| vehicle (new) | vehicle | Subjects | new: vehicle |
| creature (new) | creature | Subjects | new: creature |

- Old `clothing` is replaced by `clothing_men` and `clothing_women`. Nothing was saved under `clothing` yet.
- No gender field. Characters are named, and the name is the label. Clothing set (men or women) is chosen on the clothing card, not inferred from the character.
- The Builder `reference_type` stays the single source for a card's type. The RefMod type filters the dropdown. The mapping table lives in one JS module and one Python twin.
- New types need only: the Studio type list, one describe prompt each and one folder each. The mod pack stores whatever `concept_type` we pass.

### D7. Reference Builder behaviour
In the RefMod pipeline each card shows:
1. Source toggle: RefMod or Image.
2. RefMod source: Type dropdown, then a RefMod dropdown filtered to that folder. Each option shows a thumbnail, kind, frame count and token count.
3. Picking a mod fills: name (editable), description (from the mod file, editable), kind, tokens. No Gemma Describe and no Generate Image buttons on RefMod cards.
4. Image source: unchanged card (upload, generate, Gemma Describe), plus "Save as RefMod".
5. Strength (0 to 1) per card, set by the user with a slider and number box (default 1), applied in every scene the card is mapped to. No per-scene override in version 1.
6. A running token total for the selected scene, with a warning above a limit set after phase 0.
Locations tab: background mods and background images, same two sources.
Scene mapping tabs: unchanged. They map card ids to scenes, so the existing maps (`subject_scene_map`, `scene_map`, `extra_scene_map`) are reused as they are.

Clothing cards (decided: men and women folders, user chooses)
- A clothing card has a Clothing set selector: Men, Women or All. Default is All. The RefMod dropdown lists the matching folder (or both for All). A user can put a dress on a man by choosing Women or All.
- A clothing card is mapped to scenes like any card and can optionally be tied to one character (`wears` = the character card id). That tie sets the label order (clothing follows its character) and the description sentence ("<Video 1> (Brad) wears <Video 3> (charcoal suit)").
- A clothing card with no tied character is allowed. It then uses order-only mapping.
- A tied clothing card follows its character into every scene that character is mapped to, unless the user changes the clothing for a scene (scene-level change replaces the clothing for that scene only).

Thumbnails (decided)
- Each mod gets a preview PNG stored next to it as `<name>.preview.png`. It is made by decoding the first latent frame with the video VAE (the same result as Inspect H3 RefMod with index 0 and first-frame preview), then scaled down to about 384 px.
- It is created when the Studio makes a mod. For older mods the Reference Builder asks the backend for missing previews one at a time. The result is cached on disk and only regenerated if the safetensors file is newer than the PNG, so opening the Builder again costs nothing.
- Preview jobs share the GPU lock with renders and never run during a render.

### D8. Label order
Deterministic order for a scene, used by the Builder, Storyboard and server:
1. Characters (mapped order), 2. Extras that send to MiniMax, 3. Clothing, 4. Objects, props, vehicles, creatures, 5. Background, 6. Style.
Then labels are counted per kind (`<Video n>` and `<Picture n>` independently) in that order, skipping strength 0. This mirrors `reference_map` in the mod pack.
Implementation: one Python module `minimax/refmod_scene.py` and one JS module `refmod_labels.mjs` with parity tests, like `scene_inputs.py` and `minimax_references.mjs`.

### D9. Prompt writing
- New LLM preset in `llm/prompts/minimax.py`: `MINIMAX_H3_REFMOD_TO_VIDEO_MODE`. It tells the model the cast list as `<Video 1> (Brad)`, `<Picture 1> (location)`, and that appearance comes from the references.
- Cast list lines include the mod description from the file, so the LLM does not invent looks. For identity mods the LLM is told not to restate clothing and hair unless a garment moves (same rule as now).
- Definition lines and the mismatch check in `minimax_prompt.mjs` get a RefMod branch.
- Character budget (7,000) stays.

### D10. Singer and speaker assignment
- Speakers are identity-mod (character) cards. `miniMaxMappedSpeakersForSegment` already filters on `reference_type == character`, so it works once RefMod cards carry that type.
- Cue text refers to the speaker by label: `<Video 1> (Brad) sings <d>[English] line</d>`.
- Multi-performer rule that names `<Audio 1>` is replaced in this pipeline by a rule that names the speaker label only (until D5 option returns).

### D11. Storyboard Builder
- Reference cards in the storyboard show a RefMod badge, type, and the label that scene would get (`<Video 1> Brad`), computed with the D8 function.
- GPT payload `subject_refs` gains `refmod_label`, `kind`, `refmod_type`, and the mod description replaces any Gemma text.
- Scene prompts (`storyboard/scene_prompts.py`) refer to subjects as `<Video n> Name` in the RefMod pipeline and as before in the standard one.
- The storyboard stays a planning tool. Labels there are previews. The Builder recomputes them at render time.

### D12. Render path
New runner builders and templates, following Recipe 3 in AGENT_GUIDE:
- `_build_minimax_h3_refmod_api_prompt` (single) and `_build_minimax_h3_refmod_2pass_api_prompt`.
- Routes `/vrgdg/workflow_runner/build_minimax_h3_refmod_prompt` and `..._refmod_2pass_prompt`.
- Templates `minimax_refmod_1pass_api.json` and `minimax_refmod_2pass_api.json` in `Workflows/UsedForUIDoNotTouch`, derived from the draft workflows already made. They keep node ids 115, 119, 120, 123 to 129, 131, 132, 138, 141, 142, 171, 172 so the existing patch functions work unchanged. The reference node 136 becomes an Empty MiniMax H3 AV Latent. The loader, optional image-to-mod node and Text Encode node get fixed ids.
- Payload additions: `pipeline`, `refmod_references` (ordered list of `{source, mod_name or image_path, strength, label}`).
- Validation: every mod name must exist (like `_require_model_choice`), total count and token cap checked, clear errors.
- Model patches (LoRAs, Turbo for 2 pass, TE-Speed, feed-forward, sparse attention, fast decode, latent continuation) reused as they are.

### D13. Agent API and MCP
- New setting `pipeline` in the settings schema, defaults JSON and enum (`export_minimax_defaults.mjs`).
- `GET /vrgdg/api/v1/refmods` (list with type filter, wraps the same data as `/refmods/library`) and `GET /vrgdg/api/v1/refmods/{name}`.
- Reference items accept the RefMod fields. Same session keys as the UI (no parallel keys).
- `modes` catalog gains the RefMod pipeline entry.
- `Api_Endpoints.md`, `endpoints.json`, OpenAPI export, MCP named tools and contract tests updated per CLAUDE.md.

## 5. Data model

Settings
- `session.minimax_h3_settings.pipeline`: `standard` (default) or `refmod`.

Reference item (subjects, extras, locations; new optional fields)
```
source: "image" | "refmod"            // default "image"
refmod: {
  name: "identity/darrel",             // as shown by /refmods/library
  folder: "identity",
  type: "identity",
  kind: "video" | "image",             // copied from the file at pick time
  tokens: 4608,
  frames: 6,
  strength: 1.0                        // set by the user per card
}
clothing_set: "men" | "women" | "all"  // clothing cards only, default "all"
wears: "<character card id>"           // clothing cards only, optional
```
`description` stays the single text field. For RefMod cards it is filled from the file and editable. Existing fields and scene maps are unchanged.

Render payload (new)
```
pipeline: "refmod"
refmod_references: [ { source, mod_name | image_path, strength, label } ]   // already ordered
```

## 6. Code to add or change

New backend
- Node `VRGDG RefMod From Images`: image paths in, one image-kind mod per image out (unsaved).
- Node `VRGDG RefMod Combine`: merges mod lists (needed to mix saved mods, image-derived mods and the scene audio mod).
- Audio-mod cache per scene clip.
- Route `POST /vrgdg/refmod/preview`: decode the first latent frame, save `<name>.preview.png`, serialized behind the GPU lock.
- `minimax/refmod_scene.py`: ordering, labels, token total, validation (pure Python).
- Route `GET /vrgdg/refmod/library_ex`: library plus type, folder, frame count and thumbnail url, filtered by type. Wraps the pack's data.
- Thumbnails: RefMods Studio saves a small preview PNG next to each mod at create time. Old mods fall back to a placeholder with a "Generate preview" action.
- Runner builders, templates, routes (D12). `settings_payload.py` and `scene_inputs.py` twins.
- `llm/prompts/minimax.py` RefMod preset. Describe prompts for object, prop, vehicle, creature.

New frontend
- `refmod_labels.mjs` (twin of `refmod_scene.py`).
- `refmod_library.mjs`: fetch and cache the library, thumbnails, type filter.
- Pipeline switch in `minimax_panel_layout.mjs` and `minimax_panel.mjs`; mode and pass rows per D2; badge text in `model_settings.mjs`.
- RefMod card variant in `reference_subjects.mjs`, `reference_locations.mjs`, `reference_builder.mjs`; data normalizers in `reference_data.mjs`; catalog in `minimax_references.mjs`.
- Prompt branch in `minimax_prompt.mjs`; render payload in `video_render.mjs`.
- Storyboard: `web/storyboard_builder/references.mjs`, `gpt_payload.mjs`, and `storyboard/scene_prompts.py`.
- RefMods Studio: new type list (clothing_men, clothing_women, object, prop, vehicle, creature), thumbnail save, `kind` shown in the success message.
- Convert-image-to-RefMod action (card button and automatic before render): route `POST /vrgdg/refmod/from_image` that saves the mod, preview and description into `models/refmods/<type>/`.

Not changed
- LTX code, wizard LTX paths, post-processing, timeline, latent manager.

## 7. Phases

| Phase | Work | Exit check |
| --- | --- | --- |
| P0 | Validate labeled Text Encode on the 3-character test (`refmod_1pass_labeled.json`). Add the scene audio mod and compare lip-sync with the standard pipeline on one singing scene. Measure per-scene time (mod decode, image encode). Decide the token warning limit. | Three distinct characters in one scene, lip-sync at parity, time per scene acceptable. If not, revisit D3 to D5 before building anything else. |
| P1 | Backend core: `refmod_scene.py`, Combine and From Images nodes, audio mod, templates (input audio and built-in audio), runner builders, routes, caches, payload twins, tests. | A scene renders from a hand-written payload through the API. |
| P2 | Switch, panel rules, settings schema, defaults export, Agent API setting. | Toggle swaps the right-hand options. Standard pipeline unchanged. |
| P3 | Reference Builder RefMod cards, library route, thumbnails and preview cache, clothing set selector and character tie, image-to-RefMod conversion, mapping, token readout. | Pick three mods plus clothing, map to a scene, see labels and tokens. |
| P4 | Prompt preset, definition lines, mismatch check, singer assignment. | LLM prompt uses `<Video n>` labels that match the render. |
| P5 | Storyboard badges, payload, scene prompts. | Storyboard shows labels, GPT payload carries RefMod data. |
| P6 | New Studio types (clothing split, object, prop, vehicle, creature), describe prompts, Agent API/MCP endpoints, docs, parity and contract tests. | Full test suite plus generated docs current. |

## 8. Tests

- Python: `refmod_scene` ordering and labels (including mixed kinds, strength 0, images), payload twin parity, runner builders (node ids present, patches applied), validation errors, library route.
- JS (`tests/*.cjs`): label parity with Python, pipeline switch visibility rules, RefMod card normalizers, prompt mismatch check.
- Contract: settings enum, defaults JSON, OpenAPI, endpoints doc, MCP tools (CLAUDE.md rules).
- Manual: 3-character scene in single and 2 pass, standard pipeline regression, project saved in one pipeline opened in the other.

## 9. Risks

| Id | Risk | Mitigation |
| --- | --- | --- |
| R1 | Labeled path may not fix duplicate characters. | Phase 0 gate before any UI work. |
| R2 | Big mods swamp small ones. | Token readout, per-card strength, guidance for equal frame counts when creating mods. |
| R3 | The scene audio mod does not give lip-sync parity with the standard pipeline. | Phase 0 comparison on a singing scene. Fallback: keep the audio drive only and drop the `<Audio 1>` rule for single-performer scenes. |
| R4 | Label drift between prompt time and render time. | One ordering function in Python and JS with parity tests, plus render-time `reference_map` check. |
| R5 | RefMod files renamed or deleted after mapping. | Validate at render, show a missing-mod state on the card. |
| R6 | New concept types are not in the mod pack's own combo. | Only matters for nodes created in ComfyUI directly. Studio is the main path. |
| R7 | VRAM when encoding images and loading many mods. | Cap items per scene, log tokens, reuse the pack's mod cache. |
| R8 | Text Encode decodes every mod to pictures on every render, which could slow each scene. | Measure in phase 0. Cache the decoded frames per mod file if needed. |
| R9 | Converting a single image gives a weak identity compared with the three-view Studio flow. | Note on the card, and a link to open RefMods Studio for that character. |

## 10. Decisions

Answered
1. Q1: Whole project switches. All scenes use the RefMod pipeline when it is on.
2. Q2: Images are converted to labeled mods once, saved permanently in the right folder under the name the user gave, and reused everywhere after (D4).
3. Q3: Separate folders for object, prop, vehicle, creature (D6).
4. Q4: Music video audio and the video types (singing, speaking, no lip sync) are supported in the first release (D5).
5. Q5: Clothing split into men and women folders. The user picks the set on the clothing card (Men, Women, All) (D7).
6. Q6: First release = first version we ship. Each card has a user-set strength. No per-scene override yet.
7. Q7: Previews come from decoding the first latent frame and are cached as PNG files next to each mod (D7).
8. Q8: No gender field. Characters are identified by name only.

9. Q9: A tied clothing card follows its character into every scene unless the user changes it for a scene.

Decided
- Q10: Warn, never block. The scene status shows the RefMod token total and warns above 6,000 tokens (a four character scene at 5,030 rendered fine). It also warns when one character weighs less than half of the strongest (tokens x strength): in a four character test the two single-image mods (520 and 676 tokens) were repeatedly duplicated next to two multi-frame mods (2,394 and 1,440), and lowering the heavy mods' strength fixed it. Quality profiles are the Studio's Quality presets (see the Studio section), and each character stays one label.

## 11. Token optimization when creating a RefMod (draft, under discussion)

Why: reference tokens are paid at every sampling step of every scene. Three characters at today's sizes are about 13,000 tokens against about 23,000 for a 6 second scene at 1024x576.

### Facts from the code
- Tokens per RefMod = frames x (canvas width / 32) x (canvas height / 32). Spatial size is 16 px per latent cell and the model groups 2x2 cells.
- All images in one mod share one canvas. The canvas comes from the first image and `ref_resolution` (short edge 1024, down only). Every other image is center-cropped to that canvas (cover crop), not fitted. A tall full-body image added after a square front close-up loses its head and feet.
- When a mod is over `max_tokens` (the Studio sets 5,120) the pack drops near-duplicate latent frames, then keeps evenly spaced frames. It does not know which frames are the required front, left and right views, so it can drop the left or right close-up.
- Real examples: Brad is 22 frames at 512x512 (256 tokens per frame, 5,632). Darrel is 6 frames at 1024x768 (768 per frame, 4,608). Different shapes of the same budget.
- Compressed Reference (pooling) cuts tokens by 4x to 16x but the pack warns it removes the detail a face needs.
- Strength does not reduce tokens. Only canvas size and frame count do.

### Tokens per frame by canvas
| Canvas | Tokens per frame |
| --- | --- |
| 1024 x 1024 | 1,024 |
| 1024 x 768 | 768 |
| 768 x 768 | 576 |
| 640 x 768 | 480 |
| 512 x 768 | 384 |
| 512 x 512 | 256 |

### Levers (what costs detail and what does not)
| Id | Lever | Saves | Detail risk |
| --- | --- | --- | --- |
| L1 | Tight crop to the subject (face close-ups cropped to the face, canvas matches the crop) | 50 to 70 percent | None. Same face pixels, fewer wasted background tokens. |
| L2 | Remove near-duplicate images before encoding | one frame each | None. The pack already treats them as free to drop. |
| L3 | Smaller canvas | 44 percent (768) or 75 percent (512) vs 1024 square | Some. Needs a test per type. Backgrounds and style tolerate it better than faces. |
| L4 | Fewer frames (best N of the set) | proportional | Some. Keep the front, left, right views and the most different extras. |
| L5 | Per-type size profile | varies | Low. Faces get more pixels, backgrounds and style get fewer. |
| L6 | Split a character into a face member and a body member in one bundle file, each with its own canvas | 30 to 50 percent | Low, but each member gets its own label (`<Video 1>` face, `<Video 2>` body), so the prompt and Builder handle two labels for one character. |
| L7 | Compressed Reference | 75 to 94 percent | High for faces. Fine for style or background after a test. |

### Proposed Studio features (not built)
1. Live token meter: shows frames x canvas = tokens for the current images, and the share of the scene budget (reads image sizes in the browser, no GPU).
2. Smart crop: crop close-up slots to the face with the OpenCV face detector already used in `post_process/face_fix.py`, then fit to one canvas. Bulk images are fitted without cutting heads (pad or fit instead of center crop).
3. Duplicate finder: flags near-identical images before create and offers to drop them.
4. Size profiles per type instead of raw numbers: Compact, Balanced, Max. Profile values are fixed in code (users still do not see resolution or pool values) and come from the experiment below.
5. Keep-required rule: front, left and right are never removed by the token cap. The Studio chooses which extras to keep, then sends the final list.
6. Optional two-member character (L6) if the experiment shows it pays off.
7. A mod written by the Studio records its canvas, frame count and profile in its tags so the Builder can show them.

### Experiment (do before fixing the profiles)
Use one character (Darrel) and the same seed, scene and prompt in the 3-character test.
- Variants: canvas 512, 768, 1024 (square or matching aspect) x frames 3, 6 (front, left, right, plus extras).
- Add two crop variants: face-tight crops vs the current framing at the same token count.
- Record for each: tokens, time per scene (single and 2 pass), VRAM, and a side-by-side judgement of the face and clothing.
- Output: a table of the smallest profile per type that looks the same as Max. That table becomes the Compact, Balanced and Max values.

### Lab results (Darrel set, RTX 5090, VAE round trip)
Set: 3 head close-ups (about 600 x 660) and 3 full-body shots (about 330 to 530 x 1180). Detail score is PSNR after removing the decoder's colour cast, so only compare numbers within a row group.

| Variant | Frames | Canvas | Tokens | Detail score (dB) |
| --- | --- | --- | --- | --- |
| A. Today's Studio flow (6 images, cover crop to first image) | 6 | 608x672 | 2,394 | 21.4 to 25.0 |
| B. Close-ups, native | 3 | 608x672 | 1,197 | 22.0 to 22.5 |
| B. Close-ups, short edge 512 | 3 | 512x544 | 816 | 21.5 to 22.0 |
| B. Close-ups, short edge 384 | 3 | 384x416 | 468 | 20.4 to 20.8 |
| B. Close-ups, short edge 256 | 3 | 256x288 | 216 | 19.1 to 19.4 |
| C. Close-ups, tight crop, native | 3 | 608x608 | 1,083 | 21.5 to 22.0 |
| D. Body, tight crop and pad, long edge 1184 | 3 | 384x1184 | 1,332 | 20.2 to 22.0 |
| D. Body, long edge 960 | 3 | 320x960 | 900 | 19.3 to 21.2 |
| D. Body, long edge 768 | 3 | 256x768 | 576 | 18.5 to 20.4 |
| D. Body, long edge 640 | 3 | 224x640 | 420 | 17.9 to 20.0 |
| E. Face inside the full-body shot, long edge 1184 | 1 | 512x1184 | 592 | face only: 18.4 |
| E. Same, long edge 640 | 1 | 288x640 | 180 | face only: 15.9 |

Findings
- L-F1: Today's flow ruins the three full-body images. The first close-up sets a 608x672 canvas, and the tall images are cover-cropped to it, so the model gets torso and hand crops with no head, legs or shoes (see `lab_out/A_baseline.png`).
- L-F2: Close-ups hold up very well when small. At 384 px short edge (468 tokens for three frames) the decoded faces look the same as native by eye. Detail only falls clearly below about 320.
- L-F3: Full-body shots cannot carry the face. The face is 160 px wide at best and its score is 18.4 dB at full size, 15.9 at 640. Use the close-ups for the face and the body shots for outfit.
- L-F4: Body shots need padding, not cropping. Tight crop plus white padding keeps head to shoes and costs 1,332 tokens at full size.
- L-F5: Separate face and body sets cost about the same as today's flow at full size (1,083 + 1,332 = 2,415 against 2,394) and keep all the information. Reduced sets: Balanced 768 + 900 = 1,668 (30 percent less than today), Compact 432 + 576 = 1,008 (58 percent less).
- L-F6: One shared canvas for all six images is the wasteful option. Padding everything to a 640x960 portrait canvas would cost about 3,600 tokens.
- L-F7: Encode time is small. Six images take about 0.4 s on the 5090 once the VAE is loaded, so converting images to mods (Q2) costs almost nothing.
- L-F8: The decoder tints a white background slightly pink. Identity detail is not affected, but the tint is part of what the model is given.

Test mods written for A/B renders (delete the `lab` folder when finished): `models/refmods/lab/`
- `darrel_today` 2,394 tokens
- `darrel_face_full` 1,083, `darrel_body_full` 1,332
- `darrel_face_balanced` 768, `darrel_body_balanced` 900
- `darrel_face_compact` 432, `darrel_body_compact` 576
Workflows: `Workflows/refmod_lab_darrel_today.json`, `refmod_lab_darrel_split_balanced.json`, `refmod_lab_darrel_split_compact.json`.

### Lab follow-up: one file or two (Darrel set)
Renders of today, split balanced and split compact all looked the same, so the question became which is cheapest, not which looks best.
- Splitting does not save tokens by itself. At matching sizes the two files add up to the same as one file (face 1,083 + body 1,332 = 2,415 against 2,394 for today's single file). Every frame costs its own canvas area, and a white margin costs as much as a face.
- Savings come from canvas size and from not paying for margins. A single file with all six images padded or cropped to one square canvas: 512 = 1,536 tokens, 448 = 1,176, 384 = 864 (frames x (side / 32)^2).
- The split only helps when the images have different shapes (square close-ups and tall bodies), because a single canvas then wastes tokens on padding or loses image content to cropping.
- Extra test mods: `lab/darrel_single_512`, `darrel_single_448`, `darrel_single_384`, with workflows `refmod_lab_darrel_single_*.json` (one label, one file).

### Auto-trim and background removal (discussed)
- Auto-trim (find the subject's bounding box and cut everything outside it, with a small margin) saves tokens, because tokens follow canvas area. It is the same crop used in the lab (variants C and D).
- Transparent background does not save tokens. The video VAE reads RGB only, so transparent pixels become a solid colour and still fill the full canvas. Only the canvas width and height and the frame count change the token count. The pack's mask option (`background_retention`) blurs the area outside a mask but also leaves the token count unchanged.
- What background removal is good for: (1) trimming photos with busy backgrounds, where "cut the white margin" finds nothing, because the cut-out silhouette gives the bounding box; (2) keeping the old background from leaking into new scenes; (3) a flat neutral fill that decodes cleanly.
- Proposed Studio features: "Trim to subject" (default on; white or flat backgrounds use a colour threshold, other backgrounds use a person cut-out) and an optional "Clean background" that fills outside the cut-out with one flat colour. The cut-out model is an optional dependency, to be chosen when we build it.

### Studio quality presets and trim boxes (built)
- Quality presets replace raw canvas sizes. Each keeps a share of the original detail: Maximum 100%, High 80%, Balanced 60% (default), Compact 45%, Draft 30%. The canvas is the widest crop by the tallest crop at that share, never above 1024 px on the longest side, snapped to 32 and padded up to 320 px (the video VAE minimum). Values live in `QUALITY_PRESETS` (`refmod_trim.mjs`) and `QUALITY_SCALES` (`minimax/refmod_studio.py`).
- The Studio shows tokens for the current quality, with and without the trim boxes, and for every quality side by side. Create is blocked above 5,120 tokens.
- Trim boxes are found automatically (white or flat backgrounds), draggable by edge, corner or body, and sent to the server, which crops, scales and pads onto one canvas (padding uses each image's own border colour).
- Darrel set (6 images), tokens by quality: Maximum 3,072; High 2,700; Balanced 1,452; Compact 1,020; Draft 660. On this set auto-trim does not change the canvas, because the close-ups already fill the width and the body shots already fill the height.
- Known limit: a mixed set (square faces, tall bodies) pays for the widest-by-tallest canvas. A face set and a body set as separate files costs less at the same detail (lab: 1,008 to 1,668 tokens against 1,452 to 3,072 here). Splitting is still an option for later.

## 12. Implementation status (first testable milestone)

Built
- Setting `minimax_h3_settings.pipeline` (`standard` or `refmod`), project-wide, in the browser and in `minimax/settings_payload.py`. RefMod forces `reference_to_video`, Single or 2 Pass, and no spatial or exact-start-frame continuity. Defaults JSON, OpenAPI, endpoint docs and tests are current.
- Pipeline switch in the MiniMax panel (Standard, RefMod), badge text, mode row hidden, 2 Pass Advanced hidden, scene image controls hidden.
- Reference Builder: subject and location cards have a Source choice (image or saved RefMod), a RefMod picker filtered by type with preview, kind, frames and tokens, a strength slider, clothing set, "worn by" and "goes with this character", and "Save this image as a RefMod".
- Scene references: `refmod_labels.mjs` and `minimax/refmod_scene.py` (shared test cases) order the scene's RefMods and label them. The standard prompt writer is reused; every `<Picture n>` is swapped for the real label once, at assembly.
- Render: payload carries `pipeline` and `refmod_references`; `runner/minimax_refmod.py` rewires the built single or 2 pass graph (Text Encode with RefMods, scene audio as an audio RefMod, VRGDG RefMod Combine). Verified against the real builders and templates.
- Agent API: orchestrator renders RefMod scenes, `GET /vrgdg/api/v1/refmods`, modes catalog entry.
- RefMods Studio: types clothing_men, clothing_women, object, prop, vehicle, creature; preview PNG saved with every new RefMod; describe prompts for the new types.
- Library routes: `GET /vrgdg/refmod/library`, `GET /vrgdg/refmod/preview`, `POST /vrgdg/refmod/from_image`.

- Scene status line: RefMod token total, warning above 6,000 tokens, character balance warning (`tokenReport` in `refmod_labels.mjs`, twin `token_report` in `minimax/refmod_scene.py`). The render logs the same warnings and returns `token_report` in its summary.
- Scene clothing control in the MiniMax panel: removed. Clothing is chosen by the scene mapping (Review Lines + Map Performers) and each clothing card's "Worn by" link. A saved `segment.refmod_clothing_override` is ignored.
- Reference Builder "Save this image as a RefMod": quality choice and "Add more images" (extra images are uploaded and sent as `extra_images`; one image gives a light RefMod).
- Storyboard Builder (D11): RefMod fields survive the storyboard catalog and saved scenes (`refmodCardFields`, `_refmod_card_fields`), scene chips show a `◈ <Video n>` badge, and the GPT payload carries `refmod_label`, `refmod_kind`, `refmod_type` per subject plus a `refmod_pipeline` block (labels and how to write them) when the project uses the RefMod pipeline. Labels there are previews, the Video Builder works them out again at render.

Not built yet
- Auto-trim for the Reference Builder save (the Studio has trim boxes).
- Phase 0 checks on real renders: lip-sync parity with the standard pipeline and time per scene (decode cost per mod, R8). Three distinct characters work (scene 11 of Higher ground, four characters, after lowering the heavy mods).
