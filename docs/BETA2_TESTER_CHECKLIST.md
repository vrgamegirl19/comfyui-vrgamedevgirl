# Beta2.0 tester checklist

**Interactive form:** [Report a Beta2.0 feature test](https://form.typeform.com/to/c55P0jKl). Choose the feature ID, mark whether it passed, failed, or could not be tested, and submit one form response per feature. The form asks for the exact error and reproduction steps only when you mark **Fail**. The test steps below are your reference while filling it out.

If you prefer to work offline, use one copy of this document per tester and project and replace the blanks as you go. For every test, mark **Pass**, **Fail**, or **Not tested**. If it fails, paste the **exact error message** (or write “no message”), say what happened, and attach a screenshot or log when useful. “Not tested” is appropriate when a model, GPU, service, or account is unavailable; say which prerequisite is missing.

This covers the user-facing Beta2.0 changes merged through PR #239. It is a hands-on checklist, not a requirement to render every model on every machine. Use a **copy of a small project** (about 3–6 scenes) for tests that alter or delete media. Do not include API keys, private paths, or personal media in a shared report.

## Tester and setup

- Tester name or handle: __________
- Date and time: __________
- Beta2.0 commit/build shown in Builder: __________
- ComfyUI version and installation type: __________
- OS, browser, and browser version: __________
- GPU and VRAM / system RAM: __________
- Video engines and models installed: __________
- LLM runner used (if any): __________
- Project type and number of scenes: __________
- Link to project copy or sanitized sample, if shareable: __________

For each case, fill in all three lines. If a case fails, also add its ID to the [issue details](#issue-details) section at the end with reproduction steps. For a test with multiple bullets, mark Pass only when all listed behavior works.

## 1. Opening, saving, and layout

**UI-01 — Open the Builder.** Add/open the VRGDG AI Video Builder UI node, then open an existing project. The Builder opens maximized and the project is usable.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**UI-02 — RAM/VRAM at the top.** Verify the RAM/VRAM display is in the top toolbar, to the right of the action buttons, and updates while the Builder is open. Try a narrower window if possible.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**UI-03 — Save and in-Builder refresh.** Change a harmless project setting, click Quick Save, then click the refresh icon immediately to its right. The Builder reloads into the same project and returns to the selected scene, panels, and scroll position without leaving you at the node graph.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**UI-04 — Refresh when saving fails.** If you can safely reproduce a save error, click Refresh Builder UI. The error is shown and the page does not reload or discard the unsaved change.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**UI-05 — UI layout profiles.** Save a layout with a collapsed side panel and a taller timeline, switch layouts, and reopen the Builder. The chosen layout returns; a new project preselects the last-used layout.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**UI-06 — Floating LLM window.** Pop out LLM Prompting, move the window, edit a prompt or continuation direction, and return to the main panel. Both views show the same content.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**UI-07 — Waveform selector.** Open the waveform size/dropdown near the timeline. It fits its option text instead of spanning most of the toolbar; changing the option still updates the waveform.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

## 2. Wizard Beta and Reference Builder

**WIZ-01 — Wizard Beta route.** Open Wizard Beta. Move through Engine, Video mode, Models & LoRAs, Sound & timing, Inputs, Scenes, and Render. Each step opens, Back/Next work, and saving restores your place.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**WIZ-02 — Models page.** In MiniMax mode, check the Video Models, Audio Model, pass controls, and Video Settings tabs. Change a harmless value, leave the page, and return. The value persists and the normal Builder controls still work after closing Wizard Beta.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**WIZ-03 — RefMod choice.** Select MiniMax H3 → Reference to Video on the Video mode step. Check **Use RefMod workflow**. Open Reference Builder: subject/location cards offer saved RefMods and the guidance mentions RefMods. Uncheck it: the Standard image-reference workflow returns. Selecting another video mode also returns to Standard.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**WIZ-04 — RefMod Inputs.** With RefMod on, open Inputs. It directs you to Edit Subjects/Locations rather than asking for image uploads. Choose a saved RefMod and proceed. If none is selected, preparation should explain what is missing.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**WIZ-05 — Mapping tools.** In Scenes → Edit Mappings, open **Map Subjects / Locations**. Try the available bulk assignment and location extraction/Auto Map controls. Return, then open **Adjust & View All Mappings**; both routes reflect the same saved mappings.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**WIZ-06 — Storyboard from Wizard.** Open Story Layer from Wizard Beta, change story text or mappings, close it, then reopen the standalone Storyboard Builder. Changes are shared; prompts use the same scene data.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**REF-01 — Subject and location cards.** Add or edit a subject and a location with names/descriptions, map them to scenes, save, and reopen. Text, images or RefMods, and mappings remain attached to the intended cards.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**REF-02 — Text-only location.** Create a location with a title but no image, map it to a scene, save, and reopen. It remains selectable and counts as a mapped location for story creation.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

## 3. Storyboard and prompt creation

**STO-01 — Story Layer first.** Open the standalone Storyboard Builder. Story Layer is the first and initially active section; the story controls are visible without visiting Wizard Legacy.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**STO-02 — No mapped location guard.** In a test project with no scene-location mappings, try creating the story arc, brief, and beats. Creation stops with a clear “no locations mapped yet” message. A location in the library alone is insufficient.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**STO-03 — Combined story action.** Map a location to a scene, then click **Create Arc → Brief → Missing Beats**. Arc, brief, and only missing beats are created in order. Reopen the project to check saved results.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**STO-04 — Stop on a stage error.** If a runner error can be safely triggered, run the combined story action. It reports the failing stage and does not continue to later stages or pretend they succeeded.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**STO-05 — Per-scene cast.** Map different characters to different scenes, generate an arc, beats, and a video prompt. A character omitted from a scene must not be named in that scene’s text or prompt.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**STO-06 — GPT copy/import flow.** Use GPT Image All or GPT Video All (or selected-scenes GPT) in Storyboard. The Copy JSON / Continue to GPT / Continue to Import dialog opens. Import a response with `scene_number` and `video_prompt`; the prompt lands on the matching scene and stays saved.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**STO-07 — Storyboard scene card.** Open a scene card from the timeline, review/edit scene notes and prompt, save, and return. The edited scene stays selected and its prompt appears in the Builder.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

## 4. Timeline, media, and scene actions

**TIM-01 — Location image thumbnail.** Map a location with an image to a scene and enable location thumbnails on the timeline. That scene shows the mapped location image and title.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**TIM-02 — Location text fallback.** Map a text-only location to another scene. With location thumbnails enabled, its timeline card shows the location title rather than a broken or empty image.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**TIM-03 — Automatic continuous shots.** In Reference Builder, check **Auto continuous shots for shared locations** and apply mapping to adjacent scenes: 1–3 at Location A, 4–6 at Location B, and 7–8 at different locations. Each same-location run should form its own continuous group; scenes 7–8 must not join. Undo once and check that mapping and continuity revert together.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**TIM-04 — Continuity boundary.** Add a time gap between two same-location scenes, apply automatic continuity, and inspect their settings. The gap starts a new shot. A change of location also starts a new shot.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**TIM-05 — Timeline Tools window.** Open the movable Timeline Tools window; drag it, use Set In/Out, Clear Range, Close Gaps, Snap Scene Edge, Overlay Track, and Locations, then close it. Controls work and the window does not block the main timeline after closing.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**TIM-06 — Empty Delete All menu.** In a project with no scene images or videos, check the Delete All control. It is hidden when there is nothing eligible to delete; individual bulk actions appear only after matching content exists.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**TIM-07 — Scene right-click menu.** Right-click a scene with no media, one with an image, and one with a video. **Delete image** appears only for an image. **Delete video** and **Use frame as image** appear only for a video. **Delete scene** is not in this menu; the scene × still offers scene removal.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**TIM-08 — Delete confirmation and cancel.** On a disposable scene/project copy, try scene, image, video, and bulk deletion. Cancel/Escape/outside click leave content intact. Confirm deletes only the chosen content; Delete All Videos offers image removal as a separate unchecked choice.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**TIM-09 — Render status and selection.** Start a short render, close/reopen render status, and select scenes with Ctrl-click. Rendering/selection indicators identify the intended scenes. Try the right-click stitch-preview action on rendered scenes.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**TIM-10 — Preview joins.** Stitch two short rendered scenes and inspect the join frame by frame. Neither a repeated nor missing frame should be visible at the boundary, and clip frame counts remain consistent.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

## 5. MiniMax H3 rendering and continuity (model/GPU dependent)

**MM-01 — Video profiles.** Save a MiniMax video profile, apply it to another scene, and reopen the Builder. Video type/pass/settings restore. Audio mode, continuity, and Standard/RefMod project pipeline do not unexpectedly change. Deleting a profile asks for confirmation.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**MM-02 — LoRA target.** Add a MiniMax LoRA with no prior target. It defaults to **Pass 1 only**. Change it to Pass 2/Both, save, and reopen; the selected target remains.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**MM-03 — Shared output resolution.** Change Single, 2 Pass, and 2 Pass Advanced modes. One output-resolution preset remains selected across modes. A new project starts at the documented 1K preset; an older saved project opens with an appropriate migrated resolution.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**MM-04 — Advanced tiling.** In 2 Pass Advanced, select 8/12/16/24 GB VRAM presets and a supported output size. The automatic tile plan renders without exposing the retired manual tile controls. The output matches the chosen resolution.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**MM-05 — 2 Pass output history.** Render a disposable 2 Pass or Advanced scene. The timeline shows the final video as one take; temporary Pass 1 scratch video does not appear as an extra saved take.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**MM-06 — Built-in audio.** Render MiniMax H3 2 Pass with built-in audio and no source audio. Audio is present in the final clip. In 2 Pass Advanced, the UI still requires input audio where that mode needs it.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**MM-07 — Masked continuation quick toggle.** On scene 2 or later, use the small chain control on its timeline card. It locks the scene and selects Masked latent continuation/transition. Toggle it off and verify the previous settings return.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**MM-08 — Masked render boundary.** Render two compatible scenes with Masked continuation. Scene 2 begins from the previous scene’s ending motion without a regenerated or visibly repeated head; differing render sizes are handled. Note the engine/pass used. Image + Reference and 2 Pass Advanced are excluded from this test.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**MM-09 — Continuation direction.** Enter a scene-specific continuation direction in LLM Prompting, adjust **Direction starts at**, generate a continued-scene prompt, and reopen the project. The prompt uses the preceding scene’s final frame and respects the direction timing and scene length; the text stays saved.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**MM-10 — RefMod render.** In a small RefMod Reference-to-Video project, choose saved RefMods for mapped cards, generate a prompt, and render one scene. The prompt names the chosen references and the renderer uses the selected RefMods; switching to Standard retains any card images.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

## 6. Agent API / MCP (optional for technical testers)

Use [Api_Endpoints.md](../Api_Endpoints.md) for the exact requests. Run these against a disposable project and record the HTTP status and response body for failures. Skip this section if your beta test is UI-only.

**API-01 — Project read and revision.** List/open a project through `/vrgdg/api/v1`, request selected top-level fields with `include`, and try an unknown field. Valid fields return; an unknown field is rejected. A stale `If-Match` revision does not overwrite newer work.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error / HTTP response: __________
- Notes / request ID / log: __________

**API-02 — Lyrics and timeline replacement.** Create scenes from timed lyrics, then replace their timing through the API. Lyrics carry to overlapping new scenes instead of disappearing; the response reports how many scenes retain lyrics.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error / HTTP response: __________
- Notes / request ID / log: __________

**API-03 — Scene edits and flags.** PATCH a scene’s `lyric_text`, `no_character_present`, `lyric_no_lip_sync`, and `lyric_singers`. Read the scene again. Values persist; unsupported fields or invalid types get a validation error rather than a false success.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error / HTTP response: __________
- Notes / request ID / log: __________

**API-04 — Mapping and references.** Read/write scene mapping and list/choose a scene’s MiniMax references. Mapped subjects/locations and matching use-reference switches agree with the Builder UI after reload.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error / HTTP response: __________
- Notes / request ID / log: __________

**API-05 — Story and prompt jobs.** Run the API story arc/brief/beats and prompt pipeline on a mapped project. The job reports progress, saved prompts include the expected MiniMax reference definitions/soundscape, and the Builder shows the same results.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error / HTTP response: __________
- Notes / request ID / log: __________

**API-06 — Render and takes.** Render one scene through the API, check job progress, then list scene video takes. The final take appears once and matches the Builder timeline. For continued scenes, test masked-continuation fields and prompt context if your model setup supports them.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error / HTTP response: __________
- Notes / request ID / log: __________

**API-07 — Full song pipeline / MCP.** If you use an agent, connect through the supplied MCP instructions or call the Agent API directly. Build a small song project through timing, references, story, prompts, render, and final video. Record the first stage that differs from doing the same task in the Builder.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error / HTTP response: __________
- Notes / request ID / log: __________

## 7. Installation and regression checks

**REG-01 — Node-pack startup.** Restart ComfyUI with Beta2.0. The expected VRGDG nodes and Builder open without a missing-module startup error. Record any failed VRGDG submodules from the startup log.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**REG-02 — Stem extraction.** If you use `VRGDG_GetStems`, run it once. Demucs loads and the node returns stems; if an unrelated custom node shadows it, the error identifies the actual import failure.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**REG-03 — Existing project.** Open a project made before Beta2.0. Check scenes, lyrics, media paths, references, MiniMax mode/pass/resolution, and saved prompts. Save a copy, reopen it, and verify nothing unexpected changed.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

**REG-04 — New small project.** Create a new 3–6 scene project, save and reopen at each stage: timing → references → story → prompts → one render → preview stitch. Record the first stage that fails.
- Result: [ ] Pass [ ] Fail [ ] Not tested
- Exact error message: __________
- Notes / screenshot / log: __________

## Issue details

Copy this block once for **each failed case**. Keep one case per report so developers can reproduce it.

### Failed case: [ID]

- Severity: [ ] Blocks project [ ] Feature broken [ ] Visual/cosmetic
- Feature and test ID: __________
- Expected result: __________
- Actual result: __________
- Exact error message (copy/paste, or “no message”): __________
- Steps to reproduce from a fresh/saved project:
  1. __________
  2. __________
  3. __________
- Does it happen every time? __________
- Project state before the test (scene count, engine, mode, pass, media present): __________
- Screenshot, short recording, sanitized log, or project copy link: __________
- Workaround found, if any: __________

## Final tester summary

- Cases passed: _____  Failed: _____  Not tested: _____
- Biggest blocker: __________
- Most useful improvement: __________
- Confusing label or workflow: __________
- Other feedback: __________
