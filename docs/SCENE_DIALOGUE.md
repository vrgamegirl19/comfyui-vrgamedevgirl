# Scene dialogue with ElevenLabs

This feature appears only in **Speaking video mode**. It generates one character's speech per scene and uses the existing scene audio timeline/import logic.

## Using it

1. Enter your key in **Builder Settings → ElevenLabs**. **Save API Key to Project** makes it available after reopening the project and to API/MCP clients. Text to Speech permission is required for generation; the voice picker also uses Voices read.
2. In Reference Builder, assign and enable an ElevenLabs voice for your character, then save Reference Builder.
3. Right-click a scene and open **Audio Settings…**. The **ElevenLabs Scene Dialogue** panel contains the speaker, speech model, dialogue and LLM delivery instructions.
4. Choose a speaker and enter their words. Use **Help Craft Dialogue** to apply emotion/delivery tags through the selected LLM Runner. **Allow the LLM to rewrite the dialogue** is off by default. With it off, the helper rejects results that change the spoken words, while allowing tags and punctuation changes.
5. **Edit Dialogue LLM Instructions…** opens the existing instruction editor for this exact scene. Save an override for this scene or all project scenes, or save/load a shared preset. The key is `elevenlabs_dialogue`; voice-design instructions remain separate. LM Studio uses an already loaded model.
6. Review the tagged text, then click **Generate Speech**. This explicitly uses ElevenLabs credits with `eleven_v4` by default; `eleven_v3` is also available. No automatic retries or automatic LLM-to-provider submission occur.
7. Listen to the generated take. **Use Generated Speech** stores a preserved source file and stages it for this scene. **Save Scene Audio Settings** commits the audio attachment and dialogue draft together. With Fit enabled, scene duration becomes the actual audio duration plus opening/ending silence, and later scenes with attached audio move accordingly. Independent music/effects remain fixed.

For MiniMax video generation, select **Input Audio** in video settings to use the supplied speech. Built-in Audio uses MiniMax's own audio generation. Audio timing/replacement on an already rendered scene requires rendering it again.

Generated candidates remain temporary until used and saved. Editing the dialogue invalidates a candidate; editing a staged generated take's draft discards its pending attachment. Cancel leaves the timeline unchanged. Imported source files are preserved even if a save fails or the editor is canceled.

This step supports a single selected speaker per take. Automatic splitting at silence/word boundaries and a GPU-dependent maximum scene length are separate follow-up work. Keep dialogue within the duration limits of your selected video mode; this feature fits duration without splitting a long take.

## API / MCP

- `GET /projects/{pid}/scenes/{sid}/audio-settings` includes `dialogue`, the saved scene editor draft.
- `PATCH /projects/{pid}/scenes/{sid}/audio-settings` accepts `dialogue: {speaker_id,text,delivery,allow_rewrite,model_id}` alongside settings and optional audio import. It stores `scene_dialogue` on the same scene. Omit dialogue to preserve it. Use `If-Match` for project edits.
- `POST /projects/{pid}/scenes/{sid}/dialogue/craft` accepts an optional `dialogue` override, or uses the saved draft. Returns a job using saved LLM Runner settings and the selected character context. Wait for the job result, review `dialogue.text`, and save it if desired. Job parameters contain no API keys.
- `POST /projects/{pid}/scenes/{sid}/dialogue/generate` accepts an optional reviewed `dialogue` override, or uses the saved draft. Uses the saved ElevenLabs key and the character's enabled voice. Returns `audio_data`, `audio_name`, `voice_id` and `dialogue`, without writing project state. Pass the returned audio and draft to audio-settings PATCH to import and fit/ripple.

Models are `eleven_v4` (default, up to 10,000 characters) and `eleven_v3` (up to 5,000). Preview bytes and keys are excluded from persisted drafts. OpenAPI and `endpoints.json` expose the routes to MCP; restart the MCP server to reload its tool catalog.

Provider references: [Create speech](https://elevenlabs.io/docs/api-reference/text-to-speech/convert?explorer=true), [Audio tags for v3/v4](https://elevenlabs.io/docs/help-center/product/core-capabilities/text-to-speech/how-do-audio-tags-work-with-eleven-v3-and-v4).

## Local verification / PR notes

Restart ComfyUI and refresh the browser. Test LLM help, manual tag entry, instruction scene/all-scene overrides and shared presets, then generate a real take with your account key. Verify playback, actual duration, correct speaker/scene ownership, later-scene ripple, zero default silence and unchanged independent tracks. Reopen the project to verify draft persistence. Check disabled/unavailable voices, missing keys/permissions, canceled requests and project/mode changes. A 4-second scene with a 6-second take should become 6 seconds with zero silence and move the next scene right by 2 seconds.

Suggested PR: Add Speaking-only scene dialogue preparation and ElevenLabs speech generation to Audio Settings. Reuse the selected LLM Runner, existing instruction presets and scene audio import/fit/ripple services. Persist only the editor draft and imported source metadata; preview generated takes before committing them. Add matching MCP/API routes and provider/UI/persistence regression tests.

Automated provider calls are mocked. Focused checks cover provider requests, safe failures, word preservation, selected runner behavior, atomic instruction writes, correct scene imports, duration/ripple timing, and the full browser review/generate/stage/save flow.

Validation: 75 focused tests passed. The full suite ran 1,104 tests: 1,099 passed, with five existing environment failures from missing Krea2/MiniMax diffusion models and Node's unavailable `node:module.register` export. Live ElevenLabs generation and playback remain part of local account testing.
