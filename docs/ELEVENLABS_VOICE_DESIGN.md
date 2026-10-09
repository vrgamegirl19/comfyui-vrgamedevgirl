# ElevenLabs Voice Design

Voice Design is available on primary character cards in **Reference Builder**, only when **Speaking video mode** is enabled. Enter an ElevenLabs key in **Builder Settings → ElevenLabs**. The existing **Save API Key to Project** button remains the explicit way to persist it.

## Local testing

Restart ComfyUI and refresh the browser after updating the backend/frontend. Restart the MCP server to pick up the generated endpoint tools.

1. Open a Speaking project, then Reference Builder and a character card. Click **Design Voice**; it is available even before enabling a voice assignment.
2. Describe the voice. Click **Help Write Voice Description** to use the currently selected LLM Runner with the character description. Confirm the output remains editable and makes no ElevenLabs request. With LM Studio, only an already loaded model may be used.
3. Review/edit the description (20–1000 characters). Alternatively, write it directly without LLM help. Choose Voice Design v3 or v2. Leave sample dialogue blank or provide 100–1000 characters.
4. Click **Generate Voice Previews**. This uses ElevenLabs account credits. Listen and choose a preview. Editing the description, model or sample dialogue clears the candidates and requires regeneration.
5. Set a voice name and click **Save Voice to ElevenLabs & Assign**. This creates a reusable account voice and enables its assignment to the character. Click **Save** in Reference Builder to persist the assignment and design draft to the project.
6. Reopen the project. Confirm the voice ID and editor fields survive; generated audio and temporary preview IDs must not appear in the session file.
7. Confirm Design Voice and the rest of the ElevenLabs UI disappear in other video modes. Close the editor or change projects during a request: late responses must not update the new editor/project.
8. Test missing/restricted keys, unavailable LLM settings, expired previews and account voice limits. Errors must be useful and must not expose credentials. After an ambiguous account-save failure, refresh account voices before saving again, since the provider may have created the voice already.

A key needs access to Voice Generation for previews and Voices write for saving; Voices read is used by the existing voice picker. Provider limits and account plans still apply. This step does not generate scene dialogue or alter timeline durations.

## API and MCP

All three endpoints require Speaking mode. The ElevenLabs endpoints use the explicitly saved project key.

- `POST /projects/{pid}/references/subjects/{rid}/voice-design/description`: body `{"user_input":"A warm, calm storyteller with a soft rasp"}`. Returns a background job; wait for its result containing `voice_description`. Uses saved LLM Runner settings and the existing character, without writing project state.
- `POST /projects/{pid}/elevenlabs/voice-design/previews`: body `voice_description`, optional `model_id` (`eleven_ttv_v3` default or `eleven_multilingual_ttv_v2`), optional `text`. Returns base64 MP3 candidates with temporary `generated_voice_id` values.
- `POST /projects/{pid}/elevenlabs/voice-design/create`: body `voice_name`, reviewed `voice_description`, chosen `generated_voice_id`. Returns permanent `voice.voice_id` and `voice.name`. This mutates the ElevenLabs account, without changing the project.

Assign the permanent voice using the existing `PUT /projects/{pid}/references/subjects/{rid}` with `If-Match` and `elevenlabs_voice: {enabled:true,voice_id:"...",name:"..."}`. That endpoint also accepts `elevenlabs_voice_design` with the five durable fields above; omitting either field preserves its existing value. Separate account creation from project assignment so a revision conflict cannot lose the permanent voice ID or trigger duplicate account creation.

The generated `agent_api/endpoints.json` exposes these routes to MCP. `Api_Endpoints.md` and OpenAPI document request fields and behavior.

Provider references: [Design previews](https://elevenlabs.io/docs/api-reference/text-to-voice/design), [Create a designed voice](https://elevenlabs.io/docs/api-reference/text-to-voice/create).

## Suggested PR description

Add Speaking-only Voice Design to Reference Builder character cards. Users can write a description or use their selected LLM Runner, review ElevenLabs previews, then explicitly save and assign a reusable voice. Persist only editor drafts and permanent voice assignments; keep preview audio and temporary IDs out of project files.

Add matching API/MCP endpoints for description jobs, previews and account creation, with provider validation, bounded responses, safe errors and no automatic creation retries. Validation covers provider contracts, selected runner behavior, mode guards, draft preservation and the browser review/assignment flow.

Automated validation: 51 focused tests passed. The full suite ran 1,095 tests: 1,090 passed; five existing environment failures require the Krea2/MiniMax model files or a Node version providing `node:module.register`. Provider requests are mocked in automated tests; perform the local account preview/save checks above before merging.
