# ElevenLabs keys and character voices (step 2)

Speaking video mode exposes an ElevenLabs section in **Builder Settings**. Enter
the password-masked key, test Voices read access, and explicitly **Save API Key to
Project**. Typing or testing only changes the current session's credential. A
saved credential restores on project load; a different project starts with its
own saved key. **Remove Saved API Key** clears the saved and current key after a
successful save. Failed saves restore the previous saved credential.

In **Reference Builder**, each primary character has **Use ElevenLabs voice**.
Enable it, **Refresh Voices**, choose a voice, and play its existing preview where
available. **Load More Voices** fetches another page for larger accounts. Save
Reference Builder to keep the assignment. Disable the checkbox to retain a voice
ID without using it. Props and extra image references have no voice controls.

Assignments follow the existing subject IDs and `subject_scene_map`. This step
does not generate dialogue, insert audio, or design new voices. Those actions
will use these assignments in the next steps. Existing MiniMax native voice
settings remain separate.

## Architecture and API/MCP

- `builder/elevenlabs.py`: stdlib HTTP client for account voices, bounded response
  size and request timeout, credential validation, minimal voice metadata and
  redacted provider errors. No SDK or package installation.
- `web/music_video_builder/elevenlabs.mjs`: settings and picker UI. Account lists,
  pagination and preview URLs remain transient. Async results check the project,
  key, mode and dialog lifecycle before updating the UI.
- The shared session key is `elevenlabs_api_key_project`. Subject configuration is
  `flux_reference_builder.subjects[].elevenlabs_voice` with `enabled`, `voice_id`
  and `name`; the legacy primary `subject` mirrors the first subject.
- Builder POST `/vrgdg/music_builder/elevenlabs_voices` accepts a transient
  `api_key`, `video_type: "speaking"`, and optional `next_page_token`.
- Agent GET `/projects/{pid}/elevenlabs` returns configured status and revision.
  PUT accepts `{ "api_key": "..." }`; an empty string clears it. Supports If-Match.
- Agent GET `/projects/{pid}/elevenlabs/voices` uses the saved key; optional query
  `next_page_token` fetches another page. POST `/projects/{pid}/elevenlabs/test`
  tests Voices read access without speech generation.
- Existing PUT `/projects/{pid}/references/subjects/{rid}` accepts
  `elevenlabs_voice: { "enabled": true, "voice_id": "...", "name": "..." }` in
  Speaking mode. Omitted assignments are preserved. Supports If-Match.
- Generated `agent_api/endpoints.json` exposes the routes to MCP automatically.
  Restart the MCP process to reload the updated endpoint catalog.

Keys are stored in project files like the existing LLM runner credentials.
Existing project export credential handling and export warnings apply. Do not
commit a real key or use it in tests. Public HTTPS sample URLs come from the
[ElevenLabs voice-list API](https://elevenlabs.io/docs/api-reference/voices/search);
preview playback does not invoke speech generation.

The connection test requires **Voices → Read** (`voices_read`) access. A key that
works for text to speech can still lack permission to list voices. For restricted
keys, use ElevenLabs → Developers → API Keys → More Actions → Edit to enable voice
read access. The service reads only a bounded error body and translates known
`missing_permissions` and `invalid_api_key` statuses into fixed messages; it never
echoes provider messages or keys. An unknown HTTP 401 is reported as ambiguous
authentication/access failure rather than proof that the key is invalid. Test
Connection reads the current input value, including browser autofill changes.

## Local test checklist

1. Restart ComfyUI and reload the Builder browser page.
2. In Singing and No lip sync modes, confirm no ElevenLabs settings/picker appears.
3. Switch to Speaking. Settings → ElevenLabs: enter a key with Voices read
   permission. Test Connection. Confirm that testing does not save the key.
4. Save API Key to Project. Reload the project and verify it restores.
5. Open Reference Builder, enable a character's voice, refresh, choose and preview.
   Save and reopen; verify both voice ID and name survive. Check another character
   can use a different voice. Existing scene mappings should be unchanged.
6. Disable a voice, save and reopen; the selection survives while disabled.
7. For a larger account, load another page and confirm the selection stays intact.
8. Try an invalid/restricted key. Confirm a useful error and no key in messages.
9. Switch projects, including one without a saved key. Verify keys do not carry over.
10. Remove the saved key, reload and confirm it stays cleared while assignments
    remain. Confirm the existing scene Audio Settings dialog still works.
11. Through API/MCP, test saved-key status, discovery and subject assignment. Try
    a stale If-Match revision and a non-Speaking project; both must reject changes.

## Suggested PR text

**Title:** Add Speaking-only ElevenLabs project credentials and character voices

Speaking projects can save an ElevenLabs API key using the existing LLM runner
project-key convention, test account access, and choose/preview a voice on each
primary Reference Builder character. Voice assignments persist with existing
subject IDs and scene mappings. Matching revision-checked API routes are exposed
to MCP through the generated endpoint catalog. Voice design and speech generation
are deferred to the next steps.

Validation: `test_elevenlabs.py` covers provider errors, safe pagination, mode
guards, revisions, actual project persistence and frontend save/picker behavior.
Run the API contract, endpoint-doc, MCP and session-parity tests as well.

## Verification in this workspace

- 36 focused Python tests passed: ElevenLabs, Speaking audio settings, session-key
  parity, reference saving and atomic writes.
- API contract (7), endpoint documentation (1) and MCP endpoint coverage (13) passed.
- Settings layout harness (5 JavaScript checks) and ElevenLabs UI harness passed;
  changed frontend modules passed `node --check`; `git diff --check` passed.
- Full discovery ran 1,086 tests. Five failures were the existing environment
  issues: missing `krea2_turbo_fp8_scaled.safetensors`, two missing
  `minimax_h3_ref2va_pruned_int8_convrot.safetensors` cases, and two RefMod tests
  requiring `node:module.register` unavailable in Node 18.15.0. One new Settings
  harness failure was corrected by loading the imported ElevenLabs helper; the
  affected harness then passed all five checks.
- A real ElevenLabs credential was not supplied. Account connection, actual voice
  samples and project reload remain on the manual local checklist above.
