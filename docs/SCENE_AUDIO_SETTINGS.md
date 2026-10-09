# Speaking scene audio settings — step 1

Speaking mode now has a dedicated audio window. Right-click a base scene and choose **Audio Settings…**. The existing **Open Scene Audio Options** inspector button opens the same window in Speaking mode. These features are hidden in Singing mode and do not target overlay scenes.

The window loads/replaces scene dialogue, previews it with silence, removes scene-owned dialogue, and sets silence before/after plus duration fitting. **Use project audio defaults** is enabled initially. **Speaking Audio Defaults…**, beside the scene audio button in the Audio inspector, changes the inherited values for the project. Silence starts at 0 seconds and fitting starts enabled. Defaults are saved with the project; custom overrides belong only to their scene.

With fitting enabled, 6 seconds of dialogue plus 0.5 seconds before and 1 second after becomes a 7.5-second scene. Later base scenes and their attached audio move by the change in duration. Their settings and media do not change. Existing gaps between scenes remain. Independent music/effect clips and overlays stay in place. Multiple dialogue pieces retain their internal spacing. Added silence is represented by timeline gaps and the existing audio mixer; source files are preserved. Fitting disabled keeps scene boundaries and reports audio overflow in the dialog.

An audio edit that affects an existing rendered scene shows **Audio changed** on its scene card. Render that scene again to update its video; successfully rendering clears the marker. Rendering in progress and frozen timing block edits. Global project audio is independent and is not automatically assigned to a scene.

ElevenLabs, voice design, automatic long-dialogue splitting and continuation controls are future steps.

## Local test checklist

1. Restart ComfyUI to load the new routes and refresh the browser.
2. Open a disposable Speaking project with three 4-second scenes. Open scene 1 by right-click and then by the Audio inspector button. Confirm only audio controls appear.
3. Load a known audio file, preview it, then save. Confirm the scene fits the audio and later scenes and their attached dialogue move together.
4. Uncheck **Use project audio defaults**; set 0.5 seconds before and 1 second after. Preview both silences, save, and confirm the scene duration includes them. Reopen and save again: timing must not change again.
5. Load a shorter replacement and confirm later scenes move left. Test undo/redo, then save and reopen the project to verify settings and timing persist.
6. Set project defaults and confirm inherited scenes update while custom scenes keep their values. Check that independent background music stays in place.
7. Disable fitting and confirm the scene keeps its duration and reports overflow when applicable. Test frozen timing. Cancel an edit and confirm scene state stays unchanged.
8. Switch to Singing. Confirm the defaults button and right-click audio-settings item are absent. Ctrl+A and input editing retain their normal behavior.

## Agent API and MCP

GET/PATCH `/projects/{pid}/speaking-audio/defaults` reads/updates project defaults. PATCH body:

```json
{"settings": {"silence_before": 0, "silence_after": 0.5, "fit_duration": true}}
```

GET/PATCH `/projects/{pid}/scenes/{sid}/audio-settings` reads/updates one scene. PATCH body:

```json
{"settings": {"use_project_defaults": false, "silence_before": 0.5, "silence_after": 1, "fit_duration": true}}
```

The scene PATCH also accepts `audio_data` (base64 or a data URL) with `audio_name`, or `clear_audio: true`. Silence values must be finite numbers from 0 to 60 seconds; flags must be booleans. Send `If-Match` with the saved revision to avoid overwriting concurrent changes. Operations reject non-Speaking projects. MCP automatically exposes these routes from the committed endpoint catalog; restart the MCP connection to refresh its tools.

## PR preparation

Suggested title: **Add Speaking-mode scene audio settings with silence and ripple timing**

Suggested description:

> Speaking scenes now have an audio-only dialog through right-click Audio Settings or the Audio inspector button. Project defaults and scene overrides control opening/ending silence and fitting scene duration to dialogue, shifting later scenes and their attached audio while preserving original audio files.
>
> The browser and revision-checked Agent API share a timing service. Updated endpoint/OpenAPI exports expose the same operations through generated MCP tools. Includes timing, source-preservation, mode-gating, keyboard, persistence-path and revision-conflict tests.

Keep the existing unrelated MiniMax settings, continuation patches/tests, and `Workflows/H3MotionAdapter/` changes out of this PR. `mcp_server/` is local-only; the tracked router, endpoint descriptions and generated catalogs carry MCP support for other installations.

Validation: focused scene-audio tests, the existing audio-clip tests, OpenAPI contract checks, endpoint documentation checks, session-key parity, MCP endpoint coverage, atomic-write tests, JavaScript syntax checks and `git diff --check` pass. The scene-audio tests include a real mixer check for opening/ending silence, original WAV preservation, on-disk project persistence, dialog mode gating and keyboard behavior. Browser visual QA remains on the local checklist above.

The full suite ran 1,073 tests and reported five failures outside this change: `test_agent_api_image_jobs` could not find its Krea2 model, two `test_agent_api_latents` cases could not find their MiniMax model, and the RefMod card/storyboard JavaScript tests require `node:module.register`, which is unavailable in the installed Node 18.15.0. Those files were not changed by this feature.
