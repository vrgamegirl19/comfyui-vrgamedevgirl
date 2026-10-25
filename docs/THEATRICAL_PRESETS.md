# Theatrical presets and prompt options

Scene Defaults offers three independently selectable **Theatrical** presets:

- **Auto camera flow**: 20 setups covering low wide-angle performance masters, face-forward close-ups, Dutch tilts, whip-pans, backward/side tracking, circling, overheads, reflection portraits and tender push-ins.
- **Performance style**: cheeky theatrical presence, comic timing, purposeful struts and gestures, musical reactions and tender quieter passages.
- **Facial performance**: animated brows, sly glances, grins, winks, mock pouts, evolving supporting reactions and warm intimate expressions.

The camera selection retains the saved key `pretty_little_malfunction`; its displayed name is **Theatrical**. Performance and facial selections both use `theatrical`. These directions adapt to the current project's cast, props, locations and cue timing. They do not import the source song's characters or story.

## Agent API and MCP

Use the existing revision-checked `PATCH /projects/{pid}/scenes/{sid}` endpoint (MCP `scene_update` or its generated API tool), with the ordinary scene-default keys:

```json
{
  "scene_defaults": {
    "camera_flow": "pretty_little_malfunction",
    "performance_style": "theatrical",
    "facial_performance": "theatrical"
  }
}
```

The existing `POST /projects/{pid}/minimax-prompts` job expands these selections into the full camera, performance and facial directions for the LLM. Scene overrides take precedence over project defaults. Authored motion summaries and custom facial direction keep their existing priority. Lyric-free prompt assembly retains Theatrical facial emotion through the same shared data. Existing routes, saved keys, schemas and MCP tool contracts are unchanged.

The browser definitions are authoritative. `node scripts/export_theatrical_presets.mjs` regenerates `storyboard/theatrical_presets.json` for Python; `--check` verifies parity. No model is loaded or switched by these helpers.

## Prompt Options

The dialog shows tools and Find/Replace targets appropriate to the selected engine and mode. MiniMax reference-to-video and RefMod show a full-width video section, with image generation and LTX tools omitted. Image controls remain available for modes with generated start frames. MiniMax-specific options and validation appear only for MiniMax. LTX prompt action labels follow its current mode. This is a UI change and requires no API or MCP route.

## Checks

Run the focused JavaScript suites `prompt_options_visibility.cjs`, `camera_flow.test.mjs`, `minimax_multishot_camera.cjs` and `scene_defaults_presets.cjs`. Python coverage lives in `test_theatrical_presets.py`, `test_agent_api_minimax_prompts.py` and `test_lyric_free_performance.py`. API contract and endpoint-document checks should remain clean.
