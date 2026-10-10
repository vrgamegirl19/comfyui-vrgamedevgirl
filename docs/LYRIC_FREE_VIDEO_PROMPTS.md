# Music-video prompts without lyric lines

In **Prompt Options → Video**, enable **Do not add lyric lines to prompt** to generate
MiniMax H3 music-video prompts synchronized to supplied audio without quoting the lyrics.
The option defaults to unchecked and is saved with the project, including undo/redo.
Older projects open with the option unchecked.

- Vocal shots describe passionate singing synchronized to `<Audio 1>`.
- Single-shot prompts may describe mouth and jaw synchronization during audible vocals.
- Multi-shot prompts omit explicit mouth, lip and jaw movement wording throughout.
- Instrumental cue ranges, B-roll and scenes without characters receive no singing direction.
- Existing facial presets and custom directions contribute eyes, brows and visible expression.
- Saved lyrics, cue timing and singer assignments remain available to the builder.
- Checking the option affects newly generated prompts. Regenerate existing prompts to apply it.
- Speaking mode and built-in audio keep their existing behavior.

Agents use the same project setting through the existing settings API/MCP tool:

```json
{"project": {"omit_lyrics_from_video_prompts": true}}
```

Send this body to `PATCH /vrgdg/api/v1/projects/{pid}/settings` (with `If-Match` when
using revision checking). The MiniMax prompt job and prompt-context/assembly endpoints
honor the saved setting. No additional MCP tool or parallel project key is needed.

Policy helpers live in `web/music_video_builder/lyric_free_performance.mjs` and
`minimax/lyric_free_performance.py`. Keep their shot rules and facial presets synchronized.
