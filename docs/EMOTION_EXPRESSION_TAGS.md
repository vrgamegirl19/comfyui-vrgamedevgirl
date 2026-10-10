# Emotion/Expression Tags

Each scene has an optional **Emotion/Expression Tags** text box in the scene editor and Lyric Review. Examples: `Angry`, `Sad`, or `Start happy, then end sad`.

In the Video Builder's **Speaking** video mode, click **+ Emotion Tag** beside **+ Line Note** to show an emotion text box beneath each scene's Line Notes. The boxes use the same saved scene field, support undo, and remain saved when the lane is hidden or the video mode changes. The lane's visibility is saved as `show_timeline_emotion_tags`.

For H3 **built-in audio** speaking scenes, the LLM uses the scene beat, storyboard details, dialogue and emotion direction to choose both dialogue-header descriptors such as `[English, curious]` and inline delivery cues such as `<breath>`, `<pause>` and `<i>word</i>`. Spoken words remain unchanged. New prompt generation requires at least one inline delivery cue in each spoken shot and retries up to three attempts when the LLM returns only an emotion header. These cues follow the community examples supplied during development; tests verify prompt handling, not the renderer's interpretation of each tag. Supplied audio keeps its existing delivery unchanged.

The LLM receives the scene's lyrics, facial performance preset/custom direction, and this input. Explicit scene emotion input overrides the facial preset. Custom facial performance text such as `Angry` also supplies emotion direction. The LLM chooses fitting tags and visible acting for that lyric; the application does not expand these inputs into canned expressions.

For H3 singing with supplied audio:

- With **Do not add lyric lines to prompt** checked, lyrics remain available to the LLM for interpretation but are excluded from the generated video prompt. The LLM can write `[angry, singing]` alongside visible expressions synchronized with `<Audio 1>`.
- With lyrics included, a single shot may use `<d>[English, angry, singing] exact lyric words.</d>`. The assembly preserves these descriptors and avoids duplicating the lyric. The actual language must match the lyric.
- Instrumental and visual-only shots have no singing cues. Multi-shot lyric-free assembly removes mouth, lip, and jaw references. A single continuous shot allows articulation only during audible vocals.
- Emotion transitions are staged in time, rather than flattening `Start happy, then end sad` into one simultaneous state. Supplied audio remains unchanged.

Existing prompts are unchanged until regenerated. Tags are prompting guidance; their visual effect depends on the generation model.

The scene key is `emotion_expression_tags` (maximum 1200 characters). REST and MCP clients can set or clear it with the existing scene-update operation, `PATCH /vrgdg/api/v1/projects/{pid}/scenes/{sid}`. It is saved on the timeline segment and Storyboard card and synchronized between both editors.

Implementation: `web/music_video_builder/emotion_expression.mjs` resolves browser input; its Python twin and LLM instruction live in `llm/prompts/emotion_expression.py`. Prompt assembly preserves LLM acting while enforcing audio and instrumental boundaries. There is no separate tag preset engine or MCP-only state.
