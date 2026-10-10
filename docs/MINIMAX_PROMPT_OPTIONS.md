# MiniMax prompt options and reference scenes

**Prompt Options → Use Structured outputs** defaults to off. Reference-to-video prompts use a compact `detailed_description` with shot prose and location picture references. Enable the option to include subject definitions, summary, retention analysis and audio sections. It is saved as `use_structured_outputs` and supports undo. Regenerate existing prompts to apply a changed option.

Reference-to-video uses character pictures for identity and environment pictures for the setting. The scene beat, storyboard directions and camera settings determine the new shot's staging, framing and movement. Changing from either image mode to reference-to-video clears legacy start-frame requirements. Reference mode also ignores stale flags that would prepend the scene image or copy its composition.

Shot instructions establish props and their physical placement before referring to them, attach actions to the correct actor, and describe the camera path in chronological order. Each character is introduced by label in each shot; subsequent unambiguous actions use natural pronouns. Multiple-character scenes repeat labels when needed to identify a new actor or speaker. These instructions guide the LLM; final prose still warrants review.

**Do not add lyric lines to prompt** applies to singing with supplied audio. Built-in audio and speaking retain their spoken or sung words. See [lyric-free prompts](LYRIC_FREE_VIDEO_PROMPTS.md) and [emotion and speech tags](EMOTION_EXPRESSION_TAGS.md).
