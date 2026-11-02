# API video models in Video Builder

Choose **API Nodes** in **Builder Settings → Project Video Engine**, or cycle the engine badge to API Nodes. Open the Video panel and select **Seedance 2.5 → Reference to Video**.

Sign into ComfyUI with an account that has credits, or configure a ComfyUI API key in ComfyUI Settings. The Builder uses the frontend's existing authentication; provider keys are not stored in the project.

For each scene:

1. Assign character and location images using Reference Builder, or enter additional local image paths in the API Nodes panel. Paths refer to files on the ComfyUI computer. Optional video and audio references have separate path lists.
2. Write a Seedance prompt. Images are numbered in the displayed order, followed by any additional images. Video and audio reference numbers follow their respective path-list orders. Refer to them as `Image 1`, `Video 1`, or `Audio 1`.
3. Choose resolution, aspect ratio, and whether Seedance generates audio. These settings and the prompt are saved per scene.
4. Click **Create Seedance Scene Video**, or use **Render All** for multiple scenes and final stitching. Build Full Video uses this same render stage for API projects; enter prompts and references first. Keep the Builder tab open.

Video generation runs remotely through ComfyUI's official `ByteDance2ReferenceNodeV2` node. No local video-model weights or video-model VRAM are required. Loading references and saving, trimming, and assembling videos still run locally.

Scenes can be up to 30 seconds. The API generates whole seconds with a four-second minimum; the Builder rounds up and trims the result to the scene's timeline duration. The installed node currently supports up to 30 image, 10 video, and 10 audio references; its media validation and account restrictions also apply.

When project audio is loaded, final stitching uses that audio, with custom scene-audio overrides where supplied. With no project track, stitching uses the clips' embedded audio, including any restored custom audio. Audio references guide generation and do not guarantee lyric synchronization.

## Project/custom audio and lip-sync guidance

Load the song using the existing project Audio controls, or load a scene-specific clip in **Scene Audio Options**. In the scene's **Video → API Nodes** panel, choose **Audio input → Project / custom scene audio (lip-sync guidance)**. Custom scene audio takes priority over the project track. The separate audio-reference file list must be empty in this mode.

Choose the singer's reference image under **Lip-sync performer**, or select **Follow singer/speaker assignments in the prompt** for prompts that already identify the vocal performers. Instrumental/no-lip-sync scenes preserve their lips-closed instructions. **Use this audio input for all scenes** copies only the input mode; performer assignments remain per scene.

The Builder automatically cuts the scene's project-track excerpt or uses the custom clip's source offset, selected duration, and placement on the timeline. Silence is added before/after the custom clip where required and after the scene boundary to match Seedance's rounded-up generation duration. It never takes the next scene's lyrics to fill that padding. The excerpt is attached as `@Audio 1`, and the runtime adds timing and performer instructions. Your supplied excerpt replaces the generated audio on the returned scene video.

This is audio-reference-driven generation, not a deterministic lip-sync correction pass. Review the actual mouth motion before rendering the entire project; restoring the soundtrack does not itself fix mouth timing. The integration has automated timing/attachment tests, but those tests do not measure Seedance's generated lip sync.

This initial integration supports Reference to Video only. Automatic Seedance prompt generation, background server pipelines, draft/final passes, editing, and extension modes are not included. Failed or ambiguous paid submissions are not automatically retried. The submitted ComfyUI prompt ID is saved on the scene as `api_video_prompt_id` for checking ComfyUI history.

## Convert existing prompts

Open **Prompt Options → Convert All to Seedance Prompts**. The converter uses the selected text LLM Runner and the current engine's saved video prompts, including insert scenes. Scenes without a video prompt are skipped. The action does not switch engines or generate videos.

To convert one scene, select it on the timeline and click **Convert This Scene to Seedance Prompt** immediately above the prompt box in the right-side **Video → API Nodes** panel. This uses the same backup and preservation checks, but backs up and converts only the selected scene. The converted text appears in that scene's prompt box; other scenes stay unchanged.

Before any LLM request, it saves a separate JSON backup in `project_context/prompt_backups/`, including the original prompts, previous Seedance prompts, and reference assignments. The resulting backup path is shown in the progress window. Backups have unique filenames and a creation timestamp; another conversion never replaces an earlier backup.

Conversion preserves the source text in immutable blocks. Code changes reference labels to the actual Seedance upload order and translates MiniMax formatting markers. The LLM supplies a section-layout plan rather than rewritten prose. The converter checks that every block appears exactly once and in order before applying the result. All scene descriptions, shot instructions, lyrics, dialogue, cue times, and soundtrack requirements remain in the converted prompt. A missing reference, invalid layout, cancellation, or concurrent source edit stops the batch without applying partial results. Nothing is truncated to fit the character limit.

MiniMax `<d>` delivery tags become plain quoted lyrics/dialogue with their delivery notes, and `<i>` lyric emphasis becomes an explicit word-emphasis note. The stock instruction to preserve everyone's exact posture across cuts is changed to continuous movement without resets or frozen action. In scene-audio mode, performance timing refers directly to `@Audio 1`; whole-song preservation remains a final-assembly requirement. Existing converted prompts can be cleaned by running **Convert This Scene** again.

The converter also changes the stock “Hard cut, the same moment” instruction into immediate continuation. In a vocal close-up, “spinning once/a full turn” becomes a face-visible partial turn. These narrow staging repairs leave quoted lyrics/dialogue and all cue times untouched. Vocal-role guidance keeps instrumental-only performers silent and does not infer backing-vocal ownership from parentheses; it follows existing assignments and clearly distinguishable voices in the attached recording.

The original MiniMax/LTX prompt fields remain intact. Converted text is saved separately as each scene's `api_video_prompt`, with frozen reference-image order and a reference binding. Changing those references later requires reconversion or a manual prompt update. Review the converted text in the **API Nodes** panel before rendering. Soundtrack preservation remains a final-assembly requirement; conversion does not attach the entire song to Seedance or guarantee generated lip sync.

Use **Prompt Options → Edit Seedance Conversion Instructions** to edit the `seedance25_prompt_conversion` instructions through the existing project/scene instruction editor. Custom instructions cannot bypass the content-preservation checks.

After installing this change, restart ComfyUI and refresh the browser to load the new backend route and panel.
