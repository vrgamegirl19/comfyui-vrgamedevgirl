import { GEMMA_VIDEO_PROMPT_TIMEOUT_MS, postJson } from "./comfy_api.mjs";
import { apiVideoReferenceSignature } from "./api_video.mjs";
import { toast } from "./controls.mjs";

export function seedanceSourcePrompt(segment, engine) {
  const keys = engine === "api_nodes" ? ["api_video_prompt", "minimax_h3_prompt", "i2v_prompt"]
    : engine === "minimax_h3" ? ["minimax_h3_prompt", "i2v_prompt"] : ["i2v_prompt", "minimax_h3_prompt"];
  const key = keys.find(key => String(segment?.[key] || "").trim());
  return key ? { key, prompt: String(segment[key]) } : null;
}

export function seedanceImageAssignments(items) {
  const images = [], aliases = {}, seen = new Map();
  for (const [index, item] of items.entries()) {
    const image = item.image || item;
    const fingerprint = String(image.path || image.data || "").trim();
    if (!fingerprint || fingerprint.startsWith("refmod://")) throw new Error("Seedance conversion needs actual image references. Assign images instead of RefMods first.");
    if (!seen.has(fingerprint)) {
      images.push({ path: image.path || "", data: image.data || "", name: image.name || item.name || "", label: item.name || item.label || "Reference" });
      seen.set(fingerprint, images.length);
    }
    const target = `@Image ${seen.get(fingerprint)}`;
    const labels = [item.picture_label || `<Picture ${index + 1}>`, `<Image ${index + 1}>`, item.label].filter(label => /^<(Subject|Picture|Image) \d+>$/.test(label || ""));
    for (const label of labels) {
      const previous = aliases[label];
      if (!previous) aliases[label] = target;
      else if (!previous.split(" and ").includes(target)) aliases[label] = `${previous} and ${target}`;
    }
  }
  return { images, aliases };
}

export function createSeedanceConversion({
  state, activeSegment, allEditableSegments, activeProjectFolderForSave, updateActiveFromInputs, saveApiPanel,
  getSourceReferences, textGemmaRunnerPayload, getTextModel, createProgressWindow,
  pushHistory, saveSession, syncInspector, render,
}) {
  let running = false;
  async function convertSeedancePrompts(singleScene = null) {
    if (running) return;
    updateActiveFromInputs();
    saveApiPanel();
    const projectFolder = activeProjectFolderForSave();
    if (!projectFolder) { toast("Create or load a project before converting prompts.", true); return; }
    let progress, backupPath = "", applied = false;
    running = true;
    state.batchCancelled = false;
    try {
      const targets = (singleScene ? [singleScene] : allEditableSegments()).map(segment => ({ segment, source: seedanceSourcePrompt(segment, state.projectVideoEngine) })).filter(item => item.source);
      if (!targets.length) throw new Error("No current video prompts were found to convert.");
      const snapshot = JSON.parse(JSON.stringify(targets.map(({ segment, source }) => ({
        id: segment.id, label: segment.label, start: segment.start, end: segment.end,
        source_key: source.key, source_prompt: source.prompt,
        ...Object.fromEntries(["minimax_h3_prompt", "i2v_prompt", "api_video_prompt", "api_video_settings",
          "api_video_image_paths", "api_video_image_references", "api_video_video_paths", "api_video_audio_paths",
          "api_video_prompt_reference_binding"].map(key => [key, segment[key]])),
      }))));
      const sourceReferences = JSON.parse(JSON.stringify(targets.map(({ segment, source }) => getSourceReferences(segment, source))));
      progress = createProgressWindow(singleScene ? "Convert This Scene to Seedance Prompt" : "Convert All to Seedance Prompts");
      progress.set(singleScene ? "Saving this scene's current prompts before conversion..." : "Saving a backup of all current prompts before conversion...", 2);
      // A failed backup stops all work before the first LLM request or prompt write.
      const backup = await postJson("/vrgdg/music_builder/backup_seedance_prompts", {
        project_folder: projectFolder, video_engine: state.projectVideoEngine,
        scenes: snapshot.map((scene, index) => ({ ...scene, ...sourceReferences[index] })),
      });
      backupPath = backup.backup_path;
      if (!backup.backup_id || !backupPath) throw new Error("The prompt backup was not confirmed. Conversion stopped.");
      const results = [];
      for (let index = 0; index < targets.length; index++) {
        if (state.batchCancelled) throw new Error("Stopped by user. No prompts were changed.");
        if (activeProjectFolderForSave() !== projectFolder) throw new Error("The active project changed. No prompts were changed.");
        const { segment } = targets[index];
        progress.set(`Formatting ${segment.label || `Scene ${index + 1}`} (${index + 1}/${targets.length})...\nBackup: ${backupPath}`, 5 + 85 * index / targets.length);
        const data = await postJson("/vrgdg/music_builder/convert_seedance_prompt", {
          ...textGemmaRunnerPayload(), model_file: getTextModel(),
          project_folder: projectFolder, backup_id: backup.backup_id, scene_id: segment.id,
          unload_after: index === targets.length - 1,
        }, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
        if (!String(data.prompt || "").trim()) throw new Error("The LLM returned an empty converted prompt. No prompts were changed.");
        results.push(data.prompt);
      }
      if (state.batchCancelled) throw new Error("Stopped by user. No prompts were changed.");
      if (activeProjectFolderForSave() !== projectFolder) throw new Error("The active project changed. No prompts were changed.");
      const currentScenes = allEditableSegments();
      for (const [index, { segment, source }] of targets.entries()) {
        if (!currentScenes.includes(segment) || segment[source.key] !== source.prompt) throw new Error("A source scene changed during conversion. No prompts were changed. Run conversion again from its latest text.");
        if (segment.start !== snapshot[index].start || segment.end !== snapshot[index].end) throw new Error("Scene timing changed during conversion. No prompts were changed.");
        if (JSON.stringify(segment.api_video_prompt) !== JSON.stringify(snapshot[index].api_video_prompt)
          || JSON.stringify(getSourceReferences(segment, source)) !== JSON.stringify(sourceReferences[index])) throw new Error("A Seedance prompt or its references changed during conversion. No prompts were changed. Run conversion again.");
      }
      pushHistory();
      targets.forEach(({ segment }, index) => {
        const references = sourceReferences[index];
        segment.api_video_prompt = results[index];
        segment.api_video_image_references = references.image_references;
        segment.api_video_image_paths = [];
        segment.api_video_video_paths = references.video_paths;
        segment.api_video_audio_paths = references.audio_paths;
        segment.api_video_settings = { ...segment.api_video_settings, use_mapped_references: false,
          audio_mode: segment.api_video_settings?.audio_mode || (references.original_soundtrack && !references.audio_paths.length ? "scene_audio" : "reference_files") };
        segment.api_video_prompt_reference_binding = { prompt: results[index], signature: apiVideoReferenceSignature({
          reference_images: references.image_references, reference_videos: references.video_paths, reference_audios: references.audio_paths,
        }) };
        segment.api_video_conversion_backup = backupPath;
      });
      applied = true;
      // Refresh first so a later panel save cannot write stale API text over converted results.
      syncInspector();
      render();
      const saved = await saveSession({ quiet: true, throwOnError: true });
      if (!saved || saved.stale) throw new Error("The converted prompts could not be saved to the project.");
      progress.set(`Converted and saved ${results.length} prompts. Original prompts remain saved.\nBackup: ${backupPath}\nSelect API Nodes to review the Seedance prompts.`, 100);
      toast(`Converted ${results.length} prompts to Seedance.\nBackup: ${backupPath}`);
    } catch (error) {
      const message = `${String(error?.message || error)}\n${applied ? "Converted prompts remain in the Builder; save the project again." : "No prompts were changed."}${backupPath ? `\nBackup: ${backupPath}` : ""}`;
      progress?.set(message, 100);
      toast(message, true);
    } finally { running = false; }
  }
  async function convertAllToSeedancePrompts() {
    return convertSeedancePrompts();
  }
  async function convertSceneToSeedancePrompt() {
    const scene = activeSegment?.();
    if (!scene) { toast("Select a scene before converting its prompt.", true); return; }
    return convertSeedancePrompts(scene);
  }
  return { convertAllToSeedancePrompts, convertSceneToSeedancePrompt };
}
