import { makeButton, makeCheckbox, makeField, makeSelect, makeSettingsSection, toast } from "./controls.mjs";

export function seedanceSettings(segment) {
  return { resolution: "720p", ratio: "16:9", generate_audio: true, use_mapped_references: true,
    audio_mode: "reference_files", lip_sync_image: 0, ...segment?.api_video_settings };
}

export function seedanceSceneAudioInput(segment, projectAudioPath, editedAudio = null) {
  const sceneStart = Number(segment.start || 0);
  const duration = Number(segment.end) - sceneStart;
  if (editedAudio?.audio_path) return { source_path: editedAudio.audio_path, start: sceneStart, duration, source_duration: duration, leading_silence: 0 };
  if (!segment.custom_audio_path) return { source_path: projectAudioPath, start: sceneStart, duration, source_duration: duration, leading_silence: 0 };
  const clipStart = Number.isFinite(Number(segment.custom_audio_timeline_start)) ? Number(segment.custom_audio_timeline_start) : sceneStart;
  const elapsed = Math.max(0, sceneStart - clipStart);
  const leading = Math.max(0, clipStart - sceneStart);
  const clipDuration = Number(segment.custom_audio_duration) > 0 ? Number(segment.custom_audio_duration) : duration;
  const sourceDuration = Math.min(duration - leading, clipDuration - elapsed);
  if (sourceDuration <= 0) throw new Error("The custom audio clip does not overlap this scene. Move it into the scene before using Seedance lip-sync guidance.");
  return { source_path: segment.custom_audio_path, start: Math.max(0, Number(segment.custom_audio_source_start) || 0) + elapsed,
    duration, source_duration: sourceDuration, leading_silence: leading };
}

export function seedanceStitchAudioItems(segments, projectAudioPath) {
  return segments.map(segment => ({
    path: segment.custom_audio_path ? segment.api_video_audio_reference_path || segment.custom_audio_path : projectAudioPath,
    start: segment.custom_audio_path ? segment.api_video_audio_reference_path ? 0 : Math.max(0, Number(segment.custom_audio_source_start) || 0) : Number(segment.start || 0),
    duration: Number(segment.end) - Number(segment.start || 0),
  }));
}

export function apiVideoReferences(segment, mappedImages = []) {
  const settings = seedanceSettings(segment);
  const images = [...(settings.use_mapped_references ? mappedImages : segment?.api_video_image_references || []),
    ...(segment?.api_video_image_paths || []).map(path => ({ path }))];
  const seen = new Set();
  return {
    reference_images: images.filter(item => {
      const key = String(item?.path || item?.data || "").trim();
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    }),
    reference_videos: segment?.api_video_video_paths || [],
    reference_audios: segment?.api_video_audio_paths || [],
  };
}

export function apiVideoReferenceSignature(references) {
  return JSON.stringify([references.reference_images.map(item => String(item.path || item.data || "").trim()),
    references.reference_videos.map(item => typeof item === "string" ? item.trim() : item.path),
    references.reference_audios.map(item => typeof item === "string" ? item.trim() : item.path)]);
}

export function validateApiVideoScene(segment, references) {
  const errors = [];
  const settings = seedanceSettings(segment);
  const prompt = String(segment?.api_video_prompt || "").trim();
  const binding = segment?.api_video_prompt_reference_binding;
  if (binding?.prompt === prompt && binding.signature !== apiVideoReferenceSignature(references)) errors.push("Seedance references changed after conversion. Convert the prompt again with the current references or update its reference labels.");
  if (!prompt) errors.push("Seedance Reference to Video prompt is missing.");
  if (prompt.length > 15000) errors.push("Seedance prompt exceeds 15,000 characters.");
  const duration = Number(segment?.end) - Number(segment?.start);
  if (!Number.isFinite(duration) || duration <= 0 || duration > 30) errors.push("Seedance needs a scene longer than zero and at most 30 seconds. Split longer scenes first.");
  if (!references.reference_images.length && !references.reference_videos.length && !references.reference_audios.length && settings.audio_mode !== "scene_audio") errors.push("Seedance needs at least one image, video, or audio reference.");
  if (settings.audio_mode === "scene_audio" && references.reference_audios.length) errors.push("Scene-audio mode uses your project/custom audio as Audio 1. Clear the audio-reference file list or choose Reference files only.");
  if (settings.audio_mode === "scene_audio" && Number(settings.lip_sync_image) > references.reference_images.length) errors.push("The selected lip-sync performer image is missing. Choose a currently attached image.");
  for (const [key, limit] of [["reference_images", 30], ["reference_videos", 10], ["reference_audios", 10]]) {
    if (references[key].length > limit) errors.push(`Seedance accepts at most ${limit} ${key.replace("reference_", "")} references.`);
  }
  return errors;
}

export function createApiVideoPanel({ activeSegment, state, getMappedImages, autoSave, onRender, onConvertScene, allEditableSegments = () => [] }) {
  const model = makeSelect([{ value: "seedance_2_5", label: "Seedance 2.5" }], "seedance_2_5");
  const mode = makeSelect([{ value: "reference_to_video", label: "Reference to Video" }], "reference_to_video");
  const resolution = makeSelect(["480p", "720p", "1080p"].map(value => ({ value, label: value })), "720p");
  const ratio = makeSelect(["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"].map(value => ({ value, label: value })), "16:9");
  const audio = makeCheckbox("Generate audio", true);
  const audioMode = makeSelect([
    { value: "reference_files", label: "Reference files only" },
    { value: "scene_audio", label: "Project / custom scene audio (lip-sync guidance)" },
  ], "reference_files");
  const lipSyncImage = makeSelect([{ value: "0", label: "Follow singer/speaker assignments in the prompt" }], "0");
  const lipSyncField = makeField("Lip-sync performer", lipSyncImage);
  const applyAudioToAll = makeButton("Use this audio input for all scenes");
  const audioNote = document.createElement("div");
  audioNote.textContent = "Project/custom mode uses the existing Scene Audio controls: custom scene audio takes priority, otherwise the project track is cut at this scene's timeline position. The excerpt becomes Audio 1. Select the singer's image or follow the prompt's assignments. Your supplied audio is restored to the finished clip. Seedance mouth synchronization still needs review.";
  audioNote.style.cssText = "font-size:12px;color:#a1a1aa;line-height:1.4;";
  const mapped = makeCheckbox("Use this scene's mapped Reference Builder images", true);
  const textarea = (placeholder) => {
    const input = document.createElement("textarea");
    input.placeholder = placeholder;
    input.style.cssText = "width:100%;min-height:80px;box-sizing:border-box;background:#18181b;color:#fafafa;border:1px solid #3f3f46;border-radius:6px;padding:8px;";
    return input;
  };
  const prompt = textarea('Describe the scene. Refer to images as Image 1, Image 2, videos as Video 1, and audio as Audio 1.');
  prompt.maxLength = 15000;
  const images = textarea("One local image path per line (on the ComfyUI computer)");
  const videos = textarea("One local video path per line (optional)");
  const audios = textarea("One local audio path per line (optional)");
  const referenceList = document.createElement("div");
  referenceList.style.cssText = "font-size:12px;white-space:pre-wrap;color:#a1a1aa;";
  const note = document.createElement("div");
  note.textContent = "Uses your ComfyUI login and credits. Video generation runs remotely. Settings belong to this scene. Duration follows the timeline (up to 30 seconds); clips shorter than 4 seconds are generated at 4 seconds and trimmed. Reference audio guides generation; it does not guarantee lyric sync. Render All uses each scene's saved settings.";
  note.style.cssText = "font-size:12px;color:#a1a1aa;line-height:1.4;";
  const renderButton = makeButton("Create Seedance Scene Video", "primary");
  const convertButton = makeButton("Convert This Scene to Seedance Prompt");
  convertButton.title = "Back up and convert only the selected scene's current video prompt. Original prompts remain saved.";
  const panel = makeSettingsSection("API Nodes", [
    makeField("Video model", model), makeField("Mode", mode), note,
    makeField("Resolution", resolution), makeField("Aspect ratio", ratio), audio.wrapper,
    makeField("Audio input", audioMode), audioNote, lipSyncField, applyAudioToAll,
    mapped.wrapper, referenceList, makeField("Additional image references", images),
    makeField("Video references", videos), makeField("Audio references", audios),
    convertButton, makeField("Seedance prompt", prompt), renderButton,
  ], true);
  panel.style.display = "none";
  let panelSegment = null;
  let rendering = false;
  let converting = false;
  const paths = input => input.value.split(/\r?\n/).map(value => value.trim().replace(/^"|"$/g, "")).filter(Boolean);
  function save() {
    if (!panelSegment) return;
    panelSegment.api_video_prompt = prompt.value;
    panelSegment.api_video_settings = { resolution: resolution.value, ratio: ratio.value, generate_audio: audio.input.checked, use_mapped_references: mapped.input.checked,
      audio_mode: audioMode.value, lip_sync_image: Number(lipSyncImage.value || 0) };
    panelSegment.api_video_image_paths = paths(images);
    panelSegment.api_video_video_paths = paths(videos);
    panelSegment.api_video_audio_paths = paths(audios);
  }
  function sync() {
    const visible = state.projectVideoEngine === "api_nodes";
    panel.style.display = visible ? "block" : "none";
    if (!visible) return;
    const segment = activeSegment();
    panelSegment = segment;
    const settings = seedanceSettings(segment);
    resolution.value = settings.resolution;
    ratio.value = settings.ratio;
    audio.input.checked = settings.generate_audio;
    audioMode.value = settings.audio_mode;
    mapped.input.checked = settings.use_mapped_references;
    prompt.value = segment?.api_video_prompt || "";
    images.value = (segment?.api_video_image_paths || []).join("\n");
    videos.value = (segment?.api_video_video_paths || []).join("\n");
    audios.value = (segment?.api_video_audio_paths || []).join("\n");
    const refs = apiVideoReferences(segment, getMappedImages(segment));
    lipSyncImage.textContent = "";
    for (const [index, label] of [[0, "Follow singer/speaker assignments in the prompt"],
      ...refs.reference_images.map((item, index) => [index + 1, `Image ${index + 1}: ${item.label || item.name || item.path || "Performer"}`])]) {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = label;
      lipSyncImage.append(option);
    }
    lipSyncImage.value = String(settings.lip_sync_image || 0);
    lipSyncField.style.display = settings.audio_mode === "scene_audio" ? "" : "none";
    referenceList.textContent = refs.reference_images.map((item, index) => `Image ${index + 1}: ${item.label || item.name || item.path || "Reference Builder image"}`).join("\n") || "No image references selected.";
    for (const control of [resolution, ratio, audio.input, audioMode, lipSyncImage, applyAudioToAll, mapped.input, prompt, images, videos, audios]) control.disabled = !segment;
    renderButton.disabled = !segment || rendering || converting;
    convertButton.disabled = !segment || rendering || converting;
  }
  for (const control of [resolution, ratio, audio.input, audioMode, lipSyncImage, mapped.input, prompt, images, videos, audios]) {
    control.addEventListener("input", save);
    control.addEventListener("change", () => { save(); sync(); void autoSave("Seedance scene settings"); });
  }
  applyAudioToAll.onclick = async () => {
    save();
    for (const scene of allEditableSegments()) scene.api_video_settings = { ...seedanceSettings(scene), audio_mode: audioMode.value };
    await autoSave("Seedance audio input applied to all scenes");
    toast("Audio input applied to all scenes. Each scene keeps its own audio source and performer assignment.");
  };
  renderButton.onclick = async () => {
    if (rendering || converting) return;
    rendering = true;
    save();
    renderButton.disabled = true;
    convertButton.disabled = true;
    try { await onRender(); }
    catch (error) { toast(String(error?.message || error), true); }
    finally { rendering = false; sync(); }
  };
  convertButton.onclick = async () => {
    if (rendering || converting) return;
    save();
    converting = true;
    convertButton.disabled = true;
    renderButton.disabled = true;
    try { await onConvertScene?.(); }
    catch (error) { toast(String(error?.message || error), true); }
    finally { converting = false; sync(); }
  };
  return { panel, renderButton, convertButton, sync, save, references: segment => apiVideoReferences(segment, getMappedImages(segment)) };
}
