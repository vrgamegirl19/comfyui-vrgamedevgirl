import { storyboardGptPayload } from "../storyboard_builder/gpt_payload.mjs";
import { postJson, refreshEditorThumbnailUrl } from "./comfy_api.mjs";
import { toast } from "./controls.mjs";
import { normalizeBuilderStoryLayer } from "./model_settings.mjs";
import {
  isInstrumentalLyricText,
  segmentUsesNoLipSyncPerformance,
  syncConceptPromptToStoryBeat,
} from "./prompt_text.mjs";
import { selectedSegmentVideoPath } from "./selection_preview.mjs";
import { isBackupSceneVideoPath, mediaPathKey, normalizeSegmentVideoHistory } from "./timeline_state.mjs";

export function t2iMissingReason(segment) {
  return "";
}

export function sceneConceptPromptText(segment) {
  return String(segment?.t2i_prompt || segment?.flux_prompt || segment?.notes || segment?.flux_notes || "").trim();
}

export function sceneLyricTextForPromptValidation(segment) {
  return String(segment?.lyric_text || segment?.lyric_note || segment?.lyrics || "").trim();
}

export function builderImageInstructionKey(imageMode) {
  if (imageMode === "zimage") return "zimage_t2i";
  if (imageMode === "ernie_image") return "ernie_t2i";
  if (imageMode === "krea2_2pass") return "krea2_t2i";
  if (imageMode === "flux_klein") return "flux_klein_t2i";
  if (imageMode === "nano_banana") return "nano_b_t2i";
  if (imageMode === "flow_gpt") return "flow_gpt_t2i";
  return "";
}

export function createImagePrompts({
  activeProjectFolderForSave, activeSegment, allEditableSegments, applyImageContinuityToPromptSettings,
  applyImageTriggerToPrompt, applyMappedTriggerPhrases, ensureSegmentRuntimeFields, ernieTextGemmaModelSelect,
  flfSameLocationCameraDiversityDirection, flfStillEndpointNotes, fluxReferenceContextForSegment,
  gemmaRunnerLine, i2vTextGemmaModelSelect, nbImageSettingsForSegment, projectInput,
  promoteChainedFLFSceneImageToEndFrame, promptRunnerActionName, pushHistory, render, sceneDisplayName,
  sceneSlotNumber, segmentImageSource, segmentIndexInfo, segmentMappedLocationText, segmentMappedSubjectText,
  setActiveSegment, state, storyboardScenePayload, syncInspector, syncPreview, syncSegmentFlowGptPrompt,
  syncSegmentT2IPrompt, t2iTextGemmaModelSelect, textGemmaRunnerPayload, wizardStoryboardState,
}) {
  async function archiveGeneratedSceneImage(segment, imageInfo) {
    ensureSegmentRuntimeFields(segment);
    if (!segment || !imageInfo?.filename) return null;
    const projectFolder = projectInput.value || state.projectFolder;
    if (!projectFolder) return null;
    try {
      const sceneNumber = sceneSlotNumber(segment);
      const data = await postJson("/vrgdg/music_builder/archive_scene_image", {
        image: imageInfo,
        project_folder: projectFolder,
        scene_number: sceneNumber,
      });
      if (data.saved_path && !segment.image_history.includes(data.saved_path)) {
        segment.image_history.push(data.saved_path);
        segment.image_history_index = segment.image_history.length - 1;
      }
      if (data.saved_path) {
        refreshEditorThumbnailUrl(data.saved_path);
        segment.image_assignment_cleared = false;
        segment.preview_mode = "image";
        promoteChainedFLFSceneImageToEndFrame(segment);
      }
      return data.saved_path || null;
    } catch (error) {
      console.warn("[VRGDG Music Builder] Failed to archive preview image:", error);
      return null;
    }
  }

  function cycleSegmentImageHistory(segment) {
    ensureSegmentRuntimeFields(segment);
    if (!segment?.image_history.length) return;
    pushHistory();
    const nextIndex = (Math.max(-1, Number(segment.image_history_index ?? -1)) + 1) % segment.image_history.length;
    const imagePath = segment.image_history[nextIndex];
    segment.image_history_index = nextIndex;
    segment.custom_image_path = imagePath;
    segment.custom_image_data = "";
    segment.custom_image_name = "";
    segment.image = null;
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    setActiveSegment(segment);
    syncPreview(segment);
    render();
  }

  function addSegmentVideoHistoryPath(segment, videoPath) {
    ensureSegmentRuntimeFields(segment);
    if (!segment || !videoPath || isBackupSceneVideoPath(videoPath)) return;
    segment.video_path = videoPath;
    segment.video_cache_bust = Date.now();
    normalizeSegmentVideoHistory(segment);
    const currentIndex = segment.video_history.findIndex((item) => mediaPathKey(item) === mediaPathKey(videoPath));
    if (currentIndex >= 0) segment.video_history_index = currentIndex;
    segment.video_thumbnail_path = segment.video_thumbnail_history[segment.video_history_index] || segment.video_thumbnail_path || "";
  }

  function cycleSegmentVideoHistory(segment) {
    ensureSegmentRuntimeFields(segment);
    if (!segment?.video_history.length) return;
    pushHistory();
    const nextIndex = (Math.max(-1, Number(segment.video_history_index ?? -1)) + 1) % segment.video_history.length;
    segment.video_history_index = nextIndex;
    segment.video_path = segment.video_history[nextIndex] || segment.video_path || "";
    segment.video_thumbnail_path = segment.video_thumbnail_history[nextIndex] || "";
    segment.preview_mode = "video";
    setActiveSegment(segment);
    syncPreview(segment);
    render();
  }

  function toggleSegmentPreviewMode(segment) {
    ensureSegmentRuntimeFields(segment);
    if (!segment) return;
    const hasImage = Boolean(segmentImageSource(segment));
    const hasVideo = Boolean(selectedSegmentVideoPath(segment));
    if (!hasImage || !hasVideo) return;
    pushHistory();
    segment.preview_mode = segment.preview_mode === "image" ? "video" : "image";
    setActiveSegment(segment);
    syncPreview(segment);
    render();
  }

  function isBlankStarterProject() {
    const segments = allEditableSegments();
    if (state.overlaySegments.length) return false;
    if (segments.length !== 1) return false;
    const segment = segments[0] || {};
    const emptyMedia = !segmentImageSource(segment) && !String(selectedSegmentVideoPath(segment) || "").trim();
    const emptyText = !String(segment.notes || segment.t2i_prompt || segment.flux_prompt || segment.i2v_notes || segment.i2v_prompt || "").trim();
    return emptyMedia && emptyText && /^new scene$/i.test(String(segment.label || "").trim());
  }

  function textOnlyFallbackNotesForSegment(segment, imageMode = state.imageModelMode || "zimage") {
    const parts = [];
    const add = (title, value) => {
      const text = String(value || "").trim();
      if (text) parts.push(`${title}:\n${text}`);
    };
    add("Scene notes", segment?.notes);
    add("Flux/Klein notes", segment?.flux_notes);
    add("NanoBanana notes", segment?.nb_notes);
    add("Mapped subject / character", segmentMappedSubjectText(segment));
    add("Mapped location", segmentMappedLocationText(segment));
    add("Lyric line as still-image mood context", isInstrumentalLyricText(segment?.lyric_text) ? "" : segment?.lyric_text);
    add("Lyric section", segment?.lyric_section);
    add("Scene story beat", segment?.story_beat);
    add("First / Last Frame endpoint guidance", flfStillEndpointNotes(segment, "start"));
    add("Still shot direction", segment?.shot_type);
    const context = imageMode === "nano_banana" ? nbImageSettingsForSegment(segment).reference_context : fluxReferenceContextForSegment(segment);
    add("Reference subject description", context?.subject_description);
    add("Reference location name", context?.location_name);
    add("Reference location description", context?.location_description);
    const storyLayer = normalizeBuilderStoryLayer(state.builderStoryLayer);
    if (storyLayer.enabled !== false) {
      add("User story arc", storyLayer.user_story_arc);
      add("Song story brief", storyLayer.song_story_brief);
      add("Lyric story strength", `${storyLayer.lyric_story_strength}/10`);
    }
    if (!parts.length) {
      parts.push(`Scene:\n${sceneDisplayName(segment, segmentIndexInfo(segment).index)}`);
      parts.push("Direction:\nCreate a cinematic image prompt that fits this scene.");
    }
    if (state.imageContinuityEnabled) {
      const continuity = applyImageContinuityToPromptSettings(segment, { image_ingredients: [] }, "").userNotes;
      if (continuity) parts.push(`Previous-scene continuity:\n${continuity}`);
    }
    parts.push("Image Prep rule:\nCreate a still text-to-image prompt. The final answer must be a visual image prompt, not the lyric line. Use lyrics only for mood, symbolism, emotion, styling, and visual direction. Do not quote or return the lyrics as the prompt. Do not say the subject is singing, lip-syncing, performing vocals, or singing the lyric unless scene notes explicitly request a live singing image.");
    return parts.join("\n\n");
  }

  async function generateStoryboardT2IPromptForSegment(segment, progress = null, percent = 30, label = "Storyboard Gemma T2I", options = {}) {
    if (!segment) throw new Error("Scene is missing.");
    state.activeId = segment.id;
    syncInspector();
    const conceptText = String(segment.notes || segment.flux_notes || segment.nb_notes || "").trim();
    if (!String(segment.story_beat || "").trim() && conceptText) {
      syncConceptPromptToStoryBeat(segment, conceptText);
    }
    const scenes = storyboardScenePayload();
    const scene = scenes.find((item) => item.id === segment.id);
    if (!scene) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: storyboard scene card could not be built.`);
    const imageMode = options.imageMode || state.imageModelMode || "zimage";
    const flfImageTarget = state.videoModelMode === "flf" ? String(options.flfImageTarget || "start").trim().toLowerCase() : "";
    const diversityDirection = flfImageTarget === "start" ? flfSameLocationCameraDiversityDirection(segment, "start") : "";
    const promptScene = diversityDirection ? {
      ...scene,
      notes: [String(scene.notes || "").trim(), diversityDirection].filter(Boolean).join("\n\n"),
      prompt_summary: [String(scene.prompt_summary || "").trim(), diversityDirection].filter(Boolean).join("\n\n"),
    } : scene;
    const storyboardState = wizardStoryboardState(scenes, { promptMode: "image", imageMode });
    const imageStyle = normalizeBuilderStoryLayer(state.builderStoryLayer);
    progress?.set(`${label}: sending storyboard scene card to ${promptRunnerActionName()}...\nConcept prompt is included as the scene story beat.`, percent);
    const data = await postJson("/vrgdg/storyboard/gemma_image_prompt", {
      ...textGemmaRunnerPayload(),
      model_file: t2iTextGemmaModelSelect.value || i2vTextGemmaModelSelect.value || "",
      storyboard_payload: storyboardGptPayload(storyboardState, [promptScene]),
      project_folder: activeProjectFolderForSave(),
      scene_id: segment.id || "",
      builder_instruction_key: builderImageInstructionKey(imageMode),
      flf_image_target: flfImageTarget,
      image_world_style: imageStyle.image_world_style,
      image_custom_style_direction: imageStyle.image_custom_style_direction,
      unload_after: options.unloadAfter !== false,
      seed: options.seed,
      temperature: options.temperature ?? 0.35,
      top_p: options.topP ?? 0.90,
      max_new_tokens: options.maxNewTokens ?? 1200,
    }, 240000);
    pushHistory();
    try {
      syncSegmentT2IPrompt(segment, applyMappedTriggerPhrases(applyImageTriggerToPrompt(data.prompt, segment, imageMode, { validateJunk: true }), segment));
    } catch (error) {
      error.rawGemmaPrompt = String(data.prompt || "");
      throw error;
    }
    render();
    return { ...data, used_storyboard_prompt_writer: true };
  }

  async function generateT2IPromptForSegment(segment, progress = null, percent = 30, label = "Gemma T2I", options = {}) {
    const missing = t2iMissingReason(segment);
    if (missing) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: ${missing}`);
    state.activeId = segment.id;
    syncInspector();
    return generateStoryboardT2IPromptForSegment(segment, progress, percent, label, options);
  }

  function addSceneImageHistoryPath(segment, imagePath) {
    ensureSegmentRuntimeFields(segment);
    if (!segment || !imagePath) return;
    refreshEditorThumbnailUrl(imagePath);
    const existingIndex = segment.image_history.findIndex((item) => mediaPathKey(item) === mediaPathKey(imagePath));
    if (existingIndex >= 0) {
      segment.image_history_index = existingIndex;
    } else {
      segment.image_history.push(imagePath);
      segment.image_history_index = segment.image_history.length - 1;
    }
    segment.image_assignment_cleared = false;
    segment.preview_mode = "image";
  }

  function requireActiveSegment() {
    const segment = activeSegment();
    if (!segment) {
      toast("Hey, add a segment first.", true);
      return null;
    }
    return segment;
  }

  function assertBatchNotStopped() {
    if (state.batchCancelled) throw new Error("Stopped by user.");
  }

  function sceneVideoConceptPromptText(segment) {
    const direct = sceneConceptPromptText(segment);
    if (direct) return direct;
    const parts = [];
    const add = (title, value) => {
      const text = String(value || "").trim();
      if (text) parts.push(`${title}:\n${text}`);
    };
    add("Prompt summary", segment?.prompt_summary || segment?.summary);
    add("Scene notes", segment?.notes || segment?.director_note);
    add("Scene story beat", segment?.story_beat);
    if (!segmentUsesNoLipSyncPerformance(segment)) {
      add("Lyrics / scene text", segment?.lyric_text || segment?.lyric_note || segment?.lyrics);
    }
    add("Mapped subject / character", segment?.no_character_present ? "" : segmentMappedSubjectText(segment));
    add("Mapped location", segmentMappedLocationText(segment));
    add("Shot type", segment?.shot_type);
    const context = fluxReferenceContextForSegment(segment);
    add("Reference subject description", segment?.no_character_present ? "" : context?.subject_description);
    add("Reference location name", context?.location_name);
    add("Reference location description", context?.location_description);
    return parts.join("\n\n").trim();
  }

  async function generateTextOnlyImagePromptFallbackForSegment(segment, progress = null, percent = 30, label = "Gemma text-only fallback", options = {}) {
    state.activeId = segment.id;
    syncInspector();
    const imageMode = options.imageMode || state.imageModelMode || "zimage";
    const textModelSelect = imageMode === "ernie_image" ? ernieTextGemmaModelSelect : t2iTextGemmaModelSelect;
    let userNotes = String(options.userNotes || "").trim() || textOnlyFallbackNotesForSegment(segment, imageMode);
    ({ userNotes } = applyImageContinuityToPromptSettings(segment, { image_ingredients: [] }, userNotes));
    const referenceContext = imageMode === "nano_banana" ? nbImageSettingsForSegment(segment).reference_context : imageMode === "flux_klein" ? fluxReferenceContextForSegment(segment) : {};
    progress?.set(`${label}: creating prompt from notes with non-vision Gemma...\n${gemmaRunnerLine({ vision: false })}`, percent);
    const data = await postJson("/vrgdg/music_builder/generate_t2i", {
      ...textGemmaRunnerPayload(),
      model_file: textModelSelect.value,
      mmproj_file: "",
      use_vision: false,
      ref_image_path: "",
      scene_number: sceneSlotNumber(segment),
      lyric_text: sceneLyricTextForPromptValidation(segment),
      prompt_mode: imageMode,
      project_folder: activeProjectFolderForSave(),
      scene_id: segment.id || "",
      builder_instruction_key: ["flux_klein", "nano_banana", "flow_gpt"].includes(imageMode) ? builderImageInstructionKey(imageMode) : "",
      reference_context: referenceContext || {},
      repair_model_file: textModelSelect.value,
      user_notes: userNotes,
      theme_style_path: state.useVrgdgTextContext ? state.themeStylePath || "" : "",
      story_idea_path: state.useVrgdgTextContext ? state.storyIdeaPath || "" : "",
      subject_scene_path: state.useVrgdgTextContext ? state.subjectScenePath || "" : "",
      unload_after: true,
      seed: options.seed,
      temperature: options.temperature,
      top_p: options.topP,
    }, 120000);
    pushHistory();
    if (imageMode === "flow_gpt") {
      syncSegmentFlowGptPrompt(segment, data.prompt || "");
    } else {
      syncSegmentT2IPrompt(segment, applyMappedTriggerPhrases(applyImageTriggerToPrompt(data.prompt, segment, imageMode, { validateJunk: true }), segment));
    }
    render();
    return { ...data, used_text_only_fallback: true };
  }

  return {
    addSceneImageHistoryPath, addSegmentVideoHistoryPath, archiveGeneratedSceneImage, assertBatchNotStopped,
    cycleSegmentImageHistory, cycleSegmentVideoHistory, generateT2IPromptForSegment,
    generateTextOnlyImagePromptFallbackForSegment, isBlankStarterProject, requireActiveSegment,
    sceneVideoConceptPromptText, toggleSegmentPreviewMode,
  };
}
