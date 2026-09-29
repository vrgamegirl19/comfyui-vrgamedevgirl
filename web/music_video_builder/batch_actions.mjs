import { normalizeOverlayTrackState } from "../VRGDG_OverlayTrack.js";
import { GEMMA_VIDEO_PROMPT_TIMEOUT_MS, postJson } from "./comfy_api.mjs";
import {
  makeButton,
  makeCheckbox,
  makeField,
  makeInput,
  makeSelect,
  normalizeProjectVideoEngine,
  toast,
} from "./controls.mjs";
import { chooseBatchModeAction, timestampForProjectName } from "./project_actions.mjs";
import { normalizeBatchScope } from "./timeline_state.mjs";

export function confirmDeleteMediaAction(type, path) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #7f1d1d;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = `Delete selected ${type}?`;
    heading.style.cssText = "font-size:16px;font-weight:900;color:#fecaca;";
    const body = document.createElement("div");
    body.textContent = `This deletes the file from the current project folder and removes it from this scene history.`;
    body.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;";
    const pathBox = document.createElement("div");
    pathBox.textContent = path;
    pathBox.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:9px;color:#bae6fd;font-size:11px;overflow-wrap:anywhere;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const confirm = makeButton(`Delete ${type}`, "danger");
    confirm.style.borderColor = "#7f1d1d";
    confirm.style.background = "#991b1b";
    confirm.style.color = "#fee2e2";
    cancel.onclick = () => {
      backdrop.remove();
      resolve(false);
    };
    confirm.onclick = () => {
      backdrop.remove();
      resolve(true);
    };
    actions.append(cancel, confirm);
    box.append(heading, body, pathBox, actions);
    backdrop.append(box);
    document.body.append(backdrop);
  });
}

function showBranchProjectModal(defaultName = "") {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:20px;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(760px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow:auto;border:1px solid #155e75;border-radius:9px;background:#111827;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.65);padding:18px;display:flex;flex-direction:column;gap:14px;";
    const title = document.createElement("div");
    title.textContent = "Branch Project — Make a Safe Copy";
    title.style.cssText = "font-size:18px;font-weight:900;color:#cffafe;";
    const safety = document.createElement("div");
    safety.innerHTML = "<strong>Your current project will not be changed.</strong><br>This creates a separate new project folder, then opens the new copy.";
    safety.style.cssText = "border:1px solid #15803d;border-radius:7px;background:#052e16;color:#dcfce7;padding:11px;font-size:13px;line-height:1.5;";
    const name = makeInput(defaultName);
    name.placeholder = "Name for the new project copy";
    const preset = makeSelect([
      { value: "fresh_media", label: "Fresh Media — keep scenes, audio, and lyrics only (Recommended)" },
      { value: "full", label: "Full Copy — keep everything" },
      { value: "scenes_audio", label: "Scenes + Audio — remove lyrics and all creative/media data" },
      { value: "custom", label: "Custom Copy — choose what to keep" },
    ]);
    const explanation = document.createElement("div");
    explanation.style.cssText = "border:1px solid #334155;border-radius:7px;background:#18181b;padding:11px;font-size:12px;line-height:1.5;color:#d4d4d8;";
    const custom = document.createElement("div");
    custom.style.cssText = "display:none;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;border:1px solid #334155;border-radius:7px;padding:10px;";
    const customOptions = {
      lyrics: makeCheckbox("Keep lyrics/dialogue", true),
      notes: makeCheckbox("Keep scene and video notes", false),
      prompts: makeCheckbox("Keep image and video prompts", false),
      mappings: makeCheckbox("Keep character/location mappings", false),
      images: makeCheckbox("Keep images and image history", false),
      videos: makeCheckbox("Keep videos, redos, and backups", false),
      overlays: makeCheckbox("Keep overlay/insert clips", false),
    };
    Object.values(customOptions).forEach((option) => custom.append(option.wrapper));
    const descriptions = {
      fresh_media: "Best when you finished one version and want to create new visuals. Keeps scene order and timing, project audio, and lyrics/dialogue. Removes images, videos, histories, backups, overlays, prompts, notes, and mappings.",
      full: "Makes a complete independent copy of the project, including Storyboard Builder data, Reference Builder files, current media, histories, prompts, notes, mappings, and overlays. The new folder does not keep paths into the original project.",
      scenes_audio: "Keeps only empty scene slots with their timing and the project audio. Lyrics, notes, prompts, mappings, images, videos, and overlays are removed.",
      custom: "Choose the extra information you want in the new copy. Scene order, timing, and project audio are always kept.",
    };
    const syncPreset = () => {
      explanation.textContent = descriptions[preset.value] || descriptions.fresh_media;
      custom.style.display = preset.value === "custom" ? "grid" : "none";
    };
    preset.onchange = syncPreset;
    syncPreset();
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const create = makeButton("Create and Open New Copy", "primary");
    const finish = (result) => { backdrop.remove(); resolve(result); };
    cancel.onclick = () => finish(null);
    create.onclick = () => {
      const target = String(name.value || "").trim();
      if (!target) { name.focus(); return; }
      finish({
        target,
        preset: preset.value,
        keep: Object.fromEntries(Object.entries(customOptions).map(([key, option]) => [key, option.input.checked])),
      });
    };
    backdrop.onpointerdown = (event) => { if (event.target === backdrop) finish(null); };
    actions.append(cancel, create);
    box.append(title, safety, makeField("New project name or full folder path", name), makeField("What should the new copy keep?", preset), explanation, custom, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    name.focus();
    name.select();
  });
}

function branchKeepFlags(preset, customKeep = {}) {
  if (preset === "full") {
    return { lyrics: true, notes: true, prompts: true, mappings: true, images: true, videos: true, overlays: true };
  }
  if (preset === "custom") return { ...customKeep };
  return {
    lyrics: preset === "fresh_media",
    notes: false,
    prompts: false,
    mappings: false,
    images: false,
    videos: false,
    overlays: false,
  };
}

export function createBatchActions({
  activeSegment, allEditableSegments, audioInput, autoSaveSessionQuiet, batchScopeChoices,
  buildFullVideoPipeline, buildIndependentFLFPairs, createEndFramesAllScenes, createFLFImageChain,
  currentSessionData, currentVideoMode, deleteAllTimelineImagesButton, deleteAllTimelineVideosButton,
  deleteStaleSceneLatents, ensureSegmentRuntimeFields, ernieImageAllScenes, flowGptImageAllScenes,
  fluxKleinAllScenes, gemmaT2IAllScenes, gemmaVideoAllTextOnly, getPreferredProjectRoot,
  hasFirstLastFrameEndImage, krea2TwoPassImageAllScenes, loadSessionFromProject, nbImageAllScenes,
  pauseTimelineForEditing, previewImage, previewVideo, projectInput, promptRunnerActionName, pushHistory,
  render, renderAllScenes, renderList, rtvReferenceBehaviorGlobalValue, saveI2VVideoSettingsFromPanel,
  sceneAudio, sceneListPane, segmentImageSource, segmentLayer, state, syncFluxKleinPanel, syncInspector,
  syncPreview, updateActiveFromInputs, updateSelectedMediaTools, videoModeDisplayLabel,
  videoVisionReferenceEnabled, zEnhanceAllScenes, zEnhanceBatchTargets, zImageAllScenes,
}) {
  async function confirmAndRunZImageAll() {
    const imageMode = state.imageModelMode || "zimage";
    const useFluxKleinMode = imageMode === "flux_klein";
    const useNBMode = imageMode === "nano_banana";
    const useErnieMode = imageMode === "ernie_image";
    const useKrea2TwoPassMode = imageMode === "krea2_2pass";
    const useFlowGptMode = imageMode === "flow_gpt";
    const modelLabel = useFluxKleinMode ? "Flux/Klein" : useNBMode ? "NanoBanana" : useErnieMode ? "Ernie" : useKrea2TwoPassMode ? "Krea 2" : useFlowGptMode ? "Flow/GPT" : "ZImage";
    const imageModeChoices = [
      {
        value: "zimage",
        label: "ZImage",
        description: "Use the ZImage image workflow. Does not require Nano B reference images.",
      },
      {
        value: "flux_klein",
        label: "Flux/Klein",
        description: "Use Flux/Klein. Reference images are optional.",
      },
      {
        value: "nano_banana",
        label: "Nano B",
        description: "Use Nano B. Reference images are optional; API key is still required.",
      },
      {
        value: "ernie_image",
        label: "Ernie",
        description: "Use the Ernie image workflow.",
      },
      {
        value: "krea2_2pass",
        label: "Krea 2",
        description: "Use the Krea 2 workflow with optional image-to-image.",
      },
      {
        value: "flow_gpt",
        label: "Flow/GPT",
        description: "Browser image providers. Step 3 will wire generation; setup/login can be configured now.",
      },
    ];
    const currentChoice = imageModeChoices.find((choice) => choice.value === imageMode) || imageModeChoices[0];
    const orderedImageModeChoices = [
      currentChoice,
      ...imageModeChoices.filter((choice) => choice.value !== currentChoice.value),
    ];
    const scopeChoices = batchScopeChoices();
    const firstLastFrameMode = rtvReferenceBehaviorGlobalValue() === "first_last_frame";
    const nativeFLFMode = currentVideoMode() === "flf";
    const action = await chooseBatchModeAction({
      title: "Run Image All?",
      intro: nativeFLFMode
        ? `Current image model: ${modelLabel}. First Last Frame mode is active. Normal choices only create images; Build Independent Start + End Pairs also creates the provisional motion plans and final two-image FLF prompts required by that image workflow. It never renders or stitches videos.`
        : firstLastFrameMode
        ? `Image All only works on the image stage. Current image model: ${modelLabel}. First Last Frame reference behavior is active, so you can also create end frames from the existing scene images.`
        : `Image All only works on the image stage. It does not create I2V prompts, render videos, or stitch the final video. Current image model: ${modelLabel}. Flux ingredients, model selections, LoRAs, notes, and project paths are not reset.`,
      confirmLabel: "Run Image All",
      returnAll: true,
      choices: [
        ...(nativeFLFMode ? [{
          value: "build_independent_pairs",
          label: "Build Independent Start + End Pairs",
          description: "Recommended independent workflow. Pass 1 finishes ALL scene start images. When adjacent scenes share a mapped location, the later start uses the previous start as a do-not-copy composition reference and must move to a different camera position and area inside that 3D location. Pass 2 creates ALL provisional motion plans. Pass 3 returns to the first target scene and creates ALL end images. Pass 4 inspects each real pair and creates the final FLF prompts. Keeps completed work and resumes only missing stages.",
        }, {
          value: "redo_independent_endpoints",
          label: "Redo Independent Motion Plans + Ends",
          description: "Keep every current start image. Recreate all target motion plans, end-frame prompts, end images, and final two-image FLF prompts. Previous-end chaining is turned off.",
        }, {
          value: "redo_independent_all",
          label: "Redo ALL Independent Starts + Ends",
          description: "Rebuild all four passes for the target scenes: new start images first, then motion plans, end images, and final FLF prompts. This replaces the selected start/end results while preserving normal image history where supported.",
        }] : []),
        ...(firstLastFrameMode ? [{
          value: "create_flf_image_chain",
          label: "Resume FLF Image Chain",
          description: "Recommended for a fresh FLF project. Generate one opening image for the first scene, then only one destination image per scene. Each destination becomes the next scene's start. If adjacent scenes share a mapped location, the next destination must travel to a clearly different area and camera composition inside that location; no unused later start images are created.",
        }, {
          value: "redo_flf_image_chain_keep_prompts",
          label: "Redo FLF Image Chain — keep prompts",
          description: "Use the existing saved image prompts directly—no Gemma/LLM call—and regenerate the target scenes' destination images. When starting later in the timeline, reuse the prior scene's endpoint as the chain start.",
        }, {
          value: "create_end_frames",
          label: "Create End Frames",
          description: "First Last Frame mode only. Keep current scene images as first frames and generate missing end-frame images.",
        }] : []),
        {
          value: "resume_missing",
          label: "Resume missing images",
          description: "Safe resume mode. Keep existing image prompts and selected images. Only run Gemma/create images for scenes that do not already have an image.",
        },
        {
          value: "keep_prompts_redo_images",
          label: "Keep prompts, redo images",
          description: "Keep saved ZImage/Ernie/Flux prompts, randomize image seeds, and create a new image version for every scene.",
        },
        {
          value: "redo_prompts_images",
          label: "Redo image prompts and images",
          description: "Regenerate image prompts with Gemma, randomize image seeds, and create a new image version for every scene.",
        },
      ],
      extraGroups: [
        ...(scopeChoices.length ? [{
          key: "sceneScope",
          label: "Scenes to run",
          description: "Choose all scenes, start from the active clip, or run only the scenes selected with Select Multi.",
          choices: scopeChoices,
        }] : []),
        {
          key: "imageMode",
          label: "Image model to run",
          description: "This explicit choice controls which Image All pipeline runs.",
          choices: orderedImageModeChoices,
        },
      ],
    }, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
    if (!action?.mode) return;
    const selectedImageMode = ["zimage", "flux_klein", "nano_banana", "ernie_image", "krea2_2pass", "flow_gpt"].includes(action.imageMode) ? action.imageMode : imageMode;
    const sceneScope = normalizeBatchScope(action.sceneScope);
    state.imageModelMode = selectedImageMode;
    state.fluxKleinSettings.image_model_mode = selectedImageMode;
    state.fluxKleinSettings.enabled = selectedImageMode === "flux_klein";
    syncFluxKleinPanel();
    if (["build_independent_pairs", "redo_independent_endpoints", "redo_independent_all"].includes(action.mode)) {
      await buildIndependentFLFPairs({
        imageMode: selectedImageMode,
        sceneScope,
        runMode: action.mode === "redo_independent_all" ? "redo_all" : action.mode === "redo_independent_endpoints" ? "redo_endpoints" : "resume_missing",
      });
      return;
    }
    if (["create_flf_image_chain", "redo_flf_image_chain_keep_prompts"].includes(action.mode)) {
      await createFLFImageChain({
        imageMode: selectedImageMode,
        sceneScope,
        redoImages: action.mode === "redo_flf_image_chain_keep_prompts",
        useExistingPrompts: action.mode === "redo_flf_image_chain_keep_prompts",
      });
      return;
    }
    if (action.mode === "create_end_frames") {
      await createEndFramesAllScenes({ imageMode: selectedImageMode, sceneScope, endFrameRunMode: "resume_missing" });
      return;
    }
    if (selectedImageMode === "flux_klein") await fluxKleinAllScenes({ imageRunMode: action.mode, sceneScope });
    else if (selectedImageMode === "nano_banana") await nbImageAllScenes({ imageRunMode: action.mode, sceneScope });
    else if (selectedImageMode === "ernie_image") await ernieImageAllScenes({ imageRunMode: action.mode, sceneScope });
    else if (selectedImageMode === "krea2_2pass") await krea2TwoPassImageAllScenes({ imageRunMode: action.mode, sceneScope });
    else if (selectedImageMode === "flow_gpt") await flowGptImageAllScenes({ imageRunMode: action.mode, sceneScope });
    else await zImageAllScenes({ imageRunMode: action.mode, sceneScope });
  }

  function branchProjectSession(preset, customKeep = {}) {
    const source = currentSessionData();
    const keep = branchKeepFlags(preset, customKeep);
    if (preset === "full") return source;
    const imageKeys = ["custom_image_path", "custom_image_data", "custom_image_name", "approved_image_path", "image", "image_history", "image_history_index", "ref_image_path", "flux_subject_image_path", "flux_location_image_path", "image_output", "image_status", "minimax_h3_continuity_frame_path", "adjust_preview_image_path"];
    const videoKeys = ["video_path", "video_folder", "video_thumbnail_path", "video_history", "video_thumbnail_history", "video_backup_paths", "video_backup_thumbnail_paths", "video_history_index", "video_output", "video_original_path", "video_original_thumbnail_path", "video_source_path", "minimax_h3_continuity_source_video_path", "minimax_h3_video_references", "minimax_h3_stage1_path", "minimax_h3_stage1_source_path", "minimax_h3_stage1_backup_path", "minimax_h3_stage2_path", "minimax_h3_stage2_source_path", "minimax_h3_stage2_backup_path"];
    const noteKeys = ["timeline_note", "notes", "i2v_notes", "flux_notes", "nb_notes", "enhance_notes", "story_beat"];
    const promptKeys = ["t2i_prompt", "flux_prompt", "nb_prompt", "enhance_prompt", "i2v_prompt", "t2v_prompt", "flow_gpt_prompt", "ernie_t2i_prompt", "krea2_t2i_prompt", "minimax_h3_prompt", "minimax_h3_pass2_prompt"];
    const mappingKeys = ["subject_ids", "location_id", "reference_subject_ids", "reference_location_id", "flux_image_ingredients", "nb_image_ingredients", "minimax_h3_reference_keys"];
    const cleanSegment = (raw) => {
      const segment = typeof structuredClone === "function" ? structuredClone(raw) : JSON.parse(JSON.stringify(raw));
      if (!keep.lyrics) segment.lyric_text = "";
      if (!keep.notes) noteKeys.forEach((key) => { if (key in segment) segment[key] = ""; });
      if (!keep.prompts) promptKeys.forEach((key) => { if (key in segment) segment[key] = ""; });
      if (!keep.mappings) mappingKeys.forEach((key) => { if (key in segment) segment[key] = Array.isArray(segment[key]) ? [] : ""; });
      if (!keep.images) {
        imageKeys.forEach((key) => { if (key in segment) segment[key] = Array.isArray(segment[key]) ? [] : key.endsWith("_index") ? -1 : key === "image" || key === "image_output" ? null : key === "image_status" ? "none" : ""; });
        segment.preview_mode = "image";
      }
      if (!keep.videos) {
        videoKeys.forEach((key) => { if (key in segment) segment[key] = Array.isArray(segment[key]) ? [] : key.endsWith("_index") ? -1 : key === "video_output" ? null : ""; });
        segment.video_status = "none";
      }
      if (!keep.images && !keep.videos) {
        segment.custom_audio_path = "";
        segment.custom_audio_name = "";
      }
      return segment;
    };
    const session = { ...source, segments: source.segments.map(cleanSegment) };
    session.overlay_segments = keep.overlays ? source.overlay_segments.map(cleanSegment) : [];
    session.overlay_track = normalizeOverlayTrackState({ enabled: keep.overlays && source.overlay_track?.enabled });
    if (!keep.notes) session.timeline_markers = [];
    if (!keep.prompts) {
      session.prompt_json_path = "";
      session.i2v_motion_json_path = "";
      session.theme_style_path = "";
      session.story_idea_path = "";
      session.subject_scene_path = "";
    }
    if (!keep.mappings) {
      session.flux_reference_builder = {};
      session.id_lora_reference_builder = {};
    }
    if (!keep.images) {
      session.flux_global_image_ingredients = [];
      session.use_flux_global_image_ingredients = false;
      session.builder_agent_reference_images = [];
      session.builder_story_reference_images = [];
      session.builder_story_source_path = "";
    }
    if (!keep.lyrics && session.lyric_mapper) session.lyric_mapper = { ...session.lyric_mapper, source_text: "" };
    return session;
  }

  async function branchProject() {
    const currentProject = String(projectInput.value || state.projectFolder || "").trim();
    if (!currentProject) {
      toast("Create or load a project before making a branch copy.", true);
      return;
    }
    const currentName = currentProject.split(/[\\/]/).filter(Boolean).pop() || "VRGDG_Project";
    const choice = await showBranchProjectModal(`${currentName}_Fresh_${timestampForProjectName()}`);
    if (!choice) return;
    try {
      updateActiveFromInputs();
      saveI2VVideoSettingsFromPanel();
      const session = branchProjectSession(choice.preset, choice.keep);
      const data = await postJson("/vrgdg/music_builder/save_project_as", {
        source_project_folder: currentProject,
        target_project_folder: choice.target,
        project_root: getPreferredProjectRoot(),
        audio_path: audioInput.value,
        session,
        keep: branchKeepFlags(choice.preset, choice.keep),
      }, 120000);
      await loadSessionFromProject(data.project_folder || choice.target);
      toast(`New project copy created and opened.\nYour original project was not changed.\n${state.projectFolder}`);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  async function deleteAllTimelineVideos() {
    const segments = allEditableSegments();
    const assignedSegments = segments.filter((segment) => Boolean(
      String(segment.video_path || "").trim()
      || (Array.isArray(segment.video_history) && segment.video_history.some((path) => String(path || "").trim()))
      || (Array.isArray(segment.video_backup_paths) && segment.video_backup_paths.some((path) => String(path || "").trim()))
      || String(segment.video_original_path || "").trim()
    ));
    const videoPaths = [...new Set(assignedSegments.flatMap((segment) => [
      segment.video_path || "",
      ...(Array.isArray(segment.video_history) ? segment.video_history : []),
      ...(Array.isArray(segment.video_backup_paths) ? segment.video_backup_paths : []),
      segment.video_original_path || "",
    ]).map((path) => String(path || "").trim()).filter(Boolean))];
    const thumbnailPaths = [...new Set(assignedSegments.flatMap((segment) => [
      segment.video_thumbnail_path || "",
      ...(Array.isArray(segment.video_thumbnail_history) ? segment.video_thumbnail_history : []),
      ...(Array.isArray(segment.video_backup_thumbnail_paths) ? segment.video_backup_thumbnail_paths : []),
      segment.video_original_thumbnail_path || "",
    ]).map((path) => String(path || "").trim()).filter(Boolean))];
    const imagePaths = [...new Set(segments.flatMap((segment) => [
      ...(Array.isArray(segment.image_history) ? segment.image_history : []),
      segment.approved_image_path || "",
      segment.custom_image_path || "",
      segment.first_last_frame_end_image_path || "",
      segment.flf_rendered_start_frame_path || "",
      segment.minimax_h3_continuity_frame_path || "",
      segment.auto_img2img_source_frame_path || "",
    ]).map((path) => String(path || "").trim()).filter(Boolean))];
    const hasSceneImages = imagePaths.length > 0 || segments.some((segment) =>
      segment.custom_image_data || segment.first_last_frame_end_image_data || segment.flf_rendered_start_frame_data || segment.image);
    if (!assignedSegments.length && !hasSceneImages) {
      toast("There are no timeline videos or scene images to remove.");
      return;
    }
    const ok = window.confirm(
      `Permanently delete ALL ${videoPaths.length} video file${videoPaths.length === 1 ? "" : "s"} from ${assignedSegments.length} scene${assignedSegments.length === 1 ? "" : "s"}?\n\nThis deletes the video files and thumbnails from the project folder, the same as Delete Video does for one scene, but for every scene. This cannot be undone.\n\nSaved scene latents (Latent Continuation) are deleted too, since they no longer match any video.`
    );
    if (!ok) return;

    const deleteSceneImages = hasSceneImages && window.confirm(
      "Also permanently delete ALL timeline scene images and image history?\n\nThis removes captured video frames, first/last frames, continuity frames, and imported or generated scene images. Older captured frames cannot be distinguished from ordinary scene images. Reference Builder assets are not included.\n\nOK = delete scene images too. Cancel = keep scene images and delete videos only."
    );
    if (!assignedSegments.length && !deleteSceneImages) return;
    const mediaPaths = [...new Set([...videoPaths, ...thumbnailPaths, ...(deleteSceneImages ? imagePaths : [])])];

    try {
      deleteAllTimelineVideosButton.disabled = true;
      deleteAllTimelineVideosButton.textContent = "Removing ALL...";
      pauseTimelineForEditing();
      pushHistory();
      const projectFolder = String(state.projectFolder || projectInput.value || "").trim();
      let deleteFailures = 0;
      for (const path of mediaPaths) {
        try {
          await postJson("/vrgdg/music_builder/delete_project_media", { project_folder: projectFolder, path });
        } catch (error) {
          deleteFailures += 1;
        }
      }
      if (deleteFailures) {
        throw new Error(`Some files were deleted, but ${deleteFailures} file${deleteFailures === 1 ? "" : "s"} could not be deleted. Timeline references were kept so you can retry Delete ALL Videos.`);
      }
      for (const segment of segments) {
        segment.video_path = "";
        segment.video_source_path = "";
        segment.video_thumbnail_path = "";
        segment.video_history = [];
        segment.video_thumbnail_history = [];
        segment.video_backup_paths = [];
        segment.video_backup_thumbnail_paths = [];
        segment.video_history_index = -1;
        segment.video_output = null;
        segment.video_original_path = "";
        segment.video_original_thumbnail_path = "";
        segment.video_status = "none";
        segment.video_cache_bust = Date.now();
        segment.minimax_h3_trimmed_video = false;
        segment.minimax_h3_trim_side = "";
        segment.minimax_h3_trim_removed_seconds = 0;
        segment.minimax_h3_trim_source_start_seconds = 0;
        segment.minimax_h3_trimmed_duration_seconds = 0;
        segment.minimax_h3_continuity_frame_path = "";
        segment.minimax_h3_continuity_source_video_path = "";
        segment.minimax_h3_continuity_source_scene_id = "";
        segment.minimax_h3_continuity_image_number = 0;
        segment.flf_rendered_source_video_path = "";
        segment.preview_mode = "image";
        if (deleteSceneImages) {
          segment.image = null;
          segment.image_history = [];
          segment.image_history_index = -1;
          segment.image_assignment_cleared = true;
          for (const key of [
            "approved_image_path", "custom_image_path", "custom_image_data", "custom_image_name",
            "first_last_frame_end_image_path", "first_last_frame_end_image_data", "first_last_frame_end_image_name",
            "flf_rendered_start_frame_path", "flf_rendered_start_frame_data", "flf_rendered_start_frame_name",
            "auto_img2img_source_frame_path", "auto_img2img_source_video_path",
          ]) segment[key] = "";
          segment.flf_final_prompt_ready = false;
        }
        ensureSegmentRuntimeFields(segment);
      }
      await deleteStaleSceneLatents();
      segments.forEach((segment) => { segment._latentDirty = false; });
      previewVideo.pause();
      previewVideo.removeAttribute("src");
      previewVideo.dataset.path = "";
      previewVideo.dataset.cacheKey = "";
      previewVideo.dataset.segmentId = "";
      previewVideo.style.display = "none";
      sceneAudio.pause();
      sceneAudio.removeAttribute("src");
      sceneAudio.load();
      state.sceneAudioSegmentId = "";
      if (deleteSceneImages) previewImage.removeAttribute("src");
      syncInspector();
      syncPreview(activeSegment());
      renderList();
      render();
      await autoSaveSessionQuiet(deleteSceneImages ? "all timeline videos and scene images removed" : "all timeline videos removed");
      updateSelectedMediaTools();
      toast(`Deleted ${videoPaths.length} video file${videoPaths.length === 1 ? "" : "s"} from ${assignedSegments.length} scene${assignedSegments.length === 1 ? "" : "s"}.${deleteSceneImages ? " Timeline scene images and frame histories were also deleted." : ""}${deleteFailures ? ` ${deleteFailures} file${deleteFailures === 1 ? "" : "s"} could not be deleted.` : ""}`, Boolean(deleteFailures));
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      deleteAllTimelineVideosButton.disabled = false;
      deleteAllTimelineVideosButton.textContent = "Delete ALL Videos";
    }
  }

  async function deleteAllTimelineImages() {
    const segments = allEditableSegments();
    const imagePaths = [...new Set(segments.flatMap((segment) => [
      ...(Array.isArray(segment.image_history) ? segment.image_history : []),
      segment.approved_image_path || "",
      segment.custom_image_path || "",
      segment.first_last_frame_end_image_path || "",
      segment.flf_rendered_start_frame_path || "",
    ]).map((item) => String(item || "").trim()).filter(Boolean))];
    const assignedCount = segments.filter((segment) => Boolean(
      segmentImageSource(segment)
      || hasFirstLastFrameEndImage(segment)
      || segment.flf_rendered_start_frame_path
      || segment.flf_rendered_start_frame_data
    )).length;
    if (!assignedCount && !imagePaths.length) {
      toast("There are no regular timeline images to delete.");
      return;
    }
    const ok = window.confirm(
      `Delete ALL timeline images?\n\nThis will clear every first frame, last frame, regular scene image, and extracted chained start frame from ${assignedCount} scene${assignedCount === 1 ? "" : "s"}, and delete ${imagePaths.length} image file${imagePaths.length === 1 ? "" : "s"} from the current project folder.\n\nThis cannot be undone.`
    );
    if (!ok) return;
    try {
      deleteAllTimelineImagesButton.disabled = true;
      deleteAllTimelineImagesButton.textContent = "Deleting ALL...";
      for (const path of imagePaths) {
        await postJson("/vrgdg/music_builder/delete_project_media", {
          project_folder: projectInput.value,
          path,
        }).catch(() => null);
      }
      pushHistory();
      for (const segment of segments) {
        segment.image = null;
        segment.image_history = [];
        segment.image_history_index = -1;
        segment.approved_image_path = "";
        segment.custom_image_path = "";
        segment.custom_image_data = "";
        segment.custom_image_name = "";
        segment.first_last_frame_end_image_path = "";
        segment.first_last_frame_end_image_data = "";
        segment.first_last_frame_end_image_name = "";
        segment.flf_motion_plan = "";
        segment.flf_end_frame_prompt = "";
        segment.flf_end_frame_stale = false;
        segment.flf_final_prompt_ready = false;
        segment.flf_rendered_start_frame_path = "";
        segment.flf_rendered_start_frame_data = "";
        segment.flf_rendered_start_frame_name = "";
        segment.flf_rendered_source_video_path = "";
        segment.image_assignment_cleared = true;
        segment.preview_mode = "image";
        ensureSegmentRuntimeFields(segment);
      }
      previewImage.removeAttribute("src");
      segmentLayer.textContent = "";
      sceneListPane.textContent = "";
      syncInspector();
      syncPreview(activeSegment());
      renderList();
      render();
      await autoSaveSessionQuiet("all timeline images deleted");
      updateSelectedMediaTools();
      toast(`Deleted all timeline images, including first and last frames, from ${assignedCount} scene${assignedCount === 1 ? "" : "s"}.`);
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      deleteAllTimelineImagesButton.disabled = false;
      deleteAllTimelineImagesButton.textContent = "Delete ALL Images";
    }
  }

  async function confirmAndRunZEnhanceAll() {
    const scopeChoices = batchScopeChoices();
    const imageTargets = zEnhanceBatchTargets("all").length;
    const action = await chooseBatchModeAction({
      title: "Enhance All Images?",
      intro: `Enhance All runs the current Enhance settings on existing timeline images only. It does not create missing scene images or rewrite prompts. Found ${imageTargets} scene image${imageTargets === 1 ? "" : "s"} in the full timeline.`,
      confirmLabel: "Run Enhance All",
      returnAll: true,
      choices: [
        {
          value: "enhance_existing",
          label: "Enhance existing images",
          description: "Use each scene's current T2I/image prompt so every enhanced image keeps the right scene content.",
        },
      ],
      extraGroups: [
        ...(scopeChoices.length ? [{
          key: "sceneScope",
          label: "Scenes to run",
          description: "Choose all scenes, start from the active clip, or run only the scenes selected with Select Multi.",
          choices: scopeChoices,
        }] : []),
      ],
    }, GEMMA_VIDEO_PROMPT_TIMEOUT_MS);
    if (!action?.mode) return;
    await zEnhanceAllScenes({ sceneScope: normalizeBatchScope(action.sceneScope) });
  }

  async function confirmAndRunGemmaT2IAll() {
    const imageMode = state.imageModelMode || "zimage";
    const modelLabel = imageMode === "flux_klein" ? "Flux/Klein" : imageMode === "nano_banana" ? "NanoBanana" : imageMode === "ernie_image" ? "Ernie" : imageMode === "krea2_2pass" ? "Krea 2" : "ZImage";
    const runnerName = promptRunnerActionName();
    const scopeChoices = batchScopeChoices();
    const action = await chooseBatchModeAction({
      title: `Run ${runnerName} T2I All?`,
      intro: `This only creates text-to-image prompts for review. It will not create images, videos, or the final stitched video. Current image model: ${modelLabel}.`,
      confirmLabel: `Run ${runnerName} T2I All`,
      returnAll: true,
      choices: [
        {
          value: "missing_only",
          label: "Missing prompts only",
          description: `Keep existing T2I/Flux prompts. Only run ${runnerName} for scenes with no saved image prompt.`,
        },
        {
          value: "redo_all",
          label: "Redo all T2I prompts",
          description: `Replace every scene's saved T2I/Flux prompt with a fresh ${runnerName} prompt. Images and videos stay untouched.`,
        },
      ],
      extraGroups: scopeChoices.length ? [{
        key: "sceneScope",
        label: "Scenes to run",
        description: "Choose all scenes, start from the active clip, or run only the scenes selected with Select Multi.",
        choices: scopeChoices,
      }] : [],
    });
    if (action?.mode) await gemmaT2IAllScenes({ promptRunMode: action.mode, sceneScope: normalizeBatchScope(action.sceneScope) });
  }

  async function confirmAndRunGemmaVideoAll() {
    const videoMode = currentVideoMode();
    const videoLabel = videoModeDisplayLabel(videoMode, true);
    const runnerName = promptRunnerActionName();
    const targets = allEditableSegments().filter((segment) => !String(segment.i2v_prompt || "").trim());
    const hasVisionTargets = targets.length
      ? targets.some((segment) => videoVisionReferenceEnabled(segment))
      : allEditableSegments().some((segment) => videoVisionReferenceEnabled(segment));
    const textChoices = [
      {
        value: "missing_text",
        label: "Missing, text only",
        description: `Keep existing ${videoLabel} prompts. Use the selected text runner and T2I/concept prompt plus motion notes for scenes with no saved video prompt.`,
      },
      {
        value: "redo_text",
        label: `Redo all, text only`,
        description: "Replace every scene's saved video prompt using the selected text runner. Images and videos stay untouched.",
      },
    ];
    const enhanceChoices = [
      {
        value: "enhance_existing",
        label: "Enhance existing prompts only",
        description: `Do not create new ${videoLabel} drafts. Load the text LLM runner once, rewrite existing video prompts with the enhancement pass, then unload at the end.`,
      },
    ];
    const visionChoices = [
      {
        value: "missing_vision",
        label: "Missing, vision",
        description: `Keep existing ${videoLabel} prompts. Use the built-in vision GGUF to look at each selected scene/reference image plus motion notes for missing prompts.`,
      },
      {
        value: "redo_vision",
        label: `Redo all, vision`,
        description: "Replace every scene's saved video prompt using the built-in vision GGUF and each scene/reference image. Images and videos stay untouched.",
      },
    ];
    const scopeChoices = batchScopeChoices();
    const action = await chooseBatchModeAction({
      title: `Run ${runnerName} ${videoLabel} All?`,
      intro: `This only creates ${videoLabel} prompts for review. It will not render videos or stitch the final video. ${hasVisionTargets ? "Image-reference scenes were detected, so vision options are listed first." : "No image-reference scenes were detected, so text-only options are listed first."}`,
      confirmLabel: `Run ${runnerName} ${videoLabel} All`,
      defaultValue: hasVisionTargets ? "missing_vision" : "missing_text",
      choices: hasVisionTargets ? [...visionChoices, ...textChoices, ...enhanceChoices] : [...textChoices, ...visionChoices, ...enhanceChoices],
      returnAll: true,
      extraGroups: scopeChoices.length ? [{
        key: "sceneScope",
        label: "Scenes to run",
        description: "Choose all scenes, start from the active clip, or run only the scenes selected with Select Multi.",
        choices: scopeChoices,
      }] : [],
    });
    if (!action?.mode) return;
    if (action.mode === "enhance_existing") {
      await gemmaVideoAllTextOnly({
        promptRunMode: "enhance_existing",
        sceneScope: normalizeBatchScope(action.sceneScope),
      });
      return;
    }
    await gemmaVideoAllTextOnly({
      promptRunMode: action.mode.startsWith("missing") ? "missing_only" : "redo_all",
      gemmaInputMode: action.mode.endsWith("vision") ? "vision" : "text",
      sceneScope: normalizeBatchScope(action.sceneScope),
    });
  }

  async function confirmAndRunRenderAll() {
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    const videoMode = currentVideoMode();
    const videoLabel = miniMaxProject ? "MiniMax H3" : videoModeDisplayLabel(videoMode, true);
    const scopeChoices = batchScopeChoices();
    const action = await chooseBatchModeAction({
      title: "Run Render All?",
      intro: [
        "Render All only works on the video/render stage.",
        miniMaxProject
          ? "It uses each scene's existing MiniMax prompt and effective project-global or locked MiniMax mode, models, and video settings."
          : videoMode === "t2v"
          ? "It uses existing T2V prompts and does not require scene images."
          : videoMode === "ingredients"
            ? "It uses the current selected ingredients reference images and existing Ingredients prompts."
            : "It uses the current selected images and existing I2V prompts.",
        "All-scenes mode stitches the final video when rendering is done. From-selected mode starts at the active clip and still stitches when previous clips exist. Selected-scenes mode renders only selected scenes and does not stitch.",
      ].join(" "),
      confirmLabel: "Run Render",
      returnAll: true,
      choices: [
        {
          value: "resume_missing",
          label: "Resume missing videos",
          description: `Skip scenes that already have selected videos. Render missing ${videoLabel} videos only.`,
        },
        {
          value: "redo_videos",
          label: "Redo videos",
          description: `Create new ${videoLabel} video versions for the target scenes. Existing videos are kept as backups in the picker.`,
        },
      ],
      extraGroups: scopeChoices.length ? [{
        key: "sceneScope",
        label: "Scenes to render",
        description: "Choose all scenes, start from the active clip, or render only non-adjacent scenes selected with Select Multi. Only selected-scenes mode skips final stitching.",
        choices: scopeChoices,
      }] : [],
    });
    if (action?.mode) {
      const sceneScope = normalizeBatchScope(action.sceneScope);
      await renderAllScenes({
        sceneScope,
        forceVideos: action.mode === "redo_videos",
        skipFinalStitch: sceneScope === "selected",
      });
    }
  }

  async function confirmAndRunFullBuild() {
    const videoMode = currentVideoMode();
    const t2vMode = videoMode === "t2v";
    const rtvMode = videoMode === "rtv";
    const ingredientsMode = videoMode === "ingredients";
    const videoOnlyMode = t2vMode || rtvMode;
    const videoPromptLabel = t2vMode ? "T2V" : rtvMode ? "Reference-to-Video" : ingredientsMode ? "Ingredients-to-Video" : "I2V";
    const scopeChoices = batchScopeChoices();
    const options = await chooseBatchModeAction({
      title: "Build Full Video?",
      intro: videoOnlyMode
        ? `Build Full Video is in ${videoModeDisplayLabel(videoMode)} mode. It skips image generation, creates ${videoPromptLabel} prompts, renders scene videos, and stitches the final video. Model selections, LoRAs, notes, and project paths are not reset.`
        : `Build Full Video can run the whole pipeline: image prompts, images, ${videoPromptLabel} prompts, scene videos, and final stitching. Choose how much to regenerate. Flux ingredients, model selections, LoRAs, notes, and project paths are not reset.`,
      confirmLabel: "Build Full Video",
      returnAll: true,
      choices: [
        {
          value: "resume_missing",
          label: "Resume missing only",
          description: "Safest resume mode. Keep existing prompts, selected images, and selected videos. Only create whatever is missing, then stitch the final video.",
        },
        {
          value: "fresh_rebuild",
          label: "Fresh full rebuild",
          description: videoOnlyMode
            ? `Start fresh for generated video outputs. Regenerate ${videoPromptLabel} prompts and videos. Video seeds can be randomized below.`
            : `Start fresh for generated outputs. Regenerate image prompts, images, ${videoPromptLabel} prompts, and videos. Image seeds are randomized.`,
        },
        {
          value: "redo_i2v_prompts_videos",
          label: videoOnlyMode ? `Redo ${videoPromptLabel} prompts and videos` : `Keep images, redo ${videoPromptLabel} prompts and videos`,
          description: videoOnlyMode
            ? `Regenerate ${videoPromptLabel} prompts with Gemma, then create new video versions.`
            : `Use the current selected images. Regenerate ${videoPromptLabel} prompts with Gemma, then create new video versions.`,
        },
        {
          value: "redo_videos",
          label: videoOnlyMode ? "Keep prompts, redo videos" : "Keep images and prompts, redo videos",
          description: videoOnlyMode
            ? `Use existing ${videoPromptLabel} prompts. Only create new video versions, then stitch.`
            : `Use the current selected images and existing ${videoPromptLabel} prompts. Only create new video versions, then stitch.`,
        },
      ],
      extraGroups: [
        ...(scopeChoices.length ? [{
          key: "sceneScope",
          label: "Scenes to run",
          description: "Choose all scenes, start from the active clip, or run only scenes selected with Select Multi. Only selected-scenes mode skips final stitching.",
          choices: scopeChoices,
        }] : []),
        {
          key: "videoSeedMode",
          label: "Video seed behavior",
          description: "Used only when this build creates new scene videos.",
          choices: [
            {
              value: "keep",
              label: "Keep current video seed",
              description: "Best when you like the motion and only changed LoRAs or model settings.",
            },
            {
              value: "random",
              label: "Randomize video seed",
              description: "Best when you want new motion variations.",
            },
          ],
        },
      ],
    });
    if (options?.mode) await buildFullVideoPipeline({
      buildMode: options.mode,
      videoSeedMode: options.videoSeedMode || "keep",
      sceneScope: normalizeBatchScope(options.sceneScope),
    });
  }

  return {
    branchProject, confirmAndRunFullBuild, confirmAndRunGemmaT2IAll, confirmAndRunGemmaVideoAll,
    confirmAndRunRenderAll, confirmAndRunZEnhanceAll, confirmAndRunZImageAll, deleteAllTimelineImages,
    deleteAllTimelineVideos,
  };
}
