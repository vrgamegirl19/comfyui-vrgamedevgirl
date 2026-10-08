import { cloneSceneAsOverlay, overlayClipIsEnabled } from "../VRGDG_OverlayTrack.js";
import {
  postJson,
  queueWorkflowPrompt,
  resolveComfyVideoPath,
  sceneVideoTimeoutMessage,
  waitForVideos,
} from "./comfy_api.mjs";
import { DEFAULT_LTX_INGREDIENTS_HEIGHT, DEFAULT_LTX_INGREDIENTS_WIDTH } from "./constants.mjs";
import { makeButton, makeField, makeInput, normalizeProjectVideoEngine, toast } from "./controls.mjs";
import { showFinalVideoReadyModal } from "./dialogs.mjs";
import { formatTime } from "./format.mjs";
import { rtvReferenceImagePayload } from "./image_references.mjs";
import { miniMaxI2VLastFrame, miniMaxI2VFramePaths } from "./minimax_keyframe_state.mjs";
import { setButtonGroupState } from "./inspector.mjs";
import { isMiniMaxH3LatentContinuationMode, miniMaxH3FrameSize, miniMaxH3ModeLabel, normalizeMiniMaxH3Mode, normalizeMiniMaxH3Pipeline } from "./minimax_h3.mjs";
import { refmodPreviewUrl } from "./refmod_card.mjs";
import { attachRefmodLabels, referencePayload } from "./refmod_labels.mjs";
import { miniMaxDialogueOrderText } from "./minimax_prompt.mjs";
import { applyTriggerPhrase, segmentUsesNoLipSyncPerformance } from "./prompt_text.mjs";
import { normalizeVideoPromptOrigin, sortSegments } from "./segments.mjs";
import { isLikelyVideoPath, selectedSegmentVideoPath } from "./selection_preview.mjs";
import {
  activateSegmentVideoPath,
  audioChunkDuration,
  audioSourceStart,
  mediaPathKey,
  timelineSegmentDuration,
} from "./timeline_state.mjs";

export function showMultiSelectHint() {
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
  const heading = document.createElement("div");
  heading.textContent = "Select Multi";
  heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
  const close = makeButton("Close");
  header.append(heading, close);
  const body = document.createElement("div");
  body.style.cssText = "display:flex;flex-direction:column;gap:10px;font-size:12px;line-height:1.45;color:#d4d4d8;";
  const batch = document.createElement("div");
  batch.innerHTML = `<strong style="color:#e0f2fe;">Choose scenes your way</strong><br>Click Select Multi, then either select clips directly on the timeline or enter base scene numbers as a list. Lists accept commas, spaces, new lines, ranges such as <code>4-8</code>, and shortcuts such as <code>all</code>, <code>odd</code>, and <code>even</code>.`;
  const selectedBatch = document.createElement("div");
  selectedBatch.innerHTML = `<strong style="color:#e0f2fe;">Batch render selected scenes</strong><br>When two or more scenes are selected, Image All, LLM All, Render All, and Build Full Video offer a selected-scenes option. Use it to generate prompts, images, or videos for only those selected scenes. Selected Render/Build does not stitch a final video.`;
  const preview = document.createElement("div");
  preview.innerHTML = `<strong style="color:#e0f2fe;">Stitch Preview</strong><br>Use the Stitch Preview menu option to make a quick complete video from selected scenes or from a start/end scene range. Inserts are included automatically, and no Gemma, image generation, or video rendering is run.`;
  const note = document.createElement("div");
  note.textContent = "Selected scenes turn red. Ctrl/Cmd-click toggles scenes on or off; a plain click returns to normal single-scene editing.";
  note.style.cssText = "border:1px solid #334155;border-radius:6px;background:#0f172a;padding:9px;color:#cbd5e1;";
  body.append(batch, selectedBatch, preview, note);
  const ok = makeButton("Got it", "primary");
  box.append(header, body, ok);
  backdrop.append(box);
  document.body.append(backdrop);
  close.onclick = () => backdrop.remove();
  ok.onclick = () => backdrop.remove();
  backdrop.addEventListener("pointerdown", (event) => {
    if (event.target === backdrop) backdrop.remove();
  });
}

export function createVideoRender({
  activeSegment, activeVideoOutputFolder, advancedTwoPassControls, applyMappedTriggerPhrases,
  applySceneAdjustToRenderedVideo, applySceneFilmGrainToRenderedVideo, applySceneLutToRenderedVideo,
  applyVocalDirectiveToVideoPrompt, assertValidMiniMaxH3FinalPrompt, audioInput, audioSourceDurationForScene,
  autoSaveSessionQuiet, buildIdLoraPromptForScene, collectedSceneVideoFolder, createProgressWindow,
  createSceneVideoButtons, currentProjectAudioPath, currentVideoMode, enforceAudioTimelineEnd,
  ensureAudioOrOfferSilentTimeline, ensureBuilderManagedFx, ensureSceneAdjustsAppliedBeforeStitch,
  ensureSceneFilmGrainAppliedBeforeStitch, ensureSceneLutsAppliedBeforeStitch,
  ensureSelectedImageForSceneVideo, finishSingleSceneETA, firstLastFrameEndImageSource,
  firstLastFrameResolvedEndImageSource, firstLastFrameStartImageSource, flfChainingEnabled,
  flfRenderChainStartSource, gemmaThenCreateVideoButtons, generateI2VPromptForSegment, i2vAutoChainEnabled,
  i2vImagesFolder, i2vPrompt, i2vVideoSettingsForSegment, i2vVideoSettingsPayload, idLoraSceneContext,
  loadDirtyLatentBadges, ltx25RtvSceneOrdinal, miniMaxH3ContinuityModeForSegment, miniMaxH3ModeForSegment,
  miniMaxH3PromptVisionImages, miniMaxH3SceneImageIsPromptInspiration, miniMaxH3SettingsForSegment,
  miniMaxOrderedImageReferenceItemsForSegment, miniMaxPrompt, miniMaxPromptReferenceMismatch,
  miniMaxReferenceKeysForSegment, miniMaxRenderReferenceImagePaths, miniMaxSceneVideoButtons,
  nextOverlaySlotNumber, persistIngredientsSheetImages, prepareFLFRenderedFrameNextScene,
  previousAutoChainSourceSegment, projectInput, promptRunnerActionName, pushHistory, render, renderList,
  requireActiveSegment, rtvReferencesForSegment, runClearMemoryWorkflowQuiet, runMiniMaxH3PromptGeneration,
  saveMiniMaxH3SettingsFromPanel, saveMiniMaxSceneInputsFromPanel, saveSessionForSceneVideo, sceneDisplayName,
  sceneSlotNumber, sceneVideoDetailsHtml, segmentImageSource, segmentIndexInfo, segmentTrack,
  selectedSegmentImagePath, selectedSegmentsForBatch, setActiveSegment, startSingleSceneETA, state,
  syncInspector, syncPreview, twoPassControls, updateActiveFromInputs, updateMiniMaxPromptCharacterStatus,
  usingSceneAudioMode, validateMiniMaxSceneReadyForVideo, validateSceneReadyForVideo,
  validateSrtTimingForSceneVideo, videoModeDisplayLabel, videoTriggerPhraseForSegment,
}) {
  async function renderSceneVideoWithProgress(segment, sceneIndex, progress, options = {}) {
    const progressBase = Number(options.progressBase ?? 0);
    const progressSpan = Number(options.progressSpan ?? 100);
    const batchLabel = options.batchLabel ? `${options.batchLabel}\n` : "";
    const pct = (value) => Math.min(100, progressBase + (progressSpan * value / 100));
    const slotNumber = sceneSlotNumber(segment);
    state.activeId = segment.id;
    syncInspector();
    updateActiveFromInputs();
    const videoMode = currentVideoMode();
    const modeLabel = videoModeDisplayLabel(videoMode, true);
    const missing = validateSceneReadyForVideo(segment, sceneIndex);
    if (missing.length) throw new Error(missing.join("\n"));
    const autoChainPreFrames = i2vAutoChainEnabled() && videoMode === "i2v"
      ? Math.max(0, Number(segment.auto_chain_pre_frames || 0))
      : 0;
    segment.video_status = "running";
    renderList();
    progress?.set(`${batchLabel}Saving current UI session/SRT timing...`, pct(8));
    let srtPath = await saveSessionForSceneVideo();
    if (!srtPath) throw new Error("The builder SRT path was not created.");
    const isIdLoraMode = videoMode === "id_lora";
    if (videoMode === "i2v" || videoMode === "ingredients" || isIdLoraMode) {
      progress?.set(`${batchLabel}Preparing selected scene image for ${modeLabel}...`, pct(12));
      await ensureSelectedImageForSceneVideo(segment, sceneIndex);
    } else {
      progress?.set(`${batchLabel}Preparing text-to-video render...`, pct(12));
    }
    const videoSettingsForScene = i2vVideoSettingsForSegment(segment);
    // LTX 2.5 uses the existing custom/project-audio T2V graph with dynamic
    // diffusion and CLIP node replacement. Native-audio T2V is not used by
    // the Video Builder, so all LTX versions follow the same audio path.
    const isLtx25NativeAudio = false;
    const isLtx25Rtv = videoMode === "rtv" && videoSettingsForScene.ltx_version === "2.5";
    const idLoraContext = isIdLoraMode ? idLoraSceneContext(segment) : null;
    if (isIdLoraMode && idLoraContext?.dialogue) {
      segment.lyric_text = idLoraContext.dialogue;
    }
    let audioPathForScene = isIdLoraMode
      ? String(idLoraContext?.voicePath || "").trim()
      : options.audioPathOverride || audioInput.value;
    let promptNumberForScene = sceneIndex + 1;
    let audioModeForScene = isIdLoraMode
      ? "ID-LoRA reference voice sample"
      : options.audioPathOverride ? "Combined scene-audio track" : "Global/project audio trimmed for this scene";
    if (isLtx25NativeAudio) {
      audioPathForScene = "";
      audioModeForScene = "LTX 2.5 generated audio from prompt";
    }
    if (options.srtPathOverride) {
      srtPath = options.srtPathOverride;
    }
    if (isIdLoraMode) {
      promptNumberForScene = 1;
    } else if (!options.audioPathOverride && !isLtx25NativeAudio) {
      const fpsForPreroll = Math.max(1, Number(videoSettingsForScene.fps || 24));
      const requestedPreFrames = autoChainPreFrames > 0 ? autoChainPreFrames : Math.max(0, Number(videoSettingsForScene.pre_frames ?? 0));
      const requestedPreSeconds = requestedPreFrames / fpsForPreroll;
      const sourceAudioPath = segment.custom_audio_path || audioInput.value;
      const sceneDuration = Math.max(0.1, timelineSegmentDuration(segment) || 4);
      const sourceStart = segment.custom_audio_path ? audioSourceStart(segment) : Number(segment.start || 0);
      const sourceDuration = audioSourceDurationForScene(segment);
      if (sourceDuration > 0 && sourceStart >= sourceDuration - 0.01) {
        const sourceLabel = segment.custom_audio_path ? "custom scene audio" : "global/project audio";
        throw new Error(
          `${sceneDisplayName(segment, sceneIndex)} starts after the available ${sourceLabel} ends.\n\n` +
          `Scene audio start: ${formatTime(sourceStart)}\n` +
          `Audio length: ${formatTime(sourceDuration)}\n\n` +
          `Shorten or move this scene, load longer audio, or add silence before rendering.`
        );
      }
      const trimStart = Math.max(0, sourceStart - requestedPreSeconds);
      const actualPreSeconds = Math.max(0, sourceStart - trimStart);
      if (!String(sourceAudioPath || "").trim()) {
        throw new Error(`${sceneDisplayName(segment, sceneIndex)}: no audio path is being sent to LTX. Add custom scene audio or load project/global audio before creating the video.`);
      }
      const trimmedAudio = await postJson("/vrgdg/music_builder/trim_scene_audio", {
        project_folder: projectInput.value,
        scene_number: slotNumber,
        source_path: sourceAudioPath,
        start: trimStart,
        duration: sceneDuration + actualPreSeconds,
      }, 120000);
      audioPathForScene = trimmedAudio.audio_path || sourceAudioPath;
      const singleSrt = await postJson("/vrgdg/music_builder/save_single_scene_srt", {
        project_folder: projectInput.value,
        scene_number: slotNumber,
        start_time: actualPreSeconds,
        duration: sceneDuration,
        label: segment.label || `Scene ${sceneIndex + 1}`,
      }, 60000);
      srtPath = singleSrt.srt_path || srtPath;
      promptNumberForScene = 1;
      audioModeForScene = segment.custom_audio_path ? "Custom scene audio trimmed for this scene" : "Global/project audio trimmed for this scene";
    }
    if (isLtx25Rtv && options.audioPathOverride) {
      promptNumberForScene = ltx25RtvSceneOrdinal(segment);
    }
    if (!isLtx25NativeAudio && !String(audioPathForScene || "").trim()) {
      throw new Error(isIdLoraMode
        ? `${sceneDisplayName(segment, sceneIndex)}: ID-LoRA reference voice sample is missing.`
        : `${sceneDisplayName(segment, sceneIndex)}: no audio path is being sent to LTX. Add custom scene audio or load project/global audio before creating the video.`);
    }
    let timingCheck = null;
    if (!isIdLoraMode && !isLtx25NativeAudio) {
      progress?.set(`${batchLabel}Checking SRT timing before hidden ${modeLabel}...`, pct(14));
      const expectedDurationForScene = !options.audioPathOverride
        ? Math.max(0.1, timelineSegmentDuration(segment) || 4)
        : timelineSegmentDuration(segment);
      timingCheck = await validateSrtTimingForSceneVideo({
        segment,
        sceneIndex,
        srtPath,
        promptNumber: promptNumberForScene,
        expectedDuration: expectedDurationForScene,
      });
    } else {
      progress?.set(`${batchLabel}Preparing hidden ${modeLabel} inputs...`, pct(14));
    }
    const isGemmaPrompt = normalizeVideoPromptOrigin(segment.i2v_prompt_origin) === "gemma";
    const videoPromptForRender = isGemmaPrompt
      ? applyMappedTriggerPhrases(
          applyVocalDirectiveToVideoPrompt(
            applyTriggerPhrase(segment.i2v_prompt, videoTriggerPhraseForSegment(segment), { validateJunk: false }),
            segment,
            { suppressPrefix: true, includeFacialPerformance: false }
          ),
          segment,
          { ensureTransitionLast: true }
        )
      : String(segment.i2v_prompt || "").trim();
    const renderPromptForPayload = isIdLoraMode && isGemmaPrompt
      ? buildIdLoraPromptForScene(videoPromptForRender, segment)
      : videoPromptForRender;
    if (videoPromptForRender && videoPromptForRender !== segment.i2v_prompt) {
      segment.i2v_prompt = videoPromptForRender;
      if (segment.id === state.activeId) i2vPrompt.value = videoPromptForRender;
    }
    const payload = {
      ...i2vVideoSettingsPayload(segment),
      i2v_prompt: renderPromptForPayload,
      t2v_prompt: renderPromptForPayload,
      audio_path: audioPathForScene,
      prompt_number_one_based: promptNumberForScene,
      srt_path: srtPath,
      project_folder: projectInput.value,
    };
    if (isIdLoraMode) payload.id_lora_prompt = renderPromptForPayload;
    if (autoChainPreFrames > 0) payload.pre_frames = autoChainPreFrames;
    if (videoMode === "i2v") {
      payload.image_folder = i2vImagesFolder();
      payload.image_index_zero_based = slotNumber - 1;
    }
    if (videoMode === "ingredients") {
      payload.ingredients_image_path = segment.approved_image_path || selectedSegmentImagePath(segment);
      payload.ingredients_image_name = segment.image?.name || "ingredients_reference.png";
    }
    if (videoMode === "flf") {
      const first = firstLastFrameStartImageSource(segment) || {};
      const last = firstLastFrameResolvedEndImageSource(segment) || {};
      payload.first_frame = rtvReferenceImagePayload(first);
      payload.last_frame = rtvReferenceImagePayload(last);
    }
    if (isIdLoraMode) {
      payload.source_image_path = segment.approved_image_path || selectedSegmentImagePath(segment);
      payload.image_path = payload.source_image_path;
      payload.reference_audio_path = audioPathForScene;
      payload.id_reference_audio_path = audioPathForScene;
      payload.reference_audio_seek_seconds = Math.max(0, Number(idLoraContext?.voiceTrimStart || 0));
      payload.reference_audio_duration = Math.max(0, Number(idLoraContext?.voiceTrimDuration || 0));
      payload.identity_guidance_scale = Number(idLoraContext?.identityScale ?? payload.identity_guidance_scale ?? 3);
    }
    const rtvReferences = videoMode === "rtv" ? rtvReferencesForSegment(segment) : null;
    if (videoMode === "rtv") payload.rtv_references = rtvReferences;
    const workflowDetails = {
      audioPath: audioPathForScene,
      promptNumber: promptNumberForScene,
      audioMode: audioModeForScene,
      videoMode,
      idLoraContext,
      rtvReferences,
    };
    const defaultOutputFolder = activeVideoOutputFolder(videoMode);
    const buildEndpoint = isIdLoraMode
      ? "/vrgdg/workflow_runner/build_id_lora_prompt"
      : videoMode === "ingredients"
      ? "/vrgdg/workflow_runner/build_ingredients_prompt"
      : videoMode === "rtv"
      ? "/vrgdg/workflow_runner/build_rtv_prompt"
      : videoMode === "flf"
      ? "/vrgdg/workflow_runner/build_flf_prompt"
      : videoMode === "t2v"
        ? "/vrgdg/workflow_runner/build_t2v_prompt"
        : "/vrgdg/workflow_runner/build_i2v_prompt";
    const readyMessage = timingCheck
      ? `${batchLabel}Preparing hidden ${modeLabel} workflow...\nSRT timing verified: ${timingCheck.srt_duration.toFixed(3)}s`
      : `${batchLabel}Preparing hidden ${modeLabel} workflow...\nReference voice ready.`;
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, defaultOutputFolder, readyMessage, workflowDetails), pct(15));
    const renderStartedAt = Date.now() / 1000 - 2;
    const built = await postJson(buildEndpoint, payload);
    if (videoMode === "flf") {
      workflowDetails.flfInputs = built.flf_inputs || null;
      console.log("[VRGDG FLF] Verified queued inputs", workflowDetails.flfInputs);
      if (!workflowDetails.flfInputs?.inputs_are_different) {
        throw new Error("First Last Frame verification failed: the first and end frame resolved to the same image.");
      }
    }
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, built.output_folder || defaultOutputFolder, `${batchLabel}Queueing hidden ${modeLabel} workflow...`, workflowDetails), pct(40));
    const queued = await queueWorkflowPrompt(built.prompt);
    const promptId = queued?.prompt_id;
    if (!promptId) throw new Error("ComfyUI queued the video but did not return a prompt_id.");
    const findSceneVideoOutput = async () => {
      const found = await postJson("/vrgdg/workflow_runner/find_scene_video_output", {
        project_folder: projectInput.value,
        video_mode: videoMode,
        output_folder: built.output_folder || defaultOutputFolder,
        scene_number: slotNumber,
        prompt_number_one_based: promptNumberForScene,
        min_mtime: renderStartedAt,
        strict_output_folder: isLtx25Rtv,
      }, 30000);
      return found.video_path || "";
    };
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, built.output_folder || defaultOutputFolder, `${batchLabel}Queued prompt ID: ${promptId}\nWaiting for video...`, workflowDetails), pct(60));
    const videos = await waitForVideos(
      promptId,
      (message) => progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, built.output_folder || defaultOutputFolder, `${batchLabel}${message}\nPrompt ID: ${promptId}`, workflowDetails), pct(80)),
      () => state.batchCancelled,
      findSceneVideoOutput,
      {
        timeoutHours: state.sceneRenderWaitHours,
        timeoutMessage: () => sceneVideoTimeoutMessage({
          promptId,
          sceneLabel: sceneDisplayName(segment, sceneIndex),
          modeLabel,
          outputFolder: built.output_folder || defaultOutputFolder,
          projectFolder: projectInput.value,
          finalFolder: collectedSceneVideoFolder(),
          waitHours: state.sceneRenderWaitHours,
        }),
      }
    );
    const video = videos[videos.length - 1] || null;
    const videoPath = resolveComfyVideoPath(video);
    if (!videoPath) throw new Error(`The ${modeLabel} workflow finished, but no video path was found in history.`);
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, built.output_folder || defaultOutputFolder, `${batchLabel}Collecting scene video into builder folder...`, workflowDetails), pct(90));
    const collected = await postJson("/vrgdg/workflow_runner/collect_scene_video", {
      source_path: videoPath,
      project_folder: projectInput.value,
      scene_number: slotNumber,
      existing_action: options.existingVideoAction || "overwrite",
    }, 120000);
    let colorMatched = {
      video_path: collected.video_path || videoPath,
      thumbnail_path: collected.thumbnail_path || "",
      applied: false,
    };
    const globalFLFSettings = state.i2vVideoSettings || videoSettingsForScene;
    if (videoMode === "flf" && globalFLFSettings.flf_match_previous_clip_color === true) {
      const previousSegment = previousAutoChainSourceSegment(segment);
      const previousVideoPath = String(previousSegment ? selectedSegmentVideoPath(previousSegment) || "" : "").trim();
      if (previousVideoPath) {
        progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, collected.video_folder || collectedSceneVideoFolder(), `${batchLabel}Matching the new clip's opening color to the previous clip...`, workflowDetails), pct(93));
        colorMatched = await postJson("/vrgdg/workflow_runner/match_scene_video_start_color", {
          project_folder: projectInput.value,
          video_path: collected.video_path || videoPath,
          reference_video_path: previousVideoPath,
          strength: Math.max(0, Math.min(1, Number(globalFLFSettings.flf_color_match_strength ?? 0.85))),
          fade_seconds: Math.max(0.05, Math.min(30, Number(globalFLFSettings.flf_color_match_fade_seconds ?? 1.0))),
        }, 180000);
      }
    }
    pushHistory();
    const lutResult = await applySceneLutToRenderedVideo(
      segment,
      sceneIndex,
      colorMatched.video_path || collected.video_path || videoPath,
      colorMatched.thumbnail_path || collected.thumbnail_path || "",
      progress,
      pct,
      batchLabel,
    );
    const adjustResult = await applySceneAdjustToRenderedVideo(
      segment,
      sceneIndex,
      lutResult.video_path || collected.video_path || videoPath,
      lutResult.thumbnail_path || collected.thumbnail_path || "",
      progress,
      pct,
      batchLabel,
    );
    const grainResult = await applySceneFilmGrainToRenderedVideo(
      segment,
      sceneIndex,
      adjustResult.video_path || lutResult.video_path || collected.video_path || videoPath,
      adjustResult.thumbnail_path || lutResult.thumbnail_path || collected.thumbnail_path || "",
      progress,
      pct,
      batchLabel,
    );
    if (collected.backup_path) {
      if (!Array.isArray(segment.video_backup_paths)) segment.video_backup_paths = [];
      if (!segment.video_backup_paths.some((item) => mediaPathKey(item) === mediaPathKey(collected.backup_path))) {
        segment.video_backup_paths.push(collected.backup_path);
      }
    }
    if (collected.backup_thumbnail_path) {
      if (!Array.isArray(segment.video_backup_thumbnail_paths)) segment.video_backup_thumbnail_paths = [];
      if (!segment.video_backup_thumbnail_paths.some((item) => mediaPathKey(item) === mediaPathKey(collected.backup_thumbnail_path))) {
        segment.video_backup_thumbnail_paths.push(collected.backup_thumbnail_path);
      }
    }
    segment.video_output = video;
    segment.video_source_path = videoPath;
    activateSegmentVideoPath(segment, grainResult.video_path || adjustResult.video_path || lutResult.video_path || collected.video_path || videoPath, grainResult.thumbnail_path || adjustResult.thumbnail_path || lutResult.thumbnail_path || collected.thumbnail_path || "");
    segment.video_cache_bust = Date.now();
    segment.video_folder = collected.video_folder || collectedSceneVideoFolder();
    segment.preview_mode = "video";
    segment.video_status = "done";
    syncPreview(segment);
    render();
    if (options.autoSaveAfter !== false) {
      await autoSaveSessionQuiet(options.autoSaveReason || "scene video complete");
    }
    const backupNote = collected.backup_path ? `\n\nPrevious video backed up to:\n${collected.backup_path}` : "";
    progress?.setHtml(sceneVideoDetailsHtml(segment, sceneIndex, srtPath, segment.video_folder || collectedSceneVideoFolder(), `${batchLabel}Scene video ready.\n${segment.video_path}${backupNote}`, workflowDetails), pct(100));
    return segment.video_path;
  }

  async function prepareMiniMaxH3ContinuityReference(segment, progress = null, percent = 3, label = "MiniMax continuity") {
    if (!segment || segmentTrack(segment) === "overlay") return null;
    const continuityMode = miniMaxH3ContinuityModeForSegment(segment);
    if (continuityMode === "off") return null;
    if (continuityMode === "exact_start_frame" && segment.minimax_h3_use_scene_image_as_start_frame) {
      throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)} cannot use both its scene image and the previous rendered final frame as the exact start frame.`);
    }
    const previousSegment = previousAutoChainSourceSegment(segment);
    if (!previousSegment) return null;
    if (isMiniMaxH3LatentContinuationMode(continuityMode)) {
      const needsPromptFrame = Boolean(miniMaxH3SettingsForSegment(segment).continuity_prompt_from_last_frame);
      const slotNumber = sceneSlotNumber(segment);
      if (slotNumber <= 1) {
        throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)} is Scene 1 and cannot use Latent Continuation Masked because there is no predecessor scene. Switch Continuity Mode to Off.`);
      }
      const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
      if (!projectFolder) throw new Error("Project folder is missing.");
      progress?.set(`${label}: verifying predecessor Scene ${slotNumber - 1} latent file...`, percent);
      const checkResp = await postJson("/vrgdg/music_builder/check_latent_predecessor", {
        project_folder: projectFolder,
        scene_number: slotNumber,
      }, 10000);
      if (!checkResp?.predecessor_exists) {
        throw new Error(`Latent Continuation Masked requires Scene ${slotNumber - 1} latent file, but none was found. Render Scene ${slotNumber - 1} first.`);
      }
      // The automatic prompt loop needs the predecessor's real last frame as an image.
      let promptFramePath = "";
      if (needsPromptFrame) {
        const previousVideoPath = String(selectedSegmentVideoPath(previousSegment) || "").trim();
        if (!previousVideoPath) {
          throw new Error(`Frame-to-frame prompt creation needs Scene ${slotNumber - 1}'s rendered video to read its last frame, but it has none. Render Scene ${slotNumber - 1} first.`);
        }
        progress?.set(`${label}: extracting Scene ${slotNumber - 1}'s actual final frame...`, percent);
        const extractedFrame = await postJson("/vrgdg/music_builder/extract_video_final_frame", {
          project_folder: projectFolder,
          source_path: previousVideoPath,
          scene_number: slotNumber,
          frame_count: 1,
        }, 120000);
        promptFramePath = String(extractedFrame?.saved_path || "").trim();
        if (!promptFramePath) throw new Error("Could not extract the previous scene's last frame for frame-to-frame continuity.");
      }
      segment.minimax_h3_continuity_mode_used = continuityMode;
      segment.minimax_h3_continuity_source_scene_id = String(previousSegment.id || "");
      return {
        continuityMode,
        transitionEngine: "latent_continuation",
        overlapFrames: 0,
        framePath: "",
        framePaths: [],
        promptFramePath,
        previousSegment,
      };
    }
    const previousVideoPath = String(selectedSegmentVideoPath(previousSegment) || "").trim();
    if (!previousVideoPath) {
      progress?.set(`${label}: the previous scene has no rendered video, so this scene will render without a continuity frame.`, percent);
      segment.minimax_h3_continuity_frame_path = "";
      segment.minimax_h3_continuity_source_video_path = "";
      return null;
    }
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    if (!projectFolder) throw new Error("Project folder is missing.");
    progress?.set(`${label}: extracting the previous scene's actual final frame...`, percent);
    const extracted = await postJson("/vrgdg/music_builder/extract_video_final_frame", {
      project_folder: projectFolder,
      source_path: previousVideoPath,
      scene_number: sceneSlotNumber(segment),
    }, 120000);
    const framePath = String(extracted.saved_path || "").trim();
    if (!framePath) throw new Error("MiniMax continuity final-frame extraction did not return an image path.");
    segment.minimax_h3_continuity_frame_path = framePath;
    segment.minimax_h3_continuity_source_video_path = previousVideoPath;
    segment.minimax_h3_continuity_source_scene_id = String(previousSegment.id || "");
    segment.minimax_h3_continuity_mode_used = continuityMode;
    return { framePath, continuityMode, previousSegment, previousVideoPath };
  }

  // The saved masked mix of a scene whose Audio Mask is on, or null when the mask is off. Refuses a mix that was never
  // built or that no longer matches the scene's length, so a render never uses stale audio.
  async function audioMaskRenderOverride(segment, projectFolder, sceneSeconds, sceneLabel) {
    if (!segment?.audio_mask?.enabled) return null;
    const saved = await postJson("/vrgdg/music_builder/audio_mask/state", { project_folder: projectFolder, scene_id: segment.id }, 60000);
    const path = String(saved?.files?.masked_mix || "").trim();
    if (!saved?.exists || !saved.mix || !path) {
      throw new Error(`${sceneLabel}: Audio Mask is on but its masked mix has not been built. Open Audio Mask and build it, or turn the mask off.`);
    }
    if (Math.abs(Number(saved.mix.duration_seconds) - sceneSeconds) > 0.02) {
      throw new Error(`${sceneLabel}: the scene length changed after its Audio Mask was built. Open Audio Mask, split again and rebuild.`);
    }
    return { path };
  }

  async function createMiniMaxH3FrameContinuityPrompt(segment, sceneIndex, mode, continuityInput, progress, percent = 6, label = "MiniMax continuity") {
    const framePath = String(continuityInput?.promptFramePath || "").trim();
    if (!framePath) throw new Error(`${sceneDisplayName(segment, sceneIndex)} could not find the predecessor's extracted final frame for automatic prompt creation.`);
    const supportingImages = miniMaxH3PromptVisionImages(segment, mode);
    const seen = new Set([mediaPathKey(framePath)]);
    const visionImages = [{ path: framePath, frame_continuity_source: true }];
    for (const item of supportingImages) {
      const path = String(item?.path || "").trim();
      const data = String(item?.data || "").trim();
      const key = path ? mediaPathKey(path) : data;
      if ((!path && !data) || (key && seen.has(key))) continue;
      if (key) seen.add(key);
      visionImages.push(item);
      if (visionImages.length >= 10) break;
    }
    let lastError = null;
    for (let attempt = 1; attempt <= 10; attempt += 1) {
      try {
        progress?.set(`${label}: creating Scene ${sceneSlotNumber(segment)} prompt from Scene ${sceneSlotNumber(segment) - 1}'s actual final frame (attempt ${attempt}/10)...`, percent);
        const data = await runMiniMaxH3PromptGeneration(segment, mode, {
          projectFolder: String(projectInput.value || state.projectFolder || "").trim(),
          builderInstructionKey: "minimax_h3_frame_continuity",
          visionImages,
          frameContinuityPrompt: true,
          promptOnlySceneInspiration: miniMaxH3SceneImageIsPromptInspiration(segment),
          contextOptions: { frameContinuityPrompt: true },
          unloadAfter: true,
          finalizePrompt: (prompt) => ensureBuilderManagedFx(prompt, segment),
          emptyPromptMessage: `Attempt ${attempt}/10 returned an empty frame-to-frame continuity prompt.`,
        });
        const generatedPrompt = String(data?.prompt || "").trim();
        if (!generatedPrompt) throw new Error(`Attempt ${attempt}/10 returned an empty frame-to-frame continuity prompt.`);
        pushHistory();
        segment.minimax_h3_prompt = generatedPrompt;
        segment.minimax_h3_prompt_origin = "previous_final_frame";
        segment.minimax_h3_continuity_prompt_source_scene_id = String(continuityInput?.previousSegment?.id || "");
        segment.minimax_h3_continuity_prompt_frame_path = framePath;
        segment.minimax_h3_continuity_prompt_created_at = new Date().toISOString();
        if (segment?.id === activeSegment()?.id) miniMaxPrompt.value = generatedPrompt;
        updateMiniMaxPromptCharacterStatus(segment);
        render();
        await autoSaveSessionQuiet(`Scene ${sceneSlotNumber(segment)} frame-to-frame continuity prompt`);
        return generatedPrompt;
      } catch (error) {
        lastError = error;
        if (attempt >= 10) break;
        const delayMs = Math.min(15000, attempt * 2000);
        progress?.set(`${label}: prompt attempt ${attempt}/10 failed; retrying in ${Math.round(delayMs / 1000)} seconds...\n${String(error?.message || error)}`, percent);
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }
    throw new Error(`${sceneDisplayName(segment, sceneIndex)} could not create a valid frame-to-frame continuity prompt after 10 attempts. Last error: ${String(lastError?.message || lastError || "unknown error")}`);
  }

  async function renderMiniMaxSceneVideoWithProgress(segment, sceneIndex, progress, options = {}) {
    // The panel can still contain a newer value than the project object when a
    // render is started immediately after editing a field. Flush the active
    // scene one last time before taking the settings snapshot used to build
    // the hidden workflow.
    if (segment?.id === activeSegment()?.id) saveMiniMaxH3SettingsFromPanel();
    const panelSegmentId = activeSegment()?.id;
    const progressBase = Number(options.progressBase ?? 0);
    const progressSpan = Number(options.progressSpan ?? 100);
    const batchLabel = options.batchLabel ? `${options.batchLabel}\n` : "";
    const pct = (value) => Math.min(100, progressBase + (progressSpan * value / 100));
    const slotNumber = sceneSlotNumber(segment);
    const projectFolder = String(projectInput.value || "").trim();
    const timelineStart = Number(segment?.start);
    const timelineEnd = Number(segment?.end);
    const sceneDuration = timelineEnd - timelineStart;
    const miniMaxSettings = miniMaxH3SettingsForSegment(segment);
    const builtInAudio = miniMaxSettings.audio_mode === "built_in_audio";
    // The pipeline belongs to the whole project, and the RefMod pipeline has one mode.
    const refmodPipeline = normalizeMiniMaxH3Pipeline(state.miniMaxH3Settings?.pipeline) === "refmod";
    const mode = refmodPipeline ? "reference_to_video" : normalizeMiniMaxH3Mode(options.mode ?? miniMaxSettings.video_mode);
    const twoPass = miniMaxSettings.render_pass === "two_pass"
      && ["reference_to_video", "image_reference_to_video", "image_to_video"].includes(mode);
    const threePass = miniMaxSettings.render_pass === "three_pass"
      && ["reference_to_video", "image_reference_to_video"].includes(mode);
    if (refmodPipeline && threePass) {
      throw new Error("The RefMod pipeline supports Single and 2 Pass only. Choose one of those passes before rendering.");
    }
    if ((twoPass || threePass) && builtInAudio) {
      throw new Error(`MiniMax H3 ${threePass ? "2 Pass Advanced" : "2 Pass"} currently supports Input Audio only. Switch Audio Mode to Input Audio before rendering.`);
    }
    if (!projectFolder) throw new Error("Save or select a project folder before rendering MiniMax H3.");
    if (!Number.isFinite(timelineStart) || !Number.isFinite(timelineEnd) || sceneDuration <= 0) {
      throw new Error(`${sceneDisplayName(segment, sceneIndex)} has invalid timeline boundaries.`);
    }

    // miniMaxH3ContinuityModeForSegment is off for every mode that cannot continue, so this is a no-op there.
    const continuityInput = options.continuityInput === undefined
      ? await prepareMiniMaxH3ContinuityReference(
        segment,
        progress,
        pct(3),
        `${batchLabel}MiniMax continuity`,
      )
      : options.continuityInput;

    const generatedContinuityPrompt = miniMaxH3FrameContinuityPromptEnabled(segment)
      ? await createMiniMaxH3FrameContinuityPrompt(segment, sceneIndex, mode, continuityInput, progress, pct(6), `${batchLabel}MiniMax continuity`)
      : "";
    const promptBase = String(generatedContinuityPrompt || (options.prompt ?? (segment?.minimax_h3_prompt || segment?.i2v_prompt || ""))).trim();
    // A prompt written before the RefMod pipeline was switched on (or before its RefMods changed) may not name them yet.
    // Putting each RefMod's label next to its character is safe to repeat, so it is done again here.
    const prompt = refmodPipeline
      ? attachRefmodLabels(promptBase, miniMaxOrderedImageReferenceItemsForSegment(segment, mode).map((item) => item.refmod).filter(Boolean))
      : promptBase;
    if (!prompt) throw new Error(`${sceneDisplayName(segment, sceneIndex)} needs a MiniMax H3 prompt.`);
    assertValidMiniMaxH3FinalPrompt(prompt, segment, mode, {
      allowCueValidationWarnings: true,
      onCueValidationWarning: options.onCueValidationWarning,
    });

    const sourceAudioPath = String(
      options.audioPath
      ?? (segment?.custom_audio_path || currentProjectAudioPath() || audioInput.value || "")
    ).trim();
    if (!builtInAudio && !sourceAudioPath) throw new Error(`${sceneDisplayName(segment, sceneIndex)} needs source audio for MiniMax H3.`);
    const sourceStartSeconds = Number(
      options.sourceStartSeconds
      ?? (segment?.custom_audio_path ? audioSourceStart(segment) : timelineStart)
    );
    const sourceDurationSeconds = Number(
      options.sourceDurationSeconds
      ?? audioSourceDurationForScene(segment)
      ?? 0
    );

    // An enabled Audio Mask renders with its masked mix (kept vocals plus the music). The finished clip gets the real audio back.
    const maskedAudio = builtInAudio ? null : await audioMaskRenderOverride(segment, projectFolder, sceneDuration, sceneDisplayName(segment, sceneIndex));

    if (["reference_to_video", "video_to_video"].includes(mode) && miniMaxReferenceKeysForSegment(segment).length) {
      progress?.set(`${batchLabel}Preparing MiniMax H3 Reference Builder images...`, pct(4));
      await persistIngredientsSheetImages(projectFolder);
    }

    // The RefMod pipeline renders from saved RefMods, in scene order with their labels, and sends no reference images.
    const refmodItems = refmodPipeline
      ? miniMaxOrderedImageReferenceItemsForSegment(segment, mode).map((item) => item.refmod).filter(Boolean)
      : [];
    if (refmodPipeline && !refmodItems.length) {
      throw new Error(`${sceneDisplayName(segment, sceneIndex)} needs at least one RefMod. Pick a RefMod on a Reference Builder card and map it to this scene.`);
    }
    let imagePaths = refmodPipeline ? [] : miniMaxRenderReferenceImagePaths(segment, mode, options.imagePaths);
    let i2vLastFramePath = "";
    if (mode === "image_to_video") {
      const last = miniMaxI2VLastFrame(segment);
      if (!last.path && last.data) {
        const archived = await postJson("/vrgdg/music_builder/archive_scene_image", {
          project_folder: projectFolder, scene_number: slotNumber, image_data: last.data,
        });
        segment.first_last_frame_end_image_path = archived.saved_path;
        segment.first_last_frame_end_image_data = "";
        await autoSaveSessionQuiet("MiniMax last frame saved");
      }
      i2vLastFramePath = miniMaxI2VFramePaths(segment, imagePaths[0]).last;
      imagePaths = [imagePaths[0], ...(i2vLastFramePath ? [i2vLastFramePath] : [])];
    }
    let continuityImageNumber = 0;
    if (continuityInput?.framePath) {
      const continuityKey = mediaPathKey(continuityInput.framePath);
      const alreadyIncluded = imagePaths.some((path) => mediaPathKey(path) === continuityKey);
      if (!alreadyIncluded) {
        if (imagePaths.length >= 9) {
          throw new Error(
            `${sceneDisplayName(segment, sceneIndex)} has nine MiniMax image references already. `
            + "Remove one Reference Builder image so the previous-scene continuity frame can use the reserved ninth slot."
          );
        }
        imagePaths.push(continuityInput.framePath);
      }
      continuityImageNumber = imagePaths.findIndex((path) => mediaPathKey(path) === continuityKey) + 1;
      segment.minimax_h3_continuity_image_number = continuityImageNumber;
    } else {
      segment.minimax_h3_continuity_image_number = 0;
    }
    const rawVideoReferences = options.videoReferences
      ?? segment?.minimax_h3_video_references
      ?? segment?.minimax_video_references
      ?? [];
    const videoReferences = mode === "video_to_video" && Array.isArray(rawVideoReferences) ? rawVideoReferences : [];
    if (["image_to_video", "image_reference_to_video"].includes(mode) && !imagePaths.length) {
      throw new Error(`${sceneDisplayName(segment, sceneIndex)} needs a selected scene image for MiniMax Image to Video.`);
    }
    if (mode === "reference_to_video" && !refmodPipeline && !imagePaths.length) {
      throw new Error(`${sceneDisplayName(segment, sceneIndex)} needs at least one ordered Reference Builder image.`);
    }
    if (mode === "video_to_video" && !videoReferences.some((item) => String(item?.path || "").trim())) {
      throw new Error(`${sceneDisplayName(segment, sceneIndex)} needs at least one reference video path.`);
    }
    const referenceMismatch = miniMaxPromptReferenceMismatch(segment, prompt, mode, imagePaths);
    if (referenceMismatch) {
      throw new Error(`${sceneDisplayName(segment, sceneIndex)}: ${referenceMismatch} Regenerate this scene's MiniMax prompt after updating its reference mapping.`);
    }
    const orderedReferenceItems = miniMaxOrderedImageReferenceItemsForSegment(segment, mode);
    const progressImages = imagePaths.map((path, index) => {
      const matched = orderedReferenceItems.find((item) => mediaPathKey(item?.image?.path) === mediaPathKey(path));
      const isContinuity = continuityImageNumber === index + 1;
      let label = String(matched?.label || "").trim();
      if (isContinuity) {
        label = continuityInput?.continuityMode === "exact_start_frame"
          ? "Previous scene final frame — exact start"
          : "Previous scene final frame — spatial continuity";
      } else if (!label && mode === "image_to_video" && index === 0) {
        label = "Scene image — exact start frame";
      }
      return {
        path,
        data: String(matched?.image?.data || "").trim(),
        label: label || String(path || "").split(/[\\/]/).pop() || "Reference",
      };
    });
    const progressDialogue = miniMaxDialogueOrderText(segment);
    const progressLyric = progressDialogue
      || String(segment?.lyric_text || "").trim()
      || (segmentUsesNoLipSyncPerformance(segment) ? "[No visible vocal / no lip sync]" : "");
    // The RefMod pipeline shows the RefMods in this scene (preview, label, name and strength) in place of images.
    const progressRefmods = refmodItems.map((item) => ({
      url: refmodPreviewUrl(item.mod_name),
      label: item.name,
      caption: `${item.label} ${item.name}${item.strength < 1 ? ` · ${Math.round(item.strength * 100)}%` : ""}`,
    }));
    progress?.setSceneDetails?.({
      sceneLabel: sceneDisplayName(segment, sceneIndex),
      modeLabel: `${miniMaxH3ModeLabel(mode)}${refmodPipeline ? " (RefMods)" : ""} · ${builtInAudio ? "Built-in MiniMax Audio" : "Input Audio"}`,
      images: refmodPipeline ? progressRefmods : progressImages,
      imagesTitle: refmodPipeline ? "RefMods" : undefined,
      lyric: progressLyric,
      prompt,
    });
    // Masked continuation plans its own warm-up and tail, so the Render settings frames do not apply.
    const maskedContinuation = continuityInput?.continuityMode === "latent_continuation_masked";
    const warmupFrames = maskedContinuation ? 0 : Math.max(0, Math.trunc(Number(
      options.warmupFrames
      ?? miniMaxSettings.warmup_frames
    ) || 0));
    const cooldownFrames = maskedContinuation ? 0 : Math.max(0, Math.trunc(Number(
      options.cooldownFrames
      ?? miniMaxSettings.cooldown_frames
    ) || 0));

    state.activeId = segment.id;
    segment.video_status = "running";
    renderList();
    const activePanel = segment?.id === panelSegmentId;
    const requestedTwoPassSteps = (twoPass || threePass) && activePanel
      ? {
        pass1: Math.max(1, Math.min(1000, Math.trunc(Number((threePass ? advancedTwoPassControls : twoPassControls)[0].steps.value) || 20))),
        pass2: Math.max(1, Math.min(1000, Math.trunc(Number((threePass ? advancedTwoPassControls : twoPassControls)[1].steps.value) || 5))),
      }
      : null;
    progress?.set(`${batchLabel}Preparing exact MiniMax H3 scene timing and ${builtInAudio ? "native audio generation" : "input audio"}...`, pct(8));

    const latentContextFrames = miniMaxSettings.latent_context_frames;
    // One output resolution for every pass type: single pass renders it, 2 Pass finishes at it, and
    // 2 Pass Advanced uses it as the Pass 2 size.
    const outputMegapixels = Number(options.megapixels ?? miniMaxSettings.megapixels);
    const outputAspectRatio = String(options.aspectRatio ?? miniMaxSettings.aspect_ratio);
    const outputFrame = miniMaxH3FrameSize(outputMegapixels, outputAspectRatio);
    try {
      const payload = {
        project_folder: projectFolder,
        scene_number: slotNumber,
        audio_mode: miniMaxSettings.audio_mode,
        video_mode: mode,
        continuity_mode: continuityInput?.continuityMode || miniMaxSettings.continuity_mode || "off",
        latent_context_frames: latentContextFrames,
        minimax_h3_latent_context_frames: latentContextFrames,
        audio_path: builtInAudio ? "" : (maskedAudio?.path || sourceAudioPath),
        prompt,
        pass2_prompt: String(segment?.minimax_h3_pass2_prompt || ""),
        timeline_start_seconds: timelineStart,
        timeline_end_seconds: timelineEnd,
        source_start_seconds: builtInAudio
          ? timelineStart
          : maskedAudio
            ? 0
            : (Number.isFinite(sourceStartSeconds) ? Math.max(0, sourceStartSeconds) : timelineStart),
        pre_frames: warmupFrames,
        tail_loss_frames: cooldownFrames,
        seed: Number(options.seed ?? miniMaxSettings.seed),
        aspect_ratio: outputAspectRatio,
        megapixels: outputMegapixels,
        diffusion_model_name: miniMaxSettings.diffusion_model_name,
        clip_name: miniMaxSettings.clip_name,
        video_vae_name: miniMaxSettings.video_vae_name,
        audio_vae_name: miniMaxSettings.audio_vae_name,
        sampler_name: miniMaxSettings.sampler_name,
        scheduler: miniMaxSettings.scheduler,
        steps: miniMaxSettings.steps,
        denoise: miniMaxSettings.denoise,
        ref_image_size: miniMaxSettings.ref_image_size,
        two_pass_lora_name: miniMaxSettings.two_pass_lora_name,
        two_pass_lora_strength: miniMaxSettings.two_pass_lora_strength,
        final_width: twoPass ? outputFrame.width : undefined,
        final_height: twoPass ? outputFrame.height : undefined,
        latent_upscale_scale: twoPass ? miniMaxSettings.two_pass_latent_upscale_scale : undefined,
        latent_upscaler_name: (twoPass || threePass) ? miniMaxSettings.two_pass_latent_upscaler_name : undefined,
        use_te_speed: !twoPass && !threePass ? miniMaxSettings.use_te_speed : undefined,
        ...Object.fromEntries(["te_speed", "feedforward", "block_sparse_attention"].flatMap((key) =>
          [1, 2].map((pass) => [`pass${pass}_use_${key}`, miniMaxSettings[`pass${pass}_use_${key}`] ?? miniMaxSettings[`two_pass_use_${key}`]])
        )),
        use_fast_vae_decode: miniMaxSettings.use_fast_vae_decode,
        use_feedforward: !twoPass && !threePass ? miniMaxSettings.use_feedforward : undefined,
        two_pass_use_feedforward: (twoPass || threePass) ? miniMaxSettings.two_pass_use_feedforward : undefined,
        use_block_sparse_attention: !twoPass && !threePass ? miniMaxSettings.use_block_sparse_attention : undefined,
        two_pass_use_block_sparse_attention: (twoPass || threePass) ? miniMaxSettings.two_pass_use_block_sparse_attention : undefined,
        two_pass_use_fast_vae_decode: (twoPass || threePass) ? miniMaxSettings.two_pass_use_fast_vae_decode : undefined,
        te_speed_processing_control: miniMaxSettings.two_pass_te_speed_processing_control,
        te_speed_start_percent: miniMaxSettings.two_pass_te_speed_start_percent,
        te_speed_end_percent: miniMaxSettings.two_pass_te_speed_end_percent,
        te_speed_mcs: miniMaxSettings.two_pass_te_speed_mcs,
        te_speed_cache_depth: miniMaxSettings.two_pass_te_speed_cache_depth,
        te_speed_device: miniMaxSettings.two_pass_te_speed_device,
        final_resize_method: (twoPass || threePass) ? miniMaxSettings.two_pass_final_resize_method : undefined,
        output_crf: (twoPass || threePass) ? miniMaxSettings.two_pass_output_crf : undefined,
        three_pass_lightx_lora_name: miniMaxSettings.three_pass_lightx_lora_name,
        three_pass_lightx_lora_strength: miniMaxSettings.three_pass_lightx_lora_strength,
        pass1_steps: twoPass ? (requestedTwoPassSteps?.pass1 ?? miniMaxSettings.two_pass_pass1_steps) : threePass ? (requestedTwoPassSteps?.pass1 ?? miniMaxSettings.advanced_two_pass_pass1_steps) : miniMaxSettings.steps,
        pass1_denoise: twoPass ? miniMaxSettings.two_pass_pass1_denoise : threePass ? miniMaxSettings.advanced_two_pass_pass1_denoise : miniMaxSettings.denoise,
        pass1_sampler_name: twoPass ? miniMaxSettings.two_pass_pass1_sampler : threePass ? miniMaxSettings.advanced_two_pass_pass1_sampler : miniMaxSettings.sampler_name,
        pass1_scheduler: twoPass ? miniMaxSettings.two_pass_pass1_scheduler : threePass ? miniMaxSettings.advanced_two_pass_pass1_scheduler : miniMaxSettings.scheduler,
        pass1_seed: twoPass ? miniMaxSettings.two_pass_pass1_seed : threePass ? miniMaxSettings.advanced_two_pass_pass1_seed : miniMaxSettings.seed,
        pass2_steps: twoPass ? (requestedTwoPassSteps?.pass2 ?? miniMaxSettings.two_pass_pass2_steps) : threePass ? (requestedTwoPassSteps?.pass2 ?? miniMaxSettings.advanced_two_pass_pass2_steps) : 4,
        pass2_denoise: twoPass ? miniMaxSettings.two_pass_pass2_denoise : threePass ? miniMaxSettings.advanced_two_pass_pass2_denoise : 0.2,
        pass2_sampler_name: twoPass ? miniMaxSettings.two_pass_pass2_sampler : threePass ? miniMaxSettings.advanced_two_pass_pass2_sampler : miniMaxSettings.sampler_name,
        pass2_scheduler: twoPass ? miniMaxSettings.two_pass_pass2_scheduler : threePass ? miniMaxSettings.advanced_two_pass_pass2_scheduler : miniMaxSettings.scheduler,
        pass2_seed: twoPass ? miniMaxSettings.two_pass_pass2_seed : threePass ? miniMaxSettings.advanced_two_pass_pass2_seed : miniMaxSettings.seed,
        three_pass_pass1_megapixels: miniMaxSettings.three_pass_pass1_megapixels,
        three_pass_pass1_steps: miniMaxSettings.three_pass_pass1_steps,
        three_pass_pass1_denoise: miniMaxSettings.three_pass_pass1_denoise,
        three_pass_pass1_sampler: miniMaxSettings.three_pass_pass1_sampler,
        three_pass_pass1_scheduler: miniMaxSettings.three_pass_pass1_scheduler,
        three_pass_pass1_seed: miniMaxSettings.three_pass_pass1_seed,
        three_pass_pass1_te_speed: miniMaxSettings.three_pass_pass1_te_speed,
        three_pass_pass2_megapixels: miniMaxSettings.three_pass_pass2_megapixels,
        three_pass_pass2_steps: miniMaxSettings.three_pass_pass2_steps,
        three_pass_pass2_denoise: miniMaxSettings.three_pass_pass2_denoise,
        three_pass_pass2_sampler: miniMaxSettings.three_pass_pass2_sampler,
        three_pass_pass2_scheduler: miniMaxSettings.three_pass_pass2_scheduler,
        three_pass_pass2_seed: miniMaxSettings.three_pass_pass2_seed,
        three_pass_pass2_te_speed: miniMaxSettings.three_pass_pass2_te_speed,
        three_pass_pass3_megapixels: miniMaxSettings.three_pass_pass3_megapixels,
        three_pass_pass3_steps: miniMaxSettings.three_pass_pass3_steps,
        three_pass_pass3_denoise: miniMaxSettings.three_pass_pass3_denoise,
        three_pass_pass3_sampler: miniMaxSettings.three_pass_pass3_sampler,
        three_pass_pass3_scheduler: miniMaxSettings.three_pass_pass3_scheduler,
        three_pass_pass3_seed: miniMaxSettings.three_pass_pass3_seed,
        three_pass_pass3_te_speed: miniMaxSettings.three_pass_pass3_te_speed,
        advanced_pass1_megapixels: miniMaxSettings.advanced_two_pass_pass1_megapixels,
        advanced_pass2_megapixels: outputMegapixels,
        advanced_vram_preset: miniMaxSettings.advanced_two_pass_vram_preset,
        easy_cache_bypass: miniMaxSettings.easy_cache_bypass,
        easy_cache_reuse_threshold: miniMaxSettings.easy_cache_reuse_threshold,
        easy_cache_start_percent: miniMaxSettings.easy_cache_start_percent,
        easy_cache_end_percent: miniMaxSettings.easy_cache_end_percent,
        easy_cache_verbose: miniMaxSettings.easy_cache_verbose,
        sage_attention: miniMaxSettings.sage_attention,
        use_memory_efficient_sage_attention: miniMaxSettings.use_memory_efficient_sage_attention,
        enable_fp16_accumulation: miniMaxSettings.enable_fp16_accumulation,
        use_loras: miniMaxSettings.use_loras,
        lora_count: miniMaxSettings.lora_count,
        loras: miniMaxSettings.loras,
        use_turbo_lora: miniMaxSettings.use_turbo_lora,
        turbo_lora_name: miniMaxSettings.turbo_lora_name,
        turbo_lora_strength: miniMaxSettings.turbo_lora_strength,
        image_paths: imagePaths,
        video_references: videoReferences,
        pipeline: refmodPipeline ? "refmod" : "standard",
        ...(refmodPipeline ? { refmod_references: referencePayload(refmodItems) } : {}),
      };
      if (mode === "image_to_video") {
        if (i2vLastFramePath) payload.last_frame_path = i2vLastFramePath;
      }
      if (maskedAudio) {
        payload.source_duration_seconds = sceneDuration;
      } else if (!builtInAudio && Number.isFinite(sourceDurationSeconds) && sourceDurationSeconds > 0) {
        payload.source_duration_seconds = sourceDurationSeconds;
      }

      const renderStartedAt = Date.now() / 1000 - 2;
      const built = await postJson(
        threePass
          ? "/vrgdg/workflow_runner/build_minimax_h3_advanced_2pass_prompt"
          : twoPass
            ? "/vrgdg/workflow_runner/build_minimax_h3_2pass_prompt"
          : "/vrgdg/workflow_runner/build_minimax_h3_prompt",
        payload,
        180000,
      );
      const missingRefmodLabels = built?.refmod?.labels_missing_from_prompt || [];
      if (missingRefmodLabels.length) {
        toast(`This scene's prompt does not use ${missingRefmodLabels.join(", ")}. Regenerate the scene prompt so every RefMod is named.`, true);
      }
      const timing = built?.timing || {};
      const postTrim = built?.post_render_trim || {};
      const builtLoraSettings = built?.lora_settings || {};
      const builtTurboSettings = built?.turbo_settings || {};
      const builtAdvancedSettings = built?.advanced_settings || {};
      const builtAdvancedResolution = built?.advanced_two_pass || {};
      const debugWorkflowPath = String(built?.debug_workflow_path || "").trim();
      const exactTwoPassSteps = (twoPass || threePass)
        ? {
          pass1: Number(built?.prompt?.["124"]?.inputs?.steps),
          pass2: Number(built?.prompt?.["190"]?.inputs?.value),
        }
        : null;
      if ((twoPass || threePass) && (!Number.isInteger(exactTwoPassSteps.pass1) || !Number.isInteger(exactTwoPassSteps.pass2))) {
        throw new Error("The built MiniMax H3 two-pass prompt is missing its exact sampler step values.");
      }
      const exactTwoPassStepsLine = (twoPass || threePass)
        ? `\nRequested panel steps: Pass 1 = ${requestedTwoPassSteps?.pass1 ?? (threePass ? miniMaxSettings.advanced_two_pass_pass1_steps : miniMaxSettings.two_pass_pass1_steps)}; Pass 2 = ${requestedTwoPassSteps?.pass2 ?? (threePass ? miniMaxSettings.advanced_two_pass_pass2_steps : miniMaxSettings.two_pass_pass2_steps)}`
          + `\nExact built-prompt sampler steps: Pass 1 = ${exactTwoPassSteps.pass1}; Pass 2 = ${exactTwoPassSteps.pass2}`
        : "";
      const exactAdvancedResolutionLine = threePass
        ? `\nBuilt Pass 1 target: ${builtAdvancedResolution.pass1_width || "?"}×${builtAdvancedResolution.pass1_height || "?"} (${builtAdvancedResolution.pass1_megapixels ?? "?"} MP); Pass 2 target: ${builtAdvancedResolution.pass2_width || "?"}×${builtAdvancedResolution.pass2_height || "?"} (${builtAdvancedResolution.pass2_megapixels ?? "?"} MP)`
          + (debugWorkflowPath ? `\nAPI workflow snapshot: ${debugWorkflowPath}` : "")
        : "";
      const exactModelChainLoras = (modelRef) => {
        const loras = [];
        const visited = new Set();
        let currentRef = modelRef;
        while (Array.isArray(currentRef) && currentRef.length >= 2) {
          const nodeId = String(currentRef[0]);
          if (visited.has(nodeId)) break;
          visited.add(nodeId);
          const promptNode = built?.prompt?.[nodeId];
          if (!promptNode?.inputs) break;
          const loraName = String(promptNode.inputs.lora_name || "").trim();
          if (loraName && /lora/i.test(String(promptNode.class_type || ""))) {
            loras.push({
              nodeId,
              name: loraName,
              strength: Number(promptNode.inputs.strength_model),
            });
          }
          currentRef = promptNode.inputs.model;
        }
        return loras.reverse();
      };
      const exactTwoPassLoras = (twoPass || threePass)
        ? {
          pass1: exactModelChainLoras(built?.prompt?.["124"]?.inputs?.model),
          pass2: exactModelChainLoras(built?.prompt?.["192"]?.inputs?.model),
        }
        : null;
      const formatExactLoras = (loras) => loras.length
        ? loras.map((item) => `${item.name} @ ${Number.isFinite(item.strength) ? item.strength : "unknown strength"} [node ${item.nodeId}]`).join(" → ")
        : "none";
      const exactTwoPassLorasLine = (twoPass || threePass)
        ? `\nExact built-prompt LoRAs:\nPass 1: ${formatExactLoras(exactTwoPassLoras.pass1)}\nPass 2: ${formatExactLoras(exactTwoPassLoras.pass2)}`
        : "";
      const loraLine = (twoPass || threePass)
        ? exactTwoPassLorasLine
        : builtLoraSettings.enabled
          ? `\nLoRAs: ${Number(builtLoraSettings.count || 0)} — ${(builtLoraSettings.loras || []).map((item) => `${item.name} @ ${item.strength}`).join(", ")}`
          : "\nLoRAs: OFF";
      const turboLine = mode === "reference_to_video" && !twoPass && !threePass
        ? `\nSteps: ${Number(builtAdvancedSettings.steps || miniMaxSettings.steps)}`
        : (twoPass || threePass)
        ? ""
        : builtTurboSettings.enabled
          ? `\nTurbo: ON — effective steps ${Number(builtAdvancedSettings.effective_steps || builtTurboSettings.steps || miniMaxSettings.steps)}; LoRA ${builtTurboSettings.lora_name || miniMaxSettings.turbo_lora_name} @ ${builtTurboSettings.strength ?? miniMaxSettings.turbo_lora_strength}`
          : `\nTurbo: OFF — steps ${Number(builtAdvancedSettings.steps || miniMaxSettings.steps)}`;
      const settingsScopeLine = `\nSettings scope: ${segment?.use_scene_minimax_h3_settings ? "locked scene settings" : "project/global settings"}`;
      const finalDuration = Number(postTrim.duration);
      if (!Number.isFinite(finalDuration) || Math.abs(finalDuration - sceneDuration) > 0.001) {
        throw new Error(
          `MiniMax H3 timing verification failed for ${sceneDisplayName(segment, sceneIndex)}. `
          + `Timeline: ${sceneDuration.toFixed(6)}s; adapter: ${Number.isFinite(finalDuration) ? finalDuration.toFixed(6) : "missing"}s.`
        );
      }

      progress?.set(
        `${batchLabel}${threePass ? "Queueing MiniMax H3 2 Pass Advanced (Base → MMH3 tiled upscale)..." : twoPass ? "Queueing MiniMax H3 2 Pass (Stage 1 → Stage 2)..." : "Queueing MiniMax H3..."}\n`
        + `Timeline: ${sceneDuration.toFixed(3)}s\n`
        + `H3 render: ${Number(timing.h3_frame_count || 0)} frames`
        + (threePass ? "\nThe MMH3 tiled/chunked Pass 2 video will be used for stitching." : twoPass ? "\nPass 1 is learned-latent upscaled and refined by pass 2; the final pass-2 video will be used for stitching." : "")
        + exactTwoPassStepsLine
        + loraLine
        + turboLine
        + settingsScopeLine
        + (continuityImageNumber ? `\nContinuity: ${continuityInput.continuityMode === "exact_start_frame" ? "exact start" : "spatial reference"} (Image ${continuityImageNumber})` : ""),
        pct(30),
      );
      const queued = await queueWorkflowPrompt(built.prompt);
      const promptId = queued?.prompt_id;
      if (!promptId) throw new Error("ComfyUI queued MiniMax H3 but did not return a prompt_id.");
      const videos = await waitForVideos(
        promptId,
        (message) => {
          progress?.set(`${batchLabel}${threePass ? "MiniMax H3 2 Pass Advanced (Base → MMH3 tiled upscale)\n" : twoPass ? "MiniMax H3 2 Pass (Stage 1 → Stage 2)\n" : ""}${message}${exactTwoPassStepsLine}${exactAdvancedResolutionLine}${exactTwoPassLorasLine}\nPrompt ID: ${promptId}`, pct(62));
        },
        () => state.batchCancelled,
        null,
        {
          timeoutHours: state.sceneRenderWaitHours,
          timeoutMessage: () => sceneVideoTimeoutMessage({
            promptId,
            sceneLabel: sceneDisplayName(segment, sceneIndex),
            modeLabel: threePass ? "MiniMax H3 2 Pass Advanced (Base → MMH3 tiled upscale)" : twoPass ? "MiniMax H3 2 Pass (Stage 1 → Stage 2)" : "MiniMax H3",
            outputFolder: built.output_folder || "",
            projectFolder,
            finalFolder: collectedSceneVideoFolder(),
            waitHours: state.sceneRenderWaitHours,
          }),
        },
      );
      const videoName = (item) => String(item?.params?.filename || item?.filename || "").toLowerCase();
      const video = threePass
        ? videos.find((item) => videoName(item).includes("stage2")) || videos[videos.length - 1] || null
        : twoPass
          ? videos.find((item) => videoName(item).includes("stage2")) || videos[videos.length - 1] || null
        : videos[videos.length - 1] || null;
      const alignedVideoPath = resolveComfyVideoPath(video);
      if (!alignedVideoPath) {
        throw new Error("MiniMax H3 finished, but no aligned video path was found in history.");
      }

      progress?.set(`${batchLabel}Trimming MiniMax H3 to the exact timeline boundaries...`, pct(82));
      const trimmed = await postJson("/vrgdg/workflow_runner/trim_scene_video", {
        source_path: alignedVideoPath,
        project_folder: projectFolder,
        scene_number: slotNumber,
        start: Number(postTrim.start || 0),
        duration: finalDuration,
        frames: Number(postTrim.frames || 0),
        label: "minimax_exact",
        mark_as_audio_video: true,
        ...(maskedAudio
          ? { restore_audio_path: sourceAudioPath, restore_audio_start_seconds: Number.isFinite(sourceStartSeconds) ? Math.max(0, sourceStartSeconds) : timelineStart }
          : {}),
      }, 240000);
      const exactVideoPath = String(trimmed.video_path || "").trim();
      if (!exactVideoPath) throw new Error("MiniMax H3 exact trimming did not return a video path.");

      progress?.set(`${batchLabel}Collecting exact MiniMax H3 scene video...`, pct(92));
      const collected = await postJson("/vrgdg/workflow_runner/collect_scene_video", {
        source_path: exactVideoPath,
        project_folder: projectFolder,
        scene_number: slotNumber,
        existing_action: options.existingVideoAction || "overwrite",
      }, 120000);

      pushHistory();
      if (collected.backup_path) {
        if (!Array.isArray(segment.video_backup_paths)) segment.video_backup_paths = [];
        if (!segment.video_backup_paths.some((item) => mediaPathKey(item) === mediaPathKey(collected.backup_path))) {
          segment.video_backup_paths.push(collected.backup_path);
        }
      }
      if (collected.backup_thumbnail_path) {
        if (!Array.isArray(segment.video_backup_thumbnail_paths)) segment.video_backup_thumbnail_paths = [];
        if (!segment.video_backup_thumbnail_paths.some((item) => mediaPathKey(item) === mediaPathKey(collected.backup_thumbnail_path))) {
          segment.video_backup_thumbnail_paths.push(collected.backup_thumbnail_path);
        }
      }
      const finalSceneVideoPath = collected.video_path || exactVideoPath;
      segment.video_output = null;
      segment.video_source_path = finalSceneVideoPath;
      segment.minimax_h3_stage1_path = "";
      segment.minimax_h3_stage1_source_path = "";
      segment.minimax_h3_stage2_path = (twoPass || threePass) ? finalSceneVideoPath : "";
      segment.minimax_h3_stage2_source_path = segment.minimax_h3_stage2_path;
      segment.minimax_h3_timing = timing;
      activateSegmentVideoPath(
        segment,
        collected.video_path || exactVideoPath,
        collected.thumbnail_path || trimmed.thumbnail_path || "",
      );
      // The collected final clip must become the active timeline selection.
      // History normalization intentionally preserves a user's previous backup
      // selection, so select the newly collected final path explicitly here.
      const finalTimelineVideoPath = String(collected.video_path || exactVideoPath || "").trim();
      const finalTimelineVideoIndex = segment.video_history.findIndex(
        (item) => mediaPathKey(item) === mediaPathKey(finalTimelineVideoPath),
      );
      if (finalTimelineVideoIndex >= 0) {
        segment.video_history_index = finalTimelineVideoIndex;
        segment.video_path = finalTimelineVideoPath;
        segment.video_thumbnail_path = segment.video_thumbnail_history[finalTimelineVideoIndex]
          || collected.thumbnail_path
          || trimmed.thumbnail_path
          || segment.video_thumbnail_path
          || "";
      }
      segment.video_cache_bust = Date.now();
      segment.video_folder = collected.video_folder || collectedSceneVideoFolder();
      segment.preview_mode = "video";
      segment.video_status = "done";
      syncPreview(segment);
      render();
      loadDirtyLatentBadges();
      if (options.autoSaveAfter !== false) {
        await autoSaveSessionQuiet(options.autoSaveReason || "MiniMax H3 scene video complete");
      }
      await postJson("/vrgdg/workflow_runner/cleanup_minimax_h3_output", {
        output_folder: built.output_folder || "", project_folder: projectFolder, scene_number: slotNumber,
      }, 120000).catch(error => console.warn("MiniMax scratch cleanup failed; files retained:", error));
      progress?.set(
        `${batchLabel}${threePass ? "MiniMax H3 2 Pass Advanced complete — MMH3 Pass 2 selected." : twoPass ? "MiniMax H3 learned-latent 2 Pass complete — final pass-2 video selected." : "MiniMax H3 scene ready."}${exactAdvancedResolutionLine}\n${segment.video_path}\n`
        + `Exact duration: ${finalDuration.toFixed(3)}s`,
        pct(100),
      );
      return segment.video_path;
    } catch (error) {
      segment.video_status = "error";
      renderList();
      throw error;
    }
  }

  async function stitchRenderedScenes(progress, options = {}) {
    const baseSegments = Array.isArray(options.segments) && options.segments.length ? options.segments : state.segments;
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    const overlaySegments = state.overlayTrack.enabled
      ? (Array.isArray(options.overlaySegments) ? options.overlaySegments : state.overlaySegments)
          .filter((segment) => overlayClipIsEnabled(segment, state.overlayTrack))
      : [];
    const timelineOffset = Number(options.timelineOffset || 0);
    if (!miniMaxProject) {
      await ensureSceneLutsAppliedBeforeStitch(baseSegments, progress, options);
      await ensureSceneAdjustsAppliedBeforeStitch(baseSegments, progress, options);
      await ensureSceneFilmGrainAppliedBeforeStitch(baseSegments, progress, options);
      if (overlaySegments.length) {
        await ensureSceneLutsAppliedBeforeStitch(overlaySegments, progress, options);
        await ensureSceneAdjustsAppliedBeforeStitch(overlaySegments, progress, options);
        await ensureSceneFilmGrainAppliedBeforeStitch(overlaySegments, progress, options);
      }
    }
    const paths = baseSegments.map((segment) => String(selectedSegmentVideoPath(segment) || "").trim());
    // Every scene clip was cut to round(end * 24) - round(start * 24) frames of the whole project. A preview of a few
    // scenes shifts the times, so the shift must be a whole number of frames or the rounding differs and a clip is
    // padded with a repeated frame or loses one at the join (a stutter).
    const frameAlignedOffset = Math.round(timelineOffset * 24) / 24;
    const sceneTimingItems = miniMaxProject ? baseSegments.map((segment) => ({
      start: Math.max(0, Number(segment.start || 0) - frameAlignedOffset),
      end: Math.max(0, Number(segment.end || 0) - frameAlignedOffset),
    })) : [];
    const overlayItems = overlaySegments
      .filter((segment) => String(selectedSegmentVideoPath(segment) || "").trim())
      .map((segment, index) => ({
        path: String(selectedSegmentVideoPath(segment) || "").trim(),
        start: Math.max(0, Number(segment.start || 0) - timelineOffset),
        end: Math.max(0.05, Number(segment.end || 0) - timelineOffset),
        source_start: Math.max(0, Number(segment.overlay_source_start || 0)),
        label: segment.label || `Insert ${index + 1}`,
      }));
    const globalAudioPath = currentProjectAudioPath();
    const miniMaxBuiltInAudioMode = miniMaxProject && baseSegments.length > 0
      && baseSegments.every((segment) => miniMaxH3SettingsForSegment(segment).audio_mode === "built_in_audio");
    const embeddedSceneAudioMode = miniMaxBuiltInAudioMode || (!miniMaxProject && currentVideoMode() === "id_lora") || !!options.useEmbeddedSceneAudio;
    const sceneAudioMode = !embeddedSceneAudioMode && !globalAudioPath && usingSceneAudioMode();
    const audioPaths = sceneAudioMode ? baseSegments.map((segment) => String(segment.custom_audio_path || "").trim()) : [];
    const audioItems = sceneAudioMode ? baseSegments.map((segment) => ({
      path: String(segment.custom_audio_path || "").trim(),
      start: audioSourceStart(segment),
      duration: audioChunkDuration(segment),
    })) : [];
    const missing = [];
    paths.forEach((path, index) => {
      if (!path) missing.push(`${sceneDisplayName(baseSegments[index], index)}: rendered scene video is missing.`);
      else if (!isLikelyVideoPath(path)) missing.push(`${sceneDisplayName(baseSegments[index], index)}: selected scene media is not a video:\n${path}`);
      if (sceneAudioMode && !audioPaths[index]) {
        missing.push(`${sceneDisplayName(baseSegments[index], index)}: scene audio is missing and no global audio is available as a fallback.`);
      }
    });
    overlayItems.forEach((item) => {
      if (!isLikelyVideoPath(item.path)) missing.push(`${item.label || "Insert"}: selected insert media is not a video:\n${item.path}`);
    });
    if (missing.length) throw new Error(missing.join("\n"));
    const stitchSettings = state.i2vVideoSettings || {};
    const stitchVideoMode = currentVideoMode();
    const stitchWidth = miniMaxProject ? 0 : stitchVideoMode === "ingredients"
      ? Number(stitchSettings.ingredients_width || DEFAULT_LTX_INGREDIENTS_WIDTH)
      : Number(stitchSettings.width || 1920);
    const stitchHeight = miniMaxProject ? 0 : stitchVideoMode === "ingredients"
      ? Number(stitchSettings.ingredients_height || DEFAULT_LTX_INGREDIENTS_HEIGHT)
      : Number(stitchSettings.height || 1080);
    const stitchAudioMessage = embeddedSceneAudioMode
      ? "Stitching rendered scene videos with their embedded scene audio..."
      : sceneAudioMode
        ? "Stitching rendered scene videos with scene audio clips..."
        : "Stitching rendered scene videos with original audio...";
    progress?.set(stitchAudioMessage, 94);
    const data = await postJson("/vrgdg/workflow_runner/stitch_scene_videos", {
      scene_paths: paths,
      audio_path: embeddedSceneAudioMode ? "" : globalAudioPath,
      scene_audio_paths: audioPaths,
      scene_audio_items: audioItems,
      scene_timing_items: sceneTimingItems,
      timeline_fps: miniMaxProject ? 24 : 0,
      use_embedded_scene_audio: embeddedSceneAudioMode,
      overlay_items: overlayItems,
      project_folder: projectInput.value,
      width: stitchWidth,
      height: stitchHeight,
      audio_start: Number(options.audioStart || 0),
      audio_duration: Number(options.audioDuration || 0),
      output_prefix: options.outputPrefix || "FINAL_VIDEO",
    }, 20 * 60 * 1000);
    state.finalVideoPath = data.final_video_path || "";
    return data;
  }

  async function runGemmaThenCreateSceneVideo() {
    const segment = requireActiveSegment();
    if (!segment) return;
    const mode = currentVideoMode();
    const runnerName = promptRunnerActionName();
    if (mode === "import") {
      toast(`Imported-video mode does not use ${runnerName} prompt generation. Use Create Scene Video instead.`, true);
      return;
    }
    const modeLabel = videoModeDisplayLabel(mode, true);
    if (!window.confirm(`Run ${runnerName} to replace this scene's ${modeLabel} prompt, then immediately create the video without stopping for prompt review?`)) return;
    updateActiveFromInputs();
    let progress = null;
    try {
      setButtonGroupState(gemmaThenCreateVideoButtons, { disabled: true });
      progress = createProgressWindow(`${runnerName} → Create Scene Video`);
      progress.set(`Creating the ${modeLabel} prompt with Gemma...`, 8);
      await generateI2VPromptForSegment(
        segment,
        progress,
        18,
        `Gemma ${modeLabel}`,
        { unloadAfter: true, forceVision: mode === "flf" },
      );
      await autoSaveSessionQuiet(`Gemma ${modeLabel} prompt before scene video`);
      progress.set("Gemma prompt ready. Starting scene video automatically...", 100);
      progress.close(350);
      progress = null;
      await createSceneVideo();
    } catch (error) {
      progress?.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      setButtonGroupState(gemmaThenCreateVideoButtons, { disabled: false });
    }
  }

  async function createSceneVideo() {
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    const info = segmentIndexInfo(segment);
    const sceneIndex = info.index;
    if (sceneIndex < 0) return;
    const idLoraMode = currentVideoMode() === "id_lora";
    if (!idLoraMode && !(await ensureAudioOrOfferSilentTimeline({ segment }))) return;
    const missing = [
      ...validateSceneReadyForVideo(segment, sceneIndex),
      ...(!idLoraMode && (!String(segment.custom_audio_path || audioInput.value || "").trim() || !String(projectInput.value || "").trim()) ? ["Load global audio or add custom audio for this scene, and set the project folder first."] : []),
      ...(idLoraMode && !String(projectInput.value || "").trim() ? ["Project folder is missing."] : []),
    ];
    if (missing.length) {
      toast(missing.join("\n"), true);
      return;
    }
    let existingVideoAction = "overwrite";
    if (String(segment.video_path || "").trim()) {
      existingVideoAction = await askExistingSceneVideoAction(segment, sceneIndex);
      if (existingVideoAction === "cancel") return;
    }
    let renderTarget = segment;
    let renderIndex = sceneIndex;
    if (existingVideoAction === "overlay" && segmentTrack(segment) !== "overlay") {
      renderTarget = cloneSceneAsOverlay(
        segment,
        () => `seg_${Date.now()}_${Math.floor(Math.random() * 10000)}`,
        nextOverlaySlotNumber(),
      );
      pushHistory();
      state.overlaySegments.push(renderTarget);
      sortSegments(state.overlaySegments);
      setActiveSegment(renderTarget);
      renderIndex = segmentIndexInfo(renderTarget).index;
      existingVideoAction = "overwrite";
    }
    let progress = null;
    const etaLog = startSingleSceneETA(renderTarget);
    let etaStatus = "failed";
    try {
      state.batchCancelled = false;
      setButtonGroupState(createSceneVideoButtons, { disabled: true, text: "Creating..." });
      setButtonGroupState(gemmaThenCreateVideoButtons, { disabled: true });
      progress = createProgressWindow("Creating scene video");
      if (currentVideoMode() === "flf" && flfChainingEnabled(renderTarget) && flfRenderChainStartSource(renderTarget) === "rendered_frame") {
        const previousSegment = previousAutoChainSourceSegment(renderTarget);
        if (previousSegment && String(selectedSegmentVideoPath(previousSegment) || "").trim()) {
          await prepareFLFRenderedFrameNextScene(previousSegment, renderTarget, progress, 4, `FLF rendered-frame chain into ${sceneDisplayName(renderTarget, renderIndex)}`);
        }
      }
      const videoPath = await renderSceneVideoWithProgress(renderTarget, renderIndex, progress, {
        existingVideoAction,
      });
      await runClearMemoryWorkflowQuiet(progress, `${sceneDisplayName(renderTarget, renderIndex)} render`, 98);
      etaStatus = "complete";
      progress.close(900);
      toast(`Scene video ready:\n${videoPath}`);
    } catch (error) {
      const stopped = /stopped by user/i.test(String(error?.message || error));
      etaStatus = stopped ? "canceled" : "failed";
      renderTarget.video_status = stopped ? "none" : "error";
      progress?.set(stopped ? "Scene video creation stopped by user." : `Error:\n${String(error?.message || error)}`, 100);
      toast(stopped ? "Scene video creation stopped." : String(error?.message || error), !stopped);
      renderList();
    } finally {
      await finishSingleSceneETA(etaLog, etaStatus);
      setButtonGroupState(createSceneVideoButtons, { disabled: false, text: "Create Scene Video" });
      setButtonGroupState(gemmaThenCreateVideoButtons, { disabled: false });
    }
  }

  async function createMiniMaxSceneVideo() {
    if (normalizeProjectVideoEngine(state.projectVideoEngine) !== "minimax_h3") {
      toast("Set this project's Video Engine to MiniMax H3 in Builder Settings first.", true);
      return;
    }
    const segment = requireActiveSegment();
    if (!segment) return;
    updateActiveFromInputs();
    saveMiniMaxH3SettingsFromPanel();
    saveMiniMaxSceneInputsFromPanel();
    const info = segmentIndexInfo(segment);
    const sceneIndex = info.index;
    if (sceneIndex < 0) return;

    const missing = validateMiniMaxSceneReadyForVideo(segment, sceneIndex);
    if (!String(projectInput.value || state.projectFolder || "").trim()) missing.push("Project folder is missing.");
    if (miniMaxH3SettingsForSegment(segment).audio_mode !== "built_in_audio"
      && !String(segment.custom_audio_path || currentProjectAudioPath() || audioInput.value || "").trim()) {
      missing.push("Load global audio or add custom audio for this scene.");
    }
    if (missing.length) {
      toast(missing.join("\n"), true);
      return;
    }

    let existingVideoAction = "overwrite";
    if (String(segment.video_path || "").trim()) {
      existingVideoAction = await askExistingSceneVideoAction(segment, sceneIndex);
      if (existingVideoAction === "cancel") return;
    }
    let renderTarget = segment;
    let renderIndex = sceneIndex;
    if (existingVideoAction === "overlay" && segmentTrack(segment) !== "overlay") {
      renderTarget = cloneSceneAsOverlay(
        segment,
        () => `seg_${Date.now()}_${Math.floor(Math.random() * 10000)}`,
        nextOverlaySlotNumber(),
      );
      pushHistory();
      state.overlaySegments.push(renderTarget);
      sortSegments(state.overlaySegments);
      setActiveSegment(renderTarget);
      renderIndex = segmentIndexInfo(renderTarget).index;
      existingVideoAction = "overwrite";
    }

    let progress = null;
    const etaLog = startSingleSceneETA(renderTarget);
    let etaStatus = "failed";
    try {
      state.batchCancelled = false;
      setButtonGroupState(miniMaxSceneVideoButtons, { disabled: true, text: "Creating MiniMax H3..." });
      progress = createProgressWindow("Creating MiniMax H3 scene video");
      const videoPath = await renderMiniMaxSceneVideoWithProgress(renderTarget, renderIndex, progress, {
        existingVideoAction,
      });
      await runClearMemoryWorkflowQuiet(progress, `${sceneDisplayName(renderTarget, renderIndex)} render`, 98);
      etaStatus = "complete";
      progress.close(900);
      toast(`MiniMax H3 scene video ready:\n${videoPath}`);
    } catch (error) {
      const stopped = /stopped by user/i.test(String(error?.message || error));
      etaStatus = stopped ? "canceled" : "failed";
      renderTarget.video_status = stopped ? "none" : "error";
      progress?.set(stopped ? "MiniMax H3 scene video creation stopped by user." : `Error:\n${String(error?.message || error)}`, 100);
      toast(stopped ? "MiniMax H3 scene video creation stopped." : String(error?.message || error), !stopped);
      renderList();
    } finally {
      await finishSingleSceneETA(etaLog, etaStatus);
      setButtonGroupState(miniMaxSceneVideoButtons, { disabled: false, text: "Create MiniMax H3 Scene Video" });
    }
  }

  function overlaySegmentsForPreviewRange(startTime, endTime) {
    return state.overlaySegments
      .filter((segment) => String(selectedSegmentVideoPath(segment) || "").trim())
      .filter((segment) => Number(segment.end || 0) > startTime && Number(segment.start || 0) < endTime)
      .map((segment) => ({
        ...segment,
        overlay_source_start: Math.max(0, startTime - Number(segment.start || 0)),
        start: Math.max(startTime, Number(segment.start || 0)),
        end: Math.min(endTime, Number(segment.end || 0)),
      }));
  }

  async function stitchPreviewFromSegments(segments, label = "selected") {
    const baseSegments = (Array.isArray(segments) ? segments : []).filter((segment) => segmentTrack(segment) !== "overlay");
    if (!baseSegments.length) {
      toast("Choose at least one base scene for the preview stitch.", true);
      return;
    }
    const sorted = baseSegments.slice().sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    const baseIndexes = sorted.map((segment) => state.segments.findIndex((item) => item.id === segment.id));
    const nonContiguous = baseIndexes.some((index) => index < 0) || baseIndexes.some((index, itemIndex) => itemIndex > 0 && index !== baseIndexes[itemIndex - 1] + 1);
    if (nonContiguous) {
      toast("Preview stitching needs contiguous base scenes. Select one continuous scene range, or use Start scene / End scene.", true);
      return;
    }
    const startTime = Math.min(...sorted.map((segment) => Number(segment.start || 0)));
    const endTime = Math.max(...sorted.map((segment) => Number(segment.end || 0)));
    const overlays = overlaySegmentsForPreviewRange(startTime, endTime);
    const progress = createProgressWindow("Stitch Preview");
    try {
      progress.set(`Stitching preview from ${label}...\nScenes: ${sorted.length}`, 15);
      const stitched = await stitchRenderedScenes(progress, {
        segments: sorted,
        overlaySegments: overlays,
        timelineOffset: startTime,
        audioStart: startTime,
        audioDuration: Math.max(0.1, endTime - startTime),
        outputPrefix: `PREVIEW_SCENES_${label.replace(/[^a-z0-9_-]+/gi, "_")}`,
      });
      progress.set(`Preview stitch complete.\n\nPreview video:\n${stitched.final_video_path}`, 100);
      progress.close(6500);
      toast(`Preview stitch complete:\n${stitched.final_video_path}`);
      showFinalVideoReadyModal(stitched.final_video_path);
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    }
  }

  async function renderImageSlideshowPreview() {
    updateActiveFromInputs();
    enforceAudioTimelineEnd();
    const scenes = state.segments.slice().sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    if (!scenes.length) {
      toast("Create at least one base scene before rendering an image slideshow preview.", true);
      return;
    }
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    if (!projectFolder) {
      toast("Create or load a project before rendering an image slideshow preview.", true);
      return;
    }
    const globalAudioPath = currentProjectAudioPath();
    if (!globalAudioPath) {
      toast("Load global audio before rendering an image slideshow preview.", true);
      return;
    }
    const progress = createProgressWindow("Image Slideshow Preview");
    try {
      progress.set(`Checking ${scenes.length} scene image${scenes.length === 1 ? "" : "s"}...`, 8);
      const missing = [];
      const paths = [];
      let archivedImageData = false;
      for (let index = 0; index < scenes.length; index += 1) {
        const segment = scenes[index];
        let source = segmentImageSource(segment);
        if (source?.data && !source.path) {
          const saved = await postJson("/vrgdg/music_builder/archive_scene_image", {
            image_data: source.data,
            project_folder: projectFolder,
            scene_number: sceneSlotNumber(segment),
          }, 120000);
          if (saved.saved_path) {
            segment.custom_image_path = saved.saved_path;
            source = { path: saved.saved_path };
            archivedImageData = true;
          }
        }
        if (!String(source?.path || "").trim()) {
          missing.push(sceneDisplayName(segment, state.segments.indexOf(segment)));
        }
        paths.push(String(source?.path || "").trim());
      }
      if (missing.length) {
        throw new Error(`These scenes need an image before a slideshow can be rendered:\n${missing.join("\n")}`);
      }
      if (archivedImageData) await autoSaveSessionQuiet("slideshow source images archived");

      const imageItems = scenes.map((segment, index) => {
        const start = Number(segment.start || 0);
        const nextStart = Number(scenes[index + 1]?.start);
        const end = Number.isFinite(nextStart) && nextStart > start
          ? nextStart
          : Math.max(start + 0.05, Number(segment.end || start + 0.05));
        return { path: paths[index], duration: Math.max(0.05, end - start) };
      });
      const firstStart = Math.max(0, Number(scenes[0].start || 0));
      const settings = state.i2vVideoSettings || {};
      const mode = currentVideoMode();
      const width = mode === "ingredients"
        ? Number(settings.ingredients_width || DEFAULT_LTX_INGREDIENTS_WIDTH)
        : Number(settings.width || 1920);
      const height = mode === "ingredients"
        ? Number(settings.ingredients_height || DEFAULT_LTX_INGREDIENTS_HEIGHT)
        : Number(settings.height || 1080);
      progress.set(`Rendering image-only preview with global audio...\nScenes: ${scenes.length}`, 20);
      const result = await postJson("/vrgdg/workflow_runner/render_image_slideshow", {
        image_items: imageItems,
        audio_path: globalAudioPath,
        audio_start: firstStart,
        project_folder: projectFolder,
        width,
        height,
        fps: Number(settings.fps || 24),
        output_prefix: "IMAGE_SLIDESHOW_PREVIEW",
      }, 20 * 60 * 1000);
      state.finalVideoPath = result.final_video_path || "";
      progress.set(`Image slideshow preview complete.\n\nPreview video:\n${state.finalVideoPath}`, 100);
      progress.close(6500);
      toast(`Image slideshow preview complete:\n${state.finalVideoPath}`);
      showFinalVideoReadyModal(state.finalVideoPath);
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    }
  }

  function openStitchPreviewModal() {
    const selectedBase = selectedSegmentsForBatch({ baseOnly: true })
      .slice()
      .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = "Stitch Preview";
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const close = makeButton("Close");
    header.append(heading, close);
    const note = document.createElement("div");
    note.textContent = "Creates a preview video from already-rendered scene videos. Inserts are included automatically. This does not run Gemma, create images, or render new videos.";
    note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    const selectedCount = selectedBase.length;
    const selectedButton = makeButton(`Use Selected Scenes${selectedCount ? ` (${selectedCount})` : ""}`, "primary");
    selectedButton.disabled = !selectedCount;
    const startInput = makeInput("1");
    const endInput = makeInput(String(Math.max(1, state.segments.length)));
    startInput.type = "number";
    endInput.type = "number";
    startInput.min = "1";
    endInput.min = "1";
    startInput.max = String(Math.max(1, state.segments.length));
    endInput.max = String(Math.max(1, state.segments.length));
    const rangeGrid = document.createElement("div");
    rangeGrid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    rangeGrid.append(makeField("Start scene", startInput), makeField("End scene", endInput));
    const rangeButton = makeButton("Use Scene Range", "primary");
    const cancel = makeButton("Cancel");
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    actions.append(cancel, rangeButton);
    box.append(header, note, selectedButton, rangeGrid, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    close.onclick = () => backdrop.remove();
    cancel.onclick = () => backdrop.remove();
    selectedButton.onclick = () => {
      backdrop.remove();
      stitchPreviewFromSegments(selectedBase, "selected");
    };
    rangeButton.onclick = () => {
      const start = Math.max(1, Math.min(state.segments.length, Number(startInput.value || 1)));
      const end = Math.max(start, Math.min(state.segments.length, Number(endInput.value || start)));
      const scenes = state.segments.slice(start - 1, end);
      backdrop.remove();
      stitchPreviewFromSegments(scenes, `${String(start).padStart(3, "0")}-${String(end).padStart(3, "0")}`);
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
  }

  function askExistingSceneVideoAction(segment, sceneIndex) {
    return new Promise((resolve) => {
      const canAddToOverlayTrack = Boolean(state.overlayTrack.enabled) && segmentTrack(segment) !== "overlay";
      const backdrop = document.createElement("div");
      backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
      const box = document.createElement("div");
      box.style.cssText = "width:min(560px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
      const heading = document.createElement("div");
      heading.textContent = "This scene already has a video";
      heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
      const body = document.createElement("div");
      body.style.cssText = "display:flex;flex-direction:column;gap:8px;font-size:13px;color:#d4d4d8;line-height:1.45;";
      const scene = document.createElement("div");
      scene.textContent = `${sceneDisplayName(segment, sceneIndex)} already has a rendered clip.`;
      const path = document.createElement("div");
      path.textContent = String(segment?.video_path || "");
      path.style.cssText = "border:1px solid #303038;border-radius:6px;background:#18181b;padding:8px;color:#a5f3fc;overflow-wrap:anywhere;font-size:12px;";
      const note = document.createElement("div");
      note.textContent = canAddToOverlayTrack
        ? "Choose Add to overlay track to keep the base clip and create an alternate take above it, or replace the current clip."
        : "Choose Backup and replace to keep the old clip, or Overwrite to replace it without saving a backup.";
      note.style.cssText = "color:#fde68a;";
      body.append(scene, path, note);
      const actions = document.createElement("div");
      actions.style.cssText = `display:grid;grid-template-columns:repeat(${canAddToOverlayTrack ? 4 : 3},1fr);gap:8px;`;
      const cancel = makeButton("Cancel");
      const overwrite = makeButton("Overwrite");
      const backup = makeButton("Backup and replace", "primary");
      const overlay = makeButton("Add to overlay track", "primary");
      cancel.onclick = () => {
        backdrop.remove();
        resolve("cancel");
      };
      overwrite.onclick = () => {
        backdrop.remove();
        resolve("overwrite");
      };
      backup.onclick = () => {
        backdrop.remove();
        resolve("backup");
      };
      overlay.onclick = () => {
        backdrop.remove();
        resolve("overlay");
      };
      actions.append(cancel, overwrite, backup);
      if (canAddToOverlayTrack) actions.append(overlay);
      box.append(heading, body, actions);
      backdrop.append(box);
      document.body.append(backdrop);
    });
  }

  function miniMaxH3FrameContinuityPromptEnabled(segment) {
    if (!segment || segmentTrack(segment) === "overlay" || sceneSlotNumber(segment) <= 1) return false;
    const settings = miniMaxH3SettingsForSegment(segment);
    return Boolean(settings.continuity_prompt_from_last_frame)
      && ["reference_to_video", "video_to_video"].includes(miniMaxH3ModeForSegment(segment))
      && isMiniMaxH3LatentContinuationMode(settings.continuity_mode);
  }

  return {
    createMiniMaxSceneVideo, createSceneVideo, miniMaxH3FrameContinuityPromptEnabled, openStitchPreviewModal, stitchPreviewFromSegments,
    renderImageSlideshowPreview, renderMiniMaxSceneVideoWithProgress, renderSceneVideoWithProgress,
    runGemmaThenCreateSceneVideo, stitchRenderedScenes,
  };
}
