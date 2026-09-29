import { storyboardGptPayload } from "../storyboard_builder/gpt_payload.mjs";
import { cancelComfyExecutionAndWaitIdle, postJson } from "./comfy_api.mjs";
import { escapeHtml, normalizeProjectVideoEngine, toast } from "./controls.mjs";
import { showFinalVideoReadyModal } from "./dialogs.mjs";
import { sceneImagePromptForEnhanceAll } from "./image_generation.mjs";
import { setButtonGroupState } from "./inspector.mjs";
import { cloneFlowGptBrowserSettings, defaultI2VVideoSettings, normalizeBuilderStoryLayer } from "./model_settings.mjs";
import { isRecoverableBuildGemmaError } from "./prompt_text.mjs";
import { renderLogDuration, updateRenderLogSummary } from "./render_log.mjs";
import { renderMissingListHtml } from "./scene_render_prep.mjs";
import { selectedSegmentVideoPath } from "./selection_preview.mjs";
import { batchEmptyMessage, batchScopeLabel, normalizeBatchScope } from "./timeline_state.mjs";



export function createBatchRender({
  activeSegment, allEditableSegments, assertBatchNotStopped, autoSaveSessionQuiet, batchTargetItems,
  canImg2ImgContinuityFromPreviousRenderedScene, createEndFrameForSegment, createErnieImageForSegment,
  createFlowGptImageForSegment, createFluxKleinImageForSegment, createFluxPromptButton,
  createImageForSegmentInCurrentMode, createKrea2TwoPassImageForSegment, createNBImageForSegment,
  createNBPromptButton, createProgressWindow, createSceneVideoButtons, createT2IButton,
  createZImageForSegment, currentEnhanceSource, currentVideoMode, endFrameSegmentsForMode,
  enhanceImageForSegment, ensureAudioOrOfferSilentTimeline, ernieCreateButtons, ernieCreateT2IButton,
  firstLastFrameEndImageSource, firstLastFramePromptReferences, firstLastFrameStartImageSource,
  flfChainPreviousEndFrame, flfChainedSettingsPanel, flfChainingEnabled, flfPreGeneratePromptsEnabled,
  flfRenderChainStartSource, flfSameLocationCameraDiversityDirection, flfStructureModeSelect,
  flowGptCreateImageButton, flowGptCreatePromptButton, fluxCreateButtons, fullBuildButton, fullFLFBuildButton,
  gemmaThenCreateVideoButtons, generateFinalIndependentFLFPromptForSegment, generateFluxKleinPromptForSegment,
  generateI2VPromptForSegment, generateIndependentFLFMotionPlanForSegment, generateNBPromptForSegment,
  generateT2IPromptForSegment, hasFirstLastFrameEndImage, i2vAllScenes, i2vAutoChainEnabled,
  i2vTextGemmaModelSelect, imageAllSegmentsForMode, imageModeDisplayLabel, imageModeImg2ImgContinuityLabel,
  imageModeSupportsImg2ImgContinuity, img2imgContinuityEnabled, krea2TwoPassCreateButtons,
  krea2TwoPassCreateT2IButton, miniMaxBatchReferenceProblems, miniMaxH3ModeForSegment,
  miniMaxSceneVideoButtons, nbCreateButtons, persistRenderLog, prepareAutoChainedNextScene,
  prepareAutoImg2ImgContinuityForScene, prepareFLFRenderedFrameNextScene, prepareSceneAudioMix,
  previousAutoChainSourceSegment, previousSceneStartImageIngredient, projectInput, pushHistory,
  recoverFromBuildGemmaError, recoverSceneVideosFromProject, render, renderAllButton, renderETAScene,
  renderMiniMaxSceneVideoWithProgress, renderSceneVideoWithProgress, runClearMemoryWorkflowQuiet,
  runGemmaImagePromptPassWithRetry, runImageMemoryCleanupQuiet, saveFlowGptBrowserSettingsFromPanel,
  saveI2VVideoSettingsFromPanel, saveMiniMaxH3SettingsFromPanel, saveSessionForSceneVideo,
  saveZEnhanceSettingsFromPanel, savedImagePromptForMode, sceneDisplayName, sceneSlotNumber,
  segmentImageSource, segmentIndexInfo, segmentTrack, setImageSeedForCurrentMode, setMiniMaxH3SeedRandom,
  setVideoSeedRandom, startBuilderETA, state, stitchRenderedScenes, storyboardScenePayload,
  syncI2VVideoSettingsPanel, syncInspector, syncRTVSceneImageAnchorPanel, syncSegmentFlowGptPrompt,
  syncSegmentT2IPrompt, t2iTextGemmaModelSelect, textGemmaRunnerPayload, updateActiveFromInputs,
  upsertRenderLog, validateCreateEndFramesReady, validateRenderAllReady, validateZImageAllReady,
  videoModeDisplayLabel, wizardStoryboardState, zCreateButtons, zEnhanceAllButton, zEnhanceAllToolButton,
  zEnhanceButton, zImageAllButton,
}) {
  async function renderAllScenes(options = {}) {
    updateActiveFromInputs();
    saveI2VVideoSettingsFromPanel();
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    if (miniMaxProject) saveMiniMaxH3SettingsFromPanel();
    const forceVideos = Boolean(options.forceVideos);
    const randomizeVideoSeed = Boolean(options.randomizeVideoSeed);
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const skipFinalStitch = Boolean(options.skipFinalStitch || sceneScope === "selected");
    let renderLog = null;
    let currentSceneLog = null;
    const renderWarnings = [];
    let previousRenderEndedMs = 0;
    if (!forceVideos) {
      try {
        await recoverSceneVideosFromProject({
          autoSave: true,
          autoSaveReason: "recovered existing scene videos before Render Missing",
          toast: true,
        });
      } catch (error) {
        console.warn("[VRGDG Music Builder] Scene video recovery before Render All failed:", error);
      }
    }
    if (!(await ensureAudioOrOfferSilentTimeline({ sceneScope }))) return;
    const missing = validateRenderAllReady({ forceVideos, sceneScope });
    const progress = createProgressWindow(sceneScope === "selected" ? "Render Selected Scenes" : sceneScope === "from_selected" ? "Render From Selected Scene" : "Render All Scenes");
    if (missing.length) {
      progress.setHtml(renderMissingListHtml(missing), 100);
      toast("Render All needs a few things fixed first.", true);
      return;
    }
    const logStartedMs = Date.now();
    const logStarted = new Date(logStartedMs);
    const logStamp = logStarted.toISOString().replace(/\D/g, "").slice(0, 14);
    const modeLabel = sceneScope === "selected"
      ? "Render Selected Scenes"
      : sceneScope === "from_selected"
        ? "Render From Selected Scene"
        : "Render All";
    renderLog = {
      schema_version: 1,
      id: `render_${logStamp}_${Math.random().toString(16).slice(2, 8)}`,
      status: "running",
      mode_label: modeLabel,
      scene_scope: sceneScope,
      project_folder: String(state.projectFolder || projectInput.value || ""),
      video_engine: miniMaxProject ? "minimax_h3" : "ltx",
      video_mode: miniMaxProject ? state.miniMaxH3Settings.video_mode : currentVideoMode(),
      force_videos: forceVideos,
      randomize_video_seed: randomizeVideoSeed,
      skip_final_stitch: skipFinalStitch,
      started_at: logStarted.toISOString(),
      ended_at: "",
      setup_started_at: logStarted.toISOString(),
      setup_ms: 0,
      stitch_ms: 0,
      target_scene_count: 0,
      skipped_existing_count: 0,
      scenes: [],
      final_video_path: "",
      error: "",
    };
    startBuilderETA(renderLog);
    upsertRenderLog(renderLog);
    try {
      state.batchCancelled = false;
      renderAllButton.disabled = true;
      renderAllButton.textContent = "Rendering...";
      setButtonGroupState(createSceneVideoButtons, { disabled: true });
      setButtonGroupState(miniMaxSceneVideoButtons, { disabled: true });
      setButtonGroupState(gemmaThenCreateVideoButtons, { disabled: true });
      await persistRenderLog(renderLog);
      progress.set(`Autosaving session/SRT before ${sceneScope === "selected" ? "Render Selected" : sceneScope === "from_selected" ? "Render From Selected" : "Render All"}...`, 3);
      await saveSessionForSceneVideo();
      if (miniMaxProject) {
        const scenesAfterSave = batchTargetItems(sceneScope)
          .filter(({ segment }) => forceVideos || !String(selectedSegmentVideoPath(segment) || "").trim());
        const referenceProblems = miniMaxBatchReferenceProblems(scenesAfterSave);
        if (referenceProblems.length) {
          throw new Error(`MiniMax reference preflight found ${referenceProblems.length} stale scene prompt${referenceProblems.length === 1 ? "" : "s"} before rendering:\n${referenceProblems.map((problem) => `- ${problem}`).join("\n")}`);
        }
      }
      const ltx25RtvBatch = !miniMaxProject
        && currentVideoMode() === "rtv"
        && (state.i2vVideoSettings?.ltx_version || "2.5") === "2.5";
      const preparedAudio = miniMaxProject || skipFinalStitch
        ? { audioPath: "", srtPath: "" }
        : await prepareSceneAudioMix(progress, "Preparing combined scene-audio track for LTX", { rtvLtx25: ltx25RtvBatch });
      const scenes = batchTargetItems(sceneScope)
        .filter(({ segment }) => forceVideos || !String(selectedSegmentVideoPath(segment) || "").trim());
      const requestedSceneCount = batchTargetItems(sceneScope).length;
      renderLog.eta_plan = scenes.map(({ segment }) => renderETAScene(segment));
      renderLog.eta_video_duration = batchTargetItems(sceneScope).reduce((sum, { segment }) => sum + Math.max(0, Number(segment.end) - Number(segment.start)), 0);
      renderLog.target_scene_count = scenes.length;
      renderLog.skipped_existing_count = Math.max(0, requestedSceneCount - scenes.length);
      if (!scenes.length) {
        progress.set(skipFinalStitch ? "Selected scenes already have video. Nothing to render." : "All scenes already have video. Stitching existing scene videos...", 80);
      }
      if (!miniMaxProject && currentVideoMode() === "flf" && scenes.length && flfPreGeneratePromptsEnabled()) {
        const promptTargets = scenes.filter(({ segment }) =>
          !String(segment.i2v_prompt || "").trim()
          && firstLastFramePromptReferences(segment).length >= 2
        );
        for (let promptIndex = 0; promptIndex < promptTargets.length; promptIndex += 1) {
          assertBatchNotStopped();
          const { segment, index: sceneIndex } = promptTargets[promptIndex];
          const promptPercent = 4 + Math.floor(((promptIndex + 1) / Math.max(1, promptTargets.length)) * 8);
          progress.set(`Pre-generating FLF prompt ${promptIndex + 1}/${promptTargets.length}: ${sceneDisplayName(segment, sceneIndex)}\nUsing the previous scene image as a prompt-only provisional start reference.`, promptPercent);
          await generateI2VPromptForSegment(
            segment,
            progress,
            promptPercent,
            `FLF prompt batch ${promptIndex + 1}/${promptTargets.length}`,
            { unloadAfter: promptIndex === promptTargets.length - 1, forceVision: true },
          );
        }
        if (promptTargets.length) await autoSaveSessionQuiet("FLF provisional prompt batch complete");
      }
      renderLog.setup_ms = Math.max(0, Date.now() - logStartedMs);
      renderLog.setup_ended_at = new Date().toISOString();
      await persistRenderLog(renderLog);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const sceneStartedMs = Date.now();
        currentSceneLog = {
          ...renderETAScene(segment),
          scene_id: String(segment.id || ""),
          scene_number: sceneSlotNumber(segment),
          timeline_index: sceneIndex,
          label: sceneLabel,
          video_mode: miniMaxProject ? miniMaxH3ModeForSegment(segment) : currentVideoMode(),
          status: "running",
          phase: "preparation",
          started_at: new Date(sceneStartedMs).toISOString(),
          ended_at: "",
          preparation_ms: 0,
          render_ms: 0,
          post_ms: 0,
          gap_before_render_ms: 0,
          total_ms: 0,
          video_path: "",
          error: "",
        };
        renderLog.scenes.push(currentSceneLog);
        upsertRenderLog(renderLog);
        const base = Math.floor((index / scenes.length) * 100);
        const span = Math.max(1, Math.floor(80 / scenes.length));
        try {
        if (randomizeVideoSeed) {
          if (miniMaxProject) setMiniMaxH3SeedRandom(segment);
          else setVideoSeedRandom(segment);
        }
        if (!miniMaxProject && currentVideoMode() === "flf" && index === 0 && flfRenderChainStartSource(segment) === "rendered_frame") {
          const previousSegment = previousAutoChainSourceSegment(segment);
          if (previousSegment && String(selectedSegmentVideoPath(previousSegment) || "").trim()) {
            await prepareFLFRenderedFrameNextScene(previousSegment, segment, progress, Math.min(98, base + 1), `FLF Render Chain resume into ${sceneLabel}`);
          }
        }
        if (!miniMaxProject && currentVideoMode() === "flf" && !String(segment.i2v_prompt || "").trim()) {
          progress.set(`Creating First Last Frame prompt for ${sceneLabel}...`, Math.min(98, base + 1));
          await generateI2VPromptForSegment(segment, progress, Math.min(98, base + 2), `Render All ${index + 1}/${scenes.length}: Gemma FLF`, { unloadAfter: true, forceVision: true });
        }
        if (!miniMaxProject && i2vAutoChainEnabled() && currentVideoMode() === "i2v" && index === 0) {
          const previousSegment = previousAutoChainSourceSegment(segment);
          if (previousSegment && String(selectedSegmentVideoPath(previousSegment) || "").trim()) {
            const chainBase = Math.min(98, base + Math.max(1, Math.floor(span * 0.12)));
            await prepareAutoChainedNextScene(
              previousSegment,
              segment,
              progress,
              chainBase,
              `Auto Chain resume into ${sceneLabel}`
            );
          }
        }
        if (!miniMaxProject && img2imgContinuityEnabled() && currentVideoMode() === "i2v") {
          const previousSegment = previousAutoChainSourceSegment(segment);
          if (previousSegment && String(selectedSegmentVideoPath(previousSegment) || "").trim()) {
            const imageMode = state.imageModelMode || "zimage";
            const continuityBase = Math.min(98, base + Math.max(1, Math.floor(span * 0.10)));
            await prepareAutoImg2ImgContinuityForScene(
              previousSegment,
              segment,
              imageMode,
              progress,
              continuityBase,
              `Img2Img Continuity into ${sceneLabel}`
            );
            await createImageForSegmentInCurrentMode(
              segment,
              imageMode,
              progress,
              Math.min(98, continuityBase + 3),
              Math.max(1, Math.floor(span * 0.35)),
              `Img2Img Continuity ${sceneLabel}`
            );
            await autoSaveSessionQuiet(`Img2Img continuity image scene ${sceneIndex + 1}`);
          }
        }
        const sceneRenderStartedMs = Date.now();
        currentSceneLog.phase = "render";
        currentSceneLog.preparation_ms = Math.max(0, sceneRenderStartedMs - sceneStartedMs);
        currentSceneLog.render_started_at = new Date(sceneRenderStartedMs).toISOString();
        currentSceneLog.gap_before_render_ms = previousRenderEndedMs
          ? Math.max(0, sceneRenderStartedMs - previousRenderEndedMs)
          : 0;
        const liveSummary = updateRenderLogSummary(renderLog);
        const timingLine = liveSummary.completed_scenes
          ? `\nElapsed: ${renderLogDuration(liveSummary.total_ms)} · Estimated remaining: ${renderLogDuration(liveSummary.eta_ms)}`
          : `\nElapsed: ${renderLogDuration(liveSummary.total_ms)}`;
        progress.set(`Rendering ${sceneLabel} (${index + 1} of ${scenes.length}; ${forceVideos ? "creating a new video version" : "existing videos skipped"})...${timingLine}`, base);
        const sharedRenderOptions = {
          progressBase: base,
          progressSpan: span,
          batchLabel: `Render All ${index + 1}/${scenes.length}: ${segment.label || `Scene ${sceneIndex + 1}`}`,
          autoSaveAfter: false,
          existingVideoAction: forceVideos ? "backup" : "overwrite",
        };
        const renderedVideoPath = miniMaxProject
          ? await renderMiniMaxSceneVideoWithProgress(segment, sceneIndex, progress, {
            ...sharedRenderOptions,
            onCueValidationWarning: (warning) => renderWarnings.push(`${sceneLabel}: ${warning}`),
          })
          : await renderSceneVideoWithProgress(segment, sceneIndex, progress, {
            ...sharedRenderOptions,
            audioPathOverride: skipFinalStitch || segmentTrack(segment) === "overlay" ? "" : preparedAudio.audioPath,
            srtPathOverride: skipFinalStitch || segmentTrack(segment) === "overlay" ? "" : preparedAudio.srtPath,
          });
        const sceneRenderEndedMs = Date.now();
        previousRenderEndedMs = sceneRenderEndedMs;
        currentSceneLog.phase = "post";
        currentSceneLog.render_ended_at = new Date(sceneRenderEndedMs).toISOString();
        currentSceneLog.render_ms = Math.max(0, sceneRenderEndedMs - sceneRenderStartedMs);
        currentSceneLog.video_path = String(renderedVideoPath || selectedSegmentVideoPath(segment) || "");
        if (!miniMaxProject && currentVideoMode() === "flf" && scenes[index + 1]?.segment && flfChainingEnabled(scenes[index + 1].segment) && flfRenderChainStartSource(scenes[index + 1].segment) === "rendered_frame") {
          assertBatchNotStopped();
          const nextSegment = scenes[index + 1].segment;
          const chainBase = Math.min(98, base + Math.max(1, Math.floor(span * 0.72)));
          await prepareFLFRenderedFrameNextScene(
            segment,
            nextSegment,
            progress,
            chainBase,
            `FLF rendered-frame chain ${index + 1}->${index + 2}`,
          );
        }
        if (!miniMaxProject && i2vAutoChainEnabled() && currentVideoMode() === "i2v" && scenes[index + 1]?.segment) {
          assertBatchNotStopped();
          const nextSegment = scenes[index + 1].segment;
          const chainBase = Math.min(98, base + Math.max(1, Math.floor(span * 0.72)));
          await prepareAutoChainedNextScene(
            segment,
            nextSegment,
            progress,
            chainBase,
            `Auto Chain ${index + 1}->${index + 2}`
          );
        }
        assertBatchNotStopped();
        await runClearMemoryWorkflowQuiet(progress, sceneLabel, Math.min(98, base + span));
        const sceneEndedMs = Date.now();
        currentSceneLog.phase = "complete";
        currentSceneLog.status = "complete";
        currentSceneLog.ended_at = new Date(sceneEndedMs).toISOString();
        currentSceneLog.post_ms = Math.max(0, sceneEndedMs - sceneRenderEndedMs);
        currentSceneLog.total_ms = Math.max(0, sceneEndedMs - sceneStartedMs);
        await persistRenderLog(renderLog);
        currentSceneLog = null;
        } catch (sceneError) {
          const failedAt = Date.now();
          const message = String(sceneError?.message || sceneError);
          currentSceneLog.status = state.batchCancelled ? "canceled" : "failed";
          currentSceneLog.ended_at = new Date(failedAt).toISOString();
          currentSceneLog.total_ms = Math.max(0, failedAt - sceneStartedMs);
          currentSceneLog.error = message;
          renderLog.last_scene_error = message;
          await persistRenderLog(renderLog);
          progress.set(`${sceneLabel} failed — continuing with the remaining scenes.\n\n${message}`, Math.min(98, base + span));
          console.error(`[VRGDG Music Builder] Render All scene failed; continuing: ${sceneLabel}`, sceneError);
          currentSceneLog = null;
          if (state.batchCancelled) throw sceneError;
          continue;
        }
      }
      assertBatchNotStopped();
      const failedScenes = renderLog.scenes.filter((item) => item.status === "failed");
      if (failedScenes.length) {
        const completedAt = Date.now();
        renderLog.status = "partial";
        renderLog.ended_at = new Date(completedAt).toISOString();
        renderLog.total_ms = Math.max(0, completedAt - logStartedMs);
        renderLog.error = `${failedScenes.length} scene${failedScenes.length === 1 ? "" : "s"} failed; final stitch deferred so the batch remains resumable.`;
        await persistRenderLog(renderLog);
        await autoSaveSessionQuiet("render all partial batch complete");
        progress.set(`Render All finished with ${failedScenes.length} failed scene${failedScenes.length === 1 ? "" : "s"}.\n\nThe remaining scenes were rendered. Final stitching was deferred; run Render All again after fixing the failed scene${failedScenes.length === 1 ? "" : "s"}.\n\nRender Log:\n${renderLog.report_text_path || "saved in the project session"}`, 100);
        toast(`Render All finished with ${failedScenes.length} failed scene${failedScenes.length === 1 ? "" : "s"}. Remaining scenes continued rendering.`, true);
        return;
      }
      if (skipFinalStitch) {
        const completedAt = Date.now();
        renderLog.status = "complete";
        renderLog.ended_at = new Date(completedAt).toISOString();
        renderLog.total_ms = Math.max(0, completedAt - logStartedMs);
        await persistRenderLog(renderLog);
        await autoSaveSessionQuiet("render selected scenes complete");
        const summary = updateRenderLogSummary(renderLog);
        progress.set(`Render Selected complete.\n\nRendered/checked ${scenes.length} selected scene${scenes.length === 1 ? "" : "s"}.\nNo final stitch was run.\n\nTotal time: ${renderLogDuration(summary.total_ms)}\nActive rendering: ${renderLogDuration(summary.render_ms)}\nRender Log:\n${renderLog.report_text_path || "saved in the project session"}`, 100);
        progress.close(4500);
        toast("Render Selected complete. No final stitch was run.");
        return;
      }
      const stitchStartedMs = Date.now();
      renderLog.stitch_started_at = new Date(stitchStartedMs).toISOString();
      await persistRenderLog(renderLog);
      const stitched = await stitchRenderedScenes(progress);
      const stitchEndedMs = Date.now();
      renderLog.stitch_ended_at = new Date(stitchEndedMs).toISOString();
      renderLog.stitch_ms = Math.max(0, stitchEndedMs - stitchStartedMs);
      renderLog.final_video_path = String(stitched.final_video_path || "");
      renderLog.status = "complete";
      renderLog.warnings = renderWarnings;
      renderLog.ended_at = new Date(stitchEndedMs).toISOString();
      renderLog.total_ms = Math.max(0, stitchEndedMs - logStartedMs);
      await persistRenderLog(renderLog);
      await autoSaveSessionQuiet("render all final stitch complete");
      const summary = updateRenderLogSummary(renderLog);
      const reviewNotice = renderWarnings.length
        ? `\n\nReview after render — ${renderWarnings.length} non-blocking prompt issue${renderWarnings.length === 1 ? "" : "s"}:\n${renderWarnings.join("\n")}`
        : "";
      progress.set(`Render All complete.\n\nFinal video:\n${stitched.final_video_path}\n\nScene clips:\n${stitched.video_folder}\n\nTotal time: ${renderLogDuration(summary.total_ms)}\nActive scene rendering: ${renderLogDuration(summary.render_ms)}\nFinal stitching: ${renderLogDuration(summary.stitch_ms)}\nRender Log:\n${renderLog.report_text_path || "saved in the project session"}${reviewNotice}`, 100);
      progress.close(6500);
      toast(`Render All complete:\n${stitched.final_video_path}${reviewNotice}`, Boolean(renderWarnings.length));
      if (!options.suppressFinalModal) showFinalVideoReadyModal(stitched.final_video_path);
      return stitched;
    } catch (error) {
      const failedAt = Date.now();
      const message = String(error?.message || error);
      if (currentSceneLog && currentSceneLog.status === "running") {
        if (currentSceneLog.phase === "preparation") {
          currentSceneLog.preparation_ms = Math.max(0, failedAt - (Date.parse(currentSceneLog.started_at) || failedAt));
        } else if (currentSceneLog.phase === "render") {
          currentSceneLog.render_ms = Math.max(0, failedAt - (Date.parse(currentSceneLog.render_started_at) || failedAt));
        } else if (currentSceneLog.phase === "post") {
          currentSceneLog.post_ms = Math.max(0, failedAt - (Date.parse(currentSceneLog.render_ended_at) || failedAt));
        }
        currentSceneLog.status = state.batchCancelled ? "canceled" : "failed";
        currentSceneLog.ended_at = new Date(failedAt).toISOString();
        currentSceneLog.total_ms = Math.max(0, failedAt - (Date.parse(currentSceneLog.started_at) || failedAt));
        currentSceneLog.error = message;
      }
      if (renderLog) {
        if (!renderLog.setup_ended_at) {
          renderLog.setup_ms = Math.max(0, failedAt - logStartedMs);
          renderLog.setup_ended_at = new Date(failedAt).toISOString();
        }
        if (renderLog.stitch_started_at && !renderLog.stitch_ms) {
          renderLog.stitch_ms = Math.max(0, failedAt - (Date.parse(renderLog.stitch_started_at) || failedAt));
        }
        renderLog.status = state.batchCancelled || /\b(?:cancel|stopp?ed|interrupt)/i.test(message) ? "canceled" : "failed";
        renderLog.ended_at = new Date(failedAt).toISOString();
        renderLog.total_ms = Math.max(0, failedAt - logStartedMs);
        renderLog.error = message;
        await persistRenderLog(renderLog);
      }
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
      if (options.throwOnError) throw error;
    } finally {
      renderAllButton.disabled = false;
      renderAllButton.textContent = "Render All";
      setButtonGroupState(createSceneVideoButtons, { disabled: false, text: "Create Scene Video" });
      setButtonGroupState(miniMaxSceneVideoButtons, { disabled: false, text: "Create MiniMax H3 Scene Video" });
      setButtonGroupState(gemmaThenCreateVideoButtons, { disabled: false });
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function zImageAllScenes(options = {}) {
    updateActiveFromInputs();
    const imageRunMode = options.imageRunMode || "resume_missing";
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const forceNewImages = imageRunMode === "redo_prompts_images" || imageRunMode === "keep_prompts_redo_images";
    const redoPrompts = imageRunMode === "redo_prompts_images";
    const missing = validateZImageAllReady({ imageRunMode, sceneScope });
    const progress = createProgressWindow("Z-Image All Scenes");
    if (missing.length) {
      progress.setHtml(`
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:900;color:#fecaca;">Z-Image All cannot start yet.</div>
          <div>Fix these first, then press Z-Image All again:</div>
          <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
        </div>
      `, 100);
      toast("Z-Image All needs scene notes first.", true);
      if (options.throwOnError) throw new Error(missing.join("\n"));
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "Z-Imaging...";
      setButtonGroupState(zCreateButtons, { disabled: true });
      createT2IButton.disabled = true;
      progress.set(`Autosaving session/SRT before Z-Image All (${batchScopeLabel(sceneScope)})...`, 3);
      await saveSessionForSceneVideo();
      const scenes = imageAllSegmentsForMode(imageRunMode, "zimage", sceneScope);
      if (!scenes.length) {
        progress.set("All scenes already have images. Skipping Z-Image All.", 100);
        progress.close(1800);
        toast("All scenes already have images. Z-Image All skipped.");
        return;
      }
      if (redoPrompts) {
        scenes.forEach(({ segment }) => {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
          segment.enhance_prompt = "";
        });
      }
      const promptScenes = scenes.filter(({ segment }) => !String(segment.t2i_prompt || "").trim());
      if (promptScenes.length) {
        progress.set(`Image All: creating ${promptScenes.length} missing T2I prompt${promptScenes.length === 1 ? "" : "s"} with Gemma first...`, 6);
      } else {
        progress.set("Image All: all missing images already have T2I prompts. Skipping Gemma prompt pass...", 12);
      }
      for (let index = 0; index < promptScenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = promptScenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 6 + Math.floor((index / promptScenes.length) * 32);
        state.activeId = segment.id;
        syncInspector();
        render();
        await runGemmaImagePromptPassWithRetry(
          segment,
          progress,
          base,
          `Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}`,
          generateT2IPromptForSegment,
          { unloadAfter: false },
        );
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Image All prompt pass scene ${sceneIndex + 1}`);
      }
      if (promptScenes.length) {
        await runClearMemoryWorkflowQuiet(progress, "Image All prompt pass", 42);
      }
      progress.set(`Image All: creating ${scenes.length} ZImage image${scenes.length === 1 ? "" : "s"} from saved prompts...`, 45);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 45 + Math.floor((index / scenes.length) * 45);
        const span = Math.max(1, Math.floor(40 / scenes.length));
        state.activeId = segment.id;
        syncInspector();
        render();
        if (forceNewImages) setImageSeedForCurrentMode("zimage");
        if (img2imgContinuityEnabled() && currentVideoMode() === "i2v") {
          const previousSegment = previousAutoChainSourceSegment(segment);
          if (previousSegment) {
            await prepareAutoImg2ImgContinuityForScene(previousSegment, segment, "zimage", progress, base, `Img2Img Continuity ${index + 1}/${scenes.length}`);
          }
        }
        progress.set(`Z-Image image pass ${index + 1}/${scenes.length}: ${sceneLabel}\nCreating image from saved T2I prompt...`, base);
        await createZImageForSegment(segment, progress, base + span * 0.35, span * 0.45, `Z-Image All ${index + 1}/${scenes.length}: ZImage`);
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Z-Image All scene ${sceneIndex + 1}`);
        await runClearMemoryWorkflowQuiet(progress, sceneLabel, Math.min(98, base + span));
      }
      await autoSaveSessionQuiet("Z-Image All complete");
      progress.set("Image All complete. You can review the generated images and re-do any scenes you do not like.", 100);
      progress.close(4500);
      toast("Z-Image All complete.");
    } catch (error) {
      const errorMessage = String(error?.message || error);
      const stopped = /stopped by user/i.test(errorMessage);
      const statusLabel = stopped ? "Stopped" : "Error";
      progress.set(`${statusLabel}:\n${errorMessage}\n\nRunning memory cleanup...`, 100);
      toast(errorMessage, !stopped);
      try {
        const cleanupOutput = await runClearMemoryWorkflowQuiet(progress, stopped ? "stopped Z-Image All" : "Z-Image All error", 100);
        progress.set(`${statusLabel}:\n${errorMessage}\n\n${cleanupOutput}`, 100);
      } catch (cleanupError) {
        console.warn("[VRGDG Music Builder] Cleanup after Z-Image All stop failed:", cleanupError);
        progress.set(`${statusLabel}:\n${errorMessage}\n\nCleanup also failed:\n${String(cleanupError?.message || cleanupError)}`, 100);
      }
      if (options.throwOnError) throw error;
    } finally {
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      setButtonGroupState(zCreateButtons, { disabled: false, text: "Create Z-Image" });
      createT2IButton.disabled = false;
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  function zEnhanceBatchTargets(sceneScope = "all") {
    return batchTargetItems(sceneScope).filter(({ segment }) => {
      const source = currentEnhanceSource(segment);
      return Boolean(source?.path || source?.data || segment?.image?.filename);
    });
  }

  function validateZEnhanceAllReady({ sceneScope = "all" } = {}) {
    const missing = [];
    const allTargets = batchTargetItems(sceneScope);
    const imageTargets = zEnhanceBatchTargets(sceneScope);
    if (!allTargets.length) missing.push(batchEmptyMessage(sceneScope));
    if (!imageTargets.length) {
      missing.push(`No scene images found in ${batchScopeLabel(sceneScope)}. Create, load, or choose scene images first.`);
    }
    for (const { segment, index } of imageTargets) {
      if (!sceneImagePromptForEnhanceAll(segment).prompt) {
        missing.push(`${sceneDisplayName(segment, index)}: current T2I/image prompt is missing.`);
      }
    }
    return missing;
  }

  async function zEnhanceAllScenes(options = {}) {
    updateActiveFromInputs();
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const missing = validateZEnhanceAllReady({ sceneScope });
    const progress = createProgressWindow("Enhance All Images");
    if (missing.length) {
      progress.setHtml(`
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:900;color:#fecaca;">Enhance All cannot start yet.</div>
          <div>Fix these first, then press Enhance All again:</div>
          <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
        </div>
      `, 100);
      toast("Enhance All needs scene images and prompts first.", true);
      if (options.throwOnError) throw new Error(missing.join("\n"));
      return;
    }
    try {
      state.batchCancelled = false;
      zEnhanceAllButton.disabled = true;
      zEnhanceAllToolButton.disabled = true;
      zEnhanceAllButton.textContent = "Enhancing...";
      zEnhanceAllToolButton.textContent = "Enhancing...";
      zEnhanceButton.disabled = true;
      zEnhanceButton.textContent = "Enhancing...";
      progress.set(`Autosaving session/SRT before Enhance All (${batchScopeLabel(sceneScope)})...`, 3);
      await saveSessionForSceneVideo();
      const scenes = zEnhanceBatchTargets(sceneScope);
      const settings = saveZEnhanceSettingsFromPanel();
      pushHistory();
      progress.set(`Enhance All: enhancing ${scenes.length} timeline image${scenes.length === 1 ? "" : "s"}...`, 10);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 10 + Math.floor((index / scenes.length) * 82);
        const span = Math.max(1, Math.floor(76 / scenes.length));
        progress.set(`Enhance All ${index + 1}/${scenes.length}: ${sceneLabel}\nUpscaling/enhancing selected scene image...`, base);
        await enhanceImageForSegment(segment, progress, base + span * 0.1, span * 0.75, `Enhance All ${index + 1}/${scenes.length}: ${sceneLabel}`, {
          settings,
          promptSource: "scene_image_prompt",
        });
        if (currentVideoMode() === "flf" && scenes[index + 1]?.segment) {
          assertBatchNotStopped();
          await prepareFLFRenderedFrameNextScene(segment, scenes[index + 1].segment, progress, Math.min(98, base + Math.max(1, Math.floor(span * 0.72))), `FLF Render Chain ${index + 1}->${index + 2}`);
        }
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Enhance All scene ${sceneIndex + 1}`);
        await runImageMemoryCleanupQuiet(progress, sceneLabel, Math.min(98, base + span));
      }
      await autoSaveSessionQuiet("Enhance All complete");
      progress.set("Enhance All complete. Review the enhanced image versions in the timeline.", 100);
      progress.close(4500);
      toast("Enhance All complete.");
    } catch (error) {
      const errorMessage = String(error?.message || error);
      const stopped = /stopped by user/i.test(errorMessage);
      const statusLabel = stopped ? "Stopped" : "Error";
      progress.set(`${statusLabel}:\n${errorMessage}\n\nRunning memory cleanup...`, 100);
      toast(errorMessage, !stopped);
      try {
        const cleanupOutput = await runImageMemoryCleanupQuiet(progress, stopped ? "stopped Enhance All" : "Enhance All error", 100);
        progress.set(`${statusLabel}:\n${errorMessage}\n\n${cleanupOutput}`, 100);
      } catch (cleanupError) {
        console.warn("[VRGDG Music Builder] Cleanup after Enhance All stop failed:", cleanupError);
        progress.set(`${statusLabel}:\n${errorMessage}\n\nCleanup also failed:\n${String(cleanupError?.message || cleanupError)}`, 100);
      }
      if (options.throwOnError) throw error;
    } finally {
      zEnhanceAllButton.disabled = false;
      zEnhanceAllToolButton.disabled = false;
      zEnhanceAllButton.textContent = "Enhance All";
      zEnhanceAllToolButton.textContent = "Enhance All";
      zEnhanceButton.disabled = false;
      zEnhanceButton.textContent = "Upscale / Enhance Image";
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function ernieImageAllScenes(options = {}) {
    updateActiveFromInputs();
    const imageRunMode = options.imageRunMode || "resume_missing";
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const forceNewImages = imageRunMode === "redo_prompts_images" || imageRunMode === "keep_prompts_redo_images";
    const redoPrompts = imageRunMode === "redo_prompts_images";
    const missing = validateZImageAllReady({ imageRunMode, sceneScope });
    const progress = createProgressWindow("Ernie Image All Scenes");
    if (missing.length) {
      progress.setHtml(`
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:900;color:#fecaca;">Ernie Image All cannot start yet.</div>
          <div>Fix these first, then press Image All again:</div>
          <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
        </div>
      `, 100);
      toast("Ernie Image All needs scene notes first.", true);
      if (options.throwOnError) throw new Error(missing.join("\n"));
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "Ernie...";
      setButtonGroupState(ernieCreateButtons, { disabled: true });
      createT2IButton.disabled = true;
      ernieCreateT2IButton.disabled = true;
      progress.set(`Autosaving session/SRT before Ernie Image All (${batchScopeLabel(sceneScope)})...`, 3);
      await saveSessionForSceneVideo();
      const scenes = imageAllSegmentsForMode(imageRunMode, "ernie_image", sceneScope);
      if (!scenes.length) {
        progress.set("All scenes already have images. Skipping Ernie Image All.", 100);
        progress.close(1800);
        toast("All scenes already have images. Ernie Image All skipped.");
        return;
      }
      if (redoPrompts) {
        scenes.forEach(({ segment }) => {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
          segment.enhance_prompt = "";
        });
      }
      const promptScenes = scenes.filter(({ segment }) => !String(segment.t2i_prompt || "").trim());
      if (promptScenes.length) {
        progress.set(`Image All: creating ${promptScenes.length} missing T2I prompt${promptScenes.length === 1 ? "" : "s"} with Gemma first...`, 6);
      } else {
        progress.set("Image All: all missing images already have T2I prompts. Skipping Gemma prompt pass...", 12);
      }
      for (let index = 0; index < promptScenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = promptScenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 6 + Math.floor((index / promptScenes.length) * 32);
        state.activeId = segment.id;
        syncInspector();
        render();
        await runGemmaImagePromptPassWithRetry(
          segment,
          progress,
          base,
          `Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}`,
          generateT2IPromptForSegment,
          { unloadAfter: false },
        );
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Image All prompt pass scene ${sceneIndex + 1}`);
      }
      if (promptScenes.length) {
        await runClearMemoryWorkflowQuiet(progress, "Image All prompt pass", 42);
      }
      progress.set(`Image All: creating ${scenes.length} Ernie image${scenes.length === 1 ? "" : "s"} from saved prompts...`, 45);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 45 + Math.floor((index / scenes.length) * 45);
        const span = Math.max(1, Math.floor(40 / scenes.length));
        state.activeId = segment.id;
        syncInspector();
        render();
        if (forceNewImages) setImageSeedForCurrentMode("ernie_image");
        if (img2imgContinuityEnabled() && currentVideoMode() === "i2v") {
          const previousSegment = previousAutoChainSourceSegment(segment);
          if (previousSegment) {
            await prepareAutoImg2ImgContinuityForScene(previousSegment, segment, "ernie_image", progress, base, `Img2Img Continuity ${index + 1}/${scenes.length}`);
          }
        }
        progress.set(`Ernie image pass ${index + 1}/${scenes.length}: ${sceneLabel}\nCreating image from saved T2I prompt...`, base);
        await createErnieImageForSegment(segment, progress, base + span * 0.35, span * 0.45, `Ernie Image All ${index + 1}/${scenes.length}: Ernie`);
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Ernie Image All scene ${sceneIndex + 1}`);
        await runClearMemoryWorkflowQuiet(progress, sceneLabel, Math.min(98, base + span));
      }
      await autoSaveSessionQuiet("Ernie Image All complete");
      progress.set("Image All complete. You can review the generated images and re-do any scenes you do not like.", 100);
      progress.close(4500);
      toast("Ernie Image All complete.");
    } catch (error) {
      const errorMessage = String(error?.message || error);
      const stopped = /stopped by user/i.test(errorMessage);
      const statusLabel = stopped ? "Stopped" : "Error";
      progress.set(`${statusLabel}:\n${errorMessage}\n\nRunning memory cleanup...`, 100);
      toast(errorMessage, !stopped);
      try {
        const cleanupOutput = await runClearMemoryWorkflowQuiet(progress, stopped ? "stopped Ernie Image All" : "Ernie Image All error", 100);
        progress.set(`${statusLabel}:\n${errorMessage}\n\n${cleanupOutput}`, 100);
      } catch (cleanupError) {
        console.warn("[VRGDG Music Builder] Cleanup after Ernie Image All stop failed:", cleanupError);
        progress.set(`${statusLabel}:\n${errorMessage}\n\nCleanup also failed:\n${String(cleanupError?.message || cleanupError)}`, 100);
      }
      if (options.throwOnError) throw error;
    } finally {
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      setButtonGroupState(ernieCreateButtons, { disabled: false, text: "Create with Ernie" });
      createT2IButton.disabled = false;
      ernieCreateT2IButton.disabled = false;
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function krea2TwoPassImageAllScenes(options = {}) {
    updateActiveFromInputs();
    const imageRunMode = options.imageRunMode || "resume_missing";
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const forceNewImages = imageRunMode === "redo_prompts_images" || imageRunMode === "keep_prompts_redo_images";
    const redoPrompts = imageRunMode === "redo_prompts_images";
    const missing = validateZImageAllReady({ imageRunMode, sceneScope });
    const progress = createProgressWindow("Krea 2 Image All Scenes");
    if (missing.length) {
      progress.setHtml(`
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:900;color:#fecaca;">Krea 2 Image All cannot start yet.</div>
          <div>Fix these first, then press Image All again:</div>
          <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
        </div>
      `, 100);
      toast("Krea 2 Image All needs scene notes first.", true);
      if (options.throwOnError) throw new Error(missing.join("\n"));
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "Krea2...";
      setButtonGroupState(krea2TwoPassCreateButtons, { disabled: true });
      createT2IButton.disabled = true;
      krea2TwoPassCreateT2IButton.disabled = true;
      progress.set(`Autosaving session/SRT before Krea 2 Image All (${batchScopeLabel(sceneScope)})...`, 3);
      await saveSessionForSceneVideo();
      const scenes = imageAllSegmentsForMode(imageRunMode, "krea2_2pass", sceneScope);
      if (!scenes.length) {
        progress.set("All scenes already have images. Skipping Krea 2 Image All.", 100);
        progress.close(1800);
        toast("All scenes already have images. Krea 2 Image All skipped.");
        return;
      }
      if (redoPrompts) {
        scenes.forEach(({ segment }) => {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
          segment.enhance_prompt = "";
        });
      }
      const promptScenes = scenes.filter(({ segment }) => !String(segment.t2i_prompt || "").trim());
      if (promptScenes.length) {
        progress.set(`Image All: creating ${promptScenes.length} missing T2I prompt${promptScenes.length === 1 ? "" : "s"} with Gemma first...`, 6);
      } else {
        progress.set("Image All: all missing images already have T2I prompts. Skipping Gemma prompt pass...", 12);
      }
      for (let index = 0; index < promptScenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = promptScenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 6 + Math.floor((index / promptScenes.length) * 32);
        state.activeId = segment.id;
        syncInspector();
        render();
        await runGemmaImagePromptPassWithRetry(
          segment,
          progress,
          base,
          `Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}`,
          generateT2IPromptForSegment,
          { unloadAfter: false },
        );
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Image All prompt pass scene ${sceneIndex + 1}`);
      }
      if (promptScenes.length) {
        await runClearMemoryWorkflowQuiet(progress, "Image All prompt pass", 42);
      }
      progress.set(`Image All: creating ${scenes.length} Krea 2 image${scenes.length === 1 ? "" : "s"} from saved prompts...`, 45);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 45 + Math.floor((index / scenes.length) * 45);
        const span = Math.max(1, Math.floor(40 / scenes.length));
        state.activeId = segment.id;
        syncInspector();
        render();
        if (forceNewImages) setImageSeedForCurrentMode("krea2_2pass");
        if (img2imgContinuityEnabled() && currentVideoMode() === "i2v") {
          const previousSegment = previousAutoChainSourceSegment(segment);
          if (previousSegment) {
            await prepareAutoImg2ImgContinuityForScene(previousSegment, segment, "krea2_2pass", progress, base, `Img2Img Continuity ${index + 1}/${scenes.length}`);
          }
        }
        progress.set(`Krea 2 image pass ${index + 1}/${scenes.length}: ${sceneLabel}\nCreating image from saved T2I prompt...`, base);
        await createKrea2TwoPassImageForSegment(segment, progress, base + span * 0.35, span * 0.45, `Krea 2 Image All ${index + 1}/${scenes.length}: Krea 2`);
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Krea 2 Image All scene ${sceneIndex + 1}`);
        await runClearMemoryWorkflowQuiet(progress, sceneLabel, Math.min(98, base + span));
      }
      await autoSaveSessionQuiet("Krea 2 Image All complete");
      progress.set("Image All complete. You can review the generated images and re-do any scenes you do not like.", 100);
      progress.close(4500);
      toast("Krea 2 Image All complete.");
    } catch (error) {
      const errorMessage = String(error?.message || error);
      const stopped = /stopped by user/i.test(errorMessage);
      const statusLabel = stopped ? "Stopped" : "Error";
      progress.set(`${statusLabel}:\n${errorMessage}\n\nRunning memory cleanup...`, 100);
      toast(errorMessage, !stopped);
      try {
        const cleanupOutput = await runClearMemoryWorkflowQuiet(progress, stopped ? "stopped Krea 2 Image All" : "Krea 2 Image All error", 100);
        progress.set(`${statusLabel}:\n${errorMessage}\n\n${cleanupOutput}`, 100);
      } catch (cleanupError) {
        console.warn("[VRGDG Music Builder] Cleanup after Krea 2 Image All stop failed:", cleanupError);
        progress.set(`${statusLabel}:\n${errorMessage}\n\nCleanup also failed:\n${String(cleanupError?.message || cleanupError)}`, 100);
      }
      if (options.throwOnError) throw error;
    } finally {
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      setButtonGroupState(krea2TwoPassCreateButtons, { disabled: false, text: "Create with Krea 2" });
      createT2IButton.disabled = false;
      krea2TwoPassCreateT2IButton.disabled = false;
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function fluxKleinAllScenes(options = {}) {
    updateActiveFromInputs();
    const imageRunMode = options.imageRunMode || "resume_missing";
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const forceNewImages = imageRunMode === "redo_prompts_images" || imageRunMode === "keep_prompts_redo_images";
    const redoPrompts = imageRunMode === "redo_prompts_images";
    const progress = createProgressWindow("Flux/Klein All Scenes");
    if (img2imgContinuityEnabled() && currentVideoMode() === "i2v") {
      const message = "Img2Img continuity is not available for Flux/Klein yet. Choose ZImage, Ernie, or Krea 2 for the image model, or turn continuity Off.";
      progress.set(message, 100);
      toast(message, true);
      if (options.throwOnError) throw new Error(message);
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "Fluxing...";
      setButtonGroupState(fluxCreateButtons, { disabled: true });
      createFluxPromptButton.disabled = true;
      progress.set(`Autosaving session/SRT before Flux/Klein All (${batchScopeLabel(sceneScope)})...`, 3);
      await saveSessionForSceneVideo();
      const scenes = imageAllSegmentsForMode(imageRunMode, "flux_klein", sceneScope);
      if (!scenes.length) {
        progress.set("All scenes already have images. Skipping Flux/Klein All.", 100);
        progress.close(1800);
        toast("All scenes already have images. Flux/Klein All skipped.");
        return;
      }
      if (redoPrompts) {
        scenes.forEach(({ segment }) => {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
          segment.enhance_prompt = "";
        });
      }
      const promptScenes = scenes.filter(({ segment }) => !String(segment.flux_prompt || segment.t2i_prompt || "").trim());
      if (promptScenes.length) {
        progress.set(`Image All: creating ${promptScenes.length} missing Flux/Klein prompt${promptScenes.length === 1 ? "" : "s"} with Gemma first...`, 6);
      } else {
        progress.set("Image All: all missing images already have image prompts. Skipping Gemma prompt pass...", 12);
      }
      for (let index = 0; index < promptScenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = promptScenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 6 + Math.floor((index / promptScenes.length) * 32);
        try {
          await runGemmaImagePromptPassWithRetry(
            segment,
            progress,
            base,
            `Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}`,
            generateFluxKleinPromptForSegment,
            { clearBeforeLoad: index === 0, unloadAfter: false },
          );
          assertBatchNotStopped();
          await autoSaveSessionQuiet(`Image All prompt pass scene ${sceneIndex + 1}`);
        } catch (error) {
          throw new Error(`Image All prompt pass stopped at ${sceneLabel} (${index + 1}/${promptScenes.length}):\n${String(error?.message || error || "Unknown error")}`);
        }
      }
      if (promptScenes.length) {
        await runClearMemoryWorkflowQuiet(progress, "Image All prompt pass", 42);
      }
      progress.set(`Image All: creating ${scenes.length} Flux/Klein image${scenes.length === 1 ? "" : "s"} from saved prompts...`, 45);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 45 + Math.floor((index / scenes.length) * 45);
        const span = Math.max(1, Math.floor(40 / scenes.length));
        try {
          if (forceNewImages) setImageSeedForCurrentMode("flux_klein");
          progress.set(`Flux/Klein image pass ${index + 1}/${scenes.length}: ${sceneLabel}\nCreating image from saved Flux/Klein prompt...`, base);
          await createFluxKleinImageForSegment(segment, progress, base + span * 0.35, span * 0.45, `Flux/Klein All ${index + 1}/${scenes.length}: Image`);
          assertBatchNotStopped();
          await autoSaveSessionQuiet(`Flux/Klein All scene ${sceneIndex + 1}`);
          await runClearMemoryWorkflowQuiet(progress, sceneLabel, Math.min(98, base + span));
        } catch (error) {
          throw new Error(`Flux/Klein image pass stopped at ${sceneLabel} (${index + 1}/${scenes.length}):\n${String(error?.message || error || "Unknown error")}`);
        }
      }
      await autoSaveSessionQuiet("Flux/Klein All complete");
      progress.set("Image All complete. You can review the generated images and re-do any scenes you do not like.", 100);
      progress.close(4500);
      toast("Flux/Klein All complete.");
    } catch (error) {
      const errorMessage = String(error?.message || error);
      const stopped = /stopped by user/i.test(errorMessage);
      const statusLabel = stopped ? "Stopped" : "Error";
      progress.set(`${statusLabel}:\n${errorMessage}\n\nRunning memory cleanup...`, 100);
      toast(errorMessage, !stopped);
      try {
        const cleanupOutput = await runClearMemoryWorkflowQuiet(progress, stopped ? "stopped Flux/Klein All" : "Flux/Klein All error", 100);
        progress.set(`${statusLabel}:\n${errorMessage}\n\n${cleanupOutput}`, 100);
      } catch (cleanupError) {
        console.warn("[VRGDG Music Builder] Cleanup after Flux/Klein All stop failed:", cleanupError);
        progress.set(`${statusLabel}:\n${errorMessage}\n\nCleanup also failed:\n${String(cleanupError?.message || cleanupError)}`, 100);
      }
      if (options.throwOnError) throw error;
    } finally {
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      setButtonGroupState(fluxCreateButtons, { disabled: false, text: "Create with Flux/Klein" });
      createFluxPromptButton.disabled = false;
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function nbImageAllScenes(options = {}) {
    updateActiveFromInputs();
    const imageRunMode = options.imageRunMode || "resume_missing";
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const redoPrompts = imageRunMode === "redo_prompts_images";
    const progress = createProgressWindow("NanoBanana All Scenes");
    if (img2imgContinuityEnabled() && currentVideoMode() === "i2v") {
      const message = "Img2Img continuity is not available for NanoBanana yet. Choose ZImage, Ernie, or Krea 2 for the image model, or turn continuity Off.";
      progress.set(message, 100);
      toast(message, true);
      if (options.throwOnError) throw new Error(message);
      return;
    }
    if (!String((state.nbImageSettings || {}).api_key || "").trim() && !allEditableSegments().some((segment) => String(segment.nb_image_settings?.api_key || "").trim())) {
      const message = "NanoBanana All needs a NanoBanana API key in the NB Models tab.";
      progress.set(message, 100);
      toast(message, true);
      if (options.throwOnError) throw new Error(message);
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "NanoBanana...";
      setButtonGroupState(nbCreateButtons, { disabled: true });
      createNBPromptButton.disabled = true;
      progress.set(`Autosaving session/SRT before NanoBanana All (${batchScopeLabel(sceneScope)})...`, 3);
      await saveSessionForSceneVideo();
      const scenes = imageAllSegmentsForMode(imageRunMode, "nano_banana", sceneScope);
      if (!scenes.length) {
        progress.set("All scenes already have images. Skipping NanoBanana All.", 100);
        progress.close(1800);
        toast("All scenes already have images. NanoBanana All skipped.");
        return;
      }
      if (redoPrompts) {
        scenes.forEach(({ segment }) => {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
          segment.enhance_prompt = "";
        });
      }
      const promptScenes = scenes.filter(({ segment }) => !String(segment.nb_prompt || segment.t2i_prompt || "").trim());
      if (promptScenes.length) {
        progress.set(`Image All: creating ${promptScenes.length} missing NanoBanana prompt${promptScenes.length === 1 ? "" : "s"} with Gemma first...`, 6);
      } else {
        progress.set("Image All: all missing images already have image prompts. Skipping Gemma prompt pass...", 12);
      }
      for (let index = 0; index < promptScenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = promptScenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 6 + Math.floor((index / promptScenes.length) * 32);
        try {
          await runGemmaImagePromptPassWithRetry(
            segment,
            progress,
            base,
            `Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}`,
            generateNBPromptForSegment,
            { clearBeforeLoad: index === 0, unloadAfter: false },
          );
          assertBatchNotStopped();
          await autoSaveSessionQuiet(`Image All prompt pass scene ${sceneIndex + 1}`);
        } catch (error) {
          throw new Error(`Image All prompt pass stopped at ${sceneLabel} (${index + 1}/${promptScenes.length}):\n${String(error?.message || error || "Unknown error")}`);
        }
      }
      if (promptScenes.length) {
        await runClearMemoryWorkflowQuiet(progress, "Image All prompt pass", 42);
      }
      progress.set(`Image All: creating ${scenes.length} NanoBanana image${scenes.length === 1 ? "" : "s"} from saved prompts...`, 45);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 45 + Math.floor((index / scenes.length) * 45);
        const span = Math.max(1, Math.floor(40 / scenes.length));
        try {
          progress.set(`NanoBanana image pass ${index + 1}/${scenes.length}: ${sceneLabel}\nCreating image from saved NanoBanana prompt...`, base);
          await createNBImageForSegmentWithRetry(
            segment,
            progress,
            base + span * 0.35,
            span * 0.45,
            `NanoBanana All ${index + 1}/${scenes.length}: ${sceneLabel}`,
            { maxRetries: 10 },
          );
          assertBatchNotStopped();
          await autoSaveSessionQuiet(`NanoBanana All scene ${sceneIndex + 1}`);
          await runClearMemoryWorkflowQuiet(progress, sceneLabel, Math.min(98, base + span));
        } catch (error) {
          throw new Error(`NanoBanana image pass stopped at ${sceneLabel} (${index + 1}/${scenes.length}):\n${String(error?.message || error || "Unknown error")}`);
        }
      }
      await autoSaveSessionQuiet("NanoBanana All complete");
      progress.set("Image All complete. You can review the generated images and re-do any scenes you do not like.", 100);
      progress.close(4500);
      toast("NanoBanana All complete.");
    } catch (error) {
      const errorMessage = String(error?.message || error);
      const stopped = /stopped by user/i.test(errorMessage);
      const statusLabel = stopped ? "Stopped" : "Error";
      progress.set(`${statusLabel}:\n${errorMessage}\n\nRunning memory cleanup...`, 100);
      toast(errorMessage, !stopped);
      try {
        const cleanupOutput = await runClearMemoryWorkflowQuiet(progress, stopped ? "stopped NanoBanana All" : "NanoBanana All error", 100);
        progress.set(`${statusLabel}:\n${errorMessage}\n\n${cleanupOutput}`, 100);
      } catch (cleanupError) {
        console.warn("[VRGDG Music Builder] Cleanup after NanoBanana All stop failed:", cleanupError);
        progress.set(`${statusLabel}:\n${errorMessage}\n\nCleanup also failed:\n${String(cleanupError?.message || cleanupError)}`, 100);
      }
      if (options.throwOnError) throw error;
    } finally {
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      setButtonGroupState(nbCreateButtons, { disabled: false, text: "Create with NanoBanana" });
      createNBPromptButton.disabled = false;
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function flowGptImageAllScenes(options = {}) {
    updateActiveFromInputs();
    saveFlowGptBrowserSettingsFromPanel();
    const imageRunMode = options.imageRunMode || "resume_missing";
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const redoPrompts = imageRunMode === "redo_prompts_images";
    const progress = createProgressWindow("Flow/GPT Image All Scenes");
    if (img2imgContinuityEnabled() && currentVideoMode() === "i2v") {
      const message = "Img2Img continuity is not available for Flow/GPT yet. Choose ZImage, Ernie, or Krea 2 for the image model, or turn continuity Off.";
      progress.set(message, 100);
      toast(message, true);
      if (options.throwOnError) throw new Error(message);
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "Flow/GPT...";
      flowGptCreateImageButton.disabled = true;
      flowGptCreatePromptButton.disabled = true;
      progress.set(`Autosaving session/SRT before Flow/GPT Image All (${batchScopeLabel(sceneScope)})...`, 3);
      await saveSessionForSceneVideo();
      const scenes = imageAllSegmentsForMode(imageRunMode, "flow_gpt", sceneScope);
      if (!scenes.length) {
        progress.set("All scenes already have images. Skipping Flow/GPT Image All.", 100);
        progress.close(1800);
        toast("All scenes already have images. Flow/GPT Image All skipped.");
        return;
      }
      if (redoPrompts) {
        scenes.forEach(({ segment }) => {
          segment.t2i_prompt = "";
          segment.flux_prompt = "";
          segment.nb_prompt = "";
          segment.flow_gpt_prompt = "";
          segment.enhance_prompt = "";
        });
      }
      const promptScenes = scenes.filter(({ segment }) => !String(segment.flow_gpt_prompt || segment.nb_prompt || segment.t2i_prompt || "").trim());
      if (promptScenes.length) {
        progress.set(`Image All: creating ${promptScenes.length} missing Flow/GPT prompt${promptScenes.length === 1 ? "" : "s"} with Gemma first...`, 6);
      } else {
        progress.set("Image All: all missing images already have Flow/GPT prompts. Skipping Gemma prompt pass...", 12);
      }
      for (let index = 0; index < promptScenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = promptScenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 6 + Math.floor((index / promptScenes.length) * 32);
        try {
          await runGemmaImagePromptPassWithRetry(
            segment,
            progress,
            base,
            `Image All prompt pass ${index + 1}/${promptScenes.length}: ${sceneLabel}`,
            generateNBPromptForSegment,
            { clearBeforeLoad: index === 0, unloadAfter: false },
          );
          if (!String(segment.flow_gpt_prompt || "").trim()) {
            syncSegmentFlowGptPrompt(segment, segment.nb_prompt || segment.t2i_prompt || "");
          }
          assertBatchNotStopped();
          await autoSaveSessionQuiet(`Image All prompt pass scene ${sceneIndex + 1}`);
        } catch (error) {
          throw new Error(`Image All prompt pass stopped at ${sceneLabel} (${index + 1}/${promptScenes.length}):\n${String(error?.message || error || "Unknown error")}`);
        }
      }
      if (promptScenes.length) {
        await runClearMemoryWorkflowQuiet(progress, "Image All prompt pass", 42);
      }
      const browserSettings = cloneFlowGptBrowserSettings(state.flowGptBrowserSettings);
      const includePreviousSceneImage = Boolean(browserSettings.ask_previous_scene_image);
      progress.set(`Image All: creating ${scenes.length} Flow/GPT image${scenes.length === 1 ? "" : "s"} from saved prompts...`, 45);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 45 + Math.floor((index / scenes.length) * 45);
        const span = Math.max(1, Math.floor(40 / scenes.length));
        try {
          const sameLocationDiversity = currentVideoMode() === "flf" ? flfSameLocationCameraDiversityDirection(segment, "start") : "";
          const previousStartIngredient = sameLocationDiversity ? previousSceneStartImageIngredient(segment) : null;
          progress.set(`Flow/GPT image pass ${index + 1}/${scenes.length}: ${sceneLabel}\nCreating image from saved browser prompt...`, base);
          await createFlowGptImageForSegment(
            segment,
            progress,
            base + span * 0.35,
            span * 0.45,
            `Flow/GPT All ${index + 1}/${scenes.length}: ${sceneLabel}`,
            {
              includePreviousSceneImage: Boolean(previousStartIngredient) || includePreviousSceneImage,
              previousSceneImageIngredient: previousStartIngredient,
              previousSceneImagePurpose: previousStartIngredient ? "same_location_composition_diversity" : "continuity",
            },
          );
          assertBatchNotStopped();
          await autoSaveSessionQuiet(`Flow/GPT Image All scene ${sceneIndex + 1}`);
        } catch (error) {
          throw new Error(`Flow/GPT image pass stopped at ${sceneLabel} (${index + 1}/${scenes.length}):\n${String(error?.message || error || "Unknown error")}`);
        }
      }
      await autoSaveSessionQuiet("Flow/GPT Image All complete");
      progress.set("Image All complete. You can review the generated images and re-do any scenes you do not like.", 100);
      progress.close(4500);
      toast("Flow/GPT Image All complete.");
    } catch (error) {
      const errorMessage = String(error?.message || error);
      const stopped = /stopped by user/i.test(errorMessage);
      const statusLabel = stopped ? "Stopped" : "Error";
      progress.set(`${statusLabel}:\n${errorMessage}`, 100);
      toast(errorMessage, !stopped);
      if (options.throwOnError) throw error;
    } finally {
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      flowGptCreateImageButton.disabled = false;
      flowGptCreateImageButton.textContent = "Create with Browser AI";
      flowGptCreatePromptButton.disabled = false;
      flowGptCreatePromptButton.textContent = "Gemma Browser Prompt";
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function createEndFramesAllScenes(options = {}) {
    updateActiveFromInputs();
    const imageMode = options.imageMode || state.imageModelMode || "zimage";
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const endFrameRunMode = options.endFrameRunMode || "resume_missing";
    const missing = validateCreateEndFramesReady({ endFrameRunMode, sceneScope });
    const progress = createProgressWindow("Create End Frames");
    if (missing.length) {
      progress.setHtml(`
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div style="font-weight:900;color:#fecaca;">Create End Frames cannot start yet.</div>
          <div>Fix these first, then try again:</div>
          <div style="max-height:360px;overflow:auto;border:1px solid #7f1d1d;border-radius:6px;background:#1f0808;padding:10px;white-space:pre-wrap;">${escapeHtml(missing.map((item) => `- ${item}`).join("\n"))}</div>
        </div>
      `, 100);
      toast("Create End Frames needs First Last Frame mode and first-frame scene images.", true);
      if (options.throwOnError) throw new Error(missing.join("\n"));
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "End Frames...";
      progress.set(`Autosaving session/SRT before Create End Frames (${batchScopeLabel(sceneScope)})...`, 3);
      await saveSessionForSceneVideo();
      const scenes = endFrameSegmentsForMode(endFrameRunMode, sceneScope);
      if (!scenes.length) {
        progress.set("All target scenes already have end frames. Skipping Create End Frames.", 100);
        progress.close(1800);
        toast("All target scenes already have end frames.");
        return;
      }
      progress.set(`Create End Frames: generating ${scenes.length} last frame image${scenes.length === 1 ? "" : "s"} with ${imageModeDisplayLabel(imageMode, true)}...`, 8);
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 8 + Math.floor((index / scenes.length) * 82);
        const span = Math.max(1, Math.floor(76 / scenes.length));
        state.activeId = segment.id;
        syncInspector();
        render();
        progress.set(`Create End Frames ${index + 1}/${scenes.length}: ${sceneLabel}\nUsing selected scene image as the first frame.`, base);
        await createEndFrameForSegment(
          segment,
          imageMode,
          progress,
          base + span * 0.12,
          span * 0.68,
          `End Frame ${index + 1}/${scenes.length}: ${sceneLabel}`
        );
        assertBatchNotStopped();
        await autoSaveSessionQuiet(`Create End Frame scene ${sceneIndex + 1}`);
        await runClearMemoryWorkflowQuiet(progress, sceneLabel, Math.min(98, base + span));
      }
      await autoSaveSessionQuiet("Create End Frames complete");
      progress.set("Create End Frames complete. Reference to Video can now use the first and last frame refs.", 100);
      progress.close(4500);
      toast("Create End Frames complete.");
    } catch (error) {
      const errorMessage = String(error?.message || error);
      const stopped = /stopped by user/i.test(errorMessage);
      const statusLabel = stopped ? "Stopped" : "Error";
      progress.set(`${statusLabel}:\n${errorMessage}`, 100);
      toast(errorMessage, !stopped);
      if (options.throwOnError) throw error;
    } finally {
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function buildIndependentFLFPairs(options = {}) {
    updateActiveFromInputs();
    const imageMode = options.imageMode || state.imageModelMode || "zimage";
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const runMode = ["resume_missing", "redo_endpoints", "redo_all"].includes(options.runMode) ? options.runMode : "resume_missing";
    const redoStarts = runMode === "redo_all";
    const redoMotionAndEnds = runMode !== "resume_missing";
    const scenes = batchTargetItems(sceneScope, { baseOnly: true }).slice().sort((a, b) => Number(a.segment?.start || 0) - Number(b.segment?.start || 0));
    const progress = createProgressWindow("Build Independent Start + End Pairs");
    const missing = [];
    if (currentVideoMode() !== "flf") missing.push("Video mode must be First Last Frame.");
    if (!scenes.length) missing.push(batchEmptyMessage(sceneScope));
    if (!String(projectInput.value || state.projectFolder || "").trim()) missing.push("Project folder is missing.");
    if (missing.length) {
      progress.set(`Build Independent Start + End Pairs cannot start yet:\n\n${missing.map((item) => `- ${item}`).join("\n")}`, 100);
      toast("Independent FLF pairs need First Last Frame mode and a saved project.", true);
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "Independent FLF...";
      state.i2vVideoSettings = {
        ...(state.i2vVideoSettings || defaultI2VVideoSettings()),
        flf_chain_previous_end_frame: false,
      };
      flfChainPreviousEndFrame.input.checked = false;
      flfStructureModeSelect.value = "independent";
      flfChainedSettingsPanel.style.display = "none";
      state.segments.forEach((segment) => {
        if (segment.use_scene_i2v_video_settings && segment.i2v_video_settings) {
          segment.i2v_video_settings.flf_chain_previous_end_frame = false;
        }
      });
      scenes.forEach(({ segment }) => {
        segment.flf_rendered_start_frame_path = "";
        segment.flf_rendered_start_frame_data = "";
        segment.flf_rendered_start_frame_name = "";
        segment.flf_rendered_source_video_path = "";
        if (redoMotionAndEnds) {
          segment.flf_motion_plan = "";
          segment.flf_end_frame_prompt = "";
          segment.flf_final_prompt_ready = false;
        }
        if (redoStarts) segment.flf_final_prompt_ready = false;
      });
      progress.set(`Preparing ${scenes.length} independent FLF pair${scenes.length === 1 ? "" : "s"} (${batchScopeLabel(sceneScope)}).\n\nPass order is locked: ALL starts → ALL motion plans → ALL ends → ALL final FLF prompts.`, 2);
      await saveSessionForSceneVideo();

      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const existing = segmentImageSource(segment);
        const base = 4 + Math.floor((index / Math.max(1, scenes.length)) * 25);
        if (!redoStarts && (existing?.path || existing?.data)) {
          progress.set(`PASS 1/4 — START FRAMES ${index + 1}/${scenes.length}: ${sceneLabel}\nKeeping this scene's existing start image. No end frames will be created until every start is ready.`, base);
          continue;
        }
        progress.set(`PASS 1/4 — START FRAMES ${index + 1}/${scenes.length}: ${sceneLabel}\nCreating this scene's normal Image All start image.`, base);
        await generateFLFStartImageForSegment(
          segment,
          imageMode,
          progress,
          base,
          Math.max(2, Math.floor(23 / Math.max(1, scenes.length))),
          `Independent start ${index + 1}/${scenes.length}: ${sceneLabel}`,
          {
            useExistingPrompt: !redoStarts && Boolean(savedImagePromptForMode(segment, imageMode)),
            preserveSavedPrompts: false,
          },
        );
        segment.flf_motion_plan = "";
        segment.flf_end_frame_stale = hasFirstLastFrameEndImage(segment);
        segment.flf_final_prompt_ready = false;
        await autoSaveSessionQuiet(`Independent FLF start image ${sceneLabel}`);
      }
      progress.set("PASS 1/4 COMPLETE — Every target scene now has its own start image.\n\nClearing prompt/image memory before motion planning...", 30);
      await runClearMemoryWorkflowQuiet(progress, "independent FLF start-frame pass", 31);

      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 32 + Math.floor((index / Math.max(1, scenes.length)) * 18);
        if (!redoMotionAndEnds && String(segment.flf_motion_plan || "").trim()) {
          progress.set(`PASS 2/4 — MOTION PLANS ${index + 1}/${scenes.length}: ${sceneLabel}\nKeeping the saved provisional motion plan.`, base);
          continue;
        }
        if (!redoMotionAndEnds && hasFirstLastFrameEndImage(segment) && !segment.flf_end_frame_stale) {
          progress.set(`PASS 2/4 — MOTION PLANS ${index + 1}/${scenes.length}: ${sceneLabel}\nA manually loaded or imported end image already exists, so it remains endpoint truth. Skipping the start-only provisional plan; Pass 4 will create the final motion prompt directly from the actual pair.`, base);
          continue;
        }
        progress.set(`PASS 2/4 — MOTION PLANS ${index + 1}/${scenes.length}: ${sceneLabel}\nGemma is viewing the completed start image and defining the shot's motion and final visual moment.`, base);
        await generateIndependentFLFMotionPlanForSegment(segment, progress, base, `Independent motion ${index + 1}/${scenes.length}: ${sceneLabel}`);
        await autoSaveSessionQuiet(`Independent FLF motion plan ${sceneLabel}`);
      }
      progress.set("PASS 2/4 COMPLETE — Every target scene has either a provisional motion plan or an existing user-supplied endpoint.\n\nReturning to the first target scene to create missing end images...", 51);
      await runClearMemoryWorkflowQuiet(progress, "independent FLF motion-plan pass", 52);

      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 53 + Math.floor((index / Math.max(1, scenes.length)) * 27);
        if (!redoMotionAndEnds && hasFirstLastFrameEndImage(segment) && !segment.flf_end_frame_stale) {
          progress.set(`PASS 3/4 — END FRAMES ${index + 1}/${scenes.length}: ${sceneLabel}\nKeeping this scene's existing independent end image.`, base);
          continue;
        }
        progress.set(`PASS 3/4 — END FRAMES ${index + 1}/${scenes.length}: ${sceneLabel}\nCreating the end image from this scene's own start, motion plan, mapped descriptions, and supported references.`, base);
        await createEndFrameForSegment(
          segment,
          imageMode,
          progress,
          base,
          Math.max(3, Math.floor(24 / Math.max(1, scenes.length))),
          `Independent end ${index + 1}/${scenes.length}: ${sceneLabel}`,
        );
        await autoSaveSessionQuiet(`Independent FLF end image ${sceneLabel}`);
      }
      progress.set("PASS 3/4 COMPLETE — Every target scene now owns a start/end pair.\n\nClearing image memory before final two-image prompt generation...", 81);
      await runClearMemoryWorkflowQuiet(progress, "independent FLF end-frame pass", 82);

      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 83 + Math.floor((index / Math.max(1, scenes.length)) * 14);
        if (!redoMotionAndEnds && segment.flf_final_prompt_ready && String(segment.i2v_prompt || "").trim()) {
          progress.set(`PASS 4/4 — FINAL FLF PROMPTS ${index + 1}/${scenes.length}: ${sceneLabel}\nKeeping the saved final prompt.`, base);
          continue;
        }
        progress.set(`PASS 4/4 — FINAL FLF PROMPTS ${index + 1}/${scenes.length}: ${sceneLabel}\nGemma is inspecting this scene's actual start and end images and adapting the motion plan to those exact endpoints.`, base);
        await generateFinalIndependentFLFPromptForSegment(segment, progress, base, `Final independent FLF ${index + 1}/${scenes.length}: ${sceneLabel}`);
        await autoSaveSessionQuiet(`Independent FLF final prompt ${sceneLabel}`);
      }
      await runClearMemoryWorkflowQuiet(progress, "independent FLF final-prompt pass", 98);
      await autoSaveSessionQuiet("Build Independent Start + End Pairs complete");
      progress.set(`Build Independent Start + End Pairs complete for ${scenes.length} scene${scenes.length === 1 ? "" : "s"}.\n\n✓ All start images\n✓ All provisional motion plans\n✓ All independent end images\n✓ All final two-image FLF prompts\n\nNo end image was reused as another scene's start. Use Render All when you are ready to create the videos.`, 100);
      toast("Independent start/end pairs are ready for Render All.");
    } catch (error) {
      const message = String(error?.message || error);
      const stopped = /stopped by user/i.test(message);
      progress.set(`${stopped ? "Stopped" : "Error"}:\n${message}\n\nCompleted stages were autosaved. Run Resume Independent Pairs to continue missing work.`, 100);
      toast(message, !stopped);
      if (options.throwOnError) throw error;
    } finally {
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      state.batchCancelled = false;
      syncI2VVideoSettingsPanel();
      syncRTVSceneImageAnchorPanel();
      syncInspector();
      render();
    }
  }

  async function generateFLFStartImageForSegment(segment, imageMode, progress, percentBase, percentSpan, label, options = {}) {
    const savedPrompts = {
      t2i_prompt: segment.t2i_prompt || "",
      flux_prompt: segment.flux_prompt || "",
      nb_prompt: segment.nb_prompt || "",
      flow_gpt_prompt: segment.flow_gpt_prompt || "",
      enhance_prompt: segment.enhance_prompt || "",
    };
    state.activeId = segment.id;
    syncInspector();
    render();
    if (options.useExistingPrompt === true) {
      const existingPrompt = savedImagePromptForMode(segment, imageMode);
      if (!existingPrompt) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: saved image prompt is missing.`);
      syncSegmentT2IPrompt(segment, existingPrompt);
      if (imageMode === "flow_gpt") syncSegmentFlowGptPrompt(segment, existingPrompt);
      progress?.set(`${label}: using the existing saved image prompt; Gemma is not needed.`, percentBase);
    } else {
      progress?.set(`${label}: creating the opening-image prompt...`, percentBase);
      if (imageMode === "flux_klein") {
        await generateFluxKleinPromptForSegment(segment, progress, percentBase + percentSpan * 0.12, `${label}: Gemma Flux/Klein`, { unloadAfter: true });
      } else if (imageMode === "nano_banana" || imageMode === "flow_gpt") {
        await generateNBPromptForSegment(segment, progress, percentBase + percentSpan * 0.12, `${label}: Gemma ${imageMode === "flow_gpt" ? "Browser AI" : "NanoBanana"}`, { imageMode, unloadAfter: true, flfImageTarget: "start" });
        if (imageMode === "flow_gpt") syncSegmentFlowGptPrompt(segment, segment.nb_prompt || segment.t2i_prompt || "");
      } else {
        await generateT2IPromptForSegment(segment, progress, percentBase + percentSpan * 0.12, `${label}: Gemma start frame`, { unloadAfter: true, flfImageTarget: "start" });
      }
    }
    const sameLocationDiversity = flfSameLocationCameraDiversityDirection(segment, "start");
    if (sameLocationDiversity) {
      const generatedPrompt = savedImagePromptForMode(segment, imageMode);
      if (generatedPrompt && !generatedPrompt.includes("REQUIRED SAME-LOCATION CAMERA CHANGE:")) {
        const strengthenedPrompt = `${generatedPrompt}\n\n${sameLocationDiversity}`.trim();
        if (imageMode === "flow_gpt") syncSegmentFlowGptPrompt(segment, strengthenedPrompt);
        else syncSegmentT2IPrompt(segment, strengthenedPrompt);
      }
    }
    const previousStartIngredient = imageMode === "flow_gpt" && sameLocationDiversity
      ? previousSceneStartImageIngredient(segment)
      : null;
    progress?.set(`${label}: generating the opening image with ${imageModeDisplayLabel(imageMode, true)}...`, percentBase + percentSpan * 0.32);
    await createImageForSegmentInCurrentMode(
      segment,
      imageMode,
      progress,
      percentBase + percentSpan * 0.38,
      percentSpan * 0.55,
      label,
      {
        bypassImageToImage: true,
        includePreviousSceneImage: Boolean(previousStartIngredient),
        previousSceneImageIngredient: previousStartIngredient,
        previousSceneImagePurpose: previousStartIngredient ? "same_location_composition_diversity" : "",
      }
    );
    const generated = segmentImageSource(segment);
    if (!generated?.path && !generated?.data) throw new Error(`${sceneDisplayName(segment, segmentIndexInfo(segment).index)}: opening image did not return a saved image.`);
    if (options.preserveSavedPrompts !== false) Object.assign(segment, savedPrompts);
    if (segment.id === activeSegment()?.id) syncInspector();
    return generated;
  }

  async function createFLFImageChain(options = {}) {
    updateActiveFromInputs();
    const imageMode = options.imageMode || state.imageModelMode || "zimage";
    const redoImages = options.redoImages === true;
    const useExistingPrompts = options.useExistingPrompts === true;
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const scenes = batchTargetItems(sceneScope, { baseOnly: true }).slice().sort((a, b) => Number(a.segment?.start || 0) - Number(b.segment?.start || 0));
    const progress = createProgressWindow("Create FLF Image Chain");
    const originalRenderChainSource = state.i2vVideoSettings?.flf_render_chain_start_source || "rendered_frame";
    const missing = [];
    if (currentVideoMode() !== "flf") missing.push("Video mode must be First Last Frame.");
    if (!scenes.length) missing.push(batchEmptyMessage(sceneScope));
    if (!String(projectInput.value || state.projectFolder || "").trim()) missing.push("Project folder is missing.");
    scenes.forEach(({ segment, index }) => {
      if (!String(segment.flf_end_state || segment.story_beat || segment.lyric_text || segment.notes || "").trim()) {
        missing.push(`${sceneDisplayName(segment, index)}: endpoint beat, story beat, lyrics, or scene notes are missing.`);
      }
    });
    if (missing.length) {
      progress.set(`Create FLF Image Chain cannot start yet:\n\n${missing.map((item) => `- ${item}`).join("\n")}`, 100);
      toast("FLF image chain needs endpoint planning and a saved project first.", true);
      return;
    }
    try {
      state.batchCancelled = false;
      zImageAllButton.disabled = true;
      zImageAllButton.textContent = "FLF Chain...";
      state.i2vVideoSettings = {
        ...(state.i2vVideoSettings || defaultI2VVideoSettings()),
        flf_chain_previous_end_frame: true,
        // Image generation must use the previous generated destination. The
        // user's rendered-frame choice is restored before video rendering.
        flf_render_chain_start_source: "previous_image",
      };
      state.segments.forEach((segment) => {
        if (segment.use_scene_i2v_video_settings && segment.i2v_video_settings) {
          segment.i2v_video_settings.flf_chain_previous_end_frame = true;
        }
      });
      progress.set(`Preparing ${scenes.length} FLF scene${scenes.length === 1 ? "" : "s"} (${batchScopeLabel(sceneScope)})...\nImage-chain start source: previous generated destination.\nVideo render-chain preference will remain: ${originalRenderChainSource === "previous_image" ? "previous assigned image" : "extracted rendered frame"}.`, 3);
      await saveSessionForSceneVideo();

      const first = scenes[0];
      const firstHasPreviousChainScene = Boolean(previousAutoChainSourceSegment(first.segment));
      if (!firstLastFrameStartImageSource(first.segment) || (redoImages && !firstHasPreviousChainScene)) {
        await generateFLFStartImageForSegment(first.segment, imageMode, progress, 5, 20, `${sceneDisplayName(first.segment, first.index)} start frame`, { useExistingPrompt: useExistingPrompts });
        await autoSaveSessionQuiet(`FLF image chain opening image ${sceneDisplayName(first.segment, first.index)}`);
      }

      let chainedDestination = null;
      for (let index = 0; index < scenes.length; index += 1) {
        assertBatchNotStopped();
        const { segment, index: sceneIndex } = scenes[index];
        const sceneLabel = sceneDisplayName(segment, sceneIndex);
        const base = 25 + Math.floor((index / Math.max(1, scenes.length)) * 68);
        const span = Math.max(4, Math.floor(62 / Math.max(1, scenes.length)));
        if (!redoImages && hasFirstLastFrameEndImage(segment)) {
          chainedDestination = firstLastFrameEndImageSource(segment);
          progress.set(`FLF image chain ${index + 1}/${scenes.length}: ${sceneLabel}\nKeeping existing destination image.`, base);
          continue;
        }
        const chainedStart = index > 0 && (chainedDestination?.path || chainedDestination?.data)
          ? chainedDestination
          : firstLastFrameStartImageSource(segment);
        if (!chainedStart?.path && !chainedStart?.data) {
          throw new Error(`${sceneLabel}: chained start image could not be resolved from the previous scene's destination.`);
        }
        progress.set(`FLF image chain ${index + 1}/${scenes.length}: ${sceneLabel}\nGenerating only this scene's destination image.`, base);
        chainedDestination = await createEndFrameForSegment(
          segment,
          imageMode,
          progress,
          base + 1,
          span,
          `FLF Chain ${index + 1}/${scenes.length}: ${sceneLabel}`,
          { useExistingPrompt: useExistingPrompts, firstFrame: chainedStart }
        );
        if (!chainedDestination?.path && !chainedDestination?.data) {
          throw new Error(`${sceneLabel}: destination image generation finished without saving an FLF end image.`);
        }
        await autoSaveSessionQuiet(`FLF image chain destination ${sceneLabel}`);
      }
      state.i2vVideoSettings.flf_render_chain_start_source = originalRenderChainSource;
      await autoSaveSessionQuiet("FLF image chain complete");
      progress.set(`FLF image chain complete for ${scenes.length} scene${scenes.length === 1 ? "" : "s"}.\n\n${redoImages ? "Regenerated" : "Created or kept"} one opening image for the first target scene and one destination image per scene. Saved image prompts were preserved. No unused later start images were generated.\n\nVideo render-chain preference restored to: ${originalRenderChainSource === "previous_image" ? "previous assigned image" : "extracted rendered frame"}.\n\nThis window will remain open so you can verify the result.`, 100);
      toast("FLF image chain complete.");
    } catch (error) {
      const message = String(error?.message || error);
      const stopped = /stopped by user/i.test(message);
      progress.set(`${stopped ? "Stopped" : "Error"}:\n${message}`, 100);
      toast(message, !stopped);
      if (options.throwOnError) throw error;
    } finally {
      state.i2vVideoSettings = {
        ...(state.i2vVideoSettings || defaultI2VVideoSettings()),
        flf_chain_previous_end_frame: true,
        flf_render_chain_start_source: originalRenderChainSource,
      };
      state.segments.forEach((segment) => {
        if (segment.use_scene_i2v_video_settings && segment.i2v_video_settings) {
          segment.i2v_video_settings.flf_chain_previous_end_frame = true;
        }
      });
      await autoSaveSessionQuiet("FLF image chain render-source preference restored").catch(() => null);
      zImageAllButton.disabled = false;
      zImageAllButton.textContent = "Image All";
      state.batchCancelled = false;
      syncInspector();
      render();
    }
  }

  async function ensureFLFEndpointBeatsForBuild(progress, sceneScope = "all") {
    const targets = batchTargetItems(sceneScope, { baseOnly: true });
    const allScenes = storyboardScenePayload();
    const missingTargets = targets.filter(({ segment }) =>
      [segment.flf_start_state, segment.flf_transformation, segment.flf_end_state, segment.flf_carry_forward]
        .some((value) => !String(value || "").trim())
    );
    if (!missingTargets.length) return 0;
    const storyLayer = normalizeBuilderStoryLayer(state.builderStoryLayer);
    for (let index = 0; index < missingTargets.length; index += 1) {
      assertBatchNotStopped();
      const { segment, index: sceneIndex } = missingTargets[index];
      const scene = allScenes.find((item) => item.id === segment.id)
        || allScenes.find((item) => Number(item.scene_number) === Number(sceneIndex + 1));
      if (!scene) throw new Error(`${sceneDisplayName(segment, sceneIndex)}: storyboard scene payload is missing.`);
      const previous = sceneIndex > 0 ? state.segments[sceneIndex - 1] : null;
      const next = sceneIndex >= 0 && sceneIndex < state.segments.length - 1 ? state.segments[sceneIndex + 1] : null;
      const pct = 3 + Math.floor(((index + 1) / Math.max(1, missingTargets.length)) * 12);
      progress?.set(`Stage 1/4: creating FLF endpoint beat ${index + 1}/${missingTargets.length}\n${sceneDisplayName(segment, sceneIndex)}`, pct);
      const storyboardState = wizardStoryboardState(allScenes, { promptMode: "video", imageMode: state.imageModelMode || "zimage" });
      const data = await postJson("/vrgdg/storyboard/scene_story_beat", {
        ...textGemmaRunnerPayload(),
        model_file: i2vTextGemmaModelSelect.value || t2iTextGemmaModelSelect.value || "",
        story_layer: storyLayer,
        storyboard_payload: storyboardGptPayload(storyboardState, [scene]),
        previous_beat: String(previous?.story_beat || ""),
        previous_lyrics: String(previous?.lyric_text || ""),
        previous_end_state: String(previous?.flf_end_state || ""),
        previous_carry_forward: String(previous?.flf_carry_forward || ""),
        current_lyrics: String(segment.lyric_text || ""),
        next_lyrics: String(next?.lyric_text || ""),
        flf_mode: true,
        unload_after: index === missingTargets.length - 1,
        max_new_tokens: 700,
        temperature: 0.35,
        top_p: 0.9,
      }, 240000);
      segment.story_beat = String(data.story_beat || segment.story_beat || "").trim();
      segment.flf_start_state = sceneIndex > 0 && previous?.flf_end_state
        ? String(previous.flf_end_state).trim()
        : String(data.flf_start_state || "").trim();
      segment.flf_transformation = String(data.flf_transformation || "").trim();
      segment.flf_end_state = String(data.flf_end_state || "").trim();
      segment.flf_carry_forward = String(data.flf_carry_forward || "").trim();
      if (!segment.flf_start_state || !segment.flf_transformation || !segment.flf_end_state || !segment.flf_carry_forward) {
        throw new Error(`${sceneDisplayName(segment, sceneIndex)}: Gemma returned an incomplete FLF endpoint beat.`);
      }
      if (next && segment.flf_end_state) next.flf_start_state = segment.flf_end_state;
      await autoSaveSessionQuiet(`Build Full FLF endpoint beat scene ${sceneIndex + 1}`);
    }
    return missingTargets.length;
  }

  async function buildFullFLFVideoPipeline(options = {}) {
    const sceneScope = normalizeBatchScope(options.sceneScope);
    const redoImages = options.redoImages === true;
    const redoVideos = options.redoVideos === true;
    let redoImagesThisAttempt = redoImages;
    let redoVideosThisAttempt = redoVideos;
    const maxAttempts = Math.max(1, Math.min(4, Number(options.maxAttempts || 3)));
    let attempt = 0;
    fullFLFBuildButton.disabled = true;
    fullFLFBuildButton.textContent = "Building FLF...";
    try {
      if (currentVideoMode() !== "flf") throw new Error("Build Full FLF Video requires First Last Frame video mode.");
      if (!(await ensureAudioOrOfferSilentTimeline({ sceneScope }))) return;
      while (attempt < maxAttempts) {
        attempt += 1;
        const progress = createProgressWindow(`Build Full FLF Video${attempt > 1 ? ` — retry ${attempt}/${maxAttempts}` : ""}`);
        try {
          state.batchCancelled = false;
          progress.set("Stage 1/4: checking storyboard endpoint beats...", 2);
          await ensureFLFEndpointBeatsForBuild(progress, sceneScope);
          progress.set("Stage 2/4: creating the FLF image chain...", 18);
          progress.close(250);
          await createFLFImageChain({
            imageMode: state.imageModelMode || "zimage",
            sceneScope,
            redoImages: redoImagesThisAttempt,
            throwOnError: true,
          });
          const renderProgress = createProgressWindow("Build Full FLF Video — video prompts, renders, and stitch");
          renderProgress.set("Stage 3/4: Render All will create each missing vision prompt immediately before its scene render.\nStage 4/4: extracting each final frame and stitching the completed clips.", 55);
          renderProgress.close(1800);
          await renderAllScenes({
            sceneScope,
            forceVideos: redoVideosThisAttempt,
            randomizeVideoSeed: redoVideosThisAttempt,
            skipFinalStitch: sceneScope === "selected",
            throwOnError: true,
          });
          toast("Build Full FLF Video complete.");
          return;
        } catch (error) {
          const message = String(error?.message || error);
          progress.set(`Build Full FLF Video failed on attempt ${attempt}/${maxAttempts}:\n${message}`, 100);
          if (state.batchCancelled || attempt >= maxAttempts) throw error;
          await runClearMemoryWorkflowQuiet(progress, `Build Full FLF retry ${attempt}/${maxAttempts}`, 100).catch(() => null);
          redoImagesThisAttempt = false;
          redoVideosThisAttempt = false;
          progress.set(`Retrying in safe-resume mode. Completed beats, images, prompts, and videos will be kept.`, 100);
          progress.close(2500);
        }
      }
    } catch (error) {
      toast(`Build Full FLF Video stopped:\n${String(error?.message || error)}`, true);
    } finally {
      fullFLFBuildButton.disabled = false;
      fullFLFBuildButton.textContent = "Build Full FLF Video";
      state.batchCancelled = false;
    }
  }

  async function buildFullVideoPipeline(options = {}) {
    let buildMode = options.buildMode || "resume_missing";
    const maxAutoRetries = Math.max(0, Math.min(5, Number(options.maxAutoRetries ?? 3)));
    const sceneScope = normalizeBatchScope(options.sceneScope);
    let attempt = 0;
    let progress = null;
    try {
      fullBuildButton.disabled = true;
      fullBuildButton.textContent = "Building...";
      renderAllButton.disabled = true;
      zImageAllButton.disabled = true;
      state.batchCancelled = false;
      if (!(await ensureAudioOrOfferSilentTimeline({ sceneScope }))) return;
      while (true) {
        attempt += 1;
        progress = createProgressWindow(attempt > 1 ? `Build Full Video retry ${attempt}/${maxAutoRetries + 1}` : "Build Full Video");
        try {
          const videoMode = currentVideoMode();
          if (videoMode === "t2v" || videoMode === "rtv") {
            progress.set(`Stage 1/3: ${videoModeDisplayLabel(videoMode)} mode skips image generation.`, 20);
          } else {
            const imageStage = (state.imageModelMode || "") === "flux_klein" ? "Flux/Klein image pass" : state.imageModelMode === "nano_banana" ? "NanoBanana image pass" : state.imageModelMode === "flow_gpt" ? "Flow/GPT image pass" : state.imageModelMode === "ernie_image" ? "Ernie image pass" : state.imageModelMode === "krea2_2pass" ? "Krea 2 image pass" : "Z-Image pass";
            progress.set(`Stage 1/3: ${imageStage}...`, 5);
            const imageMode = state.imageModelMode || "zimage";
            const imageRunMode = buildMode === "fresh_rebuild" ? "redo_prompts_images" : "resume_missing";
            if (img2imgContinuityEnabled() && videoMode === "i2v") {
              if (!imageModeSupportsImg2ImgContinuity(imageMode)) {
                throw new Error(`Img2Img continuity is not available for ${imageModeImg2ImgContinuityLabel(imageMode)} yet. Choose ZImage, Ernie, or Krea 2 for the image model, or turn continuity Off.`);
              }
              const firstTarget = batchTargetItems(sceneScope)[0] || null;
              const firstSegment = firstTarget?.segment || null;
              const firstHasPreviousVideo = Boolean(firstSegment && canImg2ImgContinuityFromPreviousRenderedScene(firstSegment));
              const needsFirstImage = firstSegment && !firstHasPreviousVideo && (buildMode === "fresh_rebuild" || !segmentImageSource(firstSegment));
              if (needsFirstImage) {
                const firstIndex = firstTarget.index;
                const firstLabel = sceneDisplayName(firstSegment, firstIndex);
                progress.set(`Stage 1/3: creating starting image for ${firstLabel}. Later images will be created from previous final frames during Render All...`, 8);
                if (!String(firstSegment.t2i_prompt || "").trim()) {
                  await runGemmaImagePromptPassWithRetry(
                    firstSegment,
                    progress,
                    10,
                    `Starting image prompt: ${firstLabel}`,
                    generateT2IPromptForSegment,
                    { unloadAfter: false },
                  );
                  await runClearMemoryWorkflowQuiet(progress, "starting image prompt", 18);
                }
                state.activeId = firstSegment.id;
                syncInspector();
                render();
                await createImageForSegmentInCurrentMode(firstSegment, imageMode, progress, 20, 18, `Starting image ${firstLabel}`);
                await autoSaveSessionQuiet("Img2Img continuity starting image");
              } else {
                progress.set("Stage 1/3: Img2Img continuity will create scene images during Render All.", 20);
              }
            } else if (imageMode === "flux_klein") {
              await fluxKleinAllScenes({ throwOnError: true, imageRunMode, sceneScope });
            } else if (imageMode === "nano_banana") {
              await nbImageAllScenes({ throwOnError: true, imageRunMode, sceneScope });
            } else if (imageMode === "flow_gpt") {
              await flowGptImageAllScenes({ throwOnError: true, imageRunMode, sceneScope });
            } else if (imageMode === "ernie_image") {
              await ernieImageAllScenes({ throwOnError: true, imageRunMode, sceneScope });
            } else if (imageMode === "krea2_2pass") {
              await krea2TwoPassImageAllScenes({ throwOnError: true, imageRunMode, sceneScope });
            } else {
              await zImageAllScenes({ throwOnError: true, imageRunMode, sceneScope });
            }
          }
          assertBatchNotStopped();
          const activeVideoMode = currentVideoMode();
          if (buildMode === "redo_videos") {
            progress.set("Stage 2/3: keeping existing video prompts...", 38);
          } else {
            const videoPromptStage = activeVideoMode === "t2v"
              ? "creating T2V prompts"
              : activeVideoMode === "rtv"
                ? "creating Reference-to-Video prompts"
                : activeVideoMode === "ingredients"
                  ? "creating Ingredients-to-Video prompts"
                : "creating I2V prompts";
            progress.set(`Stage 2/3: ${videoPromptStage}...`, 38);
            await i2vAllScenes({
              throwOnError: true,
              i2vRunMode: buildMode === "fresh_rebuild" || buildMode === "redo_i2v_prompts_videos" ? "redo_prompts" : "resume_missing",
              sceneScope,
            });
          }
          assertBatchNotStopped();
          progress.set("Stage 3/3: rendering and stitching scene videos...", 68);
          progress.close(300);
          progress = null;
          const shouldRandomizeVideoSeed = buildMode === "fresh_rebuild" || buildMode === "redo_i2v_prompts_videos" || buildMode === "redo_videos";
          await renderAllScenes({
            sceneScope,
            forceVideos: buildMode === "fresh_rebuild" || buildMode === "redo_i2v_prompts_videos" || buildMode === "redo_videos",
            randomizeVideoSeed: shouldRandomizeVideoSeed && options.videoSeedMode !== "keep",
            skipFinalStitch: sceneScope === "selected",
          });
          toast(attempt > 1 ? `Build Full Video complete after ${attempt} attempts.` : "Build Full Video complete.");
          break;
        } catch (error) {
          const errorMessage = String(error?.message || error);
          const canRetry = !state.batchCancelled && attempt <= maxAutoRetries && isRecoverableBuildGemmaError(error);
          if (!canRetry) throw error;
          progress?.set(`Build Full Video hit a recoverable Gemma error on attempt ${attempt}/${maxAutoRetries + 1}:\n${errorMessage}`, 100);
          await recoverFromBuildGemmaError(error, attempt, maxAutoRetries, progress);
          progress = null;
          buildMode = "resume_missing";
        }
      }
    } catch (error) {
      const errorMessage = String(error?.message || error);
      progress?.set(`Build Full Video stopped:\n${errorMessage}`, 100);
      toast(`Full video build stopped:\n${errorMessage}`, true);
    } finally {
      fullBuildButton.disabled = false;
      fullBuildButton.textContent = "Build Full Video";
      renderAllButton.disabled = false;
      zImageAllButton.disabled = false;
      state.batchCancelled = false;
    }
  }

  async function createNBImageForSegmentWithRetry(segment, progress, percentBase, percentSpan, label, options = {}) {
    const maxRetries = Math.max(1, Number(options.maxRetries || 10));
    let lastError = null;
    for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
      assertBatchNotStopped();
      try {
        const attemptLabel = attempt === 1 ? label : `${label} retry ${attempt}/${maxRetries}`;
        progress?.set(`${attemptLabel}: creating NanoBanana image...`, percentBase);
        return await createNBImageForSegment(segment, progress, percentBase, percentSpan, attemptLabel);
      } catch (error) {
        lastError = error;
        const message = String(error?.message || error || "Unknown error");
        if (attempt >= maxRetries) break;
        progress?.set(
          `${label}: NanoBanana failed on attempt ${attempt}/${maxRetries}.\n${message}\n\nClearing memory before retry ${attempt + 1}/${maxRetries}...`,
          Math.min(99, percentBase + percentSpan),
        );
        await cancelComfyExecutionAndWaitIdle((status) => {
          progress?.set(`${label}: cancelling failed job before retry...\n${status}`, Math.min(99, percentBase + percentSpan));
        }, { shouldCancel: () => state.batchCancelled });
        try {
          await runClearMemoryWorkflowQuiet(progress, `${label} failed attempt ${attempt}/${maxRetries}`, Math.min(99, percentBase + percentSpan));
        } catch (cleanupError) {
          console.warn("[VRGDG Music Builder] Cleanup before NanoBanana retry failed:", cleanupError);
          progress?.set(
            `${label}: cleanup failed before retry ${attempt + 1}/${maxRetries}, retrying anyway.\n${String(cleanupError?.message || cleanupError)}`,
            Math.min(99, percentBase + percentSpan),
          );
        }
        await new Promise((resolve) => setTimeout(resolve, 5000));
      }
    }
    progress?.set(`${label}: failed after ${maxRetries} NanoBanana attempts. Clearing memory before stopping...`, 100);
    try {
      await runClearMemoryWorkflowQuiet(progress, `${label} failed after ${maxRetries} attempts`, 100);
    } catch (cleanupError) {
      console.warn("[VRGDG Music Builder] Final cleanup after NanoBanana retry failure failed:", cleanupError);
    }
    throw new Error(`${label}: NanoBanana failed after ${maxRetries} attempts.\nLast error:\n${String(lastError?.message || lastError || "Unknown error")}`);
  }

  return {
    buildFullFLFVideoPipeline, buildFullVideoPipeline, buildIndependentFLFPairs, createEndFramesAllScenes,
    createFLFImageChain, createNBImageForSegmentWithRetry, ernieImageAllScenes, flowGptImageAllScenes,
    fluxKleinAllScenes, krea2TwoPassImageAllScenes, nbImageAllScenes, renderAllScenes, zEnhanceAllScenes,
    zEnhanceBatchTargets, zImageAllScenes,
  };
}
