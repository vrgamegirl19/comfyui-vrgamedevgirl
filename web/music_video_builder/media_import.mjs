import { clampOverlayTiming } from "../VRGDG_OverlayTrack.js";
import { postJson } from "./comfy_api.mjs";
import { makeButton, toast } from "./controls.mjs";
import { formatTime } from "./format.mjs";
import {
  cloneI2VVideoSettings,
  defaultErnieImageSettings,
  defaultI2VVideoSettings,
  defaultKrea2TwoPassSettings,
} from "./model_settings.mjs";
import { newSegment, sortSegments } from "./segments.mjs";
import { hasLockedVideo } from "./selection_preview.mjs";

export function imageFileFromDrop(event) {
  const files = Array.from(event.dataTransfer?.files || []);
  return files.find((file) => /^image\//i.test(file.type) || /\.(png|jpe?g|webp|gif|bmp|tiff?|avif)$/i.test(file.name || ""));
}

function dropHasFiles(event) {
  return Array.from(event.dataTransfer?.types || []).some((type) => String(type || "").toLowerCase() === "files")
    || Boolean(event.dataTransfer?.files?.length);
}

export function installFileDropNavigationGuard(element) {
  const isDropZoneEvent = (event) => Boolean(event.target?.closest?.("[data-vrgdg-file-drop-zone='true']"));
  element.addEventListener("dragover", (event) => {
    if (!dropHasFiles(event)) return;
    event.preventDefault();
    if (!isDropZoneEvent(event)) event.dataTransfer.dropEffect = "none";
  }, true);
  element.addEventListener("drop", (event) => {
    if (!dropHasFiles(event) || isDropZoneEvent(event)) return;
    event.preventDefault();
    event.stopPropagation();
    toast("Drop images onto a scene block or scene row to attach them.", true);
  }, true);
}

export function audioFileFromDrop(event) {
  const files = Array.from(event.dataTransfer?.files || []);
  return files.find((file) => /^audio\//i.test(file.type) || /\.(wav|mp3|flac|m4a|ogg)$/i.test(file.name || ""));
}

function isTimelineImageFile(file) {
  const name = String(file?.name || "").trim();
  return Boolean(file) && (/^image\//i.test(file.type || "") || /\.(png|jpe?g|webp)$/i.test(name));
}

function numericImageNameParts(file) {
  const name = String(file?.name || file?.webkitRelativePath || "").trim();
  return Array.from(name.matchAll(/\d+/g)).map((match) => Number.parseInt(match[0], 10)).filter(Number.isFinite);
}

function compareNumericImageFiles(a, b) {
  const aParts = numericImageNameParts(a);
  const bParts = numericImageNameParts(b);
  if (aParts.length || bParts.length) {
    if (!aParts.length) return 1;
    if (!bParts.length) return -1;
    const length = Math.max(aParts.length, bParts.length);
    for (let index = 0; index < length; index += 1) {
      const av = aParts[index] ?? -1;
      const bv = bParts[index] ?? -1;
      if (av !== bv) return av - bv;
    }
  }
  return String(a.webkitRelativePath || a.name || "").localeCompare(String(b.webkitRelativePath || b.name || ""), undefined, { numeric: true, sensitivity: "base" });
}

export function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error(`Failed to read image file: ${file?.name || "image"}`));
    reader.readAsDataURL(file);
  });
}

function sceneIdFromDrop(event) {
  return event.dataTransfer?.getData("application/x-vrgdg-segment-id") || "";
}

export function createMediaImport({
  activeKrea2TwoPassSettings, activeSegment, activeZImageSettings, addSceneImageHistoryPath, audio,
  audioInput, autoSaveSessionQuiet, createProgressWindow, currentVideoMode, ensureSegmentRuntimeFields,
  ernieI2ISlider, ernieI2IStartStep, ernieRefImagePanel, ernieUseVisionReference, flfChainingEnabled,
  globalAudioModeSelect, handleSegmentPick, importImageFolderButton, krea2TwoPassCreativity,
  krea2TwoPassCreativityInput, krea2TwoPassRefImagePanel, krea2TwoPassUseVisionReference,
  loadFirstLastFrameEndFile, normalizeSegments, openTimelineSceneCard, previousAutoChainSourceSegment,
  projectInput, pushHistory, refImageInput, refImagePanel, render, renderFluxGlobalIngredientList,
  renderFluxIngredientList, renderList, renderNBIngredientList, sceneDisplayName, sceneSlotNumber,
  segmentIndexInfo, segmentTrack, setActiveSegment, silentAudioDurationInput, snapTimeToBeat, state,
  syncErnieImagePanel, syncFluxKleinPanel, syncGlobalAudioModeControls, syncInspector, syncKrea2TwoPassPanel,
  syncPreview, syncZImageSettingsPanel, t2vRefImagePanel, timelineViewport, useT2VVisionReference,
  useVisionReference, zI2ISlider, zI2IStartStep,
}) {
  let activeSegmentDragCleanup = null;
  let lastTimelineSceneClickTime = 0;
  let lastTimelineSceneClickId = "";

  function makeDragHandle(element, segment, mode) {
    element.style.touchAction = "none";
    element.style.userSelect = "none";
    element.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.isPrimary === false) return;
      const isOverlay = segmentTrack(segment) === "overlay";
      if (isOverlay && segment.overlay_locked !== false) {
        toast("Unlock this overlay clip before moving or trimming it.", true);
        return;
      }
      if (state.timingFrozen && !isOverlay) {
        toast("Timing is frozen. Unfreeze timing before editing segment lengths.", true);
        return;
      }
      if (!isOverlay && hasLockedVideo(segment)) {
        toast("This scene already has a generated video, so its timing is locked.", true);
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      activeSegmentDragCleanup?.();
      const pointerId = event.pointerId;
      try {
        // The scene blocks are recreated during render(), so capture on the
        // persistent viewport instead of the handle that is about to vanish.
        timelineViewport.setPointerCapture?.(pointerId);
      } catch {
        // Window listeners below still keep the drag usable without capture.
      }
      const startX = event.clientX;
      const start = segment.start;
      const end = segment.end;
      let historySaved = false;
      let dragStarted = false;
      let dragging = true;
      const cleanup = () => {
        if (!dragging) return;
        dragging = false;
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", finish);
        window.removeEventListener("pointercancel", finish);
        window.removeEventListener("blur", cleanup);
        timelineViewport.removeEventListener("lostpointercapture", cleanup);
        try {
          if (timelineViewport.hasPointerCapture?.(pointerId)) timelineViewport.releasePointerCapture(pointerId);
        } catch {
          // Capture may already have been released by the browser.
        }
        if (activeSegmentDragCleanup === cleanup) activeSegmentDragCleanup = null;
      };
      const move = (moveEvent) => {
        if (!dragging || moveEvent.pointerId !== pointerId) return;
        // If pointerup was lost, a mouse move with no left button held must
        // terminate the drag rather than continuing to edit scene timing.
        if (moveEvent.pointerType === "mouse" && (moveEvent.buttons & 1) === 0) {
          cleanup();
          return;
        }
        const pixelDelta = moveEvent.clientX - startX;
        if (!dragStarted && Math.abs(pixelDelta) < 3) return;
        dragStarted = true;
        moveEvent.preventDefault();
        if (!historySaved) {
          pushHistory();
          historySaved = true;
        }
        const delta = pixelDelta / state.pxPerSecond;
        if (mode === "start") {
          segment.start = Math.max(0, Math.min(end - 0.1, snapTimeToBeat(start + delta)));
        } else if (mode === "end") {
          segment.end = Math.max(start + 0.1, snapTimeToBeat(end + delta));
          state.duration = Math.max(Number(state.duration || 0), Number(segment.end || 0));
        } else {
          const duration = end - start;
          segment.start = Math.max(0, Math.min((state.duration || 9999) - duration, snapTimeToBeat(start + delta)));
          segment.end = segment.start + duration;
        }
        if (isOverlay) {
          const safe = clampOverlayTiming(segment, state.overlaySegments, segment.start, segment.end);
          segment.start = safe.start;
          segment.end = safe.end;
          sortSegments(state.overlaySegments);
        }
        else normalizeSegments(segment);
        syncInspector();
        render();
      };
      const finish = (finishEvent) => {
        if (finishEvent?.pointerId != null && finishEvent.pointerId !== pointerId) return;
        const shouldSelectScene = finishEvent?.type === "pointerup" && mode === "move" && !dragStarted;
        cleanup();
        if (shouldSelectScene) {
          const now = Date.now();
          if (now - lastTimelineSceneClickTime < 380 && lastTimelineSceneClickId === segment.id && !isOverlay) {
            lastTimelineSceneClickTime = 0;
            lastTimelineSceneClickId = "";
            openTimelineSceneCard(segment, finishEvent);
          } else {
            lastTimelineSceneClickTime = now;
            lastTimelineSceneClickId = segment.id;
            handleSegmentPick(segment, finishEvent);
          }
        }
      };
      activeSegmentDragCleanup = cleanup;
      window.addEventListener("pointermove", move, { passive: false });
      window.addEventListener("pointerup", finish);
      window.addEventListener("pointercancel", finish);
      window.addEventListener("blur", cleanup);
      timelineViewport.addEventListener("lostpointercapture", cleanup);
    });
  }

  async function applyCustomImageDataToSegment(segment, imageData, imageName = "custom_image", options = {}) {
    if (!segment || !imageData) return false;
    const projectFolder = projectInput.value || state.projectFolder;
    if (projectFolder) {
      try {
        const sceneNumber = sceneSlotNumber(segment);
        const saved = await postJson("/vrgdg/music_builder/archive_scene_image", {
          image_data: imageData,
          project_folder: projectFolder,
          scene_number: sceneNumber,
        });
        if (saved.saved_path) {
          addSceneImageHistoryPath(segment, saved.saved_path);
          segment.custom_image_path = saved.saved_path;
          segment.custom_image_data = "";
          segment.custom_image_name = imageName || "custom_image";
          segment.image = null;
        } else {
          segment.custom_image_data = imageData;
          segment.custom_image_name = imageName || "custom_image";
          segment.custom_image_path = "";
          segment.image = null;
        }
      } catch (error) {
        console.warn("[VRGDG Music Builder] Failed to archive custom image:", error);
        segment.custom_image_data = imageData;
        segment.custom_image_name = imageName || "custom_image";
        segment.custom_image_path = "";
        segment.image = null;
      }
    } else {
      segment.custom_image_data = imageData;
      segment.custom_image_name = imageName || "custom_image";
      segment.custom_image_path = "";
      segment.image = null;
    }
    segment.approved_image_path = "";
    segment.preview_mode = "image";
    if (options.activate !== false) setActiveSegment(segment);
    return true;
  }

  async function loadCustomImageFile(file, segment = activeSegment()) {
    if (!segment || !file) return;
    try {
      const imageData = await readFileAsDataUrl(file);
      pushHistory();
      await applyCustomImageDataToSegment(segment, imageData, file.name || "custom_image", { activate: true });
      syncPreview(segment);
      render();
      toast(`Loaded custom image for ${segment.label || "scene"}:\n${file.name}`);
      autoSaveSessionQuiet("custom image load");
    } catch (error) {
      toast(String(error?.message || error || "Failed to read the dropped image."), true);
    }
  }

  async function importTimelineImagesFromFolder(files) {
    const hasProjectAudio = Boolean(String(audioInput.value || state.audioPath || "").trim());
    const projectAudioDuration = hasProjectAudio
      ? Math.max(0, Number(audio.duration || state.duration || 0))
      : 0;
    let scenes = (Array.isArray(state.segments) ? state.segments : [])
      .filter((segment) => segment && typeof segment === "object")
      .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
    const images = Array.from(files || []).filter(isTimelineImageFile).sort(compareNumericImageFiles);
    if (!images.length) {
      toast("No PNG, JPG, JPEG, or WEBP images were found in that folder.", true);
      return;
    }
    const placeholderScene = scenes.length === 1 && ![
      scenes[0].custom_image_path,
      scenes[0].custom_image_data,
      scenes[0].approved_image_path,
      scenes[0].video_path,
      scenes[0].custom_audio_path,
      scenes[0].timeline_note,
      scenes[0].notes,
      scenes[0].i2v_notes,
      scenes[0].lyric_text,
      scenes[0].t2i_prompt,
      scenes[0].flux_prompt,
      scenes[0].nb_prompt,
      scenes[0].i2v_prompt,
    ].some((value) => String(value || "").trim());
    const createScenesFromImages = scenes.length === 0 || placeholderScene;
    if (currentVideoMode() === "flf") {
      const pairedStarts = new Map();
      const pairedEnds = new Map();
      for (const file of images) {
        const name = String(file.name || "");
        let match = name.match(/^scene_(\d+)_end\.(?:png|jpe?g|webp)$/i);
        if (match) {
          pairedEnds.set(Number(match[1]), file);
          continue;
        }
        match = name.match(/^scene_(\d+)\.(?:png|jpe?g|webp)$/i);
        if (match) pairedStarts.set(Number(match[1]), file);
      }
      const pairedNumbers = Array.from(new Set([...pairedStarts.keys(), ...pairedEnds.keys()])).sort((a, b) => a - b);
      const isStoryboardPairFolder = pairedNumbers.length > 0 && pairedNumbers.some((number) => pairedStarts.has(number) && pairedEnds.has(number));
      if (isStoryboardPairFolder) {
        const pairSceneCount = createScenesFromImages ? Math.max(...pairedNumbers) : scenes.length;
        const completePairs = pairedNumbers.filter((number) => pairedStarts.has(number) && pairedEnds.has(number) && number <= pairSceneCount).length;
        const missingStarts = Array.from({ length: pairSceneCount }, (_, index) => index + 1).filter((number) => !pairedStarts.has(number));
        const missingEnds = Array.from({ length: pairSceneCount }, (_, index) => index + 1).filter((number) => !pairedEnds.has(number));
        const confirmed = window.confirm(
          `Import independent Start Image Storyboard pairs for ${pairSceneCount} scene${pairSceneCount === 1 ? "" : "s"}?\n\n`
          + `${completePairs} complete start/end pair${completePairs === 1 ? "" : "s"} found.\n`
          + "scene_0001.png → Scene 1 start frame\n"
          + "scene_0001_end.png → Scene 1 end frame\n\n"
          + "Previous-end-to-next-start chaining will be turned OFF. Each scene will use its own start and end frame.\n"
          + (missingStarts.length ? `Missing start frames: ${missingStarts.slice(0, 12).join(", ")}${missingStarts.length > 12 ? "…" : ""}\n` : "")
          + (missingEnds.length ? `Missing end frames: ${missingEnds.slice(0, 12).join(", ")}${missingEnds.length > 12 ? "…" : ""}\n` : "")
          + "Existing frame images in matching scene slots will be replaced."
        );
        if (!confirmed) return;
        pushHistory();
        state.i2vVideoSettings = { ...(state.i2vVideoSettings || defaultI2VVideoSettings()), flf_chain_previous_end_frame: false };
        if (createScenesFromImages) {
          const duration = hasProjectAudio && projectAudioDuration > 0 ? projectAudioDuration : pairSceneCount * 4;
          const sceneDuration = duration / pairSceneCount;
          scenes = Array.from({ length: pairSceneCount }, (_, index) => newSegment(index * sceneDuration, index === pairSceneCount - 1 ? duration : (index + 1) * sceneDuration));
          state.segments = scenes; state.duration = duration; state.activeId = scenes[0]?.id || "";
          if (!hasProjectAudio) {
            globalAudioModeSelect.value = "silent";
            silentAudioDurationInput.value = String(Math.max(4, duration));
            syncGlobalAudioModeControls();
          }
        }
        const progress = createProgressWindow("Import Storyboard Start + End Frames");
        try {
          importImageFolderButton.disabled = true; importImageFolderButton.textContent = "Importing pairs...";
          let startCount = 0;
          let endCount = 0;
          for (let index = 0; index < scenes.length; index += 1) {
            const sceneNumber = index + 1;
            const segment = scenes[index];
            segment.flf_rendered_start_frame_path = "";
            segment.flf_rendered_start_frame_data = "";
            segment.flf_rendered_start_frame_name = "";
            segment.flf_rendered_source_video_path = "";
            segment.flf_motion_plan = "";
            segment.flf_end_frame_prompt = "";
            segment.flf_end_frame_stale = false;
            segment.flf_final_prompt_ready = false;
            if (segment.use_scene_i2v_video_settings && segment.i2v_video_settings) segment.i2v_video_settings.flf_chain_previous_end_frame = false;
            const startFile = pairedStarts.get(sceneNumber);
            if (startFile) {
              progress.set(`Importing Scene ${sceneNumber} start frame\n${startFile.webkitRelativePath || startFile.name}`, 5 + Math.round((index / Math.max(1, scenes.length)) * 88));
              const startData = await readFileAsDataUrl(startFile);
              await applyCustomImageDataToSegment(segment, startData, startFile.name || `scene_${sceneNumber}_start.png`, { activate: false });
              startCount += 1;
            }
            const endFile = pairedEnds.get(sceneNumber);
            if (endFile) {
              progress.set(`Importing Scene ${sceneNumber} end frame\n${endFile.webkitRelativePath || endFile.name}`, 8 + Math.round((index / Math.max(1, scenes.length)) * 88));
              const endData = await readFileAsDataUrl(endFile);
              const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
              const saved = projectFolder ? await postJson("/vrgdg/music_builder/archive_scene_image", { image_data: endData, project_folder: projectFolder, scene_number: sceneSlotNumber(segment), image_role: "end_frame" }) : { saved_path: "" };
              segment.first_last_frame_end_image_path = saved.saved_path || "";
              segment.first_last_frame_end_image_data = saved.saved_path ? "" : endData;
              segment.first_last_frame_end_image_name = endFile.name || `scene_${sceneNumber}_end.png`;
              endCount += 1;
            }
          }
          setActiveSegment(scenes[0]); render(); await autoSaveSessionQuiet("Storyboard start end frame folder import");
          progress.set(`Imported ${startCount} start frame${startCount === 1 ? "" : "s"} and ${endCount} end frame${endCount === 1 ? "" : "s"}.`, 100); progress.close(1800);
          toast("Storyboard start/end frame pairs imported. FLF chaining is off.");
        } catch (error) {
          progress.set(`Import failed:\n${String(error?.message || error)}`, 100); toast(String(error?.message || error), true);
        } finally {
          importImageFolderButton.disabled = false; importImageFolderButton.textContent = "Fill Timeline Images From Folder";
        }
        return;
      }
      if (images.length < 2) {
        toast("First Last Frame folder fill needs at least two images: Scene 1 first frame, then Scene 1 end frame.", true);
        return;
      }
      const flfSceneCount = createScenesFromImages ? images.length - 1 : scenes.length;
      const neededImages = flfSceneCount + 1;
      const confirmed = window.confirm(
        `Fill ${flfSceneCount} First Last Frame scene${flfSceneCount === 1 ? "" : "s"} from this folder?\n\n`
        + "Image 1 → Scene 1 first frame\n"
        + "Image 2 → Scene 1 end frame\n"
        + "Image 3 → Scene 2 end frame, and so on.\n\n"
        + "Each previous end frame will be reused as the next scene's first frame.\n"
        + (images.length > neededImages ? `${images.length - neededImages} extra image${images.length - neededImages === 1 ? "" : "s"} will be ignored.\n` : "")
        + (images.length < neededImages ? `${neededImages - images.length} scene end frame${neededImages - images.length === 1 ? "" : "s"} will remain missing.\n` : "")
        + "Images are sorted by numbers in their file names."
      );
      if (!confirmed) return;
      pushHistory();
      state.i2vVideoSettings = { ...(state.i2vVideoSettings || defaultI2VVideoSettings()), flf_chain_previous_end_frame: true };
      if (createScenesFromImages) {
        const duration = hasProjectAudio && projectAudioDuration > 0 ? projectAudioDuration : flfSceneCount * 4;
        const sceneDuration = duration / flfSceneCount;
        scenes = Array.from({ length: flfSceneCount }, (_, index) => newSegment(index * sceneDuration, index === flfSceneCount - 1 ? duration : (index + 1) * sceneDuration));
        state.segments = scenes; state.duration = duration; state.activeId = scenes[0]?.id || "";
        if (!hasProjectAudio) {
          globalAudioModeSelect.value = "silent";
          silentAudioDurationInput.value = String(Math.max(4, duration));
          syncGlobalAudioModeControls();
        }
      }
      scenes.forEach((segment) => {
        if (segment.use_scene_i2v_video_settings && segment.i2v_video_settings) segment.i2v_video_settings.flf_chain_previous_end_frame = true;
        segment.flf_rendered_start_frame_path = "";
        segment.flf_rendered_start_frame_data = "";
        segment.flf_rendered_start_frame_name = "";
        segment.flf_rendered_source_video_path = "";
      });
      const progress = createProgressWindow("Fill First Last Frame Images From Folder");
      try {
        importImageFolderButton.disabled = true; importImageFolderButton.textContent = "Importing FLF...";
        const firstData = await readFileAsDataUrl(images[0]);
        await applyCustomImageDataToSegment(scenes[0], firstData, images[0].name || "scene_1_first.png", { activate: false });
        const endCount = Math.min(scenes.length, images.length - 1);
        for (let index = 0; index < endCount; index += 1) {
          const file = images[index + 1];
          progress.set(`Importing end frame ${index + 1}/${endCount}\n${file.webkitRelativePath || file.name}\n→ ${sceneDisplayName(scenes[index], index)}`, 10 + Math.round((index / Math.max(1, endCount)) * 82));
          const data = await readFileAsDataUrl(file);
          const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
          const saved = projectFolder ? await postJson("/vrgdg/music_builder/archive_scene_image", { image_data: data, project_folder: projectFolder, scene_number: sceneSlotNumber(scenes[index]) }) : { saved_path: "" };
          scenes[index].first_last_frame_end_image_path = saved.saved_path || "";
          scenes[index].first_last_frame_end_image_data = saved.saved_path ? "" : data;
          scenes[index].first_last_frame_end_image_name = file.name || `scene_${index + 1}_end.png`;
          scenes[index].flf_motion_plan = "";
          scenes[index].flf_end_frame_prompt = "";
          scenes[index].flf_end_frame_stale = false;
          scenes[index].flf_final_prompt_ready = false;
        }
        setActiveSegment(scenes[0]); render(); await autoSaveSessionQuiet("First Last Frame folder fill");
        progress.set(`First Last Frame folder fill complete: Scene 1 first frame plus ${endCount} end frame${endCount === 1 ? "" : "s"}.`, 100); progress.close(1800);
        toast("First Last Frame images filled from folder.");
      } catch (error) {
        progress.set(`Import failed:\n${String(error?.message || error)}`, 100); toast(String(error?.message || error), true);
      } finally {
        importImageFolderButton.disabled = false; importImageFolderButton.textContent = "Fill Timeline Images From Folder";
      }
      return;
    }
    const plannedSceneCount = createScenesFromImages ? images.length : scenes.length;
    const plannedApplyCount = Math.min(images.length, plannedSceneCount);
    const plannedExtraCount = Math.max(0, images.length - plannedSceneCount);
    const plannedMissingCount = Math.max(0, plannedSceneCount - images.length);
    const matchLoadedAudio = createScenesFromImages && hasProjectAudio && projectAudioDuration > 0;
    const newSceneDuration = matchLoadedAudio ? projectAudioDuration / images.length : 4;
    const creationSummary = matchLoadedAudio
      ? `Create ${images.length} scene${images.length === 1 ? "" : "s"} across the full ${formatTime(projectAudioDuration)} audio timeline (about ${newSceneDuration.toFixed(2)} seconds each) and fill them`
      : `Create ${images.length} four-second scene${images.length === 1 ? "" : "s"} and fill them`;
    const confirmed = window.confirm(
      `${createScenesFromImages ? creationSummary : `Fill ${plannedApplyCount} timeline scene${plannedApplyCount === 1 ? "" : "s"}`} from this image folder?\n\n`
      + `Images are sorted by numbers in the file names.\n`
      + (createScenesFromImages && !hasProjectAudio ? "The blank project will use a silent timeline by default.\n" : "")
      + (plannedExtraCount ? `${plannedExtraCount} extra image${plannedExtraCount === 1 ? "" : "s"} will be ignored.\n` : "")
      + (plannedMissingCount ? `${plannedMissingCount} scene${plannedMissingCount === 1 ? "" : "s"} will stay blank.\n` : "")
      + "\nExisting scene images in those filled slots will be replaced."
    );
    if (!confirmed) return;

    pushHistory();
    if (createScenesFromImages) {
      const timelineDuration = matchLoadedAudio ? projectAudioDuration : images.length * 4;
      scenes = images.map((_, index) => newSegment(
        index * newSceneDuration,
        index === images.length - 1 ? timelineDuration : (index + 1) * newSceneDuration
      ));
      state.segments = scenes;
      state.duration = timelineDuration;
      state.activeId = scenes[0]?.id || "";
      state.activeTrack = "base";
      if (!hasProjectAudio) {
        globalAudioModeSelect.value = "silent";
        silentAudioDurationInput.value = String(Math.max(4, state.duration));
        syncGlobalAudioModeControls();
      }
    }
    const applyCount = Math.min(images.length, scenes.length);
    const extraCount = Math.max(0, images.length - scenes.length);
    const missingCount = Math.max(0, scenes.length - images.length);
    let progress = null;
    try {
      importImageFolderButton.disabled = true;
      importImageFolderButton.textContent = "Importing...";
      progress = createProgressWindow("Import Timeline Image Folder");
      progress.set(`Importing ${applyCount} image${applyCount === 1 ? "" : "s"} into timeline scenes...`, 4);
      for (let index = 0; index < applyCount; index += 1) {
        const file = images[index];
        const segment = scenes[index];
        const percent = 8 + Math.round((index / Math.max(1, applyCount)) * 84);
        progress.set(`Importing ${index + 1}/${applyCount}\n${file.webkitRelativePath || file.name}\n→ ${sceneDisplayName(segment, segmentIndexInfo(segment).index)}`, percent);
        const imageData = await readFileAsDataUrl(file);
        await applyCustomImageDataToSegment(segment, imageData, file.name || `folder_image_${index + 1}.png`, { activate: false });
      }
      const active = activeSegment() || scenes[0];
      if (active) {
        setActiveSegment(active);
        syncPreview(active);
      }
      render();
      await autoSaveSessionQuiet("timeline image folder import");
      progress.set(`Imported ${applyCount} image${applyCount === 1 ? "" : "s"}.${extraCount ? ` Ignored ${extraCount} extra.` : ""}${missingCount ? ` Left ${missingCount} scene${missingCount === 1 ? "" : "s"} blank.` : ""}`, 100);
      progress.close(1500);
      toast(`Imported ${applyCount} image${applyCount === 1 ? "" : "s"} into the timeline.${extraCount ? `\nIgnored ${extraCount} extra.` : ""}${missingCount ? `\nLeft ${missingCount} scene${missingCount === 1 ? "" : "s"} blank.` : ""}`);
    } catch (error) {
      progress?.set(`Import failed:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      importImageFolderButton.disabled = false;
      importImageFolderButton.textContent = "Fill Timeline Images From Folder";
    }
  }

  function setImageToImageSource({ path = "", data = "", name = "" } = {}) {
    const useErnie = (state.imageModelMode || "") === "ernie_image";
    const useKrea2TwoPass = (state.imageModelMode || "") === "krea2_2pass";
    const settings = useKrea2TwoPass
      ? (activeKrea2TwoPassSettings() || defaultKrea2TwoPassSettings())
      : useErnie
        ? (state.ernieImageSettings || defaultErnieImageSettings())
        : activeZImageSettings();
    pushHistory();
    settings.use_image_to_image = true;
    settings.image_to_image_path = path || "";
    settings.image_to_image_data = data || "";
    settings.image_to_image_name = name || "";
    if (useKrea2TwoPass) {
      settings.image_to_image_creativity = Math.max(0, Math.min(10, Number(krea2TwoPassCreativityInput.value || krea2TwoPassCreativity.value || settings.image_to_image_creativity || 5)));
      if (activeSegment()?.use_scene_krea2_2pass_settings) activeSegment().krea2_2pass_settings = settings;
      else state.krea2TwoPassSettings = settings;
      syncKrea2TwoPassPanel();
    } else {
      settings.image_to_image_start_at_step = Math.max(1, Math.min(8, Number(
        useErnie ? (ernieI2IStartStep.value || ernieI2ISlider.value || settings.image_to_image_start_at_step || 5) : (zI2IStartStep.value || zI2ISlider.value || settings.image_to_image_start_at_step || 5)
      )));
    }
    if (useErnie) {
      state.ernieImageSettings = settings;
      syncErnieImagePanel();
    } else if (!useKrea2TwoPass) {
      syncZImageSettingsPanel();
    }
    renderList();
    toast(`Image-to-image source set${path || name ? `:\n${path || name}` : "."}`);
  }

  function loadImageToImageFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImageToImageSource({ data: String(reader.result || ""), name: file.name || "image.png" });
    reader.onerror = () => toast("Failed to read the image-to-image source.", true);
    reader.readAsDataURL(file);
  }

  async function setVisionReferenceSource({ path = "", data = "", name = "", forT2V = false } = {}) {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    if (path) {
      segment.ref_image_path = path;
    } else if (data) {
      const sceneNumber = sceneSlotNumber(segment);
      const saved = await postJson("/vrgdg/music_builder/archive_scene_image", {
        image_data: data,
        project_folder: projectInput.value || state.projectFolder,
        scene_number: sceneNumber,
      });
      segment.ref_image_path = saved.saved_path || "";
    }
    if (forT2V) {
      segment.use_t2v_vision_reference = true;
      useT2VVisionReference.input.checked = true;
    } else {
      segment.use_vision_reference = true;
      useVisionReference.input.checked = true;
      ernieUseVisionReference.input.checked = true;
      krea2TwoPassUseVisionReference.input.checked = true;
    }
    refImageInput.value = segment.ref_image_path || name || "";
    refImagePanel.style.display = useVisionReference.input.checked ? "flex" : "none";
    ernieRefImagePanel.style.display = ernieUseVisionReference.input.checked ? "flex" : "none";
    krea2TwoPassRefImagePanel.style.display = krea2TwoPassUseVisionReference.input.checked ? "flex" : "none";
    t2vRefImagePanel.style.display = currentVideoMode() === "t2v" && useT2VVisionReference.input.checked ? "flex" : "none";
    renderList();
    toast(`Vision reference set${segment.ref_image_path ? `:\n${segment.ref_image_path}` : "."}`);
  }

  function loadVisionReferenceFile(file, options = {}) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setVisionReferenceSource({ data: String(reader.result || ""), name: file.name || "reference.png", forT2V: Boolean(options.forT2V) }).catch((error) => toast(String(error?.message || error), true));
    };
    reader.onerror = () => toast("Failed to read the vision reference image.", true);
    reader.readAsDataURL(file);
  }

  function droppedSceneImageSource(event) {
    const id = sceneIdFromDrop(event);
    if (!id) return null;
    const segment = state.segments.find((item) => item.id === id);
    return segment ? segmentImageSource(segment) : null;
  }

  function addFluxIngredient({ path = "", data = "", name = "", global = false } = {}) {
    pushHistory();
    const candidate = {
      path: path || "",
      data: data || "",
      name: name || path?.split?.(/[\\/]/)?.pop?.() || "image.png",
    };
    const ingredientKey = (item) => String(item?.path || item?.data || item?.name || "").trim();
    const candidateKey = ingredientKey(candidate);
    if (global) {
      if (!Array.isArray(state.fluxGlobalImageIngredients)) state.fluxGlobalImageIngredients = [];
      if (candidateKey && state.fluxGlobalImageIngredients.some((item) => ingredientKey(item) === candidateKey)) {
        renderFluxGlobalIngredientList();
        toast("That global image ingredient is already loaded.");
        return;
      }
      state.fluxGlobalImageIngredients.push(candidate);
      renderFluxGlobalIngredientList();
      render();
      toast(`Global image ingredient added${path || name ? `:\n${path || name}` : "."}`);
      return;
    }
    const segment = activeSegment();
    if (!segment) {
      toast("Add or select a scene first.", true);
      return;
    }
    if (!Array.isArray(segment.flux_image_ingredients)) segment.flux_image_ingredients = [];
    if (candidateKey && segment.flux_image_ingredients.some((item) => ingredientKey(item) === candidateKey)) {
      renderFluxIngredientList(segment);
      renderNBIngredientList(segment);
      toast("That image ingredient is already loaded for this scene.");
      return;
    }
    segment.flux_image_ingredients.push(candidate);
    const settings = state.fluxKleinSettings || {};
    settings.enabled = true;
    state.fluxKleinSettings = settings;
    syncFluxKleinPanel();
    renderFluxIngredientList(segment);
    renderNBIngredientList(segment);
    render();
    toast(`Image ingredient added${path || name ? `:\n${path || name}` : "."}`);
  }

  function loadFluxIngredientFile(file, { global = false } = {}) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => addFluxIngredient({ data: String(reader.result || ""), name: file.name || "image.png", global });
    reader.onerror = () => toast("Failed to read the image ingredient.", true);
    reader.readAsDataURL(file);
  }

  function enableFluxIngredientDrop(element, { global = false } = {}) {
    element.dataset.vrgdgFileDropZone = "true";
    element.addEventListener("dragover", (event) => {
      const types = Array.from(event.dataTransfer?.types || []).map((item) => String(item).toLowerCase());
      if (!types.includes("files") && !types.includes("application/x-vrgdg-segment-id")) return;
      event.preventDefault();
      event.stopPropagation();
      element.style.borderColor = "#a3e635";
    });
    element.addEventListener("dragleave", () => {
      element.style.borderColor = "#155e75";
    });
    element.addEventListener("drop", (event) => {
      const sceneSource = droppedSceneImageSource(event);
      const files = Array.from(event.dataTransfer?.files || []).filter((file) => file.type?.startsWith?.("image/"));
      if (!sceneSource && !files.length) return;
      event.preventDefault();
      event.stopPropagation();
      element.style.borderColor = "#155e75";
      if (sceneSource) {
        addFluxIngredient({
          path: sceneSource.path || "",
          data: sceneSource.data || "",
          name: sceneSource.name || "scene_image.png",
          global,
        });
        return;
      }
      for (const file of files) loadFluxIngredientFile(file, { global });
    });
  }

  function enableImageDrop(element, segment) {
    element.dataset.vrgdgFileDropZone = "true";
    element.addEventListener("dragenter", (event) => {
      if (!dropHasFiles(event)) return;
      event.preventDefault();
      event.stopPropagation();
      event.dataTransfer.dropEffect = "copy";
      element.style.outline = "2px solid #a3e635";
    }, true);
    element.addEventListener("dragover", (event) => {
      if (!dropHasFiles(event)) return;
      event.preventDefault();
      event.stopPropagation();
      event.dataTransfer.dropEffect = "copy";
      element.style.outline = "2px solid #a3e635";
    }, true);
    element.addEventListener("dragleave", (event) => {
      event.stopPropagation();
      element.style.outline = "";
    });
    element.addEventListener("drop", async (event) => {
      if (!dropHasFiles(event)) return;
      event.preventDefault();
      event.stopPropagation();
      element.style.outline = "";
      const file = imageFileFromDrop(event);
      if (!file) {
        toast("Drop a PNG, JPG, WEBP, GIF, BMP, TIFF, or AVIF image file.", true);
        return;
      }
      if (currentVideoMode() === "flf") {
        const role = await new Promise((resolve) => {
          const backdrop = document.createElement("div");
          backdrop.style.cssText = "position:fixed;inset:0;z-index:100030;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:20px;";
          const box = document.createElement("div");
          box.style.cssText = "width:min(520px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;padding:16px;display:flex;flex-direction:column;gap:12px;box-shadow:0 20px 70px rgba(0,0,0,.6);";
          const heading = document.createElement("div"); heading.textContent = "Add dropped image as…"; heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
          const body = document.createElement("div"); body.textContent = `${file.name || "Dropped image"} can be the opening or ending frame for ${sceneDisplayName(segment, segmentIndexInfo(segment).index)}.`; body.style.cssText = "font-size:12px;color:#d4d4d8;line-height:1.45;";
          const actions = document.createElement("div"); actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;";
          const cancel = makeButton("Cancel"); const first = makeButton("First Frame", "primary"); const last = makeButton("End Frame", "primary");
          cancel.onclick = () => { backdrop.remove(); resolve(""); };
          first.onclick = () => { backdrop.remove(); resolve("first"); };
          last.onclick = () => { backdrop.remove(); resolve("last"); };
          actions.append(cancel, first, last); box.append(heading, body, actions); backdrop.append(box); document.body.append(backdrop);
        });
        if (role === "last") await loadFirstLastFrameEndFile(file, segment);
        else if (role === "first") {
          if (previousAutoChainSourceSegment(segment) && flfChainingEnabled(segment)) {
            segment.use_scene_i2v_video_settings = true;
            segment.i2v_video_settings = { ...cloneI2VVideoSettings(state.i2vVideoSettings), flf_chain_previous_end_frame: false };
          }
          loadCustomImageFile(file, segment);
        }
        return;
      }
      loadCustomImageFile(file, segment);
    }, true);
  }

  function segmentImageSource(segment) {
    if (!segment) return null;
    ensureSegmentRuntimeFields(segment);
    const historyPath = segment.image_history?.[segment.image_history_index] || segment.image_history?.[segment.image_history.length - 1] || "";
    if (historyPath) {
      return { path: historyPath };
    }
    if (segment.custom_image_path) {
      return { path: segment.custom_image_path };
    }
    if (segment.custom_image_data) {
      return { data: segment.custom_image_data, name: segment.custom_image_name || "custom_image.png" };
    }
    if (segment.approved_image_path) {
      return { path: segment.approved_image_path };
    }
    return null;
  }

  return {
    addFluxIngredient, droppedSceneImageSource, enableFluxIngredientDrop, enableImageDrop,
    importTimelineImagesFromFolder, loadCustomImageFile, loadFluxIngredientFile, loadImageToImageFile,
    loadVisionReferenceFile, makeDragHandle, segmentImageSource, setImageToImageSource,
    setVisionReferenceSource,
  };
}
