import { storyboardCutFrequencyValue, storyboardCutPlanForDuration } from "../storyboard_builder/scenes.mjs";
import { makeEditorImageUrl, postJson } from "./comfy_api.mjs";
import {
  DEFAULT_I2V_DIFFUSION_MODEL,
  DEFAULT_I2V_PASS1_SIGMAS,
  DEFAULT_I2V_PASS2_SIGMAS,
  DEFAULT_INGREDIENTS_SAMPLER,
  DEFAULT_LTX_INGREDIENTS_HEIGHT,
  DEFAULT_LTX_INGREDIENTS_WIDTH,
  REQUIRED_LTX_ID_LORA,
  REQUIRED_LTX_INGREDIENTS_LORA,
  REQUIRED_LTX_MSR_LORA,
} from "./constants.mjs";
import { escapeHtml, normalizeProjectVideoEngine, normalizeVideoType, toast } from "./controls.mjs";
import { normalizeI2VSigmasText } from "./image_panels.mjs";
import {
  normalizeMiniMaxH3AudioMode,
  normalizeMiniMaxH3Mode,
  normalizeMiniMaxH3Voice,
  normalizeMiniMaxSpeakerAssignments,
} from "./minimax_h3.mjs";
import { miniMaxH3CompactExtraIdentity, miniMaxH3ExtraDisplayTitle } from "./minimax_prompt.mjs";
import {
  cloneI2VVideoSettings,
  normalizeBuilderStoryboardDefaults,
  normalizeBuilderStoryLayer,
  repairI2VVideoSettingDimensions,
} from "./model_settings.mjs";
import { isInstrumentalLyricText, segmentUsesNoLipSyncPerformance } from "./prompt_text.mjs";
import { normalizeVideoPromptOrigin } from "./segments.mjs";
import { timelineSegmentDuration } from "./timeline_state.mjs";

export function storyboardSubjectsForSegment(segment) {
  if (segment?.no_character_present) return [];
  const singers = Array.isArray(segment?.lyric_singers)
    ? segment.lyric_singers
    : String(segment?.lyric_singers || segment?.singers || "").split(/[,;\n]+/);
  const mapped = String(segment?.mapped_subjects || segment?.subject || "").split(/[,;\n]+/);
  return [...singers, ...mapped].map((item) => String(item || "").trim()).filter(Boolean);
}

function storyboardSummaryForSegment(segment, imagePrompt = "", videoPrompt = "", referenceData = {}) {
  const parts = [
    String(segment?.notes || segment?.director_note || "").trim(),
    String(segment?.timeline_note || "").trim(),
    String(segment?.i2v_notes || segment?.video_notes || "").trim(),
    String(referenceData?.location_ref?.description || "").trim(),
    segmentUsesNoLipSyncPerformance(segment) ? "" : String(segment?.lyric_text || segment?.lyric_note || segment?.lyrics || "").trim(),
  ].filter(Boolean);
  return parts[0] || "";
}

async function saveStoryboardPromptFromTimeline(projectFolder, fresh, promptValue) {
  const { storyboard } = await postJson("/vrgdg/storyboard/load", { project_folder: projectFolder });
  const scenes = Array.isArray(storyboard.scenes) ? storyboard.scenes.slice() : [];
  let index = scenes.findIndex((scene) => scene.id === fresh.id);
  if (index < 0 && !Array.isArray(storyboard.source_scene_ids)) {
    index = scenes.findIndex((scene) => Number(scene.scene_number) === Number(fresh.scene_number));
  }
  const prompt = { video_prompt: promptValue, video_prompt_origin: "manual" };
  if (index >= 0) scenes[index] = { ...scenes[index], ...prompt };
  else scenes.push({ ...fresh, ...prompt });
  await postJson("/vrgdg/storyboard/save", {
    project_folder: projectFolder,
    storyboard: { ...storyboard, scenes },
  });
}

export function createSceneOutput({
  activeProjectFolderForSave, activeSegment, allEditableSegments, audioInput, currentVideoMode,
  effectiveVideoPerformanceModeForSegment, firstLastFrameStartImageSource, getI2VImageReference, i2vPrompt,
  logicalExtraSubjectsForScene, logicalReferenceSubjects, logicalSubjectIdsForScene, miniMaxH3ModeForSegment,
  miniMaxH3SettingsForSegment, miniMaxH3VocalCueMapText, miniMaxPrompt, normalizeLyricCueMapForSegment,
  projectInput, saveI2VPromptButton, saveI2VVideoSettingsFromPanel, saveMiniMaxPromptButton, saveSession,
  savedI2VPrompts, savedMiniMaxPrompts, sceneReferenceMapValue, sceneSlotNumber, segmentImageSource,
  segmentIndexInfo, selectedPerformerSubjectsForSegment, selectedSegmentImagePath, state,
  storyboardReferenceBuilderWithIdLoraRefs, timelinePromptSave, updateI2VPromptSaveButtonState,
  updateMiniMaxPromptSaveButtonState,
}) {
  function i2vImagesFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}/zimage_approved`;
  }

  function i2vVideoOutputFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}/image_to_video_clips`;
  }

  function t2vVideoOutputFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\text_to_video_clips`;
  }

  function rtvVideoOutputFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\reference_to_video_clips`;
  }

  function ingredientsVideoOutputFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\ingredients_to_video_clips`;
  }
  function flfVideoOutputFolder() { return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\first_last_frame_clips`; }

  function idLoraVideoOutputFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\id_lora_i2v_clips`;
  }

  function activeVideoOutputFolder(mode = currentVideoMode()) {
    if (mode === "id_lora") return idLoraVideoOutputFolder();
    if (mode === "ingredients") return ingredientsVideoOutputFolder();
    if (mode === "rtv") return rtvVideoOutputFolder();
    if (mode === "flf") return flfVideoOutputFolder();
    return mode === "t2v" ? t2vVideoOutputFolder() : i2vVideoOutputFolder();
  }

  function collectedSceneVideoFolder() {
    return `${String(projectInput.value || "").replace(/[\\/]+$/, "")}\\rendered_scene_videos`;
  }

  function sceneVideoDetailsHtml(segment, sceneIndex, srtPath, outputFolder, statusText = "Preparing hidden video workflow...", details = {}) {
    const videoMode = details.videoMode === "id_lora" ? "id_lora" : details.videoMode === "ingredients" ? "ingredients" : details.videoMode === "flf" ? "flf" : details.videoMode === "rtv" ? "rtv" : details.videoMode === "t2v" ? "t2v" : "i2v";
    const promptNumber = Number(details.promptNumber || sceneIndex + 1);
    const imageIndex = sceneSlotNumber(segment) - 1;
    const imageSource = videoMode === "flf" ? firstLastFrameStartImageSource(segment) : segmentImageSource(segment);
    const imagePath = imageSource?.path || "";
    const imageSrc = imagePath ? makeEditorImageUrl(imagePath) : imageSource?.data || "";
    const audioPath = String(details.audioPath || audioInput.value || "");
    const audioMode = String(details.audioMode || (segment.custom_audio_path ? "Custom scene audio" : "Global/project audio"));
    const rtvSubjects = Array.isArray(details.rtvReferences?.subjects) ? details.rtvReferences.subjects : [];
    const rtvSubjectText = rtvSubjects
      .map((item, index) => {
        const label = String(item?.label || item?.name || item?.path || "").trim();
        return `${index + 1}. ${label || "Unnamed subject reference"}`;
      })
      .join("\n");
    const rtvBackground = details.rtvReferences?.background || {};
    const rtvBackgroundText = String(rtvBackground?.label || rtvBackground?.name || rtvBackground?.path || "").trim();
    const flfInputs = details.flfInputs || null;
    return `
      <div style="display:flex;flex-direction:column;gap:10px;">
        <div style="font-weight:900;color:#cffafe;">${escapeHtml(statusText)}</div>
        ${(videoMode === "i2v" || videoMode === "flf" || videoMode === "ingredients" || videoMode === "id_lora") && imageSrc ? `<img src="${imageSrc}" style="width:180px;max-height:110px;object-fit:cover;border:1px solid #155e75;border-radius:6px;background:#050505;">` : ""}
        <div style="display:grid;grid-template-columns:150px minmax(0,1fr);gap:5px 10px;font-size:11px;">
          <div style="color:#67e8f9;font-weight:900;">Scene</div><div>${escapeHtml(segment.label || `Scene ${promptNumber}`)}</div>
          <div style="color:#67e8f9;font-weight:900;">Video mode</div><div>${videoModeDisplayLabel(videoMode)}</div>
          ${videoMode === "i2v" ? `<div style="color:#67e8f9;font-weight:900;">Image index</div><div>${imageIndex} (0 based)</div>` : ""}
          <div style="color:#67e8f9;font-weight:900;">SRT prompt #</div><div>${promptNumber} (1 based)</div>
          <div style="color:#67e8f9;font-weight:900;">Audio mode</div><div>${escapeHtml(audioMode)}</div>
          ${videoMode === "i2v" ? `<div style="color:#67e8f9;font-weight:900;">Image folder</div><div style="overflow-wrap:anywhere;">${escapeHtml(i2vImagesFolder())}</div>` : ""}
          ${(videoMode === "i2v" || videoMode === "ingredients" || videoMode === "id_lora") ? `<div style="color:#67e8f9;font-weight:900;">Image path</div><div style="overflow-wrap:anywhere;">${escapeHtml(imagePath || imageSource?.name || "")}</div>` : ""}
          <div style="color:#67e8f9;font-weight:900;">${videoMode === "id_lora" ? "Reference voice" : "Audio sent to LTX"}</div><div style="overflow-wrap:anywhere;">${escapeHtml(audioPath)}</div>
          <div style="color:#67e8f9;font-weight:900;">SRT path</div><div style="overflow-wrap:anywhere;">${escapeHtml(srtPath || "")}</div>
          <div style="color:#67e8f9;font-weight:900;">Save folder</div><div style="overflow-wrap:anywhere;">${escapeHtml(outputFolder || "")}</div>
          <div style="color:#67e8f9;font-weight:900;">Collected clips</div><div style="overflow-wrap:anywhere;">${escapeHtml(collectedSceneVideoFolder())}</div>
          ${videoMode === "rtv" ? `<div style="color:#67e8f9;font-weight:900;">RTV subject refs</div><div style="overflow-wrap:anywhere;white-space:pre-wrap;">${escapeHtml(rtvSubjectText || "None")}</div>` : ""}
          ${videoMode === "rtv" ? `<div style="color:#67e8f9;font-weight:900;">RTV location ref</div><div style="overflow-wrap:anywhere;">${escapeHtml(rtvBackgroundText || "None")}</div>` : ""}
          ${videoMode === "flf" && flfInputs ? `<div style="color:#67e8f9;font-weight:900;">First frame → node ${escapeHtml(flfInputs.first_node || "950")}</div><div style="overflow-wrap:anywhere;">Source: ${escapeHtml(flfInputs.first_source || "")}` + `<br>LoadImage: ${escapeHtml(flfInputs.first_load_image || "")}</div>` : ""}
          ${videoMode === "flf" && flfInputs ? `<div style="color:#67e8f9;font-weight:900;">End frame → node ${escapeHtml(flfInputs.last_node || "945")}</div><div style="overflow-wrap:anywhere;">Source: ${escapeHtml(flfInputs.last_source || "")}` + `<br>LoadImage: ${escapeHtml(flfInputs.last_load_image || "")}</div>` : ""}
          ${videoMode === "flf" && flfInputs ? `<div style="color:#67e8f9;font-weight:900;">Input verification</div><div style="font-weight:900;color:${flfInputs.inputs_are_different ? "#86efac" : "#fca5a5"};">${flfInputs.inputs_are_different ? "PASS — two different images" : "FAILED — same image"}</div>` : ""}
          ${videoMode === "flf" && flfInputs ? `<div style="color:#67e8f9;font-weight:900;">LoRA verification → node ${escapeHtml(flfInputs.lora_node || "937")}</div><div style="font-weight:900;color:${flfInputs.loras_enabled && Number(flfInputs.lora_count || 0) > 0 ? "#86efac" : "#fbbf24"};">${flfInputs.loras_enabled && Number(flfInputs.lora_count || 0) > 0 ? `ACTIVE — ${escapeHtml(String(flfInputs.lora_count))} LoRA(s)` : "OFF — no LoRA attached"}</div>${Array.isArray(flfInputs.loras) ? flfInputs.loras.map((item) => `<div style="overflow-wrap:anywhere;">${escapeHtml(item.name || "[none]")} @ ${escapeHtml(String(item.strength ?? 1))}</div>`).join("") : ""}` : ""}
        </div>
        <div>
          <div style="color:#67e8f9;font-weight:900;margin-bottom:4px;">${videoModeDisplayLabel(videoMode, true)} prompt</div>
          <div style="border:1px solid #155e75;border-radius:6px;background:#020617;color:#e0f2fe;padding:8px;max-height:130px;overflow:auto;white-space:pre-wrap;">${escapeHtml(segment.i2v_prompt || "")}</div>
        </div>
      </div>
    `;
  }

  function i2vVideoSettingsPayload(segment = activeSegment()) {
    const settings = i2vVideoSettingsForSegment(segment);
    repairI2VVideoSettingDimensions(settings);
    const useLoras = Boolean(settings.use_loras && Number(settings.lora_count || 0) > 0);
    const count = Math.max(0, Math.min(4, Number(settings.lora_count || 0)));
    const videoMode = currentVideoMode();
    const singlePassLoras = videoMode === "flf" || (videoMode === "rtv" && settings.ltx_version === "2.3");
    let pass1SamplerName = settings.pass1_sampler_name || "euler_ancestral";
    let pass1Sigmas = settings.pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS;
    let pass2SamplerName = settings.pass2_sampler_name || "euler_ancestral";
    let pass2Sigmas = settings.pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS;
    if (videoMode === "t2v") {
      pass1SamplerName = settings.t2v_pass1_sampler_name || "euler_ancestral";
      pass1Sigmas = settings.t2v_pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS;
      pass2SamplerName = settings.t2v_pass2_sampler_name || "euler_ancestral";
      pass2Sigmas = settings.t2v_pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS;
    } else if (videoMode === "rtv") {
      pass1SamplerName = settings.rtv_pass1_sampler_name || "euler_ancestral";
      pass1Sigmas = settings.rtv_pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS;
      pass2SamplerName = settings.rtv_pass2_sampler_name || "euler_ancestral";
      pass2Sigmas = settings.rtv_pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS;
    } else if (videoMode === "ingredients") {
      pass1SamplerName = settings.ingredients_pass1_sampler_name || DEFAULT_INGREDIENTS_SAMPLER;
      pass1Sigmas = settings.ingredients_pass1_sigmas || DEFAULT_I2V_PASS1_SIGMAS;
      pass2SamplerName = settings.ingredients_pass2_sampler_name || DEFAULT_INGREDIENTS_SAMPLER;
      pass2Sigmas = settings.ingredients_pass2_sigmas || DEFAULT_I2V_PASS2_SIGMAS;
    }
    const payload = {
      ltx_version: settings.ltx_version === "2.3" ? "2.3" : "2.5",
      use_gguf_model: settings.use_gguf_model !== false,
      unet_name: settings.unet_name || "",
      diffusion_model_name: settings.diffusion_model_name || DEFAULT_I2V_DIFFUSION_MODEL,
      use_sage_attention: Boolean(settings.use_sage_attention),
      enable_fp16_accumulation: Boolean(settings.enable_fp16_accumulation),
      vae_name: settings.vae_name || "",
      clip_name1: settings.clip_name1 || "",
      clip_name2: settings.clip_name2 || "",
      upscale_model_name: settings.upscale_model_name || "",
      audio_vae_name: settings.audio_vae_name || "",
      fps: Number(settings.fps || 24),
      width: videoMode === "ingredients" ? Number(settings.ingredients_width || DEFAULT_LTX_INGREDIENTS_WIDTH) : Number(settings.width || 1920),
      height: videoMode === "ingredients" ? Number(settings.ingredients_height || DEFAULT_LTX_INGREDIENTS_HEIGHT) : Number(settings.height || 1080),
      resolution_aspect_ratio: settings.resolution_aspect_ratio || "16:9 (Widescreen)",
      resolution_megapixels: Number(settings.resolution_megapixels || 1.2),
      seed: Number(settings.seed || 1),
      tail_loss_frames: videoMode === "id_lora" ? 0 : Math.max(0, Number(settings.tail_loss_frames ?? 25)),
      pre_frames: videoMode === "id_lora" ? 0 : videoMode === "flf" ? Math.max(0, Number(settings.flf_pre_frames ?? 0)) : Math.max(0, Number(settings.pre_frames ?? 50)),
      first_guide_strength: Math.max(0, Math.min(1, Number(settings.flf_first_guide_strength ?? 0.7))),
      last_guide_strength: Math.max(0, Math.min(1, Number(settings.flf_last_guide_strength ?? 0.7))),
      first_guide_frame_idx: Math.trunc(Number(settings.flf_first_guide_frame_idx ?? 0)),
      last_guide_frame_idx: Math.trunc(Number(settings.flf_last_guide_frame_idx ?? -1)),
      first_guide_crf: Number(settings.flf_first_guide_crf ?? 29),
      last_guide_crf: Number(settings.flf_last_guide_crf ?? 29),
      first_guide_blur_radius: Number(settings.flf_first_guide_blur_radius ?? 1),
      last_guide_blur_radius: Number(settings.flf_last_guide_blur_radius ?? 1),
      first_guide_interpolation: settings.flf_first_guide_interpolation || "lanczos",
      last_guide_interpolation: settings.flf_last_guide_interpolation || "lanczos",
      first_guide_crop: settings.flf_first_guide_crop || "center",
      last_guide_crop: settings.flf_last_guide_crop || "center",
      first_attention_strength: Number(settings.flf_first_attention_strength ?? 0.9),
      last_attention_strength: Number(settings.flf_last_attention_strength ?? 1),
      msr_lora_name: settings.msr_lora_name || REQUIRED_LTX_MSR_LORA,
      msr_first_pass_strength: Number(settings.msr_first_pass_strength ?? 1),
      msr_second_pass_strength: 0,
      msr_reference_strength: settings.msr_reference_strength || "auto - based on subject count",
      msr_background_mode: settings.msr_background_mode || (settings.ltx_version === "2.3" ? "neutral placeholder (WIP/testing)" : "no background reference"),
      ingredients_lora_name: settings.ingredients_lora_name || REQUIRED_LTX_INGREDIENTS_LORA,
      ingredients_first_pass_strength: Number(settings.ingredients_first_pass_strength ?? 1),
      ingredients_second_pass_strength: 0,
      ingredients_width: Number(settings.ingredients_width || DEFAULT_LTX_INGREDIENTS_WIDTH),
      ingredients_height: Number(settings.ingredients_height || DEFAULT_LTX_INGREDIENTS_HEIGHT),
      duration: Math.max(0.25, Number(timelineSegmentDuration(segment) || settings.id_lora_duration || 5)),
      pass1_sampler_name: pass1SamplerName,
      pass1_sigmas: normalizeI2VSigmasText(pass1Sigmas, DEFAULT_I2V_PASS1_SIGMAS),
      pass1_inplace_strength: Math.max(0, Math.min(1, Number(settings.pass1_inplace_strength ?? 1))),
      pass1_inplace_bypass: Boolean(settings.pass1_inplace_bypass),
      pass2_sampler_name: pass2SamplerName,
      pass2_sigmas: normalizeI2VSigmasText(pass2Sigmas, DEFAULT_I2V_PASS2_SIGMAS),
      pass2_inplace_strength: Math.max(0, Math.min(1, Number(settings.pass2_inplace_strength ?? 1))),
      pass2_inplace_bypass: Boolean(settings.pass2_inplace_bypass),
      use_custom_loras: useLoras,
      lora_count: useLoras ? count : 0,
    };
    for (let index = 0; index < 4; index += 1) {
      const lora = settings.loras?.[index] || {};
      payload[`lora_${index + 1}`] = useLoras && index < count ? (lora.name || "[none]") : "[none]";
      payload[`first_pass_strength_${index + 1}`] = Number(lora.first_pass_strength ?? lora.strength ?? 1);
      payload[`second_pass_strength_${index + 1}`] = singlePassLoras ? 0 : Number(lora.second_pass_strength ?? lora.strength ?? 1);
    }
    if (videoMode === "id_lora") {
      payload.id_lora_name = settings.id_lora_name || REQUIRED_LTX_ID_LORA;
      payload.id_lora_first_pass_strength = Number(settings.id_lora_first_pass_strength ?? 1);
      payload.id_lora_second_pass_strength = Number(settings.id_lora_second_pass_strength ?? 1);
      payload.reference_audio_path = settings.id_lora_reference_audio_path || "";
      payload.id_reference_audio_path = settings.id_lora_reference_audio_path || "";
      payload.identity_guidance_scale = Number(settings.identity_guidance_scale ?? 3);
      payload.identity_start_percent = 0;
      payload.identity_end_percent = 1;
    }
    return payload;
  }

  function timelineSegmentLabel(segment) {
    const info = segmentIndexInfo(segment);
    const rawLabel = String(segment?.label || "").trim();
    if (info.track === "overlay") return rawLabel || `Insert ${info.index + 1}`;
    const number = Math.max(1, info.index + 1);
    if (!rawLabel || /^scene(?:\s+\d+(?:\.\d+)?)?$/i.test(rawLabel)) return `Scene ${number}`;
    const numberedDescription = rawLabel.match(/^\d+\.\s*(.+)$/);
    if (numberedDescription) return `${number}. ${numberedDescription[1]}`;
    return rawLabel;
  }

  function storyboardPromptForSegment(segment) {
    const imageMode = state.imageModelMode || "zimage";
    const candidates = [
      imageMode === "flow_gpt" ? segment?.flow_gpt_prompt : "",
      imageMode === "flux_klein" ? segment?.flux_klein_prompt : "",
      imageMode === "nano_banana" ? segment?.nb_prompt : "",
      imageMode === "ernie_image" ? segment?.ernie_t2i_prompt : "",
      segment?.flow_gpt_prompt,
      segment?.t2i_prompt,
      segment?.flux_klein_prompt,
      segment?.nb_prompt,
      segment?.ernie_t2i_prompt,
    ];
    return String(candidates.find((item) => String(item || "").trim()) || "").trim();
  }

  async function saveTimelinePrompt(kind) {
    const segment = activeSegment();
    if (!segment || timelinePromptSave.saving) return;
    const miniMax = kind === "minimax";
    const input = miniMax ? miniMaxPrompt : i2vPrompt;
    const button = miniMax ? saveMiniMaxPromptButton : saveI2VPromptButton;
    const snapshots = miniMax ? savedMiniMaxPrompts : savedI2VPrompts;
    const promptValue = input.value || "";
    const projectFolder = activeProjectFolderForSave();
    if (!projectFolder) {
      toast("Save the project before saving a scene prompt.", true);
      return;
    }
    segment[miniMax ? "minimax_h3_prompt" : "i2v_prompt"] = promptValue;
    segment[miniMax ? "minimax_h3_prompt_origin" : "i2v_prompt_origin"] = "manual";
    const fresh = storyboardScenePayload().find((scene) => scene.id === segment.id);
    if (!fresh) {
      toast("Could not find this scene in the project storyboard inputs.", true);
      return;
    }
    timelinePromptSave.saving = true;
    updateI2VPromptSaveButtonState();
    updateMiniMaxPromptSaveButtonState();
    button.textContent = "Saving...";
    try {
      const result = await saveSession({ quiet: true, throwOnError: true });
      if (result?.stale) throw new Error("The project changed during save. Save the prompt again to keep the latest edits.");
      await saveStoryboardPromptFromTimeline(projectFolder, fresh, promptValue);
      snapshots.set(segment, promptValue);
      toast("Prompt saved to scene and storyboard.");
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      timelinePromptSave.saving = false;
      button.textContent = "Save Updated Prompt";
      updateI2VPromptSaveButtonState();
      updateMiniMaxPromptSaveButtonState();
    }
  }

  function videoModeDisplayLabel(mode = currentVideoMode(), compact = false) {
    if (mode === "import") return compact ? "Import" : "Import Custom Video";
    if (mode === "id_lora") return compact ? "ID-LoRA" : "ID-LoRA I2V";
    if (mode === "ingredients") return compact ? "Ingredients" : "Ingredients to Video";
    if (mode === "rtv") return compact ? "RTV" : "Reference to Video";
    if (mode === "t2v") return compact ? "T2V" : "Text to Video";
    if (mode === "flf") return compact ? "FLF" : "First Last Frame";
    return compact ? "I2V" : "Image to Video";
  }

  function imageModeDisplayLabel(mode = state.imageModelMode || "zimage", compact = false) {
    if (mode === "flux_klein") return compact ? "Flux/Klein" : "Flux/Klein";
    if (mode === "nano_banana") return compact ? "Nano B" : "NanoBanana";
    if (mode === "ernie_image") return compact ? "Ernie" : "Ernie Image";
    if (mode === "krea2_2pass") return compact ? "Krea 2" : "Krea 2";
    if (mode === "flow_gpt") return compact ? "Flow/GPT" : "Flow/GPT";
    if (mode === "z_enhance") return compact ? "Enhance" : "Enhance";
    return compact ? "ZImage" : "ZImage";
  }

  function i2vVideoSettingsForSegment(segment = activeSegment()) {
    if (!segment || segment.id === activeSegment()?.id) {
      return saveI2VVideoSettingsFromPanel();
    }
    if (segment.use_scene_i2v_video_settings) {
      return cloneI2VVideoSettings(segment.i2v_video_settings || state.i2vVideoSettings);
    }
    return cloneI2VVideoSettings(state.i2vVideoSettings);
  }

  function sceneDisplayName(segment, sceneIndex) {
    const info = segmentIndexInfo(segment);
    if (info.track === "overlay") {
      const index = info.index >= 0 ? info.index : sceneIndex;
      return `Insert ${index + 1}. ${segment?.label || `Insert ${index + 1}`}`;
    }
    const index = sceneIndex >= 0 ? sceneIndex : info.index;
    return `${index + 1}. ${segment?.label || `Scene ${index + 1}`}`;
  }

  function storyboardReferenceDataForSegment(segment) {
    const refs = storyboardReferenceBuilderWithIdLoraRefs(state.fluxReferenceBuilder);
    const info = segmentIndexInfo(segment);
    const sceneKey = segment?.id || "";
    const numberKey = String((info.index >= 0 ? info.index : allEditableSegments().indexOf(segment)) + 1);
    const noCharacterPresent = Boolean(segment?.no_character_present);
    const subjectIds = noCharacterPresent ? [] : logicalSubjectIdsForScene(refs, segment, Number(numberKey) - 1);
    let subjectRefs = subjectIds
      .map((id) => refs.subjects.find((subject) => subject.id === id))
      .filter(Boolean);
    const logicalSubjects = logicalReferenceSubjects(refs);
    if (!noCharacterPresent && !subjectRefs.length && refs.use_subject_reference && logicalSubjects.length === 1) {
      subjectRefs = [logicalSubjects[0]];
    }
    const locId = String(sceneReferenceMapValue(refs.scene_map, segment, Number(numberKey) - 1) || "");
    const locationRef = refs.locations.find((location) => location.id === locId) || null;
    const locationsCleared = Boolean(refs.locations_cleared);
    const locationName = locationsCleared ? "" : String(locationRef?.name || segment?.mapped_location || segment?.location || "").trim();
    const locationDescription = locationsCleared ? "" : String(locationRef?.description || segment?.location_description || "").trim();
    return {
      no_character_present: noCharacterPresent,
      subject_refs: subjectRefs.map((subject) => ({
        id: subject.id,
        name: subject.name,
        description: subject.description,
        reference_type: subject.reference_type || "character",
        minimax_voice: normalizeMiniMaxH3Voice(subject.minimax_voice),
        trigger_phrase: subject.trigger_phrase || "",
        trigger_position: subject.trigger_position || "start",
        image: { ...(subject.image || {}) },
      })),
      location_ref: !locationsCleared && locationRef ? {
        id: locationRef.id,
        name: locationRef.name,
        description: locationRef.description,
        trigger_phrase: locationRef.trigger_phrase || "",
        trigger_position: locationRef.trigger_position || "start",
        image: { ...(locationRef.image || {}) },
      } : (locationName || locationDescription ? {
        id: "",
        name: locationName,
        description: locationDescription,
        image: { path: "", data: "", name: "" },
      } : null),
    };
  }

  function storyboardScenePayload() {
    const defaults = normalizeBuilderStoryboardDefaults(state.builderStoryboardDefaults);
    return allEditableSegments()
      .slice()
      .sort((a, b) => Number(a.start || 0) - Number(b.start || 0))
      .map((segment, sortedIndex) => {
        const info = segmentIndexInfo(segment);
        const index = info.index >= 0 ? info.index : sortedIndex;
        const label = sceneDisplayName(segment, index).replace(/^\d+\.\s*/, "");
        const lyric = String(segment.lyric_text || segment.lyric_note || segment.lyrics || "").trim();
        const videoNotes = String(segment.i2v_notes ?? segment.video_notes ?? "").trim();
        const sceneNotes = String(segment.notes ?? segment.director_note ?? "").trim();
        const imagePrompt = storyboardPromptForSegment(segment);
        const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
        const videoPrompt = String(miniMaxProject
          ? (segment.minimax_h3_prompt || "")
          : (segment.i2v_prompt || segment.t2v_prompt || "")).trim();
        const imageReference = getI2VImageReference(segment);
        const referenceData = storyboardReferenceDataForSegment(segment);
        const mappedExtras = logicalExtraSubjectsForScene(state.fluxReferenceBuilder, segment, index).map(({ extra, interaction }) => ({
          id: String(extra.id || "").trim(),
          name: miniMaxH3ExtraDisplayTitle(extra.title || "background performer"),
          count: Math.max(1, Math.min(100, Math.round(Number(extra.count) || 1))),
          interaction,
          identity: miniMaxH3CompactExtraIdentity(extra.description, extra.title, 150),
        }));
        const promptSummary = storyboardSummaryForSegment(segment, imagePrompt, videoPrompt, referenceData);
        const cutPlan = miniMaxProject
          ? storyboardCutPlanForDuration(timelineSegmentDuration(segment), segment.location_continuous_shot ? 0 : defaults.minimax_h3_cut_frequency)
          : null;
        const subjectRefNames = (referenceData.subject_refs || [])
          .map((subject) => String(subject?.name || "").trim())
          .filter(Boolean);
        const subjects = subjectRefNames.length
          ? Array.from(new Set(subjectRefNames))
          : Array.from(new Set(storyboardSubjectsForSegment(segment).map((item) => String(item || "").trim()).filter(Boolean)));
        const assignedPerformers = selectedPerformerSubjectsForSegment(segment, state.fluxReferenceBuilder)
          .map((subject) => String(subject?.name || "").trim())
          .filter(Boolean);
        const lyricSingers = assignedPerformers.length
          ? Array.from(new Set(assignedPerformers))
          : (Array.isArray(segment.lyric_singers) && !segment.no_character_present ? segment.lyric_singers : []);
        return {
          id: segment.id || `scene_${sortedIndex + 1}`,
          scene_number: sortedIndex + 1,
          label,
          lyrics: lyric,
          lyric_section: String(segment.lyric_section || "").trim(),
          story_beat: String(segment.story_beat || "").trim(),
          flf_start_state: String(segment.flf_start_state || "").trim(),
          flf_transformation: String(segment.flf_transformation || "").trim(),
          flf_end_state: String(segment.flf_end_state || "").trim(),
          flf_carry_forward: String(segment.flf_carry_forward || "").trim(),
          performance_mode: effectiveVideoPerformanceModeForSegment(segment),
          prompt_summary: promptSummary,
          motion_summary: videoNotes,
          lyric_singers: lyricSingers,
          lyric_cue_map: normalizeLyricCueMapForSegment(segment, undefined, { preserveBlank: true }),
          timed_lyric_cue_contract: miniMaxH3VocalCueMapText(segment, miniMaxH3ModeForSegment(segment)),
          performer_assignment: {
            singing: lyricSingers,
            cue_map: normalizeLyricCueMapForSegment(segment, undefined, { preserveBlank: true }),
          },
          speaker_assignments: normalizeMiniMaxSpeakerAssignments(segment.minimax_speaker_assignments),
          lyric_no_lip_sync: Boolean(segmentUsesNoLipSyncPerformance(segment)),
          lyric_instrumental: isInstrumentalLyricText(lyric),
          no_character_present: Boolean(segment.no_character_present),
          facial_performance: String(segment.facial_performance || "").trim(),
          facial_performance_custom: String(segment.facial_performance_custom || "").trim(),
          performance_style: String(segment.performance_style || defaults.performance_style || "").trim(),
          project_video_engine: normalizeProjectVideoEngine(state.projectVideoEngine),
          minimax_h3_mode: miniMaxH3ModeForSegment(segment),
          minimax_h3_audio_mode: miniMaxH3SettingsForSegment(segment).audio_mode,
          ...(cutPlan ? {
            minimax_h3_cut_frequency: cutPlan.frequency,
            cut_plan: cutPlan,
          } : {}),
          video_style: String(segment.minimax_h3_video_style || defaults.video_style || "").trim(),
          video_style_custom: String(segment.minimax_h3_video_style_custom || defaults.video_style_custom || "").trim(),
          temporal_world_effect_override: String(segment.temporal_world_effect_override || "global").trim(),
          temporal_world_effect_custom: String(segment.temporal_world_effect_custom || "").trim(),
          timeline_start: Number(segment.start || 0),
          timeline_end: Number(segment.end || 0),
          exact_duration: Number(timelineSegmentDuration(segment).toFixed(3)),
          video_prompt_type: ["i2v", "id_lora", "t2v", "rtv", "ingredients", "flf"].includes(String(segment.video_prompt_type || "").trim())
            ? String(segment.video_prompt_type || "").trim()
            : currentVideoMode(),
          subjects: segment.no_character_present ? [] : subjects,
          subject_refs: referenceData.subject_refs,
          extra_subjects: segment.no_character_present ? [] : mappedExtras,
          setting: String(referenceData.location_ref?.description || referenceData.location_ref?.name || segment.location || segment.mapped_location || "").trim(),
          location_ref: referenceData.location_ref,
          shot_type: String(segment.shot_type || "").trim(),
          camera_motion: String(segment.camera_motion || segment.motion_preset || "").trim(),
          camera_motion_speed: Number(defaults.camera_motion_speed ?? 4),
          character_motion_speed: Number(defaults.character_motion_speed ?? 4),
          image_prompt: imagePrompt,
          video_prompt: videoPrompt,
          minimax_h3_pass2_prompt: String(segment.minimax_h3_pass2_prompt || ""),
          video_prompt_origin: miniMaxProject
            ? normalizeVideoPromptOrigin(segment.minimax_h3_prompt_origin)
            : normalizeVideoPromptOrigin(segment.i2v_prompt_origin),
          image_path: imageReference.path || selectedSegmentImagePath(segment),
          image_data: imageReference.data || "",
          notes: sceneNotes,
          timeline_note: String(segment.timeline_note ?? ""),
          audio_direction: String(segment.audio_direction || "").trim(),
          continuity: String(segment.continuity || "").trim(),
        };
      });
  }

  function wizardStoryboardState(scenes, options = {}) {
    const defaults = normalizeBuilderStoryboardDefaults(state.builderStoryboardDefaults);
    const videoMode = options.videoMode || currentVideoMode();
    const imageMode = options.imageMode || state.imageModelMode || "zimage";
    const promptMode = options.promptMode || "image";
    return {
      projectFolder: activeProjectFolderForSave(),
      projectVideoEngine: normalizeProjectVideoEngine(state.projectVideoEngine),
      mode: promptMode === "video" ? "image_to_video_prep" : "storyboard_prompts",
      performanceMode: normalizeVideoType(state.videoType),
      videoType: normalizeVideoType(state.videoType),
      shortFilmPlanningMode: defaults.short_film_planning_mode,
      videoPromptType: videoMode,
      miniMaxH3Mode: normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3" ? normalizeMiniMaxH3Mode(options.miniMaxH3Mode || state.miniMaxH3Settings?.video_mode) : "",
      miniMaxH3AudioMode: normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3" ? normalizeMiniMaxH3AudioMode(state.miniMaxH3Settings?.audio_mode) : "",
      imageMode,
      imageModeLabel: imageModeDisplayLabel(imageMode),
      cameraFlow: defaults.camera_flow || "balanced",
      imageShotFlow: defaults.image_shot_flow || "intimate",
      imageAesthetic: defaults.image_aesthetic || "",
      videoStyle: defaults.video_style || "",
      videoStyleCustom: defaults.video_style_custom || "",
      temporalWorldEffect: defaults.temporal_world_effect || "",
      temporalWorldEffectCustom: defaults.temporal_world_effect_custom || "",
      temporalAllowBackgroundExtras: defaults.temporal_allow_background_extras !== false,
      temporalBackgroundIntensity: Number(defaults.temporal_background_intensity ?? 8),
      temporalEnvironmentTimePassage: defaults.temporal_environment_time_passage !== false,
      temporalProtectedCharacters: defaults.temporal_protected_characters || "all_referenced",
      temporalProtectedCustom: defaults.temporal_protected_custom || "",
      fxPreset: defaults.fx_preset || "",
      fxCustomJson: defaults.fx_custom_json || "",
      globalConsistencyPhrase: defaults.global_consistency_phrase || "",
      performanceStyle: defaults.performance_style || "",
      facialPerformance: state.defaultFacialPerformance || "",
      facialPerformanceCustom: state.defaultFacialPerformanceCustom || "",
      cameraMotionSpeed: Number(defaults.camera_motion_speed ?? 4),
      characterMotionSpeed: Number(defaults.character_motion_speed ?? 4),
      cutFrequency: storyboardCutFrequencyValue(defaults.minimax_h3_cut_frequency),
      motion_defaults: {
        camera_motion_speed: Number(defaults.camera_motion_speed ?? 4),
        character_motion_speed: Number(defaults.character_motion_speed ?? 4),
        minimax_h3_cut_frequency: storyboardCutFrequencyValue(defaults.minimax_h3_cut_frequency),
        camera_guidance: defaults.camera_guidance || "",
        character_guidance: defaults.character_guidance || "",
      },
      scenes,
      storyLayer: normalizeBuilderStoryLayer(state.builderStoryLayer),
      referenceBuilder: storyboardReferenceBuilderWithIdLoraRefs(state.fluxReferenceBuilder),
    };
  }

  return {
    activeVideoOutputFolder, collectedSceneVideoFolder, i2vImagesFolder, i2vVideoSettingsForSegment,
    i2vVideoSettingsPayload, imageModeDisplayLabel, saveTimelinePrompt, sceneDisplayName,
    sceneVideoDetailsHtml, storyboardReferenceDataForSegment, storyboardScenePayload, timelineSegmentLabel,
    videoModeDisplayLabel, wizardStoryboardState,
  };
}
