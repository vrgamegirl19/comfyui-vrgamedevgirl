import { getJson } from "./comfy_api.mjs";
import {
  BAD_I2V_UNET_ALIASES,
  DEFAULT_I2V_DIFFUSION_MODEL,
  DEFAULT_I2V_UNET,
  DEFAULT_NON_VISION_GEMMA_MODEL,
  REQUIRED_LTX_ID_LORA,
  REQUIRED_LTX_INGREDIENTS_LORA,
  REQUIRED_LTX_MSR_LORA,
} from "./constants.mjs";
import { makeButton, toast } from "./controls.mjs";
import { setI2VStrengthPair } from "./image_panels.mjs";
import { imageFileFromDrop } from "./media_import.mjs";
import { cloneMiniMaxH3Settings, DEFAULT_MINIMAX_H3_SETTINGS } from "./minimax_h3.mjs";
import { cloneI2VVideoSettings } from "./model_settings.mjs";
import { chooseBatchModeAction } from "./project_actions.mjs";
import { normalizeBatchScope } from "./timeline_state.mjs";

function showLegacyPromptCreatorNotice() {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:20px;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(620px,calc(100vw - 40px));border:1px solid #b45309;border-radius:10px;background:#111827;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.6);padding:18px;display:flex;flex-direction:column;gap:13px;";
    const heading = document.createElement("div");
    heading.textContent = "Prompt Creator (Legacy)";
    heading.style.cssText = "font-size:19px;font-weight:900;color:#fde68a;";
    const note = document.createElement("div");
    note.innerHTML = [
      "<strong>Prompt Creator is the older planning workflow.</strong>",
      "For new projects, use Storyboard Builder. It is the newer workflow for keeping story direction, scene beats, references, image defaults, motion defaults, and prompts together in the same project.",
      "Continue only if you need an older Prompt Creator project or its file-based prompt workflow.",
    ].map((line) => `<div style="margin-top:7px;">${line}</div>`).join("");
    note.style.cssText = "font-size:13px;color:#e5e7eb;line-height:1.5;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:3px;";
    const continueLegacy = makeButton("Continue to Prompt Creator");
    const continueBuilder = makeButton("Continue to Video Builder", "primary");
    const finish = (value) => {
      backdrop.remove();
      resolve(value);
    };
    continueLegacy.onclick = () => finish("prompt_creator");
    continueBuilder.onclick = () => finish("video_builder");
    backdrop.addEventListener("click", (event) => {
      if (event.target === backdrop) finish("video_builder");
    });
    backdrop.addEventListener("keydown", (event) => {
      if (event.key === "Escape") finish("video_builder");
    });
    actions.append(continueLegacy, continueBuilder);
    box.append(heading, note, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    backdrop.tabIndex = -1;
    backdrop.focus();
  });
}

function renderSearchableSuggestions(picker, onSelect = null, renderOptions = {}) {
  const useFilter = renderOptions.useFilter ?? picker.suggestionUseFilter ?? false;
  picker.suggestionUseFilter = Boolean(useFilter);
  const query = useFilter ? String(picker.input.value || "").trim().toLowerCase() : "";
  const choices = picker.options || [];
  const matches = choices
    .filter((name) => !query || String(name).toLowerCase().includes(query))
    .slice(0, 50);
  picker.matches = matches;
  if (!matches.length) picker.activeIndex = -1;
  else if (!Number.isInteger(picker.activeIndex) || picker.activeIndex < 0 || picker.activeIndex >= matches.length) picker.activeIndex = 0;
  picker.list.textContent = "";
  const choose = (name) => {
    picker.input.value = name;
    picker.list.style.display = "none";
    onSelect?.();
  };
  for (const [index, name] of matches.entries()) {
    const item = document.createElement("button");
    item.type = "button";
    item.textContent = name;
    item.title = name;
    item.style.cssText = `display:block;width:100%;text-align:left;border:0;background:${index === picker.activeIndex ? "#0e7490" : "#18181b"};color:#fafafa;padding:7px 8px;font-size:12px;line-height:1.35;cursor:pointer;white-space:normal;overflow-wrap:anywhere;`;
    item.onmouseenter = () => {
      picker.activeIndex = index;
      Array.from(picker.list.children).forEach((button, childIndex) => {
        button.style.background = childIndex === picker.activeIndex ? "#0e7490" : "#18181b";
      });
    };
    item.onpointerdown = (event) => {
      event.preventDefault();
      choose(name);
    };
    item.onclick = () => choose(name);
    picker.list.append(item);
  }
  picker.list.style.display = matches.length ? "block" : "none";
  const activeButton = picker.list.children[picker.activeIndex];
  activeButton?.scrollIntoView?.({ block: "nearest" });
}

export function wireSearchablePicker(picker, onChange = null) {
  const isPrintableSearchKey = (event) => (
    event.key?.length === 1
    && !event.ctrlKey
    && !event.metaKey
    && !event.altKey
  );
  const shouldClearStaleValueForSearch = () => {
    const value = String(picker.input.value || "").trim();
    if (!value || value === "[none]") return true;
    return !(picker.options || []).includes(value);
  };
  picker.input.addEventListener("focus", () => renderSearchableSuggestions(picker, onChange, { useFilter: false }));
  picker.input.addEventListener("input", () => {
    picker.activeIndex = 0;
    renderSearchableSuggestions(picker, onChange, { useFilter: true });
    onChange?.();
  });
  picker.input.addEventListener("keydown", (event) => {
    if (isPrintableSearchKey(event) && !picker.suggestionUseFilter && shouldClearStaleValueForSearch()) {
      picker.input.value = "";
    }
    if (picker.list.style.display !== "block") {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        renderSearchableSuggestions(picker, onChange, { useFilter: false });
      }
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      picker.activeIndex = Math.min((picker.matches?.length || 1) - 1, (picker.activeIndex < 0 ? 0 : picker.activeIndex + 1));
      renderSearchableSuggestions(picker, onChange);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      picker.activeIndex = Math.max(0, picker.activeIndex <= 0 ? 0 : picker.activeIndex - 1);
      renderSearchableSuggestions(picker, onChange);
    } else if (event.key === "Enter") {
      const selected = picker.matches?.[picker.activeIndex];
      if (selected) {
        event.preventDefault();
        picker.input.value = selected;
        picker.list.style.display = "none";
        onChange?.();
      }
    } else if (event.key === "Escape") {
      picker.list.style.display = "none";
    }
  });
  picker.input.addEventListener("blur", () => {
    setTimeout(() => { picker.list.style.display = "none"; }, 180);
  });
}

function basenameOnly(value) {
  return String(value || "").replaceAll("\\", "/").split("/").pop();
}

export function chooseModelValue(options = [], current = "", preferred = []) {
  const values = (options || []).filter((item) => String(item || "").trim());
  if (!values.length) return "";
  const exact = values.find((item) => item === current);
  if (exact) return exact;
  const currentBase = basenameOnly(current);
  if (currentBase) {
    const sameBase = values.find((item) => basenameOnly(item) === currentBase);
    if (sameBase) return sameBase;
  }
  for (const item of preferred) {
    const direct = values.find((value) => value === item);
    if (direct) return direct;
    const base = basenameOnly(item);
    const sameBase = values.find((value) => basenameOnly(value) === base);
    if (sameBase) return sameBase;
  }
  for (const item of preferred) {
    const needle = basenameOnly(item).toLowerCase().replace(/\.(safetensors|gguf|ckpt)$/i, "");
    const partial = values.find((value) => String(value || "").toLowerCase().includes(needle));
    if (partial) return partial;
  }
  return values[0] || "";
}

export function createModelPickers({
  activeI2VVideoSettings, activeSegment, autoSaveSessionQuiet, batchScopeChoices, buildFullFLFVideoPipeline,
  currentVideoMode, droppedSceneImageSource, ernieClipPicker, ernieGemmaModelSelect, ernieLoraSlots,
  ernieMmprojSelect, ernieTextGemmaModelSelect, ernieUnetPicker, ernieVaePicker, flfTransitionLoraNote,
  fluxClipPicker, fluxGemmaModelSelect, fluxLoraSlots, fluxMmprojSelect, fluxUnetPicker, fluxVaePicker,
  gemmaModelSelect, i2vAudioVaePicker, i2vClip1Picker, i2vClip2Picker, i2vDiffusionModelPicker,
  i2vGemmaModelSelect, i2vLoraCount, i2vLoraPanel, i2vLoraRows, i2vLoraSlots, i2vMmprojSelect,
  i2vTextGemmaModelSelect, i2vUnetPicker, i2vUpscalePicker, i2vUseLora, i2vVaePicker, krea2TwoPassClipPicker,
  krea2TwoPassLoraSlots, krea2TwoPassUnetPicker, krea2TwoPassVaePicker, loadVisionReferenceFile,
  ltxIdLoraPicker, ltxIngredientsLoraPicker, ltxMsrLoraPicker, miniMaxAdvancedLatentUpscalerPicker,
  miniMaxAudioVaePicker, miniMaxClipPicker, miniMaxDiffusionModelPicker, miniMaxGemmaModelSelect,
  miniMaxLoraSlots, miniMaxMmprojSelect, miniMaxTextGemmaModelSelect, miniMaxThreePassLoraPicker,
  miniMaxTurboLoraPicker, miniMaxTwoPassLatentUpscalerPicker, miniMaxTwoPassLoraPicker, miniMaxVideoVaePicker,
  mmprojSelect, nbGemmaModelSelect, nbMmprojSelect, openPromptCreatorPanel, pushHistory, renderList,
  saveI2VVideoSettingsFromPanel, saveMiniMaxH3SettingsFromPanel, setVisionReferenceSource, state,
  syncBuilderLlmModelSelectsFromRunner, syncI2VVideoSettingsPanel, syncKrea2TwoPassLlmSelectsFromShared,
  syncMiniMaxH3Panel, syncMiniMaxLlmSelectsFromShared, t2iTextGemmaModelSelect, zClipPicker,
  zEnhanceClipPicker, zEnhanceGemmaModelSelect, zEnhanceLoraSlots, zEnhanceMmprojSelect, zEnhanceUnetPicker,
  zEnhanceVaePicker, zLoraSlots, zUnetPicker, zVaePicker,
}) {
  async function confirmOpenLegacyPromptCreator() {
    const choice = await showLegacyPromptCreatorNotice();
    if (choice === "prompt_creator") openPromptCreatorPanel();
  }
  function setSceneI2VVideoSettingsEnabled(enabled) {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    saveI2VVideoSettingsFromPanel();
    segment.use_scene_i2v_video_settings = Boolean(enabled);
    if (segment.use_scene_i2v_video_settings && !segment.i2v_video_settings) {
      segment.i2v_video_settings = cloneI2VVideoSettings(state.i2vVideoSettings);
    }
    syncI2VVideoSettingsPanel();
    renderList();
    toast(segment.use_scene_i2v_video_settings ? "This scene now has custom video models, settings, and LoRAs." : "This scene is using global video models, settings, and LoRAs again.");
  }
  async function setSceneMiniMaxH3SettingsEnabled(enabled) {
    const segment = activeSegment();
    if (!segment) return;
    pushHistory();
    saveMiniMaxH3SettingsFromPanel();
    const useSceneSettings = Boolean(enabled);
    segment.use_scene_minimax_h3_settings = useSceneSettings;
    if (useSceneSettings) {
      segment.minimax_h3_settings = cloneMiniMaxH3Settings(state.miniMaxH3Settings);
      segment.minimax_h3_mode = segment.minimax_h3_settings.video_mode;
    }
    syncMiniMaxH3Panel();
    renderList();
    await autoSaveSessionQuiet(useSceneSettings ? "MiniMax H3 scene settings locked" : "MiniMax H3 scene settings returned to global");
    toast(useSceneSettings
      ? "This scene now has its own locked MiniMax mode, models, and video settings."
      : "This scene is following the project-global MiniMax mode, models, and video settings again.");
  }
  function wireVisionReferenceDrop(dropElement, options = {}) {
    dropElement.addEventListener("dragover", (event) => {
      const types = Array.from(event.dataTransfer?.types || []);
      if (!types.includes("Files") && !types.includes("application/x-vrgdg-segment-id")) return;
      event.preventDefault();
      event.stopPropagation();
      dropElement.style.borderColor = "#a3e635";
    });
    dropElement.addEventListener("dragleave", () => {
      dropElement.style.borderColor = "#155e75";
    });
    dropElement.addEventListener("drop", (event) => {
      const sceneSource = droppedSceneImageSource(event);
      if (sceneSource) {
        event.preventDefault();
        event.stopPropagation();
        dropElement.style.borderColor = "#155e75";
        setVisionReferenceSource({ ...sceneSource, forT2V: Boolean(options.forT2V) }).catch((error) => toast(String(error?.message || error), true));
        return;
      }
      const file = imageFileFromDrop(event);
      if (!file) return;
      event.preventDefault();
      event.stopPropagation();
      dropElement.style.borderColor = "#155e75";
      loadVisionReferenceFile(file, { forT2V: Boolean(options.forT2V) });
    });
  }

  async function refreshGemmaChoices() {
    const data = await getJson("/vrgdg/music_builder/gemma_choices");
    const models = data.models || [];
    const mmproj = data.mmproj || [];
    const validMmproj = mmproj.filter((item) => item && !/^\[No mmproj/i.test(item));
    const singleMmproj = validMmproj.length === 1 ? validMmproj[0] : "";
    for (const select of [t2iTextGemmaModelSelect, gemmaModelSelect, ernieTextGemmaModelSelect, ernieGemmaModelSelect, zEnhanceGemmaModelSelect, i2vTextGemmaModelSelect, i2vGemmaModelSelect, miniMaxTextGemmaModelSelect, miniMaxGemmaModelSelect, fluxGemmaModelSelect, nbGemmaModelSelect]) {
      select.textContent = "";
      for (const model of models) {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        select.append(option);
      }
    }
    const preferredNonVision = models.find((model) => model === DEFAULT_NON_VISION_GEMMA_MODEL)
      || models.find((model) => /supergemma4.*fast.*q4_k_m/i.test(model))
      || models.find((model) => /supergemma/i.test(model))
      || "";
    if (preferredNonVision) {
      for (const select of [t2iTextGemmaModelSelect, ernieTextGemmaModelSelect, i2vTextGemmaModelSelect, miniMaxTextGemmaModelSelect]) {
        select.value = preferredNonVision;
      }
    }
    for (const select of [mmprojSelect, ernieMmprojSelect, zEnhanceMmprojSelect, i2vMmprojSelect, miniMaxMmprojSelect, fluxMmprojSelect, nbMmprojSelect]) {
      const previousValue = select.value;
      select.textContent = "";
      for (const item of mmproj) {
        const option = document.createElement("option");
        option.value = item;
        option.textContent = item;
        select.append(option);
      }
      if (previousValue && mmproj.includes(previousValue)) {
        select.value = previousValue;
      } else if (singleMmproj) {
        select.value = singleMmproj;
      }
    }
    syncMiniMaxLlmSelectsFromShared();
    syncKrea2TwoPassLlmSelectsFromShared();
    syncBuilderLlmModelSelectsFromRunner();
  }

  async function refreshLoraChoices() {
    const data = await getJson("/vrgdg/workflow_runner/lora_list");
    const loras = data.loras || ["[none]"];
    for (const slot of [...zLoraSlots, ...ernieLoraSlots, ...fluxLoraSlots, ...i2vLoraSlots, ...zEnhanceLoraSlots, ...krea2TwoPassLoraSlots, ...miniMaxLoraSlots, { picker: miniMaxTwoPassLoraPicker }, { picker: miniMaxThreePassLoraPicker }]) {
      const current = slot.picker.input.value || "[none]";
      slot.picker.options = loras;
      slot.picker.input.value = loras.includes(current) ? current : current;
    }
    ltxMsrLoraPicker.options = loras;
    if (!ltxMsrLoraPicker.input.value) ltxMsrLoraPicker.input.value = REQUIRED_LTX_MSR_LORA;
    ltxIngredientsLoraPicker.options = loras;
    if (!ltxIngredientsLoraPicker.input.value) ltxIngredientsLoraPicker.input.value = REQUIRED_LTX_INGREDIENTS_LORA;
    ltxIdLoraPicker.options = loras;
    if (!ltxIdLoraPicker.input.value) ltxIdLoraPicker.input.value = REQUIRED_LTX_ID_LORA;
    miniMaxTurboLoraPicker.options = loras;
    if (!miniMaxTurboLoraPicker.input.value) {
      miniMaxTurboLoraPicker.input.value = DEFAULT_MINIMAX_H3_SETTINGS.turbo_lora_name;
    }
    for (const picker of [miniMaxTwoPassLoraPicker, miniMaxThreePassLoraPicker]) {
      picker.options = loras;
    }
  }

  async function confirmAndRunFullFLFBuild() {
    if (currentVideoMode() !== "flf") {
      toast("Choose Video → First Last Frame before using Build Full FLF Video.", true);
      return;
    }
    const scopeChoices = batchScopeChoices();
    const action = await chooseBatchModeAction({
      title: "Build Full FLF Video?",
      intro: "Runs the complete FLF dependency chain without further prompts: fills missing endpoint beats, creates the optimized image chain, creates each vision video prompt at the correct render step, renders every scene, extracts rendered final frames for chaining, and stitches the final video.",
      confirmLabel: "Build Full FLF Video",
      returnAll: true,
      choices: [
        {
          value: "resume_missing",
          label: "Resume missing (recommended)",
          description: "Keep completed beats, images, prompts, and videos. Create only what is missing, then stitch the final video.",
        },
        {
          value: "redo_images_videos",
          label: "Redo images and videos",
          description: "Keep storyboard planning and saved image prompts, but regenerate the FLF image chain and all scene videos.",
        },
        {
          value: "redo_videos",
          label: "Redo videos only",
          description: "Keep endpoint beats and images, then create new video versions and stitch them.",
        },
      ],
      extraGroups: scopeChoices.length ? [{
        key: "sceneScope",
        label: "Scenes to build",
        description: "All scenes produces and stitches the full project. Selected-scenes mode does not stitch.",
        choices: scopeChoices,
      }] : [],
    });
    if (!action?.mode) return;
    await buildFullFLFVideoPipeline({
      sceneScope: normalizeBatchScope(action.sceneScope),
      redoImages: action.mode === "redo_images_videos",
      redoVideos: action.mode === "redo_images_videos" || action.mode === "redo_videos",
      maxAttempts: 3,
    });
  }

  async function refreshModelChoices() {
    const data = await getJson("/vrgdg/workflow_runner/i2v_choices");
    const setOptions = (picker, options, preferred = []) => {
      const preferredList = Array.isArray(preferred) ? preferred : [preferred];
      const values = Array.from(new Set((options || []).filter((item) => String(item || "").trim())));
      picker.options = values;
      const current = BAD_I2V_UNET_ALIASES.has(picker.input.value) ? "" : picker.input.value;
      picker.input.value = chooseModelValue(values, current, preferredList);
    };
    const setMiniMaxOptions = (picker, options, fallback) => {
      const values = Array.from(new Set((options || []).filter((item) => String(item || "").trim())));
      picker.options = values;
      const current = String(picker.input.value || fallback || "").trim();
      const exactOrSameBase = values.find((item) => item === current || basenameOnly(item) === basenameOnly(current));
      // ComfyUI's newer io.Combo nodes validate against their live option list.
      // Do not preserve a stale saved filename when it is no longer available.
      picker.input.value = exactOrSameBase || values[0] || fallback;
    };
    setOptions(i2vUnetPicker, data.video_gguf_unets || data.unets, DEFAULT_I2V_UNET);
    const ltx25Selected = (state.i2vVideoSettings?.ltx_version || "2.5") !== "2.3";
    setOptions(i2vDiffusionModelPicker, data.video_diffusion_models || data.unets, ltx25Selected ? "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" : DEFAULT_I2V_DIFFUSION_MODEL);
    setOptions(i2vVaePicker, data.vae, ltx25Selected ? "ltx-2.5-video-vae-conv-bf16.safetensors" : "LTX23_video_vae_bf16.safetensors");
    setOptions(i2vClip1Picker, data.clip, ltx25Selected ? "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" : "gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors");
    setOptions(i2vClip2Picker, data.clip, "ltx-2.3_text_projection_bf16.safetensors");
    setOptions(i2vUpscalePicker, data.upscale_models, ltx25Selected ? "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" : "ltx-2.3-spatial-upscaler-x2-1.1.safetensors");
    setOptions(i2vAudioVaePicker, data.vae, ltx25Selected ? "ltx-2.5-audio-vae-bf16.safetensors" : "LTX23_audio_vae_bf16.safetensors");
    const miniMaxDiffusionChoices = (data.video_diffusion_models || data.unets || [])
      .filter((item) => !/\.gguf$/i.test(String(item || "").trim()));
    setMiniMaxOptions(miniMaxDiffusionModelPicker, miniMaxDiffusionChoices, DEFAULT_MINIMAX_H3_SETTINGS.diffusion_model_name);
    setMiniMaxOptions(miniMaxClipPicker, data.clip, DEFAULT_MINIMAX_H3_SETTINGS.clip_name);
    setMiniMaxOptions(miniMaxVideoVaePicker, data.vae, DEFAULT_MINIMAX_H3_SETTINGS.video_vae_name);
    setMiniMaxOptions(miniMaxAudioVaePicker, data.vae, DEFAULT_MINIMAX_H3_SETTINGS.audio_vae_name);
    const miniMaxLatentUpscalerChoices = (data.upscale_models || [])
      .filter((item) => /minimax_h3_latent_upscaler_3d/i.test(String(item || "")));
    setMiniMaxOptions(miniMaxTwoPassLatentUpscalerPicker, miniMaxLatentUpscalerChoices, DEFAULT_MINIMAX_H3_SETTINGS.two_pass_latent_upscaler_name);
    setMiniMaxOptions(miniMaxAdvancedLatentUpscalerPicker, miniMaxLatentUpscalerChoices, DEFAULT_MINIMAX_H3_SETTINGS.two_pass_latent_upscaler_name);
    saveMiniMaxH3SettingsFromPanel();
    setOptions(fluxUnetPicker, data.unets, ["flux\\flux-2-klein-4b-fp8.safetensors", "flux-2-klein-4b-fp8.safetensors"]);
    setOptions(fluxClipPicker, data.clip, ["qwen_3_4b.safetensors", "flux\\qwen_3_4b.safetensors"]);
    setOptions(fluxVaePicker, data.vae, ["flux\\flux2-vae.safetensors", "flux2-vae.safetensors"]);
    setOptions(zUnetPicker, data.unets, "z_image_turbo_bf16.safetensors");
    setOptions(zClipPicker, data.clip, "qwen_3_4b.safetensors");
    setOptions(zVaePicker, data.vae, "ae.safetensors");
    setOptions(zEnhanceUnetPicker, data.unets, "z_image_turbo_bf16.safetensors");
    setOptions(zEnhanceClipPicker, data.clip, "qwen_3_4b.safetensors");
    setOptions(zEnhanceVaePicker, data.vae, "ae.safetensors");
    setOptions(ernieUnetPicker, data.unets, ["ernie\\ernie-image-turbo.safetensors", "ernie-image-turbo.safetensors"]);
    setOptions(ernieClipPicker, data.clip, ["ministral-3-3b.safetensors", "ernie\\ministral-3-3b.safetensors"]);
    setOptions(ernieVaePicker, data.vae, ["flux\\flux2-vae.safetensors", "flux2-vae.safetensors"]);
    setOptions(krea2TwoPassUnetPicker, data.unets, "krea2_turbo_fp8_scaled.safetensors");
    setOptions(krea2TwoPassClipPicker, data.clip, "qwen3vl_4b_fp8_scaled.safetensors");
    setOptions(krea2TwoPassVaePicker, data.vae, "qwen_image_vae.safetensors");
  }

  async function loadCustomModelRootSetting() {
    try {
      const data = await getJson("/vrgdg/workflow_runner/model_root");
      state.customModelsRoot = data.models_root || "";
    } catch (error) {
      console.warn("[VRGDG Music Builder] Could not load custom models root:", error);
    }
  }

  function updateI2VLoraVisibility() {
    const count = Math.max(0, Math.min(4, Number(i2vLoraCount.value || 0)));
    const isSinglePassMode = currentVideoMode() === "flf" || (currentVideoMode() === "rtv" && (activeI2VVideoSettings()?.ltx_version || "2.5") === "2.3");
    i2vLoraPanel.style.display = i2vUseLora.input.checked ? "flex" : "none";
    flfTransitionLoraNote.style.display = currentVideoMode() === "flf" ? "block" : "none";
    i2vLoraRows.style.display = i2vUseLora.input.checked && count > 0 ? "flex" : "none";
    i2vLoraSlots.forEach((slot, index) => {
      slot.row.style.display = index < count ? "grid" : "none";
      slot.row.style.gridTemplateColumns = isSinglePassMode ? "1fr 84px" : "1fr 84px 84px";
      if (slot.firstPassLabel) slot.firstPassLabel.textContent = isSinglePassMode ? "Strength" : "Pass 1";
      if (slot.secondPassField) slot.secondPassField.style.display = isSinglePassMode ? "none" : "flex";
    });
  }
  function wireI2VStrengthPair(slider, input) {
    slider.addEventListener("input", () => {
      setI2VStrengthPair(slider, input, slider.value);
      saveI2VVideoSettingsFromPanel();
    });
    input.addEventListener("input", () => {
      setI2VStrengthPair(slider, input, input.value);
      saveI2VVideoSettingsFromPanel();
    });
    input.addEventListener("change", () => {
      setI2VStrengthPair(slider, input, input.value);
      saveI2VVideoSettingsFromPanel();
    });
  }

  return {
    confirmAndRunFullFLFBuild, confirmOpenLegacyPromptCreator, loadCustomModelRootSetting,
    refreshGemmaChoices, refreshLoraChoices, refreshModelChoices, setSceneI2VVideoSettingsEnabled,
    setSceneMiniMaxH3SettingsEnabled, updateI2VLoraVisibility, wireI2VStrengthPair, wireVisionReferenceDrop,
  };
}
