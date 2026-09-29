import { normalizeOwnServerTimeoutMinutes, postJson } from "./comfy_api.mjs";
import { DEFAULT_NB_IMAGE_MODEL } from "./constants.mjs";
import { toast } from "./controls.mjs";
import { createBaseProgressWindow } from "./dialogs.mjs";
import {
  normalizeGemmaContextLimit,
  normalizeGemmaGpuLayers,
  normalizeLmStudioContextLimit,
  normalizeOutputTokenLimit,
} from "./prompt_text.mjs";

export function defaultNBImageSettings() {
  return {
    api_key: "",
    model: DEFAULT_NB_IMAGE_MODEL,
    use_text_only_gemma_prompt: false,
    use_director_notes: false,
  };
}

function referenceImagePayload(image = {}) {
  return {
    image_path: String(image?.path || "").trim(),
    image_data: String(image?.data || "").trim(),
  };
}

export function hasReferenceImage(image = {}) {
  const payload = referenceImagePayload(image);
  return Boolean(payload.image_path || payload.image_data);
}

export function isLikelyEmbeddingModelId(modelId) {
  const text = String(modelId || "").toLowerCase();
  return text.includes("embedding") || text.includes("embed") || text.includes("nomic-embed") || text.includes("bge-") || text.includes("e5-");
}

export function createLlmRunner({
  autoSaveSessionQuiet, createFluxPromptButton, createI2VButton, createNBPromptButton, createT2IButton,
  currentVideoMode, ernieCreateT2IButton, ernieGemmaModelSelect, ernieMmprojSelect, flowGptCreatePromptButton,
  fluxGemmaModelSelect, fluxMmprojSelect, gemmaModelSelect, gemmaT2IAllButton, gemmaVideoAllButton,
  i2vGemmaModelSelect, i2vMmprojSelect, i2vTextGemmaModelSelect, krea2TwoPassCreateT2IButton, mmprojSelect,
  nbGemmaModelSelect, nbMmprojSelect, state, t2iTextGemmaModelSelect, zEnhanceGemmaButton,
}) {
  function textGemmaRunnerPayload() {
    const qwenLocal = state.textGemmaRunner === "qwen_local";
    return {
      text_runner: state.textGemmaRunner || "builtin",
      qwen_model_file: state.qwenModelFile || "",
      qwen_mmproj_file: state.qwenMmprojFile || "",
      gemma_model_file: state.gemmaModelFile || "",
      model_file: qwenLocal ? (state.qwenModelFile || "") : "",
      n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
      gemma_output_token_limit: normalizeOutputTokenLimit(state.gemmaOutputTokenLimit),
      n_gpu_layers: normalizeGemmaGpuLayers(state.gemmaGpuLayers),
      lmstudio_base_url: state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1",
      lmstudio_model: state.lmStudioModel || "",
      lmstudio_api_key: state.lmStudioApiKey || "",
      lmstudio_context_limit: normalizeLmStudioContextLimit(state.lmStudioContextLimit),
      lmstudio_output_token_limit: normalizeOutputTokenLimit(state.lmStudioOutputTokenLimit),
      llm_api_provider: state.llmApiProvider || "openai",
      llm_api_model: state.llmApiModel || "",
      llm_api_key_project: state.llmApiKeyProject || "",
      llm_api_key: state.llmApiKey || "",
      own_server_url: state.ownServerUrl || "http://127.0.0.1:8000/v1",
      own_server_model: state.ownServerModel || "",
      own_server_api_key: state.ownServerApiKey || "",
      own_server_api_key_project: state.ownServerApiKeyProject || "",
      own_server_output_token_limit: normalizeOutputTokenLimit(state.ownServerOutputTokenLimit),
      own_server_timeout: normalizeOwnServerTimeoutMinutes(state.ownServerTimeoutMinutes) * 60,
    };
  }

  function llmApiVisionModelSelected() {
    return Boolean(String(state.llmApiModel || "").trim());
  }

  function gemmaRunnerLabel(options = {}) {
    if (options.forceBuiltin) return options.vision ? "Built-in GGUF vision" : "Built-in GGUF";
    if (state.textGemmaRunner === "llm_api") return options.vision ? "LLM API vision" : "LLM API";
    if (state.textGemmaRunner === "own_server") return options.vision ? "Custom Server vision" : "Custom Server";
    if (state.textGemmaRunner === "qwen_local") return options.vision ? "Qwen Local vision" : "Qwen Local";
    if (options.vision) return state.textGemmaRunner === "lm_studio" ? "LM Studio vision" : "Built-in GGUF vision";
    return state.textGemmaRunner === "lm_studio" ? "LM Studio" : "Gemma Local";
  }

  function gemmaRunnerLine(options = {}) {
    return `Runner: ${gemmaRunnerLabel(options)}`;
  }

  function promptRunnerActionName() {
    if (state.textGemmaRunner === "lm_studio") return "LM Studio";
    if (state.textGemmaRunner === "llm_api") return "LLM API";
    if (state.textGemmaRunner === "own_server") return "Custom Server";
    if (state.textGemmaRunner === "qwen_local") return "Qwen Local";
    return "Gemma Local";
  }

  function runnerAwareLlmText(value) {
    return String(value || "")
      .replace(/\b(?:Vision Gemma|Gemma Vision|Gemma vision)\b/gi, gemmaRunnerLabel({ vision: true }))
      .replace(/\bGemma Local\b/gi, promptRunnerActionName())
      .replace(/\bGemma4?\b/g, promptRunnerActionName())
      .replace(/\bGemma\b/g, promptRunnerActionName())
      .replace(/\bAPI LLM\b/g, "LLM API")
      .replace(/\bOwn server\b/gi, "Custom Server");
  }

  function createProgressWindow(title, options = {}) {
    const runnerAware = options.runnerAware !== false;
    const displayedTitle = runnerAware ? runnerAwareLlmText(title) : title;
    const progress = createBaseProgressWindow(displayedTitle, options);
    if (!runnerAware) return progress;
    return {
      ...progress,
      set(message, percent = null) {
        progress.set(runnerAwareLlmText(message), percent);
      },
      setHtml(html, percent = null) {
        progress.setHtml(runnerAwareLlmText(html), percent);
      },
    };
  }

  function updatePromptRunnerButtonLabels() {
    const runner = promptRunnerActionName();
    gemmaT2IAllButton.textContent = `${runner} T2I All`;
    gemmaVideoAllButton.textContent = `${runner} Video All`;
    createT2IButton.textContent = `${runner} T2I`;
    ernieCreateT2IButton.textContent = `${runner} T2I`;
    krea2TwoPassCreateT2IButton.textContent = `${runner} T2I`;
    createFluxPromptButton.textContent = `${runner} Flux Prompt`;
    createNBPromptButton.textContent = `${runner} NB Prompt`;
    flowGptCreatePromptButton.textContent = `${runner} Browser Prompt`;
    zEnhanceGemmaButton.textContent = `${runner} Enhance Prompt`;
    const mode = currentVideoMode();
    createI2VButton.textContent = mode === "id_lora"
      ? `${runner} ID Script`
      : mode === "ingredients"
        ? `${runner} Ingredients Video`
        : mode === "flf"
          ? `${runner} First/Last Prompt`
          : mode === "rtv"
            ? `${runner} Reference Video`
            : mode === "t2v"
              ? `${runner} T2V`
              : `${runner} I2V`;
  }

  function referenceDescriptionVisionModel() {
    return String(i2vGemmaModelSelect.value || fluxGemmaModelSelect.value || nbGemmaModelSelect.value || gemmaModelSelect.value || ernieGemmaModelSelect.value || "").trim();
  }

  function referenceDescriptionMmproj() {
    return String(i2vMmprojSelect.value || fluxMmprojSelect.value || nbMmprojSelect.value || mmprojSelect.value || ernieMmprojSelect.value || "").trim();
  }

  async function describeReferenceImageWithGemma(target, referenceType = "subject", options = {}) {
    const image = target?.image || {};
    if (!hasReferenceImage(image)) {
      throw new Error(referenceType === "location" ? `This location has no image for ${gemmaRunnerLabel({ vision: true })} to describe.` : `This reference has no image for ${gemmaRunnerLabel({ vision: true })} to describe.`);
    }
    const modelFile = referenceDescriptionVisionModel();
    const mmprojFile = referenceDescriptionMmproj();
    if (!["lm_studio", "llm_api", "own_server"].includes(state.textGemmaRunner) && (!modelFile || !mmprojFile)) {
      throw new Error(`Choose a ${gemmaRunnerLabel({ vision: true })} model and Vision mmproj first.`);
    }
    const data = await postJson("/vrgdg/music_builder/describe_reference_image", {
      ...textGemmaRunnerPayload(),
      model_file: modelFile,
      mmproj_file: mmprojFile,
      reference_type: referenceType === "subject" ? (target?.reference_type || "character") : referenceType,
      name: referenceType === "extra" ? (target?.title || "") : (target?.name || ""),
      ...(referenceType === "extra" ? {
        style: String(target?.style || ""),
        count: Math.max(1, Math.min(100, Math.round(Number(target?.count) || 1))),
      } : {}),
      ...referenceImagePayload(image),
      unload_after: options.unloadAfter !== false,
      clear_before_load: Boolean(options.clearBeforeLoad),
    }, 4 * 60 * 1000);
    const description = String(data.description || "").trim();
    if (!description) throw new Error("Gemma returned an empty reference description.");
    if (referenceType === "extra" && !/\b(?:hair|bob(?:bed)?|pixie cut|buzz cut|braids?|dreadlocks?|locs?|afro|shaved head|bald)\b/i.test(description)) {
      throw new Error("Gemma omitted the extra's required hairstyle. Run Gemma Describe again so the locked identity includes hair color and style.");
    }
    const outputField = String(options.storeField || "").trim();
    if (outputField) target[outputField] = description;
    else target.description = description;
    return description;
  }

  async function createDetailedLocationDescriptionWithGemma(location, button = null, onUpdated = () => {}) {
    const name = String(location?.name || "").trim();
    const shortDescription = String(location?.description || "").trim();
    if (!name || !shortDescription) {
      toast("Run GPT Scout or Gemma Extract first so this location has a label and short description.", true);
      return;
    }
    const modelFile = String(t2iTextGemmaModelSelect.value || gemmaModelSelect.value || i2vTextGemmaModelSelect.value || i2vGemmaModelSelect.value || "").trim();
    if (!modelFile && state.textGemmaRunner === "builtin") {
      toast("Choose a Gemma4 model first in the LLM/Image model settings.", true);
      return;
    }
    const progress = createProgressWindow("Creating detailed location description", { zIndex: 100008 });
    const previousText = button?.textContent || "";
    if (button) {
      button.disabled = true;
      button.textContent = "Creating...";
    }
    try {
      progress.set(`Expanding ${name} from its short description...\n${gemmaRunnerLine()}`, 24);
      const data = await postJson("/vrgdg/gemma4/generate", {
        ...textGemmaRunnerPayload(),
        target: "location_description_detail",
        model_file: modelFile,
        location_name: name,
        location_description: shortDescription,
        notes: shortDescription,
        unload_after: true,
        n_ctx: normalizeGemmaContextLimit(state.gemmaContextLimit),
        gemma_output_token_limit: normalizeOutputTokenLimit(state.gemmaOutputTokenLimit),
        max_new_tokens: normalizeOutputTokenLimit(state.gemmaOutputTokenLimit),
      }, 10 * 60 * 1000);
      const detailed = String(data.text || "").trim();
      if (!detailed) throw new Error("Gemma4 returned an empty detailed location description.");
      location.description = detailed;
      onUpdated();
      await autoSaveSessionQuiet(`Detailed location description: ${name}`);
      progress.set("Detailed location description ready.", 100);
      progress.close(1000);
      toast(`Detailed description created for ${name}.`);
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = previousText || "Detailed Description";
      }
    }
  }

  return {
    createDetailedLocationDescriptionWithGemma, createProgressWindow, describeReferenceImageWithGemma,
    gemmaRunnerLabel, gemmaRunnerLine, llmApiVisionModelSelected, promptRunnerActionName,
    referenceDescriptionMmproj, referenceDescriptionVisionModel, textGemmaRunnerPayload,
    updatePromptRunnerButtonLabels,
  };
}
