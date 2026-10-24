// Frontend-only setup gate. No request or generation starts while configuration is open.
let activeSetup = null;

export function registerLlmSetup(setup) {
  activeSetup = setup;
  return () => {
    if (activeSetup === setup) activeSetup = null;
    setup.dispose();
  };
}

export function llmRequestUsesVision(url, payload = {}) {
  return Boolean(payload.use_vision || payload.use_vision_reference || payload.use_reference_images
    || payload.image_path || payload.image_data
    || payload.vision_images?.length || payload.reference_images?.length
    || /describe_reference_image|analyze_story_references/.test(url));
}

export function missingLlmSetup(payload = {}, vision = false) {
  const selected = value => {
    const text = String(value || "").trim();
    return text && !/^\[No |^No .*\b(found|available)\b/i.test(text);
  };
  const runner = payload.text_runner || payload.text_gemma_runner || "builtin";
  if (runner === "llm_api") {
    if (!String(payload.llm_api_key || "").trim()) return "Enter your LLM API key or choose another runner to continue.";
    if (!String(payload.llm_api_model || payload.llm_api_custom_model || "").trim()) return "Choose an LLM API model to continue.";
    if (vision && ["deepseek", "apifreellm"].includes(payload.llm_api_provider)) return "Choose a runner and model that support reference images.";
  } else if (runner === "lm_studio") {
    if (!String(payload.lmstudio_model || "").trim()) return "Choose a model already loaded in LM Studio to continue.";
  } else if (runner === "own_server") {
    if (!String(payload.own_server_url || "").trim()) return "Enter your Custom Server URL to continue.";
    if (!String(payload.own_server_model || "").trim()) return "Choose your Custom Server model to continue.";
  } else if (["builtin", "qwen_local"].includes(runner)) {
    const model = runner === "qwen_local" ? payload.qwen_model_file || payload.model_file
      : payload.gemma_model_file || payload.model_file;
    const projector = runner === "qwen_local" ? payload.qwen_mmproj_file || payload.mmproj_file : payload.mmproj_file;
    if (!selected(model)) return "Choose a local GGUF model or another runner to continue.";
    if (vision && !selected(projector)) return "Choose the vision mmproj for your local model or another vision runner to continue.";
  }
  return "";
}

export async function prepareLlmRequest(url, payload) {
  // Non-LLM calls, standalone tools and headless API clients keep their existing behavior.
  if (!activeSetup || !payload || !(payload.text_runner || payload.text_gemma_runner)) return payload;
  return activeSetup.prepare(payload, llmRequestUsesVision(url, payload));
}

export function createLlmSetup({ getRunnerPayload, openRunner }) {
  let pending = null;
  let pendingVision = false;
  let dialog = null;
  let disposed = false;
  const cancelled = () => Object.assign(new Error("LLM setup cancelled. Your current window and inputs are still available."), {
    code: "LLM_SETUP_CANCELLED",
  });
  const refresh = payload => ({ ...payload, ...getRunnerPayload() });
  async function prepare(payload, vision) {
    if (disposed) throw cancelled();
    if (!pending && !missingLlmSetup(payload, vision)) return payload;
    if (!pending && !missingLlmSetup(refresh(payload), vision)) return refresh(payload);
    pendingVision ||= vision;
    if (!pending) {
      pending = new Promise((resolve, reject) => {
        try {
          dialog = openRunner({
            setupMessage: missingLlmSetup(payload, vision),
            validateSetup: () => missingLlmSetup(getRunnerPayload(), pendingVision),
            onSaved: () => resolve(true),
            onCancel: () => resolve(false),
          });
        } catch (error) { reject(error); }
      }).finally(() => { pending = null; dialog = null; pendingVision = false; });
    }
    if (!await pending || disposed) throw cancelled();
    const updated = refresh(payload);
    const missing = missingLlmSetup(updated, vision);
    if (missing) throw new Error(missing);
    return updated;
  }
  return {
    prepare,
    dispose() { disposed = true; dialog?.cancelSetup?.(); },
  };
}
