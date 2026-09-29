import { getJson, normalizeOwnServerTimeoutMinutes, postJson } from "./comfy_api.mjs";
import { makeButton, makeField, makeInput, makeSelect, toast } from "./controls.mjs";
import { isLikelyEmbeddingModelId } from "./llm_runner.mjs";
import {
  normalizeGemmaContextLimit,
  normalizeGemmaGpuLayers,
  normalizeLmStudioContextLimit,
  normalizeOutputTokenLimit,
} from "./prompt_text.mjs";

export function createGemmaRunner({
  autoSaveSessionQuiet, createProgressWindow, saveSession, state, syncBuilderLlmModelSelectsFromRunner,
  t2iTextGemmaModelSelect, textGemmaRunnerPayload, updatePromptRunnerButtonLabels,
}) {
  function openGemmaRunnerModal() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(740px,calc(100vw - 40px));max-height:calc(100vh - 36px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">LLM Runner</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Choose the LLM runner for prompt writing. Image-reference video prompts can use Qwen Local with an mmproj, LM Studio vision, a vision-capable LLM API model, or your own OpenAI-compatible server.</div>`;
    const close = makeButton("Close");
    header.append(heading, close);
    const runner = makeSelect(["builtin", "qwen_local", "lm_studio", "llm_api", "own_server"], state.textGemmaRunner || "builtin");
    runner.options[0].textContent = "Gemma Local";
    runner.options[1].textContent = "Qwen Local";
    runner.options[2].textContent = "LM Studio";
    runner.options[3].textContent = "LLM API";
    runner.options[4].textContent = "Custom Server";
    const gemmaContextLimit = makeInput(String(normalizeGemmaContextLimit(state.gemmaContextLimit)), "number");
    gemmaContextLimit.min = "512";
    gemmaContextLimit.max = "262144";
    gemmaContextLimit.step = "256";
    const gemmaOutputTokenLimit = makeInput(String(normalizeOutputTokenLimit(state.gemmaOutputTokenLimit)), "number");
    gemmaOutputTokenLimit.min = "64";
    gemmaOutputTokenLimit.max = "262144";
    gemmaOutputTokenLimit.step = "256";
    const gemmaGpuLayers = makeInput(String(normalizeGemmaGpuLayers(state.gemmaGpuLayers)), "number");
    gemmaGpuLayers.min = "0";
    gemmaGpuLayers.max = "999";
    gemmaGpuLayers.step = "1";
    const qwenModelSelect = makeSelect([""], state.qwenModelFile || "");
    const qwenModelPath = makeInput(state.qwenModelFile || "");
    const qwenMmprojSelect = makeSelect([""], state.qwenMmprojFile || "");
    const chooseQwenModel = makeButton("Choose GGUF file");
    const qwenModelRow = document.createElement("div");
    qwenModelRow.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end;";
    const gemmaLocalSelect = makeSelect([""], state.gemmaModelFile || "");
    const gemmaLocalPath = makeInput(state.gemmaModelFile || "");
    const chooseGemmaModel = makeButton("Choose GGUF file");
    const gemmaLocalRow = document.createElement("div");
    gemmaLocalRow.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end;";
    const builtinPanel = document.createElement("div");
    builtinPanel.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:12px;";
    const builtinNote = document.createElement("div");
    builtinNote.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    builtinNote.textContent = "Advanced local GGUF settings. Context limit is the model's total input/output window; maximum output tokens is the most text one request may generate. Their combined use must fit the loaded model. Higher values can use much more RAM/VRAM and take longer.";
    const gemmaLocalPanel = document.createElement("div");
    gemmaLocalPanel.style.cssText = "display:flex;flex-direction:column;gap:10px;";
    gemmaLocalPanel.append(
      makeField("Gemma Local model in models/LLM", gemmaLocalSelect),
      gemmaLocalRow,
    );
    const qwenLocalPanel = document.createElement("div");
    qwenLocalPanel.style.cssText = "display:flex;flex-direction:column;gap:10px;";
    qwenLocalPanel.append(
      makeField("Qwen Local model in configured LLM folders", qwenModelSelect),
      makeField("Qwen vision mmproj (optional)", qwenMmprojSelect),
      qwenModelRow,
    );
    const gpuNote = document.createElement("div");
    gpuNote.style.cssText = "font-size:12px;color:#94a3b8;line-height:1.4;";
    builtinPanel.append(
      builtinNote,
      gemmaLocalPanel,
      qwenLocalPanel,
      makeField("Context limit / n_ctx", gemmaContextLimit),
      makeField("Maximum output tokens", gemmaOutputTokenLimit),
      gpuNote,
      makeField("GPU layers / n_gpu_layers", gemmaGpuLayers),
    );
    qwenModelRow.append(makeField("Qwen GGUF path (optional external file)", qwenModelPath), chooseQwenModel);
    gemmaLocalRow.append(makeField("Gemma GGUF path (optional external file)", gemmaLocalPath), chooseGemmaModel);
    gemmaLocalSelect.onchange = () => {
      gemmaLocalPath.value = gemmaLocalSelect.value || "";
      state.gemmaModelFile = gemmaLocalPath.value;
      syncBuilderLlmModelSelectsFromRunner();
    };
    gemmaLocalPath.oninput = () => {
      state.gemmaModelFile = gemmaLocalPath.value || "";
      syncBuilderLlmModelSelectsFromRunner();
    };
    chooseGemmaModel.onclick = async () => {
      try {
        const data = await postJson("/vrgdg/music_builder/pick_path", { kind: "gguf" });
        if (data.path) {
          gemmaLocalPath.value = data.path;
          state.gemmaModelFile = data.path;
          syncBuilderLlmModelSelectsFromRunner();
        }
      } catch (error) { toast(String(error?.message || error), true); }
    };
    qwenModelSelect.onchange = () => {
      qwenModelPath.value = qwenModelSelect.value || "";
      state.qwenModelFile = qwenModelPath.value;
      syncBuilderLlmModelSelectsFromRunner();
    };
    qwenModelPath.oninput = () => {
      state.qwenModelFile = qwenModelPath.value || "";
      syncBuilderLlmModelSelectsFromRunner();
    };
    qwenMmprojSelect.onchange = () => {
      state.qwenMmprojFile = qwenMmprojSelect.value || "";
      syncBuilderLlmModelSelectsFromRunner();
    };
    chooseQwenModel.onclick = async () => {
      try {
        const data = await postJson("/vrgdg/music_builder/pick_path", { kind: "gguf" });
        if (data.path) {
          qwenModelPath.value = data.path;
          state.qwenModelFile = data.path;
          syncBuilderLlmModelSelectsFromRunner();
        }
      } catch (error) { toast(String(error?.message || error), true); }
    };
    getJson("/vrgdg/music_builder/gemma_choices").then((data) => {
      const gemmaModels = Array.isArray(data.models) ? data.models.filter((item) => item && !/^\[No Gemma/i.test(item)) : [];
      gemmaLocalSelect.innerHTML = "";
      gemmaModels.forEach((item) => { const option = document.createElement("option"); option.value = item; option.textContent = item; gemmaLocalSelect.append(option); });
      if (state.gemmaModelFile && gemmaModels.includes(state.gemmaModelFile)) gemmaLocalSelect.value = state.gemmaModelFile;
      else if (gemmaModels.length && !state.gemmaModelFile) { gemmaLocalSelect.value = gemmaModels[0]; gemmaLocalPath.value = gemmaModels[0]; state.gemmaModelFile = gemmaModels[0]; }
      const models = Array.isArray(data.qwen_models) ? data.qwen_models.filter((item) => item && !/^\[No Qwen/i.test(item)) : [];
      const mmproj = Array.isArray(data.qwen_mmproj) ? data.qwen_mmproj.filter((item) => item && !/^\[No Qwen/i.test(item)) : [];
      qwenModelSelect.innerHTML = "";
      models.forEach((item) => { const option = document.createElement("option"); option.value = item; option.textContent = item; qwenModelSelect.append(option); });
      if (!models.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No Qwen GGUF found — choose a GGUF file below";
        qwenModelSelect.append(option);
      }
      if (state.qwenModelFile && models.includes(state.qwenModelFile)) qwenModelSelect.value = state.qwenModelFile;
      else if (models.length && !state.qwenModelFile) { qwenModelSelect.value = models[0]; qwenModelPath.value = models[0]; state.qwenModelFile = models[0]; }
      qwenMmprojSelect.innerHTML = "";
      mmproj.forEach((item) => { const option = document.createElement("option"); option.value = item; option.textContent = item; qwenMmprojSelect.append(option); });
      if (!mmproj.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No mmproj found (text-only Qwen is still supported)";
        qwenMmprojSelect.append(option);
      }
      if (state.qwenMmprojFile && mmproj.includes(state.qwenMmprojFile)) qwenMmprojSelect.value = state.qwenMmprojFile;
      else if (mmproj.length === 1) { qwenMmprojSelect.value = mmproj[0]; state.qwenMmprojFile = mmproj[0]; }
      syncBuilderLlmModelSelectsFromRunner();
    }).catch(() => null);
    const baseUrl = makeInput(state.lmStudioBaseUrl || "http://127.0.0.1:1234/v1");
    const model = makeInput(state.lmStudioModel || "");
    const modelSelect = makeSelect([""], "");
    const loadModels = makeButton("Load LM Studio Models");
    const modelPickerRow = document.createElement("div");
    modelPickerRow.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end;";
    const lmStudioApiKey = makeInput(state.lmStudioApiKey || "", "password");
    const lmStudioContextLimit = makeInput(String(normalizeLmStudioContextLimit(state.lmStudioContextLimit)), "number");
    lmStudioContextLimit.min = "512";
    lmStudioContextLimit.max = "262144";
    lmStudioContextLimit.step = "256";
    const lmStudioOutputTokenLimit = makeInput(String(normalizeOutputTokenLimit(state.lmStudioOutputTokenLimit)), "number");
    lmStudioOutputTokenLimit.min = "64";
    lmStudioOutputTokenLimit.max = "262144";
    lmStudioOutputTokenLimit.step = "256";
    const lmPanel = document.createElement("div");
    lmPanel.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:12px;";
    const note = document.createElement("div");
    note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    note.textContent = "In LM Studio, load your model and start the Local Server. Input context and maximum output tokens are sent with every text and vision request. Their combined use must fit the model; larger values need more memory and time.";
    const test = makeButton("Test LM Studio", "primary");
    const apiPanel = document.createElement("div");
    apiPanel.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:12px;";
    const apiNote = document.createElement("div");
    apiNote.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    apiNote.textContent = "API keys are session-only unless you explicitly save the current key to this project. A project-saved key will be included in the project session and may be included in shareable exports. For image-reference Video Prep, choose an API model that supports vision/images.";
    const apiProvider = makeSelect(["openai"], state.llmApiProvider || "openai");
    const apiModel = makeSelect([""], state.llmApiModel || "");
    const llmApiKey = makeInput(state.llmApiKey || "", "password");
    const testApi = makeButton("Test LLM API", "primary");
    const saveProjectApiKey = makeButton("Save API Key to Project", "primary");
    const apiStatus = document.createElement("div");
    apiStatus.style.cssText = "font-size:12px;color:#94a3b8;min-height:16px;";
    const providerLabel = (provider = {}) => String(provider.label || provider.id || "").trim();
    const llmProviders = () => Array.isArray(state.llmApiChoices?.providers) ? state.llmApiChoices.providers : [];
    const providerById = (id) => llmProviders().find((item) => String(item.id || "") === String(id || ""));
    const populateApiModels = () => {
      const provider = providerById(apiProvider.value) || llmProviders()[0] || { id: "openai", label: "OpenAI", models: ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-4o"], default_model: "gpt-6-luna" };
      const models = Array.isArray(provider.models) && provider.models.length ? provider.models : [provider.default_model || ""].filter(Boolean);
      apiModel.innerHTML = "";
      models.forEach((modelId) => {
        const option = document.createElement("option");
        option.value = modelId;
        option.textContent = modelId;
        apiModel.append(option);
      });
      const wanted = String(state.llmApiModel || "").trim();
      apiModel.value = wanted && models.includes(wanted) ? wanted : (provider.default_model || models[0] || "");
    };
    const populateApiProviders = () => {
      const providers = llmProviders().length ? llmProviders() : [{ id: "openai", label: "OpenAI", models: ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-4o"], default_model: "gpt-6-luna" }];
      apiProvider.innerHTML = "";
      providers.forEach((provider) => {
        const option = document.createElement("option");
        option.value = provider.id;
        option.textContent = providerLabel(provider);
        apiProvider.append(option);
      });
      const wantedRaw = String(state.llmApiProvider || "").trim();
      const wanted = wantedRaw === "xai" ? "grok" : wantedRaw;
      if (wantedRaw === "xai") state.llmApiProvider = "grok";
      apiProvider.value = providers.some((provider) => String(provider.id) === wanted) ? wanted : String(providers[0]?.id || "openai");
      populateApiModels();
    };
    const loadApiChoices = async () => {
      try {
        apiStatus.textContent = "Loading LLM API model list...";
        const data = await getJson("/vrgdg/music_builder/llm_api_choices");
        state.llmApiChoices = { providers: Array.isArray(data.providers) ? data.providers : [] };
        populateApiProviders();
        apiStatus.textContent = "";
      } catch (error) {
        apiStatus.textContent = `Could not load API model list: ${String(error?.message || error)}`;
        apiStatus.style.color = "#fca5a5";
        populateApiProviders();
      }
    };
    apiProvider.onchange = () => {
      state.llmApiProvider = apiProvider.value || "openai";
      state.llmApiModel = "";
      populateApiModels();
    };
    apiModel.onchange = () => {
      state.llmApiModel = apiModel.value || "";
    };
    llmApiKey.oninput = () => {
      state.llmApiKey = llmApiKey.value || "";
    };
    modelSelect.onchange = () => {
      if (modelSelect.value) model.value = modelSelect.value;
    };
    loadModels.onclick = async () => {
      loadModels.disabled = true;
      loadModels.textContent = "Loading...";
      try {
        const data = await postJson("/vrgdg/music_builder/lm_studio_models", {
          lmstudio_base_url: baseUrl.value || "http://127.0.0.1:1234/v1",
          lmstudio_api_key: lmStudioApiKey.value || "",
        }, 45000);
        const allIds = Array.isArray(data?.models) ? data.models.map((item) => String(item || "").trim()).filter(Boolean) : [];
        const ids = allIds.filter((id) => !isLikelyEmbeddingModelId(id));
        if (!allIds.length) throw new Error("LM Studio returned no models. Load a chat model in LM Studio and make sure the local server is running.");
        if (!ids.length) throw new Error("LM Studio only returned embedding models. Load a chat/text-generation model, then click Load LM Studio Models again.");
        modelSelect.innerHTML = "";
        ids.forEach((id) => {
          const option = document.createElement("option");
          option.value = id;
          option.textContent = id;
          modelSelect.append(option);
        });
        const current = String(model.value || "").trim();
        if (current && ids.includes(current)) modelSelect.value = current;
        else {
          modelSelect.value = ids[0];
          model.value = ids[0];
        }
        toast(`Loaded ${ids.length} LM Studio model${ids.length === 1 ? "" : "s"}.`);
      } catch (error) {
        toast(String(error?.message || error), true);
      } finally {
        loadModels.disabled = false;
        loadModels.textContent = "Load LM Studio Models";
      }
    };
    modelPickerRow.append(makeField("Available LM Studio models", modelSelect), loadModels);
    lmPanel.append(
      note,
      makeField("LM Studio base URL", baseUrl),
      modelPickerRow,
      makeField("LM Studio model name", model),
      makeField("API key (usually blank for local LM Studio)", lmStudioApiKey),
      makeField("Input context limit", lmStudioContextLimit),
      makeField("Maximum output tokens", lmStudioOutputTokenLimit),
      test,
    );
    apiPanel.append(
      apiNote,
      makeField("Provider", apiProvider),
      makeField("Model", apiModel),
      makeField("API key", llmApiKey),
      testApi,
      saveProjectApiKey,
      apiStatus,
    );
    const ownUrl = makeInput(state.ownServerUrl || "http://127.0.0.1:8000/v1");
    const ownModel = makeInput(state.ownServerModel || "");
    const ownModelSelect = makeSelect([""], "");
    const loadOwnModels = makeButton("Load Server Models");
    const ownModelPickerRow = document.createElement("div");
    ownModelPickerRow.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end;";
    const ownApiKey = makeInput(state.ownServerApiKey || "", "password");
    const ownOutputTokenLimit = makeInput(String(normalizeOutputTokenLimit(state.ownServerOutputTokenLimit)), "number");
    ownOutputTokenLimit.min = "64";
    ownOutputTokenLimit.max = "262144";
    ownOutputTokenLimit.step = "256";
    const ownTimeoutMinutes = makeInput(String(normalizeOwnServerTimeoutMinutes(state.ownServerTimeoutMinutes)), "number");
    ownTimeoutMinutes.min = "1";
    ownTimeoutMinutes.max = "6";
    ownTimeoutMinutes.step = "1";
    const ownPanel = document.createElement("div");
    ownPanel.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:12px;";
    const ownNote = document.createElement("div");
    ownNote.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
    ownNote.textContent = "Uses the OpenAI Chat Completions standard: POST {url}/v1/chat/completions. Text uses messages[].content as a string. Vision uses the official image_url content part with a JPEG data URL. Works with local addresses and HTTPS tunnels such as Cloudflare. An API key is optional and is only sent when filled. Enter the model id your server is serving, for example an Unsloth Gemma 4 12B IT QAT model.";
    const testOwn = makeButton("Test own server", "primary");
    const saveOwnProjectApiKey = makeButton("Save API Key to Project", "primary");
    const ownStatus = document.createElement("div");
    ownStatus.style.cssText = "font-size:12px;color:#94a3b8;min-height:16px;";
    const ownTestOutput = document.createElement("textarea");
    ownTestOutput.readOnly = true;
    ownTestOutput.spellcheck = false;
    ownTestOutput.placeholder = "Test response appears here and can be scrolled.";
    ownTestOutput.style.cssText = "width:100%;min-height:140px;max-height:260px;resize:vertical;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#020617;color:#e2e8f0;padding:8px;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow:auto;white-space:pre-wrap;";
    ownModelSelect.onchange = () => {
      if (ownModelSelect.value) ownModel.value = ownModelSelect.value;
    };
    loadOwnModels.onclick = async () => {
      loadOwnModels.disabled = true;
      loadOwnModels.textContent = "Loading...";
      try {
        const data = await postJson("/vrgdg/music_builder/own_server_models", {
          own_server_url: ownUrl.value || "http://127.0.0.1:8000/v1",
          own_server_api_key: ownApiKey.value || "",
        }, 45000);
        const ids = Array.isArray(data?.models) ? data.models.map((item) => String(item || "").trim()).filter(Boolean) : [];
        if (!ids.length) throw new Error("The server returned no models from GET /v1/models. Enter the model id manually.");
        ownModelSelect.innerHTML = "";
        ids.forEach((id) => {
          const option = document.createElement("option");
          option.value = id;
          option.textContent = id;
          ownModelSelect.append(option);
        });
        const current = String(ownModel.value || "").trim();
        if (current && ids.includes(current)) ownModelSelect.value = current;
        else {
          ownModelSelect.value = ids[0];
          ownModel.value = ids[0];
        }
        ownStatus.textContent = `Loaded ${ids.length} model${ids.length === 1 ? "" : "s"} from ${data.base_url || "the server"}.`;
        ownStatus.style.color = "#67e8f9";
        toast(`Loaded ${ids.length} own-server model${ids.length === 1 ? "" : "s"}.`);
      } catch (error) {
        ownStatus.textContent = `Could not load models: ${String(error?.message || error)}`;
        ownStatus.style.color = "#fca5a5";
        toast(String(error?.message || error), true);
      } finally {
        loadOwnModels.disabled = false;
        loadOwnModels.textContent = "Load Server Models";
      }
    };
    ownModelPickerRow.append(makeField("Available server models", ownModelSelect), loadOwnModels);
    ownPanel.append(
      ownNote,
      makeField("Server URL", ownUrl),
      ownModelPickerRow,
      makeField("Model name", ownModel),
      makeField("API key (optional)", ownApiKey),
      makeField("Maximum output tokens", ownOutputTokenLimit),
      makeField("Request timeout (minutes)", ownTimeoutMinutes),
      testOwn,
      saveOwnProjectApiKey,
      ownStatus,
      makeField("Test response", ownTestOutput),
    );
    const syncVisibility = () => {
      state.textGemmaRunner = runner.value || "builtin";
      state.gemmaContextLimit = normalizeGemmaContextLimit(gemmaContextLimit.value);
      state.gemmaOutputTokenLimit = normalizeOutputTokenLimit(gemmaOutputTokenLimit.value);
      state.gemmaGpuLayers = normalizeGemmaGpuLayers(gemmaGpuLayers.value);
      state.lmStudioContextLimit = normalizeLmStudioContextLimit(lmStudioContextLimit.value);
      state.lmStudioOutputTokenLimit = normalizeOutputTokenLimit(lmStudioOutputTokenLimit.value);
      syncBuilderLlmModelSelectsFromRunner();
      builtinPanel.style.display = ["builtin", "qwen_local"].includes(runner.value) ? "flex" : "none";
      gemmaLocalPanel.style.display = runner.value === "builtin" ? "flex" : "none";
      qwenLocalPanel.style.display = runner.value === "qwen_local" ? "flex" : "none";
      gpuNote.textContent = runner.value === "qwen_local"
        ? "Lower GPU layers if Qwen Local runs out of VRAM. Higher values use more VRAM and may run faster."
        : "Lower GPU layers if Gemma Local runs out of VRAM; try 12 for 10GB cards. Higher values use more VRAM and may run faster.";
      lmPanel.style.display = runner.value === "lm_studio" ? "flex" : "none";
      apiPanel.style.display = runner.value === "llm_api" ? "flex" : "none";
      ownPanel.style.display = runner.value === "own_server" ? "flex" : "none";
    };
    runner.onchange = syncVisibility;
    test.onclick = async () => {
      state.textGemmaRunner = "lm_studio";
      state.lmStudioBaseUrl = baseUrl.value || "http://127.0.0.1:1234/v1";
      state.lmStudioModel = model.value || "";
      state.lmStudioApiKey = lmStudioApiKey.value || "";
      state.lmStudioContextLimit = normalizeLmStudioContextLimit(lmStudioContextLimit.value);
      state.lmStudioOutputTokenLimit = normalizeOutputTokenLimit(lmStudioOutputTokenLimit.value);
      let progress = null;
      try {
        progress = createProgressWindow("Testing LM Studio");
        progress.set("Sending a tiny text prompt to LM Studio...", 35);
        const data = await postJson("/vrgdg/music_builder/flux_reference_zimage_prompt", {
          ...textGemmaRunnerPayload(),
          model_file: t2iTextGemmaModelSelect.value || "",
          reference_type: "location",
          source_text: "small empty test room",
          style_theme: "",
          max_new_tokens: Math.min(120, normalizeOutputTokenLimit(state.lmStudioOutputTokenLimit)),
        }, 60000);
        progress.set(`LM Studio responded successfully:\n${data.prompt}`, 100);
        progress.close(4000);
        toast("LM Studio text runner works.");
      } catch (error) {
        progress?.set(`LM Studio test failed:\n${String(error?.message || error)}`, 100);
        toast(String(error?.message || error), true);
      }
    };
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton("Cancel");
    const save = makeButton("Save Runner", "primary");
    actions.append(cancel, save);
    box.append(header, makeField("Text LLM runner", runner), builtinPanel, lmPanel, apiPanel, ownPanel, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    syncVisibility();
    loadApiChoices();
    close.onclick = cancel.onclick = () => backdrop.remove();
    save.onclick = async () => {
      state.textGemmaRunner = runner.value || "builtin";
      state.gemmaContextLimit = normalizeGemmaContextLimit(gemmaContextLimit.value);
      state.gemmaOutputTokenLimit = normalizeOutputTokenLimit(gemmaOutputTokenLimit.value);
      state.gemmaGpuLayers = normalizeGemmaGpuLayers(gemmaGpuLayers.value);
      state.lmStudioBaseUrl = baseUrl.value || "http://127.0.0.1:1234/v1";
      state.lmStudioModel = model.value || "";
      state.lmStudioApiKey = lmStudioApiKey.value || "";
      state.lmStudioContextLimit = normalizeLmStudioContextLimit(lmStudioContextLimit.value);
      state.lmStudioOutputTokenLimit = normalizeOutputTokenLimit(lmStudioOutputTokenLimit.value);
      state.llmApiProvider = apiProvider.value || "openai";
      state.llmApiModel = apiModel.value || "";
      state.llmApiKey = runner.value === "llm_api" ? llmApiKey.value || "" : state.llmApiKey || "";
      state.ownServerUrl = ownUrl.value || "http://127.0.0.1:8000/v1";
      state.ownServerModel = ownModel.value || "";
      state.ownServerApiKey = runner.value === "own_server" ? ownApiKey.value || "" : state.ownServerApiKey || "";
      state.ownServerOutputTokenLimit = normalizeOutputTokenLimit(ownOutputTokenLimit.value);
      state.ownServerTimeoutMinutes = normalizeOwnServerTimeoutMinutes(ownTimeoutMinutes.value);
      syncBuilderLlmModelSelectsFromRunner();
      await autoSaveSessionQuiet("LLM runner settings");
      toast(state.textGemmaRunner === "llm_api"
        ? "LLM API settings saved for this session. API key was not saved with the project."
        : state.textGemmaRunner === "own_server"
          ? "Custom Server settings saved for this session. API key was not saved with the project unless you use Save API Key to Project."
          : state.textGemmaRunner === "lm_studio"
          ? "Text LLM runner set to LM Studio."
          : state.textGemmaRunner === "qwen_local"
          ? "Text LLM runner set to Qwen Local."
          : "Text LLM runner set to Gemma Local.");
      updatePromptRunnerButtonLabels();
      backdrop.remove();
    };
    testApi.onclick = async () => {
      const provider = apiProvider.value || "openai";
      const modelId = apiModel.value || "";
      const key = llmApiKey.value || "";
      if (!key.trim()) {
        toast("Enter an API key before testing LLM API.", true);
        return;
      }
      state.llmApiProvider = provider;
      state.llmApiModel = modelId;
      state.llmApiKey = key;
      state.textGemmaRunner = "llm_api";
      updatePromptRunnerButtonLabels();
      let progress = null;
      try {
        testApi.disabled = true;
        testApi.textContent = "Testing...";
        progress = createProgressWindow("Testing LLM API");
        progress.set(`Sending a tiny test prompt...\nProvider: ${provider}\nModel: ${modelId || "(default)"}`, 30);
        const data = await postJson("/vrgdg/music_builder/test_llm_api", {
          provider,
          model: modelId,
          api_key: key,
          prompt: "Reply with OK only.",
        }, 120000);
        const responseText = String(data.text || "").trim();
        progress.set(`LLM API responded successfully.\nProvider: ${data.used_provider || provider}\nModel: ${data.used_model || modelId}\nResponse: ${responseText}`, 100);
        progress.close(4000);
        apiStatus.textContent = `Test passed: ${data.used_provider || provider} / ${data.used_model || modelId}`;
        apiStatus.style.color = "#67e8f9";
        toast("LLM API test passed.");
      } catch (error) {
        const message = String(error?.message || error);
        progress?.set(`LLM API test failed.\nProvider: ${provider}\nModel: ${modelId || "(default)"}\nReason: ${message}`, 100);
        apiStatus.textContent = `Test failed: ${message}`;
        apiStatus.style.color = "#fca5a5";
        toast(`LLM API test failed: ${message}`, true);
      } finally {
        testApi.disabled = false;
        testApi.textContent = "Test LLM API";
      }
    };
    testOwn.onclick = async () => {
      const url = ownUrl.value || "http://127.0.0.1:8000/v1";
      const modelId = ownModel.value || "";
      const key = ownApiKey.value || "";
      if (!String(url).trim()) {
        toast("Enter a server URL before testing.", true);
        return;
      }
      if (!String(modelId).trim()) {
        toast("Enter the model name your server is serving, or load models first.", true);
        return;
      }
      state.textGemmaRunner = "own_server";
      state.ownServerUrl = url;
      state.ownServerModel = modelId;
      state.ownServerApiKey = key;
      state.ownServerOutputTokenLimit = normalizeOutputTokenLimit(ownOutputTokenLimit.value);
      state.ownServerTimeoutMinutes = normalizeOwnServerTimeoutMinutes(ownTimeoutMinutes.value);
      updatePromptRunnerButtonLabels();
      ownTestOutput.value = "";
      try {
        testOwn.disabled = true;
        testOwn.textContent = "Testing...";
        ownStatus.textContent = `Sending a tiny test prompt to ${url}...`;
        ownStatus.style.color = "#94a3b8";
        const data = await postJson("/vrgdg/music_builder/test_own_server", {
          own_server_url: url,
          own_server_model: modelId,
          own_server_api_key: key,
          own_server_output_token_limit: normalizeOutputTokenLimit(ownOutputTokenLimit.value),
          own_server_timeout: normalizeOwnServerTimeoutMinutes(ownTimeoutMinutes.value) * 60,
          prompt: "Reply with OK only.",
        });
        const responseText = String(data.text || "").trim();
        ownTestOutput.value = [
          `Status: success`,
          `URL: ${data.base_url || url}`,
          `Model: ${data.used_model || modelId}`,
          "",
          responseText || "(empty response)",
        ].join("\n");
        ownStatus.textContent = `Test passed: ${data.used_model || modelId}`;
        ownStatus.style.color = "#67e8f9";
        toast("Custom Server test passed.");
      } catch (error) {
        const message = String(error?.message || error);
        ownTestOutput.value = `Status: failed\nURL: ${url}\nModel: ${modelId || "(missing)"}\n\n${message}`;
        ownStatus.textContent = `Test failed: ${message}`;
        ownStatus.style.color = "#fca5a5";
        toast(`Custom Server test failed: ${message}`, true);
      } finally {
        testOwn.disabled = false;
        testOwn.textContent = "Test own server";
      }
    };
    saveOwnProjectApiKey.onclick = async () => {
      const key = String(ownApiKey.value || "").trim();
      try {
        saveOwnProjectApiKey.disabled = true;
        saveOwnProjectApiKey.textContent = "Saving...";
        state.textGemmaRunner = "own_server";
        state.ownServerUrl = ownUrl.value || "http://127.0.0.1:8000/v1";
        state.ownServerModel = ownModel.value || "";
        state.ownServerApiKey = key;
        state.ownServerApiKeyProject = key;
        state.ownServerOutputTokenLimit = normalizeOutputTokenLimit(ownOutputTokenLimit.value);
        state.ownServerTimeoutMinutes = normalizeOwnServerTimeoutMinutes(ownTimeoutMinutes.value);
        await saveSession({ quiet: true, throwOnError: true });
        toast(key
          ? "Custom Server API key saved to this project. Shareable exports will warn you before including it."
          : "Cleared the project-saved own server API key. Requests will continue without a key.");
      } catch (error) {
        toast(`Could not save the project API key: ${String(error?.message || error)}`, true);
      } finally {
        saveOwnProjectApiKey.disabled = false;
        saveOwnProjectApiKey.textContent = "Save API Key to Project";
      }
    };
    saveProjectApiKey.onclick = async () => {
      const key = String(llmApiKey.value || "").trim();
      if (!key) {
        toast("Enter an API key before saving it to this project.", true);
        return;
      }
      try {
        saveProjectApiKey.disabled = true;
        saveProjectApiKey.textContent = "Saving...";
        state.textGemmaRunner = "llm_api";
        state.llmApiProvider = apiProvider.value || "openai";
        state.llmApiModel = apiModel.value || "";
        state.llmApiKey = key;
        state.llmApiKeyProject = key;
        await saveSession({ quiet: true, throwOnError: true });
        toast("LLM API key saved to this project. Shareable exports will warn you before including it.");
      } catch (error) {
        toast(`Could not save the project API key: ${String(error?.message || error)}`, true);
      } finally {
        saveProjectApiKey.disabled = false;
        saveProjectApiKey.textContent = "Save API Key to Project";
      }
    };
    return backdrop;
  }

  return { openGemmaRunnerModal };
}
