import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "VRGDG_LoraDatasetCreatorUI";
const $ = (tag, css = "", text = "") => { const el = document.createElement(tag); if (css) el.style.cssText = css; if (text) el.textContent = text; return el; };

async function post(url, payload = {}) {
  const response = await api.fetchApi(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const data = await response.json();
  if (!response.ok || data?.ok === false) throw new Error(data?.error || `Request failed (${response.status})`);
  return data;
}

function button(text, primary = false) {
  const el = $("button", `border:1px solid ${primary ? "#8b5cf6" : "#494651"};border-radius:8px;background:${primary ? "linear-gradient(135deg,#7c3aed,#9333ea)" : "#29272f"};color:#f5f3ff;padding:9px 14px;font:700 12px sans-serif;cursor:pointer;`, text);
  return el;
}
function input(value = "", type = "text") { const el = $("input", "width:100%;box-sizing:border-box;border:1px solid #494651;border-radius:8px;background:#201e26;color:#f5f3ff;padding:10px 11px;font:12px sans-serif;outline:none;"); el.type = type; el.value = value; return el; }
function select(values, value) { const el = $("select", "width:100%;box-sizing:border-box;border:1px solid #494651;border-radius:8px;background:#201e26;color:#f5f3ff;padding:10px 11px;font:12px sans-serif;"); for (const [v, label] of values) { const o = $("option", "", label); o.value = v; el.append(o); } el.value = value; return el; }
function setSelectOptions(el, values, preferred = "") { const current = preferred || el.value; el.replaceChildren(); for (const value of values || []) { const pair = Array.isArray(value) ? value : [value, value]; const option = $("option", "", pair[1]); option.value = pair[0]; el.append(option); } if ([...el.options].some((o) => o.value === current)) el.value = current; else if (el.options.length) el.selectedIndex = 0; }
function field(label, control) { const wrap = $("label", "display:flex;flex-direction:column;gap:6px;color:#d6d3df;font:700 12px sans-serif;"); wrap.append($("span", "", label), control); return wrap; }
function imageUrl(image) { const p = new URLSearchParams({ filename: image.filename || "", type: image.type || "output", rand: String(Date.now()) }); if (image.subfolder) p.set("subfolder", image.subfolder); return `/view?${p}`; }
function historyImages(data, id) { const root = data?.[id] || data || {}; return Object.values(root.outputs || {}).flatMap((o) => Array.isArray(o?.images) ? o.images : []); }
async function queue(prompt) { const r = await api.fetchApi("/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt, client_id: api.clientId || crypto.randomUUID() }) }); const d = await r.json(); if (!r.ok || d.error) throw new Error(d?.error?.message || d?.error || "Could not queue workflow."); return d.prompt_id; }
async function waitImage(id, status) {
  const start = Date.now();
  while (Date.now() - start < 20 * 60 * 1000) {
    const r = await api.fetchApi(`/history/${encodeURIComponent(id)}`); const d = await r.json();
    const root = d?.[id];
    if (root?.status?.status_str === "error" || root?.status?.completed === false && root?.status?.messages?.some?.((m) => m?.[0] === "execution_error")) throw new Error("The image workflow failed. Check the ComfyUI console for details.");
    const images = historyImages(d, id); if (images.length) return images.at(-1);
    status("Rendering image…"); await new Promise((resolve) => setTimeout(resolve, 1300));
  }
  throw new Error("Timed out waiting for the image workflow.");
}

function openCreator() {
  const state = { running: false, paused: false, stopped: false, results: [], characterAnchor: null };
  const overlay = $("div", "position:fixed;inset:0;z-index:100005;background:#0d0b11;display:flex;align-items:stretch;justify-content:stretch;padding:0;");
  const modal = $("div", "width:100vw;height:100vh;display:flex;flex-direction:column;overflow:hidden;border:0;border-radius:0;background:#17151c;color:#f7f5fb;box-shadow:none;font-family:Inter,Segoe UI,sans-serif;");
  const header = $("div", "display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid #35323c;background:linear-gradient(180deg,#211e29,#19171e);");
  const title = $("div", "display:flex;gap:11px;align-items:center;font-size:20px;font-weight:900;"); title.append($("span", "color:#a78bfa;font-size:25px;", "✦"), document.createTextNode("LoRA Dataset Creator"));
  const close = button("Close · Back to Canvas"); close.onclick = () => { if (!state.running || confirm("A dataset is still running. Close the window?")) overlay.remove(); }; header.append(title, close);
  const onEscape = (event) => { if (event.key === "Escape") close.click(); };
  window.addEventListener("keydown", onEscape);
  const removeOverlay = overlay.remove.bind(overlay);
  overlay.remove = () => { window.removeEventListener("keydown", onEscape); removeOverlay(); };
  const body = $("div", "display:grid;grid-template-columns:minmax(620px,1.45fr) minmax(380px,.85fr);gap:16px;padding:16px;min-height:0;flex:1;overflow:hidden;");
  const left = $("div", "display:flex;flex-direction:column;gap:13px;overflow:auto;padding-right:4px;");
  const right = $("div", "display:flex;flex-direction:column;min-height:0;border:1px solid #3a3741;border-radius:10px;background:#1d1b22;padding:16px;overflow:hidden;");

  const datasetType = select([["style", "Style LoRA"], ["character", "Character LoRA"], ["ic_pair", "Experimental LTX IC-LoRA · 1-frame pairs"]], "style");
  const modeNote = $("div", "border:1px solid #4c3b68;border-radius:8px;background:#211a2c;color:#cfc3e8;padding:9px 11px;font-size:11px;line-height:1.4;", "Create varied images that teach one reusable visual style.");
  const style = $("textarea", "width:100%;height:94px;resize:vertical;box-sizing:border-box;border:1px solid #494651;border-radius:8px;background:#201e26;color:#f5f3ff;padding:11px;font:13px/1.45 sans-serif;outline:none;"); style.placeholder = "Describe the art style you want the LoRA to learn…";
  const identityGrid = $("div", "display:grid;grid-template-columns:.75fr 1.35fr;gap:10px;");
  const trigger = input(); trigger.placeholder = "Created automatically";
  const phrase = input(); phrase.placeholder = "Created automatically";
  const triggerWrap = $("div", "display:grid;grid-template-columns:1fr auto;gap:5px;"); const triggerMagic = button("✦"); triggerWrap.append(trigger, triggerMagic);
  const phraseWrap = $("div", "display:grid;grid-template-columns:1fr auto;gap:5px;"); const phraseMagic = button("✦"); phraseWrap.append(phrase, phraseMagic);
  identityGrid.append(field("Trigger Word", triggerWrap), field("Style Phrase", phraseWrap));

  const conceptHead = $("div", "display:flex;align-items:center;justify-content:space-between;gap:10px;");
  const conceptTitle = $("div", "font-size:14px;font-weight:900;", "2. Concepts"); const countText = $("span", "color:#aaa5b3;font-size:11px;font-weight:500;margin-left:6px;", "one line = one image"); conceptTitle.append(countText);
  const ideas = button("✦ Generate Ideas"); conceptHead.append(conceptTitle, ideas);
  const concepts = $("textarea", "width:100%;height:clamp(230px,34vh,460px);min-height:230px;resize:none;box-sizing:border-box;border:1px solid #494651;border-radius:8px;background:#201e26;color:#f5f3ff;padding:11px;font:13px/1.65 sans-serif;outline:none;overflow:auto;"); concepts.placeholder = "a pirate captain searching for buried treasure\na knight battling a fire-breathing dragon\na wizard casting a glowing spell";
  const conceptCount = $("div", "text-align:right;color:#aaa5b3;font-size:11px;margin-top:-8px;", "0 concepts · 0 images");
  const destGrid = $("div", "display:grid;grid-template-columns:150px 1fr;gap:10px;");
  const generator = select([["zimage", "ZImage"], ["krea", "Krea 2"], ["flow", "Flow / Nano Banana"], ["gpt_image", "GPT Image"]], "zimage");
  const folder = input(); folder.placeholder = "Choose where the dataset will be saved"; const browse = button("Browse…"); const folderWrap = $("div", "display:grid;grid-template-columns:1fr auto;gap:7px;"); folderWrap.append(folder, browse);
  destGrid.append(field("Generator", generator), field("Project Root Folder", folderWrap));
  const folderStructureNote = $("div", "border:1px solid #3a3741;border-radius:8px;background:#1d1b22;padding:9px 11px;color:#aaa5b3;font-size:11px;line-height:1.45;", "The project root is created automatically. Training pairs go into dataset/. JSON and creator metadata go into project_files/.");
  const browserSetupPanel = $("div", "display:none;flex-direction:column;gap:9px;border:1px solid #624b83;border-radius:8px;background:#211a2c;padding:11px;");
  const browserSetupTitle = $("div", "font-size:12px;font-weight:900;color:#ddd6fe;", "Flow / GPT Browser Automation");
  const browserSetupNote = $("div", "font-size:11px;color:#c4b5d4;line-height:1.45;", "Flow / Nano Banana uses Google Flow in a real Chrome profile. GPT Image uses the ChatGPT image interface. Install automation first, then open and sign in to every provider you plan to use. Windows can install portable Node automatically; Playwright is installed into this extension's flow_automation folder. For Flow, set the browser UI to 1 image and the desired aspect ratio before a long run.");
  const browserSetupActions = $("div", "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;");
  const installBrowser = button("Install Browser Automation", true), checkBrowser = button("Check Setup"), loginFlow = button("Open Flow Login"), loginGpt = button("Open GPT Image Login");
  browserSetupActions.append(installBrowser, checkBrowser, loginFlow, loginGpt);
  const browserSetupStatus = $("div", "font-size:11px;color:#aaa5b3;white-space:pre-wrap;line-height:1.4;", "Setup status has not been checked yet.");
  browserSetupPanel.append(browserSetupTitle, browserSetupNote, browserSetupActions, browserSetupStatus);

  const referenceBlock = $("div", "display:none;border:1px solid #3a3741;border-radius:8px;background:#1d1b22;padding:11px;gap:9px;flex-direction:column;");
  const referenceNote = $("div", "font-size:11px;color:#aaa5b3;line-height:1.4;", "Optional: add a face image, full character reference, or turnaround. Flow/GPT will receive it with every scene. If omitted, the first generated character becomes the reference for the remaining images.");
  const referencePick = button("Choose Character Reference…"); const referenceName = $("span", "font-size:11px;color:#c4b5fd;", "No reference selected"); const referenceFile = $("input"); referenceFile.type = "file"; referenceFile.accept = "image/*"; referenceFile.style.display = "none"; referenceBlock.append($("div", "font-size:12px;font-weight:800;", "Character Reference"), referenceNote, referencePick, referenceName, referenceFile);
  let referenceIngredient = null;
  referencePick.onclick = () => referenceFile.click();
  referenceFile.onchange = () => { const file = referenceFile.files?.[0]; if (!file) return; const reader = new FileReader(); reader.onload = () => { referenceIngredient = { data: String(reader.result || ""), name: file.name }; referenceName.textContent = file.name; }; reader.readAsDataURL(file); };

  const editBlock = $("div", "display:none;flex-direction:column;gap:7px;");
  const editInstruction = $("textarea", "width:100%;height:86px;resize:vertical;box-sizing:border-box;border:1px solid #624b83;border-radius:8px;background:#201e26;color:#f5f3ff;padding:10px;font:12px/1.45 sans-serif;"); editInstruction.placeholder = "Describe the edit, for example: Make the same person visibly older while preserving identity, pose, clothing, framing, lighting, and background.";
  editBlock.append(field("Edit Transformation", editInstruction), $("div", "font-size:11px;color:#f0abfc;line-height:1.4;", "Experimental LTX export: each PNG pair is treated as a one-frame reference → target sample. This does not teach temporal consistency."));

  const advanced = $("details", "border:1px solid #3a3741;border-radius:8px;background:#1d1b22;padding:10px 12px;"); const summary = $("summary", "cursor:pointer;font-size:12px;font-weight:800;color:#ddd9e5;", "Advanced Settings");
  const advancedGrid = $("div", "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin-top:12px;");
  const width = input("1024", "number"), height = input("1024", "number"), seedMode = select([["random", "Random each image"], ["fixed", "Fixed/incrementing"]], "random"), seed = input("1", "number");
  const zEnhance = input("", "checkbox"); zEnhance.checked = true; zEnhance.style.cssText = "width:18px;height:18px;accent-color:#8b5cf6;";
  const zEnhanceStrength = input("0.5", "number"); zEnhanceStrength.min = "0.1"; zEnhanceStrength.max = "1"; zEnhanceStrength.step = "0.05";
  const zEnhanceField = field("ZImage detail enhance", zEnhance), zEnhanceStrengthField = field("Enhance strength (0.1–1.0)", zEnhanceStrength);
  const llmRunner = select([["gemma_local", "Gemma Local / Gemma 4"], ["lm_studio", "LM Studio"], ["llm_api", "LLM API"]], "gemma_local"), gemmaModel = select([], ""), gemmaMmproj = select([], ""), lmUrl = input("http://127.0.0.1:1234/v1"), lmModel = select([], ""), refreshLmModels = button("Refresh LM Studio Models"), provider = select([["openai", "OpenAI"], ["anthropic", "Anthropic"], ["google", "Google Gemini"], ["xai", "xAI / Grok"], ["openrouter", "OpenRouter"]], "openai"), apiModel = select([], ""), apiKey = input("", "password");
  const llmStatus = $("div", "grid-column:1/-1;color:#aaa5b3;font-size:11px;", "LLM model choices load dynamically for the selected runner.");
  const runnerField = field("LLM runner", llmRunner), gemmaModelField = field("Gemma / Gemma 4 model", gemmaModel), gemmaMmprojField = field("Gemma vision mmproj", gemmaMmproj), lmUrlField = field("LM Studio URL", lmUrl), lmModelField = field("LM Studio model", lmModel), providerField = field("API provider", provider), apiModelField = field("API model", apiModel), apiKeyField = field("API key (session only)", apiKey);
  advancedGrid.append(field("Width", width), field("Height", height), field("Seed mode", seedMode), field("Starting seed", seed), zEnhanceField, zEnhanceStrengthField, runnerField, gemmaModelField, gemmaMmprojField, lmUrlField, lmModelField, refreshLmModels, providerField, apiModelField, apiKeyField, llmStatus); advanced.append(summary, advancedGrid);
  const create = button("✦  Create Dataset", true); create.style.cssText += "font-size:15px;padding:13px;width:100%;";
  const descriptionField = field("1. Art Style", style);
  left.append(field("Dataset Type", datasetType), modeNote, descriptionField, identityGrid, referenceBlock, editBlock, conceptHead, concepts, conceptCount, destGrid, folderStructureNote, browserSetupPanel, advanced, create);

  const stateTitle = $("div", "color:#a78bfa;font-size:16px;font-weight:900;", "Ready to create");
  const progressLabel = $("div", "font-size:13px;font-weight:800;", "0 / 0"); const topLine = $("div", "display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;"); topLine.append(stateTitle, progressLabel);
  const track = $("div", "height:10px;border-radius:99px;background:#302d36;overflow:hidden;"); const bar = $("div", "height:100%;width:0;background:linear-gradient(90deg,#7c3aed,#a855f7);transition:width .25s;"); track.append(bar);
  const preview = $("div", "height:min(42vh,390px);min-height:220px;flex:0 0 auto;margin-top:14px;border:1px dashed #494651;border-radius:9px;background:#141219;display:flex;align-items:center;justify-content:center;overflow:hidden;color:#77717f;text-align:center;padding:12px;box-sizing:border-box;", "Your latest generated image will appear here.");
  const current = $("div", "min-height:34px;padding-top:11px;font-size:12px;color:#cec9d5;", "Describe a style, add concepts, choose a folder, and create.");
  const controls = $("div", "display:none;gap:8px;"); const pause = button("Pause"); const stop = button("Stop"); controls.append(pause, stop);
  const thumbs = $("div", "display:none;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;max-height:260px;overflow:auto;margin-top:13px;");
  const completeButtons = $("div", "display:none;grid-template-columns:1fr 1fr;gap:9px;margin-top:13px;"); const openFolder = button("Open Dataset Folder"); const more = button("Create More"); completeButtons.append(openFolder, more);
  right.append(topLine, track, preview, current, controls, thumbs, completeButtons); body.append(left, right); modal.append(header, body); overlay.append(modal); document.body.append(overlay);

  const lines = () => concepts.value.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const updateCount = () => { const n = lines().length; conceptCount.textContent = `${n} concept${n === 1 ? "" : "s"} · ${n} image${n === 1 ? "" : "s"}`; };
  concepts.addEventListener("input", updateCount);
  function syncDatasetType() {
    const mode = datasetType.value;
    trigger.value = ""; phrase.value = ""; state.characterAnchor = null;
    referenceBlock.style.display = mode === "character" ? "flex" : "none";
    editBlock.style.display = mode === "ic_pair" ? "flex" : "none";
    identityGrid.style.display = mode === "ic_pair" ? "none" : "grid";
    if (mode === "character") {
      descriptionField.firstChild.textContent = "1. Character Description";
      style.placeholder = "Describe the character's stable identity, face, hair, proportions, clothing or recurring design…";
      modeNote.textContent = "Create many varied scenes of one consistent character. Flow/GPT can use a face, character reference, or turnaround with every image.";
      if (!["flow", "gpt_image"].includes(generator.value) && referenceIngredient) generator.value = "flow";
    } else if (mode === "ic_pair") {
      descriptionField.firstChild.textContent = "1. Source Dataset Description";
      style.placeholder = "Describe the kinds of source images needed to teach this edit…";
      modeNote.textContent = "Experimental LTX-2.3 IC-LoRA dataset made from one-frame reference and edited target pairs.";
      generator.value = ["flow", "gpt_image"].includes(generator.value) ? generator.value : "flow";
    } else {
      descriptionField.firstChild.textContent = "1. Art Style";
      style.placeholder = "Describe the art style you want the LoRA to learn…";
      modeNote.textContent = "Create varied images that teach one reusable visual style.";
    }
    syncGeneratorOptions();
  }
  datasetType.onchange = syncDatasetType; syncDatasetType();
  function formatBrowserStatus(data) { return [`Chrome: ${data.chrome_ready ? "ready" : "missing"}`, `Node: ${data.node_ready ? "ready" : "missing"}`, `Playwright: ${data.playwright_ready ? "installed" : "missing"}`, data.chrome_error ? `Chrome error: ${data.chrome_error}` : ""].filter(Boolean).join("\n"); }
  function syncGeneratorOptions() {
    syncBrowserSetupPanel();
    const isKrea = generator.value === "krea";
    zEnhanceField.style.display = isKrea ? "flex" : "none";
    zEnhanceStrengthField.style.display = isKrea && zEnhance.checked ? "flex" : "none";
  }
  function syncBrowserSetupPanel() { browserSetupPanel.style.display = ["flow", "gpt_image"].includes(generator.value) ? "flex" : "none"; }
  async function checkBrowserSetup() { browserSetupStatus.textContent = "Checking browser automation setup…"; try { const response = await api.fetchApi("/vrgdg/browser_image/status"); const data = await response.json(); if (!response.ok || data.ok === false) throw new Error(data.error || "Setup check failed."); browserSetupStatus.textContent = formatBrowserStatus(data); return data; } catch (error) { browserSetupStatus.textContent = error.message; throw error; } }
  async function openProviderLogin(browserProvider) { try { const data = await post("/vrgdg/browser_image/open_login", { provider: browserProvider }); browserSetupStatus.textContent = `${data.provider_label || browserProvider} login opened. Sign in and leave the browser available for dataset generation.`; } catch (error) { browserSetupStatus.textContent = error.message; } }
  installBrowser.onclick = async () => { try { installBrowser.disabled = true; installBrowser.textContent = "Installing Node / Playwright…"; browserSetupStatus.textContent = "Installing browser automation. This can take several minutes…"; const data = await post("/vrgdg/browser_image/setup", { install_portable_node: true, install_if_missing: true, strict_ssl: false, timeout_seconds: 900 }); browserSetupStatus.textContent = data.status || formatBrowserStatus(data); } catch (error) { browserSetupStatus.textContent = error.message; } finally { installBrowser.disabled = false; installBrowser.textContent = "Install Browser Automation"; } };
  checkBrowser.onclick = () => checkBrowserSetup().catch(() => null);
  loginFlow.onclick = () => openProviderLogin("flow_nano_banana");
  loginGpt.onclick = () => openProviderLogin("gpt_image");
  generator.onchange = () => {
    if (datasetType.value === "ic_pair" && !["flow", "gpt_image"].includes(generator.value)) {
      generator.value = "flow";
      current.textContent = "Experimental IC image pairs require Flow or GPT Image because the target must edit the exact source image.";
    }
    syncGeneratorOptions();
  };
  zEnhance.onchange = syncGeneratorOptions;
  syncGeneratorOptions();
  let providerModels = {};
  let defaultModels = {};
  function syncLlmFields() {
    const runner = llmRunner.value;
    for (const el of [gemmaModelField, gemmaMmprojField]) el.style.display = runner === "gemma_local" ? "flex" : "none";
    for (const el of [lmUrlField, lmModelField, refreshLmModels]) el.style.display = runner === "lm_studio" ? "flex" : "none";
    for (const el of [providerField, apiModelField, apiKeyField]) el.style.display = runner === "llm_api" ? "flex" : "none";
  }
  function syncApiModels() { setSelectOptions(apiModel, providerModels[provider.value] || [], defaultModels[provider.value] || ""); }
  async function loadLlmChoices() {
    try {
      const data = await post("/vrgdg/lora_dataset/llm_choices");
      providerModels = data.provider_models || {}; defaultModels = data.default_models || {};
      setSelectOptions(gemmaModel, data.gemma_models || []);
      setSelectOptions(gemmaMmproj, data.gemma_mmproj || []);
      syncApiModels(); syncLlmFields();
    } catch (error) { llmStatus.textContent = `Could not load LLM choices: ${error.message}`; }
  }
  async function loadLmStudioModels() {
    try { refreshLmModels.disabled = true; const data = await post("/vrgdg/lora_dataset/lm_studio_models", { lmstudio_base_url: lmUrl.value.trim() }); setSelectOptions(lmModel, data.models || []); llmStatus.textContent = `Found ${(data.models || []).length} LM Studio model(s).`; }
    catch (error) { llmStatus.textContent = error.message; }
    finally { refreshLmModels.disabled = false; }
  }
  llmRunner.onchange = syncLlmFields; provider.onchange = syncApiModels; refreshLmModels.onclick = loadLmStudioModels; loadLlmChoices();
  const llm = () => ({ llm_runner: llmRunner.value, gemma_model: gemmaModel.value, gemma_mmproj: gemmaMmproj.value, lmstudio_base_url: lmUrl.value.trim(), lmstudio_model: lmModel.value, provider: provider.value, model: apiModel.value, api_key: apiKey.value, dataset_type: datasetType.value });
  const busy = (yes, label) => { [ideas, triggerMagic, phraseMagic, create].forEach((b) => b.disabled = yes); if (label) current.textContent = label; };
  async function ensureIdentity() { if (datasetType.value === "ic_pair" || (trigger.value.trim() && phrase.value.trim())) return; const data = await post("/vrgdg/lora_dataset/identity", { ...llm(), art_style: style.value.trim() }); trigger.value = data.trigger_word; phrase.value = data.trigger_phrase; }
  async function generateIdentity() { try { busy(true, "Creating the style identity…"); trigger.value = ""; phrase.value = ""; await ensureIdentity(); current.textContent = "Style identity ready."; } catch (e) { current.textContent = e.message; } finally { busy(false); } }
  triggerMagic.onclick = phraseMagic.onclick = generateIdentity;
  ideas.onclick = async () => { const requested = Math.max(1, Number(prompt("How many dataset images/concepts?", String(Math.max(lines().length, 20))) || 0)); if (!requested) return; try { busy(true, `Creating ${requested} varied concepts…`); await ensureIdentity(); const data = await post("/vrgdg/lora_dataset/concepts", { ...llm(), art_style: style.value.trim(), count: requested }); concepts.value = data.concepts.join("\n"); updateCount(); current.textContent = `${data.concepts.length} concepts ready.`; } catch (e) { current.textContent = e.message; } finally { busy(false); } };
  browse.onclick = async () => { try { const d = await post("/vrgdg/lora_dataset/pick_folder"); if (d.path) folder.value = d.path; } catch (e) { current.textContent = e.message; } };
  openFolder.onclick = () => post("/vrgdg/lora_dataset/open_folder", { path: folder.value.trim() }).catch((e) => current.textContent = e.message);
  more.onclick = () => { state.results = []; thumbs.innerHTML = ""; thumbs.style.display = "none"; completeButtons.style.display = "none"; preview.innerHTML = ""; preview.textContent = "Ready for another dataset run."; stateTitle.textContent = "Ready to create"; progressLabel.textContent = "0 / 0"; bar.style.width = "0"; };
  pause.onclick = () => { state.paused = !state.paused; pause.textContent = state.paused ? "Resume" : "Pause"; current.textContent = state.paused ? "Paused after the current image." : "Resuming…"; };
  stop.onclick = () => { state.stopped = true; state.paused = false; current.textContent = "Stopping after the current image…"; };
  async function waitPaused() { while (state.paused && !state.stopped) await new Promise((r) => setTimeout(r, 250)); }
  function workflowPayload(promptText, index) { const baseSeed = Number(seed.value || 1); const chosenSeed = seedMode.value === "random" ? Math.floor(Math.random() * 1125899906842624) : baseSeed + index - 1; return { prompt: promptText, unet_name: "z_image_turbo_bf16.safetensors", clip_name: "qwen_3_4b.safetensors", vae_name: "ae.safetensors", krea_unet_name: "krea2_turbo_fp8_scaled.safetensors", krea_clip_name: "qwen3vl_4b_fp8_scaled.safetensors", krea_vae_name: "qwen_image_vae.safetensors", z_unet_name: "z_image_turbo_bf16.safetensors", z_clip_name: "qwen_3_4b.safetensors", z_vae_name: "ae.safetensors", use_zimage_enhance: zEnhance.checked, zimage_enhance_strength: Math.max(0.1, Math.min(1, Number(zEnhanceStrength.value || 0.5))), first_pass_width: Number(width.value || 1024), first_pass_height: Number(height.value || 1024), second_pass_width: Number(width.value || 1024), second_pass_height: Number(height.value || 1024), width: Number(width.value || 1024), height: Number(height.value || 1024), seed: chosenSeed, seed_mode: "fixed", batch_size: 1, use_custom_loras: false, lora_count: 0 }; }
  async function generateImage(promptText, index, ingredients = []) {
    const mode = generator.value;
    if (mode === "flow" || mode === "gpt_image") {
      const built = await post("/vrgdg/workflow_runner/build_flow_gpt_image_prompt", {
        provider: mode === "gpt_image" ? "gpt_image" : "flow_nano_banana",
        prompt: promptText,
        aspect_ratio: Number(width.value || 1024) === Number(height.value || 1024) ? "1:1" : Number(width.value || 1024) > Number(height.value || 1024) ? "16:9" : "9:16",
        image_ingredients: ingredients,
        timeout_seconds: 600,
        reuse_open_project: true,
      });
      const promptId = await queue(built.prompt);
      return await waitImage(promptId, (message) => current.textContent = `${built.provider_label || "Browser image"}: ${message}`);
    }
    const wfPayload = workflowPayload(promptText, index);
    const endpoint = mode === "krea" ? "/vrgdg/workflow_runner/build_krea2_prompt" : "/vrgdg/workflow_runner/build_zimage_prompt";
    const built = await post(endpoint, wfPayload); const promptId = await queue(built.prompt);
    return await waitImage(promptId, (message) => current.textContent = message);
  }
  async function imageIngredient(image) { const data = await post("/vrgdg/lora_dataset/image_source", { image }); return { path: data.path, name: image.filename || "reference.png" }; }
  function showImage(image) { preview.innerHTML = ""; const img = $("img", "width:100%;height:100%;object-fit:contain;"); img.src = imageUrl(image); preview.append(img); }
  function addThumb(image, index) { const img = $("img", "width:100%;aspect-ratio:1;object-fit:cover;border:1px solid #6d4bad;border-radius:7px;background:#111;"); img.title = `image_${String(index).padStart(3, "0")}.png`; img.src = imageUrl(image); img.onclick = () => showImage(image); thumbs.append(img); }
  create.onclick = async () => {
    if (state.running) return; const jobs = lines();
    if (!style.value.trim() || !jobs.length || !folder.value.trim()) { current.textContent = "Add an art style, at least one concept, and a save folder."; return; }
    if (datasetType.value === "character" && referenceIngredient && !["flow", "gpt_image"].includes(generator.value)) { current.textContent = "Character references require Flow or GPT Image as the generator."; return; }
    if (["flow", "gpt_image"].includes(generator.value)) {
      try {
        const setup = await checkBrowserSetup();
        if (!setup.chrome_ready || !setup.node_ready || !setup.playwright_ready) {
          current.textContent = "Browser automation is not ready. Use Install Browser Automation, then open the provider login before creating the dataset.";
          browserSetupPanel.scrollIntoView({ behavior: "smooth", block: "center" });
          return;
        }
      } catch (_) { current.textContent = "Could not verify browser automation. Check setup before creating the dataset."; return; }
    }
    state.running = true; state.paused = false; state.stopped = false; state.results = []; state.characterAnchor = referenceIngredient; thumbs.innerHTML = ""; thumbs.style.display = "none"; completeButtons.style.display = "none"; controls.style.display = "flex"; create.disabled = true; stateTitle.textContent = "Creating Dataset";
    try {
      await ensureIdentity();
      for (let i = 0; i < jobs.length; i++) {
        await waitPaused(); if (state.stopped) break;
        const concept = jobs[i]; progressLabel.textContent = `${i} / ${jobs.length}`; bar.style.width = `${Math.round(i / jobs.length * 100)}%`; current.textContent = `Preparing ${i + 1}: ${concept}`;
        const p = await post("/vrgdg/lora_dataset/image_prompt", { ...llm(), art_style: style.value.trim(), trigger_phrase: phrase.value.trim(), concept });
        current.textContent = `Generating ${i + 1}: ${concept}`;
        const characterIngredients = datasetType.value === "character" && state.characterAnchor ? [state.characterAnchor] : [];
        const image = await generateImage(p.prompt, i + 1, characterIngredients); showImage(image);
        if (datasetType.value === "character" && !state.characterAnchor && ["flow", "gpt_image"].includes(generator.value)) state.characterAnchor = await imageIngredient(image);
        const wfPayload = workflowPayload(p.prompt, i + 1);
        if (datasetType.value === "ic_pair") {
          const transformation = editInstruction.value.trim();
          if (!transformation) throw new Error("Describe the edit transformation first.");
          current.textContent = `Editing target ${i + 1} of ${jobs.length}…`;
          const sourceIngredient = await imageIngredient(image);
          const editPrompt = `Edit the provided source image. ${transformation} Preserve the exact subject identity, pose, framing, camera angle, composition, lighting, clothing, and background unless the requested transformation specifically requires changing one of them. Return one edited image only.`;
          const target = await generateImage(editPrompt, i + 1, [sourceIngredient]); showImage(target);
          await post("/vrgdg/lora_dataset/save_ic_pair", { dataset_folder: folder.value.trim(), index: i + 1, reference: image, target, instruction: transformation });
          state.results.push({ image: target, reference: image, caption: transformation }); addThumb(target, i + 1); progressLabel.textContent = `${i + 1} / ${jobs.length}`; bar.style.width = `${Math.round((i + 1) / jobs.length * 100)}%`;
          continue;
        }
        current.textContent = `Writing caption ${i + 1} of ${jobs.length}…`; const cap = await post("/vrgdg/lora_dataset/caption", { ...llm(), image, trigger_word: trigger.value.trim(), trigger_phrase: phrase.value.trim() });
        await post("/vrgdg/lora_dataset/save_pair", { dataset_folder: folder.value.trim(), index: i + 1, image, caption: cap.caption, concept, prompt: p.prompt, art_style: style.value.trim(), trigger_word: trigger.value.trim(), trigger_phrase: phrase.value.trim(), generator: generator.value, seed: wfPayload.seed });
        state.results.push({ image, caption: cap.caption }); addThumb(image, i + 1); progressLabel.textContent = `${i + 1} / ${jobs.length}`; bar.style.width = `${Math.round((i + 1) / jobs.length * 100)}%`;
      }
      stateTitle.textContent = state.stopped ? "Dataset stopped" : "Dataset Complete"; current.textContent = `${state.results.length} image/caption pair${state.results.length === 1 ? "" : "s"} saved.`; thumbs.style.display = "grid"; completeButtons.style.display = "grid";
    } catch (e) { stateTitle.textContent = "Needs attention"; current.textContent = e.message; if (state.results.length) { thumbs.style.display = "grid"; completeButtons.style.display = "grid"; } }
    finally { state.running = false; controls.style.display = "none"; create.disabled = false; pause.textContent = "Pause"; }
  };
}

app.registerExtension({
  name: "vrgdg.lora.dataset.creator",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = original?.apply(this, args);
      if (!(this.widgets || []).some((w) => w?.name === "Open LoRA Dataset Creator")) this.addWidget("button", "Open LoRA Dataset Creator", null, openCreator);
      this.setSize?.([310, 82]);
      return result;
    };
  },
});
