import { app } from "../../scripts/app.js";

const NODE_NAME = "VRGDG_StartImageStoryboard";
const LAST_PROJECT = "vrgdg_start_storyboard_last_project";
const LAYOUT_KEY = "vrgdg_start_storyboard_layout";
const PRESETS = ["", "Extreme close-up", "Close-up", "Medium shot", "Full-body shot", "Wide shot", "Side profile", "Over-the-shoulder", "Low angle", "High angle", "Detail shot of hands", "Detail shot of feet"];
const TRANSITION_PRESETS = [
  ["", "End-frame transition preset (optional)"],
  ["wide_to_close", "Wide → close-up"],
  ["wide_to_medium", "Wide → medium"],
  ["medium_to_close", "Medium → close-up"],
  ["close_to_wide", "Close-up → wide reveal"],
  ["front_to_profile", "Front view → side profile"],
  ["front_to_mirror", "Front view → mirror view"],
  ["full_to_detail", "Full body → detail shot"],
  ["over_shoulder", "Wide/medium → over-the-shoulder"],
  ["low_to_high", "Low angle → high angle"],
];
const PROVIDERS = [
  ["gpt_image", "GPT"],
  ["flow_nano_banana", "Flow"],
  ["meta_ai", "Meta AI"],
];

function el(tag, css = "", text = "") {
  const node = document.createElement(tag);
  if (css) node.style.cssText = css;
  if (text) node.textContent = text;
  return node;
}

function button(text, primary = false) {
  const b = el("button", `border:1px solid ${primary ? "#0891b2" : "#475569"};border-radius:7px;background:${primary ? "#155e75" : "#1e293b"};color:#f8fafc;padding:8px 10px;font-weight:800;cursor:pointer;`, text);
  b.type = "button";
  return b;
}

function field(value = "", placeholder = "") {
  const input = el("input", "box-sizing:border-box;width:100%;border:1px solid #475569;border-radius:7px;background:#0f172a;color:#f8fafc;padding:8px;");
  input.value = value;
  input.placeholder = placeholder;
  return input;
}

function textarea(value = "", placeholder = "", rows = 4) {
  const input = el("textarea", "box-sizing:border-box;width:100%;resize:vertical;border:1px solid #475569;border-radius:7px;background:#0f172a;color:#f8fafc;padding:8px;line-height:1.4;");
  input.value = value;
  input.placeholder = placeholder;
  input.rows = rows;
  return input;
}

async function post(url, payload, timeout = 600000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: controller.signal });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) throw new Error(data.error || `${response.status} ${response.statusText}`);
    return data;
  } finally {
    clearTimeout(timer);
  }
}

function openStoryboardCreator() {
  const overlay = el("div", "position:fixed;inset:0;z-index:10020;background:#020617;color:#e2e8f0;font-family:Inter,Arial,sans-serif;overflow:hidden;");
  const state = {
    projectFolder: localStorage.getItem(LAST_PROJECT) || "",
    board: { global_idea: "", scenes: [] },
    provider: "gpt_image",
    runner: "builtin",
    model: "",
    lmstudioModel: "",
    apiProvider: "openai",
    apiModel: "",
    apiKey: "",
    layout: localStorage.getItem(LAYOUT_KEY) === "list" ? "list" : "grid",
  };

  const header = el("div", "height:68px;box-sizing:border-box;display:flex;gap:10px;align-items:center;padding:10px 16px;border-bottom:1px solid #334155;background:#0f172a;");
  const title = el("div", "font-size:20px;font-weight:950;color:#cffafe;white-space:nowrap;", "Start Image Storyboard · v1.13");
  const project = field(state.projectFolder, "Paste an existing Video Builder project folder...");
  const load = button("Load Video Builder Project", true);
  const reimport = button("Refresh Project Mappings");
  const save = button("Save");
  const settings = button("LLM Settings");
  const close = button("Close");
  header.append(title, project, load, reimport, save, settings, close);

  const globalBar = el("div", "display:grid;grid-template-columns:minmax(0,1fr) 180px 150px;gap:10px;padding:10px 16px;border-bottom:1px solid #334155;background:#111827;");
  const globalIdea = field("", "Optional global story idea or visual style used for every prompt...");
  const provider = el("select", "border:1px solid #475569;border-radius:7px;background:#0f172a;color:#f8fafc;padding:8px;");
  PROVIDERS.forEach(([value, label]) => { const option = el("option", "", label); option.value = value; provider.append(option); });
  const layout = el("select", "border:1px solid #475569;border-radius:7px;background:#0f172a;color:#f8fafc;padding:8px;");
  [["grid", "Grid / Tiles"], ["list", "List"]].forEach(([value, label]) => { const option = el("option", "", label); option.value = value; layout.append(option); });
  layout.value = state.layout;
  globalBar.append(globalIdea, provider, layout);
  const refBar = el("div", "display:flex;align-items:center;gap:10px;padding:9px 16px;border-bottom:1px solid #334155;background:#0f172a;");
  const globalRefPreview = el("img", "display:none;width:54px;height:54px;object-fit:cover;border:1px solid #475569;border-radius:7px;background:#020617;");
  const uploadGlobalRef = button("Upload Global Character Reference", true);
  const batchBrief = button("Batch Agent Brief", true);
  const useGlobalRef = el("input"); useGlobalRef.type = "checkbox"; useGlobalRef.checked = true;
  const useGlobalLabel = el("label", "display:flex;align-items:center;gap:7px;color:#cbd5e1;font-size:12px;font-weight:800;cursor:pointer;");
  useGlobalLabel.append(useGlobalRef, el("span", "", "Use global reference for every scene"));
  const useEndFrames = el("input"); useEndFrames.type = "checkbox";
  const useEndFramesLabel = el("label", "display:flex;align-items:center;gap:7px;color:#fde68a;font-size:12px;font-weight:900;cursor:pointer;");
  useEndFramesLabel.append(useEndFrames, el("span", "", "Start + End Frames"));
  const refNote = el("div", "margin-left:auto;color:#64748b;font-size:11px;", "Per-scene references override the global reference.");
  refBar.append(globalRefPreview, uploadGlobalRef, useGlobalLabel, useEndFramesLabel, batchBrief, refNote);
  const status = el("div", "padding:7px 16px;background:#172033;border-bottom:1px solid #334155;color:#94a3b8;font-size:12px;", "Choose a current Video Builder project to import its lyric segments.");
  const cards = el("div", "height:calc(100vh - 220px);overflow:auto;padding:14px;display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:14px;align-content:start;");
  overlay.append(header, globalBar, refBar, status, cards);
  document.body.append(overlay);

  const busy = (message, error = false) => {
    status.textContent = message;
    status.style.color = error ? "#fecaca" : "#a5f3fc";
  };

  const runnerPayload = () => ({
    text_runner: state.runner,
    model_file: state.model,
    n_ctx: 8000,
    n_gpu_layers: 99,
    lmstudio_base_url: "http://127.0.0.1:1234/v1",
    lmstudio_model: state.lmstudioModel,
    llm_api_provider: state.apiProvider,
    llm_api_model: state.apiModel,
    llm_api_key: state.apiKey,
  });

  const modalShell = (titleText, helpText = "") => {
    const shade = el("div", "position:fixed;inset:0;z-index:10040;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:20px;");
    const box = el("div", "width:min(700px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow:auto;border:1px solid #155e75;border-radius:9px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.65);padding:16px;display:flex;flex-direction:column;gap:12px;");
    const head = el("div", "display:flex;justify-content:space-between;align-items:flex-start;gap:12px;");
    const copy = el("div");
    copy.append(el("div", "font-size:17px;font-weight:950;color:#cffafe;", titleText), el("div", "font-size:12px;color:#94a3b8;margin-top:3px;line-height:1.4;", helpText));
    const x = button("Close");
    head.append(copy, x);
    box.append(head);
    shade.append(box);
    document.body.append(shade);
    const closeModal = () => shade.remove();
    x.onclick = closeModal;
    shade.onclick = (event) => { if (event.target === shade) closeModal(); };
    return { shade, box, closeModal };
  };

  const requestPromptEdit = (defaultText = "") => new Promise((resolve) => {
    const modal = modalShell("Edit Text-to-Image Prompt", "Tell the selected LLM exactly what to change. Everything else will be preserved.");
    const instruction = textarea(defaultText, "Example: Change this to a low-angle full-body shot while preserving the subject and setting.", 5);
    const actions = el("div", "display:grid;grid-template-columns:1fr 1fr;gap:8px;");
    const cancel = button("Cancel");
    const apply = button("Apply Edit", true);
    const finish = (value) => { modal.closeModal(); resolve(value); };
    cancel.onclick = () => finish("");
    apply.onclick = () => finish(instruction.value.trim());
    actions.append(cancel, apply);
    modal.box.append(instruction, actions);
    setTimeout(() => instruction.focus(), 0);
  });

  const openBatchAgentBrief = () => {
    if (!state.board.scenes?.length) return busy("Load a storyboard before creating a batch brief.", true);
    const saved = state.board.batch_agent_brief && typeof state.board.batch_agent_brief === "object" ? state.board.batch_agent_brief : {};
    const modal = modalShell("Batch Agent Brief", "Send several consecutive lyric scenes to the selected Browser AI as one continuity-focused music-video request.");
    const labeled = (label, control) => { const wrap = el("label", "display:grid;gap:5px;font-size:12px;font-weight:800;color:#bae6fd;"); wrap.append(el("span", "", label), control); return wrap; };
    const row = el("div", "display:grid;grid-template-columns:1fr 1fr;gap:10px;");
    const start = field(String(saved.start_scene || 1)); start.type = "number"; start.min = "1"; start.max = String(state.board.scenes.length);
    const end = field(String(saved.end_scene || Math.min(4, state.board.scenes.length))); end.type = "number"; end.min = "1"; end.max = String(state.board.scenes.length);
    row.append(labeled("First scene", start), labeled("Last scene", end));
    const location = textarea(saved.location || state.board.batch_location || "", "Fallback location only for selected scenes without a mapped project location...", 4);
    const direction = textarea(saved.direction || "", "Optional batch direction, such as: gradually increase intensity while keeping the same black dress and blue-gray lighting.", 3);
    const shotSequence = el("select", "width:100%;border:1px solid #475569;border-radius:7px;background:#0f172a;color:#f8fafc;padding:8px;");
    [["scene", "Use each scene card's shot preset"], ["cinematic", "Cinematic: wide → medium → close-up → detail"], ["character", "Character: close-up → full body → profile → over-shoulder"], ["intensity", "Build intensity: wide → medium → close-up → extreme close-up"], ["reveal", "Reveal environment: close-up → medium → full body → wide"]].forEach(([value, label]) => { const option = el("option", "", label); option.value = value; shotSequence.append(option); });
    shotSequence.value = saved.shot_sequence || "scene";
    const checks = el("div", "display:grid;grid-template-columns:1fr 1fr;gap:8px;border:1px solid #334155;border-radius:8px;background:#0f172a;padding:11px;");
    const check = (label, checked = true) => { const input = el("input"); input.type = "checkbox"; input.checked = checked; const wrap = el("label", "display:flex;align-items:center;gap:7px;font-size:12px;color:#cbd5e1;cursor:pointer;"); wrap.append(input, el("span", "", label)); checks.append(wrap); return input; };
    const sameIdentity = check("Same character identity", saved.same_identity !== false);
    const sameWardrobe = check("Same wardrobe", saved.same_wardrobe !== false);
    const sameStyle = check("Same visual style and lighting", saved.same_style !== false);
    const varyShots = check("Vary camera coverage", saved.vary_shots !== false);
    const usePreviousScene = check("Use previous scene image as continuity reference", Boolean(saved.use_previous_scene));
    const preview = textarea(saved.preview || "", "Generated Browser AI brief...", 15);
    let saveBriefTimer = null;
    const persistBrief = (writeToDisk = false) => {
      state.board.batch_location = location.value.trim();
      state.board.batch_agent_brief = {
        start_scene: Number(start.value) || 1,
        end_scene: Number(end.value) || Math.min(4, state.board.scenes.length),
        location: location.value,
        direction: direction.value,
        shot_sequence: shotSequence.value,
        same_identity: sameIdentity.checked,
        same_wardrobe: sameWardrobe.checked,
        same_style: sameStyle.checked,
        vary_shots: varyShots.checked,
        use_previous_scene: usePreviousScene.checked,
        preview: preview.value,
      };
      clearTimeout(saveBriefTimer);
      if (writeToDisk) return saveBoard(true);
      saveBriefTimer = setTimeout(() => saveBoard(true).catch((error) => busy(String(error.message || error), true)), 600);
      return Promise.resolve();
    };
    const sequenceFor = (count) => {
      const sets = { cinematic: ["wide shot", "medium shot", "close-up", "detail shot"], character: ["facial close-up", "full-body shot", "side profile", "over-the-shoulder shot"], intensity: ["wide shot", "medium shot", "close-up", "extreme close-up"], reveal: ["close-up", "medium shot", "full-body shot", "establishing wide shot"] };
      return sets[shotSequence.value]?.slice(0, count) || [];
    };
    const selectedScenes = () => {
      const first = Math.max(1, Math.min(state.board.scenes.length, Number(start.value) || 1));
      const last = Math.max(first, Math.min(state.board.scenes.length, Number(end.value) || first));
      return { first, last, scenes: state.board.scenes.slice(first - 1, last) };
    };
    const batchReferenceBundle = () => {
      const { first, scenes } = selectedScenes();
      const ingredients = [];
      const roles = [];
      const globalReference = String(state.board.global_reference_path || "").trim();
      if (globalReference) addUniqueIngredient(ingredients, roles, { path: globalReference, name: "character_reference.png" }, "the CHARACTER REFERENCE SHEET for identity, face, hair, body, and wardrobe");
      if (usePreviousScene.checked && first > 1) {
        const previous = state.board.scenes[first - 2];
        if (previous?.image_path) addUniqueIngredient(ingredients, roles, { path: previous.image_path, name: "previous_scene.png" }, "the completed PREVIOUS SCENE image for continuity only; do not copy its composition");
      }
      scenes.forEach((scene) => {
        if (ingredients.length >= 5) return;
        const locationRef = sceneLocationRef(scene);
        addUniqueIngredient(
          ingredients,
          roles,
          imageIngredient(sceneLocationImage(scene), `scene_${scene.number || "location"}_location.png`),
          `the MAPPED LOCATION REFERENCE named ${locationRef?.name || "location"}; use it only for scenes assigned to that location and treat it as navigable 3D space`,
        );
      });
      return { ingredients, roles };
    };
    const buildBrief = () => {
      const { first, scenes: selected } = selectedScenes();
      const shots = sequenceFor(selected.length);
      const sceneLines = selected.map((scene, index) => {
        const shot = shotSequence.value === "scene" ? scene.preset : shots[index];
        const endTransition = TRANSITION_PRESETS.find(([value]) => value === scene.end_transition_preset)?.[1] || scene.end_transition_preset || "change camera coverage and advance the action naturally";
        const paired = state.board.use_end_frames ? `\nStart frame: ${shot || "establish the shot"}${scene.note ? `; ${scene.note}` : ""}\nEnd frame: ${endTransition}; ${scene.end_frame_note || "advance the character pose or action naturally"}` : `${shot ? `\nCamera framing: ${shot}` : ""}${scene.note ? `\nScene direction: ${scene.note}` : ""}`;
        const mappedLocation = sceneLocationText(scene) || location.value.trim() || "[MISSING REQUIRED LOCATION]";
        const area = String(scene.location_area || "").trim() || "automatically choose a different believable sub-area from adjacent scenes";
        return `Scene ${first + index}\nRequired location: ${mappedLocation}\nRequired location sub-area: ${area}\nLyric inspiration: ${scene.lyric || "[instrumental / no lyric]"}${paired}`;
      }).join("\n\n");
      const continuity = [sameIdentity.checked ? "preserve the same character identity and face" : "", sameWardrobe.checked ? "preserve the same wardrobe, hair, and makeup" : "", sameStyle.checked ? "preserve one coherent visual style, color palette, time of day, and lighting logic" : "", varyShots.checked ? "vary the camera framing and composition so the images do not repeat the same shot" : ""].filter(Boolean).join("; ");
      const previousNote = usePreviousScene.checked
        ? "\n\nThe attachment guide identifies the completed image from the immediately preceding scene. Use it only as the visual continuity handoff: understand what just happened and continue naturally, but do not copy its composition unchanged."
        : "";
      const outputRequest = state.board.use_end_frames
        ? `Create ${selected.length * 2} separate images: one START FRAME and one END FRAME for each scene below. Within each pair, preserve the exact same person, face, hair, wardrobe, accessories, location, lighting, color palette, and visual style. The end frame occurs a few seconds later in the same continuous shot; change camera framing/viewpoint and advance the character pose or action in a physically plausible way.`
        : `Create ${selected.length} separate start images, one for each scene below, in the same order as the scene directions.`;
      const bundle = batchReferenceBundle();
      preview.value = `${attachmentGuide(bundle.roles)}\n\nWe are creating a continuous music-video storyboard. ${outputRequest} The images should feel like consecutive moments from one visual story.\n\nCharacter continuity: Use the attachment identified as the character reference sheet as the same person throughout. ${continuity || "Maintain visual continuity across the sequence."}.${previousNote}\n\nLOCATION SPATIAL CONTRACT: Every scene must use its individually listed mapped location. Treat every location reference as a real navigable three-dimensional environment, not a flat background plate. Place the character physically inside it with correct perspective, scale, floor contact, depth, contact shadows, reflections, environmental color spill, and natural occlusion. Move the camera through different believable sub-areas and do not repeat the same composition when a location is reused. Never paste, composite, cut out, green-screen, or layer the character over the location image.\n\n${location.value.trim() ? `Fallback location for any scene without a mapped project location: ${location.value.trim()}\n\n` : ""}${direction.value.trim() ? `Additional creative direction: ${direction.value.trim()}\n\n` : ""}${sceneLines}\n\nImportant output rules: Create separate images, not a contact sheet, collage, diptych, or split screen. Do not include captions, START/END labels, scene numbers, subtitles, lyrics, typography, signs, watermarks, borders, or written text inside any image.`.trim();
      persistBrief();
    };
    [start, end, location, direction, shotSequence, sameIdentity, sameWardrobe, sameStyle, varyShots, usePreviousScene].forEach((control) => { control.oninput = buildBrief; control.onchange = buildBrief; });
    const actions = el("div", "display:grid;grid-template-columns:1fr 1fr;gap:8px;");
    const rebuild = button("Rebuild Preview");
    const cancel = button("Cancel");
    const send = button("Send Reference + Brief", true);
    const sendWithPrevious = button("Send Previous Scene + Character Ref + Brief", true);
    rebuild.onclick = buildBrief;
    cancel.onclick = async () => { await persistBrief(true); modal.closeModal(); };
    send.onclick = async () => {
      try {
        const { scenes: selected } = selectedScenes();
        const missingLocations = selected.filter((scene) => !sceneLocationText(scene) && !location.value.trim());
        if (missingLocations.length) throw new Error(`Map locations for the selected scenes or enter a fallback location. ${missingLocations.length} scene${missingLocations.length === 1 ? " is" : "s are"} missing one.`);
        const referencePath = state.board.global_reference_path || "";
        if (!referencePath) throw new Error("Upload a global character reference before sending a batch brief.");
        if (usePreviousScene.checked) {
          const firstScene = Math.max(1, Math.min(state.board.scenes.length, Number(start.value) || 1));
          if (firstScene <= 1) throw new Error("Scene 1 has no previous scene. Turn off previous-scene continuity or begin at Scene 2 or later.");
          const previous = state.board.scenes[firstScene - 2];
          if (!previous?.image_path) throw new Error(`Scene ${firstScene - 1} has no start image to use as the previous-scene continuity reference.`);
        }
        const bundle = batchReferenceBundle();
        buildBrief();
        await persistBrief(true);
        busy(`Sending batch brief to ${provider.options[provider.selectedIndex].text}...`);
        await post("/vrgdg/browser_image/manual_upload", { provider: provider.value, prompt: preview.value.trim(), image_ingredients: bundle.ingredients });
        busy(`Character and mapped location references plus batch brief sent to ${provider.options[provider.selectedIndex].text}.`);
        modal.closeModal();
      } catch (error) { busy(String(error.message || error), true); }
    };
    sendWithPrevious.onclick = async () => {
      try {
        const { scenes: selected } = selectedScenes();
        const missingLocations = selected.filter((scene) => !sceneLocationText(scene) && !location.value.trim());
        if (missingLocations.length) throw new Error(`Map locations for the selected scenes or enter a fallback location. ${missingLocations.length} scene${missingLocations.length === 1 ? " is" : "s are"} missing one.`);
        const referencePath = state.board.global_reference_path || "";
        if (!referencePath) throw new Error("Upload a global character reference before sending a batch brief.");
        const firstScene = Math.max(1, Math.min(state.board.scenes.length, Number(start.value) || 1));
        if (firstScene <= 1) throw new Error("Scene 1 has no previous scene. Set First scene to 2 or later.");
        const previous = state.board.scenes[firstScene - 2];
        if (!previous?.image_path) throw new Error(`Scene ${firstScene - 1} has no start image to use as the previous-scene continuity reference.`);
        usePreviousScene.checked = true;
        buildBrief();
        const bundle = batchReferenceBundle();
        await persistBrief(true);
        busy(`Sending previous scene, character/location references, and batch brief to ${provider.options[provider.selectedIndex].text}...`);
        await post("/vrgdg/browser_image/manual_upload", { provider: provider.value, prompt: preview.value.trim(), image_ingredients: bundle.ingredients });
        busy(`Previous scene, character/location references, and batch brief sent to ${provider.options[provider.selectedIndex].text}.`);
        modal.closeModal();
      } catch (error) { busy(String(error.message || error), true); }
    };
    actions.append(rebuild, cancel, send, sendWithPrevious);
    modal.box.append(row, labeled("Fallback location for scenes without a mapped project location", location), labeled("Optional batch direction", direction), labeled("Shot progression", shotSequence), checks, labeled("Editable brief preview", preview), actions);
    preview.oninput = () => persistBrief();
    if (!saved.preview) buildBrief();
  };

  const openLlmSettings = async () => {
    const modal = modalShell("LLM Runner", "Choose the same text-prompt runner used by Video Builder: local Gemma GGUF, LM Studio, or an LLM API.");
    const select = (items, value = "") => {
      const node = el("select", "width:100%;border:1px solid #475569;border-radius:7px;background:#0f172a;color:#f8fafc;padding:8px;");
      items.forEach((item) => { const pair = Array.isArray(item) ? item : [item, item]; const option = el("option", "", pair[1]); option.value = pair[0]; node.append(option); });
      node.value = value;
      return node;
    };
    const labeled = (label, control) => { const wrap = el("label", "display:grid;gap:5px;font-size:12px;font-weight:800;color:#bae6fd;"); wrap.append(el("span", "", label), control); return wrap; };
    const runner = select([["builtin", "Gemma Local"], ["lm_studio", "LM Studio"], ["llm_api", "LLM API"]], state.runner);
    const panels = el("div");
    const builtin = el("div", "display:grid;gap:10px;border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;");
    const gemmaModel = select([[state.model, state.model || "Loading local Gemma models..."]], state.model);
    builtin.append(el("div", "font-size:12px;color:#cbd5e1;", "Choose the non-vision Gemma GGUF used to write image prompts."), labeled("Gemma GGUF model", gemmaModel));
    const lm = el("div", "display:grid;gap:10px;border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;");
    const baseUrl = field("http://127.0.0.1:1234/v1");
    const lmModel = field(state.lmstudioModel, "LM Studio model ID");
    const loadLm = button("Load LM Studio Models");
    lm.append(el("div", "font-size:12px;color:#cbd5e1;", "Start the LM Studio local server, then load and select its available chat model."), labeled("Base URL", baseUrl), labeled("Model", lmModel), loadLm);
    const api = el("div", "display:grid;gap:10px;border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;");
    const apiProvider = select([[state.apiProvider, state.apiProvider || "OpenAI"]], state.apiProvider);
    const apiModel = select([[state.apiModel, state.apiModel || "Loading API models..."]], state.apiModel);
    const apiKey = field(state.apiKey, "API key (kept only in this open UI session)"); apiKey.type = "password";
    api.append(el("div", "font-size:12px;color:#cbd5e1;", "Choose a configured Video Builder LLM API provider and model."), labeled("Provider", apiProvider), labeled("Model", apiModel), labeled("API key", apiKey));
    panels.append(builtin, lm, api);
    const showPanel = () => { builtin.style.display = runner.value === "builtin" ? "grid" : "none"; lm.style.display = runner.value === "lm_studio" ? "grid" : "none"; api.style.display = runner.value === "llm_api" ? "grid" : "none"; };
    runner.onchange = showPanel;
    const actions = el("div", "display:grid;grid-template-columns:1fr 1fr;gap:8px;");
    const cancel = button("Cancel");
    const saveRunner = button("Save Runner", true);
    actions.append(cancel, saveRunner);
    modal.box.append(labeled("Text LLM runner", runner), panels, actions);
    showPanel();
    cancel.onclick = modal.closeModal;
    saveRunner.onclick = () => {
      state.runner = runner.value;
      state.model = gemmaModel.value;
      state.lmstudioModel = lmModel.value.trim();
      state.apiProvider = apiProvider.value;
      state.apiModel = apiModel.value;
      state.apiKey = apiKey.value;
      busy(`LLM runner set to ${runner.options[runner.selectedIndex].text}.`);
      modal.closeModal();
    };
    loadLm.onclick = async () => {
      try {
        loadLm.disabled = true; loadLm.textContent = "Loading...";
        const data = await post("/vrgdg/music_builder/lm_studio_models", { lmstudio_base_url: baseUrl.value.trim(), lmstudio_api_key: "" }, 45000);
        const models = Array.isArray(data.models) ? data.models : [];
        if (!models.length) throw new Error("LM Studio returned no chat models.");
        lmModel.value = models[0];
        busy(`Loaded ${models.length} LM Studio model${models.length === 1 ? "" : "s"}.`);
      } catch (error) { busy(String(error.message || error), true); }
      finally { loadLm.disabled = false; loadLm.textContent = "Load LM Studio Models"; }
    };
    fetch("/vrgdg/music_builder/gemma_choices").then((r) => r.json()).then((data) => {
      const models = Array.isArray(data.models) ? data.models : [];
      gemmaModel.replaceChildren();
      models.forEach((name) => { const option = el("option", "", name); option.value = name; gemmaModel.append(option); });
      if (state.model && models.includes(state.model)) gemmaModel.value = state.model;
    }).catch(() => {});
    fetch("/vrgdg/music_builder/llm_api_choices").then((r) => r.json()).then((data) => {
      const providers = Array.isArray(data.providers) ? data.providers : [];
      apiProvider.replaceChildren();
      providers.forEach((item) => { const option = el("option", "", item.label || item.id); option.value = item.id; apiProvider.append(option); });
      if (providers.some((item) => item.id === state.apiProvider)) apiProvider.value = state.apiProvider;
      const fillModels = () => {
        const selected = providers.find((item) => item.id === apiProvider.value) || providers[0] || {};
        apiModel.replaceChildren();
        (selected.models || []).forEach((name) => { const option = el("option", "", name); option.value = name; apiModel.append(option); });
        if (state.apiModel && (selected.models || []).includes(state.apiModel)) apiModel.value = state.apiModel;
      };
      apiProvider.onchange = fillModels;
      fillModels();
    }).catch(() => {});
  };

  const saveBoard = async (quiet = false) => {
    state.projectFolder = project.value.trim();
    state.board.global_idea = globalIdea.value.trim();
    state.board.use_global_reference = useGlobalRef.checked;
    const data = await post("/vrgdg/start_storyboard/save", { project_folder: state.projectFolder, storyboard: state.board });
    if (!quiet) busy("Storyboard saved.");
    return data.storyboard;
  };

  const chooseImage = () => new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/png,image/jpeg,image/webp";
    input.onchange = () => resolve(input.files?.[0] || null);
    input.click();
  });

  const uploadReference = async (sceneNumber = null) => {
    const file = await chooseImage();
    if (!file) return;
    const dataUrl = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result || "")); reader.onerror = reject; reader.readAsDataURL(file); });
    busy(sceneNumber ? `Saving reference for scene ${sceneNumber}...` : "Saving global character reference...");
    const data = await post("/vrgdg/start_storyboard/save_reference", { project_folder: project.value.trim(), scene_number: sceneNumber, image_name: file.name, image_data: dataUrl }, 120000);
    state.board = data.storyboard;
    render();
    busy(sceneNumber ? `Scene ${sceneNumber} reference saved.` : "Global character reference saved.");
  };

  const saveDroppedSceneImage = async (file, sceneNumber, scene, frame = "start") => {
    if (!file || !/^image\/(png|jpeg|webp)$/i.test(file.type || "")) throw new Error("Drop a PNG, JPEG, or WebP image.");
    const dataUrl = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result || "")); reader.onerror = reject; reader.readAsDataURL(file); });
    busy(`Saving dropped ${frame} frame into scene ${sceneNumber}...`);
    const data = await post("/vrgdg/start_storyboard/save_scene_upload", { project_folder: project.value.trim(), scene_number: sceneNumber, frame, image_name: file.name, image_data: dataUrl }, 120000);
    if (frame === "end") {
      scene.end_image_path = data.saved_path;
      scene.end_image_url = data.image_url;
    } else {
      scene.image_path = data.saved_path;
      scene.image_url = data.image_url;
    }
    await saveBoard(true);
    render();
    busy(`Dropped image saved as scene ${sceneNumber}'s ${frame} frame.`);
  };

  const sceneLocationRef = (scene) => scene?.location_ref && typeof scene.location_ref === "object" ? scene.location_ref : null;
  const sceneLocationImage = (scene) => {
    const location = sceneLocationRef(scene);
    const image = location?.image && typeof location.image === "object" ? location.image : location || {};
    return {
      path: String(image.path || location?.image_path || "").trim(),
      data: String(image.data || location?.image_data || "").trim(),
      name: String(image.name || location?.image_name || location?.name || "mapped_location.png").trim() || "mapped_location.png",
    };
  };
  const sceneLocationText = (scene) => {
    const location = sceneLocationRef(scene);
    if (!location) return "";
    return [String(location.name || "").trim(), String(location.description || "").trim()].filter(Boolean).join(" — ");
  };
  const imageIngredient = (image, fallbackName) => {
    const source = image && typeof image === "object" ? image : {};
    const path = String(source.path || "").trim();
    const data = String(source.data || "").trim();
    if (!path && !data) return null;
    return { path, data, name: String(source.name || fallbackName || "reference.png").trim() || "reference.png" };
  };
  const characterReferenceIngredient = (scene) => {
    const path = String(scene?.reference_path || (state.board.use_global_reference !== false ? state.board.global_reference_path : "") || "").trim();
    return path ? { path, name: "character_reference.png" } : null;
  };
  const addUniqueIngredient = (ingredients, roles, ingredient, role) => {
    if (!ingredient) return;
    const key = ingredient.path || ingredient.data || ingredient.name;
    if (!key || ingredients.some((item) => (item.path || item.data || item.name) === key)) return;
    ingredients.push(ingredient);
    roles.push(role);
  };
  const sceneReferenceBundle = (scene, options = {}) => {
    const ingredients = [];
    const roles = [];
    if (options.includeStart && scene?.image_path) {
      addUniqueIngredient(ingredients, roles, { path: scene.image_path, name: "existing_start_frame.png" }, "the existing START FRAME and primary scene truth");
    }
    if (options.includeEnd && scene?.end_image_path) {
      addUniqueIngredient(ingredients, roles, { path: scene.end_image_path, name: "existing_end_frame.png" }, "the existing END FRAME for the same continuous shot");
    }
    if (options.includeCharacter !== false) {
      addUniqueIngredient(ingredients, roles, characterReferenceIngredient(scene), "the CHARACTER REFERENCE SHEET; use it for identity, face, hair, body, and wardrobe—not as a background or pose template");
    }
    if (options.includeLocation !== false) {
      addUniqueIngredient(ingredients, roles, imageIngredient(sceneLocationImage(scene), "mapped_location.png"), "the MAPPED LOCATION REFERENCE; treat it as a real navigable 3D environment, not a flat background plate");
    }
    return { ingredients, roles };
  };
  const attachmentGuide = (roles = []) => roles.length
    ? `ATTACHMENT ROLES — follow these exactly:\n${roles.map((role, index) => `Attachment ${index + 1}: ${role}.`).join("\n")}\nDo not merge the attachments into a collage or treat the character sheet as the environment.`
    : "";
  const spatialLocationContract = (scene, sceneIndex = -1) => {
    const location = sceneLocationRef(scene);
    if (!location) return "";
    const locationText = sceneLocationText(scene) || "the mapped project location";
    const area = String(scene.location_area || "").trim();
    const previous = sceneIndex > 0 ? state.board.scenes?.[sceneIndex - 1] : null;
    const previousLocation = sceneLocationRef(previous);
    const sameAsPrevious = Boolean(previousLocation && (
      (location.id && previousLocation.id && String(location.id) === String(previousLocation.id))
      || (!location.id && !previousLocation.id && String(location.name || "").trim().toLowerCase() === String(previousLocation.name || "").trim().toLowerCase())
    ));
    return [
      `REQUIRED PHYSICAL LOCATION: ${locationText}. Do not replace, rename, or move the scene to a different environment.`,
      area
        ? `REQUIRED LOCATION SUB-AREA FOR THIS SCENE: ${area}. Compose the shot from that part of the environment.`
        : "Choose a believable sub-area within this location that has not just been used by an adjacent scene; vary where inside the environment the action takes place.",
      "Treat the mapped location as a real, navigable three-dimensional space. Reconstruct its architecture, floor plane, walls, depth, materials, landmarks, lighting sources, and plausible unseen space. The camera may move to another position, height, direction, lens, or side of the environment while preserving the location's identity.",
      "Place the referenced character physically inside the location—not pasted, composited, cut out, green-screened, or layered over the location image. Match perspective, scale, floor contact, depth, contact shadows, reflections, environmental color spill, focus, and natural foreground/background occlusion.",
      sameAsPrevious
        ? "The immediately previous scene uses this same mapped location. Use a substantially different sub-area and composition. Change at least the camera position, viewing direction, shot size, subject placement, and foreground/background arrangement. Do not create a near-duplicate of the previous scene."
        : "Use a composition designed for this scene rather than copying the location reference image's exact camera position.",
    ].join("\n");
  };

  const contextFor = (scene, sceneIndex = -1) => [
    globalIdea.value.trim() ? `Global story or visual idea:\n${globalIdea.value.trim()}` : "",
    spatialLocationContract(scene, sceneIndex),
    scene.lyric ? `Current lyric segment (visual inspiration only; do not quote it in the final still-image prompt):\n${scene.lyric}` : "",
    scene.note ? `User scene note:\n${scene.note}` : "",
    scene.preset ? `Required shot preset:\n${scene.preset}` : "",
    state.board.use_end_frames && scene.end_transition_preset ? `Required start-to-end framing transition:\n${scene.end_transition_preset}` : "",
    state.board.use_end_frames && scene.end_frame_note ? `Required end-frame action/composition:\n${scene.end_frame_note}` : "",
  ].filter(Boolean).join("\n\n");

  const browserPromptFor = (scene, withImages = false, sceneIndex = -1, attachmentRoles = []) => {
    const base = String(scene.prompt || "").trim();
    const guide = attachmentGuide(attachmentRoles);
    const spatial = spatialLocationContract(scene, sceneIndex);
    if (!state.board.use_end_frames) return [guide, spatial, base].filter(Boolean).join("\n\n");
    const transition = TRANSITION_PRESETS.find(([value]) => value === scene.end_transition_preset)?.[1] || scene.end_transition_preset || "Change the camera framing or viewpoint naturally";
    const endDirection = String(scene.end_frame_note || "").trim() || "Advance the character's pose or action naturally while changing the camera coverage.";
    const hasSceneFrameAttachments = attachmentRoles.some((role) => /START FRAME|END FRAME/i.test(role));
    const attached = withImages && hasSceneFrameAttachments
      ? "Use any attached scene frames as visual continuity references. Preserve their visible identity and environment; do not combine the attachments into one image."
      : "";
    return [
      guide,
      spatial,
      "Create TWO separate 16:9 cinematic still images for one continuous video shot: a START FRAME and an END FRAME.",
      "Both images must show the exact same person, face, hair, wardrobe, accessories, physical location, time of day, lighting logic, color palette, and visual style. The end frame happens a few seconds after the start frame in the same continuous scene.",
      "Change the camera angle, framing, viewpoint, and/or character pose or action between the two frames so a video model can create meaningful motion between them. Keep the transition physically plausible.",
      attached,
      `START FRAME direction:\n${base || scene.note || "Establish the character and location."}${scene.preset ? `\nCamera framing: ${scene.preset}` : ""}`,
      `END FRAME direction:\nFraming transition: ${transition}\n${endDirection}`,
      "Return two separate images, not a collage, diptych, split screen, contact sheet, or image containing both frames. Do not add captions, START/END labels, scene numbers, lyrics, typography, signs, subtitles, watermarks, borders, or written text inside either image.",
    ].filter(Boolean).join("\n\n");
  };

  const startOptionsPromptFor = (scene, sceneIndex = -1, attachmentRoles = []) => {
    const base = String(scene.prompt || scene.note || "").trim();
    return [
      attachmentGuide(attachmentRoles),
      spatialLocationContract(scene, sceneIndex),
      "Using the attached CHARACTER REFERENCE SHEET and MAPPED LOCATION REFERENCE, create FIVE separate new 16:9 START-FRAME OPTIONS for this scene.",
      "Place the character naturally and physically within the location in five clearly different areas, poses, and compositions. Vary the camera position, viewing direction, height, angle, shot size, lens feeling, subject placement, and foreground/background arrangement for every option.",
      "Preserve the exact character identity, face, hair, body, wardrobe, accessories, and the mapped location's recognizable architecture, materials, landmarks, lighting logic, and atmosphere in all five options.",
      "Treat the location as a navigable three-dimensional environment. Integrate the character into it with correct perspective, scale, floor contact, depth, contact shadows, reflections, environmental color spill, focus, and natural occlusion. Never paste, composite, cut out, green-screen, or layer the character over the location image.",
      base ? `Scene direction that every option must interpret:\n${base}` : "Scene direction: Create five compelling opening compositions suitable as alternative start frames for this scene.",
      scene.preset ? `Preferred shot guidance to interpret with meaningful variation rather than repeating one composition:\n${scene.preset}` : "Use meaningfully different cinematic shot coverage across the five options.",
      "These are five alternative START FRAME choices, not five chronological moments and not start/end pairs. Each option must stand alone as a possible opening image for the same scene.",
      "Return five separate images, not a collage, grid, contact sheet, diptych, split screen, or one image containing multiple panels. Do not add captions, option numbers, START labels, scene numbers, lyrics, typography, signs, subtitles, watermarks, borders, or written text inside any image.",
    ].filter(Boolean).join("\n\n");
  };

  const endOnlyPromptFor = (scene, sceneIndex = -1, attachmentRoles = []) => {
    const transition = TRANSITION_PRESETS.find(([value]) => value === scene.end_transition_preset)?.[1] || scene.end_transition_preset || "Change the camera framing or viewpoint naturally";
    const endDirection = String(scene.end_frame_note || "").trim() || "Advance the character's pose or action naturally.";
    return [
      attachmentGuide(attachmentRoles),
      spatialLocationContract(scene, sceneIndex),
      "Create exactly ONE new 16:9 END FRAME for a continuous video shot.",
      "The attached existing START FRAME is the primary visual truth. Continue from it a few seconds later. Preserve the exact same person, face, hair, wardrobe, accessories, physical location, time of day, lighting logic, color palette, and visual style.",
      `Required framing transition: ${transition}.`,
      `Required end-frame action/composition: ${endDirection}`,
      scene.prompt ? `Original scene direction for context only:\n${scene.prompt}` : "",
      "Generate only the later END FRAME. Do not regenerate the start frame. Do not create a collage, diptych, split screen, comparison, contact sheet, or before-and-after layout. Do not add START/END labels, captions, scene numbers, lyrics, typography, subtitles, signs, watermarks, borders, or written text.",
    ].filter(Boolean).join("\n\n");
  };

  const render = () => {
    globalIdea.value = state.board.global_idea || "";
    useGlobalRef.checked = state.board.use_global_reference !== false;
    useEndFrames.checked = Boolean(state.board.use_end_frames);
    globalRefPreview.src = state.board.global_reference_url || "";
    globalRefPreview.style.display = state.board.global_reference_url ? "block" : "none";
    cards.replaceChildren();
    cards.style.gridTemplateColumns = state.layout === "list" ? "minmax(0,1fr)" : "repeat(auto-fill,minmax(380px,1fr))";
    if (!state.board.scenes?.length) {
      cards.append(el("div", "color:#94a3b8;padding:30px;", "No scenes loaded."));
      return;
    }
    state.board.scenes.forEach((scene, index) => {
      const mappedLocation = sceneLocationRef(scene);
      const mappedLocationImage = sceneLocationImage(scene);
      const mappedLocationHasImage = Boolean(mappedLocationImage.path || mappedLocationImage.data);
      // Use plain divs and explicit !important layout rules because some ComfyUI
      // themes globally collapse/skin semantic section elements.
      const card = el("div", state.layout === "list"
        ? "display:grid !important;grid-template-columns:260px minmax(0,1fr) !important;grid-template-rows:auto minmax(0,560px) !important;height:600px !important;min-height:600px !important;max-height:600px !important;border:1px solid #334155;border-radius:12px;background:#111827;overflow:hidden !important;box-shadow:0 10px 24px #0005;"
        : "display:flex !important;flex-direction:column !important;height:820px !important;min-height:820px !important;max-height:820px !important;border:1px solid #334155;border-radius:12px;background:#111827;overflow:hidden !important;box-shadow:0 10px 24px #0005;");
      const cardHead = el("div", "display:flex;justify-content:space-between;align-items:center;padding:9px 11px;background:#1e293b;border-bottom:1px solid #334155;");
      if (state.layout === "list") cardHead.style.cssText += "grid-column:1/-1 !important;";
      cardHead.append(
        el("strong", "color:#cffafe;", `Scene ${String(index + 1).padStart(3, "0")}`),
        el("span", `max-width:58%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;border:1px solid ${mappedLocation ? "#0e7490" : "#475569"};border-radius:999px;background:${mappedLocation ? "#083344" : "#1e293b"};color:${mappedLocation ? "#a5f3fc" : "#94a3b8"};padding:3px 8px;font-size:10px;font-weight:900;`, mappedLocation ? `Location: ${mappedLocation.name || "Mapped"}` : "No mapped location"),
      );
      const imageBox = el("div", state.layout === "list"
        ? `display:grid !important;grid-template-rows:${state.board.use_end_frames ? "1fr 1fr" : "1fr"};width:260px !important;height:100% !important;min-height:320px !important;background:#020617;border-right:1px solid #334155;`
        : `display:grid !important;grid-template-columns:${state.board.use_end_frames ? "1fr 1fr" : "1fr"};flex:0 0 240px !important;height:240px !important;min-height:240px !important;background:#020617;border-bottom:1px solid #334155;`);
      const makeFrameDrop = (frame) => {
        const isEnd = frame === "end";
        const url = isEnd ? scene.end_image_url : scene.image_url;
        const slot = el("div", `position:relative;display:flex;min-width:0;min-height:0;align-items:center;justify-content:center;background:#020617;overflow:hidden;${state.board.use_end_frames ? (state.layout === "list" ? "border-bottom:1px solid #334155;" : "border-right:1px solid #334155;") : ""}`);
        const badge = el("div", "position:absolute;left:7px;top:7px;z-index:2;padding:3px 7px;border-radius:5px;background:#0f172add;color:#cffafe;font-size:10px;font-weight:900;pointer-events:none;", `${isEnd ? "END" : "START"} FRAME`);
        slot.append(badge);
        if (url) { const img = el("img", "width:100%;height:100%;object-fit:contain;"); img.src = url; slot.append(img); }
        else slot.append(el("div", "padding:18px;color:#64748b;font-weight:800;text-align:center;line-height:1.5;white-space:pre-line;", `No ${frame} frame yet\nDrop an image here`));
        slot.ondragover = (event) => { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; slot.style.background = "#083344"; slot.style.outline = "2px dashed #22d3ee"; slot.style.outlineOffset = "-5px"; };
        slot.ondragleave = (event) => { if (event.relatedTarget && slot.contains(event.relatedTarget)) return; slot.style.background = "#020617"; slot.style.outline = "none"; };
        slot.ondrop = (event) => { event.preventDefault(); slot.style.background = "#020617"; slot.style.outline = "none"; const file = Array.from(event.dataTransfer?.files || []).find((item) => /^image\//i.test(item.type || "")); saveDroppedSceneImage(file, index + 1, scene, frame).catch((error) => busy(String(error.message || error), true)); };
        slot.onclick = async () => { const file = await chooseImage(); if (file) saveDroppedSceneImage(file, index + 1, scene, frame).catch((error) => busy(String(error.message || error), true)); };
        slot.style.cursor = "pointer";
        slot.title = `Click or drop an image for the ${frame} frame`;
        return slot;
      };
      imageBox.append(makeFrameDrop("start"));
      if (state.board.use_end_frames) imageBox.append(makeFrameDrop("end"));
      const cardBody = el("div", "display:grid !important;grid-template-columns:minmax(0,1fr) !important;grid-auto-rows:max-content !important;align-content:start !important;flex:1 1 auto !important;gap:9px !important;height:auto !important;min-height:0 !important;max-height:none !important;padding:11px 14px 28px 11px !important;box-sizing:border-box !important;visibility:visible !important;opacity:1 !important;overflow-x:hidden !important;overflow-y:scroll !important;scrollbar-gutter:stable !important;overscroll-behavior:contain !important;");
      const lyric = el("div", "min-height:38px;border-left:3px solid #0891b2;padding:7px 9px;background:#0f172a;color:#e2e8f0;white-space:pre-wrap;line-height:1.35;", scene.lyric || "(No lyric text)");
      const locationPanel = el("div", `display:grid;grid-template-columns:${mappedLocation && (scene.location_image_url || mappedLocationHasImage) ? "64px minmax(0,1fr)" : "minmax(0,1fr)"};gap:9px;align-items:start;border:1px solid ${mappedLocation ? "#0e7490" : "#475569"};border-radius:8px;background:${mappedLocation ? "#082f49" : "#0f172a"};padding:9px;`);
      if (mappedLocation && (scene.location_image_url || mappedLocationImage.data)) {
        const locationPreview = el("img", "width:64px;height:64px;object-fit:cover;border:1px solid #155e75;border-radius:7px;background:#020617;");
        locationPreview.src = scene.location_image_url || mappedLocationImage.data;
        locationPreview.title = "Mapped Video Builder location reference";
        locationPanel.append(locationPreview);
      }
      const locationCopy = el("div", "display:grid;gap:5px;min-width:0;");
      if (mappedLocation) {
        locationCopy.append(
          el("div", "font-size:12px;font-weight:950;color:#cffafe;", mappedLocation.name || "Mapped project location"),
          el("div", "font-size:11px;color:#bae6fd;line-height:1.4;white-space:pre-wrap;", mappedLocation.description || "No location description was saved; the reference image will be used as the environment."),
          el("div", `font-size:10px;font-weight:800;color:${mappedLocationHasImage ? "#86efac" : "#fde68a"};`, mappedLocationHasImage ? "Location image will be sent with the character reference." : "No location image is available; Browser AI will receive the location description only."),
        );
      } else {
        locationCopy.append(
          el("div", "font-size:12px;font-weight:900;color:#cbd5e1;", "No location mapped in the Video Builder project"),
          el("div", "font-size:11px;color:#94a3b8;line-height:1.4;", "Map a Reference Builder location to this scene, save the Video Builder project, then use Refresh Project Mappings."),
        );
      }
      const locationArea = field(scene.location_area || "", "Optional location sub-area, e.g. rear booth, bar counter, entrance, left side of stage...");
      locationArea.title = "Choose a distinct area inside the mapped location. Leave blank to tell Browser AI to select a different believable area automatically.";
      locationCopy.append(locationArea);
      locationPanel.append(locationCopy);
      const note = field(scene.note || "", "Optional scene idea or note...");
      const preset = el("select", "width:100%;border:1px solid #475569;border-radius:7px;background:#0f172a;color:#f8fafc;padding:8px;");
      PRESETS.forEach((name) => { const option = el("option", "", name || "Shot preset (optional)"); option.value = name; preset.append(option); });
      preset.value = scene.preset || "";
      const endTransition = el("select", "width:100%;border:1px solid #a16207;border-radius:7px;background:#0f172a;color:#fde68a;padding:8px;");
      TRANSITION_PRESETS.forEach(([value, label]) => { const option = el("option", "", label); option.value = value; endTransition.append(option); });
      endTransition.value = scene.end_transition_preset || "";
      const endFrameNote = field(scene.end_frame_note || "", "End frame: character action, pose, destination, or composition...");
      const prompt = textarea(scene.prompt || "", "Type a text-to-image prompt or let the LLM create one...", 6);
      const promptActions = el("div", "display:grid;grid-template-columns:1fr 1fr;gap:7px;");
      const createPrompt = button("LLM Create Prompt", true);
      const editPrompt = button("LLM Edit Prompt");
      promptActions.append(createPrompt, editPrompt);
      const browserActions = el("div", "display:grid !important;grid-template-columns:minmax(0,1fr) minmax(0,1fr) !important;gap:7px !important;height:auto !important;overflow:visible !important;");
      const sendPrompt = button("Send Prompt");
      const sendImage = button(state.board.use_end_frames ? "Send Start + End + Prompt" : "Send Image + Prompt");
      const createEndFromStart = button("Create End from Start", true);
      const importLatest = button(state.board.use_end_frames ? "Import Latest as Start" : "Import Latest", true);
      const importLatestEnd = button("Import Latest as End", true);
      browserActions.append(sendPrompt, sendImage, importLatest);
      if (state.board.use_end_frames) browserActions.append(createEndFromStart, importLatestEnd);
      else importLatest.style.gridColumn = "1 / -1";
      for (const action of [sendPrompt, sendImage, createEndFromStart, importLatest, importLatestEnd]) {
        action.style.minHeight = "40px";
        action.style.whiteSpace = "normal";
        action.style.lineHeight = "1.2";
      }
      const referenceActions = el("div", "display:grid;grid-template-columns:1fr 1fr;gap:7px;");
      const uploadSceneRef = button(scene.reference_path ? "Replace Scene Reference" : "Upload Scene Reference");
      const sendReference = button("Send Character + Location + Prompt", true);
      const createStartOptions = button("Create 5 Start Options", true);
      sendReference.title = "Send the selected character reference sheet, this scene's mapped location reference image, and the spatial-integration prompt to Browser AI.";
      createStartOptions.title = "Send the character and mapped location references to Browser AI and request five separate, compositionally different start-frame choices.";
      createStartOptions.style.gridColumn = "1 / -1";
      createStartOptions.style.minHeight = "42px";
      referenceActions.append(uploadSceneRef, sendReference, createStartOptions);
      const characterReference = characterReferenceIngredient(scene);
      const referenceLabel = el("div", "font-size:11px;color:#94a3b8;line-height:1.4;", [
        characterReference ? (scene.reference_path ? "Scene character reference ready." : "Global character reference ready.") : "No character reference selected.",
        mappedLocationHasImage ? "Mapped location reference ready." : (mappedLocation ? "Mapped location description ready; no image." : "No mapped location."),
      ].join(" "));
      cardBody.append(lyric, locationPanel, note, preset);
      if (state.board.use_end_frames) cardBody.append(endTransition, endFrameNote);
      cardBody.append(prompt, promptActions, referenceLabel, referenceActions, browserActions);
      card.append(cardHead, imageBox, cardBody);
      cards.append(card);

      const sync = () => { scene.location_area = locationArea.value.trim(); scene.note = note.value.trim(); scene.preset = preset.value; scene.end_transition_preset = endTransition.value; scene.end_frame_note = endFrameNote.value.trim(); scene.prompt = prompt.value.trim(); };
      locationArea.onchange = note.onchange = preset.onchange = endTransition.onchange = endFrameNote.onchange = prompt.onchange = sync;

      createPrompt.onclick = async () => {
        try {
          sync();
          busy(`Creating prompt for scene ${index + 1}...`);
          const data = await post("/vrgdg/music_builder/generate_t2i", { ...runnerPayload(), prompt_mode: "flow_gpt", user_notes: contextFor(scene, index), unload_after: true });
          scene.prompt = String(data.prompt || data.text || "").trim();
          prompt.value = scene.prompt;
          await saveBoard(true);
          busy(`Scene ${index + 1} prompt created.`);
        } catch (error) { busy(String(error.message || error), true); }
      };

      editPrompt.onclick = async () => {
        sync();
        if (!scene.prompt) return busy("Create or type a prompt before editing it.", true);
        const instruction = await requestPromptEdit(scene.preset ? `Apply a ${scene.preset} while preserving everything else.` : "");
        if (!instruction) return;
        try {
          busy(`Editing prompt for scene ${index + 1}...`);
          const data = await post("/vrgdg/music_builder/edit_image_prompt", { ...runnerPayload(), current_prompt: scene.prompt, edit_request: instruction, prompt_mode: "flow_gpt", use_full_scene_context: true, scene_context: { label: `Scene ${index + 1}`, lyric_text: scene.lyric, scene_notes: scene.note }, unload_after: true });
          scene.prompt = String(data.prompt || data.text || "").trim();
          prompt.value = scene.prompt;
          await saveBoard(true);
          busy(`Scene ${index + 1} prompt edited.`);
        } catch (error) { busy(String(error.message || error), true); }
      };

      const send = async (withImage) => {
        try {
          sync();
          if (withImage && !scene.image_path) throw new Error("This scene has no start frame to send.");
          const bundle = withImage
            ? sceneReferenceBundle(scene, { includeStart: true, includeEnd: Boolean(state.board.use_end_frames && scene.end_image_path) })
            : { ingredients: [], roles: [] };
          const outgoingPrompt = browserPromptFor(scene, withImage, index, bundle.roles);
          if (outgoingPrompt) await navigator.clipboard?.writeText(outgoingPrompt).catch(() => {});
          busy(`Opening ${provider.options[provider.selectedIndex].text} for scene ${index + 1}...`);
          if (withImage) {
            await post("/vrgdg/browser_image/manual_upload", { provider: provider.value, prompt: outgoingPrompt, image_ingredients: bundle.ingredients });
            busy(outgoingPrompt
              ? `Scene ${index + 1} frames, mapped references, and prompt sent to ${provider.options[provider.selectedIndex].text}.`
              : `Scene ${index + 1} image uploaded to ${provider.options[provider.selectedIndex].text}. You can now chat with the agent.`);
          } else {
            if (outgoingPrompt) {
              await post("/vrgdg/browser_image/manual_upload", { provider: provider.value, prompt: outgoingPrompt, image_ingredients: [] });
              busy(`Scene ${index + 1} prompt sent to ${provider.options[provider.selectedIndex].text}.`);
            } else {
              await post("/vrgdg/browser_image/manual_open", { provider: provider.value });
              busy(`${provider.options[provider.selectedIndex].text} opened for scene ${index + 1}. This scene has no prompt yet, so you can chat manually.`);
            }
          }
        } catch (error) { busy(String(error.message || error), true); }
      };
      sendPrompt.onclick = () => send(false);
      sendImage.onclick = () => send(true);
      createEndFromStart.onclick = async () => {
        try {
          sync();
          if (!scene.image_path) throw new Error("Add the scene's start frame first.");
          const bundle = sceneReferenceBundle(scene, { includeStart: true });
          const outgoingPrompt = endOnlyPromptFor(scene, index, bundle.roles);
          await navigator.clipboard?.writeText(outgoingPrompt).catch(() => {});
          busy(`Sending scene ${index + 1}'s start frame to create one end frame...`);
          await post("/vrgdg/browser_image/manual_upload", { provider: provider.value, prompt: outgoingPrompt, image_ingredients: bundle.ingredients });
          busy(`Start frame, character/location references, and end-frame direction sent to ${provider.options[provider.selectedIndex].text}. Download the result, then use Import Latest as End.`);
        } catch (error) { busy(String(error.message || error), true); }
      };
      uploadSceneRef.onclick = () => uploadReference(index + 1).catch((error) => busy(String(error.message || error), true));
      sendReference.onclick = async () => {
        try {
          sync();
          const bundle = sceneReferenceBundle(scene);
          const outgoingPrompt = browserPromptFor(scene, true, index, bundle.roles);
          if (!bundle.ingredients.length && !outgoingPrompt) throw new Error("Add a character reference, mapped location, or prompt first.");
          busy(`Sending character, mapped location, and prompt for scene ${index + 1}...`);
          await post("/vrgdg/browser_image/manual_upload", { provider: provider.value, prompt: outgoingPrompt, image_ingredients: bundle.ingredients });
          busy(outgoingPrompt
            ? `Character/location references and scene ${index + 1} prompt sent to ${provider.options[provider.selectedIndex].text}.`
            : `Available scene references uploaded to ${provider.options[provider.selectedIndex].text}. You can now tell the agent what to create.`);
        } catch (error) { busy(String(error.message || error), true); }
      };
      createStartOptions.onclick = async () => {
        try {
          sync();
          const characterIngredient = characterReferenceIngredient(scene);
          const locationIngredient = imageIngredient(sceneLocationImage(scene), "mapped_location.png");
          if (!characterIngredient) throw new Error("Upload or enable a character reference sheet before creating start options.");
          if (!locationIngredient) throw new Error("This scene needs a mapped location reference image before creating start options.");
          const bundle = sceneReferenceBundle(scene);
          const outgoingPrompt = startOptionsPromptFor(scene, index, bundle.roles);
          await navigator.clipboard?.writeText(outgoingPrompt).catch(() => {});
          busy(`Sending character/location references to create five start options for scene ${index + 1}...`);
          await post("/vrgdg/browser_image/manual_upload", { provider: provider.value, prompt: outgoingPrompt, image_ingredients: bundle.ingredients });
          busy(`Five varied start-frame options requested from ${provider.options[provider.selectedIndex].text} for scene ${index + 1}. Download your chosen image, then click Import Latest as Start.`);
        } catch (error) { busy(String(error.message || error), true); }
      };
      const importFrame = async (frame) => {
        try {
          sync();
          busy(`Importing latest ${provider.options[provider.selectedIndex].text} download as scene ${index + 1}'s ${frame} frame...`);
          const data = await post("/vrgdg/start_storyboard/import_latest", { project_folder: project.value.trim(), provider: provider.value, scene_number: index + 1, frame });
          if (frame === "end") { scene.end_image_path = data.saved_path; scene.end_image_url = data.image_url; }
          else { scene.image_path = data.saved_path; scene.image_url = data.image_url; }
          await saveBoard(true);
          render();
          busy(`Latest image imported as scene ${index + 1}'s ${frame} frame.`);
        } catch (error) { busy(String(error.message || error), true); }
      };
      importLatest.onclick = () => importFrame("start");
      importLatestEnd.onclick = () => importFrame("end");
    });
  };

  const loadProject = async (refresh = false) => {
    try {
      state.projectFolder = project.value.trim();
      if (!state.projectFolder) throw new Error("Paste an existing Video Builder project folder first.");
      busy(refresh ? "Refreshing lyrics and Reference Builder location mappings..." : "Loading standalone storyboard and project mappings...");
      const data = await post(refresh ? "/vrgdg/start_storyboard/reimport" : "/vrgdg/start_storyboard/load", { project_folder: state.projectFolder });
      state.board = data.storyboard;
      localStorage.setItem(LAST_PROJECT, state.projectFolder);
      render();
      const mappedLocations = Number(state.board.imported_location_count || state.board.scenes.filter((scene) => sceneLocationRef(scene)).length || 0);
      busy(`Loaded ${state.board.scenes.length} scene cards with ${mappedLocations} mapped project location${mappedLocations === 1 ? "" : "s"}.`);
    } catch (error) { busy(String(error.message || error), true); }
  };

  const chooseVideoBuilderProject = async () => {
    try {
      busy("Finding Video Builder projects...");
      const response = await fetch("/vrgdg/music_builder/list_projects");
      const data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.error || "Could not list Video Builder projects.");
      const projects = Array.isArray(data.projects) ? data.projects : [];
      return await new Promise((resolve) => {
        const shade = el("div", "position:fixed;inset:0;z-index:10030;background:#000b;display:flex;align-items:center;justify-content:center;padding:20px;");
        const box = el("div", "width:min(760px,calc(100vw - 40px));max-height:min(760px,calc(100vh - 40px));display:flex;flex-direction:column;gap:10px;border:1px solid #0e7490;border-radius:12px;background:#111827;padding:15px;box-shadow:0 24px 80px #000c;");
        const heading = el("div", "font-size:18px;font-weight:950;color:#cffafe;", "Choose a Video Builder Project");
        const note = el("div", "color:#cbd5e1;font-size:12px;line-height:1.45;", "Only folders containing a valid vrgdg_builder_session.json are shown.");
        const list = el("div", "display:flex;flex-direction:column;gap:8px;overflow:auto;min-height:100px;max-height:520px;");
        const actions = el("div", "display:flex;justify-content:flex-end;gap:8px;");
        const browse = button("Browse for Another Project Folder");
        const cancel = button("Cancel");
        const finish = (value) => { shade.remove(); resolve(value || ""); };
        if (!projects.length) list.append(el("div", "padding:18px;border:1px dashed #475569;border-radius:8px;color:#94a3b8;text-align:center;", "No Video Builder projects were found in the ComfyUI output folder."));
        projects.forEach((item) => {
          const row = el("button", "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;text-align:left;border:1px solid #334155;border-radius:8px;background:#0f172a;color:#f8fafc;padding:11px;cursor:pointer;");
          row.type = "button";
          const info = el("div");
          info.append(el("div", "font-weight:900;color:#e0f2fe;", item.name || "Video Builder project"), el("div", "margin-top:4px;font-size:11px;color:#67e8f9;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", item.project_folder || ""));
          row.append(info, el("div", "font-size:12px;color:#cbd5e1;", `${Number(item.scene_count || 0)} scenes`));
          row.onclick = () => finish(item.project_folder);
          list.append(row);
        });
        browse.onclick = async () => {
          try {
            const picked = await post("/vrgdg/music_builder/pick_path", { kind: "project_folder" }, 120000);
            if (!picked.path) return;
            // Loading performs the authoritative session-file validation.
            finish(picked.path);
          } catch (error) { busy(String(error.message || error), true); }
        };
        cancel.onclick = () => finish("");
        shade.onclick = (event) => { if (event.target === shade) finish(""); };
        actions.append(browse, cancel);
        box.append(heading, note, list, actions);
        shade.append(box);
        document.body.append(shade);
      });
    } catch (error) {
      busy(String(error.message || error), true);
      return "";
    }
  };

  settings.onclick = openLlmSettings;
  provider.onchange = () => { state.provider = provider.value; };
  uploadGlobalRef.onclick = () => uploadReference(null).catch((error) => busy(String(error.message || error), true));
  batchBrief.onclick = openBatchAgentBrief;
  useGlobalRef.onchange = () => { state.board.use_global_reference = useGlobalRef.checked; saveBoard(true).then(render).catch((error) => busy(String(error.message || error), true)); };
  useEndFrames.onchange = () => { state.board.use_end_frames = useEndFrames.checked; saveBoard(true).then(render).catch((error) => busy(String(error.message || error), true)); };
  layout.onchange = () => {
    state.layout = layout.value === "list" ? "list" : "grid";
    localStorage.setItem(LAYOUT_KEY, state.layout);
    render();
  };
  load.onclick = async () => {
    const chosen = await chooseVideoBuilderProject();
    if (!chosen) return;
    project.value = chosen;
    await loadProject(false);
  };
  reimport.onclick = () => loadProject(true);
  save.onclick = () => saveBoard(false).catch((error) => busy(String(error.message || error), true));
  close.onclick = () => overlay.remove();
  project.addEventListener("keydown", (event) => { if (event.key === "Enter") loadProject(false); });
  if (state.projectFolder) loadProject(false);
}

app.registerExtension({
  name: "vrgdg.StartImageStoryboard",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      this.size = [320, 92];
      this.addWidget("button", "Open Storyboard Creator", null, openStoryboardCreator);
    };
  },
});

window.VRGDGStartImageStoryboard = { open: openStoryboardCreator };
