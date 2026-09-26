const ICONS = {
  wand: '<path d="m15 4 5 5L7 22l-5-5Z"/><path d="m14 5 5 5M6 2v6M3 5h6M19 14v6M16 17h6"/>',
  music: '<path d="M9 18V5l12-3v13M9 8l12-3"/><ellipse cx="6" cy="18" rx="3" ry="3"/><ellipse cx="18" cy="15" rx="3" ry="3"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m3 17 6-6 4 4 3-3 5 5"/>',
  wave: '<path d="M2 10v4M6 5v14M10 2v20M14 7v10M18 4v16M22 9v6"/>',
  film: '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M7 3v18M17 3v18M3 8h4M3 16h4M17 8h4M17 16h4"/>',
  save: '<path d="M5 3h12l4 4v14H3V3h2ZM7 3v6h10V3M7 21v-8h10v8"/>',
};

function node(tag, className = "", text = "") {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

function icon(name) {
  const wrap = node("span", "wb-icon");
  wrap.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
  return wrap;
}

function button(label, action, primary = false) {
  const control = node("button", primary ? "wb-button wb-primary" : "wb-button", label);
  control.type = "button";
  control.onclick = action;
  return control;
}

function field(label, control, help = "") {
  const wrap = node("label", "wb-field");
  wrap.append(node("span", "", label), control);
  if (help) wrap.append(node("small", "wb-muted", help));
  return wrap;
}

function details(title, ...children) {
  const wrap = node("details", "wb-details");
  wrap.append(node("summary", "", title), ...children);
  return wrap;
}

function styles() {
  if (document.getElementById("vrgdg-wizard-beta-style")) return;
  const sheet = node("style");
  sheet.id = "vrgdg-wizard-beta-style";
  sheet.textContent = `
    .wb-backdrop { position:fixed;inset:0;z-index:100005;background:#02060bcc;display:grid;place-items:center;padding:18px; }
    .wb-dialog { width:min(1440px,100%);height:min(960px,calc(100dvh - 36px));background:#17212c;color:#eef3fa;border:1px solid #425263;border-radius:14px;box-shadow:0 24px 90px #0009;display:flex;flex-direction:column;overflow:hidden;font:14px/1.5 system-ui,sans-serif; }
    .wb-dialog * { box-sizing:border-box; }
    .wb-header,.wb-footer { display:flex;align-items:center;gap:18px;padding:18px 24px;background:#141e28;flex-shrink:0; }
    .wb-header { border-bottom:1px solid #344353; }
    .wb-footer { border-top:1px solid #344353; }
    .wb-brand { display:flex;align-items:center;gap:14px; }
    .wb-brand h2 { font-size:23px;line-height:1.3;margin:0; }
    .wb-badge { font-size:10px;letter-spacing:1px;background:#29465e;border:1px solid #47687e;padding:3px 6px;border-radius:4px;vertical-align:middle;margin-left:8px; }
    .wb-icon { display:inline-flex;width:26px;height:26px;flex-shrink:0;color:#62d9f4;vertical-align:middle; }
    .wb-icon svg { width:100%;height:100%; }
    .wb-brand>.wb-icon { width:34px;height:34px; }
    .wb-muted { color:#a6b7ca;font-size:12px; }
    .wb-spacer { flex:1;min-width:0; }
    .wb-steps { display:flex;gap:18px;align-items:center;justify-content:center;flex:1; }
    .wb-step { border:0;background:none;color:#a6b7ca;padding:8px;cursor:pointer;white-space:nowrap; }
    .wb-step[aria-current="step"] { color:#76e5ff; }
    .wb-step:disabled { opacity:.45;cursor:default; }
    .wb-body { flex:1;min-height:0;overflow:auto; }
    .wb-setup { display:grid;grid-template-columns:minmax(0,1.9fr) minmax(300px,1fr);min-height:100%; }
    .wb-main { padding:24px 30px;display:flex;flex-direction:column;gap:25px; }
    .wb-aside { padding:24px;border-left:1px solid #344353;background:#1b2632;display:flex;flex-direction:column;gap:18px; }
    .wb-aside h3 { font-size:20px;margin:0; }
    .wb-section h3 { margin:0 0 4px;font-size:18px;display:flex;align-items:center;gap:12px; }
    .wb-section>p { margin:0 0 14px 42px;color:#a6b7ca;font-size:12px; }
    .wb-number { width:30px;height:30px;display:inline-grid;place-items:center;border:1px solid #556c82;border-radius:50%;font-size:13px;color:#d8eafa;background:#253545; }
    .wb-pair { display:grid;grid-template-columns:1fr 1fr;gap:14px; }
    .wb-engine { display:flex;align-items:center;gap:16px;text-align:left;padding:18px;background:#1d2936;border:1px solid #4b5e72;border-radius:8px;color:#f0f5fa;cursor:pointer; }
    .wb-engine strong { display:block;font-size:19px; }
    .wb-engine[aria-pressed="true"] { border-color:#63d8fa;box-shadow:inset 0 0 0 1px #63d8fa;background:#1b3242; }
    .wb-upload { border:1px solid #45586a;border-radius:8px;padding:16px;background:#1d2936;display:flex;align-items:center;gap:14px;flex-wrap:wrap; }
    .wb-upload strong { overflow-wrap:anywhere; }
    .wb-upload .wb-button { margin-left:auto; }
    .wb-reference { border:1px solid #45586a;border-radius:8px;padding:14px;background:#1d2936;min-width:0; }
    .wb-thumbnails { display:flex;gap:8px;overflow:auto;margin:12px 0;min-height:100px;align-items:center; }
    .wb-thumbnails img { width:104px;height:100px;object-fit:cover;border:1px solid #506071;border-radius:5px; }
    .wb-empty { min-height:100px;width:100%;display:flex;align-items:center;justify-content:center;gap:12px;color:#93a9bd;border:1px dashed #506579;border-radius:6px;font-size:12px; }
    .wb-button { display:inline-flex;align-items:center;justify-content:center;gap:8px;background:#1b2733;border:1px solid #64778b;border-radius:6px;color:#f1f5fa;padding:10px 16px;cursor:pointer;font:600 13px system-ui;white-space:nowrap; }
    .wb-button:hover { background:#2c4053;border-color:#a3d4e5; }
    .wb-button:disabled { opacity:.5;cursor:not-allowed; }
    .wb-dialog[aria-busy="true"] .wb-button:disabled { cursor:wait; }
    .wb-primary { background:#55d7f6;color:#062431;border-color:#88e9ff; }
    .wb-primary:hover { background:#8be7fb;color:#062431; }
    .wb-field { display:flex;flex-direction:column;gap:7px;font-size:12px;color:#e4edf5; }
    .wb-dialog input:not([type="checkbox"]),.wb-dialog textarea,.wb-dialog select { width:100%;min-width:0;border:1px solid #536b81;background:#15202b;color:#f0f5fa;border-radius:6px;padding:10px 12px;font:13px/1.5 system-ui; }
    .wb-dialog textarea { resize:vertical;min-height:90px; }
    .wb-dialog :focus-visible { outline:2px solid #68e0ff;outline-offset:3px; }
    .wb-details { border:1px solid #405366;border-radius:7px;padding:12px; }
    .wb-details summary { cursor:pointer;font-weight:600;font-size:13px; }
    .wb-details[open]>*+* { margin-top:14px; }
    .wb-check { display:flex;align-items:center;gap:9px;font-size:12px; }
    .wb-status { padding:10px 24px;white-space:pre-wrap;font-size:13px;background:#122e3b;color:#d7f5ff;border-top:1px solid #345160; }
    .wb-status:empty { display:none; }
    .wb-status[data-error="true"] { background:#43232c;color:#ffd5da; }
    .wb-review { padding:24px; }
    .wb-review h3 { margin-top:0; }
    .wb-scene { display:grid;grid-template-columns:145px minmax(0,1fr);gap:18px;border:1px solid #405366;padding:16px;border-radius:8px;margin:12px 0; }
    .wb-scene-fields { display:grid;grid-template-columns:1fr 1fr;gap:12px; }
    .wb-scene-fields>.wb-field:last-child { grid-column:1/-1; }
    .wb-save-status { font-size:12px;color:#b1c6d6; }
    .wb-project { max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;color:#bccbd8; }
    .wb-render-card { max-width:700px;margin:30px auto;padding:30px;border:1px solid #486076;border-radius:10px;background:#1d2b39; }
    @media(max-width:950px) { .wb-setup{grid-template-columns:1fr}.wb-aside{border-left:0;border-top:1px solid #344353}.wb-header{flex-wrap:wrap}.wb-steps{order:2;flex-basis:100%}.wb-project{display:none} }
    @media(max-width:600px) { .wb-backdrop{padding:6px}.wb-dialog{height:calc(100dvh - 12px)}.wb-main,.wb-aside,.wb-review{padding:16px}.wb-pair,.wb-scene,.wb-scene-fields{grid-template-columns:1fr}.wb-footer{flex-wrap:wrap;padding:12px}.wb-footer>.wb-muted{flex-basis:100%}.wb-header{padding:12px}.wb-header .wb-button{padding:9px}.wb-steps{gap:0;justify-content:space-between} }
  `;
  document.head.append(sheet);
}

export const WIZARD_BETA_STEPS = ["Engine", "Video mode", "Inputs", "Models & LoRAs", "Sound & timing", "Scenes", "Render"];

export function wizardBetaNeeds(engine, mode) {
  return {
    images: engine === "minimax_h3" ? ["image_to_video", "image_reference_to_video"].includes(mode) : ["i2v", "flf", "id_lora"].includes(mode),
    references: engine === "minimax_h3" ? ["reference_to_video", "image_reference_to_video"].includes(mode) : ["rtv", "ingredients", "id_lora"].includes(mode),
    endFrame: engine === "ltx" && mode === "flf",
    video: engine === "minimax_h3" && mode === "video_to_video",
  };
}

export function wizardBetaIssues(draft, snapshot, preparing = false) {
  const issues = [];
  if (!(snapshot.modes[draft.engine] || []).some(mode => mode.value === draft.mode && !mode.disabled)) issues.push("Choose a supported video mode.");
  if (!snapshot.audioModes.some(item => item.value === draft.audioMode && !item.disabled)) issues.push("Choose a supported audio source for this mode and pass configuration.");
  if (preparing) {
    if (draft.audioMode === "input_audio" && !snapshot.audioPath && !draft.song) issues.push("Choose an audio file or another supported audio source.");
    const needs = wizardBetaNeeds(draft.engine, draft.mode);
    if (needs.references && !(draft.subjects?.length || draft.singer) && !draft.locations.length && !snapshot.referenceCount) issues.push("Add a reference image in Inputs or the reference editor.");
    if (needs.video && !draft.videoPath.trim() && !snapshot.hasVideoReference) issues.push("Choose a source video in Inputs.");
  }
  return issues;
}

export function openWizardBeta(api) {
  styles();
  const initial = api.snapshot();
  const draft = {
    singer: null, subjects: initial.subjects, locations: initial.locations || [], endImage: null,
    imageSource: "generate", videoPath: "", characters: "", locationsText: "", sound: "", ...initial.draft,
    engine: initial.engine, mode: initial.mode, imageMode: initial.imageMode, audioMode: initial.audioMode,
    performance: initial.performance, lyrics: initial.lyrics || "", direction: initial.direction || "", song: null,
  };
  draft.subjects = (draft.subjects || (draft.singer ? [draft.singer] : [])).map(image => ({ ...image }));
  draft.singer = draft.subjects[0] || null;
  draft.removedReferenceIds = [];
  draft.locations = draft.locations.map(image => ({ ...image }));
  let page = Math.max(0, Math.min(6, Number(initial.draft?.page) || 0));
  const sceneSteps = ["Scene Defaults", "Story direction", "Story Layer", "Align lyrics / dialogue (Optional)", "Edit Mappings", "Storyboard Scenes"];
  let sceneStep = Math.max(0, Math.min(sceneSteps.length - 1, Number(initial.draft?.sceneStep) || 0));
  let busy = false, dirty = false;
  let releasePanel = () => {};
  const sceneEdits = new Map();
  const previousFocus = document.activeElement;
  const backdrop = node("div", "wb-backdrop");
  const dialog = node("div", "wb-dialog");
  dialog.setAttribute("role", "dialog"); dialog.setAttribute("aria-modal", "true"); dialog.setAttribute("aria-label", "Wizard Beta");
  const header = node("header", "wb-header");
  const title = node("h2", "", "Wizard"); title.append(node("span", "wb-badge", "BETA"));
  const brand = node("div", "wb-brand"); brand.append(icon("wand"), title);
  const steps = node("nav", "wb-steps"); steps.setAttribute("aria-label", "Wizard steps");
  // Seven stages remain readable on narrow windows without shrinking the controls.
  steps.style.cssText = "flex:0 0 auto;overflow:auto;justify-content:flex-start;gap:4px;padding:10px 18px;border-bottom:1px solid #344353;";
  const status = node("div", "wb-status"); status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
  const body = node("main", "wb-body");
  const footer = node("footer", "wb-footer");
  const saveState = node("span", "wb-save-status", initial.draft ? "Saved draft loaded" : "Using Builder settings");
  const close = button("Close", () => {
    if (busy || (dirty && !window.confirm("Close Wizard Beta? Unsaved wizard inputs will be lost; live Builder setting changes remain applied."))) return;
    releasePanel(); api.flush(); backdrop.remove(); previousFocus?.focus();
  });
  header.append(brand, node("span", "wb-spacer"), saveState, button("Configure LLM Runner", () => api.openRunner()), button("Save Project", save), close);
  dialog.append(header, steps, body, status, footer); backdrop.append(dialog); document.body.append(backdrop);

  function report(message, error = false) { status.textContent = message; status.dataset.error = String(error); }
  function changed() { dirty = true; saveState.textContent = "Unsaved wizard inputs"; }
  function syncDraft() {
    api.flush();
    const current = api.snapshot();
    for (const key of ["engine", "mode", "imageMode", "audioMode", "performance"]) draft[key] = current[key];
  }
  async function run(action) {
    if (busy) return;
    busy = true; dialog.setAttribute("aria-busy", "true");
    const controls = [...dialog.querySelectorAll("button,input,textarea,select")];
    const disabled = controls.map(control => control.disabled);
    controls.forEach(control => { control.disabled = true; });
    try { return await action(); }
    catch (error) { report(String(error.message || error), true); return false; }
    finally { busy = false; dialog.setAttribute("aria-busy", "false"); controls.forEach((control, index) => { control.disabled = disabled[index]; }); }
  }
  async function persist() {
    syncDraft();
    const issues = wizardBetaIssues(draft, api.snapshot());
    if (issues.length) throw new Error(issues.join("\n"));
    draft.page = page; draft.sceneStep = sceneStep;
    draft.singer = draft.subjects[0] || null;
    await api.save(draft, sceneEdits);
    draft.removedReferenceIds = [];
    sceneEdits.clear(); draft.song = null; dirty = false; saveState.textContent = "Saved to project";
    return true;
  }
  async function save() { return run(async () => { await persist(); report("Project and wizard progress saved."); return true; }); }
  function editReferences(section) {
    return run(async () => {
      await persist();
      api.openReferences(section, () => {
        const saved = api.snapshot().draft;
        draft.subjects = (saved?.subjects || (saved?.singer ? [saved.singer] : [])).map(image => ({ ...image }));
        draft.singer = draft.subjects[0] || null;
        draft.locations = (saved?.locations || []).map(image => ({ ...image }));
        render();
      });
    });
  }
  function input(value, update, multiline = false, type = "text") {
    const control = node(multiline ? "textarea" : "input");
    if (!multiline) control.type = type;
    control.value = value ?? "";
    control.oninput = () => { update(type === "number" ? Number(control.value) : control.value); changed(); };
    return control;
  }
  function select(value, options, update) {
    const control = node("select");
    for (const item of options) { const option = node("option", "", item.label); option.value = item.value; option.disabled = Boolean(item.disabled); control.append(option); }
    control.value = value;
    control.onchange = () => { update(control.value); changed(); render(); };
    return control;
  }
  function configure(values) { api.configure(values); syncDraft(); }
  function upload(label, key, multiple = false, accept = "image/png,image/jpeg,image/webp") {
    const wrap = node("div", "wb-reference"); wrap.append(node("strong", "", label));
    const picker = node("input"); picker.type = "file"; picker.accept = accept; picker.multiple = multiple; picker.hidden = true;
    const images = multiple ? draft[key] : (draft[key] ? [draft[key]] : []);
    const reference = key === "subjects" || key === "locations";
    let replaceIndex = null;
    const thumbs = node("div", "wb-thumbnails");
    if (reference) thumbs.style.cssText = "flex-direction:column;align-items:stretch;overflow:visible;";
    for (const [index, item] of images.entries()) {
      const image = node("img"); image.src = item.data || (item.path ? api.imageUrl(item.path) : ""); image.alt = item.title || item.name;
      if (!reference) { thumbs.append(image); continue; }
      const entry = node("div", "wb-field");
      const prefix = key === "subjects" ? (index ? `Character ${index + 1}` : "Character") : `Location ${index + 1}`;
      entry.append(image,
        field(`${prefix} title`, input(item.title ?? item.name.replace(/\.[^.]+$/, ""), value => { item.title = value; })),
        field(`${prefix} description`, input(item.description || "", value => { item.description = value; }, true)),
      );
      entry.append(button("Replace", () => { replaceIndex = index; picker.multiple = false; picker.click(); }), button("Remove", () => {
        if (item.referenceId) draft.removedReferenceIds.push(item.referenceId);
        draft[key].splice(index, 1); draft.singer = draft.subjects[0] || null; changed(); render();
      }));
      thumbs.append(entry);
    }
    if (!images.length) { const empty = node("div", "wb-empty", "Choose an image"); empty.prepend(icon("image")); thumbs.append(empty); }
    picker.onchange = async () => {
      const files = Array.from(picker.files || []); if (!files.length) return;
      await run(async () => {
        const values = await Promise.all(files.map(file => new Promise((resolve, reject) => {
          const reader = new FileReader(); reader.onload = () => {
            const previous = reference ? (replaceIndex === null ? null : images[replaceIndex]) : images[0];
            resolve({ ...(previous?.referenceId ? { referenceId: previous.referenceId } : {}), name: file.name, path: "", data: reader.result, ...(reference ? { title: previous?.title ?? file.name.replace(/\.[^.]+$/, ""), description: previous?.description || "" } : {}) });
          }; reader.onerror = () => reject(new Error(`Could not read ${file.name}`)); reader.readAsDataURL(file);
        })));
        if (reference) {
          if (replaceIndex === null) draft[key].push(...values);
          else draft[key][replaceIndex] = values[0];
          draft.singer = draft.subjects[0] || null;
        } else draft[key] = multiple ? values : values[0];
        changed();
      }); render();
    };
    wrap.append(picker, thumbs, button(reference ? "Add Images" : images.length ? "Replace" : "Choose image", () => { replaceIndex = null; picker.multiple = multiple; picker.click(); }));
    if (!reference && images.length) wrap.append(button("Clear", () => { draft[key] = multiple ? [] : null; changed(); render(); }));
    return wrap;
  }
  function mount(parent, kind) {
    const holder = node("div", "wb-native-settings");
    parent.append(holder); releasePanel = api.mountSettings(holder, kind);
    holder.addEventListener("input", () => { saveState.textContent = "Builder settings updated · Save Project to keep progress"; });
  }
  function modes() { return api.snapshot().modes[draft.engine]; }
  function modeLabel() { return modes().find(mode => mode.value === draft.mode)?.label || draft.mode; }
  function render() {
    releasePanel(); releasePanel = () => {}; syncDraft();
    steps.replaceChildren(); body.replaceChildren(); footer.replaceChildren();
    for (const [index, label] of WIZARD_BETA_STEPS.entries()) {
      const control = button(`${index + 1} ${label}`, () => { page = index; render(); });
      control.className = "wb-step"; if (index === page) control.setAttribute("aria-current", "step");
      steps.append(control);
    }
    const main = node("div", "wb-main");
    main.append(node("h2", "", `${page + 1}. ${WIZARD_BETA_STEPS[page]}`));
    const needs = wizardBetaNeeds(draft.engine, draft.mode);
    if (page === 0) {
      main.append(node("p", "wb-muted", "Choose the engine. The next steps will only show options for that engine."));
      const pair = node("div", "wb-pair");
      for (const [value, label, symbol] of [["ltx", "LTX", "film"], ["minimax_h3", "MiniMax H3", "wave"]]) {
        const control = button(label, () => { configure({ engine: value }); changed(); render(); });
        control.className = "wb-engine"; control.prepend(icon(symbol)); control.setAttribute("aria-pressed", String(draft.engine === value)); pair.append(control);
      }
      main.append(pair);
      if (draft.engine === "ltx") main.append(field("LTX version", select(api.snapshot().ltxVersion, [{ value: "2.5", label: "LTX 2.5" }, { value: "2.3", label: "LTX 2.3" }], value => configure({ ltxVersion: value })), "Changing version applies the Builder’s matching model defaults."));
    } else if (page === 1) {
      main.append(field("Video mode", select(draft.mode, modes(), value => configure({ mode: value }))));
      const selected = modes().find(mode => mode.value === draft.mode);
      main.append(node("p", "", selected?.description || ""));
      main.append(node("p", "wb-muted", "Your selection updates the existing Builder. You can go back and change it at any time."));
    } else if (page === 2) {
      main.append(node("p", "wb-muted", `Set up ${modeLabel()}. These inputs guide new scenes; existing scene assignments are preserved until you choose to replace the timeline.`));
      if (!needs.references) main.append(field("Characters (optional)", input(draft.characters, value => { draft.characters = value; }, true)), field("Locations (optional)", input(draft.locationsText, value => { draft.locationsText = value; }, true)));
      if (needs.images) {
        main.append(field("Starting images", select(draft.imageSource, [{ value: "generate", label: "Generate scene images" }, { value: "upload", label: "Use my own images" }], value => { draft.imageSource = value; })));
        if (draft.imageSource === "generate") {
          main.append(field("Text-to-image model", select(draft.imageMode, api.snapshot().imageModes, value => configure({ imageMode: value }))));
          const advanced = details("Image model settings and LoRAs"); main.append(advanced); mount(advanced, "image");
        } else main.append(node("p", "wb-muted", "After creating scenes, upload an image for each scene in Storyboard Scenes or fill the timeline from a folder of numbered images."));
      }
      if (needs.endFrame) main.append(upload("End frame (optional when generated)", "endImage"));
      if (needs.references) {
        const pair = node("div", "wb-pair"); pair.append(upload("Subject references", "subjects", true), upload("Location references", "locations", true)); main.append(pair);
      }
      const referenceActions = node("div");
      referenceActions.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;";
      for (const [section, label] of [["subjects", "Edit Subjects"], ["locations", "Edit Locations"]]) {
        referenceActions.append(button(label, () => editReferences(section)));
      }
      main.append(referenceActions, node("p", "wb-hint", "Opening an editor saves your wizard inputs. These editors share subjects, locations and scene mappings with the main Reference Builder."));
      if (["ingredients", "id_lora"].includes(draft.mode)) main.append(button(draft.mode === "ingredients" ? "Edit Ingredients sheets" : "Edit ID-LoRA references", () => api.openReferences()));
      if (needs.video) main.append(field("Source video path", input(draft.videoPath, value => { draft.videoPath = value; }), "Full path on the ComfyUI machine. Additional video references, trimming and purpose are available in Models & LoRAs."));
      if (draft.mode === "ingredients") main.append(node("p", "wb-muted", "Use the reference editor to create your ingredients sheet and assign it to scenes."));
      if (draft.mode === "id_lora") main.append(node("p", "wb-muted", "Configure the identity LoRA and reference voice in the next step."));
    } else if (page === 3) {
      main.append(node("p", "wb-muted", "Choose project-wide models, LoRAs, quality, passes and video settings. Scenes without custom overrides use these settings. Use the normal timeline for per-scene overrides and Storyboard Scenes for prompts."));
      mount(main, "video");
    } else if (page === 4) {
      main.append(field("Performance", select(draft.performance, api.snapshot().performances, value => configure({ performance: value }))));
      main.append(field("Audio source", select(draft.audioMode, api.snapshot().audioModes, value => configure({ audioMode: value }))));
      main.append(node("p", "wb-muted", api.snapshot().audioHelp));
      if (draft.audioMode === "input_audio") {
        const picker = node("input"); picker.type = "file"; picker.accept = "audio/*,.wav,.mp3,.flac,.m4a,.ogg"; picker.hidden = true;
        picker.onchange = () => { if (picker.files?.[0]) { draft.song = picker.files[0]; changed(); render(); } };
        main.append(picker, node("div", "wb-upload", draft.song?.name || api.snapshot().audioPath || "No audio selected"), button("Choose audio", () => picker.click()));
        main.append(field("Lyrics or dialogue (optional)", input(draft.lyrics, value => { draft.lyrics = value; }, true), "Used as scene notes. You can align and map words in the scene review step."));
      } else {
        if (draft.audioMode !== "silent") main.append(field("Dialogue, music and sound direction", input(draft.sound, value => { draft.sound = value; }, true)));
      }
      const inputAudio = draft.audioMode === "input_audio";
      main.append(node("h3", "", inputAudio ? "Step 1: Transcribe" : "Scene timing"));
      main.append(node("p", "wb-muted", inputAudio
        ? "Choose how to time your scenes. Each option opens the editor for that task."
        : "Built-in audio and silence do not need transcription. Use manual timing to create scenes."));
      const choices = node("div", "wb-field");
      for (const [kind, label, description] of [
        ["existing", "1. Existing scenes", "Keep current scene timing and transcribe the audio into those scenes."],
        ["new", "2. No scenes yet", "Transcribe the audio and create scenes from the line timestamps."],
        ["manual", "3. Manual timing", inputAudio ? "Listen to your audio and tap where each scene should split." : "Enter your own scene timings in the manual editor."],
      ]) {
        const card = node("div", "wb-reference");
        const choose = button(label, async () => {
          await run(async () => {
            await persist();
            await api.openTiming(kind);
            draft.lyrics = api.snapshot().lyrics || "";
            report("Timing editor finished. Continue to Scenes to review your timeline.");
          });
          render();
        });
        const hasAudio = Boolean(draft.song || api.snapshot().audioPath);
        const sceneCount = api.snapshot().scenes.length;
        const unavailable = kind !== "manual" && !inputAudio ? "Select Input Audio to transcribe. Built-in audio and silence use Manual timing."
          : inputAudio && !hasAudio ? "Choose an audio file above first."
          : kind === "existing" && !sceneCount ? "There are no scenes to transcribe yet. Choose No scenes yet or Manual timing."
          : "";
        choose.disabled = Boolean(unavailable);
        choose.title = unavailable;
        card.append(choose, node("p", "wb-muted", description));
        if (unavailable) card.append(node("p", "wb-muted", unavailable));
        else if (kind === "new" && sceneCount) card.append(node("p", "wb-muted", `Your timeline has ${sceneCount} scene${sceneCount === 1 ? "" : "s"}. You can recreate them from the audio. Review the settings and replacement notice in the next window, then choose Create Timeline Scenes.`));
        choices.append(card);
      }
      main.append(choices);
      if (inputAudio && !draft.song && !api.snapshot().audioPath) main.append(node("p", "wb-muted", "Choose an audio file above to enable transcription and listening-based timing."));
    } else if (page === 5) {
      const navigation = node("nav", "wb-steps");
      navigation.setAttribute("aria-label", "Scene setup steps");
      navigation.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;justify-content:flex-start;";
      sceneSteps.forEach((label, index) => {
        const tab = button(`${index + 1} ${label}`, () => { sceneStep = index; render(); });
        tab.className = "wb-step";
        if (index === sceneStep) tab.setAttribute("aria-current", "step");
        navigation.append(tab);
      });
      main.append(navigation, node("h3", "", `${sceneStep + 1}. ${sceneSteps[sceneStep]}`));
      const scenes = api.snapshot().scenes;
      const openStoryboard = (section, label) => {
        const open = button(label, () => run(async () => {
          await persist();
          api.openStoryboard(section, () => {
            const current = api.snapshot();
            draft.direction = current.direction || "";
            draft.subjects = (current.draft?.subjects || (current.draft?.singer ? [current.draft.singer] : [])).map(image => ({ ...image }));
            draft.singer = draft.subjects[0] || null;
            draft.locations = (current.draft?.locations || []).map(image => ({ ...image }));
            render();
          });
        }));
        open.disabled = section === "scenes" && !scenes.length;
        main.append(open);
      };
      if (sceneStep === 0) {
        main.append(node("p", "wb-muted", "Choose the visual and performance defaults before planning your story."));
        openStoryboard("defaults", "Scene Defaults");
      } else if (sceneStep === 1) {
        main.append(field("Story direction", input(draft.direction, value => { draft.direction = value; }, true), "Describe the overall idea, mood and direction for your video."));
      } else if (sceneStep === 2) {
        main.append(node("p", "wb-muted", "Develop the story arc and scene beats with your LLM Runner."));
        openStoryboard("story", "Story Layer");
      } else if (sceneStep === 3) {
        main.append(node("p", "wb-muted", "Optional advanced review: correct scene lines and assign performers or speakers. Skip this if your transcription is already ready."));
        const align = button("Align lyrics / dialogue", () => run(async () => { await persist(); api.openLyrics(); }));
        align.disabled = !scenes.length; main.append(align);
      } else if (sceneStep === 4) {
        main.append(node("p", "wb-muted", "Choose which subjects and locations appear in each scene."));
        const mappings = button("Edit Mappings", () => editReferences("mapping"));
        mappings.disabled = !scenes.length; main.append(mappings);
      } else {
        main.append(node("p", "wb-muted", "Review scene cards and create image or video prompts in Storyboard Scenes."));
        openStoryboard("scenes", "Storyboard Scenes");
        if (needs.images && draft.imageSource === "upload") {
          main.append(node("p", "wb-muted", "Open a scene card in Storyboard Scenes to upload its image, or import a folder of numbered images into the timeline."));
          const folder = node("input"); folder.type = "file"; folder.accept = "image/png,image/jpeg,image/webp";
          folder.multiple = true; folder.hidden = true; folder.setAttribute("webkitdirectory", ""); folder.setAttribute("directory", "");
          folder.onchange = async () => {
            const files = Array.from(folder.files || []); if (!files.length) return;
            await run(async () => { await persist(); await api.importImages(files); });
            render();
          };
          const importFolder = button("Fill Timeline Images From Folder", () => { folder.value = ""; folder.click(); });
          importFolder.disabled = !scenes.length;
          main.append(folder, importFolder);
        } else if (needs.images && scenes.length) main.append(button("Generate missing scene images", () => run(async () => { await persist(); await api.generateImages(); })));
      }
      if (sceneStep >= 3 && !scenes.length) main.append(node("p", "wb-muted", "Use Sound & timing to create scenes, then return here."));
    } else {
      const snapshot = api.snapshot();
      main.append(node("h3", "", `${draft.engine === "ltx" ? "LTX" : "MiniMax H3"} · ${modeLabel()}`), node("p", "", `${snapshot.scenes.length} scenes · ${snapshot.audioModes.find(item => item.value === draft.audioMode)?.label || draft.audioMode}`));
      main.append(node("p", "wb-muted", "Render your prepared scenes and stitch the final video. Choose Resume missing videos or Redo videos. Finish missing prompts and required images in Storyboard Scenes first."));
      const build = button("Render All", () => run(async () => { await persist(); await api.render(); }), true);
      build.disabled = !snapshot.scenes.length; main.append(build);
    }
    body.append(main);
    footer.append(node("span", "wb-muted wb-spacer", `Step ${page + 1} of ${WIZARD_BETA_STEPS.length} · ${modeLabel()}`));
    if (page) footer.append(button("Back", () => { if (page === 5 && sceneStep > 0) sceneStep--; else page--; render(); }));
    footer.append(button("Save Project", save));
    if (page < 6) footer.append(button(page === 5 && sceneStep === 3 ? "Skip / Next →" : "Next →", () => { if (page === 5 && sceneStep < sceneSteps.length - 1) sceneStep++; else page++; render(); }, true));
  }
  dialog.addEventListener("keydown", event => {
    if (event.key === "Escape") { event.stopPropagation(); close.onclick(); }
    if (event.key === "Tab") {
      const controls = [...dialog.querySelectorAll("button,input,textarea,select,summary")].filter(control => !control.disabled && control.getClientRects().length);
      if (event.shiftKey && document.activeElement === controls[0]) { event.preventDefault(); controls.at(-1)?.focus(); }
      else if (!event.shiftKey && document.activeElement === controls.at(-1)) { event.preventDefault(); controls[0]?.focus(); }
    }
  });
  render(); close.focus();
}
