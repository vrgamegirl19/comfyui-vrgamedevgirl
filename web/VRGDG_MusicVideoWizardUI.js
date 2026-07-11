const WIZARD_STYLE_ID = "vrgdg-music-video-wizard-style";
const WIZARD_DRAFT_PREFIX = "vrgdg_music_video_wizard_draft:";

function injectWizardStyles() {
  if (document.getElementById(WIZARD_STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = WIZARD_STYLE_ID;
  style.textContent = `
    .vrgdg-wizard-backdrop {
      position: fixed;
      inset: 0;
      z-index: 100011;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, .68);
    }
    .vrgdg-wizard {
      width: min(1920px, calc(100vw - 36px));
      height: calc(100vh - 38px);
      max-height: calc(100vh - 38px);
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      overflow: hidden;
      border: 1px solid #155e75;
      border-radius: 10px;
      background: #0f172a;
      color: #f8fafc;
      box-shadow: 0 24px 90px rgba(0, 0, 0, .62);
      font-family: inherit;
    }
    .vrgdg-wizard-rail {
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 16px;
      border-right: 1px solid #1e3a5f;
      background: #081524;
      min-height: 0;
    }
    .vrgdg-wizard-brand {
      padding-bottom: 10px;
      border-bottom: 1px solid #1e3a5f;
    }
    .vrgdg-wizard-brand-title {
      font-size: 20px;
      font-weight: 900;
      color: #cffafe;
    }
    .vrgdg-wizard-brand-subtitle {
      margin-top: 4px;
      font-size: 12px;
      line-height: 1.4;
      color: #94a3b8;
    }
    .vrgdg-wizard-step-button {
      width: 100%;
      display: grid;
      grid-template-columns: 30px minmax(0, 1fr);
      gap: 9px;
      align-items: center;
      padding: 10px;
      border: 1px solid #1e3a5f;
      border-radius: 8px;
      background: #101b2e;
      color: #dbeafe;
      text-align: left;
      cursor: pointer;
    }
    .vrgdg-wizard-step-button:hover {
      border-color: #0891b2;
      background: #102338;
    }
    .vrgdg-wizard-step-button.is-active {
      border-color: #22d3ee;
      background: #164e63;
      color: #ecfeff;
    }
    .vrgdg-wizard-step-button.is-done .vrgdg-wizard-step-number {
      background: #0e7490;
      color: #ecfeff;
    }
    .vrgdg-wizard-step-number {
      width: 28px;
      height: 28px;
      display: grid;
      place-items: center;
      border-radius: 999px;
      background: #1f2937;
      color: #a5f3fc;
      font-size: 12px;
      font-weight: 900;
    }
    .vrgdg-wizard-step-title {
      font-size: 13px;
      font-weight: 900;
    }
    .vrgdg-wizard-step-caption {
      margin-top: 2px;
      font-size: 11px;
      line-height: 1.3;
      color: #94a3b8;
    }
    .vrgdg-wizard-main {
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      background: #111827;
    }
    .vrgdg-wizard-header {
      flex: 0 0 auto;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
      padding: 18px 20px;
      border-bottom: 1px solid #1e3a5f;
      background: #0b2236;
    }
    .vrgdg-wizard-title {
      font-size: 20px;
      font-weight: 900;
      color: #cffafe;
    }
    .vrgdg-wizard-subtitle {
      margin-top: 4px;
      max-width: 760px;
      color: #cbd5e1;
      font-size: 13px;
      line-height: 1.45;
    }
    .vrgdg-wizard-header-actions {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 10px;
      flex: 0 0 auto;
    }
    .vrgdg-wizard-content {
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
      padding: 18px 20px;
    }
    .vrgdg-wizard-footer {
      flex: 0 0 auto;
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      align-items: center;
      padding: 14px 20px;
      border-top: 1px solid #1e3a5f;
      background: #0f172a;
    }
    .vrgdg-wizard-progress {
      height: 8px;
      border-radius: 999px;
      background: #1e293b;
      overflow: hidden;
    }
    .vrgdg-wizard-progress > div {
      height: 100%;
      width: 0;
      border-radius: inherit;
      background: #22d3ee;
      transition: width .18s ease;
    }
    .vrgdg-wizard-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .vrgdg-wizard-card {
      border: 1px solid #334155;
      border-radius: 8px;
      background: #0f172a;
      padding: 14px;
      min-width: 0;
    }
    .vrgdg-wizard-card-title {
      font-size: 14px;
      font-weight: 900;
      color: #cffafe;
      margin-bottom: 6px;
    }
    .vrgdg-wizard-copy {
      color: #cbd5e1;
      font-size: 12px;
      line-height: 1.5;
    }
    .vrgdg-wizard-check {
      display: flex;
      align-items: center;
      gap: 8px;
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 8px;
      background: rgba(15, 23, 42, 0.72);
      color: #cbd5e1;
      padding: 10px 12px;
      font-size: 12px;
      font-weight: 800;
    }
    .vrgdg-wizard-button-row {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) auto auto;
      gap: 8px;
      align-items: center;
      margin: 8px 0 8px;
    }
    .vrgdg-wizard-button-row select {
      min-width: 0;
      width: 100%;
    }
    .vrgdg-wizard-settings-layout {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 12px;
    }
    .vrgdg-wizard-settings-card {
      border: 1px solid #334155;
      border-radius: 8px;
      background: #0d1b2d;
      padding: 16px;
      min-width: 0;
    }
    .vrgdg-wizard-settings-card.span-4 { grid-column: span 4; }
    .vrgdg-wizard-settings-card.span-3 { grid-column: span 3; }
    .vrgdg-wizard-settings-card.span-5 { grid-column: span 5; }
    .vrgdg-wizard-settings-card.span-6 { grid-column: span 6; }
    .vrgdg-wizard-settings-card.span-7 { grid-column: span 7; }
    .vrgdg-wizard-settings-card.span-8 { grid-column: span 8; }
    .vrgdg-wizard-settings-card.span-12 { grid-column: 1 / -1; }
    .vrgdg-wizard-settings-title {
      color: #e0f2fe;
      font-size: 16px;
      font-weight: 900;
      margin-bottom: 5px;
    }
    .vrgdg-wizard-settings-subtitle {
      color: #cbd5e1;
      font-size: 12px;
      line-height: 1.4;
      margin-bottom: 14px;
    }
    .vrgdg-wizard-settings-fields {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px 18px;
    }
    .vrgdg-wizard-settings-fields.two {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .vrgdg-wizard-settings-fields.four {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
    .vrgdg-wizard-settings-fields.compact {
      grid-template-columns: 1fr;
    }
    .vrgdg-wizard-settings-field {
      display: flex;
      flex-direction: column;
      gap: 7px;
      min-width: 0;
    }
    .vrgdg-wizard-settings-label {
      color: #e5f6ff;
      font-size: 12px;
      font-weight: 900;
    }
    .vrgdg-wizard-settings-help {
      color: #94a3b8;
      font-size: 12px;
      line-height: 1.35;
    }
    .vrgdg-wizard-status-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0;
    }
    .vrgdg-wizard-status-pill {
      border: 1px solid #334155;
      border-radius: 999px;
      background: #0f172a;
      color: #dbeafe;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 900;
    }
    .vrgdg-wizard-settings-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .vrgdg-wizard-tip {
      grid-column: 1 / -1;
      border: 1px solid #334155;
      border-radius: 8px;
      background: #0b1728;
      color: #cbd5e1;
      padding: 12px 14px;
      font-size: 12px;
      line-height: 1.4;
    }
    .vrgdg-wizard-note {
      border: 1px solid #155e75;
      border-radius: 8px;
      background: #0a2438;
      color: #dff6ff;
      padding: 10px 12px;
      font-size: 12px;
      line-height: 1.45;
    }
    .vrgdg-wizard-info-note {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      border: 1px solid #0891b2;
      border-radius: 10px;
      background: #082f49;
      color: #e0f2fe;
      padding: 13px 14px;
      font-size: 14px;
      line-height: 1.45;
    }
    .vrgdg-wizard-info-icon,
    .vrgdg-wizard-section-number {
      flex: 0 0 auto;
      display: grid;
      place-items: center;
      border-radius: 999px;
      background: #06b6d4;
      color: #001018;
      font-weight: 900;
    }
    .vrgdg-wizard-info-icon {
      width: 24px;
      height: 24px;
      font-size: 15px;
    }
    .vrgdg-wizard-section {
      border: 1px solid #334155;
      border-radius: 10px;
      background: #0f172a;
      padding: 18px 20px;
      min-width: 0;
    }
    .vrgdg-wizard-section + .vrgdg-wizard-section {
      margin-top: 12px;
    }
    .vrgdg-wizard-section-heading {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }
    .vrgdg-wizard-section-number {
      width: 28px;
      height: 28px;
      font-size: 13px;
    }
    .vrgdg-wizard-section-title {
      color: #f8fafc;
      font-size: 16px;
      font-weight: 900;
    }
    .vrgdg-wizard-section-copy {
      color: #cbd5e1;
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 12px;
    }
    .vrgdg-wizard-lyrics-editor {
      display: grid;
      grid-template-columns: 46px minmax(0, 1fr);
      height: 280px;
      max-height: 34vh;
      min-height: 180px;
      border: 1px solid #334155;
      border-radius: 8px;
      overflow: hidden;
      background: #020617;
    }
    .vrgdg-wizard-line-gutter {
      padding: 10px 9px;
      border-right: 1px solid #1e293b;
      background: #07111f;
      color: #64748b;
      text-align: right;
      font-family: monospace;
      font-size: 13px;
      line-height: 1.55;
      user-select: none;
      overflow: hidden;
      white-space: pre;
    }
    .vrgdg-wizard-lyrics-editor .vrgdg-wizard-textarea {
      height: 100%;
      min-height: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
      line-height: 1.55;
      font-size: 13px;
      outline: none;
      resize: none;
      overflow: auto;
    }
    .vrgdg-wizard-scene-settings-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .vrgdg-wizard-setting-tile {
      display: flex;
      flex-direction: column;
      gap: 8px;
      min-width: 0;
      border: 1px solid #334155;
      border-radius: 9px;
      background: #101b2e;
      padding: 13px;
    }
    .vrgdg-wizard-setting-title {
      color: #e0f2fe;
      font-size: 13px;
      font-weight: 900;
    }
    .vrgdg-wizard-setting-help {
      color: #94a3b8;
      font-size: 12px;
      line-height: 1.35;
    }
    .vrgdg-wizard-setting-help strong {
      color: #cffafe;
    }
    .vrgdg-wizard-setting-help-list {
      margin: 0;
      padding-left: 16px;
    }
    .vrgdg-wizard-setting-help-list li {
      margin: 4px 0;
    }
    .vrgdg-wizard-lyrics-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-top: 12px;
    }
    .vrgdg-wizard-lyrics-page {
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 0;
    }
    .vrgdg-wizard-button.is-working::before {
      content: "";
      width: 15px;
      height: 15px;
      display: inline-block;
      margin-right: 8px;
      vertical-align: -2px;
      border: 2px solid rgba(34, 211, 238, .35);
      border-top-color: #22d3ee;
      border-radius: 999px;
      animation: vrgdgWizardSpin .8s linear infinite;
    }
    @keyframes vrgdgWizardSpin {
      to { transform: rotate(360deg); }
    }
    .vrgdg-wizard-field {
      display: flex;
      flex-direction: column;
      gap: 5px;
      min-width: 0;
    }
    .vrgdg-wizard-label {
      color: #dbeafe;
      font-size: 12px;
      font-weight: 900;
    }
    .vrgdg-wizard-input,
    .vrgdg-wizard-textarea,
    .vrgdg-wizard-select {
      box-sizing: border-box;
      width: 100%;
      border: 1px solid #334155;
      border-radius: 7px;
      background: #020617;
      color: #f8fafc;
      padding: 9px 10px;
      font: inherit;
      font-size: 12px;
    }
    .vrgdg-wizard-textarea {
      min-height: 260px;
      resize: vertical;
      font-family: monospace;
      line-height: 1.45;
    }
    .vrgdg-wizard-button {
      min-height: 38px;
      border: 1px solid #3f3f46;
      border-radius: 7px;
      background: #27272a;
      color: #f4f4f5;
      padding: 8px 12px;
      font-weight: 900;
      cursor: pointer;
    }
    .vrgdg-wizard-button:hover {
      filter: brightness(1.08);
    }
    .vrgdg-wizard-button.primary {
      border-color: #22d3ee;
      background: #06b6d4;
      color: #001018;
    }
    .vrgdg-wizard-button.ghost {
      border-color: #334155;
      background: #111827;
      color: #dbeafe;
    }
    .vrgdg-wizard-button[disabled],
    .vrgdg-wizard-disabled {
      opacity: .55;
      cursor: not-allowed;
    }
    .vrgdg-wizard-drop {
      display: grid;
      place-items: center;
      min-height: 190px;
      border: 1px dashed #22d3ee;
      border-radius: 8px;
      background: #07111f;
      color: #cbd5e1;
      text-align: center;
      padding: 18px;
    }
    .vrgdg-wizard-pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .vrgdg-wizard-pill {
      border: 1px solid #334155;
      border-radius: 999px;
      background: #111827;
      color: #dbeafe;
      padding: 6px 9px;
      font-size: 11px;
      font-weight: 900;
    }
    .vrgdg-wizard-mode-card {
      border: 1px solid #334155;
      border-radius: 8px;
      background: #0f172a;
      padding: 14px;
      min-height: 116px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .vrgdg-wizard-mode-card.is-active {
      border-color: #22d3ee;
      background: #12314a;
    }
    .vrgdg-wizard-mode-card.is-disabled {
      opacity: .56;
    }
    @media (max-width: 840px) {
      .vrgdg-wizard {
        grid-template-columns: 1fr;
      }
      .vrgdg-wizard-rail {
        display: none;
      }
      .vrgdg-wizard-grid {
        grid-template-columns: 1fr;
      }
      .vrgdg-wizard-settings-card,
      .vrgdg-wizard-settings-card.span-4,
      .vrgdg-wizard-settings-card.span-3,
      .vrgdg-wizard-settings-card.span-5,
      .vrgdg-wizard-settings-card.span-6,
      .vrgdg-wizard-settings-card.span-7,
      .vrgdg-wizard-settings-card.span-8,
      .vrgdg-wizard-settings-card.span-12 {
        grid-column: 1 / -1;
      }
      .vrgdg-wizard-settings-fields,
      .vrgdg-wizard-settings-fields.two,
      .vrgdg-wizard-settings-fields.four {
        grid-template-columns: 1fr;
      }
      .vrgdg-wizard-scene-settings-grid {
        grid-template-columns: 1fr;
      }
      .vrgdg-wizard-footer {
        grid-template-columns: 1fr;
      }
    }
  `;
  document.head.append(style);
}

function el(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function button(label, kind = "") {
  const node = el("button", `vrgdg-wizard-button${kind ? ` ${kind}` : ""}`, label);
  node.type = "button";
  return node;
}

function field(label, control, helper = "") {
  const wrapper = el("label", "vrgdg-wizard-field");
  wrapper.append(el("div", "vrgdg-wizard-label", label), control);
  if (helper) wrapper.append(el("div", "vrgdg-wizard-copy", helper));
  return wrapper;
}

function input(value = "", type = "text") {
  const node = el("input", "vrgdg-wizard-input");
  node.type = type;
  node.value = value;
  return node;
}

function lyricStoryStrengthValue(value, fallback = 7) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(10, number)) : fallback;
}

function comboInput(value = "", options = [], listId = "") {
  const node = input(value);
  if (listId) node.setAttribute("list", listId);
  const list = document.createElement("datalist");
  list.id = listId;
  for (const item of options || []) {
    const option = document.createElement("option");
    option.value = item;
    list.append(option);
  }
  return { input: node, list };
}

function textarea(value = "", placeholder = "") {
  const node = el("textarea", "vrgdg-wizard-textarea");
  node.value = value;
  node.placeholder = placeholder;
  return node;
}

function select(options, value) {
  const node = el("select", "vrgdg-wizard-select");
  for (const item of options) {
    const option = document.createElement("option");
    option.value = typeof item === "object" ? String(item.value ?? "") : String(item);
    option.textContent = item.label || item;
    node.append(option);
  }
  node.value = value;
  return node;
}

function card(title, copy = "") {
  const node = el("div", "vrgdg-wizard-card");
  node.append(el("div", "vrgdg-wizard-card-title", title));
  if (copy) node.append(el("div", "vrgdg-wizard-copy", copy));
  return node;
}

function sampleLyrics() {
  return `[intro]
[instrumental]

I don't explain it, I let it speak
I don't chase it, it comes to me

[chorus]
Every doubt gets less important
When I do what I came to do`;
}

export function openMusicVideoWizard(api = {}) {
  injectWizardStyles();

  const steps = [
    { id: "settings", title: "Settings", caption: "Models and render setup" },
    { id: "audio", title: "Audio", caption: "Load the song file" },
    { id: "lyrics", title: "Lyrics + Scenes", caption: "Create timeline scenes" },
    { id: "mode", title: "Mode", caption: "Choose i2v or ref video" },
    { id: "references", title: "References", caption: "Subjects and locations" },
    { id: "story", title: "Story Direction", caption: "Arc and scene beats" },
    { id: "finish", title: "Gemma + Render", caption: "Prompt and build" },
  ];
  let activeIndex = 0;
  const done = new Set();
  const wizardState = {
    lyrics: "",
    language: "english",
    segmentMode: "reference_lines",
    minSceneSeconds: "1.0",
    maxSceneSeconds: "8.0",
    cameraFlow: "balanced",
    performanceStyle: "",
    facialPerformance: "",
    facialPerformanceCustom: "",
    storyCharacterMotion: 7,
    storyLayer: {
      enabled: true,
      user_story_arc: "",
      song_story_brief: "",
      lyric_story_strength: 7,
    },
  };

  const backdrop = el("div", "vrgdg-wizard-backdrop");
  const shell = el("div", "vrgdg-wizard");
  const rail = el("div", "vrgdg-wizard-rail");
  const main = el("div", "vrgdg-wizard-main");
  const header = el("div", "vrgdg-wizard-header");
  const content = el("div", "vrgdg-wizard-content");
  const footer = el("div", "vrgdg-wizard-footer");
  const progress = el("div", "vrgdg-wizard-progress");
  const progressBar = document.createElement("div");
  const prev = button("Back", "ghost");
  const next = button("Next", "primary");
  const quickSave = button("Quick Save");
  const close = button("Close");

  const brand = el("div", "vrgdg-wizard-brand");
  brand.append(
    el("div", "vrgdg-wizard-brand-title", "Video Wizard"),
    el("div", "vrgdg-wizard-brand-subtitle", "A guided path for Image to Video and Reference to Video projects. It reuses the builder tools you already have."),
  );
  rail.append(brand);

  const stepButtons = new Map();
  for (let index = 0; index < steps.length; index += 1) {
    const step = steps[index];
    const item = el("button", "vrgdg-wizard-step-button");
    item.type = "button";
    item.innerHTML = `
      <div class="vrgdg-wizard-step-number">${index + 1}</div>
      <div>
        <div class="vrgdg-wizard-step-title"></div>
        <div class="vrgdg-wizard-step-caption"></div>
      </div>
    `;
    item.querySelector(".vrgdg-wizard-step-title").textContent = step.title;
    item.querySelector(".vrgdg-wizard-step-caption").textContent = step.caption;
    item.onclick = () => {
      activeIndex = index;
      render();
    };
    stepButtons.set(step.id, item);
    rail.append(item);
  }

  progress.append(progressBar);
  footer.append(progress, prev, next);
  shell.append(rail, main);
  backdrop.append(shell);
  document.body.append(backdrop);

  const closeWizard = () => {
    persistWizardDraft();
    persistWizardDraftFile().catch(() => null);
    backdrop.remove();
  };
  close.onclick = closeWizard;
  quickSave.onclick = quickSaveWizard;
  backdrop.addEventListener("pointerdown", (event) => {
    if (event.target === backdrop) closeWizard();
  });

  function snapshot() {
    try {
      return api.snapshot?.() || {};
    } catch {
      return {};
    }
  }

  function openNestedTool(openTool, label = "wizard nested tool") {
    persistWizardDraft();
    persistWizardDraftFile().catch(() => null);
    backdrop.style.zIndex = "100005";
    try {
      openTool?.();
      saveWizardProgress(label);
    } catch {
      backdrop.style.zIndex = "";
    }
  }

  function closeAndRun(openTool, label = "wizard run action") {
    persistWizardDraft();
    persistWizardDraftFile().catch(() => null);
    backdrop.remove();
    try {
      openTool?.();
      saveWizardProgress(label);
    } catch {
      // The launched builder action owns its own error surface.
    }
  }

  function draftKey(data = snapshot()) {
    const project = String(data.projectFolder || "").trim();
    return `${WIZARD_DRAFT_PREFIX}${project || "unsaved-project"}`;
  }

  function persistWizardDraft() {
    try {
      localStorage.setItem(draftKey(), JSON.stringify(wizardDraftPayload()));
    } catch {
      // Draft persistence is a convenience only.
    }
  }

  function wizardDraftPayload() {
    return {
      wizardState,
      lyrics: wizardState.lyrics || "",
      done: Array.from(done),
      activeIndex,
      updatedAt: new Date().toISOString(),
    };
  }

  async function persistWizardDraftFile() {
    const data = snapshot();
    if (!String(data.projectFolder || "").trim()) return null;
    return api.saveWizardDraft?.(wizardDraftPayload());
  }

  let draftTimer = null;
  function queueWizardDraftSave() {
    if (draftTimer) clearTimeout(draftTimer);
    draftTimer = setTimeout(() => {
      draftTimer = null;
      persistWizardDraft();
      persistWizardDraftFile().catch(() => null);
    }, 250);
  }

  function applyWizardDraft(draft) {
    if (!draft || typeof draft !== "object") return false;
    let changed = false;
    const sourceState = draft.wizardState && typeof draft.wizardState === "object" ? draft.wizardState : draft;
      Object.assign(wizardState, {
      lyrics: String(sourceState.lyrics || draft.lyrics || ""),
      language: String(sourceState.language || "english"),
      segmentMode: String(sourceState.segmentMode || "reference_lines"),
      minSceneSeconds: String(sourceState.minSceneSeconds || "1.0"),
      maxSceneSeconds: String(sourceState.maxSceneSeconds || "8.0"),
      cameraFlow: String(sourceState.cameraFlow || "balanced"),
      performanceStyle: String(sourceState.performanceStyle || ""),
      facialPerformance: String(sourceState.facialPerformance || ""),
      facialPerformanceCustom: String(sourceState.facialPerformanceCustom || ""),
      storyCharacterMotion: Math.max(0, Math.min(10, Number(sourceState.storyCharacterMotion ?? sourceState.story_character_motion ?? wizardState.storyCharacterMotion ?? 7))),
      storyLayer: sourceState.storyLayer && typeof sourceState.storyLayer === "object"
        ? {
            enabled: sourceState.storyLayer.enabled !== false,
            user_story_arc: String(sourceState.storyLayer.user_story_arc || ""),
            song_story_brief: String(sourceState.storyLayer.song_story_brief || ""),
            lyric_story_strength: lyricStoryStrengthValue(sourceState.storyLayer.lyric_story_strength ?? sourceState.storyLayer.lyricStoryStrength),
          }
        : wizardState.storyLayer,
      });
    changed = true;
    if (Array.isArray(draft.done)) {
      draft.done.forEach((id) => {
        if (steps.some((step) => step.id === id)) done.add(id);
      });
      changed = true;
    }
    return changed;
  }

  function restoreLocalWizardDraft() {
    try {
      const raw = localStorage.getItem(draftKey());
      if (!raw) return;
      applyWizardDraft(JSON.parse(raw));
    } catch {
      // Ignore corrupt drafts.
    }
  }

  async function restoreProjectWizardDraft() {
    try {
      const result = await api.loadWizardDraft?.();
      if (applyWizardDraft(result?.draft)) render();
    } catch {
      // The wizard folder is optional until a project exists.
    }
  }

  async function saveWizardProgress(label = "wizard progress") {
    persistWizardDraft();
    await persistWizardDraftFile().catch(() => null);
    await api.updateStoryLayer?.(wizardState.storyLayer).catch(() => null);
    try {
      await api.saveProject?.({ quiet: true, label });
    } catch {
      // The project may not exist yet; the local wizard draft still protects text settings.
    }
  }

  async function quickSaveWizard() {
    const previousText = quickSave.textContent;
    quickSave.disabled = true;
    quickSave.textContent = "Saving...";
    try {
      await saveWizardProgress("wizard quick save");
    } finally {
      quickSave.disabled = false;
      quickSave.textContent = previousText;
    }
  }

  async function setWizardMode(mode = "i2v") {
    await api.setVideoMode?.(mode);
    done.add("mode");
    await saveWizardProgress("wizard mode");
    render();
  }

  async function setWizardImageMode(mode = "zimage") {
    await api.setImageMode?.(mode);
    done.add("settings");
    await saveWizardProgress("wizard image mode");
    render();
  }

  function statusPills(data) {
    const row = el("div", "vrgdg-wizard-pill-row");
    const pills = [
      `Project: ${data.projectFolder ? "saved" : "not set"}`,
      `Audio: ${data.audioPath ? "loaded" : "missing"}`,
      `Scenes: ${Number(data.sceneCount || 0)}`,
      `Mode: ${data.videoModeLabel || "Reference to Video"}`,
    ];
    for (const text of pills) row.append(el("div", "vrgdg-wizard-pill", text));
    return row;
  }

  function renderHeader(step) {
    header.textContent = "";
    const titleWrap = document.createElement("div");
    titleWrap.append(el("div", "vrgdg-wizard-title", step.title), el("div", "vrgdg-wizard-subtitle", step.caption));
    const actions = el("div", "vrgdg-wizard-header-actions");
    actions.append(quickSave, close);
    header.append(titleWrap, actions);
  }

  function renderSettings(data) {
    const modelOptions = data.modelOptions || {};
    const settings = data.videoSettings || {};
    const layout = el("div", "vrgdg-wizard-settings-layout");
    const settingsCard = el("div", "vrgdg-wizard-settings-card span-6");
    settingsCard.append(
      el("div", "vrgdg-wizard-settings-title", "Current Builder State"),
      el("div", "vrgdg-wizard-settings-subtitle", "The wizard uses the settings already selected in the Video Creator."),
    );
    const statusRow = el("div", "vrgdg-wizard-status-row");
    [
      `Project: ${data.projectFolder ? "saved" : "not set"}`,
      `Audio: ${data.audioPath ? "loaded" : "missing"}`,
      `Scenes: ${Number(data.sceneCount || 0)}`,
      `Mode: ${data.videoModeLabel || "Reference to Video"}`,
      `Image: ${data.imageModeLabel || data.imageMode || "ZImage"}`,
    ].forEach((text) => statusRow.append(el("div", "vrgdg-wizard-status-pill", text)));
    settingsCard.append(statusRow);

    const quickCard = el("div", "vrgdg-wizard-settings-card span-6");
    quickCard.append(
      el("div", "vrgdg-wizard-settings-title", "Quick Actions"),
      el("div", "vrgdg-wizard-settings-subtitle", "Set the wizard mode or change how text Gemma runs."),
    );
    const actions = el("div", "vrgdg-wizard-settings-actions");
    const activeVideoMode = String(data.videoMode || "i2v");
    const setI2v = button("Set Mode: Image to Video", activeVideoMode === "i2v" ? "primary" : "");
    const setRtv = button("Set Mode: Reference to Video", activeVideoMode === "rtv" ? "primary" : "");
    const openRunner = button("Open Gemma Runner");
    setI2v.onclick = () => setWizardMode("i2v");
    setRtv.onclick = () => setWizardMode("rtv");
    openRunner.onclick = () => {
      openNestedTool(() => api.openGemmaRunner?.(), "wizard gemma runner");
    };
    actions.append(setI2v, setRtv, openRunner);
    quickCard.append(actions);

    const settingField = (label, control, help) => {
      const wrapper = el("label", "vrgdg-wizard-settings-field");
      wrapper.append(el("div", "vrgdg-wizard-settings-label", label), control);
      if (help) wrapper.append(el("div", "vrgdg-wizard-settings-help", help));
      return wrapper;
    };

    const isReferenceToVideo = String(data.videoMode || "i2v") === "rtv";
    const imageModeOptions = data.imageModeOptions?.length ? data.imageModeOptions : [
      { value: "zimage", label: "ZImage" },
      { value: "flux_klein", label: "Flux Klein" },
      { value: "ernie_image", label: "Ernie Image" },
      { value: "krea2_2pass", label: "Krea 2" },
      { value: "nano_banana", label: "NanoBanana" },
    ];
    const imageMode = String(data.imageMode || "zimage");
    const imageModeCard = el("div", "vrgdg-wizard-settings-card span-4");
    imageModeCard.append(
      el("div", "vrgdg-wizard-settings-title", "Text-to-Image Mode"),
      el("div", "vrgdg-wizard-settings-subtitle", "Choose the image generator first. The wizard shows the matching model controls below."),
    );
    imageModeCard.style.display = isReferenceToVideo ? "none" : "";
    const imageModeSelect = select(imageModeOptions, imageMode);
    imageModeSelect.onchange = () => setWizardImageMode(imageModeSelect.value);
    imageModeCard.append(field("Image mode", imageModeSelect, "This is the image pass used before Image-to-Video."));

    const imageSettings = data.imageSettings?.[imageMode] || {};
    const imageModelOptions = data.imageModelOptions?.[imageMode] || {};
    const imageModelCard = el("div", "vrgdg-wizard-settings-card span-8");
    const isFlowGpt = imageMode === "flow_gpt";
    const isNanoBanana = imageMode === "nano_banana";
    imageModelCard.append(
      el("div", "vrgdg-wizard-settings-title", isFlowGpt ? "Flow/GPT Browser Settings" : isNanoBanana ? "NanoBanana Settings" : `${data.imageModeLabel || "Image"} Model Stack`),
      el("div", "vrgdg-wizard-settings-subtitle", isFlowGpt
        ? "Flow/GPT uses the Browser Image provider and login/settings from the main side panel."
        : isNanoBanana
        ? "NanoBanana uses API settings and reference images from the NanoBanana side panel."
        : "Primary models used by the selected text-to-image path."),
    );
    imageModelCard.style.display = isReferenceToVideo ? "none" : "";
    const imageGrid = el("div", "vrgdg-wizard-settings-fields");
    const imageUnet = comboInput(imageSettings.unet_name || "", imageModelOptions.unets || modelOptions.unets || [], "vrgdg-wizard-image-unet");
    const imageClip = comboInput(imageSettings.clip_name || "", imageModelOptions.clip || modelOptions.clip || [], "vrgdg-wizard-image-clip");
    const imageVae = comboInput(imageSettings.vae_name || "", imageModelOptions.vae || modelOptions.vae || [], "vrgdg-wizard-image-vae");
    const nanoApiKey = input(imageSettings.api_key || "", "password");
    nanoApiKey.placeholder = "NanoBanana API key...";
    const nanoModelValue = imageSettings.model || "gemini-3-pro-image-preview";
    const nanoModelOptions = Array.from(new Set([
      nanoModelValue,
      ...(imageModelOptions.models || ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"]),
    ].filter(Boolean)));
    const nanoModel = select(nanoModelOptions, nanoModelValue);
    const nanoUseTextOnly = document.createElement("label");
    nanoUseTextOnly.style.cssText = "display:flex;align-items:center;gap:8px;color:#dbeafe;font-size:12px;font-weight:900;";
    const nanoUseTextOnlyInput = document.createElement("input");
    nanoUseTextOnlyInput.type = "checkbox";
    nanoUseTextOnlyInput.checked = Boolean(imageSettings.use_text_only_gemma_prompt);
    nanoUseTextOnly.append(nanoUseTextOnlyInput, document.createTextNode("Use text-only Gemma prompts"));
    const nanoUseDirector = document.createElement("label");
    nanoUseDirector.style.cssText = nanoUseTextOnly.style.cssText;
    const nanoUseDirectorInput = document.createElement("input");
    nanoUseDirectorInput.type = "checkbox";
    nanoUseDirectorInput.checked = Boolean(imageSettings.use_director_notes);
    nanoUseDirector.append(nanoUseDirectorInput, document.createTextNode("Include Director Notes"));
    const flowGptProvider = select([
      { value: "flow_nano_banana", label: "Flow Nano Banana" },
      { value: "gpt_image", label: "GPT Image" },
    ], String(imageSettings.provider || "").toLowerCase() === "gpt_image" ? "gpt_image" : "flow_nano_banana");
    const flowGptAspectRatio = select(["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], imageSettings.aspect_ratio || "16:9");
    const flowGptFailureMode = select([
      { value: "last_successful_image", label: "Use last successful image" },
      { value: "try_other_provider", label: "Try other provider first" },
      { value: "stop", label: "Stop on failure" },
    ], imageSettings.failure_mode || "last_successful_image");
    const flowGptFlowTimeout = input(imageSettings.flow_timeout_seconds || 420, "number");
    flowGptFlowTimeout.min = "60";
    flowGptFlowTimeout.max = "1800";
    flowGptFlowTimeout.step = "10";
    const flowGptGptTimeout = input(imageSettings.gpt_timeout_seconds || imageSettings.timeout_seconds || 600, "number");
    flowGptGptTimeout.min = "60";
    flowGptGptTimeout.max = "2400";
    flowGptGptTimeout.step = "10";
    const flowGptMaxRetries = input(imageSettings.max_retries || 10, "number");
    flowGptMaxRetries.min = "1";
    flowGptMaxRetries.max = "20";
    flowGptMaxRetries.step = "1";
    const flowGptManualMode = document.createElement("label");
    flowGptManualMode.style.cssText = nanoUseTextOnly.style.cssText;
    const flowGptManualModeInput = document.createElement("input");
    flowGptManualModeInput.type = "checkbox";
    flowGptManualModeInput.checked = Boolean(imageSettings.manual_mode);
    flowGptManualMode.append(flowGptManualModeInput, document.createTextNode("Manual browser import"));
    const flowGptManualAutoAdvance = document.createElement("label");
    flowGptManualAutoAdvance.style.cssText = nanoUseTextOnly.style.cssText;
    const flowGptManualAutoAdvanceInput = document.createElement("input");
    flowGptManualAutoAdvanceInput.type = "checkbox";
    flowGptManualAutoAdvanceInput.checked = Boolean(imageSettings.manual_auto_advance);
    flowGptManualAutoAdvance.append(flowGptManualAutoAdvanceInput, document.createTextNode("Auto-advance after manual import"));
    const flowGptSettingsFromControls = () => ({
      provider: flowGptProvider.value,
      aspect_ratio: flowGptAspectRatio.value,
      manual_mode: Boolean(flowGptManualModeInput.checked),
      manual_auto_advance: Boolean(flowGptManualAutoAdvanceInput.checked),
      failure_mode: flowGptFailureMode.value,
      flow_timeout_seconds: Number(flowGptFlowTimeout.value || 420),
      gpt_timeout_seconds: Number(flowGptGptTimeout.value || 600),
      max_retries: Number(flowGptMaxRetries.value || 10),
    });
    const applyFlowGptSettingsQuietly = () => api.updateFlowGptSettings?.(flowGptSettingsFromControls());
    const flowGptStatus = el("div", "vrgdg-wizard-copy", "");
    flowGptStatus.style.cssText = "grid-column:1 / -1;white-space:pre-wrap;";
    const flowGptSetupButton = button("Install Browser Automation", "primary");
    const flowGptCheckButton = button("Test Browser Setup");
    const flowGptFlowLoginButton = button("Open Flow Login");
    const flowGptGptLoginButton = button("Open GPT Login");
    const flowGptActionRow = el("div", "vrgdg-wizard-settings-actions");
    flowGptActionRow.style.gridColumn = "1 / -1";
    flowGptActionRow.append(flowGptSetupButton, flowGptCheckButton, flowGptFlowLoginButton, flowGptGptLoginButton);
    const flowGptSection = (title, copy = "") => {
      const section = el("div", "");
      section.style.cssText = "grid-column:1 / -1;display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;border-top:1px solid #1e3a5f;padding-top:12px;";
      section.append(el("div", "vrgdg-wizard-settings-title", title));
      section.firstChild.style.gridColumn = "1 / -1";
      if (copy) {
        const note = el("div", "vrgdg-wizard-settings-help", copy);
        note.style.gridColumn = "1 / -1";
        section.append(note);
      }
      return section;
    };
    const flowGptBrowserSection = flowGptSection("Browser Actions", "Use these before a batch run to install/check browser automation or sign into the selected providers.");
    flowGptBrowserSection.append(flowGptActionRow, flowGptStatus);
    const flowGptAutoSection = flowGptSection("Automated Run Settings", "Used when Manual browser import is off.");
    flowGptAutoSection.append(
      settingField("Failure mode", flowGptFailureMode, "What Flow/GPT should do if a browser image pass fails."),
      settingField("Flow timeout", flowGptFlowTimeout, "Seconds to wait for Flow Nano Banana browser images."),
      settingField("GPT timeout", flowGptGptTimeout, "Seconds to wait for GPT Image browser images."),
      settingField("Max retries", flowGptMaxRetries, "Retry attempts before the selected failure mode is used."),
    );
    const flowGptManualSection = flowGptSection("Manual Import Settings", "Shown only when Manual browser import is checked.");
    const flowGptManualOpenButton = button("Open Manual Browser", "primary");
    const flowGptManualExportButton = button("Export Scene Refs");
    const flowGptManualArmButton = button("Arm Download Import", "primary");
    const flowGptManualLatestButton = button("Import Latest Download");
    const flowGptManualActionRow = el("div", "vrgdg-wizard-settings-actions");
    flowGptManualActionRow.style.gridColumn = "1 / -1";
    flowGptManualActionRow.append(flowGptManualOpenButton, flowGptManualExportButton, flowGptManualArmButton, flowGptManualLatestButton);
    const flowGptManualStatus = el("div", "vrgdg-wizard-copy", "");
    flowGptManualStatus.style.cssText = "grid-column:1 / -1;white-space:pre-wrap;";
    flowGptManualSection.append(flowGptManualAutoAdvance, flowGptManualActionRow, flowGptManualStatus);
    const syncFlowGptModeSections = () => {
      const manual = Boolean(flowGptManualModeInput.checked);
      flowGptManualSection.style.display = manual ? "grid" : "none";
      flowGptAutoSection.style.display = manual ? "none" : "grid";
    };
    flowGptManualModeInput.onchange = syncFlowGptModeSections;
    const runFlowGptAction = async (control, label, action) => {
      const previous = control.textContent;
      control.disabled = true;
      control.textContent = `${label}...`;
      flowGptStatus.textContent = `${label}...`;
      try {
        await applyFlowGptSettingsQuietly();
        const result = await action();
        flowGptStatus.textContent = typeof result === "string" ? result : String(result?.status || result?.message || `${label} finished.`);
      } catch (error) {
        flowGptStatus.textContent = `${label} failed:\n${String(error?.message || error)}`;
      } finally {
        control.disabled = false;
        control.textContent = previous;
      }
    };
    flowGptSetupButton.onclick = () => runFlowGptAction(flowGptSetupButton, "Installing browser automation", () => api.setupFlowGptBrowser?.());
    flowGptCheckButton.onclick = () => runFlowGptAction(flowGptCheckButton, "Testing browser setup", () => api.checkFlowGptBrowser?.());
    flowGptFlowLoginButton.onclick = () => runFlowGptAction(flowGptFlowLoginButton, "Opening Flow login", () => api.openFlowGptLogin?.("flow_nano_banana"));
    flowGptGptLoginButton.onclick = () => runFlowGptAction(flowGptGptLoginButton, "Opening GPT login", () => api.openFlowGptLogin?.("gpt_image"));
    const runManualFlowGptAction = async (control, label, action) => {
      const previous = control.textContent;
      control.disabled = true;
      control.textContent = `${label}...`;
      flowGptManualStatus.textContent = `${label}...`;
      try {
        await applyFlowGptSettingsQuietly();
        const result = await action();
        flowGptManualStatus.textContent = typeof result === "string" ? result : String(result?.status || result?.message || `${label} finished.`);
      } catch (error) {
        flowGptManualStatus.textContent = `${label} failed:\n${String(error?.message || error)}`;
      } finally {
        control.disabled = false;
        control.textContent = previous;
      }
    };
    flowGptManualOpenButton.onclick = () => runManualFlowGptAction(flowGptManualOpenButton, "Opening manual browser", () => api.openFlowGptManualBrowser?.());
    flowGptManualExportButton.onclick = () => runManualFlowGptAction(flowGptManualExportButton, "Exporting scene refs", () => api.exportFlowGptManualRefs?.());
    flowGptManualArmButton.onclick = () => runManualFlowGptAction(flowGptManualArmButton, "Arming download import", () => api.armFlowGptManualDownloadImport?.());
    flowGptManualLatestButton.onclick = () => runManualFlowGptAction(flowGptManualLatestButton, "Importing latest download", () => api.importLatestFlowGptManualDownload?.());
    if (isFlowGpt) {
      imageGrid.append(
        settingField("Provider", flowGptProvider, "Browser image provider used for Flow/GPT scenes."),
        settingField("Aspect ratio", flowGptAspectRatio, "GPT Image appends this ratio to prompts. Flow uses the ratio you set in Flow."),
        flowGptManualMode,
        flowGptBrowserSection,
        flowGptAutoSection,
        flowGptManualSection,
      );
      syncFlowGptModeSections();
    } else if (isNanoBanana) {
      imageGrid.append(
        settingField("API key", nanoApiKey, "Stored in the same NanoBanana side-panel setting. The key is hidden while typing."),
        settingField("Nano model", nanoModel, "NanoBanana image model name."),
        nanoUseTextOnly,
        nanoUseDirector,
        settingField("Reference images", el("div", "vrgdg-wizard-copy", data.subjectCount || data.locationCount ? `${Number(data.subjectCount || 0)} subject / ${Number(data.locationCount || 0)} location mapped` : "None mapped yet"), "Use Reference Builder to map subject/location references for NanoBanana."),
      );
    } else {
      imageGrid.append(
        settingField("Image UNet model", imageUnet.input, "Main model for this text-to-image mode."),
        settingField("Image CLIP", imageClip.input, "Text encoder for this image mode."),
        settingField("Image VAE", imageVae.input, "Image decoder/encoder for this image mode."),
      );
    }
    if (isFlowGpt || isNanoBanana) {
      imageModelCard.append(imageGrid);
    } else {
      imageModelCard.append(imageUnet.list, imageClip.list, imageVae.list, imageGrid);
    }

    const modelCard = el("div", "vrgdg-wizard-settings-card span-12");
    modelCard.append(
      el("div", "vrgdg-wizard-settings-title", "Video Model Stack"),
      el("div", "vrgdg-wizard-settings-subtitle", "Models used for Image to Video or Reference to Video generation and decoding."),
    );
    const modelGrid = el("div", "vrgdg-wizard-settings-fields");
    const useGgufModel = document.createElement("label");
    useGgufModel.style.cssText = "display:flex;align-items:center;gap:8px;color:#dbeafe;font-size:12px;font-weight:900;";
    const useGgufModelInput = document.createElement("input");
    useGgufModelInput.type = "checkbox";
    useGgufModelInput.checked = settings.use_gguf_model !== false;
    useGgufModel.append(useGgufModelInput, document.createTextNode("Use GGUF model?"));
    const unet = comboInput(settings.unet_name || "", modelOptions.unets || [], "vrgdg-wizard-unets");
    const diffusionModel = comboInput(settings.diffusion_model_name || "", modelOptions.diffusion_models || modelOptions.unets || [], "vrgdg-wizard-diffusion-models");
    const vae = comboInput(settings.vae_name || "", modelOptions.vae || [], "vrgdg-wizard-vae");
    const clip1 = comboInput(settings.clip_name1 || "", modelOptions.clip || [], "vrgdg-wizard-clip1");
    const clip2 = comboInput(settings.clip_name2 || "", modelOptions.clip || [], "vrgdg-wizard-clip2");
    const upscale = comboInput(settings.upscale_model_name || "", modelOptions.upscale_models || [], "vrgdg-wizard-upscale");
    const audioVae = comboInput(settings.audio_vae_name || "", modelOptions.vae || [], "vrgdg-wizard-audio-vae");
    const unetField = settingField("GGUF UNet model", unet.input, "Main GGUF video generation model.");
    const diffusionModelField = settingField("Diffusion model", diffusionModel.input, "Main safetensors video generation model.");
    const syncVideoModelPickerVisibility = () => {
      const useGguf = Boolean(useGgufModelInput.checked);
      unetField.style.display = useGguf ? "flex" : "none";
      diffusionModelField.style.display = useGguf ? "none" : "flex";
    };
    useGgufModelInput.onchange = syncVideoModelPickerVisibility;
    modelGrid.append(
      useGgufModel,
      unetField,
      diffusionModelField,
      settingField("Video VAE", vae.input, "Decodes generated video latents into final frames."),
      settingField("Gemma CLIP", clip1.input, "Model used for prompt understanding and scene guidance."),
      settingField("Text projection", clip2.input, "Projection model that aligns text features with video generation."),
      settingField("Latent upscaler", upscale.input, "Improves latent resolution before final video decoding."),
      settingField("Audio VAE", audioVae.input, "Audio model used when syncing or conditioning video from audio."),
    );
    modelCard.append(unet.list, diffusionModel.list, vae.list, clip1.list, clip2.list, upscale.list, audioVae.list, modelGrid);
    syncVideoModelPickerVisibility();

    const gemmaCard = el("div", "vrgdg-wizard-settings-card span-4");
    gemmaCard.append(
      el("div", "vrgdg-wizard-settings-title", "Gemma Assist Models"),
      el("div", "vrgdg-wizard-settings-subtitle", "Models used for prompt writing, image inspection, and location mapping."),
    );
    const gemmaGrid = el("div", "vrgdg-wizard-settings-fields two");
    const textGemma = comboInput(data.gemmaSettings?.text_model || "", modelOptions.llm || [], "vrgdg-wizard-text-gemma");
    const visionGemma = comboInput(data.gemmaSettings?.vision_model || "", modelOptions.llm || [], "vrgdg-wizard-vision-gemma");
    const mmproj = comboInput(data.gemmaSettings?.mmproj || "", modelOptions.mmproj || [], "vrgdg-wizard-mmproj");
    gemmaGrid.append(
      settingField("Text Gemma model", textGemma.input, "Used for prompt writing and location mapping."),
      settingField("Vision Gemma model", visionGemma.input, "Used when inspecting images from builder tools."),
      settingField("Vision mmproj", mmproj.input, "Connects vision features to the Gemma model."),
    );
    gemmaCard.append(textGemma.list, visionGemma.list, mmproj.list, gemmaGrid);

    const renderBasicsCard = el("div", "vrgdg-wizard-settings-card span-4");
    renderBasicsCard.append(
      el("div", "vrgdg-wizard-settings-title", "Render Basics"),
      el("div", "vrgdg-wizard-settings-subtitle", "Basic render output settings for your video."),
    );
    const fps = input(settings.fps || 24, "number");
    const width = input(settings.width || 1920, "number");
    const height = input(settings.height || 1080, "number");
    const seed = input(settings.seed || 69, "number");
    const renderGrid = el("div", "vrgdg-wizard-settings-fields compact");
    renderGrid.append(
      settingField("FPS", fps, "Frames per second for the generated video."),
      settingField("Seed", seed, "Controls reproducibility. Use the same seed for repeatable results."),
      settingField("Width", width, "Output video width in pixels."),
      settingField("Height", height, "Output video height in pixels."),
    );
    renderBasicsCard.append(renderGrid);

    const loraCard = el("div", "vrgdg-wizard-settings-card span-4");
    loraCard.append(
      el("div", "vrgdg-wizard-settings-title", "LoRA Settings"),
      el("div", "vrgdg-wizard-settings-subtitle", "LoRA models applied on top of the base video model."),
    );
    const showMsr = isReferenceToVideo;
    const msrLora = comboInput(settings.msr_lora_name || "", modelOptions.loras || [], "vrgdg-wizard-msr-lora");
    const msrStrength = input(settings.msr_first_pass_strength ?? 1, "number");
    msrStrength.step = "0.01";
    const loraTopGrid = el("div", "vrgdg-wizard-settings-fields two");
    const useExtraLoras = document.createElement("label");
    useExtraLoras.style.cssText = "display:flex;align-items:center;gap:8px;color:#dbeafe;font-size:12px;font-weight:900;";
    const useExtraLorasInput = document.createElement("input");
    useExtraLorasInput.type = "checkbox";
    useExtraLorasInput.checked = Boolean(settings.use_loras);
    useExtraLoras.append(useExtraLorasInput, document.createTextNode("Use extra video LoRAs"));
    const loraCount = input(settings.lora_count || 0, "number");
    loraCount.min = "0";
    loraCount.max = "4";
    loraCount.step = "1";
    const extraLoraGrid = el("div", "vrgdg-wizard-settings-fields");
    const extraLoraControls = [];
    for (let index = 0; index < 4; index += 1) {
      const lora = settings.loras?.[index] || {};
      const loraName = comboInput(lora.name || "[none]", modelOptions.loras || [], `vrgdg-wizard-extra-lora-${index + 1}`);
      const strength = input(lora.first_pass_strength ?? lora.strength ?? 1, "number");
      strength.step = "0.01";
      extraLoraGrid.append(
        settingField(`Extra LoRA ${index + 1}`, loraName.input, "Optional extra LoRA model."),
        settingField("Strength", strength, "Strength of the selected extra LoRA."),
      );
      extraLoraControls.push({ name: loraName.input, list: loraName.list, strength });
    }
    const syncExtraLoraVisibility = () => {
      const enabled = Boolean(useExtraLorasInput.checked);
      const count = Math.max(0, Math.min(4, Number(loraCount.value || 0)));
      loraCount.disabled = !enabled;
      extraLoraControls.forEach((control, index) => {
        const visible = enabled && index < count;
        const nameField = control.name.closest(".vrgdg-wizard-settings-field") || control.name.closest(".vrgdg-wizard-field");
        const strengthField = control.strength.closest(".vrgdg-wizard-settings-field") || control.strength.closest(".vrgdg-wizard-field");
        if (nameField) nameField.style.display = visible ? "flex" : "none";
        if (strengthField) strengthField.style.display = visible ? "flex" : "none";
      });
    };
    useExtraLorasInput.onchange = syncExtraLoraVisibility;
    loraCount.oninput = syncExtraLoraVisibility;
    if (showMsr) {
      loraTopGrid.append(
        settingField("Required MSR LoRA", msrLora.input, "Required LoRA applied for Reference to Video."),
        settingField("MSR strength", msrStrength, "Strength of the required Reference to Video LoRA."),
      );
    }
    loraTopGrid.append(
      useExtraLoras,
      settingField("Extra LoRA count", loraCount, "Number of additional LoRAs to apply."),
    );
    const apply = button("Apply Wizard Settings", "primary");
    apply.onclick = async () => {
      apply.disabled = true;
      apply.textContent = "Applying...";
      try {
        await api.applySettings?.({
          use_gguf_model: Boolean(useGgufModelInput.checked),
          unet_name: unet.input.value,
          diffusion_model_name: diffusionModel.input.value,
          vae_name: vae.input.value,
          clip_name1: clip1.input.value,
          clip_name2: clip2.input.value,
          upscale_model_name: upscale.input.value,
          audio_vae_name: audioVae.input.value,
          text_gemma_model: textGemma.input.value,
          vision_gemma_model: visionGemma.input.value,
          mmproj_file: mmproj.input.value,
          fps: Number(fps.value || 24),
          width: Number(width.value || 1920),
          height: Number(height.value || 1080),
          seed: Number(seed.value || 69),
          ...(showMsr ? {
            msr_lora_name: msrLora.input.value,
            msr_first_pass_strength: Number(msrStrength.value || 1),
          } : {}),
          image_model_mode: imageModeSelect.value,
          image_settings: imageModeSelect.value === "nano_banana"
            ? {
              api_key: nanoApiKey.value,
              model: nanoModel.value,
              use_text_only_gemma_prompt: Boolean(nanoUseTextOnlyInput.checked),
              use_director_notes: Boolean(nanoUseDirectorInput.checked),
            }
            : imageModeSelect.value === "flow_gpt"
            ? flowGptSettingsFromControls()
            : {
              unet_name: imageUnet.input.value,
              clip_name: imageClip.input.value,
              vae_name: imageVae.input.value,
            },
          use_loras: Boolean(useExtraLorasInput.checked),
          lora_count: Math.max(0, Math.min(4, Number(loraCount.value || 0))),
          loras: extraLoraControls.map((control) => ({
            name: control.name.value || "[none]",
            first_pass_strength: Number(control.strength.value || 1),
            second_pass_strength: 0,
          })),
        });
        done.add("settings");
        await saveWizardProgress("wizard settings");
        render();
      } finally {
        apply.disabled = false;
        apply.textContent = "Apply Wizard Settings";
      }
    };
    loraCard.append(msrLora.list, ...extraLoraControls.map((control) => control.list), loraTopGrid, extraLoraGrid, apply);
    syncExtraLoraVisibility();

    const tip = el("div", "vrgdg-wizard-tip", "Tip: field descriptions explain what each setting controls. Apply Wizard Settings writes these values back to the normal builder settings used during render.");
    layout.append(settingsCard, quickCard, imageModeCard, imageModelCard, modelCard, gemmaCard, renderBasicsCard, loraCard, tip);
    content.append(layout);
  }

  function renderAudio(data) {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "audio/wav,audio/mpeg,audio/flac,audio/mp4,audio/ogg,.wav,.mp3,.flac,.m4a,.ogg";
    fileInput.style.display = "none";
    const drop = el("div", "vrgdg-wizard-drop");
    drop.innerHTML = `<div><strong style="color:#cffafe;">Drop audio here</strong><br><span style="font-size:12px;color:#94a3b8;">or click to choose WAV, MP3, FLAC, M4A, or OGG.</span></div>`;
    const loaded = el("div", "vrgdg-wizard-note", data.audioPath ? `Loaded audio: ${data.audioPath}` : "No audio is loaded yet. The audio becomes the global project audio.");
    const status = el("div", "vrgdg-wizard-copy", "");
    const loadFile = async (file) => {
      if (!file) return;
      drop.textContent = "Loading audio into the project...";
      status.textContent = "";
      const result = await api.chooseAudioFile?.(file);
      const refreshed = snapshot();
      const savedPath = String(refreshed.audioPath || result?.saved_path || result?.audio_path || "").trim();
      if (!savedPath) {
        done.delete("audio");
        drop.innerHTML = `<div><strong style="color:#cffafe;">Drop audio here</strong><br><span style="font-size:12px;color:#94a3b8;">or click to choose WAV, MP3, FLAC, M4A, or OGG.</span></div>`;
        status.textContent = "Audio was not loaded. Make sure the project is saved first, then try the file again.";
        status.style.color = "#fca5a5";
        return;
      }
      done.add("audio");
      await saveWizardProgress("wizard audio");
      render();
    };
    drop.onclick = () => fileInput.click();
    fileInput.onchange = () => {
      loadFile(fileInput.files?.[0]);
      fileInput.value = "";
    };
    drop.addEventListener("dragover", (event) => {
      event.preventDefault();
      drop.style.borderColor = "#a3e635";
    });
    drop.addEventListener("dragleave", () => {
      drop.style.borderColor = "#22d3ee";
    });
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      drop.style.borderColor = "#22d3ee";
      const file = Array.from(event.dataTransfer?.files || []).find((item) => item.type?.startsWith?.("audio/") || /\.(wav|mp3|flac|m4a|ogg)$/i.test(item.name || ""));
      loadFile(file);
    });
    content.append(fileInput, loaded, drop, status);
  }

  function renderLyrics() {
    const page = el("div", "vrgdg-wizard-lyrics-page");
    const note = el("div", "vrgdg-wizard-info-note");
    note.append(
      el("div", "vrgdg-wizard-info-icon", "i"),
      el("div", "", "This runs the existing timestamped lyric scene creator with wizard defaults: instrumental gaps on, [instrumental] text, 5 second minimum instrumental gap, and 0.3 second vocal tail padding."),
    );
    const lyrics = textarea(wizardState.lyrics, sampleLyrics());
    lyrics.spellcheck = false;
    lyrics.oninput = () => {
      wizardState.lyrics = lyrics.value;
      queueWizardDraftSave();
      syncLineNumbers();
    };
    const language = input(wizardState.language);
    language.oninput = () => {
      wizardState.language = language.value || "english";
      queueWizardDraftSave();
    };
    const segmentMode = select([
      { value: "reference_lines", label: "One scene per lyric line" },
      { value: "reference_stanzas", label: "One scene per stanza" },
      { value: "whisper_chunks", label: "Whisper chunks" },
      { value: "beat_scenes", label: "Beat Mode — cut scenes on detected beats" },
    ], wizardState.segmentMode);
    const segmentationHelpHtml = {
      reference_lines: `
        <strong>What it does:</strong> treats each non-empty lyric line as its own target scene.<br>
        <strong>Use when:</strong> you want tight control and fast scene changes synced to individual lyric lines.<br>
        <strong>Result:</strong> usually more scenes with shorter durations.
      `,
      reference_stanzas: `
        <strong>What it does:</strong> groups lyric blocks separated by blank lines into scenes.<br>
        <strong>Use when:</strong> each verse, chorus, bridge, or instrumental block should stay together.<br>
        <strong>Result:</strong> fewer scenes with smoother, longer shots.
      `,
      whisper_chunks: `
        <strong>What it does:</strong> lets Whisper/stable-ts choose scene chunks from the audio transcription timing.<br>
        <strong>Use when:</strong> your pasted lyrics are only a guide, or you want timing driven mostly by the audio.<br>
        <strong>Result:</strong> less manual structure, but scene breaks may not match your pasted line breaks.
      `,
      beat_scenes: `
        <strong>What it does:</strong> detects musical beats and chooses scene cuts inside your minimum/maximum duration window.<br>
        <strong>Use when:</strong> you want scene changes to feel synchronized to the rhythm.<br>
        <strong>Result:</strong> beat-aligned scene timing. If no beat exists inside the selected window, it cuts at the maximum instead.
      `,
    };
    const segmentationHelp = el("div", "vrgdg-wizard-setting-help");
    segmentationHelp.innerHTML = segmentationHelpHtml[segmentMode.value] || segmentationHelpHtml.reference_lines;
    segmentMode.onchange = () => {
      wizardState.segmentMode = segmentMode.value || "reference_lines";
      queueWizardDraftSave();
      segmentationHelp.innerHTML = segmentationHelpHtml[wizardState.segmentMode] || segmentationHelpHtml.reference_lines;
    };
    const minScene = input(wizardState.minSceneSeconds, "number");
    minScene.step = "0.1";
    minScene.oninput = () => {
      wizardState.minSceneSeconds = minScene.value || "1.0";
      queueWizardDraftSave();
    };
    const maxScene = input(wizardState.maxSceneSeconds, "number");
    maxScene.step = "0.1";
    maxScene.oninput = () => {
      wizardState.maxSceneSeconds = maxScene.value || "8.0";
      queueWizardDraftSave();
    };
    const lyricsSection = el("div", "vrgdg-wizard-section");
    const lyricsHeading = el("div", "vrgdg-wizard-section-heading");
    lyricsHeading.append(el("div", "vrgdg-wizard-section-number", "1"), el("div", "vrgdg-wizard-section-title", "Lyrics"));
    const lyricsCopy = el("div", "vrgdg-wizard-section-copy", "Paste or edit your timestamped lyrics below. Each section will be used to generate scenes.");
    const editor = el("div", "vrgdg-wizard-lyrics-editor");
    const gutter = el("div", "vrgdg-wizard-line-gutter", "1");
    const syncLineNumbers = () => {
      const lineCount = Math.max(1, String(lyrics.value || "").split(/\r?\n/).length);
      gutter.textContent = Array.from({ length: lineCount }, (_, index) => String(index + 1)).join("\n");
    };
    lyrics.addEventListener("scroll", () => {
      gutter.scrollTop = lyrics.scrollTop;
    });
    editor.append(gutter, lyrics);
    const lyricsHint = el("div", "vrgdg-wizard-copy", "Use [instrumental], [break], or [outro] for no-vocal sections.");
    lyricsHint.style.marginTop = "8px";
    lyricsSection.append(lyricsHeading, lyricsCopy, editor, lyricsHint);

    const settingsSection = el("div", "vrgdg-wizard-section");
    const settingsHeading = el("div", "vrgdg-wizard-section-heading");
    settingsHeading.append(el("div", "vrgdg-wizard-section-number", "2"), el("div", "vrgdg-wizard-section-title", "Scene Settings"));
    const settingsCopy = el("div", "vrgdg-wizard-section-copy", "Configure how scenes are generated from your lyrics.");
    const settingTile = (title, control, help) => {
      const tile = el("div", "vrgdg-wizard-setting-tile");
      const helpNode = help instanceof HTMLElement ? help : el("div", "vrgdg-wizard-setting-help", help);
      tile.append(el("div", "vrgdg-wizard-setting-title", title), control, helpNode);
      return tile;
    };
    const grid = el("div", "vrgdg-wizard-scene-settings-grid");
    grid.append(
      settingTile("Language", language, "Tells the transcription/alignment step what language to expect. Use english for English lyrics; use the spoken lyric language for non-English songs."),
      settingTile("Scene segmentation", segmentMode, segmentationHelp),
      settingTile("Minimum scene seconds", minScene, "Scenes shorter than this are merged or stretched when possible. Raise it to avoid tiny unusable clips; lower it for quick cuts."),
      settingTile("Maximum scene seconds", maxScene, "Scenes longer than this are split or capped when possible. Lower it for faster pacing; raise it for longer cinematic shots."),
    );
    const run = button("Create Timeline Scenes From Lyrics", "primary");
    run.onclick = async () => {
      run.disabled = true;
      run.textContent = "Creating scenes...";
      run.classList.add("is-working");
      try {
        await api.createScenesFromLyrics?.({
          referenceLyrics: wizardState.lyrics || "",
          language: wizardState.language || "english",
          segmentMode: wizardState.segmentMode || "reference_lines",
          includeInstrumentalGaps: true,
          instrumentalText: "[instrumental]",
          minGapSeconds: 5.0,
          minSceneSeconds: Number(wizardState.minSceneSeconds || 1.0),
          maxSceneSeconds: Number(wizardState.maxSceneSeconds || 8.0),
          vocalTailPaddingSeconds: 0.3,
          beatUseSrtDurations: true,
          beatMinDuration: Number(wizardState.minSceneSeconds || 1.0),
          beatMaxDuration: Number(wizardState.maxSceneSeconds || 8.0),
          beatBias: 0.7,
          beatDurationPreset: "varied_no_repeat",
          beatEmptySegmentText: "Instrumental section.",
        });
        done.add("lyrics");
        await saveWizardProgress("wizard lyrics and scenes");
        render();
      } finally {
        run.disabled = false;
        run.classList.remove("is-working");
        run.textContent = "Create Timeline Scenes From Lyrics";
      }
    };
    const actions = el("div", "vrgdg-wizard-lyrics-actions");
    actions.append(run);
    settingsSection.append(settingsHeading, settingsCopy, grid, actions);
    page.append(note, lyricsSection, settingsSection);
    content.append(page);
    syncLineNumbers();
  }

  function renderMode(data) {
    const grid = el("div", "vrgdg-wizard-grid");
    const modes = [
      ["i2v", "Image to Video", "Creates image prompts first, generates scene images, then writes I2V prompts and renders videos.", true],
      ["rtv", "Reference to Video", "Uses LTX Reference-to-Video / MSR reference images and text prompts.", true],
      ["t2v", "Text to Video", "Coming soon in the wizard. Text-only mapping stays available in Reference Builder.", false],
      ["ingredients", "Ingredients to Video", "Coming soon in the wizard. Ingredients Builder stays available from the main UI.", false],
    ];
    for (const [mode, title, copy, enabled] of modes) {
      const item = el("div", `vrgdg-wizard-mode-card${data.videoMode === mode ? " is-active" : ""}${enabled ? "" : " is-disabled"}`);
      item.append(el("div", "vrgdg-wizard-card-title", title), el("div", "vrgdg-wizard-copy", copy));
      const action = button(enabled ? "Use This Mode" : "Coming Soon", enabled ? "primary" : "");
      action.disabled = !enabled;
      action.onclick = () => {
        setWizardMode(mode);
      };
      item.append(action);
      grid.append(item);
    }
    content.append(grid);
  }

  function renderReferences(data) {
    const videoMode = data.videoMode || "i2v";
    const isRtv = videoMode === "rtv";
    const note = el("div", "vrgdg-wizard-note", isRtv
      ? "Use the LTX Reference Builder and Lyric Mapping tools here. This keeps subjects, locations, singer/no-lip-sync choices, and Storyboard data synced through the same saved project data."
      : "Use Scene Text Mapping and Lyric Mapping here. This gives Gemma subject/location context for image prompts and I2V prompts without forcing Reference-to-Video image refs.");
    const grid = el("div", "vrgdg-wizard-grid");
    const ref = card(
      isRtv ? "1. LTX Reference Builder" : "1. Scene Text Mapping",
      isRtv
        ? "Add or generate subject references, create/import locations, and map them to scenes."
        : "Add subject/location descriptions and map them to scenes for Gemma image and I2V prompt writing."
    );
    const openRef = button(isRtv ? "Open LTX Reference Builder" : "Open Scene Text Mapping", "primary");
    openRef.onclick = () => {
      done.add("references");
      openNestedTool(() => api.openReferenceBuilder?.(videoMode), "wizard references");
    };
    ref.append(openRef);
    const lyric = card("2. Review Lyrics + Map Singers", "Correct lyric notes, choose who sings, mark B-roll/instrumental sections, and mark no-character scenes when needed.");
    const openLyrics = button("Open Review + Map Singers", "primary");
    openLyrics.onclick = () => {
      openNestedTool(() => (api.openLyricReview || api.openLyricMapping)?.(), "wizard lyric review");
    };
    lyric.append(openLyrics);
    const defaultsData = data.sceneDefaults || {};
    const cameraOptions = Array.isArray(defaultsData.cameraFlowOptions) ? defaultsData.cameraFlowOptions : [];
    const performanceOptions = Array.isArray(defaultsData.performanceStyleOptions) ? defaultsData.performanceStyleOptions : [];
    const facialOptions = Array.isArray(defaultsData.facialPerformanceOptions) ? defaultsData.facialPerformanceOptions : [];
    const defaults = card("3. Scene Defaults", "Fill shot direction, camera motion, performance style, and facial performance without opening Storyboard Builder.");
    const cameraSelect = select(cameraOptions.map((item) => ({ value: item.value, label: item.label })), wizardState.cameraFlow || defaultsData.cameraFlow || "balanced");
    const cameraInfo = el("div", "vrgdg-wizard-copy", "");
    const refreshCameraInfo = () => {
      const selected = cameraOptions.find((item) => item.value === cameraSelect.value) || cameraOptions[0] || {};
      cameraInfo.textContent = selected.value === "off"
        ? (selected.description || "Auto camera flow is off.")
        : `${selected.description || ""}${selected.count ? ` Cycles through ${selected.count} camera beats and only fills blank fields unless Replace All is used.` : ""}`;
    };
    const cameraFill = button("Fill Missing", "primary");
    const cameraReplace = button("Replace All");
    const cameraRow = el("div", "vrgdg-wizard-button-row");
    cameraFill.onclick = async () => {
      cameraFill.disabled = true;
      try {
        await api.applySceneDefaults?.({
          cameraFlow: wizardState.cameraFlow || cameraSelect.value || "balanced",
          performanceStyle: "",
          overwriteCamera: false,
          overwritePerformance: false,
        });
        await saveWizardProgress("wizard camera defaults");
        render();
      } finally {
        cameraFill.disabled = false;
      }
    };
    cameraReplace.onclick = async () => {
      cameraReplace.disabled = true;
      try {
        await api.applySceneDefaults?.({
          cameraFlow: wizardState.cameraFlow || cameraSelect.value || "balanced",
          performanceStyle: "",
          overwriteCamera: true,
          overwritePerformance: false,
        });
        await saveWizardProgress("wizard camera defaults replace");
        render();
      } finally {
        cameraReplace.disabled = false;
      }
    };
    cameraRow.append(cameraSelect, cameraFill, cameraReplace);
    const performanceSelect = select(performanceOptions.map((item) => ({ value: item.value, label: item.label })), wizardState.performanceStyle ?? defaultsData.performanceStyle ?? "");
    const performanceInfo = el("div", "vrgdg-wizard-copy", "");
    const refreshPerformanceInfo = () => {
      const selected = performanceOptions.find((item) => item.value === performanceSelect.value) || performanceOptions[0] || {};
      performanceInfo.textContent = performanceSelect.value
        ? `${selected.description || ""} Used for scenes without a per-scene performance style.`
        : `${selected.description || "Pick a style here to use it as the default for blank scenes."}`;
    };
    const performanceFill = button("Fill Missing", "primary");
    const performanceReplace = button("Replace All");
    const performanceRow = el("div", "vrgdg-wizard-button-row");
    performanceFill.onclick = async () => {
      performanceFill.disabled = true;
      try {
        await api.applySceneDefaults?.({
          cameraFlow: "off",
          performanceStyle: wizardState.performanceStyle ?? performanceSelect.value ?? "",
          applyPerformance: true,
          overwriteCamera: false,
          overwritePerformance: false,
        });
        await saveWizardProgress("wizard performance defaults");
        render();
      } finally {
        performanceFill.disabled = false;
      }
    };
    performanceReplace.onclick = async () => {
      performanceReplace.disabled = true;
      try {
        await api.applySceneDefaults?.({
          cameraFlow: "off",
          performanceStyle: wizardState.performanceStyle ?? performanceSelect.value ?? "",
          applyPerformance: true,
          overwriteCamera: false,
          overwritePerformance: true,
        });
        await saveWizardProgress("wizard performance defaults replace");
        render();
      } finally {
        performanceReplace.disabled = false;
      }
    };
    performanceRow.append(performanceSelect, performanceFill, performanceReplace);
    const facialSelect = select(facialOptions.map((item) => ({ value: item.value, label: item.label })), wizardState.facialPerformance ?? defaultsData.facialPerformance ?? "");
    const facialCustom = textarea(wizardState.facialPerformanceCustom ?? defaultsData.facialPerformanceCustom ?? "", "Optional custom facial performance text...");
    facialCustom.style.minHeight = "62px";
    const facialInfo = el("div", "vrgdg-wizard-copy", "");
    const refreshFacialInfo = () => {
      const selected = facialOptions.find((item) => item.value === facialSelect.value) || facialOptions[0] || {};
      facialInfo.textContent = facialSelect.value || facialCustom.value
        ? `${selected.description || "Custom facial performance text"} Used for scenes without per-scene facial performance.`
        : `${selected.description || "Pick a facial preset or enter custom text to use it as the default for blank scenes."}`;
    };
    const facialFill = button("Fill Missing", "primary");
    const facialReplace = button("Replace All");
    const facialRow = el("div", "vrgdg-wizard-button-row");
    facialFill.onclick = async () => {
      facialFill.disabled = true;
      try {
        await api.applySceneDefaults?.({
          cameraFlow: "off",
          facialPerformance: wizardState.facialPerformance ?? facialSelect.value ?? "",
          facialPerformanceCustom: wizardState.facialPerformanceCustom ?? facialCustom.value ?? "",
          applyFacialPerformance: true,
          overwriteCamera: false,
          overwriteFacialPerformance: false,
        });
        await saveWizardProgress("wizard facial defaults");
        render();
      } finally {
        facialFill.disabled = false;
      }
    };
    facialReplace.onclick = async () => {
      facialReplace.disabled = true;
      try {
        await api.applySceneDefaults?.({
          cameraFlow: "off",
          facialPerformance: wizardState.facialPerformance ?? facialSelect.value ?? "",
          facialPerformanceCustom: wizardState.facialPerformanceCustom ?? facialCustom.value ?? "",
          applyFacialPerformance: true,
          overwriteCamera: false,
          overwriteFacialPerformance: true,
        });
        await saveWizardProgress("wizard facial defaults replace");
        render();
      } finally {
        facialReplace.disabled = false;
      }
    };
    facialRow.append(facialSelect, facialFill, facialReplace);
    cameraSelect.onchange = () => {
      wizardState.cameraFlow = cameraSelect.value || "balanced";
      queueWizardDraftSave();
      refreshCameraInfo();
    };
    performanceSelect.onchange = () => {
      wizardState.performanceStyle = performanceSelect.value || "";
      queueWizardDraftSave();
      refreshPerformanceInfo();
    };
    facialSelect.onchange = () => {
      wizardState.facialPerformance = facialSelect.value || "";
      queueWizardDraftSave();
      refreshFacialInfo();
    };
    facialCustom.oninput = () => {
      wizardState.facialPerformanceCustom = facialCustom.value || "";
      queueWizardDraftSave();
      refreshFacialInfo();
    };
    defaults.append(
      el("div", "vrgdg-wizard-card-title", "Auto camera flow"),
      cameraRow,
      cameraInfo,
      el("div", "vrgdg-wizard-card-title", "Global performance style"),
      performanceRow,
      performanceInfo,
      el("div", "vrgdg-wizard-card-title", "Global facial performance"),
      facialRow,
      facialCustom,
      facialInfo,
    );
    refreshCameraInfo();
    refreshPerformanceInfo();
    refreshFacialInfo();
    const status = card("Current Mapping", `Mode: ${data.videoModeLabel || videoMode} | Subjects: ${Number(data.subjectCount || 0)} | Locations: ${Number(data.locationCount || 0)} | Scenes: ${Number(data.sceneCount || 0)}`);
    grid.append(ref, lyric, defaults, status);
    content.append(note, grid);
  }

  function renderStory(data) {
    const incomingLayer = data.storyLayer && typeof data.storyLayer === "object" ? data.storyLayer : {};
    wizardState.storyLayer = {
      enabled: incomingLayer.enabled !== false,
      user_story_arc: wizardState.storyLayer?.user_story_arc || incomingLayer.user_story_arc || "",
      song_story_brief: wizardState.storyLayer?.song_story_brief || incomingLayer.song_story_brief || "",
      lyric_story_strength: lyricStoryStrengthValue(wizardState.storyLayer?.lyric_story_strength ?? incomingLayer.lyric_story_strength ?? incomingLayer.lyricStoryStrength),
    };
    const note = el("div", "vrgdg-wizard-note", "Use this optional story layer to give Gemma a compact narrative arc. The final video prompts will use this without sending the full lyrics every time.");
    const grid = el("div", "vrgdg-wizard-grid");
    const arc = card("1. User Story Arc", "Optional direction Gemma should follow more strongly than its own interpretation.");
    const enabledLabel = el("label", "vrgdg-wizard-check");
    const enabled = document.createElement("input");
    enabled.type = "checkbox";
    enabled.checked = wizardState.storyLayer.enabled !== false;
    enabledLabel.append(enabled, document.createTextNode("Use story layer in prompt generation"));
    const lyricStrength = input(String(lyricStoryStrengthValue(wizardState.storyLayer.lyric_story_strength)), "range");
    lyricStrength.min = "0";
    lyricStrength.max = "10";
    lyricStrength.step = "1";
    lyricStrength.style.accentColor = "#22d3ee";
    const lyricStrengthValue = el("div", "vrgdg-wizard-copy");
    lyricStrengthValue.style.fontWeight = "900";
    lyricStrengthValue.style.color = "#cffafe";
    const lyricStrengthText = (value) => {
      const strength = Math.max(0, Math.min(10, Number(value || 7)));
      if (strength <= 0) return "0 / ignore lyrics";
      if (strength <= 3) return `${strength} / mood only`;
      if (strength <= 6) return `${strength} / balanced`;
      if (strength <= 8) return `${strength} / strong lyric story`;
      return `${strength} / literal lyric anchors`;
    };
    const syncLyricStrength = () => {
      wizardState.storyLayer.lyric_story_strength = lyricStoryStrengthValue(lyricStrength.value);
      lyricStrengthValue.textContent = lyricStrengthText(lyricStrength.value);
    };
    syncLyricStrength();
    const lyricHint = button("Hint");
    lyricHint.onclick = () => {
      window.alert([
        "Lyric Story Strength controls how literally Gemma should follow the lyrics when creating the story arc, story brief, scene beats, and prompt context.",
        "",
        "0: do not use lyrics as story source.",
        "1-3: use lyrics as mood and emotional timing only.",
        "4-6: balance lyrics with the story arc, subjects, and locations.",
        "7-8: lyrics strongly shape the scene story; include recognizable lyric anchors when possible.",
        "9-10: use lyrics as literally as possible; non-instrumental scenes should include a concrete object, action, emotion, or situation from the exact lyric line whenever possible.",
      ].join("\n"));
    };
    const lyricStrengthRow = el("div", "vrgdg-wizard-settings-fields two");
    lyricStrengthRow.style.gridTemplateColumns = "minmax(0,1fr) auto auto";
    lyricStrengthRow.style.alignItems = "end";
    lyricStrengthRow.append(field("Lyric Story Strength", lyricStrength), lyricStrengthValue, lyricHint);
    lyricStrength.addEventListener("input", () => {
      syncLyricStrength();
      queueWizardDraftSave();
    });
    const arcText = textarea(wizardState.storyLayer.user_story_arc || "", "Verse 1: ...\nChorus 1: ...\nVerse 2: ...");
    arcText.style.minHeight = "190px";
    arcText.addEventListener("input", () => {
      wizardState.storyLayer.user_story_arc = arcText.value;
      queueWizardDraftSave();
    });
    const motionSlider = input(String(Math.max(0, Math.min(10, Number(wizardState.storyCharacterMotion ?? 7)))), "range");
    motionSlider.min = "0";
    motionSlider.max = "10";
    motionSlider.step = "1";
    motionSlider.style.accentColor = "#22d3ee";
    const motionNumber = input(String(Math.max(0, Math.min(10, Number(wizardState.storyCharacterMotion ?? 7)))), "number");
    motionNumber.min = "0";
    motionNumber.max = "10";
    motionNumber.step = "1";
    const motionHint = el("div", "vrgdg-wizard-copy");
    const syncMotionLabel = () => {
      const value = Math.max(0, Math.min(10, Number(motionNumber.value || motionSlider.value || 7)));
      motionSlider.value = String(value);
      motionNumber.value = String(value);
      wizardState.storyCharacterMotion = value;
      motionHint.textContent = value <= 2
        ? "Mostly still: subtle gestures and restrained performance."
        : value <= 5
          ? "Moderate motion: turns, hand movement, controlled blocking."
          : value <= 8
            ? "Active: walking, moving through the space, touching objects, using the environment."
            : "Highly active: strong physical beats, dancing/running/climbing/forceful interaction.";
    };
    motionSlider.addEventListener("input", () => {
      motionNumber.value = motionSlider.value;
      syncMotionLabel();
      queueWizardDraftSave();
    });
    motionNumber.addEventListener("input", () => {
      syncMotionLabel();
      queueWizardDraftSave();
    });
    syncMotionLabel();
    enabled.addEventListener("change", () => {
      wizardState.storyLayer.enabled = Boolean(enabled.checked);
      queueWizardDraftSave();
    });
    const arcButton = button("Create User Story Arc", "primary");
    arcButton.onclick = async () => {
      arcButton.disabled = true;
      try {
        wizardState.storyLayer.user_story_arc = arcText.value;
        wizardState.storyLayer.enabled = Boolean(enabled.checked);
        wizardState.storyLayer.lyric_story_strength = lyricStoryStrengthValue(lyricStrength.value);
        const updated = await api.createStoryArc?.({
          storyLayer: wizardState.storyLayer,
          userStoryArc: arcText.value,
          storyIdea: arcText.value,
          characterMotion: wizardState.storyCharacterMotion,
        });
        if (updated) {
          wizardState.storyLayer = { ...wizardState.storyLayer, ...updated };
          arcText.value = wizardState.storyLayer.user_story_arc || "";
        }
        done.add("story");
        await saveWizardProgress("wizard story arc");
      } finally {
        arcButton.disabled = false;
      }
    };
    const motionRow = el("div", "vrgdg-wizard-settings-fields two");
    motionRow.append(field("Character motion", motionSlider, "0 = still/posed, 10 = very active and interactive"), field("Motion value", motionNumber));
    arc.append(enabledLabel, lyricStrengthRow, arcText, motionRow, motionHint, arcButton);
    const brief = card("2. Song Story Brief", "A compact Gemma summary of the song's premise, emotional arc, motifs, and scene guidance.");
    const briefText = textarea(wizardState.storyLayer.song_story_brief || "", "Create or edit the song story brief...");
    briefText.style.minHeight = "190px";
    briefText.addEventListener("input", () => {
      wizardState.storyLayer.song_story_brief = briefText.value;
      queueWizardDraftSave();
    });
    const briefButton = button("Create Story Brief", "primary");
    briefButton.onclick = async () => {
      briefButton.disabled = true;
      try {
        wizardState.storyLayer.user_story_arc = arcText.value;
        wizardState.storyLayer.enabled = Boolean(enabled.checked);
        wizardState.storyLayer.lyric_story_strength = lyricStoryStrengthValue(lyricStrength.value);
        const updated = await api.createStoryBrief?.({
          storyLayer: wizardState.storyLayer,
          userStoryArc: arcText.value,
        });
        if (updated) {
          wizardState.storyLayer = { ...wizardState.storyLayer, ...updated };
          briefText.value = wizardState.storyLayer.song_story_brief || "";
        }
        done.add("story");
        await saveWizardProgress("wizard story brief");
      } finally {
        briefButton.disabled = false;
      }
    };
    brief.append(briefText, briefButton);
    const beats = card("3. Scene Story Beats", "Create short per-scene narrative beats using lyrics, sections, subjects, locations, and the story brief.");
    const beatStatus = el("div", "vrgdg-wizard-copy", `Lyric sections: ${Number(data.lyricSectionCount || 0)} / ${Number(data.sceneCount || 0)} | Scene beats: ${Number(data.storyBeatCount || 0)} / ${Number(data.sceneCount || 0)}`);
    const detect = button("Detect Lyric Sections", "primary");
    const missing = button("Create Missing Scene Beats", "primary");
    const replace = button("Replace All Scene Beats");
    detect.onclick = async () => {
      detect.disabled = true;
      try {
        await api.detectLyricSections?.(wizardState.lyrics || "");
        done.add("story");
        render();
      } finally {
        detect.disabled = false;
      }
    };
    missing.onclick = async () => {
      missing.disabled = true;
      try {
        wizardState.storyLayer.user_story_arc = arcText.value;
        wizardState.storyLayer.song_story_brief = briefText.value;
        wizardState.storyLayer.lyric_story_strength = lyricStoryStrengthValue(lyricStrength.value);
        await api.updateStoryLayer?.(wizardState.storyLayer);
        await api.createSceneBeats?.({ overwrite: false });
        done.add("story");
        render();
      } finally {
        missing.disabled = false;
      }
    };
    replace.onclick = async () => {
      replace.disabled = true;
      try {
        wizardState.storyLayer.user_story_arc = arcText.value;
        wizardState.storyLayer.song_story_brief = briefText.value;
        wizardState.storyLayer.lyric_story_strength = lyricStoryStrengthValue(lyricStrength.value);
        await api.updateStoryLayer?.(wizardState.storyLayer);
        await api.createSceneBeats?.({ overwrite: true });
        done.add("story");
        render();
      } finally {
        replace.disabled = false;
      }
    };
    const beatActions = el("div", "vrgdg-wizard-button-row");
    beatActions.append(detect, missing, replace);
    beats.append(beatStatus, beatActions);
    grid.append(arc, brief, beats);
    content.append(note, grid);
  }

  function renderFinish(data) {
    const videoMode = data.videoMode || "i2v";
    const isRtv = videoMode === "rtv";
    const isFlowGpt = String(data.imageMode || "") === "flow_gpt";
    const imageModeLabel = data.imageModeLabel || data.imageMode || "current image mode";
    const note = el("div", "vrgdg-wizard-note", isRtv
      ? "When mapping and scene cards look right, Build Full Video can create Reference-to-Video prompts, render the clips, and stitch the final video using the current Ref to Video settings. Use the prompt button only when you want to preview or regenerate prompts first."
      : `Build Full Video can run the whole Image-to-Video pipeline: Gemma image prompts, ${imageModeLabel} images, Gemma video prompts, scene videos, and final stitching. Use the Gemma buttons only when you want to preview or regenerate prompts before the full build.`);
    const grid = el("div", "vrgdg-wizard-grid");
    if (!isRtv) {
      const imagePrompts = card("Optional: Gemma Image All", "Creates image prompts only. Build Full Video also runs this when prompts/images are missing or when you choose a rebuild mode.");
      const gemmaImage = button("Run Gemma Image All", "primary");
      gemmaImage.onclick = () => {
        openNestedTool(() => api.runGemmaImageAll?.(), "wizard gemma image all");
      };
      imagePrompts.append(gemmaImage);
      grid.append(imagePrompts);
      if (isFlowGpt) {
        const flowGptImages = card("Run Flow/GPT Image All", "Creates missing Flow/GPT prompts if needed, then runs the browser image pass only. The Wizard closes so you can watch scenes update.");
        const flowGptImageAll = button("Run Flow/GPT Image All", "primary");
        flowGptImageAll.onclick = () => {
          closeAndRun(() => api.runFlowGptImageAll?.(), "wizard Flow/GPT image all");
        };
        flowGptImages.append(flowGptImageAll);
        grid.append(flowGptImages);
      }
    }
    const prompts = card(isRtv ? "Optional: Create Video Prompts" : "Optional: Gemma Video All", isRtv
      ? "Creates video prompts only. Build Full Video can also create missing Reference-to-Video prompts."
      : "Creates I2V prompts only. Build Full Video can also create missing video prompts after the image stage.");
    const gemma = button(isRtv ? "Run Storyboard Gemma All" : "Run Gemma Video All", "primary");
    gemma.onclick = () => {
      openNestedTool(() => api.runGemmaVideoAll?.(), "wizard gemma video all");
    };
    prompts.append(gemma);
    const build = card("Build Full Video", isRtv
      ? "Runs the full Reference-to-Video render/stitch flow."
      : `Runs the full pipeline. If Flow/GPT is selected, this is where the browser image pass runs using the Wizard Flow/GPT settings.`);
    const full = button("Build Full Video", "primary");
    full.onclick = () => {
      openNestedTool(() => api.buildFullVideo?.(), "wizard full build");
    };
    build.append(full);
    grid.append(prompts, build);
    content.append(note, grid);
  }

  function render() {
    const data = snapshot();
    const step = steps[activeIndex];
    for (const [id, node] of stepButtons.entries()) {
      node.classList.toggle("is-active", id === step.id);
      node.classList.toggle("is-done", done.has(id));
    }
    progressBar.style.width = `${((activeIndex + 1) / steps.length) * 100}%`;
    header.textContent = "";
    content.textContent = "";
    main.textContent = "";
    renderHeader(step);
    try {
      if (step.id === "settings") renderSettings(data);
      else if (step.id === "audio") renderAudio(data);
      else if (step.id === "lyrics") renderLyrics();
      else if (step.id === "mode") renderMode(data);
      else if (step.id === "references") renderReferences(data);
      else if (step.id === "story") renderStory(data);
      else renderFinish(data);
    } catch (error) {
      const box = el("div", "vrgdg-wizard-info-note");
      box.style.borderColor = "#7f1d1d";
      box.style.background = "#451a1a";
      box.append(
        el("div", "vrgdg-wizard-info-icon", "!"),
        el("div", "", `Wizard page failed to render:\n${String(error?.message || error)}`),
      );
      content.append(box);
    }
    prev.disabled = activeIndex === 0;
    next.textContent = activeIndex >= steps.length - 1 ? "Done" : "Next";
    prev.onclick = () => {
      activeIndex = Math.max(0, activeIndex - 1);
      render();
    };
    next.onclick = () => {
      if (activeIndex >= steps.length - 1) {
        closeWizard();
        return;
      }
      activeIndex += 1;
      render();
    };
    main.append(header, content, footer);
  }

  restoreLocalWizardDraft();
  render();
  restoreProjectWizardDraft();
}
