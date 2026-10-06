import { makeButton, makeCheckbox, normalizeProjectVideoEngine } from "./controls.mjs";

// LLM Prompting pop-out: a separate floating window with the MiniMax H3 prompt, the Save Updated Prompt button, the
// character count line and the 2nd pass prompt. The right panel is left exactly as it is. The window holds mirrors:
// typing in either one updates the other, the button presses the real Save button, and everything follows the scene
// you select, so it does what the panel does. Drag the title bar to move it, drag its corner to resize it. A
// checkbox at the top of the right panel turns it on and off.

const MIN_WIDTH = 320;
const MIN_HEIGHT = 240;
const SYNC_MS = 200;

const FIELD_LABEL_STYLE = "font-size:11px;font-weight:700;color:#a1a1aa;margin:2px 0 4px;";
const AREA_STYLE = "width:100%;box-sizing:border-box;resize:none;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:9px;font-size:12px;line-height:1.4;";

function mirrorTextarea(original, placeholder) {
  const mirror = document.createElement("textarea");
  mirror.placeholder = placeholder;
  mirror.style.cssText = AREA_STYLE;
  // The panel's own fields keep Builder shortcuts from reacting to typing. The mirror does the same.
  ["keydown", "keypress", "keyup"].forEach((name) => mirror.addEventListener(name, (event) => event.stopPropagation()));
  // Typing here goes to the real field and runs its own handlers (dirty state, character count).
  mirror.addEventListener("input", () => {
    original.value = mirror.value;
    original.dispatchEvent(new Event("input", { bubbles: true }));
  });
  return mirror;
}

export function createLlmPopout({
  state, inspector, overlay, fields, autoSaveSessionQuiet, activeSegment, sceneDisplayName, segmentIndexInfo,
}) {
  const { prompt, saveButton, status, pass2Prompt, pass2Field } = fields;

  const win = document.createElement("div");
  win.style.cssText = `position:fixed;z-index:100002;display:none;flex-direction:column;overflow:hidden;box-sizing:border-box;min-width:${MIN_WIDTH}px;min-height:${MIN_HEIGHT}px;border:1px solid #155e75;border-radius:8px;background:#202024;color:#fafafa;box-shadow:0 18px 60px rgba(0,0,0,.6);resize:both;`;

  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 10px;border-bottom:1px solid #27272a;background:#083344;cursor:move;user-select:none;flex:0 0 auto;";
  const title = document.createElement("div");
  title.textContent = "LLM Prompting";
  title.style.cssText = "font-size:12px;font-weight:900;color:#cffafe;";
  const sceneLabel = document.createElement("div");
  sceneLabel.style.cssText = "flex:1;min-width:0;font-size:11px;color:#67e8f9;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
  const closeButton = makeButton("×");
  closeButton.title = "Close this window";
  closeButton.style.padding = "3px 9px";
  header.append(title, sceneLabel, closeButton);

  const body = document.createElement("div");
  body.style.cssText = "display:flex;flex-direction:column;gap:8px;padding:10px;overflow-y:auto;overflow-x:hidden;min-height:0;flex:1 1 auto;scrollbar-width:thin;";

  const notMiniMaxNote = document.createElement("div");
  notMiniMaxNote.textContent = "LLM Prompting is part of MiniMax H3. Click the LTX badge at the top right to switch this project to MiniMax H3.";
  notMiniMaxNote.style.cssText = "display:none;font-size:12px;line-height:1.5;color:#a1a1aa;border:1px dashed #3f3f46;border-radius:8px;padding:14px;";

  const promptLabel = document.createElement("div");
  promptLabel.textContent = "MiniMax H3 prompt";
  promptLabel.style.cssText = FIELD_LABEL_STYLE;
  const promptMirror = mirrorTextarea(prompt, prompt.placeholder);
  promptMirror.style.flex = "1 1 220px";
  promptMirror.style.minHeight = "140px";

  const saveMirror = makeButton(saveButton.textContent || "Save Updated Prompt", "primary");
  saveMirror.style.width = "100%";
  saveMirror.style.flex = "0 0 auto";
  saveMirror.addEventListener("click", () => saveButton.click());

  const statusMirror = document.createElement("div");
  statusMirror.style.flex = "0 0 auto";

  const pass2Wrap = document.createElement("div");
  pass2Wrap.style.cssText = "display:none;flex-direction:column;flex:0 0 auto;";
  const pass2Label = document.createElement("div");
  pass2Label.textContent = "2nd Pass Prompt";
  pass2Label.style.cssText = FIELD_LABEL_STYLE;
  const pass2Mirror = mirrorTextarea(pass2Prompt, pass2Prompt.placeholder);
  pass2Mirror.style.minHeight = "110px";
  pass2Wrap.append(pass2Label, pass2Mirror);

  const content = document.createElement("div");
  content.style.cssText = "display:flex;flex-direction:column;gap:8px;flex:1 1 auto;min-height:0;";
  content.append(promptLabel, promptMirror, saveMirror, statusMirror, pass2Wrap);
  body.append(notMiniMaxNote, content);
  win.append(header, body);
  overlay.append(win);

  const toggle = makeCheckbox("Pop out LLM Prompting into its own window", false);
  toggle.wrapper.style.cssText += "border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:6px 9px;font-size:12px;flex:0 0 auto;";
  toggle.wrapper.title = "Opens the MiniMax H3 prompt, Save Updated Prompt, the character count and the 2nd pass prompt in a window of their own. The right panel stays as it is.";
  inspector.prepend(toggle.wrapper);

  function isMiniMaxProject() {
    return normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
  }

  // ---- placement: where the window sits and how big it is, kept in state so it can be saved
  function clampToScreen() {
    const width = Math.max(MIN_WIDTH, Math.min(win.offsetWidth || state.llmPopoutWidth, window.innerWidth - 16));
    const height = Math.max(MIN_HEIGHT, Math.min(win.offsetHeight || state.llmPopoutHeight, window.innerHeight - 16));
    const x = Math.max(8, Math.min(Number.isFinite(state.llmPopoutX) ? state.llmPopoutX : 0, window.innerWidth - width - 8));
    const y = Math.max(8, Math.min(Number.isFinite(state.llmPopoutY) ? state.llmPopoutY : 0, window.innerHeight - height - 8));
    return { x, y, width, height };
  }

  function place() {
    // First time: beside the right panel, so it opens where it is easiest to reach.
    if (!Number.isFinite(state.llmPopoutX) || !Number.isFinite(state.llmPopoutY)) {
      const panel = inspector.getBoundingClientRect();
      state.llmPopoutX = Math.max(8, panel.left - (state.llmPopoutWidth || 460) - 10);
      state.llmPopoutY = Math.max(8, panel.top + 8);
    }
    win.style.width = `${Math.max(MIN_WIDTH, state.llmPopoutWidth || 460)}px`;
    win.style.height = `${Math.max(MIN_HEIGHT, state.llmPopoutHeight || 460)}px`;
    const spot = clampToScreen();
    win.style.left = `${spot.x}px`;
    win.style.top = `${spot.y}px`;
  }

  function rememberPlacement() {
    const rect = win.getBoundingClientRect();
    state.llmPopoutX = Math.round(rect.left);
    state.llmPopoutY = Math.round(rect.top);
    state.llmPopoutWidth = Math.round(rect.width);
    state.llmPopoutHeight = Math.round(rect.height);
    state.onLayoutChanged?.();
  }

  header.addEventListener("pointerdown", (event) => {
    if (event.target === closeButton || closeButton.contains(event.target)) return;
    event.preventDefault();
    header.setPointerCapture?.(event.pointerId);
    const startX = event.clientX;
    const startY = event.clientY;
    const startLeft = win.offsetLeft;
    const startTop = win.offsetTop;
    const move = (moveEvent) => {
      win.style.left = `${Math.max(8, Math.min(window.innerWidth - 80, startLeft + moveEvent.clientX - startX))}px`;
      win.style.top = `${Math.max(8, Math.min(window.innerHeight - 40, startTop + moveEvent.clientY - startY))}px`;
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      rememberPlacement();
      autoSaveSessionQuiet("LLM Prompting window moved");
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });

  // The corner resize handle is the browser's own. Save the new size when it settles.
  let resizeTimer = null;
  const resizeObserver = new ResizeObserver(() => {
    if (win.style.display === "none") return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      rememberPlacement();
      autoSaveSessionQuiet("LLM Prompting window resized");
    }, 400);
  });
  resizeObserver.observe(win);

  // ---- mirroring: the real fields are the source of truth. Panel code sets their values directly, which fires no
  // events, so the window compares on a short timer while it is open.
  let timer = null;

  function mirrorFromPanel() {
    if (!win.isConnected) {
      stopTimer();
      return;
    }
    if (document.activeElement !== promptMirror && promptMirror.value !== prompt.value) promptMirror.value = prompt.value;
    if (document.activeElement !== pass2Mirror && pass2Mirror.value !== pass2Prompt.value) pass2Mirror.value = pass2Prompt.value;
    if (statusMirror.textContent !== status.textContent) statusMirror.textContent = status.textContent;
    if (statusMirror.style.cssText !== status.style.cssText) statusMirror.style.cssText = `${status.style.cssText};flex:0 0 auto;`;
    statusMirror.style.display = status.style.display === "none" || !status.textContent ? "none" : "";
    if (saveMirror.textContent !== saveButton.textContent) saveMirror.textContent = saveButton.textContent;
    if (saveMirror.disabled !== saveButton.disabled) saveMirror.disabled = saveButton.disabled;
    saveMirror.style.opacity = saveButton.style.opacity;
    saveMirror.style.cursor = saveButton.style.cursor;
    const wantPass2 = pass2Field.style.display !== "none";
    pass2Wrap.style.display = wantPass2 ? "flex" : "none";
    updateSceneLabel();
  }

  function startTimer() {
    if (timer) return;
    mirrorFromPanel();
    timer = setInterval(mirrorFromPanel, SYNC_MS);
  }

  function stopTimer() {
    clearInterval(timer);
    timer = null;
  }

  function updateSceneLabel() {
    const segment = activeSegment();
    const text = segment ? sceneDisplayName(segment, segmentIndexInfo(segment).index) : "No scene selected";
    if (sceneLabel.textContent !== text) sceneLabel.textContent = text;
  }

  // Brings the window in line with state.llmPopoutOpen and the project's video engine.
  function sync() {
    const open = Boolean(state.llmPopoutOpen);
    const miniMax = isMiniMaxProject();
    toggle.input.checked = open;
    win.style.display = open ? "flex" : "none";
    notMiniMaxNote.style.display = open && !miniMax ? "block" : "none";
    content.style.display = miniMax ? "flex" : "none";
    if (open) {
      place();
      startTimer();
    } else {
      stopTimer();
    }
  }

  function setOpen(open) {
    state.llmPopoutOpen = Boolean(open);
    sync();
    state.onLayoutChanged?.();
    autoSaveSessionQuiet("LLM Prompting window toggled");
  }

  toggle.input.addEventListener("change", () => setOpen(toggle.input.checked));
  closeButton.onclick = () => setOpen(false);
  const onWindowResize = () => {
    if (!win.isConnected) {
      window.removeEventListener("resize", onWindowResize);
      return;
    }
    if (win.style.display !== "none") place();
  };
  window.addEventListener("resize", onWindowResize);

  // The hooks other modules call go live only once the Builder is fully built.
  function activate() {
    state.syncLlmPopout = sync;
    sync();
  }
  return { element: win, activate };
}
