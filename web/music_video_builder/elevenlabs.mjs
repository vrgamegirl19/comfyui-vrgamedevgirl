import { postJson } from "./comfy_api.mjs";
import { openElevenLabsVoiceDesign } from "./elevenlabs_voice_design.mjs";
import { makeButton, makeCheckbox, makeField, makeInput, makeSelect, makeSettingsSection, normalizeVideoType, toast } from "./controls.mjs";

const openPanels = new WeakMap();

export function refreshElevenLabsUI(state) {
  const panels = openPanels.get(state);
  if (!panels) return;
  for (const item of panels) {
    if (!item.panel.isConnected) { panels.delete(item); continue; }
    const visible = isElevenLabsSpeaking(state) && item.folder === state.projectFolder;
    item.panel.style.display = visible ? item.display : "none";
    if (!visible) item.panel.querySelectorAll("audio").forEach(player => { player.pause(); player.removeAttribute("src"); player.load(); });
  }
}

function trackPanel(state, panel, display = "") {
  refreshElevenLabsUI(state);
  if (!openPanels.has(state)) openPanels.set(state, new Set());
  openPanels.get(state).add({ panel, folder: state.projectFolder, display });
}

export function normalizeElevenLabsVoice(value = {}) {
  const voice = value && typeof value === "object" ? value : {};
  return {
    enabled: voice.enabled === true,
    voice_id: String(voice.voice_id || "").trim(),
    name: String(voice.name || "").trim(),
  };
}

export function isElevenLabsSpeaking(state) {
  return normalizeVideoType(state.videoType) === "speaking";
}

export async function fetchElevenLabsVoices(state, nextPageToken = "") {
  if (!isElevenLabsSpeaking(state)) throw new Error("ElevenLabs is available only in Speaking video mode.");
  const apiKey = String(state.elevenLabsApiKey || "").trim();
  if (!apiKey) throw new Error("Enter your ElevenLabs API key in Builder Settings → ElevenLabs.");
  return postJson("/vrgdg/music_builder/elevenlabs_voices", {
    video_type: "speaking", api_key: apiKey, next_page_token: nextPageToken,
  }, 35000);
}

export function makeElevenLabsSettings({ state, projectInput, saveSession }) {
  if (!isElevenLabsSpeaking(state)) return null;
  const key = makeInput(state.elevenLabsApiKey || "", "password");
  key.autocomplete = "off";
  key.placeholder = "ElevenLabs API key";
  const test = makeButton("Test Connection");
  const save = makeButton("Save API Key to Project", "primary");
  const clear = makeButton("Remove Saved API Key");
  const status = document.createElement("div");
  status.style.cssText = "font-size:12px;color:#a1a1aa;line-height:1.45;";
  const updateStatus = () => {
    status.textContent = state.elevenLabsApiKeyProject
      ? "A key is saved in this project. Changing the field affects this session until you save it again."
      : "The key is used for this session. Save it to restore it when you reopen this project.";
    clear.disabled = !state.elevenLabsApiKeyProject;
  };
  const note = document.createElement("div");
  note.textContent = "Choose character voices in Reference Builder. Saved keys are included in project files, like LLM runner keys.";
  note.style.cssText = status.style.cssText;
  const actions = document.createElement("div");
  actions.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;";
  actions.append(test, save, clear);
  const panel = makeSettingsSection("ElevenLabs", [makeField("API key", key), actions, status, note], true);
  const folder = () => String(projectInput.value || state.projectFolder || "").trim();
  const openedFolder = folder();
  const current = () => panel.isConnected && isElevenLabsSpeaking(state) && folder() === openedFolder;
  key.oninput = () => { if (current()) state.elevenLabsApiKey = key.value.trim(); };
  test.onclick = async () => {
    if (!current()) return;
    test.disabled = true;
    state.elevenLabsApiKey = key.value.trim();
    const testedKey = state.elevenLabsApiKey;
    try {
      await fetchElevenLabsVoices(state);
      if (current() && testedKey === state.elevenLabsApiKey) toast("Connected to ElevenLabs. Voice access is available.");
    } catch (error) { if (current()) toast(String(error?.message || error), true); }
    finally { test.disabled = false; }
  };
  const persist = async (value) => {
    if (!current()) return;
    if (!folder()) { toast("Create or load a project before saving an API key.", true); return; }
    const old = state.elevenLabsApiKeyProject || "";
    save.disabled = clear.disabled = true;
    if (value) state.elevenLabsApiKey = value;
    state.elevenLabsApiKeyProject = value;
    try {
      const result = await saveSession({ quiet: true, throwOnError: true });
      if (result?.stale) throw new Error("Project was modified elsewhere. Reload it before saving the API key.");
      if (current()) {
        if (!value) { state.elevenLabsApiKey = ""; key.value = ""; }
        toast(value ? "ElevenLabs API key saved to this project." : "Saved ElevenLabs API key removed.");
      }
    } catch (error) {
      if (folder() === openedFolder) state.elevenLabsApiKeyProject = old;
      if (current()) toast(String(error?.message || error), true);
    } finally { save.disabled = false; if (current()) updateStatus(); }
  };
  save.onclick = () => {
    const value = key.value.trim();
    if (!value) { toast("Enter an API key before saving it to this project.", true); return; }
    return persist(value);
  };
  clear.onclick = () => persist("");
  updateStatus();
  trackPanel(state, panel);
  return panel;
}

export function makeElevenLabsVoicePicker({ state, subject, getLlmPayload }) {
  if (!isElevenLabsSpeaking(state) || subject.reference_type !== "character" || subject.extra_reference_for) return null;
  subject.elevenlabs_voice = normalizeElevenLabsVoice(subject.elevenlabs_voice);
  const panel = document.createElement("div");
  panel.style.cssText = "border:1px solid #155e75;border-radius:7px;padding:10px;display:flex;flex-direction:column;gap:8px;";
  const enabled = makeCheckbox("Use ElevenLabs voice", subject.elevenlabs_voice.enabled);
  const fields = document.createElement("div");
  fields.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  const select = makeSelect([]);
  const refresh = makeButton("Refresh Voices");
  const more = makeButton("Load More Voices");
  const player = document.createElement("audio");
  player.controls = true;
  player.preload = "none";
  player.style.cssText = "width:100%;height:36px;";
  const hint = document.createElement("div");
  hint.style.cssText = "font-size:12px;color:#a1a1aa;";
  hint.textContent = "Refresh to load your account voices. Save Reference Builder to keep the assignment.";
  const actions = document.createElement("div");
  actions.style.cssText = "display:flex;gap:8px;";
  actions.append(refresh, more);
  fields.append(makeField("Character voice", select), actions, player, hint);
  panel.append(enabled.wrapper, fields);
  let voices = [];
  let token = "";
  let requestNumber = 0;
  const openedFolder = state.projectFolder;
  const current = () => panel.isConnected && isElevenLabsSpeaking(state) && state.projectFolder === openedFolder;
  const stop = () => { player.pause(); player.removeAttribute("src"); player.load(); };
  const update = () => {
    const assignment = subject.elevenlabs_voice;
    select.replaceChildren();
    const options = [{ value: "", label: "Choose a voice…" }, ...voices.map(v => ({ value: v.voice_id, label: `${v.name}${v.category ? ` (${v.category})` : ""}` }))];
    if (assignment.voice_id && !voices.some(v => v.voice_id === assignment.voice_id)) {
      options.push({ value: assignment.voice_id, label: `${assignment.name || assignment.voice_id} (saved; refresh to verify)` });
    }
    for (const item of options) {
      const option = document.createElement("option");
      option.value = item.value; option.textContent = item.label; select.append(option);
    }
    select.value = assignment.voice_id;
    fields.style.display = enabled.input.checked ? "flex" : "none";
    more.hidden = !token;
    stop();
    const voice = voices.find(v => v.voice_id === assignment.voice_id);
    const url = voice?.preview_url || "";
    player.hidden = !url || !enabled.input.checked;
    if (!player.hidden) player.src = url;
  };
  enabled.input.onchange = () => {
    if (!current()) return;
    // An enabled draft without a selection is allowed while editing; Save validates it.
    subject.elevenlabs_voice.enabled = enabled.input.checked;
    update();
  };
  select.onchange = () => {
    if (!current()) return;
    const voice = voices.find(v => v.voice_id === select.value);
    subject.elevenlabs_voice = {
      enabled: enabled.input.checked, voice_id: select.value,
      name: voice?.name || (subject.elevenlabs_voice.voice_id === select.value ? subject.elevenlabs_voice.name : ""),
    };
    update();
    hint.textContent = voice?.preview_url ? "Play the existing voice preview below." : "This voice has no preview available.";
  };
  const load = async (append) => {
    if (!current()) return;
    const number = ++requestNumber;
    const key = state.elevenLabsApiKey;
    refresh.disabled = more.disabled = true;
    try {
      const data = await fetchElevenLabsVoices(state, append ? token : "");
      if (!current() || number !== requestNumber || key !== state.elevenLabsApiKey) return;
      voices = [...new Map([...(append ? voices : []), ...data.voices].map(v => [v.voice_id, v])).values()];
      token = data.has_more ? data.next_page_token : "";
      update();
      const assigned = subject.elevenlabs_voice.voice_id;
      hint.textContent = assigned && !voices.some(v => v.voice_id === assigned)
        ? (token ? "Load more voices to find the saved voice." : "Saved voice is unavailable to this key. Choose another voice or disable it.")
        : `${voices.length} voices loaded.${token ? " More voices are available." : ""}`;
    } catch (error) { if (current()) { hint.textContent = String(error?.message || error); toast(hint.textContent, true); } }
    finally { refresh.disabled = more.disabled = false; }
  };
  refresh.onclick = () => load(false);
  more.onclick = () => load(true);
  const design = makeButton("Design Voice");
  design.onclick = () => openElevenLabsVoiceDesign({ state, subject, host: panel, isCurrent: current, getLlmPayload,
    onAssigned: voice => {
      voices = [...voices.filter(item => item.voice_id !== voice.voice_id), voice];
      subject.elevenlabs_voice = { enabled: true, voice_id: voice.voice_id, name: voice.name };
      enabled.input.checked = true;
      update();
      hint.textContent = "Designed voice assigned. Save Reference Builder to keep it.";
    },
  });
  panel.append(design);
  update();
  trackPanel(state, panel, "flex");
  return panel;
}
