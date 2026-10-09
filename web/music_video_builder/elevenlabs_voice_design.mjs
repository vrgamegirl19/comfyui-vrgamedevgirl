import { postJson } from "./comfy_api.mjs";
import { makeButton, makeField, makeInput, makeSelect, normalizeVideoType, toast } from "./controls.mjs";

export function normalizeElevenLabsDesignDraft(value = {}) {
  const draft = value && typeof value === "object" ? value : {};
  const result = {};
  for (const [key, limit] of Object.entries({ user_input: 4000, voice_description: 1000, preview_text: 1000, voice_name: 256 })) {
    result[key] = String(draft[key] || "").slice(0, limit);
  }
  result.model_id = ["eleven_ttv_v3", "eleven_multilingual_ttv_v2"].includes(draft.model_id) ? draft.model_id : "eleven_ttv_v3";
  return result;
}

export function openElevenLabsVoiceDesign({ state, subject, host, isCurrent, getLlmPayload, onAssigned }) {
  if (!isCurrent() || normalizeVideoType(state.videoType) !== "speaking") return;
  const previous = host.querySelector?.("[data-voice-design]");
  if (previous) return;
  const panel = document.createElement("div");
  panel.dataset.voiceDesign = "true";
  panel.style.cssText = "border:1px solid #52525b;border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:10px;background:#18181b;";
  const heading = document.createElement("strong");
  heading.textContent = `Design Voice — ${subject.name || "Character"}`;
  const draft = normalizeElevenLabsDesignDraft(subject.elevenlabs_voice_design);
  const area = (value, max, placeholder) => {
    const input = document.createElement("textarea");
    input.value = value; input.maxLength = max; input.rows = 3; input.placeholder = placeholder;
    input.style.cssText = "width:100%;box-sizing:border-box;resize:vertical;";
    return input;
  };
  const brief = area(draft.user_input, 4000, "Describe the voice you imagine: age, accent, tone, personality…");
  const description = area(draft.voice_description, 1000, "Review or write the voice description sent to ElevenLabs (20–1000 characters).");
  const previewText = area(draft.preview_text, 1000, "Optional: 100–1000 characters. Leave blank for ElevenLabs to write a sample.");
  const name = makeInput(draft.voice_name || subject.name || ""); name.maxLength = 256;
  const model = makeSelect([{ value: "eleven_ttv_v3", label: "Voice Design v3" }, { value: "eleven_multilingual_ttv_v2", label: "Voice Design v2" }]); model.value = draft.model_id;
  const help = makeButton("Help Write Voice Description");
  const generate = makeButton("Generate Voice Previews", "primary");
  const save = makeButton("Save Voice to ElevenLabs & Assign", "primary");
  const close = makeButton("Close Voice Design");
  const status = document.createElement("div"); status.style.cssText = "font-size:12px;line-height:1.5;color:#a1a1aa;";
  const note = document.createElement("div");
  note.textContent = "The optional helper uses your selected LLM Runner and character description. Review the result before generating previews. ElevenLabs generation uses account credits. Save creates a voice in your ElevenLabs account; save Reference Builder to keep its character assignment.";
  note.style.cssText = status.style.cssText;
  const previews = document.createElement("div"); previews.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  const actions = document.createElement("div"); actions.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;"; actions.append(generate, save, close);
  panel.append(heading, makeField("What should this character sound like?", brief), help,
    makeField("Voice description", description), makeField("Design model", model),
    makeField("Preview dialogue (optional)", previewText), makeField("Voice name", name), note, status, previews, actions);
  host.append(panel);
  let busy = false, selected = "", snapshot = "", candidates = [], version = 0;
  const current = () => isCurrent() && panel.isConnected;
  const signature = () => JSON.stringify([description.value.trim(), previewText.value.trim(), model.value]);
  const stop = () => previews.querySelectorAll("audio").forEach(audio => { audio.pause(); audio.removeAttribute("src"); audio.load(); });
  const controls = () => {
    help.disabled = busy || !getLlmPayload;
    generate.disabled = busy;
    save.disabled = busy || !selected || snapshot !== signature();
    // Avoid changing a provider request's inputs while it is in flight.
    for (const input of [brief, description, previewText, name, model]) input.disabled = busy;
  };
  const remember = () => {
    subject.elevenlabs_voice_design = normalizeElevenLabsDesignDraft({ user_input: brief.value, voice_description: description.value, preview_text: previewText.value, voice_name: name.value, model_id: model.value });
  };
  const invalidate = () => { version++; selected = ""; candidates = []; snapshot = ""; stop(); previews.replaceChildren(); controls(); };
  for (const input of [brief, description, previewText, name, model]) {
    input.oninput = () => { if (!current() || busy) return; remember(); if ([description, previewText, model].includes(input)) invalidate(); };
  }
  model.onchange = model.oninput;
  close.onclick = () => { version++; stop(); panel.remove(); };
  const run = async (operation) => {
    if (!current() || busy) return;
    const key = String(state.elevenLabsApiKey || "").trim();
    const requestVersion = version;
    busy = true; controls(); status.textContent = "Working…";
    const active = () => current() && requestVersion === version && key === String(state.elevenLabsApiKey || "").trim();
    try { await operation(key, active); }
    catch (error) { if (current()) { status.textContent = String(error?.message || error); toast(status.textContent, true); } }
    finally { busy = false; if (current()) controls(); }
  };
  help.onclick = () => run(async (_key, active) => {
    if (!brief.value.trim()) throw new Error("Describe the voice you want first.");
    const data = await postJson("/vrgdg/music_builder/elevenlabs_voice_description", {
      ...getLlmPayload(), video_type: "speaking", user_input: brief.value,
      character_name: subject.name || "", character_description: subject.description || "",
      unload_after: false, clear_before_load: false,
    }, 600000);
    if (!active()) return;
    description.value = data.voice_description; remember(); invalidate();
    status.textContent = "Description ready. Review and edit it before generating previews.";
  });
  generate.onclick = () => run(async (key, active) => {
    if (!key) throw new Error("Enter your ElevenLabs API key in Builder Settings → ElevenLabs.");
    const data = await postJson("/vrgdg/music_builder/elevenlabs_voice_design", {
      video_type: "speaking", api_key: key, voice_description: description.value, text: previewText.value, model_id: model.value,
    }, 150000);
    if (!active()) return;
    invalidate(); snapshot = signature(); candidates = data.previews;
    remember();
    candidates.forEach((candidate, index) => {
      const row = document.createElement("div");
      const choose = makeButton(`Choose Preview ${index + 1}`);
      const audio = document.createElement("audio"); audio.controls = true; audio.preload = "none";
      audio.style.cssText = "width:100%;height:36px;";
      audio.src = `data:audio/mpeg;base64,${candidate.audio_base_64}`;
      choose.onclick = () => {
        if (!current() || busy) return;
        selected = candidate.generated_voice_id;
        status.textContent = `Preview ${index + 1} selected. Set a name, then save and assign it.`; controls();
      };
      row.append(choose, audio); previews.append(row);
    });
    status.textContent = "Listen to the previews and choose one.";
  });
  save.onclick = () => run(async (key, active) => {
    if (!selected || snapshot !== signature()) throw new Error("Generate and choose a preview first.");
    if (!name.value.trim()) throw new Error("Give this voice a name before saving.");
    const data = await postJson("/vrgdg/music_builder/elevenlabs_voice_create", {
      video_type: "speaking", api_key: key, voice_name: name.value, voice_description: description.value, generated_voice_id: selected,
    }, 150000);
    // The account mutation succeeded. Consume the preview even if the editor changed.
    const assignHere = active();
    invalidate();
    if (!assignHere) { if (current()) status.textContent = "Voice created in ElevenLabs. Refresh account voices to assign it."; return; }
    remember(); onAssigned(data.voice);
    status.textContent = "Voice saved to ElevenLabs and assigned. Save Reference Builder to keep the assignment.";
  });
  remember(); controls();
  return panel;
}
