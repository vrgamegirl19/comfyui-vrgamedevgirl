import { postJson } from "./comfy_api.mjs";
import { makeButton, makeCheckbox, makeField, makeSelect } from "./controls.mjs";

export function normalizeSceneDialogue(value = {}) {
  const draft = value && typeof value === "object" ? value : {};
  const model = ["eleven_v4", "eleven_v3"].includes(draft.model_id) ? draft.model_id : "eleven_v4";
  return { speaker_id: String(draft.speaker_id || "").slice(0, 256),
    text: String(draft.text || "").slice(0, model === "eleven_v3" ? 5000 : 10000),
    delivery: String(draft.delivery || "").slice(0, 4000),
    allow_rewrite: draft.allow_rewrite === true, model_id: model };
}

export function createSceneDialoguePanel({ state, scene, isCurrent, projectFolder, sceneSlotNumber,
  getLlmPayload, openInstructions, task, onAttachment, onDraftChanged }) {
  const panel = document.createElement("div");
  panel.style.cssText = "border:1px solid #155e75;border-radius:7px;padding:12px;display:flex;flex-direction:column;gap:9px;";
  const title = document.createElement("strong"); title.textContent = "ElevenLabs Scene Dialogue";
  const original = normalizeSceneDialogue(scene.scene_dialogue);
  const characters = () => (state.fluxReferenceBuilder?.subjects || []).filter(subject =>
    (subject.reference_type || "character") === "character" && !subject.extra_reference_for
    && subject.elevenlabs_voice?.enabled && subject.elevenlabs_voice.voice_id);
  const speaker = makeSelect([{ value: "", label: "Choose a character…" }, ...characters().map(subject => ({
    value: subject.id, label: `${subject.name || "Character"} — ${subject.elevenlabs_voice.name || subject.elevenlabs_voice.voice_id}`,
  }))], original.speaker_id);
  const model = makeSelect([{ value: "eleven_v4", label: "Eleven v4" }, { value: "eleven_v3", label: "Eleven v3" }], original.model_id);
  const textarea = (value, max, placeholder) => {
    const field = document.createElement("textarea"); field.value = value; field.maxLength = max; field.rows = 3;
    field.placeholder = placeholder; field.style.cssText = "width:100%;box-sizing:border-box;resize:vertical;"; return field;
  };
  const text = textarea(original.text, 10000, "Words spoken by this character. You can add tags such as [whispers] directly.");
  const delivery = textarea(original.delivery, 4000, "How should they perform this line? Quiet, worried, sarcastic, relieved…");
  const rewrite = makeCheckbox("Allow the LLM to rewrite the dialogue", original.allow_rewrite);
  const help = makeButton("Help Craft Dialogue");
  const instructions = makeButton("Edit Dialogue LLM Instructions…");
  const generate = makeButton("Generate Speech", "primary");
  const use = makeButton("Use Generated Speech", "primary");
  const player = document.createElement("audio"); player.controls = true; player.preload = "none"; player.hidden = true; player.style.width = "100%";
  const status = document.createElement("div"); status.style.cssText = "font-size:12px;color:#a1a1aa;line-height:1.5;";
  status.textContent = characters().length ? "Review tagged dialogue before generating. Generation uses ElevenLabs credits. Use the take, then save this dialog to attach it and fit the scene." : "Assign and enable character voices in Reference Builder, save it, then reopen Audio Settings.";
  const actions = document.createElement("div"); actions.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;";
  actions.append(help, instructions, generate, use);
  panel.append(title, makeField("Speaker / assigned voice", speaker), makeField("Speech model", model),
    makeField("Dialogue sent to ElevenLabs", text), makeField("Delivery instructions for the LLM", delivery), rewrite.wrapper, actions, status, player);
  if (state.projectVideoEngine === "minimax_h3") {
    const renderNote = document.createElement("div");
    renderNote.style.cssText = status.style.cssText;
    renderNote.textContent = "For MiniMax video generation, choose Input Audio in video settings to use this speech take.";
    panel.append(renderNote);
  }
  let busy = false, take = null;
  const draft = () => ({ speaker_id: speaker.value, text: text.value, delivery: delivery.value,
    allow_rewrite: rewrite.input.checked, model_id: model.value });
  const character = () => characters().find(subject => subject.id === speaker.value);
  const signature = () => JSON.stringify([draft(), character()?.elevenlabs_voice?.voice_id, state.elevenLabsApiKey]);
  const current = () => isCurrent() && panel.isConnected && state.videoType === "speaking" && state.segments.includes(scene);
  const controls = () => {
    for (const field of [speaker, model, text, delivery, rewrite.input]) field.disabled = busy;
    text.maxLength = model.value === "eleven_v3" ? 5000 : 10000;
    help.disabled = busy || !getLlmPayload || !character();
    instructions.disabled = busy || !openInstructions;
    generate.disabled = busy || !character();
    use.disabled = busy || !take || take.signature !== signature();
  };
  const stop = () => { player.pause(); player.removeAttribute("src"); player.load(); player.hidden = true; };
  const invalidate = () => { take = null; stop(); controls(); };
  for (const field of [speaker, model, text, delivery, rewrite.input]) {
    const change = () => { if (!current() || busy) return; invalidate(); onDraftChanged?.(draft()); status.textContent = "Dialogue changed. Generate a new take to use these edits."; };
    field.oninput = change; field.onchange = change;
  }
  instructions.onclick = () => { if (current()) return openInstructions("elevenlabs_dialogue", scene); };
  help.onclick = () => task(async () => {
    const selected = character(); if (!selected) throw new Error("Choose a character with an assigned voice.");
    const before = signature();
    status.textContent = "Preparing dialogue with the selected LLM Runner…";
    const data = await postJson("/vrgdg/music_builder/craft_scene_dialogue", {
      ...getLlmPayload(), video_type: "speaking", project_folder: projectFolder(), scene_id: scene.id,
      dialogue: draft(), character_name: selected.name || "", character_description: selected.description || "",
      unload_after: false, clear_before_load: false,
    }, 600000);
    if (!current() || signature() !== before) return;
    text.value = data.dialogue.text; invalidate(); onDraftChanged?.(draft());
    status.textContent = "Dialogue prepared. Review the words and tags before generating speech.";
  });
  generate.onclick = () => task(async () => {
    if (!String(state.elevenLabsApiKey || "").trim()) throw new Error("Enter your ElevenLabs API key in Builder Settings → ElevenLabs.");
    const before = signature();
    status.textContent = "Generating speech with ElevenLabs…";
    const data = await postJson("/vrgdg/music_builder/generate_scene_speech", {
      video_type: "speaking", api_key: state.elevenLabsApiKey, dialogue: draft(),
      references: { subjects: characters().map(subject => ({ id: subject.id, name: subject.name,
        reference_type: "character", elevenlabs_voice: subject.elevenlabs_voice })) },
    }, 210000);
    if (!current() || signature() !== before) return;
    stop(); take = { ...data, signature: before }; player.src = take.audio_data; player.hidden = false; controls();
    status.textContent = "Speech ready. Listen, then click Use Generated Speech to stage it for this scene.";
  });
  use.onclick = () => task(async () => {
    if (!take || take.signature !== signature()) throw new Error("Generate speech for the current dialogue first.");
    const chosen = take;
    const data = await postJson("/vrgdg/music_builder/save_scene_audio", {
      project_folder: projectFolder(), scene_number: sceneSlotNumber(scene), preserve_source: true,
      audio_data: chosen.audio_data, audio_name: chosen.audio_name,
    }, 180000);
    if (!current() || chosen.signature !== signature()) return;
    if (!(Number(data.duration) > 0)) throw new Error("Generated speech has no usable audio duration.");
    onAttachment({ ...data, audio_name: chosen.audio_name, dialogue: chosen.dialogue });
    take = null; controls();
    status.textContent = `${Number(data.duration).toFixed(2)} seconds staged for this scene. Save Scene Audio Settings to attach it and fit/ripple the timeline.`;
  });
  controls();
  return { panel, getDraft: draft, setBusy: value => { busy = value; controls(); }, dispose: stop };
}
