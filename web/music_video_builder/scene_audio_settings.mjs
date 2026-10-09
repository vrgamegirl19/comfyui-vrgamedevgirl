import { audioUrl, postJson } from "./comfy_api.mjs";
import { makeButton, makeField, makeInput } from "./controls.mjs";
import { audioClipsForState } from "./audio_clip_editor.mjs";
import { createSceneDialoguePanel } from "./scene_dialogue.mjs";

export const SPEAKING_AUDIO_DEFAULTS = Object.freeze({ silence_before: 0, silence_after: 0, fit_duration: true });

const SCENE_KEYS = ["id", "label", "start", "end", "video_path", "video_output", "video_status",
  "scene_audio_settings", "scene_dialogue", "scene_audio_render_dirty", "custom_audio_path", "custom_audio_name",
  "custom_audio_duration", "custom_audio_full_duration", "custom_audio_timeline_start",
  "custom_audio_source_start", "custom_audio_peaks", "custom_audio_beats"];

export function sceneAudioSnapshot(state) {
  return {
    video_type: state.videoType, timing_frozen: state.timingFrozen,
    speaking_audio_defaults: state.speakingAudioDefaults || {},
    segments: state.segments.map(scene => Object.fromEntries(SCENE_KEYS.filter(key => key in scene).map(key => [key, scene[key]]))),
    audio_clips: audioClipsForState(state),
    flux_reference_builder: { subjects: (state.fluxReferenceBuilder?.subjects || []).map(subject => ({
      id: subject.id, name: subject.name, reference_type: subject.reference_type,
      extra_reference_for: subject.extra_reference_for, elevenlabs_voice: subject.elevenlabs_voice,
    })) },
  };
}

export function applySceneAudioResult(state, session) {
  // Keep runtime scene objects, images, prompts and selection references intact.
  for (const updated of session.segments) {
    const scene = state.segments.find(item => item.id === updated.id);
    if (scene) {
      for (const key of SCENE_KEYS) if (!(key in updated)) delete scene[key];
      Object.assign(scene, updated);
    }
  }
  state.audioClips = session.audio_clips;
  state.speakingAudioDefaults = session.speaking_audio_defaults || {};
  state.audioClipMixPath = "";
  state.audioClipMixKey = "";
  state.audioClipGenerationMixPath = "";
  state.audioClipGenerationMixKey = "";
}

export function canOpenSceneAudio(state, scene) {
  return state.videoType === "speaking" && Boolean(scene && state.segments.includes(scene));
}

export function createSceneAudioSettings({ state, projectInput, overlay, projectDefaultsAnchor,
  pushHistory, pauseTimelineForEditing, render, saveSession, sceneSlotNumber, getLlmPayload, openInstructions }) {
  let modal = null;
  let closeModal = null;
  let modalProject = "";
  const defaultsButton = makeButton("Speaking Audio Defaults…");
  defaultsButton.title = "Project silence and duration defaults for Speaking scenes";
  defaultsButton.style.display = "none";
  projectDefaultsAnchor.after(defaultsButton);

  function refresh() {
    defaultsButton.style.display = state.videoType === "speaking" ? "" : "none";
    if (modal && (state.videoType !== "speaking" || !overlay.isConnected || modalProject !== projectFolder())) closeModal?.();
  }
  function projectFolder() { return String(projectInput.value || state.projectFolder || "").trim(); }
  function assertEditable(scene = null) {
    if (state.videoType !== "speaking" || (scene && !state.segments.includes(scene))) throw new Error("Select a Speaking-mode base scene.");
    if (state.timingFrozen) throw new Error("Unfreeze timeline timing before editing scene audio.");
    if (state.segments.some(item => item.video_status === "running")) throw new Error("Wait for scene rendering to finish before editing audio.");
  }
  async function apply(settings, scene = null, attachment = undefined, dialogue = undefined) {
    assertEditable(scene);
    const folder = projectFolder();
    if (!folder) throw new Error("Set a project folder before saving audio settings.");
    const snapshot = sceneAudioSnapshot(state);
    const signature = JSON.stringify(snapshot);
    const data = await postJson("/vrgdg/music_builder/preview_scene_audio_settings", {
      session: snapshot, settings, scene_id: scene?.id ?? null, attachment, dialogue,
    });
    assertEditable(scene);
    if (folder !== projectFolder() || signature !== JSON.stringify(sceneAudioSnapshot(state))) {
      throw new Error("Project audio or timing changed while preparing this edit. Reopen Audio Settings and try again.");
    }
    pauseTimelineForEditing();
    pushHistory();
    applySceneAudioResult(state, data.session);
    render();
    try {
      const saved = await saveSession({ quiet: true, throwOnError: true });
      if (saved?.stale) throw new Error("Project changed elsewhere. Reload before saving scene audio.");
    } catch (error) {
      applySceneAudioResult(state, snapshot);
      render();
      throw error;
    }
  }

  function open(scene = null) {
    if (scene && !canOpenSceneAudio(state, scene)) return;
    if (state.videoType !== "speaking") return;
    closeModal?.();
    pauseTimelineForEditing();
    modalProject = projectFolder();
    let attachment;
    let busy = false;
    let dialoguePanel = null;
    const sceneId = scene?.id;
    const startingSignature = JSON.stringify(sceneAudioSnapshot(state));
    const defaults = { ...SPEAKING_AUDIO_DEFAULTS, ...state.speakingAudioDefaults };
    const saved = { use_project_defaults: true, ...SPEAKING_AUDIO_DEFAULTS, ...scene?.scene_audio_settings };
    const backdrop = document.createElement("div");
    modal = backdrop;
    backdrop.className = "vrgdg-scene-audio-dialog";
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.65);display:flex;align-items:center;justify-content:center;padding:20px;";
    const box = document.createElement("div");
    box.setAttribute("role", "dialog"); box.setAttribute("aria-modal", "true");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow:auto;background:#111827;border:1px solid #a78bfa;border-radius:8px;padding:18px;color:#f8fafc;display:flex;flex-direction:column;gap:12px;";
    const title = document.createElement("h2");
    title.id = "vrgdg-scene-audio-title";
    title.textContent = scene ? `${scene.label || "Scene"} — Audio Settings` : "Speaking Audio — Project Defaults";
    title.style.cssText = "margin:0;font-size:16px;color:#ddd6fe;";
    box.setAttribute("aria-labelledby", title.id);
    box.append(title);
    const inherit = document.createElement("input"); inherit.type = "checkbox"; inherit.checked = saved.use_project_defaults;
    const fit = document.createElement("input"); fit.type = "checkbox";
    fit.checked = scene ? saved.fit_duration : defaults.fit_duration;
    const before = makeInput(String(scene ? saved.silence_before : defaults.silence_before), "number");
    const after = makeInput(String(scene ? saved.silence_after : defaults.silence_after), "number");
    for (const input of [before, after]) { input.min = "0"; input.max = "60"; input.step = "0.1"; input.required = true; }
    if (scene) box.append(makeField("Use project audio defaults", inherit));
    box.append(makeField("Silence before dialogue (seconds)", before), makeField("Silence after dialogue (seconds)", after),
      makeField("Fit scene duration to audio + silence", fit));
    const info = document.createElement("div"); info.style.cssText = "font-size:12px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere;color:#d4d4d8;";
    const warning = document.createElement("div"); warning.style.cssText = "font-size:12px;color:#fcd34d;white-space:pre-wrap;";
    const player = document.createElement("audio"); player.controls = true; player.style.width = "100%";
    const preview = makeButton("Preview audio with silence");
    const load = makeButton("Load / Replace Audio…");
    const clear = makeButton("Remove Scene Dialogue");
    const file = document.createElement("input"); file.type = "file"; file.accept = "audio/*,.wav,.mp3,.flac,.m4a,.ogg"; file.hidden = true;
    const save = makeButton(scene ? "Save Scene Audio Settings" : "Apply Project Defaults", "primary");
    const cancel = makeButton("Cancel");
    const error = document.createElement("div"); error.style.cssText = "color:#fca5a5;font-size:12px;white-space:pre-wrap;";
    const note = document.createElement("div"); note.style.cssText = "color:#a1a1aa;font-size:12px;line-height:1.5;";
    note.textContent = scene
      ? "Settings apply to this scene. Fitting its duration moves later scenes and their attached audio. Independent music/effects stay fixed. Source files are preserved."
      : "Applies to scenes using project defaults. Their dialogue is repositioned and fitted durations ripple the timeline. Scenes with custom settings keep their own values.";
    box.append(info, warning);
    if (scene) box.append(load, clear, file, preview, player);
    if (scene) {
      dialoguePanel = createSceneDialoguePanel({ state, scene, getLlmPayload, openInstructions,
        projectFolder, sceneSlotNumber, task,
        isCurrent: () => backdrop.isConnected && modalProject === projectFolder(),
        onAttachment: data => { attachment = data; player.pause(); sync(); },
        onDraftChanged: draft => {
          if (attachment?.dialogue && JSON.stringify(attachment.dialogue) !== JSON.stringify(draft)) attachment = undefined;
          sync();
        },
      });
      box.append(dialoguePanel.panel);
    }
    box.append(note, error, save, cancel);
    backdrop.append(box); document.body.append(backdrop);
    const restoreFocus = document.activeElement;
    const close = () => {
      dialoguePanel?.dispose();
      player.pause(); player.removeAttribute("src"); player.load();
      backdrop.remove();
      document.removeEventListener("keydown", onKey, true);
      if (modal === backdrop) { modal = null; closeModal = null; }
      if (restoreFocus?.isConnected) restoreFocus.focus();
    };
    closeModal = close;
    function onKey(event) {
      if (backdrop.contains && !backdrop.contains(event.target)) return;
      if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); if (!busy) close(); }
      if (event.key === "Tab") {
        const controls = [...box.querySelectorAll("button,input,textarea,select,audio")].filter(item => !item.disabled && !item.hidden);
        const first = controls[0]; const last = controls.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    }
    document.addEventListener("keydown", onKey, true);
    cancel.onclick = () => { if (!busy) close(); };
    backdrop.onpointerdown = event => { if (event.target === backdrop && !busy) close(); };
    function settings() {
      return { ...(scene ? { use_project_defaults: inherit.checked } : {}),
        silence_before: Number(before.value), silence_after: Number(after.value), fit_duration: fit.checked };
    }
    function effective() { return scene && inherit.checked ? defaults : settings(); }
    function clips() {
      if (attachment !== undefined) return attachment?.saved_path ? [{ path: attachment.saved_path, name: attachment.audio_name,
        start: 0, source_start: 0, duration: attachment.duration, volume: 1 }] : [];
      return audioClipsForState(state).filter(clip => clip.scene_id === sceneId && (!clip.role || clip.role === "dialogue"));
    }
    function sync() {
      dialoguePanel?.setBusy(busy);
      const usingDefaults = scene && inherit.checked;
      before.disabled = after.disabled = fit.disabled = busy || Boolean(usingDefaults);
      inherit.disabled = busy;
      if (usingDefaults) { before.value = String(defaults.silence_before); after.value = String(defaults.silence_after); fit.checked = defaults.fit_duration; }
      load.disabled = clear.disabled = save.disabled = cancel.disabled = busy;
      const sources = clips();
      const span = sources.length ? Math.max(...sources.map(c => c.start + c.duration)) - Math.min(...sources.map(c => c.start)) : 0;
      const opts = effective();
      const total = span + opts.silence_before + opts.silence_after;
      info.textContent = scene ? (sources.length
        ? `${sources.map(c => c.name || c.path).join("\n")}\nDialogue: ${span.toFixed(2)}s · With silence: ${total.toFixed(2)}s\nCurrent scene: ${(scene.end - scene.start).toFixed(2)}s`
        : "No scene-owned dialogue. Load an audio file to fit this scene. Project-wide audio is separate.")
        : "New scenes inherit these settings. Both silence values default to 0 seconds.";
      preview.disabled = busy || !sources.length;
      warning.textContent = scene?.scene_audio_render_dirty ? "Audio changed: render this scene again to update its video." : "";
      if (scene && sources.length && !opts.fit_duration && total > scene.end - scene.start + 0.0001) {
        warning.textContent += "\nDialogue plus silence exceeds the fixed scene duration. Enable fitting or lengthen the scene before rendering.";
      }
      if (scene?.video_path && sources.length) warning.textContent += "\nChanging audio timing or replacing dialogue requires rendering this scene again.";
    }
    for (const control of [inherit, before, after, fit]) control.onchange = () => { player.pause(); sync(); };
    async function task(action) {
      error.textContent = ""; busy = true; sync();
      try { await action(); } catch (err) { error.textContent = String(err.message || err); }
      finally { busy = false; if (backdrop.isConnected) sync(); }
    }
    load.onclick = () => file.click();
    file.onchange = () => {
      const chosen = file.files?.[0]; file.value = ""; if (!chosen) return;
      task(async () => {
        assertEditable(scene);
        if (!modalProject) throw new Error("Set a project folder first.");
        const audioData = await new Promise((resolve, reject) => {
          const reader = new FileReader(); reader.onload = () => resolve(String(reader.result || ""));
          reader.onerror = () => reject(new Error("Could not read audio file.")); reader.readAsDataURL(chosen);
        });
        const data = await postJson("/vrgdg/music_builder/save_scene_audio", {
          project_folder: modalProject, scene_number: sceneSlotNumber(scene), audio_data: audioData,
          audio_name: chosen.name, preserve_source: true,
        }, 180000);
        if (!backdrop.isConnected || modalProject !== projectFolder()) return;
        if (!(Number(data.duration) > 0)) throw new Error("The imported file has no usable audio duration.");
        player.pause(); attachment = { ...data, audio_name: chosen.name };
      });
    };
    clear.onclick = () => { player.pause(); attachment = {}; sync(); };
    preview.onclick = () => task(async () => {
      const sources = clips(); const first = Math.min(...sources.map(c => c.start)); const opts = effective();
      const previewClips = sources.map(c => ({ ...c, start: c.start - first + opts.silence_before }));
      const duration = Math.max(...previewClips.map(c => c.start + c.duration)) + opts.silence_after;
      const data = await postJson("/vrgdg/music_builder/prepare_audio_clip_mix", {
        project_folder: modalProject, clips: previewClips, duration,
      }, 180000);
      if (!backdrop.isConnected || modalProject !== projectFolder()) return;
      player.src = audioUrl(data.audio_path); await player.play();
    });
    save.onclick = () => task(async () => {
      if (startingSignature !== JSON.stringify(sceneAudioSnapshot(state)) || modalProject !== projectFolder()) {
        throw new Error("Audio or timing changed since this window opened. Close and reopen it before saving.");
      }
      if (!box.querySelectorAll("input:invalid").length) {
        await apply(settings(), scene, attachment, dialoguePanel?.getDraft()); close();
      } else throw new Error("Enter silence values between 0 and 60 seconds.");
    });
    sync(); cancel.focus();
  }
  defaultsButton.onclick = () => open();
  return {
    openSceneAudioSettings: scene => open(scene), refresh,
    dispose: () => { closeModal?.(); defaultsButton.remove(); },
  };
}
