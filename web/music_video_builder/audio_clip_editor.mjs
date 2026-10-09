import { postJson } from "./comfy_api.mjs";
import { makeButton, toast } from "./controls.mjs";
import { TIMELINE_SCENE_AUDIO_TOP, TIMELINE_SCENE_AUDIO_HEIGHT } from "./constants.mjs";

export function speakingAudioEditsActive(state) {
  return state.videoType === "speaking" && Array.isArray(state.audioClips);
}

export function speakingAudioLaneCount(state) {
  if (state.videoType !== "speaking") return 1;
  return Math.max(1, ...audioClipsForState(state).map(clip => Math.max(0, Number(clip.lane || 0)) + 1));
}

export function audioClipsForState(state) {
  if (Array.isArray(state.audioClips)) return state.audioClips;
  const clips = (state.segments || []).filter(scene => scene.custom_audio_path).map(scene => ({
    id: `audio_${scene.id}`, scene_id: scene.id, path: scene.custom_audio_path,
    name: scene.custom_audio_name || scene.label || "Audio",
    start: Number(scene.custom_audio_timeline_start ?? scene.start ?? 0),
    source_start: Number(scene.custom_audio_source_start || 0),
    duration: Number(scene.custom_audio_duration || (scene.end - scene.start)),
    full_duration: Number(scene.custom_audio_full_duration || scene.custom_audio_duration || (scene.end - scene.start)),
    peaks: [...(scene.custom_audio_peaks || [])],
    lane: 0, role: "dialogue", volume: 1, muted: false, include_in_generation: true,
  }));
  if (!clips.length && state.videoType === "speaking" && state.audioPath && Number(state.audioDuration) > 0) {
    clips.push({ id: "project_audio", scene_id: "", path: state.audioPath, name: "Project dialogue",
      start: 0, source_start: 0, duration: Number(state.audioDuration), full_duration: Number(state.audioDuration),
      peaks: [...(state.peaks || [])], lane: 0, role: "dialogue", volume: 1, muted: false,
      include_in_generation: true });
  }
  return clips;
}

export function splitAudioClip(clip, time) {
  const offset = Number(time) - clip.start;
  if (offset < 0.05 || clip.duration - offset < 0.05) throw new Error("Place the playhead inside this audio clip, away from its edges.");
  return [
    { ...clip, duration: offset },
    { ...clip, id: `audio_${crypto.randomUUID()}`, start: time,
      source_start: clip.source_start + offset, duration: clip.duration - offset },
  ];
}

export function audioEditDuration(state) {
  return Math.max(0.05, ...(state.segments || []).map(scene => Number(scene.end || 0)),
    ...audioClipsForState(state).map(clip => clip.start + clip.duration));
}

export async function prepareEditedAudio(state, projectFolder, options = {}) {
  if (!speakingAudioEditsActive(state)) return null;
  const clips = state.audioClips;
  const duration = audioEditDuration(state);
  const key = JSON.stringify([projectFolder, clips, duration]);
  const prefix = options.generation ? "audioClipGenerationMix" : "audioClipMix";
  if (state[`${prefix}Key`] === key && state[`${prefix}Path`]) return { audio_path: state[`${prefix}Path`], duration };
  const data = await postJson("/vrgdg/music_builder/prepare_audio_clip_mix", {
    project_folder: projectFolder, clips, duration, generation_only: Boolean(options.generation),
  }, 180000);
  // A newer edit or a project change must never be replaced by this result.
  if (!speakingAudioEditsActive(state) || state.audioClips !== clips || JSON.stringify([projectFolder, state.audioClips, audioEditDuration(state)]) !== key) throw new Error("Audio changed while preparing playback. Please try again.");
  state[`${prefix}Key`] = key;
  state[`${prefix}Path`] = data.audio_path;
  state[`${prefix}Peaks`] = data.peaks || [];
  return data;
}

export async function prepareEmbeddedAudioWithClips(state, projectFolder, scenes, paths) {
  if (!speakingAudioEditsActive(state)) return null;
  const background = state.audioClips.filter(clip => clip.role && clip.role !== "dialogue");
  if (!background.length) return null;
  const positions = scenes.map(scene => state.segments.findIndex(item => item.id === scene.id));
  if (positions.some((position, index) => index && position !== positions[index - 1] + 1)) {
    throw new Error("Choose consecutive scenes when stitching with background audio.");
  }
  const voices = scenes.map((scene, index) => ({ path: paths[index], start: Number(scene.start || 0),
    source_start: 0, duration: Number(scene.end) - Number(scene.start), volume: 1 }));
  return postJson("/vrgdg/music_builder/prepare_audio_clip_mix", {
    project_folder: projectFolder, clips: [...voices, ...background], duration: audioEditDuration(state),
  }, 180000);
}

export function createAudioClipEditor({ state, projectInput, currentGlobalTime, pushHistory,
  render, autoSaveSessionQuiet, pauseTimelineForEditing, setActiveSegment, addAudioClipButton, refreshAudioSettings }) {
  let selectedId = "";
  let importBox = null;
  let clipMenuBox = null;
  function invalidateMix() {
    state.sceneAudioGlobalTime = currentGlobalTime();
    state.audioClipMixPath = "";
    state.audioClipMixKey = "";
    state.audioClipMixPeaks = [];
    state.audioClipGenerationMixPath = "";
    state.audioClipGenerationMixKey = "";
  }
  function edit(action) {
    if (state.videoType !== "speaking") return;
    pauseTimelineForEditing();
    pushHistory();
    state.audioClips = audioClipsForState(state).map(clip => ({ ...clip }));
    action(state.audioClips);
    invalidateMix();
    render();
    autoSaveSessionQuiet("audio clips edited").catch(error => toast(String(error), true));
  }
  function registerSceneAudio(scene) {
    if (state.videoType !== "speaking") return;
    const clip = audioClipsForState({ segments: [scene] })[0];
    edit(clips => {
      const remaining = clips.filter(item => item.scene_id !== scene.id || (item.role && item.role !== "dialogue"));
      if (clip) remaining.push({ ...clip, id: `audio_${crypto.randomUUID()}` });
      state.audioClips = remaining;
    });
  }
  async function importAdditionalAudio(files, role = "music") {
    if (state.videoType !== "speaking") return;
    const project = projectInput.value || state.projectFolder;
    if (!project) throw new Error("Set a project folder before importing audio.");
    const start = Math.max(0, currentGlobalTime());
    const imported = [];
    for (const file of files) {
      const audioData = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("Failed to read the audio file."));
        reader.readAsDataURL(file);
      });
      const data = await postJson("/vrgdg/music_builder/save_scene_audio", {
        project_folder: project, scene_number: 1, preserve_source: true,
        audio_data: audioData, audio_name: file.name,
      }, 180000);
      if ((projectInput.value || state.projectFolder) !== project || state.videoType !== "speaking") return;
      if (!(Number(data.duration) > 0)) throw new Error("The imported audio has no usable duration.");
      imported.push({ id: `audio_${crypto.randomUUID()}`, scene_id: "", path: data.saved_path,
        name: file.name, start, source_start: 0, duration: Number(data.duration),
        full_duration: Number(data.duration), peaks: data.peaks || [], role,
        volume: role === "music" ? 0.25 : 1, muted: false, include_in_generation: role === "dialogue" });
    }
    if (!imported.length) return;
    edit(clips => {
      let lane = Math.max(0, ...clips.map(item => Number(item.lane || 0)));
      for (const clip of imported) clips.push({ ...clip, lane: ++lane });
    });
  }
  function openImport() {
    if (state.videoType !== "speaking") return;
    const box = document.createElement("div");
    importBox?.remove();
    importBox = box;
    box.style.cssText = "position:fixed;inset:20% 25%;z-index:100010;background:#111827;border:1px solid #a78bfa;border-radius:8px;padding:20px;display:flex;flex-direction:column;gap:12px;";
    const title = document.createElement("strong");
    title.textContent = "Add audio clips at the playhead";
    const role = document.createElement("select");
    for (const [value, label] of [["music", "Music / Score"], ["effect", "Sound effects"], ["dialogue", "Dialogue"]]) {
      const option = document.createElement("option");
      option.value = value; option.textContent = label; role.append(option);
    }
    role.value = "music";
    const input = document.createElement("input");
    input.type = "file"; input.accept = "audio/*"; input.multiple = true;
    const help = document.createElement("div");
    help.textContent = "Each file gets a separate lane. Music starts at 25% volume and joins the final mix; dialogue also feeds speaking generation.";
    const add = makeButton("Import audio", "primary");
    add.onclick = async () => {
      add.disabled = true;
      add.textContent = "Importing audio...";
      try { await importAdditionalAudio([...(input.files || [])], role.value); box.remove(); }
      catch (error) { toast(String(error.message || error), true); }
      finally { add.disabled = false; add.textContent = "Import audio"; }
    };
    const close = makeButton("Close"); close.onclick = () => box.remove();
    box.append(title, role, input, help, add, close); document.body.append(box);
  }
  if (addAudioClipButton) addAudioClipButton.onclick = openImport;
  function menu(event, clip) {
    if (state.videoType !== "speaking") return;
    event.preventDefault();
    event.stopPropagation();
    selectedId = clip.id;
    document.querySelector(".vrgdg-builder-context-menu")?.remove();
    const box = document.createElement("div");
    clipMenuBox = box;
    const project = projectInput.value;
    const liveClip = () => {
      const current = audioClipsForState(state).find(item => item.id === clip.id);
      if (state.videoType !== "speaking" || projectInput.value !== project || !current) throw new Error("This audio clip changed. Select it again.");
      return current;
    };
    box.className = "vrgdg-builder-context-menu";
    box.style.cssText = `position:fixed;z-index:100010;left:${Math.max(8, Math.min(event.clientX, window.innerWidth - 250))}px;top:${Math.max(8, Math.min(event.clientY, window.innerHeight - 390))}px;max-height:90vh;overflow:auto;display:flex;flex-direction:column;gap:4px;padding:8px;background:#111827;border:1px solid #a78bfa;border-radius:6px;`;
    const add = (label, action) => {
      const button = makeButton(label);
      button.onclick = () => {
        box.remove();
        try { action(); } catch (error) { toast(String(error.message || error), true); }
      };
      box.append(button);
    };
    add("Split audio at playhead", () => {
      const pieces = splitAudioClip(liveClip(), currentGlobalTime());
      edit(clips => clips.splice(clips.findIndex(item => item.id === clip.id), 1, ...pieces));
    });
    add("Move audio start to playhead", () => {
      liveClip();
      edit(clips => { clips.find(item => item.id === clip.id).start = Math.max(0, currentGlobalTime()); });
    });
    add("Delete audio piece", () => {
      liveClip();
      edit(clips => clips.splice(clips.findIndex(item => item.id === clip.id), 1));
    });
    const volumeLabel = document.createElement("label");
    const volume = document.createElement("input");
    volume.type = "range"; volume.min = "0"; volume.max = "200"; volume.step = "1";
    volume.value = String(Math.round(Number(clip.volume ?? 1) * 100));
    const caption = document.createElement("span"); caption.textContent = `Volume: ${volume.value}%`;
    volume.oninput = () => { caption.textContent = `Volume: ${volume.value}%`; };
    volume.onchange = () => {
      try { liveClip(); edit(clips => { clips.find(item => item.id === clip.id).volume = Number(volume.value) / 100; }); }
      catch (error) { toast(String(error.message || error), true); }
    };
    volumeLabel.append(caption, volume); box.append(volumeLabel);
    add(clip.muted ? "Unmute audio piece" : "Mute audio piece", () => {
      const muted = !liveClip().muted;
      edit(clips => { clips.find(item => item.id === clip.id).muted = muted; });
    });
    const includesGeneration = clip.include_in_generation ?? (!clip.role || clip.role === "dialogue");
    add(includesGeneration ? "Exclude from speaking generation" : "Include in speaking generation", () => {
      liveClip();
      edit(clips => { clips.find(item => item.id === clip.id).include_in_generation = !includesGeneration; });
    });
    add("Close", () => {});
    document.body.append(box);
    const close = event => {
      if (box.contains(event.target)) return;
      box.remove();
      window.removeEventListener("pointerdown", close, true);
    };
    window.addEventListener("pointerdown", close, true);
  }
  function drag(event, clip, mode) {
    if (state.videoType !== "speaking") return;
    if (event.button !== 0 || event.isPrimary === false) return;
    event.preventDefault();
    event.stopPropagation();
    pauseTimelineForEditing();
    selectedId = clip.id;
    const scene = state.segments.find(scene => scene.id === clip.scene_id);
    if (scene) setActiveSegment(scene);
    const x = event.clientX;
    const zoom = state.pxPerSecond;
    const project = projectInput.value;
    let moved = false;
    let live;
    const move = event => {
      if (state.videoType !== "speaking" || event.pointerId !== pointer || projectInput.value !== project) return;
      const delta = (event.clientX - x) / zoom;
      if (!moved && Math.abs(delta * zoom) < 3) return;
      if (!moved) {
        pushHistory();
        state.audioClips = audioClipsForState(state).map(item => ({ ...item }));
        live = state.audioClips.find(item => item.id === clip.id);
        moved = true;
      }
      if (!live) return;
      if (mode === "move") live.start = Math.max(0, clip.start + delta);
      else if (mode === "start") {
        const change = Math.max(-clip.source_start, -clip.start, Math.min(clip.duration - 0.05, delta));
        live.start = clip.start + change;
        live.source_start = clip.source_start + change;
        live.duration = clip.duration - change;
      } else live.duration = Math.max(0.05, Math.min(clip.full_duration - clip.source_start, clip.duration + delta));
      invalidateMix();
      render();
    };
    const pointer = event.pointerId;
    const finish = event => {
      if (event?.pointerId != null && event.pointerId !== pointer) return;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      window.removeEventListener("blur", finish);
      if (moved && projectInput.value === project) autoSaveSessionQuiet("audio piece moved or trimmed").catch(error => toast(String(error), true));
      else if (!moved && event?.type === "pointerup") menu(event, clip);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
    window.addEventListener("blur", finish);
  }
  function renderAudioClips(layer) {
    refreshAudioSettings?.();
    if (addAudioClipButton) addAudioClipButton.style.display = state.videoType === "speaking" ? "" : "none";
    if (state.videoType !== "speaking") {
      importBox?.remove(); clipMenuBox?.remove();
      return;
    }
    for (const clip of audioClipsForState(state)) {
      const width = Math.max(3, clip.duration * state.pxPerSecond);
      const block = document.createElement("div");
      block.title = `${clip.name}: drag to move, drag edges to trim, click or right-click to split at the playhead. Audio gaps are silent.`;
      const top = TIMELINE_SCENE_AUDIO_TOP + Number(clip.lane || 0) * (TIMELINE_SCENE_AUDIO_HEIGHT + 4);
      block.style.cssText = `position:absolute;left:${clip.start * state.pxPerSecond}px;top:${top}px;width:${width}px;height:${TIMELINE_SCENE_AUDIO_HEIGHT}px;pointer-events:auto;z-index:2;background:${clip.role === "music" ? "#164e63" : "#581c87"};opacity:${clip.muted ? 0.5 : 1};border:1px solid ${clip.id === selectedId ? "#fef08a" : "#c084fc"};border-radius:4px;cursor:grab;overflow:hidden;box-sizing:border-box;`;
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.floor(width));
      canvas.height = TIMELINE_SCENE_AUDIO_HEIGHT;
      canvas.style.cssText = "width:100%;height:100%;pointer-events:none;";
      const ctx = canvas.getContext("2d");
      ctx.strokeStyle = "#e9d5ff";
      ctx.beginPath();
      for (let x = 0; x < canvas.width; x++) {
        const sourceTime = clip.source_start + x / canvas.width * clip.duration;
        const peak = clip.peaks?.[Math.floor(sourceTime / Math.max(0.05, clip.full_duration) * clip.peaks.length)] || 0;
        const amplitude = peak * canvas.height / 2;
        ctx.moveTo(x, canvas.height / 2 - amplitude);
        ctx.lineTo(x, canvas.height / 2 + amplitude);
      }
      ctx.stroke();
      block.append(canvas);
      const label = document.createElement("span");
      label.textContent = `${clip.role || "dialogue"}: ${clip.name} · ${clip.muted ? "Muted" : `${Math.round(Number(clip.volume ?? 1) * 100)}%`}`;
      label.style.cssText = "position:absolute;left:10px;top:2px;right:8px;overflow:hidden;white-space:nowrap;font-size:10px;color:#faf5ff;text-shadow:0 1px 2px #000;pointer-events:none;";
      block.append(label);
      block.onpointerdown = event => drag(event, clip, "move");
      block.oncontextmenu = event => menu(event, clip);
      for (const [edge, mode] of [["left", "start"], ["right", "end"]]) {
        const handle = document.createElement("div");
        handle.style.cssText = `position:absolute;${edge}:0;top:0;bottom:0;width:7px;background:#c084fc66;cursor:ew-resize;`;
        handle.onpointerdown = event => drag(event, clip, mode);
        block.append(handle);
      }
      layer.append(block);
    }
  }
  return { renderAudioClips, registerSceneAudio, importAdditionalAudio };
}
