import { postJson } from "./comfy_api.mjs";
import { makeButton, toast } from "./controls.mjs";
import { TIMELINE_SCENE_AUDIO_TOP, TIMELINE_SCENE_AUDIO_HEIGHT } from "./constants.mjs";

export function audioClipsForState(state) {
  if (Array.isArray(state.audioClips)) return state.audioClips;
  return (state.segments || []).filter(scene => scene.custom_audio_path).map(scene => ({
    id: `audio_${scene.id}`, scene_id: scene.id, path: scene.custom_audio_path,
    name: scene.custom_audio_name || scene.label || "Audio",
    start: Number(scene.custom_audio_timeline_start ?? scene.start ?? 0),
    source_start: Number(scene.custom_audio_source_start || 0),
    duration: Number(scene.custom_audio_duration || (scene.end - scene.start)),
    full_duration: Number(scene.custom_audio_full_duration || scene.custom_audio_duration || (scene.end - scene.start)),
    peaks: [...(scene.custom_audio_peaks || [])],
  }));
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

export async function prepareEditedAudio(state, projectFolder) {
  if (!Array.isArray(state.audioClips)) return null;
  const clips = state.audioClips;
  const duration = audioEditDuration(state);
  const key = JSON.stringify([projectFolder, clips, duration]);
  if (state.audioClipMixKey === key && state.audioClipMixPath) return { audio_path: state.audioClipMixPath, duration };
  const data = await postJson("/vrgdg/music_builder/prepare_audio_clip_mix", {
    project_folder: projectFolder, clips, duration,
  }, 180000);
  // A newer edit or a project change must never be replaced by this result.
  if (state.audioClips !== clips || JSON.stringify([projectFolder, state.audioClips, audioEditDuration(state)]) !== key) throw new Error("Audio changed while preparing playback. Please try again.");
  state.audioClipMixKey = key;
  state.audioClipMixPath = data.audio_path;
  state.audioClipMixPeaks = data.peaks || [];
  return data;
}

export function createAudioClipEditor({ state, projectInput, currentGlobalTime, pushHistory,
  render, autoSaveSessionQuiet, pauseTimelineForEditing, setActiveSegment }) {
  let selectedId = "";
  function invalidateMix() {
    state.sceneAudioGlobalTime = currentGlobalTime();
    state.audioClipMixPath = "";
    state.audioClipMixKey = "";
    state.audioClipMixPeaks = [];
  }
  function edit(action) {
    pauseTimelineForEditing();
    pushHistory();
    state.audioClips = audioClipsForState(state).map(clip => ({ ...clip }));
    action(state.audioClips);
    invalidateMix();
    render();
    autoSaveSessionQuiet("audio clips edited").catch(error => toast(String(error), true));
  }
  function registerSceneAudio(scene) {
    if (state.videoType !== "speaking" && !Array.isArray(state.audioClips)) return;
    const clip = audioClipsForState({ segments: [scene] })[0];
    edit(clips => {
      const remaining = clips.filter(item => item.scene_id !== scene.id);
      if (clip) remaining.push({ ...clip, id: `audio_${crypto.randomUUID()}` });
      state.audioClips = remaining;
    });
  }
  function menu(event, clip) {
    event.preventDefault();
    event.stopPropagation();
    selectedId = clip.id;
    document.querySelector(".vrgdg-builder-context-menu")?.remove();
    const box = document.createElement("div");
    const project = projectInput.value;
    const liveClip = () => {
      const current = audioClipsForState(state).find(item => item.id === clip.id);
      if (projectInput.value !== project || !current) throw new Error("This audio clip changed. Select it again.");
      return current;
    };
    box.className = "vrgdg-builder-context-menu";
    box.style.cssText = `position:fixed;z-index:100010;left:${Math.min(event.clientX, window.innerWidth - 230)}px;top:${Math.min(event.clientY, window.innerHeight - 190)}px;display:flex;flex-direction:column;gap:4px;padding:8px;background:#111827;border:1px solid #a78bfa;border-radius:6px;`;
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
      if (event.pointerId !== pointer || projectInput.value !== project) return;
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
    if (state.videoType !== "speaking" && !Array.isArray(state.audioClips)) return;
    for (const clip of audioClipsForState(state)) {
      const width = Math.max(3, clip.duration * state.pxPerSecond);
      const block = document.createElement("div");
      block.title = `${clip.name}: drag to move, drag edges to trim, click or right-click to split at the playhead. Audio gaps are silent.`;
      block.style.cssText = `position:absolute;left:${clip.start * state.pxPerSecond}px;top:${TIMELINE_SCENE_AUDIO_TOP}px;width:${width}px;height:${TIMELINE_SCENE_AUDIO_HEIGHT}px;pointer-events:auto;z-index:2;background:#581c87;border:1px solid ${clip.id === selectedId ? "#fef08a" : "#c084fc"};border-radius:4px;cursor:grab;overflow:hidden;box-sizing:border-box;`;
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
      label.textContent = clip.name;
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
  return { renderAudioClips, registerSceneAudio };
}
