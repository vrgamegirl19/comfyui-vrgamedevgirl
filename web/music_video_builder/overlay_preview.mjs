// The live preview of the overlays in the player. The player shows the selected scene's video. This puts one layer for each overlay over
// it, in step with the timeline: a media clip is a small playable copy of the file (a browser cannot play a 4K ProRes file) blended with
// the CSS blend mode, grain is a canvas of fresh noise every few frames, a vignette is a gradient with the same falloff as ffmpeg's. Fades
// and opacity follow the clip. It is a close look, not the exact render: the exact picture is the one Export makes, and LUT clips are
// not drawn here yet. Text is drawn as page text in the clip's font, placed and sized as the render places it. The rules at the top are plain functions so they can be tested without a page.
//
// Each layer is a child of the player's own container (the stage, or in the app the element that holds the player's video elements), placed
// by pixel rectangle on the picture. A layer is not wrapped in a group with a z-index or an opacity of its own: a wrapper like that makes a
// stacking context, and a blend mode only reaches what is in its own context, not the video underneath.

import { api } from "../../../scripts/api.js";
import { normalizeOverlayLayers, normalizeOverlayLayerTrack } from "./overlay_layers.mjs";

export const OVERLAY_CSS_BLEND = { screen: "screen", add: "plus-lighter", lighten: "lighten", darken: "darken", overlay: "overlay", softlight: "soft-light", hardlight: "hard-light", colordodge: "color-dodge", colorburn: "color-burn", multiply: "multiply", normal: "normal" };
const OVERLAY_FIT = { cover: "cover", contain: "contain", stretch: "fill" };
const GRAIN_CANVAS = { fine: [480, 270], medium: [320, 180], coarse: [240, 135] };
const GRAIN_STEP_MS = 70;
const DRIFT_PLAYING = 0.35;
const DRIFT_PAUSED = 0.05;

// Where a picture of `naturalW` x `naturalH` lands when it is fitted inside a box of `boxW` x `boxH` (object-fit: contain).
export function overlayPreviewFrameRect(boxW, boxH, naturalW, naturalH) {
  if (!(naturalW > 0 && naturalH > 0 && boxW > 0 && boxH > 0)) return { left: 0, top: 0, width: Math.max(0, boxW), height: Math.max(0, boxH) };
  const scale = Math.min(boxW / naturalW, boxH / naturalH);
  const width = naturalW * scale;
  const height = naturalH * scale;
  return { left: (boxW - width) / 2, top: (boxH - height) / 2, width, height };
}

// How much of a clip shows at time `t`: 0 outside it, 1 in the middle, ramping over its fade in and fade out.
export function overlayEnvelope(clip, t) {
  if (t < clip.start || t > clip.end) return 0;
  let amount = 1;
  if (clip.fade_in > 0) amount = Math.min(amount, (t - clip.start) / clip.fade_in);
  if (clip.fade_out > 0) amount = Math.min(amount, (clip.end - t) / clip.fade_out);
  return Math.max(0, Math.min(1, amount));
}

// The vignette as a gradient. ffmpeg darkens a pixel by cos(angle * r) to the fourth power, with r the distance from the middle as a share of the
// distance to a corner, and the strength slider maps to the angle as 1.2 * strength (no strength, no vignette). A circle gradient reaching the farthest corner has
// the same r, so the stops are the same numbers.
export function overlayVignetteGradient(strength) {
  const angle = 1.2 * Math.max(0, Math.min(1, Number(strength) || 0));
  const stops = [];
  for (let step = 0; step <= 10; step += 1) {
    const r = step / 10;
    const dark = 1 - Math.cos(angle * r) ** 4;
    stops.push(`rgba(0,0,0,${dark.toFixed(3)}) ${step * 10}%`);
  }
  return `radial-gradient(circle farthest-corner at 50% 50%, ${stops.join(", ")})`;
}

// Where in its file a media clip is at time `t`. The first pass starts at the clip's source start, a loop starts again from the start of the
// file (that is what ffmpeg's -stream_loop does in the render), and a clip that does not loop stays on its last frame.
export function overlayMediaTime(clip, t, duration) {
  const into = Math.max(0, t - clip.start) * (clip.speed || 1) + (clip.source_start || 0);
  if (!(duration > 0)) return into;
  if (into < duration) return into;
  return clip.loop ? (into - duration) % duration : Math.max(0, duration - 0.05);
}

// Where a text clip's words sit and how big they are, for a picture `frameHeight` pixels tall at time `t`. The render sizes everything from a
// 1080 pixel tall picture, centers the text on x and y (shares of the picture), and with the slide up animation starts it 4% of the picture's
// height low and brings it up over the fade in (at least 0.4 s).
export function overlayTextMetrics(clip, t, frameHeight) {
  const scale = (Number(frameHeight) || 0) / 1080;
  const slide = clip.animation === "slide_up" ? 0.04 * (1 - Math.max(0, Math.min(1, (t - clip.start) / Math.max(clip.fade_in || 0, 0.4)))) : 0;
  return {
    fontSize: clip.size * scale,
    outline: clip.outline_width * scale,
    shadow: clip.shadow * scale,
    left: clip.x * 100,
    top: clip.y * 100,
    offset: slide * (Number(frameHeight) || 0),
  };
}

// The font of a clip as a CSS family. The file comes from the app and is added to the document the player is in (the pop-out is another
// document). Until it has loaded the browser shows its fallback, and the text changes to the font when it arrives.
const fontFamilies = new WeakMap();
export function overlayFontFamily(doc, name) {
  const wanted = String(name || "").trim();
  if (!wanted) return "sans-serif";
  const key = wanted.toLowerCase();
  let known = fontFamilies.get(doc);
  if (!known) fontFamilies.set(doc, (known = new Map()));
  if (!known.has(key)) {
    const family = `vrgdg-overlay-${key.replace(/[^a-z0-9]+/g, "-")}`;
    known.set(key, family);
    const Face = (doc.defaultView || (typeof window !== "undefined" ? window : null))?.FontFace;
    if (Face && doc.fonts) {
      const face = new Face(family, `url(${api.apiURL(`/vrgdg/music_builder/overlay_layers/font?name=${encodeURIComponent(wanted)}`)})`);
      face.load().then(() => doc.fonts.add(face)).catch(() => {});
    }
  }
  return `"${known.get(key)}", sans-serif`;
}

export function overlayPreviewUrl(path, projectFolder) {
  return api.apiURL(`/vrgdg/music_builder/overlay_layers/preview?path=${encodeURIComponent(path)}&project_folder=${encodeURIComponent(projectFolder || "")}`);
}

// `visibleMedia()` is the picture or video the player shows now (or null), `currentGlobalTime()` the playhead in seconds,
// `isTimelinePlaying()` whether it is moving, `projectFolder()` the open project's folder.
export function createOverlayPreview({ stage, state, visibleMedia, currentGlobalTime, isTimelinePlaying, projectFolder }) {
  const layers = new Map();
  let parent = null;
  let frameRequest = 0;
  let lastTick = 0;

  function drop(id) {
    const layer = layers.get(id);
    if (!layer) return;
    layer.video?.pause();
    if (layer.video) layer.video.removeAttribute("src");
    layer.element.remove();
    layers.delete(id);
  }

  function makeLayer(clip) {
    const element = document.createElement("div");
    element.className = "vrgdg-builder-overlay-layer";
    element.dataset.overlayId = clip.id;
    element.style.cssText = "position:absolute;left:0;top:0;width:0;height:0;overflow:hidden;pointer-events:none;display:none;z-index:2;";
    const layer = { element, kind: clip.kind, key: "", video: null, canvas: null, noiseAt: 0 };
    if (clip.kind === "media") {
      const video = document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.preload = "auto";
      video.style.cssText = "position:absolute;inset:0;width:100%;height:100%;";
      video.addEventListener("error", () => { layer.failed = true; });
      element.append(video);
      layer.video = video;
    } else if (clip.kind === "text") {
      const words = document.createElement("div");
      words.style.cssText = "position:absolute;white-space:pre;line-height:normal;text-align:left;paint-order:stroke fill;";
      element.append(words);
      layer.words = words;
    } else if (clip.kind === "grain") {
      const canvas = document.createElement("canvas");
      canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated;";
      element.append(canvas);
      layer.canvas = canvas;
    }
    (parent || stage).append(element);
    layers.set(clip.id, layer);
    return layer;
  }

  function paintNoise(layer, clip) {
    const [w, h] = GRAIN_CANVAS[clip.size] || GRAIN_CANVAS.fine;
    if (layer.canvas.width !== w) {
      layer.canvas.width = w;
      layer.canvas.height = h;
    }
    const context = layer.canvas.getContext("2d");
    const image = context.createImageData(w, h);
    const amount = Math.max(0, clip.strength);
    const data = image.data;
    for (let i = 0; i < data.length; i += 4) {
      const value = 128 + (Math.random() * 2 - 1) * amount;
      data[i] = data[i + 1] = data[i + 2] = value;
      data[i + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }

  function syncText(layer, clip, t, frameHeight) {
    const metrics = overlayTextMetrics(clip, t, frameHeight);
    const words = layer.words;
    const key = [clip.text, clip.font, clip.color, clip.outline_color, metrics.fontSize.toFixed(2), metrics.outline.toFixed(2), metrics.shadow.toFixed(2), metrics.left, metrics.top].join("|");
    if (layer.key !== key) {
      layer.key = key;
      words.textContent = clip.text;
      words.style.fontFamily = overlayFontFamily(words.ownerDocument, clip.font);
      words.style.fontSize = `${metrics.fontSize}px`;
      words.style.color = clip.color;
      words.style.webkitTextStroke = metrics.outline > 0 ? `${metrics.outline * 2}px ${clip.outline_color}` : "0";
      words.style.textShadow = metrics.shadow > 0 ? `${metrics.shadow}px ${metrics.shadow}px 0 rgba(0,0,0,0.6)` : "none";
      words.style.left = `${metrics.left}%`;
      words.style.top = `${metrics.top}%`;
    }
    words.style.transform = `translate(-50%, calc(-50% + ${metrics.offset}px))`;
  }

  function syncVideo(layer, clip, t, playing, show) {
    const video = layer.video;
    const key = `${clip.path}|${projectFolder?.() || ""}`;
    if (layer.key !== key) {
      layer.key = key;
      layer.failed = false;
      video.src = overlayPreviewUrl(clip.path, projectFolder?.() || "");
    }
    video.style.objectFit = OVERLAY_FIT[clip.scale] || "cover";
    video.style.transform = clip.flip_h ? "scaleX(-1)" : "";
    video.style.filter = clip.brightness || clip.contrast !== 1 ? `brightness(${1 + clip.brightness}) contrast(${clip.contrast})` : "";
    if (!show || layer.failed || video.readyState < 1) {
      if (!video.paused) video.pause();
      return;
    }
    const target = overlayMediaTime(clip, t, video.duration);
    video.playbackRate = Math.max(0.0625, Math.min(16, clip.speed || 1));
    const drift = Math.abs(video.currentTime - target);
    if (drift > (playing ? DRIFT_PLAYING : DRIFT_PAUSED) && !video.seeking) {
      try { video.currentTime = target; } catch { /* the next tick tries again */ }
    }
    if (playing && video.paused) video.play().catch(() => {});
    if (!playing && !video.paused) video.pause();
  }

  function update() {
    const clips = state.overlayAvailable && normalizeOverlayLayerTrack(state.overlayLayerTrack).enabled !== false ? normalizeOverlayLayers(state.overlayLayers).filter((clip) => clip.enabled) : [];
    // The player is a <video> or <img> in the pack, and in the app an object that holds its own <video> elements: `element` is what is drawn
    // and `frameSource()` the video that is on screen.
    const shown = clips.length ? visibleMedia() : null;
    const media = shown ? (shown.element || shown) : null;
    const source = typeof shown?.frameSource === "function" ? shown.frameSource() : shown;
    for (const id of [...layers.keys()]) {
      if (!clips.some((clip) => clip.id === id)) drop(id);
    }
    if (!clips.length || !media) {
      for (const layer of layers.values()) {
        layer.element.style.display = "none";
        layer.video?.pause();
      }
      return;
    }
    // Layers live next to the video: in the player's own element in the app, in the stage otherwise.
    const home = shown.element || stage;
    if (parent !== home) {
      parent = home;
      for (const layer of layers.values()) home.append(layer.element);
    }
    const box = media.getBoundingClientRect();
    const around = stage.getBoundingClientRect();
    const naturalW = source?.videoWidth || source?.naturalWidth || 0;
    const naturalH = source?.videoHeight || source?.naturalHeight || 0;
    const rect = overlayPreviewFrameRect(box.width, box.height, naturalW, naturalH);
    // Where the picture is, inside the container the layers live in.
    const origin = home === media ? { left: 0, top: 0 } : { left: box.left - around.left, top: box.top - around.top };
    const place = { left: `${origin.left + rect.left}px`, top: `${origin.top + rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px` };
    const t = Number(currentGlobalTime()) || 0;
    const playing = Boolean(isTimelinePlaying());
    const now = performance.now();
    for (const clip of clips) {
      const layer = layers.get(clip.id) && layers.get(clip.id).kind === clip.kind ? layers.get(clip.id) : (drop(clip.id), makeLayer(clip));
      const amount = overlayEnvelope(clip, t);
      const show = amount > 0 && (clip.kind === "media" || clip.kind === "grain" || clip.kind === "vignette" || clip.kind === "text");
      Object.assign(layer.element.style, place, { display: show ? "block" : "none" });
      if (clip.kind === "media") {
        layer.element.style.mixBlendMode = OVERLAY_CSS_BLEND[clip.blend_mode] || "screen";
        syncVideo(layer, clip, t, playing, show);
      } else if (clip.kind === "grain") {
        layer.element.style.mixBlendMode = "overlay";
        if (show && (now - layer.noiseAt >= GRAIN_STEP_MS || layer.key !== `${clip.size}|${clip.strength}`)) {
          if (clip.animated || layer.key !== `${clip.size}|${clip.strength}`) paintNoise(layer, clip);
          layer.key = `${clip.size}|${clip.strength}`;
          layer.noiseAt = now;
        }
      } else if (clip.kind === "text") {
        layer.element.style.mixBlendMode = "normal";
        if (show) syncText(layer, clip, t, rect.height);
      } else if (clip.kind === "vignette") {
        layer.element.style.mixBlendMode = "normal";
        const background = overlayVignetteGradient(clip.strength);
        if (layer.key !== background) {
          layer.key = background;
          layer.element.style.background = background;
        }
      }
      layer.element.style.opacity = String(clip.opacity * amount);
    }
  }

  // The loop runs on the animation frames of the window the player is in now (the pop-out is another window), and a timer in this window
  // starts it again if that window went away.
  function tick() {
    lastTick = performance.now();
    try {
      update();
    } catch (error) {
      console.warn("[VRGDG Music Builder] The overlay preview could not be updated:", error);
    }
    const view = stage.ownerDocument?.defaultView || window;
    frameRequest = view.requestAnimationFrame(tick);
  }

  function start() {
    if (frameRequest) return;
    tick();
    setInterval(() => {
      if (performance.now() - lastTick > 1500) {
        frameRequest = 0;
        tick();
      }
    }, 1000);
  }

  return { start, update };
}
