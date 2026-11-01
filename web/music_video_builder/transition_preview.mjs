// The live preview in the player: while the playhead is inside a transition block, the player shows the transition drawn on the frames of the scenes, the way
// the export draws it. A block on a cut has one side playing (the scene the player shows) and the other held on its last or first frame, as in the final
// video. Inside a scene both sides are the playing scene. The canvas lives next to the video and under the overlays.

import { transitionEntries, transitionShared } from "./transition_actions.mjs";
import { transitionCover, transitionSceneFrame } from "./transition_frames.mjs";
import { createTransitionGl } from "./transition_gl.mjs";
import { loadTransitionLibrary, transitionLibrary } from "./transition_library.mjs";
import { TRANSITION_FPS, transitionFrameOf, transitionProgressAt, transitionWindow } from "./transition_model.mjs";
import { transitionElement } from "./transition_dom.mjs";
import { selectedSegmentVideoPath } from "./selection_preview.mjs";

const TRANSITION_LIVE_MAX_WIDTH = 960;

// `visibleMedia()` is the picture or video the player shows now (or null), `currentGlobalTime()` the playhead in seconds.
export function createTransitionPreview({ stage, state, visibleMedia, currentGlobalTime }) {
  transitionShared.clock = currentGlobalTime;
  const canvas = transitionElement("canvas", "position:absolute;left:0;top:0;width:0;height:0;display:none;pointer-events:none;z-index:2;", { className: "h3-transition-live" });
  let preview = null;
  let loadedFor = "";
  let lastPrefetch = 0;
  const holds = new Map();

  // The held frame of a scene (its last or its first), loaded once and scaled to cover the canvas. Null until it is there.
  function hold(segment, atEnd, width, height) {
    if (!segment) return null;
    const key = `${segment.id}|${selectedSegmentVideoPath(segment)}|${segment.video_cache_bust ?? ""}|${atEnd ? "end" : "start"}`;
    let held = holds.get(key);
    if (!held) {
      held = { frame: null, covered: null, size: "" };
      holds.set(key, held);
      transitionSceneFrame(segment, atEnd).then((frame) => { held.frame = frame; }).catch(() => {});
    }
    if (held.frame && held.size !== `${width}x${height}`) {
      held.covered = transitionCover(held.frame, width, height);
      held.size = `${width}x${height}`;
    }
    return held.covered;
  }

  // The frames of the blocks near the playhead are loaded ahead, so a block does not start without its held side.
  function prefetch(frame, segments, list) {
    const width = canvas.width || TRANSITION_LIVE_MAX_WIDTH;
    const height = canvas.height || Math.round((TRANSITION_LIVE_MAX_WIDTH * 9) / 16);
    for (const entry of list) {
      const w = transitionWindow(entry, segments);
      if (w.cut === null || frame < w.first - 5 * TRANSITION_FPS || frame > w.last + TRANSITION_FPS) continue;
      hold(segments[w.cut], true, width, height);
      hold(segments[w.cut + 1], false, width, height);
    }
  }

  function hide() {
    if (canvas.style.display !== "none") canvas.style.display = "none";
  }

  function update() {
    // A build that cannot draw transitions leaves the project's list alone.
    if (!state.transitionsAvailable) return hide();
    const list = transitionEntries(state);
    const segments = state.segments || [];
    const shown = list.length ? visibleMedia() : null;
    if (!shown || !segments.length) return hide();
    const frame = Math.floor((Number(currentGlobalTime()) || 0) * TRANSITION_FPS + 1e-6);
    const now = performance.now();
    if (now - lastPrefetch > 400) {
      lastPrefetch = now;
      prefetch(frame, segments, list);
    }
    let active = null;
    for (const entry of list) {
      const w = transitionWindow(entry, segments);
      const progress = transitionProgressAt(w, frame);
      if (progress !== null) {
        active = { entry, w, progress };
        break;
      }
    }
    if (!active) return hide();
    if (!transitionLibrary().length) {
      loadTransitionLibrary().catch(() => {});
      return hide();
    }
    const item = transitionLibrary().find((candidate) => candidate.id === active.entry.transition);
    if (!item) return hide();

    const media = shown.element || shown;
    const source = typeof shown.frameSource === "function" ? shown.frameSource() : shown;
    const naturalW = source?.videoWidth || source?.naturalWidth || 0;
    const naturalH = source?.videoHeight || source?.naturalHeight || 0;
    const ready = source && (source.tagName === "VIDEO" ? source.readyState >= 2 : source.complete);
    if (!naturalW || !naturalH || !ready) return hide();

    const width = Math.min(TRANSITION_LIVE_MAX_WIDTH, naturalW);
    const height = Math.round((width * naturalH) / naturalW);
    // The held side. Before the cut the outgoing scene plays and the incoming one is held on its first frame, after the cut the other way round.
    let held = null;
    let heldIsFrom = false;
    if (active.w.cut !== null) {
      heldIsFrom = frame >= transitionFrameOf(segments[active.w.cut].end);
      held = heldIsFrom ? hold(segments[active.w.cut], true, width, height) : hold(segments[active.w.cut + 1], false, width, height);
      if (!held) return hide();
    }
    if (!preview || canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      preview ||= createTransitionGl(canvas);
      if (!preview) return hide();
      loadedFor = "";
    }
    if (loadedFor !== item.id) {
      preview.setTransition(item);
      loadedFor = item.id;
    }

    // Layers live next to the video: in the player's own element in the app, in the stage otherwise. Overlays are drawn over the transition.
    const home = shown.element || stage;
    const overlay = home.querySelector(".vrgdg-builder-overlay-layer");
    if (canvas.parentElement !== home || (overlay && canvas.compareDocumentPosition(overlay) & Node.DOCUMENT_POSITION_PRECEDING)) {
      home.insertBefore(canvas, overlay || null);
    }
    const box = media.getBoundingClientRect();
    const around = stage.getBoundingClientRect();
    const scale = Math.min(box.width / naturalW, box.height / naturalH);
    const w = naturalW * scale;
    const h = naturalH * scale;
    const origin = home === media ? { left: 0, top: 0 } : { left: box.left - around.left, top: box.top - around.top };
    Object.assign(canvas.style, { display: "block", left: `${origin.left + (box.width - w) / 2}px`, top: `${origin.top + (box.height - h) / 2}px`, width: `${w}px`, height: `${h}px` });

    // The playing side comes from the player's frame every time. Inside a scene both sides are that frame.
    if (active.w.cut === null) preview.setFrames(source, source);
    else if (heldIsFrom) preview.setFrames(held, source);
    else preview.setFrames(source, held);
    preview.draw(active.progress, active.entry.params);
  }

  // The loop runs on the animation frames of the window the player is in now (the pop-out is another window), and a timer in this window starts it again
  // if that window went away.
  let lastTick = 0;
  let started = false;
  function tick() {
    lastTick = performance.now();
    try {
      update();
    } catch (error) {
      console.warn("[VRGDG Music Builder] The transition preview could not be updated:", error);
    }
    (stage.ownerDocument?.defaultView || window).requestAnimationFrame(tick);
  }

  function start() {
    if (started) return;
    started = true;
    tick();
    setInterval(() => {
      if (performance.now() - lastTick > 1500) tick();
    }, 1000);
  }

  return { start, update, canvas };
}
