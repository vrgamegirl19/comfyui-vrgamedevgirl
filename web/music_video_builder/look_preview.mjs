// The live preview in the player: while the playhead is inside a filter or effect block, the player shows the video's frame drawn through the looks, the way the export
// draws it (the filters that cover the frame first, then the effects, each in the order of the timeline). When a transition is showing too, the looks are drawn on
// the transition's picture, as in the final video. The canvas lives next to the video, over the transition canvas and under the overlays.

import { lookEntries } from "./look_actions.mjs";
import { createLookGl } from "./look_gl.mjs";
import { loadLookLibrary, lookLibrary } from "./look_library.mjs";
import { LOOK_FPS, LOOK_KINDS, LOOK_ORDER, lookStateAt } from "./look_model.mjs";
import { transitionElement } from "./transition_dom.mjs";

const LOOK_LIVE_MAX_WIDTH = 960;

// `visibleMedia()` is the picture or video the player shows now (or null), `currentGlobalTime()` the playhead in seconds and `transitionCanvas` the canvas of the
// transition preview.
export function createLookPreview({ stage, state, visibleMedia, currentGlobalTime, transitionCanvas }) {
  const canvas = transitionElement("canvas", "position:absolute;left:0;top:0;width:0;height:0;display:none;pointer-events:none;z-index:2;", { className: "h3-look-live" });
  let preview = null;

  function hide() {
    if (canvas.style.display !== "none") canvas.style.display = "none";
  }

  // The looks that cover `frame`, filters first and then effects, each with how far it has run.
  function activeSteps(frame) {
    const steps = [];
    for (const kind of LOOK_ORDER) {
      const config = LOOK_KINDS[kind];
      if (!state[config.availableKey]) continue;
      for (const entry of lookEntries(state, kind)) {
        const run = lookStateAt(entry, frame);
        if (!run) continue;
        const item = lookLibrary(kind).find((candidate) => candidate.id === entry.look);
        if (item) steps.push({ item, progress: run.progress, time: run.time, intensity: entry.intensity, params: entry.params });
        else if (!lookLibrary(kind).length) loadLookLibrary(kind).catch(() => {});
      }
    }
    return steps;
  }

  function update() {
    // A build that cannot draw them leaves the project's lists alone.
    if (!LOOK_ORDER.some((kind) => state[LOOK_KINDS[kind].availableKey])) return hide();
    const shown = visibleMedia();
    if (!shown) return hide();
    const frame = Math.floor((Number(currentGlobalTime()) || 0) * LOOK_FPS + 1e-6);
    const steps = activeSteps(frame);
    if (!steps.length) return hide();

    const media = shown.element || shown;
    const showingTransition = transitionCanvas.style.display !== "none" && transitionCanvas.width > 0;
    const source = showingTransition ? transitionCanvas : typeof shown.frameSource === "function" ? shown.frameSource() : shown;
    const naturalW = source?.videoWidth || source?.naturalWidth || source?.width || 0;
    const naturalH = source?.videoHeight || source?.naturalHeight || source?.height || 0;
    const ready = source && (source.tagName === "VIDEO" ? source.readyState >= 2 : source.tagName === "IMG" ? source.complete : true);
    if (!naturalW || !naturalH || !ready) return hide();

    const width = Math.min(LOOK_LIVE_MAX_WIDTH, naturalW);
    const height = Math.round((width * naturalH) / naturalW);
    if (!preview || canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      preview ||= createLookGl(canvas);
      if (!preview) return hide();
    }

    // Layers live next to the video: in the player's own element in the app, in the stage otherwise. The transition is under the looks and the overlays over them.
    const home = shown.element || stage;
    const overlay = home.querySelector(".vrgdg-builder-overlay-layer");
    const under = transitionCanvas.parentElement === home ? transitionCanvas.nextSibling : null;
    if (canvas.parentElement !== home || (overlay && canvas.compareDocumentPosition(overlay) & Node.DOCUMENT_POSITION_PRECEDING) || (under && under !== canvas && canvas.compareDocumentPosition(transitionCanvas) & Node.DOCUMENT_POSITION_FOLLOWING)) {
      home.insertBefore(canvas, under || overlay || null);
    }
    const box = media.getBoundingClientRect();
    const around = stage.getBoundingClientRect();
    const scale = Math.min(box.width / naturalW, box.height / naturalH);
    const w = naturalW * scale;
    const h = naturalH * scale;
    const origin = home === media ? { left: 0, top: 0 } : { left: box.left - around.left, top: box.top - around.top };
    Object.assign(canvas.style, { display: "block", left: `${origin.left + (box.width - w) / 2}px`, top: `${origin.top + (box.height - h) / 2}px`, width: `${w}px`, height: `${h}px` });

    preview.setFrame(source);
    preview.draw(steps);
  }

  // The loop runs on the animation frames of the window the player is in now (the pop-out is another window), and a timer in this window starts it again if that
  // window went away.
  let lastTick = 0;
  let started = false;
  function tick() {
    lastTick = performance.now();
    try {
      update();
    } catch (error) {
      console.warn("[VRGDG Music Builder] The effect and filter preview could not be updated:", error);
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
