// The settings of the selected overlay (Post Process, Overlays): opacity, fades, timing and, by kind, blend mode, speed, grain amount, vignette
// strength, LUT and strength. What each control is, how its value maps to the clip, and the presets are plain functions at the top so they
// can be tested without a page. The panel that draws them follows. Every change goes through `applyOverlayChange`, which lets the clip's
// normalizer clamp it, so a slider can never leave a clip in a state the render does not accept.

import { makeButton, toast } from "./controls.mjs";
import { OVERLAY_BLEND_MODES, OVERLAY_GRAIN_SIZES, OVERLAY_SCALE_MODES, normalizeOverlayLayers } from "./overlay_layers.mjs";

const BLEND_LABELS = { screen: "Screen (drops black)", add: "Add", lighten: "Lighten", darken: "Darken (drops white)", overlay: "Overlay", softlight: "Soft light", hardlight: "Hard light", colordodge: "Color dodge", colorburn: "Color burn", multiply: "Multiply", normal: "Normal (covers)" };
const SCALE_LABELS = { cover: "Fill the frame", contain: "Fit inside", stretch: "Stretch" };
const SIZE_LABELS = { fine: "Fine", medium: "Medium", coarse: "Coarse" };

const percent = { toUi: (v) => Math.round(v * 100), fromUi: (v) => v / 100 };
const plain = { toUi: (v) => v, fromUi: (v) => v };

const FONT_GROUPS = { bundled: "Included with the app", own: "Your font files", system: "Installed on this computer" };

// A font of the fonts list: a name, or an entry that also says where the font comes from (the app's list does), which puts it in a group.
function overlayFontChoice(font) {
  if (typeof font === "string") return { name: font, group: "" };
  const group = font?.system ? FONT_GROUPS.system : font?.bundled ? FONT_GROUPS.bundled : font?.bundled === false ? FONT_GROUPS.own : "";
  return { name: String(font?.name || ""), group };
}

// The font choices: the fonts of the list, and the clip's own font when it is not among them (a project from another computer), so the list
// never shows a different font than the clip uses.
export function overlayFontOptions(fonts, current) {
  const options = (Array.isArray(fonts) ? fonts : [])
    .map(overlayFontChoice)
    .filter((font) => font.name)
    .map(({ name, group }) => (group ? { value: name, label: name, group } : { value: name, label: name }));
  const wanted = String(current || "").trim();
  if (wanted && !options.some((option) => option.value.toLowerCase() === wanted.toLowerCase())) options.push({ value: wanted, label: `${wanted} (not installed)` });
  return options;
}

// The options of a select. Options that name a group are shown under a heading of that name, in the order the groups first appear.
export function appendOverlayOptions(select, options) {
  const groups = new Map();
  for (const option of options) {
    const element = document.createElement("option");
    element.value = option.value;
    element.textContent = option.label;
    if (!option.group) {
      select.append(element);
      continue;
    }
    if (!groups.has(option.group)) {
      const heading = document.createElement("optgroup");
      heading.label = option.group;
      groups.set(option.group, heading);
      select.append(heading);
    }
    groups.get(option.group).append(element);
  }
}

// The controls of a clip, in the order they are shown. `kind` is range (slider with a number), number (just a number box), select, check or
// color. A range or number control has `min`, `max` and `step` in what the person sees, and `toUi` / `fromUi` convert it to the clip's own value.
export function overlayControlsFor(clip, { luts = [], fonts = [] } = {}) {
  if (!clip) return [];
  const common = [
    { key: "enabled", label: "On", kind: "check" },
    { key: "locked", label: "Locked (cannot be dragged)", kind: "check" },
    { key: "opacity", label: "Opacity", kind: "range", min: 0, max: 100, step: 1, unit: "%", ...percent },
    { key: "fade_in", label: "Fade in", kind: "range", min: 0, max: 5, step: 0.1, unit: " s", ...plain },
    { key: "fade_out", label: "Fade out", kind: "range", min: 0, max: 5, step: 0.1, unit: " s", ...plain },
  ];
  const timing = [
    { key: "start", label: "Starts at", kind: "number", min: 0, max: 86400, step: 0.1, unit: " s", ...plain },
    { key: "end", label: "Ends at", kind: "number", min: 0, max: 86400, step: 0.1, unit: " s", ...plain },
  ];
  const own = {
    media: [
      { key: "blend_mode", label: "Blend mode", kind: "select", options: OVERLAY_BLEND_MODES.filter((m) => m !== "normal").concat("normal").map((value) => ({ value, label: BLEND_LABELS[value] || value })) },
      { key: "scale", label: "Size", kind: "select", options: OVERLAY_SCALE_MODES.map((value) => ({ value, label: SCALE_LABELS[value] || value })) },
      { key: "loop", label: "Loop when the clip is longer than its file", kind: "check" },
      { key: "speed", label: "Speed", kind: "range", min: 0.5, max: 2, step: 0.05, unit: "x", ...plain },
      { key: "flip_h", label: "Flip horizontally", kind: "check" },
      { key: "brightness", label: "Brightness", kind: "range", min: -100, max: 100, step: 1, unit: "", ...percent },
      { key: "contrast", label: "Contrast", kind: "range", min: 0, max: 200, step: 1, unit: "%", ...percent },
      { key: "tint", label: "Tint", kind: "color" },
    ],
    grain: [
      { key: "strength", label: "Amount", kind: "range", min: 0, max: 60, step: 1, unit: "", ...plain },
      { key: "chroma", label: "Color amount", kind: "range", min: 0, max: 20, step: 1, unit: "", ...plain },
      { key: "size", label: "Grain size", kind: "select", options: OVERLAY_GRAIN_SIZES.map((value) => ({ value, label: SIZE_LABELS[value] || value })) },
      { key: "animated", label: "Animated", kind: "check" },
    ],
    vignette: [{ key: "strength", label: "Strength", kind: "range", min: 0, max: 100, step: 1, unit: "%", ...percent }],
    text: [
      { key: "text", label: "Text", kind: "textarea" },
      { key: "font", label: "Font", kind: "select", options: overlayFontOptions(fonts, clip.font) },
      { key: "size", label: "Size", kind: "range", min: 8, max: 400, step: 1, unit: " px", ...plain },
      { key: "color", label: "Color", kind: "color", clearable: false },
      { key: "outline_width", label: "Outline", kind: "range", min: 0, max: 20, step: 1, unit: " px", ...plain },
      { key: "outline_color", label: "Outline color", kind: "color", clearable: false },
      { key: "shadow", label: "Shadow", kind: "range", min: 0, max: 20, step: 1, unit: " px", ...plain },
      { key: "x", label: "Left to right", kind: "range", min: 0, max: 100, step: 1, unit: "%", ...percent },
      { key: "y", label: "Top to bottom", kind: "range", min: 0, max: 100, step: 1, unit: "%", ...percent },
      { key: "animation", label: "Animation", kind: "select", options: [{ value: "none", label: "None" }, { value: "slide_up", label: "Slide up while it fades in" }] },
    ],
    lut: [
      { key: "lut", label: "LUT", kind: "select", options: [{ value: "", label: luts.length ? "Choose a LUT" : "No LUTs installed" }, ...luts.map((name) => ({ value: name, label: String(name).replace(/\.cube$/i, "") }))] },
      { key: "strength", label: "Strength", kind: "range", min: 0, max: 100, step: 1, unit: "%", ...percent },
    ],
  }[clip.kind] || [];
  return [...common.slice(0, 2), ...timing, ...common.slice(2), ...own];
}

// The clip list with one setting of one clip changed and clamped. A start after the end pushes the end out and an end before the start pulls
// the start in, so the clip keeps its length and a text box never leaves it backwards.
export function applyOverlayChange(clips, id, key, value) {
  const list = Array.isArray(clips) ? clips : [];
  const changed = list.map((clip) => {
    if (clip.id !== id) return clip;
    const next = { ...clip, [key]: value };
    const length = clip.end - clip.start;
    if (key === "start" && Number(value) >= clip.end) next.end = Number(value) + Math.max(0.2, length);
    if (key === "end" && Number(value) <= clip.start) next.start = Math.max(0, Number(value) - Math.max(0.2, length));
    return next;
  });
  return normalizeOverlayLayers(changed);
}

// One font set on the text clips (titles and lyric captions) that overlap a range of the timeline, or on all of them without a range. A range
// that is given never means all: one with no text in it matches nothing. `count` is how many text clips that is.
export function applyOverlayFontToText(clips, font, range = null) {
  const list = Array.isArray(clips) ? clips : [];
  const inRange = (clip) => !range || (clip.end > Number(range.start) && clip.start < Number(range.end));
  const ids = new Set(list.filter((clip) => clip?.kind === "text" && inRange(clip)).map((clip) => clip.id));
  return { clips: normalizeOverlayLayers(list.map((clip) => (ids.has(clip.id) ? { ...clip, font } : clip))), count: ids.size };
}

// A clip put back to its defaults: it keeps what it is and where it is (id, kind, times, file, LUT) and loses every setting.
export function resetOverlayLayer(clips, id) {
  const keep = ["id", "kind", "start", "end", "path", "lut", "text", "source_start"];
  const list = Array.isArray(clips) ? clips : [];
  return normalizeOverlayLayers(list.map((clip) => (clip.id === id ? Object.fromEntries(keep.filter((k) => k in clip).map((k) => [k, clip[k]])) : clip)));
}

// A clip stretched over the whole timeline, or over a scene (start and end in seconds).
export function fitOverlayLayer(clips, id, start, end) {
  const list = Array.isArray(clips) ? clips : [];
  return normalizeOverlayLayers(list.map((clip) => (clip.id === id && end > start ? { ...clip, start, end } : clip)));
}

export const OVERLAY_PRESETS = {
  subtle_film: { label: "Subtle film", clips: [{ kind: "grain", strength: 4 }, { kind: "vignette", strength: 0.03 }] },
};

// The clips a preset adds, over the selected range of the timeline or all of it. Each is unlocked so it can be dragged at once.
export function overlayPresetClips(name, { total = 0, range = null } = {}) {
  const preset = OVERLAY_PRESETS[name];
  if (!preset) return [];
  const a = Number(range?.in ?? range?.start);
  const b = Number(range?.out ?? range?.end);
  const span = Number.isFinite(a) && Number.isFinite(b) && b > a ? { start: Math.max(0, a), end: b } : { start: 0, end: Math.max(1, Number(total) || 0) };
  return preset.clips.map((clip) => ({ locked: false, ...clip, ...span }));
}

const ROW_STYLE = "display:flex;flex-direction:column;gap:3px;";
const LABEL_STYLE = "font-size:11px;font-weight:700;color:#d4d4d8;display:flex;justify-content:space-between;gap:8px;";
const FIELD_STYLE = "box-sizing:border-box;border:1px solid #3f3f46;border-radius:5px;background:#09090b;color:#fafafa;padding:4px 6px;font-size:12px;";

// Draws the settings of the selected clip into `container`. `env` gives it the page:
//   state, pushHistory(), commit(reason, redrawTimeline), luts() (names), activeSceneRange() ({start, end} or null), sceneRangeAt(time) (the
//   scene that holds that second), total()
export function renderOverlayInspector(container, env) {
  const { state } = env;
  container.textContent = "";
  // The clip as the render sees it: every setting filled in, so a clip saved by an older version or an agent never shows an empty control.
  const found = (state.overlayLayers || []).find((item) => item.id === state.activeOverlayId);
  const clip = found ? normalizeOverlayLayers([found])[0] : null;
  const card = document.createElement("div");
  card.style.cssText = "border:1px solid #3f3f46;border-radius:7px;background:#18181b;padding:10px;display:flex;flex-direction:column;gap:9px;";
  container.append(card);
  if (!clip) {
    card.innerHTML = "<div style=\"font-size:13px;font-weight:900;color:#cffafe;\">Selected clip</div><div style=\"font-size:11px;line-height:1.4;color:#a1a1aa;\">Click a clip on the overlay lane to change its opacity, fades, blend mode and the rest.</div>";
    return;
  }
  const title = document.createElement("div");
  title.style.cssText = "display:flex;justify-content:space-between;align-items:center;gap:8px;";
  const kindName = { media: "Clip", grain: "Film grain", vignette: "Vignette", lut: "LUT", text: "Text" }[clip.kind] || clip.kind;
  title.innerHTML = `<span style="font-size:13px;font-weight:900;color:#cffafe;">${kindName}</span>`;
  card.append(title);

  let gesture = false;
  const apply = (key, value, redraw) => {
    if (!gesture) {
      env.pushHistory();
      gesture = true;
    }
    state.overlayLayers = applyOverlayChange(state.overlayLayers, clip.id, key, value);
    env.commit(`Overlay ${key} changed`, redraw, false);
  };
  const finish = (redraw) => {
    gesture = false;
    env.commit("Overlay changed", redraw, true);
  };
  const current = () => {
    const raw = state.overlayLayers.find((item) => item.id === clip.id);
    return (raw && normalizeOverlayLayers([raw])[0]) || clip;
  };
  const redrawKeys = new Set(["start", "end", "enabled", "locked"]);

  // This clip's font on the other text clips: those of the scene it lies in, or all of them. A lyric caption is a clip of its own for each line,
  // so the font of one does not reach the rest until it is copied here.
  const copyFont = (everywhere) => {
    const now = current();
    const scene = everywhere ? null : env.sceneRangeAt?.((now.start + now.end) / 2) || null;
    if (!everywhere && !scene) {
      toast("This clip does not lie in a scene.", true);
      return;
    }
    const result = applyOverlayFontToText(state.overlayLayers, now.font, scene);
    if (result.count < 2) {
      toast(everywhere ? "This is the only text clip." : "This is the only text clip in its scene.", true);
      return;
    }
    env.pushHistory();
    state.overlayLayers = result.clips;
    env.commit("Overlay font changed", true, true);
    toast(`${now.font} is now the font of ${result.count} text clips.`);
  };
  const fontActions = () => {
    const line = document.createElement("div");
    line.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:6px;";
    for (const [text, title, everywhere] of [
      ["Use for this scene", "Set this font on every text clip (titles and lyric captions) in the scene this clip lies in.", false],
      ["Use for all text", "Set this font on every text clip on the timeline, titles and lyric captions.", true],
    ]) {
      const button = makeButton(text);
      button.title = title;
      button.style.cssText += "padding:5px 6px;font-size:11px;";
      button.onclick = () => copyFont(everywhere);
      line.append(button);
    }
    return line;
  };

  for (const control of overlayControlsFor(clip, { luts: env.luts(), fonts: env.fonts?.() || [] })) {
    const row = document.createElement("div");
    row.style.cssText = ROW_STYLE;
    const redraw = redrawKeys.has(control.key);
    const value = current()[control.key];
    if (control.kind === "textarea") {
      const label = document.createElement("div");
      label.textContent = control.label;
      label.style.cssText = LABEL_STYLE;
      const area = document.createElement("textarea");
      area.rows = 3;
      area.value = String(value ?? "");
      area.style.cssText = `${FIELD_STYLE}width:100%;resize:vertical;line-height:1.35;`;
      ["keydown", "keypress", "keyup"].forEach((name) => area.addEventListener(name, (event) => event.stopPropagation()));
      area.oninput = () => apply(control.key, area.value, false);
      area.onchange = () => finish(true);
      row.append(label, area);
    } else if (control.kind === "check") {
      const label = document.createElement("label");
      label.style.cssText = "display:flex;align-items:center;gap:7px;font-size:11px;color:#f4f4f5;";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = Boolean(value);
      box.onchange = () => {
        apply(control.key, box.checked, redraw);
        finish(redraw);
      };
      label.append(box, document.createTextNode(control.label));
      row.append(label);
    } else if (control.kind === "select") {
      const label = document.createElement("div");
      label.textContent = control.label;
      label.style.cssText = LABEL_STYLE;
      const select = document.createElement("select");
      select.style.cssText = FIELD_STYLE;
      appendOverlayOptions(select, control.options);
      select.value = value ?? "";
      select.onchange = () => {
        apply(control.key, select.value, redraw);
        finish(redraw);
      };
      row.append(label, select);
    } else if (control.kind === "color") {
      const label = document.createElement("div");
      label.textContent = control.label;
      label.style.cssText = LABEL_STYLE;
      const line = document.createElement("div");
      line.style.cssText = "display:flex;gap:6px;align-items:center;";
      const color = document.createElement("input");
      color.type = "color";
      color.value = /^#[0-9a-f]{6}$/i.test(value || "") ? value : "#ffffff";
      color.style.cssText = "width:42px;height:26px;padding:0;border:1px solid #3f3f46;background:none;";
      color.oninput = () => apply(control.key, color.value, false);
      color.onchange = () => finish(false);
      line.append(color);
      if (control.clearable !== false) {
        const none = makeButton("No tint");
        none.style.cssText += "padding:4px 8px;font-size:11px;";
        none.onclick = () => {
          apply(control.key, "", false);
          finish(false);
          color.value = "#ffffff";
        };
        line.append(none);
      }
      row.append(label, line);
    } else {
      // range or number
      const shown = (v) => control.toUi(Number(v));
      const label = document.createElement("div");
      label.style.cssText = LABEL_STYLE;
      const name = document.createElement("span");
      name.textContent = control.label;
      const readout = document.createElement("span");
      readout.style.cssText = "color:#67e8f9;font-variant-numeric:tabular-nums;";
      const number = document.createElement("input");
      number.type = "number";
      number.min = String(control.min);
      number.max = String(control.max);
      number.step = String(control.step);
      number.value = String(Math.round(shown(value) * 1000) / 1000);
      number.style.cssText = `${FIELD_STYLE}width:${control.kind === "number" ? "100%" : "64px"};`;
      const set = (raw, redrawNow) => {
        const next = Number(raw);
        if (!Number.isFinite(next)) return;
        apply(control.key, control.fromUi(next), redrawNow);
        const stored = current()[control.key];
        number.value = String(Math.round(shown(stored) * 1000) / 1000);
        if (slider) slider.value = String(shown(stored));
        readout.textContent = `${Math.round(shown(stored) * 100) / 100}${control.unit || ""}`;
      };
      let slider = null;
      readout.textContent = `${Math.round(shown(value) * 100) / 100}${control.unit || ""}`;
      label.append(name, readout);
      if (control.kind === "range") {
        slider = document.createElement("input");
        slider.type = "range";
        slider.min = String(control.min);
        slider.max = String(control.max);
        slider.step = String(control.step);
        slider.value = String(shown(value));
        slider.style.cssText = "width:100%;accent-color:#06b6d4;";
        slider.oninput = () => set(slider.value, false);
        slider.onchange = () => finish(false);
        const line = document.createElement("div");
        line.style.cssText = "display:flex;gap:8px;align-items:center;";
        line.append(slider, number);
        row.append(label, line);
      } else {
        row.append(label, number);
      }
      number.oninput = () => set(number.value, redraw);
      number.onchange = () => finish(redraw);
    }
    card.append(row);
    if (control.key === "font") card.append(fontActions());
  }

  const actions = document.createElement("div");
  actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:6px;border-top:1px solid #27272a;padding-top:8px;";
  const small = (text, title, onclick) => {
    const button = makeButton(text);
    button.title = title;
    button.style.cssText += "padding:5px 6px;font-size:11px;";
    button.onclick = onclick;
    actions.append(button);
  };
  small("Whole timeline", "Stretch this clip over the whole timeline.", () => {
    env.pushHistory();
    state.overlayLayers = fitOverlayLayer(state.overlayLayers, clip.id, 0, env.total());
    env.commit("Overlay stretched", true, true);
  });
  small("This scene", "Make this clip start and end with the selected scene.", () => {
    const range = env.activeSceneRange();
    if (!range) return;
    env.pushHistory();
    state.overlayLayers = fitOverlayLayer(state.overlayLayers, clip.id, range.start, range.end);
    env.commit("Overlay fitted to the scene", true, true);
  });
  small("Reset settings", "Put every setting of this clip back to its default. Its place on the timeline stays.", () => {
    env.pushHistory();
    state.overlayLayers = resetOverlayLayer(state.overlayLayers, clip.id);
    env.commit("Overlay reset", true, true);
  });
  small("Delete clip", "Remove this clip from the timeline.", () => {
    env.pushHistory();
    state.overlayLayers = state.overlayLayers.filter((item) => item.id !== clip.id);
    state.activeOverlayId = "";
    env.commit("Overlay deleted", true, true);
  });
  card.append(actions);
}
