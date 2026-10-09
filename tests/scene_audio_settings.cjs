const assert = require("assert");
const vm = require("vm");
const { readBuilderModule } = require("./builder_source.cjs");

const handlers = new Map();
const document = { body: {}, documentElement: {},
  addEventListener: (type, fn) => handlers.set(type, fn) };
const state = { videoType: "speaking", segments: [{ id: "a", start: 0, end: 4,
  image: "keep", i2v_prompt: "keep prompt", custom_audio_path: "source.wav" }], audioClips: [] };
const runtimeScene = state.segments[0];
const context = vm.createContext({ document, setTimeout: fn => fn(),
  audioClipsForState: s => s.audioClips, toast: () => {} });
vm.runInContext(readBuilderModule("scene_dialogue.mjs"), context);
vm.runInContext(readBuilderModule("scene_audio_settings.mjs"), context);
const snapshot = context.sceneAudioSnapshot(state);
assert.strictEqual(snapshot.segments[0].image, undefined);
assert.strictEqual(snapshot.segments[0].i2v_prompt, undefined);
context.applySceneAudioResult(state, { segments: [{ id: "a", start: 0, end: 7,
  scene_audio_settings: { silence_after: 1 } }], audio_clips: [{ id: "clip" }],
  speaking_audio_defaults: { silence_before: 0 } });
assert.strictEqual(state.segments[0], runtimeScene);
assert.strictEqual(runtimeScene.image, "keep");
assert.strictEqual(runtimeScene.i2v_prompt, "keep prompt");
assert.strictEqual(runtimeScene.end, 7);
assert.strictEqual(state.audioClipMixKey, "");
assert.strictEqual(context.canOpenSceneAudio(state, runtimeScene), true);
state.videoType = "singing";
assert.strictEqual(context.canOpenSceneAudio(state, runtimeScene), false);
state.videoType = "speaking";
assert.strictEqual(context.canOpenSceneAudio(state, { id: "overlay" }), false);

vm.runInContext(readBuilderModule("keyboard_shortcuts.mjs"), context);
let opens = 0;
const overlay = { isConnected: true, contains: target => target?.insideBuilder, focus: () => {} };
context.wireKeyboardShortcuts({ state, overlay, builderLifecycle: {}, activeSegment: () => runtimeScene,
  segmentTrack: () => "base", openSceneAudioSettings: () => opens++ });
const handler = handlers.get("keydown");
function event(overrides = {}) {
  return { target: { tagName: "DIV", insideBuilder: true }, key: "A", ctrlKey: true,
    shiftKey: true, preventDefault() { this.prevented = true; }, stopPropagation() {}, ...overrides };
}
const removedShortcut = event(); handler(removedShortcut); assert.strictEqual(opens, 0);
assert.strictEqual(removedShortcut.prevented, undefined);
handler(event({ repeat: true })); assert.strictEqual(opens, 0);
handler(event({ target: { tagName: "TEXTAREA", insideBuilder: true } })); assert.strictEqual(opens, 0);
handler(event({ target: { tagName: "INPUT", insideBuilder: true } })); assert.strictEqual(opens, 0);
handler(event({ target: { tagName: "DIV", insideBuilder: false } })); assert.strictEqual(opens, 0);
state.videoType = "singing";
const singing = event(); handler(singing); assert.strictEqual(opens, 0); assert.strictEqual(singing.prevented, undefined);
state.videoType = "speaking";
const selectAll = event({ shiftKey: false }); handler(selectAll);
assert.strictEqual(opens, 0); assert.strictEqual(selectAll.prevented, undefined);
overlay.isConnected = false; handler(event()); assert.strictEqual(opens, 0);

// Exercise the real dialog builder and mode-switch cleanup with a small DOM stand-in.
class Element {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.style = {}; this.isConnected = true; }
  append(...children) { this.children.push(...children); }
  after(child) { this.afterElement = child; }
  setAttribute(key, value) { this[key] = value; }
  removeAttribute(key) { delete this[key]; }
  remove() { this.isConnected = false; }
  pause() { this.paused = true; }
  load() {}
  focus() { document.activeElement = this; }
  querySelectorAll(selector) {
    if (selector === "input:invalid") return [];
    const tags = selector.split(",").map(tag => tag.toUpperCase());
    return this.children.flatMap(child => child instanceof Element
      ? [...(tags.includes(child.tagName) ? [child] : []), ...child.querySelectorAll(selector)] : []);
  }
}
document.body = new Element("body");
document.createElement = tag => new Element(tag);
document.removeEventListener = (key, fn) => { if (handlers.get(key) === fn) handlers.delete(key); };
context.makeButton = text => { const button = new Element("button"); button.textContent = text; return button; };
context.makeInput = (value, type) => { const input = new Element("input"); input.value = value; input.type = type; return input; };
context.makeField = (text, input) => { const label = new Element("label"); label.textContent = text; label.append(input); return label; };
context.makeSelect = (options, value = "") => { const select = new Element("select"); select.value = value; return select; };
context.makeCheckbox = (label, checked) => { const wrapper = new Element("label"), input = new Element("input"); input.checked = checked; wrapper.append(input); return { wrapper, input }; };
const anchor = new Element("button");
overlay.isConnected = true;
let pauses = 0;
const feature = context.createSceneAudioSettings({ state, projectInput: { value: "project" }, overlay,
  projectDefaultsAnchor: anchor, pushHistory: () => {}, pauseTimelineForEditing: () => pauses++,
  render: () => {}, saveSession: async () => {}, sceneSlotNumber: () => 1 });
state.videoType = "singing"; feature.refresh();
assert.strictEqual(anchor.afterElement.style.display, "none");
feature.openSceneAudioSettings(runtimeScene); assert.strictEqual(document.body.children.length, 0);
state.videoType = "speaking"; feature.refresh();
assert.strictEqual(anchor.afterElement.style.display, "");
feature.openSceneAudioSettings(runtimeScene);
const backdrop = document.body.children.at(-1);
const box = backdrop.children[0];
assert.strictEqual(box.role, "dialog");
assert.ok(box.children[0].textContent.includes("Audio Settings"));
assert.ok(!box.children.some(item => /Video prompt|Generate Dialogue|Design Voice/.test(item.textContent || "")));
const inputs = box.querySelectorAll("input");
const inherit = inputs.find(input => input.type === "checkbox");
assert.strictEqual(inherit.checked, true);
assert.ok(inputs.filter(input => input.type === "number").every(input => input.disabled));
inherit.checked = false; inherit.onchange();
assert.ok(inputs.filter(input => input.type === "number").every(input => !input.disabled));
const unchanged = JSON.stringify(state);
box.querySelectorAll("button").find(button => button.textContent === "Cancel").onclick();
assert.strictEqual(backdrop.isConnected, false);
assert.strictEqual(JSON.stringify(state), unchanged);
feature.openSceneAudioSettings(runtimeScene);
const modeDialog = document.body.children.at(-1);
state.videoType = "singing"; feature.refresh();
assert.strictEqual(modeDialog.isConnected, false);
assert.ok(modeDialog.children[0].querySelectorAll("audio")[0].paused);
feature.dispose(); assert.strictEqual(anchor.afterElement.isConnected, false);
assert.ok(pauses >= 2);
console.log("Scene audio state and keyboard behavior passed.");
