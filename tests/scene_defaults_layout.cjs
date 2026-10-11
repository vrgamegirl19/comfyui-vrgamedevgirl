const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.style = { cssText: "" };
    this.attributes = {};
    this.events = {};
    this.dataset = {};
    this.className = "";
    this.classList = { add: (name) => { this.className += ` ${name}`; } };
  }
  append(...children) {
    for (const child of children) {
      if (child.parentElement) child.parentElement.children = child.parentElement.children.filter((item) => item !== child);
      child.parentElement = this;
      this.children.push(child);
    }
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(key, callback) { this.events[key] = callback; }
  get firstElementChild() { return this.children[0]; }
  get lastElementChild() { return this.children.at(-1); }
  querySelectorAll(selector) {
    const tags = selector.split(",").map((tag) => tag.trim());
    return descendants(this).filter((node) => tags.includes(node.tagName));
  }
}
const descendants = (element) => element.children.flatMap((child) => [child, ...descendants(child)]);

function fixture() {
  const document = { head: new Element("head"), createElement: (tag) => new Element(tag),
    createElementNS: (_namespace, tag) => new Element(tag),
    createTextNode: (text) => Object.assign(new Element("#text"), { textContent: text }),
    getElementById: (id) => descendants(document.head).find((element) => element.id === id) };
  const context = vm.createContext({ document });
  for (const name of ["controls", "scenes", "shot_presets", "video_style", "defaults_cards", "defaults_presets", "defaults_panel_layout"]) {
    const source = fs.readFileSync(path.join(__dirname, "../web/storyboard_builder", `${name}.mjs`), "utf8")
      .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
    vm.runInContext(source, context, { filename: name });
  }
  const state = { cameraFlow: "medium_shots", cameraMotionSpeed: 4, characterMotionSpeed: 4,
    facialPerformanceCustom: "Expression sentinel", videoStyleCustom: "Style sentinel", fxCustomJson: '{"cues":["Glow"]}',
    globalConsistencyPhrase: "Consistency sentinel", storyLayer: {} };
  const panel = context.buildSceneDefaultsPanel({ state, focusedSection: "defaults", usesFilmPlanningProfile: false,
    facialPerformancePresets: [{ value: "", label: "Natural" }], imageAestheticPresets: [{ value: "", label: "Natural" }],
    performanceStylePresets: [{ value: "", label: "Natural" }], imageShotFlowPresets: { intimate: { label: "Intimate" } } });
  return { panel, state, document };
}

test("Four cards retain every original control, action button and helper", () => {
  const { panel } = fixture();
  const nodes = descendants(panel.sceneDefaultsPanel);
  assert.deepEqual(nodes.filter((node) => node.tagName === "h3").map((node) => node.textContent),
    ["Visual Style", "Camera & Editing", "Performance", "Facial & Story"]);
  for (const [name, control] of Object.entries(panel)) {
    if (control instanceof Element && !["sceneDefaultsPanel", "middleContent", "note"].includes(name)) {
      assert.equal(nodes.filter((node) => node === control).length, 1, name);
    }
  }
  const settings = nodes.filter((node) => node.className === "vrgdg-scene-defaults-setting");
  assert.equal(settings.length, 16); // Includes Image Prep settings and nested custom controls.
  for (const setting of settings) {
    assert.equal(setting.children[0].attributes["aria-expanded"], "false");
    assert.equal(setting.children[1].hidden, true);
  }
  assert.equal(panel.cameraSpeedInput.min, "0");
  assert.equal(panel.cameraSpeedInput.max, "10");
  assert.equal(panel.cameraFlowSelect.value, "medium_shots");
  assert.equal(Object.keys(panel.selectedDefaultActions).length, 6);
  for (const button of Object.values(panel.selectedDefaultActions)) {
    assert.equal(button.textContent, "Apply to selected");
    assert.equal(button.hidden, true);
    assert.equal(nodes.filter((node) => node === button).length, 1);
    assert.match(button.style.cssText, /padding:5px 8px/);
  }
  assert.match(panel.cameraFlowApply.style.cssText, /padding:5px 8px/);
  assert.match(panel.cameraFlowReplace.style.cssText, /font-size:11px/);
});

test("Opening across cards closes the previous setting without firing control handlers or modifying values", () => {
  const { panel, state } = fixture();
  const nodes = descendants(panel.sceneDefaultsPanel);
  const toggles = nodes.filter((node) => node.className === "vrgdg-scene-defaults-toggle");
  const find = (name) => toggles.find((toggle) => toggle.firstElementChild.textContent === name);
  const before = JSON.stringify(state);
  let callbacks = 0;
  panel.cameraFlowSelect.events.change = () => { callbacks++; };
  panel.cameraFlowApply.onclick = () => { callbacks++; };
  const camera = find("Auto camera flow"), face = find("Custom facial text");
  camera.events.click();
  assert.equal(camera.attributes["aria-expanded"], "true");
  face.events.click();
  assert.equal(camera.attributes["aria-expanded"], "false");
  assert.equal(face.attributes["aria-expanded"], "true");
  face.events.click();
  assert.equal(face.attributes["aria-expanded"], "false");
  assert.equal(callbacks, 0);
  assert.equal(JSON.stringify(state), before);
  assert.equal(panel.facialCustomInput.value, "Expression sentinel");
  assert.equal(panel.videoStyleCustomInput.value, "Style sentinel");
  assert.equal(panel.cameraFlowSelect.value, "medium_shots");
  panel.cameraFlowApply.onclick();
  assert.equal(callbacks, 1);
});

test("Descriptions and conditional custom options live inside their expandable setting", () => {
  const { panel, document } = fixture();
  const body = panel.temporalEffectControls.parentElement;
  for (const key of ["temporalEffectCustomControls", "temporalEffectOptions", "temporalProtectedCustomControls", "temporalEffectInfo"]) {
    assert.equal(panel[key].parentElement, body, key);
  }
  assert.equal(panel.fxCustomControls.parentElement, panel.fxControls.parentElement);
  assert.equal(panel.videoStyleCustomControls.parentElement, panel.videoStyleControls.parentElement);
  const css = document.getElementById("vrgdg-scene-defaults-card-styles").textContent;
  assert.match(css, /repeat\(2,minmax\(0,1fr\)\)/);
  assert.match(css, /@container \(max-width:720px\)/);
  assert.match(css, /:has\(.vrgdg-scene-defaults-primary/);
  assert.match(css, /prefers-reduced-motion/);
});
