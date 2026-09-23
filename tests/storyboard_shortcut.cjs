const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const { test } = require("node:test");
const path = require("node:path");
const ui = fs.readFileSync(path.join(__dirname, "../web/VRGDG_MusicVideoBuilderUI.js"), "utf8");
const story = fs.readFileSync(path.join(__dirname, "../web/VRGDG_StoryboardBuilderUI.js"), "utf8");
const section = (s, a, b) => s.slice(s.indexOf(a), s.indexOf(b, s.indexOf(a)));
for (const shiftKey of [true, false]) test(`both event paths route shift=${shiftKey} to the correct card`, () => {
  const calls = [];
  const c = { segment: { id: "scene2" }, block: {},
    openStoryboardBuilderFromProject: x => calls.push(["story", x.focusSceneId]),
    openLyricReviewModal: x => calls.push(["review", x.singleSceneId]),
    event: { shiftKey, preventDefault() {}, stopPropagation() {} },
    finishEvent: { shiftKey }, lastTimelineSceneClickTime: Date.now(), lastTimelineSceneClickId: "scene2", isOverlay: false, now: Date.now(), handleSegmentPick() {},
  };
  vm.createContext(c);
  vm.runInContext(section(ui, "  function openTimelineSceneCard(", "  let activeSegmentDragCleanup"), c);
  const native = section(ui, "        block.ondblclick =", "      enableImageDrop");
  vm.runInContext(native.slice(0, native.lastIndexOf("}")), c);
  c.block.ondblclick(c.event);
  const pointer = section(ui, "          if (now - lastTimelineSceneClickTime", "      activeSegmentDragCleanup");
  vm.runInContext(pointer.slice(0, pointer.lastIndexOf("        }")), c);
  assert.deepEqual(calls, [[shiftKey ? "story" : "review", "scene2"], [shiftKey ? "story" : "review", "scene2"]]);
});
test("repeated focused opens stop before constructing another builder", () => {
  const opening = section(story, "function openStoryboardBuilder(payload = {}) {", "  const projectFolder");
  vm.runInNewContext(opening + ' throw Error("duplicate builder"); } openStoryboardBuilder({ focusSceneId: "a" });', {
    document: { querySelector: () => ({}) },
  });
});
function editorFixture(focused = true) {
  const c = { focusedSceneOnly: focused, apply: {}, cancel: {}, closeEditor: {}, calls: [],
    saveEditorFieldsToScene: () => c.calls.push("fields"),
    saveStoryboard: async options => { assert.equal(options.throwOnError, true); c.calls.push("save"); if (c.fail) throw Error("disk full"); },
    syncReferenceMappingsToVideoCreator: () => c.calls.push("refs"),
    syncStoryLayerFromInputs: () => c.calls.push("story"),
    editorBackdrop: { remove: () => c.calls.push("close editor") }, backdrop: { remove: () => c.calls.push("close builder") },
    renderTable: () => c.calls.push("render"), createToast: m => c.calls.push(m),
  };
  vm.createContext(c); vm.runInContext(section(story, "    apply.onclick = async () => {", "\n  };"), c);
  return c;
}
test("focused Apply saves before syncing and closing", async () => {
  const c = editorFixture(); await c.apply.onclick();
  assert.deepEqual(c.calls, ["fields", "save", "refs", "story", "close editor", "close builder"]);
});
test("failed Apply keeps editor open and can be retried", async () => {
  const c = editorFixture(); c.fail = true; await c.apply.onclick();
  assert.deepEqual(c.calls, ["fields", "save", "disk full"]);
  assert.equal(c.apply.disabled, false); assert.equal(c.cancel.disabled, false);
  c.fail = false; c.calls.length = 0; await c.apply.onclick();
  assert.equal(c.calls.at(-1), "close builder");
});
test("ordinary Scene Card Apply retains its existing in-memory behavior", async () => {
  const c = editorFixture(false); await c.apply.onclick();
  assert.deepEqual(c.calls, ["fields", "refs", "story", "close editor", "render"]);
});
test("Apply cannot run twice while saving", async () => {
  const c = editorFixture(); let finish;
  c.saveStoryboard = () => new Promise(resolve => { finish = resolve; });
  const pending = c.apply.onclick(); await c.apply.onclick();
  assert.deepEqual(c.calls, ["fields"]); finish(); await pending;
});
for (const projectFolder of ["", "project"]) test(`focused persistence errors propagate with project=${projectFolder}`, async () => {
  const c = { state: { projectFolder, scenes: [] }, save: {}, syncStoryLayerFromInputs() {},
    slimStoryboardForRequest: x => x, postJson: async () => { throw Error("disk full"); }, createToast() {},
  };
  vm.createContext(c); vm.runInContext(section(story, "  async function saveStoryboard(", "  async function exportPromptFiles"), c);
  await assert.rejects(c.saveStoryboard({ throwOnError: true }));
});
