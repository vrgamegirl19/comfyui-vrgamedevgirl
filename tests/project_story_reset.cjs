const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

const root = path.join(__dirname, "..", "web", "music_video_builder");
const actionsSource = fs.readFileSync(path.join(root, "project_actions.mjs"), "utf8");
const settingsSource = fs.readFileSync(path.join(root, "model_settings.mjs"), "utf8");
const start = settingsSource.indexOf("export function normalizeBuilderStoryLayer(");
const normalizer = settingsSource.slice(start, settingsSource.indexOf("export function ", start + 1)).replace("export ", "");

function fixture(state) {
  const context = vm.createContext({ console, DEFAULT_KREA2_REFERENCE_SETTINGS: {},
    setWidgetValue() {}, newSegment: () => ({ id: "starter" }) });
  vm.runInContext(normalizer, context);
  for (const name of ["defaultErnieImageSettings", "defaultFlowGptBrowserSettings", "defaultFluxKleinSettings",
    "defaultI2VVideoSettings", "defaultKrea2TwoPassSettings", "defaultZEnhanceSettings", "defaultZImageSettings",
    "defaultFluxReferenceBuilder", "defaultIdLoraReferenceBuilder", "defaultLyricMapper", "cloneMiniMaxH3Settings"]) {
    context[name] = () => ({});
  }
  vm.runInContext(actionsSource.replace(/^import [\s\S]*?;\r?\n/gm, "").replace(/^export /gm, ""), context);
  const names = actionsSource.match(/createProjectActions\(\{([\s\S]*?)\}\)/)[1].split(",").map((s) => s.trim()).filter(Boolean);
  const deps = Object.fromEntries(names.map((name) => [name, () => {}]));
  Object.assign(deps, { state, restoreBrowserAiDownloadsQuietly: async () => {}, faceFixTool: {},
    audio: { removeAttribute() {}, load() {}, dataset: {} }, sceneAudio: { removeAttribute() {} },
    audioInput: { dataset: {} }, useVrgdgTextContext: { input: {} } });
  for (const name of names.filter((name) => name.endsWith("Input") || name.endsWith("Button"))) {
    deps[name] = { dataset: {} };
  }
  return { actions: context.createProjectActions(deps), normalize: context.normalizeBuilderStoryLayer };
}

for (const project of ["C:/projects/New Song", ""]) {
  test(`new project reset clears the previous story before wizard opening or session save (${project || "fresh startup"})`, () => {
    const oldStory = { overall_story_idea: "Old film", user_story_arc: "Old cast and plot",
      song_story_brief: "Previous song", image_world_style: "custom", image_custom_style_direction: "Old world" };
    const state = { builderStoryLayer: oldStory };
    const { actions, normalize } = fixture(state);
    actions.resetProjectState(project);
    const storyForWizardAndSave = normalize(state.builderStoryLayer);
    assert.equal(storyForWizardAndSave.overall_story_idea, "");
    assert.equal(storyForWizardAndSave.user_story_arc, "");
    assert.equal(storyForWizardAndSave.song_story_brief, "");
    assert.equal(storyForWizardAndSave.image_custom_style_direction, "");
    assert.equal(storyForWizardAndSave.image_world_style, "natural");
    assert.notEqual(state.builderStoryLayer, oldStory);
    assert.equal(oldStory.user_story_arc, "Old cast and plot");
  });
}
