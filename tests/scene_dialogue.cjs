const assert = require("assert");
const vm = require("vm");
const { readBuilderModule } = require("./builder_source.cjs");
class Element {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.style = {}; this.isConnected = true; this.value = ""; }
  append(...items) { this.children.push(...items); }
  after(item) { this.afterElement = item; }
  setAttribute(key, value) { this[key] = value; }
  removeAttribute(key) { delete this[key]; }
  remove() { this.isConnected = false; }
  pause() { this.paused = true; }
  load() {}
  focus() {}
  querySelectorAll(selector) {
    if (selector === "input:invalid") return [];
    const tags = selector.toUpperCase().split(",");
    return this.children.flatMap(child => [...(tags.includes(child.tagName) ? [child] : []), ...child.querySelectorAll(selector)]);
  }
}
const document = { body: new Element("body"), createElement: tag => new Element(tag), addEventListener() {}, removeEventListener() {} };
let resolveRequest, route, body, persisted = 0, history = 0, instructionScene;
const state = { videoType: "speaking", projectFolder: "project", elevenLabsApiKey: "secret", audioClips: [],
  segments: [{ id: "a", label: "Scene 1", start: 0, end: 4 }, { id: "b", start: 4, end: 8 }],
  fluxReferenceBuilder: { subjects: [{ id: "alice", name: "Alice", reference_type: "character", elevenlabs_voice: { enabled: true, voice_id: "voice1" } }] } };
const context = vm.createContext({ document, console,
  makeButton: text => Object.assign(new Element("button"), { textContent: text }),
  makeInput: (value, type) => Object.assign(new Element("input"), { value, type }),
  makeSelect: (_options, value = "") => Object.assign(new Element("select"), { value }),
  makeCheckbox: (_label, checked) => { const wrapper = new Element("label"), input = new Element("input"); input.checked = checked; wrapper.append(input); return { wrapper, input }; },
  makeField: (label, input) => { const field = new Element("label"); field.textContent = label; field.append(input); return field; },
  audioUrl: path => path, audioClipsForState: value => value.audioClips || [],
  postJson: async (url, payload) => {
    route = url; body = payload;
    if (url.endsWith("preview_scene_audio_settings")) {
      const saved = JSON.parse(JSON.stringify(payload.session));
      saved.segments[0].scene_dialogue = payload.dialogue;
      saved.segments[0].end = payload.attachment.duration;
      saved.segments[1].start = payload.attachment.duration;
      saved.segments[1].end = payload.attachment.duration + 4;
      saved.audio_clips = [{ scene_id: "a", path: payload.attachment.saved_path, duration: payload.attachment.duration, start: 0 }];
      return { session: saved };
    }
    return new Promise(resolve => { resolveRequest = resolve; });
  },
});
vm.runInContext(readBuilderModule("scene_dialogue.mjs"), context);
vm.runInContext(readBuilderModule("scene_audio_settings.mjs"), context);
async function main() {
  const feature = context.createSceneAudioSettings({ state, projectInput: { value: "project" }, overlay: { isConnected: true },
    projectDefaultsAnchor: new Element("button"), pushHistory: () => history++, pauseTimelineForEditing() {}, render() {},
    saveSession: async () => { persisted++; }, sceneSlotNumber: scene => state.segments.indexOf(scene) + 1,
    getLlmPayload: () => ({ text_runner: "own_server" }), openInstructions: (_key, scene) => { instructionScene = scene.id; } });
  feature.openSceneAudioSettings(state.segments[0]);
  const box = document.body.children.at(-1).children[0];
  const panel = box.children.find(child => child.children[0]?.textContent === "ElevenLabs Scene Dialogue");
  assert(panel);
  const [speaker, model, text, delivery] = [1, 2, 3, 4].map(index => panel.children[index].children[0]);
  const [help, instructions, generate, use] = panel.children[6].children;
  speaker.value = "alice"; speaker.onchange(); text.value = "Stay behind me."; text.oninput();
  delivery.value = "Quiet and worried"; delivery.oninput();
  instructions.onclick(); assert.strictEqual(instructionScene, "a");
  let request = help.onclick(); assert.strictEqual(body.text_runner, "own_server");
  assert.strictEqual(body.scene_id, "a"); assert.strictEqual(body.dialogue.allow_rewrite, false);
  resolveRequest({ dialogue: { ...body.dialogue, text: "[whispers] Stay behind me." } }); await request;
  assert.strictEqual(text.value, "[whispers] Stay behind me.");
  assert.strictEqual(state.segments[0].scene_dialogue, undefined); // Draft remains local until Save.
  request = generate.onclick(); assert.strictEqual(body.references.subjects[0].elevenlabs_voice.voice_id, "voice1");
  assert.strictEqual(body.dialogue.model_id, "eleven_v4");
  resolveRequest({ audio_data: "data:audio/mpeg;base64,YQ==", audio_name: "speech.mp3", dialogue: body.dialogue }); await request;
  assert(!use.disabled); assert.strictEqual(panel.children[8].hidden, false);
  text.value += " Please."; text.oninput(); assert(use.disabled); assert(panel.children[8].hidden);
  request = generate.onclick(); resolveRequest({ audio_data: "data:audio/mpeg;base64,YQ==", audio_name: "speech.mp3", dialogue: body.dialogue }); await request;
  request = use.onclick(); assert(route.endsWith("save_scene_audio")); assert.strictEqual(body.preserve_source, true);
  resolveRequest({ saved_path: "speech-source.mp3", duration: 6, peaks: [] }); await request;
  assert.strictEqual(state.segments[0].end, 4); // Use stages it; Save commits the timeline.
  const save = box.querySelectorAll("button").find(button => button.textContent === "Save Scene Audio Settings");
  await save.onclick();
  assert.strictEqual(state.segments[0].end, 6); assert.strictEqual(state.segments[1].start, 6);
  assert.strictEqual(state.audioClips[0].scene_id, "a");
  assert.strictEqual(state.segments[0].scene_dialogue.text, "[whispers] Stay behind me. Please.");
  assert.strictEqual(persisted, 1); assert.strictEqual(history, 1);
  assert(!JSON.stringify(state).includes("data:audio"));
  assert(!JSON.stringify(state.segments).includes("secret"));
  // Closing or changing mode during generation cannot stage a stale take.
  feature.openSceneAudioSettings(state.segments[0]);
  const nextBox = document.body.children.at(-1).children[0];
  const nextPanel = nextBox.children.find(child => child.children[0]?.textContent === "ElevenLabs Scene Dialogue");
  request = nextPanel.children[6].children[2].onclick();
  state.videoType = "singing"; feature.refresh();
  resolveRequest({ audio_data: "data:audio/mpeg;base64,YQ==", dialogue: body.dialogue }); await request;
  assert.strictEqual(nextPanel.children[8].src, undefined);
  feature.dispose();
  console.log("Scene dialogue LLM review, candidate invalidation, preview/import/save and mode guards passed.");
}
main().catch(error => { console.error(error); process.exitCode = 1; });
